import librosa.util as librosa_util
import numpy as np
import torch
import torch.nn.functional as F
from librosa.util import pad_center, tiny
from scipy.signal import get_window
from torch import nn
from torch.autograd import Variable


def window_sumsquare(
    window,
    n_frames,
    hop_length=200,
    win_length=800,
    n_fft=800,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, size=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(
        self, filter_length=800, hop_length=200, win_length=800, window="hann"
    ):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert win_length >= filter_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.to(magnitude.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class Denoiser(nn.Module):
    """Removes model bias from audio produced with hifigan"""

    def __init__(
        self,
        model,
        filter_length=1024,
        n_overlap=4,
        win_length=1024,
        mode="zeros",
    ):
        super().__init__()

        w = next(
            p
            for name, p in model.named_parameters()
            if name.endswith(".weight.original0")
        )

        self.stft = STFT(
            filter_length=filter_length,
            hop_length=int(filter_length / n_overlap),
            win_length=win_length,
        ).to(w.device)

        mel_init = {"zeros": torch.zeros, "normal": torch.randn}[mode]
        mel_input = mel_init((1, 80, 88), dtype=w.dtype, device=w.device)

        with torch.no_grad():
            bias_audio = model(mel_input).float()

            if len(bias_audio.size()) > 2:
                bias_audio = bias_audio.squeeze(0)
            elif len(bias_audio.size()) < 2:
                bias_audio = bias_audio.unsqueeze(0)
            assert len(bias_audio.size()) == 2

            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer("bias_spec", bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        audio_spec, audio_angles = self.stft.transform(audio.float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised


# class Denoiser(nn.Module):
#     """Removes model bias from audio produced with hifigan"""

#     def __init__(self, model, mode="normal", device="cpu"):
#         super().__init__()

#         filter_length = 1024
#         win_length = 1024
#         self.stft = STFT(
#             filter_length=filter_length, hop_length=256, win_length=win_length
#         )
#         self.device = device
#         self.stft = self.stft.to(self.device)
#         self.mode = mode

#         self.set_mode(mode)
#         self.generate_bias(model)

#     def set_mode(self, mode):
#         self.mode = mode
#         if self.mode == "zeros":
#             self.def_mel_input = torch.zeros(
#                 (1, 80, 88),
#                 dtype=torch.float32,
#                 device=self.device,
#             )
#         elif self.mode == "normal":
#             self.def_mel_input = torch.randn(
#                 (1, 80, 88),
#                 dtype=torch.float32,
#                 device=self.device,
#             )
#         else:
#             raise ValueError(f"Invalid mode {mode}")

#     def generate_bias(self, model):
#         with torch.no_grad():
#             bias_spec = model(self.def_mel_input).float()
#         self.register_buffer("bias_spec", bias_spec[:, :, 0][:, :, None])

#     def forward(self, audio, strength=0.1, model=None):
#         self.generate_bias(model)

#         audio = audio.squeeze(0)
#         audio_spec, audio_angles = self.stft.transform(audio.float())

#         audio_spec_denoised = audio_spec - self.bias_spec * strength
#         audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
#         audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
#         return audio_denoised
