import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchtune.modules import RotaryPositionalEmbeddings
from vector_quantize_pytorch import ResidualFSQ


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class RMSNorm(nn.Module):
    def __init__(self, dim, unit_offset=False):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim**0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1.0 - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim=-1) * self.scale * gamma


class DepthwiseConv(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dw_conv1d = nn.Conv1d(
            dim, dim, kernel_size, groups=dim, padding=kernel_size // 2, bias=False
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        # x: (B, T, C)
        x = self.dw_conv1d(x)
        x = self.activation(x)
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class MLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, intermediate_dim * 2, bias=False)
        self.geglu = GEGLU()
        self.fc2 = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.geglu(x)
        x = self.fc2(x)
        return x


class ChunkedSlidingLocalMHA(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: RotaryPositionalEmbeddings,
        window_size: int = 32,
        chunk_size: int | None = None,
    ):
        super().__init__()

        assert dim % n_heads == 0
        assert window_size > 0

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.rotary_embed = rotary_embed
        self.window_size = window_size
        self.chunk_size = chunk_size if chunk_size is not None else window_size

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        assert self.flash, "Must have scaled_dot_product_attention."

        # Keep names and shapes identical for checkpoint compatibility.
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def _build_chunk_mask(
        self,
        query_start: int,
        query_end: int,
        kv_start: int,
        kv_end: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(query_start, query_end, device=device)
        kv_positions = torch.arange(kv_start, kv_end, device=device)
        local = (
            query_positions[:, None] - kv_positions[None, :]
        ).abs() <= self.window_size
        return local.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, T, C)
        """
        batch_size, seq_len, _ = x.shape

        q, k, v = rearrange(
            self.c_attn(x),
            "b t (r h d) -> r b t h d",
            r=3,
            h=self.n_heads,
        )

        # Apply RoPE on the full sequence once so positions stay absolute.
        q = self.rotary_embed(q)
        k = self.rotary_embed(k)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, H, T, D)
        v = v.transpose(1, 2)  # (B, H, T, D)

        outputs = []
        chunk_size = self.chunk_size
        radius = self.window_size

        for query_start in range(0, seq_len, chunk_size):
            query_end = min(query_start + chunk_size, seq_len)
            kv_start = max(0, query_start - radius)
            kv_end = min(seq_len, query_end + radius)

            q_chunk = q[:, :, query_start:query_end, :]
            k_chunk = k[:, :, kv_start:kv_end, :]
            v_chunk = v[:, :, kv_start:kv_end, :]

            attn_mask = self._build_chunk_mask(
                query_start=query_start,
                query_end=query_end,
                kv_start=kv_start,
                kv_end=kv_end,
                device=x.device,
            )

            y_chunk = torch.nn.functional.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            outputs.append(y_chunk)

        y = torch.cat(outputs, dim=2)
        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.c_proj(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        n_heads: int,
        rotary_embed: RotaryPositionalEmbeddings,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.intermediate_dim = intermediate_dim

        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.att = ChunkedSlidingLocalMHA(
            dim=dim,
            n_heads=n_heads,
            rotary_embed=rotary_embed,
            window_size=32,
            chunk_size=64,
        )
        self.mlp = MLP(dim=dim, intermediate_dim=intermediate_dim)

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape
        window = self.window
        assert isinstance(window, torch.Tensor)

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2  # +2 for magnitude and phase representation
        self.out = nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, H, L), where B is the batch size,
                        H is the model dimension, and L is the sequence length.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor [B, H, L], got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.out.in_features:
            raise ValueError(
                "ISTFTHead expects channel-first input [B, H, L] with "
                f"H={self.out.in_features}, got shape {tuple(x.shape)}"
            )

        x_pred = self.out(x.transpose(1, 2))  # (B, L, out_dim)
        x_pred = x_pred.float()
        x_pred = x_pred.transpose(1, 2)  # (B, out_dim, L)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        S = torch.polar(mag, p)
        audio = self.istft(S)
        return audio.unsqueeze(1)


class Decoder(nn.Module):
    """
    Frame-rate spectral decoder inspired by the lightweight VoCodec decoder.

    This generator keeps the mel frame rate unchanged, applies a compact
    attention module at frame resolution, and predicts full STFT coefficients
    directly for ISTFT reconstruction.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        n_layers: int = 12,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.n_heads = n_heads

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        DepthwiseConv(hidden_dim, kernel_size=15),
                        TransformerBlock(
                            dim=hidden_dim,
                            intermediate_dim=intermediate_dim,
                            n_heads=n_heads,
                            rotary_embed=RotaryPositionalEmbeddings(
                                dim=hidden_dim // n_heads
                            ),
                        ),
                    ]
                )
            )

        self.final_norm = RMSNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  layers
        for dwconv, attn in self.layers:
            x = dwconv(x) + x
            x = attn(x.transpose(1, 2)).transpose(1, 2)

        # final normalization
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class CodecDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        intermediate_dim=1024,
        heads=12,
        n_layers=6,
        dropout=0.0,
        hop_length=320,
        output_hop_length=None,
        vq_dim=2048,
    ):
        super().__init__()
        self.hop_length = hop_length
        # output_hop_length controls the ISTFT synthesis rate
        # e.g. 480 for 24kHz output (20ms * 24000 = 480 samples/frame)
        self.output_hop_length = (
            output_hop_length if output_hop_length is not None else hop_length
        )

        self.quantizer = ResidualFSQ(
            dim=vq_dim, levels=[4, 4, 4, 4, 4, 4, 4, 4], num_quantizers=1
        )

        self.backbone = Decoder(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            n_heads=heads,
            dropout=dropout,
            n_layers=n_layers,
        )

        self.head = ISTFTHead(
            dim=hidden_dim,
            n_fft=self.output_hop_length * 4,
            hop_length=self.output_hop_length,
            padding="same",
        )

        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            x = x.permute(0, 2, 1)
            x, q = self.quantizer(x)
            x = x.permute(0, 2, 1)
            q = q.permute(0, 2, 1)
            return x, q, None

        x = self.backbone(x)
        x = self.head(x)

        return x

    def quantize(self, x):
        x = x.permute(0, 2, 1)
        x, q = self.quantizer(x)
        x = x.permute(0, 2, 1)
        q = q.permute(0, 2, 1)
        return x, q, None

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)


class XCodecDecoder(nn.Module):
    """Standalone decoder: VQ codes → 24 kHz waveform.

    Expects a stripped checkpoint produced by ``strip_checkpoint_for_tts.py``
    containing ``codec_decoder`` and ``fc_post_a`` state dicts.
    """

    def __init__(
        self,
        ckpt_path: str,
    ):
        super().__init__()
        self._decoder = CodecDecoder(
            hidden_dim=1024,
            intermediate_dim=1024,
            heads=16,
            n_layers=12,
            dropout=0.00,
            hop_length=480,
            output_hop_length=480,
            vq_dim=2048,
        )
        self._fc_post_a = nn.Linear(2048, 1024)

        # Load stripped checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._decoder.load_state_dict(ckpt["codec_decoder"])
        self._fc_post_a.load_state_dict(ckpt["fc_post_a"])

        self._sample_rate = ckpt.get("metadata", {}).get("sample_rate", 24000)
        self._hop_length = ckpt.get("metadata", {}).get("hop_length", 480)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def hop_length(self) -> int:
        return self._hop_length

    @torch.inference_mode()
    def decode(self, codes: torch.LongTensor) -> torch.Tensor:
        """Decode VQ code indices to a waveform.

        Args:
            codes: Integer code tensor of shape ``(batch, num_quantizers, time)``.
                For single-quantizer models (default) ``num_quantizers=1``.
                The values are those saved as ``.npy`` by ``inference_save_code.py``.

        Returns:
            Waveform tensor of shape ``(batch, samples)`` at 24 kHz.
        """
        # get_output_from_indices expects (batch, time, num_quantizers)
        vq_post_emb = self._decoder.quantizer.get_output_from_indices(
            codes.transpose(1, 2)
        )

        # Project vq_dim (2048) → hidden_dim (1024)
        # fc_post_a operates on last dim, so transpose around it
        decoder_input = self._fc_post_a(vq_post_emb).transpose(1, 2)

        # Backbone + ISTFTHead → (batch, 1, samples)
        wav = self._decoder(decoder_input, vq=False)
        return wav.squeeze(1)

    def save_wav(self, wav: torch.Tensor, path: str) -> None:
        """Write a waveform tensor to a WAV file.

        Args:
            wav: Waveform of shape ``(samples,)`` or ``(1, samples)``.
            path: Output file path.
        """
        audio = wav.detach().float().cpu().squeeze().numpy()
        sf.write(path, audio, self._sample_rate)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CodecDecoder(
        hidden_dim=768,
        intermediate_dim=1024,
        heads=12,
        num_convnext_blocks=6,
        dropout=0.0,
        output_hop_length=480,
    ).to(device)
    print(f"Model initialized. Params: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 2
    sequence_length = 50
    dummy_input = torch.randn(batch_size, 768, sequence_length).to(device)

    model.eval()
    with torch.no_grad():
        output_no_vq = model(dummy_input, vq=False)
        print(f"Output shape: {output_no_vq.shape}")
