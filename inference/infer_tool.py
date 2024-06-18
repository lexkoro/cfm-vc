import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np
import ppgs

# import onnxruntime
import soundfile
import torch
import torchaudio

import utils
from models import SynthesizerTrn
from modules.commons import dedup_seq
from utils import audio_to_energy

# from models_cf import SynthesizerTrn

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def read_temp(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write(json.dumps({"info": "temp_dict"}))
        return {}
    else:
        try:
            with open(file_name, "r") as f:
                data = f.read()
            data_dict = json.loads(data)
            if os.path.getsize(file_name) > 50 * 1024 * 1024:
                f_name = file_name.replace("\\", "/").split("/")[-1]
                print(f"clean {f_name}")
                for wav_hash in list(data_dict.keys()):
                    if (
                        int(time.time()) - int(data_dict[wav_hash]["time"])
                        > 14 * 24 * 3600
                    ):
                        del data_dict[wav_hash]
        except Exception as e:
            print(e)
            print(f"{file_name} error,auto rebuild file")
            data_dict = {"info": "temp_dict"}
        return data_dict


def write_temp(file_name, data):
    with open(file_name, "w") as f:
        f.write(json.dumps(data))


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print("executing '%s' costed %.3fs" % (func.__name__, time.time() - t))
        return res

    return run


def format_wav(audio_path):
    if Path(audio_path).suffix == ".wav":
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != "."]
        dirs[:] = [d for d in dirs if d[0] != "."]
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def get_md5(content):
    return hashlib.new("md5", content).hexdigest()


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(
            arr, (pad_left, pad_right), "constant", constant_values=(0, 0)
        )
        return padded_arr


def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i - pre if i - pre >= 0 else i : i + n]


class F0FilterException(Exception):
    pass


class Svc(object):
    def __init__(
        self,
        net_g_path,
        config_path,
        device=None,
    ):
        self.net_g_path = net_g_path
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.net_g_ms = None

        self.hps_ms = utils.get_hparams_from_file(config_path, True)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.unit_interpolate_mode = (
            self.hps_ms.data.unit_interpolate_mode
            if self.hps_ms.data.unit_interpolate_mode is not None
            else "left"
        )

        # contentvec encoder
        self.speech_encoder = (
            self.hps_ms.model.speech_encoder
            if self.hps_ms.model.speech_encoder is not None
            else "vec768l12"
        )
        # load hubert and model
        self.load_model()
        self.hubert_model = utils.get_speech_encoder(
            self.speech_encoder, device=self.dev
        )

        if not hasattr(self, "audio16k_resample_transform"):
            self.audio16k_resample_transform = torchaudio.transforms.Resample(
                self.target_sample, 16000
            ).to(self.dev)

    def load_model(self):
        # get model configuration
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.n_mel_channels,
            n_speakers=789,
            **self.hps_ms.model,
        )
        _ = utils.load_checkpoint(self.net_g_path, self.net_g_ms, None)
        self.dtype = list(self.net_g_ms.parameters())[0].dtype
        if "half" in self.net_g_path and torch.cuda.is_available():
            _ = self.net_g_ms.half().eval().to(self.dev)
        else:
            _ = self.net_g_ms.eval().to(self.dev)

    def get_unit_f0(
        self,
        wav,
        c_targets,
        tran,
        cluster_infer_ratio,
        f0_filter,
        f0_predictor,
        cr_threshold=0.05,
    ):
        if (
            not hasattr(self, "f0_predictor_object")
            or self.f0_predictor_object is None
            or f0_predictor != self.f0_predictor_object.name
        ):
            self.f0_predictor_object = utils.get_f0_predictor(
                f0_predictor,
                hop_length=self.hop_size,
                sampling_rate=self.target_sample,
                device=self.dev,
                threshold=cr_threshold,
            )
        f0, uv = self.f0_predictor_object.compute_f0_uv(wav)

        if f0_filter and sum(f0) == 0:
            raise F0FilterException("No voice detected")
        f0 = torch.FloatTensor(f0).to(self.dev)
        uv = torch.FloatTensor(uv).to(self.dev)

        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        wav = torch.from_numpy(wav).unsqueeze(0).to(self.dev)

        wav16k = self.audio16k_resample_transform(wav)[0]

        c = self.hubert_model.encoder(wav16k)
        c = utils.repeat_expand_2d(
            c.squeeze(0), f0.shape[1], self.unit_interpolate_mode
        )

        if cluster_infer_ratio != 0:
            feature_index = utils.compute_index(c_targets)

            feat_np = np.ascontiguousarray(c.transpose(0, 1).cpu().numpy())
            self.big_npy = feature_index.reconstruct_n(0, feature_index.ntotal)

            score, ix = feature_index.search(feat_np, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            c = cluster_infer_ratio * npy + (1 - cluster_infer_ratio) * feat_np
            c = torch.FloatTensor(c).to(self.dev).transpose(0, 1)

        c = c.unsqueeze(0)
        f0 = f0.unsqueeze(0)

        return c, f0, uv

    def infer(
        self,
        tran,
        raw_path,
        target_audio_path,
        cluster_infer_ratio=0,
        n_timesteps=2,
        f0_filter=False,
        f0_predictor="pm",
        cr_threshold=0.05,
        temperature=1.0,
        guidance_scale=0.0,
        solver="dopri5",
    ):
        torchaudio.set_audio_backend("soundfile")
        wav, sr = torchaudio.load(raw_path)
        if (
            not hasattr(self, "audio_resample_transform")
            or self.audio_resample_transform.orig_freq != sr
        ):
            self.audio_resample_transform = torchaudio.transforms.Resample(
                sr, self.target_sample
            )

        # resample to target sample rate
        wav = self.audio_resample_transform(wav).squeeze(0)

        # energy
        energy = (
            audio_to_energy(
                wav.unsqueeze(0),
                self.hps_ms.data.filter_length,
                self.hps_ms.data.n_mel_channels,
                self.hps_ms.data.sampling_rate,
                self.hps_ms.data.hop_length,
                self.hps_ms.data.win_length,
                self.hps_ms.data.mel_fmin,
                self.hps_ms.data.mel_fmax,
            )
            .unsqueeze(0)
            .to(self.dev)
        )

        wav = wav.numpy()

        # get the root path of the file
        c_targets = []
        mels = []
        mel_lengths = []
        for f in target_audio_path:
            wav_tgt, sr_tgt = torchaudio.load(f)

            if wav_tgt.size(0) > 1:
                wav_tgt = torch.mean(wav_tgt, dim=0, keepdim=True)

            wav_target16k = torchaudio.functional.resample(
                wav_tgt, sr_tgt, 16000
            ).squeeze(0)

            # wav_target, _ = torchaudio.load(f)
            # wav_target16k = self.audio16k_resample_transform(wav_target).squeeze(0)
            c_targets.append(self.hubert_model.encoder(wav_target16k.to(self.dev)))

            # mel spectrograms
            mel_spec_tgt = utils.mel_spectrogram_torch(
                wav_tgt,
                self.hps_ms.data.filter_length,
                self.hps_ms.data.n_mel_channels,
                self.hps_ms.data.sampling_rate,
                self.hps_ms.data.hop_length,
                self.hps_ms.data.win_length,
                self.hps_ms.data.mel_fmin,
                self.hps_ms.data.mel_fmax,
            ).to(self.dev)

            mels.append(mel_spec_tgt)
            mel_lengths.append(torch.LongTensor([mel_spec_tgt.shape[2]]).to(self.dev))

        # compute cond latent and speaker embedding
        speaker_embedding, cond, cond_mask = self.net_g_ms.compute_conditional_latent(
            mels, mel_lengths
        )

        # get contentvec, f0, uv and energy
        c, f0, uv = self.get_unit_f0(
            wav,
            c_targets,
            tran,
            cluster_infer_ratio,
            f0_filter,
            f0_predictor,
            cr_threshold=cr_threshold,
        )

        # Load speech audio at correct sample rate
        audio = ppgs.load.audio(raw_path)
        # Infer PPGs
        ppg = ppgs.from_audio(audio, ppgs.SAMPLE_RATE, gpu=None).float()
        ppg = utils.repeat_expand_2d(ppg.squeeze(0), f0.shape[-1], mode="nearest")

        sparse_ppg = ppgs.sparsify(
            ppg=ppg, method="percentile", threshold=torch.Tensor([0.85])
        )
        most_probable_ppg = torch.argmax(sparse_ppg, dim=1)
        ppg_features, ppg_durations = dedup_seq(most_probable_ppg)
        ppg_features = ppg_features.to(self.dev)
        ppg_durations = ppg_durations.to(self.dev)
        ppg_features_lengths = torch.LongTensor([ppg_features.shape[1]]).to(self.dev)

        c = c.to(self.dtype)
        f0 = f0.to(self.dtype)
        uv = uv.to(self.dtype)

        with torch.no_grad():
            o, _ = self.net_g_ms.vc(
                c,
                cond=cond,
                cond_mask=cond_mask,
                f0=f0,
                uv=uv,
                energy=energy,
                ppg=ppg_features,
                ppg_lengths=ppg_features_lengths,
                ppg_dur=ppg_durations,
                g=speaker_embedding,
                n_timesteps=n_timesteps,
                temperature=temperature,
                guidance_scale=guidance_scale,
                solver=solver,
            )

        return o

    def clear_empty(self):
        # clean up vram
        torch.cuda.empty_cache()

    def unload_model(self):
        # unload model
        self.net_g_ms = self.net_g_ms.to("cpu")
        del self.net_g_ms
        gc.collect()

    def slice_inference(
        self,
        raw_audio_path,
        raw_target_audio_path,
        tran,
        cluster_infer_ratio,
        n_timesteps=2,
        f0_predictor="rmvpe",
        cr_threshold=0.05,
        temperature=1.0,
        guidance_scale=0.0,
        solver="dopri5",
    ):
        out_audio = self.infer(
            tran,
            raw_audio_path,
            target_audio_path=raw_target_audio_path,
            cluster_infer_ratio=cluster_infer_ratio,
            n_timesteps=n_timesteps,
            f0_predictor=f0_predictor,
            cr_threshold=cr_threshold,
            temperature=temperature,
            guidance_scale=guidance_scale,
            solver=solver,
        )

        return out_audio
