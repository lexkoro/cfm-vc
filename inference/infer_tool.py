import gc
import hashlib
import io
import json
import logging
import os
import pickle
import time
from pathlib import Path

import librosa
import numpy as np

# import onnxruntime
import soundfile
import torch
import torchaudio

import cluster
import utils
from inference import slicer
from models import SynthesizerTrn
from modules.mel_processing import mel_spectrogram_torch

# from models_cf import SynthesizerTrn
from modules.speaker_encoder import ResNetSpeakerEncoder

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
        cluster_model_path="logs/44k/kmeans_10000.pt",
        speaker_encoder_path="logs/44k/speaker_encoder.pt",
        feature_retrieval=False,
    ):
        self.net_g_path = net_g_path
        self.feature_retrieval = feature_retrieval
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
        self.vol_embedding = (
            self.hps_ms.model.vol_embedding
            if self.hps_ms.model.vol_embedding is not None
            else False
        )
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
        self.volume_extractor = utils.Volume_Extractor(self.hop_size)

        if os.path.exists(cluster_model_path):
            if self.feature_retrieval:
                with open(cluster_model_path, "rb") as f:
                    self.cluster_model = pickle.load(f)
                self.big_npy = None
            else:
                self.cluster_model = cluster.get_cluster_model(cluster_model_path)
        else:
            self.feature_retrieval = False

        self.speaker_encoder = ResNetSpeakerEncoder(
            input_dim=80, proj_dim=512, log_input=True
        )
        checkpoint = torch.load(
            speaker_encoder_path,
            map_location="cpu",
        )
        self.speaker_encoder.load_state_dict(checkpoint)
        self.speaker_encoder.eval()

        if not hasattr(self, "audio16k_resample_transform"):
            self.audio16k_resample_transform = torchaudio.transforms.Resample(
                self.target_sample, 16000
            ).to(self.dev)

    def load_model(self):
        # get model configuration
        self.net_g_ms = SynthesizerTrn(
            self.hps_ms.data.n_mel_channels,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            num_mel_channels=self.hps_ms.data.n_mel_channels,
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

        # compute energy
        energy = utils.audio_to_energy(
            wav,
            filter_length=self.hps_ms.data.filter_length,
            n_mel_channels=self.hps_ms.data.n_mel_channels,
            hop_length=self.hps_ms.data.hop_length,
            win_length=self.hps_ms.data.win_length,
            sampling_rate=self.hps_ms.data.sampling_rate,
            mel_fmin=self.hps_ms.data.mel_fmin,
            mel_fmax=self.hps_ms.data.mel_fmax,
        )

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
        energy = energy.unsqueeze(0)

        return c, f0, uv, energy

    def infer(
        self,
        tran,
        raw_path,
        target_audio_path,
        target_speaker,
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
        wav = self.audio_resample_transform(wav).squeeze(0).numpy()

        # wav_tgt = self.audio_resample_transform(wav_tgt)

        # speaker_embeddings = glob(f"/mnt/datasets/VC_Dataset/{speaker}/*.emb.pt")[
        #     :20
        # ]
        # speaker_embeddings = [
        #     torch.FloatTensor(torch.load(speaker_embedding))
        #     for speaker_embedding in speaker_embeddings
        # ]
        # sid = torch.mean(torch.stack(speaker_embeddings), dim=0).to(self.dev)

        speaker_embeddings = []
        for f in target_audio_path:
            speaker_embeddings.append(
                self.speaker_encoder.compute_embedding(f).to(self.dev)
            )

        sid = torch.mean(torch.stack(speaker_embeddings), dim=0).to(self.dev)

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

        style_cond = self.net_g_ms.compute_conditional_latent(mels, mel_lengths, sid)

        # sid = self.avg_speaker_embeddings[speaker].unsqueeze(0).to(self.dev)
        # sid = torch.LongTensor([int(speaker_id)]).to(self.dev).unsqueeze(0)
        c, f0, uv, energy = self.get_unit_f0(
            wav,
            c_targets,
            tran,
            cluster_infer_ratio,
            f0_filter,
            f0_predictor,
            cr_threshold=cr_threshold,
        )

        c = c.to(self.dtype)
        f0 = f0.to(self.dtype)
        uv = uv.to(self.dtype)

        with torch.no_grad():
            o, _ = self.net_g_ms.vc(
                c,
                style_cond=style_cond,
                f0=f0,
                g=sid,
                uv=uv,
                energy=energy,
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
        if hasattr(self, "enhancer"):
            self.enhancer.enhancer = self.enhancer.enhancer.to("cpu")
            del self.enhancer.enhancer
            del self.enhancer
        gc.collect()

    def slice_inference(
        self,
        raw_audio_path,
        raw_target_audio_path,
        target_speaker,
        tran,
        cluster_infer_ratio,
        n_timesteps=2,
        f0_predictor="pm",
        cr_threshold=0.05,
        temperature=1.0,
        guidance_scale=0.0,
        solver="dopri5",
    ):
        out_audio = self.infer(
            tran,
            raw_audio_path,
            target_audio_path=raw_target_audio_path,
            target_speaker=target_speaker,
            cluster_infer_ratio=cluster_infer_ratio,
            n_timesteps=n_timesteps,
            f0_predictor=f0_predictor,
            cr_threshold=cr_threshold,
            temperature=temperature,
            guidance_scale=guidance_scale,
            solver=solver,
        )

        return out_audio

        # global_frame = 0
        # audio = []
        # for slice_tag, data in audio_data:
        #     # padd
        #     length = int(np.ceil(len(data) / audio_sr * self.target_sample))
        #     if slice_tag:
        #         _audio = np.zeros(length)
        #         audio.extend(list(pad_array(_audio, length)))
        #         global_frame += length // self.hop_size
        #         continue
        #     if per_size != 0:
        #         datas = split_list_by_n(data, per_size, lg_size)
        #     else:
        #         datas = [data]
        #     for k, dat in enumerate(datas):
        #         per_length = (
        #             int(np.ceil(len(dat) / audio_sr * self.target_sample))
        #             if clip_seconds != 0
        #             else length
        #         )
        #         if clip_seconds != 0:
        #             print(
        #                 f"###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======"
        #             )
        #         # padd
        #         pad_len = int(audio_sr * pad_seconds)
        #         dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
        #         raw_path = io.BytesIO()
        #         soundfile.write(raw_path, dat, audio_sr, format="wav")
        #         raw_path.seek(0)

        #         out_audio, out_sr, out_frame = self.infer(
        #             tran,
        #             raw_path,
        #             target_audio_path=raw_target_audio_path,
        #             cluster_infer_ratio=cluster_infer_ratio,
        #             auto_predict_f0=auto_predict_f0,
        #             noice_scale=noice_scale,
        #             f0_predictor=f0_predictor,
        #             cr_threshold=cr_threshold,
        #             f0_adain_alpha=f0_adain_alpha,
        #             loudness_envelope_adjustment=loudness_envelope_adjustment,
        #         )

        #         global_frame += out_frame
        #         _audio = out_audio.cpu().numpy()
        #         pad_len = int(self.target_sample * pad_seconds)
        #         _audio = _audio[pad_len:-pad_len]
        #         _audio = pad_array(_audio, per_length)
        #         if lg_size != 0 and k != 0:
        #             lg1 = (
        #                 audio[-(lg_size_r + lg_size_c_r) : -lg_size_c_r]
        #                 if lgr_num != 1
        #                 else audio[-lg_size:]
        #             )
        #             lg2 = (
        #                 _audio[lg_size_c_l : lg_size_c_l + lg_size_r]
        #                 if lgr_num != 1
        #                 else _audio[0:lg_size]
        #             )
        #             lg_pre = lg1 * (1 - lg) + lg2 * lg
        #             audio = (
        #                 audio[0 : -(lg_size_r + lg_size_c_r)]
        #                 if lgr_num != 1
        #                 else audio[0:-lg_size]
        #             )
        #             audio.extend(lg_pre)
        #             _audio = (
        #                 _audio[lg_size_c_l + lg_size_r :]
        #                 if lgr_num != 1
        #                 else _audio[lg_size:]
        #             )
        #         audio.extend(list(_audio))
        # return np.array(audio)


class RealTimeVC:
    def __init__(self):
        self.last_chunk = None
        self.last_o = None
        self.chunk_len = 16000  # chunk length
        self.pre_len = 3840  # cross fade length, multiples of 640

    # Input and output are 1-dimensional numpy waveform arrays

    def process(
        self,
        svc_model,
        speaker_id,
        f_pitch_change,
        input_wav_path,
        cluster_infer_ratio=0,
        auto_predict_f0=False,
        noice_scale=0.4,
        f0_filter=False,
    ):
        import maad

        audio, sr = torchaudio.load(input_wav_path)
        audio = audio.cpu().numpy()[0]
        temp_wav = io.BytesIO()
        if self.last_chunk is None:
            input_wav_path.seek(0)

            audio, sr = svc_model.infer(
                speaker_id,
                f_pitch_change,
                input_wav_path,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
                f0_filter=f0_filter,
            )

            audio = audio.cpu().numpy()
            self.last_chunk = audio[-self.pre_len :]
            self.last_o = audio
            return audio[-self.chunk_len :]
        else:
            audio = np.concatenate([self.last_chunk, audio])
            soundfile.write(temp_wav, audio, sr, format="wav")
            temp_wav.seek(0)

            audio, sr = svc_model.infer(
                speaker_id,
                f_pitch_change,
                temp_wav,
                cluster_infer_ratio=cluster_infer_ratio,
                auto_predict_f0=auto_predict_f0,
                noice_scale=noice_scale,
                f0_filter=f0_filter,
            )

            audio = audio.cpu().numpy()
            ret = maad.util.crossfade(self.last_o, audio, self.pre_len)
            self.last_chunk = audio[-self.pre_len :]
            self.last_o = audio
            return ret[self.chunk_len : 2 * self.chunk_len]
