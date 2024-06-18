import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import WeightedRandomSampler

import utils
from modules.mel_processing import mel_spectrogram_torch
from utils import audio_to_energy, load_filepaths_and_text, load_wav_to_torch

# import h5py


"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams, all_in_mem: bool = False):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.unit_interpolate_mode = hparams.data.unit_interpolate_mode
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.num_mels = hparams.data.n_mel_channels
        self.mel_fmin = hparams.data.mel_fmin
        self.mel_fmax = hparams.data.mel_fmax
        # self.min_file_length = hparams.data.min_file_length * self.sampling_rate
        # self.max_file_length = hparams.data.max_file_length * self.sampling_rate
        self.num_frames = int(4 * self.sampling_rate // self.hop_length)

        random.seed(1234)
        random.shuffle(self.audiopaths)

        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

        self.audiopaths = self._filter_long_files(self.audiopaths)

    def _filter_long_files(self, audio_paths):
        self.unique_speaker_count = len(set([x[1] for x in audio_paths]))

        print("Unique speakers:", self.unique_speaker_count)
        print("Audiopaths before filtering:", len(audio_paths))
        # print("Audiopaths after filtering:", len(filtered))

        return audio_paths

    def get_audio(self, filename):
        # filename = filename.replace("\\", "/")
        filename, speaker_id = filename
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "Sample Rate not match. Expect {} but got {} from {}".format(
                    self.sampling_rate, sampling_rate, filename
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        # compute mel spectrogram
        spec = mel_spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.num_mels,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            self.mel_fmin,
            self.mel_fmax,
        )
        spec = torch.squeeze(spec, 0)

        # energy
        energy = audio_to_energy(
            audio_norm,
            self.filter_length,
            self.num_mels,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            self.mel_fmin,
            self.mel_fmax,
        )

        # ppg unit and duration
        ppg_path = filename.replace(".wav", ".ppg_unit.pt")
        ppg = torch.load(ppg_path)
        ppg_unit = ppg["ppg_unit"]
        ppg_unit_dur = ppg["ppg_unit_dur"]

        # load f0 and uv
        f0_path = filename.replace(".wav", ".rmvpe.pt")
        loaded_data = torch.load(f0_path)
        f0 = loaded_data["f0"].unsqueeze(0)
        uv = loaded_data["uv"]

        # load hubert
        hubert_path = filename.replace(".wav", ".soft.pt")
        c = torch.load(hubert_path)
        c = utils.repeat_expand_2d(
            c.squeeze(0), f0.shape[1], mode=self.unit_interpolate_mode
        )

        # perturbate c randomly with noise and weight randomly
        if random.random() < 0.5:
            c = c + torch.randn_like(c)

        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (
            c.size(-1),
            spec.size(-1),
            filename,
        )
        assert abs(audio_norm.shape[1] - lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv, energy = (
            spec[:, :lmin],
            c[:, :lmin],
            f0[:, :lmin],
            uv[:lmin],
            energy[:, :lmin],
        )

        # speaker id
        speaker_id = torch.LongTensor([int(speaker_id)])

        return c, f0, spec, uv, energy, speaker_id, ppg_unit, ppg_unit_dur

    def random_slice(self, c, f0, spec, uv, energy, speaker_id, ppg_unit, ppg_unit_dur):
        # if spec.shape[1] > self.num_frames:
        #     start = random.randint(0, spec.shape[1] - self.num_frames)
        #     end = start + self.num_frames - 1
        #     spec, c, f0, uv, energy = (
        #         spec[:, start:end],
        #         c[:, start:end],
        #         f0[:, start:end],
        #         uv[start:end],
        #         energy[:, start:end],
        #     )

        return c, f0, spec, uv, energy, speaker_id, ppg_unit, ppg_unit_dur

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.random_slice(*self.cache[index])
        else:
            return self.random_slice(*self.get_audio(self.audiopaths[index]))

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]), dim=0, descending=True
        )

        max_c_len = max([x[0].size(1) for x in batch])
        max_ppg_len = max([x[6].size(0) for x in batch])

        lengths = torch.LongTensor(len(batch))
        ppg_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), 1, max_c_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)
        energy_padded = torch.FloatTensor(len(batch), 1, max_c_len)
        ppg_unit_padded = torch.LongTensor(len(batch), max_ppg_len)
        ppg_unit_dur_padded = torch.FloatTensor(len(batch), max_ppg_len)

        c_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        uv_padded.zero_()
        energy_padded.zero_()
        ppg_unit_padded.zero_()
        ppg_unit_dur_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, : c.size(1)] = c
            lengths[i] = c.size(1)

            f0 = row[1]
            f0_padded[i, 0, : f0.size(1)] = f0

            spec = row[2]
            spec_padded[i, :, : spec.size(1)] = spec

            uv = row[3]
            uv_padded[i, : uv.size(0)] = uv

            energy = row[4]
            energy_padded[i, 0, : energy.size(1)] = energy

            sid[i] = row[5]

            ppg = row[6]
            ppg_unit_padded[i, : ppg.size(0)] = ppg
            ppg_lengths[i] = ppg.size(0)

            ppg_dur = row[7]
            ppg_unit_dur_padded[i, : ppg_dur.size(0)] = ppg_dur

        return (
            c_padded,
            f0_padded,
            spec_padded,
            lengths,
            uv_padded,
            energy_padded,
            sid,
            ppg_unit_padded,
            ppg_lengths,
            ppg_unit_dur_padded,
        )


def get_weighted_sampler(items):
    dataset_samples_weight = 1.0

    speaker_names = np.array([item[1] for item in items])
    unique_speaker_names = np.unique(speaker_names).tolist()
    speaker_ids = [unique_speaker_names.index(l) for l in speaker_names]
    speaker_count = np.array(
        [len(np.where(speaker_names == l)[0]) for l in unique_speaker_names]
    )
    weight_speaker = 1.0 / speaker_count

    speaker_samples_weight = np.array(
        np.array([weight_speaker[l] for l in speaker_ids])
    )
    speaker_samples_weight = speaker_samples_weight / np.linalg.norm(
        speaker_samples_weight
    )
    speaker_samples_weight = torch.from_numpy(speaker_samples_weight).float()
    dataset_samples_weight += speaker_samples_weight * 2.0

    return WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))
