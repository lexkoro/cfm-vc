import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.utils.data

from modules.mel_processing import MelSpectrogramFeatures

PATH_PLACEHOLDER = "<replace_this_path>"


def _get(hparams, key, default=None):
    value = getattr(hparams, key, default)
    return default if value is None else value


def _discover_metadata_files(metadata_root: Path, patterns):
    """Discover and deduplicate metadata CSV files under *metadata_root*."""
    if isinstance(patterns, str):
        patterns = [patterns]

    metadata_files = []
    for pattern in patterns:
        metadata_files.extend(metadata_root.glob(pattern))

    metadata_files = sorted(
        {path.resolve() for path in metadata_files if path.is_file()}
    )
    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata CSV found under {metadata_root} with patterns: {patterns}"
        )
    return metadata_files


def _resolve_audio_path(metadata_root: Path, raw_path: str) -> str:
    """Resolve a raw path from a metadata CSV to an absolute path."""
    path = raw_path.replace("\\", "/").strip()
    root_str = str(metadata_root)
    if PATH_PLACEHOLDER in path:
        path = path.replace(PATH_PLACEHOLDER, root_str)
    elif not os.path.isabs(path):
        path = os.path.join(root_str, path)
    return os.path.abspath(os.path.expanduser(path))


class UnitMelLoader(torch.utils.data.Dataset):
    """Loads frame-aligned raw k-means units and mel spectrograms."""

    def __init__(self, phase: str, hparams, verbose=False):
        root_path = hparams.root_path
        self.metadata_root = Path(os.path.abspath(os.path.expanduser(root_path)))
        if not self.metadata_root.exists():
            raise FileNotFoundError(
                f"Dataset root_path does not exist: {self.metadata_root}"
            )

        self.phase = phase
        self.verbose = verbose
        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length
        self.unit_path_mode = _get(hparams, "unit_path_mode", "sidecar")
        self.allow_resample = _get(hparams, "allow_resample", True)

        codes_root_path = _get(hparams, "codes_root_path", None)
        self.codes_root = (
            Path(os.path.abspath(os.path.expanduser(codes_root_path)))
            if codes_root_path is not None
            else None
        )

        self.mel_extractor = MelSpectrogramFeatures(
            sample_rate=hparams.sampling_rate,
            n_fft=hparams.filter_length,
            hop_length=hparams.hop_length,
            n_mels=hparams.n_mel_channels,
            padding=_get(hparams, "mel_padding", "same"),
        )

        metadata_glob = _get(hparams, "metadata_glob", "*/*_metadata.csv")
        metadata_files = _discover_metadata_files(self.metadata_root, metadata_glob)
        if self.verbose:
            print(f"[UnitMelLoader] Found {len(metadata_files)} metadata file(s)")

        self.audio_paths = self._parse_metadata(metadata_files)

        rng = random.Random(_get(hparams, "seed", 1234))
        rng.shuffle(self.audio_paths)

        val_samples = int(_get(hparams, "val_samples", 4))
        val_indices = set(
            rng.sample(
                range(len(self.audio_paths)), min(val_samples, len(self.audio_paths))
            )
        )
        if phase == "val":
            self.audio_paths = [
                path for idx, path in enumerate(self.audio_paths) if idx in val_indices
            ]

        if self.verbose:
            print(f"[UnitMelLoader:{phase}] {len(self.audio_paths)} samples")

    def _parse_metadata(self, metadata_files):
        entries = []
        for metadata_file in metadata_files:
            with open(metadata_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    raw_path = line.split("|", 1)[0]
                    entries.append(_resolve_audio_path(self.metadata_root, raw_path))
        return list(dict.fromkeys(entries))

    def _build_units_path(self, audio_path: str) -> Path:
        if self.unit_path_mode == "sidecar":
            return Path(audio_path).with_suffix(".npy")
        if self.unit_path_mode == "mirror":
            if self.codes_root is None:
                raise ValueError(
                    "codes_root_path is required for mirror unit_path_mode"
                )
            rel = os.path.relpath(audio_path, self.metadata_root)
            return self.codes_root / Path(rel).with_suffix(".npy")
        raise ValueError(f"Unknown unit_path_mode: {self.unit_path_mode}")

    def get_audio(self, filename):
        try:
            target_sr = self.sampling_rate if self.allow_resample else None
            audio, sampling_rate = librosa.load(filename, sr=target_sr, mono=True)

            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR"
                )

            data = torch.from_numpy(audio).float().unsqueeze(0)

            mel = self.mel_extractor(data).squeeze(0)
        except Exception as err:
            print(err)
            print(filename)
            raise

        return mel

    def _load_units(self, audio_path: str):
        units_path = self._build_units_path(audio_path)
        if not units_path.is_file():
            raise FileNotFoundError(f"Missing unit file for {audio_path}: {units_path}")

        units = torch.from_numpy(np.load(units_path)).long().squeeze()
        if units.ndim != 1:
            raise ValueError(f"Expected 1D unit ids in {units_path}, got {units.shape}")
        return units

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        mel = self.get_audio(audio_path)
        units = self._load_units(audio_path)

        # repair mel by cropping to match unit length, if needed
        unit_len = units.size(-1)
        mel = mel[:, :unit_len]

        return {
            "mel": mel,
            "units": units,
        }

    def __len__(self):
        return len(self.audio_paths)

    def collate_fn(self, batch):
        units = [item["units"] for item in batch]
        mels = [item["mel"] for item in batch]

        unit_lengths = torch.tensor([u.size(-1) for u in units], dtype=torch.long)
        mel_lengths = torch.tensor([m.size(-1) for m in mels], dtype=torch.long)

        units_padded = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(units, layout=torch.jagged), padding=0
        )
        mels_padded = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(mels, layout=torch.jagged), padding=0
        )

        return (units_padded, unit_lengths, mels_padded, mel_lengths)
