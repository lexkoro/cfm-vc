import os
from argparse import ArgumentParser
from fnmatch import fnmatch
from pathlib import Path
from time import time
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import joblib
import librosa
import numpy as np
import torch
from espnet2.tasks.ssl import SSLTask
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PATH_PLACEHOLDER = "<replace_this_path>"


def pad_audio_batch(
    batch: Sequence[Tuple[torch.Tensor, int, str]],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    lengths = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) > 0 else 0
    padded = torch.zeros((len(batch), max_len), dtype=torch.float32)
    paths: List[str] = []

    for index, (wav_16k, wav_len, wav_path) in enumerate(batch):
        padded[index, :wav_len] = wav_16k
        paths.append(wav_path)

    return padded, lengths, paths


class WaveDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        sampling_rate: int,
        audio_norm_scale: float = 1.0,
    ) -> None:
        self.file_list = file_list
        self.sampling_rate = sampling_rate
        self.audio_norm_scale = audio_norm_scale

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        wav_path = self.file_list[index]

        wav = librosa.load(wav_path, sr=self.sampling_rate)[0]
        audio = torch.from_numpy(wav).to(torch.float32)
        if self.audio_norm_scale < 1.0:
            audio = audio * self.audio_norm_scale

        return audio, int(audio.shape[0]), wav_path

    def __len__(self) -> int:
        return len(self.file_list)


class ApplyKmeans:
    def __init__(self, model_path: str, device: torch.device) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Kmeans model not found: {model_path}")

        km_model = joblib.load(model_path)
        centers = np.asarray(km_model.cluster_centers_, dtype=np.float32)
        if centers.ndim != 2:
            raise ValueError(
                f"Expected 2D cluster_centers_, got shape {tuple(centers.shape)}"
            )

        self.device = device
        self.feature_dim = int(centers.shape[1])

        self.centers = torch.from_numpy(centers.transpose()).to(device)
        self.centers_norm = (self.centers**2).sum(0, keepdim=True)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(
                f"Expected 2D features [T, D], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Kmeans feature dim mismatch: expected {self.feature_dim}, "
                f"got {features.shape[1]}"
            )

        x = features.to(self.device, dtype=torch.float32)
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.centers)
        dist = dist + self.centers_norm
        return dist.argmin(dim=1)


def _iter_metadata_matches(metadata_root: Path, pattern: str) -> Iterable[Path]:
    normalized = pattern.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]

    if "**" in normalized:
        return metadata_root.glob(pattern)

    parts = [part for part in normalized.split("/") if part and part != "."]
    if len(parts) == 1:
        name_pattern = parts[0]
        return (
            path
            for path in metadata_root.iterdir()
            if path.is_file() and fnmatch(path.name, name_pattern)
        )

    if len(parts) == 2:
        directory_pattern, file_pattern = parts
        matches = []
        for child in metadata_root.iterdir():
            if not child.is_dir() or not fnmatch(child.name, directory_pattern):
                continue
            for candidate in child.iterdir():
                if candidate.is_file() and fnmatch(candidate.name, file_pattern):
                    matches.append(candidate)
        return matches

    return metadata_root.glob(pattern)


def _discover_metadata_files(
    metadata_root: Path, metadata_glob: List[str]
) -> List[Path]:
    metadata_files = []
    for pattern in metadata_glob:
        metadata_files.extend(_iter_metadata_matches(metadata_root, pattern))

    metadata_files = sorted(
        {path.resolve() for path in metadata_files if path.is_file()}
    )
    if not metadata_files:
        raise FileNotFoundError(
            f"No metadata CSV found under {metadata_root} with patterns: {metadata_glob}"
        )
    return metadata_files


def _resolve_audio_path(metadata_root: Path, raw_path: str) -> str:
    path = raw_path.replace("\\", "/").strip()
    root_str = str(metadata_root)
    if PATH_PLACEHOLDER in path:
        path = path.replace(PATH_PLACEHOLDER, root_str)
    elif not os.path.isabs(path):
        path = os.path.join(root_str, path)
    return os.path.abspath(os.path.expanduser(path))


def build_file_list_from_metadata(
    root_path: str, metadata_glob: List[str]
) -> List[str]:
    metadata_root = Path(os.path.abspath(os.path.expanduser(root_path)))
    if not metadata_root.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {metadata_root}")

    flist: List[str] = []
    for metadata_file in _discover_metadata_files(metadata_root, metadata_glob):
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                raw_path, _ = line.split("|", 1)
                flist.append(_resolve_audio_path(metadata_root, raw_path))

    # Keep metadata order while de-duplicating.
    flist = list(dict.fromkeys(flist))
    if not flist:
        raise RuntimeError(
            "No valid audio paths found from metadata files. Check metadata contents."
        )
    return flist


def split_for_rank(file_list: List[str], rank: int, world_size: int) -> List[str]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size - 1}] but got {rank}")
    split_file_lists = np.array_split(file_list, world_size)
    return split_file_lists[rank].tolist()


def build_code_output_path(wav_path: str) -> str:
    return str(Path(wav_path).with_suffix(".npy"))


def save_unit_code(unit_code: torch.Tensor, wav_path: str) -> None:
    code_path = build_code_output_path(wav_path=wav_path)
    Path(code_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(code_path, unit_code.detach().cpu().numpy().astype(np.int32))


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--root-path",
        type=str,
        default="/path/to/dataset_root",
        help="Dataset root path used by metadata entries",
    )
    parser.add_argument(
        "--metadata-glob",
        type=str,
        nargs="+",
        default=["*/*_metadata.csv"],
        help="Metadata CSV glob patterns under root-path",
    )
    parser.add_argument(
        "--xeus-config",
        type=str,
        default="ckpt/xeus/config.yaml",
        help="Path to Xeus config YAML",
    )
    parser.add_argument(
        "--xeus-checkpoint",
        type=str,
        default="ckpt/xeus/xeus_checkpoint_new.pth",
        help="Path to Xeus checkpoint",
    )
    parser.add_argument(
        "--kmeans-model",
        type=str,
        required=True,
        help="Path to a joblib kmeans model used for unit assignment",
    )
    parser.add_argument(
        "--xeus-layer",
        type=int,
        default=14,
        help="Hidden-state layer index from Xeus used for kmeans features",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate expected by Xeus",
    )
    parser.add_argument(
        "--audio-norm-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scaling applied after loading audio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device override, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for the DataLoader",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", 0)),
        help="Local GPU device ID",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=int(os.getenv("RANK", os.getenv("LOCAL_RANK", 0))),
        help="Global rank for file sharding",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", 1)),
        help="World size for file sharding",
    )
    return parser


def resolve_device(device_arg: Optional[str], local_rank: int) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _validate_required_files(file_paths: Sequence[str]) -> None:
    for path_str in file_paths:
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Required file not found: {path_str}")


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    is_main = args.rank == 0
    input_root = os.path.abspath(os.path.expanduser(args.root_path))
    device = resolve_device(args.device, args.local_rank)

    xeus_config = os.path.abspath(os.path.expanduser(args.xeus_config))
    xeus_checkpoint = os.path.abspath(os.path.expanduser(args.xeus_checkpoint))
    kmeans_model = os.path.abspath(os.path.expanduser(args.kmeans_model))
    _validate_required_files((xeus_config, xeus_checkpoint, kmeans_model))

    file_list = build_file_list_from_metadata(
        root_path=input_root,
        metadata_glob=args.metadata_glob,
    )
    if is_main:
        print(f"Found {len(file_list)} entries from metadata under {input_root}")

    current_file_list = split_for_rank(
        file_list=file_list,
        rank=args.rank,
        world_size=args.world_size,
    )
    print(
        f"Rank {args.rank}/{args.world_size} processing {len(current_file_list)} files"
    )

    if is_main:
        print(f"Loading Xeus from config={xeus_config}, checkpoint={xeus_checkpoint}")
        print(f"Loading kmeans model from {kmeans_model}")

    xeus_model, _ = SSLTask.build_model_from_file(
        xeus_config,
        xeus_checkpoint,
        str(device),
    )
    xeus_model = xeus_model.eval()
    xeus_model_typed: Any = xeus_model
    apply_kmeans = ApplyKmeans(model_path=kmeans_model, device=device)

    dataset = WaveDataset(
        file_list=current_file_list,
        sampling_rate=args.sample_rate,
        audio_norm_scale=args.audio_norm_scale,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_audio_batch,
    )

    st = time()
    checked_layer_bounds = False

    for wav_batch, wav_lengths, wav_paths in tqdm(
        dataloader, desc="processing", disable=not is_main
    ):
        wav_batch = wav_batch.to(device, non_blocking=True)
        wav_lengths = wav_lengths.to(device, non_blocking=True)

        with torch.no_grad():
            _, hidden_states, feat_lengths = xeus_model_typed.inference_encode(
                wav_batch,
                wav_lengths,
                use_mask=False,
            )

        n_layers = len(hidden_states)
        if not checked_layer_bounds:
            if args.xeus_layer >= n_layers or args.xeus_layer < -n_layers:
                raise ValueError(
                    f"Invalid --xeus-layer={args.xeus_layer}. "
                    f"Model returned {n_layers} hidden-state tensors."
                )
            checked_layer_bounds = True

        layer_features = hidden_states[args.xeus_layer]
        feat_lengths_list = feat_lengths.detach().cpu().tolist()

        for index, wav_path in enumerate(wav_paths):
            frame_count = int(feat_lengths_list[index])
            features = layer_features[index, :frame_count, :]
            unit_code = apply_kmeans(features)
            save_unit_code(unit_code=unit_code, wav_path=wav_path)

    et = time()
    if is_main:
        print(f"Done, time: {(et - st) / 60:.2f} mins")


if __name__ == "__main__":
    main()
