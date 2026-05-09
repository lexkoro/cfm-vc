import argparse
import json
import os
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import soundfile as sf
import torch
import torchaudio
from espnet2.tasks.ssl import SSLTask
from sklearn.cluster import MiniBatchKMeans

PATH_PLACEHOLDER = "<replace_this_path>"


def iter_audio_files(
    root: Path,
    extensions: Sequence[str],
    recursive: bool = True,
) -> Generator[Path, None, None]:
    norm_ext = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    if recursive:
        walker = root.rglob("*")
    else:
        walker = root.glob("*")

    for path in walker:
        if path.is_file() and path.suffix.lower() in norm_ext:
            yield path


def iter_manifest_files(manifest_path: Path) -> Generator[Path, None, None]:
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Path(line)


def parse_metadata_glob_arg(raw_patterns: Sequence[str] | None) -> List[str]:
    if not raw_patterns:
        return ["*/*_metadata.csv"]

    if len(raw_patterns) == 1:
        val = raw_patterns[0].strip()
        if val.startswith("["):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [val]

    return [x.strip() for x in raw_patterns if x.strip()]


def discover_metadata_files(metadata_root: Path, patterns: Sequence[str]) -> List[Path]:
    metadata_files: List[Path] = []
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


def resolve_audio_path(metadata_root: Path, raw_path: str) -> Path:
    path = raw_path.replace("\\", "/").strip()
    root_str = str(metadata_root)

    if PATH_PLACEHOLDER in path:
        path = path.replace(PATH_PLACEHOLDER, root_str)
    elif not os.path.isabs(path):
        path = os.path.join(root_str, path)

    return Path(os.path.abspath(os.path.expanduser(path)))


def iter_metadata_audio_files(
    metadata_root: Path,
    metadata_files: Sequence[Path],
    encoding: str = "utf-8",
) -> Generator[Path, None, None]:
    for mf in metadata_files:
        with mf.open("r", encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue

                raw_path = line.split("|", 1)[0].strip()
                if not raw_path:
                    continue

                yield resolve_audio_path(metadata_root, raw_path)


def load_audio(
    wav_path: Path,
    target_sr: int,
    resamplers: Dict[int, torchaudio.transforms.Resample],
) -> torch.Tensor:
    # wav, sr = torchaudio.load(str(wav_path))
    # wav = wav.float()

    wavs, sr = sf.read(str(wav_path))  # sampling rate should be 16000
    wav = torch.FloatTensor([wavs])

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, target_sr)
        wav = resamplers[sr](wav)

    return wav.squeeze(0)


def batch_audio(
    file_iter: Iterable[Path],
    batch_size: int,
    target_sr: int,
    max_files: int | None,
) -> Generator[Tuple[List[Path], torch.Tensor, torch.LongTensor], None, None]:
    files: List[Path] = []
    waves: List[torch.Tensor] = []
    resamplers: Dict[int, torchaudio.transforms.Resample] = {}
    seen = 0

    for path in file_iter:
        if max_files is not None and seen >= max_files:
            break

        try:
            wav = load_audio(path, target_sr=target_sr, resamplers=resamplers)
        except Exception as exc:
            print(f"[warn] failed to load {path}: {exc}")
            continue

        files.append(path)
        waves.append(wav)
        seen += 1

        if len(files) == batch_size:
            lengths = torch.LongTensor([w.shape[0] for w in waves])
            padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
            yield files, padded, lengths
            files, waves = [], []

    if files:
        lengths = torch.LongTensor([w.shape[0] for w in waves])
        padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        yield files, padded, lengths


def resolve_xeus_layer(outputs, layer_index: int) -> torch.Tensor:
    # Support both dict and tuple/list variants used in existing notebooks.
    if isinstance(outputs, dict):
        encoder_out = outputs.get("encoder_output")
    elif isinstance(outputs, (list, tuple)):
        encoder_out = outputs[0]
    else:
        raise TypeError(f"Unsupported xeus output type: {type(outputs)}")

    if encoder_out is None:
        raise ValueError("Could not find encoder_output in xeus outputs")

    if not isinstance(encoder_out, (list, tuple)):
        raise TypeError("Expected encoder_output to be a list/tuple of hidden states")

    if layer_index < 0:
        idx = len(encoder_out) + layer_index
    else:
        idx = layer_index

    if idx < 0 or idx >= len(encoder_out):
        raise IndexError(
            f"Requested layer {layer_index}, but encoder_output has {len(encoder_out)} layers"
        )

    feats = encoder_out[idx]
    if feats.ndim != 3:
        raise ValueError(
            f"Expected layer features with shape [B, T, C], got {tuple(feats.shape)}"
        )

    return feats


def estimate_num_frames(num_samples: int, frame_shift_samples: int = 320) -> int:
    # Xeus paper/path uses 20ms stride at 16kHz -> 320 samples per frame.
    return max(1, int(round(num_samples / frame_shift_samples)))


def chunk_rows(
    arr: np.ndarray, chunk_rows_size: int
) -> Generator[np.ndarray, None, None]:
    n = arr.shape[0]
    for start in range(0, n, chunk_rows_size):
        yield arr[start : start + chunk_rows_size]


def save_checkpoint(
    kmeans: MiniBatchKMeans,
    out_model_path: Path,
    meta_path: Path,
    stats: dict,
) -> None:
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, out_model_path)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamed Xeus k-means trainer")

    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Root directory of audio files (used when --manifest is not provided)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Text file containing one absolute/relative audio path per line",
    )
    parser.add_argument(
        "--root-path",
        type=Path,
        default=None,
        help="Dataset root path used with --metadata-glob (metadata CSV mode)",
    )
    parser.add_argument(
        "--metadata-glob",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Metadata CSV glob patterns under --root-path. "
            "Examples: --metadata-glob LJSpeech/*_metadata.csv "
            "or --metadata-glob '[\"LJSpeech/*_metadata.csv\"]'"
        ),
    )
    parser.add_argument(
        "--metadata-encoding",
        type=str,
        default="utf-8",
        help="Encoding used to read metadata CSV files",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".wav,.flac,.mp3,.ogg,.m4a",
        help="Comma-separated audio file extensions for --audio-root mode",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan --audio-root",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Disable recursive scan for --audio-root",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of audio files to process",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on total feature frames used for k-means updates",
    )

    parser.add_argument(
        "--xeus-config",
        type=Path,
        default=Path("/home/alex/Projekt/cfm-vc/ckpt/xeus/config.yaml"),
        help="Path to Xeus config.yaml",
    )
    parser.add_argument(
        "--xeus-checkpoint",
        type=Path,
        default=Path("/home/alex/Projekt/cfm-vc/ckpt/xeus/xeus_checkpoint_new.pth"),
        help="Path to Xeus checkpoint",
    )
    parser.add_argument(
        "--xeus-layer",
        type=int,
        default=14,
        help="Layer index used for clustering (paper uses 14)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate expected by Xeus",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of utterances per Xeus extraction batch",
    )

    parser.add_argument("--n-clusters", type=int, default=500)
    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=32768,
        help="MiniBatchKMeans internal batch size",
    )
    parser.add_argument(
        "--partial-fit-chunk-rows",
        type=int,
        default=100000,
        help="Rows per partial_fit call from extracted feature buffers",
    )
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--out-model",
        type=Path,
        default=Path("ckpt/xeus/kmeans_xeus_streamed.pkl"),
        help="Output path for trained MiniBatchKMeans model (joblib)",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=Path("ckpt/xeus/kmeans_xeus_streamed.meta.json"),
        help="Output path for training metadata",
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help="Optional existing joblib MiniBatchKMeans to resume from",
    )
    parser.add_argument(
        "--checkpoint-updates",
        type=int,
        default=200,
        help="Checkpoint frequency in partial_fit updates",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for Xeus extraction",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_modes = (
        int(args.manifest is not None)
        + int(args.audio_root is not None)
        + int(args.root_path is not None)
    )
    if input_modes != 1:
        raise ValueError(
            "Provide exactly one of --manifest, --audio-root, or --root-path"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[info] loading xeus model")
    xeus_model, _ = SSLTask.build_model_from_file(
        str(args.xeus_config),
        str(args.xeus_checkpoint),
        args.device,
    )
    xeus_model = xeus_model.eval()

    if args.resume_model is not None and args.resume_model.exists():
        print(f"[info] resuming MiniBatchKMeans from {args.resume_model}")
        kmeans = joblib.load(args.resume_model)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=args.n_clusters,
            init="k-means++",
            batch_size=args.kmeans_batch_size,
            n_init=1,
            compute_labels=False,
            reassignment_ratio=0.01,
            random_state=args.seed,
            verbose=0,
        )

    if args.manifest is not None:
        file_iter = iter_manifest_files(args.manifest)
        print(f"[info] input mode=manifest file={args.manifest}")
    elif args.audio_root is not None:
        extensions = [e.strip() for e in args.extensions.split(",") if e.strip()]
        file_iter = iter_audio_files(
            args.audio_root, extensions, recursive=args.recursive
        )
        print(f"[info] input mode=audio-root root={args.audio_root}")
    else:
        metadata_root = Path(os.path.abspath(os.path.expanduser(str(args.root_path))))
        if not metadata_root.exists():
            raise FileNotFoundError(f"--root-path does not exist: {metadata_root}")

        patterns = parse_metadata_glob_arg(args.metadata_glob)
        metadata_files = discover_metadata_files(metadata_root, patterns)
        print(
            "[info] input mode=metadata "
            f"root={metadata_root} patterns={patterns} files={len(metadata_files)}"
        )
        file_iter = iter_metadata_audio_files(
            metadata_root=metadata_root,
            metadata_files=metadata_files,
            encoding=args.metadata_encoding,
        )

    total_files = 0
    total_frames = 0
    total_updates = 0

    for files, wav_batch, wav_lengths in batch_audio(
        file_iter=file_iter,
        batch_size=args.batch_size,
        target_sr=args.sample_rate,
        max_files=args.max_files,
    ):
        with torch.no_grad():
            outputs = xeus_model.encode(
                wav_batch.to(args.device),
                wav_lengths.to(args.device),
                use_final_output=False,
            )

        layer_feats = resolve_xeus_layer(outputs, layer_index=args.xeus_layer)
        layer_feats = layer_feats.detach().cpu().float().numpy()

        for i, wav_path in enumerate(files):
            feat = layer_feats[i]
            est_frames = estimate_num_frames(int(wav_lengths[i].item()))
            feat = feat[: min(est_frames, feat.shape[0])]

            if feat.shape[0] == 0:
                continue

            feat = feat.astype(np.float32, copy=False)

            for chunk in chunk_rows(feat, args.partial_fit_chunk_rows):
                if chunk.shape[0] == 0:
                    continue

                kmeans.partial_fit(chunk)

                total_updates += 1
                total_frames += int(chunk.shape[0])

                if total_updates % args.checkpoint_updates == 0:
                    stats = {
                        "total_files": total_files,
                        "total_frames": total_frames,
                        "total_updates": total_updates,
                        "n_clusters": args.n_clusters,
                        "feature_dim": int(chunk.shape[1]),
                        "xeus_layer": args.xeus_layer,
                        "sample_rate": args.sample_rate,
                    }
                    save_checkpoint(kmeans, args.out_model, args.out_meta, stats)
                    print(
                        f"[ckpt] files={total_files} frames={total_frames} updates={total_updates}"
                    )

                if args.max_frames is not None and total_frames >= args.max_frames:
                    print("[info] reached --max-frames limit")
                    stats = {
                        "total_files": total_files,
                        "total_frames": total_frames,
                        "total_updates": total_updates,
                        "n_clusters": args.n_clusters,
                        "feature_dim": int(chunk.shape[1]),
                        "xeus_layer": args.xeus_layer,
                        "sample_rate": args.sample_rate,
                    }
                    save_checkpoint(kmeans, args.out_model, args.out_meta, stats)
                    return

            total_files += 1

            if total_files % 100 == 0:
                frames_m = total_frames / 1_000_000
                print(
                    f"[progress] files={total_files} frames={frames_m:.2f}M updates={total_updates}"
                )

    stats = {
        "total_files": total_files,
        "total_frames": total_frames,
        "total_updates": total_updates,
        "n_clusters": args.n_clusters,
        "feature_dim": int(kmeans.cluster_centers_.shape[1]),
        "xeus_layer": args.xeus_layer,
        "sample_rate": args.sample_rate,
    }
    save_checkpoint(kmeans, args.out_model, args.out_meta, stats)

    print(
        "[done] "
        f"files={total_files} frames={total_frames} updates={total_updates} "
        f"model={args.out_model}"
    )


if __name__ == "__main__":
    main()
