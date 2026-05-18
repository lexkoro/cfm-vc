import argparse
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import soundfile as sf
import torch
import torchaudio
from espnet2.tasks.ssl import SSLTask
from sklearn.cluster import MiniBatchKMeans

PATH_PLACEHOLDER = "<replace_this_path>"


@dataclass
class AudioRecord:
    audio_path: Path
    source_mode: str
    language: str = "unk"
    metadata_key: str = "unknown_metadata"
    metadata_file: Path | None = None
    row_index: int | None = None
    est_target_samples: int | None = None


@dataclass
class AudioChunk:
    audio_path: Path
    source_mode: str
    language: str = "unk"
    metadata_key: str = "unknown_metadata"
    metadata_file: Path | None = None
    row_index: int | None = None
    est_num_frames: int | None = None


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


def parse_language_from_metadata_name(metadata_file: Path) -> str:
    stem = metadata_file.stem.lower()
    if stem.endswith("_metadata"):
        stem = stem[: -len("_metadata")]
    match = re.match(r"^([a-z]{2,3})(?:$|[_-])", stem)
    if match:
        return match.group(1)
    return "unk"


def metadata_bucket_key(metadata_root: Path, metadata_file: Path) -> str:
    try:
        return metadata_file.resolve().relative_to(metadata_root.resolve()).as_posix()
    except ValueError:
        return metadata_file.name


def estimate_target_samples_from_audio(path: Path, target_sr: int) -> int | None:
    try:
        info = sf.info(str(path))
    except Exception:
        return None

    if info.samplerate <= 0 or info.frames <= 0:
        return None

    return int(round(info.frames * float(target_sr) / float(info.samplerate)))


def iter_metadata_audio_records(
    metadata_root: Path,
    metadata_files: Sequence[Path],
    target_sr: int,
    encoding: str = "utf-8",
) -> Generator[AudioRecord, None, None]:
    for mf in metadata_files:
        language = parse_language_from_metadata_name(mf)
        bucket_key = metadata_bucket_key(metadata_root, mf)

        with mf.open("r", encoding=encoding) as f:
            for row_index, line in enumerate(f, start=1):
                line = line.strip()
                if not line or "|" not in line:
                    continue

                raw_path = line.split("|", 1)[0].strip()
                if not raw_path:
                    continue

                audio_path = resolve_audio_path(metadata_root, raw_path)
                if not audio_path.exists():
                    continue

                est_target_samples = estimate_target_samples_from_audio(
                    audio_path, target_sr=target_sr
                )
                yield AudioRecord(
                    audio_path=audio_path,
                    source_mode="metadata",
                    language=language,
                    metadata_key=bucket_key,
                    metadata_file=mf,
                    row_index=row_index,
                    est_target_samples=est_target_samples,
                )


def estimate_num_frames(num_samples: int, frame_shift_samples: int = 320) -> int:
    # Xeus uses 20ms stride at 16kHz -> 320 samples per frame.
    return max(1, int(round(num_samples / frame_shift_samples)))


def to_audio_chunks(
    records: Sequence[AudioRecord],
) -> List[AudioChunk]:
    return [
        AudioChunk(
            audio_path=record.audio_path,
            source_mode=record.source_mode,
            language=record.language,
            metadata_key=record.metadata_key,
            metadata_file=record.metadata_file,
            row_index=record.row_index,
            est_num_frames=estimate_num_frames(record.est_target_samples)
            if record.est_target_samples is not None
            else None,
        )
        for record in records
    ]


def chunk_frame_weight(chunk: AudioChunk) -> int:
    return chunk.est_num_frames if chunk.est_num_frames is not None else 1


def pick_bucket_target(totals: Dict[str, int]) -> int | None:
    if not totals:
        return None
    return min(totals.values())


def within_quota(current: int, target: int | None, add_value: int) -> bool:
    if target is None:
        return True
    if current == 0:
        return True
    return current + add_value <= target


def interleave_groups(grouped_items: Dict[str, List[AudioChunk]]) -> List[AudioChunk]:
    order = sorted(grouped_items.keys())
    out: List[AudioChunk] = []
    indices = {k: 0 for k in order}

    while True:
        progressed = False
        for key in order:
            idx = indices[key]
            items = grouped_items[key]
            if idx >= len(items):
                continue
            out.append(items[idx])
            indices[key] += 1
            progressed = True
        if not progressed:
            break

    return out


def select_balanced_chunks(
    chunks: Sequence[AudioChunk],
    seed: int,
) -> List[AudioChunk]:
    by_language: Dict[str, Dict[str, List[AudioChunk]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for chunk in chunks:
        by_language[chunk.language][chunk.metadata_key].append(chunk)

    rng = random.Random(seed)

    selected_by_language: Dict[str, List[AudioChunk]] = {}
    language_available_totals = {
        language: sum(
            chunk_frame_weight(c) for groups in by_meta.values() for c in groups
        )
        for language, by_meta in by_language.items()
    }
    lang_target = pick_bucket_target(language_available_totals)

    for language, by_meta in sorted(by_language.items()):
        per_meta_lists: Dict[str, List[AudioChunk]] = {}
        meta_available_totals: Dict[str, int] = {}

        for meta_key, meta_chunks in sorted(by_meta.items()):
            shuffled = list(meta_chunks)
            rng.shuffle(shuffled)
            per_meta_lists[meta_key] = shuffled
            meta_available_totals[meta_key] = sum(
                chunk_frame_weight(c) for c in shuffled
            )

        meta_target = pick_bucket_target(meta_available_totals)

        meta_order = sorted(per_meta_lists.keys())
        meta_index = {meta: 0 for meta in meta_order}
        meta_frames = {meta: 0 for meta in meta_order}
        selected_meta: Dict[str, List[AudioChunk]] = {meta: [] for meta in meta_order}
        language_frames = 0

        while True:
            progressed = False
            for meta in meta_order:
                idx = meta_index[meta]
                items = per_meta_lists[meta]
                if idx >= len(items):
                    continue

                candidate = items[idx]
                meta_index[meta] += 1
                cand_frames = chunk_frame_weight(candidate)

                if not within_quota(meta_frames[meta], meta_target, cand_frames):
                    continue
                if not within_quota(language_frames, lang_target, cand_frames):
                    continue

                selected_meta[meta].append(candidate)
                meta_frames[meta] += cand_frames
                language_frames += cand_frames
                progressed = True

            if not progressed:
                break

        selected_by_language[language] = interleave_groups(selected_meta)

    return interleave_groups(selected_by_language)


def summarize_chunks(chunks: Sequence[AudioChunk]) -> dict:
    by_language: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"chunks": 0, "est_frames": 0}
    )
    by_metadata_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"chunks": 0, "est_frames": 0}
    )
    by_metadata_language: Dict[str, str] = {}

    total_est_frames = 0
    for chunk in chunks:
        frames = chunk_frame_weight(chunk)
        total_est_frames += frames

        by_language[chunk.language]["chunks"] += 1
        by_language[chunk.language]["est_frames"] += frames

        by_metadata_language[chunk.metadata_key] = chunk.language
        by_metadata_counts[chunk.metadata_key]["chunks"] += 1
        by_metadata_counts[chunk.metadata_key]["est_frames"] += frames

    return {
        "total_chunks": len(chunks),
        "total_est_frames": total_est_frames,
        "languages": {k: by_language[k] for k in sorted(by_language.keys())},
        "metadata": {
            k: {
                "language": by_metadata_language.get(k, "unk"),
                "chunks": by_metadata_counts[k]["chunks"],
                "est_frames": by_metadata_counts[k]["est_frames"],
            }
            for k in sorted(by_metadata_counts.keys())
        },
    }


def print_plan_summary(plan_summary: dict) -> None:
    available = plan_summary["available"]
    selected = plan_summary["selected"]

    print("[plan] ------------------------------")
    print(f"[plan] balance={plan_summary['balance']}")
    print(
        "[plan] available "
        f"chunks={available['total_chunks']} est_frames={available['total_est_frames']}"
    )
    print(
        "[plan] selected "
        f"chunks={selected['total_chunks']} est_frames={selected['total_est_frames']}"
    )

    for language, stats in selected["languages"].items():
        print(
            "[plan][language] "
            f"{language}: chunks={stats['chunks']} est_frames={stats['est_frames']}"
        )


def plan_brief(plan_summary: dict | None) -> dict | None:
    if plan_summary is None:
        return None

    selected = plan_summary["selected"]
    return {
        "balance": plan_summary["balance"],
        "selected_total_chunks": selected["total_chunks"],
        "selected_total_est_frames": selected["total_est_frames"],
        "selected_languages": len(selected["languages"]),
        "selected_metadata_buckets": len(selected["metadata"]),
    }


def load_audio(
    chunk: AudioChunk,
    target_sr: int,
    resamplers: Dict[int, torchaudio.transforms.Resample],
) -> torch.Tensor:
    wavs, sr = sf.read(str(chunk.audio_path))
    wav = torch.from_numpy(np.asarray(wavs)).float()

    if wav.ndim == 2:
        # soundfile returns [T, C] for multi-channel audio.
        wav = wav.mean(dim=1)
    elif wav.ndim != 1:
        wav = wav.reshape(-1)

    if sr != target_sr:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, target_sr)
        wav = resamplers[sr](wav.unsqueeze(0)).squeeze(0)

    return wav


def batch_audio(
    chunk_iter: Iterable[AudioChunk],
    batch_size: int,
    target_sr: int,
    max_files: int | None,
) -> Generator[Tuple[List[AudioChunk], torch.Tensor, torch.LongTensor], None, None]:
    chunks: List[AudioChunk] = []
    waves: List[torch.Tensor] = []
    resamplers: Dict[int, torchaudio.transforms.Resample] = {}
    seen = 0

    for chunk in chunk_iter:
        if max_files is not None and seen >= max_files:
            break

        try:
            wav = load_audio(chunk, target_sr=target_sr, resamplers=resamplers)
        except Exception as exc:
            print(f"[warn] failed to load {chunk.audio_path}: {exc}")
            continue

        chunks.append(chunk)
        waves.append(wav)
        seen += 1

        if len(chunks) == batch_size:
            lengths = torch.LongTensor([w.shape[0] for w in waves])
            padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
            yield chunks, padded, lengths
            chunks, waves = [], []

    if chunks:
        lengths = torch.LongTensor([w.shape[0] for w in waves])
        padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
        yield chunks, padded, lengths


def split_feature_buffer(
    feature_buffer: Sequence[np.ndarray],
    target_rows: int,
    min_rows: int,
    flush: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not feature_buffer:
        return [], []

    merged = np.concatenate(feature_buffer, axis=0)
    ready_batches: List[np.ndarray] = []

    while merged.shape[0] >= target_rows:
        ready_batches.append(merged[:target_rows])
        merged = merged[target_rows:]

    if flush and merged.shape[0] >= min_rows:
        ready_batches.append(merged)
        merged = merged[:0]

    remainder = [merged] if merged.shape[0] > 0 else []
    return ready_batches, remainder


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
        "--root-path",
        type=Path,
        required=True,
        help="Dataset root path used with --metadata-glob",
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
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of loaded training chunks",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on total feature frames used for k-means updates",
    )

    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance selected audio by language and metadata file",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Build and report the data plan without running Xeus or k-means",
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
        help="Number of chunks per Xeus extraction batch",
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


def build_chunks_from_args(args: argparse.Namespace) -> Tuple[List[AudioChunk], dict]:
    metadata_root = Path(os.path.abspath(os.path.expanduser(str(args.root_path))))
    if not metadata_root.exists():
        raise FileNotFoundError(f"--root-path does not exist: {metadata_root}")

    patterns = parse_metadata_glob_arg(args.metadata_glob)
    metadata_files = discover_metadata_files(metadata_root, patterns)
    print(
        "[info] input mode=metadata "
        f"root={metadata_root} patterns={patterns} files={len(metadata_files)}"
    )
    records = list(
        iter_metadata_audio_records(
            metadata_root=metadata_root,
            metadata_files=metadata_files,
            target_sr=args.sample_rate,
            encoding=args.metadata_encoding,
        )
    )

    available_chunks = to_audio_chunks(records)

    if args.balance:
        selected_chunks = select_balanced_chunks(
            chunks=available_chunks,
            seed=args.seed,
        )
    else:
        selected_chunks = available_chunks

    plan_summary = {
        "balance": args.balance,
        "available": summarize_chunks(available_chunks),
        "selected": summarize_chunks(selected_chunks),
    }
    return selected_chunks, plan_summary


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selected_chunks, plan_summary = build_chunks_from_args(args)
    print_plan_summary(plan_summary)

    if args.plan_only:
        print("[done] plan-only run completed")
        return

    print("[info] loading xeus model")
    xeus_model, _ = SSLTask.build_model_from_file(
        str(args.xeus_config),
        str(args.xeus_checkpoint),
        args.device,
    )
    xeus_model = xeus_model.eval()
    xeus_model = xeus_model  # type: ignore[assignment]
    xeus_model_typed: Any = xeus_model

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

    total_files = 0
    total_frames = 0
    total_updates = 0
    dropped_buffered_frames = 0
    source_files_seen: set[str] = set()
    feature_buffer: List[np.ndarray] = []
    partial_fit_rows = max(args.n_clusters, args.partial_fit_chunk_rows)

    def apply_fit_batch(fit_batch: np.ndarray) -> bool:
        nonlocal total_updates, total_frames

        kmeans.partial_fit(fit_batch)

        total_updates += 1
        total_frames += int(fit_batch.shape[0])

        if total_updates % args.checkpoint_updates == 0:
            stats = {
                "total_files": total_files,
                "total_frames": total_frames,
                "total_updates": total_updates,
                "n_clusters": args.n_clusters,
                "feature_dim": int(fit_batch.shape[1]),
                "xeus_layer": args.xeus_layer,
                "sample_rate": args.sample_rate,
                "unique_source_files": len(source_files_seen),
                "dropped_buffered_frames": dropped_buffered_frames,
                "plan": plan_brief(plan_summary),
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
                "feature_dim": int(fit_batch.shape[1]),
                "xeus_layer": args.xeus_layer,
                "sample_rate": args.sample_rate,
                "unique_source_files": len(source_files_seen),
                "dropped_buffered_frames": dropped_buffered_frames,
                "plan": plan_brief(plan_summary),
            }
            save_checkpoint(kmeans, args.out_model, args.out_meta, stats)
            return True

        return False

    for chunks, wav_batch, wav_lengths in batch_audio(
        chunk_iter=iter(selected_chunks),
        batch_size=args.batch_size,
        target_sr=args.sample_rate,
        max_files=args.max_files,
    ):
        with torch.no_grad():
            _, hidden_states, _ = xeus_model_typed.inference_encode(
                wav_batch.to(args.device),
                wav_lengths.to(args.device),
                use_mask=False,
            )

        layer_feats = hidden_states[args.xeus_layer]
        layer_feats = layer_feats.detach().cpu().float().numpy()

        for i, chunk_info in enumerate(chunks):
            feat = layer_feats[i]

            if feat.shape[0] == 0:
                continue

            feat = feat.astype(np.float32, copy=False)
            feature_buffer.append(feat)
            ready_batches, feature_buffer = split_feature_buffer(
                feature_buffer=feature_buffer,
                target_rows=partial_fit_rows,
                min_rows=args.n_clusters,
                flush=False,
            )
            for fit_batch in ready_batches:
                if apply_fit_batch(fit_batch):
                    return

            total_files += 1
            source_files_seen.add(str(chunk_info.audio_path))

            if total_files % 100 == 0:
                frames_m = total_frames / 1_000_000
                print(
                    f"[progress] files={total_files} frames={frames_m:.2f}M updates={total_updates}"
                )

    ready_batches, feature_buffer = split_feature_buffer(
        feature_buffer=feature_buffer,
        target_rows=partial_fit_rows,
        min_rows=args.n_clusters,
        flush=True,
    )
    for fit_batch in ready_batches:
        if apply_fit_batch(fit_batch):
            return

    if feature_buffer:
        dropped_buffered_frames = int(sum(buf.shape[0] for buf in feature_buffer))
        print(
            "[warn] dropping buffered frames "
            f"rows={dropped_buffered_frames} because MiniBatchKMeans requires at least "
            f"{args.n_clusters} rows per partial_fit"
        )

    if total_updates == 0:
        raise RuntimeError(
            "No valid MiniBatchKMeans updates were produced. "
            f"After filtering and buffering, the run never reached the required minimum of "
            f"{args.n_clusters} rows for partial_fit. "
            f"Buffered tail rows={dropped_buffered_frames}."
        )

    stats = {
        "total_files": total_files,
        "total_frames": total_frames,
        "total_updates": total_updates,
        "n_clusters": args.n_clusters,
        "feature_dim": int(kmeans.cluster_centers_.shape[1]),
        "xeus_layer": args.xeus_layer,
        "sample_rate": args.sample_rate,
        "unique_source_files": len(source_files_seen),
        "dropped_buffered_frames": dropped_buffered_frames,
        "plan": plan_brief(plan_summary),
    }
    save_checkpoint(kmeans, args.out_model, args.out_meta, stats)

    print(
        "[done] "
        f"files={total_files} frames={total_frames} updates={total_updates} "
        f"model={args.out_model}"
    )


if __name__ == "__main__":
    main()
