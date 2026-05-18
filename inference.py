import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from espnet2.tasks.ssl import SSLTask

import utils
from modules.mel_processing import MelSpectrogramFeatures
from preprocess.save_codes import ApplyKmeans
from train import VoiceConversionModule


def _get(config, key, default=None):
    if config is None:
        return default
    value = getattr(config, key, default)
    return default if value is None else value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-example GameVC inference from source and target audio."
    )
    parser.add_argument("--source-audio", type=str, required=True)
    parser.add_argument("--target-audio", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional path to the training config JSON. If omitted, the script "
            "tries to resolve config.json from the checkpoint directory."
        ),
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--kmeans-model", type=str, required=True)
    parser.add_argument(
        "--xeus-config",
        type=str,
        default="ckpt/xeus/config.yaml",
        help="Path to the Xeus config YAML.",
    )
    parser.add_argument(
        "--xeus-checkpoint",
        type=str,
        default="ckpt/xeus/xeus_checkpoint_new.pth",
        help="Path to the Xeus checkpoint.",
    )
    parser.add_argument(
        "--xeus-layer",
        type=int,
        default=14,
        help="Hidden-state layer index used for k-means unit extraction.",
    )
    parser.add_argument(
        "--xeus-sample-rate",
        type=int,
        default=16000,
        help="Sample rate expected by Xeus.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device override, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--output-mel",
        type=str,
        default=None,
        help="Optional output .npy path for the predicted mel.",
    )
    parser.add_argument("--n-timesteps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--solver", type=str, default=None)
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_required_files(file_paths):
    for path_str in file_paths:
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Required file not found: {path_str}")


def resolve_config_path(config_arg, checkpoint_path):
    if config_arg is not None:
        return os.path.abspath(os.path.expanduser(config_arg))

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    candidates = [
        checkpoint.parent.parent / "config.json",
        checkpoint.parent / "config.json",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    candidate_list = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve a training config for the checkpoint. "
        f"Tried: {candidate_list}. Pass --config explicitly."
    )


def load_audio(audio_path, sample_rate):
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return torch.from_numpy(audio).to(torch.float32).unsqueeze(0)


def build_mel_extractor(hps, device):
    return MelSpectrogramFeatures(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        hop_length=hps.data.hop_length,
        n_mels=hps.data.n_mel_channels,
        padding=_get(hps.data, "mel_padding", "same"),
    ).to(device)


def extract_units(
    audio_path, sample_rate, xeus_model, apply_kmeans, xeus_layer, device
):
    wav = load_audio(audio_path, sample_rate).to(device)
    wav_lengths = torch.tensor([wav.shape[-1]], dtype=torch.long, device=device)

    with torch.inference_mode():
        _, hidden_states, feat_lengths = xeus_model.inference_encode(
            wav,
            wav_lengths,
            use_mask=False,
        )

    n_layers = len(hidden_states)
    if xeus_layer >= n_layers or xeus_layer < -n_layers:
        raise ValueError(
            f"Invalid --xeus-layer={xeus_layer}. Model returned {n_layers} hidden-state tensors."
        )

    frame_count = int(feat_lengths[0].item())
    features = hidden_states[xeus_layer][0, :frame_count, :]
    return apply_kmeans(features).long().cpu()


def extract_target_mel(audio_path, mel_extractor, sample_rate, device):
    wav = load_audio(audio_path, sample_rate).to(device)
    with torch.inference_mode():
        mel = mel_extractor(wav).squeeze(0)
    return mel.detach().cpu()


def align_target_prompt(target_units, target_mel):
    if target_units.ndim != 1:
        raise ValueError(
            f"Expected 1D target units, got shape {tuple(target_units.shape)}"
        )
    if target_mel.ndim != 2:
        raise ValueError(f"Expected 2D target mel, got shape {tuple(target_mel.shape)}")

    frame_count = min(int(target_units.shape[0]), int(target_mel.shape[-1]))
    if frame_count <= 0:
        raise ValueError("Target prompt has no aligned frames to use for inference.")

    return target_units[:frame_count], target_mel[:, :frame_count]


def load_voice_conversion_module(checkpoint_path, hps, device):
    module = VoiceConversionModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hps=hps,
        map_location=device,
    )
    module = module.to(device)
    module.eval()
    return module


def resolve_inference_value(cli_value, config_value):
    return config_value if cli_value is None else cli_value


def build_output_path(source_audio, target_audio, output_mel):
    if output_mel is not None:
        return Path(output_mel).expanduser().resolve()

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stem = Path(source_audio).stem
    target_stem = Path(target_audio).stem
    return (output_dir / f"{source_stem}__to__{target_stem}.npy").resolve()


def main():
    args = parse_args()

    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint))
    config_path = resolve_config_path(args.config, checkpoint_path)
    xeus_config = os.path.abspath(os.path.expanduser(args.xeus_config))
    xeus_checkpoint = os.path.abspath(os.path.expanduser(args.xeus_checkpoint))
    kmeans_model = os.path.abspath(os.path.expanduser(args.kmeans_model))
    source_audio = os.path.abspath(os.path.expanduser(args.source_audio))
    target_audio = os.path.abspath(os.path.expanduser(args.target_audio))

    validate_required_files(
        (
            config_path,
            checkpoint_path,
            xeus_config,
            xeus_checkpoint,
            kmeans_model,
            source_audio,
            target_audio,
        )
    )

    device = resolve_device(args.device)
    hps = utils.get_hparams_from_file(config_path)

    print(f"Using device: {device}")
    print(f"Loading VC config from {config_path}")
    print(f"Loading VC checkpoint from {checkpoint_path}")
    print(f"Loading Xeus from config={xeus_config}, checkpoint={xeus_checkpoint}")
    print(f"Loading k-means model from {kmeans_model}")

    xeus_model, _ = SSLTask.build_model_from_file(
        xeus_config,
        xeus_checkpoint,
        str(device),
    )
    xeus_model = xeus_model.eval()
    apply_kmeans = ApplyKmeans(model_path=kmeans_model, device=device)
    mel_extractor = build_mel_extractor(hps, device)
    module = load_voice_conversion_module(checkpoint_path, hps, device)

    source_units = extract_units(
        audio_path=source_audio,
        sample_rate=args.xeus_sample_rate,
        xeus_model=xeus_model,
        apply_kmeans=apply_kmeans,
        xeus_layer=args.xeus_layer,
        device=device,
    )
    target_units = extract_units(
        audio_path=target_audio,
        sample_rate=args.xeus_sample_rate,
        xeus_model=xeus_model,
        apply_kmeans=apply_kmeans,
        xeus_layer=args.xeus_layer,
        device=device,
    )
    target_mel = extract_target_mel(
        audio_path=target_audio,
        mel_extractor=mel_extractor,
        sample_rate=hps.data.sampling_rate,
        device=device,
    )
    target_units, target_mel = align_target_prompt(target_units, target_mel)

    if source_units.ndim != 1:
        raise ValueError(
            f"Expected 1D source units, got shape {tuple(source_units.shape)}"
        )
    if target_mel.shape[0] != hps.data.n_mel_channels:
        raise ValueError(
            f"Expected {hps.data.n_mel_channels} mel channels, got {target_mel.shape[0]}"
        )

    source_units_batch = source_units.unsqueeze(0).to(device=device, dtype=torch.long)
    target_units_batch = target_units.unsqueeze(0).to(device=device, dtype=torch.long)
    target_mel_batch = target_mel.unsqueeze(0).to(device=device, dtype=torch.float32)
    source_lengths = torch.tensor(
        [source_units.shape[0]], dtype=torch.long, device=device
    )
    target_lengths = torch.tensor(
        [target_units.shape[0]], dtype=torch.long, device=device
    )

    n_timesteps = resolve_inference_value(args.n_timesteps, hps.inference.n_timesteps)
    temperature = resolve_inference_value(args.temperature, hps.inference.temperature)
    guidance_scale = resolve_inference_value(
        args.guidance_scale, hps.inference.guidance_scale
    )
    solver = resolve_inference_value(args.solver, hps.inference.solver)

    print(
        "Prepared inputs: "
        f"source_units={source_units.shape[0]}, "
        f"target_units={target_units.shape[0]}, "
        f"target_mel={tuple(target_mel.shape)}"
    )
    print(
        "Running inference with "
        f"n_timesteps={n_timesteps}, temperature={temperature}, "
        f"guidance_scale={guidance_scale}, solver={solver}"
    )

    with torch.inference_mode():
        predicted_mel = module.model.infer(
            source_units=source_units_batch,
            target_units=target_units_batch,
            target_mel=target_mel_batch,
            source_lengths=source_lengths,
            target_lengths=target_lengths,
            n_timesteps=n_timesteps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            solver=solver,
        )

    output_path = build_output_path(
        source_audio=source_audio,
        target_audio=target_audio,
        output_mel=args.output_mel,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted_mel_np = predicted_mel[0].detach().cpu().numpy().astype(np.float32)
    np.save(output_path, predicted_mel_np)

    print(f"Saved predicted mel to {output_path}")
    print(f"Predicted mel shape: {predicted_mel_np.shape}")


if __name__ == "__main__":
    main()
