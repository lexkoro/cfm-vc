# Conditional Flow Matching Voice Conversion

## 🛠️ ToDo

## Usage

Extract Xeus kmeans unit IDs from metadata-listed audio and save `.npy` files beside each audio file:

```bash
uv run python preprocess/save_codes.py \
	--root-path /path/to/dataset_root \
	--metadata-glob "*/*_metadata.csv" \
	--kmeans-model ckpt/xeus/kmeans_xeus.pkl
```

Notes:
- `--kmeans-model` is required.
- `--xeus-config` and `--xeus-checkpoint` default to files in `ckpt/xeus/`.
- Output path is `<audio_path_without_ext>.npy` (for example `foo/bar/sample.mp3 -> foo/bar/sample.npy`).

Multi-GPU (2 ranks) example:

```bash
# Terminal 1
LOCAL_RANK=0 RANK=0 WORLD_SIZE=2 uv run python preprocess/save_codes.py \
	--root-path /path/to/dataset_root \
	--metadata-glob "*/*_metadata.csv" \
	--kmeans-model ckpt/xeus/kmeans_xeus.pkl

# Terminal 2
LOCAL_RANK=1 RANK=1 WORLD_SIZE=2 uv run python preprocess/save_codes.py \
	--root-path /path/to/dataset_root \
	--metadata-glob "*/*_metadata.csv" \
	--kmeans-model ckpt/xeus/kmeans_xeus.pkl
```

## Training (WandB Logging)

The trainer logs metrics and spectrogram images to Weights & Biases via Lightning's `WandbLogger`.

Online run:

```bash
wandb login
WANDB_PROJECT=cfm-vc uv run python train.py -c configs/config.json -m exp-name
```

Offline run:

```bash
WANDB_MODE=offline WANDB_PROJECT=cfm-vc uv run python train.py -c configs/config.json -m exp-name
```

Notes:
- Run name defaults to the `-m` model name.
- Training config is attached to the WandB run automatically.

## Inference

Run single-example GameVC inference from source audio and target audio:

```bash
uv run python inference.py \
	--source-audio /path/to/source.mp3 \
	--target-audio /path/to/target.mp3 \
	--checkpoint logs/gamevc/checkpoints/last-EMA.ckpt \
	--kmeans-model ckpt/xeus/kmeans_xeus.pkl
```

Notes:
- If `--config` is omitted, the script resolves `config.json` from the checkpoint directory first, for example `logs/gamevc/checkpoints/last-EMA.ckpt -> logs/gamevc/config.json`.
- The current script saves the predicted mel spectrogram as `.npy`; waveform synthesis is not wired in because the available Vocos checkpoint does not match this repo's mel hop size.

