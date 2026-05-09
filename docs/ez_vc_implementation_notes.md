# EZ-VC Implementation Notes

This note captures the implementation-relevant facts from the EZ-VC paper and the current repository state. It is intended as a reference for a later implementation pass, not as an implementation plan.

## Paper Identity

- Paper: `EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion`
- arXiv: `2505.16691v2`
- Core claim: high-quality zero-shot any-to-any VC with a simple textless pipeline based on discrete SSL units plus a non-autoregressive conditional flow matching decoder.

## Exact EZ-VC Architecture

EZ-VC consists of four conceptual stages:

1. Audio to SSL features with a frozen `Xeus` encoder.
2. SSL features to discrete units using `k-means`.
3. Units plus a target mel reference into an `F5-TTS`-style flow-matching decoder that predicts mel.
4. Mel to waveform with `BigVGAN`.

The paper explicitly avoids:

- separate speaker encoders
- explicit speaker embeddings
- multiple disentanglement adapters
- training the SSL encoder itself

## Speech-to-Units Details

- Input audio is `16 kHz`.
- Xeus is used as a frozen multilingual SSL encoder.
- The paper uses features from the `14th` Xeus layer.
- The paper describes this as features taken from about `75%` of model depth.
- Xeus emits frame-level embeddings with:
  - `25 ms` window size
  - `20 ms` stride
  - `50` frames per second
- These 14th-layer embeddings are quantized with a `500`-cluster `k-means` model.
- After quantization, `adjacent repeated units are deduplicated`.

Important constraint:

- The paper says adjacent units are de-duplicated.
- The paper does not explicitly say that repetition counts or durations are passed as a separate conditioning stream.
- For a faithful first reproduction, the safe assumption is: `collapsed unit IDs only`, unless later code evidence shows otherwise.

## Units-to-Speech Details

- The paper uses the original `F5-TTS` implementation as the units-to-mel decoder.
- Model configuration: `base` model, approximately `300M` parameters.
- Reported architecture details:
  - `22` layers
  - `16` attention heads
- Acoustic target:
  - `80`-dimensional log mel-filterbank features
  - `16 kHz` sample rate
  - hop length `160`
- The paper says they use a tokenizer with a vocabulary containing the `500` discrete units.
- In practice, the F5-TTS text/token input is repurposed to accept unit symbols rather than phoneme or text tokens.

## Vocoder Details

- The paper uses a separate `BigVGAN base` vocoder.
- It is trained on `LibriTTS` for `1,000,000` steps.
- Its role is mel-to-waveform only.
- EZ-VC itself is effectively a `units -> mel` system, with waveform generation delegated to BigVGAN.

## Training Data Flow

For each training utterance:

1. Compute the `80`-bin mel spectrogram.
2. Pass the waveform through `Xeus`.
3. Take `encoder_output[14]`-style features.
4. Quantize with the `500`-cluster k-means model.
5. Collapse adjacent duplicate unit IDs.
6. Train the F5-TTS decoder to reconstruct mel from these collapsed units while using the unmasked mel context as the speaker/style reference.

The paper describes the decoder objective as an `infilling task`.

Interpretation that should be preserved later:

- speech content is provided by the discrete units
- speaker attributes are provided by the reference mel context
- the model learns voice conversion without an explicit speaker encoder

## Inference Data Flow

At inference time the paper uses both source and target speech:

1. `Source speech` -> `Xeus` -> `k-means` -> `collapsed source units`
2. `Target speech` -> `Xeus` -> `k-means` -> `collapsed target units`
3. `Target speech` -> target `mel spectrogram`
4. Concatenate `target units` and `source units`
5. Feed the concatenated unit sequence plus target mel reference into the F5-TTS decoder
6. Discard the target-reference mel portion at the output
7. Keep the generated continuation corresponding to the source content in the target voice
8. Vocode the resulting mel with `BigVGAN`

The paper figure makes the conditioning pattern explicit:

- target mel acts as the reference prompt
- target and source units are concatenated as the token stream
- the decoder generates a mel continuation after the reference segment

## Training Setup Reported in the Paper

- Total EZ-VC training data: `12,840` hours
- Languages: English plus `5` Indian languages
- K-means training data: `350` hours total
  - `100` hours English
  - `50` hours each from five Indian languages
- F5-TTS training:
  - batch size `64`
  - `1.35M` updates
  - `4 x NVIDIA RTX 6000 ADA`
  - peak learning rate `5e-5`
  - `100k` warmup steps
  - all other settings unchanged from original F5-TTS base

## Things The Paper Leaves Implicit

These points will likely need confirmation before implementation:

- the exact F5-TTS token formatting for unit IDs
- whether special tokens are added around the unit stream
- how the concatenated target and source unit boundary is represented
- whether deduplication keeps explicit duration counts anywhere internally
- exact masking or infilling schedule used after replacing text tokens with unit tokens
- exact cropping / prompt length strategy for target-reference mel during training
- exact way the target-reference mel segment is removed at inference

These are not blockers for understanding the architecture, but they are the main details to verify from code if the authors release it.

## What Already Exists In This Repo

### Xeus Front End

The notebook [xeus.ipynb](/home/alex/Projekt/cfm-vc/xeus.ipynb) already matches the paper on the main front-end choices:

- Xeus checkpoint is loaded from `ckpt/xeus/`
- features are taken from `feats["encoder_output"][14]`
- a multilingual `500`-unit k-means model is used

The local notebook path is effectively:

`audio -> Xeus -> layer 14 features -> k-means -> unit ids`

### Deduplication Logic

There is already an experimental dedup helper in [testing.ipynb](/home/alex/Projekt/cfm-vc/testing.ipynb).

- Function name: `dedup_seq`
- Behavior: collapses consecutive repeated unit IDs and also computes per-run counts

This is close to the paper. The only caution is that the paper only clearly requires the collapsed sequence, not necessarily the run lengths.

### Current Main VC Stack

The current main model in [models/models.py](/home/alex/Projekt/cfm-vc/models/models.py) is not EZ-VC.

Current repo assumptions:

- input is continuous SSL features, not discrete units
- default SSL dimensionality is `768`
- output is a `128`-channel mel/spectrogram-like representation
- the model uses the current repo’s own conditional flow matching stack, not F5-TTS
- pitch and UV are explicit inputs

This matters because EZ-VC is not a small tweak to the current data path. It is a different conditioning interface.

### Current Default Acoustic Config

In [configs/config.json](/home/alex/Projekt/cfm-vc/configs/config.json):

- sampling rate is `22050`
- hop length is `256`
- mel channels are `128`
- `ssl_dim` is `768`

This does not match EZ-VC’s paper settings of:

- `16 kHz`
- hop `160`
- `80` mel bins
- discrete unit tokens instead of continuous SSL frames

## Local Xeus Asset Facts

- The repository contains [ckpt/xeus/config.yaml](/home/alex/Projekt/cfm-vc/ckpt/xeus/config.yaml)
- The repository contains `kmeans_xeus_500_multilingual.pkl`
- The repository contains `xeus_checkpoint_new.pth`

One detail to remember:

- the local xeus config contains `500` numeric unit entries plus `<unk>` and `<sos/eos>`
- this means the practical symbol inventory may be `500 + specials`, depending on how the downstream tokenizer is built

## Codec Model Facts Relevant Later

### What The Local Repo Already Has

The local file [models/codec_decoder.py](/home/alex/Projekt/cfm-vc/models/codec_decoder.py) provides:

- `CodecDecoder`
- `XCodecDecoder`

`XCodecDecoder` is decoder-only:

- input: codec code indices
- output: `24 kHz` waveform
- checkpoint fields required:
  - `codec_decoder`
  - `fc_post_a`

This matches the stripped-checkpoint path and is sufficient for `code -> audio`, but not for `audio -> code`.

### External Codec Repo Confirms Audio -> Code Is Available

The external repository `lexkoro/gametts-codec` contains the missing encoder-side code.

Relevant files:

- `vq/codec_encoder.py`
- `vq/codec_decoder.py`
- `inference.py`
- `inference_save_code.py`
- `strip_checkpoint_for_tts.py`

### Codec Encoder

From the external repo:

- `CodecEncoder` accepts waveform shaped like `(B, 1, samples)`
- It has `up_ratios=[4, 4, 5, 6]` by default
- Therefore encoder hop length is `4 * 4 * 5 * 6 = 480`
- Test coverage confirms:
  - input `(B, 1, token_count * 480)`
  - output `(B, token_count, 1024)`

So the codec token frame rate is aligned to `24 kHz / 480 = 50 fps`, which matches the same 20 ms frame cadence as EZ-VC’s Xeus units.

This alignment is important because it means:

- Xeus units and codec codes are both effectively on a `50 fps` grid
- sample rate differs (`16 kHz` vs `24 kHz`), but frame cadence is compatible at the sequence level

### Full Audio -> Code Path In The External Repo

The external repo’s `audio -> code` path is not just `CodecEncoder` alone.

In `inference_save_code.py` / `inference.py`, the pipeline is:

1. audio at `24 kHz` -> `CodecEncoder` -> acoustic codec embedding
2. audio resampled to `16 kHz` -> `Wav2Vec2BertModel` -> semantic features
3. take semantic model `hidden_states[16]`
4. pass that through `SemanticEncoder`
5. concatenate semantic features with codec encoder features
6. project with `fc_prior`
7. quantize with `CodecDecoder(..., vq=True)` to get discrete codes

So the full codec encoding path depends on more than the local stripped decoder checkpoint.

Required pieces for full `audio -> code` from that repo:

- `CodecEncoder`
- `Wav2Vec2BertModel`
- `SemanticEncoder`
- `fc_prior`
- `CodecDecoder` in quantization mode

### Code Representation

From `inference_save_code.py`:

- saved codes are taken as `vq_codes[i, 0, :length]`
- codes are saved as integer `.npy`

So the codec is effectively used here as a `single-quantizer` code stream.

### Decoder Path In External Repo

The decoder path is:

1. `vq_code` -> `codec_decoder.quantizer.get_output_from_indices(...)`
2. transpose to channel-first form
3. project with `fc_post_a`
4. run `codec_decoder(..., vq=False)`
5. get waveform at `24 kHz`

This is consistent with the local `XCodecDecoder` wrapper.

### Stripped Checkpoint Facts

`strip_checkpoint_for_tts.py` in the external repo explicitly says the TTS/decoder-only checkpoint keeps:

- `codec_decoder.*`
- `fc_post_a.*`

and drops:

- encoder
- semantic path
- discriminators
- optimizer state

It also stores metadata:

- `sample_rate = 24000`
- `hop_length = 480`

This explains why the local stripped checkpoint is enough for `code -> audio`, but not enough for `audio -> code`.

## Why The Codec May Matter Later

If codec usage becomes relevant later, there are at least two distinct questions:

1. Can the existing stripped local checkpoint decode precomputed codec IDs to waveform?
   - Yes.

2. Can we also derive codec IDs from audio if needed?
   - Yes, but only if we also bring in the encoder-side and semantic-side components from the external repo, not just the stripped decoder checkpoint.

This means the repo currently has:

- a complete `Xeus -> units` front end
- a decoder-only `codec code -> audio` path

and can potentially gain:

- a full `audio -> codec code -> audio` path if the missing encoder-side components are imported or reimplemented from `lexkoro/gametts-codec`

## Important Mismatches To Remember Later

### EZ-VC vs Current Repo

- EZ-VC uses `discrete Xeus units`
- current repo uses `continuous SSL features`
- EZ-VC uses `F5-TTS base`
- current repo uses a custom CFM mel decoder stack
- EZ-VC uses `16 kHz`, hop `160`, `80` mel bins
- current repo defaults to `22.05 kHz`, hop `256`, `128` mel bins

### EZ-VC vs Codec Path

- EZ-VC content units are `Xeus + k-means` discrete units
- codec codes are a different representation entirely
- Xeus and codec both operate on roughly `50 fps` token cadence
- codec path is based on `24 kHz` audio and hop `480`
- local stripped codec checkpoint is decode-only

## High-Value Facts To Preserve For Later

- EZ-VC uses `Xeus layer 14`, not the final layer.
- EZ-VC uses a `500`-cluster multilingual k-means tokenizer.
- EZ-VC explicitly `deduplicates adjacent units`.
- EZ-VC uses `F5-TTS base` as a units-to-mel flow-matching decoder.
- EZ-VC uses `BigVGAN base` for mel-to-waveform.
- The paper’s token stream is `target units + source units`, concatenated.
- The paper’s acoustic reference is the `target mel` prompt.
- The local notebook already matches the paper’s Xeus layer choice and 500-unit tokenizer.
- The local repo does not yet match the paper’s discrete-token training path.
- The local codec checkpoint supports `code -> audio`.
- The external codec repo confirms that `audio -> code` is also possible, but requires encoder-side and semantic-side modules not present in the stripped decoder checkpoint.

## Main Unknowns To Resolve Before Coding

- exact F5-TTS token formatting for discrete unit IDs
- exact prompt / continuation segmentation used in training
- exact masking or infilling schedule after replacing text tokens with unit tokens
- whether any special tokens are inserted around target/source unit spans
- whether the implementation uses only collapsed IDs or also dedup counts internally
- whether the later implementation should stay paper-faithful or intentionally adapt parts to the existing repo’s CFM stack
