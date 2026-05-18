# Suggested LLM-VC Approach For This Repo

This note captures the suggested `LLM-VC` adaptation path for this repository. It is not a paper-faithful reproduction of EZ-VC. Instead, it describes a practical path that reuses the existing `cfm-tts` / codec-style setup while replacing text with discrete speech units.

For related background on the paper-faithful EZ-VC path and repo facts, see `ez_vc_implementation_notes.md`.

## Core Idea

Use the current text-to-codec formulation as a template, but replace text tokens with `Xeus + k-means` unit IDs extracted from source speech.

Recommended first-pass conditioning:

1. `Source speech -> Xeus layer 14 -> 500-cluster k-means -> frame-level unit IDs`
2. `Target speech -> speaker/style conditioning`
3. `Decoder / LM -> predict codec codes`
4. `Codec decoder -> waveform`

In short:

- content comes from `source Xeus units`
- voice/style comes from `target conditioning`
- output is `codec codes`, not mel

## Why This Is Plausible

The key repo-specific advantage is temporal alignment:

- Xeus units are produced at about `50 fps`
- the codec code stream is also produced at `50 fps`

That means this is not the same problem as text-to-speech, where short text must be expanded into a much longer acoustic sequence. Here the source unit stream already carries frame-level timing information on approximately the same grid as the codec targets.

This makes a direct `units -> codec codes` mapping much more plausible than a `text -> codec codes` mapping for VC.

## Main Recommendation On Dedup

For the LLM-VC path, the safest first baseline is:

- `do not deduplicate the k-means units`

Reason:

- repeated frame-level units already carry coarse duration information
- the codec targets live on the same `50 fps` cadence
- plain dedup would throw away one of the strongest alignment signals available in this setup

In EZ-VC, dedup makes sense because the downstream decoder is explicitly designed around unit-token conditioning plus a target mel prompt. For a codec-prediction LM, removing local repetition too early is more likely to hurt than help unless duration is modeled separately.

## What Target Conditioning Should Provide

The target conditioning should primarily provide:

- speaker identity
- timbre
- style / prosody prior

It should not be assumed to fully reconstruct source-utterance timing after dedup.

A pooled speaker latent can provide useful global priors such as average speaking style, but it is weaker than a richer target prompt when the goal is high-quality zero-shot VC.

Practical implication:

- `target prompt audio` or a richer latent stack is preferable to only a single pooled speaker embedding

## Recommended First Baseline

The lowest-risk experimental path is:

1. extract `frame-level` Xeus k-means units from source speech
2. do `not` dedup them
3. encode target speaker/style with the existing latent pathway or a short target prompt
4. train the decoder to predict codec codes on the same frame cadence
5. decode those codec codes with the existing codec decoder

This baseline preserves duration information and tests the core hypothesis directly.

## Better Compression Options Later

If sequence compression becomes necessary later, better options than plain dedup are:

- dedup plus explicit run-length counts
- dedup plus a learned duration predictor
- small frame grouping / chunking that reduces length without deleting duration completely

These are safer than collapsing to unique-adjacent IDs alone.

## Working Assumption To Preserve

For this repo's proposed LLM-VC route, the best current default assumption is:

- use `non-deduplicated` Xeus k-means units as the source content stream for codec prediction

Only revisit dedup after there is an explicit duration strategy or evidence that the raw frame-level sequence is too redundant for the decoder to learn from effectively.
