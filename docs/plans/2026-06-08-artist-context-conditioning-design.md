# Artist Context Conditioning Design

## Goal

Train three directly comparable shared audio-to-cover models for one epoch each
using all Ariana Grande, Drake, and LE SSERAFIM examples:

1. Musical and visual style context
2. Biographical context
3. Combined style and biographical context

## Design

The existing CLAP audio embedding remains the primary condition. Each artist's
curated context is encoded with Stable Diffusion 1.5's pretrained tokenizer and
text encoder. The resulting text tokens are concatenated with the learned audio
tokens before they are passed to UNet cross-attention. This gives useful context
from the first training step rather than requiring a new text projection layer
to learn semantic meaning from the small dataset.

Artist profiles live in a versioned JSON file and include source URLs. Cached
pairs retain the artist identifier. Training selects one profile mode and loads
the corresponding text embedding for every sample. Classifier-free dropout
replaces both audio and text context with their unconditional counterparts.

Generation accepts the artist identifier and profile mode, reconstructs the
same audio-plus-text condition, and validates that its mode matches the saved
checkpoint metadata.

## Data Compatibility

Data preparation supports both the documented `data/<artist>/audio|covers`
layout and the repository's checked-in top-level song and cover folders. PNG,
JPG, and JPEG covers are accepted and paired by normalized filename stem.

## Runs

All runs use one shared model, all available pairs, one epoch, the same seed,
and otherwise identical hyperparameters. Outputs are isolated under:

- `checkpoints/style`
- `checkpoints/biography`
- `checkpoints/combined`

## Verification

Run syntax checks and focused unit tests for pairing, profile selection, and
condition construction. Then prepare the cache once and execute the three
one-epoch training runs.
