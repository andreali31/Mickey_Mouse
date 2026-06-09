# Held-Out Context Evaluation Design

## Goal

Evaluate the existing style, biography, and combined-context checkpoints on
nine songs that were excluded from training because their original repository
files were empty or lacked matching covers.

## Test Set

- Ariana Grande: `stuckwithu`, `wecantbefriends`, `yesand`, `problem`
- Drake: `inmyfeelings`, `laughnowcrylater`, `niceforwhat`, `passionfruit`,
  `themotto`

The repaired remote files are valid MP3s. `problem` and `themotto` now also
have same-stem covers.

## Evaluation

For every song, generate one image from each existing one-epoch checkpoint:

1. Style context
2. Biography context
3. Combined context

All generations use 30 diffusion steps, seed 0, guidance 5.0, and 256 by 256
output. Each song gets a labeled three-column comparison sheet. Because these
nine audio files were absent from the training cache, this remains a held-out
test of context-conditioned generation.
