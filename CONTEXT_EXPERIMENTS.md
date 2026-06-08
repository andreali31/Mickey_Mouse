# Artist Context Experiments

This branch adds artist style and biography conditioning to the original
CLAP-to-Stable-Diffusion training pipeline.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Train All Three Context Modes

```bash
./run_context_experiments.sh
```

This prepares the cache and trains one shared checkpoint for each mode:

- `style`
- `biography`
- `combined`

Generated caches and checkpoints are intentionally excluded from Git because
they are reproducible and large.

## Generate One Cover

```bash
python3 generate.py \
  --ckpt checkpoints/combined/ckpt_e0001 \
  --audio arianasongs/breathin.mp3 \
  --artist ariana \
  --profile-mode combined \
  --steps 30 \
  --seed 0 \
  --out outputs/example.png
```

## Held-Out Evaluation

```bash
python3 evaluate_heldout.py \
  --steps 30 \
  --seed 100 \
  --output-dir outputs/heldout_varied_seeds
```

The evaluator assigns sequential seeds to the nine held-out songs while using
the same song seed across all three context modes.

## Scientific Report

The complete methodology is available at:

- `docs/Mickey_Mouse_Scientific_Methodology.pdf`
- `docs/Mickey_Mouse_Scientific_Methodology.html`

Rebuild the PDF with:

```bash
python3 scripts/build_methodology_pdf.py
```
