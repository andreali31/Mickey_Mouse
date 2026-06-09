"""
Caption-only baseline: no audio, no LoRA, no fine-tuning. Pretrained
Stable Diffusion 1.5 is prompted with the artist profile text and asked
to generate an album cover. Serves as the "text-only" reference point
opposite the audio-only ablation.

Generates one image per (artist, mode, seed) and writes them under the
provided output dir using the same filename convention as
evaluate_heldout.py so make_paper_figures.py can pick them up:

    {stem}_caption_{mode}_{steps}.png

By default it runs on the same held-out songs as evaluate_heldout.py so
the comparison sheets line up. (The audio path is unused — only the stem
matters here.)
"""

import argparse
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from artist_context import load_profiles, profile_text

SD_ID = "runwayml/stable-diffusion-v1-5"

DEFAULT_SONGS = (
    # 4 ariana, 5 drake, 3 lesserafim = 12 held-out songs
    ("ariana",     "stuckwithu",        "arianasongs/stuckwithu.mp3"),
    ("ariana",     "wecantbefriends",   "arianasongs/wecantbefriends.mp3"),
    ("ariana",     "yesand",            "arianasongs/yesand.mp3"),
    ("ariana",     "problem",           "arianasongs/problem.mp3"),
    ("drake",      "inmyfeelings",      "drakesongs/inmyfeelings.mp3"),
    ("drake",      "laughnowcrylater",  "drakesongs/laughnowcrylater.mp3"),
    ("drake",      "niceforwhat",       "drakesongs/niceforwhat.mp3"),
    ("drake",      "passionfruit",      "drakesongs/passionfruit.mp3"),
    ("drake",      "themotto",          "drakesongs/themotto.mp3"),
    ("lesserafim", "antifragile",       "lesserafimsongs/antifragile.mp3"),
    ("lesserafim", "perfectnight",      "lesserafimsongs/perfectnight.mp3"),
    ("lesserafim", "unforgiven",        "lesserafimsongs/unforgiven.mp3"),
)
CAPTION_MODES = ("style", "biography", "combined")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/heldout_varied_seeds")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=100,
                    help="base seed; each song gets (seed + index) to match evaluate_heldout.py")
    ap.add_argument("--mode", choices=(*CAPTION_MODES, "all"), default="combined",
                    help="which context mode to use as the prompt; 'all' runs every mode")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained SD 1.5 (no LoRA)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    profiles = load_profiles()
    modes = CAPTION_MODES if args.mode == "all" else (args.mode,)

    for mode in modes:
        for song_index, (artist, stem, _) in enumerate(DEFAULT_SONGS):
            song_seed = args.seed + song_index
            prompt = profile_text(profiles, artist, mode)
            print(f"[caption/{mode}] {stem} seed={song_seed}")
            generator = torch.Generator(device=device).manual_seed(song_seed)
            image = pipe(
                prompt=prompt,
                negative_prompt="",
                height=args.size,
                width=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
            image.save(out_dir / f"{stem}_caption_{mode}_{args.steps}.png")

    print(f"Saved caption baseline images to {out_dir}")


if __name__ == "__main__":
    main()
