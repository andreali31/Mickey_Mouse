"""Generate audio_only, style, biography, and combined outputs for the
held-out songs and assemble per-song comparison sheets.

Same song seed is used across all trained modes so any differences
between cells reflect the conditioning, not random init.
"""

import argparse
from pathlib import Path

import librosa
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
from transformers import ClapModel, ClapProcessor, CLIPTextModel, CLIPTokenizer

from artist_context import load_profiles, profile_text
from model import AudioProjector


SD_ID = "runwayml/stable-diffusion-v1-5"
CLAP_ID = "laion/clap-htsat-unfused"
TARGET_SR = 48000
CLIP_SECONDS = 10

TEST_SONGS = (
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
MODES = ("audio_only", "style", "biography", "combined")


@torch.no_grad()
def audio_embedding(path, clap, processor, device):
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if len(audio) > CLIP_SECONDS * TARGET_SR:
        start = (len(audio) - CLIP_SECONDS * TARGET_SR) // 2
        audio = audio[start:start + CLIP_SECONDS * TARGET_SR]
    inputs = processor(audio=audio, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = clap.get_audio_features(**inputs)
    return output.pooler_output if hasattr(output, "pooler_output") else output


@torch.no_grad()
def text_embedding(text, tokenizer, text_encoder, device):
    tokens = tokenizer(
        [text], padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt",
    )
    return text_encoder(tokens.input_ids.to(device))[0]


def load_mode(mode, device, dtype, num_tokens, ckpt_root):
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    mode_dir = Path(ckpt_root) / mode
    # Pick the latest ckpt_e* folder so this works for any --epochs value
    ckpts = sorted(mode_dir.glob("ckpt_e*"))
    if not ckpts:
        raise SystemExit(f"No checkpoint folder found under {mode_dir}")
    ckpt = ckpts[-1]
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(ckpt / "unet_lora"))
    # Merge LoRA into base UNet so dtype mismatches between fp32 LoRA weights
    # and the fp16 base UNet can't silently NaN the cross-attention. After
    # merging, pipe.unet is a plain UNet again at the base dtype.
    pipe.unet = pipe.unet.to(dtype)
    pipe.unet = pipe.unet.merge_and_unload()
    pipe.to(device)
    projector = AudioProjector(audio_dim=512, num_tokens=num_tokens, hidden_dim=768).to(device)
    projector.load_state_dict(torch.load(ckpt / "projector.pt", map_location=device))
    projector.eval()
    return pipe, projector


def _find_cover(stem):
    """Look for the ground-truth cover across all artist cover dirs."""
    for d in ("arianacovers", "drakecovers", "lesserafimcovers"):
        for ext in (".jpg", ".jpeg", ".png"):
            p = Path(d) / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def comparison_sheet(stem, seed, steps, output_dir, include_caption, include_gt):
    """Assemble a wide sheet with optional GT, optional caption baseline,
    then audio_only / style / biography / combined."""
    columns = []
    if include_gt:
        gt_path = _find_cover(stem)
        if gt_path:
            columns.append(("GROUND TRUTH", Image.open(gt_path).convert("RGB")))
    if include_caption:
        cap_path = output_dir / f"{stem}_caption_combined_{steps}.png"
        if cap_path.exists():
            columns.append(("CAPTION (TEXT-ONLY SD)", Image.open(cap_path).convert("RGB")))
    for mode in MODES:
        p = output_dir / f"{stem}_{mode}_{steps}.png"
        if p.exists():
            columns.append((mode.upper(), Image.open(p).convert("RGB")))

    if not columns:
        return
    width, height = columns[0][1].size
    # normalize sizes
    columns = [(lbl, img.resize((width, height))) for lbl, img in columns]
    header = 52
    sheet = Image.new("RGB", (width * len(columns), height + header), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
    for index, (label, image) in enumerate(columns):
        x = index * width
        sheet.paste(image, (x, header))
        full_label = f"{label} | SEED {seed}" if index == len(columns) - 1 else label
        box = draw.textbbox((0, 0), full_label, font=font)
        draw.text((x + (width - (box[2] - box[0])) / 2, 15), full_label, fill="black", font=font)
    sheet.save(output_dir / f"{stem}_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs/heldout_varied_seeds")
    parser.add_argument("--skip-trained", action="store_true",
                        help="only re-assemble comparison sheets, don't regenerate")
    parser.add_argument("--no-caption", action="store_true",
                        help="skip the caption-baseline column in the sheet")
    parser.add_argument("--no-gt", action="store_true",
                        help="skip the ground-truth column in the sheet")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()

    if not args.skip_trained:
        processor = ClapProcessor.from_pretrained(CLAP_ID)
        clap = ClapModel.from_pretrained(CLAP_ID).to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            SD_ID, subfolder="text_encoder", torch_dtype=dtype,
        ).to(device).eval()

        audio_cache = {
            stem: audio_embedding(path, clap, processor, device)
            for _, stem, path in TEST_SONGS
        }
        empty_text = text_embedding("", tokenizer, text_encoder, device)

        for mode in MODES:
            try:
                print(f"Loading {mode} checkpoint...")
                pipe, projector = load_mode(mode, device, dtype, args.num_tokens, args.ckpt_root)
            except SystemExit as e:
                print(f"[skip] {e}")
                continue
            for song_index, (artist, stem, _) in enumerate(TEST_SONGS):
                song_seed = args.seed + song_index
                print(f"  {stem}: {mode}, seed {song_seed}")
                # Build conditioning in fp32; cast to inference dtype only at
                # the very end. The projector's outputs can otherwise overflow
                # fp16 (>65504) and turn the cross-attention into NaN, which
                # collapses the latent to zero and decodes as uniform gray.
                audio_cond_f = projector(audio_cache[stem].float())
                audio_null_f = projector.null(1, device=device, dtype=torch.float32)
                text_cond_f = text_embedding(
                    profile_text(profiles, artist, mode), tokenizer, text_encoder, device,
                ).float()
                cond = torch.cat([audio_cond_f, text_cond_f], dim=1).to(dtype)
                null = torch.cat([audio_null_f, empty_text.float()], dim=1).to(dtype)
                if torch.isnan(cond).any() or torch.isinf(cond).any():
                    print(f"    [warn] NaN/Inf in cond for {stem}/{mode}; skipping")
                    continue
                generator = torch.Generator(device=device).manual_seed(song_seed)
                image = pipe(
                    prompt_embeds=cond,
                    negative_prompt_embeds=null,
                    height=args.size,
                    width=args.size,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                ).images[0]
                image.save(output_dir / f"{stem}_{mode}_{args.steps}.png")
            del pipe, projector

    seed_lines = []
    for song_index, (_, stem, _) in enumerate(TEST_SONGS):
        song_seed = args.seed + song_index
        comparison_sheet(stem, song_seed, args.steps, output_dir,
                         include_caption=not args.no_caption, include_gt=not args.no_gt)
        seed_lines.append(f"{stem}: {song_seed}")
    (output_dir / "seeds.txt").write_text("\n".join(seed_lines) + "\n", encoding="utf-8")
    print(f"Saved held-out evaluation to {output_dir}")


if __name__ == "__main__":
    main()
