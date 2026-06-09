"""Generate style, biography, and combined outputs for the held-out songs."""

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
    ("ariana", "stuckwithu", "arianasongs/stuckwithu.mp3"),
    ("ariana", "wecantbefriends", "arianasongs/wecantbefriends.mp3"),
    ("ariana", "yesand", "arianasongs/yesand.mp3"),
    ("ariana", "problem", "arianasongs/problem.mp3"),
    ("drake", "inmyfeelings", "drakesongs/inmyfeelings.mp3"),
    ("drake", "laughnowcrylater", "drakesongs/laughnowcrylater.mp3"),
    ("drake", "niceforwhat", "drakesongs/niceforwhat.mp3"),
    ("drake", "passionfruit", "drakesongs/passionfruit.mp3"),
    ("drake", "themotto", "drakesongs/themotto.mp3"),
)
MODES = ("style", "biography", "combined")


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


def load_mode(mode, device, dtype, num_tokens):
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    ckpt = Path("checkpoints") / mode / "ckpt_e0001"
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(ckpt / "unet_lora"))
    pipe.to(device)
    projector = AudioProjector(audio_dim=512, num_tokens=num_tokens, hidden_dim=768).to(device)
    projector.load_state_dict(torch.load(ckpt / "projector.pt", map_location=device))
    projector.eval()
    return pipe, projector


def comparison_sheet(stem, seed, steps, output_dir):
    items = [("MUSIC STYLE", "style"), ("BIOGRAPHY", "biography"), ("COMBINED", "combined")]
    images = [Image.open(output_dir / f"{stem}_{mode}_{steps}.png").convert("RGB") for _, mode in items]
    width, height = images[0].size
    header = 52
    sheet = Image.new("RGB", (width * 3, height + header), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    for index, ((label, _), image) in enumerate(zip(items, images)):
        x = index * width
        sheet.paste(image, (x, header))
        box = draw.textbbox((0, 0), label, font=font)
        full_label = f"{label} | SEED {seed}"
        box = draw.textbbox((0, 0), full_label, font=font)
        draw.text((x + (width - (box[2] - box[0])) / 2, 15), full_label, fill="black", font=font)
    sheet.save(output_dir / f"{stem}_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/heldout")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()

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
        print(f"Loading {mode} checkpoint...")
        pipe, projector = load_mode(mode, device, dtype, args.num_tokens)
        for song_index, (artist, stem, _) in enumerate(TEST_SONGS):
            song_seed = args.seed + song_index
            print(f"Generating {stem}: {mode}, seed {song_seed}")
            audio_cond = projector(audio_cache[stem].float()).to(dtype)
            audio_null = projector.null(1, device=device, dtype=dtype)
            text_cond = text_embedding(
                profile_text(profiles, artist, mode), tokenizer, text_encoder, device,
            ).to(dtype)
            cond = torch.cat([audio_cond, text_cond], dim=1)
            null = torch.cat([audio_null, empty_text.to(dtype)], dim=1)
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
        comparison_sheet(stem, song_seed, args.steps, output_dir)
        seed_lines.append(f"{stem}: {song_seed}")
    (output_dir / "seeds.txt").write_text("\n".join(seed_lines) + "\n", encoding="utf-8")
    print(f"Saved held-out evaluation to {output_dir}")


if __name__ == "__main__":
    main()
