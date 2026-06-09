"""
Find each artist's audio and cover folders, pair files by stem,
extract a CLAP audio embedding and the SD VAE latent for the cover, and cache
both as a single .pt per pair under cache/.

Pairing rule: data/<artist>/audio/<stem>.mp3  <->  data/<artist>/covers/<stem>.jpg
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor
from diffusers import AutoencoderKL
from torchvision import transforms

CLAP_ID = "laion/clap-htsat-unfused"
SD_ID = "runwayml/stable-diffusion-v1-5"
TARGET_SR = 48000
CLIP_SECONDS = 10
IMG_SIZE = 256


TOP_LEVEL_LAYOUT = {
    "ariana": ("arianasongs", "arianacovers"),
    "drake": ("drakesongs", "drakecovers"),
    "lesserafim": ("lesserafimsongs", "lesserafimcovers"),
}


def _cover_for_stem(cover_dir, stem):
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = cover_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def find_pairs(data_root: Path):
    pairs = []
    roots = []
    if data_root.exists():
        for artist_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
            roots.append((artist_dir.name, artist_dir / "audio", artist_dir / "covers"))
    repo_root = Path(__file__).resolve().parent
    for artist, (audio_name, cover_name) in TOP_LEVEL_LAYOUT.items():
        roots.append((artist, repo_root / audio_name, repo_root / cover_name))

    seen = set()
    for artist, audio_dir, cover_dir in roots:
        if not (audio_dir.exists() and cover_dir.exists()):
            continue
        for mp3 in sorted(audio_dir.glob("*.mp3")):
            if mp3.stat().st_size < 1024:
                print(f"[warn] skipping invalid or empty audio file {mp3}")
                continue
            cover = _cover_for_stem(cover_dir, mp3.stem)
            key = (artist, mp3.stem)
            if cover and key not in seen:
                pairs.append((artist, mp3, cover))
                seen.add(key)
            else:
                print(f"[warn] no cover for {mp3}")
    return pairs


@torch.no_grad()
def encode_audio(mp3_path: Path, clap, processor, device):
    audio, _ = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
    if len(audio) > CLIP_SECONDS * TARGET_SR:
        start = (len(audio) - CLIP_SECONDS * TARGET_SR) // 2
        audio = audio[start : start + CLIP_SECONDS * TARGET_SR]
    inputs = processor(audio=audio, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = clap.get_audio_features(**inputs)
    if hasattr(emb, "pooler_output"):
        emb = emb.pooler_output
    return emb.squeeze(0).cpu()


@torch.no_grad()
def encode_image(jpg_path: Path, vae, tfm, device):
    img = Image.open(jpg_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device, dtype=vae.dtype)
    latent = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    return latent.squeeze(0).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(data_root)
    print(f"Found {len(pairs)} (audio, cover) pairs")
    if not pairs:
        raise SystemExit("No pairs found. Drop mp3s in data/<artist>/audio and jpgs in data/<artist>/covers with matching stems.")

    print("Loading CLAP...")
    processor = ClapProcessor.from_pretrained(CLAP_ID)
    clap = ClapModel.from_pretrained(CLAP_ID).to(args.device).eval()

    print("Loading SD VAE...")
    vae = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae", torch_dtype=torch.float32).to(args.device).eval()

    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    index = []
    for artist, mp3, jpg in tqdm(pairs):
        key = f"{artist}__{mp3.stem}"
        out = cache_dir / f"{key}.pt"
        if out.exists():
            index.append({"key": key, "artist": artist, "audio": str(mp3), "cover": str(jpg), "cache": str(out)})
            continue
        audio_emb = encode_audio(mp3, clap, processor, args.device)
        latent = encode_image(jpg, vae, tfm, args.device)
        torch.save({"audio_emb": audio_emb, "latent": latent, "artist": artist, "stem": mp3.stem}, out)
        index.append({"key": key, "artist": artist, "audio": str(mp3), "cover": str(jpg), "cache": str(out)})

    torch.save(index, cache_dir / "index.pt")
    print(f"Cached {len(index)} pairs → {cache_dir}")


if __name__ == "__main__":
    main()
