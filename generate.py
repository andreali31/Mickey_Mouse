"""
Given an mp3 and a trained checkpoint, generate an album cover.

Loads SD 1.5 pipeline, attaches the LoRA-adapted UNet and the AudioProjector,
encodes the mp3 with CLAP, builds prompt_embeds from the projector, and runs
the diffusion sampler with classifier-free guidance against the projector's
learned null token.
"""

import argparse
from pathlib import Path

import librosa
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import ClapModel, ClapProcessor
from peft import PeftModel

from model import AudioProjector

SD_ID = "runwayml/stable-diffusion-v1-5"
CLAP_ID = "laion/clap-htsat-unfused"
TARGET_SR = 48000
CLIP_SECONDS = 10


@torch.no_grad()
def audio_to_cond(mp3_path, clap, processor, projector, device, dtype):
    audio, _ = librosa.load(mp3_path, sr=TARGET_SR, mono=True)
    if len(audio) > CLIP_SECONDS * TARGET_SR:
        s = (len(audio) - CLIP_SECONDS * TARGET_SR) // 2
        audio = audio[s : s + CLIP_SECONDS * TARGET_SR]
    inputs = processor(audios=audio, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = clap.get_audio_features(**inputs).to(torch.float32)
    cond = projector(emb).to(dtype)
    null = projector.null(emb.shape[0], device=device, dtype=dtype)
    return cond, null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to a checkpoints/ckpt_eXXXX folder")
    ap.add_argument("--audio", required=True, help="path to an mp3 to condition on")
    ap.add_argument("--out", default="outputs/cover.png")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--num-tokens", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    print("Attaching LoRA adapter...")
    lora_dir = Path(args.ckpt) / "unet_lora"
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_dir))
    pipe.unet.to(device, dtype=dtype)

    print("Loading projector...")
    projector = AudioProjector(audio_dim=512, num_tokens=args.num_tokens, hidden_dim=768).to(device)
    projector.load_state_dict(torch.load(Path(args.ckpt) / "projector.pt", map_location=device))
    projector.eval()

    print("Loading CLAP...")
    processor = ClapProcessor.from_pretrained(CLAP_ID)
    clap = ClapModel.from_pretrained(CLAP_ID).to(device).eval()

    cond, null = audio_to_cond(args.audio, clap, processor, projector, device, dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    out = pipe(
        prompt_embeds=cond,
        negative_prompt_embeds=null if args.guidance > 1.0 else None,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    img = out.images[0]
    img.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
