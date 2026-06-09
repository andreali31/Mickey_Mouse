"""
LoRA fine-tune of SD 1.5 UNet, conditioned on CLAP audio embeddings via
the AudioProjector (model.py). VAE latents and audio embeddings are
precomputed by prepare_data.py.

Loss: standard noise-prediction MSE.
CFG: with probability `cfg_drop`, replace the audio condition with the
projector's learned null token so we can do classifier-free guidance at
inference time.
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

from model import AudioProjector
from artist_context import PROFILE_MODES, load_profiles, profile_text

SD_ID = "runwayml/stable-diffusion-v1-5"


class CachedPairs(Dataset):
    def __init__(self, cache_dir: str):
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        self.files = [f for f in self.files if f.name != "index.pt"]
        if not self.files:
            raise RuntimeError(f"No cached pairs in {cache_dir}. Run prepare_data.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu")
        return d["latent"].float(), d["audio_emb"].float(), d["artist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--num-tokens", type=int, default=8)
    ap.add_argument("--cfg-drop", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile-mode", required=True, choices=PROFILE_MODES)
    ap.add_argument("--profile-file", default="artist_profiles.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    if device != "cuda":
        dtype = torch.float32
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading UNet + scheduler...")
    unet = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet")
    noise_sched = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder").to(device).eval()
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.to(device)
    unet.print_trainable_parameters()

    projector = AudioProjector(audio_dim=512, num_tokens=args.num_tokens, hidden_dim=768).to(device)
    profiles = load_profiles(args.profile_file)

    @torch.no_grad()
    def encode_prompts(prompts):
        tokens = tokenizer(prompts, padding="max_length", truncation=True,
                           max_length=tokenizer.model_max_length, return_tensors="pt")
        text_cond = text_encoder(tokens.input_ids.to(device))[0]
        empty = tokenizer([""] * len(prompts), padding="max_length",
                          max_length=tokenizer.model_max_length, return_tensors="pt")
        text_null = text_encoder(empty.input_ids.to(device))[0]
        return text_cond, text_null

    context_cache = {}
    for artist in profiles:
        prompt = profile_text(profiles, artist, args.profile_mode)
        cond, null = encode_prompts([prompt])
        context_cache[artist] = cond.squeeze(0).cpu()
    _, base_text_null = encode_prompts([""])
    base_text_null = base_text_null.squeeze(0).cpu()
    del text_encoder, tokenizer

    def context_for(artists):
        cond = torch.stack([context_cache[artist] for artist in artists]).to(device)
        null = base_text_null.unsqueeze(0).expand(len(artists), -1, -1).to(device)
        return cond, null

    trainable = [p for p in unet.parameters() if p.requires_grad] + list(projector.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    ds = CachedPairs(args.cache_dir)
    print(f"Dataset: {len(ds)} pairs")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    global_step = 0
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        for latents, audio_emb, artists in loader:
            latents = latents.to(device, dtype=torch.float32)
            audio_emb = audio_emb.to(device, dtype=torch.float32)
            bsz = latents.shape[0]

            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (bsz,), device=device).long()
            noisy = noise_sched.add_noise(latents, noise, t)

            audio_cond = projector(audio_emb)
            text_cond, batch_text_null = context_for(artists)
            cond = torch.cat([audio_cond, text_cond], dim=1)
            if args.cfg_drop > 0:
                drop = (torch.rand(bsz, device=device) < args.cfg_drop)
                if drop.any():
                    audio_null = projector.null(bsz, device=device, dtype=cond.dtype)
                    null = torch.cat([audio_null, batch_text_null.to(cond.dtype)], dim=1)
                    cond = torch.where(drop.view(-1, 1, 1), null, cond)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
                pred = unet(noisy.to(dtype), t, encoder_hidden_states=cond.to(dtype)).sample
                loss = F.mse_loss(pred.float(), noise.float())

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()

            global_step += 1
            pbar.set_description(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ck = Path(args.out_dir) / f"ckpt_e{epoch+1:04d}"
            ck.mkdir(parents=True, exist_ok=True)
            unet.save_pretrained(ck / "unet_lora")
            torch.save(projector.state_dict(), ck / "projector.pt")
            torch.save(vars(args), ck / "train_args.pt")
            print(f"\nSaved {ck}")


if __name__ == "__main__":
    main()
