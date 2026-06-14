#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

from run_wan_ti2v import MODEL_ID, configure_vae_tiling, is_out_of_memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode a saved Wan denoised latent without rerunning denoising."
    )
    parser.add_argument("--latent", required=True)
    parser.add_argument("--output", default="decoded_latent.mp4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vae-tile-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=24)
    return parser


def decode(vae, latents, torch_module):
    device = next(vae.parameters()).device
    latents = latents.to(device=device, dtype=vae.dtype)
    latents_mean = (
        torch_module.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch_module.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]
    return (
        (video / 2 + 0.5)
        .clamp(0, 1)
        .cpu()
        .permute(0, 2, 3, 4, 1)
        .float()
        .numpy()[0]
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.vae_tile_size < 64 or args.vae_tile_size % 8:
        raise ValueError("--vae-tile-size must be at least 64 and divisible by 8")

    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.utils import export_to_video

    latent_path = Path(args.latent)
    latents = torch.load(latent_path, map_location="cpu", weights_only=True)
    print(f"[load] Loaded latent {tuple(latents.shape)} from {latent_path}.", flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    configure_vae_tiling(vae, True, args.vae_tile_size)
    vae.to(args.device)

    try:
        frames = decode(vae, latents, torch)
    except RuntimeError as error:
        if not is_out_of_memory(error) or args.device == "cpu":
            raise
        print("[decode] GPU decode OOM; retrying on CPU.", flush=True)
        vae.to("cpu")
        if hasattr(vae, "clear_cache"):
            vae.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
        frames = decode(vae, latents, torch)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=args.fps)
    print(f"[done] Saved video to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
