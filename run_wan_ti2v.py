#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
    "背景人很多，倒着走"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a video with Wan2.2-TI2V-5B through Diffusers."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output", default="5bit2v_output.mp4")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16"),
        default="auto",
        help="Transformer dtype. Auto uses BF16 when supported and FP16 otherwise.",
    )
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use Diffusers model CPU offload instead of placing the whole pipeline on CUDA.",
    )
    placement.add_argument(
        "--balanced-device-map",
        action="store_true",
        help="Distribute pipeline components across all available GPUs.",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable VAE tiling to reduce peak decode memory.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.height < 32 or args.width < 32:
        raise ValueError("--height and --width must be at least 32")
    if args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be divisible by 32")
    if args.num_frames < 1:
        raise ValueError("--num-frames must be positive")
    if (args.num_frames - 1) % 4:
        raise ValueError("--num-frames must have the form 4n+1, for example 121")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    if args.guidance_scale <= 0:
        raise ValueError("--guidance-scale must be positive")
    if args.fps < 1:
        raise ValueError("--fps must be positive")
    if args.device_id < 0:
        raise ValueError("--device-id must be non-negative")
    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")


def resolve_transformer_dtype(torch_module: Any, dtype_name: str) -> tuple[Any, str]:
    if dtype_name == "auto":
        bf16_supported = (
            hasattr(torch_module.cuda, "is_bf16_supported")
            and torch_module.cuda.is_bf16_supported()
        )
        dtype_name = "bfloat16" if bf16_supported else "float16"
    return getattr(torch_module, dtype_name), dtype_name


def module_device(module: Any) -> Any:
    return next(module.parameters()).device


def load_pipeline(
    device_id: int,
    cpu_offload: bool,
    vae_tiling: bool,
    dtype_name: str = "auto",
    balanced_device_map: bool = False,
    torch_module: Any | None = None,
    autoencoder_class: Any | None = None,
    pipeline_class: Any | None = None,
) -> Any:
    if torch_module is None:
        import torch as torch_module
    if autoencoder_class is None or pipeline_class is None:
        from diffusers import AutoencoderKLWan, WanPipeline

        autoencoder_class = autoencoder_class or AutoencoderKLWan
        pipeline_class = pipeline_class or WanPipeline

    if not torch_module.cuda.is_available():
        raise RuntimeError("Wan2.2-TI2V-5B requires a CUDA GPU.")
    if device_id >= torch_module.cuda.device_count():
        raise ValueError(
            f"CUDA device {device_id} is unavailable; "
            f"device count is {torch_module.cuda.device_count()}"
        )

    device = f"cuda:{device_id}"
    transformer_dtype, resolved_dtype_name = resolve_transformer_dtype(
        torch_module, dtype_name
    )
    dtype_label = {"bfloat16": "BF16", "float16": "FP16"}[resolved_dtype_name]
    print(
        f"[load] CUDA devices={torch_module.cuda.device_count()}, "
        f"transformer dtype={dtype_label}.",
        flush=True,
    )
    if resolved_dtype_name == "float16":
        print(
            "[load] Using FP16 because this GPU has no native BF16 support. "
            "Use an Ampere-or-newer GPU for the reference BF16 path.",
            flush=True,
        )
    print(f"[load] Loading FP32 Wan VAE from {MODEL_ID}...", flush=True)
    vae = autoencoder_class.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch_module.float32,
    )
    print(
        f"[load] Loading {dtype_label} WanPipeline from {MODEL_ID}...",
        flush=True,
    )
    pipeline_kwargs = {
        "vae": vae,
        "torch_dtype": transformer_dtype,
    }
    if balanced_device_map:
        pipeline_kwargs["device_map"] = "balanced"
    pipe = pipeline_class.from_pretrained(MODEL_ID, **pipeline_kwargs)

    if vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if balanced_device_map:
        print(
            "[load] Using balanced device placement across available GPUs.",
            flush=True,
        )
        if hasattr(pipe, "hf_device_map"):
            print(f"[load] Device map: {pipe.hf_device_map}", flush=True)
        vae_device = module_device(pipe.vae)
        print(
            f"[load] Actual VAE decode device={vae_device}; the VAE will stay "
            f"there and final latents will be moved to it.",
            flush=True,
        )
    elif cpu_offload:
        print(f"[load] Enabling model CPU offload to {device}.", flush=True)
        pipe.enable_model_cpu_offload(gpu_id=device_id)
    else:
        print(f"[load] Moving the complete pipeline to {device}.", flush=True)
        pipe.to(device)
    return pipe


def decode_balanced_latents(pipe: Any, latents: Any, torch_module: Any) -> Any:
    vae_device = module_device(pipe.vae)
    print(
        f"[decode] Moving latent from {latents.device} to VAE on {vae_device}.",
        flush=True,
    )
    latents = latents.to(device=vae_device, dtype=pipe.vae.dtype)
    latents_mean = (
        torch_module.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch_module.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="np")


def generate(
    args: argparse.Namespace,
    pipe: Any,
    torch_module: Any | None = None,
    balanced_decoder: Any | None = None,
) -> Any:
    if torch_module is None:
        import torch as torch_module

    device = f"cuda:{args.device_id}"
    generator = torch_module.Generator(device=device).manual_seed(args.seed)
    print(
        f"[generate] {args.num_frames} frames, {args.width}x{args.height}, "
        f"{args.num_inference_steps} steps, guidance={args.guidance_scale}",
        flush=True,
    )
    forwards_per_step = 2 if args.guidance_scale > 1.0 else 1
    print(
        f"[generate] CFG requires {forwards_per_step} Transformer forward(s) "
        f"per step, {args.num_inference_steps * forwards_per_step} total.",
        flush=True,
    )
    pipeline_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )
    if not args.balanced_device_map:
        return pipe(**pipeline_kwargs).frames[0]

    latents = pipe(**pipeline_kwargs, output_type="latent").frames
    latent_path = Path(args.output).with_suffix(".latent.pt")
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(latents.detach().cpu(), latent_path)
    print(f"[decode] Saved denoised latent to {latent_path.resolve()}.", flush=True)
    if hasattr(torch_module.cuda, "empty_cache"):
        torch_module.cuda.empty_cache()
    decoder = balanced_decoder or decode_balanced_latents
    return decoder(pipe, latents, torch_module)[0]


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    from diffusers.utils import export_to_video

    pipe = load_pipeline(
        device_id=args.device_id,
        cpu_offload=args.cpu_offload,
        vae_tiling=args.vae_tiling,
        dtype_name=args.dtype,
        balanced_device_map=args.balanced_device_map,
    )
    output = generate(args, pipe)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(output, str(output_path), fps=args.fps)
    print(f"[done] Saved video to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
