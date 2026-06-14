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
        "--cpu-offload",
        action="store_true",
        help="Use Diffusers model CPU offload instead of placing the whole pipeline on CUDA.",
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


def load_pipeline(
    device_id: int,
    cpu_offload: bool,
    vae_tiling: bool,
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
    print(f"[load] Loading FP32 Wan VAE from {MODEL_ID}...", flush=True)
    vae = autoencoder_class.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch_module.float32,
    )
    print(f"[load] Loading BF16 WanPipeline from {MODEL_ID}...", flush=True)
    pipe = pipeline_class.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=torch_module.bfloat16,
    )

    if vae_tiling and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if cpu_offload:
        print(f"[load] Enabling model CPU offload to {device}.", flush=True)
        pipe.enable_model_cpu_offload(gpu_id=device_id)
    else:
        print(f"[load] Moving the complete pipeline to {device}.", flush=True)
        pipe.to(device)
    return pipe


def generate(args: argparse.Namespace, pipe: Any, torch_module: Any | None = None) -> Any:
    if torch_module is None:
        import torch as torch_module

    device = f"cuda:{args.device_id}"
    generator = torch_module.Generator(device=device).manual_seed(args.seed)
    print(
        f"[generate] {args.num_frames} frames, {args.width}x{args.height}, "
        f"{args.num_inference_steps} steps, guidance={args.guidance_scale}",
        flush=True,
    )
    return pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).frames[0]


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    from diffusers.utils import export_to_video

    pipe = load_pipeline(
        device_id=args.device_id,
        cpu_offload=args.cpu_offload,
        vae_tiling=args.vae_tiling,
    )
    output = generate(args, pipe)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(output, str(output_path), fps=args.fps)
    print(f"[done] Saved video to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
