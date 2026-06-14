#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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
    parser.add_argument(
        "--vae-tile-size",
        type=int,
        default=128,
        help="Spatial VAE tile size used with --vae-tiling. Smaller uses less memory.",
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
    if args.vae_tile_size < 64 or args.vae_tile_size % 8:
        raise ValueError("--vae-tile-size must be at least 64 and divisible by 8")
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


def mapped_component_device(pipe: Any, component_name: str, fallback: Any) -> str:
    mapped_device = getattr(pipe, "hf_device_map", {}).get(component_name, fallback)
    if isinstance(mapped_device, int):
        return f"cuda:{mapped_device}"
    return str(mapped_device)


def configure_vae_tiling(vae: Any, enabled: bool, tile_size: int) -> None:
    if not enabled or not hasattr(vae, "enable_tiling"):
        return
    stride = tile_size * 3 // 4
    print(
        f"[load] Enabling VAE tiling with tile={tile_size}, stride={stride}.",
        flush=True,
    )
    vae.enable_tiling(
        tile_sample_min_height=tile_size,
        tile_sample_min_width=tile_size,
        tile_sample_stride_height=stride,
        tile_sample_stride_width=stride,
    )


def is_out_of_memory(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def prepare_balanced_vae_decode(pipe: Any, torch_module: Any) -> Any:
    current_device = module_device(pipe.vae)
    target_device = mapped_component_device(pipe, "vae", current_device)
    if str(current_device) == target_device:
        return current_device

    released = []
    for component_name in ("text_encoder", "transformer", "transformer_2"):
        if getattr(pipe, component_name, None) is not None:
            setattr(pipe, component_name, None)
            released.append(component_name)

    print(
        f"[decode] Released {', '.join(released) or 'no model components'}; "
        f"moving VAE from {current_device} to {target_device}.",
        flush=True,
    )
    gc.collect()
    torch_module.cuda.empty_cache()

    try:
        pipe.vae.to(target_device)
    except RuntimeError as error:
        if not is_out_of_memory(error):
            raise
        print(
            f"[decode] VAE did not fit on {target_device}; falling back to CPU decode.",
            flush=True,
        )
        pipe.vae.to("cpu")
        gc.collect()
        torch_module.cuda.empty_cache()
    return module_device(pipe.vae)


def load_pipeline(
    device_id: int,
    cpu_offload: bool,
    vae_tiling: bool,
    vae_tile_size: int = 128,
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

    configure_vae_tiling(pipe.vae, vae_tiling, vae_tile_size)
    if balanced_device_map:
        print(
            "[load] Using balanced device placement across available GPUs.",
            flush=True,
        )
        if hasattr(pipe, "hf_device_map"):
            print(f"[load] Device map: {pipe.hf_device_map}", flush=True)
        vae_device = module_device(pipe.vae)
        print(
            f"[load] Initial VAE device={vae_device}; after denoising, unused "
            f"models will be released before VAE placement for decode.",
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


def decode_with_cpu_fallback(
    pipe: Any,
    latents: Any,
    torch_module: Any,
    decoder: Any = decode_balanced_latents,
) -> Any:
    try:
        return decoder(pipe, latents, torch_module)
    except RuntimeError as error:
        if not is_out_of_memory(error) or str(module_device(pipe.vae)) == "cpu":
            raise
        print(
            "[decode] GPU VAE decode ran out of memory; retrying from the saved "
            "CPU latent with CPU VAE decode.",
            flush=True,
        )

    pipe.vae.to("cpu")
    if hasattr(pipe.vae, "clear_cache"):
        pipe.vae.clear_cache()
    gc.collect()
    torch_module.cuda.empty_cache()
    return decoder(pipe, latents.detach().cpu(), torch_module)


def generate(
    args: argparse.Namespace,
    pipe: Any,
    torch_module: Any | None = None,
    balanced_decoder: Any | None = None,
    balanced_preparer: Any | None = None,
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
    cpu_latents = latents.detach().cpu()
    latent_path = Path(args.output).with_suffix(".latent.pt")
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(cpu_latents, latent_path)
    print(f"[decode] Saved denoised latent to {latent_path.resolve()}.", flush=True)
    del latents
    gc.collect()
    if hasattr(torch_module.cuda, "empty_cache"):
        torch_module.cuda.empty_cache()
    preparer = balanced_preparer or prepare_balanced_vae_decode
    preparer(pipe, torch_module)
    decoder = balanced_decoder or decode_balanced_latents
    return decode_with_cpu_fallback(pipe, cpu_latents, torch_module, decoder)[0]


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    from diffusers.utils import export_to_video

    pipe = load_pipeline(
        device_id=args.device_id,
        cpu_offload=args.cpu_offload,
        vae_tiling=args.vae_tiling,
        vae_tile_size=args.vae_tile_size,
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
