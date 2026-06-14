#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from mask_tracking.analysis import TARGET_COLORS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe global text-driven color control from a grayscale video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--color", choices=tuple(TARGET_COLORS), default="magenta")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the generated colorization prompt; --color still defines evaluation.",
    )
    parser.add_argument("--output-dir", default="outputs/gray_colorizing")
    parser.add_argument("--strength", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frame-num",
        type=int,
        default=20,
        help="Number of output frames; internally padded to 4n+1 for the Wan VAE.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--size",
        choices=("832*480", "480*832", "1280*704", "704*1280"),
        default="832*480",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--saturation-threshold", type=float, default=0.20)
    parser.add_argument("--hue-tolerance-degrees", type=float, default=30.0)
    parser.add_argument("--minimum-luma", type=float, default=0.05)
    parser.add_argument(
        "--no-save-denoise-steps",
        action="store_false",
        dest="save_denoise_steps",
        help="Disable saving noisy.mp4 and periodic denoise-step videos.",
    )
    parser.add_argument("--denoise-save-every", type=int, default=10)
    parser.set_defaults(save_denoise_steps=True)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.strength <= 1.0:
        raise ValueError("--strength must be in (0, 1]")
    if args.sampling_steps < 1:
        raise ValueError("--sampling-steps must be positive")
    if args.guide_scale <= 0.0:
        raise ValueError("--guide-scale must be positive")
    if args.max_sequence_length < 1:
        raise ValueError("--max-sequence-length must be positive")
    if args.frame_num < 1:
        raise ValueError("--frame-num must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.denoise_save_every < 1:
        raise ValueError("--denoise-save-every must be positive")
    if not 0.0 <= args.saturation_threshold <= 1.0:
        raise ValueError("--saturation-threshold must be in [0, 1]")
    if not 0.0 <= args.hue_tolerance_degrees <= 180.0:
        raise ValueError("--hue-tolerance-degrees must be in [0, 180]")
    if not 0.0 <= args.minimum_luma <= 1.0:
        raise ValueError("--minimum-luma must be in [0, 1]")
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text")


def pad_video_for_wan(video):
    import numpy as np

    if video.ndim != 4 or video.shape[-1] != 3 or len(video) < 1:
        raise ValueError("video must have shape [F,H,W,3] with at least one frame")
    padding = (1 - len(video)) % 4
    if padding == 0:
        return video, 0
    padded = np.concatenate(
        [video, np.repeat(video[-1:], padding, axis=0)],
        axis=0,
    )
    return padded, padding


def main() -> None:
    args = parse_args()
    validate_args(args)

    import numpy as np

    from mask_tracking.analysis import (
        build_global_color_prompt,
        target_color_analysis,
        to_grayscale_rec709,
        video_statistics,
    )
    from mask_tracking.video_io import (
        make_colorization_comparison,
        mask_to_rgb,
        read_video_clip,
        score_to_rgb,
        write_video,
    )
    from mask_tracking.wan_sdedit import MODEL_ID, WanFullVideoSDEdit, diffusers_version

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = (int(value) for value in args.size.split("*"))
    original, source_fps, preprocessing = read_video_clip(
        args.video,
        width,
        height,
        args.frame_num,
        args.start_frame,
        target_fps=args.fps,
    )
    grayscale = to_grayscale_rec709(original)
    model_grayscale, model_padding = pad_video_for_wan(grayscale)
    prompt = (
        args.prompt.strip()
        if args.prompt is not None
        else build_global_color_prompt(args.color)
    )
    pipeline = WanFullVideoSDEdit()
    fps = args.fps or source_fps
    snapshot_records = []

    def save_snapshot(name, frames, metadata):
        if name == "noisy":
            relative_path = Path("noisy.mp4")
        else:
            relative_path = Path("denoise_steps") / (
                f"step_{metadata['step']:03d}_timestep_{metadata['timestep']:.4f}.mp4"
            )
        write_video(output_dir / relative_path, frames[: args.frame_num], fps)
        snapshot_records.append({**metadata, "path": str(relative_path)})

    generation = pipeline.generate(
        model_grayscale,
        prompt,
        args.strength,
        args.seed,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        max_sequence_length=args.max_sequence_length,
        snapshot_callback=save_snapshot if args.save_denoise_steps else None,
        snapshot_every=args.denoise_save_every,
        negative_prompt=args.negative_prompt,
    )
    generated = generation.generated_raw[: args.frame_num]
    vae_roundtrip = generation.vae_roundtrip[: args.frame_num]
    target_mask, target_score, color_metrics = target_color_analysis(
        grayscale,
        generated,
        args.color,
        saturation_threshold=args.saturation_threshold,
        hue_tolerance_degrees=args.hue_tolerance_degrees,
        minimum_luma=args.minimum_luma,
    )
    comparison = make_colorization_comparison(
        original, grayscale, generated, target_score
    )

    write_video(output_dir / "original_color.mp4", original, fps)
    write_video(output_dir / "grayscale_input.mp4", grayscale, fps)
    write_video(output_dir / "generated_raw.mp4", generated, fps)
    write_video(output_dir / "target_color_score.mp4", score_to_rgb(target_score), fps)
    write_video(output_dir / "target_color_mask.mp4", mask_to_rgb(target_mask), fps)
    write_video(output_dir / "vae_roundtrip.mp4", vae_roundtrip, fps)
    write_video(output_dir / "side_by_side.mp4", comparison, fps)
    np.savez_compressed(
        output_dir / "colorization_arrays.npz",
        original_color=original,
        grayscale_input=grayscale,
        generated_raw=generated,
        target_color_score=target_score,
        target_color_mask=target_mask,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_warning": (
            "Global color-control diagnostic only. The target-color mask measures "
            "color coverage and is not a segmentation or tracking result."
        ),
        "conditioning_policy": (
            "The source clip is converted to three-channel Rec.709 grayscale before "
            "full-video latent SDEdit with text conditioning only."
        ),
        "video": str(Path(args.video).resolve()),
        "model_id": MODEL_ID,
        "prompt": prompt,
        "target_color": args.color,
        "target_rgb": list(TARGET_COLORS[args.color]),
        "target_hsv": color_metrics["target_hsv"],
        "parameters": vars(args),
        "preprocessing": {
            **preprocessing,
            "grayscale_conversion": "Rec.709 luma",
            "requested_output_frames": args.frame_num,
            "wan_model_input_frames": len(model_grayscale),
            "wan_temporal_padding_frames": model_padding,
            "wan_temporal_padding_policy": "repeat final frame, then trim outputs",
        },
        "outputs": {
            "original_color": "original_color.mp4",
            "grayscale_input": "grayscale_input.mp4",
            "generated_raw": "generated_raw.mp4",
            "target_color_score": "target_color_score.mp4",
            "target_color_mask": "target_color_mask.mp4",
            "vae_roundtrip": "vae_roundtrip.mp4",
            "noisy": "noisy.mp4" if args.save_denoise_steps else None,
            "denoise_steps": [record["path"] for record in snapshot_records if record["kind"] == "denoise_step"],
            "side_by_side": "side_by_side.mp4",
            "lossless_arrays": "colorization_arrays.npz",
            "manifest": "manifest.json",
        },
        "metrics": color_metrics,
        "diagnostics": {
            **generation.diagnostics,
            "latent_decode_snapshots": snapshot_records,
            "original_color": video_statistics(original),
            "grayscale_input": video_statistics(grayscale),
            "generated_raw": video_statistics(generated),
            "target_color_score": {
                "min": float(target_score.min()),
                "max": float(target_score.max()),
                "mean": float(target_score.mean()),
                "std": float(target_score.std()),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "diffusers": diffusers_version(),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"Saved grayscale colorization experiment to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
