#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe emergent mask tracking via Wan2.2 silhouette editing."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--object", required=True, dest="object_text")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the generated object-edit prompt for counterfactual tests.",
    )
    parser.add_argument("--output-dir", default="outputs/mask_tracking")
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-num", type=int, default=49)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--size",
        choices=("832*480", "480*832", "1280*704", "704*1280"),
        default="832*480",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=100,
        help="Total scheduler steps before strength truncation; actual denoise steps are approximately strength * this value.",
    )
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--mask-score-threshold", type=float, default=0.20)
    parser.add_argument("--morphology-kernel", type=int, default=3)
    parser.add_argument(
        "--no-save-denoise-steps",
        action="store_false",
        dest="save_denoise_steps",
        help="Disable saving noisy.mp4 and periodic denoise-step videos.",
    )
    parser.add_argument(
        "--denoise-save-every",
        type=int,
        default=10,
        help="Save one denoise-step video every N steps and always save the final step.",
    )
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
    if args.frame_num < 1 or (args.frame_num - 1) % 4:
        raise ValueError("--frame-num must have the form 4n+1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.denoise_save_every < 1:
        raise ValueError("--denoise-save-every must be positive")
    if not 0.0 <= args.mask_score_threshold <= 1.0:
        raise ValueError("--mask-score-threshold must be in [0, 1]")
    if args.morphology_kernel < 1:
        raise ValueError("--morphology-kernel must be positive")
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("--prompt must contain non-whitespace text")


def main() -> None:
    args = parse_args()
    validate_args(args)

    import numpy as np

    from mask_tracking.analysis import (
        composite_white_target,
        extract_silhouette_mask,
        temporal_metrics,
        video_statistics,
    )
    from mask_tracking.prompting import build_silhouette_prompt
    from mask_tracking.video_io import (
        make_comparison,
        mask_to_rgb,
        read_video_clip,
        score_to_rgb,
        write_video,
    )
    from mask_tracking.wan_sdedit import MODEL_ID, WanFullVideoSDEdit, diffusers_version

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = (int(value) for value in args.size.split("*"))
    source, source_fps, preprocessing = read_video_clip(
        args.video,
        width,
        height,
        args.frame_num,
        args.start_frame,
        target_fps=args.fps,
    )
    prompt = (
        args.prompt.strip()
        if args.prompt is not None
        else build_silhouette_prompt(args.object_text)
    )
    pipeline = WanFullVideoSDEdit()
    fps = args.fps or source_fps
    snapshot_records = []

    def save_snapshot(name, frames, metadata):
        if name == "noisy":
            relative_path = Path("noisy.mp4")
        else:
            timestep = metadata["timestep"]
            relative_path = Path("denoise_steps") / (
                f"step_{metadata['step']:03d}_timestep_{timestep:.4f}.mp4"
            )
        write_video(output_dir / relative_path, frames, fps)
        snapshot_records.append({**metadata, "path": str(relative_path)})

    generation = pipeline.generate(
        source,
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
    generated_raw = generation.generated_raw
    masks, mask_score = extract_silhouette_mask(
        source,
        generated_raw,
        score_threshold=args.mask_score_threshold,
        morphology_kernel=args.morphology_kernel,
    )
    edited = composite_white_target(source, masks)
    comparison = make_comparison(source, generated_raw, masks, edited)
    write_video(output_dir / "source.mp4", source, fps)
    write_video(output_dir / "generated_raw.mp4", generated_raw, fps)
    write_video(output_dir / "edited.mp4", edited, fps)
    np.savez_compressed(
        output_dir / "composite_arrays.npz",
        source=source,
        mask=masks,
        edited=edited,
    )
    write_video(output_dir / "raw_mask.mp4", mask_to_rgb(masks), fps)
    write_video(output_dir / "mask_score.mp4", score_to_rgb(mask_score), fps)
    write_video(output_dir / "vae_roundtrip.mp4", generation.vae_roundtrip, fps)
    write_video(output_dir / "side_by_side.mp4", comparison, fps)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_warning": (
            "No-GT discovery probe. generated_raw.mp4 is model evidence; "
            "edited.mp4 is composited from the source and extracted mask."
        ),
        "conditioning_policy": (
            "Full source-video latent SDEdit with text conditioning only. No first-frame "
            "image condition is injected; every frame is noised, denoised, and evaluated."
        ),
        "object": args.object_text,
        "prompt": prompt,
        "video": str(Path(args.video).resolve()),
        "model_id": MODEL_ID,
        "parameters": vars(args),
        "preprocessing": preprocessing,
        "metrics": temporal_metrics(masks, skip_first_frame=False),
        "diagnostics": {
            **generation.diagnostics,
            "latent_decode_snapshots": snapshot_records,
            "source": video_statistics(source),
            "edited_composite": video_statistics(edited),
            "mask_score": {
                "min": float(mask_score.min()),
                "max": float(mask_score.max()),
                "mean": float(mask_score.mean()),
                "std": float(mask_score.std()),
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
    print(f"Saved experiment to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
