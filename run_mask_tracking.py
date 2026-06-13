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
    parser.add_argument("--sampling-steps", type=int, default=30)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--white-threshold", type=int, default=220)
    parser.add_argument("--difference-threshold", type=float, default=35.0)
    parser.add_argument("--morphology-kernel", type=int, default=3)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()

    from mask_tracking.analysis import extract_silhouette_mask, temporal_metrics
    from mask_tracking.prompting import build_silhouette_prompt
    from mask_tracking.video_io import (
        make_comparison,
        make_overlay,
        mask_to_rgb,
        read_video_clip,
        write_video,
    )
    from mask_tracking.wan_sdedit import MODEL_ID, WanTI2VSDEdit, diffusers_version

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
    prompt = build_silhouette_prompt(args.object_text)
    pipeline = WanTI2VSDEdit()
    edited = pipeline.generate(
        source,
        prompt,
        args.strength,
        args.seed,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        max_sequence_length=args.max_sequence_length,
    )
    masks = extract_silhouette_mask(
        source,
        edited,
        white_threshold=args.white_threshold,
        difference_threshold=args.difference_threshold,
        morphology_kernel=args.morphology_kernel,
    )
    overlay = make_overlay(source, masks)
    comparison = make_comparison(source, edited, masks, overlay)
    fps = args.fps or source_fps
    write_video(output_dir / "source.mp4", source, fps)
    write_video(output_dir / "edited.mp4", edited, fps)
    write_video(output_dir / "raw_mask.mp4", mask_to_rgb(masks), fps)
    write_video(output_dir / "overlay.mp4", overlay, fps)
    write_video(output_dir / "side_by_side.mp4", comparison, fps)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_warning": (
            "No-GT discovery probe. Temporal metrics do not measure segmentation accuracy."
        ),
        "object": args.object_text,
        "prompt": prompt,
        "video": str(Path(args.video).resolve()),
        "model_id": MODEL_ID,
        "parameters": vars(args),
        "preprocessing": preprocessing,
        "metrics": temporal_metrics(masks),
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
