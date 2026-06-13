from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def read_video_clip(
    path: str | Path,
    width: int,
    height: int,
    frame_num: int,
    start_frame: int = 0,
    target_fps: float = 24.0,
) -> tuple[np.ndarray, float, dict]:
    if frame_num < 1 or (frame_num - 1) % 4:
        raise ValueError("--frame-num must have the form 4n+1")
    reader = imageio.get_reader(str(path))
    metadata = reader.get_meta_data()
    source_fps = float(metadata.get("fps", 24.0))
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source and target FPS must be positive")
    requested_indices = [
        start_frame + round(index * source_fps / target_fps)
        for index in range(frame_num)
    ]
    frames = []
    requested_position = 0
    for index, frame in enumerate(reader):
        if requested_position >= len(requested_indices):
            break
        if index < requested_indices[requested_position]:
            continue
        frames.append(_letterbox_rgb(frame, width, height))
        requested_position += 1
        while (
            requested_position < len(requested_indices)
            and requested_indices[requested_position] == index
        ):
            frames.append(frames[-1].copy())
            requested_position += 1
    reader.close()
    if not frames:
        raise ValueError(f"No frames could be read from {path}")

    original_count = len(frames)
    while len(frames) < frame_num:
        frames.append(frames[-1].copy())
    info = {
        "source_fps": source_fps,
        "target_fps": target_fps,
        "start_frame": start_frame,
        "real_frame_count": original_count,
        "padded_frame_count": frame_num - original_count,
        "output_width": width,
        "output_height": height,
    }
    return np.stack(frames), source_fps, info


def _letterbox_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame = np.asarray(frame)[..., :3]
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    output[y : y + resized_h, x : x + resized_w] = resized
    return output


def write_video(path: str | Path, frames: np.ndarray, fps: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def mask_to_rgb(masks: np.ndarray) -> np.ndarray:
    return np.repeat(masks[..., None], 3, axis=-1)


def score_to_rgb(score: np.ndarray) -> np.ndarray:
    if score.ndim != 3:
        raise ValueError("score must have shape [F,H,W]")
    frames = [
        cv2.cvtColor(
            cv2.applyColorMap(np.clip(frame * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO),
            cv2.COLOR_BGR2RGB,
        )
        for frame in score
    ]
    return np.stack(frames)


def make_comparison(
    source: np.ndarray, generated: np.ndarray, masks: np.ndarray, edited: np.ndarray
) -> np.ndarray:
    mask_rgb = mask_to_rgb(masks)
    return np.concatenate([source, generated, mask_rgb, edited], axis=2)
