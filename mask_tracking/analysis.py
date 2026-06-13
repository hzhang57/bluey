from __future__ import annotations

import cv2
import numpy as np


def relative_whitening_score(source: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """Score pixels that became brighter, less colorful, and different."""
    _validate_video_pair(source, generated)
    source_f = source.astype(np.float32) / 255.0
    generated_f = generated.astype(np.float32) / 255.0
    source_luma = source_f.mean(axis=-1)
    generated_luma = generated_f.mean(axis=-1)
    brightness_gain = np.clip(generated_luma - source_luma, 0.0, 1.0)
    low_chroma = 1.0 - (generated_f.max(axis=-1) - generated_f.min(axis=-1))
    difference = np.linalg.norm(generated_f - source_f, axis=-1) / np.sqrt(3.0)
    score = generated_luma * low_chroma * np.sqrt(brightness_gain * difference)
    score[0] = 0.0
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def extract_silhouette_mask(
    source: np.ndarray,
    generated: np.ndarray,
    score_threshold: float = 0.20,
    morphology_kernel: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    score = relative_whitening_score(source, generated)
    masks = (score >= score_threshold).astype(np.uint8) * 255
    if morphology_kernel > 1:
        kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
        masks = np.stack(
            [
                cv2.morphologyEx(
                    cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
                    cv2.MORPH_CLOSE,
                    kernel,
                )
                for mask in masks
            ]
        )
    masks[0] = 0
    return masks, score


def composite_white_target(source: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if source.ndim != 4 or source.shape[-1] != 3 or masks.shape != source.shape[:3]:
        raise ValueError("source must be [F,H,W,3] and masks must be [F,H,W]")
    result = source.copy()
    result[masks > 0] = 255
    result[0] = source[0]
    return result


def video_statistics(video: np.ndarray) -> dict[str, float | list[int]]:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("video must have shape [F,H,W,3]")
    values = video.astype(np.float32)
    return {
        "shape": list(video.shape),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def validate_decoded_video(
    video: np.ndarray,
    stage: str,
    expected_shape: tuple[int, ...] | None = None,
) -> dict[str, float | list[int]]:
    if expected_shape is not None and video.shape != expected_shape:
        raise RuntimeError(
            f"{stage} decoded to shape {video.shape}; expected {expected_shape}"
        )
    stats = video_statistics(video)
    if stats["max"] <= 8.0 or (stats["mean"] <= 5.0 and stats["std"] <= 5.0):
        raise RuntimeError(
            f"{stage} decoded to a near-black video: "
            f"min={stats['min']:.1f}, max={stats['max']:.1f}, "
            f"mean={stats['mean']:.1f}, std={stats['std']:.1f}"
        )
    return stats


def temporal_metrics(masks: np.ndarray, skip_first_frame: bool = True) -> dict[str, float | list[float]]:
    """Compute no-GT diagnostics. These measure stability, not segmentation accuracy."""
    if masks.ndim != 3:
        raise ValueError("masks must have shape [F, H, W]")
    binary = masks > 0
    evaluated = binary[1:] if skip_first_frame and len(binary) > 1 else binary
    areas = evaluated.reshape(evaluated.shape[0], -1).mean(axis=1)
    ious: list[float] = []
    flicker: list[float] = []
    for previous, current in zip(evaluated[:-1], evaluated[1:]):
        union = np.logical_or(previous, current).sum()
        intersection = np.logical_and(previous, current).sum()
        ious.append(float(intersection / union) if union else 1.0)
        flicker.append(float(np.logical_xor(previous, current).mean()))
    return {
        "skipped_first_frame": skip_first_frame,
        "mean_foreground_fraction": float(areas.mean()),
        "std_foreground_fraction": float(areas.std()),
        "mean_consecutive_iou": float(np.mean(ious)) if ious else 1.0,
        "mean_flicker_rate": float(np.mean(flicker)) if flicker else 0.0,
        "foreground_fraction_per_frame": [float(value) for value in areas],
    }


def _validate_video_pair(source: np.ndarray, generated: np.ndarray) -> None:
    if source.shape != generated.shape or source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError("source and generated must have identical [F,H,W,3] shapes")
