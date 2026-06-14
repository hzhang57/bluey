from __future__ import annotations

import cv2
import numpy as np


TARGET_COLORS = {
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "lime": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


def rec709_luma(video: np.ndarray) -> np.ndarray:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("video must have shape [F,H,W,3]")
    values = video.astype(np.float32)
    return (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    )


def to_grayscale_rec709(video: np.ndarray) -> np.ndarray:
    luma = np.clip(np.rint(rec709_luma(video)), 0, 255).astype(np.uint8)
    return np.repeat(luma[..., None], 3, axis=-1)


def build_global_color_prompt(color: str) -> str:
    if color not in TARGET_COLORS:
        raise ValueError(f"Unsupported target color: {color}")
    return (
        f"Colorize the entire grayscale video using a vivid {color} color palette.\n"
        f"Apply {color} color throughout the whole visible scene in every frame.\n"
        "Preserve the original scene layout, motion, camera movement, timing, "
        "and brightness structure.\n"
        "Do not leave the visible scene grayscale."
    )


def target_color_analysis(
    grayscale_input: np.ndarray,
    generated: np.ndarray,
    color: str,
    saturation_threshold: float = 0.20,
    hue_tolerance_degrees: float = 30.0,
    minimum_luma: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    _validate_video_pair(grayscale_input, generated)
    if color not in TARGET_COLORS:
        raise ValueError(f"Unsupported target color: {color}")
    if not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be in [0, 1]")
    if not 0.0 <= hue_tolerance_degrees <= 180.0:
        raise ValueError("hue_tolerance_degrees must be in [0, 180]")
    if not 0.0 <= minimum_luma <= 1.0:
        raise ValueError("minimum_luma must be in [0, 1]")

    generated_hsv = _rgb_video_to_hsv(generated)
    grayscale_hsv = _rgb_video_to_hsv(grayscale_input)
    target_rgb = np.array(TARGET_COLORS[color], dtype=np.uint8).reshape(1, 1, 3)
    target_hsv = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2HSV).reshape(3)
    target_hue_degrees = float(target_hsv[0]) * 2.0
    generated_hue_degrees = generated_hsv[..., 0] * 2.0
    raw_distance = np.abs(generated_hue_degrees - target_hue_degrees)
    hue_distance = np.minimum(raw_distance, 360.0 - raw_distance)
    saturation = generated_hsv[..., 1] / 255.0
    grayscale_saturation = grayscale_hsv[..., 1] / 255.0
    eligible = (rec709_luma(grayscale_input) / 255.0) >= minimum_luma
    hue_similarity = 1.0 - np.clip(hue_distance / 180.0, 0.0, 1.0)
    score = (hue_similarity * saturation * eligible).astype(np.float32)
    selected = (
        eligible
        & (saturation >= saturation_threshold)
        & (hue_distance <= hue_tolerance_degrees)
    )
    mask = selected.astype(np.uint8) * 255

    eligible_per_frame = eligible.reshape(len(eligible), -1).sum(axis=1)
    selected_per_frame = selected.reshape(len(selected), -1).sum(axis=1)
    coverage_per_frame = np.divide(
        selected_per_frame,
        eligible_per_frame,
        out=np.zeros_like(selected_per_frame, dtype=np.float64),
        where=eligible_per_frame > 0,
    )
    generated_luma = rec709_luma(generated) / 255.0
    grayscale_luma = rec709_luma(grayscale_input) / 255.0
    generated_f = generated.astype(np.float32) / 255.0
    grayscale_f = grayscale_input.astype(np.float32) / 255.0
    eligible_count = int(eligible.sum())

    def eligible_mean(values: np.ndarray) -> float:
        return float(values[eligible].mean()) if eligible_count else 0.0

    metrics = {
        "target_color": color,
        "target_rgb": list(TARGET_COLORS[color]),
        "target_hsv": {
            "hue_degrees": target_hue_degrees,
            "saturation": float(target_hsv[1]) / 255.0,
            "value": float(target_hsv[2]) / 255.0,
        },
        "target_hue_degrees": target_hue_degrees,
        "saturation_threshold": saturation_threshold,
        "hue_tolerance_degrees": hue_tolerance_degrees,
        "minimum_luma": minimum_luma,
        "eligible_pixel_fraction": float(eligible.mean()),
        "mean_generated_saturation": eligible_mean(saturation),
        "mean_input_saturation": eligible_mean(grayscale_saturation),
        "mean_saturation_gain": eligible_mean(saturation - grayscale_saturation),
        "mean_target_color_score": eligible_mean(score),
        "target_color_coverage": (
            float(selected.sum() / eligible_count) if eligible_count else 0.0
        ),
        "target_color_coverage_std": float(coverage_per_frame.std()),
        "target_color_coverage_per_frame": [
            float(value) for value in coverage_per_frame
        ],
        "luma_mae": eligible_mean(np.abs(generated_luma - grayscale_luma)),
        "rgb_mae": eligible_mean(np.mean(np.abs(generated_f - grayscale_f), axis=-1)),
    }
    return mask, score, metrics


def _rgb_video_to_hsv(video: np.ndarray) -> np.ndarray:
    return np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) for frame in video]
    ).astype(np.float32)


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
    return masks, score


def composite_white_target(source: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if source.ndim != 4 or source.shape[-1] != 3 or masks.shape != source.shape[:3]:
        raise ValueError("source must be [F,H,W,3] and masks must be [F,H,W]")
    result = source.copy()
    result[masks > 0] = 255
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


def temporal_metrics(masks: np.ndarray, skip_first_frame: bool = False) -> dict[str, float | list[float]]:
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
