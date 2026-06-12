from __future__ import annotations

import cv2
import numpy as np


def extract_silhouette_mask(
    source: np.ndarray,
    edited: np.ndarray,
    white_threshold: int = 220,
    difference_threshold: float = 35.0,
    morphology_kernel: int = 3,
) -> np.ndarray:
    """Extract edited-to-white pixels without using a segmentation or tracking model."""
    if source.shape != edited.shape or source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError("source and edited must have identical [F, H, W, 3] shapes")

    source_f = source.astype(np.float32)
    edited_f = edited.astype(np.float32)
    whiteness = edited.min(axis=-1) >= white_threshold
    difference = np.linalg.norm(edited_f - source_f, axis=-1) >= difference_threshold
    masks = (whiteness & difference).astype(np.uint8) * 255

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
    return masks


def temporal_metrics(masks: np.ndarray) -> dict[str, float | list[float]]:
    """Compute no-GT diagnostics. These measure stability, not segmentation accuracy."""
    if masks.ndim != 3:
        raise ValueError("masks must have shape [F, H, W]")
    binary = masks > 0
    areas = binary.reshape(binary.shape[0], -1).mean(axis=1)
    ious: list[float] = []
    flicker: list[float] = []
    for previous, current in zip(binary[:-1], binary[1:]):
        union = np.logical_or(previous, current).sum()
        intersection = np.logical_and(previous, current).sum()
        ious.append(float(intersection / union) if union else 1.0)
        flicker.append(float(np.logical_xor(previous, current).mean()))
    return {
        "mean_foreground_fraction": float(areas.mean()),
        "std_foreground_fraction": float(areas.std()),
        "mean_consecutive_iou": float(np.mean(ious)) if ious else 1.0,
        "mean_flicker_rate": float(np.mean(flicker)) if flicker else 0.0,
        "foreground_fraction_per_frame": [float(value) for value in areas],
    }
