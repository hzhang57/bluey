"""Utilities for probing emergent mask tracking in video editing models."""

from .analysis import (
    composite_white_target,
    extract_silhouette_mask,
    relative_whitening_score,
    temporal_metrics,
)
from .prompting import build_silhouette_prompt

__all__ = [
    "build_silhouette_prompt",
    "composite_white_target",
    "extract_silhouette_mask",
    "relative_whitening_score",
    "temporal_metrics",
]
