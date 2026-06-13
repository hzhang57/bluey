from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Any

import numpy as np

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REQUIRED_PIPELINE_PACKAGES = ("ftfy",)


def check_pipeline_dependencies() -> None:
    missing = [
        package
        for package in REQUIRED_PIPELINE_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        packages = " ".join(missing)
        raise ImportError(
            f"Missing Wan pipeline dependencies: {', '.join(missing)}. "
            "Install them before loading the model with:\n"
            f"  pip install {packages}"
        )


class WanTI2VSDEdit:
    """Thin Diffusers wrapper for source-video silhouette editing."""

    def __init__(self, pipeline: Any | None = None):
        import torch

        if pipeline is None:
            check_pipeline_dependencies()
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Wan2.2 video-to-video inference requires CUDA, but this Python "
                    f"environment has torch {torch.__version__} and no visible GPU. "
                    "On Kaggle, open Notebook options, set Accelerator to GPU, then "
                    "restart the session before running this command."
                )
            try:
                from diffusers import WanVideoToVideoPipeline
            except ImportError as error:
                raise ImportError(
                    "WanVideoToVideoPipeline is unavailable. Install the project "
                    "requirements with: pip install -r requirements.txt"
                ) from error
            pipeline = WanVideoToVideoPipeline.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16
            )
            pipeline.enable_model_cpu_offload()

        self.torch = torch
        self.pipeline = pipeline

    def generate(
        self,
        source_video: np.ndarray,
        prompt: str,
        strength: float,
        seed: int,
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
    ) -> np.ndarray:
        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        if source_video.ndim != 4 or source_video.shape[-1] != 3:
            raise ValueError("source_video must have shape [F, H, W, 3]")

        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        output = self.pipeline(
            prompt=prompt,
            video=list(source_video),
            strength=strength,
            num_inference_steps=sampling_steps,
            guidance_scale=guide_scale,
            generator=generator,
            output_type="np",
        )
        return _normalize_output(output.frames)


def diffusers_version() -> str:
    try:
        return importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _normalize_output(frames: Any) -> np.ndarray:
    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]
    result = np.asarray(frames)
    if result.ndim == 5 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 4 or result.shape[-1] != 3:
        raise ValueError("Diffusers returned frames with an unexpected shape")
    if np.issubdtype(result.dtype, np.floating):
        result = result * 255.0 if result.max(initial=0) <= 1.0 else result
    return np.clip(result, 0, 255).astype(np.uint8)
