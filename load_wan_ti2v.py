#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_WAN_REPO = Path("/kaggle/working/Wan2.2")
CHECKPOINT_MARKERS = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
    "config.json",
    "diffusion_pytorch_model.safetensors.index.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load Wan2.2-TI2V-5B using the official Wan2.2 GitHub code."
    )
    parser.add_argument(
        "--wan-repo",
        default=str(DEFAULT_WAN_REPO),
        help="Path to the official Wan-Video/Wan2.2 GitHub checkout.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Official non-Diffusers checkpoint directory. When omitted, reuse or "
            "download the model through the Hugging Face cache."
        ),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--text-length",
        type=int,
        default=512,
        help="T5 token limit. Official default is 512.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading a missing checkpoint.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.device_id < 0:
        raise ValueError("--device-id must be non-negative")
    if not 1 <= args.text_length <= 512:
        raise ValueError("--text-length must be in [1, 512]")


def resolve_wan_repo(path: str | Path) -> Path:
    repo = Path(path).expanduser().resolve()
    marker = repo / "wan" / "textimage2video.py"
    if marker.is_file():
        return repo
    raise FileNotFoundError(
        f"Official Wan2.2 source not found at {repo}. Missing {marker}.\n"
        "Clone it with:\n"
        "git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git "
        "/kaggle/working/Wan2.2"
    )


def is_official_checkpoint(path: Path) -> bool:
    return all((path / marker).is_file() for marker in CHECKPOINT_MARKERS)


def find_cached_checkpoint(cache_dir: Path | None = None) -> Path | None:
    if cache_dir is None:
        cache_dir = Path(
            os.environ.get(
                "HF_HUB_CACHE",
                Path.home() / ".cache" / "huggingface" / "hub",
            )
        )
    snapshots = cache_dir / "models--Wan-AI--Wan2.2-TI2V-5B" / "snapshots"
    if not snapshots.is_dir():
        return None
    for candidate in sorted(snapshots.iterdir(), reverse=True):
        if candidate.is_dir() and is_official_checkpoint(candidate):
            return candidate.resolve()
    return None


def resolve_checkpoint(
    checkpoint_dir: str | Path | None,
    allow_download: bool,
    snapshot_download_fn: Any | None = None,
) -> Path:
    if checkpoint_dir is not None:
        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        if is_official_checkpoint(checkpoint):
            return checkpoint
        raise FileNotFoundError(
            f"{checkpoint} is not a complete official {MODEL_ID} checkpoint."
        )

    cached = find_cached_checkpoint()
    if cached is not None:
        print(f"[checkpoint] Reusing Hugging Face cache: {cached}", flush=True)
        return cached

    if not allow_download:
        raise FileNotFoundError(
            f"{MODEL_ID} is not cached. Remove --no-download to download it."
        )
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download

    print(
        f"[checkpoint] Downloading {MODEL_ID} (about 31.85 GiB) to the "
        "Hugging Face cache...",
        flush=True,
    )
    checkpoint = Path(
        snapshot_download_fn(repo_id=MODEL_ID, token=os.environ.get("HF_TOKEN"))
    ).resolve()
    if not is_official_checkpoint(checkpoint):
        raise FileNotFoundError(f"Downloaded checkpoint is incomplete: {checkpoint}")
    return checkpoint


def import_official_ti2v(repo: Path) -> tuple[Any, Any]:
    package_dir = repo / "wan"
    loaded = sys.modules.get("wan")
    if loaded is None:
        # The official wan/__init__.py imports unrelated Animate/S2V modules.
        # Register the official package path and load only the TI2V modules.
        package = types.ModuleType("wan")
        package.__package__ = "wan"
        package.__path__ = [str(package_dir)]
        sys.modules["wan"] = package
    elif repo.resolve() not in {
        Path(path).resolve().parent for path in getattr(loaded, "__path__", [])
    }:
        raise RuntimeError("A different `wan` package is already imported.")

    configs = importlib.import_module("wan.configs")
    ti2v_module = importlib.import_module("wan.textimage2video")
    return configs.WAN_CONFIGS, ti2v_module.WanTI2V


def first_parameter_summary(module: Any) -> dict[str, Any]:
    parameter = next(module.parameters())
    return {
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "parameter_count": sum(value.numel() for value in module.parameters()),
    }


def load_pipeline(args: argparse.Namespace) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Wan2.2-TI2V-5B requires a CUDA GPU.")
    if args.device_id >= torch.cuda.device_count():
        raise ValueError(
            f"--device-id {args.device_id} is unavailable; "
            f"CUDA device count is {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(args.device_id)

    repo = resolve_wan_repo(args.wan_repo)
    checkpoint = resolve_checkpoint(
        args.checkpoint_dir,
        allow_download=not args.no_download,
    )
    configs, WanTI2V = import_official_ti2v(repo)
    config = copy.deepcopy(configs["ti2v-5B"])
    config.text_len = args.text_length

    print(
        "[load] Initializing official WanTI2V with T5 on CPU, DiT initialized "
        "on CPU, and VAE on CUDA...",
        flush=True,
    )
    print(
        f"[load] cuda:{args.device_id}={torch.cuda.get_device_name(args.device_id)}",
        flush=True,
    )
    pipeline = WanTI2V(
        config=config,
        checkpoint_dir=str(checkpoint),
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=True,
        convert_model_dtype=True,
    )

    print("[load] Wan2.2-TI2V-5B loaded successfully.", flush=True)
    print(f"[load] checkpoint={checkpoint}", flush=True)
    print(f"[load] DiT={first_parameter_summary(pipeline.model)}", flush=True)
    print(f"[load] VAE={first_parameter_summary(pipeline.vae.model)}", flush=True)
    print("[load] T5=cpu", flush=True)
    return pipeline


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    load_pipeline(args)


if __name__ == "__main__":
    main()
