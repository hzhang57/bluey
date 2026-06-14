from __future__ import annotations

import copy
import gc
import importlib
import importlib.metadata
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_WAN_REPO = Path("/kaggle/working/Wan2.2")
WAN_REPO_MARKER = Path("wan/textimage2video.py")


@dataclass
class OfficialSDEditResult:
    generated_raw: np.ndarray
    vae_roundtrip: np.ndarray
    diagnostics: dict[str, Any]


SnapshotCallback = Callable[[str, np.ndarray, dict[str, Any]], None]


def resolve_official_wan_repo(
    wan_repo: str | Path | None,
    search_roots: tuple[Path, ...] | None = None,
) -> Path:
    if wan_repo is not None:
        explicit = Path(wan_repo).expanduser().resolve()
        if (explicit / WAN_REPO_MARKER).is_file():
            return explicit
        raise FileNotFoundError(
            f"{explicit} is not a Wan2.2 GitHub checkout; missing "
            f"{WAN_REPO_MARKER}.\n"
            "Clone it in Kaggle with:\n"
            "!git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git "
            "/kaggle/working/Wan2.2\n"
            "Or pass the existing checkout with --wan-repo /path/to/Wan2.2"
        )

    roots = search_roots or (Path.cwd(), Path("/kaggle/working"))
    candidates = []
    environment_repo = os.environ.get("WAN_REPO")
    if environment_repo:
        candidates.append(Path(environment_repo))
    candidates.extend(
        [
            DEFAULT_WAN_REPO,
            Path.cwd() / "Wan2.2",
            Path.cwd().parent / "Wan2.2",
        ]
    )
    for root in roots:
        if root.is_dir():
            candidates.extend(
                marker.parent.parent
                for marker in root.glob("*/wan/textimage2video.py")
            )
            candidates.extend(
                marker.parent.parent
                for marker in root.glob("*/*/wan/textimage2video.py")
            )

    checked = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if (resolved / WAN_REPO_MARKER).is_file():
            print(
                f"[official preflight] Auto-detected Wan2.2 repo: {resolved}",
                flush=True,
            )
            return resolved

    checked_text = "\n".join(f"- {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find an official Wan2.2 GitHub checkout. Checked:\n"
        f"{checked_text}\n"
        "Clone it in Kaggle with:\n"
        "!git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git "
        "/kaggle/working/Wan2.2\n"
        "Then rerun, or pass --wan-repo /path/to/Wan2.2."
    )


def require_official_flash_attention(attention_module: Any) -> None:
    if attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE:
        return
    raise ImportError(
        "The official Wan2.2 WanModel requires flash-attn. Install it with "
        "`pip install flash-attn --no-build-isolation --no-deps`."
    )


def load_official_wan_components(repo: Path) -> tuple[Any, Any, Any, Any]:
    package_dir = (repo / "wan").resolve()
    loaded_package = sys.modules.get("wan")
    if loaded_package is None:
        # Upstream wan/__init__.py eagerly imports unrelated Animate and S2V
        # tasks. Registering the official package path loads only TI2V modules.
        package = types.ModuleType("wan")
        package.__package__ = "wan"
        package.__path__ = [str(package_dir)]
        sys.modules["wan"] = package
    else:
        loaded_paths = {
            Path(path).resolve() for path in getattr(loaded_package, "__path__", [])
        }
        if package_dir not in loaded_paths:
            raise RuntimeError(
                "A different `wan` package is already imported; restart Python "
                f"before loading the official checkout at {repo}"
            )

    textimage2video = importlib.import_module("wan.textimage2video")
    configs = importlib.import_module("wan.configs")
    official_attention = importlib.import_module("wan.modules.attention")
    scheduler_module = importlib.import_module("wan.utils.fm_solvers_unipc")
    return (
        textimage2video.WanTI2V,
        configs.WAN_CONFIGS,
        official_attention,
        scheduler_module.FlowUniPCMultistepScheduler,
    )


def strength_to_start_index(strength: float, sampling_steps: int) -> tuple[int, int]:
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be in (0, 1]")
    if sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")
    denoise_steps = max(1, min(round(strength * sampling_steps), sampling_steps))
    return sampling_steps - denoise_steps, denoise_steps


def pad_timestep_for_official_model(
    timestep: Any, latent: Any, patch_size: tuple[int, ...], seq_len: int, torch_module: Any
) -> Any:
    token_count = (
        int(latent.shape[1])
        * math.ceil(int(latent.shape[2]) / int(patch_size[1]))
        * math.ceil(int(latent.shape[3]) / int(patch_size[2]))
    )
    if token_count > seq_len:
        raise ValueError(f"latent token count {token_count} exceeds seq_len {seq_len}")
    # Official Wan builds this from masks_like(noise), so expanded timesteps
    # inherit the FP32 latent/noise dtype rather than the scheduler's int dtype.
    values = torch_module.ones(
        token_count, device=latent.device, dtype=latent.dtype
    ) * timestep
    if token_count < seq_len:
        values = torch_module.cat(
            [values, values.new_ones(seq_len - token_count) * timestep]
        )
    return values.unsqueeze(0)


def _ensure_finite(tensor: Any, stage: str, torch_module: Any) -> None:
    if bool(torch_module.isfinite(tensor).all().item()):
        return
    raise RuntimeError(f"{stage} contains NaN or infinity")


def _tensor_stats(tensor: Any) -> dict[str, float]:
    values = tensor.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "norm": float(values.norm().item()),
    }


def _context_difference(context: Any, context_null: Any) -> dict[str, float]:
    left = context[0].float()
    right = context_null[0].float()
    shared_length = min(left.shape[0], right.shape[0])
    difference = left[:shared_length] - right[:shared_length]
    denominator = max(float(left.norm().item()), float(right.norm().item()), 1e-12)
    return {
        "norm": float(difference.norm().item()),
        "relative_norm": float(difference.norm().item()) / denominator,
        "positive_norm": float(left.norm().item()),
        "negative_norm": float(right.norm().item()),
    }


class WanOfficialFullVideoSDEdit:
    """Full-video SDEdit built from the official Wan2.2 GitHub implementation."""

    def __init__(
        self,
        wan_repo: str | Path | None,
        checkpoint_dir: str | Path,
        max_sequence_length: int = 512,
        device_id: int = 0,
    ):
        import torch

        repo = resolve_official_wan_repo(wan_repo)
        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"Official Wan checkpoint directory does not exist: {checkpoint}"
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        WanTI2V, WAN_CONFIGS, official_attention, scheduler_class = (
            load_official_wan_components(repo)
        )
        require_official_flash_attention(official_attention)
        config = copy.deepcopy(WAN_CONFIGS["ti2v-5B"])
        official_text_length = int(config.text_len)
        if not 1 <= max_sequence_length <= official_text_length:
            raise ValueError(
                "max_sequence_length must be between 1 and the official Wan "
                f"limit of {official_text_length}"
            )
        config.text_len = max_sequence_length
        missing_checkpoint_files = [
            name
            for name in (config.t5_checkpoint, config.vae_checkpoint)
            if not (checkpoint / name).is_file()
        ]
        if missing_checkpoint_files:
            raise FileNotFoundError(
                "Checkpoint directory is not the official non-Diffusers "
                f"{MODEL_ID} layout; missing: {', '.join(missing_checkpoint_files)}"
            )
        self.torch = torch
        self.scheduler_class = scheduler_class
        self.repo = repo
        self.checkpoint_dir = checkpoint
        self.device = torch.device(f"cuda:{device_id}")
        self.vae_device = torch.device(
            "cuda:1" if torch.cuda.device_count() > 1 and device_id == 0 else self.device
        )
        self.max_sequence_length = max_sequence_length
        self.pipe = WanTI2V(
            config=config,
            checkpoint_dir=str(checkpoint),
            device_id=device_id,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=True,
        )
        self.pipe.vae.model.to(self.vae_device)
        self.pipe.vae.device = self.vae_device
        self.pipe.vae.scale = [value.to(self.vae_device) for value in self.pipe.vae.scale]
        print(
            f"[official load] repo={repo} checkpoint={checkpoint} "
            f"dit={self.device} vae={self.vae_device} t5=cpu",
            flush=True,
        )

    def _activate_model(self) -> None:
        if self.vae_device == self.device:
            self.pipe.vae.model.cpu()
        self.pipe.model.to(self.device)
        self.torch.cuda.empty_cache()

    def _activate_vae(self) -> None:
        if self.vae_device == self.device:
            self.pipe.model.cpu()
            self.pipe.vae.model.to(self.vae_device)
            self.torch.cuda.empty_cache()

    def _decode(self, latent: Any) -> np.ndarray:
        self._activate_vae()
        decoded = self.pipe.vae.decode([latent.to(self.vae_device)])[0]
        result = (
            decoded.permute(1, 2, 3, 0)
            .add(1.0)
            .mul(127.5)
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )
        return result

    def generate(
        self,
        source_video: np.ndarray,
        prompt: str,
        strength: float,
        seed: int,
        sampling_steps: int = 100,
        guide_scale: float = 5.0,
        max_sequence_length: int = 512,
        snapshot_callback: SnapshotCallback | None = None,
        snapshot_every: int = 10,
        negative_prompt: str = "",
    ) -> OfficialSDEditResult:
        torch, pipe = self.torch, self.pipe
        if max_sequence_length != self.max_sequence_length:
            raise ValueError(
                "Official Wan text length is fixed when the pipeline loads; "
                f"expected {self.max_sequence_length}, got {max_sequence_length}"
            )
        if source_video.ndim != 4 or source_video.shape[-1] != 3:
            raise ValueError("source_video must have shape [F,H,W,3]")
        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        if guide_scale <= 0.0:
            raise ValueError("guide_scale must be positive")
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be positive")

        print("[official stage 1/5] Encoding positive and negative prompts with Wan T5...", flush=True)
        context = pipe.text_encoder([prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([negative_prompt], torch.device("cpu"))
        context_difference = _context_difference(context, context_null)
        if context_difference["relative_norm"] < 1e-6:
            raise RuntimeError("Official Wan positive and negative prompt contexts are identical")
        context = [value.to(self.device) for value in context]
        context_null = [value.to(self.device) for value in context_null]
        print(
            "[official text] "
            f"context_delta_relative_norm={context_difference['relative_norm']:.6f}",
            flush=True,
        )

        print("[official stage 2/5] Encoding full source video with Wan2.2 VAE...", flush=True)
        source_tensor = (
            torch.from_numpy(source_video)
            .permute(3, 0, 1, 2)
            .to(self.vae_device, dtype=torch.float32)
            .div(127.5)
            .sub(1.0)
        )
        clean = pipe.vae.encode([source_tensor])[0]
        _ensure_finite(clean, "official VAE source latent", torch)
        vae_roundtrip = self._decode(clean)
        latent_shape = list(clean.shape)

        scheduler = self.scheduler_class(
            num_train_timesteps=pipe.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=self.device, shift=5.0)
        start_index, requested_denoise_steps = strength_to_start_index(
            strength, sampling_steps
        )
        scheduler.set_begin_index(start_index)
        timesteps = scheduler.timesteps[start_index:]
        clean = clean.to(self.device, dtype=torch.float32)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(
            clean.shape, device=self.device, dtype=torch.float32, generator=generator
        )
        latents = scheduler.add_noise(
            clean.unsqueeze(0), noise.unsqueeze(0), timesteps[:1]
        ).squeeze(0)
        _ensure_finite(latents, "official scheduler noisy latent", torch)
        initial_sigma = float(scheduler.sigmas[start_index].item())
        if snapshot_callback is not None:
            snapshot_callback(
                "noisy",
                self._decode(latents),
                {"kind": "scheduler_add_noise", "step": 0, "timestep": float(timesteps[0].item())},
            )

        seq_len = math.ceil(
            (clean.shape[2] * clean.shape[3])
            / (pipe.patch_size[1] * pipe.patch_size[2])
            * clean.shape[1]
            / pipe.sp_size
        ) * pipe.sp_size
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        self._activate_model()
        step_diagnostics = []
        last_snapshot = None
        print(
            f"[official stage 3/5] Running {len(timesteps)} official FlowUniPC denoise steps...",
            flush=True,
        )
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad():
            for index, timestep in enumerate(timesteps):
                timestep_batch = pad_timestep_for_official_model(
                    timestep, latents, pipe.patch_size, seq_len, torch
                )
                conditional = pipe.model([latents], t=timestep_batch, **arg_c)[0]
                unconditional = pipe.model([latents], t=timestep_batch, **arg_null)[0]
                _ensure_finite(conditional, "official conditional prediction", torch)
                _ensure_finite(unconditional, "official unconditional prediction", torch)
                guided = unconditional.float() + guide_scale * (
                    conditional.float() - unconditional.float()
                )
                latents = scheduler.step(
                    guided.unsqueeze(0),
                    timestep,
                    latents.unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0].squeeze(0).float()
                _ensure_finite(latents, f"official scheduler step {index + 1}", torch)
                step_number = index + 1
                delta = conditional.float() - unconditional.float()
                step_diagnostics.append(
                    {
                        "step": step_number,
                        "timestep": float(timestep.item()),
                        "conditional": _tensor_stats(conditional),
                        "unconditional": _tensor_stats(unconditional),
                        "guided": _tensor_stats(guided),
                        "conditional_minus_unconditional": _tensor_stats(delta),
                    }
                )
                should_snapshot = snapshot_callback is not None and (
                    step_number % snapshot_every == 0 or step_number == len(timesteps)
                )
                if should_snapshot:
                    last_snapshot = self._decode(latents)
                    snapshot_callback(
                        f"denoise_step_{step_number:03d}",
                        last_snapshot,
                        {
                            "kind": "denoise_step",
                            "step": step_number,
                            "timestep": float(timestep.item()),
                        },
                    )
                    self._activate_model()
                if index == 0 or step_number % 5 == 0 or step_number == len(timesteps):
                    print(
                        f"[official stage 4/5] Denoise {step_number}/{len(timesteps)} "
                        f"cond_std={step_diagnostics[-1]['conditional']['std']:.4f} "
                        f"guided_std={step_diagnostics[-1]['guided']['std']:.4f}",
                        flush=True,
                    )

        print("[official stage 5/5] Decoding final video...", flush=True)
        generated = last_snapshot if last_snapshot is not None else self._decode(latents)
        pipe.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        return OfficialSDEditResult(
            generated_raw=generated,
            vae_roundtrip=vae_roundtrip,
            diagnostics={
                "backend": "Wan-Video/Wan2.2 official GitHub",
                "official_repo": str(self.repo),
                "official_checkpoint": str(self.checkpoint_dir),
                "conditioning_mode": "official_wan_full_video_latent_sdedit_text_only",
                "clean_latent_shape": latent_shape,
                "strength": strength,
                "guidance_scale": guide_scale,
                "positive_prompt": prompt,
                "negative_prompt": negative_prompt,
                "cfg_enabled": True,
                "transformer_forwards_per_step": 2,
                "total_scheduler_steps": sampling_steps,
                "requested_denoise_steps": requested_denoise_steps,
                "denoise_start_index": start_index,
                "actual_denoise_steps": len(timesteps),
                "initial_sigma": initial_sigma,
                "text_context_difference": context_difference,
                "steps": step_diagnostics,
            },
        )


def official_environment() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "diffusers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
