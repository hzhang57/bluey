from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .analysis import validate_decoded_video

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REQUIRED_PIPELINE_PACKAGES = ("ftfy",)
OFFICIAL_SCHEDULER_CLASS = "UniPCMultistepScheduler"
OFFICIAL_FLOW_SHIFT = 5.0


@dataclass
class SDEditResult:
    generated_raw: np.ndarray
    vae_roundtrip: np.ndarray
    diagnostics: dict[str, Any]


SnapshotCallback = Callable[[str, np.ndarray, dict[str, Any]], None]


def check_pipeline_dependencies() -> None:
    missing = [name for name in REQUIRED_PIPELINE_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f"Missing Wan pipeline dependencies: {', '.join(missing)}. Install with: "
            f"pip install {' '.join(missing)}"
        )


def noise_strength_to_start_idx(strength: float, steps: int) -> int:
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be in (0, 1]")
    if steps < 1:
        raise ValueError("steps must be positive")
    return min(round((1.0 - strength) * steps), steps - 1)


def denoise_step_count(strength: float, steps: int) -> int:
    return steps - noise_strength_to_start_idx(strength, steps)


def should_save_denoise_snapshot(step: int, total_steps: int, every: int) -> bool:
    if every < 1:
        raise ValueError("snapshot interval must be positive")
    return step % every == 0 or step == total_steps


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def scheduler_info(scheduler: Any) -> dict[str, Any]:
    config = scheduler.config
    return {
        "scheduler_class": scheduler.__class__.__name__,
        "scheduler_config_class": _config_value(config, "_class_name"),
        "scheduler_flow_shift": _config_value(config, "flow_shift"),
        "scheduler_prediction_type": _config_value(config, "prediction_type"),
        "scheduler_use_flow_sigmas": _config_value(config, "use_flow_sigmas"),
        "scheduler_timestep_spacing": _config_value(config, "timestep_spacing"),
        "scheduler_solver_order": _config_value(config, "solver_order"),
        "scheduler_solver_type": _config_value(config, "solver_type"),
        "scheduler_num_train_timesteps": _config_value(config, "num_train_timesteps"),
    }


def validate_official_scheduler(scheduler: Any) -> dict[str, Any]:
    info = scheduler_info(scheduler)
    expected = {
        "scheduler_class": OFFICIAL_SCHEDULER_CLASS,
        "scheduler_flow_shift": OFFICIAL_FLOW_SHIFT,
        "scheduler_prediction_type": "flow_prediction",
        "scheduler_use_flow_sigmas": True,
        "scheduler_timestep_spacing": "linspace",
    }
    mismatches = {
        name: (info.get(name), expected_value)
        for name, expected_value in expected.items()
        if info.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(
            "Pipeline scheduler does not match official Wan2.2-TI2V-5B config: "
            f"{mismatches}"
        )
    return {**info, "scheduler_source": "checkpoint_config"}


def tensor_head_tail(tensor: Any, count: int = 5) -> tuple[list[float], list[float]]:
    values = [float(value) for value in tensor.detach().float().cpu().tolist()]
    return values[:count], values[-count:]


def add_noise_at_timestep(scheduler: Any, latents: Any, noise: Any, timestep: Any) -> Any:
    if timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)
    return scheduler.add_noise(latents, noise, timestep.to(device=latents.device))


def scheduler_noise_diagnostics(scheduler: Any, start_idx: int) -> dict[str, Any]:
    sigma = scheduler.sigmas[start_idx]
    alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma)
    config = scheduler.config
    return {
        "scheduler_class": scheduler.__class__.__name__,
        "start_timestep": float(scheduler.timesteps[start_idx].item()),
        "schedule_sigma": float(sigma.item()),
        "signal_weight": float(alpha_t.item()),
        "noise_weight": float(sigma_t.item()),
        "noise_to_signal_ratio": float((sigma_t / alpha_t).item()),
        "use_flow_sigmas": bool(getattr(config, "use_flow_sigmas", False)),
        "flow_shift": float(getattr(config, "flow_shift", 1.0)),
        "prediction_type": str(getattr(config, "prediction_type", "unknown")),
    }


def verify_scheduler_add_noise(
    clean: Any,
    noise: Any,
    noisy: Any,
    noise_diagnostics: dict[str, Any],
) -> dict[str, float]:
    clean_f = clean.float()
    noise_f = noise.float()
    noisy_f = noisy.float()
    expected = (
        noise_diagnostics["signal_weight"] * clean_f
        + noise_diagnostics["noise_weight"] * noise_f
    )
    error = (noisy_f - expected).abs()
    result = {
        "clean_latent_std": float(clean_f.std().item()),
        "sampled_noise_std": float(noise_f.std().item()),
        "noisy_latent_std": float(noisy_f.std().item()),
        "add_noise_formula_mean_abs_error": float(error.mean().item()),
        "add_noise_formula_max_abs_error": float(error.max().item()),
    }
    del clean_f, noise_f, noisy_f, expected, error
    return result


def validate_text_only_latent_contract(transformer: Any, latent: Any) -> dict[str, Any]:
    latent_channels = int(latent.shape[1])
    transformer_channels = int(transformer.config.in_channels)
    if latent_channels != transformer_channels:
        raise ValueError(
            "Text-only full-video SDEdit requires source latent channels to match "
            f"transformer input channels: latent={latent_channels}, "
            f"transformer={transformer_channels}"
        )
    return {
        "latent_channels": latent_channels,
        "transformer_in_channels": transformer_channels,
        "uses_first_frame_condition": False,
    }


def load_diffusers_pipeline(torch_module: Any, vae_class=None, pipeline_class=None) -> Any:
    if vae_class is None or pipeline_class is None:
        try:
            from diffusers import AutoencoderKLWan, WanPipeline
        except ImportError as error:
            raise ImportError("Install the project requirements before loading Wan2.2") from error
        vae_class, pipeline_class = AutoencoderKLWan, WanPipeline

    vae = vae_class.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch_module.float16)
    pipe = pipeline_class.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch_module.bfloat16)
    pipe.register_to_config(expand_timesteps=True)
    transformer_device = torch_module.device("cuda:0")
    vae_device = torch_module.device("cuda:1" if torch_module.cuda.device_count() > 1 else "cuda:0")
    text_device = torch_module.device("cpu")
    pipe.transformer.to(transformer_device)
    pipe.vae.to("cpu", dtype=torch_module.float16)
    pipe.text_encoder.to(text_device)
    pipe._vae_target_device = vae_device
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    print(
        f"[load] transformer={transformer_device} text_encoder={text_device} "
        f"vae=cpu (staged for {vae_device})",
        flush=True,
    )
    return pipe


def _retrieve_latents(encoder_output: Any) -> Any:
    if hasattr(encoder_output, "latent_dist"):
        distribution = encoder_output.latent_dist
        return distribution.mode() if hasattr(distribution, "mode") else distribution.mean
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not retrieve VAE latents")


def _vae_norm(pipe: Any, device: Any, torch_module: Any) -> tuple[Any, Any]:
    mean = torch_module.tensor(pipe.vae.config.latents_mean, dtype=torch_module.float32, device=device)
    std = torch_module.tensor(pipe.vae.config.latents_std, dtype=torch_module.float32, device=device)
    shape = (1, pipe.vae.config.z_dim, 1, 1, 1)
    return mean.view(shape), (1.0 / std).view(shape)


def _clear_memory(torch_module: Any) -> None:
    gc.collect()
    torch_module.cuda.empty_cache()


def _model_cache_context(model: Any, name: str) -> Any:
    return model.cache_context(name) if hasattr(model, "cache_context") else nullcontext()


def _text_only_timestep(pipe: Any, latents: Any, timestep: Any, torch_module: Any) -> Any:
    if not bool(_config_value(pipe.config, "expand_timesteps", False)):
        return timestep.expand(latents.shape[0])
    patch_size = pipe.transformer.config.patch_size
    mask = torch_module.ones(
        (1, 1, *latents.shape[2:]),
        device=latents.device,
        dtype=torch_module.float32,
    )
    token_mask = mask[0][0][
        :, :: int(patch_size[1]), :: int(patch_size[2])
    ]
    return (token_mask * timestep).flatten().unsqueeze(0).expand(latents.shape[0], -1)


def _clear_vae_cache(pipe: Any) -> None:
    vae = pipe.vae
    for name in ("clear_cache", "_clear_cache", "clear_context_parallel_cache"):
        function = getattr(vae, name, None)
        if callable(function):
            try:
                function()
                return
            except TypeError:
                pass
    for name in ("_feat_map", "_enc_feat_map", "_features", "_cache"):
        if hasattr(vae, name):
            setattr(vae, name, None)


def _encode_video(
    pipe: Any,
    source_video: np.ndarray,
    transformer_device: Any,
    torch_module: Any,
) -> Any:
    vae_device = next(pipe.vae.parameters()).device
    vae_dtype = next(pipe.vae.parameters()).dtype
    tensor = (
        torch_module.from_numpy(source_video)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device=vae_device, dtype=vae_dtype)
        .div(127.5)
        .sub(1.0)
    )
    _clear_vae_cache(pipe)
    try:
        with torch_module.no_grad():
            latent = _retrieve_latents(pipe.vae.encode(tensor))
    finally:
        _clear_vae_cache(pipe)
    mean, std = _vae_norm(pipe, latent.device, torch_module)
    latent = ((latent.float() - mean) * std).to(
        device=transformer_device, dtype=torch_module.float32
    )
    del tensor
    _clear_memory(torch_module)
    return latent


def _decode_video(pipe: Any, latent: Any, torch_module: Any) -> np.ndarray:
    vae_device = next(pipe.vae.parameters()).device
    vae_dtype = next(pipe.vae.parameters()).dtype
    mean, std = _vae_norm(pipe, latent.device, torch_module)
    latent = (latent.float() / std + mean).to(vae_device, vae_dtype)
    _clear_vae_cache(pipe)
    try:
        with torch_module.no_grad():
            decoded = pipe.vae.decode(latent).sample
    finally:
        _clear_vae_cache(pipe)
    result = decoded.squeeze(0).permute(1, 2, 3, 0).float().cpu().numpy()
    del decoded, latent
    _clear_memory(torch_module)
    return np.clip((result + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _encode_prompt(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    guide_scale: float,
    device: Any,
    dtype: Any,
    max_sequence_length: int,
) -> tuple:
    prompt_embeds = pipe._get_t5_prompt_embeds(
        prompt=prompt,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=pipe.text_encoder.device,
        dtype=pipe.text_encoder.dtype,
    ).to(device=device, dtype=dtype)
    negative_embeds = None
    if guide_scale > 1.0:
        negative_embeds = pipe._get_t5_prompt_embeds(
            prompt=negative_prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=pipe.text_encoder.device,
            dtype=pipe.text_encoder.dtype,
        ).to(device=device, dtype=dtype)
    return prompt_embeds, negative_embeds


def _prediction_statistics(prediction: Any) -> dict[str, float]:
    values = prediction.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "norm": float(values.norm().item()),
    }


def _ensure_finite_tensor(tensor: Any, stage: str, torch_module: Any) -> None:
    finite = torch_module.isfinite(tensor)
    if bool(finite.all().item()):
        return
    nonfinite_count = int((~finite).sum().item())
    total_count = int(tensor.numel())
    raise RuntimeError(
        f"{stage} contains {nonfinite_count}/{total_count} non-finite values. "
        "Denoising has numerically diverged; lower --guide-scale and inspect the "
        "reported stage."
    )


def _conditioning_difference_statistics(
    conditional: Any, unconditional: Any
) -> dict[str, float]:
    cond = conditional.float()
    uncond = unconditional.float()
    difference = cond - uncond
    cond_norm = float(cond.norm().item())
    uncond_norm = float(uncond.norm().item())
    difference_norm = float(difference.norm().item())
    denominator = max(cond_norm, uncond_norm, 1e-12)
    cosine = float(
        torch_cosine_similarity(cond.flatten(), uncond.flatten()).item()
    )
    return {
        **_prediction_statistics(difference),
        "relative_norm": difference_norm / denominator,
        "cosine_similarity": cosine,
    }


def torch_cosine_similarity(left: Any, right: Any) -> Any:
    import torch

    return torch.nn.functional.cosine_similarity(left, right, dim=0, eps=1e-12)


def _predict_with_cfg(
    pipe: Any,
    latents: Any,
    timestep_batch: Any,
    prompt_embeds: Any,
    negative_embeds: Any | None,
    guide_scale: float,
    torch_module: Any,
) -> tuple[Any, dict[str, Any]]:
    transformer_dtype = pipe.transformer.dtype
    hidden_states = latents.to(transformer_dtype)
    _ensure_finite_tensor(hidden_states, "transformer input", torch_module)
    with torch_module.no_grad(), _model_cache_context(pipe.transformer, "cond"):
        conditional = pipe.transformer(
            hidden_states=hidden_states,
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
    _ensure_finite_tensor(conditional, "conditional Transformer prediction", torch_module)

    unconditional = None
    guided = conditional.float()
    forward_count = 1
    if negative_embeds is not None:
        with torch_module.no_grad(), _model_cache_context(pipe.transformer, "uncond"):
            unconditional = pipe.transformer(
                hidden_states=hidden_states,
                timestep=timestep_batch,
                encoder_hidden_states=negative_embeds,
                return_dict=False,
            )[0]
        _ensure_finite_tensor(
            unconditional, "unconditional Transformer prediction", torch_module
        )
        # Keep CFG arithmetic and the scheduler state in FP32. UniPC multistep
        # updates can accumulate low-precision error until they abruptly diverge.
        conditional_f = conditional.float()
        unconditional_f = unconditional.float()
        guided = unconditional_f + guide_scale * (conditional_f - unconditional_f)
        forward_count = 2
    _ensure_finite_tensor(guided, "guided CFG prediction", torch_module)

    conditioning_difference = (
        _conditioning_difference_statistics(conditional, unconditional)
        if unconditional is not None
        else None
    )
    diagnostics = {
        "cfg_enabled": unconditional is not None,
        "transformer_forward_count": forward_count,
        "conditional": _prediction_statistics(conditional),
        "unconditional": (
            _prediction_statistics(unconditional) if unconditional is not None else None
        ),
        "conditional_minus_unconditional": conditioning_difference,
        "guided": _prediction_statistics(guided),
    }
    return guided, diagnostics


class WanFullVideoSDEdit:
    """Full-video latent SDEdit with text conditioning and no first-frame condition."""

    def __init__(self, pipeline: Any | None = None):
        import torch

        if pipeline is None:
            check_pipeline_dependencies()
            if not torch.cuda.is_available():
                raise RuntimeError("Wan2.2 TI2V SDEdit requires a CUDA GPU")
            pipeline = load_diffusers_pipeline(torch)
        self.torch = torch
        self.pipeline = pipeline

    def generate(
        self,
        source_video: np.ndarray,
        prompt: str,
        strength: float,
        seed: int,
        sampling_steps: int = 100,
        guide_scale: float = 5.0,
        max_sequence_length: int = 128,
        snapshot_callback: SnapshotCallback | None = None,
        snapshot_every: int = 10,
        negative_prompt: str = "",
    ) -> SDEditResult:
        torch, pipe = self.torch, self.pipeline
        if source_video.ndim != 4 or source_video.shape[-1] != 3:
            raise ValueError("source_video must have shape [F, H, W, 3]")
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be positive")
        start_idx = noise_strength_to_start_idx(strength, sampling_steps)
        expected_denoise_steps = denoise_step_count(strength, sampling_steps)
        scheduler_config = validate_official_scheduler(pipe.scheduler)
        transformer_device = next(pipe.transformer.parameters()).device
        transformer_dtype = pipe.transformer.dtype
        frames, height, width = source_video.shape[:3]

        print(
            "[stage 1/5] Encoding text prompt on CPU "
            f"(max_sequence_length={max_sequence_length}; this may take several minutes)...",
            flush=True,
        )
        prompt_embeds, negative_embeds = _encode_prompt(
            pipe,
            prompt,
            negative_prompt,
            guide_scale,
            transformer_device,
            transformer_dtype,
            max_sequence_length,
        )
        _ensure_finite_tensor(prompt_embeds, "positive prompt embedding", torch)
        if negative_embeds is not None:
            _ensure_finite_tensor(negative_embeds, "negative prompt embedding", torch)
        embedding_difference = (
            _conditioning_difference_statistics(prompt_embeds, negative_embeds)
            if negative_embeds is not None
            else None
        )
        text_encoder = pipe.text_encoder
        pipe.text_encoder = None
        del text_encoder
        _clear_memory(torch)
        print("[stage 1/5] Text prompt encoded; T5 released.", flush=True)
        cfg_enabled = negative_embeds is not None
        print(
            f"[cfg] enabled={cfg_enabled} guidance_scale={guide_scale} "
            f"forwards_per_step={2 if cfg_enabled else 1} "
            f"negative_prompt={negative_prompt!r}",
            flush=True,
        )
        if embedding_difference is not None:
            print(
                "[cfg] text embedding difference: "
                f"relative_norm={embedding_difference['relative_norm']:.6f} "
                f"cosine_similarity={embedding_difference['cosine_similarity']:.6f}",
                flush=True,
            )

        vae_device = getattr(pipe, "_vae_target_device", transformer_device)
        pipe.vae.to(vae_device, dtype=torch.float16)
        print(
            f"[stage 2/5] Encoding source video with VAE on {vae_device}: "
            f"{frames} frames at {width}x{height}...",
            flush=True,
        )
        clean = _encode_video(pipe, source_video, transformer_device, torch)
        _ensure_finite_tensor(clean, "encoded source latent", torch)
        latent_contract = validate_text_only_latent_contract(pipe.transformer, clean)
        print(f"[stage 2/5] Source latent shape={tuple(clean.shape)}", flush=True)
        print("[stage 2/5] Decoding VAE round-trip diagnostic...", flush=True)
        vae_roundtrip = _decode_video(pipe, clean, torch)
        vae_stats = validate_decoded_video(
            vae_roundtrip, "vae_roundtrip", expected_shape=source_video.shape
        )

        pipe.scheduler.set_timesteps(sampling_steps, device=transformer_device)
        noise_diagnostics = scheduler_noise_diagnostics(pipe.scheduler, start_idx)
        all_timesteps_head, all_timesteps_tail = tensor_head_tail(pipe.scheduler.timesteps)
        print(
            "[noise] "
            f"scheduler={noise_diagnostics['scheduler_class']} "
            f"timestep={noise_diagnostics['start_timestep']:.3f} "
            f"sigma={noise_diagnostics['schedule_sigma']:.4f} "
            f"signal_weight={noise_diagnostics['signal_weight']:.4f} "
            f"noise_weight={noise_diagnostics['noise_weight']:.4f} "
            f"flow_shift={noise_diagnostics['flow_shift']:.2f}",
            flush=True,
        )
        timesteps = pipe.scheduler.timesteps
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(start_idx)
        timesteps = timesteps[start_idx:]
        run_timesteps_head, run_timesteps_tail = tensor_head_tail(timesteps)
        if len(timesteps) != expected_denoise_steps:
            raise RuntimeError(
                "Scheduler returned an unexpected denoise-step count: "
                f"expected {expected_denoise_steps}, got {len(timesteps)}"
            )
        torch.manual_seed(seed)
        noise = torch.randn_like(clean)
        latents = add_noise_at_timestep(pipe.scheduler, clean, noise, timesteps[0])
        latents = latents.float()
        _ensure_finite_tensor(latents, "scheduler-noised source latent", torch)
        add_noise_verification = verify_scheduler_add_noise(
            clean, noise, latents, noise_diagnostics
        )
        print(
            "[noise] official add_noise formula verification: "
            f"max_abs_error={add_noise_verification['add_noise_formula_max_abs_error']:.6f} "
            f"clean_std={add_noise_verification['clean_latent_std']:.4f} "
            f"noisy_std={add_noise_verification['noisy_latent_std']:.4f}",
            flush=True,
        )
        snapshot_count = 0
        if snapshot_callback is not None:
            noisy_video = _decode_video(pipe, latents, torch)
            snapshot_callback(
                "noisy",
                noisy_video,
                {
                    "kind": "scheduler_add_noise",
                    "step": 0,
                    "timestep": float(timesteps[0].item()),
                },
            )
            snapshot_count += 1
            print("[stage 2/5] Saved direct scheduler.add_noise decode.", flush=True)

        print(
            f"[stage 3/5] Full-video latent ready; running {len(timesteps)} denoise steps "
            f"(strength={strength}, total scheduler steps={sampling_steps}).",
            flush=True,
        )

        last_step_video = None
        cfg_step_diagnostics = []
        for index, timestep in enumerate(timesteps):
            timestep_batch = _text_only_timestep(pipe, latents, timestep, torch)
            prediction, cfg_diagnostics = _predict_with_cfg(
                pipe,
                latents,
                timestep_batch,
                prompt_embeds,
                negative_embeds,
                guide_scale,
                torch,
            )
            cfg_step_diagnostics.append(
                {
                    "step": index + 1,
                    "timestep": float(timestep.item()),
                    **cfg_diagnostics,
                }
            )
            latents = pipe.scheduler.step(prediction, timestep, latents, return_dict=False)[0]
            latents = latents.float()
            _ensure_finite_tensor(
                latents, f"scheduler output after denoise step {index + 1}", torch
            )
            step_number = index + 1
            should_save_snapshot = (
                snapshot_callback is not None
                and should_save_denoise_snapshot(
                    step_number, len(timesteps), snapshot_every
                )
            )
            if should_save_snapshot:
                step_video = _decode_video(
                    pipe,
                    latents,
                    torch,
                )
                last_step_video = step_video
                snapshot_callback(
                    f"denoise_step_{step_number:03d}",
                    step_video,
                    {
                        "kind": "denoise_step",
                        "step": step_number,
                        "timestep": float(timestep.item()),
                    },
                )
                snapshot_count += 1
            if index == 0 or (index + 1) % 5 == 0 or index + 1 == len(timesteps):
                cond_stats = cfg_diagnostics["conditional"]
                guided_stats = cfg_diagnostics["guided"]
                uncond_stats = cfg_diagnostics["unconditional"]
                uncond_text = (
                    f" uncond_std={uncond_stats['std']:.4f}"
                    if uncond_stats is not None
                    else ""
                )
                difference = cfg_diagnostics["conditional_minus_unconditional"]
                difference_text = (
                    f" text_delta_relative_norm={difference['relative_norm']:.6f}"
                    if difference is not None
                    else ""
                )
                print(
                    f"[stage 4/5] Denoise {index + 1}/{len(timesteps)} "
                    f"forwards={cfg_diagnostics['transformer_forward_count']} "
                    f"cond_std={cond_stats['std']:.4f}{uncond_text} "
                    f"guided_std={guided_stats['std']:.4f}{difference_text}",
                    flush=True,
                )

        cfg_difference_summary = None
        if cfg_enabled:
            relative_norms = [
                step["conditional_minus_unconditional"]["relative_norm"]
                for step in cfg_step_diagnostics
            ]
            cfg_difference_summary = {
                "relative_norm_min": min(relative_norms),
                "relative_norm_max": max(relative_norms),
                "relative_norm_mean": sum(relative_norms) / len(relative_norms),
            }
            print(
                "[cfg] conditional-minus-unconditional summary: "
                f"relative_norm_mean={cfg_difference_summary['relative_norm_mean']:.6f} "
                f"min={cfg_difference_summary['relative_norm_min']:.6f} "
                f"max={cfg_difference_summary['relative_norm_max']:.6f}",
                flush=True,
            )
            if cfg_difference_summary["relative_norm_mean"] < 1e-4:
                print(
                    "[cfg] WARNING: text conditioning barely changes the Transformer "
                    "prediction. Check the prompt embeddings and checkpoint.",
                    flush=True,
                )

        print("[stage 5/5] Decoding edited video...", flush=True)
        generated_latent = latents
        generated_raw = (
            last_step_video
            if last_step_video is not None
            else _decode_video(pipe, generated_latent, torch)
        )
        generated_stats = validate_decoded_video(
            generated_raw, "generated_raw", expected_shape=source_video.shape
        )
        print("[stage 5/5] Decode complete.", flush=True)
        return SDEditResult(
            generated_raw=generated_raw,
            vae_roundtrip=vae_roundtrip,
            diagnostics={
                "clean_latent_shape": list(clean.shape),
                "generated_latent_shape": list(generated_latent.shape),
                "conditioning_mode": "full_video_latent_text_only",
                "first_frame_condition": False,
                "text_only_latent_contract": latent_contract,
                "text_cfg": {
                    "enabled": cfg_enabled,
                    "guidance_scale": guide_scale,
                    "positive_prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "transformer_forwards_per_step": 2 if cfg_enabled else 1,
                    "embedding_difference": embedding_difference,
                    "prediction_difference_summary": cfg_difference_summary,
                    "total_transformer_forwards": (
                        len(timesteps) * (2 if cfg_enabled else 1)
                    ),
                    "steps": cfg_step_diagnostics,
                },
                "total_scheduler_steps": sampling_steps,
                "denoise_start_index": start_idx,
                "actual_denoise_steps": len(timesteps),
                "scheduler": scheduler_config,
                "all_timesteps_head": all_timesteps_head,
                "all_timesteps_tail": all_timesteps_tail,
                "run_timesteps_head": run_timesteps_head,
                "run_timesteps_tail": run_timesteps_tail,
                "denoise_snapshot_every": snapshot_every,
                "saved_latent_decode_snapshots": snapshot_count,
                "initial_noise": noise_diagnostics,
                "add_noise_verification": add_noise_verification,
                "vae_roundtrip": vae_stats,
                "generated_raw": generated_stats,
            },
        )


WanTI2VSDEdit = WanFullVideoSDEdit


def diffusers_version() -> str:
    try:
        return importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
