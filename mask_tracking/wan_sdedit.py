from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
from typing import Any

import numpy as np

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REQUIRED_PIPELINE_PACKAGES = ("ftfy",)


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
    return min(round((1.0 - strength) * steps), steps - 1)


def load_diffusers_pipeline(torch_module: Any, vae_class=None, pipeline_class=None) -> Any:
    if vae_class is None or pipeline_class is None:
        try:
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        except ImportError as error:
            raise ImportError("Install the project requirements before loading Wan2.2") from error
        vae_class, pipeline_class = AutoencoderKLWan, WanImageToVideoPipeline

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
    pipe._text_target_device = text_device
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


def _expanded_timestep(model: Any, mask: Any, timestep: Any, batch_size: int) -> Any:
    patch_size = model.config.patch_size
    token_mask = mask[0][0][:, :: int(patch_size[1]), :: int(patch_size[2])]
    return (token_mask * timestep).flatten().unsqueeze(0).expand(batch_size, -1)


def _clear_memory(torch_module: Any) -> None:
    gc.collect()
    torch_module.cuda.empty_cache()


def _encode_prompt(
    pipe: Any,
    prompt: str,
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
            prompt="",
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=pipe.text_encoder.device,
            dtype=pipe.text_encoder.dtype,
        ).to(device=device, dtype=dtype)
    return prompt_embeds, negative_embeds


class WanTI2VSDEdit:
    """Source-video SDEdit using the official Wan2.2 TI2V expanded-timestep path."""

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
        sampling_steps: int = 30,
        guide_scale: float = 5.0,
        max_sequence_length: int = 128,
    ) -> np.ndarray:
        torch, pipe = self.torch, self.pipeline
        if source_video.ndim != 4 or source_video.shape[-1] != 3:
            raise ValueError("source_video must have shape [F, H, W, 3]")
        start_idx = noise_strength_to_start_idx(strength, sampling_steps)
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
            guide_scale,
            transformer_device,
            transformer_dtype,
            max_sequence_length,
        )
        text_encoder = pipe.text_encoder
        pipe.text_encoder = None
        del text_encoder
        _clear_memory(torch)
        print("[stage 1/5] Text prompt encoded; T5 released.", flush=True)

        vae_device = getattr(pipe, "_vae_target_device", transformer_device)
        pipe.vae.to(vae_device, dtype=torch.float16)
        vae_dtype = next(pipe.vae.parameters()).dtype
        print(
            f"[stage 2/5] Encoding source video with VAE on {vae_device}: "
            f"{frames} frames at {width}x{height}...",
            flush=True,
        )
        video = (
            torch.from_numpy(source_video)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(device=vae_device, dtype=vae_dtype)
            .div(127.5)
            .sub(1.0)
        )
        with torch.no_grad():
            clean = _retrieve_latents(pipe.vae.encode(video))
        mean, std = _vae_norm(pipe, clean.device, torch)
        clean = ((clean.float() - mean) * std).to(transformer_device, transformer_dtype)
        del video
        _clear_memory(torch)
        print(f"[stage 2/5] Source latent shape={tuple(clean.shape)}", flush=True)

        pipe.scheduler.set_timesteps(sampling_steps, device=transformer_device)
        timesteps = pipe.scheduler.timesteps
        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(start_idx)
        timesteps = timesteps[start_idx:]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.manual_seed(seed)
        noise = torch.randn_like(clean)
        latents = pipe.scheduler.add_noise(clean, noise, timesteps[:1])

        from PIL import Image

        print("[stage 3/5] Preparing official TI2V first-frame condition...", flush=True)
        image = pipe.video_processor.preprocess(
            Image.fromarray(source_video[0]), height=height, width=width
        ).to(device=vae_device, dtype=torch.float32)
        latents, condition, first_frame_mask = pipe.prepare_latents(
            image,
            batch_size=1,
            num_channels_latents=pipe.vae.config.z_dim,
            height=height,
            width=width,
            num_frames=frames,
            dtype=torch.float32,
            device=vae_device,
            generator=generator,
            latents=latents.to(device=vae_device, dtype=torch.float32),
        )
        latents = latents.to(transformer_device, transformer_dtype)
        condition = condition.to(transformer_device, transformer_dtype)
        first_frame_mask = first_frame_mask.to(transformer_device, transformer_dtype)
        print(
            f"[stage 3/5] Condition ready; running {len(timesteps)} denoise steps.",
            flush=True,
        )

        for index, timestep in enumerate(timesteps):
            model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
            timestep_batch = _expanded_timestep(pipe.transformer, first_frame_mask, timestep, latents.shape[0])
            with torch.no_grad():
                prediction = pipe.transformer(
                    hidden_states=model_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                if negative_embeds is not None:
                    uncond = pipe.transformer(
                        hidden_states=model_input,
                        timestep=timestep_batch,
                        encoder_hidden_states=negative_embeds,
                        return_dict=False,
                    )[0]
                    prediction = uncond + guide_scale * (prediction - uncond)
            latents = pipe.scheduler.step(prediction, timestep, latents, return_dict=False)[0]
            if index == 0 or (index + 1) % 5 == 0 or index + 1 == len(timesteps):
                print(f"[stage 4/5] Denoise {index + 1}/{len(timesteps)}", flush=True)

        print("[stage 5/5] Decoding edited video...", flush=True)
        latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
        mean, std = _vae_norm(pipe, latents.device, torch)
        latents = (latents.float() / std + mean).to(vae_device, vae_dtype)
        with torch.no_grad():
            decoded = pipe.vae.decode(latents).sample
        result = (
            decoded.squeeze(0)
            .permute(1, 2, 3, 0)
            .float()
            .cpu()
            .numpy()
        )
        print("[stage 5/5] Decode complete.", flush=True)
        return np.clip((result + 1.0) * 127.5, 0, 255).astype(np.uint8)


def diffusers_version() -> str:
    try:
        return importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
