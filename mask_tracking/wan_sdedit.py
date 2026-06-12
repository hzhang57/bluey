from __future__ import annotations

import gc
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def resolve_wan_repo(
    wan_repo: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> Path:
    candidates = []
    if wan_repo:
        candidates.append(Path(wan_repo))
    if os.environ.get("WAN_REPO"):
        candidates.append(Path(os.environ["WAN_REPO"]))
    if checkpoint_dir:
        checkpoint = Path(checkpoint_dir).expanduser()
        candidates.extend([checkpoint, checkpoint.parent, checkpoint.parent / "Wan2.2"])
    candidates.extend(
        [
            Path.cwd(),
            Path.cwd() / "Wan2.2",
            Path.cwd().parent / "Wan2.2",
            Path("/kaggle/working/Wan2.2"),
        ]
    )

    checked = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if (resolved / "wan" / "__init__.py").is_file():
            return resolved

    locations = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find the official Wan2.2 source repository.\n"
        "Clone it, then pass --wan-repo or set WAN_REPO:\n"
        "  git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git "
        "/kaggle/working/Wan2.2\n"
        "  pip install -r /kaggle/working/Wan2.2/requirements.txt\n"
        "Searched:\n"
        f"{locations}"
    )


def add_wan_repo_to_path(wan_repo: str | Path) -> Path:
    resolved = Path(wan_repo).expanduser().resolve()
    repo = str(resolved)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return resolved


class WanTI2VSDEdit:
    """Research adapter that adds source-video latent SDEdit to official WanTI2V."""

    def __init__(
        self,
        wan_repo: str | None,
        checkpoint_dir: str,
        device_id: int = 0,
        t5_cpu: bool = False,
        convert_model_dtype: bool = True,
    ):
        self.wan_repo = add_wan_repo_to_path(
            resolve_wan_repo(wan_repo, checkpoint_dir)
        )
        import torch
        try:
            import wan
            from wan.configs import WAN_CONFIGS
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"Found Wan2.2 at {self.wan_repo}, but its Python dependencies "
                "are incomplete. Install them with:\n"
                f"  pip install -r {self.wan_repo / 'requirements.txt'}"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.2-TI2V-5B SDEdit requires a CUDA GPU")
        self.torch = torch
        self.pipeline = wan.WanTI2V(
            config=WAN_CONFIGS["ti2v-5B"],
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
        )

    def generate(
        self,
        source_video,
        prompt: str,
        strength: float,
        seed: int,
        sampling_steps: int = 50,
        shift: float = 5.0,
        guide_scale: float = 5.0,
        negative_prompt: str = "",
        offload_model: bool = True,
        solver: str = "dpm++",
    ):
        torch = self.torch
        pipe = self.pipeline
        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        if solver not in {"dpm++", "unipc"}:
            raise ValueError("solver must be 'dpm++' or 'unipc'")

        source = (
            torch.from_numpy(source_video)
            .permute(3, 0, 1, 2)
            .float()
            .div(127.5)
            .sub(1.0)
            .to(pipe.device)
        )
        frame_num, height, width = source.shape[1:]
        seq_len = math.ceil(
            ((frame_num - 1) // pipe.vae_stride[0] + 1)
            * (height // pipe.vae_stride[1])
            * (width // pipe.vae_stride[2])
            / (pipe.patch_size[1] * pipe.patch_size[2])
            / pipe.sp_size
        ) * pipe.sp_size

        negative_prompt = negative_prompt or pipe.sample_neg_prompt
        context, context_null = self._encode_prompts(
            prompt, negative_prompt, offload_model
        )
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

        with torch.no_grad():
            source_latent = pipe.vae.encode([source])[0]
        noise = torch.randn(
            source_latent.shape,
            dtype=torch.float32,
            device=pipe.device,
            generator=generator,
        )
        scheduler = self._make_scheduler(solver, sampling_steps, shift)
        full_timesteps = scheduler.timesteps
        denoise_steps = max(1, min(sampling_steps, round(sampling_steps * strength)))
        start_index = sampling_steps - denoise_steps
        timesteps = full_timesteps[start_index:]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(start_index)
        latent = scheduler.add_noise(
            source_latent.unsqueeze(0), noise.unsqueeze(0), timesteps[:1]
        ).squeeze(0)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(pipe.model, "no_sync", noop_no_sync)
        from tqdm import tqdm

        with (
            torch.amp.autocast("cuda", dtype=pipe.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):
            if offload_model or pipe.init_on_cpu:
                pipe.model.to(pipe.device)
                torch.cuda.empty_cache()
            for timestep in tqdm(timesteps, desc=f"SDEdit strength={strength:.2f}"):
                model_timestep = timestep.expand(seq_len).unsqueeze(0)
                cond = pipe.model(
                    [latent], t=model_timestep, context=context, seq_len=seq_len
                )[0]
                uncond = pipe.model(
                    [latent], t=model_timestep, context=context_null, seq_len=seq_len
                )[0]
                prediction = uncond + guide_scale * (cond - uncond)
                latent = scheduler.step(
                    prediction.unsqueeze(0),
                    timestep,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0].squeeze(0)
            if offload_model:
                pipe.model.cpu()
                torch.cuda.empty_cache()
            video = pipe.vae.decode([latent])[0]

        result = (
            video.clamp(-1, 1)
            .add(1)
            .mul(127.5)
            .byte()
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )
        del source, source_latent, noise, latent, video, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def _encode_prompts(self, prompt: str, negative_prompt: str, offload: bool):
        torch = self.torch
        pipe = self.pipeline
        if not pipe.t5_cpu:
            pipe.text_encoder.model.to(pipe.device)
            context = pipe.text_encoder([prompt], pipe.device)
            context_null = pipe.text_encoder([negative_prompt], pipe.device)
            if offload:
                pipe.text_encoder.model.cpu()
                torch.cuda.empty_cache()
        else:
            cpu = torch.device("cpu")
            context = [value.to(pipe.device) for value in pipe.text_encoder([prompt], cpu)]
            context_null = [
                value.to(pipe.device)
                for value in pipe.text_encoder([negative_prompt], cpu)
            ]
        return context, context_null

    def _make_scheduler(self, solver: str, sampling_steps: int, shift: float):
        pipe = self.pipeline
        if solver == "unipc":
            from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=pipe.device, shift=shift)
            return scheduler

        from wan.utils.fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )

        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=pipe.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        retrieve_timesteps(
            scheduler,
            device=pipe.device,
            sigmas=get_sampling_sigmas(sampling_steps, shift),
        )
        return scheduler
