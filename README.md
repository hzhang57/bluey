# Mask Tracking as an Emergent Capability of Wan2.2

This experiment probes whether a frozen video editing model can keep a local
white silhouette edit attached to a text-specified object over time. It uses
Diffusers `WanPipeline` with full-video latent SDEdit and the fixed model:

```text
Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

No detector, segmentation model, optical flow, tracker, or Wan2.2 GitHub clone
is used.

## Setup

On a Kaggle CUDA notebook:

```bash
%cd /kaggle/working/bluey
!pip install --no-deps -r requirements-kaggle.txt
```

Before running, set **Notebook options > Accelerator > GPU**. Verify that the
environment is not using a CPU-only PyTorch build and that the required
pipeline can be imported:

```bash
python -c "import torch; from diffusers import WanPipeline; print(torch.__version__, torch.cuda.is_available())"
```

The final value must be `True`. If packages were replaced during installation,
restart the Kaggle session before inference.

The first run downloads the fixed Diffusers model from Hugging Face.
Setting an optional `HF_TOKEN` increases download rate limits. Diffusers Flax
deprecation warnings are unrelated to this PyTorch pipeline and can be ignored.

Warnings about Kaggle's unused RAPIDS packages such as `dask-cuda`, `cudf`,
`cuml`, `numba-cuda`, or `cuda-core` do not affect this experiment. Starting a
fresh GPU session and using `requirements-kaggle.txt` avoids modifying that
preinstalled CUDA stack.

## Run

```bash
python run_mask_tracking.py \
  --video input.mp4 \
  --object "the red car" \
  --negative-prompt "" \
  --strength 0.45 \
  --seed 42 \
  --output-dir outputs/red_car
```

The default `49` frames, `832*480`, and `100` total scheduler steps are selected
so `--strength 0.45` executes `45` actual denoise steps. The implementation
encodes the full source video, adds scheduler noise, and denoises every latent
frame using text conditioning only. It does not call `prepare_latents`, inject
an image condition, or preserve the first frame specially. On dual T4, the
transformer uses `cuda:0`, T5 stays on CPU because UMT5 does not fit in a
15 GiB T4, and the VAE uses `cuda:1`. Prompt length defaults to 128 tokens to
reduce CPU T5 time. Long stages print explicit progress messages.

The checkpoint's official `expand_timesteps=True` transformer interface is
still used. In text-only mode its token mask is all ones, so every spatiotemporal
token receives the current scheduler timestep; no first-frame token is assigned
timestep zero.

Text is injected through classifier-free guidance (CFG). With the default
`--guide-scale 5.0`, every denoise step runs one positive-prompt Transformer
forward and one negative-prompt Transformer forward, then combines them as
`uncond + scale * (cond - uncond)`. `--negative-prompt` defaults to an empty
string. With `--guide-scale 1.0`, only the positive-prompt forward runs. The
manifest records prompts, guidance scale, forward counts, and per-step
conditional/unconditional/guided prediction statistics under
`diagnostics.text_cfg`.

The denoise log also reports `text_delta_relative_norm`, the relative size of
the conditional-minus-unconditional prediction. A nonzero value proves that
the prompt changes the Transformer prediction at that step. If it is nonzero
but the decoded video remains close to the source, the source-video SDEdit
prior is dominating; compare stronger runs such as `--strength 0.60` or
`--strength 0.75` and `--guide-scale 7.5`. The manifest records the positive
versus negative embedding difference and every per-step prediction difference.

For a direct prompt counterfactual, bypass the generated object template:

```bash
python run_mask_tracking.py \
  --video input.mp4 \
  --object "the red car" \
  --prompt "Replace the red car with a featureless solid pure white car." \
  --strength 0.75 \
  --guide-scale 7.5
```

If the GPU still runs out of memory, restart the Kaggle session and run:

```bash
python run_mask_tracking.py \
  --video input.mp4 \
  --object "the red car" \
  --frame-num 25 \
  --size 832*480 \
  --sampling-steps 100
```

`--object` accepts an arbitrary referring expression. A run processes one
continuous clip of `--frame-num` frames, which must have the form `4n+1`.
Short clips are padded with their final frame. Use `--start-frame` for another
window. Supported sizes include `832*480`, `480*832`, `1280*704`, and
`704*1280`.

For a small sweep:

```bash
for strength in 0.30 0.45 0.60; do
  for seed in 42 123 456; do
    python run_mask_tracking.py \
      --video input.mp4 \
      --object "the red car" \
      --strength "$strength" \
      --seed "$seed" \
      --output-dir "outputs/red_car_s${strength}_seed${seed}"
  done
done
```

Each run saves:

- `generated_raw.mp4`: unmodified Wan output and the primary research evidence.
- `raw_mask.mp4`: binary mask extracted from relative whitening.
- `mask_score.mp4`: continuous relative-whitening score heatmap.
- `edited.mp4`: source video composited with pure white only inside the mask.
- `composite_arrays.npz`: lossless source, binary mask, and composited RGB
  arrays for pixel-exact locality checks; MP4 encoding itself is lossy.
- `vae_roundtrip.mp4`: source video after Wan VAE encode/decode only.
- `noisy.mp4`: direct VAE decode of the full-video latent returned by
  `scheduler.add_noise`.
- `denoise_steps/step_NNN_timestep_*.mp4`: decoded video every 10 denoise steps
  and at the final step.
- `side_by_side.mp4`: source, raw generation, mask, and final composite.
- `manifest.json`: parameters, pixel statistics, mask coverage, and no-GT
  temporal stability diagnostics.

Denoise-step snapshots are enabled by default. They are saved every 10 steps
and at the final step; for a 45-step run this saves steps 10, 20, 30, 40, and
45. Change the interval with `--denoise-save-every` or disable snapshots with
`--no-save-denoise-steps`.

`strength` controls the fraction of the official scheduler trajectory used for
denoising; it is not the final noise weight. Wan2.2 TI2V uses the model's
official `UniPCMultistepScheduler` with flow sigmas and `flow_shift=5.0`.
Consequently, `strength=0.45` starts near effective `sigma=0.80`, so
`noisy.mp4` may retain little visible source structure. The exact scheduler
class, timestep, sigma, signal weight, and noise weight are printed at runtime
and recorded under `diagnostics.initial_noise` in `manifest.json`. The manifest
also records latent standard deviations and the numerical error between the
official `scheduler.add_noise` result and its current signal/noise formula
under `diagnostics.add_noise_verification`.

The SDEdit noise path follows `codes_point_prompting/debug_denoise_moe.py`:
it validates the official checkpoint scheduler configuration, maps strength to
the descending scheduler timestep with the same rounded index calculation, and
calls `scheduler.add_noise` with the selected timestep explicitly moved to the
clean latent device. Unlike that reference script, this experiment deliberately
does not inject its TI2V first-frame condition. The full and executed timestep
head/tail values are stored in the manifest.

The mask score combines generated brightness, low chroma, brightness gain
relative to the source, and total pixel change. Tune it with
`--mask-score-threshold` (default `0.20`). Frame 0 is treated exactly like every
other frame during noising, denoising, mask extraction, compositing, and
temporal evaluation.

## Interpretation

A positive result requires the white edit in `generated_raw.mp4` to remain
attached to the same object through motion and occlusion. `edited.mp4` always
keeps non-mask pixels from the source by construction, so it is a visualization
of the extracted mask and is not evidence that Wan preserved the background.
The metrics measure output stability, not segmentation accuracy; without
ground-truth masks this remains a discovery probe.
