# Mask Tracking as an Emergent Capability of Wan2.2

This experiment probes whether a frozen video editing model can keep a local
white silhouette edit attached to a text-specified object over time. It uses
Diffusers `WanImageToVideoPipeline` with full-video latent SDEdit and the fixed
model:

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
python -c "import torch; from diffusers import WanImageToVideoPipeline; print(torch.__version__, torch.cuda.is_available())"
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
  --strength 0.45 \
  --seed 42 \
  --output-dir outputs/red_car
```

The default `49` frames, `832*480`, and `30` sampling steps are selected for
Kaggle GPUs. The implementation follows the official TI2V expanded-timestep
path: encode the full source video, add scheduler noise, inject the first-frame
condition through `prepare_latents`, then denoise. On dual T4, the transformer
uses `cuda:0`; `cuda:1` first encodes the T5 prompt, releases T5, and then runs
the VAE. Long stages print explicit progress messages, so silence after
pipeline loading should no longer look like a stalled process.

If the GPU still runs out of memory, restart the Kaggle session and run:

```bash
python run_mask_tracking.py \
  --video input.mp4 \
  --object "the red car" \
  --frame-num 25 \
  --size 832*480 \
  --sampling-steps 20
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

Each run saves the source, edited video, extracted raw mask, overlay,
side-by-side visualization, and a `manifest.json` containing parameters and
no-GT temporal stability diagnostics.

## Interpretation

A positive result requires the white silhouette to remain attached to the same
object through motion and occlusion while leaving the background unchanged.
The generated mask is extracted only from pixels that became white relative to
the source. The metrics measure output stability, not segmentation accuracy;
without ground-truth masks this remains a discovery probe.
