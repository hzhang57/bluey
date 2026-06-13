# Mask Tracking as an Emergent Capability of Wan2.2

This experiment probes whether a frozen video editing model can keep a local
white silhouette edit attached to a text-specified object over time. It uses
Diffusers `WanVideoToVideoPipeline` with the fixed model:

```text
Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

No detector, segmentation model, optical flow, tracker, or Wan2.2 GitHub clone
is used.

## Setup

On a Kaggle CUDA notebook:

```bash
%cd /kaggle/working/bluey
!pip install -r requirements.txt
```

Before running, set **Notebook options > Accelerator > GPU**. Verify that the
environment is not using a CPU-only PyTorch build:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The final value must be `True`. If packages were replaced during installation,
restart the Kaggle session before inference.

The first run downloads the fixed Diffusers model from Hugging Face.

## Run

```bash
python run_mask_tracking.py \
  --video input.mp4 \
  --object "the red car" \
  --strength 0.45 \
  --seed 42 \
  --output-dir outputs/red_car
```

`--object` accepts an arbitrary referring expression. A run processes one
continuous clip of `--frame-num` frames, which must have the form `4n+1`.
Short clips are padded with their final frame. Use `--start-frame` for another
window. Supported sizes are `1280*704` and `704*1280`.

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
