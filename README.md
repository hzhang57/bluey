# Mask Tracking as an Emergent Capability of Wan2.2

This experiment probes whether a frozen `Wan2.2-TI2V-5B` can keep a local
silhouette edit attached to a text-specified object over time. It adds a
research-only source-video Latent SDEdit adapter to the official Wan inference
code. No detector, segmentation model, optical flow, or tracker is used.

## Setup

On a CUDA machine, clone the official Wan2.2 repository, install its
requirements, download `Wan-AI/Wan2.2-TI2V-5B`, then install this project's
requirements:

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_mask_tracking.py \
  --video input.mp4 \
  --object "the red car" \
  --wan-repo /path/to/Wan2.2 \
  --wan-checkpoint /path/to/Wan2.2-TI2V-5B \
  --strength 0.45 \
  --seed 42 \
  --output-dir outputs/red_car
```

`--object` accepts an arbitrary referring expression. A run processes one
continuous clip of `--frame-num` frames, which must have the form `4n+1`.
Short clips are padded with their final frame. Use `--start-frame` for another
window. Wan2.2-TI2V-5B natively supports `1280*704` and `704*1280`.

For a small sweep:

```bash
for strength in 0.30 0.45 0.60; do
  for seed in 42 123 456; do
    python run_mask_tracking.py \
      --video input.mp4 \
      --object "the red car" \
      --wan-repo /path/to/Wan2.2 \
      --wan-checkpoint /path/to/Wan2.2-TI2V-5B \
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
# bluey
