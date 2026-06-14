# Wan2.2 TI2V 5B Diffusers Demo

This demo follows the Hugging Face implementation for
`Wan-AI/Wan2.2-TI2V-5B-Diffusers`. It does not require the official Wan2.2
GitHub repository.

```python
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
output = pipe(...).frames[0]
```

## Run

For a normal Python environment:

```bash
pip install -r requirements.txt
python run_wan_ti2v.py
```

For Kaggle, keep its preinstalled Torch, CUDA, NumPy, and RAPIDS packages
unchanged:

```bash
pip install --no-deps -r requirements-kaggle.txt
python run_wan_ti2v.py
```

The `dask-cuda`, `cudf`, `cuml`, `numba-cuda`, and `ucxx` dependency conflict
messages come from Kaggle's unused preinstalled RAPIDS environment. They do
not indicate a Wan loading failure. Using the Kaggle command above prevents
pip from replacing that CUDA stack. Restart the Kaggle session once after
installing if Python had already imported Diffusers or Transformers.

The default parameters match the reference example: `1280x704`, 121 frames,
50 inference steps, guidance scale 5.0, and 24 FPS.

On a 15 GiB Kaggle T4, placing the complete pipeline on CUDA is likely to run
out of memory. A T4 also has no native BF16 support, so the default `auto`
dtype selects FP16. Use model CPU offload and VAE tiling:

```bash
python run_wan_ti2v.py \
  --cpu-offload \
  --vae-tiling \
  --num-frames 21 \
  --height 480 \
  --width 832 \
  --output output_small.mp4
```

The progress bar only advances after a complete denoise step. With guidance
scale 5.0, CFG performs two Transformer forwards per step, so `0/50` can remain
visible for several minutes on a T4 while GPU utilization stays at 100%.
`enable_model_cpu_offload()` uses only one GPU; an idle second T4 is expected.

Use this smaller command to verify the complete pipeline first:

```bash
python run_wan_ti2v.py \
  --cpu-offload \
  --vae-tiling \
  --num-frames 5 \
  --height 320 \
  --width 512 \
  --num-inference-steps 8 \
  --guidance-scale 1.0 \
  --output smoke_test.mp4
```

To keep pipeline components on both Kaggle T4s instead of CPU-offloading them,
use the experimental balanced device map:

```bash
python run_wan_ti2v.py \
  --balanced-device-map \
  --vae-tiling \
  --vae-tile-size 128 \
  --dtype float16 \
  --num-frames 21 \
  --height 480 \
  --width 832 \
  --output output_dual_t4.mp4
```

Balanced placement distributes components across GPUs, but it does not make a
single Transformer forward run in parallel. The balanced path finishes
denoising first, saves `<output>.latent.pt`, and moves the much smaller latent
to the VAE for decoding. After denoising, it releases the text encoder and
transformer, then moves a CPU-resident VAE to its mapped GPU. If the VAE still
does not fit, or the GPU runs out of memory during decode, decoding
automatically falls back to CPU. The default 128-pixel VAE tile uses less peak
memory than the Diffusers default.

If denoising already completed and `<output>.latent.pt` exists, decode it
without rerunning the 50 denoise steps or loading the full pipeline:

```bash
python decode_wan_latent.py \
  --latent /kaggle/working/5bit2v_output.latent.pt \
  --output /kaggle/working/5bit2v_output.mp4 \
  --device cuda:0 \
  --vae-tile-size 128
```

If GPU decode still exceeds 15 GiB, the decode-only script automatically
retries on CPU.

Wan requires `num_frames` to have the form `4n+1`, such as 21, 81, or 121.
The model is downloaded to the Hugging Face cache automatically.
