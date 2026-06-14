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
out of memory. Use model CPU offload and VAE tiling:

```bash
python run_wan_ti2v.py \
  --cpu-offload \
  --vae-tiling \
  --num-frames 21 \
  --height 480 \
  --width 832 \
  --output output_small.mp4
```

Wan requires `num_frames` to have the form `4n+1`, such as 21, 81, or 121.
The model is downloaded to the Hugging Face cache automatically.
