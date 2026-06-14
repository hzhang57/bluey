# Wan2.2 TI2V 5B Load Demo

This minimal demo loads `Wan-AI/Wan2.2-TI2V-5B` with the official
`Wan-Video/Wan2.2` GitHub implementation. It only validates model loading; it
does not generate a video yet.

## Kaggle

```bash
%cd /kaggle/working
!git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git
%cd /kaggle/working/bluey
!pip install --no-deps -r requirements.txt
!python load_wan_ti2v.py
```

When no `--checkpoint-dir` is supplied, the 31.85 GiB official non-Diffusers
checkpoint is reused or downloaded under `/root/.cache/huggingface/hub`.
Kaggle's `/kaggle/working` mount may be only 20 GiB, so do not download the
checkpoint there.

The loader follows the official initialization:

```python
pipeline = WanTI2V(
    config=WAN_CONFIGS["ti2v-5B"],
    checkpoint_dir=checkpoint,
    device_id=0,
    t5_cpu=True,
    init_on_cpu=True,
    convert_model_dtype=True,
)
```

T5 and DiT remain on CPU after loading; the VAE is loaded on CUDA. The official
generation path moves the DiT to CUDA only while denoising. This load-only demo
does not require `flash-attn`; install it before adding the generation loop:

```bash
!pip install flash-attn --no-build-isolation --no-deps
```
