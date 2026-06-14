import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from run_wan_ti2v import MODEL_ID, build_parser, generate, load_pipeline, validate_args


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 2


class FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeTorch:
    cuda = FakeCuda()
    float32 = "float32"
    bfloat16 = "bfloat16"
    Generator = FakeGenerator


class FakeVAE:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.load_args = args
        cls.load_kwargs = kwargs
        return cls()


class FakePipeline:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.load_args = args
        cls.load_kwargs = kwargs
        instance = cls()
        instance.vae = kwargs["vae"]
        instance.vae.enable_tiling = MagicMock()
        instance.enable_model_cpu_offload = MagicMock()
        instance.to = MagicMock()
        return instance


class WanDiffusersDemoTests(unittest.TestCase):
    def test_kaggle_requirements_preserve_preinstalled_cuda_stack(self):
        requirements_path = Path(__file__).parents[1] / "requirements-kaggle.txt"
        requirements = [
            line.strip().lower()
            for line in requirements_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        protected_prefixes = (
            "torch",
            "torchvision",
            "numpy",
            "cuda-",
            "numba",
            "dask-cuda",
            "cudf",
            "cuml",
            "ucxx",
            "distributed-ucxx",
        )
        self.assertFalse(
            any(item.startswith(protected_prefixes) for item in requirements)
        )
        self.assertIn("diffusers==0.38.0", requirements)
        self.assertIn("transformers==4.57.6", requirements)

    def test_cli_defaults_match_reference(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.height, 704)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.num_frames, 121)
        self.assertEqual(args.num_inference_steps, 50)
        self.assertEqual(args.guidance_scale, 5.0)
        self.assertFalse(args.cpu_offload)

    def test_rejects_invalid_frame_count(self):
        args = build_parser().parse_args(["--num-frames", "20"])
        with self.assertRaisesRegex(ValueError, "4n\\+1"):
            validate_args(args)

    def test_rejects_unaligned_size(self):
        args = build_parser().parse_args(["--height", "720"])
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            validate_args(args)

    def test_loads_fp32_vae_and_bf16_pipeline(self):
        pipe = load_pipeline(
            device_id=0,
            cpu_offload=False,
            vae_tiling=False,
            torch_module=FakeTorch,
            autoencoder_class=FakeVAE,
            pipeline_class=FakePipeline,
        )
        self.assertEqual(FakeVAE.load_args, (MODEL_ID,))
        self.assertEqual(FakeVAE.load_kwargs["subfolder"], "vae")
        self.assertEqual(FakeVAE.load_kwargs["torch_dtype"], "float32")
        self.assertEqual(FakePipeline.load_args, (MODEL_ID,))
        self.assertEqual(FakePipeline.load_kwargs["torch_dtype"], "bfloat16")
        pipe.to.assert_called_once_with("cuda:0")
        pipe.enable_model_cpu_offload.assert_not_called()

    def test_cpu_offload_and_vae_tiling(self):
        pipe = load_pipeline(
            device_id=1,
            cpu_offload=True,
            vae_tiling=True,
            torch_module=FakeTorch,
            autoencoder_class=FakeVAE,
            pipeline_class=FakePipeline,
        )
        pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=1)
        pipe.to.assert_not_called()
        pipe.vae.enable_tiling.assert_called_once_with()

    def test_generation_arguments_are_forwarded(self):
        args = build_parser().parse_args([])
        pipe = MagicMock(
            return_value=SimpleNamespace(frames=[["generated video"]])
        )
        result = generate(args, pipe, torch_module=FakeTorch)
        self.assertEqual(result, ["generated video"])
        kwargs = pipe.call_args.kwargs
        self.assertEqual(kwargs["num_frames"], 121)
        self.assertEqual(kwargs["height"], 704)
        self.assertEqual(kwargs["width"], 1280)
        self.assertEqual(kwargs["generator"].device, "cuda:0")
        self.assertEqual(kwargs["generator"].seed, 42)


if __name__ == "__main__":
    unittest.main()
