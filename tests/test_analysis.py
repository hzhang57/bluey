import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from mask_tracking.analysis import extract_silhouette_mask, temporal_metrics
from mask_tracking.prompting import build_silhouette_prompt
from mask_tracking.wan_sdedit import (
    MODEL_ID,
    WanTI2VSDEdit,
    check_pipeline_dependencies,
    configure_low_memory_pipeline,
    load_diffusers_pipeline,
)
from run_mask_tracking import build_parser


class PromptTests(unittest.TestCase):
    def test_object_is_inserted(self):
        prompt = build_silhouette_prompt("the red car")
        self.assertIn("the red car", prompt)
        self.assertIn("visible parts", prompt)

    def test_empty_object_is_rejected(self):
        with self.assertRaises(ValueError):
            build_silhouette_prompt(" ")


class AnalysisTests(unittest.TestCase):
    def test_extracts_only_changed_white_pixels(self):
        source = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        edited = source.copy()
        edited[:, 2:6, 2:6] = 255
        masks = extract_silhouette_mask(source, edited, morphology_kernel=1)
        self.assertEqual(int((masks > 0).sum()), 2 * 4 * 4)

    def test_unchanged_white_is_not_a_mask(self):
        source = np.full((1, 4, 4, 3), 255, dtype=np.uint8)
        masks = extract_silhouette_mask(source, source, morphology_kernel=1)
        self.assertFalse(masks.any())

    def test_stable_mask_metrics(self):
        masks = np.zeros((3, 4, 4), dtype=np.uint8)
        masks[:, 1:3, 1:3] = 255
        metrics = temporal_metrics(masks)
        self.assertEqual(metrics["mean_consecutive_iou"], 1.0)
        self.assertEqual(metrics["mean_flicker_rate"], 0.0)


class FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeTorch:
    float32 = "float32"
    bfloat16 = "bfloat16"
    Generator = FakeGenerator

    class cuda:
        @staticmethod
        def empty_cache():
            pass


class FakeVAE:
    def __init__(self):
        self.tiling_enabled = False

    def enable_tiling(self):
        self.tiling_enabled = True


class FakePipeline:
    def __init__(self):
        self.kwargs = None
        self.sequential_offload_enabled = False
        self.vae = FakeVAE()

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        frames = np.zeros((1, 5, 4, 6, 3), dtype=np.float32)
        return SimpleNamespace(frames=frames)

    def enable_sequential_cpu_offload(self):
        self.sequential_offload_enabled = True


class DiffusersWrapperTests(unittest.TestCase):
    def test_fixed_model_id(self):
        self.assertEqual(MODEL_ID, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    def test_pipeline_dependency_preflight(self):
        with patch("mask_tracking.wan_sdedit.importlib.util.find_spec", return_value=object()):
            check_pipeline_dependencies()
        with patch("mask_tracking.wan_sdedit.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "pip install ftfy"):
                check_pipeline_dependencies()

    def test_forwards_video_editing_parameters(self):
        fake = FakePipeline()
        wrapper = WanTI2VSDEdit(pipeline=fake)
        wrapper.torch = FakeTorch
        source = np.zeros((5, 4, 6, 3), dtype=np.uint8)
        result = wrapper.generate(source, "paint it", 0.45, 123, 20, 4.0)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(fake.kwargs["strength"], 0.45)
        self.assertEqual(fake.kwargs["num_inference_steps"], 20)
        self.assertEqual(fake.kwargs["guidance_scale"], 4.0)
        self.assertEqual(fake.kwargs["generator"].seed, 123)
        self.assertEqual(len(fake.kwargs["video"]), 5)

    def test_configures_low_memory_pipeline(self):
        fake = FakePipeline()
        configure_low_memory_pipeline(fake)
        self.assertTrue(fake.sequential_offload_enabled)
        self.assertTrue(fake.vae.tiling_enabled)

    def test_loads_vae_in_float32_and_pipeline_in_bfloat16(self):
        vae = object()
        pipeline = FakePipeline()
        load_vae = MagicMock(return_value=vae)
        load_pipeline = MagicMock(return_value=pipeline)
        vae_class = SimpleNamespace(from_pretrained=load_vae)
        pipeline_class = SimpleNamespace(from_pretrained=load_pipeline)
        result = load_diffusers_pipeline(FakeTorch, vae_class, pipeline_class)
        self.assertIs(result, pipeline)
        load_vae.assert_called_once_with(
            MODEL_ID, subfolder="vae", torch_dtype=FakeTorch.float32
        )
        load_pipeline.assert_called_once_with(
            MODEL_ID, vae=vae, torch_dtype=FakeTorch.bfloat16
        )

    def test_kaggle_friendly_cli_defaults(self):
        args = build_parser().parse_args(["--video", "input.mp4", "--object", "car"])
        self.assertEqual(args.frame_num, 49)
        self.assertEqual(args.size, "832*480")
        self.assertEqual(args.sampling_steps, 30)

    def test_cli_has_no_repository_or_checkpoint_flags(self):
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--wan-repo", option_strings)
        self.assertNotIn("--wan-checkpoint", option_strings)


if __name__ == "__main__":
    unittest.main()
