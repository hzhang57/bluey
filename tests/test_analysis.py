import unittest
from types import SimpleNamespace

import numpy as np

from mask_tracking.analysis import extract_silhouette_mask, temporal_metrics
from mask_tracking.prompting import build_silhouette_prompt
from mask_tracking.wan_sdedit import MODEL_ID, WanTI2VSDEdit
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
    Generator = FakeGenerator


class FakePipeline:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        frames = np.zeros((1, 5, 4, 6, 3), dtype=np.float32)
        return SimpleNamespace(frames=frames)


class DiffusersWrapperTests(unittest.TestCase):
    def test_fixed_model_id(self):
        self.assertEqual(MODEL_ID, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

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
