import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from mask_tracking.analysis import (
    composite_white_target,
    extract_silhouette_mask,
    temporal_metrics,
    validate_decoded_video,
)
from mask_tracking.prompting import build_silhouette_prompt
from mask_tracking.wan_sdedit import (
    MODEL_ID,
    _clear_vae_cache,
    check_pipeline_dependencies,
    denoise_step_count,
    load_diffusers_pipeline,
    noise_strength_to_start_idx,
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
    def test_extracts_gray_white_target_after_first_frame(self):
        source = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        source[:, 2:6, 2:6] = [180, 20, 20]
        generated = source.copy()
        generated[:, 2:6, 2:6] = [210, 210, 210]
        masks, score = extract_silhouette_mask(source, generated, morphology_kernel=1)
        self.assertFalse(masks[0].any())
        self.assertEqual(int((masks[1:] > 0).sum()), 2 * 4 * 4)
        self.assertGreater(float(score[1, 3, 3]), 0.20)

    def test_unchanged_white_is_not_a_mask(self):
        source = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
        masks, score = extract_silhouette_mask(source, source, morphology_kernel=1)
        self.assertFalse(masks.any())
        self.assertFalse(score.any())

    def test_composite_preserves_background_and_first_frame(self):
        source = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
        masks = np.zeros((3, 4, 4), dtype=np.uint8)
        masks[:, 1:3, 1:3] = 255
        edited = composite_white_target(source, masks)
        self.assertTrue(np.array_equal(edited[0], source[0]))
        selected = masks > 0
        selected[0] = False
        self.assertTrue(np.all(edited[selected] == 255))
        self.assertTrue(np.array_equal(edited[~selected], source[~selected]))

    def test_stable_mask_metrics(self):
        masks = np.zeros((3, 4, 4), dtype=np.uint8)
        masks[1:, 1:3, 1:3] = 255
        metrics = temporal_metrics(masks)
        self.assertTrue(metrics["skipped_first_frame"])
        self.assertEqual(len(metrics["foreground_fraction_per_frame"]), 2)
        self.assertEqual(metrics["mean_consecutive_iou"], 1.0)
        self.assertEqual(metrics["mean_flicker_rate"], 0.0)

    def test_near_black_decode_has_stage_diagnostic(self):
        black = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "vae_roundtrip decoded to a near-black"):
            validate_decoded_video(black, "vae_roundtrip")

    def test_non_black_decode_returns_statistics(self):
        video = np.full((2, 4, 4, 3), 128, dtype=np.uint8)
        stats = validate_decoded_video(video, "generated_raw")
        self.assertEqual(stats["mean"], 128.0)

    def test_decode_shape_mismatch_has_stage_diagnostic(self):
        video = np.full((2, 4, 4, 3), 128, dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "generated_raw decoded to shape"):
            validate_decoded_video(
                video, "generated_raw", expected_shape=(3, 4, 4, 3)
            )


class FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeTorch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    class cuda:
        @staticmethod
        def empty_cache():
            pass

        @staticmethod
        def device_count():
            return 2

    @staticmethod
    def device(name):
        return name


class FakeVAE:
    def __init__(self):
        self.slicing_enabled = False
        self.moves = []

    def enable_slicing(self):
        self.slicing_enabled = True

    def to(self, *args, **kwargs):
        self.moves.append((args, kwargs))


class FakePipeline:
    def __init__(self):
        self.vae = FakeVAE()
        self.transformer = SimpleNamespace(to=MagicMock())
        self.text_encoder = SimpleNamespace(to=MagicMock())
        self.registered_config = {}

    def register_to_config(self, **kwargs):
        self.registered_config.update(kwargs)


class DiffusersWrapperTests(unittest.TestCase):
    def test_fixed_model_id(self):
        self.assertEqual(MODEL_ID, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    def test_pipeline_dependency_preflight(self):
        with patch("mask_tracking.wan_sdedit.importlib.util.find_spec", return_value=object()):
            check_pipeline_dependencies()
        with patch("mask_tracking.wan_sdedit.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "pip install ftfy"):
                check_pipeline_dependencies()

    def test_strength_maps_to_scheduler_start(self):
        self.assertEqual(noise_strength_to_start_idx(0.5, 30), 15)
        self.assertEqual(noise_strength_to_start_idx(1.0, 30), 0)
        self.assertEqual(noise_strength_to_start_idx(0.45, 100), 55)
        self.assertEqual(denoise_step_count(0.45, 100), 45)
        self.assertEqual(denoise_step_count(0.451, 100), 46)

    def test_scheduler_steps_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "steps must be positive"):
            noise_strength_to_start_idx(0.45, 0)

    def test_vae_uses_official_cache_reset(self):
        vae = SimpleNamespace(
            _feat_map=["stale"],
            clear_cache=MagicMock(side_effect=lambda: setattr(vae, "_feat_map", [None])),
        )
        _clear_vae_cache(SimpleNamespace(vae=vae))
        vae.clear_cache.assert_called_once_with()
        self.assertEqual(vae._feat_map, [None])

    def test_loads_official_ti2v_pipeline_configuration(self):
        vae = object()
        pipeline = FakePipeline()
        load_vae = MagicMock(return_value=vae)
        load_pipeline = MagicMock(return_value=pipeline)
        vae_class = SimpleNamespace(from_pretrained=load_vae)
        pipeline_class = SimpleNamespace(from_pretrained=load_pipeline)
        result = load_diffusers_pipeline(FakeTorch, vae_class, pipeline_class)
        self.assertIs(result, pipeline)
        load_vae.assert_called_once_with(
            MODEL_ID, subfolder="vae", torch_dtype=FakeTorch.float16
        )
        load_pipeline.assert_called_once_with(
            MODEL_ID, vae=vae, torch_dtype=FakeTorch.bfloat16
        )
        self.assertTrue(pipeline.registered_config["expand_timesteps"])
        pipeline.transformer.to.assert_called_once_with("cuda:0")
        pipeline.text_encoder.to.assert_called_once_with("cpu")
        self.assertEqual(pipeline.vae.moves[0], (("cpu",), {"dtype": FakeTorch.float16}))
        self.assertEqual(pipeline._vae_target_device, "cuda:1")
        self.assertTrue(pipeline.vae.slicing_enabled)

    def test_kaggle_friendly_cli_defaults(self):
        args = build_parser().parse_args(["--video", "input.mp4", "--object", "car"])
        self.assertEqual(args.frame_num, 49)
        self.assertEqual(args.size, "832*480")
        self.assertEqual(args.sampling_steps, 100)
        self.assertEqual(args.max_sequence_length, 128)
        self.assertEqual(args.mask_score_threshold, 0.20)

    def test_cli_has_no_repository_or_checkpoint_flags(self):
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--wan-repo", option_strings)
        self.assertNotIn("--wan-checkpoint", option_strings)
        self.assertNotIn("--white-threshold", option_strings)
        self.assertNotIn("--difference-threshold", option_strings)
        self.assertIn("--mask-score-threshold", option_strings)


if __name__ == "__main__":
    unittest.main()
