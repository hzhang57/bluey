import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from mask_tracking.analysis import (
    TARGET_COLORS,
    build_global_color_prompt,
    target_color_analysis,
    to_grayscale_rec709,
)
from run_gray_colorizing import build_parser, main, validate_args


class GrayColorAnalysisTests(unittest.TestCase):
    def test_rec709_grayscale_has_equal_channels(self):
        video = np.array([[[[255, 0, 0], [0, 255, 0], [0, 0, 255]]]], dtype=np.uint8)
        grayscale = to_grayscale_rec709(video)
        self.assertTrue(np.array_equal(grayscale[..., 0], grayscale[..., 1]))
        self.assertTrue(np.array_equal(grayscale[..., 1], grayscale[..., 2]))
        self.assertTrue(np.array_equal(grayscale[0, 0, :, 0], [54, 182, 18]))

    def test_prompt_names_requested_color(self):
        prompt = build_global_color_prompt("magenta")
        self.assertIn("vivid magenta", prompt)
        self.assertIn("whole visible scene", prompt)

    def test_exact_target_colors_are_selected(self):
        grayscale = np.full((1, 2, 2, 3), 128, dtype=np.uint8)
        for color in ("magenta", "cyan", "lime"):
            generated = np.empty_like(grayscale)
            generated[:] = TARGET_COLORS[color]
            mask, score, metrics = target_color_analysis(grayscale, generated, color)
            self.assertTrue(np.all(mask == 255))
            self.assertGreater(float(score.min()), 0.99)
            self.assertEqual(metrics["target_color_coverage"], 1.0)

    def test_gray_and_wrong_color_are_not_selected(self):
        grayscale = np.full((1, 2, 2, 3), 128, dtype=np.uint8)
        gray_mask, gray_score, _ = target_color_analysis(
            grayscale, grayscale, "magenta"
        )
        cyan = np.empty_like(grayscale)
        cyan[:] = TARGET_COLORS["cyan"]
        wrong_mask, _, _ = target_color_analysis(grayscale, cyan, "magenta")
        self.assertFalse(gray_mask.any())
        self.assertFalse(gray_score.any())
        self.assertFalse(wrong_mask.any())

    def test_dark_pixels_are_excluded_from_mask_and_coverage(self):
        grayscale = np.zeros((1, 1, 2, 3), dtype=np.uint8)
        grayscale[0, 0, 1] = 128
        generated = np.empty_like(grayscale)
        generated[:] = TARGET_COLORS["magenta"]
        mask, _, metrics = target_color_analysis(grayscale, generated, "magenta")
        self.assertEqual(mask[0, 0, 0], 0)
        self.assertEqual(mask[0, 0, 1], 255)
        self.assertEqual(metrics["eligible_pixel_fraction"], 0.5)
        self.assertEqual(metrics["target_color_coverage"], 1.0)


class GrayColorCliTests(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args(["--video", "input.mp4"])
        self.assertEqual(args.color, "magenta")
        self.assertEqual(args.strength, 0.60)
        self.assertEqual(args.frame_num, 25)
        self.assertEqual(args.guide_scale, 5.0)
        self.assertEqual(args.saturation_threshold, 0.20)
        self.assertEqual(args.hue_tolerance_degrees, 30.0)
        self.assertEqual(args.minimum_luma, 0.05)

    def test_prompt_override_and_color_are_independent(self):
        args = build_parser().parse_args(
            ["--video", "input.mp4", "--color", "cyan", "--prompt", "custom"]
        )
        self.assertEqual(args.color, "cyan")
        self.assertEqual(args.prompt, "custom")

    def test_rejects_invalid_threshold(self):
        args = build_parser().parse_args(
            ["--video", "input.mp4", "--saturation-threshold", "2"]
        )
        with self.assertRaisesRegex(ValueError, "--saturation-threshold"):
            validate_args(args)

    def test_main_passes_grayscale_video_and_writes_manifest(self):
        original = np.zeros((5, 4, 4, 3), dtype=np.uint8)
        original[..., 0] = 200
        original[..., 1] = 100
        original[..., 2] = 20
        generated = np.empty_like(original)
        generated[:] = TARGET_COLORS["magenta"]

        class FakePipeline:
            def __init__(self):
                self.received = None

            def generate(self, source, *args, **kwargs):
                self.received = source.copy()
                return SimpleNamespace(
                    generated_raw=generated,
                    vae_roundtrip=source,
                    diagnostics={"fake": True},
                )

        fake_pipeline = FakePipeline()
        write_video = MagicMock()
        fake_video_io = SimpleNamespace(
            make_colorization_comparison=lambda original, grayscale, generated, score: np.concatenate(
                [original, grayscale, generated, np.repeat(score[..., None], 3, axis=-1)],
                axis=2,
            ),
            mask_to_rgb=lambda mask: np.repeat(mask[..., None], 3, axis=-1),
            read_video_clip=MagicMock(
                return_value=(original, 24.0, {"source_fps": 24.0})
            ),
            score_to_rgb=lambda score: np.repeat(score[..., None], 3, axis=-1),
            write_video=write_video,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys,
            "argv",
            [
                "run_gray_colorizing.py",
                "--video",
                "input.mp4",
                "--frame-num",
                "5",
                "--output-dir",
                directory,
            ],
        ), patch.dict(
            sys.modules, {"mask_tracking.video_io": fake_video_io}
        ), patch(
            "mask_tracking.wan_sdedit.WanFullVideoSDEdit",
            return_value=fake_pipeline,
        ), patch(
            "mask_tracking.wan_sdedit.diffusers_version",
            return_value="test",
        ):
            main()
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            with np.load(Path(directory) / "colorization_arrays.npz") as archive:
                grayscale_array = archive["grayscale_input"].copy()

        self.assertTrue(
            np.array_equal(fake_pipeline.received[..., 0], fake_pipeline.received[..., 1])
        )
        self.assertTrue(
            np.array_equal(fake_pipeline.received[..., 1], fake_pipeline.received[..., 2])
        )
        self.assertTrue(np.array_equal(grayscale_array, fake_pipeline.received))
        self.assertEqual(manifest["target_color"], "magenta")
        self.assertEqual(manifest["target_hsv"]["hue_degrees"], 300.0)
        self.assertIn("not a segmentation", manifest["research_warning"])
        self.assertEqual(manifest["outputs"]["grayscale_input"], "grayscale_input.mp4")
        self.assertEqual(manifest["outputs"]["lossless_arrays"], "colorization_arrays.npz")
        written_names = {Path(call.args[0]).name for call in write_video.call_args_list}
        self.assertEqual(
            written_names,
            {
                "original_color.mp4",
                "grayscale_input.mp4",
                "generated_raw.mp4",
                "target_color_score.mp4",
                "target_color_mask.mp4",
                "vae_roundtrip.mp4",
                "side_by_side.mp4",
            },
        )


if __name__ == "__main__":
    unittest.main()
