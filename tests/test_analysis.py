import unittest

import numpy as np

from mask_tracking.analysis import extract_silhouette_mask, temporal_metrics
from mask_tracking.prompting import build_silhouette_prompt


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


if __name__ == "__main__":
    unittest.main()
