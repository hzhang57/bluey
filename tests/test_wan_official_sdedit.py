import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mask_tracking.wan_official_sdedit import (
    MODEL_ID,
    load_official_wan_components,
    pad_timestep_for_official_model,
    require_official_flash_attention,
    resolve_official_wan_repo,
    strength_to_start_index,
)


class OfficialWanTests(unittest.TestCase):
    def test_uses_official_non_diffusers_checkpoint(self):
        self.assertEqual(MODEL_ID, "Wan-AI/Wan2.2-TI2V-5B")

    def test_strength_controls_actual_official_denoise_step_count(self):
        self.assertEqual(strength_to_start_index(0.45, 100), (55, 45))
        self.assertEqual(strength_to_start_index(0.60, 100), (40, 60))

    def test_official_backend_requires_flash_attention(self):
        missing = SimpleNamespace(
            FLASH_ATTN_2_AVAILABLE=False, FLASH_ATTN_3_AVAILABLE=False
        )
        with self.assertRaisesRegex(ImportError, "flash-attn"):
            require_official_flash_attention(missing)
        available = SimpleNamespace(
            FLASH_ATTN_2_AVAILABLE=True, FLASH_ATTN_3_AVAILABLE=False
        )
        require_official_flash_attention(available)

    def test_repo_resolver_finds_nested_kaggle_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout = root / "some-clone"
            marker = checkout / "wan" / "textimage2video.py"
            marker.parent.mkdir(parents=True)
            marker.write_text("", encoding="utf-8")
            self.assertEqual(
                resolve_official_wan_repo(None, search_roots=(root,)),
                checkout.resolve(),
            )

    def test_repo_resolver_gives_clone_command_for_invalid_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "git clone --depth 1"):
                resolve_official_wan_repo(Path(directory) / "missing")

    def test_minimal_loader_skips_official_package_init(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            package = repo / "wan"
            for child in ("configs", "modules", "utils"):
                (package / child).mkdir(parents=True, exist_ok=True)
                (package / child / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text(
                "raise RuntimeError('optional task imports ran')\n", encoding="utf-8"
            )
            (package / "textimage2video.py").write_text(
                "class WanTI2V: pass\n", encoding="utf-8"
            )
            (package / "configs" / "__init__.py").write_text(
                "WAN_CONFIGS = {'ti2v-5B': object()}\n", encoding="utf-8"
            )
            (package / "modules" / "attention.py").write_text(
                "FLASH_ATTN_2_AVAILABLE = True\nFLASH_ATTN_3_AVAILABLE = False\n",
                encoding="utf-8",
            )
            (package / "utils" / "fm_solvers_unipc.py").write_text(
                "class FlowUniPCMultistepScheduler: pass\n", encoding="utf-8"
            )

            with patch.dict(sys.modules):
                model, configs, attention, scheduler = load_official_wan_components(repo)

            self.assertEqual(model.__name__, "WanTI2V")
            self.assertIn("ti2v-5B", configs)
            self.assertTrue(attention.FLASH_ATTN_2_AVAILABLE)
            self.assertEqual(scheduler.__name__, "FlowUniPCMultistepScheduler")

    def test_expanded_timestep_covers_all_latent_tokens(self):
        import torch

        latent = torch.zeros((48, 6, 4, 8))
        timestep = torch.tensor(750, dtype=torch.int64)
        expanded = pad_timestep_for_official_model(
            timestep, latent, (1, 2, 2), seq_len=50, torch_module=torch
        )
        self.assertEqual(tuple(expanded.shape), (1, 50))
        self.assertEqual(expanded.dtype, latent.dtype)
        self.assertTrue(torch.all(expanded == timestep))

    def test_expanded_timestep_rejects_short_seq_len(self):
        import torch

        latent = torch.zeros((48, 6, 4, 8))
        with self.assertRaisesRegex(ValueError, "exceeds seq_len"):
            pad_timestep_for_official_model(
                torch.tensor(750), latent, (1, 2, 2), seq_len=23, torch_module=torch
            )


if __name__ == "__main__":
    unittest.main()
