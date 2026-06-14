import tempfile
import unittest
from pathlib import Path

from load_wan_ti2v import (
    MODEL_ID,
    build_parser,
    find_cached_checkpoint,
    is_official_checkpoint,
    resolve_checkpoint,
    resolve_wan_repo,
    validate_args,
)


def create_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for name in (
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2_VAE.pth",
        "config.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ):
        (path / name).write_text("", encoding="utf-8")


class LoadWanTests(unittest.TestCase):
    def test_cli_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.wan_repo, "/kaggle/working/Wan2.2")
        self.assertIsNone(args.checkpoint_dir)
        self.assertEqual(args.device_id, 0)
        self.assertEqual(args.text_length, 512)
        self.assertFalse(args.no_download)

    def test_rejects_invalid_text_length(self):
        args = build_parser().parse_args(["--text-length", "513"])
        with self.assertRaisesRegex(ValueError, "--text-length"):
            validate_args(args)

    def test_resolves_official_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            marker = repo / "wan" / "textimage2video.py"
            marker.parent.mkdir()
            marker.write_text("", encoding="utf-8")
            self.assertEqual(resolve_wan_repo(repo), repo.resolve())

    def test_detects_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            create_checkpoint(checkpoint)
            self.assertTrue(is_official_checkpoint(checkpoint))
            self.assertEqual(
                resolve_checkpoint(checkpoint, allow_download=False),
                checkpoint.resolve(),
            )

    def test_finds_hugging_face_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            snapshot = (
                cache
                / "models--Wan-AI--Wan2.2-TI2V-5B"
                / "snapshots"
                / "revision"
            )
            create_checkpoint(snapshot)
            self.assertEqual(find_cached_checkpoint(cache), snapshot.resolve())

    def test_downloads_to_default_hugging_face_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "snapshot"

            def fake_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], MODEL_ID)
                self.assertNotIn("local_dir", kwargs)
                create_checkpoint(checkpoint)
                return str(checkpoint)

            self.assertEqual(
                resolve_checkpoint(
                    None,
                    allow_download=True,
                    snapshot_download_fn=fake_download,
                ),
                checkpoint.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
