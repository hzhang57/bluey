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
    _encode_prompt,
    _predict_with_cfg,
    _text_only_timestep,
    add_noise_at_timestep,
    check_pipeline_dependencies,
    denoise_step_count,
    load_diffusers_pipeline,
    noise_strength_to_start_idx,
    scheduler_noise_diagnostics,
    should_save_denoise_snapshot,
    validate_text_only_latent_contract,
    validate_official_scheduler,
)
from run_mask_tracking import build_parser, validate_args


class PromptTests(unittest.TestCase):
    def test_object_is_inserted(self):
        prompt = build_silhouette_prompt("the red car")
        self.assertIn("the red car", prompt)
        self.assertIn("visible parts", prompt)

    def test_empty_object_is_rejected(self):
        with self.assertRaises(ValueError):
            build_silhouette_prompt(" ")


class AnalysisTests(unittest.TestCase):
    def test_extracts_gray_white_target_including_first_frame(self):
        source = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        source[:, 2:6, 2:6] = [180, 20, 20]
        generated = source.copy()
        generated[:, 2:6, 2:6] = [210, 210, 210]
        masks, score = extract_silhouette_mask(source, generated, morphology_kernel=1)
        self.assertTrue(masks[0].any())
        self.assertEqual(int((masks > 0).sum()), 3 * 4 * 4)
        self.assertGreater(float(score[1, 3, 3]), 0.20)

    def test_unchanged_white_is_not_a_mask(self):
        source = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
        masks, score = extract_silhouette_mask(source, source, morphology_kernel=1)
        self.assertFalse(masks.any())
        self.assertFalse(score.any())

    def test_composite_whitens_mask_including_first_frame(self):
        source = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
        masks = np.zeros((3, 4, 4), dtype=np.uint8)
        masks[:, 1:3, 1:3] = 255
        edited = composite_white_target(source, masks)
        selected = masks > 0
        self.assertTrue(np.all(edited[selected] == 255))
        self.assertTrue(np.array_equal(edited[~selected], source[~selected]))

    def test_stable_mask_metrics(self):
        masks = np.zeros((3, 4, 4), dtype=np.uint8)
        masks[:, 1:3, 1:3] = 255
        metrics = temporal_metrics(masks)
        self.assertFalse(metrics["skipped_first_frame"])
        self.assertEqual(len(metrics["foreground_fraction_per_frame"]), 3)
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
        self.assertEqual(denoise_step_count(0.451, 100), 45)

    def test_scheduler_steps_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "steps must be positive"):
            noise_strength_to_start_idx(0.45, 0)

    def test_saves_every_tenth_and_final_denoise_step(self):
        saved = [
            step
            for step in range(1, 46)
            if should_save_denoise_snapshot(step, total_steps=45, every=10)
        ]
        self.assertEqual(saved, [10, 20, 30, 40, 45])

    def test_reports_effective_flow_scheduler_noise(self):
        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

            def __truediv__(self, other):
                return FakeScalar(self.value / other.value)

        scheduler = SimpleNamespace(
            sigmas=[FakeScalar(0.804)],
            timesteps=[FakeScalar(804.0)],
            config=SimpleNamespace(
                use_flow_sigmas=True,
                flow_shift=5.0,
                prediction_type="flow_prediction",
            ),
            _sigma_to_alpha_sigma_t=lambda sigma: (
                FakeScalar(1.0 - sigma.value),
                sigma,
            ),
        )
        diagnostics = scheduler_noise_diagnostics(scheduler, 0)
        self.assertAlmostEqual(diagnostics["signal_weight"], 0.196)
        self.assertAlmostEqual(diagnostics["noise_weight"], 0.804)
        self.assertTrue(diagnostics["use_flow_sigmas"])
        self.assertEqual(diagnostics["flow_shift"], 5.0)

    def test_validates_reference_scheduler_configuration(self):
        scheduler = type(
            "UniPCMultistepScheduler",
            (),
            {
                "config": SimpleNamespace(
                    _class_name="UniPCMultistepScheduler",
                    flow_shift=5.0,
                    prediction_type="flow_prediction",
                    use_flow_sigmas=True,
                    timestep_spacing="linspace",
                    solver_order=2,
                    solver_type="bh2",
                    num_train_timesteps=1000,
                )
            },
        )()
        info = validate_official_scheduler(scheduler)
        self.assertEqual(info["scheduler_source"], "checkpoint_config")
        scheduler.config.flow_shift = 1.0
        with self.assertRaisesRegex(ValueError, "does not match official"):
            validate_official_scheduler(scheduler)

    def test_add_noise_moves_reference_timestep_to_latent_device(self):
        timestep = MagicMock()
        timestep.ndim = 0
        expanded = timestep.unsqueeze.return_value
        moved = expanded.to.return_value
        latents = SimpleNamespace(device="cuda:0")
        noise = object()
        scheduler = SimpleNamespace(add_noise=MagicMock(return_value="noisy"))
        result = add_noise_at_timestep(scheduler, latents, noise, timestep)
        expanded.to.assert_called_once_with(device="cuda:0")
        scheduler.add_noise.assert_called_once_with(latents, noise, moved)
        self.assertEqual(result, "noisy")

    def test_text_only_latent_contract_rejects_condition_channels(self):
        transformer = SimpleNamespace(config=SimpleNamespace(in_channels=48))
        latent = np.zeros((1, 48, 7, 30, 52), dtype=np.float32)
        contract = validate_text_only_latent_contract(transformer, latent)
        self.assertFalse(contract["uses_first_frame_condition"])
        with self.assertRaisesRegex(ValueError, "latent=96, transformer=48"):
            validate_text_only_latent_contract(
                transformer, np.zeros((1, 96, 7, 30, 52), dtype=np.float32)
            )

    def test_text_only_expanded_timestep_assigns_every_token_current_time(self):
        import torch

        pipe = SimpleNamespace(
            config=SimpleNamespace(expand_timesteps=True),
            transformer=SimpleNamespace(config=SimpleNamespace(patch_size=(1, 2, 2))),
        )
        latents = torch.zeros((1, 48, 7, 4, 6))
        timestep = torch.tensor(638.0)
        expanded = _text_only_timestep(pipe, latents, timestep, torch)
        self.assertEqual(tuple(expanded.shape), (1, 7 * 2 * 3))
        self.assertTrue(torch.all(expanded == timestep))

    def test_encodes_positive_and_negative_prompts_for_cfg(self):
        import torch

        calls = []

        def encode(prompt, **kwargs):
            calls.append(prompt)
            return torch.ones((1, 2, 3))

        pipe = SimpleNamespace(
            _get_t5_prompt_embeds=encode,
            text_encoder=SimpleNamespace(device="cpu", dtype=torch.float32),
        )
        positive, negative = _encode_prompt(
            pipe,
            "paint the car white",
            "black background",
            5.0,
            "cpu",
            torch.float32,
            128,
        )
        self.assertEqual(calls, ["paint the car white", "black background"])
        self.assertEqual(tuple(positive.shape), tuple(negative.shape))

    def test_guidance_one_does_not_encode_negative_prompt(self):
        import torch

        calls = []

        def encode(prompt, **kwargs):
            calls.append(prompt)
            return torch.ones((1, 2, 3))

        pipe = SimpleNamespace(
            _get_t5_prompt_embeds=encode,
            text_encoder=SimpleNamespace(device="cpu", dtype=torch.float32),
        )
        _, negative = _encode_prompt(
            pipe,
            "paint the car white",
            "black background",
            1.0,
            "cpu",
            torch.float32,
            128,
        )
        self.assertEqual(calls, ["paint the car white"])
        self.assertIsNone(negative)

    def test_cfg_runs_two_forwards_and_uses_standard_formula(self):
        import torch

        class Transformer:
            dtype = torch.float32

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                embeds = kwargs["encoder_hidden_states"]
                self.calls.append(kwargs)
                return (embeds.clone(),)

        transformer = Transformer()
        pipe = SimpleNamespace(transformer=transformer)
        conditional = torch.tensor([[[[[3.0, 5.0]]]]])
        unconditional = torch.tensor([[[[[1.0, 2.0]]]]])
        guided, diagnostics = _predict_with_cfg(
            pipe,
            torch.zeros_like(conditional),
            torch.tensor([1.0]),
            conditional,
            unconditional,
            5.0,
            torch,
        )
        expected = unconditional + 5.0 * (conditional - unconditional)
        self.assertTrue(torch.equal(guided, expected))
        self.assertIs(transformer.calls[0]["encoder_hidden_states"], conditional)
        self.assertIs(transformer.calls[1]["encoder_hidden_states"], unconditional)
        self.assertEqual(
            tuple(transformer.calls[0]["hidden_states"].shape),
            tuple(conditional.shape),
        )
        self.assertTrue(diagnostics["cfg_enabled"])
        self.assertEqual(diagnostics["transformer_forward_count"], 2)
        self.assertIsNotNone(diagnostics["unconditional"])

    def test_guidance_one_runs_only_conditional_forward(self):
        import torch

        transformer = MagicMock()
        transformer.dtype = torch.float32
        conditional = torch.tensor([[[[[3.0, 5.0]]]]])
        transformer.return_value = (conditional,)
        pipe = SimpleNamespace(transformer=transformer)
        guided, diagnostics = _predict_with_cfg(
            pipe,
            torch.zeros_like(conditional),
            torch.tensor([1.0]),
            conditional,
            None,
            1.0,
            torch,
        )
        self.assertTrue(torch.equal(guided, conditional))
        self.assertEqual(transformer.call_count, 1)
        self.assertFalse(diagnostics["cfg_enabled"])
        self.assertEqual(diagnostics["transformer_forward_count"], 1)
        self.assertIsNone(diagnostics["unconditional"])

    def test_vae_uses_official_cache_reset(self):
        vae = SimpleNamespace(
            _feat_map=["stale"],
            clear_cache=MagicMock(side_effect=lambda: setattr(vae, "_feat_map", [None])),
        )
        _clear_vae_cache(SimpleNamespace(vae=vae))
        vae.clear_cache.assert_called_once_with()
        self.assertEqual(vae._feat_map, [None])

    def test_loads_checkpoint_with_text_only_expanded_timesteps(self):
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
        self.assertEqual(args.negative_prompt, "")
        self.assertTrue(args.save_denoise_steps)
        self.assertEqual(args.denoise_save_every, 10)

    def test_cli_can_disable_denoise_step_videos(self):
        args = build_parser().parse_args(
            [
                "--video",
                "input.mp4",
                "--object",
                "car",
                "--no-save-denoise-steps",
            ]
        )
        self.assertFalse(args.save_denoise_steps)

    def test_cli_validation_runs_before_model_loading(self):
        args = build_parser().parse_args(
            ["--video", "input.mp4", "--object", "car", "--strength", "0"]
        )
        with self.assertRaisesRegex(ValueError, "--strength"):
            validate_args(args)

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
