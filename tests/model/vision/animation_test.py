import itertools
from contextlib import nullcontext
from logging import Logger
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, call, patch

import numpy as np
from PIL import Image

from avalan.entities import (
    BetaSchedule,
    EngineSettings,
    TimestepSpacing,
)
from avalan.model.engine import Engine
from avalan.model.vision.diffusion import TextToAnimationModel
from avalan.model.vision.diffusion import animation as animation_module


class DummyDiffusionPipeline:
    def to(self, *args: object, **kwargs: object) -> object:
        return self


class TextToAnimationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_lazy_diffusers_helpers_import_targets(self) -> None:
        class MotionAdapter:
            marker = "motion"

            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

        class EulerDiscreteScheduler:
            marker = "scheduler"

            @classmethod
            def from_config(cls, *args: object, **kwargs: object) -> object:
                return ("scheduler", args, kwargs)

        def export_to_gif(*args: object, **kwargs: object) -> object:
            return ("gif", args, kwargs)

        pipeline = object()
        modules = {
            "diffusers": SimpleNamespace(
                AnimateDiffPipeline=pipeline,
                EulerDiscreteScheduler=EulerDiscreteScheduler,
                MotionAdapter=MotionAdapter,
            ),
            "diffusers.utils": SimpleNamespace(export_to_gif=export_to_gif),
        }

        with patch(
            "avalan.model.vision.diffusion.animation.import_module",
            side_effect=modules.__getitem__,
        ):
            adapter = animation_module.MotionAdapter("repo", dtype="float16")
            self.assertIsInstance(adapter, MotionAdapter)
            self.assertEqual(adapter.args, ("repo",))
            self.assertEqual(adapter.kwargs, {"dtype": "float16"})
            self.assertEqual(
                animation_module.EulerDiscreteScheduler.from_config(
                    "cfg", beta_schedule="scaled_linear"
                ),
                (
                    "scheduler",
                    ("cfg",),
                    {"beta_schedule": "scaled_linear"},
                ),
            )
            self.assertEqual(animation_module.MotionAdapter.marker, "motion")
            self.assertEqual(
                animation_module.export_to_gif(["frame"], "out.gif"),
                ("gif", (["frame"], "out.gif"), {}),
            )
            self.assertIs(animation_module._animate_diff_pipeline(), pipeline)

    def test_instantiation_with_load_model(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        animate_pipeline = MagicMock()
        with (
            patch.object(Engine, "weight", return_value="dtype"),
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch("avalan.model.engine.find_spec", return_value=None),
            patch(
                "avalan.model.engine.DiffusionPipeline",
                DummyDiffusionPipeline,
            ),
            patch(
                "avalan.model.vision.diffusion.animation.MotionAdapter"
            ) as adapter_cls,
            patch(
                "avalan.model.vision.diffusion.animation.hf_hub_download",
                return_value="ckpt_path",
            ) as download_mock,
            patch(
                "avalan.model.vision.diffusion.animation.load_file",
                return_value={"sd": 1},
            ) as load_mock,
            patch(
                "avalan.model.vision.diffusion.animation._animate_diff_pipeline",
                return_value=animate_pipeline,
            ),
        ):
            pipe_mock = animate_pipeline.from_pretrained
            adapter_instance = MagicMock()
            adapter_instance.to.return_value = adapter_instance
            adapter_cls.return_value = adapter_instance

            pipe_instance = MagicMock(spec=DummyDiffusionPipeline)
            pipe_instance.to.return_value = pipe_instance
            pipe_mock.return_value = pipe_instance

            settings = EngineSettings(base_model_id="base", checkpoint="ckpt")
            model = TextToAnimationModel(
                self.model_id, settings, logger=logger_mock
            )

            self.assertIs(model.model, pipe_instance)
            adapter_cls.assert_called_once_with()
            adapter_instance.to.assert_called_once_with("cpu", "dtype")
            download_mock.assert_called_once_with(self.model_id, "ckpt")
            load_mock.assert_called_once_with("ckpt_path", device="cpu")
            adapter_instance.load_state_dict.assert_called_once_with(
                load_mock.return_value
            )
            pipe_mock.assert_called_once_with(
                "base",
                feature_extractor=None,
                image_encoder=None,
                motion_adapter=adapter_instance,
                torch_dtype="dtype",
                use_safetensors=False,
            )
            pipe_instance.to.assert_called_once_with("cpu")


class TextToAnimationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call_all_parameter_combinations(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        animate_pipeline = MagicMock()
        with (
            patch.object(Engine, "weight", return_value="dtype"),
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch("avalan.model.engine.find_spec", return_value=None),
            patch(
                "avalan.model.engine.DiffusionPipeline",
                DummyDiffusionPipeline,
            ),
            patch(
                "avalan.model.vision.diffusion.animation.MotionAdapter"
            ) as adapter_cls,
            patch(
                "avalan.model.vision.diffusion.animation.hf_hub_download",
                return_value="ckpt",
            ),
            patch(
                "avalan.model.vision.diffusion.animation.load_file",
                return_value={},
            ),
            patch(
                "avalan.model.vision.diffusion.animation._animate_diff_pipeline",
                return_value=animate_pipeline,
            ),
            patch(
                "avalan.model.vision.diffusion.animation.EulerDiscreteScheduler.from_config"
            ) as scheduler_mock,
            patch(
                "avalan.model.vision.diffusion.animation.export_to_gif"
            ) as export_mock,
            patch(
                "avalan.model.vision.diffusion.animation.inference_mode",
                return_value=nullcontext(),
            ),
        ):
            pipe_mock = animate_pipeline.from_pretrained
            adapter_instance = MagicMock()
            adapter_instance.to.return_value = adapter_instance
            adapter_cls.return_value = adapter_instance

            pipe_instance = MagicMock(spec=DummyDiffusionPipeline)
            pipe_instance.to.return_value = pipe_instance
            pipe_instance.scheduler = MagicMock(config="cfg")
            output = MagicMock()
            output.frames = np.array(
                [[[[[0.25, 0.5, 0.75]]]]],
                dtype=np.float32,
            )
            pipe_instance.return_value = output
            pipe_mock.return_value = pipe_instance

            scheduler_instance = MagicMock()
            scheduler_mock.return_value = scheduler_instance

            settings = EngineSettings(base_model_id="base", checkpoint="ckpt")
            model = TextToAnimationModel(
                self.model_id, settings, logger=logger_mock
            )

            called = set()
            for beta in BetaSchedule:
                for spacing in TimestepSpacing:
                    for steps in [1, 2, 4, 8]:
                        config_before = pipe_instance.scheduler.config
                        path = f"{steps}-{beta.value}-{spacing.value}.gif"
                        result = await model(
                            "prompt",
                            path,
                            beta_schedule=beta,
                            guidance_scale=1.5,
                            steps=steps,
                            timestep_spacing=spacing,
                        )
                        self.assertEqual(result, path)
                        exported_frames, exported_path = (
                            export_mock.call_args.args
                        )
                        self.assertEqual(exported_path, path)
                        self.assertEqual(len(exported_frames), 1)
                        self.assertIsInstance(exported_frames[0], Image.Image)
                        self.assertEqual(
                            pipe_instance.call_args,
                            call(
                                prompt="prompt",
                                guidance_scale=1.5,
                                num_inference_steps=steps,
                                output_type="np",
                            ),
                        )
                        if (spacing, beta) not in called:
                            scheduler_mock.assert_any_call(
                                config_before,
                                timestep_spacing=spacing,
                                beta_schedule=beta,
                            )
                            called.add((spacing, beta))

            self.assertEqual(
                scheduler_mock.call_count,
                len(list(itertools.product(BetaSchedule, TimestepSpacing))),
            )

    async def test_call_invalid_steps(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        animate_pipeline = MagicMock()
        with (
            patch.object(Engine, "weight", return_value="dtype"),
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch("avalan.model.engine.find_spec", return_value=None),
            patch(
                "avalan.model.engine.DiffusionPipeline",
                DummyDiffusionPipeline,
            ),
            patch(
                "avalan.model.vision.diffusion.animation.MotionAdapter"
            ) as adapter_cls,
            patch(
                "avalan.model.vision.diffusion.animation.hf_hub_download",
                return_value="ckpt",
            ),
            patch(
                "avalan.model.vision.diffusion.animation.load_file",
                return_value={},
            ),
            patch(
                "avalan.model.vision.diffusion.animation._animate_diff_pipeline",
                return_value=animate_pipeline,
            ),
        ):
            pipe_mock = animate_pipeline.from_pretrained
            adapter_instance = MagicMock()
            adapter_instance.to.return_value = adapter_instance
            adapter_cls.return_value = adapter_instance
            pipe_instance = MagicMock(spec=DummyDiffusionPipeline)
            pipe_instance.to.return_value = pipe_instance
            pipe_mock.return_value = pipe_instance

            settings = EngineSettings(base_model_id="base", checkpoint="ckpt")
            model = TextToAnimationModel(
                self.model_id, settings, logger=logger_mock
            )

            with self.assertRaises(AssertionError):
                await model("prompt", "out.gif", steps=3)


class TextToAnimationModelFrameConversionTestCase(TestCase):
    def test_frames_to_images_sanitizes_invalid_rgb_values(self) -> None:
        frames = np.array(
            [[[[np.nan, np.inf, -np.inf], [1.2, 0.4, -0.1]]]],
            dtype=np.float32,
        )

        images = TextToAnimationModel._frames_to_images(frames)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].mode, "RGB")
        self.assertEqual(images[0].size, (2, 1))
        self.assertEqual(
            [images[0].getpixel((0, 0)), images[0].getpixel((1, 0))],
            [(0, 255, 0), (255, 102, 0)],
        )

    def test_frames_to_images_supports_grayscale_frames(self) -> None:
        frames = np.array([[[[0.5], [1.0]]]], dtype=np.float32)

        images = TextToAnimationModel._frames_to_images(frames)

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].mode, "L")
        self.assertEqual(images[0].size, (2, 1))
        self.assertEqual(
            [images[0].getpixel((0, 0)), images[0].getpixel((1, 0))],
            [128, 255],
        )

    def test_frames_to_images_requires_frame_batches(self) -> None:
        with self.assertRaises(AssertionError):
            TextToAnimationModel._frames_to_images(
                np.zeros((1, 1, 3), dtype=np.float32)
            )


class TextToAnimationModelBaseMethodsTestCase(TestCase):
    def test_uses_tokenizer(self) -> None:
        settings = EngineSettings(base_model_id="base", checkpoint="c")
        with (
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch("avalan.model.engine.find_spec", return_value=None),
            patch(
                "avalan.model.engine.DiffusionPipeline",
                DummyDiffusionPipeline,
            ),
            patch.object(
                TextToAnimationModel,
                "_load_model",
                return_value=MagicMock(spec=DummyDiffusionPipeline),
            ),
        ):
            model = TextToAnimationModel("id", settings)
        self.assertFalse(model.uses_tokenizer)

    def test_load_tokenizer_not_supported(self) -> None:
        settings = EngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            base_model_id="base",
            checkpoint="c",
        )
        with (
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch("avalan.model.engine.find_spec", return_value=None),
            patch(
                "avalan.model.engine.DiffusionPipeline",
                DummyDiffusionPipeline,
            ),
            patch.object(
                TextToAnimationModel,
                "_load_model",
                return_value=MagicMock(spec=DummyDiffusionPipeline),
            ),
        ):
            model = TextToAnimationModel("id", settings)
        self.assertFalse(hasattr(model, "_load_tokenizer"))


if __name__ == "__main__":
    main()
