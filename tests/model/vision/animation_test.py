import itertools
from contextlib import nullcontext
from logging import Logger
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, patch, call

from avalan.entities import (
    BetaSchedule,
    TimestepSpacing,
    TransformerEngineSettings,
)
from avalan.model.engine import Engine
from avalan.model.nlp import BaseNLPModel
from avalan.model.vision.animation import TextToAnimationModel
from diffusers import AnimateDiffPipeline, DiffusionPipeline


class TextToAnimationModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_with_load_model(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                BaseNLPModel, "_get_weight_type", return_value="dtype"
            ),
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch(
                "avalan.model.vision.animation.MotionAdapter"
            ) as adapter_cls,
            patch(
                "avalan.model.vision.animation.hf_hub_download",
                return_value="ckpt_path",
            ) as download_mock,
            patch(
                "avalan.model.vision.animation.load_file",
                return_value={"sd": 1},
            ) as load_mock,
            patch.object(AnimateDiffPipeline, "from_pretrained") as pipe_mock,
        ):
            adapter_instance = MagicMock()
            adapter_instance.to.return_value = adapter_instance
            adapter_cls.return_value = adapter_instance

            pipe_instance = MagicMock(spec=DiffusionPipeline)
            pipe_instance.to.return_value = pipe_instance
            pipe_mock.return_value = pipe_instance

            settings = TransformerEngineSettings(
                base_model_id="base", checkpoint="ckpt"
            )
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
                "base", motion_adapter=adapter_instance, torch_dtype="dtype"
            )
            pipe_instance.to.assert_called_once_with("cpu")


class TextToAnimationModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call_all_parameter_combinations(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                BaseNLPModel, "_get_weight_type", return_value="dtype"
            ),
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch(
                "avalan.model.vision.animation.MotionAdapter"
            ) as adapter_cls,
            patch(
                "avalan.model.vision.animation.hf_hub_download",
                return_value="ckpt",
            ),
            patch("avalan.model.vision.animation.load_file", return_value={}),
            patch.object(AnimateDiffPipeline, "from_pretrained") as pipe_mock,
            patch(
                "avalan.model.vision.animation.EulerDiscreteScheduler.from_config"
            ) as scheduler_mock,
            patch(
                "avalan.model.vision.animation.export_to_gif"
            ) as export_mock,
            patch(
                "avalan.model.vision.animation.inference_mode",
                return_value=nullcontext(),
            ),
        ):
            adapter_instance = MagicMock()
            adapter_instance.to.return_value = adapter_instance
            adapter_cls.return_value = adapter_instance

            pipe_instance = MagicMock(spec=DiffusionPipeline)
            pipe_instance.to.return_value = pipe_instance
            pipe_instance.scheduler = MagicMock(config="cfg")
            output = MagicMock()
            output.frames = [["frame"]]
            pipe_instance.return_value = output
            pipe_mock.return_value = pipe_instance

            scheduler_instance = MagicMock()
            scheduler_mock.return_value = scheduler_instance

            settings = TransformerEngineSettings(
                base_model_id="base", checkpoint="ckpt"
            )
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
                        export_mock.assert_called_with(output.frames[0], path)
                        self.assertEqual(
                            pipe_instance.call_args,
                            call(
                                prompt="prompt",
                                guidance_scale=1.5,
                                num_inference_steps=steps,
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
        with (
            patch.object(
                BaseNLPModel, "_get_weight_type", return_value="dtype"
            ),
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch(
                "avalan.model.vision.animation.MotionAdapter"
            ) as adapter_cls,
            patch(
                "avalan.model.vision.animation.hf_hub_download",
                return_value="ckpt",
            ),
            patch("avalan.model.vision.animation.load_file", return_value={}),
            patch.object(AnimateDiffPipeline, "from_pretrained") as pipe_mock,
        ):
            adapter_instance = MagicMock()
            adapter_instance.to.return_value = adapter_instance
            adapter_cls.return_value = adapter_instance
            pipe_instance = MagicMock(spec=DiffusionPipeline)
            pipe_instance.to.return_value = pipe_instance
            pipe_mock.return_value = pipe_instance

            settings = TransformerEngineSettings(
                base_model_id="base", checkpoint="ckpt"
            )
            model = TextToAnimationModel(
                self.model_id, settings, logger=logger_mock
            )

            with self.assertRaises(AssertionError):
                await model("prompt", "out.gif", steps=3)


class TextToAnimationModelBaseMethodsTestCase(TestCase):
    def test_uses_tokenizer(self) -> None:
        settings = TransformerEngineSettings(
            base_model_id="base", checkpoint="c"
        )
        with (
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch.object(
                TextToAnimationModel,
                "_load_model",
                return_value=MagicMock(spec=DiffusionPipeline),
            ),
        ):
            model = TextToAnimationModel("id", settings)
        self.assertFalse(model.uses_tokenizer)

    def test_load_tokenizer_not_implemented(self) -> None:
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            base_model_id="base",
            checkpoint="c",
        )
        with (
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch.object(
                TextToAnimationModel,
                "_load_model",
                return_value=MagicMock(spec=DiffusionPipeline),
            ),
        ):
            model = TextToAnimationModel("id", settings)
        with self.assertRaises(NotImplementedError):
            model._load_tokenizer(None, True)

    def test_tokenize_input_not_implemented(self) -> None:
        settings = TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
            base_model_id="base",
            checkpoint="c",
        )
        with (
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch.object(
                TextToAnimationModel,
                "_load_model",
                return_value=MagicMock(spec=DiffusionPipeline),
            ),
        ):
            model = TextToAnimationModel("id", settings)
        with self.assertRaises(NotImplementedError):
            model._tokenize_input("in")


if __name__ == "__main__":
    main()
