from avalan.entities import TransformerEngineSettings
from avalan.model.vision.diffusion import TextToImageDiffusionModel
from avalan.model.nlp import BaseNLPModel
from avalan.model.engine import Engine
from diffusers import DiffusionPipeline
from contextlib import nullcontext
from logging import Logger
from unittest import TestCase, IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch, call


class TextToImageDiffusionModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"
    refiner_id = "refiner/model"

    def test_missing_refiner(self) -> None:
        with self.assertRaises(AssertionError):
            TextToImageDiffusionModel(
                self.model_id, TransformerEngineSettings()
            )

    def test_instantiation_with_load_model(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                DiffusionPipeline, "from_pretrained"
            ) as pipeline_mock,
            patch.object(
                BaseNLPModel, "_get_weight_type", return_value="dtype"
            ),
            patch.object(Engine, "get_default_device", return_value="cpu"),
        ):
            base_instance = MagicMock()
            base_instance.text_encoder_2 = "te2"
            base_instance.vae = "vae"
            refiner_instance = MagicMock()
            pipeline_mock.side_effect = [base_instance, refiner_instance]

            settings = TransformerEngineSettings(
                refiner_model_id=self.refiner_id
            )
            model = TextToImageDiffusionModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            self.assertIs(model.model, refiner_instance)
            self.assertIs(model._base, base_instance)
            pipeline_mock.assert_has_calls(
                [
                    call(
                        self.model_id,
                        torch_dtype="dtype",
                        variant=settings.weight_type,
                        use_safetensors=True,
                    ),
                    call(
                        self.refiner_id,
                        text_encoder_2=base_instance.text_encoder_2,
                        vae=base_instance.vae,
                        torch_dtype="dtype",
                        use_safetensors=True,
                        variant=settings.weight_type,
                    ),
                ]
            )
            base_instance.to.assert_called_once_with("cpu")
            refiner_instance.to.assert_called_once_with("cpu")


class TextToImageDiffusionModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"
    refiner_id = "refiner/model"

    async def test_call(self) -> None:
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(
                DiffusionPipeline, "from_pretrained"
            ) as pipeline_mock,
            patch.object(
                BaseNLPModel, "_get_weight_type", return_value="dtype"
            ),
            patch.object(Engine, "get_default_device", return_value="cpu"),
            patch(
                "avalan.model.vision.diffusion.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
        ):
            base_instance = MagicMock()
            base_instance.text_encoder_2 = "te2"
            base_instance.vae = "vae"
            base_instance.__call__.return_value = MagicMock(images="latent")
            refiner_image = MagicMock()
            refiner_instance = MagicMock()
            refiner_instance.__call__.return_value = MagicMock(
                images=[refiner_image]
            )
            pipeline_mock.side_effect = [base_instance, refiner_instance]

            settings = TransformerEngineSettings(
                refiner_model_id=self.refiner_id
            )
            model = TextToImageDiffusionModel(
                self.model_id,
                settings,
                logger=logger_mock,
            )

            result = await model(
                "prompt",
                "out.jpg",
                color_model="CMYK",
                image_format="PNG",
                n_steps=10,
            )

            self.assertEqual(result, "out.jpg")
            base_instance.__call__.assert_called_once_with(
                prompt="prompt",
                num_inference_steps=10,
                denoising_end=0.8,
                output_type="latent",
            )
            refiner_instance.__call__.assert_called_once_with(
                prompt="prompt",
                num_inference_steps=10,
                denoising_start=0.8,
                image="latent",
            )
            refiner_image.convert.assert_called_once_with("CMYK")
            refiner_image.save.assert_called_once_with("out.jpg", "PNG")
            inf_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
