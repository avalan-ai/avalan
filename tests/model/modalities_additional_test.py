from argparse import Namespace
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.entities import (
    GenerationSettings,
    Modality,
    Operation,
    OperationParameters,
    OperationVisionParameters,
)
from avalan.model.modalities.audio import AudioClassificationModality
from avalan.model.modalities.vision import (
    VisionTextToAnimationModality,
    VisionTextToImageModality,
    VisionTextToVideoModality,
)


class AudioModalityArgumentsTestCase(TestCase):
    def test_audio_classification_operation_arguments(self) -> None:
        modality = AudioClassificationModality()
        args = Namespace(path="input.wav", audio_sampling_rate=44100)
        operation = modality.get_operation_from_arguments(
            args,
            input_string=None,
            settings=GenerationSettings(),
        )
        audio_params = operation.parameters["audio"]
        self.assertEqual(audio_params.path, "input.wav")
        self.assertEqual(audio_params.sampling_rate, 44100)
        self.assertEqual(operation.modality, Modality.AUDIO_CLASSIFICATION)
        self.assertFalse(operation.requires_input)


class VisionModalityArgumentsTestCase(TestCase):
    def test_text_to_image_operation_arguments(self) -> None:
        modality = VisionTextToImageModality()
        args = Namespace(
            path="seed.png",
            vision_color_model="rgb",
            vision_high_noise_frac=0.2,
            vision_image_format="png",
            vision_steps=42,
        )
        operation = modality.get_operation_from_arguments(
            args,
            input_string="a prompt",
            settings=GenerationSettings(),
        )
        vision_params = operation.parameters["vision"]
        self.assertEqual(operation.modality, Modality.VISION_TEXT_TO_IMAGE)
        self.assertTrue(operation.requires_input)
        self.assertEqual(vision_params.path, "seed.png")
        self.assertEqual(vision_params.color_model, "rgb")
        self.assertEqual(vision_params.high_noise_frac, 0.2)
        self.assertEqual(vision_params.image_format, "png")
        self.assertEqual(vision_params.n_steps, 42)

    def test_text_to_animation_operation_arguments(self) -> None:
        modality = VisionTextToAnimationModality()
        args = Namespace(
            path="clip.gif",
            vision_steps=12,
            vision_timestep_spacing="linear",
            vision_beta_schedule="cosine",
            vision_guidance_scale=2.5,
        )
        operation = modality.get_operation_from_arguments(
            args,
            input_string="animate",
            settings=GenerationSettings(),
        )
        params = operation.parameters["vision"]
        self.assertEqual(operation.modality, Modality.VISION_TEXT_TO_ANIMATION)
        self.assertTrue(operation.requires_input)
        self.assertEqual(params.path, "clip.gif")
        self.assertEqual(params.n_steps, 12)
        self.assertEqual(params.timestep_spacing, "linear")
        self.assertEqual(params.beta_schedule, "cosine")
        self.assertEqual(params.guidance_scale, 2.5)


class VisionTextToVideoCallTestCase(IsolatedAsyncioTestCase):
    async def test_text_to_video_kwargs_include_width_and_steps(self) -> None:
        modality = VisionTextToVideoModality()

        class DummyVideoModel:
            def __init__(self) -> None:
                self.called_with: tuple | None = None

            async def __call__(self, *args, **kwargs):
                self.called_with = (args, kwargs)
                return "ok"

        model = DummyVideoModel()
        vision_params = OperationVisionParameters(
            path="movie.mp4",
            reference_path="ref.mp4",
            negative_prompt="none",
            width=640,
            height=480,
            downscale=0.5,
            frames=30,
            denoise_strength=0.2,
            n_steps=8,
            inference_steps=10,
            decode_timestep=0.1,
            noise_scale=0.05,
            frames_per_second=12,
        )
        operation = Operation(
            generation_settings=GenerationSettings(),
            input="render",
            modality=Modality.VISION_TEXT_TO_VIDEO,
            parameters=OperationParameters(vision=vision_params),
            requires_input=True,
        )

        result = await modality(
            engine_uri=None,  # type: ignore[arg-type]
            model=model,
            operation=operation,
            tool=None,
        )

        self.assertEqual(result, "ok")
        self.assertIsNotNone(model.called_with)
        args, kwargs = model.called_with
        self.assertEqual(args, ("render", "movie.mp4"))
        self.assertEqual(kwargs["width"], 640)
        self.assertEqual(kwargs["steps"], 8)
        self.assertEqual(kwargs["reference_path"], "ref.mp4")
        self.assertEqual(kwargs["frames"], 30)
        self.assertEqual(kwargs["frames_per_second"], 12)
