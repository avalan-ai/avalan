import unittest
from unittest.mock import MagicMock

from avalan.agent import Specification
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Modality,
    Operation,
    OperationAudioParameters,
    OperationParameters,
    OperationTextParameters,
    OperationVisionParameters,
)
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager
from avalan.model.task import ModelTask, ModelTaskContext


class DummyModel:
    def __init__(self):
        self.tokenizer = MagicMock()
        self.called_with = None

    async def __call__(self, *args, **kwargs):
        self.called_with = (args, kwargs)
        return "ok"


class ModelManagerCallModalitiesTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hub = MagicMock(spec=HuggingfaceHub)
        self.logger = MagicMock()
        self.manager = ModelManager(self.hub, self.logger)
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.settings = GenerationSettings(max_new_tokens=1)

    async def test_call_supported_modalities(self):
        model = DummyModel()
        cases = [
            (
                Modality.AUDIO_CLASSIFICATION,
                Operation(
                    generation_settings=self.settings,
                    input=None,
                    modality=Modality.AUDIO_CLASSIFICATION,
                    parameters=OperationParameters(
                        audio=OperationAudioParameters(
                            path="a.wav",
                            sampling_rate=16000,
                        )
                    ),
                ),
                ((), {"path": "a.wav", "sampling_rate": 16000}),
            ),
            (
                Modality.AUDIO_SPEECH_RECOGNITION,
                Operation(
                    generation_settings=self.settings,
                    input=None,
                    modality=Modality.AUDIO_SPEECH_RECOGNITION,
                    parameters=OperationParameters(
                        audio=OperationAudioParameters(
                            path="a.wav",
                            sampling_rate=16000,
                        )
                    ),
                ),
                ((), {"path": "a.wav", "sampling_rate": 16000}),
            ),
            (
                Modality.AUDIO_TEXT_TO_SPEECH,
                Operation(
                    generation_settings=self.settings,
                    input="hi",
                    modality=Modality.AUDIO_TEXT_TO_SPEECH,
                    parameters=OperationParameters(
                        audio=OperationAudioParameters(
                            path="a.wav",
                            sampling_rate=16000,
                            reference_path=None,
                            reference_text=None,
                        )
                    ),
                ),
                (
                    (),
                    {
                        "path": "a.wav",
                        "prompt": "hi",
                        "max_new_tokens": 1,
                        "reference_path": None,
                        "reference_text": None,
                        "sampling_rate": 16000,
                    },
                ),
            ),
            (
                Modality.AUDIO_GENERATION,
                Operation(
                    generation_settings=self.settings,
                    input="song",
                    modality=Modality.AUDIO_GENERATION,
                    parameters=OperationParameters(
                        audio=OperationAudioParameters(
                            path="out.wav",
                            sampling_rate=16000,
                        )
                    ),
                ),
                (("song", "out.wav", 1), {}),
            ),
            (
                Modality.TEXT_GENERATION,
                Operation(
                    generation_settings=self.settings,
                    input="question",
                    modality=Modality.TEXT_GENERATION,
                    parameters=OperationParameters(
                        text=OperationTextParameters(
                            manual_sampling=False,
                            pick_tokens=0,
                            skip_special_tokens=False,
                            system_prompt=None,
                            developer_prompt=None,
                        )
                    ),
                ),
                (
                    ("question",),
                    {
                        "system_prompt": None,
                        "developer_prompt": None,
                        "settings": self.settings,
                        "stopping_criterias": None,
                        "manual_sampling": False,
                        "pick": 0,
                        "skip_special_tokens": False,
                        "tool": None,
                    },
                ),
            ),
            (
                Modality.TEXT_QUESTION_ANSWERING,
                Operation(
                    generation_settings=self.settings,
                    input="q",
                    modality=Modality.TEXT_QUESTION_ANSWERING,
                    parameters=OperationParameters(
                        text=OperationTextParameters(
                            context="ctx",
                            system_prompt=None,
                            developer_prompt=None,
                        )
                    ),
                ),
                (
                    ("q",),
                    {
                        "context": "ctx",
                        "system_prompt": None,
                        "developer_prompt": None,
                    },
                ),
            ),
            (
                Modality.TEXT_SEQUENCE_CLASSIFICATION,
                Operation(
                    generation_settings=self.settings,
                    input="txt",
                    modality=Modality.TEXT_SEQUENCE_CLASSIFICATION,
                    parameters=None,
                ),
                (("txt",), {}),
            ),
            (
                Modality.TEXT_SEQUENCE_TO_SEQUENCE,
                Operation(
                    generation_settings=self.settings,
                    input="in",
                    modality=Modality.TEXT_SEQUENCE_TO_SEQUENCE,
                    parameters=OperationParameters(
                        text=OperationTextParameters(stop_on_keywords=None)
                    ),
                ),
                (
                    ("in",),
                    {"settings": self.settings, "stopping_criterias": None},
                ),
            ),
            (
                Modality.TEXT_TOKEN_CLASSIFICATION,
                Operation(
                    generation_settings=self.settings,
                    input="tok",
                    modality=Modality.TEXT_TOKEN_CLASSIFICATION,
                    parameters=OperationParameters(
                        text=OperationTextParameters(
                            system_prompt=None,
                            developer_prompt=None,
                        )
                    ),
                ),
                (
                    ("tok",),
                    {
                        "labeled_only": False,
                        "system_prompt": None,
                        "developer_prompt": None,
                    },
                ),
            ),
            (
                Modality.TEXT_TRANSLATION,
                Operation(
                    generation_settings=self.settings,
                    input="t",
                    modality=Modality.TEXT_TRANSLATION,
                    parameters=OperationParameters(
                        text=OperationTextParameters(
                            language_source="en",
                            language_destination="fr",
                            stop_on_keywords=None,
                            skip_special_tokens=False,
                        )
                    ),
                ),
                (
                    ("t",),
                    {
                        "source_language": "en",
                        "destination_language": "fr",
                        "settings": self.settings,
                        "stopping_criterias": None,
                        "skip_special_tokens": False,
                    },
                ),
            ),
            (
                Modality.VISION_IMAGE_CLASSIFICATION,
                Operation(
                    generation_settings=self.settings,
                    input=None,
                    modality=Modality.VISION_IMAGE_CLASSIFICATION,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(path="img.png")
                    ),
                ),
                (("img.png",), {}),
            ),
            (
                Modality.VISION_IMAGE_TO_TEXT,
                Operation(
                    generation_settings=self.settings,
                    input=None,
                    modality=Modality.VISION_IMAGE_TO_TEXT,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(
                            path="img.png",
                            skip_special_tokens=False,
                        )
                    ),
                ),
                (("img.png",), {"skip_special_tokens": False}),
            ),
            (
                Modality.VISION_ENCODER_DECODER,
                Operation(
                    generation_settings=self.settings,
                    input=None,
                    modality=Modality.VISION_ENCODER_DECODER,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(
                            path="img.png",
                            skip_special_tokens=True,
                        )
                    ),
                ),
                (("img.png",), {"prompt": None, "skip_special_tokens": True}),
            ),
            (
                Modality.VISION_IMAGE_TEXT_TO_TEXT,
                Operation(
                    generation_settings=self.settings,
                    input="txt",
                    modality=Modality.VISION_IMAGE_TEXT_TO_TEXT,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(
                            path="img.png",
                            system_prompt=None,
                            developer_prompt=None,
                            width=256,
                        )
                    ),
                ),
                (
                    ("img.png", "txt"),
                    {
                        "system_prompt": None,
                        "developer_prompt": None,
                        "settings": self.settings,
                        "width": 256,
                    },
                ),
            ),
            (
                Modality.VISION_OBJECT_DETECTION,
                Operation(
                    generation_settings=self.settings,
                    input=None,
                    modality=Modality.VISION_OBJECT_DETECTION,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(
                            path="img.png",
                            threshold=0.5,
                        )
                    ),
                ),
                (("img.png",), {"threshold": 0.5}),
            ),
            (
                Modality.VISION_TEXT_TO_IMAGE,
                Operation(
                    generation_settings=self.settings,
                    input="txt",
                    modality=Modality.VISION_TEXT_TO_IMAGE,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(
                            path="out.png",
                            color_model="RGB",
                            high_noise_frac=0.9,
                            image_format="PNG",
                            n_steps=10,
                        )
                    ),
                ),
                (
                    ("txt", "out.png"),
                    {
                        "color_model": "RGB",
                        "high_noise_frac": 0.9,
                        "image_format": "PNG",
                        "n_steps": 10,
                    },
                ),
            ),
            (
                Modality.VISION_TEXT_TO_ANIMATION,
                Operation(
                    generation_settings=self.settings,
                    input="txt",
                    modality=Modality.VISION_TEXT_TO_ANIMATION,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(
                            path="out.gif",
                            n_steps=4,
                            timestep_spacing="trailing",
                            beta_schedule="linear",
                            guidance_scale=1.0,
                        )
                    ),
                ),
                (
                    ("txt", "out.gif"),
                    {
                        "beta_schedule": "linear",
                        "guidance_scale": 1.0,
                        "steps": 4,
                        "timestep_spacing": "trailing",
                    },
                ),
            ),
            (
                Modality.VISION_TEXT_TO_VIDEO,
                Operation(
                    generation_settings=self.settings,
                    input="txt",
                    modality=Modality.VISION_TEXT_TO_VIDEO,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(
                            path="video.mp4",
                            reference_path=None,
                            negative_prompt=None,
                            height=None,
                            downscale=2 / 3,
                            frames=96,
                            denoise_strength=0.4,
                            inference_steps=10,
                            decode_timestep=0.05,
                            noise_scale=0.025,
                            frames_per_second=24,
                        )
                    ),
                ),
                (
                    ("txt", "video.mp4"),
                    {
                        "reference_path": None,
                        "negative_prompt": None,
                        "height": None,
                        "downscale": 2 / 3,
                        "frames": 96,
                        "denoise_strength": 0.4,
                        "inference_steps": 10,
                        "decode_timestep": 0.05,
                        "noise_scale": 0.025,
                        "frames_per_second": 24,
                    },
                ),
            ),
            (
                Modality.VISION_TEXT_TO_VIDEO,
                Operation(
                    generation_settings=self.settings,
                    input="txt",
                    modality=Modality.VISION_TEXT_TO_VIDEO,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(
                            path="video.mp4",
                            reference_path=None,
                            negative_prompt=None,
                            height=None,
                            downscale=2 / 3,
                            frames=96,
                            denoise_strength=0.4,
                            n_steps=5,
                            inference_steps=10,
                            decode_timestep=0.05,
                            noise_scale=0.025,
                            frames_per_second=24,
                        )
                    ),
                ),
                (
                    ("txt", "video.mp4"),
                    {
                        "reference_path": None,
                        "negative_prompt": None,
                        "height": None,
                        "downscale": 2 / 3,
                        "frames": 96,
                        "denoise_strength": 0.4,
                        "steps": 5,
                        "inference_steps": 10,
                        "decode_timestep": 0.05,
                        "noise_scale": 0.025,
                        "frames_per_second": 24,
                    },
                ),
            ),
            (
                Modality.VISION_SEMANTIC_SEGMENTATION,
                Operation(
                    generation_settings=self.settings,
                    input=None,
                    modality=Modality.VISION_SEMANTIC_SEGMENTATION,
                    parameters=OperationParameters(
                        vision=OperationVisionParameters(path="img.png")
                    ),
                ),
                (("img.png",), {}),
            ),
        ]

        for modality, operation, expected in cases:
            with self.subTest(modality=modality):
                model.called_with = None
                task = ModelTask(
                    engine_uri=self.engine_uri,
                    model=model,
                    operation=operation,
                    tool=None,
                    context=ModelTaskContext(
                        specification=Specification(role=None, goal=None),
                        input=operation.input,
                        engine_args={},
                    ),
                )
                result = await self.manager(task)
                self.assertEqual(result, "ok")
                self.assertEqual(model.called_with, expected)

    async def test_call_unsupported_modality(self):
        model = DummyModel()
        operation = Operation(
            generation_settings=None,
            input=None,
            modality=Modality.EMBEDDING,
            parameters=OperationParameters(),
        )
        with self.assertRaises(NotImplementedError):
            await self.manager(
                ModelTask(
                    engine_uri=self.engine_uri,
                    model=model,
                    operation=operation,
                    tool=None,
                    context=ModelTaskContext(
                        specification=Specification(role=None, goal=None),
                        input=None,
                        engine_args={},
                    ),
                )
            )
