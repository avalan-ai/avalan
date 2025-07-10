from avalan.entities import EngineUri, Modality, TransformerEngineSettings
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager
from logging import Logger
from unittest import TestCase
from unittest.mock import MagicMock, patch


class ModelManagerLoadEngineModalitiesTestCase(TestCase):
    def setUp(self):
        self.hub = MagicMock(spec=HuggingfaceHub)
        self.logger = MagicMock(spec=Logger)

    def test_load_engine_local_modalities(self):
        modalities = {
            Modality.TEXT_GENERATION: "TextGenerationModel",
            Modality.EMBEDDING: "SentenceTransformerModel",
            Modality.AUDIO_SPEECH_RECOGNITION: "SpeechRecognitionModel",
            Modality.AUDIO_TEXT_TO_SPEECH: "TextToSpeechModel",
            Modality.VISION_OBJECT_DETECTION: "ObjectDetectionModel",
            Modality.VISION_IMAGE_CLASSIFICATION: "ImageClassificationModel",
            Modality.VISION_IMAGE_TO_TEXT: "ImageToTextModel",
            Modality.VISION_IMAGE_TEXT_TO_TEXT: "ImageTextToTextModel",
            Modality.VISION_ENCODER_DECODER: "VisionEncoderDecoderModel",
            Modality.VISION_SEMANTIC_SEGMENTATION: "SemanticSegmentationModel",
        }
        for modality, class_name in modalities.items():
            with self.subTest(modality=modality):
                with ModelManager(self.hub, self.logger) as manager:
                    uri = manager.parse_uri(
                        f"ai://local/{modality.name.lower()}"
                    )
                    settings = TransformerEngineSettings()
                    manager._stack.enter_context = MagicMock()
                    path = f"avalan.model.manager.{class_name}"
                    with patch(path) as Model:
                        result = manager.load_engine(uri, settings, modality)
                    Model.assert_called_once_with(
                        model_id=modality.name.lower(),
                        settings=settings,
                        logger=self.logger,
                    )
                    manager._stack.enter_context.assert_called_once_with(
                        Model.return_value
                    )
                    self.assertIs(result, Model.return_value)


class ModelManagerLoadModalitiesTestCase(TestCase):
    def setUp(self):
        self.hub = MagicMock(spec=HuggingfaceHub)
        self.logger = MagicMock(spec=Logger)

    def test_load_delegates_per_modality(self):
        for modality in Modality:
            with self.subTest(modality=modality):
                engine_uri = EngineUri(
                    host=None,
                    port=None,
                    user=None,
                    password=None,
                    vendor=None,
                    model_id="m",
                    params={},
                )
                with ModelManager(self.hub, self.logger) as manager:
                    with (
                        patch.object(
                            manager, "get_engine_settings"
                        ) as get_mock,
                        patch.object(manager, "load_engine") as load_mock,
                    ):
                        get_mock.return_value = TransformerEngineSettings()
                        load_mock.return_value = "model"
                        result = manager.load(engine_uri, modality=modality)
                    load_mock.assert_called_once_with(
                        engine_uri, get_mock.return_value, modality
                    )
                    self.assertEqual(result, "model")
