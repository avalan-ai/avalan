import unittest
from argparse import Namespace
from unittest.mock import MagicMock

from avalan.entities import Modality
from enum import StrEnum
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager


class FakeModality(StrEnum):
    UNKNOWN = "unknown"


class ModelManagerGetOperationTestCase(unittest.TestCase):
    def setUp(self):
        self.hub = MagicMock(spec=HuggingfaceHub)
        self.logger = MagicMock()
        self.manager = ModelManager(self.hub, self.logger)
        self.args = Namespace(
            path="file",
            audio_sampling_rate=16000,
            audio_reference_path=None,
            audio_reference_text=None,
            text_context="ctx",
            text_from_lang="en",
            text_to_lang="fr",
            stop_on_keyword=None,
            skip_special_tokens=False,
            system=None,
            display_tokens=0,
            image_width=256,
            vision_threshold=0.5,
            do_sample=False,
            enable_gradient_calculation=False,
            max_new_tokens=1,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            text_max_length=None,
            text_num_beams=None,
            quiet=False,
        )

    def _check_audio(self, op):
        self.assertEqual(op.parameters["audio"].path, "file")
        self.assertEqual(op.parameters["audio"].sampling_rate, 16000)

    def _check_text(self, op):
        self.assertIs(op.parameters["text"].system_prompt, None)

    def _check_vision(self, op):
        self.assertEqual(op.parameters["vision"].path, "file")

    def test_all_modalities(self):
        cases = {
            Modality.AUDIO_SPEECH_RECOGNITION: self._check_audio,
            Modality.AUDIO_TEXT_TO_SPEECH: self._check_audio,
            Modality.TEXT_QUESTION_ANSWERING: self._check_text,
            Modality.TEXT_SEQUENCE_CLASSIFICATION: (
                lambda op: self.assertIsNone(op.parameters)
            ),
            Modality.TEXT_SEQUENCE_TO_SEQUENCE: self._check_text,
            Modality.TEXT_TRANSLATION: self._check_text,
            Modality.TEXT_TOKEN_CLASSIFICATION: self._check_text,
            Modality.TEXT_GENERATION: self._check_text,
            Modality.VISION_IMAGE_CLASSIFICATION: self._check_vision,
            Modality.VISION_IMAGE_TO_TEXT: self._check_vision,
            Modality.VISION_IMAGE_TEXT_TO_TEXT: self._check_vision,
            Modality.VISION_ENCODER_DECODER: self._check_vision,
            Modality.VISION_OBJECT_DETECTION: self._check_vision,
            Modality.VISION_SEMANTIC_SEGMENTATION: self._check_vision,
        }
        for modality, checker in cases.items():
            with self.subTest(modality=modality):
                op = self.manager.get_operation_from_arguments(
                    modality, self.args, "i"
                )
                self.assertEqual(op.modality, modality)
                checker(op)

    def test_unknown_modality(self):
        op = self.manager.get_operation_from_arguments(
            FakeModality.UNKNOWN, self.args, None
        )
        self.assertEqual(op.modality, FakeModality.UNKNOWN)
        self.assertIsNone(op.parameters)
