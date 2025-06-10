import unittest
from argparse import Namespace
from unittest.mock import MagicMock

from avalan.cli.commands import get_model_settings


class GetModelSettingsTestCase(unittest.TestCase):
    def test_basic(self):
        engine_uri = MagicMock()
        args = Namespace(
            attention="flash",
            base_url="http://localhost:9001/v1",
            device="cpu",
            disable_loading_progress_bar=True,
            sentence_transformer=True,
            loader_class="auto",
            low_cpu_mem_usage=True,
            quiet=False,
            revision="rev",
            special_token=["<s>"],
            tokenizer="tok",
            token=["t"],
            trust_remote_code=True,
            weight_type="fp16",
        )

        result = get_model_settings(args, MagicMock(), MagicMock(), engine_uri)
        expected = {
            "engine_uri": engine_uri,
            "attention": "flash",
            "base_url": "http://localhost:9001/v1",
            "device": "cpu",
            "disable_loading_progress_bar": True,
            "is_sentence_transformer": True,
            "loader_class": "auto",
            "low_cpu_mem_usage": True,
            "quiet": False,
            "revision": "rev",
            "special_tokens": ["<s>"],
            "tokenizer": "tok",
            "tokens": ["t"],
            "trust_remote_code": True,
            "weight_type": "fp16",
        }
        self.assertEqual(result, expected)

    def test_override_sentence_transformer(self):
        engine_uri = MagicMock()
        args = Namespace(
            device="cpu",
            disable_loading_progress_bar=False,
            loader_class="auto",
            low_cpu_mem_usage=False,
            quiet=True,
            revision="main",
            special_token=None,
            tokenizer=None,
            token=None,
            weight_type="fp32",
        )

        result = get_model_settings(
            args,
            MagicMock(),
            MagicMock(),
            engine_uri,
            is_sentence_transformer=True,
        )
        self.assertTrue(result["is_sentence_transformer"])
