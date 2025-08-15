from avalan.entities import GenerationSettings, TransformerEngineSettings
import torch
from avalan.model.nlp import BaseNLPModel
from avalan.entities import (
    GenerationCacheStrategy,
)
from logging import Logger
from contextlib import nullcontext
from unittest import TestCase
from unittest.mock import MagicMock, patch


class DummyNLPModel(BaseNLPModel):
    def _load_model(self):
        return MagicMock()

    async def __call__(self, *args, **kwargs):
        return None

    def _tokenize_input(self, *args, **kwargs):
        return {}


class BaseNLPModelGenerateTestCase(TestCase):
    def test_generate_output_forced_bos(self):
        model = DummyNLPModel(
            "model",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock(eos_token_id=2)
        settings = GenerationSettings(forced_bos_token_id=1)
        with patch(
            "avalan.model.nlp.inference_mode", return_value=nullcontext()
        ) as inf_mock:
            result = model._generate_output({}, settings)
        inf_mock.assert_called_once_with()
        model._model.generate.assert_called_once()
        args, kwargs = model._model.generate.call_args
        self.assertNotIn("bos_token_id", kwargs)
        self.assertNotIn("eos_token_id", kwargs)
        self.assertIs(result, model._model.generate.return_value)

    def test_generate_output_attention_mask(self):
        model = DummyNLPModel(
            "model",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock(eos_token_id=2)
        mask = torch.tensor([[1, 1]])
        settings = GenerationSettings()
        with patch(
            "avalan.model.nlp.inference_mode", return_value=nullcontext()
        ) as inf_mock:
            model._generate_output({"attention_mask": mask}, settings)
        inf_mock.assert_called_once_with()
        model._model.generate.assert_called_once()
        _, kwargs = model._model.generate.call_args
        self.assertIs(kwargs["attention_mask"], mask)

    def test_generate_output_without_attention_mask(self):
        model = DummyNLPModel(
            "model",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock(eos_token_id=2)
        mask = torch.tensor([[1, 1]])
        settings = GenerationSettings(use_inputs_attention_mask=False)
        with patch(
            "avalan.model.nlp.inference_mode", return_value=nullcontext()
        ) as inf_mock:
            model._generate_output({"attention_mask": mask}, settings)
        inf_mock.assert_called_once_with()
        model._model.generate.assert_called_once()
        _, kwargs = model._model.generate.call_args
        self.assertNotIn("attention_mask", kwargs)

    def test_generate_output_cache_settings(self):
        model = DummyNLPModel(
            "model",
            TransformerEngineSettings(
                auto_load_model=False, auto_load_tokenizer=False
            ),
            logger=MagicMock(spec=Logger),
        )
        model._model = MagicMock()
        model._tokenizer = MagicMock(eos_token_id=2)
        settings = GenerationSettings(
            use_cache=False,
            cache_strategy=GenerationCacheStrategy.MAMBA,
        )
        with patch(
            "avalan.model.nlp.inference_mode", return_value=nullcontext()
        ):
            model._generate_output({}, settings)
        model._model.generate.assert_called_once()
        _, kwargs = model._model.generate.call_args
        self.assertFalse(kwargs["use_cache"])
        self.assertEqual(
            kwargs["cache_implementation"],
            GenerationCacheStrategy.MAMBA,
        )
