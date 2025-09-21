from contextlib import AsyncExitStack
from logging import Logger
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from avalan.entities import EngineUri, TransformerEngineSettings
from avalan.model.modalities.text import TextGenerationModality


@pytest.mark.parametrize(
    "vendor,class_name",
    [
        ("anthropic", "AnthropicModel"),
        ("openai", "OpenAIModel"),
        ("bedrock", "BedrockModel"),
        ("openrouter", "OpenRouterModel"),
        ("anyscale", "AnyScaleModel"),
        ("together", "TogetherModel"),
        ("deepseek", "DeepSeekModel"),
        ("deepinfra", "DeepInfraModel"),
        ("groq", "GroqModel"),
        ("ollama", "OllamaModel"),
        ("huggingface", "HuggingfaceModel"),
        ("hyperbolic", "HyperbolicModel"),
        ("litellm", "LiteLLMModel"),
    ],
)
def test_load_engine_per_vendor(vendor: str, class_name: str) -> None:
    loader = MagicMock()
    stub = types.ModuleType(f"avalan.model.nlp.text.vendor.{vendor}")
    setattr(stub, class_name, loader)
    engine_uri = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=vendor,
        model_id="model",
        params={},
    )
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)
    exit_stack = AsyncExitStack()
    with patch.dict(
        sys.modules, {f"avalan.model.nlp.text.vendor.{vendor}": stub}
    ):
        result = TextGenerationModality().load_engine(
            engine_uri, settings, logger, exit_stack
        )
    loader.assert_called_once_with(
        model_id="model",
        settings=settings,
        logger=logger,
        exit_stack=exit_stack,
    )
    assert result is loader.return_value
