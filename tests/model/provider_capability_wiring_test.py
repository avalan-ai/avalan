"""Test production engine wiring for structured task-input capability."""

from asyncio import run as asyncio_run
from typing import cast

import pytest

from avalan.model.capability import (
    ProviderCapabilitySupport,
    TaskInputCapabilityAdvertisement,
)
from avalan.model.engine import Engine
from avalan.model.nlp.text.ds4 import Ds4Model
from avalan.model.nlp.text.mlxlm import MlxLmModel
from avalan.model.nlp.text.vendor.anthropic import AnthropicModel
from avalan.model.nlp.text.vendor.anyscale import AnyScaleModel
from avalan.model.nlp.text.vendor.deepinfra import DeepInfraModel
from avalan.model.nlp.text.vendor.deepseek import DeepSeekModel
from avalan.model.nlp.text.vendor.google import GoogleModel
from avalan.model.nlp.text.vendor.groq import GroqModel
from avalan.model.nlp.text.vendor.hyperbolic import HyperbolicModel
from avalan.model.nlp.text.vendor.litellm import LiteLLMModel
from avalan.model.nlp.text.vendor.openai import OpenAIClient, OpenAIModel
from avalan.model.nlp.text.vendor.openrouter import OpenRouterModel
from avalan.model.nlp.text.vendor.together import TogetherModel
from avalan.model.nlp.text.vllm import VllmModel


def _openai_model(
    base_url: str | None,
    *,
    model_type: type[OpenAIModel] = OpenAIModel,
) -> OpenAIModel:
    client = object.__new__(OpenAIClient)
    client._base_url = base_url  # noqa: SLF001
    model = object.__new__(model_type)
    model._model = client  # noqa: SLF001
    model._model_id = "gpt-5"  # noqa: SLF001
    model._continuation_capability_support = None  # noqa: SLF001
    return cast(OpenAIModel, model)


@pytest.mark.parametrize(
    ("base_url", "capable"),
    (
        ("https://api.openai.com/v1", True),
        ("https://api.openai.com/v1/", True),
        ("http://api.openai.com/v1", False),
        ("https://api.openai.com/v1?mode=compatible", False),
        ("https://compatible.example/v1", False),
        (
            "https://tenant.openai.azure.com/openai/v1/",
            False,
        ),
    ),
)
def test_openai_attached_capability_requires_native_responses_endpoint(
    base_url: str | None,
    capable: bool,
) -> None:
    """Accept only the exact native OpenAI Responses engine and endpoint."""
    support = _openai_model(base_url).provider_capability_support

    assert (
        support.structured_invocation,
        support.stable_call_ids,
        support.correlated_results,
    ) == (capable, capable, capable)
    assert (
        support.task_input_advertisement
        is TaskInputCapabilityAdvertisement.INCAPABLE
    )


@pytest.mark.parametrize(
    ("base_url", "capable"),
    (
        (None, True),
        ("https://api.openai.com/v1", True),
        ("https://compatible.example/v1", False),
        (
            "https://tenant.openai.azure.com/openai/v1/",
            False,
        ),
    ),
)
def test_openai_sdk_effective_endpoint_controls_capability(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str | None,
    capable: bool,
) -> None:
    """Inspect the SDK-resolved endpoint used by a loaded native client."""
    if base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    client = OpenAIClient(api_key="test-key", base_url=None)
    model = object.__new__(OpenAIModel)
    model._model = client  # noqa: SLF001
    model._model_id = "gpt-5"  # noqa: SLF001
    model._continuation_capability_support = None  # noqa: SLF001
    try:
        support = model.provider_capability_support
        assert (
            support.structured_invocation,
            support.stable_call_ids,
            support.correlated_results,
        ) == (capable, capable, capable)
    finally:
        asyncio_run(client.aclose())


@pytest.mark.parametrize(
    "model_type",
    (
        AnthropicModel,
        GoogleModel,
        LiteLLMModel,
        AnyScaleModel,
        DeepInfraModel,
        DeepSeekModel,
        GroqModel,
        HyperbolicModel,
        OpenRouterModel,
        TogetherModel,
        VllmModel,
        MlxLmModel,
        Ds4Model,
    ),
)
def test_unproved_hosted_and_local_engines_inherit_incapable_default(
    model_type: type[Engine],
) -> None:
    """Keep schema projection support distinct from production proof."""
    model = cast(Engine, object.__new__(model_type))

    assert model.provider_capability_support == ProviderCapabilitySupport()


def test_compatible_subclass_stays_incapable_on_native_endpoint() -> None:
    """Reject a compatible adapter even when it points at api.openai.com."""
    model = _openai_model(
        "https://api.openai.com/v1",
        model_type=OpenRouterModel,
    )

    assert model.provider_capability_support == ProviderCapabilitySupport()
