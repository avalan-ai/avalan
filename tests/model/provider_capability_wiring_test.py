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
from avalan.model.provider import ProviderFamily


def _openai_model(
    base_url: str | None,
    *,
    extra_query: dict[str, str] | None = None,
    model_type: type[OpenAIModel] = OpenAIModel,
) -> OpenAIModel:
    client = object.__new__(OpenAIClient)
    client._base_url = base_url  # noqa: SLF001
    client._is_azure = OpenAIClient._is_azure_base_url(  # noqa: SLF001
        base_url
    )
    client._extra_query = extra_query  # noqa: SLF001
    model = object.__new__(model_type)
    model._model = client  # noqa: SLF001
    model._model_id = "gpt-5"  # noqa: SLF001
    model._continuation_capability_support = None  # noqa: SLF001
    return cast(OpenAIModel, model)


@pytest.mark.parametrize(
    ("base_url", "provider_family"),
    (
        ("https://api.openai.com/v1", ProviderFamily.OPENAI.value),
        ("https://api.openai.com/v1/", ProviderFamily.OPENAI.value),
        (
            "https://tenant.openai.azure.com/openai/v1/",
            ProviderFamily.AZURE_OPENAI.value,
        ),
        (
            "https://tenant.cognitiveservices.azure.com:443/openai/v1",
            ProviderFamily.AZURE_OPENAI.value,
        ),
        ("http://api.openai.com/v1", None),
        ("https://api.openai.com/v1?mode=compatible", None),
        ("https://compatible.example/v1", None),
        (
            "http://tenant.openai.azure.com/openai/v1/",
            None,
        ),
        (
            "https://tenant.openai.azure.com/openai/deployments/example",
            None,
        ),
        (
            "https://tenant.openai.azure.com:8443/openai/v1/",
            None,
        ),
        (
            "https://user@tenant.openai.azure.com/openai/v1/",
            None,
        ),
        (
            "https://tenant.openai.azure.com/openai/v1/?api-version=preview",
            None,
        ),
    ),
)
def test_openai_attached_capability_requires_native_responses_endpoint(
    base_url: str | None,
    provider_family: str | None,
) -> None:
    """Accept only exact native OpenAI and Azure Responses endpoints."""
    support = _openai_model(base_url).provider_capability_support
    capable = provider_family is not None

    assert (
        support.structured_invocation,
        support.stable_call_ids,
        support.correlated_results,
    ) == (capable, capable, capable)
    assert (
        support.task_input_advertisement
        is TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert support.provider_family == provider_family


@pytest.mark.parametrize(
    ("extra_query", "capable"),
    (
        (None, True),
        ({"api-version": "preview"}, True),
        ({"api-version": "v1"}, False),
        ({"api-version": "2025-04-01-preview"}, False),
        ({"other": "preview"}, False),
    ),
)
def test_azure_capability_requires_native_responses_api_version(
    extra_query: dict[str, str] | None,
    capable: bool,
) -> None:
    """Trust only Azure Responses API versions with a frozen contract."""
    model = _openai_model(
        "https://tenant.openai.azure.com/openai/v1/",
        extra_query=extra_query,
    )

    assert model.provider_capability_support.structured_invocation is capable


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


@pytest.mark.parametrize("azure_api_version", (None, "preview"))
def test_loaded_azure_client_advertises_exact_responses_capability(
    azure_api_version: str | None,
) -> None:
    """Inspect a loaded Azure client's effective endpoint and API version."""
    client = OpenAIClient(
        api_key="test-key",
        base_url="https://tenant.openai.azure.com/openai/v1/",
        azure_api_version=azure_api_version,
    )
    model = object.__new__(OpenAIModel)
    model._model = client  # noqa: SLF001
    model._model_id = "deployment"  # noqa: SLF001
    model._continuation_capability_support = None  # noqa: SLF001
    try:
        support = model.provider_capability_support

        assert support.provider_family == ProviderFamily.AZURE_OPENAI.value
        assert support.structured_invocation
        assert support.stable_call_ids
        assert support.correlated_results
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


def test_compatible_client_subclass_never_proves_native_provider() -> None:
    """Reject compatible client subclasses before inspecting endpoints."""

    class CompatibleOpenAIClient(OpenAIClient):
        pass

    client = cast(OpenAIClient, object.__new__(CompatibleOpenAIClient))

    assert client._native_responses_provider_family() is None  # noqa: SLF001
