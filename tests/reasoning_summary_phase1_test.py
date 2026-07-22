"""Test Phase 1 reasoning-summary settings and request capabilities."""

from asyncio import run
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from importlib import import_module
from json import dumps
from logging import getLogger
from subprocess import run as run_process
from sys import executable, modules
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from avalan.agent.engine import EngineAgent
from avalan.entities import (
    ChatSettings,
    EngineUri,
    GenerationSettings,
    Modality,
    Operation,
    OperationParameters,
    OperationTextParameters,
    ReasoningSettings,
    ReasoningSummaryMode,
)
from avalan.model.modalities.text import TextGenerationModality
from avalan.model.nlp.text.ds4 import Ds4Model
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.nlp.text.mlxlm import MlxLmModel
from avalan.model.nlp.text.vendor import TextGenerationVendorModel
from avalan.model.nlp.text.vendor.anyscale import AnyScaleClient, AnyScaleModel
from avalan.model.nlp.text.vendor.bedrock import BedrockClient, BedrockModel
from avalan.model.nlp.text.vendor.deepinfra import (
    DeepInfraClient,
    DeepInfraModel,
)
from avalan.model.nlp.text.vendor.deepseek import DeepSeekClient, DeepSeekModel
from avalan.model.nlp.text.vendor.google import GoogleClient, GoogleModel
from avalan.model.nlp.text.vendor.groq import GroqClient, GroqModel
from avalan.model.nlp.text.vendor.huggingface import (
    HuggingfaceClient,
    HuggingfaceModel,
)
from avalan.model.nlp.text.vendor.hyperbolic import (
    HyperbolicClient,
    HyperbolicModel,
)
from avalan.model.nlp.text.vendor.ollama import OllamaClient, OllamaModel
from avalan.model.nlp.text.vendor.openai import OpenAIClient, OpenAIModel
from avalan.model.nlp.text.vendor.openrouter import (
    OpenRouterClient,
    OpenRouterModel,
)
from avalan.model.nlp.text.vendor.together import TogetherClient, TogetherModel
from avalan.model.nlp.text.vllm import VllmModel
from avalan.model.reasoning import (
    ReasoningSummaryCapabilityError,
    ReasoningSummaryRequestCapability,
    validate_reasoning_summary_request,
)
from avalan.model.stream import StreamRetentionPolicy
from avalan.server.entities import ResponsesRequest


def _load_litellm_vendor_module() -> ModuleType:
    module_name = "avalan.model.nlp.text.vendor.litellm"
    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != "litellm":
            raise

    stub = ModuleType("litellm")
    setattr(stub, "acompletion", AsyncMock())
    previous_sdk = modules.get("litellm")
    modules["litellm"] = stub
    try:
        return import_module(module_name)
    finally:
        if previous_sdk is None:
            modules.pop("litellm", None)
        else:
            modules["litellm"] = previous_sdk


_LITELLM_VENDOR_MODULE = _load_litellm_vendor_module()
LiteLLMClient = cast(Any, getattr(_LITELLM_VENDOR_MODULE, "LiteLLMClient"))
LiteLLMModel = cast(Any, getattr(_LITELLM_VENDOR_MODULE, "LiteLLMModel"))


def _initialize_openai_client_state(client: OpenAIClient) -> None:
    cast(Any, client)._extra_query = None
    cast(Any, client)._stream_response_failed_retries = 24
    cast(Any, client)._stream_response_failed_retry_delay_seconds = 1.0
    cast(Any, client)._stream_retention_policy = StreamRetentionPolicy()
    cast(Any, client)._replay_owners_by_call_id = {}
    cast(Any, client)._active_replay_owners = {}
    cast(Any, client)._active_replay_streams = {}
    cast(Any, client)._active_replay_call_ids = {}
    cast(Any, client)._ambiguous_replay_call_ids = {}
    cast(Any, client)._replay_association_poisoned = False
    cast(Any, client)._closed = False


def _summary_settings(
    mode: ReasoningSummaryMode = ReasoningSummaryMode.CONCISE,
) -> GenerationSettings:
    return GenerationSettings(reasoning=ReasoningSettings(summary=mode))


def _mapping_contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _mapping_contains_key(item, key) for item in value.values()
        )
    if isinstance(value, list | tuple):
        return any(_mapping_contains_key(item, key) for item in value)
    return False


@contextmanager
def _anthropic_module() -> Iterator[Any]:
    module_name = "avalan.model.nlp.text.vendor.anthropic"
    package = import_module("avalan.model.nlp.text.vendor")
    missing = object()
    previous_module = modules.get(module_name)
    previous_attribute = getattr(package, "anthropic", missing)
    module = import_module(module_name)
    try:
        yield module
    finally:
        if previous_module is None:
            modules.pop(module_name, None)
        else:
            modules[module_name] = previous_module
        if previous_attribute is missing:
            if hasattr(package, "anthropic"):
                delattr(package, "anthropic")
        else:
            setattr(package, "anthropic", previous_attribute)


@contextmanager
def _anthropic_model_type() -> Iterator[type[TextGenerationModel]]:
    with _anthropic_module() as module:
        yield cast(
            type[TextGenerationModel],
            getattr(module, "AnthropicModel"),
        )


def test_reasoning_summary_mode_and_settings_contract() -> None:
    assert tuple(mode.value for mode in ReasoningSummaryMode) == (
        "auto",
        "concise",
        "detailed",
    )
    assert ReasoningSettings().summary is None

    for mode in ReasoningSummaryMode:
        settings = ReasoningSettings(summary=mode)
        serialized = asdict(settings)
        assert serialized["summary"] is mode
        assert f'"summary": "{mode.value}"' in dumps(serialized)

    for invalid_summary in (
        "auto",
        "unknown",
        1,
        True,
        {"mode": "auto"},
    ):
        with pytest.raises(AssertionError, match="ReasoningSummaryMode"):
            ReasoningSettings(summary=cast(Any, invalid_summary))

    with pytest.raises(AssertionError, match="disabled"):
        ReasoningSettings(
            summary=ReasoningSummaryMode.AUTO,
            enabled=False,
        )
    with pytest.raises(TypeError):
        ReasoningSettings(unexpected=True)  # type: ignore[call-arg]


def test_engine_normalizes_summary_once_and_preserves_typed_values() -> None:
    for mode in ReasoningSummaryMode:
        normalized = EngineAgent._normalize_generation_settings(
            {"reasoning": {"summary": mode.value}}
        )
        reasoning = normalized["reasoning"]
        assert isinstance(reasoning, ReasoningSettings)
        assert reasoning.summary is mode

        already_typed = EngineAgent._normalize_generation_settings(
            {"reasoning": {"summary": mode}}
        )
        typed_reasoning = already_typed["reasoning"]
        assert isinstance(typed_reasoning, ReasoningSettings)
        assert typed_reasoning.summary is mode

        with patch(
            "avalan.agent.engine.ReasoningSummaryMode",
            wraps=ReasoningSummaryMode,
        ) as summary_constructor:
            EngineAgent._normalize_generation_settings(
                {"reasoning": {"summary": mode}}
            )
        summary_constructor.assert_not_called()

    chat = ChatSettings(enable_thinking=False)
    assert (
        EngineAgent._normalize_generation_settings({"chat_settings": chat})[
            "chat_settings"
        ]
        is chat
    )
    with pytest.raises(ValueError):
        EngineAgent._normalize_generation_settings(
            {"reasoning": {"summary": "verbose"}}
        )


def test_unsupported_direct_model_entries_reject_before_provider_call() -> (
    None
):
    with _anthropic_model_type() as anthropic_model:
        providers = (
            (anthropic_model, "anthropic"),
            (BedrockModel, "bedrock"),
            (LiteLLMModel, "litellm"),
            (GoogleModel, "google"),
            (AnyScaleModel, "anyscale"),
            (DeepInfraModel, "deepinfra"),
            (DeepSeekModel, "deepseek"),
            (GroqModel, "groq"),
            (HyperbolicModel, "hyperbolic"),
            (OpenRouterModel, "openrouter"),
            (TogetherModel, "together"),
            (HuggingfaceModel, "huggingface"),
            (OllamaModel, "ollama"),
            (TextGenerationModel, "transformers"),
            (Ds4Model, "ds4"),
            (VllmModel, "vllm"),
            (MlxLmModel, "mlx"),
        )
        for model_type, provider in providers:
            model = object.__new__(model_type)
            provider_call = AsyncMock()
            model._model = provider_call

            with pytest.raises(ReasoningSummaryCapabilityError) as error:
                run(model("hello", settings=_summary_settings()))

            assert error.value.provider == provider
            assert error.value.requested_mode is ReasoningSummaryMode.CONCISE
            assert provider in str(error.value)
            assert "concise" in str(error.value)
            provider_call.assert_not_called()


class _ThirdPartyVendorModel(TextGenerationVendorModel):
    def _load_model(self) -> Any:
        return MagicMock()


def test_unsupported_vendor_clients_reject_before_request_side_effects() -> (
    None
):
    with (
        _anthropic_module() as anthropic_module,
        patch(
            "avalan.model.nlp.text.vendor.litellm.litellm.acompletion",
            new=AsyncMock(),
        ) as litellm_completion,
    ):
        client_types = (
            (getattr(anthropic_module, "AnthropicClient"), "anthropic", False),
            (BedrockClient, "bedrock", False),
            (GoogleClient, "google", False),
            (LiteLLMClient, "litellm", False),
            (HuggingfaceClient, "huggingface", False),
            (OllamaClient, "ollama", False),
            (AnyScaleClient, "anyscale", False),
            (AnyScaleClient, "anyscale", True),
            (DeepInfraClient, "deepinfra", False),
            (DeepInfraClient, "deepinfra", True),
            (DeepSeekClient, "deepseek", False),
            (DeepSeekClient, "deepseek", True),
            (GroqClient, "groq", False),
            (GroqClient, "groq", True),
            (HyperbolicClient, "hyperbolic", False),
            (HyperbolicClient, "hyperbolic", True),
            (OpenRouterClient, "openrouter", False),
            (OpenRouterClient, "openrouter", True),
            (TogetherClient, "together", False),
            (TogetherClient, "together", True),
        )
        for client_type, provider, is_azure in client_types:
            for mode in ReasoningSummaryMode:
                for use_async_generator in (False, True):
                    client = object.__new__(client_type)
                    _initialize_openai_client_state(client)
                    template_messages = MagicMock()
                    provider_client = MagicMock()
                    provider_client.responses.create = AsyncMock()
                    provider_client.chat = AsyncMock()
                    provider_client.chat_completion = AsyncMock()
                    provider_client.messages.create = AsyncMock()
                    provider_client.messages.stream = MagicMock()
                    provider_client.aio.models.generate_content = AsyncMock()
                    provider_client.aio.models.generate_content_stream = (
                        MagicMock()
                    )
                    client_instance = AsyncMock(return_value=provider_client)
                    cast(Any, client)._client = provider_client
                    cast(Any, client)._client_instance = client_instance
                    cast(Any, client)._is_azure = is_azure
                    cast(Any, client)._replay_owners_by_call_id = {
                        "kept": object()
                    }
                    cast(Any, client)._template_messages = template_messages

                    with pytest.raises(
                        ReasoningSummaryCapabilityError
                    ) as error:
                        run(
                            client(
                                "model",
                                [],
                                _summary_settings(mode),
                                use_async_generator=use_async_generator,
                            )
                        )

                    assert error.value.provider == provider
                    assert error.value.requested_mode is mode
                    template_messages.assert_not_called()
                    provider_client.responses.create.assert_not_called()
                    provider_client.chat.assert_not_called()
                    provider_client.chat_completion.assert_not_called()
                    provider_client.messages.create.assert_not_called()
                    provider_client.messages.stream.assert_not_called()
                    provider_client.aio.models.generate_content.assert_not_called()
                    provider_client.aio.models.generate_content_stream.assert_not_called()
                    client_instance.assert_not_awaited()
                    assert list(
                        cast(Any, client)._replay_owners_by_call_id
                    ) == ["kept"]

        litellm_completion.assert_not_awaited()


def test_third_party_vendor_model_identity_is_generic() -> None:
    model = object.__new__(_ThirdPartyVendorModel)

    assert model.reasoning_summary_provider == "vendor"


def test_openai_provider_identity_and_capability_are_client_derived() -> None:
    for is_azure, provider in ((False, "openai"), (True, "azure_openai")):
        model = object.__new__(OpenAIModel)
        provider_client = SimpleNamespace(_is_azure=is_azure)
        cast(Any, model)._model = provider_client

        assert model.reasoning_summary_provider == provider
        assert model.reasoning_summary_request_capability.supported_modes == (
            frozenset(ReasoningSummaryMode)
        )


def test_azure_provider_identity_falls_back_to_configured_base_url() -> None:
    model = object.__new__(OpenAIModel)
    cast(Any, model)._model = None
    cast(Any, model)._settings = SimpleNamespace(
        base_url="https://tenant.openai.azure.com/openai/v1/"
    )

    assert model.reasoning_summary_provider == "azure_openai"

    cast(Any, model)._settings = SimpleNamespace(base_url=None)

    assert model.reasoning_summary_provider == "openai"


class _PrivateCapableAdapter:
    reasoning_summary_request_capability = ReasoningSummaryRequestCapability(
        supported_modes=frozenset(ReasoningSummaryMode)
    )
    reasoning_summary_provider = "private_test"

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, *_args: object, **_kwargs: object) -> str:
        self.calls += 1
        return "ok"


def test_private_capable_adapter_crosses_shared_modality_choke() -> None:
    model = _PrivateCapableAdapter()
    settings = _summary_settings(ReasoningSummaryMode.DETAILED)
    operation = Operation(
        generation_settings=settings,
        input="hello",
        modality=Modality.TEXT_GENERATION,
        parameters=OperationParameters(text=OperationTextParameters()),
        requires_input=True,
    )
    engine_uri = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor="openai",
        model_id="private-test-model",
        params={},
    )

    result = run(
        TextGenerationModality()(engine_uri, cast(Any, model), operation)
    )

    assert result == "ok"
    assert model.calls == 1


def test_shared_modality_choke_rejects_before_adapter_invocation() -> None:
    model = _PrivateCapableAdapter()
    model.reasoning_summary_request_capability = (
        ReasoningSummaryRequestCapability()
    )
    operation = Operation(
        generation_settings=_summary_settings(ReasoningSummaryMode.AUTO),
        input="hello",
        modality=Modality.TEXT_GENERATION,
        parameters=OperationParameters(text=OperationTextParameters()),
        requires_input=True,
    )
    engine_uri = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor="openai",
        model_id="public-model",
        params={},
    )

    with pytest.raises(ReasoningSummaryCapabilityError) as error:
        run(
            TextGenerationModality()(
                engine_uri,
                cast(Any, model),
                operation,
            )
        )

    assert error.value.provider == "private_test"
    assert model.calls == 0


def test_request_capability_is_typed_scoped_and_omission_safe() -> None:
    all_modes = ReasoningSummaryRequestCapability(
        supported_modes=frozenset(ReasoningSummaryMode)
    )
    for mode in ReasoningSummaryMode:
        assert all_modes.supports(mode)

    with pytest.raises(AssertionError):
        ReasoningSummaryRequestCapability(
            supported_modes=cast(Any, tuple(ReasoningSummaryMode))
        )
    with pytest.raises(AssertionError):
        ReasoningSummaryRequestCapability(
            supported_modes=cast(Any, frozenset(("auto",)))
        )

    with _anthropic_model_type() as anthropic_model:
        for model_type in (
            anthropic_model,
            BedrockModel,
            LiteLLMModel,
            GoogleModel,
            AnyScaleModel,
            DeepInfraModel,
            DeepSeekModel,
            GroqModel,
            HyperbolicModel,
            OpenRouterModel,
            TogetherModel,
            HuggingfaceModel,
            OllamaModel,
            TextGenerationModel,
            Ds4Model,
            VllmModel,
            MlxLmModel,
        ):
            model = object.__new__(model_type)
            capability = model.reasoning_summary_request_capability
            assert capability.supported_modes == frozenset()
            model._model_id = "summary-capable-looking-name"
            capability = model.reasoning_summary_request_capability
            assert capability.supported_modes == frozenset()

    openai = object.__new__(OpenAIModel)
    cast(Any, openai)._model = None
    cast(Any, openai)._settings = SimpleNamespace(base_url=None)
    openai_capability = openai.reasoning_summary_request_capability
    assert openai_capability.supported_modes == frozenset(ReasoningSummaryMode)
    openai._model_id = "summary-capable-looking-name"
    assert openai.reasoning_summary_request_capability is openai_capability

    azure = object.__new__(OpenAIModel)
    cast(Any, azure)._model = None
    cast(Any, azure)._settings = SimpleNamespace(
        base_url="https://tenant.openai.azure.com/openai/v1/"
    )
    assert azure.reasoning_summary_provider == "azure_openai"
    assert azure.reasoning_summary_request_capability is openai_capability

    validate_reasoning_summary_request(object(), GenerationSettings())
    with pytest.raises(AssertionError, match="must declare"):
        validate_reasoning_summary_request(object(), _summary_settings())


def test_openai_capability_scope_survives_module_reload_order() -> None:
    probe = run_process(
        [
            executable,
            "-c",
            """
from importlib import import_module, reload
from types import SimpleNamespace

from avalan.entities import ReasoningSummaryMode
from avalan.model.nlp.text.vendor.anyscale import AnyScaleClient, AnyScaleModel
from avalan.model.nlp.text.vendor.openai import OpenAIClient, OpenAIModel


def native_instances(client_type, model_type):
    instances = []
    for is_azure, provider in ((False, "openai"), (True, "azure_openai")):
        client = object.__new__(client_type)
        client._is_azure = is_azure
        model = object.__new__(model_type)
        model._model = None
        model._settings = SimpleNamespace(
            base_url=(
                "https://tenant.openai.azure.com/openai/v1/"
                if is_azure
                else None
            )
        )
        instances.append((client, model, provider))
    return instances


old_native_instances = native_instances(OpenAIClient, OpenAIModel)
old_compatible_instances = (
    object.__new__(AnyScaleClient),
    object.__new__(AnyScaleModel),
)
old_compatible_instances[0]._is_azure = True
old_compatible_instances[1]._model = None
old_compatible_instances[1]._model_id = "gpt-5-summary"
old_compatible_instances[1]._settings = SimpleNamespace(
    base_url="https://tenant.openai.azure.com/openai/v1/"
)

openai_module = reload(import_module("avalan.model.nlp.text.vendor.openai"))
anyscale_module = reload(
    import_module("avalan.model.nlp.text.vendor.anyscale")
)
all_native_instances = old_native_instances + native_instances(
    openai_module.OpenAIClient,
    openai_module.OpenAIModel,
)
for client, model, provider in all_native_instances:
    assert client.reasoning_summary_provider == provider
    assert client.reasoning_summary_request_capability.supported_modes == (
        frozenset(ReasoningSummaryMode)
    )
    assert model.reasoning_summary_provider == provider
    assert model.reasoning_summary_request_capability.supported_modes == (
        frozenset(ReasoningSummaryMode)
    )

new_compatible_instances = (
    object.__new__(anyscale_module.AnyScaleClient),
    object.__new__(anyscale_module.AnyScaleModel),
)
new_compatible_instances[0]._is_azure = True
new_compatible_instances[1]._model = None
new_compatible_instances[1]._model_id = "gpt-5-summary"
new_compatible_instances[1]._settings = SimpleNamespace(
    base_url="https://tenant.openai.azure.com/openai/v1/"
)
for compatible in old_compatible_instances + new_compatible_instances:
    assert compatible.reasoning_summary_provider == "anyscale"
    assert compatible.reasoning_summary_request_capability.supported_modes == (
        frozenset()
    )
""",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout == ""


def test_hosted_provider_omission_keeps_exact_dispatch_shape() -> None:
    base_kwargs = {"capability": None, "use_async_generator": True}
    openai_kwargs = {"instructions": None, **base_kwargs}
    with _anthropic_model_type() as anthropic_model:
        providers = (
            (anthropic_model, base_kwargs),
            (BedrockModel, base_kwargs),
            (LiteLLMModel, base_kwargs),
            (GoogleModel, base_kwargs),
            (OpenAIModel, openai_kwargs),
            (AnyScaleModel, openai_kwargs),
            (DeepInfraModel, openai_kwargs),
            (DeepSeekModel, openai_kwargs),
            (GroqModel, openai_kwargs),
            (HyperbolicModel, openai_kwargs),
            (OpenRouterModel, openai_kwargs),
            (TogetherModel, openai_kwargs),
            (HuggingfaceModel, base_kwargs),
            (OllamaModel, base_kwargs),
        )
        for model_type, expected_kwargs in providers:
            model = object.__new__(model_type)
            model._model_id = "model"
            model._logger = getLogger("reasoning-summary-omission")
            model._messages = MagicMock(return_value=[])
            model._model = AsyncMock(return_value="streamer")
            settings = GenerationSettings()

            response = run(model("hello", settings=settings))

            assert response._kwargs["settings"] is settings
            model._model.assert_awaited_once_with(
                "model",
                [],
                settings,
                **expected_kwargs,
            )
            dispatched_settings = model._model.await_args.args[2]
            assert dispatched_settings.reasoning.summary is None


def test_raw_vendor_omission_keeps_exact_provider_payloads() -> None:
    default_settings = GenerationSettings()

    with _anthropic_module() as anthropic_module:
        anthropic = object.__new__(
            getattr(anthropic_module, "AnthropicClient")
        )
        anthropic_create = AsyncMock(
            return_value=SimpleNamespace(content=[], usage=None)
        )
        cast(Any, anthropic)._client = SimpleNamespace(
            messages=SimpleNamespace(create=anthropic_create)
        )
        cast(Any, anthropic)._system_prompt = MagicMock(return_value=None)
        cast(Any, anthropic)._template_messages = MagicMock(return_value=[])
        for settings in (None, default_settings):
            run(
                anthropic(
                    "model",
                    [],
                    settings,
                    use_async_generator=False,
                )
            )
        anthropic_expected = {
            "model": "model",
            "system": None,
            "messages": [],
            "max_tokens": None,
            "temperature": 1.0,
        }
        assert [
            request.kwargs for request in anthropic_create.await_args_list
        ] == [anthropic_expected, anthropic_expected]

    bedrock = object.__new__(BedrockClient)
    bedrock_provider = MagicMock()
    bedrock_provider.converse = AsyncMock(
        return_value={"output": {"message": {"content": []}}}
    )
    cast(Any, bedrock)._client_instance = AsyncMock(
        return_value=bedrock_provider
    )
    cast(Any, bedrock)._system_prompt = MagicMock(return_value=None)
    cast(Any, bedrock)._template_messages = MagicMock(return_value=[])
    for settings in (None, default_settings):
        run(
            bedrock(
                "model",
                [],
                settings,
                use_async_generator=False,
            )
        )
    assert [
        request.kwargs for request in bedrock_provider.converse.await_args_list
    ] == [
        {"modelId": "model", "messages": []},
        {
            "modelId": "model",
            "messages": [],
            "inferenceConfig": {
                "temperature": 1.0,
                "topP": 1.0,
                "topK": 50,
            },
        },
    ]

    google = object.__new__(GoogleClient)
    google_generate = AsyncMock(
        return_value=SimpleNamespace(text="", usage_metadata=None)
    )
    cast(Any, google)._client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=google_generate)
        )
    )
    cast(Any, google)._template_messages = MagicMock(return_value=[])
    cast(Any, google)._system_prompt = MagicMock(return_value=None)
    for settings in (None, default_settings):
        run(
            google(
                "model",
                [],
                settings,
                use_async_generator=False,
            )
        )
    assert [request.kwargs for request in google_generate.await_args_list] == [
        {"model": "model", "contents": []},
        {
            "model": "model",
            "contents": [],
            "config": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
            },
        },
    ]

    with patch(
        "avalan.model.nlp.text.vendor.litellm.litellm.acompletion",
        new=AsyncMock(return_value={"choices": []}),
    ) as litellm_completion:
        litellm = object.__new__(LiteLLMClient)
        cast(Any, litellm)._api_key = "key"
        cast(Any, litellm)._base_url = "https://example.test"
        cast(Any, litellm)._template_messages = MagicMock(return_value=[])
        for settings in (None, default_settings):
            run(
                litellm(
                    "model",
                    [],
                    settings,
                    use_async_generator=False,
                )
            )
        litellm_expected = {
            "model": "model",
            "messages": [],
            "api_key": "key",
            "stream": False,
            "api_base": "https://example.test",
        }
        assert [
            request.kwargs for request in litellm_completion.await_args_list
        ] == [litellm_expected, litellm_expected]

    huggingface = object.__new__(HuggingfaceClient)
    huggingface_completion = AsyncMock(return_value={"choices": []})
    cast(Any, huggingface)._client = SimpleNamespace(
        chat_completion=huggingface_completion
    )
    cast(Any, huggingface)._template_messages = MagicMock(return_value=[])
    for settings in (None, default_settings):
        run(
            huggingface(
                "model",
                [],
                settings,
                use_async_generator=False,
            )
        )
    huggingface_expected = {
        "model": "model",
        "messages": [],
        "temperature": 1.0,
        "max_tokens": None,
        "top_p": 1.0,
        "stop": None,
        "stream": False,
    }
    assert [
        request.kwargs for request in huggingface_completion.await_args_list
    ] == [huggingface_expected, huggingface_expected]

    ollama = object.__new__(OllamaClient)
    ollama_chat = AsyncMock(return_value={"message": {"content": ""}})
    cast(Any, ollama)._client = SimpleNamespace(chat=ollama_chat)
    cast(Any, ollama)._template_messages = MagicMock(return_value=[])
    for settings in (None, default_settings):
        run(
            ollama(
                "model",
                [],
                settings,
                use_async_generator=False,
            )
        )
    ollama_expected = {
        "model": "model",
        "messages": [],
        "stream": False,
    }
    assert [request.kwargs for request in ollama_chat.await_args_list] == [
        ollama_expected,
        ollama_expected,
    ]

    openai_types = (
        (OpenAIClient, False),
        (OpenAIClient, True),
        (AnyScaleClient, False),
        (DeepInfraClient, False),
        (DeepSeekClient, False),
        (GroqClient, False),
        (HyperbolicClient, False),
        (OpenRouterClient, False),
        (TogetherClient, False),
    )
    for client_type, is_azure in openai_types:
        openai = object.__new__(client_type)
        _initialize_openai_client_state(openai)
        openai_create = AsyncMock(
            return_value=SimpleNamespace(output=[], usage=None)
        )
        cast(Any, openai)._client = SimpleNamespace(
            responses=SimpleNamespace(create=openai_create)
        )
        cast(Any, openai)._is_azure = is_azure
        cast(Any, openai)._template_messages = MagicMock(return_value=[])
        for settings in (None, default_settings):
            run(
                openai(
                    "model",
                    [],
                    settings,
                    use_async_generator=False,
                )
            )
        openai_base = {
            "extra_headers": {
                "X-Title": "Avalan",
                "HTTP-Referer": "https://github.com/avalan-ai/avalan",
            },
            "model": "model",
            "input": [],
            "store": False,
            "stream": False,
        }
        expected_default = (
            openai_base
            if is_azure
            else {**openai_base, "temperature": 1.0, "top_p": 1.0}
        )
        assert [
            request.kwargs for request in openai_create.await_args_list
        ] == [
            openai_base,
            expected_default,
        ]

    payloads = [
        *[request.kwargs for request in anthropic_create.await_args_list],
        *[
            request.kwargs
            for request in bedrock_provider.converse.await_args_list
        ],
        *[request.kwargs for request in google_generate.await_args_list],
        *[
            request.kwargs
            for request in huggingface_completion.await_args_list
        ],
        *[request.kwargs for request in ollama_chat.await_args_list],
    ]
    assert all(
        not _mapping_contains_key(payload, "summary") for payload in payloads
    )


def test_local_provider_omissions_reach_unchanged_dispatch_shapes() -> None:
    settings = GenerationSettings()

    transformers = object.__new__(TextGenerationModel)
    cast(Any, transformers)._tokenizer = SimpleNamespace(
        eos_token_id=7,
        bos_token="<s>",
    )
    transformers_model = MagicMock()
    cast(Any, transformers)._model = transformers_model
    cast(Any, transformers)._logger = getLogger("transformers-omission")
    cast(Any, transformers)._tokenize_input = MagicMock(
        return_value={"input_ids": [[1]]}
    )

    transformers_response = run(transformers("hello", settings=settings))

    transformers_model.assert_not_called()
    assert transformers_response._kwargs["settings"].reasoning.summary is None
    cast(Any, transformers)._tokenize_input.assert_called_once()

    vllm = object.__new__(VllmModel)
    vllm_model = MagicMock()
    cast(Any, vllm)._model = vllm_model
    cast(Any, vllm)._prompt = MagicMock(return_value="prompt")
    cast(Any, vllm)._stream_generator = MagicMock(return_value="stream")

    vllm_result = run(vllm("hello", settings=settings))

    assert vllm_result == "stream"
    vllm_model.assert_not_called()
    vllm_generation_settings = cast(
        Any, vllm
    )._stream_generator.call_args.args[1]
    assert vllm_generation_settings.reasoning.summary is None

    mlx = object.__new__(MlxLmModel)
    mlx_model = MagicMock()
    cast(Any, mlx)._model = mlx_model
    cast(Any, mlx)._tokenizer = MagicMock()
    cast(Any, mlx)._logger = getLogger("mlx-omission")
    with patch.object(
        TextGenerationModel,
        "_tokenize_input",
        return_value={"input_ids": [[1]]},
    ) as mlx_tokenize:
        mlx_response = run(mlx("hello", settings=settings))

    mlx_model.assert_not_called()
    mlx_tokenize.assert_called_once()
    assert mlx_response._kwargs["settings"].reasoning.summary is None

    ds4_settings = GenerationSettings(use_async_generator=False)
    ds4 = object.__new__(Ds4Model)
    ds4_model = MagicMock()
    cast(Any, ds4)._model = ds4_model
    cast(Any, ds4)._logger = getLogger("ds4-omission")
    cast(Any, ds4)._uses_dsml_tools = MagicMock(return_value=False)
    cast(Any, ds4)._render_prompt_tokens_async = AsyncMock(return_value=[1])
    ds4_plan = SimpleNamespace(
        use_sampling=False,
        parse_dsml_tools=False,
    )
    cast(Any, ds4)._generation_plan = MagicMock(return_value=ds4_plan)

    ds4_response = run(ds4("hello", settings=ds4_settings))

    ds4_model.assert_not_called()
    cast(Any, ds4)._uses_dsml_tools.assert_called_once_with("hello", None)
    cast(Any, ds4)._render_prompt_tokens_async.assert_awaited_once_with(
        "hello",
        None,
        None,
        ds4_settings,
        capability=None,
    )
    cast(Any, ds4)._generation_plan.assert_called_once_with(
        ds4_settings,
        1,
        manual_sampling=False,
        parse_dsml_tools=False,
        pick=None,
    )
    dispatched_ds4_settings = cast(
        GenerationSettings,
        cast(Any, ds4)._render_prompt_tokens_async.await_args.args[3],
    )
    assert ds4_plan.parse_dsml_tools is False
    assert ds4_response._kwargs["generation_plan"] is ds4_plan
    assert dispatched_ds4_settings.reasoning.summary is None
    assert ds4_response._kwargs["settings"].reasoning.summary is None


def test_local_omission_keeps_exact_backend_request_shapes() -> None:
    settings = GenerationSettings(use_inputs_attention_mask=False)

    transformers = object.__new__(TextGenerationModel)
    transformers_generate = MagicMock(return_value=[])
    cast(Any, transformers)._model = SimpleNamespace(
        generate=transformers_generate
    )
    cast(Any, transformers)._tokenizer = SimpleNamespace(eos_token_id=7)
    with (
        patch(
            "avalan.model.nlp._tensor_type",
            return_value=type("Tensor", (), {}),
        ),
        patch("avalan.model.nlp._batch_encoding_type", return_value=dict),
        patch("avalan.model.nlp.inference_mode", return_value=nullcontext()),
    ):
        cast(Any, transformers)._generate_output(
            {"input_ids": [[1]]},
            settings,
        )
    transformers_kwargs = transformers_generate.call_args.kwargs
    assert set(transformers_kwargs) == {
        "input_ids",
        "tokenizer",
        "bos_token_id",
        "diversity_penalty",
        "do_sample",
        "early_stopping",
        "eos_token_id",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "max_length",
        "max_new_tokens",
        "max_time",
        "min_length",
        "min_new_tokens",
        "min_p",
        "num_beams",
        "num_beam_groups",
        "num_return_sequences",
        "output_attentions",
        "output_hidden_states",
        "output_logits",
        "output_scores",
        "pad_token_id",
        "penalty_alpha",
        "prompt_lookup_num_tokens",
        "repetition_penalty",
        "return_dict_in_generate",
        "stop_strings",
        "stopping_criteria",
        "streamer",
        "temperature",
        "top_k",
        "top_p",
        "use_cache",
        "cache_implementation",
    }
    assert transformers_kwargs["input_ids"] == [[1]]
    assert transformers_kwargs["temperature"] == 1.0

    vllm = object.__new__(VllmModel)
    vllm_generate = MagicMock(return_value=[])
    sampling_params = object()
    cast(Any, vllm)._model = SimpleNamespace(generate=vllm_generate)
    with patch(
        "avalan.model.nlp.text.vllm._sampling_params_class",
        return_value=MagicMock(return_value=sampling_params),
    ) as sampling_params_factory:
        vllm_result = cast(Any, vllm)._string_output("prompt", settings)
    assert vllm_result == ""
    sampling_params_factory.return_value.assert_called_once_with(
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        max_tokens=None,
        stop=None,
    )
    vllm_generate.assert_called_once_with(["prompt"], sampling_params)

    mlx = object.__new__(MlxLmModel)
    mlx_generate = MagicMock(return_value="answer")
    cast(Any, mlx)._model = "mlx-model"
    cast(Any, mlx)._tokenizer = "mlx-tokenizer"
    cast(Any, mlx)._get_sampler_and_prompt = MagicMock(
        return_value=("sampler", "prompt")
    )
    cast(Any, mlx)._run_on_worker = lambda fn: fn()
    with patch(
        "avalan.model.nlp.text.mlxlm._require_mlx_lm",
        return_value=SimpleNamespace(generate=mlx_generate),
    ):
        mlx_result = cast(Any, mlx)._string_output({}, settings, False)
    assert mlx_result == "answer"
    mlx_generate.assert_called_once_with(
        "mlx-model",
        "mlx-tokenizer",
        "prompt",
        sampler="sampler",
        max_tokens=None,
    )

    ds4 = object.__new__(Ds4Model)
    sampling_options = SimpleNamespace(temperature=1.0)
    cast(Any, ds4)._validate_generation_features = MagicMock()
    cast(Any, ds4)._sampling_options = MagicMock(return_value=sampling_options)
    cast(Any, ds4)._stop_strings = MagicMock(return_value=())
    plan = cast(Any, ds4)._generation_plan(settings, prompt_length=1)
    ds4_worker = MagicMock()
    ds4_worker.generate_string.return_value = "answer"
    cast(Any, ds4)._ds4_worker = MagicMock(return_value=ds4_worker)

    ds4_result = cast(Any, ds4)._generation_string([1], plan, settings)

    assert ds4_result == "answer"
    assert plan.max_new_tokens == 19
    assert plan.sampling_options is sampling_options
    assert plan.stop_strings == ()
    assert plan.use_sampling is False
    ds4_worker.generate_string.assert_called_once_with([1], plan)

    payloads = (
        transformers_kwargs,
        sampling_params_factory.return_value.call_args.kwargs,
        mlx_generate.call_args.kwargs,
        asdict(plan),
    )
    assert all(
        not _mapping_contains_key(payload, "summary") for payload in payloads
    )


def test_shared_local_omission_invokes_adapter_once() -> None:
    model = _PrivateCapableAdapter()
    operation = Operation(
        generation_settings=GenerationSettings(),
        input="hello",
        modality=Modality.TEXT_GENERATION,
        parameters=OperationParameters(text=OperationTextParameters()),
        requires_input=True,
    )
    engine_uri = EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=None,
        model_id="local-test-model",
        params={},
    )

    result = run(
        TextGenerationModality()(engine_uri, cast(Any, model), operation)
    )

    assert result == "ok"
    assert model.calls == 1


def test_openai_model_forwards_without_fallback_retry() -> None:
    model = object.__new__(OpenAIModel)
    model._model_id = "openai-model"
    model._logger = getLogger("openai-summary")
    model._messages = MagicMock(return_value=[])
    provider_rejection = RuntimeError("provider rejected detailed summary")
    model._model = AsyncMock(side_effect=provider_rejection)
    settings = _summary_settings(ReasoningSummaryMode.DETAILED)

    with pytest.raises(RuntimeError) as error:
        run(model("hello", settings=settings))

    assert error.value is provider_rejection
    model._model.assert_awaited_once_with(
        "openai-model",
        [],
        settings,
        instructions=None,
        capability=None,
        use_async_generator=True,
    )


def test_explicit_local_provider_error_and_zero_adapter_calls() -> None:
    model = _PrivateCapableAdapter()
    model.reasoning_summary_request_capability = (
        ReasoningSummaryRequestCapability()
    )

    with pytest.raises(ReasoningSummaryCapabilityError) as error:
        validate_reasoning_summary_request(
            model,
            _summary_settings(ReasoningSummaryMode.AUTO),
            provider="local",
        )

    assert error.value.provider == "local"
    assert error.value.requested_mode is ReasoningSummaryMode.AUTO
    assert model.calls == 0


def test_raw_reasoning_summary_preserves_authority_rejection() -> None:
    request = ResponsesRequest.model_validate(
        {
            "input": "hi",
            "reasoning": {"summary": "concise"},
        }
    )
    assert request.reasoning is not None
    assert request.reasoning.summary is ReasoningSummaryMode.CONCISE

    for reasoning in (
        {"summary": {"mode": "auto"}},
        {"summary": {"sandboxProfile": "unsafe"}},
        {"summary": "auto", "sandboxProfile": "unsafe"},
    ):
        with pytest.raises(ValidationError, match="runtime authority"):
            ResponsesRequest.model_validate(
                {
                    "input": "hi",
                    "reasoning": reasoning,
                }
            )
