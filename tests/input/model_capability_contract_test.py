"""Exercise model capability projection across every supported path."""

from ast import Import, ImportFrom, Name, parse, walk
from asyncio import run
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from json import dumps, loads
from logging import getLogger
from pathlib import Path
from subprocess import run as run_process
from types import SimpleNamespace
from typing import Any, Callable, cast, get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.types import GenerateContentConfig
from google.genai.types import Tool as GoogleTool

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    Backend,
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
    ToolCallContext,
    ToolCallResult,
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
    TransformerEngineSettings,
    Vendor,
)
from avalan.interaction import (
    RESERVED_INPUT_CAPABILITY_NAME,
    AnswerProvenance,
    CapabilityRevision,
    ContinuationRevisionBinding,
    InputAnsweredResult,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    RequirementMode,
    SingleSelectionQuestion,
    decode_continuation_snapshot,
    decode_input_model_result,
    encode_continuation_snapshot,
)
from avalan.model import (
    CapabilityBatchAccepted,
    CapabilityBatchRejected,
    CapabilityBatchRejectionCode,
    ContinuationSnapshotCodecRegistry,
    CorrelatedCapabilityResult,
    ModelCapabilityCatalog,
    ModelCapabilityKind,
    ModelCapabilityValidationError,
    ProviderCapabilityCall,
    ProviderCapabilitySupport,
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)
from avalan.model.call import ModelCallContext
from avalan.model.nlp.text.ds4 import Ds4Model, Ds4Worker
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.nlp.text.local_protocol import (
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL,
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID,
    LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME,
)
from avalan.model.nlp.text.mlxlm import MlxLmModel
from avalan.model.nlp.text.vendor.anthropic import AnthropicClient
from avalan.model.nlp.text.vendor.bedrock import BedrockClient
from avalan.model.nlp.text.vendor.google import GoogleClient
from avalan.model.nlp.text.vendor.huggingface import HuggingfaceClient
from avalan.model.nlp.text.vendor.litellm import LiteLLMClient
from avalan.model.nlp.text.vendor.ollama import OllamaClient
from avalan.model.nlp.text.vendor.openai import OpenAIClient
from avalan.model.nlp.text.vllm import VllmModel
from avalan.model.provider import ProviderFamily
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    NATIVE_STRUCTURED_OUTPUT_METADATA_KEY,
    StreamItemKind,
    StreamProviderEvent,
    TextGenerationNonStreamResult,
    TextGenerationNonStreamToolCall,
    TextGenerationSingleStream,
)
from avalan.model.vendor import TextGenerationVendor
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

_ROOT = Path(__file__).resolve().parents[2]
_PROVIDER_FIXTURES = _ROOT / "tests" / "fixtures" / "input" / "providers"

_LOCAL_MODEL_PATH_REFERENCES = {
    Backend.TRANSFORMERS: (
        "transformers",
        "avalan.model.nlp.text.generation",
        "TextGenerationModel",
    ),
    Backend.MLXLM: (
        "mlx_lm",
        "avalan.model.nlp.text.mlxlm",
        "MlxLmModel",
    ),
    Backend.VLLM: (
        "vllm",
        "avalan.model.nlp.text.vllm",
        "VllmModel",
    ),
    Backend.DS4: (
        "ds4",
        "avalan.model.nlp.text.ds4",
        "Ds4Model",
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class _TransportCall:
    request: str
    response: str


class _ScriptedTransport:
    def __init__(self, calls: tuple[_TransportCall, ...]) -> None:
        self._calls = calls
        self._index = 0

    @property
    def call_count(self) -> int:
        return self._index

    async def generate(self, request: str) -> str:
        if self._index >= len(self._calls):
            raise RuntimeError("scripted transport exhausted")
        call = self._calls[self._index]
        if request != call.request:
            raise ValueError("scripted transport request mismatch")
        self._index += 1
        return call.response


class _CapturedAsyncCall:
    """Capture one deterministic fake SDK call and return queued responses."""

    def __init__(self, *responses: object) -> None:
        self.calls: list[dict[str, object]] = []
        self._responses = list(responses)

    async def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("fake SDK response queue exhausted")
        return self._responses.pop(0)


class _BoundaryEngine:
    model_id = "fixture-model"
    tokenizer = None


def lookup(query: str) -> str:
    """Look up one deterministic value.

    Args:
        query: Value to look up.

    Returns:
        The provided value.
    """
    return query


def request_user_input(value: str) -> str:
    """Collide with the reserved capability name.

    Args:
        value: Arbitrary input.

    Returns:
        The provided value.
    """
    return value


def _fixture(name: str) -> dict[str, Any]:
    value = loads((_PROVIDER_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _production_model_path_inventory() -> dict[str, tuple[str, str]]:
    inventory: dict[str, tuple[str, str]] = {}
    for vendor in get_args(Vendor):
        assert isinstance(vendor, str)
        if vendor == "local":
            continue
        module_name = f"avalan.model.nlp.text.vendor.{vendor}"
        module = import_module(module_name)
        client_names = tuple(
            name
            for name, value in vars(module).items()
            if isinstance(value, type)
            and issubclass(value, TextGenerationVendor)
            and value is not TextGenerationVendor
            and value.__module__ == module_name
        )
        assert (
            len(client_names) == 1
        ), f"production vendor {vendor!r} must define one client adapter"
        path_id = "hugging_face" if vendor == "huggingface" else vendor
        inventory[path_id] = (module_name, client_names[0])

    openai_reference = inventory[ProviderFamily.OPENAI.value]
    inventory[ProviderFamily.AZURE_OPENAI.value] = openai_reference
    inventory[f"generic_{ProviderFamily.OPENAI_COMPATIBLE.value}"] = (
        openai_reference
    )

    assert frozenset(_LOCAL_MODEL_PATH_REFERENCES) == frozenset(
        Backend
    ), "every production local backend requires an explicit model path"
    for (
        path_id,
        module_name,
        class_name,
    ) in _LOCAL_MODEL_PATH_REFERENCES.values():
        inventory[path_id] = (module_name, class_name)
    return inventory


def _validate_provider_matrix_completeness(
    rows: tuple[dict[str, Any], ...],
) -> None:
    actual: dict[str, tuple[str, str]] = {}
    for row in rows:
        path_id = row.get("id")
        module_name = row.get("module")
        class_name = row.get("class")
        assert isinstance(path_id, str) and path_id
        assert isinstance(module_name, str) and module_name
        assert isinstance(class_name, str) and class_name
        assert (
            path_id not in actual
        ), f"duplicate provider matrix path {path_id!r}"
        actual[path_id] = (module_name, class_name)

    expected = _production_model_path_inventory()
    assert frozenset(actual) == frozenset(expected), (
        "provider matrix path IDs differ from the production model inventory: "
        f"missing={sorted(frozenset(expected) - frozenset(actual))}, "
        f"unexpected={sorted(frozenset(actual) - frozenset(expected))}"
    )
    assert actual == expected, (
        "provider matrix module/class references differ from the production "
        "model inventory"
    )


def _provider_rows() -> tuple[dict[str, Any], ...]:
    value = _fixture("provider_matrix.json")["paths"]
    assert isinstance(value, list)
    assert all(isinstance(row, dict) for row in value)
    rows = tuple(cast(list[dict[str, Any]], value))
    _validate_provider_matrix_completeness(rows)
    return rows


def _attached_support() -> ProviderCapabilitySupport:
    return ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
        attached_resolution=True,
    )


def _durable_contract() -> (
    tuple[ProviderCapabilitySupport, ContinuationRevisionBinding]
):
    binding = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("fake-provider"),
        model_id=ModelId("fake-model"),
        provider_config_revision=ProviderConfigRevision("provider-config-v1"),
        model_config_revision=ModelConfigRevision("model-config-v1"),
        capability_revision=CapabilityRevision("capability-v1"),
    )
    registry = ContinuationSnapshotCodecRegistry("fake-codec-registry")
    registry.register(
        codec_id="fake-provider-model-codec",
        revision_binding=binding,
        snapshot_kind="fake_provider_snapshot",
        export_snapshot=encode_continuation_snapshot,
        restore_snapshot=lambda value, expected: decode_continuation_snapshot(
            value,
            expected_binding=expected,
        ),
    )
    return (
        ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
            continuation_snapshot_codec_registry=registry,
            continuation_snapshot_codec=registry.reference(
                "fake-provider-model-codec"
            ),
        ),
        binding,
    )


def _manager(
    *,
    mapped_name: str = "lookup",
    tools: list[Any] | None = None,
    enable_tools: list[str] | None = None,
) -> ToolManager:
    return ToolManager.create_instance(
        available_toolsets=[ToolSet(namespace="pkg", tools=tools or [lookup])],
        enable_tools=enable_tools or ["pkg.lookup"],
        settings=ToolManagerSettings(
            tool_name_policy=ToolNamePolicySettings(
                mode=ToolNamePolicyMode.SANITIZED,
                map={"pkg.lookup": mapped_name},
            )
        ),
    )


def _catalog(
    support: ProviderCapabilitySupport,
    *,
    with_domain_tool: bool = False,
    revision_binding: ContinuationRevisionBinding | None = None,
) -> ModelCapabilityCatalog:
    seed = (
        _manager().export_model_capability_seed() if with_domain_tool else None
    )
    return ModelCapabilityCatalog.create(
        seed,
        support=support,
        revision_binding=revision_binding,
    )


def _boundary_orchestrator_response(
    output: object,
    *,
    provider_family: str,
    manager: ToolManager,
    catalog: ModelCapabilityCatalog,
    confirmation: AsyncMock,
    enable_tool_parsing: bool,
) -> OrchestratorResponse:
    operation = AgentOperation(
        specification=Specification(role=None, goal=None),
        environment=EngineEnvironment(
            engine_uri=EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="fixture-model",
                params={},
            ),
            settings=TransformerEngineSettings(),
        ),
    )
    input_message = Message(
        role=MessageRole.USER,
        content="Choose one region.",
    )
    response = (
        output
        if isinstance(output, TextGenerationResponse)
        else TextGenerationResponse(
            cast(Any, output),
            logger=getLogger(__name__),
            use_async_generator=False,
            provider_family=provider_family,
        )
    )
    agent = MagicMock(spec=EngineAgent)
    agent.engine = _BoundaryEngine()
    context = ModelCallContext(
        specification=operation.specification,
        input=input_message,
        capability=catalog,
    )
    return OrchestratorResponse(
        input_message,
        response,
        agent,
        operation,
        {},
        context,
        tool=manager,
        capability=catalog,
        tool_confirm=confirmation,
        enable_tool_parsing=enable_tool_parsing,
    )


async def _consume_boundary_response(
    response: OrchestratorResponse,
) -> list[object]:
    return [item async for item in response]


def _assert_reserved_call_side_effects_absent(
    response: OrchestratorResponse,
    items: list[object],
    confirmation: AsyncMock,
) -> None:
    confirmation.assert_not_awaited()
    assert response._call_history == []
    assert response._attempted_call_signatures == set()
    assert response._tool_cycle_signatures == set()
    assert response._tool_cycle_count == 0
    assert not any(
        getattr(item, "kind", None)
        in {
            StreamItemKind.TOOL_EXECUTION_STARTED,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            StreamItemKind.TOOL_EXECUTION_ERROR,
        }
        for item in items
    )


def _mutable_json(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _mutable_json(model_dump(mode="json", exclude_none=True))
    to_json_dict = getattr(value, "to_json_dict", None)
    if callable(to_json_dict):
        return _mutable_json(to_json_dict())
    if isinstance(value, Mapping):
        return {key: _mutable_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_mutable_json(item) for item in value]
    return value


def _native_schemas(
    row: Mapping[str, Any],
    catalog: ModelCapabilityCatalog,
) -> object | None:
    schema_format = row["schema_format"]
    family = cast(str, row["provider_family"])
    if schema_format == "none":
        return None
    if schema_format == "openai":
        return OpenAIClient._tool_schemas(
            catalog,
            provider_family=family,
        )
    if schema_format == "anthropic":
        return AnthropicClient._tool_schemas(catalog)
    if schema_format == "bedrock":
        return BedrockClient._tool_schemas(catalog)
    if schema_format == "google":
        client = object.__new__(GoogleClient)
        config = client._config(
            "model",
            [],
            None,
            capability=catalog,
        )
        if config is None:
            return None
        sdk_config = GenerateContentConfig(**config)
        assert sdk_config.tools is not None
        assert isinstance(sdk_config.tools[0], GoogleTool)
        declarations = sdk_config.tools[0].function_declarations
        assert declarations is not None
        declaration = declarations[0]
        assert declaration.parameters is None
        assert declaration.parameters_json_schema is not None
        return _mutable_json(sdk_config.tools)
    if schema_format == "litellm":
        projection = catalog.project(family)
        return None if projection.is_empty else projection.schemas
    if schema_format == "local":
        return TextGenerationModel._provider_tool_schemas(catalog)
    assert schema_format == "ds4"
    return Ds4Model._tool_schemas(catalog)


def _normalized_schema(
    row: Mapping[str, Any],
    native_schemas: object,
) -> dict[str, object]:
    schema_format = row["schema_format"]
    value = _mutable_json(native_schemas)
    if schema_format == "ds4":
        assert isinstance(value, str)
        function = loads(value)
        assert isinstance(function, dict)
        return {"type": "function", "function": function}
    assert isinstance(value, list) and len(value) == 1
    first = value[0]
    assert isinstance(first, dict)
    if schema_format == "openai":
        return {
            "type": "function",
            "function": {
                "name": first["name"],
                "description": first["description"],
                "parameters": first["parameters"],
            },
        }
    if schema_format == "anthropic":
        return {
            "type": "function",
            "function": {
                "name": first["name"],
                "description": first["description"],
                "parameters": first["input_schema"],
            },
        }
    if schema_format == "bedrock":
        tool_spec = first["toolSpec"]
        assert isinstance(tool_spec, dict)
        input_schema = tool_spec["inputSchema"]
        assert isinstance(input_schema, dict)
        return {
            "type": "function",
            "function": {
                "name": tool_spec["name"],
                "description": tool_spec["description"],
                "parameters": input_schema["json"],
            },
        }
    if schema_format == "google":
        declarations = first["function_declarations"]
        assert isinstance(declarations, list) and len(declarations) == 1
        function = declarations[0]
        assert isinstance(function, dict)
        parameters = function.get("parameters_json_schema")
        if parameters is None:
            parameters = function.get("parameters")
        return {
            "type": "function",
            "function": {
                "name": function["name"],
                "description": function["description"],
                "parameters": parameters,
            },
        }
    assert schema_format in {"litellm", "local"}
    return cast(dict[str, object], first)


def _native_response(
    row: Mapping[str, Any],
    arguments: dict[str, object],
    call_id: str,
) -> dict[str, object]:
    transcript_format = row["transcript_format"]
    if transcript_format == "openai":
        return {
            "id": "response-fixture",
            "status": "completed",
            "output": [
                {
                    "id": "function-call-fixture",
                    "type": "function_call",
                    "call_id": call_id,
                    "name": RESERVED_INPUT_CAPABILITY_NAME,
                    "arguments": dumps(arguments),
                }
            ],
        }
    if transcript_format == "anthropic":
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": RESERVED_INPUT_CAPABILITY_NAME,
                    "input": arguments,
                }
            ]
        }
    if transcript_format == "bedrock":
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": call_id,
                                "name": RESERVED_INPUT_CAPABILITY_NAME,
                                "input": arguments,
                            }
                        }
                    ]
                }
            }
        }
    if transcript_format == "google":
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": {
                                    "id": call_id,
                                    "name": RESERVED_INPUT_CAPABILITY_NAME,
                                    "args": arguments,
                                }
                            }
                        ]
                    }
                }
            ]
        }
    if transcript_format == "litellm":
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": RESERVED_INPUT_CAPABILITY_NAME,
                                    "arguments": dumps(arguments),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    assert transcript_format == "local"
    return {
        "control_frame": (
            "<tool_call>"
            + dumps(
                {
                    "id": call_id,
                    "name": RESERVED_INPUT_CAPABILITY_NAME,
                    "arguments": arguments,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "</tool_call>"
        )
    }


def _provider_response_text(
    row: Mapping[str, Any],
    response: dict[str, object],
    catalog: ModelCapabilityCatalog,
) -> str:
    transcript_format = row["transcript_format"]
    if transcript_format == "openai":
        return OpenAIClient._non_stream_response_content(
            response,
            capability=catalog,
            provider_family=cast(str, row["provider_family"]),
        )
    if transcript_format == "anthropic":
        return AnthropicClient._non_stream_response_content(
            response,
            capability=catalog,
        )
    if transcript_format == "bedrock":
        client = object.__new__(BedrockClient)
        return client._response_text(response, capability=catalog)
    if transcript_format == "google":
        return GoogleClient._response_text(response, capability=catalog)
    if transcript_format == "litellm":
        text = LiteLLMClient._message_text(response, capability=catalog)
        assert text is not None
        return text
    assert transcript_format == "local"
    control_frame = response["control_frame"]
    assert isinstance(control_frame, str)
    return control_frame


def _task_input_from_response(
    row: Mapping[str, Any],
    response: dict[str, object],
    catalog: ModelCapabilityCatalog,
) -> TaskInputCapabilityCall:
    text = _provider_response_text(row, response, catalog)
    parsed = catalog.parse_calls(text)
    assert parsed.diagnostics == []
    assert len(parsed.calls) == 1
    call = parsed.calls[0]
    classification = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id=call.id,
                provider_name=call.provider_name or call.name,
                arguments=call.arguments,
            ),
        ),
        provider_family=cast(str, row["provider_family"]),
    )
    assert isinstance(classification, CapabilityBatchAccepted)
    assert isinstance(classification.task_input, TaskInputCapabilityCall)
    return classification.task_input


def _continuation_message(
    transcript_format: str,
    result: CorrelatedCapabilityResult,
) -> dict[str, object]:
    if transcript_format == "openai":
        return cast(
            dict[str, object],
            OpenAIClient.capability_result_message(result),
        )
    if transcript_format == "anthropic":
        return cast(
            dict[str, object],
            AnthropicClient.capability_result_message(result),
        )
    if transcript_format == "bedrock":
        return cast(
            dict[str, object],
            BedrockClient.capability_result_message(result),
        )
    if transcript_format == "google":
        return cast(
            dict[str, object],
            GoogleClient.capability_result_message(result),
        )
    if transcript_format == "litellm":
        return cast(
            dict[str, object],
            LiteLLMClient.capability_result_message(result),
        )
    assert transcript_format == "local"
    message = result.local_message()
    assert isinstance(message.content, str)
    return {
        "role": str(message.role),
        "name": message.name,
        "content": message.content,
    }


class _CapturingTokenizer:
    chat_template: object = {
        "default": "{{ messages }}",
        LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: "{{ messages }}",
    }
    has_chat_template = True
    bos_token = "<bos>"
    eos_token_id = 0

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.tools: object | None = None
        self.prompt: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, list[list[int]]]:
        self.messages = messages
        self.tools = kwargs.get("tools")
        return {"input_ids": [[101, 102]]}

    def decode(self, _: object, **__: object) -> str:
        return dumps(self.messages, default=str, sort_keys=True)


class _BoundaryTokenIds:
    shape = (1, 2)

    def __getitem__(self, index: int) -> list[int]:
        assert index == 0
        return [101, 102]


class _BoundaryTokenizer(_CapturingTokenizer):
    def __init__(self, generated_text: str | Callable[[], str]) -> None:
        super().__init__()
        self._generated_text = generated_text

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        self.messages = messages
        self.tools = kwargs.get("tools")
        return {"input_ids": _BoundaryTokenIds()}

    def decode(self, token_ids: object, **__: object) -> str:
        if token_ids == [999]:
            return (
                self._generated_text()
                if callable(self._generated_text)
                else self._generated_text
            )
        return dumps(self.messages, default=str, sort_keys=True)


class _PlainTokenizer(_CapturingTokenizer):
    chat_template = None
    has_chat_template = False

    def __call__(self, prompt: str, **_: object) -> dict[str, list[list[int]]]:
        self.prompt = prompt
        return {"input_ids": [[101, 102]]}


def _local_serialized_payload(
    path_id: str,
    message: Message,
    catalog: ModelCapabilityCatalog,
) -> str:
    if path_id == "ds4":
        ds4_model = object.__new__(Ds4Model)
        _, messages = ds4_model._ds4_prompt_messages([message], None, None)
        assert len(messages) == 1
        return messages[0].content

    tokenizer = _CapturingTokenizer()
    model_type: type[TextGenerationModel]
    if path_id == "transformers":
        model_type = TextGenerationModel
    elif path_id == "vllm":
        model_type = VllmModel
    else:
        assert path_id == "mlx_lm"
        model_type = MlxLmModel
    local_model = object.__new__(model_type)
    local_model._logger = getLogger(__name__)
    local_model._model_id = "fixture-model"
    local_model._tokenizer = tokenizer
    local_model._model = SimpleNamespace()
    if path_id == "vllm":
        serialized = cast(VllmModel, local_model)._prompt(
            [message],
            None,
            capability=catalog,
        )
        assert "provider-call-001" in serialized
    else:
        local_model._tokenize_input(
            [message],
            capability=catalog,
        )
    serialized_messages = [
        item for item in tokenizer.messages if item.get("role") == message.role
    ]
    assert len(serialized_messages) == 1
    content = serialized_messages[0]["content"]
    assert isinstance(content, str)
    return content


def _local_request_serialization(
    path_id: str,
    catalog: ModelCapabilityCatalog,
    *,
    has_chat_template: bool,
    tokenizer_override: _CapturingTokenizer | None = None,
) -> tuple[str, object | None]:
    tokenizer: _CapturingTokenizer = tokenizer_override or (
        _CapturingTokenizer() if has_chat_template else _PlainTokenizer()
    )
    model_type: type[TextGenerationModel]
    if path_id == "transformers":
        model_type = TextGenerationModel
    elif path_id == "vllm":
        model_type = VllmModel
    else:
        assert path_id == "mlx_lm"
        model_type = MlxLmModel
    model = object.__new__(model_type)
    model._logger = getLogger(__name__)
    model._model_id = "fixture-model"
    model._tokenizer = tokenizer
    model._model = SimpleNamespace()
    message = Message(role=MessageRole.USER, content="Choose one region.")
    if path_id == "vllm":
        prompt = cast(VllmModel, model)._prompt(
            [message],
            None,
            capability=catalog,
        )
    else:
        model._tokenize_input([message], capability=catalog)
        prompt = (
            tokenizer.prompt
            if tokenizer.prompt is not None
            else dumps(tokenizer.messages, default=str, sort_keys=True)
        )
    return prompt, tokenizer.tools


def _ordinary_native_response(row: Mapping[str, Any]) -> dict[str, object]:
    transcript_format = row["transcript_format"]
    if row["id"] == "hugging_face":
        return {"choices": [{"message": {"content": "done"}}]}
    if row["id"] == "ollama":
        return {"message": {"content": "done"}}
    if transcript_format == "openai":
        return {
            "id": "response-fixture",
            "status": "completed",
            "output": [
                {
                    "id": "message-fixture",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        }
    if transcript_format == "anthropic":
        return {"content": [{"type": "text", "text": "done"}]}
    if transcript_format == "bedrock":
        return {"output": {"message": {"content": [{"text": "done"}]}}}
    if transcript_format == "google":
        return {
            "text": "done",
            "candidates": [{"content": {"parts": [{"text": "done"}]}}],
        }
    if transcript_format == "litellm":
        return {"choices": [{"message": {"content": "done"}}]}
    raise AssertionError(f"unsupported transcript format: {transcript_format}")


def _ordinary_text_response(
    row: Mapping[str, Any], text: str
) -> dict[str, object]:
    transcript_format = row["transcript_format"]
    if transcript_format == "anthropic":
        return {"content": [{"type": "text", "text": text}]}
    if transcript_format == "bedrock":
        return {"output": {"message": {"content": [{"text": text}]}}}
    if transcript_format == "google":
        return {
            "text": text,
            "candidates": [{"content": {"parts": [{"text": text}]}}],
        }
    assert transcript_format == "litellm"
    return {"choices": [{"message": {"content": text}}]}


def _openai_response_client(
    *responses: object,
) -> tuple[object, _CapturedAsyncCall]:
    create = _CapturedAsyncCall(*responses)

    class _Client:
        def __init__(self) -> None:
            self.responses = SimpleNamespace(create=create)

        def with_options(self, **_: object) -> "_Client":
            return self

        async def close(self) -> None:
            return None

    return _Client(), create


async def _hosted_adapter_call(
    row: Mapping[str, Any],
    catalog: ModelCapabilityCatalog,
    response: object,
    *,
    messages: list[Message] | None = None,
) -> tuple[dict[str, object], object, type[TextGenerationVendor]]:
    module = import_module(cast(str, row["module"]))
    adapter_type = cast(
        type[TextGenerationVendor],
        getattr(module, cast(str, row["class"])),
    )
    adapter_messages = messages or [
        Message(role=MessageRole.USER, content="Choose one region.")
    ]
    settings = GenerationSettings(
        max_new_tokens=64,
        temperature=0.0,
        use_async_generator=False,
    )

    if row["transcript_format"] == "openai":
        sdk_client, create = _openai_response_client(response)
        sdk_module = SimpleNamespace(
            AsyncOpenAI=MagicMock(return_value=sdk_client),
            Omit=type("Omit", (), {}),
        )
        base_url = (
            "https://fixture.openai.azure.com/openai/v1/"
            if row["id"] == "azure_openai"
            else "https://fixture.invalid/v1"
        )
        openai_module = import_module("avalan.model.nlp.text.vendor.openai")
        with patch.object(
            openai_module, "import_module", return_value=sdk_module
        ):
            openai_adapter = adapter_type("fixture-key", base_url)
        output = await cast(Any, openai_adapter)(
            "fixture-model",
            adapter_messages,
            settings,
            capability=catalog,
            use_async_generator=False,
        )
        assert len(create.calls) == 1
        return create.calls[0], output, adapter_type

    transcript_format = row["transcript_format"]

    if transcript_format == "anthropic":
        create = _CapturedAsyncCall(response)
        anthropic_adapter = object.__new__(adapter_type)
        anthropic_adapter._client = cast(
            Any, SimpleNamespace(messages=SimpleNamespace(create=create))
        )
        output = await anthropic_adapter(
            "fixture-model",
            adapter_messages,
            settings,
            capability=catalog,
            use_async_generator=False,
        )
        assert len(create.calls) == 1
        return create.calls[0], output, adapter_type

    if transcript_format == "bedrock":
        converse = _CapturedAsyncCall(response)
        bedrock_adapter = object.__new__(adapter_type)
        bedrock_adapter._client = SimpleNamespace(converse=converse)
        output = await bedrock_adapter(
            "fixture-model",
            adapter_messages,
            settings,
            capability=catalog,
            use_async_generator=False,
        )
        assert len(converse.calls) == 1
        return converse.calls[0], output, adapter_type

    if transcript_format == "google":
        generate = _CapturedAsyncCall(response)
        google_adapter = object.__new__(adapter_type)
        google_adapter._client = cast(
            Any,
            SimpleNamespace(
                aio=SimpleNamespace(
                    models=SimpleNamespace(generate_content=generate)
                )
            ),
        )
        output = await google_adapter(
            "fixture-model",
            adapter_messages,
            settings,
            capability=catalog,
            use_async_generator=False,
        )
        assert len(generate.calls) == 1
        return generate.calls[0], output, adapter_type

    if transcript_format == "litellm":
        completion = _CapturedAsyncCall(response)
        litellm_adapter = adapter_type(
            api_key="fixture-key",
            base_url="https://fixture.invalid/v1",
        )
        litellm_module = import_module("avalan.model.nlp.text.vendor.litellm")
        with patch.object(litellm_module.litellm, "acompletion", completion):
            output = await litellm_adapter(
                "fixture-model",
                adapter_messages,
                settings,
                capability=catalog,
                use_async_generator=False,
            )
        assert len(completion.calls) == 1
        return completion.calls[0], output, adapter_type

    if row["id"] == "hugging_face":
        complete = _CapturedAsyncCall(response)
        huggingface_adapter = object.__new__(adapter_type)
        huggingface_adapter._client = cast(
            Any, SimpleNamespace(chat_completion=complete)
        )
        output = await huggingface_adapter(
            "fixture-model",
            adapter_messages,
            settings,
            capability=catalog,
            use_async_generator=False,
        )
        assert len(complete.calls) == 1
        return complete.calls[0], output, adapter_type

    assert row["id"] == "ollama"
    chat = _CapturedAsyncCall(response)
    ollama_adapter = object.__new__(adapter_type)
    ollama_adapter._client = cast(Any, SimpleNamespace(chat=chat))
    output = await ollama_adapter(
        "fixture-model",
        adapter_messages,
        settings,
        capability=catalog,
        use_async_generator=False,
    )
    assert len(chat.calls) == 1
    return chat.calls[0], output, adapter_type


def _task_input_from_adapter_output(
    row: Mapping[str, Any],
    output: object,
    catalog: ModelCapabilityCatalog,
) -> TaskInputCapabilityCall:
    if isinstance(output, TextGenerationNonStreamResult):
        ready = next(
            event
            for event in output.events
            if event.kind is StreamItemKind.TOOL_CALL_READY
        )
        call_id = ready.correlation.tool_call_id
        assert call_id is not None
        data = ready.data
        assert isinstance(data, Mapping)
        provider_name = data.get("name")
        assert isinstance(provider_name, str)
        arguments = "".join(
            event.text_delta or ""
            for event in output.events
            if event.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            and event.correlation.tool_call_id == call_id
        )
        classification = catalog.classify_batch(
            (
                ProviderCapabilityCall(
                    call_id=call_id,
                    provider_name=provider_name,
                    arguments=arguments,
                ),
            ),
            provider_family=cast(str, row["provider_family"]),
        )
        assert isinstance(classification, CapabilityBatchAccepted)
        assert isinstance(classification.task_input, TaskInputCapabilityCall)
        return classification.task_input

    assert isinstance(output, TextGenerationSingleStream)
    parsed = catalog.parse_calls(output.content)
    assert parsed.diagnostics == []
    assert len(parsed.calls) == 1
    call = parsed.calls[0]
    classification = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id=call.id,
                provider_name=call.provider_name or call.name,
                arguments=call.arguments,
            ),
        ),
        provider_family=cast(str, row["provider_family"]),
    )
    assert isinstance(classification, CapabilityBatchAccepted)
    assert isinstance(classification.task_input, TaskInputCapabilityCall)
    return classification.task_input


def _request_native_schemas(
    row: Mapping[str, Any], request: Mapping[str, object]
) -> object | None:
    schema_format = row["schema_format"]
    if schema_format in {"openai", "anthropic", "litellm"}:
        return request.get("tools")
    if schema_format == "bedrock":
        tool_config = request.get("toolConfig")
        if not isinstance(tool_config, Mapping):
            return None
        return tool_config.get("tools")
    if schema_format == "google":
        raw_config = request.get("config")
        if not isinstance(raw_config, Mapping):
            return None
        sdk_config = GenerateContentConfig(**dict(raw_config))
        if sdk_config.tools is None:
            return None
        assert isinstance(sdk_config.tools[0], GoogleTool)
        declarations = sdk_config.tools[0].function_declarations
        assert declarations is not None
        declaration = declarations[0]
        assert declaration.parameters is None
        assert declaration.parameters_json_schema is not None
        config = _mutable_json(sdk_config)
        if not isinstance(config, Mapping):
            return None
        return config.get("tools")
    return None


def _json_digest(value: object) -> str:
    return sha256(
        dumps(
            _mutable_json(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _native_golden(row: Mapping[str, Any]) -> dict[str, Any]:
    formats = cast(
        dict[str, dict[str, Any]],
        _fixture("provider_native_goldens.json")["formats"],
    )
    return formats[cast(str, row["golden_format"])]


def _response_golden(
    row: Mapping[str, Any], *, capability_response: bool
) -> dict[str, object]:
    fixture = _fixture("provider_native_goldens.json")
    formats = cast(dict[str, dict[str, Any]], fixture["formats"])
    golden = _native_golden(row)
    response_format = cast(
        str,
        golden.get("response_format", row["golden_format"]),
    )
    key = "capability_response" if capability_response else "ordinary_response"
    response = formats[response_format][key]
    assert isinstance(response, dict)
    return cast(dict[str, object], loads(dumps(response)))


def _continuation_golden(row: Mapping[str, Any]) -> dict[str, object]:
    fixture = _fixture("provider_native_goldens.json")
    continuations = cast(
        dict[str, dict[str, object]], fixture["continuations"]
    )
    golden = _native_golden(row)
    continuation_format = cast(
        str,
        golden.get("continuation_format", row["transcript_format"]),
    )
    return continuations[continuation_format]


def _request_golden(
    row: Mapping[str, Any], *, capable: bool
) -> dict[str, object]:
    golden = _native_golden(row)
    request = dict(cast(Mapping[str, object], golden["request"]))
    if capable:
        request.update(
            cast(
                Mapping[str, object],
                golden.get("capable_request_extra", {}),
            )
        )
    return request


def _request_without_schema(
    row: Mapping[str, Any], request: Mapping[str, object]
) -> dict[str, object]:
    mutable_request = _mutable_json(request)
    assert isinstance(mutable_request, dict)
    projected = cast(dict[str, object], mutable_request)
    schema_format = row["schema_format"]
    if schema_format == "openai":
        projected.pop("tools", None)
        projected.pop("extra_headers", None)
    elif schema_format == "anthropic":
        projected.pop("tools", None)
    elif schema_format == "bedrock":
        tool_config = projected.pop("toolConfig", None)
        if isinstance(tool_config, Mapping):
            remaining = dict(tool_config)
            remaining.pop("tools", None)
            if remaining:
                projected["toolConfig"] = remaining
    elif schema_format == "google":
        config = projected.get("config")
        assert isinstance(config, Mapping)
        config_without_tools = dict(config)
        config_without_tools.pop("tools", None)
        projected["config"] = config_without_tools
    elif schema_format == "litellm":
        projected.pop("api_key", None)
        projected.pop("tools", None)
    else:
        assert schema_format == "none"
    return projected


def _local_initial_request(
    row: Mapping[str, Any], catalog: ModelCapabilityCatalog
) -> tuple[dict[str, object], object | None]:
    path_id = cast(str, row["id"])
    message = Message(role=MessageRole.USER, content="Choose one region.")
    if path_id == "ds4":
        model = object.__new__(Ds4Model)
        worker = object.__new__(Ds4Worker)
        worker._closed = False
        worker._engine = object()
        worker._tool_dsml_replay = {}
        model._model = worker
        effective = model._effective_ds4_capability(catalog)
        ds4_schema = model._tool_schemas(effective)
        _, messages = model._ds4_prompt_messages([message], None, None)
        assert len(messages) == 1
        return {
            "role": str(messages[0].role),
            "content": messages[0].content,
        }, ds4_schema

    prompt, local_schema = _local_request_serialization(
        path_id,
        catalog,
        has_chat_template=True,
    )
    messages = loads(prompt)
    assert isinstance(messages, list)
    user_messages = [
        item
        for item in messages
        if isinstance(item, dict) and item.get("role") == MessageRole.USER
    ]
    assert len(user_messages) == 1
    message_payload = user_messages[0]
    assert isinstance(message_payload, dict)
    return cast(dict[str, object], message_payload), local_schema


async def _unsupported_adapter_kwargs(
    row: Mapping[str, Any],
    catalog: ModelCapabilityCatalog,
) -> dict[str, object]:
    sdk_client = SimpleNamespace()
    if row["id"] == "hugging_face":
        huggingface_client = object.__new__(HuggingfaceClient)
        sdk_client.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "done"}}]}
        )
        huggingface_client._client = cast(Any, sdk_client)
        await huggingface_client(
            "model",
            [],
            capability=catalog,
            use_async_generator=False,
        )
        return cast(
            dict[str, object],
            sdk_client.chat_completion.call_args.kwargs,
        )
    sdk_client.chat = AsyncMock(return_value={"message": {"content": "done"}})
    ollama_client = object.__new__(OllamaClient)
    ollama_client._client = sdk_client
    await ollama_client(
        "model",
        [],
        capability=catalog,
        use_async_generator=False,
    )
    return cast(dict[str, object], sdk_client.chat.call_args.kwargs)


async def _provider_boundary_round_trip(
    row: Mapping[str, Any], *, capable: bool
) -> None:
    transcripts = _fixture("provider_transcripts.json")
    call_id = cast(str, transcripts["call_id"])
    catalog = _catalog(
        _attached_support() if capable else ProviderCapabilitySupport()
    )
    native_schemas = _native_schemas(row, catalog)
    expected_schema_digest = (
        _native_golden(row)["schema_sha256"]
        if capable and row["structured_input"]
        else None
    )
    actual_schema_digest = (
        _json_digest(native_schemas) if native_schemas is not None else None
    )
    assert actual_schema_digest == expected_schema_digest

    if row["transcript_format"] == "local":
        request, captured_schemas = _local_initial_request(row, catalog)
        assert request == _request_golden(row, capable=capable)
        assert _mutable_json(captured_schemas) == _mutable_json(native_schemas)
        response = _response_golden(
            row,
            capability_response=bool(capable and row["structured_input"]),
        )
        if capable and row["structured_input"] and row["id"] != "ds4":
            arguments = cast(dict[str, object], transcripts["request"])
            response_text = LOCAL_STRUCTURED_OUTPUT_PROTOCOL.control_frame(
                call_id,
                RESERVED_INPUT_CAPABILITY_NAME,
                arguments,
            )
        else:
            response_text = _provider_response_text(row, response, catalog)
        if not capable or not row["structured_input"]:
            assert native_schemas is None
            assert captured_schemas is None
            assert response_text == "done"
            assert catalog.parse_calls(response_text).calls == []
            return

        assert native_schemas is not None
        assert captured_schemas is not None
        assert _normalized_schema(row, native_schemas) == _fixture(
            "task_input_schema.json"
        )
        if row["id"] == "ds4":
            task_input = _task_input_from_response(row, response, catalog)
        else:
            local_output = LOCAL_STRUCTURED_OUTPUT_PROTOCOL.non_stream_result(
                response_text,
                provider_family=cast(str, row["id"]),
                provider_event_type=f"{row['id']}.generate",
            )
            task_input = _task_input_from_adapter_output(
                row,
                local_output,
                catalog,
            )
        assert task_input.call_id == call_id
        assert task_input.mode is RequirementMode.REQUIRED
        assert task_input.questions[0].question_id == "region"
        selection = cast(SingleSelectionQuestion, task_input.questions[0])
        assert selection.choices[0].value == "us-east"
        model_result = decode_input_model_result(transcripts["model_result"])
        assert isinstance(model_result, InputAnsweredResult)
        correlated = catalog.project_result(task_input, model_result)
        local_message = correlated.local_message()
        local_continuation = {
            "role": str(local_message.role),
            "name": local_message.name,
            "content": local_message.content,
        }
        assert local_continuation == _continuation_golden(row)
        assert (
            _local_serialized_payload(
                cast(str, row["id"]), local_message, catalog
            )
            == local_message.content
        )
        assert str(correlated.call_id) == call_id
        return

    response = _response_golden(
        row,
        capability_response=bool(capable and row["structured_input"]),
    )
    request, output, adapter_type = await _hosted_adapter_call(
        row, catalog, response
    )
    assert _request_without_schema(row, request) == _request_golden(
        row,
        capable=capable,
    )
    captured_schemas = _request_native_schemas(row, request)
    if not capable or not row["structured_input"]:
        assert native_schemas is None
        assert captured_schemas is None
        assert "tool_choice" not in request
        if isinstance(output, TextGenerationSingleStream):
            assert output.content == "done"
        else:
            assert isinstance(output, TextGenerationNonStreamResult)
            assert output.content == "done"
        return

    assert native_schemas is not None
    assert captured_schemas is not None
    assert _mutable_json(captured_schemas) == _mutable_json(native_schemas)
    assert _normalized_schema(row, captured_schemas) == _fixture(
        "task_input_schema.json"
    )
    task_input = _task_input_from_adapter_output(row, output, catalog)
    assert task_input.call_id == call_id
    assert task_input.mode is RequirementMode.REQUIRED
    assert task_input.questions[0].question_id == "region"
    selection = cast(SingleSelectionQuestion, task_input.questions[0])
    assert selection.choices[0].value == "us-east"
    model_result = decode_input_model_result(transcripts["model_result"])
    assert isinstance(model_result, InputAnsweredResult)
    correlated = catalog.project_result(task_input, model_result)
    continuation_method = getattr(adapter_type, "capability_result_message")
    native_continuation = cast(
        dict[str, object], continuation_method(correlated)
    )
    assert native_continuation == _continuation_golden(row)
    assert str(correlated.call_id) == call_id


def _decode_task_input(
    arguments: Mapping[str, object] | str,
    *,
    catalog: ModelCapabilityCatalog | None = None,
) -> TaskInputCapabilityCall:
    resolved_catalog = catalog or _catalog(_attached_support())
    decoded = resolved_catalog.decode_call(
        ProviderCapabilityCall(
            call_id="provider-call-001",
            provider_name=RESERVED_INPUT_CAPABILITY_NAME,
            arguments=arguments,
        )
    )
    assert isinstance(decoded, TaskInputCapabilityCall)
    return decoded


def _confirmation_question(question_id: str) -> dict[str, object]:
    return {
        "question_id": question_id,
        "kind": "confirmation",
        "prompt": f"Confirm {question_id}?",
        "required": True,
        "choices": [],
        "allow_other": False,
    }


def test_provider_matrix_matches_production_model_inventory() -> None:
    _validate_provider_matrix_completeness(_provider_rows())


@pytest.mark.parametrize(
    "path_id",
    ("transformers", "vllm", "mlx_lm"),
)
def test_local_provider_matrix_uses_one_exact_output_protocol(
    path_id: str,
) -> None:
    catalog = _catalog(_attached_support())

    prompt, schemas = _local_request_serialization(
        path_id,
        catalog,
        has_chat_template=True,
    )

    assert schemas is not None
    messages = loads(prompt)
    assert isinstance(messages, list)
    protocol_messages = [
        item
        for item in messages
        if isinstance(item, dict)
        and isinstance(item.get("content"), str)
        and LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID in item["content"]
    ]
    assert len(protocol_messages) == 1
    instruction = protocol_messages[0]["content"]
    assert isinstance(instruction, str)
    assert (
        "<tool_call id=JSON_STRING name=JSON_STRING>JSON_OBJECT</tool_call>"
        in instruction
    )
    assert '"name":"request_user_input"' in instruction
    assert "legacy JSON wrapper body" in instruction


@pytest.mark.parametrize(
    "path_id",
    ("transformers", "vllm", "mlx_lm"),
)
@pytest.mark.parametrize(
    "native_template",
    (
        "{% if tools %}{{ tools }}{% endif %}{{ messages }}",
        {"tool_use": "{{ tools }}{{ messages }}"},
        {LOCAL_STRUCTURED_OUTPUT_TEMPLATE_NAME: " "},
    ),
    ids=("native-tools", "unknown-adapter", "empty-exact-adapter"),
)
def test_local_provider_matrix_omits_unknown_tokenizer_protocols(
    path_id: str,
    native_template: object,
) -> None:
    catalog = _catalog(_attached_support())
    tokenizer = _CapturingTokenizer()
    tokenizer.chat_template = native_template

    prompt, schemas = _local_request_serialization(
        path_id,
        catalog,
        has_chat_template=True,
        tokenizer_override=tokenizer,
    )

    assert schemas is None
    assert LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID not in prompt
    assert RESERVED_INPUT_CAPABILITY_NAME not in prompt


@pytest.mark.parametrize("path_id", ("anthropic", "ds4"))
def test_provider_matrix_rejects_omitted_production_path(path_id: str) -> None:
    incomplete = tuple(row for row in _provider_rows() if row["id"] != path_id)

    with pytest.raises(
        AssertionError, match="provider matrix path IDs differ"
    ):
        _validate_provider_matrix_completeness(incomplete)


@pytest.mark.parametrize(
    ("row", "capable"),
    tuple(
        (row, capable) for row in _provider_rows() for capable in (True, False)
    ),
    ids=tuple(
        f"{row['id']}-{'capable' if capable else 'incapable'}"
        for row in _provider_rows()
        for capable in (True, False)
    ),
)
def test_requirement_input_n_022(
    row: Mapping[str, Any], capable: bool
) -> None:
    """Round-trip one exact model path or prove explicit omission."""
    golden_schema = _fixture("task_input_schema.json")
    module = import_module(cast(str, row["module"]))
    adapter = getattr(module, cast(str, row["class"]))
    assert isinstance(adapter, type)
    assert row["capable_coverage"] in {"attached", "omitted"}
    assert row["incapable_coverage"] == "omitted"
    assert row["real_durable_advertisement"] is False
    catalog = _catalog(
        _attached_support() if capable else ProviderCapabilitySupport()
    )
    native_schemas = _native_schemas(row, catalog)
    if capable and row["structured_input"]:
        assert native_schemas is not None
        assert _normalized_schema(row, native_schemas) == golden_schema
    else:
        assert native_schemas is None

    run(_provider_boundary_round_trip(row, capable=capable))


def _hosted_non_stream_provenance_rows() -> tuple[dict[str, Any], ...]:
    included = {"anthropic", "bedrock", "google", "litellm"}
    return tuple(row for row in _provider_rows() if row["id"] in included)


async def _hosted_non_stream_orchestrator_round_trip(
    row: Mapping[str, Any],
) -> None:
    catalog = _catalog(_attached_support())
    manager = _manager()
    response = _response_golden(row, capability_response=True)
    _, output, adapter_type = await _hosted_adapter_call(
        row, catalog, response
    )
    assert isinstance(output, TextGenerationNonStreamResult)
    confirmation = AsyncMock(return_value=True)

    with (
        patch.object(
            manager,
            "describe_tool_call",
            wraps=manager.describe_tool_call,
        ) as describe,
        patch.object(
            manager,
            "validate_tool_call",
            wraps=manager.validate_tool_call,
        ) as validate,
        patch.object(
            manager,
            "prepare_call",
            new_callable=AsyncMock,
        ) as prepare,
        patch.object(
            manager,
            "execute_call",
            new_callable=AsyncMock,
        ) as execute,
    ):
        orchestrator = _boundary_orchestrator_response(
            output,
            provider_family=cast(str, row["provider_family"]),
            manager=manager,
            catalog=catalog,
            confirmation=confirmation,
            enable_tool_parsing=True,
        )
        items = await _consume_boundary_response(orchestrator)

    describe.assert_not_called()
    validate.assert_not_called()
    prepare.assert_not_awaited()
    execute.assert_not_awaited()
    _assert_reserved_call_side_effects_absent(
        orchestrator,
        items,
        confirmation,
    )
    call = orchestrator.task_input_call
    assert isinstance(call, TaskInputCapabilityCall)
    assert call.call_id == "provider-call-001"
    assert call.mode is RequirementMode.REQUIRED
    assert call.questions[0].question_id == "region"

    model_result = decode_input_model_result(
        _fixture("provider_transcripts.json")["model_result"]
    )
    assert isinstance(model_result, InputAnsweredResult)
    assert model_result.provenance is AnswerProvenance.HUMAN
    correlated = catalog.project_result(call, model_result)
    assert isinstance(correlated, CorrelatedCapabilityResult)
    assert correlated.call_id == "provider-call-001"
    payload = correlated.provider_payload()
    assert payload["provenance"] == AnswerProvenance.HUMAN.value
    answers = payload["answers"]
    assert isinstance(answers, list)
    assert answers[0]["provenance"] == AnswerProvenance.HUMAN.value
    continuation = cast(
        dict[str, object],
        adapter_type.capability_result_message(correlated),
    )
    assert continuation == _continuation_golden(row)
    serialized_continuation = dumps(
        continuation,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert "provider-call-001" in serialized_continuation
    assert AnswerProvenance.HUMAN.value in serialized_continuation


@pytest.mark.parametrize(
    "row",
    _hosted_non_stream_provenance_rows(),
    ids=lambda row: cast(str, row["id"]),
)
def test_hosted_non_stream_native_call_reaches_orchestrator_with_provenance(
    row: Mapping[str, Any],
) -> None:
    run(_hosted_non_stream_orchestrator_round_trip(row))


async def _hosted_non_stream_text_is_untrusted(
    row: Mapping[str, Any],
    *,
    exact_control_frame: bool,
) -> None:
    catalog = _catalog(_attached_support())
    manager = _manager()
    request = cast(
        dict[str, object],
        _fixture("provider_transcripts.json")["request"],
    )
    if exact_control_frame:
        text = TextGenerationVendor.build_tool_call_text(
            "prose-provider-call",
            RESERVED_INPUT_CAPABILITY_NAME,
            request,
            tool_name_is_canonical=True,
        )
    else:
        text = (
            "The model mentions request_user_input and shows ambiguous "
            'JSON: {"name":"request_user_input","questions":[]}.'
        )
    response = _ordinary_text_response(row, text)
    _, output, _ = await _hosted_adapter_call(row, catalog, response)
    assert isinstance(output, TextGenerationSingleStream)
    confirmation = AsyncMock(return_value=True)

    with (
        patch.object(
            manager,
            "describe_tool_call",
            wraps=manager.describe_tool_call,
        ) as describe,
        patch.object(
            manager,
            "validate_tool_call",
            wraps=manager.validate_tool_call,
        ) as validate,
        patch.object(
            manager,
            "prepare_call",
            new_callable=AsyncMock,
        ) as prepare,
        patch.object(
            manager,
            "execute_call",
            new_callable=AsyncMock,
        ) as execute,
    ):
        orchestrator = _boundary_orchestrator_response(
            output,
            provider_family=cast(str, row["provider_family"]),
            manager=manager,
            catalog=catalog,
            confirmation=confirmation,
            enable_tool_parsing=True,
        )
        items = await _consume_boundary_response(orchestrator)

    describe.assert_not_called()
    validate.assert_not_called()
    prepare.assert_not_awaited()
    execute.assert_not_awaited()
    _assert_reserved_call_side_effects_absent(
        orchestrator,
        items,
        confirmation,
    )
    assert orchestrator.task_input_call is None
    diagnostics = [
        cast(dict[str, object], item.data)
        for item in orchestrator.canonical_items
        if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        and isinstance(item.data, dict)
    ]
    if exact_control_frame:
        assert {diagnostic["code"] for diagnostic in diagnostics} == {
            CapabilityBatchRejectionCode.NON_STRUCTURED_CALL.value
        }
    else:
        assert diagnostics == []


@pytest.mark.parametrize(
    ("row", "exact_control_frame"),
    tuple(
        (row, exact)
        for row in _hosted_non_stream_provenance_rows()
        for exact in (True, False)
    ),
    ids=lambda value: (
        cast(str, value["id"])
        if isinstance(value, Mapping)
        else "serialized" if value else "ambiguous"
    ),
)
def test_hosted_non_stream_text_cannot_forge_trusted_capability_call(
    row: Mapping[str, Any],
    exact_control_frame: bool,
) -> None:
    run(
        _hosted_non_stream_text_is_untrusted(
            row,
            exact_control_frame=exact_control_frame,
        )
    )


async def _local_non_stream_orchestrator_round_trip(
    path_id: str,
    *,
    generated_text: str | None = None,
    producer_call: tuple[str, str, Mapping[str, object]] | None = None,
    expect_call: bool,
) -> None:
    assert (generated_text is None) != (producer_call is None)
    catalog = _catalog(_attached_support())
    manager = _manager()

    def protocol_output() -> str:
        assert producer_call is not None
        protocol_messages: list[dict[str, object]] = []
        for item in tokenizer.messages:
            content = item.get("content")
            if (
                isinstance(content, str)
                and LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID in content
            ):
                protocol_messages.append(item)
        assert len(protocol_messages) == 1
        instruction = protocol_messages[0]["content"]
        assert isinstance(instruction, str)
        assert (
            "<tool_call id=JSON_STRING name=JSON_STRING>"
            "JSON_OBJECT</tool_call>"
            in instruction
        )
        call_id, name, arguments = producer_call
        assert f'"name":"{name}"' in instruction
        return LOCAL_STRUCTURED_OUTPUT_PROTOCOL.control_frame(
            call_id,
            name,
            arguments,
        )

    tokenizer = _BoundaryTokenizer(
        protocol_output
        if producer_call is not None
        else cast(str, generated_text)
    )
    engine_settings = TransformerEngineSettings(
        auto_load_model=False,
        auto_load_tokenizer=False,
    )
    generation_settings = GenerationSettings(
        max_new_tokens=64,
        temperature=0.0,
        use_async_generator=False,
    )
    transport_call: MagicMock
    model: TextGenerationModel

    with ExitStack() as stack:
        if path_id == "transformers":
            model = TextGenerationModel(
                "fixture-model",
                engine_settings,
                getLogger(__name__),
            )
            transport_call = MagicMock(
                return_value=[[101, 102, 999]],
            )
            model._model = SimpleNamespace(generate=transport_call)
        elif path_id == "vllm":
            model = VllmModel(
                "fixture-model",
                engine_settings,
                getLogger(__name__),
            )
            transport_call = MagicMock(
                side_effect=lambda *_args, **_kwargs: [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                text=(
                                    protocol_output()
                                    if producer_call is not None
                                    else generated_text
                                )
                            )
                        ]
                    )
                ]
            )
            model._model = SimpleNamespace(generate=transport_call)
            stack.enter_context(
                patch(
                    "avalan.model.nlp.text.vllm._sampling_params_class",
                    return_value=MagicMock(),
                )
            )
        elif path_id == "mlx_lm":
            model = MlxLmModel(
                "fixture-model",
                engine_settings,
                getLogger(__name__),
            )
            model._model = object()
            transport_call = MagicMock(
                side_effect=lambda *_args, **_kwargs: (
                    protocol_output()
                    if producer_call is not None
                    else generated_text
                )
            )
            stack.enter_context(
                patch(
                    "avalan.model.nlp.text.mlxlm._require_mlx_lm",
                    return_value=SimpleNamespace(generate=transport_call),
                )
            )
            stack.enter_context(
                patch(
                    "avalan.model.nlp.text.mlxlm.make_sampler",
                    return_value=object(),
                )
            )
        else:
            assert path_id == "ds4"
            model = Ds4Model(
                "fixture-model",
                engine_settings,
                getLogger(__name__),
            )
            worker = object.__new__(Ds4Worker)
            worker._closed = False
            worker._engine = object()
            worker._logger = getLogger(__name__)
            worker._tool_dsml_replay = {}
            cast(Any, worker).exact_dsml_for_tool_calls = MagicMock(
                return_value=None
            )
            cast(Any, worker).tokenize_rendered_chat_async = AsyncMock(
                return_value=[101, 102]
            )
            if expect_call:
                request = cast(
                    dict[str, object],
                    _fixture("provider_transcripts.json")["request"],
                )
                assert producer_call is not None
                call_id, name, _ = producer_call
                worker_result = (
                    TextGenerationNonStreamResult.from_provider_parts(
                        answer_text="",
                        calls=(
                            TextGenerationNonStreamToolCall(
                                call_id=call_id,
                                name=name,
                                arguments=dumps(
                                    request,
                                    separators=(",", ":"),
                                    sort_keys=True,
                                ),
                                provider_event_type="ds4.dsml.tool_call",
                            ),
                        ),
                        provider_family="ds4",
                        answer_event_type="ds4.text",
                        terminal_event_type="ds4.completed",
                    )
                )
            else:
                assert generated_text is not None
                worker_result = TextGenerationNonStreamResult.from_events(
                    (
                        StreamProviderEvent(
                            kind=StreamItemKind.ANSWER_DELTA,
                            text_delta=generated_text,
                            metadata={
                                NATIVE_STRUCTURED_OUTPUT_METADATA_KEY: True
                            },
                            provider_event_type="ds4.text",
                        ),
                    ),
                    provider_family="ds4",
                    terminal_event_type="ds4.completed",
                )
            transport_call = AsyncMock(return_value=worker_result)
            cast(Any, worker).generate_non_stream_result_async = transport_call
            model._model = worker

        model._tokenizer = tokenizer
        response = await model(
            Message(role=MessageRole.USER, content="Choose one region."),
            settings=generation_settings,
            capability=catalog,
        )
        assert isinstance(response, TextGenerationResponse)
        confirmation = AsyncMock(return_value=True)
        describe = stack.enter_context(
            patch.object(
                manager,
                "describe_tool_call",
                wraps=manager.describe_tool_call,
            )
        )
        validate = stack.enter_context(
            patch.object(
                manager,
                "validate_tool_call",
                wraps=manager.validate_tool_call,
            )
        )
        prepare = stack.enter_context(
            patch.object(
                manager,
                "prepare_call",
                new_callable=AsyncMock,
            )
        )
        execute = stack.enter_context(
            patch.object(
                manager,
                "execute_call",
                new_callable=AsyncMock,
            )
        )
        orchestrator = _boundary_orchestrator_response(
            response,
            provider_family=("mlx" if path_id == "mlx_lm" else path_id),
            manager=manager,
            catalog=catalog,
            confirmation=confirmation,
            enable_tool_parsing=True,
        )
        items = await _consume_boundary_response(orchestrator)
        if isinstance(model, MlxLmModel):
            model.close()

    transport_call.assert_called_once()
    describe.assert_not_called()
    validate.assert_not_called()
    prepare.assert_not_awaited()
    execute.assert_not_awaited()
    _assert_reserved_call_side_effects_absent(
        orchestrator,
        items,
        confirmation,
    )
    if expect_call:
        call = orchestrator.task_input_call
        assert isinstance(call, TaskInputCapabilityCall)
        assert producer_call is not None
        expected_call_id, expected_name, _ = producer_call
        assert call.call_id == expected_call_id
        assert call.provider_name == expected_name
        assert call.questions[0].question_id == "region"
        model_result = decode_input_model_result(
            _fixture("provider_transcripts.json")["model_result"]
        )
        assert isinstance(model_result, InputAnsweredResult)
        assert model_result.provenance is AnswerProvenance.HUMAN
        correlated = catalog.project_result(call, model_result)
        assert correlated.call_id == expected_call_id
        assert correlated.provider_payload()["provenance"] == "human"
        local_message = correlated.local_message()
        assert local_message.role is MessageRole.TOOL
        serialized = loads(cast(str, local_message.content))
        assert serialized["call_id"] == expected_call_id
        assert serialized["name"] == expected_name
        assert serialized["result"]["provenance"] == "human"
    else:
        assert orchestrator.task_input_call is None
        assert generated_text is not None
        assert (
            "".join(
                item.text_delta or ""
                for item in orchestrator.canonical_items
                if item.kind is StreamItemKind.ANSWER_DELTA
            )
            == generated_text
        )
        assert not any(
            item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            }
            for item in orchestrator.canonical_items
        )
        diagnostics = [
            cast(dict[str, object], item.data)
            for item in orchestrator.canonical_items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
            and isinstance(item.data, dict)
        ]
        assert diagnostics == []


@pytest.mark.parametrize(
    "path_id",
    ("transformers", "vllm", "mlx_lm", "ds4"),
)
def test_local_non_stream_native_control_reaches_orchestrator(
    path_id: str,
) -> None:
    request = cast(
        dict[str, object],
        _fixture("provider_transcripts.json")["request"],
    )
    call_id = 'local-provider-"quoted"-\\-001'
    run(
        _local_non_stream_orchestrator_round_trip(
            path_id,
            producer_call=(
                call_id,
                RESERVED_INPUT_CAPABILITY_NAME,
                request,
            ),
            expect_call=True,
        )
    )


@pytest.mark.parametrize(
    "path_id",
    ("transformers", "vllm", "mlx_lm", "ds4"),
)
def test_local_non_stream_ambiguous_prose_stays_non_executable(
    path_id: str,
) -> None:
    run(
        _local_non_stream_orchestrator_round_trip(
            path_id,
            generated_text=(
                "The answer mentions request_user_input with ambiguous "
                'JSON {"name":"request_user_input","questions":[]}.'
            ),
            expect_call=False,
        )
    )


@pytest.mark.parametrize(
    "path_id",
    ("transformers", "vllm", "mlx_lm", "ds4"),
)
def test_local_non_stream_legacy_serialization_is_untrusted(
    path_id: str,
) -> None:
    request = cast(
        dict[str, object],
        _fixture("provider_transcripts.json")["request"],
    )
    legacy_text = TextGenerationVendor.build_tool_call_text(
        "legacy-local-call-001",
        RESERVED_INPUT_CAPABILITY_NAME,
        request,
        tool_name_is_canonical=True,
    )
    run(
        _local_non_stream_orchestrator_round_trip(
            path_id,
            generated_text=legacy_text,
            expect_call=False,
        )
    )


async def _unclassified_local_text_round_trip(
    *, capable: bool
) -> tuple[OrchestratorResponse, list[object], AsyncMock, str]:
    request = cast(
        dict[str, object],
        _fixture("provider_transcripts.json")["request"],
    )
    legacy_text = TextGenerationVendor.build_tool_call_text(
        "unclassified-local-call-001",
        RESERVED_INPUT_CAPABILITY_NAME,
        request,
        tool_name_is_canonical=True,
    )
    output = TextGenerationNonStreamResult.from_events(
        (
            StreamProviderEvent(
                kind=StreamItemKind.ANSWER_DELTA,
                text_delta=legacy_text,
                provider_event_type="local.unclassified_text",
            ),
        ),
        provider_family="ds4",
        terminal_event_type="local.completed",
    )
    confirmation = AsyncMock(return_value=True)
    orchestrator = _boundary_orchestrator_response(
        output,
        provider_family="ds4",
        manager=_manager(),
        catalog=_catalog(
            _attached_support() if capable else ProviderCapabilitySupport()
        ),
        confirmation=confirmation,
        enable_tool_parsing=True,
    )
    items = await _consume_boundary_response(orchestrator)
    return orchestrator, items, confirmation, legacy_text


def test_unclassified_local_text_retains_generic_parser_guard() -> None:
    orchestrator, items, confirmation, _ = run(
        _unclassified_local_text_round_trip(capable=True)
    )

    _assert_reserved_call_side_effects_absent(
        orchestrator, items, confirmation
    )
    assert orchestrator.task_input_call is None
    assert {
        cast(dict[str, object], item.data)["code"]
        for item in orchestrator.canonical_items
        if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        and isinstance(item.data, dict)
    } == {CapabilityBatchRejectionCode.NON_STRUCTURED_CALL.value}


def test_incapable_local_text_retains_generic_parser_guard() -> None:
    orchestrator, items, confirmation, _ = run(
        _unclassified_local_text_round_trip(capable=False)
    )

    _assert_reserved_call_side_effects_absent(
        orchestrator, items, confirmation
    )
    assert orchestrator.task_input_call is None
    assert {
        cast(dict[str, object], item.data)["code"]
        for item in orchestrator.canonical_items
        if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        and isinstance(item.data, dict)
    } == {CapabilityBatchRejectionCode.MALFORMED_CALL.value}


def test_requirement_input_n_023() -> None:
    """Advertise only when exact attached or durable prerequisites hold."""
    incapable = _catalog(ProviderCapabilitySupport())
    sessionless = _catalog(
        ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
        )
    )
    attached = _catalog(_attached_support())
    missing_codec = _catalog(
        ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
        )
    )
    durable_support, durable_binding = _durable_contract()
    fake_durable = _catalog(
        durable_support,
        revision_binding=durable_binding,
    )
    missing_binding = _catalog(durable_support)
    wrong_binding = _catalog(
        durable_support,
        revision_binding=ContinuationRevisionBinding(
            provider_family=ProviderFamilyName("fake-provider"),
            model_id=ModelId("different-model"),
            provider_config_revision=ProviderConfigRevision(
                "provider-config-v1"
            ),
            model_config_revision=ModelConfigRevision("model-config-v1"),
            capability_revision=CapabilityRevision("capability-v1"),
        ),
    )

    assert incapable.task_input_advertisement is (
        TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert sessionless.task_input_advertisement is (
        TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert missing_codec.task_input_advertisement is (
        TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert attached.task_input_advertisement is (
        TaskInputCapabilityAdvertisement.ATTACHED
    )
    assert fake_durable.task_input_advertisement is (
        TaskInputCapabilityAdvertisement.DURABLE
    )
    assert incapable.descriptors == sessionless.descriptors == ()
    assert missing_codec.descriptors == ()
    assert missing_binding.descriptors == ()
    assert wrong_binding.descriptors == ()
    assert attached.descriptors[0].kind is ModelCapabilityKind.TASK_INPUT
    assert fake_durable.descriptors[0].kind is ModelCapabilityKind.TASK_INPUT
    assert not fake_durable.project("fake-provider").is_empty
    assert fake_durable.project("different-provider").is_empty

    empty_registry = ContinuationSnapshotCodecRegistry("empty-registry")
    with pytest.raises(ModelCapabilityValidationError) as missing_reference:
        empty_registry.reference("missing-codec")
    assert missing_reference.value.code == "capability.continuation_codec"
    with pytest.raises(AssertionError):
        ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
            continuation_snapshot_codec=cast(Any, True),
        )
    assert all(
        row["real_durable_advertisement"] is False for row in _provider_rows()
    )


@pytest.mark.parametrize(
    ("path_id", "condition"),
    (
        ("transformers", "no_chat_template"),
        ("transformers", "parser_disabled"),
        ("vllm", "no_chat_template"),
        ("vllm", "parser_disabled"),
        ("mlx_lm", "no_chat_template"),
        ("mlx_lm", "parser_disabled"),
        ("ds4", "unsupported_capability"),
    ),
    ids=(
        "transformers-no-chat-template",
        "transformers-parser-disabled",
        "vllm-no-chat-template",
        "vllm-parser-disabled",
        "mlx-lm-no-chat-template",
        "mlx-lm-parser-disabled",
        "ds4-unsupported-capability",
    ),
)
def test_local_capability_support_matrix(path_id: str, condition: str) -> None:
    """Bind unsupported local serializer conditions to exact instances."""
    if path_id == "ds4":
        capable = _catalog(_attached_support())
        model = object.__new__(Ds4Model)
        model._model = object()
        effective = model._effective_ds4_capability(capable)
        assert effective is None
        assert model._tool_schemas(effective) is None
        assert not model._uses_dsml_tools("Choose one region.", effective)
        return

    if condition == "no_chat_template":
        catalog = _catalog(_attached_support())
        prompt, tools = _local_request_serialization(
            path_id,
            catalog,
            has_chat_template=False,
        )
        assert catalog.structured_parser_enabled
    else:
        assert condition == "parser_disabled"
        catalog = _catalog(ProviderCapabilitySupport())
        prompt, tools = _local_request_serialization(
            path_id,
            catalog,
            has_chat_template=True,
        )
        assert not catalog.structured_parser_enabled
    assert tools is None
    assert RESERVED_INPUT_CAPABILITY_NAME not in prompt


def test_requirement_input_n_024() -> None:
    """Keep the golden request to one focused question."""
    request = cast(
        dict[str, object],
        _fixture("provider_transcripts.json")["request"],
    )
    call = _decode_task_input(request)

    assert len(call.questions) == 1
    assert call.questions[0].question_id == "region"


def test_requirement_input_n_025() -> None:
    """Accept one closely related bundle of at most three questions."""
    call = _decode_task_input(
        {
            "mode": "advisory",
            "reason": "Choose one coordinated rollout policy.",
            "questions": [
                _confirmation_question("deploy"),
                _confirmation_question("monitor"),
                _confirmation_question("rollback"),
            ],
        }
    )

    assert call.mode is RequirementMode.ADVISORY
    assert tuple(question.question_id for question in call.questions) == (
        "deploy",
        "monitor",
        "rollback",
    )


def test_requirement_input_n_026() -> None:
    """Reject provider requests containing more than three questions."""
    with pytest.raises(ModelCapabilityValidationError) as error:
        _decode_task_input(
            {
                "mode": "required",
                "reason": "Too many decisions.",
                "questions": [
                    _confirmation_question(f"decision-{index}")
                    for index in range(4)
                ],
            }
        )

    assert error.value.code == "capability.schema_validation"


@pytest.mark.parametrize(
    "field,value",
    (
        ("request_id", "model-chosen-request"),
        ("secret", True),
        ("authentication", {"kind": "password"}),
        ("interaction_class", "approval"),
        ("approval", {"operation": "deploy"}),
    ),
    ids=(
        "request-id",
        "secret",
        "authentication",
        "interaction-class",
        "approval",
    ),
)
def test_reserved_schema_rejects_control_and_secret_injection(
    field: str,
    value: object,
) -> None:
    request = cast(
        dict[str, object],
        _fixture("provider_transcripts.json")["request"],
    ).copy()
    request[field] = value

    with pytest.raises(ModelCapabilityValidationError) as error:
        _decode_task_input(request)

    assert error.value.code == "capability.schema_validation"


def test_requirement_input_n_027() -> None:
    """Fail closed before actions, approval, input, or provenance forgery."""
    catalog = _catalog(_attached_support(), with_domain_tool=True)
    request = cast(
        dict[str, object],
        _fixture("provider_transcripts.json")["request"],
    )
    reserved = ProviderCapabilityCall(
        call_id="input-call",
        provider_name=RESERVED_INPUT_CAPABILITY_NAME,
        arguments=request,
    )
    domain = ProviderCapabilityCall(
        call_id="domain-call",
        provider_name="lookup",
        arguments={"query": "value"},
    )

    mixed = catalog.classify_batch((domain, reserved))
    multiple = catalog.classify_batch((reserved, reserved))
    malformed = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id="input-call",
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments="{",
            ),
        )
    )
    missing_id = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id=None,
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=request,
            ),
        )
    )
    non_structured = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id="input-call",
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=request,
                structured=False,
            ),
        )
    )
    unknown = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id="unknown-call",
                provider_name="unadvertised_capability",
                arguments={},
            ),
        )
    )
    oversized = "x" * 131_073
    with pytest.raises(ModelCapabilityValidationError) as size_error:
        _decode_task_input(oversized)
    assert size_error.value.code == "capability.arguments_size"

    assert isinstance(mixed, CapabilityBatchRejected)
    assert mixed.code is CapabilityBatchRejectionCode.MIXED_TASK_INPUT_BATCH
    assert isinstance(multiple, CapabilityBatchRejected)
    assert multiple.code is (
        CapabilityBatchRejectionCode.MULTIPLE_TASK_INPUT_CALLS
    )
    assert isinstance(malformed, CapabilityBatchRejected)
    assert malformed.code is CapabilityBatchRejectionCode.MALFORMED_CALL
    assert isinstance(missing_id, CapabilityBatchRejected)
    assert missing_id.code is CapabilityBatchRejectionCode.MISSING_CALL_ID
    assert isinstance(non_structured, CapabilityBatchRejected)
    assert non_structured.code is (
        CapabilityBatchRejectionCode.NON_STRUCTURED_CALL
    )
    assert isinstance(unknown, CapabilityBatchRejected)
    assert unknown.code is CapabilityBatchRejectionCode.UNKNOWN_CAPABILITY
    for prose in (
        "Please ask the user for a region before continuing.",
        "request_user_input should probably ask which region to use.",
        (
            "The local model mentions <tool_call>request_user_input"
            "</tool_call> without a structured payload."
        ),
        '```json\n{"name": "request_user_input", "questions": []}\n```',
    ):
        assert catalog.parse_calls(prose).calls == []

    with pytest.raises(ValueError, match="reserved model capability name"):
        ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[request_user_input])],
            enable_tools=[RESERVED_INPUT_CAPABILITY_NAME],
        )

    result = decode_input_model_result(
        _fixture("provider_transcripts.json")["model_result"]
    )
    assert isinstance(result, InputAnsweredResult)
    assert result.provenance is AnswerProvenance.HUMAN
    assert result.answers[0].provenance is AnswerProvenance.HUMAN
    timed_out = decode_input_model_result(
        {
            "kind": "timed_out",
            "request_id": "input-request-001",
            "provenance": "policy",
            "resolved_at": "2026-07-21T12:05:00.000000Z",
        }
    )
    timed_out_projection = catalog.project_result(
        _decode_task_input(request, catalog=catalog),
        timed_out,
    )
    assert timed_out_projection.payload["provenance"] == "policy"


def test_requirement_input_n_028() -> None:
    """Keep task-input accounting separate from domain tool execution."""
    calls = 0

    def counted_lookup(query: str) -> str:
        """Look up and count one value.

        Args:
            query: Value to look up.

        Returns:
            The provided value.
        """
        nonlocal calls
        calls += 1
        return query

    manager = _manager(
        tools=[counted_lookup],
        enable_tools=["pkg.counted_lookup"],
    )
    catalog = ModelCapabilityCatalog.create(
        manager.export_model_capability_seed(),
        support=_attached_support(),
    )
    task_input = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id="input-call",
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=cast(
                    dict[str, object],
                    _fixture("provider_transcripts.json")["request"],
                ),
            ),
        )
    )
    assert isinstance(task_input, CapabilityBatchAccepted)
    assert task_input.domain_calls == ()
    assert task_input.task_input is not None
    assert calls == 0

    domain_name = catalog.project().provider_name("pkg.counted_lookup")
    domain_batch = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id="domain-call",
                provider_name=domain_name,
                arguments={"query": "value"},
            ),
        )
    )
    assert isinstance(domain_batch, CapabilityBatchAccepted)
    assert domain_batch.task_input is None
    assert len(domain_batch.domain_calls) == 1
    context = ToolCallContext(input="lookup")
    outcome = run(manager(domain_batch.domain_calls[0], context))
    assert isinstance(outcome, ToolCallResult)
    assert outcome.result == "value"
    assert calls == 1


def test_model_layer_tool_manager_references_match_documented_allowlist() -> (
    None
):
    completed = run_process(
        ("rg", "-n", r"\bToolManager\b", "src/avalan/model"),
        cwd=_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0
    references = {
        (path, content.strip())
        for line in completed.stdout.splitlines()
        for path, _, content in (line.split(":", 2),)
    }
    assert references == {
        (
            "src/avalan/model/capability.py",
            (
                '"""Carry a callable-free ToolManager export into the model '
                'layer."""'
            ),
        ),
        (
            "src/avalan/model/capability.py",
            (
                '"""Decode one strict callable-free ToolManager export '
                'payload."""'
            ),
        ),
    }

    for source_path in (_ROOT / "src" / "avalan" / "model").rglob("*.py"):
        tree = parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, Name) and node.id == "ToolManager"
            for node in walk(tree)
        ), source_path
        for node in walk(tree):
            if isinstance(node, Import | ImportFrom):
                assert all(alias.name != "ToolManager" for alias in node.names)
