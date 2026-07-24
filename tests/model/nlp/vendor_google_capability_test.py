from collections.abc import AsyncIterator, Mapping
from json import loads
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentConfig,
    GenerateContentResponse,
    Part,
    Tool,
)

from avalan.entities import (
    GenerationSettings,
    Message,
    MessageRole,
    MessageToolCall,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
)
from avalan.model import (
    CorrelatedCapabilityResult,
    DomainCapabilitySeed,
    ModelCapabilityCatalog,
    ModelCapabilityDescriptor,
)
from avalan.model.nlp.text.vendor.google import GoogleClient, GoogleStream
from avalan.model.provider import ProviderFamily
from avalan.model.stream import (
    StreamItemKind,
    TextGenerationNonStreamResult,
)
from avalan.types import JsonValue


def _parameter_schema() -> Mapping[str, JsonValue]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "value": {
                "oneOf": (
                    {"type": "string"},
                    {"type": "integer"},
                )
            }
        },
        "required": ("value",),
        "additionalProperties": False,
    }


def _catalog() -> ModelCapabilityCatalog:
    return ModelCapabilityCatalog.create(
        DomainCapabilitySeed(
            descriptors=(
                ModelCapabilityDescriptor(
                    canonical_name="schema.validate",
                    description="Validate one structured value.",
                    parameter_schema=_parameter_schema(),
                ),
            )
        )
    )


async def _responses(
    *responses: GenerateContentResponse,
) -> AsyncIterator[GenerateContentResponse]:
    for response in responses:
        yield response


def _native_response(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, object] | None = None,
    text: str | None = None,
) -> GenerateContentResponse:
    parts = [] if text is None else [Part(text=text)]
    parts.append(
        Part(
            function_call=FunctionCall(
                id=call_id,
                name=name,
                args=arguments,
            )
        )
    )
    return GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=parts))]
    )


def test_google_config_uses_sdk_json_schema_field_for_full_schema() -> None:
    parameter_schema = _parameter_schema()
    capability = _catalog()
    client = GoogleClient("test-api-key")

    raw_config = client._config(
        "gemini-test",
        [Message(role=MessageRole.USER, content="Validate this value.")],
        GenerationSettings(tool_choice="schema.validate"),
        capability=capability,
    )

    assert raw_config is not None
    declaration = raw_config["tools"][0]["function_declarations"][0]
    assert "parameters" not in declaration
    projected_schema = declaration["parameters_json_schema"]
    assert projected_schema["$schema"] == parameter_schema["$schema"]
    assert projected_schema["properties"]["value"]["oneOf"] == [
        {"type": "string"},
        {"type": "integer"},
    ]

    config = GenerateContentConfig(**raw_config)
    assert config.tools is not None
    assert isinstance(config.tools[0], Tool)
    declarations = config.tools[0].function_declarations
    assert declarations is not None
    sdk_declaration = declarations[0]
    assert sdk_declaration.parameters is None
    assert sdk_declaration.parameters_json_schema == projected_schema
    assert raw_config["tool_config"] == {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [
                capability.provider_name(
                    "schema.validate",
                    provider_family=ProviderFamily.GOOGLE,
                )
            ],
        }
    }


def test_google_config_skips_non_function_projection_entries() -> None:
    projection = SimpleNamespace(
        is_empty=False,
        schemas=(
            {"type": "object"},
            {"type": "function", "function": "invalid"},
        ),
    )
    capability_mock = MagicMock(spec=ModelCapabilityCatalog)
    capability_mock.project.return_value = projection

    config = GoogleClient("test-api-key")._config(
        "gemini-test",
        [Message(role=MessageRole.USER, content="Validate this value.")],
        None,
        capability=cast(ModelCapabilityCatalog, capability_mock),
    )

    assert config == {"tools": [{"function_declarations": []}]}


def test_google_function_call_extraction_handles_native_and_camel_frames() -> (
    None
):
    native = _native_response(
        call_id="native-call",
        name="native_tool",
        arguments={"value": 1},
    )
    assert GoogleClient._function_calls(native) == (
        ("native-call", "native_tool", {"value": 1}),
    )

    camel_response = {
        "candidates": [
            {"content": {"parts": None}},
            {
                "content": {
                    "parts": [
                        {"text": "ignore"},
                        {
                            "functionCall": {
                                "id": "camel-call",
                                "name": "camel_tool",
                                "args": None,
                            }
                        },
                    ]
                }
            },
        ]
    }
    assert GoogleClient._function_calls(camel_response) == (
        ("camel-call", "camel_tool", {}),
    )


@pytest.mark.parametrize(
    ("call", "message"),
    (
        ({"id": "", "name": "tool", "args": {}}, "id must be"),
        ({"id": "call", "name": "", "args": {}}, "name must be"),
        ({"id": "call", "name": "tool", "args": []}, "arguments must"),
    ),
)
def test_google_function_call_extraction_rejects_invalid_native_frames(
    call: dict[str, object],
    message: str,
) -> None:
    response = {
        "candidates": [{"content": {"parts": [{"function_call": call}]}}]
    }

    with pytest.raises(ValueError, match=message):
        GoogleClient._function_calls(response)


def test_google_continuation_messages_preserve_names_ids_and_diagnostics() -> (
    None
):
    capability = _catalog()
    provider_name = capability.provider_name(
        "schema.validate", provider_family=ProviderFamily.GOOGLE
    )
    client = GoogleClient("test-api-key")
    result_call = ToolCall(
        id="result-call",
        name="schema.validate",
        arguments={"value": "result"},
    )
    result = ToolCallResult(
        id="result-outcome",
        name=result_call.name,
        arguments=result_call.arguments,
        call=result_call,
        result={"valid": True},
    )
    error_call = ToolCall(
        id="error-call",
        name="schema.validate",
        arguments={"value": "error"},
    )
    error = ToolCallError(
        id="error-outcome",
        name=error_call.name,
        arguments=error_call.arguments,
        call=error_call,
        error=ValueError("invalid"),
        message="invalid",
    )
    unanchored = ToolCallDiagnostic(
        id="unanchored-diagnostic",
        code=ToolCallDiagnosticCode.MALFORMED_CALL,
        stage=ToolCallDiagnosticStage.PARSE,
        message="Malformed call.",
    )
    anchored = ToolCallDiagnostic(
        id="anchored-diagnostic",
        call_id="diagnostic-call",
        requested_name="schema.validate",
        code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        stage=ToolCallDiagnosticStage.VALIDATE,
        message="Invalid arguments.",
    )

    messages = client._template_messages(
        [
            Message(role=MessageRole.TOOL),
            Message(
                role=MessageRole.TOOL,
                tool_call_diagnostic=unanchored,
            ),
            Message(
                role=MessageRole.TOOL,
                tool_call_diagnostic=anchored,
            ),
            Message(role=MessageRole.TOOL, tool_call_result=result),
            Message(role=MessageRole.TOOL, tool_call_error=error),
            Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[
                    MessageToolCall(
                        id=None,
                        name="schema.validate",
                        arguments=cast(Any, {"value": "next"}),
                    )
                ],
            ),
        ],
        capability=capability,
    )

    assert len(messages) == 5
    assert messages[0]["role"] == "model"
    unanchored_payload = loads(messages[0]["parts"][0]["text"])
    assert unanchored_payload["code"] == "tool_call.malformed"
    anchored_response = messages[1]["parts"][0]["function_response"]
    assert anchored_response["id"] == "diagnostic-call"
    assert anchored_response["name"] == provider_name
    assert (
        anchored_response["response"]["output"]["code"]
        == "tool_call.arguments_invalid"
    )
    result_response = messages[2]["parts"][0]["function_response"]
    assert result_response == {
        "id": "result-call",
        "name": provider_name,
        "response": {"output": {"valid": True}},
    }
    error_response = messages[3]["parts"][0]["function_response"]
    assert error_response == {
        "id": "error-call",
        "name": provider_name,
        "response": {"output": {"error": "invalid"}},
    }
    function_call = messages[4]["parts"][0]["function_call"]
    assert function_call == {
        "id": "",
        "name": provider_name,
        "args": {"value": "next"},
    }

    correlated = CorrelatedCapabilityResult(
        call_id="continuation-call",
        canonical_name="schema.validate",
        provider_name=provider_name,
        payload={"valid": True},
    )
    assert GoogleClient.capability_result_message(correlated) == {
        "role": "user",
        "parts": [
            {
                "function_response": {
                    "id": "continuation-call",
                    "name": provider_name,
                    "response": {"valid": True},
                }
            }
        ],
    }


class GoogleCapabilityStreamTestCase(IsolatedAsyncioTestCase):
    async def test_sdk_native_stream_reports_capability_and_canonical_call(
        self,
    ) -> None:
        capability = _catalog()
        provider_name = capability.provider_name(
            "schema.validate", provider_family=ProviderFamily.GOOGLE
        )
        response = _native_response(
            call_id="provider-call",
            name=provider_name,
            arguments={"value": "streamed"},
            text="Before call.",
        )
        stream = GoogleStream(
            _responses(response),
            capability=capability,
        )

        items = [item async for item in stream]

        capabilities = cast(
            Mapping[str, object], items[0].metadata["capabilities"]
        )
        self.assertTrue(capabilities["supports_tool_calls"])
        lifecycle = [
            item
            for item in items
            if item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            }
        ]
        self.assertEqual(
            [item.kind for item in lifecycle],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        self.assertTrue(
            all(
                item.correlation.tool_call_id == "provider-call"
                for item in lifecycle
            )
        )
        self.assertEqual(lifecycle[1].data, {"name": "schema.validate"})
        self.assertTrue(
            all(item.provider_payload is not None for item in lifecycle)
        )

    async def test_incapable_stream_preserves_native_provider_name(
        self,
    ) -> None:
        response = _native_response(
            call_id="native-call",
            name="native_tool",
            arguments={"value": "native"},
        )
        stream = GoogleStream(_responses(response))

        items = [item async for item in stream]

        capabilities = cast(
            Mapping[str, object], items[0].metadata["capabilities"]
        )
        self.assertFalse(capabilities["supports_tool_calls"])
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "native_tool"})

    async def test_non_stream_native_response_builds_canonical_continuation(
        self,
    ) -> None:
        capability = _catalog()
        provider_name = capability.provider_name(
            "schema.validate", provider_family=ProviderFamily.GOOGLE
        )
        native = _native_response(
            call_id="native-response-call",
            name=provider_name,
            arguments={"value": "response"},
            text="Prefix.",
        )
        client = GoogleClient("test-api-key")
        with patch.object(
            client._client.aio.models,
            "generate_content",
            new=AsyncMock(return_value=native),
        ) as generate:
            stream = await client(
                "gemini-test",
                [Message(role=MessageRole.USER, content="Validate.")],
                capability=capability,
                use_async_generator=False,
            )
        items = [item async for item in stream]
        output = "".join(
            item.text_delta or ""
            for item in items
            if item.kind is StreamItemKind.ANSWER_DELTA
        )

        generate.assert_awaited_once()
        self.assertIsInstance(stream, TextGenerationNonStreamResult)
        self.assertIn("Prefix.", output)
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(
            ready.correlation.tool_call_id,
            "native-response-call",
        )
        self.assertEqual(ready.data, {"name": "schema.validate"})
        provider_output = GoogleClient._response_text(native)
        self.assertIn('"name": "schema.validate"', provider_output)
