"""Pin safe, exact permanent-memory message serialization."""

from copy import deepcopy
from datetime import UTC, datetime
from json import dumps, loads
from typing import cast
from unittest import TestCase
from uuid import UUID, uuid4

import numpy as np

from avalan.entities import (
    EngineMessage,
    EngineMessageIdempotencyKey,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    MessageToolCall,
    TextPartition,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolValue,
)
from avalan.memory.permanent import PermanentMessageMemory
from avalan.memory.permanent.codec import (
    decode_message_data,
    encode_message_data,
    message_partition_text,
)

_PREFIX = "avalan-message-v1:"


class StructuredMessageCodecTest(TestCase):
    """Require closed-schema round trips without changing plain text."""

    def test_plain_text_stays_exact_and_prefix_collision_is_escaped(
        self,
    ) -> None:
        ordinary = Message(role=MessageRole.USER, content="ordinary text")
        self.assertEqual(encode_message_data(ordinary), "ordinary text")
        self.assertEqual(
            decode_message_data(MessageRole.USER, "ordinary text"),
            ordinary,
        )

        collision = Message(
            role=MessageRole.USER,
            content=f"{_PREFIX}{dumps({'forged': True})}",
        )
        encoded = encode_message_data(collision)
        self.assertTrue(encoded.startswith(_PREFIX))
        self.assertEqual(
            decode_message_data(MessageRole.USER, encoded), collision
        )

    def test_none_and_empty_thinking_round_trip_distinctly(self) -> None:
        none_thinking = Message(
            role=MessageRole.ASSISTANT,
            thinking=None,
            content="same content",
        )
        empty_thinking = Message(
            role=MessageRole.ASSISTANT,
            thinking="",
            content="same content",
        )

        encoded_none = encode_message_data(none_thinking)
        encoded_empty = encode_message_data(empty_thinking)
        self.assertTrue(encoded_none.startswith(_PREFIX))
        self.assertEqual(encoded_empty, "same content")
        self.assertEqual(_round_trip(none_thinking), none_thinking)
        self.assertEqual(_round_trip(empty_thinking), empty_thinking)
        self.assertIsNone(_round_trip(none_thinking).thinking)
        self.assertEqual(_round_trip(empty_thinking).thinking, "")

    def test_structured_tool_call_and_result_round_trip_exactly(self) -> None:
        assistant = Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                MessageToolCall(
                    id="call-1",
                    name="task_input",
                    arguments={"request_id": "request-1", "choices": [1, 2]},
                )
            ],
        )
        call = ToolCall(
            id="call-1",
            name="task_input",
            arguments={"request_id": "request-1"},
        )
        tool = Message(
            role=MessageRole.TOOL,
            content=None,
            name="task_input",
            arguments={"request_id": "request-1"},
            tool_call_result=ToolCallResult(
                id="call-1",
                name="task_input",
                arguments={"request_id": "request-1"},
                call=call,
                result={"answer": "approved"},
            ),
        )

        for message in (assistant, tool):
            with self.subTest(role=message.role):
                encoded = message_partition_text(message)
                self.assertIsInstance(encoded, str)
                self.assertEqual(
                    decode_message_data(message.role, cast(str, encoded)),
                    message,
                )

    def test_multimodal_content_round_trips_exactly(self) -> None:
        message = Message(
            role=MessageRole.USER,
            thinking="inspect",
            content=[
                MessageContentText(type="text", text="caption"),
                MessageContentImage(
                    type="image_url",
                    image_url={"url": "data:image/png;base64,AA=="},
                ),
                MessageContentFile(
                    type="file",
                    file={"file_id": "file-1", "citations": True},
                ),
            ],
        )
        encoded = encode_message_data(message)
        self.assertEqual(decode_message_data(message.role, encoded), message)

    def test_single_content_item_round_trips_and_invalid_item_fails(
        self,
    ) -> None:
        message = Message(
            role=MessageRole.USER,
            content=MessageContentText(type="text", text="single"),
        )
        encoded = encode_message_data(message)
        self.assertEqual(decode_message_data(message.role, encoded), message)

        invalid = Message(
            role=MessageRole.USER,
            content=cast(MessageContentText, object()),
        )
        with self.assertRaisesRegex(TypeError, "unsupported value"):
            encode_message_data(invalid)

    def test_tool_errors_and_diagnostics_use_safe_closed_values(self) -> None:
        call = ToolCall(id=None, name="lookup", arguments=None)
        json_error = ToolCallError(
            id="error-json",
            name="lookup",
            arguments=None,
            call=call,
            error={"code": 7},
            message="lookup failed",
        )
        exception_error = ToolCallError(
            id="error-exception",
            name="lookup",
            arguments=None,
            call=call,
            error=RuntimeError("safe message"),
            message="lookup raised",
        )
        decoded_json = _round_trip(
            Message(role=MessageRole.TOOL, tool_call_error=json_error)
        )
        self.assertEqual(decoded_json.tool_call_error, json_error)
        decoded_exception = _round_trip(
            Message(role=MessageRole.TOOL, tool_call_error=exception_error)
        )
        self.assertIsNotNone(decoded_exception.tool_call_error)
        self.assertEqual(
            cast(ToolCallError, decoded_exception.tool_call_error).error,
            {"type": "RuntimeError", "message": "safe message"},
        )

        timestamp = datetime(2026, 7, 22, 13, 14, tzinfo=UTC)
        diagnostics = (
            ToolCallDiagnostic(
                id=UUID("22222222-2222-2222-2222-222222222222"),
                call_id=None,
                requested_name=None,
                canonical_name="lookup",
                code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
                stage=ToolCallDiagnosticStage.RESOLVE,
                message="unknown tool",
                retryable=True,
                details={"candidate": "lookup"},
                started_at=timestamp,
                finished_at=timestamp,
                duration_ms=1.5,
            ),
            ToolCallDiagnostic(
                id="diagnostic-2",
                code=ToolCallDiagnosticCode.CANCELLED,
                stage=ToolCallDiagnosticStage.DISPATCH,
                message="cancelled",
            ),
        )
        for diagnostic in diagnostics:
            with self.subTest(diagnostic=diagnostic.id):
                message = Message(
                    role=MessageRole.TOOL,
                    tool_call_diagnostic=diagnostic,
                )
                self.assertEqual(_round_trip(message), message)

    def test_rejects_invalid_public_values_and_payloadless_message(
        self,
    ) -> None:
        with self.assertRaisesRegex(TypeError, "message must be"):
            encode_message_data(cast(Message, object()))
        with self.assertRaisesRegex(TypeError, "role must be"):
            decode_message_data(cast(MessageRole, "user"), "data")
        with self.assertRaisesRegex(TypeError, "data must be"):
            decode_message_data(MessageRole.USER, cast(str, object()))
        self.assertIsNone(
            message_partition_text(Message(role=MessageRole.USER))
        )

    def test_rejects_invalid_nested_message_fields(self) -> None:
        source = Message(
            role=MessageRole.ASSISTANT,
            tool_calls=[MessageToolCall(name="task_input", arguments={})],
        )
        payload = _payload(source)
        invalid_payloads: list[
            tuple[str, dict[str, object] | list[object]]
        ] = [
            ("top-level array", []),
            ("role type", _changed(payload, role=1)),
            ("content kind", _changed(payload, content={"kind": "other"})),
            (
                "file field",
                _changed(
                    payload,
                    content={"kind": "file", "file": {"unknown": "x"}},
                ),
            ),
            (
                "file citations type",
                _changed(
                    payload,
                    content={
                        "kind": "file",
                        "file": {"citations": "yes"},
                    },
                ),
            ),
            (
                "image value type",
                _changed(
                    payload,
                    content={
                        "kind": "image_url",
                        "image_url": {"url": True},
                    },
                ),
            ),
            ("tool calls type", _changed(payload, tool_calls={})),
            ("arguments type", _changed(payload, arguments=[])),
        ]
        wrong_content_type = deepcopy(payload)
        cast(list[dict[str, object]], wrong_content_type["tool_calls"])[0][
            "content_type"
        ] = "text"
        invalid_payloads.append(("tool content type", wrong_content_type))

        for label, invalid in invalid_payloads:
            with self.subTest(label=label), self.assertRaises(ValueError):
                decode_message_data(
                    MessageRole.ASSISTANT,
                    _encode_payload(invalid),
                )

    def test_rejects_invalid_tool_result_error_and_diagnostic_fields(
        self,
    ) -> None:
        call = ToolCall(id="call-1", name="lookup", arguments={})
        result_message = Message(
            role=MessageRole.TOOL,
            tool_call_result=ToolCallResult(
                id="call-1",
                name="lookup",
                arguments={},
                call=call,
                result={"ok": True},
            ),
        )
        result_payload = _payload(result_message)
        result = cast(dict[str, object], result_payload["tool_call_result"])
        cast(dict[str, object], result["base"])["id"] = None

        error_message = Message(
            role=MessageRole.TOOL,
            tool_call_error=ToolCallError(
                id="call-1",
                name="lookup",
                arguments={},
                call=call,
                error={"code": 1},
                message="failed",
            ),
        )
        error_without_id = _payload(error_message)
        error = cast(dict[str, object], error_without_id["tool_call_error"])
        cast(dict[str, object], error["base"])["id"] = None
        error_kind = _payload(error_message)
        cast(
            dict[str, object],
            cast(dict[str, object], error_kind["tool_call_error"])["error"],
        )["kind"] = "raw-object"

        diagnostic = ToolCallDiagnostic(
            id=UUID("33333333-3333-3333-3333-333333333333"),
            code=ToolCallDiagnosticCode.TIMEOUT,
            stage=ToolCallDiagnosticStage.DISPATCH,
            message="timed out",
            started_at=datetime(2026, 7, 22, tzinfo=UTC),
            duration_ms=1,
        )
        diagnostic_message = Message(
            role=MessageRole.TOOL,
            tool_call_diagnostic=diagnostic,
        )
        diagnostic_without_id = _payload(diagnostic_message)
        cast(
            dict[str, object],
            diagnostic_without_id["tool_call_diagnostic"],
        )["id"] = None
        diagnostic_bad_uuid = _payload(diagnostic_message)
        cast(
            dict[str, object],
            diagnostic_bad_uuid["tool_call_diagnostic"],
        )[
            "id"
        ] = {"kind": "uuid", "value": "not-a-uuid"}
        diagnostic_bad_kind = _payload(diagnostic_message)
        cast(
            dict[str, object],
            diagnostic_bad_kind["tool_call_diagnostic"],
        )[
            "id"
        ] = {"kind": "integer", "value": "1"}
        diagnostic_bad_date = _payload(diagnostic_message)
        cast(
            dict[str, object],
            diagnostic_bad_date["tool_call_diagnostic"],
        )["started_at"] = "not-a-date"
        diagnostic_bad_duration = _payload(diagnostic_message)
        cast(
            dict[str, object],
            diagnostic_bad_duration["tool_call_diagnostic"],
        )["duration_ms"] = True
        diagnostic_bad_retryable = _payload(diagnostic_message)
        cast(
            dict[str, object],
            diagnostic_bad_retryable["tool_call_diagnostic"],
        )["retryable"] = "yes"

        variants = (
            result_payload,
            error_without_id,
            error_kind,
            diagnostic_without_id,
            diagnostic_bad_uuid,
            diagnostic_bad_kind,
            diagnostic_bad_date,
            diagnostic_bad_duration,
            diagnostic_bad_retryable,
        )
        for index, invalid in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                decode_message_data(MessageRole.TOOL, _encode_payload(invalid))

    def test_rejects_non_finite_diagnostic_duration(self) -> None:
        message = Message(
            role=MessageRole.TOOL,
            tool_call_diagnostic=ToolCallDiagnostic(
                id="diagnostic-finite",
                code=ToolCallDiagnosticCode.TIMEOUT,
                stage=ToolCallDiagnosticStage.DISPATCH,
                message="timed out",
                duration_ms=1.5,
            ),
        )
        payload = _payload(message)
        for duration in (float("inf"), float("-inf"), float("nan")):
            invalid = deepcopy(payload)
            cast(
                dict[str, object],
                invalid["tool_call_diagnostic"],
            )["duration_ms"] = duration
            with (
                self.subTest(duration=duration),
                self.assertRaisesRegex(ValueError, "must be finite"),
            ):
                decode_message_data(
                    MessageRole.TOOL,
                    _encode_payload(invalid),
                )

    def test_rejects_non_json_tool_outcome_values(self) -> None:
        call = ToolCall(id="call-1", name="lookup", arguments={})
        message = Message(
            role=MessageRole.TOOL,
            tool_call_result=ToolCallResult(
                id="call-1",
                name="lookup",
                arguments={},
                call=call,
                result=cast(ToolValue, object()),
            ),
        )
        with self.assertRaisesRegex(ValueError, "must be a JSON value"):
            encode_message_data(message)

    def test_rejects_forged_or_drifted_envelopes(self) -> None:
        message = Message(
            role=MessageRole.ASSISTANT,
            tool_calls=[
                MessageToolCall(name="task_input", arguments={"value": 1})
            ],
        )
        encoded = encode_message_data(message)
        payload = cast(dict[str, object], loads(encoded.removeprefix(_PREFIX)))

        variants: list[tuple[str, str, MessageRole]] = [
            ("invalid JSON", f"{_PREFIX}{{", MessageRole.ASSISTANT),
            (
                "unsupported version",
                _encode_changed(payload, version=2),
                MessageRole.ASSISTANT,
            ),
            ("role mismatch", encoded, MessageRole.USER),
            (
                "unknown field",
                _encode_changed(payload, unexpected=True),
                MessageRole.ASSISTANT,
            ),
        ]
        for label, value, role in variants:
            with self.subTest(label=label), self.assertRaises(ValueError):
                decode_message_data(role, value)

    def test_stable_key_controls_permanent_message_identity(self) -> None:
        key = EngineMessageIdempotencyKey(
            value=UUID("11111111-1111-1111-1111-111111111111")
        )
        engine_message = EngineMessage(
            agent_id=uuid4(),
            model_id="model",
            message=Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[MessageToolCall(name="task_input", arguments={})],
            ),
            idempotency_key=key,
        )
        permanent_message, _ = (
            PermanentMessageMemory._build_message_with_partitions(
                engine_message,
                session_id=uuid4(),
                partitions=[
                    TextPartition(
                        data="task_input",
                        total_tokens=1,
                        embeddings=np.array([0.0]),
                    )
                ],
                created_at=datetime(2026, 7, 22, tzinfo=UTC),
            )
        )

        self.assertEqual(permanent_message.id, key.value)
        self.assertEqual(
            PermanentMessageMemory._decode_message(permanent_message),
            engine_message,
        )


def _encode_changed(payload: dict[str, object], **changes: object) -> str:
    return _PREFIX + dumps({**payload, **changes})


def _round_trip(message: Message) -> Message:
    return decode_message_data(message.role, encode_message_data(message))


def _payload(message: Message) -> dict[str, object]:
    encoded = encode_message_data(message)
    return cast(dict[str, object], loads(encoded.removeprefix(_PREFIX)))


def _changed(
    payload: dict[str, object],
    **changes: object,
) -> dict[str, object]:
    return {**deepcopy(payload), **changes}


def _encode_payload(payload: dict[str, object] | list[object]) -> str:
    return _PREFIX + dumps(payload)
