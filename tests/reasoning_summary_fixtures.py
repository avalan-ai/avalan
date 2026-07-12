"""Provide reusable reasoning-summary provider trace fixtures."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import isfinite
from pathlib import Path
from typing import Literal, cast

from reasoning_summary_script_loader import (
    json_mapping_entries,
    strict_json_loads,
)

from avalan.server.entities import (
    SKILL_CONTENT_REDACTION,
    ModelVisibleServerProtocolTextRedactor,
    ServerOutputRedactionProtocol,
    ServerOutputRedactionSettings,
)

ReasoningRepresentationValue = Literal["native_text", "summary"]

REASONING_SUMMARY_TRACE_NAMES = (
    "one_part",
    "multipart",
    "empty",
    "fallback",
    "mixed_fallback",
    "malformed",
    "retry",
    "cancellation",
    "tools_answer",
    "multi_continuation",
    "sparse_indices",
    "failure_after_summary",
    "incomplete_after_summary",
    "zero_length_fallback",
)

_ALLOWED_PROVIDER_EVENT_TYPES = frozenset(
    (
        "response.output_item.added",
        "response.output_item.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.completed",
        "response.failed",
        "response.incomplete",
    )
)
_PROVIDER_TERMINAL_EVENT_TYPES = frozenset(
    (
        "response.completed",
        "response.failed",
        "response.incomplete",
    )
)
_MCP_REASONING_TRUNCATION_FIELDS = frozenset(
    (
        "truncated",
        "dropped_segments",
        "dropped_characters",
        "dropped_utf8_bytes",
        "leading_segment_partial",
    )
)
_PROVIDER_TRACE_MAPPING_CATALOG = {
    "cancellation": (
        8,
        "f2419cce60ff7683c8d24d312a4a4f96be7e606413ee44aebef8e027896afc29",
    ),
    "empty": (
        9,
        "8443c066b74a0ab4d6a05c9e944eaf55bd1a4a7fdd8d24a75713d9f8109f11d8",
    ),
    "failure_after_summary": (
        11,
        "d0a4cbe8a1de688826e9443de38910da287fde0e09f83e85e06db1f81a32424a",
    ),
    "fallback": (
        11,
        "52ff5d6ce6ec152f1ee7dcf8187bda562ebef379a90d6c5ec6b6a7b0c00b0d52",
    ),
    "incomplete_after_summary": (
        11,
        "a3868da2d9eb354e48f04f9f14f6c47f23e7d8f21f0f2a3e2c25f1bd6f0d3a02",
    ),
    "malformed": (
        26,
        "75817c4c0f078cec192ca9e38b2155910c0ac4308faa5c0a8c0f7458f8b69bbe",
    ),
    "mixed_fallback": (
        19,
        "c1da52b63f29f1be90705638de2be0121d9bc0b418d90a193efbe500816ece6e",
    ),
    "multi_continuation": (
        76,
        "af05a107177eac54b64c158703e48906bd862ba9b560e37c9bcb25869d1c8009",
    ),
    "multipart": (
        25,
        "58a2db685683f22b363f9c88e8a468fa4649e36d4a67b7efc64f148a92f7340e",
    ),
    "one_part": (
        20,
        "ca9ff6e03b4679cdbeb82d9071cd6f00fa39aaa32640c89994a0a1899d7c5c96",
    ),
    "retry": (
        20,
        "9f158d60ba6c3d311494b3c508be50861ad32619408025d96af1a5d6a516cb75",
    ),
    "sparse_indices": (
        37,
        "7eb4be11f981c97c1af8245c434149acb16277f1203963cbd8854a1f8e7c826e",
    ),
    "tools_answer": (
        53,
        "6296699bae4276c78497cf6edee72d4c9561fbe0e6a3166f98653ca2d7184b1b",
    ),
    "zero_length_fallback": (
        17,
        "5688aa75d02257ece4dabe5af33a7bfe33c7e58d511416f7ea0ff6de2f867d23",
    ),
}
_MALFORMED_ROW_CONTRACT = {
    "bad-index-type": (
        "summary_index must be an integer",
        "1d9f13d64ea12c75e26f0543d307b5536580a9463fc1eb6b46739b571e62b9fd",
    ),
    "bad-negative-index": (
        "output_index must not be negative",
        "78e285f301f077905a821fa4044a0c9274f6d3b36a8e8de84afddbf870f11d7c",
    ),
    "bad-missing-id": (
        "item_id must be a string",
        "4b4b462fb5fba0c3e02b0b4997ee3cb6830f7ed511bda0db6fd1bdfe620e2917",
    ),
    "bad-delta-type": (
        "delta must be a string",
        "47b944301cdef4f85f538789059ab94cc4961df5f6603a66a41de5930ecc056d",
    ),
}


@dataclass(frozen=True, kw_only=True, slots=True)
class ReasoningDeltaExpectation:
    """Describe one typed reasoning delta expected from a provider trace."""

    representation: ReasoningRepresentationValue
    segment_instance_ordinal: int
    text: str
    provider_item_id: str | None = None
    output_index: int | None = None
    summary_index: int | None = None
    continuation_id: str | None = None
    provider_event_type: str | None = None

    def __post_init__(self) -> None:
        assert self.representation in ("native_text", "summary")
        assert type(self.segment_instance_ordinal) is int
        assert self.segment_instance_ordinal >= 0
        assert isinstance(self.text, str)
        for string_field_name, string_value in (
            ("provider_item_id", self.provider_item_id),
            ("continuation_id", self.continuation_id),
            ("provider_event_type", self.provider_event_type),
        ):
            if string_value is not None:
                assert isinstance(
                    string_value, str
                ), f"{string_field_name} must be a string"
                assert (
                    string_value.strip()
                ), f"{string_field_name} must not be empty"
        for index_field_name, index_value in (
            ("output_index", self.output_index),
            ("summary_index", self.summary_index),
        ):
            if index_value is not None:
                assert (
                    type(index_value) is int
                ), f"{index_field_name} must be an integer"
                assert (
                    index_value >= 0
                ), f"{index_field_name} must not be negative"


@dataclass(frozen=True, kw_only=True, slots=True)
class ReasoningPartIdentity:
    """Identify one reasoning segment at a protocol redaction boundary."""

    representation: ReasoningRepresentationValue
    segment_instance_ordinal: int
    provider_item_id: str | None = None
    output_index: int | None = None
    summary_index: int | None = None
    continuation_id: str | None = None

    def __post_init__(self) -> None:
        assert self.representation in ("native_text", "summary")
        assert type(self.segment_instance_ordinal) is int
        assert self.segment_instance_ordinal >= 0
        for string_field_name, string_value in (
            ("provider_item_id", self.provider_item_id),
            ("continuation_id", self.continuation_id),
        ):
            if string_value is not None:
                assert (
                    isinstance(string_value, str) and string_value.strip()
                ), string_field_name
        for index_field_name, index_value in (
            ("output_index", self.output_index),
            ("summary_index", self.summary_index),
        ):
            if index_value is not None:
                assert type(index_value) is int
                assert index_value >= 0, index_field_name


@dataclass(frozen=True, kw_only=True, slots=True)
class TaggedRedactedText:
    """Pair safe protocol text with the identity allowed to receive it."""

    identity: ReasoningPartIdentity
    text: str


class IdentityTaggedReasoningRedactor:
    """Model the locked fail-closed cross-identity redaction contract."""

    def __init__(
        self,
        settings: ServerOutputRedactionSettings,
        *,
        protocol: ServerOutputRedactionProtocol,
    ) -> None:
        assert isinstance(settings, ServerOutputRedactionSettings)
        self._settings = settings
        self._protocol = protocol
        self._identity: ReasoningPartIdentity | None = None
        self._redactor = self._new_redactor()
        self._quarantined = False
        self._quarantine_next_identity = False
        self._redaction_latched = False

    def push(
        self,
        identity: ReasoningPartIdentity | None,
        value: str,
    ) -> tuple[TaggedRedactedText, ...]:
        """Return identity-tagged safe text for one provider delta."""
        assert identity is None or isinstance(identity, ReasoningPartIdentity)
        assert isinstance(value, str)
        if self._redaction_latched:
            return ()
        outputs: list[TaggedRedactedText] = []
        if identity is None:
            outputs.extend(self._resolve_pending_before_boundary())
            self._reset()
            self._quarantine_next_identity = True
            return tuple(outputs)
        if self._identity is None:
            self._identity = identity
            self._quarantined = self._quarantine_next_identity
            self._quarantine_next_identity = False
        elif identity != self._identity:
            pending_crossed_identity = self._redactor.has_pending
            outputs.extend(self._resolve_pending_before_boundary())
            self._identity = identity
            self._redactor = self._new_redactor()
            self._quarantined = pending_crossed_identity
            if self._redaction_latched:
                return tuple(outputs)
        if self._quarantined:
            return tuple(outputs)
        outputs.extend(
            TaggedRedactedText(identity=identity, text=text)
            for text in self._redactor.push(value)
        )
        if any(item.text == SKILL_CONTENT_REDACTION for item in outputs):
            self._redaction_latched = True
        return tuple(outputs)

    def complete(
        self,
        identity: ReasoningPartIdentity | None,
    ) -> tuple[TaggedRedactedText, ...]:
        """Resolve pending text before a segment completion boundary."""
        assert identity is None or isinstance(identity, ReasoningPartIdentity)
        if self._redaction_latched:
            self._reset()
            return ()
        identity_matches = identity is not None and identity == self._identity
        outputs = self._resolve_pending_before_boundary()
        resolved_ambiguous_pending = bool(outputs)
        self._reset()
        if not identity_matches or resolved_ambiguous_pending:
            self._quarantine_next_identity = True
        return outputs

    def _resolve_pending_before_boundary(
        self,
    ) -> tuple[TaggedRedactedText, ...]:
        if (
            self._identity is None
            or self._quarantined
            or not self._redactor.has_pending
        ):
            return ()
        self._redaction_latched = True
        return (
            TaggedRedactedText(
                identity=self._identity,
                text=SKILL_CONTENT_REDACTION,
            ),
        )

    def _reset(self) -> None:
        self._identity = None
        self._redactor = self._new_redactor()
        self._quarantined = False

    def _new_redactor(self) -> ModelVisibleServerProtocolTextRedactor:
        return ModelVisibleServerProtocolTextRedactor(
            self._settings,
            protocol=self._protocol,
            channel="reasoning",
        )


def validate_mcp_reasoning_truncation(value: object) -> None:
    """Validate one exact MCP aggregate reasoning-truncation object."""
    assert isinstance(value, dict)
    truncation = cast(dict[str, object], value)
    assert set(truncation) == _MCP_REASONING_TRUNCATION_FIELDS
    assert type(truncation["truncated"]) is bool
    assert type(truncation["leading_segment_partial"]) is bool
    for field_name in (
        "dropped_segments",
        "dropped_characters",
        "dropped_utf8_bytes",
    ):
        count = truncation[field_name]
        assert type(count) is int
        assert count >= 0


def reasoning_delta_expectation(
    text: str,
    *,
    representation: ReasoningRepresentationValue,
    segment_instance_ordinal: int,
    provider_item_id: str | None = None,
    output_index: int | None = None,
    summary_index: int | None = None,
    continuation_id: str | None = None,
    provider_event_type: str | None = None,
) -> ReasoningDeltaExpectation:
    """Return a validated typed reasoning-delta expectation."""
    return ReasoningDeltaExpectation(
        representation=representation,
        segment_instance_ordinal=segment_instance_ordinal,
        text=text,
        provider_item_id=provider_item_id,
        output_index=output_index,
        summary_index=summary_index,
        continuation_id=continuation_id,
        provider_event_type=provider_event_type,
    )


def reasoning_summary_fixture_root() -> Path:
    """Return the checked-in reasoning-summary fixture directory."""
    return Path(__file__).resolve().parent / "fixtures" / "reasoning_summary"


def load_reasoning_summary_trace(name: str) -> dict[str, object]:
    """Load and minimally validate a named provider trace fixture."""
    assert isinstance(name, str)
    assert (
        name in REASONING_SUMMARY_TRACE_NAMES
    ), f"unknown reasoning-summary trace: {name}"
    path = (
        reasoning_summary_fixture_root() / "provider_traces" / f"{name}.json"
    )
    return validate_reasoning_summary_trace_json(
        path.read_text(encoding="utf-8"),
        name=name,
    )


def validate_reasoning_summary_trace_json(
    source: str,
    *,
    name: str,
) -> dict[str, object]:
    """Parse and validate one named provider trace JSON document."""
    payload = strict_json_loads(source)
    return validate_reasoning_summary_trace_payload(payload, name=name)


def validate_reasoning_summary_trace_payload(
    payload: object,
    *,
    name: str,
) -> dict[str, object]:
    """Validate one complete named provider trace payload."""
    assert name in REASONING_SUMMARY_TRACE_NAMES
    assert isinstance(payload, dict)
    _validate_provider_trace_mapping_shape(payload, name=name)
    expected_keys = {"schema_version", "name", "responses"}
    if name == "malformed":
        expected_keys.add("expected")
    assert set(payload) == expected_keys
    schema_version = payload.get("schema_version")
    assert type(schema_version) is int and schema_version == 1
    assert payload.get("name") == name
    responses = payload.get("responses")
    assert isinstance(responses, list) and responses
    if name == "malformed":
        assert payload.get("expected") == "invalid"
        _validate_negative_responses(responses)
    else:
        _validate_positive_responses(responses)
    return cast(dict[str, object], payload)


def _validate_provider_trace_mapping_shape(
    payload: dict[str, object],
    *,
    name: str,
) -> None:
    entries = json_mapping_entries(payload)
    expected_count, expected_sha256 = _PROVIDER_TRACE_MAPPING_CATALOG[name]
    assert len(entries) == expected_count
    canonical = dumps(
        [[pointer, list(keys)] for pointer, _, keys in entries],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert sha256(canonical).hexdigest() == expected_sha256


def reasoning_summary_trace_responses(
    name: str,
) -> tuple[dict[str, object], ...]:
    """Return validated response-attempt groups from a named trace."""
    raw_responses = load_reasoning_summary_trace(name)["responses"]
    assert isinstance(raw_responses, list)
    responses: list[dict[str, object]] = []
    for response in raw_responses:
        assert isinstance(response, dict)
        continuation_id = response.get("continuation_id")
        assert isinstance(continuation_id, str) and continuation_id
        attempts = response.get("attempts")
        events = response.get("events")
        assert (attempts is None) != (
            events is None
        ), "a response must contain exactly one of events or attempts"
        if attempts is not None:
            assert isinstance(attempts, list) and attempts
            for attempt in attempts:
                assert isinstance(attempt, list) and attempt
                _validate_events(attempt)
        else:
            assert isinstance(events, list) and events
            _validate_events(events)
        responses.append(cast(dict[str, object], response))
    return tuple(responses)


def _validate_events(events: list[object]) -> None:
    for event in events:
        assert isinstance(event, dict)
        event_type = event.get("type")
        assert isinstance(event_type, str) and event_type.startswith(
            "response."
        )


def reasoning_summary_events_before_cancellation(
    name: str,
) -> tuple[dict[str, object], ...]:
    """Return only the provider events consumed before local cancellation."""
    responses = reasoning_summary_trace_responses(name)
    assert len(responses) == 1
    response = responses[0]
    cancel_after_event = response.get("cancel_after_event")
    events = response.get("events")
    assert type(cancel_after_event) is int
    assert isinstance(events, list)
    assert 0 < cancel_after_event <= len(events)
    return tuple(
        cast(dict[str, object], event) for event in events[:cancel_after_event]
    )


def validate_reasoning_summary_event_group(
    events: Sequence[object],
    *,
    cancel_after_event: int | None = None,
) -> None:
    """Validate strict provider identity, order, and lifecycle grammar."""
    assert isinstance(events, Sequence) and events
    if cancel_after_event is not None:
        assert type(cancel_after_event) is int
        assert 0 < cancel_after_event <= len(events)
    last_sequence = -1
    terminal_seen = False
    terminal_count = 0
    open_items: dict[tuple[str, int], str] = {}
    added_items: dict[tuple[str, int], dict[str, object]] = {}
    observed_items: dict[tuple[str, int], dict[str, object]] = {}
    seen_item_keys: set[tuple[str, int]] = set()
    item_id_by_output_index: dict[int, str] = {}
    output_index_by_item_id: dict[str, int] = {}
    completed_items: list[dict[str, object]] = []
    open_parts: set[tuple[str, int, int]] = set()
    part_text: dict[tuple[str, int, int], list[str]] = {}
    text_done: set[tuple[str, int, int]] = set()
    streamed_part_text: dict[tuple[str, int, int], str] = {}
    function_argument_text: dict[tuple[str, int], list[str]] = {}
    function_done: dict[tuple[str, int], str] = {}
    open_content_parts: set[tuple[str, int, int]] = set()
    content_text: dict[tuple[str, int, int], list[str]] = {}
    content_done: dict[tuple[str, int, int], str] = {}

    for raw_event in events:
        assert isinstance(raw_event, dict)
        event_type = raw_event.get("type")
        assert (
            event_type in _ALLOWED_PROVIDER_EVENT_TYPES
        ), f"unsupported provider fixture event: {event_type!r}"
        assert isinstance(event_type, str)
        assert "provider_payload" not in raw_event
        sequence_number = raw_event.get("sequence_number")
        assert type(sequence_number) is int
        assert (
            sequence_number > last_sequence
        ), "provider fixture sequence numbers must be strictly increasing"
        last_sequence = sequence_number
        assert not terminal_seen, "provider fixture event follows terminal"

        if event_type == "response.output_item.added":
            item = _event_item(raw_event)
            item_key = _item_key(item, raw_event)
            item_type = _non_empty_string(item.get("type"), "item.type")
            assert item_type in {"reasoning", "function_call", "message"}
            assert item.get("status") == "in_progress"
            if item_type == "reasoning":
                assert item.get("summary") == []
            elif item_type == "function_call":
                _non_empty_string(item.get("name"), "item.name")
                _non_empty_string(item.get("call_id"), "item.call_id")
                assert item.get("arguments") == ""
            else:
                assert item.get("role") == "assistant"
                assert item.get("content") == []
            assert item_key not in seen_item_keys
            seen_item_keys.add(item_key)
            existing_item_id = item_id_by_output_index.get(item_key[1])
            assert (
                existing_item_id is None or existing_item_id == item_key[0]
            ), "provider output index reused for a conflicting item ID"
            existing_output_index = output_index_by_item_id.get(item_key[0])
            assert (
                existing_output_index is None
                or existing_output_index == item_key[1]
            ), "provider item ID reused for a conflicting output index"
            item_id_by_output_index[item_key[1]] = item_key[0]
            output_index_by_item_id[item_key[0]] = item_key[1]
            open_items[item_key] = item_type
            added_items[item_key] = deepcopy(item)
            observed_items[item_key] = deepcopy(item)
            if item_type == "function_call":
                function_argument_text[item_key] = []
            continue

        if event_type.startswith("response.reasoning_summary_"):
            summary_key = _summary_key(raw_event)
            item_key = (summary_key[0], summary_key[1])
            assert (
                open_items.get(item_key) == "reasoning"
            ), "summary event must reference one open reasoning item"
            if event_type == "response.reasoning_summary_part.added":
                assert (
                    summary_key not in open_parts
                    and summary_key not in streamed_part_text
                )
                part = raw_event.get("part")
                assert isinstance(part, dict)
                assert part == {"type": "summary_text", "text": ""}
                open_parts.add(summary_key)
                part_text[summary_key] = []
            elif event_type == "response.reasoning_summary_text.delta":
                assert (
                    summary_key in open_parts and summary_key not in text_done
                )
                delta = raw_event.get("delta")
                assert isinstance(delta, str), "delta must be a string"
                part_text[summary_key].append(delta)
            elif event_type == "response.reasoning_summary_text.done":
                assert (
                    summary_key in open_parts and summary_key not in text_done
                )
                text = raw_event.get("text")
                assert isinstance(text, str)
                assert text == "".join(part_text[summary_key])
                text_done.add(summary_key)
                streamed_part_text[summary_key] = text
            else:
                assert event_type == "response.reasoning_summary_part.done"
                assert summary_key in open_parts and summary_key in text_done
                part = raw_event.get("part")
                assert isinstance(part, dict)
                assert part == {
                    "type": "summary_text",
                    "text": streamed_part_text[summary_key],
                }
                observed_summary = observed_items[item_key].get("summary")
                assert isinstance(observed_summary, list)
                summary_index = summary_key[2]
                while len(observed_summary) <= summary_index:
                    observed_summary.append(None)
                assert observed_summary[summary_index] is None
                observed_summary[summary_index] = deepcopy(part)
                open_parts.remove(summary_key)
            continue

        if event_type.startswith("response.function_call_arguments."):
            function_item_key = _correlated_item_key(raw_event)
            assert (
                open_items.get(function_item_key) == "function_call"
            ), "function arguments must reference an open function-call item"
            if event_type == "response.function_call_arguments.delta":
                assert function_item_key not in function_done
                delta = raw_event.get("delta")
                assert isinstance(delta, str)
                function_argument_text[function_item_key].append(delta)
            else:
                assert event_type == "response.function_call_arguments.done"
                assert function_item_key not in function_done
                added_item = added_items[function_item_key]
                name = _non_empty_string(raw_event.get("name"), "name")
                assert name == added_item.get("name")
                if "call_id" in raw_event:
                    call_id = _non_empty_string(
                        raw_event.get("call_id"), "call_id"
                    )
                    assert call_id == added_item.get("call_id")
                arguments = raw_event.get("arguments")
                assert isinstance(arguments, str)
                assert arguments == "".join(
                    function_argument_text[function_item_key]
                )
                function_done[function_item_key] = arguments
                observed_items[function_item_key]["arguments"] = arguments
            continue

        if event_type == "response.content_part.added":
            content_key = _content_key(raw_event)
            content_item_key = (content_key[0], content_key[1])
            assert (
                open_items.get(content_item_key) == "message"
            ), "content events must reference an open message item"
            assert content_key not in open_content_parts
            part = raw_event.get("part")
            assert isinstance(part, dict)
            assert part.get("type") == "output_text"
            assert part.get("text") == ""
            open_content_parts.add(content_key)
            content_text[content_key] = []
            continue

        if event_type in {
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
        }:
            content_key = _content_key(raw_event)
            content_item_key = (content_key[0], content_key[1])
            assert (
                open_items.get(content_item_key) == "message"
            ), "content events must reference an open message item"
            assert content_key in open_content_parts
            if event_type == "response.output_text.delta":
                assert content_key not in content_done
                delta = raw_event.get("delta")
                assert isinstance(delta, str)
                content_text[content_key].append(delta)
            elif event_type == "response.output_text.done":
                assert content_key not in content_done
                text = raw_event.get("text")
                assert isinstance(text, str)
                assert text == "".join(content_text[content_key])
                content_done[content_key] = text
            else:
                assert event_type == "response.content_part.done"
                assert content_key in content_done
                part = raw_event.get("part")
                assert isinstance(part, dict)
                assert part.get("type") == "output_text"
                assert part.get("text") == content_done[content_key]
                observed_content = observed_items[content_item_key].get(
                    "content"
                )
                assert isinstance(observed_content, list)
                content_index = content_key[2]
                assert content_index == len(observed_content)
                observed_content.append(deepcopy(part))
                open_content_parts.remove(content_key)
            continue

        if event_type == "response.output_item.done":
            item = _event_item(raw_event)
            completed_item_key = _item_key(item, raw_event)
            assert completed_item_key in open_items
            completed_item_type = _non_empty_string(
                item.get("type"), "item.type"
            )
            assert item.get("status") == "completed"
            assert (
                open_items[completed_item_key] == completed_item_type
            ), "completed item type conflicts with its added item"
            _validate_done_item_transition(
                item,
                added_items[completed_item_key],
                observed_items[completed_item_key],
                item_type=completed_item_type,
            )
            assert not any(
                part_key[:2] == completed_item_key and part_key in open_parts
                for part_key in part_text
            ), "reasoning item completed with an open summary part"
            if completed_item_type == "reasoning":
                summary = item.get("summary")
                assert isinstance(summary, list)
                for summary_index, raw_part in enumerate(summary):
                    assert isinstance(raw_part, dict)
                    assert raw_part.get("type") == "summary_text"
                    summary_text = raw_part.get("text")
                    assert isinstance(summary_text, str)
                    streamed = streamed_part_text.get(
                        (
                            completed_item_key[0],
                            completed_item_key[1],
                            summary_index,
                        )
                    )
                    if streamed:
                        assert streamed == summary_text
                for part_key, streamed_text in streamed_part_text.items():
                    if part_key[:2] != completed_item_key or not streamed_text:
                        continue
                    summary_index = part_key[2]
                    assert summary_index < len(
                        summary
                    ), "streamed summary part missing from completed item"
                    completed_part = summary[summary_index]
                    assert isinstance(completed_part, dict)
                    assert completed_part == {
                        "type": "summary_text",
                        "text": streamed_text,
                    }
            elif completed_item_type == "function_call":
                assert completed_item_key in function_done
                assert (
                    item.get("arguments") == function_done[completed_item_key]
                )
                del function_argument_text[completed_item_key]
                del function_done[completed_item_key]
            else:
                assert completed_item_type == "message"
                assert not any(
                    key[:2] == completed_item_key for key in open_content_parts
                ), "message item completed with an open content part"
                content = item.get("content")
                assert isinstance(content, list)
                for content_key, completed_text in content_done.items():
                    if content_key[:2] != completed_item_key:
                        continue
                    content_index = content_key[2]
                    assert content_index < len(content)
                    completed_part = content[content_index]
                    assert isinstance(completed_part, dict)
                    assert completed_part.get("type") == "output_text"
                    assert completed_part.get("text") == completed_text
            del open_items[completed_item_key]
            completed_items.append(item)
            continue

        assert event_type in _PROVIDER_TERMINAL_EVENT_TYPES
        terminal_seen = True
        terminal_count += 1
        response = raw_event.get("response")
        assert isinstance(response, dict)
        _validate_provider_terminal_response(response, event_type=event_type)
        output = response.get("output")
        assert isinstance(output, list)
        if event_type == "response.completed":
            _validate_completed_terminal_output(output, completed_items)
            assert output == completed_items
            assert not open_parts
            assert not open_content_parts
            assert not open_items
        else:
            assert event_type in {"response.failed", "response.incomplete"}
            _validate_abnormal_terminal_output(
                output,
                open_items,
                observed_items,
                completed_items,
                output_index_by_item_id,
            )
            open_parts.clear()
            open_content_parts.clear()
            open_items.clear()

    expected_terminal_count = 0 if cancel_after_event is not None else 1
    assert terminal_count == expected_terminal_count, (
        "provider fixture must have exactly one terminal unless locally "
        "cancelled"
    )


def _validate_completed_terminal_output(
    output: list[object],
    completed_items: list[dict[str, object]],
) -> None:
    assert len(output) == len(completed_items)
    for raw_item, completed_item in zip(output, completed_items, strict=True):
        assert isinstance(raw_item, dict)
        assert raw_item.get("status") == "completed"
        assert raw_item.get("id") == completed_item.get("id")
        assert raw_item.get("type") == completed_item.get("type")


def _validate_abnormal_terminal_output(
    output: list[object],
    open_items: dict[tuple[str, int], str],
    observed_items: dict[tuple[str, int], dict[str, object]],
    completed_items: list[dict[str, object]],
    output_index_by_item_id: dict[str, int],
) -> None:
    expected_by_index: list[tuple[int, dict[str, object], str | None]] = []
    for completed_item in completed_items:
        item_id = _non_empty_string(completed_item.get("id"), "item.id")
        expected_by_index.append(
            (
                output_index_by_item_id[item_id],
                deepcopy(completed_item),
                None,
            )
        )
    for item_key, item_type in open_items.items():
        expected_item = deepcopy(observed_items[item_key])
        expected_item["status"] = "incomplete"
        expected_by_index.append((item_key[1], expected_item, item_type))
    expected_by_index.sort(key=lambda entry: entry[0])
    assert len(output) == len(expected_by_index)
    for raw_item, (_, expected_item, open_item_type) in zip(
        output,
        expected_by_index,
        strict=True,
    ):
        assert isinstance(raw_item, dict)
        if open_item_type == "reasoning":
            _validate_abnormal_open_reasoning_item(raw_item, expected_item)
        else:
            assert raw_item == expected_item


def _validate_abnormal_open_reasoning_item(
    item: dict[str, object],
    observed_item: dict[str, object],
) -> None:
    actual_summary = item.get("summary")
    observed_summary = observed_item.get("summary")
    assert isinstance(actual_summary, list)
    assert isinstance(observed_summary, list)
    assert {key: value for key, value in item.items() if key != "summary"} == {
        key: value for key, value in observed_item.items() if key != "summary"
    }
    if not observed_summary:
        assert actual_summary == []
        return
    assert len(actual_summary) >= len(observed_summary)
    for summary_index, raw_part in enumerate(actual_summary):
        assert isinstance(raw_part, dict)
        assert set(raw_part) == {"type", "text"}
        assert raw_part.get("type") == "summary_text"
        assert isinstance(raw_part.get("text"), str)
        if summary_index >= len(observed_summary):
            continue
        observed_part = observed_summary[summary_index]
        if observed_part is None:
            continue
        assert isinstance(observed_part, dict)
        observed_text = observed_part.get("text")
        assert isinstance(observed_text, str)
        if observed_text:
            assert raw_part == observed_part


def _validate_done_item_transition(
    item: dict[str, object],
    added_item: dict[str, object],
    observed_item: dict[str, object],
    *,
    item_type: str,
) -> None:
    if item_type in {"function_call", "message"}:
        expected_item = deepcopy(observed_item)
        expected_item["status"] = "completed"
        assert item == expected_item
        return
    assert item_type == "reasoning"
    mutable_fields = {"status", "summary", "encrypted_content"}
    assert {
        key: value for key, value in item.items() if key not in mutable_fields
    } == {
        key: value
        for key, value in added_item.items()
        if key not in mutable_fields
    }


def validate_reasoning_summary_responses(responses: list[object]) -> None:
    """Validate response groups and cross-response fixture identity."""
    seen_continuation_ids: set[str] = set()
    for raw_response in responses:
        assert isinstance(raw_response, dict)
        continuation_id = _non_empty_string(
            raw_response.get("continuation_id"), "continuation_id"
        )
        assert (
            continuation_id not in seen_continuation_ids
        ), "continuation IDs must be unique within one trace"
        seen_continuation_ids.add(continuation_id)
        attempts = raw_response.get("attempts")
        events = raw_response.get("events")
        assert (attempts is None) != (events is None)
        if attempts is not None:
            assert isinstance(attempts, list) and attempts
            event_groups = attempts
        else:
            assert isinstance(events, list) and events
            event_groups = [events]
        for raw_event_group in event_groups:
            assert isinstance(raw_event_group, list) and raw_event_group
            _added_provider_item_ids(raw_event_group)
            cancel_after_event = (
                raw_response.get("cancel_after_event")
                if attempts is None
                else None
            )
            if cancel_after_event is not None:
                assert type(cancel_after_event) is int
            validate_reasoning_summary_event_group(
                raw_event_group,
                cancel_after_event=cancel_after_event,
            )


def _validate_positive_responses(responses: list[object]) -> None:
    validate_reasoning_summary_responses(responses)


def _validate_negative_responses(responses: list[object]) -> None:
    observed_ids: set[str] = set()
    for raw_response in responses:
        assert isinstance(raw_response, dict)
        assert set(raw_response) == {
            "continuation_id",
            "expected_error",
            "events",
        }
        continuation_id = _non_empty_string(
            raw_response.get("continuation_id"),
            "continuation_id",
        )
        assert continuation_id not in observed_ids
        observed_ids.add(continuation_id)
        assert continuation_id in _MALFORMED_ROW_CONTRACT
        expected_error = _non_empty_string(
            raw_response.get("expected_error"),
            "expected_error",
        )
        locked_error, locked_events_sha256 = _MALFORMED_ROW_CONTRACT[
            continuation_id
        ]
        assert expected_error == locked_error
        events = raw_response.get("events")
        assert isinstance(events, list) and events
        canonical_events = dumps(
            events,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        assert sha256(canonical_events).hexdigest() == locked_events_sha256
        try:
            validate_reasoning_summary_event_group(events)
        except AssertionError as exc:
            assert str(exc) == expected_error
        else:
            raise AssertionError(
                "malformed provider row must fail strict event validation"
            )
    assert observed_ids == set(_MALFORMED_ROW_CONTRACT)


def _added_provider_item_ids(events: list[object]) -> set[str]:
    item_ids: set[str] = set()
    for raw_event in events:
        assert isinstance(raw_event, dict)
        if raw_event.get("type") != "response.output_item.added":
            continue
        item = _event_item(cast(dict[str, object], raw_event))
        item_id = _non_empty_string(item.get("id"), "item.id")
        assert item_id not in item_ids
        item_ids.add(item_id)
    return item_ids


def _validate_provider_terminal_response(
    response: dict[str, object],
    *,
    event_type: str,
) -> None:
    assert event_type in _PROVIDER_TERMINAL_EVENT_TYPES
    status = response.get("status")
    error = response.get("error")
    incomplete_details = response.get("incomplete_details")
    if event_type == "response.completed":
        assert status == "completed"
        assert error is None
        assert incomplete_details is None
    elif event_type == "response.failed":
        assert status == "failed"
        assert isinstance(error, dict)
        assert set(error) == {"code", "message"}
        _non_empty_string(error.get("code"), "response.error.code")
        _non_empty_string(error.get("message"), "response.error.message")
        assert incomplete_details is None
    else:
        assert event_type == "response.incomplete"
        assert status == "incomplete"
        assert error is None
        assert isinstance(incomplete_details, dict)
        assert set(incomplete_details) == {"reason"}
        _non_empty_string(
            incomplete_details.get("reason"),
            "response.incomplete_details.reason",
        )

    created_at = response.get("created_at")
    assert type(created_at) is float
    assert isfinite(created_at) and created_at >= 0
    assert type(response.get("parallel_tool_calls")) is bool
    for field_name in ("temperature", "top_p"):
        value = response.get(field_name)
        assert value is None or type(value) is float, field_name
        if type(value) is float:
            assert isfinite(value), field_name

    usage = response.get("usage")
    if usage is None:
        return
    assert isinstance(usage, dict)
    input_tokens = _non_negative_index(
        usage.get("input_tokens"),
        "response.usage.input_tokens",
    )
    output_tokens = _non_negative_index(
        usage.get("output_tokens"),
        "response.usage.output_tokens",
    )
    total_tokens = _non_negative_index(
        usage.get("total_tokens"),
        "response.usage.total_tokens",
    )
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    assert isinstance(input_details, dict)
    assert isinstance(output_details, dict)
    cached_tokens = _non_negative_index(
        input_details.get("cached_tokens"),
        "response.usage.input_tokens_details.cached_tokens",
    )
    reasoning_tokens = _non_negative_index(
        output_details.get("reasoning_tokens"),
        "response.usage.output_tokens_details.reasoning_tokens",
    )
    assert cached_tokens <= input_tokens
    assert reasoning_tokens <= output_tokens
    assert total_tokens == input_tokens + output_tokens


def _correlated_item_key(event: dict[str, object]) -> tuple[str, int]:
    return (
        _non_empty_string(event.get("item_id"), "item_id"),
        _non_negative_index(event.get("output_index"), "output_index"),
    )


def _content_key(event: dict[str, object]) -> tuple[str, int, int]:
    item_id, output_index = _correlated_item_key(event)
    return (
        item_id,
        output_index,
        _non_negative_index(event.get("content_index"), "content_index"),
    )


def _event_item(event: dict[str, object]) -> dict[str, object]:
    item = event.get("item")
    assert isinstance(item, dict)
    _non_empty_string(item.get("id"), "item.id")
    _non_empty_string(item.get("type"), "item.type")
    return cast(dict[str, object], item)


def _item_key(
    item: dict[str, object], event: dict[str, object]
) -> tuple[str, int]:
    return (
        _non_empty_string(item.get("id"), "item.id"),
        _non_negative_index(event.get("output_index"), "output_index"),
    )


def _summary_key(event: dict[str, object]) -> tuple[str, int, int]:
    return (
        _non_empty_string(event.get("item_id"), "item_id"),
        _non_negative_index(event.get("output_index"), "output_index"),
        _non_negative_index(event.get("summary_index"), "summary_index"),
    )


def _non_empty_string(value: object, name: str) -> str:
    assert isinstance(value, str), f"{name} must be a string"
    assert value.strip(), f"{name} must not be empty"
    return value


def _non_negative_index(value: object, name: str) -> int:
    assert type(value) is int, f"{name} must be an integer"
    assert value >= 0, f"{name} must not be negative"
    return value
