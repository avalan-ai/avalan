"""Validate the locked reasoning-summary Phase 0 support artifacts."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
from json import dumps, loads
from math import ceil, isfinite
from pathlib import Path
from statistics import median
from subprocess import CompletedProcess
from typing import Any, cast

import pytest
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter
from reasoning_summary_fixtures import (
    REASONING_SUMMARY_TRACE_NAMES,
    IdentityTaggedReasoningRedactor,
    ReasoningPartIdentity,
    TaggedRedactedText,
    load_reasoning_summary_trace,
    reasoning_delta_expectation,
    reasoning_summary_events_before_cancellation,
    reasoning_summary_fixture_root,
    reasoning_summary_trace_responses,
    validate_mcp_reasoning_truncation,
    validate_reasoning_summary_event_group,
    validate_reasoning_summary_responses,
    validate_reasoning_summary_trace_json,
    validate_reasoning_summary_trace_payload,
)
from reasoning_summary_script_loader import (
    canonical_json_pointer,
    load_reasoning_summary_script,
    strict_json_loads,
    typed_json_path,
)
from reasoning_summary_script_loader import (
    json_mapping_entries as _json_mapping_entries,
)

from avalan.model.stream import StreamPerformanceBudget, StreamRetentionPolicy
from avalan.server.entities import (
    SKILL_CONTENT_REDACTION,
    ServerOutputRedactionSettings,
)

_BENCHMARK_SCRIPT = load_reasoning_summary_script(
    "benchmark_reasoning_summary"
)
_ACCEPTANCE_SCRIPT = load_reasoning_summary_script(
    "verify_reasoning_summary_acceptance"
)


_RESPONSE_STREAM_EVENT: TypeAdapter[ResponseStreamEvent] = TypeAdapter(
    ResponseStreamEvent
)
_PHASE0_MANIFEST_DIMENSIONS_SHA256 = (
    "8dd3ce6150b052cf59abd3fbd02c283870ab7ba94f77d040cbc8cb003e3e8480"
)
_PHASE0_MANIFEST_DIMENSION_NAMES = (
    "phase 0 infrastructure",
    "acceptance runner enforcement",
    "canonical native reasoning baseline",
    "OpenAI omission retry and replay baseline",
    "aggregate lifecycle baseline",
    "CLI privacy baseline",
    "protocol projection baseline",
)
_REQUIREMENTS_CATALOG_SHA256 = (
    "ffb70eee04a87b05ccf5202b6b55da3324184492a5553dd8ecf79313e7bfa368"
)
_ACCEPTANCE_INTEGER_CATALOG = (
    2,
    "f72fddf7fd023a272ef8ed8851676b4d18e6d7fe4c0720b66de01ba3259db14a",
)
_CONTRACT_INTEGER_CATALOG = (
    95,
    "9bdf2bc42dd48d9e14ed8281cdd5c7895b2e830edf493166c7927c26811f5bfd",
)
_CONTRACT_MAPPING_CATALOG = (
    54,
    "c29bc5e72b718fb3a08d172fd285213dafac93d60e02eca0957163a3f8d63e8a",
)
_BASELINE_INTEGER_CATALOG = (
    65,
    "ebb3e0bba7d612759b773bf38247a0643cd169789cc8c637c2cf9fa32035c91f",
)
_BASELINE_FLOAT_CATALOG = (
    61,
    "f3e11b389a34475410e14eb530ce557a5b83657e3d3a03cc429a6fe00b4db6b2",
)
_BASELINE_MAPPING_CATALOG = (
    17,
    "6cff31eb5ec5e387ed9a052a1daf3b91202778c84ada7fe4fa4e9754ff6db774",
)
_REQUIREMENTS_INTEGER_CATALOG = (
    155,
    "97946f4df3e823441f103e3d4c1fe6d61432710610e522b6bcfa85547a066d85",
)
_PROVIDER_TRACE_INTEGER_CATALOG = (
    358,
    "1b292ede2a5111d44be4a1ab2b57ac89405dd676d7986c293ed707bd466c229e",
)
_PROVIDER_TERMINAL_SCALAR_CATALOG = (
    69,
    "8ccfce00c1a7bf2cc4aa69a4db5d6fe72027664f627c0cada720513c7f2d61ae",
)
_PROVIDER_TERMINAL_EVENT_TYPES = frozenset(
    (
        "response.completed",
        "response.failed",
        "response.incomplete",
    )
)
_DUPLICATE_JSON_OBJECT_CASES = (
    '{"schema_version":false,"schema_version":1}',
    '{"schema_version":1,"schema_version":false}',
    '{"outer":{"count":false,"count":1}}',
    '{"outer":{"count":1,"count":false}}',
)
_NONFINITE_JSON_CONSTANTS = ("NaN", "Infinity", "-Infinity")


def _json_integer_entries(
    value: object,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str, tuple[str | int, ...], int], ...]:
    entries: list[tuple[str, tuple[str | int, ...], int]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            assert isinstance(key, str)
            entries.extend(_json_integer_entries(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            entries.extend(_json_integer_entries(child, (*path, index)))
    elif type(value) is int:
        pointer = canonical_json_pointer(path)
        entries.append((pointer, path, value))
    ordered = tuple(sorted(entries, key=lambda entry: entry[0]))
    identities = tuple(
        (pointer, typed_json_path(entry_path))
        for pointer, entry_path, _ in ordered
    )
    pointers = tuple(pointer for pointer, _, _ in ordered)
    assert len(pointers) == len(set(pointers))
    assert len(identities) == len(set(identities))
    return ordered


def _set_json_path(
    payload: object,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    assert path
    current = payload
    for component in path[:-1]:
        if isinstance(component, str):
            assert isinstance(current, dict)
            current = current[component]
        else:
            assert isinstance(current, list)
            current = current[component]
    leaf = path[-1]
    if isinstance(leaf, str):
        assert isinstance(current, dict)
        current[leaf] = value
    else:
        assert isinstance(current, list)
        current[leaf] = value


def _duplicate_json_name(
    source: str,
    *,
    name: str,
    serialized_value: str,
) -> str:
    needle = f'"{name}": {serialized_value}'
    assert source.count(needle) >= 1
    return source.replace(
        needle,
        f'"{name}": false, {needle}',
        1,
    )


def _inject_nonfinite_json_number(
    source: str,
    *,
    constant: str,
    container_marker: str | None = None,
) -> str:
    assert constant in _NONFINITE_JSON_CONSTANTS
    search_start = (
        0
        if container_marker is None
        else source.index(container_marker) + len(container_marker)
    )
    object_start = source.index("{", search_start)
    return (
        source[: object_start + 1]
        + f'"__nonfinite__": {constant},'
        + source[object_start + 1 :]
    )


def _json_float_entries(
    value: object,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str, tuple[str | int, ...], float], ...]:
    entries: list[tuple[str, tuple[str | int, ...], float]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            assert isinstance(key, str)
            entries.extend(_json_float_entries(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            entries.extend(_json_float_entries(child, (*path, index)))
    elif type(value) is float:
        pointer = canonical_json_pointer(path)
        entries.append((pointer, path, value))
    ordered = tuple(sorted(entries, key=lambda entry: entry[0]))
    identities = tuple(
        (pointer, typed_json_path(entry_path))
        for pointer, entry_path, _ in ordered
    )
    pointers = tuple(pointer for pointer, _, _ in ordered)
    assert len(pointers) == len(set(pointers))
    assert len(identities) == len(set(identities))
    return ordered


def _assert_exact_json_integer_catalog(
    payload: object,
    expected: tuple[int, str],
) -> None:
    expected_count, expected_sha256 = expected
    assert type(expected_count) is int
    entries = _json_integer_entries(payload)
    assert len(entries) == expected_count
    canonical = dumps(
        [
            [pointer, typed_json_path(path), value]
            for pointer, path, value in entries
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert sha256(canonical).hexdigest() == expected_sha256


def _assert_integer_type_mutations_fail(
    payload: object,
    expected: tuple[int, str],
) -> None:
    for _, path, integer_value in _json_integer_entries(payload):
        for invalid_value in (False, True, float(integer_value)):
            mutation = deepcopy(payload)
            _set_json_path(mutation, path, invalid_value)
            with pytest.raises(AssertionError):
                _assert_exact_json_integer_catalog(mutation, expected)


def _assert_exact_json_float_catalog(
    payload: object,
    expected: tuple[int, str],
) -> None:
    expected_count, expected_sha256 = expected
    assert type(expected_count) is int
    entries = _json_float_entries(payload)
    assert len(entries) == expected_count
    assert all(isfinite(value) for _, _, value in entries)
    canonical = dumps(
        [
            [pointer, typed_json_path(path), value]
            for pointer, path, value in entries
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert sha256(canonical).hexdigest() == expected_sha256


def _assert_float_type_mutations_fail(
    payload: object,
    expected: tuple[int, str],
) -> None:
    for _, path, float_value in _json_float_entries(payload):
        for invalid_value in (
            False,
            True,
            int(float_value),
            str(float_value),
        ):
            mutation = deepcopy(payload)
            _set_json_path(mutation, path, invalid_value)
            with pytest.raises(AssertionError):
                _assert_exact_json_float_catalog(mutation, expected)


def _assert_exact_json_mapping_catalog(
    payload: object,
    expected: tuple[int, str],
) -> None:
    expected_count, expected_sha256 = expected
    assert type(expected_count) is int
    entries = _json_mapping_entries(payload)
    assert len(entries) == expected_count
    canonical = dumps(
        [[pointer, list(keys)] for pointer, _, keys in entries],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert sha256(canonical).hexdigest() == expected_sha256


def _add_unknown_mapping_key(
    payload: object,
    path: tuple[str | int, ...],
) -> None:
    current = payload
    for component in path:
        if isinstance(component, str):
            assert isinstance(current, dict)
            current = current[component]
        else:
            assert isinstance(current, list)
            current = current[component]
    assert isinstance(current, dict)
    assert "__unexpected__" not in current
    current["__unexpected__"] = None


def _assert_phase0_contract_payload(payload: object) -> dict[str, object]:
    assert isinstance(payload, dict)
    assert set(payload) == {
        "schema_version",
        "feature",
        "activation",
        "canonical_reasoning_identity",
        "lifecycle",
        "retention",
        "retention_semantics",
        "stream_retention_policy_extensions",
        "protocol_shapes",
        "abnormal_terminal_shapes",
        "benchmark",
        "acceptance",
    }
    schema_version = payload.get("schema_version")
    assert type(schema_version) is int and schema_version == 1
    _assert_exact_json_integer_catalog(payload, _CONTRACT_INTEGER_CATALOG)
    _assert_exact_json_mapping_catalog(payload, _CONTRACT_MAPPING_CATALOG)
    return cast(dict[str, object], payload)


def _contract() -> dict[str, object]:
    path = reasoning_summary_fixture_root() / "phase0_contract.json"
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    return _assert_phase0_contract_payload(payload)


def _assert_phase0_baseline_payload(payload: object) -> dict[str, object]:
    assert isinstance(payload, dict)
    assert set(payload) == {
        "schema_version",
        "feature",
        "recorded_at",
        "git",
        "ignored_specs",
        "quality_gate",
        "preexisting_stream_benchmark",
        "locked_runner",
    }
    schema_version = payload.get("schema_version")
    assert type(schema_version) is int and schema_version == 1
    _assert_exact_json_integer_catalog(payload, _BASELINE_INTEGER_CATALOG)
    _assert_exact_json_float_catalog(payload, _BASELINE_FLOAT_CATALOG)
    _assert_exact_json_mapping_catalog(payload, _BASELINE_MAPPING_CATALOG)
    return cast(dict[str, object], payload)


def _provider_terminal_scalar_entries(
    payload: object,
) -> tuple[
    tuple[str, str, tuple[str | int, ...], object],
    ...,
]:
    entries: list[tuple[str, str, tuple[str | int, ...], object]] = []

    def append_entry(
        label: str,
        path: tuple[str | int, ...],
        value: object,
    ) -> None:
        pointer = canonical_json_pointer(path)
        entries.append((label, pointer, path, value))

    def visit(value: object, path: tuple[str | int, ...]) -> None:
        if isinstance(value, dict):
            if value.get("type") in _PROVIDER_TERMINAL_EVENT_TYPES:
                response = value.get("response")
                assert isinstance(response, dict)
                response_path = (*path, "response")
                for label, field_name in (
                    ("created_at_float", "created_at"),
                    ("parallel_tool_calls_bool", "parallel_tool_calls"),
                    ("nullable_float", "temperature"),
                    ("nullable_float", "top_p"),
                ):
                    assert field_name in response
                    append_entry(
                        label,
                        (*response_path, field_name),
                        response[field_name],
                    )
                usage = response.get("usage")
                if usage is not None:
                    assert isinstance(usage, dict)
                    for usage_path in (
                        ("input_tokens",),
                        ("input_tokens_details", "cached_tokens"),
                        ("output_tokens",),
                        ("output_tokens_details", "reasoning_tokens"),
                        ("total_tokens",),
                    ):
                        current: object = usage
                        for component in usage_path:
                            assert isinstance(current, dict)
                            current = current[component]
                        append_entry(
                            "usage_integer",
                            (*response_path, "usage", *usage_path),
                            current,
                        )
            for key, child in value.items():
                assert isinstance(key, str)
                visit(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, index))

    visit(payload, ())
    ordered = tuple(sorted(entries, key=lambda entry: entry[1]))
    identities = tuple(
        (pointer, typed_json_path(path)) for _, pointer, path, _ in ordered
    )
    pointers = tuple(pointer for _, pointer, _, _ in ordered)
    assert len(pointers) == len(set(pointers))
    assert len(identities) == len(set(identities))
    return ordered


def _provider_terminal_event_entries(
    payload: object,
) -> tuple[tuple[tuple[str | int, ...], str], ...]:
    entries: list[tuple[tuple[str | int, ...], str]] = []

    def visit(value: object, path: tuple[str | int, ...]) -> None:
        if isinstance(value, dict):
            event_type = value.get("type")
            if event_type in _PROVIDER_TERMINAL_EVENT_TYPES:
                assert isinstance(event_type, str)
                entries.append((path, event_type))
            for key, child in value.items():
                assert isinstance(key, str)
                visit(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, index))

    visit(payload, ())
    return tuple(entries)


def _assert_provider_terminal_scalar_catalog(payload: object) -> None:
    expected_count, expected_sha256 = _PROVIDER_TERMINAL_SCALAR_CATALOG
    entries = _provider_terminal_scalar_entries(payload)
    assert len(entries) == expected_count
    assert {
        label: sum(entry[0] == label for entry in entries)
        for label in {
            "created_at_float",
            "parallel_tool_calls_bool",
            "nullable_float",
            "usage_integer",
        }
    } == {
        "created_at_float": 16,
        "parallel_tool_calls_bool": 16,
        "nullable_float": 32,
        "usage_integer": 5,
    }
    canonical = dumps(
        [
            [
                label,
                pointer,
                typed_json_path(path),
                type(value).__name__,
                value,
            ]
            for label, pointer, path, value in entries
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert sha256(canonical).hexdigest() == expected_sha256


def _assert_exact_phase0_manifest_catalog(
    dimensions: dict[str, tuple[str, ...]],
) -> None:
    assert set(_PHASE0_MANIFEST_DIMENSION_NAMES).issubset(dimensions)
    phase0_dimensions = {
        dimension: dimensions[dimension]
        for dimension in _PHASE0_MANIFEST_DIMENSION_NAMES
    }
    node_ids = tuple(
        node_id for nodes in phase0_dimensions.values() for node_id in nodes
    )
    assert len(node_ids) == 49
    assert len(set(node_ids)) == 49
    canonical = dumps(
        {
            dimension: list(nodes)
            for dimension, nodes in phase0_dimensions.items()
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert sha256(canonical).hexdigest() == _PHASE0_MANIFEST_DIMENSIONS_SHA256


def _assert_self_contained_requirements_catalog(
    payload: dict[str, object],
) -> None:
    assert set(payload) == {
        "schema_version",
        "feature",
        "catalog_invariant",
        "normative_requirements",
        "acceptance_criteria",
    }
    schema_version = payload["schema_version"]
    assert type(schema_version) is int and schema_version == 1
    assert payload["feature"] == "reasoning_summary"
    invariant = cast(dict[str, object], payload["catalog_invariant"])
    assert set(invariant) == {
        "canonicalization",
        "normative_count",
        "acceptance_count",
        "sha256",
    }
    assert invariant["canonicalization"] == "compact_sorted_keys_utf8_json_v1"
    normative_count = invariant["normative_count"]
    acceptance_count = invariant["acceptance_count"]
    assert type(normative_count) is int and normative_count == 54
    assert type(acceptance_count) is int and acceptance_count == 22
    assert invariant["sha256"] == _REQUIREMENTS_CATALOG_SHA256
    raw_normative = payload["normative_requirements"]
    raw_acceptance = payload["acceptance_criteria"]
    assert isinstance(raw_normative, list)
    assert isinstance(raw_acceptance, list)
    assert all(isinstance(entry, dict) for entry in raw_normative)
    assert all(isinstance(entry, dict) for entry in raw_acceptance)
    normative = cast(list[dict[str, object]], raw_normative)
    acceptance = cast(list[dict[str, object]], raw_acceptance)
    assert len(normative) == normative_count
    assert len(acceptance) == acceptance_count
    assert [entry["id"] for entry in normative] == [
        f"RS-N-{ordinal:03d}" for ordinal in range(1, 55)
    ]
    assert [entry["id"] for entry in acceptance] == [
        f"RS-A-{ordinal:03d}" for ordinal in range(1, 23)
    ]

    entries = [*normative, *acceptance]
    for entry in entries:
        assert set(entry) == {
            "id",
            "source_section",
            "source_line",
            "phase",
            "test_target",
        }
        source_section = entry["source_section"]
        assert isinstance(source_section, str) and source_section
        source_line = entry["source_line"]
        assert type(source_line) is int and source_line > 0
        phase = entry["phase"]
        assert type(phase) is int and 0 <= phase <= 9
        target = entry["test_target"]
        assert isinstance(target, str)
        assert target.startswith("tests/") and ".py::" in target

    canonical = dumps(
        {
            "normative_requirements": normative,
            "acceptance_criteria": acceptance,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert sha256(canonical).hexdigest() == invariant["sha256"]
    assert invariant["sha256"] == _REQUIREMENTS_CATALOG_SHA256


def _event_groups(response: dict[str, object]) -> tuple[list[object], ...]:
    attempts = response.get("attempts")
    if attempts is not None:
        assert isinstance(attempts, list)
        assert all(isinstance(attempt, list) for attempt in attempts)
        return tuple(attempts)
    events = response.get("events")
    assert isinstance(events, list)
    return (events,)


def _trace_events(
    name: str,
    response_index: int = 0,
) -> tuple[dict[str, object], ...]:
    response = reasoning_summary_trace_responses(name)[response_index]
    events = response.get("events")
    assert isinstance(events, list)
    assert all(isinstance(event, dict) for event in events)
    return tuple(cast(dict[str, object], event) for event in events)


def _event_types(events: tuple[dict[str, object], ...]) -> tuple[str, ...]:
    event_types: list[str] = []
    for event in events:
        event_type = event.get("type")
        assert isinstance(event_type, str)
        event_types.append(event_type)
    return tuple(event_types)


def test_provider_trace_inventory_is_complete_and_valid() -> None:
    fixture_names = {
        path.stem
        for path in (
            reasoning_summary_fixture_root() / "provider_traces"
        ).glob("*.json")
    }
    assert fixture_names == set(REASONING_SUMMARY_TRACE_NAMES)

    trace_payloads = {
        name: load_reasoning_summary_trace(name)
        for name in REASONING_SUMMARY_TRACE_NAMES
    }
    one_part_source = dumps(trace_payloads["one_part"])
    duplicate_trace_sources = (
        _duplicate_json_name(
            one_part_source,
            name="schema_version",
            serialized_value="1",
        ),
        _duplicate_json_name(
            one_part_source,
            name="sequence_number",
            serialized_value="0",
        ),
    )
    for duplicate_source in duplicate_trace_sources:
        with pytest.raises(ValueError, match="duplicate JSON object name"):
            validate_reasoning_summary_trace_json(
                duplicate_source,
                name="one_part",
            )
    for constant in _NONFINITE_JSON_CONSTANTS:
        for container_marker in (None, '"response":'):
            nonfinite_source = _inject_nonfinite_json_number(
                one_part_source,
                constant=constant,
                container_marker=container_marker,
            )
            with pytest.raises(ValueError, match="non-finite JSON number"):
                validate_reasoning_summary_trace_json(
                    nonfinite_source,
                    name="one_part",
                )
    _assert_exact_json_integer_catalog(
        trace_payloads,
        _PROVIDER_TRACE_INTEGER_CATALOG,
    )
    _assert_integer_type_mutations_fail(
        trace_payloads,
        _PROVIDER_TRACE_INTEGER_CATALOG,
    )
    _assert_provider_terminal_scalar_catalog(trace_payloads)

    provider_mapping_mutations = 0
    for name, payload in trace_payloads.items():
        for _, path, _ in _json_mapping_entries(payload):
            mapping_mutation = deepcopy(payload)
            _add_unknown_mapping_key(mapping_mutation, path)
            with pytest.raises(AssertionError):
                validate_reasoning_summary_trace_payload(
                    mapping_mutation,
                    name=name,
                )
            provider_mapping_mutations += 1
    assert provider_mapping_mutations == 343

    direct_loader_integer_mutations = 0
    for name, payload in trace_payloads.items():
        integer_entries = _json_integer_entries(payload)
        if name == "malformed":
            integer_entries = tuple(
                entry
                for entry in integer_entries
                if entry[1] == ("schema_version",)
            )
        for _, path, integer_value in integer_entries:
            for invalid_integer_value in (
                False,
                True,
                float(integer_value),
            ):
                integer_mutation = deepcopy(payload)
                _set_json_path(
                    integer_mutation,
                    path,
                    invalid_integer_value,
                )
                with pytest.raises(AssertionError):
                    validate_reasoning_summary_trace_payload(
                        integer_mutation,
                        name=name,
                    )
                direct_loader_integer_mutations += 1
    assert direct_loader_integer_mutations == 981

    terminal_scalar_mutations = 0
    for label, _, path, value in _provider_terminal_scalar_entries(
        trace_payloads
    ):
        invalid_values: tuple[object, ...]
        if label == "created_at_float":
            assert type(value) is float
            invalid_values = (
                False,
                True,
                int(value),
                str(value),
                -1.0,
                float("inf"),
                float("nan"),
            )
        elif label == "parallel_tool_calls_bool":
            assert type(value) is bool
            invalid_values = (0, 1, 0.0, 1.0, "true", None)
        elif label == "nullable_float":
            assert value is None
            invalid_values = (
                False,
                True,
                0,
                "0.0",
                float("inf"),
                float("nan"),
            )
        else:
            assert label == "usage_integer"
            assert type(value) is int
            invalid_values = (-1, str(value))
        for invalid_terminal_value in invalid_values:
            terminal_mutation = deepcopy(trace_payloads)
            _set_json_path(
                terminal_mutation,
                path,
                invalid_terminal_value,
            )
            trace_name = path[0]
            assert isinstance(trace_name, str)
            mutated_payload = terminal_mutation[trace_name]
            assert isinstance(mutated_payload, dict)
            with pytest.raises(AssertionError):
                validate_reasoning_summary_trace_payload(
                    mutated_payload,
                    name=trace_name,
                )
            terminal_scalar_mutations += 1
    assert terminal_scalar_mutations == 410

    terminal_events = _provider_terminal_event_entries(trace_payloads)
    assert len(terminal_events) == 16
    terminal_state_mutations = 0
    terminal_types = tuple(sorted(_PROVIDER_TERMINAL_EVENT_TYPES))
    terminal_statuses = ("completed", "failed", "incomplete")
    error_states: tuple[object, ...] = (
        None,
        {"code": "server_error", "message": "synthetic"},
    )
    incomplete_states: tuple[object, ...] = (
        None,
        {"reason": "max_output_tokens"},
    )
    for event_path, _ in terminal_events:
        for event_type in terminal_types:
            for status in terminal_statuses:
                for error in error_states:
                    for incomplete_details in incomplete_states:
                        valid_state = (
                            (
                                event_type == "response.completed"
                                and status == "completed"
                                and error is None
                                and incomplete_details is None
                            )
                            or (
                                event_type == "response.failed"
                                and status == "failed"
                                and error is not None
                                and incomplete_details is None
                            )
                            or (
                                event_type == "response.incomplete"
                                and status == "incomplete"
                                and error is None
                                and incomplete_details is not None
                            )
                        )
                        if valid_state:
                            continue
                        state_mutation = deepcopy(trace_payloads)
                        _set_json_path(
                            state_mutation,
                            (*event_path, "type"),
                            event_type,
                        )
                        _set_json_path(
                            state_mutation,
                            (*event_path, "response", "status"),
                            status,
                        )
                        _set_json_path(
                            state_mutation,
                            (*event_path, "response", "error"),
                            deepcopy(error),
                        )
                        _set_json_path(
                            state_mutation,
                            (
                                *event_path,
                                "response",
                                "incomplete_details",
                            ),
                            deepcopy(incomplete_details),
                        )
                        trace_name = event_path[0]
                        assert isinstance(trace_name, str)
                        mutated_payload = state_mutation[trace_name]
                        assert isinstance(mutated_payload, dict)
                        with pytest.raises(AssertionError):
                            validate_reasoning_summary_trace_payload(
                                mutated_payload,
                                name=trace_name,
                            )
                        terminal_state_mutations += 1
    assert terminal_state_mutations == 528

    wrapper_mutations = 0
    for name, payload in trace_payloads.items():
        if name != "malformed":
            poisoned = deepcopy(payload)
            poisoned["expected"] = "invalid"
            with pytest.raises(AssertionError):
                validate_reasoning_summary_trace_payload(
                    poisoned,
                    name=name,
                )
            wrapper_mutations += 1

    malformed = trace_payloads["malformed"]
    for invalid_expected in (None, "valid", False):
        invalid_malformed = deepcopy(malformed)
        if invalid_expected is None:
            invalid_malformed.pop("expected")
        else:
            invalid_malformed["expected"] = invalid_expected
        with pytest.raises(AssertionError):
            validate_reasoning_summary_trace_payload(
                invalid_malformed,
                name="malformed",
            )
        wrapper_mutations += 1

    nonfailing_negative = deepcopy(malformed)
    malformed_rows = nonfailing_negative["responses"]
    positive_rows = trace_payloads["one_part"]["responses"]
    assert isinstance(malformed_rows, list)
    assert isinstance(positive_rows, list)
    malformed_row = malformed_rows[0]
    positive_row = positive_rows[0]
    assert isinstance(malformed_row, dict)
    assert isinstance(positive_row, dict)
    malformed_row["events"] = deepcopy(positive_row["events"])
    with pytest.raises(AssertionError):
        validate_reasoning_summary_trace_payload(
            nonfailing_negative,
            name="malformed",
        )
    wrapper_mutations += 1
    assert wrapper_mutations == 17

    for name in REASONING_SUMMARY_TRACE_NAMES:
        payload = trace_payloads[name]
        assert payload["name"] == name
        for response in reasoning_summary_trace_responses(name):
            for events in _event_groups(response):
                sequence_numbers: list[int] = []
                for raw_event in events:
                    assert isinstance(raw_event, dict)
                    event_type = raw_event["type"]
                    assert isinstance(event_type, str)
                    sequence_number = raw_event.get("sequence_number")
                    assert type(sequence_number) is int
                    assert sequence_number >= 0
                    sequence_numbers.append(sequence_number)
                    if name != "malformed":
                        _RESPONSE_STREAM_EVENT.validate_python(raw_event)
                assert all(
                    current < following
                    for current, following in zip(
                        sequence_numbers,
                        sequence_numbers[1:],
                        strict=False,
                    )
                )

    empty_events = _trace_events("empty")
    assert not any(
        "reasoning_summary_" in kind for kind in _event_types(empty_events)
    )
    empty_done = cast(dict[str, object], empty_events[1]["item"])
    assert empty_done["summary"] == []

    fallback_events = _trace_events("fallback")
    assert not any(
        "reasoning_summary_" in kind for kind in _event_types(fallback_events)
    )
    fallback_done = cast(dict[str, object], fallback_events[1]["item"])
    assert fallback_done["summary"] == [
        {
            "type": "summary_text",
            "text": "Recovered from the completed item.",
        }
    ]

    zero_length_events = _trace_events("zero_length_fallback")
    zero_length_deltas = [
        event["delta"]
        for event in zero_length_events
        if event.get("type") == "response.reasoning_summary_text.delta"
    ]
    assert zero_length_deltas == [""]
    assert not any(zero_length_deltas)
    zero_length_done = cast(dict[str, object], zero_length_events[-2]["item"])
    assert zero_length_done["summary"] == [
        {
            "type": "summary_text",
            "text": "Recovered after empty delta.",
        }
    ]

    multipart_events = _trace_events("multipart")
    assert [
        event["summary_index"]
        for event in multipart_events
        if event.get("type") == "response.reasoning_summary_part.added"
    ] == [0, 1]
    multipart_done = cast(dict[str, object], multipart_events[-2]["item"])
    assert multipart_done["summary"] == [
        {"type": "summary_text", "text": "Check records."},
        {"type": "summary_text", "text": "Choose the newest."},
    ]

    mixed_events = _trace_events("mixed_fallback")
    mixed_done = cast(dict[str, object], mixed_events[-2]["item"])
    assert mixed_done["summary"] == [
        {"type": "summary_text", "text": "Streamed part."},
        {"type": "summary_text", "text": "Fallback-only part."},
    ]
    assert [
        event["summary_index"]
        for event in mixed_events
        if event.get("type") == "response.reasoning_summary_text.delta"
    ] == [0]

    retry = reasoning_summary_trace_responses("retry")[0]
    attempts = retry.get("attempts")
    assert isinstance(attempts, list) and len(attempts) == 2
    assert _event_types(
        tuple(cast(dict[str, object], event) for event in attempts[0])
    ) == ("response.failed",)
    assert (
        _event_types(
            tuple(cast(dict[str, object], event) for event in attempts[1])
        )[-1]
        == "response.completed"
    )

    failed_events = _trace_events("failure_after_summary")
    failed_types = _event_types(failed_events)
    assert "response.reasoning_summary_text.delta" in failed_types
    assert failed_types[-1] == "response.failed"
    assert "response.output_item.done" not in failed_types

    incomplete_events = _trace_events("incomplete_after_summary")
    incomplete_types = _event_types(incomplete_events)
    assert "response.reasoning_summary_text.delta" in incomplete_types
    assert incomplete_types[-1] == "response.incomplete"
    incomplete_terminal = cast(
        dict[str, object], incomplete_events[-1]["response"]
    )
    assert incomplete_terminal["status"] == "incomplete"
    assert incomplete_terminal["incomplete_details"] == {
        "reason": "max_output_tokens"
    }

    tool_responses = reasoning_summary_trace_responses("tools_answer")
    assert len(tool_responses) == 2
    assert "response.function_call_arguments.done" in _event_types(
        _trace_events("tools_answer", 0)
    )
    assert "response.output_text.done" in _event_types(
        _trace_events("tools_answer", 1)
    )

    sparse_events = _trace_events("sparse_indices")
    assert [
        event["summary_index"]
        for event in sparse_events
        if event.get("type") == "response.reasoning_summary_part.added"
    ] == [2, 7]
    assert {
        event["output_index"]
        for event in sparse_events
        if "output_index" in event
    } == {4}
    sparse_done = cast(dict[str, object], sparse_events[-2]["item"])
    sparse_summary = sparse_done["summary"]
    assert isinstance(sparse_summary, list) and len(sparse_summary) == 8
    assert (
        cast(dict[str, object], sparse_summary[2])["text"]
        == "Earlier sparse part."
    )
    assert (
        cast(dict[str, object], sparse_summary[7])["text"]
        == "Later sparse part."
    )
    assert [index for index in range(8) if index not in {2, 7}] == [
        0,
        1,
        3,
        4,
        5,
        6,
    ]

    continuations = reasoning_summary_trace_responses("multi_continuation")
    assert [response["continuation_id"] for response in continuations] == [
        "continuation-1",
        "continuation-2",
        "continuation-3",
    ]
    assert [
        _event_types(_trace_events("multi_continuation", index))[-1]
        for index in range(3)
    ] == ["response.completed"] * 3
    assert [
        _trace_events("multi_continuation", index)[0]["output_index"]
        for index in range(3)
    ] == [0, 0, 0]
    provider_reasoning_ids = [
        cast(
            dict[str, object],
            _trace_events("multi_continuation", index)[0]["item"],
        )["id"]
        for index in range(3)
    ]
    assert provider_reasoning_ids == ["rs_1", "rs_1", "rs_3"]
    outward_template = cast(
        str,
        cast(dict[str, object], _contract()["protocol_shapes"])[
            "responses_reasoning_item_id"
        ],
    )
    outward_ids = {
        outward_template.format(
            outer_response_id="resp-outer",
            outward_output_index=index,
        )
        for index in (0, 2, 4)
    }
    assert len(outward_ids) == 3


def test_positive_summary_traces_obey_part_before_delta_order() -> None:
    for name in REASONING_SUMMARY_TRACE_NAMES:
        if name in {"fallback", "empty", "malformed"}:
            continue
        for response in reasoning_summary_trace_responses(name):
            for events in _event_groups(response):
                opened_parts: set[tuple[str, int, int]] = set()
                for raw_event in events:
                    assert isinstance(raw_event, dict)
                    event_type = raw_event.get("type")
                    if not isinstance(event_type, str):
                        continue
                    if not event_type.startswith(
                        "response.reasoning_summary_"
                    ):
                        continue
                    key = (
                        str(raw_event.get("item_id")),
                        int(raw_event.get("output_index", -1)),
                        int(raw_event.get("summary_index", -1)),
                    )
                    if event_type.endswith("part.added"):
                        opened_parts.add(key)
                    else:
                        assert key in opened_parts


def test_malformed_trace_locks_required_negative_provider_shapes() -> None:
    responses = reasoning_summary_trace_responses("malformed")
    assert {response["continuation_id"] for response in responses} == {
        "bad-index-type",
        "bad-negative-index",
        "bad-missing-id",
        "bad-delta-type",
    }
    for response in responses:
        events = response["events"]
        expected_error = response["expected_error"]
        assert isinstance(events, list)
        assert isinstance(expected_error, str)
        with pytest.raises(AssertionError, match=expected_error):
            validate_reasoning_summary_event_group(events)

    malformed_payload = load_reasoning_summary_trace("malformed")
    drifted_event_type = deepcopy(malformed_payload)
    drifted_rows = drifted_event_type["responses"]
    assert isinstance(drifted_rows, list)
    drifted_row = drifted_rows[-1]
    assert isinstance(drifted_row, dict)
    drifted_events = drifted_row["events"]
    assert isinstance(drifted_events, list)
    drifted_event = drifted_events[-1]
    assert isinstance(drifted_event, dict)
    drifted_event["type"] = "response.delta_unknown"
    with pytest.raises(AssertionError):
        validate_reasoning_summary_trace_payload(
            drifted_event_type,
            name="malformed",
        )

    broad_error = deepcopy(malformed_payload)
    broad_rows = broad_error["responses"]
    assert isinstance(broad_rows, list)
    broad_row = broad_rows[-1]
    assert isinstance(broad_row, dict)
    broad_row["expected_error"] = "delta"
    with pytest.raises(AssertionError):
        validate_reasoning_summary_trace_payload(
            broad_error,
            name="malformed",
        )


def test_strict_trace_validator_rejects_contract_mutations() -> None:
    response = reasoning_summary_trace_responses("one_part")[0]
    raw_events = response["events"]
    assert isinstance(raw_events, list)
    mutations: list[list[object]] = []

    duplicate_sequence = deepcopy(raw_events)
    cast(dict[str, object], duplicate_sequence[1])["sequence_number"] = 0
    mutations.append(duplicate_sequence)

    unknown_event = deepcopy(raw_events)
    cast(dict[str, object], unknown_event[2])["type"] = "response.unknown"
    mutations.append(unknown_event)

    conflicting_identity = deepcopy(raw_events)
    cast(dict[str, object], conflicting_identity[2])["item_id"] = "rs_other"
    mutations.append(conflicting_identity)

    premature_part_done = deepcopy(raw_events)
    cast(dict[str, object], premature_part_done[3]).update(
        {
            "type": "response.reasoning_summary_part.done",
            "part": {"type": "summary_text", "text": "Inspect inputs."},
        }
    )
    mutations.append(premature_part_done)

    completed_text_mismatch = deepcopy(raw_events)
    done_event = cast(dict[str, object], completed_text_mismatch[5])
    done_item = cast(dict[str, object], done_event["item"])
    done_summary = cast(list[object], done_item["summary"])
    cast(dict[str, object], done_summary[0])["text"] = "different"
    mutations.append(completed_text_mismatch)

    missing_streamed_part = deepcopy(raw_events)
    missing_done_event = cast(dict[str, object], missing_streamed_part[5])
    missing_done_item = cast(dict[str, object], missing_done_event["item"])
    missing_done_item["summary"] = []
    missing_terminal_event = cast(dict[str, object], missing_streamed_part[6])
    missing_response = cast(
        dict[str, object], missing_terminal_event["response"]
    )
    missing_output = cast(list[object], missing_response["output"])
    cast(dict[str, object], missing_output[0])["summary"] = []
    mutations.append(missing_streamed_part)

    missing_terminal = deepcopy(raw_events[:-1])
    mutations.append(missing_terminal)

    post_terminal = deepcopy(raw_events)
    late = deepcopy(cast(dict[str, object], raw_events[2]))
    late["sequence_number"] = 7
    post_terminal.append(late)
    mutations.append(post_terminal)

    for events in mutations:
        with pytest.raises(AssertionError):
            validate_reasoning_summary_event_group(events)

    with pytest.raises(AssertionError):
        validate_reasoning_summary_event_group([])

    tool_events = list(_trace_events("tools_answer", 0))
    wrong_function_item = deepcopy(tool_events)
    wrong_function_item[7].update(
        {
            "item_id": "rs_tool",
            "output_index": 0,
        }
    )
    with pytest.raises(AssertionError, match="function-call item"):
        validate_reasoning_summary_event_group(wrong_function_item)

    answer_events = list(_trace_events("tools_answer", 1))
    wrong_content_item = deepcopy(answer_events)
    wrong_content_item[7].update(
        {
            "item_id": "rs_answer",
            "output_index": 0,
        }
    )
    with pytest.raises(AssertionError, match="message item"):
        validate_reasoning_summary_event_group(wrong_content_item)

    duplicate_item_id = deepcopy(tool_events)
    duplicate_added = duplicate_item_id[6]
    duplicate_added_item = cast(dict[str, object], duplicate_added["item"])
    duplicate_added_item["id"] = "rs_tool"
    with pytest.raises(AssertionError, match="item ID reused"):
        validate_reasoning_summary_event_group(duplicate_item_id)

    duplicate_continuation = deepcopy(
        list(reasoning_summary_trace_responses("tools_answer"))
    )
    duplicate_continuation[1]["continuation_id"] = duplicate_continuation[0][
        "continuation_id"
    ]
    with pytest.raises(AssertionError, match="continuation IDs"):
        validate_reasoning_summary_responses(
            cast(list[object], duplicate_continuation)
        )

    truncated_sparse = deepcopy(list(_trace_events("sparse_indices")))
    sparse_done_event = truncated_sparse[-2]
    sparse_done_item = cast(dict[str, object], sparse_done_event["item"])
    sparse_summary = cast(list[object], sparse_done_item["summary"])
    sparse_done_item["summary"] = sparse_summary[:2]
    sparse_terminal = cast(dict[str, object], truncated_sparse[-1]["response"])
    sparse_output = cast(list[object], sparse_terminal["output"])
    cast(dict[str, object], sparse_output[0])["summary"] = sparse_summary[:2]
    with pytest.raises(AssertionError, match="missing from completed item"):
        validate_reasoning_summary_event_group(truncated_sparse)

    correct_arguments_call_id = deepcopy(tool_events)
    correct_arguments_call_id[8]["call_id"] = "call-tool"
    validate_reasoning_summary_event_group(correct_arguments_call_id)

    function_identity_mutations: list[list[dict[str, object]]] = []
    for field_name, invalid_value in (
        ("name", "other_lookup"),
        ("call_id", "call-other"),
    ):
        arguments_done_mutation = deepcopy(tool_events)
        arguments_done_mutation[8][field_name] = invalid_value
        function_identity_mutations.append(arguments_done_mutation)

        item_done_mutation = deepcopy(tool_events)
        item_done = cast(dict[str, object], item_done_mutation[9]["item"])
        item_done[field_name] = invalid_value
        function_identity_mutations.append(item_done_mutation)

        terminal_mutation = deepcopy(tool_events)
        terminal_response = cast(
            dict[str, object], terminal_mutation[10]["response"]
        )
        terminal_output = cast(
            list[dict[str, object]], terminal_response["output"]
        )
        terminal_output[1][field_name] = invalid_value
        function_identity_mutations.append(terminal_mutation)

        joint_mutation = deepcopy(tool_events)
        joint_mutation[8][field_name] = invalid_value
        joint_item_done = cast(dict[str, object], joint_mutation[9]["item"])
        joint_item_done[field_name] = invalid_value
        joint_terminal_response = cast(
            dict[str, object], joint_mutation[10]["response"]
        )
        joint_terminal_output = cast(
            list[dict[str, object]], joint_terminal_response["output"]
        )
        joint_terminal_output[1][field_name] = invalid_value
        function_identity_mutations.append(joint_mutation)

    for function_identity_mutation in function_identity_mutations:
        with pytest.raises(AssertionError):
            validate_reasoning_summary_event_group(function_identity_mutation)
    assert len(function_identity_mutations) == 8

    function_domain_mutations: list[list[dict[str, object]]] = []
    for field_name in ("name", "call_id"):
        for invalid_value in ("", " \t"):
            domain_mutation = deepcopy(tool_events)
            domain_added_item = cast(
                dict[str, object], domain_mutation[6]["item"]
            )
            domain_added_item[field_name] = invalid_value
            domain_mutation[8][field_name] = invalid_value
            domain_done_item = cast(
                dict[str, object], domain_mutation[9]["item"]
            )
            domain_done_item[field_name] = invalid_value
            domain_terminal_response = cast(
                dict[str, object], domain_mutation[10]["response"]
            )
            domain_terminal_output = cast(
                list[dict[str, object]],
                domain_terminal_response["output"],
            )
            domain_terminal_output[1][field_name] = invalid_value
            function_domain_mutations.append(domain_mutation)

    for function_domain_mutation in function_domain_mutations:
        with pytest.raises(AssertionError):
            validate_reasoning_summary_event_group(function_domain_mutation)
    assert len(function_domain_mutations) == 4

    message_role_mutations: list[list[dict[str, object]]] = []
    for mutate_done, mutate_terminal in (
        (True, False),
        (False, True),
        (True, True),
    ):
        role_mutation = deepcopy(answer_events)
        if mutate_done:
            role_done_item = cast(dict[str, object], role_mutation[11]["item"])
            role_done_item["role"] = "user"
        if mutate_terminal:
            role_terminal_response = cast(
                dict[str, object], role_mutation[12]["response"]
            )
            role_terminal_output = cast(
                list[dict[str, object]],
                role_terminal_response["output"],
            )
            role_terminal_output[1]["role"] = "user"
        message_role_mutations.append(role_mutation)

    for message_role_mutation in message_role_mutations:
        with pytest.raises(AssertionError):
            validate_reasoning_summary_event_group(message_role_mutation)
    assert len(message_role_mutations) == 3

    message_role_domain_mutations: list[list[dict[str, object]]] = []
    for invalid_role in ("user", "system", "developer", "tool"):
        domain_mutation = deepcopy(answer_events)
        domain_added_item = cast(dict[str, object], domain_mutation[6]["item"])
        domain_added_item["role"] = invalid_role
        domain_done_item = cast(dict[str, object], domain_mutation[11]["item"])
        domain_done_item["role"] = invalid_role
        domain_terminal_response = cast(
            dict[str, object], domain_mutation[12]["response"]
        )
        domain_terminal_output = cast(
            list[dict[str, object]], domain_terminal_response["output"]
        )
        domain_terminal_output[1]["role"] = invalid_role
        message_role_domain_mutations.append(domain_mutation)

    for message_role_domain_mutation in message_role_domain_mutations:
        with pytest.raises(AssertionError):
            validate_reasoning_summary_event_group(
                message_role_domain_mutation
            )
    assert len(message_role_domain_mutations) == 4

    def abnormal_terminal_events(
        source_events: list[dict[str, object]],
        event_type: str,
    ) -> list[dict[str, object]]:
        abnormal_events = deepcopy(source_events)
        terminal_event = abnormal_events[-1]
        terminal_event["type"] = event_type
        terminal_response = cast(dict[str, object], terminal_event["response"])
        if event_type == "response.failed":
            terminal_response["status"] = "failed"
            terminal_response["error"] = {
                "code": "server_error",
                "message": "failed after output",
            }
            terminal_response["incomplete_details"] = None
        else:
            assert event_type == "response.incomplete"
            terminal_response["status"] = "incomplete"
            terminal_response["error"] = None
            terminal_response["incomplete_details"] = {
                "reason": "max_output_tokens"
            }
        return abnormal_events

    def open_reasoning_after_part_done(
        trace_name: str,
        prefix_length: int,
    ) -> list[dict[str, object]]:
        complete_events = list(_trace_events(trace_name))
        prefix = deepcopy(complete_events[:prefix_length])
        prefix.append(deepcopy(complete_events[-1]))
        terminal_response = cast(dict[str, object], prefix[-1]["response"])
        terminal_output = cast(
            list[dict[str, object]], terminal_response["output"]
        )
        assert len(terminal_output) == 1
        terminal_output[0]["status"] = "incomplete"
        terminal_output[0].pop("encrypted_content", None)
        return prefix

    def terminal_reasoning_summary(
        events: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        terminal_response = cast(dict[str, object], events[-1]["response"])
        terminal_output = cast(
            list[dict[str, object]], terminal_response["output"]
        )
        return cast(list[dict[str, object]], terminal_output[0]["summary"])

    part_done_sources = (
        open_reasoning_after_part_done("one_part", 5),
        open_reasoning_after_part_done("sparse_indices", 9),
        open_reasoning_after_part_done("mixed_fallback", 5),
        open_reasoning_after_part_done("zero_length_fallback", 5),
    )
    part_done_positive_cases = 0
    part_done_negative_cases = 0
    for part_done_terminal_type in (
        "response.failed",
        "response.incomplete",
    ):
        valid_part_done_cases = tuple(
            abnormal_terminal_events(source, part_done_terminal_type)
            for source in part_done_sources
        )
        for valid_part_done_case in valid_part_done_cases:
            validate_reasoning_summary_event_group(valid_part_done_case)
            part_done_positive_cases += 1

        invalid_part_done_cases: list[list[dict[str, object]]] = []

        contiguous_omission = deepcopy(valid_part_done_cases[0])
        terminal_reasoning_summary(contiguous_omission).clear()
        invalid_part_done_cases.append(contiguous_omission)

        sparse_content_drift = deepcopy(valid_part_done_cases[1])
        terminal_reasoning_summary(sparse_content_drift)[2][
            "text"
        ] = "different sparse text"
        invalid_part_done_cases.append(sparse_content_drift)

        sparse_order_drift = deepcopy(valid_part_done_cases[1])
        sparse_order_summary = terminal_reasoning_summary(sparse_order_drift)
        sparse_order_summary[2], sparse_order_summary[7] = (
            sparse_order_summary[7],
            sparse_order_summary[2],
        )
        invalid_part_done_cases.append(sparse_order_drift)

        sparse_position_drift = deepcopy(valid_part_done_cases[1])
        terminal_reasoning_summary(sparse_position_drift).pop(0)
        invalid_part_done_cases.append(sparse_position_drift)

        mixed_known_part_omission = deepcopy(valid_part_done_cases[2])
        terminal_reasoning_summary(mixed_known_part_omission).pop(0)
        invalid_part_done_cases.append(mixed_known_part_omission)

        empty_part_slot_omission = deepcopy(valid_part_done_cases[3])
        terminal_reasoning_summary(empty_part_slot_omission).clear()
        invalid_part_done_cases.append(empty_part_slot_omission)

        invalid_unknown_slot = deepcopy(valid_part_done_cases[1])
        terminal_reasoning_summary(invalid_unknown_slot)[0]["type"] = "unknown"
        invalid_part_done_cases.append(invalid_unknown_slot)

        for invalid_part_done_case in invalid_part_done_cases:
            with pytest.raises(AssertionError):
                validate_reasoning_summary_event_group(invalid_part_done_case)
            part_done_negative_cases += 1

    assert part_done_positive_cases == 8
    assert part_done_negative_cases == 14

    completed_only_sources = (
        list(_trace_events("one_part")),
        list(_trace_events("sparse_indices")),
    )
    completed_and_open_source = deepcopy(tool_events[:9])
    completed_and_open_source.append(deepcopy(tool_events[-1]))
    completed_and_open_response = cast(
        dict[str, object], completed_and_open_source[-1]["response"]
    )
    completed_and_open_output = cast(
        list[dict[str, object]], completed_and_open_response["output"]
    )
    completed_and_open_output[1]["status"] = "incomplete"

    sparse_completed_and_open_source = deepcopy(completed_and_open_source)
    for sparse_event_index in (6, 7, 8):
        sparse_completed_and_open_source[sparse_event_index][
            "output_index"
        ] = 4

    abnormal_graph_positive_cases = 0
    abnormal_graph_negative_cases = 0
    for abnormal_event_type in (
        "response.failed",
        "response.incomplete",
    ):
        for completed_only_source in completed_only_sources:
            valid_completed_only = abnormal_terminal_events(
                completed_only_source,
                abnormal_event_type,
            )
            validate_reasoning_summary_event_group(valid_completed_only)
            abnormal_graph_positive_cases += 1

        valid_completed_and_open = abnormal_terminal_events(
            completed_and_open_source,
            abnormal_event_type,
        )
        validate_reasoning_summary_event_group(valid_completed_and_open)
        abnormal_graph_positive_cases += 1

        valid_sparse_completed_and_open = abnormal_terminal_events(
            sparse_completed_and_open_source,
            abnormal_event_type,
        )
        validate_reasoning_summary_event_group(valid_sparse_completed_and_open)
        abnormal_graph_positive_cases += 1

        invalid_graphs: list[list[dict[str, object]]] = []

        omitted_completed = deepcopy(valid_completed_and_open)
        omitted_response = cast(
            dict[str, object], omitted_completed[-1]["response"]
        )
        omitted_output = cast(
            list[dict[str, object]], omitted_response["output"]
        )
        omitted_output.pop(0)
        invalid_graphs.append(omitted_completed)

        duplicated_completed = deepcopy(valid_completed_and_open)
        duplicated_response = cast(
            dict[str, object], duplicated_completed[-1]["response"]
        )
        duplicated_output = cast(
            list[dict[str, object]], duplicated_response["output"]
        )
        duplicated_output.insert(0, deepcopy(duplicated_output[0]))
        invalid_graphs.append(duplicated_completed)

        reversed_output = deepcopy(valid_completed_and_open)
        reversed_response = cast(
            dict[str, object], reversed_output[-1]["response"]
        )
        reversed_items = cast(
            list[dict[str, object]], reversed_response["output"]
        )
        reversed_items.reverse()
        invalid_graphs.append(reversed_output)

        completed_content_drift = deepcopy(valid_completed_and_open)
        drift_response = cast(
            dict[str, object], completed_content_drift[-1]["response"]
        )
        drift_output = cast(list[dict[str, object]], drift_response["output"])
        drift_summary = cast(
            list[dict[str, object]], drift_output[0]["summary"]
        )
        drift_summary[0]["text"] = "different completed summary"
        invalid_graphs.append(completed_content_drift)

        open_content_drift = deepcopy(valid_completed_and_open)
        open_drift_response = cast(
            dict[str, object], open_content_drift[-1]["response"]
        )
        open_drift_output = cast(
            list[dict[str, object]], open_drift_response["output"]
        )
        open_drift_output[1]["arguments"] = '{"id":2}'
        invalid_graphs.append(open_content_drift)

        sparse_completed_omission = abnormal_terminal_events(
            completed_only_sources[1],
            abnormal_event_type,
        )
        sparse_omission_response = cast(
            dict[str, object],
            sparse_completed_omission[-1]["response"],
        )
        sparse_omission_response["output"] = []
        invalid_graphs.append(sparse_completed_omission)

        for invalid_graph in invalid_graphs:
            with pytest.raises(AssertionError):
                validate_reasoning_summary_event_group(invalid_graph)
            abnormal_graph_negative_cases += 1

    assert abnormal_graph_positive_cases == 8
    assert abnormal_graph_negative_cases == 12

    lifecycle_mutations = 0
    for name in REASONING_SUMMARY_TRACE_NAMES:
        if name in {"malformed", "cancellation"}:
            continue
        for provider_response in reasoning_summary_trace_responses(name):
            for event_group in _event_groups(provider_response):
                for event_index, raw_event in enumerate(event_group):
                    assert isinstance(raw_event, dict)
                    event_type = raw_event.get("type")
                    if event_type in {
                        "response.output_item.added",
                        "response.output_item.done",
                    }:
                        expected_status = (
                            "in_progress"
                            if event_type == "response.output_item.added"
                            else "completed"
                        )
                        for invalid_status in {
                            "in_progress",
                            "completed",
                            "incomplete",
                        } - {expected_status}:
                            mutation = deepcopy(event_group)
                            mutated_event = cast(
                                dict[str, object],
                                mutation[event_index],
                            )
                            mutated_item = cast(
                                dict[str, object],
                                mutated_event["item"],
                            )
                            mutated_item["status"] = invalid_status
                            with pytest.raises(AssertionError):
                                validate_reasoning_summary_event_group(
                                    mutation
                                )
                            lifecycle_mutations += 1

                    if event_type not in _PROVIDER_TERMINAL_EVENT_TYPES:
                        continue
                    response_payload = raw_event.get("response")
                    assert isinstance(response_payload, dict)
                    output = response_payload.get("output")
                    assert isinstance(output, list)
                    expected_status = (
                        "completed"
                        if event_type == "response.completed"
                        else "incomplete"
                    )
                    for item_index, raw_item in enumerate(output):
                        assert isinstance(raw_item, dict)
                        for invalid_status in {
                            "in_progress",
                            "completed",
                            "incomplete",
                        } - {expected_status}:
                            mutation = deepcopy(event_group)
                            terminal_event = cast(
                                dict[str, object],
                                mutation[event_index],
                            )
                            terminal_response = cast(
                                dict[str, object],
                                terminal_event["response"],
                            )
                            terminal_output = cast(
                                list[dict[str, object]],
                                terminal_response["output"],
                            )
                            terminal_output[item_index][
                                "status"
                            ] = invalid_status
                            with pytest.raises(AssertionError):
                                validate_reasoning_summary_event_group(
                                    mutation
                                )
                            lifecycle_mutations += 1

                        for field_name, invalid_value in (
                            ("id", "rs_other"),
                            (
                                "type",
                                (
                                    "message"
                                    if raw_item.get("type") != "message"
                                    else "reasoning"
                                ),
                            ),
                        ):
                            mutation = deepcopy(event_group)
                            terminal_event = cast(
                                dict[str, object],
                                mutation[event_index],
                            )
                            terminal_response = cast(
                                dict[str, object],
                                terminal_event["response"],
                            )
                            terminal_output = cast(
                                list[dict[str, object]],
                                terminal_response["output"],
                            )
                            terminal_output[item_index][
                                field_name
                            ] = invalid_value
                            with pytest.raises(AssertionError):
                                validate_reasoning_summary_event_group(
                                    mutation
                                )
                            lifecycle_mutations += 1
    assert lifecycle_mutations == 156

    content_mutations = 0
    multipart_events = list(_trace_events("multipart"))
    for mutation_kind in ("text", "order"):
        multipart_mutation = deepcopy(multipart_events)
        terminal_response = cast(
            dict[str, object],
            multipart_mutation[-1]["response"],
        )
        terminal_output = cast(
            list[dict[str, object]],
            terminal_response["output"],
        )
        terminal_summary = cast(
            list[dict[str, object]],
            terminal_output[0]["summary"],
        )
        if mutation_kind == "text":
            terminal_summary[0]["text"] = "different summary text"
        else:
            terminal_summary.reverse()
        with pytest.raises(AssertionError):
            validate_reasoning_summary_event_group(multipart_mutation)
        content_mutations += 1

    for name in ("failure_after_summary", "incomplete_after_summary"):
        abnormal_events = list(_trace_events(name))
        for summary in (
            [{"type": "summary_text", "text": "different"}],
            [
                {"type": "summary_text", "text": "first"},
                {"type": "summary_text", "text": "second"},
            ],
            [
                {"type": "summary_text", "text": "second"},
                {"type": "summary_text", "text": "first"},
            ],
        ):
            abnormal_mutation = deepcopy(abnormal_events)
            terminal_response = cast(
                dict[str, object],
                abnormal_mutation[-1]["response"],
            )
            terminal_output = cast(
                list[dict[str, object]],
                terminal_response["output"],
            )
            terminal_output[0]["summary"] = summary
            with pytest.raises(AssertionError):
                validate_reasoning_summary_event_group(abnormal_mutation)
            content_mutations += 1
    assert content_mutations == 8


def test_cancellation_fixture_stops_before_pending_provider_pull() -> None:
    consumed = reasoning_summary_events_before_cancellation("cancellation")
    response = reasoning_summary_trace_responses("cancellation")[0]
    events = response["events"]
    assert isinstance(events, list)

    assert len(consumed) == 3
    assert consumed[-1]["delta"] == "Visible before cancellation."
    assert cast(dict[str, object], events[3])["delta"] == "Must not be pulled."
    assert cast(dict[str, object], events[3]) not in consumed


def test_typed_reasoning_delta_builder_preserves_identity() -> None:
    expectation = reasoning_delta_expectation(
        "plan",
        representation="summary",
        segment_instance_ordinal=4,
        provider_item_id="rs_1",
        output_index=2,
        summary_index=3,
        continuation_id="continuation-1",
        provider_event_type="response.reasoning_summary_text.delta",
    )

    assert expectation.representation == "summary"
    assert expectation.segment_instance_ordinal == 4
    assert expectation.text == "plan"
    assert expectation.provider_item_id == "rs_1"
    assert expectation.output_index == 2
    assert expectation.summary_index == 3
    assert expectation.continuation_id == "continuation-1"


def test_typed_reasoning_delta_builder_rejects_invalid_identity() -> None:
    try:
        reasoning_delta_expectation(
            "plan",
            representation="summary",
            segment_instance_ordinal=0,
            summary_index=cast(Any, True),
        )
    except AssertionError as exc:
        assert "summary_index" in str(exc)
    else:
        raise AssertionError("boolean summary index must be rejected")

    with pytest.raises(AssertionError):
        ReasoningPartIdentity(
            representation="native_text",
            segment_instance_ordinal=cast(Any, True),
        )
    for invalid_ordinal in (None, True, "0", -1):
        with pytest.raises(AssertionError):
            reasoning_delta_expectation(
                "plan",
                representation="native_text",
                segment_instance_ordinal=cast(Any, invalid_ordinal),
            )
        with pytest.raises(AssertionError):
            ReasoningPartIdentity(
                representation="native_text",
                segment_instance_ordinal=cast(Any, invalid_ordinal),
            )


def test_phase0_retention_contract_reconciles_existing_policy_defaults() -> (
    None
):
    payload = _contract()
    retention = payload["retention"]
    assert isinstance(retention, dict)
    policy = StreamRetentionPolicy()

    assert retention["canonical_item_limit"] == policy.accumulator_item_limit
    assert retention["replay_reasoning_item_limit"] == 1024
    assert retention["replay_reasoning_summary_node_limit"] == 4096
    assert retention["replay_reasoning_summary_character_limit"] == 262144
    assert (
        retention["replay_reasoning_summary_serialized_byte_limit"] == 1048576
    )
    extensions = payload["stream_retention_policy_extensions"]
    assert isinstance(extensions, dict)
    assert extensions["reasoning_segment_limit"] == 1024
    assert extensions["reasoning_character_limit"] == 262144
    assert extensions["reasoning_text_byte_limit"] == 1048576
    assert extensions["openai_replay_reasoning_item_limit"] == 1024
    assert extensions["openai_replay_reasoning_summary_node_limit"] == 4096
    assert (
        extensions["openai_replay_reasoning_summary_character_limit"] == 262144
    )
    assert (
        extensions["openai_replay_reasoning_summary_serialized_byte_limit"]
        == 1048576
    )
    assert extensions["responses_reasoning_item_segment_limit"] == 1024
    assert extensions["mcp_reasoning_segment_limit"] == 512
    assert extensions["mcp_reasoning_character_limit"] == 1048576
    assert extensions["mcp_reasoning_text_byte_limit"] == 1048576
    assert extensions["a2a_reasoning_artifact_segment_limit"] == 512
    assert extensions["a2a_reasoning_artifact_character_limit"] == 1048576
    assert extensions["a2a_reasoning_artifact_text_byte_limit"] == 1048576
    assert "recording_reasoning_character_limit" not in extensions


def test_phase0_retention_contract_locks_owners_and_overflow() -> None:
    contract = _contract()
    _assert_integer_type_mutations_fail(
        contract,
        _CONTRACT_INTEGER_CATALOG,
    )
    contract_mapping_mutations = 0
    for _, path, _ in _json_mapping_entries(contract):
        unknown_contract = deepcopy(contract)
        _add_unknown_mapping_key(unknown_contract, path)
        with pytest.raises(AssertionError):
            _assert_phase0_contract_payload(unknown_contract)
        contract_mapping_mutations += 1
    assert contract_mapping_mutations == 54
    pointer_fixture: dict[str, object] = {
        "": {},
        "/": {},
        "~": {},
        "~1": {},
        "a/b": {},
        "a~1b": {},
    }
    pointer_entries = _json_mapping_entries(pointer_fixture)
    pointers = {pointer for pointer, _, _ in pointer_entries}
    assert canonical_json_pointer(()) == ""
    assert pointers == {
        "",
        "/",
        "/~1",
        "/~0",
        "/~01",
        "/a~1b",
        "/a~01b",
    }
    assert len(pointer_entries) == len(pointers)
    assert _json_integer_entries(7) == (("", (), 7),)
    assert _json_float_entries(7.0) == (("", (), 7.0),)
    scalar_fixture = {
        "": 1,
        "~": 2,
        "/": 3,
        "items": [4],
        "mapping": {"0": 5},
    }
    scalar_entries = _json_integer_entries(scalar_fixture)
    assert {pointer for pointer, _, _ in scalar_entries} == {
        "/",
        "/~0",
        "/~1",
        "/items/0",
        "/mapping/0",
    }
    assert canonical_json_pointer(("items", 0)) == "/items/0"
    assert canonical_json_pointer(("items", "0")) == "/items/0"
    assert typed_json_path(("items", 0)) != typed_json_path(("items", "0"))
    for duplicate_source in _DUPLICATE_JSON_OBJECT_CASES:
        with pytest.raises(ValueError, match="duplicate JSON object name"):
            strict_json_loads(duplicate_source)

    contract_source = (
        reasoning_summary_fixture_root() / "phase0_contract.json"
    ).read_text(encoding="utf-8")
    duplicate_contract_sources = (
        _duplicate_json_name(
            contract_source,
            name="schema_version",
            serialized_value="1",
        ),
        _duplicate_json_name(
            contract_source,
            name="warmups",
            serialized_value="3",
        ),
    )
    for duplicate_source in duplicate_contract_sources:
        with pytest.raises(ValueError, match="duplicate JSON object name"):
            strict_json_loads(duplicate_source)
    for constant in _NONFINITE_JSON_CONSTANTS:
        for container_marker in (None, '"benchmark":'):
            nonfinite_source = _inject_nonfinite_json_number(
                contract_source,
                constant=constant,
                container_marker=container_marker,
            )
            with pytest.raises(ValueError, match="non-finite JSON number"):
                strict_json_loads(nonfinite_source)
    semantics = contract["retention_semantics"]
    assert isinstance(semantics, dict)

    sdk = semantics["sdk_reasoning"]
    replay = semantics["openai_replay_reasoning"]
    separator = semantics["derived_paragraph_separator"]
    responses = semantics["responses_open_item"]
    redaction_pending = semantics["redaction_pending"]
    mcp = semantics["mcp_reasoning"]
    a2a = semantics["a2a_reasoning"]
    generic = semantics["generic_persistence"]
    assert isinstance(sdk, dict)
    assert isinstance(replay, dict)
    assert isinstance(separator, dict)
    assert isinstance(responses, dict)
    assert isinstance(redaction_pending, dict)
    assert isinstance(mcp, dict)
    assert isinstance(a2a, dict)
    assert isinstance(generic, dict)
    assert "same retained segment text" in cast(str, sdk["flat_view"])
    assert replay["failure_code"] == "reasoning_replay_retention_exceeded"
    assert "whole item before mutation" in cast(str, replay["overflow"])
    assert replay["outward_projection"] == "none"
    assert "compact UTF-8 JSON" in cast(str, replay["structure_accounting"])
    assert (
        replay["serialized_byte_accounting"]
        == 'len(json.dumps(value, ensure_ascii=False, separators=(",",":"), '
        'sort_keys=True, allow_nan=False).encode("utf-8"))'
    )
    serialized_unicode = dumps(
        {"résumé": ["雪"]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    assert serialized_unicode == '{"résumé":["雪"]}'
    assert len(serialized_unicode.encode("utf-8")) > len(serialized_unicode)
    with pytest.raises(ValueError):
        dumps(
            {"invalid": float("nan")},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    assert replay["allowed_scalars"] == [
        "string",
        "finite_integer",
        "finite_float",
        "boolean",
        "null",
    ]
    assert "non_finite_float" in replay["invalid_values"]
    assert separator == {
        "owners": [
            "sdk_reasoning",
            "cli_visible_reasoning",
            "mcp_reasoning",
        ],
        "text": (
            "minimum line feeds needed for two across adjacent non-empty "
            "segment boundaries"
        ),
        "admission": (
            "count derived line-feed characters and UTF-8 bytes before the "
            "following segment"
        ),
        "truncation": (
            "include in dropped_characters and dropped_utf8_bytes but not "
            "dropped_segments"
        ),
        "raw_canonical_stream": False,
    }
    assert "never emit a truncated successful item" in cast(
        str, responses["overflow"]
    )
    assert "including empty" in cast(str, responses["overflow"])
    assert responses["output_item_status"] == "incomplete"
    assert responses["failure_code"] == "reasoning_summary_retention_exceeded"
    assert responses["admission_points"] == [
        "before part.added",
        "before each raw delta",
        "before completed-item fallback parts",
        "before redaction marker",
    ]
    assert responses["non_stream_overflow_http_status"] == 500
    assert responses["non_stream_overflow_body"] == {
        "error": {
            "type": "server_error",
            "code": "reasoning_summary_retention_exceeded",
            "message": (
                "Reasoning summary exceeded the configured retention limit."
            ),
        }
    }
    assert redaction_pending["literal_overflow_flush"] is False
    assert redaction_pending["unbounded_host_path_tail"] is False
    assert redaction_pending["marker_reservation_text"] == (
        SKILL_CONTENT_REDACTION
    )
    assert (
        redaction_pending["marker_reservation_character_expression"]
        == "len(SKILL_CONTENT_REDACTION)"
    )
    assert (
        redaction_pending["marker_reservation_byte_expression"]
        == 'len(SKILL_CONTENT_REDACTION.encode("utf-8"))'
    )
    assert redaction_pending["marker_reservation_characters"] == len(
        SKILL_CONTENT_REDACTION
    )
    assert redaction_pending["marker_reservation_utf8_bytes"] == len(
        SKILL_CONTENT_REDACTION.encode("utf-8")
    )
    assert "before the first raw delta" in cast(
        str, redaction_pending["marker_reservation_timing"]
    )
    assert redaction_pending["required_identity_fields"] == [
        "representation",
        "segment_instance_ordinal",
    ]
    assert redaction_pending["optional_correlation_fields"] == [
        "provider_item_id",
        "output_index",
        "summary_index",
        "continuation_id",
    ]
    assert redaction_pending["boundaries"] == [
        "representation_change",
        "segment_completion",
        "correlation_change",
        "identity_loss",
    ]
    assert "every later identity and any completion flush" in cast(
        str, redaction_pending["confirmed_redaction_latch"]
    )
    assert "preserving structural close" in cast(
        str, redaction_pending["confirmed_redaction_latch"]
    )
    assert "fresh protocol stream owner" in cast(
        str, redaction_pending["confirmed_redaction_latch_reset"]
    )
    assert "exactly the next known segment" in cast(
        str, redaction_pending["identity_loss_without_marker"]
    )
    assert "confirmed-redaction latch" in cast(
        str, redaction_pending["identity_loss_with_marker"]
    )
    assert "confirmed-redaction latch" in cast(
        str, redaction_pending["completion_with_pending"]
    )
    assert "always emits the old marker" in cast(
        str, redaction_pending["marker_admission"]
    )
    assert "no pending marker is created" in cast(
        str, redaction_pending["mcp_overflow"]
    )
    assert "no pending marker is created" in cast(
        str, redaction_pending["a2a_overflow"]
    )
    assert "all later deltas for the identity" in cast(str, a2a["overflow"])
    assert "charge paragraph separators" in cast(str, mcp["flat_view"])
    assert "empty string when all reasoning was dropped" in cast(
        str, mcp["flat_view"]
    )
    assert "one adjacent structuredContent.reasoningTruncation" in cast(
        str, mcp["truncation_owner"]
    )
    assert "never repeat collection-wide counters on segments" in cast(
        str, mcp["truncation_owner"]
    )
    assert "never synthesize a metadata segment" in cast(
        str, mcp["truncation_owner"]
    )
    assert "dropped or rejected" in cast(str, mcp["truncation_presence"])
    assert a2a["live_event_retraction"] is False
    assert "reasoning content is prohibited" in cast(str, generic["overflow"])

    activation = cast(dict[str, object], contract["activation"])
    assert activation == {
        "phases_0_through_8_public_adapters": (
            "all production and publicly reachable adapters reject summary "
            "requests"
        ),
        "phases_0_through_8_capable_adapters": (
            "explicit private test-only adapters"
        ),
        "phase_9": (
            "official OpenAI and Azure OpenAI activate atomically; compatible "
            "production adapters may then opt in through explicit typed "
            "capability"
        ),
    }
    lifecycle = cast(dict[str, object], contract["lifecycle"])
    assert lifecycle == {
        "provider_terminal_reasoning_done": (
            "exactly once after non-empty reasoning"
        ),
        "consumer_cancel_reasoning_done": "none",
        "consumer_cancel_owned_terminal": "exactly one STREAM_CANCELLED",
        "consumer_aclose_or_generator_exit_yield": "none",
    }
    canonical_identity = cast(
        dict[str, object], contract["canonical_reasoning_identity"]
    )
    assert canonical_identity == {
        "required_fields": ["representation", "segment_instance_ordinal"],
        "optional_correlation_fields": [
            "provider_item_id",
            "output_index",
            "summary_index",
            "continuation_id",
        ],
        "segment_instance_ordinal": (
            "zero-based per normalized provider response and prefixed by "
            "continuation in aggregate state"
        ),
        "native_missing_correlations": "valid including summary_index null",
        "ordinal_allowed_kinds": ["REASONING_DELTA"],
        "ordinal_invalid_values": [
            "null",
            "boolean",
            "string",
            "float",
            "object",
            "negative_integer",
        ],
        "boundaries": [
            "representation_change",
            "segment_completion",
            "correlation_change",
            "identity_loss",
        ],
    }
    abnormal = cast(dict[str, object], contract["abnormal_terminal_shapes"])
    assert abnormal["responses_item_status"] == "incomplete"
    assert abnormal["responses_failed_order"] == [
        "reasoning_summary_text.done",
        "reasoning_summary_part.done",
        "output_item.done",
        "response.failed",
    ]
    assert abnormal["responses_cancelled_order"] == [
        "reasoning_summary_text.done",
        "reasoning_summary_part.done",
        "output_item.done",
        "response.cancelled",
    ]
    assert abnormal["provider_response_incomplete"] == {
        "canonical_terminal": "STREAM_ERRORED",
        "error_code": "response_incomplete",
        "outward_terminal": "response.failed",
    }
    assert abnormal["response_cancelled"] == {
        "surface": "Avalan outward Responses extension",
        "provider_openai_sdk_event": False,
        "canonical_terminal": "STREAM_CANCELLED",
    }
    assert abnormal["mcp_segment_metadata"] == {
        "completed": False,
        "status": "incomplete",
        "terminal_outcome": ["failed", "cancelled"],
    }
    assert abnormal["a2a_artifact_metadata"] == {
        "status": "incomplete",
        "terminal_outcome": ["failed", "cancelled"],
    }
    assert abnormal["terminal_count"] == 1
    assert abnormal["semantic_output_after_terminal"] is False
    assert abnormal["local_consumer_aclose_output"] == "none"
    protocol_shapes = cast(dict[str, object], contract["protocol_shapes"])
    assert (
        protocol_shapes["responses_sse_sequence"]
        == "one response-local zero-based contiguous allocator with no gaps "
        "or reuse across all events and continuations"
    )
    assert protocol_shapes["responses_sse_sequence_start"] == 0
    assert (
        protocol_shapes["responses_output_index"]
        == "one zero-based contiguous index exactly once per outward"
        " reasoning, "
        "tool, or message item across continuations"
    )
    assert protocol_shapes["responses_output_index_start"] == 0
    assert (
        protocol_shapes["responses_reasoning_item_id"]
        == "rs_{outer_response_id}_{outward_output_index}"
    )
    sparse_mapping = protocol_shapes["responses_sparse_summary_index_example"]
    assert isinstance(sparse_mapping, dict)
    assert sparse_mapping == {
        "completed_provider_array_length": 8,
        "streamed_provider_summary_indices": [2, 7],
        "fallback_provider_summary_indices": [0, 1, 3, 4, 5, 6],
        "canonical_provider_emission_order": [2, 7, 0, 1, 3, 4, 5, 6],
        "outward_summary_indices": list(range(8)),
    }
    assert protocol_shapes["completed_item_fallback"] == {
        "non_empty_streamed_delta_disables_part_fallback": True,
        "zero_length_delta_is_visible": False,
        "zero_length_delta_disables_part_fallback": False,
    }
    assert [
        event["summary_index"]
        for event in _trace_events("sparse_indices")
        if event.get("type") == "response.reasoning_summary_part.added"
    ] == sparse_mapping["streamed_provider_summary_indices"]
    assert protocol_shapes["responses_non_stream_required_fields"] == [
        "id",
        "type",
        "status",
        "summary",
    ]
    assert protocol_shapes["mcp_reasoning_segment_required_fields"] == [
        "representation",
        "segment_instance_ordinal",
        "text",
        "completed",
        "status",
        "terminal_outcome",
    ]
    assert protocol_shapes["mcp_reasoning_segment_optional_correlations"] == [
        "provider_item_id",
        "output_index",
        "summary_index",
        "continuation_id",
    ]
    assert protocol_shapes["mcp_reasoning_segment_forbidden_fields"] == [
        "completion",
        "truncation",
        "truncated",
        "dropped_segments",
        "dropped_characters",
        "dropped_utf8_bytes",
        "leading_segment_partial",
    ]
    assert protocol_shapes["mcp_reasoning_segment_field_types"] == {
        "representation": ["native_text", "summary"],
        "segment_instance_ordinal": "non_negative_integer_not_boolean",
        "text": "string",
        "completed": "boolean",
        "status": ["in_progress", "completed", "incomplete"],
        "terminal_outcome": [None, "completed", "failed", "cancelled"],
    }
    assert protocol_shapes["mcp_reasoning_truncation_required_fields"] == [
        "truncated",
        "dropped_segments",
        "dropped_characters",
        "dropped_utf8_bytes",
        "leading_segment_partial",
    ]
    assert protocol_shapes["mcp_reasoning_truncation_field_types"] == {
        "truncated": "boolean",
        "dropped_segments": "non_negative_integer_not_boolean",
        "dropped_characters": "non_negative_integer_not_boolean",
        "dropped_utf8_bytes": "non_negative_integer_not_boolean",
        "leading_segment_partial": "boolean",
    }
    assert (
        protocol_shapes["mcp_final_reasoning_container"] == "structuredContent"
    )
    assert protocol_shapes["mcp_final_reasoning_fields"] == [
        "reasoning",
        "reasoningSegments",
        "reasoningTruncation",
    ]
    assert "include all three fields" in cast(
        str, protocol_shapes["mcp_final_reasoning_presence"]
    )
    assert protocol_shapes["mcp_final_reasoning_presence_matrix"] == {
        "retained_nonempty_no_drop": True,
        "retained_nonempty_with_drop": True,
        "no_retained_with_drop_or_rejection": True,
        "no_retained_no_drop": False,
    }
    assert "empty string" in cast(
        str, protocol_shapes["mcp_flat_reasoning_value"]
    )
    assert protocol_shapes["mcp_reasoning_segment_state_matrix"] == {
        "open": {
            "completed": False,
            "status": "in_progress",
            "terminal_outcome": None,
        },
        "completed": {
            "completed": True,
            "status": "completed",
            "terminal_outcome": "completed",
        },
        "failed": {
            "completed": False,
            "status": "incomplete",
            "terminal_outcome": "failed",
        },
        "cancelled": {
            "completed": False,
            "status": "incomplete",
            "terminal_outcome": "cancelled",
        },
    }
    retention_examples = cast(
        dict[str, object],
        protocol_shapes["mcp_reasoning_retention_examples"],
    )
    assert set(retention_examples) == {
        "no_overflow",
        "oldest_segment_and_separator_drop",
        "leading_partial",
        "no_retained_oversize_first_candidate",
    }
    final_reasoning_fields = set(
        cast(list[str], protocol_shapes["mcp_final_reasoning_fields"])
    )
    required_segment_fields = set(
        cast(
            list[str],
            protocol_shapes["mcp_reasoning_segment_required_fields"],
        )
    )
    optional_segment_fields = set(
        cast(
            list[str],
            protocol_shapes["mcp_reasoning_segment_optional_correlations"],
        )
    )
    forbidden_segment_fields = set(
        cast(
            list[str],
            protocol_shapes["mcp_reasoning_segment_forbidden_fields"],
        )
    )
    truncation_fields = set(
        cast(
            list[str],
            protocol_shapes["mcp_reasoning_truncation_required_fields"],
        )
    )
    for raw_example in retention_examples.values():
        example = cast(dict[str, object], raw_example)
        payload = cast(dict[str, object], example["payload"])
        assert set(payload) == final_reasoning_fields
        segments = cast(list[dict[str, object]], payload["reasoningSegments"])
        truncation = cast(dict[str, object], payload["reasoningTruncation"])
        assert set(truncation) == truncation_fields
        validate_mcp_reasoning_truncation(truncation)
        for segment in segments:
            assert required_segment_fields <= set(segment)
            assert set(segment) <= (
                required_segment_fields | optional_segment_fields
            )
            assert forbidden_segment_fields.isdisjoint(segment)
        assert payload["reasoning"] == "\n\n".join(
            cast(str, segment["text"]) for segment in segments
        )

    no_overflow = cast(dict[str, object], retention_examples["no_overflow"])
    no_overflow_payload = cast(dict[str, object], no_overflow["payload"])
    baseline_truncation = deepcopy(
        cast(
            dict[str, object],
            no_overflow_payload["reasoningTruncation"],
        )
    )
    for flag_name in ("truncated", "leading_segment_partial"):
        for invalid_flag in (0, 1):
            invalid_truncation = deepcopy(baseline_truncation)
            invalid_truncation[flag_name] = invalid_flag
            with pytest.raises(AssertionError):
                validate_mcp_reasoning_truncation(invalid_truncation)
    for count_name in (
        "dropped_segments",
        "dropped_characters",
        "dropped_utf8_bytes",
    ):
        for invalid_count in (False, True):
            invalid_truncation = deepcopy(baseline_truncation)
            invalid_truncation[count_name] = invalid_count
            with pytest.raises(AssertionError):
                validate_mcp_reasoning_truncation(invalid_truncation)
        negative_truncation = deepcopy(baseline_truncation)
        negative_truncation[count_name] = -1
        with pytest.raises(AssertionError):
            validate_mcp_reasoning_truncation(negative_truncation)

    assert no_overflow_payload["reasoningTruncation"] == {
        "truncated": False,
        "dropped_segments": 0,
        "dropped_characters": 0,
        "dropped_utf8_bytes": 0,
        "leading_segment_partial": False,
    }

    oldest_drop = cast(
        dict[str, object],
        retention_examples["oldest_segment_and_separator_drop"],
    )
    oldest_source = cast(list[str], oldest_drop["source_segments"])
    oldest_separator = cast(str, oldest_drop["derived_separator"])
    oldest_payload = cast(dict[str, object], oldest_drop["payload"])
    oldest_truncation = cast(
        dict[str, object], oldest_payload["reasoningTruncation"]
    )
    oldest_dropped_text = oldest_source[0] + oldest_separator
    assert oldest_payload["reasoning"] == oldest_source[-1]
    assert oldest_truncation == {
        "truncated": True,
        "dropped_segments": 1,
        "dropped_characters": len(oldest_dropped_text),
        "dropped_utf8_bytes": len(oldest_dropped_text.encode("utf-8")),
        "leading_segment_partial": False,
    }

    leading_partial = cast(
        dict[str, object], retention_examples["leading_partial"]
    )
    leading_source = cast(list[str], leading_partial["source_segments"])[0]
    leading_payload = cast(dict[str, object], leading_partial["payload"])
    leading_text = cast(str, leading_payload["reasoning"])
    leading_truncation = cast(
        dict[str, object], leading_payload["reasoningTruncation"]
    )
    assert leading_source.endswith(leading_text)
    leading_dropped = leading_source[: -len(leading_text)]
    assert leading_truncation == {
        "truncated": True,
        "dropped_segments": 0,
        "dropped_characters": len(leading_dropped),
        "dropped_utf8_bytes": len(leading_dropped.encode("utf-8")),
        "leading_segment_partial": True,
    }

    rejected = cast(
        dict[str, object],
        retention_examples["no_retained_oversize_first_candidate"],
    )
    rejected_text = cast(list[str], rejected["source_segments"])[0]
    rejected_payload = cast(dict[str, object], rejected["payload"])
    assert rejected["prospectively_rejected"] is True
    assert rejected_payload["reasoning"] == ""
    assert rejected_payload["reasoningSegments"] == []
    assert rejected_payload["reasoningTruncation"] == {
        "truncated": True,
        "dropped_segments": 1,
        "dropped_characters": len(rejected_text),
        "dropped_utf8_bytes": len(rejected_text.encode("utf-8")),
        "leading_segment_partial": False,
    }
    assert "confirmed-redaction latch" in cast(
        str, protocol_shapes["redaction_identity_change_with_pending"]
    )
    assert "exactly the next known segment" in cast(
        str, protocol_shapes["redaction_identity_loss_without_marker"]
    )
    assert protocol_shapes["responses_overflow_delta"] == "suppressed"
    assert (
        protocol_shapes["responses_overflow_output_item_status"]
        == "incomplete"
    )
    assert (
        protocol_shapes["responses_overflow_failure_code"]
        == "reasoning_summary_retention_exceeded"
    )
    assert (
        protocol_shapes["a2a_reasoning_artifact_id"]
        == "reasoning-{run_id}-{continuation_ordinal}-{segment_ordinal}"
    )
    a2a_template = protocol_shapes["a2a_reasoning_artifact_id"]
    a2a_ids = [
        a2a_template.format(
            run_id="run",
            continuation_ordinal=continuation_ordinal,
            segment_ordinal=segment_ordinal,
        )
        for continuation_ordinal, segment_ordinal in ((0, 0), (1, 0), (1, 1))
    ]
    assert len(set(a2a_ids)) == 3
    assert a2a_ids[1] == a2a_template.format(
        run_id="run",
        continuation_ordinal=1,
        segment_ordinal=0,
    )

    settings = ServerOutputRedactionSettings(
        enabled=True,
        rules=frozenset({"skill_body_echoes"}),
    )
    old_identity = ReasoningPartIdentity(
        representation="summary",
        segment_instance_ordinal=0,
        provider_item_id="rs-redaction",
        output_index=0,
        summary_index=0,
        continuation_id="continuation-1",
    )
    new_identity = ReasoningPartIdentity(
        representation="summary",
        segment_instance_ordinal=1,
        provider_item_id="rs-redaction",
        output_index=0,
        summary_index=1,
        continuation_id="continuation-1",
    )
    for protocol in ("openai", "mcp", "a2a"):
        redactor = IdentityTaggedReasoningRedactor(
            settings,
            protocol=cast(Any, protocol),
        )
        assert redactor.push(old_identity, "# Demo Skill\n\n") == ()
        emitted = redactor.push(
            new_identity,
            "Use when private.\nCROSS_PART_SECRET",
        )
        assert [(item.identity, item.text) for item in emitted] == [
            (old_identity, SKILL_CONTENT_REDACTION),
        ]
        assert all("CROSS_PART_SECRET" not in item.text for item in emitted)
        assert redactor.push(new_identity, "still secret") == ()
        assert redactor.complete(new_identity) == ()
        direct_second_later_identity = ReasoningPartIdentity(
            representation="summary",
            segment_instance_ordinal=2,
            provider_item_id="rs-redaction-next",
            output_index=1,
            summary_index=0,
            continuation_id="continuation-2",
        )
        assert (
            redactor.push(
                direct_second_later_identity,
                "SECOND_LATER_IDENTITY_SECRET",
            )
            == ()
        )
        assert redactor.complete(direct_second_later_identity) == ()

        native_identity = ReasoningPartIdentity(
            representation="native_text",
            segment_instance_ordinal=0,
        )
        native_redactor = IdentityTaggedReasoningRedactor(
            settings,
            protocol=cast(Any, protocol),
        )
        assert native_redactor.push(native_identity, "# Demo Skill\n\n") == ()
        assert native_redactor.complete(native_identity) == (
            TaggedRedactedText(
                identity=native_identity,
                text=SKILL_CONTENT_REDACTION,
            ),
        )
        completion_later_identities = (
            ReasoningPartIdentity(
                representation="native_text",
                segment_instance_ordinal=1,
            ),
            ReasoningPartIdentity(
                representation="native_text",
                segment_instance_ordinal=2,
            ),
        )
        for later_identity in completion_later_identities:
            assert (
                native_redactor.push(
                    later_identity,
                    "COMPLETION_SPLIT_SECRET",
                )
                == ()
            )
            assert native_redactor.complete(later_identity) == ()

        fresh_redactor = IdentityTaggedReasoningRedactor(
            settings,
            protocol=cast(Any, protocol),
        )
        fresh_identity = ReasoningPartIdentity(
            representation="native_text",
            segment_instance_ordinal=0,
        )
        assert fresh_redactor.push(fresh_identity, "fresh stream") == (
            TaggedRedactedText(identity=fresh_identity, text="fresh stream"),
        )

        summary_only_identity = ReasoningPartIdentity(
            representation="summary",
            segment_instance_ordinal=0,
        )
        native_only_identity = ReasoningPartIdentity(
            representation="native_text",
            segment_instance_ordinal=1,
        )
        representation_redactor = IdentityTaggedReasoningRedactor(
            settings,
            protocol=cast(Any, protocol),
        )
        assert (
            representation_redactor.push(
                summary_only_identity, "# Demo Skill\n\n"
            )
            == ()
        )
        representation_boundary = representation_redactor.push(
            native_only_identity,
            "REPRESENTATION_BOUNDARY_SECRET",
        )
        assert [
            (item.identity, item.text) for item in representation_boundary
        ] == [(summary_only_identity, SKILL_CONTENT_REDACTION)]
        assert representation_redactor.complete(native_only_identity) == ()

        lost_with_marker_redactor = IdentityTaggedReasoningRedactor(
            settings,
            protocol=cast(Any, protocol),
        )
        assert (
            lost_with_marker_redactor.push(old_identity, "# Demo Skill\n\n")
            == ()
        )
        lost_boundary = lost_with_marker_redactor.push(
            None, "UNIDENTIFIED_SECRET"
        )
        assert [(item.identity, item.text) for item in lost_boundary] == [
            (old_identity, SKILL_CONTENT_REDACTION)
        ]
        assert (
            lost_with_marker_redactor.push(
                new_identity,
                "LATCHED_AFTER_IDENTITY_LOSS",
            )
            == ()
        )

        lost_without_marker_redactor = IdentityTaggedReasoningRedactor(
            settings,
            protocol=cast(Any, protocol),
        )
        assert lost_without_marker_redactor.push(
            old_identity, "ordinary before identity loss"
        ) == (
            TaggedRedactedText(
                identity=old_identity,
                text="ordinary before identity loss",
            ),
        )
        assert (
            lost_without_marker_redactor.push(None, "UNIDENTIFIED_SECRET")
            == ()
        )
        quarantined_identity = ReasoningPartIdentity(
            representation="native_text",
            segment_instance_ordinal=2,
        )
        assert (
            lost_without_marker_redactor.push(
                quarantined_identity, "QUARANTINED_SECRET"
            )
            == ()
        )
        assert (
            lost_without_marker_redactor.complete(quarantined_identity) == ()
        )
        following_identity = ReasoningPartIdentity(
            representation="native_text",
            segment_instance_ordinal=3,
        )
        following = lost_without_marker_redactor.push(
            following_identity, "ordinary"
        )
        assert [(item.identity, item.text) for item in following] == [
            (following_identity, "ordinary")
        ]


def test_benchmark_protocol_is_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = _BENCHMARK_SCRIPT.benchmark_protocol()
    contract_payload = _contract()
    contract_source = (
        reasoning_summary_fixture_root() / "phase0_contract.json"
    ).read_text(encoding="utf-8")
    duplicate_contract_sources = (
        _duplicate_json_name(
            contract_source,
            name="schema_version",
            serialized_value="1",
        ),
        _duplicate_json_name(
            contract_source,
            name="warmups",
            serialized_value="3",
        ),
    )
    original_contract_path = _BENCHMARK_SCRIPT.contract_path
    for index, duplicate_source in enumerate(duplicate_contract_sources):
        duplicate_path = tmp_path / f"duplicate-contract-{index}.json"
        duplicate_path.write_text(duplicate_source, encoding="utf-8")
        monkeypatch.setattr(
            _BENCHMARK_SCRIPT,
            "contract_path",
            lambda duplicate_path=duplicate_path: duplicate_path,
        )
        with pytest.raises(ValueError, match="duplicate JSON object name"):
            _BENCHMARK_SCRIPT.benchmark_protocol()
    monkeypatch.setattr(
        _BENCHMARK_SCRIPT,
        "contract_path",
        original_contract_path,
    )
    for index, (constant, container_marker) in enumerate(
        (
            (constant, container_marker)
            for constant in _NONFINITE_JSON_CONSTANTS
            for container_marker in (None, '"benchmark":')
        )
    ):
        nonfinite_path = tmp_path / f"nonfinite-contract-{index}.json"
        nonfinite_path.write_text(
            _inject_nonfinite_json_number(
                contract_source,
                constant=constant,
                container_marker=container_marker,
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _BENCHMARK_SCRIPT,
            "contract_path",
            lambda nonfinite_path=nonfinite_path: nonfinite_path,
        )
        with pytest.raises(ValueError, match="non-finite JSON number"):
            _BENCHMARK_SCRIPT.benchmark_protocol()
    monkeypatch.setattr(
        _BENCHMARK_SCRIPT,
        "contract_path",
        original_contract_path,
    )
    benchmark_integer_paths = (
        ("schema_version",),
        ("benchmark", "warmups"),
        ("benchmark", "samples"),
        ("benchmark", "delta_counts", 0),
        ("benchmark", "delta_counts", 1),
        ("benchmark", "memory_subprocess_timeout_seconds"),
    )
    contract_integers = {
        path: value
        for _, path, value in _json_integer_entries(contract_payload)
    }
    invalid_value: object
    for path in benchmark_integer_paths:
        integer_value = contract_integers[path]
        for invalid_value in (False, True, float(integer_value)):
            mutation = deepcopy(contract_payload)
            _set_json_path(mutation, path, invalid_value)
            with pytest.raises(AssertionError):
                _BENCHMARK_SCRIPT._benchmark_protocol_from_payload(mutation)

    raw_benchmark = contract_payload["benchmark"]
    assert isinstance(raw_benchmark, dict)
    for bool_field in (
        "gc_before_each_sample",
        "reset_state_before_each_sample",
        "gc_before_memory_probe",
        "network_allowed",
    ):
        for invalid_value in (0, 1, 0.0, 1.0, "true", None):
            mutation = deepcopy(contract_payload)
            mutated_benchmark = mutation["benchmark"]
            assert isinstance(mutated_benchmark, dict)
            mutated_benchmark[bool_field] = invalid_value
            with pytest.raises(AssertionError):
                _BENCHMARK_SCRIPT._benchmark_protocol_from_payload(mutation)

    unknown_benchmark_field = deepcopy(contract_payload)
    mutated_benchmark = unknown_benchmark_field["benchmark"]
    assert isinstance(mutated_benchmark, dict)
    mutated_benchmark["unexpected"] = None
    with pytest.raises(AssertionError):
        _BENCHMARK_SCRIPT._benchmark_protocol_from_payload(
            unknown_benchmark_field
        )

    assert protocol.warmups == 3
    assert protocol.samples == 20
    assert protocol.delta_counts == (4096, 8192)
    assert protocol.delta_text == "x"
    assert protocol.percentile_method == "nearest_rank"
    assert protocol.gc_before_each_sample
    assert protocol.reset_state_before_each_sample
    assert protocol.gc_before_memory_probe
    assert protocol.elapsed_clock == "perf_counter"
    assert protocol.peak_memory_probe == "tracemalloc"
    assert protocol.memory_probe_scopes == (
        "processing_excluding_source_fixture",
        "retained_total_including_source_fixture",
    )
    assert protocol.memory_subprocess_timeout_seconds == 60
    assert not protocol.network_allowed
    assert protocol.network_guard == "deny_socket_creation_and_resolution"

    deterministic = _BENCHMARK_SCRIPT.expected_deterministic_counts(1)
    deterministic_fields = asdict(deterministic)
    assert len(deterministic_fields) == 14
    deterministic_mutations = 0
    for field_name, field_value in deterministic_fields.items():
        assert type(field_value) is int
        for invalid_value in (False, True, float(field_value)):
            mutation = dict(deterministic_fields)
            mutation[field_name] = invalid_value
            with pytest.raises(AssertionError):
                _BENCHMARK_SCRIPT.DeterministicCounts(**mutation)
            deterministic_mutations += 1
    assert deterministic_mutations == 42

    valid_workload: dict[str, Any] = {
        "delta_count": 1,
        "item_count": 5,
        "sample_microseconds": (1.0,),
        "median_microseconds": 1.0,
        "p95_microseconds": 1.0,
        "median_per_item_microseconds": 0.2,
        "p95_per_item_microseconds": 0.2,
        "peak_processing_bytes_excluding_source_fixture": 1,
        "current_retained_bytes_including_source_fixture": 1,
        "peak_total_bytes_including_source_fixture": 1,
        "deterministic": deterministic,
    }
    _BENCHMARK_SCRIPT.WorkloadResult(**valid_workload)
    workload_integer_mutations = 0
    for field_name in (
        "delta_count",
        "item_count",
        "peak_processing_bytes_excluding_source_fixture",
        "current_retained_bytes_including_source_fixture",
        "peak_total_bytes_including_source_fixture",
    ):
        for invalid_value in (False, True, 1.0):
            mutation = dict(valid_workload)
            mutation[field_name] = invalid_value
            with pytest.raises(AssertionError):
                _BENCHMARK_SCRIPT.WorkloadResult(**mutation)
            workload_integer_mutations += 1
    assert workload_integer_mutations == 15

    workload_float_mutations = 0
    for field_name in (
        "median_microseconds",
        "p95_microseconds",
        "median_per_item_microseconds",
        "p95_per_item_microseconds",
    ):
        for invalid_value in (
            False,
            True,
            1,
            "1.0",
            float("inf"),
            float("nan"),
            None,
        ):
            mutation = dict(valid_workload)
            mutation[field_name] = invalid_value
            with pytest.raises(AssertionError):
                _BENCHMARK_SCRIPT.WorkloadResult(**mutation)
            workload_float_mutations += 1
    assert workload_float_mutations == 28

    invalid_sample_sets: tuple[object, ...] = (
        (False,),
        (True,),
        (1,),
        ("1.0",),
        (float("inf"),),
        (float("nan"),),
        (),
        [1.0],
    )
    for invalid_samples in invalid_sample_sets:
        mutation = dict(valid_workload)
        mutation["sample_microseconds"] = invalid_samples
        with pytest.raises(AssertionError):
            _BENCHMARK_SCRIPT.WorkloadResult(**mutation)

    benchmark_fixture = _BENCHMARK_SCRIPT._fixture_items(1, "x")
    for invalid_factory_count in (False, True, 1.0):
        with pytest.raises(AssertionError):
            _BENCHMARK_SCRIPT.BenchmarkFixture(
                items=benchmark_fixture.items,
                harness_canonical_item_factory_calls=invalid_factory_count,
            )

    with _BENCHMARK_SCRIPT._network_denied():
        socket_module = import_module("socket")
        with pytest.raises(AssertionError, match="network access"):
            socket_module.create_connection(("127.0.0.1", 9))

    collection_calls: list[None] = []
    states: list[object] = []
    accumulators: list[object] = []
    original_create_state = (
        _BENCHMARK_SCRIPT._BenchmarkObserver.create_projection_state
    )
    original_isolated_memory_probe = _BENCHMARK_SCRIPT._isolated_memory_probe

    def create_observed_state(observer: object) -> object:
        state = original_create_state(observer)
        states.append(state)
        accumulators.append(state.accumulator)
        return state

    monkeypatch.setattr(
        _BENCHMARK_SCRIPT,
        "collect",
        lambda: collection_calls.append(None),
    )
    monkeypatch.setattr(
        _BENCHMARK_SCRIPT._BenchmarkObserver,
        "create_projection_state",
        create_observed_state,
    )
    monkeypatch.setattr(
        _BENCHMARK_SCRIPT,
        "_isolated_memory_probe",
        lambda *_args: _BENCHMARK_SCRIPT.MemoryProbeResult(
            peak_processing_bytes_excluding_source_fixture=1,
            current_retained_bytes_including_source_fixture=1,
            peak_total_bytes_including_source_fixture=1,
        ),
    )

    _BENCHMARK_SCRIPT._run_workload(1, protocol)

    expected_projection_runs = protocol.warmups + protocol.samples
    assert len(collection_calls) == expected_projection_runs + 1
    assert len(states) == expected_projection_runs
    assert len({id(state) for state in states}) == expected_projection_runs
    assert len({id(item) for item in accumulators}) == expected_projection_runs

    captured_timeout: list[int] = []
    probe_returncode: list[object] = [0]
    probe_raw_payload: list[str | None] = [None]
    probe_payload: dict[str, object] = {
        "current_retained_bytes_including_source_fixture": 1,
        "peak_processing_bytes_excluding_source_fixture": 1,
        "peak_total_bytes_including_source_fixture": 1,
    }

    def completed_memory_probe(*_args: object, **kwargs: object) -> object:
        timeout = kwargs.get("timeout")
        assert type(timeout) is int
        captured_timeout.append(timeout)
        return CompletedProcess(
            args=["python"],
            returncode=cast(Any, probe_returncode[0]),
            stdout=(
                "__REASONING_BENCHMARK_MEMORY__"
                + (
                    probe_raw_payload[0]
                    if probe_raw_payload[0] is not None
                    else dumps(probe_payload, sort_keys=True)
                )
                + "\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(_BENCHMARK_SCRIPT, "run", completed_memory_probe)
    original_isolated_memory_probe(1, "x", protocol)
    assert captured_timeout == [60]

    for invalid_delta_count in (False, True, 1.0):
        with pytest.raises(AssertionError):
            original_isolated_memory_probe(
                invalid_delta_count,
                "x",
                protocol,
            )

    for invalid_returncode in (1, False, True, 0.0, 1.0, "0", None):
        probe_returncode[0] = invalid_returncode
        with pytest.raises(AssertionError):
            original_isolated_memory_probe(1, "x", protocol)
    probe_returncode[0] = 0

    memory_probe_mutations = 0
    for field_name in tuple(probe_payload):
        original_value = probe_payload[field_name]
        for invalid_value in (False, True, 1.0, -1, "1"):
            probe_payload[field_name] = invalid_value
            with pytest.raises(AssertionError):
                original_isolated_memory_probe(1, "x", protocol)
            memory_probe_mutations += 1
        probe_payload[field_name] = original_value
    assert memory_probe_mutations == 15

    probe_payload["unexpected"] = None
    with pytest.raises(AssertionError):
        original_isolated_memory_probe(1, "x", protocol)
    probe_payload.pop("unexpected")

    memory_source = dumps(probe_payload, sort_keys=True)
    probe_raw_payload[0] = _duplicate_json_name(
        memory_source,
        name="peak_total_bytes_including_source_fixture",
        serialized_value="1",
    )
    with pytest.raises(ValueError, match="duplicate JSON object name"):
        original_isolated_memory_probe(1, "x", protocol)
    probe_raw_payload[0] = None
    for constant in _NONFINITE_JSON_CONSTANTS:
        probe_raw_payload[0] = _inject_nonfinite_json_number(
            memory_source,
            constant=constant,
        )
        with pytest.raises(ValueError, match="non-finite JSON number"):
            original_isolated_memory_probe(1, "x", protocol)
    probe_raw_payload[0] = None


def test_benchmark_execution_report_matches_locked_protocol() -> None:
    report = _BENCHMARK_SCRIPT.run_benchmark()
    protocol = _BENCHMARK_SCRIPT.benchmark_protocol()
    workloads = report["workloads"]
    assert isinstance(workloads, list)
    assert len(workloads) == len(protocol.delta_counts)
    budget = StreamPerformanceBudget()
    assert report["protocol"] == asdict(protocol)
    assert report["stream_performance_budget"] == asdict(budget)

    for raw_workload, delta_count in zip(
        workloads,
        protocol.delta_counts,
        strict=True,
    ):
        assert isinstance(raw_workload, dict)
        samples = raw_workload["sample_microseconds"]
        deterministic = raw_workload["deterministic"]
        assert isinstance(samples, list)
        assert isinstance(deterministic, dict)
        assert len(samples) == protocol.samples
        assert raw_workload["delta_count"] == delta_count
        assert deterministic == asdict(
            _BENCHMARK_SCRIPT.expected_deterministic_counts(delta_count)
        )
        numeric_samples = [float(sample) for sample in samples]
        expected_median = median(numeric_samples)
        expected_p95 = sorted(numeric_samples)[
            ceil(0.95 * len(numeric_samples)) - 1
        ]
        item_count = delta_count + 4
        assert raw_workload["item_count"] == item_count
        assert raw_workload["median_microseconds"] == pytest.approx(
            expected_median
        )
        assert raw_workload["p95_microseconds"] == pytest.approx(expected_p95)
        assert raw_workload["median_per_item_microseconds"] == pytest.approx(
            expected_median / item_count
        )
        assert raw_workload["p95_per_item_microseconds"] == pytest.approx(
            expected_p95 / item_count
        )
        processing_peak = raw_workload[
            "peak_processing_bytes_excluding_source_fixture"
        ]
        retained_current = raw_workload[
            "current_retained_bytes_including_source_fixture"
        ]
        total_peak = raw_workload["peak_total_bytes_including_source_fixture"]
        assert type(processing_peak) is int and processing_peak > 0
        assert type(retained_current) is int and retained_current > 0
        assert type(total_peak) is int and total_peak > 0
        assert retained_current <= total_peak <= budget.max_memory_bytes
        assert processing_peak <= budget.max_memory_bytes

    assert report["python"]
    assert report["platform"]
    assert report["machine"]
    assert report["protocol"]["network_allowed"] is False


def test_benchmark_observer_counts_each_materialization_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _BENCHMARK_SCRIPT._fixture_items(4, "x")
    deterministic = _BENCHMARK_SCRIPT._project_fixture(fixture)
    assert deterministic.production_accumulator_instances == 1
    assert deterministic.production_accumulator_add_calls == 8
    assert deterministic.production_reasoning_text_property_reads == 1

    original_materialize = (
        _BENCHMARK_SCRIPT._BenchmarkObserver.materialize_reasoning
    )

    def materialize_twice(observer: object, accumulator: object) -> str:
        _ = cast(Any, accumulator).reasoning_text
        return cast(str, original_materialize(observer, accumulator))

    monkeypatch.setattr(
        _BENCHMARK_SCRIPT._BenchmarkObserver,
        "materialize_reasoning",
        materialize_twice,
    )
    with pytest.raises(AssertionError, match="deterministic benchmark work"):
        _BENCHMARK_SCRIPT._project_fixture(fixture)


def test_benchmark_cli_writes_json_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        _BENCHMARK_SCRIPT,
        "run_benchmark",
        lambda: {"schema_version": 1, "suite": "smoke"},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_reasoning_summary.py", "--json-out", str(output)],
    )

    assert _BENCHMARK_SCRIPT.main() == 0
    assert strict_json_loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "suite": "smoke",
    }


def test_phase0_baseline_records_quality_and_benchmark_evidence() -> None:
    path = reasoning_summary_fixture_root() / "phase0_baseline.json"
    baseline_source = path.read_text(encoding="utf-8")
    baseline = _assert_phase0_baseline_payload(
        strict_json_loads(baseline_source)
    )
    duplicate_baseline_sources = (
        _duplicate_json_name(
            baseline_source,
            name="schema_version",
            serialized_value="1",
        ),
        _duplicate_json_name(
            baseline_source,
            name="seconds",
            serialized_value="14.75",
        ),
    )
    for duplicate_source in duplicate_baseline_sources:
        with pytest.raises(ValueError, match="duplicate JSON object name"):
            strict_json_loads(duplicate_source)
    for constant in _NONFINITE_JSON_CONSTANTS:
        for container_marker in (None, '"git":'):
            nonfinite_source = _inject_nonfinite_json_number(
                baseline_source,
                constant=constant,
                container_marker=container_marker,
            )
            with pytest.raises(ValueError, match="non-finite JSON number"):
                strict_json_loads(nonfinite_source)
    _assert_integer_type_mutations_fail(
        baseline,
        _BASELINE_INTEGER_CATALOG,
    )
    _assert_float_type_mutations_fail(
        baseline,
        _BASELINE_FLOAT_CATALOG,
    )
    baseline_mapping_mutations = 0
    for _, mapping_path, _ in _json_mapping_entries(baseline):
        unknown_baseline = deepcopy(baseline)
        _add_unknown_mapping_key(unknown_baseline, mapping_path)
        with pytest.raises(AssertionError):
            _assert_phase0_baseline_payload(unknown_baseline)
        baseline_mapping_mutations += 1
    assert baseline_mapping_mutations == 17
    git_evidence = cast(dict[str, object], baseline["git"])
    assert git_evidence["head"] == "7cf4d79fe45a4909afae1c146259be833c6cdae9"
    quality_gate = cast(
        list[dict[str, object]],
        baseline["quality_gate"],
    )
    gates = {entry["command"]: entry for entry in quality_gate}
    assert gates["poetry run pytest --verbose -s"]["passed"] == 9452
    assert gates["poetry run pytest --verbose -s"]["skipped"] == 66
    coverage = gates[
        "jq -e '[.files[] | select(.summary.percent_covered < 100)] | "
        "length == 0' coverage.json"
    ]
    assert coverage["covered_statements"] == coverage["total_statements"]
    assert coverage["result"] is True

    runner = cast(dict[str, object], baseline["locked_runner"])
    protocol = _BENCHMARK_SCRIPT.benchmark_protocol()
    assert runner["protocol"] == loads(dumps(asdict(protocol)))
    workloads = cast(list[dict[str, object]], runner["workloads"])
    assert [workload["delta_count"] for workload in workloads] == [4096, 8192]
    budget = StreamPerformanceBudget()
    for workload in workloads:
        delta_count = workload["delta_count"]
        assert type(delta_count) is int
        samples = workload["sample_microseconds"]
        assert isinstance(samples, list) and len(samples) == protocol.samples
        numeric_samples = [float(sample) for sample in samples]
        assert workload["median_microseconds"] == pytest.approx(
            median(numeric_samples)
        )
        assert workload["p95_microseconds"] == pytest.approx(
            sorted(numeric_samples)[ceil(0.95 * len(numeric_samples)) - 1]
        )
        assert workload["deterministic"] == asdict(
            _BENCHMARK_SCRIPT.expected_deterministic_counts(delta_count)
        )
        processing_peak = workload[
            "peak_processing_bytes_excluding_source_fixture"
        ]
        retained_current = workload[
            "current_retained_bytes_including_source_fixture"
        ]
        total_peak = workload["peak_total_bytes_including_source_fixture"]
        assert type(processing_peak) is int
        assert type(retained_current) is int
        assert type(total_peak) is int
        assert 0 < processing_peak <= budget.max_memory_bytes
        assert 0 < retained_current <= total_peak
        assert total_peak <= budget.max_memory_bytes
    known_phase5_target = runner["known_phase5_target"]
    assert isinstance(known_phase5_target, str)
    assert "Phase 5" in known_phase5_target


def test_requirement_traceability_is_complete() -> None:
    traceability_source = (
        reasoning_summary_fixture_root() / "requirements_traceability.json"
    ).read_text(encoding="utf-8")
    traceability = strict_json_loads(traceability_source)
    duplicate_traceability_sources = (
        _duplicate_json_name(
            traceability_source,
            name="schema_version",
            serialized_value="1",
        ),
        _duplicate_json_name(
            traceability_source,
            name="source_line",
            serialized_value="98",
        ),
    )
    for duplicate_source in duplicate_traceability_sources:
        with pytest.raises(ValueError, match="duplicate JSON object name"):
            strict_json_loads(duplicate_source)
    for constant in _NONFINITE_JSON_CONSTANTS:
        for container_marker in (None, '"catalog_invariant":'):
            nonfinite_source = _inject_nonfinite_json_number(
                traceability_source,
                constant=constant,
                container_marker=container_marker,
            )
            with pytest.raises(ValueError, match="non-finite JSON number"):
                strict_json_loads(nonfinite_source)
    assert isinstance(traceability, dict)
    catalog = cast(dict[str, object], traceability)
    _assert_self_contained_requirements_catalog(catalog)
    _assert_exact_json_integer_catalog(
        catalog,
        _REQUIREMENTS_INTEGER_CATALOG,
    )
    _assert_integer_type_mutations_fail(
        catalog,
        _REQUIREMENTS_INTEGER_CATALOG,
    )

    deleted = deepcopy(catalog)
    deleted_normative = cast(list[object], deleted["normative_requirements"])
    deleted_normative.pop()

    mutated = deepcopy(catalog)
    mutated_acceptance = cast(
        list[dict[str, object]], mutated["acceptance_criteria"]
    )
    mutated_acceptance[0]["phase"] = 8

    wrong_digest = deepcopy(catalog)
    wrong_invariant = cast(
        dict[str, object], wrong_digest["catalog_invariant"]
    )
    wrong_invariant["sha256"] = "0" * 64

    unknown = deepcopy(catalog)
    unknown["unexpected"] = None

    for invalid_catalog in (deleted, mutated, wrong_digest, unknown):
        with pytest.raises(AssertionError):
            _assert_self_contained_requirements_catalog(invalid_catalog)

    for invalid_schema_version in (False, True, 1.0):
        invalid_schema = deepcopy(catalog)
        invalid_schema["schema_version"] = invalid_schema_version
        with pytest.raises(AssertionError):
            _assert_self_contained_requirements_catalog(invalid_schema)

    for count_name, expected_count in (
        ("normative_count", 54),
        ("acceptance_count", 22),
    ):
        for invalid_count in (False, True, float(expected_count)):
            invalid_catalog = deepcopy(catalog)
            invalid_invariant = cast(
                dict[str, object], invalid_catalog["catalog_invariant"]
            )
            invalid_invariant[count_name] = invalid_count
            with pytest.raises(AssertionError):
                _assert_self_contained_requirements_catalog(invalid_catalog)


def test_acceptance_manifest_is_schema_versioned_and_phase_scoped() -> None:
    manifest_path = _ACCEPTANCE_SCRIPT.default_manifest_path()
    manifest = _ACCEPTANCE_SCRIPT.load_manifest(manifest_path)
    raw_manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(raw_manifest, dict)
    _assert_exact_json_integer_catalog(
        raw_manifest,
        _ACCEPTANCE_INTEGER_CATALOG,
    )
    _assert_integer_type_mutations_fail(
        raw_manifest,
        _ACCEPTANCE_INTEGER_CATALOG,
    )

    assert manifest.active_phase == 2
    assert len(manifest.node_ids) == len(set(manifest.node_ids))
    assert set(_PHASE0_MANIFEST_DIMENSION_NAMES).issubset(manifest.dimensions)
    _assert_exact_phase0_manifest_catalog(manifest.dimensions)


def test_acceptance_manifest_pins_complete_phase0_catalog() -> None:
    manifest = _ACCEPTANCE_SCRIPT.load_manifest(
        _ACCEPTANCE_SCRIPT.default_manifest_path()
    )
    _assert_exact_phase0_manifest_catalog(manifest.dimensions)

    missing_baseline = deepcopy(manifest.dimensions)
    protocol_nodes = missing_baseline["protocol projection baseline"]
    missing_baseline["protocol projection baseline"] = protocol_nodes[:-1]

    wrong_file = deepcopy(manifest.dimensions)
    infrastructure = wrong_file["phase 0 infrastructure"]
    wrong_file["phase 0 infrastructure"] = (
        infrastructure[0].replace(
            "tests/reasoning_summary_phase0_test.py",
            "tests/reasoning_summary_wrong_test.py",
        ),
        *infrastructure[1:],
    )

    wrong_class = deepcopy(manifest.dimensions)
    openai_nodes = wrong_class["OpenAI omission retry and replay baseline"]
    wrong_class["OpenAI omission retry and replay baseline"] = (
        openai_nodes[0].replace("OpenAITestCase", "WrongOpenAITestCase"),
        *openai_nodes[1:],
    )

    for mutation in (missing_baseline, wrong_file, wrong_class):
        with pytest.raises(AssertionError):
            _assert_exact_phase0_manifest_catalog(mutation)


def test_phase0_contract_file_is_checked_in_location() -> None:
    contract = reasoning_summary_fixture_root() / "phase0_contract.json"
    baseline = strict_json_loads(
        (reasoning_summary_fixture_root() / "phase0_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(baseline, dict)

    assert contract.is_file()
    assert contract == Path(contract)
    ignored_specs = baseline["ignored_specs"]
    assert ignored_specs == {
        "rule": ".gitignore:6:specs/",
        "policy": "never_track_remain_ignored",
        "must_remain_ignored": True,
        "must_never_be_committed": True,
        "tracked_tests_are_self_contained": True,
        "local_human_reference_paths": [
            "specs/TOOL-REASONING.md",
            "specs/TOOL-REASONING-AGENDA.md",
        ],
    }
