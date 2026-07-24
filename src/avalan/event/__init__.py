from ..interaction import (
    AgentId,
    AnswerProvenance,
    BranchId,
    InputErrorCode,
    InputRequestId,
    RequestState,
    ResolutionStatus,
    RunId,
    TaskId,
    TurnId,
)
from ..interaction.validation import validate_opaque_id
from ..observability import observability_key_sample
from ..types import (
    LooseJsonValue,
    assert_non_negative_int,
    assert_non_negative_number,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import NewType, TypeVar, cast


class EventType(StrEnum):
    CALL_PREPARE_BEFORE = "call_prepare_before"
    CALL_PREPARE_AFTER = "call_prepare_after"
    END = "end"
    ENGINE_AGENT_CALL_BEFORE = "engine_agent_call_before"
    ENGINE_AGENT_CALL_AFTER = "engine_agent_call_after"
    ENGINE_RUN_BEFORE = "engine_run_before"
    ENGINE_RUN_AFTER = "engine_run_after"
    INPUT_TOKEN_COUNT_BEFORE = "input_token_count_before"
    INPUT_TOKEN_COUNT_AFTER = "input_token_count_after"
    INTERACTION_LIFECYCLE = "interaction_lifecycle"
    MEMORY_APPEND_BEFORE = "memory_append_before"
    MEMORY_APPEND_AFTER = "memory_append_after"
    MEMORY_PERMANENT_MESSAGE_ADD = "memory_permanent_message_add"
    MEMORY_PERMANENT_MESSAGE_ADDED = "memory_permanent_message_added"
    MEMORY_PERMANENT_MESSAGE_SESSION_CONTINUE = (
        "memory_permanent_message_session_continue"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_CONTINUED = (
        "memory_permanent_message_session_continued"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_START = (
        "memory_permanent_message_session_start"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_STARTED = (
        "memory_permanent_message_session_started"
    )
    MODEL_EXECUTE_BEFORE = "model_execute_before"
    MODEL_EXECUTE_AFTER = "model_execute_after"
    MODEL_MANAGER_CALL_BEFORE = "model_manager_call_before"
    MODEL_MANAGER_CALL_AFTER = "model_manager_call_after"
    FLOW_CANCELLED = "flow_cancelled"
    FLOW_COMPLETED = "flow_completed"
    FLOW_CONDITION_EVALUATED = "flow_condition_evaluated"
    FLOW_CONTAINER_EVENT = "flow_container_event"
    FLOW_EDGE_ELIGIBLE = "flow_edge_eligible"
    FLOW_EDGE_ROUTED = "flow_edge_routed"
    FLOW_JOIN_READY = "flow_join_ready"
    FLOW_MANAGER_CALL_BEFORE = "flow_manager_call_before"
    FLOW_MANAGER_CALL_AFTER = "flow_manager_call_after"
    FLOW_NODE_CANCELLED = "flow_node_cancelled"
    FLOW_NODE_COMPLETED = "flow_node_completed"
    FLOW_NODE_FAILED = "flow_node_failed"
    FLOW_NODE_PAUSED = "flow_node_paused"
    FLOW_NODE_RESUMED = "flow_node_resumed"
    FLOW_NODE_RETRYING = "flow_node_retrying"
    FLOW_NODE_SKIPPED = "flow_node_skipped"
    FLOW_NODE_STARTED = "flow_node_started"
    FLOW_OUTPUT_SELECTED = "flow_output_selected"
    FLOW_STARTED = "flow_started"
    FLOW_VALIDATION = "flow_validation"
    SKILL_CHECK_DIAGNOSTICS_PRODUCED = "skill_check_diagnostics_produced"
    SKILL_DISABLED = "skill_disabled"
    SKILL_DUPLICATE = "skill_duplicate"
    SKILL_MALFORMED = "skill_malformed"
    SKILL_MATCH_AMBIGUOUS = "skill_match_ambiguous"
    SKILL_MATCH_CANDIDATES_RETURNED = "skill_match_candidates_returned"
    SKILL_MATCH_EMPTY = "skill_match_empty"
    SKILL_MATCH_QUERY_EVALUATED = "skill_match_query_evaluated"
    SKILL_READ_ALLOWED = "skill_read_allowed"
    SKILL_READ_BLOCKED = "skill_read_blocked"
    SKILL_READ_DELETED = "skill_read_deleted"
    SKILL_READ_DENIED = "skill_read_denied"
    SKILL_READ_STALE = "skill_read_stale"
    SKILL_READ_TRUNCATED = "skill_read_truncated"
    SKILL_REGISTERED = "skill_registered"
    SKILL_REGISTRY_BUILD_COMPLETED = "skill_registry_build_completed"
    SKILL_REGISTRY_BUILD_FAILED = "skill_registry_build_failed"
    SKILL_REGISTRY_BUILD_STARTED = "skill_registry_build_started"
    SKILL_SHADOWED = "skill_shadowed"
    SKILL_SOURCE_ACCEPTED = "skill_source_accepted"
    SKILL_SOURCE_POLICY_DENIED = "skill_source_policy_denied"
    SKILL_SOURCE_SKIPPED = "skill_source_skipped"
    SKILL_SOURCE_UNAVAILABLE = "skill_source_unavailable"
    START = "start"
    STREAM_END = "stream_end"
    TOKEN_GENERATED = "token_generated"
    TOOL_DETECT = "tool_detect"
    TOOL_DIAGNOSTIC = "tool_diagnostic"
    TOOL_EXECUTE = "tool_execute"
    TOOL_MODEL_RUN = "tool_model_run"
    TOOL_MODEL_RESPONSE = "tool_model_response"
    TOOL_PROCESS = "tool_process"
    TOOL_PROGRESS = "tool_progress"
    TOOL_RESULT = "tool_result"


TOOL_TYPES = {et for et in EventType if et.value.startswith("tool_")}


class EventPayloadKind(StrEnum):
    CANONICAL_STREAM = "canonical_stream"
    INTERACTION_LIFECYCLE = "interaction_lifecycle"
    TEMPORARY_LEGACY = "temporary_legacy"


_INTERACTION_LIFECYCLE_REQUIRED_FIELDS = frozenset(
    {
        "agent_id",
        "branch_id",
        "request_id",
        "run_id",
        "state",
        "turn_id",
    }
)
_INTERACTION_LIFECYCLE_OPTIONAL_FIELDS = frozenset(
    {
        "duplicate",
        "provenance_category",
        "resolution_category",
        "stale",
        "surface",
        "task_id",
        "validation_code",
        "wait_duration_ms",
    }
)
_INTERACTION_LIFECYCLE_TERMINAL_STATES = frozenset(
    {
        RequestState.ANSWERED,
        RequestState.CANCELLED,
        RequestState.DECLINED,
        RequestState.EXPIRED,
        RequestState.SUPERSEDED,
        RequestState.TIMED_OUT,
        RequestState.UNAVAILABLE,
    }
)
_TELEMETRY_NAME_CHARACTERS = frozenset("._:-")
_TELEMETRY_NAME_LIMIT = 256
_OBSERVER_ID_PREFIX = "oid_"
_OBSERVER_ID_HEX_LENGTH = 64
_OBSERVER_ID_LENGTH = len(_OBSERVER_ID_PREFIX) + _OBSERVER_ID_HEX_LENGTH
_LOWER_HEX_CHARACTERS = frozenset("0123456789abcdef")

ObserverId = NewType("ObserverId", str)
_StrEnumValue = TypeVar("_StrEnumValue", bound=StrEnum)


def validate_observer_id(
    value: object,
    field_name: str = "observer_id",
) -> ObserverId:
    """Return one hashed identifier safe for observer surfaces."""
    assert isinstance(field_name, str), "field_name must be a string"
    assert field_name, "field_name must not be empty"
    assert isinstance(value, str), f"{field_name} must be a string"
    assert (
        len(value) == _OBSERVER_ID_LENGTH
    ), f"{field_name} must be a canonical observer token"
    assert value.startswith(
        _OBSERVER_ID_PREFIX
    ), f"{field_name} must be a canonical observer token"
    assert all(
        character in _LOWER_HEX_CHARACTERS
        for character in value[len(_OBSERVER_ID_PREFIX) :]
    ), f"{field_name} must be a canonical observer token"
    return ObserverId(value)


def project_observer_id(
    value: object,
    field_name: str = "observer_id",
) -> ObserverId:
    """Project one canonical opaque identifier into an observer token."""
    assert isinstance(field_name, str), "field_name must be a string"
    assert field_name, "field_name must not be empty"
    canonical = validate_opaque_id(value, field_name)
    digest = sha256(f"{field_name}\0{canonical}".encode("utf-8")).hexdigest()
    return ObserverId(f"{_OBSERVER_ID_PREFIX}{digest}")


@dataclass(frozen=True, kw_only=True, slots=True)
class InteractionLifecyclePayload:
    """Project one interaction transition into content-safe telemetry."""

    request_id: ObserverId
    run_id: ObserverId
    turn_id: ObserverId
    agent_id: ObserverId
    branch_id: ObserverId
    state: RequestState
    task_id: ObserverId | None = None
    resolution_category: ResolutionStatus | None = None
    surface: str | None = None
    wait_duration_ms: int | None = None
    validation_code: InputErrorCode | None = None
    duplicate: bool | None = None
    stale: bool | None = None
    provenance_category: AnswerProvenance | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("request_id", self.request_id),
            ("run_id", self.run_id),
            ("turn_id", self.turn_id),
            ("agent_id", self.agent_id),
            ("branch_id", self.branch_id),
        ):
            object.__setattr__(
                self,
                field_name,
                validate_observer_id(value, field_name),
            )
        if self.task_id is not None:
            object.__setattr__(
                self,
                "task_id",
                validate_observer_id(self.task_id, "task_id"),
            )
        assert isinstance(self.state, RequestState)
        if self.resolution_category is not None:
            assert isinstance(self.resolution_category, ResolutionStatus)
        if self.validation_code is not None:
            assert isinstance(self.validation_code, InputErrorCode)
        if self.provenance_category is not None:
            assert isinstance(self.provenance_category, AnswerProvenance)
        if self.surface is not None:
            _assert_telemetry_name(self.surface, "surface")
        if self.wait_duration_ms is not None:
            assert_non_negative_int(self.wait_duration_ms, "wait_duration_ms")
        for field_name, boolean_value in (
            ("duplicate", self.duplicate),
            ("stale", self.stale),
        ):
            if boolean_value is not None:
                assert isinstance(
                    boolean_value, bool
                ), f"{field_name} must be boolean"
        _assert_resolution_fields(
            self.state,
            self.resolution_category,
            self.provenance_category,
        )

    @classmethod
    def from_canonical_ids(
        cls,
        *,
        request_id: InputRequestId,
        run_id: RunId,
        turn_id: TurnId,
        agent_id: AgentId,
        branch_id: BranchId,
        state: RequestState,
        task_id: TaskId | None = None,
        resolution_category: ResolutionStatus | None = None,
        surface: str | None = None,
        wait_duration_ms: int | None = None,
        validation_code: InputErrorCode | None = None,
        duplicate: bool | None = None,
        stale: bool | None = None,
        provenance_category: AnswerProvenance | None = None,
    ) -> "InteractionLifecyclePayload":
        """Project canonical interaction IDs into one observer payload."""
        return cls(
            request_id=project_observer_id(request_id, "request_id"),
            run_id=project_observer_id(run_id, "run_id"),
            turn_id=project_observer_id(turn_id, "turn_id"),
            task_id=(
                project_observer_id(task_id, "task_id")
                if task_id is not None
                else None
            ),
            agent_id=project_observer_id(agent_id, "agent_id"),
            branch_id=project_observer_id(branch_id, "branch_id"),
            state=state,
            resolution_category=resolution_category,
            surface=surface,
            wait_duration_ms=wait_duration_ms,
            validation_code=validation_code,
            duplicate=duplicate,
            stale=stale,
            provenance_category=provenance_category,
        )

    def to_dict(self) -> dict[str, LooseJsonValue]:
        """Return the strict interaction telemetry projection."""
        data: dict[str, LooseJsonValue] = {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "branch_id": self.branch_id,
            "state": self.state.value,
        }
        optional_values: tuple[tuple[str, LooseJsonValue | None], ...] = (
            (
                "task_id",
                self.task_id,
            ),
            (
                "resolution_category",
                (
                    self.resolution_category.value
                    if self.resolution_category is not None
                    else None
                ),
            ),
            ("surface", self.surface),
            ("wait_duration_ms", self.wait_duration_ms),
            (
                "validation_code",
                (
                    self.validation_code.value
                    if self.validation_code is not None
                    else None
                ),
            ),
            ("duplicate", self.duplicate),
            ("stale", self.stale),
            (
                "provenance_category",
                (
                    self.provenance_category.value
                    if self.provenance_category is not None
                    else None
                ),
            ),
        )
        for field_name, value in optional_values:
            if value is not None:
                data[field_name] = value
        return data


@dataclass(frozen=True, kw_only=True, slots=True)
class EventObservabilityPayload:
    kind: EventPayloadKind
    data: Mapping[str, LooseJsonValue]
    owner: str | None = None
    removal_condition: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.kind, EventPayloadKind)
        assert isinstance(self.data, Mapping)
        assert self.data
        for key in self.data:
            assert isinstance(key, str)
            assert key.strip()
        if self.kind is EventPayloadKind.INTERACTION_LIFECYCLE:
            _validate_interaction_lifecycle_data(self.data)
            object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        if self.kind is EventPayloadKind.TEMPORARY_LEGACY:
            assert isinstance(self.owner, str)
            assert self.owner.strip()
            assert isinstance(self.removal_condition, str)
            assert self.removal_condition.strip()
        else:
            assert self.owner is None
            assert self.removal_condition is None

    @classmethod
    def canonical_stream(
        cls, data: Mapping[str, LooseJsonValue]
    ) -> "EventObservabilityPayload":
        return cls(kind=EventPayloadKind.CANONICAL_STREAM, data=data)

    @classmethod
    def interaction_lifecycle(
        cls,
        payload: InteractionLifecyclePayload,
    ) -> "EventObservabilityPayload":
        """Wrap a typed content-safe interaction lifecycle payload."""
        assert isinstance(payload, InteractionLifecyclePayload)
        return cls(
            kind=EventPayloadKind.INTERACTION_LIFECYCLE,
            data=payload.to_dict(),
        )

    @classmethod
    def temporary_legacy(
        cls,
        data: Mapping[str, LooseJsonValue],
        *,
        owner: str,
        removal_condition: str,
    ) -> "EventObservabilityPayload":
        return cls(
            kind=EventPayloadKind.TEMPORARY_LEGACY,
            data=data,
            owner=owner,
            removal_condition=removal_condition,
        )

    def to_dict(self) -> dict[str, LooseJsonValue]:
        payload: dict[str, LooseJsonValue] = {
            "kind": self.kind.value,
            "data": dict(self.data),
        }
        if self.owner is not None:
            payload["owner"] = self.owner
        if self.removal_condition is not None:
            payload["removal_condition"] = self.removal_condition
        return payload


@dataclass(frozen=True, kw_only=True, slots=True)
class Event:
    type: EventType
    payload: object | None = None
    observability_payload: EventObservabilityPayload | None = None
    started: float | None = None
    finished: float | None = None
    elapsed: float | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.type, EventType | str)
        if self.observability_payload is not None:
            assert isinstance(
                self.observability_payload, EventObservabilityPayload
            )
        if self.type == EventType.INTERACTION_LIFECYCLE:
            self._project_interaction_event()
        for field_name, value in (
            ("started", self.started),
            ("finished", self.finished),
            ("elapsed", self.elapsed),
        ):
            if value is not None:
                assert_non_negative_number(value, field_name)

    @classmethod
    def from_observability_payload(
        cls,
        *,
        type: EventType,
        observability_payload: EventObservabilityPayload,
        started: float | None = None,
        finished: float | None = None,
        elapsed: float | None = None,
    ) -> "Event":
        return cls(
            type=type,
            payload=dict(observability_payload.data),
            observability_payload=observability_payload,
            started=started,
            finished=finished,
            elapsed=elapsed,
        )

    @classmethod
    def from_interaction_lifecycle(
        cls,
        payload: InteractionLifecyclePayload,
        *,
        started: float | None = None,
        finished: float | None = None,
        elapsed: float | None = None,
    ) -> "Event":
        """Create one content-safe interaction lifecycle event."""
        return cls.from_observability_payload(
            type=EventType.INTERACTION_LIFECYCLE,
            observability_payload=(
                EventObservabilityPayload.interaction_lifecycle(payload)
            ),
            started=started,
            finished=finished,
            elapsed=elapsed,
        )

    @property
    def observability(self) -> EventObservabilityPayload:
        if self.observability_payload is not None:
            return self.observability_payload
        return EventObservabilityPayload.temporary_legacy(
            self._legacy_observability_data(),
            owner="event-listener-facade",
            removal_condition=(
                "Remove after CLI, server, and protocol listeners consume "
                "canonical stream projections."
            ),
        )

    @property
    def interaction_lifecycle_payload(
        self,
    ) -> InteractionLifecyclePayload | None:
        """Return the typed interaction lifecycle payload when available."""
        observability_payload = self.observability_payload
        if (
            self.type != EventType.INTERACTION_LIFECYCLE
            or observability_payload is None
            or observability_payload.kind
            is not EventPayloadKind.INTERACTION_LIFECYCLE
        ):
            return None
        return _interaction_lifecycle_payload(observability_payload.data)

    def for_history(self) -> "Event":
        if self.type == EventType.INTERACTION_LIFECYCLE:
            return Event(
                type=self.type,
                observability_payload=self.observability,
                started=self.started,
                finished=self.finished,
                elapsed=self.elapsed,
            )
        if self.payload is None:
            return self
        observability_payload = self.observability
        if (
            self.observability_payload is observability_payload
            and isinstance(self.payload, Mapping)
            and dict(self.payload) == dict(observability_payload.data)
        ):
            return self
        return Event(
            type=self.type,
            observability_payload=observability_payload,
            started=self.started,
            finished=self.finished,
            elapsed=self.elapsed,
        )

    def _project_interaction_event(self) -> None:
        observability_payload = self.observability_payload
        if observability_payload is None:
            object.__setattr__(self, "payload", None)
            return
        if observability_payload.kind is EventPayloadKind.CANONICAL_STREAM:
            projected = _project_interaction_lifecycle_data(
                observability_payload.data,
                require_terminal_provenance=False,
            )
            object.__setattr__(
                self,
                "observability_payload",
                EventObservabilityPayload.canonical_stream(
                    MappingProxyType(dict(projected))
                ),
            )
            if self.payload is not None:
                object.__setattr__(
                    self,
                    "payload",
                    MappingProxyType(dict(projected)),
                )
            return
        if (
            observability_payload.kind
            is EventPayloadKind.INTERACTION_LIFECYCLE
        ):
            if self.payload is not None:
                object.__setattr__(
                    self,
                    "payload",
                    MappingProxyType(dict(observability_payload.data)),
                )
            return
        object.__setattr__(self, "payload", None)

    def _legacy_observability_data(self) -> dict[str, LooseJsonValue]:
        event_type = (
            self.type.value if isinstance(self.type, EventType) else self.type
        )
        data: dict[str, LooseJsonValue] = {"event_type": event_type}
        if self.started is not None:
            data["started"] = float(self.started)
        if self.finished is not None:
            data["finished"] = float(self.finished)
        if self.elapsed is not None:
            data["elapsed"] = float(self.elapsed)
        if self.payload is not None:
            data["payload_summary"] = cast(
                dict[str, object], self._payload_summary()
            )
        return data

    def _payload_summary(self) -> dict[str, LooseJsonValue]:
        payload = self.payload
        summary: dict[str, LooseJsonValue] = {
            "type": type(payload).__name__,
        }
        if isinstance(payload, Mapping):
            keys, keys_truncated = observability_key_sample(payload)
            summary["keys"] = cast(list[object], keys)
            summary["size"] = len(payload)
            if keys_truncated:
                summary["keys_truncated"] = True
        elif isinstance(payload, list | tuple | set | frozenset):
            summary["size"] = len(payload)
        return summary


def _validate_interaction_lifecycle_data(
    data: Mapping[str, LooseJsonValue],
) -> None:
    fields = frozenset(data)
    assert _INTERACTION_LIFECYCLE_REQUIRED_FIELDS <= fields
    assert fields <= (
        _INTERACTION_LIFECYCLE_REQUIRED_FIELDS
        | _INTERACTION_LIFECYCLE_OPTIONAL_FIELDS
    )
    projected = _project_interaction_lifecycle_data(
        data,
        require_terminal_provenance=True,
    )
    assert dict(data) == projected


def _interaction_lifecycle_payload(
    data: Mapping[str, LooseJsonValue],
) -> InteractionLifecyclePayload:
    task_id_value = data.get("task_id")
    task_id = (
        validate_observer_id(task_id_value, "task_id")
        if task_id_value is not None
        else None
    )
    resolution_value = data.get("resolution_category")
    resolution = (
        _validated_str_enum(
            ResolutionStatus,
            resolution_value,
            "resolution_category",
        )
        if resolution_value is not None
        else None
    )
    surface_value = data.get("surface")
    if surface_value is not None:
        _assert_telemetry_name(surface_value, "surface")
        assert isinstance(surface_value, str)
    wait_duration_value = data.get("wait_duration_ms")
    if wait_duration_value is not None:
        assert_non_negative_int(wait_duration_value, "wait_duration_ms")
        assert isinstance(wait_duration_value, int)
        assert not isinstance(wait_duration_value, bool)
    validation_value = data.get("validation_code")
    validation_code = (
        _validated_str_enum(
            InputErrorCode,
            validation_value,
            "validation_code",
        )
        if validation_value is not None
        else None
    )
    duplicate_value = data.get("duplicate")
    if duplicate_value is not None:
        assert isinstance(duplicate_value, bool)
    stale_value = data.get("stale")
    if stale_value is not None:
        assert isinstance(stale_value, bool)
    provenance_value = data.get("provenance_category")
    provenance = (
        _validated_str_enum(
            AnswerProvenance,
            provenance_value,
            "provenance_category",
        )
        if provenance_value is not None
        else None
    )
    return InteractionLifecyclePayload(
        request_id=validate_observer_id(data.get("request_id"), "request_id"),
        run_id=validate_observer_id(data.get("run_id"), "run_id"),
        turn_id=validate_observer_id(data.get("turn_id"), "turn_id"),
        task_id=task_id,
        agent_id=validate_observer_id(data.get("agent_id"), "agent_id"),
        branch_id=validate_observer_id(data.get("branch_id"), "branch_id"),
        state=_validated_str_enum(RequestState, data.get("state"), "state"),
        resolution_category=resolution,
        surface=surface_value,
        wait_duration_ms=wait_duration_value,
        validation_code=validation_code,
        duplicate=duplicate_value,
        stale=stale_value,
        provenance_category=provenance,
    )


def _project_interaction_lifecycle_data(
    data: Mapping[str, LooseJsonValue],
    *,
    require_terminal_provenance: bool,
) -> dict[str, LooseJsonValue]:
    _validate_interaction_observer_ids(data)
    projected: dict[str, LooseJsonValue] = {
        field_name: data[field_name]
        for field_name in (
            "agent_id",
            "branch_id",
            "request_id",
            "run_id",
            "turn_id",
        )
    }
    task_id = data.get("task_id")
    if task_id is not None:
        projected["task_id"] = cast(LooseJsonValue, task_id)
    state = _validated_str_enum(
        RequestState,
        data["state"],
        "state",
    )
    projected["state"] = state.value
    resolution_value = data.get("resolution_category")
    resolution = None
    if resolution_value is not None:
        resolution = _validated_str_enum(
            ResolutionStatus,
            resolution_value,
            "resolution_category",
        )
        projected["resolution_category"] = resolution.value
    provenance_value = data.get("provenance_category")
    provenance = None
    if provenance_value is not None:
        provenance = _validated_str_enum(
            AnswerProvenance,
            provenance_value,
            "provenance_category",
        )
        projected["provenance_category"] = provenance.value
    if require_terminal_provenance:
        _assert_resolution_fields(state, resolution, provenance)
    elif state in _INTERACTION_LIFECYCLE_TERMINAL_STATES:
        assert resolution is ResolutionStatus(state.value)
    else:
        assert resolution is None
        assert provenance is None
    surface = data.get("surface")
    if surface is not None:
        _assert_telemetry_name(surface, "surface")
        projected["surface"] = cast(str, surface)
    wait_duration_ms = data.get("wait_duration_ms")
    if wait_duration_ms is not None:
        assert_non_negative_int(wait_duration_ms, "wait_duration_ms")
        projected["wait_duration_ms"] = cast(int, wait_duration_ms)
    validation_code = data.get("validation_code")
    if validation_code is not None:
        projected["validation_code"] = _validated_str_enum(
            InputErrorCode,
            validation_code,
            "validation_code",
        ).value
    for field_name in ("duplicate", "stale"):
        value = data.get(field_name)
        if value is not None:
            assert isinstance(value, bool), f"{field_name} must be boolean"
            projected[field_name] = value
    return projected


def _validate_interaction_observer_ids(
    data: Mapping[str, LooseJsonValue],
) -> None:
    for field_name in (
        "agent_id",
        "branch_id",
        "request_id",
        "run_id",
        "turn_id",
    ):
        validate_observer_id(data.get(field_name), field_name)
    task_id = data.get("task_id")
    if task_id is not None:
        validate_observer_id(task_id, "task_id")


def _assert_resolution_fields(
    state: RequestState,
    resolution: ResolutionStatus | None,
    provenance: AnswerProvenance | None,
) -> None:
    if state in _INTERACTION_LIFECYCLE_TERMINAL_STATES:
        assert resolution is ResolutionStatus(state.value)
        assert provenance is not None
        return
    assert resolution is None
    assert provenance is None


def _assert_non_empty_string(value: object, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def _validated_str_enum(
    enum_type: type[_StrEnumValue],
    value: object,
    field_name: str,
) -> _StrEnumValue:
    assert isinstance(value, str), f"{field_name} must be a string"
    for member in enum_type:
        if value == member.value:
            return member
    raise AssertionError(f"{field_name} must be a recognized value")


def _assert_telemetry_name(value: object, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert isinstance(value, str)
    assert len(value) <= _TELEMETRY_NAME_LIMIT
    assert value == value.strip()
    assert value.isascii()
    assert all(
        character.isalnum() or character in _TELEMETRY_NAME_CHARACTERS
        for character in value
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class EventStatsSnapshot:
    triggers: Mapping[EventType, int]
    total_triggers: int
    published: int
    delivered: int
    dropped: int
    coalesced: int
    failed: int
    queue_depth: int
    max_queue_depth: int
    listener_lag: float
    critical_wait_time: float


@dataclass(slots=True)
class EventStats:
    triggers: dict[EventType, int] = field(default_factory=dict)
    total_triggers: int = 0
    published: int = 0
    delivered: int = 0
    dropped: int = 0
    coalesced: int = 0
    failed: int = 0
    queue_depth: int = 0
    max_queue_depth: int = 0
    listener_lag: float = 0.0
    critical_wait_time: float = 0.0

    def record_trigger(self, event_type: EventType) -> None:
        assert isinstance(event_type, EventType)
        self.total_triggers += 1
        self.triggers[event_type] = self.triggers.get(event_type, 0) + 1

    def record_published(
        self, event_type: EventType, *, queue_depth: int = 0
    ) -> None:
        assert_non_negative_int(queue_depth, "queue_depth")
        self.record_trigger(event_type)
        self.published += 1
        self.record_queue_depth(queue_depth)

    def record_delivered(self, *, queue_depth: int = 0) -> None:
        assert_non_negative_int(queue_depth, "queue_depth")
        self.delivered += 1
        self.record_queue_depth(queue_depth)

    def record_dropped(
        self, count: int = 1, *, queue_depth: int | None = None
    ) -> None:
        assert_non_negative_int(count, "count")
        if queue_depth is not None:
            assert_non_negative_int(queue_depth, "queue_depth")
        self.dropped += count
        if queue_depth is not None:
            self.record_queue_depth(queue_depth)

    def record_queue_depth(self, queue_depth: int) -> None:
        assert_non_negative_int(queue_depth, "queue_depth")
        self.queue_depth = queue_depth
        self.max_queue_depth = max(self.max_queue_depth, queue_depth)

    def record_coalesced(self, count: int = 1) -> None:
        assert_non_negative_int(count, "count")
        self.coalesced += count

    def record_failed(self, count: int = 1) -> None:
        assert_non_negative_int(count, "count")
        self.failed += count

    def record_listener_lag(self, seconds: float) -> None:
        assert_non_negative_number(seconds, "seconds")
        self.listener_lag = max(self.listener_lag, float(seconds))

    def record_critical_wait_time(self, seconds: float) -> None:
        assert_non_negative_number(seconds, "seconds")
        self.critical_wait_time = max(self.critical_wait_time, float(seconds))

    def snapshot(self) -> EventStatsSnapshot:
        return EventStatsSnapshot(
            triggers=MappingProxyType(dict(self.triggers)),
            total_triggers=self.total_triggers,
            published=self.published,
            delivered=self.delivered,
            dropped=self.dropped,
            coalesced=self.coalesced,
            failed=self.failed,
            queue_depth=self.queue_depth,
            max_queue_depth=self.max_queue_depth,
            listener_lag=self.listener_lag,
            critical_wait_time=self.critical_wait_time,
        )
