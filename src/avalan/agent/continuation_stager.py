"""Stage portable agent continuations without optional service imports."""

from ..entities import Message
from ..interaction.broker import InteractionBrokerRequest
from ..interaction.continuation import (
    ContinuationClaim,
    ContinuationFencingToken,
    ContinuationStoreRevision,
    PortableContinuation,
)
from ..interaction.durable import DurableInteractionSuspension
from ..interaction.entities import (
    InputRequestId,
    StateRevision,
    create_input_request,
)
from ..interaction.error import InputErrorCode, InputValidationError
from ..interaction.store import CreateInteractionCommand
from ..interaction.validation import validate_aware_datetime
from ..memory.permanent.codec import encode_message_data
from ..types import JsonValue
from .execution import (
    AgentExecution,
    DurableInteractionStagingContext,
    ExecutionCorrelationError,
)
from .orchestrator_response_contract import (
    DurableOrchestratorResponse,
)

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, timedelta
from datetime import datetime as datetime
from enum import Enum
from json import dumps, loads
from typing import cast, final
from uuid import uuid4

_TRANSCRIPT_VERSION = 1
_OBSERVATION_VERSION = 1
_EXECUTION_OBSERVATION_KIND = "agent_execution"


@final
class PortableAgentContinuationStager:
    """Build portable continuation state before durable persistence."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock or (lambda: datetime.now(UTC))

    async def __call__(
        self,
        request: InteractionBrokerRequest,
        *,
        execution: AgentExecution,
        response: object,
        stream_sequence: int,
        staging: DurableInteractionStagingContext,
    ) -> DurableInteractionSuspension:
        """Return one exact uncommitted request and portable checkpoint."""
        if type(request) is not InteractionBrokerRequest:
            raise TypeError("request must be an interaction broker request")
        if not isinstance(execution, AgentExecution):
            raise TypeError("execution must be an agent execution")
        if not isinstance(response, DurableOrchestratorResponse):
            raise TypeError("response must be an orchestrator response")
        if type(staging) is not DurableInteractionStagingContext:
            raise TypeError("staging must be a durable staging context")
        now = validate_aware_datetime(
            self._clock(),
            "continuation_stager.clock",
        )
        request_id = InputRequestId(f"request-{uuid4()}")
        continuation_id = staging.continuation_id
        created = create_input_request(
            request_id=request_id,
            continuation_id=continuation_id,
            origin=request.origin,
            mode=request.mode,
            reason=request.reason,
            questions=request.questions,
            created_at=now,
            continuation_ttl_seconds=request.continuation_ttl_seconds,
            advisory_wait_seconds=request.advisory_wait_seconds,
        )
        snapshot = execution.snapshot
        reservation = next(
            (
                entry
                for entry in reversed(snapshot.ledger)
                if entry.task_input_call is not None
                and entry.interaction_assistant_message is not None
            ),
            None,
        )
        if (
            reservation is None
            or reservation.task_input_call != staging.task_input_call
            or snapshot.active_interaction_fingerprint is None
        ):
            raise ExecutionCorrelationError(
                "durable checkpoint changed its active interaction"
            )
        assistant_message = reservation.interaction_assistant_message
        assert assistant_message is not None
        transcript = tuple(
            _encode_message_record(message) for message in execution.messages
        )
        observations = (
            cast(
                Mapping[str, JsonValue],
                {
                    "version": _OBSERVATION_VERSION,
                    "kind": _EXECUTION_OBSERVATION_KIND,
                    "active_interaction_fingerprint": (
                        snapshot.active_interaction_fingerprint
                    ),
                    "interaction_fingerprint_counts": [
                        {"fingerprint": fingerprint, "count": count}
                        for fingerprint, count in (
                            snapshot.interaction_fingerprint_counts
                        )
                    ],
                    "assistant_message": _encode_message_record(
                        assistant_message
                    ),
                },
            ),
        )
        generation_settings = _portable_json_mapping(
            response.continuation_generation_settings,
            "continuation.generation_settings",
        )
        continuation = PortableContinuation(
            continuation_id=continuation_id,
            request_id=request_id,
            origin=request.origin,
            provider_call_id=request.origin.model_call_id,
            provider_call_correlation_id=(
                staging.provider_call_correlation_id
            ),
            definition=execution.definition,
            operation_cursor=execution.operation_index,
            generation_settings=generation_settings,
            transcript=transcript,
            observations=observations,
            revision_binding=staging.revision_binding,
            interaction_count=execution.interaction_count,
            tool_loop_count=response.continuation_tool_loop_count,
            stream_sequence=stream_sequence,
            state_revision=StateRevision(execution.revision),
            store_revision=ContinuationStoreRevision(0),
            created_at=now,
            updated_at=now,
            expires_at=now
            + timedelta(seconds=request.continuation_ttl_seconds),
            claim=ContinuationClaim(),
            fencing_token=ContinuationFencingToken(0),
            provider_snapshot=staging.provider_snapshot,
        )
        return DurableInteractionSuspension(
            command=CreateInteractionCommand(
                actor=request.actor,
                request=created,
            ),
            continuation=continuation,
        )


def _encode_message_record(message: Message) -> Mapping[str, JsonValue]:
    return cast(
        Mapping[str, JsonValue],
        {
            "version": _TRANSCRIPT_VERSION,
            "role": message.role.value,
            "data": encode_message_data(message),
        },
    )


def _portable_json_mapping(
    value: Mapping[str, object],
    path: str,
) -> Mapping[str, JsonValue]:
    try:
        encoded = dumps(
            value,
            default=_json_default,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = loads(encoded)
    except (TypeError, ValueError) as error:
        raise InputValidationError(
            InputErrorCode.NON_JSON_VALUE,
            path,
            "value must contain only portable JSON settings",
        ) from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise InputValidationError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "value must be a JSON object",
        )
    return cast(Mapping[str, JsonValue], decoded)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not portable JSON")
