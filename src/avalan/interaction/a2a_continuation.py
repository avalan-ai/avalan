"""Checkpoint one negotiated A2A tool input continuation."""

from ..types import JsonValue
from .a2a import (
    A2AInputRequestMetadata,
    decode_a2a_input_request_metadata,
    decode_a2a_input_resolution_metadata,
    encode_a2a_input_request_metadata,
    encode_a2a_input_resolution_metadata,
)
from .entities import (
    AnsweredResolution,
    CancelledResolution,
    DeclinedResolution,
    InputRequest,
    InputResolution,
    RequirementMode,
    TimedOutResolution,
    UnavailableResolution,
    _freeze_snapshot_object,
)
from .error import InputErrorCode, InputValidationError
from .validation import validate_opaque_id

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import NoReturn, cast, final

_KIND = "a2a_tool"
_VERSION = 1
_FIELDS = frozenset(
    (
        "version kind call_id canonical_name provider_name "
        "provider_name_encoded call_arguments request request_text "
        "task_id context_id prior_message_id "
        "prior_content ttl_seconds input_cycle_count interaction_counts"
    ).split()
)
_MAX_PRIOR_CONTENT_BYTES = 65_536
_MAX_PRIOR_CONTENT_ITEM_BYTES = 16_384
_MAX_PRIOR_CONTENT_ITEMS = 32


@final
class A2AInputRequiredError(RuntimeError):
    """Carry one durable A2A suspension across the tool boundary."""

    def __init__(self, continuation: "A2ARemoteInputContinuation") -> None:
        self.continuation = continuation
        super().__init__("A2A task requires durable input")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class A2ARemoteInputContinuation:
    """Carry the remote half of one suspended A2A input request."""

    request: A2AInputRequestMetadata
    request_text: str
    task_id: str
    context_id: str
    prior_message_id: str
    prior_content: tuple[str, ...]
    ttl_seconds: int
    input_cycle_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_text, str)
            or not self.request_text.strip()
        ):
            _invalid("a2a_continuation", "contains empty text")
        for name in ("task_id", "context_id", "prior_message_id"):
            object.__setattr__(
                self,
                name,
                validate_opaque_id(
                    getattr(self, name),
                    f"a2a_continuation.{name}",
                    maximum_characters=1_024,
                    maximum_bytes=4_096,
                ),
            )
        if type(self.request) is not A2AInputRequestMetadata:
            _invalid("a2a_continuation.request", "is invalid")
        if (
            type(self.ttl_seconds) is not int
            or not 60 <= self.ttl_seconds <= 604_800
            or type(self.input_cycle_count) is not int
            or self.input_cycle_count < 1
        ):
            _invalid("a2a_continuation", "has invalid bounds")
        if not isinstance(self.prior_content, tuple):
            _invalid("a2a_continuation.prior_content", "exceeds safe bounds")
        try:
            bound_a2a_prior_content(self.prior_content)
        except RuntimeError:
            _invalid("a2a_continuation.prior_content", "exceeds safe bounds")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class A2AToolContinuationCheckpoint:
    """Bind one remote continuation to its exact provider tool call."""

    call_id: str
    canonical_name: str
    provider_name: str
    provider_name_encoded: bool
    arguments: Mapping[str, JsonValue]
    remote: A2ARemoteInputContinuation
    interaction_fingerprint_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "call_id",
            validate_opaque_id(
                self.call_id,
                "a2a_continuation.call_id",
                maximum_characters=256,
                maximum_bytes=1_024,
            ),
        )
        if (
            any(
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
                for name in ("canonical_name", "provider_name")
            )
            or type(self.provider_name_encoded) is not bool
            or type(self.remote) is not A2ARemoteInputContinuation
            or tuple(sorted(self.interaction_fingerprint_counts))
            != self.interaction_fingerprint_counts
            or any(
                not fingerprint or type(count) is not int or count < 1
                for fingerprint, count in self.interaction_fingerprint_counts
            )
        ):
            _invalid("a2a_continuation.call", "is invalid")
        arguments = _json_object(
            self.arguments,
            "a2a_continuation.call_arguments",
        )
        object.__setattr__(self, "arguments", arguments)
        target_arguments = arguments.get("arguments")
        if (
            not isinstance(arguments.get("uri"), str)
            or not cast(str, arguments["uri"]).strip()
            or not isinstance(arguments.get("name"), str)
            or not cast(str, arguments["name"]).strip()
            or (
                target_arguments is not None
                and not isinstance(target_arguments, Mapping)
            )
        ):
            _invalid("a2a_continuation.call_arguments", "is invalid")

    @property
    def mode(self) -> RequirementMode:
        """Return the local interaction requirement mode."""
        return (
            RequirementMode.REQUIRED
            if self.remote.request.required
            else RequirementMode.ADVISORY
        )


def encode_a2a_tool_continuation_observation(
    checkpoint: A2AToolContinuationCheckpoint,
) -> Mapping[str, JsonValue]:
    """Encode one typed A2A tool checkpoint as portable JSON."""
    if type(checkpoint) is not A2AToolContinuationCheckpoint:
        raise TypeError("checkpoint must be an A2A tool continuation")
    remote = checkpoint.remote
    return cast(
        Mapping[str, JsonValue],
        {
            "version": _VERSION,
            "kind": _KIND,
            "call_id": checkpoint.call_id,
            "canonical_name": checkpoint.canonical_name,
            "provider_name": checkpoint.provider_name,
            "provider_name_encoded": checkpoint.provider_name_encoded,
            "call_arguments": checkpoint.arguments,
            "request": encode_a2a_input_request_metadata(remote.request),
            "request_text": remote.request_text,
            "task_id": remote.task_id,
            "context_id": remote.context_id,
            "prior_message_id": remote.prior_message_id,
            "prior_content": remote.prior_content,
            "ttl_seconds": remote.ttl_seconds,
            "input_cycle_count": remote.input_cycle_count,
            "interaction_counts": dict(
                checkpoint.interaction_fingerprint_counts
            ),
        },
    )


def decode_a2a_tool_continuation_observation(
    observations: tuple[Mapping[str, JsonValue], ...],
) -> A2AToolContinuationCheckpoint | None:
    """Decode one A2A checkpoint or return None for another kind."""
    if len(observations) != 1 or observations[0].get("kind") != _KIND:
        return None
    payload = observations[0]
    if set(payload) != _FIELDS or payload["version"] != _VERSION:
        _invalid("continuation.observations[0]", "has invalid fields")
    raw_counts = payload["interaction_counts"]
    if not isinstance(raw_counts, Mapping) or any(
        not isinstance(key, str) or type(value) is not int or value < 1
        for key, value in raw_counts.items()
    ):
        _invalid("continuation.observations[0]", "has invalid counts")
    counts = tuple(sorted(cast(Mapping[str, int], raw_counts).items()))
    arguments = _json_object(
        payload["call_arguments"],
        "continuation.observations[0].call_arguments",
    )
    return A2AToolContinuationCheckpoint(
        call_id=cast(str, payload["call_id"]),
        canonical_name=cast(str, payload["canonical_name"]),
        provider_name=cast(str, payload["provider_name"]),
        provider_name_encoded=cast(
            bool,
            payload["provider_name_encoded"],
        ),
        arguments=arguments,
        remote=A2ARemoteInputContinuation(
            request=decode_a2a_input_request_metadata(payload["request"]),
            request_text=cast(str, payload["request_text"]),
            task_id=cast(str, payload["task_id"]),
            context_id=cast(str, payload["context_id"]),
            prior_message_id=cast(str, payload["prior_message_id"]),
            prior_content=cast(tuple[str, ...], payload["prior_content"]),
            ttl_seconds=cast(int, payload["ttl_seconds"]),
            input_cycle_count=cast(int, payload["input_cycle_count"]),
        ),
        interaction_fingerprint_counts=counts,
    )


def project_a2a_remote_resolution(
    request: InputRequest,
    remote: A2ARemoteInputContinuation,
) -> InputResolution | None:
    """Project one authorized local terminal result to its remote identity."""
    return project_a2a_input_resolution(request.resolution, remote.request)


def project_a2a_input_resolution(
    resolution: InputResolution | None,
    request: A2AInputRequestMetadata,
) -> InputResolution | None:
    """Project one terminal result to an A2A request identity."""
    if isinstance(resolution, TimedOutResolution | UnavailableResolution):
        return None
    if not isinstance(
        resolution,
        AnsweredResolution | DeclinedResolution | CancelledResolution,
    ):
        raise InputValidationError(
            InputErrorCode.ILLEGAL_TRANSITION,
            "resume.interaction.resolution",
            "A2A continuation requires answer, decline, or request cancel",
        )
    projected = replace(resolution, request_id=request.request_id)
    decode_a2a_input_resolution_metadata(
        encode_a2a_input_resolution_metadata(projected),
        request=request,
        resolved_at=projected.resolved_at,
    )
    return projected


def bound_a2a_prior_content(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return remote text only when every semantic byte fits safely."""
    encoded = tuple(
        value.encode() if isinstance(value, str) else b"" for value in values
    )
    if (
        len(values) > _MAX_PRIOR_CONTENT_ITEMS
        or any(
            not raw or len(raw) > _MAX_PRIOR_CONTENT_ITEM_BYTES
            for raw in encoded
        )
        or sum(map(len, encoded)) > _MAX_PRIOR_CONTENT_BYTES
    ):
        raise RuntimeError("A2A prior content exceeds safe bounds")
    return values


def _json_object(
    value: object,
    path: str,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        _invalid(path, "must be an object")
    return _freeze_snapshot_object(cast(Mapping[str, object], value), path)


def _invalid(path: str, message: str) -> NoReturn:
    raise InputValidationError(InputErrorCode.SNAPSHOT_INVALID, path, message)
