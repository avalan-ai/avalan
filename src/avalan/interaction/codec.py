"""Encode and decode canonical interactions as strict JSON values."""

from ..types import JsonObject, JsonValue, MutableJsonValue
from .entities import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancellationScope,
    CancelledResolution,
    CapabilityRevision,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    FreeFormOther,
    InputAnswer,
    InputAnsweredResult,
    InputCancelledResult,
    InputContinuationOutcome,
    InputDeclinedResult,
    InputModelResult,
    InputQuestion,
    InputRequest,
    InputRequestId,
    InputRequiredResult,
    InputResolution,
    InputResultKind,
    InputTimedOutResult,
    InputUnavailableResult,
    InteractionClass,
    InteractionSnapshot,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    ParticipantId,
    PresentationHint,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    QuestionId,
    QuestionType,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    ResumeInputContinuation,
    RunId,
    SelectedChoice,
    SelectionValidationConstraints,
    SelectionValue,
    SelectionValueType,
    SessionId,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    SupersededResolution,
    TaskId,
    TenantId,
    TerminateInputContinuation,
    TextAnswer,
    TextQuestion,
    TextValidationConstraints,
    TimedOutResolution,
    TurnId,
    UnavailableResolution,
    UserId,
    _is_input_answer_variant,
    _is_input_continuation_outcome_variant,
    _is_input_model_result_variant,
    _is_input_question_variant,
    _is_input_resolution_variant,
    _is_selection_value_variant,
    _validate_snapshot_kind,
)
from .error import (
    InputCodecError,
    InputErrorCode,
    InputSnapshotError,
    InputValidationError,
)
from .validation import validate_finite_number

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from typing import NoReturn, TypeVar, cast

_SNAPSHOT_VERSION = 1
_MAX_SNAPSHOT_UTF8_BYTES = 1_048_576
_PROHIBITED_QUESTION_TAGS = frozenset(
    {
        "access_token",
        "api_key",
        "authentication",
        "authentication_challenge",
        "bank_credential",
        "mfa",
        "mfa_challenge",
        "password",
        "payment",
        "payment_card",
        "private_key",
        "refresh_token",
        "token",
        "url_auth",
        "url_authentication",
    }
)

EnumValue = TypeVar("EnumValue", bound=StrEnum)
NewValue = TypeVar("NewValue")


def encode_input_request(request: InputRequest) -> JsonObject:
    """Encode a validated input request into JSON-safe values."""
    if type(request) is not InputRequest:
        _codec_error("request", "value must be an input request")
    questions: list[MutableJsonValue] = [
        encode_input_question(question) for question in request.questions
    ]
    result: JsonObject = {
        "interaction_class": request.interaction_class.value,
        "request_id": str(request.request_id),
        "continuation_id": str(request.continuation_id),
        "origin": encode_execution_origin(request.origin),
        "mode": request.mode.value,
        "required": request.required,
        "reason": request.reason,
        "questions": questions,
        "created_at": _encode_datetime(request.created_at),
        "continuation_ttl_seconds": request.continuation_ttl_seconds,
        "state": request.state.value,
        "state_revision": int(request.state_revision),
    }
    if request.advisory_wait_seconds is not None:
        result["advisory_wait_seconds"] = request.advisory_wait_seconds
    if request.advisory_deadline is not None:
        result["advisory_deadline"] = _encode_datetime(
            request.advisory_deadline
        )
    if request.resolution is not None:
        result["resolution"] = encode_input_resolution(request.resolution)
    return result


def decode_input_request(value: object) -> InputRequest:
    """Decode and validate a canonical input request."""
    item = _object(value, "request")
    _keys(
        item,
        required={
            "interaction_class",
            "request_id",
            "continuation_id",
            "origin",
            "mode",
            "required",
            "reason",
            "questions",
            "created_at",
            "continuation_ttl_seconds",
            "state",
            "state_revision",
        },
        optional={
            "advisory_wait_seconds",
            "advisory_deadline",
            "resolution",
        },
        path="request",
    )
    interaction_class = _enum(
        item["interaction_class"],
        InteractionClass,
        "request.interaction_class",
    )
    if interaction_class is not InteractionClass.TASK_INPUT:
        raise InputCodecError(
            InputErrorCode.PROHIBITED_INPUT,
            "request.interaction_class",
            "only task-input requests are accepted",
        )
    mode = _enum(item["mode"], RequirementMode, "request.mode")
    required = _bool(item["required"], "request.required")
    if required != (mode is RequirementMode.REQUIRED):
        _codec_error("request.required", "mode and required flag disagree")
    raw_questions = _array(item["questions"], "request.questions")
    questions = tuple(
        decode_input_question(question) for question in raw_questions
    )
    raw_resolution = item.get("resolution")
    resolution = (
        None
        if raw_resolution is None
        else decode_input_resolution(raw_resolution)
    )
    return InputRequest(
        request_id=InputRequestId(
            _string(item["request_id"], "request.request_id")
        ),
        continuation_id=ContinuationId(
            _string(item["continuation_id"], "request.continuation_id")
        ),
        origin=decode_execution_origin(item["origin"]),
        mode=mode,
        reason=_string(item["reason"], "request.reason"),
        questions=questions,
        created_at=_datetime(item["created_at"], "request.created_at"),
        continuation_ttl_seconds=_integer(
            item["continuation_ttl_seconds"],
            "request.continuation_ttl_seconds",
        ),
        advisory_wait_seconds=_optional_integer(
            item.get("advisory_wait_seconds"),
            "request.advisory_wait_seconds",
        ),
        advisory_deadline=(
            _datetime(
                item["advisory_deadline"],
                "request.advisory_deadline",
            )
            if "advisory_deadline" in item
            else None
        ),
        state=_enum(item["state"], RequestState, "request.state"),
        state_revision=StateRevision(
            _integer(item["state_revision"], "request.state_revision")
        ),
        resolution=resolution,
    )


def encode_execution_origin(origin: ExecutionOrigin) -> JsonObject:
    """Encode complete logical-execution correlation."""
    if not isinstance(origin, ExecutionOrigin):
        _codec_error("origin", "value must be an execution origin")
    result: JsonObject = {
        "run_id": str(origin.run_id),
        "turn_id": str(origin.turn_id),
        "agent_id": str(origin.agent_id),
        "branch_id": str(origin.branch_id),
        "model_call_id": str(origin.model_call_id),
        "stream_session_id": str(origin.stream_session_id),
        "definition": _encode_definition(origin.definition),
        "principal": _encode_principal(origin.principal),
    }
    if origin.task_id is not None:
        result["task_id"] = str(origin.task_id)
    if origin.parent_branch_id is not None:
        result["parent_branch_id"] = str(origin.parent_branch_id)
    return result


def decode_execution_origin(value: object) -> ExecutionOrigin:
    """Decode and validate complete logical-execution correlation."""
    item = _object(value, "origin")
    _keys(
        item,
        required={
            "run_id",
            "turn_id",
            "agent_id",
            "branch_id",
            "model_call_id",
            "stream_session_id",
            "definition",
            "principal",
        },
        optional={"task_id", "parent_branch_id"},
        path="origin",
    )
    return ExecutionOrigin(
        run_id=RunId(_string(item["run_id"], "origin.run_id")),
        turn_id=TurnId(_string(item["turn_id"], "origin.turn_id")),
        task_id=_optional_new_type(
            item.get("task_id"),
            "origin.task_id",
            TaskId,
        ),
        agent_id=AgentId(_string(item["agent_id"], "origin.agent_id")),
        branch_id=BranchId(_string(item["branch_id"], "origin.branch_id")),
        parent_branch_id=_optional_new_type(
            item.get("parent_branch_id"),
            "origin.parent_branch_id",
            BranchId,
        ),
        model_call_id=ModelCallId(
            _string(item["model_call_id"], "origin.model_call_id")
        ),
        stream_session_id=StreamSessionId(
            _string(item["stream_session_id"], "origin.stream_session_id")
        ),
        definition=_decode_definition(item["definition"]),
        principal=_decode_principal(item["principal"]),
    )


def encode_input_question(question: InputQuestion) -> JsonObject:
    """Encode one typed semantic question."""
    if not _is_input_question_variant(question):
        _codec_error(
            "question",
            "value must be a supported input question variant",
        )
    result: JsonObject = {
        "question_id": str(question.question_id),
        "kind": question.kind.value,
        "prompt": question.prompt,
        "required": question.required,
        "choices": [],
        "allow_other": False,
    }
    if question.header is not None:
        result["header"] = question.header
    if question.help_text is not None:
        result["help"] = question.help_text
    if question.presentation_hint is not None:
        result["presentation_hint"] = question.presentation_hint.value
    if type(question) is ConfirmationQuestion:
        if question.default_value is not None:
            result["default_value"] = question.default_value
    elif type(question) is TextQuestion:
        _encode_text_question_fields(result, question)
    elif type(question) is MultilineTextQuestion:
        _encode_text_question_fields(result, question)
    elif type(question) is SingleSelectionQuestion:
        _encode_selection_question_fields(result, question)
        if question.default_value is not None:
            result["default_value"] = str(question.default_value)
    else:
        assert type(question) is MultipleSelectionQuestion
        _encode_selection_question_fields(result, question)
        if question.default_value is not None:
            result["default_value"] = [
                str(value) for value in question.default_value
            ]
        result["constraints"] = {
            "minimum": question.constraints.minimum,
            "maximum": question.constraints.maximum,
        }
    return result


def _encode_text_question_fields(
    result: JsonObject,
    question: TextQuestion | MultilineTextQuestion,
) -> None:
    if question.default_value is not None:
        result["default_value"] = question.default_value
    result["constraints"] = {
        "minimum_length": question.constraints.minimum_length,
        "maximum_length": question.constraints.maximum_length,
    }


def _encode_selection_question_fields(
    result: JsonObject,
    question: SingleSelectionQuestion | MultipleSelectionQuestion,
) -> None:
    result["choices"] = [_encode_choice(choice) for choice in question.choices]
    result["allow_other"] = question.allow_other
    if question.recommended_choice is not None:
        result["recommended_choice"] = str(question.recommended_choice)


def decode_input_question(value: object) -> InputQuestion:
    """Decode and validate one typed semantic question."""
    item = _object(value, "question")
    _reject_prohibited_question_tag(item)
    kind = _enum(item.get("kind"), QuestionType, "question.kind")
    optional_fields = {
        "header",
        "help",
        "default_value",
        "presentation_hint",
    }
    if kind in {QuestionType.TEXT, QuestionType.MULTILINE_TEXT}:
        optional_fields.add("constraints")
    elif kind is QuestionType.SINGLE_SELECTION:
        optional_fields.add("recommended_choice")
    elif kind is QuestionType.MULTIPLE_SELECTION:
        optional_fields.update({"constraints", "recommended_choice"})
    _keys(
        item,
        required={
            "question_id",
            "kind",
            "prompt",
            "required",
            "choices",
            "allow_other",
        },
        optional=optional_fields,
        path="question",
    )
    _reject_explicit_nulls(item, optional_fields, "question")
    question_id = QuestionId(
        _string(item["question_id"], "question.question_id")
    )
    prompt = _string(item["prompt"], "question.prompt")
    required = _bool(item["required"], "question.required")
    header = _optional_string(item.get("header"), "question.header")
    help_text = _optional_string(item.get("help"), "question.help")
    presentation_hint = _optional_enum(
        item.get("presentation_hint"),
        PresentationHint,
        "question.presentation_hint",
    )
    raw_choices = _array(item["choices"], "question.choices")
    allow_other = _bool(item["allow_other"], "question.allow_other")
    if kind is QuestionType.CONFIRMATION:
        _require_empty_selection_fields(raw_choices, allow_other, item)
        return ConfirmationQuestion(
            question_id=question_id,
            prompt=prompt,
            required=required,
            header=header,
            help_text=help_text,
            presentation_hint=presentation_hint,
            default_value=_optional_bool(
                item.get("default_value"),
                "question.default_value",
            ),
        )
    if kind is QuestionType.TEXT:
        _require_empty_selection_fields(raw_choices, allow_other, item)
        return TextQuestion(
            question_id=question_id,
            prompt=prompt,
            required=required,
            header=header,
            help_text=help_text,
            presentation_hint=presentation_hint,
            default_value=_optional_string(
                item.get("default_value"),
                "question.default_value",
            ),
            constraints=_decode_text_constraints(
                item.get("constraints"),
                maximum=4_096,
            ),
        )
    if kind is QuestionType.MULTILINE_TEXT:
        _require_empty_selection_fields(raw_choices, allow_other, item)
        return MultilineTextQuestion(
            question_id=question_id,
            prompt=prompt,
            required=required,
            header=header,
            help_text=help_text,
            presentation_hint=presentation_hint,
            default_value=_optional_string(
                item.get("default_value"),
                "question.default_value",
            ),
            constraints=_decode_text_constraints(
                item.get("constraints"),
                maximum=65_536,
            ),
        )
    choices = tuple(_decode_choice(choice) for choice in raw_choices)
    recommended_choice = _optional_new_type(
        item.get("recommended_choice"),
        "question.recommended_choice",
        ChoiceValue,
    )
    if kind is QuestionType.SINGLE_SELECTION:
        return SingleSelectionQuestion(
            question_id=question_id,
            prompt=prompt,
            required=required,
            header=header,
            help_text=help_text,
            presentation_hint=presentation_hint,
            choices=choices,
            allow_other=allow_other,
            recommended_choice=recommended_choice,
            default_value=_optional_new_type(
                item.get("default_value"),
                "question.default_value",
                ChoiceValue,
            ),
        )
    raw_default = item.get("default_value")
    default_value = (
        None
        if raw_default is None
        else tuple(
            ChoiceValue(_string(entry, "question.default_value"))
            for entry in _array(raw_default, "question.default_value")
        )
    )
    return MultipleSelectionQuestion(
        question_id=question_id,
        prompt=prompt,
        required=required,
        header=header,
        help_text=help_text,
        presentation_hint=presentation_hint,
        choices=choices,
        allow_other=allow_other,
        recommended_choice=recommended_choice,
        default_value=default_value,
        constraints=_decode_selection_constraints(
            item.get("constraints"),
            maximum=min(20, len(choices) + int(allow_other)),
        ),
    )


def encode_input_answer(answer: InputAnswer) -> JsonObject:
    """Encode one typed answer keyed by stable question identifier."""
    if not _is_input_answer_variant(answer):
        _codec_error(
            "answer",
            "value must be a supported input answer variant",
        )
    result: JsonObject = {
        "question_id": str(answer.question_id),
        "kind": answer.question_type.value,
        "provenance": answer.provenance.value,
    }
    if type(answer) is ConfirmationAnswer:
        result["value"] = answer.value
    elif type(answer) is TextAnswer:
        result["value"] = answer.value
    elif type(answer) is MultilineTextAnswer:
        result["value"] = answer.value
    elif type(answer) is SingleSelectionAnswer:
        result["value"] = _encode_selection_value(answer.value)
    else:
        assert type(answer) is MultipleSelectionAnswer
        result["values"] = [
            _encode_selection_value(value) for value in answer.values
        ]
    return result


def decode_input_answer(value: object) -> InputAnswer:
    """Decode and validate one typed answer."""
    item = _object(value, "answer")
    kind = _enum(item.get("kind"), QuestionType, "answer.kind")
    expected = {"question_id", "kind", "provenance"}
    value_key = (
        "values" if kind is QuestionType.MULTIPLE_SELECTION else "value"
    )
    _keys(item, required=expected | {value_key}, optional=set(), path="answer")
    question_id = QuestionId(
        _string(item["question_id"], "answer.question_id")
    )
    provenance = _enum(
        item["provenance"],
        AnswerProvenance,
        "answer.provenance",
    )
    if kind is QuestionType.CONFIRMATION:
        return ConfirmationAnswer(
            question_id=question_id,
            provenance=provenance,
            value=_bool(item["value"], "answer.value"),
        )
    if kind is QuestionType.TEXT:
        return TextAnswer(
            question_id=question_id,
            provenance=provenance,
            value=_string(item["value"], "answer.value"),
        )
    if kind is QuestionType.MULTILINE_TEXT:
        return MultilineTextAnswer(
            question_id=question_id,
            provenance=provenance,
            value=_string(item["value"], "answer.value"),
        )
    if kind is QuestionType.SINGLE_SELECTION:
        return SingleSelectionAnswer(
            question_id=question_id,
            provenance=provenance,
            value=_decode_selection_value(item["value"]),
        )
    return MultipleSelectionAnswer(
        question_id=question_id,
        provenance=provenance,
        values=tuple(
            _decode_selection_value(entry)
            for entry in _array(item["values"], "answer.values")
        ),
    )


def encode_input_resolution(resolution: InputResolution) -> JsonObject:
    """Encode one terminal resolution with explicit provenance."""
    if not _is_input_resolution_variant(resolution):
        _codec_error(
            "resolution",
            "value must be a supported input resolution variant",
        )
    result: JsonObject = {
        "request_id": str(resolution.request_id),
        "status": resolution.status.value,
        "provenance": resolution.provenance.value,
        "resolved_at": _encode_datetime(resolution.resolved_at),
    }
    if type(resolution) is AnsweredResolution:
        result["answers"] = [
            encode_input_answer(answer) for answer in resolution.answers
        ]
    elif type(resolution) is CancelledResolution:
        result["scope"] = resolution.scope.value
    return result


def decode_input_resolution(value: object) -> InputResolution:
    """Decode and validate one terminal resolution."""
    item = _object(value, "resolution")
    status = _enum(
        item.get("status"),
        ResolutionStatus,
        "resolution.status",
    )
    required = {"request_id", "status", "provenance", "resolved_at"}
    if status is ResolutionStatus.ANSWERED:
        optional = {"answers"}
    elif status is ResolutionStatus.CANCELLED:
        required.add("scope")
        optional = set()
    else:
        optional = set()
    _keys(item, required=required, optional=optional, path="resolution")
    request_id = InputRequestId(
        _string(item["request_id"], "resolution.request_id")
    )
    provenance = _enum(
        item["provenance"],
        AnswerProvenance,
        "resolution.provenance",
    )
    resolved_at = _datetime(
        item["resolved_at"],
        "resolution.resolved_at",
    )
    if status is ResolutionStatus.ANSWERED:
        return AnsweredResolution(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
            answers=tuple(
                decode_input_answer(answer)
                for answer in _array(
                    item.get("answers"),
                    "resolution.answers",
                )
            ),
        )
    if status is ResolutionStatus.DECLINED:
        return DeclinedResolution(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
        )
    if status is ResolutionStatus.CANCELLED:
        return CancelledResolution(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
            scope=_enum(
                item["scope"],
                CancellationScope,
                "resolution.scope",
            ),
        )
    if status is ResolutionStatus.TIMED_OUT:
        return TimedOutResolution(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
        )
    if status is ResolutionStatus.UNAVAILABLE:
        return UnavailableResolution(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
        )
    if status is ResolutionStatus.EXPIRED:
        return ExpiredResolution(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
        )
    return SupersededResolution(
        request_id=request_id,
        provenance=provenance,
        resolved_at=resolved_at,
    )


def encode_input_required_result(result: InputRequiredResult) -> JsonObject:
    """Encode a segment-level input-required result."""
    if type(result) is not InputRequiredResult:
        _codec_error("result", "value must be an input-required result")
    return {
        "kind": result.kind.value,
        "request_id": str(result.request_id),
        "continuation_id": str(result.continuation_id),
        "detached_resumption_available": result.detached_resumption_available,
    }


def decode_input_required_result(value: object) -> InputRequiredResult:
    """Decode and validate a segment-level input-required result."""
    item = _object(value, "result")
    _keys(
        item,
        required={
            "kind",
            "request_id",
            "continuation_id",
            "detached_resumption_available",
        },
        optional=set(),
        path="result",
    )
    kind = _enum(item["kind"], InputResultKind, "result.kind")
    if kind is not InputResultKind.INPUT_REQUIRED:
        _codec_error("result.kind", "result is not input-required")
    return InputRequiredResult(
        request_id=InputRequestId(
            _string(item["request_id"], "result.request_id")
        ),
        continuation_id=ContinuationId(
            _string(item["continuation_id"], "result.continuation_id")
        ),
        detached_resumption_available=_bool(
            item["detached_resumption_available"],
            "result.detached_resumption_available",
        ),
    )


def encode_input_model_result(result: InputModelResult) -> JsonObject:
    """Encode an explicit result returned to the pending model request."""
    if not _is_input_model_result_variant(result):
        _codec_error(
            "result",
            "value must be a supported input model result variant",
        )
    encoded: JsonObject = {
        "kind": result.kind.value,
        "request_id": str(result.request_id),
        "provenance": result.provenance.value,
        "resolved_at": _encode_datetime(result.resolved_at),
    }
    if type(result) is InputAnsweredResult:
        encoded["answers"] = [
            encode_input_answer(answer) for answer in result.answers
        ]
    return encoded


def decode_input_model_result(value: object) -> InputModelResult:
    """Decode an explicit result returned to the pending model request."""
    item = _object(value, "result")
    kind = _enum(item.get("kind"), InputResultKind, "result.kind")
    if kind is InputResultKind.INPUT_REQUIRED:
        _codec_error("result.kind", "model result cannot be input-required")
    required = {"kind", "request_id", "provenance", "resolved_at"}
    optional = {"answers"} if kind is InputResultKind.ANSWERED else set()
    _keys(item, required=required, optional=optional, path="result")
    request_id = InputRequestId(
        _string(item["request_id"], "result.request_id")
    )
    provenance = _enum(
        item["provenance"],
        AnswerProvenance,
        "result.provenance",
    )
    resolved_at = _datetime(item["resolved_at"], "result.resolved_at")
    if kind is InputResultKind.ANSWERED:
        return InputAnsweredResult(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
            answers=tuple(
                decode_input_answer(answer)
                for answer in _array(item.get("answers"), "result.answers")
            ),
        )
    if kind is InputResultKind.DECLINED:
        return InputDeclinedResult(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
        )
    if kind is InputResultKind.CANCELLED:
        return InputCancelledResult(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
        )
    if kind is InputResultKind.TIMED_OUT:
        return InputTimedOutResult(
            request_id=request_id,
            provenance=provenance,
            resolved_at=resolved_at,
        )
    return InputUnavailableResult(
        request_id=request_id,
        provenance=provenance,
        resolved_at=resolved_at,
    )


def encode_continuation_outcome(
    outcome: InputContinuationOutcome,
) -> JsonObject:
    """Encode a resume-or-terminate continuation decision."""
    if not _is_input_continuation_outcome_variant(outcome):
        _codec_error(
            "outcome",
            "value must be a supported continuation outcome variant",
        )
    if type(outcome) is ResumeInputContinuation:
        return {
            "disposition": outcome.disposition.value,
            "request_id": str(outcome.request_id),
            "result": encode_input_model_result(outcome.result),
        }
    assert type(outcome) is TerminateInputContinuation
    return {
        "disposition": outcome.disposition.value,
        "request_id": str(outcome.request_id),
        "status": outcome.status.value,
    }


def encode_interaction_snapshot(snapshot: InteractionSnapshot) -> str:
    """Encode a versioned snapshot as canonical strict JSON."""
    if not isinstance(snapshot, InteractionSnapshot):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "snapshot",
            "value must be an interaction snapshot",
        )
    payload: JsonObject = {
        "version": snapshot.version,
        "request": encode_input_request(snapshot.request),
        "content_sha256": interaction_snapshot_digest(snapshot),
    }
    encoded = _canonical_json_text(payload, "snapshot")
    if len(encoded.encode("utf-8")) > _MAX_SNAPSHOT_UTF8_BYTES:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "snapshot",
            "snapshot exceeds its byte bound",
        )
    return encoded


def canonical_interaction_snapshot_bytes(
    snapshot: InteractionSnapshot,
) -> bytes:
    """Return the documented interaction-snapshot hash preimage."""
    if not isinstance(snapshot, InteractionSnapshot):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "snapshot",
            "value must be an interaction snapshot",
        )
    preimage: JsonObject = {
        "version": snapshot.version,
        "request": encode_input_request(snapshot.request),
    }
    return _canonical_json_text(preimage, "snapshot").encode("utf-8")


def interaction_snapshot_digest(snapshot: InteractionSnapshot) -> str:
    """Return the SHA-256 digest of canonical snapshot preimage bytes."""
    return sha256(canonical_interaction_snapshot_bytes(snapshot)).hexdigest()


def decode_interaction_snapshot(value: str | bytes) -> InteractionSnapshot:
    """Decode strict canonical JSON into a versioned snapshot."""
    text = _snapshot_text(value, "snapshot")
    raw = _strict_json(text, "snapshot")
    item = _object(raw, "snapshot")
    _keys(
        item,
        required={"version", "request", "content_sha256"},
        optional=set(),
        path="snapshot",
        snapshot=True,
    )
    _require_canonical_snapshot_text(text, item, "snapshot")
    version = _integer(item["version"], "snapshot.version")
    if version != _SNAPSHOT_VERSION:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_UNSUPPORTED,
            "snapshot.version",
            "snapshot version is unsupported",
        )
    digest = _sha256_string(
        item["content_sha256"],
        "snapshot.content_sha256",
    )
    try:
        request = decode_input_request(item["request"])
    except InputValidationError as exc:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            exc.path,
            exc.safe_message,
        ) from exc
    snapshot = InteractionSnapshot(version=version, request=request)
    if digest != interaction_snapshot_digest(snapshot):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "snapshot.content_sha256",
            "snapshot content hash does not match its canonical preimage",
        )
    return snapshot


def encode_continuation_snapshot(snapshot: ContinuationSnapshot) -> str:
    """Encode a provider continuation snapshot as canonical strict JSON."""
    if not isinstance(snapshot, ContinuationSnapshot):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation_snapshot",
            "value must be a continuation snapshot",
        )
    binding = snapshot.revision_binding
    payload: JsonObject = {
        "version": snapshot.version,
        "snapshot_kind": snapshot.snapshot_kind,
        "provider_family": str(binding.provider_family),
        "model_id": str(binding.model_id),
        "provider_config_revision": str(binding.provider_config_revision),
        "model_config_revision": str(binding.model_config_revision),
        "capability_revision": str(binding.capability_revision),
        "model_call_id": str(snapshot.model_call_id),
        "provider_idempotency_key": str(snapshot.provider_idempotency_key),
        "payload": _mutable_json_mapping(snapshot.payload),
        "content_sha256": continuation_snapshot_digest(snapshot),
    }
    encoded = _canonical_json_text(payload, "continuation_snapshot")
    if len(encoded.encode("utf-8")) > _MAX_SNAPSHOT_UTF8_BYTES:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation_snapshot",
            "snapshot exceeds its byte bound",
        )
    return encoded


def canonical_continuation_snapshot_bytes(
    snapshot: ContinuationSnapshot,
) -> bytes:
    """Return the documented continuation-snapshot hash preimage."""
    if not isinstance(snapshot, ContinuationSnapshot):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation_snapshot",
            "value must be a continuation snapshot",
        )
    binding = snapshot.revision_binding
    preimage: JsonObject = {
        "version": snapshot.version,
        "snapshot_kind": snapshot.snapshot_kind,
        "provider_family": str(binding.provider_family),
        "model_id": str(binding.model_id),
        "provider_config_revision": str(binding.provider_config_revision),
        "model_config_revision": str(binding.model_config_revision),
        "capability_revision": str(binding.capability_revision),
        "model_call_id": str(snapshot.model_call_id),
        "provider_idempotency_key": str(snapshot.provider_idempotency_key),
        "payload": _mutable_json_mapping(snapshot.payload),
    }
    return _canonical_json_text(
        preimage,
        "continuation_snapshot",
    ).encode("utf-8")


def continuation_snapshot_digest(snapshot: ContinuationSnapshot) -> str:
    """Return the SHA-256 digest of canonical continuation bytes."""
    return sha256(canonical_continuation_snapshot_bytes(snapshot)).hexdigest()


def decode_continuation_snapshot(
    value: str | bytes,
    *,
    expected_binding: ContinuationRevisionBinding,
) -> ContinuationSnapshot:
    """Decode a snapshot and reject provider identity or revision drift."""
    text = _snapshot_text(value, "continuation_snapshot")
    raw = _strict_json(text, "continuation_snapshot")
    item = _object(raw, "continuation_snapshot")
    fields = {
        "version",
        "snapshot_kind",
        "provider_family",
        "model_id",
        "provider_config_revision",
        "model_config_revision",
        "capability_revision",
        "model_call_id",
        "provider_idempotency_key",
        "payload",
        "content_sha256",
    }
    _keys(
        item,
        required=fields,
        optional=set(),
        path="continuation_snapshot",
        snapshot=True,
    )
    _require_canonical_snapshot_text(text, item, "continuation_snapshot")
    version = _integer(item["version"], "continuation_snapshot.version")
    if version != _SNAPSHOT_VERSION:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_UNSUPPORTED,
            "continuation_snapshot.version",
            "snapshot version is unsupported",
        )
    try:
        snapshot_kind = _validate_snapshot_kind(item["snapshot_kind"])
    except InputValidationError as exc:
        raise InputSnapshotError(exc.code, exc.path, exc.safe_message) from exc
    digest = _sha256_string(
        item["content_sha256"],
        "continuation_snapshot.content_sha256",
    )
    if digest != _raw_continuation_snapshot_digest(item):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation_snapshot.content_sha256",
            "snapshot content hash does not match its canonical preimage",
        )
    try:
        binding = ContinuationRevisionBinding(
            provider_family=ProviderFamilyName(
                _string(
                    item["provider_family"],
                    "continuation_snapshot.provider_family",
                )
            ),
            model_id=ModelId(
                _string(item["model_id"], "continuation_snapshot.model_id")
            ),
            provider_config_revision=ProviderConfigRevision(
                _string(
                    item["provider_config_revision"],
                    "continuation_snapshot.provider_config_revision",
                )
            ),
            model_config_revision=ModelConfigRevision(
                _string(
                    item["model_config_revision"],
                    "continuation_snapshot.model_config_revision",
                )
            ),
            capability_revision=CapabilityRevision(
                _string(
                    item["capability_revision"],
                    "continuation_snapshot.capability_revision",
                )
            ),
        )
    except InputValidationError as exc:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            exc.path,
            exc.safe_message,
        ) from exc
    _validate_revision_binding(binding, expected_binding)
    try:
        snapshot = ContinuationSnapshot(
            version=version,
            snapshot_kind=snapshot_kind,
            revision_binding=binding,
            model_call_id=ModelCallId(
                _string(
                    item["model_call_id"],
                    "continuation_snapshot.model_call_id",
                )
            ),
            provider_idempotency_key=ProviderIdempotencyKey(
                _string(
                    item["provider_idempotency_key"],
                    "continuation_snapshot.provider_idempotency_key",
                )
            ),
            payload=_immutable_json_mapping(
                item["payload"],
                "continuation_snapshot.payload",
            ),
        )
    except InputValidationError as exc:
        raise InputSnapshotError(exc.code, exc.path, exc.safe_message) from exc
    return snapshot


def canonical_resolution_digest(resolution: InputResolution) -> str:
    """Return the canonical semantic digest of a typed resolution.

    Exclude the trusted commit timestamp so transport retries can compare the
    submitted meaning without treating a new observation time as new content.
    """
    payload = encode_input_resolution(resolution)
    del payload["resolved_at"]
    if type(resolution) is AnsweredResolution:
        encoded_answers = payload["answers"]
        assert isinstance(encoded_answers, list)
        payload["answers"] = sorted(
            encoded_answers,
            key=_encoded_answer_question_id,
        )
    encoded = dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _encoded_answer_question_id(value: MutableJsonValue) -> str:
    """Return one encoded answer identifier for semantic ordering."""
    assert isinstance(value, dict)
    question_id = value["question_id"]
    assert isinstance(question_id, str)
    return question_id


def semantic_request_fingerprint(request: InputRequest) -> str:
    """Return a digest excluding runtime identity and display-only hints."""
    if type(request) is not InputRequest:
        _codec_error("request", "value must be an input request")
    questions: list[MutableJsonValue] = []
    for question in request.questions:
        if not _is_input_question_variant(question):
            _codec_error(
                "question",
                "value must be a supported input question variant",
            )
        item: JsonObject = {
            "question_id": str(question.question_id),
            "kind": question.kind.value,
            "prompt": question.prompt,
            "required": question.required,
        }
        if type(question) is TextQuestion:
            _encode_semantic_text_constraints(item, question)
        elif type(question) is MultilineTextQuestion:
            _encode_semantic_text_constraints(item, question)
        elif type(question) is SingleSelectionQuestion:
            _encode_semantic_selection_fields(item, question)
        elif type(question) is MultipleSelectionQuestion:
            _encode_semantic_selection_fields(item, question)
            item["constraints"] = {
                "minimum": question.constraints.minimum,
                "maximum": question.constraints.maximum,
            }
        questions.append(item)
    payload: JsonObject = {
        "required": request.required,
        "reason": request.reason,
        "questions": questions,
    }
    encoded = dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _encode_semantic_text_constraints(
    item: JsonObject,
    question: TextQuestion | MultilineTextQuestion,
) -> None:
    item["constraints"] = {
        "minimum_length": question.constraints.minimum_length,
        "maximum_length": question.constraints.maximum_length,
    }


def _encode_semantic_selection_fields(
    item: JsonObject,
    question: SingleSelectionQuestion | MultipleSelectionQuestion,
) -> None:
    item["choice_values"] = [str(choice.value) for choice in question.choices]
    item["allow_other"] = question.allow_other


def _encode_definition(value: ExecutionDefinitionRef) -> JsonObject:
    return {
        "agent_definition_locator": value.agent_definition_locator,
        "agent_definition_revision": value.agent_definition_revision,
        "operation_id": value.operation_id,
        "operation_index": value.operation_index,
        "model_config_reference": value.model_config_reference,
        "tool_revision": value.tool_revision,
        "capability_revision": value.capability_revision,
    }


def _decode_definition(value: object) -> ExecutionDefinitionRef:
    item = _object(value, "origin.definition")
    keys = {
        "agent_definition_locator",
        "agent_definition_revision",
        "operation_id",
        "operation_index",
        "model_config_reference",
        "tool_revision",
        "capability_revision",
    }
    _keys(item, required=keys, optional=set(), path="origin.definition")
    return ExecutionDefinitionRef(
        agent_definition_locator=_string(
            item["agent_definition_locator"],
            "origin.definition.agent_definition_locator",
        ),
        agent_definition_revision=_string(
            item["agent_definition_revision"],
            "origin.definition.agent_definition_revision",
        ),
        operation_id=_string(
            item["operation_id"],
            "origin.definition.operation_id",
        ),
        operation_index=_integer(
            item["operation_index"],
            "origin.definition.operation_index",
        ),
        model_config_reference=_string(
            item["model_config_reference"],
            "origin.definition.model_config_reference",
        ),
        tool_revision=_string(
            item["tool_revision"],
            "origin.definition.tool_revision",
        ),
        capability_revision=_string(
            item["capability_revision"],
            "origin.definition.capability_revision",
        ),
    )


def _encode_principal(value: PrincipalScope) -> JsonObject:
    result: JsonObject = {}
    for name in ("user_id", "tenant_id", "participant_id", "session_id"):
        item = getattr(value, name)
        if item is not None:
            result[name] = str(item)
    return result


def _decode_principal(value: object) -> PrincipalScope:
    item = _object(value, "origin.principal")
    fields = {"user_id", "tenant_id", "participant_id", "session_id"}
    _keys(item, required=set(), optional=fields, path="origin.principal")
    return PrincipalScope(
        user_id=_optional_new_type(
            item.get("user_id"),
            "origin.principal.user_id",
            UserId,
        ),
        tenant_id=_optional_new_type(
            item.get("tenant_id"),
            "origin.principal.tenant_id",
            TenantId,
        ),
        participant_id=_optional_new_type(
            item.get("participant_id"),
            "origin.principal.participant_id",
            ParticipantId,
        ),
        session_id=_optional_new_type(
            item.get("session_id"),
            "origin.principal.session_id",
            SessionId,
        ),
    )


def _encode_choice(choice: Choice) -> JsonObject:
    result: JsonObject = {
        "value": str(choice.value),
        "label": choice.label,
    }
    if choice.description is not None:
        result["description"] = choice.description
    return result


def _decode_choice(value: object) -> Choice:
    item = _object(value, "choice")
    _keys(
        item,
        required={"value", "label"},
        optional={"description"},
        path="choice",
    )
    return Choice(
        value=ChoiceValue(_string(item["value"], "choice.value")),
        label=_string(item["label"], "choice.label"),
        description=_optional_string(
            item.get("description"),
            "choice.description",
        ),
    )


def _encode_selection_value(value: SelectionValue) -> JsonObject:
    if not _is_selection_value_variant(value):
        _codec_error("selection", "selection value variant is unsupported")
    if type(value) is SelectedChoice:
        return {"kind": value.kind.value, "value": str(value.value)}
    assert type(value) is FreeFormOther
    return {"kind": value.kind.value, "text": value.text}


def _decode_selection_value(value: object) -> SelectionValue:
    item = _object(value, "selection")
    kind = _enum(item.get("kind"), SelectionValueType, "selection.kind")
    if kind is SelectionValueType.SELECTED_CHOICE:
        _keys(
            item,
            required={"kind", "value"},
            optional=set(),
            path="selection",
        )
        return SelectedChoice(
            value=ChoiceValue(_string(item["value"], "selection.value"))
        )
    _keys(
        item,
        required={"kind", "text"},
        optional=set(),
        path="selection",
    )
    return FreeFormOther(text=_string(item["text"], "selection.text"))


def _decode_text_constraints(
    value: object | None,
    *,
    maximum: int,
) -> TextValidationConstraints:
    if value is None:
        return TextValidationConstraints(maximum_length=maximum)
    item = _object(value, "question.constraints")
    _keys(
        item,
        required={"minimum_length", "maximum_length"},
        optional=set(),
        path="question.constraints",
    )
    return TextValidationConstraints(
        minimum_length=_integer(
            item["minimum_length"],
            "question.constraints.minimum_length",
        ),
        maximum_length=_integer(
            item["maximum_length"],
            "question.constraints.maximum_length",
        ),
    )


def _decode_selection_constraints(
    value: object | None,
    *,
    maximum: int,
) -> SelectionValidationConstraints:
    if value is None:
        return SelectionValidationConstraints(maximum=maximum)
    item = _object(value, "question.constraints")
    _keys(
        item,
        required={"minimum", "maximum"},
        optional=set(),
        path="question.constraints",
    )
    return SelectionValidationConstraints(
        minimum=_integer(item["minimum"], "question.constraints.minimum"),
        maximum=_integer(item["maximum"], "question.constraints.maximum"),
    )


def _require_empty_selection_fields(
    choices: list[object],
    allow_other: bool,
    item: dict[str, object],
) -> None:
    if choices or allow_other or "recommended_choice" in item:
        _codec_error(
            "question",
            "non-selection question contains selection fields",
        )


def _validate_revision_binding(
    actual: ContinuationRevisionBinding,
    expected: ContinuationRevisionBinding,
) -> None:
    if not isinstance(expected, ContinuationRevisionBinding):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            "continuation_snapshot.revision_binding",
            "expected binding must be a typed revision binding",
        )
    if (
        actual.provider_family != expected.provider_family
        or actual.model_id != expected.model_id
    ):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE,
            "continuation_snapshot.revision_binding",
            "snapshot provider or model is unavailable",
        )
    if (
        actual.provider_config_revision != expected.provider_config_revision
        or actual.model_config_revision != expected.model_config_revision
        or actual.capability_revision != expected.capability_revision
    ):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_REVISION_DRIFT,
            "continuation_snapshot.revision_binding",
            "snapshot configuration revision has drifted",
        )


def _snapshot_text(value: object, path: str) -> str:
    if not isinstance(value, str | bytes):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "snapshot must be text or UTF-8 bytes",
        )
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputSnapshotError(
                InputErrorCode.SNAPSHOT_INVALID,
                path,
                "snapshot bytes must be UTF-8",
            ) from exc
    else:
        text = value
    if len(text.encode("utf-8")) > _MAX_SNAPSHOT_UTF8_BYTES:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "snapshot exceeds its byte bound",
        )
    return text


def _require_canonical_snapshot_text(
    text: str,
    value: dict[str, object],
    path: str,
) -> None:
    canonical = _canonical_json_text(cast(JsonObject, value), path)
    if text != canonical:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "snapshot does not use its canonical JSON representation",
        )


def _raw_continuation_snapshot_digest(value: dict[str, object]) -> str:
    preimage = cast(
        JsonObject,
        {key: item for key, item in value.items() if key != "content_sha256"},
    )
    return sha256(
        _canonical_json_text(preimage, "continuation_snapshot").encode("utf-8")
    ).hexdigest()


def _strict_json(value: str, path: str) -> object:
    try:
        raw = cast(
            object,
            loads(value, object_pairs_hook=_object_without_duplicates),
        )
    except (JSONDecodeError, InputSnapshotError, ValueError) as exc:
        if isinstance(exc, InputSnapshotError):
            raise
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "snapshot is not strict JSON",
        ) from exc
    _validate_json_tree(raw, path)
    return raw


def _canonical_json_text(value: JsonObject, path: str) -> str:
    try:
        return dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "snapshot contains a non-JSON value",
        ) from exc


def _sha256_string(value: object, path: str) -> str:
    raw = _string(value, path)
    if len(raw) != 64 or any(
        character not in "0123456789abcdef" for character in raw
    ):
        raise InputSnapshotError(
            InputErrorCode.SNAPSHOT_INVALID,
            path,
            "content hash must be lowercase SHA-256 hexadecimal",
        )
    return raw


def _mutable_json_mapping(value: Mapping[str, JsonValue]) -> JsonObject:
    return {key: _mutable_json_value(item) for key, item in value.items()}


def _mutable_json_value(value: JsonValue) -> MutableJsonValue:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, tuple):
        return [_mutable_json_value(item) for item in value]
    return _mutable_json_mapping(value)


def _immutable_json_mapping(
    value: object,
    path: str,
) -> dict[str, JsonValue]:
    item = _object(value, path)
    return {
        key: _immutable_json_value(
            child,
            _json_mapping_value_path(path, key),
        )
        for key, child in item.items()
    }


def _immutable_json_value(value: object, path: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        validate_finite_number(value, path)
        return value
    if isinstance(value, list):
        return tuple(
            _immutable_json_value(item, _json_sequence_item_path(path, index))
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        return _immutable_json_mapping(value, path)
    raise InputSnapshotError(
        InputErrorCode.SNAPSHOT_INVALID,
        path,
        "snapshot contains a non-JSON value",
    )


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _codec_error(path, "value must be a JSON object")
    return cast(dict[str, object], value)


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        _codec_error(path, "value must be a JSON array")
    return cast(list[object], value)


def _reject_prohibited_question_tag(value: dict[str, object]) -> None:
    """Reject explicit secret request tags at the synchronous boundary.

    Classification of semantic free text and submitted values belongs to the
    trusted host policy and is intentionally not inferred here.
    """
    for field_name in ("kind", "semantic_type"):
        raw = value.get(field_name)
        if (
            isinstance(raw, str)
            and raw.casefold() in _PROHIBITED_QUESTION_TAGS
        ):
            raise InputCodecError(
                InputErrorCode.PROHIBITED_INPUT,
                f"question.{field_name}",
                "secret and authentication requests are prohibited",
            )


def _reject_explicit_nulls(
    value: dict[str, object],
    fields: set[str],
    path: str,
) -> None:
    for field_name in fields:
        if field_name in value and value[field_name] is None:
            raise InputCodecError(
                InputErrorCode.INVALID_FORMAT,
                f"{path}.{field_name}",
                "optional fields must be omitted instead of null",
            )


def _keys(
    value: dict[str, object],
    *,
    required: set[str],
    optional: set[str],
    path: str,
    snapshot: bool = False,
) -> None:
    actual = set(value)
    if actual == required | (actual & optional) and required <= actual:
        return
    error_type = InputSnapshotError if snapshot else InputCodecError
    raise error_type(
        (
            InputErrorCode.SNAPSHOT_INVALID
            if snapshot
            else InputErrorCode.INVALID_FORMAT
        ),
        path,
        "object fields do not match the canonical schema",
    )


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        _codec_error(path, "value must be a string")
    return value


def _optional_string(value: object | None, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        _codec_error(path, "value must be a boolean")
    return value


def _optional_bool(value: object | None, path: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, path)


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _codec_error(path, "value must be an integer")
    return value


def _optional_integer(value: object | None, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _enum(
    value: object,
    enum_type: type[EnumValue],
    path: str,
) -> EnumValue:
    raw = _string(value, path)
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise InputCodecError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value is not a supported discriminator",
        ) from exc


def _optional_enum(
    value: object | None,
    enum_type: type[EnumValue],
    path: str,
) -> EnumValue | None:
    if value is None:
        return None
    return _enum(value, enum_type, path)


def _optional_new_type(
    value: object | None,
    path: str,
    constructor: Callable[[str], NewValue],
) -> NewValue | None:
    if value is None:
        return None
    return constructor(_string(value, path))


def _datetime(value: object, path: str) -> datetime:
    raw = _string(value, path)
    if not raw.endswith("Z"):
        raise InputCodecError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "timestamp must use canonical UTC RFC 3339 form",
        )
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise InputCodecError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value must be an ISO 8601 timestamp",
        ) from exc
    if _encode_datetime(parsed) != raw:
        raise InputCodecError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "timestamp must use canonical microsecond precision",
        )
    return parsed


def _encode_datetime(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _codec_error(path: str, message: str) -> NoReturn:
    raise InputCodecError(InputErrorCode.INVALID_TYPE, path, message)


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InputSnapshotError(
                InputErrorCode.SNAPSHOT_INVALID,
                "snapshot",
                "snapshot contains a duplicate object name",
            )
        result[key] = value
    return result


def _validate_json_tree(value: object, path: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        try:
            validate_finite_number(value, path)
        except Exception as exc:
            raise InputSnapshotError(
                InputErrorCode.SNAPSHOT_INVALID,
                path,
                "snapshot contains a non-finite number",
            ) from exc
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, _json_sequence_item_path(path, index))
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in cast(dict[str, object], value).items():
            _validate_json_tree(item, _json_mapping_value_path(path, key))
        return
    raise InputSnapshotError(
        InputErrorCode.SNAPSHOT_INVALID,
        path,
        "snapshot contains a non-JSON value",
    )


def _json_mapping_value_path(path: str, key: str) -> str:
    if _is_continuation_payload_path(path):
        return f"{path}.value"
    return f"{path}.{key}"


def _json_sequence_item_path(path: str, index: int) -> str:
    if _is_continuation_payload_path(path):
        return f"{path}.item"
    return f"{path}[{index}]"


def _is_continuation_payload_path(path: str) -> bool:
    prefix = "continuation_snapshot.payload"
    return path == prefix or path.startswith(f"{prefix}.")
