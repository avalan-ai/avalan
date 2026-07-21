"""Exercise strict structured-interaction domain wire codecs."""

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import dumps, loads
from typing import ClassVar, cast

import pytest

from avalan.interaction import (
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
    InputCodecError,
    InputContinuationOutcome,
    InputDeclinedResult,
    InputErrorCode,
    InputModelResult,
    InputQuestion,
    InputRequest,
    InputRequestId,
    InputRequiredResult,
    InputResolution,
    InputResultKind,
    InputSnapshotError,
    InputTimedOutResult,
    InputUnavailableResult,
    InputValidationError,
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
    canonical_continuation_snapshot_bytes,
    canonical_interaction_snapshot_bytes,
    canonical_resolution_digest,
    continuation_snapshot_digest,
    create_input_request,
    decode_continuation_snapshot,
    decode_execution_origin,
    decode_input_answer,
    decode_input_model_result,
    decode_input_question,
    decode_input_request,
    decode_input_required_result,
    decode_input_resolution,
    decode_interaction_snapshot,
    encode_continuation_outcome,
    encode_continuation_snapshot,
    encode_execution_origin,
    encode_input_answer,
    encode_input_model_result,
    encode_input_question,
    encode_input_request,
    encode_input_required_result,
    encode_input_resolution,
    encode_interaction_snapshot,
    interaction_snapshot_digest,
    semantic_request_fingerprint,
)
from avalan.interaction.codec import (
    _canonical_json_text,
    _decode_selection_value,
    _encode_selection_value,
    _immutable_json_value,
    _validate_json_tree,
)
from avalan.types import JsonObject

_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


class _UnsupportedAnswer(InputAnswer):
    """Provide a typed but unsupported answer variant for boundary testing."""

    question_type: ClassVar[QuestionType] = QuestionType.TEXT

    def __post_init__(self) -> None:
        pass


class _UnsupportedQuestion(InputQuestion):
    """Provide an unregistered question variant for boundary testing."""

    kind: ClassVar[QuestionType] = QuestionType.CONFIRMATION

    def __post_init__(self) -> None:
        pass


class _UnsupportedResolution(InputResolution):
    """Provide an unregistered resolution variant for boundary testing."""

    status: ClassVar[ResolutionStatus] = ResolutionStatus.DECLINED

    def __post_init__(self) -> None:
        pass


class _UnsupportedSelectionValue(SelectionValue):
    """Provide an unregistered selection variant for boundary testing."""

    kind: ClassVar[SelectionValueType] = SelectionValueType.SELECTED_CHOICE

    def __post_init__(self) -> None:
        pass


def _skip_variant_validation(_: object) -> None:
    pass


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        task_id=TaskId("task-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        parent_branch_id=BranchId("parent-1"),
        model_call_id=ModelCallId("call-1"),
        stream_session_id=StreamSessionId("stream-1"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://support",
            agent_definition_revision="agent-r1",
            operation_id="operation-1",
            operation_index=2,
            model_config_reference="model-config-1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
        principal=PrincipalScope(
            user_id=UserId("user-1"),
            tenant_id=TenantId("tenant-1"),
            participant_id=ParticipantId("participant-1"),
            session_id=SessionId("session-1"),
        ),
    )


def _choices() -> tuple[Choice, Choice]:
    return (
        Choice(value=ChoiceValue("one"), label="One", description="First"),
        Choice(value=ChoiceValue("two"), label="Two", description="Second"),
    )


def _questions() -> tuple[InputQuestion, ...]:
    return (
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
            header="Confirmation",
            help_text="Confirm the operation.",
            presentation_hint=PresentationHint.COMPACT,
            default_value=True,
        ),
        TextQuestion(
            question_id=QuestionId("text"),
            prompt="Name?",
            required=True,
            header="Identity",
            help_text="Provide a public display name.",
            presentation_hint=PresentationHint.SINGLE_LINE,
            default_value="Ada",
            constraints=TextValidationConstraints(
                minimum_length=1,
                maximum_length=20,
            ),
        ),
        MultilineTextQuestion(
            question_id=QuestionId("multiline"),
            prompt="Notes?",
            required=False,
            header="Notes",
            help_text="Provide optional notes.",
            presentation_hint=PresentationHint.EDITOR,
            default_value="one\ntwo",
            constraints=TextValidationConstraints(maximum_length=200),
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("single"),
            prompt="Choose one.",
            required=True,
            header="Strategy",
            help_text="Select a stable value.",
            presentation_hint=PresentationHint.RADIO,
            choices=_choices(),
            allow_other=True,
            recommended_choice=ChoiceValue("one"),
            default_value=ChoiceValue("two"),
        ),
        MultipleSelectionQuestion(
            question_id=QuestionId("multiple"),
            prompt="Choose several.",
            required=True,
            header="Checks",
            help_text="Select one or more values.",
            presentation_hint=PresentationHint.CHECKBOX,
            choices=_choices(),
            allow_other=True,
            recommended_choice=ChoiceValue("two"),
            default_value=(ChoiceValue("one"),),
            constraints=SelectionValidationConstraints(minimum=1, maximum=3),
        ),
    )


def _request(
    *,
    questions: tuple[InputQuestion, ...] | None = None,
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> InputRequest:
    return create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=_origin(),
        mode=mode,
        reason="A typed response is required.",
        questions=questions or _questions()[:3],
        created_at=_NOW,
        advisory_wait_seconds=30 if mode is RequirementMode.ADVISORY else None,
    )


def _answers() -> tuple[
    ConfirmationAnswer,
    TextAnswer,
    MultilineTextAnswer,
    SingleSelectionAnswer,
    MultipleSelectionAnswer,
]:
    return (
        ConfirmationAnswer(
            question_id=QuestionId("confirm"),
            provenance=AnswerProvenance.HUMAN,
            value=True,
        ),
        TextAnswer(
            question_id=QuestionId("text"),
            provenance=AnswerProvenance.TRUSTED_DEFAULT,
            value="Ada",
        ),
        MultilineTextAnswer(
            question_id=QuestionId("multiline"),
            provenance=AnswerProvenance.POLICY,
            value="one\ntwo",
        ),
        SingleSelectionAnswer(
            question_id=QuestionId("single"),
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            value=SelectedChoice(value=ChoiceValue("one")),
        ),
        MultipleSelectionAnswer(
            question_id=QuestionId("multiple"),
            provenance=AnswerProvenance.HUMAN,
            values=(
                SelectedChoice(value=ChoiceValue("one")),
                FreeFormOther(text="custom"),
            ),
        ),
    )


def _binding() -> ContinuationRevisionBinding:
    return ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("openai"),
        model_id=ModelId("gpt-5"),
        provider_config_revision=ProviderConfigRevision("provider-r1"),
        model_config_revision=ModelConfigRevision("model-r1"),
        capability_revision=CapabilityRevision("capability-r1"),
    )


def _continuation_snapshot() -> ContinuationSnapshot:
    return ContinuationSnapshot(
        snapshot_kind="responses.v1",
        revision_binding=_binding(),
        model_call_id=ModelCallId("call-1"),
        provider_idempotency_key=ProviderIdempotencyKey("key-1"),
        payload={
            "cursor": "cursor-1",
            "sequence": (None, True, 2, 3.5, {"nested": ("value",)}),
        },
    )


def _canonical_wire(value: object) -> str:
    return dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rehash_continuation_wire(
    value: dict[str, object],
) -> dict[str, object]:
    result = dict(value)
    preimage = {
        key: item for key, item in result.items() if key != "content_sha256"
    }
    result["content_sha256"] = sha256(
        _canonical_wire(preimage).encode("utf-8")
    ).hexdigest()
    return result


def _assert_payload_diagnostic_redacted(
    error: InputValidationError,
    *private_values: str,
) -> None:
    diagnostic_fields = (error.path, error.safe_message, str(error))
    assert "[" not in error.path
    assert "]" not in error.path
    for private_value in private_values:
        assert all(
            private_value not in diagnostic for diagnostic in diagnostic_fields
        )


def test_origin_and_all_question_variants_round_trip() -> None:
    """Round-trip complete correlation and every semantic question type."""
    origin = _origin()
    assert decode_execution_origin(encode_execution_origin(origin)) == origin

    for question in _questions():
        encoded = encode_input_question(question)
        assert decode_input_question(encoded) == question
        assert encoded["kind"] == question.kind.value

    bare_text = {
        "question_id": "bare-text",
        "kind": "text",
        "prompt": "Text?",
        "required": False,
        "choices": [],
        "allow_other": False,
    }
    bare_multiple = {
        "question_id": "bare-multiple",
        "kind": "multiple_selection",
        "prompt": "Choose.",
        "required": False,
        "choices": [
            {"value": "one", "label": "One"},
            {"value": "two", "label": "Two"},
        ],
        "allow_other": False,
    }
    assert isinstance(decode_input_question(bare_text), TextQuestion)
    decoded_multiple = decode_input_question(bare_multiple)
    assert isinstance(decoded_multiple, MultipleSelectionQuestion)
    assert decoded_multiple.constraints.maximum == 2


def test_all_answer_and_resolution_variants_round_trip() -> None:
    """Round-trip every answer tag and terminal resolution status."""
    answers = _answers()
    for answer in answers:
        assert decode_input_answer(encode_input_answer(answer)) == answer

    resolutions: tuple[InputResolution, ...] = (
        AnsweredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            resolved_at=_NOW,
            answers=answers,
        ),
        DeclinedResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ),
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            scope=CancellationScope.CONTAINING_RUN,
        ),
        TimedOutResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        UnavailableResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        ExpiredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        SupersededResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
    )
    for resolution in resolutions:
        encoded = encode_input_resolution(resolution)
        assert decode_input_resolution(encoded) == resolution
        assert canonical_resolution_digest(
            resolution
        ) == canonical_resolution_digest(resolution)


def test_request_segment_and_model_results_round_trip() -> None:
    """Round-trip requests and explicit transport/model results."""
    advisory = _request(mode=RequirementMode.ADVISORY)
    assert decode_input_request(encode_input_request(advisory)) == advisory

    resolution = DeclinedResolution(
        request_id=advisory.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    terminal = InputRequest(
        request_id=advisory.request_id,
        continuation_id=advisory.continuation_id,
        origin=advisory.origin,
        mode=advisory.mode,
        reason=advisory.reason,
        questions=advisory.questions,
        created_at=advisory.created_at,
        continuation_ttl_seconds=advisory.continuation_ttl_seconds,
        advisory_wait_seconds=advisory.advisory_wait_seconds,
        advisory_deadline=_NOW + timedelta(seconds=30),
        state=RequestState.DECLINED,
        state_revision=StateRevision(2),
        resolution=resolution,
    )
    assert decode_input_request(encode_input_request(terminal)) == terminal

    required = InputRequiredResult(
        request_id=advisory.request_id,
        continuation_id=advisory.continuation_id,
        detached_resumption_available=True,
    )
    assert (
        decode_input_required_result(encode_input_required_result(required))
        == required
    )

    answered = InputAnsweredResult(
        request_id=advisory.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(_answers()[0],),
    )
    results: tuple[InputModelResult, ...] = (
        answered,
        InputDeclinedResult(
            request_id=advisory.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ),
        InputCancelledResult(
            request_id=advisory.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ),
        InputTimedOutResult(
            request_id=advisory.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
        InputUnavailableResult(
            request_id=advisory.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW,
        ),
    )
    for result in results:
        assert (
            decode_input_model_result(encode_input_model_result(result))
            == result
        )

    outcomes: tuple[InputContinuationOutcome, ...] = (
        ResumeInputContinuation(
            request_id=advisory.request_id, result=answered
        ),
        TerminateInputContinuation(
            request_id=advisory.request_id,
            status=ResolutionStatus.EXPIRED,
        ),
    )
    assert [
        encode_continuation_outcome(outcome)["disposition"]
        for outcome in outcomes
    ] == [
        "resume",
        "terminate",
    ]


def test_interaction_snapshot_round_trip_binds_canonical_hash() -> None:
    """Round-trip interaction snapshots with a canonical hash preimage."""
    snapshot = InteractionSnapshot(request=_request())
    encoded = encode_interaction_snapshot(snapshot)
    preimage = canonical_interaction_snapshot_bytes(snapshot)

    assert decode_interaction_snapshot(encoded) == snapshot
    assert decode_interaction_snapshot(encoded.encode("utf-8")) == snapshot
    assert interaction_snapshot_digest(
        snapshot
    ) == interaction_snapshot_digest(snapshot)
    assert b"content_sha256" not in preimage
    assert loads(encoded)["content_sha256"] == interaction_snapshot_digest(
        snapshot
    )


def test_continuation_snapshot_round_trip_binds_provider_revisions() -> None:
    """Round-trip provider state with typed identity and revision binding."""
    snapshot = _continuation_snapshot()
    encoded = encode_continuation_snapshot(snapshot)
    preimage = canonical_continuation_snapshot_bytes(snapshot)

    assert (
        decode_continuation_snapshot(
            encoded,
            expected_binding=snapshot.revision_binding,
        )
        == snapshot
    )
    assert (
        decode_continuation_snapshot(
            encoded.encode("utf-8"),
            expected_binding=snapshot.revision_binding,
        )
        == snapshot
    )
    assert continuation_snapshot_digest(
        snapshot
    ) == continuation_snapshot_digest(snapshot)
    assert b"content_sha256" not in preimage
    assert loads(encoded)["content_sha256"] == continuation_snapshot_digest(
        snapshot
    )


def test_semantic_fingerprint_covers_constraints_and_stable_values() -> None:
    """Hash semantic constraints while excluding display-only metadata."""
    text_request = _request(questions=_questions()[1:3])
    selection_request = _request(questions=_questions()[3:5])
    changed = _request(
        questions=(
            SingleSelectionQuestion(
                question_id=QuestionId("single"),
                prompt="Choose one.",
                required=True,
                choices=(
                    Choice(value=ChoiceValue("one"), label="Renamed"),
                    Choice(value=ChoiceValue("two"), label="Also renamed"),
                ),
                allow_other=True,
            ),
            _questions()[4],
        )
    )

    assert semantic_request_fingerprint(text_request)
    assert semantic_request_fingerprint(selection_request)
    assert semantic_request_fingerprint(
        selection_request
    ) == semantic_request_fingerprint(changed)


def test_codec_rejects_invalid_root_types_and_unsupported_variants() -> None:
    """Reject arbitrary objects before unchecked serialization."""
    invalid = object()
    encoders = (
        encode_input_request,
        encode_execution_origin,
        encode_input_question,
        encode_input_answer,
        encode_input_resolution,
        encode_input_required_result,
        encode_input_model_result,
        encode_continuation_outcome,
    )
    for encoder in encoders:
        with pytest.raises(InputCodecError):
            encoder(invalid)

    for decoder in (
        decode_input_request,
        decode_execution_origin,
        decode_input_question,
        decode_input_answer,
        decode_input_resolution,
        decode_input_required_result,
        decode_input_model_result,
    ):
        with pytest.raises(InputCodecError):
            decoder(invalid)

    private_value = "PRIVATE_ROGUE_VARIANT_SENTINEL"
    unsupported_answer = _UnsupportedAnswer(
        question_id=QuestionId(private_value),
        provenance=AnswerProvenance.HUMAN,
    )
    unsupported_question = _UnsupportedQuestion(
        question_id=QuestionId(private_value),
        prompt=private_value,
        required=True,
    )
    unsupported_resolution = _UnsupportedResolution(
        request_id=InputRequestId(private_value),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    unsupported_selection = _UnsupportedSelectionValue()
    rogue_model_result_type = type(
        "RogueInputDeclinedResult",
        (InputDeclinedResult,),
        {"__post_init__": _skip_variant_validation},
    )
    unsupported_model_result = rogue_model_result_type(
        request_id=InputRequestId(private_value),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    valid_model_result = InputDeclinedResult(
        request_id=InputRequestId(private_value),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    rogue_outcome_type = type(
        "RogueResumeInputContinuation",
        (ResumeInputContinuation,),
        {"__post_init__": _skip_variant_validation},
    )
    unsupported_outcome = rogue_outcome_type(
        request_id=InputRequestId(private_value),
        result=valid_model_result,
    )
    rogue_required_result_type = type(
        "RogueInputRequiredResult",
        (InputRequiredResult,),
        {"__post_init__": _skip_variant_validation},
    )
    unsupported_required_result = rogue_required_result_type(
        request_id=InputRequestId(private_value),
        continuation_id=ContinuationId("continuation-1"),
        detached_resumption_available=True,
    )
    valid_request = _request()
    rogue_request_type = type(
        "RogueInputRequest",
        (InputRequest,),
        {"__post_init__": _skip_variant_validation},
    )
    unsupported_request = rogue_request_type(
        request_id=valid_request.request_id,
        continuation_id=valid_request.continuation_id,
        origin=valid_request.origin,
        mode=valid_request.mode,
        reason=valid_request.reason,
        questions=valid_request.questions,
        created_at=valid_request.created_at,
        continuation_ttl_seconds=valid_request.continuation_ttl_seconds,
        advisory_wait_seconds=valid_request.advisory_wait_seconds,
        advisory_deadline=valid_request.advisory_deadline,
        state=valid_request.state,
        state_revision=valid_request.state_revision,
        resolution=valid_request.resolution,
    )
    unsupported_encoders = (
        (encode_input_request, unsupported_request),
        (encode_input_question, unsupported_question),
        (encode_input_answer, unsupported_answer),
        (encode_input_resolution, unsupported_resolution),
        (encode_input_model_result, unsupported_model_result),
        (encode_continuation_outcome, unsupported_outcome),
        (encode_input_required_result, unsupported_required_result),
        (_encode_selection_value, unsupported_selection),
    )
    for encoder, unsupported in unsupported_encoders:
        with pytest.raises(InputCodecError) as captured:
            encoder(unsupported)
        assert captured.value.code is InputErrorCode.INVALID_TYPE
        assert private_value not in captured.value.path
        assert private_value not in captured.value.safe_message
        assert private_value not in str(captured.value)

    with pytest.raises(InputCodecError):
        semantic_request_fingerprint(unsupported_request)
    forged_request = _request()
    object.__setattr__(forged_request, "questions", (unsupported_question,))
    with pytest.raises(InputCodecError):
        semantic_request_fingerprint(forged_request)

    with pytest.raises(InputCodecError):
        _encode_selection_value(cast(SelectedChoice, invalid))
    with pytest.raises(InputCodecError):
        _decode_selection_value(invalid)
    with pytest.raises(InputSnapshotError):
        _canonical_json_text(cast(JsonObject, {"value": invalid}), "snapshot")
    with pytest.raises(InputSnapshotError):
        _immutable_json_value(invalid, "snapshot.value")
    with pytest.raises(InputSnapshotError):
        _validate_json_tree(invalid, "snapshot.value")


def test_decoders_reject_schema_type_and_discriminator_mismatches() -> None:
    """Reject missing, additional, mistyped, and unknown wire fields."""
    question = encode_input_question(_questions()[0])
    malformed_questions: tuple[dict[str, object], ...] = (
        {key: value for key, value in question.items() if key != "prompt"},
        {**question, "extra": True},
        {**question, "kind": "unknown"},
        {**question, "required": 1},
        {**question, "choices": {}},
        {**question, "allow_other": "false"},
        {**question, "default_value": "true"},
        {**question, "choices": [{"value": "one", "label": "One"}]},
        {**question, "allow_other": True},
        {**question, "recommended_choice": "one"},
    )
    for malformed in malformed_questions:
        with pytest.raises(InputCodecError):
            decode_input_question(malformed)

    variant_fields = (
        (_questions()[0], "constraints", {"minimum_length": 0}),
        (_questions()[1], "recommended_choice", "one"),
        (_questions()[2], "recommended_choice", "one"),
        (_questions()[3], "constraints", {"minimum": 0, "maximum": 1}),
    )
    for variant, field_name, field_value in variant_fields:
        encoded_variant = encode_input_question(variant)
        with pytest.raises(InputCodecError):
            decode_input_question({**encoded_variant, field_name: field_value})

    nullable_fields = (
        (_questions()[0], "header"),
        (_questions()[0], "help"),
        (_questions()[0], "default_value"),
        (_questions()[0], "presentation_hint"),
        (_questions()[1], "constraints"),
        (_questions()[3], "recommended_choice"),
        (_questions()[4], "constraints"),
        (_questions()[4], "recommended_choice"),
    )
    for variant, field_name in nullable_fields:
        encoded_variant = encode_input_question(variant)
        with pytest.raises(InputCodecError) as error:
            decode_input_question({**encoded_variant, field_name: None})
        assert error.value.path == f"question.{field_name}"

    request = encode_input_request(_request())
    with pytest.raises(InputCodecError):
        decode_input_request({**request, "required": False})
    with pytest.raises(InputCodecError):
        decode_input_request({**request, "questions": {}})
    with pytest.raises(InputCodecError):
        decode_input_request({**request, "state_revision": True})
    with pytest.raises(InputCodecError):
        decode_input_request({**request, "mode": "unknown"})
    with pytest.raises(InputCodecError):
        decode_input_request({**request, "reason": 1})
    for invalid_timestamp in (
        "2026-07-20T12:00:00.000000+00:00",
        "not-a-dateZ",
        "2026-07-20T12:00:00Z",
    ):
        with pytest.raises(InputCodecError):
            decode_input_request({**request, "created_at": invalid_timestamp})

    required = encode_input_required_result(
        InputRequiredResult(
            request_id=InputRequestId("request-1"),
            continuation_id=ContinuationId("continuation-1"),
            detached_resumption_available=True,
        )
    )
    with pytest.raises(InputCodecError):
        decode_input_required_result(
            {
                **required,
                "kind": InputResultKind.ANSWERED.value,
            }
        )
    with pytest.raises(InputCodecError):
        decode_input_model_result(required)

    answer = encode_input_answer(_answers()[0])
    resolution = encode_input_resolution(
        DeclinedResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        )
    )
    model_result = encode_input_model_result(
        InputDeclinedResult(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        )
    )
    selection = _encode_selection_value(
        SelectedChoice(value=ChoiceValue("one"))
    )
    unknown_discriminators = (
        (decode_input_answer, {**answer, "kind": "rogue"}),
        (decode_input_resolution, {**resolution, "status": "rogue"}),
        (decode_input_model_result, {**model_result, "kind": "rogue"}),
        (_decode_selection_value, {**selection, "kind": "rogue"}),
    )
    for decoder, malformed in unknown_discriminators:
        with pytest.raises(InputCodecError):
            decoder(malformed)


def test_request_decoder_revalidates_terminal_aggregate_invariants() -> None:
    """Reject forged lifecycle revisions, resolutions, timing, and answers."""
    request = _request()
    wire = encode_input_request(request)
    declined = encode_input_resolution(
        DeclinedResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=1),
        )
    )
    timed_out = encode_input_resolution(
        TimedOutResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=60),
        )
    )
    predating = encode_input_resolution(
        DeclinedResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW - timedelta(microseconds=1),
        )
    )
    wrong_answer = encode_input_resolution(
        AnsweredResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(seconds=1),
            answers=(
                TextAnswer(
                    question_id=QuestionId("confirm"),
                    provenance=AnswerProvenance.HUMAN,
                    value="yes",
                ),
            ),
        )
    )
    invalid_requests = (
        {**wire, "state_revision": 99},
        {**wire, "state": "pending", "state_revision": 99},
        {
            **wire,
            "state": "declined",
            "state_revision": 99,
            "resolution": declined,
        },
        {
            **wire,
            "state": "timed_out",
            "state_revision": 0,
            "resolution": timed_out,
        },
        {
            **wire,
            "state": "timed_out",
            "state_revision": 2,
            "resolution": timed_out,
        },
        {
            **wire,
            "state": "declined",
            "state_revision": 2,
            "resolution": predating,
        },
        {
            **wire,
            "state": "answered",
            "state_revision": 2,
            "resolution": wrong_answer,
        },
        {
            **wire,
            "state": "expired",
            "state_revision": 2,
            "resolution": declined,
        },
    )
    for invalid in invalid_requests:
        with pytest.raises(InputValidationError):
            decode_input_request(invalid)

    advisory_wire = encode_input_request(
        _request(mode=RequirementMode.ADVISORY)
    )
    deadline = cast(str, timed_out["resolved_at"])
    early_timeout = encode_input_resolution(
        TimedOutResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_NOW + timedelta(seconds=59),
        )
    )
    invalid_advisory_requests = (
        {
            **advisory_wire,
            "state": "pending",
            "state_revision": 1,
        },
        {**advisory_wire, "advisory_deadline": deadline},
        {**wire, "advisory_deadline": deadline},
        {**advisory_wire, "advisory_deadline": None},
        {
            **advisory_wire,
            "state": "pending",
            "state_revision": 1,
            "advisory_deadline": advisory_wire["created_at"],
        },
        {
            **advisory_wire,
            "state": "timed_out",
            "state_revision": 2,
            "advisory_deadline": deadline,
            "resolution": early_timeout,
        },
    )
    for invalid in invalid_advisory_requests:
        with pytest.raises(InputValidationError):
            decode_input_request(invalid)

    pending_advisory = decode_input_request(
        {
            **advisory_wire,
            "state": "pending",
            "state_revision": 1,
            "advisory_deadline": deadline,
        }
    )
    terminal_advisory = decode_input_request(
        {
            **advisory_wire,
            "state": "timed_out",
            "state_revision": 2,
            "advisory_deadline": deadline,
            "resolution": timed_out,
        }
    )

    assert pending_advisory.advisory_deadline == _NOW + timedelta(seconds=60)
    assert (
        encode_input_request(terminal_advisory)["advisory_deadline"]
        == deadline
    )


def test_model_result_decoder_rejects_duplicate_answer_ids() -> None:
    """Reject duplicate answer identities before exposing a model result."""
    answered = InputAnsweredResult(
        request_id=InputRequestId("request-1"),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
        answers=(_answers()[0],),
    )
    wire = encode_input_model_result(answered)
    answer = cast(list[object], wire["answers"])[0]

    with pytest.raises(InputValidationError) as error:
        decode_input_model_result({**wire, "answers": [answer, answer]})
    assert error.value.code is InputErrorCode.DUPLICATE


def test_snapshot_decoders_reject_malformed_or_tampered_content() -> None:
    """Reject malformed JSON, invalid hashes, and oversized snapshots."""
    snapshot = InteractionSnapshot(request=_request())
    encoded = encode_interaction_snapshot(snapshot)
    wire = cast(dict[str, object], loads(encoded))

    invalid_interaction_values: tuple[object, ...] = (
        object(),
        b"\xff",
        "{" + '"version":1,' * 600_000 + "}",
        "{",
        '{"version":1,"version":1}',
        '{"number":NaN}',
    )
    for invalid in invalid_interaction_values:
        with pytest.raises(InputSnapshotError):
            decode_interaction_snapshot(cast(str | bytes, invalid))

    tampered_values = (
        {**wire, "version": 2},
        {**wire, "content_sha256": "ABC"},
        {**wire, "content_sha256": "0" * 64},
        {**wire, "request": {"invalid": True}},
        {**wire, "extra": True},
    )
    for tampered in tampered_values:
        with pytest.raises(InputSnapshotError):
            decode_interaction_snapshot(_canonical_wire(tampered))

    with pytest.raises(InputSnapshotError):
        encode_interaction_snapshot(cast(InteractionSnapshot, object()))
    with pytest.raises(InputSnapshotError):
        canonical_interaction_snapshot_bytes(
            cast(InteractionSnapshot, object())
        )
    oversized_request = _request()
    object.__setattr__(oversized_request, "reason", "x" * 1_048_576)
    with pytest.raises(InputSnapshotError):
        encode_interaction_snapshot(
            InteractionSnapshot(request=oversized_request)
        )


def test_snapshot_decoders_require_exact_canonical_json_text() -> None:
    """Reject alternate whitespace, key ordering, and number spellings."""
    interaction = encode_interaction_snapshot(
        InteractionSnapshot(request=_request())
    )
    interaction_wire = cast(dict[str, object], loads(interaction))
    interaction_alternates = (
        dumps(interaction_wire, ensure_ascii=False, sort_keys=True),
        dumps(
            dict(reversed(tuple(interaction_wire.items()))),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    for alternate in interaction_alternates:
        assert alternate != interaction
        with pytest.raises(InputSnapshotError) as error:
            decode_interaction_snapshot(alternate)
        assert error.value.code is InputErrorCode.SNAPSHOT_INVALID

    continuation = encode_continuation_snapshot(_continuation_snapshot())
    continuation_wire = cast(dict[str, object], loads(continuation))
    alternate_number = continuation.replace("3.5", "3.50")
    continuation_alternates = (
        dumps(continuation_wire, ensure_ascii=False, sort_keys=True),
        dumps(
            dict(reversed(tuple(continuation_wire.items()))),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        alternate_number,
    )
    for alternate in continuation_alternates:
        assert alternate != continuation
        with pytest.raises(InputSnapshotError) as error:
            decode_continuation_snapshot(
                alternate,
                expected_binding=_binding(),
            )
        assert error.value.code is InputErrorCode.SNAPSHOT_INVALID


def test_continuation_snapshot_rejects_drift_secrets_and_invalid_json() -> (
    None
):
    """Reject provider drift, secrets, and invalid JSON."""
    snapshot = _continuation_snapshot()
    encoded = encode_continuation_snapshot(snapshot)
    wire = cast(dict[str, object], loads(encoded))
    private_key = "PRIVATE_PROVIDER_KEY_SENTINEL"
    private_value = "PRIVATE_PROVIDER_VALUE_SENTINEL"
    private_number_key = "PRIVATE_NUMBER_KEY_SENTINEL"

    provider_drift = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("other"),
        model_id=snapshot.revision_binding.model_id,
        provider_config_revision=snapshot.revision_binding.provider_config_revision,
        model_config_revision=snapshot.revision_binding.model_config_revision,
        capability_revision=snapshot.revision_binding.capability_revision,
    )
    revision_drift = ContinuationRevisionBinding(
        provider_family=snapshot.revision_binding.provider_family,
        model_id=snapshot.revision_binding.model_id,
        provider_config_revision=ProviderConfigRevision("provider-r2"),
        model_config_revision=snapshot.revision_binding.model_config_revision,
        capability_revision=snapshot.revision_binding.capability_revision,
    )
    for binding, code in (
        (provider_drift, InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE),
        (revision_drift, InputErrorCode.SNAPSHOT_REVISION_DRIFT),
    ):
        with pytest.raises(InputSnapshotError) as error:
            decode_continuation_snapshot(encoded, expected_binding=binding)
        assert error.value.code is code

    wrong_provider_and_secret = _rehash_continuation_wire(
        {
            **wire,
            "provider_family": "other",
            "payload": {private_key: {"password": private_value}},
        }
    )
    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            _canonical_wire(wrong_provider_and_secret),
            expected_binding=snapshot.revision_binding,
        )
    assert error.value.code is InputErrorCode.SNAPSHOT_PROVIDER_UNAVAILABLE
    _assert_payload_diagnostic_redacted(
        error.value,
        private_key,
        private_value,
        "password",
    )

    revision_drift_and_invalid_payload = _rehash_continuation_wire(
        {
            **wire,
            "provider_config_revision": "provider-r2",
            "payload": [],
        }
    )
    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            _canonical_wire(revision_drift_and_invalid_payload),
            expected_binding=snapshot.revision_binding,
        )
    assert error.value.code is InputErrorCode.SNAPSHOT_REVISION_DRIFT

    tampered_and_drifted = {
        **wire,
        "payload": {"cursor": "tampered"},
    }
    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            _canonical_wire(tampered_and_drifted),
            expected_binding=provider_drift,
        )
    assert error.value.code is InputErrorCode.SNAPSHOT_INVALID
    assert error.value.path == "continuation_snapshot.content_sha256"

    for field_name in (
        "provider_family",
        "model_id",
        "provider_config_revision",
        "model_config_revision",
        "capability_revision",
    ):
        with pytest.raises(InputSnapshotError) as error:
            decode_continuation_snapshot(
                _canonical_wire(
                    _rehash_continuation_wire({**wire, field_name: ""})
                ),
                expected_binding=snapshot.revision_binding,
            )
        assert error.value.code is InputErrorCode.SNAPSHOT_INVALID

    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            _canonical_wire(
                _rehash_continuation_wire(
                    {
                        **wire,
                        "payload": {
                            private_key: {"e\u0301": "first", "é": "second"}
                        },
                    }
                )
            ),
            expected_binding=snapshot.revision_binding,
        )
    assert error.value.code is InputErrorCode.SNAPSHOT_INVALID
    _assert_payload_diagnostic_redacted(
        error.value,
        private_key,
        "e\u0301",
        "é",
    )

    bad_hash_and_secret = {
        **wire,
        "content_sha256": "0" * 64,
        "payload": {private_key: {"password": private_value}},
    }
    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            _canonical_wire(bad_hash_and_secret),
            expected_binding=snapshot.revision_binding,
        )
    assert error.value.code is InputErrorCode.SNAPSHOT_INVALID
    assert error.value.path == "continuation_snapshot.content_sha256"
    _assert_payload_diagnostic_redacted(
        error.value,
        private_key,
        private_value,
        "password",
    )

    unsupported_version = _rehash_continuation_wire({**wire, "version": 2})
    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            _canonical_wire(unsupported_version),
            expected_binding=snapshot.revision_binding,
        )
    assert error.value.code is InputErrorCode.SNAPSHOT_UNSUPPORTED

    prohibited_secret = _rehash_continuation_wire(
        {
            **wire,
            "payload": {private_key: {"password": private_value}},
        }
    )
    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            _canonical_wire(prohibited_secret),
            expected_binding=snapshot.revision_binding,
        )
    assert error.value.code is InputErrorCode.SNAPSHOT_SECRET_PROHIBITED
    _assert_payload_diagnostic_redacted(
        error.value,
        private_key,
        private_value,
        "password",
    )

    empty_nested_key = _rehash_continuation_wire(
        {
            **wire,
            "payload": {private_key: {"": private_value}},
        }
    )
    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            _canonical_wire(empty_nested_key),
            expected_binding=snapshot.revision_binding,
        )
    _assert_payload_diagnostic_redacted(
        error.value,
        private_key,
        private_value,
    )

    nonfinite_payload = {
        **wire,
        "payload": {private_key: {private_number_key: float("nan")}},
    }
    with pytest.raises(InputSnapshotError) as error:
        decode_continuation_snapshot(
            dumps(
                nonfinite_payload,
                allow_nan=True,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            expected_binding=snapshot.revision_binding,
        )
    _assert_payload_diagnostic_redacted(
        error.value,
        private_key,
        private_number_key,
    )

    for tampered in (
        _rehash_continuation_wire({**wire, "snapshot_kind": "INVALID"}),
        _rehash_continuation_wire({**wire, "payload": []}),
    ):
        with pytest.raises(InputSnapshotError):
            decode_continuation_snapshot(
                _canonical_wire(tampered),
                expected_binding=snapshot.revision_binding,
            )

    malformed_values = (
        {**wire, "content_sha256": "invalid"},
        {**wire, "content_sha256": "0" * 64},
        {**wire, "extra": True},
    )
    for tampered in malformed_values:
        with pytest.raises(InputSnapshotError):
            decode_continuation_snapshot(
                _canonical_wire(tampered),
                expected_binding=snapshot.revision_binding,
            )

    for invalid in (object(), b"\xff", "{", '{"value":NaN}'):
        with pytest.raises(InputSnapshotError):
            decode_continuation_snapshot(
                cast(str | bytes, invalid),
                expected_binding=snapshot.revision_binding,
            )
    with pytest.raises(InputSnapshotError):
        decode_continuation_snapshot(
            "x" * 1_048_577,
            expected_binding=snapshot.revision_binding,
        )

    with pytest.raises(InputSnapshotError):
        decode_continuation_snapshot(
            encoded,
            expected_binding=cast(ContinuationRevisionBinding, object()),
        )
    with pytest.raises(InputSnapshotError):
        encode_continuation_snapshot(cast(ContinuationSnapshot, object()))
    with pytest.raises(InputSnapshotError):
        canonical_continuation_snapshot_bytes(
            cast(ContinuationSnapshot, object())
        )

    oversized = ContinuationSnapshot(
        snapshot_kind="responses.v1",
        revision_binding=_binding(),
        model_call_id=ModelCallId("call-1"),
        provider_idempotency_key=ProviderIdempotencyKey("key-1"),
        payload={"large": "x" * 1_048_576},
    )
    with pytest.raises(InputSnapshotError):
        encode_continuation_snapshot(oversized)
