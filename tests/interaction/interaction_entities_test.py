"""Exercise immutable structured-interaction domain values."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
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
    FreeFormOther,
    HostCapabilities,
    HostHandling,
    InputAnswer,
    InputAnsweredResult,
    InputDeclinedResult,
    InputErrorCode,
    InputQuestion,
    InputRequest,
    InputRequestId,
    InputRequiredResult,
    InputResolution,
    InputResultKind,
    InputTransitionApplied,
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
    TaskId,
    TenantId,
    TerminateInputContinuation,
    TextAnswer,
    TextQuestion,
    TextValidationConstraints,
    TimedOutResolution,
    TurnId,
    UserId,
    create_input_request,
)
from avalan.types import JsonValue

_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


class _UncheckedQuestion(InputQuestion):
    """Bypass base validation to exercise aggregate boundaries."""

    kind: ClassVar[QuestionType] = QuestionType.CONFIRMATION

    def __post_init__(self) -> None:
        pass


class _UncheckedAnswer(InputAnswer):
    """Bypass base validation to exercise aggregate boundaries."""

    question_type: ClassVar[QuestionType] = QuestionType.CONFIRMATION

    def __post_init__(self) -> None:
        pass


class _UncheckedResolution(InputResolution):
    """Bypass base validation to exercise aggregate boundaries."""

    status: ClassVar[ResolutionStatus] = ResolutionStatus.DECLINED

    def __post_init__(self) -> None:
        pass


class _UncheckedSelectionValue(SelectionValue):
    """Bypass base validation to exercise aggregate boundaries."""

    kind: ClassVar[SelectionValueType] = SelectionValueType.SELECTED_CHOICE

    def __post_init__(self) -> None:
        pass


def _skip_variant_validation(_: object) -> None:
    pass


def _definition() -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="agent://support",
        agent_definition_revision="agent-r1",
        operation_id="operation-1",
        operation_index=2,
        model_config_reference="model-config-1",
        tool_revision="tools-r1",
        capability_revision="capabilities-r1",
    )


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
        definition=_definition(),
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


def _question() -> ConfirmationQuestion:
    return ConfirmationQuestion(
        question_id=QuestionId("confirm"),
        prompt="Continue?",
        required=True,
        default_value=True,
    )


def _request(
    *,
    mode: RequirementMode = RequirementMode.REQUIRED,
    questions: tuple[ConfirmationQuestion, ...] | None = None,
    state: RequestState = RequestState.CREATED,
    revision: int = 0,
    resolution: DeclinedResolution | None = None,
    advisory_deadline: datetime | None = None,
) -> InputRequest:
    return InputRequest(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=_origin(),
        mode=mode,
        reason="A decision is required.",
        questions=questions or (_question(),),
        created_at=_NOW,
        advisory_wait_seconds=60 if mode is RequirementMode.ADVISORY else None,
        advisory_deadline=advisory_deadline,
        state=state,
        state_revision=StateRevision(revision),
        resolution=resolution,
    )


def _binding() -> ContinuationRevisionBinding:
    return ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("openai"),
        model_id=ModelId("gpt-5"),
        provider_config_revision=ProviderConfigRevision("provider-r1"),
        model_config_revision=ModelConfigRevision("model-r1"),
        capability_revision=CapabilityRevision("capability-r1"),
    )


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


def test_public_tagged_union_constructors_reject_subclasses() -> None:
    """Reject unregistered subclasses before validating their content."""
    private_value = "PRIVATE_ROGUE_CONSTRUCTOR_SENTINEL"
    request = _request()
    result = InputDeclinedResult(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    rogue_question_type = type(
        "RogueQuestion",
        (InputQuestion,),
        {"kind": QuestionType.CONFIRMATION},
    )
    rogue_answer_type = type(
        "RogueAnswer",
        (InputAnswer,),
        {"question_type": QuestionType.CONFIRMATION},
    )
    rogue_resolution_type = type(
        "RogueResolution",
        (InputResolution,),
        {"status": ResolutionStatus.DECLINED},
    )
    rogue_selection_type = type(
        "RogueSelectionValue",
        (SelectionValue,),
        {"kind": SelectionValueType.SELECTED_CHOICE},
    )
    rogue_model_result_type = type(
        "RogueInputDeclinedResult",
        (InputDeclinedResult,),
        {},
    )
    rogue_outcome_type = type(
        "RogueResumeInputContinuation",
        (ResumeInputContinuation,),
        {},
    )
    rogue_transition_type = type(
        "RogueInputTransitionApplied",
        (InputTransitionApplied,),
        {},
    )
    rogue_required_result_type = type(
        "RogueInputRequiredResult",
        (InputRequiredResult,),
        {},
    )
    factories = (
        lambda: rogue_question_type(
            question_id=QuestionId(private_value),
            prompt=private_value,
            required=True,
        ),
        lambda: rogue_answer_type(
            question_id=QuestionId(private_value),
            provenance=AnswerProvenance.HUMAN,
        ),
        lambda: rogue_resolution_type(
            request_id=InputRequestId(private_value),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ),
        rogue_selection_type,
        lambda: rogue_model_result_type(
            request_id=InputRequestId(private_value),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ),
        lambda: rogue_outcome_type(
            request_id=InputRequestId(private_value),
            result=result,
        ),
        lambda: rogue_transition_type(
            previous=request,
            request=request,
            mutation_applied=False,
        ),
        lambda: rogue_required_result_type(
            request_id=InputRequestId(private_value),
            continuation_id=ContinuationId("continuation-1"),
            detached_resumption_available=True,
        ),
    )

    for factory in factories:
        with pytest.raises(InputValidationError) as captured:
            factory()
        assert captured.value.code is InputErrorCode.INVALID_TYPE
        assert private_value not in captured.value.path
        assert private_value not in captured.value.safe_message
        assert private_value not in str(captured.value)


def test_aggregate_constructors_reject_unchecked_union_subclasses() -> None:
    """Reject subclasses that deliberately bypass their base validation."""
    private_value = "PRIVATE_ROGUE_AGGREGATE_SENTINEL"
    question = _UncheckedQuestion(
        question_id=QuestionId(private_value),
        prompt=private_value,
        required=True,
    )
    answer = _UncheckedAnswer(
        question_id=QuestionId(private_value),
        provenance=AnswerProvenance.HUMAN,
    )
    resolution = _UncheckedResolution(
        request_id=InputRequestId(private_value),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    selection = _UncheckedSelectionValue()
    rogue_model_result_type = type(
        "UncheckedInputDeclinedResult",
        (InputDeclinedResult,),
        {"__post_init__": _skip_variant_validation},
    )
    model_result = rogue_model_result_type(
        request_id=InputRequestId(private_value),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    request_arguments = {
        "request_id": InputRequestId("request-1"),
        "continuation_id": ContinuationId("continuation-1"),
        "origin": _origin(),
        "mode": RequirementMode.REQUIRED,
        "reason": "A decision is required.",
        "created_at": _NOW,
    }
    aggregate_factories = (
        lambda: InputRequest(
            **request_arguments,
            questions=(question,),
        ),
        lambda: AnsweredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(answer,),
        ),
        lambda: InputAnsweredResult(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(answer,),
        ),
        lambda: SingleSelectionAnswer(
            question_id=QuestionId("single"),
            provenance=AnswerProvenance.HUMAN,
            value=selection,
        ),
        lambda: MultipleSelectionAnswer(
            question_id=QuestionId("multiple"),
            provenance=AnswerProvenance.HUMAN,
            values=(selection,),
        ),
        lambda: InputRequest(
            **request_arguments,
            questions=(_question(),),
            state=RequestState.DECLINED,
            state_revision=StateRevision(2),
            resolution=resolution,
        ),
        lambda: ResumeInputContinuation(
            request_id=InputRequestId(private_value),
            result=model_result,
        ),
    )

    for factory in aggregate_factories:
        with pytest.raises(InputValidationError) as captured:
            factory()
        assert captured.value.code is InputErrorCode.INVALID_TYPE
        assert private_value not in captured.value.path
        assert private_value not in captured.value.safe_message
        assert private_value not in str(captured.value)


def test_identity_values_validate_and_remain_frozen() -> None:
    """Validate trusted execution identity and immutable values."""
    origin = _origin()

    assert origin.task_id == "task-1"
    assert origin.principal.user_id == "user-1"
    assert origin.definition.operation_index == 2
    with pytest.raises(FrozenInstanceError):
        origin.run_id = RunId("other")

    with pytest.raises(InputValidationError):
        ExecutionDefinitionRef(
            agent_definition_locator="agent://support",
            agent_definition_revision="agent-r1",
            operation_id="operation-1",
            operation_index=True,
            model_config_reference="model-config-1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        )
    with pytest.raises(InputValidationError):
        ExecutionOrigin(
            run_id=RunId("run-1"),
            turn_id=TurnId("turn-1"),
            agent_id=AgentId("agent-1"),
            branch_id=BranchId("branch-1"),
            parent_branch_id=BranchId("branch-1"),
            model_call_id=ModelCallId("call-1"),
            stream_session_id=StreamSessionId("stream-1"),
            definition=_definition(),
        )
    with pytest.raises(InputValidationError):
        ExecutionOrigin(
            run_id=RunId("run-1"),
            turn_id=TurnId("turn-1"),
            agent_id=AgentId("agent-1"),
            branch_id=BranchId("branch-1"),
            model_call_id=ModelCallId("call-1"),
            stream_session_id=StreamSessionId("stream-1"),
            definition=cast(ExecutionDefinitionRef, object()),
        )
    with pytest.raises(InputValidationError):
        ExecutionOrigin(
            run_id=RunId("run-1"),
            turn_id=TurnId("turn-1"),
            agent_id=AgentId("agent-1"),
            branch_id=BranchId("branch-1"),
            model_call_id=ModelCallId("call-1"),
            stream_session_id=StreamSessionId("stream-1"),
            definition=_definition(),
            principal=cast(PrincipalScope, object()),
        )


def test_host_capabilities_select_the_strongest_handling() -> None:
    """Derive host handling without conflating availability modes."""
    assert (
        HostCapabilities(attached_resolution=True).handling
        is HostHandling.ATTACHED
    )
    detached = HostCapabilities(durable_resolution=True)
    assert detached.handling is HostHandling.DETACHED
    assert detached.can_advertise
    unavailable = HostCapabilities()
    assert unavailable.handling is HostHandling.UNAVAILABLE
    assert not unavailable.can_advertise
    with pytest.raises(InputValidationError):
        HostCapabilities(attached_resolution=cast(bool, 1))


def test_continuation_snapshot_freezes_json_and_rejects_secrets() -> None:
    """Freeze safe JSON provider state and reject invalid snapshots."""
    snapshot = ContinuationSnapshot(
        snapshot_kind="responses.v1",
        revision_binding=_binding(),
        model_call_id=ModelCallId("call-1"),
        provider_idempotency_key=ProviderIdempotencyKey("key-1"),
        payload={
            "none": None,
            "string": "value",
            "bool": True,
            "integer": 3,
            "float": 1.5,
            "object": {"items": [1, "two", False]},
            "tuple": ("three",),
        },
    )

    assert snapshot.payload["object"] == {"items": (1, "two", False)}
    with pytest.raises(TypeError):
        snapshot.payload["new"] = "value"

    invalid_arguments: tuple[dict[str, object], ...] = (
        {"version": 2},
        {"version": True},
        {"snapshot_kind": "Responses"},
        {"revision_binding": object()},
        {"payload": ["not", "an", "object"]},
        {"payload": {"access_token": "secret"}},
        {"payload": {"number": float("inf")}},
        {"payload": {"callback": object()}},
        {"payload": {"e\u0301": 1, "\u00e9": 2}},
    )
    base: dict[str, object] = {
        "snapshot_kind": "responses.v1",
        "revision_binding": _binding(),
        "model_call_id": ModelCallId("call-1"),
        "provider_idempotency_key": ProviderIdempotencyKey("key-1"),
        "payload": {},
    }
    for overrides in invalid_arguments:
        with pytest.raises(InputValidationError):
            ContinuationSnapshot(**cast(dict[str, object], base | overrides))

    with pytest.raises(InputValidationError):
        ContinuationRevisionBinding(
            provider_family=ProviderFamilyName("x" * 65),
            model_id=ModelId("gpt-5"),
            provider_config_revision=ProviderConfigRevision("provider-r1"),
            model_config_revision=ModelConfigRevision("model-r1"),
            capability_revision=CapabilityRevision("capability-r1"),
        )


def _assert_continuation_snapshot_key_rejected(
    key: str,
    nested: bool,
) -> None:
    private_value = "PRIVATE_CREDENTIAL_VALUE_SENTINEL"
    payload: dict[str, JsonValue]
    if nested:
        payload = {"provider_state": {key: private_value}}
        expected_path = "continuation_snapshot.payload.value.key"
    else:
        payload = {key: private_value}
        expected_path = "continuation_snapshot.payload.key"

    with pytest.raises(InputValidationError) as captured:
        ContinuationSnapshot(
            snapshot_kind="responses.v1",
            revision_binding=_binding(),
            model_call_id=ModelCallId("call-1"),
            provider_idempotency_key=ProviderIdempotencyKey("key-1"),
            payload=payload,
        )

    assert captured.value.code is InputErrorCode.SNAPSHOT_SECRET_PROHIBITED
    assert captured.value.path == expected_path
    _assert_payload_diagnostic_redacted(captured.value, private_value)


@pytest.mark.parametrize(
    ("key", "nested"),
    (
        ("token", False),
        ("ToKeN", True),
        ("authorization", False),
        ("AUTHORIZATION", True),
        ("client_secret", False),
        ("CLIENT-SECRET", True),
        ("client secret", False),
        ("Secret", True),
        ("credential", False),
        ("CrEdEnTiAl", True),
        ("ACCESS-TOKEN", False),
        ("API KEY", True),
        ("private key", False),
        ("refresh-token", True),
        ("auth_token", False),
        ("bearer-token", True),
        ("session token", False),
        ("api_secret", True),
        ("database-password", False),
        ("ssh private key", True),
        ("BearerToken", False),
        ("SSHPrivateKey", True),
        ("client.secret", False),
        ("ACCESS.TOKEN", True),
        ("api.key", False),
        ("PRIVATE/KEY", True),
        ("refresh/token", False),
        ("DATABASE:PASSWORD", True),
        ("authorization_header", False),
        ("TOKEN_VALUE", True),
        ("api_key_value", False),
        ("ClientSecretValue", True),
        ("bearer_token_value", False),
        ("requestAuthorizationHeader", True),
        ("credentials", False),
        ("CLIENT_CREDENTIALS", True),
        ("service_account_credentials", False),
        ("Secrets", True),
        ("tokens", False),
        ("API_KEYS", True),
        ("CLIENT\u2022SECRET", False),
        ("ＡＰＩ．ＫＥＹ", True),
    ),
)
def test_continuation_snapshot_rejects_credential_key_aliases(
    key: str,
    nested: bool,
) -> None:
    """Reject credential-bearing keys across case and word boundaries."""
    _assert_continuation_snapshot_key_rejected(key, nested)


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize(
    "key",
    (
        "BEARERTOKEN",
        "bearertoken",
        "SESSIONTOKEN",
        "DATABASEPASSWORD",
        "SSHPRIVATEKEY",
        "REQUESTAUTHORIZATIONHEADER",
        "OAUTHCLIENTSECRET",
        "AWSSECRET",
        "ＢＥＡＲＥＲＴＯＫＥＮ",
        "ｒｅｑｕｅｓｔａｕｔｈｏｒｉｚａｔｉｏｎｈｅａｄｅｒ",
    ),
)
def test_continuation_snapshot_rejects_compact_credential_keys(
    key: str,
    nested: bool,
) -> None:
    """Reject compact credential compounds at every recursive depth."""
    _assert_continuation_snapshot_key_rejected(key, nested)


def test_continuation_snapshot_preserves_noncredential_near_miss_keys() -> (
    None
):
    """Preserve safe metadata keys that merely contain credential words."""
    payload: dict[str, JsonValue] = {
        "token_count": 42,
        "authorization_url": "https://example.test/authorize",
        "client_secret_configured": False,
        "secret_store_revision": "revision-1",
        "credential_type": "oauth",
        "tokenizer": {
            "credentialed_request_count": 0,
            "bearer_token_count": 0,
            "database_password_policy": "policy-1",
            "ssh_private_key_fingerprint": "sha256:public-metadata",
        },
        "TokenCount": 42,
        "AuthorizationURL": "https://example.test/authorize",
        "ClientSecretConfigured": False,
        "CredentialType": "oauth",
        "SecretStoreRevision": "revision-1",
        "Tokenizer": {"CredentialedRequestCount": 0},
    }

    snapshot = ContinuationSnapshot(
        snapshot_kind="responses.v1",
        revision_binding=_binding(),
        model_call_id=ModelCallId("call-1"),
        provider_idempotency_key=ProviderIdempotencyKey("key-1"),
        payload=payload,
    )

    assert snapshot.payload == payload


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize(
    "key",
    (
        "completion_tokens",
        "prompt_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "max_tokens",
    ),
)
def test_continuation_snapshot_preserves_numeric_token_metrics(
    key: str,
    nested: bool,
) -> None:
    """Preserve explicit numeric provider token-usage metadata."""
    value: int | float = 1.5 if key == "total_tokens" else 3
    payload: dict[str, JsonValue]
    if nested:
        payload = {"usage": {key: value}}
    else:
        payload = {key: value}

    snapshot = ContinuationSnapshot(
        snapshot_kind="responses.v1",
        revision_binding=_binding(),
        model_call_id=ModelCallId("call-1"),
        provider_idempotency_key=ProviderIdempotencyKey("key-1"),
        payload=payload,
    )

    assert snapshot.payload == payload


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("completion_tokens", "PRIVATE_METRIC_VALUE_SENTINEL"),
        ("prompt_tokens", ["PRIVATE_METRIC_VALUE_SENTINEL"]),
        ("input_tokens", {"value": "PRIVATE_METRIC_VALUE_SENTINEL"}),
        ("output_tokens", True),
        ("total_tokens", -1),
        ("max_tokens", float("inf")),
        ("max_tokens", float("nan")),
        ("tokens", 3),
        ("secrets", 3),
        ("api_keys", 3),
    ),
)
def test_continuation_snapshot_rejects_unsafe_token_metric_values(
    key: str,
    value: object,
    nested: bool,
) -> None:
    """Reject nonnumeric metric payloads and unqualified sensitive keys."""
    private_value = "PRIVATE_METRIC_VALUE_SENTINEL"
    if nested:
        payload = {"usage": {key: value}}
        expected_path = "continuation_snapshot.payload.value.key"
    else:
        payload = {key: value}
        expected_path = "continuation_snapshot.payload.key"

    with pytest.raises(InputValidationError) as captured:
        ContinuationSnapshot(
            snapshot_kind="responses.v1",
            revision_binding=_binding(),
            model_call_id=ModelCallId("call-1"),
            provider_idempotency_key=ProviderIdempotencyKey("key-1"),
            payload=cast(dict[str, JsonValue], payload),
        )

    assert captured.value.code is InputErrorCode.SNAPSHOT_SECRET_PROHIBITED
    assert captured.value.path == expected_path
    _assert_payload_diagnostic_redacted(captured.value, private_value)


def test_continuation_payload_failures_never_disclose_provider_content() -> (
    None
):
    """Keep nested payload diagnostics structural and content-free."""
    private_key = "PRIVATE_PROVIDER_KEY_SENTINEL"
    private_value = "PRIVATE_PROVIDER_VALUE_SENTINEL"
    private_invalid_key = "PRIVATE_INVALID_KEY_SENTINEL"
    private_number_key = "PRIVATE_NUMBER_KEY_SENTINEL"
    invalid_payloads = (
        {private_key: {"password": private_value}},
        {private_key: {private_invalid_key: {private_value}}},
        {private_key: [{private_invalid_key: {private_value}}]},
        {private_key: {private_number_key: float("inf")}},
        {private_key: {1: private_value}},
        {private_key: {"e\u0301": 1, "é": 2}},
    )

    for payload in invalid_payloads:
        with pytest.raises(InputValidationError) as captured:
            ContinuationSnapshot(
                snapshot_kind="responses.v1",
                revision_binding=_binding(),
                model_call_id=ModelCallId("call-1"),
                provider_idempotency_key=ProviderIdempotencyKey("key-1"),
                payload=cast(dict[str, JsonValue], payload),
            )
        _assert_payload_diagnostic_redacted(
            captured.value,
            private_key,
            private_value,
            private_invalid_key,
            private_number_key,
            "password",
            "e\u0301",
            "é",
        )


def test_question_variants_validate_defaults_and_constraints() -> None:
    """Validate question-specific defaults, choices, and constraints."""
    text = TextQuestion(
        question_id=QuestionId("name"),
        prompt="Name?",
        required=True,
        header="Profile",
        help_text="Enter a display name.",
        presentation_hint=PresentationHint.SINGLE_LINE,
        default_value="Ada",
        constraints=TextValidationConstraints(
            minimum_length=2,
            maximum_length=8,
        ),
    )
    multiline = MultilineTextQuestion(
        question_id=QuestionId("notes"),
        prompt="Notes?",
        required=False,
        default_value="one\r\ntwo\rthree",
        constraints=TextValidationConstraints(maximum_length=20),
    )
    single = SingleSelectionQuestion(
        question_id=QuestionId("single"),
        prompt="Choose one.",
        required=True,
        choices=_choices(),
        recommended_choice=ChoiceValue("one"),
        default_value=ChoiceValue("two"),
    )
    multiple = MultipleSelectionQuestion(
        question_id=QuestionId("multiple"),
        prompt="Choose several.",
        required=True,
        choices=_choices(),
        allow_other=True,
        recommended_choice=ChoiceValue("two"),
        default_value=(ChoiceValue("one"),),
        constraints=SelectionValidationConstraints(minimum=1, maximum=3),
    )
    default_multiple = MultipleSelectionQuestion(
        question_id=QuestionId("default-multiple"),
        prompt="Choose several.",
        required=False,
        choices=_choices(),
    )

    assert text.default_value == "Ada"
    assert multiline.default_value == "one\ntwo\nthree"
    assert single.recommended_choice == "one"
    assert multiple.default_value == ("one",)
    assert default_multiple.constraints.maximum == 2

    with pytest.raises(InputValidationError):
        Choice(value=ChoiceValue("other"), label="Other")
    with pytest.raises(InputValidationError):
        TextValidationConstraints(minimum_length=2, maximum_length=1)
    with pytest.raises(InputValidationError):
        SelectionValidationConstraints(minimum=2, maximum=1)
    with pytest.raises(InputValidationError):
        ConfirmationQuestion(
            question_id=QuestionId("bad-hint"),
            prompt="Continue?",
            required=True,
            presentation_hint=cast(PresentationHint, "radio"),
        )
    with pytest.raises(InputValidationError):
        TextQuestion(
            question_id=QuestionId("bad-default"),
            prompt="Name?",
            required=True,
            default_value="x",
            constraints=TextValidationConstraints(minimum_length=2),
        )
    with pytest.raises(InputValidationError):
        MultilineTextQuestion(
            question_id=QuestionId("bad-default"),
            prompt="Notes?",
            required=False,
            default_value="long",
            constraints=TextValidationConstraints(maximum_length=3),
        )
    with pytest.raises(InputValidationError):
        TextQuestion(
            question_id=QuestionId("bad-constraints"),
            prompt="Name?",
            required=False,
            constraints=cast(TextValidationConstraints, object()),
        )
    with pytest.raises(InputValidationError):
        TextQuestion(
            question_id=QuestionId("wide-constraints"),
            prompt="Name?",
            required=False,
            constraints=TextValidationConstraints(maximum_length=4_097),
        )


@pytest.mark.parametrize(
    "choices",
    [
        cast(tuple[Choice, ...], [*_choices()]),
        (Choice(value=ChoiceValue("one"), label="One"),),
        cast(tuple[Choice, ...], (_choices()[0], object())),
        (
            Choice(value=ChoiceValue("same"), label="One"),
            Choice(value=ChoiceValue("same"), label="Two"),
        ),
    ],
)
def test_selection_questions_reject_invalid_choice_collections(
    choices: tuple[Choice, ...],
) -> None:
    """Reject mutable, undersized, untyped, and duplicate choices."""
    with pytest.raises(InputValidationError):
        SingleSelectionQuestion(
            question_id=QuestionId("choice"),
            prompt="Choose.",
            required=True,
            choices=choices,
        )


def test_selection_questions_reject_invalid_references() -> None:
    """Reject invalid recommendations, defaults, and cardinalities."""
    with pytest.raises(InputValidationError):
        SingleSelectionQuestion(
            question_id=QuestionId("bad-recommendation"),
            prompt="Choose.",
            required=True,
            choices=_choices(),
            recommended_choice=ChoiceValue("unknown"),
        )
    with pytest.raises(InputValidationError):
        SingleSelectionQuestion(
            question_id=QuestionId("bad-default"),
            prompt="Choose.",
            required=True,
            choices=_choices(),
            default_value=ChoiceValue("unknown"),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionQuestion(
            question_id=QuestionId("bad-constraints"),
            prompt="Choose.",
            required=False,
            choices=_choices(),
            constraints=cast(SelectionValidationConstraints, object()),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionQuestion(
            question_id=QuestionId("too-wide"),
            prompt="Choose.",
            required=False,
            choices=_choices(),
            constraints=SelectionValidationConstraints(maximum=3),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionQuestion(
            question_id=QuestionId("bad-recommendation"),
            prompt="Choose.",
            required=False,
            choices=_choices(),
            recommended_choice=ChoiceValue("unknown"),
            constraints=SelectionValidationConstraints(maximum=2),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionQuestion(
            question_id=QuestionId("mutable-default"),
            prompt="Choose.",
            required=False,
            choices=_choices(),
            default_value=cast(tuple[ChoiceValue, ...], [ChoiceValue("one")]),
            constraints=SelectionValidationConstraints(maximum=2),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionQuestion(
            question_id=QuestionId("duplicate-default"),
            prompt="Choose.",
            required=False,
            choices=_choices(),
            default_value=(ChoiceValue("one"), ChoiceValue("one")),
            constraints=SelectionValidationConstraints(maximum=2),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionQuestion(
            question_id=QuestionId("empty-default"),
            prompt="Choose.",
            required=True,
            choices=_choices(),
            default_value=(),
            constraints=SelectionValidationConstraints(maximum=2),
        )


def test_answer_variants_reject_invalid_tagged_values() -> None:
    """Validate answer provenance and tagged selection structure."""
    text = TextAnswer(
        question_id=QuestionId("text"),
        provenance=AnswerProvenance.HUMAN,
        value="Ada",
    )
    multiline = MultilineTextAnswer(
        question_id=QuestionId("notes"),
        provenance=AnswerProvenance.HUMAN,
        value="one\r\ntwo",
    )
    single = SingleSelectionAnswer(
        question_id=QuestionId("single"),
        provenance=AnswerProvenance.HUMAN,
        value=SelectedChoice(value=ChoiceValue("one")),
    )
    assert text.value == "Ada"
    assert multiline.value == "one\ntwo"
    assert single.value.kind.value == "selected_choice"

    with pytest.raises(InputValidationError):
        ConfirmationAnswer(
            question_id=QuestionId("confirm"),
            provenance=cast(AnswerProvenance, "human"),
            value=True,
        )
    with pytest.raises(InputValidationError):
        SingleSelectionAnswer(
            question_id=QuestionId("single"),
            provenance=AnswerProvenance.HUMAN,
            value=cast(SelectedChoice, object()),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionAnswer(
            question_id=QuestionId("multiple"),
            provenance=AnswerProvenance.HUMAN,
            values=cast(tuple[SelectedChoice, ...], []),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionAnswer(
            question_id=QuestionId("multiple"),
            provenance=AnswerProvenance.HUMAN,
            values=tuple(
                SelectedChoice(value=ChoiceValue(f"choice-{index}"))
                for index in range(21)
            ),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionAnswer(
            question_id=QuestionId("multiple"),
            provenance=AnswerProvenance.HUMAN,
            values=(
                SelectedChoice(value=ChoiceValue("one")),
                SelectedChoice(value=ChoiceValue("one")),
            ),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionAnswer(
            question_id=QuestionId("multiple"),
            provenance=AnswerProvenance.HUMAN,
            values=cast(tuple[SelectedChoice, ...], (object(),)),
        )
    with pytest.raises(InputValidationError):
        MultipleSelectionAnswer(
            question_id=QuestionId("multiple"),
            provenance=AnswerProvenance.HUMAN,
            values=(FreeFormOther(text="first"), FreeFormOther(text="second")),
        )


def test_resolution_and_result_values_validate_structure() -> None:
    """Validate terminal resolutions and continuation result correlation."""
    answer = ConfirmationAnswer(
        question_id=QuestionId("confirm"),
        provenance=AnswerProvenance.HUMAN,
        value=True,
    )
    resolution = AnsweredResolution(
        request_id=InputRequestId("request-1"),
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        resolved_at=_NOW,
        answers=(answer,),
    )
    result = InputAnsweredResult(
        request_id=resolution.request_id,
        provenance=resolution.provenance,
        resolved_at=resolution.resolved_at,
        answers=resolution.answers,
    )
    continuation = ResumeInputContinuation(
        request_id=resolution.request_id,
        result=result,
    )

    assert continuation.result.kind is InputResultKind.ANSWERED
    assert (
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ).scope
        is CancellationScope.REQUEST
    )
    assert (
        InputDeclinedResult(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
        ).kind
        is InputResultKind.DECLINED
    )

    with pytest.raises(InputValidationError):
        DeclinedResolution(
            request_id=InputRequestId("request-1"),
            provenance=cast(AnswerProvenance, "human"),
            resolved_at=_NOW,
        )
    with pytest.raises(InputValidationError):
        AnsweredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=cast(tuple[ConfirmationAnswer, ...], [answer]),
        )
    with pytest.raises(InputValidationError):
        AnsweredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=cast(tuple[ConfirmationAnswer, ...], (object(),)),
        )
    with pytest.raises(InputValidationError):
        AnsweredResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(answer, answer),
        )
    with pytest.raises(InputValidationError):
        CancelledResolution(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            scope=cast(CancellationScope, "request"),
        )
    with pytest.raises(InputValidationError):
        InputAnsweredResult(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=cast(tuple[ConfirmationAnswer, ...], [answer]),
        )
    with pytest.raises(InputValidationError):
        InputAnsweredResult(
            request_id=InputRequestId("request-1"),
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(answer, answer),
        )
    with pytest.raises(InputValidationError):
        InputDeclinedResult(
            request_id=InputRequestId("request-1"),
            provenance=cast(AnswerProvenance, "human"),
            resolved_at=_NOW,
        )
    with pytest.raises(InputValidationError):
        ResumeInputContinuation(
            request_id=InputRequestId("request-1"),
            result=cast(InputAnsweredResult, object()),
        )
    with pytest.raises(InputValidationError):
        ResumeInputContinuation(
            request_id=InputRequestId("other-request"),
            result=result,
        )
    with pytest.raises(InputValidationError):
        TerminateInputContinuation(
            request_id=InputRequestId("request-1"),
            status=cast(ResolutionStatus, "expired"),
        )


def test_request_enforces_lifecycle_and_aggregate_invariants() -> None:
    """Reject malformed request structure and inconsistent terminal state."""
    request = create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=_origin(),
        mode=RequirementMode.ADVISORY,
        reason="A decision is required.",
        questions=(_question(),),
        created_at=_NOW,
    )
    assert request.advisory_wait_seconds == 60
    assert request.advisory_deadline is None
    assert not request.required

    terminal_resolution = DeclinedResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW,
    )
    terminal = _request(
        state=RequestState.DECLINED,
        revision=2,
        resolution=terminal_resolution,
    )
    assert terminal.resolution is terminal_resolution

    invalid_values: tuple[dict[str, object], ...] = (
        {"origin": object()},
        {"mode": "required"},
        {"questions": [_question()]},
        {"questions": ()},
        {"questions": (_question(),) * 4},
        {"questions": (object(),)},
        {
            "questions": (
                _question(),
                ConfirmationQuestion(
                    question_id=QuestionId("confirm"),
                    prompt="Again?",
                    required=False,
                ),
            )
        },
        {"state": "created"},
        {"state": RequestState.DECLINED},
    )
    base: dict[str, object] = {
        "request_id": InputRequestId("request-1"),
        "continuation_id": ContinuationId("continuation-1"),
        "origin": _origin(),
        "mode": RequirementMode.REQUIRED,
        "reason": "A decision is required.",
        "questions": (_question(),),
        "created_at": _NOW,
    }
    for overrides in invalid_values:
        with pytest.raises(InputValidationError):
            InputRequest(**cast(dict[str, object], base | overrides))

    with pytest.raises(InputValidationError):
        InputRequest(
            **cast(dict[str, object], base),
            advisory_wait_seconds=60,
        )
    with pytest.raises(InputValidationError):
        InputRequest(
            **cast(dict[str, object], base),
            advisory_deadline=_NOW + timedelta(seconds=60),
        )
    with pytest.raises(InputValidationError):
        InputRequest(
            **cast(dict[str, object], base),
            state=RequestState.DECLINED,
            resolution=DeclinedResolution(
                request_id=InputRequestId("other"),
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
            ),
        )
    with pytest.raises(InputValidationError):
        InputRequest(
            **cast(dict[str, object], base),
            state=RequestState.EXPIRED,
            resolution=terminal_resolution,
        )
    with pytest.raises(InputValidationError):
        InputRequest(
            **cast(
                dict[str, object],
                base
                | {
                    "questions": (
                        MultilineTextQuestion(
                            question_id=QuestionId("huge"),
                            prompt="Text?",
                            required=False,
                            default_value="x" * 32_769,
                        ),
                    )
                },
            ),
        )


def test_advisory_hydration_requires_request_owned_deadline() -> None:
    """Reject missing, forged, or unreached advisory lifecycle timing."""
    created = _request(mode=RequirementMode.ADVISORY)
    deadline = _NOW + timedelta(seconds=60)
    base: dict[str, object] = {
        "request_id": created.request_id,
        "continuation_id": created.continuation_id,
        "origin": created.origin,
        "mode": created.mode,
        "reason": created.reason,
        "questions": created.questions,
        "created_at": created.created_at,
        "advisory_wait_seconds": created.advisory_wait_seconds,
    }

    invalid_requests: tuple[dict[str, object], ...] = (
        {"advisory_deadline": deadline},
        {
            "state": RequestState.PENDING,
            "state_revision": StateRevision(1),
        },
        {
            "state": RequestState.PENDING,
            "state_revision": StateRevision(1),
            "advisory_deadline": deadline - timedelta(microseconds=1),
        },
        {
            "state": RequestState.PENDING,
            "state_revision": StateRevision(1),
            "advisory_deadline": deadline.replace(tzinfo=None),
        },
        {
            "state": RequestState.TIMED_OUT,
            "state_revision": StateRevision(2),
            "resolution": TimedOutResolution(
                request_id=created.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=deadline,
            ),
        },
        {
            "state": RequestState.TIMED_OUT,
            "state_revision": StateRevision(2),
            "advisory_deadline": deadline,
            "resolution": TimedOutResolution(
                request_id=created.request_id,
                provenance=AnswerProvenance.POLICY,
                resolved_at=deadline - timedelta(microseconds=1),
            ),
        },
    )
    for invalid in invalid_requests:
        with pytest.raises(InputValidationError):
            InputRequest(**cast(dict[str, object], base | invalid))

    pending = InputRequest(
        **cast(dict[str, object], base),
        state=RequestState.PENDING,
        state_revision=StateRevision(1),
        advisory_deadline=deadline,
    )
    terminal = InputRequest(
        **cast(dict[str, object], base),
        state=RequestState.TIMED_OUT,
        state_revision=StateRevision(2),
        advisory_deadline=deadline,
        resolution=TimedOutResolution(
            request_id=created.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=deadline,
        ),
    )

    assert pending.advisory_deadline == deadline
    assert terminal.resolution is not None


def test_terminal_request_hydration_revalidates_semantic_invariants() -> None:
    """Reject forged revisions, terminal timing, modes, and answers."""
    request = _request()
    base = {
        "request_id": request.request_id,
        "continuation_id": request.continuation_id,
        "origin": request.origin,
        "mode": request.mode,
        "reason": request.reason,
        "questions": request.questions,
        "created_at": request.created_at,
    }
    declined = DeclinedResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
    )
    wrong_answer = AnsweredResolution(
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
    required_timeout = TimedOutResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW + timedelta(seconds=60),
    )
    invalid_states: tuple[dict[str, object], ...] = (
        {"state_revision": StateRevision(99)},
        {
            "state": RequestState.PENDING,
            "state_revision": StateRevision(99),
        },
        {
            "state": RequestState.DECLINED,
            "state_revision": StateRevision(99),
            "resolution": declined,
        },
        {
            "state": RequestState.TIMED_OUT,
            "state_revision": StateRevision(2),
            "resolution": required_timeout,
        },
        {
            "state": RequestState.DECLINED,
            "state_revision": StateRevision(2),
            "resolution": DeclinedResolution(
                request_id=request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW - timedelta(seconds=1),
            ),
        },
        {
            "state": RequestState.ANSWERED,
            "state_revision": StateRevision(2),
            "resolution": wrong_answer,
        },
        {
            "state": RequestState.EXPIRED,
            "state_revision": StateRevision(2),
            "resolution": declined,
        },
    )
    for invalid in invalid_states:
        with pytest.raises(InputValidationError):
            InputRequest(**cast(dict[str, object], base | invalid))


def test_interaction_snapshot_and_continuation_values_validate_types() -> None:
    """Validate snapshot roots and simple continuation values."""
    request = _request()
    assert InteractionSnapshot(request=request).version == 1
    required = InputRequiredResult(
        request_id=request.request_id,
        continuation_id=request.continuation_id,
        detached_resumption_available=True,
    )
    assert required.kind is InputResultKind.INPUT_REQUIRED
    assert (
        TerminateInputContinuation(
            request_id=request.request_id,
            status=ResolutionStatus.EXPIRED,
        ).status
        is ResolutionStatus.EXPIRED
    )

    with pytest.raises(InputValidationError):
        InteractionSnapshot(request=request, version=2)
    with pytest.raises(InputValidationError):
        InteractionSnapshot(request=request, version=True)
    with pytest.raises(InputValidationError):
        InteractionSnapshot(request=cast(InputRequest, object()))
    with pytest.raises(InputValidationError):
        InputRequiredResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            detached_resumption_available=cast(bool, 1),
        )
