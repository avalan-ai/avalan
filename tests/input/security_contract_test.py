"""Exercise interaction-class security separation."""

from asyncio import run
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from json import dumps
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast

sys_path.append(str(Path(__file__).parents[1] / "interaction" / "stores"))

import interaction_memory_store_test as memory_support  # noqa: E402
import interaction_pgsql_store_test as durable_support  # noqa: E402
import pytest
from pgsql_support import FakePgsqlDatabase  # noqa: E402

import avalan.sdk as sdk_module
from avalan.cli.interaction_renderer import _literal_text
from avalan.event import InteractionLifecyclePayload
from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    Choice,
    ChoiceValue,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputCodecError,
    InputErrorCode,
    InputRequest,
    InputRequestId,
    InputValidationError,
    InteractionActor,
    InteractionClass,
    InteractionCorrelation,
    InteractionPolicy,
    InteractionRecord,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    RunId,
    ScopedInteractionLookup,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StreamSessionId,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TextAnswer,
    TextQuestion,
    TurnId,
    UserId,
    create_input_request,
    decode_input_request,
    decode_input_resolution,
    encode_input_request,
    encode_input_resolution,
)
from avalan.interaction.a2a import (  # noqa: E402
    A2AInputRequestMetadata,
    a2a_input_request_text,
    encode_a2a_input_request_metadata,
)
from avalan.interaction.security import (  # noqa: E402
    _has_unnegated_collection_intent,
    enforce_task_input_request_policy,
)
from avalan.interaction.stores.pgsql import PgsqlInteractionStorePolicy
from avalan.server.mcp_session import (  # noqa: E402
    MCPFormErrorCode,
    MCPFormSessionError,
    project_mcp_form_params,
)


def test_requirement_input_n_015() -> None:
    """Keep task input separate from approval, steering, and authentication."""
    origin = ExecutionOrigin(
        run_id=RunId("run"),
        turn_id=TurnId("turn"),
        agent_id=AgentId("agent"),
        branch_id=BranchId("branch"),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://safe",
            agent_definition_revision="r1",
            operation_id="op",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
    )
    request = create_input_request(
        request_id=InputRequestId("request"),
        continuation_id=ContinuationId("continuation"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Need non-secret task context.",
        questions=(
            TextQuestion(
                question_id=QuestionId("context"),
                prompt="Which public environment?",
                required=True,
            ),
        ),
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert request.interaction_class is InteractionClass.TASK_INPUT
    assert "interaction_class" not in {
        field.name
        for field in InputRequest.__dataclass_fields__.values()
        if field.init
    }
    forged = encode_input_request(request)
    for prohibited in (
        InteractionClass.ACTION_APPROVAL,
        InteractionClass.STEERING,
        InteractionClass.AUTHENTICATION,
    ):
        forged["interaction_class"] = prohibited.value
        with pytest.raises(InputCodecError):
            decode_input_request(forged)


def test_prohibited_tags_are_rejected_without_scanning_free_text() -> None:
    """Reject explicit secret semantics while leaving text to host policy."""
    origin = ExecutionOrigin(
        run_id=RunId("run"),
        turn_id=TurnId("turn"),
        agent_id=AgentId("agent"),
        branch_id=BranchId("branch"),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://safe",
            agent_definition_revision="r1",
            operation_id="op",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
    )
    request = create_input_request(
        request_id=InputRequestId("request"),
        continuation_id=ContinuationId("continuation"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason="Explain the boundary without submitting a secret.",
        questions=(
            TextQuestion(
                question_id=QuestionId("context"),
                prompt="Explain why password and token collection is unsafe.",
                required=True,
            ),
        ),
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
    )
    wire = encode_input_request(request)

    assert decode_input_request(wire) == request
    questions = wire["questions"]
    assert isinstance(questions, list)
    question = questions[0]
    assert isinstance(question, dict)
    for field_name, tag in (
        ("kind", "password"),
        ("kind", "api_key"),
        ("kind", "token"),
        ("kind", "private_key"),
        ("kind", "payment"),
        ("kind", "mfa"),
        ("semantic_type", "authentication_challenge"),
    ):
        forged_question = dict(question)
        forged_question[field_name] = tag
        forged = dict(wire)
        forged["questions"] = [forged_question]
        with pytest.raises(InputCodecError) as error:
            decode_input_request(forged)
        assert error.value.code is InputErrorCode.PROHIBITED_INPUT


def _security_origin(
    *,
    run_id: str = "security-run",
    user_id: str = "security-owner",
) -> ExecutionOrigin:
    """Return one deterministic security-test origin."""
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId("security-turn"),
        agent_id=AgentId("security-agent"),
        branch_id=BranchId("security-branch"),
        model_call_id=ModelCallId("security-model-call"),
        stream_session_id=StreamSessionId("security-stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://security",
            agent_definition_revision="r1",
            operation_id="security",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
        principal=PrincipalScope(user_id=UserId(user_id)),
    )


def _security_request(
    name: str,
    *questions: TextQuestion | MultilineTextQuestion | SingleSelectionQuestion,
    origin: ExecutionOrigin | None = None,
    reason: str = "Need public task context.",
    created_at: datetime = datetime(2026, 7, 24, tzinfo=UTC),
) -> InputRequest:
    """Return one deterministic task-input request."""
    return create_input_request(
        request_id=InputRequestId(name),
        continuation_id=ContinuationId(f"continuation-{name}"),
        origin=origin or _security_origin(),
        mode=RequirementMode.REQUIRED,
        reason=reason,
        questions=questions
        or (
            TextQuestion(
                question_id=QuestionId("context"),
                prompt="Which public environment?",
                required=True,
            ),
        ),
        created_at=created_at,
        continuation_ttl_seconds=600,
    )


def _answered(
    request: InputRequest,
    value: str,
    *,
    provenance: AnswerProvenance = AnswerProvenance.HUMAN,
) -> AnsweredResolution:
    """Return one typed text resolution."""
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=provenance,
        resolved_at=request.created_at + timedelta(seconds=1),
        answers=(
            TextAnswer(
                question_id=request.questions[0].question_id,
                provenance=provenance,
                value=value,
            ),
        ),
    )


def test_requirement_input_n_016() -> None:
    """Keep task answers incapable of granting action approval."""
    request = _security_request("no-approval")
    resolution = _answered(request, "yes")
    wire = encode_input_resolution(resolution)

    assert decode_input_resolution(wire) == resolution
    answers = cast(list[dict[str, object]], wire["answers"])
    assert answers[0]["value"] == "yes"
    assert set(wire) == {
        "request_id",
        "status",
        "provenance",
        "resolved_at",
        "answers",
    }
    assert not {
        "approval",
        "approved",
        "authorization",
        "authentication",
    }.intersection(wire)


def test_requirement_input_n_017() -> None:
    """Reject approval responses presented as ordinary task content."""
    request = _security_request("approval-response")
    resolution = _answered(request, "continue")
    wire = encode_input_resolution(resolution)

    for field_name in ("approval_response", "action_approval"):
        forged = dict(wire)
        forged[field_name] = True
        with pytest.raises(InputCodecError) as error:
            decode_input_resolution(forged)
        assert error.value.code is InputErrorCode.INVALID_FORMAT
        assert "continue" not in error.value.safe_message


def test_requirement_input_n_018() -> None:
    """Reject authentication collection at the task-input boundary."""
    prompts = (
        "Enter your password.",
        "Enter your passphrase.",
        "Provide your SSH passphrase.",
        "Provide the API key.",
        "Submit the bearer token.",
        "Paste the private key.",
        "Enter payment card details.",
        "Provide the MFA code.",
        "Complete the authentication challenge.",
    )
    for index, prompt in enumerate(prompts):
        request = _security_request(
            f"authentication-{index}",
            TextQuestion(
                question_id=QuestionId(f"auth-{index}"),
                prompt=prompt,
                required=True,
            ),
        )
        with pytest.raises(InputValidationError) as error:
            enforce_task_input_request_policy(request, "host.request")
        assert error.value.code is InputErrorCode.PROHIBITED_INPUT
        assert prompt not in error.value.safe_message


@pytest.mark.parametrize(
    "prompt",
    (
        "Enter your pass\u200bword.",
        "Enter your p@ssword.",
        "Enter your p4ssword.",
        "Provide passw0rd.",
        "Type p a s s w o r d.",
        "Provide the API k\u0435y.",
        "Provide A P I k e y.",
        "Enter the one\u2011time code.",
        "Provide the sign\u2060-in code.",
        "Enter the login code.",
        "Type your phone verification code.",
        "Provide the code sent to your phone.",
        "Phone verification code",
        "Enter the account PIN.",
        "Provide the sign-in pin.",
        "Enter your PIN.",
        "Enter an authentication PIN on the map.",
        "Enter an account PIN in the map.",
    ),
)
def test_security_admission_normalizes_obfuscated_authentication(
    prompt: str,
) -> None:
    """Reject obfuscated credentials and common authentication codes."""
    request = _security_request(
        "normalized-authentication",
        TextQuestion(
            question_id=QuestionId("context"),
            prompt=prompt,
            required=True,
        ),
    )

    with pytest.raises(InputValidationError) as error:
        enforce_task_input_request_policy(request, "host.request")

    assert error.value.code is InputErrorCode.PROHIBITED_INPUT
    assert prompt not in error.value.safe_message


@pytest.mark.parametrize(
    "prompt",
    (
        "Explain why password and token collection is unsafe.",
        "Do not enter your password here.",
        "Do not enter your PIN here.",
        "Explain why login codes should never be shared.",
        "Can you explain how your password is protected?",
        "Discuss phone verification code retention policy.",
        "Place a map pin for the office.",
        "Enter a pin on the map.",
        "Enter a pin in the map.",
        "Choose the map pin color.",
        "Choose the location pin color.",
        "Provide a summary of the password policy architecture.",
        "Collect requirements for API key rotation architecture.",
    ),
)
def test_security_admission_allows_discussion_and_negation(
    prompt: str,
) -> None:
    """Allow discussion and explicit warnings that collect no secret."""
    request = _security_request(
        "authentication-discussion",
        TextQuestion(
            question_id=QuestionId("discussion"),
            prompt=prompt,
            required=True,
        ),
    )

    enforce_task_input_request_policy(request, "host.request")


def test_security_admission_preserves_clause_and_field_boundaries() -> None:
    """Bind collection intent only to its credential clause or field role."""
    attack = _security_request(
        "credential-clause",
        TextQuestion(
            question_id=QuestionId("context"),
            prompt="Explain the password policy; then enter your p@ssword.",
            required=True,
        ),
    )
    with pytest.raises(InputValidationError) as error:
        enforce_task_input_request_policy(attack, "host.request")
    assert error.value.code is InputErrorCode.PROHIBITED_INPUT

    benign = _security_request(
        "field-boundaries",
        TextQuestion(
            question_id=QuestionId("deployment"),
            prompt="Enter the deployment code name.",
            header="API key policy architecture",
            required=True,
        ),
        reason="Discuss authentication architecture.",
    )
    enforce_task_input_request_policy(benign, "host.request")


def test_security_admission_combines_collection_intent_and_auth_context() -> (
    None
):
    """Reject collection intent split across reason and question text."""
    request = _security_request(
        "authentication-context",
        TextQuestion(
            question_id=QuestionId("code"),
            prompt="Enter the code.",
            required=True,
        ),
        reason="Sign-in verification is required.",
    )

    with pytest.raises(InputValidationError) as error:
        enforce_task_input_request_policy(request, "host.request")

    assert error.value.code is InputErrorCode.PROHIBITED_INPUT


def test_security_admission_checks_all_presentation_locations() -> None:
    """Reject credential semantics in context, choices, and PIN context."""
    requests = (
        _security_request("reason-pin", reason="PIN"),
        _security_request(
            "choice-pin",
            SingleSelectionQuestion(
                question_id=QuestionId("method"),
                prompt="Choose a public method.",
                required=True,
                choices=(
                    Choice(value=ChoiceValue("pin"), label="PIN"),
                    Choice(
                        value=ChoiceValue("alternative"), label="Alternative"
                    ),
                ),
            ),
        ),
        _security_request(
            "context-pin",
            TextQuestion(
                question_id=QuestionId("context"),
                prompt="Account PIN for access.",
                required=True,
            ),
        ),
    )

    for request in requests:
        with pytest.raises(InputValidationError) as error:
            enforce_task_input_request_policy(request, "host.request")
        assert error.value.code is InputErrorCode.PROHIBITED_INPUT


def test_collection_intent_resets_at_boundaries_and_contrast() -> None:
    """Keep negation scoped to its clause while honoring later intent."""
    assert _has_unnegated_collection_intent(
        "do not share. enter the password."
    )
    assert _has_unnegated_collection_intent(
        "do not share but enter the password."
    )


def test_requirement_input_n_097() -> None:
    """Treat yes-like clarification answers only as canonical task data."""
    request = _security_request("yes-like")
    for phrase in ("yes", "continue", "do it"):
        resolution = _answered(request, phrase)
        decoded = decode_input_resolution(encode_input_resolution(resolution))
        assert isinstance(decoded, AnsweredResolution)
        answer = decoded.answers[0]
        assert isinstance(answer, TextAnswer)
        assert answer.value == phrase
        assert decoded.status is ResolutionStatus.ANSWERED
        assert decoded.provenance is AnswerProvenance.HUMAN


def test_requirement_input_n_098() -> None:
    """Refuse secret and authentication questions on public projections."""
    requests = tuple(
        _security_request(
            f"sensitive-{index}",
            TextQuestion(
                question_id=QuestionId(f"sensitive-{index}"),
                prompt=prompt,
                required=True,
            ),
        )
        for index, prompt in enumerate(
            (
                "Enter your password.",
                "Enter your passphrase.",
                "Provide your SSH passphrase.",
                "Share the API token.",
                "Paste the private key.",
                "Provide the card security code.",
                "Enter the one-time verification code.",
                "Respond to the authentication challenge.",
                "Provide your GitHub token.",
                "Enter the authenticator-app code.",
                "Provide the card expiry date.",
                "Provide the challenge response from your sign-in page.",
            )
        )
    )
    for request in requests:
        with pytest.raises(MCPFormSessionError) as mcp_error:
            project_mcp_form_params(request)
        assert mcp_error.value.code is MCPFormErrorCode.UNSAFE_REQUEST
        assert request.questions[0].prompt not in mcp_error.value.safe_message

        with pytest.raises(InputValidationError) as a2a_error:
            encode_a2a_input_request_metadata(request)
        assert a2a_error.value.code is InputErrorCode.PROHIBITED_INPUT
        assert request.questions[0].prompt not in a2a_error.value.safe_message
        with pytest.raises(InputValidationError):
            encode_a2a_input_request_metadata(
                A2AInputRequestMetadata(
                    request_id=request.request_id,
                    required=request.required,
                    questions=request.questions,
                )
            )


def test_requirement_input_n_099() -> None:
    """Treat model-authored presentation wording as untrusted data."""
    prompt = (
        "Choose a GitHub repository, authenticator app, card theme, and "
        "sign-in method. <admin> [run](file:///tmp/run) "
        "\x1b[31mred\x1b[0m."
    )
    request = _security_request(
        "untrusted-presentation",
        TextQuestion(
            question_id=QuestionId("untrusted"),
            prompt=prompt,
            required=True,
        ),
    )
    rendered = _literal_text(request.questions[0].prompt)
    fallback = a2a_input_request_text(request)

    assert request.questions[0].prompt == prompt
    assert "\x1b" not in rendered
    assert "\x1b" not in fallback
    assert "<admin>" in rendered
    assert "[run](file:///tmp/run)" in rendered
    assert "<admin>" in fallback
    assert "[run](file:///tmp/run)" in fallback


def test_requirement_input_n_100() -> None:
    """Render terminal controls and embedded instructions as inert text."""
    prompt = (
        "Review \x1b]8;;https://example.test\x07link\x1b]8;;\x07"
        "\u202e.gnp `deploy --force`"
    )
    request = _security_request(
        "literal-rendering",
        TextQuestion(
            question_id=QuestionId("literal"),
            prompt=prompt,
            required=True,
        ),
    )
    terminal = _literal_text(f"{prompt}\nsecond line")
    fallback = a2a_input_request_text(request)

    assert "\x1b" not in terminal
    assert "\x07" not in terminal
    assert "\u202e" not in terminal
    assert "\n" not in terminal
    assert "\\n" in terminal
    assert "`deploy --force`" in terminal
    assert "\x1b" not in fallback
    assert "\x07" not in fallback
    assert "\u202e" not in fallback
    assert "\\u202e" in fallback
    assert "`deploy --force`" in fallback
    assert request.questions[0].prompt == prompt


def test_requirement_input_n_101() -> None:
    """Expose only lifecycle categories and bounded operational metadata."""
    payload = InteractionLifecyclePayload.from_canonical_ids(
        request_id=InputRequestId("telemetry-request"),
        run_id=RunId("telemetry-run"),
        turn_id=TurnId("telemetry-turn"),
        agent_id=AgentId("telemetry-agent"),
        branch_id=BranchId("telemetry-branch"),
        state=RequestState.ANSWERED,
        resolution_category=ResolutionStatus.ANSWERED,
        surface="sdk",
        wait_duration_ms=17,
        validation_code=InputErrorCode.STALE_REVISION,
        duplicate=False,
        stale=True,
        provenance_category=AnswerProvenance.HUMAN,
    ).to_dict()

    assert set(payload) == {
        "request_id",
        "run_id",
        "turn_id",
        "agent_id",
        "branch_id",
        "state",
        "resolution_category",
        "surface",
        "wait_duration_ms",
        "validation_code",
        "duplicate",
        "stale",
        "provenance_category",
    }
    assert payload["state"] == RequestState.ANSWERED.value
    assert payload["resolution_category"] == ResolutionStatus.ANSWERED.value


def test_requirement_input_n_102() -> None:
    """Correlate telemetry without prompts, choices, or answers."""
    private_values = (
        "telemetry-request",
        "telemetry-run",
        "private prompt",
        "private choice",
        "private answer",
    )
    payload = InteractionLifecyclePayload.from_canonical_ids(
        request_id=InputRequestId(private_values[0]),
        run_id=RunId(private_values[1]),
        turn_id=TurnId("telemetry-turn"),
        agent_id=AgentId("telemetry-agent"),
        branch_id=BranchId("telemetry-branch"),
        state=RequestState.PENDING,
        surface="server",
    ).to_dict()
    serialized = dumps(payload, sort_keys=True)

    assert all(
        str(payload[field_name]).startswith("oid_")
        for field_name in (
            "request_id",
            "run_id",
            "turn_id",
            "agent_id",
            "branch_id",
        )
    )
    assert all(value not in serialized for value in private_values)
    assert not {"prompt", "choices", "answers"}.intersection(payload)


def test_requirement_input_n_103() -> None:
    """Encrypt, scope, retain, and delete every interaction content class."""

    async def exercise() -> None:
        database = FakePgsqlDatabase()
        cipher = durable_support._Cipher()
        store = await durable_support._store(
            database,
            cipher=cipher,
            store_policy=PgsqlInteractionStorePolicy(retention_days=1),
        )
        request = _security_request(
            "privacy-record",
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="private prompt sentinel",
                required=True,
                default_value="private default sentinel",
            ),
            SingleSelectionQuestion(
                question_id=QuestionId("choice"),
                prompt="private choice prompt",
                required=True,
                choices=(
                    Choice(
                        value=ChoiceValue("private-choice-value"),
                        label="private choice label",
                    ),
                    Choice(
                        value=ChoiceValue("public-alternative"),
                        label="Public alternative",
                    ),
                ),
                default_value=ChoiceValue("private-choice-value"),
            ),
            MultilineTextQuestion(
                question_id=QuestionId("multiline"),
                prompt="private multiline prompt",
                required=True,
            ),
            origin=durable_support._request("privacy-record").origin,
            reason="private reason sentinel",
            created_at=durable_support._NOW,
        )
        continuation = replace(
            durable_support._portable(request),
            transcript=(
                {"role": "user", "content": "private transcript sentinel"},
            ),
            observations=(
                {"kind": "tool", "value": "private observation sentinel"},
            ),
        )
        created = await store.create_durable(
            durable_support._create_command(request),
            continuation,
        )
        resolution = AnsweredResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=request.created_at + timedelta(seconds=1),
            answers=(
                TextAnswer(
                    question_id=QuestionId("text"),
                    provenance=AnswerProvenance.HUMAN,
                    value="private text answer",
                ),
                SingleSelectionAnswer(
                    question_id=QuestionId("choice"),
                    provenance=AnswerProvenance.HUMAN,
                    value=SelectedChoice(
                        value=ChoiceValue("private-choice-value")
                    ),
                ),
                MultilineTextAnswer(
                    question_id=QuestionId("multiline"),
                    provenance=AnswerProvenance.HUMAN,
                    value="private multiline\nanswer",
                ),
            ),
        )
        resolved = await store.resolve(
            ResolveInteractionCommand(
                actor=created.command.actor,
                correlation=created.record.correlation,
                expected_state_revision=created.record.request.state_revision,
                idempotency_key=ResolutionIdempotencyKey("privacy-resolution"),
                proposed_resolution=resolution,
            )
        )
        assert isinstance(resolved, ResolveInteractionApplied)
        owner = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=created.command.actor,
                correlation=created.record.correlation,
            )
        )
        assert isinstance(owner, InteractionRecord)
        stored_resolution = owner.request.resolution
        assert isinstance(stored_resolution, AnsweredResolution)
        assert stored_resolution.answers == resolution.answers
        assert stored_resolution.provenance is resolution.provenance
        intruder = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=InteractionActor(
                    principal=PrincipalScope(user_id=UserId("intruder"))
                ),
                correlation=created.record.correlation,
            )
        )
        assert intruder is None

        private_values = (
            "private prompt sentinel",
            "private default sentinel",
            "private choice label",
            "private transcript sentinel",
            "private observation sentinel",
            "private text answer",
            "private multiline",
        )
        persisted = repr(database.snapshot())
        assert all(value not in persisted for value in private_values)

        invalidated = await store.sweep(
            now=request.created_at + timedelta(minutes=11)
        )
        assert invalidated.invalidated == (request.continuation_id,)
        deleted = await store.sweep(now=request.created_at + timedelta(days=2))
        assert deleted.deleted == (request.continuation_id,)
        assert database.records == {}
        assert database.continuations == {}
        assert database.resolution_keys == {}
        assert database.outbox == {}
        assert database.branches == {}
        await store.aclose()

    run(exercise())


class _RejectingClassifier:
    """Reject submitted free-form content without echoing its value."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self.policy = policy

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return one fully correlated secret-like classification."""
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.REJECT_SECRET,
            classifier_id=self.policy.task_input_classifier_id,
            classification_id=f"classification-{request.question_id}",
            policy_revision=self.policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


def test_requirement_input_n_104() -> None:
    """Redact rejected secret-like input while preserving canonical data."""

    async def exercise() -> None:
        policy = InteractionPolicy()
        factory, _, _ = memory_support._factory(
            policy=policy,
            classifier=_RejectingClassifier(policy),
        )
        store = await factory.open()
        created = await memory_support._create(
            store,
            memory_support._text_request("secret-like"),
        )
        secret = "sk-proj-abcdefghijklmnopqrstuv"
        command = memory_support._text_answer(
            created.record,
            "secret-like-answer",
        )
        answer = cast(TextAnswer, command.proposed_resolution.answers[0])
        command = replace(
            command,
            proposed_resolution=replace(
                command.proposed_resolution,
                answers=(replace(answer, value=secret),),
            ),
        )
        rejected = await store.resolve(command)

        assert isinstance(rejected, ResolveInteractionRejected)
        assert rejected.error.code is InputErrorCode.PROHIBITED_INPUT
        assert secret not in repr(rejected.error)
        proposed = cast(
            AnsweredResolution,
            rejected.command.proposed_resolution,
        )
        proposed_answer = cast(TextAnswer, proposed.answers[0])
        assert proposed_answer.value == secret
        stored = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=created.command.actor,
                correlation=created.record.correlation,
            )
        )
        assert isinstance(stored, InteractionRecord)
        assert stored.request.state is RequestState.PENDING
        assert stored.request.resolution is None
        assert stored.request.interaction_class is InteractionClass.TASK_INPUT
        await store.aclose()

    run(exercise())


def test_requirement_input_n_105() -> None:
    """Keep human, trusted-default, and policy provenance distinguishable."""
    categories = (
        AnswerProvenance.HUMAN,
        AnswerProvenance.TRUSTED_DEFAULT,
        AnswerProvenance.POLICY,
    )
    payloads = tuple(
        InteractionLifecyclePayload.from_canonical_ids(
            request_id=InputRequestId(f"provenance-{category.value}"),
            run_id=RunId("provenance-run"),
            turn_id=TurnId("provenance-turn"),
            agent_id=AgentId("provenance-agent"),
            branch_id=BranchId("provenance-branch"),
            state=RequestState.ANSWERED,
            resolution_category=ResolutionStatus.ANSWERED,
            provenance_category=category,
        ).to_dict()
        for category in categories
    )

    assert tuple(
        payload["provenance_category"] for payload in payloads
    ) == tuple(category.value for category in categories)
    assert len({dumps(payload, sort_keys=True) for payload in payloads}) == 3


def test_forged_opaque_reference_and_wrong_scope_are_isolated() -> None:
    """Reject forged, mismatched, and cross-principal interaction access."""
    first = _security_request("opaque-first")
    second = _security_request(
        "opaque-second",
        origin=_security_origin(run_id="other-run", user_id="other-owner"),
    )
    first_correlation = InteractionCorrelation.from_request(first)
    second_correlation = InteractionCorrelation.from_request(second)
    request_ref = sdk_module._encode_correlation_ref(
        "request",
        first_correlation,
    )
    continuation_ref = sdk_module._encode_correlation_ref(
        "continuation",
        first_correlation,
    )
    other_continuation_ref = sdk_module._encode_correlation_ref(
        "continuation",
        second_correlation,
    )
    forged = f"{request_ref[:-1]}{'0' if request_ref[-1] != '0' else '1'}"

    with pytest.raises(InputValidationError) as forged_error:
        sdk_module._decode_correlation_ref(forged, "request")
    assert forged_error.value.code is InputErrorCode.INVALID_FORMAT
    assert request_ref not in forged_error.value.safe_message

    assert (
        sdk_module._decode_correlation_pair(
            cast(Any, request_ref),
            cast(Any, continuation_ref),
        )
        == first_correlation
    )
    with pytest.raises(InputValidationError) as mismatch_error:
        sdk_module._decode_correlation_pair(
            cast(Any, request_ref),
            cast(Any, other_continuation_ref),
        )
    assert mismatch_error.value.code is InputErrorCode.CORRELATION_MISMATCH
    assert continuation_ref not in mismatch_error.value.safe_message
