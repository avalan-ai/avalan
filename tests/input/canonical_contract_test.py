"""Exercise the public canonical interaction contract."""

from ast import Attribute, Call, Import, ImportFrom, Name, parse, walk
from collections.abc import Mapping
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

import avalan.model.capability as model_capability_module
from avalan.interaction import (
    RESERVED_INPUT_CAPABILITY_NAME,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CancelledResolution,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    FreeFormOther,
    HostCapabilities,
    HostHandling,
    InputAnsweredResult,
    InputCodecError,
    InputQuestion,
    InputRequest,
    InputRequestId,
    InputTimedOutResult,
    InputTransitionApplied,
    InputTransitionRejected,
    InteractionClass,
    ModelCallId,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PresentationHint,
    QuestionId,
    QuestionType,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    ResumeInputContinuation,
    RunId,
    SelectedChoice,
    SelectionValidationConstraints,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    SupersededResolution,
    TaskId,
    TextAnswer,
    TextQuestion,
    TimedOutResolution,
    TurnId,
    UnavailableResolution,
    create_input_request,
    decode_input_question,
    decode_input_request,
    encode_input_question,
    encode_input_request,
    mark_request_pending,
    project_resolution_to_model,
    resolve_request,
    semantic_request_fingerprint,
)
from avalan.interaction.state import _anchor_request_presentation
from avalan.model import (
    CapabilityBatchAccepted,
    ModelCapabilityCatalog,
    ModelCapabilityKind,
    ProviderCapabilityCall,
    ProviderCapabilitySupport,
    TaskInputCapabilityCall,
)

_CREATED_AT = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        task_id=TaskId("task-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        parent_branch_id=BranchId("branch-parent"),
        model_call_id=ModelCallId("model-call-1"),
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
    )


def _choices(*, second_label: str = "Careful") -> tuple[Choice, ...]:
    return (
        Choice(
            value=ChoiceValue("fast"),
            label="Fast",
            description="Finish sooner with fewer checks.",
        ),
        Choice(
            value=ChoiceValue("careful"),
            label=second_label,
            description="Run the complete validation set.",
        ),
    )


def _questions() -> tuple[
    ConfirmationQuestion,
    SingleSelectionQuestion,
    MultipleSelectionQuestion,
]:
    choices = _choices()
    return (
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
            default_value=True,
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("strategy"),
            prompt="Choose a strategy.",
            required=True,
            header="Strategy",
            help_text="This controls validation depth.",
            presentation_hint=PresentationHint.RADIO,
            choices=choices,
            recommended_choice=ChoiceValue("careful"),
            default_value=ChoiceValue("careful"),
        ),
        MultipleSelectionQuestion(
            question_id=QuestionId("checks"),
            prompt="Choose checks.",
            required=False,
            choices=choices,
            allow_other=True,
            default_value=(ChoiceValue("fast"),),
            constraints=SelectionValidationConstraints(
                minimum=0,
                maximum=3,
            ),
            presentation_hint=PresentationHint.CHECKBOX,
        ),
    )


def _request(
    *,
    questions: tuple[InputQuestion, ...] | None = None,
    mode: RequirementMode = RequirementMode.REQUIRED,
) -> InputRequest:
    typed_questions = _questions() if questions is None else questions
    return create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=_origin(),
        mode=mode,
        reason="A decision is required to continue safely.",
        questions=typed_questions,
        created_at=_CREATED_AT,
    )


def _pending(request: InputRequest) -> InputRequest:
    result = mark_request_pending(
        request,
        expected_state_revision=StateRevision(0),
    )
    assert isinstance(result, InputTransitionApplied)
    assert result.mutation_applied
    if request.mode is RequirementMode.ADVISORY:
        return _anchor_request_presentation(
            result.request,
            request.created_at,
        )
    return result.request


def _answered(request: InputRequest) -> AnsweredResolution:
    return AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        resolved_at=_CREATED_AT + timedelta(seconds=1),
        answers=(
            ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
            SingleSelectionAnswer(
                question_id=QuestionId("strategy"),
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("careful")),
            ),
            MultipleSelectionAnswer(
                question_id=QuestionId("checks"),
                provenance=AnswerProvenance.POLICY,
                values=(
                    SelectedChoice(value=ChoiceValue("fast")),
                    FreeFormOther(text="security"),
                ),
            ),
        ),
    )


def _attached_model_catalog() -> ModelCapabilityCatalog:
    return ModelCapabilityCatalog.create(
        support=ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            attached_resolution=True,
        )
    )


def _model_input_arguments() -> dict[str, object]:
    return {
        "mode": "required",
        "reason": "Choose the deployment region.",
        "questions": [
            {
                "question_id": "region",
                "kind": "single_selection",
                "header": "Region",
                "prompt": "Which region should be used?",
                "required": True,
                "choices": [
                    {
                        "value": "us-east",
                        "label": "US East",
                        "description": "Use the eastern region.",
                    },
                    {
                        "value": "eu-west",
                        "label": "EU West",
                        "description": "Use the western Europe region.",
                    },
                ],
                "allow_other": False,
                "recommended_choice": "us-east",
            }
        ],
    }


def test_requirement_input_n_001() -> None:
    """Expose task input as one structured model capability."""
    catalog = _attached_model_catalog()
    projection = catalog.project("openai")

    assert len(catalog.descriptors) == 1
    assert catalog.descriptors[0].kind is ModelCapabilityKind.TASK_INPUT
    assert catalog.descriptors[0].canonical_name == (
        RESERVED_INPUT_CAPABILITY_NAME
    )
    assert projection.schemas[0]["type"] == "function"
    function = cast(Mapping[str, object], projection.schemas[0]["function"])
    assert function["name"] == RESERVED_INPUT_CAPABILITY_NAME
    parameters = cast(Mapping[str, object], function["parameters"])
    assert parameters["type"] == "object"


def test_requirement_input_n_002() -> None:
    """Classify a reserved call without publishing or suspending a run."""
    catalog = _attached_model_catalog()
    classification = catalog.classify_batch(
        (
            ProviderCapabilityCall(
                call_id="provider-call-001",
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=_model_input_arguments(),
            ),
        )
    )

    assert isinstance(classification, CapabilityBatchAccepted)
    assert classification.domain_calls == ()
    assert isinstance(classification.task_input, TaskInputCapabilityCall)
    assert classification.task_input.call_id == "provider-call-001"
    assert classification.task_input.questions[0].question_id == "region"


def test_requirement_input_n_003() -> None:
    """Allow only pure dependencies and exercise runtime I/O sentinels."""
    source = Path(model_capability_module.__file__).read_text(encoding="utf-8")
    tree = parse(source)
    imported_modules = {
        name.name
        for node in walk(tree)
        if isinstance(node, Import)
        for name in node.names
    }
    imported_modules.update(
        f"{'.' * node.level}{node.module or ''}"
        for node in walk(tree)
        if isinstance(node, ImportFrom)
    )

    def call_name(call: Call) -> str | None:
        value = call.func
        parts: list[str] = []
        while isinstance(value, Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, Name):
            parts.append(value.id)
            return ".".join(reversed(parts))
        return None

    called_names = {
        name
        for node in walk(tree)
        if isinstance(node, Call)
        if (name := call_name(node)) is not None
    }
    allowed_internal_dependencies = {
        "..entities",
        "..interaction.codec",
        "..interaction.entities",
        "..interaction.error",
        "..interaction.validation",
        "..tool.name_policy",
        "..tool.parser",
        "..types",
    }
    internal_dependencies = {
        module for module in imported_modules if module.startswith(".")
    }
    assert internal_dependencies <= allowed_internal_dependencies
    assert allowed_internal_dependencies - {"..interaction.error"} <= (
        internal_dependencies
    )
    forbidden_dependency_roots = {
        "curses",
        "getpass",
        "rich",
        "sqlite3",
        "sys",
        "termios",
        "textual",
        "tty",
        "..interaction.broker",
        "..interaction.handler",
        "..interaction.store",
        "..interaction.stores",
    }
    assert imported_modules.isdisjoint(forbidden_dependency_roots)
    assert called_names.isdisjoint(
        {
            "input",
            "open",
            "sys.stdin.read",
            "sys.stdin.readline",
        }
    )

    catalog = _attached_model_catalog()
    sentinels = {
        name: patch(name, side_effect=AssertionError(name))
        for name in (
            "builtins.input",
            "getpass.getpass",
            "pathlib.Path.open",
            "sqlite3.connect",
        )
    }
    started = [sentinel.start() for sentinel in sentinels.values()]
    try:
        projection = catalog.project("openai")
        classification = catalog.classify_batch(
            (
                ProviderCapabilityCall(
                    call_id="provider-call-io-sentinel",
                    provider_name=projection.provider_name(
                        RESERVED_INPUT_CAPABILITY_NAME
                    ),
                    arguments=_model_input_arguments(),
                ),
            ),
            provider_family="openai",
        )
        assert isinstance(classification, CapabilityBatchAccepted)
        assert isinstance(classification.task_input, TaskInputCapabilityCall)
    finally:
        for sentinel in reversed(tuple(sentinels.values())):
            sentinel.stop()
    assert all(not mock.called for mock in started)


def test_requirement_input_n_004() -> None:
    """Describe every supported semantic input type."""
    question_types = {
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        ).kind,
        TextQuestion(
            question_id=QuestionId("name"),
            prompt="Name?",
            required=True,
        ).kind,
        MultilineTextQuestion(
            question_id=QuestionId("notes"),
            prompt="Notes?",
            required=False,
        ).kind,
        _questions()[1].kind,
        _questions()[2].kind,
    }
    assert question_types == set(QuestionType)


def test_requirement_input_n_005() -> None:
    """Keep stable values independent of native display labels."""
    first = _request(
        questions=(
            SingleSelectionQuestion(
                question_id=QuestionId("strategy"),
                prompt="Choose a strategy.",
                required=True,
                choices=_choices(),
            ),
        )
    )
    second = _request(
        questions=(
            SingleSelectionQuestion(
                question_id=QuestionId("strategy"),
                prompt="Choose a strategy.",
                required=True,
                choices=_choices(second_label="Thorough"),
            ),
        )
    )
    assert semantic_request_fingerprint(first) == semantic_request_fingerprint(
        second
    )


def test_requirement_input_n_006() -> None:
    """Resume an answered request with the same logical origin."""
    pending = _pending(_request())
    transition = resolve_request(
        pending,
        _answered(pending),
        expected_state_revision=StateRevision(1),
    )
    assert isinstance(transition, InputTransitionApplied)
    outcome = project_resolution_to_model(
        transition.request,
        containing_run_exists=True,
    )
    assert isinstance(outcome, ResumeInputContinuation)
    assert isinstance(outcome.result, InputAnsweredResult)
    assert transition.request.origin == pending.origin


def test_requirement_input_n_007() -> None:
    """Keep the canonical core independent of provider call syntax."""
    encoded = encode_input_request(_request())
    assert RESERVED_INPUT_CAPABILITY_NAME == "request_user_input"
    assert not ({"provider", "tool_call", "function_call"} & set(encoded))
    assert decode_input_request(encoded) == _request()


def test_requirement_input_n_008() -> None:
    """Advertise only for attached or durable host handling."""
    assert HostCapabilities(attached_resolution=True).can_advertise
    assert HostCapabilities(durable_resolution=True).can_advertise
    assert not HostCapabilities().can_advertise
    assert HostCapabilities().handling is HostHandling.UNAVAILABLE


def test_requirement_input_n_009() -> None:
    """Keep host availability independent of domain-tool permissions."""
    names = {item.name for item in fields(HostCapabilities)}
    assert names == {"attached_resolution", "durable_resolution"}


def test_requirement_input_n_010() -> None:
    """Represent every terminal outcome explicitly."""
    assert {status.value for status in ResolutionStatus} == {
        "answered",
        "declined",
        "cancelled",
        "timed_out",
        "unavailable",
        "expired",
        "superseded",
    }


def test_requirement_input_n_011() -> None:
    """Never manufacture a human answer for a timeout."""
    request = _pending(_request(mode=RequirementMode.ADVISORY))
    resolution = TimedOutResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.POLICY,
        resolved_at=_CREATED_AT + timedelta(seconds=60),
    )
    transition = resolve_request(
        request,
        resolution,
        expected_state_revision=StateRevision(1),
    )
    assert isinstance(transition, InputTransitionApplied)
    outcome = project_resolution_to_model(
        transition.request,
        containing_run_exists=True,
    )
    assert isinstance(outcome, ResumeInputContinuation)
    assert isinstance(outcome.result, InputTimedOutResult)
    assert outcome.result.provenance is AnswerProvenance.POLICY
    assert not hasattr(outcome.result, "answers")


def test_requirement_input_n_012() -> None:
    """Expose distinct interaction classes while hardcoding task input."""
    assert set(InteractionClass) == {
        InteractionClass.TASK_INPUT,
        InteractionClass.ACTION_APPROVAL,
        InteractionClass.STEERING,
        InteractionClass.AUTHENTICATION,
    }
    assert _request().interaction_class is InteractionClass.TASK_INPUT


def test_requirement_input_n_013() -> None:
    """Require a material user-facing reason for interruption."""
    request = _request()
    assert request.reason == "A decision is required to continue safely."


def test_requirement_input_n_014() -> None:
    """Support one focused question and consequential choice descriptions."""
    question = _questions()[1]
    request = _request(questions=(question,))
    assert len(request.questions) == 1
    assert all(choice.description for choice in question.choices)


def test_requirement_input_n_029() -> None:
    """Carry complete immutable correlation and lifecycle identity."""
    request = _request()
    assert request.request_id == InputRequestId("request-1")
    assert request.origin.task_id == TaskId("task-1")
    assert request.origin.model_call_id == ModelCallId("model-call-1")
    assert request.mode is RequirementMode.REQUIRED
    assert request.created_at.tzinfo is UTC
    assert request.state is RequestState.CREATED


def test_requirement_input_n_030() -> None:
    """Carry the required fields for every question."""
    question = _questions()[0]
    assert question.question_id == QuestionId("confirm")
    assert question.prompt == "Continue?"
    assert question.kind is QuestionType.CONFIRMATION
    assert question.required


def test_requirement_input_n_031() -> None:
    """Round-trip every optional request presentation field."""
    request = _request()
    assert decode_input_request(encode_input_request(request)) == request
    selection = request.questions[1]
    assert isinstance(selection, SingleSelectionQuestion)
    assert selection.header == "Strategy"
    assert selection.help_text
    assert selection.recommended_choice == ChoiceValue("careful")
    assert selection.default_value == ChoiceValue("careful")
    assert selection.presentation_hint is PresentationHint.RADIO


def test_requirement_input_n_032() -> None:
    """Key answers by stable IDs rather than displayed wording."""
    answer = _answered(_request()).answers[1]
    reason = encode_input_request(_request())["reason"]
    assert answer.question_id == QuestionId("strategy")
    assert isinstance(reason, str)
    assert "Choose a strategy." not in reason


def test_requirement_input_n_033() -> None:
    """Keep question variants closed to the supported flat types."""
    for question in (
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
        ),
        TextQuestion(
            question_id=QuestionId("text"),
            prompt="Text?",
            required=True,
        ),
        MultilineTextQuestion(
            question_id=QuestionId("multiline"),
            prompt="Details?",
            required=False,
        ),
        _questions()[1],
        _questions()[2],
    ):
        assert (
            decode_input_question(encode_input_question(question)) == question
        )
    invalid = encode_input_question(_questions()[0])
    invalid["kind"] = "nested_form"
    with pytest.raises(InputCodecError):
        decode_input_question(invalid)


def test_requirement_input_n_034() -> None:
    """Give every choice a stable value and readable label."""
    assert [(choice.value, choice.label) for choice in _choices()] == [
        (ChoiceValue("fast"), "Fast"),
        (ChoiceValue("careful"), "Careful"),
    ]


def test_requirement_input_n_035() -> None:
    """Allow display labels to change without changing returned values."""
    original = _choices()[1]
    changed = _choices(second_label="Thorough")[1]
    assert original.value == changed.value
    assert original.label != changed.label


def test_requirement_input_n_036() -> None:
    """Represent free-form Other as a tagged alternative."""
    other = FreeFormOther(text="A custom safe choice")
    assert other.kind.value == "free_form_other"
    assert not isinstance(other, SelectedChoice)


def test_requirement_input_n_037() -> None:
    """Encode structured choices before opening a free-form alternative."""
    encoded = encode_input_question(_questions()[2])
    encoded_choices = encoded["choices"]
    assert isinstance(encoded_choices, list)
    assert encoded_choices
    assert encoded["allow_other"] is True
    labels: set[object] = set()
    for choice in encoded_choices:
        assert isinstance(choice, dict)
        labels.add(choice["label"])
    assert "Other" not in labels


def test_requirement_input_n_038() -> None:
    """Express compact and native-control preferences as typed hints."""
    assert {hint.value for hint in PresentationHint} == {
        "compact",
        "expanded",
        "radio",
        "list",
        "checkbox",
        "single_line",
        "editor",
    }


def test_requirement_input_n_039() -> None:
    """Keep hints advisory and outside semantic fingerprinting."""
    base = _questions()[1]
    alternate = SingleSelectionQuestion(
        question_id=base.question_id,
        prompt=base.prompt,
        required=base.required,
        choices=base.choices,
        allow_other=base.allow_other,
        recommended_choice=base.recommended_choice,
        default_value=base.default_value,
        presentation_hint=PresentationHint.LIST,
    )
    assert semantic_request_fingerprint(
        _request(questions=(base,))
    ) == semantic_request_fingerprint(_request(questions=(alternate,)))


def test_requirement_input_n_040() -> None:
    """Resolve each request once into one of seven terminal states."""
    constructors = (
        DeclinedResolution,
        CancelledResolution,
        UnavailableResolution,
        ExpiredResolution,
        SupersededResolution,
    )
    for constructor in constructors:
        pending = _pending(_request())
        resolution = constructor(
            request_id=pending.request_id,
            provenance=AnswerProvenance.POLICY,
            resolved_at=_CREATED_AT + timedelta(seconds=1),
        )
        result = resolve_request(
            pending,
            resolution,
            expected_state_revision=StateRevision(1),
        )
        assert isinstance(result, InputTransitionApplied)
        assert result.request.state.value == resolution.status.value
        replay = resolve_request(
            result.request,
            resolution,
            expected_state_revision=StateRevision(2),
        )
        assert isinstance(replay, InputTransitionApplied)
        assert not replay.mutation_applied
    assert ResolutionStatus.ANSWERED.value == "answered"
    assert ResolutionStatus.TIMED_OUT.value == "timed_out"


def test_requirement_input_n_041() -> None:
    """Identify request, keyed answers, provenance, and resolution time."""
    request = _request()
    resolution = _answered(request)
    assert resolution.request_id == request.request_id
    assert {answer.question_id for answer in resolution.answers} == {
        QuestionId("confirm"),
        QuestionId("strategy"),
        QuestionId("checks"),
    }
    assert all(answer.provenance for answer in resolution.answers)
    assert resolution.resolved_at == _CREATED_AT + timedelta(seconds=1)


def test_requirement_input_n_042() -> None:
    """Distinguish human, trusted-default, and policy provenance."""
    question = ConfirmationQuestion(
        question_id=QuestionId("confirm"),
        prompt="Continue?",
        required=True,
        default_value=True,
    )
    pending = _pending(_request(questions=(question,)))
    for provenance in (
        AnswerProvenance.HUMAN,
        AnswerProvenance.TRUSTED_DEFAULT,
        AnswerProvenance.POLICY,
    ):
        answer = ConfirmationAnswer(
            question_id=question.question_id,
            provenance=provenance,
            value=True,
        )
        resolution = AnsweredResolution(
            request_id=pending.request_id,
            provenance=provenance,
            resolved_at=_CREATED_AT + timedelta(seconds=1),
            answers=(answer,),
        )
        result = resolve_request(
            pending,
            resolution,
            expected_state_revision=StateRevision(1),
        )
        assert isinstance(result, InputTransitionApplied)


def test_requirement_input_n_043() -> None:
    """Reject invalid answers before creating a continuation result."""
    pending = _pending(_request())
    invalid = AnsweredResolution(
        request_id=pending.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_CREATED_AT + timedelta(seconds=1),
        answers=(
            TextAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value="yes",
            ),
        ),
    )
    result = resolve_request(
        pending,
        invalid,
        expected_state_revision=StateRevision(1),
    )
    assert isinstance(result, InputTransitionRejected)
    assert result.previous is pending
    assert pending.state is RequestState.PENDING
