"""Exercise bounded inbound MCP form elicitation."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    Task,
    create_task,
    gather,
    run,
    sleep,
    wait_for,
)
from asyncio import (
    shield as asyncio_shield,
)
from collections.abc import Awaitable
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast
from unittest.mock import patch

from pytest import raises

from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    BranchId,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    FreeFormOther,
    InputHandlerContext,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
    InputRequest,
    InputRequestId,
    InputTransitionApplied,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    ParticipantId,
    PrincipalScope,
    QuestionId,
    RequirementMode,
    RunId,
    SelectedChoice,
    SelectionValidationConstraints,
    SessionId,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StateRevision,
    StreamSessionId,
    TaskId,
    TenantId,
    TextAnswer,
    TextQuestion,
    TextValidationConstraints,
    TurnId,
    UserId,
    mark_request_pending,
)
from avalan.interaction.handler import InputDisconnectReason
from avalan.server.mcp_session import (
    MCP_CONFLICT,
    MCP_ELICITATION_CREATE_METHOD,
    MCP_FORM_OTHER_PROPERTY_PREFIX,
    MCP_FORM_RESPONSE_MAX_BYTES,
    MCP_INVALID_PARAMS,
    MCP_PROTOCOL_VERSION,
    MCP_RELATED_TASK_METADATA_KEY,
    MCP_REQUIRED_OTHER_METADATA_KEY,
    MCPElicitationCapabilities,
    MCPFormErrorCode,
    MCPFormSessionError,
    MCPFormSessionRegistry,
    MCPFormStatus,
    MCPFormStatusEvent,
    MCPFormStatusHook,
    mcp_form_other_property_name,
    normalize_mcp_elicitation_capabilities,
    project_mcp_form_params,
)

_NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
_OWNER = PrincipalScope(
    user_id=UserId("owner"),
    session_id=SessionId("owner-session"),
)
_OTHER_OWNER = PrincipalScope(
    user_id=UserId("other"),
    session_id=SessionId("other-session"),
)
_CHOICES = (
    Choice(
        value=ChoiceValue("stable-a"),
        label="Alpha",
        description="First choice.",
    ),
    Choice(value=ChoiceValue("stable-b"), label="Beta"),
)


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run"),
        turn_id=TurnId("turn"),
        task_id=TaskId("task"),
        agent_id=AgentId("agent"),
        branch_id=BranchId("branch"),
        model_call_id=ModelCallId("model-call"),
        stream_session_id=StreamSessionId("stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://test",
            agent_definition_revision="revision",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model",
            tool_revision="tools",
            capability_revision="capabilities",
        ),
        principal=_OWNER,
    )


def _request(
    *questions: ConfirmationQuestion
    | TextQuestion
    | MultilineTextQuestion
    | SingleSelectionQuestion
    | MultipleSelectionQuestion,
    request_id: str = "request",
    reason: str = "More information is needed.",
) -> InputRequest:
    request = InputRequest(
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(f"continuation-{request_id}"),
        origin=_origin(),
        mode=RequirementMode.REQUIRED,
        reason=reason,
        questions=questions,
        created_at=_NOW,
    )
    result = mark_request_pending(
        request,
        expected_state_revision=StateRevision(0),
    )
    assert isinstance(result, InputTransitionApplied)
    return result.request


async def _ready(
    registry: MCPFormSessionRegistry,
    *,
    session_id: str = "session",
    capabilities: object = {"elicitation": {"form": {}}},
    owner: PrincipalScope = _OWNER,
    can_route: bool = True,
    preserves_newlines: bool = True,
) -> None:
    view = await registry.initialize(
        session_id=session_id,
        owner=owner,
        protocol_version=MCP_PROTOCOL_VERSION,
        capabilities=capabilities,
        can_route_and_resume=can_route,
        preserves_newlines=preserves_newlines,
    )
    assert view.protocol_version == MCP_PROTOCOL_VERSION
    await registry.mark_initialized(session_id, owner)


def _handler_task(
    registry: MCPFormSessionRegistry,
    request: InputRequest,
    *,
    session_id: str = "session",
    owner: PrincipalScope = _OWNER,
    hook: MCPFormStatusHook | None = None,
    related_request_id: str = "client-call",
    related_task_id: str | None = None,
) -> Task[InputHandlerOutcome]:
    handler = registry.handler(
        session_id=session_id,
        owner=owner,
        related_request_id=related_request_id,
        related_task_id=related_task_id,
        status_hook=hook,
    )
    return create_task(handler(InputHandlerContext(request=request)))


def _response(response_id: str, result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": response_id, "result": result}


def test_capability_normalization_and_projection() -> None:
    assert normalize_mcp_elicitation_capabilities({}) == (
        MCPElicitationCapabilities(False, False)
    )
    assert normalize_mcp_elicitation_capabilities(
        {"elicitation": {}}
    ) == MCPElicitationCapabilities(True, False, True)
    assert normalize_mcp_elicitation_capabilities(
        {"elicitation": {"form": {}, "url": {}}}
    ) == MCPElicitationCapabilities(True, True)
    assert normalize_mcp_elicitation_capabilities(
        {"elicitation": {"url": {}}}
    ) == MCPElicitationCapabilities(False, True)
    invalid_capabilities: tuple[object, ...] = (
        [],
        {"elicitation": True},
        {"elicitation": {"form": True}},
        {"elicitation": {"unknown": {}}},
    )
    for value in invalid_capabilities:
        with raises(MCPFormSessionError) as caught:
            normalize_mcp_elicitation_capabilities(value)
        assert caught.value.code is MCPFormErrorCode.INVALID_CAPABILITIES
        assert caught.value.rpc_code == MCP_INVALID_PARAMS

    questions = (
        ConfirmationQuestion(
            question_id=QuestionId("confirm"),
            prompt="Continue?",
            required=True,
            default_value=True,
        ),
        TextQuestion(
            question_id=QuestionId("name"),
            prompt="Name?",
            required=False,
            header="Name",
            help_text="Use one line.",
            default_value="Ada",
            constraints=TextValidationConstraints(
                minimum_length=0,
                maximum_length=20,
            ),
        ),
        MultilineTextQuestion(
            question_id=QuestionId("notes"),
            prompt="Notes?",
            required=True,
            default_value="Default notes.",
            constraints=TextValidationConstraints(
                minimum_length=2,
                maximum_length=100,
            ),
        ),
    )
    params = project_mcp_form_params(_request(*questions))
    assert params["mode"] == "form"
    assert "_meta" not in params
    schema = cast(dict[str, object], params["requestedSchema"])
    assert set(schema) == {"$schema", "type", "properties", "required"}
    assert schema["required"] == ["confirm", "notes"]
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    assert properties["confirm"] == {
        "title": "Continue?",
        "type": "boolean",
        "default": True,
    }
    assert properties["name"] == {
        "title": "Name",
        "description": "Name? Use one line.",
        "type": "string",
        "minLength": 0,
        "maxLength": 20,
        "default": "Ada",
    }
    assert properties["notes"]["pattern"] == r"^(?:[^\r]|\r\n)*$"
    assert properties["notes"]["default"] == "Default notes."
    with raises(TypeError):
        project_mcp_form_params(cast(InputRequest, object()))

    selections = _request(
        SingleSelectionQuestion(
            question_id=QuestionId("single"),
            prompt="Choose one.",
            required=True,
            choices=_CHOICES,
            allow_other=True,
            default_value=ChoiceValue("stable-a"),
        ),
        MultipleSelectionQuestion(
            question_id=QuestionId("multiple"),
            prompt="Choose several.",
            required=True,
            choices=_CHOICES,
            allow_other=True,
            default_value=(ChoiceValue("stable-b"),),
            constraints=SelectionValidationConstraints(
                minimum=1,
                maximum=3,
            ),
        ),
    )
    legacy = project_mcp_form_params(
        selections,
        legacy_form_only=True,
    )
    assert "mode" not in legacy
    schema = cast(dict[str, object], legacy["requestedSchema"])
    assert "required" not in schema
    assert legacy["_meta"] == {
        MCP_REQUIRED_OTHER_METADATA_KEY: ["single", "multiple"],
    }
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    assert properties["single"]["enum"] == ["stable-a", "stable-b"]
    assert properties["multiple"]["items"] == {
        "type": "string",
        "enum": ["stable-a", "stable-b"],
    }
    assert properties["multiple"]["minItems"] == 0
    assert properties["multiple"]["uniqueItems"] is True
    other = mcp_form_other_property_name(QuestionId("single"))
    assert other == f"{MCP_FORM_OTHER_PROPERTY_PREFIX}single"
    assert properties[other]["type"] == "string"
    with raises(MCPFormSessionError) as caught:
        project_mcp_form_params(
            _request(questions[2]),
            preserves_newlines=False,
        )
    assert caught.value.code is MCPFormErrorCode.MULTILINE_UNAVAILABLE


def test_sensitive_values_are_rejected_before_projection() -> None:
    unsafe_requests = (
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Enter OTP.",
                required=True,
            ),
            request_id="unsafe-otp",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Enter the value.",
                header="PIN",
                required=True,
            ),
            request_id="unsafe-pin",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Enter your pin.",
                required=True,
            ),
            request_id="unsafe-contextual-pin",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="What is your pin?",
                required=True,
            ),
            request_id="unsafe-interrogative-pin",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Your pin, please.",
                required=True,
            ),
            request_id="unsafe-possessive-pin",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Enter the current pin.",
                required=True,
            ),
            request_id="unsafe-modified-pin",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="What is your access pin?",
                required=True,
            ),
            request_id="unsafe-modified-possessive-pin",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Provide your access pin.",
                required=True,
            ),
            request_id="unsafe-modified-request-pin",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Choose an access pin.",
                required=True,
            ),
            request_id="unsafe-qualified-pin",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Enter the value.",
                help_text="Use the credit card number.",
                required=True,
            ),
            request_id="unsafe-card",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Enter the value.",
                required=True,
            ),
            request_id="unsafe-reason",
            reason="Complete payment authorization.",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Confirm the saved value.",
                required=True,
                default_value="secret-token-material",
            ),
            request_id="unsafe-text-default",
        ),
        _request(
            MultilineTextQuestion(
                question_id=QuestionId("value"),
                prompt="Confirm the saved value.",
                required=True,
                default_value="one-time password",
            ),
            request_id="unsafe-multiline-default",
        ),
        _request(
            SingleSelectionQuestion(
                question_id=QuestionId("value"),
                prompt="Choose.",
                required=True,
                choices=(
                    Choice(
                        value=ChoiceValue("otp"),
                        label="First choice",
                    ),
                    Choice(
                        value=ChoiceValue("second"),
                        label="Second choice",
                    ),
                ),
                default_value=ChoiceValue("otp"),
            ),
            request_id="unsafe-choice-value",
        ),
        _request(
            SingleSelectionQuestion(
                question_id=QuestionId("value"),
                prompt="Choose.",
                required=True,
                choices=(
                    Choice(
                        value=ChoiceValue("first"),
                        label="Authentication",
                    ),
                    Choice(
                        value=ChoiceValue("second"),
                        label="Second choice",
                    ),
                ),
            ),
            request_id="unsafe-choice-label",
        ),
        _request(
            SingleSelectionQuestion(
                question_id=QuestionId("value"),
                prompt="Choose.",
                required=True,
                choices=(
                    Choice(
                        value=ChoiceValue("first"),
                        label="First choice",
                        description="Use the verification code.",
                    ),
                    Choice(
                        value=ChoiceValue("second"),
                        label="Second choice",
                    ),
                ),
            ),
            request_id="unsafe-choice-description",
        ),
    )
    for request in unsafe_requests:
        with raises(MCPFormSessionError) as caught:
            project_mcp_form_params(request)
        assert caught.value.code is MCPFormErrorCode.UNSAFE_REQUEST
        assert caught.value.rpc_code == MCP_INVALID_PARAMS

    credential_lifecycle_prompts = (
        "Set your pin.",
        "Change a pin.",
        "Reset your pin.",
        "Choose a pin.",
        "Retype your pin.",
        "Verify a pin.",
        "Create your pin.",
        "Update a pin.",
        "Select your pin.",
        "Pick a pin.",
        "Reenter your pin.",
        "Re-enter a pin.",
        "Repeat your pin.",
        "Validate a pin.",
    )
    for index, prompt in enumerate(credential_lifecycle_prompts):
        request = _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt=prompt,
                required=True,
            ),
            request_id=f"unsafe-pin-lifecycle-{index}",
        )
        with raises(MCPFormSessionError) as caught:
            project_mcp_form_params(request)
        assert caught.value.code is MCPFormErrorCode.UNSAFE_REQUEST

    unsafe_labels = (
        "Passwords",
        "Credit cards",
        "OTPs",
        "TOTPs",
        "HOTPs",
        "PINs",
        "2FAs",
        "Verification codes",
        "Recovery codes",
        "Security codes",
        "API keys",
        "API tokens",
        "Access tokens",
        "Auth tokens",
        "Session tokens",
        "Identity tokens",
        "Private keys",
        "Payments",
        "Card details",
        "Routing numbers",
        "IBANs",
        "Bank accounts",
        "Account material",
    )
    for index, label in enumerate(unsafe_labels):
        request = _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt=f"Enter {label}.",
                required=True,
            ),
            request_id=f"unsafe-label-{index}",
        )
        with raises(MCPFormSessionError) as caught:
            project_mcp_form_params(request)
        assert caught.value.code is MCPFormErrorCode.UNSAFE_REQUEST

    secret_value_requests = (
        _request(
            TextQuestion(
                question_id=QuestionId("value"),
                prompt="Confirm the saved value.",
                required=True,
                default_value="4111 1111 1111 1111",
            ),
            request_id="unsafe-card-value",
        ),
        _request(
            SingleSelectionQuestion(
                question_id=QuestionId("value"),
                prompt="Choose the saved reference.",
                required=True,
                choices=(
                    Choice(
                        value=ChoiceValue(
                            "sk-proj-AbCdEf0123456789AbCdEf0123456789"
                        ),
                        label="First reference",
                    ),
                    Choice(
                        value=ChoiceValue("ordinary-reference"),
                        label="Second reference",
                    ),
                ),
            ),
            request_id="unsafe-api-key-value",
        ),
    )
    for request in secret_value_requests:
        with raises(MCPFormSessionError) as caught:
            project_mcp_form_params(request)
        assert caught.value.code is MCPFormErrorCode.UNSAFE_REQUEST

    safe_request = _request(
        TextQuestion(
            question_id=QuestionId("instruction"),
            prompt="Pin this report to the planning board.",
            required=False,
            default_value="123e4567-e89b-12d3-a456-426614174000",
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("layout"),
            prompt="Choose cards for the planning board.",
            required=True,
            choices=(
                Choice(
                    value=ChoiceValue("release-candidate-20260724"),
                    label="Release candidate",
                ),
                Choice(
                    value=ChoiceValue("current"),
                    label="Current",
                ),
            ),
        ),
        SingleSelectionQuestion(
            question_id=QuestionId("marker"),
            prompt="Choose a map marker.",
            required=True,
            choices=(
                Choice(value=ChoiceValue("pin"), label="Map pin"),
                Choice(value=ChoiceValue("dot"), label="Map dot"),
            ),
            default_value=ChoiceValue("pin"),
        ),
        request_id="ordinary-values",
        reason="Collect ordinary display preferences.",
    )
    assert project_mcp_form_params(safe_request)["mode"] == "form"
    benign_pin_requests = (
        _request(
            TextQuestion(
                question_id=QuestionId("map-question"),
                prompt="Which map pin should move?",
                required=False,
            ),
            TextQuestion(
                question_id=QuestionId("map-action"),
                prompt="Move the red pin.",
                required=False,
            ),
            TextQuestion(
                question_id=QuestionId("report-action"),
                prompt="Pin report 42.",
                required=False,
            ),
            request_id="ordinary-pin-nouns",
        ),
        _request(
            TextQuestion(
                question_id=QuestionId("archive"),
                prompt="Pin the report to the archive.",
                required=False,
            ),
            request_id="ordinary-pin-imperative",
        ),
    )
    for request in benign_pin_requests:
        assert project_mcp_form_params(request)["mode"] == "form"


def test_session_initialization_guards_and_capacity() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(maximum_sessions=1)
        with raises(TypeError):
            await registry.initialize(
                session_id="bad-owner",
                owner=cast(PrincipalScope, object()),
                protocol_version=MCP_PROTOCOL_VERSION,
                capabilities={"elicitation": {"form": {}}},
                can_route_and_resume=True,
            )
        with raises(MCPFormSessionError):
            await registry.initialize(
                session_id="bad-version",
                owner=_OWNER,
                protocol_version=1,
                capabilities={"elicitation": {"form": {}}},
                can_route_and_resume=True,
            )
        with raises(TypeError):
            await registry.initialize(
                session_id="bad-routing",
                owner=_OWNER,
                protocol_version=MCP_PROTOCOL_VERSION,
                capabilities={"elicitation": {"form": {}}},
                can_route_and_resume=cast(bool, 1),
            )
        with raises(TypeError):
            await registry.initialize(
                session_id="bad-newlines",
                owner=_OWNER,
                protocol_version=MCP_PROTOCOL_VERSION,
                capabilities={"elicitation": {"form": {}}},
                can_route_and_resume=True,
                preserves_newlines=cast(bool, 1),
            )

        with raises(TypeError):
            registry.handler(
                session_id="session",
                owner=cast(PrincipalScope, object()),
                related_request_id="request",
            )
        with raises(MCPFormSessionError):
            registry.handler(
                session_id="session",
                owner=_OWNER,
                related_request_id=True,
            )
        with raises(TypeError):
            await registry.negotiation(
                "session",
                cast(PrincipalScope, object()),
            )

        await _ready(registry)
        with raises(MCPFormSessionError) as duplicate:
            await registry.initialize(
                session_id="session",
                owner=_OWNER,
                protocol_version=MCP_PROTOCOL_VERSION,
                capabilities={"elicitation": {"form": {}}},
                can_route_and_resume=True,
            )
        assert duplicate.value.code is MCPFormErrorCode.SESSION_CONFLICT
        with raises(MCPFormSessionError) as capacity:
            await registry.initialize(
                session_id="second",
                owner=_OWNER,
                protocol_version=MCP_PROTOCOL_VERSION,
                capabilities={"elicitation": {"form": {}}},
                can_route_and_resume=True,
            )
        assert capacity.value.code is MCPFormErrorCode.CAPABILITY_UNAVAILABLE

    run(scenario())


def test_status_hook_precedes_publication_and_cancellation_cleans_up() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await _ready(registry)
        request = _request(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            )
        )
        hook_entered = Event()
        release_hook = Event()

        async def blocking_hook(event: MCPFormStatusEvent) -> None:
            if event.status is MCPFormStatus.INPUT_REQUIRED:
                hook_entered.set()
                await release_hook.wait()

        task = _handler_task(registry, request, hook=blocking_hook)
        await wait_for(hook_entered.wait(), timeout=0.1)
        state = registry._sessions["session"]
        pending = next(iter(state.pending.values()))
        assert not pending.published
        assert not state.outbound
        assert (
            await registry.next_outbound(
                "session",
                _OWNER,
                timeout_seconds=0.01,
            )
            is None
        )
        release_hook.set()
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is pending.outbound
        assert pending.published
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound.jsonrpc_id),
                {"action": "decline"},
            ),
        )
        assert isinstance(
            cast(InputHandlerResolution, await task).resolution,
            DeclinedResolution,
        )

        cancel_entered = Event()
        never_release = Event()

        async def cancellable_hook(event: MCPFormStatusEvent) -> None:
            if event.status is MCPFormStatus.INPUT_REQUIRED:
                cancel_entered.set()
                await never_release.wait()

        cancelled_task = _handler_task(
            registry,
            replace(
                request,
                request_id=InputRequestId("cancelled-hook"),
                continuation_id=ContinuationId("cancelled-hook-continuation"),
            ),
            hook=cancellable_hook,
        )
        await wait_for(cancel_entered.wait(), timeout=0.1)
        cancelled_pending = next(iter(state.pending.values()))
        assert not cancelled_pending.published
        cancelled_task.cancel()
        with raises(CancelledError):
            await cancelled_task
        assert cancelled_pending.future.cancelled()
        assert not state.pending
        assert not state.outbound

        failure_entered = Event()
        fail_hook = Event()

        async def failing_hook(event: MCPFormStatusEvent) -> None:
            if event.status is MCPFormStatus.INPUT_REQUIRED:
                failure_entered.set()
                await fail_hook.wait()
                raise RuntimeError("private status failure")

        failed_task = _handler_task(
            registry,
            replace(
                request,
                request_id=InputRequestId("failed-hook"),
                continuation_id=ContinuationId("failed-hook-continuation"),
            ),
            hook=failing_hook,
        )
        await wait_for(failure_entered.wait(), timeout=0.1)
        assert not state.outbound
        assert (
            await registry.next_outbound(
                "session",
                _OWNER,
                timeout_seconds=0.01,
            )
            is None
        )
        fail_hook.set()
        failed = await failed_task
        assert isinstance(failed, InputHandlerDisconnected)
        assert failed.reason is InputDisconnectReason.HANDLER_UNAVAILABLE
        assert not state.pending
        assert not state.outbound

        unconsumed_task = _handler_task(
            registry,
            replace(
                request,
                request_id=InputRequestId("cancelled-published"),
                continuation_id=ContinuationId(
                    "cancelled-published-continuation"
                ),
            ),
        )

        async def wait_until_published() -> None:
            while not state.outbound:
                await sleep(0)

        await wait_for(wait_until_published(), timeout=0.1)
        unconsumed_pending = next(iter(state.pending.values()))
        assert unconsumed_pending.published
        unconsumed_task.cancel()
        with raises(CancelledError):
            await unconsumed_task
        assert unconsumed_pending.future.cancelled()
        assert not state.pending
        assert not state.outbound

        withdraw_entered = Event()
        release_withdraw_hook = Event()

        async def withdraw_hook(event: MCPFormStatusEvent) -> None:
            if event.status is MCPFormStatus.INPUT_REQUIRED:
                withdraw_entered.set()
                await release_withdraw_hook.wait()

        withdrawn_task = _handler_task(
            registry,
            replace(
                request,
                request_id=InputRequestId("withdrawn-before-publication"),
                continuation_id=ContinuationId(
                    "withdrawn-before-publication-continuation"
                ),
            ),
            hook=withdraw_hook,
        )
        await wait_for(withdraw_entered.wait(), timeout=0.1)
        await registry.withdraw_form("session", _OWNER)
        release_withdraw_hook.set()
        withdrawn = await withdrawn_task
        assert isinstance(withdrawn, InputHandlerDisconnected)
        assert withdrawn.reason is InputDisconnectReason.HANDLER_UNAVAILABLE
        assert not state.pending
        assert not state.outbound

    run(scenario())


def test_request_principal_must_exactly_match_session_owner() -> None:
    async def scenario() -> None:
        owner = PrincipalScope(
            user_id=UserId("owner"),
            tenant_id=TenantId("tenant"),
            participant_id=ParticipantId("participant"),
            session_id=SessionId("owner-session"),
        )
        mismatched_owners = (
            replace(owner, user_id=UserId("other")),
            replace(owner, tenant_id=TenantId("other")),
            replace(owner, participant_id=ParticipantId("other")),
            replace(owner, session_id=SessionId("other")),
        )
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await _ready(registry, owner=owner)
        for index, request_owner in enumerate(mismatched_owners):
            request = _request(
                TextQuestion(
                    question_id=QuestionId("text"),
                    prompt="Text?",
                    required=True,
                ),
                request_id=f"wrong-owner-{index}",
            )
            request = replace(
                request,
                origin=replace(request.origin, principal=request_owner),
            )
            outcome = await _handler_task(
                registry,
                request,
                owner=owner,
            )
            assert isinstance(outcome, InputHandlerDisconnected)
            assert outcome.reason is InputDisconnectReason.HANDLER_UNAVAILABLE
        assert await registry.pending_count("session", owner) == 0
        assert (
            await registry.next_outbound(
                "session",
                owner,
                timeout_seconds=0.01,
            )
            is None
        )

        matching = _request(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            ),
            request_id="matching-owner",
        )
        matching = replace(
            matching,
            origin=replace(matching.origin, principal=owner),
        )
        task = _handler_task(registry, matching, owner=owner)
        outbound = await registry.next_outbound("session", owner)
        assert outbound is not None
        await registry.dispatch_response(
            "session",
            owner,
            _response(
                cast(str, outbound.jsonrpc_id),
                {"action": "decline"},
            ),
        )
        assert isinstance(
            cast(InputHandlerResolution, await task).resolution,
            DeclinedResolution,
        )

    run(scenario())


def test_accept_decline_cancel_and_status_mapping() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await _ready(registry)
        events: list[MCPFormStatusEvent] = []

        async def hook(event: MCPFormStatusEvent) -> None:
            events.append(event)

        request = _request(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            ),
            SingleSelectionQuestion(
                question_id=QuestionId("single"),
                prompt="Choose.",
                required=True,
                choices=_CHOICES,
                allow_other=True,
            ),
            MultipleSelectionQuestion(
                question_id=QuestionId("multiple"),
                prompt="Choose several.",
                required=False,
                choices=_CHOICES,
                allow_other=True,
                constraints=SelectionValidationConstraints(
                    minimum=0,
                    maximum=3,
                ),
            ),
        )
        task = _handler_task(
            registry,
            request,
            hook=hook,
            related_task_id="mcp-task",
        )
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        assert outbound.method == MCP_ELICITATION_CREATE_METHOD
        assert outbound.related_request_id == "client-call"
        assert outbound.related_task_id == "mcp-task"
        assert outbound.params["_meta"] == {
            MCP_RELATED_TASK_METADATA_KEY: {"taskId": "mcp-task"},
            MCP_REQUIRED_OTHER_METADATA_KEY: ["single"],
        }
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound.jsonrpc_id),
                {
                    "action": "accept",
                    "_meta": {
                        MCP_RELATED_TASK_METADATA_KEY: {"taskId": "mcp-task"}
                    },
                    "content": {
                        "text": "hello",
                        mcp_form_other_property_name("single"): "custom",
                        "multiple": ["stable-b"],
                        mcp_form_other_property_name("multiple"): "extra",
                    },
                },
            ),
        )
        outcome = await task
        assert isinstance(outcome, InputHandlerResolution)
        assert isinstance(outcome.resolution, AnsweredResolution)
        assert isinstance(outcome.resolution.answers[0], TextAnswer)
        single = cast(SingleSelectionAnswer, outcome.resolution.answers[1])
        assert single.value == FreeFormOther(text="custom")
        multiple = cast(MultipleSelectionAnswer, outcome.resolution.answers[2])
        assert multiple.values == (
            SelectedChoice(value=ChoiceValue("stable-b")),
            FreeFormOther(text="extra"),
        )
        assert [event.status for event in events] == [
            MCPFormStatus.INPUT_REQUIRED,
            MCPFormStatus.ANSWERED,
        ]
        assert all(event.related_task_id == "mcp-task" for event in events)

        other_only_request = _request(
            MultipleSelectionQuestion(
                question_id=QuestionId("multiple"),
                prompt="Choose several.",
                required=True,
                choices=_CHOICES,
                allow_other=True,
                constraints=SelectionValidationConstraints(
                    minimum=1,
                    maximum=3,
                ),
            ),
            request_id="other-only",
        )
        task = _handler_task(registry, other_only_request)
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        assert outbound.params["_meta"] == {
            MCP_REQUIRED_OTHER_METADATA_KEY: ["multiple"],
        }
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound.jsonrpc_id),
                {
                    "action": "accept",
                    "content": {
                        mcp_form_other_property_name("multiple"): "custom",
                    },
                },
            ),
        )
        outcome = cast(InputHandlerResolution, await task)
        resolution = cast(AnsweredResolution, outcome.resolution)
        multiple = cast(MultipleSelectionAnswer, resolution.answers[0])
        assert multiple.values == (FreeFormOther(text="custom"),)

        for action, expected in (
            ("decline", DeclinedResolution),
            ("cancel", InputHandlerDisconnected),
        ):
            action_request = replace(
                request,
                request_id=InputRequestId(f"request-{action}"),
                continuation_id=ContinuationId(f"continuation-{action}"),
            )
            task = _handler_task(registry, action_request)
            outbound = await registry.next_outbound("session", _OWNER)
            assert outbound is not None
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(cast(str, outbound.jsonrpc_id), {"action": action}),
            )
            outcome = await task
            if action == "decline":
                assert isinstance(outcome, InputHandlerResolution)
                assert isinstance(outcome.resolution, expected)
            else:
                assert isinstance(outcome, InputHandlerDisconnected)
                assert (
                    outcome.reason is InputDisconnectReason.HANDLER_CANCELLED
                )
        assert await registry.pending_count("session", _OWNER) == 0

    run(scenario())


def test_sibling_filters_and_related_task_response_metadata() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await _ready(registry)
        request = _request(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            )
        )
        tasks = {
            name: _handler_task(
                registry,
                replace(
                    request,
                    request_id=InputRequestId(f"request-{name}"),
                    continuation_id=ContinuationId(f"continuation-{name}"),
                ),
                related_request_id=f"call-{name}",
                related_task_id=f"task-{name}",
            )
            for name in ("a", "b", "c")
        }
        outbound_b = await registry.next_outbound(
            "session",
            _OWNER,
            related_request_id="call-b",
            related_task_id="task-b",
        )
        assert outbound_b is not None
        assert outbound_b.canonical_request_id == "request-b"
        with raises(MCPFormSessionError):
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, outbound_b.jsonrpc_id),
                    {"action": "decline"},
                ),
            )
        assert not tasks["b"].done()
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound_b.jsonrpc_id),
                {
                    "action": "decline",
                    "_meta": {
                        MCP_RELATED_TASK_METADATA_KEY: {"taskId": "task-b"}
                    },
                },
            ),
        )
        assert isinstance(
            cast(InputHandlerResolution, await tasks["b"]).resolution,
            DeclinedResolution,
        )
        assert not tasks["a"].done()
        assert not tasks["c"].done()

        outbound_c = await registry.next_outbound(
            "session",
            _OWNER,
            related_request_id="call-c",
            related_task_id="task-c",
        )
        assert outbound_c is not None
        with raises(MCPFormSessionError):
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, outbound_c.jsonrpc_id),
                    {
                        "action": "decline",
                        "_meta": {
                            MCP_RELATED_TASK_METADATA_KEY: {
                                "taskId": "task-wrong"
                            }
                        },
                    },
                ),
            )
        assert not tasks["c"].done()
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound_c.jsonrpc_id),
                {
                    "action": "decline",
                    "_meta": {
                        MCP_RELATED_TASK_METADATA_KEY: {"taskId": "task-c"}
                    },
                },
            ),
        )
        assert isinstance(
            cast(InputHandlerResolution, await tasks["c"]).resolution,
            DeclinedResolution,
        )
        assert not tasks["a"].done()

        outbound_a = await registry.next_outbound(
            "session",
            _OWNER,
            related_request_id="call-a",
            related_task_id="task-a",
        )
        assert outbound_a is not None
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound_a.jsonrpc_id),
                {
                    "action": "decline",
                    "_meta": {
                        MCP_RELATED_TASK_METADATA_KEY: {"taskId": "task-a"}
                    },
                },
            ),
        )
        assert isinstance(
            cast(InputHandlerResolution, await tasks["a"]).resolution,
            DeclinedResolution,
        )
        assert await registry.pending_count("session", _OWNER) == 0

    run(scenario())


def test_successful_response_replays_are_bounded_and_private() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(
            response_wait_seconds=1,
            stale_response_limit=2,
        )
        await _ready(registry)
        request = _request(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            ),
            request_id="replay",
        )
        task = _handler_task(
            registry,
            request,
            related_task_id="task-replay",
        )
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        secret = "private-response-value-47"
        accepted_response: dict[str, object] = {
            "jsonrpc": "2.0",
            "id": outbound.jsonrpc_id,
            "result": {
                "action": "accept",
                "_meta": {
                    MCP_RELATED_TASK_METADATA_KEY: {
                        "taskId": "task-replay",
                    }
                },
                "content": {"text": secret},
            },
        }
        await registry.dispatch_response(
            "session",
            _OWNER,
            accepted_response,
        )
        outcome = cast(InputHandlerResolution, await task)
        assert isinstance(outcome.resolution, AnsweredResolution)

        state = registry._sessions["session"]
        response_id = outbound.jsonrpc_id
        fingerprint = state.replays[response_id]
        assert isinstance(fingerprint, bytes)
        assert len(fingerprint) == 32
        assert secret not in repr(state)
        assert secret not in repr(state.replays)
        assert secret.encode() not in fingerprint

        await registry.dispatch_response(
            "session",
            _OWNER,
            accepted_response,
        )
        reordered_response: dict[str, object] = {
            "result": {
                "content": {"text": secret},
                "_meta": {
                    MCP_RELATED_TASK_METADATA_KEY: {
                        "taskId": "task-replay",
                    }
                },
                "action": "accept",
            },
            "id": response_id,
            "jsonrpc": "2.0",
        }
        await registry.dispatch_response(
            "session",
            _OWNER,
            reordered_response,
        )

        conflicting_responses = (
            {
                "jsonrpc": "2.0",
                "id": response_id,
                "result": {
                    "action": "accept",
                    "_meta": {
                        MCP_RELATED_TASK_METADATA_KEY: {
                            "taskId": "task-replay",
                        }
                    },
                    "content": {"text": "different"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": response_id,
                "result": {
                    "action": "accept",
                    "_meta": {
                        MCP_RELATED_TASK_METADATA_KEY: {
                            "taskId": "wrong-task",
                        }
                    },
                    "content": {"text": secret},
                },
            },
        )
        for conflicting in conflicting_responses:
            with raises(MCPFormSessionError) as caught:
                await registry.dispatch_response(
                    "session",
                    _OWNER,
                    conflicting,
                )
            assert caught.value.code is MCPFormErrorCode.STALE_RESPONSE
            assert caught.value.rpc_code == MCP_CONFLICT

        with raises(MCPFormSessionError) as malformed:
            await registry.dispatch_response(
                "session",
                _OWNER,
                {
                    "jsonrpc": "2.0",
                    "id": response_id,
                    "result": [],
                },
            )
        assert malformed.value.code is MCPFormErrorCode.INVALID_RESPONSE
        assert malformed.value.rpc_code == MCP_INVALID_PARAMS

        with raises(MCPFormSessionError) as oversized:
            await registry.dispatch_response(
                "session",
                _OWNER,
                {
                    "jsonrpc": "2.0",
                    "id": response_id,
                    "result": {
                        "action": "accept",
                        "content": {
                            "text": "x" * MCP_FORM_RESPONSE_MAX_BYTES,
                        },
                    },
                },
            )
        assert oversized.value.code is MCPFormErrorCode.OVERSIZED_RESPONSE
        assert oversized.value.rpc_code == MCP_INVALID_PARAMS

        await _ready(registry, session_id="other-session")
        other_task = _handler_task(
            registry,
            request,
            session_id="other-session",
            related_task_id="task-replay",
        )
        other_outbound = await registry.next_outbound(
            "other-session",
            _OWNER,
        )
        assert other_outbound is not None
        assert other_outbound.jsonrpc_id == response_id
        await registry.dispatch_response(
            "other-session",
            _OWNER,
            accepted_response,
        )
        assert isinstance(await other_task, InputHandlerResolution)
        other_state = registry._sessions["other-session"]
        assert other_state.replays[response_id] != fingerprint
        await registry.close("other-session", _OWNER)
        assert tuple(other_state.stale) == (response_id,)
        assert not other_state.replays

        for index in range(2):
            eviction_request = _request(
                TextQuestion(
                    question_id=QuestionId("text"),
                    prompt="Text?",
                    required=True,
                ),
                request_id=f"eviction-{index}",
            )
            eviction_task = _handler_task(registry, eviction_request)
            eviction_outbound = await registry.next_outbound(
                "session",
                _OWNER,
            )
            assert eviction_outbound is not None
            eviction_response = _response(
                cast(str, eviction_outbound.jsonrpc_id),
                {"action": "decline"},
            )
            await registry.dispatch_response(
                "session",
                _OWNER,
                eviction_response,
            )
            assert isinstance(await eviction_task, InputHandlerResolution)

        assert len(state.stale) == 2
        assert len(state.replays) == 2
        assert response_id not in state.stale
        assert response_id not in state.replays
        with raises(MCPFormSessionError) as evicted:
            await registry.dispatch_response(
                "session",
                _OWNER,
                accepted_response,
            )
        assert evicted.value.code is MCPFormErrorCode.RESPONSE_NOT_PENDING
        assert evicted.value.rpc_code == MCP_INVALID_PARAMS

    run(scenario())


def test_all_answer_types_and_canonical_validation() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await _ready(registry)
        request = _request(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
            MultilineTextQuestion(
                question_id=QuestionId("notes"),
                prompt="Notes?",
                required=True,
            ),
            SingleSelectionQuestion(
                question_id=QuestionId("single"),
                prompt="Choose.",
                required=True,
                choices=_CHOICES,
            ),
        )
        task = _handler_task(registry, request)
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound.jsonrpc_id),
                {
                    "action": "accept",
                    "content": {
                        "confirm": True,
                        "notes": "line 1\r\nline 2",
                        "single": "stable-a",
                    },
                },
            ),
        )
        outcome = await task
        assert isinstance(outcome, InputHandlerResolution)
        resolution = cast(AnsweredResolution, outcome.resolution)
        assert isinstance(resolution.answers[0], ConfirmationAnswer)
        multiline = cast(MultilineTextAnswer, resolution.answers[1])
        assert multiline.value == "line 1\nline 2"
        selection = cast(SingleSelectionAnswer, resolution.answers[2])
        assert selection.value == SelectedChoice(value=ChoiceValue("stable-a"))

        invalid_results = (
            {"action": "accept"},
            {"action": "accept", "content": {"confirm": "yes"}},
            {"action": "accept", "content": {"unknown": "value"}},
            {"action": "accept", "content": {}},
            {"action": "decline", "content": {}},
            {"action": "cancel", "content": {}},
            {"action": "other"},
        )
        for index, result in enumerate(invalid_results):
            item = replace(
                request,
                request_id=InputRequestId(f"invalid-{index}"),
                continuation_id=ContinuationId(f"invalid-cont-{index}"),
            )
            task = _handler_task(registry, item)
            outbound = await registry.next_outbound("session", _OWNER)
            assert outbound is not None
            with raises(MCPFormSessionError) as caught:
                await registry.dispatch_response(
                    "session",
                    _OWNER,
                    _response(cast(str, outbound.jsonrpc_id), result),
                )
            assert caught.value.code is MCPFormErrorCode.INVALID_RESPONSE
            assert caught.value.rpc_code == MCP_INVALID_PARAMS
            assert not task.done()
            assert await registry.pending_count("session", _OWNER) == 1
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, outbound.jsonrpc_id),
                    {"action": "decline"},
                ),
            )
            assert isinstance(
                cast(InputHandlerResolution, await task).resolution,
                DeclinedResolution,
            )

        optional_requests = (
            _request(
                TextQuestion(
                    question_id=QuestionId("text"),
                    prompt="Text?",
                    required=False,
                ),
                MultilineTextQuestion(
                    question_id=QuestionId("notes"),
                    prompt="Notes?",
                    required=False,
                ),
                MultipleSelectionQuestion(
                    question_id=QuestionId("multiple"),
                    prompt="Choose.",
                    required=False,
                    choices=_CHOICES,
                ),
                request_id="optional-text",
            ),
            _request(
                SingleSelectionQuestion(
                    question_id=QuestionId("single"),
                    prompt="Choose.",
                    required=False,
                    choices=_CHOICES,
                ),
                request_id="optional-single",
            ),
        )
        for item in optional_requests:
            task = _handler_task(registry, item)
            outbound = await registry.next_outbound("session", _OWNER)
            assert outbound is not None
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, outbound.jsonrpc_id),
                    {"action": "accept", "content": {}},
                ),
            )
            outcome = cast(InputHandlerResolution, await task)
            assert cast(AnsweredResolution, outcome.resolution).answers == ()

        other_name = mcp_form_other_property_name("single")
        invalid_answer_cases = (
            (
                _request(
                    TextQuestion(
                        question_id=QuestionId("text"),
                        prompt="Text?",
                        required=True,
                    ),
                    request_id="bad-text",
                ),
                {"text": 1},
            ),
            (
                _request(
                    MultilineTextQuestion(
                        question_id=QuestionId("notes"),
                        prompt="Notes?",
                        required=True,
                    ),
                    request_id="bad-multiline",
                ),
                {"notes": 1},
            ),
            (
                _request(
                    SingleSelectionQuestion(
                        question_id=QuestionId("single"),
                        prompt="Choose.",
                        required=True,
                        choices=_CHOICES,
                    ),
                    request_id="bad-selection",
                ),
                {"single": 1},
            ),
            (
                _request(
                    SingleSelectionQuestion(
                        question_id=QuestionId("single"),
                        prompt="Choose.",
                        required=True,
                        choices=_CHOICES,
                        allow_other=True,
                    ),
                    request_id="ambiguous-selection",
                ),
                {"single": "stable-a", other_name: "other"},
            ),
            (
                _request(
                    SingleSelectionQuestion(
                        question_id=QuestionId("single"),
                        prompt="Choose.",
                        required=True,
                        choices=_CHOICES,
                        allow_other=True,
                    ),
                    request_id="bad-other",
                ),
                {other_name: 1},
            ),
            (
                _request(
                    MultipleSelectionQuestion(
                        question_id=QuestionId("multiple"),
                        prompt="Choose.",
                        required=True,
                        choices=_CHOICES,
                    ),
                    request_id="bad-multiple",
                ),
                {"multiple": "stable-a"},
            ),
            (
                _request(
                    TextQuestion(
                        question_id=QuestionId("text"),
                        prompt="Text?",
                        required=True,
                        constraints=TextValidationConstraints(
                            minimum_length=1,
                            maximum_length=2,
                        ),
                    ),
                    request_id="bad-constraint",
                ),
                {"text": "too long"},
            ),
        )
        for item, content in invalid_answer_cases:
            task = _handler_task(registry, item)
            outbound = await registry.next_outbound("session", _OWNER)
            assert outbound is not None
            with raises(MCPFormSessionError) as caught:
                await registry.dispatch_response(
                    "session",
                    _OWNER,
                    _response(
                        cast(str, outbound.jsonrpc_id),
                        {"action": "accept", "content": content},
                    ),
                )
            assert caught.value.rpc_code == MCP_INVALID_PARAMS
            assert not task.done()
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, outbound.jsonrpc_id),
                    {"action": "decline"},
                ),
            )
            assert isinstance(
                cast(InputHandlerResolution, await task).resolution,
                DeclinedResolution,
            )

    run(scenario())


def test_session_authority_lifecycle_and_bounded_cleanup() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(
            maximum_sessions=2,
            maximum_pending_per_session=2,
            response_wait_seconds=0.1,
            stale_response_limit=1,
        )
        initial = await registry.initialize(
            session_id="session",
            owner=_OWNER,
            protocol_version="older-client-version",
            capabilities={"elicitation": {}},
            can_route_and_resume=True,
        )
        assert initial.form_available is False
        assert initial.protocol_version == MCP_PROTOCOL_VERSION
        request = _request(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            )
        )
        before_ready = await _handler_task(registry, request)
        assert isinstance(before_ready, InputHandlerDisconnected)
        assert registry.session_count == 1
        await registry.mark_initialized("session", _OWNER)
        assert (await registry.negotiation("session", _OWNER)).form_available

        unrouted = await _handler_task(
            registry,
            replace(
                request,
                request_id=InputRequestId("unrouted"),
                continuation_id=ContinuationId("unrouted-continuation"),
            ),
        )
        assert isinstance(unrouted, InputHandlerDisconnected)
        assert len(registry._sessions["session"].outbound) == 0

        timed_out = _handler_task(registry, request)
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        with raises(MCPFormSessionError) as invalid:
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, outbound.jsonrpc_id),
                    {"action": "accept", "content": {}},
                ),
            )
        assert invalid.value.rpc_code == MCP_INVALID_PARAMS
        assert not timed_out.done()
        assert isinstance(await timed_out, InputHandlerDisconnected)
        assert await registry.pending_count("session", _OWNER) == 0
        with raises(MCPFormSessionError) as caught:
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, outbound.jsonrpc_id),
                    {"action": "decline"},
                ),
            )
        assert caught.value.code is MCPFormErrorCode.STALE_RESPONSE
        assert caught.value.rpc_code == MCP_CONFLICT

        pending = _handler_task(registry, request)
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        second_pending = _handler_task(
            registry,
            replace(
                request,
                request_id=InputRequestId("second-pending"),
                continuation_id=ContinuationId("second-pending-continuation"),
            ),
        )
        second_outbound = await registry.next_outbound("session", _OWNER)
        assert second_outbound is not None
        over_capacity = await _handler_task(
            registry,
            replace(
                request,
                request_id=InputRequestId("over-capacity"),
                continuation_id=ContinuationId("over-capacity-continuation"),
            ),
        )
        assert isinstance(over_capacity, InputHandlerDisconnected)
        for operation in (
            lambda: registry.negotiation("missing", _OWNER),
            lambda: registry.negotiation("session", _OTHER_OWNER),
            lambda: registry.dispatch_response(
                "session",
                _OTHER_OWNER,
                _response(
                    cast(str, outbound.jsonrpc_id),
                    {"action": "decline"},
                ),
            ),
        ):
            with raises(MCPFormSessionError) as caught:
                await operation()
            assert caught.value.code is MCPFormErrorCode.SESSION_NOT_FOUND
        assert caught.value.safe_message == "MCP session is unavailable"
        outbound_waiter = create_task(
            registry.next_outbound(
                "session",
                _OWNER,
                timeout_seconds=1,
            )
        )
        await sleep(0)
        await registry.withdraw_form("session", _OWNER)
        assert await wait_for(outbound_waiter, timeout=0.1) is None
        for unavailable in await gather(pending, second_pending):
            assert isinstance(unavailable, InputHandlerDisconnected)
            assert (
                unavailable.reason is InputDisconnectReason.HANDLER_UNAVAILABLE
            )
        with raises(MCPFormSessionError) as withdrawn:
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, second_outbound.jsonrpc_id),
                    {"action": "decline"},
                ),
            )
        assert withdrawn.value.code is MCPFormErrorCode.STALE_RESPONSE
        assert withdrawn.value.rpc_code == MCP_CONFLICT
        after_withdrawal = await _handler_task(registry, request)
        assert isinstance(after_withdrawal, InputHandlerDisconnected)
        negotiation = await registry.negotiation("session", _OWNER)
        assert not negotiation.form_available

        await _ready(registry, session_id="close-session")
        closing = _handler_task(
            registry,
            request,
            session_id="close-session",
        )
        assert (
            await registry.next_outbound("close-session", _OWNER) is not None
        )
        close_state = registry._sessions["close-session"]
        close_outbound = next(iter(close_state.pending.values())).outbound
        with raises(MCPFormSessionError):
            await registry.dispatch_response(
                "close-session",
                _OWNER,
                _response(
                    cast(str, close_outbound.jsonrpc_id),
                    {"action": "accept", "content": {}},
                ),
            )
        assert not closing.done()
        await registry.close("close-session", _OWNER)
        disconnected = await closing
        assert isinstance(disconnected, InputHandlerDisconnected)
        assert (
            disconnected.reason is InputDisconnectReason.CONTROL_CHANNEL_CLOSED
        )
        assert not close_state.pending
        assert not close_state.outbound
        assert registry.session_count == 1

        await _ready(registry, session_id="all-session")
        close_all_pending = _handler_task(
            registry,
            request,
            session_id="all-session",
        )
        assert await registry.next_outbound("all-session", _OWNER) is not None
        await registry.close_all()
        disconnected = await close_all_pending
        assert isinstance(disconnected, InputHandlerDisconnected)
        assert (
            disconnected.reason is InputDisconnectReason.CONTROL_CHANNEL_CLOSED
        )
        assert registry.session_count == 0

    run(scenario())


def test_session_close_races_fail_closed() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await registry.initialize(
            session_id="initializing",
            owner=_OWNER,
            protocol_version=MCP_PROTOCOL_VERSION,
            capabilities={"elicitation": {"form": {}}},
            can_route_and_resume=True,
        )
        original_owned = registry._owned
        obtained = Event()
        resume = Event()

        async def paused_owned(
            session_id: str,
            owner: PrincipalScope,
        ) -> object:
            state = await original_owned(session_id, owner)
            obtained.set()
            await resume.wait()
            return state

        with patch.object(registry, "_owned", paused_owned):
            initializing = create_task(
                registry.mark_initialized("initializing", _OWNER)
            )
            await obtained.wait()
            await registry.close_all()
            resume.set()
            with raises(MCPFormSessionError):
                await initializing

        await _ready(registry, session_id="closing")
        original_owned = registry._owned
        obtained = Event()
        resume = Event()

        async def paused_close_owned(
            session_id: str,
            owner: PrincipalScope,
        ) -> object:
            state = await original_owned(session_id, owner)
            obtained.set()
            await resume.wait()
            return state

        with patch.object(registry, "_owned", paused_close_owned):
            closing = create_task(registry.close("closing", _OWNER))
            await obtained.wait()
            await registry.close_all()
            resume.set()
            with raises(MCPFormSessionError):
                await closing

        await _ready(registry, session_id="waiting")
        waiting = create_task(
            registry.next_outbound(
                "waiting",
                _OWNER,
                timeout_seconds=1,
            )
        )
        await sleep(0)
        await registry.close("waiting", _OWNER)
        with raises(MCPFormSessionError):
            await waiting

    run(scenario())


def test_session_close_cleanup_survives_caller_cancellation() -> None:
    async def exercise(*, all_sessions: bool) -> None:
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await _ready(registry)
        request = _request(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            )
        )
        pending = _handler_task(registry, request)
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        state = registry._sessions["session"]
        await state.lock.acquire()
        shield_entries = (Event(), Event(), Event())
        shield_calls = 0

        def observed_shield(
            awaitable: Awaitable[None],
        ) -> Future[None]:
            nonlocal shield_calls
            shield_entries[shield_calls].set()
            shield_calls += 1
            return asyncio_shield(awaitable)

        try:
            with patch(
                "avalan.server.mcp_session.shield",
                new=observed_shield,
            ):
                closing = create_task(
                    registry.close_all()
                    if all_sessions
                    else registry.close("session", _OWNER)
                )
                await wait_for(shield_entries[0].wait(), timeout=0.1)
                assert registry.session_count == 0
                assert closing.cancel()
                await wait_for(shield_entries[1].wait(), timeout=0.1)
                assert not closing.done()
                assert closing.cancel()
                await wait_for(shield_entries[2].wait(), timeout=0.1)
                assert not closing.done()
                state.lock.release()
                with raises(CancelledError):
                    await closing
        finally:
            if state.lock.locked():
                state.lock.release()
        assert shield_calls == 3
        outcome = await pending
        assert isinstance(outcome, InputHandlerDisconnected)
        assert outcome.reason is InputDisconnectReason.CONTROL_CHANNEL_CLOSED
        assert state.closed
        assert not state.pending
        assert not state.outbound

    async def scenario() -> None:
        await exercise(all_sessions=False)
        await exercise(all_sessions=True)

    run(scenario())


def test_malformed_oversized_peer_error_and_races_do_not_leak() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await _ready(registry)
        request = _request(
            TextQuestion(
                question_id=QuestionId("text"),
                prompt="Text?",
                required=True,
            )
        )

        stale_outbound = _handler_task(
            registry,
            replace(
                request,
                request_id=InputRequestId("stale-outbound"),
                continuation_id=ContinuationId("stale-outbound-continuation"),
            ),
        )
        await sleep(0)
        with raises(MCPFormSessionError) as malformed_unconsumed:
            await registry.dispatch_response("session", _OWNER, [])
        assert malformed_unconsumed.value.rpc_code == MCP_INVALID_PARAMS
        assert not stale_outbound.done()
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound.jsonrpc_id),
                {"action": "decline"},
            ),
        )
        assert isinstance(
            cast(InputHandlerResolution, await stale_outbound).resolution,
            DeclinedResolution,
        )
        assert not registry._sessions["session"].outbound
        assert (
            await registry.next_outbound(
                "session",
                _OWNER,
                timeout_seconds=0.01,
            )
            is None
        )

        malformed: tuple[dict[str, object], ...] = (
            {},
            {"jsonrpc": "1.0", "id": "x", "result": {}},
            {"jsonrpc": "2.0", "id": True, "result": {}},
            {"jsonrpc": "2.0", "result": {}},
            {
                "jsonrpc": "2.0",
                "id": "x",
                "result": {},
                "error": {},
            },
            {"jsonrpc": "2.0", "id": "x", "result": []},
            {
                "jsonrpc": "2.0",
                "id": "x",
                "error": {"code": "bad", "message": 1},
            },
            {"jsonrpc": "2.0", "id": "x", "result": {"value": float("nan")}},
            {
                "jsonrpc": "2.0",
                "id": "x",
                "result": {1: "unexpected", "action": "decline"},
            },
        )
        for index, response in enumerate(malformed):
            item = replace(
                request,
                request_id=InputRequestId(f"malformed-{index}"),
                continuation_id=ContinuationId(f"malformed-cont-{index}"),
            )
            task = _handler_task(registry, item)
            outbound = await registry.next_outbound("session", _OWNER)
            assert outbound is not None
            if response.get("id") == "x":
                response["id"] = outbound.jsonrpc_id
            with raises(MCPFormSessionError) as malformed_response:
                await registry.dispatch_response(
                    "session",
                    _OWNER,
                    response,
                )
            assert malformed_response.value.rpc_code == MCP_INVALID_PARAMS
            assert not task.done()
            assert await registry.pending_count("session", _OWNER) == 1
            await registry.dispatch_response(
                "session",
                _OWNER,
                _response(
                    cast(str, outbound.jsonrpc_id),
                    {"action": "decline"},
                ),
            )
            assert isinstance(
                cast(InputHandlerResolution, await task).resolution,
                DeclinedResolution,
            )

        await _ready(registry, session_id="ambiguous")
        ambiguous_tasks = tuple(
            _handler_task(
                registry,
                replace(
                    request,
                    request_id=InputRequestId(f"ambiguous-{index}"),
                    continuation_id=ContinuationId(
                        f"ambiguous-continuation-{index}"
                    ),
                ),
                session_id="ambiguous",
            )
            for index in range(2)
        )
        await sleep(0)
        assert await registry.pending_count("ambiguous", _OWNER) == 2
        with raises(MCPFormSessionError) as ambiguous:
            await registry.dispatch_response("ambiguous", _OWNER, {})
        assert ambiguous.value.code is MCPFormErrorCode.AMBIGUOUS_RESPONSE
        await registry.close("ambiguous", _OWNER)
        assert all(
            isinstance(outcome, InputHandlerDisconnected)
            for outcome in await gather(*ambiguous_tasks)
        )

        secret = "do-not-leak"
        task = _handler_task(registry, request)
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        huge = _response(
            cast(str, outbound.jsonrpc_id),
            {
                "action": "accept",
                "content": {"text": secret * MCP_FORM_RESPONSE_MAX_BYTES},
            },
        )
        with raises(MCPFormSessionError) as caught:
            await registry.dispatch_response("session", _OWNER, huge)
        assert caught.value.code is MCPFormErrorCode.OVERSIZED_RESPONSE
        assert caught.value.rpc_code == MCP_INVALID_PARAMS
        assert secret not in str(caught.value)
        assert not task.done()
        await registry.dispatch_response(
            "session",
            _OWNER,
            _response(
                cast(str, outbound.jsonrpc_id),
                {"action": "decline"},
            ),
        )
        assert isinstance(
            cast(InputHandlerResolution, await task).resolution,
            DeclinedResolution,
        )

        recoverable_peer_error_task = _handler_task(registry, request)
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        with raises(MCPFormSessionError) as recoverable:
            await registry.dispatch_response(
                "session",
                _OWNER,
                {
                    "jsonrpc": "2.0",
                    "id": outbound.jsonrpc_id,
                    "error": {"code": -32000, "message": secret},
                },
            )
        assert recoverable.value.code is MCPFormErrorCode.PEER_ERROR
        assert secret not in str(recoverable.value)
        assert isinstance(
            await recoverable_peer_error_task,
            InputHandlerDisconnected,
        )
        assert (await registry.negotiation("session", _OWNER)).form_available

        peer_error_task = _handler_task(registry, request)
        outbound = await registry.next_outbound("session", _OWNER)
        assert outbound is not None
        with raises(MCPFormSessionError) as caught:
            await registry.dispatch_response(
                "session",
                _OWNER,
                {
                    "jsonrpc": "2.0",
                    "id": outbound.jsonrpc_id,
                    "error": {"code": -32602, "message": secret},
                },
            )
        assert caught.value.code is MCPFormErrorCode.PEER_ERROR
        assert secret not in str(caught.value)
        assert isinstance(await peer_error_task, InputHandlerDisconnected)
        negotiation = await registry.negotiation("session", _OWNER)
        assert not negotiation.form_available

        await _ready(registry, session_id="race")
        for index in range(10):
            item = replace(
                request,
                request_id=InputRequestId(f"race-{index}"),
                continuation_id=ContinuationId(f"race-cont-{index}"),
            )
            task = _handler_task(registry, item, session_id="race")
            outbound = await registry.next_outbound("race", _OWNER)
            assert outbound is not None
            dispatch = create_task(
                registry.dispatch_response(
                    "race",
                    _OWNER,
                    _response(
                        cast(str, outbound.jsonrpc_id),
                        {"action": "decline"},
                    ),
                )
            )
            if index % 2:
                task.cancel()
            await gather(dispatch, return_exceptions=True)
            try:
                await task
            except CancelledError:
                pass
        assert await registry.pending_count("race", _OWNER) == 0

    run(scenario())


def test_unsafe_multiline_and_failed_status_hook_fail_closed() -> None:
    async def scenario() -> None:
        registry = MCPFormSessionRegistry(response_wait_seconds=1)
        await _ready(registry, preserves_newlines=False)
        multiline = _request(
            MultilineTextQuestion(
                question_id=QuestionId("notes"),
                prompt="Notes?",
                required=True,
            )
        )
        outcome = await _handler_task(registry, multiline)
        assert isinstance(outcome, InputHandlerDisconnected)
        assert (
            await registry.next_outbound(
                "session",
                _OWNER,
                timeout_seconds=0.01,
            )
            is None
        )

        unsafe = _request(
            TextQuestion(
                question_id=QuestionId("password"),
                prompt="Enter it.",
                required=True,
            ),
            request_id="unsafe",
        )
        outcome = await _handler_task(registry, unsafe)
        assert isinstance(outcome, InputHandlerDisconnected)

        await _ready(registry, session_id="hook")

        async def broken_hook(event: MCPFormStatusEvent) -> None:
            assert event.request_id == "request"
            raise RuntimeError("private hook failure")

        outcome = await _handler_task(
            registry,
            _request(
                TextQuestion(
                    question_id=QuestionId("text"),
                    prompt="Text?",
                    required=True,
                )
            ),
            session_id="hook",
            hook=broken_hook,
        )
        assert isinstance(outcome, InputHandlerDisconnected)
        assert await registry.pending_count("hook", _OWNER) == 0
        assert not registry._sessions["hook"].outbound

        await _ready(registry, session_id="final-hook")

        async def final_broken_hook(event: MCPFormStatusEvent) -> None:
            if event.status is not MCPFormStatus.INPUT_REQUIRED:
                raise RuntimeError("private final hook failure")

        task = _handler_task(
            registry,
            _request(
                TextQuestion(
                    question_id=QuestionId("text"),
                    prompt="Text?",
                    required=True,
                )
            ),
            session_id="final-hook",
            hook=final_broken_hook,
        )
        outbound = await registry.next_outbound("final-hook", _OWNER)
        assert outbound is not None
        await registry.dispatch_response(
            "final-hook",
            _OWNER,
            _response(
                cast(str, outbound.jsonrpc_id),
                {"action": "decline"},
            ),
        )
        assert isinstance(await task, InputHandlerDisconnected)

    run(scenario())
