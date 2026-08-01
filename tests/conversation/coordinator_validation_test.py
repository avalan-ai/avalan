"""Exercise closed coordinator validation and fail-closed branches."""

from asyncio import CancelledError
from collections.abc import Generator
from copy import copy
from dataclasses import replace
from datetime import datetime
from typing import cast

import pytest
from phase2_fixtures import (
    NOW,
    authority,
    binding,
    child_identity,
    empty_stateless_plan,
    first_stored_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation
from avalan.conversation import coordinator as coordinator_module
from avalan.conversation import fakes as fakes_module
from avalan.conversation.coordinator import _AttemptStaging
from avalan.conversation.protocols import CoordinatorBoundaryHook

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic coordinator validation on asyncio only."""
    return "asyncio"


class _ExplodingObserver:
    async def publish(
        self, observation: conversation.ConversationObservation
    ) -> None:
        raise RuntimeError("observer failure")


class _SpoofProvider:
    """Mimic an ordinary provider without repository construction."""

    def __init__(self, result: conversation.ProviderResult) -> None:
        self._result = result
        self.dispatches = 0
        self.streams = 0
        self.awaits = 0

    async def dispatch(
        self,
        plan: conversation.ProviderPlan,
    ) -> conversation.ProviderResult:
        self.dispatches += 1
        return self._result

    async def stream(
        self,
        plan: conversation.ProviderPlan,
    ) -> conversation.ConversationProviderStream:
        self.streams += 1
        raise AssertionError("spoof provider must never stream")

    def __await__(self) -> Generator[None, None, None]:
        self.awaits += 1
        if False:
            yield None
        raise AssertionError("spoof provider must never be awaited")


class _MissingCapabilityProvider:
    """Implement provider effects without any fake capability proof."""

    def __init__(self, result: conversation.ProviderResult) -> None:
        self._result = result

    async def dispatch(
        self,
        plan: conversation.ProviderPlan,
    ) -> conversation.ProviderResult:
        return self._result

    async def stream(
        self,
        plan: conversation.ProviderPlan,
    ) -> conversation.ConversationProviderStream:
        raise AssertionError("unproved provider must never dispatch")


class _WrappedFakeProvider:
    """Delegate to a real fake while presenting the proof on a wrapper."""

    def __init__(
        self,
        provider: conversation.ConversationProvider,
    ) -> None:
        self._provider = provider

    async def dispatch(
        self,
        plan: conversation.ProviderPlan,
    ) -> conversation.ProviderResult:
        return await self._provider.dispatch(plan)

    async def stream(
        self,
        plan: conversation.ProviderPlan,
    ) -> conversation.ConversationProviderStream:
        return await self._provider.stream(plan)


def _runtime(
    lane_binding: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
    *,
    max_output_items: int = 1_024,
) -> conversation.ConversationLaneRuntime:
    return conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=results
        ),
        max_output_items=max_output_items,
    )


def _engine(
    *,
    store: conversation.ConversationStore,
    scope: conversation.AuthorityScope,
    lanes: tuple[conversation.ConversationLaneRuntime, ...],
    observer: conversation.ConversationObserver | None = None,
    max_attempts: int = 2,
    max_active_executions: int = 128,
    boundary_hook: CoordinatorBoundaryHook | None = None,
) -> conversation.RunScopedConversationCoordinator:
    return conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=observer or conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=lanes,
        max_attempts=max_attempts,
        max_active_executions=max_active_executions,
        boundary_hook=boundary_hook,
    )


def _tool_item(
    lane_id: conversation.ProviderLaneId,
) -> conversation.ProviderItem:
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId("tool-output-item"),
        lane_id=lane_id,
        model_call_id=conversation.ConversationModelCallId("tool-model-call"),
        kind=conversation.ProviderItemKind.FUNCTION_CALL_OUTPUT,
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.TOOL,
        caller=conversation.ProviderItemCaller.TOOL,
        canonical_input={
            "call_id": "tool-call",
            "output": "safe-output",
            "type": "function_call_output",
        },
        normalization_version=(
            conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
        ),
        call_id=conversation.ProviderCallId("tool-call"),
    )


def _compaction_item(
    lane_id: conversation.ProviderLaneId,
) -> conversation.ProviderItem:
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId("compaction-item"),
        lane_id=lane_id,
        model_call_id=conversation.ConversationModelCallId(
            "compaction-model-call"
        ),
        kind=conversation.ProviderItemKind.COMPACTION,
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.COMPACTION,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input={"type": "compaction"},
        normalization_version=(
            conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
        ),
        opaque_state=conversation.OpaqueProviderState(_value=b"compaction"),
    )


def test_lane_runtime_coordinator_and_staging_validation_is_closed() -> None:
    """Reject invalid lane runtimes, coordinator inputs, and staged items."""
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    runtime = _runtime(lane_binding, (result,))
    profile = conversation.fake_capability_profile(lane_binding)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_lane_runtime(object())
    missing_runtime = object.__new__(conversation.ConversationLaneRuntime)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_lane_runtime(missing_runtime)
    missing_state_runtime = object.__new__(
        conversation.ConversationLaneRuntime
    )
    object.__setattr__(missing_state_runtime, "binding", lane_binding)
    object.__setattr__(missing_state_runtime, "capability_profile", profile)
    object.__setattr__(
        missing_state_runtime,
        "provider_script",
        runtime.provider_script,
    )
    object.__setattr__(
        missing_state_runtime,
        "retention_policy",
        conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    object.__setattr__(missing_state_runtime, "max_output_items", 1_024)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_lane_runtime(missing_state_runtime)
    for values in (
        {"binding": cast(conversation.ProviderLaneBinding, object())},
        {
            "capability_profile": cast(
                conversation.ConversationCapabilityProfile, object()
            )
        },
        {
            "retention_policy": cast(
                conversation.ChildLaneRetentionPolicy, "invalid"
            )
        },
        {"max_output_items": 0},
        {"max_output_items": cast(int, True)},
    ):
        with pytest.raises(conversation.ConversationError):
            replace(runtime, **values)
    non_fake_binding = replace(
        lane_binding,
        adapter_type="external.Adapter",
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        conversation.ConversationLaneRuntime(
            binding=non_fake_binding,
            capability_profile=conversation.fake_capability_profile(
                non_fake_binding
            ),
            provider_script=conversation.DeterministicFakeProviderScript(
                results=(result,)
            ),
        )

    for lanes in (
        cast(tuple[conversation.ConversationLaneRuntime, ...], ()),
        cast(tuple[conversation.ConversationLaneRuntime, ...], (object(),)),
        (runtime, runtime),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            _engine(
                store=conversation.InMemoryConversationStore(),
                scope=authority(),
                lanes=lanes,
            )
    with pytest.raises(conversation.ConversationValidationError):
        _engine(
            store=conversation.InMemoryConversationStore(),
            scope=authority(),
            lanes=(runtime,),
            max_attempts=0,
        )
    for max_active_executions in (0, cast(int, True)):
        with pytest.raises(conversation.ConversationValidationError):
            _engine(
                store=conversation.InMemoryConversationStore(),
                scope=authority(),
                lanes=(runtime,),
                max_active_executions=max_active_executions,
            )
    assert profile.test_only

    diagnostics_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=authority(),
        lanes=(runtime,),
    )
    with pytest.raises(conversation.ConversationValidationError):
        diagnostics_engine.fake_provider_diagnostics(
            conversation.ProviderLaneId("missing-lane")
        )

    staging = _AttemptStaging(lane_id=lane_binding.lane_id, items=[])
    with pytest.raises(conversation.ConversationValidationError):
        staging.accept(cast(conversation.ProviderItem, object()))
    with pytest.raises(conversation.ConversationValidationError):
        staging.finish(result)
    tool_staging = _AttemptStaging(lane_id=lane_binding.lane_id, items=[])
    tool_staging.accept(_tool_item(lane_binding.lane_id))
    assert tool_staging.tool_effect


def test_fake_provider_admission_is_inert_exact_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admit only inert exact scripts without registry or closure authority."""
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    profile = conversation.fake_capability_profile(lane_binding)
    script_type = conversation.DeterministicFakeProviderScript
    script = script_type(results=(result,))
    copied_script = copy(script)
    assert not hasattr(script, "dispatch")
    assert not hasattr(script, "stream")
    assert not hasattr(script, "__await__")

    for removed_name in (
        "is_repository_deterministic_fake_provider",
        "repository_deterministic_fake_provider_count",
        "_build_deterministic_fake_provider",
    ):
        assert not hasattr(conversation, removed_name)
        assert not hasattr(fakes_module, removed_name)
    assert fakes_module._validate_fake_provider_script.__closure__ is None
    provider_names = tuple(
        name for name in vars(fakes_module) if "fake_provider" in name.lower()
    )
    assert not any(
        authority_name in name.lower()
        for name in provider_names
        for authority_name in (
            "registry",
            "token",
            "issuer",
            "marker",
            "capability",
        )
    )

    spoof = _SpoofProvider(result)
    spoof_subclass_type = type(
        "SpoofProviderSubclass",
        (_SpoofProvider,),
        {},
    )
    spoof_subclass = spoof_subclass_type(result)
    wrapped = _WrappedFakeProvider(spoof)
    missing = _MissingCapabilityProvider(result)
    raw_spoof = object.__new__(_SpoofProvider)
    raw_spoof._result = result
    raw_spoof.dispatches = 0
    raw_spoof.streams = 0
    raw_spoof.awaits = 0
    copied_spoof = copy(spoof)

    script_subclass = type("ScriptSubclass", (script_type,), {})
    with pytest.raises(conversation.ConversationValidationError):
        script_subclass(results=(result,))

    monkeypatch.setattr(
        script_type,
        "__eq__",
        lambda _self, _other: True,
        raising=False,
    )
    monkeypatch.setattr(
        script_type,
        "__hash__",
        lambda _self: 0,
        raising=False,
    )
    for candidate in (
        spoof,
        spoof_subclass,
        wrapped,
        missing,
        raw_spoof,
        copied_spoof,
        object(),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.ConversationLaneRuntime(
                binding=lane_binding,
                capability_profile=profile,
                provider_script=cast(
                    conversation.DeterministicFakeProviderScript,
                    candidate,
                ),
            )

    accepted = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=profile,
        provider_script=copied_script,
    )
    assert accepted.provider.plans == ()
    assert accepted.provider.remaining_results == 1

    class _ReplacementScript:
        pass

    with monkeypatch.context() as replaced:
        replaced.setattr(
            fakes_module,
            "DeterministicFakeProviderScript",
            _ReplacementScript,
        )
        replaced.setattr(
            conversation,
            "DeterministicFakeProviderScript",
            _ReplacementScript,
        )
        still_accepted = conversation.ConversationLaneRuntime(
            binding=lane_binding,
            capability_profile=profile,
            provider_script=script,
        )
        assert still_accepted.provider.plans == ()

    assert spoof.dispatches == 0
    assert spoof.streams == 0
    assert spoof.awaits == 0
    assert raw_spoof.dispatches == 0
    assert raw_spoof.streams == 0
    assert raw_spoof.awaits == 0
    assert copied_spoof.dispatches == 0
    assert copied_spoof.streams == 0
    assert copied_spoof.awaits == 0
    assert spoof_subclass.dispatches == 0
    assert spoof_subclass.streams == 0
    assert spoof_subclass.awaits == 0


async def test_provider_runtime_tampering_cannot_execute_custom_effects() -> (
    None
):
    """Reject raw, copied, substituted, and replaced effect authority."""
    scope = authority()
    lane_binding = binding()
    result = conversation.fake_provider_result(
        empty_stateless_plan(lane_binding),
        turn=1,
    )
    profile = conversation.fake_capability_profile(lane_binding)
    script = conversation.DeterministicFakeProviderScript(results=(result,))
    spy = _SpoofProvider(result)

    raw_runtime = object.__new__(conversation.ConversationLaneRuntime)
    object.__setattr__(raw_runtime, "binding", lane_binding)
    object.__setattr__(raw_runtime, "capability_profile", profile)
    object.__setattr__(raw_runtime, "provider_script", spy)
    object.__setattr__(
        raw_runtime,
        "retention_policy",
        conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    object.__setattr__(raw_runtime, "max_output_items", 1_024)
    object.__setattr__(raw_runtime, "_provider_runtime", spy)
    with pytest.raises(conversation.ConversationValidationError):
        _engine(
            store=conversation.InMemoryConversationStore(),
            scope=scope,
            lanes=(raw_runtime,),
        )

    copied_runtime = copy(
        conversation.ConversationLaneRuntime(
            binding=lane_binding,
            capability_profile=profile,
            provider_script=script,
        )
    )
    object.__setattr__(copied_runtime, "provider_script", spy)
    with pytest.raises(conversation.ConversationValidationError):
        _engine(
            store=conversation.InMemoryConversationStore(),
            scope=scope,
            lanes=(copied_runtime,),
        )

    substituted_runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=profile,
        provider_script=script,
    )
    substituted_store = conversation.InMemoryConversationStore()
    substituted_engine = _engine(
        store=substituted_store,
        scope=scope,
        lanes=(substituted_runtime,),
    )
    object.__setattr__(substituted_runtime, "provider_script", spy)
    with pytest.raises(conversation.ConversationValidationError):
        await substituted_engine.execute(
            request(
                scope=scope,
                identity=root_identity("substituted-script"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="substituted-script",
                key="key-substituted-script",
            )
        )
    assert substituted_store.diagnostics.staged_executions == 0

    replaced_runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=profile,
        provider_script=conversation.DeterministicFakeProviderScript(
            results=(result,)
        ),
    )
    replaced_store = conversation.InMemoryConversationStore()
    replaced_engine = _engine(
        store=replaced_store,
        scope=scope,
        lanes=(replaced_runtime,),
    )
    cast(
        dict[conversation.ProviderLaneId, object],
        replaced_engine._fake_runtimes,
    )[lane_binding.lane_id] = spy
    with pytest.raises(conversation.ConversationValidationError):
        await replaced_engine.execute(
            request(
                scope=scope,
                identity=root_identity("replaced-provider-state"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="replaced-provider-state",
                key="key-replaced-provider-state",
            )
        )
    assert replaced_store.diagnostics.staged_executions == 0
    assert spy.dispatches == 0
    assert spy.streams == 0
    assert spy.awaits == 0


async def test_raw_valid_script_runs_only_canonical_repository_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run raw inert data canonically and reject malformed internal state."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    script_type = conversation.DeterministicFakeProviderScript
    raw_script = object.__new__(script_type)
    object.__setattr__(raw_script, "results", (result,))
    controller = conversation.DeterministicFaultController()
    object.__setattr__(raw_script, "controller", controller)
    runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=raw_script,
    )
    provider_runtime = runtime._provider_runtime
    assert (
        type(provider_runtime)
        is fakes_module._DeterministicFakeProviderRuntime
    )
    assert getattr(type(provider_runtime), "__final__", False)
    assert not hasattr(fakes_module, "_DeterministicFakeProviderExecutor")

    class _ReplacementController:
        reaches = 0

        async def reach(self, label: str) -> None:
            type(self).reaches += 1

    patched_reaches = 0

    async def patched_reach(
        _controller: object,
        _label: str,
    ) -> None:
        nonlocal patched_reaches
        patched_reaches += 1

    monkeypatch.setattr(
        fakes_module,
        "DeterministicFaultController",
        _ReplacementController,
    )
    monkeypatch.setattr(
        conversation,
        "DeterministicFaultController",
        _ReplacementController,
    )
    monkeypatch.setattr(type(controller), "reach", patched_reach)
    engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(runtime,),
    )
    await engine.execute(
        request(
            scope=scope,
            identity=root_identity("raw-inert-script"),
            advance=conversation.FirstTurnAdvance(),
            response_suffix="raw-inert-script",
            key="key-raw-inert-script",
        )
    )
    diagnostics = engine.fake_provider_diagnostics(lane_binding.lane_id)
    assert diagnostics.plans == (plan,)
    assert diagnostics.remaining_results == 0
    assert controller.visited.count("provider:dispatch") == 1
    assert _ReplacementController.reaches == 0
    assert patched_reaches == 0

    missing_script = object.__new__(script_type)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ConversationLaneRuntime(
            binding=lane_binding,
            capability_profile=conversation.fake_capability_profile(
                lane_binding
            ),
            provider_script=missing_script,
        )

    malformed_runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=script_type(results=(result,)),
    )
    raw_state = object.__new__(fakes_module._DeterministicFakeProviderRuntime)
    object.__setattr__(malformed_runtime, "_provider_runtime", raw_state)
    with pytest.raises(conversation.ConversationValidationError):
        _engine(
            store=conversation.InMemoryConversationStore(),
            scope=scope,
            lanes=(malformed_runtime,),
        )


async def test_provider_execution_ignores_post_import_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call captured provider functions despite module and class patches."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    dispatch_runtime = _runtime(lane_binding, (result,))
    stream_runtime = _runtime(lane_binding, (result,))
    dispatch_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(dispatch_runtime,),
    )
    stream_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(stream_runtime,),
    )
    calls = {
        "dispatch": 0,
        "open": 0,
        "item": 0,
        "terminal": 0,
        "close": 0,
        "validate": 0,
        "helper": 0,
        "controller": 0,
    }

    def replacement(name: str) -> object:
        def invoke(*_args: object, **_kwargs: object) -> None:
            calls[name] += 1
            raise AssertionError(f"replacement {name} must never execute")

        return invoke

    def async_replacement(name: str) -> object:
        async def invoke(*_args: object, **_kwargs: object) -> None:
            calls[name] += 1
            raise AssertionError(f"replacement {name} must never execute")

        return invoke

    async_replacements = {
        "_dispatch_deterministic_fake_provider": "dispatch",
        "_open_deterministic_fake_provider_stream": "open",
        "_next_deterministic_fake_provider_item": "item",
        "_terminal_deterministic_fake_provider_stream": "terminal",
        "_close_deterministic_fake_provider_stream": "close",
        "_reach_fault_controller": "controller",
        "_fault_controller_reach": "controller",
    }
    sync_replacements = {
        "_validate_deterministic_fake_provider_runtime": "validate",
        "_validate_deterministic_fake_provider_stream": "helper",
        "_owned_deterministic_fake_provider_stream": "helper",
        "_next_deterministic_fake_provider_result": "helper",
        "_validate_provider_plan": "helper",
        "_validate_fake_provider_script": "helper",
        "_validate_fault_controller": "controller",
        "_validate_fault_controller_state": "controller",
        "_canonical_provider_result": "helper",
        "_canonical_provider_item": "helper",
        "_validate_frozen_json_value": "helper",
        "isfinite": "helper",
    }
    for module in (fakes_module, coordinator_module):
        for name, counter in async_replacements.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, async_replacement(counter))
        for name, counter in sync_replacements.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, replacement(counter))

    controller_type = type(dispatch_runtime._provider_runtime.controller)
    monkeypatch.setattr(
        controller_type,
        "reach",
        async_replacement("controller"),
    )

    dispatch_receipt = await dispatch_engine.execute(
        request(
            scope=scope,
            identity=root_identity("captured-dispatch"),
            advance=conversation.FirstTurnAdvance(),
            response_suffix="captured-dispatch",
            key="key-captured-dispatch",
        )
    )
    stream_receipt = await stream_engine.stream(
        request(
            scope=scope,
            identity=root_identity("captured-stream"),
            advance=conversation.FirstTurnAdvance(),
            response_suffix="captured-stream",
            key="key-captured-stream",
        )
    )
    assert dispatch_receipt.result is not None
    assert stream_receipt.result is not None
    assert (
        dispatch_engine.fake_provider_diagnostics(
            lane_binding.lane_id
        ).remaining_results
        == 0
    )
    assert (
        stream_engine.fake_provider_diagnostics(
            lane_binding.lane_id
        ).remaining_results
        == 0
    )
    assert calls == {name: 0 for name in calls}


def test_fake_fault_controller_rejects_override_authority() -> None:
    """Reject controller wrappers, subclasses, and malformed raw state."""

    async def subclass_reach(_self: object, _label: str) -> None:
        raise AssertionError("subclass controller must never execute")

    controller_subclass = cast(
        type[conversation.DeterministicFaultController],
        type(
            "ControllerSubclass",
            (conversation.DeterministicFaultController,),
            {"reach": subclass_reach},
        ),
    )

    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFakeProviderScript(
            results=(
                conversation.fake_provider_result(
                    empty_stateless_plan(binding()),
                    turn=1,
                ),
            ),
            controller=controller_subclass(),
        )

    controller = conversation.DeterministicFaultController()
    with pytest.raises(AttributeError):
        object.__setattr__(controller, "reach", lambda _label: None)

    raw_controller = object.__new__(conversation.DeterministicFaultController)
    object.__setattr__(raw_controller, "_actions", object())
    object.__setattr__(raw_controller, "_visited", [])
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFakeProviderScript(
            results=(
                conversation.fake_provider_result(
                    empty_stateless_plan(binding()),
                    turn=1,
                ),
            ),
            controller=raw_controller,
        )


async def test_stream_close_fault_precedence_and_active_limit() -> None:
    """Preserve canonical close faults and bound concurrent run ownership."""
    scope = authority()
    lane_binding = binding(streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)

    cancelled_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:close",
                exception=CancelledError(),
            ),
            conversation.FaultAction(
                label="provider:close:retry:1",
                exception=RuntimeError("retry-close"),
            ),
        )
    )
    cancelled_runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=(result,),
            controller=cancelled_controller,
        ),
    )
    cancelled_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(cancelled_runtime,),
    )
    with pytest.raises(CancelledError) as cancelled_error:
        await cancelled_engine.stream(
            request(
                scope=scope,
                identity=root_identity("canonical-close-cancel"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="canonical-close-cancel",
                key="key-canonical-close-cancel",
            )
        )
    assert isinstance(cancelled_error.value.__cause__, RuntimeError)
    assert str(cancelled_error.value.__cause__) == "retry-close"
    cancelled_diagnostics = cancelled_engine.fake_provider_diagnostics(
        lane_binding.lane_id
    )
    assert cancelled_diagnostics.streams[0].close_attempts == 2
    assert cancelled_diagnostics.streams[0].closed

    hook_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_stream_close",
                exception=RuntimeError("hook-before-runtime-close"),
            ),
        )
    )
    runtime_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:close",
                exception=RuntimeError("stream-close"),
            ),
        )
    )
    doubly_failed_runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=(result,),
            controller=runtime_controller,
        ),
    )
    doubly_failed_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(doubly_failed_runtime,),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            hook_controller
        ),
    )
    with pytest.raises(
        RuntimeError, match="hook-before-runtime-close"
    ) as raised:
        await doubly_failed_engine.stream(
            request(
                scope=scope,
                identity=root_identity("canonical-double-close"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="canonical-double-close",
                key="key-canonical-double-close",
            )
        )
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "stream-close"
    assert (
        doubly_failed_engine.fake_provider_diagnostics(lane_binding.lane_id)
        .streams[0]
        .closed
    )

    hook_cancellation = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_stream_close",
                exception=CancelledError(),
            ),
        )
    )
    cleanup_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:close",
                exception=RuntimeError("cleanup-after-cancellation"),
            ),
        )
    )
    cleanup_runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=(result,),
            controller=cleanup_controller,
        ),
    )
    cleanup_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(cleanup_runtime,),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            hook_cancellation
        ),
    )
    with pytest.raises(CancelledError) as cleanup_cancelled:
        await cleanup_engine.stream(
            request(
                scope=scope,
                identity=root_identity("canonical-cleanup-cancel"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="canonical-cleanup-cancel",
                key="key-canonical-cleanup-cancel",
            )
        )
    assert isinstance(cleanup_cancelled.value.__cause__, RuntimeError)
    assert (
        str(cleanup_cancelled.value.__cause__) == "cleanup-after-cancellation"
    )

    primary_only_hook = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_stream_close",
                exception=RuntimeError("primary-only-close"),
            ),
        )
    )
    primary_only_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(_runtime(lane_binding, (result,)),),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(
            primary_only_hook
        ),
    )
    with pytest.raises(RuntimeError, match="primary-only-close"):
        await primary_only_engine.stream(
            request(
                scope=scope,
                identity=root_identity("canonical-primary-only-close"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="canonical-primary-only-close",
                key="key-canonical-primary-only-close",
            )
        )

    combined_hook = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_stream_close",
                exception=RuntimeError("combined-primary-close"),
            ),
        )
    )
    combined_cleanup = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="provider:close",
                exception=CancelledError(),
            ),
            conversation.FaultAction(
                label="provider:close:retry:1",
                exception=RuntimeError("combined-cleanup-close"),
            ),
        )
    )
    combined_runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=(result,),
            controller=combined_cleanup,
        ),
    )
    combined_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(combined_runtime,),
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(combined_hook),
    )
    with pytest.raises(CancelledError) as combined_cancelled:
        await combined_engine.stream(
            request(
                scope=scope,
                identity=root_identity("canonical-combined-close"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="canonical-combined-close",
                key="key-canonical-combined-close",
            )
        )
    assert isinstance(combined_cancelled.value.__cause__, RuntimeError)
    assert str(combined_cancelled.value.__cause__) == "combined-primary-close"
    assert isinstance(
        combined_cancelled.value.__cause__.__cause__,
        RuntimeError,
    )
    assert (
        str(combined_cancelled.value.__cause__.__cause__)
        == "combined-cleanup-close"
    )

    active_runtime = _runtime(lane_binding, (result,))
    active_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(active_runtime,),
        max_active_executions=1,
    )
    active_engine._active_attempts.add("manual-active")
    with pytest.raises(conversation.ConversationConflictError):
        await active_engine.execute(
            request(
                scope=scope,
                identity=root_identity("active-limit"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="active-limit",
                key="key-active-limit",
            )
        )
    assert active_engine.diagnostics.active_attempts == 1


async def test_close_internal_replay_observer_and_commit_fail_closed() -> None:
    """Close idempotently, replay internal commits, and isolate observers."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(_runtime(lane_binding, (result,)),),
    )
    engine._active_attempts.add("active-attempt")
    with pytest.raises(conversation.ConversationConflictError):
        await engine.close()
    engine._active_attempts.clear()
    await engine.close()
    await engine.close()
    with pytest.raises(conversation.ConversationValidationError):
        await engine.execute(
            cast(conversation.ConversationRunRequest, object())
        )

    result = conversation.fake_provider_result(plan, turn=2)
    engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(_runtime(lane_binding, (result,)),),
    )
    internal = request(
        scope=scope,
        identity=root_identity("internal-replay"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="internal-replay",
        key="internal-replay-key",
        boundary=conversation.ConversationCommitBoundary.INTERNAL_SEGMENT,
    )
    first = await engine.execute(internal)
    replay = await engine.execute(internal)
    assert replay.checkpoint == first.checkpoint
    assert replay.result is None

    result = conversation.fake_provider_result(plan, turn=3)
    engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(_runtime(lane_binding, (result,)),),
        observer=_ExplodingObserver(),
    )
    observed = await engine.execute(
        request(
            scope=scope,
            identity=root_identity("observer-failure"),
            advance=conversation.FirstTurnAdvance(),
            response_suffix="observer-failure",
            key="observer-failure-key",
        )
    )
    assert (
        observed.checkpoint.lifecycle
        is conversation.CheckpointLifecycle.COMMITTED
    )

    result = conversation.fake_provider_result(plan, turn=4)
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="store:commit_atomic",
                exception=RuntimeError("untyped commit failure"),
            ),
        )
    )
    failed_observer = conversation.DeterministicFakeObserver()
    engine = _engine(
        store=conversation.InMemoryConversationStore(
            boundary_hook=conversation.FakeStoreBoundaryHook(controller)
        ),
        scope=scope,
        lanes=(_runtime(lane_binding, (result,)),),
        observer=failed_observer,
    )
    with pytest.raises(conversation.ConversationCommitError):
        await engine.execute(
            request(
                scope=scope,
                identity=root_identity("commit-wrapper"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="commit-wrapper",
                key="commit-wrapper-key",
            )
        )
    assert failed_observer.observations == ()


async def test_parent_and_mode_mismatches_reject_before_dispatch() -> None:
    """Reject invalid child identity and prior-lane mode combinations."""
    scope = authority()
    lane_binding = binding()
    first_plan = empty_stateless_plan(lane_binding)
    first_result = conversation.fake_provider_result(first_plan, turn=1)
    runtime = _runtime(lane_binding, (first_result,))
    store = conversation.InMemoryConversationStore()
    engine = _engine(store=store, scope=scope, lanes=(runtime,))
    root = await engine.execute(
        request(
            scope=scope,
            identity=root_identity("parent-validation"),
            advance=conversation.FirstTurnAdvance(),
            response_suffix="parent-validation",
            key="parent-validation-key",
        )
    )
    parent_id = root.checkpoint.identity.checkpoint_id
    base_identity = child_identity(root.checkpoint, "invalid-child")
    invalid_identities = (
        replace(
            base_identity,
            conversation_id=conversation.ConversationId(
                "different-conversation"
            ),
        ),
        replace(
            base_identity,
            branch_id=conversation.ConversationBranchId("different-branch"),
            checkpoint_id=conversation.CheckpointId("branch-mismatch"),
        ),
    )
    for index, identity in enumerate(invalid_identities):
        child = request(
            scope=scope,
            identity=identity,
            advance=conversation.OrdinaryChildAdvance(
                parent_checkpoint_id=parent_id
            ),
            response_suffix=f"invalid-{index}",
            key=f"invalid-key-{index}",
        )
        with pytest.raises(conversation.ConversationValidationError):
            await engine.execute(child)
    same_branch = child_identity(root.checkpoint, "same-branch")
    with pytest.raises(conversation.ConversationValidationError):
        await engine.execute(
            request(
                scope=scope,
                identity=same_branch,
                advance=conversation.ExplicitBranchAdvance(
                    parent_checkpoint_id=parent_id,
                    branch_id=same_branch.branch_id,
                ),
                response_suffix="same-branch",
                key="same-branch-key",
            )
        )

    with pytest.raises(conversation.ConversationValidationError):
        await engine.execute(
            request(
                scope=scope,
                identity=child_identity(root.checkpoint, "stored-mismatch"),
                advance=conversation.OrdinaryChildAdvance(
                    parent_checkpoint_id=parent_id
                ),
                modes=(conversation.ConversationMode.STORED,),
                stored_retention=True,
                response_suffix="stored-mismatch",
                key="stored-mismatch-key",
            )
        )

    stored_binding = binding("stored-prior")
    first_stored = first_stored_plan(stored_binding)
    first_stored_result = conversation.fake_provider_result(
        first_stored, turn=1
    )
    continued_plan = conversation.StoredProviderPlan(
        binding=stored_binding,
        upstream_response_id=cast(
            conversation.UpstreamResponseId,
            first_stored_result.upstream_response_id,
        ),
        reasoning=first_stored.reasoning,
    )
    continued_result = conversation.fake_provider_result(
        continued_plan, turn=2
    )
    stored_runtime = _runtime(
        stored_binding, (first_stored_result, continued_result)
    )
    stored_store = conversation.InMemoryConversationStore()
    stored_engine = _engine(
        store=stored_store, scope=scope, lanes=(stored_runtime,)
    )
    stored_root = await stored_engine.execute(
        request(
            scope=scope,
            identity=root_identity("stored-prior"),
            advance=conversation.FirstTurnAdvance(),
            lane_ids=("stored-prior",),
            modes=(conversation.ConversationMode.STORED,),
            stored_retention=True,
            response_suffix="stored-prior",
            key="stored-prior-key",
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await stored_engine.execute(
            request(
                scope=scope,
                identity=child_identity(
                    stored_root.checkpoint, "stateless-mismatch"
                ),
                advance=conversation.OrdinaryChildAdvance(
                    parent_checkpoint_id=(
                        stored_root.checkpoint.identity.checkpoint_id
                    )
                ),
                lane_ids=("stored-prior",),
                modes=(conversation.ConversationMode.STATELESS,),
                response_suffix="stateless-mismatch",
                key="stateless-mismatch-key",
            )
        )
    stored_child = await stored_engine.execute(
        request(
            scope=scope,
            identity=child_identity(stored_root.checkpoint, "stored-child"),
            advance=conversation.OrdinaryChildAdvance(
                parent_checkpoint_id=stored_root.checkpoint.identity.checkpoint_id
            ),
            lane_ids=("stored-prior",),
            modes=(conversation.ConversationMode.STORED,),
            stored_retention=True,
            response_suffix="stored-child",
            key="stored-child-key",
        )
    )
    assert isinstance(
        stored_child.checkpoint.content.lanes[0],
        conversation.StoredProviderLaneSnapshot,
    )


async def test_capability_and_request_limits_reject_before_dispatch() -> None:
    """Exercise requested reasoning and every coordinator input limit."""
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    results = tuple(
        conversation.fake_provider_result(plan, turn=turn) for turn in (1, 2)
    )
    runtime = _runtime(lane_binding, results)
    engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(runtime,),
    )
    for suffix, context in (
        ("current", conversation.ReasoningContext.CURRENT_TURN),
        ("all", conversation.ReasoningContext.ALL_TURNS),
    ):
        run = request(
            scope=scope,
            identity=root_identity(f"reasoning-{suffix}"),
            advance=conversation.FirstTurnAdvance(),
            response_suffix=f"reasoning-{suffix}",
            key=f"reasoning-{suffix}-key",
        )
        run = replace(
            run,
            semantics=replace(run.semantics, reasoning_context=context),
            lanes=(replace(run.lanes[0], reasoning_context=context),),
        )
        await engine.execute(run)

    limit_runtime = _runtime(lane_binding, (results[0],))
    limit_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(limit_runtime,),
    )
    missing_lane = request(
        scope=scope,
        identity=root_identity("missing-lane"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=("unconfigured-lane",),
        response_suffix="missing-lane",
        key="missing-lane-key",
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await limit_engine.execute(missing_lane)
    with pytest.raises(conversation.ConversationCapabilityError):
        limit_engine._plan_lanes(
            missing_lane,
            None,
            streaming=False,
        )

    semantic_limit = request(
        scope=scope,
        identity=root_identity("semantic-limit"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="semantic-limit",
        key="semantic-limit-key",
    )
    semantic_limit = replace(
        semantic_limit,
        semantics=replace(
            semantic_limit.semantics,
            semantic_input="x" * 1_048_576,
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await limit_engine.execute(semantic_limit)

    visible_limit = request(
        scope=scope,
        identity=root_identity("visible-limit"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="visible-limit",
        key="visible-limit-key",
    )
    visible_limit = replace(
        visible_limit,
        visible_delta=(
            conversation.VisibleTranscriptEntry(
                role=conversation.VisibleTranscriptRole.USER,
                content="x" * 600_000,
            ),
            conversation.VisibleTranscriptEntry(
                role=conversation.VisibleTranscriptRole.USER,
                content="y" * 600_000,
            ),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await limit_engine.execute(visible_limit)

    output_limit_runtime = _runtime(
        lane_binding, (results[0],), max_output_items=10_001
    )
    output_limit_engine = _engine(
        store=conversation.InMemoryConversationStore(),
        scope=scope,
        lanes=(output_limit_runtime,),
    )
    with pytest.raises(conversation.ConversationValidationError):
        await output_limit_engine.execute(
            request(
                scope=scope,
                identity=root_identity("output-limit"),
                advance=conversation.FirstTurnAdvance(),
                response_suffix="output-limit",
                key="output-limit-key",
            )
        )


def test_lane_snapshot_compaction_and_public_helpers_are_closed() -> None:
    """Validate lane result shape, compaction, builder, and reducer values."""
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    runtime = _runtime(
        lane_binding,
        (conversation.fake_provider_result(plan, turn=1),),
    )
    lane_request = conversation.ConversationLaneRequest(
        lane_id=lane_binding.lane_id,
        mode=conversation.ConversationMode.STATELESS,
    )
    result = conversation.fake_provider_result(plan, turn=1)
    execution_receipt = conversation.provider_lane_execution_receipt(
        authority=authority(),
        identity=root_identity("lane-snapshot"),
        binding=lane_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        upstream_response_id=None,
    )
    wrong_item = replace(
        result.items[0], lane_id=conversation.ProviderLaneId("wrong-lane")
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.RunScopedConversationCoordinator._lane_snapshot(
            lane_request,
            runtime,
            plan,
            replace(result, items=(wrong_item,)),
            execution_receipt,
        )
    stored_plan = first_stored_plan(lane_binding)
    stored_result = conversation.fake_provider_result(stored_plan, turn=3)
    stored_receipt = conversation.provider_lane_execution_receipt(
        authority=authority(),
        identity=root_identity("stored-lane-snapshot"),
        binding=lane_binding,
        mode=conversation.ConversationMode.STORED,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=stored_result.items,
        reasoning=stored_result.reasoning,
        usage=stored_result.usage,
        upstream_response_id=stored_result.upstream_response_id,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.RunScopedConversationCoordinator._lane_snapshot(
            lane_request,
            runtime,
            stored_plan,
            conversation.fake_provider_result(stored_plan, turn=2),
            stored_receipt,
        )
    stored_request = replace(
        lane_request, mode=conversation.ConversationMode.STORED
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.RunScopedConversationCoordinator._lane_snapshot(
            stored_request,
            runtime,
            stored_plan,
            replace(stored_result, upstream_response_id=None),
            stored_receipt,
        )

    compaction = _compaction_item(lane_binding.lane_id)
    compaction_result = conversation.ProviderResult(
        items=(compaction,),
        reasoning=plan.reasoning,
    )
    compaction_receipt = conversation.provider_lane_execution_receipt(
        authority=authority(),
        identity=root_identity("compaction-lane-snapshot"),
        binding=lane_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=compaction_result.items,
        reasoning=compaction_result.reasoning,
        usage=compaction_result.usage,
        upstream_response_id=None,
    )
    snapshot = conversation.RunScopedConversationCoordinator._lane_snapshot(
        lane_request,
        runtime,
        plan,
        compaction_result,
        compaction_receipt,
    )
    assert isinstance(snapshot, conversation.StatelessProviderLaneSnapshot)
    assert snapshot.compaction_boundary is not None
    assert snapshot.compaction_boundary.boundary_item_id == compaction.item_id

    run = request(
        scope=authority(),
        identity=root_identity("invalid-builder"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="invalid-builder",
        key="invalid-builder-key",
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.build_checkpoint_candidate(
            cast(conversation.ConversationRunRequest, object()),
            parent=None,
            completed_lanes=(snapshot,),
            created_at=NOW,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.build_checkpoint_candidate(
            run,
            parent=None,
            completed_lanes=cast(
                tuple[conversation.ProviderLaneSnapshot, ...], []
            ),
            created_at=NOW,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.build_checkpoint_candidate(
            run,
            parent=None,
            completed_lanes=(snapshot,),
            created_at=datetime.min,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.reduce_failure(
            cast(conversation.FailureBoundary, "invalid"),
            visible_output=False,
            tool_effect=False,
            committed=False,
            ambiguous=False,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.reduce_failure(
            conversation.FailureBoundary.VALIDATION_BEFORE_DISPATCH,
            visible_output=cast(bool, 1),
            tool_effect=False,
            committed=False,
            ambiguous=False,
        )
