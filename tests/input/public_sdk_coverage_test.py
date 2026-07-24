"""Exercise every public SDK validation and policy result branch."""

from asyncio import get_running_loop, run
from base64 import urlsafe_b64encode
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from json import dumps
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast

import pytest

import avalan.sdk as sdk_module
from avalan.agent.execution import (
    AttachedInteractionRuntime,
    ExecutionInputRequiredError,
    ExecutionTerminatedError,
)
from avalan.interaction.durable import DurableInteractionSuspension
from avalan.interaction.entities import (
    AnswerProvenance,
    BranchId,
    ContinuationId,
    InputAnswer,
    InputRequest,
    InputRequestId,
    InputRequiredResult,
    QuestionId,
    QuestionType,
    RequestState,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    RunId,
    StateRevision,
    TerminateInputContinuation,
    UserId,
)
from avalan.interaction.error import (
    InputAlreadyResolvedError,
    InputAuthorizationError,
    InputContractError,
    InputErrorCode,
    InputExpiredError,
    InputNotFoundError,
    InputSupersededError,
    InputValidationError,
)
from avalan.interaction.handler import (
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputHandlerResolution,
)
from avalan.interaction.headless import (
    DeclineInputPolicy,
    DurableHandoffInputPolicy,
    ExternalControllerInputPolicy,
    PolicyValueInputPolicy,
    PredeclaredInputPolicy,
    TrustedDefaultInputPolicy,
    UnavailableInputPolicy,
)
from avalan.interaction.policy import (
    InteractionActor,
    InteractionBranchAuthorizationTarget,
    InteractionDisclosure,
    InteractionOperation,
    InteractionPolicy,
    InteractionRequestAuthorizationTarget,
    InteractionScopeAuthorizationTarget,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
)
from avalan.interaction.state import InputTransitionError
from avalan.interaction.store import (
    CreateInteractionCommand,
    InteractionCorrelation,
    InteractionRecord,
    InteractionStoreReplayed,
    InteractionTerminalMetadata,
    ResolutionDecisionStage,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    ScopedInteractionLookup,
    apply_create_interaction,
)

sys_path.append(str(Path(__file__).parent))

import sdk_contract_test as sdk_support  # noqa: E402


def _view(
    record: InteractionRecord | None = None,
) -> sdk_module.InputRequestView:
    selected = sdk_support._pending_record() if record is None else record
    return sdk_module._request_view(selected.request)


def _refs(
    record: InteractionRecord,
) -> tuple[sdk_module.InputRequestRef, sdk_module.InputContinuationRef]:
    correlation = record.correlation
    return (
        sdk_module.InputRequestRef(
            sdk_module._encode_correlation_ref("request", correlation)
        ),
        sdk_module.InputContinuationRef(
            sdk_module._encode_correlation_ref("continuation", correlation)
        ),
    )


def _other_record() -> InteractionRecord:
    request = replace(
        sdk_support._created_request(),
        request_id=InputRequestId("request-sdk-other"),
        continuation_id=ContinuationId("continuation-sdk-other"),
        origin=replace(
            sdk_support._origin(),
            run_id=RunId("run-sdk-other"),
        ),
    )
    return apply_create_interaction(
        CreateInteractionCommand(
            actor=InteractionActor(principal=request.origin.principal),
            request=request,
        ),
        InteractionPolicy(),
    ).record


def _assert_invalid(
    path: str,
    operation: Callable[[], object],
    *,
    code: InputErrorCode = InputErrorCode.INVALID_TYPE,
) -> InputValidationError:
    with pytest.raises(InputValidationError) as captured:
        operation()
    assert captured.value.code is code
    assert captured.value.path == path
    return captured.value


def _persistence_request(
    *,
    request: object | None = None,
    request_payload: object = "{}",
    continuation_payload: object = "continuation",
    persistence_digest: object = "0" * 64,
) -> sdk_module.DurableInputPersistenceRequest:
    record = sdk_support._pending_record()
    request_id, continuation_id = _refs(record)
    return sdk_module.DurableInputPersistenceRequest(
        request_id=request_id,
        continuation_id=continuation_id,
        request=cast(
            sdk_module.InputRequestView,
            _view(record) if request is None else request,
        ),
        request_payload=cast(
            sdk_module.DurableInputRequestPayload,
            request_payload,
        ),
        continuation_payload=cast(
            sdk_module.DurableInputContinuationPayload,
            continuation_payload,
        ),
        persistence_digest=cast(str, persistence_digest),
    )


class _ControllerBridge:
    def __init__(self) -> None:
        self.inspection: object = None
        self.resolution: object = None
        self.inspection_requests: list[sdk_module.InputInspectionRequest] = []
        self.resolution_requests: list[sdk_module.InputResolutionRequest] = []

    async def inspect_input(
        self,
        request: sdk_module.InputInspectionRequest,
    ) -> object:
        self.inspection_requests.append(request)
        return self.inspection

    async def resolve_input(
        self,
        request: sdk_module.InputResolutionRequest,
    ) -> object:
        self.resolution_requests.append(request)
        return self.resolution


class _SyncInspectionBridge:
    def inspect_input(
        self,
        request: sdk_module.InputInspectionRequest,
    ) -> sdk_module.InputInspection:
        del request
        raise AssertionError("synchronous inspection must not run")

    async def resolve_input(
        self,
        request: sdk_module.InputResolutionRequest,
    ) -> sdk_module.InputResolutionResult:
        del request
        raise AssertionError("resolution must not run")


class _SyncPersistenceBridge:
    async def inspect_input(
        self,
        request: sdk_module.InputInspectionRequest,
    ) -> sdk_module.InputInspection:
        del request
        raise AssertionError("inspection must not run")

    async def resolve_input(
        self,
        request: sdk_module.InputResolutionRequest,
    ) -> sdk_module.InputResolutionResult:
        del request
        raise AssertionError("resolution must not run")

    def persist_input(
        self,
        request: sdk_module.DurableInputPersistenceRequest,
    ) -> sdk_module.DurableInputPersistenceAccepted:
        del request
        raise AssertionError("synchronous persistence must not run")


class _ProjectionBroker:
    def __init__(self, projection: object | BaseException) -> None:
        self.projection = projection

    async def inspect(self, query: ScopedInteractionLookup) -> object:
        del query
        if isinstance(self.projection, BaseException):
            raise self.projection
        return self.projection


class _PersistenceBridge:
    def __init__(self, result: str) -> None:
        self.result = result
        self.received: sdk_module.DurableInputPersistenceRequest | None = None

    async def inspect_input(
        self,
        request: sdk_module.InputInspectionRequest,
    ) -> sdk_module.InputInspection:
        del request
        raise AssertionError("inspection must not run")

    async def resolve_input(
        self,
        request: sdk_module.InputResolutionRequest,
    ) -> sdk_module.InputResolutionResult:
        del request
        raise AssertionError("resolution must not run")

    async def persist_input(
        self,
        request: sdk_module.DurableInputPersistenceRequest,
    ) -> object:
        self.received = request
        if self.result == "untyped":
            return object()
        digest = (
            request.persistence_digest
            if self.result == "accepted"
            else "0" * 64
        )
        return sdk_module.DurableInputPersistenceAccepted(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            persistence_digest=digest,
        )


def _signed_ref(
    payload: object,
) -> sdk_module.InputRequestRef:
    encoded = urlsafe_b64encode(
        dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).rstrip(b"=")
    checksum = sha256(b"avalan.public-input-ref.v1\x00" + encoded).hexdigest()
    return sdk_module.InputRequestRef(
        f"avl-input-v1.{encoded.decode('ascii')}.{checksum}"
    )


def _ref_payload(
    correlation: InteractionCorrelation,
    *,
    kind: str = "request",
) -> dict[str, object]:
    return {
        "agent_id": str(correlation.agent_id),
        "branch_id": str(correlation.branch_id),
        "continuation_id": str(correlation.continuation_id),
        "kind": kind,
        "model_call_id": str(correlation.model_call_id),
        "request_id": str(correlation.request_id),
        "run_id": str(correlation.run_id),
        "task_id": (
            None if correlation.task_id is None else str(correlation.task_id)
        ),
        "turn_id": str(correlation.turn_id),
        "version": 1,
    }


def test_public_value_objects_reject_each_invalid_member() -> None:
    """Reject invalid semantic values at every public value-object boundary."""
    record = sdk_support._pending_record()
    request_id, continuation_id = _refs(record)
    view = _view(record)

    invalid_request_views: tuple[
        tuple[str, Callable[[], object]],
        ...,
    ] = (
        (
            "request.mode",
            lambda: replace(view, mode=cast(Any, "required")),
        ),
        (
            "request.reason",
            lambda: replace(view, reason=cast(Any, object())),
        ),
        (
            "request.questions",
            lambda: replace(view, questions=cast(Any, (object(),))),
        ),
        (
            "request.created_at",
            lambda: replace(view, created_at=datetime(2026, 7, 24)),
        ),
        (
            "request.state",
            lambda: replace(view, state=cast(Any, "pending")),
        ),
        (
            "request.state_revision",
            lambda: replace(view, state_revision=cast(Any, True)),
        ),
    )
    for path, operation in invalid_request_views:
        _assert_invalid(path, operation)

    _assert_invalid(
        "handler.validation_error.code",
        lambda: sdk_module.InputValidationFeedback(
            code=cast(Any, "input.invalid_type"),
            path="answer",
            message="invalid",
        ),
    )
    _assert_invalid(
        "handler.validation_error.path",
        lambda: sdk_module.InputValidationFeedback(
            code=InputErrorCode.INVALID_TYPE,
            path="",
            message="invalid",
        ),
    )
    _assert_invalid(
        "handler.validation_error.message",
        lambda: sdk_module.InputValidationFeedback(
            code=InputErrorCode.INVALID_TYPE,
            path="answer",
            message="",
        ),
    )
    feedback = sdk_module.InputValidationFeedback(
        code=InputErrorCode.INVALID_TYPE,
        path="answer",
        message="invalid",
    )
    _assert_invalid(
        "handler.request",
        lambda: sdk_module.AttachedInputContext(request=cast(Any, object())),
    )
    _assert_invalid(
        "handler.validation_error",
        lambda: sdk_module.AttachedInputContext(
            request=view,
            validation_error=cast(Any, object()),
        ),
    )
    assert (
        sdk_module.AttachedInputContext(
            request=view,
            validation_error=feedback,
        ).validation_error
        is feedback
    )

    _assert_invalid(
        "handler.disconnect_reason",
        lambda: sdk_module.AttachedInputDisconnected(
            reason=cast(Any, "handler_unavailable")
        ),
    )
    _assert_invalid(
        "inspection.request",
        lambda: sdk_module.InputInspection(
            request_id=request_id,
            continuation_id=continuation_id,
            request=cast(Any, object()),
            detached_resumption_available=True,
        ),
    )
    _assert_invalid(
        "inspection.detached_resumption_available",
        lambda: sdk_module.InputInspection(
            request_id=request_id,
            continuation_id=continuation_id,
            request=view,
            detached_resumption_available=cast(Any, 1),
        ),
    )
    _assert_invalid(
        "submission.answers",
        lambda: sdk_module.InputAnswerSubmission(
            answers=cast(Any, [sdk_support._answer()]),
            provenance=AnswerProvenance.HUMAN,
        ),
    )
    _assert_invalid(
        "submission.provenance",
        lambda: sdk_module.InputAnswerSubmission(
            answers=(),
            provenance=cast(Any, AnswerProvenance.POLICY),
        ),
        code=InputErrorCode.FORBIDDEN,
    )
    _assert_invalid(
        "submission.provenance",
        lambda: sdk_module.InputDeclineSubmission(
            provenance=cast(Any, AnswerProvenance.POLICY)
        ),
        code=InputErrorCode.FORBIDDEN,
    )
    _assert_invalid(
        "resolution.interaction_state",
        lambda: sdk_module.InputResolutionAccepted(
            interaction_state=cast(Any, "cancelled"),
            idempotent=False,
        ),
    )
    _assert_invalid(
        "resolution.idempotent",
        lambda: sdk_module.InputResolutionAccepted(
            interaction_state="answered",
            idempotent=cast(Any, 1),
        ),
    )
    decline = sdk_module.InputDeclineSubmission()
    _assert_invalid(
        "idempotency_key",
        lambda: sdk_module.InputResolutionRequest(
            request_id=request_id,
            continuation_id=continuation_id,
            submission=decline,
            idempotency_key=ResolutionIdempotencyKey(""),
        ),
    )
    _assert_invalid(
        "input_bridge.resolution",
        lambda: sdk_module.InputResolutionResult(
            request_id=request_id,
            continuation_id=continuation_id,
            resolution=cast(Any, object()),
        ),
    )
    _assert_invalid(
        "durable_bridge.request",
        lambda: _persistence_request(request=object()),
    )


def test_public_persistence_values_reject_empty_payloads_and_bad_digests() -> (
    None
):
    """Reject empty serialized payloads and non-canonical SHA-256 values."""
    _assert_invalid(
        "durable_bridge.request_payload",
        lambda: _persistence_request(request_payload=""),
    )
    _assert_invalid(
        "durable_bridge.continuation_payload",
        lambda: _persistence_request(continuation_payload=object()),
    )
    _assert_invalid(
        "durable_bridge.persistence_digest",
        lambda: _persistence_request(persistence_digest="A" * 64),
        code=InputErrorCode.INVALID_FORMAT,
    )
    record = sdk_support._pending_record()
    request_id, continuation_id = _refs(record)
    _assert_invalid(
        "durable_bridge.persistence_digest",
        lambda: sdk_module.DurableInputPersistenceAccepted(
            request_id=request_id,
            continuation_id=continuation_id,
            persistence_digest="short",
        ),
        code=InputErrorCode.INVALID_FORMAT,
    )


def test_public_controller_aliases_and_bridge_validation() -> None:
    """Delegate aliases and reject untyped, mismatched bridge responses."""

    async def exercise() -> None:
        record = sdk_support._pending_record()
        request_id, continuation_id = _refs(record)
        bridge = _ControllerBridge()
        controller = sdk_module.create_input_controller(
            cast(sdk_module.InputControllerBridge, bridge)
        )
        bridge.inspection = sdk_module.InputInspection(
            request_id=request_id,
            continuation_id=continuation_id,
            request=_view(record),
            detached_resumption_available=True,
        )
        inspected = await controller.inspect(request_id, continuation_id)
        assert inspected is bridge.inspection
        assert bridge.inspection_requests == [
            sdk_module.InputInspectionRequest(
                request_id=request_id,
                continuation_id=continuation_id,
            )
        ]

        decline = sdk_module.InputDeclineSubmission()
        accepted = sdk_module.InputResolutionAccepted(
            interaction_state="declined",
            idempotent=False,
        )
        bridge.resolution = sdk_module.InputResolutionResult(
            request_id=request_id,
            continuation_id=continuation_id,
            resolution=accepted,
        )
        resolved = await controller.resolve(
            request_id,
            continuation_id,
            decline,
            idempotency_key=ResolutionIdempotencyKey("resolve-alias"),
        )
        assert resolved is accepted
        assert len(bridge.resolution_requests) == 1

        bridge.inspection = object()
        with pytest.raises(InputValidationError) as invalid_inspection:
            await controller.inspect_input(request_id, continuation_id)
        assert invalid_inspection.value.path == "input_bridge.inspection"

        bridge.resolution = object()
        with pytest.raises(InputValidationError) as invalid_resolution:
            await controller.resolve_input(
                request_id,
                continuation_id,
                decline,
                idempotency_key=ResolutionIdempotencyKey("invalid-resolution"),
            )
        assert invalid_resolution.value.path == "input_bridge.resolution"

        bridge.resolution = sdk_module.InputResolutionResult(
            request_id=request_id,
            continuation_id=continuation_id,
            resolution=sdk_module.InputResolutionAccepted(
                interaction_state="answered",
                idempotent=False,
            ),
        )
        with pytest.raises(InputValidationError) as wrong_state:
            await controller.resolve_input(
                request_id,
                continuation_id,
                decline,
                idempotency_key=ResolutionIdempotencyKey("wrong-state"),
            )
        assert (
            wrong_state.value.path
            == "input_bridge.resolution.interaction_state"
        )

        other = _other_record()
        other_request_id, other_continuation_id = _refs(other)
        bridge.inspection = sdk_module.InputInspection(
            request_id=other_request_id,
            continuation_id=other_continuation_id,
            request=_view(other),
            detached_resumption_available=False,
        )
        with pytest.raises(InputValidationError) as mismatched:
            await controller.inspect_input(request_id, continuation_id)
        assert mismatched.value.code is InputErrorCode.CORRELATION_MISMATCH
        assert mismatched.value.path == "input_bridge.inspection"

    with pytest.raises(InputValidationError) as sync_inspection:
        sdk_module.create_input_controller(
            cast(sdk_module.InputControllerBridge, _SyncInspectionBridge())
        )
    assert sync_inspection.value.path == "input_bridge"

    with pytest.raises(InputValidationError) as sync_persistence:
        sdk_module.create_durable_input_integration(
            cast(
                sdk_module.DurableInputBridge,
                _SyncPersistenceBridge(),
            )
        )
    assert sync_persistence.value.path == "durable_bridge"
    run(exercise())


def test_owned_runtime_and_policy_factories_validate_callbacks() -> None:
    """Exercise owned runtime lifecycle and public policy factories."""

    async def handler(
        context: sdk_module.AttachedInputContext,
    ) -> sdk_module.AttachedInputOutcome:
        assert context.request.state is RequestState.PENDING
        return sdk_module.AttachedInputDetached()

    async def provider(
        context: sdk_module.AttachedInputContext,
    ) -> tuple[InputAnswer, ...]:
        assert context.request.state is RequestState.PENDING
        return (sdk_support._answer(AnswerProvenance.POLICY),)

    cancelled: list[sdk_module.AttachedInputContext] = []

    async def cancellation_handler(
        context: sdk_module.AttachedInputContext,
    ) -> None:
        cancelled.append(context)

    def sync_handler(
        context: sdk_module.AttachedInputContext,
    ) -> sdk_module.AttachedInputOutcome:
        del context
        return sdk_module.AttachedInputDetached()

    def sync_provider(
        context: sdk_module.AttachedInputContext,
    ) -> tuple[InputAnswer, ...]:
        del context
        return ()

    async def exercise() -> None:
        with pytest.raises(InputValidationError) as invalid_handler:
            await sdk_module.create_attached_input_runtime(
                cast(sdk_module.AttachedInputHandler, sync_handler)
            )
        assert invalid_handler.value.path == "handler"

        runtime = await sdk_module.create_attached_input_runtime()
        async with runtime as entered:
            assert entered is runtime
            internal = cast(AttachedInteractionRuntime, runtime._runtime)
            detached = await internal.handler(
                InputHandlerContext(
                    request=sdk_support._pending_record().request
                )
            )
            assert isinstance(detached, InputHandlerDetached)

        predeclared = sdk_module.create_predeclared_input_policy(
            (sdk_support._answer(AnswerProvenance.POLICY),),
            cancellation_handler=cancellation_handler,
        )
        internal_predeclared = cast(
            PredeclaredInputPolicy,
            predeclared._policy,
        )
        callback = internal_predeclared.cancellation_handler
        assert callback is not None
        context = InputHandlerContext(
            request=sdk_support._pending_record().request
        )
        await callback(context)
        assert len(cancelled) == 1
        assert cancelled[0].request.state is RequestState.PENDING

        value_policy = sdk_module.create_policy_value_input_policy(provider)
        internal_value_policy = cast(
            PolicyValueInputPolicy,
            value_policy._policy,
        )
        value_outcome = await internal_value_policy(context)
        value_resolution = cast(Any, value_outcome).resolution
        assert value_resolution.status is ResolutionStatus.ANSWERED

        assert isinstance(
            sdk_module.create_trusted_default_input_policy()._policy,
            TrustedDefaultInputPolicy,
        )
        assert isinstance(
            sdk_module.create_decline_input_policy()._policy,
            DeclineInputPolicy,
        )
        assert isinstance(
            sdk_module.create_unavailable_input_policy()._policy,
            UnavailableInputPolicy,
        )
        assert isinstance(
            sdk_module.create_external_controller_input_policy(
                handler
            )._policy,
            ExternalControllerInputPolicy,
        )

    with pytest.raises(TypeError):
        sdk_module.AgentInteractionRuntime()
    with pytest.raises(TypeError):
        sdk_module.AgentHeadlessInputPolicy()
    with pytest.raises(InputValidationError) as invalid_provider:
        sdk_module.create_policy_value_input_policy(
            cast(sdk_module.InputPolicyValueProvider, sync_provider)
        )
    assert invalid_provider.value.path == "headless.provider"
    with pytest.raises(InputValidationError) as invalid_controller:
        sdk_module.create_external_controller_input_policy(
            cast(sdk_module.AttachedInputHandler, sync_handler)
        )
    assert invalid_controller.value.path == "headless.controller"
    with pytest.raises(InputValidationError) as invalid_cancellation:
        sdk_module.create_decline_input_policy(
            cancellation_handler=cast(
                sdk_module.AttachedInputCancellationHandler,
                sync_handler,
            )
        )
    assert invalid_cancellation.value.path == "headless.cancellation_handler"
    run(exercise())


def test_durable_integration_rejects_unowned_components() -> None:
    """Reject each non-factory durable integration component independently."""

    async def exercise() -> None:
        runtime = await sdk_module.create_attached_input_runtime()
        policy = sdk_module.create_decline_input_policy()
        controller = sdk_module.create_input_controller(
            cast(sdk_module.InputControllerBridge, _ControllerBridge())
        )
        try:
            with pytest.raises(InputValidationError) as invalid_runtime:
                sdk_module.DurableInputIntegration(
                    runtime=cast(Any, object()),
                    headless_policy=policy,
                    controller=controller,
                )
            assert invalid_runtime.value.path == "durable_integration.runtime"

            with pytest.raises(InputValidationError) as invalid_policy:
                sdk_module.DurableInputIntegration(
                    runtime=runtime,
                    headless_policy=cast(Any, object()),
                    controller=controller,
                )
            assert (
                invalid_policy.value.path
                == "durable_integration.headless_policy"
            )

            with pytest.raises(InputValidationError) as invalid_controller:
                sdk_module.DurableInputIntegration(
                    runtime=runtime,
                    headless_policy=policy,
                    controller=cast(Any, object()),
                )
            assert (
                invalid_controller.value.path
                == "durable_integration.controller"
            )
        finally:
            await runtime.aclose()

    run(exercise())


def test_legacy_async_controller_aliases_and_projection_failures() -> None:
    """Exercise compatibility aliases and fail-closed projection handling."""

    async def exercise() -> None:
        broker = sdk_support._Broker(sdk_support._pending_record())
        request_id, continuation_id = _refs(broker.record)

        async def authority(correlation: InteractionCorrelation) -> bool:
            return correlation == broker.record.correlation

        async def resolver(
            command: ResolveInteractionCommand,
        ) -> ResolveInteractionApplied | InteractionStoreReplayed:
            result = (await broker.resolve(command)).store_result
            assert isinstance(
                result,
                ResolveInteractionApplied | InteractionStoreReplayed,
            )
            return result

        controller = sdk_module.AsyncInputController(
            broker=cast(Any, broker),
            actor=InteractionActor(
                principal=broker.record.request.origin.principal
            ),
            clock=sdk_support._Clock(),
            durable_authority=authority,
            durable_resolver=resolver,
        )
        inspection = await controller.inspect(
            request_id,
            continuation_id,
        )
        assert inspection.detached_resumption_available
        accepted = await controller.resolve(
            request_id,
            continuation_id,
            sdk_module.InputDeclineSubmission(),
            idempotency_key=ResolutionIdempotencyKey("legacy-alias"),
        )
        assert accepted.interaction_state == "declined"

        rejected_broker = sdk_support._Broker(sdk_support._pending_record())
        rejected_request_id, rejected_continuation_id = _refs(
            rejected_broker.record
        )

        async def rejected_authority(
            correlation: InteractionCorrelation,
        ) -> bool:
            return correlation == rejected_broker.record.correlation

        async def rejected_resolver(
            command: ResolveInteractionCommand,
        ) -> ResolveInteractionRejected:
            return ResolveInteractionRejected(
                command=command,
                error=InputTransitionError(
                    code=InputErrorCode.EXPIRED,
                    path="request.state",
                    message="request expired",
                ),
                decision_stage=ResolutionDecisionStage.DEADLINE,
            )

        with pytest.raises(InputExpiredError):
            await sdk_module.AsyncInputController(
                broker=cast(Any, rejected_broker),
                actor=InteractionActor(
                    principal=rejected_broker.record.request.origin.principal
                ),
                clock=sdk_support._Clock(),
                durable_authority=rejected_authority,
                durable_resolver=rejected_resolver,
            ).resolve_input(
                rejected_request_id,
                rejected_continuation_id,
                sdk_module.InputDeclineSubmission(),
                idempotency_key=ResolutionIdempotencyKey("rejected"),
            )

        invalid_authority_broker = sdk_support._Broker(
            sdk_support._pending_record()
        )
        invalid_request_id, invalid_continuation_id = _refs(
            invalid_authority_broker.record
        )

        async def invalid_authority(
            correlation: InteractionCorrelation,
        ) -> object:
            assert correlation == invalid_authority_broker.record.correlation
            return "yes"

        with pytest.raises(InputValidationError) as invalid_authority_result:
            await sdk_module.AsyncInputController(
                broker=cast(Any, invalid_authority_broker),
                actor=InteractionActor(
                    principal=invalid_authority_broker.record.request.origin.principal
                ),
                clock=sdk_support._Clock(),
                durable_authority=cast(Any, invalid_authority),
            ).inspect_input(
                invalid_request_id,
                invalid_continuation_id,
            )
        assert invalid_authority_result.value.path == "durable_authority"

        actor = InteractionActor(
            principal=sdk_support._pending_record().request.origin.principal
        )

        def controller_for(
            projection: object | BaseException,
        ) -> sdk_module.AsyncInputController:
            return sdk_module.AsyncInputController(
                broker=cast(Any, _ProjectionBroker(projection)),
                actor=actor,
                clock=sdk_support._Clock(),
            )

        with pytest.raises(InputAuthorizationError):
            await controller_for(InputAuthorizationError()).inspect_input(
                request_id, continuation_id
            )
        with pytest.raises(InputAuthorizationError):
            await controller_for(
                InputValidationError(
                    InputErrorCode.FORBIDDEN,
                    "store",
                    "forbidden",
                )
            ).inspect_input(request_id, continuation_id)
        with pytest.raises(InputNotFoundError):
            await controller_for(
                InputValidationError(
                    InputErrorCode.NOT_FOUND,
                    "store",
                    "missing",
                )
            ).inspect_input(request_id, continuation_id)

        unrelated_error = InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "store",
            "invalid",
        )
        with pytest.raises(InputValidationError) as reraised:
            await controller_for(unrelated_error).inspect_input(
                request_id,
                continuation_id,
            )
        assert reraised.value is unrelated_error

        missing_projections: tuple[object, ...] = (
            InteractionTerminalMetadata(
                status=ResolutionStatus.EXPIRED,
                resolved_at=sdk_support._NOW,
            ),
            object(),
            _other_record(),
        )
        for projection in missing_projections:
            with pytest.raises(InputNotFoundError):
                await controller_for(projection).inspect_input(
                    request_id,
                    continuation_id,
                )

    run(exercise())


def test_run_agent_maps_untyped_values_failures_and_input_handoff() -> None:
    """Return strict outcomes for raw values, bad wrappers, and handoffs."""

    async def return_mapping(value: object, **kwargs: object) -> object:
        assert value == "mapping"
        assert "interaction_runtime" in kwargs
        return {"value": 1}

    async def return_string(value: object, **kwargs: object) -> object:
        del value, kwargs
        return "complete"

    async def raw_policy(context: InputHandlerContext) -> InputHandlerDetached:
        del context
        return InputHandlerDetached()

    async def durable_handoff(
        suspension: DurableInteractionSuspension,
    ) -> InputRequest:
        del suspension
        raise RuntimeError("persistence unavailable")

    async def invalid_durable_handoff(
        suspension: DurableInteractionSuspension,
    ) -> InputRequest:
        del suspension
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "durable_bridge",
            "invalid suspension",
        )

    async def exercise() -> None:
        mapping = await sdk_module.run_agent(
            cast(Any, return_mapping),
            "mapping",
        )
        assert isinstance(mapping, sdk_module.AgentRunCompleted)
        assert mapping.value == {"value": 1}

        request = sdk_support._created_request()
        terminated = ExecutionTerminatedError(
            TerminateInputContinuation(
                request_id=request.request_id,
                status=ResolutionStatus.EXPIRED,
            )
        )

        async def terminate(value: object, **kwargs: object) -> object:
            del value, kwargs
            raise terminated

        failed_termination = await sdk_module.run_agent(
            cast(Any, terminate),
            "terminated",
        )
        assert isinstance(failed_termination, sdk_module.AgentRunFailed)
        assert failed_termination.code == "input.expired"

        cancelled = ExecutionTerminatedError(
            TerminateInputContinuation(
                request_id=request.request_id,
                status=ResolutionStatus.CANCELLED,
            )
        )

        async def cancel(value: object, **kwargs: object) -> object:
            del value, kwargs
            raise cancelled

        cancelled_result = await sdk_module.run_agent(
            cast(Any, cancel),
            "cancelled",
        )
        assert isinstance(cancelled_result, sdk_module.AgentRunCancelled)

        async def crash(value: object, **kwargs: object) -> object:
            del value, kwargs
            raise RuntimeError("secret internal detail")

        crashed = await sdk_module.run_agent(cast(Any, crash), "crash")
        assert isinstance(crashed, sdk_module.AgentRunFailed)
        assert crashed.code == "agent.execution_failed"
        assert crashed.message == "agent execution failed"

        corrupt_runtime = object.__new__(sdk_module.AgentInteractionRuntime)
        corrupt_runtime._runtime = object()
        corrupt_runtime._closer = sdk_module._noop_close
        corrupt_runtime_result = await sdk_module.run_agent(
            cast(Any, return_string),
            "corrupt-runtime",
            interaction_runtime=corrupt_runtime,
        )
        assert isinstance(
            corrupt_runtime_result,
            sdk_module.AgentRunFailed,
        )
        assert corrupt_runtime_result.code == InputErrorCode.INVALID_TYPE.value

        raw_runtime_result = await sdk_module.run_agent(
            cast(Any, return_string),
            "raw-runtime",
            interaction_runtime=cast(Any, object()),
        )
        assert isinstance(raw_runtime_result, sdk_module.AgentRunFailed)
        assert raw_runtime_result.code == InputErrorCode.INVALID_TYPE.value

        corrupt_policy = object.__new__(sdk_module.AgentHeadlessInputPolicy)
        corrupt_policy._policy = object()
        corrupt_policy_result = await sdk_module.run_agent(
            cast(Any, return_string),
            "corrupt-policy",
            headless_policy=corrupt_policy,
        )
        assert isinstance(corrupt_policy_result, sdk_module.AgentRunFailed)
        assert corrupt_policy_result.code == InputErrorCode.INVALID_TYPE.value

        raw_policy_result = await sdk_module.run_agent(
            cast(Any, return_string),
            "raw-policy",
            headless_policy=cast(Any, object()),
        )
        assert isinstance(raw_policy_result, sdk_module.AgentRunFailed)
        assert raw_policy_result.code == InputErrorCode.INVALID_TYPE.value

        runtime = await sdk_module.create_attached_input_runtime()
        try:
            callable_policy_result = await sdk_module.run_agent(
                cast(Any, return_string),
                "callable-policy",
                interaction_runtime=runtime,
                headless_policy=cast(Any, raw_policy),
            )
            assert isinstance(
                callable_policy_result,
                sdk_module.AgentRunCompleted,
            )

            durable_with_attached = await sdk_module.run_agent(
                cast(Any, return_string),
                "durable-with-attached",
                interaction_runtime=runtime,
                headless_policy=cast(
                    Any,
                    DurableHandoffInputPolicy(handoff=durable_handoff),
                ),
            )
            assert isinstance(
                durable_with_attached,
                sdk_module.AgentRunFailed,
            )
            assert (
                durable_with_attached.code == InputErrorCode.INVALID_TYPE.value
            )
        finally:
            await runtime.aclose()

        attached_without_runtime = await sdk_module.run_agent(
            cast(Any, return_string),
            "attached-without-runtime",
            headless_policy=cast(Any, DeclineInputPolicy()),
        )
        assert isinstance(
            attached_without_runtime,
            sdk_module.AgentRunFailed,
        )
        assert (
            attached_without_runtime.code == InputErrorCode.INVALID_TYPE.value
        )

        required = InputRequiredResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            detached_resumption_available=False,
        )

        async def require_without_request(
            value: object,
            **kwargs: object,
        ) -> object:
            del value, kwargs
            raise ExecutionInputRequiredError(required)

        unavailable_correlation = await sdk_module.run_agent(
            cast(Any, require_without_request),
            "required",
        )
        assert isinstance(
            unavailable_correlation,
            sdk_module.AgentRunFailed,
        )
        assert unavailable_correlation.code == "input.correlation_unavailable"

        async def require_attached_request(
            value: object,
            **kwargs: object,
        ) -> object:
            del value, kwargs
            raise ExecutionInputRequiredError(required, request=request)

        durable_without_suspension = await sdk_module.run_agent(
            cast(Any, require_attached_request),
            "required",
            headless_policy=cast(
                Any,
                DurableHandoffInputPolicy(handoff=durable_handoff),
            ),
        )
        assert isinstance(
            durable_without_suspension,
            sdk_module.AgentRunFailed,
        )
        assert (
            durable_without_suspension.code
            == "input.durable_handoff_unavailable"
        )

        suspension = sdk_support._suspension()
        durable_required = InputRequiredResult(
            request_id=suspension.command.request.request_id,
            continuation_id=suspension.command.request.continuation_id,
            detached_resumption_available=True,
        )

        async def require_durable(
            value: object,
            **kwargs: object,
        ) -> object:
            del value, kwargs
            raise ExecutionInputRequiredError(
                durable_required,
                durable=suspension,
            )

        failed_handoff = await sdk_module.run_agent(
            cast(Any, require_durable),
            "durable",
            headless_policy=cast(
                Any,
                DurableHandoffInputPolicy(handoff=durable_handoff),
            ),
        )
        assert isinstance(failed_handoff, sdk_module.AgentRunFailed)
        assert failed_handoff.code == "input.durable_handoff_failed"
        assert failed_handoff.retryable

        invalid_handoff = await sdk_module.run_agent(
            cast(Any, require_durable),
            "durable",
            headless_policy=cast(
                Any,
                DurableHandoffInputPolicy(handoff=invalid_durable_handoff),
            ),
        )
        assert isinstance(invalid_handoff, sdk_module.AgentRunFailed)
        assert invalid_handoff.code == InputErrorCode.INVALID_FORMAT.value
        assert not invalid_handoff.retryable

    run(exercise())


def test_principal_clock_identifiers_authorization_and_classifier() -> None:
    """Exercise runtime infrastructure with allowed and denied scopes."""

    async def exercise() -> None:
        with pytest.raises(InputValidationError) as invalid_principal:
            sdk_module._principal_scope(cast(Any, object()))
        assert invalid_principal.value.path == "principal"

        principal = sdk_module.InputPrincipal(
            user_id="user",
            tenant_id="tenant",
            participant_id="participant",
            session_id="session",
        )
        scope = sdk_module._principal_scope(principal)
        assert scope.user_id == UserId("user")

        clock = sdk_module._SystemInteractionClock()
        before = get_running_loop().time()
        observed = await clock.read()
        assert observed.monotonic_seconds >= before
        assert observed.wall_time.tzinfo is not None
        await clock.wait_until(0.0)

        identifiers = sdk_module._UuidInteractionIdFactory()
        request_id = await identifiers.new_request_id()
        continuation_id = await identifiers.new_continuation_id()
        idempotency_key = await identifiers.new_idempotency_key()
        lease_nonce = await identifiers.new_active_control_lease_nonce()
        assert str(request_id).startswith("request-")
        assert str(continuation_id).startswith("continuation-")
        assert str(idempotency_key).startswith("resolution-")
        assert str(lease_nonce).startswith("lease-")

        actor = InteractionActor(principal=scope)
        authorizer = sdk_module._BoundPrincipalAuthorizer(scope)
        request_target = InteractionRequestAuthorizationTarget(
            request_id=sdk_support._created_request().request_id,
            origin=replace(
                sdk_support._origin(),
                principal=scope,
            ),
        )
        allowed = await authorizer.authorize(
            actor,
            InteractionOperation.INSPECT,
            request_target,
        )
        assert allowed.allowed
        assert allowed.disclosure is InteractionDisclosure.FULL

        other_scope = sdk_module._principal_scope(
            sdk_module.InputPrincipal(user_id="other")
        )
        branch_target = InteractionBranchAuthorizationTarget(
            run_id=RunId("run"),
            branch_id=BranchId("child"),
            parent_branch_id=BranchId("parent"),
            principal=other_scope,
        )
        denied = await authorizer.authorize(
            actor,
            InteractionOperation.REGISTER_BRANCH,
            branch_target,
        )
        assert not denied.allowed
        assert denied.disclosure is InteractionDisclosure.NONE
        assert (
            sdk_module._authorization_target_principal(
                InteractionScopeAuthorizationTarget(
                    run_id=RunId("run"),
                    principal=scope,
                )
            )
            == scope
        )
        with pytest.raises(InputValidationError) as invalid_target:
            sdk_module._authorization_target_principal(cast(Any, object()))
        assert invalid_target.value.path == "authorization.target"

        policy = InteractionPolicy()
        classifier = sdk_module._AllowTaskInputClassifier(policy)
        classification_request = TaskInputClassificationRequest(
            value="candidate",
            request_id=InputRequestId("request-classification"),
            candidate_digest="0" * 64,
            question_id=QuestionId("question-classification"),
            semantic_type=QuestionType.TEXT,
            policy_revision="policy-revision",
        )
        classification = await classifier.classify_task_input(
            classification_request
        )
        assert classification.decision is TaskInputClassificationDecision.ALLOW
        assert classification.request_id == classification_request.request_id
        assert classification.classification_id.startswith("classification-")

    run(exercise())


def test_public_handler_adapters_preserve_each_typed_outcome() -> None:
    """Translate answer, decline, detach, disconnect, and cancellation."""
    context = InputHandlerContext(
        request=sdk_support._pending_record().request
    )

    async def answer_handler(
        public: sdk_module.AttachedInputContext,
    ) -> sdk_module.AttachedInputOutcome:
        assert public.request.state is RequestState.PENDING
        return sdk_module.InputAnswerSubmission(
            answers=(sdk_support._answer(),),
            provenance=AnswerProvenance.HUMAN,
        )

    async def decline_handler(
        public: sdk_module.AttachedInputContext,
    ) -> sdk_module.AttachedInputOutcome:
        assert public.validation_error is None
        return sdk_module.InputDeclineSubmission()

    async def disconnect_handler(
        public: sdk_module.AttachedInputContext,
    ) -> sdk_module.AttachedInputOutcome:
        assert public.request.reason == context.request.reason
        return sdk_module.AttachedInputDisconnected(
            reason=sdk_module.AttachedInputDisconnectReason.HANDLER_CANCELLED
        )

    async def invalid_handler(
        public: sdk_module.AttachedInputContext,
    ) -> sdk_module.AttachedInputOutcome:
        del public
        return cast(sdk_module.AttachedInputOutcome, object())

    cancelled: list[sdk_module.AttachedInputContext] = []

    async def cancellation_handler(
        public: sdk_module.AttachedInputContext,
    ) -> None:
        cancelled.append(public)

    async def exercise() -> None:
        answered = await sdk_module._AttachedInputHandlerAdapter(
            cast(sdk_module.AttachedInputHandler, answer_handler),
            sdk_support._Clock(),
        )(context)
        assert isinstance(answered, InputHandlerResolution)
        assert answered.resolution.status is ResolutionStatus.ANSWERED

        declined = await sdk_module._AttachedInputHandlerAdapter(
            cast(sdk_module.AttachedInputHandler, decline_handler),
            sdk_support._Clock(),
        )(context)
        assert isinstance(declined, InputHandlerResolution)
        assert declined.resolution.status is ResolutionStatus.DECLINED

        detached = await sdk_module._AttachedInputHandlerAdapter(
            sdk_module._DetachedAttachedInputHandler(),
            sdk_support._Clock(),
        )(context)
        assert isinstance(detached, InputHandlerDetached)

        disconnected = await sdk_module._AttachedInputHandlerAdapter(
            cast(sdk_module.AttachedInputHandler, disconnect_handler),
            sdk_support._Clock(),
        )(context)
        assert isinstance(disconnected, InputHandlerDisconnected)
        assert disconnected.reason is InputDisconnectReason.HANDLER_CANCELLED

        with pytest.raises(InputValidationError) as invalid:
            await sdk_module._AttachedInputHandlerAdapter(
                cast(sdk_module.AttachedInputHandler, invalid_handler),
                sdk_support._Clock(),
            )(context)
        assert invalid.value.path == "handler.outcome"

        adapter = sdk_module._cancellation_adapter(
            cast(
                sdk_module.AttachedInputCancellationHandler,
                cancellation_handler,
            )
        )
        assert adapter is not None
        await adapter(context)
        assert len(cancelled) == 1
        assert cancelled[0].request.state is RequestState.PENDING

        with pytest.raises(InputValidationError) as invalid_submission:
            sdk_module._validate_submission(object())
        assert invalid_submission.value.path == "submission"

    run(exercise())


def test_durable_handoff_rejects_untyped_and_mismatched_acknowledgements() -> (
    None
):
    """Require exact typed persistence acknowledgement and digest echo."""

    async def exercise() -> None:
        invalid = sdk_module._DurableInputBridgeHandoff(
            cast(sdk_module.DurableInputBridge, _PersistenceBridge("accepted"))
        )
        with pytest.raises(InputValidationError) as invalid_suspension:
            await invalid(cast(Any, object()))
        assert invalid_suspension.value.path == "durable_bridge.suspension"

        untyped_bridge = _PersistenceBridge("untyped")
        with pytest.raises(InputValidationError) as untyped:
            await sdk_module._DurableInputBridgeHandoff(
                cast(sdk_module.DurableInputBridge, untyped_bridge)
            )(sdk_support._suspension())
        assert untyped.value.path == "durable_bridge.persistence"
        assert untyped_bridge.received is not None

        mismatched_bridge = _PersistenceBridge("mismatched")
        with pytest.raises(InputValidationError) as mismatched:
            await sdk_module._DurableInputBridgeHandoff(
                cast(sdk_module.DurableInputBridge, mismatched_bridge)
            )(sdk_support._suspension())
        assert mismatched.value.code is InputErrorCode.CORRELATION_MISMATCH
        assert mismatched.value.path == "durable_bridge.persistence_digest"

        accepted_bridge = _PersistenceBridge("accepted")
        pending = await sdk_module._DurableInputBridgeHandoff(
            cast(sdk_module.DurableInputBridge, accepted_bridge)
        )(sdk_support._suspension())
        assert pending.state is RequestState.PENDING
        assert pending.state_revision == StateRevision(1)

    run(exercise())


def test_opaque_references_reject_type_shape_payload_and_pair_mismatches() -> (
    None
):
    """Reject malformed or differently correlated opaque public references."""
    record = sdk_support._pending_record()
    request_id, _continuation_id = _refs(record)

    _assert_invalid(
        "input_ref",
        lambda: sdk_module._decode_correlation_ref(
            cast(Any, object()),
            "request",
        ),
    )
    _assert_invalid(
        "input_ref",
        lambda: sdk_module._decode_correlation_ref(
            sdk_module.InputRequestRef("invalid"),
            "request",
        ),
        code=InputErrorCode.INVALID_FORMAT,
    )
    _assert_invalid(
        "input_ref.payload",
        lambda: sdk_module._decode_correlation_ref(
            request_id,
            "continuation",
        ),
        code=InputErrorCode.INVALID_FORMAT,
    )

    malformed_payload = _ref_payload(record.correlation)
    malformed_payload["request_id"] = ""
    _assert_invalid(
        "input_ref.payload",
        lambda: sdk_module._decode_correlation_ref(
            _signed_ref(malformed_payload),
            "request",
        ),
        code=InputErrorCode.INVALID_FORMAT,
    )

    other = _other_record()
    _other_request_id, other_continuation_id = _refs(other)
    _assert_invalid(
        "input_refs",
        lambda: sdk_module._decode_correlation_pair(
            request_id,
            other_continuation_id,
        ),
        code=InputErrorCode.CORRELATION_MISMATCH,
    )


def test_public_state_errors_map_to_exact_exception_classes() -> None:
    """Map terminal states and transition codes without leaking internals."""
    for state, error_type in (
        (RequestState.EXPIRED, InputExpiredError),
        (RequestState.SUPERSEDED, InputSupersededError),
        (RequestState.CANCELLED, InputAlreadyResolvedError),
    ):
        with pytest.raises(error_type):
            sdk_module._raise_for_non_candidate_state(state)

    with pytest.raises(InputAlreadyResolvedError):
        sdk_module._require_pending_resolution_state(RequestState.CANCELLED)
    with pytest.raises(InputExpiredError):
        sdk_module._require_pending_resolution_state(RequestState.EXPIRED)
    with pytest.raises(InputSupersededError):
        sdk_module._require_pending_resolution_state(RequestState.SUPERSEDED)

    transition_errors: tuple[
        tuple[InputErrorCode, type[InputContractError]],
        ...,
    ] = (
        (InputErrorCode.ALREADY_RESOLVED, InputAlreadyResolvedError),
        (InputErrorCode.EXPIRED, InputExpiredError),
        (InputErrorCode.SUPERSEDED, InputSupersededError),
        (InputErrorCode.NOT_FOUND, InputNotFoundError),
        (InputErrorCode.FORBIDDEN, InputAuthorizationError),
    )
    for code, transition_error_type in transition_errors:
        with pytest.raises(transition_error_type):
            sdk_module._raise_public_transition_error(
                code,
                "transition",
                "transition rejected",
            )

    with pytest.raises(InputValidationError) as validation:
        sdk_module._raise_public_transition_error(
            InputErrorCode.INVALID_FORMAT,
            "transition",
            "transition rejected",
        )
    assert validation.value.path == "transition"


def test_completed_json_rejects_every_non_finite_float() -> None:
    """Preserve the public conversion error for NaN and both infinities."""
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(
            TypeError,
            match="completed value is not JSON-compatible",
        ):
            sdk_module.AgentRunCompleted(value=value).to_json()

    completed = sdk_module.AgentRunCompleted(value={"result": True})
    assert completed.to_json() == {"result": True}
    with pytest.raises(TypeError, match="completed value is not a string"):
        completed.to_str()
