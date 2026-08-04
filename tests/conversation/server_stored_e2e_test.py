"""Exercise the authenticated durable Avalan Responses boundary."""

from asyncio import create_task, gather, run, sleep, to_thread
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import dumps, loads
from logging import Logger, getLogger
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from os import environ
from pathlib import Path
from socket import AF_INET, SOCK_STREAM, socket
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from openai import AsyncOpenAI, ConflictError, NotFoundError
from phase2_fixtures import binding, empty_stateless_plan, next_stateless_plan
from pydantic import ValidationError
from uvicorn import Config, Server

import avalan.conversation as conversation
import avalan.server.routers as server_routers
from avalan.agent.conversation_child import AgentConversationChildBinding
from avalan.agent.execution import InteractionRuntime
from avalan.agent.orchestrator import Orchestrator
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.conversation import security
from avalan.entities import GenerationSettings
from avalan.model.response.text import TextGenerationResponse
from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.server import di_get_logger, di_get_orchestrator
from avalan.server.entities import (
    ResponsesRequest,
    ServerOutputRedactionSettings,
)
from avalan.server.responses_lifecycle import (
    PreparedServedResponsesTurn,
    ResponsesAuthorityResolver,
    ServedResponseLifecycle,
    ServedResponseLifecycleRecord,
    ServedResponsesConfiguration,
    ServedResponsesDurableStore,
    ServedResponsesPolicy,
    ServedResponsesService,
    ServedResponsesTurnPlan,
    StoredResponsesResource,
    close_served_responses,
    configure_served_responses,
    start_served_responses,
)
from avalan.server.responses_schema import (
    ResponsesDeletedResource,
    ResponsesErrorEnvelope,
    ResponsesInputFile,
    ResponsesInputImage,
    ResponsesResource,
)
from avalan.server.routers import responses as responses_router
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)

_NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)
_AGENT_ID = "agent-phase9-served"
_ENDPOINT_ID = "endpoint-phase9-served"
_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
_REQUEST_FINGERPRINT = "f" * 64


@pytest.fixture
def anyio_backend() -> str:
    """Run the official-client acceptance path on asyncio only."""
    return "asyncio"


@dataclass(slots=True)
class _MutableClock:
    """Expose deterministic async wall time for lifecycle tests."""

    value: datetime = _NOW

    async def now(self) -> datetime:
        """Return the configured aware instant."""
        return self.value


class _DurableMemoryFacade:
    """Mark an in-memory semantic test double as durable for route tests."""

    durable = True

    def __init__(self, store: conversation.InMemoryConversationStore) -> None:
        self.store = store

    async def retrieve(
        self,
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationResult:
        """Delegate authorized retrieval."""
        return await self.store.retrieve(public_response_id, authority)

    async def load(
        self,
        checkpoint_id: conversation.CheckpointId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationCheckpoint:
        """Delegate authorized checkpoint loading."""
        return await self.store.load(checkpoint_id, authority)

    async def sweep(
        self,
        now: datetime,
        *,
        limit: int,
    ) -> conversation.SweepReceipt:
        """Delegate bounded expiry."""
        return await self.store.sweep(now, limit=limit)

    async def prepare_deletion(
        self,
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.LocalDeletionPreparation:
        """Delegate deletion preparation."""
        return await self.store.prepare_deletion(public_response_id, authority)

    async def tombstone(
        self,
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
        at: datetime,
    ) -> conversation.ConversationCheckpoint:
        """Delegate immediate tombstoning."""
        return await self.store.tombstone(public_response_id, authority, at)

    async def close(self) -> conversation.StoreCloseResolution:
        """Delegate owned store closure."""
        return await self.store.close()


class _ResponsesOrchestrator(Orchestrator):
    """Provide only the router-owned behavior used by the test boundary."""

    model_ids = {"served-model"}

    async def sync_messages(self, response: object) -> None:
        """Accept one completed response without legacy message retention."""
        assert isinstance(response, TextGenerationResponse)


def _authority(principal: str = "owner") -> conversation.AuthorityScope:
    """Return one network-authenticated served authority."""
    return conversation.AuthorityScope(
        source=conversation.AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=conversation.AuthorityTenantId("tenant-phase9-served"),
        principal_id=conversation.AuthorityPrincipalId(principal),
        agent_id=conversation.ConversationAgentId(_AGENT_ID),
        endpoint_id=conversation.AuthorityEndpointId(_ENDPOINT_ID),
        network_exposed=True,
    )


def _retention(ttl: int = 60) -> conversation.RetentionLimits:
    """Return durable local retention with stateless private provider lanes."""
    return conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.DURABLE,
            upstream=conversation.ProviderLaneStorage.STATELESS,
        ),
        upstream_lifetime_status=(
            conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
        local_ttl_seconds=ttl,
    )


def _provider_lane(
    plan: ServedResponsesTurnPlan,
) -> tuple[
    conversation.AgentProviderLane,
    conversation.StatelessProviderPlan,
]:
    """Build or restore one deterministic parent-agent provider lane."""
    parent = plan.parent
    conversation_id = (
        parent.identity.conversation_id
        if parent is not None
        else conversation.ConversationId(
            f"conversation-{str(plan.public_response_id)[-16:]}"
        )
    )
    model_slot = conversation.AgentModelSlot("primary")
    topology_path = conversation.parent_agent_topology_path(
        plan.authority.agent_id,
        model_slot,
    )
    if parent is None:
        seed = replace(
            binding("phase9-seed", agent=str(plan.authority.agent_id)),
            model_or_deployment="served-model",
        )
        lane_id = conversation.derive_agent_provider_lane_id(
            conversation_id=conversation_id,
            owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
            topology_path=topology_path,
            model_slot=model_slot,
            binding=seed,
        )
        lane_binding = replace(seed, lane_id=lane_id)
        provider_plan = empty_stateless_plan(
            lane_binding,
            requested=plan.reasoning_context,
        )
    else:
        snapshot = parent.content.lanes[0]
        assert isinstance(snapshot, conversation.StatelessProviderLaneSnapshot)
        lane_binding = snapshot.binding
        provider_plan = next_stateless_plan(
            lane_binding,
            snapshot.ledger.items,
            requested=plan.reasoning_context,
        )
    lane = conversation.AgentProviderLane(
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        agent_id=plan.authority.agent_id,
        topology_path=topology_path,
        model_slot=model_slot,
        binding=lane_binding,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    return lane, provider_plan


def _turn_resolver(
    store: conversation.ConversationStore,
    clock: _MutableClock,
    provider_controller: (
        conversation.DeterministicFaultController | None
    ) = None,
) -> Callable[
    [ServedResponsesTurnPlan],
    Awaitable[PreparedServedResponsesTurn],
]:
    """Return one async resolver using the actual conversation coordinator."""

    async def resolve(
        plan: ServedResponsesTurnPlan,
    ) -> PreparedServedResponsesTurn:
        lane, provider_plan = _provider_lane(plan)
        turn_number = (
            1
            if plan.parent is None
            else int(plan.parent.identity.sequence) + 2
        )
        provider_result = conversation.fake_provider_result(
            provider_plan,
            turn=turn_number,
            text=f"turn-{turn_number}:{plan.input_text}",
        )
        coordinator = conversation.RunScopedConversationCoordinator(
            store=store,
            authority_resolver=(
                conversation.DeterministicFakeAuthorityResolver(plan.authority)
            ),
            clock=clock,
            publisher=conversation.DeterministicFakePublisher(),
            observer=conversation.DeterministicFakeObserver(),
            retry_waiter=conversation.DeterministicFakeRetryWaiter(),
            lanes=(
                conversation.ConversationLaneRuntime(
                    binding=lane.binding,
                    capability_profile=conversation.fake_capability_profile(
                        lane.binding
                    ),
                    provider_script=(
                        conversation.DeterministicFakeProviderScript(
                            results=(provider_result,),
                            controller=provider_controller,
                        )
                    ),
                ),
            ),
            boundary_hook=plan.hardening_hook,
            hardening_required=plan.hardening_required,
        )
        suffix = str(plan.public_response_id).removeprefix("resp_avl_")
        topology = conversation.AgentLaneTopology(
            conversation_id=(
                plan.parent.identity.conversation_id
                if plan.parent is not None
                else conversation.ConversationId(
                    f"conversation-{suffix[-16:]}"
                )
            ),
            lanes=(lane,),
        )
        turn = conversation.AgentConversationTurn(
            coordinator=coordinator,
            authority=plan.authority,
            topology=topology,
            lanes=(
                conversation.AgentConversationLane(
                    lane_id=lane.lane_id,
                    mode=conversation.ConversationMode.STATELESS,
                    reasoning_context=plan.reasoning_context,
                    compaction=(
                        conversation.InlineCompaction(
                            compact_threshold=plan.compact_threshold
                        )
                        if plan.compact_threshold is not None
                        else conversation.DisabledCompaction()
                    ),
                ),
            ),
            logical_turn_id=conversation.LogicalTurnId(f"turn-{suffix}"),
            execution_segment_id=conversation.ExecutionSegmentId(
                f"segment-{suffix}"
            ),
            checkpoint_id=conversation.CheckpointId(f"checkpoint-{suffix}"),
            branch_id=(
                plan.parent.identity.branch_id
                if plan.parent is not None
                else conversation.ConversationBranchId(f"branch-{suffix}")
            ),
            provisional_response_id=plan.provisional_response_id,
            public_response_id=plan.public_response_id,
            idempotency_key=plan.idempotency_key,
            retention=plan.retention,
            parent=plan.parent,
        )
        return PreparedServedResponsesTurn(turn=turn)

    return resolve


def _configuration(
    store: conversation.InMemoryConversationStore,
    clock: _MutableClock,
    *,
    ttl: int = 60,
    provider_controller: (
        conversation.DeterministicFaultController | None
    ) = None,
    sweep_limit: int = 100,
    hardening_hook: (
        security.ConversationHardeningCoordinatorHook | None
    ) = None,
) -> ServedResponsesConfiguration:
    """Return one strict service configuration for a durable test facade."""

    async def resolve_authority(
        request: Request,
    ) -> conversation.AuthorityScope | None:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            return None
        return _authority(header.removeprefix("Bearer "))

    return ServedResponsesConfiguration(
        store=_DurableMemoryFacade(store),
        authority_resolver=resolve_authority,
        turn_resolver=_turn_resolver(store, clock, provider_controller),
        policy=ServedResponsesPolicy(
            agent_id=_AGENT_ID,
            endpoint_id=_ENDPOINT_ID,
            retention=_retention(ttl),
            public_model="served-model",
            allowed_reasoning_contexts=frozenset(
                conversation.ReasoningContext
            ),
            allowed_includes=frozenset(
                {
                    "reasoning.encrypted_content",
                    "file_search_call.results",
                }
            ),
            allowed_tool_names=frozenset({"lookup"}),
            min_compact_threshold=10,
            max_compact_threshold=1000,
            sweep_limit=sweep_limit,
            hardening_required=hardening_hook is not None,
        ),
        clock=clock,
        hardening_hook=hardening_hook,
    )


def _hardening_hook(
    store: conversation.InMemoryConversationStore,
    clock: _MutableClock,
) -> security.ConversationHardeningCoordinatorHook:
    """Return production hardening bound to the served test authority."""
    authority = _authority()
    authority_scope_digest = conversation.AuthorityDigest(
        conversation.authority_digest(authority)
    )
    key_ring = security.AsyncConversationKeyRing(
        {
            authority_scope_digest: (
                security.ConversationOperationalKey(
                    key_id="served-hardening-key",
                    revision=1,
                    status=(security.ConversationOperationalKeyStatus.ACTIVE),
                    purposes=frozenset(security.ConversationKeyPurpose),
                    key_bytes=b"h" * 32,
                    activated_at=_NOW,
                ),
            )
        },
        clock=clock,
    )
    worker = security.ConversationMaintenanceWorker(
        (
            security.ConversationRetentionMaintenanceOperation(
                store=store,
                clock=clock,
            ),
        ),
        batch_size=10,
        interval_seconds=60,
        shutdown_timeout_seconds=0.1,
    )

    async def backend() -> security.ConversationBackendHealth:
        return security.ConversationBackendHealth(
            migration_ready=True,
            schema_version=1,
            application_version=1,
            outbox_lag=0,
            maximum_outbox_lag=0,
        )

    async def capability() -> security.ConversationCapabilityHealth:
        return security.ConversationCapabilityHealth(
            resolver_available=True,
            active_profiles=0,
            resolvable_profiles=0,
        )

    activation_digest = conversation.IntegrityDigest("a" * 64)
    readiness = security.ConversationReadinessChecker(
        backend_probe=backend,
        key_ring=key_ring,
        authority=authority_scope_digest,
        workers=(worker,),
        capability_probe=capability,
        activation=security.ConversationActivationHealth(
            expected_digest=activation_digest,
            loaded_digest=activation_digest,
        ),
    )
    policy = security.resolve_conversation_policy(
        security.ConversationHardeningPolicy(
            default_mode=conversation.ConversationMode.STATELESS,
            allowed_modes=frozenset({conversation.ConversationMode.STATELESS}),
            allowed_reasoning_contexts=frozenset(
                conversation.ReasoningContext
            ),
            compaction=security.ConversationCompactionPolicy(
                allowed_operations=frozenset(conversation.CompactionOperation)
            ),
            backend=security.ConversationCheckpointBackend.POSTGRESQL,
            retention=_retention(),
            resources=security.ConversationResourcePolicy(),
            checkpoint_keys=security.ConversationKeyRotationPolicy(),
            envelope_keys=security.ConversationKeyRotationPolicy(),
            capability_profiles=(),
            telemetry=security.ConversationTelemetryPolicy(),
        )
    )
    return security.ConversationHardeningCoordinatorHook(
        policy=policy,
        admission=security.FairConversationAdmissionController(
            policy.resources
        ),
        admission_key=security.ConversationAdmissionKey(
            authority_digest=authority_scope_digest,
            conversation_digest=conversation.IntegrityDigest("b" * 64),
        ),
        readiness=readiness,
        telemetry=security.BoundedConversationTelemetry(max_events=32),
    )


def _response(text: str) -> TextGenerationResponse:
    """Return one canonical response consumed by both router transports."""
    return TextGenerationResponse(
        lambda: text,
        logger=getLogger(__name__),
        use_async_generator=False,
        generation_settings=GenerationSettings(),
    )


def _stored_orchestrate(
    dispatches: list[str],
) -> Callable[..., Any]:
    """Return an async route adapter that executes the coordinated turn."""

    async def dispatch(
        request: ResponsesRequest,
        logger: Logger,
        orchestrator: Orchestrator,
        interaction_runtime: InteractionRuntime | None = None,
        *,
        conversation_turn: conversation.AgentConversationTurn | None = None,
        conversation_children: tuple[AgentConversationChildBinding, ...] = (),
        outward_response_id: str | None = None,
    ) -> tuple[OrchestratorResponse, str, int]:
        del logger, orchestrator, interaction_runtime, conversation_children
        dispatches.append(str(outward_response_id))
        if conversation_turn is None:
            text = "one-shot"
        else:
            result = await conversation_turn.execute(
                "\n".join(str(message.content) for message in request.messages)
            )
            text = result.output
        assert outward_response_id is not None or not request.store
        response_id = outward_response_id or "resp_avl_" + "0" * 32
        return _response(text), response_id, int(_NOW.timestamp())

    return dispatch


def _client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    configuration: ServedResponsesConfiguration | None,
    dispatches: list[str] | None = None,
) -> TestClient:
    """Return the real FastAPI validation and route boundary."""
    observed = dispatches if dispatches is not None else []
    monkeypatch.setattr(
        responses_router,
        "orchestrate",
        _stored_orchestrate(observed),
    )
    app = FastAPI()
    app.include_router(responses_router.router)
    app.dependency_overrides[di_get_logger] = lambda: getLogger(__name__)
    orchestrator = object.__new__(_ResponsesOrchestrator)
    app.dependency_overrides[di_get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[
        responses_router._server_output_redaction_settings
    ] = lambda: ServerOutputRedactionSettings()
    configure_served_responses(app, configuration)
    return TestClient(app)


_INPUT_ITEMS: tuple[dict[str, object], ...] = (
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
        "status": "completed",
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "answer", "annotations": []}
        ],
        "id": "message-1",
        "status": "completed",
        "phase": "final_answer",
    },
    {
        "type": "file_search_call",
        "id": "fs-1",
        "queries": ["hello"],
        "status": "completed",
    },
    {
        "type": "computer_call",
        "id": "computer-1",
        "call_id": "call-1",
        "pending_safety_checks": [],
        "status": "completed",
    },
    {"type": "computer_call_output", "call_id": "call-1", "output": "ok"},
    {
        "type": "web_search_call",
        "id": "web-1",
        "action": {"type": "search", "query": "hello"},
        "status": "completed",
    },
    {
        "type": "function_call",
        "arguments": "{}",
        "call_id": "call-2",
        "name": "lookup",
    },
    {"type": "function_call_output", "call_id": "call-2", "output": "ok"},
    {
        "type": "tool_search_call",
        "arguments": {"query": "lookup"},
        "call_id": "call-3",
    },
    {"type": "tool_search_output", "call_id": "call-3", "tools": []},
    {"type": "additional_tools", "role": "developer", "tools": []},
    {"type": "reasoning", "id": "reasoning-1", "summary": []},
    {"type": "compaction", "encrypted_content": "opaque"},
    {
        "type": "image_generation_call",
        "id": "image-1",
        "result": "image-data",
        "status": "completed",
    },
    {
        "type": "code_interpreter_call",
        "code": "print(1)",
        "container_id": "container-1",
        "id": "code-1",
        "outputs": [],
        "status": "completed",
    },
    {
        "type": "local_shell_call",
        "action": {"type": "exec", "command": ["pwd"], "env": {}},
        "call_id": "call-4",
        "id": "shell-1",
        "status": "completed",
    },
    {
        "type": "local_shell_call_output",
        "id": "shell-output-1",
        "output": "ok",
    },
    {
        "type": "shell_call",
        "action": {"commands": ["pwd"]},
        "call_id": "call-5",
    },
    {
        "type": "shell_call_output",
        "call_id": "call-5",
        "output": [
            {
                "outcome": {"type": "exit", "exit_code": 0},
                "stderr": "",
                "stdout": "ok",
            }
        ],
    },
    {
        "type": "apply_patch_call",
        "call_id": "call-6",
        "operation": {"type": "create_file", "path": "safe.txt"},
        "status": "completed",
    },
    {
        "type": "apply_patch_call_output",
        "call_id": "call-6",
        "status": "completed",
    },
    {
        "type": "mcp_list_tools",
        "id": "mcp-list-1",
        "server_label": "safe",
        "tools": [],
    },
    {
        "type": "mcp_approval_request",
        "arguments": "{}",
        "id": "mcp-approval-1",
        "name": "lookup",
        "server_label": "safe",
    },
    {
        "type": "mcp_approval_response",
        "approval_request_id": "mcp-approval-1",
        "approve": True,
    },
    {
        "type": "mcp_call",
        "arguments": "{}",
        "id": "mcp-call-1",
        "name": "lookup",
        "server_label": "safe",
    },
    {
        "type": "custom_tool_call",
        "call_id": "call-7",
        "input": "hello",
        "name": "lookup",
    },
    {"type": "custom_tool_call_output", "call_id": "call-7", "output": "ok"},
    {"type": "compaction_trigger"},
    {"id": "item-1"},
)


@pytest.mark.parametrize(
    "item",
    _INPUT_ITEMS,
    ids=tuple(
        str(item.get("type", "item_reference")) for item in _INPUT_ITEMS
    ),
)
def test_supported_input_items_cross_fastapi_boundary(
    item: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept every frozen tagged input item through the actual route model."""
    client = _client(monkeypatch, configuration=None)

    response = client.post(
        "/responses",
        json={"input": [item], "model": "served-model"},
    )

    assert response.status_code == 200, response.text
    request = ResponsesRequest.model_validate({"input": [item]})
    content = request.messages[0].content
    assert isinstance(content, str)
    assert loads(content) == item


def test_strict_schema_defaults_and_extension_boundary() -> None:
    """Default to transient mode while keeping only safe future namespaces."""
    request = ResponsesRequest.model_validate(
        {
            "input": "hello",
            "extensions": {
                "task_input": {"version": "1", "handling": "attached"},
                "future_safe": {"revision": 2},
                "avalan": {"version": "1"},
            },
        }
    )

    assert request.store is False
    assert request.extensions is not None
    assert request.extensions.model_extra == {"future_safe": {"revision": 2}}
    stored_request = ResponsesRequest.model_validate(
        {
            "input": "hello",
            "store": True,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "idempotency_key": "request-1",
                    },
                }
            },
        }
    )
    assert (
        responses_router._request_idempotency_key(stored_request)
        == "request-1"
    )
    phased = ResponsesRequest.model_validate(
        {
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "working",
                    "phase": "commentary",
                }
            ]
        }
    )
    phased_content = phased.messages[0].content
    assert isinstance(phased_content, str)
    assert loads(phased_content)["phase"] == "commentary"
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(
            {
                "input": "hello",
                "reasoning": {"context": "auto", "extra": 1},
            }
        )
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(
            {
                "input": "hello",
                "extensions": {
                    "avalan": {"version": "1", "authority": "owner"}
                },
            }
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"input": ["primitive-input-item"]},
        {"input": "hello", "max_tokens": 1, "max_output_tokens": 1},
        {"input": "hello", "previous_response_id": "resp_avl_parent"},
        {
            "input": "hello",
            "store": True,
            "previous_response_id": " ",
        },
        {
            "input": "hello",
            "stream_options": {"include_obfuscation": False},
        },
        {"input": "hello", "context_management": []},
        {
            "input": "hello",
            "context_management": [
                {"type": "compaction", "compact_threshold": 10},
                {"type": "compaction", "compact_threshold": 20},
            ],
        },
        {
            "input": "hello",
            "include": [
                "reasoning.encrypted_content",
                "reasoning.encrypted_content",
            ],
        },
        {"input": "hello", "tools": []},
        {
            "input": "hello",
            "tools": [
                {"type": "function", "name": "lookup", "parameters": {}},
                {"type": "function", "name": "lookup", "parameters": {}},
            ],
        },
        {"input": "hello", "background": True},
        {
            "input": "hello",
            "metadata": {f"key-{index}": index for index in range(17)},
        },
    ),
)
def test_strict_schema_rejects_cross_field_conflicts(
    payload: dict[str, object],
) -> None:
    """Reject ambiguous request combinations before dispatch."""
    with pytest.raises(ValidationError):
        ResponsesRequest.model_validate(payload)


def test_served_text_projection_handles_all_content_sources() -> None:
    """Project replay content safely and reject empty served input."""
    request = ResponsesRequest.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "declined"},
                        {"type": "input_image", "file_id": "file-image"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.test/image.png",
                        },
                    ],
                }
            ]
        }
    )

    assert (
        responses_router._served_input_text(request)
        == "declined\n[uploaded image input]\n[non-text input]"
    )
    extension_only = ResponsesRequest.model_validate(
        {
            "input": "hello",
            "extensions": {"avalan": {"version": "1"}},
        }
    )
    assert responses_router._request_idempotency_key(extension_only) is None
    with pytest.raises(conversation.ConversationValidationError):
        responses_router._served_input_text(
            ResponsesRequest.model_validate({"input": " "})
        )


def test_orchestration_forwards_prepared_conversation_context() -> None:
    """Forward prepared turns only through the explicit orchestrator lane."""
    observed: dict[str, object] = {}
    marker = cast(OrchestratorResponse, object())

    async def recording_orchestrator(
        input_value: object,
        **kwargs: Any,
    ) -> OrchestratorResponse:
        del input_value
        observed.update(kwargs)
        return marker

    turn = cast(conversation.AgentConversationTurn, object())
    response, response_id, _ = run(
        server_routers.orchestrate(
            ResponsesRequest.model_validate(
                {
                    "input": "hello",
                    "max_tokens": 17,
                }
            ),
            getLogger(__name__),
            cast(Orchestrator, recording_orchestrator),
            conversation_turn=turn,
            conversation_children=(),
            outward_response_id="provisional-avl-test",
        )
    )

    assert response is marker
    assert response_id == "provisional-avl-test"
    assert observed["conversation_turn"] is turn
    assert observed["conversation_children"] == ()
    settings = observed["settings"]
    assert isinstance(settings, GenerationSettings)
    assert settings.max_new_tokens == 17


@pytest.mark.parametrize(
    ("error", "status_code"),
    (
        (conversation.ConversationValidationError(), 400),
        (conversation.ConversationLimitError(), 400),
        (conversation.ConversationConflictError(), 409),
        (
            conversation.ConversationError(
                conversation.ConversationErrorCode.BINDING_DRIFT
            ),
            500,
        ),
    ),
)
def test_served_error_projection_is_stable(
    error: conversation.ConversationError,
    status_code: int,
) -> None:
    """Map each remaining conversation error class to a safe status."""
    response = responses_router._conversation_error_response(error)

    assert response.status_code == status_code


@pytest.mark.parametrize(
    ("error_type", "status_code", "code"),
    (
        (
            conversation.ConversationAuthorizationError,
            404,
            "response_not_found",
        ),
        (
            conversation.ConversationStorageError,
            500,
            "conversation_storage_failed",
        ),
        (
            conversation.ConversationKeyMissingError,
            500,
            "conversation_key_missing",
        ),
        (
            conversation.ConversationIntegrityError,
            500,
            "conversation_integrity_failed",
        ),
    ),
)
def test_lifecycle_routes_preserve_safe_storage_error_categories(
    error_type: type[conversation.ConversationError],
    status_code: int,
    code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conceal only absence and preserve frozen persistence categories."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    configuration = _configuration(store, clock)
    client = _client(monkeypatch, configuration=configuration)
    identifier = "resp_avl_" + "8" * 32
    headers = {"Authorization": "Bearer owner"}

    async def fail_retrieve(*args: object) -> conversation.ConversationResult:
        del args
        raise error_type()

    async def fail_deletion(
        *args: object,
    ) -> conversation.LocalDeletionPreparation:
        del args
        raise error_type()

    monkeypatch.setattr(configuration.store, "retrieve", fail_retrieve)
    retrieved = client.get(f"/responses/{identifier}", headers=headers)
    monkeypatch.setattr(
        configuration.store,
        "prepare_deletion",
        fail_deletion,
    )
    deleted = client.delete(f"/responses/{identifier}", headers=headers)

    for response in (retrieved, deleted):
        assert response.status_code == status_code
        assert response.json()["error"]["code"] == code
        assert identifier not in response.text


def test_retrieve_commit_failure_preserves_existing_mapping(
    record_property: Callable[[str, object], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a stored response addressable after retrieval storage failure."""
    record_property("conversation_acceptance_evidence", "negative")
    store = conversation.InMemoryConversationStore()
    configuration = _configuration(store, _MutableClock())
    client = _client(monkeypatch, configuration=configuration)
    headers = {"Authorization": "Bearer owner"}
    created = client.post(
        "/responses",
        headers=headers,
        json={"input": "hello", "store": True},
    )
    assert created.status_code == 200
    identifier = created.json()["id"]
    original_retrieve = store.retrieve

    async def fail_retrieve(*args: object) -> conversation.ConversationResult:
        del args
        raise conversation.ConversationCommitError()

    monkeypatch.setattr(store, "retrieve", fail_retrieve)
    response = client.get(f"/responses/{identifier}", headers=headers)

    assert response.status_code == 500
    assert (
        response.json()["error"]["code"] == "conversation_state_commit_failed"
    )
    retained = run(
        original_retrieve(
            conversation.PublicResponseId(identifier),
            _authority(),
        )
    )
    assert retained.public_response_id == identifier


def test_delete_commit_failure_preserves_existing_mapping(
    record_property: Callable[[str, object], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a stored response addressable after deletion storage failure."""
    record_property("conversation_acceptance_evidence", "negative")
    store = conversation.InMemoryConversationStore()
    configuration = _configuration(store, _MutableClock())
    client = _client(monkeypatch, configuration=configuration)
    headers = {"Authorization": "Bearer owner"}
    created = client.post(
        "/responses",
        headers=headers,
        json={"input": "hello", "store": True},
    )
    assert created.status_code == 200
    identifier = created.json()["id"]
    original_retrieve = store.retrieve

    async def fail_deletion(
        *args: object,
    ) -> conversation.LocalDeletionPreparation:
        del args
        raise conversation.ConversationCommitError()

    monkeypatch.setattr(store, "prepare_deletion", fail_deletion)
    response = client.delete(f"/responses/{identifier}", headers=headers)

    assert response.status_code == 500
    assert (
        response.json()["error"]["code"] == "conversation_state_commit_failed"
    )
    retained = run(
        original_retrieve(
            conversation.PublicResponseId(identifier),
            _authority(),
        )
    )
    assert retained.public_response_id == identifier


def test_served_routes_require_service_and_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed on lifecycle routes before touching durable state."""
    identifier = "resp_avl_" + "f" * 32
    unavailable = _client(monkeypatch, configuration=None)

    assert unavailable.get(f"/responses/{identifier}").status_code == 404
    assert unavailable.delete(f"/responses/{identifier}").status_code == 404

    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    configuration = _configuration(store, clock)
    protected = _client(monkeypatch, configuration=configuration)
    assert protected.get(f"/responses/{identifier}").status_code == 401
    assert protected.delete(f"/responses/{identifier}").status_code == 401
    assert (
        protected.post(
            "/responses",
            json={"input": "hello", "store": True},
        ).status_code
        == 401
    )

    direct = run(
        responses_router.create_response(
            ResponsesRequest.model_validate({"input": "hello", "store": True}),
            logger=getLogger(__name__),
            orchestrator=object.__new__(_ResponsesOrchestrator),
            output_redaction_settings=ServerOutputRedactionSettings(),
            served_responses_service=ServedResponsesService(configuration),
        )
    )
    assert isinstance(direct, JSONResponse)
    assert direct.status_code == 401


def test_create_maps_conversation_dispatch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return one safe error when stateless dispatch rejects the request."""
    client = _client(monkeypatch, configuration=None)

    async def reject_dispatch(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise conversation.ConversationValidationError()

    monkeypatch.setattr(responses_router, "orchestrate", reject_dispatch)
    response = client.post("/responses", json={"input": "hello"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "conversation_validation_failed"


def test_stored_create_continue_retrieve_delete_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish only committed IDs and enforce the full local lifecycle."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    dispatches: list[str] = []
    client = _client(
        monkeypatch,
        configuration=_configuration(store, clock, ttl=60),
        dispatches=dispatches,
    )
    headers = {"Authorization": "Bearer owner"}
    first = client.post(
        "/responses",
        headers=headers,
        json={
            "input": "alpha",
            "model": "served-model",
            "store": True,
            "max_output_tokens": 32,
            "reasoning": {"context": "auto"},
            "context_management": [
                {"type": "compaction", "compact_threshold": 100}
            ],
            "stream_options": None,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "idempotency_key": "first-request",
                    },
                }
            },
        },
    )
    assert first.status_code == 200, first.text
    first_body = first.json()
    first_id = first_body["id"]
    assert first_id.startswith("resp_avl_")
    assert dispatches[0].startswith("provisional-avl-")
    assert dispatches[0] != first_id
    assert first_body["object"] == "response"
    assert first_body["parallel_tool_calls"] is False
    assert first_body["output"][0]["content"][0]["text"] == "turn-1:alpha"
    assert "private" not in str(first_body).lower()

    retrieved = client.get(f"/responses/{first_id}", headers=headers)
    assert retrieved.status_code == 200
    assert retrieved.json() == first_body

    second = client.post(
        "/responses",
        headers=headers,
        json={
            "input": "beta",
            "model": "served-model",
            "store": True,
            "previous_response_id": first_id,
        },
    )
    assert second.status_code == 200, second.text
    second_body = second.json()
    second_id = second_body["id"]
    assert second_id != first_id
    assert second_body["output"][0]["content"][0]["text"] == "turn-2:beta"

    concealed = {"Authorization": "Bearer another-principal"}
    concealed_get = client.get(
        f"/responses/{first_id}",
        headers=concealed,
    )
    assert concealed_get.status_code == 404
    assert (
        client.post(
            "/responses",
            headers=concealed,
            json={
                "input": "forbidden",
                "store": True,
                "previous_response_id": first_id,
            },
        ).status_code
        == 404
    )
    deleted = client.delete(f"/responses/{first_id}", headers=headers)
    assert deleted.status_code == 200
    repeated = client.delete(f"/responses/{first_id}", headers=headers)
    assert repeated.status_code == 200
    metadata = repeated.json()["metadata"]
    assert metadata["avalan_local_deletion"] == "tombstoned"
    deleted_get = client.get(f"/responses/{first_id}", headers=headers)
    assert deleted_get.status_code == 404

    clock.value += timedelta(seconds=61)
    expired_get = client.get(f"/responses/{second_id}", headers=headers)
    assert expired_get.status_code == 404


def test_explicit_idempotency_replays_without_provider_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay lost responses and conflict changed requests under one key."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    provider_controller = conversation.DeterministicFaultController()
    client = _client(
        monkeypatch,
        configuration=_configuration(
            store,
            clock,
            provider_controller=provider_controller,
        ),
    )
    headers = {"Authorization": "Bearer owner"}
    payload: dict[str, object] = {
        "input": "retry-safe",
        "store": True,
        "extensions": {
            "avalan": {
                "version": "1",
                "conversation": {
                    "version": "1",
                    "idempotency_key": "retry-safe-request",
                },
            }
        },
    }

    first = client.post("/responses", headers=headers, json=payload)
    assert first.status_code == 200, first.text

    async def retry_concurrently() -> tuple[Any, Any]:
        retries = await gather(
            to_thread(
                client.post, "/responses", headers=headers, json=payload
            ),
            to_thread(
                client.post, "/responses", headers=headers, json=payload
            ),
        )
        return retries[0], retries[1]

    retry_one, retry_two = run(retry_concurrently())
    retry_responses = (retry_one, retry_two)
    for retry in retry_responses:
        assert retry.status_code == 200, retry.text
        assert retry.json() == first.json()
    assert provider_controller.visited.count("provider:dispatch") == 1

    changed = client.post(
        "/responses",
        headers=headers,
        json={**payload, "input": "changed"},
    )
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "conversation_conflict"
    assert provider_controller.visited.count("provider:dispatch") == 1


@pytest.mark.anyio
async def test_phase11_served_dispatch_installs_required_hardening(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Dispatch store=true only through the trusted production hook."""
    record_property("conversation_acceptance_evidence", "public")
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    hook = _hardening_hook(store, clock)
    configuration = _configuration(
        store,
        clock,
        hardening_hook=hook,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            configuration.policy,
            hardening_required=cast(bool, 1),
        )
    bound_plan = ServedResponsesTurnPlan(
        authority=_authority(),
        input_text="hardening-binding",
        public_response_id=conversation.PublicResponseId(
            "resp_avl_" + "e" * 32
        ),
        provisional_response_id=conversation.ProvisionalResponseId(
            "provisional-avl-" + "e" * 32
        ),
        idempotency_key=conversation.RequestIdempotencyKey(
            "hardening-binding"
        ),
        request_fingerprint="e" * 64,
        retention=_retention(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        compact_threshold=None,
        includes=(),
        tool_names=(),
        parent=None,
        streaming=False,
        hardening_hook=hook,
        hardening_required=True,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(bound_plan, hardening_required=False)

    forged_dispatches: list[str] = []

    class _ForgedCoordinator:
        async def execute(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            forged_dispatches.append("dispatch")
            raise AssertionError("forged coordinator dispatched")

    forged_coordinator = _ForgedCoordinator()
    forged_store = conversation.InMemoryConversationStore()

    async def forged_turn_resolver(
        plan: ServedResponsesTurnPlan,
    ) -> PreparedServedResponsesTurn:
        prepared = await _turn_resolver(forged_store, clock)(plan)
        coordinator = cast(
            conversation.RunScopedConversationCoordinator,
            prepared.turn.coordinator,
        )
        await coordinator.close()
        return replace(
            prepared,
            turn=replace(
                prepared.turn,
                coordinator=cast(
                    conversation.ConversationCoordinator,
                    forged_coordinator,
                ),
            ),
        )

    forged_service = ServedResponsesService(
        replace(configuration, turn_resolver=forged_turn_resolver)
    )
    with pytest.raises(conversation.ConversationValidationError):
        await forged_service.prepare_turn(
            authority=_authority(),
            input_text="forged",
            parent=None,
            reasoning_context=conversation.ReasoningContext.AUTO,
            compact_threshold=None,
            includes=(),
            tool_names=(),
            streaming=False,
            idempotency_key="forged-hardening",
            request_fingerprint="d" * 64,
        )
    assert forged_dispatches == []

    dispatches: list[str] = []
    monkeypatch.setattr(
        responses_router,
        "orchestrate",
        _stored_orchestrate(dispatches),
    )
    app = FastAPI()
    app.include_router(responses_router.router)
    app.dependency_overrides[di_get_logger] = lambda: getLogger(__name__)
    app.dependency_overrides[di_get_orchestrator] = lambda: object.__new__(
        _ResponsesOrchestrator
    )
    app.dependency_overrides[
        responses_router._server_output_redaction_settings
    ] = lambda: ServerOutputRedactionSettings()
    configure_served_responses(app, configuration)
    service = app.state.served_responses_service
    assert isinstance(service, ServedResponsesService)
    await start_served_responses(app)
    await start_served_responses(app)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://served.test",
        ) as client:
            response = await client.post(
                "/responses",
                headers={"Authorization": "Bearer owner"},
                json={
                    "input": "hardened",
                    "model": "served-model",
                    "store": True,
                },
            )
        assert response.status_code == 200, response.text
        assert (
            response.json()["output"][0]["content"][0]["text"]
            == "turn-1:hardened"
        )
        assert len(dispatches) == 1
    finally:
        await service.aclose()

    with pytest.raises(TypeError, match="hardening_hook"):
        replace(
            configuration,
            policy=replace(
                configuration.policy,
                hardening_required=True,
            ),
            hardening_hook=None,
        )


def test_target_ttl_is_independent_of_bounded_sweep_backlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an expired target even while older expiry work remains."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    client = _client(
        monkeypatch,
        configuration=_configuration(
            store,
            clock,
            ttl=60,
            sweep_limit=1,
        ),
    )
    headers = {"Authorization": "Bearer owner"}
    for index in range(4):
        blocker = client.post(
            "/responses",
            headers=headers,
            json={"input": f"blocker-{index}", "store": True},
        )
        assert blocker.status_code == 200, blocker.text
    target = client.post(
        "/responses",
        headers=headers,
        json={"input": "target", "store": True},
    )
    assert target.status_code == 200, target.text
    target_id = target.json()["id"]
    clock.value += timedelta(seconds=61)

    retrieved = client.get(f"/responses/{target_id}", headers=headers)
    continued = client.post(
        "/responses",
        headers=headers,
        json={
            "input": "expired-child",
            "store": True,
            "previous_response_id": target_id,
        },
    )
    deleted = client.delete(f"/responses/{target_id}", headers=headers)

    assert retrieved.status_code == 404
    assert continued.status_code == 404
    assert deleted.status_code == 404
    raw_target = run(
        store.retrieve(
            conversation.PublicResponseId(target_id),
            _authority(),
        )
    )
    assert str(raw_target.public_response_id) == target_id


def test_store_true_requires_durable_service_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject durable intent before dispatch when storage is absent."""
    dispatches: list[str] = []
    client = _client(
        monkeypatch,
        configuration=None,
        dispatches=dispatches,
    )

    response = client.post(
        "/responses",
        json={"input": "hello", "store": True},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "avalan_durable_store_required"
    assert dispatches == []


def test_stored_stream_withholds_committed_id_until_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project a provisional stream ID and publish the committed ID once."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    client = _client(monkeypatch, configuration=_configuration(store, clock))
    headers = {"Authorization": "Bearer owner"}

    response = client.post(
        "/responses",
        headers=headers,
        json={"input": "streamed", "store": True, "stream": True},
    )

    assert response.status_code == 200, response.text
    events = [
        loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    event_types = [event["type"] for event in events]
    assert event_types[0] == "response.created"
    assert event_types[-1] == "response.completed"
    assert event_types.count("response.completed") == 1
    provisional_id = events[0]["response"]["id"]
    public_id = events[-1]["response"]["id"]
    assert provisional_id.startswith("provisional-avl-")
    assert public_id.startswith("resp_avl_")
    assert public_id not in str(events[:-1])
    provisional = client.get(
        f"/responses/{provisional_id}",
        headers=headers,
    )
    committed = client.get(f"/responses/{public_id}", headers=headers)
    assert provisional.status_code == 404
    assert committed.status_code == 200


@pytest.mark.parametrize("stream", (False, True), ids=("body", "stream"))
def test_commit_failure_never_publishes_success(
    stream: bool,
    record_property: Callable[[str, object], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Withhold success after provider output without a durable commit."""
    record_property("conversation_acceptance_evidence", "negative")
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    client = _client(monkeypatch, configuration=_configuration(store, clock))
    dispatches = 0

    async def complete_without_commit(
        request: ResponsesRequest,
        logger: Logger,
        orchestrator: Orchestrator,
        interaction_runtime: InteractionRuntime | None = None,
        *,
        conversation_turn: conversation.AgentConversationTurn | None = None,
        conversation_children: tuple[AgentConversationChildBinding, ...] = (),
        outward_response_id: str | None = None,
    ) -> tuple[OrchestratorResponse, str, int]:
        nonlocal dispatches
        del (
            request,
            logger,
            orchestrator,
            interaction_runtime,
            conversation_children,
        )
        dispatches += 1
        assert conversation_turn is not None
        assert outward_response_id == str(
            conversation_turn.provisional_response_id
        )
        return (
            _response("visible-before-commit-failure"),
            outward_response_id,
            int(_NOW.timestamp()),
        )

    monkeypatch.setattr(
        responses_router,
        "orchestrate",
        complete_without_commit,
    )
    response = client.post(
        "/responses",
        headers={"Authorization": "Bearer owner"},
        json={"input": "hello", "store": True, "stream": stream},
    )

    assert dispatches == 1
    assert store.diagnostics.public_responses == 0
    if not stream:
        assert response.status_code == 500
        assert (
            response.json()["error"]["code"]
            == "conversation_state_commit_failed"
        )
        return
    events = [
        loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    event_types = [event["type"] for event in events]
    assert event_types[-1] == "response.failed"
    assert "response.completed" not in event_types
    assert "resp_avl_" not in str(events)


@pytest.mark.parametrize(
    ("payload", "status"),
    (
        ({"input": "hello", "unknown": True}, 400),
        ({"input": "hello", "reasoning": {"unknown": True}}, 400),
        (
            {
                "input": "hello",
                "context_management": [
                    {"type": "compaction", "compact_threshold": 10, "x": 1}
                ],
            },
            400,
        ),
        (
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "hello",
                        "unknown": True,
                    }
                ]
            },
            400,
        ),
        (
            {
                "input": "hello",
                "extensions": {
                    "avalan": {"version": "1", "authority": "owner"}
                },
            },
            400,
        ),
    ),
    ids=(
        "top-level",
        "reasoning",
        "context-management",
        "input-item",
        "authority-extension",
    ),
)
def test_unknown_stateful_fields_fail_before_dispatch(
    payload: dict[str, object],
    status: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject every unknown state-bearing field before model execution."""
    dispatches: list[str] = []
    client = _client(
        monkeypatch,
        configuration=None,
        dispatches=dispatches,
    )

    response = client.post("/responses", json=payload)

    assert response.status_code == status
    assert response.json()["error"] == {
        "message": "Request validation failed.",
        "type": "invalid_request_error",
        "code": "conversation_validation_failed",
    }
    assert dispatches == []


def test_request_cannot_broaden_served_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject include, tool, reasoning, and compaction policy broadening."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    dispatches: list[str] = []
    client = _client(
        monkeypatch,
        configuration=_configuration(store, clock),
        dispatches=dispatches,
    )
    headers = {"Authorization": "Bearer owner"}
    payloads = (
        {
            "context_management": [
                {"type": "compaction", "compact_threshold": 1}
            ]
        },
    )

    for extra in payloads:
        response = client.post(
            "/responses",
            headers=headers,
            json={"input": "hello", "store": True, **extra},
        )
        assert response.status_code == 400
        assert (
            response.json()["error"]["code"]
            == "conversation_capability_unsupported"
        )
    assert dispatches == []


@pytest.mark.parametrize(
    "extra",
    (
        {"reasoning": {"context": "auto"}},
        {
            "context_management": [
                {"type": "compaction", "compact_threshold": 100}
            ]
        },
        {"include": ["reasoning.encrypted_content"]},
        {
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "idempotency_key": "stateless-request",
                    },
                }
            }
        },
        {"tools": [{"type": "function", "name": "lookup", "parameters": {}}]},
        {"tool_choice": "auto"},
    ),
    ids=(
        "reasoning-context",
        "context-management",
        "include",
        "idempotency",
        "tools",
        "tool-choice",
    ),
)
def test_unsupported_controls_fail_before_stateless_dispatch(
    extra: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject every unsupported stateless control before model execution."""
    dispatches: list[str] = []
    client = _client(
        monkeypatch,
        configuration=None,
        dispatches=dispatches,
    )

    response = client.post("/responses", json={"input": "hello", **extra})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "conversation_validation_failed"
    assert dispatches == []


def test_public_id_boundaries_conceal_malformed_and_upstream_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conceal malformed, upstream-owned, and absent public identifiers."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    client = _client(monkeypatch, configuration=_configuration(store, clock))
    headers = {"Authorization": "Bearer owner"}

    for identifier in ("resp_upstream_123", "response-123", "resp_avl_bad"):
        retrieved = client.get(f"/responses/{identifier}", headers=headers)
        deleted = client.delete(f"/responses/{identifier}", headers=headers)
        assert retrieved.status_code == 404
        assert deleted.status_code == 404
        continued = client.post(
            "/responses",
            headers=headers,
            json={
                "input": "hello",
                "store": True,
                "previous_response_id": identifier,
            },
        )
        assert continued.status_code == 404


def test_lifecycle_transitions_and_configuration_fail_closed() -> None:
    """Accept only legal publication transitions and durable async wiring."""
    identifier = conversation.PublicResponseId("resp_avl_" + "a" * 32)
    staged = ServedResponseLifecycleRecord(public_response_id=identifier)
    completed = staged.transition(ServedResponseLifecycle.COMPLETED_PROVIDER)
    committed = completed.transition(
        ServedResponseLifecycle.CHECKPOINT_COMMITTED
    )
    published = committed.transition(ServedResponseLifecycle.PUBLISHED)
    assert published.transition(ServedResponseLifecycle.TOMBSTONED).state is (
        ServedResponseLifecycle.TOMBSTONED
    )
    with pytest.raises(conversation.ConversationValidationError):
        staged.transition(ServedResponseLifecycle.PUBLISHED)

    async def authority_resolver(
        request: Request,
    ) -> conversation.AuthorityScope | None:
        del request
        return _authority()

    async def turn_resolver(
        plan: ServedResponsesTurnPlan,
    ) -> PreparedServedResponsesTurn:
        del plan
        raise conversation.ConversationCapabilityError()

    with pytest.raises(TypeError, match="durable"):
        ServedResponsesConfiguration(
            store=conversation.InMemoryConversationStore(),
            authority_resolver=authority_resolver,
            turn_resolver=turn_resolver,
            policy=ServedResponsesPolicy(
                agent_id=_AGENT_ID,
                endpoint_id=_ENDPOINT_ID,
                retention=_retention(),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        ServedResponsesPolicy(
            agent_id=_AGENT_ID,
            endpoint_id=_ENDPOINT_ID,
            retention=replace(
                _retention(),
                storage=conversation.StoragePolicy(
                    local=conversation.LocalResponseStorage.PROCESS_LOCAL,
                    upstream=conversation.ProviderLaneStorage.STATELESS,
                ),
            ),
        )


def test_schema_sources_and_lifecycle_values_fail_closed() -> None:
    """Reject ambiguous inputs and every malformed lifecycle value shape."""
    invalid_inputs = (
        {
            "type": "input_image",
            "image_url": "https://example.test/image.png",
            "file_id": "file-1",
        },
        {"type": "input_image", "image_url": ""},
        {
            "type": "input_file",
            "file_data": "ZmlsZQ==",
            "file_url": "https://example.test/file.txt",
        },
        {"type": "input_file"},
        {
            "type": "message",
            "id": "caller-owned-input-id",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "role": "user",
            "content": "hello",
            "phase": "commentary",
        },
        {
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "declined"}],
        },
    )
    for item in invalid_inputs:
        with pytest.raises(ValidationError):
            ResponsesRequest.model_validate({"input": [item]})
    with pytest.raises(ValidationError):
        ResponsesInputImage(
            type="input_image",
            image_url="https://example.test/image.png",
            file_id="file-image",
        )
    with pytest.raises(ValidationError):
        ResponsesInputFile(type="input_file")
    ResponsesRequest.model_validate(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "file_id": "file-image",
                        },
                        {
                            "type": "input_file",
                            "file_url": "https://example.test/file.txt",
                        },
                    ],
                }
            ]
        }
    )

    identifier = conversation.PublicResponseId("resp_avl_" + "b" * 32)
    with pytest.raises(conversation.ConversationValidationError):
        ServedResponseLifecycleRecord(
            public_response_id=identifier,
            state=cast(ServedResponseLifecycle, "staged"),
        )
    valid = ServedResponsesTurnPlan(
        authority=_authority(),
        input_text="hello",
        public_response_id=identifier,
        provisional_response_id=conversation.ProvisionalResponseId(
            "provisional-avl-" + "b" * 32
        ),
        idempotency_key=conversation.RequestIdempotencyKey("request-b"),
        request_fingerprint=_REQUEST_FINGERPRINT,
        retention=_retention(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        compact_threshold=None,
        includes=(),
        tool_names=(),
        parent=None,
        streaming=False,
    )
    invalid_plans = (
        lambda: replace(
            valid,
            authority=cast(conversation.AuthorityScope, object()),
        ),
        lambda: replace(
            valid,
            retention=cast(conversation.RetentionLimits, object()),
        ),
        lambda: replace(
            valid,
            retention=replace(
                _retention(),
                storage=conversation.StoragePolicy(
                    local=conversation.LocalResponseStorage.PROCESS_LOCAL,
                    upstream=conversation.ProviderLaneStorage.STATELESS,
                ),
            ),
        ),
        lambda: replace(
            valid,
            reasoning_context=cast(conversation.ReasoningContext, "bad"),
        ),
        lambda: replace(valid, compact_threshold=0),
        lambda: replace(valid, includes=cast(tuple[str, ...], ["bad"])),
        lambda: replace(valid, tool_names=("",)),
        lambda: replace(valid, request_fingerprint="not-a-digest"),
        lambda: replace(
            valid,
            parent=cast(conversation.ConversationCheckpoint, object()),
        ),
        lambda: replace(valid, streaming=cast(bool, 1)),
    )
    for invalid_plan in invalid_plans:
        with pytest.raises(conversation.ConversationValidationError):
            invalid_plan()

    with pytest.raises(conversation.ConversationValidationError):
        PreparedServedResponsesTurn(
            turn=cast(conversation.AgentConversationTurn, object())
        )
    clock = _MutableClock()
    store = conversation.InMemoryConversationStore()
    prepared = run(_turn_resolver(store, clock)(valid))
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            prepared,
            children=cast(
                tuple[AgentConversationChildBinding, ...],
                (object(),),
            ),
        )

    invalid_policies = (
        lambda: replace(
            _configuration(store, clock).policy,
            retention=cast(conversation.RetentionLimits, object()),
        ),
        lambda: replace(
            _configuration(store, clock).policy,
            allowed_reasoning_contexts=cast(
                frozenset[conversation.ReasoningContext],
                frozenset(),
            ),
        ),
        lambda: replace(
            _configuration(store, clock).policy,
            min_compact_threshold=100,
            max_compact_threshold=10,
        ),
        lambda: replace(
            _configuration(store, clock).policy,
            sweep_limit=0,
        ),
    )
    for invalid_policy in invalid_policies:
        with pytest.raises(conversation.ConversationValidationError):
            invalid_policy()
    with pytest.raises(conversation.ConversationCapabilityError):
        _configuration(store, clock).policy.validate_capabilities(
            reasoning_context=conversation.ReasoningContext.AUTO,
            includes=("unsupported",),
            tool_names=(),
            compact_threshold=None,
        )


@pytest.mark.anyio
async def test_lifecycle_service_failure_and_shutdown_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise safe service failures, mismatches, clocks, and shutdown."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    configuration = _configuration(store, clock)
    service = ServedResponsesService(configuration)
    assert await service.now() == _NOW
    default_clock = ServedResponsesService(
        ServedResponsesConfiguration(
            store=configuration.store,
            authority_resolver=configuration.authority_resolver,
            turn_resolver=configuration.turn_resolver,
            policy=configuration.policy,
        )
    )
    assert (await default_clock.now()).utcoffset() == timedelta(0)
    plan = ServedResponsesTurnPlan(
        authority=_authority(),
        input_text="hello",
        public_response_id=conversation.PublicResponseId(
            "resp_avl_" + "c" * 32
        ),
        provisional_response_id=conversation.ProvisionalResponseId(
            "provisional-avl-" + "c" * 32
        ),
        idempotency_key=conversation.RequestIdempotencyKey("request-c"),
        request_fingerprint=_REQUEST_FINGERPRINT,
        retention=_retention(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        compact_threshold=None,
        includes=(),
        tool_names=(),
        parent=None,
        streaming=False,
    )

    with pytest.raises(TypeError, match="configuration"):
        ServedResponsesService(cast(ServedResponsesConfiguration, object()))
    with pytest.raises(conversation.ConversationValidationError):
        await service.prepare_turn(
            authority=plan.authority,
            input_text=plan.input_text,
            parent=None,
            reasoning_context=plan.reasoning_context,
            compact_threshold=None,
            includes=(),
            tool_names=(),
            streaming=False,
            idempotency_key=None,
            request_fingerprint="not-a-digest",
        )

    async def raising_authority(
        request: Request,
    ) -> conversation.AuthorityScope | None:
        del request
        raise RuntimeError("private authority failure")

    rejected = ServedResponsesService(
        replace(configuration, authority_resolver=raising_authority)
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await rejected.authenticate(cast(Request, object()))

    naive = ServedResponsesService(
        replace(configuration, clock=_MutableClock(datetime(2026, 8, 3)))
    )
    with pytest.raises(conversation.ConversationStorageError):
        await naive.now()

    async def wrong_result(
        current: ServedResponsesTurnPlan,
    ) -> PreparedServedResponsesTurn:
        del current
        return cast(PreparedServedResponsesTurn, object())

    invalid_result = ServedResponsesService(
        replace(configuration, turn_resolver=wrong_result)
    )
    with pytest.raises(conversation.ConversationValidationError):
        await invalid_result.prepare_turn(
            authority=plan.authority,
            input_text=plan.input_text,
            parent=None,
            reasoning_context=plan.reasoning_context,
            compact_threshold=None,
            includes=(),
            tool_names=(),
            streaming=False,
            idempotency_key=None,
            request_fingerprint=_REQUEST_FINGERPRINT,
        )

    async def mismatched_result(
        current: ServedResponsesTurnPlan,
    ) -> PreparedServedResponsesTurn:
        prepared = await _turn_resolver(store, clock)(current)
        return replace(
            prepared,
            turn=replace(
                prepared.turn,
                public_response_id=conversation.PublicResponseId(
                    "resp_avl_" + "d" * 32
                ),
            ),
        )

    mismatched = ServedResponsesService(
        replace(configuration, turn_resolver=mismatched_result)
    )
    with pytest.raises(conversation.ConversationValidationError):
        await mismatched.prepare_turn(
            authority=plan.authority,
            input_text=plan.input_text,
            parent=None,
            reasoning_context=plan.reasoning_context,
            compact_threshold=None,
            includes=(),
            tool_names=(),
            streaming=False,
            idempotency_key="mismatch",
            request_fingerprint=_REQUEST_FINGERPRINT,
        )

    prepared = await configuration.turn_resolver(plan)
    await prepared.turn.execute("seed")
    resource = await service.retrieve(
        str(plan.public_response_id), plan.authority
    )
    wrong_kind = replace(
        resource.checkpoint,
        kind=conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY,
    )

    async def load_wrong_kind(
        checkpoint_id: conversation.CheckpointId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationCheckpoint:
        del checkpoint_id, authority
        return wrong_kind

    with monkeypatch.context() as scoped:
        scoped.setattr(configuration.store, "load", load_wrong_kind)
        with pytest.raises(conversation.ConversationAuthorizationError):
            await service.resolve_parent(
                str(plan.public_response_id),
                plan.authority,
            )

    async def retrieve_wrong_id(
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationResult:
        del public_response_id, authority
        return replace(
            resource.result,
            public_response_id=conversation.PublicResponseId(
                "resp_avl_" + "e" * 32
            ),
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(configuration.store, "retrieve", retrieve_wrong_id)
        with pytest.raises(conversation.ConversationAuthorizationError):
            await service.retrieve(
                str(plan.public_response_id),
                plan.authority,
            )

    async def fail_deletion(
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.LocalDeletionPreparation:
        del public_response_id, authority
        raise conversation.ConversationStorageError()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            configuration.store,
            "prepare_deletion",
            fail_deletion,
        )
        with pytest.raises(conversation.ConversationStorageError):
            await service.tombstone(
                str(plan.public_response_id),
                plan.authority,
            )

    class InvalidActivePreparation:
        state = conversation.LocalDeletionState.ACTIVE
        checkpoint = None

    async def invalid_active_deletion(
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.LocalDeletionPreparation:
        del public_response_id, authority
        return cast(
            conversation.LocalDeletionPreparation,
            InvalidActivePreparation(),
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            configuration.store,
            "prepare_deletion",
            invalid_active_deletion,
        )
        with pytest.raises(conversation.ConversationStorageError):
            await service.tombstone(
                str(plan.public_response_id),
                plan.authority,
            )

    tombstoned_checkpoint = await configuration.store.tombstone(
        plan.public_response_id,
        plan.authority,
        _NOW,
    )
    active_preparation = conversation.LocalDeletionPreparation(
        state=conversation.LocalDeletionState.ACTIVE,
        checkpoint=resource.checkpoint,
    )
    settled_preparation = conversation.LocalDeletionPreparation(
        state=conversation.LocalDeletionState.TOMBSTONED,
        checkpoint=tombstoned_checkpoint,
    )
    preparations = iter((active_preparation, settled_preparation))

    async def prepare_racing_deletion(
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.LocalDeletionPreparation:
        del public_response_id, authority
        return next(preparations)

    async def lose_tombstone_race(
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
        at: datetime,
    ) -> conversation.ConversationCheckpoint:
        del public_response_id, authority, at
        raise conversation.ConversationConflictError()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            configuration.store,
            "prepare_deletion",
            prepare_racing_deletion,
        )
        scoped.setattr(
            configuration.store,
            "tombstone",
            lose_tombstone_race,
        )
        assert (
            await service.tombstone(
                str(plan.public_response_id),
                plan.authority,
            )
            is conversation.LocalDeletionState.TOMBSTONED
        )

    async def prepare_unsettled_deletion(
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.LocalDeletionPreparation:
        del public_response_id, authority
        return active_preparation

    with monkeypatch.context() as scoped:
        scoped.setattr(
            configuration.store,
            "prepare_deletion",
            prepare_unsettled_deletion,
        )
        scoped.setattr(
            configuration.store,
            "tombstone",
            lose_tombstone_race,
        )
        with pytest.raises(conversation.ConversationConflictError):
            await service.tombstone(
                str(plan.public_response_id),
                plan.authority,
            )

    class SyncStore:
        durable = True

        def retrieve(self) -> None:
            return None

    with pytest.raises(TypeError, match="store.retrieve"):
        replace(
            configuration,
            store=cast(ServedResponsesDurableStore, SyncStore()),
        )

    def sync_authority(
        request: Request,
    ) -> conversation.AuthorityScope | None:
        del request
        return _authority()

    with pytest.raises(TypeError, match="authority_resolver"):
        replace(
            configuration,
            authority_resolver=cast(
                ResponsesAuthorityResolver,
                sync_authority,
            ),
        )
    with pytest.raises(TypeError, match="policy"):
        replace(
            configuration,
            policy=cast(ServedResponsesPolicy, object()),
        )
    with pytest.raises(TypeError, match="close_store_on_shutdown"):
        replace(
            configuration,
            close_store_on_shutdown=cast(bool, 1),
        )

    app = FastAPI()
    configure_served_responses(app, configuration)
    configure_served_responses(app, None)
    assert not hasattr(app.state, "served_responses_service")
    close_calls = 0

    async def close_store() -> conversation.StoreCloseResolution:
        nonlocal close_calls
        close_calls += 1
        return conversation.StoreCloseResolution(
            disposition=conversation.StoreCloseDisposition.CLOSED
        )

    monkeypatch.setattr(configuration.store, "close", close_store)
    owned_configuration = replace(
        configuration,
        close_store_on_shutdown=True,
    )
    configure_served_responses(app, owned_configuration)
    await close_served_responses(app)
    assert close_calls == 1

    with pytest.raises(conversation.ConversationValidationError):
        StoredResponsesResource(
            result=cast(conversation.ConversationResult, object()),
            checkpoint=cast(conversation.ConversationCheckpoint, object()),
            public_model="served-model",
        )


def test_stored_projection_exposes_only_one_parent_lane() -> None:
    """Hide child output, lane order, and private provider deployment state."""
    store = conversation.InMemoryConversationStore()
    clock = _MutableClock()
    configuration = _configuration(store, clock)
    service = ServedResponsesService(configuration)
    plan = ServedResponsesTurnPlan(
        authority=_authority(),
        input_text="parent-visible",
        public_response_id=conversation.PublicResponseId(
            "resp_avl_" + "6" * 32
        ),
        provisional_response_id=conversation.ProvisionalResponseId(
            "provisional-avl-" + "6" * 32
        ),
        idempotency_key=conversation.RequestIdempotencyKey(
            "projection-request"
        ),
        request_fingerprint=_REQUEST_FINGERPRINT,
        retention=_retention(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        compact_threshold=None,
        includes=(),
        tool_names=(),
        parent=None,
        streaming=False,
    )
    prepared = run(configuration.turn_resolver(plan))
    run(prepared.turn.execute("parent-visible"))
    resource = run(
        service.retrieve(str(plan.public_response_id), plan.authority)
    )
    parent_output = resource.result.lane_outputs[0]
    topology = resource.checkpoint.content.lane_topology
    assert topology is not None
    parent_entry = topology.entries[0]
    child_lane_id = conversation.ProviderLaneId("private-child-lane")
    child_entry = conversation.ProviderLaneTopologyEntry(
        lane_id=child_lane_id,
        owner_kind=conversation.ProviderLaneOwnerKind.CHILD_AGENT,
        agent_id=conversation.ConversationAgentId("private-child-agent"),
        topology_path=(
            f"{parent_entry.topology_path}/child/private-child-agent/primary"
        ),
        model_slot="primary",
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        binding_digest=conversation.IntegrityDigest("a" * 64),
        parent_lane_id=parent_entry.lane_id,
    )
    private_parent_snapshot = replace(
        resource.checkpoint.content.lanes[0],
        binding=replace(
            resource.checkpoint.content.lanes[0].binding,
            model_or_deployment="private-provider-deployment",
        ),
    )
    checkpoint = replace(
        resource.checkpoint,
        content=replace(
            resource.checkpoint.content,
            lanes=(private_parent_snapshot,),
            lane_topology=conversation.ProviderLaneTopology(
                schema_version=1,
                entries=(child_entry, parent_entry),
            ),
        ),
    )
    child_output = replace(
        parent_output,
        lane_id=child_lane_id,
        binding_alias=conversation.SafeAlias("private-child-binding"),
        items=(
            conversation.VisibleTranscriptEntry(
                role=conversation.VisibleTranscriptRole.ASSISTANT,
                content="private child output",
            ),
        ),
        usage=conversation.ProviderUsage(
            input_tokens=700,
            output_tokens=900,
        ),
    )
    result = replace(
        resource.result,
        lane_outputs=(child_output, parent_output),
    )
    projection = StoredResponsesResource(
        result=result,
        checkpoint=checkpoint,
        public_model="served-model",
    ).response_body()
    reordered = StoredResponsesResource(
        result=replace(
            resource.result,
            lane_outputs=(parent_output, child_output),
        ),
        checkpoint=checkpoint,
        public_model="served-model",
    ).response_body()

    assert projection["model"] == "served-model"
    assert projection["output"] == reordered["output"]
    assert projection["usage"] == reordered["usage"]
    assert "turn-1:parent-visible" in str(projection)
    assert "private" not in str(projection)
    usage = cast(dict[str, object], projection["usage"])
    assert usage["input_tokens"] == parent_output.usage.input_tokens
    assert usage["output_tokens"] == parent_output.usage.output_tokens

    second_parent_entry = replace(
        parent_entry,
        lane_id=conversation.ProviderLaneId("second-parent-lane"),
        agent_id=conversation.ConversationAgentId("second-parent-agent"),
        topology_path="agent/second-parent-agent/model/primary",
        binding_digest=conversation.IntegrityDigest("b" * 64),
    )
    duplicate_parent_checkpoint = replace(
        checkpoint,
        content=replace(
            checkpoint.content,
            lane_topology=conversation.ProviderLaneTopology(
                schema_version=1,
                entries=(parent_entry, second_parent_entry),
            ),
        ),
    )
    with pytest.raises(conversation.ConversationStorageError):
        StoredResponsesResource(
            result=resource.result,
            checkpoint=duplicate_parent_checkpoint,
            public_model="served-model",
        ).response_body()
    with pytest.raises(conversation.ConversationStorageError):
        StoredResponsesResource(
            result=replace(
                resource.result,
                lane_outputs=(child_output,),
            ),
            checkpoint=checkpoint,
            public_model="served-model",
        ).response_body()


def test_openapi_exposes_closed_request_and_lifecycle_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match every served Responses schema and surface to its golden."""
    client = _client(monkeypatch, configuration=None)
    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["ResponsesRequest"]
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "conversation"
        / "served_responses_openapi.phase10.json"
    )
    fixture = loads(fixture_path.read_text(encoding="utf-8"))
    fixture_without_digest = {
        key: value
        for key, value in fixture.items()
        if key != "canonical_sha256"
    }

    assert (
        fixture["canonical_sha256"]
        == sha256(
            dumps(
                fixture_without_digest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )

    assert request_schema["additionalProperties"] is False
    assert request_schema["properties"]["store"]["default"] is False
    schema_hashes = {
        name: (
            sha256(
                dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        for name, value in schema["components"]["schemas"].items()
        if name.startswith("Responses")
    }
    assert schema_hashes == fixture["schema_sha256"]

    create = schema["paths"]["/responses"]["post"]
    compact = schema["paths"]["/responses/compact"]["post"]
    item = schema["paths"]["/responses/{response_id}"]
    assert fixture["operations"] == {
        "create": {
            "method": "post",
            "path": "/responses",
            "request": create["requestBody"]["content"]["application/json"][
                "schema"
            ]["$ref"],
            "response": create["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"],
            "stream": create["responses"]["200"]["content"][
                "text/event-stream"
            ]["schema"],
            "error": create["responses"]["400"]["content"]["application/json"][
                "schema"
            ]["$ref"],
        },
        "compact": {
            "method": "post",
            "path": "/responses/compact",
            "request": compact["requestBody"]["content"]["application/json"][
                "schema"
            ]["$ref"],
            "response": compact["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"],
            "error": compact["responses"]["400"]["content"][
                "application/json"
            ]["schema"]["$ref"],
        },
        "retrieve": {
            "method": "get",
            "path": "/responses/{response_id}",
            "response": item["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"],
            "error": item["get"]["responses"]["404"]["content"][
                "application/json"
            ]["schema"]["$ref"],
        },
        "delete": {
            "method": "delete",
            "path": "/responses/{response_id}",
            "response": item["delete"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"],
            "error": item["delete"]["responses"]["404"]["content"][
                "application/json"
            ]["schema"]["$ref"],
        },
    }
    examples = fixture["examples"]
    ResponsesRequest.model_validate(examples["request"])
    ResponsesResource.model_validate(examples["response"])
    assert examples["retrieval"] == {
        "method": "GET",
        "path": "/responses/resp_avl_0123456789abcdef0123456789abcdef",
        "result": "response",
    }
    ResponsesDeletedResource.model_validate(examples["deletion"])
    ResponsesErrorEnvelope.model_validate(examples["error"])
    public_id = examples["response"]["id"]
    assert public_id not in str(examples["stream"][:-1])
    assert public_id in str(examples["stream"][-1])


def _pgsql_store(
    dsn: str,
    schema: str,
    clock: _MutableClock,
) -> conversation.PgsqlConversationStore:
    """Return one encrypted durable store with restart-stable keys."""
    scope = _authority()
    key = conversation.ConversationDataKey(
        key_id="phase9-served-key",
        revision=1,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"9" * 32,
    )
    resolver = conversation.InMemoryConversationKeyResolver(
        {conversation.authority_digest(scope): (key,)}
    )
    return conversation.PgsqlConversationStore.from_settings(
        conversation.PgsqlConversationStoreSettings(
            dsn=dsn,
            schema=schema,
            pool_minimum=1,
            pool_maximum=2,
        ),
        key_resolver=resolver,
        cipher=conversation.AesGcmConversationCipher(),
        clock=clock,
    )


def _pgsql_configuration(
    store: conversation.PgsqlConversationStore,
    clock: _MutableClock,
) -> ServedResponsesConfiguration:
    """Return production-shaped served configuration for a PostgreSQL host."""

    async def resolve_authority(
        request: Request,
    ) -> conversation.AuthorityScope | None:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            return None
        return _authority(header.removeprefix("Bearer "))

    return ServedResponsesConfiguration(
        store=store,
        authority_resolver=resolve_authority,
        turn_resolver=_turn_resolver(store, clock),
        policy=ServedResponsesPolicy(
            agent_id=_AGENT_ID,
            endpoint_id=_ENDPOINT_ID,
            retention=_retention(3600),
            public_model="served-model",
            allowed_reasoning_contexts=frozenset(
                conversation.ReasoningContext
            ),
            allowed_includes=frozenset({"file_search_call.results"}),
            allowed_tool_names=frozenset({"lookup"}),
            min_compact_threshold=10,
            max_compact_threshold=1000,
        ),
        clock=clock,
    )


async def _serve_pgsql_responses(
    dsn: str,
    schema: str,
    port: int,
    readiness: Connection,
    shutdown: Connection,
) -> None:
    """Run one fresh loopback server process over durable PostgreSQL."""
    clock = _MutableClock()
    store = _pgsql_store(dsn, schema, clock)
    await store.open()
    app = FastAPI()
    app.include_router(responses_router.router, prefix="/v1")
    app.dependency_overrides[di_get_logger] = lambda: getLogger(__name__)
    orchestrator = object.__new__(_ResponsesOrchestrator)
    app.dependency_overrides[di_get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[
        responses_router._server_output_redaction_settings
    ] = lambda: ServerOutputRedactionSettings()
    configure_served_responses(app, _pgsql_configuration(store, clock))
    responses_router.orchestrate = _stored_orchestrate([])
    server = Server(
        Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="critical",
            lifespan="off",
        )
    )

    async def announce() -> None:
        for _ in range(1000):
            if server.started:
                readiness.send((True, "ready"))
                return
            await sleep(0.01)
        readiness.send((False, "server did not start"))
        server.should_exit = True

    async def stop_when_requested() -> None:
        while not server.should_exit:
            if shutdown.poll():
                shutdown.recv()
                server.should_exit = True
                return
            await sleep(0.01)

    ready_task = create_task(announce())
    stop_task = create_task(stop_when_requested())
    try:
        await server.serve()
        await ready_task
    finally:
        server.should_exit = True
        await stop_task
        await store.close()


def _pgsql_server_target(
    dsn: str,
    schema: str,
    port: int,
    readiness: Connection,
    shutdown: Connection,
) -> None:
    """Bridge the async server into one spawn-safe process target."""
    try:
        run(_serve_pgsql_responses(dsn, schema, port, readiness, shutdown))
    except BaseException as error:
        try:
            readiness.send((False, (type(error).__name__, str(error))))
        except (BrokenPipeError, OSError):
            pass
        raise
    finally:
        readiness.close()
        shutdown.close()


@dataclass(slots=True)
class _ServerProcess:
    """Own one spawned TCP server and its selected loopback port."""

    process: BaseProcess
    port: int
    shutdown: Connection

    @property
    def base_url(self) -> str:
        """Return the official-client compatible API prefix."""
        return f"http://127.0.0.1:{self.port}/v1"


def _unused_loopback_port() -> int:
    """Return one currently unbound loopback port for a child process."""
    listener = socket(AF_INET, SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        address = cast(tuple[str, int], listener.getsockname())
        return address[1]
    finally:
        listener.close()


async def _start_pgsql_server(dsn: str, schema: str) -> _ServerProcess:
    """Start and verify one fresh Responses server process."""
    context = get_context("spawn")
    parent_readiness, child_readiness = context.Pipe(duplex=False)
    child_shutdown, parent_shutdown = context.Pipe(duplex=False)
    port = _unused_loopback_port()
    process = context.Process(
        target=_pgsql_server_target,
        args=(dsn, schema, port, child_readiness, child_shutdown),
    )
    process.start()
    child_readiness.close()
    child_shutdown.close()
    available = await to_thread(parent_readiness.poll, 20)
    if not available:
        process.terminate()
        await to_thread(process.join, 5)
        parent_shutdown.close()
        pytest.fail("fresh Responses server did not report readiness")
    success, detail = parent_readiness.recv()
    parent_readiness.close()
    if not success:
        process.terminate()
        await to_thread(process.join, 5)
        parent_shutdown.close()
        pytest.fail(f"fresh Responses server failed: {detail!r}")
    assert process.is_alive()
    return _ServerProcess(
        process=process,
        port=port,
        shutdown=parent_shutdown,
    )


async def _stop_pgsql_server(server: _ServerProcess) -> None:
    """Stop one owned child server and verify resource settlement."""
    if server.process.is_alive():
        server.shutdown.send("stop")
    server.shutdown.close()
    await to_thread(server.process.join, 10)
    if server.process.is_alive():
        server.process.kill()
        await to_thread(server.process.join, 5)
        pytest.fail("fresh Responses server did not stop")
    assert server.process.exitcode == 0


async def _drop_schema(dsn: str, schema: str) -> None:
    """Drop only the isolated schema owned by this acceptance test."""
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


@pytest.mark.anyio
@pytest.mark.skipif(
    _DSN is None,
    reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
)
async def test_normative_server_stored_contract(
    record_property: Callable[[str, object], None],
) -> None:
    """Survive real TCP server restarts through the official async client."""
    record_property("conversation_acceptance_evidence", "public")
    assert _DSN is not None
    schema = f"conv_phase9_server_{uuid4().hex}"
    assert schema.startswith("conv_phase9_server_")
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    first_server: _ServerProcess | None = None
    second_server: _ServerProcess | None = None
    third_server: _ServerProcess | None = None
    try:
        first_server = await _start_pgsql_server(_DSN, schema)
        async with AsyncOpenAI(
            api_key="owner",
            base_url=first_server.base_url,
        ) as client:
            first = await client.responses.create(
                model="served-model",
                input="alpha",
                store=True,
                max_output_tokens=32,
            )
            assert first.id.startswith("resp_avl_")
            assert first.output_text == "turn-1:alpha"
            retrieved_first = await client.responses.retrieve(first.id)
            assert retrieved_first.model_dump() == first.model_dump()
        await _stop_pgsql_server(first_server)
        first_server = None

        second_server = await _start_pgsql_server(_DSN, schema)
        async with AsyncOpenAI(
            api_key="owner",
            base_url=second_server.base_url,
        ) as client:
            second = await client.responses.create(
                model="served-model",
                input="beta",
                store=True,
                previous_response_id=first.id,
            )
            assert second.id != first.id
            assert second.output_text == "turn-2:beta"
            first_again = await client.responses.retrieve(first.id)
            second_again = await client.responses.retrieve(second.id)
            assert first_again.output_text == "turn-1:alpha"
            assert second_again.output_text == "turn-2:beta"

            stream = await client.responses.create(
                model="served-model",
                input="gamma",
                store=True,
                stream=True,
                previous_response_id=second.id,
                stream_options={"include_obfuscation": False},
            )
            events = [event async for event in stream]
            assert events[0].type == "response.created"
            assert events[-1].type == "response.completed"
            terminal = events[-1]
            assert terminal.type == "response.completed"
            third = terminal.response
            assert third.output_text == "turn-3:gamma"
            assert third.id not in str(events[:-1])
            third_again = await client.responses.retrieve(third.id)
            assert third_again.output_text == "turn-3:gamma"

            idempotency_extension = {
                "extensions": {
                    "avalan": {
                        "version": "1",
                        "conversation": {
                            "version": "1",
                            "idempotency_key": "official-client-retry",
                        },
                    }
                }
            }
            retry_origin = await client.responses.create(
                model="served-model",
                input="official-retry",
                store=True,
                extra_body=idempotency_extension,
            )
            retry_replay = await client.responses.create(
                model="served-model",
                input="official-retry",
                store=True,
                extra_body=idempotency_extension,
            )
            assert retry_replay.model_dump() == retry_origin.model_dump()
            with pytest.raises(ConflictError):
                await client.responses.create(
                    model="served-model",
                    input="changed-official-retry",
                    store=True,
                    extra_body=idempotency_extension,
                )

            async with AsyncOpenAI(
                api_key="another-principal",
                base_url=second_server.base_url,
            ) as other:
                with pytest.raises(NotFoundError):
                    await other.responses.retrieve(first.id)
                with pytest.raises(NotFoundError):
                    await other.responses.create(
                        model="served-model",
                        input="forbidden",
                        store=True,
                        previous_response_id=first.id,
                    )
                with pytest.raises(NotFoundError):
                    await other.responses.delete(first.id)

            concurrent_delete_target = await client.responses.create(
                model="served-model",
                input="concurrent-first-delete",
                store=True,
            )
            concurrent_first_deletes = await gather(
                client.responses.delete(concurrent_delete_target.id),
                client.responses.delete(concurrent_delete_target.id),
            )
            assert all(result is None for result in concurrent_first_deletes)

            race_parent = await client.responses.create(
                model="served-model",
                input="race-parent",
                store=True,
            )
            continued, deleted = await gather(
                client.responses.create(
                    model="served-model",
                    input="race-child",
                    store=True,
                    previous_response_id=race_parent.id,
                ),
                client.responses.delete(race_parent.id),
                return_exceptions=True,
            )
            assert deleted is None
            assert isinstance(continued, NotFoundError) or (
                not isinstance(continued, BaseException)
                and continued.output_text == "turn-2:race-child"
            )
            with pytest.raises(NotFoundError):
                await client.responses.retrieve(race_parent.id)
            duplicate_deletes = await gather(
                client.responses.delete(race_parent.id),
                client.responses.delete(race_parent.id),
            )
            assert duplicate_deletes == [None, None]

            assert await client.responses.delete(third.id) is None
            assert await client.responses.delete(third.id) is None
        await _stop_pgsql_server(second_server)
        second_server = None

        third_server = await _start_pgsql_server(_DSN, schema)
        async with AsyncOpenAI(
            api_key="owner",
            base_url=third_server.base_url,
        ) as client:
            with pytest.raises(NotFoundError):
                await client.responses.retrieve(third.id)
            with pytest.raises(NotFoundError):
                await client.responses.create(
                    model="served-model",
                    input="deleted-parent",
                    store=True,
                    previous_response_id=third.id,
                )
            retained = await client.responses.retrieve(second.id)
            assert retained.output_text == "turn-2:beta"
            await client.responses.create(
                model="served-model",
                input="expiry-race",
                store=True,
            )
        await _stop_pgsql_server(third_server)
        third_server = None

        expiry_clock = _MutableClock(_NOW + timedelta(hours=2))
        expiry_store = _pgsql_store(_DSN, schema, expiry_clock)
        await expiry_store.open()
        try:
            retrieved, swept = await gather(
                expiry_store.retrieve(
                    conversation.PublicResponseId(second.id),
                    _authority(),
                ),
                expiry_store.sweep(expiry_clock.value, limit=100),
                return_exceptions=True,
            )
            assert isinstance(
                retrieved,
                conversation.ConversationResult
                | conversation.ConversationAuthorizationError,
            )
            assert isinstance(swept, conversation.SweepReceipt)
            with pytest.raises(conversation.ConversationAuthorizationError):
                await expiry_store.retrieve(
                    conversation.PublicResponseId(second.id),
                    _authority(),
                )
        finally:
            await expiry_store.close()
    finally:
        for server in (first_server, second_server, third_server):
            if server is not None:
                await _stop_pgsql_server(server)
        await _drop_schema(_DSN, schema)
