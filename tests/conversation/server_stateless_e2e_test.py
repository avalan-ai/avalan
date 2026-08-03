"""Exercise stateless Responses lifecycle and compact HTTP boundaries."""

from asyncio import (
    CancelledError,
    StreamReader,
    StreamWriter,
    create_task,
    gather,
    run,
    sleep,
    start_server,
    to_thread,
)
from asyncio import (
    Server as AsyncioServer,
)
from base64 import b64decode, urlsafe_b64encode
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import dumps, loads
from logging import getLogger
from multiprocessing import get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from os import getpid
from socket import AF_INET, SOCK_STREAM, socket
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from openai import AsyncOpenAI, NotFoundError
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
from uvicorn import Config, Server

import avalan.conversation as conversation
import avalan.server.stateless_responses as stateless_module
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import ReasoningSummaryMode
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
)
from avalan.server.entities import ResponsesCompactRequest, ResponsesRequest
from avalan.server.interaction import ServerInteractionHandling
from avalan.server.responses_lifecycle import (
    ResponsesAuthorityResolver,
    ResponsesClock,
)
from avalan.server.responses_schema import ResponsesConversationExtension
from avalan.server.routers import responses
from avalan.server.stateless_responses import (
    InMemoryStatelessNamedHeadLedger,
    InMemoryStatelessResponsesAuditHook,
    PreparedStatelessResponse,
    PreparedStatelessResponsesTurn,
    StatelessCompactPlan,
    StatelessCompactResult,
    StatelessResponseOutcome,
    StatelessResponsesConfiguration,
    StatelessResponsesPolicy,
    StatelessResponsesService,
    StatelessResponsesTurnPlan,
)
from avalan.types import JsonValue
from avalan.utils import to_json

pytestmark = pytest.mark.anyio

_NOW = datetime(2036, 1, 1, tzinfo=UTC)


def _tamper_authenticated_ciphertext(token: str) -> str:
    encoded = token.removeprefix(conversation.CONTINUATION_ENVELOPE_PREFIX)
    wrapper = cast(
        dict[str, object],
        loads(b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_")),
    )
    ciphertext_value = cast(str, wrapper["ciphertext"])
    ciphertext = bytearray(
        b64decode(
            ciphertext_value + "=" * (-len(ciphertext_value) % 4),
            altchars=b"-_",
        )
    )
    ciphertext[-1] ^= 1
    wrapper["ciphertext"] = (
        urlsafe_b64encode(bytes(ciphertext)).rstrip(b"=").decode("ascii")
    )
    return conversation.CONTINUATION_ENVELOPE_PREFIX + (
        urlsafe_b64encode(
            dumps(
                wrapper,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        .rstrip(b"=")
        .decode("ascii")
    )


def _rewrite_envelope_key(
    token: str,
    *,
    key_id: str,
    revision: int,
) -> str:
    encoded = token.removeprefix(conversation.CONTINUATION_ENVELOPE_PREFIX)
    wrapper = cast(
        dict[str, object],
        loads(b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_")),
    )
    wrapper["key_id"] = key_id
    wrapper["key_revision"] = revision
    return conversation.CONTINUATION_ENVELOPE_PREFIX + (
        urlsafe_b64encode(
            dumps(
                wrapper,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        .rstrip(b"=")
        .decode("ascii")
    )


@pytest.fixture
def anyio_backend() -> str:
    """Run served stateless checks on asyncio."""
    return "asyncio"


def _authority(
    principal: str = "principal-stateless",
) -> conversation.AuthorityScope:
    return conversation.AuthorityScope(
        source=conversation.AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=conversation.AuthorityTenantId("tenant-stateless"),
        principal_id=conversation.AuthorityPrincipalId(principal),
        agent_id=conversation.ConversationAgentId("agent-stateless"),
        endpoint_id=conversation.AuthorityEndpointId("endpoint-stateless"),
        network_exposed=True,
    )


def _retention() -> conversation.RetentionLimits:
    return conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.TRANSIENT,
            upstream=conversation.ProviderLaneStorage.STATELESS,
        ),
        upstream_lifetime_status=(
            conversation.UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
        local_ttl_seconds=3_600,
    )


def _binding_seed() -> conversation.ProviderLaneBinding:
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId("lane-seed"),
        adapter_type="tests.ServedStatelessProvider",
        provider_family=conversation.ProviderFamily.SYNTHETIC,
        normalized_endpoint="https://stateless.provider.test/v1",
        model_or_deployment="server-model",
        provider_api_revision=conversation.ProviderApiRevision("api-v1"),
        sdk_revision=conversation.ProviderSdkRevision("sdk-v1"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("config-v1")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-v1")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-v1"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-v1")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.NON_STREAMING,
        agent_id=conversation.ConversationAgentId("agent-stateless"),
    )


def _runtime_topology() -> tuple[
    conversation.AgentLaneTopology,
    conversation.AgentProviderLane,
]:
    conversation_id = conversation.ConversationId("conversation-stateless")
    model_slot = conversation.AgentModelSlot("primary")
    agent_id = conversation.ConversationAgentId("agent-stateless")
    topology_path = conversation.parent_agent_topology_path(
        agent_id,
        model_slot,
    )
    seed = _binding_seed()
    lane_id = conversation.derive_agent_provider_lane_id(
        conversation_id=conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        topology_path=topology_path,
        model_slot=model_slot,
        binding=seed,
    )
    lane = conversation.AgentProviderLane(
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        agent_id=agent_id,
        topology_path=topology_path,
        model_slot=model_slot,
        binding=replace(seed, lane_id=lane_id),
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    return (
        conversation.AgentLaneTopology(
            conversation_id=conversation_id,
            lanes=(lane,),
        ),
        lane,
    )


def _provider_message(
    lane: conversation.AgentProviderLane,
    identifier: str,
    text: str,
) -> conversation.ProviderItem:
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId(identifier),
        lane_id=lane.lane_id,
        model_call_id=conversation.ConversationModelCallId(
            f"call-{identifier}"
        ),
        kind=conversation.ProviderItemKind.MESSAGE,
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.FINAL,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input=cast(
            Mapping[str, JsonValue],
            {
                "content": [
                    {
                        "annotations": [],
                        "text": text,
                        "type": "output_text",
                    }
                ],
                "id": identifier,
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ),
        normalization_version=conversation.ConversationCodecVersion(1),
    )


class _UnusedCoordinator:
    async def execute(self, request: object) -> object:
        del request
        raise AssertionError("test resolver must not dispatch")


class _TransientStore:
    durable = False

    def __init__(
        self,
        result: conversation.ConversationResult,
        checkpoint: conversation.ConversationCheckpoint,
    ) -> None:
        self._result: conversation.ConversationResult | None = result
        self._checkpoint: conversation.ConversationCheckpoint | None = (
            checkpoint
        )
        self.closed = False

    async def retrieve(
        self,
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationResult:
        if (
            self.closed
            or self._result is None
            or self._checkpoint is None
            or public_response_id != self._result.public_response_id
            or authority != self._checkpoint.authority
        ):
            raise conversation.ConversationStorageError()
        return self._result

    async def load(
        self,
        checkpoint_id: conversation.CheckpointId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationCheckpoint:
        if (
            self.closed
            or self._checkpoint is None
            or checkpoint_id != self._checkpoint.identity.checkpoint_id
            or authority != self._checkpoint.authority
        ):
            raise conversation.ConversationStorageError()
        return self._checkpoint

    async def close(self) -> conversation.StoreCloseResolution:
        self.closed = True
        self._result = None
        self._checkpoint = None
        return conversation.StoreCloseResolution(
            disposition=conversation.StoreCloseDisposition.CLOSED
        )

    async def audit_non_retention(
        self,
    ) -> conversation.StoreNonRetentionAudit:
        count = 0 if self.closed else 1
        return conversation.StoreNonRetentionAudit(
            checkpoints=count,
            provider_ledgers=count,
            public_mappings=count,
            provisional_mappings=count,
            idempotency_records=count,
            named_heads=0,
            queues=0,
            outbox_records=0,
            task_state=0,
            envelope_plaintexts=0,
            temporary_files=0,
        )


class _TurnResolver:
    def __init__(self) -> None:
        self.topology, self.lane = _runtime_topology()
        self.stores: list[_TransientStore] = []
        self.plans: list[StatelessResponsesTurnPlan] = []

    async def __call__(
        self,
        plan: StatelessResponsesTurnPlan,
    ) -> PreparedStatelessResponsesTurn:
        self.plans.append(plan)
        parent = (
            plan.opened_parent.checkpoint
            if plan.opened_parent is not None
            else None
        )
        branch_id = (
            plan.opened_parent.target_branch_id
            if plan.opened_parent is not None
            else conversation.ConversationBranchId("branch-stateless")
        )
        sequence = (
            int(parent.identity.sequence) + 1 if parent is not None else 0
        )
        identity = conversation.CheckpointIdentity(
            conversation_id=self.topology.conversation_id,
            logical_turn_id=conversation.LogicalTurnId(
                f"turn-{plan.public_response_id}"
            ),
            execution_segment_id=conversation.ExecutionSegmentId(
                f"segment-{plan.public_response_id}"
            ),
            checkpoint_id=conversation.CheckpointId(
                f"checkpoint-{plan.public_response_id}"
            ),
            branch_id=branch_id,
            sequence=conversation.CheckpointSequence(sequence),
            parent_checkpoint_id=(
                parent.identity.checkpoint_id if parent is not None else None
            ),
            parent_sequence=(
                parent.identity.sequence if parent is not None else None
            ),
        )
        item = _provider_message(
            self.lane,
            f"item-{plan.public_response_id}",
            "safe terminal",
        )
        checkpoint = conversation.with_checkpoint_integrity(
            conversation.ConversationCheckpoint(
                identity=identity,
                kind=conversation.CheckpointKind.COMPLETED_OUTWARD_TURN,
                lifecycle=conversation.CheckpointLifecycle.COMMITTED,
                authority=plan.authority,
                content=conversation.MultiLaneCheckpointContent(
                    visible_transcript=conversation.VisibleTranscript(
                        entries=(
                            conversation.VisibleTranscriptEntry(
                                role=(
                                    conversation.VisibleTranscriptRole.ASSISTANT
                                ),
                                content="safe terminal",
                            ),
                        )
                    ),
                    lanes=(
                        conversation.StatelessProviderLaneSnapshot(
                            binding=self.lane.binding,
                            ledger=conversation.ProviderItemLedger(
                                lane_id=self.lane.lane_id,
                                normalization_version=(
                                    self.lane.binding.continuation_codec_version
                                ),
                                items=(item,),
                            ),
                            reasoning=(
                                conversation.EffectiveReasoningMetadata(
                                    requested=plan.reasoning_context,
                                    effective=None,
                                )
                            ),
                            lifecycle=(
                                conversation.ProviderLaneLifecycle.COMMITTED
                            ),
                            retention_policy=self.lane.retention_policy,
                        ),
                    ),
                    lane_topology=self.topology.checkpoint_topology(),
                ),
                timestamps=conversation.CheckpointTimestamps(
                    created_at=_NOW + timedelta(seconds=sequence),
                    committed_at=_NOW + timedelta(seconds=sequence),
                    expires_at=_NOW + timedelta(hours=1),
                ),
                retention=plan.retention,
            )
        )
        result = conversation.ConversationResult(
            handle=conversation.StatelessConversationHandle(
                conversation_id=identity.conversation_id,
                checkpoint_id=identity.checkpoint_id,
                branch_id=identity.branch_id,
            ),
            reasoning=conversation.EffectiveReasoningMetadata(
                requested=plan.reasoning_context,
                effective=None,
            ),
            checkpoint_digest=cast(
                conversation.IntegrityDigest,
                (
                    checkpoint.integrity.digest
                    if checkpoint.integrity is not None
                    else ""
                ),
            ),
            lane_outputs=(
                conversation.ProviderLaneOutput(
                    lane_id=self.lane.lane_id,
                    binding_alias=self.lane.binding.safe_alias,
                    mode=conversation.ConversationMode.STATELESS,
                    scope=(conversation.ProviderLaneOutputScope.CURRENT_CALL),
                    items=(
                        conversation.VisibleTranscriptEntry(
                            role=conversation.VisibleTranscriptRole.ASSISTANT,
                            content="safe terminal",
                        ),
                    ),
                    reasoning=conversation.EffectiveReasoningMetadata(
                        requested=plan.reasoning_context,
                        effective=None,
                    ),
                    usage=conversation.ProviderUsage(
                        input_tokens=3,
                        output_tokens=2,
                    ),
                ),
            ),
            public_response_id=plan.public_response_id,
        )
        store = _TransientStore(result, checkpoint)
        self.stores.append(store)
        advance: (
            conversation.ExplicitBranchAdvance
            | conversation.NamedHeadAdvance
            | None
        ) = None
        if parent is not None and (
            plan.advance.mode is conversation.ParentAdvanceMode.EXPLICIT_BRANCH
        ):
            assert plan.advance.branch_id is not None
            advance = conversation.ExplicitBranchAdvance(
                parent_checkpoint_id=parent.identity.checkpoint_id,
                branch_id=plan.advance.branch_id,
            )
        elif parent is not None and (
            plan.advance.mode is conversation.ParentAdvanceMode.NAMED_HEAD
        ):
            assert plan.advance.head_id is not None
            assert plan.advance.expected_head_revision is not None
            advance = conversation.NamedHeadAdvance(
                head_id=plan.advance.head_id,
                parent_checkpoint_id=parent.identity.checkpoint_id,
                expected_revision=plan.advance.expected_head_revision,
            )
        turn = conversation.AgentConversationTurn(
            coordinator=cast(
                conversation.ConversationCoordinator,
                _UnusedCoordinator(),
            ),
            authority=plan.authority,
            topology=self.topology,
            lanes=(
                conversation.AgentConversationLane(
                    lane_id=self.lane.lane_id,
                    mode=conversation.ConversationMode.STATELESS,
                    reasoning_context=plan.reasoning_context,
                ),
            ),
            logical_turn_id=identity.logical_turn_id,
            execution_segment_id=identity.execution_segment_id,
            checkpoint_id=identity.checkpoint_id,
            branch_id=identity.branch_id,
            provisional_response_id=plan.provisional_response_id,
            public_response_id=plan.public_response_id,
            idempotency_key=plan.idempotency_key,
            retention=plan.retention,
            parent=parent,
            advance=advance,
        )
        return PreparedStatelessResponsesTurn(turn=turn, store=store)


class _Clock:
    async def now(self) -> datetime:
        return _NOW


@dataclass(frozen=True, slots=True)
class _FixedClock:
    value: datetime

    async def now(self) -> datetime:
        return self.value


class _CompactResolver:
    def __init__(self) -> None:
        self.plans: list[StatelessCompactPlan] = []

    async def __call__(
        self,
        plan: StatelessCompactPlan,
    ) -> StatelessCompactResult:
        self.plans.append(plan)
        return StatelessCompactResult(
            id="resp_compacted",
            created_at=int(_NOW.timestamp()),
            output=(
                {
                    "content": (
                        {
                            "annotations": (),
                            "text": "provider compact output",
                            "type": "output_text",
                        },
                    ),
                    "id": "message-compacted",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ),
            usage=conversation.ProviderUsage(
                input_tokens=7,
                output_tokens=1,
            ),
            checkpoint=plan.checkpoint,
        )


async def _authority_resolver(request: Request) -> conversation.AuthorityScope:
    principal = request.headers.get("x-test-principal")
    if principal is None:
        authorization = request.headers.get("authorization", "")
        principal = (
            "principal-other"
            if authorization.endswith("another-principal")
            else "principal-stateless"
        )
    return _authority(principal)


def _configuration() -> tuple[
    StatelessResponsesConfiguration,
    _TurnResolver,
    _CompactResolver,
    InMemoryStatelessResponsesAuditHook,
    InMemoryStatelessNamedHeadLedger,
]:
    scope = _authority()
    turn_resolver = _TurnResolver()
    compact_resolver = _CompactResolver()
    audit = InMemoryStatelessResponsesAuditHook()
    heads = InMemoryStatelessNamedHeadLedger()
    key = conversation.ContinuationEnvelopeKey(
        key_id="key-stateless",
        revision=1,
        status=conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
        key_bytes=b"s" * 32,
    )
    codec = conversation.ContinuationEnvelopeCodec(
        key_resolver=conversation.InMemoryContinuationEnvelopeKeyResolver(
            {conversation.authority_digest(scope): (key,)}
        )
    )
    return (
        StatelessResponsesConfiguration(
            authority_resolver=_authority_resolver,
            turn_resolver=turn_resolver,
            envelope_codec=codec,
            policy=StatelessResponsesPolicy(
                agent_id="agent-stateless",
                endpoint_id="endpoint-stateless",
                deployment_id="deployment-stateless",
                retention=_retention(),
                compact_lane_id=str(turn_resolver.lane.lane_id),
                public_model="server-model",
            ),
            compact_resolver=compact_resolver,
            named_head_ledger=heads,
            audit_hook=audit,
            clock=_Clock(),
        ),
        turn_resolver,
        compact_resolver,
        audit,
        heads,
    )


def _non_retention(
    count: int = 0,
) -> conversation.StoreNonRetentionAudit:
    """Return one content-free transient-store audit fixture."""
    return conversation.StoreNonRetentionAudit(
        checkpoints=count,
        provider_ledgers=0,
        public_mappings=0,
        provisional_mappings=0,
        idempotency_records=0,
        named_heads=0,
        queues=0,
        outbox_records=0,
        task_state=0,
        envelope_plaintexts=0,
        temporary_files=0,
    )


async def test_request_validation_and_memory_audit_close_all_states() -> None:
    memory = conversation.InMemoryConversationStore()
    audit = await memory.audit_non_retention()
    assert audit == _non_retention()
    with pytest.raises(conversation.ConversationValidationError):
        replace(audit, checkpoints=-1)

    invalid_extensions = (
        {
            "version": "1",
            "continuation_envelope": "avl_ce1.valid",
        },
        {
            "version": "1",
            "mode": "caller_held",
            "operation": "continue",
            "branch_id": "branch-invalid",
        },
        {
            "version": "1",
            "mode": "caller_held",
            "operation": "branch",
            "branch_id": "branch-without-envelope",
        },
        {
            "version": "1",
            "mode": "caller_held",
            "operation": "named_head",
            "head_id": "head-without-parent",
            "expected_head_revision": 1,
        },
    )
    for extension in invalid_extensions:
        with pytest.raises(ValueError):
            ResponsesConversationExtension.model_validate(extension)
    envelope_token = conversation.ContinuationEnvelopeToken.from_request(
        "avl_ce1.valid",
        max_chars=6_000_000,
    )
    validated_token = ResponsesConversationExtension.model_validate(
        {
            "version": "1",
            "mode": "caller_held",
            "continuation_envelope": envelope_token,
        }
    )
    assert validated_token.continuation_envelope is envelope_token
    for invalid_token in (1, "not-an-envelope"):
        with pytest.raises(ValueError, match="invalid continuation envelope"):
            ResponsesConversationExtension.model_validate(
                {
                    "version": "1",
                    "mode": "caller_held",
                    "continuation_envelope": invalid_token,
                }
            )
    assert (
        ResponsesConversationExtension.model_validate(
            {
                "version": "1",
                "mode": "caller_held",
                "operation": "branch",
                "branch_id": "branch-valid",
                "continuation_envelope": "avl_ce1.valid",
            }
        ).operation
        == "branch"
    )
    assert (
        ResponsesConversationExtension.model_validate(
            {
                "version": "1",
                "mode": "caller_held",
                "operation": "named_head",
                "head_id": "head-root",
                "expected_head_revision": 0,
            }
        ).operation
        == "named_head"
    )

    caller_held_extension = {
        "avalan": {
            "version": "1",
            "conversation": {
                "version": "1",
                "mode": "caller_held",
                "continuation_envelope": "avl_ce1.valid",
            },
        }
    }
    with pytest.raises(ValueError):
        ResponsesRequest.model_validate(
            {
                "model": "server-model",
                "input": "invalid retained caller state",
                "store": True,
                "extensions": caller_held_extension,
            }
        )

    tagged_message = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "complete context"}],
    }
    with pytest.raises(ValueError):
        ResponsesCompactRequest.model_validate(
            {
                "model": "server-model",
                "input": [tagged_message],
                "previous_response_id": "resp_forbidden",
            }
        )
    with pytest.raises(ValueError):
        ResponsesCompactRequest.model_validate({"model": "server-model"})
    envelope_only = ResponsesCompactRequest.model_validate(
        {
            "model": "server-model",
            "extensions": caller_held_extension,
        }
    )
    assert envelope_only.input is None

    invalid_compact_inputs = (
        [{"role": "user", "content": "easy input"}],
        [{**tagged_message, "status": "incomplete"}],
        [
            {
                "type": "function_call",
                "arguments": "{}",
                "call_id": "call-wrong-adjacency",
                "name": "lookup",
                "status": "completed",
            },
            tagged_message,
        ],
        [
            {
                "type": "function_call_output",
                "call_id": "call-leading-output",
                "output": "orphaned",
                "status": "completed",
            }
        ],
        [
            tagged_message,
            {
                "type": "function_call_output",
                "call_id": "call-trailing-output",
                "output": "orphaned",
                "status": "completed",
            },
        ],
    )
    for compact_input in invalid_compact_inputs:
        with pytest.raises(ValueError):
            ResponsesCompactRequest.model_validate(
                {
                    "model": "server-model",
                    "input": compact_input,
                }
            )
    with pytest.raises(
        ValueError,
        match="compact tool pair requires a correlation identity",
    ):
        ResponsesCompactRequest.model_validate(
            {
                "model": "server-model",
                "input": [
                    {
                        "type": "tool_search_call",
                        "arguments": {"query": "missing identity"},
                        "status": "completed",
                    },
                    {
                        "type": "tool_search_output",
                        "tools": [],
                        "status": "completed",
                    },
                ],
            }
        )


def test_caller_envelope_is_opaque_in_diagnostics_and_utilities(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expose caller state only through the exact terminal response field."""
    raw = "avl_ce1.secret_caller_state"
    request = ResponsesRequest.model_validate(
        {
            "input": "safe input",
            "model": "server-model",
            "store": False,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "mode": "caller_held",
                        "continuation_envelope": raw,
                    },
                }
            },
        }
    )
    assert request.extensions is not None
    assert request.extensions.avalan is not None
    extension = request.extensions.avalan.conversation
    assert extension is not None
    token = extension.continuation_envelope
    assert type(token) is conversation.ContinuationEnvelopeToken
    assert raw not in repr(request)
    assert raw not in repr(extension)
    assert raw not in repr(token)
    assert raw not in repr(request.model_dump())
    with pytest.raises(PydanticSerializationError):
        request.model_dump_json()
    with pytest.raises(TypeError):
        dumps(request.model_dump())
    with pytest.raises(TypeError):
        asdict(token)
    with pytest.raises(TypeError):
        to_json(token)

    safe_payload = responses._responses_request_safe_payload(request)
    assert raw not in dumps(safe_payload, sort_keys=True)
    assert token.digest in dumps(safe_payload, sort_keys=True)
    with caplog.at_level("WARNING"):
        getLogger(__name__).warning("request diagnostic: %r", request)
    assert raw not in caplog.text

    with pytest.raises(ValidationError) as raised:
        ResponsesRequest.model_validate(
            {
                "input": "safe input",
                "extensions": {
                    "avalan": {
                        "version": "1",
                        "conversation": {
                            "version": "1",
                            "mode": "caller_held",
                            "operation": "branch",
                            "continuation_envelope": raw,
                        },
                    }
                },
            }
        )
    assert raw not in str(raised.value)
    exact_response = stateless_module.continuation_response_extension(token)
    assert (
        exact_response["avalan"]["conversation"]["continuation_envelope"]
        == raw
    )


_COMPACT_PAIR_CASES: tuple[
    tuple[str, dict[str, object], dict[str, object], str], ...
] = (
    (
        "function",
        {
            "type": "function_call",
            "arguments": "{}",
            "call_id": "pair-function",
            "name": "lookup",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "pair-function",
            "output": "ok",
            "status": "completed",
        },
        "call_id",
    ),
    (
        "computer",
        {
            "type": "computer_call",
            "id": "computer-item",
            "call_id": "pair-computer",
            "pending_safety_checks": [],
            "status": "completed",
        },
        {
            "type": "computer_call_output",
            "call_id": "pair-computer",
            "output": "ok",
            "status": "completed",
        },
        "call_id",
    ),
    (
        "tool_search",
        {
            "type": "tool_search_call",
            "arguments": {"query": "lookup"},
            "call_id": "pair-tool-search",
            "status": "completed",
        },
        {
            "type": "tool_search_output",
            "call_id": "pair-tool-search",
            "tools": [],
            "status": "completed",
        },
        "call_id",
    ),
    (
        "local_shell",
        {
            "type": "local_shell_call",
            "action": {"type": "exec", "command": ["pwd"], "env": {}},
            "call_id": "pair-local-shell",
            "id": "local-shell-item",
            "status": "completed",
        },
        {
            "type": "local_shell_call_output",
            "id": "pair-local-shell",
            "output": "ok",
            "status": "completed",
        },
        "id",
    ),
    (
        "shell",
        {
            "type": "shell_call",
            "action": {"commands": ["pwd"]},
            "call_id": "pair-shell",
            "status": "completed",
        },
        {
            "type": "shell_call_output",
            "call_id": "pair-shell",
            "output": [
                {
                    "outcome": {"type": "exit", "exit_code": 0},
                    "stderr": "",
                    "stdout": "ok",
                }
            ],
            "status": "completed",
        },
        "call_id",
    ),
    (
        "apply_patch",
        {
            "type": "apply_patch_call",
            "call_id": "pair-patch",
            "operation": {"type": "create_file", "path": "safe.txt"},
            "status": "completed",
        },
        {
            "type": "apply_patch_call_output",
            "call_id": "pair-patch",
            "status": "completed",
        },
        "call_id",
    ),
    (
        "mcp_approval",
        {
            "type": "mcp_approval_request",
            "arguments": "{}",
            "id": "pair-mcp",
            "name": "lookup",
            "server_label": "safe",
        },
        {
            "type": "mcp_approval_response",
            "approval_request_id": "pair-mcp",
            "approve": True,
        },
        "approval_request_id",
    ),
    (
        "custom_tool",
        {
            "type": "custom_tool_call",
            "call_id": "pair-custom",
            "input": "lookup",
            "name": "lookup",
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "pair-custom",
            "output": "ok",
        },
        "call_id",
    ),
)


@pytest.mark.parametrize(
    ("pair_name", "call", "output", "output_id_field"),
    _COMPACT_PAIR_CASES,
    ids=tuple(case[0] for case in _COMPACT_PAIR_CASES),
)
def test_compact_pair_matrix_rejects_before_provider_dispatch(
    pair_name: str,
    call: dict[str, object],
    output: dict[str, object],
    output_id_field: str,
) -> None:
    """Require exact adjacency and identity for every frozen pair kind."""
    configuration, _, resolver, _, _ = _configuration()
    app = FastAPI()
    app.state.stateless_responses_service = StatelessResponsesService(
        configuration
    )
    app.include_router(responses.router)
    client = TestClient(app)
    payload = {
        "model": "server-model",
        "input": [call, output],
    }
    accepted = client.post("/responses/compact", json=payload)
    assert accepted.status_code == 200, (pair_name, accepted.text)
    assert len(resolver.plans) == 1

    mismatched = dict(output)
    mismatched[output_id_field] = "pair-mismatch"
    wrong_output = dict(
        _COMPACT_PAIR_CASES[1 if pair_name == "function" else 0][2]
    )
    invalid_inputs = (
        [call],
        [output],
        [output, call],
        [call, mismatched],
        [call, wrong_output],
        [call, output, call, output],
        [
            call,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "reordered"}],
            },
            output,
        ],
    )
    for invalid_input in invalid_inputs:
        rejected = client.post(
            "/responses/compact",
            json={"model": "server-model", "input": invalid_input},
        )
        assert rejected.status_code == 400, (pair_name, rejected.text)
        assert len(resolver.plans) == 1


async def test_finalize_removes_state_and_named_head_cas() -> None:
    configuration, resolver, _, audit, heads = _configuration()
    service = StatelessResponsesService(configuration)
    scope = _authority()
    first = await service.prepare_turn(
        authority=scope,
        input_text="first",
        request_fingerprint=sha256(b"first").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key="first-key",
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.NAMED_HEAD,
            head_id=conversation.NamedHeadId("head-stateless"),
            expected_head_revision=conversation.NamedHeadRevision(0),
        ),
    )
    committed = await service.finalize(
        first,
        request_bytes=20,
        response_bytes=40,
        input_items=1,
        output_items=1,
    )
    assert committed.audit.reconstructable_state_count == 0
    assert resolver.stores[0].closed
    assert audit.records[-1].reconstructable_state_count == 0
    extensions = cast(
        Mapping[str, object],
        committed.body["extensions"],
    )
    avalan_extension = cast(Mapping[str, object], extensions["avalan"])
    conversation_extension = cast(
        Mapping[str, object],
        avalan_extension["conversation"],
    )
    assert "continuation_envelope" in conversation_extension
    head = await heads.inspect(
        str(conversation.authority_digest(scope)),
        conversation.NamedHeadId("head-stateless"),
    )
    assert head is not None and head[0] == 1

    second = await service.prepare_turn(
        authority=scope,
        input_text="second",
        request_fingerprint=sha256(b"second").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key="second-key",
        continuation_value=committed.continuation,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.NAMED_HEAD,
            head_id=conversation.NamedHeadId("head-stateless"),
            expected_head_revision=conversation.NamedHeadRevision(1),
        ),
    )
    await service.finalize(
        second,
        request_bytes=20,
        response_bytes=40,
        input_items=1,
        output_items=1,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await service.prepare_turn(
            authority=scope,
            input_text="stale",
            request_fingerprint=sha256(b"stale").hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key="stale-key",
            continuation_value=committed.continuation,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.NAMED_HEAD,
                head_id=conversation.NamedHeadId("head-stateless"),
                expected_head_revision=conversation.NamedHeadRevision(1),
            ),
        )


async def test_abort_and_standard_replay_leave_only_bounded_audit() -> None:
    configuration, resolver, _, audit, _ = _configuration()
    service = StatelessResponsesService(configuration)
    prepared = await service.prepare_turn(
        authority=_authority(),
        input_text="cancelled",
        request_fingerprint=sha256(b"cancelled").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=True,
        idempotency_key=None,
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
    )
    await service.abort(
        prepared,
        outcome=StatelessResponseOutcome.DISCONNECTED,
        request_bytes=10,
        input_items=1,
    )
    await service.record_standard_terminal(
        _authority(),
        outcome=StatelessResponseOutcome.COMPLETED,
        request_bytes=11,
        response_bytes=12,
        input_items=2,
        output_items=1,
    )
    assert resolver.stores[-1].closed
    assert [record.operation for record in audit.records] == [
        "response_create",
        "canonical_replay",
    ]
    assert all(
        record.reconstructable_state_count == 0 for record in audit.records
    )


async def test_compact_route_is_canonical_and_tool_free() -> None:
    configuration, _, compact_resolver, audit, _ = _configuration()
    app = FastAPI()
    app.state.stateless_responses_service = StatelessResponsesService(
        configuration
    )
    app.include_router(responses.router)
    client = TestClient(app)
    payload = {
        "model": "server-model",
        "input": [
            {
                "content": [
                    {
                        "text": "complete caller context",
                        "type": "input_text",
                    }
                ],
                "role": "user",
                "type": "message",
            }
        ],
        "instructions": "compact natively",
    }
    response = client.post("/responses/compact", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response.compaction"
    assert body["output"][0]["content"][0]["text"] == "provider compact output"
    assert compact_resolver.plans[0].instructions == "compact natively"
    assert compact_resolver.plans[0].input[0]["type"] == "message"
    assert audit.records[-1].operation == "response_compact"

    for invalid in (
        {"model": "server-model", "input": []},
        {
            "model": "server-model",
            "input": [{"type": "item_reference", "id": "x"}],
        },
        {"model": "server-model", "input": [{"type": "unknown"}]},
        {
            "model": "server-model",
            "input": [
                {
                    "arguments": "{}",
                    "call_id": "call-one",
                    "name": "lookup",
                    "status": "completed",
                    "type": "function_call",
                }
            ],
        },
        {
            **payload,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "mode": "caller_held",
                        "operation": "branch",
                        "branch_id": "branch-invalid",
                        "continuation_envelope": "avl_ce1.invalid",
                    },
                }
            },
        },
        {"model": "other-model", "input": payload["input"]},
    ):
        rejected = client.post("/responses/compact", json=invalid)
        assert rejected.status_code == 400
    assert len(compact_resolver.plans) == 1


class _RouteOrchestrator(Orchestrator):
    def __init__(self) -> None:
        self._model_ids = {"server-model"}
        self.synced = 0

    async def __call__(
        self,
        messages: object,
        settings: object = None,
        **kwargs: object,
    ) -> TextGenerationResponse:
        del kwargs
        stream = bool(getattr(settings, "use_async_generator", False))
        disconnect = "disconnect" in str(messages)
        return TextGenerationResponse(
            (
                _slow_stream_items
                if stream and disconnect
                else _stream_items if stream else lambda **_: "safe terminal"
            ),
            logger=getLogger(__name__),
            use_async_generator=stream,
            inputs={"input_ids": [[1, 2, 3]]},
        )

    async def sync_messages(self, response: object) -> None:
        del response
        self.synced += 1


@dataclass(slots=True)
class _ControlledResponse:
    items: tuple[CanonicalStreamItem, ...]
    source_error: BaseException | None = None
    close_error: Exception | None = None
    input_token_count: int = 3
    output_token_count: int = 2
    close_count: int = 0
    cancel_count: int = 0

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        async def generate() -> AsyncIterator[CanonicalStreamItem]:
            for item in self.items:
                yield item
            if self.source_error is not None:
                raise self.source_error

        return generate()

    async def aclose(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error

    async def cancel(self) -> None:
        self.cancel_count += 1


class _DetachedInteractionRun:
    handling = ServerInteractionHandling.DETACHED
    runtime = object()

    async def extension_events(
        self,
        projection: object,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        del projection
        return ()


def _terminal_items(
    outcome: StreamTerminalOutcome,
    *,
    close: bool = True,
) -> tuple[CanonicalStreamItem, ...]:
    common = {
        "stream_session_id": "stateless-controlled-stream",
        "run_id": "stateless-controlled-run",
        "turn_id": "stateless-controlled-turn",
    }
    items = [
        CanonicalStreamItem(
            **common,
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
    ]
    sequence = 1
    if outcome is StreamTerminalOutcome.COMPLETED:
        items.extend(
            (
                CanonicalStreamItem(
                    **common,
                    sequence=sequence,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="controlled answer",
                ),
                CanonicalStreamItem(
                    **common,
                    sequence=sequence + 1,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
            )
        )
        sequence += 2
    terminal_kind = {
        StreamTerminalOutcome.COMPLETED: StreamItemKind.STREAM_COMPLETED,
        StreamTerminalOutcome.ERRORED: StreamItemKind.STREAM_ERRORED,
        StreamTerminalOutcome.CANCELLED: StreamItemKind.STREAM_CANCELLED,
    }[outcome]
    items.append(
        CanonicalStreamItem(
            **common,
            sequence=sequence,
            kind=terminal_kind,
            channel=StreamChannel.CONTROL,
            data=(
                {"message": "controlled failure"}
                if outcome is StreamTerminalOutcome.ERRORED
                else None
            ),
            usage=(
                {"input_tokens": 3, "output_tokens": 2}
                if outcome is StreamTerminalOutcome.COMPLETED
                else None
            ),
            terminal_outcome=outcome,
        )
    )
    if close:
        items.append(
            CanonicalStreamItem(
                **common,
                sequence=sequence + 1,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
            )
        )
    return tuple(items)


def _interaction_pending_items() -> tuple[CanonicalStreamItem, ...]:
    common = {
        "stream_session_id": "stateless-interaction-stream",
        "run_id": "stateless-interaction-run",
        "turn_id": "stateless-interaction-turn",
    }
    correlation = StreamItemCorrelation(
        request_id="request-stateless",
        continuation_id="continuation-stateless",
        agent_id="agent-stateless",
        branch_id="branch-stateless",
    )
    return (
        CanonicalStreamItem(
            **common,
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        ),
        CanonicalStreamItem(
            **common,
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="preface before interaction",
        ),
        CanonicalStreamItem(
            **common,
            sequence=2,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        ),
        CanonicalStreamItem(
            **common,
            sequence=3,
            kind=StreamItemKind.INTERACTION_CREATED,
            channel=StreamChannel.INTERACTION,
            correlation=correlation,
        ),
        CanonicalStreamItem(
            **common,
            sequence=4,
            kind=StreamItemKind.INTERACTION_PENDING,
            channel=StreamChannel.INTERACTION,
            correlation=correlation,
        ),
    )


def _caller_held_request(*, stream: bool = False) -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "input": "controlled caller state",
            "model": "server-model",
            "store": False,
            "stream": stream,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "mode": "caller_held",
                    },
                }
            },
        }
    )


def _standard_request(*, stream: bool = False) -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "input": "controlled standard request",
            "model": "server-model",
            "store": False,
            "stream": stream,
        }
    )


def _http_request() -> Request:
    return Request({"type": "http", "headers": []})


async def _invoke_controlled_response(
    request: ResponsesRequest,
    *,
    service: StatelessResponsesService | None,
    response: _ControlledResponse | None = None,
    dispatch_error: BaseException | None = None,
    interaction_run: _DetachedInteractionRun | None = None,
    interaction_error: BaseException | None = None,
) -> object:
    async def controlled_orchestrate(
        request_value: object,
        logger: object,
        orchestrator: object,
        **kwargs: object,
    ) -> tuple[TextGenerationResponse, UUID, int]:
        del request_value, logger, orchestrator, kwargs
        if dispatch_error is not None:
            raise dispatch_error
        assert response is not None
        return (
            cast(TextGenerationResponse, response),
            UUID("00000000-0000-0000-0000-000000000020"),
            2,
        )

    async def controlled_interaction_run(
        *args: object,
        **kwargs: object,
    ) -> _DetachedInteractionRun | None:
        del args, kwargs
        if interaction_error is not None:
            raise interaction_error
        return interaction_run

    with (
        patch.object(
            responses,
            "orchestrate",
            new=controlled_orchestrate,
        ),
        patch.object(
            responses,
            "prepare_openai_interaction_run",
            new=controlled_interaction_run,
        ),
    ):
        return await responses.create_response(
            request,
            logger=getLogger(__name__),
            orchestrator=_RouteOrchestrator(),
            output_redaction_settings=responses.ServerOutputRedactionSettings(),
            served_responses_service=None,
            stateless_responses_service=service,
            http_request=_http_request(),
        )


async def _consume_stream(response: object) -> str:
    assert isinstance(response, responses.StreamingResponse)
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


async def _stream_items(
    **_: object,
) -> AsyncIterator[CanonicalStreamItem]:
    yield CanonicalStreamItem(
        stream_session_id="stateless-stream",
        run_id="stateless-run",
        turn_id="stateless-turn",
        sequence=0,
        kind=StreamItemKind.STREAM_STARTED,
        channel=StreamChannel.CONTROL,
    )
    yield CanonicalStreamItem(
        stream_session_id="stateless-stream",
        run_id="stateless-run",
        turn_id="stateless-turn",
        sequence=1,
        kind=StreamItemKind.ANSWER_DELTA,
        channel=StreamChannel.ANSWER,
        text_delta="safe terminal",
    )
    yield CanonicalStreamItem(
        stream_session_id="stateless-stream",
        run_id="stateless-run",
        turn_id="stateless-turn",
        sequence=2,
        kind=StreamItemKind.ANSWER_DONE,
        channel=StreamChannel.ANSWER,
    )
    yield CanonicalStreamItem(
        stream_session_id="stateless-stream",
        run_id="stateless-run",
        turn_id="stateless-turn",
        sequence=3,
        kind=StreamItemKind.STREAM_COMPLETED,
        channel=StreamChannel.CONTROL,
        usage={"input_tokens": 3, "output_tokens": 2},
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
    )
    yield CanonicalStreamItem(
        stream_session_id="stateless-stream",
        run_id="stateless-run",
        turn_id="stateless-turn",
        sequence=4,
        kind=StreamItemKind.STREAM_CLOSED,
        channel=StreamChannel.CONTROL,
    )


async def _slow_stream_items(
    **_: object,
) -> AsyncIterator[CanonicalStreamItem]:
    yield CanonicalStreamItem(
        stream_session_id="stateless-disconnect-stream",
        run_id="stateless-disconnect-run",
        turn_id="stateless-disconnect-turn",
        sequence=0,
        kind=StreamItemKind.STREAM_STARTED,
        channel=StreamChannel.CONTROL,
    )
    await sleep(10)
    yield CanonicalStreamItem(
        stream_session_id="stateless-disconnect-stream",
        run_id="stateless-disconnect-run",
        turn_id="stateless-disconnect-turn",
        sequence=1,
        kind=StreamItemKind.STREAM_COMPLETED,
        channel=StreamChannel.CONTROL,
        usage={"input_tokens": 1, "output_tokens": 0},
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
    )


async def _route_orchestrate(
    request: object,
    logger: object,
    orchestrator: object,
    **kwargs: object,
) -> tuple[TextGenerationResponse, UUID, int]:
    del logger, orchestrator, kwargs
    stream = bool(getattr(request, "stream", False))
    response = TextGenerationResponse(
        _stream_items if stream else lambda **_: "safe terminal",
        logger=getLogger(__name__),
        use_async_generator=stream,
        inputs={"input_ids": [[1, 2, 3]]},
    )
    return response, UUID("00000000-0000-0000-0000-000000000010"), 1


async def test_router_control_and_authentication_boundaries_are_closed() -> (
    None
):
    ordinary = responses._request_continuation_advance(_standard_request())
    assert ordinary.mode is conversation.ParentAdvanceMode.ORDINARY_CHILD
    branch_request = ResponsesRequest.model_validate(
        {
            "input": "explicit branch",
            "model": "server-model",
            "store": False,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "mode": "caller_held",
                        "operation": "branch",
                        "branch_id": "branch-explicit",
                        "continuation_envelope": "avl_ce1.valid",
                    },
                }
            },
        }
    )
    branch = responses._request_continuation_advance(branch_request)
    assert branch.mode is conversation.ParentAdvanceMode.EXPLICIT_BRANCH
    assert branch.branch_id == conversation.ConversationBranchId(
        "branch-explicit"
    )
    named_request = ResponsesRequest.model_validate(
        {
            "input": "new named head",
            "model": "server-model",
            "store": False,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "mode": "caller_held",
                        "operation": "named_head",
                        "head_id": "head-root",
                        "expected_head_revision": 0,
                    },
                }
            },
        }
    )
    named = responses._request_continuation_advance(named_request)
    assert named.mode is conversation.ParentAdvanceMode.NAMED_HEAD
    assert named.head_id == conversation.NamedHeadId("head-root")
    assert named.expected_head_revision == conversation.NamedHeadRevision(0)

    not_configured = await _invoke_controlled_response(
        _caller_held_request(),
        service=None,
    )
    assert isinstance(not_configured, responses.JSONResponse)
    assert not_configured.status_code == 400

    configuration, _, _, _, _ = _configuration()
    service = StatelessResponsesService(configuration)
    missing_request = await responses.create_response(
        _caller_held_request(),
        logger=getLogger(__name__),
        orchestrator=_RouteOrchestrator(),
        output_redaction_settings=responses.ServerOutputRedactionSettings(),
        served_responses_service=None,
        stateless_responses_service=service,
        http_request=cast(Request, object()),
    )
    assert isinstance(missing_request, responses.JSONResponse)
    assert missing_request.status_code == 401

    async def reject_authority(
        request: Request,
    ) -> conversation.AuthorityScope:
        del request
        raise RuntimeError("authority unavailable")

    rejecting_service = StatelessResponsesService(
        replace(configuration, authority_resolver=reject_authority)
    )
    rejected = await _invoke_controlled_response(
        _standard_request(),
        service=rejecting_service,
    )
    assert isinstance(rejected, responses.JSONResponse)
    assert rejected.status_code == 401

    tagged_message = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "compact context"}],
    }
    compact_request = ResponsesCompactRequest.model_validate(
        {
            "model": "server-model",
            "input": [tagged_message],
        }
    )
    compact_not_configured = await responses.compact_response(
        compact_request,
        _http_request(),
        None,
    )
    assert isinstance(compact_not_configured, responses.JSONResponse)
    assert compact_not_configured.status_code == 400
    compact_unauthorized = await responses.compact_response(
        compact_request,
        _http_request(),
        rejecting_service,
    )
    assert isinstance(compact_unauthorized, responses.JSONResponse)
    assert compact_unauthorized.status_code == 401

    compact_with_extension = ResponsesCompactRequest.model_validate(
        {
            "model": "server-model",
            "input": [tagged_message],
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "mode": "caller_held",
                        "lane_id": configuration.policy.compact_lane_id,
                    },
                }
            },
        }
    )
    compact_result = await responses.compact_response(
        compact_with_extension,
        _http_request(),
        service,
    )
    assert isinstance(compact_result, dict)
    assert compact_result["object"] == "response.compaction"


async def test_router_dispatch_and_detached_interaction_fail_closed() -> None:
    configuration, resolver, _, audit, _ = _configuration()
    dispatch_failure = await _invoke_controlled_response(
        _caller_held_request(),
        service=StatelessResponsesService(configuration),
        dispatch_error=conversation.ConversationStorageError(),
    )
    assert isinstance(dispatch_failure, responses.JSONResponse)
    assert dispatch_failure.status_code == 500
    assert resolver.stores[-1].closed
    assert audit.records[-1].reconstructable_state_count == 0

    (
        interaction_configuration,
        interaction_resolver,
        _,
        interaction_audit,
        _,
    ) = _configuration()
    interaction_failure = await _invoke_controlled_response(
        _caller_held_request(),
        service=StatelessResponsesService(interaction_configuration),
        interaction_error=responses.ServerInteractionHTTPError.forbidden(),
    )
    assert isinstance(interaction_failure, responses.JSONResponse)
    assert interaction_failure.status_code == 403
    assert interaction_resolver.stores[-1].closed
    assert interaction_audit.records[-1].reconstructable_state_count == 0

    reasoning_configuration, reasoning_resolver, _, reasoning_audit, _ = (
        _configuration()
    )
    with pytest.raises(responses.HTTPException) as reasoning_failure:
        await _invoke_controlled_response(
            _caller_held_request(),
            service=StatelessResponsesService(reasoning_configuration),
            dispatch_error=responses.ReasoningSummaryCapabilityError(
                provider="controlled-provider",
                requested_mode=ReasoningSummaryMode.AUTO,
            ),
        )
    assert reasoning_failure.value.status_code == 400
    assert reasoning_resolver.stores[-1].closed
    assert reasoning_audit.records[-1].reconstructable_state_count == 0

    runtime_configuration, runtime_resolver, _, runtime_audit, _ = (
        _configuration()
    )
    with pytest.raises(RuntimeError, match="arbitrary dispatch failure"):
        await _invoke_controlled_response(
            _caller_held_request(),
            service=StatelessResponsesService(runtime_configuration),
            dispatch_error=RuntimeError("arbitrary dispatch failure"),
        )
    assert runtime_resolver.stores[-1].closed
    assert runtime_audit.records[-1].reconstructable_state_count == 0

    for stream in (False, True):
        interaction_configuration, interaction_resolver, _, _, _ = (
            _configuration()
        )
        interaction_response = await _invoke_controlled_response(
            _caller_held_request(stream=stream),
            service=StatelessResponsesService(interaction_configuration),
            response=_ControlledResponse(_interaction_pending_items()),
            interaction_run=_DetachedInteractionRun(),
        )
        if stream:
            stream_body = await _consume_stream(interaction_response)
            assert "conversation_capability_unsupported" in stream_body
        else:
            assert isinstance(interaction_response, responses.JSONResponse)
            assert interaction_response.status_code == 400
        assert interaction_resolver.stores[-1].closed


async def test_route_ownership_guards_reject_invalid_lifecycle() -> None:
    configuration, resolver, _, _, _ = _configuration()
    service = StatelessResponsesService(configuration)
    prepared = await service.prepare_turn(
        authority=_authority(),
        input_text="ownership guards",
        request_fingerprint=sha256(b"ownership-guards").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key=None,
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
    )
    ownership = responses._StatelessResponseOwnership()
    with pytest.raises(conversation.ConversationValidationError):
        ownership.acquire(
            service,
            prepared,
            request_bytes=1,
            input_items=-1,
        )
    ownership.acquire(
        service,
        prepared,
        request_bytes=1,
        input_items=1,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await ownership.abort(StatelessResponseOutcome.COMPLETED)
    assert not ownership.disposed
    assert await ownership.abort(StatelessResponseOutcome.FAILED)
    with pytest.raises(conversation.ConversationValidationError):
        await ownership.finalize(response_bytes=1, output_items=1)
    assert resolver.stores[-1].closed


async def test_streaming_stateless_failure_lifecycle_matrix() -> None:
    source_configuration, source_resolver, _, _, _ = _configuration()
    source_after_terminal = await _invoke_controlled_response(
        _caller_held_request(stream=True),
        service=StatelessResponsesService(source_configuration),
        response=_ControlledResponse(
            _terminal_items(StreamTerminalOutcome.COMPLETED, close=False),
            source_error=RuntimeError("source failed after terminal"),
        ),
    )
    with pytest.raises(responses._ResponsesSourceAfterTerminalError):
        await _consume_stream(source_after_terminal)
    assert source_resolver.stores[-1].closed

    gap_configuration, gap_resolver, _, _, _ = _configuration()
    original_capture = responses._DetachedResponsesProjection._capture

    def capture_with_gap(
        adapter: responses._DetachedResponsesProjection,
        event: responses._ResponsesSSEEvent,
    ) -> None:
        original_capture(adapter, event)
        if 0 in adapter.indexed_output:
            adapter.indexed_output[1] = adapter.indexed_output.pop(0)

    with patch.object(
        responses._DetachedResponsesProjection,
        "_capture",
        new=capture_with_gap,
    ):
        gap_response = await _invoke_controlled_response(
            _caller_held_request(stream=True),
            service=StatelessResponsesService(gap_configuration),
            response=_ControlledResponse(
                _terminal_items(StreamTerminalOutcome.COMPLETED)
            ),
        )
        with pytest.raises(StreamValidationError):
            await _consume_stream(gap_response)
    assert gap_resolver.stores[-1].closed

    standard_configuration, _, _, standard_audit, _ = _configuration()
    standard_response = await _invoke_controlled_response(
        _standard_request(stream=True),
        service=StatelessResponsesService(standard_configuration),
        response=_ControlledResponse(
            _terminal_items(StreamTerminalOutcome.COMPLETED)
        ),
    )
    assert "response.completed" in await _consume_stream(standard_response)
    assert standard_audit.records[-1].operation == "canonical_replay"

    failed_configuration, failed_resolver, _, _, _ = _configuration()
    failed_response = await _invoke_controlled_response(
        _caller_held_request(stream=True),
        service=StatelessResponsesService(failed_configuration),
        response=_ControlledResponse(
            _terminal_items(StreamTerminalOutcome.ERRORED)
        ),
    )
    assert "response.failed" in await _consume_stream(failed_response)
    assert failed_resolver.stores[-1].closed

    commit_configuration, commit_resolver, _, _, _ = _configuration()
    commit_service = StatelessResponsesService(commit_configuration)

    async def fail_finalize(
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        raise conversation.ConversationStorageError()

    with patch.object(commit_service, "finalize", new=fail_finalize):
        commit_response = await _invoke_controlled_response(
            _caller_held_request(stream=True),
            service=commit_service,
            response=_ControlledResponse(
                _terminal_items(StreamTerminalOutcome.COMPLETED)
            ),
        )
        commit_body = await _consume_stream(commit_response)
    assert "conversation_state_commit_failed" in commit_body
    assert commit_resolver.stores[-1].closed

    abort_configuration, abort_resolver, _, _, _ = _configuration()
    abort_service = StatelessResponsesService(abort_configuration)

    async def fail_abort(
        *args: object,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        raise conversation.ConversationStorageError()

    with patch.object(abort_service, "abort", new=fail_abort):
        missing_terminal = await _invoke_controlled_response(
            _caller_held_request(stream=True),
            service=abort_service,
            response=_ControlledResponse(
                (_terminal_items(StreamTerminalOutcome.COMPLETED)[0],)
            ),
        )
        with pytest.raises(StreamValidationError):
            await _consume_stream(missing_terminal)
    await abort_resolver.stores[-1].close()


async def test_non_stream_stateless_failure_lifecycle_matrix() -> None:
    cancelled_configuration, cancelled_resolver, _, _, _ = _configuration()
    with pytest.raises(CancelledError):
        await _invoke_controlled_response(
            _caller_held_request(),
            service=StatelessResponsesService(cancelled_configuration),
            response=_ControlledResponse(
                (_terminal_items(StreamTerminalOutcome.COMPLETED)[0],),
                source_error=CancelledError(),
            ),
        )
    assert cancelled_resolver.stores[-1].closed

    projection_configuration, projection_resolver, _, _, _ = _configuration()

    def observe_failure(
        adapter: responses._DetachedResponsesProjection,
        projection: object,
    ) -> None:
        del projection
        adapter.projector.failure = {
            "error": {
                "type": "server_error",
                "code": "controlled_projection_failure",
                "message": "controlled projection failure",
            }
        }

    with patch.object(
        responses._DetachedResponsesProjection,
        "observe_json",
        new=observe_failure,
    ):
        projection_failure = await _invoke_controlled_response(
            _caller_held_request(),
            service=StatelessResponsesService(projection_configuration),
            response=_ControlledResponse(
                _terminal_items(StreamTerminalOutcome.COMPLETED)
            ),
        )
    assert isinstance(projection_failure, responses.JSONResponse)
    assert projection_failure.status_code == 500
    assert projection_resolver.stores[-1].closed

    cleanup_configuration, cleanup_resolver, _, _, _ = _configuration()
    with pytest.raises(responses._ResponsesCleanupError):
        await _invoke_controlled_response(
            _caller_held_request(),
            service=StatelessResponsesService(cleanup_configuration),
            response=_ControlledResponse(
                _terminal_items(StreamTerminalOutcome.COMPLETED),
                close_error=RuntimeError("cleanup failed"),
            ),
        )
    assert cleanup_resolver.stores[-1].closed

    missing_configuration, missing_resolver, _, _, _ = _configuration()
    missing_error = RuntimeError("source failed before terminal")
    with pytest.raises(RuntimeError, match="source failed before terminal"):
        await _invoke_controlled_response(
            _caller_held_request(),
            service=StatelessResponsesService(missing_configuration),
            response=_ControlledResponse(
                (_terminal_items(StreamTerminalOutcome.COMPLETED)[0],),
                source_error=missing_error,
            ),
        )
    assert missing_resolver.stores[-1].closed

    source_configuration, source_resolver, _, _, _ = _configuration()
    with pytest.raises(responses._ResponsesSourceAfterTerminalError):
        await _invoke_controlled_response(
            _caller_held_request(),
            service=StatelessResponsesService(source_configuration),
            response=_ControlledResponse(
                _terminal_items(StreamTerminalOutcome.COMPLETED, close=False),
                source_error=RuntimeError("source failed after terminal"),
            ),
        )
    assert source_resolver.stores[-1].closed

    commit_configuration, commit_resolver, _, _, _ = _configuration()
    commit_service = StatelessResponsesService(commit_configuration)

    async def fail_finalize(
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        await commit_resolver.stores[-1].close()
        raise conversation.ConversationStorageError()

    with patch.object(commit_service, "finalize", new=fail_finalize):
        commit_failure = await _invoke_controlled_response(
            _caller_held_request(),
            service=commit_service,
            response=_ControlledResponse(
                _terminal_items(StreamTerminalOutcome.COMPLETED)
            ),
        )
    assert isinstance(commit_failure, responses.JSONResponse)
    assert commit_failure.status_code == 500
    assert commit_resolver.stores[-1].closed

    failed_configuration, failed_resolver, _, _, _ = _configuration()
    failed_terminal = await _invoke_controlled_response(
        _caller_held_request(),
        service=StatelessResponsesService(failed_configuration),
        response=_ControlledResponse(
            _terminal_items(StreamTerminalOutcome.ERRORED)
        ),
    )
    assert isinstance(failed_terminal, dict)
    assert failed_terminal["status"] == "failed"
    assert failed_resolver.stores[-1].closed

    standard_configuration, _, _, standard_audit, _ = _configuration()
    standard_failed = await _invoke_controlled_response(
        _standard_request(),
        service=StatelessResponsesService(standard_configuration),
        response=_ControlledResponse(
            _terminal_items(StreamTerminalOutcome.ERRORED)
        ),
    )
    assert isinstance(standard_failed, dict)
    assert standard_failed["status"] == "failed"
    assert standard_audit.records[-1].operation == "canonical_replay"
    assert (
        standard_audit.records[-1].outcome is StatelessResponseOutcome.FAILED
    )

    completed_configuration, _, _, completed_audit, _ = _configuration()
    standard_completed = await _invoke_controlled_response(
        _standard_request(),
        service=StatelessResponsesService(completed_configuration),
        response=_ControlledResponse(
            _terminal_items(StreamTerminalOutcome.COMPLETED)
        ),
    )
    assert isinstance(standard_completed, dict)
    assert standard_completed["status"] == "completed"
    assert completed_audit.records[-1].operation == "canonical_replay"
    assert (
        completed_audit.records[-1].outcome
        is StatelessResponseOutcome.COMPLETED
    )


async def test_create_response_logs_failed_owned_cleanup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    for cancelled in (True, False):
        configuration, resolver, _, _, _ = _configuration()
        service = StatelessResponsesService(configuration)
        prepared = await service.prepare_turn(
            authority=_authority(),
            input_text=f"cleanup failure {cancelled}",
            request_fingerprint=sha256(
                f"cleanup-failure-{cancelled}".encode()
            ).hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=None,
            continuation_value=None,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
            ),
        )
        original_abort = service.abort

        async def fail_abort(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("controlled cleanup failure")

        async def controlled_create(
            request: ResponsesRequest,
            ownership: responses._StatelessResponseOwnership,
            **kwargs: object,
        ) -> dict[str, Any]:
            del request, kwargs
            ownership.acquire(
                service,
                prepared,
                request_bytes=1,
                input_items=1,
            )
            if cancelled:
                raise CancelledError()
            return {}

        caplog.clear()
        with (
            patch.object(service, "abort", new=fail_abort),
            patch.object(
                responses,
                "_create_response_impl",
                new=controlled_create,
            ),
            caplog.at_level("ERROR"),
        ):
            if cancelled:
                with pytest.raises(CancelledError):
                    await responses.create_response(
                        _caller_held_request(),
                        logger=getLogger(__name__),
                        orchestrator=_RouteOrchestrator(),
                        output_redaction_settings=(
                            responses.ServerOutputRedactionSettings()
                        ),
                        served_responses_service=None,
                        stateless_responses_service=service,
                        http_request=_http_request(),
                    )
            else:
                assert (
                    await responses.create_response(
                        _caller_held_request(),
                        logger=getLogger(__name__),
                        orchestrator=_RouteOrchestrator(),
                        output_redaction_settings=(
                            responses.ServerOutputRedactionSettings()
                        ),
                        served_responses_service=None,
                        stateless_responses_service=service,
                        http_request=_http_request(),
                    )
                    == {}
                )
        assert (
            caplog.messages.count(responses._RESPONSES_CLEANUP_ERROR_MESSAGE)
            == 1
        )
        await original_abort(
            prepared,
            outcome=StatelessResponseOutcome.FAILED,
            request_bytes=1,
            input_items=1,
        )
        assert resolver.stores[-1].closed


async def test_http_terminal_extension_and_non_retention_parity() -> None:
    configuration, resolver, _, audit, _ = _configuration()
    app = FastAPI()
    orchestrator = _RouteOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.logger = getLogger(__name__)
    app.state.stateless_responses_service = StatelessResponsesService(
        configuration
    )
    app.include_router(responses.router)
    client = TestClient(app)
    base = {
        "input": "hello",
        "model": "server-model",
        "store": False,
        "extensions": {
            "avalan": {
                "version": "1",
                "conversation": {"version": "1", "mode": "caller_held"},
            }
        },
    }
    with patch(
        "avalan.server.routers.responses.orchestrate",
        new=_route_orchestrate,
    ):
        non_stream = client.post("/responses", json=base)
        stream = client.post(
            "/responses",
            json={**base, "stream": True},
        )
    assert non_stream.status_code == 200, non_stream.text
    token = non_stream.json()["extensions"]["avalan"]["conversation"][
        "continuation_envelope"
    ]
    assert token.startswith(conversation.CONTINUATION_ENVELOPE_PREFIX)
    assert stream.status_code == 200
    terminal_lines = [
        line.removeprefix("data: ")
        for line in stream.text.splitlines()
        if line.startswith("data: ") and "response.completed" in line
    ]
    assert len(terminal_lines) == 1
    assert "continuation_envelope" in terminal_lines[0]
    assert all(store.closed for store in resolver.stores)
    assert all(
        record.reconstructable_state_count == 0 for record in audit.records
    )
    assert orchestrator.synced == 0


async def test_http_accepts_caller_state_only_in_strict_body_field() -> None:
    configuration, resolver, _, _, _ = _configuration()
    app = FastAPI()
    app.state.orchestrator = _RouteOrchestrator()
    app.state.logger = getLogger(__name__)
    app.state.stateless_responses_service = StatelessResponsesService(
        configuration
    )
    app.include_router(responses.router)
    client = TestClient(app)
    base = {
        "input": "strict boundary",
        "model": "server-model",
        "store": False,
        "extensions": {
            "avalan": {
                "version": "1",
                "conversation": {"version": "1", "mode": "caller_held"},
            }
        },
    }
    with patch(
        "avalan.server.routers.responses.orchestrate",
        new=_route_orchestrate,
    ):
        created = client.post("/responses", json=base)
        assert created.status_code == 200
        token = created.json()["extensions"]["avalan"]["conversation"][
            "continuation_envelope"
        ]
        before = len(resolver.plans)
        header_only = client.post(
            "/responses?continuation_envelope=" + token,
            headers={"x-continuation-envelope": token},
            json=base,
        )
        assert header_only.status_code == 200
        assert resolver.plans[-1].opened_parent is None
        assert len(resolver.plans) == before + 1

        tampered = _tamper_authenticated_ciphertext(token)
        tamper_body = {
            **base,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "mode": "caller_held",
                        "continuation_envelope": tampered,
                    },
                }
            },
        }
        rejected = client.post("/responses", json=tamper_body)
        bare_parent = client.post(
            "/responses",
            json={
                "input": "bare parent",
                "model": "server-model",
                "previous_response_id": created.json()["id"],
                "store": False,
            },
        )
        lane_on_create = client.post(
            "/responses",
            json={
                **base,
                "extensions": {
                    "avalan": {
                        "version": "1",
                        "conversation": {
                            "version": "1",
                            "mode": "caller_held",
                            "lane_id": "lane-not-create",
                        },
                    }
                },
            },
        )
    assert rejected.status_code == 404
    assert bare_parent.status_code == 400
    assert lane_on_create.status_code == 400
    assert len(resolver.plans) == before + 1


async def test_http_envelope_key_oracle_is_closed_after_authentication() -> (
    None
):
    configuration, resolver, _, _, _ = _configuration()
    app = FastAPI()
    app.state.orchestrator = _RouteOrchestrator()
    app.state.logger = getLogger(__name__)
    app.state.stateless_responses_service = StatelessResponsesService(
        configuration
    )
    app.include_router(responses.router)
    client = TestClient(app)
    base = {
        "input": "key lifecycle",
        "model": "server-model",
        "store": False,
        "extensions": {
            "avalan": {
                "version": "1",
                "conversation": {"version": "1", "mode": "caller_held"},
            }
        },
    }

    def with_token(token: str) -> dict[str, object]:
        return {
            **base,
            "extensions": {
                "avalan": {
                    "version": "1",
                    "conversation": {
                        "version": "1",
                        "mode": "caller_held",
                        "continuation_envelope": token,
                    },
                }
            },
        }

    with patch(
        "avalan.server.routers.responses.orchestrate",
        new=_route_orchestrate,
    ):
        created = client.post("/responses", json=base)
        assert created.status_code == 200
        token = cast(
            str,
            created.json()["extensions"]["avalan"]["conversation"][
                "continuation_envelope"
            ],
        )
        before = len(resolver.plans)
        scope_digest = conversation.authority_digest(_authority())
        key_resolver = cast(
            conversation.InMemoryContinuationEnvelopeKeyResolver,
            configuration.envelope_codec.key_resolver,
        )
        active = conversation.ContinuationEnvelopeKey(
            key_id="key-stateless",
            revision=1,
            status=conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
            key_bytes=b"s" * 32,
        )
        retired_decoy = conversation.ContinuationEnvelopeKey(
            key_id="key-retired",
            revision=9,
            status=conversation.ContinuationEnvelopeKeyStatus.RETIRED,
            key_bytes=b"r" * 32,
        )
        await key_resolver.replace_keys(
            scope_digest,
            (active, retired_decoy),
        )
        oracle_attempts = (
            _tamper_authenticated_ciphertext(token),
            _rewrite_envelope_key(
                token,
                key_id="key-retired",
                revision=9,
            ),
            _rewrite_envelope_key(
                token,
                key_id="key-unknown",
                revision=7,
            ),
        )
        oracle_responses = [
            client.post("/responses", json=with_token(value))
            for value in oracle_attempts
        ]
        assert {response.status_code for response in oracle_responses} == {404}
        assert len({response.text for response in oracle_responses}) == 1

        next_active = conversation.ContinuationEnvelopeKey(
            key_id="key-next",
            revision=2,
            status=conversation.ContinuationEnvelopeKeyStatus.ACTIVE,
            key_bytes=b"n" * 32,
        )
        await key_resolver.replace_keys(
            scope_digest,
            (
                replace(
                    active,
                    status=conversation.ContinuationEnvelopeKeyStatus.RETIRED,
                ),
                next_active,
            ),
        )
        retired = client.post("/responses", json=with_token(token))
        assert retired.status_code == 500
        assert retired.json()["error"]["code"] == "conversation_key_retired"

        await key_resolver.replace_keys(
            scope_digest,
            (
                replace(
                    active,
                    status=(
                        conversation.ContinuationEnvelopeKeyStatus.COMPROMISED
                    ),
                ),
                next_active,
            ),
        )
        compromised = client.post("/responses", json=with_token(token))
        assert compromised.status_code == 500
        assert (
            compromised.json()["error"]["code"]
            == "conversation_key_compromised"
        )

        await key_resolver.replace_keys(scope_digest, (active,))
        app.state.stateless_responses_service = StatelessResponsesService(
            replace(
                configuration,
                clock=_FixedClock(_NOW + timedelta(hours=2)),
            )
        )
        expired = client.post("/responses", json=with_token(token))
        assert expired.status_code == 500
        assert expired.json()["error"]["code"] == "conversation_expired"

    assert len(resolver.plans) == before


async def test_named_head_race_has_exactly_one_reservation_winner() -> None:
    configuration, _, _, _, _ = _configuration()
    service = StatelessResponsesService(configuration)
    scope = _authority()
    root = await service.prepare_turn(
        authority=scope,
        input_text="root",
        request_fingerprint=sha256(b"race-root").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key="race-root",
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.NAMED_HEAD,
            head_id=conversation.NamedHeadId("head-race"),
            expected_head_revision=conversation.NamedHeadRevision(0),
        ),
    )
    committed = await service.finalize(
        root,
        request_bytes=1,
        response_bytes=1,
        input_items=1,
        output_items=1,
    )

    async def contend(label: str) -> PreparedStatelessResponse:
        return await service.prepare_turn(
            authority=scope,
            input_text=label,
            request_fingerprint=sha256(label.encode()).hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=label,
            continuation_value=committed.continuation,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.NAMED_HEAD,
                head_id=conversation.NamedHeadId("head-race"),
                expected_head_revision=conversation.NamedHeadRevision(1),
            ),
        )

    results = await gather(
        contend("contender-one"),
        contend("contender-two"),
        return_exceptions=True,
    )
    winners = tuple(
        result
        for result in results
        if isinstance(result, PreparedStatelessResponse)
    )
    losers = tuple(
        result
        for result in results
        if isinstance(result, conversation.ConversationConflictError)
    )
    assert len(winners) == len(losers) == 1
    await service.abort(
        winners[0],
        outcome=StatelessResponseOutcome.CANCELLED,
        request_bytes=1,
        input_items=1,
    )


async def test_stateless_value_and_configuration_validation_is_total() -> None:
    configuration, _, _, audit_hook, _ = _configuration()
    service = StatelessResponsesService(configuration)
    scope = _authority()
    scope_digest = str(conversation.authority_digest(scope))
    record = stateless_module.StatelessOperationalAuditRecord(
        authority_scope_digest=scope_digest,
        operation="validation",
        outcome=StatelessResponseOutcome.COMPLETED,
        request_bytes=1,
        response_bytes=1,
        input_items=1,
        output_items=1,
        provider_lanes=1,
        reconstructable_state_count=0,
    )
    with pytest.raises(conversation.ConversationValidationError):
        stateless_module.StatelessOperationalAuditRecord(
            authority_scope_digest=scope_digest,
            operation="validation",
            outcome=cast(StatelessResponseOutcome, "invalid"),
            request_bytes=1,
            response_bytes=1,
            input_items=1,
            output_items=1,
            provider_lanes=1,
            reconstructable_state_count=0,
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(record, request_bytes=-1)
    with pytest.raises(conversation.ConversationValidationError):
        await audit_hook.record(
            cast(stateless_module.StatelessOperationalAuditRecord, object())
        )
    noop = stateless_module._NoopStatelessResponsesAuditHook()
    with pytest.raises(conversation.ConversationValidationError):
        await noop.record(
            cast(stateless_module.StatelessOperationalAuditRecord, object())
        )

    reservation = stateless_module.StatelessNamedHeadReservation(
        authority_scope_digest=scope_digest,
        head_id=conversation.NamedHeadId("head-validation"),
        expected_revision=conversation.NamedHeadRevision(0),
        request_digest="a" * 64,
        lease_id="lease-validation",
    )
    assert "expected_revision=0" in repr(reservation)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            reservation,
            expected_revision=cast(conversation.NamedHeadRevision, -1),
        )
    ledger = InMemoryStatelessNamedHeadLedger()
    with pytest.raises(conversation.ConversationValidationError):
        await ledger.reserve(
            authority_scope_digest=scope_digest,
            head_id=conversation.NamedHeadId("head-validation"),
            expected_revision=cast(conversation.NamedHeadRevision, -1),
            request_digest="a" * 64,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await ledger.commit(
            cast(stateless_module.StatelessNamedHeadReservation, object()),
            continuation_digest="b" * 64,
        )
    with pytest.raises(conversation.ConversationConflictError):
        await ledger.commit(
            reservation,
            continuation_digest="b" * 64,
        )
    reserved = await ledger.reserve(
        authority_scope_digest=scope_digest,
        head_id=conversation.NamedHeadId("head-validation"),
        expected_revision=conversation.NamedHeadRevision(0),
        request_digest="a" * 64,
    )
    ledger._heads[(scope_digest, reserved.head_id)] = (1, "old")
    with pytest.raises(conversation.ConversationConflictError):
        await ledger.commit(reserved, continuation_digest="b" * 64)
    with pytest.raises(conversation.ConversationValidationError):
        await ledger.release(
            cast(stateless_module.StatelessNamedHeadReservation, object())
        )
    await ledger.release(reserved)

    prepared = await service.prepare_turn(
        authority=scope,
        input_text="value validation",
        request_fingerprint=sha256(b"value-validation").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key=None,
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
    )
    plan = prepared.plan
    assert "parent_present=False" in repr(plan)
    with pytest.raises(conversation.ConversationValidationError):
        replace(plan, authority=cast(conversation.AuthorityScope, object()))
    with pytest.raises(conversation.ConversationValidationError):
        replace(plan, request_fingerprint="not-a-digest")
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            prepared.prepared,
            turn=cast(conversation.AgentConversationTurn, object()),
        )
    sync_store = SimpleNamespace(
        durable=False,
        retrieve=lambda *_: None,
        load=lambda *_: None,
        close=lambda: None,
        audit_non_retention=lambda: None,
    )
    with pytest.raises(conversation.ConversationValidationError):
        PreparedStatelessResponsesTurn(
            turn=prepared.prepared.turn,
            store=cast(
                stateless_module.StatelessResponsesTransientStore,
                sync_store,
            ),
        )
    checkpoint = await prepared.prepared.store.load(
        prepared.prepared.turn.checkpoint_id,
        scope,
    )
    canonical_item = cast(
        Mapping[str, JsonValue],
        {"type": "message", "role": "user"},
    )
    compact_plan = StatelessCompactPlan(
        authority=scope,
        model="server-model",
        lane_id=str(prepared.prepared.turn.lanes[0].lane_id),
        input=(canonical_item,),
    )
    assert "item_count=1" in repr(compact_plan)
    compact_plan_cases = (
        {"authority": cast(conversation.AuthorityScope, object())},
        {"instructions": ""},
        {"input": ()},
        {
            "input": cast(
                tuple[Mapping[str, JsonValue], ...],
                ("not-a-mapping",),
            )
        },
        {
            "checkpoint": replace(
                checkpoint,
                authority=_authority("principal-other"),
            )
        },
    )
    for plan_changes in compact_plan_cases:
        with pytest.raises(conversation.ConversationValidationError):
            replace(compact_plan, **plan_changes)

    compact_result = StatelessCompactResult(
        id="resp_compact_validation",
        created_at=1,
        output=(canonical_item,),
        usage=conversation.ProviderUsage(input_tokens=1, output_tokens=1),
    )
    assert "item_count=1" in repr(compact_result)
    compact_result_cases = (
        {"created_at": -1},
        {"output": ()},
        {
            "output": cast(
                tuple[Mapping[str, JsonValue], ...],
                ("not-a-mapping",),
            )
        },
        {"usage": cast(conversation.ProviderUsage, object())},
        {
            "checkpoint": cast(
                conversation.ConversationCheckpoint,
                SimpleNamespace(
                    lifecycle=conversation.CheckpointLifecycle.STAGED,
                ),
            )
        },
    )
    for result_changes in compact_result_cases:
        with pytest.raises(conversation.ConversationValidationError):
            replace(compact_result, **result_changes)

    with pytest.raises(conversation.ConversationValidationError):
        replace(configuration.policy, allowed_reasoning_contexts=frozenset())
    with pytest.raises(conversation.ConversationValidationError):
        replace(configuration.policy, max_canonical_items=0)
    with pytest.raises(TypeError):
        replace(
            configuration,
            authority_resolver=cast(
                ResponsesAuthorityResolver,
                lambda request: scope,
            ),
        )
    with pytest.raises(TypeError):
        replace(
            configuration,
            envelope_codec=cast(
                conversation.ContinuationEnvelopeCodec,
                object(),
            ),
        )
    with pytest.raises(TypeError):
        replace(
            configuration,
            compact_resolver=cast(
                stateless_module.StatelessCompactResolver,
                lambda value: value,
            ),
        )
    with pytest.raises(TypeError):
        replace(
            configuration,
            named_head_ledger=cast(
                stateless_module.StatelessNamedHeadLedger,
                object(),
            ),
        )
    with pytest.raises(TypeError):
        replace(
            configuration,
            clock=cast(
                ResponsesClock,
                SimpleNamespace(now=lambda: _NOW),
            ),
        )
    with pytest.raises(TypeError):
        StatelessResponsesService(
            cast(StatelessResponsesConfiguration, object())
        )

    async def broken_authority(
        request: Request,
    ) -> conversation.AuthorityScope:
        del request
        raise RuntimeError("authority unavailable")

    broken_auth_service = StatelessResponsesService(
        replace(configuration, authority_resolver=broken_authority)
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await broken_auth_service.authenticate(
            Request({"type": "http", "headers": []})
        )

    class _NaiveClock:
        async def now(self) -> datetime:
            return datetime(2036, 1, 1)

    with pytest.raises(conversation.ConversationStorageError):
        await StatelessResponsesService(
            replace(configuration, clock=_NaiveClock())
        ).now()
    restricted = replace(
        configuration,
        policy=replace(
            configuration.policy,
            allowed_reasoning_contexts=frozenset(
                {conversation.ReasoningContext.AUTO}
            ),
        ),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await StatelessResponsesService(restricted).prepare_turn(
            authority=scope,
            input_text="unsupported reasoning",
            request_fingerprint=sha256(b"unsupported").hexdigest(),
            reasoning_context=conversation.ReasoningContext.ALL_TURNS,
            streaming=False,
            idempotency_key=None,
            continuation_value=None,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await service.prepare_turn(
            authority=scope,
            input_text="branch without parent",
            request_fingerprint=sha256(b"branch-without-parent").hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=None,
            continuation_value=None,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.EXPLICIT_BRANCH,
                branch_id=conversation.ConversationBranchId("branch-new"),
            ),
        )

    async def failed_turn_resolver(
        plan: StatelessResponsesTurnPlan,
    ) -> PreparedStatelessResponsesTurn:
        del plan
        raise conversation.ConversationStorageError()

    failed_configuration, _, _, _, failed_heads = _configuration()
    failed_service = StatelessResponsesService(
        replace(failed_configuration, turn_resolver=failed_turn_resolver)
    )
    with pytest.raises(conversation.ConversationStorageError):
        await failed_service.prepare_turn(
            authority=scope,
            input_text="reserved failure",
            request_fingerprint=sha256(b"reserved-failure").hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=None,
            continuation_value=None,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.NAMED_HEAD,
                head_id=conversation.NamedHeadId("head-resolver-failure"),
                expected_head_revision=conversation.NamedHeadRevision(0),
            ),
        )
    assert (
        await failed_heads.inspect(
            scope_digest,
            conversation.NamedHeadId("head-resolver-failure"),
        )
        is None
    )
    with pytest.raises(conversation.ConversationValidationError):
        await service.finalize(
            cast(PreparedStatelessResponse, object()),
            request_bytes=1,
            response_bytes=1,
            input_items=1,
            output_items=1,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await service.abort(
            prepared,
            outcome=StatelessResponseOutcome.COMPLETED,
            request_bytes=1,
            input_items=1,
        )
    committed = await service.finalize(
        prepared,
        request_bytes=1,
        response_bytes=1,
        input_items=1,
        output_items=1,
    )
    assert "reconstructable_state_count=0" in repr(committed)
    compact_commit = stateless_module.StatelessCompactCommit(
        result=compact_result,
        continuation=committed.continuation,
    )
    assert "continuation_present=True" in repr(compact_commit)
    assert "extensions" in compact_commit.response_body()
    assert stateless_module.canonical_compact_digest(compact_result)
    with pytest.raises(conversation.ConversationValidationError):
        stateless_module.canonical_compact_digest(
            cast(StatelessCompactResult, object())
        )
    assert stateless_module.is_public_stateless_response_id(
        str(committed.body["id"])
    )
    assert not stateless_module.is_public_stateless_response_id("invalid")
    with pytest.raises(conversation.ConversationValidationError):
        stateless_module.continuation_response_extension(
            cast(conversation.ContinuationEnvelopeToken, object())
        )
    app = FastAPI()
    stateless_module.configure_stateless_responses(app, configuration)
    assert isinstance(
        app.state.stateless_responses_service,
        StatelessResponsesService,
    )
    stateless_module.configure_stateless_responses(app, None)
    assert not hasattr(app.state, "stateless_responses_service")
    stateless_module.configure_stateless_responses(app, None)


async def test_finalize_failure_paths_always_dispose_transient_state() -> None:
    async def prepare(
        service: StatelessResponsesService,
        label: str,
        *,
        named_head: bool = False,
    ) -> PreparedStatelessResponse:
        return await service.prepare_turn(
            authority=_authority(),
            input_text=label,
            request_fingerprint=sha256(label.encode()).hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=None,
            continuation_value=None,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=(
                    conversation.ParentAdvanceMode.NAMED_HEAD
                    if named_head
                    else conversation.ParentAdvanceMode.ORDINARY_CHILD
                ),
                head_id=(
                    conversation.NamedHeadId(f"head-{label}")
                    if named_head
                    else None
                ),
                expected_head_revision=(
                    conversation.NamedHeadRevision(0) if named_head else None
                ),
            ),
        )

    configuration, _, _, _, _ = _configuration()
    service = StatelessResponsesService(configuration)
    mismatched = await prepare(service, "mismatched-checkpoint")
    mismatched_store = cast(_TransientStore, mismatched.prepared.store)

    async def load_mismatched(
        checkpoint_id: conversation.CheckpointId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationCheckpoint:
        del checkpoint_id, authority
        return cast(
            conversation.ConversationCheckpoint,
            SimpleNamespace(
                identity=SimpleNamespace(
                    checkpoint_id=conversation.CheckpointId(
                        "checkpoint-unexpected"
                    )
                )
            ),
        )

    with patch.object(mismatched_store, "load", new=load_mismatched):
        with pytest.raises(conversation.ConversationStorageError):
            await service.finalize(
                mismatched,
                request_bytes=1,
                response_bytes=1,
                input_items=1,
                output_items=1,
            )
    assert mismatched_store.closed

    open_close = await prepare(service, "open-close")
    open_store = cast(_TransientStore, open_close.prepared.store)

    async def close_as_open() -> conversation.StoreCloseResolution:
        await _TransientStore.close(open_store)
        return conversation.StoreCloseResolution(
            disposition=conversation.StoreCloseDisposition.OPEN
        )

    with patch.object(open_store, "close", new=close_as_open):
        with pytest.raises(conversation.ConversationStorageError):
            await service.finalize(
                open_close,
                request_bytes=1,
                response_bytes=1,
                input_items=1,
                output_items=1,
            )
    assert open_store.closed

    retained = await prepare(service, "retained-state")
    retained_store = cast(_TransientStore, retained.prepared.store)

    async def close_without_disposal() -> conversation.StoreCloseResolution:
        return conversation.StoreCloseResolution(
            disposition=conversation.StoreCloseDisposition.CLOSED
        )

    async def nonzero_audit() -> conversation.StoreNonRetentionAudit:
        return _non_retention(1)

    with (
        patch.object(
            retained_store,
            "close",
            new=close_without_disposal,
        ),
        patch.object(
            retained_store,
            "audit_non_retention",
            new=nonzero_audit,
        ),
    ):
        with pytest.raises(conversation.ConversationStorageError):
            await service.finalize(
                retained,
                request_bytes=1,
                response_bytes=1,
                input_items=1,
                output_items=1,
            )
    await _TransientStore.close(retained_store)

    head_configuration, _, _, _, heads = _configuration()
    head_service = StatelessResponsesService(head_configuration)
    wrong_revision = await prepare(
        head_service,
        "wrong-head-revision",
        named_head=True,
    )

    async def commit_wrong_revision(
        reservation: stateless_module.StatelessNamedHeadReservation,
        *,
        continuation_digest: str,
    ) -> conversation.NamedHeadRevision:
        del reservation, continuation_digest
        return conversation.NamedHeadRevision(2)

    with patch.object(heads, "commit", new=commit_wrong_revision):
        with pytest.raises(conversation.ConversationConflictError):
            await head_service.finalize(
                wrong_revision,
                request_bytes=1,
                response_bytes=1,
                input_items=1,
                output_items=1,
            )
    assert (
        await heads.inspect(
            str(conversation.authority_digest(_authority())),
            conversation.NamedHeadId("head-wrong-head-revision"),
        )
        is None
    )


async def test_defensive_prepare_and_cleanup_failures_dispose_state() -> None:
    configuration, resolver, _, _, heads = _configuration()
    service = StatelessResponsesService(configuration)
    scope = _authority()
    advance = conversation.ContinuationEnvelopeAdvance(
        mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
    )
    with pytest.raises(conversation.ConversationValidationError):
        await service.prepare_turn(
            authority=scope,
            input_text="invalid continuation type",
            request_fingerprint=sha256(
                b"invalid-continuation-type"
            ).hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=None,
            continuation_value=cast(
                conversation.ContinuationEnvelopeToken,
                "avl_ce1.invalid",
            ),
            advance=advance,
        )
    with pytest.raises(conversation.ConversationValidationError):
        await service.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(
                cast(
                    Mapping[str, JsonValue],
                    {"role": "user", "type": "message"},
                ),
            ),
            continuation_value=cast(
                conversation.ContinuationEnvelopeToken,
                "avl_ce1.invalid",
            ),
            lane_id=None,
        )

    original_resolver = configuration.turn_resolver

    async def mismatched_resolver(
        plan: StatelessResponsesTurnPlan,
    ) -> PreparedStatelessResponsesTurn:
        prepared = await original_resolver(plan)
        return replace(
            prepared,
            turn=replace(
                prepared.turn,
                public_response_id=conversation.PublicResponseId(
                    "resp_avl_mismatched-resolver"
                ),
            ),
        )

    mismatched_service = StatelessResponsesService(
        replace(configuration, turn_resolver=mismatched_resolver)
    )
    with pytest.raises(conversation.ConversationValidationError):
        await mismatched_service.prepare_turn(
            authority=scope,
            input_text="mismatched prepared turn",
            request_fingerprint=sha256(
                b"mismatched-prepared-turn"
            ).hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=None,
            continuation_value=None,
            advance=advance,
        )
    assert resolver.stores[-1].closed

    async def prepare(
        label: str,
        *,
        named_head: bool = False,
    ) -> PreparedStatelessResponse:
        return await service.prepare_turn(
            authority=scope,
            input_text=label,
            request_fingerprint=sha256(label.encode()).hexdigest(),
            reasoning_context=conversation.ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=None,
            continuation_value=None,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=(
                    conversation.ParentAdvanceMode.NAMED_HEAD
                    if named_head
                    else conversation.ParentAdvanceMode.ORDINARY_CHILD
                ),
                head_id=(
                    conversation.NamedHeadId("head-cleanup-failure")
                    if named_head
                    else None
                ),
                expected_head_revision=(
                    conversation.NamedHeadRevision(0) if named_head else None
                ),
            ),
        )

    release_failure = await prepare("release failure", named_head=True)

    async def fail_release(
        reservation: stateless_module.StatelessNamedHeadReservation,
    ) -> None:
        del reservation
        raise RuntimeError("release cleanup failed")

    with patch.object(heads, "release", new=fail_release):
        with pytest.raises(RuntimeError, match="release cleanup failed"):
            await service.abort(
                release_failure,
                outcome=StatelessResponseOutcome.FAILED,
                request_bytes=1,
                input_items=1,
            )
    assert cast(_TransientStore, release_failure.prepared.store).closed
    assert release_failure.reservation is not None
    await heads.release(release_failure.reservation)

    close_failure = await prepare("close failure")
    close_store = cast(_TransientStore, close_failure.prepared.store)

    async def fail_after_close() -> conversation.StoreCloseResolution:
        await _TransientStore.close(close_store)
        raise RuntimeError("store cleanup failed")

    with patch.object(close_store, "close", new=fail_after_close):
        with pytest.raises(RuntimeError, match="store cleanup failed"):
            await service.abort(
                close_failure,
                outcome=StatelessResponseOutcome.FAILED,
                request_bytes=1,
                input_items=1,
            )
    assert close_store.closed
    assert (
        await close_store.audit_non_retention()
    ).reconstructable_state_count == 0

    audit_failure = await prepare("audit failure")
    audit_store = cast(_TransientStore, audit_failure.prepared.store)

    async def fail_audit() -> conversation.StoreNonRetentionAudit:
        raise RuntimeError("non-retention audit failed")

    with patch.object(audit_store, "audit_non_retention", new=fail_audit):
        with pytest.raises(RuntimeError, match="non-retention audit failed"):
            await service.abort(
                audit_failure,
                outcome=StatelessResponseOutcome.FAILED,
                request_bytes=1,
                input_items=1,
            )
    assert audit_store.closed


async def test_compact_and_prepared_fault_boundaries_are_closed() -> None:
    configuration, _, _, _, _ = _configuration()
    service = StatelessResponsesService(configuration)
    scope = _authority()
    canonical_item = cast(
        Mapping[str, JsonValue],
        {"role": "user", "type": "message"},
    )

    without_compact = StatelessResponsesService(
        replace(configuration, compact_resolver=None)
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await without_compact.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(canonical_item,),
            continuation_value=None,
            lane_id=None,
        )
    with pytest.raises(conversation.ConversationCapabilityError):
        await service.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(canonical_item,),
            continuation_value=None,
            lane_id="lane-not-configured",
        )
    with pytest.raises(conversation.ConversationValidationError):
        await service.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(),
            continuation_value=None,
            lane_id=None,
        )
    bounded = StatelessResponsesService(
        replace(
            configuration,
            policy=replace(configuration.policy, max_canonical_items=1),
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await bounded.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(canonical_item, canonical_item),
            continuation_value=None,
            lane_id=None,
        )

    prepared = await service.prepare_turn(
        authority=scope,
        input_text="compact fault checkpoint",
        request_fingerprint=sha256(b"compact-fault-checkpoint").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key=None,
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
    )
    checkpoint = await prepared.prepared.store.load(
        prepared.prepared.turn.checkpoint_id,
        scope,
    )

    async def seal(
        value: conversation.ConversationCheckpoint,
    ) -> conversation.ContinuationEnvelopeToken:
        return await configuration.envelope_codec.seal(
            value,
            authority=conversation.ContinuationEnvelopeAuthority(
                authority=scope,
                deployment_id=configuration.policy.deployment_id,
            ),
            public_parent=prepared.plan.public_response_id,
            issued_at=_NOW,
        )

    token = await seal(checkpoint)
    missing_lane = StatelessResponsesService(
        replace(
            configuration,
            policy=replace(
                configuration.policy,
                compact_lane_id="lane-missing",
            ),
        )
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await missing_lane.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(),
            continuation_value=token,
            lane_id=None,
        )

    stateless_lane = cast(
        conversation.StatelessProviderLaneSnapshot,
        checkpoint.content.lanes[0],
    )
    stored_lane = conversation.StoredProviderLaneSnapshot(
        binding=stateless_lane.binding,
        upstream_response_id=conversation.UpstreamResponseId(
            "upstream-private"
        ),
        reasoning=stateless_lane.reasoning,
        lifecycle=stateless_lane.lifecycle,
        retention_policy=stateless_lane.retention_policy,
    )
    stored_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            checkpoint,
            content=replace(checkpoint.content, lanes=(stored_lane,)),
            integrity=None,
        )
    )
    stored_token = await seal(stored_checkpoint)
    with pytest.raises(conversation.ConversationCapabilityError):
        await service.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(),
            continuation_value=stored_token,
            lane_id=None,
        )
    await service.abort(
        prepared,
        outcome=StatelessResponseOutcome.CANCELLED,
        request_bytes=1,
        input_items=1,
    )

    async def invalid_compact_result(
        plan: StatelessCompactPlan,
    ) -> StatelessCompactResult:
        del plan
        return cast(StatelessCompactResult, object())

    invalid_result_service = StatelessResponsesService(
        replace(configuration, compact_resolver=invalid_compact_result)
    )
    with pytest.raises(conversation.ConversationStorageError):
        await invalid_result_service.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(canonical_item,),
            continuation_value=None,
            lane_id=None,
        )

    other_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            checkpoint,
            authority=_authority("principal-other"),
            integrity=None,
        )
    )

    async def cross_authority_result(
        plan: StatelessCompactPlan,
    ) -> StatelessCompactResult:
        del plan
        return StatelessCompactResult(
            id="resp_compact_cross_authority",
            created_at=1,
            output=(canonical_item,),
            usage=conversation.ProviderUsage(
                input_tokens=1,
                output_tokens=1,
            ),
            checkpoint=other_checkpoint,
        )

    cross_authority_service = StatelessResponsesService(
        replace(configuration, compact_resolver=cross_authority_result)
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await cross_authority_service.compact(
            authority=scope,
            model="server-model",
            instructions=None,
            canonical_input=(canonical_item,),
            continuation_value=None,
            lane_id=None,
        )

    malformed = conversation.ContinuationEnvelopeToken.from_request(
        "avl_ce1.bad",
        max_chars=configuration.envelope_codec.limits.max_token_chars,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await service._open_envelope(
            malformed,
            authority=scope,
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
            ),
            now=_NOW,
        )
    with pytest.raises(conversation.ConversationValidationError):
        service._response_suffix(
            scope,
            "not-a-digest",
            idempotency_key=None,
        )
    with pytest.raises(conversation.ConversationValidationError):
        service._validate_prepared(
            cast(StatelessResponsesTurnPlan, object()),
            cast(PreparedStatelessResponsesTurn, object()),
        )

    root = await service.prepare_turn(
        authority=scope,
        input_text="explicit root",
        request_fingerprint=sha256(b"explicit-root").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key=None,
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
    )
    root_commit = await service.finalize(
        root,
        request_bytes=1,
        response_bytes=1,
        input_items=1,
        output_items=1,
    )
    explicit = await service.prepare_turn(
        authority=scope,
        input_text="explicit child",
        request_fingerprint=sha256(b"explicit-child").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key=None,
        continuation_value=root_commit.continuation,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.EXPLICIT_BRANCH,
            branch_id=conversation.ConversationBranchId("branch-explicit"),
        ),
    )
    await service.abort(
        explicit,
        outcome=StatelessResponseOutcome.CANCELLED,
        request_bytes=1,
        input_items=1,
    )

    mismatch = await service.prepare_turn(
        authority=scope,
        input_text="prepared mismatch",
        request_fingerprint=sha256(b"prepared-mismatch").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key=None,
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
    )
    mismatched_prepared = replace(
        mismatch.prepared,
        turn=replace(
            mismatch.prepared.turn,
            public_response_id=conversation.PublicResponseId(
                "resp_avl_mismatched"
            ),
        ),
    )
    with pytest.raises(conversation.ConversationValidationError):
        service._validate_prepared(mismatch.plan, mismatched_prepared)
    await service.abort(
        mismatch,
        outcome=StatelessResponseOutcome.CANCELLED,
        request_bytes=1,
        input_items=1,
    )


async def test_envelope_only_compact_and_cross_principal_replay() -> None:
    configuration, _, compact_resolver, _, _ = _configuration()
    service = StatelessResponsesService(configuration)
    first = await service.prepare_turn(
        authority=_authority(),
        input_text="first",
        request_fingerprint=sha256(b"first-compact").hexdigest(),
        reasoning_context=conversation.ReasoningContext.AUTO,
        streaming=False,
        idempotency_key=None,
        continuation_value=None,
        advance=conversation.ContinuationEnvelopeAdvance(
            mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
        ),
    )
    committed = await service.finalize(
        first,
        request_bytes=1,
        response_bytes=1,
        input_items=1,
        output_items=1,
    )
    result = await service.compact(
        authority=_authority(),
        model="server-model",
        instructions=None,
        canonical_input=(),
        continuation_value=committed.continuation,
        lane_id=None,
    )
    assert result.continuation is not None
    assert compact_resolver.plans[-1].input[0]["type"] == "message"
    with pytest.raises(conversation.ConversationError):
        await service.compact(
            authority=_authority("principal-other"),
            model="server-model",
            instructions=None,
            canonical_input=(),
            continuation_value=committed.continuation,
            lane_id=None,
        )


def _native_usage() -> dict[str, object]:
    return {
        "input_tokens": 4,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 6,
        "output_tokens_details": {"reasoning_tokens": 3},
        "total_tokens": 10,
    }


def _native_response(
    *,
    model: str,
    ordinal: int,
) -> dict[str, object]:
    prefix = model.removeprefix("native-")
    return {
        "id": f"native-{prefix}-response-{ordinal}",
        "object": "response",
        "created_at": ordinal,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": [
            {
                "encrypted_content": f"opaque-{prefix}-{ordinal}",
                "id": f"reasoning-{prefix}-{ordinal}",
                "status": "completed",
                "summary": [],
                "type": "reasoning",
            },
            {
                "content": [
                    {
                        "annotations": [],
                        "text": f"{prefix} output {ordinal}",
                        "type": "output_text",
                    }
                ],
                "id": f"message-{prefix}-{ordinal}",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ],
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {"context": "current_turn"},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": _native_usage(),
    }


@dataclass(frozen=True, slots=True)
class _NativeWireRequest:
    path: str
    payload: Mapping[str, object]


@dataclass(slots=True)
class _NativeUpstream:
    requests: list[_NativeWireRequest] = field(default_factory=list)
    server: AsyncioServer | None = None
    base_url: str | None = None

    async def start(self) -> None:
        server = await start_server(self._handle, "127.0.0.1", 0)
        assert server.sockets
        port = cast(tuple[str, int], server.sockets[0].getsockname())[1]
        self.server = server
        self.base_url = f"http://127.0.0.1:{port}/v1"

    async def close(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def _handle(
        self,
        reader: StreamReader,
        writer: StreamWriter,
    ) -> None:
        header = await reader.readuntil(b"\r\n\r\n")
        lines = header.decode("ascii").split("\r\n")
        path = lines[0].split(" ", 2)[1]
        lengths = tuple(
            int(line.split(":", 1)[1].strip())
            for line in lines[1:]
            if line.casefold().startswith("content-length:")
        )
        assert len(lengths) == 1
        body = await reader.readexactly(lengths[0])
        decoded = loads(body)
        assert isinstance(decoded, dict)
        payload = {str(key): value for key, value in decoded.items()}
        self.requests.append(_NativeWireRequest(path=path, payload=payload))
        if path.endswith("/responses/compact"):
            raw_input = payload.get("input")
            assert isinstance(raw_input, list)
            response = {
                "created_at": len(self.requests),
                "id": f"native-compact-response-{len(self.requests)}",
                "object": "response.compaction",
                "output": [
                    *raw_input,
                    {
                        "created_by": "native-upstream",
                        "encrypted_content": "opaque-native-compaction",
                        "id": "native-compaction-boundary",
                        "type": "compaction",
                    },
                ],
                "usage": _native_usage(),
            }
        else:
            model = payload.get("model")
            assert model in {"native-child", "native-parent"}
            ordinal = sum(
                request.payload.get("model") == model
                for request in self.requests
            )
            response = _native_response(
                model=cast(str, model), ordinal=ordinal
            )
        response_body = dumps(response).encode("utf-8")
        writer.write(
            (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(response_body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            + response_body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()


def _native_binding(
    *,
    endpoint: str,
    lane_id: str,
    model: str,
    agent_id: conversation.ConversationAgentId,
    compaction_limits: conversation.NativeOpenAICompactionLimits | None = None,
) -> conversation.ProviderLaneBinding:
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=(
            "avalan.conversation.providers.openai."
            "NativeOpenAIStatelessProvider"
        ),
        provider_family=conversation.ProviderFamily.OPENAI,
        normalized_endpoint=endpoint,
        model_or_deployment=model,
        provider_api_revision=conversation.ProviderApiRevision(
            "openapi-2.3.0"
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("native-model-v1")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("native-capability-v1")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("no-tools-v1"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("native-execution-v1")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.NON_STREAMING,
        agent_id=agent_id,
        compaction_policy_digest=(
            conversation.native_openai_compaction_policy_digest(
                compaction_limits
            )
            if compaction_limits is not None
            else None
        ),
    )


def _native_capabilities(
    binding: conversation.ProviderLaneBinding,
    *,
    compact: bool = False,
) -> conversation.ConversationCapabilityProfile:
    supported = {
        conversation.ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY,
        conversation.ConversationCapability.REASONING_CONTEXT_CURRENT_TURN,
        conversation.ConversationCapability.REASONING_CONTEXT_ALL_TURNS,
    }
    if compact:
        supported.add(
            conversation.ConversationCapability.STANDALONE_COMPACTION
        )
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId(
            f"native-{binding.lane_id}"
        ),
        schema_version=1,
        revision=binding.capability_profile_revision,
        binding_alias=binding.safe_alias,
        capabilities=tuple(
            conversation.CapabilityEvidence(
                capability=capability,
                state=(
                    conversation.CapabilityEvidenceState.TEST_ONLY
                    if capability in supported
                    else conversation.CapabilityEvidenceState.INCAPABLE
                ),
                evidence_ids=(
                    (f"native-{capability.value}",)
                    if capability in supported
                    else ()
                ),
            )
            for capability in conversation.ConversationCapability
        ),
        test_only=True,
    )


def _native_provider(
    binding: conversation.ProviderLaneBinding,
    *,
    compaction_limits: conversation.NativeOpenAICompactionLimits | None = None,
) -> conversation.NativeOpenAIStatelessProvider:
    return conversation.NativeOpenAIStatelessProvider(
        client=AsyncOpenAI(
            api_key="native-upstream-key",
            base_url=binding.normalized_endpoint,
            max_retries=0,
        ),
        profile=conversation.NativeOpenAIStatelessProfile(
            profile_id=f"native-{binding.lane_id}",
            binding=binding,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
            compaction_limits=compaction_limits,
            scripted_tcp_test=True,
        ),
        capability_profile=_native_capabilities(
            binding,
            compact=compaction_limits is not None,
        ),
        tools=(),
    )


def _native_topology(
    endpoint: str,
) -> tuple[
    conversation.AgentLaneTopology,
    conversation.AgentProviderLane,
    conversation.AgentProviderLane,
]:
    conversation_id = conversation.ConversationId(
        "conversation-native-stateless"
    )
    parent_agent_id = conversation.ConversationAgentId("agent-stateless")
    parent_slot = conversation.AgentModelSlot("parent")
    parent_path = conversation.parent_agent_topology_path(
        parent_agent_id,
        parent_slot,
    )
    parent_seed = _native_binding(
        endpoint=endpoint,
        lane_id="lane-native-parent-seed",
        model="native-parent",
        agent_id=parent_agent_id,
    )
    parent_lane_id = conversation.derive_agent_provider_lane_id(
        conversation_id=conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        topology_path=parent_path,
        model_slot=parent_slot,
        binding=parent_seed,
    )
    parent = conversation.AgentProviderLane(
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        agent_id=parent_agent_id,
        topology_path=parent_path,
        model_slot=parent_slot,
        binding=replace(parent_seed, lane_id=parent_lane_id),
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
    )
    child_agent_id = conversation.ConversationAgentId("agent-native-child")
    child_slot = conversation.AgentModelSlot("child")
    child_path = conversation.child_agent_topology_path(
        parent_path,
        child_agent_id,
        child_slot,
    )
    child_seed = _native_binding(
        endpoint=endpoint,
        lane_id="lane-native-child-seed",
        model="native-child",
        agent_id=child_agent_id,
    )
    child_lane_id = conversation.derive_agent_provider_lane_id(
        conversation_id=conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.CHILD_AGENT,
        topology_path=child_path,
        model_slot=child_slot,
        binding=child_seed,
    )
    child = conversation.AgentProviderLane(
        owner_kind=conversation.ProviderLaneOwnerKind.CHILD_AGENT,
        agent_id=child_agent_id,
        topology_path=child_path,
        model_slot=child_slot,
        binding=replace(child_seed, lane_id=child_lane_id),
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        parent_lane_id=parent.lane_id,
    )
    return (
        conversation.AgentLaneTopology(
            conversation_id=conversation_id,
            lanes=(parent, child),
        ),
        parent,
        child,
    )


class _CoordinatorOwnedStore:
    durable = False

    def __init__(
        self,
        core: conversation.InMemoryConversationStore,
        coordinator: conversation.RunScopedConversationCoordinator,
    ) -> None:
        self.core = core
        self.coordinator = coordinator
        self.closed = False

    async def retrieve(
        self,
        public_response_id: conversation.PublicResponseId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationResult:
        return await self.core.retrieve(public_response_id, authority)

    async def load(
        self,
        checkpoint_id: conversation.CheckpointId,
        authority: conversation.AuthorityScope,
    ) -> conversation.ConversationCheckpoint:
        return await self.core.load(checkpoint_id, authority)

    async def close(self) -> conversation.StoreCloseResolution:
        try:
            await self.coordinator.close()
        finally:
            resolution = await self.core.inspect_close()
            self.closed = (
                resolution.disposition
                is conversation.StoreCloseDisposition.CLOSED
            )
        return resolution

    async def audit_non_retention(
        self,
    ) -> conversation.StoreNonRetentionAudit:
        return await self.core.audit_non_retention()


class _NativeTurnResolver:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.topology, self.parent, self.child = _native_topology(endpoint)
        self.stores: list[_CoordinatorOwnedStore] = []
        self.plans: list[StatelessResponsesTurnPlan] = []

    async def __call__(
        self,
        plan: StatelessResponsesTurnPlan,
    ) -> PreparedStatelessResponsesTurn:
        self.plans.append(plan)
        parent_checkpoint = (
            plan.opened_parent.checkpoint
            if plan.opened_parent is not None
            else None
        )
        branch_id = (
            plan.opened_parent.target_branch_id
            if plan.opened_parent is not None
            else conversation.ConversationBranchId("branch-native-stateless")
        )
        providers = tuple(
            _native_provider(lane.binding)
            for lane in (self.parent, self.child)
        )
        core = conversation.InMemoryConversationStore()
        if plan.opened_parent is not None:
            staged_parent = conversation.with_checkpoint_integrity(
                replace(
                    plan.opened_parent.checkpoint,
                    lifecycle=conversation.CheckpointLifecycle.STAGED,
                    timestamps=replace(
                        plan.opened_parent.checkpoint.timestamps,
                        committed_at=None,
                    ),
                    integrity=None,
                )
            )
            await core.create(
                conversation.OutwardTurnCheckpointCandidate(
                    checkpoint=staged_parent,
                    public_response_id=plan.opened_parent.public_parent,
                )
            )
        coordinator = conversation.RunScopedConversationCoordinator(
            store=core,
            authority_resolver=conversation.DeterministicFakeAuthorityResolver(
                plan.authority
            ),
            clock=conversation.DeterministicFakeClock(_NOW),
            publisher=conversation.DeterministicFakePublisher(),
            observer=conversation.DeterministicFakeObserver(),
            retry_waiter=conversation.DeterministicFakeRetryWaiter(),
            lanes=tuple(
                conversation.NativeOpenAIConversationLaneRuntime(
                    provider=provider,
                    retention_policy=lane.retention_policy,
                )
                for provider, lane in zip(
                    providers,
                    (self.parent, self.child),
                    strict=True,
                )
            ),
        )
        store = _CoordinatorOwnedStore(core, coordinator)
        self.stores.append(store)
        advance: (
            conversation.ExplicitBranchAdvance
            | conversation.NamedHeadAdvance
            | None
        ) = None
        if parent_checkpoint is not None and (
            plan.advance.mode is conversation.ParentAdvanceMode.EXPLICIT_BRANCH
        ):
            assert plan.advance.branch_id is not None
            advance = conversation.ExplicitBranchAdvance(
                parent_checkpoint_id=(
                    parent_checkpoint.identity.checkpoint_id
                ),
                branch_id=plan.advance.branch_id,
            )
        elif parent_checkpoint is not None and (
            plan.advance.mode is conversation.ParentAdvanceMode.NAMED_HEAD
        ):
            assert plan.advance.head_id is not None
            assert plan.advance.expected_head_revision is not None
            advance = conversation.NamedHeadAdvance(
                head_id=plan.advance.head_id,
                parent_checkpoint_id=(
                    parent_checkpoint.identity.checkpoint_id
                ),
                expected_revision=plan.advance.expected_head_revision,
            )
        turn = conversation.AgentConversationTurn(
            coordinator=coordinator,
            authority=plan.authority,
            topology=self.topology,
            lanes=tuple(
                conversation.AgentConversationLane(
                    lane_id=lane.lane_id,
                    mode=conversation.ConversationMode.STATELESS,
                    reasoning_context=plan.reasoning_context,
                )
                for lane in self.topology.lanes
            ),
            logical_turn_id=conversation.LogicalTurnId(
                f"turn-{plan.public_response_id}"
            ),
            execution_segment_id=conversation.ExecutionSegmentId(
                f"segment-{plan.public_response_id}"
            ),
            checkpoint_id=conversation.CheckpointId(
                f"checkpoint-{plan.public_response_id}"
            ),
            branch_id=branch_id,
            provisional_response_id=plan.provisional_response_id,
            public_response_id=plan.public_response_id,
            idempotency_key=plan.idempotency_key,
            retention=plan.retention,
            parent=parent_checkpoint,
            advance=advance,
        )
        return PreparedStatelessResponsesTurn(turn=turn, store=store)


class _NativeCompactResolver:
    def __init__(self, endpoint: str) -> None:
        self.plans: list[StatelessCompactPlan] = []
        self.limits = conversation.NativeOpenAICompactionLimits(
            min_compact_threshold=64,
            max_compact_threshold=4_096,
            max_input_items=32,
            max_input_bytes=65_536,
            max_output_items=32,
            max_output_bytes=65_536,
        )
        self.binding = _native_binding(
            endpoint=endpoint,
            lane_id="lane-native-compact",
            model="native-compact",
            agent_id=conversation.ConversationAgentId("agent-stateless"),
            compaction_limits=self.limits,
        )

    async def __call__(
        self,
        plan: StatelessCompactPlan,
    ) -> StatelessCompactResult:
        self.plans.append(plan)
        model_call_id = conversation.ConversationModelCallId(
            "native-compact-input-call"
        )
        items = tuple(
            conversation.ProviderItem(
                item_id=conversation.ProviderItemId(
                    f"native-compact-input-{index}"
                ),
                lane_id=self.binding.lane_id,
                model_call_id=model_call_id,
                kind=conversation.ProviderItemKind.MESSAGE,
                order=conversation.ProviderItemOrder(index),
                provider_index=conversation.ProviderItemIndex(index),
                phase=conversation.ProviderItemPhase.INPUT,
                caller=conversation.ProviderItemCaller.CALLER,
                canonical_input=item,
                normalization_version=(
                    conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
                ),
            )
            for index, item in enumerate(plan.input)
        )
        provider = _native_provider(
            self.binding,
            compaction_limits=self.limits,
        )
        try:
            result = await provider.compact(
                conversation.StandaloneCompactProviderPlan(
                    binding=self.binding,
                    ledger=conversation.ProviderItemLedger(
                        lane_id=self.binding.lane_id,
                        normalization_version=(
                            conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
                        ),
                        items=items,
                    ),
                    reasoning=conversation.EffectiveReasoningMetadata(
                        requested=conversation.ReasoningContext.AUTO,
                        effective=None,
                    ),
                )
            )
        finally:
            await provider.aclose()
        return StatelessCompactResult(
            id=str(result.upstream_response_id or "native-compact-response"),
            created_at=int(_NOW.timestamp()),
            output=tuple(
                cast(
                    Mapping[str, JsonValue],
                    conversation.thaw_json_value(item.canonical_input),
                )
                for item in result.items
            ),
            usage=result.usage,
            checkpoint=plan.checkpoint,
        )


class _NativeRouteOrchestrator(_RouteOrchestrator):
    @staticmethod
    def _input_text(messages: object) -> str:
        assert isinstance(messages, list) and messages
        content = getattr(messages[-1], "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, Mapping):
            text = content.get("text")
            if isinstance(text, str):
                return text
        text = getattr(content, "text", None)
        if isinstance(text, str):
            return text
        if isinstance(content, list):
            values = tuple(
                (
                    part.get("text")
                    if isinstance(part, Mapping)
                    else getattr(part, "text", None)
                )
                for part in content
                if isinstance(
                    (
                        part.get("text")
                        if isinstance(part, Mapping)
                        else getattr(part, "text", None)
                    ),
                    str,
                )
            )
            if values:
                return "\n".join(cast(tuple[str, ...], values))
        raise AssertionError("native Responses input must be text")

    async def __call__(
        self,
        messages: object,
        settings: object = None,
        **kwargs: object,
    ) -> TextGenerationResponse:
        turn = kwargs.get("conversation_turn")
        if not isinstance(turn, conversation.AgentConversationTurn):
            return await super().__call__(messages, settings, **kwargs)
        input_text = self._input_text(messages)
        result = await turn.execute(input_text)
        stream = bool(getattr(settings, "use_async_generator", False))
        return TextGenerationResponse(
            (
                _slow_stream_items
                if stream and "disconnect" in input_text
                else _stream_items if stream else lambda **_: result.output
            ),
            logger=getLogger(__name__),
            use_async_generator=stream,
            inputs={"input_ids": [[1, 2, 3]]},
        )


def _native_configuration(
    endpoint: str,
) -> tuple[
    StatelessResponsesConfiguration,
    _NativeTurnResolver,
    _NativeCompactResolver,
    InMemoryStatelessResponsesAuditHook,
    InMemoryStatelessNamedHeadLedger,
]:
    base, _, _, audit, heads = _configuration()
    resolver = _NativeTurnResolver(endpoint)
    compact_resolver = _NativeCompactResolver(endpoint)
    return (
        replace(
            base,
            turn_resolver=resolver,
            compact_resolver=compact_resolver,
            policy=replace(
                base.policy,
                compact_lane_id=str(compact_resolver.binding.lane_id),
            ),
        ),
        resolver,
        compact_resolver,
        audit,
        heads,
    )


async def _serve_stateless_responses(
    port: int,
    readiness: Connection,
    shutdown: Connection,
    upstream_endpoint: str,
) -> None:
    configuration, resolver, compact_resolver, audit, _ = (
        _native_configuration(upstream_endpoint)
    )
    app = FastAPI()
    orchestrator = _NativeRouteOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.logger = getLogger(__name__)
    app.state.stateless_responses_service = StatelessResponsesService(
        configuration
    )
    app.include_router(responses.router, prefix="/v1")

    @app.get("/test/state")
    async def test_state() -> dict[str, object]:
        return {
            "audit": [
                {
                    "operation": record.operation,
                    "outcome": record.outcome.value,
                    "reconstructable_state_count": (
                        record.reconstructable_state_count
                    ),
                }
                for record in audit.records
            ],
            "closed_stores": sum(store.closed for store in resolver.stores),
            "compact_plans": len(compact_resolver.plans),
            "open_stores": sum(not store.closed for store in resolver.stores),
            "pid": getpid(),
            "turn_plans": len(resolver.plans),
        }

    server = Server(
        Config(
            app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="error",
            lifespan="off",
        )
    )

    async def announce() -> None:
        for _ in range(1_000):
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


def _stateless_server_target(
    port: int,
    readiness: Connection,
    shutdown: Connection,
    upstream_endpoint: str,
) -> None:
    try:
        run(
            _serve_stateless_responses(
                port,
                readiness,
                shutdown,
                upstream_endpoint,
            )
        )
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
    process: BaseProcess
    port: int
    shutdown: Connection

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"


def _unused_loopback_port() -> int:
    listener = socket(AF_INET, SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        address = cast(tuple[str, int], listener.getsockname())
        return address[1]
    finally:
        listener.close()


async def _start_stateless_server(
    upstream_endpoint: str,
) -> _ServerProcess:
    context = get_context("spawn")
    parent_readiness, child_readiness = context.Pipe(duplex=False)
    child_shutdown, parent_shutdown = context.Pipe(duplex=False)
    port = _unused_loopback_port()
    process = context.Process(
        target=_stateless_server_target,
        args=(port, child_readiness, child_shutdown, upstream_endpoint),
    )
    process.start()
    child_readiness.close()
    child_shutdown.close()
    available = await to_thread(parent_readiness.poll, 20)
    if not available:
        process.terminate()
        await to_thread(process.join, 5)
        parent_shutdown.close()
        pytest.fail("stateless Responses server did not report readiness")
    success, detail = parent_readiness.recv()
    parent_readiness.close()
    if not success:
        process.terminate()
        await to_thread(process.join, 5)
        parent_shutdown.close()
        pytest.fail(f"stateless Responses server failed: {detail!r}")
    assert process.is_alive()
    return _ServerProcess(
        process=process,
        port=port,
        shutdown=parent_shutdown,
    )


async def _stop_stateless_server(server: _ServerProcess) -> None:
    if server.process.is_alive():
        server.shutdown.send("stop")
    server.shutdown.close()
    await to_thread(server.process.join, 10)
    if server.process.is_alive():
        server.process.kill()
        await to_thread(server.process.join, 5)
        pytest.fail("stateless Responses server did not stop")
    assert server.process.exitcode == 0


def _continuation_from_model(value: object) -> str:
    dump = getattr(value, "model_dump", None)
    assert callable(dump)
    body = cast(Mapping[str, object], dump())
    extensions = cast(Mapping[str, object], body["extensions"])
    avalan_extension = cast(Mapping[str, object], extensions["avalan"])
    conversation_extension = cast(
        Mapping[str, object],
        avalan_extension["conversation"],
    )
    token = conversation_extension["continuation_envelope"]
    assert isinstance(token, str)
    return token


async def _server_state(server: _ServerProcess) -> dict[str, object]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{server.base_url}/test/state")
    response.raise_for_status()
    return cast(dict[str, object], response.json())


def _caller_held_body(token: str | None = None) -> dict[str, object]:
    conversation_extension: dict[str, object] = {
        "version": "1",
        "mode": "caller_held",
    }
    if token is not None:
        conversation_extension["continuation_envelope"] = token
    return {
        "extensions": {
            "avalan": {
                "version": "1",
                "conversation": conversation_extension,
            }
        }
    }


async def test_normative_server_stateless_contract(
    record_property: Callable[[str, object], None],
) -> None:
    """Run native multi-lane continuity through public and provider TCP."""
    record_property("conversation_acceptance_evidence", "public")
    upstream = _NativeUpstream()
    first_server: _ServerProcess | None = None
    second_server: _ServerProcess | None = None
    try:
        await upstream.start()
        assert upstream.base_url is not None
        first_server = await _start_stateless_server(upstream.base_url)
        first_pid = cast(int, (await _server_state(first_server))["pid"])
        async with AsyncOpenAI(
            api_key="owner",
            base_url=first_server.openai_base_url,
        ) as client:
            enveloped = await client.responses.create(
                model="server-model",
                input="envelope one",
                store=False,
                extra_body=_caller_held_body(),
            )
            assert enveloped.output_text == "parent output 1"
            first_token = _continuation_from_model(enveloped)

        state = await _server_state(first_server)
        assert state["open_stores"] == 0
        assert state["closed_stores"] == 1
        assert state["turn_plans"] == 1
        first_audit = cast(list[dict[str, object]], state["audit"])
        assert first_audit
        assert all(
            record["reconstructable_state_count"] == 0
            for record in first_audit
        )
        await _stop_stateless_server(first_server)
        first_server = None

        probe_configuration, _, _, _, _ = _native_configuration(
            upstream.base_url
        )
        probe_token = conversation.ContinuationEnvelopeToken.from_request(
            first_token,
            max_chars=(
                probe_configuration.envelope_codec.limits.max_token_chars
            ),
        )
        probe = await StatelessResponsesService(
            probe_configuration
        )._open_envelope(
            probe_token,
            authority=_authority(),
            advance=conversation.ContinuationEnvelopeAdvance(
                mode=conversation.ParentAdvanceMode.ORDINARY_CHILD
            ),
            now=_NOW,
        )
        assert probe.checkpoint.identity.conversation_id == (
            conversation.ConversationId("conversation-native-stateless")
        )

        second_server = await _start_stateless_server(upstream.base_url)
        second_pid = cast(int, (await _server_state(second_server))["pid"])
        assert second_pid != first_pid
        async with AsyncOpenAI(
            api_key="owner",
            base_url=second_server.openai_base_url,
        ) as client:
            try:
                continued = await client.responses.create(
                    model="server-model",
                    input="envelope two",
                    store=False,
                    extra_body=_caller_held_body(first_token),
                )
            except NotFoundError:
                pytest.fail(
                    f"restart state: {await _server_state(second_server)!r}"
                )
            assert continued.output_text == "parent output 2"
            second_token = _continuation_from_model(continued)
            assert second_token != first_token
            compacted = await client.responses.compact(
                model="server-model",
                input=cast(
                    Any,
                    [
                        {
                            "content": [
                                {
                                    "text": "compact context",
                                    "type": "input_text",
                                }
                            ],
                            "role": "user",
                            "type": "message",
                        }
                    ],
                ),
                instructions="compact natively",
            )
            assert compacted.output[0].type == "message"
            assert compacted.output[-1].type == "compaction"
        async with AsyncOpenAI(
            api_key="another-principal",
            base_url=second_server.openai_base_url,
        ) as other:
            with pytest.raises(NotFoundError):
                await other.responses.create(
                    model="server-model",
                    input="forbidden",
                    store=False,
                    extra_body=_caller_held_body(first_token),
                )

        async with httpx.AsyncClient(timeout=3) as raw_client:
            lines: list[str] = []
            async with raw_client.stream(
                "POST",
                f"{second_server.openai_base_url}/responses",
                headers={"authorization": "Bearer owner"},
                json={
                    "input": "disconnect",
                    "model": "server-model",
                    "store": False,
                    "stream": True,
                    **_caller_held_body(),
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    lines.append(line)
                    if "response.created" in line:
                        break
        assert "continuation_envelope" not in "\n".join(lines)
        for _ in range(100):
            state = await _server_state(second_server)
            if state["open_stores"] == 0:
                break
            await sleep(0.02)
        assert state["open_stores"] == 0
        audit_records = cast(list[dict[str, object]], state["audit"])
        assert all(
            record["reconstructable_state_count"] == 0
            for record in audit_records
        )
        assert state["compact_plans"] == 1
        assert state["closed_stores"] == 2
        assert state["turn_plans"] == 2

        paths = [request.path for request in upstream.requests]
        assert paths == [
            "/v1/responses",
            "/v1/responses",
            "/v1/responses",
            "/v1/responses",
            "/v1/responses/compact",
            "/v1/responses",
            "/v1/responses",
        ]
        models = [
            request.payload.get("model") for request in upstream.requests
        ]
        assert models == [
            "native-child",
            "native-parent",
            "native-child",
            "native-parent",
            "native-compact",
            "native-child",
            "native-parent",
        ]
        assert all(
            request.payload.get("tools", []) == []
            for request in upstream.requests
        )
        child_second = dumps(upstream.requests[2].payload, sort_keys=True)
        parent_second = dumps(upstream.requests[3].payload, sort_keys=True)
        assert "envelope two" in child_second
        assert "reasoning-child-1" not in child_second
        assert "reasoning-parent-1" not in child_second
        assert "reasoning-parent-1" in parent_second
        assert "message-parent-1" in parent_second
        assert "reasoning-child-1" not in parent_second
        assert "child output 2" in parent_second
        compact_wire = upstream.requests[4].payload
        assert compact_wire["input"] == [
            {
                "content": [{"text": "compact context", "type": "input_text"}],
                "role": "user",
                "type": "message",
            }
        ]
    finally:
        for server in (second_server, first_server):
            if server is not None:
                await _stop_stateless_server(server)
        if upstream.server is not None:
            await upstream.close()
