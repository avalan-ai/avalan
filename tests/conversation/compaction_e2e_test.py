"""Exercise native compaction over real TCP and durable checkpoints."""

from asyncio import (
    Server,
    StreamReader,
    StreamWriter,
    run,
    start_server,
    to_thread,
)
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from json import dumps, loads
from multiprocessing import get_context
from multiprocessing.connection import Connection
from os import environ
from typing import cast
from uuid import uuid4

import pytest
from openai import AsyncOpenAI
from phase2_fixtures import authority, retention

import avalan
import avalan.conversation as conversation
from avalan.conversation.providers.openai import (
    _native_openai_test_authority,
)
from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)
from avalan.types import JsonValue

_STATELESS_ADAPTER = (
    "avalan.conversation.providers.openai.NativeOpenAIStatelessProvider"
)
_STORED_ADAPTER = (
    "avalan.conversation.providers.openai_stored.NativeOpenAIStoredProvider"
)
_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
_NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        _DSN is None,
        reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run durable compaction evidence on asyncio only."""
    return "asyncio"


def _key() -> conversation.ConversationDataKey:
    return conversation.ConversationDataKey(
        key_id="native-compaction-key",
        revision=1,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"7" * 32,
    )


def _resolver() -> conversation.InMemoryConversationKeyResolver:
    scope = authority()
    return conversation.InMemoryConversationKeyResolver(
        {conversation.authority_digest(scope): (_key(),)}
    )


def _store(dsn: str, schema: str) -> conversation.PgsqlConversationStore:
    return conversation.PgsqlConversationStore.from_settings(
        conversation.PgsqlConversationStoreSettings(
            dsn=dsn,
            schema=schema,
            pool_minimum=1,
            pool_maximum=2,
        ),
        key_resolver=_resolver(),
        cipher=conversation.AesGcmConversationCipher(),
        clock=conversation.DeterministicFakeClock(_NOW),
    )


async def _drop_schema(dsn: str, schema: str) -> None:
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


@pytest.fixture
async def pgsql_schema() -> AsyncIterator[tuple[str, str]]:
    """Yield one migrated isolated schema for compact operations."""
    assert _DSN is not None
    schema = f"conv_compaction_{uuid4().hex}"
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    try:
        yield _DSN, schema
    finally:
        await _drop_schema(_DSN, schema)


def _usage() -> dict[str, object]:
    return {
        "input_tokens": 8,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 2},
        "total_tokens": 13,
    }


def _reasoning(identifier: str, opaque: str) -> dict[str, object]:
    return {
        "encrypted_content": opaque,
        "id": identifier,
        "status": "completed",
        "summary": [],
        "type": "reasoning",
    }


def _message(identifier: str, text: str) -> dict[str, object]:
    return {
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "id": identifier,
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _input_message(text: str) -> dict[str, object]:
    return {
        "content": [{"text": text, "type": "input_text"}],
        "role": "user",
        "type": "message",
    }


def _compaction(identifier: str, opaque: str) -> dict[str, object]:
    return {
        "created_by": "scripted-provider",
        "encrypted_content": opaque,
        "id": identifier,
        "type": "compaction",
    }


def _function_call(identifier: str, call_id: str) -> dict[str, object]:
    return {
        "arguments": '{"value":1}',
        "call_id": call_id,
        "id": identifier,
        "name": "lookup",
        "status": "completed",
        "type": "function_call",
    }


def _response(
    identifier: str,
    output: list[dict[str, object]],
    *,
    stored: bool = False,
    previous_response_id: str | None = None,
    execution: conversation.NativeOpenAIStoredExecution | None = None,
) -> dict[str, object]:
    return {
        "created_at": 1,
        "error": None,
        "id": identifier,
        "incomplete_details": None,
        "instructions": execution.instructions if execution else None,
        "max_output_tokens": (
            execution.max_output_tokens if execution else None
        ),
        "max_tool_calls": execution.max_tool_calls if execution else None,
        "model": "gpt-5",
        "object": "response",
        "output": output,
        "parallel_tool_calls": (
            execution.parallel_tool_calls if execution else False
        ),
        "previous_response_id": previous_response_id,
        "reasoning": {"context": "current_turn"},
        "safety_identifier": (
            execution.safety_identifier if execution else None
        ),
        "status": "completed",
        "store": stored,
        "temperature": execution.temperature if execution else None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": execution.top_p if execution else None,
        "truncation": execution.truncation if execution else "disabled",
        "usage": _usage(),
    }


def _scripted_response(
    scenario: str,
    path: str,
    request_index: int,
) -> dict[str, object]:
    if scenario == "long":
        if request_index < 4:
            number = request_index + 1
            return _response(
                f"long-response-{number}",
                [
                    _reasoning(
                        f"long-reasoning-{number}",
                        f"long-private-{number}",
                    ),
                    _message(f"long-message-{number}", f"turn-{number}"),
                ],
            )
        if request_index == 4:
            return _response(
                "long-response-compacted",
                [
                    _compaction("long-compact", "long-compact-private"),
                    _message("long-message-compacted", "compacted-turn"),
                ],
            )
        return _response(
            "long-response-restarted",
            [_message("long-message-restarted", "after-restart")],
        )
    if scenario == "tools":
        outputs = (
            [_function_call("tools-pre-one", "call-pre-one")],
            [
                _compaction("tools-compact-one", "tools-private-one"),
                _function_call("tools-post-one", "call-post-one"),
            ],
            [_message("tools-final-one", "first tool turn")],
            [_function_call("tools-pre-two", "call-pre-two")],
            [
                _compaction("tools-compact-two", "tools-private-two"),
                _function_call("tools-post-two", "call-post-two"),
            ],
            [_message("tools-final-two", "second tool turn")],
            [_message("tools-final-three", "third tool turn")],
        )
        return _response(
            f"tools-response-{request_index + 1}",
            outputs[request_index],
        )
    if scenario == "stored":
        execution = _stored_execution()
        parent = None if request_index == 0 else "stored-response-1"
        return _response(
            f"stored-response-{request_index + 1}",
            [
                _compaction(
                    f"stored-compact-{request_index + 1}",
                    f"stored-private-{request_index + 1}",
                ),
                _message(
                    f"stored-message-{request_index + 1}",
                    f"stored turn {request_index + 1}",
                ),
            ],
            stored=True,
            previous_response_id=parent,
            execution=execution,
        )
    if scenario == "standalone":
        if path.endswith("/responses/compact"):
            return {
                "created_at": 2,
                "id": "standalone-compact-response",
                "object": "response.compaction",
                "output": [
                    _input_message("retained original input"),
                    _compaction(
                        "standalone-compact",
                        "standalone-compact-private",
                    ),
                ],
                "usage": _usage(),
            }
        response_number = sum(
            1 for index in range(request_index + 1) if index != 1
        )
        output = {
            1: [
                _reasoning(
                    "standalone-original-reasoning", "original-private"
                ),
                _message("standalone-original-message", "original"),
            ],
            2: [_message("standalone-fork-message", "fork continued")],
            3: [_message("standalone-branch-message", "original branched")],
        }[response_number]
        return _response(f"standalone-response-{response_number}", output)
    raise AssertionError("unknown scripted compaction scenario")


@dataclass(frozen=True, slots=True)
class _WireRequest:
    path: str
    body: bytes
    payload: dict[str, object]


@dataclass(slots=True)
class _ScriptedTcpProvider:
    scenario: str
    requests: list[_WireRequest] = field(default_factory=list)
    _server: Server | None = None
    base_url: str | None = None

    async def start(self) -> None:
        """Start one real loopback HTTP provider."""
        server = await start_server(self._handle, "127.0.0.1", 0)
        socket = server.sockets[0]
        port = cast(tuple[str, int], socket.getsockname())[1]
        self._server = server
        self.base_url = f"http://127.0.0.1:{port}"

    async def close(self) -> None:
        """Close the loopback provider."""
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(
        self,
        reader: StreamReader,
        writer: StreamWriter,
    ) -> None:
        header = await reader.readuntil(b"\r\n\r\n")
        lines = header.decode("ascii").split("\r\n")
        path = lines[0].split(" ", 2)[1]
        lengths = [
            int(line.split(":", 1)[1].strip())
            for line in lines[1:]
            if line.casefold().startswith("content-length:")
        ]
        assert len(lengths) == 1
        body = await reader.readexactly(lengths[0])
        decoded = loads(body)
        assert isinstance(decoded, dict)
        payload = {str(key): value for key, value in decoded.items()}
        request_index = len(self.requests)
        self.requests.append(
            _WireRequest(path=path, body=body, payload=payload)
        )
        response = _scripted_response(
            self.scenario,
            path,
            request_index,
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


def _limits() -> conversation.NativeOpenAICompactionLimits:
    return conversation.NativeOpenAICompactionLimits(
        min_compact_threshold=64,
        max_compact_threshold=4_096,
        max_input_items=128,
        max_input_bytes=1_048_576,
        max_output_items=128,
        max_output_bytes=1_048_576,
    )


def _binding(
    endpoint: str,
    lane_id: str,
    *,
    stored: bool = False,
) -> conversation.ProviderLaneBinding:
    base = conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=_STORED_ADAPTER if stored else _STATELESS_ADAPTER,
        provider_family=conversation.ProviderFamily.OPENAI,
        normalized_endpoint=endpoint,
        model_or_deployment="gpt-5",
        provider_api_revision=conversation.ProviderApiRevision(
            "openapi-2.3.0"
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("compaction-model")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("compaction-capability")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision(
            "compaction-tools"
        ),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("compaction-execution")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=conversation.ProviderTransport.NON_STREAMING,
        agent_id=authority().agent_id,
        compaction_policy_digest=(
            conversation.native_openai_compaction_policy_digest(_limits())
        ),
    )
    if not stored:
        return base
    return replace(
        base,
        execution_definition_digest=(
            conversation.native_openai_stored_execution_digest(
                binding=base,
                execution=_stored_execution(),
                encrypted_content=(
                    conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
                ),
                compaction_limits=_limits(),
            )
        ),
    )


def _capabilities(
    binding: conversation.ProviderLaneBinding,
    *,
    stored: bool = False,
) -> conversation.ConversationCapabilityProfile:
    capability = conversation.ConversationCapability
    supported = {
        capability.INLINE_COMPACTION,
        capability.REASONING_CONTEXT_CURRENT_TURN,
        capability.REASONING_CONTEXT_ALL_TURNS,
        (
            capability.STORED_RESPONSES_CHAINING
            if stored
            else capability.STATELESS_ENCRYPTED_REASONING_REPLAY
        ),
    }
    if not stored:
        supported.add(
            conversation.ConversationCapability.STANDALONE_COMPACTION
        )
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId(
            f"compaction-profile-{binding.lane_id}"
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
                    (f"compaction-{capability.value}",)
                    if capability in supported
                    else ()
                ),
            )
            for capability in conversation.ConversationCapability
        ),
        test_only=True,
    )


async def _lookup(arguments: Mapping[str, JsonValue]) -> str:
    assert arguments == {"value": 1}
    return "lookup-result"


def _stateless_provider(
    binding: conversation.ProviderLaneBinding,
    *,
    tools: bool = False,
) -> conversation.NativeOpenAIStatelessProvider:
    configured_tools = (
        (
            conversation.NativeOpenAIFunctionTool(
                name="lookup",
                description="Return one deterministic value.",
                parameters={
                    "additionalProperties": False,
                    "properties": {"value": {"type": "integer"}},
                    "required": ("value",),
                    "type": "object",
                },
                handler=_lookup,
            ),
        )
        if tools
        else ()
    )
    client = AsyncOpenAI(
        api_key="compaction-test-key",
        base_url=binding.normalized_endpoint,
        max_retries=0,
    )
    profile = conversation.NativeOpenAIStatelessProfile(
        profile_id=f"compaction-{binding.lane_id}",
        binding=binding,
        encrypted_content=(
            conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
        ),
        compaction_limits=_limits(),
        scripted_tcp_test=True,
    )
    capabilities = _capabilities(binding)
    return conversation.NativeOpenAIStatelessProvider(
        client=client,
        profile=profile,
        capability_profile=capabilities,
        tools=configured_tools,
        test_authority=_native_openai_test_authority(
            client=client,
            binding=binding,
            scripted_tcp_test=profile.scripted_tcp_test,
            capability_profile=capabilities,
        ),
    )


def _stored_execution() -> conversation.NativeOpenAIStoredExecution:
    return conversation.NativeOpenAIStoredExecution(
        instructions="Use the frozen compaction test execution.",
        max_output_tokens=256,
        max_tool_calls=8,
    )


def _stored_provider(
    binding: conversation.ProviderLaneBinding,
) -> conversation.NativeOpenAIStoredProvider:
    client = AsyncOpenAI(
        api_key="stored-compaction-test-key",
        base_url=binding.normalized_endpoint,
        max_retries=0,
    )
    profile = conversation.NativeOpenAIStoredProfile(
        profile_id=f"stored-compaction-{binding.lane_id}",
        binding=binding,
        execution=_stored_execution(),
        encrypted_content=(
            conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
        ),
        compaction_limits=_limits(),
        scripted_tcp_test=True,
    )
    capabilities = _capabilities(binding, stored=True)
    return conversation.NativeOpenAIStoredProvider(
        client=client,
        profile=profile,
        capability_profile=capabilities,
        test_authority=_native_openai_test_authority(
            client=client,
            binding=binding,
            scripted_tcp_test=profile.scripted_tcp_test,
            capability_profile=capabilities,
        ),
    )


def _stored_retention() -> conversation.RetentionLimits:
    return conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.DURABLE,
            upstream=conversation.ProviderLaneStorage.STORED,
            provider_storage_disclosed=True,
        ),
        upstream_lifetime_status=conversation.UpstreamLifetimeStatus.UNKNOWN,
        local_ttl_seconds=3_600,
    )


def _client(
    store: conversation.PgsqlConversationStore,
    provider: (
        conversation.NativeOpenAIStatelessProvider
        | conversation.NativeOpenAIStoredProvider
    ),
    namespace: str,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.RunScopedConversationCoordinator,
]:
    scope = authority()
    lane_runtime: (
        conversation.NativeOpenAIConversationLaneRuntime
        | conversation.NativeOpenAIStoredLaneRuntime
    )
    if type(provider) is conversation.NativeOpenAIStatelessProvider:
        lane_runtime = conversation.NativeOpenAIConversationLaneRuntime(
            provider=provider
        )
        selected_retention = retention()
    else:
        lane_runtime = conversation.NativeOpenAIStoredLaneRuntime(
            provider=provider
        )
        selected_retention = _stored_retention()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(lane_runtime,),
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=store,
        authority=scope,
        lane=provider.binding,
        retention=selected_retention,
        id_namespace=namespace,
    )
    return avalan.DirectConversationClient(runtime), coordinator


def _handle_values(
    handle: avalan.StatelessConversationHandle,
) -> tuple[str, str, str]:
    return (
        str(handle.conversation_id),
        str(handle.checkpoint_id),
        str(handle.branch_id),
    )


def _parent(values: tuple[str, str, str]) -> avalan.StatelessParent:
    return avalan.StatelessParent(
        handle=avalan.StatelessConversationHandle(
            conversation_id=conversation.ConversationId(values[0]),
            checkpoint_id=conversation.CheckpointId(values[1]),
            branch_id=conversation.ConversationBranchId(values[2]),
        )
    )


async def _child_continue(
    dsn: str,
    schema: str,
    endpoint: str,
    lane_id: str,
    handle_values: tuple[str, str, str],
    input_text: str,
    namespace: str,
) -> tuple[str, tuple[str, str, str]]:
    store = _store(dsn, schema)
    provider = _stateless_provider(_binding(endpoint, lane_id))
    client, coordinator = _client(store, provider, namespace)
    try:
        await store.open()
        result = await client.continue_conversation(
            input_text,
            avalan.StatelessConversationSettings(
                parent=_parent(handle_values)
            ),
        )
        assert type(result.handle) is avalan.StatelessConversationHandle
        return result.output, _handle_values(result.handle)
    finally:
        await coordinator.close()
        await store.close()


def _child_target(
    dsn: str,
    schema: str,
    endpoint: str,
    lane_id: str,
    handle_values: tuple[str, str, str],
    input_text: str,
    namespace: str,
    connection: Connection,
) -> None:
    try:
        payload = run(
            _child_continue(
                dsn,
                schema,
                endpoint,
                lane_id,
                handle_values,
                input_text,
                namespace,
            )
        )
        connection.send((True, payload))
    except BaseException as error:
        connection.send((False, (type(error).__name__, str(error))))
    finally:
        connection.close()


async def _spawn_child(
    dsn: str,
    schema: str,
    endpoint: str,
    lane_id: str,
    handle_values: tuple[str, str, str],
    input_text: str,
    namespace: str,
) -> tuple[str, tuple[str, str, str]]:
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_target,
        args=(
            dsn,
            schema,
            endpoint,
            lane_id,
            handle_values,
            input_text,
            namespace,
            child_connection,
        ),
    )
    process.start()
    child_connection.close()
    await to_thread(process.join, 45)
    if process.is_alive():
        process.terminate()
        await to_thread(process.join, 5)
        pytest.fail("fresh-process compact continuation timed out")
    assert process.exitcode == 0
    success, payload = parent_connection.recv()
    parent_connection.close()
    assert success, payload
    return cast(tuple[str, tuple[str, str, str]], payload)


def _input_items(request: _WireRequest) -> list[dict[str, object]]:
    items = request.payload["input"]
    assert type(items) is list
    assert all(type(item) is dict for item in items)
    return cast(list[dict[str, object]], items)


async def test_long_stateless_inline_compaction_restarts_at_latest_boundary(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Restart from one exact latest compact item and retained suffix."""
    record_property("conversation_acceptance_evidence", "database")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("long")
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"
    store = _store(dsn, schema)
    provider = _stateless_provider(_binding(endpoint, "lane-compact-long"))
    client, coordinator = _client(store, provider, "compact-long-parent")
    try:
        await store.open()
        result = await client.create(
            "long input 1",
            avalan.StatelessConversationSettings(),
        )
        assert type(result.handle) is avalan.StatelessConversationHandle
        handle = result.handle
        for number in range(2, 5):
            result = await client.continue_conversation(
                f"long input {number}",
                avalan.StatelessConversationSettings(
                    parent=avalan.StatelessParent(handle=handle)
                ),
            )
            assert type(result.handle) is avalan.StatelessConversationHandle
            handle = result.handle
        compacted = await client.continue_conversation(
            "long input 5",
            avalan.StatelessConversationSettings(
                parent=avalan.StatelessParent(handle=handle),
                compaction=avalan.InlineCompaction(compact_threshold=128),
            ),
        )
        assert type(compacted.handle) is avalan.StatelessConversationHandle
        await coordinator.close()
        await store.close()
        restarted = await _spawn_child(
            dsn,
            schema,
            endpoint,
            "lane-compact-long",
            _handle_values(compacted.handle),
            "after restart input",
            "compact-long-child",
        )
        assert restarted[0] == "after-restart"
        assert server.requests[4].payload["context_management"] == [
            {"compact_threshold": 128, "type": "compaction"}
        ]
        replay = _input_items(server.requests[5])
        assert [item.get("id") for item in replay[:2]] == [
            "long-compact",
            "long-message-compacted",
        ]
        assert replay[0]["encrypted_content"] == "long-compact-private"
        assert "created_by" not in replay[0]
        serialized = dumps(replay)
        assert "long-reasoning-1" not in serialized
        assert "long-message-4" not in serialized
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_tool_cycles_across_two_boundaries_keep_exact_final_order(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Retain every post-boundary call/output pair in provider order."""
    record_property("conversation_acceptance_evidence", "wire")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("tools")
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"
    store = _store(dsn, schema)
    provider = _stateless_provider(
        _binding(endpoint, "lane-compact-tools"),
        tools=True,
    )
    client, coordinator = _client(store, provider, "compact-tools")
    try:
        await store.open()
        first = await client.create(
            "first tool input",
            avalan.StatelessConversationSettings(
                compaction=avalan.InlineCompaction(compact_threshold=128)
            ),
        )
        assert type(first.handle) is avalan.StatelessConversationHandle
        second = await client.continue_conversation(
            "second tool input",
            avalan.StatelessConversationSettings(
                parent=avalan.StatelessParent(handle=first.handle),
                compaction=avalan.InlineCompaction(compact_threshold=128),
            ),
        )
        assert type(second.handle) is avalan.StatelessConversationHandle
        third = await client.continue_conversation(
            "third tool input",
            avalan.StatelessConversationSettings(
                parent=avalan.StatelessParent(handle=second.handle)
            ),
        )
        assert third.output == "third tool turn"
        final_replay = _input_items(server.requests[6])
        assert [item["type"] for item in final_replay[:-1]] == [
            "compaction",
            "function_call",
            "function_call_output",
            "message",
        ]
        assert final_replay[0]["id"] == "tools-compact-two"
        assert final_replay[1]["call_id"] == "call-post-two"
        assert final_replay[2]["call_id"] == "call-post-two"
        assert final_replay[3]["id"] == "tools-final-two"
        serialized = dumps(final_replay)
        assert "tools-compact-one" not in serialized
        assert "call-pre-two" not in serialized
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_stored_inline_compaction_uses_only_immediate_upstream_parent(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Keep stored history provider-managed across inline compaction."""
    record_property("conversation_acceptance_evidence", "database")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("stored")
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"
    store = _store(dsn, schema)
    provider = _stored_provider(
        _binding(endpoint, "lane-compact-stored", stored=True)
    )
    client, coordinator = _client(store, provider, "compact-stored")
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True,
        compaction=avalan.InlineCompaction(compact_threshold=128),
    )
    try:
        await store.open()
        first = await client.create("stored first", settings)
        assert type(first.handle) is avalan.StoredConversationHandle
        parent = avalan.StoredParent(handle=first.handle)
        second = await client.continue_conversation(
            "stored second",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=parent,
                compaction=avalan.InlineCompaction(compact_threshold=128),
            ),
        )
        assert second.output == "stored turn 2"
        request = server.requests[1].payload
        assert request["previous_response_id"] == "stored-response-1"
        assert request["context_management"] == [
            {"compact_threshold": 128, "type": "compaction"}
        ]
        items = _input_items(server.requests[1])
        assert items == [_input_message("stored second")]
        serialized = dumps(request)
        assert "stored-private-1" not in serialized
        assert "stored-message-1" not in serialized
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def _verify_standalone_fork_restart_and_original_branch(
    pgsql_schema: tuple[str, str],
) -> tuple[str, str]:
    """Persist an explicit compact fork while preserving its source parent."""
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("standalone")
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"
    store = _store(dsn, schema)
    binding = _binding(endpoint, "lane-compact-standalone")
    provider = _stateless_provider(binding)
    client, coordinator = _client(store, provider, "compact-standalone")
    scope = authority()
    try:
        await store.open()
        original = await client.create(
            "original input",
            avalan.StatelessConversationSettings(),
        )
        assert type(original.handle) is avalan.StatelessConversationHandle
        original_checkpoint = await store.load(
            original.handle.checkpoint_id,
            scope,
        )
        compacted = await client.compact(
            avalan.StandaloneCompactRequest(
                parent=avalan.StatelessParent(handle=original.handle)
            )
        )
        fork = await client.fork_compact(
            compacted,
            conversation.ConversationBranchId("explicit-compact-fork"),
        )
        await coordinator.close()
        await store.close()
        restarted = await _spawn_child(
            dsn,
            schema,
            endpoint,
            "lane-compact-standalone",
            _handle_values(fork),
            "continue compact fork",
            "compact-standalone-child",
        )
        assert restarted[0] == "fork continued"

        reopened_store = _store(dsn, schema)
        reopened_provider = _stateless_provider(binding)
        reopened_client, reopened_coordinator = _client(
            reopened_store,
            reopened_provider,
            "compact-standalone-original-branch",
        )
        try:
            await reopened_store.open()
            parent = avalan.StatelessParent(handle=original.handle)
            branched = await reopened_client.branch(
                "branch original parent",
                avalan.StatelessConversationSettings(
                    parent=parent,
                    branch=avalan.ConversationBranchIntent(
                        parent=parent,
                        branch_id=conversation.ConversationBranchId(
                            "original-parent-branch"
                        ),
                    ),
                ),
            )
            assert branched.output == "original branched"
            assert (
                await reopened_store.load(
                    original.handle.checkpoint_id,
                    scope,
                )
                == original_checkpoint
            )
        finally:
            await reopened_coordinator.close()
            await reopened_store.close()

        assert server.requests[1].path == "/v1/responses/compact"
        fork_replay = _input_items(server.requests[2])
        assert [item["type"] for item in fork_replay[:-1]] == [
            "message",
            "compaction",
        ]
        assert fork_replay[1]["id"] == "standalone-compact"
        assert (
            fork_replay[1]["encrypted_content"] == "standalone-compact-private"
        )
        original_replay = _input_items(server.requests[3])
        assert original_replay[0]["id"] == "standalone-original-reasoning"
        assert "standalone-compact" not in dumps(original_replay)
        return restarted[0], branched.output
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_pgsql_named_head_compact_commit_closes_atomic_validation(
    pgsql_schema: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit one compact head and reject every malformed atomic candidate."""
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("standalone")
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"
    store = _store(dsn, schema)
    provider = _stateless_provider(
        _binding(endpoint, "lane-compact-atomic-head")
    )
    client, coordinator = _client(store, provider, "compact-atomic-head")
    scope = authority()
    try:
        await store.open()
        root = await client.create(
            "atomic head root",
            avalan.StatelessConversationSettings(),
        )
        assert type(root.handle) is avalan.StatelessConversationHandle
        head_id = conversation.NamedHeadId("compact-atomic-head")
        await store.create_head(
            conversation.NamedHeadSnapshot(
                head_id=head_id,
                revision=conversation.NamedHeadRevision(0),
                checkpoint_id=root.handle.checkpoint_id,
            ),
            scope,
        )
        parent = avalan.StatelessParent(handle=root.handle)
        head_parent = avalan.NamedHeadParent(
            head_id=head_id,
            expected_revision=conversation.NamedHeadRevision(0),
            parent=parent,
        )
        request = avalan.StandaloneCompactRequest(
            parent=parent,
            named_head=head_parent,
        )
        first = await client.compact(request)

        original_create = (
            conversation.PgsqlConversationStore.create_with_named_head
        )
        captured: list[
            tuple[
                conversation.CheckpointCandidate,
                conversation.NamedHeadAdvance,
            ]
        ] = []

        async def capture_candidate(
            active: conversation.PgsqlConversationStore,
            candidate: conversation.CheckpointCandidate,
            advance: conversation.NamedHeadAdvance,
        ) -> conversation.ConversationCheckpoint:
            if active is store:
                captured.append((candidate, advance))
            return await original_create(active, candidate, advance)

        monkeypatch.setattr(
            conversation.PgsqlConversationStore,
            "create_with_named_head",
            capture_candidate,
        )
        committed = await client.commit_compact(first)
        assert (await store.load_head(head_id, scope)).checkpoint_id == (
            committed.checkpoint_id
        )
        assert len(captured) == 1
        candidate, advance = captured[0]
        assert type(candidate) is (
            conversation.ExecutionSegmentCheckpointCandidate
        )
        monkeypatch.setattr(
            conversation.PgsqlConversationStore,
            "create_with_named_head",
            original_create,
        )

        with pytest.raises(conversation.ConversationConflictError):
            await original_create(store, candidate, advance)

        with pytest.raises(conversation.ConversationValidationError):
            await original_create(
                store,
                candidate,
                cast(conversation.NamedHeadAdvance, object()),
            )

        missing_head = conversation.ExecutionSegmentCheckpointCandidate(
            checkpoint=conversation.with_checkpoint_integrity(
                replace(candidate.checkpoint, head=None, integrity=None)
            )
        )
        with pytest.raises(conversation.ConversationValidationError):
            await original_create(store, missing_head, advance)

        compact_source = await store.load(first.handle.checkpoint_id, scope)
        root_id = compact_source.identity.parent_checkpoint_id
        root_sequence = compact_source.identity.parent_sequence
        assert root_id is not None and root_sequence is not None
        ordinary_parent_identity = replace(
            candidate.checkpoint.identity,
            parent_checkpoint_id=root_id,
            parent_sequence=root_sequence,
            sequence=conversation.CheckpointSequence(root_sequence + 1),
        )
        ordinary_parent = conversation.ExecutionSegmentCheckpointCandidate(
            checkpoint=conversation.with_checkpoint_integrity(
                replace(
                    candidate.checkpoint,
                    identity=ordinary_parent_identity,
                    integrity=None,
                )
            )
        )
        with pytest.raises(conversation.ConversationValidationError):
            await original_create(
                store,
                ordinary_parent,
                replace(
                    advance,
                    parent_checkpoint_id=conversation.CheckpointId(
                        "unrelated-compact-grandparent"
                    ),
                ),
            )
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_standalone_explicit_fork_restarts_and_original_parent_branches(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Persist an explicit compact fork while preserving its source parent."""
    record_property("conversation_acceptance_evidence", "database")
    await _verify_standalone_fork_restart_and_original_branch(pgsql_schema)


async def test_standalone_fork_restart_and_original_branch_remain_exact(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Preserve standalone fork restart and original-parent branching."""
    record_property("conversation_acceptance_evidence", "database")
    assert await _verify_standalone_fork_restart_and_original_branch(
        pgsql_schema
    ) == ("fork continued", "original branched")
