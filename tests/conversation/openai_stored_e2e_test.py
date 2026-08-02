"""Exercise native stored chaining over real TCP and PostgreSQL."""

from asyncio import (
    CancelledError,
    IncompleteReadError,
    Server,
    StreamReader,
    StreamWriter,
    gather,
    run,
    start_server,
    to_thread,
)
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from multiprocessing import get_context
from multiprocessing.connection import Connection
from os import environ
from typing import cast
from uuid import uuid4

import httpx
import pytest
from openai import AsyncOpenAI
from phase2_fixtures import authority
from store_conformance_test import _stored_atomic_commit

import avalan
import avalan.conversation as conversation
from avalan.conversation import sdk as sdk_module
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

_ADAPTER = (
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
    """Run the stored wire contract under asyncio only."""
    return "asyncio"


def _key() -> conversation.ConversationDataKey:
    return conversation.ConversationDataKey(
        key_id="phase6-native-stored-key",
        revision=1,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"6" * 32,
    )


def _key_resolver() -> conversation.InMemoryConversationKeyResolver:
    scope = authority()
    return conversation.InMemoryConversationKeyResolver(
        {conversation.authority_digest(scope): (_key(),)}
    )


def _store(
    dsn: str,
    schema: str,
    *,
    fault_hook: conversation.PgsqlConversationFaultHook | None = None,
    policy: conversation.PgsqlConversationStorePolicy | None = None,
) -> conversation.PgsqlConversationStore:
    return conversation.PgsqlConversationStore.from_settings(
        conversation.PgsqlConversationStoreSettings(
            dsn=dsn,
            schema=schema,
            pool_minimum=1,
            pool_maximum=2,
        ),
        key_resolver=_key_resolver(),
        cipher=conversation.AesGcmConversationCipher(),
        clock=conversation.DeterministicFakeClock(_NOW),
        fault_hook=fault_hook,
        policy=policy or conversation.PgsqlConversationStorePolicy(),
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
    """Yield one migrated isolated schema for stored provider evidence."""
    assert _DSN is not None
    schema = f"conv_phase6_stored_{uuid4().hex}"
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    try:
        yield _DSN, schema
    finally:
        await _drop_schema(_DSN, schema)


def _message(identifier: str, text: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_call(identifier: str, call_id: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": "lookup",
        "arguments": '{"value":1}',
    }


def _response(
    identifier: str,
    output: list[dict[str, object]],
    *,
    previous_response_id: str | None,
) -> dict[str, object]:
    return {
        "id": identifier,
        "object": "response",
        "created_at": 1.0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": "Use the exact Phase 6 stored execution.",
        "max_output_tokens": 512,
        "max_tool_calls": 4,
        "model": "gpt-5",
        "output": output,
        "parallel_tool_calls": False,
        "previous_response_id": previous_response_id,
        "reasoning": {"context": "current_turn"},
        "safety_identifier": "avalan-conversation",
        "store": True,
        "temperature": 0.2,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 0.8,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 10,
        },
    }


@dataclass(slots=True)
class _WireRequest:
    method: str
    path: str
    body: bytes
    payload: dict[str, object] | None


class _LoopbackTcpTransport(httpx.AsyncBaseTransport):
    """Route an unchanged classified URL through a loopback TCP socket."""

    def __init__(self, loopback_base_url: str) -> None:
        target = httpx.URL(loopback_base_url)
        assert target.scheme == "http"
        assert target.host in {"127.0.0.1", "localhost"}
        assert target.port is not None
        self._target = target
        self._transport = httpx.AsyncHTTPTransport()
        self.request_urls: list[str] = []

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        """Send one exact classified request over the loopback transport."""
        self.request_urls.append(str(request.url))
        mapped_url = request.url.copy_with(
            scheme=self._target.scheme,
            host=self._target.host,
            port=self._target.port,
        )
        mapped_request = httpx.Request(
            method=request.method,
            url=mapped_url,
            headers=request.headers,
            stream=request.stream,
            extensions=request.extensions,
        )
        return await self._transport.handle_async_request(mapped_request)

    async def aclose(self) -> None:
        """Close the owned real TCP transport."""
        await self._transport.aclose()


@dataclass(slots=True)
class _ScriptedTcpProvider:
    scenario: str
    delete_failures_remaining: int = 0
    requests: list[_WireRequest] = field(default_factory=list)
    responses: dict[str, dict[str, object]] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    _server: Server | None = None
    base_url: str | None = None

    async def start(self) -> None:
        """Start one loopback HTTP/1.1 provider server."""
        server = await start_server(self._handle, "127.0.0.1", 0)
        socket = server.sockets[0]
        port = cast(tuple[str, int], socket.getsockname())[1]
        self._server = server
        self.base_url = f"http://127.0.0.1:{port}"

    async def close(self) -> None:
        """Close the loopback provider server."""
        server = self._server
        assert server is not None
        server.close()
        await server.wait_closed()

    def _post_response(
        self,
        payload: Mapping[str, object],
        index: int,
    ) -> dict[str, object]:
        previous = payload.get("previous_response_id")
        previous_id = previous if type(previous) is str else None
        if self.scenario == "restart":
            identifier = f"private-restart-{index + 1}"
            return _response(
                identifier,
                [
                    _message(
                        f"restart-message-{index + 1}", f"restart-{index + 1}"
                    )
                ],
                previous_response_id=previous_id,
            )
        if self.scenario == "tools":
            outputs = (
                [_function_call("tool-call-one", "call-one")],
                [_message("tool-message-one", "tool-first")],
                [_message("tool-message-two", "tool-second")],
            )
            return _response(
                f"private-tool-{index + 1}",
                outputs[index],
                previous_response_id=previous_id,
            )
        if self.scenario == "branches":
            suffix = ("parent", "alpha", "beta")[index]
            return _response(
                f"private-branch-{suffix}",
                [_message(f"branch-message-{suffix}", suffix)],
                previous_response_id=previous_id,
            )
        if self.scenario == "quarantine":
            return _response(
                f"private-quarantine-{index + 1}",
                [
                    _message(
                        f"quarantine-message-{index + 1}",
                        f"quarantine-{index + 1}",
                    )
                ],
                previous_response_id=previous_id,
            )
        if self.scenario == "head":
            suffix = ("root", "alpha", "beta")[index]
            return _response(
                f"private-head-{suffix}",
                [_message(f"head-message-{suffix}", suffix)],
                previous_response_id=previous_id,
            )
        if self.scenario == "alias":
            identifier = "direct-phase6-alias-pg-1-create-response"
            return _response(
                identifier,
                [_message("alias-message", "alias")],
                previous_response_id=previous_id,
            )
        if self.scenario == "checkpoint_alias":
            identifier = (
                "direct-phase6-checkpoint-alias-pg-1-create-checkpoint"
            )
            return _response(
                identifier,
                [_message("checkpoint-alias-message", "alias")],
                previous_response_id=previous_id,
            )
        if self.scenario == "azure":
            response = _response(
                f"private-azure-{index + 1}",
                [_message(f"azure-message-{index + 1}", "azure-stored")],
                previous_response_id=previous_id,
            )
            response["model"] = "deployment-stored"
            return response
        raise AssertionError("unknown Phase 6 scenario")

    async def _handle(
        self,
        reader: StreamReader,
        writer: StreamWriter,
    ) -> None:
        try:
            header = await reader.readuntil(b"\r\n\r\n")
        except IncompleteReadError:
            writer.close()
            await writer.wait_closed()
            return
        lines = header.decode("ascii").split("\r\n")
        method, path, _ = lines[0].split(" ", 2)
        lengths = [
            int(line.split(":", 1)[1].strip())
            for line in lines[1:]
            if line.casefold().startswith("content-length:")
        ]
        length = lengths[0] if lengths else 0
        body = await reader.readexactly(length) if length else b""
        payload: dict[str, object] | None = None
        if body:
            decoded = loads(body)
            assert isinstance(decoded, dict)
            payload = {str(key): value for key, value in decoded.items()}
        request = _WireRequest(
            method=method,
            path=path,
            body=body,
            payload=payload,
        )
        self.requests.append(request)
        status = "200 OK"
        content_type = "application/json"
        response_body = b""
        if method == "POST":
            assert payload is not None
            post_index = (
                sum(item.method == "POST" for item in self.requests) - 1
            )
            response = self._post_response(payload, post_index)
            identifier = cast(str, response["id"])
            self.responses[identifier] = response
            if payload.get("stream") is True:
                output = cast(list[dict[str, object]], response["output"])
                events = [
                    {
                        "type": "response.output_item.done",
                        "sequence_number": index,
                        "output_index": index,
                        "item": item,
                    }
                    for index, item in enumerate(output)
                ]
                events.append(
                    {
                        "type": "response.completed",
                        "sequence_number": len(events),
                        "response": response,
                    }
                )
                response_body = (
                    "".join(f"data: {dumps(event)}\n\n" for event in events)
                    + "data: [DONE]\n\n"
                ).encode()
                content_type = "text/event-stream"
            else:
                response_body = dumps(response).encode()
        else:
            identifier = path.rsplit("/", 1)[-1]
            if method == "GET" and identifier in self.responses:
                response_body = dumps(self.responses[identifier]).encode()
            elif method == "DELETE":
                if self.delete_failures_remaining:
                    self.delete_failures_remaining -= 1
                    status = "500 Internal Server Error"
                    response_body = dumps(
                        {"error": {"message": "temporary deletion outage"}}
                    ).encode()
                else:
                    self.deleted.add(identifier)
                    status = "204 No Content"
            else:
                status = "404 Not Found"
                response_body = dumps(
                    {"error": {"message": "missing"}}
                ).encode()
        writer.write(
            (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(response_body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            + response_body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()


def _binding(
    endpoint: str,
    lane_id: str,
    *,
    streaming: bool,
    azure: bool = False,
    azure_resource_identity: str | None = None,
    tools: bool = False,
) -> conversation.ProviderLaneBinding:
    binding = conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=_ADAPTER,
        provider_family=(
            conversation.ProviderFamily.AZURE_OPENAI
            if azure
            else conversation.ProviderFamily.OPENAI
        ),
        normalized_endpoint=endpoint,
        azure_resource_identity=(
            azure_resource_identity or "phase6-resource" if azure else None
        ),
        model_or_deployment="deployment-stored" if azure else "gpt-5",
        provider_api_revision=conversation.ProviderApiRevision(
            "azure-openai-v1" if azure else "openapi-2.3.0"
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=conversation.ModelConfigurationRevision(
            "model-config-phase6"
        ),
        capability_profile_revision=conversation.CapabilityProfileRevision(
            "capability-phase6"
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-phase6"),
        execution_definition_revision=conversation.ExecutionDefinitionRevision(
            "execution-phase6"
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=(
            conversation.ProviderTransport.STREAMING
            if streaming
            else conversation.ProviderTransport.NON_STREAMING
        ),
        agent_id=authority().agent_id,
    )
    configured_tools = (
        (
            conversation.NativeOpenAIFunctionTool(
                name="lookup",
                description="Return one deterministic test value.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ("value",),
                    "additionalProperties": False,
                },
                handler=_lookup,
            ),
        )
        if tools
        else ()
    )
    execution = conversation.NativeOpenAIStoredExecution(
        instructions="Use the exact Phase 6 stored execution.",
        max_output_tokens=512,
        max_tool_calls=4,
        parallel_tool_calls=False,
        temperature=0.2,
        top_p=0.8,
        truncation="disabled",
    )
    encrypted_content = (
        conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
        if azure
        else conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
    )
    return replace(
        binding,
        execution_definition_digest=(
            conversation.native_openai_stored_execution_digest(
                binding=binding,
                execution=execution,
                encrypted_content=encrypted_content,
                tools=configured_tools,
            )
        ),
    )


def _capabilities(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ConversationCapabilityProfile:
    supported = {
        conversation.ConversationCapability.STORED_RESPONSES_CHAINING,
        conversation.ConversationCapability.STORED_RESPONSE_RETRIEVAL,
        conversation.ConversationCapability.STORED_RESPONSE_DELETION,
        conversation.ConversationCapability.REASONING_CONTEXT_CURRENT_TURN,
        conversation.ConversationCapability.REASONING_CONTEXT_ALL_TURNS,
    }
    if binding.transport is conversation.ProviderTransport.STREAMING:
        supported.add(
            conversation.ConversationCapability.STREAMING_ITEM_FIDELITY
        )
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId(
            f"stored-profile-{binding.lane_id}"
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
                    (f"phase6-{capability.value}",)
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


def _provider(
    binding: conversation.ProviderLaneBinding,
    *,
    tools: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> conversation.NativeOpenAIStoredProvider:
    client = AsyncOpenAI(
        api_key="phase6-test-key",
        base_url=binding.normalized_endpoint,
        http_client=(
            httpx.AsyncClient(transport=transport)
            if transport is not None
            else None
        ),
        max_retries=0,
    )
    configured_tools = (
        (
            conversation.NativeOpenAIFunctionTool(
                name="lookup",
                description="Return one deterministic test value.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ("value",),
                    "additionalProperties": False,
                },
                handler=_lookup,
            ),
        )
        if tools
        else ()
    )
    return conversation.NativeOpenAIStoredProvider(
        client=client,
        profile=conversation.NativeOpenAIStoredProfile(
            profile_id=f"stored-{binding.lane_id}",
            binding=binding,
            execution=conversation.NativeOpenAIStoredExecution(
                instructions="Use the exact Phase 6 stored execution.",
                max_output_tokens=512,
                max_tool_calls=4,
                parallel_tool_calls=False,
                temperature=0.2,
                top_p=0.8,
                truncation="disabled",
            ),
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
                if binding.provider_family
                is conversation.ProviderFamily.AZURE_OPENAI
                else (
                    conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
                )
            ),
            scripted_tcp_test=transport is None,
        ),
        capability_profile=_capabilities(binding),
        tools=configured_tools,
    )


async def _resolver_clock() -> datetime:
    return _NOW


def _retention() -> conversation.RetentionLimits:
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
    provider: conversation.NativeOpenAIStoredProvider,
    namespace: str,
    *,
    boundary_hook: conversation.CoordinatorBoundaryHook | None = None,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.RunScopedConversationCoordinator,
]:
    scope = authority()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(conversation.NativeOpenAIStoredLaneRuntime(provider=provider),),
        boundary_hook=boundary_hook,
    )
    resolver = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=provider,
                revision="phase6-resolver",
                valid_from=_NOW - timedelta(minutes=1),
                valid_until=_NOW + timedelta(days=1),
            ),
        ),
        clock=_resolver_clock,
    )
    reconciler = conversation.ProviderLifecycleReconciler(
        store=store,
        resolver=resolver,
        authority=scope,
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=store,
        authority=scope,
        lane=provider.binding,
        retention=_retention(),
        id_namespace=namespace,
        provider_resolver=resolver,
        lifecycle_reconciler=reconciler,
    )
    return avalan.DirectConversationClient(runtime), coordinator


def _handle_values(
    handle: avalan.StoredConversationHandle,
) -> tuple[str, str, str, str]:
    assert handle.public_response_id is not None
    return (
        str(handle.conversation_id),
        str(handle.checkpoint_id),
        str(handle.branch_id),
        str(handle.public_response_id),
    )


def _parent(values: tuple[str, str, str, str]) -> avalan.StoredParent:
    return avalan.StoredParent(
        handle=avalan.StoredConversationHandle(
            conversation_id=conversation.ConversationId(values[0]),
            checkpoint_id=conversation.CheckpointId(values[1]),
            branch_id=conversation.ConversationBranchId(values[2]),
            public_response_id=conversation.PublicResponseId(values[3]),
        )
    )


async def _child_restart(
    dsn: str,
    schema: str,
    endpoint: str,
    handle_values: tuple[str, str, str, str],
) -> tuple[str, str, bool, str]:
    store = _store(dsn, schema)
    provider = _provider(_binding(endpoint, "lane-restart", streaming=False))
    client, coordinator = _client(store, provider, "phase6-restart-child")
    try:
        await store.open()
        result = await client.continue_conversation(
            "restart second",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=_parent(handle_values),
            ),
        )
        assert type(result.handle) is avalan.StoredConversationHandle
        assert result.handle.public_response_id is not None
        retrieved = await client.retrieve(result.handle.public_response_id)
        deleted = await client.delete(result.handle.public_response_id)
        return (
            result.output,
            retrieved.output,
            deleted.upstream_pending,
            str(result.handle.public_response_id),
        )
    finally:
        await coordinator.close()
        await store.close()


def _child_target(
    dsn: str,
    schema: str,
    endpoint: str,
    handle_values: tuple[str, str, str, str],
    connection: Connection,
) -> None:
    try:
        connection.send(
            (
                True,
                run(_child_restart(dsn, schema, endpoint, handle_values)),
            )
        )
    except BaseException as error:
        connection.send((False, (type(error).__name__, str(error))))
    finally:
        connection.close()


async def _spawn_child(
    dsn: str,
    schema: str,
    endpoint: str,
    handle_values: tuple[str, str, str, str],
) -> tuple[str, str, bool, str]:
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_target,
        args=(dsn, schema, endpoint, handle_values, child_connection),
    )
    process.start()
    child_connection.close()
    await to_thread(process.join, 45)
    if process.is_alive():
        process.terminate()
        await to_thread(process.join, 5)
        pytest.fail("Phase 6 fresh-process continuation timed out")
    assert process.exitcode == 0
    success, payload = parent_connection.recv()
    parent_connection.close()
    assert success, payload
    return cast(tuple[str, str, bool, str], payload)


async def _stream_result(
    stream: avalan.DirectConversationStream,
) -> avalan.DirectConversationResult:
    events = [event async for event in stream]
    terminal = events[-1]
    assert type(terminal) is avalan.DirectConversationStreamTerminal
    return terminal.result


def _post_payloads(server: _ScriptedTcpProvider) -> list[dict[str, object]]:
    return [
        request.payload
        for request in server.requests
        if request.method == "POST" and request.payload is not None
    ]


def _assert_frozen(payloads: list[dict[str, object]]) -> None:
    for payload in payloads:
        assert payload["store"] is True
        assert (
            payload["instructions"]
            == "Use the exact Phase 6 stored execution."
        )
        assert payload["max_output_tokens"] == 512
        assert payload["max_tool_calls"] == 4
        assert payload["parallel_tool_calls"] is False
        assert payload["temperature"] == 0.2
        assert payload["top_p"] == 0.8
        assert payload["truncation"] == "disabled"


class _CommitFailureHook:
    def __init__(self) -> None:
        self.failed = False

    async def reach(
        self,
        point: conversation.PgsqlConversationFaultPoint,
    ) -> None:
        if (
            not self.failed
            and point.boundary
            is conversation.PgsqlConversationFaultBoundary.COMMIT_BEFORE
            and point.operation == "checkpoint_atomic_commit"
        ):
            self.failed = True
            raise conversation.ConversationStorageError()


class _StreamCloseFailureHook:
    def __init__(self, *, cancel: bool) -> None:
        self.cancel = cancel
        self.failed = False

    async def reach(
        self,
        boundary: conversation.CoordinatorAwaitBoundary,
    ) -> None:
        if (
            not self.failed
            and boundary
            is conversation.CoordinatorAwaitBoundary.PROVIDER_STREAM_CLOSE
        ):
            self.failed = True
            if self.cancel:
                raise CancelledError()
            raise conversation.ConversationCommitError()


@pytest.mark.parametrize("streaming", (False, True), ids=("sync", "stream"))
async def test_native_azure_stored_exact_identity_over_loopback_transport(
    pgsql_schema: tuple[str, str],
    streaming: bool,
) -> None:
    """Keep exact stored Azure identity over loopback TCP and PostgreSQL."""
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("azure")
    await server.start()
    assert server.base_url is not None
    exact_resource = "resource.openai.azure.com"
    exact_endpoint = f"https://{exact_resource}/openai/v1"
    loopback_transport = _LoopbackTcpTransport(server.base_url)
    store = _store(dsn, schema)
    binding = _binding(
        exact_endpoint,
        f"lane-azure-stored-{'stream' if streaming else 'sync'}",
        streaming=streaming,
        azure=True,
        azure_resource_identity=exact_resource,
    )
    provider = _provider(binding, transport=loopback_transport)
    client, coordinator = _client(
        store,
        provider,
        f"phase6-azure-stored-{'stream' if streaming else 'sync'}",
    )
    try:
        await store.open()
        with pytest.raises(conversation.ConversationValidationError):
            await store.prepare_deletion(
                conversation.PublicResponseId("phase6-invalid-authority"),
                cast(conversation.AuthorityScope, object()),
            )
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.prepare_deletion(
                conversation.PublicResponseId("phase6-unknown-response"),
                authority(),
            )
        created = await client.create(
            "azure stored exact wire",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
            stream=streaming,
        )
        result = (
            await _stream_result(created)
            if type(created) is avalan.DirectConversationStream
            else created
        )
        assert type(result) is avalan.DirectConversationResult
        assert result.output == "azure-stored"
        assert binding.normalized_endpoint == exact_endpoint
        assert binding.azure_resource_identity == exact_resource
        assert binding.model_or_deployment == "deployment-stored"
        assert binding.provider_api_revision == "azure-openai-v1"
        assert provider._profile.scripted_tcp_test is False
        assert loopback_transport.request_urls == [
            exact_endpoint + "/responses"
        ]
        assert server.requests[0].path == "/openai/v1/responses"
        payload = server.requests[0].payload
        assert payload is not None
        assert payload["model"] == "deployment-stored"
        assert payload["include"] == ["reasoning.encrypted_content"]
        assert payload["store"] is True
        assert payload["stream"] is streaming
        assert "previous_response_id" not in payload
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_stored_restart_retrieval_and_deletion(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Prove fresh-process chaining, retrieval, and deletion over real I/O."""
    record_property("conversation_acceptance_evidence", "public")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("restart", delete_failures_remaining=1)
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"
    store = _store(dsn, schema)
    provider = _provider(_binding(endpoint, "lane-restart", streaming=False))
    client, coordinator = _client(store, provider, "phase6-restart-parent")
    try:
        await store.open()
        first = await client.create(
            "restart first",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        assert first.output == "restart-1"
        assert type(first.handle) is avalan.StoredConversationHandle
        await coordinator.close()
        await store.close()
        child = await _spawn_child(
            dsn,
            schema,
            endpoint,
            _handle_values(first.handle),
        )
        assert child[:3] == ("restart-2", "restart-2", True)
        cleanup_store = _store(dsn, schema)
        cleanup_provider = _provider(
            _binding(endpoint, "lane-restart", streaming=False)
        )
        cleanup_client, cleanup_coordinator = _client(
            cleanup_store,
            cleanup_provider,
            "phase6-delete-retry",
        )
        try:
            await cleanup_store.open()
            public_response_id = conversation.PublicResponseId(child[3])
            deletion = await cleanup_client.delete(public_response_id)
            repeated_deletion = await cleanup_client.delete(public_response_id)
            assert not deletion.upstream_pending
            assert repeated_deletion == deletion
            assert (
                await cleanup_store.claim_provider_lifecycle(
                    authority(), limit=10
                )
                == ()
            )
            with pytest.raises(conversation.ConversationAuthorizationError):
                await cleanup_store.retrieve(
                    public_response_id,
                    authority(),
                )
        finally:
            await cleanup_coordinator.close()
            await cleanup_store.close()
        payloads = _post_payloads(server)
        assert len(payloads) == 2
        assert "previous_response_id" not in payloads[0]
        assert payloads[1]["previous_response_id"] == "private-restart-1"
        assert payloads[0]["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "restart first"}],
            }
        ]
        assert payloads[1]["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "restart second"}],
            }
        ]
        _assert_frozen(payloads)
        methods = [request.method for request in server.requests]
        assert methods == ["POST", "POST", "GET", "DELETE", "DELETE"]
        assert server.deleted == {"private-restart-2"}
        assert first.handle.public_response_id != "private-restart-1"
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_retired_runtime_continues_after_pgsql_store_restart(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Use retired credentials for an old parent and current ones for roots."""
    record_property("conversation_acceptance_evidence", "database")
    dsn, schema = pgsql_schema
    old_server = _ScriptedTcpProvider("restart")
    new_server = _ScriptedTcpProvider("restart")
    await old_server.start()
    await new_server.start()
    assert old_server.base_url is not None
    assert new_server.base_url is not None
    lane_id = "lane-retired-pgsql-restart"
    old_binding = _binding(
        f"{old_server.base_url}/v1",
        lane_id,
        streaming=False,
    )
    old_store = _store(dsn, schema)
    old_provider = _provider(old_binding)
    old_client, old_coordinator = _client(
        old_store,
        old_provider,
        "phase6-retired-pgsql-old",
    )
    try:
        await old_store.open()
        old_root = await old_client.create(
            "old root",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        assert type(old_root.handle) is avalan.StoredConversationHandle
    finally:
        await old_coordinator.close()
        await old_store.close()

    restarted_store = _store(dsn, schema)
    retired_provider = _provider(old_binding)
    new_binding = _binding(
        f"{new_server.base_url}/v1",
        lane_id,
        streaming=False,
    )
    current_provider = _provider(new_binding)
    resolver = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=retired_provider,
                revision="phase6-retired-pgsql-runtime",
                valid_from=_NOW - timedelta(minutes=1),
                valid_until=_NOW + timedelta(days=1),
                continuation_runtime=(
                    conversation.NativeOpenAIStoredLaneRuntime(
                        provider=retired_provider
                    )
                ),
            ),
        ),
        clock=_resolver_clock,
    )
    coordinator = conversation.RunScopedConversationCoordinator(
        store=restarted_store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            authority()
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.NativeOpenAIStoredLaneRuntime(
                provider=current_provider
            ),
        ),
    )
    client = avalan.DirectConversationClient(
        avalan.DirectConversationRuntime(
            coordinator=coordinator,
            store=restarted_store,
            authority=authority(),
            lane=new_binding,
            retention=_retention(),
            id_namespace="phase6-retired-pgsql-current",
            provider_resolver=resolver,
        )
    )
    try:
        await restarted_store.open()
        continued = await client.continue_conversation(
            "continue old root",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=avalan.StoredParent(handle=old_root.handle),
            ),
        )
        current_root = await client.create(
            "new root",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        assert (continued.output, current_root.output) == (
            "restart-2",
            "restart-1",
        )
        old_payloads = _post_payloads(old_server)
        new_payloads = _post_payloads(new_server)
        assert len(old_payloads) == 2
        assert old_payloads[1]["previous_response_id"] == "private-restart-1"
        assert len(new_payloads) == 1
        assert "previous_response_id" not in new_payloads[0]
    finally:
        await coordinator.close()
        await retired_provider.aclose()
        await restarted_store.close()
        await old_server.close()
        await new_server.close()


async def test_stored_streaming_tool_cycle_uses_terminal_id(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Use the final internal response ID for the next outward turn."""
    record_property("conversation_acceptance_evidence", "wire")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("tools")
    await server.start()
    assert server.base_url is not None
    store = _store(dsn, schema)
    binding = _binding(
        f"{server.base_url}/v1",
        "lane-tools",
        streaming=True,
        tools=True,
    )
    provider = _provider(binding, tools=True)
    client, coordinator = _client(store, provider, "phase6-tools")
    try:
        await store.open()
        first_stream = await client.create(
            "tool first",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
            stream=True,
        )
        first = await _stream_result(first_stream)
        assert first.output == "tool-first"
        assert type(first.handle) is avalan.StoredConversationHandle
        second_stream = await client.continue_conversation(
            "tool second",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=avalan.StoredParent(handle=first.handle),
            ),
            stream=True,
        )
        second = await _stream_result(second_stream)
        assert second.output == "tool-second"
        payloads = _post_payloads(server)
        assert len(payloads) == 3
        assert payloads[1]["previous_response_id"] == "private-tool-1"
        assert payloads[1]["input"] == [
            {
                "type": "function_call_output",
                "call_id": "call-one",
                "output": "lookup-result",
            }
        ]
        assert payloads[2]["previous_response_id"] == "private-tool-2"
        _assert_frozen(payloads)
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_stored_branches_reuse_immutable_parent(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Create two provider branches from one immutable private parent."""
    record_property("conversation_acceptance_evidence", "database")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("branches")
    await server.start()
    assert server.base_url is not None
    store = _store(dsn, schema)
    provider = _provider(
        _binding(f"{server.base_url}/v1", "lane-branches", streaming=False)
    )
    client, coordinator = _client(store, provider, "phase6-branches")
    try:
        await store.open()
        root = await client.create(
            "branch parent",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        assert type(root.handle) is avalan.StoredConversationHandle
        parent = avalan.StoredParent(handle=root.handle)
        before = await store.load(root.handle.checkpoint_id, authority())
        alpha = await client.branch(
            "branch alpha",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=parent,
                branch=avalan.ConversationBranchIntent(
                    parent=parent,
                    branch_id=conversation.ConversationBranchId(
                        "phase6-alpha"
                    ),
                ),
            ),
        )
        beta = await client.branch(
            "branch beta",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=parent,
                branch=avalan.ConversationBranchIntent(
                    parent=parent,
                    branch_id=conversation.ConversationBranchId("phase6-beta"),
                ),
            ),
        )
        assert (alpha.output, beta.output) == ("alpha", "beta")
        assert (
            await store.load(root.handle.checkpoint_id, authority()) == before
        )
        payloads = _post_payloads(server)
        assert payloads[1]["previous_response_id"] == "private-branch-parent"
        assert payloads[2]["previous_response_id"] == "private-branch-parent"
        assert b"branch beta" not in server.requests[1].body
        assert b"branch alpha" not in server.requests[2].body
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_stored_public_upstream_alias_never_commits_pgsql_mapping(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Reject one provider alias and retain only private cleanup work."""
    record_property("conversation_acceptance_evidence", "security")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("alias")
    await server.start()
    assert server.base_url is not None
    store = _store(dsn, schema)
    binding = _binding(
        f"{server.base_url}/v1",
        "lane-alias-pg",
        streaming=False,
    )
    provider = _provider(binding)
    client, coordinator = _client(
        store,
        provider,
        "phase6-alias-pg",
    )
    public_id = conversation.PublicResponseId(
        "direct-phase6-alias-pg-1-create-response"
    )
    try:
        await store.open()
        with pytest.raises(conversation.ConversationValidationError):
            await client.create(
                "reject public upstream alias",
                avalan.StoredConversationSettings(
                    provider_storage_disclosed=True
                ),
            )
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.retrieve(public_id, authority())
        page = await store.list_checkpoints(authority(), cursor=None, limit=10)
        assert len(page.checkpoints) == 1
        assert str(page.checkpoints[0].identity.checkpoint_id).startswith(
            "quarantine-"
        )
        work = await store.claim_provider_lifecycle(authority(), limit=10)
        assert len(work) == 1
        assert str(work[0].upstream_response_id) == str(public_id)
        assert str(public_id) not in repr(work[0])
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_generated_checkpoint_alias_fails_pgsql_codec_and_sdk(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Reject a generated checkpoint alias at every durable boundary."""
    record_property("conversation_acceptance_evidence", "security")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("checkpoint_alias")
    await server.start()
    assert server.base_url is not None
    store = _store(dsn, schema)
    binding = _binding(
        f"{server.base_url}/v1",
        "lane-checkpoint-alias-pg",
        streaming=False,
    )
    provider = _provider(binding)
    client, coordinator = _client(
        store,
        provider,
        "phase6-checkpoint-alias-pg",
    )
    public_response_id = conversation.PublicResponseId(
        "direct-phase6-checkpoint-alias-pg-1-create-response"
    )
    try:
        await store.open()
        with pytest.raises(conversation.ConversationValidationError):
            await client.create(
                "reject generated checkpoint alias",
                avalan.StoredConversationSettings(
                    provider_storage_disclosed=True
                ),
            )
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.retrieve(public_response_id, authority())
        page = await store.list_checkpoints(authority(), cursor=None, limit=10)
        assert len(page.checkpoints) == 1
        assert str(page.checkpoints[0].identity.checkpoint_id).startswith(
            "quarantine-"
        )
        work = await store.claim_provider_lifecycle(authority(), limit=10)
        assert len(work) == 1
        assert (
            work[0].upstream_response_id
            == "direct-phase6-checkpoint-alias-pg-1-create-checkpoint"
        )

        forged = _stored_atomic_commit("phase6-pg-checkpoint-bypass")
        checkpoint = forged.candidate.checkpoint
        lane = checkpoint.content.lanes[0]
        assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
        object.__setattr__(
            lane,
            "upstream_response_id",
            conversation.UpstreamResponseId(
                str(checkpoint.identity.checkpoint_id)
            ),
        )
        with pytest.raises(conversation.ConversationCodecError):
            conversation.ConversationCheckpointCodec().encode(checkpoint)
        with pytest.raises(conversation.ConversationValidationError):
            await store.commit_atomic(forged)
        result = conversation.InMemoryConversationStore._build_result(
            forged,
            checkpoint,
        )
        assert result is not None
        receipt = object.__new__(conversation.AtomicCommitReceipt)
        object.__setattr__(receipt, "checkpoint", checkpoint)
        object.__setattr__(receipt, "result", result)
        object.__setattr__(
            receipt,
            "output_candidates",
            forged.output_candidates,
        )
        with pytest.raises(conversation.ConversationValidationError):
            sdk_module._direct_result(receipt)
        unchanged = await store.list_checkpoints(
            authority(), cursor=None, limit=10
        )
        assert unchanged == page
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_retrieve_execution_drift_fails_over_tcp_and_pgsql(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Reject retrieved execution drift against the durable local binding."""
    record_property("conversation_acceptance_evidence", "security")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("restart")
    await server.start()
    assert server.base_url is not None
    store = _store(dsn, schema)
    binding = _binding(
        f"{server.base_url}/v1",
        "lane-retrieve-drift-pg",
        streaming=False,
    )
    provider = _provider(binding)
    client, coordinator = _client(
        store,
        provider,
        "phase6-retrieve-drift-pg",
    )
    try:
        await store.open()
        created = await client.create(
            "persist exact retrieval binding",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        assert type(created.handle) is avalan.StoredConversationHandle
        public_response_id = created.handle.public_response_id
        assert public_response_id is not None
        checkpoint = await store.load(
            created.handle.checkpoint_id,
            authority(),
        )
        lane = checkpoint.content.lanes[0]
        assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
        assert lane.binding == binding
        assert lane.binding.execution_definition_digest is not None
        upstream_response_id = str(lane.upstream_response_id)
        original = server.responses[upstream_response_id]

        drifted_values: tuple[tuple[str, dict[str, object]], ...] = (
            ("instructions", {"instructions": "drifted instructions"}),
            ("temperature", {"temperature": 0.3}),
            (
                "tools",
                {
                    "tools": [
                        {
                            "type": "function",
                            "name": "drifted_lookup",
                            "parameters": {"type": "object"},
                            "strict": True,
                        }
                    ]
                },
            ),
            (
                "combined",
                {
                    "instructions": "combined drift",
                    "max_tool_calls": 5,
                    "safety_identifier": "combined-drift",
                    "temperature": 0.4,
                },
            ),
        )
        for _, changes in drifted_values:
            server.responses[upstream_response_id] = original | changes
            with pytest.raises(conversation.ConversationProviderResponseError):
                await client.retrieve(public_response_id)

        server.responses[upstream_response_id] = original | {
            "reasoning": {"context": "all_turns"}
        }
        with pytest.raises(conversation.ConversationBindingDriftError):
            await client.retrieve(public_response_id)

        server.responses[upstream_response_id] = original
        retrieved = await client.retrieve(public_response_id)
        assert retrieved.output == created.output
        assert await store.load(created.handle.checkpoint_id, authority()) == (
            checkpoint
        )
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


async def test_stored_quarantine_survives_pgsql_capacity_and_restart(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Reserve durable cleanup capacity and preserve its fence on restart."""
    record_property("conversation_acceptance_evidence", "database")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("quarantine")
    await server.start()
    assert server.base_url is not None
    binding = _binding(
        f"{server.base_url}/v1",
        "lane-quarantine-capacity",
        streaming=False,
    )

    seed_store = _store(dsn, schema)
    seed_provider = _provider(binding)
    seed_client, seed_coordinator = _client(
        seed_store,
        seed_provider,
        "phase6-quarantine-capacity-seed",
    )
    try:
        await seed_store.open()
        seed = await seed_client.create(
            "capacity seed",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        seed_checkpoint = await seed_store.load(
            seed.handle.checkpoint_id,
            authority(),
        )
    finally:
        await seed_coordinator.close()
        await seed_store.close()

    policy = conversation.PgsqlConversationStorePolicy(
        limits=replace(conversation.StoreLimits(), max_checkpoints=1)
    )
    failure_store = _store(dsn, schema, policy=policy)
    failure_provider = _provider(binding)
    failure_client, failure_coordinator = _client(
        failure_store,
        failure_provider,
        "phase6-quarantine-capacity-failure",
    )
    key = conversation.RequestIdempotencyKey("phase6-quarantine-capacity-key")
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True
    )
    try:
        await failure_store.open()
        with pytest.raises(conversation.ConversationLimitError):
            await failure_client.create(
                "capacity failure",
                settings,
                idempotency_key=key,
            )
        assert len(_post_payloads(server)) == 2
        with pytest.raises(conversation.ConversationAmbiguousDispatchError):
            await failure_client.create(
                "capacity failure",
                settings,
                idempotency_key=key,
            )
        assert len(_post_payloads(server)) == 2
    finally:
        await failure_coordinator.close()
        await failure_store.close()

    restarted_store = _store(dsn, schema, policy=policy)
    restarted_provider = _provider(binding)
    restarted_client, restarted_coordinator = _client(
        restarted_store,
        restarted_provider,
        "phase6-quarantine-capacity-failure",
    )
    try:
        await restarted_store.open()
        with pytest.raises(conversation.ConversationAmbiguousDispatchError):
            await restarted_client.create(
                "capacity failure",
                settings,
                idempotency_key=key,
            )
        assert len(_post_payloads(server)) == 2
        page = await restarted_store.list_checkpoints(
            authority(), cursor=None, limit=10
        )
        assert len(page.checkpoints) == 2
        assert (
            sum(
                str(item.identity.checkpoint_id).startswith("quarantine-")
                for item in page.checkpoints
            )
            == 1
        )
        quarantine = next(
            item
            for item in page.checkpoints
            if str(item.identity.checkpoint_id).startswith("quarantine-")
        )
        staged_quarantine = conversation.with_checkpoint_integrity(
            replace(
                quarantine,
                lifecycle=conversation.CheckpointLifecycle.STAGED,
                timestamps=replace(
                    quarantine.timestamps,
                    committed_at=None,
                ),
                integrity=None,
            )
        )
        replay = await restarted_store.quarantine_provider_checkpoint(
            conversation.ProviderQuarantineRequest(
                candidate=conversation.ExecutionSegmentCheckpointCandidate(
                    checkpoint=staged_quarantine
                ),
                created_at=_NOW,
            )
        )
        assert replay.checkpoint_id == quarantine.identity.checkpoint_id
        assert replay.target_count == 1
        assert (
            await restarted_store.load(
                seed.handle.checkpoint_id,
                authority(),
            )
            == seed_checkpoint
        )
        reconciler = restarted_client._runtime.lifecycle_reconciler
        assert reconciler is not None
        assert await reconciler.run_once(limit=10) == 1
        assert server.deleted == {"private-quarantine-2"}
        assert (
            await restarted_store.claim_provider_lifecycle(
                authority(), limit=10
            )
            == ()
        )
    finally:
        await restarted_coordinator.close()
        await restarted_store.close()
        await server.close()


@pytest.mark.parametrize(
    "cancel_close", (False, True), ids=("fault", "cancel")
)
async def test_stream_close_quarantine_survives_pgsql_restart_capacity(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
    cancel_close: bool,
) -> None:
    """Persist one validated terminal before stream-close settlement."""
    record_property("conversation_acceptance_evidence", "database")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("quarantine")
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"

    seed_store = _store(dsn, schema)
    seed_binding = _binding(
        endpoint,
        f"lane-stream-close-seed-{'cancel' if cancel_close else 'fault'}",
        streaming=False,
    )
    seed_provider = _provider(seed_binding)
    seed_client, seed_coordinator = _client(
        seed_store,
        seed_provider,
        f"phase6-stream-close-seed-{'cancel' if cancel_close else 'fault'}",
    )
    try:
        await seed_store.open()
        seed = await seed_client.create(
            "fill ordinary checkpoint capacity",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        seed_checkpoint = await seed_store.load(
            seed.handle.checkpoint_id,
            authority(),
        )
    finally:
        await seed_coordinator.close()
        await seed_store.close()

    policy = conversation.PgsqlConversationStorePolicy(
        limits=replace(conversation.StoreLimits(), max_checkpoints=1)
    )
    suffix = "cancel" if cancel_close else "fault"
    namespace = f"phase6-stream-close-{suffix}"
    binding = _binding(
        endpoint,
        f"lane-stream-close-{suffix}",
        streaming=True,
    )
    failure_store = _store(dsn, schema, policy=policy)
    failure_provider = _provider(binding)
    failure_client, failure_coordinator = _client(
        failure_store,
        failure_provider,
        namespace,
        boundary_hook=_StreamCloseFailureHook(cancel=cancel_close),
    )
    key = conversation.RequestIdempotencyKey(
        f"phase6-stream-close-{suffix}-key"
    )
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True
    )
    public_response_id = conversation.PublicResponseId(
        f"direct-{namespace}-1-create-response"
    )
    try:
        await failure_store.open()
        stream = await failure_client.create(
            "validated terminal then close failure",
            settings,
            stream=True,
            idempotency_key=key,
        )
        expected_error = (
            avalan.DirectConversationCancelledError
            if cancel_close
            else conversation.ConversationCommitError
        )
        with pytest.raises(expected_error):
            _ = [event async for event in stream]
        assert len(_post_payloads(server)) == 2
        with pytest.raises(conversation.ConversationAuthorizationError):
            await failure_store.retrieve(public_response_id, authority())
        retry = await failure_client.create(
            "validated terminal then close failure",
            settings,
            stream=True,
            idempotency_key=key,
        )
        with pytest.raises(conversation.ConversationAmbiguousDispatchError):
            _ = [event async for event in retry]
        assert len(_post_payloads(server)) == 2
    finally:
        await failure_coordinator.close()
        await failure_store.close()

    restarted_store = _store(dsn, schema, policy=policy)
    restarted_provider = _provider(binding)
    restarted_client, restarted_coordinator = _client(
        restarted_store,
        restarted_provider,
        namespace,
    )
    try:
        await restarted_store.open()
        retry = await restarted_client.create(
            "validated terminal then close failure",
            settings,
            stream=True,
            idempotency_key=key,
        )
        with pytest.raises(conversation.ConversationAmbiguousDispatchError):
            _ = [event async for event in retry]
        assert len(_post_payloads(server)) == 2
        page = await restarted_store.list_checkpoints(
            authority(), cursor=None, limit=10
        )
        assert len(page.checkpoints) == 2
        quarantines = tuple(
            checkpoint
            for checkpoint in page.checkpoints
            if str(checkpoint.identity.checkpoint_id).startswith("quarantine-")
        )
        assert len(quarantines) == 1
        quarantine = quarantines[0]
        lane = quarantine.content.lanes[0]
        assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
        assert lane.upstream_response_id == "private-quarantine-2"
        staged_quarantine = conversation.with_checkpoint_integrity(
            replace(
                quarantine,
                lifecycle=conversation.CheckpointLifecycle.STAGED,
                timestamps=replace(
                    quarantine.timestamps,
                    committed_at=None,
                ),
                integrity=None,
            )
        )
        replay = await restarted_store.quarantine_provider_checkpoint(
            conversation.ProviderQuarantineRequest(
                candidate=conversation.ExecutionSegmentCheckpointCandidate(
                    checkpoint=staged_quarantine
                ),
                created_at=_NOW,
            )
        )
        assert replay.checkpoint_id == quarantine.identity.checkpoint_id
        assert replay.target_count == 1
        assert (
            await restarted_store.load(
                seed.handle.checkpoint_id,
                authority(),
            )
            == seed_checkpoint
        )
        reconciler = restarted_client._runtime.lifecycle_reconciler
        assert reconciler is not None
        assert await reconciler.run_once(limit=10) == 1
        assert await reconciler.run_once(limit=10) == 0
        assert server.deleted == {"private-quarantine-2"}
        assert (
            sum(request.method == "DELETE" for request in server.requests) == 1
        )
        assert (
            await restarted_store.claim_provider_lifecycle(
                authority(), limit=10
            )
            == ()
        )
    finally:
        await restarted_coordinator.close()
        await restarted_store.close()
        await server.close()


async def test_stored_commit_failure_reconciles_after_restart(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Quarantine a completed child and delete it after a store restart."""
    record_property("conversation_acceptance_evidence", "security")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("quarantine")
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"
    binding = _binding(endpoint, "lane-quarantine", streaming=False)

    parent_store = _store(dsn, schema)
    parent_provider = _provider(binding)
    parent_client, parent_coordinator = _client(
        parent_store,
        parent_provider,
        "phase6-quarantine-parent",
    )
    parent_result: avalan.DirectConversationResult
    parent_checkpoint: conversation.ConversationCheckpoint
    try:
        await parent_store.open()
        parent_result = await parent_client.create(
            "quarantine parent",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        assert type(parent_result.handle) is avalan.StoredConversationHandle
        parent_checkpoint = await parent_store.load(
            parent_result.handle.checkpoint_id,
            authority(),
        )
    finally:
        await parent_coordinator.close()
        await parent_store.close()

    failed_store = _store(dsn, schema, fault_hook=_CommitFailureHook())
    failed_provider = _provider(binding)
    failed_client, failed_coordinator = _client(
        failed_store,
        failed_provider,
        "phase6-quarantine-failed-child",
    )
    try:
        await failed_store.open()
        with pytest.raises(conversation.ConversationStorageError):
            await failed_client.continue_conversation(
                "quarantine child",
                avalan.StoredConversationSettings(
                    provider_storage_disclosed=True,
                    parent=avalan.StoredParent(handle=parent_result.handle),
                ),
            )
        assert (
            await failed_store.load(
                parent_result.handle.checkpoint_id,
                authority(),
            )
            == parent_checkpoint
        )
    finally:
        await failed_coordinator.close()
        await failed_store.close()

    restarted_store = _store(dsn, schema)
    restarted_provider = _provider(binding)
    resolver = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=restarted_provider,
                revision="phase6-quarantine-restart",
                valid_from=_NOW - timedelta(minutes=1),
                valid_until=_NOW + timedelta(days=1),
            ),
        ),
        clock=_resolver_clock,
    )
    reconciler = conversation.ProviderLifecycleReconciler(
        store=restarted_store,
        resolver=resolver,
        authority=authority(),
    )
    try:
        await restarted_store.open()
        assert await reconciler.run_once(limit=10) == 1
        assert (
            await restarted_store.claim_provider_lifecycle(
                authority(), limit=10
            )
            == ()
        )
        assert server.deleted == {"private-quarantine-2"}
        assert (
            await restarted_store.load(
                parent_result.handle.checkpoint_id,
                authority(),
            )
            == parent_checkpoint
        )
        page = await restarted_store.list_checkpoints(
            authority(), cursor=None, limit=10
        )
        assert any(
            str(item.identity.checkpoint_id).startswith("quarantine-")
            for item in page.checkpoints
        )
    finally:
        await restarted_provider.aclose()
        await restarted_store.close()
        await server.close()


async def test_stored_named_head_race_commits_one_child(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Commit one CAS winner and quarantine the completed losing child."""
    record_property("conversation_acceptance_evidence", "database")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("head")
    await server.start()
    assert server.base_url is not None
    store = _store(dsn, schema)
    binding = _binding(f"{server.base_url}/v1", "lane-head", streaming=False)
    provider = _provider(binding)
    client, coordinator = _client(store, provider, "phase6-head")
    try:
        await store.open()
        root = await client.create(
            "head root",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
        assert type(root.handle) is avalan.StoredConversationHandle
        parent = avalan.StoredParent(handle=root.handle)
        root_checkpoint = await store.load(
            root.handle.checkpoint_id,
            authority(),
        )
        head_id = conversation.NamedHeadId("phase6-main")
        await store.create_head(
            conversation.NamedHeadSnapshot(
                head_id=head_id,
                revision=conversation.NamedHeadRevision(0),
                checkpoint_id=root.handle.checkpoint_id,
            ),
            authority(),
        )

        async def advance(label: str) -> avalan.DirectConversationResult:
            result = await client.continue_conversation(
                f"head {label}",
                avalan.StoredConversationSettings(
                    provider_storage_disclosed=True,
                    parent=parent,
                    named_head=avalan.NamedHeadParent(
                        head_id=head_id,
                        expected_revision=conversation.NamedHeadRevision(0),
                        parent=parent,
                    ),
                ),
            )
            assert type(result) is avalan.DirectConversationResult
            return result

        outcomes = await gather(
            advance("alpha"),
            advance("beta"),
            return_exceptions=True,
        )
        successes = [
            item
            for item in outcomes
            if type(item) is avalan.DirectConversationResult
        ]
        conflicts = [
            item
            for item in outcomes
            if type(item) is conversation.ConversationConflictError
        ]
        assert len(successes) == len(conflicts) == 1
        head = await store.load_head(head_id, authority())
        assert head.revision == 1
        assert head.checkpoint_id == successes[0].handle.checkpoint_id
        assert await store.load(root.handle.checkpoint_id, authority()) == (
            root_checkpoint
        )
        payloads = _post_payloads(server)
        assert len(payloads) == 3
        assert all(
            payload["previous_response_id"] == "private-head-root"
            for payload in payloads[1:]
        )

        resolver = conversation.StoredProviderResolver(
            (
                conversation.StoredProviderResolverEntry(
                    adapter=provider,
                    revision="phase6-head-reconciler",
                    valid_from=_NOW - timedelta(minutes=1),
                    valid_until=_NOW + timedelta(days=1),
                ),
            ),
            clock=_resolver_clock,
        )
        reconciler = conversation.ProviderLifecycleReconciler(
            store=store,
            resolver=resolver,
            authority=authority(),
        )
        assert await reconciler.run_once(limit=10) == 1
        assert len(server.deleted) == 1
        assert server.deleted <= {"private-head-alpha", "private-head-beta"}
    finally:
        await coordinator.close()
        await store.close()
        await server.close()
