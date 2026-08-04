"""Exercise native stateless replay over real TCP and PostgreSQL."""

from asyncio import (
    Server,
    StreamReader,
    StreamWriter,
    run,
    start_server,
    to_thread,
)
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from json import dumps, loads
from multiprocessing import get_context
from multiprocessing.connection import Connection
from os import environ
from typing import cast
from uuid import uuid4

import httpx
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

_ADAPTER = "avalan.conversation.providers.openai.NativeOpenAIStatelessProvider"
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
    """Run the normative wire contract on asyncio only."""
    return "asyncio"


def _key() -> conversation.ConversationDataKey:
    return conversation.ConversationDataKey(
        key_id="phase5-native-openai-key",
        revision=1,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"5" * 32,
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
    """Yield one migrated isolated schema for native replay evidence."""
    assert _DSN is not None
    schema = f"conv_phase5_native_{uuid4().hex}"
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    try:
        yield _DSN, schema
    finally:
        await _drop_schema(_DSN, schema)


def _reasoning(identifier: str, opaque: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "reasoning",
        "status": "completed",
        "encrypted_content": opaque,
        "summary": [],
    }


def _message(identifier: str, text: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_call(
    identifier: str,
    call_id: str,
) -> dict[str, object]:
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
    model: str = "gpt-5",
) -> dict[str, object]:
    return {
        "id": identifier,
        "object": "response",
        "created_at": 1.0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"context": "current_turn"},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 10,
        },
    }


def _last_user_text(payload: Mapping[str, object]) -> str | None:
    raw_input = payload.get("input")
    if type(raw_input) is not list:
        return None
    for item in reversed(raw_input):
        if type(item) is not dict or item.get("role") != "user":
            continue
        content = item.get("content")
        if type(content) is not list or not content:
            return None
        part = content[0]
        if type(part) is not dict:
            return None
        text = part.get("text")
        return text if type(text) is str else None
    return None


def _scripted_response(
    scenario: str,
    payload: Mapping[str, object],
    request_index: int,
) -> dict[str, object]:
    if scenario == "restart":
        restart_outputs = (
            [
                _reasoning("restart-reasoning-1", "restart-opaque-1"),
                _message("restart-message-1", "restart-first"),
            ]
            if request_index == 0
            else [
                _reasoning("restart-reasoning-2", "restart-opaque-2"),
                _message("restart-message-2", "restart-second"),
            ]
        )
        return _response(
            f"restart-response-{request_index + 1}", restart_outputs
        )
    if scenario == "tools":
        tool_outputs = (
            [
                _reasoning("tools-reasoning-1", "tools-opaque-1"),
                _message("tools-message-1", "tools-first"),
            ],
            [
                _reasoning("tools-reasoning-2a", "tools-opaque-2a"),
                _function_call("tools-call-2", "call-two"),
            ],
            [
                _reasoning("tools-reasoning-2b", "tools-opaque-2b"),
                _message("tools-message-2", "tools-second"),
            ],
            [
                _reasoning("tools-reasoning-3a", "tools-opaque-3a"),
                _function_call("tools-call-3", "call-three"),
            ],
            [
                _reasoning("tools-reasoning-3b", "tools-opaque-3b"),
                _message("tools-message-3", "tools-third"),
            ],
        )
        return _response(
            f"tools-response-{request_index + 1}",
            tool_outputs[request_index],
        )
    if scenario == "branches":
        text = _last_user_text(payload)
        suffix = {
            "branch parent": "parent",
            "branch alpha": "alpha",
            "branch beta": "beta",
        }[cast(str, text)]
        return _response(
            f"branch-response-{suffix}",
            [
                _reasoning(
                    f"branch-reasoning-{suffix}",
                    f"branch-opaque-{suffix}",
                ),
                _message(f"branch-message-{suffix}", suffix),
            ],
        )
    if scenario == "azure":
        return _response(
            "azure-response",
            [
                _reasoning("azure-reasoning", "azure-opaque"),
                _message("azure-message", "azure-result"),
            ],
            model="deployment-native",
        )
    raise AssertionError("unknown scripted scenario")


@dataclass(slots=True)
class _WireRequest:
    path: str
    body: bytes
    payload: dict[str, object]


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

    def mock_transport(self) -> httpx.MockTransport:
        """Return an exact mocked SDK boundary backed by loopback TCP."""
        return httpx.MockTransport(self.handle_async_request)

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
    requests: list[_WireRequest] = field(default_factory=list)
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
            payload,
            request_index,
        )
        if payload.get("stream") is True:
            raw_output = response["output"]
            assert type(raw_output) is list
            events = [
                {
                    "type": "response.output_item.done",
                    "sequence_number": index,
                    "output_index": index,
                    "item": item,
                }
                for index, item in enumerate(raw_output)
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
            ).encode("utf-8")
            content_type = "text/event-stream"
        else:
            response_body = dumps(response).encode("utf-8")
            content_type = "application/json"
        writer.write(
            (
                "HTTP/1.1 200 OK\r\n"
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
) -> conversation.ProviderLaneBinding:
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=_ADAPTER,
        provider_family=(
            conversation.ProviderFamily.AZURE_OPENAI
            if azure
            else conversation.ProviderFamily.OPENAI
        ),
        normalized_endpoint=endpoint,
        azure_resource_identity=(
            azure_resource_identity or "phase5-resource" if azure else None
        ),
        model_or_deployment="deployment-native" if azure else "gpt-5",
        provider_api_revision=conversation.ProviderApiRevision(
            "azure-openai-v1-preview" if azure else "openapi-2.3.0"
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("model-config-phase5")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-phase5")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-phase5"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-phase5")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=(
            conversation.ProviderTransport.STREAMING
            if streaming
            else conversation.ProviderTransport.NON_STREAMING
        ),
        agent_id=authority().agent_id,
    )


def _capabilities(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ConversationCapabilityProfile:
    supported = {
        conversation.ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY,
        conversation.ConversationCapability.REASONING_CONTEXT_CURRENT_TURN,
        conversation.ConversationCapability.REASONING_CONTEXT_ALL_TURNS,
    }
    if binding.transport is conversation.ProviderTransport.STREAMING:
        supported.add(
            conversation.ConversationCapability.STREAMING_ITEM_FIDELITY
        )
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId(
            f"profile-{binding.lane_id}"
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
                    (f"phase5-{capability.value}",)
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
) -> conversation.NativeOpenAIStatelessProvider:
    client = AsyncOpenAI(
        api_key="phase5-test-key",
        base_url=binding.normalized_endpoint,
        default_query=(
            {"api-version": "preview"}
            if binding.provider_api_revision == "azure-openai-v1-preview"
            else None
        ),
        http_client=(
            httpx.AsyncClient(transport=transport)
            if transport is not None
            else None
        ),
        max_retries=0,
    )
    profile = conversation.NativeOpenAIStatelessProfile(
        profile_id=f"profile-{binding.lane_id}",
        binding=binding,
        encrypted_content=(
            conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
            if binding.provider_family
            is conversation.ProviderFamily.AZURE_OPENAI
            else conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
        ),
        scripted_tcp_test=transport is None,
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


def _client(
    store: conversation.PgsqlConversationStore,
    provider: conversation.NativeOpenAIStatelessProvider,
    namespace: str,
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
        lanes=(
            conversation.NativeOpenAIConversationLaneRuntime(
                provider=provider
            ),
        ),
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=store,
        authority=scope,
        lane=provider.binding,
        retention=retention(),
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


def _parent(
    values: tuple[str, str, str],
) -> avalan.StatelessParent:
    return avalan.StatelessParent(
        handle=avalan.StatelessConversationHandle(
            conversation_id=conversation.ConversationId(values[0]),
            checkpoint_id=conversation.CheckpointId(values[1]),
            branch_id=conversation.ConversationBranchId(values[2]),
        )
    )


async def _child_continue_once(
    dsn: str,
    schema: str,
    endpoint: str,
    handle_values: tuple[str, str, str],
) -> tuple[str, tuple[str, str, str]]:
    store = _store(dsn, schema)
    binding = _binding(endpoint, "lane-restart", streaming=False)
    provider = _provider(binding)
    client, coordinator = _client(store, provider, "phase5-restart-child")
    try:
        await store.open()
        result = await client.continue_conversation(
            "restart second",
            avalan.StatelessConversationSettings(
                parent=_parent(handle_values)
            ),
        )
        assert type(result.handle) is avalan.StatelessConversationHandle
        return result.output, _handle_values(result.handle)
    finally:
        await coordinator.close()
        await store.close()


async def _stream_result(
    stream: avalan.DirectConversationStream,
) -> avalan.DirectConversationResult:
    events = [event async for event in stream]
    terminal = events[-1]
    assert type(terminal) is avalan.DirectConversationStreamTerminal
    return terminal.result


async def _child_continue_tools(
    dsn: str,
    schema: str,
    endpoint: str,
    handle_values: tuple[str, str, str],
) -> tuple[str, str, tuple[str, str, str]]:
    store = _store(dsn, schema)
    binding = _binding(endpoint, "lane-tools", streaming=True)
    provider = _provider(binding, tools=True)
    client, coordinator = _client(store, provider, "phase5-tools-child")
    try:
        await store.open()
        second_stream = await client.continue_conversation(
            "tools second",
            avalan.StatelessConversationSettings(
                parent=_parent(handle_values)
            ),
            stream=True,
        )
        second = await _stream_result(second_stream)
        assert type(second.handle) is avalan.StatelessConversationHandle
        third_stream = await client.continue_conversation(
            "tools third",
            avalan.StatelessConversationSettings(
                parent=avalan.StatelessParent(handle=second.handle)
            ),
            stream=True,
        )
        third = await _stream_result(third_stream)
        assert type(third.handle) is avalan.StatelessConversationHandle
        return second.output, third.output, _handle_values(third.handle)
    finally:
        await coordinator.close()
        await store.close()


def _child_target(
    operation: str,
    dsn: str,
    schema: str,
    endpoint: str,
    handle_values: tuple[str, str, str],
    connection: Connection,
) -> None:
    try:
        payload: object
        if operation == "restart":
            payload = run(
                _child_continue_once(
                    dsn,
                    schema,
                    endpoint,
                    handle_values,
                )
            )
        elif operation == "tools":
            payload = run(
                _child_continue_tools(
                    dsn,
                    schema,
                    endpoint,
                    handle_values,
                )
            )
        else:
            raise ValueError("unknown child operation")
        connection.send((True, payload))
    except BaseException as error:
        connection.send((False, (type(error).__name__, str(error))))
    finally:
        connection.close()


async def _spawn_child(
    operation: str,
    dsn: str,
    schema: str,
    endpoint: str,
    handle_values: tuple[str, str, str],
) -> object:
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_target,
        args=(
            operation,
            dsn,
            schema,
            endpoint,
            handle_values,
            child_connection,
        ),
    )
    process.start()
    child_connection.close()
    await to_thread(process.join, 45)
    if process.is_alive():
        process.terminate()
        await to_thread(process.join, 5)
        pytest.fail("Phase 5 fresh-process continuation timed out")
    assert process.exitcode == 0
    success, payload = parent_connection.recv()
    parent_connection.close()
    assert success, payload
    return payload


def _assert_common_wire(requests: list[_WireRequest]) -> None:
    for request in requests:
        assert request.payload["store"] is False
        assert "previous_response_id" not in request.payload
        assert loads(request.body) == request.payload


async def test_native_openai_fresh_process_durable_replay(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Replay encrypted native state from PostgreSQL in a fresh process."""
    record_property("conversation_acceptance_evidence", "database")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("restart")
    await server.start()
    assert server.base_url is not None
    endpoint = f"{server.base_url}/v1"
    store = _store(dsn, schema)
    provider = _provider(_binding(endpoint, "lane-restart", streaming=False))
    client, coordinator = _client(
        store,
        provider,
        "phase5-durable-parent",
    )
    try:
        await store.open()
        first = await client.create(
            "restart first",
            avalan.StatelessConversationSettings(),
        )
        assert type(first.handle) is avalan.StatelessConversationHandle
        await coordinator.close()
        await store.close()
        child_payload = cast(
            tuple[str, tuple[str, str, str]],
            await _spawn_child(
                "restart",
                dsn,
                schema,
                endpoint,
                _handle_values(first.handle),
            ),
        )
        assert child_payload[0] == "restart-second"
        assert child_payload[1][0] == str(first.handle.conversation_id)
        assert child_payload[1][1] != str(first.handle.checkpoint_id)
        assert [request.path for request in server.requests] == [
            "/v1/responses",
            "/v1/responses",
        ]
        replay = server.requests[1].payload["input"]
        assert type(replay) is list
        assert [item["id"] for item in replay[:-1]] == [
            "restart-reasoning-1",
            "restart-message-1",
        ]
        assert replay[0]["encrypted_content"] == "restart-opaque-1"
        _assert_common_wire(server.requests)
    finally:
        await coordinator.close()
        await store.close()
        await server.close()


@pytest.mark.parametrize("streaming", (False, True), ids=("sync", "stream"))
async def test_native_azure_exact_identity_over_loopback_transport(
    pgsql_schema: tuple[str, str],
    streaming: bool,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep exact Azure identity while routing test traffic over loopback."""
    record_property("conversation_acceptance_evidence", "wire")
    dsn, schema = pgsql_schema
    server = _ScriptedTcpProvider("azure")
    await server.start()
    assert server.base_url is not None
    exact_resource = "phase5-resource.openai.azure.com"
    exact_endpoint = f"https://{exact_resource}/openai/v1"
    loopback_transport = _LoopbackTcpTransport(server.base_url)
    store = _store(dsn, schema)
    binding = _binding(
        exact_endpoint,
        f"lane-azure-exact-{'stream' if streaming else 'sync'}",
        streaming=streaming,
        azure=True,
        azure_resource_identity=exact_resource,
    )
    provider = _provider(
        binding,
        transport=loopback_transport.mock_transport(),
    )
    client, coordinator = _client(
        store,
        provider,
        f"phase5-azure-exact-{'stream' if streaming else 'sync'}",
    )
    try:
        await store.open()
        if streaming:
            stream = await client.create(
                "azure exact wire",
                avalan.StatelessConversationSettings(),
                stream=True,
            )
            assert type(stream) is avalan.DirectConversationStream
            result = await _stream_result(stream)
        else:
            result = await client.create(
                "azure exact wire",
                avalan.StatelessConversationSettings(),
                stream=False,
            )
            assert type(result) is avalan.DirectConversationResult
        assert result.output == "azure-result"
        assert binding.normalized_endpoint == exact_endpoint
        assert binding.azure_resource_identity == exact_resource
        assert binding.model_or_deployment == "deployment-native"
        assert binding.provider_api_revision == "azure-openai-v1-preview"
        assert provider._profile.scripted_tcp_test is False
        assert loopback_transport.request_urls == [
            exact_endpoint + "/responses?api-version=preview"
        ]
        assert (
            server.requests[0].path
            == "/openai/v1/responses?api-version=preview"
        )
        payload = server.requests[0].payload
        assert payload["model"] == "deployment-native"
        assert payload["include"] == ["reasoning.encrypted_content"]
        assert payload["stream"] is streaming
        _assert_common_wire(server.requests)
    finally:
        await coordinator.close()
        await store.close()
        await loopback_transport.aclose()
        await server.close()


async def test_normative_stateless_contract(
    pgsql_schema: tuple[str, str],
    record_property: Callable[[str, object], None],
) -> None:
    """Prove restart, tools, branching, and Azure stateless wire behavior."""
    record_property("conversation_acceptance_evidence", "wire")
    dsn, schema = pgsql_schema

    restart_server = _ScriptedTcpProvider("restart")
    await restart_server.start()
    assert restart_server.base_url is not None
    restart_endpoint = f"{restart_server.base_url}/v1"
    restart_store = _store(dsn, schema)
    restart_provider = _provider(
        _binding(restart_endpoint, "lane-restart", streaming=False)
    )
    restart_client, restart_coordinator = _client(
        restart_store,
        restart_provider,
        "phase5-restart-parent",
    )
    try:
        await restart_store.open()
        first = await restart_client.create(
            "restart first",
            avalan.StatelessConversationSettings(),
        )
        assert first.output == "restart-first"
        assert type(first.handle) is avalan.StatelessConversationHandle
        await restart_coordinator.close()
        await restart_store.close()
        child_payload = cast(
            tuple[str, tuple[str, str, str]],
            await _spawn_child(
                "restart",
                dsn,
                schema,
                restart_endpoint,
                _handle_values(first.handle),
            ),
        )
        assert child_payload[0] == "restart-second"
        assert child_payload[1][0] == str(first.handle.conversation_id)
        assert child_payload[1][1] != str(first.handle.checkpoint_id)
        assert [request.path for request in restart_server.requests] == [
            "/v1/responses",
            "/v1/responses",
        ]
        replay = restart_server.requests[1].payload["input"]
        assert type(replay) is list
        assert [item["id"] for item in replay[:-1]] == [
            "restart-reasoning-1",
            "restart-message-1",
        ]
        assert (
            _last_user_text(restart_server.requests[1].payload)
            == "restart second"
        )
        _assert_common_wire(restart_server.requests)
    finally:
        await restart_coordinator.close()
        await restart_store.close()
        await restart_server.close()

    tools_server = _ScriptedTcpProvider("tools")
    await tools_server.start()
    assert tools_server.base_url is not None
    tools_endpoint = f"{tools_server.base_url}/v1"
    tools_store = _store(dsn, schema)
    tools_provider = _provider(
        _binding(tools_endpoint, "lane-tools", streaming=True),
        tools=True,
    )
    tools_client, tools_coordinator = _client(
        tools_store,
        tools_provider,
        "phase5-tools-parent",
    )
    try:
        await tools_store.open()
        first_stream = await tools_client.create(
            "tools first",
            avalan.StatelessConversationSettings(),
            stream=True,
        )
        tools_first = await _stream_result(first_stream)
        assert type(tools_first.handle) is avalan.StatelessConversationHandle
        await tools_coordinator.close()
        await tools_store.close()
        child_tools = cast(
            tuple[str, str, tuple[str, str, str]],
            await _spawn_child(
                "tools",
                dsn,
                schema,
                tools_endpoint,
                _handle_values(tools_first.handle),
            ),
        )
        assert child_tools[0:2] == ("tools-second", "tools-third")
        assert len(tools_server.requests) == 5
        assert all(
            request.path == "/v1/responses"
            for request in tools_server.requests
        )
        assert all(
            request.payload["stream"] is True
            for request in tools_server.requests
        )
        second_tool_input = tools_server.requests[2].payload["input"]
        third_turn_input = tools_server.requests[3].payload["input"]
        third_tool_input = tools_server.requests[4].payload["input"]
        assert type(second_tool_input) is list
        assert type(third_turn_input) is list
        assert type(third_tool_input) is list
        assert [item["type"] for item in second_tool_input[-3:]] == [
            "reasoning",
            "function_call",
            "function_call_output",
        ]
        assert [item["call_id"] for item in second_tool_input[-2:]] == [
            "call-two",
            "call-two",
        ]
        assert (
            _last_user_text(tools_server.requests[3].payload) == "tools third"
        )
        assert [item["call_id"] for item in third_tool_input[-2:]] == [
            "call-three",
            "call-three",
        ]
        assert (
            sum(
                item.get("id") == "tools-reasoning-1"
                for item in third_turn_input
            )
            == 1
        )
        _assert_common_wire(tools_server.requests)
    finally:
        await tools_coordinator.close()
        await tools_store.close()
        await tools_server.close()

    branch_server = _ScriptedTcpProvider("branches")
    await branch_server.start()
    assert branch_server.base_url is not None
    branch_store = _store(dsn, schema)
    branch_provider = _provider(
        _binding(
            f"{branch_server.base_url}/v1",
            "lane-branches",
            streaming=False,
        )
    )
    branch_client, branch_coordinator = _client(
        branch_store,
        branch_provider,
        "phase5-branches",
    )
    try:
        await branch_store.open()
        branch_parent = await branch_client.create(
            "branch parent",
            avalan.StatelessConversationSettings(),
        )
        assert type(branch_parent.handle) is avalan.StatelessConversationHandle
        parent = avalan.StatelessParent(handle=branch_parent.handle)
        parent_before = await branch_store.load(
            branch_parent.handle.checkpoint_id,
            authority(),
        )
        alpha = await branch_client.branch(
            "branch alpha",
            avalan.StatelessConversationSettings(
                parent=parent,
                branch=avalan.ConversationBranchIntent(
                    parent=parent,
                    branch_id=conversation.ConversationBranchId(
                        "phase5-alpha"
                    ),
                ),
            ),
        )
        beta = await branch_client.branch(
            "branch beta",
            avalan.StatelessConversationSettings(
                parent=parent,
                branch=avalan.ConversationBranchIntent(
                    parent=parent,
                    branch_id=conversation.ConversationBranchId("phase5-beta"),
                ),
            ),
        )
        assert (alpha.output, beta.output) == ("alpha", "beta")
        assert (
            await branch_store.load(
                branch_parent.handle.checkpoint_id,
                authority(),
            )
            == parent_before
        )
        alpha_body = branch_server.requests[1].body
        beta_body = branch_server.requests[2].body
        assert b"branch beta" not in alpha_body
        assert b"branch alpha" not in beta_body
        _assert_common_wire(branch_server.requests)
    finally:
        await branch_coordinator.close()
        await branch_store.close()
        await branch_server.close()

    azure_server = _ScriptedTcpProvider("azure")
    await azure_server.start()
    assert azure_server.base_url is not None
    azure_store = _store(dsn, schema)
    azure_provider = _provider(
        _binding(
            f"{azure_server.base_url}/openai/v1",
            "lane-azure-wire",
            streaming=False,
            azure=True,
        )
    )
    azure_client, azure_coordinator = _client(
        azure_store,
        azure_provider,
        "phase5-azure",
    )
    try:
        await azure_store.open()
        azure = await azure_client.create(
            "azure wire",
            avalan.StatelessConversationSettings(),
        )
        assert azure.output == "azure-result"
        assert (
            azure_server.requests[0].path
            == "/openai/v1/responses?api-version=preview"
        )
        azure_payload = azure_server.requests[0].payload
        assert azure_payload["model"] == "deployment-native"
        assert azure_payload["include"] == ["reasoning.encrypted_content"]
        _assert_common_wire(azure_server.requests)
    finally:
        await azure_coordinator.close()
        await azure_store.close()
        await azure_server.close()
