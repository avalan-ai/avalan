from ...agent.orchestrator import Orchestrator
from ...entities import MessageRole
from ...model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
)
from ..entities import ChatCompletionRequest, ChatMessage
from ..routers import orchestrate, resolve_model_id
from ..routers.streaming import (
    cleanup_stream_sources,
    stream_consumer_iterator,
    stream_terminal_succeeded,
)

from asyncio import CancelledError
from collections.abc import AsyncIterable, AsyncIterator
from copy import deepcopy
from importlib import import_module
from logging import Logger
from typing import Any, cast
from urllib.parse import urljoin

from fastapi import FastAPI


def install_a2a_routes(
    app: FastAPI,
    *,
    prefix: str,
    name: str,
    description: str | None,
) -> None:
    """Install A2A SDK v1 routes on ``app``."""
    try:
        _ensure_typing_override()
        a2a_pb2 = import_module("a2a.types.a2a_pb2")
        constants = import_module("a2a.utils.constants")
        route_module = import_module("a2a.server.routes.fastapi_routes")
        jsonrpc_routes_module = import_module(
            "a2a.server.routes.jsonrpc_routes"
        )
        rest_routes_module = import_module("a2a.server.routes.rest_routes")
        response_helpers_module = import_module(
            "a2a.server.request_handlers.response_helpers"
        )
        handler_module = import_module(
            "a2a.server.request_handlers.default_request_handler_v2"
        )
        task_store_module = import_module(
            "a2a.server.tasks.inmemory_task_store"
        )
        responses_module = import_module("starlette.responses")
        routing_module = import_module("starlette.routing")
    except ImportError as exc:
        raise ImportError("A2A router requires the a2a-sdk package") from exc

    card = _build_agent_card(
        a2a_pb2=a2a_pb2,
        constants=constants,
        interface_url=prefix,
        name=name,
        description=description,
    )
    request_handler = handler_module.DefaultRequestHandlerV2(
        agent_executor=AvalanA2AAgentExecutor(app),
        task_store=task_store_module.InMemoryTaskStore(),
        agent_card=card,
    )
    route_module.add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=_agent_card_routes(
            agent_card=card,
            interface_url=prefix,
            agent_card_to_dict=response_helpers_module.agent_card_to_dict,
            json_response=responses_module.JSONResponse,
            route_class=routing_module.Route,
        ),
        jsonrpc_routes=jsonrpc_routes_module.create_jsonrpc_routes(
            request_handler, rpc_url=prefix, enable_v0_3_compat=False
        ),
        rest_routes=rest_routes_module.create_rest_routes(
            request_handler,
            path_prefix=prefix,
            enable_v0_3_compat=False,
        ),
    )


def _agent_card_routes(
    *,
    agent_card: Any,
    interface_url: str,
    agent_card_to_dict: Any,
    json_response: Any,
    route_class: Any,
) -> list[Any]:
    async def _get_agent_card(request: Any) -> Any:
        card = deepcopy(agent_card)
        absolute_interface_url = _absolute_url(request, interface_url)
        for supported_interface in card.supported_interfaces:
            supported_interface.url = absolute_interface_url
        return json_response(agent_card_to_dict(card))

    return [
        route_class(
            path="/.well-known/agent-card.json",
            endpoint=_get_agent_card,
            methods=["GET"],
        )
    ]


def _absolute_url(request: Any, path: str) -> str:
    return urljoin(str(request.base_url), path.lstrip("/"))


def _ensure_typing_override() -> None:
    typing_module = import_module("typing")
    if hasattr(typing_module, "override"):
        return
    typing_extensions_module = import_module("typing_extensions")
    setattr(typing_module, "override", typing_extensions_module.override)


def _build_agent_card(
    *,
    a2a_pb2: Any,
    constants: Any,
    interface_url: str,
    name: str,
    description: str | None,
) -> Any:
    skill_description = description or "Execute the Avalan agent."
    return a2a_pb2.AgentCard(
        name=name,
        description=skill_description,
        version="1.0.0",
        supported_interfaces=[
            a2a_pb2.AgentInterface(
                url=interface_url,
                protocol_binding=constants.TransportProtocol.JSONRPC,
                protocol_version=constants.PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=a2a_pb2.AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            a2a_pb2.AgentSkill(
                id=name,
                name=name,
                description=skill_description,
                tags=["avalan", "agent"],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ],
    )


class AvalanA2AAgentExecutor:
    """Execute Avalan orchestrator calls for A2A SDK routes."""

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    async def execute(self, context: Any, event_queue: Any) -> None:
        _ensure_typing_override()
        task_id = cast(str, context.task_id)
        context_id = cast(str, context.context_id)
        updater_module = import_module("a2a.server.tasks.task_updater")
        updater = updater_module.TaskUpdater(
            event_queue, task_id=task_id, context_id=context_id
        )
        a2a_pb2 = import_module("a2a.types.a2a_pb2")
        if context.current_task is None:
            await event_queue.enqueue_event(
                a2a_pb2.Task(
                    id=task_id,
                    context_id=context_id,
                    status=a2a_pb2.TaskStatus(
                        state=a2a_pb2.TaskState.TASK_STATE_SUBMITTED
                    ),
                )
            )
        await updater.update_status(
            a2a_pb2.TaskState.TASK_STATE_WORKING,
            metadata={"source": "avalan"},
        )

        response: AsyncIterable[object] | None = None
        iterator: AsyncIterator[object] | None = None
        try:
            orchestrator = await self._orchestrator()
            request = await self._chat_request(context, orchestrator)
            logger = cast(Logger, self._app.state.logger)
            response, _response_uuid, _timestamp = await orchestrate(
                request, logger, orchestrator
            )
            translator = A2AResponseTranslator(updater)
            iterator = stream_consumer_iterator(
                response,
                stream_session_id="a2a-stream",
                run_id=task_id,
                turn_id=context_id,
                unsupported_message="unsupported A2A stream item",
                close_source_on_generator_exit=False,
            )
            async for item in iterator:
                await translator.process(item)
            await translator.finish()
            if translator.succeeded:
                await orchestrator.sync_messages()
        except CancelledError:
            if response is not None:
                await cleanup_stream_sources(
                    response, iterator, cancelled=True
                )
            await updater.cancel()
            raise
        except Exception:
            if response is not None:
                await cleanup_stream_sources(
                    response, iterator, cancelled=False
                )
            await updater.failed()
            raise
        else:
            if response is not None:
                await cleanup_stream_sources(
                    response, iterator, cancelled=False
                )

    async def cancel(self, context: Any, event_queue: Any) -> None:
        _ensure_typing_override()
        task_id = cast(str, context.task_id)
        context_id = cast(str, context.context_id)
        updater_module = import_module("a2a.server.tasks.task_updater")
        updater = updater_module.TaskUpdater(
            event_queue, task_id=task_id, context_id=context_id
        )
        await updater.cancel()

    async def _orchestrator(self) -> Orchestrator:
        server_module = import_module("avalan.server")
        return cast(
            Orchestrator,
            await server_module.di_get_orchestrator_from_app(self._app),
        )

    async def _chat_request(
        self, context: Any, orchestrator: Orchestrator
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=resolve_model_id(orchestrator),
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=context.get_user_input(),
                )
            ],
            stream=True,
        )


class A2AResponseTranslator:
    """Translate canonical Avalan stream items to A2A SDK events."""

    def __init__(self, updater: Any) -> None:
        self._updater = updater
        self._a2a_pb2 = import_module("a2a.types.a2a_pb2")
        self._terminal_outcome: StreamTerminalOutcome | None = None
        self._open_artifacts: set[str] = set()

    @property
    def succeeded(self) -> bool:
        return stream_terminal_succeeded(self._terminal_outcome)

    async def process(self, item: object) -> None:
        if isinstance(item, CanonicalStreamItem):
            canonical_item = item
        elif isinstance(item, StreamConsumerProjection):
            canonical_item = canonical_item_from_consumer_projection(item)
        else:
            raise StreamValidationError("unsupported A2A stream item")
        await self._process_canonical_item(canonical_item)

    async def finish(self) -> None:
        for artifact_id in tuple(self._open_artifacts):
            await self._finish_artifact(artifact_id)
        if self._terminal_outcome is StreamTerminalOutcome.CANCELLED:
            await self._updater.cancel()
        elif self._terminal_outcome is StreamTerminalOutcome.ERRORED:
            await self._updater.failed()
        else:
            await self._updater.complete()

    async def _process_canonical_item(self, item: CanonicalStreamItem) -> None:
        if item.is_stream_terminal:
            self._terminal_outcome = item.terminal_outcome
            return
        if item.kind is StreamItemKind.ANSWER_DELTA:
            text = item.text_delta or ""
            if text:
                await self._add_text_artifact(
                    artifact_id="answer",
                    text=text,
                    metadata={"kind": "answer", "channel": "output"},
                    name="Answer",
                )
            return
        if item.kind is StreamItemKind.REASONING_DELTA:
            text = item.text_delta or ""
            if text:
                await self._add_text_artifact(
                    artifact_id="reasoning",
                    text=text,
                    metadata={"kind": "reasoning", "channel": "reasoning"},
                    name="Reasoning",
                )
            return
        if item.channel in (
            StreamChannel.TOOL_CALL,
            StreamChannel.TOOL_EXECUTION,
        ):
            await self._process_tool_item(item)

    async def _process_tool_item(self, item: CanonicalStreamItem) -> None:
        tool_call_id = item.correlation.tool_call_id or "tool"
        data = item.data if isinstance(item.data, dict) else {}
        metadata: dict[str, Any] = {
            "kind": "tool",
            "channel": item.channel.value,
            "phase": item.kind.value,
            "tool_call_id": tool_call_id,
        }
        tool_name = data.get("name") or item.metadata.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            metadata["tool_name"] = tool_name
        text = item.text_delta or ""
        if not text and data:
            text = str(data)
        if text:
            await self._add_text_artifact(
                artifact_id=tool_call_id,
                text=text,
                metadata=metadata,
                name=cast(str | None, tool_name),
            )
        await self._updater.update_status(
            self._a2a_pb2.TaskState.TASK_STATE_WORKING,
            metadata=metadata,
        )
        if item.kind in (
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            StreamItemKind.TOOL_EXECUTION_CANCELLED,
        ):
            await self._finish_artifact(tool_call_id)

    async def _add_text_artifact(
        self,
        *,
        artifact_id: str,
        text: str,
        metadata: dict[str, Any],
        name: str | None,
    ) -> None:
        append = artifact_id in self._open_artifacts
        self._open_artifacts.add(artifact_id)
        await self._updater.add_artifact(
            [self._a2a_pb2.Part(text=text)],
            artifact_id=artifact_id,
            name=name,
            metadata=metadata,
            append=append,
        )

    async def _finish_artifact(self, artifact_id: str) -> None:
        await self._updater.add_artifact(
            [],
            artifact_id=artifact_id,
            append=True,
            last_chunk=True,
        )
        self._open_artifacts.discard(artifact_id)
