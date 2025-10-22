from ..agent.loader import OrchestratorLoader
from ..agent.orchestrator import Orchestrator
from ..entities import OrchestratorSettings
from ..model.hubs.huggingface import HuggingfaceHub
from ..tool.context import ToolSettingsContext
from ..utils import logger_replace
from .a2a.store import TaskStore
from .entities import OrchestratorContext
from .routers import mcp as mcp_router

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from importlib import import_module
from logging import Logger
from typing import TYPE_CHECKING, Mapping
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from uvicorn import Server


_ALLOWED_PROTOCOLS = frozenset({"a2a", "mcp", "openai"})
_OPENAI_ENDPOINTS = frozenset({"completions", "responses"})


def _normalize_protocols(
    protocols: Mapping[str, set[str]] | None,
) -> dict[str, set[str]]:
    if protocols is None:
        return {
            "openai": set(_OPENAI_ENDPOINTS),
            "mcp": set(),
            "a2a": set(),
        }

    normalized: dict[str, set[str]] = {}
    for name, endpoints in protocols.items():
        protocol = name.lower()
        assert protocol in _ALLOWED_PROTOCOLS, f"Unsupported protocol '{name}'"
        if protocol == "openai":
            normalized_endpoints = {endpoint.lower() for endpoint in endpoints}
            if not normalized_endpoints:
                normalized_endpoints = set(_OPENAI_ENDPOINTS)
            else:
                missing = normalized_endpoints - _OPENAI_ENDPOINTS
                assert (
                    not missing
                ), f"Unsupported OpenAI endpoints: {sorted(missing)}"
            normalized[protocol] = {
                endpoint
                for endpoint in normalized_endpoints
                if endpoint in _OPENAI_ENDPOINTS
            }
        else:
            assert (
                not endpoints
            ), f"Protocol '{protocol}' does not accept endpoint selection"
            normalized[protocol] = set()
    return normalized


def _create_lifespan(
    *,
    hub: HuggingfaceHub,
    logger: Logger,
    specs_path: str | None,
    settings: OrchestratorSettings | None,
    tool_settings: ToolSettingsContext | None,
    mcp_prefix: str,
    mcp_name: str,
    mcp_description: str | None,
    a2a_tool_name: str,
    a2a_tool_description: str | None,
    selected_protocols: Mapping[str, set[str]],
    agent_id: UUID | None,
    participant_id: UUID | None,
) -> Callable[[FastAPI], AsyncIterator[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing app lifespan")
        from os import environ

        environ["TOKENIZERS_PARALLELISM"] = "false"
        async with AsyncExitStack() as stack:
            logger.info("Loading OrchestratorLoader in app lifespan")
            pid = participant_id or uuid4()
            loader = OrchestratorLoader(
                hub=hub,
                logger=logger,
                participant_id=pid,
                stack=stack,
            )
            tool_ctx = tool_settings
            ctx = OrchestratorContext(
                participant_id=pid,
                specs_path=specs_path,
                settings=settings,
                tool_settings=tool_ctx,
            )
            app.state.ctx = ctx
            app.state.stack = stack
            app.state.loader = loader
            app.state.logger = logger
            app.state.agent_id = agent_id
            if "a2a" in selected_protocols:
                app.state.a2a_store = TaskStore()
                app.state.a2a_tool_name = a2a_tool_name or "run"
                if a2a_tool_description:
                    app.state.a2a_tool_description = a2a_tool_description
            if "mcp" in selected_protocols:
                app.state.mcp_resource_store = mcp_router.MCPResourceStore()
                app.state.mcp_resource_base_path = mcp_prefix
                app.state.mcp_tool_name = mcp_name or "run"
                if mcp_description:
                    app.state.mcp_tool_description = mcp_description
            yield

    return lifespan


def _configure_cors(
    app: FastAPI,
    *,
    allow_origins: list[str] | None,
    allow_origin_regex: str | None,
    allow_methods: list[str] | None,
    allow_headers: list[str] | None,
    allow_credentials: bool,
) -> None:
    if any(
        [
            allow_origins,
            allow_origin_regex,
            allow_methods,
            allow_headers,
            allow_credentials,
        ]
    ):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins or [],
            allow_origin_regex=allow_origin_regex,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods or ["*"],
            allow_headers=allow_headers or ["*"],
        )


def _include_protocol_routers(
    app: FastAPI,
    *,
    selected_protocols: Mapping[str, set[str]],
    openai_prefix: str,
    mcp_prefix: str,
    a2a_prefix: str,
) -> None:
    openai_endpoints = selected_protocols.get("openai")
    if openai_endpoints:
        if "completions" in openai_endpoints:
            chat_router_module = import_module("avalan.server.routers.chat")
            app.include_router(chat_router_module.router, prefix=openai_prefix)
        if "responses" in openai_endpoints:
            responses_router_module = import_module(
                "avalan.server.routers.responses"
            )
            app.include_router(
                responses_router_module.router, prefix=openai_prefix
            )
        engine_router_module = import_module("avalan.server.routers.engine")
        app.include_router(engine_router_module.router)

    if "a2a" in selected_protocols:
        a2a_module = import_module("avalan.server.a2a")
        app.include_router(a2a_module.router, prefix=a2a_prefix)
        app.include_router(a2a_module.well_known_router)

    if "mcp" in selected_protocols:
        mcp_http_router = mcp_router.create_router()
        app.include_router(mcp_http_router, prefix=mcp_prefix)


def _attach_lifespan(
    app: FastAPI, lifespan: Callable[[FastAPI], AsyncIterator[None]]
) -> None:
    existing = app.router.lifespan_context

    if existing is None:
        app.router.lifespan_context = lifespan
        return

    @asynccontextmanager
    async def combined(app_: FastAPI):
        async with existing(app_):
            async with lifespan(app_):
                yield

    app.router.lifespan_context = combined


def register_agent_endpoints(
    app: FastAPI,
    *,
    hub: HuggingfaceHub,
    logger: Logger,
    specs_path: str | None,
    settings: OrchestratorSettings | None,
    tool_settings: ToolSettingsContext | None,
    mcp_prefix: str,
    openai_prefix: str,
    mcp_name: str,
    mcp_description: str | None = None,
    a2a_prefix: str = "/a2a",
    a2a_tool_name: str = "run",
    a2a_tool_description: str | None = None,
    agent_id: UUID | None = None,
    participant_id: UUID | None = None,
    allow_origins: list[str] | None = None,
    allow_origin_regex: str | None = None,
    allow_methods: list[str] | None = None,
    allow_headers: list[str] | None = None,
    allow_credentials: bool = False,
    protocols: Mapping[str, set[str]] | None = None,
) -> None:
    assert (specs_path is None) ^ (
        settings is None
    ), "Provide either specs_path or settings, but not both"

    selected_protocols = _normalize_protocols(protocols)

    lifespan = _create_lifespan(
        hub=hub,
        logger=logger,
        specs_path=specs_path,
        settings=settings,
        tool_settings=tool_settings,
        mcp_prefix=mcp_prefix,
        mcp_name=mcp_name,
        mcp_description=mcp_description,
        a2a_tool_name=a2a_tool_name,
        a2a_tool_description=a2a_tool_description,
        selected_protocols=selected_protocols,
        agent_id=agent_id,
        participant_id=participant_id,
    )

    _attach_lifespan(app, lifespan)
    _configure_cors(
        app,
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        allow_credentials=allow_credentials,
    )
    _include_protocol_routers(
        app,
        selected_protocols=selected_protocols,
        openai_prefix=openai_prefix,
        mcp_prefix=mcp_prefix,
        a2a_prefix=a2a_prefix,
    )


def agents_server(
    hub: HuggingfaceHub,
    name: str,
    version: str,
    host: str,
    port: int,
    reload: bool,
    specs_path: str | None,
    settings: OrchestratorSettings | None,
    tool_settings: ToolSettingsContext | None,
    mcp_prefix: str,
    openai_prefix: str,
    mcp_name: str,
    logger: Logger,
    mcp_description: str | None = None,
    a2a_prefix: str = "/a2a",
    a2a_tool_name: str = "run",
    a2a_tool_description: str | None = None,
    agent_id: UUID | None = None,
    participant_id: UUID | None = None,
    allow_origins: list[str] | None = None,
    allow_origin_regex: str | None = None,
    allow_methods: list[str] | None = None,
    allow_headers: list[str] | None = None,
    allow_credentials: bool = False,
    protocols: Mapping[str, set[str]] | None = None,
) -> "Server":
    """Build a configured Uvicorn server for Avalan agents."""
    assert (specs_path is None) ^ (
        settings is None
    ), "Provide either specs_path or settings, but not both"

    from uvicorn import Config, Server

    logger.debug("Creating %s server", name)
    selected_protocols = _normalize_protocols(protocols)
    lifespan = _create_lifespan(
        hub=hub,
        logger=logger,
        specs_path=specs_path,
        settings=settings,
        tool_settings=tool_settings,
        mcp_prefix=mcp_prefix,
        mcp_name=mcp_name,
        mcp_description=mcp_description,
        a2a_tool_name=a2a_tool_name,
        a2a_tool_description=a2a_tool_description,
        selected_protocols=selected_protocols,
        agent_id=agent_id,
        participant_id=participant_id,
    )
    app = FastAPI(title=name, version=version, lifespan=lifespan)

    _configure_cors(
        app,
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        allow_credentials=allow_credentials,
    )

    logger.debug("Adding routes to %s server", name)
    _include_protocol_routers(
        app,
        selected_protocols=selected_protocols,
        openai_prefix=openai_prefix,
        mcp_prefix=mcp_prefix,
        a2a_prefix=a2a_prefix,
    )

    logger.debug("Starting %s server at %s:%d", name, host, port)
    config = Config(app, host=host, port=port, reload=reload)
    server = Server(config)
    logger_replace(
        logger,
        [
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
            "uvicorn.asgi",
            "uvicorn.lifespan",
        ],
    )
    return server


def di_set(app: FastAPI, logger: Logger, orchestrator: Orchestrator) -> None:
    """Store dependencies on the application state."""
    assert logger is not None
    assert orchestrator is not None
    app.state.logger = logger
    app.state.orchestrator = orchestrator


def di_get_logger(request: Request) -> Logger:
    """Retrieve the application logger from the request."""
    assert hasattr(request.app.state, "logger")
    logger = request.app.state.logger
    assert isinstance(logger, Logger)
    return logger


async def di_get_orchestrator(request: Request) -> Orchestrator:
    """Retrieve the orchestrator from the request."""
    if not hasattr(request.app.state, "orchestrator"):
        ctx: OrchestratorContext = request.app.state.ctx
        loader: OrchestratorLoader = request.app.state.loader
        stack: AsyncExitStack = request.app.state.stack
        if ctx.specs_path:
            orchestrator_cm = await loader.from_file(
                ctx.specs_path,
                agent_id=request.app.state.agent_id,
                tool_settings=ctx.tool_settings,
            )
        else:
            assert ctx.settings
            orchestrator_cm = await loader.from_settings(
                ctx.settings,
                tool_settings=ctx.tool_settings,
            )
        orchestrator = await stack.enter_async_context(orchestrator_cm)
        request.app.state.orchestrator = orchestrator
        request.app.state.agent_id = orchestrator.id
    orchestrator = request.app.state.orchestrator
    assert orchestrator is not None
    return orchestrator
