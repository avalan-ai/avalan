from ...agent.loader import OrchestratorLoader
from ...agent.orchestrator import Orchestrator
from ...agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from ...cli import get_input, has_input
from ...cli.commands.model import token_generation
from ...cli.display import cli_stream_display_config
from ...cli.stream_coordinator import CliStreamCoordinator
from ...cli.theme import Theme
from ...entities import (
    Backend,
    EngineMessageScored,
    GenerationCacheStrategy,
    Message,
    Model,
    OrchestratorSettings,
    PermanentMemoryStoreSettings,
    ToolCall,
    ToolCallRecoveryFormat,
    ToolFormat,
)
from ...event import EventStats, EventType
from ...event.manager import EventManagerMode
from ...model.hubs.huggingface import HuggingfaceHub
from ...model.input import input_files
from ...model.nlp.text.generation import TextGenerationModel
from ...model.nlp.text.vendor import TextGenerationVendorModel
from ...model.response.text import TextGenerationResponse
from ...server import agents_server
from ...tool.browser import BrowserToolSettings
from ...tool.context import ToolSettingsContext
from ...tool.database.settings import DatabaseToolSettings
from ...tool.graph_settings import GraphToolSettings
from ...tool.shell import ShellToolSettings

from argparse import Namespace
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import fields
from logging import Logger
from os.path import dirname, getmtime, join
from typing import Any, cast, overload
from uuid import UUID, uuid4

from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax


class _Unset:
    pass


_UNSET = _Unset()


def _parse_permanent_memory_items(
    items: Iterable[str],
) -> dict[str, PermanentMemoryStoreSettings]:
    stores: dict[str, PermanentMemoryStoreSettings] = {}
    for item in items:
        namespace, value = item.split("@", 1)
        namespace = namespace.strip()
        assert namespace, "Permanent memory namespace must be provided"
        stores[namespace] = OrchestratorLoader.parse_permanent_store_value(
            value
        )
    return stores


def get_orchestrator_settings(
    args: Namespace,
    *,
    agent_id: UUID,
    name: str | None = None,
    role: str | None = None,
    task: str | None = None,
    instructions: str | None = None,
    goal_instructions: str | None = None,
    system: str | None = None,
    developer: str | None = None,
    user: str | None = None,
    user_template: str | None = None,
    engine_uri: str | None = None,
    memory_recent: bool | None = None,
    memory_permanent_message: str | None = None,
    memory_permanent: list[str] | None = None,
    maximum_tool_cycles: int | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    tools: list[str] | None | _Unset = _UNSET,
    tool_choice: str | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool | None = None,
    cache_strategy: GenerationCacheStrategy | None = None,
) -> OrchestratorSettings:
    """Create ``OrchestratorSettings`` from CLI arguments."""
    assert not (
        (user or getattr(args, "user", None))
        and (user_template or getattr(args, "user_template", None))
    )
    memory_recent = (
        memory_recent
        if memory_recent is not None
        else (
            args.memory_recent
            if args.memory_recent is not None
            else not getattr(args, "no_session", False)
        )
    )
    engine_uri = engine_uri or args.engine_uri
    call_tokens = (
        max_new_tokens
        if max_new_tokens is not None
        else args.run_max_new_tokens
    )
    call_maximum_tool_cycles = (
        maximum_tool_cycles
        if maximum_tool_cycles is not None
        else getattr(args, "run_maximum_tool_cycles", None)
    )
    call_tool_choice = (
        tool_choice
        if tool_choice is not None
        else getattr(args, "tool_choice", None)
    )

    chat_settings = {
        k[len("run_chat_") :]: v
        for k, v in vars(args).items()
        if k.startswith("run_chat_") and v is not None
    }
    reasoning_settings = {
        k[len("run_reasoning_") :]: v
        for k, v in vars(args).items()
        if k.startswith("run_reasoning_") and v is not None
    }
    call_options = {
        "max_new_tokens": call_tokens,
        "skip_special_tokens": args.run_skip_special_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        **({"chat_settings": chat_settings} if chat_settings else {}),
        **({"reasoning": reasoning_settings} if reasoning_settings else {}),
    }
    if use_cache is not None:
        call_options["use_cache"] = use_cache
    if cache_strategy is not None:
        call_options["cache_strategy"] = cache_strategy
    if call_maximum_tool_cycles is not None:
        call_options["maximum_tool_cycles"] = call_maximum_tool_cycles
    if call_tool_choice is not None:
        call_options["tool_choice"] = call_tool_choice
    engine_config: dict[str, Any] = {
        "backend": getattr(args, "backend", Backend.TRANSFORMERS.value)
    }
    engine_base_url = getattr(args, "engine_base_url", None)
    if engine_base_url is not None:
        engine_config["base_url"] = engine_base_url

    return OrchestratorSettings(
        agent_id=agent_id,
        orchestrator_type=None,
        agent_config={
            k: v
            for k, v in {
                "name": name if name is not None else args.name,
                "role": role if role is not None else args.role,
                "task": task if task is not None else args.task,
                "instructions": (
                    instructions
                    if instructions is not None
                    else getattr(args, "instructions", None)
                ),
                "goal_instructions": (
                    goal_instructions
                    if goal_instructions is not None
                    else getattr(args, "goal_instructions", None)
                ),
                "system": (
                    system
                    if system is not None
                    else getattr(args, "system", None)
                ),
                "developer": (
                    developer
                    if developer is not None
                    else getattr(args, "developer", None)
                ),
                "user": (
                    user if user is not None else getattr(args, "user", None)
                ),
                "user_template": (
                    user_template
                    if user_template is not None
                    else getattr(args, "user_template", None)
                ),
            }.items()
            if v is not None
        },
        uri=engine_uri,
        engine_config=engine_config,
        call_options=call_options,
        template_vars=None,
        memory_permanent_message=(
            memory_permanent_message
            if memory_permanent_message is not None
            else args.memory_permanent_message
        ),
        permanent_memory=(
            _parse_permanent_memory_items(memory_permanent)
            if memory_permanent is not None
            else (
                _parse_permanent_memory_items(args.memory_permanent)
                if args.memory_permanent
                else None
            )
        ),
        memory_recent=memory_recent,
        sentence_model_id=(
            args.memory_engine_model_id
            or OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID
        ),
        sentence_model_engine_config=None,
        sentence_model_max_tokens=args.memory_engine_max_tokens,
        sentence_model_overlap_size=args.memory_engine_overlap,
        sentence_model_window_size=args.memory_engine_window,
        json_config=None,
        tools=(
            tools
            if not isinstance(tools, _Unset)
            else (args.tool or []) + (getattr(args, "tools", None) or [])
        ),
        log_events=True,
    )


def _tool_settings_from_mapping(
    mapping: Mapping[str, object] | Namespace,
    *,
    prefix: str | None = None,
    settings_cls: (
        type[BrowserToolSettings]
        | type[DatabaseToolSettings]
        | type[GraphToolSettings]
        | type[ShellToolSettings]
    ),
    open_files: bool = True,
) -> (
    BrowserToolSettings
    | DatabaseToolSettings
    | GraphToolSettings
    | ShellToolSettings
    | None
):
    """Return tool settings from a mapping using dataclass ``settings_cls``."""
    values: dict[str, object] = {}
    for field in fields(settings_cls):
        key = f"tool_{prefix}_{field.name}" if prefix else field.name
        if isinstance(mapping, Namespace):
            if hasattr(mapping, key):
                value = getattr(mapping, key)
            else:
                continue
        else:
            if key in mapping:
                value = mapping[key]
            elif prefix and field.name in mapping:
                value = mapping[field.name]
            else:
                continue

        if value is not None:
            if (
                field.name == "debug_source"
                and open_files
                and isinstance(value, str)
            ):
                value = open(value)
            if settings_cls is DatabaseToolSettings:
                value = _coerce_database_tool_setting_value(
                    field.name,
                    value,
                    from_cli=isinstance(mapping, Namespace),
                )
            if settings_cls is ShellToolSettings:
                value = _coerce_shell_tool_setting_value(field.name, value)
            values[field.name] = value

    if not values:
        return None

    settings = cast(
        BrowserToolSettings
        | DatabaseToolSettings
        | GraphToolSettings
        | ShellToolSettings,
        cast(Any, settings_cls)(**values),
    )
    return settings


def _coerce_database_tool_setting_value(
    field_name: str, value: object, *, from_cli: bool
) -> object:
    if field_name != "allowed_commands":
        return value
    if from_cli and isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    return value


def _coerce_shell_tool_setting_value(field_name: str, value: object) -> object:
    if field_name == "executable_paths":
        return _coerce_shell_executable_paths(value)
    return value


def _coerce_shell_executable_paths(value: object) -> object:
    if isinstance(value, Mapping):
        return value
    if not _is_tuple_pair_sequence(value):
        return value

    executable_paths: dict[str, str] = {}
    for command, executable in cast(Sequence[tuple[str, str]], value):
        executable_paths[command] = executable
    return executable_paths


def _is_tuple_pair_sequence(
    value: object,
) -> bool:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return False
    return all(
        isinstance(item, tuple)
        and len(item) == 2
        and all(isinstance(part, str) for part in item)
        for item in value
    )


@overload
def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type[BrowserToolSettings],
    open_files: bool = True,
) -> BrowserToolSettings | None: ...


@overload
def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type[DatabaseToolSettings],
    open_files: bool = True,
) -> DatabaseToolSettings | None: ...


@overload
def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type[GraphToolSettings],
    open_files: bool = True,
) -> GraphToolSettings | None: ...


@overload
def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type[ShellToolSettings],
    open_files: bool = True,
) -> ShellToolSettings | None: ...


def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: (
        type[BrowserToolSettings]
        | type[DatabaseToolSettings]
        | type[GraphToolSettings]
        | type[ShellToolSettings]
    ),
    open_files: bool = True,
) -> (
    BrowserToolSettings
    | DatabaseToolSettings
    | GraphToolSettings
    | ShellToolSettings
    | None
):
    return _tool_settings_from_mapping(
        args, prefix=prefix, settings_cls=settings_cls, open_files=open_files
    )


def _shell_tool_template_settings(
    settings: ShellToolSettings | None,
) -> (
    dict[
        str,
        bool | int | float | str | tuple[str, ...] | dict[str, str],
    ]
    | None
):
    if settings is None:
        return None

    default_settings = ShellToolSettings()
    rendered: dict[
        str,
        bool | int | float | str | tuple[str, ...] | dict[str, str],
    ] = {}
    for field in fields(ShellToolSettings):
        name = field.name
        value = getattr(settings, name)
        if value == getattr(default_settings, name):
            continue
        if isinstance(value, bool | int | float | str):
            rendered[name] = value
        elif _is_simple_string_mapping(value):
            rendered[name] = dict(value)
        elif _is_simple_string_sequence(value):
            rendered[name] = tuple(value)
    return rendered


def _is_simple_string_mapping(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    )


def _is_simple_string_sequence(value: object) -> bool:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return False
    return all(isinstance(item, str) for item in value)


def _agent_enabled_tools(args: Namespace) -> list[str] | None:
    tools = (args.tool or []) + (getattr(args, "tools", None) or [])
    if tools:
        return tools
    return None


def _agent_tool_format(args: Namespace) -> ToolFormat | None:
    if getattr(args, "tool_format", None):
        return ToolFormat(args.tool_format)
    if not _agent_enabled_tools(args):
        return None
    backend = getattr(args, "backend", None)
    if backend in (
        Backend.DS4,
        Backend.DS4.value,
        Backend.MLXLM,
        Backend.MLXLM.value,
    ):
        return ToolFormat.REACT
    return None


def _agent_tool_settings(args: Namespace) -> ToolSettingsContext:
    return ToolSettingsContext(
        browser=get_tool_settings(
            args, prefix="browser", settings_cls=BrowserToolSettings
        ),
        database=get_tool_settings(
            args, prefix="database", settings_cls=DatabaseToolSettings
        ),
        graph=get_tool_settings(
            args, prefix="graph", settings_cls=GraphToolSettings
        ),
        shell=get_tool_settings(
            args, prefix="shell", settings_cls=ShellToolSettings
        ),
    )


async def _agent_run_input(
    input_string: str | None, file_paths: list[str] | None
) -> Message | str | None:
    """Build agent run input from text and local file paths."""
    return await input_files(input_string, file_paths)


def _uses_ds4_backend(orchestrator: Orchestrator) -> bool:
    """Return whether the active orchestrator engine uses DS4."""
    engine_agent = orchestrator.engine_agent
    if not engine_agent:
        return False

    engine_uri = getattr(engine_agent, "engine_uri", None)
    params = getattr(engine_uri, "params", None)
    if isinstance(params, dict):
        backend = params.get("backend")
        if backend == Backend.DS4 or backend == Backend.DS4.value:
            return True

    engine = orchestrator.engine
    model_type = getattr(engine, "model_type", None)
    return isinstance(model_type, str) and model_type.lower().startswith("ds4")


def _agent_display_models(
    orchestrator: Orchestrator,
    hub: HuggingfaceHub,
    *,
    is_local: bool,
) -> list[Model | str]:
    """Return model display payloads without querying DS4 local paths."""
    if not is_local or _uses_ds4_backend(orchestrator):
        return list(orchestrator.model_ids)

    return [hub.model(model_id) for model_id in orchestrator.model_ids]


async def agent_message_search(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
    refresh_per_second: int,
) -> None:
    _, _i = theme._, theme.icons

    specs_path = args.specifications_file
    engine_uri = getattr(args, "engine_uri", None)
    assert not (
        specs_path and engine_uri
    ), "specifications file and --engine-uri are mutually exclusive"
    assert (
        specs_path or engine_uri
    ), "specifications file or --engine-uri must be specified"
    agent_id = args.id
    participant_id = args.participant
    session_id = args.session

    assert agent_id and participant_id and session_id

    tty_path = getattr(args, "tty", "/dev/tty") or "/dev/tty"

    input_string = get_input(
        console,
        _i["user_input"] + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
        tty_path=tty_path,
    )
    if not input_string:
        return

    limit = args.limit

    async with AsyncExitStack() as stack:
        loader = OrchestratorLoader(
            hub=hub,
            logger=logger,
            participant_id=participant_id,
            stack=stack,
        )
        with console.status(
            _("Loading agent..."),
            spinner=theme.get_spinner("agent_loading") or "dots",
            refresh_per_second=refresh_per_second,
        ):
            if specs_path:
                logger.debug(
                    "Loading agent from %s for participant %s",
                    specs_path,
                    participant_id,
                )

                orchestrator = await loader.from_file(
                    specs_path,
                    agent_id=agent_id,
                    tool_settings=_agent_tool_settings(args),
                    event_manager_mode=EventManagerMode.CLI,
                )
            else:
                assert (
                    args.engine_uri
                ), "--engine-uri required when no specifications file"
                logger.debug("Loading agent from inline settings")
                tool_settings = _agent_tool_settings(args)
                memory_recent = (
                    args.memory_recent
                    if args.memory_recent is not None
                    else True
                )
                settings = get_orchestrator_settings(
                    args,
                    agent_id=agent_id,
                    memory_recent=memory_recent,
                    tools=_agent_enabled_tools(args),
                )
                orchestrator = await loader.from_settings(
                    settings,
                    tool_settings=tool_settings,
                    event_manager_mode=EventManagerMode.CLI,
                )
            orchestrator = await stack.enter_async_context(orchestrator)

            assert orchestrator.engine_agent
            assert orchestrator.engine and orchestrator.engine.model_id

            is_local = not isinstance(
                orchestrator.engine, TextGenerationVendorModel
            )
            can_access = (
                True
                if _uses_ds4_backend(orchestrator)
                else args.skip_hub_access_check
                or hub.can_access(orchestrator.engine.model_id)
            )
            models = _agent_display_models(
                orchestrator, hub, is_local=is_local
            )

            console.print(
                theme.agent(orchestrator, models=models, can_access=can_access)
            )

            logger.debug(
                'Searching for "%s" across messages on session %s between '
                "agent %s and participant %s",
                input_string,
                session_id,
                agent_id,
                participant_id,
            )
            messages = await orchestrator.memory.search_messages(
                search=input_string,
                agent_id=agent_id,
                search_user_messages=False,
                session_id=session_id,
                participant_id=participant_id,
                function=args.function,
                limit=limit,
            )
            console.print(
                theme.search_message_matches(
                    participant_id,
                    orchestrator,
                    cast(list[EngineMessageScored], messages),
                )
            )


async def agent_run(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
    refresh_per_second: int,
) -> None:
    _, _i = theme._, theme.icons
    display_config = cli_stream_display_config(
        args,
        refresh_per_second=refresh_per_second,
        interactive=bool(getattr(console, "is_terminal", True)),
    )

    specs_path = args.specifications_file
    engine_uri = getattr(args, "engine_uri", None)
    assert not (
        specs_path and engine_uri
    ), "specifications file and --engine-uri are mutually exclusive"
    assert (
        specs_path or engine_uri
    ), "specifications file or --engine-uri must be specified"
    use_async_generator = not args.use_sync_generator
    display_tokens = display_config.display_tokens
    dtokens_pick = 10 if display_tokens > 0 else 0
    with_stats = display_config.show_stats
    agent_id = args.id
    participant_id = args.participant
    session_id = args.session if not args.no_session else None
    load_recent_messages = (
        not args.skip_load_recent_messages and not args.no_session
    )
    load_recent_messages_limit = args.load_recent_messages_limit

    event_stats = EventStats()
    tty_path = getattr(args, "tty", "/dev/tty") or "/dev/tty"
    coordinator_container: dict[str, CliStreamCoordinator | None] = {
        "coordinator": None
    }

    async def _confirm_call(call: ToolCall) -> str:
        coordinator = coordinator_container["coordinator"]
        if coordinator is None:
            async with CliStreamCoordinator(
                console,
                display_config,
            ) as fallback_coordinator:
                return await fallback_coordinator.confirm_tool_call(
                    call,
                    tty_path=tty_path,
                )
        return await coordinator.confirm_tool_call(call, tty_path=tty_path)

    async def _event_listener(event: object) -> None:
        nonlocal event_stats
        event_type = getattr(event, "type", None)
        assert isinstance(event_type, EventType)
        event_stats.record_trigger(event_type)

    async def _init_orchestrator() -> Orchestrator:
        loader = OrchestratorLoader(
            hub=hub,
            logger=logger,
            participant_id=participant_id,
            stack=stack,
        )
        if specs_path:
            logger.debug(
                "Loading agent from %s for participant %s",
                specs_path,
                participant_id,
            )

            orchestrator = await loader.from_file(
                specs_path,
                agent_id=agent_id,
                disable_memory=args.no_session,
                tool_settings=_agent_tool_settings(args),
                event_manager_mode=EventManagerMode.CLI,
            )
        else:
            assert (
                args.engine_uri
            ), "--engine-uri required when no specifications file"
            assert not args.specifications_file or not args.engine_uri
            tool_settings = _agent_tool_settings(args)
            memory_recent = (
                args.memory_recent
                if args.memory_recent is not None
                else not args.no_session
            )
            settings = get_orchestrator_settings(
                args,
                agent_id=agent_id or uuid4(),
                memory_recent=memory_recent,
                tools=_agent_enabled_tools(args),
                max_new_tokens=getattr(args, "run_max_new_tokens", None),
                temperature=getattr(args, "run_temperature", None),
                top_k=getattr(args, "run_top_k", None),
                top_p=getattr(args, "run_top_p", None),
                use_cache=getattr(args, "run_use_cache", None),
                cache_strategy=getattr(args, "run_cache_strategy", None),
            )
            logger.debug("Loading agent from inline settings")
            tool_format = _agent_tool_format(args)
            tool_recovery_formats = [
                ToolCallRecoveryFormat(value)
                for value in getattr(args, "tool_recovery_format", None) or []
            ]
            if tool_recovery_formats:
                orchestrator = await loader.from_settings(
                    settings,
                    tool_settings=tool_settings,
                    tool_format=tool_format,
                    tool_recovery_formats=tool_recovery_formats,
                    event_manager_mode=EventManagerMode.CLI,
                )
            else:
                orchestrator = await loader.from_settings(
                    settings,
                    tool_settings=tool_settings,
                    tool_format=tool_format,
                    event_manager_mode=EventManagerMode.CLI,
                )
        event_manager = orchestrator.event_manager

        def event_manager_method(name: str) -> Any | None:
            method = getattr(event_manager, name, None)
            has_method = callable(
                getattr(type(event_manager), name, None)
            ) or name in getattr(event_manager, "__dict__", {})
            return method if has_method and callable(method) else None

        add_ui_listener = event_manager_method("add_ui_listener")
        if add_ui_listener is not None:
            add_ui_listener(_event_listener)
        else:
            add_listener = event_manager_method("add_listener")
            assert callable(add_listener)
            add_listener(_event_listener)
        remove_listener = event_manager_method("remove_listener")
        register_cleanup = getattr(stack, "callback", None)
        has_cleanup = callable(
            getattr(type(stack), "callback", None)
        ) or "callback" in getattr(stack, "__dict__", {})
        if remove_listener is not None and has_cleanup:
            assert callable(register_cleanup)
            register_cleanup(remove_listener, _event_listener)

        orchestrator = await stack.enter_async_context(orchestrator)

        if args.tools_confirm:
            assert (
                not orchestrator.tool.is_empty
            ), "--tools-confirm requires tools"

        logger.debug(
            "Agent loaded from %s, models used: %s, with recent message "
            "memory: %s, with permanent message memory: %s",
            specs_path,
            orchestrator.model_ids,
            "yes" if orchestrator.memory.has_recent_message else "no",
            (
                "yes, with session #"
                + str(orchestrator.memory.permanent_message.session_id)
                if (
                    orchestrator.memory.has_permanent_message
                    and orchestrator.memory.permanent_message is not None
                )
                else "no"
            ),
        )

        if not display_config.answer_stdout_only:
            assert orchestrator.engine_agent
            assert orchestrator.engine and orchestrator.engine.model_id

            is_local = not isinstance(
                orchestrator.engine, TextGenerationVendorModel
            )

            can_access = (
                True
                if _uses_ds4_backend(orchestrator)
                else args.skip_hub_access_check
                or not is_local
                or hub.can_access(orchestrator.engine.model_id)
            )
            models = _agent_display_models(
                orchestrator, hub, is_local=is_local
            )

            console.print(
                theme.agent(orchestrator, models=models, can_access=can_access)
            )

        if not args.no_session:
            if session_id:
                await orchestrator.memory.continue_session(
                    session_id=session_id,
                    load_recent_messages=load_recent_messages,
                    load_recent_messages_limit=load_recent_messages_limit,
                )
            else:
                await orchestrator.memory.start_session()

        if (
            load_recent_messages
            and orchestrator.memory.has_recent_message
            and not display_config.answer_stdout_only
        ):
            recent_message = orchestrator.memory.recent_message
            assert recent_message is not None
            if recent_message.is_empty:
                return orchestrator
            console.print(
                theme.recent_messages(
                    participant_id,
                    orchestrator,
                    recent_message.data,
                )
            )

        return orchestrator

    async with AsyncExitStack() as stack:
        if display_config.answer_stdout_only:
            orchestrator = await _init_orchestrator()
        else:
            with console.status(
                _("Loading agent..."),
                spinner=theme.get_spinner("agent_loading") or "dots",
                refresh_per_second=display_config.refresh_per_second,
            ):
                orchestrator = await _init_orchestrator()

        watch_spec = bool(specs_path and args.conversation and args.watch)
        if watch_spec:
            specs_mtime = getmtime(specs_path)

        input_string: str | None = None
        input_file_paths = cast(
            list[str] | None, getattr(args, "input_file", None)
        )
        in_conversation = False
        while not input_string or in_conversation:
            current_input_file_paths = (
                None if in_conversation else input_file_paths
            )
            if watch_spec and not has_input(console):
                new_mtime = getmtime(specs_path)
                if new_mtime != specs_mtime:
                    logger.debug("Reloading agent from %s", specs_path)
                    orchestrator = await _init_orchestrator()
                    specs_mtime = new_mtime
                    in_conversation = False
                    continue
            logger.debug(
                "Waiting for new message to add to orchestrator's existing "
                + str(orchestrator.memory.recent_message.size)
                if orchestrator.memory
                and orchestrator.memory.has_recent_message
                and orchestrator.memory.recent_message is not None
                else "0" + " messages"
            )
            input_string = get_input(
                console,
                _i["user_input"] + " ",
                echo_stdin=not args.no_repl,
                force_prompt=in_conversation,
                is_quiet=display_config.answer_stdout_only,
                tty_path=tty_path,
            )
            if not input_string and not current_input_file_paths:
                logger.debug("Finishing session with orchestrator")
                return

            logger.debug('Agent about to process input "%s"', input_string)
            agent_input = await _agent_run_input(
                input_string, current_input_file_paths
            )
            assert agent_input is not None
            output = await orchestrator(
                agent_input,
                use_async_generator=use_async_generator,
                tool_confirm=_confirm_call if args.tools_confirm else None,
            )

            if (
                not display_config.answer_stdout_only
                and not display_config.show_stats
                and not (
                    isinstance(theme, Theme) and theme.prefix_stream_answers
                )
            ):
                console.print(_i["agent_output"] + " ", end="")

            assert isinstance(output, OrchestratorResponse)
            assert orchestrator.engine is not None
            text_output = cast(TextGenerationResponse, output)
            text_engine = cast(TextGenerationModel, orchestrator.engine)

            await token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=orchestrator,
                event_stats=event_stats,
                lm=text_engine,
                input_string=input_string or "",
                refresh_per_second=display_config.refresh_per_second,
                response=text_output,
                dtokens_pick=dtokens_pick,
                display_tokens=display_tokens,
                tool_events_limit=display_config.display_tools_events,
                with_stats=with_stats,
                coordinator_container=coordinator_container,
                display_config=display_config,
                answer_prefix=(
                    _i["agent_output"] + " "
                    if isinstance(theme, Theme)
                    and theme.prefix_stream_answers
                    and not display_config.answer_stdout_only
                    else None
                ),
            )

            if args.conversation:
                console.print("")
                if not in_conversation:
                    in_conversation = True
            else:
                break


async def agent_serve(
    args: Namespace,
    hub: HuggingfaceHub,
    logger: Logger,
    name: str,
    version: str,
) -> None:
    assert args.host and args.port
    specs_path = args.specifications_file
    agent_id = getattr(args, "id", None)
    participant_id = args.participant
    engine_uri = getattr(args, "engine_uri", None)
    assert not (
        specs_path and engine_uri
    ), "specifications file and --engine-uri are mutually exclusive"
    assert (
        specs_path or engine_uri
    ), "specifications file or --engine-uri must be specified"

    settings: OrchestratorSettings | None = None
    tool_settings = _agent_tool_settings(args)

    protocols = await OrchestratorLoader.resolve_serve_protocols(
        specs_path=specs_path,
        cli_protocols=getattr(args, "protocol", None),
    )

    if not specs_path:
        memory_recent = (
            args.memory_recent if args.memory_recent is not None else True
        )
        settings = get_orchestrator_settings(
            args,
            agent_id=agent_id or uuid4(),
            memory_recent=memory_recent,
            tools=_agent_enabled_tools(args),
        )

    server = agents_server(
        hub=hub,
        name=name,
        version=version,
        mcp_prefix=getattr(args, "mcp_prefix", "/mcp") or "/mcp",
        openai_prefix=args.openai_prefix,
        a2a_prefix=getattr(args, "a2a_prefix", "/a2a") or "/a2a",
        mcp_name=getattr(args, "mcp_name", "run") or "run",
        mcp_description=getattr(args, "mcp_description", None),
        a2a_tool_name=getattr(args, "a2a_name", "run") or "run",
        a2a_tool_description=getattr(args, "a2a_description", None),
        specs_path=specs_path,
        settings=settings,
        tool_settings=tool_settings,
        host=args.host,
        port=args.port,
        reload=args.reload,
        logger=logger,
        agent_id=agent_id,
        participant_id=participant_id,
        allow_origins=args.cors_origin,
        allow_origin_regex=args.cors_origin_regex,
        allow_methods=args.cors_method,
        allow_headers=args.cors_header,
        allow_credentials=args.cors_credentials,
        protocols=protocols,
    )
    await server.serve()


async def agent_proxy(
    args: Namespace,
    hub: HuggingfaceHub,
    logger: Logger,
    name: str,
    version: str,
) -> None:
    args.name = getattr(args, "name", "Proxy") or "Proxy"
    args.memory_recent = getattr(args, "memory_recent", True) or True
    args.memory_permanent_message = (
        getattr(args, "memory_permanent_message", None)
        or "postgresql://avalan:password@localhost:5432/avalan"
    )
    args.specifications_file = None

    assert getattr(args, "engine_uri", None), "--engine-uri is required"

    await agent_serve(args, hub, logger, name, version)


async def agent_init(args: Namespace, console: Console, theme: Theme) -> None:
    _ = theme._
    tty_path = getattr(args, "tty", "/dev/tty") or "/dev/tty"

    name = args.name or Prompt.ask(_("Agent name"))
    role = args.role or get_input(
        console,
        _("Agent role") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
        tty_path=tty_path,
    )

    task = args.task or get_input(
        console,
        _("Agent task") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
        tty_path=tty_path,
    )
    goal_instructions = getattr(args, "goal_instructions", None) or get_input(
        console,
        _("Agent goal instructions") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
        tty_path=tty_path,
    )

    memory_recent = (
        args.memory_recent
        if args.memory_recent is not None
        else Confirm.ask(_("Use recent message memory?"))
    )
    memory_permanent_message = (
        args.memory_permanent_message
        if args.memory_permanent_message is not None
        else Prompt.ask(_("Permanent memory DSN"), default="")
    )
    engine_uri = args.engine_uri or Prompt.ask(
        _("Engine URI"),
        default="microsoft/Phi-4-mini-instruct",
    )

    settings = get_orchestrator_settings(
        args,
        agent_id=uuid4(),
        name=name,
        role=role,
        task=task,
        instructions=getattr(args, "instructions", None),
        goal_instructions=goal_instructions,
        engine_uri=engine_uri,
        memory_recent=memory_recent,
        memory_permanent_message=memory_permanent_message,
        max_new_tokens=(
            args.run_max_new_tokens
            if args.run_max_new_tokens is not None
            else 1024
        ),
        tools=(args.tool or []) + (getattr(args, "tools", None) or []),
        temperature=getattr(args, "run_temperature", None),
        top_k=getattr(args, "run_top_k", None),
        top_p=getattr(args, "run_top_p", None),
        use_cache=getattr(args, "run_use_cache", None),
        cache_strategy=getattr(args, "run_cache_strategy", None),
    )

    browser_tool = get_tool_settings(
        args,
        prefix="browser",
        settings_cls=BrowserToolSettings,
        open_files=False,
    )
    database_tool = get_tool_settings(
        args,
        prefix="database",
        settings_cls=DatabaseToolSettings,
        open_files=False,
    )
    graph_tool = get_tool_settings(
        args,
        prefix="graph",
        settings_cls=GraphToolSettings,
        open_files=False,
    )
    shell_tool = get_tool_settings(
        args,
        prefix="shell",
        settings_cls=ShellToolSettings,
        open_files=False,
    )

    env = Environment(
        loader=FileSystemLoader(
            join(dirname(__file__), "..", "..", "agent", "templates")
        ),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("blueprint.toml")
    tool_format = getattr(args, "tool_format", None)
    tool_recovery_formats = getattr(args, "tool_recovery_format", None) or []
    rendered = template.render(
        orchestrator=settings,
        browser_tool=browser_tool,
        database_tool=database_tool,
        graph_tool=graph_tool,
        shell_tool=_shell_tool_template_settings(shell_tool),
        tool_format=tool_format,
        tool_recovery_formats=tool_recovery_formats,
    )
    console.print(Syntax(rendered, "toml"))
