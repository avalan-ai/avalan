from ...agent.loader import OrchestratorLoader
from ...agent.orchestrator import Orchestrator
from ...agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from ...cli import get_input, has_input, confirm_tool_call
from ...cli.commands.model import token_generation
from ...entities import OrchestratorSettings, ToolCall
from ...event import EventStats
from ...model.hubs.huggingface import HuggingfaceHub
from ...model.nlp.text.vendor import TextGenerationVendorModel
from ...server import agents_server
from ...tool.browser import BrowserToolSettings
from argparse import Namespace
from contextlib import AsyncExitStack
from dataclasses import fields
from jinja2 import Environment, FileSystemLoader
from logging import Logger
from os.path import dirname, join, getmtime
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.theme import Theme
from typing import Mapping
from uuid import UUID, uuid4


def get_orchestrator_settings(
    args: Namespace,
    *,
    agent_id: UUID,
    name: str | None = None,
    role: str | None = None,
    task: str | None = None,
    instructions: str | None = None,
    engine_uri: str | None = None,
    memory_recent: bool | None = None,
    memory_permanent_message: str | None = None,
    memory_permanent: list[str] | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    tools: list[str] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> OrchestratorSettings:
    """Create ``OrchestratorSettings`` from CLI arguments."""
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

    chat_template_settings = {
        k[len("run_chat_") :]: v
        for k, v in vars(args).items()
        if k.startswith("run_chat_") and v is not None
    }

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
                    else args.instructions
                ),
            }.items()
            if v is not None
        },
        uri=engine_uri,
        engine_config=None,
        call_options={
            "max_new_tokens": call_tokens,
            "skip_special_tokens": args.run_skip_special_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            **(
                {"chat_template_settings": chat_template_settings}
                if chat_template_settings
                else {}
            ),
        },
        template_vars=None,
        memory_permanent_message=(
            memory_permanent_message
            if memory_permanent_message is not None
            else args.memory_permanent_message
        ),
        permanent_memory=(
            {
                ns: dsn
                for ns, dsn in (
                    item.split("@", 1)
                    for item in (
                        memory_permanent or args.memory_permanent or []
                    )
                )
            }
            if (memory_permanent or args.memory_permanent)
            else None
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
        tools=tools if tools is not None else args.tool or [],
        log_events=True,
    )


def _tool_settings_from_mapping(
    mapping: Mapping[str, object] | Namespace,
    *,
    prefix: str | None = None,
    settings_cls: type,
    open_files: bool = True,
) -> object:
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
            values[field.name] = value

    if not values:
        return None

    return settings_cls(**values)


def get_tool_settings(
    args: Namespace,
    *,
    prefix: str,
    settings_cls: type,
    open_files: bool = True,
) -> object:
    return _tool_settings_from_mapping(
        args, prefix=prefix, settings_cls=settings_cls, open_files=open_files
    )


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

    input_string = get_input(
        console,
        _i["user_input"] + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
    )
    if not input_string:
        return

    limit = args.limit

    async with AsyncExitStack() as stack:
        with console.status(
            _("Loading agent..."),
            spinner=theme.get_spinner("agent_loading"),
            refresh_per_second=refresh_per_second,
        ):
            if specs_path:
                logger.debug(
                    "Loading agent from %s for participant %s",
                    specs_path,
                    participant_id,
                )

                orchestrator = await OrchestratorLoader.from_file(
                    specs_path,
                    agent_id=agent_id,
                    hub=hub,
                    logger=logger,
                    participant_id=participant_id,
                    stack=stack,
                )
            else:
                assert (
                    args.engine_uri
                ), "--engine-uri required when no specifications file"
                logger.debug("Loading agent from inline settings")
                memory_recent = (
                    args.memory_recent
                    if args.memory_recent is not None
                    else True
                )
                settings = get_orchestrator_settings(
                    args,
                    agent_id=agent_id,
                    memory_recent=memory_recent,
                    tools=args.tool,
                )
                browser_settings = get_tool_settings(
                    args, prefix="browser", settings_cls=BrowserToolSettings
                )
                orchestrator = await OrchestratorLoader.load_from_settings(
                    settings,
                    hub=hub,
                    logger=logger,
                    participant_id=participant_id,
                    stack=stack,
                    browser_settings=browser_settings,
                )
            orchestrator = await stack.enter_async_context(orchestrator)

            assert orchestrator.engine_agent and orchestrator.engine.model_id

            can_access = args.skip_hub_access_check or hub.can_access(
                orchestrator.engine.model_id
            )
            is_local = not isinstance(
                orchestrator.engine, TextGenerationVendorModel
            )
            models = [
                hub.model(model_id) if is_local else model_id
                for model_id in orchestrator.model_ids
            ]

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
                    participant_id, orchestrator, messages
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

    specs_path = args.specifications_file
    engine_uri = getattr(args, "engine_uri", None)
    assert not (
        specs_path and engine_uri
    ), "specifications file and --engine-uri are mutually exclusive"
    assert (
        specs_path or engine_uri
    ), "specifications file or --engine-uri must be specified"
    use_async_generator = not args.use_sync_generator
    display_tokens = args.display_tokens or 0
    dtokens_pick = 10 if display_tokens > 0 else 0
    with_stats = args.stats and not args.quiet
    agent_id = args.id
    participant_id = args.participant
    session_id = args.session if not args.no_session else None
    load_recent_messages = (
        not args.skip_load_recent_messages and not args.no_session
    )
    load_recent_messages_limit = args.load_recent_messages_limit

    event_stats = EventStats()

    def _confirm_call(call: ToolCall) -> str:
        return confirm_tool_call(console, call)

    async def _event_listener(event):
        nonlocal event_stats
        event_stats.total_triggers += 1
        if event.type not in event_stats.triggers:
            event_stats.triggers[event.type] = 1
        else:
            event_stats.triggers[event.type] += 1

    async def _init_orchestrator() -> Orchestrator:
        if specs_path:
            logger.debug(
                "Loading agent from %s for participant %s",
                specs_path,
                participant_id,
            )

            orchestrator = await OrchestratorLoader.from_file(
                specs_path,
                agent_id=agent_id,
                hub=hub,
                logger=logger,
                participant_id=participant_id,
                stack=stack,
                disable_memory=args.no_session,
            )
        else:
            assert (
                args.engine_uri and args.role
            ), "--engine-uri and --role required when no specifications file"
            assert not args.specifications_file or not args.engine_uri
            memory_recent = (
                args.memory_recent
                if args.memory_recent is not None
                else not args.no_session
            )
            settings = get_orchestrator_settings(
                args,
                agent_id=agent_id or uuid4(),
                memory_recent=memory_recent,
                tools=args.tool,
            )
            logger.debug("Loading agent from inline settings")
            browser_settings = get_tool_settings(
                args, prefix="browser", settings_cls=BrowserToolSettings
            )
            orchestrator = await OrchestratorLoader.load_from_settings(
                settings,
                hub=hub,
                logger=logger,
                participant_id=participant_id,
                stack=stack,
                browser_settings=browser_settings,
            )
        orchestrator.event_manager.add_listener(_event_listener)

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
                if orchestrator.memory.has_permanent_message
                else "no"
            ),
        )

        if not args.quiet:
            assert orchestrator.engine_agent and orchestrator.engine.model_id

            is_local = not isinstance(
                orchestrator.engine, TextGenerationVendorModel
            )

            can_access = (
                args.skip_hub_access_check
                or not is_local
                or hub.can_access(orchestrator.engine.model_id)
            )
            models = [
                hub.model(model_id) if is_local else model_id
                for model_id in orchestrator.model_ids
            ]

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
            and not orchestrator.memory.recent_message.is_empty
            and not args.quiet
        ):
            console.print(
                theme.recent_messages(
                    participant_id,
                    orchestrator,
                    orchestrator.memory.recent_message.data,
                )
            )

        return orchestrator

    async with AsyncExitStack() as stack:
        with console.status(
            _("Loading agent..."),
            spinner=theme.get_spinner("agent_loading"),
            refresh_per_second=refresh_per_second,
        ):
            orchestrator = await _init_orchestrator()

        watch_spec = bool(specs_path and args.conversation and args.watch)
        if watch_spec:
            specs_mtime = getmtime(specs_path)

        input_string: str | None = None
        in_conversation = False
        while not input_string or in_conversation:
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
                else "0" + " messages"
            )
            input_string = get_input(
                console,
                _i["user_input"] + " ",
                echo_stdin=not args.no_repl,
                force_prompt=in_conversation,
                is_quiet=args.quiet,
                tty_path=args.tty,
            )
            if not input_string:
                logger.debug("Finishing session with orchestrator")
                return

            logger.debug('Agent about to process input "%s"', input_string)
            output = await orchestrator(
                input_string,
                use_async_generator=use_async_generator,
                tool_confirm=_confirm_call if args.tools_confirm else None,
            )

            if not args.quiet and not args.stats:
                console.print(_i["agent_output"] + " ", end="")

            if args.quiet:
                console.print(await output.to_str())
                return

            assert isinstance(output, OrchestratorResponse)

            await token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=orchestrator,
                event_stats=event_stats,
                lm=orchestrator.engine,
                input_string=input_string,
                refresh_per_second=refresh_per_second,
                response=output,
                dtokens_pick=dtokens_pick,
                display_tokens=display_tokens,
                tool_events_limit=args.display_tools_events,
                with_stats=with_stats,
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
    engine_uri = getattr(args, "engine_uri", None)
    assert not (
        specs_path and engine_uri
    ), "specifications file and --engine-uri are mutually exclusive"
    assert (
        specs_path or engine_uri
    ), "specifications file or --engine-uri must be specified"

    async with AsyncExitStack() as stack:
        if specs_path:
            logger.debug("Loading agent from %s", specs_path)

            orchestrator = await OrchestratorLoader.from_file(
                specs_path,
                agent_id=uuid4(),
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
            )
        else:
            assert args.role
            memory_recent = (
                args.memory_recent if args.memory_recent is not None else True
            )
            settings = get_orchestrator_settings(
                args,
                agent_id=uuid4(),
                memory_recent=memory_recent,
                tools=args.tool,
            )
            logger.debug("Loading agent from inline settings")
            browser_settings = get_tool_settings(
                args, prefix="browser", settings_cls=BrowserToolSettings
            )
            orchestrator = await OrchestratorLoader.load_from_settings(
                settings,
                hub=hub,
                logger=logger,
                participant_id=uuid4(),
                stack=stack,
                browser_settings=browser_settings,
            )
        orchestrator = await stack.enter_async_context(orchestrator)

        logger.debug(
            "Agent loaded from"
            f" {specs_path if specs_path else 'inline settings'}"
        )
        server = agents_server(
            name=name,
            version=version,
            prefix_openai=args.prefix_openai,
            prefix_mcp=args.prefix_mcp,
            orchestrators=[orchestrator],
            host=args.host,
            port=args.port,
            reload=args.reload,
            logger=logger,
        )
        await server.serve()


async def agent_init(args: Namespace, console: Console, theme: Theme) -> None:
    _ = theme._

    name = args.name or Prompt.ask(_("Agent name"))
    role = args.role or get_input(
        console,
        _("Agent role") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
    )
    if not role:
        return

    task = args.task or get_input(
        console,
        _("Agent task") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
    )
    instructions = args.instructions or get_input(
        console,
        _("Agent instructions") + " ",
        echo_stdin=not args.no_repl,
        is_quiet=args.quiet,
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
        instructions=instructions,
        engine_uri=engine_uri,
        memory_recent=memory_recent,
        memory_permanent_message=memory_permanent_message,
        max_new_tokens=args.run_max_new_tokens or 1024,
        tools=args.tool or [],
    )

    browser_tool = get_tool_settings(
        args,
        prefix="browser",
        settings_cls=BrowserToolSettings,
        open_files=False,
    )

    env = Environment(
        loader=FileSystemLoader(join(dirname(__file__), "..", "..", "agent")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("blueprint.toml")
    rendered = template.render(
        orchestrator=settings, browser_tool=browser_tool
    )
    console.print(Syntax(rendered, "toml"))
