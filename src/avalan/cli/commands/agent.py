from argparse import Namespace
from ...agent.loader import OrchestrationLoader
from ...agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from ...cli import get_input
from ...cli.commands.model import token_generation
from ...event import EventStats
from ...memory.permanent import VectorFunction
from ...model.hubs.huggingface import HuggingfaceHub
from ...model.nlp.text.vendor import TextGenerationVendorModel
from ...server import agents_server
from contextlib import AsyncExitStack
from logging import Logger
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.theme import Theme
from typing import Optional
from uuid import uuid4
from jinja2 import Environment, FileSystemLoader
from os.path import dirname, join


async def agent_message_search(
    args: Namespace,
    console: Console,
    theme: Theme,
    hub: HuggingfaceHub,
    logger: Logger,
    refresh_per_second: int,
) -> None:
    assert args.specifications_file
    _, _i = theme._, theme.icons

    specs_path = args.specifications_file
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
            logger.debug(
                f"Loading agent from {specs_path} for "
                f"participant {participant_id}"
            )

            orchestrator = await OrchestrationLoader.from_file(
                specs_path,
                agent_id=agent_id,
                hub=hub,
                logger=logger,
                participant_id=participant_id,
                stack=stack,
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

            messages = await orchestrator.memory.search_messages(
                search=input_string,
                agent_id=agent_id,
                session_id=session_id,
                participant_id=participant_id,
                function=VectorFunction.L2_DISTANCE,
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
    assert args.specifications_file
    _, _i = theme._, theme.icons

    specs_path = args.specifications_file
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

    async def _event_listener(event):
        nonlocal event_stats
        event_stats.total_triggers += 1
        if event.type not in event_stats.triggers:
            event_stats.triggers[event.type] = 1
        else:
            event_stats.triggers[event.type] += 1

    async with AsyncExitStack() as stack:
        with console.status(
            _("Loading agent..."),
            spinner=theme.get_spinner("agent_loading"),
            refresh_per_second=refresh_per_second,
        ):
            logger.debug(
                f"Loading agent from {specs_path} for "
                f"participant {participant_id}"
            )

            orchestrator = await OrchestrationLoader.from_file(
                specs_path,
                agent_id=agent_id,
                hub=hub,
                logger=logger,
                participant_id=participant_id,
                stack=stack,
                disable_memory=args.no_session,
            )
            orchestrator.event_manager.add_listener(_event_listener)

            orchestrator = await stack.enter_async_context(orchestrator)

            logger.debug(
                f"Agent loaded from {specs_path}, models "
                f"used: {orchestrator.model_ids}, with recent message memory: "
                f"{'yes' if orchestrator.memory.has_recent_message else 'no'},"
                " with permanent message memory: yes, with session #"
                + str(orchestrator.memory.permanent_message.session_id)
                if orchestrator.memory.has_permanent_message
                else "no"
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

        input_string: Optional[str] = None
        in_conversation = False
        while not input_string or in_conversation:
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

            logger.debug(f'Agent about to process input "{input_string}"')
            output = await orchestrator(
                input_string, use_async_generator=use_async_generator
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
                tool_events_limit=args.tool_events,
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
    assert args.host and args.port and args.specifications_file
    specs_path = args.specifications_file

    async with AsyncExitStack() as stack:
        logger.debug(f"Loading agent from {specs_path}")

        orchestrator = await OrchestrationLoader.from_file(
            specs_path,
            agent_id=uuid4(),
            hub=hub,
            logger=logger,
            participant_id=uuid4(),
            stack=stack,
        )
        orchestrator = await stack.enter_async_context(orchestrator)

        logger.debug(f"Agent loaded from {specs_path}")
        server = agents_server(
            name=name,
            version=version,
            prefix_openai=args.prefix_openai,
            prefix_mcp=args.prefix_mcp,
            agents=[orchestrator],
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
    memory_permanent = (
        args.memory_permanent
        if args.memory_permanent is not None
        else Prompt.ask(_("Permanent memory DSN"), default="")
    )
    memory_engine_model_id = (
        args.memory_engine_model_id
        or OrchestrationLoader.DEFAULT_SENTENCE_MODEL_ID
    )
    engine_uri = args.engine_uri or Prompt.ask(
        _("Engine URI"),
        default="microsoft/Phi-4-mini-instruct",
    )
    run_use_cache = (
        args.use_cache
        if args.use_cache is not None
        else Confirm.ask(_("Cache model locally?"))
    )
    run_skip_special_tokens = args.skip_special_tokens
    max_new_tokens = args.max_new_tokens or 1024

    data = dict(
        name=name,
        role=role,
        task=task,
        instructions=instructions,
        memory_recent=memory_recent,
        memory_permanent=memory_permanent,
        memory_engine_model_id=memory_engine_model_id,
        engine_uri=engine_uri,
        run_use_cache=run_use_cache,
        run_skip_special_tokens=run_skip_special_tokens,
        max_new_tokens=max_new_tokens,
    )

    env = Environment(
        loader=FileSystemLoader(join(dirname(__file__), "..", "..", "agent")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("blueprint.toml")
    rendered = template.render(**data)
    console.print(Syntax(rendered, "toml"))
