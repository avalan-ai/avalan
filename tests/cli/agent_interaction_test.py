from argparse import Namespace
from asyncio import Event
from asyncio.exceptions import CancelledError
from contextlib import AsyncExitStack, asynccontextmanager
from logging import Logger
from typing import Any, AsyncIterator, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent.execution import InteractionRuntime
from avalan.cli.commands import agent as agent_cmds
from avalan.cli.interaction_renderer import (
    CliInteractionCommandDisposition,
    CliRunCancellationCommand,
    CliSteeringCommand,
)
from avalan.cli.stream_coordinator import CliStreamCoordinator
from avalan.sdk import AttachedInputContext, AttachedInputOutcome


class _AsyncResource:
    """Record ownership and closure through an asynchronous exit stack."""

    def __init__(self, name: str, exits: list[str]) -> None:
        self.name = name
        self.exits = exits

    async def __aenter__(self) -> "_AsyncResource":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.exits.append(self.name)


class _Renderer:
    """Return an attached outcome without touching process streams."""

    def __init__(self, outcome: AttachedInputOutcome) -> None:
        self.outcome = outcome
        self.contexts: list[AttachedInputContext] = []

    async def __call__(
        self,
        context: AttachedInputContext,
    ) -> AttachedInputOutcome:
        self.contexts.append(context)
        return self.outcome


class _Coordinator:
    """Record the exact prompt pause boundary used by the attached handler."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @asynccontextmanager
    async def paused(self) -> AsyncIterator[None]:
        self.events.append("pause")
        try:
            yield
        finally:
            self.events.append("resume")


class CliInteractionRuntimeTestCase(IsolatedAsyncioTestCase):
    async def test_missing_control_terminal_disables_attached_runtime(
        self,
    ) -> None:
        create_runtime = AsyncMock()
        unwrap_runtime = MagicMock()

        async with AsyncExitStack() as stack:
            with (
                patch(
                    "avalan.cli.commands.agent.CliInteractionChannel.open",
                    return_value=None,
                ) as open_channel,
                patch.object(
                    agent_cmds,
                    "create_attached_input_runtime",
                    create_runtime,
                ),
                patch.object(
                    agent_cmds,
                    "_unwrap_interaction_runtime",
                    unwrap_runtime,
                ),
            ):
                runtime = await agent_cmds._cli_interaction_runtime(
                    stack,
                    "/missing/tty",
                    {"coordinator": None},
                    Event(),
                    participant_id="participant",
                    session_id="session",
                )

        self.assertIsNone(runtime)
        open_channel.assert_called_once_with("/missing/tty")
        create_runtime.assert_not_awaited()
        unwrap_runtime.assert_not_called()

    async def test_runtime_owns_channel_and_pauses_only_active_display(
        self,
    ) -> None:
        for active in (False, True):
            with self.subTest(active=active):
                exits: list[str] = []
                channel = _AsyncResource("channel", exits)
                public_runtime = _AsyncResource("runtime", exits)
                internal_runtime = cast(InteractionRuntime, object())
                outcome = cast(AttachedInputOutcome, object())
                renderer = _Renderer(outcome)
                coordinator = _Coordinator()
                container: dict[str, CliStreamCoordinator | None] = {
                    "coordinator": (
                        cast(CliStreamCoordinator, coordinator)
                        if active
                        else None
                    )
                }
                captured_handler: list[Any] = []
                captured_principal: list[Any] = []
                run_cancellation = Event()

                async def create_runtime(
                    handler: Any,
                    *,
                    principal: Any,
                ) -> Any:
                    captured_handler.append(handler)
                    captured_principal.append(principal)
                    return public_runtime

                renderer_factory = MagicMock(return_value=renderer)
                async with AsyncExitStack() as stack:
                    with (
                        patch(
                            "avalan.cli.commands.agent.CliInteractionChannel.open",
                            return_value=channel,
                        ),
                        patch.object(
                            agent_cmds,
                            "CliInteractionRenderer",
                            renderer_factory,
                        ),
                        patch.object(
                            agent_cmds,
                            "create_attached_input_runtime",
                            new=AsyncMock(side_effect=create_runtime),
                        ),
                        patch.object(
                            agent_cmds,
                            "_unwrap_interaction_runtime",
                            return_value=internal_runtime,
                        ) as unwrap_runtime,
                    ):
                        result = await agent_cmds._cli_interaction_runtime(
                            stack,
                            "/tmp/control-tty",
                            container,
                            run_cancellation,
                            participant_id="participant",
                            session_id="session",
                        )
                        command_handler = renderer_factory.call_args.kwargs[
                            "command_handler"
                        ]
                        disposition = await command_handler(
                            CliRunCancellationCommand()
                        )
                        self.assertIs(
                            disposition,
                            CliInteractionCommandDisposition.ACCEPTED,
                        )
                        self.assertFalse(run_cancellation.is_set())

                        context = cast(AttachedInputContext, object())
                        rendered = await captured_handler[0](context)

                    self.assertIs(result, internal_runtime)
                    self.assertIs(rendered, outcome)
                    self.assertTrue(run_cancellation.is_set())
                    self.assertEqual(renderer.contexts, [context])
                    self.assertEqual(
                        coordinator.events,
                        ["pause", "resume"] if active else [],
                    )
                    unwrap_runtime.assert_called_once_with(public_runtime)

                    renderer_factory.assert_called_once()
                    disposition = await command_handler(
                        CliSteeringCommand(text="focus elsewhere")
                    )
                    self.assertIs(
                        disposition,
                        CliInteractionCommandDisposition.UNAVAILABLE,
                    )
                    principal = captured_principal[0]
                    self.assertEqual(principal.participant_id, "participant")
                    self.assertEqual(principal.session_id, "session")

                self.assertEqual(exits, ["runtime", "channel"])


class AgentRunInteractionInjectionTestCase(IsolatedAsyncioTestCase):
    async def test_agent_run_passes_opened_runtime_per_call_and_reads_once(
        self,
    ) -> None:
        args = Namespace(
            specifications_file="agent.toml",
            engine_uri=None,
            use_sync_generator=False,
            id="agent",
            participant="participant",
            session=None,
            no_session=True,
            skip_load_recent_messages=True,
            load_recent_messages_limit=None,
            tools_confirm=False,
            conversation=False,
            watch=False,
            no_repl=False,
            quiet=True,
            tty="/tmp/control-tty",
            input_file=None,
        )
        console = MagicMock()
        console.is_terminal = False
        theme = MagicMock()
        theme._ = lambda value: value
        theme.icons = {"user_input": ">", "agent_output": "<"}
        logger = MagicMock(spec=Logger)
        hub = MagicMock()
        runtime = cast(InteractionRuntime, object())

        orchestrator = AsyncMock()
        orchestrator._call_options = None
        orchestrator.engine = MagicMock()
        orchestrator.event_manager.add_ui_listener = MagicMock()
        orchestrator.event_manager.remove_listener = MagicMock()
        orchestrator.memory.has_recent_message = False
        orchestrator.memory.has_permanent_message = False
        orchestrator.memory.recent_message = MagicMock(size=0)
        orchestrator.tool.is_empty = True

        class Response:
            def __init__(self) -> None:
                self.cancellation_checker: Any = None

            def set_cancellation_checker(self, checker: Any) -> None:
                self.cancellation_checker = checker

        response = Response()
        orchestrator.return_value = response
        stack = AsyncMock()
        stack.__aenter__.return_value = stack
        stack.__aexit__.return_value = False
        stack.enter_async_context = AsyncMock(return_value=orchestrator)
        stack.callback = MagicMock()
        input_reader = MagicMock(return_value="initial prompt")
        interaction_factory = AsyncMock(return_value=runtime)

        with (
            patch.object(agent_cmds, "AsyncExitStack", return_value=stack),
            patch(
                "avalan.cli.commands.agent.OrchestratorLoader.from_file",
                new=AsyncMock(return_value=orchestrator),
            ),
            patch.object(
                agent_cmds,
                "_agent_tool_settings",
                return_value=MagicMock(),
            ),
            patch.object(
                agent_cmds,
                "_agent_tool_name_policy_kwargs",
                return_value={},
            ),
            patch.object(
                agent_cmds,
                "_agent_reasoning_overrides",
                return_value={},
            ),
            patch.object(agent_cmds, "get_input", input_reader),
            patch.object(
                agent_cmds,
                "_agent_run_input",
                new=AsyncMock(return_value="built input"),
            ),
            patch.object(
                agent_cmds,
                "_cli_interaction_runtime",
                interaction_factory,
            ),
            patch.object(agent_cmds, "OrchestratorResponse", Response),
            patch.object(
                agent_cmds,
                "token_generation",
                new_callable=AsyncMock,
            ) as token_generation,
        ):
            await agent_cmds.agent_run(
                args,
                console,
                theme,
                hub,
                logger,
                refresh_per_second=4,
            )

        interaction_factory.assert_awaited_once()
        interaction_call = interaction_factory.await_args
        assert interaction_call is not None
        interaction_kwargs = interaction_call.kwargs
        self.assertEqual(interaction_call.args[1], "/tmp/control-tty")
        run_cancellation = interaction_call.args[3]
        self.assertIsInstance(run_cancellation, Event)
        self.assertEqual(interaction_kwargs["participant_id"], "participant")
        self.assertIsNone(interaction_kwargs["session_id"])
        input_reader.assert_called_once()
        orchestrator.assert_awaited_once_with(
            "built input",
            use_async_generator=True,
            tool_confirm=None,
            interaction_runtime=runtime,
        )
        token_generation.assert_awaited_once()
        self.assertIsNotNone(response.cancellation_checker)
        await response.cancellation_checker()
        run_cancellation.set()
        with self.assertRaises(CancelledError):
            await response.cancellation_checker()
