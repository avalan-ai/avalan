"""Exercise complete failure inventory and presentation coherence."""

from argparse import Namespace
from asyncio import CancelledError, Event, Task, create_task, run
from dataclasses import replace
from pathlib import Path
from sys import path as sys_path
from typing import Any

from avalan import sdk
from avalan.agent.execution import (
    AttachedInteractionRuntime,
    create_agent_execution,
    ensure_interaction_runtime_branch,
)
from avalan.cli.display import cli_stream_display_config
from avalan.cli.display_reducer import CliStreamSnapshotReducer
from avalan.cli.interaction_renderer import CliInteractionRenderer
from avalan.interaction import (
    AgentId,
    InteractionCorrelation,
    RequestState,
    ResolutionStatus,
    RunId,
    TaskId,
)
from avalan.model.stream import (
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
)

_ROOT = Path(__file__).parents[2]
sys_path.append(str(_ROOT / "scripts"))
sys_path.append(str(_ROOT / "tests/cli"))
sys_path.append(str(_ROOT / "tests/input"))

import broker_contract_test as broker_support  # noqa: E402
import interaction_renderer_test as renderer_support  # noqa: E402
import orchestration_contract_test as nested_support  # noqa: E402
from verify_input_acceptance import load_failure_matrix  # noqa: E402


def test_requirement_input_n_106() -> None:
    """Bind all 15 conditions to surface-owned dynamic evidence nodes."""
    matrix = load_failure_matrix(
        _ROOT / "tests/fixtures/input/failure_matrix.json"
    )
    condition_ids = {condition.id for condition in matrix.conditions}
    assert condition_ids == {
        f"INPUT-F-{number:02d}" for number in range(1, 16)
    }
    assert {rule.condition_id for rule in matrix.rules} == condition_ids
    assert all(rule.surface_ids for rule in matrix.rules)


def test_requirement_input_n_107() -> None:
    """Keep prior streams and persisted input coherent on cancellation."""

    async def exercise() -> None:
        harness = await broker_support._harness()
        channel = renderer_support._BlockingChannel()
        reducer = CliStreamSnapshotReducer(
            cli_stream_display_config(
                Namespace(quiet=True),
                refresh_per_second=1,
                interactive=False,
            )
        )
        reducer.reduce_projection(
            StreamConsumerProjection(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
                text_delta="partial-answer",
            )
        )

        class Handler:
            execution: Any
            before: tuple[Any, Any, Any]
            correlation: InteractionCorrelation
            renderer_task: Task[object]
            propagated = False
            started = Event()

            async def __call__(self, context: object) -> object:
                assert isinstance(context, broker_support.InputHandlerContext)
                self.correlation = InteractionCorrelation.from_request(
                    context.request
                )
                persisted = await broker_support._inspect(
                    harness.broker, self.correlation
                )
                self.before = (
                    tuple(self.execution.messages),
                    reducer.snapshot(),
                    persisted,
                )
                self.renderer_task = create_task(
                    CliInteractionRenderer(channel).render(
                        sdk._public_handler_context(context)
                    )
                )
                self.started.set()
                try:
                    return await self.renderer_task
                except CancelledError:
                    self.propagated = True
                    raise

        handler = Handler()
        runtime = await ensure_interaction_runtime_branch(
            AttachedInteractionRuntime(
                broker=harness.broker,
                actor=broker_support._actor(),
                handler=handler,
                run_id=RunId("run"),
                task_id=TaskId("task"),
            )
        )
        execution = await create_agent_execution(
            definition=nested_support._definition(),
            agent_id=AgentId("agent"),
            principal=runtime.actor.principal,
            initial_messages=(nested_support._assistant_message(),),
            interaction_runtime=runtime,
        )
        handler.execution = execution
        request = replace(
            broker_support._request(
                handler, run_id="run", reason="Need input."
            ),
            origin=execution.origin,
        )
        try:
            requested = create_task(harness.broker.request(request))
            await handler.started.wait()
            await channel.reading.wait()
            handler.renderer_task.cancel()
            result = await requested
            assert result.delivery is not None
            after = await broker_support._inspect(
                harness.broker, handler.correlation
            )
            transcript, answer, persisted = handler.before
            assert tuple(execution.messages) == transcript
            answer_after = reducer.snapshot()
            assert (
                replace(answer_after, build_stats=answer.build_stats) == answer
            )
            assert answer_after.answer_text == "partial-answer"
            assert persisted.request.origin == after.request.origin
            assert persisted.request.questions == after.request.questions
            assert after.request.state is RequestState.UNAVAILABLE
            assert after.request.resolution is not None
            assert (
                after.request.resolution.status is ResolutionStatus.UNAVAILABLE
            )
            assert handler.propagated
            assert "Input required" in channel.output
        finally:
            await harness.broker.aclose()

    run(exercise())
