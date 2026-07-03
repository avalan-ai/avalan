from collections.abc import Awaitable, Callable
from json import dumps
from pathlib import Path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.tool.shell import (
    ExecutionPolicy,
    ShellCommandDefinition,
    ShellCompositionResult,
    ShellCompositionSpec,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellFormattedCompositionResult,
    ShellStreamRef,
    ShellToolSettings,
)
from avalan.tool.shell.tools import PipelineTool


class PipelineToolTest(IsolatedAsyncioTestCase):
    async def test_pipeline_tool_formats_result_and_forwards_streams(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            allow_pipelines=True,
            workspace_root=str(fixture_root),
        )
        executor = _StreamingCompositionExecutor()
        tool = PipelineTool(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            executor=executor,
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        output = await tool(
            steps=[
                {
                    "id": "read",
                    "command": "cat",
                    "paths": ["filesystem/visible.txt"],
                },
                {
                    "id": "count",
                    "command": "wc",
                    "options": {"lines": True},
                    "stdin_from": {"step_id": "read", "stream": "stdout"},
                },
            ],
            context=ToolCallContext(stream_event=record),
        )

        self.assertIsInstance(output, ShellFormattedCompositionResult)
        self.assertEqual(executor.calls, 1)
        self.assertIn("tool: shell.pipeline", output)
        self.assertIn("stage_chain: cat | wc", output)
        self.assertIn("stdout:\n2\n", output)
        self.assertNotIn("INTERMEDIATE_STDOUT_SHOULD_NOT_LEAK", output)
        self.assertEqual(
            [(event.kind, event.content, event.metadata) for event in events],
            [
                (
                    ToolExecutionStreamKind.STDERR,
                    "warning\n",
                    {"stage_id": "read", "stage_index": 0},
                ),
                (
                    ToolExecutionStreamKind.STDOUT,
                    "2\n",
                    {"stage_id": "count", "stage_index": 1},
                ),
            ],
        )

    async def test_disabled_pipeline_policy_denial_skips_executor(
        self,
    ) -> None:
        settings = ShellToolSettings(allow_pipelines=False)
        executor = _StreamingCompositionExecutor()
        tool = PipelineTool(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            executor=executor,
        )

        output = await tool(
            steps=[{"id": "list", "command": "ls"}],
            context=ToolCallContext(),
        )

        self.assertIsInstance(output, ShellFormattedCompositionResult)
        self.assertEqual(executor.calls, 0)
        self.assertIn("status: policy_denied", output)
        self.assertIn("error_message: shell pipelines are disabled", output)

    async def test_pipeline_tool_executes_without_stream_callback(
        self,
    ) -> None:
        fixture_root = Path(__file__).parent / "fixtures"
        settings = ShellToolSettings(
            allow_pipelines=True,
            workspace_root=str(fixture_root),
        )
        executor = _StreamingCompositionExecutor()
        tool = PipelineTool(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            executor=executor,
        )

        output = await tool(
            steps=[
                {
                    "id": "read",
                    "command": "cat",
                    "paths": ["filesystem/visible.txt"],
                },
                {
                    "id": "count",
                    "command": "wc",
                    "stdin_from": {"step_id": "read", "stream": "stdout"},
                },
            ],
            context=ToolCallContext(),
        )

        self.assertIsInstance(output, ShellFormattedCompositionResult)
        self.assertEqual(executor.calls, 1)
        self.assertIn("stage_chain: cat | wc", output)

    async def test_unknown_command_policy_denial_redacts_result_surface(
        self,
    ) -> None:
        settings = ShellToolSettings(allow_pipelines=True)
        executor = _StreamingCompositionExecutor()
        tool = PipelineTool(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            executor=executor,
        )
        raw_command = "unknown PRIVATE_RAW_PAYLOAD_DO_NOT_LEAK"
        raw_step_id = "PRIVATE_STEP_ID_DO_NOT_LEAK"
        arguments = {
            "steps": [
                {
                    "id": raw_step_id,
                    "command": raw_command,
                }
            ]
        }

        output = await tool(
            steps=[
                {
                    "id": raw_step_id,
                    "command": raw_command,
                }
            ],
            context=ToolCallContext(),
        )

        self.assertIsInstance(output, ShellFormattedCompositionResult)
        self.assertEqual(executor.calls, 0)
        self.assertIn("status: policy_denied", output)
        self.assertIn("stage_chain: [redacted]", output)
        self.assertIn("  id: [redacted]", output)
        self.assertIn("  command: [redacted]", output)
        self.assertNotIn(raw_command, output)
        self.assertNotIn(raw_step_id, output)

        call = ToolCall(
            id="call-pipeline",
            name="shell.pipeline",
            arguments=arguments,
        )
        outcome = ToolCallResult(
            id="result-pipeline",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=output,
        )
        projection = tool.tool_display_projector(call, outcome)

        assert projection is not None
        payload = dumps(projection.to_payload(), sort_keys=True)
        self.assertIn("[redacted]", payload)
        self.assertNotIn(raw_command, payload)
        self.assertNotIn(raw_step_id, payload)

    async def test_pipeline_denial_redacts_ids_and_only_unknown_commands(
        self,
    ) -> None:
        settings = ShellToolSettings(allow_pipelines=False)
        executor = _StreamingCompositionExecutor()
        tool = PipelineTool(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=_AllResolved()),
            executor=executor,
        )
        raw_command = "unknown PRIVATE_COMMAND_DO_NOT_LEAK"

        safe_call = ToolCall(
            id="safe",
            name="shell.pipeline",
            arguments={"steps": [{"id": "read", "command": "cat"}]},
        )
        unsafe_call = ToolCall(
            id="unsafe",
            name="shell.pipeline",
            arguments={"steps": [{"id": "secret", "command": raw_command}]},
        )
        output = await tool(
            steps=[
                {"id": "read", "command": "cat"},
                {"id": "secret", "command": raw_command},
            ],
            context=ToolCallContext(),
        )

        self.assertIsNotNone(tool.tool_display_projector(safe_call))
        self.assertIsNone(tool.tool_display_projector(unsafe_call))
        self.assertIsInstance(output, ShellFormattedCompositionResult)
        formatted_output = cast(ShellFormattedCompositionResult, output)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(
            tuple(
                step.id for step in formatted_output.composition_result.steps
            ),
            ("[redacted]-0", "[redacted]-1"),
        )
        self.assertEqual(
            tuple(
                step.command
                for step in formatted_output.composition_result.steps
            ),
            ("cat", "[redacted]"),
        )
        self.assertEqual(
            tuple(
                step.metadata["stdout_visible"]
                for step in formatted_output.composition_result.steps
            ),
            (False, True),
        )
        self.assertNotIn("secret", output)
        self.assertNotIn(raw_command, output)

    def test_build_request_accepts_none_paths_and_stream_ref_stdin(
        self,
    ) -> None:
        tool = PipelineTool(
            settings=ShellToolSettings(allow_pipelines=True),
            policy=ExecutionPolicy(resolver=_AllResolved()),
            executor=_StreamingCompositionExecutor(),
        )
        stdin_from = ShellStreamRef(step_id="read", stream="stdout")

        request = tool._build_request(
            steps=cast(
                Any,
                [
                    {
                        "id": "read",
                        "command": "cat",
                        "paths": None,
                    },
                    {
                        "id": "count",
                        "command": "wc",
                        "stdin_from": stdin_from,
                    },
                ],
            )
        )

        self.assertEqual(request.steps[0].paths, ())
        self.assertIs(request.steps[1].stdin_from, stdin_from)


class _AllResolved:
    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        return f"/trusted/bin/{command.executable_name}"


class _StreamingCompositionExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute_composition(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ShellCompositionResult:
        self.calls += 1
        if stream is not None:
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDERR,
                    content="warning\n",
                    metadata={"stage_id": "read", "stage_index": 0},
                )
            )
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content="2\n",
                    metadata={"stage_id": "count", "stage_index": 1},
                )
            )
        return ShellCompositionResult(
            mode=spec.mode,
            status=ShellExecutionStatus.COMPLETED,
            stdout="2\n",
            stderr="[read:cat]\nwarning\n",
            steps=(
                ShellExecutionStepResult(
                    id="read",
                    command="cat",
                    status=ShellExecutionStatus.COMPLETED,
                    exit_code=0,
                    stdout="INTERMEDIATE_STDOUT_SHOULD_NOT_LEAK",
                    stderr="warning\n",
                    stdout_bytes=35,
                    stderr_bytes=8,
                    stdout_truncated=False,
                    stderr_truncated=False,
                    duration_ms=1,
                ),
                ShellExecutionStepResult(
                    id="count",
                    command="wc",
                    status=ShellExecutionStatus.COMPLETED,
                    exit_code=0,
                    stdout="2\n",
                    stderr="",
                    stdout_bytes=2,
                    stderr_bytes=0,
                    stdout_truncated=False,
                    stderr_truncated=False,
                    duration_ms=1,
                ),
            ),
            stdout_bytes=2,
            stderr_bytes=19,
            duration_ms=2,
        )


if __name__ == "__main__":
    main()
