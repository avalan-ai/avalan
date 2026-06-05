from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.task import (
    CallableTaskTargetRunner,
    TaskDefinition,
    TaskExecutionContext,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskTargetContext,
    TaskTargetRunner,
    TaskTargetRunnerRegistry,
    TaskTargetType,
    TaskValidationContext,
    TaskValidationIssue,
)


class RecordingRunner(TaskTargetRunner):
    def __init__(self, name: str) -> None:
        self.name = name
        self.validated: list[str] = []
        self.ran: list[str] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = context
        self.validated.append(definition.execution.type.value)
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.ran.append(context.definition.execution.type.value)
        return self.name


class TaskTargetRunnerRegistryTestCase(IsolatedAsyncioTestCase):
    async def test_registry_uses_mapped_runner_only_for_target_type(
        self,
    ) -> None:
        default = RecordingRunner("default")
        flow = RecordingRunner("flow")
        registry = TaskTargetRunnerRegistry(
            default,
            {TaskTargetType.FLOW: flow},
        )
        agent_definition = _definition(TaskExecutionTarget.agent("agent.toml"))
        flow_definition = _definition(TaskExecutionTarget.flow("flow.toml"))

        self.assertIs(registry.runner_for(TaskTargetType.AGENT), default)
        self.assertIs(registry.runner_for(TaskTargetType.FLOW), flow)
        self.assertEqual(
            await registry.validate_definition(
                agent_definition,
                TaskValidationContext(),
            ),
            (),
        )
        self.assertEqual(
            await registry.validate_definition(
                flow_definition,
                TaskValidationContext(),
            ),
            (),
        )
        self.assertEqual(
            await registry.run(_context(agent_definition)),
            "default",
        )
        self.assertEqual(await registry.run(_context(flow_definition)), "flow")
        self.assertEqual(default.validated, ["agent"])
        self.assertEqual(flow.validated, ["flow"])
        self.assertEqual(default.ran, ["agent"])
        self.assertEqual(flow.ran, ["flow"])


class CallableTaskTargetRunnerTestCase(IsolatedAsyncioTestCase):
    async def test_callable_runner_validates_and_runs_target(self) -> None:
        definition = _definition(TaskExecutionTarget.agent("agent.toml"))
        context = _context(definition)

        async def target(received: TaskTargetContext) -> object:
            self.assertIs(received, context)
            return "ok"

        runner = CallableTaskTargetRunner(target)

        self.assertEqual(
            await runner.validate_definition(
                definition,
                TaskValidationContext(),
            ),
            (),
        )
        self.assertEqual(await runner.run(context), "ok")


class TaskValidationContextTestCase(TestCase):
    def test_validation_context_rejects_invalid_roots(self) -> None:
        self.assertEqual(
            TaskValidationContext(
                execution_roots=(Path.cwd(),)
            ).execution_roots,
            (Path.cwd(),),
        )
        with self.assertRaises(AssertionError):
            TaskValidationContext(
                execution_roots=(object(),),  # type: ignore[arg-type]
            )


def _definition(execution: TaskExecutionTarget) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="target_registry", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=execution,
    )


def _context(definition: TaskDefinition) -> TaskTargetContext:
    return TaskTargetContext(
        definition=definition,
        execution=TaskExecutionContext(
            run_id="run-1",
            attempt_id="attempt-1",
            attempt_number=1,
        ),
        input_value="ready",
        files=(),
        metadata={},
    )


if __name__ == "__main__":
    main()
