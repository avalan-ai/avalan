from copy import copy, deepcopy
from dataclasses import replace as dataclass_replace
from pathlib import Path
from pickle import dumps
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch

from avalan.task import (
    CallableTaskTargetRunner,
    TaskDefinition,
    TaskExecutionContext,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskTargetContext,
    TaskTargetOutcome,
    TaskTargetRunner,
    TaskTargetRunnerRegistry,
    TaskTargetType,
    TaskValidationContext,
    TaskValidationIssue,
    completed_task_target_outcome,
)
from avalan.task.context import TaskDurableResumeHandle
from avalan.task.target import (
    PreparedTaskDurableResumeTarget,
    TaskDurableResumeTargetRunner,
    task_target_outcome,
)


class RecordingRunner(TaskTargetRunner):
    def __init__(
        self,
        name: str,
        *,
        durable_suspension: bool = False,
    ) -> None:
        self.name = name
        self.durable_suspension = durable_suspension
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

    async def run(
        self,
        context: TaskTargetContext,
    ) -> TaskTargetOutcome:
        self.ran.append(context.definition.execution.type.value)
        return completed_task_target_outcome(self.name)

    def supports_durable_suspension(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        return self.durable_suspension and target_type is TaskTargetType.AGENT


class RecordingDurableResumeRunner(RecordingRunner):
    def __init__(
        self,
        name: str,
        *,
        advertisement: object,
        supported_target: TaskTargetType = TaskTargetType.AGENT,
    ) -> None:
        super().__init__(name)
        self.advertisement = advertisement
        self.supported_target = supported_target
        self.capability_calls = 0
        self.resume_contexts: list[TaskTargetContext] = []
        self.resume_handles: list[TaskDurableResumeHandle] = []

    def supports_durable_resume(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        self.capability_calls += 1
        if target_type is not self.supported_target:
            return False
        return cast(bool, self.advertisement)

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        self.resume_contexts.append(context)
        self.resume_handles.append(durable_resume)
        return completed_task_target_outcome(self.name)


class StatefulDurableResumeRunner(RecordingDurableResumeRunner):
    def supports_durable_resume(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        self.capability_calls += 1
        return (
            target_type is self.supported_target and self.capability_calls == 1
        )


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
            completed_task_target_outcome("default"),
        )
        self.assertEqual(
            await registry.run(_context(flow_definition)),
            completed_task_target_outcome("flow"),
        )
        self.assertEqual(default.validated, ["agent"])
        self.assertEqual(flow.validated, ["flow"])
        self.assertEqual(default.ran, ["agent"])
        self.assertEqual(flow.ran, ["flow"])

    def test_registry_copies_runner_mapping(self) -> None:
        default = RecordingRunner("default")
        flow = RecordingRunner("flow")
        runners = {TaskTargetType.FLOW: flow}
        registry = TaskTargetRunnerRegistry(default, runners)

        runners.clear()
        runners[TaskTargetType.AGENT] = flow

        self.assertIs(registry.runner_for(TaskTargetType.AGENT), default)
        self.assertIs(registry.runner_for(TaskTargetType.FLOW), flow)

    def test_registry_routes_durable_suspension_capability_exactly(
        self,
    ) -> None:
        default = RecordingRunner("default", durable_suspension=True)
        flow = RecordingRunner("flow")
        registry = TaskTargetRunnerRegistry(
            default,
            {TaskTargetType.FLOW: flow},
        )

        self.assertTrue(
            registry.supports_durable_suspension(TaskTargetType.AGENT)
        )
        self.assertFalse(
            registry.supports_durable_suspension(TaskTargetType.FLOW)
        )

    def test_registry_rejects_invalid_target_outcome(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "completed or suspended outcome",
        ):
            task_target_outcome(object())

    def test_registry_rejects_non_durable_resume_runner(self) -> None:
        registry = TaskTargetRunnerRegistry(RecordingRunner("default"))

        self.assertIsNone(
            registry.prepare_durable_resume(TaskTargetType.AGENT)
        )

    async def test_registry_resume_requires_exact_positive_capability(
        self,
    ) -> None:
        definition = _definition(TaskExecutionTarget.agent("agent.toml"))
        context = _context(definition)
        handle = cast(TaskDurableResumeHandle, object())

        for advertisement in (False, 1, "true"):
            with self.subTest(advertisement=advertisement):
                runner = RecordingDurableResumeRunner(
                    "unreachable",
                    advertisement=advertisement,
                )
                registry = TaskTargetRunnerRegistry(runner)

                self.assertIsInstance(
                    runner,
                    TaskDurableResumeTargetRunner,
                )
                self.assertFalse(
                    registry.supports_durable_resume(TaskTargetType.AGENT)
                )
                with self.assertRaisesRegex(
                    TypeError,
                    "does not support durable resume",
                ):
                    await registry.resume(context, handle)
                self.assertEqual(runner.resume_contexts, [])
                self.assertEqual(runner.resume_handles, [])

    async def test_registry_resume_rejects_wrong_selected_target(
        self,
    ) -> None:
        agent = RecordingDurableResumeRunner(
            "agent",
            advertisement=True,
        )
        flow = RecordingDurableResumeRunner(
            "unreachable",
            advertisement=True,
            supported_target=TaskTargetType.AGENT,
        )
        registry = TaskTargetRunnerRegistry(
            agent,
            {TaskTargetType.FLOW: flow},
        )
        context = _context(_definition(TaskExecutionTarget.flow("flow.toml")))
        handle = cast(TaskDurableResumeHandle, object())

        self.assertFalse(registry.supports_durable_resume(TaskTargetType.FLOW))
        with self.assertRaisesRegex(
            TypeError,
            "does not support durable resume",
        ):
            await registry.resume(context, handle)
        self.assertEqual(agent.resume_contexts, [])
        self.assertEqual(flow.resume_contexts, [])

    async def test_registry_resume_routes_exact_positive_capability(
        self,
    ) -> None:
        runner = RecordingDurableResumeRunner(
            "resumed",
            advertisement=True,
        )
        registry = TaskTargetRunnerRegistry(runner)
        context = _context(
            _definition(TaskExecutionTarget.agent("agent.toml"))
        )
        handle = cast(TaskDurableResumeHandle, object())

        result = await registry.resume(context, handle)

        self.assertEqual(
            result,
            completed_task_target_outcome("resumed"),
        )
        self.assertEqual(runner.resume_contexts, [context])
        self.assertEqual(runner.resume_handles, [handle])
        self.assertEqual(runner.capability_calls, 1)

    async def test_registry_public_resume_rejects_untrusted_preparation(
        self,
    ) -> None:
        runner = RecordingDurableResumeRunner(
            "unreachable",
            advertisement=True,
        )
        registry = TaskTargetRunnerRegistry(runner)
        context = _context(
            _definition(TaskExecutionTarget.agent("agent.toml"))
        )
        handle = cast(TaskDurableResumeHandle, object())
        untrusted = cast(PreparedTaskDurableResumeTarget, object())

        with patch.object(
            registry,
            "prepare_durable_resume",
            return_value=untrusted,
        ) as prepare:
            self.assertFalse(
                registry.supports_durable_resume(TaskTargetType.AGENT)
            )
            with self.assertRaisesRegex(
                TypeError,
                "does not support durable resume",
            ):
                await registry.resume(context, handle)

        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(runner.capability_calls, 0)
        self.assertEqual(runner.resume_contexts, [])
        self.assertEqual(runner.resume_handles, [])

    async def test_registry_prepares_sealed_target_bound_runner_once(
        self,
    ) -> None:
        runner = StatefulDurableResumeRunner(
            "resumed",
            advertisement=True,
        )
        registry = TaskTargetRunnerRegistry(runner)
        other_registry = TaskTargetRunnerRegistry(runner)
        context = _context(
            _definition(TaskExecutionTarget.agent("agent.toml"))
        )
        handle = cast(TaskDurableResumeHandle, object())

        prepared = registry.prepare_durable_resume(TaskTargetType.AGENT)

        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertIs(type(prepared), PreparedTaskDurableResumeTarget)
        self.assertIs(prepared.target_type, TaskTargetType.AGENT)
        self.assertIs(prepared.runner, runner)
        self.assertTrue(prepared.is_bound_to(registry, TaskTargetType.AGENT))
        self.assertFalse(prepared.is_bound_to(object(), TaskTargetType.AGENT))
        self.assertFalse(
            prepared.is_bound_to(other_registry, TaskTargetType.AGENT)
        )
        self.assertFalse(prepared.is_bound_to(registry, TaskTargetType.FLOW))
        with self.assertRaises(AttributeError):
            object.__setattr__(prepared, "runner", other_registry)

        result = await prepared.runner.resume(context, handle)

        self.assertEqual(
            result,
            completed_task_target_outcome("resumed"),
        )
        self.assertEqual(runner.capability_calls, 1)
        self.assertEqual(runner.resume_contexts, [context])
        self.assertEqual(runner.resume_handles, [handle])

    async def test_prepared_target_rejects_identity_forgeries(self) -> None:
        runner = RecordingDurableResumeRunner(
            "resumed",
            advertisement=True,
        )
        replacement = RecordingDurableResumeRunner(
            "forged",
            advertisement=True,
        )
        registry = TaskTargetRunnerRegistry(runner)
        prepared = registry.prepare_durable_resume(TaskTargetType.AGENT)

        self.assertIsNotNone(prepared)
        assert prepared is not None
        forged_values = list(prepared)
        forged_values[1] = replacement
        forged = tuple.__new__(
            PreparedTaskDurableResumeTarget,
            tuple(forged_values),
        )
        copied_identity = tuple.__new__(
            PreparedTaskDurableResumeTarget,
            tuple(prepared),
        )
        subclass = type(
            "PreparedTaskDurableResumeTargetSubclass",
            (PreparedTaskDurableResumeTarget,),
            {},
        )
        subclass_forgery: PreparedTaskDurableResumeTarget = tuple.__new__(
            subclass,
            tuple(prepared),
        )
        blank = tuple.__new__(PreparedTaskDurableResumeTarget)
        context = _context(
            _definition(TaskExecutionTarget.agent("agent.toml"))
        )
        handle = cast(TaskDurableResumeHandle, object())

        for untrusted in (
            forged,
            copied_identity,
            subclass_forgery,
            blank,
        ):
            with self.subTest(untrusted=type(untrusted).__name__):
                self.assertFalse(
                    untrusted.is_bound_to(
                        registry,
                        TaskTargetType.AGENT,
                    )
                )
                with (
                    patch.object(
                        registry,
                        "prepare_durable_resume",
                        return_value=untrusted,
                    ),
                    self.assertRaisesRegex(
                        TypeError,
                        "does not support durable resume",
                    ),
                ):
                    await registry.resume(context, handle)

        self.assertEqual(runner.capability_calls, 1)
        self.assertEqual(runner.resume_contexts, [])
        self.assertEqual(replacement.resume_contexts, [])

    def test_prepared_target_rejects_copy_replace_and_pickle(self) -> None:
        runner = RecordingDurableResumeRunner(
            "resumed",
            advertisement=True,
        )
        prepared = TaskTargetRunnerRegistry(runner).prepare_durable_resume(
            TaskTargetType.AGENT
        )

        self.assertIsNotNone(prepared)
        assert prepared is not None
        for operation in (
            lambda: copy(prepared),
            lambda: deepcopy(prepared),
            lambda: dataclass_replace(cast(Any, prepared)),
            lambda: dumps(prepared),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(TypeError):
                    operation()

    def test_prepared_target_cannot_be_constructed_directly(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "come from a target registry",
        ):
            PreparedTaskDurableResumeTarget()
        with self.assertRaises(TypeError):
            object.__new__(PreparedTaskDurableResumeTarget)
        runner = RecordingDurableResumeRunner(
            "unreachable",
            advertisement=True,
        )
        with self.assertRaisesRegex(TypeError, "not registry-minted"):
            PreparedTaskDurableResumeTarget._mint(
                target_type=TaskTargetType.AGENT,
                runner=runner,
                preparer=object(),
                proof=object(),
            )


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
        self.assertEqual(
            await runner.run(context),
            completed_task_target_outcome("ok"),
        )


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
