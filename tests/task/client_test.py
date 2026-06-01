from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import Event, EventType
from avalan.task import (
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskClient,
    TaskClientUnsupportedOperationError,
    TaskClientValidationResult,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskInputContract,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskRun,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    UsageSource,
)
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import AgentTaskTargetRunner


class FakeResponse:
    input_token_count = 3
    output_token_count = 2

    def __init__(self, text: str) -> None:
        self.text = text

    async def to_str(self) -> str:
        return self.text


class FakeEventManager:
    def __init__(self) -> None:
        self.listeners: list[Callable[[Event], Awaitable[None] | None]] = []

    def add_listener(
        self,
        listener: Callable[[Event], Awaitable[None] | None],
    ) -> None:
        self.listeners.append(listener)

    def remove_listener(
        self,
        listener: Callable[[Event], Awaitable[None] | None],
    ) -> None:
        self.listeners.remove(listener)

    async def trigger(self, event: Event) -> None:
        for listener in tuple(self.listeners):
            result = listener(event)
            if result is not None:
                await result


class FakeOrchestrator:
    def __init__(self, loader: "FakeLoader") -> None:
        self._loader = loader
        self.event_manager = loader.event_manager

    async def __aenter__(self) -> "FakeOrchestrator":
        self._loader.entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None:
        self._loader.exited += 1
        return None

    async def __call__(self, input: object) -> FakeResponse:
        self._loader.inputs.append(input)
        await self.event_manager.trigger(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload={
                    "token": "secret-token",
                    "token_id": 9,
                    "status": "ok",
                },
            )
        )
        return FakeResponse("short summary")


class FakeLoader:
    def __init__(self) -> None:
        self.event_manager = FakeEventManager()
        self.inputs: list[object] = []
        self.entered = 0
        self.exited = 0

    async def from_file(
        self,
        path: str,
        *,
        agent_id: object | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> FakeOrchestrator:
        _ = path, agent_id, disable_memory, uri, tool_settings
        return FakeOrchestrator(self)


class StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"test-secret",
        )


class RejectingTarget(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return (
            TaskValidationIssue(
                code="execution.unknown_target",
                path="execution.ref",
                message="Task target could not be loaded.",
                hint="Use a supported execution target.",
                category=TaskValidationCategory.UNSUPPORTED,
            ),
        )

    async def run(self, context: TaskTargetContext) -> object:
        return "unused"


class TaskClientTest(IsolatedAsyncioTestCase):
    async def test_agent_backed_direct_run_is_inspectable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"
instructions = "Be brief."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            store = InMemoryTaskStore()
            loader = FakeLoader()
            client = TaskClient(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "client-direct-hash",
                execution_roots=(root,),
            )

            result = await client.run(
                _definition(),
                input_value="private prompt",
                metadata={"request": 1},
            )
            output = await client.output(result.run.run_id)
            events = await client.events(result.run.run_id)
            usage = await client.usage(result.run.run_id)
            inspection = await client.inspect(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "short summary")
        self.assertEqual(loader.inputs, ["private prompt"])
        self.assertEqual(loader.entered, 1)
        self.assertEqual(loader.exited, 1)
        self.assertTrue(output.ready)
        self.assertEqual(output.output_summary, {"privacy": "<redacted>"})
        self.assertIsNone(output.error)
        self.assertEqual(
            [event.event_type for event in events], ["token_generated"]
        )
        self.assertNotIn("secret-token", str(events[0].payload))
        self.assertNotIn("token_id", str(events[0].payload))
        self.assertEqual(usage[0].source, UsageSource.ESTIMATED)
        self.assertEqual(usage[0].totals.input_tokens, 3)
        self.assertEqual(usage[0].totals.output_tokens, 2)
        self.assertIsNone(usage[0].totals.total_tokens)
        self.assertEqual(inspection.run.run_id, result.run.run_id)
        self.assertEqual(inspection.output, output)
        self.assertEqual(inspection.events, events)
        self.assertEqual(inspection.usage, usage)
        self.assertIsNone(inspection.usage_totals.total_tokens)
        self.assertEqual(inspection.artifacts, ())
        self.assertEqual(loader.event_manager.listeners, [])

    async def test_direct_callable_target_uses_shared_validation(
        self,
    ) -> None:
        async def target(context: TaskTargetContext) -> object:
            _ = context
            return "callable summary"

        client = TaskClient(
            InMemoryTaskStore(),
            target=target,
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "client-callable-hash",
        )

        validation = await client.validate(
            _definition(),
            input_value="private prompt",
        )
        result = await client.run(_definition(), input_value="private prompt")

        self.assertTrue(validation.valid)
        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "callable summary")

    async def test_validate_aggregates_definition_input_and_target_issues(
        self,
    ) -> None:
        client = TaskClient(
            InMemoryTaskStore(),
            target=RejectingTarget(),
        )

        result = await client.validate(
            _definition(privacy=TaskPrivacyPolicy()),
            input_value={"raw": "not text"},
        )

        self.assertFalse(result.valid)
        self.assertEqual(
            [issue.code for issue in result.issues],
            [
                "privacy.hmac_key_missing",
                "input.invalid_type",
                "execution.unknown_target",
            ],
        )
        self.assertNotIn("not text", str(result.issues))
        with self.assertRaises(TaskValidationError):
            result.raise_for_issues()

    async def test_validation_result_accepts_empty_issue_set(self) -> None:
        result = TaskClientValidationResult()

        self.assertTrue(result.valid)
        result.raise_for_issues()

    async def test_queued_run_and_enqueue_return_stable_diagnostic(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        client = TaskClient(
            store,
            target=RejectingTarget(),
            hmac_provider=StaticHmacProvider(),
        )
        definition = _definition(run=TaskRunPolicy.queued("private-queue"))

        with self.assertRaises(TaskClientUnsupportedOperationError) as run:
            await client.run(definition, input_value="private prompt")
        with self.assertRaises(TaskClientUnsupportedOperationError) as enqueue:
            await client.enqueue(definition, input_value="private prompt")

        self.assertEqual(run.exception.code, "task.queue_unsupported")
        self.assertEqual(run.exception.operation, "run")
        self.assertEqual(enqueue.exception.operation, "enqueue")
        self.assertNotIn("private-queue", str(run.exception))

    async def test_output_and_artifacts_reflect_store_records(self) -> None:
        store = InMemoryTaskStore()
        definition = _definition()
        await store.register_definition(
            definition,
            definition_hash="hash-manual",
        )
        pending = await store.create_run(
            TaskExecutionRequest(definition_id="hash-manual")
        )
        persisted = await store.create_run(
            TaskExecutionRequest(definition_id="hash-manual")
        )
        tuple_refs = await store.create_run(
            TaskExecutionRequest(definition_id="hash-manual")
        )
        mapping_ref = await store.create_run(
            TaskExecutionRequest(definition_id="hash-manual")
        )
        client = TaskClient(
            store,
            target=RejectingTarget(),
            hmac_provider=StaticHmacProvider(),
        )

        pending_output = await client.output(pending.run_id)
        pending_artifacts = await client.artifacts(pending.run_id)
        await store.append_artifact(
            persisted.run_id,
            ref=TaskArtifactRef(
                artifact_id="artifact-persisted",
                store="local",
                storage_key="ar/artifact-persisted",
                media_type="text/plain",
                size_bytes=4,
                sha256="a" * 64,
            ),
            purpose=TaskArtifactPurpose.OUTPUT,
            metadata={"safe": "metadata"},
        )
        tuple_refs = await _fail_run(
            store,
            tuple_refs.run_id,
            metadata={"artifacts": ("artifact-1", "artifact-2")},
        )
        mapping_ref = await _fail_run(
            store,
            mapping_ref.run_id,
            metadata={"artifacts": {"artifact_id": "artifact-3"}},
        )

        self.assertFalse(pending_output.ready)
        self.assertIsNone(pending_output.output_summary)
        self.assertEqual(pending_artifacts, ())
        self.assertEqual(
            (await client.output(tuple_refs.run_id)).error,
            {"code": "runnable.failed"},
        )
        persisted_artifacts = await client.artifacts(persisted.run_id)
        self.assertEqual(len(persisted_artifacts), 1)
        self.assertNotIn("storage_key", str(persisted_artifacts))
        self.assertIn("artifact-persisted", str(persisted_artifacts))
        self.assertEqual(
            await client.artifacts(tuple_refs.run_id),
            ("artifact-1", "artifact-2"),
        )
        self.assertEqual(
            await client.artifacts(mapping_ref.run_id),
            ({"artifact_id": "artifact-3"},),
        )


def _definition(
    *,
    privacy: TaskPrivacyPolicy | None = None,
    run: TaskRunPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="agent", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/valid.toml"),
        privacy=privacy or TaskPrivacyPolicy(),
        run=run or TaskRunPolicy.direct(),
    )


async def _fail_run(
    store: InMemoryTaskStore,
    run_id: str,
    *,
    metadata: dict[str, object],
) -> TaskRun:
    await store.transition_run(
        run_id,
        from_states={TaskRunState.CREATED},
        to_state=TaskRunState.VALIDATED,
        reason="validated",
    )
    await store.transition_run(
        run_id,
        from_states={TaskRunState.VALIDATED},
        to_state=TaskRunState.RUNNING,
        reason="started",
    )
    return await store.transition_run(
        run_id,
        from_states={TaskRunState.RUNNING},
        to_state=TaskRunState.FAILED,
        reason="failed",
        result=TaskExecutionResult(
            error={"code": "runnable.failed"},
            metadata=metadata,
        ),
    )


if __name__ == "__main__":
    main()
