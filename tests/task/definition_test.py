from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    IdempotencyMode,
    ObservabilitySinkType,
    PrivacyAction,
    RetryBackoff,
    RunMode,
    TaskArtifactPolicy,
    TaskDefinition,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInputType,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskOutputType,
    TaskPrivacyPolicy,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskTargetType,
)


class TaskDefinitionTest(TestCase):
    def test_minimal_agent_definition_uses_local_defaults(self) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(
                name="person_explainer",
                version="1",
                description="Explain who the person named in the input is.",
            ),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent(
                "agents/person_explainer.toml"
            ),
        )

        self.assertEqual(definition.task.name, "person_explainer")
        self.assertEqual(definition.input.type, TaskInputType.STRING)
        self.assertEqual(definition.output.type, TaskOutputType.TEXT)
        self.assertEqual(definition.execution.type, TaskTargetType.AGENT)
        self.assertEqual(definition.run.mode, RunMode.DIRECT)
        self.assertEqual(definition.run.timeout_seconds, 300)
        self.assertEqual(definition.run.idempotency, IdempotencyMode.NONE)
        self.assertEqual(definition.retry.max_attempts, 1)
        self.assertEqual(definition.retry.backoff, RetryBackoff.NONE)
        self.assertEqual(definition.observability.sinks, ("pgsql",))
        self.assertIsNone(definition.definition_base)

    def test_sdk_definition_base_can_be_declared(self) -> None:
        definition = TaskDefinition(
            task=TaskMetadata(name="sdk_base", version="1"),
            input=TaskInputContract.string(),
            output=TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/sdk_base.toml"),
            definition_base=Path("tasks/sdk_base.task.toml"),
        )

        self.assertEqual(
            definition.definition_base,
            Path("tasks/sdk_base.task.toml"),
        )

    def test_enum_values_preserve_task_definition_vocabulary(self) -> None:
        self.assertEqual(TaskInputType.FILE_ARRAY.value, "file[]")
        self.assertEqual(TaskOutputType.FILE_ARRAY.value, "file[]")
        self.assertEqual(
            TaskOutputType.ARTIFACT_ARRAY.value,
            "artifact[]",
        )
        self.assertEqual(TaskTargetType.CALLABLE.value, "callable")
        self.assertEqual(
            IdempotencyMode.INPUT_AND_FILES_HASH.value,
            "input_and_files_hash",
        )
        self.assertEqual(PrivacyAction.ENCRYPT.value, "encrypt")
        self.assertEqual(ObservabilitySinkType.OTEL.value, "otel")

    def test_structured_contracts_freeze_nested_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        contract = TaskInputContract.object(schema)

        schema["type"] = "array"
        cast(dict[str, object], schema["properties"])["name"] = {
            "type": "number"
        }

        self.assertIsNotNone(contract.schema)
        frozen_schema = cast(Mapping[str, object], contract.schema)
        properties = cast(Mapping[str, object], frozen_schema["properties"])
        name = cast(Mapping[str, object], properties["name"])
        self.assertEqual(frozen_schema["type"], "object")
        self.assertEqual(name["type"], "string")
        self.assertEqual(frozen_schema["required"], ("name",))
        with self.assertRaises(TypeError):
            cast(dict[str, object], contract.schema)["type"] = "array"

    def test_metadata_and_variables_reject_mutation_attempts(self) -> None:
        annotations = {
            "labels": ["initial"],
            "nested": {"owner": "ops"},
        }
        variables = {
            "settings": {"locale": "en"},
            "steps": ["draft"],
        }

        metadata = TaskMetadata(
            name="hostile_metadata",
            version="1",
            annotations=annotations,
        )
        target = TaskExecutionTarget.agent(
            "agents/a.toml",
            variables=variables,
        )
        annotations["labels"].append("mutated")
        cast(dict[str, object], annotations["nested"])["owner"] = "attacker"
        cast(dict[str, object], variables["settings"])["locale"] = "es"
        variables["steps"].append("publish")

        self.assertEqual(metadata.annotations["labels"], ("initial",))
        self.assertEqual(
            cast(Mapping[str, object], metadata.annotations["nested"])[
                "owner"
            ],
            "ops",
        )
        self.assertEqual(
            cast(Mapping[str, object], target.variables["settings"])["locale"],
            "en",
        )
        self.assertEqual(target.variables["steps"], ("draft",))
        with self.assertRaises(TypeError):
            cast(dict[str, object], metadata.annotations)["added"] = "raw"
        with self.assertRaises(AssertionError):
            TaskMetadata(
                name="bad_metadata",
                version="1",
                annotations=cast(Mapping[str, object], {1: "bad"}),
            )

    def test_contract_helpers_cover_scalar_structured_and_file_types(
        self,
    ) -> None:
        self.assertEqual(
            TaskInputContract.integer().type,
            TaskInputType.INTEGER,
        )
        self.assertEqual(TaskInputContract.number().type, TaskInputType.NUMBER)
        self.assertEqual(
            TaskInputContract.boolean().type,
            TaskInputType.BOOLEAN,
        )
        self.assertEqual(TaskInputContract.array().type, TaskInputType.ARRAY)
        file_input = TaskInputContract.file_array(
            conversions=("markdown", "text"),
            mime_types=("application/pdf",),
        )
        self.assertEqual(file_input.type, TaskInputType.FILE_ARRAY)
        self.assertEqual(file_input.file_conversions, ("markdown", "text"))
        self.assertEqual(file_input.mime_types, ("application/pdf",))

        self.assertEqual(TaskOutputContract.json().type, TaskOutputType.JSON)
        self.assertEqual(
            TaskOutputContract.object().type,
            TaskOutputType.OBJECT,
        )
        self.assertEqual(TaskOutputContract.array().type, TaskOutputType.ARRAY)
        self.assertEqual(TaskOutputContract.file().type, TaskOutputType.FILE)
        self.assertEqual(
            TaskOutputContract.file_array().type,
            TaskOutputType.FILE_ARRAY,
        )
        self.assertEqual(
            TaskOutputContract.artifact_array().type,
            TaskOutputType.ARTIFACT_ARRAY,
        )

    def test_execution_target_helpers_cover_deferred_target_types(
        self,
    ) -> None:
        variables = {"locale": "en", "limits": {"top_k": 3}}
        targets = (
            TaskExecutionTarget.flow("flows/report.toml"),
            TaskExecutionTarget.task("tasks/child.task.toml"),
            TaskExecutionTarget.model("ai://env:OPENAI_API_KEY@openai/gpt-4o"),
            TaskExecutionTarget.callable("package.module:function"),
            TaskExecutionTarget.tool("search"),
        )

        self.assertEqual(
            [target.type for target in targets],
            [
                TaskTargetType.FLOW,
                TaskTargetType.TASK,
                TaskTargetType.MODEL,
                TaskTargetType.CALLABLE,
                TaskTargetType.TOOL,
            ],
        )
        target = TaskExecutionTarget.agent(
            "agents/a.toml", variables=variables
        )
        variables["locale"] = "es"
        self.assertEqual(target.variables["locale"], "en")

    def test_policy_helpers_represent_queue_privacy_artifacts_limits(
        self,
    ) -> None:
        run = TaskRunPolicy.queued(
            "documents",
            timeout_seconds=120,
            priority=10,
            concurrency=2,
        )
        retry = TaskRetryPolicy(
            max_attempts=3,
            backoff=RetryBackoff.EXPONENTIAL,
            max_delay_seconds=60,
            jitter=True,
        )
        privacy = TaskPrivacyPolicy(
            input=PrivacyAction.HASH,
            output=PrivacyAction.ENCRYPT,
            raw_retention_days=7,
        )
        artifact = TaskArtifactPolicy.raw_storage(
            retention_days=30,
            storage="local",
        )
        limits = TaskLimitsPolicy(
            input_bytes=1024,
            file_count=2,
            file_bytes=2048,
            output_bytes=4096,
            artifact_count=3,
            artifact_bytes=8192,
            total_tokens=1000,
        )
        observability = TaskObservabilityPolicy(
            sinks=(
                ObservabilitySinkType.PGSQL,
                ObservabilitySinkType.PROMETHEUS,
                ObservabilitySinkType.OTEL,
            ),
        )

        self.assertEqual(run.mode, RunMode.QUEUE)
        self.assertEqual(run.queue, "documents")
        self.assertEqual(
            run.idempotency, (IdempotencyMode.INPUT_AND_FILES_HASH)
        )
        self.assertEqual(retry.max_attempts, 3)
        self.assertEqual(privacy.output, PrivacyAction.ENCRYPT)
        self.assertTrue(artifact.store_bytes)
        self.assertTrue(artifact.encrypt)
        self.assertEqual(limits.total_tokens, 1000)
        self.assertEqual(observability.sinks[-1], ObservabilitySinkType.OTEL)

    def test_noop_observability_disables_metrics_trace_and_events(
        self,
    ) -> None:
        observability = TaskObservabilityPolicy.noop()

        self.assertEqual(observability.sinks, (ObservabilitySinkType.NOOP,))
        self.assertFalse(observability.metrics)
        self.assertFalse(observability.trace)
        self.assertFalse(observability.capture_events)

    def test_remaining_constructor_branches_preserve_values(self) -> None:
        input_contract = TaskInputContract.file(
            conversions=("native",),
            description="Upload a report.",
        )
        output_contract = TaskOutputContract.json(
            {"type": "object"},
            description="Return structured JSON.",
        )
        input_schema_ref = TaskInputContract.object(
            schema_ref="schemas/input.json"
        )
        output_schema_ref = TaskOutputContract.object(
            schema_ref="schemas/output.json"
        )
        run = TaskRunPolicy.direct(
            timeout_seconds=60,
            idempotency=IdempotencyMode.INPUT_HASH,
        )
        direct_with_queue = TaskRunPolicy(queue="debug")
        custom_with_key = TaskRunPolicy(
            idempotency=IdempotencyMode.CUSTOM,
            idempotency_key_path="input.request_id",
        )
        non_custom_with_key = TaskRunPolicy(
            idempotency_key_path="input.request_id"
        )
        privacy = TaskPrivacyPolicy.default()
        artifact = TaskArtifactPolicy.references_only(retention_days=3)

        self.assertEqual(input_contract.type, TaskInputType.FILE)
        self.assertEqual(input_contract.description, "Upload a report.")
        self.assertEqual(
            output_contract.description, "Return structured JSON."
        )
        self.assertEqual(
            cast(Mapping[str, object], output_contract.schema)["type"],
            "object",
        )
        self.assertEqual(input_schema_ref.schema_ref, "schemas/input.json")
        self.assertEqual(output_schema_ref.schema_ref, "schemas/output.json")
        self.assertEqual(run.timeout_seconds, 60)
        self.assertEqual(run.idempotency, IdempotencyMode.INPUT_HASH)
        self.assertEqual(direct_with_queue.queue, "debug")
        self.assertEqual(
            custom_with_key.idempotency_key_path,
            "input.request_id",
        )
        self.assertEqual(
            non_custom_with_key.idempotency_key_path,
            "input.request_id",
        )
        self.assertEqual(privacy.input, PrivacyAction.HASH)
        self.assertEqual(artifact.retention_days, 3)
        self.assertFalse(artifact.store_bytes)

    def test_definition_instances_are_frozen_and_slotted(self) -> None:
        metadata = TaskMetadata(name="immutable", version="1")

        with self.assertRaises(FrozenInstanceError):
            setattr(metadata, "name", "changed")
        with self.assertRaises((AttributeError, TypeError)):
            setattr(metadata, "unknown", "value")

    def test_invalid_authoring_values_fail_fast(self) -> None:
        with self.assertRaises(AssertionError):
            TaskMetadata(name="", version="1")
        with self.assertRaises(AssertionError):
            TaskInputContract(type=cast(TaskInputType, "string"))
        with self.assertRaises(AssertionError):
            TaskInputContract(
                type=TaskInputType.STRING,
                schema={"type": "string"},
            )
        with self.assertRaises(AssertionError):
            TaskInputContract(
                type=TaskInputType.STRING,
                schema_ref="schemas/input.json",
            )
        with self.assertRaises(AssertionError):
            TaskInputContract.object(
                schema={"type": "object"},
                schema_ref="schemas/input.json",
            )
        with self.assertRaises(AssertionError):
            TaskInputContract(
                type=TaskInputType.STRING,
                file_conversions=("markdown",),
            )
        with self.assertRaises(AssertionError):
            TaskOutputContract(
                type=TaskOutputType.TEXT,
                schema={"type": "string"},
            )
        with self.assertRaises(AssertionError):
            TaskOutputContract(
                type=TaskOutputType.TEXT,
                schema_ref="schemas/output.json",
            )
        with self.assertRaises(AssertionError):
            TaskOutputContract.json(
                schema={"type": "object"},
                schema_ref="schemas/output.json",
            )
        with self.assertRaises(AssertionError):
            TaskExecutionTarget.agent("")
        with self.assertRaises(AssertionError):
            TaskDefinition(
                task=TaskMetadata(name="invalid_base", version="1"),
                input=TaskInputContract.string(),
                output=TaskOutputContract.text(),
                execution=TaskExecutionTarget.agent("agents/a.toml"),
                definition_base="",
            )

    def test_invalid_policy_boundaries_fail_fast(self) -> None:
        with self.assertRaises(AssertionError):
            TaskRunPolicy(timeout_seconds=0)
        with self.assertRaises(AssertionError):
            TaskRunPolicy(mode=RunMode.QUEUE)
        with self.assertRaises(AssertionError):
            TaskRunPolicy(priority=-1)
        with self.assertRaises(AssertionError):
            TaskRunPolicy(
                idempotency=IdempotencyMode.CUSTOM,
                idempotency_key_path="",
            )
        with self.assertRaises(AssertionError):
            TaskRetryPolicy(max_attempts=0)
        with self.assertRaises(AssertionError):
            TaskPrivacyPolicy(raw_retention_days=-1)
        with self.assertRaises(AssertionError):
            TaskArtifactPolicy(retention_days=0)
        with self.assertRaises(AssertionError):
            TaskLimitsPolicy(total_tokens=0)
        with self.assertRaises(AssertionError):
            TaskObservabilityPolicy(sinks=())
        with self.assertRaises(AssertionError):
            TaskObservabilityPolicy(
                sinks=(
                    ObservabilitySinkType.NOOP,
                    ObservabilitySinkType.PGSQL,
                )
            )


if __name__ == "__main__":
    main()
