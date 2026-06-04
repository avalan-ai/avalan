from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from math import inf
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest import TestCase, main
from unittest.mock import patch

from avalan.task import (
    TASK_VALIDATION_ISSUE_CODES,
    EncryptedPrivacyValue,
    PrivacyAction,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactState,
    TaskDefinition,
    TaskExecutionTarget,
    TaskFeature,
    TaskFileConversionRequest,
    TaskFileConversionResult,
    TaskFileConverterCapability,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputType,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskTargetType,
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    TaskValidationSeverity,
    raise_task_validation_error,
    validate_task_definition,
    validate_task_input,
    validate_task_output,
    validate_task_sections,
)
from avalan.task.converters import TaskFileConversionError


class ConfiguredHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or "hmac-v1",
            algorithm="hmac-sha256",
            secret=b"test-hmac-key",
        )


class ConfiguredEncryptionProvider:
    def encrypt(
        self,
        value: bytes,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> EncryptedPrivacyValue:
        return EncryptedPrivacyValue(
            ciphertext=b"encrypted",
            key_id=key_id or "enc-v1",
            algorithm="test-aead",
        )


class HostileMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("private raw mapping value")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("private raw mapping key")

    def __len__(self) -> int:
        return 1


class HostileDescriptorMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("private raw descriptor value")

    def __iter__(self) -> Iterator[str]:
        return iter(("source_kind", "reference"))

    def __len__(self) -> int:
        return 2

    def get(self, key: str, default: object = None) -> object:
        raise RuntimeError("private raw descriptor lookup")


class HostileTuple(tuple[object, ...]):
    def __iter__(self) -> Iterator[object]:
        raise RuntimeError("private raw array value")


class CapabilityConverter:
    name = "convert"
    version = "1"

    def __init__(
        self,
        *,
        source_mime_types: tuple[str, ...],
        output_mime_types: tuple[str, ...] = ("text/plain",),
        max_input_bytes: int | None = 1024,
        reject_options: bool = False,
        dependency_gates: tuple[TaskFeature, ...] = (),
    ) -> None:
        self._reject_options = reject_options
        self._capability = TaskFileConverterCapability(
            source_mime_types=source_mime_types,
            output_mime_types=output_mime_types,
            supports_streaming=False,
            max_input_bytes=max_input_bytes,
            max_output_bytes=1024,
            dependency_gates=dependency_gates,
        )

    @property
    def capability(self) -> TaskFileConverterCapability:
        return self._capability

    def validate_options(
        self,
        options: Mapping[str, object] | None = None,
    ) -> None:
        if self._reject_options:
            raise TaskFileConversionError("private option value")

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        return TaskFileConversionResult(
            content=content,
            media_type="text/plain",
            metadata={},
        )


class TaskValidationTest(TestCase):
    def test_validation_issue_serializes_stable_fields(self) -> None:
        issue = TaskValidationIssue(
            code="input.invalid_type",
            path="input",
            message="Task input must match the declared scalar type.",
            hint="Pass a string value.",
            category=TaskValidationCategory.VALUE,
        )

        self.assertEqual(
            issue.as_dict(),
            {
                "code": "input.invalid_type",
                "path": "input",
                "category": "value",
                "severity": "error",
                "message": "Task input must match the declared scalar type.",
                "hint": "Pass a string value.",
            },
        )
        self.assertEqual(issue.severity, TaskValidationSeverity.ERROR)
        warning = TaskValidationIssue(
            code="limits.invalid_value",
            path="limits.total_tokens",
            message="Task limit value is invalid.",
            hint="Use a positive integer value.",
            category=TaskValidationCategory.VALUE,
            severity=TaskValidationSeverity.WARNING,
        )
        self.assertEqual(warning.as_dict()["severity"], "warning")

    def test_validation_error_summarizes_without_raw_values(self) -> None:
        issue = TaskValidationIssue(
            code="input.invalid_type",
            path="input",
            message="Task input must match the declared scalar type.",
            hint="Pass a string value.",
            category=TaskValidationCategory.VALUE,
        )

        with self.assertRaises(TaskValidationError) as error:
            raise_task_validation_error((issue,))

        self.assertEqual(error.exception.issues, (issue,))
        self.assertEqual(
            str(error.exception),
            "task validation failed: input.invalid_type at input",
        )
        self.assertNotIn("private prompt", str(error.exception))
        raise_task_validation_error(())

    def test_stable_issue_codes_include_initial_validation_codes(self) -> None:
        self.assertEqual(
            TASK_VALIDATION_ISSUE_CODES,
            {
                "artifact.bytes_unsupported",
                "artifact.retention_required",
                "dependency.jsonschema_missing",
                "dependency.task_documents_missing",
                "execution.path_escape",
                "execution.unknown_target",
                "execution.unsupported_flow",
                "feature.flow_backed_tasks_disabled",
                "feature.remote_url_file_inputs_disabled",
                "input.invalid_file",
                "input.invalid_schema",
                "input.invalid_type",
                "limits.invalid_value",
                "observability.unsupported_sink",
                "output.invalid_schema",
                "output.invalid_type",
                "privacy.encryption_key_missing",
                "privacy.hmac_key_missing",
                "privacy.raw_retention_required",
                "privacy.unknown_action",
                "task.missing_section",
            },
        )

    def test_missing_sections_are_aggregated(self) -> None:
        issues = validate_task_sections({"task": {}, "input": {}})

        self.assertEqual(
            [issue.path for issue in issues],
            ["output", "execution"],
        )
        self.assertEqual(
            {issue.code for issue in issues},
            {"task.missing_section"},
        )
        self.assertTrue(
            all(
                issue.category == TaskValidationCategory.STRUCTURE
                for issue in issues
            )
        )

    def test_valid_agent_definition_has_no_issues(self) -> None:
        definition = self._definition()
        object.__setattr__(definition.run, "priority", 0)

        self.assertEqual(validate_task_definition(definition), ())

    def test_definition_validation_aggregates_safe_diagnostics(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.object(schema={"type": "array"}),
            output_contract=TaskOutputContract.text(),
            execution=TaskExecutionTarget.flow("flows/report.toml"),
            limits=TaskLimitsPolicy(total_tokens=100),
        )
        object.__setattr__(definition.output, "schema", {"type": "object"})
        object.__setattr__(definition.limits, "total_tokens", 0)

        issues = validate_task_definition(definition)

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "input.invalid_schema",
                "output.invalid_schema",
                "limits.invalid_value",
            ],
        )
        self.assertEqual(
            [issue.path for issue in issues],
            [
                "input.schema",
                "output.schema",
                "limits.total_tokens",
            ],
        )
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("flows/report.toml", rendered)

    def test_unknown_contract_types_return_type_issues(self) -> None:
        definition = self._definition()
        object.__setattr__(definition.input, "type", "message")
        object.__setattr__(definition.output, "type", "yaml")

        issues = validate_task_definition(definition)

        self.assertEqual(
            [issue.code for issue in issues],
            ["input.invalid_type", "output.invalid_type"],
        )
        self.assertEqual(
            [issue.path for issue in issues],
            ["input.type", "output.type"],
        )
        self.assertEqual(
            validate_task_input(definition, "raw")[0].code,
            "input.invalid_type",
        )
        self.assertEqual(
            validate_task_output(definition, "raw")[0].code,
            "output.invalid_type",
        )

    def test_unsupported_targets_return_stable_issue_codes(self) -> None:
        deferred_targets = (
            (TaskExecutionTarget.task("tasks/child.task.toml"),),
            (TaskExecutionTarget.model("ai://env:KEY@openai/gpt-4o"),),
            (TaskExecutionTarget.callable("package.module:function"),),
            (TaskExecutionTarget.tool("search"),),
        )

        for (target,) in deferred_targets:
            with self.subTest(target=target.type):
                issues = validate_task_definition(
                    self._definition(execution=target)
                )
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0].code, "execution.unknown_target")
                self.assertEqual(
                    issues[0].category,
                    TaskValidationCategory.UNSUPPORTED,
                )

        unknown = self._definition()
        object.__setattr__(unknown.execution, "type", "workflow")
        issues = validate_task_definition(unknown)
        self.assertEqual(issues[0].code, "execution.unknown_target")

    def test_execution_refs_reject_path_escape_without_raw_refs(
        self,
    ) -> None:
        cases = (
            "../private/secret-agent.toml",
            "/private/secret-agent.toml",
            "agents\\secret-agent.toml",
            "ai://env:KEY@vendor/model",
        )

        for ref in cases:
            with self.subTest(ref=ref):
                issues = validate_task_definition(
                    self._definition(execution=TaskExecutionTarget.agent(ref))
                )
                self.assertEqual(
                    [issue.code for issue in issues],
                    ["execution.path_escape"],
                )
                rendered = " ".join(
                    value
                    for issue in issues
                    for value in issue.as_dict().values()
                )
                self.assertNotIn("secret-agent", rendered)
                self.assertNotIn(ref, rendered)

    def test_execution_refs_stay_inside_allowed_roots(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "allowed"
            agents = root / "agents"
            outside = base / "outside"
            agents.mkdir(parents=True)
            outside.mkdir()
            (agents / "valid.toml").write_text("", encoding="utf-8")
            (outside / "secret.toml").write_text("", encoding="utf-8")
            symlink = agents / "link.toml"
            symlink.symlink_to(outside / "secret.toml")

            valid = validate_task_definition(
                self._definition(
                    execution=TaskExecutionTarget.agent("agents/valid.toml")
                ),
                execution_roots=(root,),
            )
            escaped = validate_task_definition(
                self._definition(
                    execution=TaskExecutionTarget.agent("agents/link.toml")
                ),
                execution_roots=(root,),
            )

        self.assertEqual(valid, ())
        self.assertEqual(
            [issue.code for issue in escaped],
            ["execution.path_escape"],
        )

    def test_execution_ref_validation_handles_invalid_refs(self) -> None:
        missing_ref = self._definition()
        object.__setattr__(missing_ref.execution, "ref", None)

        with TemporaryDirectory() as tmp:
            nul_ref = validate_task_definition(
                self._definition(
                    execution=TaskExecutionTarget.agent("agents/\0bad.toml")
                ),
                execution_roots=(Path(tmp),),
            )

        self.assertEqual(
            [issue.code for issue in validate_task_definition(missing_ref)],
            ["execution.path_escape"],
        )
        self.assertEqual(
            [issue.code for issue in nul_ref],
            ["execution.path_escape"],
        )

    def test_privacy_validation_rejects_unknown_actions_and_unsafe_raw(
        self,
    ) -> None:
        definition = self._definition(
            privacy=TaskPrivacyPolicy(
                output=PrivacyAction.ENCRYPT,
                tool_results=PrivacyAction.STORE,
            )
        )
        object.__setattr__(definition.privacy, "input", "sha256")

        issues = validate_task_definition(
            definition,
            hmac_provider=ConfiguredHmacProvider(),
            require_configured_keys=True,
            raw_storage_allowed=False,
        )

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "privacy.unknown_action",
                "privacy.raw_retention_required",
                "privacy.raw_retention_required",
                "privacy.encryption_key_missing",
                "privacy.encryption_key_missing",
            ],
        )
        self.assertEqual(
            [issue.path for issue in issues],
            [
                "privacy.input",
                "privacy.raw_retention_days",
                "privacy",
                "privacy.tool_results",
                "privacy.output",
            ],
        )

    def test_privacy_validation_handles_invalid_policy_and_retention(
        self,
    ) -> None:
        invalid_policy = self._definition()
        object.__setattr__(invalid_policy, "privacy", object())
        invalid_retention = self._definition(
            privacy=TaskPrivacyPolicy(output=PrivacyAction.ENCRYPT)
        )
        object.__setattr__(
            invalid_retention.privacy,
            "raw_retention_days",
            -1,
        )

        self.assertEqual(
            [issue.code for issue in validate_task_definition(invalid_policy)],
            ["privacy.unknown_action"],
        )
        self.assertEqual(
            [
                issue.code
                for issue in validate_task_definition(invalid_retention)
            ],
            [
                "limits.invalid_value",
                "privacy.raw_retention_required",
            ],
        )

    def test_configured_key_validation_fails_closed_when_keys_missing(
        self,
    ) -> None:
        missing_hmac = validate_task_definition(
            self._definition(),
            require_configured_keys=True,
        )
        missing_encryption = validate_task_definition(
            self._definition(
                privacy=TaskPrivacyPolicy(
                    output=PrivacyAction.ENCRYPT,
                    raw_retention_days=1,
                )
            ),
            hmac_provider=ConfiguredHmacProvider(),
            require_configured_keys=True,
            raw_storage_allowed=True,
        )
        configured = validate_task_definition(
            self._definition(
                privacy=TaskPrivacyPolicy(
                    output=PrivacyAction.ENCRYPT,
                    raw_retention_days=1,
                )
            ),
            hmac_provider=ConfiguredHmacProvider(),
            encryption_provider=ConfiguredEncryptionProvider(),
            require_configured_keys=True,
            raw_storage_allowed=True,
        )

        self.assertEqual(
            [issue.code for issue in missing_hmac],
            ["privacy.hmac_key_missing"],
        )
        self.assertEqual(missing_hmac[0].path, "privacy.input")
        self.assertEqual(
            [issue.code for issue in missing_encryption],
            ["privacy.encryption_key_missing"],
        )
        self.assertEqual(configured, ())

    def test_artifact_raw_storage_requires_retention_and_encryption(
        self,
    ) -> None:
        definition = self._definition(
            artifact=TaskArtifactPolicy(store_bytes=True, encrypt=False)
        )

        issues = validate_task_definition(
            definition,
            raw_storage_allowed=False,
        )

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "artifact.retention_required",
                "artifact.bytes_unsupported",
                "privacy.encryption_key_missing",
            ],
        )
        self.assertEqual(
            [issue.path for issue in issues],
            [
                "artifact.retention_days",
                "artifact.store_bytes",
                "artifact.encrypt",
            ],
        )

    def test_artifact_raw_storage_requires_configured_encryption_key(
        self,
    ) -> None:
        definition = self._definition(
            artifact=TaskArtifactPolicy(
                store_bytes=True,
                retention_days=7,
                encrypt=True,
            )
        )

        issues = validate_task_definition(
            definition,
            hmac_provider=ConfiguredHmacProvider(),
            require_configured_keys=True,
            raw_storage_allowed=True,
        )

        self.assertEqual(
            [issue.code for issue in issues],
            ["privacy.encryption_key_missing"],
        )
        self.assertEqual(issues[0].path, "artifact.encrypt")

    def test_invalid_schema_shapes_return_safe_diagnostics(self) -> None:
        non_json_schema = self._definition(
            input_contract=TaskInputContract.object(
                schema={"type": "object", "default": object()}
            ),
            output_contract=TaskOutputContract.array(schema={"type": "array"}),
        )
        non_mapping_schema = self._definition(
            input_contract=TaskInputContract.object(schema={"type": "object"}),
        )
        object.__setattr__(non_mapping_schema.input, "schema", [])
        missing_schema = self._definition(
            input_contract=TaskInputContract.array(),
            output_contract=TaskOutputContract.object(),
        )
        bad_ref = self._definition(
            input_contract=TaskInputContract.object(schema={"type": "object"}),
            output_contract=TaskOutputContract.json(
                schema_ref="schemas/output.json"
            ),
        )
        object.__setattr__(bad_ref.output, "schema_ref", "")
        both_schema_forms = self._definition(
            input_contract=TaskInputContract.object(schema={"type": "object"}),
        )
        object.__setattr__(
            both_schema_forms.input,
            "schema_ref",
            "schemas/input.json",
        )
        schema_type_array = self._definition(
            output_contract=TaskOutputContract.object(
                schema={"type": ["object", "null"]}
            )
        )

        self.assertEqual(
            [
                issue.code
                for issue in validate_task_definition(non_json_schema)
            ],
            ["input.invalid_schema"],
        )
        self.assertEqual(
            [
                issue.path
                for issue in validate_task_definition(non_mapping_schema)
            ],
            ["input.schema"],
        )
        self.assertEqual(
            [issue.path for issue in validate_task_definition(missing_schema)],
            ["input.schema", "output.schema"],
        )
        self.assertEqual(
            [issue.path for issue in validate_task_definition(bad_ref)],
            ["output.schema_ref"],
        )
        self.assertEqual(
            [
                issue.path
                for issue in validate_task_definition(both_schema_forms)
            ],
            ["input.schema"],
        )
        self.assertEqual(validate_task_definition(schema_type_array), ())

    def test_json_schema_syntax_validation_rejects_invalid_schemas(
        self,
    ) -> None:
        object_schema = self._definition(
            input_contract=TaskInputContract.object(
                schema={
                    "type": "object",
                    "properties": {"name": {"type": "not-a-type"}},
                }
            )
        )
        array_schema = self._definition(
            output_contract=TaskOutputContract.array(
                schema={"type": "array", "items": {"type": "invalid"}}
            )
        )

        object_issues = validate_task_definition(object_schema)
        array_issues = validate_task_definition(array_schema)

        self.assertEqual(
            [issue.code for issue in object_issues],
            [
                "input.invalid_schema",
            ],
        )
        self.assertEqual(object_issues[0].path, "input.schema")
        self.assertEqual(
            [issue.code for issue in array_issues],
            [
                "output.invalid_schema",
            ],
        )
        rendered = " ".join(
            value
            for issue in object_issues + array_issues
            for value in issue.as_dict().values()
        )
        self.assertNotIn("not-a-type", rendered)

    def test_json_schema_runtime_validation_for_structured_contracts(
        self,
    ) -> None:
        object_definition = self._definition(
            input_contract=TaskInputContract.object(
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["question"],
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                    },
                }
            ),
            output_contract=TaskOutputContract.object(
                schema={
                    "type": "object",
                    "required": ["answer"],
                    "properties": {"answer": {"type": "string"}},
                }
            ),
        )
        array_definition = self._definition(
            input_contract=TaskInputContract.array(
                schema={
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "integer", "minimum": 1},
                }
            ),
            output_contract=TaskOutputContract.json(
                schema={
                    "type": "array",
                    "items": {"type": "string"},
                }
            ),
        )

        self.assertEqual(
            validate_task_input(object_definition, {"question": "status"}),
            (),
        )
        self.assertEqual(
            validate_task_output(object_definition, {"answer": "ok"}),
            (),
        )
        self.assertEqual(
            validate_task_input(array_definition, [1, 2]),
            (),
        )
        self.assertEqual(
            validate_task_output(array_definition, ["ok"]),
            (),
        )
        self.assertEqual(
            [
                issue.code
                for issue in validate_task_input(
                    object_definition,
                    {"question": ""},
                )
            ],
            ["input.invalid_type"],
        )
        self.assertEqual(
            [
                issue.code
                for issue in validate_task_input(
                    array_definition,
                    [0],
                )
            ],
            ["input.invalid_type"],
        )
        self.assertEqual(
            [
                issue.code
                for issue in validate_task_output(
                    object_definition,
                    {"answer": 1},
                )
            ],
            ["output.invalid_type"],
        )
        self.assertEqual(
            [
                issue.code
                for issue in validate_task_output(
                    array_definition,
                    [1],
                )
            ],
            ["output.invalid_type"],
        )

    def test_structured_validation_rejects_hostile_mapping_safely(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.object(schema={"type": "object"}),
            output_contract=TaskOutputContract.object(
                schema={"type": "object"}
            ),
        )

        input_issues = validate_task_input(definition, HostileMapping())
        output_issues = validate_task_output(definition, HostileMapping())

        self.assertEqual(
            [issue.code for issue in input_issues],
            ["input.invalid_type"],
        )
        self.assertEqual(
            [issue.code for issue in output_issues],
            ["output.invalid_type"],
        )
        rendered = f"{input_issues} {output_issues}"
        self.assertNotIn("private raw", rendered)

    def test_array_validation_rejects_hostile_tuple_safely(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.array(schema={"type": "array"}),
        )

        issues = validate_task_input(definition, HostileTuple(("private",)))

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "input.invalid_type",
            ],
        )
        self.assertNotIn("private raw", str(issues))

    def test_file_validation_rejects_hostile_descriptor_safely(self) -> None:
        definition = self._definition(input_contract=TaskInputContract.file())

        issues = validate_task_input(definition, HostileDescriptorMapping())

        self.assertEqual(
            [(issue.code, issue.path) for issue in issues],
            [("input.invalid_file", "input")],
        )
        self.assertNotIn("private raw", str(issues))

    def test_file_conversion_capabilities_validate_mime_examples(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(
                conversions=("text", "markdown")
            )
        )
        converters = {
            "text": CapabilityConverter(
                source_mime_types=(
                    "text/*",
                    "application/json",
                    "application/*+json",
                )
            ),
            "markdown": CapabilityConverter(
                source_mime_types=(
                    "text/markdown",
                    "text/plain",
                    "text/html",
                ),
                output_mime_types=("text/markdown",),
            ),
        }
        accepted = (
            ("text/plain", "text"),
            ("application/json", "text"),
            ("application/activity+json", "text"),
            ("text/markdown", "markdown"),
            ("text/html", "markdown"),
        )
        rejected = (
            ("application/pdf", "markdown"),
            (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "markdown",
            ),
            ("image/png", "markdown"),
            ("application/octet-stream", "text"),
            ("application/x-custom", "text"),
        )

        for mime_type, conversion in accepted:
            with self.subTest(mime_type=mime_type, conversion=conversion):
                issues = validate_task_input(
                    definition,
                    TaskFileDescriptor.local_path(
                        "uploads/private-input",
                        mime_type=mime_type,
                        size_bytes=8,
                        conversions=(
                            TaskFileConversionRequest(name=conversion),
                        ),
                    ),
                    file_converters=converters,
                )
                self.assertEqual(issues, ())

        for mime_type, conversion in rejected:
            with self.subTest(mime_type=mime_type, conversion=conversion):
                issues = validate_task_input(
                    definition,
                    TaskFileDescriptor.local_path(
                        "uploads/private-input",
                        mime_type=mime_type,
                        size_bytes=8,
                        conversions=(
                            TaskFileConversionRequest(name=conversion),
                        ),
                    ),
                    file_converters=converters,
                )
                self.assertEqual(
                    [issue.code for issue in issues],
                    ["input.invalid_file"],
                )
                self.assertEqual(issues[0].path, "input.conversions[0]")
                self.assertNotIn("private-input", str(issues))

    def test_file_conversion_capabilities_reject_bad_requests(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(
                conversions=("text", "blocked")
            )
        )

        unknown = validate_task_input(
            definition,
            TaskFileDescriptor.local_path(
                "uploads/private.txt",
                mime_type="text/plain",
                size_bytes=4,
                conversions=(TaskFileConversionRequest(name="text"),),
            ),
            file_converters={},
        )
        bad_options = validate_task_input(
            definition,
            TaskFileDescriptor.local_path(
                "uploads/private.txt",
                mime_type="text/plain",
                size_bytes=4,
                conversions=(
                    TaskFileConversionRequest(
                        name="text",
                        options={"private": "value"},
                    ),
                ),
            ),
            file_converters={
                "text": CapabilityConverter(
                    source_mime_types=("text/plain",),
                    reject_options=True,
                )
            },
        )
        oversized = validate_task_input(
            definition,
            TaskFileDescriptor.local_path(
                "uploads/private.txt",
                mime_type="text/plain",
                size_bytes=5,
                conversions=(TaskFileConversionRequest(name="text"),),
            ),
            file_converters={
                "text": CapabilityConverter(
                    source_mime_types=("text/plain",),
                    max_input_bytes=4,
                )
            },
        )
        missing_cap = validate_task_input(
            definition,
            TaskFileDescriptor.local_path(
                "uploads/private.txt",
                mime_type="text/plain",
                size_bytes=4,
                conversions=(TaskFileConversionRequest(name="text"),),
            ),
            file_converters={
                "text": CapabilityConverter(
                    source_mime_types=("text/plain",),
                    max_input_bytes=None,
                )
            },
        )
        with patch(
            "avalan.task.converters.feature_available",
            return_value=False,
        ):
            dependency_blocked = validate_task_input(
                definition,
                TaskFileDescriptor.local_path(
                    "uploads/private.txt",
                    mime_type="text/plain",
                    size_bytes=4,
                    conversions=(TaskFileConversionRequest(name="blocked"),),
                ),
                file_converters={
                    "text": CapabilityConverter(
                        source_mime_types=("text/plain",),
                    ),
                    "blocked": CapabilityConverter(
                        source_mime_types=("text/plain",),
                        dependency_gates=(TaskFeature.DOCUMENT_CONVERSION,),
                    ),
                },
            )

        self.assertEqual(
            unknown[0].message, "Task file conversion is not supported."
        )
        self.assertEqual(bad_options[0].code, "input.invalid_file")
        self.assertEqual(oversized[0].code, "input.invalid_file")
        self.assertEqual(missing_cap[0].code, "input.invalid_file")
        self.assertEqual(
            dependency_blocked[0].code,
            "dependency.task_documents_missing",
        )
        rendered = (
            f"{unknown} {bad_options} {oversized} "
            f"{missing_cap} {dependency_blocked}"
        )
        self.assertNotIn("private.txt", rendered)
        self.assertNotIn("private option value", rendered)

    def test_definition_validation_checks_registered_converters(self) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.file(
                conversions=("unknown", "blocked")
            )
        )

        with patch(
            "avalan.task.converters.feature_available",
            return_value=False,
        ):
            issues = validate_task_definition(
                definition,
                file_converters={
                    "blocked": CapabilityConverter(
                        source_mime_types=("text/plain",),
                        dependency_gates=(TaskFeature.DOCUMENT_CONVERSION,),
                    )
                },
            )

        self.assertEqual(
            [(issue.code, issue.path) for issue in issues],
            [
                ("input.invalid_file", "input.file_conversions[0]"),
                (
                    "dependency.task_documents_missing",
                    "input.file_conversions[1]",
                ),
            ],
        )

    def test_missing_json_schema_dependency_returns_safe_diagnostic(
        self,
    ) -> None:
        definition = self._definition(
            input_contract=TaskInputContract.object(schema={"type": "object"}),
            output_contract=TaskOutputContract.json(schema={"type": "array"}),
        )

        with patch(
            "avalan.task.validation.import_module",
            side_effect=ModuleNotFoundError("private traceback"),
        ):
            definition_issues = validate_task_definition(definition)
            input_issues = validate_task_input(definition, {})
            output_issues = validate_task_output(definition, [])
            scalar_issues = validate_task_input(
                self._definition(input_contract=TaskInputContract.string()),
                "plain",
            )

        self.assertEqual(
            [issue.path for issue in definition_issues],
            ["input.schema", "output.schema"],
        )
        self.assertEqual(
            {issue.code for issue in definition_issues},
            {"dependency.jsonschema_missing"},
        )
        self.assertEqual(input_issues[0].code, "dependency.jsonschema_missing")
        self.assertEqual(
            output_issues[0].code, "dependency.jsonschema_missing"
        )
        self.assertEqual(scalar_issues, ())
        rendered = " ".join(
            value
            for issue in definition_issues + input_issues + output_issues
            for value in issue.as_dict().values()
        )
        self.assertNotIn("private traceback", rendered)

    def test_json_schema_adapter_fails_closed_for_incomplete_module(
        self,
    ) -> None:
        module = ModuleType("jsonschema")
        module.Draft202012Validator = object()
        module.SchemaError = object()
        module.ValidationError = ValueError
        definition = self._definition(
            input_contract=TaskInputContract.object(schema={"type": "object"})
        )

        with patch(
            "avalan.task.validation.import_module", return_value=module
        ):
            issues = validate_task_definition(definition)

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "dependency.jsonschema_missing",
            ],
        )

    def test_json_schema_runtime_schema_errors_are_safe(self) -> None:
        class RuntimeSchemaError(Exception):
            pass

        class RuntimeValidationError(Exception):
            pass

        class BrokenValidator:
            def __init__(self, schema: Mapping[str, object]) -> None:
                self.schema = schema

            def validate(self, instance: object) -> None:
                raise RuntimeSchemaError("raw schema details")

            @staticmethod
            def check_schema(schema: Mapping[str, object]) -> None:
                return None

        module = ModuleType("jsonschema")
        module.Draft202012Validator = BrokenValidator
        module.SchemaError = RuntimeSchemaError
        module.ValidationError = RuntimeValidationError
        definition = self._definition(
            output_contract=TaskOutputContract.array(schema={"type": "array"})
        )

        with patch(
            "avalan.task.validation.import_module", return_value=module
        ):
            issues = validate_task_output(definition, [])

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "output.invalid_schema",
            ],
        )
        self.assertNotIn(
            "raw schema details",
            " ".join(
                value for issue in issues for value in issue.as_dict().values()
            ),
        )

    def test_runtime_schema_validation_skips_absent_inline_schema(
        self,
    ) -> None:
        definition = self._definition(
            output_contract=TaskOutputContract.array()
        )

        issues = validate_task_output(definition, [])

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "output.invalid_schema",
            ],
        )

    def test_limit_validation_covers_common_boundaries(self) -> None:
        definition = self._definition(limits=TaskLimitsPolicy(file_count=1))
        object.__setattr__(definition.run, "timeout_seconds", 0)
        object.__setattr__(definition.run, "priority", -1)
        object.__setattr__(definition.run, "concurrency", True)
        object.__setattr__(definition.retry, "max_attempts", 0)
        object.__setattr__(definition.retry, "max_delay_seconds", False)
        object.__setattr__(definition.artifact, "retention_days", 0)
        object.__setattr__(definition.artifact, "max_count", -1)
        object.__setattr__(definition.artifact, "max_bytes", True)
        object.__setattr__(definition.limits, "file_count", 0)

        issues = validate_task_definition(definition)

        self.assertEqual(
            {issue.code for issue in issues}, {"limits.invalid_value"}
        )
        self.assertEqual(
            [issue.path for issue in issues],
            [
                "run.timeout_seconds",
                "run.priority",
                "run.concurrency",
                "retry.max_attempts",
                "retry.max_delay_seconds",
                "artifact.retention_days",
                "artifact.max_count",
                "artifact.max_bytes",
                "limits.file_count",
            ],
        )

    def test_scalar_input_validation_rejects_wrong_value_types(self) -> None:
        cases = (
            (
                self._definition(input_contract=TaskInputContract.string()),
                10,
                "Pass a string value.",
            ),
            (
                self._definition(input_contract=TaskInputContract.integer()),
                True,
                "Pass an integer value.",
            ),
            (
                self._definition(input_contract=TaskInputContract.number()),
                inf,
                "Pass a finite numeric value.",
            ),
            (
                self._definition(input_contract=TaskInputContract.number()),
                True,
                "Pass a finite numeric value.",
            ),
            (
                self._definition(input_contract=TaskInputContract.boolean()),
                "true",
                "Pass a boolean value.",
            ),
        )

        for definition, value, hint in cases:
            with self.subTest(input_type=definition.input.type):
                issues = validate_task_input(definition, value)
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0].code, "input.invalid_type")
                self.assertEqual(issues[0].hint, hint)

    def test_input_validation_handles_required_and_structured_values(
        self,
    ) -> None:
        optional = self._definition(
            input_contract=TaskInputContract(
                type=TaskInputType.STRING,
                required=False,
            )
        )
        required = self._definition(input_contract=TaskInputContract.string())
        object_definition = self._definition(
            input_contract=TaskInputContract.object(schema={"type": "object"})
        )
        array_definition = self._definition(
            input_contract=TaskInputContract.array(schema={"type": "array"})
        )

        self.assertEqual(validate_task_input(optional, None), ())
        self.assertEqual(
            validate_task_input(required, None)[0].code,
            "input.invalid_type",
        )
        self.assertEqual(
            validate_task_input(object_definition, [])[0].hint,
            "Pass an object value.",
        )
        self.assertEqual(
            validate_task_input(array_definition, {})[0].hint,
            "Pass an array value.",
        )
        self.assertEqual(validate_task_input(array_definition, []), ())
        self.assertEqual(
            validate_task_input(object_definition, {1: "bad"})[0].hint,
            "Pass an object with JSON-compatible values.",
        )
        self.assertEqual(
            validate_task_input(array_definition, [object()])[0].hint,
            "Pass an array with JSON-compatible values.",
        )
        self.assertEqual(
            validate_task_input(
                self._definition(input_contract=TaskInputContract.number()),
                1,
            ),
            (),
        )
        self.assertEqual(
            validate_task_input(
                self._definition(input_contract=TaskInputContract.file()),
                TaskFileDescriptor.local_path("uploads/private.txt"),
            ),
            (),
        )
        self.assertEqual(
            validate_task_input(
                self._definition(
                    input_contract=TaskInputContract.file_array()
                ),
                [TaskFileDescriptor.local_path("uploads/private.txt")],
            ),
            (),
        )

    def test_output_validation_rejects_wrong_value_types(self) -> None:
        cases = (
            (
                self._definition(output_contract=TaskOutputContract.text()),
                10,
                "Return a text value.",
            ),
            (
                self._definition(
                    output_contract=TaskOutputContract.object(
                        schema={"type": "object"}
                    )
                ),
                [],
                "Return an object value.",
            ),
            (
                self._definition(
                    output_contract=TaskOutputContract.array(
                        schema={"type": "array"}
                    )
                ),
                {},
                "Return an array value.",
            ),
            (
                self._definition(
                    output_contract=TaskOutputContract.json(
                        schema={"type": "object"}
                    )
                ),
                {"bad": object()},
                "Return a JSON-compatible value.",
            ),
        )

        for definition, value, hint in cases:
            with self.subTest(output_type=definition.output.type):
                issues = validate_task_output(definition, value)
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0].code, "output.invalid_type")
                self.assertEqual(issues[0].hint, hint)
        self.assertEqual(
            validate_task_output(
                self._definition(
                    output_contract=TaskOutputContract.object(
                        schema={"type": "object"}
                    )
                ),
                {1: "bad"},
            )[0].hint,
            "Return an object with JSON-compatible values.",
        )
        self.assertEqual(
            validate_task_output(
                self._definition(
                    output_contract=TaskOutputContract.array(
                        schema={"type": "array"}
                    )
                ),
                [object()],
            )[0].hint,
            "Return an array with JSON-compatible values.",
        )

    def test_output_validation_accepts_non_scalar_deferred_values(
        self,
    ) -> None:
        self.assertEqual(
            validate_task_output(
                self._definition(
                    output_contract=TaskOutputContract.json(
                        schema={"type": "number"}
                    )
                ),
                1,
            ),
            (),
        )
        self.assertEqual(
            validate_task_output(
                self._definition(
                    output_contract=TaskOutputContract.json(
                        schema={"type": "number"}
                    )
                ),
                1.5,
            ),
            (),
        )
        self.assertEqual(
            validate_task_output(
                self._definition(
                    output_contract=TaskOutputContract.json(
                        schema={"type": "array"}
                    )
                ),
                [],
            ),
            (),
        )

    def test_output_validation_checks_artifact_contracts(self) -> None:
        ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
            media_type="text/plain",
            size_bytes=4,
            sha256="a" * 64,
        )
        lost_record = TaskArtifactRecord(
            artifact_id="artifact-2",
            run_id="run-1",
            purpose=TaskArtifactPurpose.OUTPUT,
            state=TaskArtifactState.LOST,
            ref=TaskArtifactRef(
                artifact_id="artifact-2",
                store="local",
                storage_key="ar/artifact-2",
            ),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

        self.assertEqual(
            validate_task_output(
                self._definition(output_contract=TaskOutputContract.file()),
                ref,
            ),
            (),
        )
        self.assertEqual(
            validate_task_output(
                self._definition(
                    output_contract=TaskOutputContract.file_array()
                ),
                [ref],
            ),
            (),
        )
        self.assertEqual(
            validate_task_output(
                self._definition(
                    output_contract=TaskOutputContract.artifact_array()
                ),
                [lost_record],
            ),
            (),
        )

        malformed = validate_task_output(
            self._definition(output_contract=TaskOutputContract.file()),
            object(),
        )
        deleted_file = validate_task_output(
            self._definition(output_contract=TaskOutputContract.file_array()),
            [lost_record],
        )
        non_array = validate_task_output(
            self._definition(
                output_contract=TaskOutputContract.artifact_array()
            ),
            ref,
        )
        wrong_purpose = validate_task_output(
            self._definition(
                output_contract=TaskOutputContract.artifact_array()
            ),
            [
                TaskArtifactRecord(
                    artifact_id="artifact-3",
                    run_id="run-1",
                    purpose=TaskArtifactPurpose.INPUT,
                    state=TaskArtifactState.READY,
                    ref=TaskArtifactRef(
                        artifact_id="artifact-3",
                        store="local",
                        storage_key="ar/artifact-3",
                    ),
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    updated_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            ],
        )
        too_many = validate_task_output(
            self._definition(
                output_contract=TaskOutputContract.file_array(),
                artifact=TaskArtifactPolicy(max_count=1),
            ),
            [ref, ref],
        )
        too_large = validate_task_output(
            self._definition(
                output_contract=TaskOutputContract.file(),
                limits=TaskLimitsPolicy(artifact_bytes=3),
            ),
            ref,
        )

        self.assertEqual(malformed[0].code, "output.invalid_type")
        self.assertEqual(malformed[0].path, "output")
        self.assertEqual(deleted_file[0].path, "output[0].state")
        self.assertEqual(non_array[0].path, "output")
        self.assertEqual(wrong_purpose[0].path, "output[0].purpose")
        self.assertEqual(too_many[0].path, "output")
        self.assertEqual(too_large[0].path, "output.size_bytes")

    def test_input_and_output_diagnostics_do_not_include_raw_values(
        self,
    ) -> None:
        raw_input = "private prompt with /Users/person/secret.pdf"
        raw_output = {"secret": "raw model output"}
        input_issues = validate_task_input(
            self._definition(input_contract=TaskInputContract.integer()),
            raw_input,
        )
        output_issues = validate_task_output(
            self._definition(output_contract=TaskOutputContract.text()),
            raw_output,
        )

        rendered = " ".join(
            value
            for issue in input_issues + output_issues
            for value in issue.as_dict().values()
        )

        self.assertNotIn("private prompt", rendered)
        self.assertNotIn("/Users/person/secret.pdf", rendered)
        self.assertNotIn("raw model output", rendered)

    def _definition(
        self,
        *,
        input_contract: TaskInputContract | None = None,
        output_contract: TaskOutputContract | None = None,
        execution: TaskExecutionTarget | None = None,
        privacy: TaskPrivacyPolicy | None = None,
        artifact: TaskArtifactPolicy | None = None,
        limits: TaskLimitsPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="validation", version="1"),
            input=input_contract or TaskInputContract.string(),
            output=output_contract or TaskOutputContract.text(),
            execution=execution
            or TaskExecutionTarget(
                type=TaskTargetType.AGENT,
                ref="agents/validation.toml",
            ),
            privacy=privacy or TaskPrivacyPolicy(),
            artifact=artifact or TaskArtifactPolicy(),
            limits=limits or TaskLimitsPolicy(),
        )


if __name__ == "__main__":
    main()
