from asyncio import CancelledError
from asyncio import run as asyncio_run
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import patch
from uuid import UUID, uuid4

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
)
from avalan.event import Event, EventType
from avalan.model import (
    FileDeliveryDecision,
    FileDeliveryLimit,
    FileDeliveryMode,
    FileDeliveryProfile,
    LocalFileDeliveryProfile,
)
from avalan.task import (
    ArtifactStorePolicyError,
    DirectTaskRunner,
    TaskArtifactPolicy,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactStat,
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionTarget,
    TaskFileConversionRequest,
    TaskFileDeliveryPlan,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskOutputContract,
    TaskOutputParseError,
    TaskProviderReference,
    TaskProviderReferenceKind,
    TaskProviderStructuredOutputError,
    TaskRetryPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    UsageSource,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.store import TaskExecutionContext
from avalan.task.stores import InMemoryTaskStore
from avalan.task.targets import AgentTaskTargetRunner
from avalan.task.targets import agent as agent_module


class FakeResponse:
    def __init__(
        self,
        text: str,
        *,
        input_token_count: int | None = None,
        output_token_count: int | None = None,
    ) -> None:
        self.text = text
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count

    async def to_str(self) -> str:
        return self.text

    async def to_json(self) -> str:
        return self.text


class CancellableResponse(FakeResponse):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.cancellation_checker: Callable[[], Awaitable[None]] | None = None

    def set_cancellation_checker(
        self,
        checker: Callable[[], Awaitable[None]] | None,
    ) -> None:
        self.cancellation_checker = checker

    async def to_str(self) -> str:
        if self.cancellation_checker is not None:
            await self.cancellation_checker()
        return await super().to_str()


class TextOnlyResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    async def to_str(self) -> str:
        return self.text


class FailingJsonResponse(TextOnlyResponse):
    async def to_json(self) -> str:
        raise RuntimeError("private provider body with api key sk-test-secret")


class NonStringJsonResponse(TextOnlyResponse):
    async def to_json(self) -> object:
        return {"private": "raw object"}


class BareResponse:
    pass


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
        for listener in list(self.listeners):
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
        trigger = getattr(self.event_manager, "trigger", None)
        if self._loader.emit_event and callable(trigger):
            await trigger(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "token": "secret-token",
                        "token_id": 7,
                        "model_id": "private-model",
                    },
                )
            )
        response = self._loader.next_response()
        if isinstance(response, BaseException):
            raise response
        if response is not None:
            return response
        return FakeResponse(self._loader.response_text)


class FakeLoader:
    def __init__(
        self,
        *,
        response_text: str = "summary",
        response: object | None = None,
        responses: tuple[object, ...] = (),
        emit_event: bool = False,
    ) -> None:
        self.response_text = response_text
        self.response = response
        self.responses = list(responses)
        self.emit_event = emit_event
        self.event_manager: object = FakeEventManager()
        self.paths: list[str] = []
        self.agent_ids: list[UUID | None] = []
        self.disable_memory_values: list[bool] = []
        self.uris: list[str | None] = []
        self.inputs: list[object] = []
        self.entered = 0
        self.exited = 0

    def next_response(self) -> object | None:
        if self.responses:
            return self.responses.pop(0)
        return self.response

    async def from_file(
        self,
        path: str,
        *,
        agent_id: UUID | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> FakeOrchestrator:
        self.paths.append(path)
        self.agent_ids.append(agent_id)
        self.disable_memory_values.append(disable_memory)
        self.uris.append(uri)
        return FakeOrchestrator(self)


class FakeArtifactStore:
    def __init__(self, data: bytes = b"private bytes") -> None:
        self.data = data
        self.max_bytes_values: list[int | None] = []

    async def open(self, ref: TaskArtifactRef) -> BytesIO:
        return BytesIO(self.data)

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> BytesIO:
        self.max_bytes_values.append(max_bytes)
        if max_bytes is not None and len(self.data) > max_bytes:
            raise ArtifactStorePolicyError("private artifact bytes exceeded")
        return BytesIO(self.data)

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        return TaskArtifactStat(
            ref=ref,
            size_bytes=len(self.data),
            sha256=("0" * 64 if ref.sha256 is None else ref.sha256),
        )


class FailingReadArtifactStore(FakeArtifactStore):
    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> BytesIO:
        self.max_bytes_values.append(max_bytes)
        return FailingReader(self.data)


class FailingReader(BytesIO):
    def read(self, size: int = -1) -> bytes:
        raise ArtifactStorePolicyError("private read failure")


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


class AgentTaskTargetRunnerValidationTest(TestCase):
    def test_valid_agent_reference_loads_through_loader_boundary(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"
goal_instructions = "Be brief."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(
                FakeLoader(),
                ref_base=root,
            )
            issues = self._run_validate(runner, self._definition())
            rooted_issues = asyncio_run(
                runner.validate_definition(
                    self._definition(),
                    TaskValidationContext(execution_roots=(root,)),
                )
            )

        self.assertEqual(issues, ())
        self.assertEqual(rooted_issues, ())

    def test_simple_prompt_agent_reference_loads_through_loader_boundary(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
system = "Answer task inputs directly."
user = "Use one sentence."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(
                FakeLoader(),
                ref_base=root,
            )
            issues = self._run_validate(runner, self._definition())

        self.assertEqual(issues, ())

    def test_agent_response_schema_must_match_task_output_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "agents" / "schemas" / "answer.json"
            agent_path = root / "agents" / "valid.toml"
            schema_path.parent.mkdir(parents=True)
            schema_path.write_text(
                """
                {
                  "type": "object",
                  "required": ["answer"],
                  "properties": {"answer": {"type": "string"}}
                }
                """,
                encoding="utf-8",
            )
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"

[run.response_format]
type = "json_schema"
name = "answer"
schema_ref = "schemas/answer.json"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)
            issues = self._run_validate(
                runner,
                self._definition(
                    output_contract=TaskOutputContract.object(
                        schema={
                            "type": "object",
                            "required": ["answer"],
                            "properties": {
                                "answer": {"type": "integer"},
                            },
                        }
                    ),
                ),
            )

        self.assertEqual(
            [(issue.code, issue.path) for issue in issues],
            [("output.invalid_schema", "output.schema")],
        )
        rendered = " ".join(issues[0].as_dict().values())
        self.assertNotIn("answer.json", rendered)

    def test_agent_response_schema_can_match_task_output_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "agents" / "schemas" / "answer.json"
            agent_path = root / "agents" / "valid.toml"
            schema_path.parent.mkdir(parents=True)
            schema_path.write_text(
                """
                {
                  "required": ["answer"],
                  "properties": {"answer": {"type": "string"}},
                  "type": "object"
                }
                """,
                encoding="utf-8",
            )
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"

[run.response_format]
type = "json_schema"
name = "answer"
schema_ref = "schemas/answer.json"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)
            issues = self._run_validate(
                runner,
                self._definition(
                    output_contract=TaskOutputContract.object(
                        schema={
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"},
                            },
                            "required": ["answer"],
                        }
                    ),
                ),
            )

        self.assertEqual(issues, ())

    def test_agent_schema_helpers_cover_invalid_shapes(self) -> None:
        definition = self._definition(
            output_contract=TaskOutputContract.object(
                schema={"type": "object"}
            )
        )
        object.__setattr__(definition.output, "schema", {"bad": object()})
        invalid_schema_issues = agent_module._validate_agent_output_schema(
            definition,
            {
                "run": {
                    "response_format": {
                        "type": "json_schema",
                        "schema": {"type": "object"},
                    }
                }
            },
        )

        self.assertEqual(
            [(issue.code, issue.path) for issue in invalid_schema_issues],
            [("output.invalid_schema", "output.schema")],
        )
        self.assertIsNone(
            agent_module._agent_response_format_schema(
                {"run": {"response_format": "bad"}}
            )
        )
        self.assertIsNone(
            agent_module._agent_response_format_schema(
                {"run": {"response_format": {"type": "text"}}}
            )
        )
        self.assertIsNone(
            agent_module._agent_response_format_schema(
                {
                    "run": {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": "bad",
                        }
                    }
                }
            )
        )
        self.assertIsNone(
            agent_module._agent_response_format_schema(
                {
                    "run": {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"name": "answer"},
                        }
                    }
                }
            )
        )
        self.assertEqual(
            agent_module._agent_response_format_schema(
                {
                    "run": {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "answer",
                                "schema": {"type": "object"},
                            },
                        }
                    }
                }
            ),
            {"type": "object"},
        )

    def test_simple_prompt_agent_reference_rejects_invalid_user_prompt(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
system = "Answer task inputs directly."
user = 1

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            issues = self._run_validate(runner, self._definition())

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "execution.unknown_target",
            ],
        )
        self.assertEqual(issues[0].path, "execution.ref")

    def test_invalid_agent_reference_returns_safe_diagnostic(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "secret-agent.toml"
            agent_path.parent.mkdir()
            agent_path.write_text("[agent]\n", encoding="utf-8")
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            issues = self._run_validate(runner, self._definition())

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "execution.unknown_target",
            ],
        )
        self.assertEqual(issues[0].path, "execution.ref")
        rendered = " ".join(
            value for issue in issues for value in issue.as_dict().values()
        )
        self.assertNotIn("secret-agent", rendered)
        self.assertNotIn("[agent]", rendered)

    def test_non_agent_target_returns_unsupported_issue(self) -> None:
        runner = AgentTaskTargetRunner(FakeLoader())

        issues = self._run_validate(
            runner,
            self._definition(
                execution=TaskExecutionTarget.model(
                    "ai://env:KEY@openai/gpt-4o-mini"
                )
            ),
        )

        self.assertEqual(
            [issue.code for issue in issues],
            [
                "execution.unknown_target",
            ],
        )
        self.assertEqual(issues[0].path, "execution.type")

    def test_file_input_rejects_agent_without_file_support(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://local/model"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(input_contract=TaskInputContract.file()),
            )

        self.assertEqual(
            [issue.code for issue in issues], ["input.invalid_file"]
        )
        self.assertEqual(issues[0].path, "input.file_conversions")
        rendered = " ".join(issues[0].as_dict().values())
        self.assertNotIn("local/model", rendered)

    def test_file_input_requires_conversion_for_text_only_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://env:KEY@bedrock/anthropic.claude"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",),
                    )
                ),
            )
            converted_issues = self._run_validate(
                runner,
                self._definition(
                    input_contract=TaskInputContract.file(
                        conversions=("markdown",),
                        mime_types=("application/pdf",),
                    )
                ),
            )

        self.assertEqual(
            [issue.code for issue in issues], ["input.invalid_file"]
        )
        self.assertEqual(issues[0].path, "input.file_conversions")
        self.assertEqual(converted_issues, ())

    def test_google_file_target_accepts_native_file_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://env:KEY@google/gemini-2.0-flash"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(input_contract=TaskInputContract.file()),
            )

        self.assertEqual(issues, ())

    def test_agent_uri_without_engine_returns_no_file_capability(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                '[agent]\nname = "Valid"\n', encoding="utf-8"
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            uri = runner._agent_uri(self._definition())
            profile = runner._agent_local_file_delivery_profile(
                self._definition()
            )

        self.assertIsNone(uri)
        self.assertEqual(profile, LocalFileDeliveryProfile.TEXT)

    def test_local_text_target_requires_compatible_text_delivery(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://local/model"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            pdf_issues = self._run_validate(
                runner,
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",),
                    )
                ),
            )
            text_issues = self._run_validate(
                runner,
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("text/plain",),
                    )
                ),
            )
            converted_issues = self._run_validate(
                runner,
                self._definition(
                    input_contract=TaskInputContract.file(
                        conversions=("text",),
                        mime_types=("application/pdf",),
                    )
                ),
            )

        self.assertEqual(
            [issue.code for issue in pdf_issues], ["input.invalid_file"]
        )
        self.assertEqual(pdf_issues[0].path, "input.file_conversions")
        self.assertEqual(text_issues, ())
        self.assertEqual(converted_issues, ())

    def test_local_multimodal_profile_accepts_supported_media(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://local/model"
file_delivery_profile = "multimodal"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            image_issues = self._run_validate(
                runner,
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("image/png",),
                    )
                ),
            )
            text_issues = self._run_validate(
                runner,
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("text/plain",),
                    )
                ),
            )
            pdf_issues = self._run_validate(
                runner,
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",),
                    )
                ),
            )

        self.assertEqual(image_issues, ())
        self.assertEqual(
            [issue.code for issue in text_issues], ["input.invalid_file"]
        )
        self.assertEqual(text_issues[0].path, "input.file_conversions")
        self.assertEqual(
            [issue.code for issue in pdf_issues], ["input.invalid_file"]
        )
        self.assertEqual(pdf_issues[0].path, "input.file_conversions")

    def test_invalid_file_delivery_profile_hint_is_safe(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://local/model"
file_delivery_profile = "binary"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(input_contract=TaskInputContract.file()),
            )

        self.assertEqual(
            [issue.code for issue in issues], ["execution.unknown_target"]
        )
        self.assertEqual(issues[0].path, "execution.ref")
        self.assertNotIn("binary", str(issues[0].as_dict()))

    def test_invalid_file_delivery_profile_hint_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://local/model"
file_delivery_profile = "binary"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            profile = runner._agent_local_file_delivery_profile(
                self._definition()
            )

        self.assertEqual(profile, LocalFileDeliveryProfile.TEXT)

    def test_unknown_agent_provider_rejects_file_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://env:KEY@unknown/model"
""",
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            issues = self._run_validate(
                runner,
                self._definition(input_contract=TaskInputContract.file()),
            )

        self.assertEqual(
            [issue.code for issue in issues], ["input.invalid_file"]
        )
        self.assertEqual(issues[0].path, "input.type")
        rendered = " ".join(issues[0].as_dict().values())
        self.assertNotIn("unknown/model", rendered)

    def _run_validate(
        self,
        runner: AgentTaskTargetRunner,
        definition: TaskDefinition,
    ) -> tuple[TaskValidationIssue, ...]:
        return asyncio_run(
            runner.validate_definition(
                definition,
                TaskValidationContext(),
            )
        )

    def _definition(
        self,
        *,
        input_contract: TaskInputContract | None = None,
        output_contract: TaskOutputContract | None = None,
        execution: TaskExecutionTarget | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="agent", version="1"),
            input=input_contract or TaskInputContract.string(),
            output=output_contract or TaskOutputContract.text(),
            execution=execution
            or TaskExecutionTarget.agent("agents/valid.toml"),
        )


class AgentTaskTargetRunnerTest(IsolatedAsyncioTestCase):
    async def test_run_uses_fresh_orchestrator_and_consumes_text(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                '[agent]\nname = "Valid"\n\n[engine]\nuri = "ai://x"\n',
                encoding="utf-8",
            )
            loader = FakeLoader(response_text="short summary")
            agent_id = uuid4()
            runner = AgentTaskTargetRunner(
                loader,
                agent_id=agent_id,
                disable_memory=True,
                ref_base=root,
                uri="ai://override",
            )

            output = await runner.run(
                self._context(
                    self._definition(
                        output=TaskOutputContract.text(),
                    ),
                    "private prompt",
                )
            )
            second_output = await runner.run(
                self._context(
                    self._definition(
                        output=TaskOutputContract.text(),
                    ),
                    "second prompt",
                )
            )

        self.assertEqual(output, "short summary")
        self.assertEqual(second_output, "short summary")
        self.assertEqual(loader.entered, 2)
        self.assertEqual(loader.exited, 2)
        self.assertEqual(loader.agent_ids, [agent_id, agent_id])
        self.assertEqual(loader.disable_memory_values, [True, True])
        self.assertEqual(loader.uris, ["ai://override", "ai://override"])
        self.assertEqual(loader.inputs, ["private prompt", "second prompt"])
        self.assertTrue(loader.paths[0].endswith("agents/valid.toml"))

    async def test_run_maps_json_input_and_structured_output(self) -> None:
        loader = FakeLoader(response_text='{"answer":"ok"}')
        runner = AgentTaskTargetRunner(loader)

        output = await runner.run(
            self._context(
                self._definition(
                    input_contract=TaskInputContract.object(
                        schema={"type": "object"}
                    ),
                    output=TaskOutputContract.object(
                        schema={"type": "object"}
                    ),
                ),
                {"question": "status"},
            )
        )

        self.assertEqual(output, {"answer": "ok"})
        self.assertEqual(loader.inputs, ['{"question":"status"}'])

    async def test_structured_output_rejects_provider_failure_safely(
        self,
    ) -> None:
        loader = FakeLoader(response=FailingJsonResponse("{}"))
        runner = AgentTaskTargetRunner(loader)

        with self.assertRaises(TaskProviderStructuredOutputError) as error:
            await runner.run(
                self._context(
                    self._definition(
                        output=TaskOutputContract.object(
                            schema={"type": "object"}
                        )
                    ),
                    "private prompt",
                )
            )

        self.assertNotIn("sk-test-secret", str(error.exception))
        self.assertEqual(loader.entered, 1)
        self.assertEqual(loader.exited, 1)

    async def test_structured_output_rejects_invalid_json_safely(
        self,
    ) -> None:
        cases = (
            TextOnlyResponse("{private: output"),
            NonStringJsonResponse("{}"),
        )
        for response in cases:
            with self.subTest(response=type(response).__name__):
                loader = FakeLoader(response=response)
                runner = AgentTaskTargetRunner(loader)

                with self.assertRaises(TaskOutputParseError) as error:
                    await runner.run(
                        self._context(
                            self._definition(
                                output=TaskOutputContract.object(
                                    schema={"type": "object"}
                                )
                            ),
                            "private prompt",
                        )
                    )

                self.assertNotIn("private", str(error.exception))
                self.assertEqual(loader.entered, 1)
                self.assertEqual(loader.exited, 1)

    async def test_run_attaches_cancellation_checker_to_response(
        self,
    ) -> None:
        checks = 0

        async def check_cancelled() -> None:
            nonlocal checks
            checks += 1

        response = CancellableResponse("short summary")
        loader = FakeLoader(response=response)
        runner = AgentTaskTargetRunner(loader)

        output = await runner.run(
            self._context(
                self._definition(output=TaskOutputContract.text()),
                "private prompt",
                cancellation_checker=check_cancelled,
            )
        )

        self.assertEqual(output, "short summary")
        self.assertIsNotNone(response.cancellation_checker)
        self.assertEqual(checks, 4)

    async def test_run_maps_agent_input_shapes(self) -> None:
        message = Message(role=MessageRole.USER, content="hello")
        loader = FakeLoader()
        runner = AgentTaskTargetRunner(loader)

        await runner.run(self._context(self._definition(), ["a", "b"]))
        await runner.run(self._context(self._definition(), [message]))
        await runner.run(self._context(self._definition(), (1, 2)))
        await runner.run(self._context(self._definition(), object()))

        self.assertEqual(loader.inputs[0], ["a", "b"])
        self.assertEqual(loader.inputs[1], [message])
        self.assertEqual(loader.inputs[2], "[1,2]")
        self.assertEqual(loader.inputs[3], "object")

    async def test_structured_output_can_fall_back_to_text_response(
        self,
    ) -> None:
        loader = FakeLoader(response=TextOnlyResponse('{"items":[1]}'))
        runner = AgentTaskTargetRunner(loader)

        output = await runner.run(
            self._context(
                self._definition(
                    output=TaskOutputContract.array(schema={"type": "array"})
                ),
                "private prompt",
            )
        )

        self.assertEqual(output, {"items": [1]})

    async def test_plain_output_without_text_method_returns_type_name(
        self,
    ) -> None:
        loader = FakeLoader(response=BareResponse())
        runner = AgentTaskTargetRunner(loader)

        output = await runner.run(
            self._context(
                self._definition(output=TaskOutputContract.text()),
                "private prompt",
            )
        )

        self.assertEqual(output, "BareResponse")

    async def test_plain_string_response_is_returned(self) -> None:
        loader = FakeLoader(response="plain response")
        runner = AgentTaskTargetRunner(loader)

        output = await runner.run(
            self._context(
                self._definition(output=TaskOutputContract.text()),
                "private prompt",
            )
        )

        self.assertEqual(output, "plain response")

    async def test_run_maps_provider_file_reference_at_execution_time(
        self,
    ) -> None:
        loader = FakeLoader(response="accepted")
        runner = AgentTaskTargetRunner(
            loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )

        output = await runner.run(
            TaskTargetContext(
                definition=self._definition(),
                execution=TaskExecutionContext(
                    run_id="run-1",
                    attempt_id="attempt-1",
                    attempt_number=1,
                ),
                input_value="summarize",
                files=(
                    TaskInputFile(
                        logical_path="provider:openai:provider_file_id",
                        media_type="application/pdf",
                        provider_reference=_provider_reference(
                            "openai",
                            "file-test",
                            media_type="application/pdf",
                        ),
                    ),
                ),
            )
        )

        self.assertEqual(output, "accepted")
        message = loader.inputs[0]
        self.assertIsInstance(message, Message)
        content = cast(Message, message).content
        self.assertIsInstance(content, list)
        blocks = cast(list[Any], content)
        self.assertEqual(blocks[0].type, "text")
        self.assertEqual(blocks[1].type, "file")
        file_block = blocks[1]
        self.assertEqual(file_block.file["file_id"], "file-test")
        self.assertEqual(file_block.file["mime_type"], "application/pdf")

    async def test_run_maps_typed_provider_file_reference(self) -> None:
        loader = FakeLoader(response="accepted")
        runner = AgentTaskTargetRunner(
            loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )

        await runner.run(
            self._context(
                self._definition(),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="provider:openai:provider_file_id",
                        media_type="application/pdf",
                        provider_reference=(
                            TaskFileDescriptor.provider_reference_descriptor(
                                "file-test",
                                kind=(
                                    TaskProviderReferenceKind.PROVIDER_FILE_ID
                                ),
                                provider="openai",
                                mime_type="application/pdf",
                                owner_scope="tenant-a",
                                identity_hmac="hmac-value",
                            ).provider_reference
                        ),
                    ),
                ),
            )
        )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(content[1].file["file_id"], "file-test")
        self.assertNotIn("tenant-a", str(content[1].file))

    async def test_run_prefixes_file_input_with_agent_user_prompt(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
system = "Keep system separate."
user = "Extract {{ files[0].role }}."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader(response="accepted")
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            await runner.run(
                self._context(
                    self._definition(
                        input_contract=TaskInputContract.file(
                            mime_types=("application/pdf",),
                        )
                    ),
                    "ignored prompt",
                    files=(
                        TaskInputFile(
                            logical_path="/private/uploads/sentinel.pdf",
                            media_type="application/pdf",
                            size_bytes=512,
                            metadata={
                                "display_name": "sentinel.pdf",
                                "role": "primary",
                            },
                            provider_reference=_provider_reference(
                                "openai",
                                "file-test",
                                media_type="application/pdf",
                            ),
                        ),
                    ),
                )
            )

        message = cast(Message, loader.inputs[0])
        content = cast(list[Any], message.content)
        self.assertEqual(message.role, MessageRole.USER)
        self.assertEqual(
            content[0].text,
            "Extract primary.",
        )
        self.assertEqual(content[1].file["file_id"], "file-test")
        self.assertNotIn("Keep system separate", str(message))
        self.assertNotIn("ignored prompt", str(message))
        self.assertNotIn("sentinel.pdf", str(message))
        self.assertNotIn("/private/uploads", str(message))

    async def test_run_renders_file_array_user_template_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            template_path = root / "agents" / "user.md"
            agent_path.parent.mkdir()
            template_path.write_text(
                "{% for file in files %}"
                "{{ file.index }}:{{ file.mime_type }}:"
                "{{ file.size_bucket }}:{{ file.identity_hmac }};"
                "{% endfor %}",
                encoding="utf-8",
            )
            agent_path.write_text(
                """
[agent]
name = "Valid"
user_template = "user.md"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            first = TaskFileDescriptor.provider_reference_descriptor(
                "file-first",
                kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                provider="openai",
                mime_type="application/pdf",
                identity_hmac="hmac-first",
            ).provider_reference
            second = TaskFileDescriptor.provider_reference_descriptor(
                "file-second",
                kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                provider="openai",
                mime_type="application/pdf",
                identity_hmac="hmac-second",
            ).provider_reference
            assert first is not None
            assert second is not None
            loader = FakeLoader(response="accepted")
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            await runner.run(
                self._context(
                    self._definition(
                        input_contract=TaskInputContract.file_array(
                            mime_types=("application/pdf",),
                        )
                    ),
                    "ignored prompt",
                    files=(
                        TaskInputFile(
                            logical_path="provider:openai:one",
                            media_type="application/pdf",
                            provider_reference=first,
                            size_bytes=0,
                        ),
                        TaskInputFile(
                            logical_path="provider:openai:two",
                            media_type="application/pdf",
                            provider_reference=second,
                            size_bytes=2_048,
                        ),
                    ),
                )
            )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(
            content[0].text,
            "0:application/pdf:0B:hmac-first;"
            "1:application/pdf:1KB-1MB:hmac-second;",
        )
        self.assertEqual(content[1].file["file_id"], "file-first")
        self.assertEqual(content[2].file["file_id"], "file-second")
        self.assertNotIn("provider:openai", str(loader.inputs[0]))

    async def test_run_prefixes_text_input_before_file_blocks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
user = "Review the attachment."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader(response="accepted")
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            await runner.run(
                self._context(
                    self._definition(),
                    "include this detail",
                    files=(
                        TaskInputFile(
                            logical_path="provider:openai:provider_file_id",
                            provider_reference=_provider_reference(
                                "openai",
                                "file-test",
                            ),
                        ),
                    ),
                )
            )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(
            content[0].text,
            "Review the attachment.\n\ninclude this detail",
        )
        self.assertEqual(content[1].file["file_id"], "file-test")

    async def test_run_prefixes_message_text_before_file_blocks(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
user = "Review the attachment."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader(response="accepted")
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            await runner.run(
                self._context(
                    self._definition(),
                    Message(
                        role=MessageRole.USER,
                        content=MessageContentText(
                            type="text",
                            text="include this detail",
                        ),
                    ),
                    files=(
                        TaskInputFile(
                            logical_path="/private/uploads/sentinel.pdf",
                            metadata={"display_name": "sentinel.pdf"},
                            provider_reference=_provider_reference(
                                "openai",
                                "file-test",
                            ),
                        ),
                    ),
                )
            )

        message = cast(Message, loader.inputs[0])
        content = cast(list[Any], message.content)
        self.assertEqual(
            content[0].text,
            "Review the attachment.\n\ninclude this detail",
        )
        self.assertEqual(content[1].file["file_id"], "file-test")
        self.assertNotIn("sentinel.pdf", str(message))
        self.assertNotIn("/private/uploads", str(message))

    async def test_run_prefixes_last_message_text_before_file_blocks(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
user = "Use the final instruction."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader(response="accepted")
            runner = AgentTaskTargetRunner(loader, ref_base=root)
            messages = [
                Message(role=MessageRole.USER, content="previous"),
                Message(role=MessageRole.ASSISTANT, content="ok"),
                Message(role=MessageRole.USER, content="final detail"),
            ]

            await runner.run(
                self._context(
                    self._definition(),
                    messages,
                    files=(
                        TaskInputFile(
                            logical_path="provider:openai:provider_file_id",
                            provider_reference=_provider_reference(
                                "openai",
                                "file-test",
                            ),
                        ),
                    ),
                )
            )

        captured = cast(list[Message], loader.inputs[0])
        content = cast(list[Any], captured[2].content)
        self.assertEqual(captured[0].content, "previous")
        self.assertEqual(captured[1].content, "ok")
        self.assertEqual(
            content[0].text,
            "Use the final instruction.\n\nfinal detail",
        )
        self.assertEqual(content[1].file["file_id"], "file-test")

    async def test_run_rejects_message_prompt_template_variable_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
user = "Extract {{ filename }}."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader()
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    self._context(
                        self._definition(),
                        Message(
                            role=MessageRole.USER,
                            content="private detail",
                        ),
                        files=(
                            TaskInputFile(
                                logical_path="/private/uploads/sentinel.pdf",
                                metadata={"display_name": "sentinel.pdf"},
                                provider_reference=_provider_reference(
                                    "openai",
                                    "file-test",
                                ),
                            ),
                        ),
                    )
                )

        self.assertEqual(
            error.exception.issues[0].code, "input.invalid_prompt"
        )
        self.assertEqual(loader.inputs, [])
        self.assertNotIn("private detail", str(error.exception))
        self.assertNotIn("sentinel.pdf", str(error.exception))
        self.assertNotIn("/private/uploads", str(error.exception))

    async def test_run_preserves_mixed_message_content_with_prompt(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
user = "Review safely."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            file_loader = FakeLoader(response="accepted")
            text_loader = FakeLoader(response="accepted")
            insert_loader = FakeLoader(response="accepted")
            file_runner = AgentTaskTargetRunner(file_loader, ref_base=root)
            text_runner = AgentTaskTargetRunner(text_loader, ref_base=root)
            insert_runner = AgentTaskTargetRunner(insert_loader, ref_base=root)
            existing_file = MessageContentFile(
                type="file",
                file={"file_id": "existing-file"},
            )
            new_file = TaskInputFile(
                logical_path="provider:openai:provider_file_id",
                provider_reference=_provider_reference("openai", "file-test"),
            )

            await file_runner.run(
                self._context(
                    self._definition(
                        input_contract=TaskInputContract.file(
                            mime_types=("application/pdf",),
                        )
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContentText(
                                type="text",
                                text="ignored input",
                            ),
                            existing_file,
                        ],
                    ),
                    files=(new_file,),
                )
            )
            await text_runner.run(
                self._context(
                    self._definition(),
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContentText(
                                type="text",
                                text="keep this",
                            ),
                            existing_file,
                        ],
                    ),
                    files=(new_file,),
                )
            )
            await insert_runner.run(
                self._context(
                    self._definition(),
                    Message(
                        role=MessageRole.USER,
                        content=existing_file,
                    ),
                    files=(new_file,),
                )
            )

        file_content = cast(
            list[Any], cast(Message, file_loader.inputs[0]).content
        )
        text_content = cast(
            list[Any], cast(Message, text_loader.inputs[0]).content
        )
        insert_content = cast(
            list[Any], cast(Message, insert_loader.inputs[0]).content
        )
        self.assertEqual(file_content[0].text, "Review safely.")
        self.assertEqual(file_content[1].file["file_id"], "existing-file")
        self.assertEqual(file_content[2].file["file_id"], "file-test")
        self.assertNotIn("ignored input", str(file_loader.inputs[0]))
        self.assertEqual(
            text_content[0].text,
            "Review safely.\n\nkeep this",
        )
        self.assertEqual(text_content[1].file["file_id"], "existing-file")
        self.assertEqual(text_content[2].file["file_id"], "file-test")
        self.assertEqual(insert_content[0].text, "Review safely.")
        self.assertEqual(insert_content[1].file["file_id"], "existing-file")
        self.assertEqual(insert_content[2].file["file_id"], "file-test")

    async def test_run_uses_agent_user_input_reference_once(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
user = "Prompt: {{ input }}"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader(response="accepted")
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            await runner.run(
                self._context(
                    self._definition(),
                    "include this detail",
                    files=(
                        TaskInputFile(
                            logical_path="provider:openai:provider_file_id",
                            provider_reference=_provider_reference(
                                "openai",
                                "file-test",
                            ),
                        ),
                    ),
                )
            )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(content[0].text, "Prompt: include this detail")

    async def test_run_prefixes_template_prompt_before_text_input(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            template_path = root / "agents" / "user.md"
            agent_path.parent.mkdir()
            template_path.write_text("Template {{ input }}", encoding="utf-8")
            agent_path.write_text(
                """
[agent]
name = "Valid"
user_template = "user.md"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader(response="accepted")
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            await runner.run(
                self._context(
                    self._definition(),
                    "include this detail",
                    files=(
                        TaskInputFile(
                            logical_path="provider:openai:provider_file_id",
                            provider_reference=_provider_reference(
                                "openai",
                                "file-test",
                            ),
                        ),
                    ),
                )
            )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(
            content[0].text,
            "Template include this detail\n\ninclude this detail",
        )

    async def test_run_rejects_missing_prompt_template_variable_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
user = "Extract {{ filename }}."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader()
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    self._context(
                        self._definition(
                            input_contract=TaskInputContract.file(
                                mime_types=("application/pdf",),
                            )
                        ),
                        "ignored prompt",
                        files=(
                            TaskInputFile(
                                logical_path="/private/uploads/sentinel.pdf",
                                media_type="application/pdf",
                                metadata={"display_name": "sentinel.pdf"},
                                provider_reference=_provider_reference(
                                    "openai",
                                    "file-test",
                                    media_type="application/pdf",
                                ),
                            ),
                        ),
                    )
                )

        self.assertEqual(
            error.exception.issues[0].code, "input.invalid_prompt"
        )
        self.assertEqual(error.exception.issues[0].path, "execution.ref")
        self.assertEqual(loader.inputs, [])
        self.assertNotIn("sentinel.pdf", str(error.exception))
        self.assertNotIn("/private/uploads", str(error.exception))

    async def test_run_rejects_missing_user_template_safely(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
user_template = "missing.md"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader()
            runner = AgentTaskTargetRunner(loader, ref_base=root)

            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    self._context(
                        self._definition(
                            input_contract=TaskInputContract.file(
                                mime_types=("application/pdf",),
                            )
                        ),
                        "ignored prompt",
                        files=(
                            TaskInputFile(
                                logical_path="/private/uploads/sentinel.pdf",
                                media_type="application/pdf",
                                provider_reference=_provider_reference(
                                    "openai",
                                    "file-test",
                                    media_type="application/pdf",
                                ),
                            ),
                        ),
                    )
                )

        self.assertEqual(
            error.exception.issues[0].code, "input.invalid_prompt"
        )
        self.assertEqual(loader.inputs, [])
        self.assertNotIn("missing.md", str(error.exception))
        self.assertNotIn("sentinel.pdf", str(error.exception))

    async def test_run_rejects_legacy_provider_reference_metadata(
        self,
    ) -> None:
        for metadata in (
            {"provider_file_id": "file-private"},
            {
                "provider_reference": {
                    "kind": "provider_file_id",
                    "provider": "openai",
                    "reference": "file-private",
                }
            },
        ):
            with self.subTest(metadata=metadata):
                loader = FakeLoader(response="accepted")
                runner = AgentTaskTargetRunner(
                    loader,
                    uri="ai://env:KEY@openai/gpt-4o-mini",
                )

                with self.assertRaises(TaskValidationError) as error:
                    await runner.run(
                        self._context(
                            self._definition(),
                            "summarize",
                            files=(
                                TaskInputFile(
                                    logical_path="provider:file",
                                    media_type="application/pdf",
                                    metadata=metadata,
                                ),
                            ),
                        ),
                    )

                self.assertEqual(
                    error.exception.issues[0].code,
                    "input.invalid_file",
                )
                self.assertEqual(
                    error.exception.issues[0].path,
                    "input.files[0]",
                )
                self.assertNotIn("file-private", str(error.exception))
                self.assertEqual(loader.inputs, [])

    async def test_run_rejects_expired_provider_reference(self) -> None:
        runner = AgentTaskTargetRunner(
            FakeLoader(),
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="provider:openai:handle",
                            provider_reference=(
                                TaskFileDescriptor.provider_reference_descriptor(
                                    "https://example.test/private",
                                    kind=(
                                        TaskProviderReferenceKind.EXPIRING_PROVIDER_HANDLE
                                    ),
                                    provider="openai",
                                    expires_at=(
                                        datetime.now(UTC)
                                        - timedelta(seconds=1)
                                    ),
                                    durable=False,
                                ).provider_reference
                            ),
                        ),
                    ),
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")

    async def test_run_maps_provider_urls_at_execution_time(self) -> None:
        url_loader = FakeLoader(response="accepted")
        uri_loader = FakeLoader(response="accepted")
        s3_loader = FakeLoader(response="accepted")

        await AgentTaskTargetRunner(
            url_loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        ).run(
            self._context(
                self._definition(),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="provider:openai:hosted_url",
                        media_type="application/pdf",
                        provider_reference=_provider_reference(
                            "openai",
                            "https://example.test/file",
                            kind=TaskProviderReferenceKind.HOSTED_URL,
                            media_type="application/pdf",
                        ),
                    ),
                ),
            )
        )
        await AgentTaskTargetRunner(
            uri_loader,
            uri="ai://env:KEY@google/gemini-2.0-flash",
        ).run(
            self._context(
                self._definition(),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="provider:google:object_store_uri",
                        media_type="application/pdf",
                        provider_reference=_provider_reference(
                            "google",
                            "gs://bucket/object",
                            kind=(TaskProviderReferenceKind.OBJECT_STORE_URI),
                            media_type="application/pdf",
                        ),
                    ),
                ),
            )
        )
        await AgentTaskTargetRunner(
            s3_loader,
            uri="ai://env:KEY@bedrock/us.anthropic.claude",
        ).run(
            self._context(
                self._definition(),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="provider:bedrock:object_store_uri",
                        media_type="text/plain",
                        provider_reference=_provider_reference(
                            "bedrock",
                            "s3://bucket/object",
                            kind=(TaskProviderReferenceKind.OBJECT_STORE_URI),
                            media_type="text/plain",
                        ),
                    ),
                ),
            )
        )

        url_file = cast(Message, url_loader.inputs[0]).content
        uri_file = cast(Message, uri_loader.inputs[0]).content
        s3_file = cast(Message, s3_loader.inputs[0]).content
        self.assertEqual(
            cast(list[Any], url_file)[1].file["file_url"],
            "https://example.test/file",
        )
        self.assertEqual(
            cast(list[Any], uri_file)[1].file["file_url"],
            "gs://bucket/object",
        )
        self.assertEqual(
            cast(list[Any], s3_file)[1].file["file_url"],
            "s3://bucket/object",
        )

    async def test_run_uses_injected_file_delivery_profile(self) -> None:
        loader = FakeLoader(response="accepted")
        seen_uris: list[str | None] = []
        profile = FileDeliveryProfile(
            name="id_only",
            delivery_modes=frozenset({FileDeliveryMode.PROVIDER_FILE_ID}),
        )

        def resolver(uri: str | None) -> FileDeliveryProfile:
            seen_uris.append(uri)
            return profile

        runner = AgentTaskTargetRunner(
            loader,
            file_delivery_resolver=resolver,
            uri="ai://env:KEY@unknown/model",
        )

        await runner.run(
            self._context(
                self._definition(),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="provider:id_only:provider_file_id",
                        provider_reference=_provider_reference(
                            "id_only",
                            "file-test",
                        ),
                    ),
                ),
            )
        )

        message = cast(Message, loader.inputs[0])
        self.assertEqual(
            cast(list[Any], message.content)[1].file["file_id"],
            "file-test",
        )
        self.assertEqual(seen_uris, ["ai://env:KEY@unknown/model"])

    async def test_run_maps_local_text_artifact_to_text_block(self) -> None:
        loader = FakeLoader(response="accepted")
        tokenized_texts: list[str] = []
        runner = AgentTaskTargetRunner(
            loader,
            token_counter=lambda text: (
                tokenized_texts.append(text) or len(text.split())
            ),
            uri="ai://local/model",
        )
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        await runner.run(
            self._context(
                self._definition(
                    limits=TaskLimitsPolicy(total_tokens=3),
                ),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="artifact:artifact-1",
                        artifact_ref=artifact_ref,
                        media_type="text/plain",
                        size_bytes=12,
                    ),
                ),
                artifact_store=FakeArtifactStore(b"private text"),
            )
        )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(content[0].type, "text")
        self.assertEqual(content[0].text, "summarize")
        self.assertEqual(content[1].type, "text")
        self.assertEqual(content[1].text, "private text")
        self.assertEqual(tokenized_texts, ["summarize", "private text"])

    async def test_run_maps_explicit_local_multimodal_image_to_file_block(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://local/model"
file_delivery_profile = "multimodal"
""",
                encoding="utf-8",
            )
            loader = FakeLoader(response="accepted")
            runner = AgentTaskTargetRunner(loader, ref_base=root)
            artifact_ref = TaskArtifactRef(
                artifact_id="artifact-1",
                store="local",
                storage_key="ar/artifact-1",
            )

            await runner.run(
                self._context(
                    self._definition(),
                    "describe",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="image/png",
                            size_bytes=4,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(b"\x89PNG"),
                )
            )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(content[0].type, "text")
        self.assertEqual(content[0].text, "describe")
        self.assertEqual(content[1].type, "file")
        self.assertEqual(content[1].file["file_data"], "iVBORw==")
        self.assertEqual(content[1].file["filename"], "task-file.png")
        self.assertEqual(content[1].file["mime_type"], "image/png")

    async def test_run_reads_inline_bytes_with_bounded_raw_limit(
        self,
    ) -> None:
        loader = FakeLoader(response="accepted")
        profile = FileDeliveryProfile(
            name="inline-bytes",
            delivery_modes=frozenset({FileDeliveryMode.INLINE_BYTES}),
            inline_byte_limit=FileDeliveryLimit(
                name="inline_file_bytes",
                source="test",
                max_bytes=8,
            ),
        )
        runner = AgentTaskTargetRunner(
            loader,
            file_delivery_resolver=lambda uri: profile,
            uri="ai://env:KEY@fake/model",
        )
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )
        artifact_store = FakeArtifactStore(b"abcdef")

        await runner.run(
            self._context(
                self._definition(),
                "describe",
                files=(
                    TaskInputFile(
                        logical_path="artifact:artifact-1",
                        artifact_ref=artifact_ref,
                        media_type="application/octet-stream",
                        size_bytes=6,
                    ),
                ),
                artifact_store=artifact_store,
            )
        )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(content[1].file["file_data"], "YWJjZGVm")
        self.assertEqual(artifact_store.max_bytes_values, [6])

    async def test_run_rejects_inline_bytes_when_bounded_read_exceeds_limit(
        self,
    ) -> None:
        loader = FakeLoader()
        profile = FileDeliveryProfile(
            name="inline-bytes",
            delivery_modes=frozenset({FileDeliveryMode.INLINE_BYTES}),
            inline_byte_limit=FileDeliveryLimit(
                name="inline_file_bytes",
                source="test",
                max_bytes=8,
            ),
        )
        runner = AgentTaskTargetRunner(
            loader,
            file_delivery_resolver=lambda uri: profile,
            uri="ai://env:KEY@fake/model",
        )
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )
        artifact_store = FakeArtifactStore(b"abcdefg")

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(),
                    "describe",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="application/octet-stream",
                            size_bytes=6,
                        ),
                    ),
                    artifact_store=artifact_store,
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")
        self.assertEqual(artifact_store.max_bytes_values, [6])
        self.assertEqual(loader.inputs, [])
        self.assertNotIn("private artifact bytes", str(error.exception))

    async def test_run_reads_inline_text_with_task_limit_cap(self) -> None:
        loader = FakeLoader(response="accepted")
        profile = FileDeliveryProfile(
            name="inline-text",
            delivery_modes=frozenset({FileDeliveryMode.INLINE_TEXT}),
            inline_text_limit=FileDeliveryLimit(
                name="inline_text_bytes",
                source="test",
                max_bytes=12,
            ),
        )
        runner = AgentTaskTargetRunner(
            loader,
            file_delivery_resolver=lambda uri: profile,
            uri="ai://env:KEY@fake/model",
        )
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )
        artifact_store = FakeArtifactStore(b"hello")

        await runner.run(
            self._context(
                self._definition(limits=TaskLimitsPolicy(file_bytes=5)),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="artifact:artifact-1",
                        artifact_ref=artifact_ref,
                        media_type="text/plain",
                        size_bytes=5,
                    ),
                ),
                artifact_store=artifact_store,
            )
        )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(content[1].text, "hello")
        self.assertEqual(artifact_store.max_bytes_values, [5])

    async def test_run_rejects_local_native_media_without_profile(
        self,
    ) -> None:
        loader = FakeLoader()
        runner = AgentTaskTargetRunner(loader, uri="ai://local/model")
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(),
                    "describe",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="image/png",
                            size_bytes=4,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(b"\x89PNG"),
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")
        self.assertEqual(loader.inputs, [])

    async def test_run_rejects_local_text_when_prompt_exhausts_budget(
        self,
    ) -> None:
        loader = FakeLoader()
        runner = AgentTaskTargetRunner(loader, uri="ai://local/model")
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(
                        limits=TaskLimitsPolicy(total_tokens=1),
                    ),
                    "summarize now",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="text/plain",
                            size_bytes=18,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(b"one two three four"),
                )
            )

        self.assertEqual(
            error.exception.issues[0].code, "limits.invalid_value"
        )
        self.assertEqual(error.exception.issues[0].path, "limits.total_tokens")
        self.assertNotIn("one two three four", str(error.exception))
        self.assertEqual(loader.inputs, [])

    async def test_run_retrieves_matching_local_text_chunks(self) -> None:
        loader = FakeLoader(response="accepted")
        runner = AgentTaskTargetRunner(loader, uri="ai://local/model")
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        await runner.run(
            self._context(
                self._definition(limits=TaskLimitsPolicy(total_tokens=5)),
                "needle",
                files=(
                    TaskInputFile(
                        logical_path="artifact:artifact-1",
                        artifact_ref=artifact_ref,
                        media_type="text/plain",
                        size_bytes=64,
                    ),
                ),
                artifact_store=FakeArtifactStore(
                    b"zero one two three needle five six seven eight nine"
                ),
            )
        )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(
            [block.text for block in content],
            ["needle", "needle five six seven"],
        )
        self.assertNotIn("zero one two three", str(loader.inputs[0]))
        self.assertTrue(
            all(isinstance(block, MessageContentText) for block in content)
        )

    async def test_run_plans_file_input_without_prompt_text(self) -> None:
        loader = FakeLoader(response="accepted")
        runner = AgentTaskTargetRunner(loader, uri="ai://local/model")
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        await runner.run(
            self._context(
                self._definition(
                    input_contract=TaskInputContract.file(
                        conversions=("text",),
                        mime_types=("text/plain",),
                    ),
                    limits=TaskLimitsPolicy(total_tokens=3),
                ),
                "ignored file prompt",
                files=(
                    TaskInputFile(
                        logical_path="artifact:artifact-1",
                        artifact_ref=artifact_ref,
                        media_type="text/plain",
                        size_bytes=11,
                    ),
                ),
                artifact_store=FakeArtifactStore(b"alpha beta"),
            )
        )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual([block.text for block in content], ["alpha beta"])

    async def test_run_uses_map_reduce_for_unmatched_local_text(
        self,
    ) -> None:
        responses = (
            FakeResponse("one", input_token_count=3),
            FakeResponse("two", input_token_count=3),
            FakeResponse(
                "final summary",
                input_token_count=4,
                output_token_count=2,
            ),
        )
        loader = FakeLoader(responses=responses)
        observed: list[object] = []
        runner = AgentTaskTargetRunner(loader, uri="ai://local/model")
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        result = await runner.run(
            self._context(
                self._definition(limits=TaskLimitsPolicy(total_tokens=4)),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="artifact:artifact-1",
                        artifact_ref=artifact_ref,
                        media_type="text/plain",
                        size_bytes=36,
                    ),
                ),
                artifact_store=FakeArtifactStore(
                    b"alpha beta gamma delta epsilon zeta"
                ),
                usage_observer=lambda response: observed.append(response),
            )
        )

        self.assertEqual(result, "final summary")
        self.assertEqual(len(loader.inputs), 3)
        self.assertEqual(
            [
                [
                    block.text
                    for block in cast(
                        list[Any],
                        cast(Message, input_value).content,
                    )
                ]
                for input_value in loader.inputs
            ],
            [
                ["summarize", "alpha beta gamma"],
                ["summarize", "delta epsilon zeta"],
                ["summarize", "one", "two"],
            ],
        )
        self.assertEqual(observed, list(responses))

    async def test_run_checks_cancellation_between_map_calls(self) -> None:
        loader = FakeLoader(
            responses=(
                FakeResponse("map one"),
                FakeResponse("map two"),
                FakeResponse("final summary"),
            )
        )
        calls = 0
        runner = AgentTaskTargetRunner(loader, uri="ai://local/model")
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        async def check_cancelled() -> None:
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise CancelledError()

        with self.assertRaises(CancelledError):
            await runner.run(
                self._context(
                    self._definition(limits=TaskLimitsPolicy(total_tokens=4)),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="text/plain",
                            size_bytes=36,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(
                        b"alpha beta gamma delta epsilon zeta"
                    ),
                    cancellation_checker=check_cancelled,
                )
            )

        self.assertEqual(len(loader.inputs), 1)

    async def test_run_raises_invalid_reduce_output_safely(self) -> None:
        loader = FakeLoader(
            responses=(
                FakeResponse('{"partial": 1}'),
                FakeResponse('{"partial": 2}'),
                FakeResponse("private invalid json"),
            )
        )
        runner = AgentTaskTargetRunner(loader, uri="ai://local/model")
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        with self.assertRaises(ValueError) as error:
            await runner.run(
                self._context(
                    self._definition(
                        output=TaskOutputContract.json(),
                        limits=TaskLimitsPolicy(total_tokens=4),
                    ),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="text/plain",
                            size_bytes=36,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(
                        b"alpha beta gamma delta epsilon zeta"
                    ),
                )
            )

        self.assertNotIn("alpha beta gamma", str(error.exception))

    async def test_run_rejects_reduce_input_over_token_limit_safely(
        self,
    ) -> None:
        responses = (
            FakeResponse("private map one two three"),
            FakeResponse("private map four five six"),
            FakeResponse("final summary"),
        )
        loader = FakeLoader(responses=responses)
        observed: list[object] = []
        runner = AgentTaskTargetRunner(loader, uri="ai://local/model")
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(limits=TaskLimitsPolicy(total_tokens=4)),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="text/plain",
                            size_bytes=36,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(
                        b"alpha beta gamma delta epsilon zeta"
                    ),
                    usage_observer=lambda response: observed.append(response),
                )
            )

        self.assertEqual(len(loader.inputs), 2)
        self.assertEqual(observed, list(responses[:2]))
        self.assertEqual(error.exception.issues[0].path, "limits.total_tokens")
        self.assertNotIn("alpha beta gamma", str(error.exception))
        self.assertNotIn("private map", str(error.exception))

    async def test_direct_runner_retries_transient_map_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"
goal_instructions = "Be brief."

[engine]
uri = "ai://local/model"
""",
                encoding="utf-8",
            )
            loader = FakeLoader(
                responses=(
                    RuntimeError("private transient failure"),
                    FakeResponse("one"),
                    FakeResponse("two"),
                    FakeResponse("final summary"),
                )
            )
            runner = DirectTaskRunner(
                InMemoryTaskStore(),
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                artifact_store=FakeArtifactStore(
                    b"alpha beta gamma delta epsilon zeta"
                ),
                definition_hash=lambda task: "agent-map-retry-hash",
            )
            artifact_ref = TaskArtifactRef(
                artifact_id="artifact-1",
                store="local",
                storage_key="ar/artifact-1",
            )

            result = await runner.run(
                self._definition(
                    limits=TaskLimitsPolicy(total_tokens=4),
                    retry=TaskRetryPolicy(max_attempts=2),
                ),
                input_value="summarize",
                files=(
                    TaskInputFile(
                        logical_path="artifact:artifact-1",
                        artifact_ref=artifact_ref,
                        media_type="text/plain",
                        size_bytes=36,
                    ),
                ),
            )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "final summary")
        self.assertEqual(len(loader.inputs), 4)

    async def test_run_rejects_unsupported_provider_uri_scheme(self) -> None:
        runner = AgentTaskTargetRunner(
            FakeLoader(),
            uri="ai://env:KEY@bedrock/us.anthropic.claude",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="provider:bedrock:object_store_uri",
                            media_type="text/plain",
                            provider_reference=_provider_reference(
                                "bedrock",
                                "gs://bucket/object",
                                kind=(
                                    TaskProviderReferenceKind.OBJECT_STORE_URI
                                ),
                                media_type="text/plain",
                            ),
                        ),
                    ),
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")

    async def test_run_rejects_invalid_inline_text_bytes_safely(self) -> None:
        runner = AgentTaskTargetRunner(
            FakeLoader(),
            uri="ai://local/model",
        )
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="text/plain",
                            size_bytes=1,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(b"\xff"),
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")
        self.assertNotIn(
            "artifact-1", str(error.exception.issues[0].as_dict())
        )

    async def test_run_rejects_invalid_strategy_text_bytes_safely(
        self,
    ) -> None:
        runner = AgentTaskTargetRunner(
            FakeLoader(),
            uri="ai://local/model",
        )
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(
                        input_contract=TaskInputContract.file(
                            conversions=("text",),
                            mime_types=("application/pdf",),
                        )
                    ),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                            media_type="application/pdf",
                            size_bytes=1,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(b"\xff"),
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")
        self.assertNotIn(
            "artifact-1", str(error.exception.issues[0].as_dict())
        )

    async def test_run_maps_strategy_text_artifact_to_text_block(self) -> None:
        loader = FakeLoader(response="accepted")
        tokenized_texts: list[str] = []
        runner = AgentTaskTargetRunner(
            loader,
            token_counter=lambda text: (
                tokenized_texts.append(text) or len(text.split())
            ),
            uri="ai://local/model",
        )
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        await runner.run(
            self._context(
                self._definition(
                    input_contract=TaskInputContract.file(
                        conversions=("text",),
                        mime_types=("application/pdf",),
                    )
                ),
                "summarize",
                files=(
                    TaskInputFile(
                        logical_path="artifact:artifact-1",
                        artifact_ref=artifact_ref,
                        media_type="application/pdf",
                        size_bytes=12,
                    ),
                ),
                artifact_store=FakeArtifactStore(b"converted text"),
            )
        )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual([block.text for block in content], ["converted text"])
        self.assertEqual(tokenized_texts, ["converted text"])

    async def test_run_rejects_local_multimodal_text_only_file_block(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://local/model"
file_delivery_profile = "multimodal"
""",
                encoding="utf-8",
            )
            loader = FakeLoader()
            runner = AgentTaskTargetRunner(loader, ref_base=root)
            artifact_ref = TaskArtifactRef(
                artifact_id="artifact-1",
                store="local",
                storage_key="ar/artifact-1",
            )

            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    self._context(
                        self._definition(),
                        "describe",
                        files=(
                            TaskInputFile(
                                logical_path="artifact:artifact-1",
                                artifact_ref=artifact_ref,
                                media_type="text/plain",
                                size_bytes=12,
                            ),
                        ),
                        artifact_store=FakeArtifactStore(b"private text"),
                    )
                )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")
        self.assertNotIn("private text", str(error.exception))
        self.assertEqual(loader.inputs, [])

    async def test_run_rejects_profile_file_count_limit(self) -> None:
        profile = FileDeliveryProfile(
            name="single",
            delivery_modes=frozenset({FileDeliveryMode.PROVIDER_FILE_ID}),
            file_count_limit=FileDeliveryLimit(
                name="files",
                source="test",
                max_count=1,
            ),
        )
        loader = FakeLoader()
        runner = AgentTaskTargetRunner(
            loader,
            file_delivery_resolver=lambda uri: profile,
            uri="ai://env:KEY@fake/model",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                self._context(
                    self._definition(),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="provider:single:provider_file_id",
                            provider_reference=_provider_reference(
                                "single",
                                "file-test",
                            ),
                        ),
                        TaskInputFile(
                            logical_path="provider:single:provider_file_id",
                            provider_reference=_provider_reference(
                                "single",
                                "file-test-2",
                            ),
                        ),
                    ),
                )
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files")
        self.assertEqual(loader.paths, [])

    async def test_artifact_bytes_rejects_missing_artifact_backend(
        self,
    ) -> None:
        with self.assertRaises(TaskValidationError) as error:
            await agent_module._artifact_bytes(
                TaskInputFile(logical_path="artifact:missing"),
                context=self._context(self._definition(), "summarize"),
                path="input.files[0]",
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")

    async def test_artifact_bytes_rejects_read_failure_safely(self) -> None:
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )

        with self.assertRaises(TaskValidationError) as error:
            await agent_module._artifact_bytes(
                TaskInputFile(
                    logical_path="artifact:artifact-1",
                    artifact_ref=artifact_ref,
                ),
                context=self._context(
                    self._definition(),
                    "summarize",
                    artifact_store=FailingReadArtifactStore(),
                ),
                path="input.files[0]",
                max_bytes=5,
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files[0]")
        self.assertNotIn("private read failure", str(error.exception))

    def test_artifact_read_max_bytes_returns_none_without_limits(self) -> None:
        value = agent_module._artifact_read_max_bytes(
            self._definition(),
            decision=FileDeliveryDecision(
                mode=FileDeliveryMode.PROVIDER_FILE_ID,
                reference="file-test",
            ),
            profile=FileDeliveryProfile(
                name="id-only",
                delivery_modes=frozenset({FileDeliveryMode.PROVIDER_FILE_ID}),
            ),
        )

        self.assertIsNone(value)

    def test_decision_reference_rejects_missing_reference(self) -> None:
        with self.assertRaises(TaskValidationError) as error:
            agent_module._decision_reference(
                FileDeliveryDecision(mode=FileDeliveryMode.PROVIDER_FILE_ID)
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(error.exception.issues[0].path, "input.files")

    def test_text_block_helpers_cover_input_shapes(self) -> None:
        text_message = Message(role=MessageRole.USER, content="one")
        block_message = Message(
            role=MessageRole.USER,
            content=MessageContentText(type="text", text="two"),
        )
        list_message = Message(
            role=MessageRole.USER,
            content=[MessageContentText(type="text", text="three")],
        )

        self.assertEqual(agent_module._input_text_blocks("zero"), ("zero",))
        self.assertEqual(
            agent_module._input_text_blocks(["one", "two"]),
            ("one", "two"),
        )
        self.assertEqual(
            agent_module._input_text_blocks([text_message]),
            ("one",),
        )
        self.assertEqual(
            agent_module._input_text_blocks(cast(Any, object())),
            (),
        )
        self.assertEqual(
            agent_module._message_text_blocks(
                Message(role=MessageRole.USER, content=None)
            ),
            (),
        )
        self.assertEqual(
            agent_module._message_text_blocks(text_message),
            ("one",),
        )
        self.assertEqual(
            agent_module._message_text_blocks(block_message),
            ("two",),
        )
        self.assertEqual(
            agent_module._message_text_blocks(list_message),
            ("three",),
        )
        self.assertEqual(
            agent_module._message_text_blocks(
                Message(
                    role=MessageRole.USER,
                    content=MessageContentFile(type="file", file={}),
                )
            ),
            (),
        )
        self.assertEqual(agent_module._estimated_token_count("one two"), 2)

    def test_agent_prompt_and_file_metadata_helpers_cover_fallbacks(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                'agent = "invalid"\n[engine]\nuri = "ai://x"\n',
                encoding="utf-8",
            )
            runner = AgentTaskTargetRunner(FakeLoader(), ref_base=root)

            prompt = runner._agent_prompt(self._definition())

        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-test",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            size_bucket="provider-bucket",
        )
        assert descriptor.provider_reference is not None
        metadata = agent_module._safe_file_template_metadata(
            TaskInputFile(
                logical_path="provider:openai:file",
                provider_reference=descriptor.provider_reference,
            ),
            index=0,
            plan=TaskFileDeliveryPlan(
                decision=FileDeliveryDecision(
                    mode=FileDeliveryMode.PROVIDER_FILE_ID,
                    reference="file-test",
                )
            ),
        )

        self.assertIsNone(prompt.user)
        self.assertIsNone(prompt.user_template)
        self.assertEqual(metadata["size_bucket"], "provider-bucket")

    async def test_run_appends_files_to_message_inputs(self) -> None:
        message_loader = FakeLoader(response="accepted")
        empty_message_loader = FakeLoader(response="accepted")
        list_content_loader = FakeLoader(response="accepted")
        content_block_loader = FakeLoader(response="accepted")
        messages_loader = FakeLoader(response="accepted")
        runner = AgentTaskTargetRunner(
            message_loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )
        empty_runner = AgentTaskTargetRunner(
            empty_message_loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )
        list_content_runner = AgentTaskTargetRunner(
            list_content_loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )
        content_block_runner = AgentTaskTargetRunner(
            content_block_loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )
        list_runner = AgentTaskTargetRunner(
            messages_loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )
        file = TaskInputFile(
            logical_path="provider:openai:provider_file_id",
            provider_reference=_provider_reference("openai", "file-test"),
        )

        await runner.run(
            self._context(
                self._definition(),
                Message(role=MessageRole.USER, content="summarize"),
                files=(file,),
            )
        )
        await empty_runner.run(
            self._context(
                self._definition(),
                Message(role=MessageRole.USER, content=None),
                files=(file,),
            )
        )
        await list_content_runner.run(
            self._context(
                self._definition(),
                Message(
                    role=MessageRole.USER,
                    content=[MessageContentText(type="text", text="one")],
                ),
                files=(file,),
            )
        )
        await content_block_runner.run(
            self._context(
                self._definition(),
                Message(
                    role=MessageRole.USER,
                    content=MessageContentText(type="text", text="one"),
                ),
                files=(file,),
            )
        )
        await list_runner.run(
            self._context(
                self._definition(),
                [Message(role=MessageRole.USER, content=None)],
                files=(file,),
            )
        )

        message = cast(Message, message_loader.inputs[0])
        empty_message = cast(Message, empty_message_loader.inputs[0])
        list_content_message = cast(Message, list_content_loader.inputs[0])
        content_block_message = cast(Message, content_block_loader.inputs[0])
        messages = cast(list[Message], messages_loader.inputs[0])
        self.assertEqual(cast(list[Any], message.content)[0].text, "summarize")
        self.assertEqual(
            cast(list[Any], message.content)[1].file["file_id"], "file-test"
        )
        self.assertEqual(
            cast(list[Any], empty_message.content)[0].type, "file"
        )
        self.assertEqual(
            cast(list[Any], list_content_message.content)[0].text,
            "one",
        )
        self.assertEqual(
            cast(list[Any], content_block_message.content)[0].text,
            "one",
        )
        self.assertEqual(messages[1].role, MessageRole.USER)
        self.assertEqual(
            cast(list[Any], messages[1].content)[0].file["file_id"],
            "file-test",
        )

    async def test_run_uses_string_list_prompt_with_file_input(self) -> None:
        loader = FakeLoader(response="accepted")
        runner = AgentTaskTargetRunner(
            loader,
            uri="ai://env:KEY@openai/gpt-4o-mini",
        )

        await runner.run(
            self._context(
                self._definition(),
                ["one", "two"],
                files=(
                    TaskInputFile(
                        logical_path="provider:openai:provider_file_id",
                        provider_reference=_provider_reference(
                            "openai",
                            "file-test",
                        ),
                    ),
                ),
            )
        )

        content = cast(list[Any], cast(Message, loader.inputs[0]).content)
        self.assertEqual(content[0].text, "one\ntwo")

    async def test_run_rejects_unsupported_file_mapping_safely(self) -> None:
        artifact_ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )
        local_runner = AgentTaskTargetRunner(
            FakeLoader(), uri="ai://local/model"
        )
        bedrock_runner = AgentTaskTargetRunner(
            FakeLoader(),
            uri="ai://env:KEY@bedrock/anthropic.claude",
        )

        with self.assertRaises(TaskValidationError) as local_error:
            await local_runner.run(
                self._context(
                    self._definition(),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(),
                )
            )
        with self.assertRaises(TaskValidationError) as bedrock_error:
            await bedrock_runner.run(
                self._context(
                    self._definition(),
                    "summarize",
                    files=(
                        TaskInputFile(
                            logical_path="artifact:artifact-1",
                            artifact_ref=artifact_ref,
                        ),
                    ),
                    artifact_store=FakeArtifactStore(),
                )
            )

        self.assertEqual(
            local_error.exception.issues[0].path, "input.files[0]"
        )
        self.assertEqual(
            bedrock_error.exception.issues[0].code, "input.invalid_file"
        )
        self.assertIn(
            "artifact",
            TaskInputFile(
                logical_path="artifact:artifact-1",
                artifact_ref=artifact_ref,
            ).summary(),
        )

    async def test_orchestrator_response_branches_are_consumed(self) -> None:
        with patch(
            "avalan.task.targets.agent.OrchestratorResponse",
            FakeResponse,
        ):
            structured_loader = FakeLoader(response_text='{"answer":"ok"}')
            text_loader = FakeLoader(response_text="plain")
            structured_runner = AgentTaskTargetRunner(structured_loader)
            text_runner = AgentTaskTargetRunner(text_loader)

            structured = await structured_runner.run(
                self._context(
                    self._definition(
                        output=TaskOutputContract.object(
                            schema={"type": "object"}
                        )
                    ),
                    "private prompt",
                )
            )
            text = await text_runner.run(
                self._context(
                    self._definition(output=TaskOutputContract.text()),
                    "private prompt",
                )
            )

        self.assertEqual(structured, {"answer": "ok"})
        self.assertEqual(text, "plain")

    async def test_direct_runner_records_sanitized_events_and_usage(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"
goal_instructions = "Be brief."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            store = InMemoryTaskStore()
            loader = FakeLoader(
                response=FakeResponse(
                    "short summary",
                    input_token_count=3,
                    output_token_count=2,
                ),
                emit_event=True,
            )
            runner = DirectTaskRunner(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "agent-direct-hash",
            )

            result = await runner.run(
                self._definition(output=TaskOutputContract.text()),
                input_value="private prompt",
            )
            events = await store.list_events(result.run.run_id)
            usage = await store.list_usage(result.run.run_id)
            event_manager = cast(FakeEventManager, loader.event_manager)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.attempt.state, TaskAttemptState.SUCCEEDED)
        self.assertEqual(result.output, "short summary")
        self.assertEqual(
            [event.event_type for event in events], ["token_generated"]
        )
        self.assertNotIn("secret-token", str(events[0].payload))
        self.assertNotIn("private-model", str(events[0].payload))
        self.assertEqual(usage[0].source, UsageSource.ESTIMATED)
        self.assertEqual(usage[0].totals.input_tokens, 3)
        self.assertEqual(usage[0].totals.output_tokens, 2)
        self.assertIsNone(usage[0].totals.total_tokens)
        self.assertEqual(usage[0].attempt_id, result.attempt.attempt_id)
        self.assertEqual(event_manager.listeners, [])

    async def test_direct_runner_maps_materialized_file_to_agent_message(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp, TemporaryDirectory() as artifacts:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            input_path = root / "uploads" / "input.txt"
            agent_path.parent.mkdir()
            input_path.parent.mkdir()
            input_path.write_bytes(b"private text")
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"
goal_instructions = "Be brief."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            task_store = InMemoryTaskStore()
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: "artifact-1",
            )
            loader = FakeLoader(response=FakeResponse("short summary"))
            runner = DirectTaskRunner(
                task_store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                artifact_store=artifact_store,
                execution_roots=(root,),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "agent-file-hash",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("text/plain",),
                    ),
                    output=TaskOutputContract.text(),
                ),
                input_value=TaskFileDescriptor.local_path(
                    "uploads/input.txt",
                    mime_type="text/plain",
                ),
            )
            records = await task_store.list_artifacts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        message = cast(Message, loader.inputs[0])
        self.assertEqual(message.role, MessageRole.USER)
        content = cast(list[Any], message.content)
        self.assertEqual(len(content), 1)
        file_block = content[0]
        self.assertEqual(file_block.type, "file")
        self.assertEqual(file_block.file["file_data"], "cHJpdmF0ZSB0ZXh0")
        self.assertEqual(file_block.file["filename"], "task-file.txt")
        self.assertEqual(file_block.file["mime_type"], "text/plain")
        self.assertNotIn("input.txt", str(message))
        self.assertEqual(records[0].artifact_id, "artifact-1")

    async def test_direct_runner_composes_prompt_and_pdf_reference(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
system = "Use structured extraction."
user = "Extract the attached {{ files[0].mime_type }}."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            task_store = InMemoryTaskStore()
            loader = FakeLoader(response=FakeResponse("short summary"))
            runner = DirectTaskRunner(
                task_store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                execution_roots=(root,),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "agent-pdf-reference-hash",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("application/pdf",),
                    ),
                    output=TaskOutputContract.text(),
                ),
                input_value=TaskFileDescriptor.provider_reference_descriptor(
                    "file-pdf",
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    provider="openai",
                    mime_type="application/pdf",
                    identity_hmac="hmac-pdf",
                ),
            )
            records = await task_store.list_artifacts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(records, ())
        message = cast(Message, loader.inputs[0])
        content = cast(list[Any], message.content)
        self.assertEqual(len(content), 2)
        self.assertEqual(
            content[0].text,
            "Extract the attached application/pdf.",
        )
        self.assertEqual(content[1].file["file_id"], "file-pdf")
        self.assertEqual(content[1].file["mime_type"], "application/pdf")
        self.assertNotIn("Use structured extraction", str(message))
        self.assertNotIn("hmac-pdf", str(message))

    async def test_direct_file_task_runs_with_conversion_and_inspection(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp, TemporaryDirectory() as artifacts:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            input_path = root / "uploads" / "private.html"
            agent_path.parent.mkdir()
            input_path.parent.mkdir()
            input_path.write_text("<h1>Private</h1>", encoding="utf-8")
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"
goal_instructions = "Be brief."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            artifact_ids = iter(("input-artifact", "converted-artifact"))
            task_store = InMemoryTaskStore()
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            loader = FakeLoader(
                response=FakeResponse(
                    "short summary",
                    input_token_count=5,
                    output_token_count=4,
                ),
                emit_event=True,
            )
            runner = DirectTaskRunner(
                task_store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                artifact_store=artifact_store,
                execution_roots=(root,),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "agent-file-converted-hash",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        conversions=("text",),
                        mime_types=("text/html",),
                    ),
                    output=TaskOutputContract.text(),
                    artifact=TaskArtifactPolicy.references_only(
                        retention_days=7,
                    ),
                ),
                input_value=TaskFileDescriptor.local_path(
                    "uploads/private.html",
                    mime_type="text/html",
                    conversions=(TaskFileConversionRequest(name="text"),),
                    metadata={"display_name": "private.html"},
                ),
            )
            records = await task_store.list_artifacts(result.run.run_id)
            events = await task_store.list_events(result.run.run_id)
            usage = await task_store.list_usage(result.run.run_id)
            run = await task_store.get_run(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.attempt.state, TaskAttemptState.SUCCEEDED)
        self.assertEqual(result.output, "short summary")
        self.assertEqual(
            [record.purpose for record in records],
            [TaskArtifactPurpose.INPUT, TaskArtifactPurpose.CONVERTED],
        )
        self.assertEqual(
            [record.retention.delete_after_days for record in records],
            [7, 7],
        )
        self.assertEqual(
            records[1].provenance.source_artifact_id,
            "input-artifact",
        )
        self.assertEqual(records[1].provenance.converter, "text")
        self.assertNotIn("private.html", str(run.request.input_summary))
        artifact_summaries = " ".join(
            str(record.summary()) for record in records
        )
        self.assertNotIn("private.html", artifact_summaries)
        self.assertNotIn("<h1>Private</h1>", str(run))
        self.assertNotIn("secret-token", str(events[0].payload))
        self.assertEqual(usage[0].totals.input_tokens, 5)
        self.assertEqual(usage[0].totals.output_tokens, 4)
        message = cast(Message, loader.inputs[0])
        content = cast(list[Any], message.content)
        self.assertEqual(content[0].type, "file")
        self.assertEqual(content[0].file["mime_type"], "text/plain")

    async def test_direct_file_task_rejects_unknown_conversion_safely(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp, TemporaryDirectory() as artifacts:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            input_path = root / "uploads" / "private.bin"
            agent_path.parent.mkdir()
            input_path.parent.mkdir()
            input_path.write_bytes(b"private bytes")
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            task_store = InMemoryTaskStore()
            loader = FakeLoader()
            runner = DirectTaskRunner(
                task_store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "input-artifact",
                ),
                execution_roots=(root,),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "agent-file-bad-conversion-hash",
            )

            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    self._definition(
                        input_contract=TaskInputContract.file(
                            conversions=("image",),
                        ),
                    ),
                    input_value=TaskFileDescriptor.local_path(
                        "uploads/private.bin",
                        conversions=(TaskFileConversionRequest(name="image"),),
                    ),
                )

        self.assertEqual(loader.inputs, [])
        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(
            error.exception.issues[0].path,
            "input.file_conversions[0]",
        )
        self.assertNotIn("private.bin", str(error.exception))
        self.assertNotIn("private bytes", str(error.exception))

    async def test_direct_runner_rejects_file_without_artifact_backend(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            input_path = root / "uploads" / "input.txt"
            agent_path.parent.mkdir()
            input_path.parent.mkdir()
            input_path.write_bytes(b"private text")
            agent_path.write_text(
                """
[agent]
name = "Valid"

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            loader = FakeLoader()
            store = InMemoryTaskStore()
            runner = DirectTaskRunner(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                execution_roots=(root,),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "agent-file-no-backend-hash",
            )

            result = await runner.run(
                self._definition(
                    input_contract=TaskInputContract.file(
                        mime_types=("text/plain",),
                    ),
                ),
                input_value=TaskFileDescriptor.local_path(
                    "uploads/input.txt",
                    mime_type="text/plain",
                ),
            )
            attempts = await store.list_attempts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(attempts[0].state, TaskAttemptState.FAILED)
        error_summary = cast(
            dict[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["code"], "input_contract.failed")
        self.assertIn("artifact.bytes_unsupported", str(error_summary))
        self.assertNotIn("uploads/input.txt", str(error_summary))
        self.assertEqual(loader.inputs, [])

    async def test_direct_runner_allows_missing_event_api_and_usage(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agents" / "valid.toml"
            agent_path.parent.mkdir()
            agent_path.write_text(
                """
[agent]
name = "Valid"
task = "Answer"
goal_instructions = "Be brief."

[engine]
uri = "ai://env:KEY@openai/gpt-4o-mini"
""",
                encoding="utf-8",
            )
            store = InMemoryTaskStore()
            loader = FakeLoader(response=FakeResponse("short summary"))
            loader.event_manager = object()
            runner = DirectTaskRunner(
                store,
                target=AgentTaskTargetRunner(loader, ref_base=root),
                hmac_provider=StaticHmacProvider(),
                definition_hash=lambda task: "agent-direct-no-usage-hash",
            )

            result = await runner.run(
                self._definition(output=TaskOutputContract.text()),
                input_value="private prompt",
            )
            usage = await store.list_usage(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "short summary")
        self.assertEqual(usage, ())

    def _context(
        self,
        definition: TaskDefinition,
        input_value: object,
        *,
        files: tuple[TaskInputFile, ...] = (),
        artifact_store: FakeArtifactStore | None = None,
        cancellation_checker: Callable[[], Awaitable[None]] | None = None,
        usage_observer: (
            Callable[[object], Awaitable[None] | None] | None
        ) = None,
    ) -> TaskTargetContext:
        return TaskTargetContext(
            definition=definition,
            execution=TaskExecutionContext(
                run_id="run-1",
                attempt_id="attempt-1",
                attempt_number=1,
            ),
            input_value=input_value,
            files=files,
            artifact_store=artifact_store,
            cancellation_checker=cancellation_checker,
            usage_observer=usage_observer,
        )

    def _definition(
        self,
        *,
        input_contract: TaskInputContract | None = None,
        output: TaskOutputContract | None = None,
        artifact: TaskArtifactPolicy | None = None,
        limits: TaskLimitsPolicy | None = None,
        retry: TaskRetryPolicy | None = None,
    ) -> TaskDefinition:
        return TaskDefinition(
            task=TaskMetadata(name="agent", version="1"),
            input=input_contract or TaskInputContract.string(),
            output=output or TaskOutputContract.text(),
            execution=TaskExecutionTarget.agent("agents/valid.toml"),
            artifact=artifact or TaskArtifactPolicy(),
            limits=limits or TaskLimitsPolicy(),
            retry=retry or TaskRetryPolicy(),
        )


def _provider_reference(
    provider: str,
    reference: str,
    *,
    kind: TaskProviderReferenceKind = (
        TaskProviderReferenceKind.PROVIDER_FILE_ID
    ),
    media_type: str | None = None,
) -> TaskProviderReference:
    descriptor = TaskFileDescriptor.provider_reference_descriptor(
        reference,
        kind=kind,
        provider=provider,
        mime_type=media_type,
    )
    assert descriptor.provider_reference is not None
    return descriptor.provider_reference


if __name__ == "__main__":
    main()
