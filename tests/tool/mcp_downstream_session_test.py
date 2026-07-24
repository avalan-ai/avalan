from asyncio import CancelledError, Event, create_task
from contextlib import suppress
from datetime import UTC, datetime
from subprocess import run
from sys import executable
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch
from uuid import UUID

from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.session import ElicitationFnT
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import BaseModel, Field

from avalan.agent.execution import (
    AgentExecution,
    AttachedInteractionRuntime,
    BranchInteractionBroker,
    DurableInteractionRuntime,
)
from avalan.entities import ToolCallContext
from avalan.interaction.broker import (
    InteractionBrokerRequest,
    InteractionRequestResult,
)
from avalan.interaction.entities import (
    AnsweredResolution,
    AnswerProvenance,
    CancelledResolution,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    ExpiredResolution,
    FreeFormOther,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PrincipalScope,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    SupersededResolution,
    TextAnswer,
    TextQuestion,
    TimedOutResolution,
    UnavailableResolution,
)
from avalan.interaction.error import InputErrorCode, InputValidationError
from avalan.interaction.policy import InteractionActor
from avalan.tool import mcp as mcp_module
from avalan.tool import mcp_session as session_module

_AGENT_ID = "00000000-0000-0000-0000-000000000001"


class _NameForm(BaseModel):
    name: str = Field(title="Name", min_length=1, max_length=20)


def _origin(
    *,
    task_id: str | None = "task-origin",
    branch_id: str = "branch-origin",
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id="run-origin",
        turn_id="turn-origin",
        agent_id=_AGENT_ID,
        branch_id=branch_id,
        model_call_id="model-call-origin",
        stream_session_id="stream-origin",
        task_id=task_id,
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent.toml",
            agent_definition_revision="agent-revision",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model",
            tool_revision="tool-revision",
            capability_revision="capability-revision",
        ),
        principal=PrincipalScope(user_id="user-origin"),
    )


def _terminal_result(
    resolution: object,
) -> InteractionRequestResult:
    return cast(
        InteractionRequestResult,
        SimpleNamespace(
            delivery=SimpleNamespace(
                record=SimpleNamespace(
                    request=SimpleNamespace(resolution=resolution)
                )
            )
        ),
    )


def _resolution(
    kind: type[
        AnsweredResolution
        | DeclinedResolution
        | CancelledResolution
        | TimedOutResolution
        | UnavailableResolution
        | ExpiredResolution
        | SupersededResolution
    ],
    *,
    answers: tuple[object, ...] = (),
) -> object:
    values: dict[str, object] = {
        "request_id": "request-origin",
        "provenance": AnswerProvenance.HUMAN,
        "resolved_at": datetime.now(UTC),
    }
    if kind is AnsweredResolution:
        values["answers"] = answers
    return kind(**values)


class _Broker:
    def __init__(
        self,
        result: InteractionRequestResult | None = None,
        *,
        error: BaseException | None = None,
        block: bool = False,
    ) -> None:
        self.result = result
        self.error = error
        self.block = block
        self.requests: list[InteractionBrokerRequest] = []
        self.started = Event()
        self.cancelled = False

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        self.requests.append(request)
        self.started.set()
        try:
            if self.block:
                await Event().wait()
            if self.error is not None:
                raise self.error
            assert self.result is not None
            return self.result
        except CancelledError:
            self.cancelled = True
            raise

    async def cancel_scope(self, _command: object) -> object:
        return object()


async def _unused_handler(_context: object) -> object:
    raise AssertionError("the fake broker must not invoke the handler")


async def _unused_stager(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("downstream MCP must not claim durable suspension")


def _context(
    result: InteractionRequestResult | None = None,
    *,
    error: BaseException | None = None,
    block: bool = False,
    origin: ExecutionOrigin | None = None,
) -> tuple[ToolCallContext, _Broker]:
    bound_origin = origin or _origin()
    broker = _Broker(result, error=error, block=block)
    runtime = AttachedInteractionRuntime(
        broker=cast(Any, broker),
        actor=InteractionActor(principal=bound_origin.principal),
        handler=_unused_handler,
    )
    execution = cast(
        AgentExecution,
        SimpleNamespace(
            origin=bound_origin,
            interaction_runtime=runtime,
        ),
    )
    return (
        ToolCallContext(
            agent_id=UUID(_AGENT_ID),
            execution=execution,
            execution_origin=bound_origin,
            interaction_broker=cast(BranchInteractionBroker, broker),
        ),
        broker,
    )


def _form(
    schema: dict[str, object],
    *,
    message: str = "Choose a value",
    related_task_id: str | None = None,
    required_other: list[str] | None = None,
    task: bool = False,
) -> mcp_types.ElicitRequestFormParams:
    metadata: dict[str, object] = {}
    if related_task_id is not None:
        metadata[session_module.MCP_RELATED_TASK_METADATA_KEY] = {
            "taskId": related_task_id
        }
    if required_other is not None:
        metadata[session_module.MCP_REQUIRED_OTHER_METADATA_KEY] = (
            required_other
        )
    meta = (
        mcp_types.RequestParams.Meta.model_validate(metadata)
        if metadata
        else None
    )
    return mcp_types.ElicitRequestFormParams(
        message=message,
        requestedSchema=schema,
        **({"_meta": meta} if meta is not None else {}),
        task=mcp_types.TaskMetadata(ttl=60) if task else None,
    )


class McpSessionProjectionTestCase(TestCase):
    def test_capabilities_advertise_only_routable_form(self) -> None:
        self.assertEqual(
            session_module._client_capabilities(False).model_dump(
                mode="json",
                exclude_none=True,
            ),
            {},
        )
        self.assertEqual(
            session_module._client_capabilities(True).model_dump(
                mode="json",
                exclude_none=True,
            ),
            {"elicitation": {"form": {}}},
        )

    def test_projects_supported_flat_form_questions(self) -> None:
        schema = {
            "type": "object",
            "title": "Details",
            "additionalProperties": False,
            "properties": {
                "ready": {
                    "type": "boolean",
                    "title": "Ready",
                    "default": True,
                },
                "name": {
                    "type": "string",
                    "description": "Display name",
                    "minLength": 1,
                    "maxLength": 20,
                    "default": "Ada",
                },
                "color": {
                    "type": "string",
                    "oneOf": [
                        {"const": "r", "title": "Red"},
                        {"const": "b", "title": "Blue"},
                    ],
                    "default": "b",
                },
            },
            "required": ["ready", "name", "color"],
        }

        questions = session_module._questions(schema)

        self.assertIsInstance(questions[0], ConfirmationQuestion)
        self.assertIsInstance(questions[1], TextQuestion)
        self.assertIsInstance(questions[2], SingleSelectionQuestion)
        self.assertTrue(questions[0].default_value)
        self.assertEqual(questions[1].default_value, "Ada")
        selection = cast(SingleSelectionQuestion, questions[2])
        self.assertEqual(
            [
                (str(choice.value), choice.label)
                for choice in selection.choices
            ],
            [("r", "Red"), ("b", "Blue")],
        )

    def test_projects_multiple_selection_and_literal_other_choice(
        self,
    ) -> None:
        questions = session_module._questions(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "title": "Colors",
                        "items": {
                            "type": "string",
                            "enum": ["red", "Other"],
                        },
                        "uniqueItems": True,
                        "minItems": 1,
                        "maxItems": 2,
                        "default": ["red"],
                    }
                },
                "required": ["colors"],
            }
        )

        question = cast(MultipleSelectionQuestion, questions[0])
        self.assertEqual(question.default_value, (ChoiceValue("red"),))
        self.assertEqual(question.constraints.minimum, 1)
        self.assertEqual(question.constraints.maximum, 2)
        self.assertFalse(question.allow_other)
        self.assertEqual(question.choices[1].label, "Other option")

    def test_inverts_avalan_multiline_and_other_degradations(self) -> None:
        for pattern in (
            "^[^\\r]*$",
            "^(?:[^\\r]|\\r\\n)*$",
        ):
            with self.subTest(pattern=pattern):
                schema = {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "title": "Notes",
                            "minLength": 1,
                            "maxLength": 65_536,
                            "pattern": pattern,
                        },
                        "single": {
                            "type": "string",
                            "enum": ["stable-a", "stable-b"],
                        },
                        "__avalan_other__single": {
                            "type": "string",
                            "title": "Other",
                            "description": (
                                "Free-form alternative for Choose one."
                            ),
                            "minLength": 1,
                            "maxLength": 4_096,
                        },
                        "multiple": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["stable-a", "stable-b"],
                            },
                            "uniqueItems": True,
                            "minItems": 0,
                            "maxItems": 3,
                        },
                        "__avalan_other__multiple": {
                            "type": "string",
                            "title": "Other",
                            "description": (
                                "Free-form alternative for Choose several."
                            ),
                            "minLength": 1,
                            "maxLength": 4_096,
                        },
                    },
                    "required": ["notes"],
                }
                questions = session_module._questions(
                    schema,
                    required_other=frozenset({"single", "multiple"}),
                )

                self.assertIsInstance(questions[0], MultilineTextQuestion)
                single = cast(SingleSelectionQuestion, questions[1])
                multiple = cast(MultipleSelectionQuestion, questions[2])
                self.assertTrue(single.allow_other)
                self.assertTrue(single.required)
                self.assertTrue(multiple.allow_other)
                self.assertTrue(multiple.required)
                self.assertEqual(multiple.constraints.maximum, 3)
                optional = session_module._questions(schema)
                self.assertFalse(optional[1].required)
                self.assertFalse(optional[2].required)

        answers = (
            MultilineTextAnswer(
                question_id="notes",
                provenance=AnswerProvenance.HUMAN,
                value="line 1\r\nline 2",
            ),
            SingleSelectionAnswer(
                question_id="single",
                provenance=AnswerProvenance.HUMAN,
                value=FreeFormOther(text="custom"),
            ),
            MultipleSelectionAnswer(
                question_id="multiple",
                provenance=AnswerProvenance.HUMAN,
                values=(
                    SelectedChoice(value=ChoiceValue("stable-a")),
                    FreeFormOther(text="extra"),
                ),
            ),
        )
        self.assertEqual(
            session_module._answer_content(answers),
            {
                "notes": "line 1\nline 2",
                "__avalan_other__single": "custom",
                "multiple": ["stable-a"],
                "__avalan_other__multiple": "extra",
            },
        )

    def test_rejects_malformed_unsafe_and_unsupported_forms(self) -> None:
        valid_property = {"type": "string"}
        cases = {
            "unknown root": {
                "type": "object",
                "properties": {"value": valid_property},
                "allOf": [],
            },
            "nested": {
                "type": "object",
                "properties": {"value": {"type": "object"}},
            },
            "non object": {
                "type": "object",
                "properties": {"value": "string"},
            },
            "too many": {
                "type": "object",
                "properties": {
                    str(index): valid_property for index in range(4)
                },
            },
            "unknown required": {
                "type": "object",
                "properties": {"value": valid_property},
                "required": ["missing"],
            },
            "unsafe": {
                "type": "object",
                "properties": {
                    "api_key": {"type": "string"},
                },
            },
            "unsafe verification code": {
                "type": "object",
                "properties": {"totp": {"type": "string"}},
            },
            "unsafe bank detail": {
                "type": "object",
                "properties": {"routing-number": {"type": "string"}},
            },
            "unsafe debit card": {
                "type": "object",
                "properties": {"debit_card": {"type": "string"}},
            },
            "unsafe card details": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "title": "Card details",
                    }
                },
            },
            "unsafe default": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "default": "session token",
                    }
                },
            },
            "nested array": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "array",
                        "items": {"type": "array", "enum": ["x"]},
                    }
                },
            },
            "repeated selection": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["x"]},
                        "uniqueItems": False,
                    }
                },
            },
            "unsupported keyword": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "pattern": ".*"},
                },
            },
            "orphan other": {
                "type": "object",
                "properties": {
                    "__avalan_other__missing": {
                        "type": "string",
                        "title": "Other",
                        "description": "Free-form alternative for Missing.",
                        "minLength": 1,
                        "maxLength": 4_096,
                    }
                },
            },
            "other on text": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "__avalan_other__value": {
                        "type": "string",
                        "title": "Other",
                        "description": "Free-form alternative for Value.",
                        "minLength": 1,
                        "maxLength": 4_096,
                    },
                },
            },
            "other on boolean": {
                "type": "object",
                "properties": {
                    "value": {"type": "boolean"},
                    "__avalan_other__value": {
                        "type": "string",
                        "title": "Other",
                        "description": "Free-form alternative for Value.",
                        "minLength": 1,
                        "maxLength": 4_096,
                    },
                },
            },
            "malformed other": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": ["a"]},
                    "__avalan_other__value": {
                        "type": "string",
                        "title": "Alternative",
                        "description": "Free-form alternative for Value.",
                        "minLength": 1,
                        "maxLength": 4_096,
                    },
                },
            },
            "unknown property keyword": {
                "type": "object",
                "properties": {
                    "value": {"type": "boolean", "minLength": 0},
                },
            },
            "open object": {
                "type": "object",
                "properties": {"value": valid_property},
                "additionalProperties": True,
            },
        }
        for name, schema in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                session_module._questions(schema)
        self.assertEqual(
            len(
                session_module._questions(
                    {
                        "type": "object",
                        "properties": {
                            "debit_summary": {"type": "string"},
                            "cardinality": {"type": "string"},
                        },
                    }
                )
            ),
            2,
        )
        with self.assertRaises(ValueError):
            session_module._questions(
                {
                    "type": "object",
                    "properties": {"value": valid_property},
                },
                required_other=frozenset({"missing"}),
            )

    def test_validates_required_other_metadata(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "choice": {
                    "type": "string",
                    "enum": ["stable-a", "stable-b"],
                },
                "__avalan_other__choice": {
                    "type": "string",
                    "title": "Other",
                    "description": "Free-form alternative for Choice.",
                    "minLength": 1,
                    "maxLength": 4_096,
                },
            },
        }
        marker = session_module._required_other(
            _form(schema, required_other=["choice"])
        )
        self.assertEqual(marker, frozenset({"choice"}))
        question = session_module._questions(
            schema,
            required_other=cast(frozenset[str], marker),
        )[0]
        self.assertTrue(question.required)
        self.assertFalse(session_module._questions(schema)[0].required)

        for value in ([], ["choice", "choice"], ["choice", 1], "choice"):
            with self.subTest(value=value):
                malformed = session_module._required_other(
                    _form(schema, required_other=cast(Any, value))
                )
                self.assertIsInstance(malformed, mcp_types.ErrorData)
                self.assertEqual(malformed.code, mcp_types.INVALID_PARAMS)

    def test_rejects_invalid_defaults_constraints_and_choices(self) -> None:
        properties = (
            {"type": "boolean", "default": "yes"},
            {"type": "string", "default": 1},
            {"type": "string", "enum": ["a"], "default": "b"},
            {"type": "string", "enum": ["a", "a"]},
            {
                "type": "string",
                "enum": ["a"],
                "oneOf": [{"const": "a", "title": "A"}],
            },
            {"type": "string", "enum": "a"},
            {"type": "string", "oneOf": "a"},
            {"type": "string", "oneOf": [{"const": "a"}]},
            {"type": "string", "title": ""},
            {
                "type": "array",
                "items": {"type": "string", "enum": ["a"]},
                "minItems": 2,
                "maxItems": 1,
            },
            {
                "type": "array",
                "items": {"type": "string", "enum": ["a"]},
                "default": [1],
            },
        )
        for value in properties:
            with self.subTest(value=value), self.assertRaises(ValueError):
                session_module._questions(
                    {
                        "type": "object",
                        "properties": {"value": value},
                    }
                )

    def test_maps_canonical_answers_and_terminal_outcomes(self) -> None:
        answers = (
            ConfirmationAnswer(
                question_id="confirm",
                provenance=AnswerProvenance.HUMAN,
                value=True,
            ),
            TextAnswer(
                question_id="text",
                provenance=AnswerProvenance.HUMAN,
                value="Ada",
            ),
            SingleSelectionAnswer(
                question_id="single",
                provenance=AnswerProvenance.HUMAN,
                value=SelectedChoice(value=ChoiceValue("one")),
            ),
            MultipleSelectionAnswer(
                question_id="many",
                provenance=AnswerProvenance.HUMAN,
                values=(
                    SelectedChoice(value=ChoiceValue("one")),
                    FreeFormOther(text="custom"),
                ),
            ),
        )
        accepted = session_module._broker_result(
            _terminal_result(_resolution(AnsweredResolution, answers=answers))
        )
        self.assertIsInstance(accepted, mcp_types.ElicitResult)
        self.assertEqual(accepted.action, "accept")
        self.assertEqual(
            accepted.content,
            {
                "confirm": True,
                "text": "Ada",
                "single": "one",
                "many": ["one"],
                "__avalan_other__many": "custom",
            },
        )
        declined = session_module._broker_result(
            _terminal_result(_resolution(DeclinedResolution))
        )
        self.assertEqual(declined.action, "decline")
        for kind in (
            CancelledResolution,
            TimedOutResolution,
        ):
            with self.subTest(kind=kind):
                result = session_module._broker_result(
                    _terminal_result(_resolution(kind))
                )
                self.assertEqual(result.action, "cancel")
                self.assertIsNone(result.content)
        errors = (
            (
                ExpiredResolution,
                -32_010,
                {"code": "avalan.input.expired"},
            ),
            (
                UnavailableResolution,
                -32_001,
                {"code": "avalan.input.unavailable"},
            ),
            (
                SupersededResolution,
                -32_009,
                {"code": "avalan.input.already_resolved"},
            ),
        )
        for kind, code, data in errors:
            with self.subTest(kind=kind):
                error = session_module._broker_result(
                    _terminal_result(_resolution(kind))
                )
                self.assertIsInstance(error, mcp_types.ErrorData)
                self.assertEqual(error.code, code)
                self.assertEqual(error.data, data)
                self.assertTrue(error.message)

    def test_rejects_unsupported_answer_and_nonterminal_result(self) -> None:
        with self.assertRaises(TypeError):
            session_module._answer_content(
                (cast(Any, SimpleNamespace(question_id="value")),)
            )
        missing = cast(
            InteractionRequestResult,
            SimpleNamespace(delivery=None),
        )
        self.assertEqual(
            session_module._broker_result(missing).code,
            mcp_types.INVALID_PARAMS,
        )
        pending = _terminal_result(None)
        self.assertEqual(
            session_module._broker_result(pending).code,
            mcp_types.INTERNAL_ERROR,
        )

    def test_call_parameters_are_bounded_and_correlated(self) -> None:
        self.assertEqual(session_module._call_timeout({}), 300.0)
        self.assertEqual(session_module._call_timeout({"timeout": 1}), 1.0)
        for value in (True, 0, -1, 3_601, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                session_module._call_timeout({"timeout": value})
        self.assertEqual(
            session_module._call_meta(
                {"request_id": "request"},
                "task",
            ),
            {
                "progressToken": "request",
                session_module.MCP_RELATED_TASK_METADATA_KEY: {
                    "taskId": "task"
                },
            },
        )
        self.assertEqual(
            session_module._call_meta(
                {"progress_token": False},
                None,
            ),
            {},
        )
        self.assertEqual(
            session_module._http_options(
                {"headers": {"Authorization": "Bearer value"}}
            ),
            {
                "headers": {
                    "Authorization": "Bearer value",
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                "timeout": None,
            },
        )

    def test_related_task_and_context_label_validation(self) -> None:
        self.assertIsNone(session_module._expected_related_task({}))
        self.assertEqual(
            session_module._expected_related_task({"related_task_id": "task"}),
            "task",
        )
        for value in ("", "has space", "x" * 129, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                session_module._expected_related_task(
                    {"related_task_id": value}
                )
        malformed = mcp_types.ElicitRequestFormParams.model_validate(
            {
                "message": "Value",
                "requestedSchema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                "_meta": {
                    session_module.MCP_RELATED_TASK_METADATA_KEY: {
                        "taskId": "has space"
                    }
                },
            }
        )
        related_error = session_module._related_task(malformed)
        self.assertEqual(related_error.code, mcp_types.INVALID_PARAMS)
        label = session_module._context_label(
            "https://server.example/mcp",
            "tool\nname",
            "task-123456789",
            _AGENT_ID,
            "branch-123456789",
        )
        self.assertLessEqual(len(label), 80)
        self.assertNotIn("\n", label)
        self.assertIn("server.example", label)
        for value in (FreeFormOther(text="custom"), object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                session_module._selection(cast(Any, value))


class McpSessionRouterTestCase(IsolatedAsyncioTestCase):
    async def test_routes_accept_to_exact_origin(self) -> None:
        resolution = _resolution(
            AnsweredResolution,
            answers=(
                SingleSelectionAnswer(
                    question_id="choice",
                    provenance=AnswerProvenance.HUMAN,
                    value=SelectedChoice(value=ChoiceValue("stable-a")),
                ),
            ),
        )
        context, broker = _context(_terminal_result(resolution))
        router = session_module._ElicitationRouter(
            uri="https://server.example/mcp",
            tool_name="remote.tool",
            context=context,
            related_task_id="mcp-task",
            ttl_seconds=60,
        )

        result = await router(
            cast(Any, None),
            _form(
                {
                    "type": "object",
                    "properties": {
                        "choice": {
                            "type": "string",
                            "enum": ["stable-a", "stable-b"],
                        },
                        "__avalan_other__choice": {
                            "type": "string",
                            "title": "Other",
                            "description": "Free-form alternative for Choice.",
                            "minLength": 1,
                            "maxLength": 4_096,
                        },
                    },
                },
                related_task_id="mcp-task",
                required_other=["choice"],
            ),
        )

        self.assertEqual(result.action, "accept")
        self.assertEqual(result.content, {"choice": "stable-a"})
        self.assertEqual(
            result.meta,
            {
                session_module.MCP_RELATED_TASK_METADATA_KEY: {
                    "taskId": "mcp-task"
                }
            },
        )
        request = broker.requests[0]
        question = cast(SingleSelectionQuestion, request.questions[0])
        self.assertTrue(question.required)
        self.assertTrue(question.allow_other)
        self.assertIs(request.origin, context.execution_origin)
        self.assertEqual(request.actor.principal, request.origin.principal)
        self.assertIn("server.example", request.context_label or "")
        self.assertIn("remote.tool", request.context_label or "")
        self.assertIn("task", request.context_label or "")
        self.assertIn("agent", request.context_label or "")
        self.assertIn("branch", request.context_label or "")
        self.assertIsNotNone(request.handler)

    async def test_routes_explicit_durable_controller_broker(self) -> None:
        origin = _origin(branch_id="durable-branch")
        resolution = _resolution(
            AnsweredResolution,
            answers=(
                TextAnswer(
                    question_id="value",
                    provenance=AnswerProvenance.HUMAN,
                    value="durable answer",
                ),
            ),
        )
        broker = _Broker(_terminal_result(resolution))
        runtime = DurableInteractionRuntime(
            actor=InteractionActor(principal=origin.principal),
            stager=cast(Any, _unused_stager),
        )
        context = ToolCallContext(
            agent_id=UUID(_AGENT_ID),
            execution=cast(
                AgentExecution,
                SimpleNamespace(
                    origin=origin,
                    interaction_runtime=runtime,
                ),
            ),
            execution_origin=origin,
            interaction_broker=cast(BranchInteractionBroker, broker),
        )
        router = session_module._ElicitationRouter(
            uri="https://server.example/mcp",
            tool_name="durable.tool",
            context=context,
            related_task_id=None,
            ttl_seconds=60,
        )

        self.assertTrue(router.form_capable)
        result = await router(
            cast(Any, None),
            _form(
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                }
            ),
        )

        self.assertEqual(result.action, "accept")
        self.assertEqual(result.content, {"value": "durable answer"})
        self.assertEqual(len(broker.requests), 1)
        request = broker.requests[0]
        self.assertIs(request.origin, origin)
        self.assertEqual(request.actor.principal, origin.principal)
        self.assertIsNone(request.handler)

    async def test_rejects_wrong_origin_mode_task_and_sensitive_input(
        self,
    ) -> None:
        context, broker = _context(
            _terminal_result(_resolution(DeclinedResolution))
        )
        router = session_module._ElicitationRouter(
            uri="https://server.example/mcp",
            tool_name="remote.tool",
            context=context,
            related_task_id="mcp-task",
            ttl_seconds=60,
        )
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        malformed_related = mcp_types.ElicitRequestFormParams.model_validate(
            {
                "message": "Value",
                "requestedSchema": schema,
                "_meta": {
                    session_module.MCP_RELATED_TASK_METADATA_KEY: {
                        "taskId": "has space"
                    }
                },
            }
        )
        cases: tuple[object, ...] = (
            mcp_types.ElicitRequestURLParams(
                message="Authenticate",
                url="https://server.example/auth",
                elicitationId="auth",
            ),
            cast(mcp_types.ElicitRequestParams, object()),
            malformed_related,
            _form(
                schema,
                related_task_id="mcp-task",
                required_other=[],
            ),
            _form(schema),
            _form(schema, related_task_id="wrong-task"),
            _form(schema, related_task_id="mcp-task", task=True),
            _form(
                schema,
                related_task_id="mcp-task",
                message="Enter an API key",
            ),
            _form(
                {"type": "object", "properties": {}},
                related_task_id="mcp-task",
            ),
        )
        for params in cases:
            with self.subTest(params=type(params).__name__):
                result = await router(cast(Any, None), cast(Any, params))
                self.assertEqual(result.code, mcp_types.INVALID_PARAMS)
        self.assertEqual(broker.requests, [])

        execution = cast(Any, context.execution)
        execution.origin = _origin(branch_id="sibling")
        result = await router(
            cast(Any, None),
            _form(schema, related_task_id="mcp-task"),
        )
        self.assertEqual(result.code, mcp_types.INVALID_PARAMS)
        self.assertEqual(broker.requests, [])

        origin = _origin()
        durable = DurableInteractionRuntime(
            actor=InteractionActor(principal=origin.principal),
            stager=cast(Any, _unused_stager),
        )
        durable_context = ToolCallContext(
            agent_id=UUID(_AGENT_ID),
            execution=cast(
                AgentExecution,
                SimpleNamespace(
                    origin=origin,
                    interaction_runtime=durable,
                ),
            ),
            execution_origin=origin,
        )
        durable_router = session_module._ElicitationRouter(
            uri="https://server.example/mcp",
            tool_name="remote.tool",
            context=durable_context,
            related_task_id=None,
            ttl_seconds=60,
        )
        self.assertFalse(durable_router.form_capable)
        unavailable = await durable_router(
            cast(Any, None),
            _form(schema),
        )
        self.assertEqual(unavailable.code, mcp_types.INVALID_PARAMS)

    async def test_maps_broker_failure_and_propagates_cancellation(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }
        context, _broker = _context(error=RuntimeError("private failure"))
        router = session_module._ElicitationRouter(
            uri="memory://server",
            tool_name="tool",
            context=context,
            related_task_id=None,
            ttl_seconds=60,
        )
        result = await router(cast(Any, None), _form(schema))
        self.assertEqual(result.code, mcp_types.INTERNAL_ERROR)
        self.assertNotIn("private", result.message)

        context, _broker = _context(
            error=InputValidationError(
                InputErrorCode.FORBIDDEN,
                "mcp.origin",
                "safe origin rejection",
            )
        )
        router = session_module._ElicitationRouter(
            uri="memory://server",
            tool_name="tool",
            context=context,
            related_task_id=None,
            ttl_seconds=60,
        )
        result = await router(cast(Any, None), _form(schema))
        self.assertEqual(result.code, mcp_types.INVALID_PARAMS)
        self.assertEqual(result.message, "safe origin rejection")

        execution = cast(Any, context.execution)
        execution.origin = _origin(branch_id="raced-sibling")
        with self.assertRaisesRegex(RuntimeError, "no longer active"):
            await router._request(
                "Value",
                session_module._questions(schema),
            )

        context, broker = _context(block=True)
        router = session_module._ElicitationRouter(
            uri="memory://server",
            tool_name="tool",
            context=context,
            related_task_id=None,
            ttl_seconds=60,
        )
        task = create_task(router(cast(Any, None), _form(schema)))
        await broker.started.wait()
        task.cancel()
        with self.assertRaises(CancelledError):
            await task
        self.assertTrue(broker.cancelled)

    async def test_initialize_rejects_wrong_protocol_version(self) -> None:
        session = AsyncMock()
        session.send_request.return_value = mcp_types.InitializeResult(
            protocolVersion="2025-06-18",
            capabilities=mcp_types.ServerCapabilities(),
            serverInfo=mcp_types.Implementation(
                name="server",
                version="1",
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "2025-11-25"):
            await session_module._initialize(
                cast(ClientSession, session),
                form_capable=True,
            )
        session.send_notification.assert_not_awaited()

    async def test_real_initialized_session_handles_reverse_form(self) -> None:
        resolution = _resolution(
            AnsweredResolution,
            answers=(
                TextAnswer(
                    question_id="name",
                    provenance=AnswerProvenance.HUMAN,
                    value="Ada",
                ),
            ),
        )
        context, broker = _context(_terminal_result(resolution))
        router = session_module._ElicitationRouter(
            uri="memory://server",
            tool_name="ask",
            context=context,
            related_task_id=None,
            ttl_seconds=60,
        )
        server = FastMCP("downstream-test")

        @server.tool()
        async def ask(ctx: Context) -> str:
            elicited = await ctx.elicit(
                message="Who should be greeted?",
                schema=_NameForm,
            )
            return elicited.data.name if elicited.data is not None else "none"

        async with create_client_server_memory_streams() as (
            client_streams,
            server_streams,
        ):
            server_task = create_task(
                server._mcp_server.run(
                    *server_streams,
                    server._mcp_server.create_initialization_options(),
                    raise_exceptions=True,
                )
            )
            try:
                async with ClientSession(
                    *client_streams,
                    elicitation_callback=cast(ElicitationFnT, router),
                ) as session:
                    initialized = await session_module._initialize(
                        session,
                        form_capable=True,
                    )
                    result = await session.call_tool("ask", {})
            finally:
                server_task.cancel()
                with suppress(CancelledError):
                    await server_task

        self.assertEqual(
            initialized.protocolVersion,
            session_module.MCP_PROTOCOL_VERSION,
        )
        self.assertEqual(result.content[0].text, "Ada")
        self.assertEqual(len(broker.requests), 1)


class McpOptionalDependencyBoundaryTestCase(IsolatedAsyncioTestCase):
    def test_mcp_tool_module_imports_without_optional_sdk(self) -> None:
        script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "mcp" or name.startswith("mcp."):
        raise ModuleNotFoundError("blocked optional MCP SDK", name=name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import avalan.tool.mcp
"""
        completed = run(
            [executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    async def test_missing_sdk_fails_only_when_call_is_executed(self) -> None:
        error = ModuleNotFoundError("No module named 'mcp'", name="mcp")
        with patch.object(
            mcp_module,
            "import_module",
            side_effect=error,
        ):
            with self.assertRaisesRegex(RuntimeError, "optional 'server'"):
                await mcp_module._call_initialized_mcp_tool()
        other = ModuleNotFoundError("No module named 'other'", name="other")
        with patch.object(
            mcp_module,
            "import_module",
            side_effect=other,
        ):
            with self.assertRaises(ModuleNotFoundError):
                await mcp_module._call_initialized_mcp_tool()

    async def test_proxy_invokes_lazily_loaded_session(self) -> None:
        call = AsyncMock(return_value={"content": []})
        module = SimpleNamespace(call_initialized_mcp_tool=call)
        with patch.object(
            mcp_module,
            "import_module",
            return_value=module,
        ):
            result = await mcp_module._call_initialized_mcp_tool(
                uri="memory://server"
            )
        self.assertEqual(result, {"content": []})
        call.assert_awaited_once_with(uri="memory://server")


class McpSessionTransportTestCase(IsolatedAsyncioTestCase):
    async def test_call_uses_initialized_session_and_capability(self) -> None:
        check_cancelled = AsyncMock()
        context = ToolCallContext(cancellation_checker=check_cancelled)
        client = object()
        client_manager = AsyncMock()
        client_manager.__aenter__.return_value = client
        streams_manager = AsyncMock()
        streams_manager.__aenter__.return_value = (
            object(),
            object(),
            "session",
        )
        session = AsyncMock()
        session.call_tool.return_value = mcp_types.CallToolResult(content=[])
        session_manager = AsyncMock()
        session_manager.__aenter__.return_value = session
        initialized = SimpleNamespace(
            capabilities=SimpleNamespace(tools=object())
        )
        initialize = AsyncMock(return_value=initialized)

        with (
            patch.object(
                session_module,
                "AsyncClient",
                return_value=client_manager,
            ),
            patch.object(
                session_module,
                "streamable_http_client",
                return_value=streams_manager,
            ) as stream_factory,
            patch.object(
                session_module,
                "ClientSession",
                return_value=session_manager,
            ),
            patch.object(
                session_module,
                "_initialize",
                initialize,
            ),
        ):
            result = await session_module.call_initialized_mcp_tool(
                uri="https://server.example/mcp",
                name="remote",
                arguments={"value": 1},
                context=context,
                client_params={},
                call_params={"timeout": 1, "request_id": "request"},
                progress_callback=AsyncMock(),
                logging_callback=AsyncMock(),
                message_handler=AsyncMock(),
            )
            self.assertEqual(result, {"content": []})
            stream_factory.assert_called_once_with(
                "https://server.example/mcp",
                http_client=client,
            )
            session.call_tool.assert_awaited_once()

            initialized.capabilities.tools = None
            with self.assertRaisesRegex(RuntimeError, "negotiate tools"):
                await session_module.call_initialized_mcp_tool(
                    uri="https://server.example/mcp",
                    name="remote",
                    arguments={},
                    context=context,
                    client_params={},
                    call_params={"timeout": 1},
                    progress_callback=AsyncMock(),
                    logging_callback=AsyncMock(),
                    message_handler=AsyncMock(),
                )

            initialized.capabilities.tools = object()
            await session_module.call_initialized_mcp_tool(
                uri="https://server.example/mcp",
                name="remote",
                arguments={},
                context=ToolCallContext(),
                client_params={},
                call_params={"timeout": 1},
                progress_callback=AsyncMock(),
                logging_callback=AsyncMock(),
                message_handler=AsyncMock(),
            )

        self.assertEqual(check_cancelled.await_count, 3)

    async def test_call_timeout_is_bounded(self) -> None:
        client_manager = AsyncMock()

        async def block() -> object:
            await Event().wait()
            return object()

        client_manager.__aenter__.side_effect = block
        with patch.object(
            session_module,
            "AsyncClient",
            return_value=client_manager,
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await session_module.call_initialized_mcp_tool(
                    uri="https://server.example/mcp",
                    name="remote",
                    arguments={},
                    context=ToolCallContext(),
                    client_params={},
                    call_params={"timeout": 0.001},
                    progress_callback=AsyncMock(),
                    logging_callback=AsyncMock(),
                    message_handler=AsyncMock(),
                )
