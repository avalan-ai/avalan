from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from typing import get_args
from unittest import TestCase, main
from uuid import uuid4

from avalan.entities import (
    PreparedToolCall,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallDiagnosticStatus,
    ToolCallError,
    ToolCallOutcome,
    ToolCallParseOutcome,
    ToolCallResult,
    ToolDescriptor,
    ToolNameResolution,
    ToolNameResolutionStatus,
)


class ToolCallDiagnosticTestCase(TestCase):
    def test_fields_are_stable(self):
        self.assertEqual(
            [field.name for field in fields(ToolCallDiagnostic)],
            [
                "id",
                "call_id",
                "requested_name",
                "canonical_name",
                "status",
                "code",
                "stage",
                "message",
                "retryable",
                "details",
                "started_at",
                "finished_at",
                "duration_ms",
            ],
        )

    def test_codes_cover_non_executed_categories(self):
        self.assertEqual(
            {code.value for code in ToolCallDiagnosticCode},
            {
                "tool.unknown",
                "tool.disabled",
                "tool.ambiguous_name",
                "tool_call.malformed",
                "tool_call.arguments_malformed",
                "tool_call.arguments_invalid",
                "tool_call.policy_suppressed",
                "tool_call.filter_suppressed",
                "tool_call.user_rejected",
                "tool_call.repeated",
                "tool_call.maximum_depth",
                "tool_call.cancelled",
                "tool_call.timeout",
                "tool_call.loop_guard",
                "tool_call.runaway_guard",
            },
        )

    def test_stages_cover_parse_and_execution_preconditions(self):
        self.assertEqual(
            {stage.value for stage in ToolCallDiagnosticStage},
            {
                "parse",
                "resolve",
                "validate",
                "policy",
                "filter",
                "confirm",
                "dispatch",
                "guard",
            },
        )

    def test_create_non_executed_diagnostic(self):
        diagnostic_id = uuid4()
        call_id = uuid4()
        started_at = datetime(2026, 6, 5, 1, 2, 3, tzinfo=UTC)
        finished_at = datetime(2026, 6, 5, 1, 2, 4, tzinfo=UTC)

        diagnostic = ToolCallDiagnostic(
            id=diagnostic_id,
            call_id=call_id,
            requested_name=" calculator ",
            canonical_name="math.calculator",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Tool is not enabled.",
            retryable=True,
            details={"available": ["math.calculator"]},
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=1.5,
        )

        self.assertEqual(diagnostic.id, diagnostic_id)
        self.assertEqual(diagnostic.call_id, call_id)
        self.assertEqual(diagnostic.requested_name, " calculator ")
        self.assertEqual(diagnostic.canonical_name, "math.calculator")
        self.assertIs(diagnostic.status, ToolCallDiagnosticStatus.NON_EXECUTED)
        self.assertTrue(diagnostic.retryable)
        self.assertEqual(diagnostic.details["available"], ["math.calculator"])
        self.assertEqual(diagnostic.started_at, started_at)
        self.assertEqual(diagnostic.finished_at, finished_at)
        self.assertEqual(diagnostic.duration_ms, 1.5)
        self.assertNotIsInstance(diagnostic, ToolCallError)

    def test_default_details_are_independent(self):
        first = ToolCallDiagnostic(
            id="diag-1",
            code=ToolCallDiagnosticCode.MALFORMED_CALL,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Malformed call.",
        )
        second = ToolCallDiagnostic(
            id="diag-2",
            code=ToolCallDiagnosticCode.MALFORMED_CALL,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Malformed call.",
        )

        first.details["source"] = "tag"

        self.assertEqual(second.details, {})

    def test_rejects_invalid_identifiers(self):
        for field_name in ("id", "call_id"):
            with self.subTest(field_name=field_name):
                kwargs = {
                    "id": "diag",
                    "code": ToolCallDiagnosticCode.MALFORMED_CALL,
                    "stage": ToolCallDiagnosticStage.PARSE,
                    "message": "Malformed call.",
                    field_name: "",
                }
                with self.assertRaises(AssertionError):
                    ToolCallDiagnostic(**kwargs)

    def test_rejects_invalid_names(self):
        for field_name in ("requested_name", "canonical_name"):
            with self.subTest(field_name=field_name):
                kwargs = {
                    "id": "diag",
                    "code": ToolCallDiagnosticCode.UNKNOWN_TOOL,
                    "stage": ToolCallDiagnosticStage.RESOLVE,
                    "message": "Tool is unknown.",
                    field_name: " ",
                }
                with self.assertRaises(AssertionError):
                    ToolCallDiagnostic(**kwargs)

    def test_rejects_invalid_enums(self):
        with self.assertRaises(AssertionError):
            ToolCallDiagnostic(
                id="diag",
                code="tool.unknown",
                stage=ToolCallDiagnosticStage.RESOLVE,
                message="Tool is unknown.",
            )
        with self.assertRaises(AssertionError):
            ToolCallDiagnostic(
                id="diag",
                code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
                stage="resolve",
                message="Tool is unknown.",
            )
        with self.assertRaises(AssertionError):
            ToolCallDiagnostic(
                id="diag",
                code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
                stage=ToolCallDiagnosticStage.RESOLVE,
                status="non_executed",
                message="Tool is unknown.",
            )

    def test_rejects_invalid_message_retry_details_and_timing(self):
        invalid_kwargs = (
            {"message": ""},
            {"retryable": 1},
            {"details": []},
            {"started_at": "2026-06-05T00:00:00Z"},
            {"finished_at": "2026-06-05T00:00:01Z"},
            {"duration_ms": True},
            {"duration_ms": -1},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                diagnostic_kwargs = {
                    "id": "diag",
                    "code": ToolCallDiagnosticCode.MALFORMED_CALL,
                    "stage": ToolCallDiagnosticStage.PARSE,
                    "message": "Malformed call.",
                }
                diagnostic_kwargs.update(kwargs)
                with self.assertRaises(AssertionError):
                    ToolCallDiagnostic(**diagnostic_kwargs)


class ToolCallParseOutcomeTestCase(TestCase):
    def test_fields_are_stable(self):
        self.assertEqual(
            [field.name for field in fields(ToolCallParseOutcome)],
            ["calls", "diagnostics"],
        )

    def test_defaults_to_empty_calls_and_diagnostics(self):
        outcome = ToolCallParseOutcome()

        self.assertEqual(outcome.calls, [])
        self.assertEqual(outcome.diagnostics, [])

    def test_create_parse_outcome(self):
        call = ToolCall(id="call-1", name="calculator", arguments={})
        diagnostic = ToolCallDiagnostic(
            id="diag-1",
            call_id="call-2",
            requested_name="calculator",
            code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Arguments must be an object.",
        )

        outcome = ToolCallParseOutcome(
            calls=[call],
            diagnostics=[diagnostic],
        )

        self.assertEqual(outcome.calls, [call])
        self.assertEqual(outcome.diagnostics, [diagnostic])

    def test_rejects_invalid_calls_and_diagnostics(self):
        diagnostic = ToolCallDiagnostic(
            id="diag-1",
            code=ToolCallDiagnosticCode.MALFORMED_CALL,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Malformed call.",
        )
        invalid_cases = (
            {"calls": (ToolCall(id="call-1", name="calculator"),)},
            {"calls": ["call"]},
            {"diagnostics": (diagnostic,)},
            {"diagnostics": ["diagnostic"]},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ToolCallParseOutcome(**kwargs)

    def test_is_frozen(self):
        outcome = ToolCallParseOutcome()

        with self.assertRaises(FrozenInstanceError):
            outcome.calls = []


class ToolDescriptorTestCase(TestCase):
    def test_fields_are_stable(self):
        self.assertEqual(
            [field.name for field in fields(ToolDescriptor)],
            [
                "name",
                "callable",
                "aliases",
                "schema",
                "parameter_schema",
                "return_schema",
                "provider_safe_schema",
                "namespace",
                "policy",
                "metadata",
            ],
        )

    def test_create_descriptor(self):
        def calculator() -> int:
            return 1

        descriptor = ToolDescriptor(
            name="math.calculator",
            callable=calculator,
            aliases=["calc"],
            schema={"type": "function"},
            parameter_schema={"type": "object"},
            return_schema={"type": "integer"},
            provider_safe_schema={"type": "function"},
            namespace="math",
            policy={"confirmation": False},
            metadata={"source": "test"},
        )

        self.assertEqual(descriptor.name, "math.calculator")
        self.assertIs(descriptor.callable, calculator)
        self.assertEqual(descriptor.aliases, ["calc"])
        self.assertEqual(descriptor.schema, {"type": "function"})
        self.assertEqual(descriptor.parameter_schema, {"type": "object"})
        self.assertEqual(descriptor.return_schema, {"type": "integer"})
        self.assertEqual(descriptor.provider_safe_schema, {"type": "function"})
        self.assertEqual(descriptor.namespace, "math")
        self.assertEqual(descriptor.policy, {"confirmation": False})
        self.assertEqual(descriptor.metadata, {"source": "test"})

    def test_rejects_invalid_values(self):
        invalid_cases = (
            {"name": " "},
            {"callable": "calculator"},
            {"aliases": ("calc",)},
            {"aliases": [" "]},
            {"schema": []},
            {"parameter_schema": []},
            {"return_schema": []},
            {"provider_safe_schema": []},
            {"namespace": " "},
            {"policy": []},
            {"metadata": []},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                valid = {"name": "calculator"}
                valid.update(kwargs)
                with self.assertRaises(AssertionError):
                    ToolDescriptor(**valid)


class PreparedToolCallTestCase(TestCase):
    def test_fields_are_stable(self):
        self.assertEqual(
            [field.name for field in fields(PreparedToolCall)],
            ["call", "callable", "descriptor", "arguments", "context"],
        )

    def test_create_prepared_tool_call(self):
        def calculator(expression: str) -> str:
            return expression

        call = ToolCall(
            id="call-1",
            name="calculator",
            arguments={"expression": "1"},
        )
        descriptor = ToolDescriptor(name="calculator", callable=calculator)
        context = ToolCallContext()

        prepared = PreparedToolCall(
            call=call,
            callable=calculator,
            descriptor=descriptor,
            arguments={"expression": "1"},
            context=context,
        )

        self.assertEqual(prepared.call, call)
        self.assertIs(prepared.callable, calculator)
        self.assertEqual(prepared.descriptor, descriptor)
        self.assertEqual(prepared.arguments, {"expression": "1"})
        self.assertEqual(prepared.context, context)

    def test_rejects_invalid_values(self):
        def calculator(expression: str) -> str:
            return expression

        call = ToolCall(
            id="call-1",
            name="calculator",
            arguments={"expression": "1"},
        )
        descriptor = ToolDescriptor(name="calculator", callable=calculator)
        valid = {
            "call": call,
            "callable": calculator,
            "descriptor": descriptor,
            "arguments": {"expression": "1"},
            "context": ToolCallContext(),
        }
        invalid_cases = (
            {"call": "call"},
            {"callable": "calculator"},
            {"descriptor": "descriptor"},
            {"arguments": [("expression", "1")]},
            {"context": "context"},
            {
                "call": ToolCall(
                    id="call-1",
                    name="other",
                    arguments={"expression": "1"},
                )
            },
            {"arguments": {"expression": "2"}},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                candidate = dict(valid)
                candidate.update(kwargs)
                with self.assertRaises(AssertionError):
                    PreparedToolCall(**candidate)


class ToolNameResolutionTestCase(TestCase):
    def test_statuses_are_stable(self):
        self.assertEqual(
            {status.value for status in ToolNameResolutionStatus},
            {"exact", "alias", "ambiguous", "unknown", "disabled"},
        )

    def test_create_resolution(self):
        resolution = ToolNameResolution(
            requested_name="calc",
            status=ToolNameResolutionStatus.ALIAS,
            canonical_name="calculator",
            candidates=["calculator"],
            diagnostic_code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )

        self.assertEqual(resolution.requested_name, "calc")
        self.assertIs(resolution.status, ToolNameResolutionStatus.ALIAS)
        self.assertEqual(resolution.canonical_name, "calculator")
        self.assertEqual(resolution.candidates, ["calculator"])
        self.assertIs(
            resolution.diagnostic_code, ToolCallDiagnosticCode.UNKNOWN_TOOL
        )

    def test_rejects_invalid_values(self):
        invalid_cases = (
            {"requested_name": " "},
            {"status": "exact"},
            {"canonical_name": " "},
            {"candidates": ("calculator",)},
            {"candidates": [" "]},
            {"diagnostic_code": "tool.unknown"},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                valid = {
                    "requested_name": "calculator",
                    "status": ToolNameResolutionStatus.EXACT,
                }
                valid.update(kwargs)
                with self.assertRaises(AssertionError):
                    ToolNameResolution(**valid)


class ToolCallOutcomeTestCase(TestCase):
    def test_public_outcome_union(self):
        self.assertEqual(
            set(get_args(ToolCallOutcome)),
            {ToolCallResult, ToolCallError, ToolCallDiagnostic},
        )


if __name__ == "__main__":
    main()
