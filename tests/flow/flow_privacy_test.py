from collections.abc import Mapping
from unittest import IsolatedAsyncioTestCase, TestCase, main

from async_helpers import run_async

from avalan.event import Event
from avalan.flow import (
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowDiagnosticSeverity,
    FlowExecutionPlan,
    FlowExecutionTrace,
    FlowInputDefinition,
    FlowInputType,
    FlowLoadIssue,
    FlowLoadIssueCategory,
    FlowMappingKind,
    FlowMappingPlan,
    FlowNodeKind,
    FlowNodePlan,
    FlowNodeState,
    FlowNodeTrace,
    FlowOutputDefinition,
    FlowOutputType,
    FlowPlanExecutionResult,
    FlowSourceSpan,
    InMemoryFlowStateStore,
    execute_flow_plan,
    export_sanitized_flow_trace,
    loads_flow_definition_result,
    parse_flow_selector,
)

_async_loads_flow_definition_result = loads_flow_definition_result


def loads_flow_definition_result(
    *args: object,
    **kwargs: object,
) -> object:
    return run_async(_async_loads_flow_definition_result(*args, **kwargs))


_SENSITIVE_VALUES = (
    "private-toml-password",
    "prompt: close account 821",
    "raw-file-bytes-marker",
    "sk-live-secret-token",
    "provider-body-card-number",
    "token-text-marker",
    "model-output-private-summary",
    "/private/customer/source.toml",
    "private-output-selection",
    "private-record-output",
)


class FlowMalformedInputPrivacyTestCase(TestCase):
    def test_malformed_toml_public_diagnostics_omit_source_values(
        self,
    ) -> None:
        load_result = loads_flow_definition_result(
            """
            [flow]
            name = "malformed-privacy"
            private = "private-toml-password"
            prompt = "prompt: close account 821"
            raw_bytes = "raw-file-bytes-marker"
            secret = "sk-live-secret-token"
            provider_body = "provider-body-card-number"
            token_text = "token-text-marker"
            model_output = "model-output-private-summary"
            broken = "unterminated
            """,
            source_path="/private/customer/source.toml",
        )

        self.assertFalse(load_result.ok)
        codes = [
            diagnostic["code"] for diagnostic in load_result.public_diagnostics
        ]
        self.assertEqual(
            codes,
            ["flow.malformed_toml"],
        )
        _assert_no_sensitive_values(self, load_result.public_diagnostics)


class FlowExecutionFailurePrivacyTestCase(IsolatedAsyncioTestCase):
    async def test_execution_failure_public_surfaces_omit_runtime_values(
        self,
    ) -> None:
        plan = FlowExecutionPlan(
            name="execution-privacy",
            version="2026-06-08",
            revision=None,
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.TEXT,
                ),
            ),
            entry_node="worker",
            output_selectors={
                "answer": parse_flow_selector("worker.value.answer"),
            },
            nodes=(
                FlowNodePlan(
                    name="worker",
                    type="select",
                    kind=FlowNodeKind.SELECT,
                    mappings=(
                        FlowMappingPlan(
                            target="value",
                            kind=FlowMappingKind.SELECT,
                            source=parse_flow_selector("inputs.payload"),
                        ),
                    ),
                ),
            ),
        )
        events: list[Event] = []

        def runner(
            _: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            raise RuntimeError(f"provider failed: {inputs!r}")

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={
                "payload": {
                    "private_toml": "private-toml-password",
                    "prompt": "prompt: close account 821",
                    "raw_file": b"raw-file-bytes-marker",
                    "secret": "sk-live-secret-token",
                    "provider_body": {
                        "error": "provider-body-card-number",
                    },
                    "token_text": "token-text-marker",
                    "model_output": "model-output-private-summary",
                },
            },
            event_listener=events.append,
        )
        exported = export_sanitized_flow_trace(result, plan=plan)

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic["code"] for diagnostic in result.public_diagnostics],
            [
                "flow.execution.node_failed",
                "flow.execution.missing_failure_route",
                "flow.execution.missing_output",
            ],
        )
        self.assertEqual(result.outputs, {})
        _assert_no_sensitive_values(self, result.public_diagnostics)
        _assert_no_sensitive_values(self, exported)
        _assert_no_sensitive_values(
            self,
            tuple(event.payload for event in events),
        )

    async def test_output_selection_failure_public_surfaces_are_safe(
        self,
    ) -> None:
        plan = FlowExecutionPlan(
            name="output-privacy",
            version="2026-06-08",
            revision=None,
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.TEXT,
                ),
            ),
            entry_node="worker",
            output_selectors={
                "answer": parse_flow_selector("worker.missing"),
            },
            nodes=(
                FlowNodePlan(
                    name="worker",
                    type="select",
                    kind=FlowNodeKind.SELECT,
                    mappings=(
                        FlowMappingPlan(
                            target="value",
                            kind=FlowMappingKind.SELECT,
                            source=parse_flow_selector("inputs.payload"),
                        ),
                    ),
                ),
            ),
        )
        events: list[Event] = []

        def runner(
            _: FlowNodePlan,
            inputs: Mapping[str, object],
        ) -> object:
            return {
                "value": {
                    "private": inputs["value"],
                    "marker": "private-output-selection",
                }
            }

        result = await execute_flow_plan(
            plan,
            runner,
            inputs={"payload": "private-output-selection"},
            event_listener=events.append,
        )
        exported = export_sanitized_flow_trace(result, plan=plan)

        self.assertFalse(result.ok)
        self.assertEqual(result.outputs, {})
        self.assertEqual(
            result.public_diagnostics,
            (
                {
                    "code": "flow.execution.missing_output",
                    "category": "execution",
                    "severity": "error",
                    "message": "Flow output selection failed.",
                    "path": "flow.output_behavior.outputs.answer",
                    "hint": "Select an output produced by an executed node.",
                },
            ),
        )
        _assert_no_sensitive_values(self, result.public_diagnostics)
        _assert_no_sensitive_values(self, exported)
        _assert_no_sensitive_values(
            self,
            tuple(event.payload for event in events),
        )

    async def test_persisted_record_public_projection_uses_safe_diagnostics(
        self,
    ) -> None:
        diagnostic = _private_source_diagnostic()
        trace = FlowExecutionTrace(
            nodes=(
                FlowNodeTrace(
                    node="worker",
                    state=FlowNodeState.FAILED,
                    attempts=1,
                    diagnostics=(diagnostic,),
                ),
            )
        )
        result = FlowPlanExecutionResult(
            trace=trace,
            diagnostics=(diagnostic,),
            outputs={"answer": "private-record-output"},
        )
        store = InMemoryFlowStateStore()

        record = await store.create_flow_execution(
            "run-private",
            trace=trace,
            selected_outputs={"answer": "private-record-output"},
            diagnostics=(diagnostic,),
        )
        snapshot = record.as_snapshot()
        exported_result = export_sanitized_flow_trace(result)
        exported_record = export_sanitized_flow_trace(record)

        self.assertNotIn(
            "/private/customer/source.toml",
            str(snapshot["diagnostics"]),
        )
        self.assertNotIn(
            "/private/customer/source.toml",
            str(snapshot["trace"]),
        )
        _assert_no_sensitive_values(self, result.public_diagnostics)
        _assert_no_sensitive_values(self, exported_result)
        _assert_no_sensitive_values(self, exported_record)


class FlowLoadIssueProjectionPrivacyTestCase(TestCase):
    def test_load_issue_projection_is_canonical_and_value_free(self) -> None:
        issue = FlowLoadIssue(
            code="flow.invalid_value",
            path="nodes.worker.config",
            category=FlowLoadIssueCategory.VALUE,
            message="Flow node configuration is invalid.",
            hint="Use supported configuration fields.",
        )

        public = issue.as_public_diagnostic_dict()

        self.assertEqual(
            public,
            {
                "code": "flow.invalid_value",
                "category": "flow_definition_validation",
                "severity": "error",
                "message": "Flow node configuration is invalid.",
                "path": "nodes.worker.config",
                "hint": "Use supported configuration fields.",
            },
        )
        _assert_no_sensitive_values(self, public)


def _private_source_diagnostic() -> FlowDiagnostic:
    return FlowDiagnostic(
        code="flow.execution.node_failed",
        path="nodes.worker",
        category=FlowDiagnosticCategory.EXECUTION,
        severity=FlowDiagnosticSeverity.ERROR,
        source_span=FlowSourceSpan(
            source="/private/customer/source.toml",
            start_line=3,
            start_column=5,
        ),
        message="Flow node failed.",
        hint="Inspect error routes for this node.",
    )


def _assert_no_sensitive_values(test_case: TestCase, value: object) -> None:
    rendered = str(value)
    for sensitive in _SENSITIVE_VALUES:
        with test_case.subTest(sensitive=sensitive):
            test_case.assertNotIn(sensitive, rendered)


if __name__ == "__main__":
    main()
