from os import environ
from pathlib import Path
from shutil import which
from subprocess import run
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from async_helpers import run_async

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallOutcome,
    ToolCallResult,
    ToolDescriptor,
    ToolManagerSettings,
    ToolNameResolution,
    ToolNameResolutionStatus,
)
from avalan.flow import (
    FLOW_TOOL_NODE_TYPE,
    FlowCondition,
    FlowConditionOperator,
    FlowConditionValueType,
    FlowDefinition,
    FlowDiagnostic,
    FlowDiagnosticCategory,
    FlowEdgeDefinition,
    FlowEdgeKind,
    FlowEntryBehavior,
    FlowExecutor,
    FlowInputDefinition,
    FlowInputMapping,
    FlowInputType,
    FlowJoinPolicy,
    FlowJoinPolicyType,
    FlowLoopPolicy,
    FlowMappingKind,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputBehavior,
    FlowOutputDefinition,
    FlowOutputType,
    FlowRetryBackoffStrategy,
    FlowRetryPolicy,
    FlowRouteMatchPolicy,
    FlowTimeoutPolicy,
    FlowValidationResult,
    parse_flow_selector,
    tool_flow_node_registry,
    validate_flow_definition,
)
from avalan.flow import loader as flow_loader
from avalan.flow import validator as flow_validator
from avalan.flow.node import Node
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    ShellGitToolSettings,
    ShellToolSet,
    ShellToolSettings,
)


async def validator_flow_adder(a: int, b: int) -> int:
    return a + b


validator_flow_adder.aliases = ["sum"]  # type: ignore[attr-defined]


async def validator_flow_adder_alt(a: int, b: int) -> int:
    return a + b


validator_flow_adder_alt.aliases = ["sum"]  # type: ignore[attr-defined]


async def validator_flow_status() -> str:
    return "ready"


async def validator_flow_disabled(value: int) -> int:
    return value


def _write_shell_git_example_repo(root: Path, git_binary: str) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _run_git(git_binary, "init", cwd=repo)
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    _run_git(git_binary, "add", "README.md", cwd=repo)
    _run_git(
        git_binary,
        "-c",
        "user.name=Avalan Test",
        "-c",
        "user.email=avalan@example.test",
        "commit",
        "-m",
        "initial commit",
        cwd=repo,
    )
    (repo / "README.md").write_text("initial\nupdated\n", encoding="utf-8")
    return repo


def _run_git(git_binary: str, *args: str, cwd: Path) -> None:
    git_env = dict(environ)
    git_env.update(
        {
            "GIT_CONFIG_GLOBAL": "/nonexistent",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    run(
        (git_binary, *args),
        cwd=cwd,
        env=git_env,
        check=True,
        capture_output=True,
        text=True,
    )


def _tool_manager(
    *,
    enable_tools: list[str] | None = None,
) -> ToolManager:
    return ToolManager.create_instance(
        enable_tools=enable_tools
        or ["validator_flow_adder", "validator_flow_status"],
        available_toolsets=[
            ToolSet(
                tools=[
                    validator_flow_adder,
                    validator_flow_adder_alt,
                    validator_flow_status,
                ]
            ),
            ToolSet(namespace="disabled", tools=[validator_flow_disabled]),
        ],
        settings=ToolManagerSettings(),
    )


class StaticToolResolver:
    def __init__(
        self,
        descriptors: list[ToolDescriptor],
        *,
        disabled_names: set[str] | None = None,
    ) -> None:
        self.descriptors = descriptors
        self.disabled_names = disabled_names or set()

    def list_tools(self) -> list[ToolDescriptor]:
        return self.descriptors

    def resolve_tool_name(
        self, name: str, *, provider_originated: bool = False
    ) -> ToolNameResolution:
        _ = provider_originated
        if name in self.disabled_names:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.DISABLED,
                candidates=[name],
            )
        names = {descriptor.name for descriptor in self.descriptors}
        if name in names:
            return ToolNameResolution(
                requested_name=name,
                status=ToolNameResolutionStatus.EXACT,
                canonical_name=name,
                candidates=[name],
            )
        return ToolNameResolution(
            requested_name=name,
            status=ToolNameResolutionStatus.UNKNOWN,
        )

    def validate_tool_call(self, call: ToolCall) -> ToolCallDiagnostic | None:
        _ = call
        return None

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> ToolCallOutcome:
        _ = context
        return ToolCallResult(
            id="result-1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=call.arguments or {},
        )


class FlowValidatorTestCase(TestCase):
    def _strict_definition(
        self,
        *,
        nodes: tuple[FlowNodeDefinition, ...],
        output_selector: str = "start.result",
    ) -> FlowDefinition:
        return FlowDefinition(
            name="strict",
            version="2026-06-07",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node=nodes[0].name),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": output_selector},
            ),
            nodes=nodes,
        )

    def test_flow_validation_load_category_marks_privacy(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.safe",
            category=FlowDiagnosticCategory.PRIVACY,
            message="Private value was rejected.",
            path="nodes.start.ref",
        )

        self.assertEqual(
            flow_validator.flow_validation_diagnostic_load_category(
                diagnostic
            ),
            "privacy",
        )

    def test_validate_flow_definition_accepts_valid_definition(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="valid",
                entrypoint="start",
                output_node="finish",
                nodes=(
                    FlowNodeDefinition(name="start", type="input"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(FlowEdgeDefinition(source="start", target="finish"),),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual(result.public_diagnostics, ())

    def test_validate_flow_definition_rejects_builtin_refs(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="invalid",
                entrypoint="start",
                output_node="start",
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type="input",
                        ref="local_ref",
                    ),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.untrusted_callable")
        self.assertEqual(result.diagnostics[0].path, "nodes.start.ref")

    def test_validate_flow_definition_accepts_strict_definition(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "finish.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="prepare", type="input"),
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(source="prepare", target="start"),
                    FlowEdgeDefinition(source="start", target="finish"),
                ),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())

    def test_validate_flow_definition_accepts_revision_identity(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                revision="rev-1",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "start.result"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_unknown_edge_refs(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="invalid",
                entrypoint="start",
                output_node="start",
                nodes=(FlowNodeDefinition(name="start", type="input"),),
                edges=(
                    FlowEdgeDefinition(
                        source="missing_source",
                        target="missing_target",
                    ),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.path for diagnostic in result.diagnostics],
            ["edges.source", "edges.target"],
        )
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.bad_reference", "flow.bad_reference"],
        )

    def test_validate_flow_definition_rejects_legacy_topology_gaps(
        self,
    ) -> None:
        cases = (
            (
                FlowDefinition(
                    name="legacy",
                    entrypoint="missing",
                    output_node="finish",
                    nodes=(
                        FlowNodeDefinition(name="start", type="input"),
                        FlowNodeDefinition(name="finish", type="echo"),
                    ),
                    edges=(
                        FlowEdgeDefinition(source="start", target="finish"),
                    ),
                ),
                "flow.unknown_entrypoint",
            ),
            (
                FlowDefinition(
                    name="legacy",
                    entrypoint="middle",
                    output_node="finish",
                    nodes=(
                        FlowNodeDefinition(name="start", type="input"),
                        FlowNodeDefinition(name="middle", type="echo"),
                        FlowNodeDefinition(name="finish", type="echo"),
                    ),
                    edges=(
                        FlowEdgeDefinition(source="start", target="middle"),
                        FlowEdgeDefinition(source="middle", target="finish"),
                    ),
                ),
                "flow.invalid_entrypoint",
            ),
            (
                FlowDefinition(
                    name="legacy",
                    entrypoint="start",
                    output_node="finish",
                    nodes=(
                        FlowNodeDefinition(name="start", type="input"),
                        FlowNodeDefinition(name="finish", type="echo"),
                        FlowNodeDefinition(name="other", type="echo"),
                    ),
                    edges=(
                        FlowEdgeDefinition(source="start", target="finish"),
                    ),
                ),
                "flow.multiple_outputs",
            ),
            (
                FlowDefinition(
                    name="legacy",
                    entrypoint="start",
                    output_node="middle",
                    nodes=(
                        FlowNodeDefinition(name="start", type="input"),
                        FlowNodeDefinition(name="middle", type="echo"),
                        FlowNodeDefinition(name="finish", type="echo"),
                    ),
                    edges=(
                        FlowEdgeDefinition(source="start", target="middle"),
                        FlowEdgeDefinition(source="middle", target="finish"),
                    ),
                ),
                "flow.invalid_output_node",
            ),
        )

        for definition, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(definition)

                self.assertFalse(result.ok)
                self.assertIn(code, [item.code for item in result.diagnostics])

    def test_validate_flow_definition_rejects_strict_contract_gaps(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                runtime_limits={"timeout_seconds": 30},
                input=FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
                output=FlowOutputDefinition(
                    name="result",
                    type=FlowOutputType.OBJECT,
                ),
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.STRING,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.JSON,
                    ),
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.missing_identity",
                "flow.scalar_input_alias",
                "flow.scalar_output_alias",
                "flow.duplicate_input",
                "flow.duplicate_output",
                "flow.missing_entry_behavior",
                "flow.missing_output_behavior",
            ],
        )

    def test_validate_flow_definition_rejects_unknown_strict_references(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                    FlowOutputDefinition(
                        name="audit",
                        type=FlowOutputType.JSON,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="missing"),
                output_behavior=FlowOutputBehavior(
                    outputs={
                        "answer": "missing.result",
                        "extra": "start.result",
                    },
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.unknown_entry_node",
                "flow.unknown_output",
                "flow.missing_output_selection",
                "flow.unknown_output_selector_node",
            ],
        )

    def test_validate_flow_definition_rejects_strict_aliases(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                entrypoint="start",
                output_node="start",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "start.value"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.entrypoint_alias", "flow.output_node_alias"],
        )

    def test_validate_flow_definition_rejects_output_contract_mismatch(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"typed": lambda definition: Node(definition.name)},
            {
                "typed": FlowNodeMetadata(
                    kind=FlowNodeKind.PASS_THROUGH,
                    output_contract=FlowNodeContract(
                        name="result",
                        type=FlowOutputType.TEXT,
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "start.result"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="typed"),),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.incompatible_output_selection"],
        )
        self.assertEqual(
            result.diagnostics[0].path,
            "flow.output_behavior.outputs.answer",
        )

    def test_validate_flow_definition_requires_explicit_output_map(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                    FlowOutputDefinition(
                        name="audit",
                        type=FlowOutputType.JSON,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.missing_output_behavior"],
        )

    def test_validate_flow_definition_accepts_non_terminal_output_selection(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "start.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(FlowEdgeDefinition(source="start", target="finish"),),
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.diagnostics, ())

    def test_validate_flow_definition_rejects_invalid_output_selector(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "start"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.invalid_output_selector"],
        )

    def test_validate_flow_definition_accepts_nested_output_selector(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"schema": lambda definition: Node(definition.name)},
            {
                "schema": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(FlowNodeDefinition(name="start", type="schema"),),
                output_selector="start.result.items[0].name",
            ),
            registry,
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_unsafe_output_selectors(
        self,
    ) -> None:
        cases = (
            ("answer", "env.SECRET", "flow.reserved_selector"),
            ("answer", "start.result/{{secret}}", "flow.unsafe_selector"),
        )

        for output, selector, code in cases:
            with self.subTest(selector=selector):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="strict",
                        version="2026-06-07",
                        inputs=(
                            FlowInputDefinition(
                                name="payload",
                                type=FlowInputType.OBJECT,
                            ),
                        ),
                        outputs=(
                            FlowOutputDefinition(
                                name=output,
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="start"),
                        output_behavior=FlowOutputBehavior(
                            outputs={output: selector},
                        ),
                        nodes=(FlowNodeDefinition(name="start", type="echo"),),
                    )
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)
                self.assertEqual(
                    result.diagnostics[0].category,
                    FlowDiagnosticCategory.PRIVACY,
                )
                self.assertNotIn("secret", str(result.public_diagnostics))

    def test_validate_flow_definition_rejects_unknown_selector_path(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"schema": lambda definition: Node(definition.name)},
            {
                "schema": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": "object",
                            "properties": {
                                "known": {"type": "string"},
                                "items": {"type": "array"},
                            },
                        },
                    ),
                ),
            },
        )

        cases = (
            "start.result.missing",
            "start.result.known[0]",
            "start.result.known.name",
        )
        for selector in cases:
            with self.subTest(selector=selector):
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(
                            FlowNodeDefinition(name="start", type="schema"),
                        ),
                        output_selector=selector,
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(
                    result.diagnostics[-1].code,
                    "flow.unknown_selector_path",
                )

    def test_validate_flow_definition_accepts_unknown_nested_schema(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"schema": lambda definition: Node(definition.name)},
            {
                "schema": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": ["object", "null"],
                            "properties": {"items": {"type": "array"}},
                        },
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(FlowNodeDefinition(name="start", type="schema"),),
                output_selector="start.result.items[0].name",
            ),
            registry,
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_accepts_open_selector_contracts(
        self,
    ) -> None:
        cases = (
            (
                FlowNodeMetadata(kind=FlowNodeKind.SELECT),
                "start.result.any[0]",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        metadata={"dynamic": True},
                    ),
                ),
                "start.result.any",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema_ref="schemas/result.json",
                    ),
                ),
                "start.result.any",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(name="result"),
                ),
                "start.result.any",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={"type": "object"},
                    ),
                ),
                "start.result.any",
            ),
            (
                FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={"type": 7},
                    ),
                ),
                "start.result.any",
            ),
        )

        for metadata, selector in cases:
            with self.subTest(selector=selector, metadata=metadata):
                registry = FlowNodeRegistry(
                    {"open": lambda definition: Node(definition.name)},
                    {"open": metadata},
                )
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(FlowNodeDefinition(name="start", type="open"),),
                        output_selector=selector,
                    ),
                    registry,
                )

                self.assertTrue(result.ok)

        selector = parse_flow_selector("start.result.any")
        self.assertFalse(
            flow_validator._supports_output_path(  # type: ignore[attr-defined]
                FlowNodeRegistry(
                    {"open": lambda definition: Node(definition.name)},
                    {
                        "open": FlowNodeMetadata(
                            kind=FlowNodeKind.SELECT,
                            output_contract=FlowNodeContract(name="other"),
                        ),
                    },
                ),
                "open",
                selector,
            )
        )

    def test_validate_flow_definition_rejects_agent_file_selector_contracts(
        self,
    ) -> None:
        cases = (
            (
                "render.missing",
                "flow.unknown_node_output",
                FlowNodeContract(name="files"),
            ),
            (
                "render.files.missing",
                "flow.unknown_selector_path",
                FlowNodeContract(
                    name="files",
                    schema={
                        "type": "object",
                        "properties": {"known": {"type": "string"}},
                    },
                ),
            ),
        )

        for selector, code, output_contract in cases:
            with self.subTest(selector=selector):
                registry = FlowNodeRegistry(
                    {
                        "agent": lambda definition: Node(definition.name),
                        "render": lambda definition: Node(definition.name),
                    },
                    {
                        "agent": FlowNodeMetadata(kind=FlowNodeKind.AGENT),
                        "render": FlowNodeMetadata(
                            kind=FlowNodeKind.FILE_CONVERSION,
                            output_contract=output_contract,
                        ),
                    },
                )
                result = validate_flow_definition(
                    FlowDefinition(
                        name="files",
                        entrypoint="render",
                        output_node="agent",
                        nodes=(
                            FlowNodeDefinition(name="render", type="render"),
                            FlowNodeDefinition(
                                name="agent",
                                type="agent",
                                config={"files_input": selector},
                            ),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="render",
                                target="agent",
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)

    def test_validate_flow_definition_rejects_agent_file_selector_shapes(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "agent": lambda definition: Node(definition.name),
                "bridge": lambda definition: Node(definition.name),
                "render": lambda definition: Node(definition.name),
            },
            {
                "agent": FlowNodeMetadata(kind=FlowNodeKind.AGENT),
                "bridge": FlowNodeMetadata(kind=FlowNodeKind.PASS_THROUGH),
                "render": FlowNodeMetadata(
                    kind=FlowNodeKind.FILE_CONVERSION,
                    output_contract=FlowNodeContract(name="files"),
                ),
            },
        )
        cases = (
            ("", "flow.invalid_type"),
            ("env.SECRET", "flow.reserved_selector"),
            ("missing.files", "flow.bad_reference"),
            ("render.files", "flow.bad_reference"),
        )

        for selector, code in cases:
            with self.subTest(selector=selector):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="files",
                        entrypoint="render",
                        output_node="agent",
                        nodes=(
                            FlowNodeDefinition(name="render", type="render"),
                            FlowNodeDefinition(name="bridge", type="bridge"),
                            FlowNodeDefinition(
                                name="agent",
                                type="agent",
                                config={"files_input": selector},
                            ),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="render",
                                target="bridge",
                            ),
                            FlowEdgeDefinition(
                                source="bridge",
                                target="agent",
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertIn(code, [item.code for item in result.diagnostics])

    def test_validate_flow_definition_rejects_topology_output_inference(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                output_node="finish",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(FlowEdgeDefinition(source="start", target="finish"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.output_node_alias", "flow.missing_output_behavior"],
        )

    def test_validation_result_requires_diagnostic_tuple(self) -> None:
        diagnostic = FlowDiagnostic(
            code="flow.invalid_entrypoint",
            path="flow.entrypoint",
            category=FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
            message="Flow entrypoint is invalid.",
        )

        result = FlowValidationResult(diagnostics=(diagnostic,))

        self.assertFalse(result.ok)
        self.assertEqual(
            result.public_diagnostics,
            (diagnostic.as_public_dict(),),
        )
        with self.assertRaises(AssertionError):
            FlowValidationResult(
                diagnostics=[diagnostic],  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            FlowValidationResult(
                diagnostics=(object(),),  # type: ignore[arg-type]
            )

    def test_validate_flow_definition_rejects_node_type_cases(self) -> None:
        registry = FlowNodeRegistry(
            {"external": lambda definition: Node(definition.name)},
            {"external": FlowNodeMetadata(supports_ref=True)},
        )
        cases = (
            ("unknown", "flow.unknown_node_type", "nodes.start.type"),
            ("agent", "flow.unsupported_node_type", "nodes.start.type"),
            ("python", "flow.untrusted_callable", "nodes.start.type"),
        )

        for node_type, code, path in cases:
            with self.subTest(node_type=node_type):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="invalid",
                        entrypoint="start",
                        output_node="start",
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type=node_type,
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)
                self.assertEqual(result.diagnostics[0].path, path)

    def test_validate_flow_definition_uses_node_metadata_contracts(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"tool": lambda definition: Node(definition.name)},
            {
                "tool": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    supports_ref=True,
                    async_only=True,
                    output_contract=FlowNodeContract(name="result"),
                    capabilities=(FlowNodeCapability.ASYNC_ONLY,),
                    requires_ref=True,
                    required_config_keys=("mode",),
                ),
            },
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type="tool",
                        ref="weather",
                        config={"mode": "safe"},
                    ),
                )
            ),
            registry,
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_accepts_strict_tool_refs(
        self,
    ) -> None:
        registry = tool_flow_node_registry(_tool_manager())
        cases = (
            (
                "validator_flow_adder",
                {"arguments": {"a": "left", "b": "right"}},
            ),
            ("sum", {"arguments": {"a": "left", "b": "right"}}),
            ("validator_flow_status", {}),
        )

        for ref, config in cases:
            with self.subTest(ref=ref):
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type=FLOW_TOOL_NODE_TYPE,
                                ref=ref,
                                config=config,
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertTrue(result.ok, result.public_diagnostics)

    def test_validate_flow_definition_accepts_nested_tool_literals(
        self,
    ) -> None:
        descriptor = ToolDescriptor(
            name="shell.pipeline",
            parameter_schema={
                "type": "object",
                "properties": {
                    "steps": {"type": "array"},
                    "max_stdout_bytes": {"type": "integer"},
                },
                "required": ["steps"],
            },
        )
        registry = tool_flow_node_registry(StaticToolResolver([descriptor]))

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="shell.pipeline",
                        config={
                            "arguments": {
                                "steps": [
                                    {
                                        "id": "read",
                                        "command": "cat",
                                        "paths": ["visible.txt"],
                                    },
                                    {
                                        "id": "count",
                                        "command": "wc",
                                        "stdin_from": {
                                            "step_id": "read",
                                            "stream": "stdout",
                                        },
                                    },
                                ],
                                "max_stdout_bytes": 1024,
                            }
                        },
                    ),
                )
            ),
            registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)

    def test_docs_shell_pipeline_flow_examples_validate_with_runtime(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["shell.pipeline"],
            available_toolsets=[
                ShellToolSet(settings=ShellToolSettings(allow_pipelines=True))
            ],
            settings=ToolManagerSettings(),
        )
        loader = flow_loader.FlowDefinitionLoader(
            tool_flow_node_registry(manager)
        )
        root = Path(__file__).resolve().parents[2]
        examples = (
            root / "docs" / "examples" / "flows" / "shell_pipeline.flow.toml",
            root / "docs" / "examples" / "tasks" / "pipeline_flow.flow.toml",
        )

        for example_path in examples:
            with self.subTest(example=example_path.name):
                result = run_async(loader.load_validation_result(example_path))

                self.assertTrue(result.ok, result.public_diagnostics)
                assert result.definition is not None
                self.assertEqual(
                    result.definition.nodes[0].ref,
                    "shell.pipeline",
                )

    def test_docs_shell_git_flow_example_validates_and_runs(
        self,
    ) -> None:
        git_binary = which("git")
        if git_binary is None:
            self.skipTest("git executable is required for shell Git examples")

        root = Path(__file__).resolve().parents[2]
        example_path = (
            root
            / "docs"
            / "examples"
            / "flows"
            / "shell_git_readonly.flow.toml"
        )
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            repo = _write_shell_git_example_repo(workspace, git_binary)
            manager = ToolManager.create_instance(
                enable_tools=[
                    "shell.git_status",
                    "shell.git_diff",
                    "shell.git_log",
                ],
                available_toolsets=[
                    ShellToolSet(
                        settings=ShellToolSettings(
                            executable_search_paths=(
                                str(Path(git_binary).parent),
                            ),
                            git=ShellGitToolSettings(
                                workspace_root=str(workspace),
                                cwd=repo.name,
                                capabilities=("read",),
                                allowed_commands=("status", "diff", "log"),
                            ),
                        )
                    )
                ],
                settings=ToolManagerSettings(),
            )
            loader = flow_loader.FlowDefinitionLoader(
                tool_flow_node_registry(manager)
            )

            result = run_async(loader.load_result(example_path))

            self.assertTrue(result.ok, result.public_diagnostics)
            assert result.definition is not None
            self.assertEqual(
                [node.ref for node in result.definition.nodes if node.ref],
                ["shell.git_status", "shell.git_diff", "shell.git_log"],
            )
            run_result = run_async(
                FlowExecutor(registry=tool_flow_node_registry(manager)).run(
                    result.definition
                )
            )

        self.assertTrue(run_result.ok, run_result.public_diagnostics)
        output = run_result.outputs
        self.assertIn("status", output)
        self.assertIn("diff", output)
        self.assertIn("log", output)
        self.assertIn("status: success", output["status"])
        self.assertIn("status: success", output["diff"])
        self.assertIn("status: success", output["log"])
        self.assertIn("updated", output["diff"])
        self.assertIn("initial commit", output["log"])

    def test_validate_flow_definition_accepts_shell_git_read_refs(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=[
                "shell.git_status",
                "shell.git_diff",
                "shell.git_log",
            ],
            available_toolsets=[
                ShellToolSet(
                    settings=ShellToolSettings(
                        git=ShellGitToolSettings(
                            capabilities=("read",),
                            allowed_commands=("status", "diff", "log"),
                        )
                    )
                )
            ],
        )
        registry = tool_flow_node_registry(manager)
        cases = (
            ("shell.git_status", {"arguments": {}}),
            (
                "shell.git_diff",
                {
                    "arguments": {
                        "mode": "staged",
                        "paths": ["src/avalan/tool"],
                    }
                },
            ),
            (
                "shell.git_log",
                {"arguments": {"max_count": 3, "format": "oneline"}},
            ),
        )

        for ref, config in cases:
            with self.subTest(ref=ref):
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type=FLOW_TOOL_NODE_TYPE,
                                ref=ref,
                                config=config,
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertTrue(result.ok, result.public_diagnostics)

    def test_validate_flow_definition_rejects_shell_git_capability_bypass(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["shell.git_commit"],
            available_toolsets=[
                ShellToolSet(
                    settings=ShellToolSettings(
                        git=ShellGitToolSettings(
                            capabilities=("read",),
                            allowed_commands=("commit",),
                        )
                    )
                )
            ],
        )
        registry = tool_flow_node_registry(manager)

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="shell.git_commit",
                        config={"arguments": {"message": "blocked"}},
                    ),
                )
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.tool_disabled")
        self.assertEqual(result.diagnostics[0].path, "nodes.start.ref")
        self.assertEqual(
            result.diagnostics[0].message,
            "shell.git_commit is disabled for this flow.",
        )
        self.assertIn("history", result.diagnostics[0].hint)
        self.assertIn("allowed_commands", result.diagnostics[0].hint)
        self.assertIn("commit", result.diagnostics[0].hint)

    def test_validate_flow_definition_rejects_shell_git_allowed_command_bypass(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["shell.git_status"],
            available_toolsets=[
                ShellToolSet(
                    settings=ShellToolSettings(
                        git=ShellGitToolSettings(
                            capabilities=("read",),
                            allowed_commands=("diff",),
                        )
                    )
                )
            ],
        )
        registry = tool_flow_node_registry(manager)

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="shell.git_status",
                        config={"arguments": {}},
                    ),
                )
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.tool_disabled")
        self.assertEqual(
            result.diagnostics[0].message,
            "shell.git_status is disabled for this flow.",
        )
        self.assertIn("capabilities", result.diagnostics[0].hint)
        self.assertIn("read", result.diagnostics[0].hint)
        self.assertIn("allowed_commands", result.diagnostics[0].hint)
        self.assertIn("status", result.diagnostics[0].hint)

    def test_validate_flow_definition_rejects_unavailable_shell_git_runtime(
        self,
    ) -> None:
        registry = tool_flow_node_registry(_tool_manager())

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="shell.git_status",
                        config={"arguments": {}},
                    ),
                )
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.tool_unknown")
        self.assertEqual(
            result.diagnostics[0].message,
            "shell.git_status is not available in this flow runtime.",
        )
        self.assertIn("shell tool resolver", result.diagnostics[0].hint)
        self.assertIn("required capability", result.diagnostics[0].hint)

    def test_validate_flow_definition_rejects_disabled_unknown_shell_git_name(
        self,
    ) -> None:
        registry = tool_flow_node_registry(
            StaticToolResolver(
                [],
                disabled_names={"shell.git_custom_operation"},
            )
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="shell.git_custom_operation",
                        config={"arguments": {}},
                    ),
                )
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.tool_disabled")
        self.assertEqual(
            result.diagnostics[0].message,
            "shell.git_custom_operation is disabled for this flow.",
        )
        self.assertIn(
            "the required Git capability",
            result.diagnostics[0].hint,
        )
        self.assertIn("custom-operation", result.diagnostics[0].hint)

    def test_validate_flow_definition_rejects_strict_tool_refs(
        self,
    ) -> None:
        registry = tool_flow_node_registry(
            _tool_manager(
                enable_tools=[
                    "validator_flow_adder",
                    "validator_flow_adder_alt",
                    "validator_flow_status",
                ]
            )
        )
        cases = (
            (None, "flow.missing_ref"),
            ("tools/adder.py", "flow.invalid_ref"),
            ("mcp://server/tool", "flow.invalid_ref"),
            ("avl_dmFsaWRhdG9yX2Zsb3dfYWRkZXI", "flow.invalid_ref"),
            ("functions.validator_flow_adder", "flow.invalid_ref"),
            ("missing", "flow.tool_unknown"),
            ("disabled.validator_flow_disabled", "flow.tool_disabled"),
            ("sum", "flow.tool_ambiguous"),
        )

        for ref, code in cases:
            with self.subTest(ref=ref):
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type=FLOW_TOOL_NODE_TYPE,
                                ref=ref,
                                config={
                                    "arguments": {
                                        "a": "left",
                                        "b": "right",
                                    }
                                },
                            ),
                        )
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)
                self.assertEqual(result.diagnostics[0].path, "nodes.start.ref")
                if ref is not None:
                    self.assertNotIn(ref, str(result.public_diagnostics))

    def test_validate_flow_definition_rejects_pipeline_without_allow(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["shell.pipeline"],
            available_toolsets=[
                ShellToolSet(settings=ShellToolSettings(allow_pipelines=False))
            ],
            settings=ToolManagerSettings(),
        )
        registry = tool_flow_node_registry(manager)

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="shell.pipeline",
                        config={
                            "arguments": {
                                "steps": [
                                    {
                                        "id": "read",
                                        "command": "cat",
                                    }
                                ],
                            }
                        },
                    ),
                )
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.tool_disabled")
        self.assertEqual(result.diagnostics[0].path, "nodes.start.ref")
        self.assertIn("allow_pipelines", result.diagnostics[0].hint)

    def test_validate_flow_definition_rejects_unavailable_pipeline(
        self,
    ) -> None:
        registry = tool_flow_node_registry(StaticToolResolver([]))

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="shell.pipeline",
                        config={
                            "arguments": {
                                "steps": [
                                    {
                                        "id": "read",
                                        "command": "cat",
                                    }
                                ],
                            }
                        },
                    ),
                )
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.tool_unknown")
        self.assertEqual(result.diagnostics[0].path, "nodes.start.ref")
        self.assertEqual(
            result.diagnostics[0].message,
            "shell.pipeline is not available in this flow runtime.",
        )
        self.assertIn("shell tool resolver", result.diagnostics[0].hint)
        self.assertIn("allow_pipelines=true", result.diagnostics[0].hint)

    def test_validate_flow_definition_rejects_unavailable_unknown_git_ref(
        self,
    ) -> None:
        registry = tool_flow_node_registry(StaticToolResolver([]))

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(
                        name="start",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="shell.git_custom_command",
                        config={"arguments": {}},
                    ),
                )
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.tool_unknown")
        self.assertEqual(result.diagnostics[0].path, "nodes.start.ref")
        self.assertEqual(
            result.diagnostics[0].message,
            "shell.git_custom_command is not available in this flow runtime.",
        )
        self.assertIn("shell tool resolver", result.diagnostics[0].hint)

    def test_validate_flow_definition_rejects_strict_tool_bindings(
        self,
    ) -> None:
        registry = tool_flow_node_registry(_tool_manager())
        cases = (
            (
                "validator_flow_adder",
                {},
                "flow.missing_argument_binding",
                "nodes.start.config.arguments.a",
            ),
            (
                "validator_flow_adder",
                {"arguments": "bad"},
                "flow.invalid_arguments",
                "nodes.start.config.arguments",
            ),
            (
                "validator_flow_adder",
                {"arguments": {"a": "left", "b": "right", "c": "other"}},
                "flow.unknown_argument_binding",
                "nodes.start.config.arguments.c",
            ),
            (
                "validator_flow_adder",
                {"arguments": {"a": "left"}},
                "flow.missing_argument_binding",
                "nodes.start.config.arguments.b",
            ),
            (
                "validator_flow_adder",
                {"arguments": {"a": "left", "b": ""}},
                "flow.invalid_argument_selector",
                "nodes.start.config.arguments.b",
            ),
            (
                "validator_flow_adder",
                {"arguments": {"a": object(), "b": "right"}},
                "flow.invalid_argument_literal",
                "nodes.start.config.arguments.a",
            ),
            (
                "validator_flow_adder",
                {
                    "arguments": {"a": "left", "b": "right"},
                    "output_mode": "wrapped",
                },
                "flow.invalid_output_mode",
                "nodes.start.config.output_mode",
            ),
            (
                "validator_flow_status",
                {"arguments": {"value": "payload"}},
                "flow.unknown_argument_binding",
                "nodes.start.config.arguments.value",
            ),
        )

        for ref, config, code, path in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type=FLOW_TOOL_NODE_TYPE,
                                ref=ref,
                                config=config,
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(result.diagnostics[0].code, code)
                self.assertEqual(result.diagnostics[0].path, path)

    def test_validate_flow_definition_handles_tool_schema_edges(
        self,
    ) -> None:
        cases = (
            (
                ToolDescriptor(name="raw", parameter_schema=None),
                {"arguments": {}},
            ),
            (
                ToolDescriptor(
                    name="bad_required",
                    parameter_schema={
                        "type": "object",
                        "properties": {},
                        "required": "bad",
                    },
                ),
                {},
            ),
            (
                ToolDescriptor(
                    name="bad_properties",
                    parameter_schema={
                        "type": "object",
                        "properties": [],
                    },
                ),
                {"arguments": {}},
            ),
        )

        for descriptor, config in cases:
            with self.subTest(name=descriptor.name):
                result = validate_flow_definition(
                    self._strict_definition(
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type=FLOW_TOOL_NODE_TYPE,
                                ref=descriptor.name,
                                config=config,
                            ),
                        )
                    ),
                    tool_flow_node_registry(StaticToolResolver([descriptor])),
                )

                self.assertTrue(result.ok, result.public_diagnostics)

    def test_validate_flow_definition_accepts_declarative_mappings(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "document": lambda definition: Node(definition.name),
                "prepare": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "document": FlowNodeMetadata(
                    kind=FlowNodeKind.FILE_CONVERSION,
                    output_contracts=(
                        FlowNodeContract(
                            name="file",
                            type=FlowOutputType.FILE,
                        ),
                        FlowNodeContract(
                            name="files",
                            type=FlowOutputType.FILE_ARRAY,
                        ),
                    ),
                ),
                "prepare": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="payload",
                        type=FlowOutputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "units": {"type": "string"},
                                "items": {"type": "array"},
                            },
                        },
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contracts=(
                        FlowNodeContract(
                            name="arguments",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="aliases",
                            type=FlowInputType.ARRAY,
                        ),
                        FlowNodeContract(
                            name="document",
                            type=FlowInputType.FILE,
                        ),
                        FlowNodeContract(
                            name="attachments",
                            type=FlowInputType.FILE_ARRAY,
                        ),
                        FlowNodeContract(
                            name="renamed",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="merged",
                            type=FlowInputType.OBJECT,
                        ),
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="mapped",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="prepare"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "target.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="prepare", type="prepare"),
                    FlowNodeDefinition(name="document", type="document"),
                    FlowNodeDefinition(
                        name="target",
                        type="target",
                        join_policy=FlowJoinPolicy(
                            type=FlowJoinPolicyType.ALL_SUCCESS,
                        ),
                        mappings=(
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.OBJECT,
                                fields={
                                    "city": "prepare.payload.city",
                                    "units": "prepare.payload.units",
                                },
                            ),
                            FlowInputMapping(
                                target="aliases",
                                kind=FlowMappingKind.ARRAY,
                                items=(
                                    "prepare.payload.items[0]",
                                    "input.payload.items[0]",
                                ),
                            ),
                            FlowInputMapping(
                                target="document",
                                kind=FlowMappingKind.FILE,
                                source="document.file",
                            ),
                            FlowInputMapping(
                                target="attachments",
                                kind=FlowMappingKind.FILE_ARRAY,
                                source="document.files",
                            ),
                            FlowInputMapping(
                                target="renamed",
                                kind=FlowMappingKind.RENAME,
                                source="input.payload",
                            ),
                            FlowInputMapping(
                                target="merged",
                                kind=FlowMappingKind.MERGE,
                                sources=(
                                    "input.payload",
                                    "prepare.payload",
                                ),
                            ),
                        ),
                    ),
                ),
                edges=(
                    FlowEdgeDefinition(source="prepare", target="target"),
                    FlowEdgeDefinition(source="document", target="target"),
                ),
            ),
            registry,
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_accepts_dynamic_mapping_output_contract(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "source": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "source": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        type=FlowOutputType.OBJECT,
                        metadata={"dynamic": True},
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contract=FlowNodeContract(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                    output_contract=FlowNodeContract(name="result"),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="mapped",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="source"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "target.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="source", type="source"),
                    FlowNodeDefinition(
                        name="target",
                        type="target",
                        mappings=(
                            FlowInputMapping(
                                target="payload",
                                source="source.anything",
                            ),
                        ),
                    ),
                ),
                edges=(FlowEdgeDefinition(source="source", target="target"),),
            ),
            registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)

    def test_validate_flow_definition_rejects_declarative_mappings(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "document": lambda definition: Node(definition.name),
                "prepare": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "document": FlowNodeMetadata(
                    kind=FlowNodeKind.FILE_CONVERSION,
                    output_contract=FlowNodeContract(
                        name="file",
                        type=FlowOutputType.FILE,
                    ),
                ),
                "prepare": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="payload",
                        type=FlowOutputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {"known": {"type": "string"}},
                        },
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contracts=(
                        FlowNodeContract(
                            name="arguments",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="count",
                            type=FlowInputType.INTEGER,
                        ),
                        FlowNodeContract(
                            name="document",
                            type=FlowInputType.FILE,
                        ),
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="mapped",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="prepare"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "target.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="prepare", type="prepare"),
                    FlowNodeDefinition(name="document", type="document"),
                    FlowNodeDefinition(
                        name="target",
                        type="target",
                        mappings=(
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.OBJECT,
                            ),
                            FlowInputMapping(
                                target="arguments",
                                source="missing.payload",
                            ),
                            FlowInputMapping(
                                target="unknown",
                                source="input.payload",
                            ),
                            FlowInputMapping(
                                target="count",
                                source="input.payload",
                            ),
                            FlowInputMapping(
                                target="document",
                                kind=FlowMappingKind.FILE,
                                source="input.payload",
                            ),
                        ),
                    ),
                ),
                edges=(FlowEdgeDefinition(source="prepare", target="target"),),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.empty_mapping",
                "flow.duplicate_mapping_target",
                "flow.unknown_mapping_source",
                "flow.unknown_mapping_target",
                "flow.incompatible_mapping",
                "flow.incompatible_mapping",
            ],
        )

    def test_validate_flow_definition_rejects_mapping_edge_cases(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "document": lambda definition: Node(definition.name),
                "prepare": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "document": FlowNodeMetadata(
                    kind=FlowNodeKind.FILE_CONVERSION,
                    output_contracts=(
                        FlowNodeContract(
                            name="file",
                            type=FlowOutputType.FILE,
                        ),
                        FlowNodeContract(
                            name="files",
                            type=FlowOutputType.FILE_ARRAY,
                        ),
                    ),
                ),
                "prepare": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="payload",
                        type=FlowOutputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {"known": {"type": "string"}},
                        },
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contracts=(
                        FlowNodeContract(
                            name="required",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="arguments",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="items",
                            type=FlowInputType.ARRAY,
                        ),
                        FlowNodeContract(
                            name="text",
                            type=FlowInputType.STRING,
                        ),
                        FlowNodeContract(
                            name="files",
                            type=FlowInputType.FILE_ARRAY,
                        ),
                        FlowNodeContract(
                            name="array_bad",
                            type=FlowInputType.STRING,
                        ),
                        FlowNodeContract(
                            name="object_bad",
                            type=FlowInputType.STRING,
                        ),
                        FlowNodeContract(
                            name="merge_bad",
                            type=FlowInputType.STRING,
                        ),
                        FlowNodeContract(
                            name="coalesce_bad",
                            type=FlowInputType.STRING,
                        ),
                        FlowNodeContract(
                            name="coalesce_good",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="coalesce_type_bad",
                            type=FlowInputType.STRING,
                        ),
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="mapped",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="prepare"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "target.result"},
                ),
                nodes=(
                    FlowNodeDefinition(name="prepare", type="prepare"),
                    FlowNodeDefinition(name="document", type="document"),
                    FlowNodeDefinition(
                        name="target",
                        type="target",
                        mappings=(
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.MERGE,
                            ),
                            FlowInputMapping(
                                target="items",
                                kind=FlowMappingKind.ARRAY,
                            ),
                            FlowInputMapping(target="text"),
                            FlowInputMapping(
                                target="files",
                                kind=FlowMappingKind.FILE_ARRAY,
                                source="document.file",
                            ),
                            FlowInputMapping(
                                target="array_bad",
                                kind=FlowMappingKind.ARRAY,
                                items=("input.payload",),
                            ),
                            FlowInputMapping(
                                target="object_bad",
                                kind=FlowMappingKind.OBJECT,
                                fields={"payload": "input.payload"},
                            ),
                            FlowInputMapping(
                                target="merge_bad",
                                kind=FlowMappingKind.MERGE,
                                sources=("input.payload",),
                            ),
                            FlowInputMapping(
                                target="coalesce_bad",
                                kind=FlowMappingKind.COALESCE,
                            ),
                            FlowInputMapping(
                                target="coalesce_good",
                                kind=FlowMappingKind.COALESCE,
                                sources=("input.payload",),
                            ),
                            FlowInputMapping(
                                target="coalesce_type_bad",
                                kind=FlowMappingKind.COALESCE,
                                sources=("input.payload",),
                            ),
                            FlowInputMapping(
                                target="arguments",
                                kind=FlowMappingKind.OBJECT,
                                fields={
                                    "unknown_input": "input.missing",
                                    "disconnected": "document.file",
                                    "unknown_output": "prepare.missing",
                                    "unknown_path": "prepare.payload.missing",
                                },
                            ),
                        ),
                    ),
                ),
                edges=(FlowEdgeDefinition(source="prepare", target="target"),),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.empty_mapping",
                "flow.empty_mapping",
                "flow.missing_mapping_source",
                "flow.bad_reference",
                "flow.incompatible_mapping",
                "flow.incompatible_mapping",
                "flow.incompatible_mapping",
                "flow.empty_mapping",
                "flow.incompatible_mapping",
                "flow.duplicate_mapping_target",
                "flow.unknown_mapping_source",
                "flow.bad_reference",
                "flow.unknown_node_output",
                "flow.unknown_selector_path",
                "flow.missing_input_mapping",
            ],
        )

    def test_validate_flow_definition_rejects_invalid_mapping_selector(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"target": lambda definition: Node(definition.name)},
            {
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contract=FlowNodeContract(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="mapped",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="target"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "target.result"},
                ),
                nodes=(
                    FlowNodeDefinition(
                        name="target",
                        type="target",
                        mappings=(
                            FlowInputMapping(
                                target="payload",
                                source="input.payload/{{secret}}",
                            ),
                        ),
                    ),
                ),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.unsafe_selector")
        self.assertEqual(
            result.diagnostics[0].category,
            FlowDiagnosticCategory.PRIVACY,
        )
        self.assertNotIn("secret", str(result.public_diagnostics))

    def test_validate_flow_definition_rejects_final_mapping_matrix(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "source": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "source": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="value",
                        type=FlowOutputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {
                                "customer": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                    },
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    input_contracts=(
                        FlowNodeContract(
                            name="required",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="payload",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="count",
                            type=FlowInputType.INTEGER,
                        ),
                        FlowNodeContract(
                            name="document",
                            type=FlowInputType.FILE,
                        ),
                    ),
                ),
            },
        )
        valid_required = FlowInputMapping(
            target="required",
            source="input.payload",
        )
        valid_payload = FlowInputMapping(
            target="payload",
            source="input.payload",
        )
        valid_count = FlowInputMapping(target="count", source="input.count")
        valid_document = FlowInputMapping(
            target="document",
            kind=FlowMappingKind.FILE,
            source="input.document",
        )
        cases = (
            (
                "missing required input",
                (valid_required, valid_count, valid_document),
                ("flow.missing_input_mapping",),
                ("nodes.target.mapping.payload",),
            ),
            (
                "invalid object path",
                (
                    valid_required,
                    FlowInputMapping(
                        target="payload",
                        kind=FlowMappingKind.OBJECT,
                        fields={"bad": "source.value.tags.name"},
                    ),
                    valid_count,
                    valid_document,
                ),
                ("flow.unknown_selector_path",),
                ("nodes.target.mapping.payload.fields.bad",),
            ),
            (
                "invalid array path",
                (
                    valid_required,
                    FlowInputMapping(
                        target="payload",
                        kind=FlowMappingKind.ARRAY,
                        items=("source.value.customer[0]",),
                    ),
                    valid_count,
                    valid_document,
                ),
                (
                    "flow.unknown_selector_path",
                    "flow.incompatible_mapping",
                ),
                (
                    "nodes.target.mapping.payload.items[0]",
                    "nodes.target.mapping.payload",
                ),
            ),
            (
                "incompatible target type",
                (
                    valid_required,
                    valid_payload,
                    FlowInputMapping(
                        target="count",
                        source="input.payload",
                    ),
                    valid_document,
                ),
                ("flow.incompatible_mapping",),
                ("nodes.target.mapping.count.source",),
            ),
            (
                "file mapping violation",
                (
                    valid_required,
                    valid_payload,
                    valid_count,
                    FlowInputMapping(
                        target="document",
                        kind=FlowMappingKind.FILE,
                        source="input.payload",
                    ),
                ),
                ("flow.incompatible_mapping",),
                ("nodes.target.mapping.document.source",),
            ),
        )

        for name, mappings, codes, paths in cases:
            with self.subTest(name=name):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="mapped",
                        version="2026-06-07",
                        inputs=(
                            FlowInputDefinition(
                                name="payload",
                                type=FlowInputType.OBJECT,
                            ),
                            FlowInputDefinition(
                                name="count",
                                type=FlowInputType.INTEGER,
                            ),
                            FlowInputDefinition(
                                name="document",
                                type=FlowInputType.FILE,
                            ),
                        ),
                        outputs=(
                            FlowOutputDefinition(
                                name="answer",
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="source"),
                        output_behavior=FlowOutputBehavior(
                            outputs={"answer": "target.result"},
                        ),
                        nodes=(
                            FlowNodeDefinition(
                                name="source",
                                type="source",
                            ),
                            FlowNodeDefinition(
                                name="target",
                                type="target",
                                mappings=mappings,
                            ),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="source",
                                target="target",
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(
                    tuple(
                        diagnostic.code for diagnostic in result.diagnostics
                    ),
                    codes,
                )
                self.assertEqual(
                    tuple(
                        diagnostic.path for diagnostic in result.diagnostics
                    ),
                    paths,
                )

    def test_validate_flow_definition_accepts_declarative_conditions(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="conditioned",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "finish.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="finish",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.ALL,
                            conditions=(
                                FlowCondition(
                                    operator=FlowConditionOperator.EQ,
                                    selector="start.value.status",
                                    value_selector="input.payload.expected",
                                ),
                                FlowCondition(
                                    operator=FlowConditionOperator.IS_TYPE,
                                    selector="start.value.score",
                                    value_type=FlowConditionValueType.NUMBER,
                                ),
                                FlowCondition(
                                    operator=FlowConditionOperator.NOT,
                                    condition=FlowCondition(
                                        operator=FlowConditionOperator.EXISTS,
                                        selector="start.value.blocked",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_accepts_compatible_value_selector(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "typed": lambda definition: Node(definition.name),
                "echo": lambda definition: Node(definition.name),
            },
            {
                "typed": FlowNodeMetadata(
                    kind=FlowNodeKind.PASS_THROUGH,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                            },
                        },
                    ),
                ),
                "echo": FlowNodeMetadata(kind=FlowNodeKind.PASS_THROUGH),
            },
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="conditioned",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {
                                "expected": {"type": "string"},
                            },
                        },
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "finish.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="typed"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="finish",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.EQ,
                            selector="start.result.status",
                            value_selector="input.payload.expected",
                        ),
                    ),
                ),
            ),
            registry,
        )

        self.assertTrue(result.ok, result.public_diagnostics)

    def test_validate_flow_definition_rejects_declarative_conditions(
        self,
    ) -> None:
        cases = (
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    value="ready",
                ),
                "flow.missing_condition_selector",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                ),
                "flow.missing_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.GT,
                    selector="start.value.score",
                    value="3",
                ),
                "flow.invalid_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.CONTAINS,
                    selector="start.value.status",
                    value=3,
                ),
                "flow.invalid_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IS_TYPE,
                    selector="start.value.status",
                ),
                "flow.missing_condition_value_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="start.value.status",
                ),
                "flow.missing_condition_values",
            ),
            (
                FlowCondition(operator=FlowConditionOperator.ALL),
                "flow.missing_condition_children",
            ),
            (
                FlowCondition(operator=FlowConditionOperator.NOT),
                "flow.missing_condition_child",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="start.value.status",
                    value="ready",
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="start.value.status",
                    value_selector="start.value.other",
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value="ready",
                    value_type=FlowConditionValueType.STRING,
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value="ready",
                    values=("ready",),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value="ready",
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.ALL,
                    selector="start.value.status",
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.ALL,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                    values=("ready",),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.ANY,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                    values=("ready",),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.ALL,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EXISTS,
                        selector="start.value.other",
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.NOT,
                    conditions=(
                        FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.status",
                        ),
                    ),
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EXISTS,
                        selector="start.value.other",
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.NOT,
                    values=("ready",),
                    condition=FlowCondition(
                        operator=FlowConditionOperator.EXISTS,
                        selector="start.value.other",
                    ),
                ),
                "flow.unsupported_condition_field",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="input.missing",
                ),
                "flow.unknown_condition_source",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="missing.value",
                ),
                "flow.unknown_condition_source",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="other.value.status",
                ),
                "flow.bad_reference",
            ),
        )

        for condition, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="conditioned",
                        version="2026-06-07",
                        inputs=(
                            FlowInputDefinition(
                                name="payload",
                                type=FlowInputType.OBJECT,
                            ),
                        ),
                        outputs=(
                            FlowOutputDefinition(
                                name="answer",
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="start"),
                        output_behavior=FlowOutputBehavior(
                            outputs={"answer": "finish.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="start", type="echo"),
                            FlowNodeDefinition(name="other", type="echo"),
                            FlowNodeDefinition(name="finish", type="echo"),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="start",
                                target="finish",
                                condition=condition,
                            ),
                        ),
                    )
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_rejects_private_conditions(
        self,
    ) -> None:
        cases = (
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="env.SECRET",
                ),
                "flow.reserved_selector",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value="{{secret}}",
                ),
                "flow.unsafe_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.value.status",
                    value={"token": "{{secret}}"},
                ),
                "flow.unsafe_condition_value",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="start.value.status",
                    values=(("{{secret}}",),),
                ),
                "flow.unsafe_condition_value",
            ),
        )

        for condition, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="conditioned",
                        version="2026-06-07",
                        inputs=(
                            FlowInputDefinition(
                                name="payload",
                                type=FlowInputType.OBJECT,
                            ),
                        ),
                        outputs=(
                            FlowOutputDefinition(
                                name="answer",
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="start"),
                        output_behavior=FlowOutputBehavior(
                            outputs={"answer": "finish.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="start", type="echo"),
                            FlowNodeDefinition(name="finish", type="echo"),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="start",
                                target="finish",
                                condition=condition,
                            ),
                        ),
                    )
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )
                self.assertIn(
                    FlowDiagnosticCategory.PRIVACY,
                    {
                        diagnostic.category
                        for diagnostic in result.diagnostics
                        if diagnostic.code == code
                    },
                )
                self.assertNotIn("secret", str(result.public_diagnostics))

    def test_validate_flow_definition_rejects_nondeterministic_conditions(
        self,
    ) -> None:
        cases = (
            FlowCondition(
                operator=FlowConditionOperator.EQ,
                selector="start.value.status",
                value="random()",
            ),
            FlowCondition(
                operator=FlowConditionOperator.IN,
                selector="start.value.status",
                values=({"clock": "datetime.now()"},),
            ),
        )

        for condition in cases:
            with self.subTest(condition=condition):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="conditioned",
                        version="2026-06-07",
                        inputs=(
                            FlowInputDefinition(
                                name="payload",
                                type=FlowInputType.OBJECT,
                            ),
                        ),
                        outputs=(
                            FlowOutputDefinition(
                                name="answer",
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="start"),
                        output_behavior=FlowOutputBehavior(
                            outputs={"answer": "finish.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="start", type="echo"),
                            FlowNodeDefinition(name="finish", type="echo"),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="start",
                                target="finish",
                                condition=condition,
                            ),
                        ),
                    )
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    "flow.nondeterministic_condition_value",
                    [diagnostic.code for diagnostic in result.diagnostics],
                )
                self.assertIn(
                    FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
                    {
                        diagnostic.category
                        for diagnostic in result.diagnostics
                        if diagnostic.code
                        == "flow.nondeterministic_condition_value"
                    },
                )
                self.assertNotIn("random()", str(result.public_diagnostics))
                self.assertNotIn(
                    "datetime.now()", str(result.public_diagnostics)
                )

    def test_validate_flow_definition_rejects_condition_contract_mismatches(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"schema": lambda definition: Node(definition.name)},
            {
                "schema": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="result",
                        schema={
                            "type": "object",
                            "properties": {
                                "known": {"type": "string"},
                                "score": {"type": "number"},
                            },
                        },
                    ),
                ),
            },
        )
        cases = (
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="start.missing",
                ),
                "flow.unknown_node_output",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="start.result.missing",
                ),
                "flow.unknown_selector_path",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.GT,
                    selector="start.result.known",
                    value=3,
                ),
                "flow.incompatible_condition_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.CONTAINS,
                    selector="start.result.score",
                    value="3",
                ),
                "flow.incompatible_condition_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.result.known",
                    value=3,
                ),
                "flow.incompatible_condition_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.EQ,
                    selector="start.result.score",
                    value_selector="start.result.known",
                ),
                "flow.incompatible_condition_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="start.result.known",
                    values=("ready", 3),
                ),
                "flow.incompatible_condition_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="start.result.known",
                    value=("ready", 3),
                ),
                "flow.incompatible_condition_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.GT,
                    selector="input.payload.status",
                    value=3,
                ),
                "flow.incompatible_condition_type",
            ),
            (
                FlowCondition(
                    operator=FlowConditionOperator.CONTAINS,
                    selector="input.flag",
                    value="true",
                ),
                "flow.incompatible_condition_type",
            ),
        )

        for condition, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="conditioned",
                        version="2026-06-07",
                        inputs=(
                            FlowInputDefinition(
                                name="other",
                                type=FlowInputType.STRING,
                            ),
                            FlowInputDefinition(
                                name="payload",
                                type=FlowInputType.OBJECT,
                                schema={
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": ("string",)},
                                        "score": {"type": "number"},
                                    },
                                },
                            ),
                            FlowInputDefinition(
                                name="flag",
                                type=FlowInputType.BOOLEAN,
                            ),
                            FlowInputDefinition(
                                name="opaque",
                                type=FlowInputType.OBJECT,
                            ),
                        ),
                        outputs=(
                            FlowOutputDefinition(
                                name="answer",
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="start"),
                        output_behavior=FlowOutputBehavior(
                            outputs={"answer": "finish.result"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="start", type="schema"),
                            FlowNodeDefinition(name="finish", type="schema"),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="start",
                                target="finish",
                                condition=condition,
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_condition_private_helpers_cover_type_edges(self) -> None:
        definition = FlowDefinition(
            name="conditioned",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
                FlowInputDefinition(
                    name="opaque",
                    type=FlowInputType.OBJECT,
                ),
            ),
            nodes=(FlowNodeDefinition(name="start", type="schema"),),
        )
        schema_definition = FlowDefinition(
            name="conditioned",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                    schema={
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                    },
                ),
            ),
            nodes=(FlowNodeDefinition(name="start", type="schema"),),
        )
        field_path = parse_flow_selector("start.result.status").path
        index_path = parse_flow_selector("start.result[0]").path

        self.assertIsNone(
            flow_validator._flow_input_selector_type(  # type: ignore[attr-defined]
                definition,
                parse_flow_selector("input.missing"),
            )
        )
        self.assertIsNone(
            flow_validator._flow_input_selector_type(  # type: ignore[attr-defined]
                definition,
                parse_flow_selector("input.opaque.status"),
            )
        )
        self.assertIsNone(
            flow_validator._node_output_selector_type(  # type: ignore[attr-defined]
                FlowNodeRegistry(
                    {"plain": lambda node: Node(node.name)},
                ),
                "plain",
                parse_flow_selector("plain.result"),
            )
        )
        self.assertIsNone(
            flow_validator._node_output_selector_type(  # type: ignore[attr-defined]
                FlowNodeRegistry(
                    {"plain": lambda node: Node(node.name)},
                    {
                        "plain": FlowNodeMetadata(
                            output_contract=FlowNodeContract(name="other"),
                        ),
                    },
                ),
                "plain",
                parse_flow_selector("plain.result"),
            )
        )
        self.assertIsNone(
            flow_validator._schema_selector_type(  # type: ignore[attr-defined]
                {"type": "object"},
                field_path,
            )
        )
        self.assertIsNone(
            flow_validator._schema_selector_type(  # type: ignore[attr-defined]
                {
                    "type": "object",
                    "properties": {"status": "bad"},
                },
                field_path,
            )
        )
        self.assertIsNone(
            flow_validator._schema_selector_type(  # type: ignore[attr-defined]
                {"type": "array"},
                index_path,
            )
        )
        self.assertEqual(
            flow_validator._schema_selector_type(  # type: ignore[attr-defined]
                {"type": "array", "items": {"type": "string"}},
                index_path,
            ),
            "string",
        )
        self.assertIsNone(
            flow_validator._schema_type_name(  # type: ignore[attr-defined]
                ("string", "null")
            )
        )
        self.assertIsNone(
            flow_validator._schema_type_name(  # type: ignore[attr-defined]
                (1,)
            )
        )
        self.assertIsNone(
            flow_validator._contract_semantic_type(  # type: ignore[attr-defined]
                None
            )
        )
        self.assertEqual(
            flow_validator._contract_semantic_type(  # type: ignore[attr-defined]
                "custom"
            ),
            "custom",
        )
        self.assertEqual(
            flow_validator._contract_semantic_type(  # type: ignore[attr-defined]
                FlowInputType.STRING
            ),
            "string",
        )
        self.assertEqual(
            flow_validator._condition_membership_literals(  # type: ignore[attr-defined]
                FlowCondition(
                    operator=FlowConditionOperator.EXISTS,
                    selector="start.value",
                )
            ),
            (),
        )
        self.assertEqual(
            flow_validator._validate_condition_type_compatibility(  # type: ignore[attr-defined]
                schema_definition,
                FlowCondition(
                    operator=FlowConditionOperator.IN,
                    selector="input.payload.status",
                    values=(object(),),
                ),
                path="edges[0].condition",
                edge_source="start",
                input_names={"payload"},
                node_names={"start"},
                registry=FlowNodeRegistry(
                    {"schema": lambda node: Node(node.name)}
                ),
            ),
            (),
        )
        literal_cases = (
            (True, "boolean"),
            (3.5, "number"),
            ({"status": "ready"}, "object"),
            (["ready"], "array"),
            (object(), None),
        )

        for value, expected_type in literal_cases:
            with self.subTest(value_type=expected_type):
                self.assertEqual(
                    flow_validator._condition_literal_type(  # type: ignore[attr-defined]
                        value
                    ),
                    expected_type,
                )

    def test_validate_flow_definition_accepts_edge_routing_policy(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="routed",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "approved.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="approved", type="echo"),
                    FlowNodeDefinition(name="rejected", type="echo"),
                    FlowNodeDefinition(name="fallback", type="echo"),
                    FlowNodeDefinition(name="timeout", type="echo"),
                    FlowNodeDefinition(name="cleanup", type="echo"),
                    FlowNodeDefinition(name="cancel", type="echo"),
                    FlowNodeDefinition(name="pause", type="echo"),
                    FlowNodeDefinition(name="resume", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="approved",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.approved",
                        ),
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="rejected",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value.rejected",
                        ),
                        priority=1,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="fallback",
                        default=True,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="timeout",
                        kind=FlowEdgeKind.TIMEOUT,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="cleanup",
                        kind=FlowEdgeKind.FINALLY,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="cancel",
                        kind=FlowEdgeKind.CANCELLATION,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="pause",
                        kind=FlowEdgeKind.PAUSE,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="resume",
                        kind=FlowEdgeKind.RESUME,
                    ),
                ),
            )
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_accepts_all_matching_routes(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="routed",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "left.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="left", type="echo"),
                    FlowNodeDefinition(name="right", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="right",
                        routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                    ),
                ),
            )
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_bad_route_policy(
        self,
    ) -> None:
        cases = (
            (
                (
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        default=True,
                    ),
                    FlowEdgeDefinition(
                        source="start",
                        target="right",
                        default=True,
                    ),
                ),
                "flow.duplicate_default_route",
            ),
            (
                (
                    FlowEdgeDefinition(source="start", target="left"),
                    FlowEdgeDefinition(source="start", target="right"),
                ),
                "flow.ambiguous_route",
            ),
            (
                (
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        routing_policy=FlowRouteMatchPolicy.ALL_MATCHING,
                    ),
                    FlowEdgeDefinition(source="start", target="right"),
                ),
                "flow.mixed_routing_policy",
            ),
            (
                (
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        priority=-1,
                    ),
                ),
                "flow.invalid_route_priority",
            ),
            (
                (
                    FlowEdgeDefinition(
                        source="start",
                        target="left",
                        condition=FlowCondition(
                            operator=FlowConditionOperator.EXISTS,
                            selector="start.value",
                        ),
                        default=True,
                    ),
                ),
                "flow.default_route_condition",
            ),
        )

        for edges, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="routed",
                        version="2026-06-07",
                        inputs=(
                            FlowInputDefinition(
                                name="payload",
                                type=FlowInputType.OBJECT,
                            ),
                        ),
                        outputs=(
                            FlowOutputDefinition(
                                name="answer",
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="start"),
                        output_behavior=FlowOutputBehavior(
                            outputs={"answer": "left.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="start", type="echo"),
                            FlowNodeDefinition(name="left", type="echo"),
                            FlowNodeDefinition(name="right", type="echo"),
                        ),
                        edges=edges,
                    )
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_rejects_legacy_edge_policy(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="legacy",
                entrypoint="start",
                output_node="finish",
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(
                        source="start",
                        target="finish",
                        kind=FlowEdgeKind.ERROR,
                    ),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.unsupported_edge_policy"],
        )

    def test_validate_flow_definition_accepts_join_policies(self) -> None:
        cases = (
            FlowJoinPolicy(type=FlowJoinPolicyType.ALL_SUCCESS),
            FlowJoinPolicy(type=FlowJoinPolicyType.ALL_DONE),
            FlowJoinPolicy(type=FlowJoinPolicyType.ANY_SUCCESS),
            FlowJoinPolicy(type=FlowJoinPolicyType.QUORUM, quorum=2),
            FlowJoinPolicy(type=FlowJoinPolicyType.FIRST_SUCCESS),
            FlowJoinPolicy(type=FlowJoinPolicyType.FAIL_FAST),
            FlowJoinPolicy(type=FlowJoinPolicyType.COLLECT),
        )

        for join_policy in cases:
            with self.subTest(join_policy=join_policy.type.value):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="joined",
                        version="2026-06-07",
                        inputs=(
                            FlowInputDefinition(
                                name="payload",
                                type=FlowInputType.OBJECT,
                            ),
                        ),
                        outputs=(
                            FlowOutputDefinition(
                                name="answer",
                                type=FlowOutputType.OBJECT,
                            ),
                        ),
                        entry_behavior=FlowEntryBehavior(node="left"),
                        output_behavior=FlowOutputBehavior(
                            outputs={"answer": "finish.value"},
                        ),
                        nodes=(
                            FlowNodeDefinition(name="left", type="echo"),
                            FlowNodeDefinition(name="right", type="echo"),
                            FlowNodeDefinition(
                                name="finish",
                                type="echo",
                                join_policy=join_policy,
                            ),
                        ),
                        edges=(
                            FlowEdgeDefinition(
                                source="left",
                                target="finish",
                            ),
                            FlowEdgeDefinition(
                                source="right",
                                target="finish",
                            ),
                        ),
                    )
                )

                self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_missing_join_policy(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="joined",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="left"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "finish.value"},
                ),
                nodes=(
                    FlowNodeDefinition(name="left", type="echo"),
                    FlowNodeDefinition(name="right", type="echo"),
                    FlowNodeDefinition(name="finish", type="echo"),
                ),
                edges=(
                    FlowEdgeDefinition(source="left", target="finish"),
                    FlowEdgeDefinition(source="right", target="finish"),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            result.diagnostics[0].code, "flow.missing_join_policy"
        )
        self.assertEqual(
            result.diagnostics[0].path,
            "nodes.finish.join_policy",
        )

    def test_validate_flow_definition_rejects_join_policy_details(
        self,
    ) -> None:
        cases = (
            (
                FlowJoinPolicy(type=FlowJoinPolicyType.QUORUM),
                "flow.missing_join_quorum",
            ),
            (
                FlowJoinPolicy(type=FlowJoinPolicyType.QUORUM, quorum=0),
                "flow.invalid_join_quorum",
            ),
            (
                FlowJoinPolicy(type=FlowJoinPolicyType.QUORUM, quorum=3),
                "flow.invalid_join_quorum",
            ),
            (
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_SUCCESS,
                    quorum=1,
                ),
                "flow.unsupported_join_field",
            ),
            (
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_DONE,
                    optional_inputs=("audit", "audit"),
                ),
                "flow.duplicate_join_optional_input",
            ),
            (
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_DONE,
                    optional_inputs=("missing",),
                ),
                "flow.unknown_join_optional_input",
            ),
        )

        for join_policy, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    self._join_policy_contract_definition(join_policy),
                    self._join_policy_registry(),
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_uses_join_optional_inputs(
        self,
    ) -> None:
        result = validate_flow_definition(
            self._join_policy_contract_definition(
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_DONE,
                    optional_inputs=("audit",),
                ),
                include_payload_mapping=True,
            ),
            self._join_policy_registry(),
        )

        self.assertTrue(result.ok)

        missing_required = validate_flow_definition(
            self._join_policy_contract_definition(
                FlowJoinPolicy(
                    type=FlowJoinPolicyType.ALL_DONE,
                    optional_inputs=("audit",),
                ),
            ),
            self._join_policy_registry(),
        )

        self.assertFalse(missing_required.ok)
        self.assertIn(
            "flow.missing_input_mapping",
            [diagnostic.code for diagnostic in missing_required.diagnostics],
        )

    def test_validate_flow_definition_accepts_node_policies(self) -> None:
        result = validate_flow_definition(
            self._policy_definition(
                FlowNodeDefinition(
                    name="worker",
                    type="worker",
                    retry_policy=FlowRetryPolicy(
                        max_attempts=3,
                        backoff=FlowRetryBackoffStrategy.EXPONENTIAL,
                        initial_delay_seconds=1,
                        max_delay_seconds=8,
                        retryable_categories=("transient",),
                        non_retryable_categories=("validation",),
                        exhausted_route="failed",
                    ),
                    timeout_policy=FlowTimeoutPolicy(
                        per_attempt_seconds=30,
                    ),
                    loop_policy=FlowLoopPolicy(
                        max_iterations=4,
                        max_elapsed_seconds=60,
                        continue_condition=self._loop_condition("more"),
                        exit_condition=self._loop_condition("done"),
                        output_selector="worker.result.safe",
                        limit_route="limited",
                    ),
                ),
            ),
            self._policy_registry(),
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_retry_policy_details(
        self,
    ) -> None:
        cases = (
            (
                FlowRetryPolicy(),
                "flow.missing_retry_attempts",
                None,
            ),
            (
                FlowRetryPolicy(max_attempts=0),
                "flow.invalid_retry_attempts",
                None,
            ),
            (
                FlowRetryPolicy(
                    max_attempts=2,
                    backoff=FlowRetryBackoffStrategy.EXPONENTIAL,
                ),
                "flow.missing_retry_backoff_delay",
                None,
            ),
            (
                FlowRetryPolicy(
                    max_attempts=2,
                    backoff=FlowRetryBackoffStrategy.CONSTANT,
                    initial_delay_seconds=0,
                ),
                "flow.invalid_retry_backoff_delay",
                None,
            ),
            (
                FlowRetryPolicy(
                    max_attempts=2,
                    backoff=FlowRetryBackoffStrategy.LINEAR,
                    initial_delay_seconds=5,
                    max_delay_seconds=1,
                ),
                "flow.invalid_retry_max_delay",
                None,
            ),
            (
                FlowRetryPolicy(max_attempts=2, initial_delay_seconds=1),
                "flow.unsupported_retry_backoff_field",
                None,
            ),
            (
                FlowRetryPolicy(max_attempts=2, max_delay_seconds=1),
                "flow.unsupported_retry_backoff_field",
                None,
            ),
            (
                FlowRetryPolicy(
                    max_attempts=2,
                    backoff=FlowRetryBackoffStrategy.LINEAR,
                    initial_delay_seconds=1,
                    max_delay_seconds=0,
                ),
                "flow.invalid_retry_max_delay",
                None,
            ),
            (
                FlowRetryPolicy(
                    max_attempts=2,
                    retryable_categories=("transient", "transient"),
                ),
                "flow.duplicate_retry_category",
                None,
            ),
            (
                FlowRetryPolicy(
                    max_attempts=2,
                    non_retryable_categories=("validation", "validation"),
                ),
                "flow.duplicate_retry_category",
                None,
            ),
            (
                FlowRetryPolicy(
                    max_attempts=2,
                    retryable_categories=("transient",),
                    non_retryable_categories=("transient",),
                ),
                "flow.conflicting_retry_category",
                None,
            ),
            (
                FlowRetryPolicy(max_attempts=2, exhausted_route="missing"),
                "flow.unknown_retry_exhaustion_route",
                None,
            ),
            (
                FlowRetryPolicy(max_attempts=2, exhausted_route="failed"),
                "flow.missing_retry_exhaustion_route",
                (FlowEdgeDefinition(source="worker", target="finish"),),
            ),
        )

        for retry_policy, code, edges in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    self._policy_definition(
                        FlowNodeDefinition(
                            name="worker",
                            type="worker",
                            retry_policy=retry_policy,
                        ),
                        edges=edges,
                    ),
                    self._policy_registry(),
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_rejects_timeout_policy_details(
        self,
    ) -> None:
        cases = (
            (FlowTimeoutPolicy(), "flow.missing_timeout"),
            (
                FlowTimeoutPolicy(per_attempt_seconds=0),
                "flow.invalid_timeout",
            ),
        )

        for timeout_policy, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    self._policy_definition(
                        FlowNodeDefinition(
                            name="worker",
                            type="worker",
                            timeout_policy=timeout_policy,
                        ),
                    ),
                    self._policy_registry(),
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_accepts_human_review_contract(
        self,
    ) -> None:
        result = validate_flow_definition(
            self._human_review_definition(
                FlowNodeDefinition(
                    name="review",
                    type="human_review",
                    mappings=(
                        FlowInputMapping(
                            target="payload",
                            source="inputs.payload",
                        ),
                    ),
                    config=self._human_review_config(
                        audit_metadata={
                            "queue": "risk",
                            "labels": {"tier": "medium"},
                        }
                    ),
                ),
            ),
            self._human_review_registry(),
        )

        self.assertTrue(result.ok)

    def test_validate_flow_definition_rejects_human_review_gaps(
        self,
    ) -> None:
        cases = (
            (
                {},
                {
                    "flow.missing_human_review_config",
                    "flow.missing_human_review_decisions",
                    "flow.missing_human_review_decision_schema",
                    "flow.missing_human_review_payload_schema",
                    "flow.missing_human_review_timeout",
                },
            ),
            (
                self._human_review_config(allowed_decisions="approved"),
                {"flow.invalid_human_review_decisions"},
            ),
            (
                self._human_review_config(
                    allowed_decisions=(),
                    decision_schema={
                        "type": "object",
                        "properties": {"decision": {"enum": ("approved",)}},
                    },
                ),
                {"flow.missing_human_review_decisions"},
            ),
            (
                self._human_review_config(
                    allowed_decisions=("approved", "approved"),
                ),
                {"flow.duplicate_human_review_decision"},
            ),
            (
                self._human_review_config(
                    allowed_decisions=("",),
                    decision_schema={
                        "type": "object",
                        "properties": {"decision": {"enum": ("",)}},
                    },
                ),
                {"flow.invalid_human_review_decision"},
            ),
            (
                self._human_review_config(
                    allowed_decisions=("Approved",),
                    decision_schema={
                        "type": "object",
                        "properties": {"decision": {"enum": ("Approved",)}},
                    },
                ),
                {"flow.invalid_human_review_decision"},
            ),
            (
                self._human_review_config(payload_schema="schema"),
                {"flow.invalid_human_review_payload_schema"},
            ),
            (
                self._human_review_config(payload_schema={"type": "string"}),
                {"flow.invalid_human_review_payload_schema"},
            ),
            (
                self._human_review_config(decision_schema="schema"),
                {"flow.invalid_human_review_decision_schema"},
            ),
            (
                self._human_review_config(decision_schema={"type": "string"}),
                {"flow.invalid_human_review_decision_schema"},
            ),
            (
                self._human_review_config(
                    decision_schema={
                        "type": "object",
                        "properties": "schema",
                    },
                ),
                {"flow.invalid_human_review_decision_schema"},
            ),
            (
                self._human_review_config(
                    decision_schema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                {"flow.invalid_human_review_decision_schema"},
            ),
            (
                self._human_review_config(
                    decision_schema={
                        "type": "object",
                        "properties": {"decision": {"enum": ()}},
                    },
                ),
                {"flow.invalid_human_review_decision_schema"},
            ),
            (
                self._human_review_config(
                    decision_schema={
                        "type": "object",
                        "properties": {"decision": {"enum": ("approved", 1)}},
                    },
                ),
                {"flow.invalid_human_review_decision_schema"},
            ),
            (
                self._human_review_config(
                    decision_schema={
                        "type": "object",
                        "properties": {"decision": {"enum": ("approved",)}},
                    },
                ),
                {"flow.human_review_decision_schema_mismatch"},
            ),
            (
                self._human_review_config(timeout_seconds=0),
                {"flow.invalid_human_review_timeout"},
            ),
            (
                self._human_review_config(
                    audit_metadata={"raw_prompt": "private-token"}
                ),
                {"flow.unsafe_human_review_audit_metadata"},
            ),
            (
                self._human_review_config(
                    audit_metadata={"": "private-token"}
                ),
                {"flow.unsafe_human_review_audit_metadata"},
            ),
            (
                self._human_review_config(
                    audit_metadata={"queue": {"secret_label": "private-token"}}
                ),
                {"flow.unsafe_human_review_audit_metadata"},
            ),
            (
                self._human_review_config(audit_metadata="audit"),
                {"flow.invalid_human_review_audit_metadata"},
            ),
        )

        for config, expected_codes in cases:
            with self.subTest(expected_codes=expected_codes):
                result = validate_flow_definition(
                    self._human_review_definition(
                        FlowNodeDefinition(
                            name="review",
                            type="human_review",
                            mappings=(
                                FlowInputMapping(
                                    target="payload",
                                    source="inputs.payload",
                                ),
                            ),
                            config=config,
                        ),
                    ),
                    self._human_review_registry(),
                )

                codes = {diagnostic.code for diagnostic in result.diagnostics}
                self.assertFalse(result.ok)
                self.assertTrue(expected_codes.issubset(codes))
                self.assertNotIn(
                    "private-token", str(result.public_diagnostics)
                )

    def test_validate_flow_definition_rejects_human_review_routes(
        self,
    ) -> None:
        review = FlowNodeDefinition(
            name="review",
            type="human_review",
            mappings=(
                FlowInputMapping(
                    target="payload",
                    source="inputs.payload",
                ),
            ),
            config=self._human_review_config(),
        )
        cases = (
            (
                (
                    FlowEdgeDefinition(
                        source="review",
                        target="finish",
                        label="approved",
                        kind=FlowEdgeKind.RESUME,
                    ),
                    FlowEdgeDefinition(
                        source="review",
                        target="rejected",
                        label="rejected",
                        kind=FlowEdgeKind.RESUME,
                        priority=1,
                    ),
                    FlowEdgeDefinition(
                        source="review",
                        target="correction",
                        label="needs_correction",
                        kind=FlowEdgeKind.RESUME,
                        priority=2,
                    ),
                ),
                "flow.missing_human_review_timeout_route",
            ),
            (
                (
                    FlowEdgeDefinition(
                        source="review",
                        target="finish",
                        label="approved",
                        kind=FlowEdgeKind.RESUME,
                    ),
                    FlowEdgeDefinition(
                        source="review",
                        target="expired",
                        label="expired",
                        kind=FlowEdgeKind.TIMEOUT,
                    ),
                ),
                "flow.missing_human_review_resume_route",
            ),
        )

        for edges, code in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    self._human_review_definition(review, edges=edges),
                    self._human_review_registry(),
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_rejects_human_review_no_durable_support(
        self,
    ) -> None:
        result = validate_flow_definition(
            self._human_review_definition(
                FlowNodeDefinition(
                    name="review",
                    type="human_review",
                    mappings=(
                        FlowInputMapping(
                            target="payload",
                            source="inputs.payload",
                        ),
                    ),
                    config=self._human_review_config(),
                ),
            ),
            self._human_review_registry(durable=False),
        )

        self.assertFalse(result.ok)
        self.assertIn(
            "flow.unsupported_human_review_direct_mode",
            [diagnostic.code for diagnostic in result.diagnostics],
        )

    def test_validate_flow_definition_rejects_loop_policy_details(
        self,
    ) -> None:
        cases = (
            (
                FlowLoopPolicy(
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.safe",
                    limit_route="limited",
                ),
                "flow.unbounded_loop",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=0,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.safe",
                    limit_route="limited",
                ),
                "flow.invalid_loop_iterations",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_elapsed_seconds=0,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.safe",
                    limit_route="limited",
                ),
                "flow.invalid_loop_elapsed_time",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.safe",
                    limit_route="limited",
                ),
                "flow.missing_loop_continue_condition",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    output_selector="worker.result.safe",
                    limit_route="limited",
                ),
                "flow.missing_loop_exit_condition",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    limit_route="limited",
                ),
                "flow.missing_loop_output",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="env.SECRET",
                    limit_route="limited",
                ),
                "flow.reserved_selector",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="other.result.safe",
                    limit_route="limited",
                ),
                "flow.bad_reference",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.missing",
                    limit_route="limited",
                ),
                "flow.unknown_node_output",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.missing",
                    limit_route="limited",
                ),
                "flow.unknown_selector_path",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.safe",
                ),
                "flow.missing_loop_limit_route",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.safe",
                    limit_route="missing",
                ),
                "flow.unknown_loop_limit_route",
                None,
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=self._loop_condition("more"),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.safe",
                    limit_route="limited",
                ),
                "flow.missing_loop_limit_route",
                (FlowEdgeDefinition(source="worker", target="finish"),),
            ),
            (
                FlowLoopPolicy(
                    max_iterations=2,
                    continue_condition=FlowCondition(
                        operator=FlowConditionOperator.EXISTS,
                        selector="other.result.more",
                    ),
                    exit_condition=self._loop_condition("done"),
                    output_selector="worker.result.safe",
                    limit_route="limited",
                ),
                "flow.bad_reference",
                None,
            ),
        )

        for loop_policy, code, edges in cases:
            with self.subTest(code=code):
                result = validate_flow_definition(
                    self._policy_definition(
                        FlowNodeDefinition(
                            name="worker",
                            type="worker",
                            loop_policy=loop_policy,
                        ),
                        edges=edges,
                    ),
                    self._policy_registry(),
                )

                self.assertFalse(result.ok)
                self.assertIn(
                    code,
                    [diagnostic.code for diagnostic in result.diagnostics],
                )

    def test_validate_flow_definition_rejects_graph_cycle_with_policy(
        self,
    ) -> None:
        result = validate_flow_definition(
            self._policy_definition(
                FlowNodeDefinition(
                    name="worker",
                    type="worker",
                    loop_policy=FlowLoopPolicy(
                        max_iterations=2,
                        continue_condition=self._loop_condition("more"),
                        exit_condition=self._loop_condition("done"),
                        output_selector="worker.result.safe",
                        limit_route="limited",
                    ),
                ),
                edges=(
                    FlowEdgeDefinition(source="worker", target="finish"),
                    FlowEdgeDefinition(source="finish", target="worker"),
                    FlowEdgeDefinition(source="worker", target="limited"),
                ),
            ),
            self._policy_registry(),
        )

        self.assertFalse(result.ok)
        self.assertIn(
            "flow.cycle",
            [diagnostic.code for diagnostic in result.diagnostics],
        )

    def test_mapping_private_helpers_cover_type_edges(self) -> None:
        registry = FlowNodeRegistry({"open": lambda definition: Node("open")})
        schema_definition = FlowDefinition(
            name="schema",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                    schema={
                        "type": "object",
                        "properties": {"known": {"type": "string"}},
                    },
                ),
            ),
            nodes=(FlowNodeDefinition(name="start", type="schema_node"),),
        )
        schema_registry = FlowNodeRegistry(
            {"schema_node": lambda definition: Node(definition.name)},
            {
                "schema_node": FlowNodeMetadata(
                    output_contracts=(
                        FlowNodeContract(
                            name="result",
                            schema={
                                "type": "object",
                                "properties": {
                                    "known": {"type": "string"},
                                },
                            },
                        ),
                    ),
                )
            },
        )
        unnamed_output_registry = FlowNodeRegistry(
            {"schema_node": lambda definition: Node(definition.name)},
            {
                "schema_node": FlowNodeMetadata(
                    output_contracts=(FlowNodeContract(name=None),),
                )
            },
        )

        self.assertIsNone(
            flow_validator._flow_input_type(  # type: ignore[attr-defined]
                FlowDefinition(name="flow", nodes=()),
                "missing",
            )
        )
        self.assertIsNone(
            flow_validator._node_output_type(  # type: ignore[attr-defined]
                registry,
                "missing",
                "result",
            )
        )
        self.assertIsNone(
            flow_validator._node_output_type(  # type: ignore[attr-defined]
                registry,
                "open",
                "result",
            )
        )
        self.assertEqual(
            flow_validator._flow_input_selector_type(  # type: ignore[attr-defined]
                schema_definition,
                parse_flow_selector("inputs.payload.known"),
            ),
            "string",
        )
        self.assertIsNone(
            flow_validator._flow_input_selector_type(  # type: ignore[attr-defined]
                schema_definition,
                parse_flow_selector("inputs.payload.missing"),
            )
        )
        self.assertEqual(
            flow_validator._node_output_selector_type(  # type: ignore[attr-defined]
                schema_registry,
                "schema_node",
                parse_flow_selector("start.result.known"),
            ),
            "string",
        )
        self.assertIsNone(
            flow_validator._node_output_selector_type(  # type: ignore[attr-defined]
                schema_registry,
                "schema_node",
                parse_flow_selector("start.result.missing"),
            )
        )
        self.assertFalse(
            flow_validator._supports_output_name(  # type: ignore[attr-defined]
                unnamed_output_registry,
                "schema_node",
                "result",
            )
        )
        self.assertEqual(
            flow_validator._validate_mapping(  # type: ignore[attr-defined]
                schema_definition,
                FlowNodeDefinition(name="target", type="schema_node"),
                FlowInputMapping(
                    target="value",
                    kind=FlowMappingKind.COALESCE,
                    sources=("input.payload",),
                ),
                path="nodes.target.mappings[0]",
                input_names={"payload"},
                incoming_sources=set(),
                node_names={"target"},
                registry=schema_registry,
                target_contract=FlowNodeContract(
                    name="value",
                    type=FlowInputType.OBJECT,
                ),
            ),
            (),
        )
        self.assertIsNone(
            flow_validator._mapping_type_compatibility(  # type: ignore[attr-defined]
                FlowInputMapping(
                    target="value",
                    source="input.payload",
                ),
                source_type=FlowInputType.OBJECT,
                target_contract=FlowNodeContract(name="value"),
            )
        )
        self.assertIsNone(
            flow_validator._mapping_type_compatibility(  # type: ignore[attr-defined]
                FlowInputMapping(
                    target="value",
                    source="input.payload",
                ),
                source_type=FlowInputType.OBJECT,
                target_contract=FlowNodeContract(name="value", type="custom"),
            )
        )
        self.assertIsNone(
            flow_validator._mapping_type_compatibility(  # type: ignore[attr-defined]
                FlowInputMapping(
                    target="value",
                    kind=FlowMappingKind.FILE_ARRAY,
                    source="input.files",
                ),
                source_type=FlowInputType.FILE_ARRAY,
                target_contract=FlowNodeContract(
                    name="value",
                    type=FlowInputType.FILE_ARRAY,
                ),
            )
        )
        self.assertEqual(
            flow_validator._mapping_type_compatibility(  # type: ignore[attr-defined]
                FlowInputMapping(
                    target="value",
                    kind=FlowMappingKind.FILE_ARRAY,
                    source="input.payload",
                ),
                source_type=FlowInputType.OBJECT,
                target_contract=FlowNodeContract(
                    name="value",
                    type=FlowInputType.FILE_ARRAY,
                ),
            ),
            "flow.incompatible_mapping",
        )
        cases = (
            (FlowInputType.FILE, "file"),
            (FlowInputType.FILE_ARRAY, "file[]"),
            (FlowInputType.ARRAY, "array"),
            (FlowInputType.STRING, "string"),
            (FlowInputType.INTEGER, "integer"),
            (FlowInputType.NUMBER, "number"),
            (FlowInputType.BOOLEAN, "boolean"),
        )
        for input_type, expected in cases:
            with self.subTest(input_type=input_type):
                self.assertEqual(
                    flow_validator._semantic_type(  # type: ignore[attr-defined]
                        input_type,
                    ),
                    expected,
                )
        self.assertIsNone(
            flow_validator._semantic_type(  # type: ignore[attr-defined]
                "custom",
            )
        )

    def test_validate_flow_definition_rejects_metadata_gaps(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {
                "custom": lambda definition: Node(definition.name),
                "tool": lambda definition: Node(definition.name),
            },
            {
                "tool": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    supports_ref=True,
                    output_contract=FlowNodeContract(name="result"),
                    requires_ref=True,
                    required_config_keys=("mode",),
                ),
            },
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(
                    FlowNodeDefinition(name="start", type="custom"),
                    FlowNodeDefinition(name="tool", type="tool"),
                ),
                output_selector="tool.missing",
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            [
                "flow.missing_node_kind",
                "flow.missing_ref",
                "flow.missing_node_config",
                "flow.unknown_node_output",
            ],
        )

    def test_validate_flow_definition_rejects_conflicting_node_kind(
        self,
    ) -> None:
        registry = FlowNodeRegistry(
            {"input": lambda definition: Node(definition.name)},
            {"input": FlowNodeMetadata(kind=FlowNodeKind.TOOL)},
        )

        result = validate_flow_definition(
            self._strict_definition(
                nodes=(FlowNodeDefinition(name="start", type="input"),)
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.invalid_node_kind"],
        )

    def test_validate_flow_definition_rejects_duplicate_nodes(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="duplicate",
                entrypoint="start",
                output_node="start",
                nodes=(
                    FlowNodeDefinition(name="start", type="echo"),
                    FlowNodeDefinition(name="start", type="echo"),
                ),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.duplicate_node")
        self.assertEqual(result.diagnostics[0].path, "nodes.start")

    def test_validate_flow_definition_rejects_missing_legacy_run_fields(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="missing",
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.missing_entrypoint", "flow.missing_output_node"],
        )

    def test_validate_flow_definition_rejects_missing_strict_contracts(
        self,
    ) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="strict",
                version="2026-06-07",
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(outputs={}),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(
            [diagnostic.code for diagnostic in result.diagnostics],
            ["flow.missing_inputs", "flow.missing_outputs"],
        )

    def test_validate_flow_definition_rejects_private_refs(self) -> None:
        registry = FlowNodeRegistry(
            {"external": lambda definition: Node(definition.name)},
            {"external": FlowNodeMetadata(supports_ref=True)},
        )
        cases = ("../secret", "/secret", "https://host/secret", "dir\\secret")

        for ref in cases:
            with self.subTest(ref=ref):
                result = validate_flow_definition(
                    FlowDefinition(
                        name="private_ref",
                        entrypoint="start",
                        output_node="start",
                        nodes=(
                            FlowNodeDefinition(
                                name="start",
                                type="external",
                                ref=ref,
                            ),
                        ),
                    ),
                    registry,
                )

                self.assertFalse(result.ok)
                self.assertEqual(
                    result.diagnostics[0].code, "flow.path_escape"
                )
                self.assertEqual(
                    result.diagnostics[0].category,
                    FlowDiagnosticCategory.PRIVACY,
                )
                self.assertNotIn("secret", str(result.public_diagnostics))

    def test_validate_flow_definition_rejects_private_schema_refs(
        self,
    ) -> None:
        cases = (
            (
                FlowDefinition(
                    name="private_schema_ref",
                    entrypoint="start",
                    output_node="start",
                    input=FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                        schema_ref="../private/input.json",
                    ),
                    nodes=(FlowNodeDefinition(name="start", type="echo"),),
                ),
                "flow.input.schema_ref",
                "flow.path_escape",
            ),
            (
                FlowDefinition(
                    name="private_schema_ref",
                    entrypoint="start",
                    output_node="start",
                    output=FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                        schema_ref="https://host/secret.json",
                    ),
                    nodes=(FlowNodeDefinition(name="start", type="echo"),),
                ),
                "flow.output.schema_ref",
                "flow.path_escape",
            ),
            (
                FlowDefinition(
                    name="private_schema_ref",
                    version="2026-06-07",
                    inputs=(
                        FlowInputDefinition(
                            name="payload",
                            type=FlowInputType.OBJECT,
                            schema_ref="schema.json#/secret",
                        ),
                    ),
                    outputs=(
                        FlowOutputDefinition(
                            name="answer",
                            type=FlowOutputType.OBJECT,
                        ),
                    ),
                    entry_behavior=FlowEntryBehavior(node="start"),
                    output_behavior=FlowOutputBehavior(
                        outputs={"answer": "start.value"},
                    ),
                    nodes=(FlowNodeDefinition(name="start", type="echo"),),
                ),
                "flow.inputs.payload.schema_ref",
                "flow.invalid_schema_ref",
            ),
            (
                FlowDefinition(
                    name="private_schema_ref",
                    version="2026-06-07",
                    inputs=(
                        FlowInputDefinition(
                            name="payload",
                            type=FlowInputType.OBJECT,
                        ),
                    ),
                    outputs=(
                        FlowOutputDefinition(
                            name="answer",
                            type=FlowOutputType.OBJECT,
                            schema_ref="C:/secret/schema.json",
                        ),
                    ),
                    entry_behavior=FlowEntryBehavior(node="start"),
                    output_behavior=FlowOutputBehavior(
                        outputs={"answer": "start.value"},
                    ),
                    nodes=(FlowNodeDefinition(name="start", type="echo"),),
                ),
                "flow.outputs.answer.schema_ref",
                "flow.path_escape",
            ),
        )

        for definition, path, code in cases:
            with self.subTest(path=path):
                result = validate_flow_definition(definition)

                self.assertFalse(result.ok)
                matching = [
                    diagnostic
                    for diagnostic in result.diagnostics
                    if diagnostic.path == path
                ]
                self.assertEqual(matching[0].code, code)
                self.assertEqual(
                    matching[0].category,
                    FlowDiagnosticCategory.PRIVACY,
                )
                self.assertNotIn("secret", str(result.public_diagnostics))
                self.assertNotIn("private", str(result.public_diagnostics))

    def test_validate_flow_definition_accepts_safe_schema_refs(self) -> None:
        result = validate_flow_definition(
            FlowDefinition(
                name="safe_schema_ref",
                version="2026-06-07",
                inputs=(
                    FlowInputDefinition(
                        name="payload",
                        type=FlowInputType.OBJECT,
                        schema_ref="schemas/input.json",
                    ),
                ),
                outputs=(
                    FlowOutputDefinition(
                        name="answer",
                        type=FlowOutputType.OBJECT,
                        schema_ref="schemas/output.json",
                    ),
                ),
                entry_behavior=FlowEntryBehavior(node="start"),
                output_behavior=FlowOutputBehavior(
                    outputs={"answer": "start.value"},
                ),
                nodes=(FlowNodeDefinition(name="start", type="echo"),),
            )
        )

        self.assertTrue(result.ok, result.public_diagnostics)

    def test_loader_validates_semantics_before_node_factories(self) -> None:
        calls: list[str] = []

        def factory(definition: FlowNodeDefinition) -> Node:
            calls.append(definition.name)
            return Node(definition.name)

        result = run_async(
            flow_loader.FlowDefinitionLoader(
                FlowNodeRegistry({"counted": factory})
            ).loads_result(
                """
            [flow]
            name = "invalid"
            entrypoint = "start"
            output_node = "missing"

            [nodes.start]
            type = "counted"
            """
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.unknown_output_node")
        self.assertEqual(calls, [])

    def test_loader_maps_validation_diagnostics_to_load_issues(self) -> None:
        result = run_async(flow_loader.loads_flow_definition_result("""
            [flow]
            name = "invalid"
            entrypoint = "start"
            output_node = "start"

            [nodes.start]
            type = "missing"
            """))

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].code, "flow.unknown_node_type")
        self.assertEqual(result.issues[0].category.value, "unsupported")
        self.assertEqual(
            result.diagnostics[0].category,
            FlowDiagnosticCategory.FLOW_DEFINITION_VALIDATION,
        )

    def _join_policy_registry(self) -> FlowNodeRegistry:
        return FlowNodeRegistry(
            {
                "source": lambda definition: Node(definition.name),
                "target": lambda definition: Node(definition.name),
            },
            {
                "source": FlowNodeMetadata(
                    kind=FlowNodeKind.SELECT,
                    output_contract=FlowNodeContract(
                        name="payload",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                "target": FlowNodeMetadata(
                    kind=FlowNodeKind.JOIN,
                    input_contracts=(
                        FlowNodeContract(
                            name="payload",
                            type=FlowInputType.OBJECT,
                        ),
                        FlowNodeContract(
                            name="audit",
                            type=FlowInputType.OBJECT,
                        ),
                    ),
                    output_contract=FlowNodeContract(
                        name="result",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
            },
        )

    def _join_policy_contract_definition(
        self,
        join_policy: FlowJoinPolicy,
        *,
        include_payload_mapping: bool = False,
    ) -> FlowDefinition:
        mappings: tuple[FlowInputMapping, ...] = ()
        if include_payload_mapping:
            mappings = (
                FlowInputMapping(
                    target="payload",
                    source="left.payload",
                ),
            )
        return FlowDefinition(
            name="joined",
            version="2026-06-07",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="left"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "finish.result"},
            ),
            nodes=(
                FlowNodeDefinition(name="left", type="source"),
                FlowNodeDefinition(name="right", type="source"),
                FlowNodeDefinition(
                    name="finish",
                    type="target",
                    join_policy=join_policy,
                    mappings=mappings,
                ),
            ),
            edges=(
                FlowEdgeDefinition(source="left", target="finish"),
                FlowEdgeDefinition(source="right", target="finish"),
            ),
        )

    def _loop_condition(self, field: str) -> FlowCondition:
        return FlowCondition(
            operator=FlowConditionOperator.EXISTS,
            selector=f"worker.result.{field}",
        )

    def _policy_registry(self) -> FlowNodeRegistry:
        return FlowNodeRegistry(
            {"worker": lambda definition: Node(definition.name)},
            {
                "worker": FlowNodeMetadata(
                    kind=FlowNodeKind.TOOL,
                    output_contract=FlowNodeContract(
                        name="result",
                        type=FlowOutputType.OBJECT,
                        schema={
                            "type": "object",
                            "properties": {
                                "done": {"type": "boolean"},
                                "more": {"type": "boolean"},
                                "safe": {"type": "object"},
                            },
                        },
                    ),
                ),
            },
        )

    def _policy_definition(
        self,
        worker: FlowNodeDefinition,
        *,
        edges: tuple[FlowEdgeDefinition, ...] | None = None,
    ) -> FlowDefinition:
        if edges is None:
            edges = (
                FlowEdgeDefinition(source="worker", target="finish"),
                FlowEdgeDefinition(
                    source="worker",
                    target="failed",
                    kind=FlowEdgeKind.ERROR,
                ),
                FlowEdgeDefinition(
                    source="worker",
                    target="limited",
                    kind=FlowEdgeKind.TIMEOUT,
                ),
            )
        return FlowDefinition(
            name="policies",
            version="2026-06-07",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="worker"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "finish.result"},
            ),
            nodes=(
                worker,
                FlowNodeDefinition(name="finish", type="worker"),
                FlowNodeDefinition(name="failed", type="worker"),
                FlowNodeDefinition(name="limited", type="worker"),
                FlowNodeDefinition(name="other", type="worker"),
            ),
            edges=edges,
        )

    def _human_review_config(
        self,
        *,
        allowed_decisions: object = (
            "approved",
            "rejected",
            "needs_correction",
        ),
        decision_schema: object | None = None,
        payload_schema: object | None = None,
        timeout_seconds: int | float = 3600,
        audit_metadata: object | None = None,
    ) -> dict[str, object]:
        config: dict[str, object] = {
            "allowed_decisions": allowed_decisions,
            "decision_schema": (
                decision_schema
                or {
                    "type": "object",
                    "properties": {
                        "decision": {"enum": allowed_decisions},
                        "comment": {"type": "string"},
                    },
                    "required": ("decision",),
                }
            ),
            "payload_schema": (
                payload_schema
                or {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ("summary",),
                }
            ),
            "timeout_seconds": timeout_seconds,
        }
        if audit_metadata is not None:
            config["audit_metadata"] = audit_metadata
        return config

    def _human_review_registry(
        self, *, durable: bool = True
    ) -> FlowNodeRegistry:
        capabilities: tuple[FlowNodeCapability, ...] = ()
        if durable:
            capabilities = (FlowNodeCapability.DURABLE_PAUSE,)
        return FlowNodeRegistry(
            {
                "human_review": lambda definition: Node(definition.name),
                "review_sink": lambda definition: Node(definition.name),
            },
            {
                "human_review": FlowNodeMetadata(
                    kind=FlowNodeKind.HUMAN_REVIEW,
                    async_only=True,
                    capabilities=capabilities,
                    input_contract=FlowNodeContract(
                        name="payload",
                        type=FlowInputType.OBJECT,
                    ),
                    output_contract=FlowNodeContract(
                        name="result",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
                "review_sink": FlowNodeMetadata(
                    kind=FlowNodeKind.PASS_THROUGH,
                    output_contract=FlowNodeContract(
                        name="result",
                        type=FlowOutputType.OBJECT,
                    ),
                ),
            },
        )

    def _human_review_definition(
        self,
        review: FlowNodeDefinition,
        *,
        edges: tuple[FlowEdgeDefinition, ...] | None = None,
    ) -> FlowDefinition:
        if edges is None:
            edges = (
                FlowEdgeDefinition(
                    source="review",
                    target="finish",
                    label="approved",
                    kind=FlowEdgeKind.RESUME,
                ),
                FlowEdgeDefinition(
                    source="review",
                    target="rejected",
                    label="rejected",
                    kind=FlowEdgeKind.RESUME,
                    priority=1,
                ),
                FlowEdgeDefinition(
                    source="review",
                    target="correction",
                    label="needs_correction",
                    kind=FlowEdgeKind.RESUME,
                    priority=2,
                ),
                FlowEdgeDefinition(
                    source="review",
                    target="expired",
                    label="expired",
                    kind=FlowEdgeKind.TIMEOUT,
                ),
            )
        return FlowDefinition(
            name="review-flow",
            version="2026-06-08",
            inputs=(
                FlowInputDefinition(
                    name="payload",
                    type=FlowInputType.OBJECT,
                ),
            ),
            outputs=(
                FlowOutputDefinition(
                    name="answer",
                    type=FlowOutputType.OBJECT,
                ),
            ),
            entry_behavior=FlowEntryBehavior(node="review"),
            output_behavior=FlowOutputBehavior(
                outputs={"answer": "finish.result"},
            ),
            nodes=(
                review,
                FlowNodeDefinition(name="finish", type="review_sink"),
                FlowNodeDefinition(name="rejected", type="review_sink"),
                FlowNodeDefinition(name="correction", type="review_sink"),
                FlowNodeDefinition(name="expired", type="review_sink"),
            ),
            edges=edges,
        )


if __name__ == "__main__":
    main()
