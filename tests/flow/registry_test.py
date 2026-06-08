from asyncio import CancelledError
from collections.abc import Mapping
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallDiagnosticStatus,
    ToolCallOutcome,
    ToolCallRecoveryFormat,
    ToolCallResult,
    ToolDescriptor,
    ToolFilterResult,
    ToolFilterResultStatus,
    ToolManagerSettings,
    ToolNameResolution,
    ToolNameResolutionStatus,
)
from avalan.flow import (
    FLOW_INPUT_KEY,
    FLOW_TOOL_NODE_TYPE,
    FlowDefinition,
    FlowInputDefinition,
    FlowInputType,
    FlowNodeCapability,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeKind,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputType,
    default_flow_node_registry,
    flow_input_binding,
    tool_flow_node_registry,
    validate_flow_definition,
)
from avalan.flow.node import Node
from avalan.flow.registry import FlowNodeConfigurationError
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.mcp import McpToolSet


async def flow_adder(a: int, b: int) -> int:
    return a + b


flow_adder.aliases = ["sum"]  # type: ignore[attr-defined]


async def flow_adder_alt(a: int, b: int) -> int:
    return a + b


flow_adder_alt.aliases = ["sum"]  # type: ignore[attr-defined]


async def flow_multiplier(a: int, b: int) -> int:
    return a * b


async def flow_disabled(a: int) -> int:
    return a


async def flow_identity(value: int) -> int:
    return value


async def flow_none(value: int) -> None:
    return None


async def flow_status() -> str:
    return "ready"


async def flow_fails(value: int) -> None:
    raise ValueError(f"bad {value}")


async def flow_private_fails(value: str) -> None:
    raise ValueError(f"bad {value}")


def _tool_manager(
    *,
    enable_tools: list[str] | None = None,
) -> ToolManager:
    return ToolManager.create_instance(
        enable_tools=enable_tools
        or [
            "flow_adder",
            "flow_identity",
            "flow_none",
            "mcp.call",
        ],
        available_toolsets=[
            ToolSet(
                tools=[
                    flow_adder,
                    flow_adder_alt,
                    flow_multiplier,
                    flow_identity,
                    flow_none,
                    flow_status,
                    flow_fails,
                ]
            ),
            ToolSet(namespace="disabled", tools=[flow_disabled]),
            McpToolSet(),
        ],
        settings=ToolManagerSettings(),
    )


class RecordingToolResolver:
    def __init__(self, manager: ToolManager) -> None:
        self.manager = manager
        self.calls: list[ToolCall] = []
        self.contexts: list[ToolCallContext] = []

    def list_tools(self) -> list[ToolDescriptor]:
        return self.manager.list_tools()

    def resolve_tool_name(
        self, name: str, *, provider_originated: bool = False
    ) -> ToolNameResolution:
        return self.manager.resolve_tool_name(
            name,
            provider_originated=provider_originated,
        )

    def validate_tool_call(self, call: ToolCall) -> ToolCallDiagnostic | None:
        return self.manager.validate_tool_call(call)

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> ToolCallOutcome:
        self.calls.append(call)
        self.contexts.append(context)
        return await self.manager.execute_call(call, context)


class StaticToolResolver:
    def __init__(self, descriptors: list[ToolDescriptor]) -> None:
        self.descriptors = descriptors

    def list_tools(self) -> list[ToolDescriptor]:
        return self.descriptors

    def resolve_tool_name(
        self, name: str, *, provider_originated: bool = False
    ) -> ToolNameResolution:
        return ToolNameResolution(
            requested_name=name,
            status=ToolNameResolutionStatus.EXACT,
            canonical_name=name,
            candidates=[name],
        )

    def validate_tool_call(self, call: ToolCall) -> ToolCallDiagnostic | None:
        return None

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> ToolCallOutcome:
        return ToolCallResult(
            id="result-1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=call.arguments or {},
        )


class ValidatingToolResolver(StaticToolResolver):
    def __init__(self, descriptors: list[ToolDescriptor]) -> None:
        super().__init__(descriptors)
        self.executed = False
        self.validated: list[ToolCall] = []

    def validate_tool_call(self, call: ToolCall) -> ToolCallDiagnostic | None:
        self.validated.append(call)
        return ToolCallDiagnostic(
            id="diagnostic-1",
            call_id=call.id,
            requested_name=call.name,
            canonical_name=call.name,
            code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
            stage=ToolCallDiagnosticStage.VALIDATE,
            message="Tool node arguments are invalid.",
        )

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
    ) -> ToolCallOutcome:
        self.executed = True
        return await super().execute_call(call, context)


class FlowNodeRegistryTestCase(IsolatedAsyncioTestCase):
    async def test_default_nodes_execute_without_mutating_inputs(
        self,
    ) -> None:
        registry = default_flow_node_registry()
        payload = {"items": ["first"]}
        binding = flow_input_binding(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            payload,
        )
        payload["items"].append("changed")

        input_node = registry.build(
            FlowNodeDefinition(name="input", type="input", input="payload")
        )
        echo_node = registry.build(
            FlowNodeDefinition(name="echo", type="echo", input="input")
        )
        select_node = registry.build(
            FlowNodeDefinition(
                name="select",
                type="select",
                input="input",
                config={"path": "items.0"},
            )
        )
        constant_node = registry.build(
            FlowNodeDefinition(
                name="constant",
                type="constant",
                config={"value": {"answer": "ok"}},
            )
        )

        input_value = await input_node.execute_async(binding)
        self.assertEqual(input_value, {"items": ["first"]})
        self.assertEqual(
            await echo_node.execute_async({"input": input_value}),
            input_value,
        )
        self.assertEqual(
            await select_node.execute_async({"input": input_value}),
            "first",
        )
        self.assertEqual(
            await constant_node.execute_async({}),
            {"answer": "ok"},
        )
        self.assertEqual(payload, {"items": ["first", "changed"]})

    def test_binding_handles_scalar_mapping_and_missing_definition(
        self,
    ) -> None:
        scalar = flow_input_binding(
            FlowInputDefinition(name="text", type=FlowInputType.STRING),
            "ready",
        )
        no_definition = flow_input_binding(None, "ready")
        no_definition_empty = flow_input_binding(None, None)
        mapping = flow_input_binding(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            {"name": "Ada"},
        )

        self.assertEqual(scalar[FLOW_INPUT_KEY], "ready")
        self.assertEqual(scalar["text"], "ready")
        self.assertEqual(scalar["value"], "ready")
        self.assertEqual(no_definition["value"], "ready")
        self.assertEqual(no_definition_empty, {FLOW_INPUT_KEY: None})
        self.assertEqual(mapping["payload"], {"name": "Ada"})
        self.assertNotIn("value", mapping)

    def test_registry_definition_validator_reports_without_building(
        self,
    ) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            raise AssertionError(definition.name)

        def validator(
            definition: FlowDefinition,
            node: FlowNodeDefinition,
        ) -> tuple[FlowNodeConfigurationError, ...]:
            self.assertEqual(definition.name, "checked")
            return (
                FlowNodeConfigurationError(
                    code="flow.invalid_node",
                    path=f"nodes.{node.name}.config",
                    message="Flow node configuration is invalid.",
                    hint="Use valid node configuration.",
                ),
            )

        registry = FlowNodeRegistry(
            {"checked": factory},
            {"checked": FlowNodeMetadata(kind=FlowNodeKind.PASS_THROUGH)},
            {"checked": validator},
        )

        result = validate_flow_definition(
            FlowDefinition(
                name="checked",
                entrypoint="start",
                output_node="start",
                nodes=(FlowNodeDefinition(name="start", type="checked"),),
            ),
            registry,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "flow.invalid_node")
        self.assertEqual(
            registry.validate_node_definition(
                FlowDefinition(
                    name="empty",
                    entrypoint="start",
                    output_node="start",
                    nodes=(FlowNodeDefinition(name="start", type="missing"),),
                ),
                FlowNodeDefinition(name="start", type="missing"),
            ),
            (),
        )

    async def test_echo_node_falls_back_to_available_inputs(self) -> None:
        node = default_flow_node_registry().build(
            FlowNodeDefinition(name="echo", type="echo")
        )

        self.assertEqual(
            await node.execute_async({FLOW_INPUT_KEY: "flow"}), "flow"
        )
        self.assertEqual(await node.execute_async({"only": "value"}), "value")
        self.assertEqual(
            await node.execute_async({"left": "L", "right": "R"}),
            {"left": "L", "right": "R"},
        )

    async def test_custom_registry_and_select_errors(self) -> None:
        registry = FlowNodeRegistry()

        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(
                definition.name,
                func=lambda inputs: str(inputs["value"]).upper(),
            )

        registry.register("upper", factory)

        self.assertTrue(registry.supports("upper"))
        self.assertEqual(
            await registry.build(
                FlowNodeDefinition(name="upper", type="upper")
            ).execute_async({"value": "ok"}),
            "OK",
        )

        select_node = default_flow_node_registry().build(
            FlowNodeDefinition(
                name="select",
                type="select",
                input="input",
                config={"field": "missing"},
            )
        )
        with self.assertRaises(KeyError):
            await select_node.execute_async({"input": {"answer": "ok"}})

    async def test_native_strict_nodes_execute_data_primitives(self) -> None:
        registry = default_flow_node_registry()
        input_node = registry.build(
            FlowNodeDefinition(name="raw", type="input")
        )
        passthrough = registry.build(
            FlowNodeDefinition(name="pass", type="pass-through")
        )
        select_node = registry.build(
            FlowNodeDefinition(name="project", type="select")
        )
        validation = registry.build(
            FlowNodeDefinition(
                name="check",
                type="validation",
                config={
                    "value_type": "object",
                    "required_fields": ("name",),
                },
            )
        )
        decision = registry.build(
            FlowNodeDefinition(name="decide", type="decision")
        )
        notification = registry.build(
            FlowNodeDefinition(
                name="notice",
                type="notification",
                config={"channel": "audit"},
            )
        )
        default_notification = registry.build(
            FlowNodeDefinition(name="notice_default", type="notification")
        )
        join = registry.build(FlowNodeDefinition(name="join", type="join"))

        value = {"name": "Ada", "approved": True}

        self.assertEqual(
            await input_node.execute_async({"value": value}), value
        )
        self.assertEqual(
            await passthrough.execute_async({"value": value}), value
        )
        self.assertEqual(
            await select_node.execute_async({"value": value}), value
        )
        self.assertEqual(
            await validation.execute_async({"value": value}), value
        )
        self.assertEqual(await decision.execute_async({"value": value}), value)
        self.assertEqual(
            await notification.execute_async({"value": value}),
            {
                "status": "notified",
                "payload": value,
                "channel": "audit",
            },
        )
        self.assertEqual(
            await default_notification.execute_async({"value": value}),
            {"status": "notified", "payload": value},
        )
        self.assertEqual(
            await join.execute_async({"left": "L", "right": "R"}),
            {"left": "L", "right": "R"},
        )

    async def test_native_validation_node_supports_value_types(self) -> None:
        cases: tuple[tuple[str, object], ...] = (
            ("array", ["one"]),
            ("boolean", True),
            ("integer", 1),
            ("null", None),
            ("number", 1.5),
            ("string", "ready"),
        )
        registry = default_flow_node_registry()

        for value_type, value in cases:
            with self.subTest(value_type=value_type):
                node = registry.build(
                    FlowNodeDefinition(
                        name="check",
                        type="validation",
                        config={"value_type": value_type},
                    )
                )

                self.assertEqual(
                    await node.execute_async({"value": value}),
                    value,
                )

    async def test_native_strict_nodes_reject_invalid_config(self) -> None:
        cases = (
            (
                FlowNodeDefinition(
                    name="project",
                    type="select",
                    config={"path": ""},
                ),
                "nodes.project.config.path",
            ),
            (
                FlowNodeDefinition(
                    name="check",
                    type="validation",
                    config={"required_fields": "name"},
                ),
                "nodes.check.config.required_fields",
            ),
            (
                FlowNodeDefinition(
                    name="check",
                    type="validation",
                    config={"required_fields": ("",)},
                ),
                "nodes.check.config.required_fields",
            ),
            (
                FlowNodeDefinition(
                    name="check",
                    type="validation",
                    config={"value_type": "unsupported"},
                ),
                "nodes.check.config.value_type",
            ),
            (
                FlowNodeDefinition(
                    name="notice",
                    type="notification",
                    config={"channel": ""},
                ),
                "nodes.notice.config.channel",
            ),
        )
        registry = default_flow_node_registry()

        for definition, path in cases:
            with self.subTest(node=definition.name, type=definition.type):
                with self.assertRaises(FlowNodeConfigurationError) as context:
                    await registry.build(definition).execute_async(
                        {"value": {"name": "Ada"}}
                    )
                self.assertEqual(
                    context.exception.code,
                    "flow.invalid_node_config",
                )
                self.assertEqual(context.exception.path, path)

    async def test_native_validation_node_rejects_invalid_values(
        self,
    ) -> None:
        cases = (
            (
                FlowNodeDefinition(
                    name="check",
                    type="validation",
                    config={"value_type": "object"},
                ),
                {"value": "not-object"},
            ),
            (
                FlowNodeDefinition(
                    name="check",
                    type="validation",
                    config={"required_fields": ("name",)},
                ),
                {"value": "not-object"},
            ),
            (
                FlowNodeDefinition(
                    name="check",
                    type="validation",
                    config={"required_fields": ("name",)},
                ),
                {"value": {"age": 37}},
            ),
        )
        registry = default_flow_node_registry()

        for definition, inputs in cases:
            with self.subTest(config=definition.config):
                with self.assertRaises(FlowNodeConfigurationError):
                    await registry.build(definition).execute_async(inputs)

    def test_registry_tracks_subflow_resolver_metadata(self) -> None:
        class Resolver:
            def compile_subflow(
                self,
                ref: str,
                *,
                parent_definition: FlowDefinition,
                node: FlowNodeDefinition,
                registry: FlowNodeRegistry,
            ) -> Mapping[str, object]:
                return {
                    "ref": ref,
                    "parent": parent_definition.name,
                    "node": node.name,
                    "registered": registry.supports_subflow_resolution(
                        node.type
                    ),
                }

        registry = FlowNodeRegistry()
        resolver = Resolver()

        self.assertIs(
            registry.register_subflow_resolver("subflow", resolver),
            registry,
        )
        self.assertTrue(registry.supports_subflow_resolution("subflow"))
        self.assertEqual(
            registry.subflow_metadata(
                FlowDefinition(
                    name="parent",
                    entrypoint="child",
                    output_node="child",
                    nodes=(
                        FlowNodeDefinition(
                            name="child",
                            type="subflow",
                            ref="flows/child.toml",
                        ),
                    ),
                ),
                FlowNodeDefinition(
                    name="child",
                    type="subflow",
                    ref="flows/child.toml",
                ),
            ),
            {
                "ref": "flows/child.toml",
                "parent": "parent",
                "node": "child",
                "registered": True,
            },
        )
        with self.assertRaises(AssertionError):
            registry.register_subflow_resolver("", resolver)
        with self.assertRaises(AssertionError):
            registry.register_subflow_resolver(
                "bad",
                object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            registry.subflow_metadata(
                object(),  # type: ignore[arg-type]
                FlowNodeDefinition(name="child", type="subflow"),
            )
        with self.assertRaises(AssertionError):
            registry.subflow_metadata(
                FlowDefinition(
                    name="parent",
                    entrypoint="child",
                    output_node="child",
                    nodes=(FlowNodeDefinition(name="child", type="subflow"),),
                ),
                object(),  # type: ignore[arg-type]
            )

    def test_registry_exposes_node_metadata(self) -> None:
        registry = default_flow_node_registry()

        self.assertFalse(registry.supports_ref("echo"))
        self.assertFalse(registry.is_async_only("echo"))
        self.assertIsNone(registry.metadata("missing"))
        self.assertIsNone(registry.input_contract("missing"))
        self.assertIsNone(registry.output_contract("missing"))
        echo_input = registry.input_contract("echo")
        echo_output = registry.output_contract("echo")
        echo_metadata = registry.metadata("echo")
        decision_metadata = registry.metadata("decision")
        join_metadata = registry.metadata("join")
        notification_metadata = registry.metadata("notification")
        validation_metadata = registry.metadata("validation")

        assert echo_input is not None
        assert echo_output is not None
        assert echo_metadata is not None
        assert decision_metadata is not None
        assert join_metadata is not None
        assert notification_metadata is not None
        assert validation_metadata is not None
        self.assertEqual(echo_metadata.kind, FlowNodeKind.PASS_THROUGH)
        self.assertEqual(decision_metadata.kind, FlowNodeKind.DECISION)
        self.assertEqual(join_metadata.kind, FlowNodeKind.JOIN)
        self.assertEqual(
            notification_metadata.kind,
            FlowNodeKind.NOTIFICATION,
        )
        self.assertEqual(validation_metadata.kind, FlowNodeKind.VALIDATION)
        self.assertIn(
            FlowNodeCapability.DIRECT_ASYNC,
            echo_metadata.capabilities,
        )
        self.assertEqual(echo_input.name, "value")
        self.assertTrue(echo_input.metadata["dynamic"])
        self.assertEqual(echo_output.name, "value")
        self.assertTrue(echo_output.metadata["dynamic"])

    async def test_custom_registry_metadata_supports_refs_and_contracts(
        self,
    ) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(definition.name, func=lambda _: definition.ref)

        metadata = FlowNodeMetadata(
            kind=FlowNodeKind.SUBFLOW,
            supports_ref=True,
            async_only=True,
            input_contract=FlowNodeContract(
                name="payload",
                type=FlowInputType.OBJECT,
                schema={"type": "object"},
            ),
            output_contract=FlowNodeContract(
                name="result",
                type=FlowOutputType.JSON,
                schema={"type": "string"},
            ),
            capabilities=(FlowNodeCapability.TASK_BACKED,),
            requires_ref=True,
            required_config_keys=("mode",),
        )
        registry = FlowNodeRegistry({"remote": factory}, {"remote": metadata})

        self.assertTrue(registry.supports("remote"))
        self.assertTrue(registry.supports_ref("remote"))
        self.assertTrue(registry.is_async_only("remote"))
        remote_metadata = registry.metadata("remote")
        assert remote_metadata is not None
        self.assertEqual(remote_metadata.kind, FlowNodeKind.SUBFLOW)
        self.assertTrue(remote_metadata.requires_ref)
        self.assertEqual(
            remote_metadata.required_config_keys,
            ("mode",),
        )
        self.assertEqual(
            registry.input_contract("remote"), metadata.input_contract
        )
        self.assertEqual(
            registry.output_contract("remote"),
            metadata.output_contract,
        )
        self.assertEqual(
            await registry.build(
                FlowNodeDefinition(
                    name="remote",
                    type="remote",
                    ref="safe.toml",
                )
            ).execute_async({}),
            "safe.toml",
        )

    def test_registry_rejects_invalid_metadata(self) -> None:
        registry = FlowNodeRegistry()

        with self.assertRaises(AssertionError):
            registry.register(
                "bad",
                lambda definition: Node(definition.name),
                metadata=object(),  # type: ignore[arg-type]
            )

    def test_configuration_error_carries_public_fields(self) -> None:
        error = FlowNodeConfigurationError(
            code="flow.invalid_node",
            path="nodes.start",
            message="Flow node configuration is invalid.",
            hint="Fix the node configuration.",
        )

        self.assertEqual(str(error), "flow.invalid_node")
        self.assertEqual(error.code, "flow.invalid_node")
        self.assertEqual(error.path, "nodes.start")
        self.assertEqual(
            error.message,
            "Flow node configuration is invalid.",
        )
        self.assertEqual(error.hint, "Fix the node configuration.")
        with self.assertRaises(AssertionError):
            FlowNodeConfigurationError(
                code="",
                path="nodes.start",
                message="Flow node configuration is invalid.",
                hint="Fix the node configuration.",
            )


class FlowToolNodeRegistryTestCase(IsolatedAsyncioTestCase):
    async def test_tool_registry_builds_async_tool_node_with_metadata(
        self,
    ) -> None:
        registry = tool_flow_node_registry(_tool_manager())

        self.assertTrue(registry.supports(FLOW_TOOL_NODE_TYPE))
        self.assertTrue(registry.supports_ref(FLOW_TOOL_NODE_TYPE))
        self.assertTrue(registry.is_async_only(FLOW_TOOL_NODE_TYPE))
        metadata = registry.metadata(FLOW_TOOL_NODE_TYPE)
        input_contract = registry.input_contract(FLOW_TOOL_NODE_TYPE)
        output_contract = registry.output_contract(FLOW_TOOL_NODE_TYPE)
        assert metadata is not None
        assert input_contract is not None
        assert output_contract is not None
        tools = metadata.metadata["tools"]
        assert isinstance(tools, Mapping)
        self.assertIn("flow_adder", tools)
        self.assertEqual(input_contract.name, "arguments")
        self.assertEqual(input_contract.type, FlowInputType.OBJECT)
        self.assertEqual(output_contract.name, "result")
        self.assertEqual(output_contract.type, FlowOutputType.JSON)

        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
            )
        )

        self.assertEqual(node.label, "flow_adder")
        self.assertTrue(node.async_only)
        self.assertEqual(
            await node.execute_async({"payload": {"a": 1, "b": 2}}),
            3,
        )

    async def test_tool_node_binds_explicit_argument_selectors(
        self,
    ) -> None:
        resolver = RecordingToolResolver(_tool_manager())
        registry = tool_flow_node_registry(resolver)
        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
                input="payload",
                config={"arguments": {"b": "right", "a": "left"}},
            )
        )

        self.assertEqual(
            await node.execute_async(
                {"payload": {"left": 2, "right": 3, "ignored": 4}}
            ),
            5,
        )

        self.assertEqual(len(resolver.calls), 1)
        self.assertEqual(resolver.calls[0].name, "flow_adder")
        self.assertEqual(resolver.calls[0].arguments, {"b": 3, "a": 2})
        self.assertTrue(resolver.contexts[0].flow_tool_node)

    async def test_tool_node_binds_and_rejects_array_argument_selectors(
        self,
    ) -> None:
        resolver = RecordingToolResolver(_tool_manager())
        registry = tool_flow_node_registry(resolver)
        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
                input="payload",
                config={"arguments": {"a": "items.0", "b": "items.1"}},
            )
        )

        self.assertEqual(
            await node.execute_async({"payload": {"items": [8, 5]}}),
            13,
        )
        self.assertEqual(resolver.calls[0].arguments, {"a": 8, "b": 5})

        with self.assertRaises(FlowNodeConfigurationError) as context:
            await node.execute_async({"payload": {"items": [8]}})

        self.assertEqual(
            context.exception.code,
            "flow.unresolved_argument_selector",
        )
        self.assertEqual(
            context.exception.path,
            "nodes.calculate.config.arguments.b",
        )

    async def test_tool_node_binds_implicit_object_by_parameter_name(
        self,
    ) -> None:
        resolver = RecordingToolResolver(_tool_manager())
        registry = tool_flow_node_registry(resolver)
        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
            )
        )

        self.assertEqual(
            await node.execute_async({"payload": {"b": 5, "a": 4}}),
            9,
        )

        self.assertEqual(resolver.calls[0].arguments, {"a": 4, "b": 5})

    async def test_tool_node_binds_single_parameter_from_whole_input(
        self,
    ) -> None:
        resolver = RecordingToolResolver(
            _tool_manager(enable_tools=["flow_identity"])
        )
        registry = tool_flow_node_registry(resolver)
        node = registry.build(
            FlowNodeDefinition(
                name="identity",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_identity",
            )
        )

        self.assertEqual(await node.execute_async({"payload": 7}), 7)

        self.assertEqual(resolver.calls[0].arguments, {"value": 7})

    async def test_tool_node_executes_no_argument_tool_without_bindings(
        self,
    ) -> None:
        resolver = RecordingToolResolver(
            _tool_manager(enable_tools=["flow_status"])
        )
        registry = tool_flow_node_registry(resolver)
        node = registry.build(
            FlowNodeDefinition(
                name="status",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_status",
            )
        )

        self.assertEqual(await node.execute_async({}), "ready")

        self.assertEqual(resolver.calls[0].arguments, {})

    async def test_tool_node_ignores_manager_recovery_formats(self) -> None:
        async def echo_text(value: str) -> str:
            return value

        payload = (
            "```tool_call\n"
            '{"name": "flow_adder", "arguments": {"a": 2, "b": 3}}\n'
            "```"
        )
        manager = ToolManager.create_instance(
            enable_tools=["echo_text", "flow_adder"],
            available_toolsets=[ToolSet(tools=[echo_text, flow_adder])],
            settings=ToolManagerSettings(
                recovery_formats=[ToolCallRecoveryFormat.FENCED],
            ),
        )
        resolver = RecordingToolResolver(manager)
        node = tool_flow_node_registry(resolver).build(
            FlowNodeDefinition(
                name="echo",
                type=FLOW_TOOL_NODE_TYPE,
                ref="echo_text",
            )
        )

        self.assertEqual(
            await node.execute_async({"payload": payload}), payload
        )
        self.assertEqual(len(resolver.calls), 1)
        self.assertEqual(resolver.calls[0].name, "echo_text")
        self.assertEqual(resolver.calls[0].arguments, {"value": payload})

    async def test_tool_node_passes_flow_cancellation_checker(self) -> None:
        resolver = RecordingToolResolver(_tool_manager())
        registry = tool_flow_node_registry(resolver)
        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
            )
        )

        async def cancel() -> None:
            return None

        self.assertEqual(
            await node.execute_async(
                {"payload": {"a": 2, "b": 4}},
                cancellation_checker=cancel,
            ),
            6,
        )
        self.assertIs(resolver.contexts[0].cancellation_checker, cancel)

    async def test_tool_node_handles_descriptor_schema_edge_cases(
        self,
    ) -> None:
        cases = (
            (
                ToolDescriptor(name="raw", parameter_schema=None),
                {"arguments": {}},
                {},
                None,
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
                {"arguments": {}},
                {},
                None,
            ),
            (
                ToolDescriptor(
                    name="bad_properties",
                    parameter_schema={
                        "type": "object",
                        "properties": [],
                    },
                ),
                {},
                {"payload": {"value": 1}},
                "flow.ambiguous_argument_binding",
            ),
        )

        for descriptor, config, inputs, code in cases:
            with self.subTest(name=descriptor.name):
                registry = tool_flow_node_registry(
                    StaticToolResolver([descriptor])
                )
                node = registry.build(
                    FlowNodeDefinition(
                        name=descriptor.name,
                        type=FLOW_TOOL_NODE_TYPE,
                        ref=descriptor.name,
                        config=config,
                    )
                )

                if code is None:
                    self.assertEqual(await node.execute_async(inputs), {})
                    continue

                with self.assertRaises(FlowNodeConfigurationError) as context:
                    await node.execute_async(inputs)
                self.assertEqual(context.exception.code, code)

    async def test_tool_node_validates_bound_arguments_before_execution(
        self,
    ) -> None:
        descriptor = ToolDescriptor(
            name="validated",
            parameter_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        )
        raw_resolver = ValidatingToolResolver([descriptor])
        raw_node = tool_flow_node_registry(raw_resolver).build(
            FlowNodeDefinition(
                name="validated",
                type=FLOW_TOOL_NODE_TYPE,
                ref="validated",
                config={"arguments": {"value": "value"}},
            )
        )

        with self.assertRaises(FlowNodeConfigurationError) as raw_context:
            await raw_node.execute_async({"payload": {"value": "bad"}})

        self.assertEqual(raw_context.exception.code, "flow.invalid_arguments")
        self.assertFalse(raw_resolver.executed)
        self.assertEqual(len(raw_resolver.validated), 1)
        self.assertEqual(
            raw_resolver.validated[0].arguments,
            {"value": "bad"},
        )

        envelope_resolver = ValidatingToolResolver([descriptor])
        envelope_node = tool_flow_node_registry(envelope_resolver).build(
            FlowNodeDefinition(
                name="validated",
                type=FLOW_TOOL_NODE_TYPE,
                ref="validated",
                config={
                    "arguments": {"value": "value"},
                    "output_mode": "envelope",
                },
            )
        )

        envelope = await envelope_node.execute_async(
            {"payload": {"value": "bad"}}
        )

        assert isinstance(envelope, dict)
        diagnostic = envelope["diagnostic"]
        assert isinstance(diagnostic, dict)
        self.assertEqual(envelope["status"], "diagnostic")
        self.assertEqual(
            diagnostic["code"],
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED.value,
        )
        self.assertFalse(envelope_resolver.executed)
        self.assertEqual(len(envelope_resolver.validated), 1)

    async def test_tool_node_envelope_exports_manager_diagnostic_safely(
        self,
    ) -> None:
        node = tool_flow_node_registry(_tool_manager()).build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
                config={
                    "arguments": {"a": "left", "b": "right"},
                    "output_mode": "envelope",
                },
            )
        )

        envelope = await node.execute_async(
            {"payload": {"left": "private-secret", "right": 2}}
        )

        assert isinstance(envelope, dict)
        diagnostic = envelope["diagnostic"]
        assert isinstance(diagnostic, dict)
        self.assertEqual(envelope["status"], "diagnostic")
        self.assertEqual(envelope["canonical_name"], "flow_adder")
        self.assertIsNone(envelope["result"])
        self.assertIsNone(envelope["error"])
        self.assertEqual(
            diagnostic["status"],
            ToolCallDiagnosticStatus.NON_EXECUTED.value,
        )
        self.assertEqual(
            diagnostic["code"],
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED.value,
        )
        self.assertEqual(
            diagnostic["stage"],
            ToolCallDiagnosticStage.VALIDATE.value,
        )
        self.assertEqual(diagnostic["requested_name"], "flow_adder")
        self.assertEqual(diagnostic["canonical_name"], "flow_adder")
        self.assertFalse(diagnostic["retryable"])
        self.assertNotIn("private-secret", str(envelope))
        self.assertNotIn("arguments", diagnostic)

    async def test_tool_node_output_modes_cover_outcomes(self) -> None:
        success = tool_flow_node_registry(_tool_manager()).build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
                config={"output_mode": "envelope"},
            )
        )
        diagnostic = tool_flow_node_registry(_tool_manager()).build(
            FlowNodeDefinition(
                name="typed",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
                config={
                    "arguments": {"a": "left", "b": "right"},
                    "output_mode": "envelope",
                },
            )
        )
        error = tool_flow_node_registry(
            _tool_manager(enable_tools=["flow_fails"])
        ).build(
            FlowNodeDefinition(
                name="fail",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_fails",
                config={"output_mode": "envelope"},
            )
        )

        success_envelope = await success.execute_async(
            {"payload": {"a": 2, "b": 3}}
        )
        diagnostic_envelope = await diagnostic.execute_async(
            {
                "left": "one",
                "right": 2,
            }
        )
        error_envelope = await error.execute_async({"payload": 5})

        assert isinstance(success_envelope, dict)
        self.assertEqual(success_envelope["status"], "result")
        self.assertEqual(success_envelope["canonical_name"], "flow_adder")
        self.assertEqual(success_envelope["result"], 5)
        self.assertIsNone(success_envelope["error"])
        self.assertIsNone(success_envelope["diagnostic"])
        assert isinstance(diagnostic_envelope, dict)
        self.assertEqual(diagnostic_envelope["status"], "diagnostic")
        self.assertEqual(
            diagnostic_envelope["canonical_name"],
            "flow_adder",
        )
        self.assertIsNotNone(diagnostic_envelope["diagnostic"])
        assert isinstance(error_envelope, dict)
        self.assertEqual(error_envelope["status"], "error")
        self.assertEqual(error_envelope["canonical_name"], "flow_fails")
        self.assertEqual(
            error_envelope["error"],
            {"type": "ValueError", "message": "Tool call failed."},
        )
        self.assertNotIn("bad 5", str(error_envelope))

    async def test_tool_node_raw_mode_preserves_null_result(self) -> None:
        registry = tool_flow_node_registry(
            _tool_manager(enable_tools=["flow_none"])
        )
        node = registry.build(
            FlowNodeDefinition(
                name="null",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_none",
            )
        )

        self.assertIsNone(await node.execute_async({"payload": 1}))

    async def test_tool_node_raw_mode_raises_on_execution_error(self) -> None:
        registry = tool_flow_node_registry(
            _tool_manager(enable_tools=["flow_fails"])
        )
        node = registry.build(
            FlowNodeDefinition(
                name="fail",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_fails",
            )
        )

        with self.assertRaisesRegex(RuntimeError, "ValueError") as context:
            await node.execute_async({"payload": 3})
        self.assertNotIn("bad 3", str(context.exception))

    async def test_tool_node_error_projection_omits_private_values(
        self,
    ) -> None:
        manager = ToolManager.create_instance(
            enable_tools=["flow_private_fails"],
            available_toolsets=[ToolSet(tools=[flow_private_fails])],
            settings=ToolManagerSettings(),
        )
        registry = tool_flow_node_registry(manager)
        envelope_node = registry.build(
            FlowNodeDefinition(
                name="fail",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_private_fails",
                config={"output_mode": "envelope"},
            )
        )
        raw_node = registry.build(
            FlowNodeDefinition(
                name="fail_raw",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_private_fails",
            )
        )

        envelope = await envelope_node.execute_async(
            {"payload": "private-call-argument"}
        )
        with self.assertRaises(RuntimeError) as context:
            await raw_node.execute_async({"payload": "private-call-argument"})

        assert isinstance(envelope, dict)
        self.assertEqual(envelope["status"], "error")
        self.assertEqual(
            envelope["error"],
            {"type": "ValueError", "message": "Tool call failed."},
        )
        self.assertNotIn("private-call-argument", str(envelope))
        self.assertNotIn("private-call-argument", str(context.exception))

    async def test_tool_node_raw_mode_raises_on_diagnostic(self) -> None:
        def suppress(
            _call: ToolCall,
            _context: ToolCallContext,
        ) -> ToolFilterResult:
            return ToolFilterResult(status=ToolFilterResultStatus.SUPPRESS)

        manager = ToolManager.create_instance(
            enable_tools=["flow_adder"],
            available_toolsets=[ToolSet(tools=[flow_adder])],
            settings=ToolManagerSettings(filters=[suppress]),
        )
        registry = tool_flow_node_registry(manager)
        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
            )
        )

        with self.assertRaises(FlowNodeConfigurationError) as context:
            await node.execute_async({"payload": {"a": 1, "b": 2}})

        self.assertEqual(context.exception.code, "flow.tool_diagnostic")

    async def test_tool_node_envelope_keeps_filter_guard_after_context_replace(
        self,
    ) -> None:
        def replace_context(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            return call, ToolCallContext()

        def rename(
            call: ToolCall,
            context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            return (
                ToolCall(
                    id=call.id,
                    name="flow_multiplier",
                    arguments=call.arguments,
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["flow_adder", "flow_multiplier"],
            available_toolsets=[ToolSet(tools=[flow_adder, flow_multiplier])],
            settings=ToolManagerSettings(filters=[replace_context, rename]),
        )
        registry = tool_flow_node_registry(manager)
        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
                config={"output_mode": "envelope"},
            )
        )

        envelope = await node.execute_async({"payload": {"a": 2, "b": 3}})

        assert isinstance(envelope, dict)
        diagnostic = envelope["diagnostic"]
        assert isinstance(diagnostic, dict)
        self.assertEqual(envelope["status"], "diagnostic")
        self.assertEqual(envelope["canonical_name"], "flow_adder")
        self.assertIsNone(envelope["result"])
        self.assertEqual(
            diagnostic["code"],
            ToolCallDiagnosticCode.FILTER_SUPPRESSED.value,
        )
        self.assertEqual(
            diagnostic["details"], {"filtered_name": "flow_multiplier"}
        )

    async def test_tool_node_accepts_filter_alias_for_same_tool(
        self,
    ) -> None:
        def set_alias(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            return (
                ToolCall(
                    id=call.id,
                    name="sum",
                    arguments=call.arguments,
                ),
                ToolCallContext(),
            )

        manager = ToolManager.create_instance(
            enable_tools=["flow_adder"],
            available_toolsets=[ToolSet(tools=[flow_adder])],
            settings=ToolManagerSettings(filters=[set_alias]),
        )
        registry = tool_flow_node_registry(manager)
        node = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
            )
        )

        self.assertEqual(
            await node.execute_async({"payload": {"a": 2, "b": 3}}),
            5,
        )

    async def test_tool_node_propagates_cancellation_after_context_replace(
        self,
    ) -> None:
        cases = (
            {},
            {"output_mode": "envelope"},
        )

        for config in cases:
            with self.subTest(config=config):
                called: list[str] = []

                async def cancellable_adder(a: int, b: int) -> int:
                    called.append("tool")
                    return a + b

                def replace_context(
                    call: ToolCall,
                    _context: ToolCallContext,
                ) -> tuple[ToolCall, ToolCallContext]:
                    called.append("filter")
                    return call, ToolCallContext()

                async def cancel() -> None:
                    called.append("cancel")
                    if called.count("cancel") == 4:
                        raise CancelledError()

                manager = ToolManager.create_instance(
                    enable_tools=["cancellable_adder"],
                    available_toolsets=[ToolSet(tools=[cancellable_adder])],
                    settings=ToolManagerSettings(filters=[replace_context]),
                )
                registry = tool_flow_node_registry(manager)
                node = registry.build(
                    FlowNodeDefinition(
                        name="calculate",
                        type=FLOW_TOOL_NODE_TYPE,
                        ref="cancellable_adder",
                        config=config,
                    )
                )

                with self.assertRaises(CancelledError):
                    await node.execute_async(
                        {"payload": {"a": 2, "b": 3}},
                        cancellation_checker=cancel,
                    )

                self.assertEqual(
                    called,
                    ["cancel", "cancel", "cancel", "filter", "cancel"],
                )

    async def test_tool_node_rejects_invalid_runtime_bindings(
        self,
    ) -> None:
        registry = tool_flow_node_registry(_tool_manager())
        unresolved = registry.build(
            FlowNodeDefinition(
                name="calculate",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
                config={"arguments": {"a": "left", "b": "missing"}},
            )
        )
        ambiguous = registry.build(
            FlowNodeDefinition(
                name="sum",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
            )
        )
        invalid = registry.build(
            FlowNodeDefinition(
                name="typed",
                type=FLOW_TOOL_NODE_TYPE,
                ref="flow_adder",
                config={"arguments": {"a": "left", "b": "right"}},
            )
        )

        cases = (
            (
                unresolved,
                {"left": 1},
                "flow.unresolved_argument_selector",
            ),
            (ambiguous, {"value": 1}, "flow.ambiguous_argument_binding"),
            (
                invalid,
                {"left": "one", "right": 2},
                "flow.invalid_arguments",
            ),
        )
        for node, inputs, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(FlowNodeConfigurationError) as context:
                    await node.execute_async(inputs)

                self.assertEqual(context.exception.code, code)

    def test_tool_node_factory_rejects_invalid_argument_bindings(self) -> None:
        registry = tool_flow_node_registry(_tool_manager())
        cases = (
            (
                {"arguments": "a"},
                "flow.invalid_arguments",
                "nodes.calculate.config.arguments",
            ),
            (
                {"arguments": {"c": "left", "a": "left", "b": "right"}},
                "flow.unknown_argument_binding",
                "nodes.calculate.config.arguments.c",
            ),
            (
                {"arguments": {"a": "left", "b": ""}},
                "flow.invalid_argument_selector",
                "nodes.calculate.config.arguments.b",
            ),
            (
                {"arguments": {"a": "left"}},
                "flow.missing_argument_binding",
                "nodes.calculate.config.arguments.b",
            ),
            (
                {"output_mode": "wrapped"},
                "flow.invalid_output_mode",
                "nodes.calculate.config.output_mode",
            ),
        )

        for config, code, path in cases:
            with self.subTest(code=code):
                definition = FlowNodeDefinition(
                    name="calculate",
                    type=FLOW_TOOL_NODE_TYPE,
                    ref="flow_adder",
                    config=config,
                )
                with self.assertRaises(FlowNodeConfigurationError) as context:
                    registry.build(definition)

                self.assertEqual(context.exception.code, code)
                self.assertEqual(context.exception.path, path)

        no_arg_registry = tool_flow_node_registry(
            _tool_manager(enable_tools=["flow_status"])
        )
        with self.assertRaises(FlowNodeConfigurationError) as context:
            no_arg_registry.build(
                FlowNodeDefinition(
                    name="status",
                    type=FLOW_TOOL_NODE_TYPE,
                    ref="flow_status",
                    config={"arguments": {"value": "payload"}},
                )
            )

        self.assertEqual(
            context.exception.code,
            "flow.unknown_argument_binding",
        )
        self.assertEqual(
            context.exception.path,
            "nodes.status.config.arguments.value",
        )

    def test_tool_registry_can_extend_custom_base_registry(self) -> None:
        def factory(definition: FlowNodeDefinition) -> Node:
            return Node(definition.name, func=lambda _: "custom")

        base_registry = FlowNodeRegistry({"custom": factory})

        registry = tool_flow_node_registry(
            _tool_manager(),
            base_registry=base_registry,
        )

        self.assertIs(registry, base_registry)
        self.assertTrue(registry.supports("custom"))
        self.assertTrue(registry.supports(FLOW_TOOL_NODE_TYPE))

    def test_tool_registry_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(AssertionError):
            tool_flow_node_registry(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            tool_flow_node_registry(
                _tool_manager(),
                base_registry=object(),  # type: ignore[arg-type]
            )

    def test_tool_node_factory_rejects_bad_refs(self) -> None:
        manager = _tool_manager(enable_tools=["flow_adder", "flow_adder_alt"])
        registry = tool_flow_node_registry(manager)
        cases = (
            (None, "flow.missing_ref"),
            ("tools/adder.py", "flow.invalid_ref"),
            ("mcp://server/tool", "flow.invalid_ref"),
            ("missing", "flow.tool_unknown"),
            ("disabled.flow_disabled", "flow.tool_disabled"),
            ("sum", "flow.tool_ambiguous"),
        )

        for ref, code in cases:
            with self.subTest(ref=ref):
                definition = FlowNodeDefinition(
                    name="calculate",
                    type=FLOW_TOOL_NODE_TYPE,
                    ref=ref,
                )
                with self.assertRaises(FlowNodeConfigurationError) as context:
                    registry.build(definition)

                self.assertEqual(context.exception.code, code)
                self.assertEqual(context.exception.path, "nodes.calculate.ref")


if __name__ == "__main__":
    main()
