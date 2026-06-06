from collections.abc import Mapping
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallOutcome,
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
    FlowInputDefinition,
    FlowInputType,
    FlowNodeContract,
    FlowNodeDefinition,
    FlowNodeMetadata,
    FlowNodeRegistry,
    FlowOutputType,
    default_flow_node_registry,
    flow_input_binding,
    tool_flow_node_registry,
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


async def flow_disabled(a: int) -> int:
    return a


async def flow_identity(value: int) -> int:
    return value


async def flow_none(value: int) -> None:
    return None


async def flow_fails(value: int) -> None:
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
                    flow_identity,
                    flow_none,
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
        mapping = flow_input_binding(
            FlowInputDefinition(name="payload", type=FlowInputType.OBJECT),
            {"name": "Ada"},
        )

        self.assertEqual(scalar[FLOW_INPUT_KEY], "ready")
        self.assertEqual(scalar["text"], "ready")
        self.assertEqual(scalar["value"], "ready")
        self.assertEqual(no_definition["value"], "ready")
        self.assertEqual(mapping["payload"], {"name": "Ada"})
        self.assertNotIn("value", mapping)

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

    def test_registry_exposes_node_metadata(self) -> None:
        registry = default_flow_node_registry()

        self.assertFalse(registry.supports_ref("echo"))
        self.assertFalse(registry.is_async_only("echo"))
        self.assertIsNone(registry.metadata("missing"))
        self.assertIsNone(registry.input_contract("missing"))
        self.assertIsNone(registry.output_contract("missing"))
        echo_input = registry.input_contract("echo")
        echo_output = registry.output_contract("echo")

        assert echo_input is not None
        assert echo_output is not None
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
        )
        registry = FlowNodeRegistry({"remote": factory}, {"remote": metadata})

        self.assertTrue(registry.supports("remote"))
        self.assertTrue(registry.supports_ref("remote"))
        self.assertTrue(registry.is_async_only("remote"))
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
            {"type": "ValueError", "message": "bad 5"},
        )

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

        with self.assertRaisesRegex(RuntimeError, "ValueError: bad 3"):
            await node.execute_async({"payload": 3})

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
