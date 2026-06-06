from asyncio import CancelledError, sleep
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, patch
from uuid import uuid4 as _uuid4

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolFilter,
    ToolFormat,
    ToolManagerExecutionMode,
    ToolManagerMissingCallMode,
    ToolManagerSettings,
    ToolNameResolutionStatus,
    ToolParserReturnMode,
    ToolProviderArgumentsMode,
    ToolTransformer,
)
from avalan.model.vendor import TextGenerationVendor
from avalan.tool import Tool, ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.math import CalculatorTool
from avalan.tool.parser import ToolCallParser


class ToolManagerCreationTestCase(TestCase):
    def test_settings_default_to_legacy_compatibility(self):
        settings = ToolManagerSettings()

        self.assertIs(settings.execution_mode, ToolManagerExecutionMode.LEGACY)
        self.assertIs(
            settings.missing_call_mode,
            ToolManagerMissingCallMode.LEGACY_NONE,
        )
        self.assertIs(
            settings.parser_return_mode,
            ToolParserReturnMode.LEGACY,
        )
        self.assertIs(
            settings.provider_arguments_mode,
            ToolProviderArgumentsMode.EMPTY_ON_MALFORMED,
        )

    def test_settings_accept_outcome_compatibility_modes(self):
        settings = ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES,
            missing_call_mode=ToolManagerMissingCallMode.DIAGNOSTIC,
            parser_return_mode=ToolParserReturnMode.OUTCOME,
            provider_arguments_mode=(
                ToolProviderArgumentsMode.DIAGNOSTIC_ON_MALFORMED
            ),
        )

        self.assertIs(
            settings.execution_mode, ToolManagerExecutionMode.OUTCOMES
        )
        self.assertIs(
            settings.missing_call_mode,
            ToolManagerMissingCallMode.DIAGNOSTIC,
        )
        self.assertIs(
            settings.parser_return_mode,
            ToolParserReturnMode.OUTCOME,
        )
        self.assertIs(
            settings.provider_arguments_mode,
            ToolProviderArgumentsMode.DIAGNOSTIC_ON_MALFORMED,
        )

    def test_settings_reject_invalid_compatibility_modes(self):
        invalid_cases = (
            {"execution_mode": "legacy"},
            {"missing_call_mode": "legacy_none"},
            {"parser_return_mode": "legacy"},
            {"provider_arguments_mode": "empty_on_malformed"},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ToolManagerSettings(**kwargs)

    def test_default_instance_empty(self):
        manager = ToolManager.create_instance(
            enable_tools=[], settings=ToolManagerSettings()
        )
        self.assertTrue(manager.is_empty)
        self.assertIsNone(manager.tools)

    def test_instance_with_enabled_tool(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=[calculator.__name__],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertFalse(manager.is_empty)
        self.assertEqual(manager.tools, [calculator])

    def test_no_enabled_tools(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=[],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertTrue(manager.is_empty)
        self.assertIsNone(manager.tools)

    def test_tool_format_property(self):
        settings = ToolManagerSettings(tool_format=ToolFormat.HARMONY)
        manager = ToolManager.create_instance(settings=settings)
        self.assertEqual(manager.tool_format, ToolFormat.HARMONY)

    def test_toolset_without_tools(self):
        manager = ToolManager.create_instance(
            enable_tools=None,
            available_toolsets=[ToolSet(tools=[])],
            settings=ToolManagerSettings(),
        )
        self.assertTrue(manager.is_empty)
        self.assertIsNone(manager.tools)

    def test_enable_tools_partial_namespace(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=["math"],
            available_toolsets=[ToolSet(namespace="math", tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertEqual(manager.tools, [calculator])

    def test_enable_tools_full_namespace(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=["math.calculator"],
            available_toolsets=[ToolSet(namespace="math", tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertEqual(manager.tools, [calculator])

    def test_enable_tools_no_namespace_match(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=["science"],
            available_toolsets=[ToolSet(namespace="math", tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        self.assertTrue(manager.is_empty)

    def test_nested_toolset_names_are_enabled_with_parent_namespace(self):
        inner = ToolSet(namespace="inner", tools=[CalculatorTool()])
        outer = ToolSet(namespace="outer", tools=[CalculatorTool(), inner])
        manager = ToolManager.create_instance(
            enable_tools=["outer"],
            available_toolsets=[outer],
            settings=ToolManagerSettings(),
        )

        self.assertEqual(
            [descriptor.name for descriptor in manager.list_tools()],
            ["outer.calculator", "outer.inner.calculator"],
        )
        resolution = manager.resolve_tool_name("outer.inner.calculator")
        self.assertIs(resolution.status, ToolNameResolutionStatus.EXACT)

    def test_nested_toolset_can_enable_exact_nested_tool(self):
        inner = ToolSet(namespace="inner", tools=[CalculatorTool()])
        outer = ToolSet(namespace="outer", tools=[CalculatorTool(), inner])
        manager = ToolManager.create_instance(
            enable_tools=["outer.inner.calculator"],
            available_toolsets=[outer],
            settings=ToolManagerSettings(),
        )

        self.assertEqual(
            [descriptor.name for descriptor in manager.list_tools()],
            ["outer.inner.calculator"],
        )
        self.assertIsNone(manager.describe_tool("outer.calculator"))
        resolution = manager.resolve_tool_name("outer.calculator")
        self.assertIs(resolution.status, ToolNameResolutionStatus.DISABLED)

    def test_list_and_describe_enabled_tools(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(),
        )

        descriptors = manager.list_tools()

        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0].name, "adder")
        self.assertIs(descriptors[0].callable, adder)
        self.assertEqual(descriptors[0].aliases, ["sum"])
        assert descriptors[0].schema is not None
        self.assertEqual(
            descriptors[0].schema["function"]["name"],
            "adder",
        )
        self.assertEqual(
            descriptors[0].parameter_schema,
            descriptors[0].schema["function"]["parameters"],
        )
        self.assertEqual(
            descriptors[0].return_schema,
            descriptors[0].schema["function"]["return"],
        )
        self.assertEqual(
            descriptors[0].provider_safe_schema,
            descriptors[0].schema,
        )
        self.assertIsNone(descriptors[0].namespace)
        self.assertEqual(descriptors[0].policy, {})
        self.assertEqual(descriptors[0].metadata, {})
        self.assertEqual(manager.describe_tool("adder"), descriptors[0])
        self.assertEqual(manager.describe_tool("sum"), descriptors[0])
        self.assertIsNone(manager.describe_tool("missing"))

    def test_list_tools_prefixes_function_schema(self):
        def multiply(a: int, b: int) -> int:
            """Multiply numbers.

            Args:
                a: Left number.
                b: Right number.
            """
            return a * b

        manager = ToolManager.create_instance(
            enable_tools=["math.multiply"],
            available_toolsets=[ToolSet(namespace="math", tools=[multiply])],
            settings=ToolManagerSettings(),
        )

        descriptor = manager.list_tools()[0]

        assert descriptor.schema is not None
        self.assertEqual(
            descriptor.schema["function"]["name"],
            "math.multiply",
        )
        self.assertEqual(descriptor.namespace, "math")
        assert descriptor.provider_safe_schema is not None
        self.assertEqual(
            descriptor.provider_safe_schema["function"]["name"],
            "avl_bWF0aC5tdWx0aXBseQ",
        )
        self.assertEqual(
            descriptor.provider_safe_schema["function"]["parameters"],
            descriptor.schema["function"]["parameters"],
        )

    def test_tool_descriptor_accepts_missing_schema(self):
        adder = DummyAdder()

        descriptor = ToolManager._tool_descriptor(
            canonical_name="adder",
            tool=adder,
            aliases=[],
            namespace=None,
            schema=None,
        )

        self.assertEqual(descriptor.name, "adder")
        self.assertIs(descriptor.callable, adder)
        self.assertIsNone(descriptor.schema)
        self.assertIsNone(descriptor.parameter_schema)
        self.assertIsNone(descriptor.return_schema)
        self.assertIsNone(descriptor.provider_safe_schema)

    def test_resolve_tool_name_exact_alias_ambiguous_disabled_unknown(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder", "adder_alt"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyAdderAlt()]),
                ToolSet(namespace="disabled", tools=[CalculatorTool()]),
            ],
            settings=ToolManagerSettings(),
        )

        exact = manager.resolve_tool_name("adder")
        self.assertIs(exact.status, ToolNameResolutionStatus.EXACT)
        self.assertEqual(exact.canonical_name, "adder")
        self.assertEqual(exact.candidates, ["adder"])

        ambiguous = manager.resolve_tool_name("sum")
        self.assertIs(ambiguous.status, ToolNameResolutionStatus.AMBIGUOUS)
        self.assertEqual(ambiguous.candidates, ["adder", "adder_alt"])
        self.assertIs(
            ambiguous.diagnostic_code,
            ToolCallDiagnosticCode.AMBIGUOUS_TOOL_NAME,
        )

        disabled = manager.resolve_tool_name("disabled.calculator")
        self.assertIs(disabled.status, ToolNameResolutionStatus.DISABLED)
        self.assertIs(
            disabled.diagnostic_code, ToolCallDiagnosticCode.DISABLED_TOOL
        )

        unknown = manager.resolve_tool_name("missing")
        self.assertIs(unknown.status, ToolNameResolutionStatus.UNKNOWN)
        self.assertIs(
            unknown.diagnostic_code, ToolCallDiagnosticCode.UNKNOWN_TOOL
        )

    def test_resolve_tool_name_alias(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name("sum")

        self.assertIs(resolution.status, ToolNameResolutionStatus.ALIAS)
        self.assertEqual(resolution.canonical_name, "adder")
        self.assertEqual(resolution.candidates, ["adder"])

    def test_resolve_tool_name_exact_canonical_wins_before_alias(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder", "sum"],
            available_toolsets=[ToolSet(tools=[DummyAdder(), DummySum()])],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name("sum")

        self.assertIs(resolution.status, ToolNameResolutionStatus.EXACT)
        self.assertEqual(resolution.canonical_name, "sum")
        self.assertEqual(resolution.candidates, ["sum"])

    def test_resolve_tool_name_strips_one_functions_prefix(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )

        exact = manager.resolve_tool_name("functions.adder")
        alias = manager.resolve_tool_name("functions.sum")
        unknown = manager.resolve_tool_name("functions.functions.adder")

        self.assertIs(exact.status, ToolNameResolutionStatus.EXACT)
        self.assertEqual(exact.canonical_name, "adder")
        self.assertIs(alias.status, ToolNameResolutionStatus.ALIAS)
        self.assertEqual(alias.canonical_name, "adder")
        self.assertIs(unknown.status, ToolNameResolutionStatus.UNKNOWN)

    def test_resolve_tool_name_decodes_only_provider_originated_names(self):
        manager = ToolManager.create_instance(
            enable_tools=["math.adder"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[DummyAdder()])
            ],
            settings=ToolManagerSettings(),
        )
        encoded = TextGenerationVendor.encode_tool_name("math.adder")

        without_provenance = manager.resolve_tool_name(encoded)
        with_provenance = manager.resolve_tool_name(
            encoded,
            provider_originated=True,
        )

        self.assertIs(
            without_provenance.status, ToolNameResolutionStatus.UNKNOWN
        )
        self.assertIs(with_provenance.status, ToolNameResolutionStatus.EXACT)
        self.assertEqual(with_provenance.canonical_name, "math.adder")
        self.assertEqual(with_provenance.requested_name, encoded)

    def test_resolve_tool_name_provider_origin_plain_name(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name(
            "adder",
            provider_originated=True,
        )

        self.assertIs(resolution.status, ToolNameResolutionStatus.EXACT)
        self.assertEqual(resolution.canonical_name, "adder")

    def test_resolve_tool_name_rejects_invalid_provider_origin_name(self):
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(),
        )

        invalid_names = ("pkg.tool", "avl_notbase64")
        for name in invalid_names:
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    manager.resolve_tool_name(
                        name,
                        provider_originated=True,
                    )

    def test_resolve_tool_name_alias_recovery_uses_enabled_tools_only(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder()]),
                ToolSet(namespace="disabled", tools=[DummyAdderAlt()]),
            ],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name("sum")

        self.assertIs(resolution.status, ToolNameResolutionStatus.ALIAS)
        self.assertEqual(resolution.canonical_name, "adder")
        self.assertEqual(resolution.candidates, ["adder"])

    def test_resolve_tool_name_disabled_alias_is_not_recovered(self):
        manager = ToolManager.create_instance(
            enable_tools=[],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name("sum")

        self.assertIs(resolution.status, ToolNameResolutionStatus.DISABLED)
        self.assertIsNone(resolution.canonical_name)
        self.assertEqual(resolution.candidates, ["adder"])
        self.assertIs(
            resolution.diagnostic_code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )

    def test_resolve_tool_name_rejects_empty_name(self):
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(),
        )

        with self.assertRaises(AssertionError):
            manager.resolve_tool_name(" ")

    def test_invalid_tool_aliases_are_rejected(self):
        with self.assertRaises(AssertionError):
            ToolManager.create_instance(
                enable_tools=["invalid_aliases"],
                available_toolsets=[ToolSet(tools=[InvalidAliasesTool()])],
                settings=ToolManagerSettings(),
            )

    def test_validate_tool_call_accepts_resolved_alias(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call-1", name="sum", arguments={"a": 1, "b": 2})

        self.assertIsNone(manager.validate_tool_call(call))

    def test_validate_tool_call_returns_resolution_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(),
        )
        diagnostic_id = _uuid4()
        call = ToolCall(id="call-1", name="missing", arguments={})

        with patch("avalan.tool.manager.uuid4", return_value=diagnostic_id):
            diagnostic = manager.validate_tool_call(call)

        assert diagnostic is not None
        self.assertEqual(diagnostic.id, diagnostic_id)
        self.assertEqual(diagnostic.call_id, "call-1")
        self.assertEqual(diagnostic.requested_name, "missing")
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.UNKNOWN_TOOL)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)

    def test_validate_tool_call_returns_malformed_arguments_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        diagnostic_id = _uuid4()
        call = ToolCall(id="call-1", name="adder", arguments=cast(Any, ["a"]))

        with patch("avalan.tool.manager.uuid4", return_value=diagnostic_id):
            diagnostic = manager.validate_tool_call(call)

        assert diagnostic is not None
        self.assertEqual(diagnostic.id, diagnostic_id)
        self.assertIs(
            diagnostic.code, ToolCallDiagnosticCode.MALFORMED_ARGUMENTS
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.VALIDATE)

    def test_validate_tool_call_returns_argument_validation_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        diagnostic_id = _uuid4()
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1})

        with patch("avalan.tool.manager.uuid4", return_value=diagnostic_id):
            diagnostic = manager.validate_tool_call(call)

        assert diagnostic is not None
        self.assertEqual(diagnostic.id, diagnostic_id)
        self.assertEqual(diagnostic.canonical_name, "adder")
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.VALIDATE)


class DummyAdder:
    def __init__(self) -> None:
        self.__name__ = "adder"
        self.aliases = ["sum"]

    async def __call__(self, a: int, b: int) -> int:
        """Return the sum of ``a`` and ``b``."""
        return a + b


class DummyAdderAlt(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "adder_alt"
        self.aliases = ["sum"]


class DummySum(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "sum"
        self.aliases = []


class InvalidAliasesTool(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "invalid_aliases"
        self.aliases = "sum"


class NativeAdderTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "native_adder"

    async def __call__(self, a: int, b: int, context: ToolCallContext) -> int:
        return a + b


class NativeNoArgTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "native_noarg"

    async def __call__(self, context: ToolCallContext) -> str:
        return "hi"


class ToolManagerToolTypesTestCase(IsolatedAsyncioTestCase):
    async def test_native_tool_with_arguments(self):
        tool = NativeAdderTool()
        manager = ToolManager.create_instance(
            enable_tools=[tool.__name__],
            available_toolsets=[ToolSet(tools=[tool])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(), name=tool.__name__, arguments={"a": 1, "b": 2}
        )
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, 3)

    async def test_native_tool_without_arguments(self):
        tool = NativeNoArgTool()
        manager = ToolManager.create_instance(
            enable_tools=[tool.__name__],
            available_toolsets=[ToolSet(tools=[tool])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id=_uuid4(), name=tool.__name__, arguments={})
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, "hi")

    async def test_non_native_tool_with_arguments(self):
        async def adder(a: int, b: int) -> int:
            return a + b

        manager = ToolManager.create_instance(
            enable_tools=[adder.__name__],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(), name=adder.__name__, arguments={"a": 1, "b": 2}
        )
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, 3)

    async def test_non_native_tool_without_arguments(self):
        def greet() -> str:
            return "hi"

        manager = ToolManager.create_instance(
            enable_tools=[greet.__name__],
            available_toolsets=[ToolSet(tools=[greet])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id=_uuid4(), name=greet.__name__, arguments={})
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, "hi")


class ToolManagerCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        calculator = CalculatorTool()
        self.manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )

    async def test_callable_class_tool(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1, "b": 2})
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, 3)

    async def test_call_no_tool_call(self):
        calls = self.manager.get_calls("no tools here")
        self.assertIsNone(calls)

    async def test_get_calls_normalizes_configured_tuple_formats(self):
        cases = (
            (
                ToolFormat.JSON,
                '{"tool": "calculator", "arguments": {"expression": "1"}}',
                {"expression": "1"},
            ),
            (
                ToolFormat.REACT,
                'Action: calculator\nAction Input: {"expression": "2"}',
                {"expression": "2"},
            ),
            (
                ToolFormat.BRACKET,
                "[calculator](3)",
                {"input": "3"},
            ),
            (
                ToolFormat.OPENAI,
                '{"name": "calculator", "arguments": {"expression": "4"}}',
                {"expression": "4"},
            ),
        )

        for tool_format, text, arguments in cases:
            with self.subTest(tool_format=tool_format):
                manager = ToolManager.create_instance(
                    enable_tools=["calculator"],
                    available_toolsets=[ToolSet(tools=[CalculatorTool()])],
                    settings=ToolManagerSettings(tool_format=tool_format),
                )
                call_id = _uuid4()

                with patch("avalan.tool.parser.uuid4", return_value=call_id):
                    self.assertEqual(
                        manager.get_calls(text),
                        [
                            ToolCall(
                                id=call_id,
                                name="calculator",
                                arguments=arguments,
                            )
                        ],
                    )

    async def test_get_calls_returns_none_for_parse_diagnostics_only(self):
        cases = (
            (
                ToolFormat.JSON,
                '{"tool": "calculator", "arguments": ["1"]}',
            ),
            (
                ToolFormat.REACT,
                'Action: calculator\nAction Input: ["2"]',
            ),
            (
                ToolFormat.OPENAI,
                '{"name": "calculator", "arguments": ["3"]}',
            ),
            (
                ToolFormat.JSON,
                '{"tool": "calculator", "arguments": ',
            ),
        )

        for tool_format, text in cases:
            with self.subTest(tool_format=tool_format):
                manager = ToolManager.create_instance(
                    enable_tools=["calculator"],
                    available_toolsets=[ToolSet(tools=[CalculatorTool()])],
                    settings=ToolManagerSettings(tool_format=tool_format),
                )

                self.assertIsNone(manager.get_calls(text))

    async def test_call_with_tool(self):
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "1 + 1"}}</tool_call>'
        )
        call_id = _uuid4()
        result_id = _uuid4()
        with (
            patch("avalan.tool.parser.uuid4", return_value=call_id),
            patch("avalan.tool.manager.uuid4", return_value=result_id),
        ):
            calls = self.manager.get_calls(text)
            expected_call = ToolCall(
                id=call_id,
                name="calculator",
                arguments={"expression": "1 + 1"},
            )
            self.assertEqual(calls, [expected_call])

            results = await self.manager(calls[0], context=ToolCallContext())

            expected_result = ToolCallResult(
                id=result_id,
                call=expected_call,
                name="calculator",
                arguments={"expression": "1 + 1"},
                result="2",
            )
            self.assertEqual(results, expected_result)

    async def test_tool_exception(self):
        async def failing_tool(a: int) -> None:
            raise ValueError("boom")

        manager = ToolManager.create_instance(
            enable_tools=["failing_tool"],
            available_toolsets=[ToolSet(tools=[failing_tool])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id=_uuid4(), name="failing_tool", arguments={"a": 1})
        result = await manager(call, context=ToolCallContext())
        self.assertIsInstance(result, ToolCallError)
        self.assertIsInstance(result.error, ValueError)
        self.assertEqual(result.message, "boom")

    async def test_tool_keyboard_interrupt_propagates(self):
        async def interrupted_tool(a: int) -> None:
            raise KeyboardInterrupt()

        manager = ToolManager.create_instance(
            enable_tools=["interrupted_tool"],
            available_toolsets=[ToolSet(tools=[interrupted_tool])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(), name="interrupted_tool", arguments={"a": 1}
        )

        with self.assertRaises(KeyboardInterrupt):
            await manager(call, context=ToolCallContext())

    async def test_tool_cancelled_error_propagates(self):
        async def interrupted_tool(a: int) -> None:
            raise CancelledError()

        manager = ToolManager.create_instance(
            enable_tools=["interrupted_tool"],
            available_toolsets=[ToolSet(tools=[interrupted_tool])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(), name="interrupted_tool", arguments={"a": 1}
        )

        with self.assertRaises(CancelledError):
            await manager(call, context=ToolCallContext())

    async def test_context_cancellation_checker_runs_before_filters(self):
        called: list[str] = []

        async def adder(a: int) -> int:
            called.append("tool")
            return a + 1

        def filter_call(call: ToolCall, context: ToolCallContext):
            called.append("filter")
            return call, context

        async def cancel() -> None:
            called.append("cancel")
            raise CancelledError()

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(filters=[filter_call]),
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1})

        with self.assertRaises(CancelledError):
            await manager(
                call,
                context=ToolCallContext(cancellation_checker=cancel),
            )

        self.assertEqual(called, ["cancel"])

    async def test_context_cancellation_checker_runs_after_filters(self):
        called: list[str] = []

        async def adder(a: int) -> int:
            called.append("tool")
            return a + 1

        async def first_check() -> None:
            called.append("first")

        async def second_check() -> None:
            called.append("second")
            raise CancelledError()

        def filter_call(call: ToolCall, context: ToolCallContext):
            called.append("filter")
            return (
                call,
                ToolCallContext(
                    input=context.input,
                    agent_id=context.agent_id,
                    participant_id=context.participant_id,
                    session_id=context.session_id,
                    calls=context.calls,
                    cancellation_checker=second_check,
                ),
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(filters=[filter_call]),
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1})

        with self.assertRaises(CancelledError):
            await manager(
                call,
                context=ToolCallContext(cancellation_checker=first_check),
            )

        self.assertEqual(called, ["first", "filter", "second"])


class ToolManagerPotentialCallTestCase(TestCase):
    def setUp(self):
        self.manager = ToolManager.create_instance(
            enable_tools=[], settings=ToolManagerSettings()
        )

    def test_is_potential_tool_call_true(self):
        with patch.object(
            self.manager._parser,
            "is_potential_tool_call",
            return_value=True,
        ) as called:
            result = self.manager.is_potential_tool_call("buf", "tok")
            self.assertTrue(result)
            called.assert_called_once_with("buf", "tok")

    def test_is_potential_tool_call_false(self):
        with patch.object(
            self.manager._parser,
            "is_potential_tool_call",
            return_value=False,
        ) as called:
            result = self.manager.is_potential_tool_call("", "")
            self.assertFalse(result)
            called.assert_called_once_with("", "")

    def test_tool_call_status_proxies_parser(self):
        self.assertIs(
            self.manager.tool_call_status("<tool_call>"),
            ToolCallParser.ToolCallBufferStatus.OPEN,
        )


class ToolManagerSchemasTestCase(TestCase):
    def test_json_schemas(self):
        calculator = CalculatorTool()
        manager = ToolManager.create_instance(
            enable_tools=["math.calculator"],
            available_toolsets=[ToolSet(namespace="math", tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        schemas = manager.json_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "math.calculator")


class ToolManagerExtraCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        calculator = CalculatorTool()
        self.manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )

    async def test_avoid_repetition_different_arguments(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(avoid_repetition=True),
        )
        history_call = ToolCall(
            id=_uuid4(),
            name="adder",
            arguments={"a": 2, "b": 2},
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1, "b": 2})
        result = await manager(
            call, context=ToolCallContext(calls=[history_call])
        )
        self.assertIsNotNone(result)

    async def test_avoid_repetition_different_name(self):
        adder = DummyAdder()
        alt = DummyAdderAlt()
        manager = ToolManager.create_instance(
            enable_tools=["adder", "adder_alt"],
            available_toolsets=[ToolSet(tools=[adder, alt])],
            settings=ToolManagerSettings(avoid_repetition=True),
        )
        history_call = ToolCall(
            id=_uuid4(),
            name="adder",
            arguments={"a": 1, "b": 2},
        )
        call = ToolCall(
            id=_uuid4(),
            name="adder_alt",
            arguments={"a": 1, "b": 2},
        )
        result = await manager(
            call, context=ToolCallContext(calls=[history_call])
        )
        self.assertIsNotNone(result)

    async def test_avoid_repetition_different_name_and_args(self):
        adder = DummyAdder()
        alt = DummyAdderAlt()
        manager = ToolManager.create_instance(
            enable_tools=["adder", "adder_alt"],
            available_toolsets=[ToolSet(tools=[adder, alt])],
            settings=ToolManagerSettings(avoid_repetition=True),
        )
        history_call = ToolCall(
            id=_uuid4(),
            name="adder",
            arguments={"a": 2, "b": 3},
        )
        call = ToolCall(
            id=_uuid4(),
            name="adder_alt",
            arguments={"a": 1, "b": 2},
        )
        result = await manager(
            call, context=ToolCallContext(calls=[history_call])
        )
        self.assertIsNotNone(result)

    async def test_call_name_not_found(self):
        call = ToolCall(id=_uuid4(), name="missing", arguments={})
        result = await self.manager(call, context=ToolCallContext())
        self.assertIsNone(result)

    async def test_call_alias_still_returns_none_without_canonical_name(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id=_uuid4(), name="sum", arguments={"a": 1, "b": 2})

        result = await manager(call, context=ToolCallContext())

        self.assertIsNone(result)

    async def test_aenter_no_toolsets(self):
        manager = ToolManager.create_instance(
            enable_tools=None,
            available_toolsets=None,
            settings=ToolManagerSettings(),
        )
        manager._stack.enter_async_context = AsyncMock()
        manager._stack.__aexit__ = AsyncMock(return_value=False)

        async with manager:
            manager._stack.enter_async_context.assert_not_called()

        manager._stack.__aexit__.assert_awaited_once()

    async def test_avoid_repetition(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(avoid_repetition=True),
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1, "b": 2})
        result1 = await manager(call, context=ToolCallContext())
        self.assertIsNotNone(result1)
        result2 = await manager(call, context=ToolCallContext(calls=[call]))
        self.assertIsNone(result2)

    async def test_maximum_depth(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(maximum_depth=1),
        )
        call = ToolCall(id=_uuid4(), name="adder", arguments={"a": 1, "b": 2})
        result1 = await manager(call, context=ToolCallContext())
        self.assertIsNotNone(result1)
        result2 = await manager(call, context=ToolCallContext(calls=[call]))
        self.assertIsNone(result2)

    async def test_async_context(self):
        calculator = CalculatorTool()
        toolset = ToolSet(tools=[calculator])
        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[toolset],
            settings=ToolManagerSettings(),
        )
        manager._stack.enter_async_context = AsyncMock()
        manager._stack.__aexit__ = AsyncMock(return_value=False)

        async with manager:
            manager._stack.enter_async_context.assert_awaited_once()

        manager._stack.__aexit__.assert_awaited_once()

    async def test_async_context_interrupt_close_timeout_suppressed(self):
        manager = ToolManager.create_instance(
            enable_tools=[], settings=ToolManagerSettings()
        )

        async def slow_close(*args):
            await sleep(10)

        manager._stack.__aexit__ = AsyncMock(side_effect=slow_close)

        with patch.object(ToolManager, "_INTERRUPT_CLOSE_TIMEOUT", 0.01):
            result = await manager.__aexit__(
                KeyboardInterrupt, KeyboardInterrupt(), None
            )

        self.assertFalse(result)
        manager._stack.__aexit__.assert_awaited_once()

    async def test_async_context_interrupt_close_error_suppressed(self):
        manager = ToolManager.create_instance(
            enable_tools=[], settings=ToolManagerSettings()
        )
        manager._stack.__aexit__ = AsyncMock(
            side_effect=RuntimeError("close failed")
        )

        result = await manager.__aexit__(
            KeyboardInterrupt, KeyboardInterrupt(), None
        )

        self.assertFalse(result)
        manager._stack.__aexit__.assert_awaited_once()

    async def test_async_context_interrupt_close_success_returns_false(self):
        manager = ToolManager.create_instance(
            enable_tools=[], settings=ToolManagerSettings()
        )
        manager._stack.__aexit__ = AsyncMock(return_value=True)

        result = await manager.__aexit__(
            KeyboardInterrupt, KeyboardInterrupt(), None
        )

        self.assertFalse(result)
        manager._stack.__aexit__.assert_awaited_once()

    async def test_set_eos_token(self):
        self.manager.set_eos_token("<END>")
        text = (
            '<tool_call>{"name": "calculator", '
            '"arguments": {"expression": "2"}}</tool_call><END>'
        )

        call_id = _uuid4()
        result_id = _uuid4()
        with (
            patch("avalan.tool.parser.uuid4", return_value=call_id),
            patch("avalan.tool.manager.uuid4", return_value=result_id),
        ):
            calls = self.manager.get_calls(text)
            expected_call = ToolCall(
                id=call_id,
                name="calculator",
                arguments={"expression": "2"},
            )
            self.assertEqual(calls, [expected_call])

            results = await self.manager(calls[0], context=ToolCallContext())

            expected_result = ToolCallResult(
                id=result_id,
                call=expected_call,
                name="calculator",
                arguments={"expression": "2"},
                result="2",
            )
            self.assertEqual(results, expected_result)
            self.assertEqual(self.manager._parser._eos_token, "<END>")


class ToolManagerFiltersTransformersTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        calculator = CalculatorTool()
        self.manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[calculator])],
            settings=ToolManagerSettings(),
        )

    async def test_filters(self):
        def modify(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"expression": "2 + 2"},
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[CalculatorTool()])],
            settings=ToolManagerSettings(filters=[modify]),
        )

        call = ToolCall(
            id=_uuid4(), name="calculator", arguments={"expression": "1 + 1"}
        )
        result_id = _uuid4()
        with patch("avalan.tool.manager.uuid4", return_value=result_id):
            result = await manager(call, context=ToolCallContext())

        self.assertEqual(result.call.arguments, {"expression": "2 + 2"})
        self.assertEqual(result.result, "4")

    async def test_transformers(self):
        def transform(_: ToolCall, __: ToolCallContext, result: str | None):
            return f"{result}!"

        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[CalculatorTool()])],
            settings=ToolManagerSettings(transformers=[transform]),
        )

        call = ToolCall(
            id=_uuid4(), name="calculator", arguments={"expression": "1 + 1"}
        )
        result_id = _uuid4()
        with patch("avalan.tool.manager.uuid4", return_value=result_id):
            result = await manager(call, context=ToolCallContext())

        self.assertEqual(result.result, "2!")

    async def test_filters_and_transformers(self):
        def modify(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"expression": "3 + 3"},
                ),
                context,
            )

        def transform(_: ToolCall, __: ToolCallContext, result: str | None):
            return int(result) * 2

        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[CalculatorTool()])],
            settings=ToolManagerSettings(
                filters=[modify], transformers=[transform]
            ),
        )

        call = ToolCall(
            id=_uuid4(), name="calculator", arguments={"expression": "1 + 1"}
        )
        result_id = _uuid4()
        with patch("avalan.tool.manager.uuid4", return_value=result_id):
            result = await manager(call, context=ToolCallContext())

        self.assertEqual(result.result, 12)

    async def test_namespaced_tool(self):
        calculator = CalculatorTool()
        namespaced_manager = ToolManager.create_instance(
            enable_tools=["math.calculator"],
            available_toolsets=[ToolSet(namespace="math", tools=[calculator])],
            settings=ToolManagerSettings(),
        )
        text = (
            '<tool_call>{"name": "math.calculator", '
            '"arguments": {"expression": "3"}}</tool_call>'
        )

        call_id = _uuid4()
        result_id = _uuid4()
        with (
            patch("avalan.tool.parser.uuid4", return_value=call_id),
            patch("avalan.tool.manager.uuid4", return_value=result_id),
        ):
            calls = namespaced_manager.get_calls(text)
            expected_call = ToolCall(
                id=call_id,
                name="math.calculator",
                arguments={"expression": "3"},
            )
            self.assertEqual(calls, [expected_call])

            results = await namespaced_manager(
                calls[0], context=ToolCallContext()
            )

            expected_result = ToolCallResult(
                id=result_id,
                call=expected_call,
                name="math.calculator",
                arguments={"expression": "3"},
                result="3",
            )
            self.assertEqual(results, expected_result)

    async def test_filter_scoped_to_namespace(self):
        def modify(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"expression": "2 + 2"},
                ),
                context,
            )

        math_set = ToolSet(namespace="math", tools=[CalculatorTool()])
        other_set = ToolSet(namespace="other", tools=[CalculatorTool()])
        manager = ToolManager.create_instance(
            enable_tools=["math.calculator", "other.calculator"],
            available_toolsets=[math_set, other_set],
            settings=ToolManagerSettings(
                filters=[ToolFilter(func=modify, namespace="math")]
            ),
        )

        call_math = ToolCall(
            id=_uuid4(), name="math.calculator", arguments={"expression": "1"}
        )
        call_other = ToolCall(
            id=_uuid4(), name="other.calculator", arguments={"expression": "1"}
        )
        with patch(
            "avalan.tool.manager.uuid4",
            side_effect=[_uuid4(), _uuid4()],
        ):
            res_math = await manager(call_math, context=ToolCallContext())
            res_other = await manager(call_other, context=ToolCallContext())

        self.assertEqual(res_math.call.arguments, {"expression": "2 + 2"})
        self.assertEqual(res_math.result, "4")
        self.assertEqual(res_other.call.arguments, {"expression": "1"})
        self.assertEqual(res_other.result, "1")

    async def test_transformer_scoped_to_full_namespace(self):
        def transform(_: ToolCall, __: ToolCallContext, result: str | None):
            return f"{result}!"

        manager = ToolManager.create_instance(
            enable_tools=["math.calculator", "calculator"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[CalculatorTool()]),
                ToolSet(tools=[CalculatorTool()]),
            ],
            settings=ToolManagerSettings(
                transformers=[
                    ToolTransformer(
                        func=transform, namespace="math.calculator"
                    )
                ]
            ),
        )

        call_math = ToolCall(
            id=_uuid4(), name="math.calculator", arguments={"expression": "1"}
        )
        call_plain = ToolCall(
            id=_uuid4(), name="calculator", arguments={"expression": "1"}
        )
        with patch(
            "avalan.tool.manager.uuid4",
            side_effect=[_uuid4(), _uuid4()],
        ):
            res_math = await manager(call_math, context=ToolCallContext())
            res_plain = await manager(call_plain, context=ToolCallContext())

        self.assertEqual(res_math.result, "1!")
        self.assertEqual(res_plain.result, "1")


if __name__ == "__main__":
    main()
