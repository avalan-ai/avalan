from asyncio import CancelledError, sleep
from dataclasses import replace
from enum import Enum
from typing import Any, Literal, TypedDict, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, patch
from uuid import uuid4 as _uuid4

from avalan.entities import (
    PreparedToolCall,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallRecoveryFormat,
    ToolCallResult,
    ToolCapabilities,
    ToolFilter,
    ToolFilterResult,
    ToolFilterResultStatus,
    ToolFormat,
    ToolManagerExecutionMode,
    ToolManagerMissingCallMode,
    ToolManagerSettings,
    ToolNameResolutionStatus,
    ToolParserReturnMode,
    ToolProviderArgumentsMode,
    ToolTransformer,
    ToolTransformerResult,
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
        self.assertIsNone(settings.maximum_argument_depth)
        self.assertIsNone(settings.maximum_argument_size)
        self.assertIsNone(settings.maximum_parser_input_size)
        self.assertIsNone(settings.maximum_parser_payload_depth)
        self.assertIsNone(settings.maximum_parser_payload_size)
        self.assertFalse(settings.parallel_tool_calls)
        self.assertEqual(settings.maximum_parallel_tool_calls, 4)
        self.assertEqual(settings.recovery_formats, [])

    def test_settings_accept_outcome_compatibility_modes(self):
        settings = ToolManagerSettings(
            execution_mode=ToolManagerExecutionMode.OUTCOMES,
            missing_call_mode=ToolManagerMissingCallMode.DIAGNOSTIC,
            parser_return_mode=ToolParserReturnMode.OUTCOME,
            provider_arguments_mode=(
                ToolProviderArgumentsMode.DIAGNOSTIC_ON_MALFORMED
            ),
            maximum_argument_depth=3,
            maximum_argument_size=64,
            maximum_parser_input_size=128,
            maximum_parser_payload_depth=4,
            maximum_parser_payload_size=96,
            parallel_tool_calls=True,
            maximum_parallel_tool_calls=2,
            recovery_formats=[ToolCallRecoveryFormat.FENCED],
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
        self.assertEqual(settings.maximum_argument_depth, 3)
        self.assertEqual(settings.maximum_argument_size, 64)
        self.assertEqual(settings.maximum_parser_input_size, 128)
        self.assertEqual(settings.maximum_parser_payload_depth, 4)
        self.assertEqual(settings.maximum_parser_payload_size, 96)
        self.assertTrue(settings.parallel_tool_calls)
        self.assertEqual(settings.maximum_parallel_tool_calls, 2)
        self.assertEqual(
            settings.recovery_formats, [ToolCallRecoveryFormat.FENCED]
        )

    def test_settings_reject_invalid_compatibility_modes(self):
        invalid_cases = (
            {"execution_mode": "legacy"},
            {"missing_call_mode": "legacy_none"},
            {"parser_return_mode": "legacy"},
            {"provider_arguments_mode": "empty_on_malformed"},
            {"maximum_depth": 0},
            {"maximum_argument_depth": 0},
            {"maximum_argument_size": -1},
            {"maximum_parser_input_size": True},
            {"maximum_parser_payload_depth": "1"},
            {"maximum_parser_payload_size": 0},
            {"parallel_tool_calls": 1},
            {"maximum_parallel_tool_calls": 0},
            {"recovery_formats": "fenced"},
            {"recovery_formats": ["fenced"]},
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(AssertionError):
                    ToolManagerSettings(**kwargs)

    def test_tool_call_error_type_uses_exception_or_projection(self):
        call = ToolCall(id="call-1", name="tool", arguments={})

        exception_error = ToolCallError(
            id="error-1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            error=ValueError("boom"),
            message="boom",
        )
        projected_error = ToolCallError(
            id="error-2",
            call=call,
            name=call.name,
            arguments=call.arguments,
            error={"type": "ProjectedError"},
            message="boom",
        )
        unknown_error = ToolCallError(
            id="error-3",
            call=call,
            name=call.name,
            arguments=call.arguments,
            error={"message": "boom"},
            message="boom",
        )

        self.assertEqual(exception_error.error_type, "ValueError")
        self.assertEqual(projected_error.error_type, "ProjectedError")
        self.assertEqual(unknown_error.error_type, "ToolCallError")

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

    def test_recovery_formats_pass_to_parser(self):
        settings = ToolManagerSettings(
            recovery_formats=[
                ToolCallRecoveryFormat.FENCED,
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
            ]
        )
        manager = ToolManager.create_instance(settings=settings)

        self.assertEqual(
            manager._parser.recovery_formats,
            (
                ToolCallRecoveryFormat.FENCED,
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
            ),
        )

    def test_get_calls_consumes_recovery_parser_output(self):
        call_id = _uuid4()
        manager = ToolManager.create_instance(
            settings=ToolManagerSettings(
                recovery_formats=[ToolCallRecoveryFormat.TOOL_CALL_BLOCK]
            )
        )

        with patch("avalan.tool.parser.uuid4", return_value=call_id):
            calls = manager.get_calls(
                '[TOOL_CALL]{"name": "calculator", "arguments": '
                '{"expression": "1 + 1"}}[/TOOL_CALL]'
            )

        self.assertEqual(
            calls,
            [
                ToolCall(
                    id=call_id,
                    name="calculator",
                    arguments={"expression": "1 + 1"},
                )
            ],
        )

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

    def test_enable_tools_namespace_uses_segment_boundaries(self):
        manager = ToolManager.create_instance(
            enable_tools=["math"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[CalculatorTool()]),
                ToolSet(namespace="mathx", tools=[CalculatorTool()]),
            ],
            settings=ToolManagerSettings(),
        )

        self.assertEqual(
            [descriptor.name for descriptor in manager.list_tools()],
            ["math.calculator"],
        )

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
        self.assertEqual(descriptors[0].capabilities, ToolCapabilities())
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

    def test_descriptors_expose_canonical_return_schemas(self):
        class Payload(TypedDict):
            name: str
            tags: list[str]

        def scalar() -> int:
            """Count records.

            Returns:
                Record count.
            """
            return 1

        def object_value() -> Payload:
            """Build a payload.

            Returns:
                Payload object.
            """
            return {"name": "ok", "tags": ["ready"]}

        def array_value() -> list[str]:
            """List values.

            Returns:
                Values.
            """
            return ["ready"]

        def nullable_value() -> str | None:
            """Maybe return a value.

            Returns:
                Optional value.
            """
            return None

        def union_value() -> str | int:
            """Return an identifier.

            Returns:
                Identifier.
            """
            return "ready"

        manager = ToolManager.create_instance(
            enable_tools=None,
            available_toolsets=[
                ToolSet(
                    tools=[
                        scalar,
                        object_value,
                        array_value,
                        nullable_value,
                        union_value,
                    ]
                )
            ],
            settings=ToolManagerSettings(),
        )

        descriptors = {
            descriptor.name: descriptor for descriptor in manager.list_tools()
        }

        self.assertEqual(
            descriptors["scalar"].return_schema,
            {"type": "integer", "description": "Record count."},
        )
        self.assertEqual(
            descriptors["object_value"].return_schema,
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "tags"],
                "additionalProperties": False,
                "description": "Payload object.",
            },
        )
        self.assertEqual(
            descriptors["array_value"].return_schema,
            {
                "type": "array",
                "items": {"type": "string"},
                "description": "Values.",
            },
        )
        self.assertEqual(
            descriptors["nullable_value"].return_schema,
            {
                "type": ["string", "null"],
                "description": "Optional value.",
            },
        )
        self.assertEqual(
            descriptors["union_value"].return_schema,
            {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
                "description": "Identifier.",
            },
        )
        for descriptor in descriptors.values():
            assert descriptor.schema is not None
            self.assertEqual(
                descriptor.return_schema,
                descriptor.schema["function"]["return"],
            )

    def test_tool_descriptor_ignores_malformed_return_schema(self):
        adder = DummyAdder()
        schema = {
            "type": "function",
            "function": {
                "name": "adder",
                "parameters": {"type": "object"},
                "return": ["integer"],
            },
        }

        descriptor = ToolManager._tool_descriptor(
            canonical_name="adder",
            tool=adder,
            aliases=[],
            namespace=None,
            schema=schema,
        )

        self.assertEqual(descriptor.parameter_schema, {"type": "object"})
        self.assertIsNone(descriptor.return_schema)

    def test_descriptors_expose_tool_capabilities_from_dataclass(self):
        streaming = StreamingSafeTool()
        manager = ToolManager.create_instance(
            enable_tools=["streaming_safe"],
            available_toolsets=[ToolSet(tools=[streaming])],
            settings=ToolManagerSettings(),
        )

        descriptor = manager.list_tools()[0]

        self.assertEqual(
            descriptor.capabilities,
            ToolCapabilities(
                supports_streaming=True,
                side_effecting=False,
                parallel_safe=True,
            ),
        )

    def test_descriptors_expose_tool_capabilities_from_mapping(self):
        mapped = MappedCapabilitiesTool()
        manager = ToolManager.create_instance(
            enable_tools=["mapped_capabilities"],
            available_toolsets=[ToolSet(tools=[mapped])],
            settings=ToolManagerSettings(),
        )

        descriptor = manager.list_tools()[0]

        self.assertEqual(
            descriptor.capabilities,
            ToolCapabilities(
                supports_streaming=True,
                side_effecting=True,
                parallel_safe=False,
            ),
        )

    def test_descriptors_expose_tool_capabilities_from_attributes(self):
        parallel = AttributeCapabilitiesTool()
        manager = ToolManager.create_instance(
            enable_tools=["attribute_capabilities"],
            available_toolsets=[ToolSet(tools=[parallel])],
            settings=ToolManagerSettings(),
        )

        descriptor = manager.list_tools()[0]

        self.assertEqual(
            descriptor.capabilities,
            ToolCapabilities(
                supports_streaming=False,
                side_effecting=False,
                parallel_safe=True,
            ),
        )

    def test_parallel_settings_are_exposed(self):
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(
                parallel_tool_calls=True,
                maximum_parallel_tool_calls=3,
            ),
        )

        self.assertTrue(manager.parallel_tool_calls)
        self.assertEqual(manager.maximum_parallel_tool_calls, 3)

    def test_tool_call_parallel_safety_uses_descriptor_capabilities(self):
        parallel = AttributeCapabilitiesTool()
        manager = ToolManager.create_instance(
            enable_tools=["attribute_capabilities"],
            available_toolsets=[ToolSet(tools=[parallel])],
            settings=ToolManagerSettings(),
        )

        self.assertTrue(
            manager.is_tool_call_parallel_safe(
                ToolCall(
                    id="call-1",
                    name="attribute_capabilities",
                    arguments={},
                )
            )
        )
        self.assertFalse(
            manager.is_tool_call_parallel_safe(
                ToolCall(id="call-2", name="missing", arguments={})
            )
        )
        self.assertFalse(
            manager.is_tool_call_parallel_safe(
                ToolCall(id="call-3", name="", arguments={})
            )
        )

    def test_tool_capabilities_reject_malformed_mapping(self):
        invalid_tools = (
            InvalidCapabilityMappingTool(),
            UnknownCapabilityMappingTool(),
        )

        for invalid in invalid_tools:
            with self.subTest(tool=invalid.__name__):
                with self.assertRaises(AssertionError):
                    ToolManager.create_instance(
                        enable_tools=[invalid.__name__],
                        available_toolsets=[ToolSet(tools=[invalid])],
                        settings=ToolManagerSettings(),
                    )

    def test_tool_capabilities_reject_malformed_attribute(self):
        invalid = InvalidCapabilityAttributeTool()

        with self.assertRaises(AssertionError):
            ToolManager.create_instance(
                enable_tools=["invalid_capability_attribute"],
                available_toolsets=[ToolSet(tools=[invalid])],
                settings=ToolManagerSettings(),
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
        self.assertEqual(descriptor.capabilities, ToolCapabilities())

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

    def test_resolve_tool_name_disabled_exact_wins_over_enabled_alias(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder()]),
                ToolSet(tools=[DummySum()]),
            ],
            settings=ToolManagerSettings(),
        )

        resolution = manager.resolve_tool_name("sum")

        self.assertIs(resolution.status, ToolNameResolutionStatus.DISABLED)
        self.assertIsNone(resolution.canonical_name)
        self.assertEqual(resolution.candidates, ["sum"])
        self.assertIs(
            resolution.diagnostic_code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )

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

    def test_validate_tool_call_rejects_disabled_exact_alias_collision(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder()]),
                ToolSet(tools=[DummySum()]),
            ],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call-1", name="sum", arguments={"a": 1, "b": 2})

        diagnostic = manager.validate_tool_call(call)

        assert diagnostic is not None
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.DISABLED_TOOL,
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)
        self.assertEqual(diagnostic.requested_name, "sum")
        self.assertIsNone(diagnostic.canonical_name)
        self.assertEqual(diagnostic.details["candidates"], ["sum"])

    def test_validate_tool_call_resolves_provider_encoded_name(self):
        manager = ToolManager.create_instance(
            enable_tools=["math.adder"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[DummyAdder()])
            ],
            settings=ToolManagerSettings(),
        )
        encoded_name = TextGenerationVendor.encode_tool_name("math.adder")
        call = ToolCall(
            id="call-1",
            name="math.adder",
            arguments={"a": 1, "b": 2},
            provider_name=encoded_name,
            provider_name_encoded=True,
        )

        self.assertIsNone(manager.validate_tool_call(call))

    def test_validate_tool_call_rejects_malformed_provider_arguments(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(
                provider_arguments_mode=(
                    ToolProviderArgumentsMode.DIAGNOSTIC_ON_MALFORMED
                )
            ),
        )
        call = ToolCall(
            id="call-1",
            name="adder",
            arguments={},
            provider_name="adder",
            provider_arguments_malformed=True,
        )

        diagnostic = manager.validate_tool_call(call)

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertEqual(diagnostic.call_id, "call-1")
        self.assertEqual(diagnostic.requested_name, "adder")
        self.assertEqual(diagnostic.canonical_name, "adder")
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertEqual(
            diagnostic.message,
            "Provider tool call arguments are malformed.",
        )

    def test_validate_tool_call_accepts_legacy_provider_arguments_mode(self):
        async def provider_no_args() -> str:
            return "ok"

        manager = ToolManager.create_instance(
            enable_tools=["provider_no_args"],
            available_toolsets=[ToolSet(tools=[provider_no_args])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id="call-1",
            name="provider_no_args",
            arguments={},
            provider_name="provider_no_args",
            provider_arguments_malformed=True,
        )

        self.assertIsNone(manager.validate_tool_call(call))

    def test_validate_tool_call_rejects_ambiguous_provider_alias(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder", "adder_alt"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyAdderAlt()])
            ],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id="call-1",
            name="sum",
            arguments={"a": 1, "b": 2},
            provider_name="sum",
        )

        diagnostic = manager.validate_tool_call(call)

        assert diagnostic is not None
        self.assertEqual(diagnostic.requested_name, "sum")
        self.assertIs(
            diagnostic.code, ToolCallDiagnosticCode.AMBIGUOUS_TOOL_NAME
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)
        self.assertEqual(
            diagnostic.details["candidates"], ["adder", "adder_alt"]
        )

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
        calls = (
            ToolCall(
                id="call-1",
                name="adder",
                arguments=cast(Any, ["a"]),
            ),
            ToolCall(id="call-2", name="adder", arguments=cast(Any, [])),
        )

        for call in calls:
            with self.subTest(call_id=call.id):
                diagnostic_id = _uuid4()

                with patch(
                    "avalan.tool.manager.uuid4",
                    return_value=diagnostic_id,
                ):
                    diagnostic = manager.validate_tool_call(call)

                assert diagnostic is not None
                self.assertEqual(diagnostic.id, diagnostic_id)
                self.assertIs(
                    diagnostic.code,
                    ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
                )
                self.assertIs(
                    diagnostic.stage,
                    ToolCallDiagnosticStage.VALIDATE,
                )

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

    def test_validate_tool_call_rejects_excessive_argument_depth(self):
        manager = ToolManager.create_instance(
            enable_tools=["nested_value"],
            available_toolsets=[ToolSet(tools=[nested_value])],
            settings=ToolManagerSettings(maximum_argument_depth=2),
        )

        diagnostic = manager.validate_tool_call(
            ToolCall(
                id="call-1",
                name="nested_value",
                arguments={"payload": {"nested": {"value": 1}}},
            )
        )

        assert diagnostic is not None
        self.assertEqual(diagnostic.canonical_name, "nested_value")
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.MAXIMUM_DEPTH)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertEqual(diagnostic.details["limit"], 2)

    def test_validate_tool_call_rejects_oversized_arguments(self):
        manager = ToolManager.create_instance(
            enable_tools=["nested_value"],
            available_toolsets=[ToolSet(tools=[nested_value])],
            settings=ToolManagerSettings(maximum_argument_size=8),
        )

        diagnostic = manager.validate_tool_call(
            ToolCall(
                id="call-1",
                name="nested_value",
                arguments={"payload": "large value"},
            )
        )

        assert diagnostic is not None
        self.assertEqual(diagnostic.canonical_name, "nested_value")
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.MAXIMUM_SIZE)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertEqual(diagnostic.details["limit"], 8)

    def test_validate_tool_call_rejects_wrong_json_types(self):
        async def typed_tool(
            payload: RuntimePayload,
            count: int,
            ratio: float,
            enabled: bool,
            mode: Literal["fast", "slow"],
            status: RuntimeMode,
        ) -> dict[str, Any]:
            return {"payload": payload, "count": count}

        manager = ToolManager.create_instance(
            enable_tools=["typed_tool"],
            available_toolsets=[ToolSet(tools=[typed_tool])],
            settings=ToolManagerSettings(),
        )
        cases = (
            (
                {"payload": {"name": "ok", "scores": [1], "note": None}},
                "$.count is required.",
            ),
            (
                {
                    "payload": {"name": "ok", "scores": [1], "note": None},
                    "count": 1,
                    "ratio": 1.5,
                    "enabled": True,
                    "mode": "fast",
                    "status": "slow",
                    "extra": "no",
                },
                "$.extra is not allowed.",
            ),
            (
                {
                    "payload": {"name": "ok", "scores": [1], "note": None},
                    "count": "1",
                    "ratio": 1.5,
                    "enabled": True,
                    "mode": "fast",
                    "status": "slow",
                },
                "$.count must be integer.",
            ),
            (
                {
                    "payload": {"name": "ok", "scores": [True], "note": None},
                    "count": 1,
                    "ratio": 1.5,
                    "enabled": True,
                    "mode": "fast",
                    "status": "slow",
                },
                "$.payload.scores[0] must be integer.",
            ),
            (
                {
                    "payload": {"name": "ok", "scores": [1], "note": 1},
                    "count": 1,
                    "ratio": 1.5,
                    "enabled": True,
                    "mode": "fast",
                    "status": "slow",
                },
                "$.payload.note must be string or null.",
            ),
            (
                {
                    "payload": {"name": "ok", "scores": [1], "note": None},
                    "count": 1,
                    "ratio": 1.5,
                    "enabled": True,
                    "mode": "medium",
                    "status": "slow",
                },
                "$.mode must be one of ['fast', 'slow'].",
            ),
            (
                {
                    "payload": {"name": "ok", "scores": [1], "note": None},
                    "count": 1,
                    "ratio": 1.5,
                    "enabled": True,
                    "mode": "fast",
                    "status": "medium",
                },
                "$.status must be one of ['fast', 'slow'].",
            ),
        )

        for arguments, message in cases:
            with self.subTest(message=message):
                diagnostic = manager.validate_tool_call(
                    ToolCall(
                        id="call-1",
                        name="typed_tool",
                        arguments=arguments,
                    )
                )

                assert diagnostic is not None
                self.assertIs(
                    diagnostic.code,
                    ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
                )
                self.assertIs(
                    diagnostic.stage,
                    ToolCallDiagnosticStage.VALIDATE,
                )
                self.assertEqual(diagnostic.message, message)

    def test_schema_validation_helper_covers_composite_shapes(self):
        self.assertIsNone(
            ToolManager._schema_validation_error(
                "ok",
                {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "$",
            )
        )
        self.assertEqual(
            ToolManager._schema_validation_error(
                False,
                {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "$",
            ),
            "$ does not match any allowed schema.",
        )
        self.assertIsNone(
            ToolManager._object_schema_validation_error(
                "not-object",
                {"type": "object"},
                "$",
            )
        )
        self.assertIsNone(
            ToolManager._array_schema_validation_error(
                "not-array",
                {"type": "array"},
                "$",
            )
        )
        self.assertEqual(
            ToolManager._array_schema_validation_error(
                [],
                {"type": "array", "minItems": 1},
                "$",
            ),
            "$ must contain at least 1 item(s).",
        )
        self.assertEqual(
            ToolManager._array_schema_validation_error(
                [1, 2],
                {"type": "array", "maxItems": 1},
                "$",
            ),
            "$ must contain at most 1 item(s).",
        )
        self.assertEqual(
            ToolManager._array_schema_validation_error(
                ["ok", "bad"],
                {
                    "type": "array",
                    "prefixItems": [
                        {"type": "string"},
                        {"type": "integer"},
                    ],
                    "minItems": 2,
                    "maxItems": 2,
                },
                "$",
            ),
            "$[1] must be integer.",
        )
        self.assertIsNone(
            ToolManager._array_schema_validation_error(
                ["ok"],
                {"type": "array", "prefixItems": [None]},
                "$",
            )
        )
        self.assertIsNone(
            ToolManager._array_schema_validation_error(
                ["ok", 1],
                {
                    "type": "array",
                    "prefixItems": [
                        {"type": "string"},
                        {"type": "integer"},
                    ],
                },
                "$",
            )
        )
        self.assertTrue(
            ToolManager._matches_schema_type(object(), "unhandled")
        )

    def test_validate_tool_call_uses_signature_when_schema_is_missing(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        descriptor = manager._descriptors["adder"]
        manager._descriptors["adder"] = replace(
            descriptor,
            parameter_schema=None,
        )

        diagnostic = manager.validate_tool_call(
            ToolCall(id="call-1", name="adder", arguments={"a": 1})
        )

        assert diagnostic is not None
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIn("missing", diagnostic.message)


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


class DummyMultiplier(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "multiplier"
        self.aliases = []

    async def __call__(self, a: int, b: int) -> int:
        """Return the product of ``a`` and ``b``."""
        return a * b


class DummySum(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "sum"
        self.aliases = []


class InvalidAliasesTool(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "invalid_aliases"
        self.aliases = "sum"


class StreamingSafeTool(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "streaming_safe"
        self.aliases = []
        self.tool_capabilities = ToolCapabilities(
            supports_streaming=True,
            side_effecting=False,
            parallel_safe=True,
        )


class MappedCapabilitiesTool(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "mapped_capabilities"
        self.aliases = []
        self.tool_capabilities = {"supports_streaming": True}


class AttributeCapabilitiesTool(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "attribute_capabilities"
        self.aliases = []
        self.side_effecting = False
        self.parallel_safe = True


class InvalidCapabilityMappingTool(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "invalid_capability_mapping"
        self.aliases = []
        self.tool_capabilities = {"supports_streaming": "yes"}


class UnknownCapabilityMappingTool(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "unknown_capability_mapping"
        self.aliases = []
        self.tool_capabilities = {"streaming": True}


class InvalidCapabilityAttributeTool(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "invalid_capability_attribute"
        self.aliases = []
        self.parallel_safe = "yes"


class RuntimePayload(TypedDict):
    name: str
    scores: list[int]
    note: str | None


class RuntimeMode(str, Enum):
    FAST = "fast"
    SLOW = "slow"


async def nested_value(payload: Any) -> Any:
    return payload


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


class ToolManagerPrepareCallTestCase(IsolatedAsyncioTestCase):
    async def test_prepare_call_returns_canonical_plan_for_alias(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call-1", name="sum", arguments={"a": 1, "b": 2})
        context = ToolCallContext()

        prepared = await manager.prepare_call(call, context=context)

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(
            prepared.call,
            ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2}),
        )
        self.assertIs(prepared.callable, adder)
        self.assertEqual(prepared.descriptor.name, "adder")
        self.assertEqual(prepared.arguments, {"a": 1, "b": 2})
        self.assertEqual(prepared.context, context)

    async def test_prepare_call_resolves_provider_encoded_name(self):
        adder = DummyAdder()
        manager = ToolManager.create_instance(
            enable_tools=["math.adder"],
            available_toolsets=[ToolSet(namespace="math", tools=[adder])],
            settings=ToolManagerSettings(),
        )
        provider_name = TextGenerationVendor.encode_tool_name("math.adder")
        call = ToolCall(
            id="call-1",
            name="math.adder",
            arguments={"a": 1, "b": 2},
            provider_name=provider_name,
            provider_name_encoded=True,
        )
        context = ToolCallContext()

        prepared = await manager.prepare_call(call, context=context)

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(
            prepared.call,
            ToolCall(
                id="call-1",
                name="math.adder",
                arguments={"a": 1, "b": 2},
                provider_name=provider_name,
                provider_name_encoded=True,
            ),
        )
        self.assertIs(prepared.callable, adder)

    async def test_prepare_call_rejects_ambiguous_provider_encoded_alias(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder", "adder_alt"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyAdderAlt()])
            ],
            settings=ToolManagerSettings(),
        )
        provider_name = TextGenerationVendor.encode_tool_name("sum")
        call = ToolCall(
            id="call-1",
            name="sum",
            arguments={"a": 1, "b": 2},
            provider_name=provider_name,
            provider_name_encoded=True,
        )

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertEqual(diagnostic.requested_name, provider_name)
        self.assertIs(
            diagnostic.code, ToolCallDiagnosticCode.AMBIGUOUS_TOOL_NAME
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)

    async def test_execute_call_runs_provider_encoded_name(self):
        manager = ToolManager.create_instance(
            enable_tools=["math.adder"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[DummyAdder()])
            ],
            settings=ToolManagerSettings(),
        )
        provider_name = TextGenerationVendor.encode_tool_name("math.adder")
        call = ToolCall(
            id="call-1",
            name="math.adder",
            arguments={"a": 2, "b": 3},
            provider_name=provider_name,
            provider_name_encoded=True,
        )
        result_id = _uuid4()

        with patch("avalan.tool.manager.uuid4", return_value=result_id):
            outcome = await manager.execute_call(
                call,
                context=ToolCallContext(),
            )

        self.assertEqual(
            outcome,
            ToolCallResult(
                id=result_id,
                call=ToolCall(
                    id="call-1",
                    name="math.adder",
                    arguments={"a": 2, "b": 3},
                    provider_name=provider_name,
                    provider_name_encoded=True,
                ),
                name="math.adder",
                arguments={"a": 2, "b": 3},
                provider_name=provider_name,
                provider_name_encoded=True,
                result=5,
            ),
        )

    async def test_execute_call_reports_malformed_provider_encoded_name(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        token = TextGenerationVendor.build_tool_call_token(
            call_id="call-1",
            tool_name="avl_notbase64",
            arguments={"a": 2, "b": 3},
        )
        diagnostic_id = _uuid4()

        with patch("avalan.tool.manager.uuid4", return_value=diagnostic_id):
            outcome = await manager.execute_call(
                token.call,
                context=ToolCallContext(),
            )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertEqual(outcome.id, diagnostic_id)
        self.assertEqual(outcome.call_id, "call-1")
        self.assertEqual(outcome.requested_name, "avl_notbase64")
        self.assertIsNone(outcome.canonical_name)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.MALFORMED_CALL)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.RESOLVE)

    async def test_execute_call_reports_empty_name_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id="call-1",
            name="",
            arguments={"a": 2, "b": 3},
        )
        diagnostic_id = _uuid4()

        with patch("avalan.tool.manager.uuid4", return_value=diagnostic_id):
            outcome = await manager.execute_call(
                call,
                context=ToolCallContext(),
            )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertEqual(outcome.id, diagnostic_id)
        self.assertEqual(outcome.call_id, "call-1")
        self.assertIsNone(outcome.requested_name)
        self.assertIsNone(outcome.canonical_name)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.MALFORMED_CALL)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.RESOLVE)
        self.assertEqual(outcome.message, "Tool call name must not be empty.")

    async def test_prepare_call_returns_resolution_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=[],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call-1", name="missing", arguments={})
        diagnostic_id = _uuid4()

        with patch("avalan.tool.manager.uuid4", return_value=diagnostic_id):
            diagnostic = await manager.prepare_call(
                call,
                context=ToolCallContext(),
            )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertEqual(diagnostic.id, diagnostic_id)
        self.assertEqual(diagnostic.call_id, "call-1")
        self.assertEqual(diagnostic.requested_name, "missing")
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.UNKNOWN_TOOL)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)

    async def test_prepare_call_returns_malformed_arguments_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        calls = (
            ToolCall(
                id="call-1",
                name="adder",
                arguments=cast(Any, ["a"]),
            ),
            ToolCall(id="call-2", name="adder", arguments=cast(Any, [])),
        )

        for call in calls:
            with self.subTest(call_id=call.id):
                diagnostic = await manager.prepare_call(
                    call,
                    context=ToolCallContext(),
                )

                self.assertIsInstance(diagnostic, ToolCallDiagnostic)
                assert isinstance(diagnostic, ToolCallDiagnostic)
                self.assertIs(
                    diagnostic.code,
                    ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
                )
                self.assertIs(
                    diagnostic.stage,
                    ToolCallDiagnosticStage.VALIDATE,
                )

    async def test_prepare_call_rejects_malformed_provider_arguments(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(
                provider_arguments_mode=(
                    ToolProviderArgumentsMode.DIAGNOSTIC_ON_MALFORMED
                )
            ),
        )
        call = ToolCall(
            id="call-1",
            name="adder",
            arguments={},
            provider_name="adder",
            provider_arguments_malformed=True,
        )

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertEqual(diagnostic.canonical_name, "adder")

    async def test_prepare_call_returns_repeated_call_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(avoid_repetition=True),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(calls=[call]),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.REPEATED_CALL)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.GUARD)

    async def test_prepare_call_returns_maximum_depth_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(maximum_depth=1),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(calls=[call]),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.MAXIMUM_DEPTH)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.GUARD)

    async def test_prepare_call_accepts_native_tool_arguments(self):
        tool = NativeAdderTool()
        manager = ToolManager.create_instance(
            enable_tools=["native_adder"],
            available_toolsets=[ToolSet(tools=[tool])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id="call-1",
            name="native_adder",
            arguments={"a": 1, "b": 2},
        )

        prepared = await manager.prepare_call(call, context=ToolCallContext())

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.arguments, {"a": 1, "b": 2})

    async def test_prepare_call_validates_arguments_after_filters(self):
        def drop_argument(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(id=call.id, name=call.name, arguments={"a": 1}),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[drop_argument]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertEqual(diagnostic.canonical_name, "adder")
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.VALIDATE)

    async def test_prepare_call_applies_argument_limits_after_filters(self):
        def deepen_argument(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"payload": {"nested": {"value": 1}}},
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["nested_value"],
            available_toolsets=[ToolSet(tools=[nested_value])],
            settings=ToolManagerSettings(
                filters=[deepen_argument],
                maximum_argument_depth=2,
            ),
        )
        call = ToolCall(
            id="call-1",
            name="nested_value",
            arguments={"payload": 1},
        )

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertEqual(diagnostic.canonical_name, "nested_value")
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.MAXIMUM_DEPTH)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.VALIDATE)

    async def test_prepare_call_rechecks_repetition_after_filters(self):
        def rewrite_to_adder(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name="adder",
                    arguments={"a": 2, "b": 3},
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder", "multiplier"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyMultiplier()])
            ],
            settings=ToolManagerSettings(
                avoid_repetition=True,
                filters=[rewrite_to_adder],
            ),
        )
        previous = ToolCall(
            id="call-0",
            name="adder",
            arguments={"a": 2, "b": 3},
        )
        call = ToolCall(
            id="call-1",
            name="multiplier",
            arguments={"a": 2, "b": 3},
        )

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(calls=[previous]),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.REPEATED_CALL)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.GUARD)
        self.assertEqual(diagnostic.requested_name, "adder")

    async def test_execute_prepared_call_does_not_rerun_filters(self):
        filter_calls = 0

        def replace_argument(call: ToolCall, context: ToolCallContext):
            nonlocal filter_calls
            filter_calls += 1
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"a": 3, "b": 4},
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[replace_argument]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})
        result_id = _uuid4()

        prepared = await manager.prepare_call(call, context=ToolCallContext())
        self.assertEqual(filter_calls, 1)
        assert isinstance(prepared, PreparedToolCall)

        with patch("avalan.tool.manager.uuid4", return_value=result_id):
            result = await manager.execute_prepared_call(prepared)

        self.assertEqual(filter_calls, 1)
        self.assertEqual(
            result,
            ToolCallResult(
                id=result_id,
                call=ToolCall(
                    id="call-1",
                    name="adder",
                    arguments={"a": 3, "b": 4},
                ),
                name="adder",
                arguments={"a": 3, "b": 4},
                result=7,
            ),
        )

    async def test_prepare_call_filter_can_repair_arguments(self):
        def add_argument(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"a": 1, "b": 2},
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[add_argument]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1})

        prepared = await manager.prepare_call(call, context=ToolCallContext())

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.arguments, {"a": 1, "b": 2})

    async def test_prepare_call_resolves_filter_name_rewrite(self):
        def rename(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name="multiplier",
                    arguments=call.arguments,
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder", "multiplier"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyMultiplier()])
            ],
            settings=ToolManagerSettings(filters=[rename]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        prepared = await manager.prepare_call(call, context=ToolCallContext())

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call.name, "multiplier")
        result = await manager.execute_prepared_call(prepared)
        self.assertIsInstance(result, ToolCallResult)
        assert isinstance(result, ToolCallResult)
        self.assertEqual(result.result, 6)

    async def test_prepare_call_returns_diagnostic_for_filter_unknown_rewrite(
        self,
    ):
        def rename(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(id=call.id, name="missing", arguments=call.arguments),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[rename]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertEqual(diagnostic.requested_name, "missing")
        self.assertIs(diagnostic.code, ToolCallDiagnosticCode.UNKNOWN_TOOL)
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.RESOLVE)

    async def test_legacy_call_returns_none_for_filter_unknown_rewrite(self):
        def rename(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(id=call.id, name="missing", arguments=call.arguments),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[rename]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        result = await manager(call, context=ToolCallContext())

        self.assertIsNone(result)

    async def test_prepare_call_accepts_explicit_filter_pass(self):
        def pass_call(
            _call: ToolCall,
            _context: ToolCallContext,
        ) -> ToolFilterResult:
            return ToolFilterResult(status=ToolFilterResultStatus.PASS)

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[pass_call]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        prepared = await manager.prepare_call(call, context=ToolCallContext())

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call, call)

    async def test_prepare_call_preserves_legacy_filter_none(self):
        def pass_call(
            _call: ToolCall,
            _context: ToolCallContext,
        ) -> None:
            return None

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[pass_call]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        prepared = await manager.prepare_call(call, context=ToolCallContext())

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call, call)

    async def test_prepare_call_accepts_explicit_filter_modify(self):
        def modify(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolFilterResult:
            return ToolFilterResult(
                status=ToolFilterResultStatus.MODIFY,
                call=ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"a": 3, "b": 4},
                ),
                context=context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[modify]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        prepared = await manager.prepare_call(call, context=ToolCallContext())

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.arguments, {"a": 3, "b": 4})

    async def test_prepare_call_returns_explicit_filter_suppress_diagnostic(
        self,
    ):
        diagnostic_id = _uuid4()

        def suppress(
            _call: ToolCall,
            _context: ToolCallContext,
        ) -> ToolFilterResult:
            return ToolFilterResult(
                status=ToolFilterResultStatus.SUPPRESS,
                message="Blocked by policy.",
                details={"reason": "policy"},
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[suppress]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        with patch("avalan.tool.manager.uuid4", return_value=diagnostic_id):
            diagnostic = await manager.prepare_call(
                call,
                context=ToolCallContext(),
            )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertEqual(diagnostic.id, diagnostic_id)
        self.assertEqual(diagnostic.canonical_name, "adder")
        self.assertIs(
            diagnostic.code, ToolCallDiagnosticCode.FILTER_SUPPRESSED
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.FILTER)
        self.assertEqual(diagnostic.message, "Blocked by policy.")
        self.assertEqual(diagnostic.details, {"reason": "policy"})

    async def test_legacy_call_returns_none_for_explicit_filter_suppress(self):
        def suppress(
            _call: ToolCall,
            _context: ToolCallContext,
        ) -> ToolFilterResult:
            return ToolFilterResult(status=ToolFilterResultStatus.SUPPRESS)

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[suppress]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        result = await manager(call, context=ToolCallContext())

        self.assertIsNone(result)

    async def test_prepare_call_rejects_flow_tool_node_filter_name_rewrite(
        self,
    ):
        def rename(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name="multiplier",
                    arguments=call.arguments,
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder", "multiplier"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyMultiplier()])
            ],
            settings=ToolManagerSettings(filters=[rename]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(flow_tool_node=True),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(
            diagnostic.code, ToolCallDiagnosticCode.FILTER_SUPPRESSED
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.FILTER)
        self.assertEqual(diagnostic.canonical_name, "adder")
        self.assertEqual(diagnostic.details, {"filtered_name": "multiplier"})

    async def test_prepare_call_keeps_unknown_filter_suppression_uncanonical(
        self,
    ) -> None:
        for rewritten_name in ("missing", ""):
            with self.subTest(rewritten_name=rewritten_name):

                def rename(
                    call: ToolCall,
                    context: ToolCallContext,
                ) -> tuple[ToolCall, ToolCallContext]:
                    return (
                        ToolCall(
                            id=call.id,
                            name=rewritten_name,
                            arguments=call.arguments,
                        ),
                        context,
                    )

                def suppress(
                    _call: ToolCall,
                    _context: ToolCallContext,
                ) -> ToolFilterResult:
                    return ToolFilterResult(
                        status=ToolFilterResultStatus.SUPPRESS
                    )

                manager = ToolManager.create_instance(
                    enable_tools=["adder"],
                    available_toolsets=[ToolSet(tools=[DummyAdder()])],
                    settings=ToolManagerSettings(filters=[rename, suppress]),
                )
                call = ToolCall(
                    id="call-1",
                    name="adder",
                    arguments={"a": 2, "b": 3},
                )

                diagnostic = await manager.prepare_call(
                    call,
                    context=ToolCallContext(),
                )

                self.assertIsInstance(diagnostic, ToolCallDiagnostic)
                assert isinstance(diagnostic, ToolCallDiagnostic)
                self.assertIsNone(diagnostic.canonical_name)
                self.assertIs(
                    diagnostic.code,
                    ToolCallDiagnosticCode.FILTER_SUPPRESSED,
                )

    async def test_prepare_call_keeps_flow_filter_guard_after_context_replace(
        self,
    ) -> None:
        seen_flow_markers: list[bool] = []

        def replace_context(
            call: ToolCall,
            context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            seen_flow_markers.append(context.flow_tool_node)
            return call, ToolCallContext()

        def rename(
            call: ToolCall,
            context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            seen_flow_markers.append(context.flow_tool_node)
            return (
                ToolCall(
                    id=call.id,
                    name="multiplier",
                    arguments=call.arguments,
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder", "multiplier"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyMultiplier()])
            ],
            settings=ToolManagerSettings(filters=[replace_context, rename]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(flow_tool_node=True),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(
            diagnostic.code, ToolCallDiagnosticCode.FILTER_SUPPRESSED
        )
        self.assertIs(diagnostic.stage, ToolCallDiagnosticStage.FILTER)
        self.assertEqual(diagnostic.canonical_name, "adder")
        self.assertEqual(diagnostic.details, {"filtered_name": "multiplier"})
        self.assertEqual(seen_flow_markers, [True, True])

    async def test_prepare_call_rejects_flow_provider_name_rewrite(
        self,
    ) -> None:
        cases = ("multiplier", "pkg.tool")

        for provider_name in cases:
            with self.subTest(provider_name=provider_name):

                def set_provider_name(
                    call: ToolCall,
                    _context: ToolCallContext,
                ) -> tuple[ToolCall, ToolCallContext]:
                    return (
                        ToolCall(
                            id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                            provider_name=provider_name,
                        ),
                        ToolCallContext(),
                    )

                manager = ToolManager.create_instance(
                    enable_tools=["adder", "multiplier"],
                    available_toolsets=[
                        ToolSet(tools=[DummyAdder(), DummyMultiplier()])
                    ],
                    settings=ToolManagerSettings(filters=[set_provider_name]),
                )
                call = ToolCall(
                    id="call-1",
                    name="adder",
                    arguments={"a": 2, "b": 3},
                )

                diagnostic = await manager.prepare_call(
                    call,
                    context=ToolCallContext(flow_tool_node=True),
                )

                self.assertIsInstance(diagnostic, ToolCallDiagnostic)
                assert isinstance(diagnostic, ToolCallDiagnostic)
                self.assertIs(
                    diagnostic.code,
                    ToolCallDiagnosticCode.FILTER_SUPPRESSED,
                )
                self.assertIs(
                    diagnostic.stage,
                    ToolCallDiagnosticStage.FILTER,
                )
                self.assertEqual(
                    diagnostic.details, {"filtered_name": provider_name}
                )

    async def test_prepare_call_accepts_flow_same_provider_name(
        self,
    ) -> None:
        def set_provider_name(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    provider_name="adder",
                ),
                ToolCallContext(),
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[set_provider_name]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        prepared = await manager.prepare_call(
            call,
            context=ToolCallContext(flow_tool_node=True),
        )

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call.name, "adder")

    async def test_prepare_call_accepts_flow_alias_for_same_tool(
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
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[set_alias]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        prepared = await manager.prepare_call(
            call,
            context=ToolCallContext(flow_tool_node=True),
        )

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call.name, "adder")

    async def test_prepare_call_matches_filter_after_flow_alias_rewrite(
        self,
    ) -> None:
        called: list[str] = []

        def set_alias(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            assert call.arguments is not None
            called.append("alias")
            return (
                ToolCall(
                    id=call.id,
                    name="sum",
                    arguments={"a": call.arguments["a"]},
                ),
                ToolCallContext(),
            )

        def add_argument(
            call: ToolCall,
            context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            assert call.arguments is not None
            called.append("adder")
            return (
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments={"a": call.arguments["a"], "b": 5},
                ),
                context,
            )

        def unrelated(
            call: ToolCall,
            context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            called.append("multiplier")
            return call, context

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(
                filters=[
                    set_alias,
                    ToolFilter(func=unrelated, namespace="multiplier"),
                    ToolFilter(func=add_argument, namespace="adder"),
                ],
            ),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        prepared = await manager.prepare_call(
            call,
            context=ToolCallContext(flow_tool_node=True),
        )

        self.assertIsInstance(prepared, PreparedToolCall)
        assert isinstance(prepared, PreparedToolCall)
        self.assertEqual(prepared.call.name, "adder")
        self.assertEqual(prepared.arguments, {"a": 2, "b": 5})
        self.assertEqual(called, ["alias", "adder"])

    async def test_prepare_call_rejects_flow_alias_for_other_tool(
        self,
    ) -> None:
        multiplier = DummyMultiplier()
        multiplier.aliases = ["product"]

        def set_alias(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            return (
                ToolCall(
                    id=call.id,
                    name="product",
                    arguments=call.arguments,
                ),
                ToolCallContext(),
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder", "multiplier"],
            available_toolsets=[ToolSet(tools=[DummyAdder(), multiplier])],
            settings=ToolManagerSettings(filters=[set_alias]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(flow_tool_node=True),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.FILTER_SUPPRESSED,
        )
        self.assertEqual(diagnostic.details, {"filtered_name": "multiplier"})

    async def test_prepare_call_rejects_flow_blank_filter_name(
        self,
    ) -> None:
        def clear_name(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            return (
                ToolCall(
                    id=call.id,
                    name="",
                    arguments=call.arguments,
                ),
                ToolCallContext(),
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[clear_name]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        diagnostic = await manager.prepare_call(
            call,
            context=ToolCallContext(flow_tool_node=True),
        )

        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.FILTER_SUPPRESSED,
        )
        self.assertEqual(diagnostic.details, {"filtered_name": ""})

    async def test_prepare_call_keeps_flow_cancellation_after_context_replace(
        self,
    ) -> None:
        called: list[str] = []

        async def cancel() -> None:
            called.append("cancel")
            if len(called) == 3:
                raise CancelledError()

        def replace_context(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            called.append("filter")
            return call, ToolCallContext()

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[replace_context]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        with self.assertRaises(CancelledError):
            await manager.prepare_call(
                call,
                context=ToolCallContext(
                    cancellation_checker=cancel,
                    flow_tool_node=True,
                ),
            )

        self.assertEqual(called, ["cancel", "filter", "cancel"])

    async def test_prepare_call_keeps_explicit_flow_cancellation_replacement(
        self,
    ) -> None:
        called: list[str] = []

        async def first_check() -> None:
            called.append("first")

        async def second_check() -> None:
            called.append("second")
            raise CancelledError()

        def replace_context(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            called.append("filter")
            return call, ToolCallContext(cancellation_checker=second_check)

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[replace_context]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        with self.assertRaises(CancelledError):
            await manager.prepare_call(
                call,
                context=ToolCallContext(
                    cancellation_checker=first_check,
                    flow_tool_node=True,
                ),
            )

        self.assertEqual(called, ["first", "filter", "second"])


class ToolManagerExecuteCallTestCase(IsolatedAsyncioTestCase):
    async def test_execute_call_returns_result_for_prepared_call(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})
        result_id = _uuid4()

        with patch("avalan.tool.manager.uuid4", return_value=result_id):
            outcome = await manager.execute_call(
                call,
                context=ToolCallContext(),
            )

        self.assertEqual(
            outcome,
            ToolCallResult(
                id=result_id,
                call=call,
                name="adder",
                arguments={"a": 1, "b": 2},
                result=3,
            ),
        )

    async def test_execute_call_validates_arguments_before_dispatch(self):
        calls: list[dict[str, Any]] = []

        async def typed_tool(
            payload: RuntimePayload,
            count: int,
            ratio: float,
            enabled: bool,
            mode: Literal["fast", "slow"],
            status: RuntimeMode,
        ) -> dict[str, Any]:
            calls.append(
                {
                    "payload": payload,
                    "count": count,
                    "ratio": ratio,
                    "enabled": enabled,
                    "mode": mode,
                    "status": status,
                }
            )
            return calls[-1]

        manager = ToolManager.create_instance(
            enable_tools=["typed_tool"],
            available_toolsets=[ToolSet(tools=[typed_tool])],
            settings=ToolManagerSettings(),
        )
        valid_arguments = {
            "payload": {"name": "ok", "scores": [1, 2], "note": None},
            "count": 2,
            "ratio": 1.5,
            "enabled": False,
            "mode": "fast",
            "status": "slow",
        }

        result = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="typed_tool",
                arguments=valid_arguments,
            ),
            context=ToolCallContext(),
        )
        diagnostic = await manager.execute_call(
            ToolCall(
                id="call-2",
                name="typed_tool",
                arguments={**valid_arguments, "count": True},
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(result, ToolCallResult)
        assert isinstance(result, ToolCallResult)
        self.assertEqual(result.result, valid_arguments)
        self.assertEqual(calls, [valid_arguments])
        self.assertIsInstance(diagnostic, ToolCallDiagnostic)
        assert isinstance(diagnostic, ToolCallDiagnostic)
        self.assertIs(
            diagnostic.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertEqual(diagnostic.message, "$.count must be integer.")
        self.assertEqual(calls, [valid_arguments])

    async def test_execute_call_rejects_argument_limits_before_dispatch(self):
        calls: list[Any] = []

        async def limited_tool(payload: Any) -> Any:
            calls.append(payload)
            return payload

        manager = ToolManager.create_instance(
            enable_tools=["limited_tool"],
            available_toolsets=[ToolSet(tools=[limited_tool])],
            settings=ToolManagerSettings(maximum_argument_size=8),
        )

        outcome = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="limited_tool",
                arguments={"payload": "large value"},
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.MAXIMUM_SIZE)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertEqual(calls, [])

    async def test_execute_call_rejects_malformed_provider_arguments(self):
        calls = 0

        async def provider_no_args() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        manager = ToolManager.create_instance(
            enable_tools=["provider_no_args"],
            available_toolsets=[ToolSet(tools=[provider_no_args])],
            settings=ToolManagerSettings(
                provider_arguments_mode=(
                    ToolProviderArgumentsMode.DIAGNOSTIC_ON_MALFORMED
                )
            ),
        )
        malformed = await manager.execute_call(
            ToolCall(
                id="call-1",
                name="provider_no_args",
                arguments={},
                provider_name="provider_no_args",
                provider_arguments_malformed=True,
            ),
            context=ToolCallContext(),
        )
        valid_empty = await manager.execute_call(
            ToolCall(
                id="call-2",
                name="provider_no_args",
                arguments={},
                provider_name="provider_no_args",
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(malformed, ToolCallDiagnostic)
        assert isinstance(malformed, ToolCallDiagnostic)
        self.assertIs(
            malformed.code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )
        self.assertIs(malformed.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertIsInstance(valid_empty, ToolCallResult)
        assert isinstance(valid_empty, ToolCallResult)
        self.assertEqual(valid_empty.result, "ok")
        self.assertEqual(calls, 1)

    async def test_execute_call_rejects_disabled_exact_alias_collision(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder()]),
                ToolSet(tools=[DummySum()]),
            ],
            settings=ToolManagerSettings(),
        )

        outcome = await manager.execute_call(
            ToolCall(id="call-1", name="sum", arguments={"a": 1, "b": 2}),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.DISABLED_TOOL)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.RESOLVE)
        self.assertEqual(outcome.requested_name, "sum")
        self.assertIsNone(outcome.canonical_name)
        self.assertEqual(outcome.details["candidates"], ["sum"])

    async def test_execute_call_returns_execution_error(self):
        async def failing_tool(a: int) -> None:
            raise ValueError("boom")

        manager = ToolManager.create_instance(
            enable_tools=["failing_tool"],
            available_toolsets=[ToolSet(tools=[failing_tool])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call-1", name="failing_tool", arguments={"a": 1})

        outcome = await manager.execute_call(call, context=ToolCallContext())

        self.assertIsInstance(outcome, ToolCallError)
        assert isinstance(outcome, ToolCallError)
        self.assertEqual(outcome.error, {"type": "ValueError"})
        self.assertEqual(outcome.error_type, "ValueError")
        self.assertEqual(outcome.message, "boom")

    async def test_execute_call_returns_resolution_diagnostics(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder", "adder_alt"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyAdderAlt()]),
                ToolSet(namespace="disabled", tools=[CalculatorTool()]),
            ],
            settings=ToolManagerSettings(),
        )
        cases = (
            (
                ToolCall(id="call-1", name="missing", arguments={}),
                ToolCallDiagnosticCode.UNKNOWN_TOOL,
            ),
            (
                ToolCall(
                    id="call-2",
                    name="disabled.calculator",
                    arguments={},
                ),
                ToolCallDiagnosticCode.DISABLED_TOOL,
            ),
            (
                ToolCall(id="call-3", name="sum", arguments={}),
                ToolCallDiagnosticCode.AMBIGUOUS_TOOL_NAME,
            ),
        )

        for call, code in cases:
            with self.subTest(code=code):
                outcome = await manager.execute_call(
                    call,
                    context=ToolCallContext(),
                )

                self.assertIsInstance(outcome, ToolCallDiagnostic)
                assert isinstance(outcome, ToolCallDiagnostic)
                self.assertIs(outcome.code, code)
                self.assertIs(outcome.stage, ToolCallDiagnosticStage.RESOLVE)

    async def test_execute_call_returns_filter_diagnostic(self):
        def suppress(
            _call: ToolCall,
            _context: ToolCallContext,
        ) -> ToolFilterResult:
            return ToolFilterResult(status=ToolFilterResultStatus.SUPPRESS)

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(filters=[suppress]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})

        outcome = await manager.execute_call(call, context=ToolCallContext())

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.FILTER_SUPPRESSED)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.FILTER)

    async def test_execute_call_returns_guard_diagnostics(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(
                avoid_repetition=True,
                maximum_depth=1,
            ),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1, "b": 2})
        cases = (
            (
                ToolCallContext(calls=[call]),
                ToolCallDiagnosticCode.REPEATED_CALL,
            ),
            (
                ToolCallContext(
                    calls=[
                        ToolCall(
                            id="call-2",
                            name="adder",
                            arguments={"a": 3, "b": 4},
                        )
                    ]
                ),
                ToolCallDiagnosticCode.MAXIMUM_DEPTH,
            ),
        )

        for context, code in cases:
            with self.subTest(code=code):
                outcome = await manager.execute_call(call, context=context)

                self.assertIsInstance(outcome, ToolCallDiagnostic)
                assert isinstance(outcome, ToolCallDiagnostic)
                self.assertIs(outcome.code, code)
                self.assertIs(outcome.stage, ToolCallDiagnosticStage.GUARD)

    async def test_execute_call_rechecks_depth_after_filter_context(self):
        previous = ToolCall(
            id="call-0",
            name="adder",
            arguments={"a": 2, "b": 3},
        )

        def add_history(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> tuple[ToolCall, ToolCallContext]:
            return call, ToolCallContext(calls=[previous])

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(
                maximum_depth=1,
                filters=[add_history],
            ),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 2, "b": 3})

        outcome = await manager.execute_call(call, context=ToolCallContext())

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.MAXIMUM_DEPTH)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.GUARD)

    async def test_execute_call_returns_cancellation_before_filters(self):
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
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1})

        outcome = await manager.execute_call(
            call,
            context=ToolCallContext(cancellation_checker=cancel),
        )

        self.assertEqual(called, ["cancel"])
        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.CANCELLED)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.GUARD)
        self.assertEqual(outcome.details, {})

    async def test_execute_call_returns_cancellation_after_confirmation(self):
        called: list[str] = []

        async def adder(a: int) -> int:
            called.append("tool")
            return a + 1

        async def first_check() -> None:
            called.append("first")

        second_checks = 0

        async def second_check() -> None:
            nonlocal second_checks
            second_checks += 1
            called.append(f"second:{second_checks}")
            if second_checks == 2:
                raise CancelledError()

        def replace_checker(call: ToolCall, context: ToolCallContext):
            called.append("filter")
            return (
                call,
                ToolCallContext(
                    calls=context.calls,
                    cancellation_checker=second_check,
                ),
            )

        def confirm(call: ToolCall) -> str:
            called.append(f"confirm:{call.name}")
            return "y"

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[adder])],
            settings=ToolManagerSettings(filters=[replace_checker]),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1})

        outcome = await manager.execute_call(
            call,
            context=ToolCallContext(cancellation_checker=first_check),
            confirm=confirm,
        )

        self.assertEqual(
            called,
            ["first", "filter", "second:1", "confirm:adder", "second:2"],
        )
        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(outcome.code, ToolCallDiagnosticCode.CANCELLED)
        self.assertEqual(outcome.canonical_name, "adder")

    async def test_execute_call_returns_confirmation_rejection(self):
        async def reject(_call: ToolCall) -> str:
            return "n"

        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call-1", name="sum", arguments={"a": 1, "b": 2})

        outcome = await manager.execute_call(
            call,
            context=ToolCallContext(),
            confirm=reject,
        )

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertEqual(outcome.requested_name, "adder")
        self.assertEqual(outcome.canonical_name, "adder")
        self.assertIs(outcome.code, ToolCallDiagnosticCode.USER_REJECTED)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.CONFIRM)


class ToolManagerOutcomeModeCallTestCase(IsolatedAsyncioTestCase):
    async def test_call_uses_outcome_mode_to_execute_alias(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(
                execution_mode=ToolManagerExecutionMode.OUTCOMES
            ),
        )
        call = ToolCall(id="call-1", name="sum", arguments={"a": 1, "b": 2})
        result_id = _uuid4()

        with patch("avalan.tool.manager.uuid4", return_value=result_id):
            outcome = await manager(call, context=ToolCallContext())

        self.assertEqual(
            outcome,
            ToolCallResult(
                id=result_id,
                call=ToolCall(
                    id="call-1",
                    name="adder",
                    arguments={"a": 1, "b": 2},
                ),
                name="adder",
                arguments={"a": 1, "b": 2},
                result=3,
            ),
        )

    async def test_legacy_call_still_returns_none_for_alias(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call-1", name="sum", arguments={"a": 1, "b": 2})

        outcome = await manager(call, context=ToolCallContext())

        self.assertIsNone(outcome)

    async def test_call_uses_outcome_mode_for_resolution_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=[],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(
                execution_mode=ToolManagerExecutionMode.OUTCOMES
            ),
        )
        call = ToolCall(id="call-1", name="sum", arguments={"a": 1, "b": 2})
        diagnostic_id = _uuid4()

        with patch("avalan.tool.manager.uuid4", return_value=diagnostic_id):
            outcome = await manager(call, context=ToolCallContext())

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertEqual(outcome.id, diagnostic_id)
        self.assertEqual(outcome.call_id, "call-1")
        self.assertEqual(outcome.requested_name, "sum")
        self.assertIs(outcome.code, ToolCallDiagnosticCode.DISABLED_TOOL)
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.RESOLVE)

    async def test_call_uses_outcome_mode_for_validation_diagnostic(self):
        manager = ToolManager.create_instance(
            enable_tools=["adder"],
            available_toolsets=[ToolSet(tools=[DummyAdder()])],
            settings=ToolManagerSettings(
                execution_mode=ToolManagerExecutionMode.OUTCOMES
            ),
        )
        call = ToolCall(id="call-1", name="adder", arguments={"a": 1})

        outcome = await manager(call, context=ToolCallContext())

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertEqual(outcome.call_id, "call-1")
        self.assertIs(
            outcome.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.VALIDATE)

    async def test_call_uses_outcome_mode_for_provider_argument_diagnostic(
        self,
    ):
        calls = 0

        async def provider_no_args() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        manager = ToolManager.create_instance(
            enable_tools=["provider_no_args"],
            available_toolsets=[ToolSet(tools=[provider_no_args])],
            settings=ToolManagerSettings(
                execution_mode=ToolManagerExecutionMode.OUTCOMES,
                provider_arguments_mode=(
                    ToolProviderArgumentsMode.DIAGNOSTIC_ON_MALFORMED
                ),
            ),
        )
        call = ToolCall(
            id="call-1",
            name="provider_no_args",
            arguments={},
            provider_name="provider_no_args",
            provider_arguments_malformed=True,
        )

        outcome = await manager(call, context=ToolCallContext())

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(
            outcome.code,
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS,
        )
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.VALIDATE)
        self.assertEqual(calls, 0)


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

    async def test_non_native_tool_dispatches_arguments_by_keyword(self):
        async def subtract(a: int, b: int) -> int:
            return a - b

        manager = ToolManager.create_instance(
            enable_tools=[subtract.__name__],
            available_toolsets=[ToolSet(tools=[subtract])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(), name=subtract.__name__, arguments={"b": 2, "a": 5}
        )
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, 3)

    async def test_non_native_tool_supports_positional_only_arguments(self):
        async def subtract(a: int, /, b: int) -> int:
            return a - b

        manager = ToolManager.create_instance(
            enable_tools=[subtract.__name__],
            available_toolsets=[ToolSet(tools=[subtract])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(), name=subtract.__name__, arguments={"b": 2, "a": 5}
        )
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, 3)

    async def test_execute_call_rejects_varargs_arguments(self):
        async def collect(*values: int) -> int:
            return sum(values)

        manager = ToolManager.create_instance(
            enable_tools=[collect.__name__],
            available_toolsets=[ToolSet(tools=[collect])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(),
            name=collect.__name__,
            arguments={"values": cast(Any, [1, 2])},
        )

        outcome = await manager.execute_call(call, context=ToolCallContext())

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(
            outcome.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertIs(outcome.stage, ToolCallDiagnosticStage.VALIDATE)

    async def test_non_native_tool_supports_variadic_keywords(self):
        async def collect(**values: int) -> dict[str, int]:
            return values

        manager = ToolManager.create_instance(
            enable_tools=[collect.__name__],
            available_toolsets=[ToolSet(tools=[collect])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(), name=collect.__name__, arguments={"b": 2, "a": 1}
        )
        result = await manager(call, context=ToolCallContext())
        self.assertEqual(result.result, {"b": 2, "a": 1})

    async def test_execute_call_rejects_invalid_variadic_keyword_value(self):
        async def collect(**values: int) -> dict[str, int]:
            return values

        manager = ToolManager.create_instance(
            enable_tools=[collect.__name__],
            available_toolsets=[ToolSet(tools=[collect])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id=_uuid4(),
            name=collect.__name__,
            arguments={"a": "1"},
        )

        outcome = await manager.execute_call(call, context=ToolCallContext())

        self.assertIsInstance(outcome, ToolCallDiagnostic)
        assert isinstance(outcome, ToolCallDiagnostic)
        self.assertIs(
            outcome.code,
            ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        )
        self.assertEqual(outcome.message, "$.a must be integer.")

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

    async def test_react_calls_after_malformed_input_execute(self):
        async def echo_input(input: str) -> str:
            return input

        manager = ToolManager.create_instance(
            enable_tools=["calculator", "echo_input"],
            available_toolsets=[ToolSet(tools=[CalculatorTool(), echo_input])],
            settings=ToolManagerSettings(tool_format=ToolFormat.REACT),
        )
        outcome = manager.parse_calls(
            'Action: broken\nAction Input: {"expression": '
            "\nAction: calculator\n"
            'Action Input: {"expression": "1 + 1"}\n'
            "Action: echo_input\n"
            'Action Input: {"input": "done"}'
        )

        self.assertEqual(len(outcome.calls), 2)
        self.assertEqual(len(outcome.diagnostics), 1)
        self.assertEqual(
            outcome.diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )

        first = await manager.execute_call(
            outcome.calls[0],
            context=ToolCallContext(),
        )
        second = await manager.execute_call(
            outcome.calls[1],
            context=ToolCallContext(),
        )

        self.assertIsInstance(first, ToolCallResult)
        self.assertIsInstance(second, ToolCallResult)
        assert isinstance(first, ToolCallResult)
        assert isinstance(second, ToolCallResult)
        self.assertEqual(first.result, "2")
        self.assertEqual(second.result, "done")

    def test_get_calls_preserves_multiple_tag_calls_after_xml_parse_error(
        self,
    ):
        text = (
            '<tool_call name="calculator">{"expression": "1"}</tool_call>'
            "&"
            '<tool_call name="calculator">{"expression": "2"}</tool_call>'
        )
        first_id = _uuid4()
        second_id = _uuid4()

        with patch(
            "avalan.tool.parser.uuid4", side_effect=[first_id, second_id]
        ):
            calls = self.manager.get_calls(text)

        self.assertEqual(
            calls,
            [
                ToolCall(
                    id=first_id,
                    name="calculator",
                    arguments={"expression": "1"},
                ),
                ToolCall(
                    id=second_id,
                    name="calculator",
                    arguments={"expression": "2"},
                ),
            ],
        )

    async def test_react_namespaced_tool_executes(self):
        manager = ToolManager.create_instance(
            enable_tools=["math.calculator"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[CalculatorTool()])
            ],
            settings=ToolManagerSettings(tool_format=ToolFormat.REACT),
        )
        call_id = _uuid4()
        result_id = _uuid4()

        with (
            patch("avalan.tool.parser.uuid4", return_value=call_id),
            patch("avalan.tool.manager.uuid4", return_value=result_id),
        ):
            calls = manager.get_calls(
                'Action: math.calculator\nAction Input: {"expression": "2"}'
            )
            assert calls is not None
            result = await manager(calls[0], context=ToolCallContext())

        expected_call = ToolCall(
            id=call_id,
            name="math.calculator",
            arguments={"expression": "2"},
        )
        self.assertEqual(calls, [expected_call])
        self.assertEqual(
            result,
            ToolCallResult(
                id=result_id,
                call=expected_call,
                name="math.calculator",
                arguments={"expression": "2"},
                result="2",
            ),
        )

    async def test_bracket_namespaced_tool_executes(self):
        async def echo_input(input: str) -> str:
            return input

        manager = ToolManager.create_instance(
            enable_tools=["text.echo_input"],
            available_toolsets=[ToolSet(namespace="text", tools=[echo_input])],
            settings=ToolManagerSettings(tool_format=ToolFormat.BRACKET),
        )
        call_id = _uuid4()
        result_id = _uuid4()

        with (
            patch("avalan.tool.parser.uuid4", return_value=call_id),
            patch("avalan.tool.manager.uuid4", return_value=result_id),
        ):
            calls = manager.get_calls("[text.echo_input](hello)")
            assert calls is not None
            result = await manager(calls[0], context=ToolCallContext())

        expected_call = ToolCall(
            id=call_id,
            name="text.echo_input",
            arguments={"input": "hello"},
        )
        self.assertEqual(calls, [expected_call])
        self.assertEqual(
            result,
            ToolCallResult(
                id=result_id,
                call=expected_call,
                name="text.echo_input",
                arguments={"input": "hello"},
                result="hello",
            ),
        )

    def test_parse_calls_reports_malformed_bracket_with_valid_call(self):
        manager = ToolManager.create_instance(
            settings=ToolManagerSettings(tool_format=ToolFormat.BRACKET),
        )
        call_id = _uuid4()
        diagnostic_id = _uuid4()

        with patch(
            "avalan.tool.parser.uuid4",
            side_effect=[call_id, diagnostic_id],
        ):
            outcome = manager.parse_calls(
                "[math..calculator](bad)\n[math.calculator](2)"
            )

        self.assertEqual(
            outcome.calls,
            [
                ToolCall(
                    id=call_id,
                    name="math.calculator",
                    arguments={"input": "2"},
                )
            ],
        )
        self.assertEqual(len(outcome.diagnostics), 1)
        self.assertEqual(outcome.diagnostics[0].id, diagnostic_id)
        self.assertEqual(
            outcome.diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(
            outcome.diagnostics[0].stage,
            ToolCallDiagnosticStage.PARSE,
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
            (
                ToolFormat.JSON,
                '{"tool": "math..calculator", "arguments": {}}',
            ),
            (
                ToolFormat.OPENAI,
                '{"name": "math/calculator", "arguments": {}}',
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

    async def test_get_calls_returns_none_for_parser_resource_diagnostics(
        self,
    ):
        cases = (
            (
                ToolManagerSettings(
                    tool_format=ToolFormat.JSON,
                    maximum_parser_input_size=4,
                ),
                '{"tool": "calculator", "arguments": {}}',
            ),
            (
                ToolManagerSettings(
                    tool_format=ToolFormat.JSON,
                    maximum_parser_payload_depth=1,
                ),
                (
                    '{"tool": "calculator", "arguments": '
                    '{"payload": {"value": 1}}}'
                ),
            ),
            (
                ToolManagerSettings(
                    tool_format=ToolFormat.JSON,
                    maximum_parser_payload_size=8,
                ),
                '{"tool": "calculator", "arguments": {"payload": "large"}}',
            ),
        )

        for settings, text in cases:
            with self.subTest(settings=settings):
                manager = ToolManager.create_instance(
                    enable_tools=["calculator"],
                    available_toolsets=[ToolSet(tools=[CalculatorTool()])],
                    settings=settings,
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
        assert isinstance(result, ToolCallError)
        self.assertEqual(result.error, {"type": "ValueError"})
        self.assertEqual(result.error_type, "ValueError")
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

    def test_tool_call_status_proxies_final_parser_status(self):
        self.assertIs(
            self.manager.tool_call_status("<tool_call>", final=True),
            ToolCallParser.ToolCallBufferStatus.UNTERMINATED,
        )

    def test_parse_calls_returns_parser_outcome(self):
        outcome = self.manager.parse_calls(
            '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>'
        )

        self.assertEqual(len(outcome.calls), 1)
        self.assertEqual(outcome.calls[0].name, "calculator")

    def test_stream_buffer_diagnostics_proxies_parser(self):
        diagnostics = self.manager.stream_buffer_diagnostics("<tool_call>")

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
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

    async def test_legacy_call_rechecks_repetition_after_filters(self):
        def rewrite_to_adder(call: ToolCall, context: ToolCallContext):
            return (
                ToolCall(
                    id=call.id,
                    name="adder",
                    arguments={"a": 2, "b": 3},
                ),
                context,
            )

        manager = ToolManager.create_instance(
            enable_tools=["adder", "multiplier"],
            available_toolsets=[
                ToolSet(tools=[DummyAdder(), DummyMultiplier()])
            ],
            settings=ToolManagerSettings(
                avoid_repetition=True,
                filters=[rewrite_to_adder],
            ),
        )
        previous = ToolCall(
            id="call-0",
            name="adder",
            arguments={"a": 2, "b": 3},
        )
        call = ToolCall(
            id="call-1",
            name="multiplier",
            arguments={"a": 2, "b": 3},
        )

        result = await manager(call, context=ToolCallContext(calls=[previous]))

        self.assertIsNone(result)

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

    async def test_transformer_none_keeps_current_result(self):
        def transform(
            _: ToolCall,
            __: ToolCallContext,
            ___: str | None,
        ) -> None:
            return None

        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[CalculatorTool()])],
            settings=ToolManagerSettings(transformers=[transform]),
        )

        call = ToolCall(
            id=_uuid4(), name="calculator", arguments={"expression": "1 + 1"}
        )
        result = await manager(call, context=ToolCallContext())

        self.assertEqual(result.result, "2")

    async def test_transformer_result_can_set_null_result(self):
        def transform(
            _: ToolCall,
            __: ToolCallContext,
            ___: str | None,
        ) -> ToolTransformerResult:
            return ToolTransformerResult(result=None)

        manager = ToolManager.create_instance(
            enable_tools=["calculator"],
            available_toolsets=[ToolSet(tools=[CalculatorTool()])],
            settings=ToolManagerSettings(transformers=[transform]),
        )

        call = ToolCall(
            id=_uuid4(), name="calculator", arguments={"expression": "1 + 1"}
        )
        result = await manager(call, context=ToolCallContext())

        self.assertIsNone(result.result)

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

    async def test_filter_namespace_uses_segment_boundaries(self):
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
            enable_tools=["math.calculator", "mathx.calculator"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[CalculatorTool()]),
                ToolSet(namespace="mathx", tools=[CalculatorTool()]),
            ],
            settings=ToolManagerSettings(
                filters=[ToolFilter(func=modify, namespace="math")]
            ),
        )
        call_math = ToolCall(
            id=_uuid4(), name="math.calculator", arguments={"expression": "1"}
        )
        call_mathx = ToolCall(
            id=_uuid4(), name="mathx.calculator", arguments={"expression": "1"}
        )

        with patch(
            "avalan.tool.manager.uuid4",
            side_effect=[_uuid4(), _uuid4()],
        ):
            res_math = await manager(call_math, context=ToolCallContext())
            res_mathx = await manager(call_mathx, context=ToolCallContext())

        self.assertEqual(res_math.call.arguments, {"expression": "2 + 2"})
        self.assertEqual(res_math.result, "4")
        self.assertEqual(res_mathx.call.arguments, {"expression": "1"})
        self.assertEqual(res_mathx.result, "1")

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

    async def test_transformer_namespace_uses_segment_boundaries(self):
        def transform(_: ToolCall, __: ToolCallContext, result: str | None):
            return f"{result}!"

        manager = ToolManager.create_instance(
            enable_tools=["math.calculator", "mathx.calculator"],
            available_toolsets=[
                ToolSet(namespace="math", tools=[CalculatorTool()]),
                ToolSet(namespace="mathx", tools=[CalculatorTool()]),
            ],
            settings=ToolManagerSettings(
                transformers=[
                    ToolTransformer(func=transform, namespace="math")
                ]
            ),
        )
        call_math = ToolCall(
            id=_uuid4(), name="math.calculator", arguments={"expression": "1"}
        )
        call_mathx = ToolCall(
            id=_uuid4(), name="mathx.calculator", arguments={"expression": "1"}
        )

        with patch(
            "avalan.tool.manager.uuid4",
            side_effect=[_uuid4(), _uuid4()],
        ):
            res_math = await manager(call_math, context=ToolCallContext())
            res_mathx = await manager(call_mathx, context=ToolCallContext())

        self.assertEqual(res_math.result, "1!")
        self.assertEqual(res_mathx.result, "1")


if __name__ == "__main__":
    main()
