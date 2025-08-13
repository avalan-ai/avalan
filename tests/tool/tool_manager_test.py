from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolManagerSettings,
    ToolFilter,
    ToolTransformer,
)
from avalan.tool import ToolSet
from avalan.tool.math import CalculatorTool
from avalan.tool.manager import ToolManager
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, patch
from uuid import uuid4 as _uuid4


class ToolManagerCreationTestCase(TestCase):
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


class DummyAdder:
    def __init__(self) -> None:
        self.__name__ = "adder"

    async def __call__(self, a: int, b: int) -> int:
        """Return the sum of ``a`` and ``b``."""
        return a + b


class DummyAdderAlt(DummyAdder):
    def __init__(self) -> None:
        self.__name__ = "adder_alt"


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
