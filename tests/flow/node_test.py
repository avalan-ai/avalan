from asyncio import CancelledError, sleep
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.flow.flow import Flow
from avalan.flow.node import Node


class NodeExecuteTestCase(TestCase):
    def test_execute_with_function(self):
        def inc(inputs):
            return inputs["x"] + 1

        node = Node("n", func=inc)
        result = node.execute({"x": 1})
        self.assertEqual(result, 2)

    def test_execute_function_varargs(self):
        def add(a, b):
            return a + b

        node = Node("n", func=add)
        result = node.execute({"a": 1, "b": 2})
        self.assertEqual(result, 3)

    def test_execute_no_function(self):
        node = Node("n")
        self.assertEqual(node.execute({"a": 1}), 1)
        self.assertEqual(node.execute({"a": 1, "b": 2}), {"a": 1, "b": 2})
        self.assertIsNone(node.execute({}))

    def test_schema_validation(self):
        node = Node(
            "n",
            func=lambda inputs: str(inputs["x"]),
            input_schema=int,
            output_schema=str,
        )
        self.assertEqual(node.execute({"x": 2}), "2")
        with self.assertRaises(TypeError):
            node.execute({"x": "a"})
        bad_node = Node("b", func=lambda inputs: 1, output_schema=str)
        with self.assertRaises(TypeError):
            bad_node.execute({"x": 1})

    def test_execute_subgraph(self):
        sub = Flow()
        sub.add_node(Node("A", func=lambda inp: inp["__init__"] * 2))
        sub.add_node(Node("B", func=lambda inp: inp["A"] + 3))
        sub.add_connection("A", "B")
        node = Node("sub", subgraph=sub, output_schema=int)
        result = node.execute({"val": 5})
        self.assertEqual(result, 13)


class NodeExtraTestCase(TestCase):
    def test_execute_output_schema_mismatch_subgraph(self):
        sub = Flow()
        sub.add_node(Node("A", func=lambda inp: 42))
        node = Node("sub", subgraph=sub, output_schema=str)
        with self.assertRaises(TypeError):
            node.execute({"val": 1})

    def test_execute_input_schema_mismatch_multiple_inputs(self):
        node = Node(
            "n", input_schema=int, func=lambda inputs: sum(inputs.values())
        )
        with self.assertRaises(TypeError):
            node.execute({"a": 1, "b": 2})

    def test_repr(self):
        node = Node("repr")
        self.assertEqual(repr(node), "<Node repr>")

    def test_execute_rejects_async_function(self) -> None:
        async def inc(inputs: dict[str, int]) -> int:
            return inputs["x"] + 1

        node = Node("n", func=inc)

        with self.assertRaisesRegex(TypeError, "execute_async"):
            node.execute({"x": 1})


class NodeAsyncTestCase(IsolatedAsyncioTestCase):
    async def test_execute_async_with_function(self) -> None:
        async def inc(inputs: dict[str, int]) -> int:
            await sleep(0)
            return inputs["x"] + 1

        node = Node("n", func=inc, output_schema=int)

        self.assertEqual(await node.execute_async({"x": 1}), 2)

    async def test_execute_async_function_varargs(self) -> None:
        async def add(a: int, b: int) -> int:
            await sleep(0)
            return a + b

        node = Node("n", func=add)

        self.assertEqual(await node.execute_async({"a": 1, "b": 2}), 3)

    async def test_execute_async_subgraph(self) -> None:
        sub = Flow()

        async def double(inp: dict[str, int]) -> int:
            await sleep(0)
            return inp["__init__"] * 2

        sub.add_node(Node("A", func=double))
        sub.add_node(Node("B", func=lambda inp: inp["A"] + 3))
        sub.add_connection("A", "B")
        node = Node("sub", subgraph=sub, output_schema=int)

        self.assertEqual(await node.execute_async({"val": 5}), 13)

    async def test_execute_async_checks_cancellation(self) -> None:
        async def cancelled() -> None:
            raise CancelledError()

        node = Node("n", func=lambda _: 1)

        with self.assertRaises(CancelledError):
            await node.execute_async({}, cancellation_checker=cancelled)


class NodeImportTestCase(TestCase):
    def test_import_with_type_checking(self):
        import importlib
        import sys
        import typing

        module_name = "avalan.flow.node"
        with self.subTest("import with TYPE_CHECKING=True"):
            if module_name in sys.modules:
                del sys.modules[module_name]
            typing.TYPE_CHECKING = True
            try:
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, "Node"))
            finally:
                typing.TYPE_CHECKING = False
