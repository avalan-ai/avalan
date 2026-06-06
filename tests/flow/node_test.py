from asyncio import CancelledError, sleep
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.flow.flow import Flow
from avalan.flow.node import Node


class NodeExecuteTestCase(TestCase):
    def test_sync_execute_is_not_exposed(self) -> None:
        self.assertFalse(hasattr(Node("n"), "execute"))

    def test_repr(self):
        node = Node("repr")
        self.assertEqual(repr(node), "<Node repr>")


class NodeAsyncTestCase(IsolatedAsyncioTestCase):
    async def test_execute_async_with_function(self) -> None:
        def inc(inputs):
            return inputs["x"] + 1

        node = Node("n", func=inc)

        self.assertEqual(await node.execute_async({"x": 1}), 2)

    async def test_execute_async_function_varargs(self) -> None:
        def add(a, b):
            return a + b

        node = Node("n", func=add)

        self.assertEqual(await node.execute_async({"a": 1, "b": 2}), 3)

    async def test_execute_async_no_function(self) -> None:
        node = Node("n")
        self.assertEqual(await node.execute_async({"a": 1}), 1)
        self.assertEqual(
            await node.execute_async({"a": 1, "b": 2}),
            {"a": 1, "b": 2},
        )
        self.assertIsNone(await node.execute_async({}))

    async def test_execute_async_sync_none_output_bypasses_output_schema(
        self,
    ) -> None:
        node = Node("n", func=lambda _: None, output_schema=str)

        self.assertIsNone(await node.execute_async({"x": 1}))

    async def test_schema_validation(self) -> None:
        node = Node(
            "n",
            func=lambda inputs: str(inputs["x"]),
            input_schema=int,
            output_schema=str,
        )
        self.assertEqual(await node.execute_async({"x": 2}), "2")
        with self.assertRaises(TypeError):
            await node.execute_async({"x": "a"})
        bad_node = Node("b", func=lambda inputs: 1, output_schema=str)
        with self.assertRaises(TypeError):
            await bad_node.execute_async({"x": 1})

    async def test_execute_async_subgraph(self) -> None:
        sub = Flow()
        sub.add_node(Node("A", func=lambda inp: inp["__init__"] * 2))
        sub.add_node(Node("B", func=lambda inp: inp["A"] + 3))
        sub.add_connection("A", "B")
        node = Node("sub", subgraph=sub, output_schema=int)

        self.assertEqual(await node.execute_async({"val": 5}), 13)

    async def test_execute_async_awaits_function(self) -> None:
        async def inc(inputs: dict[str, int]) -> int:
            await sleep(0)
            return inputs["x"] + 1

        node = Node("n", func=inc, output_schema=int)

        self.assertEqual(await node.execute_async({"x": 1}), 2)

    async def test_execute_async_awaits_varargs_function(self) -> None:
        async def add(a: int, b: int) -> int:
            await sleep(0)
            return a + b

        node = Node("n", func=add)

        self.assertEqual(await node.execute_async({"a": 1, "b": 2}), 3)

    async def test_execute_async_output_schema_mismatch_subgraph(self) -> None:
        sub = Flow()
        sub.add_node(Node("A", func=lambda inp: 42))
        node = Node("sub", subgraph=sub, output_schema=str)
        with self.assertRaises(TypeError):
            await node.execute_async({"val": 1})

    async def test_execute_async_input_schema_mismatch_multiple_inputs(
        self,
    ) -> None:
        node = Node(
            "n", input_schema=int, func=lambda inputs: sum(inputs.values())
        )
        with self.assertRaises(TypeError):
            await node.execute_async({"a": 1, "b": 2})

    async def test_execute_async_checks_cancellation(self) -> None:
        async def cancelled() -> None:
            raise CancelledError()

        node = Node("n", func=lambda _: 1)

        with self.assertRaises(CancelledError):
            await node.execute_async({}, cancellation_checker=cancelled)

    async def test_execute_async_none_output_bypasses_output_schema(
        self,
    ) -> None:
        async def no_output(_: dict[str, int]) -> None:
            await sleep(0)

        node = Node("n", func=no_output, output_schema=str)

        self.assertIsNone(await node.execute_async({"x": 1}))


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
