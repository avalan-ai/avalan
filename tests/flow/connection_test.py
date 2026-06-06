from asyncio import sleep
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.flow.connection import Connection
from avalan.flow.node import Node


class ConnectionTestCase(TestCase):
    def test_sync_helpers_are_not_exposed(self) -> None:
        conn = Connection(Node("A"), Node("B"))
        self.assertFalse(hasattr(conn, "check_conditions"))
        self.assertFalse(hasattr(conn, "apply_filters"))

    def test_repr(self) -> None:
        conn = Connection(Node("A"), Node("B"))

        self.assertEqual(repr(conn), "<Conn A->B>")


class ConnectionAsyncTestCase(IsolatedAsyncioTestCase):
    async def test_check_conditions_async_with_sync_callbacks(self) -> None:
        calls = []

        def c1(x):
            calls.append("c1")
            return True

        def c2(x):
            calls.append("c2")
            return False

        conn = Connection(Node("A"), Node("B"), conditions=[c1, c2])

        self.assertFalse(await conn.check_conditions_async(1))
        self.assertEqual(calls, ["c1", "c2"])

    async def test_apply_filters_async_with_sync_callbacks(self) -> None:
        def f1(x):
            return x + 1

        def f2(x):
            return x * 2

        conn = Connection(Node("A"), Node("B"), filters=[f1, f2])
        self.assertEqual(await conn.apply_filters_async(1), 4)

    async def test_check_conditions_async(self) -> None:
        calls: list[str] = []

        async def c1(x: object) -> bool:
            await sleep(0)
            calls.append("c1")
            return True

        async def c2(x: object) -> bool:
            await sleep(0)
            calls.append("c2")
            return False

        conn = Connection(Node("A"), Node("B"), conditions=[c1, c2])

        self.assertFalse(await conn.check_conditions_async(1))
        self.assertEqual(calls, ["c1", "c2"])

    async def test_apply_filters_async(self) -> None:
        async def f1(x: object) -> int:
            await sleep(0)
            assert isinstance(x, int)
            return x + 1

        async def f2(x: object) -> int:
            await sleep(0)
            assert isinstance(x, int)
            return x * 2

        conn = Connection(Node("A"), Node("B"), filters=[f1, f2])

        self.assertEqual(await conn.apply_filters_async(1), 4)
