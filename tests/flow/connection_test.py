from asyncio import sleep
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.flow.connection import Connection
from avalan.flow.node import Node


class ConnectionTestCase(TestCase):
    def test_check_conditions(self):
        calls = []

        def c1(x):
            calls.append("c1")
            return True

        def c2(x):
            calls.append("c2")
            return False

        conn = Connection(Node("A"), Node("B"), conditions=[c1, c2])
        self.assertFalse(conn.check_conditions(1))
        self.assertEqual(calls, ["c1", "c2"])

    def test_apply_filters(self):
        def f1(x):
            return x + 1

        def f2(x):
            return x * 2

        conn = Connection(Node("A"), Node("B"), filters=[f1, f2])
        self.assertEqual(conn.apply_filters(1), 4)


class ConnectionAsyncTestCase(IsolatedAsyncioTestCase):
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
