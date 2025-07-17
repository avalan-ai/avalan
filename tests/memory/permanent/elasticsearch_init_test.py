from avalan.memory.permanent.elasticsearch import to_thread
from unittest import IsolatedAsyncioTestCase


class ToThreadTestCase(IsolatedAsyncioTestCase):
    async def test_to_thread_runs_function(self):
        def add(a: int, b: int) -> int:
            return a + b

        result = await to_thread(add, a=1, b=2)
        self.assertEqual(result, 3)
