import unittest
from asyncio import CancelledError, sleep
from unittest.mock import AsyncMock, MagicMock

from avalan.agent.loader import OrchestratorLoader
from avalan.event import EventType
from avalan.flow.flow import Flow
from avalan.flow.manager import FlowManager
from avalan.flow.node import Node


class FlowManagerCallTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_call_triggers_events(self):
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: 1))
        loader = MagicMock(spec=OrchestratorLoader)
        loader.event_manager = MagicMock()
        loader.event_manager.trigger = AsyncMock()
        manager = FlowManager(loader, logger=MagicMock())

        result = await manager(flow)

        self.assertEqual(result, 1)
        called_types = [
            c.args[0].type
            for c in loader.event_manager.trigger.await_args_list
        ]
        self.assertIn(EventType.FLOW_MANAGER_CALL_BEFORE, called_types)
        self.assertIn(EventType.FLOW_MANAGER_CALL_AFTER, called_types)
        before = next(
            c.args[0]
            for c in loader.event_manager.trigger.await_args_list
            if c.args[0].type == EventType.FLOW_MANAGER_CALL_BEFORE
        )
        after = next(
            c.args[0]
            for c in loader.event_manager.trigger.await_args_list
            if c.args[0].type == EventType.FLOW_MANAGER_CALL_AFTER
        )
        self.assertIsNone(before.finished)
        self.assertIsNone(before.elapsed)
        self.assertEqual(after.started, before.started)
        self.assertIsNotNone(after.finished)
        self.assertIsNotNone(after.elapsed)

    async def test_call_passes_initial_data_and_timeout(self) -> None:
        async def slow(_: dict[str, object]) -> str:
            await sleep(0.05)
            return "done"

        flow = Flow()
        flow.add_node(Node("A", func=slow))
        loader = MagicMock(spec=OrchestratorLoader)
        loader.event_manager = None
        manager = FlowManager(loader, logger=MagicMock())

        with self.assertRaises(TimeoutError):
            await manager(flow, timeout_seconds=0.001)

    async def test_call_honors_cancellation_checker(self) -> None:
        async def cancelled() -> None:
            raise CancelledError()

        flow = Flow()
        flow.add_node(Node("A", func=lambda _: 1))
        loader = MagicMock(spec=OrchestratorLoader)
        loader.event_manager = None
        manager = FlowManager(loader, logger=MagicMock())

        with self.assertRaises(CancelledError):
            await manager(flow, cancellation_checker=cancelled)


if __name__ == "__main__":
    unittest.main()
