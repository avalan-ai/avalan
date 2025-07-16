import unittest
from unittest.mock import AsyncMock, MagicMock

from avalan.flow.flow import Flow
from avalan.flow.node import Node
from avalan.flow.manager import FlowManager
from avalan.event import EventType
from avalan.agent.loader import OrchestratorLoader


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


if __name__ == "__main__":
    unittest.main()
