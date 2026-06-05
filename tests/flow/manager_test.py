import unittest
from asyncio import CancelledError, sleep
from unittest.mock import AsyncMock, MagicMock

from avalan.agent.loader import OrchestratorLoader
from avalan.event import EventType
from avalan.flow.flow import Flow
from avalan.flow.manager import FlowManager
from avalan.flow.node import Node


class FlowManagerCallTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_call_triggers_events_without_raw_result(self) -> None:
        class PrivateResult:
            def __repr__(self) -> str:
                return "private flow result"

        flow = Flow()
        flow.add_node(Node("A", func=lambda _: PrivateResult()))
        loader = MagicMock(spec=OrchestratorLoader)
        loader.event_manager = MagicMock()
        loader.event_manager.trigger = AsyncMock()
        manager = FlowManager(loader, logger=MagicMock())

        result = await manager(flow)

        self.assertEqual(repr(result), "private flow result")
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
        self.assertEqual(before.payload["status"], "started")
        self.assertEqual(before.payload["name"], "flow")
        self.assertEqual(after.payload["status"], "succeeded")
        self.assertEqual(after.payload["name"], "flow")
        self.assertNotIn("flow", before.payload)
        self.assertNotIn("flow", after.payload)
        self.assertNotIn("result", after.payload)
        self.assertNotIn("private flow result", str(after.payload))
        self.assertIsNone(before.finished)
        self.assertIsNone(before.elapsed)
        self.assertEqual(after.started, before.started)
        self.assertIsNotNone(after.finished)
        self.assertIsNotNone(after.elapsed)

    async def test_call_triggers_failed_event_on_node_error(self) -> None:
        def fail(_: dict[str, object]) -> str:
            raise RuntimeError("private failure")

        flow = Flow()
        flow.add_node(Node("A", func=fail))
        loader = MagicMock(spec=OrchestratorLoader)
        loader.event_manager = MagicMock()
        loader.event_manager.trigger = AsyncMock()
        manager = FlowManager(loader, logger=MagicMock())

        with self.assertRaisesRegex(RuntimeError, "private failure"):
            await manager(flow)

        calls = loader.event_manager.trigger.await_args_list
        self.assertEqual(len(calls), 2)
        before = calls[0].args[0]
        after = calls[1].args[0]
        self.assertEqual(before.type, EventType.FLOW_MANAGER_CALL_BEFORE)
        self.assertEqual(after.type, EventType.FLOW_MANAGER_CALL_AFTER)
        self.assertEqual(before.payload["status"], "started")
        self.assertEqual(after.payload["status"], "failed")
        self.assertEqual(after.payload["name"], "flow")
        self.assertNotIn("flow", after.payload)
        self.assertNotIn("result", after.payload)
        self.assertNotIn("private failure", str(after.payload))
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

    async def test_call_triggers_failed_event_on_timeout(self) -> None:
        async def slow(_: dict[str, object]) -> str:
            await sleep(0.05)
            return "done"

        flow = Flow()
        flow.add_node(Node("A", func=slow))
        loader = MagicMock(spec=OrchestratorLoader)
        loader.event_manager = MagicMock()
        loader.event_manager.trigger = AsyncMock()
        manager = FlowManager(loader, logger=MagicMock())

        with self.assertRaises(TimeoutError):
            await manager(flow, timeout_seconds=0.001)

        calls = loader.event_manager.trigger.await_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0].args[0].type,
            EventType.FLOW_MANAGER_CALL_BEFORE,
        )
        after = calls[1].args[0]
        self.assertEqual(after.type, EventType.FLOW_MANAGER_CALL_AFTER)
        self.assertEqual(after.payload["status"], "failed")
        self.assertEqual(after.payload["name"], "flow")
        self.assertNotIn("flow", after.payload)
        self.assertNotIn("result", after.payload)
        self.assertNotIn("done", str(after.payload))

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
