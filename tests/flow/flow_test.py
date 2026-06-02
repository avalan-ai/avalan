from asyncio import CancelledError, sleep, wait_for
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.flow.flow import Flow
from avalan.flow.node import Node


class FlowParseMermaidTestCase(TestCase):
    def test_parse_mermaid(self):
        mermaid = """
        graph LR
        A[Start] -- go --> B{Decision}
        B --> C(((End)))
        """
        flow = Flow()
        flow.parse_mermaid(mermaid)

        self.assertIn("A", flow.nodes)
        self.assertEqual(flow.nodes["A"].label, "Start")
        self.assertEqual(flow.nodes["A"].shape, "rect")

        self.assertIn("B", flow.nodes)
        self.assertEqual(flow.nodes["B"].label, "Decision")
        self.assertEqual(flow.nodes["B"].shape, "diamond")

        self.assertIn("C", flow.nodes)
        self.assertEqual(flow.nodes["C"].label, "End")
        self.assertEqual(flow.nodes["C"].shape, "circle")

        self.assertEqual(len(flow.connections), 2)
        self.assertEqual(flow.connections[0].src.name, "A")
        self.assertEqual(flow.connections[0].dest.name, "B")
        self.assertEqual(flow.connections[0].label, "go")
        self.assertEqual(flow.connections[1].src.name, "B")
        self.assertEqual(flow.connections[1].dest.name, "C")

    def test_parse_mermaid_roundrect(self):
        mermaid = """
        graph LR
        A(Start) --> B[End]
        """
        flow = Flow()
        flow.parse_mermaid(mermaid)

        self.assertIn("A", flow.nodes)
        self.assertEqual(flow.nodes["A"].label, "Start")
        self.assertEqual(flow.nodes["A"].shape, "roundrect")

        self.assertIn("B", flow.nodes)
        self.assertEqual(flow.nodes["B"].label, "End")
        self.assertEqual(flow.nodes["B"].shape, "rect")

        self.assertEqual(len(flow.connections), 1)
        self.assertEqual(flow.connections[0].src.name, "A")
        self.assertEqual(flow.connections[0].dest.name, "B")

    def test_parse_mermaid_pipe_label(self):
        mermaid = """
        graph LR
        A -->|edge| B
        """
        flow = Flow()
        flow.parse_mermaid(mermaid)

        self.assertEqual(len(flow.connections), 1)
        self.assertEqual(flow.connections[0].label, "edge")
        self.assertIn("A", flow.nodes)
        self.assertIn("B", flow.nodes)


class FlowExecutionTestCase(TestCase):
    def test_manual_execution(self):
        executed = []

        def start(_):
            executed.append("A")
            return 1

        def plus_one(inputs):
            executed.append("B")
            val = list(inputs.values())[0]
            return val + 1

        def times_two(inputs):
            executed.append("C")
            val = list(inputs.values())[0]
            return val * 2

        flow = Flow()
        flow.add_node(Node("A", func=start))
        flow.add_node(Node("B", func=plus_one))
        flow.add_node(Node("C", func=times_two))
        flow.add_connection("A", "B")
        flow.add_connection("B", "C")

        result = flow.execute()

        self.assertEqual(result, 4)
        self.assertEqual(executed, ["A", "B", "C"])

    def test_skip_node_without_inputs(self):
        executed = []

        def start(_):
            executed.append("A")
            return "ignored"

        def should_not_run(_):
            executed.append("B")
            raise AssertionError("Callback should not run")

        flow = Flow()
        flow.add_node(Node("A", func=start))
        flow.add_node(Node("B", func=should_not_run))
        flow.add_connection("A", "B", conditions=[lambda _: False])

        result = flow.execute()

        self.assertIsNone(result)
        self.assertEqual(executed, ["A"])

    def test_execute_with_initial_node(self):
        executed = []

        def start(_):
            executed.append("A")
            return 1

        def plus_one(inputs):
            executed.append("B")
            val = list(inputs.values())[0]
            return val + 1

        def times_two(inputs):
            executed.append("C")
            val = list(inputs.values())[0]
            return val * 2

        flow = Flow()
        flow.add_node(Node("A", func=start))
        flow.add_node(Node("B", func=plus_one))
        flow.add_node(Node("C", func=times_two))
        flow.add_connection("A", "B")
        flow.add_connection("B", "C")

        result = flow.execute(initial_node="B", initial_data=5)

        self.assertEqual(result, 12)
        self.assertEqual(executed, ["B", "C"])

    def test_execute_raises_when_no_start_nodes(self):
        flow = Flow()
        flow.add_node(Node("A"))
        flow.add_node(Node("B"))
        flow.add_connection("A", "B")
        flow.add_connection("B", "A")
        with self.assertRaises(ValueError) as context:
            flow.execute()
        self.assertIn("cycle", str(context.exception))

    def test_execute_detects_cycle_with_initial_node(self):
        flow = Flow()
        flow.add_node(Node("A"))
        flow.add_node(Node("B"))
        flow.add_connection("A", "B")
        flow.add_connection("B", "A")
        with self.assertRaises(ValueError) as context:
            flow.execute(initial_node="A", initial_data=1)
        self.assertIn("cycle", str(context.exception))


class FlowAsyncExecutionTestCase(IsolatedAsyncioTestCase):
    async def test_execute_async_runs_async_nodes_edges_and_filters(
        self,
    ) -> None:
        executed: list[str] = []

        async def start(_: dict[str, object]) -> int:
            await sleep(0)
            executed.append("A")
            return 1

        async def should_forward(value: object) -> bool:
            await sleep(0)
            return value == 1

        async def plus_two(value: object) -> int:
            await sleep(0)
            assert isinstance(value, int)
            return value + 2

        async def finish(inputs: dict[str, int]) -> int:
            await sleep(0)
            executed.append("B")
            return inputs["A"] * 3

        flow = Flow()
        flow.add_node(Node("A", func=start))
        flow.add_node(Node("B", func=finish))
        flow.add_connection(
            "A",
            "B",
            conditions=[should_forward],
            filters=[plus_two],
        )

        result = await flow.execute_async()

        self.assertEqual(result, 9)
        self.assertEqual(executed, ["A", "B"])

    async def test_execute_async_checks_cancellation_between_nodes(
        self,
    ) -> None:
        executed: list[str] = []

        def start(_: dict[str, object]) -> int:
            executed.append("A")
            return 1

        def should_not_run(_: dict[str, object]) -> int:
            executed.append("B")
            return 2

        async def cancel_after_first_node() -> None:
            if executed == ["A"]:
                raise CancelledError()

        flow = Flow()
        flow.add_node(Node("A", func=start))
        flow.add_node(Node("B", func=should_not_run))
        flow.add_connection("A", "B")

        with self.assertRaises(CancelledError):
            await flow.execute_async(
                cancellation_checker=cancel_after_first_node
            )

        self.assertEqual(executed, ["A"])

    async def test_execute_async_timeout_covers_node_work(self) -> None:
        async def slow(_: dict[str, object]) -> str:
            await sleep(0.05)
            return "done"

        flow = Flow()
        flow.add_node(Node("A", func=slow))

        with self.assertRaises(TimeoutError):
            await wait_for(flow.execute_async(), timeout=0.001)

    async def test_execute_async_skip_node_without_inputs(self) -> None:
        executed: list[str] = []

        def start(_: dict[str, object]) -> str:
            executed.append("A")
            return "ignored"

        def should_not_run(_: dict[str, object]) -> str:
            executed.append("B")
            return "unexpected"

        flow = Flow()
        flow.add_node(Node("A", func=start))
        flow.add_node(Node("B", func=should_not_run))
        flow.add_connection("A", "B", conditions=[lambda _: False])

        result = await flow.execute_async()

        self.assertIsNone(result)
        self.assertEqual(executed, ["A"])

    async def test_execute_async_missing_input_and_multi_terminal(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: 2))
        flow.add_node(Node("B"))
        flow.add_node(Node("C"))
        flow.add_connection("A", "B")
        flow.add_connection("A", "C")

        result = await flow.execute_async()

        self.assertEqual(result, {"B": 2, "C": 2})

    async def test_execute_async_raises_when_no_start_nodes(self) -> None:
        flow = Flow()
        flow.add_node(Node("A"))
        flow.add_node(Node("B"))
        flow.add_connection("A", "B")
        flow.add_connection("B", "A")

        with self.assertRaises(ValueError) as context:
            await flow.execute_async()

        self.assertIn("cycle", str(context.exception))

    async def test_execute_async_detects_reachable_cycle(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: 1))
        flow.add_node(Node("B", func=lambda _: 2))
        flow.add_node(Node("C", func=lambda _: 3))
        flow.add_connection("A", "B")
        flow.add_connection("B", "C")
        flow.add_connection("C", "B")

        with self.assertRaises(ValueError) as context:
            await flow.execute_async()

        self.assertIn("cycle", str(context.exception))

    async def test_execute_async_detects_cycle_with_initial_node(self) -> None:
        flow = Flow()
        flow.add_node(Node("A"))
        flow.add_node(Node("B"))
        flow.add_connection("A", "B")
        flow.add_connection("B", "A")

        with self.assertRaises(ValueError) as context:
            await flow.execute_async(initial_node="A", initial_data=1)

        self.assertIn("cycle", str(context.exception))


class FlowAddConnectionTestCase(TestCase):
    def test_add_connection_unknown_src(self):
        flow = Flow()
        flow.add_node(Node("A"))
        with self.assertRaises(KeyError):
            flow.add_connection("B", "A")

    def test_add_connection_unknown_dest(self):
        flow = Flow()
        flow.add_node(Node("A"))
        with self.assertRaises(KeyError):
            flow.add_connection("A", "B")


if __name__ == "__main__":
    main()
