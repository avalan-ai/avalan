from avalan.flow.flow import Flow
from avalan.flow.node import Node
from unittest import TestCase, main


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
