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

if __name__ == "__main__":
    main()
