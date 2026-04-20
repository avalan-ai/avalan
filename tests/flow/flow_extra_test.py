import unittest

from avalan.flow.connection import Connection
from avalan.flow.flow import Flow
from avalan.flow.node import Node


class FlowExtraTestCase(unittest.TestCase):
    def test_parse_mermaid_ignores_invalid_lines(self) -> None:
        mermaid = """graph LR
A --> B
invalid
"""
        flow = Flow()
        flow.parse_mermaid(mermaid)
        self.assertEqual(len(flow.connections), 1)
        self.assertIn("A", flow.nodes)
        self.assertIn("B", flow.nodes)

    def test_parse_mermaid_updates_existing_node(self) -> None:
        flow = Flow()
        flow.add_node(Node("A"))
        flow.add_node(Node("B"))
        flow.parse_mermaid("A[Start] --> B")
        self.assertEqual(flow.nodes["A"].label, "Start")
        self.assertEqual(flow.nodes["A"].shape, "rect")

    def test_parse_node_branches(self) -> None:
        flow = Flow()
        self.assertEqual(flow._parse_node("?bad"), ("?bad", None, None))
        self.assertEqual(
            flow._parse_node("A some label"), ("A", "some label", None)
        )

    def test_execute_missing_input_and_multi_terminal(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: 2))
        flow.add_node(Node("B"))
        flow.add_connection("A", "B")
        # Start execution from node B without initial data
        result_none = flow.execute(initial_node="B")
        self.assertIsNone(result_none)

        # Add a second terminal node and run the full graph
        flow.add_node(Node("C"))
        flow.add_connection("A", "C")
        result = flow.execute()
        self.assertEqual(result, {"B": 2, "C": 2})

    def test_execute_detects_reachable_cycle(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda _: 1))
        flow.add_node(Node("B", func=lambda _: 2))
        flow.add_node(Node("C", func=lambda _: 3))
        flow.add_connection("A", "B")
        flow.add_connection("B", "C")
        flow.add_connection("C", "B")

        with self.assertRaises(ValueError) as context:
            flow.execute()

        self.assertIn("cycle", str(context.exception))

    def test_resolve_start_nodes_with_unknown_node_inputs(self) -> None:
        flow = Flow()
        flow.add_node(Node("A"))

        with self.assertRaises(KeyError):
            flow.execute(initial_node=Node("B"))

        with self.assertRaises(KeyError):
            flow.execute(initial_node="B")

    def test_execute_with_existing_node_instance(self) -> None:
        flow = Flow()
        flow.add_node(Node("A", func=lambda inputs: inputs.get("__init__", 0)))

        result = flow.execute(initial_node=Node("A"), initial_data=7)

        self.assertEqual(result, 7)

    def test_detect_cycle_nodes_handles_previously_visited_branch(
        self,
    ) -> None:
        flow = Flow()
        flow.add_node(Node("A"))
        flow.add_node(Node("B"))
        flow.add_node(Node("C"))
        flow.add_node(Node("D"))
        flow.add_connection("A", "B")
        flow.add_connection("A", "C")
        flow.add_connection("B", "D")
        flow.add_connection("C", "D")

        cycle_nodes = flow._detect_cycle_nodes([flow.nodes["A"]])

        self.assertEqual(cycle_nodes, set())


class ConnectionReprTestCase(unittest.TestCase):
    def test_repr(self) -> None:
        conn = Connection(Node("A"), Node("B"))
        self.assertEqual(repr(conn), "<Conn A->B>")


if __name__ == "__main__":
    unittest.main()
