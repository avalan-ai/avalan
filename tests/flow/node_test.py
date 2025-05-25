from avalan.flow.node import Node
from avalan.flow.flow import Flow
from unittest import TestCase

class NodeExecuteTestCase(TestCase):
    def test_execute_with_function(self):
        def inc(inputs):
            return inputs['x'] + 1
        node = Node('n', func=inc)
        result = node.execute({'x': 1})
        self.assertEqual(result, 2)

    def test_execute_function_varargs(self):
        def add(a, b):
            return a + b
        node = Node('n', func=add)
        result = node.execute({'a': 1, 'b': 2})
        self.assertEqual(result, 3)

    def test_execute_no_function(self):
        node = Node('n')
        self.assertEqual(node.execute({'a': 1}), 1)
        self.assertEqual(node.execute({'a': 1, 'b': 2}), {'a': 1, 'b': 2})
        self.assertIsNone(node.execute({}))

    def test_schema_validation(self):
        node = Node('n', func=lambda inputs: str(inputs['x']),
                    input_schema=int, output_schema=str)
        self.assertEqual(node.execute({'x': 2}), '2')
        with self.assertRaises(TypeError):
            node.execute({'x': 'a'})
        bad_node = Node('b', func=lambda inputs: 1, output_schema=str)
        with self.assertRaises(TypeError):
            bad_node.execute({'x': 1})

    def test_execute_subgraph(self):
        sub = Flow()
        sub.add_node(Node('A', func=lambda inp: inp['__init__'] * 2))
        sub.add_node(Node('B', func=lambda inp: inp['A'] + 3))
        sub.add_connection('A', 'B')
        node = Node('sub', subgraph=sub, output_schema=int)
        result = node.execute({'val': 5})
        self.assertEqual(result, 13)

