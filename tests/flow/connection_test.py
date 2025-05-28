from avalan.flow.connection import Connection
from avalan.flow.node import Node
from unittest import TestCase


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
