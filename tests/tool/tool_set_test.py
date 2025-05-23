from avalan.tool import ToolSet
from avalan.tool.calculator import calculator
from dataclasses import FrozenInstanceError, is_dataclass
from unittest import TestCase, main

class ToolSetTestCase(TestCase):
    def test_dataclass_properties(self):
        self.assertTrue(is_dataclass(ToolSet))
        toolset = ToolSet(tools=[calculator], namespace="math")
        self.assertEqual(toolset.namespace, "math")
        self.assertEqual(list(toolset.tools), [calculator])
        with self.assertRaises(FrozenInstanceError):
            toolset.namespace = "other"

if __name__ == "__main__":
    main()
