import unittest

from avalan.entities import ParallelStrategy
from avalan.model.engine import Engine


class GetTPPlanTestCase(unittest.TestCase):
    def test_none(self) -> None:
        self.assertIsNone(Engine._get_tp_plan(None))

    def test_string(self) -> None:
        self.assertEqual(
            Engine._get_tp_plan(ParallelStrategy.COLWISE),
            "colwise",
        )

    def test_dict(self) -> None:
        plan = Engine._get_tp_plan({"a": ParallelStrategy.ROWWISE})
        self.assertEqual(plan, {"a": "rowwise"})
