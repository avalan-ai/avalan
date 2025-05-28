from avalan.tool.calculator import CalculatorTool
from pytest import raises
from sympy.core.sympify import SympifyError
from unittest import TestCase, main


class CalculatorToolTestCase(TestCase):
    def setUp(self):
        self.calc = CalculatorTool()

    def test_addition(self):
        self.assertEqual(self.calc("1 + 1"), "2")

    def test_parentheses(self):
        self.assertEqual(self.calc("2*(3+4)"), "14")

    def test_invalid(self):
        with raises(SympifyError) as exc:
            self.calc("2**")

        assert (
            "Sympify of expression 'could not parse '2**'' failed, "
            + "because of exception being raised"
            in str(exc.value)
        )


if __name__ == "__main__":
    main()
