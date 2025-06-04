from . import Tool
from sympy import sympify


class CalculatorTool(Tool):
    """
    Calculate the result of the arithmetic expression.

    Args:
        expression: Expression to calculate.

    Returns:
        Result of the calculated expression
    """

    def __init__(self) -> None:
        self.__name__ = "calculator"

    async def __call__(self, expression: str) -> str:
        result = sympify(expression, evaluate=True)
        return str(result)
