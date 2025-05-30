from sympy import sympify


class CalculatorTool:
    """Safely evaluate arithmetic expressions using sympy."""

    def __init__(self) -> None:
        self.__name__ = "calculator"

    def __call__(self, expression: str) -> str:
        """Return the result of the expression as a string."""
        result = sympify(expression, evaluate=True)
        return str(result)


tool = CalculatorTool()


async def calculator(expression: str) -> str:
    """
    Calculate the result of the arithmetic expression.

    Args:
        expression: Expression to calculate.

    Returns:
        Result of the calculated expression
    """
    return tool(expression)
