from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext
from contextlib import AsyncExitStack
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
        super().__init__()
        self.__name__ = "calculator"

    async def __call__(self, expression: str, context: ToolCallContext) -> str:
        result = sympify(expression, evaluate=True)
        return str(result)


class MathToolSet(ToolSet):
    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ):
        tools = [CalculatorTool()]
        return super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )
