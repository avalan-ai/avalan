from contextlib import AsyncExitStack
from importlib import import_module

from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext


class CodeTool(Tool):
    """Execute Python code in a restricted environment.

    Args:
        code: Python code to execute.

    Returns:
        Value of a variable named ``result`` defined in the executed code,
        or an empty string if not present.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "code"

    async def __call__(self, code: str, *, context: ToolCallContext) -> str:
        compile_restricted = import_module(
            "RestrictedPython"
        ).compile_restricted
        safe_builtins = import_module("RestrictedPython.Guards").safe_builtins
        compiled = compile_restricted(code, "<tool>", "exec")
        globals_dict: dict[str, object] = {"__builtins__": safe_builtins}
        locals_dict: dict[str, object] = {}
        exec(compiled, globals_dict, locals_dict)
        return str(locals_dict.get("result", ""))


class CodeToolSet(ToolSet):
    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ) -> None:
        tools = [CodeTool()]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )
