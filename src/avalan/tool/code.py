from . import Tool, ToolSet
from ..compat import override
from ..entities import ToolCallContext
import asyncio
from contextlib import AsyncExitStack
from RestrictedPython import (
    compile_restricted,
    RestrictingNodeTransformer,
    safe_globals,
)


class CodeTool(Tool):
    """Execute Python function code in a restricted environment.

    Args:
        code: Python code to execute, which should define a function named
        `run`, with whatever arguments needed, and returning its result on
        a string.
        args: Positional arguments to send the `run` function.
        kwargs: Keyword arguments to send the `run` function.

    Returns:
        A string, which is the value returned by the `run` function.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "run"

    async def __call__(
        self, code: str, *args: any, context: ToolCallContext, **kwargs: any
    ) -> str:
        locals_dict = {}
        byte_code = compile_restricted(
            code,
            filename="<avalan:tool:code>",
            mode="exec",
            flags=0,
            dont_inherit=False,
            policy=RestrictingNodeTransformer,
        )
        exec(byte_code, safe_globals, locals_dict)
        assert "run" in locals_dict

        function = locals_dict["run"]

        if args and not kwargs and isinstance(args, tuple) and len(args) == 2:
            (args, kwargs) = args
            if args and not kwargs and isinstance(args, dict):
                kwargs = args
                args = None

        result = (
            function(*args, **kwargs)
            if args and kwargs
            else (
                function(*args)
                if args
                else function(**kwargs) if kwargs else function()
            )
        )

        return str(result)


class AstGrepTool(Tool):
    """Search or rewrite code using the ast-grep CLI.

    Args:
        pattern: Code pattern to search for.
        context: Tool call context.
        lang: Programming language of the files.
        rewrite: Template used to rewrite matches.
        paths: Files or directories to search. Defaults to the current
            directory.

    Returns:
        Output from ast-grep.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "search.ast.grep"

    async def __call__(
        self,
        pattern: str,
        *,
        context: ToolCallContext,
        lang: str,
        rewrite: str | None = None,
        paths: list[str] | None = None,
    ) -> str:
        assert pattern
        assert lang

        args = ["ast-grep", "--pattern", pattern, "--lang", lang]
        if rewrite is not None:
            args.extend(["--rewrite", rewrite])
        if paths:
            args.extend(paths)

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.decode() or stdout.decode())
        return stdout.decode()


class CodeToolSet(ToolSet):
    @override
    def __init__(
        self,
        *,
        exit_stack: AsyncExitStack | None = None,
        namespace: str | None = None,
    ) -> None:
        tools = [CodeTool(), AstGrepTool()]
        super().__init__(
            exit_stack=exit_stack, namespace=namespace, tools=tools
        )
