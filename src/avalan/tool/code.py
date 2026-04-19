from ..compat import override
from ..entities import ToolCallContext
from . import Tool, ToolSet

from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
from contextlib import AsyncExitStack
from typing import Any

try:
    from RestrictedPython import (
        RestrictingNodeTransformer,
        compile_restricted,
        safe_globals,
    )

    HAS_CODE_DEPENDENCIES = True
except ImportError:
    HAS_CODE_DEPENDENCIES = False
    RestrictingNodeTransformer = None
    compile_restricted = None
    safe_globals = None


class CodeTool(Tool):
    """Execute Python code in a restricted environment.

    Args:
        code: Python source that defines a callable named `run`.
        args: Positional arguments forwarded to `run`.
        kwargs: Keyword arguments forwarded to `run`.

    Returns:
        Text representation of the value returned by `run`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = "run"

    async def __call__(
        self,
        code: str,
        *args: Any,
        context: ToolCallContext,
        **kwargs: Any,
    ) -> str:
        _ = context
        locals_dict: dict[str, Any] = {}
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
        positional_args: tuple[Any, ...] = args
        keyword_args: dict[str, Any] = kwargs

        if (
            positional_args
            and not keyword_args
            and len(positional_args) == 2
            and isinstance(positional_args[1], dict)
        ):
            unpacked_args, unpacked_kwargs = positional_args
            if isinstance(unpacked_args, tuple):
                positional_args = unpacked_args
            elif isinstance(unpacked_args, dict):
                positional_args = ()
                unpacked_kwargs = unpacked_args
            keyword_args = unpacked_kwargs

        result = (
            function(*positional_args, **keyword_args)
            if positional_args and keyword_args
            else (
                function(*positional_args)
                if positional_args
                else function(**keyword_args) if keyword_args else function()
            )
        )

        return str(result)


class AstGrepTool(Tool):
    """Search or rewrite code using the ast-grep CLI.

    Args:
        pattern: Code pattern to search for.
        lang: Programming language of the files.
        rewrite: Template used to rewrite matches.
        paths: Files or directories to search.

    Returns:
        Output produced by ast-grep.
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

        process = await create_subprocess_exec(
            *args,
            stdout=PIPE,
            stderr=PIPE,
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
