from ..entities import ToolCall

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from io import UnsupportedOperation
from json import dumps
from select import select
from sys import stdin

from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.text import Text


@contextmanager
def _pause_live(live: Live) -> Iterator[None]:
    """Temporarily disable ``live`` auto refresh."""
    auto_refresh = live.auto_refresh
    live.auto_refresh = False
    live.refresh()
    try:
        yield
    finally:
        live.auto_refresh = auto_refresh
        live.refresh()


class PromptWithoutPrefix(Prompt):
    prompt_suffix = ""


class CommandAbortException(BaseException):
    pass


def confirm(console: Console, prompt: str) -> bool:
    return Confirm.ask(prompt)


def confirm_tool_call(
    console: Console,
    call: ToolCall,
    *,
    tty_path: str = "/dev/tty",
    live: Live | None = None,
) -> str:
    prompt = "Execute tool call?"
    options = {
        "choices": ["y", "a", "n"],
        "default": "n",
        "console": console,
        "show_choices": True,
        "show_default": True,
    }

    with _pause_live(live) if live else nullcontext():
        call_element = Syntax(
            dumps({"name": call.name, "arguments": call.arguments}, indent=2),
            "json",
        )
        console.print(call_element)
        if live:
            prompt_element = Text.from_markup(prompt, style="prompt")
            prompt_element.end = ""
            choices = "/".join(options["choices"])
            prompt_element.append(" ")
            prompt_element.append(f"[{choices}]", "prompt.choices")

            if options["show_default"]:
                default = Text(f"({options['default']})", "prompt.default")
                prompt_element.append(" ")
                prompt_element.append(default)

            console.print(prompt_element)

        stdin_is_tty = stdin.isatty()
        with open(tty_path) if not stdin_is_tty else nullcontext() as tty:
            if not stdin_is_tty:
                options["stream"] = tty
            return Prompt.ask(prompt, **options)


def has_input(console: Console) -> bool:
    try:
        stdin_ready, __, __ = select([stdin], [], [], 0.0)
        return bool(stdin_ready)
    except UnsupportedOperation:
        return False


def get_input(
    console: Console,
    prompt: str | None,
    *,
    echo_stdin: bool = True,
    force_prompt: bool = False,
    is_quiet: bool = False,
    prefix_line: bool = True,
    strip_prompt: bool = True,
    strip_stdin: bool = True,
    suffix_line: bool = True,
    tty_path: str = "/dev/tty",
) -> str | None:
    full_prompt = f"{prompt} "
    is_input_available = has_input(console)
    if not force_prompt and is_input_available:
        input_string = stdin.read()
        if strip_stdin:
            input_string = input_string.strip()
        if prompt and not is_quiet and echo_stdin:
            console.print(
                Padding(f"{full_prompt}{input_string}", pad=(1, 0, 1, 0))
            )
    elif prompt:
        if prefix_line and not is_quiet:
            console.print("")
        try:
            with (
                open(tty_path)
                if is_input_available and force_prompt
                else nullcontext()
            ) as tty:
                kwargs = {}
                if is_input_available and force_prompt:
                    kwargs["stream"] = tty
                input_string = PromptWithoutPrefix.ask(full_prompt, **kwargs)
            if strip_prompt:
                input_string = input_string.strip()
        except EOFError:
            raise CommandAbortException()
        if suffix_line and not is_quiet:
            console.print("")
    else:
        input_string = None
    return input_string
