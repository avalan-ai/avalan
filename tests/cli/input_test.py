from unittest import TestCase
from unittest.mock import MagicMock, patch, call

from uuid import uuid4

from rich.syntax import Syntax

from avalan.cli import (
    CommandAbortException,
    confirm,
    confirm_tool_call,
    get_input,
    has_input,
)
from avalan.entities import ToolCall


class CliGetInputTestCase(TestCase):
    def setUp(self):
        self.console = MagicMock()

    def test_stdin_echo(self):
        stdin_mock = MagicMock(read=MagicMock(return_value=" text "))
        with (
            patch("avalan.cli.has_input", return_value=True),
            patch("avalan.cli.stdin", stdin_mock),
        ):
            result = get_input(self.console, "prompt")

        self.assertEqual(result, "text")
        padding = self.console.print.call_args.args[0]
        self.assertEqual(padding.renderable, "prompt text")
        self.assertEqual(
            (padding.top, padding.right, padding.bottom, padding.left),
            (1, 0, 1, 0),
        )

    def test_prompt_when_no_stdin(self):
        with (
            patch("avalan.cli.has_input", return_value=False),
            patch(
                "avalan.cli.PromptWithoutPrefix.ask", return_value=" value "
            ) as ask,
        ):
            result = get_input(self.console, "prompt")

        ask.assert_called_once_with("prompt ")
        self.assertEqual(result, "value")
        self.assertEqual(
            self.console.print.call_args_list, [call(""), call("")]
        )

    def test_force_prompt_uses_tty(self):
        fake_tty = MagicMock()
        ctx = MagicMock()
        ctx.__enter__.return_value = fake_tty
        ctx.__exit__.return_value = False
        with (
            patch("avalan.cli.has_input", return_value=True),
            patch("avalan.cli.open", return_value=ctx) as open_patch,
            patch(
                "avalan.cli.PromptWithoutPrefix.ask", return_value="a"
            ) as ask,
        ):
            result = get_input(
                self.console, "p", force_prompt=True, tty_path="/dev/test"
            )

        open_patch.assert_called_once_with("/dev/test")
        ask.assert_called_once_with("p ", stream=fake_tty)
        self.assertEqual(result, "a")
        self.assertEqual(
            self.console.print.call_args_list, [call(""), call("")]
        )

    def test_prompt_eof_raises(self):
        with (
            patch("avalan.cli.has_input", return_value=False),
            patch("avalan.cli.PromptWithoutPrefix.ask", side_effect=EOFError),
        ):
            with self.assertRaises(CommandAbortException):
                get_input(self.console, "prompt")

    def test_no_prompt_returns_none(self):
        with patch("avalan.cli.has_input", return_value=False):
            result = get_input(self.console, None)

        self.assertIsNone(result)
        self.console.print.assert_not_called()


class CliConfirmHasInputTestCase(TestCase):
    def test_confirm(self):
        with patch("avalan.cli.Confirm.ask", return_value=True) as ask:
            result = confirm(MagicMock(), "p?")
        ask.assert_called_once_with("p?")
        self.assertTrue(result)

    def test_has_input_true(self):
        with patch("avalan.cli.select", return_value=([object()], [], [])):
            self.assertTrue(has_input(MagicMock()))

    def test_has_input_false(self):
        with patch("avalan.cli.select", return_value=([], [], [])):
            self.assertFalse(has_input(MagicMock()))


class CliConfirmToolCallTestCase(TestCase):
    def test_confirm_tool_call(self):
        console = MagicMock()
        call = ToolCall(id=uuid4(), name="calc", arguments={"x": 1})
        with patch("avalan.cli.Prompt.ask", return_value="y") as ask:
            result = confirm_tool_call(console, call)

        printed = console.print.call_args.args[0]
        self.assertIsInstance(printed, Syntax)
        self.assertIn('"calc"', printed.code)
        self.assertIn('"x": 1', printed.code)
        ask.assert_called_once_with(
            "Execute tool call? ([y]es/[a]ll/[n]o)",
            choices=["y", "a", "n"],
            default="n",
        )
        self.assertEqual(result, "y")
