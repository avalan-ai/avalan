import sys
import logging
from argparse import ArgumentParser, Namespace, _SubParsersAction
from types import SimpleNamespace
from pathlib import Path
from contextlib import ExitStack
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.cli.__main__ import CLI, HuggingfaceHub
from avalan.cli import CommandAbortException


def _collect_progs(parser: ArgumentParser) -> list[str]:
    progs: list[str] = []
    stack = [parser]
    while stack:
        p = stack.pop()
        progs.append(p.prog)
        for action in p._actions:
            if isinstance(action, _SubParsersAction):
                for sub in action.choices.values():
                    stack.append(sub)
    return progs


class CliInitTestCase(TestCase):
    def test_constructor_creates_parser_and_sets_help_full(self):
        logger = MagicMock()
        with (
            patch.object(sys, "argv", ["prog"]),
            patch(
                "avalan.cli.__main__.getlocale",
                return_value=("en_US", "UTF-8"),
            ),
            patch(
                "avalan.cli.__main__.TransformerModel.get_default_device",
                return_value="cpu",
            ),
            patch.object(
                CLI, "_create_parser", wraps=CLI._create_parser
            ) as cp,
        ):
            cli = CLI(logger)

        locales_path = str(
            Path(sys.modules[CLI.__module__].__file__).resolve().parents[3]
            / "locale"
        )
        cp.assert_called_once_with(
            "cpu",
            HuggingfaceHub.DEFAULT_CACHE_DIR,
            locales_path,
            "en_US",
        )
        args = cli._parser.parse_args(["--help-full"])
        self.assertTrue(hasattr(args, "help_full"))


class CliParallelOptionTestCase(TestCase):
    def test_parallel_argument(self) -> None:
        logger = MagicMock()
        with patch.object(sys, "argv", ["prog"]):
            cli = CLI(logger)
        args = cli._parser.parse_args(["--parallel", "colwise"])
        self.assertEqual(args.parallel, "colwise")

    def test_parallel_count_argument(self) -> None:
        logger = MagicMock()
        with (
            patch.object(sys, "argv", ["prog"]),
            patch.object(CLI, "_default_parallel_count", return_value=3),
        ):
            cli = CLI(logger)
        args = cli._parser.parse_args(
            [
                "--parallel",
                "colwise",
                "--parallel-count",
                "5",
            ]
        )
        self.assertEqual(args.parallel_count, 5)


class CliCallTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        from logging import getLogger

        self.logger = getLogger("cli-test")
        with patch.object(sys, "argv", ["prog"]):
            self.cli = CLI(self.logger)
        self.translator = SimpleNamespace(
            gettext=lambda s: s, ngettext=lambda s, p, n: s if n == 1 else p
        )

    async def test_call_runs_main(self):
        console = MagicMock(export_text=lambda: "")
        with (
            patch.object(sys, "argv", ["prog"]),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=console),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch.object(CLI, "_help") as help_mock,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        main_mock.assert_awaited_once()
        help_mock.assert_not_called()

    async def test_call_help_full_outputs_all_commands(self):
        records_console = None

        def create_console(*args, **kwargs):
            nonlocal records_console
            from rich.console import Console

            records_console = Console(record=True)
            return records_console

        with (
            patch.object(sys, "argv", ["prog", "--help-full"]),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", side_effect=create_console),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
            patch.object(CLI, "_help", wraps=self.cli._help) as help_mock,
        ):
            await self.cli()

        main_mock.assert_not_called()
        self.assertEqual(
            help_mock.call_count, len(_collect_progs(self.cli._parser))
        )
        output = records_console.export_text()
        for prog in _collect_progs(self.cli._parser):
            self.assertIn(prog, output)

    async def test_call_version_outputs_version_and_exits(self):
        with (
            patch.object(sys, "argv", ["prog", "--version"]),
            patch("builtins.print") as print_patch,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
            patch.object(CLI, "_help") as help_mock,
        ):
            await self.cli()

        print_patch.assert_called_once_with(
            f"{self.cli._name} {self.cli._version}"
        )
        main_mock.assert_not_called()
        help_mock.assert_not_called()

    async def test_call_parallel_invokes_torchrun(self):
        with (
            patch.object(sys, "argv", ["prog", "--parallel", "colwise"]),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=MagicMock()),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch("avalan.cli.__main__.run") as run_patch,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        run_patch.assert_called_once()
        main_mock.assert_not_called()

    async def test_call_parallel_child_destroys_group(self):
        with (
            patch.dict("os.environ", {"LOCAL_RANK": "0"}, clear=False),
            patch.object(sys, "argv", ["prog", "--parallel", "colwise"]),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=MagicMock()),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch("avalan.cli.__main__.destroy_process_group") as destroy,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        main_mock.assert_awaited_once()
        destroy.assert_called_once()

    async def test_call_parallel_child_ignores_destroy_assertion_error(self):
        with (
            patch.dict("os.environ", {"LOCAL_RANK": "0"}, clear=False),
            patch.object(sys, "argv", ["prog", "--parallel", "colwise"]),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=MagicMock()),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch(
                "avalan.cli.__main__.destroy_process_group",
                side_effect=AssertionError,
            ) as destroy,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        main_mock.assert_awaited_once()
        destroy.assert_called_once()

    async def test_call_parallel_child_keeps_non_cuda_device(self):
        with (
            patch.dict("os.environ", {"LOCAL_RANK": "1"}, clear=False),
            patch.object(
                sys,
                "argv",
                ["prog", "--parallel", "colwise", "--device", "cpu"],
            ),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=MagicMock()),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch("avalan.cli.__main__.destroy_process_group"),
            patch("avalan.cli.__main__.set_device") as set_device_mock,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        main_mock.assert_awaited_once()
        self.assertEqual(main_mock.await_args.args[0].device, "cpu")
        set_device_mock.assert_not_called()

    async def test_call_parallel_child_sets_rank_on_cuda_device(self):
        rank = 2
        with (
            patch.dict("os.environ", {"LOCAL_RANK": str(rank)}, clear=False),
            patch.object(
                sys,
                "argv",
                ["prog", "--parallel", "colwise", "--device", "cuda"],
            ),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=MagicMock()),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch("avalan.cli.__main__.destroy_process_group"),
            patch("avalan.cli.__main__.set_device") as set_device_mock,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        main_mock.assert_awaited_once()
        self.assertEqual(main_mock.await_args.args[0].device, f"cuda:{rank}")
        set_device_mock.assert_called_once_with(rank)

    async def test_call_parallel_child_leaves_explicit_cuda_device(self):
        with (
            patch.dict("os.environ", {"LOCAL_RANK": "0"}, clear=False),
            patch.object(
                sys,
                "argv",
                ["prog", "--parallel", "colwise", "--device", "cuda:1"],
            ),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=MagicMock()),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch("avalan.cli.__main__.destroy_process_group"),
            patch("avalan.cli.__main__.set_device") as set_device_mock,
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        main_mock.assert_awaited_once()
        self.assertEqual(main_mock.await_args.args[0].device, "cuda:1")
        set_device_mock.assert_not_called()

    async def test_call_adds_mask_spec_embed_filter(self):
        hf_logger = MagicMock()
        added_filters: list = []

        def capture_filter(f: logging.Filter) -> None:
            added_filters.append(f)

        hf_logger.addFilter.side_effect = capture_filter

        with (
            patch.object(sys, "argv", ["prog"]),
            patch(
                "avalan.cli.__main__.translation", return_value=self.translator
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=MagicMock()),
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch(
                "avalan.cli.__main__.hf_logging.get_logger",
                return_value=hf_logger,
            ),
            patch("avalan.cli.__main__.find_spec", return_value=None),
            patch("avalan.cli.__main__.logger_replace"),
            patch("avalan.cli.__main__.filterwarnings"),
            patch("avalan.cli.__main__.has_input", return_value=True),
            patch("avalan.cli.__main__.Confirm.ask", return_value=False),
        ):
            await self.cli()

        self.assertEqual(len(added_filters), 1)
        filt = added_filters[0]
        ok_record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="ok",
            args=(),
            exc_info=None,
        )
        bad_record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="wav2vec2.masked_spec_embed warning",
            args=(),
            exc_info=None,
        )
        self.assertTrue(filt.filter(ok_record))
        self.assertFalse(filt.filter(bad_record))


class CliMainDispatchTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        from logging import getLogger

        self.logger = getLogger("cli-dispatch-test")
        with patch.object(sys, "argv", ["prog"]):
            self.cli = CLI(self.logger)

    async def test_main_dispatch(self):
        args_base = Namespace(
            verbose=None,
            quiet=True,
            login=False,
            hf_token="tok",
        )
        theme = MagicMock()
        theme._ = lambda s: s
        console = MagicMock()
        hub = MagicMock(domain="hf")
        with ExitStack() as stack:
            stack.enter_context(
                patch("avalan.cli.__main__.has_input", return_value=True)
            )
            stack.enter_context(
                patch("avalan.cli.__main__.Confirm.ask", return_value=False)
            )
            stack.enter_context(
                patch("avalan.cli.__main__.find_spec", return_value=None)
            )
            stack.enter_context(patch("avalan.cli.__main__.logger_replace"))
            stack.enter_context(patch("avalan.cli.__main__.filterwarnings"))
            run_mock = stack.enter_context(
                patch("avalan.cli.__main__.agent_run", AsyncMock())
            )
            msg_mock = stack.enter_context(
                patch("avalan.cli.__main__.agent_message_search", AsyncMock())
            )
            serve_mock = stack.enter_context(
                patch("avalan.cli.__main__.agent_serve", AsyncMock())
            )
            proxy_mock = stack.enter_context(
                patch("avalan.cli.__main__.agent_proxy", AsyncMock())
            )
            init_mock = stack.enter_context(
                patch("avalan.cli.__main__.agent_init", AsyncMock())
            )
            cache_del = stack.enter_context(
                patch("avalan.cli.__main__.cache_delete")
            )
            cache_down = stack.enter_context(
                patch("avalan.cli.__main__.cache_download")
            )
            cache_list = stack.enter_context(
                patch("avalan.cli.__main__.cache_list")
            )
            mem_doc = stack.enter_context(
                patch("avalan.cli.__main__.memory_document_index", AsyncMock())
            )
            mem_search = stack.enter_context(
                patch("avalan.cli.__main__.memory_search", AsyncMock())
            )
            mem_emb = stack.enter_context(
                patch("avalan.cli.__main__.memory_embeddings", AsyncMock())
            )
            model_disp = stack.enter_context(
                patch("avalan.cli.__main__.model_display")
            )
            model_install = stack.enter_context(
                patch("avalan.cli.__main__.model_install")
            )
            model_run = stack.enter_context(
                patch("avalan.cli.__main__.model_run", AsyncMock())
            )
            model_search = stack.enter_context(
                patch("avalan.cli.__main__.model_search", AsyncMock())
            )
            model_uninstall = stack.enter_context(
                patch("avalan.cli.__main__.model_uninstall")
            )
            deploy_run_mock = stack.enter_context(
                patch("avalan.cli.__main__.deploy_run", AsyncMock())
            )
            tokenize_mock = stack.enter_context(
                patch("avalan.cli.__main__.tokenize", AsyncMock())
            )
            scenarios = [
                ("agent", "run", run_mock),
                ("agent", "message", msg_mock),
                ("agent", "serve", serve_mock),
                ("agent", "proxy", proxy_mock),
                ("agent", "init", init_mock),
                ("cache", "delete", cache_del),
                ("cache", "download", cache_down),
                ("cache", "list", cache_list),
                ("memory", "document", mem_doc),
                ("memory", "search", mem_search),
                ("memory", "embeddings", mem_emb),
                ("model", "display", model_disp),
                ("model", "install", model_install),
                ("model", "run", model_run),
                ("model", "search", model_search),
                ("model", "uninstall", model_uninstall),
                ("deploy", "run", deploy_run_mock),
                ("tokenizer", None, tokenize_mock),
            ]
            for cmd, subcmd, fn in scenarios:
                args = Namespace(**vars(args_base))
                args.command = cmd
                if cmd == "agent":
                    if subcmd == "message":
                        args.agent_command = "message"
                        args.agent_message_command = "search"
                    else:
                        args.agent_command = subcmd
                elif cmd == "cache":
                    args.cache_command = subcmd
                elif cmd == "memory":
                    if subcmd == "document":
                        args.memory_command = "document"
                        args.memory_document_command = "index"
                    else:
                        args.memory_command = subcmd
                elif cmd == "model":
                    args.model_command = subcmd
                elif cmd == "deploy":
                    args.deploy_command = subcmd
                await self.cli._main(args, theme, console, hub)
                self.assertTrue(fn.called)
                fn.reset_mock()


class CliMainLoginTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        from logging import getLogger

        self.logger = getLogger("cli-login-test")
        with patch.object(sys, "argv", ["prog"]):
            self.cli = CLI(self.logger)

    async def test_main_login_and_logger_replace(self):
        args = Namespace(
            command="cache",
            cache_command="list",
            verbose=1,
            quiet=False,
            login=True,
            hf_token="tok",
        )
        theme = MagicMock()
        theme._ = lambda s: s
        theme.welcome.return_value = "welcome"
        console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm
        hub = MagicMock(domain="hf")
        with (
            patch("avalan.cli.__main__.has_input", return_value=True),
            patch("avalan.cli.__main__.Confirm.ask", return_value=True),
            patch("avalan.cli.__main__.find_spec", return_value=True),
            patch("avalan.cli.__main__.logger_replace") as lr,
            patch("avalan.cli.__main__.filterwarnings"),
            patch("avalan.cli.__main__.cache_list") as cache_list,
        ):
            await self.cli._main(args, theme, console, hub, suggest_login=True)
        lr.assert_any_call(self.cli._logger, ["sentence_transformers"])
        lr.assert_any_call(self.cli._logger, ["httpx"])
        hub.login.assert_called_once()
        hub.user.assert_called_once()
        cache_list.assert_called_once()
        console.print.assert_any_call("welcome")


class CliMainAdditionalTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        from logging import getLogger

        self.logger = getLogger("cli-additional-test")
        with patch.object(sys, "argv", ["prog"]):
            self.cli = CLI(self.logger)

    def test_add_tool_settings_arguments(self):
        from dataclasses import dataclass

        @dataclass
        class Settings:
            flag: bool = False
            count: int = 0
            ratio: float = 0.0
            name: str = ""

        parser = ArgumentParser()
        CLI._add_tool_settings_arguments(
            parser, prefix="x", settings_cls=Settings
        )
        args = parser.parse_args(
            [
                "--tool-x-flag",
                "--tool-x-count",
                "5",
                "--tool-x-ratio",
                "0.5",
                "--tool-x-name",
                "Bob",
            ]
        )
        self.assertTrue(args.tool_x_flag)
        self.assertEqual(args.tool_x_count, 5)
        self.assertEqual(args.tool_x_ratio, 0.5)
        self.assertEqual(args.tool_x_name, "Bob")

    def test_add_tool_settings_arguments_list(self):
        from dataclasses import dataclass

        @dataclass
        class Settings:
            names: list[str]

        parser = ArgumentParser()
        CLI._add_tool_settings_arguments(
            parser, prefix="y", settings_cls=Settings
        )
        args = parser.parse_args(["--tool-y-names", "Alice"])
        self.assertEqual(args.tool_y_names, "Alice")

    async def test_call_prompts_for_token_and_handles_exception(self):
        with (
            patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "agent",
                    "run",
                    "--engine-uri",
                    "e",
                    "--role",
                    "r",
                    "--run-chat-x",
                    "--hf-token",
                    "",
                ],
            ),
            patch(
                "avalan.cli.__main__.translation",
                side_effect=FileNotFoundError(),
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch("avalan.cli.__main__.Console", return_value=MagicMock()),
            patch.object(CLI, "_needs_hf_token", return_value=True),
            patch("avalan.cli.__main__.has_input", return_value=True),
            patch("builtins.open", side_effect=OSError()),
            patch(
                "avalan.cli.__main__.Prompt.ask", return_value="tok"
            ) as ask_patch,
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch.object(CLI, "_help"),
            patch.object(CLI, "_main", AsyncMock()) as main_mock,
        ):
            await self.cli()
        ask_patch.assert_called_once()
        main_mock.assert_awaited_once()
        self.assertTrue(hasattr(self.cli._parser.parse_args([]), "help_full"))

    async def test_call_handles_abort_exception(self):
        with (
            patch.object(sys, "argv", ["prog"]),
            patch(
                "avalan.cli.__main__.translation",
                return_value=SimpleNamespace(
                    gettext=lambda s: s, ngettext=lambda s, p, n: s
                ),
            ),
            patch(
                "avalan.cli.__main__.FancyTheme",
                return_value=MagicMock(get_styles=lambda: {}),
            ),
            patch(
                "avalan.cli.__main__.Console", return_value=MagicMock()
            ) as console_patch,
            patch.object(CLI, "_needs_hf_token", return_value=False),
            patch("avalan.cli.__main__.HuggingfaceHub"),
            patch.object(
                CLI, "_main", AsyncMock(side_effect=CommandAbortException)
            ),
        ):
            await self.cli()
        console_patch.return_value.print.assert_called()


class CliMainFunctionTestCase(TestCase):
    def test_main_invokes_run(self):
        with (
            patch("avalan.cli.__main__.run_in_loop") as run,
            patch("avalan.cli.__main__.CLI") as cli,
        ):
            from avalan.cli.__main__ import main

            main()
        run.assert_called_once()
        cli.assert_called_once()
