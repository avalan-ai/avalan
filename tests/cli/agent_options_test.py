from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from logging import Logger
from tempfile import NamedTemporaryFile
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.cli.__main__ import CLI
from avalan.cli.commands import agent as agent_cmds
from avalan.entities import Modality


class AgentParserOptionsTestCase(TestCase):
    def setUp(self) -> None:
        self.parser = CLI._create_parser(
            default_device="cpu",
            cache_dir="/tmp",
            default_locales_path="/tmp",
            default_locale="en_US",
        )

    def test_run_parser_with_common_options(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "run",
                "spec.toml",
                "--participant",
                "pid",
                "--id",
                "aid",
            ]
        )
        self.assertEqual(args.specifications_file, "spec.toml")
        self.assertEqual(args.participant, "pid")
        self.assertEqual(args.id, "aid")

    def test_serve_parser_with_common_options(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "serve",
                "spec.toml",
                "--participant",
                "pid",
                "--id",
                "aid",
            ]
        )
        self.assertEqual(args.specifications_file, "spec.toml")
        self.assertEqual(args.participant, "pid")
        self.assertEqual(args.id, "aid")

    def test_run_parser_defaults(self) -> None:
        args = self.parser.parse_args(["agent", "run", "spec.toml"])
        self.assertTrue(args.participant)
        self.assertIsNone(args.id)
        self.assertIsNone(args.run_temperature)

    def test_serve_parser_defaults(self) -> None:
        args = self.parser.parse_args(["agent", "serve", "spec.toml"])
        self.assertTrue(args.participant)
        self.assertIsNone(args.id)
        self.assertIsNone(args.run_temperature)

    def test_run_parser_accepts_temperature(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "run",
                "spec.toml",
                "--run-temperature",
                "0.5",
            ]
        )
        self.assertEqual(args.run_temperature, 0.5)

    def test_run_parser_accepts_engine_base_url(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "run",
                "--engine-uri",
                "ai://env:KEY@openai/deployment",
                "--engine-base-url",
                "https://tenant.openai.azure.com/openai/v1/",
            ]
        )
        self.assertEqual(
            args.engine_base_url,
            "https://tenant.openai.azure.com/openai/v1/",
        )

    def test_run_parser_accepts_graph_file(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "run",
                "spec.toml",
                "--tool-graph-file",
                "chart.png",
            ]
        )
        self.assertEqual(args.tool_graph_file, "chart.png")

    def test_run_parser_accepts_input_files(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "run",
                "spec.toml",
                "--input-file",
                "doc-1.pdf",
                "--input-file",
                "doc-2.pdf",
            ]
        )
        self.assertEqual(args.input_file, ["doc-1.pdf", "doc-2.pdf"])

    def test_run_parser_accepts_tool_choice(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "run",
                "spec.toml",
                "--tool-choice",
                "mcp.call",
            ]
        )
        self.assertEqual(args.tool_choice, "mcp.call")

    def test_serve_parser_protocol_option(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "serve",
                "spec.toml",
                "--protocol",
                "openai:responses",
                "--protocol",
                "mcp",
            ]
        )
        self.assertEqual(args.protocol, ["openai:responses", "mcp"])

    def test_serve_parser_server_output_redaction_options(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "serve",
                "spec.toml",
                "--server-output-redaction",
                "--server-output-redaction-rule",
                "host_paths",
                "--server-output-redaction-protocol",
                "mcp",
                "--server-output-redaction-channel",
                "reasoning",
            ]
        )

        self.assertTrue(args.server_output_redaction_enabled)
        self.assertEqual(args.server_output_redaction_rules, ["host_paths"])
        self.assertEqual(args.server_output_redaction_protocols, ["mcp"])
        self.assertEqual(
            args.server_output_redaction_channels,
            ["reasoning"],
        )

    def test_serve_parser_server_output_redaction_filter_help(
        self,
    ) -> None:
        output = StringIO()

        with redirect_stdout(output), self.assertRaises(SystemExit) as exc:
            self.parser.parse_args(["agent", "serve", "--help"])

        self.assertEqual(exc.exception.code, 0)
        help_text = " ".join(output.getvalue().split())
        self.assertIn(
            "Enable opt-in redaction for server protocol output.",
            help_text,
        )
        self.assertIn(
            "Enable server output redaction and restrict it to a rule",
            help_text,
        )
        self.assertIn(
            "Enable server output redaction and restrict it to a protocol",
            help_text,
        )
        self.assertIn(
            "Enable server output redaction and restrict it to a channel",
            help_text,
        )

    def test_serve_parser_reasoning_effort_alias(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "serve",
                "spec.toml",
                "--reasoning-effort",
                "xhigh",
            ]
        )
        self.assertEqual(args.run_reasoning_effort, "xhigh")

    def test_run_parser_reasoning_effort_alias(self) -> None:
        args = self.parser.parse_args(
            [
                "agent",
                "run",
                "spec.toml",
                "--reasoning-effort",
                "xhigh",
            ]
        )
        self.assertEqual(args.run_reasoning_effort, "xhigh")

    def test_run_and_init_reasoning_summary_aliases(self) -> None:
        for command, option in (
            (("agent", "run", "spec.toml"), "--reasoning-summary"),
            (("agent", "run", "spec.toml"), "--run-reasoning-summary"),
            (("agent", "init"), "--reasoning-summary"),
            (("agent", "init"), "--run-reasoning-summary"),
        ):
            with self.subTest(command=command, option=option):
                args = self.parser.parse_args([*command, option, "detailed"])
                self.assertEqual(args.run_reasoning_summary, "detailed")

    def test_reasoning_summary_choices_are_enum_derived(self) -> None:
        for mode in ("auto", "concise", "detailed"):
            with self.subTest(mode=mode):
                model_args = self.parser.parse_args(
                    [
                        "model",
                        "run",
                        "model-id",
                        "--reasoning-summary",
                        mode,
                    ]
                )
                agent_args = self.parser.parse_args(
                    [
                        "agent",
                        "run",
                        "spec.toml",
                        "--reasoning-summary",
                        mode,
                    ]
                )
                self.assertEqual(model_args.reasoning_summary, mode)
                self.assertEqual(agent_args.run_reasoning_summary, mode)

    def test_invalid_reasoning_summary_choices_exit_two(self) -> None:
        for command in (
            ["model", "run", "model-id"],
            ["agent", "run", "spec.toml"],
            ["agent", "init"],
        ):
            stderr = StringIO()
            with (
                self.subTest(command=command),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                self.parser.parse_args(
                    [
                        *command,
                        "--reasoning-summary",
                        "verbose",
                    ]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertIn("invalid choice", stderr.getvalue())

    def test_reasoning_summary_and_disabled_reasoning_exit_two(self) -> None:
        stderr = StringIO()
        with (
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as error,
        ):
            self.parser.parse_args(
                [
                    "model",
                    "run",
                    "model-id",
                    "--reasoning-summary",
                    "auto",
                    "--no-reasoning",
                ]
            )

        self.assertEqual(error.exception.code, 2)
        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_non_text_reasoning_summary_choices_exit_two(self) -> None:
        for modality in Modality:
            if modality is Modality.TEXT_GENERATION:
                continue
            stderr = StringIO()
            with (
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                self.parser.parse_args(
                    [
                        "model",
                        "run",
                        "model-id",
                        "--modality",
                        modality.value,
                        "--reasoning-summary",
                        "auto",
                    ]
                )

            self.assertEqual(error.exception.code, 2)
            self.assertIn(
                "requires --modality text_generation", stderr.getvalue()
            )

    def test_unhonored_agent_surfaces_reject_reasoning_summary(self) -> None:
        for command in (
            ["agent", "serve", "spec.toml"],
            ["agent", "proxy", "spec.toml"],
            [
                "agent",
                "message",
                "search",
                "spec.toml",
                "--function",
                "l2_distance",
                "--id",
                "agent-id",
                "--participant",
                "participant-id",
                "--session",
                "session-id",
            ],
        ):
            stderr = StringIO()
            with (
                self.subTest(command=command),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                self.parser.parse_args(
                    [
                        *command,
                        "--reasoning-summary",
                        "auto",
                    ]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertIn("unrecognized arguments", stderr.getvalue())


class AgentServeForwardOptionsTestCase(IsolatedAsyncioTestCase):
    def _make_args(self, **overrides):
        defaults = dict(
            specifications_file=None,
            engine_uri=None,
            memory_recent=None,
            tool=None,
            tools=None,
            host="localhost",
            port=9001,
            openai_prefix="/v1",
            mcp_prefix="/mcp",
            a2a_prefix="/a2a",
            mcp_name="run",
            mcp_description=None,
            reload=False,
            id="aid",
            participant="pid",
            cors_origin=None,
            cors_origin_regex=None,
            cors_method=None,
            cors_header=None,
            cors_credentials=False,
            protocol=None,
            a2a_name="run",
            a2a_description=None,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    async def test_agent_serve_passes_ids(self) -> None:
        hub = MagicMock()
        logger = MagicMock(spec=Logger)

        with NamedTemporaryFile("w") as spec:
            spec.write("[agent]\nname='a'\n[engine]\nuri='m'\n")
            spec.flush()

            args = self._make_args(specifications_file=spec.name)
            server = MagicMock()
            server.serve = AsyncMock()
            with patch(
                "avalan.cli.commands.agent.agents_server", return_value=server
            ) as srv_patch:
                await agent_cmds.agent_serve(args, hub, logger, "name", "v")

        srv_patch.assert_called_once()
        self.assertEqual(srv_patch.call_args.kwargs["agent_id"], "aid")
        self.assertEqual(srv_patch.call_args.kwargs["participant_id"], "pid")
        self.assertEqual(srv_patch.call_args.kwargs["a2a_prefix"], "/a2a")
        self.assertIsNone(srv_patch.call_args.kwargs["protocols"])
        server.serve.assert_awaited_once()

    async def test_agent_serve_forwards_server_output_redaction(
        self,
    ) -> None:
        hub = MagicMock()
        logger = MagicMock(spec=Logger)
        server = MagicMock()
        server.serve = AsyncMock()

        with NamedTemporaryFile("w") as spec:
            spec.write("[agent]\nname='a'\n[engine]\nuri='m'\n")
            spec.flush()
            args = self._make_args(
                specifications_file=spec.name,
                server_output_redaction_rules=["host_paths"],
                server_output_redaction_protocols=["mcp"],
                server_output_redaction_channels=["reasoning"],
            )
            with patch(
                "avalan.cli.commands.agent.agents_server",
                return_value=server,
            ) as srv_patch:
                await agent_cmds.agent_serve(args, hub, logger, "name", "v")

        settings = srv_patch.call_args.kwargs["output_redaction_settings"]
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.rules, frozenset({"host_paths"}))
        self.assertEqual(settings.protocols, frozenset({"mcp"}))
        self.assertEqual(settings.channels, frozenset({"reasoning"}))

    async def test_agent_serve_protocols_forwarded(self) -> None:
        args = self._make_args(
            specifications_file="spec.toml",
            protocol=["openai:responses", "a2a"],
        )
        hub = MagicMock()
        logger = MagicMock(spec=Logger)
        server = MagicMock()
        server.serve = AsyncMock()
        with patch(
            "avalan.cli.commands.agent.agents_server", return_value=server
        ) as srv_patch:
            await agent_cmds.agent_serve(args, hub, logger, "name", "v")

        expected_protocols = {
            "openai": {"responses"},
            "a2a": set(),
        }
        self.assertEqual(
            srv_patch.call_args.kwargs["protocols"], expected_protocols
        )

    async def test_agent_serve_cli_protocol_variations(self) -> None:
        hub = MagicMock()
        logger = MagicMock(spec=Logger)
        cases = [
            (["openai"], {"openai": {"completions", "responses"}}),
            (["openai:responses"], {"openai": {"responses"}}),
            (
                ["openai:completion", "mcp"],
                {"openai": {"completions"}, "mcp": set()},
            ),
            (
                ["openai:responses,completion", "a2a", "flow"],
                {
                    "openai": {"completions", "responses"},
                    "a2a": set(),
                    "flow": set(),
                },
            ),
        ]

        with NamedTemporaryFile("w") as spec:
            spec.write("[agent]\nname='a'\n[engine]\nuri='m'\n")
            spec.flush()

            for raw_protocols, expected in cases:
                with self.subTest(protocols=raw_protocols):
                    args = self._make_args(
                        specifications_file=spec.name,
                        protocol=raw_protocols,
                    )
                    server = MagicMock()
                    server.serve = AsyncMock()
                    with patch(
                        "avalan.cli.commands.agent.agents_server",
                        return_value=server,
                    ) as srv_patch:
                        await agent_cmds.agent_serve(
                            args, hub, logger, "name", "v"
                        )

                    server.serve.assert_awaited_once()
                    self.assertEqual(
                        srv_patch.call_args.kwargs["protocols"], expected
                    )

    async def test_agent_serve_spec_protocol_variations(self) -> None:
        hub = MagicMock()
        logger = MagicMock(spec=Logger)
        cases = [
            (["openai"], {"openai": {"completions", "responses"}}),
            (["openai:responses"], {"openai": {"responses"}}),
            (
                ["openai:responses", "openai:completion", "mcp"],
                {"openai": {"completions", "responses"}, "mcp": set()},
            ),
            (
                ["mcp", "a2a", "flow"],
                {"mcp": set(), "a2a": set(), "flow": set()},
            ),
        ]

        base_config = "[agent]\nname='a'\n[engine]\nuri='m'\n"

        for config_protocols, expected in cases:
            protocols_literal = ", ".join(
                f'"{value}"' for value in config_protocols
            )
            with self.subTest(protocols=config_protocols):
                with NamedTemporaryFile("w") as spec:
                    spec.write(base_config)
                    spec.write(f"[serve]\nprotocols = [{protocols_literal}]\n")
                    spec.flush()

                    args = self._make_args(specifications_file=spec.name)
                    server = MagicMock()
                    server.serve = AsyncMock()
                    with patch(
                        "avalan.cli.commands.agent.agents_server",
                        return_value=server,
                    ) as srv_patch:
                        await agent_cmds.agent_serve(
                            args, hub, logger, "name", "v"
                        )

                    server.serve.assert_awaited_once()
                    self.assertEqual(
                        srv_patch.call_args.kwargs["protocols"], expected
                    )

    async def test_agent_serve_cli_protocols_override_spec(self) -> None:
        hub = MagicMock()
        logger = MagicMock(spec=Logger)

        with NamedTemporaryFile("w") as spec:
            spec.write("[agent]\nname='a'\n[engine]\nuri='m'\n")
            spec.write('[serve]\nprotocols = ["mcp"]\n')
            spec.flush()

            args = self._make_args(
                specifications_file=spec.name,
                protocol=["openai:responses"],
            )
            server = MagicMock()
            server.serve = AsyncMock()

            with patch(
                "avalan.cli.commands.agent.agents_server", return_value=server
            ) as srv_patch:
                await agent_cmds.agent_serve(args, hub, logger, "name", "v")

        server.serve.assert_awaited_once()
        self.assertEqual(
            srv_patch.call_args.kwargs["protocols"], {"openai": {"responses"}}
        )
