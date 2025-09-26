from argparse import Namespace
from logging import Logger
from tempfile import NamedTemporaryFile
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.cli.__main__ import CLI
from avalan.cli.commands import agent as agent_cmds


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

    def test_serve_parser_defaults(self) -> None:
        args = self.parser.parse_args(["agent", "serve", "spec.toml"])
        self.assertTrue(args.participant)
        self.assertIsNone(args.id)

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
        self.assertEqual(srv_patch.call_args.kwargs["protocols"], expected_protocols)

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
                ["openai:responses,completion", "a2a"],
                {"openai": {"completions", "responses"}, "a2a": set()},
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
                    self.assertEqual(srv_patch.call_args.kwargs["protocols"], expected)

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
            (["mcp", "a2a"], {"mcp": set(), "a2a": set()}),
        ]

        base_config = "[agent]\nname='a'\n[engine]\nuri='m'\n"

        for config_protocols, expected in cases:
            protocols_literal = ", ".join(
                f"\"{value}\"" for value in config_protocols
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
                    self.assertEqual(srv_patch.call_args.kwargs["protocols"], expected)

    async def test_agent_serve_cli_protocols_override_spec(self) -> None:
        hub = MagicMock()
        logger = MagicMock(spec=Logger)

        with NamedTemporaryFile("w") as spec:
            spec.write("[agent]\nname='a'\n[engine]\nuri='m'\n")
            spec.write("[serve]\nprotocols = [\"mcp\"]\n")
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
