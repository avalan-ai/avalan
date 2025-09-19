from argparse import Namespace
from logging import Logger
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


class AgentServeForwardOptionsTestCase(IsolatedAsyncioTestCase):
    async def test_agent_serve_passes_ids(self) -> None:
        args = Namespace(
            specifications_file="spec.toml",
            engine_uri=None,
            memory_recent=None,
            tool=None,
            tools=None,
            host="localhost",
            port=9001,
            openai_prefix="/v1",
            mcp_prefix="/mcp",
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
        )
        hub = MagicMock()
        logger = MagicMock(spec=Logger)
        server = MagicMock()
        server.serve = AsyncMock()
        with patch(
            "avalan.cli.commands.agent.agents_server", return_value=server
        ) as srv_patch:
            await agent_cmds.agent_serve(args, hub, logger, "name", "v")

        srv_patch.assert_called_once()
        self.assertEqual(srv_patch.call_args.kwargs["agent_id"], "aid")
        self.assertEqual(srv_patch.call_args.kwargs["participant_id"], "pid")
        server.serve.assert_awaited_once()
