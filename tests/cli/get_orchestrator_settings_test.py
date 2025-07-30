import unittest
from argparse import Namespace
from uuid import UUID

from avalan.cli.commands import agent as agent_cmds
from avalan.agent.loader import OrchestratorLoader


class GetOrchestratorSettingsTestCase(unittest.TestCase):
    def test_defaults(self):
        args = Namespace(
            name="a",
            role="r",
            task=None,
            instructions=None,
            engine_uri="ai://m",
            backend="transformers",
            run_max_new_tokens=10,
            run_skip_special_tokens=False,
            memory_recent=None,
            no_session=False,
            memory_permanent_message="dsn",
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=200,
            memory_engine_overlap=20,
            memory_engine_window=40,
            tool=None,
        )
        uid = UUID("00000000-0000-0000-0000-000000000001")
        result = agent_cmds.get_orchestrator_settings(args, agent_id=uid)
        self.assertTrue(result.memory_recent)
        self.assertEqual(result.uri, "ai://m")
        self.assertEqual(result.call_options["max_new_tokens"], 10)
        self.assertEqual(result.engine_config, {"backend": "transformers"})
        self.assertEqual(result.agent_config, {"name": "a", "role": "r"})
        self.assertEqual(result.tools, [])
        self.assertEqual(
            result.sentence_model_id,
            OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
        )

    def test_overrides(self):
        args = Namespace(
            name="n",
            role="r",
            task="t",
            instructions="i",
            engine_uri="old",
            backend="transformers",
            run_max_new_tokens=5,
            run_skip_special_tokens=True,
            memory_recent=False,
            no_session=True,
            memory_permanent_message="old_dsn",
            memory_permanent=["ns@dsn1"],
            memory_engine_model_id="m",
            memory_engine_max_tokens=300,
            memory_engine_overlap=30,
            memory_engine_window=60,
            tool=["a"],
        )
        uid = UUID("00000000-0000-0000-0000-000000000002")
        result = agent_cmds.get_orchestrator_settings(
            args,
            agent_id=uid,
            name="x",
            role="y",
            task="z",
            instructions="j",
            engine_uri="new",
            memory_recent=True,
            memory_permanent_message="dsn",
            memory_permanent=["ns@dsn2"],
            max_new_tokens=20,
            tools=["b"],
        )
        self.assertTrue(result.memory_recent)
        self.assertEqual(result.uri, "new")
        self.assertEqual(result.call_options["max_new_tokens"], 20)
        self.assertEqual(
            result.agent_config,
            {"name": "x", "role": "y", "task": "z", "instructions": "j"},
        )
        self.assertEqual(result.memory_permanent_message, "dsn")
        self.assertEqual(result.permanent_memory, {"ns": "dsn2"})
        self.assertEqual(result.tools, ["b"])
        self.assertEqual(result.sentence_model_id, "m")
        self.assertEqual(result.engine_config, {"backend": "transformers"})

    def test_chat_settings(self):
        args = Namespace(
            name="a",
            role="r",
            task=None,
            instructions=None,
            engine_uri="ai://m",
            backend="transformers",
            run_max_new_tokens=10,
            run_skip_special_tokens=False,
            memory_recent=None,
            no_session=False,
            memory_permanent_message="dsn",
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=200,
            memory_engine_overlap=20,
            memory_engine_window=40,
            tool=None,
            run_chat_enable_thinking=True,
        )
        uid = UUID("00000000-0000-0000-0000-000000000003")
        result = agent_cmds.get_orchestrator_settings(args, agent_id=uid)
        self.assertTrue(
            result.call_options["chat_settings"]["enable_thinking"]
        )
        self.assertEqual(result.engine_config, {"backend": "transformers"})
