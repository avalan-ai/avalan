import unittest
from types import SimpleNamespace
from argparse import Namespace
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from avalan.cli.__main__ import CLI, ModelManager


class NeedsHfTokenTestCase(unittest.TestCase):
    def test_model_run_local_requires_token(self):
        args = Namespace(command="model", model_command="run", model="m")
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=True),
        ) as parse_patch:
            self.assertTrue(CLI._needs_hf_token(args))
        parse_patch.assert_called_once_with("m")

    def test_model_run_remote_no_token(self):
        args = Namespace(command="model", model_command="run", model="m")
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=False),
        ):
            self.assertFalse(CLI._needs_hf_token(args))

    def test_agent_run_local_requires_token(self):
        args = Namespace(command="agent", agent_command="run", engine_uri="e")
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=True),
        ) as parse_patch:
            self.assertTrue(CLI._needs_hf_token(args))
        parse_patch.assert_called_once_with("e")

    def test_agent_serve_remote_no_token(self):
        args = Namespace(
            command="agent", agent_command="serve", engine_uri="e"
        )
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=False),
        ):
            self.assertFalse(CLI._needs_hf_token(args))

    def test_agent_missing_engine_defaults_true(self):
        args = Namespace(command="agent", agent_command="serve")
        self.assertTrue(CLI._needs_hf_token(args))

    def test_agent_proxy_remote_no_token(self):
        args = Namespace(
            command="agent", agent_command="proxy", engine_uri="e"
        )
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=False),
        ):
            self.assertFalse(CLI._needs_hf_token(args))

    def test_agent_proxy_missing_engine_defaults_true(self):
        args = Namespace(command="agent", agent_command="proxy")
        self.assertTrue(CLI._needs_hf_token(args))

    def test_agent_run_specs_remote_no_token(self):
        with NamedTemporaryFile("w", suffix=".toml") as spec:
            spec.write('[engine]\nuri="e"')
            spec.flush()
            args = Namespace(
                command="agent",
                agent_command="run",
                engine_uri=None,
                specifications_file=spec.name,
            )
            with patch.object(
                ModelManager,
                "parse_uri",
                return_value=SimpleNamespace(is_local=False),
            ) as parse_patch:
                self.assertFalse(CLI._needs_hf_token(args))
        parse_patch.assert_called_once_with("e")

    def test_agent_run_specs_local_requires_token(self):
        with NamedTemporaryFile("w", suffix=".toml") as spec:
            spec.write('[engine]\nuri="e"')
            spec.flush()
            args = Namespace(
                command="agent",
                agent_command="run",
                engine_uri=None,
                specifications_file=spec.name,
            )
            with patch.object(
                ModelManager,
                "parse_uri",
                return_value=SimpleNamespace(is_local=True),
            ) as parse_patch:
                self.assertTrue(CLI._needs_hf_token(args))
        parse_patch.assert_called_once_with("e")
