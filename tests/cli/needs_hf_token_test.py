import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from async_helpers import run_async

from avalan.cli.__main__ import CLI, ModelManager


def _needs_hf_token(args: Namespace) -> bool:
    return run_async(CLI._needs_hf_token(args))


def _task_agent_engine_uri(args: Namespace) -> str | None:
    return run_async(CLI._task_agent_engine_uri(args))


class NeedsHfTokenTestCase(unittest.TestCase):
    def test_flow_commands_do_not_require_token(self):
        args = Namespace(command="flow", flow_command="run")

        self.assertFalse(_needs_hf_token(args))

    def test_model_run_local_requires_token(self):
        args = Namespace(command="model", model_command="run", model="m")
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=True),
        ) as parse_patch:
            self.assertTrue(_needs_hf_token(args))
        parse_patch.assert_called_once_with("m")

    def test_model_run_local_ds4_cli_backend_no_token(self):
        args = Namespace(
            command="model",
            model_command="run",
            model="ai://local/./model.gguf",
            backend="ds4",
        )
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=True, params={}),
        ) as parse_patch:
            self.assertFalse(_needs_hf_token(args))
        parse_patch.assert_called_once_with("ai://local/./model.gguf")

    def test_model_run_local_ds4_uri_backend_no_token(self):
        args = Namespace(
            command="model",
            model_command="run",
            model="ai://local/./model.gguf?backend=ds4",
            backend="transformers",
        )
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(
                is_local=True, params={"backend": "ds4"}
            ),
        ) as parse_patch:
            self.assertFalse(_needs_hf_token(args))
        parse_patch.assert_called_once_with(
            "ai://local/./model.gguf?backend=ds4"
        )

    def test_model_run_remote_no_token(self):
        args = Namespace(command="model", model_command="run", model="m")
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=False),
        ):
            self.assertFalse(_needs_hf_token(args))

    def test_agent_run_local_requires_token(self):
        args = Namespace(command="agent", agent_command="run", engine_uri="e")
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=True),
        ) as parse_patch:
            self.assertTrue(_needs_hf_token(args))
        parse_patch.assert_called_once_with("e")

    def test_agent_run_local_ds4_uri_backend_no_token(self):
        args = Namespace(
            command="agent",
            agent_command="run",
            engine_uri="ai://local/./model.gguf?backend=ds4",
            backend="transformers",
        )
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(
                is_local=True, params={"backend": "ds4"}
            ),
        ) as parse_patch:
            self.assertFalse(_needs_hf_token(args))
        parse_patch.assert_called_once_with(
            "ai://local/./model.gguf?backend=ds4"
        )

    def test_agent_serve_remote_no_token(self):
        args = Namespace(
            command="agent", agent_command="serve", engine_uri="e"
        )
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=False),
        ):
            self.assertFalse(_needs_hf_token(args))

    def test_agent_missing_engine_defaults_true(self):
        args = Namespace(command="agent", agent_command="serve")
        self.assertTrue(_needs_hf_token(args))

    def test_task_command_does_not_need_token(self):
        args = Namespace(command="task", task_command="validate")

        self.assertFalse(_needs_hf_token(args))

    def test_task_run_local_agent_requires_token(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "agents"
            agents.mkdir()
            (agents / "local.toml").write_text(
                '[engine]\nuri = "ai://local/private-model"',
                encoding="utf-8",
            )
            task = root / "task.toml"
            task.write_text(
                '[execution]\ntype = "agent"\nref = "agents/local.toml"',
                encoding="utf-8",
            )
            args = Namespace(
                command="task",
                task_command="run",
                definition=str(task),
            )
            with patch.object(
                ModelManager,
                "parse_uri",
                return_value=SimpleNamespace(is_local=True, params={}),
            ) as parse_patch:
                self.assertTrue(_needs_hf_token(args))
        parse_patch.assert_called_once_with("ai://local/private-model")

    def test_task_enqueue_remote_agent_does_not_need_token(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "agents"
            agents.mkdir()
            (agents / "remote.toml").write_text(
                '[engine]\nuri = "ai://openai/gpt-4o-mini"',
                encoding="utf-8",
            )
            task = root / "task.toml"
            task.write_text(
                '[execution]\ntype = "agent"\nref = "agents/remote.toml"',
                encoding="utf-8",
            )
            args = Namespace(
                command="task",
                task_command="enqueue",
                definition=str(task),
            )
            with patch.object(
                ModelManager,
                "parse_uri",
                return_value=SimpleNamespace(is_local=False, params={}),
            ) as parse_patch:
                self.assertFalse(_needs_hf_token(args))
        parse_patch.assert_called_once_with("ai://openai/gpt-4o-mini")

    def test_task_run_local_ds4_agent_does_not_need_token(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "agents"
            agents.mkdir()
            (agents / "ds4.toml").write_text(
                '[engine]\nuri = "ai://local/./model.gguf?backend=ds4"',
                encoding="utf-8",
            )
            task = root / "task.toml"
            task.write_text(
                '[execution]\ntype = "agent"\nref = "agents/ds4.toml"',
                encoding="utf-8",
            )
            args = Namespace(
                command="task",
                task_command="run",
                definition=str(task),
                backend="transformers",
            )
            with patch.object(
                ModelManager,
                "parse_uri",
                return_value=SimpleNamespace(
                    is_local=True, params={"backend": "ds4"}
                ),
            ) as parse_patch:
                self.assertFalse(_needs_hf_token(args))
        parse_patch.assert_called_once_with(
            "ai://local/./model.gguf?backend=ds4"
        )

    def test_task_run_missing_agent_defers_to_task_validation(self):
        with TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.toml"
            task.write_text(
                '[execution]\ntype = "agent"\nref = "agents/missing.toml"',
                encoding="utf-8",
            )
            args = Namespace(
                command="task",
                task_command="run",
                definition=str(task),
            )
            with patch.object(ModelManager, "parse_uri") as parse_patch:
                self.assertFalse(_needs_hf_token(args))
        parse_patch.assert_not_called()

    def test_task_agent_engine_uri_rejects_missing_definition_argument(self):
        args = Namespace(command="task", task_command="run")

        self.assertIsNone(_task_agent_engine_uri(args))

    def test_task_agent_engine_uri_rejects_missing_execution_section(self):
        with TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.toml"
            task.write_text("[task]\nname = 'example'", encoding="utf-8")
            args = Namespace(
                command="task",
                task_command="run",
                definition=str(task),
            )

            self.assertIsNone(_task_agent_engine_uri(args))

    def test_task_agent_engine_uri_rejects_non_agent_target(self):
        with TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.toml"
            task.write_text(
                '[execution]\ntype = "flow"\nref = "flows/example.toml"',
                encoding="utf-8",
            )
            args = Namespace(
                command="task",
                task_command="run",
                definition=str(task),
            )

            self.assertIsNone(_task_agent_engine_uri(args))

    def test_task_agent_engine_uri_rejects_missing_agent_ref(self):
        with TemporaryDirectory() as tmp:
            task = Path(tmp) / "task.toml"
            task.write_text(
                '[execution]\ntype = "agent"',
                encoding="utf-8",
            )
            args = Namespace(
                command="task",
                task_command="run",
                definition=str(task),
            )

            self.assertIsNone(_task_agent_engine_uri(args))

    def test_task_agent_engine_uri_rejects_missing_engine_section(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agents = root / "agents"
            agents.mkdir()
            (agents / "missing_engine.toml").write_text(
                "[agent]\nrole = 'assistant'",
                encoding="utf-8",
            )
            task = root / "task.toml"
            task.write_text(
                '[execution]\ntype = "agent"\n'
                'ref = "agents/missing_engine.toml"',
                encoding="utf-8",
            )
            args = Namespace(
                command="task",
                task_command="run",
                definition=str(task),
            )

            self.assertIsNone(_task_agent_engine_uri(args))

    def test_agent_proxy_remote_no_token(self):
        args = Namespace(
            command="agent", agent_command="proxy", engine_uri="e"
        )
        with patch.object(
            ModelManager,
            "parse_uri",
            return_value=SimpleNamespace(is_local=False),
        ):
            self.assertFalse(_needs_hf_token(args))

    def test_agent_proxy_missing_engine_defaults_true(self):
        args = Namespace(command="agent", agent_command="proxy")
        self.assertTrue(_needs_hf_token(args))

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
                self.assertFalse(_needs_hf_token(args))
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
                self.assertTrue(_needs_hf_token(args))
        parse_patch.assert_called_once_with("e")

    def test_agent_run_specs_local_ds4_backend_no_token(self):
        with NamedTemporaryFile("w", suffix=".toml") as spec:
            spec.write('[engine]\nuri="e"')
            spec.flush()
            args = Namespace(
                command="agent",
                agent_command="run",
                engine_uri=None,
                specifications_file=spec.name,
                backend="ds4",
            )
            with patch.object(
                ModelManager,
                "parse_uri",
                return_value=SimpleNamespace(is_local=True, params={}),
            ) as parse_patch:
                self.assertFalse(_needs_hf_token(args))
        parse_patch.assert_called_once_with("e")
