import unittest
from types import SimpleNamespace
from argparse import Namespace
from unittest.mock import MagicMock, patch, call

from avalan.cli.commands import model as model_cmds


class CliModelTestCase(unittest.TestCase):
    def setUp(self):
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme.ask_secret_password.side_effect = lambda k: f"ask-{k}"
        self.logger = MagicMock()
        self.hub = MagicMock()

    def test_get_model_settings(self):
        engine_uri = MagicMock()
        args = Namespace(
            attention="flash",
            device="cpu",
            disable_loading_progress_bar=True,
            sentence_transformer=True,
            loader_class="auto",
            low_cpu_mem_usage=True,
            quiet=False,
            revision="rev",
            special_token=["<s>"],
            tokenizer="tok",
            token=["t"],
            trust_remote_code=True,
            weight_type="fp16",
        )

        result = model_cmds.get_model_settings(args, self.hub, self.logger, engine_uri)
        expected = {
            "engine_uri": engine_uri,
            "attention": "flash",
            "device": "cpu",
            "disable_loading_progress_bar": True,
            "is_sentence_transformer": True,
            "loader_class": "auto",
            "low_cpu_mem_usage": True,
            "quiet": False,
            "revision": "rev",
            "special_tokens": ["<s>"],
            "tokenizer": "tok",
            "tokens": ["t"],
            "trust_remote_code": True,
            "weight_type": "fp16",
        }
        self.assertEqual(result, expected)

    def test_model_install_secret_creates_secret(self):
        args = Namespace(model="m")
        engine_uri = SimpleNamespace(vendor="openai", password="pw", user="secret")
        secrets = MagicMock()
        secrets.read.return_value = None
        with patch.object(model_cmds.ModelManager, "parse_uri", return_value=engine_uri), \
             patch.object(model_cmds, "KeyringSecrets", return_value=secrets) as ks, \
             patch.object(model_cmds.Prompt, "ask", return_value="val") as ask, \
             patch.object(model_cmds, "cache_download") as cache_download, \
             patch.object(model_cmds, "confirm") as confirm:
            model_cmds.model_install(args, self.console, self.theme, self.hub)

        ks.assert_called_once_with()
        secrets.read.assert_called_once_with("pw")
        ask.assert_called_once_with("ask-pw")
        secrets.write.assert_called_once_with("pw", "val")
        confirm.assert_not_called()
        cache_download.assert_called_once_with(args, self.console, self.theme, self.hub)

    def test_model_uninstall_secret(self):
        args = Namespace(model="m")
        engine_uri = SimpleNamespace(vendor="openai", password="pw", user="secret")
        secrets = MagicMock()
        with patch.object(model_cmds.ModelManager, "parse_uri", return_value=engine_uri), \
             patch.object(model_cmds, "KeyringSecrets", return_value=secrets) as ks, \
             patch.object(model_cmds, "cache_delete") as cache_delete:
            model_cmds.model_uninstall(args, self.console, self.theme, self.hub)

        ks.assert_called_once_with()
        secrets.delete.assert_called_once_with("pw")
        cache_delete.assert_called_once_with(args, self.console, self.theme, self.hub, is_full_deletion=True)

    def test_model_display_uses_provided_model(self):
        args = Namespace(model="id", skip_hub_access_check=False, summary=False, load=False)
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri.return_value = "uri"
        model = SimpleNamespace(config="cfg", tokenizer_config="tok_cfg")
        self.hub.can_access.return_value = True
        self.hub.model.return_value = "hub_model"
        with patch.object(model_cmds, "ModelManager", return_value=manager):
            model_cmds.model_display(args, self.console, self.theme, self.hub, self.logger, model=model)

        manager.parse_uri.assert_called_once_with("id")
        self.hub.can_access.assert_called_once_with("id")
        self.hub.model.assert_called_once_with("id")
        manager.load.assert_not_called()
        self.theme.model.assert_called_once_with(
            "hub_model",
            can_access=True,
            expand=True,
            summary=False,
        )
        self.theme.model_display.assert_called_once_with(
            model.config,
            model.tokenizer_config,
            summary=False,
        )


from unittest import IsolatedAsyncioTestCase


class CliTokenGenerationTestCase(IsolatedAsyncioTestCase):
    async def test_token_generation_no_stats(self):
        async def gen():
            for t in ["a", "b"]:
                yield t
        args = Namespace()
        console = MagicMock()
        await model_cmds.token_generation(
            args=args,
            console=console,
            theme=MagicMock(),
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=MagicMock(),
            input_string="i",
            response=gen(),
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
        )
        console.print.assert_has_calls([call("a", end=""), call("b", end="")])


if __name__ == "__main__":
    unittest.main()
