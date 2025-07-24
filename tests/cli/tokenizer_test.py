import sys
import importlib
from types import ModuleType, SimpleNamespace
from dataclasses import dataclass
from argparse import Namespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch, call


class CliTokenizerTestCase(IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        sys.path.insert(0, "src")

        # Preserve existing modules
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in [
                "avalan.entities",
                "avalan.model.hubs.huggingface",
                "avalan.model.nlp.text.generation",
            ]
        }

        # Stub avalan.entities
        entities = ModuleType("avalan.entities")

        @dataclass(frozen=True, kw_only=True)
        class Token:
            id: int
            token: str
            probability: float | None = None

        class TransformerEngineSettings:
            def __init__(
                self,
                device=None,
                cache_dir=None,
                **kwargs,
            ) -> None:
                self.device = device
                self.cache_dir = cache_dir
                for key, value in kwargs.items():
                    setattr(self, key, value)

            backend = "transformers"

        entities.Token = Token
        entities.TransformerEngineSettings = TransformerEngineSettings
        sys.modules["avalan.entities"] = entities

        # Stub avalan.model.hubs.huggingface
        hubs = ModuleType("avalan.model.hubs.huggingface")

        class HuggingfaceHub:
            def __init__(self, cache_dir="/cache"):
                self.cache_dir = cache_dir

        hubs.HuggingfaceHub = HuggingfaceHub
        sys.modules["avalan.model.hubs.huggingface"] = hubs

        # Stub avalan.model.nlp.text.generation
        nlp = ModuleType("avalan.model.nlp.text.generation")

        class TextGenerationModel:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def tokenize(self, text):
                return [text]

            def save_tokenizer(self, path):
                return [path]

            tokenizer_config = SimpleNamespace(
                tokens=["t"],
                special_tokens=["s"],
                tokenizer_model_max_length=10,
            )
            config = "cfg"

        nlp.TextGenerationModel = TextGenerationModel
        sys.modules["avalan.model.nlp.text.generation"] = nlp

        cls.tokenizer_mod = importlib.import_module(
            "avalan.cli.commands.tokenizer"
        )
        cls.TransformerEngineSettings = TransformerEngineSettings

    @classmethod
    def tearDownClass(cls):
        for name, mod in cls._saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)
        sys.path.remove("src")
        super().tearDownClass()

    def setUp(self):
        self.args = Namespace(
            tokenizer="dummy/tokenizer",
            device="cpu",
            token=None,
            special_token=None,
            disable_loading_progress_bar=False,
            low_cpu_mem_usage=False,
            loader_class="auto",
            backend="transformers",
            weight_type="auto",
            save=None,
            no_repl=False,
            quiet=False,
        )
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme._n = lambda s, p, n: s if n == 1 else p
        self.theme.icons = {"user_input": ">"}
        self.theme.tokenizer_config.return_value = "cfg_panel"
        self.theme.saved_tokenizer_files.return_value = "save_panel"
        self.theme.tokenizer_tokens.return_value = "tokens_panel"
        self.hub = SimpleNamespace(cache_dir="/cache")
        self.logger = MagicMock()

    async def test_save_tokenizer(self):
        args = Namespace(**vars(self.args))
        args.save = "/tmp/tokenizer"

        dummy_model = MagicMock()
        dummy_model.__enter__.return_value = dummy_model
        dummy_model.__exit__.return_value = False
        dummy_model.tokenizer_config = self.TransformerEngineSettings()
        dummy_model.save_tokenizer.return_value = ["f1", "f2"]

        with (
            patch.object(
                self.tokenizer_mod,
                "TextGenerationModel",
                return_value=dummy_model,
            ) as Model,
            patch.object(self.tokenizer_mod, "get_input") as get_input,
        ):
            await self.tokenizer_mod.tokenize(
                args, self.console, self.theme, self.hub, self.logger
            )

        Model.assert_called_once()
        settings = Model.call_args.kwargs["settings"]
        self.assertEqual(settings.device, args.device)
        self.assertEqual(settings.cache_dir, self.hub.cache_dir)
        self.assertFalse(settings.auto_load_model)
        self.assertTrue(settings.auto_load_tokenizer)
        dummy_model.save_tokenizer.assert_called_once_with(args.save)
        self.theme.tokenizer_config.assert_called_once_with(
            dummy_model.tokenizer_config
        )
        self.theme.saved_tokenizer_files.assert_called_once_with(args.save, 2)
        self.assertEqual(
            self.console.print.call_args_list,
            [call("cfg_panel"), call("save_panel")],
        )
        get_input.assert_not_called()

    async def test_tokenize_input(self):
        args = Namespace(**vars(self.args))

        dummy_model = MagicMock()
        dummy_model.__enter__.return_value = dummy_model
        dummy_model.__exit__.return_value = False
        dummy_model.tokenizer_config = SimpleNamespace(
            tokens=["x"], special_tokens=["y"]
        )
        dummy_model.tokenize.return_value = ["tok"]

        with (
            patch.object(
                self.tokenizer_mod,
                "TextGenerationModel",
                return_value=dummy_model,
            ) as Model,
            patch.object(
                self.tokenizer_mod, "get_input", return_value="hello"
            ) as get_input,
        ):
            await self.tokenizer_mod.tokenize(
                args, self.console, self.theme, self.hub, self.logger
            )

        Model.assert_called_once()
        dummy_model.tokenize.assert_called_once_with("hello")
        get_input.assert_called_once_with(
            self.console,
            self.theme.icons["user_input"] + " ",
            echo_stdin=not args.no_repl,
            is_quiet=args.quiet,
        )
        self.theme.tokenizer_tokens.assert_called_once_with(
            ["tok"],
            dummy_model.tokenizer_config.tokens,
            dummy_model.tokenizer_config.special_tokens,
            display_details=True,
        )
        self.assertEqual(
            self.console.print.call_args_list,
            [call("cfg_panel"), call("tokens_panel")],
        )
