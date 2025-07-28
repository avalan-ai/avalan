from avalan.entities import EngineUri, Modality, TransformerEngineSettings
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager
from logging import Logger
from unittest import TestCase
from unittest.mock import MagicMock, patch


class ModelManagerExtraTestCase(TestCase):
    def setUp(self):
        self.hub = MagicMock(spec=HuggingfaceHub)
        self.hub.cache_dir = "cache"
        self.logger = MagicMock(spec=Logger)

    def test_parse_uri_invalid_scheme(self):
        manager = ModelManager(self.hub, self.logger)
        with self.assertRaises(ValueError):
            manager.parse_uri("http://openai/gpt-4o")

    def test_get_engine_settings_user_password_no_secret(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host="openai",
            port=None,
            user="token",
            password="pass",
            vendor="openai",
            model_id="gpt",
            params={},
        )
        settings = manager.get_engine_settings(uri)
        self.assertIsNone(settings.access_token)

    def test_load_passes_arguments(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            result = manager.load(
                uri,
                base_url="url",
                quiet=True,
                attention="sd",
                trust_remote_code=True,
                backend="mlxlm",
            )
        args = get_mock.call_args.args[1]
        self.assertTrue(args["disable_loading_progress_bar"])
        self.assertEqual(args["base_url"], "url")
        self.assertEqual(args["attention"], "sd")
        self.assertEqual(args["backend"], "mlxlm")
        self.assertTrue(args["trust_remote_code"])
        load_mock.assert_called_once_with(
            uri, get_mock.return_value, Modality.TEXT_GENERATION
        )
        self.assertEqual(result, "model")

    def test_load_engine_invalid_modality(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://tok@openai/gpt-4o")
        settings = TransformerEngineSettings()
        with self.assertRaises(NotImplementedError):
            manager.load_engine(uri, settings, "invalid")  # type: ignore[arg-type]

    def test_load_invalid_modality(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://tok@openai/gpt-4o")
        with self.assertRaises(NotImplementedError):
            manager.load(uri, modality="invalid")  # type: ignore[arg-type]

    def test_load_output_hidden_states(self):
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        manager = ModelManager(self.hub, self.logger)
        for modality in (Modality.TEXT_GENERATION, Modality.EMBEDDING):
            for value in (True, False, None):
                with self.subTest(modality=modality, value=value):
                    with (
                        patch.object(manager, "get_engine_settings") as ges,
                        patch.object(manager, "load_engine") as le,
                    ):
                        manager._stack.enter_context = MagicMock()
                        ges.return_value = TransformerEngineSettings()
                        le.return_value = "model"
                        result = manager.load(
                            uri, modality=modality, output_hidden_states=value
                        )
                    args = ges.call_args.args[1]
                    if value is None:
                        self.assertNotIn("output_hidden_states", args)
                    else:
                        self.assertEqual(args["output_hidden_states"], value)
                    le.assert_called_once_with(uri, ges.return_value, modality)
                    self.assertEqual(result, "model")
