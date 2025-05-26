from avalan.model.entities import EngineUri, TransformerEngineSettings
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager
from avalan.secrets import KeyringSecrets
from logging import Logger
from unittest import main, TestCase
from unittest.mock import MagicMock, patch
import sys
import types

class ManagerTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_uris = [
            (
                "tiiuae/Falcon-E-3B-Instruct",
                None,
                "tiiuae/Falcon-E-3B-Instruct",
                None,
                None
            ),
            (
                "ai://local/tiiuae/Falcon-E-3B-Instruct",
                None,
                "tiiuae/Falcon-E-3B-Instruct",
                None,
                None
            ),
            (
                "meta-llama/Meta-Llama-3-8B-Instruct",
                None,
                "meta-llama/Meta-Llama-3-8B-Instruct",
                None,
                None
            ),
            (
                "ai://local/meta-llama/Meta-Llama-3-8B-Instruct",
                None,
                "meta-llama/Meta-Llama-3-8B-Instruct",
                None,
                None
            ),
            (
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                None,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                None,
                None
            ),
            (
                "ai://local/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                None,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                None,
                None
            ),
            (
                "Qwen/Qwen2.5-7B-Instruct",
                None,
                "Qwen/Qwen2.5-7B-Instruct",
                None,
                None
            ),
            (
                "ai://local/Qwen/Qwen2.5-7B-Instruct",
                None,
                "Qwen/Qwen2.5-7B-Instruct",
                None,
                None
            ),
            (
                "ai://messi_api_key:@openai/gpt-4o",
                "openai",
                "gpt-4o",
                "messi_api_key",
                None
            ),
            (
                "ai://messi_api_key@openai/gpt-4o",
                "openai",
                "gpt-4o",
                "messi_api_key",
                None
            ),
            (
                "ai://router_api_key:@openrouter/gpt-3.5-turbo",
                "openrouter",
                "gpt-3.5-turbo",
                "router_api_key",
                None
            ),
            (
                "ai://router_api_key@openrouter/gpt-3.5-turbo",
                "openrouter",
                "gpt-3.5-turbo",
                "router_api_key",
                None
            ),
            (
                "ai://scale_key:@anyscale/gpt-3.5-turbo",
                "anyscale",
                "gpt-3.5-turbo",
                "scale_key",
                None
            ),
            (
                "ai://scale_key@anyscale/gpt-3.5-turbo",
                "anyscale",
                "gpt-3.5-turbo",
                "scale_key",
                None
            ),
            (
                "ai://tog_key:@together/mistral-7b",
                "together",
                "mistral-7b",
                "tog_key",
                None
            ),
            (
                "ai://tog_key@together/mistral-7b",
                "together",
                "mistral-7b",
                "tog_key",
                None
            ),
            (
                "ai://seek_key:@deepseek/deepseek-chat",
                "deepseek",
                "deepseek-chat",
                "seek_key",
                None
            ),
            (
                "ai://seek_key@deepseek/deepseek-chat",
                "deepseek",
                "deepseek-chat",
                "seek_key",
                None
            ),
            (
                "ai://groq_key:@groq/llama3-8b-8192",
                "groq",
                "llama3-8b-8192",
                "groq_key",
                None
            ),
            (
                "ai://groq_key@groq/llama3-8b-8192",
                "groq",
                "llama3-8b-8192",
                "groq_key",
                None
            ),
            (
                "ai://hf_key:@huggingface/meta-llama/Llama-3-8B-Instruct",
                "huggingface",
                "meta-llama/Llama-3-8B-Instruct",
                "hf_key",
                None
            ),
            (
                "ai://hf_key@huggingface/meta-llama/Llama-3-8B-Instruct",
                "huggingface",
                "meta-llama/Llama-3-8B-Instruct",
                "hf_key",
                None
            ),
            (
                "ai://secret:openai_key@openai/gpt-4o",
                "openai",
                "gpt-4o",
                "secret",
                "openai_key"
            ),
            (
                "ai://ollama/llama3",
                "ollama",
                "llama3",
                None,
                None
            ),
            (
                "ai://litellm/gpt-3.5-turbo",
                "litellm",
                "gpt-3.5-turbo",
                None,
                None
            )
        ]

    def test_parse_uri(self):
        hub_mock = MagicMock(spec=HuggingfaceHub)
        logger_mock = MagicMock(spec=Logger)
        with ModelManager(hub_mock, logger_mock) as manager:
            for fixture_uri in self.fixture_uris:
                uri, vendor, model_id, user, password = fixture_uri
                with self.subTest():
                    result = manager.parse_uri(uri)
                    self.assertIsInstance(result, EngineUri)
                    if not vendor:
                        self.assertTrue(result.is_local)
                        self.assertIsNone(result.vendor)
                        self.assertEqual(model_id, result.model_id)
                        self.assertIsNone(result.host)
                        self.assertIsNone(result.port)
                        self.assertIsNone(result.user)
                        self.assertEqual(password, result.password)
                    else:
                        self.assertFalse(result.is_local)
                        self.assertEqual(vendor, result.vendor)
                        self.assertEqual(model_id, result.model_id)
                        self.assertEqual(vendor, result.host)
                        self.assertIsNone(result.port)
                        self.assertEqual(user, result.user)
                        self.assertEqual(password, result.password)
        logger_mock.assert_not_called()


class ManagerEngineSettingsTestCase(TestCase):
    def setUp(self):
        self.hub_mock = MagicMock(spec=HuggingfaceHub)
        self.logger_mock = MagicMock(spec=Logger)
        self.secrets_mock = MagicMock(spec=KeyringSecrets)

    def test_local_uri_does_not_add_token(self):
        with ModelManager(self.hub_mock, self.logger_mock, self.secrets_mock) as manager:
            uri = manager.parse_uri("ai://local/llama3")
            settings = manager.get_engine_settings(uri, {"device": "cpu"})
            self.assertIsInstance(settings, TransformerEngineSettings)
            self.assertIsNone(settings.access_token)
            self.assertEqual(settings.device, "cpu")
        self.secrets_mock.read.assert_not_called()

    def test_user_token_added(self):
        with ModelManager(self.hub_mock, self.logger_mock, self.secrets_mock) as manager:
            uri = manager.parse_uri("ai://tok@openai/gpt-4o")
            settings = manager.get_engine_settings(uri)
            self.assertEqual(settings.access_token, "tok")
        self.secrets_mock.read.assert_not_called()

    def test_secret_token_lookup(self):
        self.secrets_mock.read.return_value = "val"
        with ModelManager(self.hub_mock, self.logger_mock, self.secrets_mock) as manager:
            uri = manager.parse_uri("ai://secret:pass@openai/gpt-4o")
            settings = manager.get_engine_settings(uri)
            self.assertEqual(settings.access_token, "val")
        self.secrets_mock.read.assert_called_once_with("pass")

    def test_sentence_transformer_no_token(self):
        with ModelManager(self.hub_mock, self.logger_mock, self.secrets_mock) as manager:
            uri = manager.parse_uri("ai://tok@openai/gpt-4o")
            settings = manager.get_engine_settings(uri, is_sentence_transformer=True)
            self.assertIsNone(settings.access_token)
        self.secrets_mock.read.assert_not_called()


class ManagerLoadEngineTestCase(TestCase):
    def setUp(self):
        self.hub_mock = MagicMock(spec=HuggingfaceHub)
        self.logger_mock = MagicMock(spec=Logger)

    def test_load_engine_per_vendor(self):
        vendor_data = {
            "local": (
                "ai://local/tiiuae/Falcon-E-3B-Instruct",
                "avalan.model.manager.TextGenerationModel",
                "tiiuae/Falcon-E-3B-Instruct",
                False,
            ),
            "sentence": (
                "ai://local/tiiuae/Falcon-E-3B-Instruct",
                "avalan.model.manager.SentenceTransformerModel",
                "tiiuae/Falcon-E-3B-Instruct",
                True,
            ),
            "openai": (
                "ai://tok@openai/gpt-4o",
                "avalan.model.nlp.text.vendor.openai.OpenAIModel",
                "gpt-4o",
                False,
            ),
            "openrouter": (
                "ai://router@openrouter/gpt-3.5-turbo",
                "avalan.model.nlp.text.vendor.openrouter.OpenRouterModel",
                "gpt-3.5-turbo",
                False,
            ),
            "anyscale": (
                "ai://as@anyscale/gpt-3.5-turbo",
                "avalan.model.nlp.text.vendor.anyscale.AnyScaleModel",
                "gpt-3.5-turbo",
                False,
            ),
            "together": (
                "ai://tg@together/mistral-7b",
                "avalan.model.nlp.text.vendor.together.TogetherModel",
                "mistral-7b",
                False,
            ),
            "deepseek": (
                "ai://d@deepseek/deepseek-chat",
                "avalan.model.nlp.text.vendor.deepseek.DeepSeekModel",
                "deepseek-chat",
                False,
            ),
            "groq": (
                "ai://g@groq/llama3-8b-8192",
                "avalan.model.nlp.text.vendor.groq.GroqModel",
                "llama3-8b-8192",
                False,
            ),
            "ollama": (
                "ai://ollama/llama3",
                "avalan.model.nlp.text.vendor.ollama.OllamaModel",
                "llama3",
                False,
            ),
            "litellm": (
                "ai://litellm/gpt-3.5-turbo",
                "avalan.model.nlp.text.vendor.litellm.LiteLLMModel",
                "gpt-3.5-turbo",
                False,
            ),
            "huggingface": (
                "ai://hf@huggingface/meta-llama/Llama-3-8B-Instruct",
                "avalan.model.nlp.text.vendor.huggingface.HuggingfaceModel",
                "meta-llama/Llama-3-8B-Instruct",
                False,
            ),
        }

        for vendor, (uri, path, model_id, is_sentence) in vendor_data.items():
            with self.subTest(vendor=vendor):
                with ModelManager(self.hub_mock, self.logger_mock) as manager:
                    engine_uri = manager.parse_uri(uri)
                    settings = TransformerEngineSettings()
                    manager._stack.enter_context = MagicMock()
                    if path.startswith("avalan.model.manager"):
                        with patch(path) as Model:
                            result = manager.load_engine(
                                engine_uri,
                                settings,
                                is_sentence_transformer=is_sentence,
                            )
                            Model.assert_called_once_with(
                                model_id=model_id,
                                settings=settings,
                                logger=self.logger_mock,
                            )
                            manager._stack.enter_context.assert_called_once_with(Model.return_value)
                            self.assertIs(result, Model.return_value)
                    else:
                        module_path, class_name = path.rsplit('.', 1)
                        dummy_module = types.SimpleNamespace()
                        Model = MagicMock()
                        setattr(dummy_module, class_name, Model)
                        with patch.dict(sys.modules, {module_path: dummy_module}):
                            result = manager.load_engine(
                                engine_uri,
                                settings,
                                is_sentence_transformer=is_sentence,
                            )

                        Model.assert_called_once_with(
                            model_id=model_id,
                            settings=settings,
                            logger=self.logger_mock,
                        )
                        manager._stack.enter_context.assert_called_once_with(Model.return_value)
                        self.assertIs(result, Model.return_value)

if __name__ == '__main__':
    main()
