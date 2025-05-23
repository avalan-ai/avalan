from avalan.model.entities import EngineUri
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager
from logging import Logger
from unittest import main, TestCase
from unittest.mock import MagicMock

class ManagerTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_uris = [
            (
                "tiiuae/Falcon-E-3B-Instruct",
                None,
                "tiiuae/Falcon-E-3B-Instruct",
                None
            ),
            (
                "ai://local/tiiuae/Falcon-E-3B-Instruct",
                None,
                "tiiuae/Falcon-E-3B-Instruct",
                None
            ),
            (
                "meta-llama/Meta-Llama-3-8B-Instruct",
                None,
                "meta-llama/Meta-Llama-3-8B-Instruct",
                None
            ),
            (
                "ai://local/meta-llama/Meta-Llama-3-8B-Instruct",
                None,
                "meta-llama/Meta-Llama-3-8B-Instruct",
                None
            ),
            (
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                None,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                None
            ),
            (
                "ai://local/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                None,
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                None
            ),
            (
                "Qwen/Qwen2.5-7B-Instruct",
                None,
                "Qwen/Qwen2.5-7B-Instruct",
                None
            ),
            (
                "ai://local/Qwen/Qwen2.5-7B-Instruct",
                None,
                "Qwen/Qwen2.5-7B-Instruct",
                None
            ),
            (
                "ai://messi_api_key:@openai/gpt-4o",
                "openai",
                "gpt-4o",
                "messi_api_key"
            ),
            (
                "ai://messi_api_key@openai/gpt-4o",
                "openai",
                "gpt-4o",
                "messi_api_key"
            ),
            (
                "ai://router_api_key:@openrouter/gpt-3.5-turbo",
                "openrouter",
                "gpt-3.5-turbo",
                "router_api_key"
            ),
            (
                "ai://router_api_key@openrouter/gpt-3.5-turbo",
                "openrouter",
                "gpt-3.5-turbo",
                "router_api_key"
            ),
            (
                "ai://seek_key:@deepseek/deepseek-chat",
                "deepseek",
                "deepseek-chat",
                "seek_key"
            ),
            (
                "ai://seek_key@deepseek/deepseek-chat",
                "deepseek",
                "deepseek-chat",
                "seek_key"
            ),
            (
                "ai://groq_key:@groq/llama3-8b-8192",
                "groq",
                "llama3-8b-8192",
                "groq_key"
            ),
            (
                "ai://groq_key@groq/llama3-8b-8192",
                "groq",
                "llama3-8b-8192",
                "groq_key"
            ),
            (
                "ai://ollama/llama3",
                "ollama",
                "llama3",
                None
            )
        ]

    def test_parse_uri(self):
        hub_mock = MagicMock(spec=HuggingfaceHub)
        logger_mock = MagicMock(spec=Logger)
        with ModelManager(hub_mock, logger_mock) as manager:
            for fixture_uri in self.fixture_uris:
                uri, vendor, model_id, user = fixture_uri
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
                        self.assertIsNone(result.password)
                    else:
                        self.assertFalse(result.is_local)
                        self.assertEqual(vendor, result.vendor)
                        self.assertEqual(model_id, result.model_id)
                        self.assertEqual(vendor, result.host)
                        self.assertIsNone(result.port)
                        self.assertEqual(user, result.user)
                        self.assertIsNone(result.password)
        logger_mock.assert_not_called()

if __name__ == '__main__':
    main()
