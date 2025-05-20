from avalan.model.hubs.huggingface import HuggingfaceHub
from logging import Logger
from unittest import main, TestCase
from unittest.mock import MagicMock

class HuggingfaceTestCase(TestCase):
    def test_instantiation(self):
        logger_mock = MagicMock(spec=Logger)
        hub = HuggingfaceHub(
            access_token="access_token",
            cache_dir="/tmp",
            logger=logger_mock
        )
        self.assertIsInstance(hub, HuggingfaceHub)
        logger_mock.assert_not_called()

if __name__ == '__main__':
    main()
