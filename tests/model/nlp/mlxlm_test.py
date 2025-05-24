from avalan.model.entities import TransformerEngineSettings
from avalan.model.nlp.text.mlxlm import MlxLmModel
from logging import Logger
from pytest import importorskip
from unittest import TestCase, main, skip
from unittest.mock import MagicMock, patch

importorskip("mlx_lm", reason="mlx-lm not installed")

class MlxLmModelTestCase(TestCase):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    @skip("Patching MLXLm.load is not working out")
    @patch("avalan.model.nlp.text.mlxlm")
    def test_instantiation_with_load_model_and_tokenizer(self, MLXLm):
        model_instance = MagicMock()
        tokenizer_instance = MagicMock()

        MLXLm.load.return_value = (model_instance, tokenizer_instance)
        MLXLm.generate.return_value = ""

        logger_mock = MagicMock(spec=Logger)
        model = MlxLmModel(
            self.model_id,
            TransformerEngineSettings(),
            logger=logger_mock,
        )
        self.assertIsNotNone(model)
        #self.assertIs(model._model, model_instance)
        #self.assertIs(model._tokenizer, tokenizer_instance)

        #MLXLm.load.assert_called_once_with(self.model_id)

if __name__ == "__main__":
    main()

