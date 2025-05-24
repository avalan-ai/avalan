from avalan.model.entities import TransformerEngineSettings
from avalan.model.nlp.text.mlxlm import MlxLmModel
from logging import Logger
from pytest import importorskip
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

importorskip("mlx_lm", reason="mlx-lm not installed")


class MlxLmModelTestCase(TestCase):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    def test_instantiation_with_load_model_and_tokenizer(self):
        mlx_mock = MagicMock()
        model_instance = MagicMock()
        tokenizer_instance = MagicMock()
        mlx_mock.load.return_value = (model_instance, tokenizer_instance)
        mlx_mock.generate.return_value = ""

        with patch.dict("sys.modules", {"mlx_lm": mlx_mock}):
            logger_mock = MagicMock(spec=Logger)
            model = MlxLmModel(
                self.model_id,
                TransformerEngineSettings(),
                logger=logger_mock,
            )
            self.assertIs(model._model, model_instance)
            self.assertIs(model._tokenizer, tokenizer_instance)
            mlx_mock.load.assert_called_once_with(self.model_id)


if __name__ == "__main__":
    main()

