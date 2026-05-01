from contextlib import nullcontext
from logging import Logger
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import MagicMock, patch

from transformers import PreTrainedModel

from avalan.entities import EngineSettings
from avalan.model.audio.speech import (
    AutoFeatureExtractor,
    AutoTokenizer,
    DiaConfig,
    DiaForConditionalGeneration,
    DiaProcessor,
    TextToSpeechModel,
)
from avalan.model.engine import Engine


class TextToSpeechModelInstantiationTestCase(TestCase):
    model_id = "dummy/model"

    def test_instantiation_no_load(self):
        logger_mock = MagicMock(spec=Logger)
        with (
            patch.object(TextToSpeechModel, "_load_dia_config") as config_mock,
            patch.object(
                TextToSpeechModel, "_load_processor"
            ) as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
        ):
            settings = EngineSettings(auto_load_model=False)
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )
            self.assertIsInstance(model, TextToSpeechModel)
            config_mock.assert_not_called()
            processor_mock.assert_not_called()
            model_mock.assert_not_called()

    def test_instantiation_with_load_model(self):
        logger_mock = MagicMock(spec=Logger)
        config = MagicMock()
        with (
            patch.object(
                TextToSpeechModel, "_load_dia_config", return_value=config
            ) as config_mock,
            patch.object(
                TextToSpeechModel, "_load_processor"
            ) as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )
            self.assertIs(model.model, model_instance)
            config_mock.assert_called_once_with()
            processor_mock.assert_called_once_with(config)
            model_mock.assert_called_once_with(
                self.model_id,
                config=config,
                trust_remote_code=False,
                device_map=Engine.get_default_device(),
                tp_plan=None,
                distributed_config=None,
                subfolder="",
            )

    def test_normalize_dia_config_dict_moves_token_ids(self):
        config_dict = {
            "bos_token_id": 1026,
            "decoder_config": {"bos_token_id": 1, "vocab_size": 1028},
            "eos_token_id": 1024,
            "pad_token_id": 1025,
        }

        normalized = TextToSpeechModel._normalize_dia_config_dict(config_dict)

        self.assertNotIn("bos_token_id", normalized)
        self.assertNotIn("eos_token_id", normalized)
        self.assertNotIn("pad_token_id", normalized)
        self.assertEqual(normalized["decoder_config"]["bos_token_id"], 1026)
        self.assertEqual(normalized["decoder_config"]["eos_token_id"], 1024)
        self.assertEqual(normalized["decoder_config"]["pad_token_id"], 1025)
        self.assertEqual(
            config_dict["decoder_config"]["bos_token_id"],
            1,
        )

    def test_normalize_dia_config_dict_adds_decoder_config(self):
        normalized = TextToSpeechModel._normalize_dia_config_dict({})

        self.assertEqual(normalized["decoder_config"], {})

    def test_normalize_dia_config_dict_requires_decoder_config_mapping(self):
        with self.assertRaises(AssertionError):
            TextToSpeechModel._normalize_dia_config_dict(
                {"decoder_config": "bad"}
            )

    def test_sync_dia_config_token_ids(self):
        config = DiaConfig(
            decoder_config={
                "bos_token_id": 3,
                "eos_token_id": 2,
                "pad_token_id": 1,
            }
        )

        TextToSpeechModel._sync_dia_config_token_ids(config)

        self.assertEqual(config.bos_token_id, 3)
        self.assertEqual(config.eos_token_id, 2)
        self.assertEqual(config.pad_token_id, 1)

    def test_load_dia_config_normalizes_checkpoint_config(self):
        logger_mock = MagicMock(spec=Logger)
        config_dict = {
            "bos_token_id": 3,
            "decoder_config": {},
            "eos_token_id": 2,
            "pad_token_id": 1,
        }
        settings = EngineSettings(auto_load_model=False)
        model = TextToSpeechModel(self.model_id, settings, logger=logger_mock)

        with patch.object(
            DiaConfig,
            "get_config_dict",
            return_value=(config_dict, {}),
        ) as get_config_dict_mock:
            config = model._load_dia_config()

        get_config_dict_mock.assert_called_once_with(
            self.model_id,
            subfolder="",
        )
        self.assertEqual(config.bos_token_id, 3)
        self.assertEqual(config.decoder_config.bos_token_id, 3)
        self.assertEqual(config.eos_token_id, 2)
        self.assertEqual(config.decoder_config.eos_token_id, 2)
        self.assertEqual(config.pad_token_id, 1)
        self.assertEqual(config.decoder_config.pad_token_id, 1)

    def test_load_processor_passes_config_to_tokenizer(self):
        logger_mock = MagicMock(spec=Logger)
        config = MagicMock(spec=DiaConfig)
        feature_extractor = MagicMock()
        processor = MagicMock(spec=DiaProcessor)
        tokenizer = MagicMock()
        settings = EngineSettings(
            access_token="token",
            auto_load_model=False,
            cache_dir="cache",
            revision="rev",
            tokenizer_subfolder="tokenizer",
            trust_remote_code=True,
        )
        model = TextToSpeechModel(self.model_id, settings, logger=logger_mock)
        processor_dict = {"audio_tokenizer": "dac"}
        processor_kwargs = {
            "cache_dir": "cache",
            "revision": "rev",
            "subfolder": "tokenizer",
            "token": "token",
            "trust_remote_code": True,
        }

        with (
            patch.object(
                DiaProcessor,
                "get_processor_dict",
                return_value=(processor_dict, {"unused": "ok"}),
            ) as get_processor_dict_mock,
            patch.object(
                AutoFeatureExtractor,
                "from_pretrained",
                return_value=feature_extractor,
            ) as feature_extractor_mock,
            patch.object(
                AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_mock,
            patch.object(
                DiaProcessor,
                "from_args_and_dict",
                return_value=processor,
            ) as from_args_mock,
        ):
            result = model._load_processor(config)

        self.assertIs(result, processor)
        get_processor_dict_mock.assert_called_once_with(
            self.model_id,
            **processor_kwargs,
        )
        feature_extractor_mock.assert_called_once_with(
            self.model_id,
            **processor_kwargs,
        )
        tokenizer_mock.assert_called_once_with(
            self.model_id,
            config=config,
            **processor_kwargs,
        )
        from_args_mock.assert_called_once_with(
            [feature_extractor, tokenizer],
            processor_dict,
            unused="ok",
        )


class TextToSpeechModelCallTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_call(self):
        logger_mock = MagicMock(spec=Logger)
        config = MagicMock()
        with (
            patch.object(
                TextToSpeechModel, "_load_dia_config", return_value=config
            ),
            patch.object(
                TextToSpeechModel, "_load_processor"
            ) as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch(
                "avalan.model.audio.speech.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
        ):
            call_result = MagicMock()
            inputs = {"input_ids": [1]}
            call_result.to.return_value = inputs
            processor_instance = MagicMock(return_value=call_result)
            processor_instance.batch_decode.return_value = ["audio"]
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_instance.generate = MagicMock(return_value=[2])
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model(["hi"], "file.wav", 3, padding=False)

            self.assertEqual(result, "file.wav")
            processor_instance.assert_called_with(
                text=["hi"],
                audio=None,
                padding=False,
                return_tensors="pt",
                sampling_rate=44_100,
            )
            call_result.to.assert_called_once_with(model._device)
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=3
            )
            processor_instance.batch_decode.assert_called_once_with([2])
            processor_instance.save_audio.assert_called_once_with(
                ["audio"], "file.wav"
            )
            inf_mock.assert_called_once_with()


class TextToSpeechModelReferenceTestCase(IsolatedAsyncioTestCase):
    model_id = "dummy/model"

    async def test_reference_validation(self):
        logger_mock = MagicMock(spec=Logger)
        config = MagicMock()
        with (
            patch.object(
                TextToSpeechModel, "_load_dia_config", return_value=config
            ),
            patch.object(
                TextToSpeechModel, "_load_processor"
            ) as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
        ):
            processor_instance = MagicMock()
            processor_mock.return_value = processor_instance
            model_instance = MagicMock(spec=PreTrainedModel)
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )

            with self.assertRaises(AssertionError):
                await model("hi", "file.wav", 3, reference_path="voice.wav")

    async def test_call_with_reference_resample(self):
        logger_mock = MagicMock(spec=Logger)
        config = MagicMock()
        with (
            patch.object(
                TextToSpeechModel, "_load_dia_config", return_value=config
            ),
            patch.object(
                TextToSpeechModel, "_load_processor"
            ) as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch.object(TextToSpeechModel, "_resample") as resample_method,
            patch(
                "avalan.model.audio.speech.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
        ):
            call_result = MagicMock()
            inputs = {"input_ids": [1], "decoder_attention_mask": "mask"}
            call_result.to.return_value = inputs
            processor_instance = MagicMock(return_value=call_result)
            processor_instance.batch_decode.return_value = ["audio"]
            processor_instance.get_audio_prompt_len.return_value = 2
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            outputs = MagicMock(shape=(1, 5))
            model_instance.generate = MagicMock(return_value=outputs)
            model_mock.return_value = model_instance

            resample_method.return_value = "voice"

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model(
                "hi",
                "file.wav",
                3,
                reference_path="ref.wav",
                reference_text="ref",
                sampling_rate=16000,
            )

            self.assertEqual(result, "file.wav")
            resample_method.assert_called_once_with("ref.wav", 16000)
            processor_instance.assert_called_with(
                text="ref\nhi",
                audio="voice",
                padding=True,
                return_tensors="pt",
                sampling_rate=16000,
            )
            call_result.to.assert_called_once_with(model._device)
            processor_instance.get_audio_prompt_len.assert_called_once_with(
                inputs["decoder_attention_mask"]
            )
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=3
            )
            processor_instance.batch_decode.assert_called_once_with(
                outputs, audio_prompt_len=2
            )
            processor_instance.save_audio.assert_called_once_with(
                ["audio"], "file.wav"
            )
            inf_mock.assert_called_once_with()

    async def test_call_with_reference_no_resample(self):
        logger_mock = MagicMock(spec=Logger)
        config = MagicMock()
        with (
            patch.object(
                TextToSpeechModel, "_load_dia_config", return_value=config
            ),
            patch.object(
                TextToSpeechModel, "_load_processor"
            ) as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch.object(TextToSpeechModel, "_resample") as resample_method,
            patch(
                "avalan.model.audio.speech.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
        ):
            call_result = MagicMock()
            inputs = {"input_ids": [1], "decoder_attention_mask": "mask"}
            call_result.to.return_value = inputs
            processor_instance = MagicMock(return_value=call_result)
            processor_instance.batch_decode.return_value = ["audio"]
            processor_instance.get_audio_prompt_len.return_value = 2
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            outputs = MagicMock(shape=(1, 1))
            model_instance.generate = MagicMock(return_value=outputs)
            model_mock.return_value = model_instance

            resample_method.return_value = "voice"

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model(
                "hi",
                "file.wav",
                3,
                reference_path="ref.wav",
                reference_text="ref",
                sampling_rate=16000,
            )

            self.assertEqual(result, "file.wav")
            resample_method.assert_called_once_with("ref.wav", 16000)
            processor_instance.assert_called_with(
                text="ref\nhi",
                audio="voice",
                padding=True,
                return_tensors="pt",
                sampling_rate=16000,
            )
            call_result.to.assert_called_once_with(model._device)
            processor_instance.get_audio_prompt_len.assert_called_once_with(
                inputs["decoder_attention_mask"]
            )
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=3
            )
            processor_instance.batch_decode.assert_called_once_with(outputs)
            processor_instance.save_audio.assert_called_once_with(
                ["audio"], "file.wav"
            )
            inf_mock.assert_called_once_with()

    async def test_call_sampling_rate(self):
        logger_mock = MagicMock(spec=Logger)
        config = MagicMock()
        with (
            patch.object(
                TextToSpeechModel, "_load_dia_config", return_value=config
            ),
            patch.object(
                TextToSpeechModel, "_load_processor"
            ) as processor_mock,
            patch.object(
                DiaForConditionalGeneration, "from_pretrained"
            ) as model_mock,
            patch(
                "avalan.model.audio.speech.inference_mode",
                return_value=nullcontext(),
            ) as inf_mock,
        ):
            call_result = MagicMock()
            inputs = {"input_ids": [1]}
            call_result.to.return_value = inputs
            processor_instance = MagicMock(return_value=call_result)
            processor_instance.batch_decode.return_value = ["audio"]
            processor_mock.return_value = processor_instance

            model_instance = MagicMock(spec=PreTrainedModel)
            model_instance.generate = MagicMock(return_value=[2])
            model_mock.return_value = model_instance

            settings = EngineSettings()
            model = TextToSpeechModel(
                self.model_id, settings, logger=logger_mock
            )

            result = await model(
                ["hi"], "file.wav", 3, padding=False, sampling_rate=16000
            )

            self.assertEqual(result, "file.wav")
            processor_instance.assert_called_with(
                text=["hi"],
                audio=None,
                padding=False,
                return_tensors="pt",
                sampling_rate=16000,
            )
            call_result.to.assert_called_once_with(model._device)
            model_instance.generate.assert_called_once_with(
                **inputs, max_new_tokens=3
            )
            processor_instance.batch_decode.assert_called_once_with([2])
            processor_instance.save_audio.assert_called_once_with(
                ["audio"], "file.wav"
            )
            inf_mock.assert_called_once_with()


if __name__ == "__main__":
    main()
