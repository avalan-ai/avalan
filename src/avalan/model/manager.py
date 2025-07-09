from ..entities import (
    AttentionImplementation,
    EngineUri,
    Modality,
    ParallelStrategy,
    TextGenerationLoaderClass,
    TransformerEngineSettings,
    Vendor,
    WeightType,
)
from ..model.hubs.huggingface import HuggingfaceHub
from ..model.nlp.sentence import SentenceTransformerModel
from ..model.nlp.text.generation import TextGenerationModel
from ..model.audio import SpeechRecognitionModel, TextToSpeechModel
from ..model.vision.detection import ObjectDetectionModel
from ..model.vision.image import (
    ImageClassificationModel,
    ImageTextToTextModel,
    VisionEncoderDecoderModel,
)
from ..model.vision.segmentation import SemanticSegmentationModel
from ..secrets import KeyringSecrets
from contextlib import ContextDecorator, ExitStack
from logging import Logger
from typing import Any, get_args
from urllib.parse import urlparse, parse_qsl


class ModelManager(ContextDecorator):
    _hub: HuggingfaceHub
    _stack: ExitStack
    _logger: Logger
    _secrets: KeyringSecrets

    def __init__(
        self,
        hub: HuggingfaceHub,
        logger: Logger,
        secrets: KeyringSecrets | None = None,
    ):
        self._hub, self._logger = hub, logger
        self._stack = ExitStack()
        self._secrets = secrets or KeyringSecrets()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ):
        return self._stack.__exit__(exc_type, exc_value, traceback)

    def get_engine_settings(
        self,
        engine_uri: EngineUri,
        settings: dict | None = None,
        modality: Modality | None = None,
    ) -> TransformerEngineSettings:
        engine_settings_args = settings or {}

        if modality != Modality.EMBEDDING and not engine_uri.is_local:
            token = None
            if engine_uri.password and engine_uri.user:
                if engine_uri.user == "secret":
                    token = self._secrets.read(engine_uri.password)
                else:
                    token = None
            elif engine_uri.user:
                token = engine_uri.user

            if token:
                engine_settings_args.update(access_token=token)

        engine_settings = TransformerEngineSettings(**engine_settings_args)
        return engine_settings

    def load(
        self,
        engine_uri: EngineUri,
        modality: Modality = Modality.TEXT_GENERATION,
        *args,
        attention: AttentionImplementation | None = None,
        base_url: str | None = None,
        device: str | None = None,
        disable_loading_progress_bar: bool = False,
        loader_class: TextGenerationLoaderClass | None = "auto",
        low_cpu_mem_usage: bool = False,
        parallel: ParallelStrategy | None = None,
        quiet: bool = False,
        revision: str | None = None,
        special_tokens: list[str] | None = None,
        tokenizer: str | None = None,
        tokens: list[str] | None = None,
        trust_remote_code: bool | None = None,
        weight_type: WeightType = "auto",
    ) -> (
        SentenceTransformerModel
        | TextGenerationModel
        | SpeechRecognitionModel
        | TextToSpeechModel
        | ObjectDetectionModel
        | ImageClassificationModel
        | ImageTextToTextModel
        | VisionEncoderDecoderModel
        | SemanticSegmentationModel
    ):
        engine_settings_args = dict(
            base_url=base_url,
            cache_dir=self._hub.cache_dir,
            device=device,
            disable_loading_progress_bar=quiet or disable_loading_progress_bar,
            low_cpu_mem_usage=low_cpu_mem_usage,
            loader_class=loader_class,
            parallel=parallel,
            revision=revision,
            special_tokens=special_tokens or None,
            tokenizer_name_or_path=tokenizer,
            tokens=tokens or None,
            weight_type=weight_type,
        )
        if modality != Modality.EMBEDDING:
            engine_settings_args.update(
                attention=attention or None,
                trust_remote_code=trust_remote_code or None,
            )

        engine_settings = self.get_engine_settings(
            engine_uri,
            engine_settings_args,
            modality=modality,
        )
        return self.load_engine(engine_uri, engine_settings, modality)

    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        modality: Modality = Modality.TEXT_GENERATION,
    ) -> (
        SentenceTransformerModel
        | TextGenerationModel
        | SpeechRecognitionModel
        | TextToSpeechModel
        | ObjectDetectionModel
        | ImageClassificationModel
        | ImageTextToTextModel
        | VisionEncoderDecoderModel
        | SemanticSegmentationModel
    ):
        assert isinstance(engine_uri, EngineUri)
        model_load_args = dict(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=self._logger,
        )

        # Load local model, or lazy-import per vendor
        if engine_uri.is_local:
            match modality:
                case Modality.EMBEDDING:
                    model = SentenceTransformerModel(**model_load_args)
                case Modality.AUDIO_SPEECH_RECOGNITION:
                    model = SpeechRecognitionModel(**model_load_args)
                case Modality.AUDIO_TEXT_TO_SPEECH:
                    model = TextToSpeechModel(**model_load_args)
                case Modality.VISION_OBJECT_DETECTION:
                    model = ObjectDetectionModel(**model_load_args)
                case Modality.VISION_IMAGE_CLASSIFICATION:
                    model = ImageClassificationModel(**model_load_args)
                case Modality.VISION_IMAGE_TO_TEXT:
                    model = ImageTextToTextModel(**model_load_args)
                case Modality.VISION_ENCODER_DECODER:
                    model = VisionEncoderDecoderModel(**model_load_args)
                case Modality.VISION_SEMANTIC_SEGMENTATION:
                    model = SemanticSegmentationModel(**model_load_args)
                case _:
                    model = TextGenerationModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "openai"
        ):
            from ..model.nlp.text.vendor.openai import OpenAIModel

            model = OpenAIModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "openrouter"
        ):
            from ..model.nlp.text.vendor.openrouter import OpenRouterModel

            model = OpenRouterModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "anyscale"
        ):
            from ..model.nlp.text.vendor.anyscale import AnyScaleModel

            model = AnyScaleModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "together"
        ):
            from ..model.nlp.text.vendor.together import TogetherModel

            model = TogetherModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "deepseek"
        ):
            from ..model.nlp.text.vendor.deepseek import DeepSeekModel

            model = DeepSeekModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "deepinfra"
        ):
            from ..model.nlp.text.vendor.deepinfra import DeepInfraModel

            model = DeepInfraModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "groq"
        ):
            from ..model.nlp.text.vendor.groq import GroqModel

            model = GroqModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "ollama"
        ):
            from ..model.nlp.text.vendor.ollama import OllamaModel

            model = OllamaModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "huggingface"
        ):
            from ..model.nlp.text.vendor.huggingface import HuggingfaceModel

            model = HuggingfaceModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "hyperbolic"
        ):
            from ..model.nlp.text.vendor.hyperbolic import HyperbolicModel

            model = HyperbolicModel(**model_load_args)
        elif (
            modality == Modality.TEXT_GENERATION
            and engine_uri.vendor == "litellm"
        ):
            from ..model.nlp.text.vendor.litellm import LiteLLMModel

            model = LiteLLMModel(**model_load_args)
        else:
            raise NotImplementedError()

        self._stack.enter_context(model)
        return model

    @staticmethod
    def parse_uri(uri: str) -> EngineUri:
        parsed = urlparse(uri)
        if not parsed.scheme:
            uri = f"ai://{uri}"
            parsed = urlparse(uri)

        if parsed.scheme != "ai":
            raise ValueError(
                f"Invalid scheme {parsed.scheme!r}, expected 'ai'"
            )

        vendor = parsed.hostname
        if not vendor or vendor not in get_args(Vendor) or vendor == "local":
            vendor = None
        use_host = bool(vendor)
        path_prefixed = parsed.path.startswith("/")
        params = dict(parse_qsl(parsed.query))

        # urlparse() normalizes hostname to lowercase, so keep original case
        authority = parsed.netloc.rsplit("@", 1)[-1]
        hostname = authority.split(":", 1)[0]

        model_id = (
            hostname + ("/" if path_prefixed else "")
            if not vendor and hostname != "local"
            else ""
        ) + (parsed.path[1:] if path_prefixed else parsed.path)
        engine_uri = EngineUri(
            vendor=vendor,
            host=hostname if use_host else None,
            port=(parsed.port or None) if use_host else None,
            user=parsed.username or None,
            password=parsed.password or None,
            model_id=model_id,
            params=params,
        )
        return engine_uri
