from .modalities import ModalityRegistry
from ..entities import (
    AttentionImplementation,
    Backend,
    EngineUri,
    Input,
    Modality,
    Operation,
    ParallelStrategy,
    TextGenerationLoaderClass,
    TransformerEngineSettings,
    Vendor,
    WeightType,
)
from ..event import Event, EventType
from ..event.manager import EventManager
from ..model.audio.classification import AudioClassificationModel
from ..model.audio.generation import AudioGenerationModel
from ..model.audio.speech import TextToSpeechModel
from ..model.audio.speech_recognition import SpeechRecognitionModel
from ..model.hubs.huggingface import HuggingfaceHub
from ..model.nlp.question import QuestionAnsweringModel
from ..model.nlp.sentence import SentenceTransformerModel
from ..model.nlp.sequence import (
    SequenceClassificationModel,
    SequenceToSequenceModel,
    TranslationModel,
)
from ..model.nlp.text.generation import TextGenerationModel
from ..model.nlp.token import TokenClassificationModel
from ..model.vision.classification import ImageClassificationModel
from ..model.vision.decoder import VisionEncoderDecoderModel
from ..model.vision.detection import ObjectDetectionModel
from ..model.vision.diffusion import (
    TextToAnimationModel,
    TextToImageModel,
    TextToVideoModel,
)
from ..model.vision.segmentation import SemanticSegmentationModel
from ..model.vision.text import ImageTextToTextModel, ImageToTextModel
from ..secrets import KeyringSecrets
from ..tool.manager import ToolManager
from argparse import Namespace
from contextlib import ContextDecorator, ExitStack
from logging import Logger
from time import perf_counter
from typing import Any, TypeAlias, get_args
from urllib.parse import parse_qsl, urlparse

ModelType: TypeAlias = (
    AudioClassificationModel
    | AudioGenerationModel
    | ImageClassificationModel
    | ImageTextToTextModel
    | ImageToTextModel
    | ObjectDetectionModel
    | QuestionAnsweringModel
    | SemanticSegmentationModel
    | SentenceTransformerModel
    | SequenceClassificationModel
    | SequenceToSequenceModel
    | SpeechRecognitionModel
    | TextGenerationModel
    | TextToAnimationModel
    | TextToImageModel
    | TextToSpeechModel
    | TextToVideoModel
    | TokenClassificationModel
    | TranslationModel
    | VisionEncoderDecoderModel
)


class ModelManager(ContextDecorator):
    _hub: HuggingfaceHub
    _stack: ExitStack
    _logger: Logger
    _secrets: KeyringSecrets
    _event_manager: EventManager | None

    def __init__(
        self,
        hub: HuggingfaceHub,
        logger: Logger,
        secrets: KeyringSecrets | None = None,
        event_manager: EventManager | None = None,
    ):
        self._hub, self._logger = hub, logger
        self._stack = ExitStack()
        self._secrets = secrets or KeyringSecrets()
        self._event_manager = event_manager

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ):
        return self._stack.__exit__(exc_type, exc_value, traceback)

    async def __call__(
        self,
        engine_uri: EngineUri,
        model: ModelType,
        operation: Operation,
        tool: ToolManager | None = None,
    ):
        modality = operation.modality

        self._logger.info("ModelManager call process started for %s", modality)

        start = perf_counter()
        if self._event_manager:
            await self._event_manager.trigger(
                Event(
                    type=EventType.MODEL_MANAGER_CALL_BEFORE,
                    payload={
                        "engine_uri": engine_uri,
                        "modality": modality,
                        "operation": operation,
                    },
                    started=start,
                )
            )

        handler = ModalityRegistry.get(modality)
        result = await handler(engine_uri, model, operation, tool)

        end = perf_counter()
        if self._event_manager:
            await self._event_manager.trigger(
                Event(
                    type=EventType.MODEL_MANAGER_CALL_AFTER,
                    payload={
                        "engine_uri": engine_uri,
                        "modality": modality,
                        "operation": operation,
                        "result": result,
                    },
                    started=start,
                    finished=end,
                    elapsed=end - start,
                )
            )

        self._logger.info("ModelManager call processed for %s", modality)

        return result

    @staticmethod
    def get_operation_from_arguments(
        modality: Modality,
        args: Namespace,
        input_string: Input | None,
    ) -> Operation:
        return ModalityRegistry.get_operation_from_arguments(
            modality, args, input_string
        )

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
        backend: Backend = Backend.TRANSFORMERS,
        low_cpu_mem_usage: bool = False,
        parallel: ParallelStrategy | None = None,
        quiet: bool = False,
        output_hidden_states: bool | None = None,
        base_model_id: str | None = None,
        checkpoint: str | None = None,
        refiner_model_id: str | None = None,
        upsampler_model_id: str | None = None,
        revision: str | None = None,
        special_tokens: list[str] | None = None,
        subfolder: str | None = None,
        tokenizer: str | None = None,
        tokenizer_subfolder: str | None = None,
        tokens: list[str] | None = None,
        trust_remote_code: bool | None = None,
        weight_type: WeightType = "auto",
    ) -> ModelType:
        engine_settings_args = dict(
            base_url=base_url,
            cache_dir=self._hub.cache_dir,
            device=device,
            disable_loading_progress_bar=quiet or disable_loading_progress_bar,
            low_cpu_mem_usage=low_cpu_mem_usage,
            loader_class=loader_class,
            backend=backend,
            parallel=parallel,
            base_model_id=base_model_id or None,
            checkpoint=checkpoint or None,
            refiner_model_id=refiner_model_id or None,
            upsampler_model_id=upsampler_model_id or None,
            revision=revision,
            special_tokens=special_tokens or None,
            subfolder=subfolder or None,
            tokenizer_name_or_path=tokenizer,
            tokenizer_subfolder=tokenizer_subfolder or None,
            tokens=tokens or None,
            weight_type=weight_type,
        )

        if output_hidden_states is not None:
            engine_settings_args["output_hidden_states"] = output_hidden_states

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
    ) -> ModelType:
        if modality is Modality.EMBEDDING:
            from ..model.nlp.sentence import SentenceTransformerModel

            model = SentenceTransformerModel(
                model_id=engine_uri.model_id,
                settings=engine_settings,
                logger=self._logger,
            )
        else:
            model = ModalityRegistry.load_engine(
                engine_uri, engine_settings, modality, self._logger
            )
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
