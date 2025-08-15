from .registry import ModalityRegistry
from ..audio.classification import AudioClassificationModel
from ..audio.generation import AudioGenerationModel
from ..audio.speech import TextToSpeechModel
from ..audio.speech_recognition import SpeechRecognitionModel
from ...entities import (
    EngineUri,
    GenerationSettings,
    Input,
    Modality,
    Operation,
    OperationAudioParameters,
    OperationParameters,
    TransformerEngineSettings,
)
from ...tool.manager import ToolManager

from argparse import Namespace
from logging import Logger
from typing import Any


@ModalityRegistry.register(Modality.AUDIO_CLASSIFICATION)
class AudioClassificationModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> AudioClassificationModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return AudioClassificationModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            audio=OperationAudioParameters(
                path=args.path,
                sampling_rate=args.audio_sampling_rate,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.AUDIO_CLASSIFICATION,
            parameters=parameters,
            requires_input=False,
        )

    async def __call__(
        self,
        engine_uri: EngineUri,
        model: AudioClassificationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["audio"]
            and operation.parameters["audio"].path
            and operation.parameters["audio"].sampling_rate
        )

        return await model(
            path=operation.parameters["audio"].path,
            sampling_rate=operation.parameters["audio"].sampling_rate,
        )


@ModalityRegistry.register(Modality.AUDIO_SPEECH_RECOGNITION)
class AudioSpeechRecognitionModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> SpeechRecognitionModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return SpeechRecognitionModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            audio=OperationAudioParameters(
                path=args.path,
                sampling_rate=args.audio_sampling_rate,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.AUDIO_SPEECH_RECOGNITION,
            parameters=parameters,
            requires_input=False,
        )

    async def __call__(
        self,
        engine_uri: EngineUri,
        model: SpeechRecognitionModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["audio"]
            and operation.parameters["audio"].path
            and operation.parameters["audio"].sampling_rate
        )

        return await model(
            path=operation.parameters["audio"].path,
            sampling_rate=operation.parameters["audio"].sampling_rate,
        )


@ModalityRegistry.register(Modality.AUDIO_TEXT_TO_SPEECH)
class AudioTextToSpeechModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> TextToSpeechModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return TextToSpeechModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            audio=OperationAudioParameters(
                path=args.path,
                reference_path=args.audio_reference_path,
                reference_text=args.audio_reference_text,
                sampling_rate=args.audio_sampling_rate,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.AUDIO_TEXT_TO_SPEECH,
            parameters=parameters,
            requires_input=True,
        )

    async def __call__(
        self,
        engine_uri: EngineUri,
        model: TextToSpeechModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.parameters["audio"]
            and operation.parameters["audio"].path
            and operation.parameters["audio"].sampling_rate
        )

        return await model(
            path=operation.parameters["audio"].path,
            prompt=operation.input,
            max_new_tokens=operation.generation_settings.max_new_tokens,
            reference_path=operation.parameters["audio"].reference_path,
            reference_text=operation.parameters["audio"].reference_text,
            sampling_rate=operation.parameters["audio"].sampling_rate,
        )


@ModalityRegistry.register(Modality.AUDIO_GENERATION)
class AudioGenerationModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> AudioGenerationModel:
        if not engine_uri.is_local:
            raise NotImplementedError()
        return AudioGenerationModel(
            model_id=engine_uri.model_id,
            settings=engine_settings,
            logger=logger,
        )

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            audio=OperationAudioParameters(
                path=args.path,
                sampling_rate=args.audio_sampling_rate,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.AUDIO_GENERATION,
            parameters=parameters,
            requires_input=True,
        )

    async def __call__(
        self,
        engine_uri: EngineUri,
        model: AudioGenerationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.input
            and operation.parameters["audio"]
            and operation.parameters["audio"].path
        )

        return await model(
            operation.input,
            operation.parameters["audio"].path,
            operation.generation_settings.max_new_tokens,
        )
