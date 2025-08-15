from .registry import ModalityRegistry
from ..audio.classification import AudioClassificationModel
from ..audio.generation import AudioGenerationModel
from ..audio.speech import TextToSpeechModel
from ..audio.speech_recognition import SpeechRecognitionModel
from ...entities import EngineUri, Modality, Operation
from ...tool.manager import ToolManager

from typing import Any


@ModalityRegistry.register(Modality.AUDIO_CLASSIFICATION)
class AudioClassificationModality:
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
