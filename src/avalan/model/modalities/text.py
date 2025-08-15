from .registry import ModalityRegistry
from ..criteria import KeywordStoppingCriteria
from ..nlp.question import QuestionAnsweringModel
from ..nlp.sequence import (
    SequenceClassificationModel,
    SequenceToSequenceModel,
    TranslationModel,
)
from ..nlp.text.generation import TextGenerationModel
from ..nlp.text.mlxlm import MlxLmModel
from ..nlp.token import TokenClassificationModel
from ...entities import EngineUri, Modality, Operation
from ...tool.manager import ToolManager

from typing import Any


def _stopping_criteria(
    operation: Operation, model: Any
) -> KeywordStoppingCriteria | None:
    text_params = (
        operation.parameters["text"] if operation.parameters else None
    )
    if text_params and text_params.stop_on_keywords:
        return KeywordStoppingCriteria(
            text_params.stop_on_keywords, model.tokenizer
        )
    return None


@ModalityRegistry.register(Modality.TEXT_GENERATION)
class TextGenerationModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: TextGenerationModel | MlxLmModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert operation.input and operation.parameters["text"]

        criteria = _stopping_criteria(operation, model)
        is_mlx = isinstance(model, MlxLmModel)
        if engine_uri.is_local and not is_mlx:
            return await model(
                operation.input,
                system_prompt=operation.parameters["text"].system_prompt,
                settings=operation.generation_settings,
                stopping_criterias=[criteria] if criteria else None,
                manual_sampling=operation.parameters["text"].manual_sampling,
                pick=operation.parameters["text"].pick_tokens,
                skip_special_tokens=operation.parameters[
                    "text"
                ].skip_special_tokens,
                tool=tool,
            )
        return await model(
            operation.input,
            system_prompt=operation.parameters["text"].system_prompt,
            settings=operation.generation_settings,
            tool=tool,
        )


@ModalityRegistry.register(Modality.TEXT_QUESTION_ANSWERING)
class TextQuestionAnsweringModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: QuestionAnsweringModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.input
            and operation.parameters["text"]
            and operation.parameters["text"].context
        )

        return await model(
            operation.input,
            context=operation.parameters["text"].context,
            system_prompt=operation.parameters["text"].system_prompt,
        )


@ModalityRegistry.register(Modality.TEXT_SEQUENCE_CLASSIFICATION)
class TextSequenceClassificationModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: SequenceClassificationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert operation.input
        return await model(operation.input)


@ModalityRegistry.register(Modality.TEXT_SEQUENCE_TO_SEQUENCE)
class TextSequenceToSequenceModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: SequenceToSequenceModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert operation.input and operation.parameters["text"]
        criteria = _stopping_criteria(operation, model)
        return await model(
            operation.input,
            settings=operation.generation_settings,
            stopping_criterias=[criteria] if criteria else None,
        )


@ModalityRegistry.register(Modality.TEXT_TOKEN_CLASSIFICATION)
class TextTokenClassificationModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: TokenClassificationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert operation.input and operation.parameters["text"]
        return await model(
            operation.input,
            labeled_only=operation.parameters["text"].labeled_only or False,
            system_prompt=operation.parameters["text"].system_prompt,
        )


@ModalityRegistry.register(Modality.TEXT_TRANSLATION)
class TextTranslationModality:
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: TranslationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert (
            operation.input
            and operation.parameters["text"]
            and operation.parameters["text"].language_source
            and operation.parameters["text"].language_destination
        )
        criteria = _stopping_criteria(operation, model)
        return await model(
            operation.input,
            source_language=operation.parameters["text"].language_source,
            destination_language=operation.parameters[
                "text"
            ].language_destination,
            settings=operation.generation_settings,
            stopping_criterias=[criteria] if criteria else None,
            skip_special_tokens=operation.parameters[
                "text"
            ].skip_special_tokens,
        )
