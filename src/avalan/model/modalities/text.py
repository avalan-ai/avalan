from ...entities import (
    Backend,
    EngineUri,
    GenerationSettings,
    Input,
    Modality,
    Operation,
    OperationParameters,
    OperationTextParameters,
    TransformerEngineSettings,
)
from ...tool.manager import ToolManager
from ..criteria import KeywordStoppingCriteria
from ..nlp.question import QuestionAnsweringModel
from ..nlp.sequence import (
    SequenceClassificationModel,
    SequenceToSequenceModel,
    TranslationModel,
)
from ..nlp.text.generation import TextGenerationModel
from ..nlp.token import TokenClassificationModel
from .registry import ModalityRegistry

from argparse import Namespace
from contextlib import AsyncExitStack
from functools import lru_cache
from importlib.util import find_spec
from logging import Logger
from typing import Any, cast


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


@lru_cache(maxsize=1)
def _get_mlx_model() -> type[TextGenerationModel] | None:
    if not find_spec("mlx_lm"):
        return None
    try:
        from ..nlp.text.mlxlm import MlxLmModel as loader
    except ModuleNotFoundError:
        return None
    return loader if loader.is_available() else None


@ModalityRegistry.register(Modality.TEXT_GENERATION)
class TextGenerationModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        exit_stack: AsyncExitStack,
    ) -> TextGenerationModel:
        assert engine_uri.model_id is not None
        if engine_uri.is_local:
            match engine_settings.backend:
                case Backend.MLXLM:
                    mlx_loader = _get_mlx_model()
                    if mlx_loader is None:
                        msg = (
                            "The mlx-lm dependency is not installed. "
                            "Install avalan[mlx] to enable the MLX backend."
                        )
                        raise ModuleNotFoundError(msg)

                    return mlx_loader(
                        model_id=engine_uri.model_id,
                        settings=engine_settings,
                        logger=logger,
                    )
                case Backend.VLLM:
                    from ..nlp.text.vllm import VllmModel

                    return VllmModel(
                        model_id=engine_uri.model_id,
                        settings=engine_settings,
                        logger=logger,
                    )
                case _:
                    return TextGenerationModel(
                        model_id=engine_uri.model_id,
                        settings=engine_settings,
                        logger=logger,
                    )
        match engine_uri.vendor:
            case "anthropic":
                from ..nlp.text.vendor.anthropic import (
                    AnthropicModel,
                )

                return AnthropicModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )

            case "openai":
                from ..nlp.text.vendor.openai import OpenAIModel

                return OpenAIModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "bedrock":
                from ..nlp.text.vendor.bedrock import BedrockModel

                return BedrockModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "openrouter":
                from ..nlp.text.vendor.openrouter import (
                    OpenRouterModel,
                )

                return OpenRouterModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "anyscale":
                from ..nlp.text.vendor.anyscale import AnyScaleModel

                return AnyScaleModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "together":
                from ..nlp.text.vendor.together import TogetherModel

                return TogetherModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "deepseek":
                from ..nlp.text.vendor.deepseek import DeepSeekModel

                return DeepSeekModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "deepinfra":
                from ..nlp.text.vendor.deepinfra import (
                    DeepInfraModel,
                )

                return DeepInfraModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "groq":
                from ..nlp.text.vendor.groq import GroqModel

                return GroqModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "ollama":
                from ..nlp.text.vendor.ollama import OllamaModel

                return OllamaModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "huggingface":
                from ..nlp.text.vendor.huggingface import (
                    HuggingfaceModel,
                )

                return HuggingfaceModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "hyperbolic":
                from ..nlp.text.vendor.hyperbolic import (
                    HyperbolicModel,
                )

                return HyperbolicModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
            case "litellm":
                from ..nlp.text.vendor.litellm import LiteLLMModel

                return LiteLLMModel(
                    model_id=engine_uri.model_id,
                    settings=engine_settings,
                    logger=logger,
                    exit_stack=exit_stack,
                )
        raise NotImplementedError()

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation:
        parameters = OperationParameters(
            text=OperationTextParameters(
                manual_sampling=cast(bool | None, args.display_tokens or 0),
                pick_tokens=(
                    10
                    if args.display_tokens and args.display_tokens > 0
                    else 0
                ),
                stop_on_keywords=args.stop_on_keyword,
                skip_special_tokens=args.quiet or args.skip_special_tokens,
                system_prompt=args.system or None,
                developer_prompt=getattr(args, "developer", None) or None,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.TEXT_GENERATION,
            parameters=parameters,
            requires_input=True,
        )

    async def __call__(
        self,
        engine_uri: EngineUri,
        model: TextGenerationModel,
        operation: Operation,
        tool: ToolManager | None = None,
    ) -> Any:
        assert operation.input and operation.parameters["text"]

        criteria = _stopping_criteria(operation, model)
        mlx_model = _get_mlx_model() if engine_uri.is_local else None
        is_mlx = mlx_model is not None and isinstance(model, mlx_model)
        if engine_uri.is_local and not is_mlx:
            return await model(
                operation.input,
                system_prompt=operation.parameters["text"].system_prompt,
                developer_prompt=operation.parameters["text"].developer_prompt,
                settings=operation.generation_settings or GenerationSettings(),
                stopping_criterias=[criteria] if criteria else None,
                manual_sampling=operation.parameters["text"].manual_sampling
                or False,
                pick=operation.parameters["text"].pick_tokens,
                skip_special_tokens=operation.parameters[
                    "text"
                ].skip_special_tokens
                or False,
                tool=tool,
            )
        return await model(
            operation.input,
            system_prompt=operation.parameters["text"].system_prompt,
            developer_prompt=operation.parameters["text"].developer_prompt,
            settings=operation.generation_settings or GenerationSettings(),
            tool=tool,
        )


@ModalityRegistry.register(Modality.TEXT_QUESTION_ANSWERING)
class TextQuestionAnsweringModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        exit_stack: AsyncExitStack,
    ) -> QuestionAnsweringModel:
        _ = exit_stack
        if not engine_uri.is_local:
            raise NotImplementedError()
        assert engine_uri.model_id is not None
        return QuestionAnsweringModel(
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
            text=OperationTextParameters(
                context=args.text_context,
                system_prompt=args.system or None,
                developer_prompt=getattr(args, "developer", None) or None,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.TEXT_QUESTION_ANSWERING,
            parameters=parameters,
            requires_input=True,
        )

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
            developer_prompt=operation.parameters["text"].developer_prompt,
        )


@ModalityRegistry.register(Modality.TEXT_SEQUENCE_CLASSIFICATION)
class TextSequenceClassificationModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        exit_stack: AsyncExitStack,
    ) -> SequenceClassificationModel:
        _ = exit_stack
        if not engine_uri.is_local:
            raise NotImplementedError()
        assert engine_uri.model_id is not None
        return SequenceClassificationModel(
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
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.TEXT_SEQUENCE_CLASSIFICATION,
            parameters=cast(OperationParameters, None),
            requires_input=True,
        )

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
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        exit_stack: AsyncExitStack,
    ) -> SequenceToSequenceModel:
        _ = exit_stack
        if not engine_uri.is_local:
            raise NotImplementedError()
        assert engine_uri.model_id is not None
        return SequenceToSequenceModel(
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
            text=OperationTextParameters(
                stop_on_keywords=args.stop_on_keyword,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.TEXT_SEQUENCE_TO_SEQUENCE,
            parameters=parameters,
            requires_input=True,
        )

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
            settings=operation.generation_settings or GenerationSettings(),
            stopping_criterias=[criteria] if criteria else None,
        )


@ModalityRegistry.register(Modality.TEXT_TOKEN_CLASSIFICATION)
class TextTokenClassificationModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        exit_stack: AsyncExitStack,
    ) -> TokenClassificationModel:
        _ = exit_stack
        if not engine_uri.is_local:
            raise NotImplementedError()
        assert engine_uri.model_id is not None
        return TokenClassificationModel(
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
            text=OperationTextParameters(
                labeled_only=getattr(args, "text_labeled_only", None),
                system_prompt=args.system or None,
                developer_prompt=getattr(args, "developer", None) or None,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.TEXT_TOKEN_CLASSIFICATION,
            parameters=parameters,
            requires_input=True,
        )

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
            developer_prompt=operation.parameters["text"].developer_prompt,
        )


@ModalityRegistry.register(Modality.TEXT_TRANSLATION)
class TextTranslationModality:
    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        exit_stack: AsyncExitStack,
    ) -> TranslationModel:
        _ = exit_stack
        if not engine_uri.is_local:
            raise NotImplementedError()
        assert engine_uri.model_id is not None
        return TranslationModel(
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
            text=OperationTextParameters(
                language_destination=args.text_to_lang,
                language_source=args.text_from_lang,
                stop_on_keywords=args.stop_on_keyword,
                skip_special_tokens=args.skip_special_tokens,
            )
        )
        return Operation(
            generation_settings=settings,
            input=input_string,
            modality=Modality.TEXT_TRANSLATION,
            parameters=parameters,
            requires_input=True,
        )

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
            settings=operation.generation_settings or GenerationSettings(),
            stopping_criterias=[criteria] if criteria else None,
            skip_special_tokens=operation.parameters[
                "text"
            ].skip_special_tokens
            or False,
        )
