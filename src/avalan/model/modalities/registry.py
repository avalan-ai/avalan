from ...entities import (
    ChatSettings,
    EngineUri,
    GenerationSettings,
    GenerationCacheStrategy,
    Input,
    Modality,
    Operation,
    ReasoningSettings,
    ReasoningTag,
    TransformerEngineSettings,
)
from ...tool.manager import ToolManager

from argparse import Namespace
from collections.abc import Callable
from inspect import isclass
from logging import Logger
from typing import Any, Protocol


class ModalityHandler(Protocol):
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: Any,
        operation: Operation,
        tool: ToolManager | None,
    ) -> Any: ...

    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
    ) -> Any: ...

    def get_operation_from_arguments(
        self,
        args: Namespace,
        input_string: Input | None,
        settings: GenerationSettings,
    ) -> Operation: ...


class ModalityRegistry:
    _handlers: dict[Modality, ModalityHandler] = {}

    @classmethod
    def register(
        cls, modality: Modality
    ) -> Callable[[ModalityHandler | type], ModalityHandler]:
        def decorator(handler: ModalityHandler | type) -> ModalityHandler:
            cls._handlers[modality] = (
                handler() if isclass(handler) else handler
            )
            return handler

        return decorator

    @classmethod
    def get(cls, modality: Modality) -> ModalityHandler:
        if modality not in cls._handlers:
            raise NotImplementedError(f"Modality {modality} not registered")
        return cls._handlers[modality]

    @classmethod
    def load_engine(
        cls,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        modality: Modality,
        logger: Logger,
    ) -> Any:
        handler = cls.get(modality)
        return handler.load_engine(engine_uri, engine_settings, logger)

    @classmethod
    def get_operation_from_arguments(
        cls,
        modality: Modality,
        args: Namespace,
        input_string: Input | None,
    ) -> Operation:
        reasoning_settings = ReasoningSettings(
            max_new_tokens=getattr(args, "reasoning_max_new_tokens", None),
            enabled=not getattr(args, "no_reasoning", False),
            stop_on_max_new_tokens=getattr(
                args,
                "reasoning_stop_on_max_new_tokens",
                False,
            ),
            tag=(
                ReasoningTag(getattr(args, "reasoning_tag"))
                if getattr(args, "reasoning_tag", None)
                else None
            ),
        )
        settings = GenerationSettings(
            do_sample=args.do_sample,
            enable_gradient_calculation=args.enable_gradient_calculation,
            max_new_tokens=args.max_new_tokens,
            max_length=getattr(args, "text_max_length", None),
            min_p=args.min_p,
            num_beams=getattr(args, "text_num_beams", None),
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=args.use_cache,
            cache_strategy=(
                GenerationCacheStrategy(args.cache_strategy)
                if getattr(args, "cache_strategy", None)
                else None
            ),
            chat_settings=ChatSettings(
                enable_thinking=not getattr(
                    args,
                    "chat_disable_thinking",
                    not reasoning_settings.enabled,
                )
            ),
            reasoning=reasoning_settings,
        )
        try:
            handler = cls.get(modality)
        except NotImplementedError:
            return Operation(
                generation_settings=settings,
                input=input_string,
                modality=modality,
                parameters=None,
                requires_input=False,
            )
        return handler.get_operation_from_arguments(
            args, input_string, settings
        )
