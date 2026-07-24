from ...entities import (
    ChatSettings,
    EngineUri,
    GenerationCacheStrategy,
    GenerationSettings,
    Input,
    Modality,
    Operation,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryMode,
    ReasoningTag,
    TransformerEngineSettings,
)
from ..capability import ModelCapabilityCatalog

from argparse import Namespace
from collections.abc import Callable
from contextlib import AsyncExitStack
from importlib import import_module
from inspect import isclass
from logging import Logger
from typing import Any, Protocol, TypeVar, cast

HandlerType = TypeVar("HandlerType")
_HANDLER_MODALITY_ATTRIBUTE = "__avalan_modality__"


class ModalityHandler(Protocol):
    async def __call__(
        self,
        engine_uri: EngineUri,
        model: Any,
        operation: Operation,
        capability: ModelCapabilityCatalog | None,
    ) -> Any: ...

    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        logger: Logger,
        exit_stack: AsyncExitStack,
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
    def _normalize_modality(cls, modality: Modality | str) -> Modality | str:
        if isinstance(modality, Modality):
            return modality
        try:
            return Modality(modality)
        except ValueError:
            return modality

    @classmethod
    def _load_handlers(cls, modality: Modality | str) -> None:
        if not isinstance(modality, Modality):
            return
        module_name = (
            "audio"
            if modality.value.startswith("audio_")
            else ("vision" if modality.value.startswith("vision_") else "text")
        )
        module = import_module(f"{__package__}.{module_name}")
        if modality not in cls._handlers:
            cls._register_cached_handlers(module)

    @classmethod
    def _register_cached_handlers(cls, module: object) -> None:
        for handler in vars(module).values():
            modality = getattr(handler, _HANDLER_MODALITY_ATTRIBUTE, None)
            modality = cls._normalize_modality(modality) if modality else None
            if not isinstance(modality, Modality):
                continue
            if isclass(handler):
                class_handler = cast(type[ModalityHandler], handler)
                cls._handlers[modality] = class_handler()
            else:
                cls._handlers[modality] = cast(ModalityHandler, handler)

    @classmethod
    def register(
        cls, modality: Modality
    ) -> Callable[[HandlerType], HandlerType]:
        def decorator(handler: HandlerType) -> HandlerType:
            setattr(handler, _HANDLER_MODALITY_ATTRIBUTE, modality)
            if isclass(handler):
                class_handler = cast(type[ModalityHandler], handler)
                resolved_handler = class_handler()
            else:
                resolved_handler = cast(ModalityHandler, handler)
            cls._handlers[modality] = resolved_handler
            return handler

        return decorator

    @classmethod
    def get(cls, modality: Modality | str) -> ModalityHandler:
        modality = cls._normalize_modality(modality)
        if not isinstance(modality, Modality):
            raise NotImplementedError(f"Modality {modality} not registered")
        if modality not in cls._handlers:
            cls._load_handlers(modality)
        if modality not in cls._handlers:
            raise NotImplementedError(f"Modality {modality} not registered")
        return cls._handlers[modality]

    @classmethod
    def load_engine(
        cls,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        modality: Modality | str,
        logger: Logger,
        exit_stack: AsyncExitStack,
    ) -> Any:
        handler = cls.get(modality)
        return handler.load_engine(
            engine_uri, engine_settings, logger, exit_stack
        )

    @classmethod
    def get_operation_from_arguments(
        cls,
        modality: Modality | str,
        args: Namespace,
        input_string: Input | None,
    ) -> Operation:
        modality = cls._normalize_modality(modality)
        if (
            getattr(args, "reasoning_summary", None) is not None
            and modality is not Modality.TEXT_GENERATION
        ):
            raise ValueError(
                "reasoning summary requires text_generation modality"
            )
        reasoning_settings = ReasoningSettings(
            effort=(
                ReasoningEffort(getattr(args, "reasoning_effort"))
                if getattr(args, "reasoning_effort", None)
                else None
            ),
            summary=(
                ReasoningSummaryMode(getattr(args, "reasoning_summary"))
                if getattr(args, "reasoning_summary", None)
                else None
            ),
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
            openai_max_retries=getattr(args, "openai_max_retries", None),
            openai_response_failed_retries=getattr(
                args,
                "openai_response_failed_retries",
                None,
            ),
            openai_response_failed_retry_delay_seconds=getattr(
                args,
                "openai_response_failed_retry_delay_seconds",
                None,
            ),
            openai_timeout_seconds=getattr(
                args,
                "openai_timeout_seconds",
                None,
            ),
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
                modality=cast(Modality, modality),
                parameters=cast(Any, None),
                requires_input=False,
            )
        return handler.get_operation_from_arguments(
            args, input_string, settings
        )
