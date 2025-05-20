from abc import ABC, abstractmethod
from avalan.agent import Specification
from avalan.memory.manager import MemoryManager
from avalan.model.engine import Engine
from avalan.model.entities import (
    EngineMessage,
    GenerationSettings,
    Input,
    Message,
    MessageRole,
)
from avalan.model.nlp.text import TextGenerationResponse
from avalan.model.nlp.text.vendor import TextGenerationVendorModel
from avalan.tool.manager import ToolManager
from dataclasses import replace
from typing import Any, Optional, Union, Tuple
from uuid import UUID, uuid4

class EngineAgent(ABC):
    _id: UUID
    _name: Optional[str]
    _model: Engine
    _memory: MemoryManager
    _tool: ToolManager
    _last_output: Optional[TextGenerationResponse]=None
    _last_prompt: Optional[Tuple[Input,Optional[str]]]=None

    @abstractmethod
    def _prepare_call(
        self,
        specification: Specification,
        input: str,
        **kwargs: Any
    ) -> Any:
        raise NotImplementedError()

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def engine(self) -> Engine:
        return self._model

    @property
    def output(self) -> Optional[TextGenerationResponse]:
        return self._last_output

    @property
    def input_token_count(self) -> Optional[int]:
        return self._model.input_token_count(
            self._last_prompt[0],
            system_prompt=self._last_prompt[1]
        ) if self._last_prompt else None

    def __init__(
        self,
        model: Engine,
        memory: MemoryManager,
        tool: ToolManager,
        *args,
        name: Optional[str]=None,
        id: Optional[UUID]=None,
    ):
        self._id = id or uuid4()
        self._name = name
        self._model = model
        self._memory = memory
        self._tool = tool

    async def __call__(
        self,
        specification: Specification,
        input: str,
        **kwargs
    ) -> Union[
        TextGenerationResponse,
        str
    ]:
        run_args = self._prepare_call(specification, input, **kwargs)
        return await self._run(input, **run_args)

    async def _run(
        self,
        input: str,
        *args,
        settings: Optional[GenerationSettings]=None,
        system_prompt: Optional[str]=None,
        skip_special_tokens=True,
        **kwargs
    ) -> Union[
        TextGenerationResponse
    ]:
        # Process settings
        if settings and kwargs:
            settings = replace(settings, **kwargs)
        elif not settings:
            kwargs.setdefault("temperature", None)
            kwargs.setdefault("do_sample", False)
            settings = GenerationSettings(**kwargs)
        assert settings

        # Prepare memory
        assert not self._memory.has_recent_message \
               or self._memory.recent_message is not None

        # Should always be stored, with or without memory
        self._last_prompt = (input, system_prompt)

        # Transform input (by adding memory, if necessary)
        if (
            self._memory.has_permanent_message or
            self._memory.has_recent_message
        ) and isinstance(input,Message):
            previous_message: Optional[Message]=None
            new_message: Message = input

            # Handle last message if not already consumed
            previous_output = self._last_output
            if previous_output \
               and isinstance(previous_output, TextGenerationResponse):
                previous_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=await previous_output.to_str()
                )

            # Append messages
            if previous_message:
                await self._memory.append_message(EngineMessage(
                    agent_id=self._id,
                    model_id=self._model.model_id,
                    message=previous_message
                ))
            await self._memory.append_message(EngineMessage(
                agent_id=self._id,
                model_id=self._model.model_id,
                message=new_message
            ))

            # Make recent memory the new model input
            input = [ rm.message for rm in self._memory.recent_messages ]

        # Have model generate output from input

        model_settings = dict(
            system_prompt=system_prompt,
            settings=settings,
            tool=self._tool
        )
        if not isinstance(self._model, TextGenerationVendorModel):
            model_settings["skip_special_tokens"] = skip_special_tokens

        output = await self._model(input, **model_settings)

        # Update memory
        if self._memory.has_recent_message:
            self._last_output = output

        return output

