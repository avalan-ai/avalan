from .parser import ReasoningOutputParser

from types import TracebackType
from typing import Any

from avalan.agent.orchestrator import Orchestrator
from avalan.entities import Input, Message, ReasoningOrchestratorResponse


class ReasoningOrchestrator(Orchestrator):
    """Wrap another orchestrator adding Chain-of-Thought reasoning."""

    TEMPLATE_ID = "reasoning/cot.md"

    def __init__(
        self,
        orchestrator: Orchestrator,
        parser: ReasoningOutputParser | None = None,
    ) -> None:
        assert orchestrator
        self._orchestrator = orchestrator
        self._parser = parser or ReasoningOutputParser()
        super().__init__(
            orchestrator._logger,
            orchestrator._model_manager,
            orchestrator._memory,
            orchestrator._tool,
            orchestrator._event_manager,
            orchestrator.operations,
            call_options=orchestrator._call_options,
            exit_memory=orchestrator._exit_memory,
            id=orchestrator.id,
            name=orchestrator.name,
            renderer=orchestrator.renderer,
        )

    async def __aenter__(self) -> "ReasoningOrchestrator":
        await self._orchestrator.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return await self._orchestrator.__aexit__(
            exc_type, exc_value, traceback
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orchestrator, name)

    async def __call__(self, input: Input, **kwargs: Any) -> Any:
        template_vars = {}
        if self._orchestrator.operations:
            template_vars = (
                self._orchestrator.operations[0].specification.template_vars
                or {}
            )
        prompt_text = input.content if isinstance(input, Message) else input
        assert isinstance(prompt_text, str)
        prompt_text = self._renderer.from_string(prompt_text, template_vars)
        rendered_input = self._renderer(
            self.TEMPLATE_ID, prompt=prompt_text, **template_vars
        )
        response = await self._orchestrator(rendered_input, **kwargs)
        text = await response.to_str()
        reasoning, answer = self._parser.parse(text)
        return ReasoningOrchestratorResponse(
            answer=answer, reasoning=reasoning
        )
