from avalan.agent import EngineEnvironment, Goal, Operation, Specification
from avalan.agent.orchestrator import Orchestrator
from logging import Logger
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model.entities import EngineUri, TransformerEngineSettings
from avalan.model.manager import ModelManager
from avalan.tool.manager import ToolManager
from typing import Optional
from uuid import UUID

class DefaultOrchestrator(Orchestrator):
    def __init__(
        self,
        engine_uri: EngineUri,
        logger: Logger,
        model_manager: ModelManager,
        memory: MemoryManager,
        tool: ToolManager,
        event_manager: EventManager,
        *,
        name: Optional[str],
        role: str,
        task: str,
        instructions: str,
        rules: Optional[list[str]],
        template_id: Optional[str]=None,
        settings: Optional[TransformerEngineSettings]=None,
        call_options: Optional[dict]=None,
        template_vars: Optional[dict]=None,
        id: Optional[UUID]=None
    ):
        specification = Specification(
            role=role,
            goal=Goal(
                task=task,
                instructions=[instructions]
            ) if task and instructions else None,
            rules=rules,
            template_id=template_id or "agent.md",
            template_vars=template_vars
        )
        super().__init__(
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            Operation(
                specification=specification,
                environment=EngineEnvironment(
                    engine_uri=engine_uri,
                    settings=settings
                ),
            ),
            call_options=call_options,
            id=id,
            name=name
        )

