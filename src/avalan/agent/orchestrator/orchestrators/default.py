from ....agent import EngineEnvironment, AgentOperation, Goal, Specification
from ....agent.orchestrator import Orchestrator
from ....entities import EngineUri, Modality, TransformerEngineSettings
from ....event.manager import EventManager
from ....memory.manager import MemoryManager
from ....model.manager import ModelManager
from ....tool.manager import ToolManager
from logging import Logger
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
        name: str | None,
        role: str | None,
        task: str | None,
        instructions: str | None,
        rules: list[str] | None,
        template_id: str | None = None,
        settings: TransformerEngineSettings | None = None,
        call_options: dict | None = None,
        template_vars: dict | None = None,
        id: UUID | None = None,
    ):
        specification = Specification(
            role=role,
            goal=(
                Goal(task=task, instructions=[instructions])
                if task and instructions
                else None
            ),
            rules=rules,
            template_id=template_id or "agent.md",
            template_vars=template_vars,
        )
        super().__init__(
            logger,
            model_manager,
            memory,
            tool,
            event_manager,
            AgentOperation(
                specification=specification,
                environment=EngineEnvironment(
                    engine_uri=engine_uri, settings=settings
                ),
                modality=Modality.TEXT_GENERATION,
            ),
            call_options=call_options,
            id=id,
            name=name,
        )
