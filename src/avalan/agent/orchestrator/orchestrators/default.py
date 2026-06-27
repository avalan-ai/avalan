from ....agent import (
    AgentOperation,
    EngineEnvironment,
    Goal,
    Role,
    Specification,
)
from ....agent.orchestrator import Orchestrator
from ....entities import (
    EngineSettings,
    EngineUri,
    Modality,
    TransformerEngineSettings,
)
from ....event.manager import EventManager
from ....memory.manager import MemoryManager
from ....model.manager import ModelManager
from ....tool.manager import ToolManager
from ....tool.shell.settings import ShellToolSettings

from logging import Logger
from typing import Any, cast
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
        instructions: str | None = None,
        goal_instructions: str | None = None,
        rules: list[str] | None = None,
        system: str | None = None,
        developer: str | None = None,
        user: str | None = None,
        user_template: str | None = None,
        template_id: str | None = None,
        settings: TransformerEngineSettings | None = None,
        shell_input_file_settings: ShellToolSettings | None = None,
        call_options: dict[str, Any] | None = None,
        template_vars: dict[str, Any] | None = None,
        id: UUID | None = None,
    ) -> None:
        if system is not None or developer is not None:
            specification = Specification(
                role=None,
                goal=None,
                instructions=instructions,
                system_prompt=system,
                developer_prompt=developer,
                rules=rules,
                template_id=template_id or "agent.md",
                template_vars=template_vars,
            )
        else:
            specification = Specification(
                role=cast(Role | None, role),
                goal=(
                    Goal(task=task, goal_instructions=[goal_instructions])
                    if task and goal_instructions
                    else None
                ),
                instructions=instructions,
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
                    engine_uri=engine_uri,
                    settings=settings or EngineSettings(),
                ),
                modality=Modality.TEXT_GENERATION,
            ),
            call_options=call_options,
            id=id,
            name=name,
            shell_input_file_settings=shell_input_file_settings,
            user=user,
            user_template=user_template,
        )
