from ....agent import (
    AgentOperation,
    EngineEnvironment,
    Goal,
    OutputType,
    Role,
    Specification,
)
from ....agent.execution import InteractionRuntime
from ....agent.orchestrator import Orchestrator
from ....entities import (
    EngineSettings,
    EngineUri,
    Input,
    Modality,
    TransformerEngineSettings,
)
from ....event.manager import EventManager
from ....memory.manager import MemoryManager
from ....model.manager import ModelManager
from ....tool.manager import ToolManager
from ....tool.shell.settings import ShellToolSettings

from dataclasses import dataclass
from logging import Logger
from typing import Annotated, Any, get_args, get_origin
from uuid import UUID


@dataclass(frozen=True, kw_only=True, slots=True)
class Property:
    TYPE_MAP = {
        int: int.__name__,
        float: float.__name__,
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    name: str
    data_type: str
    description: str | None = None


class JsonOrchestratorOutput(str):
    _usage: object | None
    _usage_responses: tuple[object, ...]

    def __new__(
        cls,
        value: str,
        *,
        usage: object | None = None,
        usage_responses: tuple[object, ...] = (),
    ) -> "JsonOrchestratorOutput":
        output = str.__new__(cls, value)
        output._usage = usage
        output._usage_responses = usage_responses
        return output

    @property
    def usage(self) -> object | None:
        return self._usage

    @property
    def usage_responses(self) -> tuple[object, ...]:
        return self._usage_responses


class JsonSpecification(Specification):
    def __init__(
        self,
        output: type | list[Property],
        role: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(output, list):
            annotations = getattr(output, "__annotations__", None)
            assert annotations

            # Read annotated properties from output_class
            properties: list[Property] = []
            for name, type_info in annotations.items():
                if get_origin(type_info) is Annotated:
                    data_class, *metadata = get_args(type_info)
                    data_type = Property.TYPE_MAP.get(
                        data_class, data_class.__name__
                    )
                    description = metadata[0] if metadata else None
                    properties.append(
                        Property(
                            name=name,
                            data_type=data_type,
                            description=(
                                description.strip()
                                if isinstance(description, str)
                                else None
                            ),
                        )
                    )
        else:
            properties = output

        # Set specification defaults
        assert properties
        template_vars = {
            **(
                kwargs["template_vars"]
                if "template_vars" in kwargs and kwargs["template_vars"]
                else {}
            ),
            **{"output_properties": properties},
        }

        if role is not None:
            kwargs.setdefault("role", Role(persona=[role]))
        kwargs.setdefault("output_type", OutputType.JSON)
        kwargs.setdefault("template_vars", template_vars)
        super().__init__(**kwargs)


class JsonOrchestrator(Orchestrator):
    DEFAULT_FILE_NAME = "agent_json.md"

    def __init__(
        self,
        engine_uri: EngineUri,
        logger: Logger,
        model_manager: ModelManager,
        memory: MemoryManager,
        tool: ToolManager,
        event_manager: EventManager,
        output: type | list[Property],
        *,
        role: str | None = None,
        task: str | None = None,
        instructions: str | None = None,
        goal_instructions: str | None = None,
        name: str | None = None,
        rules: list[str] | None = [],
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
        interaction_runtime: InteractionRuntime | None = None,
    ) -> None:
        if system is not None or developer is not None:
            specification = JsonSpecification(
                output=output,
                goal=None,
                instructions=instructions,
                rules=rules,
                system_prompt=system,
                developer_prompt=developer,
                template_id=template_id or JsonOrchestrator.DEFAULT_FILE_NAME,
                template_vars=template_vars,
            )
        else:
            specification = JsonSpecification(
                output=output,
                role=role,
                goal=(
                    Goal(task=task, goal_instructions=[goal_instructions])
                    if task and goal_instructions
                    else None
                ),
                instructions=instructions,
                rules=rules,
                template_id=template_id or JsonOrchestrator.DEFAULT_FILE_NAME,
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
            interaction_runtime=interaction_runtime,
            name=name,
            shell_input_file_settings=shell_input_file_settings,
            user=user,
            user_template=user_template,
        )

    async def __call__(self, input: Input, **kwargs: Any) -> str:
        text_response = await super().__call__(input, **kwargs)
        output = await text_response.to_json()
        return JsonOrchestratorOutput(
            output,
            usage=text_response.usage,
            usage_responses=_usage_responses_from(text_response),
        )


def _usage_responses_from(response: object) -> tuple[object, ...]:
    usage_responses = getattr(response, "usage_responses", None)
    if callable(usage_responses):
        usage_responses = usage_responses()
    if isinstance(usage_responses, tuple):
        return usage_responses
    if isinstance(usage_responses, list):
        return tuple(usage_responses)
    return ()
