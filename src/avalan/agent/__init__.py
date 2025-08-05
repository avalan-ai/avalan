from ..entities import (
    EngineSettings,
    EngineUri,
    GenerationSettings,
    Modality,
    TransformerEngineSettings,
)
from dataclasses import dataclass, field
from enum import StrEnum


class NoOperationAvailableException(Exception):
    pass


class InputType(StrEnum):
    TEXT = "text"


class OutputType(StrEnum):
    JSON = "json"
    TEXT = "text"


@dataclass(frozen=True, kw_only=True, slots=True)
class Goal:
    task: str
    instructions: list[str]


@dataclass(frozen=True, kw_only=True, slots=True)
class Role:
    persona: list[str]


@dataclass(frozen=True, kw_only=True, slots=True)
class Specification:
    role: Role | None
    goal: Goal | None
    rules: list[str] | None = field(default_factory=list)
    input_type: InputType = InputType.TEXT
    output_type: OutputType = OutputType.TEXT
    settings: GenerationSettings | None = None
    template_id: str | None = None
    template_vars: dict | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class EngineEnvironment:
    engine_uri: EngineUri
    settings: EngineSettings | TransformerEngineSettings


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentOperation:
    specification: Specification
    environment: EngineEnvironment
    modality: Modality = Modality.TEXT_GENERATION
