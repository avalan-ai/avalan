from ..entities import (
    EngineSettings,
    EngineUri,
    GenerationSettings,
    TransformerEngineSettings,
)
from dataclasses import dataclass, field
from enum import StrEnum


class NoOperationAvailableException(Exception):
    pass


class EngineType(StrEnum):
    TEXT_GENERATION = "text_generation"


class InputType(StrEnum):
    TEXT = "text"


class OutputType(StrEnum):
    JSON = "json"
    TEXT = "text"


@dataclass(frozen=True, kw_only=True)
class Goal:
    task: str
    instructions: list[str]


@dataclass(frozen=True, kw_only=True)
class Role:
    persona: list[str]


@dataclass(frozen=True, kw_only=True)
class Specification:
    role: Role | None
    goal: Goal | None
    rules: list[str] | None = field(default_factory=list)
    input_type: InputType = InputType.TEXT
    output_type: OutputType = OutputType.TEXT
    settings: GenerationSettings | None = None
    template_id: str | None = None
    template_vars: dict | None = None


@dataclass(frozen=True, kw_only=True)
class EngineEnvironment:
    engine_uri: EngineUri
    settings: EngineSettings | TransformerEngineSettings
    type: EngineType = EngineType.TEXT_GENERATION


@dataclass(frozen=True, kw_only=True)
class Operation:
    specification: Specification
    environment: EngineEnvironment
