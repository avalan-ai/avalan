from ..entities import (
    EngineSettings,
    EngineUri,
    GenerationSettings,
    TransformerEngineSettings,
)
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Union


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
    role: Role
    goal: Optional[Goal]
    rules: list[str] = field(default_factory=list)
    input_type: InputType = InputType.TEXT
    output_type: OutputType = OutputType.TEXT
    settings: Optional[GenerationSettings] = None
    template_id: Optional[str] = None
    template_vars: Optional[dict] = None


@dataclass(frozen=True, kw_only=True)
class EngineEnvironment:
    engine_uri: EngineUri
    settings: Union[EngineSettings, TransformerEngineSettings]
    type: EngineType = EngineType.TEXT_GENERATION


@dataclass(frozen=True, kw_only=True)
class Operation:
    specification: Specification
    environment: EngineEnvironment
