from ..container import ContainerToolRuntimeSettings
from ..isolation import IsolationToolRuntimeSettings
from ..patch.toolset import PatchToolSettings
from ..skill.settings import TrustedSkillSettings
from .browser import BrowserToolSettings
from .database import DatabaseToolSettings
from .graph_settings import GraphToolSettings
from .shell import ShellToolSettings

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import final

_PATCH_TOOL_SETTINGS_INTEGRATION = "trusted_runtime_settings"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class A2AToolSettings:
    """Configure one trusted A2A client runtime."""

    client_params: Mapping[str, object] = field(
        default_factory=dict,
        repr=False,
    )
    call_params: Mapping[str, object] = field(
        default_factory=dict,
        repr=False,
    )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ToolSettingsContext:
    a2a: A2AToolSettings | None = None
    browser: BrowserToolSettings | None = None
    database: DatabaseToolSettings | None = None
    graph: GraphToolSettings | None = None
    skills: TrustedSkillSettings | None = None
    shell: ShellToolSettings | None = None
    shell_explicit_fields: frozenset[str] | None = None
    container: ContainerToolRuntimeSettings | None = None
    isolation: IsolationToolRuntimeSettings | None = None
    patch: PatchToolSettings | None = None
    extra: dict[str, object] | None = None
