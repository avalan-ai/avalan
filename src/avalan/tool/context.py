from ..container import ContainerToolRuntimeSettings
from ..isolation import IsolationToolRuntimeSettings
from .browser import BrowserToolSettings
from .database import DatabaseToolSettings
from .graph_settings import GraphToolSettings
from .shell import ShellToolSettings

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ToolSettingsContext:
    browser: BrowserToolSettings | None = None
    database: DatabaseToolSettings | None = None
    graph: GraphToolSettings | None = None
    shell: ShellToolSettings | None = None
    shell_explicit_fields: frozenset[str] | None = None
    container: ContainerToolRuntimeSettings | None = None
    isolation: IsolationToolRuntimeSettings | None = None
    extra: dict[str, object] | None = None
