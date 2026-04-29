from .browser import BrowserToolSettings
from .database import DatabaseToolSettings
from .graph_settings import GraphToolSettings

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ToolSettingsContext:
    browser: BrowserToolSettings | None = None
    database: DatabaseToolSettings | None = None
    graph: GraphToolSettings | None = None
    extra: dict[str, object] | None = None
