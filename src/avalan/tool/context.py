from .browser import BrowserToolSettings
from .database import DatabaseToolSettings
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolSettingsContext:
    browser: BrowserToolSettings | None = None
    database: DatabaseToolSettings | None = None
    extra: dict[str, object] | None = None
