from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class GraphToolSettings(dict[str, object]):
    """Store configuration settings for graph tools.

    This class is separated from the main graph module to allow importing
    settings without importing matplotlib or running graph backend setup.
    """

    file: str | None = None

    def __post_init__(self) -> None:
        self["file"] = self.file
