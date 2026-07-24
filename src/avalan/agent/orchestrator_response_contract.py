"""Define service-light durable orchestrator response state."""

from abc import ABC, abstractmethod
from collections.abc import Mapping


class DurableOrchestratorResponse(ABC):
    """Expose response state needed to stage a durable continuation."""

    @property
    @abstractmethod
    def continuation_generation_settings(self) -> Mapping[str, object]:
        """Return provider-neutral continuation settings."""
        ...

    @property
    @abstractmethod
    def continuation_tool_loop_count(self) -> int:
        """Return completed domain-tool cycles before suspension."""
        ...
