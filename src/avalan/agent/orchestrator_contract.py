"""Define the service-light public orchestrator call contract."""

from ..entities import Input

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Orchestrator(Protocol):
    """Run one agent operation through the public SDK."""

    async def __call__(
        self,
        input: Input,
        **kwargs: Any,
    ) -> object:
        """Return one asynchronously produced agent response."""
        ...
