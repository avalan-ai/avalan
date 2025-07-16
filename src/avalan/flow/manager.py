from logging import Logger
from time import perf_counter
from typing import Any

from ..agent.loader import OrchestratorLoader
from ..event import Event, EventType
from .flow import Flow


class FlowManager:
    """Manage execution of a :class:`Flow`."""

    _loader: OrchestratorLoader
    _logger: Logger

    def __init__(
        self, orchestrator_loader: OrchestratorLoader, logger: Logger
    ) -> None:
        self._loader = orchestrator_loader
        self._logger = logger

    async def __call__(self, flow: Flow) -> Any:
        """Execute ``flow`` and return its result."""
        start = perf_counter()
        await self._loader.event_manager.trigger(
            Event(
                type=EventType.FLOW_MANAGER_CALL_BEFORE,
                payload={"flow": flow},
                started=start,
            )
        )
        result = flow.execute()
        end = perf_counter()
        await self._loader.event_manager.trigger(
            Event(
                type=EventType.FLOW_MANAGER_CALL_AFTER,
                payload={"flow": flow, "result": result},
                started=start,
                finished=end,
                elapsed=end - start,
            )
        )
        return result
