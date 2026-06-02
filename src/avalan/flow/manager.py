from ..agent.loader import OrchestratorLoader
from ..event import Event, EventType
from .flow import Flow
from .node import CancellationChecker, Node

from asyncio import wait_for
from logging import Logger
from time import perf_counter
from typing import Any, cast


class FlowManager:
    """Manage execution of a :class:`Flow`."""

    _loader: OrchestratorLoader
    _logger: Logger

    def __init__(
        self, orchestrator_loader: OrchestratorLoader, logger: Logger
    ) -> None:
        self._loader = orchestrator_loader
        self._logger = logger

    async def __call__(
        self,
        flow: Flow,
        *,
        initial_node: str | Node | None = None,
        initial_data: Any = None,
        cancellation_checker: CancellationChecker | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Execute ``flow`` and return its result."""
        start = perf_counter()
        event_manager = cast(
            Any,
            getattr(self._loader, "event_manager", None),
        )
        if event_manager is not None:
            await event_manager.trigger(
                Event(
                    type=EventType.FLOW_MANAGER_CALL_BEFORE,
                    payload={"name": "flow", "status": "started"},
                    started=start,
                )
            )
        execution = flow.execute_async(
            initial_node=initial_node,
            initial_data=initial_data,
            cancellation_checker=cancellation_checker,
        )
        try:
            if timeout_seconds is None:
                result = await execution
            else:
                result = await wait_for(execution, timeout=timeout_seconds)
        except BaseException:
            end = perf_counter()
            if event_manager is not None:
                await event_manager.trigger(
                    Event(
                        type=EventType.FLOW_MANAGER_CALL_AFTER,
                        payload={"name": "flow", "status": "failed"},
                        started=start,
                        finished=end,
                        elapsed=end - start,
                    )
                )
            raise
        else:
            end = perf_counter()
            if event_manager is not None:
                await event_manager.trigger(
                    Event(
                        type=EventType.FLOW_MANAGER_CALL_AFTER,
                        payload={
                            "name": "flow",
                            "status": "succeeded",
                        },
                        started=start,
                        finished=end,
                        elapsed=end - start,
                    )
                )
            return result
