"""Test configuration for Avalan's pytest suite."""

import sys
import types
from typing import Any


def _install_a2a_stub() -> None:
    if "a2a" in sys.modules:
        return

    class _DynamicEnum(str):
        def __new__(cls, value: str) -> "_DynamicEnum":
            return str.__new__(cls, value)

        @property
        def value(self) -> str:
            return str(self)

    class TaskStatus(_DynamicEnum):
        pass

    TaskStatus.CREATED = TaskStatus("created")  # type: ignore[attr-defined]
    TaskStatus.RUNNING = TaskStatus("running")  # type: ignore[attr-defined]
    TaskStatus.COMPLETED = TaskStatus("completed")  # type: ignore[attr-defined]
    TaskStatus.FAILED = TaskStatus("failed")  # type: ignore[attr-defined]

    class _SDKModel:
        def __init__(self, **data: Any) -> None:
            self._data = dict(data)

        @classmethod
        def model_validate(cls, payload: dict[str, Any]) -> "_SDKModel":
            return cls(**payload)

        def model_dump(
            self, *, by_alias: bool = True, exclude_none: bool = True
        ) -> dict[str, Any]:
            if exclude_none:
                return {
                    key: value for key, value in self._data.items() if value is not None
                }
            return dict(self._data)

        def dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
            return self.model_dump(exclude_none=exclude_none)

    class AgentCard(_SDKModel):
        pass

    class Event(_SDKModel):
        pass

    class Task(_SDKModel):
        pass

    a2a_module = types.ModuleType("a2a")
    types_module = types.ModuleType("a2a.types")
    agent_module = types.ModuleType("a2a.types.agent")
    event_module = types.ModuleType("a2a.types.event")
    task_module = types.ModuleType("a2a.types.task")

    agent_module.AgentCard = AgentCard
    event_module.Event = Event
    task_module.Task = Task
    task_module.TaskStatus = TaskStatus

    types_module.AgentCard = AgentCard
    types_module.Event = Event
    types_module.Task = Task
    types_module.TaskStatus = TaskStatus
    types_module.agent = agent_module
    types_module.event = event_module
    types_module.task = task_module

    a2a_module.types = types_module

    sys.modules["a2a"] = a2a_module
    sys.modules["a2a.types"] = types_module
    sys.modules["a2a.types.agent"] = agent_module
    sys.modules["a2a.types.event"] = event_module
    sys.modules["a2a.types.task"] = task_module


_install_a2a_stub()
