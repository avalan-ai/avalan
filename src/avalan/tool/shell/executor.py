from typing import Protocol


class CommandExecutor(Protocol):
    async def execute(self, spec: object) -> object:
        raise NotImplementedError


class LocalCommandExecutor:
    async def execute(self, spec: object) -> object:
        raise NotImplementedError("local shell execution is not implemented")
