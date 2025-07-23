from typing import Any, Iterable, Protocol


class StreamParser(Protocol):
    async def push(self, token_str: str) -> Iterable[Any]: ...

    async def flush(self) -> Iterable[Any]: ...
