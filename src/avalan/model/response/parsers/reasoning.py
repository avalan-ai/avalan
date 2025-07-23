"""Parser labeling reasoning tokens."""

from typing import Any, Iterable

from ....entities import ReasoningToken


class ReasoningParser:
    """Label reasoning tokens inside ``<think>`` blocks or after prefixes."""

    def __init__(
        self,
        *,
        start_tag: str = "<think>",
        end_tag: str = "</think>",
        prefixes: list[str] | None = None,
    ) -> None:
        self._start_tag = start_tag
        self._end_tag = end_tag
        self._prefixes = prefixes or ["Think:"]
        self._thinking = False

    async def push(self, token_str: str) -> Iterable[Any]:
        if token_str.strip() == self._start_tag:
            self._thinking = True
            return [token_str]
        if token_str.strip() == self._end_tag:
            self._thinking = False
            return [token_str]
        if any(token_str.startswith(p) for p in self._prefixes):
            self._thinking = True
            return [token_str]
        if self._thinking:
            return [ReasoningToken(token_str)]
        return [token_str]

    async def flush(self) -> Iterable[Any]:
        return []
