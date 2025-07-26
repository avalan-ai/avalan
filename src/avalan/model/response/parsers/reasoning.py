from typing import Any, Iterable

from ....entities import ReasoningSettings, ReasoningToken


class ReasoningTokenLimitExceeded(Exception):
    """Raised when the reasoning token limit is reached."""


class ReasoningParser:
    def __init__(
        self,
        *,
        reasoning_settings: ReasoningSettings,
        start_tag: str = "<think>",
        end_tag: str = "</think>",
        prefixes: list[str] | None = None,
    ) -> None:
        self._settings = reasoning_settings
        self._start_tag = start_tag
        self._end_tag = end_tag
        self._prefixes = prefixes or ["Think:"]
        self._thinking = False
        self._token_count = 0

    def set_thinking(self, thinking: bool) -> None:
        self._thinking = thinking

    @property
    def is_thinking(self) -> bool:
        return self._thinking

    async def push(self, token_str: str) -> Iterable[Any]:
        token_clean = token_str.strip()
        if token_clean == self._start_tag:
            self._thinking = True
            self._token_count += 1
            return [ReasoningToken(token_str)]
        if token_clean == self._end_tag:
            self._thinking = False
            self._token_count += 1
            return [ReasoningToken(token_str)]
        if any(token_clean.startswith(p) for p in self._prefixes):
            self._thinking = True
            self._token_count += 1
            return [ReasoningToken(token_str)]
        if self._thinking:
            if (
                self._settings.max_new_tokens is None
                or self._token_count < self._settings.max_new_tokens
            ):
                self._token_count += 1
                return [ReasoningToken(token_str)]
            if self._settings.stop_on_max_new_tokens:
                raise ReasoningTokenLimitExceeded
            return [token_str]
        return [token_str]

    async def flush(self) -> Iterable[Any]:
        return []
