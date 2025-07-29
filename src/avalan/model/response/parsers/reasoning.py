from ....entities import ReasoningSettings, ReasoningToken
from typing import Any, Iterable


class ReasoningTokenLimitExceeded(Exception):
    pass


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

    async def push(self, token: str) -> Iterable[Any]:
        token_clean = token.strip()

        def wrap(t: str) -> list[Any]:
            self._token_count += 1
            return [ReasoningToken(t)]

        if token_clean in (self._start_tag, self._end_tag) or any(
            token_clean.startswith(p) for p in self._prefixes
        ):
            self._thinking = token_clean != self._end_tag
            return wrap(token)

        if self._thinking:
            within_budget = (
                self._settings.max_new_tokens is None
                or self._token_count < self._settings.max_new_tokens
            )
            if within_budget:
                return wrap(token)
            if self._settings.stop_on_max_new_tokens:
                raise ReasoningTokenLimitExceeded
            return [token]

        return [token]

    async def flush(self) -> Iterable[Any]:
        return []
