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
        self._pending_tag: list[str] = []
        self._pending_length = 0
        self._max_tag_len = max(len(self._start_tag), len(self._end_tag))

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

        def flush_pending(as_reasoning: bool) -> list[Any]:
            result: list[Any] = []
            if self._pending_tag:
                for t in self._pending_tag:
                    result.extend(wrap(t) if as_reasoning else [t])
                self._pending_tag.clear()
                self._pending_length = 0
            return result

        expecting_tag = self._end_tag if self._thinking else self._start_tag

        result: list[Any] = []
        if self._pending_tag:
            candidate = (
                "".join(t.strip() for t in self._pending_tag) + token_clean
            )
            if expecting_tag.startswith(candidate):
                self._pending_tag.append(token)
                self._pending_length += len(token_clean)
                while self._pending_length > len(expecting_tag):
                    removed = self._pending_tag.pop(0)
                    self._pending_length -= len(removed.strip())
                if candidate == expecting_tag:
                    self._thinking = expecting_tag == self._start_tag
                    result.extend(flush_pending(True))
                    return result
                return result
            result.extend(flush_pending(False))

        if token_clean in (self._start_tag, self._end_tag) or (
            not self._thinking
            and any(token_clean.startswith(p) for p in self._prefixes)
        ):
            self._thinking = token_clean != self._end_tag
            result.extend(flush_pending(True))
            result.extend(wrap(token))
            return result

        if expecting_tag.startswith(token_clean):
            self._pending_tag.append(token)
            self._pending_length += len(token_clean)
            while self._pending_length > len(expecting_tag):
                removed = self._pending_tag.pop(0)
                self._pending_length -= len(removed.strip())
            if token_clean == expecting_tag:
                self._thinking = expecting_tag == self._start_tag
                result.extend(flush_pending(True))
                return result
            return result

        if self._thinking:
            within_budget = (
                self._settings.max_new_tokens is None
                or self._token_count < self._settings.max_new_tokens
            )
            if within_budget:
                result.extend(wrap(token))
                return result
            if self._settings.stop_on_max_new_tokens:
                raise ReasoningTokenLimitExceeded
            result.append(token)
            return result

        result.append(token)
        return result

    async def flush(self) -> Iterable[Any]:
        return []
