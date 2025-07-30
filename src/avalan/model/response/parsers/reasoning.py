from ....entities import ReasoningSettings, ReasoningToken
from typing import Any, Iterable


class ReasoningTokenLimitExceeded(Exception):
    pass


class ReasoningParser:
    def __init__(
        self,
        *,
        end_tag: str = "</think>",
        prefixes: list[str] | None = None,
        reasoning_settings: ReasoningSettings,
        start_tag: str = "<think>",
        max_thinking_turns: int = 1,
    ) -> None:
        self._settings = reasoning_settings
        self._start_tag = start_tag
        self._end_tag = end_tag
        self._prefixes = prefixes or ["Think:"]
        self._thinking = False
        self._thinking_turns = 0
        self._max_thinking_turns = 1
        self._thinking_budget_exhausted = False
        self._token_count = 0
        self._pending_tag: list[str] = []
        self._pending_length = 0
        self._max_tag_len = max(len(self._start_tag), len(self._end_tag))

    def set_thinking(self, thinking: bool) -> None:
        self._thinking = thinking

    @property
    def is_thinking(self) -> bool:
        return self._thinking

    @property
    def is_thinking_budget_exhausted(self) -> bool:
        return self._thinking_budget_exhausted

    async def push(self, token: str) -> Iterable[Any]:
        if self._thinking_budget_exhausted and not self._thinking:
            return [token]

        token_clean = token.strip()
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
                    return self._set_thinking(
                        result=result,
                        is_start=(expecting_tag == self._start_tag),
                    )
                return result
            result.extend(self._flush_pending(False))

        if token_clean in (self._start_tag, self._end_tag) or (
            not self._thinking
            and any(token_clean.startswith(p) for p in self._prefixes)
        ):
            return self._set_thinking(
                token=token,
                result=result,
                is_start=(token_clean != self._end_tag),
            )

        if expecting_tag.startswith(token_clean):
            self._pending_tag.append(token)
            self._pending_length += len(token_clean)
            while self._pending_length > len(expecting_tag):
                removed = self._pending_tag.pop(0)
                self._pending_length -= len(removed.strip())
            if token_clean == expecting_tag:
                return self._set_thinking(
                    result=result, is_start=(expecting_tag == self._start_tag)
                )
            return result

        if self._thinking:
            within_budget = (
                self._settings.max_new_tokens is None
                or self._token_count < self._settings.max_new_tokens
            )
            if within_budget:
                result.extend(self._wrap(token))
                return result
            if self._settings.stop_on_max_new_tokens:
                raise ReasoningTokenLimitExceeded
            result.append(token)
            return result

        result.append(token)
        return result

    async def flush(self) -> Iterable[Any]:
        result: list[Any] = []
        if self._pending_tag:
            as_reasoning = self._thinking
            for t in self._pending_tag:
                if as_reasoning:
                    self._token_count += 1
                    result.append(ReasoningToken(t))
                else:
                    result.append(t)
            self._pending_tag.clear()
            self._pending_length = 0
        return result

    def _set_thinking(
        self, result: list[Any], is_start: bool, token: str | None = None
    ) -> list[Any]:
        self._thinking = is_start
        if is_start:
            self._thinking_turns += 1
            if self._thinking_turns >= self._max_thinking_turns:
                self._thinking_budget_exhausted = True
        result.extend(self._flush_pending(True))
        if token is not None:
            result.extend(self._wrap(token))
        return result

    def _wrap(self, t: str) -> list[Any]:
        self._token_count += 1
        return [ReasoningToken(t)]

    def _flush_pending(self, as_reasoning: bool) -> list[Any]:
        result: list[Any] = []
        if self._pending_tag:
            for t in self._pending_tag:
                result.extend(self._wrap(t) if as_reasoning else [t])
            self._pending_tag.clear()
            self._pending_length = 0
        return result
