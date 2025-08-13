from ....entities import ReasoningSettings, ReasoningTag, ReasoningToken
from logging import Logger
from typing import Any, Iterable


class ReasoningTokenLimitExceeded(Exception):
    pass


class ReasoningParser:
    tags: dict[ReasoningTag, tuple[str, str]] = {
        ReasoningTag.THINK: ("<think>", "</think>"),
        ReasoningTag.CHANNEL: (
            "<|channel|>analysis<|message|>",
            "<|end|>",
        ),
    }

    _settings: ReasoningSettings
    _start_tag: str
    _end_tag: str
    _prefixes: tuple[str, ...]
    _thinking: bool
    _thinking_turns: int
    _max_thinking_turns: int
    _thinking_budget_exhausted: bool
    _token_count: int
    _pending_tokens: list[str]
    _pending_str: str
    _logger: Logger

    def __init__(
        self,
        *,
        reasoning_settings: ReasoningSettings,
        logger: Logger,
        bos_token: str | None = None,
        start_tag: str | None = None,
        end_tag: str | None = None,
        prefixes: list[str] | None = None,
        max_thinking_turns: int = 1,
    ) -> None:
        self._settings = reasoning_settings
        self._logger = logger
        tag = reasoning_settings.tag
        if not tag:
            if bos_token == "<|startoftext|>":
                tag = ReasoningTag.CHANNEL
            else:
                tag = ReasoningTag.THINK
        default_start, default_end = self.tags[tag]
        self._start_tag = start_tag or default_start
        self._end_tag = end_tag or default_end
        self._prefixes = tuple(prefixes or ["Think:"])
        self._thinking = False
        self._thinking_turns = 0
        self._max_thinking_turns = max_thinking_turns
        self._thinking_budget_exhausted = False
        self._token_count = 0
        self._pending_tokens = []
        self._pending_str = ""

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
            self._logger.debug(
                "Thinking budget exhausted and no longer thinking"
            )
            return [token]

        token_clean = token.strip()
        expecting_tag = self._end_tag if self._thinking else self._start_tag
        result: list[Any] = []

        if self._pending_tokens:
            candidate = self._pending_str + token_clean
            if expecting_tag.startswith(candidate):
                self._pending_tokens.append(token)
                self._pending_str += token_clean
                while len(self._pending_str) > len(expecting_tag):
                    removed = self._pending_tokens.pop(0)
                    removed_clean = removed.strip()
                    self._pending_str = self._pending_str[len(removed_clean) :]
                if candidate == expecting_tag:
                    return self._set_thinking(
                        result, expecting_tag == self._start_tag
                    )
                return result
            result.extend(self._flush_pending(self._thinking))

        if token_clean in (self._start_tag, self._end_tag) or (
            not self._thinking and token_clean.startswith(self._prefixes)
        ):
            return self._set_thinking(
                result, token_clean != self._end_tag, token=token
            )

        if expecting_tag.startswith(token_clean):
            self._pending_tokens.append(token)
            self._pending_str += token_clean
            while len(self._pending_str) > len(expecting_tag):
                removed = self._pending_tokens.pop(0)
                removed_clean = removed.strip()
                self._pending_str = self._pending_str[len(removed_clean) :]
            if token_clean == expecting_tag:
                return self._set_thinking(
                    result, expecting_tag == self._start_tag
                )
            return result

        if self._thinking:
            within_budget = (
                self._settings.max_new_tokens is None
                or self._token_count < self._settings.max_new_tokens
            )
            if within_budget:
                self._logger.debug('Adding reasoning token "%s"', token)
                result.extend(self._wrap(token))
                return result
            if self._settings.stop_on_max_new_tokens:
                self._logger.debug(
                    "Maximum token limit %s reached",
                    self._settings.max_new_tokens,
                )
                raise ReasoningTokenLimitExceeded

            self._logger.debug(
                'Adding reasoning token "%s" after budget exceeded', token
            )
            result.append(token)
            return result

        result.append(token)
        return result

    async def flush(self) -> Iterable[Any]:
        result: list[Any] = []
        if self._pending_tokens:
            as_reasoning = self._thinking
            for t in self._pending_tokens:
                if as_reasoning:
                    self._token_count += 1
                    self._logger.debug('Flushing reasoning token "%s"', t)
                    result.append(ReasoningToken(t))
                else:
                    self._logger.debug('Flushing token "%s"', t)
                    result.append(t)
            self._pending_tokens.clear()
            self._pending_str = ""
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
            self._logger.debug('Adding reasoning token "%s"', token)
            result.extend(self._wrap(token))
        return result

    def _wrap(self, t: str) -> list[Any]:
        self._token_count += 1
        return [ReasoningToken(t)]

    def _flush_pending(self, as_reasoning: bool) -> list[Any]:
        result: list[Any] = []
        if self._pending_tokens:
            for t in self._pending_tokens:
                self._logger.debug(
                    'Flushing pending parser token "%s". Reasoning: %s',
                    t,
                    as_reasoning,
                )
                result.extend(self._wrap(t) if as_reasoning else [t])
            self._pending_tokens.clear()
            self._pending_str = ""
        return result
