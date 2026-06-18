from ....entities import ReasoningSettings, ReasoningTag, ReasoningToken
from ...stream import StreamItemKind, StreamProviderEvent, StreamVisibility

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
    _reasoning_delta_emitted: bool
    _token_count: int
    _pending_tokens: list[str]
    _pending_str: str
    _logger: Logger
    _legacy_fixture: bool

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
        legacy_fixture: bool = False,
    ) -> None:
        self._settings = reasoning_settings
        self._logger = logger
        self._legacy_fixture = legacy_fixture
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
        self._reasoning_delta_emitted = False
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
        if not token:
            return []

        token_clean = token.strip()
        expecting_start = not self._thinking
        expecting_tag = self._marker_text(expecting_start)
        result: list[Any] = []

        if self._pending_tokens:
            if not token_clean:
                result.extend(self._flush_pending(self._thinking))
                if self._thinking:
                    return self._thinking_token_result(token, result)
                result.append(token)
                return result
            candidate = self._pending_str + token_clean
            if expecting_tag.startswith(candidate):
                self._pending_tokens.append(token)
                self._pending_str += token_clean
                while len(self._pending_str) > len(expecting_tag):
                    removed = self._pending_tokens.pop(0)
                    removed_clean = removed.strip()
                    self._pending_str = self._pending_str[len(removed_clean) :]
                if candidate == expecting_tag:
                    return self._set_thinking_from_pending(
                        result, expecting_start
                    )
                return result
            if isinstance(expecting_tag, str) and candidate.startswith(
                expecting_tag
            ):
                needed_length = len(expecting_tag) - len(self._pending_str)
                tag_start = len(token) - len(token.lstrip())
                tag_end = tag_start + needed_length
                self._pending_tokens.append(token[:tag_end])
                self._pending_str += token_clean[:needed_length]
                result = self._set_thinking_from_pending(
                    result, expecting_start
                )
                remainder = token[tag_end:]
                if remainder:
                    result.extend(await self.push(remainder))
                return result
            result.extend(self._flush_pending(self._thinking))

        end_tag = self._marker_text(False)
        if token_clean in (self._marker_text(True), end_tag):
            return self._set_thinking_from_token(
                result, token_clean != end_tag, token
            )

        if not self._thinking and token_clean.startswith(self._prefixes):
            return self._set_thinking(result, True, token=token)

        if token_clean and expecting_tag.startswith(token_clean):
            self._pending_tokens.append(token)
            self._pending_str += token_clean
            return result

        embedded = await self._push_embedded_marker(token)
        if embedded is not None:
            return embedded

        if self._thinking:
            return self._thinking_token_result(token, result)

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
                    result.append(self._reasoning_delta(t))
                else:
                    self._logger.debug('Flushing token "%s"', t)
                    result.append(t)
            self._pending_tokens.clear()
            self._pending_str = ""
        if self._thinking and not self._legacy_fixture:
            self._thinking = False
            result.extend(self._reasoning_done_result())
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
        if not is_start and not self._legacy_fixture:
            result.extend(self._reasoning_done_result())
        return result

    def _set_thinking_from_pending(
        self, result: list[Any], is_start: bool
    ) -> list[Any]:
        raw = "".join(self._pending_tokens)
        marker = self._marker_text(is_start)
        marker_index = raw.find(marker)
        if marker_index == -1:
            result.extend(self._flush_pending(self._thinking))
            return result
        marker_end = marker_index + len(marker)
        prefix, marker_parts, suffix = self._pending_marker_segments(
            marker_index, marker_end
        )
        self._pending_tokens.clear()
        self._pending_str = ""
        return self._set_thinking_from_segments(
            result, is_start, prefix, marker_parts, suffix
        )

    def _set_thinking_from_token(
        self, result: list[Any], is_start: bool, token: str
    ) -> list[Any]:
        marker = self._marker_text(is_start)
        marker_index = token.find(marker)
        assert marker_index != -1
        marker_end = marker_index + len(marker)
        return self._set_thinking_from_segments(
            result,
            is_start,
            [token[:marker_index]],
            [marker],
            [token[marker_end:]],
        )

    def _set_thinking_from_segments(
        self,
        result: list[Any],
        is_start: bool,
        prefix: list[str],
        marker_parts: list[str],
        suffix: list[str],
    ) -> list[Any]:
        if is_start:
            for part in prefix:
                if part:
                    result.extend(
                        self._wrap(part) if self._thinking else [part]
                    )
            self._thinking = True
            self._thinking_turns += 1
            if self._thinking_turns >= self._max_thinking_turns:
                self._thinking_budget_exhausted = True
            for part in (*marker_parts, *suffix):
                if part:
                    result.extend(self._wrap(part))
            return result

        for part in (*prefix, *marker_parts):
            if part:
                result.extend(self._wrap(part))
        self._thinking = False
        if not self._legacy_fixture:
            result.extend(self._reasoning_done_result())
        for part in suffix:
            if part:
                result.append(part)
        return result

    def _pending_marker_segments(
        self, marker_index: int, marker_end: int
    ) -> tuple[list[str], list[str], list[str]]:
        prefix: list[str] = []
        marker_parts: list[str] = []
        suffix: list[str] = []
        position = 0
        for token in self._pending_tokens:
            token_end = position + len(token)
            if position < marker_index:
                prefix_end = min(token_end, marker_index)
                prefix.append(token[: prefix_end - position])
            if token_end > marker_index and position < marker_end:
                part_start = max(marker_index, position) - position
                part_end = min(token_end, marker_end) - position
                marker_parts.append(token[part_start:part_end])
            if token_end > marker_end:
                suffix_start = max(marker_end, position) - position
                suffix.append(token[suffix_start:])
            position = token_end
        return prefix, marker_parts, suffix

    def _marker_text(self, is_start: bool) -> str:
        marker = self._start_tag if is_start else self._end_tag
        marker_value = getattr(marker, "value", marker)
        assert isinstance(marker_value, str)
        return marker_value

    async def _push_embedded_marker(self, token: str) -> list[Any] | None:
        is_start = not self._thinking
        marker = self._marker_text(is_start)

        marker_index = token.find(marker)
        if marker_index != -1:
            result: list[Any] = []
            before = token[:marker_index]
            if before:
                if self._thinking:
                    result = self._thinking_token_result(before, result)
                else:
                    result.append(before)
            marker_text = token[marker_index : marker_index + len(marker)]
            result = self._set_thinking(result, is_start, token=marker_text)
            remainder = token[marker_index + len(marker) :]
            if remainder:
                result.extend(await self.push(remainder))
            return result

        suffix_length = self._marker_suffix_length(token, marker)
        if suffix_length and suffix_length < len(token):
            result = []
            before = token[:-suffix_length]
            if before:
                if self._thinking:
                    result = self._thinking_token_result(before, result)
                else:
                    result.append(before)
            pending = token[-suffix_length:]
            self._pending_tokens.append(pending)
            self._pending_str += pending.strip()
            return result

        return None

    def _thinking_token_result(
        self, token: str, result: list[Any]
    ) -> list[Any]:
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

    def _wrap(self, t: str) -> list[Any]:
        self._token_count += 1
        return [self._reasoning_delta(t)]

    def _reasoning_delta(
        self, token: str
    ) -> ReasoningToken | StreamProviderEvent:
        if self._legacy_fixture:
            return ReasoningToken(token)
        self._reasoning_delta_emitted = True
        return StreamProviderEvent(
            kind=StreamItemKind.REASONING_DELTA,
            text_delta=token,
            visibility=StreamVisibility.PRIVATE,
        )

    def _reasoning_done_result(self) -> list[Any]:
        result: list[Any] = []
        if not self._reasoning_delta_emitted:
            result.append(self._reasoning_delta(""))
        self._reasoning_delta_emitted = False
        return [
            *result,
            StreamProviderEvent(
                kind=StreamItemKind.REASONING_DONE,
                visibility=StreamVisibility.PRIVATE,
            ),
        ]

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

    @staticmethod
    def _marker_suffix_length(token: str, marker: str) -> int:
        max_suffix = min(len(token), len(marker) - 1)
        for length in range(max_suffix, 0, -1):
            if marker.startswith(token[-length:]):
                return length
        return 0
