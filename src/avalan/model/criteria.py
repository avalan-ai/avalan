from io import StringIO
from re import Pattern, compile, escape
from typing import Any, Protocol

from transformers.generation.stopping_criteria import StoppingCriteria


class TokenDecoder(Protocol):
    """Define the tokenizer contract required by keyword stopping criteria."""

    def decode(
        self, token_ids: int | list[int], skip_special_tokens: bool = False
    ) -> str: ...


class KeywordStoppingCriteria(StoppingCriteria):
    _buffer: StringIO
    _tokenizer: TokenDecoder
    _pattern: Pattern[str]
    _keywords: list[str]
    _keyword_count: int

    def __init__(
        self,
        keywords: list[str],
        tokenizer: TokenDecoder,
        all_must_be_present: bool = False,
    ) -> None:
        assert keywords
        escaped_keywords = [escape(k) for k in keywords]
        self._pattern = compile(
            r"^" + "".join(f"(?=.*{k})" for k in escaped_keywords) + r".*$"
            if all_must_be_present
            else r"(" + "|".join(escaped_keywords) + r")$"
        )
        self._buffer = StringIO()
        self._tokenizer = tokenizer
        self._keywords = keywords
        self._keyword_count = len(keywords)

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        token_id = input_ids[0][-1]
        token = self._tokenizer.decode(token_id, skip_special_tokens=False)
        self._buffer.write(token)
        buffer_contents = self._buffer.getvalue()
        return (
            self._keyword_count == 1
            and buffer_contents.endswith(self._keywords[0])
        ) or bool(self._pattern.search(buffer_contents))
