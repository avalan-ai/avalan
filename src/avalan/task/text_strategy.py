from .validation import (
    TaskValidationCategory,
    TaskValidationIssue,
)

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from re import findall
from types import MappingProxyType

TokenCounter = Callable[[str], int]


class TextStrategyKind(StrEnum):
    INLINE = "inline"
    RETRIEVAL = "retrieval"
    MAP_REDUCE = "map_reduce"
    REJECT = "reject"


@dataclass(frozen=True, slots=True, kw_only=True)
class TextChunk:
    file_index: int
    chunk_index: int
    start_token: int
    end_token: int
    text: str
    token_count: int

    def __post_init__(self) -> None:
        assert isinstance(self.file_index, int)
        assert not isinstance(self.file_index, bool)
        assert self.file_index >= 0
        assert isinstance(self.chunk_index, int)
        assert not isinstance(self.chunk_index, bool)
        assert self.chunk_index >= 0
        assert isinstance(self.start_token, int)
        assert not isinstance(self.start_token, bool)
        assert self.start_token >= 0
        assert isinstance(self.end_token, int)
        assert not isinstance(self.end_token, bool)
        assert self.end_token > self.start_token
        assert isinstance(self.text, str)
        assert self.text
        assert isinstance(self.token_count, int)
        assert not isinstance(self.token_count, bool)
        assert self.token_count == self.end_token - self.start_token

    def summary(self) -> MappingProxyType[str, object]:
        return MappingProxyType(
            {
                "file_index": self.file_index,
                "chunk_index": self.chunk_index,
                "start_token": self.start_token,
                "end_token": self.end_token,
                "token_count": self.token_count,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TextStrategyPlan:
    kind: TextStrategyKind
    prompt_texts: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    chunks: tuple[TextChunk, ...] = ()
    selected_chunks: tuple[TextChunk, ...] = ()
    issues: tuple[TaskValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.kind, TextStrategyKind)
        assert isinstance(self.prompt_texts, tuple)
        assert isinstance(self.texts, tuple)
        assert isinstance(self.chunks, tuple)
        assert isinstance(self.selected_chunks, tuple)
        assert isinstance(self.issues, tuple)
        for text in (*self.prompt_texts, *self.texts):
            assert isinstance(text, str)
        for chunk in (*self.chunks, *self.selected_chunks):
            assert isinstance(chunk, TextChunk)
        for issue in self.issues:
            assert isinstance(issue, TaskValidationIssue)


def chunk_text_documents(
    documents: Sequence[str],
    *,
    chunk_tokens: int,
    overlap_tokens: int = 0,
) -> tuple[TextChunk, ...]:
    assert isinstance(documents, Sequence)
    assert isinstance(chunk_tokens, int)
    assert not isinstance(chunk_tokens, bool)
    assert chunk_tokens > 0
    assert isinstance(overlap_tokens, int)
    assert not isinstance(overlap_tokens, bool)
    assert overlap_tokens >= 0
    assert overlap_tokens < chunk_tokens
    chunks: list[TextChunk] = []
    step = chunk_tokens - overlap_tokens
    for file_index, document in enumerate(documents):
        assert isinstance(document, str)
        tokens = _text_tokens(document)
        for chunk_index, start_token in enumerate(range(0, len(tokens), step)):
            window = tokens[start_token : start_token + chunk_tokens]
            end_token = start_token + len(window)
            chunks.append(
                TextChunk(
                    file_index=file_index,
                    chunk_index=chunk_index,
                    start_token=start_token,
                    end_token=end_token,
                    text=" ".join(window),
                    token_count=len(window),
                )
            )
            if end_token == len(tokens):
                break
    return tuple(chunks)


def select_retrieval_chunks(
    chunks: Sequence[TextChunk],
    *,
    query: str,
    top_k: int,
    neighbor_count: int = 0,
) -> tuple[TextChunk, ...]:
    assert isinstance(chunks, Sequence)
    assert isinstance(query, str)
    assert isinstance(top_k, int)
    assert not isinstance(top_k, bool)
    assert top_k > 0
    assert isinstance(neighbor_count, int)
    assert not isinstance(neighbor_count, bool)
    assert neighbor_count >= 0
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
    query_terms = _query_terms(query)
    scored = [
        (_chunk_score(chunk.text, query_terms), chunk)
        for chunk in chunks
        if _chunk_score(chunk.text, query_terms) > 0
    ]
    if scored:
        anchors = [
            chunk
            for _, chunk in sorted(
                scored,
                key=lambda item: (
                    -item[0],
                    item[1].file_index,
                    item[1].chunk_index,
                ),
            )[:top_k]
        ]
    else:
        anchors = list(chunks[:top_k])
    selected = {
        (chunk.file_index, chunk.chunk_index)
        for anchor in anchors
        for chunk in _neighbor_chunks(
            chunks,
            anchor=anchor,
            neighbor_count=neighbor_count,
        )
    }
    return tuple(
        chunk
        for chunk in chunks
        if (chunk.file_index, chunk.chunk_index) in selected
    )


def plan_text_strategy(
    *,
    prompt_texts: Sequence[str],
    document_texts: Sequence[str],
    token_limit: int,
    token_counter: TokenCounter,
    retrieval_top_k: int = 2,
    retrieval_neighbor_count: int = 1,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> TextStrategyPlan:
    assert isinstance(prompt_texts, Sequence)
    assert isinstance(document_texts, Sequence)
    assert isinstance(token_limit, int)
    assert not isinstance(token_limit, bool)
    assert token_limit > 0
    assert callable(token_counter)
    prompts = tuple(prompt_texts)
    documents = tuple(document_texts)
    for text in (*prompts, *documents):
        assert isinstance(text, str)
    inline_texts = (*prompts, *documents)
    if _token_total(inline_texts, token_counter) <= token_limit:
        return TextStrategyPlan(
            kind=TextStrategyKind.INLINE,
            prompt_texts=prompts,
            texts=inline_texts,
        )
    prompt_tokens = _token_total(prompts, token_counter)
    available_tokens = token_limit - prompt_tokens
    if available_tokens <= 0 or not documents:
        return _reject_plan(prompts)
    chunk_token_count = chunk_tokens or min(available_tokens, 256)
    chunk_token_count = max(1, min(chunk_token_count, available_tokens))
    overlap_token_count = (
        overlap_tokens
        if overlap_tokens is not None
        else min(chunk_token_count - 1, max(0, chunk_token_count // 8))
    )
    chunks = chunk_text_documents(
        documents,
        chunk_tokens=chunk_token_count,
        overlap_tokens=overlap_token_count,
    )
    if not chunks:
        return _reject_plan(prompts)
    query = "\n".join(prompts)
    selected_chunks = select_retrieval_chunks(
        chunks,
        query=query,
        top_k=retrieval_top_k,
        neighbor_count=retrieval_neighbor_count,
    )
    retrieval_chunks = _bounded_retrieval_chunks(
        selected_chunks,
        query=query,
        available_tokens=available_tokens,
        token_counter=token_counter,
    )
    if retrieval_chunks and _has_positive_retrieval_match(
        retrieval_chunks,
        query,
    ):
        return TextStrategyPlan(
            kind=TextStrategyKind.RETRIEVAL,
            prompt_texts=prompts,
            texts=(
                *prompts,
                *(chunk.text for chunk in retrieval_chunks),
            ),
            chunks=chunks,
            selected_chunks=retrieval_chunks,
        )
    return TextStrategyPlan(
        kind=TextStrategyKind.MAP_REDUCE,
        prompt_texts=prompts,
        chunks=chunks,
    )


def _bounded_retrieval_chunks(
    chunks: Sequence[TextChunk],
    *,
    query: str,
    available_tokens: int,
    token_counter: TokenCounter,
) -> tuple[TextChunk, ...]:
    query_terms = _query_terms(query)
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            -_chunk_score(chunk.text, query_terms),
            chunk.file_index,
            chunk.chunk_index,
        ),
    )
    selected: list[TextChunk] = []
    total = 0
    for chunk in ordered:
        count = _safe_token_count(chunk.text, token_counter)
        if total + count > available_tokens:
            continue
        selected.append(chunk)
        total += count
    return tuple(
        chunk
        for chunk in chunks
        if (chunk.file_index, chunk.chunk_index)
        in {
            (selected_chunk.file_index, selected_chunk.chunk_index)
            for selected_chunk in selected
        }
    )


def _has_positive_retrieval_match(
    chunks: Sequence[TextChunk],
    query: str,
) -> bool:
    query_terms = _query_terms(query)
    return bool(query_terms) and any(
        _chunk_score(chunk.text, query_terms) > 0 for chunk in chunks
    )


def _neighbor_chunks(
    chunks: Sequence[TextChunk],
    *,
    anchor: TextChunk,
    neighbor_count: int,
) -> tuple[TextChunk, ...]:
    lower = anchor.chunk_index - neighbor_count
    upper = anchor.chunk_index + neighbor_count
    return tuple(
        chunk
        for chunk in chunks
        if chunk.file_index == anchor.file_index
        and lower <= chunk.chunk_index <= upper
    )


def _reject_plan(prompt_texts: tuple[str, ...]) -> TextStrategyPlan:
    return TextStrategyPlan(
        kind=TextStrategyKind.REJECT,
        prompt_texts=prompt_texts,
        issues=(
            TaskValidationIssue(
                code="limits.invalid_value",
                path="limits.total_tokens",
                message="Task input exceeds the configured token limit.",
                hint=(
                    "Reduce the input text, raise the token limit, or use a "
                    "smaller file input."
                ),
                category=TaskValidationCategory.VALUE,
            ),
        ),
    )


def _token_total(
    texts: Sequence[str],
    token_counter: TokenCounter,
) -> int:
    return sum(_safe_token_count(text, token_counter) for text in texts)


def _safe_token_count(text: str, token_counter: TokenCounter) -> int:
    count = token_counter(text)
    assert isinstance(count, int)
    assert not isinstance(count, bool)
    assert count >= 0
    return count


def _chunk_score(text: str, query_terms: frozenset[str]) -> int:
    if not query_terms:
        return 0
    text_terms = _query_terms(text)
    return sum(1 for term in text_terms if term in query_terms)


def _query_terms(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _text_tokens(text))


def _text_tokens(text: str) -> tuple[str, ...]:
    return tuple(findall(r"\S+", text))
