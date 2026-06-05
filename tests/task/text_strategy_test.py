from collections.abc import Sequence
from unittest import TestCase, main

from avalan.task import (
    TextChunk,
    TextStrategyKind,
    chunk_text_documents,
    plan_text_strategy,
    select_retrieval_chunks,
)


def _count_tokens(text: str) -> int:
    return len(text.split())


class TaskTextStrategyTest(TestCase):
    def test_chunk_text_documents_uses_deterministic_overlap(self) -> None:
        chunks = chunk_text_documents(
            ("one two three four five six",),
            chunk_tokens=3,
            overlap_tokens=1,
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["one two three", "three four five", "five six"],
        )
        self.assertEqual(
            [chunk.summary() for chunk in chunks],
            [
                {
                    "file_index": 0,
                    "chunk_index": 0,
                    "start_token": 0,
                    "end_token": 3,
                    "token_count": 3,
                },
                {
                    "file_index": 0,
                    "chunk_index": 1,
                    "start_token": 2,
                    "end_token": 5,
                    "token_count": 3,
                },
                {
                    "file_index": 0,
                    "chunk_index": 2,
                    "start_token": 4,
                    "end_token": 6,
                    "token_count": 2,
                },
            ],
        )

    def test_chunk_metadata_does_not_include_raw_text(self) -> None:
        chunks = chunk_text_documents(
            ("private-token visible",),
            chunk_tokens=1,
        )

        self.assertNotIn("private-token", str(chunks[0].summary()))

    def test_chunk_text_documents_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(AssertionError):
            chunk_text_documents(
                ("one two",),
                chunk_tokens=2,
                overlap_tokens=2,
            )

    def test_select_retrieval_chunks_includes_neighbors(self) -> None:
        chunks = chunk_text_documents(
            ("alpha beta gamma needle delta epsilon zeta eta theta",),
            chunk_tokens=3,
        )

        selected = select_retrieval_chunks(
            chunks,
            query="needle",
            top_k=1,
            neighbor_count=1,
        )

        self.assertEqual(
            [chunk.text for chunk in selected],
            [
                "alpha beta gamma",
                "needle delta epsilon",
                "zeta eta theta",
            ],
        )

    def test_select_retrieval_chunks_falls_back_without_query(self) -> None:
        chunks = chunk_text_documents(
            ("alpha beta gamma delta",),
            chunk_tokens=2,
        )

        selected = select_retrieval_chunks(
            chunks,
            query="",
            top_k=1,
        )

        self.assertEqual([chunk.text for chunk in selected], ["alpha beta"])

    def test_plan_uses_inline_when_text_fits_budget(self) -> None:
        plan = plan_text_strategy(
            prompt_texts=("summarize",),
            document_texts=("short document",),
            token_limit=3,
            token_counter=_count_tokens,
        )

        self.assertEqual(plan.kind, TextStrategyKind.INLINE)
        self.assertEqual(plan.texts, ("summarize", "short document"))
        self.assertEqual(plan.chunks, ())

    def test_plan_uses_retrieval_when_query_matches(self) -> None:
        plan = plan_text_strategy(
            prompt_texts=("needle",),
            document_texts=(
                "zero one two three needle five six seven eight nine",
            ),
            token_limit=5,
            token_counter=_count_tokens,
            chunk_tokens=4,
            overlap_tokens=0,
        )

        self.assertEqual(plan.kind, TextStrategyKind.RETRIEVAL)
        self.assertEqual(plan.texts, ("needle", "needle five six seven"))
        self.assertEqual(
            [chunk.summary() for chunk in plan.selected_chunks],
            [
                {
                    "file_index": 0,
                    "chunk_index": 1,
                    "start_token": 4,
                    "end_token": 8,
                    "token_count": 4,
                }
            ],
        )

    def test_plan_uses_map_reduce_without_retrieval_match(self) -> None:
        plan = plan_text_strategy(
            prompt_texts=("summarize",),
            document_texts=("alpha beta gamma delta epsilon zeta",),
            token_limit=4,
            token_counter=_count_tokens,
            chunk_tokens=3,
            overlap_tokens=0,
        )

        self.assertEqual(plan.kind, TextStrategyKind.MAP_REDUCE)
        self.assertEqual(
            [chunk.text for chunk in plan.chunks],
            ["alpha beta gamma", "delta epsilon zeta"],
        )

    def test_plan_uses_supplied_retrieval_selector_without_lexical_match(
        self,
    ) -> None:
        def selector(
            chunks: Sequence[TextChunk],
            query: str,
            top_k: int,
            neighbor_count: int,
        ) -> tuple[TextChunk, ...]:
            self.assertEqual(query, "semantic")
            self.assertEqual(top_k, 2)
            self.assertEqual(neighbor_count, 1)
            return (chunks[1],)

        plan = plan_text_strategy(
            prompt_texts=("semantic",),
            document_texts=("alpha beta gamma delta",),
            token_limit=3,
            token_counter=_count_tokens,
            retrieval_selector=selector,
            chunk_tokens=2,
            overlap_tokens=0,
        )

        self.assertEqual(plan.kind, TextStrategyKind.RETRIEVAL)
        self.assertEqual(plan.texts, ("semantic", "gamma delta"))

    def test_plan_rejects_unbounded_map_reduce(self) -> None:
        plan = plan_text_strategy(
            prompt_texts=("summarize",),
            document_texts=("alpha beta gamma delta epsilon zeta",),
            token_limit=2,
            token_counter=_count_tokens,
            chunk_tokens=1,
            overlap_tokens=0,
            max_map_chunks=2,
        )

        self.assertEqual(plan.kind, TextStrategyKind.REJECT)
        self.assertEqual(plan.issues[0].path, "limits.total_tokens")
        self.assertNotIn("alpha beta", str(plan.issues))

    def test_plan_rejects_selector_chunks_outside_plan(self) -> None:
        def selector(
            chunks: Sequence[TextChunk],
            query: str,
            top_k: int,
            neighbor_count: int,
        ) -> tuple[TextChunk, ...]:
            return (
                TextChunk(
                    file_index=9,
                    chunk_index=9,
                    start_token=0,
                    end_token=1,
                    text="outside",
                    token_count=1,
                ),
            )

        with self.assertRaises(AssertionError):
            plan_text_strategy(
                prompt_texts=("semantic",),
                document_texts=("alpha beta gamma delta",),
                token_limit=3,
                token_counter=_count_tokens,
                retrieval_selector=selector,
                chunk_tokens=2,
                overlap_tokens=0,
            )

    def test_plan_rejects_invalid_map_chunk_limit(self) -> None:
        with self.assertRaises(AssertionError):
            plan_text_strategy(
                prompt_texts=("summarize",),
                document_texts=("alpha beta gamma delta",),
                token_limit=2,
                token_counter=_count_tokens,
                max_map_chunks=0,
            )

    def test_plan_rejects_when_prompt_exhausts_budget_safely(self) -> None:
        plan = plan_text_strategy(
            prompt_texts=("one two three",),
            document_texts=("private document",),
            token_limit=2,
            token_counter=_count_tokens,
        )

        self.assertEqual(plan.kind, TextStrategyKind.REJECT)
        self.assertEqual(plan.issues[0].path, "limits.total_tokens")
        self.assertNotIn("private document", str(plan.issues))

    def test_plan_rejects_when_counted_document_cannot_chunk(self) -> None:
        plan = plan_text_strategy(
            prompt_texts=(),
            document_texts=("   ",),
            token_limit=1,
            token_counter=lambda text: 2 if text else 0,
        )

        self.assertEqual(plan.kind, TextStrategyKind.REJECT)
        self.assertEqual(plan.issues[0].path, "limits.total_tokens")


if __name__ == "__main__":
    main()
