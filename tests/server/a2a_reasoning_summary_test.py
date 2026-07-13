from types import SimpleNamespace

import pytest

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamVisibility,
)
from avalan.server.a2a import router as a2a_router
from avalan.server.a2a.router import A2AResponseTranslator
from avalan.server.entities import (
    SKILL_CONTENT_REDACTION,
    ServerOutputRedactionSettings,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake_a2a_imports(monkeypatch) -> None:
    real_import_module = a2a_router.import_module
    fake_pb2 = SimpleNamespace(
        Part=lambda **kwargs: SimpleNamespace(**kwargs),
        TaskState=SimpleNamespace(TASK_STATE_WORKING="working"),
    )

    def fake_import_module(name: str):
        if name == "a2a.types.a2a_pb2":
            return fake_pb2
        return real_import_module(name)

    monkeypatch.setattr(a2a_router, "import_module", fake_import_module)


class _Updater:
    def __init__(self) -> None:
        self.artifacts: list[dict[str, object]] = []
        self.events: list[tuple[str, object]] = []
        self.completed = 0
        self.cancelled = 0
        self.failed_count = 0

    async def add_artifact(self, parts, **kwargs: object) -> None:
        event = {"parts": parts, **kwargs}
        self.artifacts.append(event)
        self.events.append(("artifact", event))

    async def update_status(self, state, metadata=None) -> None:
        self.events.append(("status", {"state": state, "metadata": metadata}))

    async def complete(self) -> None:
        self.completed += 1
        self.events.append(("complete", None))

    async def cancel(self) -> None:
        self.cancelled += 1
        self.events.append(("cancel", None))

    async def failed(self) -> None:
        self.failed_count += 1
        self.events.append(("failed", None))


def _reasoning(
    sequence: int,
    text: str,
    *,
    run_id: str = "run",
    representation: StreamReasoningRepresentation = (
        StreamReasoningRepresentation.SUMMARY
    ),
    ordinal: int = 0,
    provider_item_id: str | None = "provider-repeat",
    output_index: int | None = 0,
    summary_index: int | None = 0,
    continuation_id: str | None = "continuation-a",
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream",
        run_id=run_id,
        turn_id="turn",
        sequence=sequence,
        kind=StreamItemKind.REASONING_DELTA,
        channel=StreamChannel.REASONING,
        correlation=StreamItemCorrelation(
            protocol_item_id=provider_item_id,
            provider_output_index=output_index,
            provider_summary_index=summary_index,
            model_continuation_id=continuation_id,
        ),
        text_delta=text,
        visibility=StreamVisibility.PRIVATE,
        reasoning_representation=representation,
        segment_instance_ordinal=ordinal,
    )


def _answer(sequence: int, text: str) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream",
        run_id="run",
        turn_id="turn",
        sequence=sequence,
        kind=StreamItemKind.ANSWER_DELTA,
        channel=StreamChannel.ANSWER,
        text_delta=text,
    )


def _terminal(
    sequence: int,
    outcome: StreamTerminalOutcome,
) -> CanonicalStreamItem:
    kinds = {
        StreamTerminalOutcome.COMPLETED: StreamItemKind.STREAM_COMPLETED,
        StreamTerminalOutcome.ERRORED: StreamItemKind.STREAM_ERRORED,
        StreamTerminalOutcome.CANCELLED: StreamItemKind.STREAM_CANCELLED,
    }
    return CanonicalStreamItem(
        stream_session_id="stream",
        run_id="run",
        turn_id="turn",
        sequence=sequence,
        kind=kinds[outcome],
        channel=StreamChannel.CONTROL,
        terminal_outcome=outcome,
    )


def _continuation_started(
    sequence: int,
    continuation_id: str,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream",
        run_id="run",
        turn_id="turn",
        sequence=sequence,
        kind=StreamItemKind.MODEL_CONTINUATION_STARTED,
        channel=StreamChannel.CONTROL,
        correlation=StreamItemCorrelation(
            model_continuation_id=continuation_id
        ),
    )


def _artifact_text(events: list[dict[str, object]]) -> str:
    return "".join(
        getattr(part, "text", "")
        for event in events
        for part in event["parts"]  # type: ignore[union-attr]
    )


@pytest.mark.anyio
async def test_reasoning_artifact_ids_grouping_and_answer_isolation(
    fake_a2a_imports,
) -> None:
    updater = _Updater()
    translator = A2AResponseTranslator(updater)

    await translator.process(_reasoning(0, "sum"))
    await translator.process(_reasoning(1, "mary"))
    await translator.process(
        _reasoning(2, "second", ordinal=1, summary_index=1)
    )
    await translator.process(_answer(3, "answer-only"))
    await translator.process(
        _reasoning(
            4,
            "native",
            representation=StreamReasoningRepresentation.NATIVE_TEXT,
            continuation_id="continuation-b",
            summary_index=None,
        )
    )
    await translator.process(_terminal(5, StreamTerminalOutcome.COMPLETED))
    await translator.finish()

    reasoning = [
        event
        for event in updater.artifacts
        if str(event["artifact_id"]).startswith("reasoning-")
    ]
    assert [event["artifact_id"] for event in reasoning] == [
        "reasoning-run-0-0",
        "reasoning-run-0-0",
        "reasoning-run-0-0",
        "reasoning-run-0-1",
        "reasoning-run-0-1",
        "reasoning-run-1-0",
        "reasoning-run-1-0",
    ]
    first_metadata = reasoning[0]["metadata"]
    assert first_metadata == {
        "kind": "reasoning",
        "channel": "reasoning",
        "representation": "summary",
        "segment_instance_ordinal": 0,
        "status": "in_progress",
        "terminal_outcome": None,
        "truncation": {
            "truncated": False,
            "dropped_artifacts": 0,
            "dropped_characters": 0,
            "dropped_utf8_bytes": 0,
        },
        "provider_item_id": "provider-repeat",
        "output_index": 0,
        "summary_index": 0,
        "continuation_id": "continuation-a",
    }
    assert reasoning[-1]["metadata"]["terminal_outcome"] == "completed"  # type: ignore[index]
    answer_events = [
        event
        for event in updater.artifacts
        if event["artifact_id"] == "answer"
    ]
    assert _artifact_text(answer_events) == "answer-only"
    assert "summary" not in _artifact_text(answer_events)
    assert updater.completed == 1

    hinted_updater = _Updater()
    hinted = A2AResponseTranslator(hinted_updater)
    await hinted.process(_continuation_started(0, "continuation-hint"))
    await hinted.process(
        _reasoning(
            1,
            "provider-neutral",
            provider_item_id=None,
            output_index=None,
            summary_index=None,
            continuation_id=None,
        )
    )
    assert hinted_updater.artifacts[0]["artifact_id"] == "reasoning-run-0-0"
    assert "continuation_id" not in hinted_updater.artifacts[0]["metadata"]  # type: ignore[operator]


@pytest.mark.anyio
async def test_reasoning_retention_boundaries_drop_and_recover(
    fake_a2a_imports,
) -> None:
    cases = (
        ("ascii", 3, 3, ("ab", "c", "d", "e"), "z"),
        ("unicode", 10, 6, ("雪é", "a", "b", "c"), "雪"),
    )
    for label, character_limit, byte_limit, chunks, recovery in cases:
        updater = _Updater()
        translator = A2AResponseTranslator(
            updater,
            retention_policy=StreamRetentionPolicy(
                a2a_reasoning_segment_limit=2,
                a2a_reasoning_character_limit=character_limit,
                a2a_reasoning_text_byte_limit=byte_limit,
            ),
        )
        for sequence, chunk in enumerate(chunks):
            await translator.process(_reasoning(sequence, chunk))
        await translator.process(_reasoning(10, recovery, ordinal=1))
        await translator.process(
            _terminal(11, StreamTerminalOutcome.COMPLETED)
        )
        await translator.finish()

        projected = _artifact_text(updater.artifacts)
        assert chunks[0] + chunks[1] in projected, label
        assert chunks[2] not in projected, label
        assert chunks[3] not in projected, label
        assert recovery in projected, label
        first_close = next(
            event
            for event in updater.artifacts
            if event["artifact_id"] == "reasoning-run-0-0"
            and event.get("last_chunk") is True
        )
        truncation = first_close["metadata"]["truncation"]  # type: ignore[index]
        assert truncation["truncated"] is True
        assert truncation["dropped_characters"] == len(chunks[2] + chunks[3])
        assert truncation["dropped_utf8_bytes"] == len(
            (chunks[2] + chunks[3]).encode("utf-8")
        )

    dropped_updater = _Updater()
    dropped = A2AResponseTranslator(
        dropped_updater,
        retention_policy=StreamRetentionPolicy(
            a2a_reasoning_segment_limit=1,
            a2a_reasoning_character_limit=16,
            a2a_reasoning_text_byte_limit=16,
        ),
    )
    await dropped.process(_reasoning(0, "old"))
    await dropped.process(_reasoning(1, "new", ordinal=1))
    await dropped.process(_terminal(2, StreamTerminalOutcome.COMPLETED))
    await dropped.finish()
    new_event = next(
        event
        for event in dropped_updater.artifacts
        if event["artifact_id"] == "reasoning-run-0-1" and event["parts"]
    )
    assert new_event["metadata"]["truncation"] == {  # type: ignore[index]
        "truncated": True,
        "dropped_artifacts": 1,
        "dropped_characters": 3,
        "dropped_utf8_bytes": 3,
    }

    rejected_updater = _Updater()
    rejected = A2AResponseTranslator(
        rejected_updater,
        retention_policy=StreamRetentionPolicy(
            a2a_reasoning_segment_limit=1,
            a2a_reasoning_character_limit=3,
            a2a_reasoning_text_byte_limit=3,
        ),
    )
    await rejected.process(_reasoning(0, "four"))
    await rejected.process(_reasoning(1, "ok", ordinal=1))
    first_rejected = rejected_updater.artifacts[0]
    assert first_rejected["parts"] == []
    assert first_rejected["last_chunk"] is True
    assert first_rejected["metadata"]["truncation"]["dropped_characters"] == 4  # type: ignore[index]
    assert _artifact_text(rejected_updater.artifacts) == "ok"


@pytest.mark.anyio
async def test_reasoning_redaction_identity_latch_and_quarantine(
    fake_a2a_imports,
) -> None:
    settings = ServerOutputRedactionSettings(enabled=True)
    marker_characters = len(SKILL_CONTENT_REDACTION)
    marker_bytes = len(SKILL_CONTENT_REDACTION.encode("utf-8"))
    admission_cases = (
        (
            "summary-part-characters",
            "#",
            marker_characters + 1,
            128,
            {"ordinal": 1, "summary_index": 1},
        ),
        (
            "representation-utf8-bytes",
            "# 雪",
            128,
            marker_bytes + len("# 雪".encode("utf-8")),
            {
                "ordinal": 0,
                "representation": StreamReasoningRepresentation.NATIVE_TEXT,
            },
        ),
    )
    for (
        label,
        pending,
        character_boundary,
        byte_boundary,
        changes,
    ) in admission_cases:
        for offset in (-1, 0, 1):
            updater = _Updater()
            character_limit = (
                character_boundary + offset
                if label.endswith("characters")
                else character_boundary
            )
            byte_limit = (
                byte_boundary + offset
                if label.endswith("bytes")
                else byte_boundary
            )
            translator = A2AResponseTranslator(
                updater,
                output_redaction_settings=settings,
                retention_policy=StreamRetentionPolicy(
                    a2a_reasoning_segment_limit=4,
                    a2a_reasoning_character_limit=character_limit,
                    a2a_reasoning_text_byte_limit=byte_limit,
                ),
            )
            await translator.process(_reasoning(0, pending))
            await translator.process(_reasoning(1, "x", **changes))
            await translator.process(_reasoning(2, "recovered", ordinal=2))
            await translator.process(
                _terminal(3, StreamTerminalOutcome.COMPLETED)
            )
            await translator.finish()

            projected = _artifact_text(updater.artifacts)
            if offset < 0:
                assert SKILL_CONTENT_REDACTION not in projected, label
                assert "xrecovered" in projected, label
                rejected_close = next(
                    event
                    for event in updater.artifacts
                    if event["artifact_id"] == "reasoning-run-0-0"
                    and event.get("last_chunk") is True
                )
                assert rejected_close["metadata"]["status"] == "incomplete"  # type: ignore[index]
                assert rejected_close["metadata"]["truncation"][  # type: ignore[index]
                    "dropped_characters"
                ] == len(
                    pending
                )
            else:
                assert projected == SKILL_CONTENT_REDACTION, label
                assert "recovered" not in projected, label
            assert updater.completed == 1

    rejected_marker_updater = _Updater()
    rejected_marker = A2AResponseTranslator(
        rejected_marker_updater,
        output_redaction_settings=settings,
        retention_policy=StreamRetentionPolicy(
            a2a_reasoning_segment_limit=4,
            a2a_reasoning_character_limit=128,
            a2a_reasoning_text_byte_limit=128,
        ),
    )
    await rejected_marker.process(_reasoning(0, "#"))
    rejected_marker._reasoning_character_limit = marker_characters - 1
    await rejected_marker.process(
        _reasoning(1, "REJECTED_SECRET", ordinal=1, summary_index=1)
    )
    rejected_marker._reasoning_character_limit = 128
    await rejected_marker.process(_reasoning(2, "recovered", ordinal=2))
    await rejected_marker.process(
        _terminal(3, StreamTerminalOutcome.COMPLETED)
    )
    await rejected_marker.finish()
    rejected_marker_text = _artifact_text(rejected_marker_updater.artifacts)
    assert SKILL_CONTENT_REDACTION not in rejected_marker_text
    assert "REJECTED_SECRET" not in rejected_marker_text
    assert rejected_marker_text == "recovered"
    rejected_closes = [
        event
        for event in rejected_marker_updater.artifacts
        if event.get("last_chunk") is True
        and event["metadata"]["status"] == "incomplete"  # type: ignore[index]
    ]
    assert [
        event["metadata"]["truncation"]["dropped_characters"]  # type: ignore[index]
        for event in rejected_closes
    ] == [1, len("REJECTED_SECRET")]
    assert rejected_marker_updater.completed == 1

    boundary_cases = (
        {"ordinal": 1, "summary_index": 1},
        {
            "ordinal": 0,
            "representation": StreamReasoningRepresentation.NATIVE_TEXT,
        },
    )
    for changes in boundary_cases:
        updater = _Updater()
        translator = A2AResponseTranslator(
            updater,
            output_redaction_settings=settings,
        )
        await translator.process(_reasoning(0, "# Demo Skill\n\n"))
        await translator.process(_reasoning(1, "CROSS_PART_SECRET", **changes))
        await translator.process(_reasoning(2, "LATER_SECRET", ordinal=2))
        await translator.process(
            _reasoning(
                3,
                "SECOND_LATER_SECRET",
                ordinal=3,
                continuation_id="continuation-b",
            )
        )
        projected = _artifact_text(updater.artifacts)
        assert projected == SKILL_CONTENT_REDACTION
        assert "SECRET" not in projected

    fresh_updater = _Updater()
    fresh = A2AResponseTranslator(
        fresh_updater,
        output_redaction_settings=settings,
    )
    await fresh.process(_reasoning(0, "fresh owner"))
    assert _artifact_text(fresh_updater.artifacts) == "fresh owner"

    quarantine_updater = _Updater()
    quarantine = A2AResponseTranslator(quarantine_updater)
    await quarantine.process(_reasoning(0, "ordinary"))
    await quarantine.process(
        _reasoning(
            1,
            "QUARANTINED_SECRET",
            ordinal=1,
            provider_item_id=None,
            output_index=None,
            summary_index=None,
            continuation_id=None,
        )
    )
    await quarantine.process(
        _reasoning(
            2,
            "STILL_QUARANTINED",
            ordinal=1,
            provider_item_id=None,
            output_index=None,
            summary_index=None,
            continuation_id=None,
        )
    )
    await quarantine.process(_reasoning(3, "resumed", ordinal=2))
    quarantine_text = _artifact_text(quarantine_updater.artifacts)
    assert quarantine_text == "ordinaryresumed"


@pytest.mark.anyio
async def test_reasoning_abnormal_terminal_and_local_close_order(
    fake_a2a_imports,
) -> None:
    for outcome, terminal_name, terminal_event_name in (
        (StreamTerminalOutcome.ERRORED, "failed", "failed"),
        (StreamTerminalOutcome.CANCELLED, "cancelled", "cancel"),
    ):
        updater = _Updater()
        translator = A2AResponseTranslator(updater)
        await translator.process(_reasoning(0, "observed-prefix"))
        await translator.process(_terminal(1, outcome))
        await translator.process(_answer(2, "LATE_ANSWER"))
        await translator.finish()
        await translator.finish()
        await translator.abort(outcome)

        close_index = next(
            index
            for index, (kind, event) in enumerate(updater.events)
            if kind == "artifact"
            and event["artifact_id"] == "reasoning-run-0-0"  # type: ignore[index]
            and event.get("last_chunk") is True  # type: ignore[union-attr]
        )
        terminal_index = next(
            index
            for index, (kind, _event) in enumerate(updater.events)
            if kind == terminal_event_name
        )
        assert close_index < terminal_index
        close = updater.events[close_index][1]
        assert close["metadata"]["status"] == "incomplete"  # type: ignore[index]
        assert close["metadata"]["terminal_outcome"] == terminal_name  # type: ignore[index]
        assert "LATE_ANSWER" not in _artifact_text(updater.artifacts)
        assert updater.failed_count + updater.cancelled == 1

    pending_updater = _Updater()
    pending = A2AResponseTranslator(
        pending_updater,
        output_redaction_settings=ServerOutputRedactionSettings(enabled=True),
    )
    await pending.process(_reasoning(0, "#"))
    await pending.process(_terminal(1, StreamTerminalOutcome.COMPLETED))
    await pending.finish()
    assert _artifact_text(pending_updater.artifacts) == SKILL_CONTENT_REDACTION
    assert pending_updater.completed == 1

    local_updater = _Updater()
    local = A2AResponseTranslator(local_updater)
    await local.process(_reasoning(0, "prefix"))
    before_close = len(local_updater.events)
    await local.process(
        CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=1,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )
    )
    await local.finish()
    assert len(local_updater.events) == before_close
