"""Exercise the response-owned private provider-state sidecar."""

from asyncio import CancelledError, Event, create_task
from collections.abc import AsyncIterator
from logging import getLogger
from typing import cast

import pytest

from avalan.conversation import (
    EffectiveReasoningContext,
    EffectiveReasoningMetadata,
    ProviderUsage,
    ReasoningContext,
)
from avalan.entities import GenerationSettings
from avalan.model import (
    ProviderStateError,
    ProviderStateFinalization,
    ProviderStateSink,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
)

pytestmark = pytest.mark.anyio

_PRIVATE_SENTINELS = (
    "opaque-provider-secret",
    "opaque-finalize-secret",
    "opaque-cleanup-secret",
    "opaque-cancellation-secret",
    "opaque-nested-secret",
)


@pytest.fixture
def anyio_backend() -> str:
    """Run cancellation-sensitive response tests on asyncio only."""
    return "asyncio"


class _ProviderStateSink:
    def __init__(
        self,
        *,
        finalize_error: BaseException | None = None,
        finalize_entered: Event | None = None,
        finalize_release: Event | None = None,
        cleanup_error: BaseException | None = None,
    ) -> None:
        self.finalize_error = finalize_error
        self.finalize_entered = finalize_entered
        self.finalize_release = finalize_release
        self.cleanup_error = cleanup_error
        self.finalize_calls = 0
        self.cleanup_calls = 0

    async def finalize(self) -> ProviderStateFinalization:
        self.finalize_calls += 1
        if self.finalize_entered is not None:
            self.finalize_entered.set()
        if self.finalize_release is not None:
            await self.finalize_release.wait()
        if self.finalize_error is not None:
            raise self.finalize_error
        return ProviderStateFinalization(
            reasoning=EffectiveReasoningMetadata(
                requested=ReasoningContext.ALL_TURNS,
                effective=EffectiveReasoningContext.ALL_TURNS,
            ),
            usage=ProviderUsage(input_tokens=7, output_tokens=3),
            item_count=2,
        )

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class _InvalidFinalizationSink(_ProviderStateSink):
    async def finalize(self) -> ProviderStateFinalization:
        self.finalize_calls += 1
        return cast(ProviderStateFinalization, object())


def _private_error(
    message: str,
    *,
    cancellation: bool = False,
) -> BaseException:
    error: BaseException = (
        CancelledError(message) if cancellation else RuntimeError(message)
    )
    error.__cause__ = ValueError("opaque-nested-secret-cause")
    error.__context__ = LookupError("opaque-nested-secret-context")
    return error


def _assert_public_exception_is_chain_free(error: BaseException) -> None:
    """Require a recursively content-safe exception with no linked chain."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered = (str(current), repr(current), repr(current.args))
        assert all(
            sentinel not in value
            for sentinel in _PRIVATE_SENTINELS
            for value in rendered
        )
        cause = current.__cause__
        context = current.__context__
        assert cause is None
        assert context is None
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)


def _response(
    sink: ProviderStateSink,
    *,
    output: str = "visible",
) -> TextGenerationResponse:
    return TextGenerationResponse(
        lambda: output,
        logger=getLogger("provider-state-response"),
        use_async_generator=False,
        generation_settings=GenerationSettings(),
        provider_state_sink=sink,
    )


async def _canonical_output() -> AsyncIterator[CanonicalStreamItem]:
    for item in (
        CanonicalStreamItem(
            stream_session_id="provider-state-stream",
            run_id="provider-state-run",
            turn_id="provider-state-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        ),
        CanonicalStreamItem(
            stream_session_id="provider-state-stream",
            run_id="provider-state-run",
            turn_id="provider-state-turn",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="visible",
        ),
        CanonicalStreamItem(
            stream_session_id="provider-state-stream",
            run_id="provider-state-run",
            turn_id="provider-state-turn",
            sequence=2,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        ),
        CanonicalStreamItem(
            stream_session_id="provider-state-stream",
            run_id="provider-state-run",
            turn_id="provider-state-turn",
            sequence=3,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    ):
        yield item


async def test_nonstream_finalizes_and_cleans_exactly_once() -> None:
    """Finish private state before returning a successful visible result."""
    sink = _ProviderStateSink()
    response = _response(sink)

    assert await response.to_str() == "visible"
    assert await response.to_str() == "visible"
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1
    await response.aclose()
    assert sink.cleanup_calls == 1
    assert response.cleanup_complete


async def test_canonical_stream_finalizes_without_public_sidecar_payload() -> (
    None
):
    """Keep sidecar metadata outside consumer-visible canonical items."""
    sink = _ProviderStateSink()
    response = TextGenerationResponse(
        lambda: _canonical_output(),
        logger=getLogger("provider-state-stream"),
        use_async_generator=True,
        generation_settings=GenerationSettings(),
        provider_state_sink=sink,
    )

    items = [item async for item in response]
    assert [item.kind for item in items] == [
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.ANSWER_DONE,
        StreamItemKind.STREAM_COMPLETED,
    ]
    assert all("provider_state" not in repr(item) for item in items)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_close_without_consuming_only_cleans_sidecar() -> None:
    """Release unconsumed state without falsely finalizing it."""
    sink = _ProviderStateSink()
    response = _response(sink)

    await response.aclose()
    await response.aclose()
    assert sink.finalize_calls == 0
    assert sink.cleanup_calls == 1
    assert response.cleanup_complete


async def test_finalization_failure_is_content_safe_and_cleans() -> None:
    """Replace private sink failures with one stable public error."""
    sink = _ProviderStateSink(
        finalize_error=_private_error("opaque-provider-secret")
    )
    response = _response(sink)

    with pytest.raises(ProviderStateError) as failure:
        await response.to_str()
    _assert_public_exception_is_chain_free(failure.value)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_invalid_finalization_type_is_content_safe_and_cleans() -> None:
    """Map malformed private completion output to the stable public error."""
    sink = _InvalidFinalizationSink()
    response = _response(sink)

    with pytest.raises(ProviderStateError) as failure:
        await response.to_str()
    _assert_public_exception_is_chain_free(failure.value)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_sink_owned_finalization_cancellation_still_cleans() -> None:
    """Propagate sink cancellation after releasing its private resources."""
    sink = _ProviderStateSink(
        finalize_error=_private_error(
            "opaque-cancellation-secret",
            cancellation=True,
        )
    )
    response = _response(sink)

    with pytest.raises(CancelledError) as failure:
        await response.to_str()
    _assert_public_exception_is_chain_free(failure.value)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_sink_owned_cleanup_cancellation_is_propagated() -> None:
    """Preserve cleanup cancellation without exposing private state."""
    sink = _ProviderStateSink(
        cleanup_error=_private_error(
            "opaque-cancellation-secret",
            cancellation=True,
        )
    )
    response = _response(sink)

    with pytest.raises(CancelledError) as failure:
        await response.to_str()
    _assert_public_exception_is_chain_free(failure.value)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_cleanup_failure_is_content_safe() -> None:
    """Map private cleanup failure to the stable public error."""
    sink = _ProviderStateSink(
        cleanup_error=_private_error("opaque-cleanup-secret")
    )
    response = _response(sink)

    with pytest.raises(ProviderStateError) as failure:
        await response.to_str()
    _assert_public_exception_is_chain_free(failure.value)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_finalization_and_cleanup_failures_are_content_safe() -> None:
    """Discard both private lifecycle chains from the public error."""
    sink = _ProviderStateSink(
        finalize_error=_private_error("opaque-finalize-secret"),
        cleanup_error=_private_error("opaque-cleanup-secret"),
    )
    response = _response(sink)

    with pytest.raises(ProviderStateError) as failure:
        await response.to_str()
    _assert_public_exception_is_chain_free(failure.value)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_cancellation_waits_for_finalize_and_cleanup() -> None:
    """Shield single-owner finalization and then restore cancellation."""
    entered = Event()
    release = Event()
    sink = _ProviderStateSink(
        finalize_entered=entered,
        finalize_release=release,
    )
    response = _response(sink)
    task = create_task(response.to_str())
    await entered.wait()

    task.cancel()
    release.set()
    with pytest.raises(CancelledError) as failure:
        await task
    _assert_public_exception_is_chain_free(failure.value)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_cancellation_discards_cleanup_failure_chain() -> None:
    """Restore caller cancellation without retaining private cleanup state."""
    entered = Event()
    release = Event()
    sink = _ProviderStateSink(
        finalize_entered=entered,
        finalize_release=release,
        cleanup_error=_private_error("opaque-cleanup-secret"),
    )
    response = _response(sink)
    task = create_task(response.to_str())
    await entered.wait()

    task.cancel()
    release.set()
    with pytest.raises(CancelledError) as failure:
        await task
    _assert_public_exception_is_chain_free(failure.value)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1


async def test_concurrent_completion_reuses_single_owned_tasks() -> None:
    """Reuse in-flight finalization and cleanup across concurrent joiners."""
    entered = Event()
    release = Event()
    sink = _ProviderStateSink(
        finalize_entered=entered,
        finalize_release=release,
    )
    response = _response(sink)
    first = create_task(response._complete_provider_state())
    await entered.wait()
    second = create_task(response._complete_provider_state())

    release.set()
    await first
    await second
    await response._complete_provider_state()

    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1
    assert response._provider_state_cleaned


@pytest.mark.parametrize(
    ("cleanup_error", "public_error"),
    (
        (
            _private_error(
                "opaque-cancellation-secret",
                cancellation=True,
            ),
            CancelledError,
        ),
        (
            _private_error("opaque-cleanup-secret"),
            ProviderStateError,
        ),
    ),
)
async def test_direct_cleanup_join_discards_private_task_failures(
    cleanup_error: BaseException,
    public_error: type[BaseException],
) -> None:
    """Map an owned cleanup task result without retaining its exception."""
    response = _response(_ProviderStateSink(cleanup_error=cleanup_error))

    with pytest.raises(public_error) as failure:
        await response._cleanup_provider_state()

    _assert_public_exception_is_chain_free(failure.value)


async def test_direct_completion_discards_failed_finalization_result() -> None:
    """Map a failed finalization join through the public lifecycle type."""
    response = _response(
        _ProviderStateSink(
            finalize_error=_private_error("opaque-finalize-secret")
        )
    )

    with pytest.raises(ProviderStateError) as failure:
        await response._complete_provider_state()

    _assert_public_exception_is_chain_free(failure.value)


def test_response_string_and_repr_do_not_render_private_sidecar() -> None:
    """Keep the response display surfaces independent of private state."""
    response = _response(
        _ProviderStateSink(
            finalize_error=_private_error("opaque-provider-secret")
        )
    )

    assert str(response) == "visible"
    assert all(
        sentinel not in repr(response) for sentinel in _PRIVATE_SENTINELS
    )


def test_synchronous_sidecar_is_rejected_at_construction() -> None:
    """Reject a structurally similar synchronous provider-state object."""

    class SyncSink:
        def finalize(self) -> ProviderStateFinalization:
            raise AssertionError

        def cleanup(self) -> None:
            raise AssertionError

    with pytest.raises(TypeError, match="asynchronously"):
        _response(cast(ProviderStateSink, SyncSink()))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("reasoning", object(), "reasoning"),
        ("usage", object(), "usage"),
        ("item_count", -1, "item_count"),
    ),
)
def test_provider_state_finalization_rejects_invalid_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject malformed private sidecar completion metadata."""
    values: dict[str, object] = {
        "reasoning": EffectiveReasoningMetadata(
            requested=ReasoningContext.AUTO,
            effective=EffectiveReasoningContext.CURRENT_TURN,
        ),
        "usage": ProviderUsage(input_tokens=1, output_tokens=1),
        "item_count": 1,
    }
    values[field] = value
    with pytest.raises(TypeError, match=message):
        ProviderStateFinalization(
            reasoning=cast(
                EffectiveReasoningMetadata,
                values["reasoning"],
            ),
            usage=cast(ProviderUsage, values["usage"]),
            item_count=cast(int, values["item_count"]),
        )
