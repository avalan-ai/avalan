"""Reject synchronous substitutes for every Phase 4 direct SDK effect."""

from collections.abc import Awaitable, Callable

from avalan import (
    DirectConversationStreamItem,
    StandaloneCompactRequest,
    StandaloneCompactResult,
)
from avalan.conversation import (
    AtomicCommitReceipt,
    ConversationCoordinator,
    ConversationProviderStateSink,
    ConversationRunRequest,
    ProviderItem,
    ProviderLaneOutputCandidate,
)
from avalan.model import (
    ProviderStateFinalization,
    ProviderStateSink,
)


class SyncCoordinator:
    """Run every coordinator operation synchronously."""

    def execute(self, request: ConversationRunRequest) -> AtomicCommitReceipt:
        """Execute without an awaitable boundary."""
        raise NotImplementedError

    def stream(self, request: ConversationRunRequest) -> AtomicCommitReceipt:
        """Stream without an awaitable boundary."""
        raise NotImplementedError

    def stream_with_sink(
        self,
        request: ConversationRunRequest,
        sink: ConversationProviderStateSink,
    ) -> AtomicCommitReceipt:
        """Stream private state without an awaitable boundary."""
        raise NotImplementedError

    def compact(self, request: ConversationRunRequest) -> AtomicCommitReceipt:
        """Compact without an awaitable boundary."""
        raise NotImplementedError


class SyncConversationSink:
    """Stage and clean coordinator provider state synchronously."""

    def stage(self, item: ProviderItem) -> None:
        """Stage without an awaitable boundary."""
        raise NotImplementedError

    def finalize(
        self,
        outputs: tuple[ProviderLaneOutputCandidate, ...],
    ) -> None:
        """Finalize without an awaitable boundary."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Clean without an awaitable boundary."""
        raise NotImplementedError


class SyncResponseSink:
    """Finalize response-owned provider state synchronously."""

    def finalize(self) -> ProviderStateFinalization:
        """Finalize without an awaitable boundary."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Clean without an awaitable boundary."""
        raise NotImplementedError


def sync_compact(
    request: StandaloneCompactRequest,
) -> StandaloneCompactResult:
    """Compact synchronously."""
    raise NotImplementedError


def sync_callback(item: DirectConversationStreamItem) -> None:
    """Observe a direct stream item synchronously."""
    raise NotImplementedError


def sync_cleanup() -> None:
    """Clean a direct operation synchronously."""
    raise NotImplementedError


coordinator: ConversationCoordinator = SyncCoordinator()
conversation_sink: ConversationProviderStateSink = SyncConversationSink()
response_sink: ProviderStateSink = SyncResponseSink()
compact_handler: Callable[
    [StandaloneCompactRequest], Awaitable[StandaloneCompactResult]
] = sync_compact
callback: Callable[[DirectConversationStreamItem], Awaitable[None]] = (
    sync_callback
)
cleanup: Callable[[], Awaitable[None]] = sync_cleanup
