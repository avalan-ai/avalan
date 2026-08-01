"""Reject synchronous implementations of every async effect protocol."""

from collections.abc import AsyncIterator

from avalan.conversation import (
    AuthorityScope,
    CheckpointCandidate,
    CheckpointId,
    ConversationCheckpoint,
    ConversationCoordinator,
    ConversationObservation,
    ConversationObserver,
    ConversationProvider,
    ConversationProviderStream,
    ConversationRequestSemantics,
    ConversationResult,
    ConversationSettings,
    ConversationStore,
    ConversationStreamTerminal,
    ProviderItem,
    ProviderPlan,
    ProviderResult,
)


class SyncCoordinator:
    """Execute coordinator effects synchronously."""

    def execute(
        self,
        request: ConversationRequestSemantics,
        settings: ConversationSettings,
    ) -> ConversationResult:
        """Fail because execution is not awaitable."""
        del request, settings
        raise NotImplementedError

    def stream(
        self,
        request: ConversationRequestSemantics,
        settings: ConversationSettings,
    ) -> ConversationStreamTerminal:
        """Fail because stream completion is not awaitable."""
        del request, settings
        raise NotImplementedError


class SyncStore:
    """Execute persistence effects synchronously."""

    def load(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        """Fail because loading is not awaitable."""
        del checkpoint_id, authority
        raise NotImplementedError

    def commit(
        self,
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        """Fail because commit is not awaitable."""
        del candidate
        raise NotImplementedError

    def close(self) -> None:
        """Fail because close is not awaitable."""


class SyncProvider:
    """Execute provider dispatch effects synchronously."""

    def dispatch(self, plan: ProviderPlan) -> ProviderResult:
        """Fail because dispatch is not awaitable."""
        del plan
        raise NotImplementedError

    def stream(self, plan: ProviderPlan) -> ConversationProviderStream:
        """Fail because opening a provider stream is not awaitable."""
        del plan
        raise NotImplementedError


class SyncProviderStream:
    """Execute provider stream terminal effects synchronously."""

    def __aiter__(self) -> AsyncIterator[ProviderItem]:
        """Return an asynchronous item iterator."""
        raise NotImplementedError

    def terminal(self) -> ProviderResult:
        """Fail because terminal metadata is not awaitable."""
        raise NotImplementedError

    def aclose(self) -> None:
        """Fail because close is not awaitable."""


class SyncObserver:
    """Publish observations synchronously."""

    def publish(self, observation: ConversationObservation) -> None:
        """Fail because publication is not awaitable."""
        del observation


INVALID_COORDINATOR: ConversationCoordinator = SyncCoordinator()
INVALID_STORE: ConversationStore = SyncStore()
INVALID_PROVIDER: ConversationProvider = SyncProvider()
INVALID_PROVIDER_STREAM: ConversationProviderStream = SyncProviderStream()
INVALID_OBSERVER: ConversationObserver = SyncObserver()
