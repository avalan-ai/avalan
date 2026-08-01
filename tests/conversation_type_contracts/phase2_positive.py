"""Prove the Phase 2 coordinator and storage effects remain async-only."""

from avalan.conversation import (
    ConversationAuthorityResolver,
    ConversationClock,
    ConversationCoordinator,
    ConversationObserver,
    ConversationProvider,
    ConversationPublisher,
    ConversationRetryWaiter,
    ConversationStore,
    ConversationUnitOfWork,
    DeterministicFakeAuthorityResolver,
    DeterministicFakeClock,
    DeterministicFakeObserver,
    DeterministicFakePublisher,
    DeterministicFakeRetryWaiter,
    InMemoryConversationStore,
    InMemoryConversationUnitOfWork,
    RunScopedConversationCoordinator,
)


def prove_phase2_protocols(
    coordinator: RunScopedConversationCoordinator,
    store: InMemoryConversationStore,
    unit_of_work: InMemoryConversationUnitOfWork,
    provider: ConversationProvider,
    observer: DeterministicFakeObserver,
    authority: DeterministicFakeAuthorityResolver,
    clock: DeterministicFakeClock,
    waiter: DeterministicFakeRetryWaiter,
    publisher: DeterministicFakePublisher,
) -> tuple[
    ConversationCoordinator,
    ConversationStore,
    ConversationUnitOfWork,
    ConversationProvider,
    ConversationObserver,
    ConversationAuthorityResolver,
    ConversationClock,
    ConversationRetryWaiter,
    ConversationPublisher,
]:
    """Return concrete Phase 2 implementations through public protocols."""
    coordinator_protocol: ConversationCoordinator = coordinator
    store_protocol: ConversationStore = store
    unit_of_work_protocol: ConversationUnitOfWork = unit_of_work
    provider_protocol: ConversationProvider = provider
    observer_protocol: ConversationObserver = observer
    authority_protocol: ConversationAuthorityResolver = authority
    clock_protocol: ConversationClock = clock
    waiter_protocol: ConversationRetryWaiter = waiter
    publisher_protocol: ConversationPublisher = publisher
    return (
        coordinator_protocol,
        store_protocol,
        unit_of_work_protocol,
        provider_protocol,
        observer_protocol,
        authority_protocol,
        clock_protocol,
        waiter_protocol,
        publisher_protocol,
    )
