"""Define async-only coordinator, store, and provider protocols."""

from .binding import ProviderLaneBinding
from .contract import AuthorityScope, CheckpointId, UpstreamResponseId
from .errors import ConversationValidationError
from .items import ProviderItem, ProviderItemLedger
from .observability import (
    ConversationObservation,
    ConversationRequestSemantics,
)
from .settings import (
    ConversationResult,
    ConversationSettings,
    ConversationStreamTerminal,
    EffectiveReasoningMetadata,
)
from .state import (
    CheckpointCandidate,
    ConversationCheckpoint,
)
from .value import validate_identifier

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, TypeAlias, final


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StatelessProviderPlan:
    """Dispatch canonical stateless context without an upstream ID."""

    binding: ProviderLaneBinding
    ledger: ProviderItemLedger
    reasoning: EffectiveReasoningMetadata

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not ProviderLaneBinding
            or type(self.ledger) is not ProviderItemLedger
            or type(self.reasoning) is not EffectiveReasoningMetadata
            or self.binding.lane_id != self.ledger.lane_id
        ):
            raise ConversationValidationError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class StoredProviderPlan:
    """Dispatch new input using one private upstream response ID."""

    binding: ProviderLaneBinding
    upstream_response_id: UpstreamResponseId
    reasoning: EffectiveReasoningMetadata

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not ProviderLaneBinding
            or type(self.reasoning) is not EffectiveReasoningMetadata
        ):
            raise ConversationValidationError()
        validate_identifier(self.upstream_response_id, "upstream_response_id")


ProviderPlan: TypeAlias = StatelessProviderPlan | StoredProviderPlan


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderResult:
    """Return complete validated items and effective reasoning metadata."""

    items: tuple[ProviderItem, ...]
    reasoning: EffectiveReasoningMetadata
    upstream_response_id: UpstreamResponseId | None = None

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or any(
            type(item) is not ProviderItem for item in self.items
        ):
            raise ConversationValidationError()
        if type(self.reasoning) is not EffectiveReasoningMetadata:
            raise ConversationValidationError()
        if self.upstream_response_id is not None:
            validate_identifier(
                self.upstream_response_id,
                "upstream_response_id",
            )


class ConversationProviderStream(Protocol):
    """Yield complete provider items and close asynchronously."""

    def __aiter__(self) -> AsyncIterator[ProviderItem]:
        """Return the asynchronous item iterator."""
        ...

    async def terminal(self) -> ProviderResult:
        """Return validated terminal provider metadata."""
        ...

    async def aclose(self) -> None:
        """Close and await the owned provider stream."""
        ...


class ConversationProvider(Protocol):
    """Dispatch typed provider plans using asynchronous effects only."""

    async def dispatch(self, plan: ProviderPlan) -> ProviderResult:
        """Dispatch one non-streaming provider request."""
        ...

    async def stream(self, plan: ProviderPlan) -> ConversationProviderStream:
        """Open one owned asynchronous provider stream."""
        ...


class ConversationStore(Protocol):
    """Persist and resolve immutable checkpoints asynchronously."""

    async def load(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint:
        """Load one authorized immutable checkpoint."""
        ...

    async def commit(
        self,
        candidate: CheckpointCandidate,
    ) -> ConversationCheckpoint:
        """Commit one validated immutable checkpoint candidate."""
        ...

    async def close(self) -> None:
        """Close and await every owned storage resource."""
        ...


class ConversationCoordinator(Protocol):
    """Coordinate one typed conversation operation asynchronously."""

    async def execute(
        self,
        request: ConversationRequestSemantics,
        settings: ConversationSettings,
    ) -> ConversationResult:
        """Execute and commit one non-streaming conversation operation."""
        ...

    async def stream(
        self,
        request: ConversationRequestSemantics,
        settings: ConversationSettings,
    ) -> ConversationStreamTerminal:
        """Execute one streaming operation and return its terminal result."""
        ...


class ConversationObserver(Protocol):
    """Publish content-safe lifecycle observations asynchronously."""

    async def publish(self, observation: ConversationObservation) -> None:
        """Publish one content-safe post-transition observation."""
        ...
