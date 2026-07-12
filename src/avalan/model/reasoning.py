from ..entities import GenerationSettings, ReasoningSummaryMode

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class ReasoningSummaryRequestCapability:
    """Declare request-time reasoning-summary modes for one model adapter."""

    supported_modes: frozenset[ReasoningSummaryMode] = frozenset()

    def __post_init__(self) -> None:
        assert isinstance(self.supported_modes, frozenset)
        assert all(
            isinstance(mode, ReasoningSummaryMode)
            for mode in self.supported_modes
        )

    def supports(self, mode: ReasoningSummaryMode) -> bool:
        """Return whether the adapter supports the requested summary mode."""
        assert isinstance(mode, ReasoningSummaryMode)
        return mode in self.supported_modes


class ReasoningSummaryCapabilityError(RuntimeError):
    """Report a valid reasoning-summary request rejected before dispatch."""

    provider: str
    requested_mode: ReasoningSummaryMode

    def __init__(
        self,
        *,
        provider: str,
        requested_mode: ReasoningSummaryMode,
    ) -> None:
        assert isinstance(provider, str) and provider.strip()
        assert isinstance(requested_mode, ReasoningSummaryMode)
        self.provider = provider
        self.requested_mode = requested_mode
        super().__init__(
            f"Provider {provider!r} does not support reasoning summary mode "
            f"{requested_mode.value!r}"
        )


def validate_reasoning_summary_request(
    model: object,
    settings: GenerationSettings,
    *,
    provider: str | None = None,
) -> None:
    """Reject an unsupported reasoning-summary request before dispatch."""
    assert isinstance(settings, GenerationSettings)
    requested_mode = settings.reasoning.summary
    if requested_mode is None:
        return

    capability = getattr(model, "reasoning_summary_request_capability", None)
    assert isinstance(capability, ReasoningSummaryRequestCapability), (
        "text generation model must declare a reasoning summary request "
        "capability"
    )
    if capability.supports(requested_mode):
        return

    resolved_provider = provider or getattr(
        model,
        "reasoning_summary_provider",
        None,
    )
    assert isinstance(resolved_provider, str) and resolved_provider.strip()
    raise ReasoningSummaryCapabilityError(
        provider=resolved_provider,
        requested_mode=requested_mode,
    )
