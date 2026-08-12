"""Provide an explicitly configured deterministic model adapter for tests."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from json import dumps
from logging import Logger
from types import SimpleNamespace, TracebackType
from typing import Literal

from avalan.entities import EngineUri, Modality, TransformerEngineSettings
from avalan.model.call import ModelCall
from avalan.model.capability import ProviderCapabilitySupport
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)


@dataclass(frozen=True, slots=True)
class DeterministicToolPlan:
    """Describe one canonical provider-authored deterministic tool call."""

    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        """Require one named call with an immutable mapping shape."""
        if not self.name or not isinstance(self.arguments, Mapping):
            raise ValueError("deterministic tool plan is invalid")


class DeterministicTextEngine:
    """Expose the engine surface required by production agent orchestration."""

    model_id = "deterministic-model"
    model_type = "fake"
    provider_capability_support = ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
    )

    def __init__(self) -> None:
        """Create deterministic tokenizer metadata for EngineAgent."""
        self.tokenizer = SimpleNamespace(eos_token="<deterministic-eos>")

    def __enter__(self) -> "DeterministicTextEngine":
        """Enter this non-resource-bearing deterministic engine."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Leave this deterministic engine without suppression."""
        del exc_type, exc_value, traceback
        return False


class DeterministicModelManager(ModelManager):
    """Run a configured provider script through the normal manager API."""

    def __init__(
        self,
        plans: list[DeterministicToolPlan | str],
        logger: Logger,
    ) -> None:
        """Bind an explicit finite script and record every production call."""
        if not plans:
            raise ValueError("deterministic model script is empty")
        self.plans = list(plans)
        self.calls: list[ModelCall] = []
        self.engine = DeterministicTextEngine()
        self._logger = logger

    def __enter__(self) -> "DeterministicModelManager":
        """Enter this configured deterministic provider adapter."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Leave this deterministic adapter without suppression."""
        del exc_type, exc_value, traceback
        return False

    def get_engine_settings(
        self,
        engine_uri: EngineUri,
        settings: Mapping[str, object] | None = None,
        modality: Modality | None = None,
    ) -> TransformerEngineSettings:
        """Return closed settings for the explicitly configured fake engine."""
        del engine_uri, settings, modality
        return TransformerEngineSettings(
            auto_load_model=False,
            auto_load_tokenizer=False,
        )

    def load_engine(
        self,
        engine_uri: EngineUri,
        engine_settings: TransformerEngineSettings,
        modality: Modality = Modality.TEXT_GENERATION,
    ) -> DeterministicTextEngine:
        """Return the deterministic engine through normal loading."""
        del engine_uri, engine_settings, modality
        return self.engine

    async def __call__(self, model_task: ModelCall) -> object:
        """Return the next canonical provider response in exact call order."""
        index = len(self.calls)
        self.calls.append(model_task)
        if index >= len(self.plans):
            raise RuntimeError("deterministic model script exhausted")
        plan = self.plans[index]
        if isinstance(plan, str):
            return TextGenerationResponse(
                lambda: plan,
                logger=self._logger,
                use_async_generator=False,
            )
        return _tool_response(
            "deterministic-call-" + str(index + 1),
            plan.name,
            dict(plan.arguments),
            self._logger,
        )


def _tool_response(
    call_id: str,
    canonical_name: str,
    arguments: Mapping[str, object],
    logger: Logger,
) -> TextGenerationResponse:
    """Return one complete canonical deterministic tool-call stream."""

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        """Yield one stable provider-authored tool call and terminal frame."""
        correlation = StreamItemCorrelation(tool_call_id=call_id)
        yield CanonicalStreamItem(
            stream_session_id="deterministic-stream",
            run_id="deterministic-run",
            turn_id=call_id,
            provider_family="deterministic",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="deterministic-stream",
            run_id="deterministic-run",
            turn_id=call_id,
            provider_family="deterministic",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            text_delta=dumps(arguments, separators=(",", ":")),
            correlation=correlation,
        )
        yield CanonicalStreamItem(
            stream_session_id="deterministic-stream",
            run_id="deterministic-run",
            turn_id=call_id,
            provider_family="deterministic",
            sequence=2,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            data={"name": canonical_name, "arguments": arguments},
            correlation=correlation,
        )
        yield CanonicalStreamItem(
            stream_session_id="deterministic-stream",
            run_id="deterministic-run",
            turn_id=call_id,
            provider_family="deterministic",
            sequence=3,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=correlation,
        )
        yield CanonicalStreamItem(
            stream_session_id="deterministic-stream",
            run_id="deterministic-run",
            turn_id=call_id,
            provider_family="deterministic",
            sequence=4,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield CanonicalStreamItem(
            stream_session_id="deterministic-stream",
            run_id="deterministic-run",
            turn_id=call_id,
            provider_family="deterministic",
            sequence=5,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )

    return TextGenerationResponse(
        lambda: items(),
        logger=logger,
        use_async_generator=True,
    )
