from ...types import assert_non_empty_string as _assert_non_empty_string
from ..observability import (
    ObservabilitySink,
    ObservabilitySinkHealth,
    TaskObservedEvent,
)
from ..usage import UsageSource, UsageTotals, freeze_usage_metadata

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(slots=True, kw_only=True)
class NoopObservabilitySink(ObservabilitySink):
    name: str = "noop"
    _event_count: int = field(default=0, init=False)
    _usage_count: int = field(default=0, init=False)

    async def record_event(self, event: TaskObservedEvent) -> None:
        assert event is not None
        self._event_count += 1

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        _assert_non_empty_string(run_id, "run_id")
        if attempt_id is not None:
            _assert_non_empty_string(attempt_id, "attempt_id")
        assert isinstance(source, UsageSource)
        assert isinstance(totals, UsageTotals)
        freeze_usage_metadata(metadata)
        self._usage_count += 1

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name=self.name,
            event_count=self._event_count,
            usage_count=self._usage_count,
        )
