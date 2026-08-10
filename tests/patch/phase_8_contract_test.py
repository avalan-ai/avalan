"""Pin Phase 8 durable continuation to the internal authenticated test host."""

import pytest

from avalan.patch.durable_coordinator import (
    DurablePatchTestHost,
    DurablePatchTestHostProfile,
)
from avalan.patch.durable_store import (
    DurableStoreError,
    DurableStoreErrorCode,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)


def test_patch_phase_8_requirements() -> None:
    """Require explicit authenticated test-host activation before pending."""
    store = InMemoryDurablePatchStore(InMemoryDurablePatchBackend())
    with pytest.raises(DurableStoreError) as raised:
        DurablePatchTestHost(store, DurablePatchTestHostProfile())
    assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
    host = DurablePatchTestHost(
        store, DurablePatchTestHostProfile(enabled=True, authenticated=True)
    )
    assert isinstance(host, DurablePatchTestHost)
