"""Close defensive Phase 8 headless and SDK coverage gaps."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from headless_policy_test import _suspension

import avalan.sdk as sdk_module
from avalan.interaction.continuation import (
    PortableConversationCheckpointReference,
)
from avalan.interaction.durable import DurableInteractionSuspension
from avalan.interaction.entities import (
    InputRequest,
    RequestState,
    StateRevision,
)
from avalan.interaction.error import InputValidationError
from avalan.interaction.headless import DurableHandoffInputPolicy


@pytest.fixture
def anyio_backend() -> str:
    """Run Phase 8 async coverage checks on asyncio."""
    return "asyncio"


def _conversation_suspension() -> DurableInteractionSuspension:
    """Attach one exact portable conversation reference to a suspension."""
    base = _suspension()
    return replace(
        base,
        continuation=replace(
            base.continuation,
            version=2,
            conversation_checkpoint_reference=(
                PortableConversationCheckpointReference(
                    checkpoint_id="conversation-checkpoint",
                    execution_segment_id="conversation-segment",
                )
            ),
        ),
    )


class _AtomicHandoff:
    """Persist a suspension through both ordinary and atomic host paths."""

    def __init__(self) -> None:
        self.atomic_calls: list[tuple[object, object]] = []

    async def __call__(
        self,
        suspension: DurableInteractionSuspension,
    ) -> InputRequest:
        """Return the exact pending request for ordinary persistence."""
        return replace(
            suspension.command.request,
            state=RequestState.PENDING,
            state_revision=StateRevision(1),
        )

    async def persist_atomic(
        self,
        suspension: DurableInteractionSuspension,
        participant: object,
    ) -> InputRequest:
        """Record and persist one exact atomic conversation participant."""
        self.atomic_calls.append((suspension, participant))
        return await self(suspension)


@pytest.mark.anyio
async def test_durable_handoff_enforces_and_uses_atomic_participants() -> None:
    """Reject mismatched participants and invoke a valid atomic host."""
    handoff = _AtomicHandoff()
    policy = DurableHandoffInputPolicy(handoff=handoff)
    ordinary = _suspension()
    with pytest.raises(InputValidationError):
        await policy.persist(
            ordinary,
            conversation_unit=cast(Any, SimpleNamespace()),
        )

    suspension = _conversation_suspension()
    reference = suspension.continuation.conversation_checkpoint_reference
    assert reference is not None
    valid = SimpleNamespace(
        checkpoint_id=reference.checkpoint_id,
        execution_segment_id=reference.execution_segment_id,
        continuation_id=str(suspension.continuation.continuation_id),
        continuation_state_revision=int(
            suspension.continuation.state_revision
        ),
    )
    with pytest.raises(InputValidationError):
        await policy.persist(
            suspension,
            conversation_unit=cast(
                Any,
                SimpleNamespace(
                    checkpoint_id="wrong-checkpoint",
                    execution_segment_id=reference.execution_segment_id,
                    continuation_id=valid.continuation_id,
                    continuation_state_revision=(
                        valid.continuation_state_revision
                    ),
                ),
            ),
        )

    expected = await policy.persist(
        suspension,
        conversation_unit=cast(Any, valid),
    )
    assert expected.state is RequestState.PENDING
    assert handoff.atomic_calls == [(suspension, valid)]


@pytest.mark.anyio
async def test_sdk_rolls_back_async_conversation_participant() -> None:
    """Await a supported rollback callback after a failed public handoff."""
    rolled_back = False

    class Participant:
        async def rollback(self) -> None:
            nonlocal rolled_back
            rolled_back = True

    await sdk_module._rollback_conversation_handoff(Participant())
    assert rolled_back
