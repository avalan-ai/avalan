"""Prove caller-held served continuation operations remain strictly typed."""

from typing import assert_type

from fastapi import Request

from avalan.conversation import (
    AuthorityScope,
    ContinuationEnvelopeAdvance,
    ParentAdvanceMode,
    ReasoningContext,
)
from avalan.server.stateless_responses import (
    PreparedStatelessResponse,
    StatelessCompactCommit,
    StatelessResponseCommit,
    StatelessResponsesService,
)


async def prove_phase10_caller_held_responses(
    service: StatelessResponsesService,
    request: Request,
) -> tuple[StatelessResponseCommit, StatelessCompactCommit]:
    """Return exact authenticated, terminal, and compact result types."""
    authority = assert_type(
        await service.authenticate(request),
        AuthorityScope,
    )
    prepared = assert_type(
        await service.prepare_turn(
            authority=authority,
            input_text="typed continuation",
            request_fingerprint="f" * 64,
            reasoning_context=ReasoningContext.AUTO,
            streaming=False,
            idempotency_key=None,
            continuation_value=None,
            advance=ContinuationEnvelopeAdvance(
                mode=ParentAdvanceMode.ORDINARY_CHILD
            ),
        ),
        PreparedStatelessResponse,
    )
    committed = assert_type(
        await service.finalize(
            prepared,
            request_bytes=1,
            response_bytes=1,
            input_items=1,
            output_items=1,
        ),
        StatelessResponseCommit,
    )
    compacted = assert_type(
        await service.compact(
            authority=authority,
            model="configured-model",
            instructions=None,
            canonical_input=(),
            continuation_value=None,
            lane_id=None,
        ),
        StatelessCompactCommit,
    )
    return committed, compacted
