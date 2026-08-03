"""Reject synchronous use of caller-held served continuation operations."""

from fastapi import Request

from avalan.conversation import AuthorityScope
from avalan.server.stateless_responses import (
    PreparedStatelessResponse,
    StatelessCompactCommit,
    StatelessResponseOutcome,
    StatelessResponsesService,
)


def reject_sync_authentication(
    service: StatelessResponsesService,
    request: Request,
) -> AuthorityScope:
    """Reject authentication whose coroutine is not awaited."""
    return service.authenticate(request)


def reject_sync_compaction(
    service: StatelessResponsesService,
    authority: AuthorityScope,
) -> StatelessCompactCommit:
    """Reject compaction whose coroutine is not awaited."""
    return service.compact(
        authority=authority,
        model="configured-model",
        instructions=None,
        canonical_input=(),
        continuation_value=None,
        lane_id=None,
    )


def reject_sync_abort(
    service: StatelessResponsesService,
    prepared: PreparedStatelessResponse,
) -> None:
    """Reject cleanup whose coroutine is not awaited."""
    return service.abort(
        prepared,
        outcome=StatelessResponseOutcome.CANCELLED,
        request_bytes=1,
        input_items=1,
    )
