"""Prove the Phase 9 served Responses lifecycle is strictly async and typed."""

from typing import assert_type

from fastapi import Request

from avalan.conversation import AuthorityScope, LocalDeletionState
from avalan.server.responses_lifecycle import (
    ServedResponsesService,
    StoredResponsesResource,
)


async def prove_phase9_served_responses(
    service: ServedResponsesService,
    request: Request,
    response_id: str,
) -> tuple[StoredResponsesResource, LocalDeletionState]:
    """Return exact authenticated retrieval and deletion result types."""
    authority = assert_type(
        await service.authenticate(request),
        AuthorityScope,
    )
    resource = assert_type(
        await service.retrieve(response_id, authority),
        StoredResponsesResource,
    )
    deletion = assert_type(
        await service.tombstone(response_id, authority),
        LocalDeletionState,
    )
    return resource, deletion
