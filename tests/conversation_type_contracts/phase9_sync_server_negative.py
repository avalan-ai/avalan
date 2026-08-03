"""Reject use of async served Responses operations as synchronous results."""

from fastapi import Request

from avalan.conversation import AuthorityScope
from avalan.server.responses_lifecycle import (
    ServedResponsesService,
    StoredResponsesResource,
)


def reject_sync_authentication(
    service: ServedResponsesService,
    request: Request,
) -> AuthorityScope:
    """Reject authentication whose coroutine is not awaited."""
    return service.authenticate(request)


def reject_sync_retrieval(
    service: ServedResponsesService,
    response_id: str,
    authority: AuthorityScope,
) -> StoredResponsesResource:
    """Reject retrieval whose coroutine is not awaited."""
    return service.retrieve(response_id, authority)
