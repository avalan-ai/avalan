"""Reject a provider-stored parent for standalone compaction."""

from avalan.conversation import StandaloneCompactRequest, StoredParent


def reject_stored_compact_parent(
    parent: StoredParent,
) -> StandaloneCompactRequest:
    """Reject implicit conversion from stored to stateless mode."""
    return StandaloneCompactRequest(parent=parent)
