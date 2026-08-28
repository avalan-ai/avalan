"""Reject mutable caller replacement of authenticated remote authority."""

from avalan.patch.domain import Capability
from avalan.patch.policy import PatchTenantId
from avalan.server.patch import RemotePatchAuthority


def reject_remote_authority_mutation(authority: RemotePatchAuthority) -> None:
    """Attempt to replace the authenticated tenant after server binding."""
    authority.tenant = PatchTenantId("caller-tenant")
    authority.capabilities = frozenset((Capability.UPDATE,))
