"""Assert authenticated remote patch test-server types stay closed."""

from typing import assert_type

from avalan.patch.domain import Capability
from avalan.patch.toolset import RemotePatchRuntimeWitness
from avalan.server.patch import (
    RemotePatchAuthority,
    RemotePatchTestServerProfile,
)


def assert_remote_test_server_types(
    authority: RemotePatchAuthority,
    profile: RemotePatchTestServerProfile,
    witness: RemotePatchRuntimeWitness,
) -> None:
    """Require immutable authenticated authority and explicit activation."""
    assert_type(authority.correlation, str)
    assert_type(authority.execution_scope, str)
    assert_type(authority.capabilities, frozenset[Capability])
    assert_type(profile.active, bool)
    assert_type(witness.capabilities, frozenset[Capability])
