"""Reject interchangeable controller, lease, and transport identities."""

from avalan.interaction import (
    ActiveControlLeaseNonce,
    ControllerId,
    ResolutionIdempotencyKey,
)

controller_id = ControllerId("controller")
lease_nonce = ActiveControlLeaseNonce("lease")
idempotency_key = ResolutionIdempotencyKey("key")

expects_controller: ControllerId = lease_nonce
expects_lease: ActiveControlLeaseNonce = idempotency_key
expects_key: ResolutionIdempotencyKey = controller_id
