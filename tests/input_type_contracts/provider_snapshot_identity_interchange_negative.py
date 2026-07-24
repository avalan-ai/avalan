"""Reject provider snapshot identity, revision, and token interchange."""

from avalan.interaction import (
    CapabilityRevision,
    ContinuationFencingToken,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    ContinuationStoreRevision,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    StateRevision,
)

provider_family: ProviderFamilyName = ModelId("model")
model_id: ModelId = ProviderFamilyName("provider")
provider_revision: ProviderConfigRevision = ModelConfigRevision("model-r1")
model_revision: ModelConfigRevision = CapabilityRevision("capability-r1")
capability_revision: CapabilityRevision = ProviderConfigRevision("provider-r1")
state_revision: StateRevision = ContinuationStoreRevision(1)
store_revision: ContinuationStoreRevision = ContinuationFencingToken(1)
fencing_token: ContinuationFencingToken = StateRevision(1)

binding = ContinuationRevisionBinding(
    provider_family=ProviderFamilyName("openai"),
    model_id=ModelId("gpt-5"),
    provider_config_revision=ProviderConfigRevision("provider-r1"),
    model_config_revision=ModelConfigRevision("model-r1"),
    capability_revision=CapabilityRevision("capability-r1"),
)
snapshot = ContinuationSnapshot(
    snapshot_kind="openai.responses.reasoning",
    revision_binding=binding,
    model_call_id=ContinuationId("continuation"),
    provider_idempotency_key=ModelCallId("model-call"),
    payload={},
)
provider_key: ProviderIdempotencyKey = ContinuationId("continuation")
