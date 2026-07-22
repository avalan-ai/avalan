"""Reject interchangeable provider, model, revision, and binding identities."""

from avalan.interaction import (
    CapabilityRevision,
    ContinuationRevisionBinding,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
)

provider_family = ProviderFamilyName("provider")
model_id = ModelId("model")
provider_revision = ProviderConfigRevision("provider-v1")
model_revision = ModelConfigRevision("model-v1")
capability_revision = CapabilityRevision("capability-v1")

wrong_provider_family: ProviderFamilyName = model_id
wrong_model_id: ModelId = provider_family
wrong_provider_revision: ProviderConfigRevision = model_revision
wrong_model_revision: ModelConfigRevision = capability_revision
wrong_capability_revision: CapabilityRevision = provider_revision
wrong_binding: ContinuationRevisionBinding = capability_revision

print(
    wrong_provider_family,
    wrong_model_id,
    wrong_provider_revision,
    wrong_model_revision,
    wrong_capability_revision,
    wrong_binding,
)
