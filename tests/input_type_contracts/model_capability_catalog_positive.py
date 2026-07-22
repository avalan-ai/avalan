"""Lock model capability catalog composition and context identity types."""

from collections.abc import Mapping
from typing import assert_type

from avalan.agent import Specification
from avalan.model import (
    DomainCapabilitySeed,
    ModelCallContext,
    ModelCapabilityCatalog,
    ModelCapabilityDescriptor,
    ProviderCapabilityProjection,
    ProviderCapabilitySupport,
)
from avalan.types import JsonValue

parameter_schema: Mapping[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
}
descriptor = ModelCapabilityDescriptor(
    canonical_name="domain.inspect",
    description="Inspect one domain value.",
    parameter_schema=parameter_schema,
)
seed = DomainCapabilitySeed(descriptors=(descriptor,))
support = ProviderCapabilitySupport(
    structured_invocation=True,
    stable_call_ids=True,
    correlated_results=True,
    attached_resolution=True,
)
catalog = ModelCapabilityCatalog.create(seed, support=support)
projection = catalog.project("openai")

assert_type(descriptor, ModelCapabilityDescriptor)
assert_type(seed, DomainCapabilitySeed)
assert_type(support, ProviderCapabilitySupport)
assert_type(catalog, ModelCapabilityCatalog)
assert_type(catalog.descriptors, tuple[ModelCapabilityDescriptor, ...])
assert_type(projection, ProviderCapabilityProjection)

context = ModelCallContext(
    specification=Specification(),
    input="inspect",
    capability=catalog,
)
assert_type(context, ModelCallContext)
assert context.capability is catalog
assert_type(context.capability, ModelCapabilityCatalog)
