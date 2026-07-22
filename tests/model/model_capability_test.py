from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from gc import collect
from math import inf
from types import MappingProxyType
from typing import Any, cast
from unittest.mock import MagicMock
from weakref import ref

import pytest

import avalan.model.capability as capability_module
from avalan.entities import (
    MessageRole,
    ToolCall,
    ToolCallRecoveryFormat,
    ToolFormat,
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
)
from avalan.interaction import (
    RESERVED_INPUT_CAPABILITY_NAME,
    AnswerProvenance,
    CapabilityRevision,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    InputDeclinedResult,
    InputRequestId,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    ProviderIdempotencyKey,
    RequirementMode,
    decode_continuation_snapshot,
    encode_continuation_snapshot,
)
from avalan.model import (
    CapabilityBatchAccepted,
    CapabilityBatchRejected,
    CapabilityBatchRejectionCode,
    ContinuationSnapshotCodecRegistry,
    CorrelatedCapabilityResult,
    DomainCapabilitySeed,
    ModelCapabilityCatalog,
    ModelCapabilityDescriptor,
    ModelCapabilityKind,
    ModelCapabilityValidationError,
    ProviderCapabilityCall,
    ProviderCapabilitySupport,
    RegisteredContinuationSnapshotCodec,
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.parser import ToolCallParser


def lookup(query: str) -> str:
    """Look up one value.

    Args:
        query: Value to look up.

    Returns:
        The provided value.
    """
    return query


def request_user_input(value: str) -> str:
    """Collide with the reserved model capability.

    Args:
        value: Arbitrary value.

    Returns:
        The provided value.
    """
    return value


def _manager(
    *,
    mapped_name: str = "lookup",
    enable_tools: list[str] | None = None,
    tools: list[Any] | None = None,
) -> ToolManager:
    return ToolManager.create_instance(
        available_toolsets=[ToolSet(namespace="pkg", tools=tools or [lookup])],
        enable_tools=enable_tools or ["pkg.lookup"],
        settings=ToolManagerSettings(
            tool_format=ToolFormat.JSON,
            tool_name_policy=ToolNamePolicySettings(
                mode=ToolNamePolicyMode.SANITIZED,
                map={"pkg.lookup": mapped_name},
            ),
            recovery_formats=[ToolCallRecoveryFormat.TOOL_CALL_BLOCK],
            maximum_parser_input_size=8_192,
            maximum_parser_payload_depth=10,
            maximum_parser_payload_size=4_096,
        ),
    )


def _attached_support() -> ProviderCapabilitySupport:
    return ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
        attached_resolution=True,
    )


def _revision_binding(
    *,
    provider_family: str = "openai",
    model_id: str = "test-model",
    provider_config_revision: str = "provider-v1",
    model_config_revision: str = "model-v1",
    capability_revision: str = "capability-v1",
) -> ContinuationRevisionBinding:
    return ContinuationRevisionBinding(
        provider_family=ProviderFamilyName(provider_family),
        model_id=ModelId(model_id),
        provider_config_revision=ProviderConfigRevision(
            provider_config_revision
        ),
        model_config_revision=ModelConfigRevision(model_config_revision),
        capability_revision=CapabilityRevision(capability_revision),
    )


def _restore_snapshot(
    value: str, binding: ContinuationRevisionBinding
) -> ContinuationSnapshot:
    return decode_continuation_snapshot(value, expected_binding=binding)


def _registered_codec() -> tuple[
    ContinuationSnapshotCodecRegistry,
    RegisteredContinuationSnapshotCodec,
]:
    registry = ContinuationSnapshotCodecRegistry("test-registry")
    registry.register(
        codec_id="test-codec-v1",
        revision_binding=_revision_binding(),
        snapshot_kind="test.provider-response",
        export_snapshot=encode_continuation_snapshot,
        restore_snapshot=_restore_snapshot,
    )
    return registry, registry.reference("test-codec-v1")


def _durable_support(
    registry: ContinuationSnapshotCodecRegistry | None = None,
    codec: RegisteredContinuationSnapshotCodec | None = None,
) -> ProviderCapabilitySupport:
    if registry is None or codec is None:
        assert registry is None and codec is None
        registry, codec = _registered_codec()
    return ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
        durable_store=True,
        registered_resumer=True,
        continuation_snapshot_codec_registry=registry,
        continuation_snapshot_codec=codec,
    )


def _input_arguments() -> dict[str, object]:
    return {
        "mode": "required",
        "reason": "Choose the deployment region.",
        "questions": [
            {
                "question_id": "region",
                "kind": "single_selection",
                "header": "Region",
                "prompt": "Which region should be used?",
                "required": True,
                "choices": [
                    {
                        "value": "us-east",
                        "label": "US East",
                        "description": "Use the eastern region.",
                    },
                    {
                        "value": "eu-west",
                        "label": "EU West",
                        "description": "Use the western Europe region.",
                    },
                ],
                "allow_other": False,
                "recommended_choice": "us-east",
            }
        ],
    }


def _contains_callable(value: object) -> bool:
    if callable(value):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_callable(key) or _contains_callable(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_callable(item) for item in value)
    return False


def _nested_mapping(depth: int, *, proxy: bool) -> object:
    value: object = "leaf"
    for _ in range(depth):
        layer = {"nested": value}
        value = MappingProxyType(layer) if proxy else layer
    return value


def _replace_export_schema(
    manager: ToolManager,
    attribute: str,
    value: object,
) -> None:
    descriptor = manager.describe_tool("pkg.lookup")
    assert descriptor is not None
    schema = getattr(descriptor, attribute)
    assert isinstance(schema, dict)
    schema.clear()
    schema["nested"] = value


def test_tool_manager_exports_isolated_callable_free_seed() -> None:
    manager = _manager()

    exported = manager.export_model_capability_seed()

    assert exported["version"] == 1
    assert not _contains_callable(exported)
    assert exported["name_policy"] == {
        "mode": "sanitized",
        "prefix": "avl_",
        "replacement": "_",
        "collapse_replacement": True,
        "map": {"pkg.lookup": "lookup"},
        "provider_family": None,
    }
    assert exported["parser"] == {
        "tool_format": "json",
        "eos_token": None,
        "recovery_formats": ["tool_call_block"],
        "maximum_input_size": 8_192,
        "maximum_payload_depth": 10,
        "maximum_payload_size": 4_096,
    }
    descriptors = cast(list[dict[str, Any]], exported["descriptors"])
    assert descriptors == [
        {
            "canonical_name": "pkg.lookup",
            "description": "Look up one value.",
            "aliases": [],
            "parameter_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Value to look up.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "result_schema": {
                "type": "string",
                "description": "The provided value.",
            },
        }
    ]

    descriptors[0]["description"] = "mutated"
    cast(dict[str, Any], descriptors[0]["parameter_schema"])[
        "additionalProperties"
    ] = True
    fresh = manager.export_model_capability_seed()
    fresh_descriptor = cast(list[dict[str, Any]], fresh["descriptors"])[0]
    assert fresh_descriptor["description"] == "Look up one value."
    assert (
        cast(dict[str, Any], fresh_descriptor["parameter_schema"])[
            "additionalProperties"
        ]
        is False
    )


@pytest.mark.parametrize("attribute", ("parameter_schema", "return_schema"))
@pytest.mark.parametrize("proxy", (False, True), ids=("dict", "mapping"))
def test_tool_manager_seed_export_accepts_schema_depth_boundary(
    attribute: str,
    proxy: bool,
) -> None:
    manager = _manager()
    _replace_export_schema(
        manager,
        attribute,
        _nested_mapping(63, proxy=proxy),
    )

    exported = manager.export_model_capability_seed()

    descriptor = cast(list[dict[str, object]], exported["descriptors"])[0]
    exported_name = (
        "parameter_schema"
        if attribute == "parameter_schema"
        else "result_schema"
    )
    value = descriptor[exported_name]
    for _ in range(64):
        assert isinstance(value, dict)
        value = value["nested"]
    assert value == "leaf"


@pytest.mark.parametrize("attribute", ("parameter_schema", "return_schema"))
@pytest.mark.parametrize("proxy", (False, True), ids=("dict", "mapping"))
def test_tool_manager_seed_export_rejects_extreme_schema_depth(
    attribute: str,
    proxy: bool,
) -> None:
    manager = _manager()
    _replace_export_schema(
        manager,
        attribute,
        _nested_mapping(1_100, proxy=proxy),
    )

    with pytest.raises(ModelCapabilityValidationError) as captured:
        manager.export_model_capability_seed()

    assert captured.value.code == "capability.non_json"
    assert "too deeply nested" in str(captured.value)


def test_catalog_is_immutable_and_reconstructs_parser_configuration() -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed()
    )

    assert not catalog.is_empty
    assert catalog.structured_parser_enabled
    assert catalog.tool_format is ToolFormat.JSON
    assert catalog.recovery_formats == (
        ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
    )
    assert isinstance(catalog.domain_seed, DomainCapabilitySeed)
    assert catalog.descriptors[0].kind is ModelCapabilityKind.DOMAIN_TOOL
    assert isinstance(
        catalog.descriptors[0].parameter_schema, MappingProxyType
    )
    with pytest.raises(TypeError):
        cast(dict[str, object], catalog.descriptors[0].parameter_schema)[
            "type"
        ] = "string"

    assert not catalog.is_potential_tool_call("", " ")
    assert catalog.is_potential_tool_call("", "<")
    prose = catalog.parse_calls("Please use pkg.lookup if helpful.")
    assert prose.calls == []
    parsed = catalog.parse_calls(
        '<tool_call>{"name":"lookup",'
        '"arguments":{"query":"value"}}</tool_call>'
    )
    assert len(parsed.calls) == 1
    assert parsed.calls[0].name == "pkg.lookup"
    assert parsed.calls[0].arguments == {"query": "value"}
    assert catalog.stream_buffer_diagnostics("ordinary text") == []


def test_seed_and_catalog_parser_accessors_cover_edge_states() -> None:
    empty = ModelCapabilityCatalog.create(
        DomainCapabilitySeed(eos_token="</s>")
    )

    assert not empty.is_potential_tool_call("ordinary", " text")
    assert empty.tool_call_status("ordinary text") is (
        ToolCallParser.ToolCallBufferStatus.NONE
    )
    assert empty.get_calls("ordinary text") is None


def test_support_prerequisites_control_reserved_advertisement() -> None:
    incapable = ModelCapabilityCatalog.create()
    missing_call_ids = ModelCapabilityCatalog.create(
        support=ProviderCapabilitySupport(
            structured_invocation=True,
            correlated_results=True,
            attached_resolution=True,
        )
    )
    partial_durable = ModelCapabilityCatalog.create(
        support=ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
        )
    )
    attached = ModelCapabilityCatalog.create(support=_attached_support())
    durable_registry, durable_codec = _registered_codec()
    durable = ModelCapabilityCatalog.create(
        support=_durable_support(durable_registry, durable_codec),
        revision_binding=durable_codec.revision_binding,
    )

    assert incapable.is_empty
    assert not incapable.structured_parser_enabled
    assert (
        incapable.task_input_advertisement
        is TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert missing_call_ids.is_empty
    assert partial_durable.is_empty
    assert (
        attached.task_input_advertisement
        is TaskInputCapabilityAdvertisement.ATTACHED
    )
    assert (
        durable.task_input_advertisement
        is TaskInputCapabilityAdvertisement.DURABLE
    )
    assert attached.descriptors[0].canonical_name == (
        RESERVED_INPUT_CAPABILITY_NAME
    )


def test_durable_support_requires_registry_validated_codec_evidence() -> None:
    registry, codec = _registered_codec()
    snapshot = ContinuationSnapshot(
        snapshot_kind=codec.snapshot_kind,
        revision_binding=codec.revision_binding,
        model_call_id=ModelCallId("call-1"),
        provider_idempotency_key=ProviderIdempotencyKey("idempotency-1"),
        payload={"response_id": "response-1"},
    )

    assert codec.registry_id == "test-registry"
    assert codec.codec_id == "test-codec-v1"
    assert codec.is_registry_minted
    assert registry.is_registered(codec)
    assert codec.accepts(snapshot)
    assert not codec.accepts(
        ContinuationSnapshot(
            snapshot_kind=codec.snapshot_kind,
            revision_binding=_revision_binding(model_id="other-model"),
            model_call_id=ModelCallId("call-1"),
            provider_idempotency_key=ProviderIdempotencyKey("idempotency-1"),
            payload={"response_id": "response-1"},
        )
    )

    with pytest.raises(TypeError):
        RegisteredContinuationSnapshotCodec()
    with pytest.raises(AssertionError):
        ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
            continuation_snapshot_codec=cast(Any, True),
        )


def test_codec_registry_live_lookup_exports_and_restores_by_reference() -> (
    None
):
    registry, codec = _registered_codec()
    snapshot = ContinuationSnapshot(
        snapshot_kind=codec.snapshot_kind,
        revision_binding=codec.revision_binding,
        model_call_id=ModelCallId("call-live"),
        provider_idempotency_key=ProviderIdempotencyKey("idempotency-live"),
        payload={"response_id": "response-live"},
    )

    assert registry.reference(codec.codec_id) is codec
    assert registry.is_registered(codec)
    encoded = registry.export_snapshot(codec, snapshot)
    assert (
        registry.restore_snapshot(
            codec,
            encoded,
            codec.revision_binding,
        )
        == snapshot
    )


def test_durable_support_retains_the_live_registry_across_gc() -> None:
    registry, codec = _registered_codec()
    registry_reference = ref(registry)
    support = _durable_support(registry, codec)
    catalog = ModelCapabilityCatalog.create(
        support=support,
        revision_binding=codec.revision_binding,
    )

    del registry
    collect()

    assert registry_reference() is support.continuation_snapshot_codec_registry
    assert (
        catalog.task_input_advertisement
        is TaskInputCapabilityAdvertisement.DURABLE
    )
    assert catalog.descriptors[0].kind is ModelCapabilityKind.TASK_INPUT


def test_durable_advertisement_rejects_wrong_and_dangling_registries() -> None:
    registry, codec = _registered_codec()
    foreign_registry, foreign_codec = _registered_codec()
    snapshot = ContinuationSnapshot(
        snapshot_kind=codec.snapshot_kind,
        revision_binding=codec.revision_binding,
        model_call_id=ModelCallId("call-registry"),
        provider_idempotency_key=ProviderIdempotencyKey(
            "idempotency-registry"
        ),
        payload={"response_id": "response-registry"},
    )

    assert codec is not foreign_codec
    assert codec != foreign_codec
    assert registry.is_registered(codec)
    assert foreign_registry.is_registered(foreign_codec)
    assert not foreign_registry.is_registered(codec)
    wrong_registry_catalog = ModelCapabilityCatalog.create(
        support=_durable_support(foreign_registry, codec),
        revision_binding=codec.revision_binding,
    )
    assert (
        wrong_registry_catalog.task_input_advertisement
        is TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert wrong_registry_catalog.descriptors == ()
    with pytest.raises(ModelCapabilityValidationError):
        foreign_registry.export_snapshot(codec, snapshot)
    with pytest.raises(ModelCapabilityValidationError):
        foreign_registry.restore_snapshot(
            codec,
            encode_continuation_snapshot(snapshot),
            codec.revision_binding,
        )

    live_catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed(),
        support=_durable_support(registry, codec),
        revision_binding=codec.revision_binding,
    )
    registry._registrations.clear()  # noqa: SLF001
    assert not registry.is_registered(codec)
    assert (
        live_catalog.task_input_advertisement
        is TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert [
        descriptor.canonical_name for descriptor in live_catalog.descriptors
    ] == ["pkg.lookup"]
    assert [
        descriptor.canonical_name
        for descriptor in live_catalog.project("openai").descriptors
    ] == ["pkg.lookup"]
    with pytest.raises(ModelCapabilityValidationError):
        registry.export_snapshot(codec, snapshot)
    with pytest.raises(ModelCapabilityValidationError):
        registry.restore_snapshot(
            codec,
            encode_continuation_snapshot(snapshot),
            codec.revision_binding,
        )


def test_durable_advertisement_is_bound_to_exact_provider_and_model() -> None:
    registry, codec = _registered_codec()
    support = _durable_support(registry, codec)
    catalog = ModelCapabilityCatalog.create(
        support=support,
        revision_binding=codec.revision_binding,
    )
    wrong_model = ModelCapabilityCatalog.create(
        support=support,
        revision_binding=_revision_binding(model_id="other-model"),
    )

    assert not catalog.project("openai").is_empty
    assert catalog.project("google").is_empty
    assert (
        wrong_model.task_input_advertisement
        is TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert wrong_model.project("openai").is_empty


@pytest.mark.parametrize(
    "revision_binding",
    (
        _revision_binding(provider_family="other-provider"),
        _revision_binding(model_id="other-model"),
        _revision_binding(provider_config_revision="provider-v2"),
        _revision_binding(model_config_revision="model-v2"),
        _revision_binding(capability_revision="capability-v2"),
    ),
    ids=(
        "provider-family",
        "model-id",
        "provider-config-revision",
        "model-config-revision",
        "capability-revision",
    ),
)
def test_durable_advertisement_rejects_each_binding_mismatch(
    revision_binding: ContinuationRevisionBinding,
) -> None:
    registry, codec = _registered_codec()
    expected = codec.revision_binding
    fields = (
        "provider_family",
        "model_id",
        "provider_config_revision",
        "model_config_revision",
        "capability_revision",
    )
    assert (
        sum(
            getattr(revision_binding, field) != getattr(expected, field)
            for field in fields
        )
        == 1
    )

    catalog = ModelCapabilityCatalog.create(
        support=_durable_support(registry, codec),
        revision_binding=revision_binding,
    )

    assert (
        catalog.task_input_advertisement
        is TaskInputCapabilityAdvertisement.INCAPABLE
    )
    assert catalog.descriptors == ()
    assert catalog.project("openai").is_empty


def test_codec_registry_rejects_unvalidated_or_non_roundtripping_codecs() -> (
    None
):
    registry = ContinuationSnapshotCodecRegistry("test-registry")
    binding = _revision_binding()

    with pytest.raises(ModelCapabilityValidationError):
        registry.reference("missing")
    with pytest.raises(ModelCapabilityValidationError):
        registry.register(
            codec_id="broken-codec",
            revision_binding=binding,
            snapshot_kind="test.provider-response",
            export_snapshot=lambda _snapshot: "{}",
            restore_snapshot=lambda _value, _binding: cast(
                ContinuationSnapshot, object()
            ),
        )


def test_provider_projection_has_strict_schemas_names_and_tool_choice() -> (
    None
):
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed(),
        support=_attached_support(),
    )

    projection = catalog.project("openai")

    assert not projection.is_empty
    assert projection.provider_name("pkg.lookup") == "lookup"
    assert projection.canonical_name("lookup") == "pkg.lookup"
    assert projection.provider_name(RESERVED_INPUT_CAPABILITY_NAME) == (
        RESERVED_INPUT_CAPABILITY_NAME
    )
    assert projection.tool_choice("pkg.lookup") == "lookup"
    assert projection.tool_choice(RESERVED_INPUT_CAPABILITY_NAME) == (
        RESERVED_INPUT_CAPABILITY_NAME
    )
    functions = [
        cast(Mapping[str, Any], schema["function"])
        for schema in projection.schemas
    ]
    assert [function["name"] for function in functions] == [
        "lookup",
        RESERVED_INPUT_CAPABILITY_NAME,
    ]
    reserved = functions[1]
    parameters = cast(Mapping[str, Any], reserved["parameters"])
    properties = cast(Mapping[str, Any], parameters["properties"])
    assert set(properties) == {
        "mode",
        "reason",
        "questions",
    }
    assert parameters["additionalProperties"] is False
    with pytest.raises(ModelCapabilityValidationError):
        projection.tool_choice("pkg.missing")
    with pytest.raises(ModelCapabilityValidationError) as unknown_provider:
        projection.canonical_name("missing")
    assert unknown_provider.value.code == "capability.unknown"
    assert (
        catalog.provider_name("pkg.lookup", provider_family="openai")
        == "lookup"
    )
    assert (
        catalog.canonical_name("lookup", provider_family="openai")
        == "pkg.lookup"
    )


def test_decode_call_validates_domain_and_reserved_arguments() -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed(),
        support=_attached_support(),
    )

    domain = catalog.decode_call(
        ProviderCapabilityCall(
            call_id="domain-call",
            provider_name="lookup",
            arguments='{"query":"value"}',
        ),
        provider_family="openai",
    )
    reserved = catalog.decode_call(
        ProviderCapabilityCall(
            call_id="input-call",
            provider_name=RESERVED_INPUT_CAPABILITY_NAME,
            arguments=_input_arguments(),
        ),
        provider_family="openai",
    )

    assert domain == ToolCall(
        id="domain-call",
        name="pkg.lookup",
        arguments={"query": "value"},
        provider_name="lookup",
        provider_name_encoded=True,
    )
    assert isinstance(reserved, TaskInputCapabilityCall)
    assert reserved.call_id == "input-call"
    assert reserved.canonical_name == RESERVED_INPUT_CAPABILITY_NAME
    assert reserved.mode is RequirementMode.REQUIRED
    assert reserved.reason == "Choose the deployment region."
    assert reserved.questions[0].question_id == "region"
    assert reserved.advertisement is TaskInputCapabilityAdvertisement.ATTACHED
    with pytest.raises(TypeError):
        cast(dict[str, object], reserved.arguments)["mode"] = "advisory"


@pytest.mark.parametrize(
    ("provider_name", "arguments"),
    [
        ("lookup", '{"query":"one","query":"two"}'),
        ("lookup", '{"query":NaN}'),
        ("lookup", {"query": object()}),
        ("lookup", {"unexpected": "value"}),
        (
            RESERVED_INPUT_CAPABILITY_NAME,
            {**_input_arguments(), "request_id": "model-chosen"},
        ),
        (
            RESERVED_INPUT_CAPABILITY_NAME,
            {**_input_arguments(), "interaction_class": "authentication"},
        ),
    ],
)
def test_decode_call_rejects_non_json_and_schema_invalid_arguments(
    provider_name: str,
    arguments: object,
) -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed(),
        support=_attached_support(),
    )

    with pytest.raises(ModelCapabilityValidationError):
        catalog.decode_call(
            ProviderCapabilityCall(
                call_id="call-1",
                provider_name=provider_name,
                arguments=cast(Any, arguments),
            ),
            provider_family="openai",
        )


def test_decode_call_rejects_deep_json_text_with_typed_depth_error() -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed()
    )
    arguments = '{"query":' + ("[" * 1_100) + '"value"' + ("]" * 1_100) + "}"

    with pytest.raises(ModelCapabilityValidationError) as captured:
        catalog.decode_call(
            ProviderCapabilityCall(
                call_id="call-1",
                provider_name="lookup",
                arguments=arguments,
            ),
            provider_family="openai",
        )

    assert captured.value.code == "capability.arguments_depth"


def test_decode_call_rejects_deep_mapping_with_typed_depth_error() -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed()
    )
    arguments: dict[str, object] = {}
    cursor = arguments
    for _ in range(1_100):
        nested: dict[str, object] = {}
        cursor["nested"] = nested
        cursor = nested

    with pytest.raises(ModelCapabilityValidationError) as captured:
        catalog.decode_call(
            ProviderCapabilityCall(
                call_id="call-1",
                provider_name="lookup",
                arguments=MappingProxyType(arguments),
            ),
            provider_family="openai",
        )

    assert captured.value.code == "capability.arguments_depth"


def test_batch_classification_is_fail_closed_before_domain_dispatch() -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed(),
        support=_attached_support(),
    )
    domain = ProviderCapabilityCall(
        call_id="domain-call",
        provider_name="lookup",
        arguments={"query": "value"},
    )
    reserved = ProviderCapabilityCall(
        call_id="input-call",
        provider_name=RESERVED_INPUT_CAPABILITY_NAME,
        arguments=_input_arguments(),
    )

    mixed = catalog.classify_batch(
        [domain, reserved], provider_family="openai"
    )
    multiple = catalog.classify_batch(
        [reserved, reserved], provider_family="openai"
    )
    missing_id = catalog.classify_batch(
        [
            ProviderCapabilityCall(
                call_id=None,
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=_input_arguments(),
            )
        ],
        provider_family="openai",
    )
    non_structured = catalog.classify_batch(
        [
            ProviderCapabilityCall(
                call_id="input-call",
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=_input_arguments(),
                structured=False,
            )
        ],
        provider_family="openai",
    )
    accepted_domain = catalog.classify_batch(
        [domain], provider_family="openai"
    )
    accepted_input = catalog.classify_batch(
        [reserved], provider_family="openai"
    )

    assert mixed == CapabilityBatchRejected(
        code=CapabilityBatchRejectionCode.MIXED_TASK_INPUT_BATCH,
        message="Task input cannot be mixed with domain calls.",
    )
    assert multiple.code is (  # type: ignore[union-attr]
        CapabilityBatchRejectionCode.MULTIPLE_TASK_INPUT_CALLS
    )
    assert missing_id.code is (  # type: ignore[union-attr]
        CapabilityBatchRejectionCode.MISSING_CALL_ID
    )
    assert non_structured.code is (  # type: ignore[union-attr]
        CapabilityBatchRejectionCode.NON_STRUCTURED_CALL
    )
    assert isinstance(accepted_domain, CapabilityBatchAccepted)
    assert [call.name for call in accepted_domain.domain_calls] == [
        "pkg.lookup"
    ]
    assert accepted_domain.task_input is None
    assert isinstance(accepted_input, CapabilityBatchAccepted)
    assert accepted_input.domain_calls == ()
    assert accepted_input.task_input is not None


def test_reserved_and_provider_name_collisions_fail() -> None:
    with pytest.raises(ValueError, match="reserved model capability name"):
        ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[request_user_input])],
            enable_tools=[RESERVED_INPUT_CAPABILITY_NAME],
            settings=ToolManagerSettings(),
        )

    with pytest.raises(ModelCapabilityValidationError) as reserved:
        ModelCapabilityCatalog.create(
            DomainCapabilitySeed(
                descriptors=(_descriptor(RESERVED_INPUT_CAPABILITY_NAME),)
            )
        )
    assert reserved.value.code == "capability.reserved_name"

    with pytest.raises(ValueError, match="reserved model capability name"):
        _manager(mapped_name=RESERVED_INPUT_CAPABILITY_NAME)


def test_project_result_preserves_exact_provider_call_correlation() -> None:
    catalog = ModelCapabilityCatalog.create(support=_attached_support())
    decoded = catalog.decode_call(
        ProviderCapabilityCall(
            call_id="provider-call/exact:01",
            provider_name=RESERVED_INPUT_CAPABILITY_NAME,
            arguments=_input_arguments(),
        ),
        provider_family="openai",
    )
    assert isinstance(decoded, TaskInputCapabilityCall)
    model_result = InputDeclinedResult(
        request_id=InputRequestId("request-01"),
        provenance=AnswerProvenance.HUMAN,
        resolved_at=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
    )

    projected = catalog.project_result(decoded, model_result)

    assert isinstance(projected, CorrelatedCapabilityResult)
    assert projected.call_id == "provider-call/exact:01"
    assert projected.canonical_name == RESERVED_INPUT_CAPABILITY_NAME
    assert projected.provider_name == RESERVED_INPUT_CAPABILITY_NAME
    assert projected.provider_payload() == {
        "kind": "declined",
        "request_id": "request-01",
        "provenance": "human",
        "resolved_at": "2026-07-21T12:00:00.000000Z",
    }
    mutable = projected.provider_payload()
    mutable["kind"] = "changed"
    assert projected.payload["kind"] == "declined"
    local = projected.local_message()
    assert local.role is MessageRole.TOOL
    assert local.name == RESERVED_INPUT_CAPABILITY_NAME
    assert (
        local.content
        == '{"call_id":"provider-call/exact:01","name":"request_user_input",'
        '"result":{"kind":"declined","provenance":"human",'
        '"request_id":"request-01",'
        '"resolved_at":"2026-07-21T12:00:00.000000Z"}}'
    )


def _codec_snapshot(
    reference: RegisteredContinuationSnapshotCodec,
) -> ContinuationSnapshot:
    return ContinuationSnapshot(
        snapshot_kind=reference.snapshot_kind,
        revision_binding=reference.revision_binding,
        model_call_id=ModelCallId("call-configurable"),
        provider_idempotency_key=ProviderIdempotencyKey(
            "idempotency-configurable"
        ),
        payload={"response_id": "response-configurable"},
    )


def _configurable_codec_registry() -> tuple[
    ContinuationSnapshotCodecRegistry,
    RegisteredContinuationSnapshotCodec,
    dict[str, str],
]:
    state = {"export": "ok", "restore": "ok"}

    def export_snapshot(snapshot: ContinuationSnapshot) -> str:
        if state["export"] == "validation_error":
            raise ModelCapabilityValidationError(
                "codec.export", "codec rejected the snapshot"
            )
        if state["export"] == "type_error":
            raise TypeError("codec export failed")
        if state["export"] == "empty":
            return ""
        return encode_continuation_snapshot(snapshot)

    def restore_snapshot(
        value: str, binding: ContinuationRevisionBinding
    ) -> ContinuationSnapshot:
        if state["restore"] == "validation_error":
            raise ModelCapabilityValidationError(
                "codec.restore", "codec rejected the payload"
            )
        if state["restore"] == "type_error":
            raise TypeError("codec restore failed")
        restored = _restore_snapshot(value, binding)
        if state["restore"] == "wrong_snapshot":
            return replace(restored, snapshot_kind="wrong.snapshot-kind")
        return restored

    registry = ContinuationSnapshotCodecRegistry("configurable-registry")
    registry.register(
        codec_id="configurable-codec",
        revision_binding=_revision_binding(),
        snapshot_kind="test.provider-response",
        export_snapshot=export_snapshot,
        restore_snapshot=restore_snapshot,
    )
    return registry, registry.reference("configurable-codec"), state


def test_registry_rejects_forged_codec_evidence_and_invalid_identifier() -> (
    None
):
    with pytest.raises(ModelCapabilityValidationError) as invalid_registry:
        ContinuationSnapshotCodecRegistry("")
    assert invalid_registry.value.code == "capability.continuation_codec"

    with pytest.raises(ModelCapabilityValidationError) as forged:
        RegisteredContinuationSnapshotCodec._mint(  # noqa: SLF001
            registry_id="forged-registry",
            codec_id="forged-codec",
            revision_binding=_revision_binding(),
            snapshot_kind="test.provider-response",
            proof=object(),
        )
    assert forged.value.code == "capability.continuation_codec"


def test_registry_registration_rejects_each_invalid_codec_contract() -> None:
    registry = ContinuationSnapshotCodecRegistry("registration-registry")
    binding = _revision_binding()

    with pytest.raises(ModelCapabilityValidationError) as binding_error:
        registry.register(
            codec_id="invalid-binding",
            revision_binding=cast(Any, object()),
            snapshot_kind="test.provider-response",
            export_snapshot=encode_continuation_snapshot,
            restore_snapshot=_restore_snapshot,
        )
    assert binding_error.value.code == "capability.continuation_codec"

    with pytest.raises(ModelCapabilityValidationError) as callable_error:
        registry.register(
            codec_id="invalid-callable",
            revision_binding=binding,
            snapshot_kind="test.provider-response",
            export_snapshot=cast(Any, None),
            restore_snapshot=_restore_snapshot,
        )
    assert callable_error.value.code == "capability.continuation_codec"

    with pytest.raises(ModelCapabilityValidationError) as content_error:
        registry.register(
            codec_id="empty-export",
            revision_binding=binding,
            snapshot_kind="test.provider-response",
            export_snapshot=lambda _snapshot: "",
            restore_snapshot=_restore_snapshot,
        )
    assert content_error.value.code == "capability.continuation_codec"

    def reject_export(_snapshot: ContinuationSnapshot) -> str:
        raise ModelCapabilityValidationError(
            "codec.registration", "registration rejected"
        )

    with pytest.raises(ModelCapabilityValidationError) as validation_error:
        registry.register(
            codec_id="validation-error",
            revision_binding=binding,
            snapshot_kind="test.provider-response",
            export_snapshot=reject_export,
            restore_snapshot=_restore_snapshot,
        )
    assert validation_error.value.code == "codec.registration"

    def fail_export(_snapshot: ContinuationSnapshot) -> str:
        raise TypeError("registration failed")

    with pytest.raises(ModelCapabilityValidationError) as wrapped_error:
        registry.register(
            codec_id="type-error",
            revision_binding=binding,
            snapshot_kind="test.provider-response",
            export_snapshot=fail_export,
            restore_snapshot=_restore_snapshot,
        )
    assert wrapped_error.value.code == "capability.continuation_codec"

    registry.register(
        codec_id="duplicate-codec",
        revision_binding=binding,
        snapshot_kind="test.provider-response",
        export_snapshot=encode_continuation_snapshot,
        restore_snapshot=_restore_snapshot,
    )
    with pytest.raises(ModelCapabilityValidationError) as duplicate_error:
        registry.register(
            codec_id="duplicate-codec",
            revision_binding=binding,
            snapshot_kind="test.provider-response",
            export_snapshot=encode_continuation_snapshot,
            restore_snapshot=_restore_snapshot,
        )
    assert duplicate_error.value.code == "capability.continuation_codec"


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    (
        ("validation_error", "codec.export"),
        ("type_error", "capability.continuation_codec"),
        ("empty", "capability.continuation_codec"),
    ),
)
def test_registry_export_is_fail_closed(mode: str, expected_code: str) -> None:
    registry, reference, state = _configurable_codec_registry()
    state["export"] = mode

    with pytest.raises(ModelCapabilityValidationError) as captured:
        registry.export_snapshot(reference, _codec_snapshot(reference))

    assert captured.value.code == expected_code


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    (
        ("validation_error", "codec.restore"),
        ("type_error", "capability.continuation_codec"),
        ("wrong_snapshot", "capability.continuation_codec"),
    ),
)
def test_registry_restore_is_fail_closed(
    mode: str, expected_code: str
) -> None:
    registry, reference, state = _configurable_codec_registry()
    snapshot = _codec_snapshot(reference)
    encoded = registry.export_snapshot(reference, snapshot)
    state["restore"] = mode

    with pytest.raises(ModelCapabilityValidationError) as captured:
        registry.restore_snapshot(
            reference,
            encoded,
            reference.revision_binding,
        )

    assert captured.value.code == expected_code


def test_registry_rejects_wrong_types_snapshots_and_restore_binding() -> None:
    registry, reference, _state = _configurable_codec_registry()
    snapshot = _codec_snapshot(reference)

    assert not registry.is_registered(cast(Any, object()))
    with pytest.raises(ModelCapabilityValidationError) as snapshot_error:
        registry.export_snapshot(
            reference,
            replace(snapshot, snapshot_kind="other.provider-response"),
        )
    assert snapshot_error.value.code == "capability.continuation_codec"

    encoded = registry.export_snapshot(reference, snapshot)
    with pytest.raises(ModelCapabilityValidationError) as restore_error:
        registry.restore_snapshot(
            reference,
            encoded,
            _revision_binding(model_id="other-model"),
        )
    assert restore_error.value.code == "capability.continuation_codec"


def test_durable_binding_mismatch_falls_back_to_attached_resolution() -> None:
    registry, reference = _registered_codec()
    support = ProviderCapabilitySupport(
        structured_invocation=True,
        stable_call_ids=True,
        correlated_results=True,
        attached_resolution=True,
        durable_store=True,
        registered_resumer=True,
        continuation_snapshot_codec_registry=registry,
        continuation_snapshot_codec=reference,
    )

    catalog = ModelCapabilityCatalog.create(
        support=support,
        revision_binding=_revision_binding(model_id="other-model"),
    )

    assert (
        catalog.task_input_advertisement
        is TaskInputCapabilityAdvertisement.ATTACHED
    )


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("version", "capability.seed_version"),
        ("descriptors_type", "capability.seed"),
        ("map_value", "capability.seed"),
        ("provider_family", "capability.seed"),
        ("mode_type", "capability.seed"),
        ("mode_value", "capability.seed"),
        ("prefix", "capability.seed"),
        ("collapse", "capability.seed"),
        ("tool_format_type", "capability.seed"),
        ("tool_format_value", "capability.seed"),
        ("eos_token", "capability.seed"),
        ("recovery_formats_type", "capability.seed"),
        ("recovery_format_value", "capability.seed"),
        ("parser_limit", "capability.seed"),
        ("descriptor_name", "capability.seed"),
        ("descriptor_alias", "capability.seed"),
        ("nested_object", "capability.seed"),
        ("extra_field", "capability.seed"),
    ),
)
def test_domain_seed_decode_rejects_each_malformed_field(
    case: str, expected_code: str
) -> None:
    seed = deepcopy(_manager().export_model_capability_seed())
    policy = cast(dict[str, Any], seed["name_policy"])
    parser = cast(dict[str, Any], seed["parser"])
    descriptors = cast(list[dict[str, Any]], seed["descriptors"])

    match case:
        case "version":
            seed["version"] = 2
        case "descriptors_type":
            seed["descriptors"] = "not-an-array"
        case "map_value":
            cast(dict[str, Any], policy["map"])["pkg.lookup"] = 1
        case "provider_family":
            policy["provider_family"] = 1
        case "mode_type":
            policy["mode"] = 1
        case "mode_value":
            policy["mode"] = "unsupported"
        case "prefix":
            policy["prefix"] = 1
        case "collapse":
            policy["collapse_replacement"] = 1
        case "tool_format_type":
            parser["tool_format"] = 1
        case "tool_format_value":
            parser["tool_format"] = "unsupported"
        case "eos_token":
            parser["eos_token"] = 1
        case "recovery_formats_type":
            parser["recovery_formats"] = "not-an-array"
        case "recovery_format_value":
            parser["recovery_formats"] = ["unsupported"]
        case "parser_limit":
            parser["maximum_input_size"] = 0
        case "descriptor_name":
            descriptors[0]["canonical_name"] = 1
        case "descriptor_alias":
            descriptors[0]["aliases"] = [1]
        case "nested_object":
            seed["name_policy"] = []
        case "extra_field":
            seed["unexpected"] = True
        case _:
            raise AssertionError(f"unknown mutation case {case}")

    with pytest.raises(ModelCapabilityValidationError) as captured:
        DomainCapabilitySeed.decode(seed)

    assert captured.value.code == expected_code


def test_domain_seed_accepts_explicit_name_policy_provider_family() -> None:
    seed = deepcopy(_manager().export_model_capability_seed())
    policy = cast(dict[str, Any], seed["name_policy"])
    policy["provider_family"] = "openai"

    decoded = DomainCapabilitySeed.decode(seed)

    assert decoded.name_policy_provider_family == "openai"


def _descriptor(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    kind: ModelCapabilityKind = ModelCapabilityKind.DOMAIN_TOOL,
) -> ModelCapabilityDescriptor:
    return ModelCapabilityDescriptor(
        canonical_name=name,
        description=f"Invoke {name}.",
        aliases=aliases,
        parameter_schema={"type": "object"},
        kind=kind,
    )


@pytest.mark.parametrize(
    "descriptors",
    (
        (_descriptor("duplicate"), _descriptor("duplicate")),
        (_descriptor("first"), _descriptor("second", aliases=("first",))),
    ),
    ids=("canonical-name", "alias"),
)
def test_catalog_rejects_duplicate_names_and_aliases(
    descriptors: tuple[ModelCapabilityDescriptor, ...],
) -> None:
    with pytest.raises(ModelCapabilityValidationError) as captured:
        ModelCapabilityCatalog.create(
            DomainCapabilitySeed(descriptors=descriptors)
        )

    assert captured.value.code == "capability.duplicate"


def test_catalog_rejects_reserved_descriptor_kind_and_wrong_binding_type() -> (
    None
):
    with pytest.raises(ModelCapabilityValidationError) as descriptor_error:
        ModelCapabilityCatalog.create(
            DomainCapabilitySeed(
                descriptors=(
                    _descriptor(
                        "domain.input", kind=ModelCapabilityKind.TASK_INPUT
                    ),
                )
            )
        )
    assert descriptor_error.value.code == "capability.seed"

    with pytest.raises(ModelCapabilityValidationError) as binding_error:
        ModelCapabilityCatalog.create(revision_binding=cast(Any, object()))
    assert binding_error.value.code == "capability.continuation_binding"


def test_catalog_detects_provider_collision_after_policy_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = deepcopy(_manager().export_model_capability_seed())
    descriptors = cast(list[dict[str, Any]], seed["descriptors"])
    descriptors.append(
        {
            **deepcopy(descriptors[0]),
            "canonical_name": "pkg.other",
        }
    )
    policy = cast(dict[str, Any], seed["name_policy"])
    policy["map"] = {"pkg.lookup": "same", "pkg.other": "same"}
    catalog = ModelCapabilityCatalog.create(seed)
    monkeypatch.setattr(
        capability_module.ToolNamePolicy,
        "bind",
        lambda self, _names: self,
    )

    with pytest.raises(ModelCapabilityValidationError) as captured:
        catalog.project("openai")

    assert captured.value.code == "capability.provider_collision"


def test_catalog_call_and_result_boundaries_reject_wrong_types() -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed(),
        support=_attached_support(),
    )

    with pytest.raises(ModelCapabilityValidationError) as call_error:
        catalog.decode_call(cast(Any, object()), provider_family="openai")
    assert call_error.value.code == "capability.call_type"

    rejected = catalog.classify_batch(
        [cast(Any, object())], provider_family="openai"
    )
    assert isinstance(rejected, CapabilityBatchRejected)
    assert rejected.code is CapabilityBatchRejectionCode.UNKNOWN_CAPABILITY

    with pytest.raises(ModelCapabilityValidationError) as correlation_error:
        catalog.project_result(cast(Any, object()), cast(Any, object()))
    assert correlation_error.value.code == "capability.result_correlation"

    task_input = catalog.decode_call(
        ProviderCapabilityCall(
            call_id="result-call",
            provider_name=RESERVED_INPUT_CAPABILITY_NAME,
            arguments=_input_arguments(),
        ),
        provider_family="openai",
    )
    assert isinstance(task_input, TaskInputCapabilityCall)
    with pytest.raises(ModelCapabilityValidationError) as result_error:
        catalog.project_result(task_input, cast(Any, object()))
    assert result_error.value.code == "capability.result"


def test_local_parser_resolves_alias_and_rejects_unadvertised_name() -> None:
    seed = deepcopy(_manager().export_model_capability_seed())
    descriptor = cast(list[dict[str, Any]], seed["descriptors"])[0]
    descriptor["aliases"] = ["lookup-alias"]
    catalog = ModelCapabilityCatalog.create(seed)

    aliased = catalog.parse_calls(
        '<tool_call>{"name":"lookup-alias","arguments":{"query":"value"}}'
        "</tool_call>"
    )
    unknown = catalog.parse_calls(
        '<tool_call>{"name":"missing","arguments":{}}</tool_call>'
    )

    assert aliased.calls[0].name == "pkg.lookup"
    assert aliased.calls[0].provider_name == "lookup-alias"
    assert unknown.calls[0].name == ""
    assert unknown.calls[0].provider_name == "missing"


def test_local_parser_preserves_exact_name_and_resolves_harmony_prefix() -> (
    None
):
    exact = ModelCapabilityCatalog.create(
        DomainCapabilitySeed(descriptors=(_descriptor("plain"),))
    )
    exact_call = exact.parse_calls(
        '<tool_call>{"name":"plain","arguments":{}}</tool_call>'
    ).calls[0]
    assert exact_call.name == "plain"
    assert exact_call.provider_name is None

    harmony = ModelCapabilityCatalog.create(
        DomainCapabilitySeed(
            descriptors=(_descriptor("functions.calculate"),),
            tool_format=ToolFormat.HARMONY,
        )
    )
    harmony_call = harmony._canonical_provider_originated_call(  # noqa: SLF001
        ToolCall(id="harmony-call", name="calculate", arguments={})
    )
    assert harmony_call.name == "functions.calculate"
    assert harmony_call.provider_name == "calculate"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mode", 1),
        ("mode", "unsupported"),
        ("questions", {}),
    ),
)
def test_decode_task_input_rejects_invalid_mode_and_questions(
    field: str, value: object
) -> None:
    catalog = ModelCapabilityCatalog.create(support=_attached_support())
    arguments = _input_arguments()
    arguments[field] = value

    with pytest.raises(ModelCapabilityValidationError) as captured:
        catalog.decode_call(
            ProviderCapabilityCall(
                call_id="input-call",
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=arguments,
            ),
            provider_family="openai",
        )

    assert captured.value.code == "capability.schema_validation"


def test_decode_task_input_rejects_semantically_invalid_reason() -> None:
    catalog = ModelCapabilityCatalog.create(support=_attached_support())
    arguments = _input_arguments()
    arguments["reason"] = "line one\nline two"

    with pytest.raises(ModelCapabilityValidationError) as captured:
        catalog.decode_call(
            ProviderCapabilityCall(
                call_id="input-call",
                provider_name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=arguments,
            ),
            provider_family="openai",
        )

    assert captured.value.code == "capability.arguments"


def _domain_call(
    catalog: ModelCapabilityCatalog,
    arguments: str | Mapping[str, object] | None,
) -> ToolCall:
    decoded = catalog.decode_call(
        ProviderCapabilityCall(
            call_id="domain-call",
            provider_name="lookup",
            arguments=arguments,
        ),
        provider_family="openai",
    )
    assert isinstance(decoded, ToolCall)
    return decoded


def test_argument_json_handles_escapes_and_rejects_lone_surrogate() -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed()
    )

    decoded = _domain_call(
        catalog,
        '{"query":"quote: \\" slash: \\\\"}',
    )
    assert decoded.arguments == {"query": 'quote: " slash: \\'}

    with pytest.raises(ModelCapabilityValidationError) as captured:
        _domain_call(catalog, '{"query":"\ud800"}')
    assert captured.value.code == "capability.arguments_json"


def test_argument_decoder_wraps_recursion_and_serialization_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed()
    )
    monkeypatch.setattr(
        capability_module,
        "loads",
        MagicMock(side_effect=RecursionError("decoder recursion")),
    )
    with pytest.raises(ModelCapabilityValidationError) as decode_error:
        _domain_call(catalog, '{"query":"value"}')
    assert decode_error.value.code == "capability.arguments_depth"


@pytest.mark.parametrize(
    ("exception", "expected_code"),
    (
        (RecursionError("encoder recursion"), "capability.arguments_depth"),
        (TypeError("encoder type"), "capability.arguments_json"),
    ),
    ids=("recursion", "type"),
)
def test_argument_decoder_wraps_encoder_failures(
    monkeypatch: pytest.MonkeyPatch,
    exception: BaseException,
    expected_code: str,
) -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed()
    )
    monkeypatch.setattr(
        capability_module,
        "dumps",
        MagicMock(side_effect=exception),
    )

    with pytest.raises(ModelCapabilityValidationError) as captured:
        _domain_call(catalog, {"query": "value"})

    assert captured.value.code == expected_code


def test_argument_decoder_enforces_object_size_depth_and_json_types() -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed()
    )

    with pytest.raises(ModelCapabilityValidationError) as array_error:
        _domain_call(catalog, "[]")
    assert array_error.value.code == "capability.arguments_type"

    with pytest.raises(ModelCapabilityValidationError) as empty_error:
        _domain_call(catalog, None)
    assert empty_error.value.code == "capability.schema_validation"

    oversized_json = '{"query":"' + ("x" * 1_048_576) + '"}'
    with pytest.raises(ModelCapabilityValidationError) as raw_size_error:
        _domain_call(catalog, oversized_json)
    assert raw_size_error.value.code == "capability.arguments_size"

    with pytest.raises(ModelCapabilityValidationError) as size_error:
        _domain_call(catalog, {"query": "x" * 1_048_576})
    assert size_error.value.code == "capability.arguments_size"

    with pytest.raises(ModelCapabilityValidationError) as finite_error:
        _domain_call(catalog, {"query": inf})
    assert finite_error.value.code == "capability.arguments_json"

    with pytest.raises(ModelCapabilityValidationError) as schema_error:
        _domain_call(catalog, {"query": 1.5})
    assert schema_error.value.code == "capability.schema_validation"

    nested_list: object = "value"
    for _ in range(40):
        nested_list = [nested_list]
    with pytest.raises(ModelCapabilityValidationError) as depth_error:
        _domain_call(catalog, {"query": nested_list})
    assert depth_error.value.code == "capability.arguments_depth"

    with pytest.raises(ModelCapabilityValidationError) as key_error:
        _domain_call(catalog, cast(Any, {1: "value"}))
    assert key_error.value.code == "capability.arguments_json"


def test_schema_validation_dependency_failure_is_content_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = ModelCapabilityCatalog.create(
        _manager().export_model_capability_seed()
    )
    monkeypatch.setattr(
        capability_module,
        "import_module",
        MagicMock(side_effect=ImportError("jsonschema unavailable")),
    )

    with pytest.raises(ModelCapabilityValidationError) as captured:
        _domain_call(catalog, {"query": "value"})

    assert captured.value.code == "capability.schema_unavailable"


def test_descriptor_json_freezing_accepts_and_rejects_values() -> None:
    descriptor = ModelCapabilityDescriptor(
        canonical_name="finite",
        description="Accept a finite number.",
        parameter_schema={"value": 1.5},
    )
    assert descriptor.parameter_schema["value"] == 1.5

    for schema in (
        {"value": inf},
        cast(Any, {1: "value"}),
        {"value": object()},
    ):
        with pytest.raises(ModelCapabilityValidationError) as captured:
            ModelCapabilityDescriptor(
                canonical_name="invalid",
                description="Reject invalid JSON.",
                parameter_schema=schema,
            )
        assert captured.value.code == "capability.non_json"

    nested_list: object = "value"
    for _ in range(65):
        nested_list = [nested_list]
    with pytest.raises(ModelCapabilityValidationError) as depth_error:
        ModelCapabilityDescriptor(
            canonical_name="deep",
            description="Reject deep JSON.",
            parameter_schema={"value": nested_list},
        )
    assert depth_error.value.code == "capability.non_json"
