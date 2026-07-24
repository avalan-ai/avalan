"""Define immutable model capability catalogs and provider projections."""

from ..entities import (
    Message,
    MessageRole,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallParseOutcome,
    ToolCallRecoveryFormat,
    ToolCallResult,
    ToolFormat,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
    ToolValue,
    normalize_tool_arguments,
)
from ..interaction.codec import (
    decode_input_question,
    encode_input_model_result,
)
from ..interaction.entities import (
    RESERVED_INPUT_CAPABILITY_NAME,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    InputModelResult,
    InputQuestion,
    ModelCallId,
    ProviderIdempotencyKey,
    RequirementMode,
)
from ..interaction.error import InputContractError
from ..interaction.validation import (
    validate_opaque_id,
    validate_presentation_text,
)
from ..tool.name_policy import ToolNamePolicy
from ..tool.parser import ToolCallParser
from ..types import JsonValue

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from importlib import import_module
from json import dumps, loads
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias, cast, final
from uuid import UUID

_DOMAIN_SEED_VERSION = 1
_MAX_ARGUMENT_UTF8_BYTES = 1_048_576
_MAX_ARGUMENT_DEPTH = 32
_MAX_INPUT_ARGUMENT_UTF8_BYTES = 131_072
_MAX_INPUT_ARGUMENT_DEPTH = 12
_MAX_FROZEN_JSON_DEPTH = 64
_REGISTERED_CODEC_PROOF = object()


class ModelCapabilityKind(StrEnum):
    """Identify a model-visible capability category."""

    DOMAIN_TOOL = "domain_tool"
    TASK_INPUT = "task_input"


class TaskInputCapabilityAdvertisement(StrEnum):
    """Identify the strongest safe task-input advertisement."""

    INCAPABLE = "incapable"
    ATTACHED = "attached"
    DURABLE = "durable"


class CapabilityBatchRejectionCode(StrEnum):
    """Identify a fail-closed provider-call batch rejection."""

    UNKNOWN_CAPABILITY = "capability.unknown"
    MALFORMED_CALL = "capability.malformed_call"
    MISSING_CALL_ID = "capability.missing_call_id"
    NON_STRUCTURED_CALL = "capability.non_structured_call"
    MIXED_TASK_INPUT_BATCH = "capability.mixed_task_input_batch"
    MULTIPLE_TASK_INPUT_CALLS = "capability.multiple_task_input_calls"


class ModelCapabilityError(ValueError):
    """Report a content-safe model capability failure."""


class ModelCapabilityValidationError(ModelCapabilityError):
    """Report invalid capability configuration or provider content."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        assert isinstance(code, str) and code.strip()
        assert isinstance(message, str) and message.strip()
        self.code = code
        super().__init__(message)


@final
@dataclass(frozen=True, slots=True, init=False, eq=False)
class RegisteredContinuationSnapshotCodec:
    """Carry registry-minted evidence for one validated snapshot codec."""

    registry_id: str
    codec_id: str
    revision_binding: ContinuationRevisionBinding
    snapshot_kind: str
    _proof: object = field(repr=False, compare=False)

    def __new__(cls) -> "RegisteredContinuationSnapshotCodec":
        raise TypeError(
            "registered continuation snapshot codecs come from a registry"
        )

    @classmethod
    def _mint(
        cls,
        *,
        registry_id: str,
        codec_id: str,
        revision_binding: ContinuationRevisionBinding,
        snapshot_kind: str,
        proof: object,
    ) -> "RegisteredContinuationSnapshotCodec":
        if proof is not _REGISTERED_CODEC_PROOF:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec evidence is not registry-minted",
            )
        reference = object.__new__(cls)
        object.__setattr__(reference, "registry_id", registry_id)
        object.__setattr__(reference, "codec_id", codec_id)
        object.__setattr__(reference, "revision_binding", revision_binding)
        object.__setattr__(reference, "snapshot_kind", snapshot_kind)
        object.__setattr__(reference, "_proof", proof)
        return reference

    @property
    def is_registry_minted(self) -> bool:
        """Return whether a validating registry minted this evidence."""
        return self._proof is _REGISTERED_CODEC_PROOF

    def accepts(self, snapshot: ContinuationSnapshot) -> bool:
        """Return whether this registration owns the snapshot contract."""
        return (
            self.is_registry_minted
            and type(snapshot) is ContinuationSnapshot
            and snapshot.revision_binding == self.revision_binding
            and snapshot.snapshot_kind == self.snapshot_kind
        )


ContinuationSnapshotExporter: TypeAlias = Callable[[ContinuationSnapshot], str]
ContinuationSnapshotRestorer: TypeAlias = Callable[
    [str, ContinuationRevisionBinding], ContinuationSnapshot
]


@dataclass(frozen=True, slots=True)
class _ContinuationSnapshotCodecRegistration:
    reference: RegisteredContinuationSnapshotCodec
    export_snapshot: ContinuationSnapshotExporter = field(repr=False)
    restore_snapshot: ContinuationSnapshotRestorer = field(repr=False)


@final
class ContinuationSnapshotCodecRegistry:
    """Register validated codecs without leaking callables into catalogs."""

    def __init__(self, registry_id: str) -> None:
        try:
            self._registry_id = validate_opaque_id(
                registry_id,
                "continuation_snapshot_codec_registry.registry_id",
            )
        except InputContractError as exc:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec registry identifier is invalid",
            ) from exc
        self._registrations: dict[
            str, _ContinuationSnapshotCodecRegistration
        ] = {}

    def register(
        self,
        *,
        codec_id: str,
        revision_binding: ContinuationRevisionBinding,
        snapshot_kind: str,
        export_snapshot: ContinuationSnapshotExporter,
        restore_snapshot: ContinuationSnapshotRestorer,
    ) -> None:
        """Validate and register one exact snapshot codec contract."""
        if type(revision_binding) is not ContinuationRevisionBinding:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec requires an exact revision "
                "binding",
            )
        if not callable(export_snapshot) or not callable(restore_snapshot):
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec requires export and restore",
            )
        try:
            validated_codec_id = validate_opaque_id(
                codec_id,
                "continuation_snapshot_codec.codec_id",
            )
            probe = ContinuationSnapshot(
                snapshot_kind=snapshot_kind,
                revision_binding=revision_binding,
                model_call_id=ModelCallId("codec-registration-probe"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "codec-registration-probe"
                ),
                payload={"registration_probe": True},
            )
            encoded = export_snapshot(probe)
            if not isinstance(encoded, str) or not encoded:
                raise ModelCapabilityValidationError(
                    "capability.continuation_codec",
                    "continuation snapshot exporter returned invalid content",
                )
            restored = restore_snapshot(encoded, revision_binding)
        except ModelCapabilityValidationError:
            raise
        except (InputContractError, TypeError, ValueError) as exc:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec registration is invalid",
            ) from exc
        if type(restored) is not ContinuationSnapshot or restored != probe:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec failed its validation round trip",
            )
        if validated_codec_id in self._registrations:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec is already registered",
            )
        reference = RegisteredContinuationSnapshotCodec._mint(
            registry_id=self._registry_id,
            codec_id=validated_codec_id,
            revision_binding=revision_binding,
            snapshot_kind=probe.snapshot_kind,
            proof=_REGISTERED_CODEC_PROOF,
        )
        self._registrations[validated_codec_id] = (
            _ContinuationSnapshotCodecRegistration(
                reference=reference,
                export_snapshot=export_snapshot,
                restore_snapshot=restore_snapshot,
            )
        )

    def reference(self, codec_id: str) -> RegisteredContinuationSnapshotCodec:
        """Return immutable evidence from an exact registry lookup."""
        try:
            validated_codec_id = validate_opaque_id(
                codec_id,
                "continuation_snapshot_codec.codec_id",
            )
            registration = self._registrations[validated_codec_id]
        except (InputContractError, KeyError) as exc:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec is not registered",
            ) from exc
        return registration.reference

    def is_registered(
        self, reference: RegisteredContinuationSnapshotCodec
    ) -> bool:
        """Return whether this registry still owns the exact reference."""
        if type(reference) is not RegisteredContinuationSnapshotCodec:
            return False
        registration = self._registrations.get(reference.codec_id)
        return (
            reference.registry_id == self._registry_id
            and reference.is_registry_minted
            and registration is not None
            and registration.reference is reference
        )

    def export_snapshot(
        self,
        reference: RegisteredContinuationSnapshotCodec,
        snapshot: ContinuationSnapshot,
    ) -> str:
        """Export a snapshot through one live registered reference."""
        registration = self._registration(reference)
        if not reference.accepts(snapshot):
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot does not match the registered codec",
            )
        try:
            encoded = registration.export_snapshot(snapshot)
        except ModelCapabilityValidationError:
            raise
        except (InputContractError, TypeError, ValueError) as exc:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot export failed",
            ) from exc
        if not isinstance(encoded, str) or not encoded:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot exporter returned invalid content",
            )
        return encoded

    def restore_snapshot(
        self,
        reference: RegisteredContinuationSnapshotCodec,
        value: str,
        revision_binding: ContinuationRevisionBinding,
    ) -> ContinuationSnapshot:
        """Restore a snapshot through one live registered reference."""
        registration = self._registration(reference)
        if (
            not isinstance(value, str)
            or not value
            or type(revision_binding) is not ContinuationRevisionBinding
            or revision_binding != reference.revision_binding
        ):
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot restore contract does not match the "
                "registered codec",
            )
        try:
            restored = registration.restore_snapshot(value, revision_binding)
        except ModelCapabilityValidationError:
            raise
        except (InputContractError, TypeError, ValueError) as exc:
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot restore failed",
            ) from exc
        if not reference.accepts(restored):
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot restorer returned an invalid snapshot",
            )
        return restored

    def _registration(
        self, reference: RegisteredContinuationSnapshotCodec
    ) -> _ContinuationSnapshotCodecRegistration:
        if not self.is_registered(reference):
            raise ModelCapabilityValidationError(
                "capability.continuation_codec",
                "continuation snapshot codec reference is not live in this "
                "registry",
            )
        return self._registrations[reference.codec_id]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderCapabilitySupport:
    """Declare trusted prerequisites for structured task input."""

    structured_invocation: bool = False
    stable_call_ids: bool = False
    correlated_results: bool = False
    attached_resolution: bool = False
    durable_store: bool = False
    registered_resumer: bool = False
    continuation_snapshot_codec_registry: (
        ContinuationSnapshotCodecRegistry | None
    ) = None
    continuation_snapshot_codec: RegisteredContinuationSnapshotCodec | None = (
        None
    )

    def __post_init__(self) -> None:
        for name in (
            "structured_invocation",
            "stable_call_ids",
            "correlated_results",
            "attached_resolution",
            "durable_store",
            "registered_resumer",
        ):
            assert type(getattr(self, name)) is bool, f"{name} must be boolean"
        assert self.continuation_snapshot_codec is None or (
            type(self.continuation_snapshot_codec)
            is RegisteredContinuationSnapshotCodec
            and self.continuation_snapshot_codec.is_registry_minted
        ), "continuation_snapshot_codec must be a registered codec reference"
        assert self.continuation_snapshot_codec_registry is None or (
            type(self.continuation_snapshot_codec_registry)
            is ContinuationSnapshotCodecRegistry
        ), "continuation_snapshot_codec_registry must be a codec registry"

    @property
    def task_input_advertisement(
        self,
    ) -> TaskInputCapabilityAdvertisement:
        """Return the strongest task-input mode whose prerequisites hold."""
        structured = (
            self.structured_invocation
            and self.stable_call_ids
            and self.correlated_results
        )
        if not structured:
            return TaskInputCapabilityAdvertisement.INCAPABLE
        if (
            self.durable_store
            and self.registered_resumer
            and self.continuation_snapshot_codec_registry is not None
            and self.continuation_snapshot_codec is not None
            and self.continuation_snapshot_codec_registry.is_registered(
                self.continuation_snapshot_codec
            )
        ):
            return TaskInputCapabilityAdvertisement.DURABLE
        if self.attached_resolution:
            return TaskInputCapabilityAdvertisement.ATTACHED
        return TaskInputCapabilityAdvertisement.INCAPABLE

    def task_input_advertisement_for(
        self,
        revision_binding: ContinuationRevisionBinding | None,
    ) -> TaskInputCapabilityAdvertisement:
        """Return advertisement strength for one exact model revision path."""
        advertisement = self.task_input_advertisement
        if advertisement is not TaskInputCapabilityAdvertisement.DURABLE:
            return advertisement
        codec = self.continuation_snapshot_codec
        assert codec is not None
        if revision_binding == codec.revision_binding:
            return advertisement
        if self.attached_resolution:
            return TaskInputCapabilityAdvertisement.ATTACHED
        return TaskInputCapabilityAdvertisement.INCAPABLE


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapabilityDescriptor:
    """Describe one callable-free model-visible capability."""

    canonical_name: str
    description: str
    parameter_schema: Mapping[str, JsonValue]
    result_schema: Mapping[str, JsonValue] | None = None
    aliases: tuple[str, ...] = ()
    kind: ModelCapabilityKind = ModelCapabilityKind.DOMAIN_TOOL

    def __post_init__(self) -> None:
        _assert_name(self.canonical_name, "canonical_name")
        assert isinstance(self.description, str)
        assert self.description.strip(), "description must not be empty"
        object.__setattr__(
            self,
            "parameter_schema",
            _freeze_json_object(self.parameter_schema, "parameter_schema"),
        )
        if self.result_schema is not None:
            object.__setattr__(
                self,
                "result_schema",
                _freeze_json_object(self.result_schema, "result_schema"),
            )
        assert isinstance(self.aliases, tuple)
        for alias in self.aliases:
            _assert_name(alias, "alias")
        assert isinstance(self.kind, ModelCapabilityKind)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DomainCapabilitySeed:
    """Carry a callable-free ToolManager export into the model layer."""

    descriptors: tuple[ModelCapabilityDescriptor, ...] = ()
    name_policy_mode: ToolNamePolicyMode = ToolNamePolicyMode.ENCODED
    name_policy_prefix: str = "avl_"
    name_policy_replacement: str = "_"
    name_policy_collapse_replacement: bool = True
    name_policy_map: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    name_policy_provider_family: str | None = None
    tool_format: ToolFormat | None = None
    eos_token: str | None = None
    recovery_formats: tuple[ToolCallRecoveryFormat, ...] = ()
    maximum_parser_input_size: int | None = None
    maximum_parser_payload_depth: int | None = None
    maximum_parser_payload_size: int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.descriptors, tuple)
        assert all(
            isinstance(descriptor, ModelCapabilityDescriptor)
            for descriptor in self.descriptors
        )
        assert isinstance(self.name_policy_mode, ToolNamePolicyMode)
        assert isinstance(self.name_policy_prefix, str)
        assert isinstance(self.name_policy_replacement, str)
        assert type(self.name_policy_collapse_replacement) is bool
        name_map: dict[str, str] = {}
        for canonical_name, provider_name in self.name_policy_map.items():
            _assert_name(canonical_name, "name_policy_map key")
            _assert_name(provider_name, "name_policy_map value")
            name_map[canonical_name] = provider_name
        object.__setattr__(self, "name_policy_map", MappingProxyType(name_map))
        if self.name_policy_provider_family is not None:
            _assert_name(
                self.name_policy_provider_family,
                "name_policy_provider_family",
            )
        if self.tool_format is not None:
            assert isinstance(self.tool_format, ToolFormat)
        if self.eos_token is not None:
            assert isinstance(self.eos_token, str)
        assert isinstance(self.recovery_formats, tuple)
        assert all(
            isinstance(recovery_format, ToolCallRecoveryFormat)
            for recovery_format in self.recovery_formats
        )
        for limit in (
            self.maximum_parser_input_size,
            self.maximum_parser_payload_depth,
            self.maximum_parser_payload_size,
        ):
            assert limit is None or (
                isinstance(limit, int)
                and not isinstance(limit, bool)
                and limit > 0
            )
        self.name_policy_settings()

    @classmethod
    def decode(cls, value: Mapping[str, object]) -> "DomainCapabilitySeed":
        """Decode one strict callable-free ToolManager export payload."""
        _exact_keys(
            value,
            {"version", "name_policy", "parser", "descriptors"},
            "seed",
        )
        if value["version"] != _DOMAIN_SEED_VERSION:
            raise ModelCapabilityValidationError(
                "capability.seed_version",
                "unsupported capability seed version",
            )
        raw_policy = _object(value["name_policy"], "seed.name_policy")
        _exact_keys(
            raw_policy,
            {
                "mode",
                "prefix",
                "replacement",
                "collapse_replacement",
                "map",
                "provider_family",
            },
            "seed.name_policy",
        )
        raw_parser = _object(value["parser"], "seed.parser")
        _exact_keys(
            raw_parser,
            {
                "tool_format",
                "eos_token",
                "recovery_formats",
                "maximum_input_size",
                "maximum_payload_depth",
                "maximum_payload_size",
            },
            "seed.parser",
        )
        raw_descriptors = value["descriptors"]
        if not isinstance(raw_descriptors, list | tuple):
            raise ModelCapabilityValidationError(
                "capability.seed", "seed descriptors must be an array"
            )
        descriptors = tuple(
            _decode_domain_descriptor(item, index)
            for index, item in enumerate(raw_descriptors)
        )
        raw_map = _object(raw_policy["map"], "seed.name_policy.map")
        if not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in raw_map.items()
        ):
            raise ModelCapabilityValidationError(
                "capability.seed", "name policy map must contain strings"
            )
        raw_provider_family = raw_policy["provider_family"]
        if raw_provider_family is not None and not isinstance(
            raw_provider_family, str
        ):
            raise ModelCapabilityValidationError(
                "capability.seed", "provider family must be a string or null"
            )
        raw_mode = raw_policy["mode"]
        if not isinstance(raw_mode, str):
            raise ModelCapabilityValidationError(
                "capability.seed", "name policy mode is invalid"
            )
        try:
            mode = ToolNamePolicyMode(raw_mode)
        except (TypeError, ValueError) as exc:
            raise ModelCapabilityValidationError(
                "capability.seed", "name policy mode is invalid"
            ) from exc
        prefix = raw_policy["prefix"]
        replacement = raw_policy["replacement"]
        collapse = raw_policy["collapse_replacement"]
        if not isinstance(prefix, str) or not isinstance(replacement, str):
            raise ModelCapabilityValidationError(
                "capability.seed", "name policy strings are invalid"
            )
        if type(collapse) is not bool:
            raise ModelCapabilityValidationError(
                "capability.seed", "name policy collapse flag is invalid"
            )
        raw_tool_format = raw_parser["tool_format"]
        if raw_tool_format is not None and not isinstance(
            raw_tool_format, str
        ):
            raise ModelCapabilityValidationError(
                "capability.seed", "parser tool format is invalid"
            )
        try:
            tool_format = (
                None
                if raw_tool_format is None
                else ToolFormat(raw_tool_format)
            )
        except (TypeError, ValueError) as exc:
            raise ModelCapabilityValidationError(
                "capability.seed", "parser tool format is invalid"
            ) from exc
        eos_token = raw_parser["eos_token"]
        if eos_token is not None and not isinstance(eos_token, str):
            raise ModelCapabilityValidationError(
                "capability.seed", "parser eos token must be a string or null"
            )
        raw_recovery_formats = raw_parser["recovery_formats"]
        if not isinstance(raw_recovery_formats, list | tuple):
            raise ModelCapabilityValidationError(
                "capability.seed", "parser recovery formats must be an array"
            )
        try:
            recovery_formats = tuple(
                ToolCallRecoveryFormat(recovery_format)
                for recovery_format in raw_recovery_formats
            )
        except (TypeError, ValueError) as exc:
            raise ModelCapabilityValidationError(
                "capability.seed", "parser recovery format is invalid"
            ) from exc
        parser_limits: list[int | None] = []
        for name in (
            "maximum_input_size",
            "maximum_payload_depth",
            "maximum_payload_size",
        ):
            limit = raw_parser[name]
            if limit is not None and (
                not isinstance(limit, int)
                or isinstance(limit, bool)
                or limit <= 0
            ):
                raise ModelCapabilityValidationError(
                    "capability.seed", f"parser {name} is invalid"
                )
            if limit is None:
                parser_limits.append(None)
            else:
                assert isinstance(limit, int)
                parser_limits.append(limit)
        return cls(
            descriptors=descriptors,
            name_policy_mode=mode,
            name_policy_prefix=prefix,
            name_policy_replacement=replacement,
            name_policy_collapse_replacement=collapse,
            name_policy_map=cast(Mapping[str, str], raw_map),
            name_policy_provider_family=raw_provider_family,
            tool_format=tool_format,
            eos_token=eos_token,
            recovery_formats=recovery_formats,
            maximum_parser_input_size=parser_limits[0],
            maximum_parser_payload_depth=parser_limits[1],
            maximum_parser_payload_size=parser_limits[2],
        )

    def name_policy_settings(self) -> ToolNamePolicySettings:
        """Return an isolated ToolNamePolicy configuration."""
        return ToolNamePolicySettings(
            mode=self.name_policy_mode,
            prefix=self.name_policy_prefix,
            replacement=self.name_policy_replacement,
            collapse_replacement=self.name_policy_collapse_replacement,
            map=dict(self.name_policy_map),
            provider_family=self.name_policy_provider_family,
        )

    def parser(self) -> ToolCallParser:
        """Return a fresh parser configured from the immutable seed."""
        return ToolCallParser(
            tool_format=self.tool_format,
            eos_token=self.eos_token,
            recovery_formats=self.recovery_formats,
            maximum_text_size=self.maximum_parser_input_size,
            maximum_payload_depth=self.maximum_parser_payload_depth,
            maximum_payload_size=self.maximum_parser_payload_size,
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderCapabilityCall:
    """Carry one provider-native structured capability call."""

    call_id: UUID | str | None
    provider_name: str
    arguments: str | Mapping[str, object] | None = None
    structured: bool = True

    def __post_init__(self) -> None:
        if self.call_id is not None:
            assert isinstance(self.call_id, UUID | str)
            if isinstance(self.call_id, str):
                assert self.call_id.strip(), "call_id must not be empty"
        _assert_name(self.provider_name, "provider_name")
        assert self.arguments is None or isinstance(
            self.arguments, str | Mapping
        )
        assert type(self.structured) is bool


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class TaskInputCapabilityCall:
    """Carry one validated reserved task-input invocation."""

    call_id: UUID | str
    provider_name: str
    arguments: Mapping[str, JsonValue]
    mode: RequirementMode
    reason: str
    questions: tuple[InputQuestion, ...]
    advertisement: TaskInputCapabilityAdvertisement
    canonical_name: str = field(
        init=False, default=RESERVED_INPUT_CAPABILITY_NAME
    )

    def __post_init__(self) -> None:
        assert isinstance(self.call_id, UUID | str)
        _assert_name(self.provider_name, "provider_name")
        object.__setattr__(
            self,
            "arguments",
            _freeze_json_object(self.arguments, "arguments"),
        )
        assert isinstance(self.mode, RequirementMode)
        assert isinstance(self.reason, str) and self.reason.strip()
        assert isinstance(self.questions, tuple) and self.questions
        assert all(
            isinstance(question, InputQuestion) for question in self.questions
        )
        assert isinstance(self.advertisement, TaskInputCapabilityAdvertisement)


DomainCapabilityCall: TypeAlias = ToolCall


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderCapabilityProjection:
    """Expose one immutable provider-specific capability projection."""

    provider_family: Enum | str | None
    schemas: tuple[Mapping[str, JsonValue], ...]
    descriptors: tuple[ModelCapabilityDescriptor, ...]
    _canonical_to_provider: Mapping[str, str] = field(repr=False)
    _provider_to_canonical: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        assert isinstance(self.schemas, tuple)
        assert isinstance(self.descriptors, tuple)
        object.__setattr__(
            self,
            "_canonical_to_provider",
            MappingProxyType(dict(self._canonical_to_provider)),
        )
        object.__setattr__(
            self,
            "_provider_to_canonical",
            MappingProxyType(dict(self._provider_to_canonical)),
        )

    @property
    def is_empty(self) -> bool:
        """Return whether this projection advertises no capabilities."""
        return not self.schemas

    def provider_name(self, canonical_name: str) -> str:
        """Return the advertised provider name for a canonical name."""
        try:
            return self._canonical_to_provider[canonical_name]
        except KeyError as exc:
            raise ModelCapabilityValidationError(
                "capability.unknown", "canonical capability is not advertised"
            ) from exc

    def canonical_name(self, provider_name: str) -> str:
        """Return the canonical name for an advertised provider name."""
        try:
            return self._provider_to_canonical[provider_name]
        except KeyError as exc:
            raise ModelCapabilityValidationError(
                "capability.unknown", "provider capability is not advertised"
            ) from exc

    def tool_choice(self, canonical_name: str) -> str:
        """Return a validated provider-facing forced capability name."""
        return self.provider_name(canonical_name)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityBatchAccepted:
    """Carry a fully decoded batch safe for one next dispatch decision."""

    domain_calls: tuple[DomainCapabilityCall, ...] = ()
    task_input: TaskInputCapabilityCall | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.domain_calls, tuple)
        assert all(isinstance(call, ToolCall) for call in self.domain_calls)
        assert self.task_input is None or isinstance(
            self.task_input, TaskInputCapabilityCall
        )
        assert not (self.domain_calls and self.task_input is not None)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityBatchRejected:
    """Carry a content-safe batch rejection with no executable calls."""

    code: CapabilityBatchRejectionCode
    message: str

    def __post_init__(self) -> None:
        assert isinstance(self.code, CapabilityBatchRejectionCode)
        assert isinstance(self.message, str) and self.message.strip()


CapabilityBatchClassification: TypeAlias = (
    CapabilityBatchAccepted | CapabilityBatchRejected
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CorrelatedCapabilityResult:
    """Carry a model result correlated to its exact provider call."""

    call_id: UUID | str
    canonical_name: str
    provider_name: str
    payload: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        assert isinstance(self.call_id, UUID | str)
        _assert_name(self.canonical_name, "canonical_name")
        _assert_name(self.provider_name, "provider_name")
        object.__setattr__(
            self,
            "payload",
            _freeze_json_object(self.payload, "payload"),
        )

    def provider_payload(self) -> dict[str, object]:
        """Return an isolated JSON-ready result frame payload."""
        return cast(dict[str, object], _thaw_json(self.payload))

    def local_message(self) -> Message:
        """Return a provider-neutral local continuation message."""
        content = dumps(
            {
                "call_id": str(self.call_id),
                "name": self.provider_name,
                "result": self.provider_payload(),
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return Message(
            role=MessageRole.TOOL,
            content=content,
            name=self.provider_name,
        )

    def tool_result_message(
        self,
        call: TaskInputCapabilityCall,
    ) -> Message:
        """Return one correlated provider-neutral tool-result message."""
        if (
            not isinstance(call, TaskInputCapabilityCall)
            or call.call_id != self.call_id
            or call.canonical_name != self.canonical_name
            or call.provider_name != self.provider_name
        ):
            raise ModelCapabilityValidationError(
                "capability.result_correlation",
                "result does not match its task-input capability call",
            )
        arguments = normalize_tool_arguments(call.arguments)
        provider_call = ToolCall(
            id=call.call_id,
            name=call.canonical_name,
            arguments=arguments,
            provider_name=call.provider_name,
            provider_name_encoded=(call.provider_name != call.canonical_name),
        )
        local = self.local_message()
        return Message(
            role=local.role,
            content=local.content,
            name=local.name,
            tool_call_result=ToolCallResult(
                id=call.call_id,
                call=provider_call,
                name=provider_call.name,
                arguments=provider_call.arguments,
                provider_name=provider_call.provider_name,
                provider_name_encoded=(provider_call.provider_name_encoded),
                result=cast(
                    ToolValue,
                    normalize_tool_arguments(self.provider_payload()),
                ),
            ),
        )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapabilityCatalog:
    """Catalog strict model-facing capabilities without execution state."""

    domain_seed: DomainCapabilitySeed
    support: ProviderCapabilitySupport
    revision_binding: ContinuationRevisionBinding | None = None

    @classmethod
    def create(
        cls,
        domain_seed: Mapping[str, object] | DomainCapabilitySeed | None = None,
        *,
        support: ProviderCapabilitySupport | None = None,
        revision_binding: ContinuationRevisionBinding | None = None,
    ) -> "ModelCapabilityCatalog":
        """Create a catalog from a callable-free domain seed."""
        seed = (
            DomainCapabilitySeed()
            if domain_seed is None
            else (
                domain_seed
                if isinstance(domain_seed, DomainCapabilitySeed)
                else DomainCapabilitySeed.decode(domain_seed)
            )
        )
        resolved_support = support or ProviderCapabilitySupport()
        if revision_binding is not None and (
            type(revision_binding) is not ContinuationRevisionBinding
        ):
            raise ModelCapabilityValidationError(
                "capability.continuation_binding",
                "model capability catalog requires an exact revision binding",
            )
        names: set[str] = set()
        for descriptor in seed.descriptors:
            if descriptor.kind is not ModelCapabilityKind.DOMAIN_TOOL:
                raise ModelCapabilityValidationError(
                    "capability.seed",
                    "domain seed contains a reserved descriptor",
                )
            if (
                descriptor.canonical_name == RESERVED_INPUT_CAPABILITY_NAME
                or (RESERVED_INPUT_CAPABILITY_NAME in descriptor.aliases)
            ):
                raise ModelCapabilityValidationError(
                    "capability.reserved_name",
                    "domain capability uses reserved name",
                )
            if descriptor.canonical_name in names:
                raise ModelCapabilityValidationError(
                    "capability.duplicate",
                    "duplicate canonical capability name",
                )
            names.add(descriptor.canonical_name)
            for alias in descriptor.aliases:
                if alias in names:
                    raise ModelCapabilityValidationError(
                        "capability.duplicate",
                        "duplicate capability name or alias",
                    )
                names.add(alias)
        return cls(
            domain_seed=seed,
            support=resolved_support,
            revision_binding=revision_binding,
        )

    def __post_init__(self) -> None:
        assert isinstance(self.domain_seed, DomainCapabilitySeed)
        assert isinstance(self.support, ProviderCapabilitySupport)
        assert self.revision_binding is None or (
            type(self.revision_binding) is ContinuationRevisionBinding
        )

    @property
    def descriptors(self) -> tuple[ModelCapabilityDescriptor, ...]:
        """Return descriptors allowed by the current live support state."""
        descriptors = self.domain_seed.descriptors
        if (
            self.task_input_advertisement
            is not TaskInputCapabilityAdvertisement.INCAPABLE
        ):
            descriptors += (_task_input_descriptor(),)
        return descriptors

    @property
    def is_empty(self) -> bool:
        """Return whether the catalog has no model-visible capability."""
        return not self.descriptors

    @property
    def structured_parser_enabled(self) -> bool:
        """Return whether structured capability parsing should be enabled."""
        return not self.is_empty

    @property
    def task_input_advertisement(
        self,
    ) -> TaskInputCapabilityAdvertisement:
        """Return the reserved capability advertisement state."""
        return self.support.task_input_advertisement_for(self.revision_binding)

    @property
    def tool_format(self) -> ToolFormat | None:
        """Return the immutable parser tool format."""
        return self.domain_seed.tool_format

    @property
    def recovery_formats(self) -> tuple[ToolCallRecoveryFormat, ...]:
        """Return explicitly enabled structured recovery formats."""
        return self.domain_seed.recovery_formats

    def is_potential_tool_call(self, buffer: str, token_str: str) -> bool:
        """Return whether structured parser detection should run."""
        if not self.structured_parser_enabled:
            return False
        return self.domain_seed.parser().is_potential_tool_call(
            buffer, token_str
        )

    def tool_call_status(
        self, buffer: str, *, final: bool = False
    ) -> ToolCallParser.ToolCallBufferStatus:
        """Return the structured parser status for one buffer."""
        return self.domain_seed.parser().tool_call_status(buffer, final=final)

    def parse_calls(self, text: str) -> ToolCallParseOutcome:
        """Parse text control frames and canonicalize advertised names."""
        outcome = self.domain_seed.parser().parse(text)
        if not outcome.calls:
            return outcome
        return ToolCallParseOutcome(
            calls=[
                self._canonical_provider_originated_call(call)
                for call in outcome.calls
            ],
            diagnostics=outcome.diagnostics,
        )

    def get_calls(self, text: str) -> list[ToolCall] | None:
        """Return parsed canonical calls or ``None`` when absent."""
        calls = self.parse_calls(text).calls
        return calls or None

    def stream_buffer_diagnostics(
        self, buffer: str
    ) -> list[ToolCallDiagnostic]:
        """Return diagnostics for one terminal structured parser buffer."""
        return self.domain_seed.parser().stream_buffer_diagnostics(buffer)

    def project(
        self, provider_family: Enum | str | None = None
    ) -> ProviderCapabilityProjection:
        """Project immutable schemas and exact names for one provider."""
        descriptors = self._descriptors_for_provider(provider_family)
        try:
            policy = ToolNamePolicy(
                settings=self.domain_seed.name_policy_settings()
            ).for_provider(provider_family)
            policy = policy.bind(
                descriptor.canonical_name for descriptor in descriptors
            )
            canonical_to_provider = {
                descriptor.canonical_name: policy.provider_name(
                    descriptor.canonical_name
                )
                for descriptor in descriptors
            }
        except AssertionError as exc:
            raise ModelCapabilityValidationError(
                "capability.provider_projection",
                "capability names cannot be projected for this provider",
            ) from exc
        provider_to_canonical = {
            provider_name: canonical_name
            for canonical_name, provider_name in canonical_to_provider.items()
        }
        if len(provider_to_canonical) != len(canonical_to_provider):
            raise ModelCapabilityValidationError(
                "capability.provider_collision",
                "provider capability name collision",
            )
        schemas = tuple(
            _provider_schema(
                descriptor,
                canonical_to_provider[descriptor.canonical_name],
            )
            for descriptor in descriptors
        )
        return ProviderCapabilityProjection(
            provider_family=provider_family,
            schemas=schemas,
            descriptors=descriptors,
            _canonical_to_provider=canonical_to_provider,
            _provider_to_canonical=provider_to_canonical,
        )

    def _descriptors_for_provider(
        self, provider_family: Enum | str | None
    ) -> tuple[ModelCapabilityDescriptor, ...]:
        advertisement = self.task_input_advertisement
        if advertisement is not TaskInputCapabilityAdvertisement.DURABLE:
            return self.descriptors
        binding = self.revision_binding
        assert binding is not None
        provider_value = (
            provider_family.value
            if isinstance(provider_family, Enum)
            else provider_family
        )
        if provider_value == str(binding.provider_family):
            return self.descriptors
        return tuple(
            descriptor
            for descriptor in self.descriptors
            if descriptor.kind is not ModelCapabilityKind.TASK_INPUT
        )

    def provider_name(
        self,
        canonical_name: str,
        *,
        provider_family: Enum | str | None = None,
    ) -> str:
        """Return an advertised provider-facing capability name."""
        return self.project(provider_family).provider_name(canonical_name)

    def canonical_name(
        self,
        provider_name: str,
        *,
        provider_family: Enum | str | None = None,
    ) -> str:
        """Return the canonical name for an advertised provider name."""
        return self.project(provider_family).canonical_name(provider_name)

    def decode_call(
        self,
        call: ProviderCapabilityCall,
        *,
        provider_family: Enum | str | None = None,
    ) -> DomainCapabilityCall | TaskInputCapabilityCall:
        """Decode and validate one provider-native structured call."""
        if not isinstance(call, ProviderCapabilityCall):
            raise ModelCapabilityValidationError(
                "capability.call_type",
                "call must be a provider capability call",
            )
        if not call.structured:
            raise ModelCapabilityValidationError(
                "capability.non_structured_call",
                "capability calls must originate from a structured frame",
            )
        projection = self.project(provider_family)
        canonical_name = projection.canonical_name(call.provider_name)
        descriptor = next(
            descriptor
            for descriptor in self.descriptors
            if descriptor.canonical_name == canonical_name
        )
        maximum_bytes = (
            _MAX_INPUT_ARGUMENT_UTF8_BYTES
            if canonical_name == RESERVED_INPUT_CAPABILITY_NAME
            else _MAX_ARGUMENT_UTF8_BYTES
        )
        maximum_depth = (
            _MAX_INPUT_ARGUMENT_DEPTH
            if canonical_name == RESERVED_INPUT_CAPABILITY_NAME
            else _MAX_ARGUMENT_DEPTH
        )
        arguments = _decode_arguments(
            call.arguments,
            maximum_bytes=maximum_bytes,
            maximum_depth=maximum_depth,
        )
        _validate_schema_instance(descriptor.parameter_schema, arguments)
        if canonical_name == RESERVED_INPUT_CAPABILITY_NAME:
            return self._decode_task_input(call, arguments)
        return ToolCall(
            id=call.call_id,
            name=canonical_name,
            arguments=cast(dict[str, ToolValue], arguments),
            provider_name=call.provider_name,
            provider_name_encoded=(call.provider_name != canonical_name),
        )

    def classify_batch(
        self,
        calls: Sequence[ProviderCapabilityCall],
        *,
        provider_family: Enum | str | None = None,
    ) -> CapabilityBatchClassification:
        """Classify a complete provider batch before any domain execution."""
        assert isinstance(calls, Sequence)
        projection = self.project(provider_family)
        canonical_names: list[str] = []
        try:
            for call in calls:
                if not isinstance(call, ProviderCapabilityCall):
                    raise ModelCapabilityValidationError(
                        "capability.call_type",
                        "batch contains an invalid call",
                    )
                canonical_names.append(
                    projection.canonical_name(call.provider_name)
                )
        except ModelCapabilityValidationError:
            return CapabilityBatchRejected(
                code=CapabilityBatchRejectionCode.UNKNOWN_CAPABILITY,
                message="Batch contains an unadvertised capability.",
            )
        input_count = canonical_names.count(RESERVED_INPUT_CAPABILITY_NAME)
        if input_count > 1:
            return CapabilityBatchRejected(
                code=CapabilityBatchRejectionCode.MULTIPLE_TASK_INPUT_CALLS,
                message="A response cannot contain multiple task-input calls.",
            )
        if input_count == 1 and len(calls) != 1:
            return CapabilityBatchRejected(
                code=CapabilityBatchRejectionCode.MIXED_TASK_INPUT_BATCH,
                message="Task input cannot be mixed with domain calls.",
            )
        try:
            decoded = tuple(
                self.decode_call(call, provider_family=provider_family)
                for call in calls
            )
        except ModelCapabilityValidationError as exc:
            code = CapabilityBatchRejectionCode.MALFORMED_CALL
            if exc.code == "capability.missing_call_id":
                code = CapabilityBatchRejectionCode.MISSING_CALL_ID
            elif exc.code == "capability.non_structured_call":
                code = CapabilityBatchRejectionCode.NON_STRUCTURED_CALL
            return CapabilityBatchRejected(
                code=code,
                message="Batch contains an invalid capability call.",
            )
        if input_count == 1:
            task_input = decoded[0]
            assert isinstance(task_input, TaskInputCapabilityCall)
            return CapabilityBatchAccepted(task_input=task_input)
        return CapabilityBatchAccepted(
            domain_calls=cast(tuple[ToolCall, ...], decoded)
        )

    def project_result(
        self,
        call: TaskInputCapabilityCall,
        result: InputModelResult,
    ) -> CorrelatedCapabilityResult:
        """Project a result using the exact originating provider call ID."""
        if not isinstance(call, TaskInputCapabilityCall):
            raise ModelCapabilityValidationError(
                "capability.result_correlation",
                "result requires a task-input capability call",
            )
        try:
            payload = encode_input_model_result(result)
        except InputContractError as exc:
            raise ModelCapabilityValidationError(
                "capability.result", "task-input result is invalid"
            ) from exc
        return CorrelatedCapabilityResult(
            call_id=call.call_id,
            canonical_name=call.canonical_name,
            provider_name=call.provider_name,
            payload=cast(Mapping[str, JsonValue], payload),
        )

    def _canonical_provider_originated_call(self, call: ToolCall) -> ToolCall:
        provider_name = call.provider_name or call.name
        canonical_names = {
            descriptor.canonical_name for descriptor in self.descriptors
        }
        aliases = {
            alias: descriptor.canonical_name
            for descriptor in self.descriptors
            for alias in descriptor.aliases
        }
        try:
            policy = ToolNamePolicy(
                settings=self.domain_seed.name_policy_settings()
            ).for_provider("local")
            local_policy = policy.bind(canonical_names)
            harmony_name = f"functions.{provider_name}"
            if provider_name in aliases:
                canonical_name = aliases[provider_name]
            elif (
                provider_name in canonical_names
                and local_policy.provider_name(provider_name) == provider_name
            ):
                canonical_name = provider_name
            elif (
                self.tool_format is ToolFormat.HARMONY
                and harmony_name in canonical_names
            ):
                canonical_name = harmony_name
            else:
                canonical_name = local_policy.canonical_name(provider_name)
                if (
                    canonical_name not in canonical_names
                    or local_policy.provider_name(canonical_name)
                    != provider_name
                ):
                    raise AssertionError(
                        "provider capability is not advertised"
                    )
        except AssertionError:
            return ToolCall(
                id=call.id,
                name="",
                arguments=call.arguments,
                provider_name=provider_name,
                provider_name_encoded=call.provider_name_encoded,
                provider_arguments_malformed=(
                    call.provider_arguments_malformed
                ),
            )
        if canonical_name == call.name:
            return call
        return ToolCall(
            id=call.id,
            name=canonical_name,
            arguments=call.arguments,
            provider_name=provider_name,
            provider_name_encoded=(provider_name != canonical_name),
            provider_arguments_malformed=call.provider_arguments_malformed,
        )

    def _decode_task_input(
        self,
        call: ProviderCapabilityCall,
        arguments: dict[str, object],
    ) -> TaskInputCapabilityCall:
        if call.call_id is None:
            raise ModelCapabilityValidationError(
                "capability.missing_call_id",
                "task-input calls require a provider call ID",
            )
        try:
            mode = RequirementMode(cast(str, arguments["mode"]))
            reason = validate_presentation_text(
                arguments["reason"],
                "reason",
                minimum=1,
                maximum=500,
                maximum_bytes=2_000,
            )
            questions = tuple(
                decode_input_question(question)
                for question in cast(list[object], arguments["questions"])
            )
        except (InputContractError, KeyError, TypeError, ValueError) as exc:
            raise ModelCapabilityValidationError(
                "capability.arguments", "task-input arguments are invalid"
            ) from exc
        return TaskInputCapabilityCall(
            call_id=call.call_id,
            provider_name=call.provider_name,
            arguments=cast(Mapping[str, JsonValue], arguments),
            mode=mode,
            reason=reason,
            questions=questions,
            advertisement=self.task_input_advertisement,
        )


def _decode_domain_descriptor(
    value: object, index: int
) -> ModelCapabilityDescriptor:
    item = _object(value, f"seed.descriptors[{index}]")
    _exact_keys(
        item,
        {
            "canonical_name",
            "description",
            "aliases",
            "parameter_schema",
            "result_schema",
        },
        f"seed.descriptors[{index}]",
    )
    name = item["canonical_name"]
    description = item["description"]
    aliases = item["aliases"]
    if not isinstance(name, str) or not isinstance(description, str):
        raise ModelCapabilityValidationError(
            "capability.seed",
            "descriptor name and description must be strings",
        )
    if not isinstance(aliases, list | tuple) or not all(
        isinstance(alias, str) for alias in aliases
    ):
        raise ModelCapabilityValidationError(
            "capability.seed", "descriptor aliases must be strings"
        )
    parameter_schema = _object(
        item["parameter_schema"],
        f"seed.descriptors[{index}].parameter_schema",
    )
    raw_result_schema = item["result_schema"]
    result_schema = (
        None
        if raw_result_schema is None
        else _object(
            raw_result_schema,
            f"seed.descriptors[{index}].result_schema",
        )
    )
    return ModelCapabilityDescriptor(
        canonical_name=name,
        description=description,
        aliases=tuple(aliases),
        parameter_schema=cast(Mapping[str, JsonValue], parameter_schema),
        result_schema=cast(Mapping[str, JsonValue] | None, result_schema),
    )


def _task_input_descriptor() -> ModelCapabilityDescriptor:
    choice_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["value", "label"],
        "properties": {
            "value": {"type": "string", "minLength": 1, "maxLength": 128},
            "label": {"type": "string", "minLength": 1, "maxLength": 80},
            "description": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
        },
    }
    question_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "question_id",
            "kind",
            "prompt",
            "required",
            "choices",
            "allow_other",
        ],
        "properties": {
            "question_id": {
                "type": "string",
                "pattern": "^[A-Za-z][A-Za-z0-9._-]{0,63}$",
            },
            "kind": {
                "enum": [
                    "confirmation",
                    "text",
                    "multiline_text",
                    "single_selection",
                    "multiple_selection",
                ]
            },
            "header": {"type": "string", "minLength": 1, "maxLength": 40},
            "prompt": {"type": "string", "minLength": 1, "maxLength": 500},
            "help": {"type": "string", "minLength": 1, "maxLength": 1_000},
            "required": {"type": "boolean"},
            "choices": {
                "type": "array",
                "maxItems": 20,
                "items": choice_schema,
            },
            "allow_other": {"type": "boolean"},
            "recommended_choice": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
            },
            "default_value": {
                "oneOf": [
                    {"type": "boolean"},
                    {"type": "string"},
                    {
                        "type": "array",
                        "uniqueItems": True,
                        "maxItems": 20,
                        "items": {"type": "string"},
                    },
                ]
            },
            "constraints": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "minimum_length": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 65_536,
                    },
                    "maximum_length": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 65_536,
                    },
                    "minimum": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 20,
                    },
                    "maximum": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 20,
                    },
                },
            },
            "presentation_hint": {
                "enum": [
                    "compact",
                    "expanded",
                    "radio",
                    "list",
                    "checkbox",
                    "single_line",
                    "editor",
                ]
            },
        },
    }
    parameter_schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["mode", "reason", "questions"],
        "properties": {
            "mode": {"enum": ["required", "advisory"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 500},
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": question_schema,
            },
        },
    }
    return ModelCapabilityDescriptor(
        canonical_name=RESERVED_INPUT_CAPABILITY_NAME,
        description=(
            "Request bounded structured task input from the user without "
            "approving actions or requesting secrets."
        ),
        parameter_schema=cast(Mapping[str, JsonValue], parameter_schema),
        kind=ModelCapabilityKind.TASK_INPUT,
    )


def _provider_schema(
    descriptor: ModelCapabilityDescriptor, provider_name: str
) -> Mapping[str, JsonValue]:
    parameters = cast(
        dict[str, object], _thaw_json(descriptor.parameter_schema)
    )
    for keyword in ("anyOf", "oneOf", "allOf", "not"):
        parameters.pop(keyword, None)
    parameters.setdefault("type", "object")
    return _freeze_json_object(
        {
            "type": "function",
            "function": {
                "name": provider_name,
                "description": descriptor.description,
                "parameters": parameters,
            },
        },
        "provider_schema",
    )


def _decode_arguments(
    value: str | Mapping[str, object] | None,
    *,
    maximum_bytes: int,
    maximum_depth: int,
) -> dict[str, object]:
    assert maximum_bytes > 0
    assert maximum_depth > 0
    if value is None:
        decoded: object = {}
    elif isinstance(value, str):
        try:
            encoded_value = value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ModelCapabilityValidationError(
                "capability.arguments_json",
                "capability arguments are not strict JSON",
            ) from exc
        if len(encoded_value) > maximum_bytes:
            raise ModelCapabilityValidationError(
                "capability.arguments_size",
                "capability arguments are too large",
            )
        _validate_json_text_depth(value, maximum_depth=maximum_depth)
        try:
            decoded = loads(
                value,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except RecursionError as exc:
            raise ModelCapabilityValidationError(
                "capability.arguments_depth",
                "capability arguments are too deeply nested",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ModelCapabilityValidationError(
                "capability.arguments_json",
                "capability arguments are not strict JSON",
            ) from exc
    else:
        decoded = _strict_json_copy(
            value,
            "arguments",
            maximum_depth=maximum_depth,
        )
    if not isinstance(decoded, dict):
        raise ModelCapabilityValidationError(
            "capability.arguments_type",
            "capability arguments must be an object",
        )
    try:
        encoded = dumps(
            decoded,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except RecursionError as exc:
        raise ModelCapabilityValidationError(
            "capability.arguments_depth",
            "capability arguments are too deeply nested",
        ) from exc
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ModelCapabilityValidationError(
            "capability.arguments_json",
            "capability arguments are not strict JSON",
        ) from exc
    if len(encoded) > maximum_bytes:
        raise ModelCapabilityValidationError(
            "capability.arguments_size", "capability arguments are too large"
        )
    return cast(dict[str, object], decoded)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _validate_json_text_depth(value: str, *, maximum_depth: int) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum_depth:
                raise ModelCapabilityValidationError(
                    "capability.arguments_depth",
                    "capability arguments are too deeply nested",
                )
        elif character in "]}":
            depth = max(0, depth - 1)


def _strict_json_copy(
    value: object,
    path: str,
    *,
    maximum_depth: int,
    depth: int = 0,
) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ModelCapabilityValidationError(
                "capability.arguments_json",
                f"{path} contains a non-finite number",
            )
        return value
    if isinstance(value, list):
        container_depth = depth + 1
        if container_depth > maximum_depth:
            raise ModelCapabilityValidationError(
                "capability.arguments_depth",
                "capability arguments are too deeply nested",
            )
        return [
            _strict_json_copy(
                item,
                f"{path}[{index}]",
                maximum_depth=maximum_depth,
                depth=container_depth,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        container_depth = depth + 1
        if container_depth > maximum_depth:
            raise ModelCapabilityValidationError(
                "capability.arguments_depth",
                "capability arguments are too deeply nested",
            )
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ModelCapabilityValidationError(
                    "capability.arguments_json", f"{path} has a non-string key"
                )
            result[key] = _strict_json_copy(
                item,
                f"{path}.{key}",
                maximum_depth=maximum_depth,
                depth=container_depth,
            )
        return result
    raise ModelCapabilityValidationError(
        "capability.arguments_json", f"{path} contains a non-JSON value"
    )


def _validate_schema_instance(
    schema: Mapping[str, JsonValue], instance: dict[str, object]
) -> None:
    try:
        module = import_module("jsonschema")
        validator_class = getattr(module, "Draft202012Validator")
        validator_class.check_schema(_thaw_json(schema))
        validator = validator_class(_thaw_json(schema))
        if next(iter(validator.iter_errors(instance)), None) is not None:
            raise ModelCapabilityValidationError(
                "capability.schema_validation",
                "capability arguments do not match the advertised schema",
            )
    except ModelCapabilityValidationError:
        raise
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise ModelCapabilityValidationError(
            "capability.schema_unavailable",
            "strict capability schema validation is unavailable",
        ) from exc


def _copy_model_capability_json(value: object, path: str) -> object:
    """Return a bounded mutable copy of one model capability JSON value."""
    return _thaw_json(_freeze_json(value, path))


def _freeze_json_object(
    value: Mapping[str, object] | Mapping[str, JsonValue], path: str
) -> Mapping[str, JsonValue]:
    frozen = _freeze_json(value, path)
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_json(value: object, path: str, *, depth: int = 0) -> JsonValue:
    if value is None or type(value) in (bool, int, str):
        return cast(JsonValue, value)
    if type(value) is float:
        if not isfinite(value):
            raise ModelCapabilityValidationError(
                "capability.non_json", f"{path} contains a non-finite number"
            )
        return value
    if isinstance(value, list | tuple):
        container_depth = depth + 1
        if container_depth > _MAX_FROZEN_JSON_DEPTH:
            raise ModelCapabilityValidationError(
                "capability.non_json", f"{path} is too deeply nested"
            )
        return tuple(
            _freeze_json(
                item,
                f"{path}[{index}]",
                depth=container_depth,
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        container_depth = depth + 1
        if container_depth > _MAX_FROZEN_JSON_DEPTH:
            raise ModelCapabilityValidationError(
                "capability.non_json", f"{path} is too deeply nested"
            )
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ModelCapabilityValidationError(
                    "capability.non_json", f"{path} has a non-string key"
                )
            frozen[key] = _freeze_json(
                item,
                f"{path}.{key}",
                depth=container_depth,
            )
        return MappingProxyType(frozen)
    raise ModelCapabilityValidationError(
        "capability.non_json", f"{path} contains a non-JSON value"
    )


def _thaw_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ModelCapabilityValidationError(
            "capability.seed", f"{path} must be an object"
        )
    return dict(cast(Mapping[str, object], value))


def _exact_keys(
    value: Mapping[str, object], expected: set[str], path: str
) -> None:
    if set(value) != expected:
        raise ModelCapabilityValidationError(
            "capability.seed", f"{path} has invalid fields"
        )


def _assert_name(value: object, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
