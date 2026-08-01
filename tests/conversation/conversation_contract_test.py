"""Verify frozen conversation state and storage contracts."""

from asyncio import gather
from collections.abc import Callable
from datetime import datetime
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from typing import cast

import pytest
from conversation_fixtures import (
    AsyncConversationBarrier,
    ConversationFaultInjector,
    ConversationIdFactory,
    InjectedConversationFault,
    ManualConversationClock,
    TestResponseResource,
    fixture_authority_scope,
)

from avalan.conversation import (
    CHECKPOINT_COMMIT_TRANSITIONS,
    CHECKPOINT_VISIBILITY,
    CONFIGURATION_PRECEDENCE,
    CONVERSATION_CONTRACT_VERSION,
    FAILURE_FENCES,
    LOCAL_DELETION_TRANSITIONS,
    PUBLIC_RESPONSE_ID_TRANSITIONS,
    RESPONSE_OPERATION_POLICY,
    RESPONSE_RESOURCE_TRANSITIONS,
    UPSTREAM_DELETION_TRANSITIONS,
    AuthorityEndpointId,
    AuthorityPrincipalId,
    AuthorityScope,
    AuthoritySource,
    AuthorityTenantId,
    CanonicalRequestDigest,
    CheckpointCommitState,
    CheckpointId,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointSequence,
    CheckpointVisibility,
    ChildLaneRetentionPolicy,
    ConfigurationSource,
    ContinuationDigest,
    ConversationAgentId,
    ConversationBranchId,
    ConversationId,
    ConversationOperation,
    ConversationSurface,
    ExecutionSegmentId,
    FailureBoundary,
    FailureFence,
    IdempotencyDisposition,
    IdempotencyRecord,
    IdempotencyRecordState,
    LocalDeletionState,
    LocalResponseStorage,
    LogicalTurnId,
    MigrationDisposition,
    NamedHeadAdvanceDisposition,
    NamedHeadRevision,
    ParentAdvanceMode,
    PortableContinuationReference,
    ProviderLaneOwnerKind,
    ProviderLaneStorage,
    PublicResponseIdState,
    PublicResponseMappingState,
    RequestIdempotencyIdentity,
    RequestIdempotencyKey,
    ResponseOperation,
    ResponseOperationDisposition,
    ResponseResourceState,
    ResponseStorageContext,
    RetentionLimits,
    RetryRule,
    StoragePolicy,
    StructuredInputContinuationId,
    SurfaceDisposition,
    UpstreamDeletionState,
    UpstreamLifetimeStatus,
    capability_revision,
    idempotency_disposition,
    named_head_advance_disposition,
    response_operation_disposition,
    response_transition_allowed,
    terminal_publication_allowed,
)
from avalan.interaction import (
    CapabilityRevision,
    ContinuationId,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ModelConfigRevision,
    ModelId,
    ProviderConfigRevision,
    ProviderFamilyName,
    StateRevision,
)

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _ROOT / "tests" / "fixtures" / "conversation"
_CONTRACT_DECISIONS_DIGEST = (
    "efe0ea3422048d7f78819402b87f31a89c25f855090fbb871b7140a69c1f5416"
)
_DETERMINISTIC_FIXTURES_DIGEST = (
    "39a1bd331f63954425516b0cfb1437ad9934feeaaf550121486924dc0caf7547"
)


@pytest.fixture
def anyio_backend() -> str:
    """Run async contract fixtures on their native asyncio backend."""
    return "asyncio"


def _json_object(path: Path) -> dict[str, object]:
    decoded: object = loads(path.read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    assert all(isinstance(key, str) for key in decoded)
    return cast(dict[str, object], decoded)


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


def _string(value: object) -> str:
    assert isinstance(value, str)
    return value


def _integer(value: object) -> int:
    assert type(value) is int
    return value


def _boolean(value: object) -> bool:
    assert type(value) is bool
    return value


def _definition(
    capability: str = "capability-revision-1",
) -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="agent.toml",
        agent_definition_revision="agent-revision-1",
        operation_id="operation-1",
        operation_index=0,
        model_config_reference="model-config",
        tool_revision="tool-revision-1",
        capability_revision=capability,
    )


def _binding(
    capability: str = "capability-revision-1",
) -> ContinuationRevisionBinding:
    return ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("synthetic"),
        model_id=ModelId("model-1"),
        provider_config_revision=ProviderConfigRevision("provider-revision-1"),
        model_config_revision=ModelConfigRevision("model-revision-1"),
        capability_revision=CapabilityRevision(capability),
    )


def _idempotency_identity(
    *,
    authority: AuthorityScope | None = None,
    operation: ConversationOperation = ConversationOperation.CONTINUE,
    key: str = "request-key-1",
    digest: str = "digest-1",
) -> RequestIdempotencyIdentity:
    return RequestIdempotencyIdentity(
        authority=authority or fixture_authority_scope(),
        operation=operation,
        key=RequestIdempotencyKey(key),
        request_digest=CanonicalRequestDigest(digest),
    )


def _storage_context(
    *,
    local: LocalResponseStorage = LocalResponseStorage.PROCESS_LOCAL,
    upstream: ProviderLaneStorage = ProviderLaneStorage.STATELESS,
    public_mapping: PublicResponseMappingState = (
        PublicResponseMappingState.ADDRESSABLE
    ),
) -> ResponseStorageContext:
    return ResponseStorageContext(
        policy=StoragePolicy(
            local=local,
            upstream=upstream,
            provider_storage_disclosed=(
                upstream is ProviderLaneStorage.STORED
            ),
        ),
        public_mapping=public_mapping,
    )


def _public_mapping_allowed(
    local: LocalResponseStorage,
    mapping: PublicResponseMappingState,
) -> bool:
    """Return whether one local store can own the requested mapping."""
    match local:
        case LocalResponseStorage.NONE:
            return mapping is PublicResponseMappingState.ABSENT
        case LocalResponseStorage.PROCESS_LOCAL | LocalResponseStorage.DURABLE:
            return mapping in (
                PublicResponseMappingState.ABSENT,
                PublicResponseMappingState.ADDRESSABLE,
                PublicResponseMappingState.TOMBSTONED,
            )
        case LocalResponseStorage.TRANSIENT:
            return mapping in (
                PublicResponseMappingState.ABSENT,
                PublicResponseMappingState.PRIVATE_TRANSIENT,
            )


def _expected_response_operation_disposition(
    state: ResponseResourceState,
    operation: ResponseOperation,
    public_id_state: PublicResponseIdState,
    local: LocalResponseStorage,
    upstream: ProviderLaneStorage,
    mapping: PublicResponseMappingState,
) -> ResponseOperationDisposition:
    """Derive a disposition independently from implementation tables."""
    assert _public_mapping_allowed(local, mapping)
    if public_id_state is not PublicResponseIdState.COMMITTED:
        return ResponseOperationDisposition.NOT_ADDRESSABLE
    concealed_state = state in (
        ResponseResourceState.TOMBSTONED,
        ResponseResourceState.DELETED,
        ResponseResourceState.EXPIRED,
    )
    if concealed_state or mapping is PublicResponseMappingState.TOMBSTONED:
        return ResponseOperationDisposition.CONCEALED
    if mapping is not PublicResponseMappingState.ADDRESSABLE:
        return ResponseOperationDisposition.NOT_ADDRESSABLE
    if state in (
        ResponseResourceState.ALLOCATED,
        ResponseResourceState.DISPATCHING,
        ResponseResourceState.STREAMING,
        ResponseResourceState.COMMITTING,
    ):
        return ResponseOperationDisposition.NOT_ADDRESSABLE
    if state is ResponseResourceState.INPUT_REQUIRED:
        if operation in (ResponseOperation.RETRIEVE, ResponseOperation.DELETE):
            return ResponseOperationDisposition.ALLOWED
        if operation is ResponseOperation.CONTINUE:
            if upstream is ProviderLaneStorage.OFF:
                return ResponseOperationDisposition.DENIED_STATE
            return ResponseOperationDisposition.STRUCTURED_INPUT_ONLY
        return ResponseOperationDisposition.DENIED_STATE
    if state is ResponseResourceState.COMPLETED:
        if operation in (ResponseOperation.RETRIEVE, ResponseOperation.DELETE):
            return ResponseOperationDisposition.ALLOWED
        if operation is ResponseOperation.CONTINUE:
            if upstream is ProviderLaneStorage.OFF:
                return ResponseOperationDisposition.DENIED_STATE
            return ResponseOperationDisposition.ALLOWED
        if upstream is ProviderLaneStorage.STATELESS:
            return ResponseOperationDisposition.ALLOWED
        return ResponseOperationDisposition.DENIED_STATE
    assert state is ResponseResourceState.FAILED
    if operation in (ResponseOperation.RETRIEVE, ResponseOperation.DELETE):
        return ResponseOperationDisposition.ALLOWED
    return ResponseOperationDisposition.DENIED_STATE


def test_contract_fixture_freezes_every_phase_zero_decision_group() -> None:
    """Keep the domain contract versioned, dormant, and explicit."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    canonical = dumps(
        fixture,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert sha256(canonical).hexdigest() == _CONTRACT_DECISIONS_DIGEST
    assert set(fixture) == {
        "schema_version",
        "contract_version",
        "feature",
        "owner",
        "activation",
        "identity",
        "checkpoint",
        "public_response_id",
        "response_resource",
        "provider_lane_binding",
        "branching",
        "idempotency",
        "failure_fences",
        "failure_fence_tuple_fields",
        "atomic_boundaries",
        "storage",
        "retention",
        "descendants",
        "deletion",
        "migration",
        "configuration",
        "authority",
        "surfaces",
    }
    assert fixture["schema_version"] == 1
    assert fixture["contract_version"] == CONVERSATION_CONTRACT_VERSION
    assert fixture["feature"] == "conversation_continuity"
    assert fixture["owner"] == "conversation_domain_state_storage"
    assert fixture["activation"] == "dormant"

    identities = _object(_object(fixture["identity"])["types"])
    assert set(identities) == {
        "conversation_id",
        "logical_turn_id",
        "execution_segment_id",
        "checkpoint_id",
        "branch_id",
        "named_head_id",
        "provider_lane_id",
        "model_call_id",
        "public_response_id",
        "provisional_response_id",
        "upstream_response_id",
        "task_id",
        "agent_id",
        "structured_input_continuation_id",
    }
    surfaces = _object(fixture["surfaces"])
    expected_surface_names = {
        "direct_model_sdk",
        "agent_sdk",
        "cli",
        "flow",
        "mcp",
        "a2a",
        "served_responses",
    }
    assert {surface.value for surface in ConversationSurface} == (
        expected_surface_names
    )
    assert set(surfaces) == expected_surface_names | {
        "disposition_values",
        "legacy_cli_conversation",
        "legacy_cli_session",
        "deferred_field_policy",
    }
    disposition_values = [
        _string(value) for value in _list(surfaces["disposition_values"])
    ]
    assert disposition_values == [
        SurfaceDisposition.ACTIVATED.value,
        SurfaceDisposition.DEFERRED.value,
        SurfaceDisposition.INCAPABLE.value,
    ]
    assert disposition_values == ["activated", "deferred", "incapable"]
    assert len(disposition_values) == len(SurfaceDisposition)
    assert set(disposition_values) == {
        disposition.value for disposition in SurfaceDisposition
    }
    expected_surfaces = {
        ConversationSurface.DIRECT_MODEL_SDK: {
            "initial_release": SurfaceDisposition.ACTIVATED.value,
            "phase_0": "dormant",
        },
        ConversationSurface.AGENT_SDK: {
            "initial_release": SurfaceDisposition.ACTIVATED.value,
            "phase_0": "dormant",
        },
        ConversationSurface.CLI: {
            "initial_release": SurfaceDisposition.DEFERRED.value,
            "phase_0": "dormant",
        },
        ConversationSurface.FLOW: {
            "initial_release": SurfaceDisposition.DEFERRED.value,
            "phase_0": "dormant",
        },
        ConversationSurface.MCP: {
            "initial_release": SurfaceDisposition.DEFERRED.value,
            "phase_0": "dormant",
        },
        ConversationSurface.A2A: {
            "initial_release": SurfaceDisposition.DEFERRED.value,
            "phase_0": "dormant",
        },
        ConversationSurface.SERVED_RESPONSES: {
            "initial_release": SurfaceDisposition.ACTIVATED.value,
            "phase_0": "dormant",
        },
    }
    assert set(expected_surfaces) == set(ConversationSurface)
    assert {
        surface: _object(surfaces[surface.value])
        for surface in ConversationSurface
    } == expected_surfaces
    assert (
        surfaces["legacy_cli_conversation"]
        == "visible repeated-message loop only"
    )
    assert surfaces["legacy_cli_session"] == "message memory only"
    assert (
        surfaces["deferred_field_policy"]
        == "omit unsupported stateful fields or reject them before dispatch; "
        "never accept and discard"
    )


def test_provider_binding_fixture_is_generic_versioned_and_secret_free() -> (
    None
):
    """Freeze lane-binding fields without embedding provider credentials."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    binding = _object(fixture["provider_lane_binding"])
    assert binding["version"] == 1
    assert _list(binding["required"]) == [
        "provider_family",
        "normalized_endpoint",
        "model_or_deployment",
        "provider_api_revision",
        "sdk_revision",
        "model_configuration_revision",
        "capability_profile_revision",
        "tool_schema_revision",
        "execution_definition_revision",
        "continuation_codec_version",
    ]
    normalization = _object(binding["normalization"])
    prohibited = {_string(item) for item in _list(normalization["prohibited"])}
    assert {"credentials", "query", "api_key"} <= prohibited
    lane_identity = _object(binding["lane_identity"])
    assert lane_identity["version"] == 1
    assert lane_identity["prefix"] == "lane-v1-"
    assert lane_identity["digest"] == "sha256"
    assert _list(lane_identity["components"]) == [
        "contract_version",
        "conversation_id",
        "owner_kind",
        "agent_topology_path",
        "model_slot",
        "provider_family",
        "normalized_endpoint",
        "azure_resource_identity_or_empty",
        "model_or_deployment",
        "model_configuration_revision",
        "execution_definition_revision",
    ]
    assert set(_object(lane_identity["owner_paths"])) == {
        kind.value for kind in ProviderLaneOwnerKind
    }
    assert set(_list(lane_identity["prohibited_components"])) == {
        "credential",
        "query_secret",
        "upstream_response_id",
        "process_counter",
        "random_nonce",
    }
    child_retention = _object(binding["child_lane_retention"])
    assert child_retention == {
        "suspended": ChildLaneRetentionPolicy.RETAIN.value,
        "explicitly_resumable": ChildLaneRetentionPolicy.RETAIN.value,
        "terminal": ChildLaneRetentionPolicy.DISCARD_TERMINAL.value,
        "default": ChildLaneRetentionPolicy.DISCARD_TERMINAL.value,
        "recorded_in_checkpoint": True,
    }
    serialized = (_FIXTURES / "contract_decisions.json").read_text(
        encoding="utf-8"
    )
    assert "sk-" not in serialized
    assert "api-version=" not in serialized


def test_branch_migration_and_authority_policies_use_closed_values() -> None:
    """Keep owned policy decisions closed over their typed contract enums."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    branching = _object(fixture["branching"])
    assert set(branching) == {mode.value for mode in ParentAdvanceMode}

    migration = _object(fixture["migration"])
    assert migration["structured_input_records"] == (
        MigrationDisposition.REFERENCE_EXISTING.value
    )
    assert migration["legacy_replay_snapshots"] == (
        MigrationDisposition.VERSIONED_MIGRATION.value
    )
    assert migration["n_and_n_plus_1"] == (
        MigrationDisposition.COMPATIBLE_READ.value
    )
    assert migration["rollback_after_authority_cutover"] == (
        MigrationDisposition.REJECT_ROLLBACK.value
    )

    authority = _object(fixture["authority"])
    assert (
        authority["direct_sdk"] == AuthoritySource.TRUSTED_HOST_CONTEXT.value
    )
    assert authority["embedded"] == AuthoritySource.TRUSTED_HOST_CONTEXT.value
    assert authority["server"] == (
        AuthoritySource.AUTHENTICATED_SERVER_CONTEXT.value
    )
    assert authority["authenticated_server_requires_tenant"] is True
    assert authority["fixed_local_requires_no_tenant"] is True
    assert authority["authority_source_required"] is True


def test_checkpoint_tables_exactly_match_the_tracked_contract() -> None:
    """Keep checkpoint kinds and commit transitions total and deterministic."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    checkpoint = _object(fixture["checkpoint"])
    expected_kinds = {
        name: _string(visibility)
        for name, visibility in _object(checkpoint["kinds"]).items()
    }
    assert expected_kinds == {
        kind.value: CHECKPOINT_VISIBILITY[kind].value
        for kind in CheckpointKind
    }
    assert set(CHECKPOINT_VISIBILITY) == set(CheckpointKind)

    expected_transitions = {
        name: {_string(item) for item in _list(targets)}
        for name, targets in _object(checkpoint["commit_transitions"]).items()
    }
    assert expected_transitions == {
        state.value: {target.value for target in targets}
        for state, targets in CHECKPOINT_COMMIT_TRANSITIONS.items()
    }
    assert set(CHECKPOINT_COMMIT_TRANSITIONS) == set(CheckpointCommitState)


def test_response_tables_exactly_match_the_tracked_contract() -> None:
    """Keep every resource transition and operation disposition frozen."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    response = _object(fixture["response_resource"])
    expected_transitions = {
        name: {_string(item) for item in _list(targets)}
        for name, targets in _object(response["transitions"]).items()
    }
    assert expected_transitions == {
        state.value: {target.value for target in targets}
        for state, targets in RESPONSE_RESOURCE_TRANSITIONS.items()
    }
    assert set(RESPONSE_RESOURCE_TRANSITIONS) == set(ResponseResourceState)
    for source in ResponseResourceState:
        for target in ResponseResourceState:
            assert response_transition_allowed(source, target) is (
                target.value in expected_transitions[source.value]
            )

    expected_operations = {
        state_name: {
            operation_name: _string(disposition)
            for operation_name, disposition in _object(operations).items()
        }
        for state_name, operations in _object(
            response["state_operations_for_addressable_id"]
        ).items()
    }
    assert expected_operations == {
        state.value: {
            operation.value: disposition.value
            for operation, disposition in operations.items()
        }
        for state, operations in RESPONSE_OPERATION_POLICY.items()
    }
    assert set(RESPONSE_OPERATION_POLICY) == set(ResponseResourceState)
    addressable = _storage_context()
    for state in ResponseResourceState:
        assert set(RESPONSE_OPERATION_POLICY[state]) == set(ResponseOperation)
        for operation in ResponseOperation:
            assert (
                response_operation_disposition(
                    state,
                    operation,
                    PublicResponseIdState.COMMITTED,
                    addressable,
                )
                is RESPONSE_OPERATION_POLICY[state][operation]
            )
            assert (
                response_operation_disposition(
                    state,
                    operation,
                    PublicResponseIdState.PROVISIONAL,
                    addressable,
                )
                is ResponseOperationDisposition.NOT_ADDRESSABLE
            )
            assert (
                response_operation_disposition(
                    state,
                    operation,
                    PublicResponseIdState.WITHHELD,
                    addressable,
                )
                is ResponseOperationDisposition.NOT_ADDRESSABLE
            )


def test_response_addressability_crosses_independent_storage_axes() -> None:
    """Cross every response axis against an independent policy oracle."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    response = _object(fixture["response_resource"])
    expected_mappings = {
        name: {_string(item) for item in _list(states)}
        for name, states in _object(
            response["permitted_mappings_by_local_storage"]
        ).items()
    }
    assert expected_mappings == {
        LocalResponseStorage.NONE.value: {
            PublicResponseMappingState.ABSENT.value
        },
        LocalResponseStorage.PROCESS_LOCAL.value: {
            PublicResponseMappingState.ABSENT.value,
            PublicResponseMappingState.ADDRESSABLE.value,
            PublicResponseMappingState.TOMBSTONED.value,
        },
        LocalResponseStorage.TRANSIENT.value: {
            PublicResponseMappingState.ABSENT.value,
            PublicResponseMappingState.PRIVATE_TRANSIENT.value,
        },
        LocalResponseStorage.DURABLE.value: {
            PublicResponseMappingState.ABSENT.value,
            PublicResponseMappingState.ADDRESSABLE.value,
            PublicResponseMappingState.TOMBSTONED.value,
        },
    }
    assert set(_list(response["mapping_states"])) == {
        state.value for state in PublicResponseMappingState
    }
    assert _object(response["provider_mode_operation_overrides"]) == {
        ProviderLaneStorage.OFF.value: {
            ResponseOperation.CONTINUE.value: (
                ResponseOperationDisposition.DENIED_STATE.value
            ),
            ResponseOperation.COMPACT.value: (
                ResponseOperationDisposition.DENIED_STATE.value
            ),
        },
        ProviderLaneStorage.STATELESS.value: {
            ResponseOperation.CONTINUE.value: "state_policy",
            ResponseOperation.COMPACT.value: "state_policy",
        },
        ProviderLaneStorage.STORED.value: {
            ResponseOperation.CONTINUE.value: "state_policy",
            ResponseOperation.COMPACT.value: (
                ResponseOperationDisposition.DENIED_STATE.value
            ),
        },
    }

    valid_context_count = 0
    invalid_context_count = 0
    evaluated_disposition_count = 0
    for local in LocalResponseStorage:
        for upstream in ProviderLaneStorage:
            for mapping in PublicResponseMappingState:
                allowed_mapping = _public_mapping_allowed(local, mapping)
                assert allowed_mapping is (
                    mapping.value in expected_mappings[local.value]
                )
                if not allowed_mapping:
                    with pytest.raises(AssertionError):
                        _storage_context(
                            local=local,
                            upstream=upstream,
                            public_mapping=mapping,
                        )
                    invalid_context_count += 1
                    continue
                storage = _storage_context(
                    local=local,
                    upstream=upstream,
                    public_mapping=mapping,
                )
                valid_context_count += 1
                for state in ResponseResourceState:
                    for operation in ResponseOperation:
                        for public_id_state in PublicResponseIdState:
                            expected = (
                                _expected_response_operation_disposition(
                                    state,
                                    operation,
                                    public_id_state,
                                    local,
                                    upstream,
                                    mapping,
                                )
                            )
                            assert (
                                response_operation_disposition(
                                    state,
                                    operation,
                                    public_id_state,
                                    storage,
                                )
                                is expected
                            )
                            evaluated_disposition_count += 1

    assert valid_context_count == 27
    assert invalid_context_count == 21
    assert evaluated_disposition_count == 3240

    expected_input_required_continuations = {
        ProviderLaneStorage.OFF: ResponseOperationDisposition.DENIED_STATE,
        ProviderLaneStorage.STATELESS: (
            ResponseOperationDisposition.STRUCTURED_INPUT_ONLY
        ),
        ProviderLaneStorage.STORED: (
            ResponseOperationDisposition.STRUCTURED_INPUT_ONLY
        ),
    }
    for upstream, expected in expected_input_required_continuations.items():
        assert (
            response_operation_disposition(
                ResponseResourceState.INPUT_REQUIRED,
                ResponseOperation.CONTINUE,
                PublicResponseIdState.COMMITTED,
                _storage_context(upstream=upstream),
            )
            is expected
        )


def test_public_id_transitions_and_terminal_publication_are_commit_first() -> (
    None
):
    """Prevent provisional IDs from promising a resumable checkpoint."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    public_id = _object(fixture["public_response_id"])
    expected = {
        name: {_string(item) for item in _list(targets)}
        for name, targets in _object(public_id["transitions"]).items()
    }
    assert expected == {
        state.value: {target.value for target in targets}
        for state, targets in PUBLIC_RESPONSE_ID_TRANSITIONS.items()
    }
    assert set(PUBLIC_RESPONSE_ID_TRANSITIONS) == set(PublicResponseIdState)
    for response_state in ResponseResourceState:
        for checkpoint_state in CheckpointCommitState:
            for public_id_state in PublicResponseIdState:
                expected_allowed = (
                    response_state is ResponseResourceState.COMPLETED
                    and checkpoint_state is CheckpointCommitState.COMMITTED
                    and public_id_state is PublicResponseIdState.COMMITTED
                )
                assert (
                    terminal_publication_allowed(
                        response_state,
                        checkpoint_state,
                        public_id_state,
                    )
                    is expected_allowed
                )


def test_checkpoint_identity_separates_turns_from_execution_segments() -> None:
    """Represent private segments without changing their outward turn ID."""
    root = CheckpointIdentity(
        conversation_id=ConversationId("conversation-1"),
        logical_turn_id=LogicalTurnId("turn-1"),
        execution_segment_id=ExecutionSegmentId("segment-1"),
        checkpoint_id=CheckpointId("checkpoint-1"),
        branch_id=ConversationBranchId("branch-1"),
        sequence=CheckpointSequence(0),
    )
    child = CheckpointIdentity(
        conversation_id=root.conversation_id,
        logical_turn_id=root.logical_turn_id,
        execution_segment_id=ExecutionSegmentId("segment-2"),
        checkpoint_id=CheckpointId("checkpoint-2"),
        branch_id=root.branch_id,
        sequence=CheckpointSequence(1),
        parent_checkpoint_id=root.checkpoint_id,
        parent_sequence=root.sequence,
    )
    assert child.logical_turn_id == root.logical_turn_id
    assert child.execution_segment_id != root.execution_segment_id
    assert child.parent_checkpoint_id == root.checkpoint_id


@pytest.mark.parametrize(
    ("sequence", "parent", "parent_sequence", "checkpoint"),
    [
        (1, None, None, "checkpoint-1"),
        (0, "checkpoint-0", 0, "checkpoint-1"),
        (1, "checkpoint-1", 0, "checkpoint-1"),
        (2, "checkpoint-0", 0, "checkpoint-1"),
        (1, "checkpoint-0", None, "checkpoint-1"),
        (0, None, 0, "checkpoint-1"),
    ],
)
def test_checkpoint_identity_rejects_invalid_parent_sequence_relationships(
    sequence: int,
    parent: str | None,
    parent_sequence: int | None,
    checkpoint: str,
) -> None:
    """Reject roots, children, and self-parenting with invalid sequences."""
    with pytest.raises(AssertionError):
        CheckpointIdentity(
            conversation_id=ConversationId("conversation-1"),
            logical_turn_id=LogicalTurnId("turn-1"),
            execution_segment_id=ExecutionSegmentId("segment-1"),
            checkpoint_id=CheckpointId(checkpoint),
            branch_id=ConversationBranchId("branch-1"),
            sequence=CheckpointSequence(sequence),
            parent_checkpoint_id=(
                CheckpointId(parent) if parent is not None else None
            ),
            parent_sequence=(
                CheckpointSequence(parent_sequence)
                if parent_sequence is not None
                else None
            ),
        )


@pytest.mark.parametrize(
    "field",
    [
        "conversation_id",
        "logical_turn_id",
        "execution_segment_id",
        "checkpoint_id",
        "branch_id",
    ],
)
def test_checkpoint_identity_rejects_blank_identifiers(field: str) -> None:
    """Reject empty values for every checkpoint identity component."""
    values = {
        "conversation_id": "conversation-1",
        "logical_turn_id": "turn-1",
        "execution_segment_id": "segment-1",
        "checkpoint_id": "checkpoint-1",
        "branch_id": "branch-1",
    }
    values[field] = " "
    with pytest.raises(AssertionError):
        CheckpointIdentity(
            conversation_id=ConversationId(values["conversation_id"]),
            logical_turn_id=LogicalTurnId(values["logical_turn_id"]),
            execution_segment_id=ExecutionSegmentId(
                values["execution_segment_id"]
            ),
            checkpoint_id=CheckpointId(values["checkpoint_id"]),
            branch_id=ConversationBranchId(values["branch_id"]),
            sequence=CheckpointSequence(0),
        )


def test_portable_continuation_reference_reuses_existing_typed_bindings() -> (
    None
):
    """Reference structured-input state without embedding provider payloads."""
    reference = PortableContinuationReference(
        continuation_id=StructuredInputContinuationId("continuation-1"),
        state_revision=StateRevision(3),
        digest=ContinuationDigest("digest-1"),
        definition=_definition(),
        revision_binding=_binding(),
    )
    assert reference.continuation_id == "continuation-1"
    assert StructuredInputContinuationId is ContinuationId
    assert capability_revision(reference) == "capability-revision-1"
    assert set(reference.__dataclass_fields__) == {
        "continuation_id",
        "state_revision",
        "digest",
        "definition",
        "revision_binding",
    }
    assert "provider_snapshot" not in reference.__dataclass_fields__


def test_portable_continuation_reference_rejects_drift_and_bad_values() -> (
    None
):
    """Fail before state can refer to a mismatched portable continuation."""
    with pytest.raises(AssertionError):
        PortableContinuationReference(
            continuation_id=ContinuationId("continuation-1"),
            state_revision=StateRevision(0),
            digest=ContinuationDigest("digest-1"),
            definition=_definition("capability-a"),
            revision_binding=_binding("capability-b"),
        )
    with pytest.raises(AssertionError):
        PortableContinuationReference(
            continuation_id=ContinuationId(" "),
            state_revision=StateRevision(0),
            digest=ContinuationDigest("digest-1"),
            definition=_definition(),
            revision_binding=_binding(),
        )
    with pytest.raises(AssertionError):
        PortableContinuationReference(
            continuation_id=ContinuationId("continuation-1"),
            state_revision=StateRevision(-1),
            digest=ContinuationDigest("digest-1"),
            definition=_definition(),
            revision_binding=_binding(),
        )
    with pytest.raises(AssertionError):
        PortableContinuationReference(
            continuation_id=ContinuationId("continuation-1"),
            state_revision=StateRevision(0),
            digest=ContinuationDigest(" "),
            definition=_definition(),
            revision_binding=_binding(),
        )
    with pytest.raises(AssertionError):
        capability_revision(cast(PortableContinuationReference, object()))


def test_named_head_cas_is_separate_from_immutable_parent_reuse() -> None:
    """Apply revision CAS only to an explicitly selected named head."""
    fixture = _json_object(_FIXTURES / "deterministic_fixtures.json")
    cases = _list(fixture["named_head_cases"])
    for raw_case in cases:
        case = _object(raw_case)
        disposition = named_head_advance_disposition(
            NamedHeadRevision(_integer(case["expected_revision"])),
            NamedHeadRevision(_integer(case["current_revision"])),
        )
        assert disposition.value == case["disposition"]
    assert (
        named_head_advance_disposition(
            NamedHeadRevision(0), NamedHeadRevision(0)
        )
        is NamedHeadAdvanceDisposition.ADVANCE
    )
    with pytest.raises(AssertionError):
        named_head_advance_disposition(
            NamedHeadRevision(-1), NamedHeadRevision(0)
        )
    with pytest.raises(AssertionError):
        named_head_advance_disposition(
            NamedHeadRevision(0), cast(NamedHeadRevision, False)
        )


def test_idempotency_contract_covers_replay_conflict_and_failure_fences() -> (
    None
):
    """Scope idempotency and fence every ambiguous or in-progress attempt."""
    request = _idempotency_identity()
    assert (
        idempotency_disposition(request, None)
        is IdempotencyDisposition.EXECUTE
    )
    for state, expected in (
        (
            IdempotencyRecordState.COMMITTED,
            IdempotencyDisposition.REPLAY_COMMITTED,
        ),
        (
            IdempotencyRecordState.FAILED_NO_DISPATCH,
            IdempotencyDisposition.EXECUTE,
        ),
        (IdempotencyRecordState.IN_PROGRESS, IdempotencyDisposition.FENCED),
        (IdempotencyRecordState.AMBIGUOUS, IdempotencyDisposition.FENCED),
    ):
        record = IdempotencyRecord(identity=request, state=state)
        assert idempotency_disposition(request, record) is expected

    conflict = IdempotencyRecord(
        identity=_idempotency_identity(digest="different-digest"),
        state=IdempotencyRecordState.COMMITTED,
    )
    assert (
        idempotency_disposition(request, conflict)
        is IdempotencyDisposition.CONFLICT
    )
    other_key = IdempotencyRecord(
        identity=_idempotency_identity(key="different-key"),
        state=IdempotencyRecordState.COMMITTED,
    )
    assert (
        idempotency_disposition(request, other_key)
        is IdempotencyDisposition.EXECUTE
    )
    other_operation = IdempotencyRecord(
        identity=_idempotency_identity(operation=ConversationOperation.BRANCH),
        state=IdempotencyRecordState.COMMITTED,
    )
    assert (
        idempotency_disposition(request, other_operation)
        is IdempotencyDisposition.EXECUTE
    )
    other_authority = AuthorityScope(
        source=AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=AuthorityTenantId("tenant-fixture"),
        principal_id=AuthorityPrincipalId("other-principal"),
        agent_id=ConversationAgentId("agent-fixture"),
        endpoint_id=AuthorityEndpointId("endpoint-fixture"),
        network_exposed=True,
    )
    scoped_record = IdempotencyRecord(
        identity=_idempotency_identity(authority=other_authority),
        state=IdempotencyRecordState.COMMITTED,
    )
    assert (
        idempotency_disposition(request, scoped_record)
        is IdempotencyDisposition.EXECUTE
    )


def test_idempotency_types_reject_untrusted_or_empty_components() -> None:
    """Reject unscoped, untyped, or empty idempotency identities."""
    with pytest.raises(AssertionError):
        _idempotency_identity(authority=cast(AuthorityScope, object()))
    with pytest.raises(AssertionError):
        _idempotency_identity(
            operation=cast(ConversationOperation, "continue")
        )
    with pytest.raises(AssertionError):
        _idempotency_identity(key=" ")
    with pytest.raises(AssertionError):
        _idempotency_identity(digest=" ")
    with pytest.raises(AssertionError):
        IdempotencyRecord(
            identity=cast(RequestIdempotencyIdentity, object()),
            state=IdempotencyRecordState.COMMITTED,
        )
    with pytest.raises(AssertionError):
        IdempotencyRecord(
            identity=_idempotency_identity(),
            state=cast(IdempotencyRecordState, "committed"),
        )
    with pytest.raises(AssertionError):
        idempotency_disposition(
            cast(RequestIdempotencyIdentity, object()), None
        )
    with pytest.raises(AssertionError):
        idempotency_disposition(
            _idempotency_identity(), cast(IdempotencyRecord, object())
        )


def test_authority_scope_rejects_missing_or_blank_trusted_ids() -> None:
    """Require trusted authority independent of caller metadata."""
    authority = fixture_authority_scope()
    assert authority.tenant_id == "tenant-fixture"
    assert authority.source is AuthoritySource.AUTHENTICATED_SERVER_CONTEXT
    local = AuthorityScope(
        source=AuthoritySource.FIXED_LOCAL_SINGLE_USER,
        principal_id=AuthorityPrincipalId("local-principal"),
        agent_id=ConversationAgentId("local-agent"),
        endpoint_id=AuthorityEndpointId("local-endpoint"),
        local_single_user_configured=True,
    )
    assert local.tenant_id is None
    constructors: tuple[Callable[[], AuthorityScope], ...] = (
        lambda: AuthorityScope(
            source=AuthoritySource.TRUSTED_HOST_CONTEXT,
            principal_id=AuthorityPrincipalId(" "),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.TRUSTED_HOST_CONTEXT,
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId(" "),
            endpoint_id=AuthorityEndpointId("endpoint"),
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.TRUSTED_HOST_CONTEXT,
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId(" "),
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
            tenant_id=AuthorityTenantId(" "),
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.FIXED_LOCAL_SINGLE_USER,
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.FIXED_LOCAL_SINGLE_USER,
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
            local_single_user_configured=True,
            network_exposed=True,
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.FIXED_LOCAL_SINGLE_USER,
            tenant_id=AuthorityTenantId("tenant"),
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
            local_single_user_configured=True,
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.TRUSTED_HOST_CONTEXT,
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
            local_single_user_configured=True,
        ),
        lambda: AuthorityScope(
            source=cast(AuthoritySource, "trusted_host_context"),
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
        ),
        lambda: AuthorityScope(
            source=AuthoritySource.TRUSTED_HOST_CONTEXT,
            principal_id=AuthorityPrincipalId("principal"),
            agent_id=ConversationAgentId("agent"),
            endpoint_id=AuthorityEndpointId("endpoint"),
            network_exposed=cast(bool, 1),
        ),
    )
    for constructor in constructors:
        with pytest.raises(AssertionError):
            constructor()


def test_failure_fences_exactly_match_the_tracked_matrix() -> None:
    """Keep retry and reconciliation decisions total across boundaries."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    expected = _object(fixture["failure_fences"])
    assert set(expected) == {boundary.value for boundary in FailureBoundary}
    assert set(FAILURE_FENCES) == set(FailureBoundary)
    for boundary in FailureBoundary:
        raw = _list(expected[boundary.value])
        fence = FAILURE_FENCES[boundary]
        assert [
            fence.retry_rule.value,
            fence.fence_duplicate_dispatch,
            fence.preserve_parent,
            fence.quarantine_completed_upstream,
            fence.reconciliation_required,
        ] == raw
    assert (
        FAILURE_FENCES[FailureBoundary.KNOWN_NO_DISPATCH_TRANSPORT].retry_rule
        is RetryRule.BOUNDED_EFFECT_FREE
    )
    assert all(fence.preserve_parent for fence in FAILURE_FENCES.values())


def test_failure_fence_rejects_untyped_contract_values() -> None:
    """Reject booleans and retry rules that could weaken a failure fence."""
    with pytest.raises(AssertionError):
        FailureFence(
            retry_rule=cast(RetryRule, "never"),
            fence_duplicate_dispatch=False,
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=False,
        )
    with pytest.raises(AssertionError):
        FailureFence(
            retry_rule=RetryRule.NEVER,
            fence_duplicate_dispatch=cast(bool, 0),
            preserve_parent=True,
            quarantine_completed_upstream=False,
            reconciliation_required=False,
        )


def test_storage_axes_are_independent_and_provider_storage_is_disclosed() -> (
    None
):
    """Permit every pair without deriving upstream mode from local mode."""
    policies = []
    for local in LocalResponseStorage:
        for upstream in ProviderLaneStorage:
            policy = StoragePolicy(
                local=local,
                upstream=upstream,
                provider_storage_disclosed=(
                    upstream is ProviderLaneStorage.STORED
                ),
            )
            policies.append(policy)
    assert len(policies) == len(LocalResponseStorage) * len(
        ProviderLaneStorage
    )
    assert {(policy.local, policy.upstream) for policy in policies} == {
        (local, upstream)
        for local in LocalResponseStorage
        for upstream in ProviderLaneStorage
    }
    with pytest.raises(AssertionError):
        StoragePolicy(
            local=LocalResponseStorage.DURABLE,
            upstream=ProviderLaneStorage.STORED,
        )
    with pytest.raises(AssertionError):
        StoragePolicy(
            local=cast(LocalResponseStorage, "durable"),
            upstream=ProviderLaneStorage.OFF,
        )
    with pytest.raises(AssertionError):
        StoragePolicy(
            local=LocalResponseStorage.DURABLE,
            upstream=cast(ProviderLaneStorage, "off"),
        )
    with pytest.raises(AssertionError):
        StoragePolicy(
            local=LocalResponseStorage.DURABLE,
            upstream=ProviderLaneStorage.OFF,
            provider_storage_disclosed=cast(bool, 1),
        )
    with pytest.raises(AssertionError):
        ResponseStorageContext(
            policy=cast(StoragePolicy, object()),
            public_mapping=PublicResponseMappingState.ABSENT,
        )
    with pytest.raises(AssertionError):
        ResponseStorageContext(
            policy=policies[0],
            public_mapping=cast(PublicResponseMappingState, "absent"),
        )


def test_retention_uses_the_minimum_applicable_lifetime() -> None:
    """Never advertise retention beyond a known independent lifetime."""
    fixture = _json_object(_FIXTURES / "deterministic_fixtures.json")
    for raw_case in _list(fixture["retention_cases"]):
        case = _object(raw_case)
        storage = StoragePolicy(
            local=LocalResponseStorage(_string(case["local_storage"])),
            upstream=ProviderLaneStorage(_string(case["provider_storage"])),
            provider_storage_disclosed=_boolean(
                case["provider_storage_disclosed"]
            ),
        )
        limits = RetentionLimits(
            storage=storage,
            upstream_lifetime_status=UpstreamLifetimeStatus(
                _string(case["upstream_status"])
            ),
            local_ttl_seconds=cast(int | None, case["local"]),
            envelope_ttl_seconds=cast(int | None, case["envelope"]),
            known_upstream_ttl_seconds=cast(int | None, case["upstream"]),
        )
        assert limits.effective_ttl_seconds == case["effective"]
    off_storage = StoragePolicy(
        local=LocalResponseStorage.NONE,
        upstream=ProviderLaneStorage.OFF,
    )
    stored_storage = StoragePolicy(
        local=LocalResponseStorage.DURABLE,
        upstream=ProviderLaneStorage.STORED,
        provider_storage_disclosed=True,
    )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=off_storage,
            upstream_lifetime_status=UpstreamLifetimeStatus.NOT_APPLICABLE,
            local_ttl_seconds=0,
        )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=off_storage,
            upstream_lifetime_status=UpstreamLifetimeStatus.NOT_APPLICABLE,
            envelope_ttl_seconds=-1,
        )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=stored_storage,
            upstream_lifetime_status=UpstreamLifetimeStatus.KNOWN,
            known_upstream_ttl_seconds=cast(int, True),
        )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=stored_storage,
            upstream_lifetime_status=UpstreamLifetimeStatus.NOT_APPLICABLE,
        )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=off_storage,
            upstream_lifetime_status=UpstreamLifetimeStatus.UNKNOWN,
        )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=stored_storage,
            upstream_lifetime_status=UpstreamLifetimeStatus.KNOWN,
        )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=stored_storage,
            upstream_lifetime_status=UpstreamLifetimeStatus.UNKNOWN,
            known_upstream_ttl_seconds=3600,
        )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=cast(StoragePolicy, object()),
            upstream_lifetime_status=UpstreamLifetimeStatus.NOT_APPLICABLE,
        )
    with pytest.raises(AssertionError):
        RetentionLimits(
            storage=off_storage,
            upstream_lifetime_status=cast(
                UpstreamLifetimeStatus,
                "not_applicable",
            ),
        )


def test_deletion_tables_match_local_first_outbox_contract() -> None:
    """Keep local access closed while upstream deletion is retried."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    deletion = _object(fixture["deletion"])
    expected_local = {
        name: {_string(item) for item in _list(targets)}
        for name, targets in _object(deletion["local_transitions"]).items()
    }
    expected_upstream = {
        name: {_string(item) for item in _list(targets)}
        for name, targets in _object(deletion["upstream_transitions"]).items()
    }
    assert expected_local == {
        state.value: {target.value for target in targets}
        for state, targets in LOCAL_DELETION_TRANSITIONS.items()
    }
    assert expected_upstream == {
        state.value: {target.value for target in targets}
        for state, targets in UPSTREAM_DELETION_TRANSITIONS.items()
    }
    assert set(LOCAL_DELETION_TRANSITIONS) == set(LocalDeletionState)
    assert set(UPSTREAM_DELETION_TRANSITIONS) == set(UpstreamDeletionState)
    assert (
        UpstreamDeletionState.PENDING
        in UPSTREAM_DELETION_TRANSITIONS[UpstreamDeletionState.FAILED]
    )


def test_configuration_precedence_matches_the_frozen_policy() -> None:
    """Keep requests below authority policy and above provider defaults."""
    fixture = _json_object(_FIXTURES / "contract_decisions.json")
    configuration = _object(fixture["configuration"])
    assert [source.value for source in CONFIGURATION_PRECEDENCE] == _list(
        configuration["precedence_high_to_low"]
    )
    assert CONFIGURATION_PRECEDENCE == (
        ConfigurationSource.SERVER_POLICY,
        ConfigurationSource.SERVED_AGENT,
        ConfigurationSource.MODEL_PROVIDER,
        ConfigurationSource.REQUEST,
        ConfigurationSource.PROVIDER_DEFAULT,
    )
    defaults = _object(configuration["defaults"])
    assert defaults["conversation_mode"] == "off"
    assert defaults["served_store"] is False
    assert defaults["feature_activation"] == "dormant"


def test_deterministic_id_clock_and_public_resource_fixtures() -> None:
    """Keep typed IDs, clocks, and response fixtures stable across runs."""
    fixture = _json_object(_FIXTURES / "deterministic_fixtures.json")
    factory_data = _object(fixture["id_factory"])
    factory = ConversationIdFactory(_string(factory_data["prefix"]))
    generated = {
        "conversation_id": factory.conversation_id(),
        "logical_turn_id": factory.logical_turn_id(),
        "execution_segment_id": factory.execution_segment_id(),
        "checkpoint_id": factory.checkpoint_id(),
        "branch_id": factory.branch_id(),
        "named_head_id": factory.named_head_id(),
        "provider_lane_id": factory.provider_lane_id(),
        "model_call_id": factory.model_call_id(),
        "provisional_response_id": factory.provisional_response_id(),
        "public_response_id": factory.public_response_id(),
        "upstream_response_id": factory.upstream_response_id(),
        "task_id": factory.task_id(),
        "agent_id": factory.agent_id(),
        "structured_input_continuation_id": (
            factory.structured_input_continuation_id()
        ),
        "tenant_id": factory.tenant_id(),
        "principal_id": factory.principal_id(),
        "endpoint_id": factory.endpoint_id(),
        "idempotency_key": factory.idempotency_key(),
        "request_digest": factory.request_digest(),
    }
    assert generated == _object(factory_data["expected"])
    assert factory.checkpoint_sequence() == 19

    clock_data = _object(fixture["clock"])
    clock = ManualConversationClock(
        datetime.fromisoformat(_string(clock_data["initial"]))
    )
    observed = []
    for seconds in _list(clock_data["advances_seconds"]):
        clock.advance(_integer(seconds))
        observed.append(clock.now().isoformat())
    assert observed == _list(clock_data["expected"])

    for raw_resource in _list(fixture["public_response_resources"]):
        resource = _object(raw_resource)
        typed = TestResponseResource(
            name=_string(resource["name"]),
            terminal_publication_allowed=_boolean(
                resource["terminal_publication_allowed"]
            ),
        )
        assert (
            typed.terminal_publication_allowed
            is terminal_publication_allowed(
                ResponseResourceState(_string(resource["resource_state"])),
                CheckpointCommitState(_string(resource["checkpoint_state"])),
                PublicResponseIdState(_string(resource["public_id_state"])),
            )
        )


def test_deterministic_fixture_inventory_is_complete_and_dormant() -> None:
    """Freeze nested fixture values without activating a provider."""
    fixture = _json_object(_FIXTURES / "deterministic_fixtures.json")
    canonical = {
        key: value for key, value in fixture.items() if key != "fixture_sha256"
    }
    canonical_bytes = dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert fixture["fixture_sha256"] == _DETERMINISTIC_FIXTURES_DIGEST
    observed_digest = sha256(canonical_bytes).hexdigest()
    assert observed_digest == _DETERMINISTIC_FIXTURES_DIGEST
    logical_turn = "conversation-fixture-logical-turn-0002"
    provider_lane = "conversation-fixture-provider-lane-0007"
    idempotency_key = "conversation-fixture-idempotency-key-0018"
    request_digest = "conversation-fixture-request-digest-0019"
    material_source = "ephemeral test harness only; never serialized"
    authority = fixture_authority_scope()
    assert _object(fixture["principal"]) == {
        "tenant_id": authority.tenant_id,
        "principal_id": authority.principal_id,
        "agent_id": authority.agent_id,
        "endpoint_id": authority.endpoint_id,
    }
    assert fixture == {
        "schema_version": 1,
        "contract_version": 1,
        "fixture_sha256": _DETERMINISTIC_FIXTURES_DIGEST,
        "clock": {
            "initial": "2030-01-01T00:00:00+00:00",
            "advances_seconds": [0, 1, 59, 3600],
            "expected": [
                "2030-01-01T00:00:00+00:00",
                "2030-01-01T00:00:01+00:00",
                "2030-01-01T00:01:00+00:00",
                "2030-01-01T01:01:00+00:00",
            ],
        },
        "id_factory": {
            "prefix": "conversation-fixture",
            "expected": {
                "conversation_id": "conversation-fixture-conversation-0001",
                "logical_turn_id": logical_turn,
                "execution_segment_id": (
                    "conversation-fixture-execution-segment-0003"
                ),
                "checkpoint_id": "conversation-fixture-checkpoint-0004",
                "branch_id": "conversation-fixture-branch-0005",
                "named_head_id": "conversation-fixture-named-head-0006",
                "provider_lane_id": provider_lane,
                "model_call_id": "conversation-fixture-model-call-0008",
                "provisional_response_id": (
                    "conversation-fixture-provisional-response-0009"
                ),
                "public_response_id": (
                    "conversation-fixture-public-response-0010"
                ),
                "upstream_response_id": (
                    "conversation-fixture-upstream-response-0011"
                ),
                "task_id": "conversation-fixture-task-0012",
                "agent_id": "conversation-fixture-agent-0013",
                "structured_input_continuation_id": (
                    "conversation-fixture-structured-input-continuation-0014"
                ),
                "tenant_id": "conversation-fixture-tenant-0015",
                "principal_id": "conversation-fixture-principal-0016",
                "endpoint_id": "conversation-fixture-endpoint-0017",
                "idempotency_key": idempotency_key,
                "request_digest": request_digest,
            },
        },
        "principal": {
            "tenant_id": "tenant-fixture",
            "principal_id": "principal-fixture",
            "agent_id": "agent-fixture",
            "endpoint_id": "endpoint-fixture",
        },
        "keys": {
            "test_only": True,
            "checkpoint_key_ids": [
                "checkpoint-key-2029-12",
                "checkpoint-key-2030-01",
            ],
            "envelope_key_ids": [
                "envelope-key-2029-12",
                "envelope-key-2030-01",
            ],
            "idempotency_digest_key_id": "idempotency-key-2030-01",
            "material_source": material_source,
        },
        "provider_capability": {
            "profile_id": "synthetic-provider-profile-v1",
            "test_only": True,
            "activation": "dormant",
            "stored_chaining": True,
            "stateless_encrypted_replay": True,
            "current_turn_reasoning": True,
            "all_turns_reasoning": True,
            "inline_compaction": True,
            "standalone_compaction": True,
            "streaming_item_fidelity": True,
            "retrieval": True,
            "deletion": True,
        },
        "provider_item_trace": {
            "complete": True,
            "items": [
                {
                    "index": 0,
                    "kind": "reasoning",
                    "provider_id": "reasoning-1",
                    "status": "completed",
                    "opaque_payload": "opaque-fixture-ciphertext",
                },
                {
                    "index": 1,
                    "kind": "function_call",
                    "provider_id": "call-item-1",
                    "status": "completed",
                    "call_id": "call-1",
                    "phase": "assistant",
                },
                {
                    "index": 2,
                    "kind": "function_call_output",
                    "provider_id": "output-item-1",
                    "status": "completed",
                    "call_id": "call-1",
                    "phase": "tool",
                },
                {
                    "index": 3,
                    "kind": "message",
                    "provider_id": "message-1",
                    "status": "completed",
                    "phase": "final",
                },
            ],
        },
        "async_barrier": {
            "parties": 2,
            "arrival_order": ["advance-a", "advance-b"],
        },
        "fault_injection": [boundary.value for boundary in FailureBoundary],
        "named_head_cases": [
            {
                "expected_revision": 4,
                "current_revision": 4,
                "disposition": "advance",
            },
            {
                "expected_revision": 3,
                "current_revision": 4,
                "disposition": "conflict",
            },
            {
                "expected_revision": 5,
                "current_revision": 4,
                "disposition": "conflict",
            },
        ],
        "retention_cases": [
            {
                "local_storage": "durable",
                "provider_storage": "stored",
                "provider_storage_disclosed": True,
                "upstream_status": "known",
                "local": 86400,
                "envelope": 3600,
                "upstream": 7200,
                "effective": 3600,
            },
            {
                "local_storage": "durable",
                "provider_storage": "stored",
                "provider_storage_disclosed": True,
                "upstream_status": "known",
                "local": 86400,
                "envelope": None,
                "upstream": 7200,
                "effective": 7200,
            },
            {
                "local_storage": "durable",
                "provider_storage": "stored",
                "provider_storage_disclosed": True,
                "upstream_status": "unknown",
                "local": 86400,
                "envelope": 3600,
                "upstream": None,
                "effective": None,
            },
            {
                "local_storage": "none",
                "provider_storage": "off",
                "provider_storage_disclosed": False,
                "upstream_status": "not_applicable",
                "local": None,
                "envelope": None,
                "upstream": None,
                "effective": None,
            },
        ],
        "public_response_resources": [
            {
                "name": "provisional_stream",
                "resource_state": "streaming",
                "checkpoint_state": "staged",
                "public_id_state": "provisional",
                "terminal_publication_allowed": False,
            },
            {
                "name": "committed_completion",
                "resource_state": "completed",
                "checkpoint_state": "committed",
                "public_id_state": "committed",
                "terminal_publication_allowed": True,
            },
            {
                "name": "commit_failure",
                "resource_state": "failed",
                "checkpoint_state": "quarantined",
                "public_id_state": "withheld",
                "terminal_publication_allowed": False,
            },
            {
                "name": "local_tombstone",
                "resource_state": "tombstoned",
                "checkpoint_state": "committed",
                "public_id_state": "committed",
                "terminal_publication_allowed": False,
            },
        ],
    }


@pytest.mark.anyio
async def test_async_barrier_and_fault_injector_are_deterministic() -> None:
    """Coordinate races and inject failures without timing sleeps."""
    barrier = AsyncConversationBarrier(2)
    await gather(
        barrier.arrive_and_wait("advance-a"),
        barrier.arrive_and_wait("advance-b"),
    )
    assert barrier.arrivals == ("advance-a", "advance-b")

    duplicate_barrier = AsyncConversationBarrier(1)
    await duplicate_barrier.arrive_and_wait("advance-a")
    with pytest.raises(ValueError, match="more than once"):
        await duplicate_barrier.arrive_and_wait("advance-a")
    with pytest.raises(RuntimeError, match="too many"):
        await duplicate_barrier.arrive_and_wait("advance-b")

    injector = ConversationFaultInjector(
        (FailureBoundary.AMBIGUOUS_POSSIBLE_DISPATCH,),
    )
    await injector.reach(FailureBoundary.VALIDATION_BEFORE_DISPATCH)
    with pytest.raises(InjectedConversationFault) as raised:
        await injector.reach(FailureBoundary.AMBIGUOUS_POSSIBLE_DISPATCH)
    assert raised.value.boundary is FailureBoundary.AMBIGUOUS_POSSIBLE_DISPATCH
    assert injector.visited == (
        FailureBoundary.VALIDATION_BEFORE_DISPATCH,
        FailureBoundary.AMBIGUOUS_POSSIBLE_DISPATCH,
    )


def test_fixture_helpers_reject_invalid_values() -> None:
    """Reject invalid deterministic clocks, factories, barriers, and faults."""
    with pytest.raises(AssertionError):
        ManualConversationClock(datetime(2030, 1, 1))
    clock = ManualConversationClock(
        datetime.fromisoformat("2030-01-01T00:00:00+00:00")
    )
    with pytest.raises(AssertionError):
        clock.advance(-1)
    with pytest.raises(AssertionError):
        ConversationIdFactory(" ")
    factory = ConversationIdFactory("fixture")
    with pytest.raises(AssertionError):
        factory._next("")
    with pytest.raises(AssertionError):
        AsyncConversationBarrier(0)
    with pytest.raises(AssertionError):
        ConversationFaultInjector(
            (cast(FailureBoundary, "validation_before_dispatch"),),
        )
    with pytest.raises(AssertionError):
        InjectedConversationFault(
            cast(FailureBoundary, "validation_before_dispatch")
        )
    with pytest.raises(AssertionError):
        TestResponseResource(name=" ", terminal_publication_allowed=False)
    with pytest.raises(AssertionError):
        TestResponseResource(
            name="resource",
            terminal_publication_allowed=cast(bool, 0),
        )


def test_contract_query_functions_reject_untyped_enums() -> None:
    """Keep stringly typed lifecycle values outside the contract boundary."""
    storage = _storage_context()
    with pytest.raises(AssertionError):
        response_transition_allowed(
            cast(ResponseResourceState, "allocated"),
            ResponseResourceState.DISPATCHING,
        )
    with pytest.raises(AssertionError):
        response_transition_allowed(
            ResponseResourceState.ALLOCATED,
            cast(ResponseResourceState, "dispatching"),
        )
    with pytest.raises(AssertionError):
        response_operation_disposition(
            cast(ResponseResourceState, "completed"),
            ResponseOperation.RETRIEVE,
            PublicResponseIdState.COMMITTED,
            storage,
        )
    with pytest.raises(AssertionError):
        response_operation_disposition(
            ResponseResourceState.COMPLETED,
            cast(ResponseOperation, "retrieve"),
            PublicResponseIdState.COMMITTED,
            storage,
        )
    with pytest.raises(AssertionError):
        response_operation_disposition(
            ResponseResourceState.COMPLETED,
            ResponseOperation.RETRIEVE,
            cast(PublicResponseIdState, "committed"),
            storage,
        )
    with pytest.raises(AssertionError):
        response_operation_disposition(
            ResponseResourceState.COMPLETED,
            ResponseOperation.RETRIEVE,
            PublicResponseIdState.COMMITTED,
            cast(ResponseStorageContext, object()),
        )
    with pytest.raises(AssertionError):
        terminal_publication_allowed(
            cast(ResponseResourceState, "completed"),
            CheckpointCommitState.COMMITTED,
            PublicResponseIdState.COMMITTED,
        )
    with pytest.raises(AssertionError):
        terminal_publication_allowed(
            ResponseResourceState.COMPLETED,
            cast(CheckpointCommitState, "committed"),
            PublicResponseIdState.COMMITTED,
        )
    with pytest.raises(AssertionError):
        terminal_publication_allowed(
            ResponseResourceState.COMPLETED,
            CheckpointCommitState.COMMITTED,
            cast(PublicResponseIdState, "committed"),
        )


def test_immutable_contract_tables_reject_mutation() -> None:
    """Prevent runtime mutation of frozen state policy tables."""
    with pytest.raises(TypeError):
        cast(
            dict[CheckpointKind, CheckpointVisibility], CHECKPOINT_VISIBILITY
        )[
            CheckpointKind.COMPLETED_OUTWARD_TURN
        ] = CheckpointVisibility.PRIVATE_EXECUTION
    with pytest.raises(TypeError):
        cast(
            dict[ResponseResourceState, frozenset[ResponseResourceState]],
            RESPONSE_RESOURCE_TRANSITIONS,
        )[ResponseResourceState.DELETED] = frozenset(
            (ResponseResourceState.COMPLETED,),
        )
