#!/usr/bin/env python
"""Validate and execute conversation-continuity acceptance evidence."""

from argparse import ArgumentParser, Namespace
from ast import AsyncFunctionDef, ClassDef, Constant, FunctionDef, walk
from ast import parse as parse_python
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from re import compile as compile_regex
from sys import stderr
from tempfile import TemporaryDirectory

from contract_gate import (
    POSTGRESQL_TEST_DSN_ENV,
    ContractGateError,
    StrictJsonError,
    _validate_node_sources,
    canonical_sha256,
    execute_pytest_nodes,
    mapping,
    object_list,
    strict_json_path,
)
from verify_conversation_types import (
    ConversationTypeContractError,
    validate_type_source_phase_anchors,
)
from verify_conversation_types import (
    load_manifest as load_type_contract_manifest,
)


def _frozen(*values: str) -> frozenset[str]:
    return frozenset(values)


_FEATURE = "conversation_continuity"
_MARKDOWN_SUFFIX = "." + "m" + "d"
_MIN_PHASE = 0
_MAX_PHASE = 12
_NORMATIVE_OCCURRENCES = 144
_CATEGORIES = _frozen(
    "positive",
    "negative",
    "race",
    "security",
    "persistence",
    "wire",
    "integration",
    "public_e2e",
)
_DIMENSIONS = {
    "provider": _frozen(
        "native_openai",
        "native_azure",
        "incapable_generic_compatible",
    ),
    "provider_mode": _frozen(
        "off",
        "stateless_encrypted_replay",
        "provider_stored_chain",
    ),
    "local_retention": _frozen(
        "direct_process_local",
        "durable_local",
        "served_store_false",
        "served_store_true",
    ),
    "transport": _frozen("streaming", "non_streaming"),
    "execution": _frozen(
        "no_tool",
        "one_tool",
        "multiple_tool_cycles",
        "structured_input_suspension",
        "multiple_agents_lanes",
    ),
    "turn_topology": _frozen(
        "first_turn",
        "ordinary_child",
        "explicit_branch",
        "named_head_conflict",
        "retry",
        "reset",
    ),
    "reasoning_context": _frozen(
        "auto_omitted",
        "current_turn",
        "all_turns",
        "unsupported",
    ),
    "compaction": _frozen(
        "none",
        "inline_no_boundary",
        "inline_boundary",
        "repeated_boundary",
        "standalone",
    ),
    "lifecycle": _frozen(
        "same_process",
        "fresh_process",
        "expiry",
        "deletion",
        "tombstone",
        "key_rotation",
    ),
    "failure": _frozen(
        "validation",
        "known_no_dispatch",
        "ambiguous_dispatch",
        "before_output",
        "after_visible_output",
        "malformed_item",
        "commit_failure",
        "publication_failure",
    ),
    "authority": _frozen(
        "correct_principal",
        "wrong_tenant",
        "wrong_principal",
        "wrong_agent",
        "no_authenticated_authority",
    ),
    "limit": _frozen(
        "item_count",
        "bytes",
        "depth",
        "branch_count",
        "concurrency",
        "envelope_size",
        "ttl",
    ),
}
_NODE_PATTERN = compile_regex(r"^tests/[A-Za-z0-9_./-]+\.py::[^\s]+$")
_REQUIREMENT_PATTERN = compile_regex(r"^CONV-N-[0-9]{3}$")
_PHASE11_REQUIREMENT_IDS = frozenset(
    f"CONV-N-{ordinal:03d}" for ordinal in range(118, 131)
)
_ACTIVE_INTEGRATED_FIXTURES = (
    "contract_decisions.json",
    "deterministic_fixtures.json",
    "provider_contract.json",
    "provider_conformance.json",
)
_PHASE5_PROVIDER_CONFORMANCE = "provider_conformance.phase5.json"
_PHASE6_PROVIDER_CONFORMANCE = "provider_conformance.phase6.json"
_PHASE7_PROVIDER_CONFORMANCE = "provider_conformance.phase7.json"
_PHASE8_PROVIDER_CONFORMANCE = "provider_conformance.phase8.json"
_PHASE5_PROVIDER_CONFORMANCE_BYTE_SHA256 = (
    "c2cee698687f15d6147ba367450b68e863ec579866f6fc63982db8e73b7bf2f4"
)
_PHASE6_PROVIDER_CONFORMANCE_BYTE_SHA256 = (
    "7d17cbb33d159025a874bf82a5e29236661c01664987376d219ec307db306a70"
)
_PHASE7_PROVIDER_CONFORMANCE_BYTE_SHA256 = (
    "d8eb36b662f604c59589b52be2b8c972bdaef0b347a2aae2ec4d1b91ab1ec936"
)
_THREAT_IDS = _frozen(
    "opaque-state-disclosure",
    "envelope-theft",
    "confused-deputy",
    "cross-tenant-equality",
    "replay-and-rollback",
    "decompression-size-bomb",
    "orphaned-upstream-state",
    "deletion-race",
)
_EVIDENCE_CLASSES = _frozen(
    "audit",
    "contract",
    "database",
    "live",
    "matrix",
    "negative",
    "pre_dispatch_rejection",
    "public",
    "runtime",
    "security",
    "wire",
)
_PHASE8_TOOL_EVIDENCE_NODES = _frozen(
    "tests/conversation/agent_integration_e2e_test.py::"
    "test_parent_tool_effect_failure_fences_unsafe_retry",
    "tests/conversation/agent_integration_contract_test.py::"
    "test_durable_tool_crash_points_have_one_safe_recovery_action",
    "tests/conversation/agent_integration_pgsql_test.py::"
    "test_pgsql_recovery_admission_is_exact_and_single_owner",
    "tests/conversation/agent_integration_pgsql_test.py::"
    "test_pgsql_tool_boundaries_recover_without_duplicate_effect",
    "tests/conversation/native_openai_provider_validation_test.py::"
    "test_native_function_tool_rejects_invalid_schema_arguments_before_effect",
    "tests/conversation/native_openai_provider_validation_test.py::"
    "test_native_function_tool_rejects_nonlocal_schema_before_effect",
    "tests/conversation/native_openai_provider_validation_test.py::"
    "test_native_function_tool_persists_only_validated_arguments",
)
_PHASE8_DURABLE_EVIDENCE_NODES = _frozen(
    "tests/conversation/agent_integration_pgsql_test.py::"
    "test_pgsql_recovery_admission_is_exact_and_single_owner",
    "tests/conversation/agent_integration_pgsql_test.py::"
    "test_pgsql_tool_boundaries_recover_without_duplicate_effect",
    "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
    "test_fresh_worker_applies_atomic_conversation_answer_once",
)
_PHASE8_FRESH_PROCESS_EVIDENCE_NODES = _frozen(
    "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
    "test_fresh_worker_applies_atomic_conversation_answer_once",
)
_PHASE8_MULTI_AGENT_EVIDENCE_NODES = _frozen(
    "tests/conversation/agent_integration_e2e_test.py::"
    "test_parent_two_children_persist_isolation_and_restart",
    "tests/conversation/agent_integration_e2e_test.py::"
    "test_child_merge_rejects_wrong_provider_and_model_binding",
    "tests/conversation/agent_integration_e2e_test.py::"
    "test_failed_child_never_dispatches_or_publishes_parent",
)
_PHASE8_FAILURE_EVIDENCE = {
    "tool_effect--direct_sdk": (
        "tests/conversation/agent_integration_pgsql_test.py::"
        "test_pgsql_tool_boundaries_recover_without_duplicate_effect"
    ),
    "tool_effect--agent_sdk": (
        "tests/conversation/agent_integration_e2e_test.py::"
        "test_parent_tool_effect_failure_fences_unsafe_retry"
    ),
    "tool_effect--stream": (
        "tests/conversation/native_openai_provider_validation_test.py::"
        "test_native_output_byte_limit_precedes_tool_effect_and_commit"
    ),
    "tool_effect--structured_input": (
        "tests/agent/execution_wrapper_input_required_test.py::"
        "test_default_stream_has_exact_input_required_order"
    ),
    "structured_input_suspension--direct_sdk": (
        "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
        "test_fresh_worker_applies_atomic_conversation_answer_once"
    ),
    "structured_input_suspension--agent_sdk": (
        "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
        "test_fresh_worker_applies_atomic_conversation_answer_once"
    ),
    "structured_input_suspension--stream": (
        "tests/agent/execution_wrapper_input_required_test.py::"
        "test_default_stream_has_exact_input_required_order"
    ),
    "structured_input_suspension--structured_input": (
        "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
        "test_atomic_suspension_commits_every_durable_surface"
    ),
    "tool_effect--durable_checkpoint_store": (
        "tests/conversation/agent_integration_pgsql_test.py::"
        "test_pgsql_tool_boundaries_recover_without_duplicate_effect"
    ),
    "structured_input_suspension--durable_checkpoint_store": (
        "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
        "test_fresh_worker_applies_atomic_conversation_answer_once"
    ),
    "durable_transaction_failure--agent_sdk": (
        "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
        "test_atomic_suspension_rolls_back_every_durable_surface"
    ),
    "durable_transaction_failure--structured_input": (
        "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
        "test_atomic_suspension_rolls_back_every_durable_surface"
    ),
}
_PHASE8_AGENT_TOOL_FAILURE_SEMANTICS = {
    "expected_dispatch_count": 1,
    "visible_output_count": 0,
    "tool_effect_count": 1,
    "checkpoint_commit_count": 0,
    "public_mapping": "absent",
    "retry_decision": "reconcile_only",
    "parent_state": "unchanged",
    "public_error": "conversation_effect_boundary",
    "reconciliation_state": "required",
}
_PHASE8_SCHEMA_FAILURE_EVIDENCE = {
    "validation_before_dispatch--direct_sdk": (
        "tests/conversation/native_openai_provider_validation_test.py::"
        "test_native_function_tool_rejects_invalid_schema_arguments_"
        "before_effect"
    ),
    "validation_before_dispatch--provider_adapter": (
        "tests/conversation/native_openai_provider_validation_test.py::"
        "test_native_function_tool_rejects_nonlocal_schema_before_effect"
    ),
    "validation_before_dispatch--agent_sdk": (
        "tests/conversation/agent_integration_e2e_test.py::"
        "test_public_agent_rejects_lossy_input_before_dispatch"
    ),
    "validation_before_dispatch--structured_input": (
        "tests/interaction/headless_policy_test.py::"
        "test_conversation_handoff_fails_before_non_atomic_persistence"
    ),
}
_PHASE0_NODE_PAYLOAD_SHA256 = (
    "0440d0f24548c5b9ddcead0ad6f4e238416f3e0dc6683414f1eeb16dd92d046b"
)
_PHASE0_REQUIREMENTS_SHA256 = (
    "596f3f62b99be967aa09bdb1f543447d8f7580dfea533ddcaa3aaaa95e2994fe"
)
_PHASE0_FAILURE_STRUCTURE_SHA256 = (
    "ce4d56793e95d86b9b49bd1338d132c5ab5f3970c548e9827e8a56b9ca7f4956"
)
_PHASE0_THREAT_STRUCTURE_SHA256 = (
    "7d3e7470e5d978da1c5bfaba2c734c15de169f97045f33188633abc77266f239"
)
_PHASE0_PROVIDER_CANONICAL_SHA256 = (
    "2c5e6e8fd1757bcf669ffdcb6e433b4ca5b35b64f5e26d31d6aa0900e918750f"
)
_PHASE0_PROVIDER_SOURCE_SHA256 = (
    "47d250ded5a4e0006fe3116ed51b9552f3a2b1caa313c73d77581e09e9ee5a0d"
)
_PHASE0_PROVIDER_BYTE_ANCHORS = {
    "tests/fixtures/conversation/provider_contract.json": (
        34_882,
        "4ed471bf7a4018e0baca9c691c039fb0c0c9befc1931f176ad82265f04147fe5",
    ),
    "tests/model/nlp/vendor_openai_conversation_phase0_test.py": (
        96_247,
        "953066734fc2c292c26e1fa78b0a2f2ec26ad96035e01f3f0522493e94079ce8",
    ),
    "src/avalan/model/nlp/text/vendor/openai.py": (
        336_124,
        _PHASE0_PROVIDER_SOURCE_SHA256,
    ),
}
_PHASE5_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase5.json"
)
_PHASE5_PROVIDER_TARGET_BYTE_ANCHORS = {
    "src/avalan/model/nlp/text/vendor/openai.py": (
        337_354,
        "7fcedb4274ecbe56134c7a921c0fa0b4adc1ee02477afb1406151ac135c6c0c5",
    ),
    "tests/conversation/domain_contract_test.py": (
        160_196,
        "6a00228d6ab78a5e2ec6e29da5745f2fe6c083b93e1893e6e8f4fca4bafcce15",
    ),
    "tests/model/nlp/vendor_openai_conversation_phase0_test.py": (
        97_796,
        "67614ab06d27b44fa49c8553b5f7a7a0cad2de3979dd3cd44ee8adf8c134e08b",
    ),
    "tests/conversation_phase0_contract_test.py": (
        18_110,
        "da96d1a90cf07648d33cba6c3b8701dc9dfe3e8116729071a3660fcdc584a6b5",
    ),
}
_PHASE6_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase6.json"
)
_PHASE6_PROVIDER_SOURCE_BYTE_ANCHORS = {
    "src/avalan/__init__.py": (
        9_617,
        "2b41467ac4dc6d6e637d342e69e5efc96a344e15097a24f693b9b045e2ee5498",
        9_978,
        "496ce5f01ed2c12d3bb37712c37e5a97cf0a5a814acbf05f54a70ebf68518847",
    ),
    "src/avalan/conversation/__init__.py": (
        26_055,
        "044b5d410f12222266e623ed04b05ce7c0fe8288553ef436d860aed67237e0fa",
        28_421,
        "1c157fe247d198e07dff583eead46f21de8c70c9cefdc1abe80991ddd8667ddb",
    ),
    "src/avalan/conversation/coordinator.py": (
        86_533,
        "4239e49d847faee802a92da6b1089f0e4db1779973d2812abd946db4a4c4efd3",
        111_286,
        "b586445a0d5910238d191508a5a534cc8f0b8910e72081c5551e9038021e2547",
    ),
    "src/avalan/conversation/lifecycle.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        21_208,
        "edea170ab18293bcb91dcef38a95e7359cc0ee099636472cf24a52b1e288e943",
    ),
    "src/avalan/conversation/protocols.py": (
        17_492,
        "061ab8bbba83a8fef21143ab56af6b9b8a529267d58055d75ed1cbd8422b02b9",
        20_226,
        "f9faba0869f6f156139d5e88c384fbfe61caa32d5d72f2439d8914698ef37db0",
    ),
    "src/avalan/conversation/providers/__init__.py": (
        626,
        "edb3542550c21497a599ba31956a0d8054dd98715e458d948078a5d6769e8230",
        1_133,
        "4c6c408a95d86e0f254a19d70a9fd77ba98524a13da73c20f7cec28491035aa4",
    ),
    "src/avalan/conversation/providers/openai.py": (
        39_320,
        "eaf92e24c7e57bc6ba4d8db7868cd36e97489d4b4b42e5de94c9a9ce154c1fa9",
        41_406,
        "4081b066e549a014a3dd3b9b1210d0c9ca51e4b9e0c720330073d66db04e347c",
    ),
    "src/avalan/conversation/providers/openai_stored.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        29_169,
        "9c0132bb7c48669c1ee157f7a58fb8582d4f51bca116db65ff4543856cda6019",
    ),
    "src/avalan/conversation/sdk.py": (
        34_109,
        "bd650c6f99f98b896a2d0b7bde64b41c5a88d57a9c613b064b4f1273480ea1f5",
        46_113,
        "fdadba814639ce92e08e79be36dafff25e4cd793c2f94b69de109f05a8b2884d",
    ),
    "src/avalan/conversation/store.py": (
        86_275,
        "572c48d0528552a8c4fe1e8217174c700da3496c70fb120defc33c2730c14936",
        99_760,
        "6718b06f63d1ea7e097dd78aebdcbaeca85ead83277f7019114aa89b0f34e222",
    ),
    "src/avalan/conversation/stores/pgsql.py": (
        175_913,
        "351eb6d72d7f19624d83d30721cba47742ae26e3f2e4c1e98467e5201fe42c1f",
        190_583,
        "8e3afd42d2e3455400bfd19bd1329a6c2267155b5b9d8899535a7083ce49c29c",
    ),
    "src/avalan/sdk.py": (
        76_975,
        "114d6540ab9ef0c4ca3a025f0550a2c716993f90b19e551be342db4d0f23e51c",
        77_463,
        "afdf0f4056e33a39d00bf0d5b5587a26ac0d40eacf47f53bb67f278b8aecf162",
    ),
    "tests/conversation/pgsql_conformance_test.py": (
        42_709,
        "9d6ae807cec43b4b006fdc14113f9fb5c26290cc69c465522adb36e1acbfc5d5",
        47_841,
        "752c8e8abfbf7f62bbcf02974229ceb81c808fd1fcee4362a9dfd31a0b3538d6",
    ),
}
_PHASE7_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase7.json"
)
_PHASE8_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase8.json"
)
_PHASE8_PROVIDER_TRANSITION_CANONICAL_SHA256 = (
    "7ffc46ca679f9e9fbac82eee530ae0474a8e1cf90a8b8ee11d71357ac95e4284"
)
_PHASE9_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase9.json"
)
_PHASE9_PROVIDER_TRANSITION_CANONICAL_SHA256 = (
    "3908043c7a3505c885b1ca551b7070e8b2032eadc17f97d563c0fb1b79870a64"
)
_PHASE10_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase10.json"
)
_PHASE10_PROVIDER_TRANSITION_CANONICAL_SHA256 = (
    "6ac6074869fc20f3be0e7e5ecd6bab853be859b690443e68540791a1d8d0c688"
)
_PHASE11_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase11.json"
)
_PHASE11_PROVIDER_TRANSITION_CANONICAL_SHA256 = (
    "f46d632fc85b254deb9e256dd29e84b1fd26c673f7709663352f7c0866ee6829"
)
_PHASE12_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase12.json"
)
_PHASE12_PROVIDER_TRANSITION_CANONICAL_SHA256 = (
    "a1fe168b70e6266baf2ec7498ea98de096b980e5528e047ce598e899ffe57136"
)
_PHASE13_PROVIDER_TRANSITION_PATH = (
    "tests/fixtures/conversation/provider_transition.phase13.json"
)
_PHASE13_PROVIDER_TRANSITION_CANONICAL_SHA256 = (
    "9b438f5c835f622352c6c7b05f65a31991fc8e30555a187fdef9eae8b4d1e9ff"
)
_PHASE13_PROVIDER_TARGET_BYTE_ANCHORS = {
    "src/avalan/model/nlp/text/vendor/openai.py": (
        363_588,
        "cf925379d546ac052813b724fa7307d772b8ad41019cf36bfe935708507bded3",
    ),
    "tests/conversation/domain_contract_test.py": (
        161_490,
        "0ac0cbc3e5a1f4204829e102cd64478f06ca25e19a9ea08ef6d0a4672b032760",
    ),
    "tests/model/nlp/vendor_openai_conversation_phase0_test.py": (
        100_430,
        "ce413aff06e5b71bc1e2e545af4f5f0fc197e7e71831ead2c0df23a379f33d61",
    ),
}
_PHASE13_PROVIDER_EVIDENCE_NODES = (
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_private_replay_replacement_open_uses_shared_retry_budget"
    ),
    (
        "tests/model/nlp/vendor_openai_conversation_phase0_test.py::"
        "test_no_production_conversation_dispatch_or_advertisement"
    ),
    (
        "tests/model/nlp/vendor_openai_test.py::"
        "OpenAIAdditionalCoverageTestCase::"
        "test_tool_result_images_are_native_ordered_continuation_content"
    ),
    (
        "tests/model/nlp/vendor_openai_test.py::"
        "OpenAIAdditionalCoverageTestCase::"
        "test_tool_result_image_without_pixels_fails_explicitly"
    ),
    (
        "tests/model/nlp/vendor_openai_test.py::"
        "OpenAIAdditionalCoverageTestCase::"
        "test_repeated_tool_image_calls_keep_call_association"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_transport_timeout_stream_failure_retries_privately"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_transport_timeout_retry_exhaustion_reports_safe_category"
    ),
    (
        "tests/model/nlp/vendor_openai_reasoning_summary_test.py::"
        "test_private_replay_adapter_failure_reports_safe_status"
    ),
)
_PHASE12_TRACEABILITY_CANDIDATE_PATH = (
    "tests/fixtures/conversation/acceptance_candidate.phase12.json"
)
_PHASE12_TRACEABILITY_CANDIDATE_BYTE_SHA256 = (
    "069bfd96411f55a84eba36d600cc7f77c9566bde1a3129a20149bd88addd54ce"
)
_PHASE12_TRACEABILITY_CANDIDATE_CANONICAL_SHA256 = (
    "0517adf8b2cfd0d35c78eb330c56167a1c23773ede08e444c9ee8cfae957dcdc"
)
_PHASE12_TRACEABILITY_MAPPING_CANONICAL_SHA256 = (
    "463e962f400c96c61a63671cfb3b0c06acbe2f5e8b29f50c428008e6129ea160"
)
_PHASE12_ACTIVATION_DECISION_PATH = (
    "tests/fixtures/conversation/activation_manifest.phase12.json"
)
_PHASE12_ACTIVATION_DECISION_BYTE_SHA256 = (
    "23a54ef76395508fa407f319646b7c5d95677f5a0d19c559b19914042de98ade"
)
_PHASE12_ACTIVATION_DECISION_CANONICAL_SHA256 = (
    "f890ff3a5f5ee26368070fc29fd3a5a4be4efe2585efc72b5b4750fc75a0824c"
)
_PHASE12_LIVE_RESULTS_PATH = (
    "tests/fixtures/conversation/live_conformance_results.phase12.json"
)
_PHASE12_LIVE_RESULTS_BYTE_SHA256 = (
    "40ea57a6b1300259d28e5aca81676ce770ae7fdc31c2d7a2376b64fd92a6cf31"
)
_PHASE12_LIVE_RESULTS_CANONICAL_SHA256 = (
    "d3aab6c4e4c83be848a304126c2d933898e499909bc65594959f74bb00c66e44"
)
_PHASE12_LIVE_PROOF_PREFIX = "conversation-live-receipt-v1"
_PHASE12_LIVE_NODE_ID = (
    "tests/conversation/live_conformance_test.py::"
    "test_normative_completion_contract"
)
_PHASE12_MATRIX_NODE_ID = (
    "tests/conversation/full_matrix_e2e_test.py::"
    "test_required_matrix_cross_product"
)
_PHASE12_LIVE_CASES = (
    "inline_compaction",
    "standalone_compaction_and_unpruned_replay",
    "stateless_all_turns_replay",
    "stateless_current_turn_tool",
    "stored_create",
    "stored_previous_response_chain",
    "stored_retrieve_delete",
    "streaming_tool",
)
_PHASE12_MATRIX_CASES = (
    "activation_apply_exact",
    "authorization_rejected_before_dispatch",
    "dispatch_failure_withholds_commit",
    "durable_storage_absence_rejects_stored_mode",
    "expired_manifest_rejected_before_dispatch",
    "generic_compatible_rejected_before_dispatch",
    "historical_deletion_survives_revocation",
    "key_rotation_retains_grace_read",
    "native_transport_mode_reasoning_compaction_cross_product",
    "revocation_blocks_new_dispatch",
    "rollback_restores_dormant_state",
    "stored_standalone_compaction_rejected",
)
_PHASE12_EXTERNAL_BLOCKER_STATES = {
    "native_openai_live_receipt": (
        "authorized_model_present_account_credit_quota_blocked_before_inference"
    ),
}
_PHASE12_ROLLBACK_NODE_ID = (
    "tests/conversation/activation_test.py::"
    "test_rollback_restores_prior_manifest_or_dormant_state"
)
_PHASE12_DOCUMENTATION_INDEX_NODE_ID = (
    "tests/conversation/documentation_test.py::"
    "test_documentation_and_examples_are_safe_and_indexed"
)
_PHASE12_RUNBOOK_NODE_ID = (
    "tests/conversation/documentation_test.py::"
    "test_runbook_commands_reference_real_nodes_and_tracked_fixtures"
)
_PHASE12_SECURITY_DOCUMENTATION_NODE_ID = (
    "tests/conversation/documentation_test.py::"
    "test_security_migration_and_operator_contracts_are_complete"
)
_PHASE12_FAKE_SDK_MATRIX_NODE_ID = (
    "tests/conversation/live_conformance_harness_test.py::"
    "test_full_fake_sdk_matrix_is_typed_redacted_and_closed"
)
_PHASE12_STRUCTURAL_ASSERTIONS_NODE_ID = (
    "tests/conversation/live_conformance_harness_test.py::"
    "test_structural_assertions_reject_each_invalid_live_branch"
)
_PHASE12_CANDIDATE_ONLY_EVIDENCE = {
    _PHASE12_ROLLBACK_NODE_ID: "runtime",
    _PHASE12_DOCUMENTATION_INDEX_NODE_ID: "contract",
    _PHASE12_RUNBOOK_NODE_ID: "contract",
    _PHASE12_SECURITY_DOCUMENTATION_NODE_ID: "contract",
    _PHASE12_FAKE_SDK_MATRIX_NODE_ID: "wire",
    _PHASE12_STRUCTURAL_ASSERTIONS_NODE_ID: "negative",
}
_PHASE7_PROVIDER_SOURCE_BYTE_ANCHORS = {
    "src/avalan/__init__.py": (
        9_978,
        "496ce5f01ed2c12d3bb37712c37e5a97cf0a5a814acbf05f54a70ebf68518847",
        10_046,
        "34094438bb8b6b2b3ba6197127c6903e5f8c7dc76cc6873f61976f58d1ab7ed3",
    ),
    "src/avalan/conversation/__init__.py": (
        28_421,
        "1c157fe247d198e07dff583eead46f21de8c70c9cefdc1abe80991ddd8667ddb",
        28_960,
        "98a018f55684c85faca8eac4cc0e650a7f6ab6d0068d8f18ee6fe8c7cd0a04a0",
    ),
    "src/avalan/conversation/binding.py": (
        11_628,
        "6677821422ae77ecef8ee2939d60ac8f11ce51b23d752974fea4b27df283f527",
        12_110,
        "d2a31bc6bf27a33cb41795eeb6dab6efffa1aaff5ccd071aaa6b9555d286c59d",
    ),
    "src/avalan/conversation/codec.py": (
        32_829,
        "64d9fc44e494453625538a4250b46d83a686f8e406c3b6340f31eb6ae5ef92e9",
        33_233,
        "eb7dfd3f39946fe8bbad368f73082cb47acfcd9b130bdac51bba390e5aa3ce1a",
    ),
    "src/avalan/conversation/coordinator.py": (
        111_286,
        "b586445a0d5910238d191508a5a534cc8f0b8910e72081c5551e9038021e2547",
        127_460,
        "7fd15c415a3a28a9f7fa477a9c01cfc076542fce1a2fe6fbb18686151630e838",
    ),
    "src/avalan/conversation/fakes.py": (
        61_349,
        "feeab4cbcc14e4882d7b8de19481353cff60e0eb787655e22cab4faf2e782957",
        61_559,
        "642941f1bf429e0187242da7fbf89f4debc4c4746ec15585ad959ca742cc4b35",
    ),
    "src/avalan/conversation/items.py": (
        69_302,
        "dce99215796d7274fd8948bce9a1574cf8e5f34dc2087039208c7ef4809ef1f8",
        71_181,
        "8d5d2e2342513f688aa8d7478dec666a86d857ca21d23d79b734fba8062594e8",
    ),
    "src/avalan/conversation/protocols.py": (
        20_226,
        "f9faba0869f6f156139d5e88c384fbfe61caa32d5d72f2439d8914698ef37db0",
        21_895,
        "35a00f98479afca3182663e9e2a8b8199b324a703650ffa6e8636b0b769a46ba",
    ),
    "src/avalan/conversation/providers/__init__.py": (
        1_133,
        "4c6c408a95d86e0f254a19d70a9fd77ba98524a13da73c20f7cec28491035aa4",
        1_347,
        "c28bddfd6a1465d8539d852a7bbb0dca3e9543f8d40f03e95897b704e761098a",
    ),
    "src/avalan/conversation/providers/openai.py": (
        41_406,
        "4081b066e549a014a3dd3b9b1210d0c9ca51e4b9e0c720330073d66db04e347c",
        59_231,
        "9ccca73e880c89dd7699482b730476fd920036d559ab076a430323f775832d2b",
    ),
    "src/avalan/conversation/providers/openai_stored.py": (
        29_169,
        "9c0132bb7c48669c1ee157f7a58fb8582d4f51bca116db65ff4543856cda6019",
        34_824,
        "aed336669d63d4784f9facd901190951956f389bda16b06d033ac160a139857c",
    ),
    "src/avalan/conversation/runtime.py": (
        38_395,
        "5c798ee3a69fc8dabed5b61fcabc59f9d3f026cbf3a646f294ca641c378eefe2",
        38_477,
        "a8bf7538728adb7999d698fcc18dc116785e3dd200363d7f5bd687719871fba9",
    ),
    "src/avalan/conversation/sdk.py": (
        46_113,
        "fdadba814639ce92e08e79be36dafff25e4cd793c2f94b69de109f05a8b2884d",
        52_924,
        "0e127adcfd4677fb9184f0e9704f33906bc371e3629f1b79a07e30d38110a216",
    ),
    "src/avalan/conversation/settings.py": (
        22_624,
        "30668a8c786172a7b3463c9bd898782029efcf354d0ec17a600495e016a415d7",
        25_654,
        "f8e6fdbc44356f65aeae175d0e4c353457909ecbe98c5965ce797d874a2ebef1",
    ),
    "src/avalan/conversation/state.py": (
        25_490,
        "d852b5a9019ce18c5221757f00833b0f7d9ba5b03ec917f2ad74be24c8913265",
        25_880,
        "1cc077709f50104307c078e7304a11782059d7fdc322d4c0d6f80d1f1f445f82",
    ),
    "src/avalan/conversation/store.py": (
        99_760,
        "6718b06f63d1ea7e097dd78aebdcbaeca85ead83277f7019114aa89b0f34e222",
        102_450,
        "92c0864f3eef0974f1eed606b802635aed6e51714bbabedfb211f2cabb010ce3",
    ),
    "src/avalan/conversation/stores/pgsql.py": (
        190_583,
        "8e3afd42d2e3455400bfd19bd1329a6c2267155b5b9d8899535a7083ce49c29c",
        193_033,
        "20bcfc86a6d5a66a5fb6089ddf76e9598c3765ca05b2e596ae3a84469cf602e3",
    ),
    "src/avalan/sdk.py": (
        77_463,
        "afdf0f4056e33a39d00bf0d5b5587a26ac0d40eacf47f53bb67f278b8aecf162",
        77_558,
        "00091d7b4092444777217918e04a30116792fa88617d3e80857a4e16a84b042b",
    ),
    "tests/conversation/direct_sdk_test.py": (
        25_942,
        "a8a10e69b27842a3b0ec1138d04c1e758a618d091678623fff6e4f3a18d566a8",
        26_694,
        "a0a7aa6cddac7865eeefbefe4726c2112a9041170347ef85d6f658ae522e5295",
    ),
    "tests/conversation/domain_contract_test.py": (
        160_196,
        "6a00228d6ab78a5e2ec6e29da5745f2fe6c083b93e1893e6e8f4fca4bafcce15",
        161_460,
        "3d20fe77585eed53afcddec0127de6a40e3af2971a42fb4d343db9a90853eb80",
    ),
}
_PHASE0_ACTIVE_SOURCE_SHA256 = {
    "tests/conversation_contract_gate_test.py": (
        "c014a962f1e0384370bc70113acc7189de48bbd0e7ecba54c041054eee4de349"
    ),
    "tests/conversation_phase0_contract_test.py": (
        "1b3c5fd038c7e8436a42e6456afbc7189465cc80be9ea06c6a4e5f23334a10fc"
    ),
    "tests/conversation_response_dormancy_test.py": (
        "47ee7631ee6f9928ffcfdd5550325a98f16c1f828f51a36eba7c104b298647bb"
    ),
}
_ACTIVE_SOURCE_SHA256_BY_PHASE = {
    0: _PHASE0_ACTIVE_SOURCE_SHA256,
    1: {
        "tests/conversation/domain_contract_test.py": (
            "b339ff4b902b43fa9f39487e073c256b7b52d994b793e670d87e16fe91078141"
        )
    },
    2: {
        "tests/conversation/coordinator_e2e_test.py": (
            "7a65b802d76f1dc5c2b573550f5ae85d5e836befb7048810dac4a8acb4a43d10"
        ),
        "tests/conversation/coordinator_failure_matrix_test.py": (
            "c84b03d197a486df3ee79ab396c69de4cbc5037786e25054debdea71f81a3519"
        ),
    },
    3: {
        "tests/conversation/pgsql_restart_e2e_test.py": (
            "9321dc34c4a5148233ef31a68c4d2b2672fbb892c8cab7b9a9c4c14889813d43"
        ),
        "tests/conversation/pgsql_conformance_test.py": (
            "9d6ae807cec43b4b006fdc14113f9fb5c26290cc69c465522adb36e1acbfc5d5"
        ),
        "tests/conversation/pgsql_store_test.py": (
            "b585ff0ad94903b254b674112adadd55f7e2c9b42c69e312508288c7c88b6edb"
        ),
        "tests/interaction/stores/conversation_atomic_pgsql_test.py": (
            "d31c5f667fff1a4a54ed51cc0e16c8953b7c4e7126128ee1098ce5958354ee74"
        ),
    },
    4: {
        "tests/conversation/direct_generation_contract_test.py": (
            "11bc20f6645bb8e47d1768c8ccaa0b62aea79a2ba4a8297f48fcd8a2c2517ed2"
        ),
        "tests/conversation/direct_sdk_test.py": (
            "a8a10e69b27842a3b0ec1138d04c1e758a618d091678623fff6e4f3a18d566a8"
        ),
        "tests/conversation/direct_sdk_pgsql_test.py": (
            "b8f5f49ecdb3d9a39a232bc381159c5c78c9f2aba71b432810c611c08c2b29d8"
        ),
        "tests/conversation/sdk_e2e_test.py": (
            "1359c82abdac356fe9c40d527f306ae48c00bdb91c7e8e594e9cfd525626c835"
        ),
    },
    5: {
        "tests/conversation/coordinator_e2e_test.py": (
            "7a65b802d76f1dc5c2b573550f5ae85d5e836befb7048810dac4a8acb4a43d10"
        ),
        "tests/conversation/native_openai_provider_test.py": (
            "efc8b5908542e8c9cabf5da3aabe991cff52d7a57f06b7a5c57f2062a48bdc60"
        ),
        "tests/conversation/native_openai_provider_validation_test.py": (
            "37d0e30e733e56c9e388ae0492aac931d2b98245497b4359ca9740e950c1dc74"
        ),
        "tests/conversation/openai_stateless_e2e_test.py": (
            "3e9d28ace23a848add8cd479d1060a86f77c0abb9d0e34f979197098f32accb2"
        ),
    },
    6: {
        "tests/conversation/native_openai_stored_provider_test.py": (
            "9688ce9407c1b33940aafec26d1a108b74c59579ee7d0e77bf79a013b75723ef"
        ),
        "tests/conversation/openai_stored_e2e_test.py": (
            "a0fcaa163b736849392e9c51f32de70a95a7da0b5b5b903e17ca44a8db540892"
        ),
        "tests/conversation/pgsql_conformance_test.py": (
            "9d6ae807cec43b4b006fdc14113f9fb5c26290cc69c465522adb36e1acbfc5d5"
        ),
        "tests/conversation/phase6_validation_test.py": (
            "43c337c9f430a36c2472b2b5f40f5d5513b804c86d71e961afd2724f7b2a2d08"
        ),
    },
    7: {
        "tests/conversation/compaction_contract_test.py": (
            "afa41bbfd11b0323d4af0a4f6771c7bfc77241f0b2d9a13307207d7899e931e0"
        ),
        "tests/conversation/compaction_e2e_test.py": (
            "6b1186ecedf1e40e7b34d3ad8335e166a059c12c39878ed7742c3f0ec4e277bf"
        ),
        "tests/conversation/native_openai_compaction_test.py": (
            "7e1f208d3a44a756dcdeca127117c75844ce22ec935ef16b48576bf484b41211"
        ),
    },
    8: {
        "tests/agent/durable_continuation_resume_test.py": (
            "759ccfe458701fe9e0ffa0a549afc0b0a0a9ec0258c2385e4c25062d946b326c"
        ),
        "tests/agent/execution_wrapper_input_required_test.py": (
            "fd868276a07f8b9893641bdeadd16448c15c2336772a554b484c01149981f81e"
        ),
        "tests/conversation/agent_integration_contract_test.py": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "tests/conversation/agent_integration_e2e_test.py": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "tests/conversation/agent_integration_pgsql_test.py": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "tests/conversation/native_openai_provider_validation_test.py": (
            "37d0e30e733e56c9e388ae0492aac931d2b98245497b4359ca9740e950c1dc74"
        ),
        "tests/interaction/headless_policy_test.py": (
            "531b4a900e33ad05836c55415c587979833efb60b19019cc3d9a1c762fc06c5e"
        ),
        "tests/interaction/stores/conversation_atomic_pgsql_test.py": (
            "d31c5f667fff1a4a54ed51cc0e16c8953b7c4e7126128ee1098ce5958354ee74"
        ),
    },
    9: {
        "tests/conversation/server_stored_e2e_test.py": (
            "824e502fb7bcf9f6a97d720e20a5fd77e4ec41f4e1af10f69a0c268de45fb555"
        ),
    },
    10: {
        "tests/conversation/server_stateless_e2e_test.py": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
    },
    11: {
        "tests/conversation/security_e2e_test.py": (
            "aca93bf163c7497de675ac296e93f612a6c59d8b768644622a9101a053ce6ebb"
        ),
        "tests/conversation/server_stored_e2e_test.py": (
            "f0cfbc664524d7378bafcd4a9a38bb8748e91d5cbadf207e659a55f487380a9c"
        ),
    },
}
_NODE_PAYLOAD_SHA256_BY_PHASE = {
    0: _PHASE0_NODE_PAYLOAD_SHA256,
    1: "9a85447f5de838051a3801b66eccd865ecc62b6e72ecfd9d3084603468ff8663",
    2: "8014ee73f5334290be612a567836f02aace953764a14add8d02b1407b487d441",
    3: "a0f3d780942570e794f1134e2da69754f6c8eabbd419285481653839104126ef",
    4: "4882ad8775ecb064daf399f2f83c32c913f4a0d5168396d18c5936936075a3eb",
    5: "b520005336f99a95a381c18887e5d319daf71dcc9bd7e03e2cc8fde05f2143f0",
    6: "c975653c7ae9bcd020f91be6785ddeb9a28ca9e8f39fa17e0a906e2cfee1a701",
    7: "c791ea6bc92ad8ee8140bdddf2396c7f485306d946ce315723c274f654224227",
    8: "c922e9081f5944d65b923222372029f365f65e7a7ce4350cdbe43c74d0ce9110",
    9: "2002bcaa8330005dc9cd01574c75ee4f2be2ddaca3bbff93dd9d77996e4ff527",
    10: "1f610ea0f07a58132dc7a829c44b31b8b4c697f680dd896836d3de5b8f63d1c2",
    11: "6cd469aa39cd7948f4c83fa99d24bd1177b92ce7c8e3f096d9b897fa857b0ce3",
}
_ACTIVATION_HISTORY_BY_PHASE = {
    0: "b8385b1c2ee8c56e7118ccd6c27a25d746974378808e92699953e5c846567f74",
    1: "cc98a83a046019ac7bb1f2c16469cc3a67fa6408885e87ff1fb6b265c6aa6161",
    2: "8a6da13a0627cd0a167649dc9708e3585a717f6cb71db3c17a1c686686c295ca",
    3: "3b1d94a50ca44b715a02b989646bef9daea8d471d649491465a1431cf277194f",
    4: "869ac471b5bd88df84ff12ccae1d4c7929a70aa8e339a410a7ab031c873cf0b1",
    5: "b9d2247bc0db892b1e6da8b5b718fb814a19cc2641cb3a5ec54dc9d17e4b4bc5",
    6: "9e21f629f5fa6f3023321a873af7ccd934621fb55d398e52a7aa69abd4e2b3fa",
    7: "b8fea5ce7efd5ef3f44913da597a7befec1a96084282a3524d58b807e88a2bc9",
    8: "f7dfd4dc132cd9007d09a73a6f884688e7084324ba01773b6be585b7ad261365",
    9: "3ba2b5064970f4fbfabd203b9fe0c453c2ddbd2751c511224df807850e1c3355",
    10: "4cbc71ed23fca55866c3b2f0b5d2c1eb0efa2cb05c1649b0bd92ef0002a0cb4c",
    11: "4238a684e10e5e484a08c360444288b3bea188e133a196901f3d208906c9ecf6",
}
_REPLACEMENT_HISTORY_BY_PHASE = {
    0: (
        0,
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    1: (
        1,
        "c8982b4da6b6603a382d3319688e73d9a495ecee29d2301b2c4962cdb62b1e8b",
    ),
    2: (
        2,
        "e0208c0580ae9f450254951d1bac8e761b28502d98d89586ec6d616138fb73e1",
    ),
    3: (
        3,
        "924ca90b4812ca9dcbf9776a6c9d845d2ddceb2b8081d15c9eaa345b94bb6453",
    ),
    4: (
        5,
        "1ed1d510bf0a2a09729884cf42b12e078d2acb16cffaa32d0e1fefc5427279a6",
    ),
    5: (
        6,
        "188ba85ef7d6039495c5814f68e5ffbfa1e0581f73102b13a12b21770332123a",
    ),
    6: (
        12,
        "292080771fdfc270bf429b31b265371982b0b9b8b037f0541cce7eb47f99d870",
    ),
    7: (
        19,
        "28b06b757ec0cf38561aab854b11956b70f86d0fa52cf110af0edd26b9795144",
    ),
    8: (
        21,
        "20bb89ea58062fd7588ecab0a2a2e7f7d33ffc7c2057157ca9969c2bdfcd976d",
    ),
    9: (
        22,
        "39fe2f2258df939cdf93338a2ed4bf797f9bfbdd1324dbe6856f1e8fe12d0395",
    ),
    10: (
        22,
        "39fe2f2258df939cdf93338a2ed4bf797f9bfbdd1324dbe6856f1e8fe12d0395",
    ),
    11: (
        23,
        "c158a591a5bfb68718fb08bfd2300611bada6a908e2d7a565a31a14d40f0daa1",
    ),
}
_FAILURE_STRUCTURE_BY_PHASE = {
    0: (11, 9, 99, _PHASE0_FAILURE_STRUCTURE_SHA256),
    1: (11, 9, 99, _PHASE0_FAILURE_STRUCTURE_SHA256),
    2: (11, 9, 99, _PHASE0_FAILURE_STRUCTURE_SHA256),
    3: (
        12,
        10,
        120,
        "dc99eabb897899280ae9ce8a9ea20c377edda2fae0c838fc824d90e6d5ec543b",
    ),
    4: (
        12,
        10,
        120,
        "dc99eabb897899280ae9ce8a9ea20c377edda2fae0c838fc824d90e6d5ec543b",
    ),
    5: (
        12,
        10,
        120,
        "dc99eabb897899280ae9ce8a9ea20c377edda2fae0c838fc824d90e6d5ec543b",
    ),
    6: (
        13,
        10,
        130,
        "83cb33eeb336ce865843cbf5b7872f47f428ef8c68ff90805a96b857a807a0d0",
    ),
    7: (
        19,
        10,
        190,
        "d44224b1acbb51572b311d60297f597dfcc00145334b163c0efeef2d37c4b5ab",
    ),
    8: (
        19,
        10,
        190,
        "d44224b1acbb51572b311d60297f597dfcc00145334b163c0efeef2d37c4b5ab",
    ),
}
_PHASE8_CORRECTED_FAILURE_STRUCTURE_BY_PHASE = {
    0: (
        11,
        9,
        99,
        "66bb92a99d4392fa9d05038dbacee97c42aebfeb6e843bcdf553d4bc62a33c5f",
    ),
    1: (
        11,
        9,
        99,
        "66bb92a99d4392fa9d05038dbacee97c42aebfeb6e843bcdf553d4bc62a33c5f",
    ),
    2: (
        11,
        9,
        99,
        "66bb92a99d4392fa9d05038dbacee97c42aebfeb6e843bcdf553d4bc62a33c5f",
    ),
    3: (
        12,
        10,
        120,
        "c06779f69ff721a928eb4dc6956867db50ac010cb1c0f6e9471b24f17c0bef80",
    ),
    4: (
        12,
        10,
        120,
        "c06779f69ff721a928eb4dc6956867db50ac010cb1c0f6e9471b24f17c0bef80",
    ),
    5: (
        12,
        10,
        120,
        "c06779f69ff721a928eb4dc6956867db50ac010cb1c0f6e9471b24f17c0bef80",
    ),
    6: (
        13,
        10,
        130,
        "0a8eb708d8778242affc06006238a6784b6a0a63039ed214e6e640fb572864ce",
    ),
    7: (
        19,
        10,
        190,
        "30a5e1c30de04d41ea1226f4f2b06ee2984bb719616ce3af24a4858b16b2815a",
    ),
    8: (
        19,
        10,
        190,
        "30a5e1c30de04d41ea1226f4f2b06ee2984bb719616ce3af24a4858b16b2815a",
    ),
    9: (
        19,
        10,
        190,
        "30a5e1c30de04d41ea1226f4f2b06ee2984bb719616ce3af24a4858b16b2815a",
    ),
}
_PHASE9_CORRECTED_FAILURE_STRUCTURE_BY_PHASE = {
    **_PHASE8_CORRECTED_FAILURE_STRUCTURE_BY_PHASE,
    3: (
        12,
        10,
        120,
        "27d718faa873746e7ca32798c5a836589aca1590287f5aeef1a133f5954d63a4",
    ),
    4: (
        12,
        10,
        120,
        "27d718faa873746e7ca32798c5a836589aca1590287f5aeef1a133f5954d63a4",
    ),
    5: (
        12,
        10,
        120,
        "27d718faa873746e7ca32798c5a836589aca1590287f5aeef1a133f5954d63a4",
    ),
    6: (
        13,
        10,
        130,
        "9dcacf052c2c3a26e3fdde298e59bdf02b52665915ce0d877b9a464de21f533d",
    ),
    7: (
        19,
        10,
        190,
        "7527fe33d610969b1e5f1380315181ea7488c103613e17c2ba1a6d5b90842ba4",
    ),
    8: (
        19,
        10,
        190,
        "7527fe33d610969b1e5f1380315181ea7488c103613e17c2ba1a6d5b90842ba4",
    ),
    9: (
        19,
        10,
        190,
        "7527fe33d610969b1e5f1380315181ea7488c103613e17c2ba1a6d5b90842ba4",
    ),
    10: (
        19,
        10,
        190,
        "7527fe33d610969b1e5f1380315181ea7488c103613e17c2ba1a6d5b90842ba4",
    ),
    11: (
        19,
        10,
        190,
        "7527fe33d610969b1e5f1380315181ea7488c103613e17c2ba1a6d5b90842ba4",
    ),
}
_THREAT_STRUCTURE_BY_PHASE = {
    0: (5, 5, 8, _PHASE0_THREAT_STRUCTURE_SHA256),
    1: (5, 5, 8, _PHASE0_THREAT_STRUCTURE_SHA256),
    2: (5, 5, 8, _PHASE0_THREAT_STRUCTURE_SHA256),
    3: (
        9,
        8,
        14,
        "c973ba36785f5f28d51ed744a6184b1f40ec61e6f99eb2e0677bc6594fee5b88",
    ),
    4: (
        9,
        8,
        15,
        "25399ab5bab61e83943cdc21ce8278b012e2f64792be0e6f6a2bd8a2e9715569",
    ),
    5: (
        9,
        8,
        16,
        "57e2cd5e05338824a56132b2011292897a34b9c2bc5673f9c14ef9176b364667",
    ),
    6: (
        10,
        9,
        23,
        "0e90f43443ae9106ebb1f371db0f250417856c73db1ff4c9fb2e9170f9cef8c1",
    ),
    7: (
        11,
        10,
        25,
        "9949053c95e1e85ad3e1ba0bd28f6f4a6a2ed91f001ef4ee2cae30c5b7fa8fbc",
    ),
    8: (
        15,
        14,
        29,
        "045c3597d6633aeb595b5081e9944b52742d09fb132aae62cf6f4c73826192d8",
    ),
    9: (
        15,
        14,
        29,
        "045c3597d6633aeb595b5081e9944b52742d09fb132aae62cf6f4c73826192d8",
    ),
    10: (
        15,
        14,
        29,
        "045c3597d6633aeb595b5081e9944b52742d09fb132aae62cf6f4c73826192d8",
    ),
    11: (
        15,
        14,
        35,
        "2a4fbfa4ade6e77a4517219259ff2179aed76b4bac19ef1844623720f079dce5",
    ),
}
_PHASE0_NODE_INVENTORY = (
    (
        "phase0-positive-fixtures",
        "tests/conversation_phase0_contract_test.py::test_phase0_contract_fixtures_are_frozen",
        0,
        "contract",
    ),
    (
        "phase0-negative-dormancy",
        "tests/conversation_phase0_contract_test.py::test_all_production_capabilities_remain_dormant",
        0,
        "negative",
    ),
    (
        "phase0-race-sealed-inventory",
        "tests/conversation_contract_gate_test.py::test_sealed_inventory_rejects_mid_run_mutation",
        0,
        "runtime",
    ),
    (
        "phase0-security-threats",
        "tests/conversation_phase0_contract_test.py::test_phase0_threat_controls_are_complete",
        0,
        "security",
    ),
    (
        "phase0-persistence-state-table",
        "tests/conversation_phase0_contract_test.py::test_contract_state_tables_are_total",
        0,
        "contract",
    ),
    (
        "phase0-wire-provider-evidence",
        "tests/conversation_phase0_contract_test.py::test_provider_contract_evidence_is_typed_and_dormant",
        0,
        "wire",
    ),
    (
        "phase0-public-fail-closed",
        "tests/conversation_response_dormancy_test.py::test_responses_reject_dormant_conversation_fields_before_dispatch",
        0,
        "pre_dispatch_rejection",
    ),
    (
        "phase0-one-shot-regression",
        "tests/conversation_phase0_contract_test.py::test_one_shot_behavior_omits_conversation_state",
        0,
        "runtime",
    ),
    (
        "phase0-source-isolation",
        "tests/conversation_phase0_contract_test.py::test_tracked_gate_sources_do_not_depend_on_ignored_material",
        0,
        "audit",
    ),
)
_PHASE0_FAILURE_BOUNDARIES = (
    ("validation_before_dispatch", 11),
    ("provider_rejection", 11),
    ("known_no_dispatch", 11),
    ("ambiguous_dispatch", 11),
    ("before_visible_output", 11),
    ("after_visible_output", 11),
    ("malformed_stream_item", 11),
    ("tool_effect", 8),
    ("structured_input_suspension", 8),
    ("checkpoint_commit_failure", 11),
    ("outward_publication_failure", 11),
)
_PHASE0_FAILURE_SURFACES = (
    ("direct_sdk", 4),
    ("provider_adapter", 5),
    ("agent_sdk", 8),
    ("served_responses", 9),
    ("compact", 7),
    ("retrieve", 9),
    ("delete", 9),
    ("stream", 5),
    ("structured_input", 8),
)
_FAILURE_COUNT_VALUES = frozenset((0, 1))
_PUBLIC_MAPPING_VALUES = _frozen(
    "absent",
    "committed_unpublished",
    "input_required",
    "not_applicable",
)
_RETRY_DECISION_VALUES = _frozen(
    "bounded_if_proven_safe",
    "fenced",
    "never",
    "not_applicable",
    "reconcile_only",
    "resume_only",
)
_PARENT_STATE_VALUES = _frozen("not_applicable", "unchanged")
_PUBLIC_ERROR_VALUES = _frozen(
    "conversation_cancelled",
    "conversation_dispatch_ambiguous",
    "conversation_effect_boundary",
    "conversation_failed_after_output",
    "conversation_input_required",
    "conversation_limit_exceeded",
    "conversation_provider_failed",
    "conversation_provider_rejected",
    "conversation_publication_failed",
    "conversation_state_commit_failed",
    "conversation_stream_item_invalid",
    "conversation_transport_no_dispatch",
    "conversation_validation_failed",
    "not_applicable",
)
_RECONCILIATION_STATE_VALUES = _frozen(
    "none",
    "not_applicable",
    "pending",
    "quarantined",
    "required",
    "suspended",
)


class ConversationAcceptanceError(RuntimeError):
    """Report invalid or non-passing conversation evidence."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _Phase12LiveReceiptIdentity:
    """Identify one exact current live provider receipt."""

    provider_family: str
    profile: str
    revision: str
    structural_observations_digest: str

    @property
    def identity_digest(self) -> str:
        """Return the digest binding provider, profile, revision, and proof."""
        identity = {
            "profile": self.profile,
            "provider_family": self.provider_family,
            "revision": self.revision,
            "structural_observations_digest": (
                self.structural_observations_digest
            ),
        }
        return canonical_sha256(identity)

    @property
    def proof_id(self) -> str:
        """Return a deterministic identity- and full-digest-bound proof ID."""
        return (
            f"{_PHASE12_LIVE_PROOF_PREFIX}:identity-sha256:"
            f"{self.identity_digest}:structural-sha256:"
            f"{self.structural_observations_digest}"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceNode:
    """Store one lifecycle-aware acceptance node."""

    id: str
    category: str
    lifecycle: str
    active_from_phase: int
    requirement_ids: tuple[str, ...]
    node_id: str
    surface: str
    dimensions: dict[str, tuple[str, ...]]
    evidence_class: str


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceReplacement:
    """Store one reviewed append-only acceptance evidence replacement."""

    phase: int
    old_node_id: str
    replacement_node_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceManifest:
    """Store the validated conversation acceptance inventory."""

    path: Path
    current_phase: int
    nodes: tuple[AcceptanceNode, ...]
    replacements: tuple[AcceptanceReplacement, ...]

    def active_nodes(self, through_phase: int) -> tuple[AcceptanceNode, ...]:
        """Return active nodes introduced through one phase."""
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle == "active"
            and node.active_from_phase <= through_phase
        )

    def planned_nodes(self) -> tuple[AcceptanceNode, ...]:
        """Return all future planned nodes."""
        return tuple(
            node for node in self.nodes if node.lifecycle == "planned"
        )

    def ever_activated_nodes(
        self,
        through_phase: int,
    ) -> tuple[AcceptanceNode, ...]:
        """Return retained active and replaced records through one phase."""
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle in {"active", "replaced"}
            and node.active_from_phase <= through_phase
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class Requirement:
    """Store one ordinal normative requirement."""

    id: str
    normative_ordinal: int
    source_section: str
    normative_level: str
    paraphrase: str
    owner_phase: int
    production_artifact: str
    test_node_ids: tuple[str, ...]
    replacement_root_node_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureBoundary:
    """Store one failure boundary and its owning requirements."""

    id: str
    owner_phase: int
    requirement_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureSurface:
    """Store one public or runtime failure surface."""

    id: str
    owner_phase: int


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureCell:
    """Store one explicit failure-boundary and surface intersection."""

    id: str
    boundary_id: str
    surface_id: str
    applicability: str
    lifecycle: str
    active_from_phase: int
    evidence_node_id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureMatrix:
    """Store the complete failure-boundary Cartesian matrix."""

    boundaries: tuple[FailureBoundary, ...]
    surfaces: tuple[FailureSurface, ...]
    cells: tuple[FailureCell, ...]


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def fixture_root() -> Path:
    """Return the tracked conversation fixture directory."""
    return repository_root() / "tests" / "fixtures" / "conversation"


def default_manifest_path() -> Path:
    """Return the tracked acceptance manifest path."""
    return fixture_root() / "acceptance_manifest.phase11.json"


def companion_fixture_path(manifest_path: Path, stem: str) -> Path:
    """Return a phase-qualified companion beside an acceptance manifest."""
    name = manifest_path.name
    prefix = "acceptance_manifest"
    qualifier = ""
    if name.startswith(prefix) and name.endswith(".json"):
        qualifier = name[len(prefix) : -len(".json")]
    return manifest_path.parent / f"{stem}{qualifier}.json"


def load_manifest(path: Path) -> AcceptanceManifest:
    """Load and validate the lifecycle-aware acceptance manifest."""
    payload = _strict_mapping(path, "acceptance manifest")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "categories",
            "required_dimensions",
            "replacements",
            "activation_history",
            "nodes",
            "manifest_sha256",
        },
        "acceptance manifest",
    )
    _header(payload, "acceptance manifest")
    current_phase = _phase(payload.get("current_phase"), "current_phase")
    categories = _string_list(payload.get("categories"), "categories")
    if frozenset(categories) != _CATEGORIES or len(categories) != len(
        _CATEGORIES
    ):
        raise ConversationAcceptanceError(
            "acceptance categories differ from the required inventory"
        )
    required_dimensions = _required_dimensions(
        payload.get("required_dimensions")
    )
    raw_nodes = object_list(payload.get("nodes"), "acceptance nodes")
    if not raw_nodes:
        raise ConversationAcceptanceError("acceptance nodes must be non-empty")
    nodes = tuple(_acceptance_node(raw, current_phase) for raw in raw_nodes)
    _unique((node.id for node in nodes), "acceptance node ID")
    _unique((node.node_id for node in nodes), "pytest node ID")
    _validate_phase8_semantic_axes(nodes)
    if frozenset(node.category for node in nodes) != _CATEGORIES:
        raise ConversationAcceptanceError(
            "every acceptance category must own a node"
        )
    active_categories = frozenset(
        node.category for node in nodes if node.lifecycle == "active"
    )
    if active_categories != _CATEGORIES:
        raise ConversationAcceptanceError(
            "every acceptance category must have active Phase 0 evidence"
        )
    if current_phase < _MAX_PHASE and not any(
        node.lifecycle == "planned" for node in nodes
    ):
        raise ConversationAcceptanceError(
            "future acceptance nodes must remain explicitly planned"
        )
    for phase in range(current_phase + 1):
        if not any(
            node.lifecycle in {"active", "replaced"}
            and node.active_from_phase == phase
            for node in nodes
        ):
            raise ConversationAcceptanceError(
                f"implemented acceptance inventory has a gap at phase {phase}"
            )
    replacements = _validate_replacements(
        payload.get("replacements"), nodes, current_phase
    )
    activation_history = _validate_activation_history(
        payload.get("activation_history"), nodes, current_phase
    )
    _validate_replacement_transitions(
        replacements,
        nodes,
        activation_history,
    )
    observed_dimensions = {
        name: frozenset(
            value for node in nodes for value in node.dimensions[name]
        )
        for name in _DIMENSIONS
    }
    if observed_dimensions != required_dimensions:
        raise ConversationAcceptanceError(
            "acceptance nodes do not cover every mandatory dimension"
        )
    active_dimensions = {
        name: frozenset(
            value
            for node in nodes
            if node.lifecycle == "active"
            for value in node.dimensions[name]
        )
        for name in _DIMENSIONS
    }
    if active_dimensions != required_dimensions:
        raise ConversationAcceptanceError(
            "active evidence lacks an explicit disposition for mandatory "
            "dimensions"
        )
    if not any(
        node.lifecycle == "active"
        and node.evidence_class == "pre_dispatch_rejection"
        and node.surface == "served_responses"
        for node in nodes
    ):
        raise ConversationAcceptanceError(
            "active served dimensions require executable pre-dispatch "
            "rejection evidence"
        )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if payload.get("manifest_sha256") != canonical_sha256(canonical):
        raise ConversationAcceptanceError(
            "acceptance manifest digest is invalid"
        )
    observed_phase0_nodes = tuple(
        (
            node.id,
            node.node_id,
            node.active_from_phase,
            node.evidence_class,
        )
        for node in nodes
        if node.active_from_phase == 0
    )
    if observed_phase0_nodes != _PHASE0_NODE_INVENTORY:
        raise ConversationAcceptanceError(
            "Phase 0 acceptance node tuple inventory drifted"
        )
    _validate_node_phase_anchors(raw_nodes, nodes, current_phase)
    return AcceptanceManifest(
        path=path,
        current_phase=current_phase,
        nodes=nodes,
        replacements=replacements,
    )


def _validate_phase8_semantic_axes(
    nodes: tuple[AcceptanceNode, ...],
) -> None:
    """Reject Phase 8 evidence metadata that contradicts its test boundary."""
    by_node_id = {
        node.node_id: node
        for node in nodes
        if node.lifecycle == "active" and node.active_from_phase == 8
    }
    for node_id in _PHASE8_TOOL_EVIDENCE_NODES:
        node = by_node_id.get(node_id)
        if node is not None and node.dimensions["execution"] != ("one_tool",):
            raise ConversationAcceptanceError(
                "Phase 8 tool evidence must declare one_tool execution"
            )
    for node_id in _PHASE8_DURABLE_EVIDENCE_NODES:
        node = by_node_id.get(node_id)
        if node is not None and node.dimensions["local_retention"] != (
            "durable_local",
        ):
            raise ConversationAcceptanceError(
                "Phase 8 PostgreSQL evidence must declare durable_local "
                "retention"
            )
    for node_id in _PHASE8_FRESH_PROCESS_EVIDENCE_NODES:
        node = by_node_id.get(node_id)
        if node is not None and node.dimensions["lifecycle"] != (
            "fresh_process",
        ):
            raise ConversationAcceptanceError(
                "Phase 8 fresh-worker evidence must declare fresh_process "
                "lifecycle"
            )
    for node_id in _PHASE8_MULTI_AGENT_EVIDENCE_NODES:
        node = by_node_id.get(node_id)
        if node is not None and node.dimensions["execution"] != (
            "multiple_agents_lanes",
        ):
            raise ConversationAcceptanceError(
                "Phase 8 multi-agent evidence must declare "
                "multiple_agents_lanes execution"
            )


def load_requirements(
    path: Path,
    manifest: AcceptanceManifest,
    *,
    repo_root: Path,
) -> tuple[Requirement, ...]:
    """Load and validate all ordinal normative requirements."""
    payload = _strict_mapping(path, "requirements traceability")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "source_sections",
            "normative_occurrence_count",
            "requirements",
            "catalog_sha256",
        },
        "requirements traceability",
    )
    _header(payload, "requirements traceability")
    sections = _string_list(payload.get("source_sections"), "source sections")
    if sections != tuple(str(section) for section in range(9, 27)):
        raise ConversationAcceptanceError(
            "source sections must be the contiguous 9 through 26 inventory"
        )
    if payload.get("normative_occurrence_count") != _NORMATIVE_OCCURRENCES:
        raise ConversationAcceptanceError(
            "normative occurrence count must be exactly 144"
        )
    raw_requirements = object_list(payload.get("requirements"), "requirements")
    if len(raw_requirements) != _NORMATIVE_OCCURRENCES:
        raise ConversationAcceptanceError(
            "requirement catalog does not contain every normative occurrence"
        )
    requirements = tuple(
        _requirement(raw, repo_root=repo_root) for raw in raw_requirements
    )
    if manifest.current_phase >= 11:
        requirements = _apply_phase11_requirement_evidence(
            path.parent / "requirements_evidence.phase11.json",
            requirements,
        )
    expected_ids = tuple(
        f"CONV-N-{ordinal:03d}"
        for ordinal in range(1, _NORMATIVE_OCCURRENCES + 1)
    )
    if tuple(requirement.id for requirement in requirements) != expected_ids:
        raise ConversationAcceptanceError(
            "requirement IDs must be stable and ordinal"
        )
    if tuple(
        requirement.normative_ordinal for requirement in requirements
    ) != tuple(range(1, _NORMATIVE_OCCURRENCES + 1)):
        raise ConversationAcceptanceError(
            "normative occurrence ordinals must be contiguous"
        )
    node_by_id = {node.node_id: node for node in manifest.nodes}
    reverse: dict[str, set[str]] = {}
    for node in manifest.nodes:
        for requirement_id in node.requirement_ids:
            reverse.setdefault(requirement_id, set()).add(node.node_id)
    replacement_by_old = {
        replacement.old_node_id: replacement.replacement_node_ids
        for replacement in manifest.replacements
    }
    for requirement in requirements:
        for node_id in (
            *requirement.replacement_root_node_ids,
            *requirement.test_node_ids,
        ):
            owner_node = node_by_id.get(node_id)
            if owner_node is None:
                raise ConversationAcceptanceError(
                    f"requirement references an unknown node: {node_id}"
                )
            if owner_node.active_from_phase != requirement.owner_phase:
                raise ConversationAcceptanceError(
                    "requirement owner phase differs from its exact nodes: "
                    f"{requirement.id}"
                )
        allowed = _replacement_closure(
            requirement.replacement_root_node_ids,
            replacement_by_old,
            node_by_id,
            requirement.id,
        )
        if allowed != reverse.get(requirement.id, set()):
            raise ConversationAcceptanceError(
                "requirement ownership is outside its reviewed replacement "
                f"chain: {requirement.id}"
            )
        if requirement.owner_phase <= manifest.current_phase:
            leaves = allowed - set(replacement_by_old)
            exact_phase11_evidence = (
                requirement.owner_phase != 11
                or leaves == set(requirement.test_node_ids)
            )
            if (
                not leaves
                or not exact_phase11_evidence
                or any(
                    node_by_id[node_id].lifecycle != "active"
                    for node_id in leaves
                )
            ):
                evidence_label = (
                    "exact active replacement evidence"
                    if requirement.owner_phase == 11
                    else "active replacement-chain evidence"
                )
                raise ConversationAcceptanceError(
                    f"implemented requirement lacks {evidence_label}: "
                    f"{requirement.id}"
                )
    if payload.get("catalog_sha256") != canonical_sha256(raw_requirements):
        raise ConversationAcceptanceError(
            "requirement catalog digest is invalid"
        )
    if canonical_sha256(raw_requirements) != _PHASE0_REQUIREMENTS_SHA256:
        raise ConversationAcceptanceError(
            "requirement catalog differs from the independent Phase 0 anchor"
        )
    return requirements


def _apply_phase11_requirement_evidence(
    path: Path,
    requirements: tuple[Requirement, ...],
) -> tuple[Requirement, ...]:
    """Apply exact Phase 11 leaves without rewriting the frozen catalog."""
    payload = _strict_mapping(path, "Phase 11 requirement evidence")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "requirements",
            "evidence_sha256",
        },
        "Phase 11 requirement evidence",
    )
    _header(payload, "Phase 11 requirement evidence")
    if payload.get("phase") != 11:
        raise ConversationAcceptanceError(
            "Phase 11 requirement evidence has the wrong phase"
        )
    raw_records = object_list(
        payload.get("requirements"), "Phase 11 requirement records"
    )
    expected_ids = tuple(
        f"CONV-N-{ordinal:03d}" for ordinal in range(118, 131)
    )
    replacements: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    observed_ids: list[str] = []
    for raw_record in raw_records:
        record = mapping(raw_record, "Phase 11 requirement record")
        _exact_keys(
            record,
            {"id", "test_node_ids", "replacement_root_node_ids"},
            "Phase 11 requirement record",
        )
        identifier = _nonempty_string(
            record.get("id"), "Phase 11 requirement ID"
        )
        observed_ids.append(identifier)
        test_nodes = _string_list(
            record.get("test_node_ids"), "Phase 11 exact test nodes"
        )
        roots = _string_list(
            record.get("replacement_root_node_ids"),
            "Phase 11 replacement roots",
        )
        if not test_nodes or not roots:
            raise ConversationAcceptanceError(
                "Phase 11 requirement evidence must be non-empty"
            )
        for node_id in (*test_nodes, *roots):
            _test_node(node_id)
        _unique(test_nodes, "Phase 11 exact test node")
        _unique(roots, "Phase 11 replacement root")
        replacements[identifier] = (test_nodes, roots)
    if tuple(observed_ids) != expected_ids:
        raise ConversationAcceptanceError(
            "Phase 11 requirement evidence must cover ordinals 118-130"
        )
    if payload.get("evidence_sha256") != canonical_sha256(raw_records):
        raise ConversationAcceptanceError(
            "Phase 11 requirement evidence digest is invalid"
        )
    return tuple(
        (
            replace(
                requirement,
                test_node_ids=replacements[requirement.id][0],
                replacement_root_node_ids=replacements[requirement.id][1],
            )
            if requirement.id in replacements
            else requirement
        )
        for requirement in requirements
    )


def _replacement_closure(
    roots: tuple[str, ...],
    replacement_by_old: dict[str, tuple[str, ...]],
    node_by_id: dict[str, AcceptanceNode],
    requirement_id: str,
) -> set[str]:
    """Return roots and reviewed descendants owning one requirement."""
    observed = set(roots)
    pending = list(roots)
    while pending:
        current = pending.pop()
        for target in replacement_by_old.get(current, ()):
            if (
                requirement_id in node_by_id[target].requirement_ids
                and target not in observed
            ):
                observed.add(target)
                pending.append(target)
    return observed


def load_failure_matrix(
    path: Path,
    *,
    manifest: AcceptanceManifest,
    requirement_ids: frozenset[str],
) -> FailureMatrix:
    """Load and validate the complete explicit failure matrix."""
    payload = _strict_mapping(path, "failure matrix")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "observation_window",
            "tool_effect_scope",
            "boundaries",
            "surfaces",
            "cells",
            "matrix_sha256",
        },
        "failure matrix",
    )
    _header(payload, "failure matrix")
    if payload.get("current_phase") != manifest.current_phase:
        raise ConversationAcceptanceError(
            "failure matrix and acceptance phases differ"
        )
    _nonempty_string(payload.get("observation_window"), "observation window")
    _nonempty_string(payload.get("tool_effect_scope"), "tool-effect scope")
    raw_boundaries = object_list(
        payload.get("boundaries"), "failure boundaries"
    )
    boundaries = tuple(
        _failure_boundary(raw, requirement_ids) for raw in raw_boundaries
    )
    raw_surfaces = object_list(payload.get("surfaces"), "failure surfaces")
    surfaces = tuple(_failure_surface(raw) for raw in raw_surfaces)
    if not boundaries or not surfaces:
        raise ConversationAcceptanceError(
            "failure boundaries and surfaces must be non-empty"
        )
    _unique((item.id for item in boundaries), "failure boundary ID")
    _unique((item.id for item in surfaces), "failure surface ID")
    phase0_boundary_count = len(_PHASE0_FAILURE_BOUNDARIES)
    if (
        tuple(
            (item.id, item.owner_phase)
            for item in boundaries[:phase0_boundary_count]
        )
        != _PHASE0_FAILURE_BOUNDARIES
    ):
        raise ConversationAcceptanceError(
            "failure boundary inventory differs from the Phase 0 anchor"
        )
    phase0_surface_count = len(_PHASE0_FAILURE_SURFACES)
    if (
        tuple(
            (item.id, item.owner_phase)
            for item in surfaces[:phase0_surface_count]
        )
        != _PHASE0_FAILURE_SURFACES
    ):
        raise ConversationAcceptanceError(
            "failure surface inventory differs from the Phase 0 anchor"
        )
    boundary_by_id = {item.id: item for item in boundaries}
    surface_by_id = {item.id: item for item in surfaces}
    node_by_id = {node.node_id: node for node in manifest.nodes}
    raw_cells = object_list(payload.get("cells"), "failure cells")
    cells = tuple(
        _failure_cell(
            raw,
            boundary_by_id=boundary_by_id,
            surface_by_id=surface_by_id,
            node_by_id=node_by_id,
            current_phase=manifest.current_phase,
        )
        for raw in raw_cells
    )
    _unique((cell.id for cell in cells), "failure cell ID")
    observed = {(cell.boundary_id, cell.surface_id) for cell in cells}
    expected = {
        (boundary.id, surface.id)
        for boundary in boundaries
        for surface in surfaces
    }
    if len(cells) != len(expected) or observed != expected:
        raise ConversationAcceptanceError(
            "failure matrix must cover the complete Cartesian inventory"
        )
    if not any(
        cell.applicability == "applicable" for cell in cells
    ) or not any(cell.applicability == "not_applicable" for cell in cells):
        raise ConversationAcceptanceError(
            "failure matrix needs applicable and explicit non-applicable cells"
        )
    if manifest.current_phase >= 8:
        _validate_phase8_failure_evidence(cells, raw_cells=raw_cells)
    canonical = {
        key: value for key, value in payload.items() if key != "matrix_sha256"
    }
    if payload.get("matrix_sha256") != canonical_sha256(canonical):
        raise ConversationAcceptanceError("failure matrix digest is invalid")
    _validate_failure_structure_anchors(
        payload,
        raw_boundaries,
        raw_surfaces,
        raw_cells,
        manifest.current_phase,
    )
    return FailureMatrix(
        boundaries=boundaries,
        surfaces=surfaces,
        cells=cells,
    )


def _validate_phase8_failure_evidence(
    cells: tuple[FailureCell, ...],
    *,
    raw_cells: list[object],
) -> None:
    """Pin Phase 8 runtime and schema failure evidence independently."""
    by_id = {cell.id: cell for cell in cells}
    observed = {
        cell_id: by_id[cell_id].evidence_node_id
        for cell_id in _PHASE8_FAILURE_EVIDENCE
    }
    if observed != _PHASE8_FAILURE_EVIDENCE:
        raise ConversationAcceptanceError(
            "Phase 8 failure cells differ from exact executable evidence"
        )
    raw_by_id = {
        _nonempty_string(
            mapping(raw, "failure cell").get("id"),
            "failure cell ID",
        ): mapping(raw, "failure cell")
        for raw in raw_cells
    }
    agent_tool = raw_by_id["tool_effect--agent_sdk"]
    semantics = {
        field: agent_tool.get(field)
        for field in _PHASE8_AGENT_TOOL_FAILURE_SEMANTICS
    }
    if semantics != _PHASE8_AGENT_TOOL_FAILURE_SEMANTICS:
        raise ConversationAcceptanceError(
            "Phase 8 agent tool failure semantics contradict its durable "
            "negative evidence"
        )
    schema_observed = {
        cell_id: by_id[cell_id].evidence_node_id
        for cell_id in _PHASE8_SCHEMA_FAILURE_EVIDENCE
    }
    if schema_observed != _PHASE8_SCHEMA_FAILURE_EVIDENCE:
        raise ConversationAcceptanceError(
            "Phase 8 schema failure cells differ from zero-effect evidence"
        )


def _validate_failure_structure_anchors(
    payload: dict[str, object],
    raw_boundaries: list[object],
    raw_surfaces: list[object],
    raw_cells: list[object],
    current_phase: int,
) -> None:
    """Validate append-only failure topology apart from mutable evidence."""
    if current_phase >= 9:
        anchors = _PHASE9_CORRECTED_FAILURE_STRUCTURE_BY_PHASE
    elif current_phase >= 8:
        anchors = _PHASE8_CORRECTED_FAILURE_STRUCTURE_BY_PHASE
    else:
        anchors = _FAILURE_STRUCTURE_BY_PHASE
    _require_phase_anchor_keys(
        anchors,
        current_phase,
        "failure structure",
    )
    previous = (0, 0, 0)
    for phase in range(current_phase + 1):
        boundary_count, surface_count, cell_count, expected_sha256 = anchors[
            phase
        ]
        counts = (boundary_count, surface_count, cell_count)
        available = (
            len(raw_boundaries),
            len(raw_surfaces),
            len(raw_cells),
        )
        if any(
            before > after for before, after in zip(previous, counts)
        ) or any(count > maximum for count, maximum in zip(counts, available)):
            raise ConversationAcceptanceError(
                "failure structure phase anchors are not append-only"
            )
        normalized_cells = [
            {
                key: value
                for key, value in mapping(raw, "failure cell").items()
                if key not in {"evidence_node_id", "lifecycle"}
            }
            for raw in raw_cells[:cell_count]
        ]
        structure = {
            "observation_window": payload.get("observation_window"),
            "tool_effect_scope": payload.get("tool_effect_scope"),
            "boundaries": raw_boundaries[:boundary_count],
            "surfaces": raw_surfaces[:surface_count],
            "cells": normalized_cells,
        }
        if canonical_sha256(structure) != expected_sha256:
            raise ConversationAcceptanceError(
                "failure structure differs from its immutable phase "
                f"anchor at phase {phase}"
            )
        previous = counts
    if previous != (
        len(raw_boundaries),
        len(raw_surfaces),
        len(raw_cells),
    ):
        raise ConversationAcceptanceError(
            "failure structure has unanchored appended payload"
        )


def verify_acceptance(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
    execute: bool = True,
) -> AcceptanceManifest:
    """Validate all fixtures and execute selected active nodes."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    if not _MIN_PHASE <= through_phase <= manifest.current_phase:
        raise ConversationAcceptanceError(
            "through-phase must be implemented by the current manifest"
        )
    fixtures = path.parent
    requirements = load_requirements(
        fixtures / "requirements_traceability.json",
        manifest,
        repo_root=root,
    )
    if manifest.current_phase >= 11:
        _validate_phase12_live_proof_resolution(root)
        _validate_phase12_traceability_candidate(root, manifest)
    requirement_ids = frozenset(item.id for item in requirements)
    load_failure_matrix(
        companion_fixture_path(path, "failure_matrix"),
        manifest=manifest,
        requirement_ids=requirement_ids,
    )
    _validate_threat_model(
        companion_fixture_path(path, "threat_model"),
        manifest=manifest,
        requirement_ids=requirement_ids,
    )
    _validate_integrated_fixtures(
        fixtures,
        current_phase=manifest.current_phase,
    )
    _validate_type_manifest(
        fixtures,
        manifest.current_phase,
        root,
        acceptance_path=path,
    )
    verify_gate_source_isolation(root, manifest)
    nodes = manifest.active_nodes(through_phase)
    if not nodes:
        raise ConversationAcceptanceError(
            "the selected acceptance inventory has no active nodes"
        )
    if execute:
        with TemporaryDirectory(
            prefix="avalan-conversation-acceptance-"
        ) as temporary:
            try:
                execute_pytest_nodes(
                    root,
                    tuple(node.node_id for node in nodes),
                    junit_path=Path(temporary) / "pytest.xml",
                    expected_evidence={
                        node.node_id: node.evidence_class for node in nodes
                    },
                    inherited_names=(POSTGRESQL_TEST_DSN_ENV,),
                )
            except ContractGateError as exc:
                raise ConversationAcceptanceError(str(exc)) from exc
    return manifest


def verify_gate_source_isolation(
    root: Path,
    manifest: AcceptanceManifest,
) -> None:
    """Reject Markdown dependencies from tracked gate and active tests."""
    transitions_by_phase = (
        _phase5_provider_transitions(root),
        _phase6_provider_transitions(root),
        _phase7_provider_transitions(root),
        _phase8_provider_transitions(root),
        _phase9_provider_transitions(root),
        _phase10_provider_transitions(root),
        (
            _phase11_provider_transitions(root)
            if manifest.current_phase >= 11
            or (root / _PHASE11_PROVIDER_TRANSITION_PATH).is_file()
            else {}
        ),
        (
            _phase12_provider_transitions(root)
            if (root / _PHASE12_PROVIDER_TRANSITION_PATH).is_file()
            else {}
        ),
        (
            _phase13_provider_transitions(root)
            if (root / _PHASE13_PROVIDER_TRANSITION_PATH).is_file()
            else {}
        ),
    )
    transition_chains: dict[
        str,
        list[tuple[int, str, int, str]],
    ] = {}
    for transitions in transitions_by_phase:
        for relative, transition in transitions.items():
            chain = transition_chains.setdefault(relative, [])
            if chain and chain[-1][2:] != transition[:2]:
                raise ConversationAcceptanceError(
                    "reviewed provider transition chain is discontinuous: "
                    f"{relative}"
                )
            chain.append(transition)
    for relative, chain in transition_chains.items():
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise ConversationAcceptanceError(
                f"reviewed provider transition target is missing: {relative}"
            )
        target_bytes = target.read_bytes()
        final_size, final_sha256 = chain[-1][2:]
        if (
            len(target_bytes) != final_size
            or sha256(target_bytes).hexdigest() != final_sha256
        ):
            raise ConversationAcceptanceError(
                f"reviewed provider transition target changed: {relative}"
            )
    _require_phase_anchor_keys(
        _ACTIVE_SOURCE_SHA256_BY_PHASE,
        manifest.current_phase,
        "active source",
    )
    for phase in range(manifest.current_phase + 1):
        observed = {
            node.node_id.split("::", 1)[0]
            for node in manifest.nodes
            if node.lifecycle in {"active", "replaced"}
            and node.active_from_phase == phase
        }
        expected = _ACTIVE_SOURCE_SHA256_BY_PHASE[phase]
        if observed != set(expected):
            raise ConversationAcceptanceError(
                "active acceptance source inventory differs from its "
                f"phase anchor at phase {phase}"
            )
        for relative, expected_sha256 in expected.items():
            source = root / relative
            if not source.is_file():
                raise ConversationAcceptanceError(
                    f"active acceptance source is missing: {relative}"
                )
            current_sha256 = expected_sha256
            chain = transition_chains.get(relative, [])
            if chain:
                matching = tuple(
                    index
                    for index, transition in enumerate(chain)
                    if transition[1] == expected_sha256
                )
                if len(matching) != 1:
                    raise ConversationAcceptanceError(
                        "reviewed provider transition source differs from "
                        f"its historical anchor: {relative}"
                    )
                current_sha256 = chain[-1][3]
            if sha256(source.read_bytes()).hexdigest() != current_sha256:
                raise ConversationAcceptanceError(
                    f"active acceptance source digest changed: {relative}"
                )
    candidates = {
        root
        / "scripts"
        / "contract_startup"
        / "avalan_contract_gate_plugin.py",
        root / "scripts" / "contract_startup" / "sitecustomize.py",
        root / "scripts" / "verify_conversation_acceptance.py",
        root / "scripts" / "verify_conversation_types.py",
        root / "scripts" / "run_conversation_contract_gate.py",
        root / "scripts" / "contract_gate.py",
        *(
            root / node.node_id.split("::", 1)[0]
            for node in manifest.nodes
            if node.lifecycle in {"active", "replaced"}
        ),
    }
    for path in candidates:
        if not path.is_file():
            raise ConversationAcceptanceError(
                f"tracked gate source is missing: {path.relative_to(root)}"
            )
        try:
            tree = parse_python(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise ConversationAcceptanceError(
                f"cannot audit tracked gate source: {path.relative_to(root)}"
            ) from exc
        markdown_literals = tuple(
            node.value
            for node in walk(tree)
            if isinstance(node, Constant)
            and isinstance(node.value, str)
            and node.value.casefold().endswith(_MARKDOWN_SUFFIX)
        )
        if markdown_literals:
            raise ConversationAcceptanceError(
                "tracked gate sources must not depend on Markdown inputs: "
                f"{path.relative_to(root)}"
            )


def _acceptance_node(raw: object, current_phase: int) -> AcceptanceNode:
    item = mapping(raw, "acceptance node")
    expected = {
        "id",
        "category",
        "lifecycle",
        "active_from_phase",
        "requirement_ids",
        "node_id",
        "surface",
        "provider",
        "provider_mode",
        "local_retention",
        "transport",
        "execution",
        "turn_topology",
        "reasoning_context",
        "compaction",
        "scenario_lifecycle",
        "failure",
        "authority",
        "limit",
        "evidence_class",
    }
    _exact_keys(item, expected, "acceptance node")
    category = _nonempty_string(item.get("category"), "node category")
    if category not in _CATEGORIES:
        raise ConversationAcceptanceError(f"invalid node category: {category}")
    phase = _phase(item.get("active_from_phase"), "active_from_phase")
    lifecycle = _nonempty_string(item.get("lifecycle"), "node lifecycle")
    if lifecycle not in {"active", "planned", "replaced"} or (
        (phase > current_phase) != (lifecycle == "planned")
    ):
        raise ConversationAcceptanceError(
            "node lifecycle disagrees with active_from_phase"
        )
    requirement_ids = _string_list(
        item.get("requirement_ids"), "node requirement IDs"
    )
    if not requirement_ids:
        raise ConversationAcceptanceError(
            "acceptance node must cover at least one requirement"
        )
    _unique(requirement_ids, "node requirement ID")
    for requirement_id in requirement_ids:
        if _REQUIREMENT_PATTERN.fullmatch(requirement_id) is None:
            raise ConversationAcceptanceError(
                f"invalid requirement ID: {requirement_id}"
            )
    dimensions = {
        name: _dimension_values(
            item.get("scenario_lifecycle" if name == "lifecycle" else name),
            name,
        )
        for name in _DIMENSIONS
    }
    evidence_class = _nonempty_string(
        item.get("evidence_class"), "evidence class"
    )
    if evidence_class not in _EVIDENCE_CLASSES:
        raise ConversationAcceptanceError(
            f"invalid evidence class: {evidence_class}"
        )
    return AcceptanceNode(
        id=_nonempty_string(item.get("id"), "node ID"),
        category=category,
        lifecycle=lifecycle,
        active_from_phase=phase,
        requirement_ids=requirement_ids,
        node_id=_test_node(item.get("node_id")),
        surface=_nonempty_string(item.get("surface"), "node surface"),
        dimensions=dimensions,
        evidence_class=evidence_class,
    )


def _required_dimensions(raw: object) -> dict[str, frozenset[str]]:
    item = mapping(raw, "required dimensions")
    _exact_keys(item, set(_DIMENSIONS), "required dimensions")
    observed = {
        name: frozenset(_string_list(item.get(name), f"{name} dimension"))
        for name in _DIMENSIONS
    }
    if observed != _DIMENSIONS:
        raise ConversationAcceptanceError(
            "mandatory dimension inventory changed"
        )
    return observed


def _dimension_values(raw: object, name: str) -> tuple[str, ...]:
    values = _string_list(raw, f"{name} dimension values")
    if not values or not set(values) <= _DIMENSIONS[name]:
        raise ConversationAcceptanceError(
            f"node has invalid or empty {name} dimensions"
        )
    _unique(values, f"node {name} dimension")
    return values


def _requirement(raw: object, *, repo_root: Path) -> Requirement:
    item = mapping(raw, "requirement")
    phase = _phase(item.get("owner_phase"), "requirement owner phase")
    expected_keys = {
        "id",
        "normative_ordinal",
        "source_section",
        "normative_level",
        "paraphrase",
        "owner_phase",
        "production_artifact",
        "test_node_ids",
    }
    _exact_keys(
        item,
        expected_keys,
        "requirement",
    )
    identifier = _nonempty_string(item.get("id"), "requirement ID")
    ordinal = _positive_int(item.get("normative_ordinal"), "normative ordinal")
    if identifier != f"CONV-N-{ordinal:03d}":
        raise ConversationAcceptanceError("requirement ID and ordinal differ")
    section = _nonempty_string(item.get("source_section"), "source section")
    try:
        major = int(section.split(".", 1)[0])
    except ValueError as exc:
        raise ConversationAcceptanceError(
            f"invalid source section: {section}"
        ) from exc
    if not 9 <= major <= 26:
        raise ConversationAcceptanceError(f"invalid source section: {section}")
    level = _nonempty_string(item.get("normative_level"), "normative level")
    if level not in {"MUST", "MUST NOT"}:
        raise ConversationAcceptanceError(f"invalid normative level: {level}")
    paraphrase = _nonempty_string(
        item.get("paraphrase"), "requirement paraphrase"
    )
    artifact = _relative_path(
        item.get("production_artifact"), "production artifact"
    )
    if phase == 0 and not (repo_root / artifact).is_file():
        raise ConversationAcceptanceError(
            f"active production artifact is missing: {artifact}"
        )
    nodes = _string_list(item.get("test_node_ids"), "requirement test nodes")
    if not nodes:
        raise ConversationAcceptanceError(
            f"requirement has no exact test nodes: {identifier}"
        )
    for node in nodes:
        _test_node(node)
    _unique(nodes, "requirement test node")
    return Requirement(
        id=identifier,
        normative_ordinal=ordinal,
        source_section=section,
        normative_level=level,
        paraphrase=paraphrase,
        owner_phase=phase,
        production_artifact=artifact,
        test_node_ids=nodes,
        replacement_root_node_ids=nodes,
    )


def _failure_boundary(
    raw: object,
    requirement_ids: frozenset[str],
) -> FailureBoundary:
    item = mapping(raw, "failure boundary")
    _exact_keys(
        item,
        {"id", "description", "owner_phase", "requirement_ids"},
        "failure boundary",
    )
    _nonempty_string(item.get("description"), "boundary description")
    owned = _string_list(item.get("requirement_ids"), "boundary requirements")
    if not owned or not set(owned) <= requirement_ids:
        raise ConversationAcceptanceError(
            "failure boundary references unknown requirements"
        )
    return FailureBoundary(
        id=_nonempty_string(item.get("id"), "boundary ID"),
        owner_phase=_phase(item.get("owner_phase"), "boundary owner phase"),
        requirement_ids=owned,
    )


def _failure_surface(raw: object) -> FailureSurface:
    item = mapping(raw, "failure surface")
    _exact_keys(
        item,
        {"id", "description", "owner_phase"},
        "failure surface",
    )
    _nonempty_string(item.get("description"), "surface description")
    return FailureSurface(
        id=_nonempty_string(item.get("id"), "surface ID"),
        owner_phase=_phase(item.get("owner_phase"), "surface owner phase"),
    )


def _failure_cell(
    raw: object,
    *,
    boundary_by_id: dict[str, FailureBoundary],
    surface_by_id: dict[str, FailureSurface],
    node_by_id: dict[str, AcceptanceNode],
    current_phase: int,
) -> FailureCell:
    item = mapping(raw, "failure cell")
    _exact_keys(
        item,
        {
            "id",
            "boundary_id",
            "surface_id",
            "applicability",
            "lifecycle",
            "active_from_phase",
            "evidence_node_id",
            "expected_dispatch_count",
            "visible_output_count",
            "tool_effect_count",
            "checkpoint_commit_count",
            "public_mapping",
            "retry_decision",
            "parent_state",
            "public_error",
            "reconciliation_state",
            "rationale",
        },
        "failure cell",
    )
    boundary_id = _nonempty_string(item.get("boundary_id"), "boundary ID")
    surface_id = _nonempty_string(item.get("surface_id"), "surface ID")
    boundary = boundary_by_id.get(boundary_id)
    surface = surface_by_id.get(surface_id)
    if boundary is None or surface is None:
        raise ConversationAcceptanceError(
            "failure cell references an unknown boundary or surface"
        )
    identifier = _nonempty_string(item.get("id"), "failure cell ID")
    if identifier != f"{boundary_id}--{surface_id}":
        raise ConversationAcceptanceError(
            "failure cell ID differs from its coordinates"
        )
    applicability = _nonempty_string(
        item.get("applicability"), "cell applicability"
    )
    if applicability not in {"applicable", "not_applicable"}:
        raise ConversationAcceptanceError(
            "failure cell applicability is invalid"
        )
    phase = _phase(item.get("active_from_phase"), "cell active phase")
    lifecycle = _nonempty_string(item.get("lifecycle"), "cell lifecycle")
    if lifecycle not in {"active", "planned"}:
        raise ConversationAcceptanceError(
            "failure cell lifecycle must be active or planned"
        )
    evidence_node_id = _test_node(item.get("evidence_node_id"))
    evidence = node_by_id.get(evidence_node_id)
    if evidence is None:
        raise ConversationAcceptanceError("failure cell evidence is missing")
    if lifecycle == "active" and (
        phase > current_phase
        or evidence.lifecycle == "planned"
        or evidence.active_from_phase > current_phase
    ):
        raise ConversationAcceptanceError(
            "active failure cell evidence is not active"
        )
    if (
        lifecycle == "planned"
        and phase <= current_phase
        and (
            evidence.lifecycle != "planned"
            or evidence.active_from_phase <= current_phase
        )
    ):
        raise ConversationAcceptanceError(
            "planned failure cell evidence is not future evidence"
        )
    counts = tuple(
        _nonnegative_int(item.get(field), f"failure cell {field}")
        for field in (
            "expected_dispatch_count",
            "visible_output_count",
            "tool_effect_count",
            "checkpoint_commit_count",
        )
    )
    if any(value not in _FAILURE_COUNT_VALUES for value in counts):
        raise ConversationAcceptanceError(
            "failure cell counts must use the closed zero-or-one inventory"
        )
    states = tuple(
        _nonempty_string(item.get(field), f"failure cell {field}")
        for field in (
            "public_mapping",
            "retry_decision",
            "parent_state",
            "public_error",
            "reconciliation_state",
        )
    )
    allowed_states = (
        _PUBLIC_MAPPING_VALUES,
        _RETRY_DECISION_VALUES,
        _PARENT_STATE_VALUES,
        _PUBLIC_ERROR_VALUES,
        _RECONCILIATION_STATE_VALUES,
    )
    if any(
        value not in allowed
        for value, allowed in zip(states, allowed_states, strict=True)
    ):
        raise ConversationAcceptanceError(
            "failure cell uses a state outside the closed Phase 0 inventory"
        )
    _nonempty_string(item.get("rationale"), "failure cell rationale")
    if applicability == "not_applicable":
        if phase != 0 or any(counts) or set(states) != {"not_applicable"}:
            raise ConversationAcceptanceError(
                "non-applicable failure cells need exact Phase 0 evidence"
            )
    elif phase < max(boundary.owner_phase, surface.owner_phase):
        raise ConversationAcceptanceError(
            "applicable failure cell activates before its owners"
        )
    return FailureCell(
        id=identifier,
        boundary_id=boundary_id,
        surface_id=surface_id,
        applicability=applicability,
        lifecycle=lifecycle,
        active_from_phase=phase,
        evidence_node_id=evidence_node_id,
    )


def _validate_threat_model(
    path: Path,
    *,
    manifest: AcceptanceManifest,
    requirement_ids: frozenset[str],
) -> None:
    payload = _strict_mapping(path, "threat model")
    expected_payload_keys = {
        "schema_version",
        "feature",
        "current_phase",
        "assets",
        "trust_boundaries",
        "threats",
        "threat_model_sha256",
    }
    if manifest.current_phase >= 11:
        expected_payload_keys.add("inherited_traceability")
    _exact_keys(payload, expected_payload_keys, "threat model")
    _header(payload, "threat model")
    if payload.get("current_phase") != manifest.current_phase:
        raise ConversationAcceptanceError(
            "threat model and acceptance phases differ"
        )
    assets = _string_list(payload.get("assets"), "threat assets")
    trust_boundaries = _string_list(
        payload.get("trust_boundaries"), "trust boundaries"
    )
    if not assets or not trust_boundaries:
        raise ConversationAcceptanceError(
            "threat assets and trust boundaries must be non-empty"
        )
    node_ids = {
        node.node_id
        for node in manifest.nodes
        if node.lifecycle in {"active", "replaced"}
    }
    active_node_ids = {
        node.node_id
        for node in manifest.nodes
        if node.lifecycle == "active"
        and node.active_from_phase <= manifest.current_phase
    }
    observed: list[str] = []
    threat_items: dict[str, Mapping[str, object]] = {}
    raw_threats = object_list(payload.get("threats"), "threats")
    phase10_threat_count = _THREAT_STRUCTURE_BY_PHASE[10][2]
    for index, raw in enumerate(raw_threats):
        item = mapping(raw, "threat")
        hardened = index >= phase10_threat_count
        expected_keys = {
            "id",
            "asset",
            "actor",
            "boundary",
            "attack",
            "controls",
            "requirement_ids",
            "owner_phase",
            "lifecycle",
            "evidence_node_ids",
        }
        if hardened:
            expected_keys.update(
                {
                    "control_owners",
                    "positive_evidence_node_ids",
                    "negative_evidence_node_ids",
                    "operator_detection",
                    "incident_response",
                    "residual_risk",
                }
            )
        _exact_keys(
            item,
            expected_keys,
            "threat",
        )
        identifier = _nonempty_string(item.get("id"), "threat ID")
        observed.append(identifier)
        threat_items[identifier] = item
        for field in ("asset", "actor", "boundary", "attack"):
            _nonempty_string(item.get(field), f"threat {field}")
        controls = _string_list(item.get("controls"), "threat controls")
        if not controls:
            raise ConversationAcceptanceError(
                "threat controls must be non-empty"
            )
        owned = _string_list(
            item.get("requirement_ids"), "threat requirements"
        )
        if not owned or not set(owned) <= requirement_ids:
            raise ConversationAcceptanceError(
                "threat references unknown requirements"
            )
        owner_phase = _phase(item.get("owner_phase"), "threat owner phase")
        if item.get("lifecycle") != "active":
            raise ConversationAcceptanceError(
                "Phase 0 threat entries must be active"
            )
        evidence = _string_list(
            item.get("evidence_node_ids"), "threat evidence nodes"
        )
        if not evidence or not set(evidence) <= node_ids:
            raise ConversationAcceptanceError(
                "threat evidence must use active exact nodes"
            )
        if hardened:
            if owner_phase != 11:
                raise ConversationAcceptanceError(
                    "Phase 11 hardening threats must be owned by Phase 11"
                )
            _validate_threat_traceability(
                item,
                identifier=identifier,
                controls=controls,
                active_node_ids=active_node_ids,
                expected_evidence=frozenset(evidence),
            )
    if manifest.current_phase >= 11:
        required_inherited = {
            identifier
            for identifier, item in threat_items.items()
            if identifier in set(observed[:phase10_threat_count])
            and (
                item.get("owner_phase") == 11
                or bool(
                    set(
                        _string_list(
                            item.get("requirement_ids"),
                            "threat requirements",
                        )
                    )
                    & _PHASE11_REQUIREMENT_IDS
                )
            )
        }
        inherited = object_list(
            payload.get("inherited_traceability"),
            "inherited threat traceability",
        )
        inherited_ids: list[str] = []
        for raw_trace in inherited:
            trace = mapping(raw_trace, "inherited threat traceability")
            _exact_keys(
                trace,
                {
                    "threat_id",
                    "control_owners",
                    "positive_evidence_node_ids",
                    "negative_evidence_node_ids",
                    "operator_detection",
                    "incident_response",
                    "residual_risk",
                },
                "inherited threat traceability",
            )
            threat_id = _nonempty_string(
                trace.get("threat_id"), "inherited threat ID"
            )
            inherited_ids.append(threat_id)
            source = threat_items.get(threat_id)
            if source is None:
                raise ConversationAcceptanceError(
                    "inherited traceability references an unknown threat"
                )
            _validate_threat_traceability(
                trace,
                identifier=threat_id,
                controls=_string_list(
                    source.get("controls"), "threat controls"
                ),
                active_node_ids=active_node_ids,
                expected_evidence=None,
            )
        _unique(inherited_ids, "inherited threat traceability ID")
        if set(inherited_ids) != required_inherited:
            raise ConversationAcceptanceError(
                "Phase 11 inherited threat traceability is incomplete"
            )
    phase0_threat_count = len(_THREAT_IDS)
    _unique(observed, "threat ID")
    if (
        len(observed) < phase0_threat_count
        or frozenset(observed[:phase0_threat_count]) != _THREAT_IDS
    ):
        raise ConversationAcceptanceError(
            "threat inventory is incomplete or duplicated"
        )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "threat_model_sha256"
    }
    if payload.get("threat_model_sha256") != canonical_sha256(canonical):
        raise ConversationAcceptanceError("threat model digest is invalid")
    _validate_threat_structure_anchors(
        assets,
        trust_boundaries,
        raw_threats,
        manifest.current_phase,
    )


def _validate_threat_traceability(
    item: Mapping[str, object],
    *,
    identifier: str,
    controls: tuple[str, ...],
    active_node_ids: set[str],
    expected_evidence: frozenset[str] | None,
) -> None:
    """Validate one complete operational control and evidence mapping."""
    for value in (
        identifier,
        *controls,
        _nonempty_string(
            item.get("operator_detection"),
            "threat operator detection",
        ),
        _nonempty_string(
            item.get("incident_response"),
            "threat incident response",
        ),
        _nonempty_string(
            item.get("residual_risk"),
            "threat residual risk",
        ),
    ):
        _reject_threat_placeholder(value)
    raw_owners = object_list(
        item.get("control_owners"), "threat control owners"
    )
    owners: dict[str, str] = {}
    for raw_owner in raw_owners:
        owner = mapping(raw_owner, "threat control owner")
        _exact_keys(owner, {"control_id", "owner"}, "threat control owner")
        control_id = _nonempty_string(
            owner.get("control_id"), "threat control ID"
        )
        owner_name = _nonempty_string(
            owner.get("owner"), "threat control owner"
        )
        _reject_threat_placeholder(control_id)
        _reject_threat_placeholder(owner_name)
        if control_id in owners:
            raise ConversationAcceptanceError(
                "threat control ownership is duplicated"
            )
        owners[control_id] = owner_name
    if set(owners) != set(controls):
        raise ConversationAcceptanceError(
            "every threat control needs one explicit owner"
        )
    positive = _string_list(
        item.get("positive_evidence_node_ids"),
        "positive threat evidence",
    )
    negative = _string_list(
        item.get("negative_evidence_node_ids"),
        "negative threat evidence",
    )
    _unique(positive, "positive threat evidence node")
    _unique(negative, "negative threat evidence node")
    if (
        not positive
        or not negative
        or set(positive) & set(negative)
        or not set(positive) <= active_node_ids
        or not set(negative) <= active_node_ids
        or (
            expected_evidence is not None
            and expected_evidence != frozenset((*positive, *negative))
        )
    ):
        raise ConversationAcceptanceError(
            "Phase 11 threats need distinct active positive and negative "
            "evidence"
        )


def _reject_threat_placeholder(value: str) -> None:
    """Reject placeholder traceability text from active threat evidence."""
    normalized = value.casefold()
    markers = (
        "placeholder",
        "positive-evidence",
        "negative-evidence",
        "production-control",
        "todo",
        "tbd",
    )
    if any(marker in normalized for marker in markers):
        raise ConversationAcceptanceError(
            "active threat traceability contains placeholder text"
        )


def _validate_threat_structure_anchors(
    assets: tuple[str, ...],
    trust_boundaries: tuple[str, ...],
    raw_threats: list[object],
    current_phase: int,
) -> None:
    """Validate cumulative append-only threat structure snapshots."""
    _require_phase_anchor_keys(
        _THREAT_STRUCTURE_BY_PHASE,
        current_phase,
        "threat structure",
    )
    previous = (0, 0, 0)
    for phase in range(current_phase + 1):
        asset_count, boundary_count, threat_count, expected_sha256 = (
            _THREAT_STRUCTURE_BY_PHASE[phase]
        )
        counts = (asset_count, boundary_count, threat_count)
        available = (len(assets), len(trust_boundaries), len(raw_threats))
        if any(
            before > after for before, after in zip(previous, counts)
        ) or any(count > maximum for count, maximum in zip(counts, available)):
            raise ConversationAcceptanceError(
                "threat structure phase anchors are not append-only"
            )
        structure = {
            "assets": assets[:asset_count],
            "trust_boundaries": trust_boundaries[:boundary_count],
            "threats": raw_threats[:threat_count],
        }
        if canonical_sha256(structure) != expected_sha256:
            raise ConversationAcceptanceError(
                "threat structure differs from its immutable phase "
                f"anchor at phase {phase}"
            )
        previous = counts
    if previous != (len(assets), len(trust_boundaries), len(raw_threats)):
        raise ConversationAcceptanceError(
            "threat structure has unanchored appended payload"
        )


def _validate_integrated_fixtures(
    fixtures: Path,
    *,
    current_phase: int,
) -> None:
    """Validate integrated contract/provider fixtures once they arrive."""
    names: tuple[str, ...] = _ACTIVE_INTEGRATED_FIXTURES
    if current_phase >= 5:
        names += (_PHASE5_PROVIDER_CONFORMANCE,)
    if current_phase >= 6:
        names += (_PHASE6_PROVIDER_CONFORMANCE,)
    if current_phase >= 7:
        names += (_PHASE7_PROVIDER_CONFORMANCE,)
    if current_phase >= 8:
        names += (_PHASE8_PROVIDER_CONFORMANCE,)
    authoritative = fixtures.resolve() == fixture_root().resolve()
    if authoritative:
        missing = tuple(
            name for name in names if not (fixtures / name).is_file()
        )
        if missing:
            raise ConversationAcceptanceError(
                f"integrated Phase 0 fixtures are missing: {missing}"
            )
        _validate_phase0_provider_byte_anchors(fixtures.parents[2])
    for name in names:
        path = fixtures / name
        if not path.exists():
            continue
        payload = _strict_mapping(path, f"integrated fixture {name}")
        if name == "contract_decisions.json":
            _validate_contract_decisions(payload)
        elif name == "deterministic_fixtures.json":
            _validate_deterministic_fixtures(payload)
        elif name == "provider_contract.json":
            _validate_provider_contract(payload)
        elif name == "provider_conformance.json":
            _validate_provider_conformance(payload)
        elif name == _PHASE5_PROVIDER_CONFORMANCE:
            base = _strict_mapping(
                fixtures / "provider_conformance.json",
                "Phase 0 provider conformance",
            )
            _validate_phase5_provider_conformance(payload, base)
        elif name == _PHASE6_PROVIDER_CONFORMANCE:
            phase5_path = fixtures / _PHASE5_PROVIDER_CONFORMANCE
            phase5 = _strict_mapping(
                phase5_path,
                "Phase 5 provider conformance",
            )
            _validate_phase6_provider_conformance(
                payload,
                phase5,
                phase5_path=phase5_path,
            )
        elif name == _PHASE7_PROVIDER_CONFORMANCE:
            phase6_path = fixtures / _PHASE6_PROVIDER_CONFORMANCE
            phase6 = _strict_mapping(
                phase6_path,
                "Phase 6 provider conformance",
            )
            _validate_phase7_provider_conformance(
                payload,
                phase6,
                phase6_path=phase6_path,
            )
        else:
            phase7_path = fixtures / _PHASE7_PROVIDER_CONFORMANCE
            phase7 = _strict_mapping(
                phase7_path,
                "Phase 7 provider conformance",
            )
            _validate_phase8_provider_conformance(
                payload,
                phase7,
                phase7_path=phase7_path,
            )


def _validate_phase0_provider_byte_anchors(root: Path) -> None:
    """Validate exact provider fixture, test, and production source bytes."""
    transition_maps = (
        _phase5_provider_transitions(root),
        _phase6_provider_transitions(root),
        _phase7_provider_transitions(root),
        _phase8_provider_transitions(root),
        _phase9_provider_transitions(root),
        _phase10_provider_transitions(root),
        (
            _phase11_provider_transitions(root)
            if (root / _PHASE11_PROVIDER_TRANSITION_PATH).is_file()
            else {}
        ),
        (
            _phase12_provider_transitions(root)
            if (root / _PHASE12_PROVIDER_TRANSITION_PATH).is_file()
            else {}
        ),
        (
            _phase13_provider_transitions(root)
            if (root / _PHASE13_PROVIDER_TRANSITION_PATH).is_file()
            else {}
        ),
    )
    for relative, (
        expected_size,
        expected_sha256,
    ) in _PHASE0_PROVIDER_BYTE_ANCHORS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ConversationAcceptanceError(
                f"anchored Phase 0 provider source is missing: {relative}"
            )
        current_size = expected_size
        current_sha256 = expected_sha256
        for transitions in transition_maps:
            transitioned = transitions.get(relative)
            if transitioned is None:
                continue
            from_size, from_sha256, to_size, to_sha256 = transitioned
            if from_size != current_size or from_sha256 != current_sha256:
                raise ConversationAcceptanceError(
                    "reviewed provider transition differs from its Phase 0 "
                    f"anchor: {relative}"
                )
            current_size = to_size
            current_sha256 = to_sha256
        payload = path.read_bytes()
        if (
            len(payload) != current_size
            or sha256(payload).hexdigest() != current_sha256
        ):
            raise ConversationAcceptanceError(
                f"anchored Phase 0 provider source changed: {relative}"
            )


def _phase5_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Return the exact reviewed Phase 5 provider byte transitions."""
    path = root / _PHASE5_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE5_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 5 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 5 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 5
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by") != "phase5-native-provider-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 5 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if observed_digest != canonical_sha256(canonical):
        raise ConversationAcceptanceError(
            "Phase 5 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    evidence = _string_list(
        payload.get("evidence_node_ids"),
        "provider transition evidence nodes",
    )
    if evidence != (
        (
            "tests/conversation/native_openai_provider_test.py::"
            "test_native_openai_two_turn_replay_is_exact_and_private"
        ),
        (
            "tests/conversation/native_openai_provider_validation_test.py::"
            "test_new_input_freezing_and_legacy_facade_is_stateless"
        ),
    ):
        raise ConversationAcceptanceError(
            "Phase 5 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    for raw in object_list(
        payload.get("transitions"),
        "provider byte transitions",
    ):
        entry = mapping(raw, "provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "provider byte transition",
        )
        relative = _nonempty_string(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"), "transition source digest"
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"), "transition target digest"
        )
        if (
            type(from_size) is not int
            or from_size <= 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "provider byte transition is invalid"
            )
        transitions[relative] = (
            from_size,
            from_sha256,
            to_size,
            to_sha256,
        )
        if (to_size, to_sha256) != _PHASE5_PROVIDER_TARGET_BYTE_ANCHORS.get(
            relative
        ):
            raise ConversationAcceptanceError(
                "provider byte transition target differs from its "
                f"independent anchor: {relative}"
            )
    if set(transitions) != set(_PHASE5_PROVIDER_TARGET_BYTE_ANCHORS):
        raise ConversationAcceptanceError(
            "Phase 5 provider transition inventory is invalid"
        )
    return transitions


def _phase6_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Validate exact reviewed Phase 6 stored-provider source transitions."""
    path = root / _PHASE6_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE6_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 6 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 6 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 6
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by") != "phase6-native-stored-provider-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 6 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if observed_digest != canonical_sha256(canonical):
        raise ConversationAcceptanceError(
            "Phase 6 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    evidence = _string_list(
        payload.get("evidence_node_ids"),
        "Phase 6 provider transition evidence nodes",
    )
    if evidence != (
        (
            "tests/conversation/openai_stored_e2e_test.py::"
            "test_stored_public_upstream_alias_never_commits_pgsql_mapping"
        ),
        (
            "tests/conversation/native_openai_stored_provider_test.py::"
            "test_unproven_lifecycle_and_conversion_fail_closed"
        ),
        (
            "tests/conversation/native_openai_stored_provider_test.py::"
            "test_execution_definition_bytes_fail_before_dispatch"
        ),
        (
            "tests/conversation/openai_stored_e2e_test.py::"
            "test_stored_quarantine_survives_pgsql_capacity_and_restart"
        ),
        (
            "tests/conversation/native_openai_stored_provider_test.py::"
            "test_known_provider_rejection_releases_idempotency_fence"
        ),
        (
            "tests/conversation/pgsql_conformance_test.py::"
            "test_pgsql_ambiguous_dispatch_reconciliation_survives_restart_race"
        ),
        (
            "tests/conversation/openai_stored_e2e_test.py::"
            "test_retired_runtime_continues_after_pgsql_store_restart"
        ),
        (
            "tests/conversation/openai_stored_e2e_test.py::"
            "test_stream_close_quarantine_survives_pgsql_restart_capacity"
        ),
        (
            "tests/conversation/openai_stored_e2e_test.py::"
            "test_generated_checkpoint_alias_fails_pgsql_codec_and_sdk"
        ),
        (
            "tests/conversation/openai_stored_e2e_test.py::"
            "test_retrieve_execution_drift_fails_over_tcp_and_pgsql"
        ),
    ):
        raise ConversationAcceptanceError(
            "Phase 6 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    for raw in object_list(
        payload.get("transitions"),
        "Phase 6 provider byte transitions",
    ):
        entry = mapping(raw, "Phase 6 provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "Phase 6 provider byte transition",
        )
        relative = _nonempty_string(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"),
            "transition source digest",
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"),
            "transition target digest",
        )
        if (
            type(from_size) is not int
            or from_size < 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "Phase 6 provider byte transition is invalid"
            )
        transition = (from_size, from_sha256, to_size, to_sha256)
        if transition != _PHASE6_PROVIDER_SOURCE_BYTE_ANCHORS.get(relative):
            raise ConversationAcceptanceError(
                "Phase 6 provider transition differs from its independent "
                f"anchor: {relative}"
            )
        transitions[relative] = transition
    if set(transitions) != set(_PHASE6_PROVIDER_SOURCE_BYTE_ANCHORS):
        raise ConversationAcceptanceError(
            "Phase 6 provider transition inventory is invalid"
        )
    return transitions


def _phase7_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Validate exact reviewed native compaction source transitions."""
    path = root / _PHASE7_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE7_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 7 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 7 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 7
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by")
        != "phase7-native-compaction-provider-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 7 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if observed_digest != canonical_sha256(canonical):
        raise ConversationAcceptanceError(
            "Phase 7 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    evidence = _string_list(
        payload.get("evidence_node_ids"),
        "Phase 7 provider transition evidence nodes",
    )
    if evidence != (
        (
            "tests/conversation/native_openai_compaction_test.py::"
            "test_native_inline_latest_replay_and_standalone_wire"
        ),
        (
            "tests/conversation/native_openai_compaction_test.py::"
            "test_streamed_compaction_commits_complete_done_items"
        ),
        (
            "tests/conversation/compaction_e2e_test.py::"
            "test_long_stateless_inline_compaction_restarts_at_latest_boundary"
        ),
        (
            "tests/conversation/compaction_e2e_test.py::"
            "test_stored_inline_compaction_uses_only_immediate_upstream_parent"
        ),
        (
            "tests/conversation/compaction_e2e_test.py::"
            "test_tool_cycles_across_two_boundaries_keep_exact_final_order"
        ),
        (
            "tests/conversation/compaction_e2e_test.py::"
            "test_standalone_explicit_fork_restarts_and_original_parent_branches"
        ),
    ):
        raise ConversationAcceptanceError(
            "Phase 7 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    for raw in object_list(
        payload.get("transitions"),
        "Phase 7 provider byte transitions",
    ):
        entry = mapping(raw, "Phase 7 provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "Phase 7 provider byte transition",
        )
        relative = _nonempty_string(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"),
            "transition source digest",
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"),
            "transition target digest",
        )
        if (
            type(from_size) is not int
            or from_size <= 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "Phase 7 provider byte transition is invalid"
            )
        transition = (from_size, from_sha256, to_size, to_sha256)
        if transition != _PHASE7_PROVIDER_SOURCE_BYTE_ANCHORS.get(relative):
            raise ConversationAcceptanceError(
                "Phase 7 provider transition differs from its independent "
                f"anchor: {relative}"
            )
        transitions[relative] = transition
    if set(transitions) != set(_PHASE7_PROVIDER_SOURCE_BYTE_ANCHORS):
        raise ConversationAcceptanceError(
            "Phase 7 provider transition inventory is invalid"
        )
    return transitions


def _phase8_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Validate exact reviewed agent and tool source transitions."""
    path = root / _PHASE8_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE8_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 8 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 8 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 8
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by") != "phase8-agent-tool-provider-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 8 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if (
        observed_digest != canonical_sha256(canonical)
        or observed_digest != _PHASE8_PROVIDER_TRANSITION_CANONICAL_SHA256
    ):
        raise ConversationAcceptanceError(
            "Phase 8 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    evidence = _string_list(
        payload.get("evidence_node_ids"),
        "Phase 8 provider transition evidence nodes",
    )
    if evidence != (
        (
            "tests/conversation/native_openai_provider_test.py::"
            "test_native_function_cycles_use_the_coordinator_ledger"
        ),
        (
            "tests/conversation/native_openai_provider_validation_test.py::"
            "test_native_function_tool_rejects_invalid_schema_arguments_"
            "before_effect"
        ),
        (
            "tests/conversation/native_openai_provider_validation_test.py::"
            "test_native_function_tool_rejects_nonlocal_schema_before_effect"
        ),
        (
            "tests/conversation/native_openai_provider_validation_test.py::"
            "test_native_function_tool_persists_only_validated_arguments"
        ),
        (
            "tests/conversation/native_openai_provider_validation_test.py::"
            "test_native_output_byte_limit_precedes_tool_effect_and_commit"
        ),
        (
            "tests/conversation/native_openai_provider_test.py::"
            "test_agent_turn_propagates_typed_structured_input_suspension"
        ),
        (
            "tests/conversation/native_openai_stored_provider_test.py::"
            "test_stored_tool_cycle_uses_only_immediate_id_and_tool_output"
        ),
        (
            "tests/conversation/agent_integration_e2e_test.py::"
            "test_parent_tool_effect_failure_fences_unsafe_retry"
        ),
        (
            "tests/conversation/agent_integration_e2e_test.py::"
            "test_parent_two_children_persist_isolation_and_restart"
        ),
        (
            "tests/conversation/agent_integration_e2e_test.py::"
            "test_child_merge_rejects_wrong_provider_and_model_binding"
        ),
        (
            "tests/conversation/agent_integration_contract_test.py::"
            "test_parent_kind_policy_rejects_agent_coordinator_and_store_"
            "bypasses"
        ),
        (
            "tests/conversation/agent_integration_contract_test.py::"
            "test_agent_conversation_surfaces_are_explicit_and_fail_closed"
        ),
        (
            "tests/conversation/agent_integration_pgsql_test.py::"
            "test_pgsql_tool_boundaries_recover_without_duplicate_effect"
        ),
        (
            "tests/interaction/stores/conversation_atomic_pgsql_test.py::"
            "test_fresh_worker_applies_atomic_conversation_answer_once"
        ),
    ):
        raise ConversationAcceptanceError(
            "Phase 8 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    empty_sha256 = sha256(b"").hexdigest()
    for raw in object_list(
        payload.get("transitions"),
        "Phase 8 provider byte transitions",
    ):
        entry = mapping(raw, "Phase 8 provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "Phase 8 provider byte transition",
        )
        relative = _relative_path(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"),
            "transition source digest",
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"),
            "transition target digest",
        )
        if (
            type(from_size) is not int
            or from_size < 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or (from_size == 0 and from_sha256 != empty_sha256)
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "Phase 8 provider byte transition is invalid"
            )
        transitions[relative] = (
            from_size,
            from_sha256,
            to_size,
            to_sha256,
        )
    if len(transitions) != 44:
        raise ConversationAcceptanceError(
            "Phase 8 provider transition inventory is invalid"
        )
    return transitions


def _phase9_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Validate exact served Responses source transitions."""
    path = root / _PHASE9_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE9_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 9 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 9 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 9
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by") != "phase9-served-responses-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 9 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if (
        observed_digest != canonical_sha256(canonical)
        or observed_digest != _PHASE9_PROVIDER_TRANSITION_CANONICAL_SHA256
    ):
        raise ConversationAcceptanceError(
            "Phase 9 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    evidence = _string_list(
        payload.get("evidence_node_ids"),
        "Phase 9 provider transition evidence nodes",
    )
    if evidence != (
        (
            "tests/conversation/server_stored_e2e_test.py::"
            "test_normative_server_stored_contract"
        ),
    ):
        raise ConversationAcceptanceError(
            "Phase 9 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    empty_sha256 = sha256(b"").hexdigest()
    for raw in object_list(
        payload.get("transitions"),
        "Phase 9 provider byte transitions",
    ):
        entry = mapping(raw, "Phase 9 provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "Phase 9 provider byte transition",
        )
        relative = _relative_path(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"),
            "transition source digest",
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"),
            "transition target digest",
        )
        if (
            type(from_size) is not int
            or from_size < 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or (from_size == 0 and from_sha256 != empty_sha256)
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "Phase 9 provider byte transition is invalid"
            )
        transitions[relative] = (
            from_size,
            from_sha256,
            to_size,
            to_sha256,
        )
    if len(transitions) != 15:
        raise ConversationAcceptanceError(
            "Phase 9 provider transition inventory is invalid"
        )
    return transitions


def _phase10_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Validate exact caller-held Responses source transitions."""
    path = root / _PHASE10_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE10_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 10 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 10 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 10
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by") != "phase10-stateless-responses-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 10 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if (
        observed_digest != canonical_sha256(canonical)
        or observed_digest != _PHASE10_PROVIDER_TRANSITION_CANONICAL_SHA256
    ):
        raise ConversationAcceptanceError(
            "Phase 10 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    evidence = _string_list(
        payload.get("evidence_node_ids"),
        "Phase 10 provider transition evidence nodes",
    )
    if evidence != (
        (
            "tests/conversation/server_stateless_e2e_test.py::"
            "test_normative_server_stateless_contract"
        ),
    ):
        raise ConversationAcceptanceError(
            "Phase 10 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    empty_sha256 = sha256(b"").hexdigest()
    for raw in object_list(
        payload.get("transitions"),
        "Phase 10 provider byte transitions",
    ):
        entry = mapping(raw, "Phase 10 provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "Phase 10 provider byte transition",
        )
        relative = _relative_path(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"),
            "transition source digest",
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"),
            "transition target digest",
        )
        if (
            type(from_size) is not int
            or from_size < 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or (from_size == 0 and from_sha256 != empty_sha256)
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "Phase 10 provider byte transition is invalid"
            )
        transitions[relative] = (
            from_size,
            from_sha256,
            to_size,
            to_sha256,
        )
    if len(transitions) != 17:
        raise ConversationAcceptanceError(
            "Phase 10 provider transition inventory is invalid"
        )
    return transitions


def _phase11_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Validate exact hardening integration source transitions."""
    path = root / _PHASE11_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE11_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 11 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 11 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 11
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by") != "phase11-hardening-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 11 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if (
        observed_digest != canonical_sha256(canonical)
        or observed_digest != _PHASE11_PROVIDER_TRANSITION_CANONICAL_SHA256
    ):
        raise ConversationAcceptanceError(
            "Phase 11 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    evidence = _string_list(
        payload.get("evidence_node_ids"),
        "Phase 11 provider transition evidence nodes",
    )
    if evidence != (
        (
            "tests/conversation/server_stored_e2e_test.py::"
            "test_phase11_served_dispatch_installs_required_hardening"
        ),
    ):
        raise ConversationAcceptanceError(
            "Phase 11 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    for raw in object_list(
        payload.get("transitions"),
        "Phase 11 provider byte transitions",
    ):
        entry = mapping(raw, "Phase 11 provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "Phase 11 provider byte transition",
        )
        relative = _relative_path(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"), "transition source digest"
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"), "transition target digest"
        )
        if (
            type(from_size) is not int
            or from_size <= 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "Phase 11 provider byte transition is invalid"
            )
        transitions[relative] = (
            from_size,
            from_sha256,
            to_size,
            to_sha256,
        )
    if len(transitions) != 5:
        raise ConversationAcceptanceError(
            "Phase 11 provider transition inventory is invalid"
        )
    return transitions


def _phase12_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Validate activation and PostgreSQL compatibility source transitions."""
    path = root / _PHASE12_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE12_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 12 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 12 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 12
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by") != "phase12-activation-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 12 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if (
        observed_digest != canonical_sha256(canonical)
        or observed_digest != _PHASE12_PROVIDER_TRANSITION_CANONICAL_SHA256
    ):
        raise ConversationAcceptanceError(
            "Phase 12 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    evidence = _string_list(
        payload.get("evidence_node_ids"),
        "Phase 12 provider transition evidence nodes",
    )
    if evidence != (
        (
            "tests/conversation/activation_test.py::"
            "test_native_provider_dispatch_requires_exact_active_registry"
        ),
        (
            "tests/conversation/activation_test.py::"
            "test_stored_dispatch_and_lifecycle_use_registry_boundaries"
        ),
        (
            "tests/conversation/full_matrix_e2e_test.py::"
            "test_required_matrix_cross_product"
        ),
    ):
        raise ConversationAcceptanceError(
            "Phase 12 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    empty_sha256 = sha256(b"").hexdigest()
    for raw in object_list(
        payload.get("transitions"),
        "Phase 12 provider byte transitions",
    ):
        entry = mapping(raw, "Phase 12 provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "Phase 12 provider byte transition",
        )
        relative = _relative_path(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"), "transition source digest"
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"), "transition target digest"
        )
        if (
            type(from_size) is not int
            or from_size < 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or (from_size == 0 and from_sha256 != empty_sha256)
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "Phase 12 provider byte transition is invalid"
            )
        transitions[relative] = (
            from_size,
            from_sha256,
            to_size,
            to_sha256,
        )
    if len(transitions) != 15:
        raise ConversationAcceptanceError(
            "Phase 12 provider transition inventory is invalid"
        )
    return transitions


def _phase13_provider_transitions(
    root: Path,
) -> dict[str, tuple[int, str, int, str]]:
    """Validate exact reviewed provider retry source transitions."""
    path = root / _PHASE13_PROVIDER_TRANSITION_PATH
    if not path.is_file():
        path = repository_root() / _PHASE13_PROVIDER_TRANSITION_PATH
    payload = _strict_mapping(path, "Phase 13 provider transition")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "kind",
            "reviewed_by",
            "reason",
            "transitions",
            "evidence_node_ids",
            "canonical_sha256",
        },
        "Phase 13 provider transition",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 13
        or payload.get("kind") != "reviewed_provider_source_transition"
        or payload.get("reviewed_by")
        != "phase13-provider-retry-tool-image-and-transport-review"
    ):
        raise ConversationAcceptanceError(
            "Phase 13 provider transition header is invalid"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if (
        observed_digest != canonical_sha256(canonical)
        or observed_digest != _PHASE13_PROVIDER_TRANSITION_CANONICAL_SHA256
    ):
        raise ConversationAcceptanceError(
            "Phase 13 provider transition digest is invalid"
        )
    _nonempty_string(payload.get("reason"), "provider transition reason")
    if (
        _string_list(
            payload.get("evidence_node_ids"),
            "Phase 13 provider transition evidence nodes",
        )
        != _PHASE13_PROVIDER_EVIDENCE_NODES
    ):
        raise ConversationAcceptanceError(
            "Phase 13 provider transition evidence is invalid"
        )
    transitions: dict[str, tuple[int, str, int, str]] = {}
    for raw in object_list(
        payload.get("transitions"),
        "Phase 13 provider byte transitions",
    ):
        entry = mapping(raw, "Phase 13 provider byte transition")
        _exact_keys(
            entry,
            {
                "path",
                "from_size",
                "from_sha256",
                "to_size",
                "to_sha256",
            },
            "Phase 13 provider byte transition",
        )
        relative = _relative_path(entry.get("path"), "transition path")
        from_size = entry.get("from_size")
        to_size = entry.get("to_size")
        from_sha256 = _nonempty_string(
            entry.get("from_sha256"), "transition source digest"
        )
        to_sha256 = _nonempty_string(
            entry.get("to_sha256"), "transition target digest"
        )
        if (
            type(from_size) is not int
            or from_size <= 0
            or type(to_size) is not int
            or to_size <= 0
            or len(from_sha256) != 64
            or len(to_sha256) != 64
            or relative in transitions
        ):
            raise ConversationAcceptanceError(
                "Phase 13 provider byte transition is invalid"
            )
        transition = (from_size, from_sha256, to_size, to_sha256)
        if (to_size, to_sha256) != _PHASE13_PROVIDER_TARGET_BYTE_ANCHORS.get(
            relative
        ):
            raise ConversationAcceptanceError(
                "Phase 13 provider transition target differs from its "
                f"independent anchor: {relative}"
            )
        transitions[relative] = transition
    if set(transitions) != set(_PHASE13_PROVIDER_TARGET_BYTE_ANCHORS):
        raise ConversationAcceptanceError(
            "Phase 13 provider transition inventory is invalid"
        )
    return transitions


def _phase12_sha256(value: object, label: str) -> str:
    """Return one exact lowercase SHA-256 value."""
    digest = _nonempty_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ConversationAcceptanceError(f"{label} is not a SHA-256 digest")
    return digest


def _phase12_parse_live_proof_id(proof_id: str) -> tuple[str, str]:
    """Return the identity and referenced structural digests in one proof."""
    parts = proof_id.split(":")
    if (
        len(parts) != 5
        or parts[0] != _PHASE12_LIVE_PROOF_PREFIX
        or parts[1] != "identity-sha256"
        or parts[3] != "structural-sha256"
    ):
        raise ConversationAcceptanceError(
            "Phase 12 activation live proof format is invalid"
        )
    return (
        _phase12_sha256(parts[2], "Phase 12 live proof identity digest"),
        _phase12_sha256(parts[4], "Phase 12 live proof structural digest"),
    )


def _phase12_scoped_digest(
    payload: dict[str, object],
    label: str,
) -> str:
    """Validate and return one Phase 12 whole-object canonical digest."""
    digest = mapping(payload.get("canonical_digest"), f"{label} digest")
    _exact_keys(
        digest,
        {"algorithm", "encoding", "scope", "value"},
        f"{label} digest",
    )
    if (
        digest.get("algorithm") != "sha256"
        or digest.get("encoding")
        != "utf-8 canonical JSON with sorted keys and compact separators"
        or digest.get("scope")
        != "all top-level fields except canonical_digest"
    ):
        raise ConversationAcceptanceError(
            f"{label} digest metadata is invalid"
        )
    unsigned = dict(payload)
    unsigned.pop("canonical_digest", None)
    observed = _phase12_sha256(digest.get("value"), f"{label} digest")
    if observed != canonical_sha256(unsigned):
        raise ConversationAcceptanceError(f"{label} digest is invalid")
    return observed


def _phase12_anchored_payload(
    root: Path,
    relative: str,
    *,
    expected_byte_sha256: str,
    expected_canonical_sha256: str,
    label: str,
) -> tuple[dict[str, object], bytes, str]:
    """Load one source-anchored Phase 12 evidence object."""
    path = root / relative
    try:
        payload_bytes = path.read_bytes()
    except OSError as exc:
        raise ConversationAcceptanceError(
            f"cannot read {label} bytes"
        ) from exc
    if sha256(payload_bytes).hexdigest() != expected_byte_sha256:
        raise ConversationAcceptanceError(f"{label} byte anchor is invalid")
    payload = _strict_mapping(path, label)
    canonical = _phase12_scoped_digest(payload, label)
    if canonical != expected_canonical_sha256:
        raise ConversationAcceptanceError(
            f"{label} canonical anchor is invalid"
        )
    return payload, payload_bytes, canonical


def _phase12_validate_review_signature(
    activation: dict[str, object],
) -> None:
    """Validate the activation decision's content-digest review signature."""
    signature = mapping(
        activation.get("review_signature"),
        "Phase 12 activation review signature",
    )
    _exact_keys(
        signature,
        {"algorithm", "encoding", "scope", "value"},
        "Phase 12 activation review signature",
    )
    if (
        signature.get("algorithm") != "sha256"
        or signature.get("encoding")
        != "utf-8 canonical JSON with sorted keys and compact separators"
        or signature.get("scope")
        != "all top-level fields except review_signature and canonical_digest"
    ):
        raise ConversationAcceptanceError(
            "Phase 12 activation review signature metadata is invalid"
        )
    unsigned = dict(activation)
    unsigned.pop("review_signature", None)
    unsigned.pop("canonical_digest", None)
    observed = _phase12_sha256(
        signature.get("value"),
        "Phase 12 activation review signature",
    )
    if observed != canonical_sha256(unsigned):
        raise ConversationAcceptanceError(
            "Phase 12 activation review signature is invalid"
        )


def _phase12_live_receipt_identities(
    live_results: dict[str, object],
) -> tuple[_Phase12LiveReceiptIdentity, ...]:
    """Return exact identities for every current complete live receipt."""
    native = mapping(
        live_results.get("native_openai_attempt"),
        "Phase 12 native OpenAI result",
    )
    native_execution = mapping(
        native.get("matrix_execution"),
        "Phase 12 native OpenAI matrix execution",
    )
    if (
        native_execution.get("state")
        != "inactive_account_credit_exhausted_before_inference"
        or native.get("live_capability_receipt") is not False
    ):
        raise ConversationAcceptanceError(
            "Phase 12 native OpenAI quota blocker drifted"
        )
    execution_order = _string_list(
        live_results.get("execution_order"),
        "Phase 12 live execution order",
    )
    if len(execution_order) != len(_PHASE12_LIVE_CASES) or frozenset(
        execution_order
    ) != frozenset(_PHASE12_LIVE_CASES):
        raise ConversationAcceptanceError(
            "Phase 12 live execution order is invalid"
        )
    azure = mapping(
        live_results.get("azure_openai_matrix"),
        "Phase 12 Azure OpenAI matrix",
    )
    identities: list[_Phase12LiveReceiptIdentity] = []
    for raw in object_list(
        azure.get("results"),
        "Phase 12 Azure OpenAI results",
    ):
        profile = mapping(raw, "Phase 12 Azure OpenAI profile")
        if "tracked_cli_receipt" not in profile:
            continue
        receipt = mapping(
            profile.get("tracked_cli_receipt"),
            "Phase 12 tracked CLI receipt",
        )
        provider_family = _nonempty_string(
            receipt.get("provider_family"),
            "Phase 12 receipt provider family",
        )
        model_or_deployment = _nonempty_string(
            receipt.get("model_or_deployment"),
            "Phase 12 receipt profile",
        )
        revision = _nonempty_string(
            receipt.get("model_or_deployment_revision"),
            "Phase 12 receipt revision",
        )
        structural_digest = _phase12_sha256(
            receipt.get("structural_observations_digest"),
            "Phase 12 receipt structural digest",
        )
        if (
            provider_family != "azure_openai"
            or model_or_deployment != profile.get("deployment")
            or revision != profile.get("deployment_revision")
            or profile.get("state")
            != "inactive_complete_live_matrix_pending_review"
            or profile.get("failed_case") is not None
            or profile.get("safe_error") is not None
            or _string_list(
                profile.get("completed_cases"),
                "Phase 12 profile completed cases",
            )
            != execution_order
            or _string_list(
                receipt.get("completed_cases"),
                "Phase 12 receipt completed cases",
            )
            != execution_order
            or receipt.get("production_activation_granted") is not False
            or receipt.get("opaque_payloads_logged") is not False
        ):
            raise ConversationAcceptanceError(
                "Phase 12 receipt identity differs from its provider profile"
            )
        identities.append(
            _Phase12LiveReceiptIdentity(
                provider_family=provider_family,
                profile=model_or_deployment,
                revision=revision,
                structural_observations_digest=structural_digest,
            )
        )
    completed_count = live_results.get("completed_full_matrix_profile_count")
    if (
        type(completed_count) is not int
        or completed_count != len(identities)
        or not identities
        or live_results.get("active_profile_count") != 0
        or live_results.get("activation_decision") != "remain_inactive"
    ):
        raise ConversationAcceptanceError(
            "Phase 12 current live receipt inventory is invalid"
        )
    return tuple(identities)


def _validate_phase12_live_proof_resolution(root: Path) -> None:
    """Resolve every activation proof to one exact current live receipt."""
    activation, _, _ = _phase12_anchored_payload(
        root,
        _PHASE12_ACTIVATION_DECISION_PATH,
        expected_byte_sha256=_PHASE12_ACTIVATION_DECISION_BYTE_SHA256,
        expected_canonical_sha256=(
            _PHASE12_ACTIVATION_DECISION_CANONICAL_SHA256
        ),
        label="Phase 12 activation decision",
    )
    _phase12_validate_review_signature(activation)
    if (
        activation.get("activation_state") != "inactive"
        or activation.get("production_dispatch_enabled") is not False
        or activation.get("production_advertisement_enabled") is not False
        or activation.get("active_production_rows") != []
    ):
        raise ConversationAcceptanceError(
            "Phase 12 activation decision is not zero-active"
        )
    live_results, live_bytes, live_canonical = _phase12_anchored_payload(
        root,
        _PHASE12_LIVE_RESULTS_PATH,
        expected_byte_sha256=_PHASE12_LIVE_RESULTS_BYTE_SHA256,
        expected_canonical_sha256=_PHASE12_LIVE_RESULTS_CANONICAL_SHA256,
        label="Phase 12 live results",
    )
    live_link = mapping(
        activation.get("live_evidence"),
        "Phase 12 activation live evidence link",
    )
    _exact_keys(
        live_link,
        {"path", "byte_sha256", "canonical_digest"},
        "Phase 12 activation live evidence link",
    )
    if (
        live_link.get("path") != PurePosixPath(_PHASE12_LIVE_RESULTS_PATH).name
        or live_link.get("byte_sha256") != sha256(live_bytes).hexdigest()
        or live_link.get("canonical_digest") != live_canonical
    ):
        raise ConversationAcceptanceError(
            "Phase 12 activation live evidence link is invalid"
        )
    proof_ids = _string_list(
        activation.get("live_proof_ids"),
        "Phase 12 activation live proof ID",
    )
    if (
        not proof_ids
        or proof_ids != tuple(sorted(proof_ids))
        or len(proof_ids) != len(set(proof_ids))
    ):
        raise ConversationAcceptanceError(
            "Phase 12 activation live proof IDs are duplicate or noncanonical"
        )
    identities = _phase12_live_receipt_identities(live_results)
    receipts_by_identity: dict[str, list[_Phase12LiveReceiptIdentity]] = {}
    for identity in identities:
        receipts_by_identity.setdefault(identity.identity_digest, []).append(
            identity
        )
    for proof_id in proof_ids:
        identity_digest, referenced_structural_digest = (
            _phase12_parse_live_proof_id(proof_id)
        )
        matches = receipts_by_identity.get(identity_digest, [])
        if not matches:
            raise ConversationAcceptanceError(
                "Phase 12 activation live proof does not resolve"
            )
        if len(matches) != 1:
            raise ConversationAcceptanceError(
                "Phase 12 activation live proof resolves ambiguously"
            )
        identity = matches[0]
        if (
            referenced_structural_digest
            != identity.structural_observations_digest
        ):
            raise ConversationAcceptanceError(
                "Phase 12 activation live proof digest does not match"
            )
        if proof_id != identity.proof_id:
            raise ConversationAcceptanceError(
                "Phase 12 activation live proof is noncanonical"
            )
    if set(proof_ids) != {identity.proof_id for identity in identities}:
        raise ConversationAcceptanceError(
            "Phase 12 current live receipt is missing an activation proof"
        )


def _validate_phase12_traceability_candidate(
    root: Path,
    manifest: AcceptanceManifest,
) -> None:
    """Validate the non-promoting exact Phase 12 evidence candidate."""
    path = root / _PHASE12_TRACEABILITY_CANDIDATE_PATH
    try:
        candidate_bytes = path.read_bytes()
    except OSError as exc:
        raise ConversationAcceptanceError(
            "cannot read Phase 12 traceability candidate bytes"
        ) from exc
    if (
        sha256(candidate_bytes).hexdigest()
        != _PHASE12_TRACEABILITY_CANDIDATE_BYTE_SHA256
    ):
        raise ConversationAcceptanceError(
            "Phase 12 traceability candidate byte anchor is invalid"
        )
    payload = _strict_mapping(path, "Phase 12 traceability candidate")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "phase",
            "authoritative_phase",
            "candidate_state",
            "planned_nodes",
            "public_e2e_inventory",
            "normative_requirements",
            "external_blockers",
            "canonical_sha256",
        },
        "Phase 12 traceability candidate",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("phase") != 12
        or payload.get("authoritative_phase") != 11
        or payload.get("candidate_state")
        != "azure_live_complete_openai_blocked_zero_active"
        or manifest.current_phase != 11
    ):
        raise ConversationAcceptanceError(
            "Phase 12 traceability candidate promoted or drifted"
        )
    canonical = dict(payload)
    observed_digest = canonical.pop("canonical_sha256")
    if (
        observed_digest != _PHASE12_TRACEABILITY_CANDIDATE_CANONICAL_SHA256
        or observed_digest != canonical_sha256(canonical)
    ):
        raise ConversationAcceptanceError(
            "Phase 12 traceability candidate digest is invalid"
        )
    mapping_authority = {
        "public_e2e_inventory": payload.get("public_e2e_inventory"),
        "normative_requirements": payload.get("normative_requirements"),
    }
    if (
        canonical_sha256(mapping_authority)
        != _PHASE12_TRACEABILITY_MAPPING_CANONICAL_SHA256
    ):
        raise ConversationAcceptanceError(
            "Phase 12 traceability mapping authority digest is invalid"
        )

    planned = object_list(
        payload.get("planned_nodes"), "Phase 12 planned candidate nodes"
    )
    if len(planned) != 2:
        raise ConversationAcceptanceError(
            "Phase 12 candidate must retain exactly two planned nodes"
        )
    expected_plans = (
        (
            "phase12-live-completion",
            _PHASE12_LIVE_NODE_ID,
            "live",
            "planned_external",
            ("native_openai", "native_azure"),
            _PHASE12_LIVE_CASES,
        ),
        (
            "phase12-full-matrix",
            _PHASE12_MATRIX_NODE_ID,
            "matrix",
            "candidate_deterministic",
            (
                "native_openai",
                "native_azure",
                "incapable_generic_compatible",
            ),
            _PHASE12_MATRIX_CASES,
        ),
    )
    manifest_by_id = {node.id: node for node in manifest.nodes}
    for raw, expected in zip(planned, expected_plans, strict=True):
        record = mapping(raw, "Phase 12 planned candidate node")
        _exact_keys(
            record,
            {
                "id",
                "node_id",
                "evidence_class",
                "evidence_state",
                "provider_families",
                "observable_cases",
            },
            "Phase 12 planned candidate node",
        )
        observed = (
            record.get("id"),
            record.get("node_id"),
            record.get("evidence_class"),
            record.get("evidence_state"),
            _string_list(
                record.get("provider_families"),
                "Phase 12 candidate provider families",
            ),
            _string_list(
                record.get("observable_cases"),
                "Phase 12 candidate observable cases",
            ),
        )
        if observed != expected:
            raise ConversationAcceptanceError(
                "Phase 12 planned node contains label-only or broad claims"
            )
        manifest_node = manifest_by_id.get(expected[0])
        if (
            manifest_node is None
            or manifest_node.node_id != expected[1]
            or manifest_node.evidence_class != expected[2]
            or manifest_node.lifecycle != "planned"
            or manifest_node.active_from_phase != 12
        ):
            raise ConversationAcceptanceError(
                "Phase 12 candidate differs from the frozen planned node"
            )
    try:
        _validate_node_sources(
            root,
            (_PHASE12_LIVE_NODE_ID, _PHASE12_MATRIX_NODE_ID),
            {
                _PHASE12_LIVE_NODE_ID: "live",
                _PHASE12_MATRIX_NODE_ID: "matrix",
            },
        )
    except ContractGateError as exc:
        raise ConversationAcceptanceError(str(exc)) from exc

    manifest_by_node = {node.node_id: node for node in manifest.nodes}
    public_rows = object_list(
        payload.get("public_e2e_inventory"),
        "Phase 12 public E2E inventory",
    )
    expected_public_ids = tuple(
        f"CONV-E2E-{ordinal:03d}" for ordinal in range(1, 16)
    )
    observed_public_ids: list[str] = []
    public_evidence: dict[str, tuple[tuple[str, str, str], ...]] = {}
    for raw in public_rows:
        record = mapping(raw, "Phase 12 public E2E record")
        _exact_keys(record, {"id", "evidence"}, "Phase 12 public E2E record")
        identifier = _nonempty_string(
            record.get("id"), "Phase 12 public E2E ID"
        )
        observed_public_ids.append(identifier)
        public_evidence[identifier] = _phase12_candidate_evidence(
            root,
            record.get("evidence"),
            manifest_by_node,
        )
    if tuple(observed_public_ids) != expected_public_ids:
        raise ConversationAcceptanceError(
            "Phase 12 public E2E inventory must be exactly 001 through 015"
        )
    expected_provider_evidence = (
        (_PHASE12_LIVE_NODE_ID, "live", "planned_external"),
        (
            _PHASE12_MATRIX_NODE_ID,
            "matrix",
            "candidate_deterministic",
        ),
        (
            (
                "tests/conversation/native_openai_provider_test.py::"
                "test_unproven_or_drifted_profiles_fail_without_dispatch"
            ),
            "pre_dispatch_rejection",
            "active",
        ),
    )
    if public_evidence["CONV-E2E-015"] != expected_provider_evidence:
        raise ConversationAcceptanceError(
            "CONV-E2E-015 provider evidence is not exact"
        )

    normative_rows = object_list(
        payload.get("normative_requirements"),
        "Phase 12 normative requirement evidence",
    )
    expected_normative_ids = tuple(
        f"CONV-N-{ordinal:03d}" for ordinal in range(136, 145)
    )
    normative: dict[str, tuple[tuple[str, str, str], ...]] = {}
    for raw in normative_rows:
        record = mapping(raw, "Phase 12 normative requirement record")
        _exact_keys(
            record,
            {"id", "evidence"},
            "Phase 12 normative requirement record",
        )
        identifier = _nonempty_string(
            record.get("id"), "Phase 12 normative requirement ID"
        )
        normative[identifier] = _phase12_candidate_evidence(
            root,
            record.get("evidence"),
            manifest_by_node,
        )
    if tuple(normative) != expected_normative_ids:
        raise ConversationAcceptanceError(
            "Phase 12 normative mappings must be exactly 136 through 144"
        )
    if normative["CONV-N-142"] != (
        (_PHASE12_LIVE_NODE_ID, "live", "planned_external"),
    ):
        raise ConversationAcceptanceError(
            "live provider verification must remain externally planned"
        )
    if normative["CONV-N-143"] != (
        (
            (
                "tests/conversation/native_openai_provider_test.py::"
                "test_unproven_or_drifted_profiles_fail_without_dispatch"
            ),
            "pre_dispatch_rejection",
            "active",
        ),
        (
            _PHASE12_MATRIX_NODE_ID,
            "matrix",
            "candidate_deterministic",
        ),
    ):
        raise ConversationAcceptanceError(
            "generic-compatible rejection evidence is not exact"
        )
    for node_id in (_PHASE12_LIVE_NODE_ID, _PHASE12_MATRIX_NODE_ID):
        if not any(
            node_id == evidence[0]
            for rows in (*public_evidence.values(), *normative.values())
            for evidence in rows
        ):
            raise ConversationAcceptanceError(
                "Phase 12 planned node lacks an exact evidence owner"
            )

    blockers = object_list(
        payload.get("external_blockers"), "Phase 12 external blockers"
    )
    expected_blockers = ("native_openai_live_receipt",)
    observed_blockers: list[str] = []
    for raw in blockers:
        record = mapping(raw, "Phase 12 external blocker")
        _exact_keys(
            record,
            {"id", "state", "node_id"},
            "Phase 12 external blocker",
        )
        identifier = _nonempty_string(record.get("id"), "external blocker ID")
        observed_blockers.append(identifier)
        if (
            record.get("state")
            != _PHASE12_EXTERNAL_BLOCKER_STATES.get(identifier)
            or record.get("node_id") != _PHASE12_LIVE_NODE_ID
        ):
            raise ConversationAcceptanceError(
                "Phase 12 external blocker is not precise"
            )
    if tuple(observed_blockers) != expected_blockers:
        raise ConversationAcceptanceError(
            "Phase 12 external blocker inventory is invalid"
        )


def _phase12_candidate_evidence(
    root: Path,
    value: object,
    manifest_by_node: Mapping[str, AcceptanceNode],
) -> tuple[tuple[str, str, str], ...]:
    """Validate exact node-level evidence in the Phase 12 candidate."""
    rows: list[tuple[str, str, str]] = []
    for raw in object_list(value, "Phase 12 candidate evidence"):
        record = mapping(raw, "Phase 12 candidate evidence row")
        _exact_keys(
            record,
            {"node_id", "evidence_class", "evidence_state"},
            "Phase 12 candidate evidence row",
        )
        node_id = _test_node(record.get("node_id"))
        evidence_class = _nonempty_string(
            record.get("evidence_class"), "candidate evidence class"
        )
        evidence_state = _nonempty_string(
            record.get("evidence_state"), "candidate evidence state"
        )
        manifest_node = manifest_by_node.get(node_id)
        expected_class = (
            manifest_node.evidence_class
            if manifest_node is not None
            else _PHASE12_CANDIDATE_ONLY_EVIDENCE.get(node_id)
        )
        if expected_class != evidence_class:
            raise ConversationAcceptanceError(
                f"candidate evidence class is not exact: {node_id}"
            )
        if evidence_state == "active":
            if manifest_node is None or manifest_node.lifecycle != "active":
                raise ConversationAcceptanceError(
                    f"candidate labels inactive evidence active: {node_id}"
                )
        elif evidence_state == "planned_external":
            if (
                node_id != _PHASE12_LIVE_NODE_ID
                or manifest_node is None
                or manifest_node.lifecycle != "planned"
            ):
                raise ConversationAcceptanceError(
                    "only the exact live node may remain externally planned"
                )
        elif evidence_state != "candidate_deterministic":
            raise ConversationAcceptanceError(
                f"candidate evidence state is invalid: {node_id}"
            )
        _phase12_candidate_node_exists(root, node_id)
        rows.append((node_id, evidence_class, evidence_state))
    if not rows or len(rows) != len(set(rows)):
        raise ConversationAcceptanceError(
            "Phase 12 candidate evidence is empty or duplicated"
        )
    return tuple(rows)


def _phase12_candidate_node_exists(root: Path, node_id: str) -> None:
    """Require a candidate node to resolve to one real test function."""
    relative, *parts = node_id.split("::")
    path = (root / relative).resolve()
    test_root = (root / "tests").resolve()
    if not path.is_relative_to(test_root) or not path.is_file():
        raise ConversationAcceptanceError(
            f"Phase 12 candidate test does not exist: {node_id}"
        )
    try:
        children = tuple(parse_python(path.read_text(encoding="utf-8")).body)
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ConversationAcceptanceError(
            f"Phase 12 candidate test cannot be inspected: {node_id}"
        ) from exc
    target: AsyncFunctionDef | ClassDef | FunctionDef | None = None
    for part in parts:
        target = next(
            (
                child
                for child in children
                if isinstance(
                    child,
                    (AsyncFunctionDef, ClassDef, FunctionDef),
                )
                and child.name == part.split("[", 1)[0]
            ),
            None,
        )
        if target is None:
            raise ConversationAcceptanceError(
                f"Phase 12 candidate node cannot be resolved: {node_id}"
            )
        children = tuple(target.body)
    if not isinstance(target, (AsyncFunctionDef, FunctionDef)):
        raise ConversationAcceptanceError(
            f"Phase 12 candidate node is not a test: {node_id}"
        )


def _validate_contract_decisions(payload: dict[str, object]) -> None:
    expected = {
        "activation",
        "atomic_boundaries",
        "authority",
        "branching",
        "checkpoint",
        "configuration",
        "contract_version",
        "deletion",
        "descendants",
        "failure_fence_tuple_fields",
        "failure_fences",
        "feature",
        "idempotency",
        "identity",
        "migration",
        "owner",
        "provider_lane_binding",
        "public_response_id",
        "response_resource",
        "retention",
        "schema_version",
        "storage",
        "surfaces",
    }
    _exact_keys(payload, expected, "contract decisions")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("activation") != "dormant"
    ):
        raise ConversationAcceptanceError(
            "contract decisions are not the dormant version 1 contract"
        )
    for field in expected - {
        "activation",
        "contract_version",
        "feature",
        "owner",
        "schema_version",
    }:
        if (
            not isinstance(payload.get(field), (dict, list))
            or not payload[field]
        ):
            raise ConversationAcceptanceError(
                f"contract decision group is empty: {field}"
            )


def _validate_deterministic_fixtures(payload: dict[str, object]) -> None:
    expected = {
        "async_barrier",
        "clock",
        "contract_version",
        "fault_injection",
        "fixture_sha256",
        "id_factory",
        "keys",
        "named_head_cases",
        "principal",
        "provider_capability",
        "provider_item_trace",
        "public_response_resources",
        "retention_cases",
        "schema_version",
    }
    _exact_keys(payload, expected, "deterministic fixtures")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_version") != 1
    ):
        raise ConversationAcceptanceError(
            "deterministic fixtures are not the version 1 inventory"
        )
    for field in expected - {"contract_version", "schema_version"}:
        if field == "fixture_sha256":
            continue
        if (
            not isinstance(payload.get(field), (dict, list))
            or not payload[field]
        ):
            raise ConversationAcceptanceError(
                f"deterministic fixture group is empty: {field}"
            )
    canonical = {
        key: value for key, value in payload.items() if key != "fixture_sha256"
    }
    if payload.get("fixture_sha256") != canonical_sha256(canonical):
        raise ConversationAcceptanceError(
            "deterministic fixture digest is invalid"
        )


def _validate_provider_contract(payload: dict[str, object]) -> None:
    expected = {
        "activation_state",
        "canonical_digest",
        "conformance_digest",
        "contract_version",
        "current_phase",
        "feature",
        "owner",
        "retrieved_date",
        "schema_version",
        "sdk_boundary",
        "snapshot_id",
        "sources",
    }
    _exact_keys(payload, expected, "provider contract")
    _validate_provider_header(payload, "provider contract")
    sdk = mapping(payload.get("sdk_boundary"), "provider SDK boundary")
    policy = mapping(
        sdk.get("conversation_state_transport_policy"),
        "conversation state transport policy",
    )
    _exact_keys(
        policy,
        {
            "scope",
            "runtime_disposition",
            "legacy_generic_request_kwargs_acknowledged",
            "legacy_generic_request_kwargs_description",
            "prohibited_routes",
            "provider_wire_paths",
            "public_request_fields",
            "reasoning_mapping_policy",
            "stateful_create_field_policy",
        },
        "conversation state transport policy",
    )
    if (
        policy.get("scope") != "conversation_state_and_stateful_create_fields"
        or policy.get("runtime_disposition") != "dormant_fail_closed"
        or policy.get("legacy_generic_request_kwargs_acknowledged") is not True
    ):
        raise ConversationAcceptanceError(
            "provider conversation-state transport policy is not fail closed"
        )
    _nonempty_string(
        policy.get("legacy_generic_request_kwargs_description"),
        "legacy generic request kwargs description",
    )
    if _string_list(
        policy.get("prohibited_routes"), "prohibited provider routes"
    ) != (
        "extra_body",
        "conversation_state_dict[str, A" + "ny]",
        "conversation_state_mapping_unpack",
        "untyped_generation_override",
        "caller_or_dynamic_store_control",
        "background_dispatch",
        "alternate_response_create_mapping_unpack",
        "responses_lifecycle_alias_or_getattr",
        "tracked_request_binding_rebind",
        "trusted_helper_shadow",
        "runtime_namespace_or_frame_reflection",
        "reasoning_mapping_alias_or_mutator",
        "phase0_provider_source_integrity_drift",
        "runtime_non_create_response_lifecycle",
    ):
        raise ConversationAcceptanceError(
            "provider prohibited transport routes changed"
        )
    if _string_list(
        policy.get("provider_wire_paths"), "provider wire paths"
    ) != (
        "background",
        "previous_response_id",
        "conversation",
        "context_management",
        "context_management.compact_threshold",
        "reasoning.context",
        "store",
    ):
        raise ConversationAcceptanceError(
            "provider wire path inventory changed"
        )
    if _string_list(
        policy.get("public_request_fields"), "public request fields"
    ) != (
        "background",
        "previous_response_id",
        "conversation",
        "context_management",
        "reasoning_context",
        "conversation_handle",
        "continuation_envelope",
        "store",
    ):
        raise ConversationAcceptanceError(
            "provider public conversation request field inventory changed"
        )
    _validate_reasoning_mapping_policy(policy.get("reasoning_mapping_policy"))
    _validate_stateful_create_field_policy(
        policy.get("stateful_create_field_policy")
    )
    _validate_scoped_digest(payload, "provider contract")
    digest = mapping(
        payload.get("canonical_digest"), "provider contract digest"
    )
    if digest.get("value") != _PHASE0_PROVIDER_CANONICAL_SHA256:
        raise ConversationAcceptanceError(
            "provider contract differs from its independent Phase 0 anchor"
        )


def _validate_reasoning_mapping_policy(raw: object) -> None:
    """Validate the closed static reasoning mapping policy."""
    policy = mapping(raw, "reasoning mapping policy")
    if policy != {
        "mapping_name": "reasoning",
        "allowed_static_keys": ["effort", "summary"],
        "forbidden_path": "reasoning.context",
        "dynamic_keys_allowed": False,
        "aliases_allowed": False,
        "mutator_calls_allowed": False,
    }:
        raise ConversationAcceptanceError(
            "provider reasoning mapping policy changed"
        )


def _validate_stateful_create_field_policy(raw: object) -> None:
    """Validate fixed Phase 0 provider retention and dispatch policy."""
    policy = mapping(raw, "stateful create field policy")
    _exact_keys(
        policy,
        {
            "forbidden_provider_wire_roots",
            "typed_sdk_create_fields",
            "legacy_fixed_provider_values",
            "provider_mapping_flow",
            "closed_ast_gate",
            "public_runtime_disposition",
        },
        "stateful create field policy",
    )
    typed = mapping(
        policy.get("typed_sdk_create_fields"),
        "typed stateful create fields",
    )
    _exact_keys(typed, {"background", "store"}, "typed stateful create fields")
    annotation_sha256 = (
        "80624365ea5db072b2ea31b2a3bf9d483b05fd5f828c7fc0ed7da554518892a5"
    )
    background = mapping(typed.get("background"), "background field policy")
    _exact_keys(
        background,
        {
            "sdk_parameter_kind",
            "sdk_default_contract",
            "sdk_resolved_annotation_sha256",
            "provider_runtime_disposition",
            "allowed_provider_write_count",
            "public_runtime_disposition",
        },
        "background field policy",
    )
    store = mapping(typed.get("store"), "store field policy")
    _exact_keys(
        store,
        {
            "sdk_parameter_kind",
            "sdk_default_contract",
            "sdk_resolved_annotation_sha256",
            "provider_runtime_disposition",
            "allowed_provider_write_count",
            "allowed_provider_value",
            "public_runtime_disposition",
        },
        "store field policy",
    )
    shared = {
        "sdk_parameter_kind": "KEYWORD_ONLY",
        "sdk_default_contract": "singleton:openai.Omit",
        "sdk_resolved_annotation_sha256": annotation_sha256,
        "public_runtime_disposition": "dormant_fail_closed",
    }
    if (
        any(background.get(key) != value for key, value in shared.items())
        or background.get("provider_runtime_disposition") != "prohibited"
        or type(background.get("allowed_provider_write_count")) is not int
        or background.get("allowed_provider_write_count") != 0
    ):
        raise ConversationAcceptanceError(
            "background provider policy is not fail closed"
        )
    if (
        any(store.get(key) != value for key, value in shared.items())
        or store.get("provider_runtime_disposition")
        != "legacy_fixed_false_only"
        or type(store.get("allowed_provider_write_count")) is not int
        or store.get("allowed_provider_write_count") != 1
        or store.get("allowed_provider_value") is not False
    ):
        raise ConversationAcceptanceError(
            "store provider policy is not fixed to false"
        )
    legacy_values = mapping(
        policy.get("legacy_fixed_provider_values"),
        "legacy fixed provider values",
    )
    if (
        set(legacy_values) != {"store"}
        or legacy_values.get("store") is not False
    ):
        raise ConversationAcceptanceError(
            "legacy provider values are not fixed to store=false"
        )
    mapping_flow = mapping(
        policy.get("provider_mapping_flow"),
        "provider mapping flow",
    )
    if mapping_flow != {
        "initial_request_mapping": "kwargs",
        "normalization_temporary": "normalized_request_kwargs",
        "normalized_request_mapping": "request_kwargs",
        "attempt_request_mapping": "attempt_kwargs",
        "copy_function": "_strict_replay_json_copy",
        "create_target": "request_client.responses.create",
        "create_unpack_source": "attempt_kwargs",
        "create_call_count": 1,
        "mapping_unpack_count": 1,
    } or any(
        type(mapping_flow.get(field)) is not int
        for field in ("create_call_count", "mapping_unpack_count")
    ):
        raise ConversationAcceptanceError("provider mapping flow changed")
    if _string_list(
        policy.get("forbidden_provider_wire_roots"),
        "forbidden provider wire roots",
    ) != (
        "background",
        "compact_threshold",
        "context_management",
        "conversation",
        "extra_body",
        "previous_response_id",
        "store",
    ):
        raise ConversationAcceptanceError(
            "forbidden provider wire roots changed"
        )
    closed_gate = mapping(policy.get("closed_ast_gate"), "closed AST gate")
    if closed_gate != {
        "tracked_bindings": [
            "attempt_kwargs",
            "kwargs",
            "normalized_request_kwargs",
            "request_client",
            "request_kwargs",
        ],
        "trusted_helpers": ["_strict_replay_json_copy", "cast"],
        "forbidden_reflection_names": [
            "eval",
            "exec",
            "globals",
            "locals",
            "vars",
        ],
        "forbidden_frame_attributes": [
            "_getframe",
            "ag_frame",
            "cr_frame",
            "currentframe",
            "f_back",
            "f_globals",
            "f_locals",
            "gi_frame",
            "tb_frame",
        ],
        "phase0_source_integrity": {
            "phase": 0,
            "kind": "exact_source_sha256",
            "algorithm": "sha256",
            "encoding": "sha256 of exact UTF-8 provider module source bytes",
            "source_path": "src/avalan/model/nlp/text/vendor/openai.py",
            "covers": [
                "module_import_and_binding_topology",
                "_strict_replay_json_copy",
                "OpenAIClient.__call__",
                "OpenAIClient._reasoning_config",
            ],
            "rotation_policy": "reviewed_provider_phase_transition_only",
            "value": _PHASE0_PROVIDER_SOURCE_SHA256,
        },
    }:
        raise ConversationAcceptanceError("provider closed AST gate changed")
    source_integrity = mapping(
        closed_gate.get("phase0_source_integrity"),
        "Phase 0 provider source integrity",
    )
    if type(source_integrity.get("phase")) is not int:
        raise ConversationAcceptanceError(
            "provider source-integrity phase must be an integer"
        )
    if policy.get("public_runtime_disposition") != "dormant_fail_closed":
        raise ConversationAcceptanceError(
            "stateful public create fields are not fail closed"
        )


def _validate_provider_conformance(payload: dict[str, object]) -> None:
    expected = {
        "activation_state",
        "canonical_digest",
        "capability_names",
        "capability_states",
        "current_phase",
        "feature",
        "identity_dimensions",
        "inference_policy",
        "owner",
        "production_advertisement_enabled",
        "production_dispatch_enabled",
        "profile_schema_version",
        "profiles",
        "rejected_inference_cases",
        "schema_version",
    }
    _exact_keys(payload, expected, "provider conformance")
    _validate_provider_header(payload, "provider conformance")
    if (
        payload.get("production_advertisement_enabled") is not False
        or payload.get("production_dispatch_enabled") is not False
    ):
        raise ConversationAcceptanceError(
            "Phase 0 provider capabilities must not advertise or dispatch"
        )
    profiles = object_list(payload.get("profiles"), "provider profiles")
    if not profiles:
        raise ConversationAcceptanceError(
            "provider profiles must be non-empty"
        )
    for raw in profiles:
        profile = mapping(raw, "provider profile")
        activation = profile.get("activation_state")
        lifecycle = profile.get("lifecycle")
        if activation not in {"dormant", "incapable"} or lifecycle not in {
            "planned",
            "incapable",
        }:
            raise ConversationAcceptanceError(
                "provider profiles must remain planned/dormant or incapable"
            )
        if profile.get("identity_complete") is not False:
            raise ConversationAcceptanceError(
                "Phase 0 provider profiles cannot claim complete identity"
            )
        capabilities = mapping(
            profile.get("capabilities"), "provider capabilities"
        )
        if not capabilities or any(
            value not in {"dormant", "incapable"}
            for value in capabilities.values()
        ):
            raise ConversationAcceptanceError(
                "provider capability state is prematurely active"
            )
        if object_list(
            profile.get("activation_evidence"), "activation evidence"
        ):
            raise ConversationAcceptanceError(
                "Phase 0 profiles cannot contain activation evidence"
            )
    _validate_scoped_digest(payload, "provider conformance")


def _validate_phase5_provider_conformance(
    payload: dict[str, object],
    phase0: dict[str, object],
) -> None:
    """Validate append-only test evidence for exact native stateless lanes."""
    expected = {
        "activation_state",
        "canonical_digest",
        "capability_names",
        "capability_states",
        "current_phase",
        "feature",
        "identity_dimensions",
        "inference_policy",
        "owner",
        "production_advertisement_enabled",
        "production_dispatch_enabled",
        "profile_schema_version",
        "profiles",
        "rejected_inference_cases",
        "schema_version",
    }
    _exact_keys(payload, expected, "Phase 5 provider conformance")
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("current_phase") != 5
        or payload.get("activation_state") != "test_only"
        or payload.get("production_advertisement_enabled") is not False
        or payload.get("production_dispatch_enabled") is not False
    ):
        raise ConversationAcceptanceError(
            "Phase 5 provider conformance must remain test-only"
        )
    if _string_list(
        payload.get("capability_states"),
        "Phase 5 provider capability states",
    ) != ("test_only", "incapable"):
        raise ConversationAcceptanceError(
            "Phase 5 provider capability states are invalid"
        )
    profiles = object_list(payload.get("profiles"), "Phase 5 profiles")
    phase0_profiles = object_list(phase0.get("profiles"), "Phase 0 profiles")
    if profiles[: len(phase0_profiles)] != phase0_profiles:
        raise ConversationAcceptanceError(
            "Phase 5 provider profiles rewrote Phase 0 history"
        )
    active = profiles[len(phase0_profiles) :]
    expected_identities = (
        (
            "native-openai-stateless-non-streaming-phase5-test",
            "openai",
            "non_streaming",
        ),
        (
            "native-openai-stateless-streaming-phase5-test",
            "openai",
            "streaming",
        ),
        (
            "native-azure-stateless-non-streaming-phase5-test",
            "azure_openai",
            "non_streaming",
        ),
        (
            "native-azure-stateless-streaming-phase5-test",
            "azure_openai",
            "streaming",
        ),
    )
    exact_evidence = {
        "native-openai-stateless-non-streaming-phase5-test": (
            (
                "phase5-exact-request-matrix",
                "phase5-loopback-postgresql-fresh-process",
            ),
            (
                (
                    "tests/conversation/native_openai_provider_test.py::"
                    "test_exact_profile_request_matrix"
                ),
                (
                    "tests/conversation/openai_stateless_e2e_test.py::"
                    "test_native_openai_fresh_process_durable_replay"
                ),
            ),
        ),
        "native-openai-stateless-streaming-phase5-test": (
            ("phase5-exact-request-matrix",),
            (
                (
                    "tests/conversation/native_openai_provider_test.py::"
                    "test_exact_profile_request_matrix"
                ),
            ),
        ),
        "native-azure-stateless-non-streaming-phase5-test": (
            (
                "phase5-exact-request-matrix",
                "phase5-exact-azure-loopback-postgresql-wire",
            ),
            (
                (
                    "tests/conversation/native_openai_provider_test.py::"
                    "test_exact_profile_request_matrix"
                ),
                (
                    "tests/conversation/openai_stateless_e2e_test.py::"
                    "test_native_azure_exact_identity_over_loopback_transport"
                ),
            ),
        ),
        "native-azure-stateless-streaming-phase5-test": (
            (
                "phase5-exact-request-matrix",
                "phase5-exact-azure-loopback-postgresql-wire",
            ),
            (
                (
                    "tests/conversation/native_openai_provider_test.py::"
                    "test_exact_profile_request_matrix"
                ),
                (
                    "tests/conversation/openai_stateless_e2e_test.py::"
                    "test_native_azure_exact_identity_over_loopback_transport"
                ),
            ),
        ),
    }
    if len(active) != len(expected_identities):
        raise ConversationAcceptanceError(
            "Phase 5 native provider profile inventory is incomplete"
        )
    capability_names = set(
        _string_list(
            payload.get("capability_names"),
            "Phase 5 provider capability names",
        )
    )
    for raw, (profile_id, family, transport) in zip(
        active,
        expected_identities,
        strict=True,
    ):
        profile = mapping(raw, "Phase 5 provider profile")
        if (
            profile.get("profile_id") != profile_id
            or profile.get("lifecycle") != "active"
            or profile.get("active_from_phase") != 5
            or profile.get("activation_state") != "test_only"
            or profile.get("identity_complete") is not True
        ):
            raise ConversationAcceptanceError(
                "Phase 5 native provider profile activation is invalid"
            )
        binding = mapping(profile.get("binding"), "Phase 5 binding")
        if (
            binding.get("adapter_type")
            != "avalan.conversation.providers.openai."
            "NativeOpenAIStatelessProvider"
            or binding.get("provider_family") != family
            or binding.get("transport") != transport
            or binding.get("sdk_revision") != "openai-python-2.42.0"
            or binding.get("continuation_codec_version") != "1"
        ):
            raise ConversationAcceptanceError(
                "Phase 5 native provider binding is not exact"
            )
        capabilities = mapping(
            profile.get("capabilities"),
            "Phase 5 native provider capabilities",
        )
        if set(capabilities) != capability_names or any(
            state not in {"test_only", "incapable"}
            for state in capabilities.values()
        ):
            raise ConversationAcceptanceError(
                "Phase 5 native provider capabilities are invalid"
            )
        required = {
            "stateless_encrypted_reasoning_replay",
            "reasoning_context_current_turn",
            "reasoning_context_all_turns",
        }
        if transport == "streaming":
            required.add("streaming_item_fidelity")
        if any(capabilities.get(name) != "test_only" for name in required):
            raise ConversationAcceptanceError(
                "Phase 5 native profile lacks required test evidence"
            )
        if any(
            capabilities.get(name) != "incapable"
            for name in capability_names - required
        ):
            raise ConversationAcceptanceError(
                "Phase 5 native profile activates an unproven capability"
            )
        evidence = _string_list(
            profile.get("activation_evidence"),
            "Phase 5 provider activation evidence",
        )
        expected_activation, expected_nodes = exact_evidence[profile_id]
        if evidence != expected_activation:
            raise ConversationAcceptanceError(
                "Phase 5 provider activation evidence is invalid"
            )
        nodes = _string_list(
            profile.get("evidence_node_ids"),
            "Phase 5 provider evidence nodes",
        )
        if nodes != expected_nodes:
            raise ConversationAcceptanceError(
                "Phase 5 provider evidence node is invalid"
            )
    _validate_scoped_digest(payload, "Phase 5 provider conformance")


def _validate_phase6_provider_conformance(
    payload: dict[str, object],
    phase5: dict[str, object],
    *,
    phase5_path: Path,
) -> None:
    """Validate exact stored profiles against independently pinned evidence."""
    _exact_keys(
        payload,
        {
            "activation_state",
            "base",
            "canonical_digest",
            "capability_names",
            "capability_states",
            "current_phase",
            "feature",
            "generic_compatible_state",
            "identity_dimensions",
            "owner",
            "production_advertisement_enabled",
            "production_dispatch_enabled",
            "profile_schema_version",
            "profiles",
            "rejected_profile_evidence",
            "schema_version",
        },
        "Phase 6 provider conformance",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("profile_schema_version")
        != "conversation-provider-profile-v1"
        or payload.get("feature") != _FEATURE
        or payload.get("owner") != "provider_runtime"
        or payload.get("current_phase") != 6
        or payload.get("activation_state") != "test_only"
        or payload.get("production_advertisement_enabled") is not False
        or payload.get("production_dispatch_enabled") is not False
        or payload.get("generic_compatible_state") != "incapable"
    ):
        raise ConversationAcceptanceError(
            "Phase 6 provider conformance must remain exact and test-only"
        )
    if _string_list(
        payload.get("capability_names"),
        "Phase 6 provider capability names",
    ) != _string_list(
        phase5.get("capability_names"),
        "Phase 5 provider capability names",
    ):
        raise ConversationAcceptanceError(
            "Phase 6 provider capabilities differ from the frozen Phase 5 axes"
        )
    phase5_identity_dimensions = _string_list(
        phase5.get("identity_dimensions"),
        "Phase 5 provider identity dimensions",
    )
    if _string_list(
        payload.get("identity_dimensions"),
        "Phase 6 provider identity dimensions",
    ) != (*phase5_identity_dimensions, "execution_definition_digest"):
        raise ConversationAcceptanceError(
            "Phase 6 provider identity axes must append the exact execution "
            "definition digest to the frozen Phase 5 axes"
        )
    if _string_list(
        payload.get("capability_states"),
        "Phase 6 provider capability states",
    ) != ("test_only", "incapable"):
        raise ConversationAcceptanceError(
            "Phase 6 provider capability states are invalid"
        )
    base = mapping(payload.get("base"), "Phase 6 provider base")
    _exact_keys(
        base,
        {"path", "byte_sha256", "canonical_digest"},
        "Phase 6 provider base",
    )
    phase5_digest = mapping(
        phase5.get("canonical_digest"),
        "Phase 5 provider canonical digest",
    )
    if (
        base.get("path") != _PHASE5_PROVIDER_CONFORMANCE
        or base.get("byte_sha256") != _PHASE5_PROVIDER_CONFORMANCE_BYTE_SHA256
        or sha256(phase5_path.read_bytes()).hexdigest()
        != _PHASE5_PROVIDER_CONFORMANCE_BYTE_SHA256
        or base.get("canonical_digest") != phase5_digest.get("value")
    ):
        raise ConversationAcceptanceError(
            "Phase 6 provider base is not the frozen Phase 5 evidence"
        )

    profiles = object_list(payload.get("profiles"), "Phase 6 profiles")
    expected_identities = (
        (
            "native-openai-stored-non-streaming-phase6-test",
            "openai",
            "https://api.openai.com/v1",
            None,
            "gpt-5",
            "openapi-2.3.0",
            "non_streaming",
        ),
        (
            "native-openai-stored-streaming-phase6-test",
            "openai",
            "https://api.openai.com/v1",
            None,
            "gpt-5",
            "openapi-2.3.0",
            "streaming",
        ),
        (
            "native-azure-stored-non-streaming-phase6-test",
            "azure_openai",
            "https://resource.openai.azure.com/openai/v1",
            "resource.openai.azure.com",
            "deployment-stored",
            "azure-openai-v1",
            "non_streaming",
        ),
        (
            "native-azure-stored-streaming-phase6-test",
            "azure_openai",
            "https://resource.openai.azure.com/openai/v1",
            "resource.openai.azure.com",
            "deployment-stored",
            "azure-openai-v1",
            "streaming",
        ),
    )
    exact_evidence = {
        "native-openai-stored-non-streaming-phase6-test": (
            (
                "phase6-exact-stored-reasoning-transport-matrix",
                "phase6-exact-stored-lifecycle-matrix",
                "phase6-loopback-postgresql-fresh-process",
                "phase6-execution-definition-digest",
            ),
            (
                (
                    "tests/conversation/phase6_validation_test.py::"
                    "test_stored_reasoning_context_is_sent_explicitly"
                ),
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_exact_stored_lifecycle_profile_matrix"
                ),
                (
                    "tests/conversation/openai_stored_e2e_test.py::"
                    "test_stored_restart_retrieval_and_deletion"
                ),
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_execution_definition_bytes_fail_before_dispatch"
                ),
            ),
        ),
        "native-openai-stored-streaming-phase6-test": (
            (
                "phase6-exact-stored-reasoning-transport-matrix",
                "phase6-exact-stored-lifecycle-matrix",
                "phase6-loopback-postgresql-streaming-tool-cycle",
                "phase6-execution-definition-digest",
            ),
            (
                (
                    "tests/conversation/phase6_validation_test.py::"
                    "test_stored_reasoning_context_is_sent_explicitly"
                ),
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_exact_stored_lifecycle_profile_matrix"
                ),
                (
                    "tests/conversation/openai_stored_e2e_test.py::"
                    "test_stored_streaming_tool_cycle_uses_terminal_id"
                ),
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_execution_definition_bytes_fail_before_dispatch"
                ),
            ),
        ),
        "native-azure-stored-non-streaming-phase6-test": (
            (
                "phase6-exact-stored-reasoning-transport-matrix",
                "phase6-exact-stored-lifecycle-matrix",
                "phase6-exact-azure-loopback-postgresql-wire",
                "phase6-execution-definition-digest",
            ),
            (
                (
                    "tests/conversation/phase6_validation_test.py::"
                    "test_stored_reasoning_context_is_sent_explicitly"
                ),
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_exact_stored_lifecycle_profile_matrix"
                ),
                (
                    "tests/conversation/openai_stored_e2e_test.py::"
                    "test_native_azure_stored_exact_identity_over_loopback_transport"
                ),
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_execution_definition_bytes_fail_before_dispatch"
                ),
            ),
        ),
        "native-azure-stored-streaming-phase6-test": (
            (
                "phase6-exact-stored-reasoning-transport-matrix",
                "phase6-exact-stored-lifecycle-matrix",
                "phase6-exact-azure-loopback-postgresql-wire",
                "phase6-execution-definition-digest",
            ),
            (
                (
                    "tests/conversation/phase6_validation_test.py::"
                    "test_stored_reasoning_context_is_sent_explicitly"
                ),
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_exact_stored_lifecycle_profile_matrix"
                ),
                (
                    "tests/conversation/openai_stored_e2e_test.py::"
                    "test_native_azure_stored_exact_identity_over_loopback_transport"
                ),
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_execution_definition_bytes_fail_before_dispatch"
                ),
            ),
        ),
    }
    execution_definition_digests = {
        "native-openai-stored-non-streaming-phase6-test": (
            "6942129db4435f33c3aa67d1b86be64adec98a272912961c46a1fddc8618878e"
        ),
        "native-openai-stored-streaming-phase6-test": (
            "b49822dca816625835c6214e2df3f513e0681a84bc867b143b91c47cbcb116fd"
        ),
        "native-azure-stored-non-streaming-phase6-test": (
            "084aaf6a2b0c994639e9786134421213678ca7af637a0f6af401243a20ae07d2"
        ),
        "native-azure-stored-streaming-phase6-test": (
            "7d7c335aba635819c1b9f510f3305810832acff1464e73aaa073df3db9d71a63"
        ),
    }
    if len(profiles) != len(expected_identities):
        raise ConversationAcceptanceError(
            "Phase 6 stored provider profile inventory is incomplete"
        )
    capability_names = set(
        _string_list(
            payload.get("capability_names"),
            "Phase 6 provider capability names",
        )
    )
    binding_keys = set(
        _string_list(
            payload.get("identity_dimensions"),
            "Phase 6 provider identity dimensions",
        )
    )
    for raw, identity in zip(profiles, expected_identities, strict=True):
        profile = mapping(raw, "Phase 6 stored provider profile")
        _exact_keys(
            profile,
            {
                "profile_id",
                "lifecycle",
                "active_from_phase",
                "activation_state",
                "identity_complete",
                "binding",
                "capabilities",
                "activation_evidence",
                "evidence_node_ids",
            },
            "Phase 6 stored provider profile",
        )
        (
            profile_id,
            family,
            endpoint,
            azure_resource,
            model,
            api_revision,
            transport,
        ) = identity
        if (
            profile.get("profile_id") != profile_id
            or profile.get("lifecycle") != "active"
            or profile.get("active_from_phase") != 6
            or profile.get("activation_state") != "test_only"
            or profile.get("identity_complete") is not True
        ):
            raise ConversationAcceptanceError(
                "Phase 6 stored provider activation is invalid"
            )
        binding = mapping(profile.get("binding"), "Phase 6 stored binding")
        _exact_keys(binding, binding_keys, "Phase 6 stored binding")
        expected_binding = {
            "adapter_type": (
                "avalan.conversation.providers.openai_stored."
                "NativeOpenAIStoredProvider"
            ),
            "provider_family": family,
            "normalized_endpoint": endpoint,
            "azure_resource_identity": azure_resource,
            "model_or_deployment": model,
            "provider_api_revision": api_revision,
            "sdk_revision": "openai-python-2.42.0",
            "model_configuration_revision": "model-config-phase6",
            "capability_profile_revision": "capability-phase6",
            "tool_schema_revision": "tools-phase6",
            "execution_definition_revision": "execution-phase6",
            "continuation_codec_version": "1",
            "transport": transport,
            "execution_definition_digest": execution_definition_digests[
                profile_id
            ],
        }
        if binding != expected_binding:
            raise ConversationAcceptanceError(
                "Phase 6 stored provider binding is not exact"
            )
        capabilities = mapping(
            profile.get("capabilities"),
            "Phase 6 stored provider capabilities",
        )
        if set(capabilities) != capability_names:
            raise ConversationAcceptanceError(
                "Phase 6 stored provider capability inventory is invalid"
            )
        required = {
            "stored_responses_chaining",
            "reasoning_context_current_turn",
            "reasoning_context_all_turns",
            "stored_response_retrieval",
            "stored_response_deletion",
        }
        if transport == "streaming":
            required.add("streaming_item_fidelity")
        if any(capabilities.get(name) != "test_only" for name in required):
            raise ConversationAcceptanceError(
                "Phase 6 stored profile lacks required test evidence"
            )
        if any(
            capabilities.get(name) != "incapable"
            for name in capability_names - required
        ):
            raise ConversationAcceptanceError(
                "Phase 6 stored profile activates an unproven capability"
            )
        expected_activation, expected_nodes = exact_evidence[profile_id]
        if (
            _string_list(
                profile.get("activation_evidence"),
                "Phase 6 provider activation evidence",
            )
            != expected_activation
            or _string_list(
                profile.get("evidence_node_ids"),
                "Phase 6 provider evidence nodes",
            )
            != expected_nodes
        ):
            raise ConversationAcceptanceError(
                "Phase 6 provider evidence is not independently pinned"
            )
    rejected = object_list(
        payload.get("rejected_profile_evidence"),
        "Phase 6 rejected provider evidence",
    )
    if rejected != [
        {
            "provider_family": "openai_compatible",
            "expected_state": "incapable",
            "evidence_node_id": (
                "tests/conversation/phase6_validation_test.py::"
                "test_stored_profile_rejects_unproven_provider_forms"
            ),
        }
    ]:
        raise ConversationAcceptanceError(
            "generic compatible stored profiles must remain incapable"
        )
    _validate_scoped_digest(payload, "Phase 6 provider conformance")


def _validate_phase7_provider_conformance(
    payload: dict[str, object],
    phase6: dict[str, object],
    *,
    phase6_path: Path,
) -> None:
    """Validate exact native compaction profiles and evidence."""
    _exact_keys(
        payload,
        {
            "activation_state",
            "base",
            "canonical_digest",
            "capability_names",
            "capability_states",
            "current_phase",
            "feature",
            "generic_compatible_state",
            "identity_dimensions",
            "owner",
            "production_advertisement_enabled",
            "production_dispatch_enabled",
            "profile_schema_version",
            "profiles",
            "rejected_profile_evidence",
            "schema_version",
        },
        "Phase 7 provider conformance",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("profile_schema_version")
        != "conversation-provider-profile-v1"
        or payload.get("feature") != _FEATURE
        or payload.get("owner") != "provider_runtime"
        or payload.get("current_phase") != 7
        or payload.get("activation_state") != "test_only"
        or payload.get("production_advertisement_enabled") is not False
        or payload.get("production_dispatch_enabled") is not False
        or payload.get("generic_compatible_state") != "incapable"
    ):
        raise ConversationAcceptanceError(
            "Phase 7 provider conformance must remain exact and test-only"
        )
    capability_names = _string_list(
        payload.get("capability_names"),
        "Phase 7 provider capability names",
    )
    if capability_names != _string_list(
        phase6.get("capability_names"),
        "Phase 6 provider capability names",
    ):
        raise ConversationAcceptanceError(
            "Phase 7 provider capabilities differ from the frozen axes"
        )
    phase6_identity_dimensions = _string_list(
        phase6.get("identity_dimensions"),
        "Phase 6 provider identity dimensions",
    )
    identity_dimensions = _string_list(
        payload.get("identity_dimensions"),
        "Phase 7 provider identity dimensions",
    )
    if identity_dimensions != (
        *phase6_identity_dimensions,
        "compaction_limits_digest",
    ):
        raise ConversationAcceptanceError(
            "Phase 7 provider identity must bind exact compaction limits"
        )
    if _string_list(
        payload.get("capability_states"),
        "Phase 7 provider capability states",
    ) != ("test_only", "incapable"):
        raise ConversationAcceptanceError(
            "Phase 7 provider capability states are invalid"
        )
    base = mapping(payload.get("base"), "Phase 7 provider base")
    _exact_keys(
        base,
        {"path", "byte_sha256", "canonical_digest"},
        "Phase 7 provider base",
    )
    phase6_digest = mapping(
        phase6.get("canonical_digest"),
        "Phase 6 provider canonical digest",
    )
    if (
        base.get("path") != _PHASE6_PROVIDER_CONFORMANCE
        or base.get("byte_sha256") != _PHASE6_PROVIDER_CONFORMANCE_BYTE_SHA256
        or sha256(phase6_path.read_bytes()).hexdigest()
        != _PHASE6_PROVIDER_CONFORMANCE_BYTE_SHA256
        or base.get("canonical_digest") != phase6_digest.get("value")
    ):
        raise ConversationAcceptanceError(
            "Phase 7 provider base is not the frozen Phase 6 evidence"
        )

    profile_specs = (
        (
            "native-openai-stateless-non-streaming-phase7-test",
            (
                "avalan.conversation.providers.openai."
                "NativeOpenAIStatelessProvider"
            ),
            "non_streaming",
            False,
            (
                "exact-inline-and-standalone-wire",
                "bounded-profile-limits",
                "loopback-postgresql-latest-boundary-restart",
            ),
            (
                (
                    "tests/conversation/native_openai_compaction_test.py::"
                    "test_native_inline_latest_replay_and_standalone_wire"
                ),
                (
                    "tests/conversation/compaction_contract_test.py::"
                    "test_compaction_models_and_limits_are_closed"
                ),
                (
                    "tests/conversation/compaction_e2e_test.py::"
                    "test_long_stateless_inline_compaction_restarts_at_latest_boundary"
                ),
            ),
        ),
        (
            "native-openai-stateless-streaming-phase7-test",
            (
                "avalan.conversation.providers.openai."
                "NativeOpenAIStatelessProvider"
            ),
            "streaming",
            False,
            (
                "fragmented-compaction-event-assembly",
                "bounded-profile-limits",
                "two-boundary-tool-order",
            ),
            (
                (
                    "tests/conversation/native_openai_compaction_test.py::"
                    "test_streamed_compaction_commits_complete_done_items"
                ),
                (
                    "tests/conversation/compaction_contract_test.py::"
                    "test_compaction_models_and_limits_are_closed"
                ),
                (
                    "tests/conversation/compaction_e2e_test.py::"
                    "test_tool_cycles_across_two_boundaries_keep_exact_final_order"
                ),
            ),
        ),
        (
            "native-openai-stored-non-streaming-phase7-test",
            (
                "avalan.conversation.providers.openai_stored."
                "NativeOpenAIStoredProvider"
            ),
            "non_streaming",
            True,
            (
                "immediate-upstream-parent-only",
                "stored-execution-compaction-limit-digest",
                "loopback-postgresql-provider-chain",
            ),
            (
                (
                    "tests/conversation/native_openai_compaction_test.py::"
                    "test_stored_inline_compaction_only_sends_new_input_and_parent"
                ),
                (
                    "tests/conversation/native_openai_compaction_test.py::"
                    "test_compact_input_and_output_limits_are_enforced"
                ),
                (
                    "tests/conversation/compaction_e2e_test.py::"
                    "test_stored_inline_compaction_uses_only_immediate_upstream_parent"
                ),
            ),
        ),
    )
    profiles = object_list(payload.get("profiles"), "Phase 7 profiles")
    if len(profiles) != len(profile_specs):
        raise ConversationAcceptanceError(
            "Phase 7 compaction provider profile inventory is incomplete"
        )
    capability_inventory = set(capability_names)
    binding_keys = set(identity_dimensions)
    for raw, spec in zip(profiles, profile_specs, strict=True):
        profile = mapping(raw, "Phase 7 compaction provider profile")
        _exact_keys(
            profile,
            {
                "profile_id",
                "lifecycle",
                "active_from_phase",
                "activation_state",
                "identity_complete",
                "binding",
                "capabilities",
                "activation_evidence",
                "evidence_node_ids",
            },
            "Phase 7 compaction provider profile",
        )
        profile_id, adapter, transport, stored, activation, evidence = spec
        if (
            profile.get("profile_id") != profile_id
            or profile.get("lifecycle") != "active"
            or profile.get("active_from_phase") != 7
            or profile.get("activation_state") != "test_only"
            or profile.get("identity_complete") is not True
        ):
            raise ConversationAcceptanceError(
                "Phase 7 compaction provider activation is invalid"
            )
        binding = mapping(profile.get("binding"), "Phase 7 binding")
        _exact_keys(binding, binding_keys, "Phase 7 binding")
        expected_binding = {
            "adapter_type": adapter,
            "provider_family": "openai",
            "normalized_endpoint": "https://api.openai.com/v1",
            "azure_resource_identity": None,
            "model_or_deployment": "gpt-5",
            "provider_api_revision": "openapi-2.3.0",
            "sdk_revision": "openai-python-2.42.0",
            "model_configuration_revision": "model-config-compact",
            "capability_profile_revision": "capability-compact",
            "tool_schema_revision": "tools-compact",
            "execution_definition_revision": "execution-compact",
            "continuation_codec_version": "1",
            "transport": transport,
            "execution_definition_digest": (
                "12b476fdd37c3fb116ff04e344c3dd91361b653d352e75b6b14f376734807500"
                if stored
                else None
            ),
            "compaction_limits_digest": (
                "198b029b6478edd284c816098d82810daa19edef7d927522ac601c2765b6ba7b"
            ),
        }
        if binding != expected_binding:
            raise ConversationAcceptanceError(
                "Phase 7 compaction provider binding is not exact"
            )
        capabilities = mapping(
            profile.get("capabilities"),
            "Phase 7 provider capabilities",
        )
        required = {
            "reasoning_context_current_turn",
            "reasoning_context_all_turns",
            "inline_compaction",
            (
                "stored_responses_chaining"
                if stored
                else "stateless_encrypted_reasoning_replay"
            ),
        }
        if not stored:
            required.add("standalone_compaction")
        else:
            required.update(
                {
                    "stored_response_retrieval",
                    "stored_response_deletion",
                }
            )
        if transport == "streaming":
            required.add("streaming_item_fidelity")
        if (
            set(capabilities) != capability_inventory
            or any(capabilities.get(name) != "test_only" for name in required)
            or any(
                capabilities.get(name) != "incapable"
                for name in capability_inventory - required
            )
        ):
            raise ConversationAcceptanceError(
                "Phase 7 compaction capability evidence is incomplete"
            )
        if (
            _string_list(
                profile.get("activation_evidence"),
                "Phase 7 activation evidence",
            )
            != activation
            or _string_list(
                profile.get("evidence_node_ids"),
                "Phase 7 provider evidence nodes",
            )
            != evidence
        ):
            raise ConversationAcceptanceError(
                "Phase 7 provider evidence is not independently pinned"
            )
    rejected = object_list(
        payload.get("rejected_profile_evidence"),
        "Phase 7 rejected provider evidence",
    )
    if rejected != [
        {
            "provider_family": "openai_compatible",
            "expected_state": "incapable",
            "evidence_node_id": (
                "tests/conversation/native_openai_compaction_test.py::"
                "test_unproven_or_out_of_range_compaction_rejects_before_wire"
            ),
        }
    ]:
        raise ConversationAcceptanceError(
            "generic compatible compaction profiles must remain incapable"
        )
    _validate_scoped_digest(payload, "Phase 7 provider conformance")


def _validate_phase8_provider_conformance(
    payload: dict[str, object],
    phase7: dict[str, object],
    *,
    phase7_path: Path,
) -> None:
    """Validate exact agent tool, suspension, and lane evidence."""
    _exact_keys(
        payload,
        {
            "activation_state",
            "base",
            "canonical_digest",
            "capability_names",
            "capability_states",
            "current_phase",
            "feature",
            "generic_compatible_state",
            "identity_dimensions",
            "owner",
            "production_advertisement_enabled",
            "production_dispatch_enabled",
            "profile_schema_version",
            "profiles",
            "rejected_profile_evidence",
            "schema_version",
        },
        "Phase 8 provider conformance",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("profile_schema_version")
        != "conversation-provider-profile-v1"
        or payload.get("feature") != _FEATURE
        or payload.get("owner") != "provider_runtime"
        or payload.get("current_phase") != 8
        or payload.get("activation_state") != "test_only"
        or payload.get("production_advertisement_enabled") is not False
        or payload.get("production_dispatch_enabled") is not False
        or payload.get("generic_compatible_state") != "incapable"
    ):
        raise ConversationAcceptanceError(
            "Phase 8 provider conformance must remain exact and test-only"
        )
    phase7_capabilities = _string_list(
        phase7.get("capability_names"),
        "Phase 7 provider capability names",
    )
    capability_names = _string_list(
        payload.get("capability_names"),
        "Phase 8 provider capability names",
    )
    new_capabilities = (
        "durable_tool_execution_segments",
        "structured_input_suspension",
        "deterministic_agent_lane_topology",
    )
    if capability_names != (*phase7_capabilities, *new_capabilities):
        raise ConversationAcceptanceError(
            "Phase 8 provider capabilities differ from the frozen axes"
        )
    if _string_list(
        payload.get("identity_dimensions"),
        "Phase 8 provider identity dimensions",
    ) != _string_list(
        phase7.get("identity_dimensions"),
        "Phase 7 provider identity dimensions",
    ):
        raise ConversationAcceptanceError(
            "Phase 8 provider identity dimensions changed"
        )
    if _string_list(
        payload.get("capability_states"),
        "Phase 8 provider capability states",
    ) != ("test_only", "incapable"):
        raise ConversationAcceptanceError(
            "Phase 8 provider capability states are invalid"
        )
    base = mapping(payload.get("base"), "Phase 8 provider base")
    _exact_keys(
        base,
        {"path", "byte_sha256", "canonical_digest"},
        "Phase 8 provider base",
    )
    phase7_digest = mapping(
        phase7.get("canonical_digest"),
        "Phase 7 provider canonical digest",
    )
    if (
        base.get("path") != _PHASE7_PROVIDER_CONFORMANCE
        or base.get("byte_sha256") != _PHASE7_PROVIDER_CONFORMANCE_BYTE_SHA256
        or sha256(phase7_path.read_bytes()).hexdigest()
        != _PHASE7_PROVIDER_CONFORMANCE_BYTE_SHA256
        or base.get("canonical_digest") != phase7_digest.get("value")
    ):
        raise ConversationAcceptanceError(
            "Phase 8 provider base is not the frozen Phase 7 evidence"
        )
    evidence_specs = (
        (
            (
                "durable-tool-segment-recovery",
                "draft-2020-12-tool-schema-zero-effect",
                "typed-structured-input-suspension",
                "actual-agent-lane-restart-isolation",
            ),
            (
                (
                    "tests/conversation/native_openai_provider_test.py::"
                    "test_native_function_cycles_use_the_coordinator_ledger"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_rejects_invalid_schema_"
                    "arguments_before_effect"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_rejects_nonlocal_schema_"
                    "before_effect"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_persists_only_validated_"
                    "arguments"
                ),
                (
                    "tests/conversation/native_openai_provider_test.py::"
                    "test_agent_turn_propagates_typed_structured_input_suspension"
                ),
                (
                    "tests/conversation/agent_integration_pgsql_test.py::"
                    "test_pgsql_tool_boundaries_recover_without_duplicate_"
                    "effect"
                ),
                (
                    "tests/conversation/agent_integration_e2e_test.py::"
                    "test_parent_two_children_persist_isolation_and_restart"
                ),
                (
                    "tests/conversation/agent_integration_e2e_test.py::"
                    "test_child_merge_rejects_wrong_provider_and_model_binding"
                ),
            ),
            "test_only",
        ),
        (
            (
                "streaming-provider-boundary",
                "streaming-tool-effect-zero-commit",
                "draft-2020-12-tool-schema-zero-effect",
                "actual-agent-lane-isolation",
            ),
            (
                (
                    "tests/conversation/native_openai_provider_test.py::"
                    "test_native_stream_matches_non_stream_and_closes"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_output_byte_limit_precedes_tool_effect_and_"
                    "commit"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_rejects_invalid_schema_"
                    "arguments_before_effect"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_rejects_nonlocal_schema_"
                    "before_effect"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_persists_only_validated_"
                    "arguments"
                ),
                (
                    "tests/conversation/agent_integration_e2e_test.py::"
                    "test_parent_two_children_persist_isolation_and_restart"
                ),
            ),
            "incapable",
        ),
        (
            (
                "stored-immediate-parent-tool-segments",
                "draft-2020-12-tool-schema-zero-effect",
                "actual-agent-lane-isolation",
            ),
            (
                (
                    "tests/conversation/native_openai_stored_provider_test.py::"
                    "test_stored_tool_cycle_uses_only_immediate_id_and_tool_output"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_rejects_invalid_schema_"
                    "arguments_before_effect"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_rejects_nonlocal_schema_"
                    "before_effect"
                ),
                (
                    "tests/conversation/"
                    "native_openai_provider_validation_test.py::"
                    "test_native_function_tool_persists_only_validated_"
                    "arguments"
                ),
                (
                    "tests/conversation/agent_integration_e2e_test.py::"
                    "test_parent_two_children_persist_isolation_and_restart"
                ),
            ),
            "incapable",
        ),
    )
    profiles = object_list(payload.get("profiles"), "Phase 8 profiles")
    phase7_profiles = object_list(
        phase7.get("profiles"),
        "Phase 7 profiles",
    )
    if len(profiles) != 3 or len(phase7_profiles) != 3:
        raise ConversationAcceptanceError(
            "Phase 8 agent provider profile inventory is incomplete"
        )
    profile_keys = {
        "profile_id",
        "lifecycle",
        "active_from_phase",
        "activation_state",
        "identity_complete",
        "binding",
        "capabilities",
        "activation_evidence",
        "evidence_node_ids",
    }
    for raw, old_raw, spec in zip(
        profiles,
        phase7_profiles,
        evidence_specs,
        strict=True,
    ):
        profile = mapping(raw, "Phase 8 provider profile")
        old = mapping(old_raw, "Phase 7 provider profile")
        _exact_keys(profile, profile_keys, "Phase 8 provider profile")
        old_id = _nonempty_string(old.get("profile_id"), "Phase 7 profile ID")
        if (
            profile.get("profile_id") != old_id.replace("phase7", "phase8")
            or profile.get("lifecycle") != "active"
            or profile.get("active_from_phase") != 8
            or profile.get("activation_state") != "test_only"
            or profile.get("identity_complete") is not True
            or profile.get("binding") != old.get("binding")
        ):
            raise ConversationAcceptanceError(
                "Phase 8 provider profile identity is invalid"
            )
        capabilities = mapping(
            profile.get("capabilities"),
            "Phase 8 provider capabilities",
        )
        old_capabilities = mapping(
            old.get("capabilities"),
            "Phase 7 provider capabilities",
        )
        if (
            set(capabilities) != set(capability_names)
            or any(
                capabilities.get(name) != old_capabilities.get(name)
                for name in phase7_capabilities
            )
            or capabilities.get("durable_tool_execution_segments")
            != "test_only"
            or capabilities.get("structured_input_suspension") != spec[2]
            or capabilities.get("deterministic_agent_lane_topology")
            != "test_only"
        ):
            raise ConversationAcceptanceError(
                "Phase 8 agent provider capability evidence is invalid"
            )
        if (
            _string_list(
                profile.get("activation_evidence"),
                "Phase 8 activation evidence",
            )
            != spec[0]
            or _string_list(
                profile.get("evidence_node_ids"),
                "Phase 8 provider evidence nodes",
            )
            != spec[1]
        ):
            raise ConversationAcceptanceError(
                "Phase 8 provider evidence is not independently pinned"
            )
    rejected = object_list(
        payload.get("rejected_profile_evidence"),
        "Phase 8 rejected provider evidence",
    )
    if rejected != [
        {
            "provider_family": "openai_compatible",
            "expected_state": "incapable",
            "evidence_node_id": (
                "tests/conversation/native_openai_provider_test.py::"
                "test_unproven_or_drifted_profiles_fail_without_dispatch"
            ),
        }
    ]:
        raise ConversationAcceptanceError(
            "generic compatible agent profiles must remain incapable"
        )
    _validate_scoped_digest(payload, "Phase 8 provider conformance")


def _validate_provider_header(
    payload: dict[str, object],
    label: str,
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("current_phase") != 0
        or payload.get("activation_state") != "dormant"
    ):
        raise ConversationAcceptanceError(
            f"{label} is not the dormant Phase 0 version"
        )


def _validate_scoped_digest(
    payload: dict[str, object],
    label: str,
) -> None:
    digest = mapping(payload.get("canonical_digest"), f"{label} digest")
    _exact_keys(
        digest,
        {"algorithm", "encoding", "scope", "value"},
        f"{label} digest",
    )
    if digest.get("algorithm") != "sha256":
        raise ConversationAcceptanceError(
            f"{label} digest algorithm must be sha256"
        )
    scope = _string_list(digest.get("scope"), f"{label} digest scope")
    expected_scope = tuple(
        field for field in payload if field != "canonical_digest"
    )
    if scope != expected_scope:
        raise ConversationAcceptanceError(f"{label} digest scope is invalid")
    scoped = {field: payload[field] for field in scope}
    if digest.get("value") != canonical_sha256(scoped):
        raise ConversationAcceptanceError(f"{label} digest is invalid")


def _validate_type_manifest(
    fixtures: Path,
    current_phase: int,
    root: Path,
    *,
    acceptance_path: Path | None = None,
) -> None:
    """Reuse complete type-manifest and source-anchor validation."""
    path = (
        companion_fixture_path(acceptance_path, "type_contract_manifest")
        if acceptance_path is not None
        else fixtures / "type_contract_manifest.json"
    )
    if not path.is_file():
        raise ConversationAcceptanceError(
            "conversation type-contract manifest is missing"
        )
    try:
        manifest = load_type_contract_manifest(path)
        if manifest.current_phase != current_phase:
            raise ConversationAcceptanceError(
                "type and acceptance manifest phases differ"
            )
        validate_type_source_phase_anchors(manifest, root)
    except (
        ContractGateError,
        ConversationTypeContractError,
        StrictJsonError,
    ) as exc:
        raise ConversationAcceptanceError(
            f"type-contract validation failed: {exc}"
        ) from exc


def _validate_activation_history(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> tuple[tuple[str, ...], ...]:
    history = object_list(raw, "activation history")
    _require_phase_anchor_keys(
        _ACTIVATION_HISTORY_BY_PHASE,
        current_phase,
        "activation history",
    )
    if len(history) != current_phase + 1:
        raise ConversationAcceptanceError(
            "activation history must preserve every implemented phase"
        )
    previous: set[str] = set()
    snapshots: list[tuple[str, ...]] = []
    for expected_phase, raw_entry in enumerate(history):
        entry = mapping(raw_entry, "activation history entry")
        _exact_keys(entry, {"phase", "node_ids", "sha256"}, "activation entry")
        if _phase(entry.get("phase"), "activation phase") != expected_phase:
            raise ConversationAcceptanceError(
                "activation history phases must be contiguous"
            )
        node_ids = _string_list(entry.get("node_ids"), "activation node IDs")
        _unique(node_ids, "activation node ID")
        expected_ids = tuple(
            node.node_id
            for node in nodes
            if node.lifecycle in {"active", "replaced"}
            and node.active_from_phase <= expected_phase
        )
        if node_ids != expected_ids or not previous <= set(node_ids):
            raise ConversationAcceptanceError(
                "activation history is not monotonic at phase"
                f" {expected_phase}"
            )
        if entry.get("sha256") != _text_digest(node_ids):
            raise ConversationAcceptanceError(
                "activation history digest is invalid at phase"
                f" {expected_phase}"
            )
        if entry.get("sha256") != _ACTIVATION_HISTORY_BY_PHASE[expected_phase]:
            raise ConversationAcceptanceError(
                "activation history differs from its immutable phase "
                f"anchor at phase {expected_phase}"
            )
        previous = set(node_ids)
        snapshots.append(node_ids)
    return tuple(snapshots)


def _validate_node_phase_anchors(
    raw_nodes: list[object],
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> None:
    """Validate independently anchored node payloads by activation phase."""
    _require_phase_anchor_keys(
        _NODE_PAYLOAD_SHA256_BY_PHASE,
        current_phase,
        "acceptance node payload",
    )
    for phase in range(current_phase + 1):
        payload = [
            {
                key: value
                for key, value in mapping(raw, "acceptance node").items()
                if key != "lifecycle"
            }
            for raw, node in zip(raw_nodes, nodes, strict=True)
            if node.active_from_phase == phase
        ]
        if canonical_sha256(payload) != _NODE_PAYLOAD_SHA256_BY_PHASE[phase]:
            raise ConversationAcceptanceError(
                "acceptance node payload differs from its independent "
                f"phase anchor at phase {phase}"
            )


def _require_phase_anchor_keys(
    anchors: Mapping[int, object],
    current_phase: int,
    label: str,
) -> None:
    """Require one append-only independent anchor per implemented phase."""
    expected = set(range(current_phase + 1))
    if not expected <= set(anchors):
        raise ConversationAcceptanceError(
            f"{label} anchors must cover every implemented phase"
        )


def _validate_replacements(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> tuple[AcceptanceReplacement, ...]:
    replacements = object_list(raw, "acceptance replacements")
    current_ids = {node.node_id for node in nodes}
    parsed: list[AcceptanceReplacement] = []
    old_ids: list[str] = []
    targets: list[str] = []
    phases: list[int] = []
    for raw_entry in replacements:
        entry = mapping(raw_entry, "acceptance replacement")
        _exact_keys(
            entry,
            {
                "phase",
                "old_node_id",
                "replacement_node_ids",
                "reviewed_by",
                "evidence",
            },
            "acceptance replacement",
        )
        phase = _phase(entry.get("phase"), "replacement phase")
        if phase > current_phase:
            raise ConversationAcceptanceError(
                "future replacements cannot alter activation history"
            )
        old = _test_node(entry.get("old_node_id"))
        replacements_for_old = _string_list(
            entry.get("replacement_node_ids"), "replacement nodes"
        )
        if (
            old not in current_ids
            or not replacements_for_old
            or not set(replacements_for_old) <= current_ids
            or old in replacements_for_old
        ):
            raise ConversationAcceptanceError(
                "replacement tombstone differs from current inventory"
            )
        old_ids.append(old)
        targets.extend(replacements_for_old)
        phases.append(phase)
        parsed.append(
            AcceptanceReplacement(
                phase=phase,
                old_node_id=old,
                replacement_node_ids=replacements_for_old,
            )
        )
        _nonempty_string(entry.get("reviewed_by"), "replacement reviewer")
        _nonempty_string(entry.get("evidence"), "replacement evidence")
    _unique(old_ids, "replaced node ID")
    _unique(targets, "replacement target")
    _validate_replacement_phase_anchors(
        replacements,
        tuple(phases),
        current_phase,
    )
    return tuple(parsed)


def _validate_replacement_transitions(
    replacements: tuple[AcceptanceReplacement, ...],
    nodes: tuple[AcceptanceNode, ...],
    activation_history: tuple[tuple[str, ...], ...],
) -> None:
    """Validate retained tombstones against adjacent activation snapshots."""
    node_by_id = {node.node_id: node for node in nodes}
    replacement_by_old = {
        replacement.old_node_id: replacement for replacement in replacements
    }
    replaced_ids = {
        node.node_id for node in nodes if node.lifecycle == "replaced"
    }
    if replaced_ids != set(replacement_by_old):
        raise ConversationAcceptanceError(
            "replaced acceptance records and reviewed ledger entries differ"
        )
    for replacement in replacements:
        if replacement.phase == 0:
            raise ConversationAcceptanceError(
                "acceptance replacements require a preceding phase snapshot"
            )
        old = node_by_id[replacement.old_node_id]
        previous = set(activation_history[replacement.phase - 1])
        current = set(activation_history[replacement.phase])
        additions = current - previous
        same_phase_split = old.active_from_phase == replacement.phase
        retained_prior = (
            old.active_from_phase < replacement.phase
            and replacement.old_node_id in previous
        )
        introduced_split = (
            same_phase_split
            and replacement.old_node_id not in previous
            and replacement.old_node_id in additions
        )
        if old.lifecycle != "replaced" or not (
            retained_prior or introduced_split
        ):
            raise ConversationAcceptanceError(
                "acceptance replacement old record is neither a retained "
                "prior member nor a reviewed same-phase split"
            )
        target_requirement_sets: list[set[str]] = []
        for target_id in replacement.replacement_node_ids:
            target = node_by_id[target_id]
            if (
                target.active_from_phase != replacement.phase
                or target.lifecycle not in {"active", "replaced"}
                or target_id not in additions
            ):
                raise ConversationAcceptanceError(
                    "acceptance replacement targets must be new same-phase "
                    "records"
                )
            target_requirement_sets.append(set(target.requirement_ids))
        old_requirements = set(old.requirement_ids)
        exact_cover = (
            all(target_requirement_sets)
            and set().union(*target_requirement_sets) == old_requirements
            and all(
                requirements <= old_requirements
                for requirements in target_requirement_sets
            )
        )
        if not exact_cover:
            raise ConversationAcceptanceError(
                "acceptance replacement targets must form an exact "
                "nonempty cover of preserved requirement ownership"
            )


def _validate_replacement_phase_anchors(
    replacements: list[object],
    phases: tuple[int, ...],
    current_phase: int,
) -> None:
    """Validate cumulative append-only acceptance replacement history."""
    _require_phase_anchor_keys(
        _REPLACEMENT_HISTORY_BY_PHASE,
        current_phase,
        "acceptance replacement history",
    )
    previous_count = 0
    for phase in range(current_phase + 1):
        count, expected_sha256 = _REPLACEMENT_HISTORY_BY_PHASE[phase]
        if (
            count < previous_count
            or count > len(replacements)
            or any(value > phase for value in phases[:count])
            or any(value <= phase for value in phases[count:])
        ):
            raise ConversationAcceptanceError(
                "acceptance replacement history anchors are not append-only"
            )
        if canonical_sha256(replacements[:count]) != expected_sha256:
            raise ConversationAcceptanceError(
                "acceptance replacement history differs from its immutable "
                f"phase anchor at phase {phase}"
            )
        previous_count = count
    if previous_count != len(replacements):
        raise ConversationAcceptanceError(
            "acceptance replacement history has unanchored appended payload"
        )


def _strict_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        return mapping(strict_json_path(path), label)
    except (ContractGateError, StrictJsonError) as exc:
        raise ConversationAcceptanceError(
            f"cannot read {label}: {exc}"
        ) from exc


def _header(payload: dict[str, object], label: str) -> None:
    if payload.get("schema_version") != 1:
        raise ConversationAcceptanceError(f"{label} schema_version must be 1")
    if payload.get("feature") != _FEATURE:
        raise ConversationAcceptanceError(
            f"{label} feature must be {_FEATURE}"
        )


def _phase(value: object, label: str) -> int:
    if type(value) is not int or not _MIN_PHASE <= value <= _MAX_PHASE:
        raise ConversationAcceptanceError(
            f"{label} must be an integer from {_MIN_PHASE} through"
            f" {_MAX_PHASE}"
        )
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ConversationAcceptanceError(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConversationAcceptanceError(
            f"{label} must be a positive integer"
        )
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationAcceptanceError(
            f"{label} must be a non-empty string"
        )
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    return tuple(
        _nonempty_string(item, label) for item in object_list(value, label)
    )


def _relative_path(value: object, label: str) -> str:
    raw = _nonempty_string(value, label)
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ConversationAcceptanceError(f"{label} escapes the repository")
    return raw


def _test_node(value: object) -> str:
    node_id = _nonempty_string(value, "pytest node ID")
    relative = node_id.split("::", 1)[0]
    if (
        _NODE_PATTERN.fullmatch(node_id) is None
        or "\\" in node_id
        or ".." in PurePosixPath(relative).parts
    ):
        raise ConversationAcceptanceError(f"invalid pytest node ID: {node_id}")
    return node_id


def _unique(values: Iterable[object], label: str) -> None:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise ConversationAcceptanceError(f"duplicate {label}")


def _exact_keys(
    value: dict[str, object],
    expected: Iterable[str],
    label: str,
) -> None:
    expected_keys = set(expected)
    if set(value) != expected_keys:
        raise ConversationAcceptanceError(
            f"{label} has invalid keys: {sorted(set(value) ^ expected_keys)}"
        )


def _text_digest(values: tuple[str, ...]) -> str:
    return sha256("\n".join(values).encode("utf-8")).hexdigest()


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Collect and execute active conversation acceptance nodes without "
            "skips, xfails, deselection, or placeholder evidence."
        )
    )
    parser.add_argument("--through-phase", required=True, type=int)
    parser.add_argument(
        "--manifest", type=Path, default=default_manifest_path()
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run conversation acceptance verification from the command line."""
    args = _parse_args()
    try:
        manifest = verify_acceptance(
            args.manifest,
            repo_root=args.repo_root,
            through_phase=args.through_phase,
            execute=not args.validate_only,
        )
    except (
        ContractGateError,
        ConversationAcceptanceError,
        StrictJsonError,
    ) as exc:
        print(f"conversation acceptance failed: {exc}", file=stderr)
        return 1
    active = len(manifest.active_nodes(args.through_phase))
    planned = len(manifest.planned_nodes())
    print(
        "conversation acceptance passed: "
        f"through_phase={args.through_phase} active={active} planned={planned}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
