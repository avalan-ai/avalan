"""Validate executable conversation documentation and tracked contracts."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import fields
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from runpy import run_path
from typing import cast

import pytest

import avalan.conversation as conversation
from avalan.server.entities import ResponsesCompactRequest, ResponsesRequest

pytestmark = pytest.mark.anyio

_ROOT = Path(__file__).parents[2]
_FIXTURE = (
    _ROOT
    / "tests"
    / "fixtures"
    / "conversation"
    / "documentation_contract.phase12.json"
)
_ExampleResult = dict[str, object]
_ExampleRun = Callable[[], Awaitable[_ExampleResult]]


@pytest.fixture
def anyio_backend() -> str:
    """Run documentation examples on asyncio only."""
    return "asyncio"


def _json(path: Path) -> dict[str, object]:
    """Load one tracked JSON object."""
    return cast(dict[str, object], loads(path.read_text(encoding="utf-8")))


def _contract() -> dict[str, object]:
    """Return the tracked documentation contract."""
    return _json(_FIXTURE)


def _strings(value: object) -> tuple[str, ...]:
    """Return one exact JSON string sequence."""
    assert isinstance(value, list)
    assert all(type(item) is str for item in value)
    return tuple(cast(list[str], value))


def _mapping(value: object) -> Mapping[str, object]:
    """Return one exact JSON object view."""
    assert isinstance(value, dict)
    assert all(type(key) is str for key in value)
    return cast(Mapping[str, object], value)


def _document_texts() -> dict[str, str]:
    """Return every document named by the tracked contract."""
    paths = _strings(_contract()["documents"])
    return {path: (_ROOT / path).read_text(encoding="utf-8") for path in paths}


def _load_example(path: str, entrypoint: str) -> _ExampleRun:
    """Load one tracked example entrypoint without executing its main block."""
    namespace = run_path(str(_ROOT / path), run_name="conversation_docs_test")
    return cast(_ExampleRun, namespace[entrypoint])


def test_documentation_contract_has_canonical_integrity() -> None:
    """Detect unreviewed drift in the tracked documentation contract."""
    contract = _contract()
    expected = cast(str, contract.pop("canonical_sha256"))
    actual = sha256(
        dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert actual == expected


async def test_every_documented_example_executes_locally() -> None:
    """Run every tracked SDK and served example without network access."""
    examples = cast(list[dict[str, str]], _contract()["examples"])
    assert {item["path"] for item in examples} == {
        "docs/examples/conversation_continuity_sdk.py",
        "docs/examples/conversation_responses_local.py",
    }
    results = {
        item["path"]: await _load_example(item["path"], item["entrypoint"])()
        for item in examples
    }

    sdk = results["docs/examples/conversation_continuity_sdk.py"]
    assert sdk["stream_event_types"] == [
        "DirectConversationOutputDelta",
        "DirectConversationStreamTerminal",
    ]
    assert sdk["terminal_handle_committed"] is True
    assert sdk["branch_isolated"] is True
    assert sdk["compact_committed"] is True
    assert sdk["reset_is_fresh_root"] is True

    served = results["docs/examples/conversation_responses_local.py"]
    assert served["retrieved_same_public_id"] is True
    assert served["deleted"] is True
    assert served["stream_event_types"] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    rendered = repr(results)
    assert conversation.CONTINUATION_ENVELOPE_PREFIX not in rendered
    assert "encrypted_content" not in rendered
    assert "upstream" not in rendered


def test_generated_schema_and_configuration_match_fixture() -> None:
    """Pin strict served schemas and public settings to tracked fields."""
    contract = _contract()
    request_schema = ResponsesRequest.model_json_schema()
    compact_schema = ResponsesCompactRequest.model_json_schema()

    assert request_schema["additionalProperties"] is False
    assert request_schema["properties"]["store"]["default"] is False
    assert sorted(ResponsesRequest.model_fields) == list(
        _strings(contract["served_request_fields"])
    )
    assert sorted(ResponsesCompactRequest.model_fields) == list(
        _strings(contract["served_compact_fields"])
    )
    assert set(compact_schema["required"]) == {"model"}
    assert sorted(
        field.name
        for field in fields(conversation.StatelessConversationSettings)
    ) == list(_strings(contract["stateless_settings_fields"]))
    assert sorted(
        field.name for field in fields(conversation.StoredConversationSettings)
    ) == list(_strings(contract["stored_settings_fields"]))

    openapi = _json(_ROOT / cast(str, contract["openapi_fixture"]))
    operations = _mapping(openapi["operations"])
    expected = _mapping(contract["openapi_operations"])
    assert set(operations) == set(expected)
    for name, expected_pair_value in expected.items():
        operation = _mapping(operations[name])
        expected_pair = _strings(expected_pair_value)
        assert (operation["method"], operation["path"]) == expected_pair


def test_error_catalog_and_surface_dispositions_match_runtime() -> None:
    """Keep documented errors and intentionally deferred surfaces exact."""
    contract = _contract()
    error_codes = tuple(
        item.value for item in conversation.ConversationErrorCode
    )
    durable_codes = tuple(
        item.value for item in conversation.DurableConversationErrorCode
    )
    assert error_codes == _strings(contract["error_codes"])
    assert durable_codes == _strings(contract["durable_error_codes"])

    expected_surfaces = _mapping(contract["surface_dispositions"])
    actual_surfaces = {
        surface.value: (
            conversation.agent_conversation_surface_disposition(surface).value
        )
        for surface in conversation.ConversationSurface
    }
    assert actual_surfaces == expected_surfaces

    guide = _document_texts()["docs/CONVERSATIONS.md"]
    for code in (*error_codes, *durable_codes):
        assert f"`{code}`" in guide
    for surface, disposition in actual_surfaces.items():
        if disposition == "deferred":
            assert surface.upper() in guide or surface.title() in guide


def test_capability_table_matches_inactive_activation_fixture() -> None:
    """Prevent a pending-review inactive decision from becoming activation."""
    contract = _contract()
    expectation = _mapping(contract["activation_expectation"])
    activation = _json(_ROOT / cast(str, expectation["fixture"]))
    review = _mapping(activation["review"])

    for field in (
        "activation_state",
        "production_dispatch_enabled",
        "production_advertisement_enabled",
    ):
        assert activation[field] == expectation[field]
    assert len(cast(list[object], activation["active_production_rows"])) == (
        expectation["active_production_rows"]
    )
    assert len(cast(list[object], activation["live_proof_ids"])) == (
        expectation["live_proof_ids"]
    )
    assert review["status"] == expectation["review_status"]
    candidate_states = {
        cast(str, item["provider_family"]): cast(str, item["state"])
        for item in cast(
            list[dict[str, object]], activation["candidate_lanes"]
        )
    }
    assert candidate_states == _mapping(expectation["provider_states"])

    canonical = dict(activation)
    canonical_digest = _mapping(canonical.pop("canonical_digest"))
    assert (
        sha256(
            dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == canonical_digest["value"]
    )
    signed = dict(activation)
    review_signature = _mapping(signed.pop("review_signature"))
    signed.pop("canonical_digest")
    assert (
        sha256(
            dumps(signed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == review_signature["value"]
    )

    live_link = _mapping(activation["live_evidence"])
    live_path = (
        _ROOT
        / "tests"
        / "fixtures"
        / "conversation"
        / cast(str, live_link["path"])
    )
    live_payload = _json(live_path)
    live_digest = _mapping(live_payload["canonical_digest"])
    assert (
        sha256(live_path.read_bytes()).hexdigest() == live_link["byte_sha256"]
    )
    assert live_digest["value"] == live_link["canonical_digest"]
    live_expectation = _mapping(contract["live_evidence_expectation"])
    assert live_path == _ROOT / cast(str, live_expectation["fixture"])
    assert live_payload["observed_at"] == live_expectation["observed_at"]
    assert tuple(cast(list[str], live_payload["provider_families"])) == (
        _strings(live_expectation["provider_families"])
    )
    native_openai = _mapping(live_payload["native_openai_attempt"])
    native_execution = _mapping(native_openai["matrix_execution"])
    assert native_openai["total_http_call_count"] == (
        live_expectation["native_openai_total_http_calls"]
    )
    assert len(cast(list[object], native_execution["completed_cases"])) == (
        live_expectation["native_openai_completed_cases"]
    )
    assert native_execution["state"] == live_expectation["native_openai_state"]
    azure_openai = _mapping(live_payload["azure_openai_matrix"])
    azure_results = cast(list[dict[str, object]], azure_openai["results"])
    assert len(azure_results) == (live_expectation["evaluated_azure_profiles"])
    terra = next(
        row for row in azure_results if row["deployment"] == "gpt-5.6-terra"
    )
    receipt = _mapping(terra["tracked_cli_receipt"])
    accounting = _mapping(receipt["accounting"])
    assert accounting["logical_operation_count"] == (
        live_expectation["azure_tracked_cli_logical_operations"]
    )
    assert accounting["http_request_count"] == (
        live_expectation["azure_tracked_cli_http_requests"]
    )
    assert accounting["sdk_configured_max_retries"] == (
        live_expectation["azure_tracked_cli_sdk_configured_max_retries"]
    )
    assert accounting["observed_sdk_retry_count"] == (
        live_expectation["azure_tracked_cli_observed_sdk_retries"]
    )
    assert accounting["unexpected_request_count"] == (
        live_expectation["azure_tracked_cli_unexpected_requests"]
    )
    assert accounting["cleanup_completed"] == (
        live_expectation["azure_tracked_cli_cleanup_completed"]
    )
    assert live_expectation["evaluated_openai_profiles"] == 1
    assert live_payload["completed_full_matrix_profile_count"] == (
        live_expectation["completed_full_matrix_profiles"]
    )
    assert (
        live_payload["active_profile_count"]
        == live_expectation["active_profiles"]
    )
    assert (
        live_payload["activation_decision"]
        == live_expectation["activation_decision"]
    )

    zero_active = _mapping(activation["zero_active_evidence_mapping"])
    assert zero_active["mapping_state"] == "non_vacuous_fail_closed"
    assert zero_active["evaluated_openai_profile_count"] == 1
    assert zero_active["evaluated_azure_profile_count"] == 6
    assert zero_active["completed_live_profile_count"] == 1
    assert zero_active["active_row_count"] == 0
    public_e2e = _mapping(zero_active["public_e2e"])
    assert public_e2e == {
        "id": "CONV-E2E-015",
        "state": "incomplete_cross_provider_live_matrix",
        "completed_live_profile_count": 1,
        "negative_evidence_nodes": [
            (
                "tests/conversation/full_matrix_e2e_test.py::"
                "test_required_matrix_cross_product"
            ),
            (
                "tests/conversation/native_openai_provider_test.py::"
                "test_unproven_or_drifted_profiles_fail_without_dispatch"
            ),
        ],
    }
    for node in _strings(zero_active["deterministic_evidence_nodes"]):
        path_value, test_name = node.split("::", maxsplit=1)
        source = (_ROOT / path_value).read_text(encoding="utf-8")
        assert f"def {test_name}(" in source

    guide = _document_texts()["docs/CONVERSATIONS.md"].casefold()
    assert "inactive pending-review decision" in guide
    assert "`active_production_rows` is" in guide
    assert "empty" in guide
    assert "generic openai-compatible" in guide
    assert "incapable; reject before dispatch" in guide


def test_provider_evidence_sources_are_dated_and_exact() -> None:
    """Tie provider claims to tracked documentation and SDK evidence."""
    contract = _contract()
    expectation = _mapping(contract["provider_evidence_expectation"])
    evidence = _json(_ROOT / cast(str, expectation["fixture"]))
    sdk = _mapping(evidence["sdk"])
    activation = _json(
        _ROOT
        / cast(
            str,
            _mapping(contract["activation_expectation"])["fixture"],
        )
    )

    assert evidence["accessed_at"] == expectation["accessed_at"]
    assert sdk["distribution"] == expectation["sdk_distribution"]
    assert sdk["installed_version"] == expectation["installed_version"]
    assert sdk["declared_supported_range"] == expectation["sdk_range"]
    assert sdk["untyped_extra_body_permitted"] is False
    assert activation["sdk_range"] == expectation["sdk_range"]
    source_urls = {
        cast(str, source["url"])
        for provider in cast(list[dict[str, object]], evidence["providers"])
        for source in cast(list[dict[str, object]], provider["sources"])
    }
    assert source_urls == set(_strings(expectation["source_urls"]))
    assert all(
        source["accessed_at"] == expectation["accessed_at"]
        for provider in cast(list[dict[str, object]], evidence["providers"])
        for source in cast(list[dict[str, object]], provider["sources"])
    )


def test_security_migration_and_operator_contracts_are_complete() -> None:
    """Require every security, migration, and operations topic."""
    contract = _contract()
    documents = _document_texts()
    security = documents["docs/CONVERSATION_SECURITY.md"].casefold()
    operations = documents["docs/CONVERSATION_OPERATIONS.md"].casefold()
    migration = documents["docs/CONVERSATION_MIGRATION_V1.md"].casefold()

    for term in _strings(contract["security_terms"]):
        assert term in security
    for term in _strings(contract["operator_terms"]):
        assert term in operations
    for path in _strings(contract["migration_paths"]):
        assert f"## {path}" in migration
    assert "visible transcript" in migration
    assert "explicit reset" in migration
    assert "provider continuity not implied" in migration


def test_runbook_commands_reference_real_nodes_and_tracked_fixtures() -> None:
    """Keep activation and rollback commands executable."""
    contract = _contract()
    operations = _document_texts()["docs/CONVERSATION_OPERATIONS.md"]
    for node in _strings(contract["runbook_pytest_nodes"]):
        path_value, test_name = node.split("::", maxsplit=1)
        path = _ROOT / path_value
        assert path.is_file()
        assert f"def {test_name}(" in path.read_text(encoding="utf-8")
        assert f"poetry run pytest -q {node}" in operations
    for expectation_name in (
        "activation_expectation",
        "live_evidence_expectation",
        "provider_evidence_expectation",
    ):
        expectation = _mapping(contract[expectation_name])
        assert (_ROOT / cast(str, expectation["fixture"])).is_file()


def test_documentation_and_examples_are_safe_and_indexed() -> None:
    """Reject stale fields, unsafe tokens, and unsupported claims."""
    contract = _contract()
    documents = _document_texts()
    examples = cast(list[dict[str, str]], contract["examples"])
    combined_docs = "\n".join(documents.values())
    combined_examples = "\n".join(
        (_ROOT / item["path"]).read_text(encoding="utf-8") for item in examples
    )
    index = (_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    example_index = (_ROOT / "docs" / "examples" / "README.md").read_text(
        encoding="utf-8"
    )

    for path in _strings(contract["documents"]):
        assert Path(path).name in index
    for item in examples:
        assert Path(item["path"]).name in example_index
        assert item["command"] in combined_docs

    for forbidden in (
        "extra_body",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "api.openai.com",
        ".openai.azure.com",
        conversation.CONTINUATION_ENVELOPE_PREFIX,
    ):
        assert forbidden not in combined_examples
    assert "continuation_envelope=" not in combined_docs
    assert "silently switch" not in combined_docs.casefold()
    assert (
        "all native provider rows are active" not in combined_docs.casefold()
    )
    assert "inactive pending-review decision" in combined_docs.casefold()
    assert "store` defaults to `false`" in combined_docs
