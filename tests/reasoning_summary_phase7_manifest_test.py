"""Test the exact Phase 7 reasoning-summary acceptance inventory."""

from asyncio import run
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from sys import modules
from types import ModuleType
from typing import Any, cast

from reasoning_summary_script_loader import load_reasoning_summary_script

_ACCEPTANCE_SCRIPT = load_reasoning_summary_script(
    "verify_reasoning_summary_acceptance"
)
_PHASE7_DIMENSION = "phase 7 OpenAI-compatible Responses ingress and egress"
_PHASE7_TEST = "tests/reasoning_summary_phase7_manifest_test.py::"
_PHASE7_NODE_IDS = (
    f"{_PHASE7_TEST}test_phase7_acceptance_manifest_pins_exact_catalog",
    f"{_PHASE7_TEST}test_phase7_ingress_acceptance",
    f"{_PHASE7_TEST}test_phase7_stream_representation_acceptance",
    f"{_PHASE7_TEST}test_phase7_allocator_and_coalescing_acceptance",
    f"{_PHASE7_TEST}test_phase7_abnormal_terminal_acceptance",
    f"{_PHASE7_TEST}test_phase7_source_lifecycle_acceptance",
    f"{_PHASE7_TEST}test_phase7_retention_admission_acceptance",
    f"{_PHASE7_TEST}test_phase7_redaction_identity_acceptance",
    f"{_PHASE7_TEST}test_phase7_non_stream_structure_acceptance",
    f"{_PHASE7_TEST}test_phase7_non_stream_outcome_acceptance",
    f"{_PHASE7_TEST}test_phase7_stream_non_stream_parity_acceptance",
    f"{_PHASE7_TEST}test_phase7_cross_protocol_golden_acceptance",
)
_RESPONSES_CONTRACT = (
    "server/responses_phase7_contract_test.py",
    "ResponsesPhase7ContractTestCase",
)
_SERVER_MODULE_PREFIX = "avalan.server"


def _is_server_module_name(module_name: str) -> bool:
    return module_name == _SERVER_MODULE_PREFIX or module_name.startswith(
        f"{_SERVER_MODULE_PREFIX}."
    )


def _snapshot_server_module_state() -> tuple[
    dict[str, ModuleType],
    dict[str, tuple[ModuleType, bool, Any]],
]:
    server_modules = {
        module_name: module
        for module_name, module in modules.items()
        if _is_server_module_name(module_name)
    }
    parent_bindings: dict[str, tuple[ModuleType, bool, Any]] = {}
    for module_name in server_modules:
        parent_name, _, attribute_name = module_name.rpartition(".")
        parent = modules.get(parent_name)
        if isinstance(parent, ModuleType):
            has_attribute = hasattr(parent, attribute_name)
            parent_bindings[module_name] = (
                parent,
                has_attribute,
                getattr(parent, attribute_name, None),
            )
    return server_modules, parent_bindings


def _restore_server_module_state(
    server_modules: dict[str, ModuleType],
    parent_bindings: dict[str, tuple[ModuleType, bool, Any]],
) -> None:
    wrapper_modules = {
        module_name: module
        for module_name, module in modules.items()
        if _is_server_module_name(module_name)
    }
    for module_name in wrapper_modules:
        modules.pop(module_name, None)
    modules.update(server_modules)

    for module_name, module in wrapper_modules.items():
        if module_name in server_modules:
            continue
        parent_name, _, attribute_name = module_name.rpartition(".")
        parent = modules.get(parent_name)
        if (
            isinstance(parent, ModuleType)
            and getattr(parent, attribute_name, None) is module
        ):
            delattr(parent, attribute_name)

    for module_name, (
        parent,
        has_attribute,
        value,
    ) in parent_bindings.items():
        attribute_name = module_name.rpartition(".")[2]
        if has_attribute:
            setattr(parent, attribute_name, value)
        elif hasattr(parent, attribute_name):
            delattr(parent, attribute_name)


def _load_test_module(relative_path: str) -> ModuleType:
    assert isinstance(relative_path, str) and relative_path
    path = Path(__file__).resolve().parent / relative_path
    module_name = f"_phase7_acceptance_{path.stem}"
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        modules.pop(module_name, None)
    return module


async def _run_async_test_case_method(
    relative_path: str,
    class_name: str,
    method_name: str,
) -> None:
    server_modules, parent_bindings = _snapshot_server_module_state()
    try:
        module = _load_test_module(relative_path)
        test_class = getattr(module, class_name)
        test_case = test_class(methodName=method_name)
        test_case.setUp()
        try:
            await getattr(test_case, method_name)()
        finally:
            test_case.tearDown()
    finally:
        _restore_server_module_state(server_modules, parent_bindings)


async def _run_acceptance_methods(
    methods: tuple[tuple[str, str, str], ...],
) -> None:
    for method in methods:
        await _run_async_test_case_method(*method)


def _contract_method(method_name: str) -> tuple[str, str, str]:
    return (*_RESPONSES_CONTRACT, method_name)


def test_phase7_wrapped_methods_restore_server_module_identity() -> None:
    import_module("avalan.server")
    import_module("avalan.server.routers.responses")
    before_modules, before_bindings = _snapshot_server_module_state()
    original_loader = globals()["_load_test_module"]

    def assert_identity_preserved() -> None:
        after_modules, after_bindings = _snapshot_server_module_state()
        assert after_modules.keys() == before_modules.keys()
        for module_name, module in before_modules.items():
            assert after_modules[module_name] is module
        assert after_bindings.keys() == before_bindings.keys()
        for module_name, binding in before_bindings.items():
            after_parent, after_has_attribute, after_value = after_bindings[
                module_name
            ]
            parent, has_attribute, value = binding
            assert after_parent is parent
            assert after_has_attribute is has_attribute
            assert after_value is value
        assert "avalan.server.wrapper_created" not in modules

    for failure_stage in (None, "setUp", "method", "tearDown"):
        synthetic_module = ModuleType("_phase7_synthetic_isolation")

        class SyntheticTestCase:
            def __init__(self, **kwargs: str) -> None:
                assert kwargs["methodName"] == "exercise"

            @staticmethod
            def _pollute_modules(stage: str) -> None:
                for module_name in tuple(modules):
                    if _is_server_module_name(module_name):
                        modules.pop(module_name, None)
                server = ModuleType("avalan.server")
                wrapper_created = ModuleType("avalan.server.wrapper_created")
                modules["avalan.server"] = server
                modules["avalan.server.wrapper_created"] = wrapper_created
                avalan = modules["avalan"]
                setattr(avalan, "server", server)
                setattr(server, "wrapper_created", wrapper_created)
                setattr(wrapper_created, "stage", stage)

            def setUp(self) -> None:
                self._pollute_modules("setUp")
                if failure_stage == "setUp":
                    raise RuntimeError(failure_stage)

            async def exercise(self) -> None:
                self._pollute_modules("method")
                if failure_stage == "method":
                    raise RuntimeError(failure_stage)

            def tearDown(self) -> None:
                self._pollute_modules("tearDown")
                if failure_stage == "tearDown":
                    raise RuntimeError(failure_stage)

        setattr(synthetic_module, "SyntheticTestCase", SyntheticTestCase)

        def load_synthetic_module(_relative_path: str) -> ModuleType:
            return synthetic_module

        globals()["_load_test_module"] = load_synthetic_module
        try:
            run(
                _run_async_test_case_method(
                    "unused.py",
                    "SyntheticTestCase",
                    "exercise",
                )
            )
        except RuntimeError as error:
            assert failure_stage is not None
            assert str(error) == failure_stage
        else:
            assert failure_stage is None
        finally:
            globals()["_load_test_module"] = original_loader
        assert_identity_preserved()


def test_phase7_ingress_acceptance() -> None:
    run(
        _run_acceptance_methods(
            (
                (
                    "server/responses_test.py",
                    "ResponsesEndpointTestCase",
                    "test_response_endpoint_rejects_invalid_summary_with_422",
                ),
                (
                    "server/responses_test.py",
                    "ResponsesEndpointTestCase",
                    (
                        "test_response_endpoint_rejects_unsupported_before_"
                        "sse_headers"
                    ),
                ),
                (
                    "server/chat_router_unit_test.py",
                    "ChatRouterUnitTest",
                    (
                        "test_create_response_forwards_only_explicit_"
                        "reasoning_fields"
                    ),
                ),
            )
        )
    )


def test_phase7_stream_representation_acceptance() -> None:
    run(
        _run_acceptance_methods(
            (
                _contract_method("test_summary_uses_exact_summary_lifecycle"),
                _contract_method(
                    "test_native_reasoning_remains_reasoning_text"
                ),
            )
        )
    )


def test_phase7_allocator_and_coalescing_acceptance() -> None:
    run(
        _run_acceptance_methods(
            (
                _contract_method(
                    "test_allocators_span_continuations_and_item_kinds"
                ),
                _contract_method(
                    "test_interleaved_tool_calls_keep_one_item_each_with_parity"
                ),
                _contract_method(
                    "test_summary_coalescing_stops_at_part_identity"
                ),
            )
        )
    )


def test_phase7_abnormal_terminal_acceptance() -> None:
    run(
        _run_acceptance_methods(
            tuple(
                _contract_method(method_name)
                for method_name in (
                    "test_failure_and_cancellation_close_summary_incomplete",
                    "test_failure_and_cancellation_close_empty_observed_prefix",
                    "test_response_incomplete_projects_exact_safe_error",
                    "test_response_cancelled_is_only_canonical_cancellation",
                )
            )
        )
    )


def test_phase7_source_lifecycle_acceptance() -> None:
    run(
        _run_acceptance_methods(
            tuple(
                _contract_method(method_name)
                for method_name in (
                    "test_source_terminal_then_reraises_content_free_once",
                    "test_source_after_terminal_wins_over_cleanup_error",
                    "test_pre_terminal_source_error_wins_over_cleanup_error",
                    "test_missing_terminal_and_validation_error_have_no_terminal",
                    "test_local_body_aclose_has_no_late_write_and_cleans_source",
                    "test_cleanup_error_after_terminal_is_not_second_terminal",
                )
            )
        )
    )


def test_phase7_retention_admission_acceptance() -> None:
    run(
        _run_acceptance_methods(
            tuple(
                _contract_method(method_name)
                for method_name in (
                    "test_retention_rejects_1025th_outward_empty_part",
                    (
                        "test_retention_rejects_oversized_first_delta_after_"
                        "part_added"
                    ),
                    "test_retention_reserves_marker_before_redactor_mutation",
                    "test_retention_bounds_pending_host_path_tail",
                    "test_streaming_retention_failure_wins_over_cleanup_error",
                )
            )
        )
    )


def test_phase7_redaction_identity_acceptance() -> None:
    run(
        _run_acceptance_methods(
            tuple(
                _contract_method(method_name)
                for method_name in (
                    "test_marker_latch_suppresses_later_identities_and_resets",
                    "test_pending_marker_resolves_at_every_identity_boundary",
                    (
                        "test_no_marker_identity_loss_quarantines_once_then_"
                        "recovers"
                    ),
                )
            )
        )
    )


def test_phase7_non_stream_structure_acceptance() -> None:
    run(
        _run_acceptance_methods(
            tuple(
                _contract_method(method_name)
                for method_name in (
                    "test_non_stream_preserves_ordered_item_representations",
                    "test_non_stream_rebases_sparse_summary_emission_order",
                    (
                        "test_non_stream_repeated_provider_ids_stay_response_"
                        "unique"
                    ),
                )
            )
        )
    )


def test_phase7_non_stream_outcome_acceptance() -> None:
    run(
        _run_acceptance_methods(
            tuple(
                _contract_method(method_name)
                for method_name in (
                    "test_non_stream_abnormal_status_and_sync_ownership",
                    "test_non_stream_source_error_raises_and_never_syncs",
                    (
                        "test_non_stream_late_source_error_is_content_free_"
                        "and_no_sync"
                    ),
                    (
                        "test_non_stream_retention_is_exact_all_or_nothing_"
                        "http_500"
                    ),
                    (
                        "test_non_stream_retention_masks_cleanup_with_locked_"
                        "error"
                    ),
                    "test_non_stream_summary_isolated_from_json_answer",
                )
            )
        )
    )


def test_phase7_stream_non_stream_parity_acceptance() -> None:
    run(
        _run_acceptance_methods(
            (
                _contract_method(
                    "test_fixed_id_streaming_and_non_stream_output_parity"
                ),
            )
        )
    )


def test_phase7_cross_protocol_golden_acceptance() -> None:
    run(
        _run_acceptance_methods(
            (
                (
                    "server/primary_consumer_projection_golden_test.py",
                    "PrimaryConsumerProjectionGoldenTestCase",
                    "test_primary_consumers_project_same_canonical_trace",
                ),
            )
        )
    )


def test_phase7_acceptance_manifest_pins_exact_catalog() -> None:
    manifest = cast(
        Any,
        _ACCEPTANCE_SCRIPT.load_manifest(
            _ACCEPTANCE_SCRIPT.default_manifest_path()
        ),
    )

    assert manifest.active_phase == 7
    assert manifest.dimensions[_PHASE7_DIMENSION] == _PHASE7_NODE_IDS
    assert len(_PHASE7_NODE_IDS) == 12
    assert len(manifest.node_ids) == 355
    assert len(manifest.node_ids) == len(set(manifest.node_ids))
