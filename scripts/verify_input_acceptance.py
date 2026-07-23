#!/usr/bin/env python
"""Collect and execute the exact structured-input acceptance inventory."""

from argparse import ArgumentParser, Namespace
from ast import (
    AST,
    AnnAssign,
    Assert,
    Assign,
    AsyncFor,
    AsyncFunctionDef,
    AsyncWith,
    Attribute,
    AugAssign,
    BinOp,
    BoolOp,
    Break,
    Call,
    ClassDef,
    Compare,
    Constant,
    Continue,
    Delete,
    DictComp,
    Eq,
    Expr,
    For,
    FunctionDef,
    GeneratorExp,
    If,
    IfExp,
    Import,
    ImportFrom,
    Is,
    Lambda,
    ListComp,
    Load,
    Match,
    MatchAs,
    MatchMapping,
    MatchStar,
    Module,
    Name,
    NamedExpr,
    Pass,
    Raise,
    Return,
    SetComp,
    Starred,
    Subscript,
    Try,
    TryStar,
    While,
    With,
    dump,
    iter_child_nodes,
    parse,
    walk,
)
from ast import (
    Dict as AstDict,
)
from ast import (
    List as AstList,
)
from ast import (
    Set as AstSet,
)
from ast import (
    Tuple as AstTuple,
)
from ast import (
    expr as AstExpression,
)
from ast import (
    stmt as AstStatement,
)
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from json import dumps
from os import environ
from pathlib import Path, PurePosixPath
from re import IGNORECASE
from re import compile as compile_regex
from stat import S_IXGRP, S_IXOTH, S_IXUSR
from subprocess import CompletedProcess, TimeoutExpired, run
from sys import executable, stderr
from typing import Protocol, cast

from coverage import Coverage
from input_contract_json import (
    StrictJsonError,
    strict_json_loads,
    strict_json_path,
)
from verify_src_coverage import (
    CoverageVerificationError,
    verify_src_coverage,
)

_FEATURE = "structured_task_input"
_MIN_PHASE = 0
_MAX_PHASE = 12
_CATEGORIES = frozenset(
    (
        "unit",
        "integration",
        "negative",
        "race",
        "security",
        "public_e2e",
    )
)
_DISALLOWED_MARKERS = frozenset(("skip", "skipif", "xfail"))
_PROHIBITED_TEST_CONTROLS = frozenset(
    (
        "skip",
        "skipIf",
        "skipUnless",
        "importorskip",
        "xfail",
    )
)
_PROHIBITED_EXECUTION_NAMES = frozenset(("exec", "compile"))
_PROHIBITED_TEST_SYMBOLS = (
    _PROHIBITED_TEST_CONTROLS | _PROHIBITED_EXECUTION_NAMES
)
_ALIAS_CONTEXTLIB = "contextlib_module"
_ALIAS_BUILTINS = "builtins_module"
_ALIAS_PYTEST = "pytest_module"
_ALIAS_TEMPFILE = "tempfile_module"
_ALIAS_UNITTEST = "unittest_module"
_ALIAS_UNITTEST_MOCK = "unittest_mock_module"
_ALIAS_UNITTEST_PATCH = "unittest_mock_patch"
_ALIAS_UNITTEST_TEST_CASE = "unittest_test_case"
_ALIAS_SUPPRESS = "contextlib_suppress"
_ALIAS_NULLCONTEXT = "contextlib_nullcontext"
_ALIAS_CHECK_CONTEXT = "recognized_check_context"
_ALIAS_SAFE_CONTEXT = "recognized_safe_context"
_ALIAS_SAFE_CONTEXT_INSTANCE = "recognized_safe_context_instance"
_ALIAS_EXCEPTIONS = "exception_names"
_ALIAS_UNKNOWN = "unknown"
_ALIAS_END_POSITION = (1_000_000_000, 1_000_000_000)
_CHECK_FAILURE_EXCEPTION_NAMES = frozenset(
    (
        "AssertionError",
        "BaseException",
        "Exception",
    )
)
_KNOWN_BUILTIN_EXCEPTION_NAMES = frozenset(
    (
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "BaseExceptionGroup",
        "BlockingIOError",
        "BrokenPipeError",
        "BufferError",
        "BytesWarning",
        "ChildProcessError",
        "ConnectionAbortedError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "DeprecationWarning",
        "EOFError",
        "EncodingWarning",
        "EnvironmentError",
        "Exception",
        "ExceptionGroup",
        "FileExistsError",
        "FileNotFoundError",
        "FloatingPointError",
        "FutureWarning",
        "GeneratorExit",
        "IOError",
        "ImportError",
        "ImportWarning",
        "IndentationError",
        "IndexError",
        "InterruptedError",
        "IsADirectoryError",
        "KeyError",
        "KeyboardInterrupt",
        "LookupError",
        "MemoryError",
        "ModuleNotFoundError",
        "NameError",
        "NotADirectoryError",
        "NotImplementedError",
        "OSError",
        "OverflowError",
        "PendingDeprecationWarning",
        "PermissionError",
        "ProcessLookupError",
        "RecursionError",
        "ReferenceError",
        "ResourceWarning",
        "RuntimeError",
        "RuntimeWarning",
        "StopAsyncIteration",
        "StopIteration",
        "SyntaxError",
        "SyntaxWarning",
        "SystemError",
        "SystemExit",
        "TabError",
        "TimeoutError",
        "TypeError",
        "UnboundLocalError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "UnicodeError",
        "UnicodeTranslateError",
        "UnicodeWarning",
        "UserWarning",
        "ValueError",
        "Warning",
        "ZeroDivisionError",
    )
)
_PYTEST_CHECK_CONTEXT_NAMES = frozenset(("raises", "warns"))
_TEMPFILE_SAFE_CONTEXT_NAMES = frozenset(
    (
        "NamedTemporaryFile",
        "TemporaryDirectory",
        "TemporaryFile",
    )
)
_UNITTEST_CHECK_CONTEXT_NAMES = frozenset(
    (
        "assertRaises",
        "assertRaisesRegex",
        "assertWarns",
        "assertWarnsRegex",
    )
)
_UNITTEST_SAFE_CONTEXT_NAMES = frozenset(("subTest",))
_UNITTEST_TEST_CASE_NAMES = frozenset(("IsolatedAsyncioTestCase", "TestCase"))
_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_PUBLIC_SCHEMA_MUTATIONS = (
    "missing_required_field",
    "extra_field",
    "wrong_const",
    "wrong_type",
    "cross_field_invariant",
)
_EXPECTED_PROTOCOL_SCHEMA_SHA256 = {
    "a2a_message_metadata": (
        "f2226918b7d610aeabd987434cd7e186902486add3fb4e4ad2abebe6ca9fbeb3"
    ),
    "mcp_params_task": (
        "ecd4f03f32ca078f6e47650385ff7abe968f5f1bbaf4ba2c32b41e52fd08c834"
    ),
    "mcp_task": (
        "f1fb83bcc7c798c59985258b55df13883efa1f314f1d329ede4995dc657efa0e"
    ),
    "mcp_create_task_result": (
        "185f091128f71fc8fd3ad202b2625a0b2bf8a70b2d6bbdc8c7ea2304d7041f86"
    ),
}
_PUBLIC_CROSS_FIELD_INVARIANTS = {
    "a2a.task_working.v1": frozenset(
        {
            "a2a_resolution_task_id",
            "a2a_resolution_context_id",
            "a2a_resolution_request_id",
        }
    ),
    "a2a.task_input_required.v1": frozenset(
        {
            "a2a_task_message_task_id",
            "a2a_task_message_context_id",
            "a2a_task_message_request_id",
        }
    ),
    "mcp.task_cancelled.v1": frozenset(
        {
            "mcp_related_task_id",
            "mcp_canonical_request_id",
        }
    ),
    "mcp.task_input_required.v1": frozenset(
        {
            "mcp_related_task_id",
            "mcp_canonical_request_id",
        }
    ),
    "mcp.task_working.v1": frozenset(
        {
            "mcp_related_task_id",
            "mcp_canonical_request_id",
        }
    ),
}
_COLLECT_SENTINEL = "__INPUT_ACCEPTANCE_COLLECT__"
_EXECUTE_SENTINEL = "__INPUT_ACCEPTANCE_EXECUTE__"
_PROCESS_TIMEOUT_SECONDS = 300
_EXPECTED_CURRENT_RUNTIME_FILES = (
    "tests/agent/execution_attached_boundaries_test.py",
    "tests/agent/execution_cancellation_integration_test.py",
    "tests/agent/execution_direct_iteration_cancellation_test.py",
    "tests/agent/execution_isolation_integration_test.py",
    "tests/agent/execution_memory_idempotency_test.py",
    "tests/agent/execution_message_exactness_test.py",
    "tests/agent/execution_response_ownership_adversarial_test.py",
    "tests/agent/execution_sequential_response_sync_test.py",
    "tests/agent/execution_strict_invariants_test.py",
    "tests/agent/execution_suspension_adversarial_test.py",
    "tests/agent/execution_test.py",
    "tests/agent/execution_transcript_adversarial_test.py",
    "tests/agent/execution_wrapper_input_required_test.py",
    "tests/agent/json_orchestrator_test.py",
    "tests/agent/orchestrator_cleanup_ownership_test.py",
    "tests/agent/orchestrator_convergence_coverage_test.py",
    "tests/agent/orchestrator_response_convergence_coverage_test.py",
    "tests/agent/orchestrator_test.py",
    "tests/agent/renderer_test.py",
    "tests/input/attached_runtime_e2e_test.py",
    "tests/input/attached_runtime_matrix_test.py",
    "tests/input/broker_contract_test.py",
    "tests/memory/permanent/elasticsearch_message_memory_test.py",
    "tests/memory/permanent/pgsql_test.py",
    "tests/memory/permanent/s3vectors_message_memory_test.py",
    "tests/memory/permanent/structured_message_codec_test.py",
    "tests/model/engine_test.py",
    "tests/model/model_stream_interaction_test.py",
)
_EXPECTED_CURRENT_RUNTIME_NODE_COUNT = 249
_EXPECTED_CURRENT_RUNTIME_NODE_SHA256 = (
    "41ec33710c3a014309798077e3ef75484d315e0781309aa4f62756253c5e0be4"
)
_EXPECTED_CURRENT_FOCUSED_COMMAND = (
    "poetry run python scripts/verify_input_acceptance.py --through-phase 4"
    " --runtime-only"
)
_COVERAGE_EXCLUSION_PATTERN = compile_regex(
    r"#\s*(?:pragma\s*:?\s*no\s*cover|coverage\s*:?\s*ignore)",
    IGNORECASE,
)
_TRANSITION_PATTERN = compile_regex(r"([a-z][a-z0-9_]*)->([a-z][a-z0-9_]*)")
_PUBLIC_RESULT_PATTERN = compile_regex(r"envelope=([a-z][a-z0-9._-]*)")
_STATUS_OR_EXIT_PATTERN = compile_regex(
    r"([a-z][a-z0-9_]*)=(-?[A-Za-z0-9][A-Za-z0-9._-]*)"
)
_INTERACTION_STATES = frozenset(
    {
        "created",
        "pending",
        "running",
        "answered",
        "declined",
        "cancelled",
        "timed_out",
        "unavailable",
        "expired",
        "superseded",
    }
)
_STATUS_OR_EXIT_KEYS = frozenset(
    {
        "exit",
        "interaction_state",
        "result",
        "exception",
        "http",
        "jsonrpc_error",
        "task_status",
        "task_state",
        "flow_state",
        "branch_state",
        "capability",
        "client_result",
    }
)
_BEHAVIOR_REQUIREMENT_IDS = frozenset(
    tuple(f"INPUT-N-{index:03d}" for index in range(1, 108))
    + tuple(f"INPUT-26.{index}" for index in range(1, 13))
)
_GATE_REQUIREMENT_IDS = frozenset(
    f"INPUT-GATE-{index:03d}" for index in range(1, 13)
)
_EXPECTED_REQUIREMENT_IDS = _BEHAVIOR_REQUIREMENT_IDS | _GATE_REQUIREMENT_IDS
_EXPECTED_FAILURE_CONDITIONS = frozenset(
    f"INPUT-F-{index:02d}" for index in range(1, 16)
)
_EXPECTED_NO_BC_IDS = frozenset(
    {
        "tool-manager-provider-coupling",
        "completion-only-results",
        "engine-agent-reusable-mutable-fields",
        "orchestrator-reusable-mutable-fields",
        "orchestrator-response-continuation-state",
        "chat-sse-fake-turn",
        "responses-sse-fake-turn",
        "responses-non-stream-fake-turn",
        "mcp-fake-turn",
        "flow-run-equals-turn",
        "a2a-task-context-correlation",
        "model-single-stream-correlation",
        "model-non-stream-correlation",
        "legacy-chat-sse-correlation-assertion",
        "legacy-responses-sse-correlation-assertion",
        "legacy-mcp-correlation-assertion",
    }
)
_EXPECTED_REQUIREMENTS_SHA256 = (
    "f333a13de1678a7d139acc384d673df15f861c66cfbbffbcd77b8dcc2dc79c80"
)
_EXPECTED_FAILURE_MATRIX_SHA256 = (
    "e5ce3aac0d441897b80a09d6a693853c65d4a446ed7e4c0184b3e3bc0b212c08"
)
_EXPECTED_DECISIONS_SHA256 = (
    "c13bcff64c0b28905c64c8e92b040d56e2312a99b45303b4e3a5d4d4490c882d"
)
_EXPECTED_NO_BC_SHA256 = (
    "c75145467fe15a1cd55b6bb10e7dd16fc5ff8e4b25b530c2d7f147ab3c641887"
)
_EXPECTED_ACCEPTANCE_LEDGER_SHA256 = (
    "51c73d84e7292cff0dbc0698708ae0a8d162309c41d77b79d5880b72decbc2a2"
)
_EXPECTED_EVIDENCE_SHA256 = (
    "e3546c8702c933b8861db39a72e499f7d5bec80523eb9650c3f2bb7a52c0ecba"
)
_EXPECTED_REVIEW_HISTORY_SHA256 = (
    "ce375482081e5180ba4904bf4d7517af8069cd21f82a1e7771c5077c7ba0cfdd"
)
_EXPECTED_PHASE0_REVIEW_SHA256 = (
    "573625598e6f7501e5d3cbc158be7b630427143e1cdd7658814a52b6374d8f6b"
)
_EXPECTED_PHASE1_REVIEW_SHA256 = (
    "42ee51f1041cc975bcdd750247d3e61a08fe453f1f332d76f9dd47e18b8e4a85"
)
_EXPECTED_PHASE2_REVIEW_SHA256 = (
    "7c94eb4806501ecb3ae82f1447fd94ed95e31d185d41e9fbcba2f31ce448a408"
)
_EXPECTED_PRIOR_REVIEW_SHA256 = (
    "f59a5cb66ee765407b15134bfe8e2a2c19600b807dcca550a89ef68b2caaee1c"
)
_EXPECTED_PHASE2_PENDING_REVIEW_SHA256 = (
    "a83a4e9545ac72c99c23d6fd316c7661f5a6bfef86c8c39a5c209ee6185a852a"
)
_EXPECTED_PHASE1_QUALITY_SHA256 = (
    "f58bd16d9bf57bb3f2972982ff8bcf19a6125715a40194effecb8141c8ebd5ea"
)
_EXPECTED_PHASE1_EVIDENCE_SHA256 = (
    "a4c16a90cf2d451b423da22ba763b50742e47f583230ded87c9997d77e1b93b8"
)
_EXPECTED_PHASE2_QUALITY_SHA256 = (
    "d004e9f765e9167d31debb7642883e774e42a03503f32f8869eb6b4e084e3953"
)
_EXPECTED_PHASE2_EVIDENCE_SHA256 = (
    "d0e276493609d2e7254c576bf50552a933e4e54cb67c9ec6e6a71f94a17f0302"
)
_EXPECTED_PRIOR_QUALITY_SHA256 = (
    "62c94da810be0c995525580c19df034b35aee2700f6e9e8fa51c69ba778e0102"
)
_EXPECTED_PRIOR_EVIDENCE_SHA256 = (
    "59788e2441bec0bd34a61ff94f8b14459ca229a37fcf693ae6b94fb8106e8ab9"
)
_EXPECTED_QUALITY_HISTORY_SHA256 = (
    "c1798ede412d1b56848dede3f7d242b1067ced5a871e60bbb9a4f9098df5b875"
)
_EXPECTED_IMPLEMENTATION_OWNER = "/root"
_EXPECTED_INDEPENDENT_REVIEWER = "/root/input_contract_audit"
_EXPECTED_REVIEW_OCCURRENCES = (
    (0, "baseline", "/root/input_contract_audit", "approved"),
    (1, "semantic", "/root/interaction_round4_semantic", "pending"),
    (1, "gate", "/root/interaction_round4_gates", "pending"),
    (1, "semantic", "/root/interaction_round4_semantic", "approved"),
    (1, "gate", "/root/interaction_round4_gates", "approved"),
    (2, "semantic", "/root/broker_review", "pending"),
    (2, "gate", "/root/phase2_acceptance_review", "pending"),
    (2, "semantic", "/root/phase2_acceptance_review", "approved"),
    (2, "gate", "/root/phase2_metadata_review", "approved"),
    (3, "gate", "/root/terminal_review", "pending"),
    (3, "semantic", "/root/acceptance_review", "approved"),
    (3, "closure", "/root/phase3_closure_audit", "pending"),
    (3, "closure", "/root/phase3_closure_audit", "approved"),
    (
        3,
        "coverage-closure",
        "/root/phase3_closure_audit/turn3_toolmanager_readonly",
        "pending",
    ),
    (
        3,
        "coverage-closure",
        "/root/phase3_closure_audit/turn3_toolmanager_readonly",
        "approved",
    ),
    (3, "gate", "/root/terminal_review", "approved"),
    (4, "semantic", "/root/execution_runtime_review", "approved"),
    (4, "gate", "/root/execution_gate_review", "approved"),
)
_EXPECTED_CURRENT_SEMANTIC_REVIEW_STATUS = "approved"
_EXPECTED_CURRENT_GATE_REVIEW_STATUS = "approved"
_EXPECTED_BASELINE_HEAD = "609aa091c17756ab952cf5fe668ca3d867f0e311"
_EXPECTED_BASELINE_SUBJECT = "Bump version to v1.5.8 (#1067)"
_EXPECTED_CURRENT_BASELINE_HEAD = "d538fba47d9721755675fa8752403203e08fe025"
_EXPECTED_CURRENT_REGRESSION_NODE_COUNT = 26
_EXPECTED_CURRENT_SUPPORT_SURFACE_COUNT = 43
_EXPECTED_CURRENT_TEST_FILE_COUNT = 51
_EXPECTED_CURRENT_UNCHANGED_SUPPORT_SURFACE_COUNT = 8
_ABSENT_TEST_DEFINITION_SHA256 = (
    "d6f5bc657cdeb0be6ee6c3f042458c9981e5bcb0a4dbe6a9f6d6c39f464f0479"
)
_ABSENT_TEST_SUPPORT_SHA256 = (
    "6b21bdd337b5554cd17fe6cf861b9b1f457568a5a2e05a41e7ee744686ed0872"
)
_EXPECTED_CURRENT_CHANGED_SUPPORT_PATHS = frozenset(
    (
        "tests/agent/additional_coverage_test.py",
        "tests/agent/default_orchestrator_test.py",
        "tests/agent/execution_attached_boundaries_test.py",
        "tests/agent/execution_cancellation_integration_test.py",
        "tests/agent/execution_coverage_regression_test.py",
        "tests/agent/execution_direct_iteration_cancellation_test.py",
        "tests/agent/execution_isolation_integration_test.py",
        "tests/agent/execution_memory_idempotency_test.py",
        "tests/agent/execution_message_exactness_test.py",
        "tests/agent/execution_response_ownership_adversarial_test.py",
        "tests/agent/execution_sequential_response_sync_test.py",
        "tests/agent/execution_strict_invariants_test.py",
        "tests/agent/execution_suspension_adversarial_test.py",
        "tests/agent/execution_test.py",
        "tests/agent/execution_transcript_adversarial_test.py",
        "tests/agent/execution_wrapper_input_required_test.py",
        "tests/agent/json_orchestrator_test.py",
        "tests/agent/loader_test.py",
        "tests/agent/orchestrator_cleanup_gap_coverage_test.py",
        "tests/agent/orchestrator_cleanup_ownership_test.py",
        "tests/agent/orchestrator_convergence_coverage_test.py",
        "tests/agent/orchestrator_response_cleanup_coverage_test.py",
        "tests/agent/orchestrator_response_convergence_coverage_test.py",
        "tests/agent/orchestrator_response_test.py",
        "tests/agent/orchestrator_test.py",
        "tests/input/attached_runtime_e2e_test.py",
        "tests/input/attached_runtime_matrix_test.py",
        "tests/input/broker_contract_test.py",
        "tests/input_acceptance_verifier_test.py",
        "tests/input_contract_test.py",
        "tests/memory/permanent/elasticsearch_message_memory_test.py",
        "tests/memory/permanent/pgsql_test.py",
        "tests/memory/permanent/s3vectors_message_memory_test.py",
        "tests/memory/permanent/structured_message_codec_test.py",
        "tests/model/model_stream_interaction_test.py",
        "tests/model/text_generation_response_additional_test.py",
        "tests/project_metadata_test.py",
        "tests/server/protocol_streaming_e2e_test.py",
        "tests/server/reasoning_summary_protocol_test.py",
        "tests/server/responses_test.py",
        "tests/server/streaming_conformance_test.py",
        "tests/src_coverage_verifier_test.py",
        "tests/tool/a2a_tool_test.py",
    )
)
_CURRENT_DUPLICATE_PGSQL_NODE = (
    "tests/memory/permanent/pgsql_test.py::"
    "PgsqlMessageMemoryTestCase::test_search_messages"
)
_EXPECTED_CURRENT_DUPLICATE_TEST_DEFINITIONS = {
    _CURRENT_DUPLICATE_PGSQL_NODE: (
        2,
        "4631cf0ad47b3977417207f7d6afaf5194b9aed18bcd1d8a05f3cf691ac680c2",
        "4631cf0ad47b3977417207f7d6afaf5194b9aed18bcd1d8a05f3cf691ac680c2",
    ),
}
_EXPECTED_CURRENT_REGRESSION_SHA256 = (
    "e7e18a868cdda568f185b8620cf1460cb51916dd1c6bc14080398c63fd08c24a"
)
_EXPECTED_CURRENT_ACTIVE_LEGACY_NODE_COUNT = 10
_EXPECTED_CURRENT_ACTIVE_LEGACY_SHA256 = (
    "9d54e8c0522016cb357d2042553297d1e0022b23e0928ea1fafe5ba1e98b625c"
)
_EXPECTED_CURRENT_ACTIVE_LEGACY_GATE_NODES = frozenset(
    (
        (
            "tests/input_acceptance_verifier_test.py::"
            "test_evidence_state_and_review_history_fail_closed"
        ),
    )
)
_EXPECTED_CURRENT_SEMANTIC_REPLACEMENTS = (
    (
        (
            "tests/agent/orchestrator_test.py::OrchestratorCallTestCase::"
            "test_aexit_skips_message_sync_on_keyboard_interrupt"
        ),
        "811e85fdb87372d567e68dc4c10d81b1732fc520bb98b763aea4e279713ff12e",
        (
            "tests/agent/orchestrator_test.py::OrchestratorCallTestCase::"
            "test_aexit_runs_all_cleanup_on_keyboard_interrupt"
        ),
        "835f38e34bc443fbd1a5388c2f9299cbb9bf71565c49673f759f11ad16628d71",
    ),
    (
        (
            "tests/agent/orchestrator_test.py::OrchestratorCallTestCase::"
            "test_aexit_skips_message_sync_on_cancelled_error"
        ),
        "d2a0ff40f479e33ce7a75e1cbf5aa095a3f4308d41d05a7fad549fa0807abbc0",
        (
            "tests/agent/orchestrator_test.py::OrchestratorCallTestCase::"
            "test_aexit_runs_response_cleanup_on_cancelled_error"
        ),
        "fd4c2a19fd5c30d826e5ca4a392ede1bb23c2c3b0718614c4f70469cfdf3f4e7",
    ),
)
_EXPECTED_PENDING_SOURCE_INVENTORY = (
    "a803978249761cdf9b9f8ebf019ca4df9fa7e33d18b9281a424c104dca4c4563",
    426,
    111511,
    1356,
)
_EXPECTED_BOUNDARY_PATHS = frozenset(
    {
        ".github/workflows/test.yml",
        "Makefile",
        "scripts/input_contract_json.py",
        "scripts/run_input_contract_gate.py",
        "scripts/task_pgsql_test_database.py",
        "scripts/verify_input_acceptance.py",
        "scripts/verify_input_types.py",
        "scripts/verify_src_coverage.py",
        "src/avalan/agent/",
        "src/avalan/cli/",
        "src/avalan/entities.py",
        "src/avalan/event/__init__.py",
        "src/avalan/event/manager.py",
        "src/avalan/flow/registry.py",
        "src/avalan/interaction/",
        "src/avalan/memory/",
        "src/avalan/model/",
        "src/avalan/server/a2a/router.py",
        "src/avalan/server/routers/chat.py",
        "src/avalan/server/routers/mcp.py",
        "src/avalan/server/routers/responses.py",
        "src/avalan/task/event.py",
        "src/avalan/tool/",
        "tests/agent/",
        "tests/cli/",
        "tests/event/interaction_lifecycle_test.py",
        "tests/fixtures/input/",
        "tests/flow/",
        "tests/input/",
        "tests/input_acceptance_verifier_test.py",
        "tests/input_contract_fixtures.py",
        "tests/input_contract_harness_test.py",
        "tests/input_contract_metadata_test.py",
        "tests/input_contract_test.py",
        "tests/input_type_contract_test.py",
        "tests/input_type_contracts/",
        "tests/interaction/",
        "tests/interaction_type_contracts/",
        "tests/memory/",
        "tests/model/",
        "tests/project_metadata_test.py",
        "tests/reasoning_summary_phase1_test.py",
        "tests/server/",
        "tests/src_coverage_verifier_test.py",
        "tests/task/",
        "tests/tool/",
    }
)
_EXPECTED_PRODUCTION_SOURCE_PATHS = frozenset(
    {
        "src/avalan/agent/",
        "src/avalan/cli/",
        "src/avalan/entities.py",
        "src/avalan/event/__init__.py",
        "src/avalan/event/manager.py",
        "src/avalan/flow/registry.py",
        "src/avalan/interaction/",
        "src/avalan/memory/",
        "src/avalan/model/",
        "src/avalan/server/a2a/router.py",
        "src/avalan/server/routers/chat.py",
        "src/avalan/server/routers/mcp.py",
        "src/avalan/server/routers/responses.py",
        "src/avalan/task/event.py",
        "src/avalan/tool/",
    }
)
_EXPECTED_ORDERED_COMMON_GATE_COMMANDS = (
    "poetry run pytest --verbose -s",
    "make test-coverage -- -100 src/",
    "make test-coverage-exact no-install",
    (
        "poetry run python scripts/verify_input_acceptance.py"
        + " --through-phase 4"
    ),
    "make typecheck-input-contract INPUT_PHASE=4",
    "make lint",
    "git diff --check",
)
_EXPECTED_COMMON_GATE_COMMANDS = frozenset(
    _EXPECTED_ORDERED_COMMON_GATE_COMMANDS
)

_COLLECT_DRIVER = f"""
from json import dumps
from sys import argv
from pytest import main

class Probe:
    def __init__(self):
        self.items = []
        self.deselected = []
        self.collection_reports = []

    def pytest_collection_finish(self, session):
        self.items = [
            {{
                "nodeid": item.nodeid,
                "markers": sorted(
                    marker.name for marker in item.iter_markers()
                ),
            }}
            for item in session.items
        ]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection_reports.append(
                {{
                    "nodeid": report.nodeid,
                    "outcome": report.outcome,
                    "detail": str(report.longrepr),
                }}
            )

probe = Probe()
exit_code = main(
    [
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "anyio.pytest_plugin",
        *argv[1:],
    ],
    plugins=[probe],
)
print("{_COLLECT_SENTINEL}" + dumps({{
    "exit_code": int(exit_code),
    "items": probe.items,
    "deselected": probe.deselected,
    "collection_reports": probe.collection_reports,
}}, sort_keys=True))
"""

_EXECUTE_DRIVER = f"""
from json import dumps
from sys import argv
from pytest import main

class Probe:
    def __init__(self):
        self.items = []
        self.deselected = []
        self.collection_reports = []
        self.reports = []

    def pytest_collection_finish(self, session):
        self.items = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection_reports.append({{
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "detail": str(report.longrepr),
            }})

    def pytest_runtest_logreport(self, report):
        self.reports.append({{
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "wasxfail": str(getattr(report, "wasxfail", "")),
            "detail": (
                str(report.longrepr)
                if report.failed or report.skipped
                else ""
            ),
        }})

probe = Probe()
exit_code = main(
    ["-q", "-p", "no:cacheprovider", "-p", "anyio.pytest_plugin", *argv[1:]],
    plugins=[probe],
)
print("{_EXECUTE_SENTINEL}" + dumps({{
    "exit_code": int(exit_code),
    "items": probe.items,
    "deselected": probe.deselected,
    "collection_reports": probe.collection_reports,
    "reports": probe.reports,
}}, sort_keys=True))
"""


class AcceptanceVerificationError(RuntimeError):
    """Report an invalid or non-passing acceptance inventory."""


class _JsonSchemaValidator(Protocol):
    """Describe the JSON Schema operation used by the verifier."""

    def is_valid(self, instance: object) -> bool: ...


class _JsonSchemaValidatorFactory(Protocol):
    """Describe the dynamically loaded Draft 2020-12 validator."""

    def __call__(self, schema: dict[str, object]) -> _JsonSchemaValidator: ...

    def check_schema(self, schema: dict[str, object]) -> None: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceNode:
    """Store one lifecycle-aware acceptance node."""

    id: str
    category: str
    lifecycle: str
    active_from_phase: int
    requirement_ids: tuple[str, ...]
    node_id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RequirementActivationSlice:
    """Store reviewed ownership for one partially active requirement."""

    requirement_id: str
    phase: int
    active_owner: str
    active_scope: str
    active_node_ids: tuple[str, ...]
    remaining_owner: str
    remaining_scope: str
    planned_node_ids: tuple[str, ...]
    reviewed_by: str
    evidence: str


@dataclass(frozen=True, kw_only=True, slots=True)
class ParameterExpansion:
    """Store the exact pytest instances owned by one parametrized node."""

    node_id: str
    instance_node_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceManifest:
    """Store the validated acceptance inventory."""

    path: Path
    current_phase: int
    nodes: tuple[AcceptanceNode, ...]
    requirement_activation_slices: tuple[RequirementActivationSlice, ...]
    parameter_expansions: tuple[ParameterExpansion, ...]

    def active_nodes(self, through_phase: int) -> tuple[AcceptanceNode, ...]:
        """Return active nodes introduced no later than the requested gate."""
        assert _MIN_PHASE <= through_phase <= self.current_phase
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle == "active"
            and node.active_from_phase <= through_phase
        )

    def active_pytest_instances(self, through_phase: int) -> tuple[str, ...]:
        """Return the exact pytest instances required by the selected gate."""
        expansions = {
            expansion.node_id: expansion.instance_node_ids
            for expansion in self.parameter_expansions
        }
        return tuple(
            instance
            for node in self.active_nodes(through_phase)
            for instance in expansions.get(node.node_id, (node.node_id,))
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _CheckPaths:
    """Store check state for every reachable control-flow outcome."""

    next_states: frozenset[bool] = frozenset()
    return_states: frozenset[bool] = frozenset()
    break_states: frozenset[bool] = frozenset()
    continue_states: frozenset[bool] = frozenset()
    failed_check_states: frozenset[bool] = frozenset()


@dataclass(frozen=True, kw_only=True, slots=True)
class _SuppressContextAliases:
    """Store lexical scopes used to resolve context-manager aliases."""

    scopes: tuple[AST, ...]
    class_scope_index: int | None = None
    instance_names: frozenset[str] = frozenset()


@dataclass(frozen=True, kw_only=True, slots=True)
class _AliasBinding:
    """Store one ordered lexical name binding."""

    name: str
    position: tuple[int, int]
    value: AST | None = None
    direct_kind: str | None = None
    direct_exception_names: frozenset[str] = frozenset()
    imported_attribute: str | None = None
    definite: bool = True


@dataclass(frozen=True, kw_only=True, slots=True)
class _ResolvedAlias:
    """Store one resolved module, callable, or exception alias."""

    kind: str
    exception_names: frozenset[str] = frozenset()


@dataclass(frozen=True, kw_only=True, slots=True)
class _AttributeMutation:
    """Store exact or dynamically unknown attribute mutation targets."""

    names: frozenset[str] | None
    owners: tuple[AST, ...] | None


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def fixture_root() -> Path:
    """Return the tracked input-contract fixture directory."""
    return repository_root() / "tests" / "fixtures" / "input"


def default_manifest_path() -> Path:
    """Return the tracked acceptance-manifest path."""
    return fixture_root() / "acceptance_manifest.json"


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
            "activation_history",
            "activation_snapshots",
            "requirement_activation_slices",
            "parameter_expansions",
            "replacements",
            "nodes",
        },
        "acceptance manifest",
    )
    _header(payload, "acceptance manifest")
    current_phase = _phase(payload.get("current_phase"), "current_phase")
    categories = _string_list(payload.get("categories"), "categories")
    if frozenset(categories) != _CATEGORIES or len(categories) != len(
        _CATEGORIES
    ):
        raise AcceptanceVerificationError(
            "acceptance categories must contain the exact required inventory"
        )
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise AcceptanceVerificationError(
            "acceptance nodes must be a non-empty list"
        )
    nodes = tuple(_acceptance_node(item, current_phase) for item in raw_nodes)
    _unique((node.id for node in nodes), "acceptance node ID")
    _unique((node.node_id for node in nodes), "pytest node ID")
    if frozenset(node.category for node in nodes) != _CATEGORIES:
        raise AcceptanceVerificationError(
            "every acceptance category must own at least one node"
        )
    requirement_activation_slices = _requirement_activation_slices(
        payload.get("requirement_activation_slices"),
        nodes,
        current_phase,
    )
    parameter_expansions = _parameter_expansions(
        payload.get("parameter_expansions"),
        nodes,
    )
    _activation_history(
        payload.get("activation_history"),
        nodes,
        current_phase,
    )
    _activation_snapshots(
        payload.get("activation_snapshots"),
        payload.get("replacements"),
        payload.get("requirement_activation_slices"),
        payload.get("parameter_expansions"),
        nodes,
        current_phase,
    )
    if current_phase >= 4:
        _validate_current_manifest_inventory(nodes)
    return AcceptanceManifest(
        path=path,
        current_phase=current_phase,
        nodes=nodes,
        requirement_activation_slices=requirement_activation_slices,
        parameter_expansions=parameter_expansions,
    )


def _current_runtime_node_ids(
    nodes: tuple[AcceptanceNode, ...],
) -> tuple[str, ...]:
    """Return exact current behavioral nodes from the inventory."""
    return tuple(
        node.node_id
        for node in nodes
        if node.active_from_phase == 4
        and any(
            requirement_id in _BEHAVIOR_REQUIREMENT_IDS
            for requirement_id in node.requirement_ids
        )
    )


def _validate_current_manifest_inventory(
    nodes: tuple[AcceptanceNode, ...],
) -> None:
    """Require the reviewed current behavioral files and node digest."""
    runtime_files = frozenset(_EXPECTED_CURRENT_RUNTIME_FILES)
    runtime_nodes = tuple(
        node
        for node in nodes
        if node.active_from_phase == 4
        and any(
            requirement_id in _BEHAVIOR_REQUIREMENT_IDS
            for requirement_id in node.requirement_ids
        )
    )
    invalid = tuple(
        node.node_id
        for node in runtime_nodes
        if node.lifecycle != "active"
        or node.category == "public_e2e"
        or any(
            requirement_id in _GATE_REQUIREMENT_IDS
            for requirement_id in node.requirement_ids
        )
    )
    observed_files = frozenset(
        node.node_id.split("::", 1)[0] for node in runtime_nodes
    )
    node_ids = tuple(node.node_id for node in runtime_nodes)
    digest = sha256("\n".join(sorted(node_ids)).encode("utf-8")).hexdigest()
    if (
        invalid
        or observed_files != runtime_files
        or len(node_ids) != _EXPECTED_CURRENT_RUNTIME_NODE_COUNT
        or digest != _EXPECTED_CURRENT_RUNTIME_NODE_SHA256
    ):
        raise AcceptanceVerificationError(
            "current runtime acceptance inventory changed:"
            f" invalid={list(invalid)},"
            f" missing_files={sorted(runtime_files - observed_files)},"
            f" unexpected_files={sorted(observed_files - runtime_files)},"
            f" nodes={len(node_ids)}, digest={digest}"
        )


def _validate_current_runtime_collection(
    manifest: AcceptanceManifest,
    root: Path,
) -> None:
    """Collect every reviewed current node and reject node or file drift."""
    missing_files = tuple(
        relative
        for relative in _EXPECTED_CURRENT_RUNTIME_FILES
        if not (root / relative).is_file()
    )
    if missing_files:
        raise AcceptanceVerificationError(
            "current runtime test-file inventory changed:"
            f" missing={list(missing_files)}"
        )
    expected = _current_runtime_node_ids(manifest.nodes)
    collection = _run_probe(
        _COLLECT_DRIVER,
        _COLLECT_SENTINEL,
        expected,
        root,
    )
    _verify_collection(expected, collection)


def _current_runtime_instance_ids(
    manifest: AcceptanceManifest,
) -> tuple[str, ...]:
    """Return exact collected instances for current behavioral nodes."""
    expansions = {
        expansion.node_id: expansion.instance_node_ids
        for expansion in manifest.parameter_expansions
    }
    return tuple(
        instance
        for node_id in _current_runtime_node_ids(manifest.nodes)
        for instance in expansions.get(node_id, (node_id,))
    )


def verify_current_runtime(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> AcceptanceManifest:
    """Validate and execute only the exact current behavioral inventory."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    if manifest.current_phase < 4:
        raise AcceptanceVerificationError(
            "current runtime verification requires an implemented current"
            " manifest"
        )
    _validate_current_runtime_collection(manifest, root)
    node_ids = _current_runtime_node_ids(manifest.nodes)
    instance_node_ids = _current_runtime_instance_ids(manifest)
    _validate_execution_scope(path, node_ids, root)
    for node_id in node_ids:
        _validate_test_implementation(node_id, root)
    collection = _run_probe(
        _COLLECT_DRIVER,
        _COLLECT_SENTINEL,
        node_ids,
        root,
    )
    collected_node_ids = _verify_collection(instance_node_ids, collection)
    execution = _run_probe(
        _EXECUTE_DRIVER,
        _EXECUTE_SENTINEL,
        node_ids,
        root,
    )
    _verify_execution(instance_node_ids, execution, collected_node_ids)
    return manifest


def verify_acceptance(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
    contract_fixture_root: Path | None = None,
) -> AcceptanceManifest:
    """Validate fixtures and require exact passing node execution."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    if through_phase < _MIN_PHASE or through_phase > manifest.current_phase:
        raise AcceptanceVerificationError(
            "through-phase must be implemented by the current manifest: "
            f"requested={through_phase}, current={manifest.current_phase}"
        )
    fixtures = contract_fixture_root or path.parent
    if through_phase >= 4:
        _validate_current_runtime_collection(manifest, root)
    _validate_contract_fixtures(manifest, fixtures, root)
    active = manifest.active_nodes(through_phase)
    if not active:
        raise AcceptanceVerificationError(
            "the selected acceptance inventory has no active nodes"
        )
    node_ids = tuple(node.node_id for node in active)
    instance_node_ids = manifest.active_pytest_instances(through_phase)
    _validate_execution_scope(path, node_ids, root)
    for node_id in node_ids:
        _validate_test_implementation(node_id, root)
    collection = _run_probe(
        _COLLECT_DRIVER,
        _COLLECT_SENTINEL,
        node_ids,
        root,
    )
    collected_node_ids = _verify_collection(instance_node_ids, collection)
    execution = _run_probe(
        _EXECUTE_DRIVER,
        _EXECUTE_SENTINEL,
        node_ids,
        root,
    )
    _verify_execution(instance_node_ids, execution, collected_node_ids)
    return manifest


def _strict_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = strict_json_path(path)
    except StrictJsonError as exc:
        raise AcceptanceVerificationError(
            f"cannot read {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AcceptanceVerificationError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _header(payload: dict[str, object], label: str) -> None:
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise AcceptanceVerificationError(
            f"{label} schema_version must be the integer 1"
        )
    if payload.get("feature") != _FEATURE:
        raise AcceptanceVerificationError(
            f"{label} feature must be {_FEATURE}"
        )


def _acceptance_node(raw: object, current_phase: int) -> AcceptanceNode:
    if not isinstance(raw, dict):
        raise AcceptanceVerificationError("acceptance node must be an object")
    item = cast(dict[str, object], raw)
    _exact_keys(
        item,
        {
            "id",
            "category",
            "lifecycle",
            "active_from_phase",
            "requirement_ids",
            "node_id",
        },
        "acceptance node",
    )
    identifier = _nonempty_string(item.get("id"), "acceptance node id")
    category = _nonempty_string(item.get("category"), "acceptance category")
    if category not in _CATEGORIES:
        raise AcceptanceVerificationError(
            f"unknown acceptance category: {category}"
        )
    lifecycle = _nonempty_string(item.get("lifecycle"), "node lifecycle")
    if lifecycle not in {"planned", "active"}:
        raise AcceptanceVerificationError(
            f"invalid acceptance lifecycle: {lifecycle}"
        )
    active_from_phase = _phase(
        item.get("active_from_phase"), "active_from_phase"
    )
    expected_lifecycle = (
        "active" if active_from_phase <= current_phase else "planned"
    )
    if lifecycle != expected_lifecycle:
        raise AcceptanceVerificationError(
            f"acceptance lifecycle regression for {identifier}: "
            f"expected {expected_lifecycle}, observed {lifecycle}"
        )
    requirement_ids = _string_list(
        item.get("requirement_ids"), "requirement_ids"
    )
    _unique(requirement_ids, f"requirement ID on {identifier}")
    node_id = _node_id(item.get("node_id"))
    return AcceptanceNode(
        id=identifier,
        category=category,
        lifecycle=lifecycle,
        active_from_phase=active_from_phase,
        requirement_ids=requirement_ids,
        node_id=node_id,
    )


def _parameter_expansions(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
) -> tuple[ParameterExpansion, ...]:
    if not isinstance(raw, list):
        raise AcceptanceVerificationError(
            "parameter expansions must be a list"
        )
    by_node_id = {node.node_id: node for node in nodes}
    manifest_node_ids = frozenset(by_node_id)
    expansions: list[ParameterExpansion] = []
    all_instances: list[str] = []
    for value in raw:
        if not isinstance(value, dict):
            raise AcceptanceVerificationError(
                "parameter expansion must be an object"
            )
        item = cast(dict[str, object], value)
        _exact_keys(
            item,
            {"node_id", "instance_node_ids", "sha256"},
            "parameter expansion",
        )
        node_id = _node_id(item.get("node_id"))
        node = by_node_id.get(node_id)
        if node is None:
            raise AcceptanceVerificationError(
                "parameter expansion must own one exact manifest node:"
                f" {node_id}"
            )
        if "[" in node_id.rsplit("::", 1)[-1]:
            raise AcceptanceVerificationError(
                "explicit parameter instance must remain exact-only:"
                f" {node_id}"
            )
        instances = _string_list(
            item.get("instance_node_ids"),
            "parameter instance node IDs",
        )
        _unique(instances, f"parameter instance for {node_id}")
        for instance in instances:
            _node_id(instance)
            if not instance.startswith(f"{node_id}[") or not instance.endswith(
                "]"
            ):
                raise AcceptanceVerificationError(
                    "parameter instance does not belong to its base node:"
                    f" {instance}"
                )
            if instance in manifest_node_ids:
                raise AcceptanceVerificationError(
                    "parameter instance duplicates a manifest node:"
                    f" {instance}"
                )
        digest = _nonempty_string(
            item.get("sha256"),
            "parameter expansion SHA-256",
        )
        calculated = sha256("\n".join(instances).encode("utf-8")).hexdigest()
        if digest != calculated:
            raise AcceptanceVerificationError(
                f"parameter expansion digest mismatch: {node_id}"
            )
        expansions.append(
            ParameterExpansion(
                node_id=node_id,
                instance_node_ids=instances,
            )
        )
        all_instances.extend(instances)
    _unique(
        (expansion.node_id for expansion in expansions),
        "parameter expansion base node",
    )
    _unique(all_instances, "parameter instance across expansions")
    node_order = {node.node_id: index for index, node in enumerate(nodes)}
    expansion_order = [
        node_order[expansion.node_id] for expansion in expansions
    ]
    if expansion_order != sorted(expansion_order):
        raise AcceptanceVerificationError(
            "parameter expansions must preserve manifest node order"
        )
    return tuple(expansions)


def _requirement_activation_slices(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> tuple[RequirementActivationSlice, ...]:
    if not isinstance(raw, list):
        raise AcceptanceVerificationError(
            "requirement activation slices must be a list"
        )
    active_by_requirement: dict[str, tuple[AcceptanceNode, ...]] = {}
    planned_by_requirement: dict[str, tuple[AcceptanceNode, ...]] = {}
    for requirement_id in _EXPECTED_REQUIREMENT_IDS:
        active = tuple(
            node
            for node in nodes
            if requirement_id in node.requirement_ids
            and node.lifecycle == "active"
        )
        planned = tuple(
            node
            for node in nodes
            if requirement_id in node.requirement_ids
            and node.lifecycle == "planned"
        )
        if active:
            active_by_requirement[requirement_id] = active
        if planned:
            planned_by_requirement[requirement_id] = planned
    mixed_requirements = set(active_by_requirement) & set(
        planned_by_requirement
    )
    slices: list[RequirementActivationSlice] = []
    expected_keys = {
        "requirement_id",
        "phase",
        "active_owner",
        "active_scope",
        "active_node_ids",
        "remaining_owner",
        "remaining_scope",
        "planned_node_ids",
        "reviewed_by",
        "evidence",
    }
    for value in raw:
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise AcceptanceVerificationError(
                "requirement activation slice has invalid shape"
            )
        item = cast(dict[str, object], value)
        requirement_id = _nonempty_string(
            item.get("requirement_id"),
            "slice requirement_id",
        )
        if requirement_id not in mixed_requirements:
            raise AcceptanceVerificationError(
                "requirement activation slice is not mixed-lifecycle:"
                f" {requirement_id}"
            )
        phase = _phase(item.get("phase"), "slice phase")
        if phase > current_phase:
            raise AcceptanceVerificationError(
                "requirement activation slice phase is not implemented"
            )
        active_nodes = active_by_requirement[requirement_id]
        planned_nodes = planned_by_requirement[requirement_id]
        expected_phase = min(node.active_from_phase for node in active_nodes)
        if phase != expected_phase:
            raise AcceptanceVerificationError(
                "requirement activation slice phase differs from its first"
                f" active node: {requirement_id}"
            )
        active_node_ids = _string_list(
            item.get("active_node_ids"),
            "slice active_node_ids",
        )
        planned_node_ids = _string_list(
            item.get("planned_node_ids"),
            "slice planned_node_ids",
        )
        _unique(active_node_ids, "slice active node ID")
        _unique(planned_node_ids, "slice planned node ID")
        if active_node_ids != tuple(node.node_id for node in active_nodes):
            raise AcceptanceVerificationError(
                "requirement activation slice active inventory changed:"
                f" {requirement_id}"
            )
        if planned_node_ids != tuple(node.node_id for node in planned_nodes):
            raise AcceptanceVerificationError(
                "requirement activation slice planned inventory changed:"
                f" {requirement_id}"
            )
        active_owner = _slice_detail(item, "active_owner")
        active_scope = _slice_detail(item, "active_scope")
        remaining_owner = _slice_detail(item, "remaining_owner")
        remaining_scope = _slice_detail(item, "remaining_scope")
        reviewed_by = _nonempty_string(
            item.get("reviewed_by"),
            "slice reviewed_by",
        )
        if reviewed_by != _EXPECTED_IMPLEMENTATION_OWNER:
            raise AcceptanceVerificationError(
                "requirement activation slice lacks implementation review"
            )
        evidence = _slice_detail(item, "evidence")
        slices.append(
            RequirementActivationSlice(
                requirement_id=requirement_id,
                phase=phase,
                active_owner=active_owner,
                active_scope=active_scope,
                active_node_ids=active_node_ids,
                remaining_owner=remaining_owner,
                remaining_scope=remaining_scope,
                planned_node_ids=planned_node_ids,
                reviewed_by=reviewed_by,
                evidence=evidence,
            )
        )
    _unique(
        (item.requirement_id for item in slices),
        "requirement activation slice ID",
    )
    observed = {item.requirement_id for item in slices}
    if observed != mixed_requirements:
        raise AcceptanceVerificationError(
            "mixed-lifecycle requirements lack exact activation slices:"
            f" expected={sorted(mixed_requirements)},"
            f" observed={sorted(observed)}"
        )
    return tuple(slices)


def _slice_detail(item: dict[str, object], field: str) -> str:
    value = _nonempty_string(item.get(field), f"slice {field}").strip()
    if len(value) < 20 or value.lower() in {
        "pending",
        "placeholder",
        "tbd",
        "todo",
    }:
        raise AcceptanceVerificationError(
            f"requirement activation slice {field} is not concrete"
        )
    return value


def _activation_history(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> None:
    if not isinstance(raw, list) or len(raw) != current_phase + 1:
        raise AcceptanceVerificationError(
            "activation history must contain every implemented phase"
        )
    observed: list[str] = []
    for expected_phase, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise AcceptanceVerificationError(
                "activation history entries must be objects"
            )
        item = cast(dict[str, object], entry)
        _exact_keys(item, {"phase", "node_ids"}, "activation history entry")
        phase = _phase(item.get("phase"), "activation history phase")
        if phase != expected_phase:
            raise AcceptanceVerificationError(
                "activation history phases must be contiguous"
            )
        node_ids = _string_list(item.get("node_ids"), "activation node_ids")
        _unique(node_ids, f"activation nodes at phase {phase}")
        expected = tuple(
            node.id for node in nodes if node.active_from_phase == phase
        )
        if set(node_ids) != set(expected) or len(node_ids) != len(expected):
            raise AcceptanceVerificationError(
                f"activation history mismatch at phase {phase}"
            )
        observed.extend(node_ids)
    active_ids = [node.id for node in nodes if node.lifecycle == "active"]
    if set(observed) != set(active_ids) or len(observed) != len(active_ids):
        raise AcceptanceVerificationError(
            "activation history does not exactly preserve active nodes"
        )


def _activation_snapshots(
    raw_snapshots: object,
    raw_replacements: object,
    raw_requirement_slices: object,
    raw_parameter_expansions: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> None:
    if (
        not isinstance(raw_snapshots, list)
        or len(raw_snapshots) != current_phase + 1
    ):
        raise AcceptanceVerificationError(
            "activation snapshots must preserve every implemented phase"
        )
    if not isinstance(raw_replacements, list):
        raise AcceptanceVerificationError("replacements must be a list")
    if not isinstance(raw_parameter_expansions, list):
        raise AcceptanceVerificationError(
            "parameter expansions must be a list"
        )
    replacements: dict[str, tuple[str, ...]] = {}
    replacement_phases: dict[str, int] = {}
    replacement_requirements: dict[str, frozenset[str]] = {}
    replacement_targets: set[str] = set()
    node_by_id = {node.node_id: node for node in nodes}
    for raw in raw_replacements:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "acceptance replacement must be an object"
            )
        replacement = cast(dict[str, object], raw)
        _exact_keys(
            replacement,
            {
                "phase",
                "old_node_id",
                "replacement_node_ids",
                "requirement_ids",
                "reviewed_by",
                "evidence",
            },
            "acceptance replacement",
        )
        phase = _phase(replacement.get("phase"), "replacement phase")
        if phase > current_phase:
            raise AcceptanceVerificationError(
                "replacement phase is not implemented"
            )
        old_node_id = _node_id(replacement.get("old_node_id"))
        if old_node_id in replacements:
            raise AcceptanceVerificationError(
                f"acceptance node is replaced more than once: {old_node_id}"
            )
        replacement_ids = _string_list(
            replacement.get("replacement_node_ids"),
            "replacement node_ids",
        )
        _unique(replacement_ids, "replacement node ID")
        for node_id in replacement_ids:
            _node_id(node_id)
            if node_id in replacement_targets:
                raise AcceptanceVerificationError(
                    f"replacement target is reused: {node_id}"
                )
            replacement_targets.add(node_id)
        requirement_ids = _string_list(
            replacement.get("requirement_ids"),
            "replacement requirement_ids",
        )
        _unique(requirement_ids, "replacement requirement ID")
        if not set(requirement_ids) <= _EXPECTED_REQUIREMENT_IDS:
            raise AcceptanceVerificationError(
                f"replacement owns unknown requirements: {old_node_id}"
            )
        _nonempty_string(
            replacement.get("reviewed_by"), "replacement reviewed_by"
        )
        _nonempty_string(replacement.get("evidence"), "replacement evidence")
        replacements[old_node_id] = replacement_ids
        replacement_phases[old_node_id] = phase
        replacement_requirements[old_node_id] = frozenset(requirement_ids)
    ledger_digest = sha256(
        dumps(
            {
                "activation_snapshots": raw_snapshots,
                "replacements": raw_replacements,
                "requirement_activation_slices": raw_requirement_slices,
                "parameter_expansions": raw_parameter_expansions,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if ledger_digest != _EXPECTED_ACCEPTANCE_LEDGER_SHA256:
        raise AcceptanceVerificationError(
            "acceptance activation ledger changed without verifier review"
        )
    snapshots: list[tuple[str, ...]] = []
    for expected_phase, raw in enumerate(raw_snapshots):
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "activation snapshot must be an object"
            )
        snapshot = cast(dict[str, object], raw)
        _exact_keys(
            snapshot, {"phase", "node_ids", "sha256"}, "activation snapshot"
        )
        if _phase(snapshot.get("phase"), "snapshot phase") != expected_phase:
            raise AcceptanceVerificationError(
                "activation snapshot phases must be contiguous"
            )
        node_ids = _string_list(snapshot.get("node_ids"), "snapshot node_ids")
        _unique(node_ids, f"snapshot node ID at phase {expected_phase}")
        for node_id in node_ids:
            _node_id(node_id)
        digest = _nonempty_string(snapshot.get("sha256"), "snapshot SHA-256")
        calculated = sha256("\n".join(node_ids).encode("utf-8")).hexdigest()
        if digest != calculated:
            raise AcceptanceVerificationError(
                "activation snapshot digest mismatch at phase"
                f" {expected_phase}"
            )
        snapshots.append(node_ids)

    replacements_by_phase = {
        phase: {
            old_node_id
            for old_node_id, replacement_phase in replacement_phases.items()
            if replacement_phase == phase
        }
        for phase in range(current_phase + 1)
    }
    targets_by_phase = {
        phase: {
            target
            for old_node_id in replacements_by_phase[phase]
            for target in replacements[old_node_id]
        }
        for phase in range(current_phase + 1)
    }
    previous: set[str] = set()
    all_snapshot_ids: set[str] = set()
    for phase, snapshot_ids in enumerate(snapshots):
        current = set(snapshot_ids)
        added = current - previous
        removed = previous - current
        expected_removed = replacements_by_phase[phase]
        if removed != expected_removed:
            raise AcceptanceVerificationError(
                "activation snapshot removals lack exact reviewed tombstones"
                f" at phase {phase}: expected={sorted(expected_removed)},"
                f" observed={sorted(removed)}"
            )
        missing_targets = targets_by_phase[phase] - added
        if missing_targets:
            raise AcceptanceVerificationError(
                "replacement targets are not same-phase snapshot additions:"
                f" phase={phase}, missing={sorted(missing_targets)}"
            )
        expected_current_additions = {
            node.node_id
            for node in nodes
            if node.lifecycle == "active" and node.active_from_phase == phase
        }
        missing_current = expected_current_additions - added
        if missing_current:
            raise AcceptanceVerificationError(
                "active nodes are absent from their activation snapshot:"
                f" phase={phase}, missing={sorted(missing_current)}"
            )
        for node_id in added:
            node = node_by_id.get(node_id)
            if node is not None:
                if (
                    node.lifecycle != "active"
                    or node.active_from_phase != phase
                ):
                    raise AcceptanceVerificationError(
                        "snapshot node was added outside its activation phase:"
                        f" {node_id}"
                    )
            elif node_id not in replacements:
                raise AcceptanceVerificationError(
                    "historical snapshot node lacks a later tombstone:"
                    f" {node_id}"
                )
        previous = current
        all_snapshot_ids.update(current)

    historical_only = all_snapshot_ids - set(node_by_id)
    if not historical_only <= set(replacements):
        raise AcceptanceVerificationError(
            "historical acceptance nodes lack reviewed tombstones"
        )
    for old_node_id, replacement_ids in replacements.items():
        phase = replacement_phases[old_node_id]
        if phase == 0 or old_node_id not in set(snapshots[phase - 1]):
            raise AcceptanceVerificationError(
                "replacement old node was not active immediately before its"
                f" tombstone: {old_node_id}"
            )
        if old_node_id in set(snapshots[phase]):
            raise AcceptanceVerificationError(
                f"replacement old node remains active: {old_node_id}"
            )
        target_requirements: set[str] = set()
        for target_id in replacement_ids:
            target = node_by_id.get(target_id)
            if target is not None:
                if target.active_from_phase != phase:
                    raise AcceptanceVerificationError(
                        "replacement target activated in another phase:"
                        f" {target_id}"
                    )
                target_requirements.update(target.requirement_ids)
                continue
            target_phase = replacement_phases.get(target_id)
            if target_phase is None or target_phase <= phase:
                raise AcceptanceVerificationError(
                    "replacement chain is cyclic or lacks a later tombstone:"
                    f" {target_id}"
                )
            target_requirements.update(replacement_requirements[target_id])
        if target_requirements != set(replacement_requirements[old_node_id]):
            raise AcceptanceVerificationError(
                "replacement does not exactly preserve requirements:"
                f" {old_node_id}"
            )

    active = tuple(
        node.node_id for node in nodes if node.lifecycle == "active"
    )
    if snapshots[-1] != active:
        raise AcceptanceVerificationError(
            "latest activation snapshot differs from active inventory"
        )


def _validate_contract_fixtures(
    manifest: AcceptanceManifest,
    fixtures: Path,
    root: Path,
) -> None:
    decision_surfaces, public_envelopes = _validate_decisions(
        fixtures / "contract_decisions.json"
    )
    requirements = _validate_requirements(
        fixtures / "requirements_traceability.json", manifest
    )
    _validate_failure_matrix(
        fixtures / "failure_matrix.json",
        manifest,
        requirements,
        decision_surfaces,
        public_envelopes,
    )
    _validate_type_manifest(
        fixtures / "type_contract_manifest.json",
        manifest.current_phase,
        root,
    )
    _validate_no_bc(fixtures / "no_bc_removals.json")
    _validate_evidence(fixtures / "baseline_evidence.json", manifest, root)


def _validate_requirements(
    path: Path,
    manifest: AcceptanceManifest,
) -> frozenset[str]:
    payload = _strict_mapping(path, "requirements traceability")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "source_sections",
            "catalog_sha256",
            "requirements",
        },
        "requirements traceability",
    )
    _header(payload, "requirements traceability")
    sections = _string_list(payload.get("source_sections"), "source_sections")
    expected_sections = tuple(str(value) for value in range(7, 27))
    if sections != expected_sections:
        raise AcceptanceVerificationError(
            "requirements source sections must be the frozen 7 through 26"
            " inventory"
        )
    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list):
        raise AcceptanceVerificationError("requirements must be a list")
    manifest_by_node = {node.node_id: node for node in manifest.nodes}
    requirement_nodes: dict[str, tuple[str, ...]] = {}
    observed_ids: list[str] = []
    mapped_nodes: set[str] = set()
    for raw in raw_requirements:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError("requirement must be an object")
        item = cast(dict[str, object], raw)
        _exact_keys(
            item,
            {
                "id",
                "source_section",
                "normative_level",
                "paraphrase",
                "owner",
                "implementation_artifacts",
                "test_node_ids",
            },
            "requirement",
        )
        requirement_id = _nonempty_string(item.get("id"), "requirement id")
        observed_ids.append(requirement_id)
        _nonempty_string(
            item.get("source_section"), "requirement source_section"
        )
        level = _nonempty_string(
            item.get("normative_level"), "normative_level"
        )
        if level not in {"MUST", "SHOULD", "MAY", "SCENARIO"}:
            raise AcceptanceVerificationError(
                f"invalid normative level for {requirement_id}: {level}"
            )
        _nonempty_string(item.get("paraphrase"), "requirement paraphrase")
        _nonempty_string(item.get("owner"), "requirement owner")
        _string_list(
            item.get("implementation_artifacts"), "implementation_artifacts"
        )
        node_ids = _string_list(item.get("test_node_ids"), "test_node_ids")
        _unique(node_ids, f"test node for {requirement_id}")
        requirement_nodes[requirement_id] = node_ids
        for node_id in node_ids:
            node = manifest_by_node.get(node_id)
            if node is None:
                raise AcceptanceVerificationError(
                    f"unmapped requirement node {node_id} for {requirement_id}"
                )
            if requirement_id not in node.requirement_ids:
                raise AcceptanceVerificationError(
                    f"non-reciprocal requirement mapping: {requirement_id},"
                    f" {node_id}"
                )
            mapped_nodes.add(node_id)
    _unique(observed_ids, "requirement ID")
    _verify_digest(
        raw_requirements,
        payload.get("catalog_sha256"),
        _EXPECTED_REQUIREMENTS_SHA256,
        "requirements catalog",
    )
    if frozenset(observed_ids) != _EXPECTED_REQUIREMENT_IDS:
        missing = sorted(_EXPECTED_REQUIREMENT_IDS - frozenset(observed_ids))
        unexpected = sorted(
            frozenset(observed_ids) - _EXPECTED_REQUIREMENT_IDS
        )
        raise AcceptanceVerificationError(
            f"requirements inventory mismatch: missing={missing},"
            f" unexpected={unexpected}"
        )
    for node in manifest.nodes:
        reciprocal = tuple(
            requirement_id
            for requirement_id, node_ids in requirement_nodes.items()
            if node.node_id in node_ids
        )
        if set(reciprocal) != set(node.requirement_ids) or len(
            reciprocal
        ) != len(node.requirement_ids):
            raise AcceptanceVerificationError(
                "acceptance node requirement mapping is not exact:"
                f" {node.node_id}"
            )
        for requirement_id in node.requirement_ids:
            if requirement_id not in _EXPECTED_REQUIREMENT_IDS:
                raise AcceptanceVerificationError(
                    "acceptance node owns unknown requirement:"
                    f" {requirement_id}"
                )
        if node.node_id not in mapped_nodes:
            raise AcceptanceVerificationError(
                "acceptance node has no reciprocal requirement:"
                f" {node.node_id}"
            )
    return frozenset(observed_ids)


def _validate_failure_matrix(
    path: Path,
    manifest: AcceptanceManifest,
    requirements: frozenset[str],
    decision_surfaces: frozenset[str],
    public_envelopes: frozenset[str],
) -> None:
    payload = _strict_mapping(path, "failure matrix")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "matrix_sha256",
            "observation_window",
            "domain_side_effect_scope",
            "surfaces",
            "conditions",
            "cells",
        },
        "failure matrix",
    )
    _header(payload, "failure matrix")
    _nonempty_string(
        payload.get("observation_window"),
        "failure observation_window",
    )
    _nonempty_string(
        payload.get("domain_side_effect_scope"),
        "failure domain_side_effect_scope",
    )
    surfaces = _record_ids(
        payload.get("surfaces"),
        {"id", "description", "active_from_phase"},
        "failure surface",
    )
    if frozenset(surfaces) != decision_surfaces:
        raise AcceptanceVerificationError(
            "failure surfaces differ from the public capability inventory"
        )
    conditions = _record_ids(
        payload.get("conditions"),
        {"id", "description", "requirement_id", "active_from_phase"},
        "failure condition",
    )
    if frozenset(conditions) != _EXPECTED_FAILURE_CONDITIONS:
        raise AcceptanceVerificationError(
            "failure condition inventory must contain all fifteen conditions"
        )
    raw_conditions = cast(list[dict[str, object]], payload["conditions"])
    condition_phases: dict[str, int] = {}
    condition_requirements: dict[str, str] = {}
    for condition in raw_conditions:
        requirement_id = _nonempty_string(
            condition.get("requirement_id"), "failure requirement_id"
        )
        if requirement_id not in requirements:
            raise AcceptanceVerificationError(
                f"failure condition owns unknown requirement: {requirement_id}"
            )
        condition_id = _nonempty_string(condition.get("id"), "condition id")
        condition_phases[condition_id] = _phase(
            condition.get("active_from_phase"),
            "condition active_from_phase",
        )
        condition_requirements[condition_id] = requirement_id
    raw_surfaces = cast(list[dict[str, object]], payload["surfaces"])
    surface_phases = {
        _nonempty_string(item.get("id"), "surface id"): _phase(
            item.get("active_from_phase"), "surface active_from_phase"
        )
        for item in raw_surfaces
    }
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        raise AcceptanceVerificationError("failure cells must be a list")
    expected_cells = {
        (condition, surface)
        for condition in conditions
        for surface in surfaces
    }
    observed_cells: set[tuple[str, str]] = set()
    manifest_nodes = {node.node_id: node for node in manifest.nodes}
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError("failure cell must be an object")
        cell = cast(dict[str, object], raw)
        _exact_keys(
            cell,
            {
                "condition_id",
                "surface_id",
                "applicable",
                "active_from_phase",
                "expected_transition",
                "public_result",
                "status_or_exit",
                "provider_call_count",
                "domain_side_effect_count",
                "negative_e2e_node",
                "non_applicability_reason",
                "reviewed_by",
            },
            "failure cell",
        )
        condition_id = _nonempty_string(
            cell.get("condition_id"), "condition_id"
        )
        surface_id = _nonempty_string(cell.get("surface_id"), "surface_id")
        key = (condition_id, surface_id)
        if key in observed_cells:
            raise AcceptanceVerificationError(f"duplicate failure cell: {key}")
        observed_cells.add(key)
        if key not in expected_cells:
            raise AcceptanceVerificationError(f"unknown failure cell: {key}")
        active_from_phase = _phase(
            cell.get("active_from_phase"), "failure active_from_phase"
        )
        expected_phase = max(
            surface_phases[surface_id],
            condition_phases[condition_id],
        )
        if active_from_phase != expected_phase:
            raise AcceptanceVerificationError(
                f"failure cell activation differs from its surface: {key}"
            )
        applicable = cell.get("applicable")
        if type(applicable) is not bool:
            raise AcceptanceVerificationError(
                f"failure applicability must be boolean: {key}"
            )
        transition = _nonempty_string(
            cell.get("expected_transition"), "expected_transition"
        )
        public_result = _nonempty_string(
            cell.get("public_result"), "public_result"
        )
        status_or_exit = _nonempty_string(
            cell.get("status_or_exit"), "status_or_exit"
        )
        provider_calls = _nonnegative_int(
            cell.get("provider_call_count"), "provider_call_count"
        )
        side_effects = _nonnegative_int(
            cell.get("domain_side_effect_count"), "domain_side_effect_count"
        )
        node_id = cell.get("negative_e2e_node")
        reason = cell.get("non_applicability_reason")
        reviewed_by = cell.get("reviewed_by")
        if key == ("INPUT-F-15", "mcp-inbound-task") and cell != {
            "condition_id": "INPUT-F-15",
            "surface_id": "mcp-inbound-task",
            "applicable": True,
            "active_from_phase": 10,
            "expected_transition": "running->running",
            "public_result": "envelope=mcp.ordinary_result.v1",
            "status_or_exit": "result=ordinary",
            "provider_call_count": 1,
            "domain_side_effect_count": 0,
            "negative_e2e_node": (
                "tests/input/failure_matrix_mcp_e2e_test.py::test_input_f_15"
            ),
            "non_applicability_reason": None,
            "reviewed_by": None,
        }:
            raise AcceptanceVerificationError(
                "MCP capability-absent fallback must preserve ordinary"
                " execution"
            )
        if applicable:
            for label, value in (
                ("expected_transition", transition),
                ("public_result", public_result),
                ("status_or_exit", status_or_exit),
            ):
                _unambiguous_outcome(value, label, key)
            transition_match = _TRANSITION_PATTERN.fullmatch(transition)
            if (
                transition_match is None
                or not set(transition_match.groups()) <= _INTERACTION_STATES
            ):
                raise AcceptanceVerificationError(
                    "failure transition is not a canonical interaction-state"
                    f" transition: {key}"
                )
            public_result_match = _PUBLIC_RESULT_PATTERN.fullmatch(
                public_result
            )
            if (
                public_result_match is None
                or public_result_match.group(1) not in public_envelopes
            ):
                raise AcceptanceVerificationError(
                    "failure public result is not a cataloged literal"
                    f" envelope: {key}"
                )
            status_match = _STATUS_OR_EXIT_PATTERN.fullmatch(status_or_exit)
            if (
                status_match is None
                or status_match.group(1) not in _STATUS_OR_EXIT_KEYS
            ):
                raise AcceptanceVerificationError(
                    f"failure status or exit is not one machine literal: {key}"
                )
            if side_effects != 0:
                raise AcceptanceVerificationError(
                    f"failure cell permits a domain side effect: {key}"
                )
            exact_node = _node_id(node_id)
            manifest_node = manifest_nodes.get(exact_node)
            if manifest_node is None or manifest_node.category != "public_e2e":
                raise AcceptanceVerificationError(
                    f"failure cell lacks an owned negative E2E node: {key}"
                )
            if (
                condition_requirements[condition_id]
                not in manifest_node.requirement_ids
            ):
                raise AcceptanceVerificationError(
                    f"failure cell node does not own its requirement: {key}"
                )
            if manifest_node.active_from_phase != active_from_phase:
                raise AcceptanceVerificationError(
                    "failure cell node activation does not match its cell:"
                    f" {key}"
                )
            if reason is not None:
                raise AcceptanceVerificationError(
                    "applicable failure cell has a non-applicability reason:"
                    f" {key}"
                )
            if reviewed_by is not None:
                raise AcceptanceVerificationError(
                    f"applicable failure cell has an N/A reviewer: {key}"
                )
            if (
                transition == "not_applicable"
                or public_result == "none"
                or status_or_exit == "not_applicable"
            ):
                raise AcceptanceVerificationError(
                    f"applicable failure cell has placeholder behavior: {key}"
                )
        else:
            if (
                node_id is not None
                or transition != "not_applicable"
                or public_result != "none"
                or status_or_exit != "not_applicable"
                or provider_calls != 0
                or side_effects != 0
            ):
                raise AcceptanceVerificationError(
                    "non-applicable failure cell has executable behavior:"
                    f" {key}"
                )
            concrete = _nonempty_string(reason, "non_applicability_reason")
            normalized = concrete.lower().replace("_", " ").strip(" .")
            if (
                len(concrete) < 80
                or surface_id not in concrete
                or condition_id not in concrete
                or normalized in {"n/a", "na", "not applicable"}
                or "another lifecycle owner" in normalized
                or "declared ownership cannot exercise" in normalized
            ):
                raise AcceptanceVerificationError(
                    "non-applicable failure cell lacks a concrete reviewed"
                    f" reason: {key}"
                )
            reviewer = _nonempty_string(reviewed_by, "reviewed_by")
            reviewer_label = reviewer.lower().replace("_", "-")
            if any(
                claim in reviewer_label
                for claim in ("audit", "auditor", "independent")
            ):
                raise AcceptanceVerificationError(
                    "N/A reviewer claims unrecorded independent approval:"
                    f" {key}"
                )
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        raise AcceptanceVerificationError(
            f"failure matrix is incomplete: missing={missing}"
        )
    _verify_digest(
        {
            "observation_window": payload["observation_window"],
            "domain_side_effect_scope": payload["domain_side_effect_scope"],
            "surfaces": payload["surfaces"],
            "conditions": payload["conditions"],
            "cells": payload["cells"],
        },
        payload.get("matrix_sha256"),
        _EXPECTED_FAILURE_MATRIX_SHA256,
        "failure matrix",
    )


def _validate_type_manifest(
    path: Path,
    acceptance_phase: int,
    root: Path,
) -> None:
    payload = _strict_mapping(path, "type-contract manifest")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "activation_history",
            "activation_snapshots",
            "planned_replacements",
            "replacements",
            "fixtures",
        },
        "type-contract manifest",
    )
    _header(payload, "type-contract manifest")
    current_phase = _phase(payload.get("current_phase"), "type current_phase")
    if current_phase != acceptance_phase:
        raise AcceptanceVerificationError(
            "type and acceptance manifests must implement the same phase"
        )
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise AcceptanceVerificationError(
            "type fixtures must be a non-empty list"
        )
    identifiers: list[str] = []
    active_paths: list[str] = []
    active_ids: list[str] = []
    for raw in raw_fixtures:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError("type fixture must be an object")
        item = cast(dict[str, object], raw)
        _exact_keys(
            item,
            {
                "id",
                "kind",
                "lifecycle",
                "active_from_phase",
                "path",
                "expected_diagnostics",
            },
            "type fixture",
        )
        identifier = _nonempty_string(item.get("id"), "type fixture id")
        identifiers.append(identifier)
        kind = _nonempty_string(item.get("kind"), "type fixture kind")
        if kind not in {"positive", "negative"}:
            raise AcceptanceVerificationError(
                f"invalid type fixture kind: {kind}"
            )
        active_from = _phase(
            item.get("active_from_phase"), "type active_from_phase"
        )
        lifecycle = _nonempty_string(item.get("lifecycle"), "type lifecycle")
        if lifecycle not in {"active", "planned", "replaced"}:
            raise AcceptanceVerificationError(
                f"invalid type fixture lifecycle: {lifecycle}"
            )
        expected = "active" if active_from <= current_phase else "planned"
        if lifecycle != "replaced" and lifecycle != expected:
            raise AcceptanceVerificationError(
                f"type fixture lifecycle regression: {item.get('id')}"
            )
        if lifecycle == "replaced" and active_from > current_phase:
            raise AcceptanceVerificationError(
                f"unimplemented type fixture replacement: {item.get('id')}"
            )
        if lifecycle == "active":
            active_ids.append(identifier)
        raw_path = _nonempty_string(item.get("path"), "type fixture path")
        if lifecycle != "replaced":
            active_paths.append(raw_path)
        _type_fixture_path(raw_path, root)
        diagnostics = item.get("expected_diagnostics")
        if not isinstance(diagnostics, list) or not all(
            isinstance(value, str) and value for value in diagnostics
        ):
            raise AcceptanceVerificationError(
                "type expected_diagnostics must be a string list"
            )
        if (kind == "positive") is bool(diagnostics):
            raise AcceptanceVerificationError(
                f"type diagnostics do not match fixture kind: {item.get('id')}"
            )
    _unique(identifiers, "type fixture ID")
    _unique(active_paths, "type fixture path")
    history = payload.get("activation_history")
    if not isinstance(history, list) or len(history) != current_phase + 1:
        raise AcceptanceVerificationError(
            "type activation history must contain every implemented phase"
        )
    observed_active: list[str] = []
    for expected_phase, raw in enumerate(history):
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "type activation history entry must be an object"
            )
        entry = cast(dict[str, object], raw)
        _exact_keys(entry, {"phase", "fixture_ids"}, "type activation entry")
        if (
            _phase(entry.get("phase"), "type activation phase")
            != expected_phase
        ):
            raise AcceptanceVerificationError(
                "type activation history phases must be contiguous"
            )
        phase_ids = _string_list(entry.get("fixture_ids"), "type fixture_ids")
        expected_ids = tuple(
            cast(str, item["id"])
            for item in cast(list[dict[str, object]], raw_fixtures)
            if item.get("active_from_phase") == expected_phase
            and item.get("lifecycle") == "active"
        )
        if set(phase_ids) != set(expected_ids) or len(phase_ids) != len(
            expected_ids
        ):
            raise AcceptanceVerificationError(
                f"type activation history mismatch at phase {expected_phase}"
            )
        observed_active.extend(phase_ids)
    if set(observed_active) != set(active_ids) or len(observed_active) != len(
        active_ids
    ):
        raise AcceptanceVerificationError(
            "type activation history does not preserve active fixtures"
        )


def _validate_decisions(
    path: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    payload = _strict_mapping(path, "contract decisions")
    required = {
        "schema_version",
        "feature",
        "identity",
        "request_bounds",
        "question_contracts",
        "state_transitions",
        "outcome_to_model",
        "execution",
        "capability_matrix",
        "protocol_projection",
        "privacy",
        "error_status",
        "repeated_requests",
        "activation",
        "capacity_budgets",
        "contract_sha256",
    }
    _exact_keys(payload, required, "contract decisions")
    _header(payload, "contract decisions")
    for key in required - {"schema_version", "feature", "contract_sha256"}:
        value = payload.get(key)
        if not isinstance(value, (dict, list)) or not value:
            raise AcceptanceVerificationError(
                f"contract decision {key} must be populated"
            )
    activation = cast(dict[str, object], payload["activation"])
    if activation.get("production_default") != "absent":
        raise AcceptanceVerificationError(
            "structured input must remain absent in production"
        )
    content = {
        key: value
        for key, value in payload.items()
        if key != "contract_sha256"
    }
    _verify_digest(
        content,
        payload.get("contract_sha256"),
        _EXPECTED_DECISIONS_SHA256,
        "contract decisions",
    )
    _validate_protocol_decision_shapes(payload)
    capability_matrix = cast(dict[str, object], payload["capability_matrix"])
    surface_ids = _string_list(
        capability_matrix.get("public_failure_surface_ids"),
        "public failure surface IDs",
    )
    _unique(surface_ids, "public failure surface ID")
    error_status = cast(dict[str, object], payload["error_status"])
    raw_catalog = error_status.get("public_envelope_catalog")
    if not isinstance(raw_catalog, dict) or not raw_catalog:
        raise AcceptanceVerificationError(
            "public envelope catalog must be a non-empty object"
        )
    envelope_ids: list[str] = []
    for raw_id, envelope in raw_catalog.items():
        if (
            not isinstance(raw_id, str)
            or _PUBLIC_RESULT_PATTERN.fullmatch(f"envelope={raw_id}") is None
            or not isinstance(envelope, dict)
            or not envelope
        ):
            raise AcceptanceVerificationError(
                "public envelope catalog entry has an invalid literal shape"
            )
        envelope_ids.append(raw_id)
    _unique(envelope_ids, "public envelope catalog ID")
    return frozenset(surface_ids), frozenset(envelope_ids)


def _validate_protocol_decision_shapes(payload: dict[str, object]) -> None:
    projection = _decision_mapping(
        payload.get("protocol_projection"), "protocol"
    )
    a2a = _decision_mapping(projection.get("a2a"), "A2A")
    expected_task_states = [
        "TASK_STATE_UNSPECIFIED",
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
        "TASK_STATE_FAILED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_REJECTED",
        "TASK_STATE_AUTH_REQUIRED",
    ]
    if a2a.get("task_states") != expected_task_states:
        raise AcceptanceVerificationError(
            "A2A task states must contain the complete ordered 1.0 enum"
        )
    error_status = _decision_mapping(
        payload.get("error_status"), "error status"
    )
    a2a_errors = _decision_mapping(error_status.get("a2a"), "A2A errors")
    core_errors = _decision_mapping(a2a_errors.get("core"), "A2A core errors")
    if (
        core_errors.get("push_notification_not_supported") != -32003
        or core_errors.get("version_not_supported") != -32009
    ):
        raise AcceptanceVerificationError(
            "A2A reserved core error codes are incorrect"
        )

    mcp = _decision_mapping(projection.get("mcp"), "MCP")
    elicitation = _decision_mapping(mcp.get("elicitation"), "MCP elicitation")
    requested = _decision_mapping(
        elicitation.get("requestedSchema"), "MCP requestedSchema"
    )
    _exact_keys(
        requested,
        {"allowed_top_level_keys", "type", "properties", "required"},
        "MCP requestedSchema",
    )
    if requested.get("allowed_top_level_keys") != [
        "$schema",
        "type",
        "properties",
        "required",
    ]:
        raise AcceptanceVerificationError(
            "MCP requestedSchema top-level keys are incorrect"
        )
    requested_type = _decision_mapping(
        requested.get("type"), "MCP requestedSchema type"
    )
    if (
        set(requested_type) != {"const"}
        or requested_type.get("const") != "object"
    ):
        raise AcceptanceVerificationError(
            "MCP requestedSchema type must be the object literal"
        )
    properties = _decision_mapping(
        requested.get("properties"), "MCP requestedSchema properties"
    )
    _exact_keys(
        properties,
        {"shape", "primitive_types", "single_select", "multiple_select"},
        "MCP requestedSchema properties",
    )
    if (
        properties.get("primitive_types")
        != [
            "string",
            "number",
            "integer",
            "boolean",
        ]
        or requested.get("required")
        != "optional array of unique property names"
    ):
        raise AcceptanceVerificationError(
            "MCP requestedSchema primitive or required fields are incorrect"
        )
    single = _decision_mapping(
        properties.get("single_select"), "MCP single-select schema"
    )
    _exact_keys(single, {"type", "enum"}, "MCP single-select schema")
    if single != {
        "type": "string",
        "enum": "non-empty unique stable string values",
    }:
        raise AcceptanceVerificationError(
            "MCP single-select schema must use a string enum"
        )
    multiple = _decision_mapping(
        properties.get("multiple_select"), "MCP multiple-select schema"
    )
    _exact_keys(
        multiple,
        {"type", "items", "uniqueItems"},
        "MCP multiple-select schema",
    )
    multiple_items = _decision_mapping(
        multiple.get("items"), "MCP multiple-select items"
    )
    _exact_keys(
        multiple_items,
        {"type", "enum"},
        "MCP multiple-select items",
    )
    if (
        multiple.get("type") != "array"
        or multiple.get("uniqueItems") is not True
        or multiple_items.get("type") != "string"
        or multiple_items.get("enum")
        != "non-empty unique stable string values"
    ):
        raise AcceptanceVerificationError(
            "MCP multiple-select schema must use a unique enum array"
        )

    tasks = _decision_mapping(mcp.get("tasks"), "MCP tasks")
    params = _decision_mapping(
        tasks.get("params_task_schema"), "MCP task params"
    )
    params_properties = _decision_mapping(
        params.get("properties"), "MCP task params properties"
    )
    _exact_keys(
        params,
        {"$schema", "type", "additionalProperties", "properties"},
        "MCP task params",
    )
    _exact_keys(params_properties, {"ttl"}, "MCP task params properties")
    ttl = _decision_mapping(params_properties.get("ttl"), "MCP task TTL")
    if (
        set(ttl) != {"type", "unit"}
        or params.get("type") != "object"
        or params.get("additionalProperties") is not False
        or ttl.get("unit") != "milliseconds"
        or ttl.get("type") != "number"
        or tasks.get("ttl_mapping")
        != "canonical continuation TTL seconds multiplied by 1000 without"
        " rounding"
    ):
        raise AcceptanceVerificationError(
            "MCP task TTL must use exact milliseconds"
        )
    create_result = _decision_mapping(
        tasks.get("CreateTaskResult"), "MCP CreateTaskResult"
    )
    result_properties = _decision_mapping(
        create_result.get("properties"), "MCP CreateTaskResult properties"
    )
    _exact_keys(
        create_result,
        {
            "$schema",
            "type",
            "additionalProperties",
            "required",
            "properties",
        },
        "MCP CreateTaskResult",
    )
    _exact_keys(
        result_properties,
        {"task", "_meta"},
        "MCP CreateTaskResult properties",
    )
    required = create_result.get("required")
    if (
        create_result.get("type") != "object"
        or required != ["task"]
        or "_meta" in cast(list[object], required)
        or create_result.get("additionalProperties") is not True
        or result_properties.get("_meta") != {"type": "object"}
    ):
        raise AcceptanceVerificationError(
            "MCP CreateTaskResult must permit optional _meta and extensions"
        )
    task_schema = _decision_mapping(tasks.get("task_schema"), "MCP Task")
    task_properties = _decision_mapping(
        task_schema.get("properties"), "MCP Task properties"
    )
    _exact_keys(
        task_schema,
        {
            "$schema",
            "type",
            "additionalProperties",
            "required",
            "properties",
        },
        "MCP Task",
    )
    expected_task_properties = {
        "taskId",
        "status",
        "statusMessage",
        "createdAt",
        "lastUpdatedAt",
        "ttl",
        "pollInterval",
    }
    task_id = _decision_mapping(task_properties.get("taskId"), "MCP taskId")
    task_status = _decision_mapping(
        task_properties.get("status"), "MCP task status"
    )
    status_message = _decision_mapping(
        task_properties.get("statusMessage"), "MCP task statusMessage"
    )
    created_at = _decision_mapping(
        task_properties.get("createdAt"), "MCP task createdAt"
    )
    last_updated_at = _decision_mapping(
        task_properties.get("lastUpdatedAt"), "MCP task lastUpdatedAt"
    )
    task_ttl = _decision_mapping(task_properties.get("ttl"), "MCP Task ttl")
    poll_interval = _decision_mapping(
        task_properties.get("pollInterval"), "MCP Task pollInterval"
    )
    if (
        set(task_properties) != expected_task_properties
        or task_schema.get("type") != "object"
        or task_schema.get("additionalProperties") is not False
        or task_schema.get("required")
        != ["taskId", "status", "createdAt", "lastUpdatedAt", "ttl"]
        or task_id != {"type": "string", "minLength": 1}
        or task_status
        != {
            "enum": [
                "working",
                "input_required",
                "completed",
                "failed",
                "cancelled",
            ]
        }
        or status_message != {"type": "string"}
        or created_at != {"type": "string", "format": "date-time"}
        or last_updated_at != {"type": "string", "format": "date-time"}
        or task_ttl != {"type": ["number", "null"]}
        or poll_interval != {"type": "number"}
        or result_properties.get("task")
        != {
            key: value
            for key, value in task_schema.items()
            if key != "$schema"
        }
    ):
        raise AcceptanceVerificationError(
            "MCP Task schema differs from the complete protocol contract"
        )
    if (
        tasks.get("request_type_task_capability_absent")
        != "receiver MUST process request normally and ignore params.task"
        " augmentation"
    ):
        raise AcceptanceVerificationError(
            "MCP request-type task capability fallback is incorrect"
        )
    generic_requirement = _decision_mapping(
        tasks.get("generic_receiver_task_requirement"),
        "MCP generic receiver task requirement",
    )
    if generic_requirement != {
        "omission_behavior": (
            "receiver MAY require task augmentation for a request type with"
            " declared support; omission MAY return -32600"
        ),
        "omission_error": -32600,
    }:
        raise AcceptanceVerificationError(
            "MCP generic receiver task requirement is incorrect"
        )
    tool_support = _decision_mapping(
        tasks.get("tool_execution_task_support"),
        "MCP tool execution task support",
    )
    if tool_support != {
        "absent": (
            "defaults to forbidden; attempted params.task SHOULD return -32601"
        ),
        "forbidden": "attempted params.task SHOULD return -32601",
        "optional": "client MAY invoke normally or with params.task",
        "required": (
            "client MUST invoke with params.task; omission MUST return -32601"
        ),
    }:
        raise AcceptanceVerificationError(
            "MCP tool execution task-support behavior is incorrect"
        )
    if tasks.get("initial_state") != "working":
        raise AcceptanceVerificationError(
            "MCP tasks must begin in the working state"
        )
    if tasks.get("legal_transitions") != [
        ["working", "input_required"],
        ["input_required", "working"],
        ["input_required", "completed"],
        ["input_required", "failed"],
        ["working", "completed"],
        ["working", "failed"],
        ["working", "cancelled"],
        ["input_required", "cancelled"],
    ]:
        raise AcceptanceVerificationError(
            "MCP task transitions differ from the frozen state graph"
        )
    mcp_errors = _decision_mapping(error_status.get("mcp"), "MCP errors")
    if mcp_errors != {
        "invalid_params": -32602,
        "unavailable": -32001,
        "unauthorized": -32003,
        "conflict": -32009,
        "expired": -32010,
        "receiver_task_augmentation_required": -32600,
        "tool_task_augmentation_forbidden": -32601,
        "tool_task_augmentation_required": -32601,
    }:
        raise AcceptanceVerificationError(
            "MCP receiver and tool task errors are conflated"
        )

    _validate_a2a_message_metadata(a2a)
    _validate_mcp_schema_examples(tasks)
    _validate_public_envelope_contract(error_status)


def _validate_a2a_message_metadata(a2a: dict[str, object]) -> None:
    extension = _decision_mapping(a2a.get("extension"), "A2A extension")
    schema = _decision_mapping(
        extension.get("message_metadata_schema"),
        "A2A message metadata schema",
    )
    schema_digest = _EXPECTED_PROTOCOL_SCHEMA_SHA256["a2a_message_metadata"]
    _verify_digest(
        schema,
        schema_digest,
        schema_digest,
        "A2A message metadata schema",
    )
    examples = _decision_mapping(
        extension.get("message_metadata_examples"),
        "A2A message metadata examples",
    )
    if set(examples) != {"request", "accept", "decline", "cancel"}:
        raise AcceptanceVerificationError(
            "A2A message metadata examples must cover every action"
        )
    _validate_schema_examples(
        schema,
        examples,
        "A2A message metadata",
        exercise_mutations=True,
    )


def _validate_mcp_schema_examples(tasks: dict[str, object]) -> None:
    params_schema = _decision_mapping(
        tasks.get("params_task_schema"), "MCP task params"
    )
    task_schema = _decision_mapping(tasks.get("task_schema"), "MCP Task")
    create_result_schema = _decision_mapping(
        tasks.get("CreateTaskResult"), "MCP CreateTaskResult"
    )
    for schema_name, schema in (
        ("mcp_params_task", params_schema),
        ("mcp_task", task_schema),
        ("mcp_create_task_result", create_result_schema),
    ):
        expected_digest = _EXPECTED_PROTOCOL_SCHEMA_SHA256[schema_name]
        _verify_digest(
            schema,
            expected_digest,
            expected_digest,
            f"protocol schema {schema_name}",
        )
    schemas = {
        "params_task_with_ttl": params_schema,
        "params_task_without_ttl": params_schema,
        "Task": task_schema,
        "CreateTaskResult": create_result_schema,
    }
    examples = _decision_mapping(
        tasks.get("schema_examples"), "MCP schema examples"
    )
    if set(examples) != set(schemas):
        raise AcceptanceVerificationError(
            "MCP schema examples must cover every frozen schema variant"
        )
    for name, schema in schemas.items():
        _validate_schema_examples(
            schema,
            {name: examples[name]},
            f"MCP {name}",
            exercise_mutations=False,
            allow_empty=name == "params_task_without_ttl",
        )
    _validate_mcp_schema_probes(
        params_schema,
        task_schema,
        create_result_schema,
    )


def _validate_mcp_schema_probes(
    params_schema: dict[str, object],
    task_schema: dict[str, object],
    create_result_schema: dict[str, object],
) -> None:
    validator_factory = _draft202012_validator()
    params_validator = validator_factory(params_schema)
    task_validator = validator_factory(task_schema)
    create_result_validator = validator_factory(create_result_schema)
    task = {
        "taskId": "task-0001",
        "status": "working",
        "statusMessage": "",
        "createdAt": "2026-07-20T00:00:00Z",
        "lastUpdatedAt": "2026-07-20T00:00:00Z",
        "ttl": 0,
        "pollInterval": -1,
    }
    if not all(
        params_validator.is_valid(probe)
        for probe in ({}, {"ttl": 0}, {"ttl": -1})
    ):
        raise AcceptanceVerificationError(
            "MCP task params reject a protocol-valid numeric TTL"
        )
    if any(
        params_validator.is_valid(probe)
        for probe in ({"ttl": None}, {"ttl": "1"}, {"extra": True})
    ):
        raise AcceptanceVerificationError(
            "MCP task params accept an invalid task augmentation"
        )
    if not task_validator.is_valid(task):
        raise AcceptanceVerificationError(
            "MCP Task rejects optional fields or protocol-valid bounds"
        )
    minimal_task = {
        key: value
        for key, value in task.items()
        if key not in {"statusMessage", "pollInterval"}
    }
    minimal_task["ttl"] = None
    if not task_validator.is_valid(minimal_task):
        raise AcceptanceVerificationError(
            "MCP Task rejects omitted optional fields or unlimited TTL"
        )
    invalid_tasks = []
    for key, value in (
        ("statusMessage", 1),
        ("pollInterval", None),
        ("ttl", "0"),
    ):
        invalid = deepcopy(task)
        invalid[key] = value
        invalid_tasks.append(invalid)
    missing_required = deepcopy(task)
    del missing_required["taskId"]
    invalid_tasks.append(missing_required)
    task_with_meta = deepcopy(task)
    task_with_meta["_meta"] = {}
    invalid_tasks.append(task_with_meta)
    if any(task_validator.is_valid(probe) for probe in invalid_tasks):
        raise AcceptanceVerificationError(
            "MCP Task accepts an invalid field, type, or omission"
        )
    valid_results = (
        {"task": task},
        {"task": task, "_meta": {}},
        {
            "task": task,
            "_meta": {"vendor": True},
            "vendor_extension": {"enabled": True},
        },
    )
    if not all(
        create_result_validator.is_valid(probe) for probe in valid_results
    ):
        raise AcceptanceVerificationError(
            "MCP CreateTaskResult rejects optional metadata or extensions"
        )
    if any(
        create_result_validator.is_valid(probe)
        for probe in ({}, {"task": task, "_meta": []})
    ):
        raise AcceptanceVerificationError(
            "MCP CreateTaskResult accepts an invalid required field or _meta"
        )


def _validate_public_envelope_contract(
    error_status: dict[str, object],
) -> None:
    catalog = _decision_mapping(
        error_status.get("public_envelope_catalog"),
        "public envelope catalog",
    )
    examples = _decision_mapping(
        error_status.get("public_envelope_examples"),
        "public envelope examples",
    )
    contract = _decision_mapping(
        error_status.get("public_envelope_catalog_contract"),
        "public envelope catalog contract",
    )
    _exact_keys(
        contract,
        {
            "dialect",
            "each_entry",
            "validation",
            "cross_field_invariants",
            "mutation_requirements",
        },
        "public envelope catalog contract",
    )
    if contract.get("dialect") != _JSON_SCHEMA_DIALECT or contract.get(
        "mutation_requirements"
    ) != list(_PUBLIC_SCHEMA_MUTATIONS):
        raise AcceptanceVerificationError(
            "public envelope schema proof requirements are incomplete"
        )
    if set(catalog) != set(examples):
        raise AcceptanceVerificationError(
            "public envelope schemas and representative examples differ"
        )
    for envelope_id, raw_schema in catalog.items():
        if not isinstance(raw_schema, dict):
            raise AcceptanceVerificationError(
                f"public envelope schema is invalid: {envelope_id}"
            )
        _validate_schema_examples(
            cast(dict[str, object], raw_schema),
            {envelope_id: examples[envelope_id]},
            f"public envelope {envelope_id}",
            exercise_mutations=True,
        )
    _validate_public_cross_field_mutations(
        catalog,
        examples,
        error_status.get("public_envelope_cross_field_mutations"),
    )


def _draft202012_validator() -> _JsonSchemaValidatorFactory:
    module = import_module("jsonschema")
    return cast(
        _JsonSchemaValidatorFactory,
        getattr(module, "Draft202012Validator"),
    )


def _validate_schema_examples(
    schema: dict[str, object],
    examples: dict[str, object],
    label: str,
    *,
    exercise_mutations: bool,
    allow_empty: bool = False,
) -> None:
    if schema.get("$schema") != _JSON_SCHEMA_DIALECT:
        raise AcceptanceVerificationError(
            f"{label} must declare the frozen JSON Schema dialect"
        )
    validator_factory = _draft202012_validator()
    try:
        validator_factory.check_schema(schema)
    except Exception as exc:
        raise AcceptanceVerificationError(
            f"{label} is not a valid JSON Schema: {exc}"
        ) from exc
    validator = validator_factory(schema)
    for name, example in examples.items():
        if not isinstance(example, dict) or (not example and not allow_empty):
            raise AcceptanceVerificationError(
                f"{label} example must be a populated object: {name}"
            )
        if not validator.is_valid(example):
            raise AcceptanceVerificationError(
                f"{label} representative example does not validate: {name}"
            )
        if exercise_mutations:
            _exercise_schema_mutations(
                schema,
                cast(dict[str, object], example),
                validator,
                f"{label} example {name}",
            )


def _exercise_schema_mutations(
    schema: dict[str, object],
    example: dict[str, object],
    validator: _JsonSchemaValidator,
    label: str,
) -> None:
    object_schema = _matching_object_schema(schema, example, label)
    required = object_schema.get("required")
    properties = object_schema.get("properties")
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(item, str) and item for item in required)
        or not isinstance(properties, dict)
        or object_schema.get("additionalProperties") is not False
    ):
        raise AcceptanceVerificationError(
            f"{label} lacks a closed required object contract"
        )
    missing = deepcopy(example)
    missing.pop(cast(str, required[0]), None)
    extra = deepcopy(example)
    extra["__unexpected_contract_field__"] = True
    constant_name = next(
        (
            name
            for name, raw_property in properties.items()
            if isinstance(name, str)
            and isinstance(raw_property, dict)
            and "const" in raw_property
        ),
        None,
    )
    if constant_name is None or constant_name not in example:
        raise AcceptanceVerificationError(
            f"{label} lacks a representative constant field"
        )
    raw_constant_schema = properties[constant_name]
    assert isinstance(raw_constant_schema, dict)
    wrong_constant = deepcopy(example)
    wrong_constant[constant_name] = _different_json_value(
        raw_constant_schema["const"]
    )
    mutations = {
        "missing_required_field": missing,
        "extra_field": extra,
        "wrong_const": wrong_constant,
        "wrong_type": [],
    }
    for mutation_name, mutation in mutations.items():
        if validator.is_valid(mutation):
            raise AcceptanceVerificationError(
                f"{label} accepts prohibited {mutation_name} mutation"
            )


def _matching_object_schema(
    schema: dict[str, object],
    example: dict[str, object],
    label: str,
) -> dict[str, object]:
    if schema.get("type") == "object":
        return schema
    raw_branches = schema.get("oneOf")
    if not isinstance(raw_branches, list):
        raise AcceptanceVerificationError(
            f"{label} must be an object schema or an object union"
        )
    matches: list[dict[str, object]] = []
    for raw_branch in raw_branches:
        if not isinstance(raw_branch, dict):
            continue
        branch = deepcopy(cast(dict[str, object], raw_branch))
        branch["$schema"] = _JSON_SCHEMA_DIALECT
        if "$defs" in schema:
            branch["$defs"] = schema["$defs"]
        if _draft202012_validator()(branch).is_valid(example):
            matches.append(cast(dict[str, object], raw_branch))
    if len(matches) != 1 or matches[0].get("type") != "object":
        raise AcceptanceVerificationError(
            f"{label} does not select exactly one object branch"
        )
    return matches[0]


def _different_json_value(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "-invalid"
    if type(value) in {int, float}:
        return cast(int | float, value) + 1
    if value is None:
        return "invalid"
    return {"invalid": True}


def _validate_public_cross_field_mutations(
    catalog: dict[str, object],
    examples: dict[str, object],
    raw_mutations: object,
) -> None:
    if not isinstance(raw_mutations, dict) or set(raw_mutations) != set(
        _PUBLIC_CROSS_FIELD_INVARIANTS
    ):
        raise AcceptanceVerificationError(
            "public cross-field mutation inventory is incomplete"
        )
    for envelope_id, expected_ids in _PUBLIC_CROSS_FIELD_INVARIANTS.items():
        raw_vectors = raw_mutations.get(envelope_id)
        if not isinstance(raw_vectors, list) or not raw_vectors:
            raise AcceptanceVerificationError(
                f"public cross-field mutations are empty: {envelope_id}"
            )
        schema = cast(dict[str, object], catalog[envelope_id])
        example = cast(dict[str, object], examples[envelope_id])
        validator = _draft202012_validator()(schema)
        observed_ids: list[str] = []
        for raw_vector in raw_vectors:
            if not isinstance(raw_vector, dict):
                raise AcceptanceVerificationError(
                    f"public cross-field mutation is invalid: {envelope_id}"
                )
            vector = cast(dict[str, object], raw_vector)
            invariant_id = _nonempty_string(
                vector.get("invariant_id"), "cross-field invariant ID"
            )
            observed_ids.append(invariant_id)
            equals_path: tuple[str, ...] | None = None
            if "expected" in vector:
                _exact_keys(
                    vector,
                    {"invariant_id", "path", "expected", "replacement"},
                    "public expected-value cross-field mutation",
                )
                comparison = vector.get("expected")
            else:
                _exact_keys(
                    vector,
                    {
                        "invariant_id",
                        "path",
                        "equals_path",
                        "replacement",
                    },
                    "public equality cross-field mutation",
                )
                equals_path = _json_path_parts(
                    vector.get("equals_path"), "cross-field equals_path"
                )
                comparison = _json_path(
                    example,
                    equals_path,
                )
            path = _json_path_parts(vector.get("path"), "cross-field path")
            if equals_path is not None and path == equals_path:
                raise AcceptanceVerificationError(
                    f"cross-field mutation compares one field: {invariant_id}"
                )
            original = _json_path(example, path)
            if original != comparison:
                raise AcceptanceVerificationError(
                    f"public cross-field invariant is false: {invariant_id}"
                )
            replacement = vector.get("replacement")
            if replacement == original:
                raise AcceptanceVerificationError(
                    f"public cross-field mutation is a no-op: {invariant_id}"
                )
            mutated = deepcopy(example)
            _replace_json_path(mutated, path, replacement)
            if not validator.is_valid(mutated):
                raise AcceptanceVerificationError(
                    "cross-field mutation must remain schema-valid:"
                    f" {invariant_id}"
                )
            mutated_value = _json_path(mutated, path)
            if mutated_value == comparison:
                raise AcceptanceVerificationError(
                    f"cross-field mutation did not violate: {invariant_id}"
                )
        if set(observed_ids) != expected_ids or len(observed_ids) != len(
            expected_ids
        ):
            raise AcceptanceVerificationError(
                f"public cross-field invariant IDs differ: {envelope_id}"
            )


def _json_path_parts(value: object, label: str) -> tuple[str, ...]:
    parts = _string_list(value, label)
    if not parts:
        raise AcceptanceVerificationError(f"{label} must not be empty")
    return parts


def _json_path(value: object, parts: Sequence[str]) -> object:
    current = value
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise AcceptanceVerificationError(
                f"cross-field JSON path does not exist: {'.'.join(parts)}"
            )
        current = current[part]
    return current


def _replace_json_path(
    value: dict[str, object],
    parts: Sequence[str],
    replacement: object,
) -> None:
    parent = _json_path(value, parts[:-1]) if len(parts) > 1 else value
    if not isinstance(parent, dict):
        raise AcceptanceVerificationError(
            f"cross-field JSON path parent is invalid: {'.'.join(parts)}"
        )
    parent[parts[-1]] = replacement


def _decision_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise AcceptanceVerificationError(
            f"contract decision {label} must be a populated object"
        )
    return cast(dict[str, object], value)


def _validate_no_bc(path: Path) -> None:
    payload = _strict_mapping(path, "no-BC removal inventory")
    _exact_keys(
        payload,
        {"schema_version", "feature", "inventory_sha256", "removals"},
        "no-BC removal inventory",
    )
    _header(payload, "no-BC removal inventory")
    raw = payload.get("removals")
    if not isinstance(raw, list) or len(raw) < 5:
        raise AcceptanceVerificationError(
            "no-BC removal inventory must contain all known replacement paths"
        )
    ids: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise AcceptanceVerificationError(
                "no-BC removal must be an object"
            )
        item = cast(dict[str, object], entry)
        _exact_keys(
            item,
            {
                "id",
                "current_path",
                "remove_by_phase",
                "replacement",
                "evidence",
            },
            "no-BC removal",
        )
        ids.append(_nonempty_string(item.get("id"), "no-BC id"))
        _nonempty_string(item.get("current_path"), "no-BC current_path")
        _phase(item.get("remove_by_phase"), "no-BC remove_by_phase")
        _nonempty_string(item.get("replacement"), "no-BC replacement")
        _nonempty_string(item.get("evidence"), "no-BC evidence")
    _unique(ids, "no-BC removal ID")
    if frozenset(ids) != _EXPECTED_NO_BC_IDS:
        raise AcceptanceVerificationError(
            "no-BC removal inventory does not contain the frozen paths"
        )
    _verify_digest(
        raw,
        payload.get("inventory_sha256"),
        _EXPECTED_NO_BC_SHA256,
        "no-BC removal inventory",
    )


def _validate_evidence(
    path: Path,
    manifest: AcceptanceManifest,
    root: Path,
) -> None:
    payload = _strict_mapping(path, "implementation evidence")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "recorded_at",
            "implementation_owner",
            "independent_reviewer",
            "review_history_sha256",
            "review_history_phase0_sha256",
            "review_history_phase1_sha256",
            "review_history_phase2_sha256",
            "review_history_prior_sha256",
            "review_history",
            "quality_history_sha256",
            "quality_history",
            "active_test_node_ids",
            "git",
            "baseline",
            "boundary",
            "pending_structural_inventory",
            "current_regression_classification",
            "inventory",
            "quality_gate",
            "typing_async_audit",
            "unresolved_risks",
        },
        "implementation evidence",
    )
    _header(payload, "implementation evidence")
    _nonempty_string(payload.get("recorded_at"), "evidence recorded_at")
    implementation_owner = _nonempty_string(
        payload.get("implementation_owner"), "implementation_owner"
    )
    independent_reviewer = _nonempty_string(
        payload.get("independent_reviewer"), "independent_reviewer"
    )
    if (
        implementation_owner != _EXPECTED_IMPLEMENTATION_OWNER
        or independent_reviewer != _EXPECTED_INDEPENDENT_REVIEWER
        or implementation_owner == independent_reviewer
    ):
        raise AcceptanceVerificationError(
            "implementation evidence ownership identities changed"
        )
    _validate_review_history(
        payload.get("review_history"),
        payload.get("review_history_sha256"),
        payload.get("review_history_phase0_sha256"),
        payload.get("review_history_phase1_sha256"),
        payload.get("review_history_phase2_sha256"),
        payload.get("review_history_prior_sha256"),
        manifest.current_phase,
        implementation_owner,
    )
    _validate_quality_history(
        payload.get("quality_history"),
        payload.get("quality_history_sha256"),
        manifest.current_phase,
    )
    active_test_node_ids = _string_list(
        payload.get("active_test_node_ids"), "active_test_node_ids"
    )
    _unique(active_test_node_ids, "active evidence pytest node ID")
    expected_active_test_node_ids = tuple(
        node.node_id for node in manifest.nodes if node.lifecycle == "active"
    )
    if active_test_node_ids != expected_active_test_node_ids:
        raise AcceptanceVerificationError(
            "implementation evidence active pytest nodes differ from the"
            " manifest"
        )

    git = _evidence_mapping(payload.get("git"), "git")
    _exact_keys(
        git,
        {
            "branch",
            "head",
            "head_subject",
            "production_changes_before_baseline",
            "preserved_untracked",
        },
        "implementation evidence git",
    )
    preserved_untracked = _string_list(
        git.get("preserved_untracked"), "preserved untracked paths"
    )
    if (
        git.get("branch") != "input"
        or git.get("head") != _EXPECTED_BASELINE_HEAD
        or git.get("head_subject") != _EXPECTED_BASELINE_SUBJECT
        or git.get("production_changes_before_baseline") != []
        or preserved_untracked != ("docs/examples/skills/code/",)
    ):
        raise AcceptanceVerificationError(
            "implementation evidence git baseline changed"
        )
    if (
        _git_output(
            root,
            "log",
            "-1",
            "--format=%s",
            _EXPECTED_BASELINE_HEAD,
        )
        != _EXPECTED_BASELINE_SUBJECT
        or _git_returncode(
            root,
            "merge-base",
            "--is-ancestor",
            _EXPECTED_BASELINE_HEAD,
            "HEAD",
        )
        != 0
    ):
        raise AcceptanceVerificationError(
            "live git baseline differs from implementation evidence"
        )

    baseline = _evidence_mapping(payload.get("baseline"), "baseline")
    _exact_keys(
        baseline,
        {
            "command",
            "exit_code",
            "collected",
            "passed",
            "skipped",
            "subtests_passed",
            "seconds",
        },
        "implementation evidence baseline",
    )
    baseline_counts = {
        name: _nonnegative_int(baseline.get(name), f"baseline {name}")
        for name in (
            "exit_code",
            "collected",
            "passed",
            "skipped",
            "subtests_passed",
        )
    }
    seconds = baseline.get("seconds")
    if (
        baseline.get("command") != "poetry run pytest --verbose -s"
        or baseline_counts["exit_code"] != 0
        or baseline_counts["collected"]
        != baseline_counts["passed"] + baseline_counts["skipped"]
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or seconds <= 0
    ):
        raise AcceptanceVerificationError(
            "implementation evidence baseline result is inconsistent"
        )

    boundary = _evidence_mapping(payload.get("boundary"), "boundary")
    _exact_keys(
        boundary,
        {
            "production_capability",
            "production_capability_history",
            "production_source_changes",
            "changed_paths",
        },
        "implementation evidence boundary",
    )
    changed_paths = _string_list(
        boundary.get("changed_paths"), "boundary changed_paths"
    )
    _unique(changed_paths, "boundary changed path")
    production_source_changes = _string_list(
        boundary.get("production_source_changes"),
        "production source changes",
    )
    _unique(production_source_changes, "production source change")
    capability_history = _production_capability_history(
        boundary.get("production_capability_history"),
        manifest.current_phase,
    )
    if (
        boundary.get("production_capability") != capability_history[-1]
        or frozenset(production_source_changes)
        != _EXPECTED_PRODUCTION_SOURCE_PATHS
        or len(production_source_changes)
        != len(_EXPECTED_PRODUCTION_SOURCE_PATHS)
        or frozenset(changed_paths) != _EXPECTED_BOUNDARY_PATHS
        or len(changed_paths) != len(_EXPECTED_BOUNDARY_PATHS)
    ):
        raise AcceptanceVerificationError(
            "implementation evidence production boundary is stale"
        )
    _validate_live_boundary(
        root,
        changed_paths,
        production_source_changes,
        preserved_untracked,
    )

    _validate_pending_structural_inventory(
        payload.get("pending_structural_inventory"), root
    )
    _validate_current_regression_classification(
        payload.get("current_regression_classification"), manifest, root
    )

    failure = _strict_mapping(path.with_name("failure_matrix.json"), "failure")
    raw_surfaces = failure.get("surfaces")
    raw_conditions = failure.get("conditions")
    raw_cells = failure.get("cells")
    if not all(
        isinstance(value, list)
        for value in (raw_surfaces, raw_conditions, raw_cells)
    ):
        raise AcceptanceVerificationError(
            "implementation evidence cannot derive failure counts"
        )
    surfaces = cast(list[object], raw_surfaces)
    conditions = cast(list[object], raw_conditions)
    cells = cast(list[object], raw_cells)
    applicable_cells = len(
        [
            cell
            for cell in cells
            if isinstance(cell, dict) and cell.get("applicable") is True
        ]
    )
    inventory = _evidence_mapping(payload.get("inventory"), "inventory")
    _exact_keys(
        inventory,
        {
            "behavior_requirements",
            "public_scenarios",
            "delivery_requirements",
            "active_acceptance_nodes",
            "active_pytest_instances",
            "planned_acceptance_nodes",
            "failure_conditions",
            "failure_surfaces",
            "failure_cells",
            "applicable_failure_cells",
            "non_applicable_failure_cells",
        },
        "implementation evidence inventory",
    )
    risks = payload.get("unresolved_risks")
    if not isinstance(risks, list) or not all(
        isinstance(item, str) and item for item in risks
    ):
        raise AcceptanceVerificationError(
            "implementation evidence unresolved_risks must be a string list"
        )
    active = len(
        [node for node in manifest.nodes if node.lifecycle == "active"]
    )
    planned = len(
        [node for node in manifest.nodes if node.lifecycle == "planned"]
    )
    expected_inventory = {
        "behavior_requirements": 107,
        "public_scenarios": 12,
        "delivery_requirements": 12,
        "active_acceptance_nodes": active,
        "active_pytest_instances": len(
            manifest.active_pytest_instances(manifest.current_phase)
        ),
        "planned_acceptance_nodes": planned,
        "failure_conditions": len(conditions),
        "failure_surfaces": len(surfaces),
        "failure_cells": len(cells),
        "applicable_failure_cells": applicable_cells,
        "non_applicable_failure_cells": len(cells) - applicable_cells,
    }
    if inventory != expected_inventory or len(cells) != len(surfaces) * len(
        conditions
    ):
        raise AcceptanceVerificationError(
            "implementation evidence inventory counts are stale"
        )

    type_manifest = _strict_mapping(
        path.with_name("type_contract_manifest.json"),
        "type manifest evidence",
    )
    raw_type_fixtures = type_manifest.get("fixtures")
    if not isinstance(raw_type_fixtures, list):
        raise AcceptanceVerificationError(
            "implementation evidence cannot derive type counts"
        )
    type_fixtures = [
        fixture for fixture in raw_type_fixtures if isinstance(fixture, dict)
    ]
    active_type_fixtures = len(
        [
            fixture
            for fixture in type_fixtures
            if fixture.get("lifecycle") == "active"
        ]
    )
    planned_negative_fixtures = len(
        [
            fixture
            for fixture in type_fixtures
            if fixture.get("lifecycle") == "planned"
            and fixture.get("kind") == "negative"
        ]
    )
    typing_audit = _evidence_mapping(
        payload.get("typing_async_audit"), "typing_async_audit"
    )
    _exact_keys(
        typing_audit,
        {
            "strict_type_fixture_count",
            "negative_type_fixtures_planned",
            "effect_interfaces_async",
            "blocking_waits",
            "timing_sleeps",
            "network_dependencies",
        },
        "implementation evidence typing audit",
    )
    if (
        typing_audit.get("strict_type_fixture_count") != active_type_fixtures
        or typing_audit.get("negative_type_fixtures_planned")
        != planned_negative_fixtures
        or typing_audit.get("effect_interfaces_async")
        != ["scripted provider", "async barrier", "local protocol peer"]
        or typing_audit.get("blocking_waits") != 0
        or typing_audit.get("timing_sleeps") != 0
        or typing_audit.get("network_dependencies") != 0
    ):
        raise AcceptanceVerificationError(
            "implementation evidence type or async audit is stale"
        )

    _validate_quality_gate_evidence(
        payload.get("quality_gate"),
        active_acceptance_nodes=active,
        active_pytest_instances=len(
            manifest.active_pytest_instances(manifest.current_phase)
        ),
        active_type_fixtures=active_type_fixtures,
        root=root,
        preserved_untracked=preserved_untracked,
        evidence_payload=payload,
    )
    _verify_digest(
        payload,
        _EXPECTED_EVIDENCE_SHA256,
        _EXPECTED_EVIDENCE_SHA256,
        "implementation evidence",
    )


def _production_capability_history(
    raw: object,
    current_phase: int,
) -> tuple[str, ...]:
    """Return the exact production-capability state through this phase."""
    if not isinstance(raw, list) or len(raw) != current_phase + 1:
        raise AcceptanceVerificationError(
            "production capability history must contain every implemented"
            " phase"
        )
    states: list[str] = []
    for expected_phase, value in enumerate(raw):
        if not isinstance(value, dict):
            raise AcceptanceVerificationError(
                "production capability history entries must be objects"
            )
        entry = cast(dict[str, object], value)
        _exact_keys(
            entry,
            {"phase", "state"},
            "production capability history entry",
        )
        phase = _phase(entry.get("phase"), "production capability phase")
        if phase != expected_phase:
            raise AcceptanceVerificationError(
                "production capability phases must be contiguous"
            )
        state = _nonempty_string(
            entry.get("state"), "production capability state"
        )
        expected_state = (
            "absent"
            if phase == 0
            else "active" if phase == _MAX_PHASE else "dormant_unadvertised"
        )
        if state != expected_state:
            raise AcceptanceVerificationError(
                "production capability activated outside the atomic"
                f" boundary: phase={phase}, expected={expected_state},"
                f" observed={state}"
            )
        states.append(state)
    return tuple(states)


def _validate_live_boundary(
    root: Path,
    declared_paths: Sequence[str],
    declared_source_paths: Sequence[str],
    preserved_untracked: Sequence[str],
) -> None:
    tracked = set(
        _git_lines(
            root,
            "diff",
            "--name-only",
            f"{_EXPECTED_BASELINE_HEAD}..HEAD",
            "--",
        )
    )
    tracked.update(_git_lines(root, "diff", "--name-only", "HEAD", "--"))
    untracked = set(
        _git_lines(root, "ls-files", "--others", "--exclude-standard", "--")
    )
    for prefix in preserved_untracked:
        if not prefix.endswith("/"):
            raise AcceptanceVerificationError(
                f"preserved untracked path must be a directory: {prefix}"
            )
        if any(path.startswith(prefix) for path in tracked):
            raise AcceptanceVerificationError(
                f"preserved untracked path became tracked: {prefix}"
            )
    live_files = tracked | {
        path
        for path in untracked
        if not any(path.startswith(prefix) for prefix in preserved_untracked)
    }
    directory_claims = tuple(
        path for path in declared_paths if path.endswith("/")
    )
    normalized = {
        next(
            (prefix for prefix in directory_claims if path.startswith(prefix)),
            path,
        )
        for path in live_files
    }
    if normalized != set(declared_paths):
        raise AcceptanceVerificationError(
            "live changed paths differ from implementation evidence:"
            f" declared={sorted(declared_paths)}, live={sorted(normalized)}"
        )
    source_directory_claims = tuple(
        path for path in declared_source_paths if path.endswith("/")
    )
    source_changes = {
        next(
            (
                prefix
                for prefix in source_directory_claims
                if path.startswith(prefix)
            ),
            path,
        )
        for path in live_files
        if path == "src" or path.startswith("src/")
    }
    if source_changes != set(declared_source_paths):
        raise AcceptanceVerificationError(
            "live production source changes differ from implementation"
            f" evidence: declared={sorted(declared_source_paths)},"
            f" live={sorted(source_changes)}"
        )


def _validate_pending_structural_inventory(raw: object, root: Path) -> None:
    """Validate pending source inventory without relying on key order."""
    inventory = _evidence_mapping(raw, "pending structural inventory")
    _exact_keys(
        inventory,
        {
            "source_inventory_sha256",
            "source_file_count",
            "statement_count",
            "excluded_line_count",
        },
        "pending structural inventory",
    )
    observed = (
        _sha256_string(
            inventory.get("source_inventory_sha256"),
            "pending source inventory SHA-256",
        ),
        _nonnegative_int(
            inventory.get("source_file_count"),
            "pending source file count",
        ),
        _nonnegative_int(
            inventory.get("statement_count"),
            "pending statement count",
        ),
        _nonnegative_int(
            inventory.get("excluded_line_count"),
            "pending excluded line count",
        ),
    )
    if (
        observed != _EXPECTED_PENDING_SOURCE_INVENTORY
        or _source_statement_inventory(root)
        != _EXPECTED_PENDING_SOURCE_INVENTORY
    ):
        raise AcceptanceVerificationError(
            "pending structural inventory differs from the live source tree"
        )


def _test_definition_digests(
    source: str,
    relative: str,
) -> dict[str, str]:
    """Return AST digests for every pytest definition in one source."""
    tree = _parse_test_source(source, relative)
    definition_digests: dict[str, list[str]] = {}

    def visit(body: Sequence[AST], parents: tuple[str, ...]) -> None:
        for statement in body:
            if isinstance(statement, ClassDef):
                visit(statement.body, (*parents, statement.name))
                continue
            if isinstance(statement, (FunctionDef, AsyncFunctionDef)):
                if not statement.name.startswith("test_"):
                    continue
                node_id = "::".join((relative, *parents, statement.name))
                definition_digests.setdefault(node_id, []).append(
                    sha256(
                        dump(statement, include_attributes=False).encode(
                            "utf-8"
                        )
                    ).hexdigest()
                )
                continue
            for nested_body in _structural_statement_bodies(statement):
                visit(nested_body, parents)

    visit(tree.body, ())
    definitions: dict[str, str] = {}
    for node_id, digests in definition_digests.items():
        if len(digests) > 1:
            if _EXPECTED_CURRENT_DUPLICATE_TEST_DEFINITIONS.get(
                node_id,
                (0, "", ""),
            )[0] != len(digests):
                raise AcceptanceVerificationError(
                    f"duplicate changed test definition: {node_id}"
                )
            definitions[node_id] = sha256(
                dumps(
                    digests,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            continue
        definitions[node_id] = digests[0]
    return definitions


def _test_support_surface_digest(source: str, relative: str) -> str:
    """Return one normalized digest for non-test collection support."""
    tree = deepcopy(_parse_test_source(source, relative))

    def replace_tests(
        body: Sequence[AstStatement],
    ) -> list[AstStatement]:
        normalized: list[AstStatement] = []
        has_tests = False
        for statement in body:
            if isinstance(statement, ClassDef):
                statement.body = replace_tests(statement.body)
                normalized.append(statement)
                continue
            if isinstance(
                statement, (FunctionDef, AsyncFunctionDef)
            ) and statement.name.startswith("test_"):
                has_tests = True
                continue
            _replace_structural_test_bodies(statement, replace_tests)
            normalized.append(statement)
        if has_tests:
            normalized.append(
                Expr(value=Constant(value="current-test-placeholder"))
            )
        return normalized

    tree.body = replace_tests(tree.body)
    return sha256(
        dump(tree, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _structural_statement_bodies(
    statement: AST,
) -> tuple[Sequence[AstStatement], ...]:
    """Return structural child bodies without entering function scopes."""
    if isinstance(statement, If):
        return (statement.body, statement.orelse)
    if isinstance(statement, (For, AsyncFor, While)):
        return (statement.body, statement.orelse)
    if isinstance(statement, (With, AsyncWith)):
        return (statement.body,)
    if isinstance(statement, (Try, TryStar)):
        return (
            statement.body,
            *(handler.body for handler in statement.handlers),
            statement.orelse,
            statement.finalbody,
        )
    if isinstance(statement, Match):
        return tuple(case.body for case in statement.cases)
    return ()


def _replace_structural_test_bodies(
    statement: AstStatement,
    replace_tests: Callable[
        [Sequence[AstStatement]],
        list[AstStatement],
    ],
) -> None:
    """Replace tests recursively in structural child bodies."""
    if isinstance(statement, If):
        statement.body = replace_tests(statement.body)
        statement.orelse = replace_tests(statement.orelse)
    elif isinstance(statement, (For, AsyncFor, While)):
        statement.body = replace_tests(statement.body)
        statement.orelse = replace_tests(statement.orelse)
    elif isinstance(statement, (With, AsyncWith)):
        statement.body = replace_tests(statement.body)
    elif isinstance(statement, (Try, TryStar)):
        statement.body = replace_tests(statement.body)
        statement.orelse = replace_tests(statement.orelse)
        for handler in statement.handlers:
            handler.body = replace_tests(handler.body)
        statement.finalbody = replace_tests(statement.finalbody)
    elif isinstance(statement, Match):
        for case in statement.cases:
            case.body = replace_tests(case.body)


def _parse_test_source(source: str, relative: str) -> Module:
    """Parse one changed test file or fail with stable context."""
    try:
        return parse(source, filename=relative)
    except SyntaxError as exc:
        raise AcceptanceVerificationError(
            f"cannot inspect changed test definitions in {relative}: {exc}"
        ) from exc


def _current_baseline_source(root: Path, relative: str) -> str | None:
    """Return one test source from the frozen current baseline."""
    revision_path = f"{_EXPECTED_CURRENT_BASELINE_HEAD}:{relative}"
    if _git_returncode(root, "cat-file", "-e", revision_path) != 0:
        return None
    completed = run(
        ("git", "show", revision_path),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AcceptanceVerificationError(
            "cannot read current baseline test source:"
            f" {relative}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _current_changed_test_definitions(
    root: Path,
) -> tuple[dict[str, str], dict[str, str], frozenset[str]]:
    """Return baseline/current definitions for every changed test file."""
    if (
        _git_returncode(
            root,
            "merge-base",
            "--is-ancestor",
            _EXPECTED_CURRENT_BASELINE_HEAD,
            "HEAD",
        )
        != 0
    ):
        raise AcceptanceVerificationError(
            "current test classification baseline is not an ancestor"
        )
    changed = set(
        _git_lines(
            root,
            "diff",
            "--name-only",
            f"{_EXPECTED_CURRENT_BASELINE_HEAD}..HEAD",
            "--",
            "tests",
        )
    )
    changed.update(
        _git_lines(root, "diff", "--name-only", "HEAD", "--", "tests")
    )
    changed.update(
        _git_lines(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "tests",
        )
    )
    paths = frozenset(
        relative for relative in changed if _is_default_pytest_path(relative)
    )
    baseline: dict[str, str] = {}
    current: dict[str, str] = {}
    live_paths: list[str] = []
    for relative in sorted(paths):
        source = _current_baseline_source(root, relative)
        baseline_file: dict[str, str] = {}
        if source is not None:
            baseline_file = _test_definition_digests(source, relative)
            baseline.update(baseline_file)
        path = root / relative
        if path.is_file():
            live_paths.append(relative)
            try:
                live_source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise AcceptanceVerificationError(
                    f"cannot read changed test source {relative}: {exc}"
                ) from exc
            current_file = _test_definition_digests(live_source, relative)
            current.update(current_file)
            if not _common_test_definition_order_is_preserved(
                baseline_file,
                current_file,
            ):
                raise AcceptanceVerificationError(
                    "changed test definitions reordered relative to the"
                    f" current baseline: {relative}"
                )
        elif source is not None:
            raise AcceptanceVerificationError(
                f"current baseline test file was deleted: {relative}"
            )
    _validate_frozen_duplicate_test_definitions(baseline, current)
    if live_paths:
        payload = _run_probe(
            _COLLECT_DRIVER,
            _COLLECT_SENTINEL,
            tuple(live_paths),
            root,
        )
        collected_bases: set[str] = set()
        for node_id in _collection_node_ids(
            payload,
            reject_disallowed_markers=False,
        ):
            matches = tuple(
                base
                for base in current
                if node_id == base or node_id.startswith(f"{base}[")
            )
            if len(matches) != 1:
                raise AcceptanceVerificationError(
                    "collected changed test does not map to one static"
                    f" definition: {node_id}"
                )
            collected_bases.add(matches[0])
        if collected_bases != current.keys():
            raise AcceptanceVerificationError(
                "collected changed test definitions differ from static"
                " definitions:"
                f" missing={sorted(current.keys() - collected_bases)},"
                f" extra={sorted(collected_bases - current.keys())}"
            )
    return baseline, current, paths


def _is_default_pytest_path(relative: str) -> bool:
    """Return whether a path matches either default pytest file pattern."""
    name = PurePosixPath(relative).name
    return name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py")
    )


def _common_test_definition_order_is_preserved(
    baseline: dict[str, str],
    current: dict[str, str],
) -> bool:
    """Return whether common pytest node IDs retain their source order."""
    common = baseline.keys() & current.keys()
    baseline_order = tuple(
        node_id for node_id in baseline if node_id in common
    )
    current_order = tuple(node_id for node_id in current if node_id in common)
    return baseline_order == current_order


def _validate_frozen_duplicate_test_definitions(
    baseline: dict[str, str],
    current: dict[str, str],
) -> None:
    """Require the exact ordered legacy duplicate definitions."""
    for node_id, (
        _,
        baseline_digest,
        current_digest,
    ) in _EXPECTED_CURRENT_DUPLICATE_TEST_DEFINITIONS.items():
        if (
            baseline.get(node_id) != baseline_digest
            or current.get(node_id) != current_digest
        ):
            raise AcceptanceVerificationError(
                "frozen duplicate test definition changed or disappeared:"
                f" {node_id}"
            )


def _current_changed_test_support_surfaces(
    root: Path,
    paths: frozenset[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return normalized residual AST digests for every changed test file."""
    baseline: dict[str, str] = {}
    current: dict[str, str] = {}
    for relative in sorted(paths):
        source = _current_baseline_source(root, relative)
        if source is None:
            baseline[relative] = _ABSENT_TEST_SUPPORT_SHA256
        else:
            baseline[relative] = _test_support_surface_digest(
                source,
                relative,
            )
        path = root / relative
        if not path.is_file():
            raise AcceptanceVerificationError(
                f"current baseline test file was deleted: {relative}"
            )
        try:
            live_source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise AcceptanceVerificationError(
                f"cannot read changed test source {relative}: {exc}"
            ) from exc
        current[relative] = _test_support_surface_digest(
            live_source,
            relative,
        )
    return baseline, current


def _validate_current_regression_classification(
    raw: object,
    manifest: AcceptanceManifest,
    root: Path,
) -> None:
    """Validate every changed current test definition exactly."""
    classification = _evidence_mapping(
        raw, "current regression classification"
    )
    _exact_keys(
        classification,
        {
            "catalog_sha256",
            "mechanical_nodes",
            "reviewed_nonsemantic_nodes",
            "support_surfaces",
        },
        "current regression classification",
    )
    raw_mechanical = classification.get("mechanical_nodes")
    raw_nonsemantic = classification.get("reviewed_nonsemantic_nodes")
    raw_support = classification.get("support_surfaces")
    if (
        not isinstance(raw_mechanical, list)
        or not isinstance(raw_nonsemantic, list)
        or not isinstance(raw_support, list)
    ):
        raise AcceptanceVerificationError(
            "current regression classification buckets must be lists"
        )
    _verify_digest(
        {
            "mechanical_nodes": raw_mechanical,
            "reviewed_nonsemantic_nodes": raw_nonsemantic,
            "support_surfaces": raw_support,
        },
        classification.get("catalog_sha256"),
        _EXPECTED_CURRENT_REGRESSION_SHA256,
        "current regression classification",
    )
    if len(raw_mechanical) != _EXPECTED_CURRENT_REGRESSION_NODE_COUNT:
        raise AcceptanceVerificationError(
            "current mechanical regression node inventory changed"
        )
    if len(raw_support) != _EXPECTED_CURRENT_SUPPORT_SURFACE_COUNT:
        raise AcceptanceVerificationError(
            "current support-surface inventory changed"
        )
    baseline, current, changed_paths = _current_changed_test_definitions(root)
    baseline_support, current_support = _current_changed_test_support_surfaces(
        root, changed_paths
    )
    changed_support_paths = frozenset(
        relative
        for relative in baseline_support.keys() & current_support.keys()
        if baseline_support[relative] != current_support[relative]
    )
    unchanged_support_paths = frozenset(baseline_support) - (
        changed_support_paths
    )
    if (
        baseline_support.keys() != current_support.keys()
        or len(baseline_support) != _EXPECTED_CURRENT_TEST_FILE_COUNT
        or changed_support_paths != _EXPECTED_CURRENT_CHANGED_SUPPORT_PATHS
        or len(unchanged_support_paths)
        != _EXPECTED_CURRENT_UNCHANGED_SUPPORT_SURFACE_COUNT
    ):
        raise AcceptanceVerificationError(
            "current changed test support-surface inventory differs from"
            " the frozen baseline"
        )
    support_paths: list[str] = []
    for value in raw_support:
        entry = _evidence_mapping(value, "current support surface")
        _exact_keys(
            entry,
            {
                "path",
                "baseline_ast_sha256",
                "current_ast_sha256",
                "change_kind",
                "reviewed_by",
                "evidence",
            },
            "current support surface",
        )
        relative = _nonempty_string(
            entry.get("path"),
            "current support-surface path",
        )
        support_paths.append(relative)
        baseline_digest = _sha256_string(
            entry.get("baseline_ast_sha256"),
            "support-surface baseline AST SHA-256",
        )
        current_digest = _sha256_string(
            entry.get("current_ast_sha256"),
            "support-surface current AST SHA-256",
        )
        if (
            entry.get("change_kind")
            not in {
                "gate_support",
                "mechanical_fixture_migration",
                "semantic_support",
            }
            or entry.get("reviewed_by") != _EXPECTED_IMPLEMENTATION_OWNER
            or len(
                _nonempty_string(
                    entry.get("evidence"),
                    "support-surface evidence",
                )
            )
            < 20
            or baseline_support.get(relative) != baseline_digest
            or current_support.get(relative) != current_digest
            or baseline_digest == current_digest
            or relative not in changed_paths
        ):
            raise AcceptanceVerificationError(
                "current support surface differs from its exact reviewed"
                f" file: {relative}"
            )
    _unique(support_paths, "current support-surface path")
    if frozenset(support_paths) != changed_support_paths:
        raise AcceptanceVerificationError(
            "changed test support surfaces lack exact classification:"
            f" missing={sorted(changed_support_paths - set(support_paths))},"
            f" extra={sorted(set(support_paths) - changed_support_paths)}"
        )
    manifest_node_ids = frozenset(node.node_id for node in manifest.nodes)
    active_manifest_ids = frozenset(
        node.node_id for node in manifest.nodes if node.lifecycle == "active"
    )
    mechanical_ids: list[str] = []
    for value in raw_mechanical:
        entry = _evidence_mapping(value, "current mechanical regression")
        _exact_keys(
            entry,
            {
                "node_id",
                "baseline_ast_sha256",
                "current_ast_sha256",
                "disposition",
                "evidence",
            },
            "current mechanical regression",
        )
        node_id = _node_id(entry.get("node_id"))
        mechanical_ids.append(node_id)
        baseline_digest = _sha256_string(
            entry.get("baseline_ast_sha256"),
            "mechanical baseline AST SHA-256",
        )
        current_digest = _sha256_string(
            entry.get("current_ast_sha256"),
            "mechanical current AST SHA-256",
        )
        if (
            entry.get("disposition")
            != "mechanical_fixture_or_assertion_migration"
            or len(
                _nonempty_string(
                    entry.get("evidence"), "mechanical regression evidence"
                )
            )
            < 20
            or baseline.get(node_id) != baseline_digest
            or current.get(node_id) != current_digest
            or baseline_digest == current_digest
            or node_id in manifest_node_ids
            or node_id.split("::", 1)[0] not in changed_paths
        ):
            raise AcceptanceVerificationError(
                "current mechanical regression differs from its exact"
                f" reviewed node: {node_id}"
            )
    _unique(mechanical_ids, "current mechanical regression node ID")

    reviewed_ids: list[str] = []
    for value in raw_nonsemantic:
        entry = _evidence_mapping(value, "reviewed support test")
        _exact_keys(
            entry,
            {
                "node_id",
                "baseline_ast_sha256",
                "current_ast_sha256",
                "disposition",
                "reviewed_by",
                "evidence",
            },
            "reviewed support test",
        )
        node_id = _node_id(entry.get("node_id"))
        reviewed_ids.append(node_id)
        in_baseline = node_id in baseline
        in_current = node_id in current
        baseline_digest = _sha256_string(
            entry.get("baseline_ast_sha256"),
            "reviewed support baseline AST SHA-256",
        )
        current_digest = _sha256_string(
            entry.get("current_ast_sha256"),
            "reviewed support current AST SHA-256",
        )
        if (
            entry.get("disposition")
            not in {
                "gate_support",
                "reviewed_nonsemantic",
                "semantic_support",
            }
            or entry.get("reviewed_by") != _EXPECTED_IMPLEMENTATION_OWNER
            or len(
                _nonempty_string(
                    entry.get("evidence"), "support review evidence"
                )
            )
            < 20
            or node_id in manifest_node_ids
            or (
                baseline.get(node_id, _ABSENT_TEST_DEFINITION_SHA256)
                != baseline_digest
            )
            or (
                current.get(node_id, _ABSENT_TEST_DEFINITION_SHA256)
                != current_digest
            )
            or in_baseline == in_current
            and (not in_baseline or baseline[node_id] == current[node_id])
            or node_id.split("::", 1)[0] not in changed_paths
        ):
            raise AcceptanceVerificationError(
                f"invalid reviewed support test disposition: {node_id}"
            )
    _unique(reviewed_ids, "reviewed support test node ID")
    if set(mechanical_ids) & set(reviewed_ids):
        raise AcceptanceVerificationError(
            "current test classifications overlap"
        )
    baseline_ids = frozenset(baseline)
    current_ids = frozenset(current)
    new_definitions = current_ids - baseline_ids
    removed_definitions = baseline_ids - current_ids
    modified_definitions = frozenset(
        node_id
        for node_id in baseline_ids & current_ids
        if baseline[node_id] != current[node_id]
    )
    semantic_new = frozenset(
        node.node_id
        for node in manifest.nodes
        if node.lifecycle == "active" and node.active_from_phase == 4
    )
    reviewed_new = frozenset(
        node_id for node_id in reviewed_ids if node_id in new_definitions
    )
    reviewed_modified = frozenset(
        node_id for node_id in reviewed_ids if node_id in modified_definitions
    )
    reviewed_removed = frozenset(
        node_id for node_id in reviewed_ids if node_id in removed_definitions
    )
    expected_new = semantic_new | reviewed_new
    if new_definitions != expected_new:
        raise AcceptanceVerificationError(
            "new test definitions lack semantic acceptance or explicit"
            " nonsemantic review:"
            f" missing={sorted(expected_new - new_definitions)},"
            f" unclassified={sorted(new_definitions - expected_new)}"
        )

    semantic_removed: set[str] = set()
    replacement_targets: set[str] = set()
    for (
        removed_id,
        removed_digest,
        replacement_id,
        replacement_digest,
    ) in _EXPECTED_CURRENT_SEMANTIC_REPLACEMENTS:
        semantic_removed.add(removed_id)
        replacement_targets.add(replacement_id)
        if (
            baseline.get(removed_id) != removed_digest
            or removed_id in current
            or replacement_id in baseline
            or current.get(replacement_id) != replacement_digest
            or replacement_id not in semantic_new
        ):
            raise AcceptanceVerificationError(
                "current semantic replacement differs from its exact"
                f" reviewed transition: {removed_id} -> {replacement_id}"
            )
    _unique(
        (
            *semantic_removed,
            *replacement_targets,
        ),
        "current semantic replacement node ID",
    )
    expected_removed = frozenset(semantic_removed) | reviewed_removed
    if removed_definitions != expected_removed:
        raise AcceptanceVerificationError(
            "removed test definitions lack exact semantic replacement or"
            " explicit nonsemantic review:"
            f" missing={sorted(expected_removed - removed_definitions)},"
            f" unclassified={sorted(removed_definitions - expected_removed)}"
        )

    active_legacy_ids = (
        modified_definitions & active_manifest_ids
    ) | _EXPECTED_CURRENT_ACTIVE_LEGACY_GATE_NODES
    if (
        not _EXPECTED_CURRENT_ACTIVE_LEGACY_GATE_NODES <= modified_definitions
        or _EXPECTED_CURRENT_ACTIVE_LEGACY_GATE_NODES & manifest_node_ids
    ):
        raise AcceptanceVerificationError(
            "current active legacy gate definition inventory changed"
        )
    active_legacy_entries = [
        {
            "node_id": node_id,
            "baseline_ast_sha256": baseline[node_id],
            "current_ast_sha256": current[node_id],
        }
        for node_id in sorted(active_legacy_ids)
    ]
    active_legacy_digest = sha256(
        dumps(
            active_legacy_entries,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if (
        len(active_legacy_entries)
        != _EXPECTED_CURRENT_ACTIVE_LEGACY_NODE_COUNT
        or active_legacy_digest != _EXPECTED_CURRENT_ACTIVE_LEGACY_SHA256
    ):
        raise AcceptanceVerificationError(
            "current active legacy acceptance definition inventory changed"
        )
    expected_modified = (
        active_legacy_ids | frozenset(mechanical_ids) | reviewed_modified
    )
    if modified_definitions != expected_modified:
        raise AcceptanceVerificationError(
            "legacy test-definition changes lack exact classification:"
            f" unclassified={sorted(modified_definitions - expected_modified)}"
        )


def _git_lines(root: Path, *arguments: str) -> tuple[str, ...]:
    completed = run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AcceptanceVerificationError(
            "cannot verify live git evidence: "
            f"git {' '.join(arguments)}: {completed.stderr.strip()}"
        )
    return tuple(line for line in completed.stdout.splitlines() if line)


def _git_returncode(root: Path, *arguments: str) -> int:
    completed = run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    return completed.returncode


def _git_output(root: Path, *arguments: str) -> str:
    lines = _git_lines(root, *arguments)
    if len(lines) != 1:
        raise AcceptanceVerificationError(
            f"live git evidence returned {len(lines)} lines:"
            f" {' '.join(arguments)}"
        )
    return lines[0]


def _validate_review_history(
    raw: object,
    raw_digest: object,
    raw_phase0_digest: object,
    raw_phase1_digest: object,
    raw_phase2_digest: object,
    raw_prior_digest: object,
    current_phase: int,
    implementation_owner: str,
) -> None:
    if not isinstance(raw, list) or not raw:
        raise AcceptanceVerificationError(
            "implementation evidence review history must be non-empty"
        )
    _verify_digest(
        raw[:1],
        raw_phase0_digest,
        _EXPECTED_PHASE0_REVIEW_SHA256,
        "phase-0 review prefix",
    )
    if len(raw) < 5:
        raise AcceptanceVerificationError(
            "review history lost its phase-1 prefix"
        )
    _verify_digest(
        raw[:5],
        raw_phase1_digest,
        _EXPECTED_PHASE1_REVIEW_SHA256,
        "phase-1 review prefix",
    )
    if len(raw) < 7:
        raise AcceptanceVerificationError(
            "review history lost its phase-2 pending prefix"
        )
    _verify_digest(
        raw[:7],
        _EXPECTED_PHASE2_PENDING_REVIEW_SHA256,
        _EXPECTED_PHASE2_PENDING_REVIEW_SHA256,
        "phase-2 pending review prefix",
    )
    if len(raw) < 9:
        raise AcceptanceVerificationError(
            "review history lost its phase-2 prefix"
        )
    _verify_digest(
        raw[:9],
        raw_phase2_digest,
        _EXPECTED_PHASE2_REVIEW_SHA256,
        "phase-2 review prefix",
    )
    if len(raw) < 16:
        raise AcceptanceVerificationError(
            "review history lost its historical prefix"
        )
    _verify_digest(
        raw[:16],
        raw_prior_digest,
        _EXPECTED_PRIOR_REVIEW_SHA256,
        "historical review prefix",
    )
    latest_status: dict[tuple[int, str], str] = {}
    recorded_times: list[str] = []
    for expected_sequence, value in enumerate(raw):
        record = _evidence_mapping(value, "review history record")
        _exact_keys(
            record,
            {
                "sequence",
                "phase",
                "role",
                "reviewer",
                "status",
                "recorded_at",
                "evidence",
            },
            "review history record",
        )
        if record.get("sequence") != expected_sequence:
            raise AcceptanceVerificationError(
                "review history sequences must be contiguous and append-only"
            )
        phase = _phase(record.get("phase"), "review history phase")
        if phase > current_phase:
            raise AcceptanceVerificationError(
                "review history phase is not implemented"
            )
        role = _nonempty_string(record.get("role"), "review role")
        reviewer = _nonempty_string(record.get("reviewer"), "reviewer")
        status = _nonempty_string(record.get("status"), "review status")
        recorded_at = _nonempty_string(
            record.get("recorded_at"),
            "review recorded_at",
        )
        evidence = _nonempty_string(
            record.get("evidence"),
            "review evidence",
        )
        if len(evidence) < 20:
            raise AcceptanceVerificationError(
                "review history evidence must be concrete"
            )
        if reviewer == implementation_owner:
            raise AcceptanceVerificationError(
                "implementation owner cannot review its own evidence"
            )
        if expected_sequence >= len(_EXPECTED_REVIEW_OCCURRENCES):
            raise AcceptanceVerificationError(
                "review history contains an unexpected occurrence"
            )
        expected_occurrence = _EXPECTED_REVIEW_OCCURRENCES[expected_sequence]
        if (phase, role, reviewer, status) != expected_occurrence:
            raise AcceptanceVerificationError(
                "review history occurrence identity or status changed"
            )
        if status not in {"pending", "approved", "rejected"}:
            raise AcceptanceVerificationError(
                f"invalid review status: {status}"
            )
        key = (phase, role)
        previous = latest_status.get(key)
        if previous is None:
            if phase == 0 and status != "approved":
                raise AcceptanceVerificationError(
                    "phase-0 review must preserve its approval"
                )
            direct_current_approval = (
                phase == current_phase
                and role in {"semantic", "gate"}
                and status == "approved"
            )
            preserved_prior_semantic_approval = (
                phase == 3
                and role == "semantic"
                and reviewer == "/root/acceptance_review"
                and status == "approved"
            )
            if (
                phase > 0
                and status != "pending"
                and not direct_current_approval
                and not preserved_prior_semantic_approval
            ):
                raise AcceptanceVerificationError(
                    "new review roles must begin pending"
                )
        elif previous != "pending" or status not in {"approved", "rejected"}:
            raise AcceptanceVerificationError(
                "review history rewrites or extends a terminal decision"
            )
        latest_status[key] = status
        recorded_times.append(recorded_at)
    if recorded_times != sorted(recorded_times):
        raise AcceptanceVerificationError(
            "review history timestamps must be monotonic"
        )
    if latest_status.get((0, "baseline")) != "approved":
        raise AcceptanceVerificationError("phase-0 review approval is missing")
    expected_current_statuses = {
        "semantic": _EXPECTED_CURRENT_SEMANTIC_REVIEW_STATUS,
        "gate": _EXPECTED_CURRENT_GATE_REVIEW_STATUS,
    }
    for role, expected_status in expected_current_statuses.items():
        if latest_status.get((current_phase, role)) != expected_status:
            raise AcceptanceVerificationError(
                f"current {role} review status is not {expected_status}"
            )
    _verify_digest(
        raw,
        raw_digest,
        _EXPECTED_REVIEW_HISTORY_SHA256,
        "review history",
    )


def _validate_quality_history(
    raw: object,
    raw_digest: object,
    current_phase: int,
) -> None:
    """Validate append-only digests for prior completed quality records."""
    if not isinstance(raw, list) or len(raw) != max(0, current_phase - 1):
        raise AcceptanceVerificationError(
            "quality history must preserve every prior completed gate"
        )
    for expected_phase, value in enumerate(raw, start=1):
        record = _evidence_mapping(value, "quality history record")
        _exact_keys(
            record,
            {
                "phase",
                "state",
                "quality_gate_sha256",
                "evidence_sha256",
            },
            "quality history record",
        )
        phase = _phase(record.get("phase"), "quality history phase")
        if phase != expected_phase or record.get("state") != "complete":
            raise AcceptanceVerificationError(
                "quality history phases must be contiguous completed gates"
            )
        quality_digest = _sha256_string(
            record.get("quality_gate_sha256"),
            "historical quality gate SHA-256",
        )
        evidence_digest = _sha256_string(
            record.get("evidence_sha256"),
            "historical evidence SHA-256",
        )
        if quality_digest == evidence_digest:
            raise AcceptanceVerificationError(
                "quality history cannot reuse its evidence digest"
            )
        expected_historical_digests = {
            1: (
                _EXPECTED_PHASE1_QUALITY_SHA256,
                _EXPECTED_PHASE1_EVIDENCE_SHA256,
            ),
            2: (
                _EXPECTED_PHASE2_QUALITY_SHA256,
                _EXPECTED_PHASE2_EVIDENCE_SHA256,
            ),
            3: (
                _EXPECTED_PRIOR_QUALITY_SHA256,
                _EXPECTED_PRIOR_EVIDENCE_SHA256,
            ),
        }
        expected_digests = expected_historical_digests.get(phase)
        if (
            expected_digests is not None
            and (
                quality_digest,
                evidence_digest,
            )
            != expected_digests
        ):
            raise AcceptanceVerificationError(
                f"quality history lost its phase-{phase} record"
            )
    _verify_digest(
        raw,
        raw_digest,
        _EXPECTED_QUALITY_HISTORY_SHA256,
        "quality history",
    )


def _evidence_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise AcceptanceVerificationError(
            f"implementation evidence {label} must be a populated object"
        )
    return cast(dict[str, object], value)


def _validate_quality_gate_evidence(
    raw: object,
    *,
    active_acceptance_nodes: int,
    active_pytest_instances: int,
    active_type_fixtures: int,
    root: Path,
    preserved_untracked: tuple[str, ...],
    evidence_payload: dict[str, object],
) -> None:
    quality_gate = _evidence_mapping(raw, "quality gate")
    _exact_keys(
        quality_gate,
        {
            "state",
            "required_commands",
            "state_details",
            "results",
            "tree_binding",
            "coverage_binding",
        },
        "quality gate",
    )
    state = _nonempty_string(quality_gate.get("state"), "quality state")
    if state not in {"pending", "complete"}:
        raise AcceptanceVerificationError(
            f"invalid quality gate evidence state: {state}"
        )
    required_commands = _string_list(
        quality_gate.get("required_commands"),
        "required quality commands",
    )
    _unique(required_commands, "required quality command")
    if len(required_commands) != 8:
        raise AcceptanceVerificationError(
            "implementation evidence must require eight exact gate commands"
        )
    if required_commands[:7] != _EXPECTED_ORDERED_COMMON_GATE_COMMANDS:
        raise AcceptanceVerificationError(
            "required common quality commands changed order or identity"
        )
    required = frozenset(required_commands)
    if not _EXPECTED_COMMON_GATE_COMMANDS <= required:
        raise AcceptanceVerificationError(
            "implementation evidence omits a common gate command"
        )
    focused = required - _EXPECTED_COMMON_GATE_COMMANDS
    if focused != frozenset((_EXPECTED_CURRENT_FOCUSED_COMMAND,)):
        raise AcceptanceVerificationError(
            "implementation evidence lacks the exact current focused pytest"
            " command"
        )
    raw_results = quality_gate.get("results")
    if not isinstance(raw_results, list):
        raise AcceptanceVerificationError(
            "implementation evidence quality results must be a list"
        )
    state_details = quality_gate.get("state_details")
    tree_binding = quality_gate.get("tree_binding")
    coverage_binding = quality_gate.get("coverage_binding")
    if state == "pending":
        details = _evidence_mapping(state_details, "pending quality state")
        _exact_keys(
            details,
            {"requested_at", "reason"},
            "pending quality state",
        )
        _nonempty_string(details.get("requested_at"), "quality requested_at")
        reason = _nonempty_string(details.get("reason"), "quality reason")
        if len(reason) < 20:
            raise AcceptanceVerificationError(
                "pending quality evidence requires a concrete reason"
            )
        if raw_results or tree_binding != {} or coverage_binding != {}:
            raise AcceptanceVerificationError(
                "pending quality evidence cannot claim completed results or"
                " bindings"
            )
        return

    details = _evidence_mapping(state_details, "complete quality state")
    _exact_keys(
        details,
        {"completed_at", "gate_run_id"},
        "complete quality state",
    )
    _nonempty_string(details.get("completed_at"), "quality completed_at")
    gate_run_id = _nonempty_string(
        details.get("gate_run_id"),
        "quality gate_run_id",
    )
    if len(gate_run_id) < 12:
        raise AcceptanceVerificationError(
            "completed quality evidence requires a concrete gate run ID"
        )
    if len(raw_results) != len(required_commands):
        raise AcceptanceVerificationError(
            "completed quality evidence lacks exact gate results"
        )
    results: dict[str, dict[str, object]] = {}
    observed_commands: list[str] = []
    for value in raw_results:
        result = _evidence_mapping(value, "quality gate result")
        command = _nonempty_string(result.get("command"), "quality command")
        observed_commands.append(command)
        if (
            command in results
            or type(result.get("exit_code")) is not int
            or result.get("exit_code") != 0
        ):
            raise AcceptanceVerificationError(
                "quality gate commands must be unique successful results"
            )
        if _contains_none(result):
            raise AcceptanceVerificationError(
                f"quality gate result contains null evidence: {command}"
            )
        results[command] = result
    if tuple(observed_commands) != required_commands:
        raise AcceptanceVerificationError(
            "completed quality results must preserve required command order"
        )
    focused_command = next(iter(focused))
    test_command = "poetry run pytest --verbose -s"
    test_result = results[test_command]
    _exact_keys(
        test_result,
        {
            "command",
            "exit_code",
            "passed",
            "skipped",
            "subtests_passed",
            "seconds",
            "deselected",
            "xfail",
            "xpass",
        },
        "test quality-gate evidence",
    )
    test_counts = {
        name: _nonnegative_int(test_result.get(name), f"quality {name}")
        for name in (
            "passed",
            "skipped",
            "subtests_passed",
            "deselected",
            "xfail",
            "xpass",
        )
    }
    _positive_number(
        test_result.get("seconds"),
        f"quality seconds: {test_command}",
    )
    if (
        test_counts["passed"] == 0
        or test_counts["deselected"] != 0
        or test_counts["xfail"] != 0
        or test_counts["xpass"] != 0
    ):
        raise AcceptanceVerificationError(
            f"test quality-gate evidence is incomplete: {test_command}"
        )
    focused_result = results[focused_command]
    _exact_keys(
        focused_result,
        {"command", "exit_code", "active_nodes", "active_instances"},
        "focused runtime acceptance evidence",
    )
    if (
        focused_result.get("active_nodes")
        != _EXPECTED_CURRENT_RUNTIME_NODE_COUNT
        or focused_result.get("active_instances")
        != _EXPECTED_CURRENT_RUNTIME_NODE_COUNT
    ):
        raise AcceptanceVerificationError(
            "current focused result differs from the exact collected and"
            " passing runtime inventory"
        )
    legacy_coverage = results["make test-coverage -- -100 src/"]
    _exact_keys(
        legacy_coverage,
        {"command", "exit_code", "output_lines"},
        "legacy coverage evidence",
    )
    if legacy_coverage.get("output_lines") != []:
        raise AcceptanceVerificationError(
            "the legacy exact-coverage audit must have zero output lines"
        )
    exact_coverage = results["make test-coverage-exact no-install"]
    _exact_keys(
        exact_coverage,
        {
            "command",
            "exit_code",
            "covered_statements",
            "total_statements",
            "source_files",
            "missing_lines",
            "missing_files",
            "passed",
            "skipped",
            "subtests_passed",
            "seconds",
        },
        "exact coverage evidence",
    )
    exact_test_counts = {
        name: _nonnegative_int(
            exact_coverage.get(name),
            f"exact coverage {name}",
        )
        for name in ("passed", "skipped", "subtests_passed")
    }
    _positive_number(
        exact_coverage.get("seconds"),
        "exact coverage seconds",
    )
    if exact_test_counts["passed"] == 0:
        raise AcceptanceVerificationError(
            "exact coverage evidence has no passing tests"
        )
    try:
        verified_coverage = verify_src_coverage(
            report_path=root / "coverage.json",
            repo_root=root,
        )
    except CoverageVerificationError as exc:
        raise AcceptanceVerificationError(
            f"live exact source coverage is invalid: {exc}"
        ) from exc
    derived_coverage = {
        "covered_statements": verified_coverage.summary.covered_lines,
        "total_statements": verified_coverage.summary.num_statements,
        "source_files": len(verified_coverage.files),
        "missing_lines": verified_coverage.summary.missing_lines,
        "missing_files": 0,
    }
    observed_coverage_result = {
        key: exact_coverage.get(key) for key in derived_coverage
    }
    if observed_coverage_result != derived_coverage:
        raise AcceptanceVerificationError(
            "exact source-coverage evidence differs from the validated live"
            " report"
        )
    acceptance = results[
        "poetry run python scripts/verify_input_acceptance.py"
        " --through-phase 4"
    ]
    _exact_keys(
        acceptance,
        {"command", "exit_code", "active_nodes", "active_instances"},
        "acceptance evidence",
    )
    if (
        acceptance.get("active_nodes") != active_acceptance_nodes
        or acceptance.get("active_instances") != active_pytest_instances
    ):
        raise AcceptanceVerificationError(
            "acceptance gate evidence has stale node or instance counts"
        )
    type_result = results["make typecheck-input-contract INPUT_PHASE=4"]
    _exact_keys(
        type_result,
        {"command", "exit_code", "active_fixtures"},
        "type evidence",
    )
    if type_result.get("active_fixtures") != active_type_fixtures:
        raise AcceptanceVerificationError(
            "type gate evidence has a stale fixture count"
        )
    lint = results["make lint"]
    _exact_keys(
        lint,
        {
            "command",
            "exit_code",
            "source_files_typechecked",
            "script_files_typechecked",
        },
        "lint quality evidence",
    )
    lint_source_files = _nonnegative_int(
        lint.get("source_files_typechecked"),
        "lint source files typechecked",
    )
    lint_script_files = _nonnegative_int(
        lint.get("script_files_typechecked"),
        "lint script files typechecked",
    )
    if lint_source_files == 0 or lint_script_files == 0:
        raise AcceptanceVerificationError(
            "lint quality evidence has empty typechecked inventories"
        )
    _exact_keys(
        results["git diff --check"],
        {"command", "exit_code"},
        "quality command evidence",
    )
    live_tree_binding = _current_tree_binding(
        root,
        preserved_untracked,
        evidence_payload,
    )
    if tree_binding != live_tree_binding:
        raise AcceptanceVerificationError(
            "completed quality evidence does not match the live git tree"
        )
    coverage = _evidence_mapping(
        coverage_binding,
        "coverage binding",
    )
    _exact_keys(
        coverage,
        {
            "report_sha256",
            "source_inventory_sha256",
            "source_file_count",
            "statement_count",
            "excluded_line_count",
        },
        "coverage binding",
    )
    report_digest = _sha256_string(
        coverage.get("report_sha256"),
        "coverage report SHA-256",
    )
    inventory_digest = _sha256_string(
        coverage.get("source_inventory_sha256"),
        "coverage source inventory SHA-256",
    )
    if report_digest == "0" * 64 or report_digest == inventory_digest:
        raise AcceptanceVerificationError(
            "coverage report digest is missing or reused"
        )
    live_inventory = _source_statement_inventory(root)
    live_report = _coverage_report_binding(root)
    if report_digest != live_report[0]:
        raise AcceptanceVerificationError(
            "coverage report digest does not match the live report"
        )
    if live_report[1:] != live_inventory:
        raise AcceptanceVerificationError(
            "coverage report source inventory differs from live source"
        )
    if (
        len(verified_coverage.files) != live_inventory[1]
        or verified_coverage.summary.num_statements != live_inventory[2]
        or verified_coverage.summary.excluded_lines != live_inventory[3]
    ):
        raise AcceptanceVerificationError(
            "validated exact coverage inventory differs from live source"
        )
    expected_coverage = {
        "source_inventory_sha256": live_inventory[0],
        "source_file_count": live_inventory[1],
        "statement_count": live_inventory[2],
        "excluded_line_count": live_inventory[3],
    }
    observed_coverage = {key: coverage.get(key) for key in expected_coverage}
    if observed_coverage != expected_coverage:
        raise AcceptanceVerificationError(
            "coverage source inventory does not match the live source tree"
        )
    if exact_coverage.get("total_statements") != live_inventory[2]:
        raise AcceptanceVerificationError(
            "exact coverage statement count differs from its source inventory"
        )
    if lint_source_files != live_inventory[1]:
        raise AcceptanceVerificationError(
            "lint source-file count differs from the live source inventory"
        )


def _current_tree_binding(
    root: Path,
    preserved_untracked: tuple[str, ...],
    evidence_payload: dict[str, object],
) -> dict[str, object]:
    evidence_path = "tests/fixtures/input/baseline_evidence.json"
    verifier_path = "scripts/verify_input_acceptance.py"
    ignored_paths = frozenset((evidence_path, verifier_path))
    tracked_modes = _git_stage_modes(root)
    untracked = _git_null_paths(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
    )
    tracked_paths = set(tracked_modes)
    untracked_paths = set(untracked)
    if tracked_paths & untracked_paths:
        raise AcceptanceVerificationError(
            "git tree inventory classifies one path twice"
        )
    included_untracked = {
        relative
        for relative in untracked_paths
        if not any(
            relative.startswith(prefix) for prefix in preserved_untracked
        )
    }
    staged_changed = set(
        _git_null_paths(
            root,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--",
        )
    )
    worktree_changed = set(
        _git_null_paths(
            root,
            "diff",
            "--name-only",
            "-z",
            "--",
        )
    )
    ambiguous_paths = staged_changed & worktree_changed
    if ambiguous_paths:
        raise AcceptanceVerificationError(
            "git tree inventory has staged and unstaged changes for: "
            + ", ".join(sorted(ambiguous_paths))
        )
    inventory: list[dict[str, object]] = []
    resolved_root = root.resolve()
    for relative in sorted(
        (tracked_paths | included_untracked) - ignored_paths
    ):
        pure_path = PurePosixPath(relative)
        if (
            not relative
            or pure_path.is_absolute()
            or "." in pure_path.parts
            or ".." in pure_path.parts
        ):
            raise AcceptanceVerificationError(
                f"unsafe git tree inventory path: {relative}"
            )
        path = resolved_root.joinpath(*pure_path.parts)
        if path.is_symlink():
            raise AcceptanceVerificationError(
                f"git tree inventory entry is a symlink: {relative}"
            )
        if not path.exists():
            if relative in tracked_paths:
                continue
            raise AcceptanceVerificationError(
                f"untracked git tree entry disappeared: {relative}"
            )
        kind = _tree_entry_kind(path, resolved_root, relative)
        if (
            relative in tracked_modes
            and kind != tracked_modes[relative]
            and relative not in worktree_changed
        ):
            raise AcceptanceVerificationError(
                f"git index and working-tree modes differ for: {relative}"
            )
        inventory.append(
            {
                "path": relative,
                "kind": kind,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    normalized_evidence = deepcopy(evidence_payload)
    normalized_quality = normalized_evidence.get("quality_gate")
    if not isinstance(normalized_quality, dict):
        raise AcceptanceVerificationError(
            "cannot normalize quality evidence tree binding"
        )
    cast(dict[str, object], normalized_quality)["tree_binding"] = {}
    normalized_evidence_kind = _bound_tree_entry_kind(
        resolved_root / evidence_path,
        resolved_root,
        evidence_path,
        tracked_modes,
        worktree_changed,
    )
    evidence_digest = sha256(
        dumps(
            normalized_evidence,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    verifier_file = resolved_root / verifier_path
    normalized_verifier_kind = _bound_tree_entry_kind(
        verifier_file,
        resolved_root,
        verifier_path,
        tracked_modes,
        worktree_changed,
    )
    verifier_source = verifier_file.read_text(encoding="utf-8")
    normalized_verifier = verifier_source.replace(
        _EXPECTED_EVIDENCE_SHA256,
        "0" * 64,
    )
    verifier_digest = sha256(normalized_verifier.encode("utf-8")).hexdigest()
    inventory_digest = sha256(
        dumps(
            inventory,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    values: dict[str, object] = {
        "baseline_head": _EXPECTED_BASELINE_HEAD,
        "inventory_file_count": len(inventory),
        "inventory_sha256": inventory_digest,
        "normalized_evidence_kind": normalized_evidence_kind,
        "normalized_evidence_sha256": evidence_digest,
        "normalized_verifier_kind": normalized_verifier_kind,
        "normalized_verifier_sha256": verifier_digest,
    }
    values["tree_sha256"] = sha256(
        dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return values


def _tree_entry_kind(path: Path, root: Path, relative: str) -> str:
    if path.is_symlink():
        raise AcceptanceVerificationError(
            f"git tree inventory entry is a symlink: {relative}"
        )
    if not path.exists():
        raise AcceptanceVerificationError(
            f"git tree inventory entry is missing: {relative}"
        )
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise AcceptanceVerificationError(
            f"git tree inventory path escapes repository: {relative}"
        ) from exc
    if resolved_path != path or not path.is_file():
        raise AcceptanceVerificationError(
            f"git tree inventory entry is not a regular file: {relative}"
        )
    mode = path.stat().st_mode
    if mode & (S_IXUSR | S_IXGRP | S_IXOTH):
        return "executable"
    return "regular"


def _bound_tree_entry_kind(
    path: Path,
    root: Path,
    relative: str,
    tracked_modes: dict[str, str],
    worktree_changed: set[str],
) -> str:
    kind = _tree_entry_kind(path, root, relative)
    if (
        relative in tracked_modes
        and kind != tracked_modes[relative]
        and relative not in worktree_changed
    ):
        raise AcceptanceVerificationError(
            f"git index and working-tree modes differ for: {relative}"
        )
    return kind


def _git_stage_modes(root: Path) -> dict[str, str]:
    raw = _git_bytes(root, "ls-files", "--stage", "-z", "--")
    if not raw:
        return {}
    if not raw.endswith(b"\0"):
        raise AcceptanceVerificationError(
            "git index inventory is not NUL terminated"
        )
    values: dict[str, str] = {}
    for raw_entry in raw.removesuffix(b"\0").split(b"\0"):
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_id, stage = metadata.split(b" ")
            relative = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise AcceptanceVerificationError(
                "git index inventory contains an invalid entry"
            ) from exc
        if (
            stage != b"0"
            or len(object_id) not in {40, 64}
            or any(value not in b"0123456789abcdef" for value in object_id)
        ):
            raise AcceptanceVerificationError(
                f"git index inventory contains an unresolved entry: {relative}"
            )
        match mode:
            case b"100644":
                kind = "regular"
            case b"100755":
                kind = "executable"
            case _:
                raise AcceptanceVerificationError(
                    f"git index inventory contains an unsafe mode: {relative}"
                )
        if relative in values:
            raise AcceptanceVerificationError(
                f"git index inventory contains a duplicate path: {relative}"
            )
        values[relative] = kind
    return values


def _git_null_paths(root: Path, *arguments: str) -> tuple[str, ...]:
    raw = _git_bytes(root, *arguments)
    if not raw:
        return ()
    if not raw.endswith(b"\0"):
        raise AcceptanceVerificationError(
            "git tree inventory is not NUL terminated"
        )
    try:
        values = tuple(
            value.decode("utf-8")
            for value in raw.removesuffix(b"\0").split(b"\0")
        )
    except UnicodeDecodeError as exc:
        raise AcceptanceVerificationError(
            "git tree inventory contains a non-UTF-8 path"
        ) from exc
    if len(set(values)) != len(values):
        raise AcceptanceVerificationError(
            "git tree inventory contains a duplicate path"
        )
    return values


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
        text=False,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AcceptanceVerificationError(
            "cannot verify live git tree binding:"
            f" git {' '.join(arguments)}: {detail}"
        )
    return completed.stdout


def _source_statement_inventory(root: Path) -> tuple[str, int, int, int]:
    source_root = root / "src"
    if not source_root.is_dir():
        raise AcceptanceVerificationError(
            "coverage source inventory root is missing"
        )
    analyzer = Coverage(config_file=False, data_file=None)
    inventory: list[dict[str, object]] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            _, statements, excluded, _, _ = analyzer.analysis2(str(path))
        except Exception as exc:
            raise AcceptanceVerificationError(
                f"cannot analyze coverage source inventory: {relative}: {exc}"
            ) from exc
        inventory.append(
            {
                "path": relative,
                "statements": len(statements),
                "excluded_lines": len(excluded),
            }
        )
    digest = sha256(
        dumps(
            inventory,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return (
        digest,
        len(inventory),
        sum(cast(int, entry["statements"]) for entry in inventory),
        sum(cast(int, entry["excluded_lines"]) for entry in inventory),
    )


def _coverage_report_binding(
    root: Path,
) -> tuple[str, str, int, int, int]:
    report_path = root / "coverage.json"
    if not report_path.is_file():
        raise AcceptanceVerificationError(
            "completed quality evidence requires the live coverage report"
        )
    report_digest = sha256(report_path.read_bytes()).hexdigest()
    report = _strict_mapping(report_path, "coverage report evidence")
    raw_files = report.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise AcceptanceVerificationError(
            "coverage report evidence has no source files"
        )
    inventory: list[dict[str, object]] = []
    for relative, raw in sorted(raw_files.items()):
        if not isinstance(relative, str) or not relative.startswith("src/"):
            raise AcceptanceVerificationError(
                f"coverage report contains a non-source path: {relative!r}"
            )
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                f"coverage report file entry must be an object: {relative}"
            )
        summary = cast(dict[str, object], raw).get("summary")
        if not isinstance(summary, dict):
            raise AcceptanceVerificationError(
                f"coverage report file summary is missing: {relative}"
            )
        statements = summary.get("num_statements")
        excluded = summary.get("excluded_lines")
        if (
            type(statements) is not int
            or statements < 0
            or type(excluded) is not int
            or excluded < 0
        ):
            raise AcceptanceVerificationError(
                f"coverage report file counts are invalid: {relative}"
            )
        inventory.append(
            {
                "path": relative,
                "statements": statements,
                "excluded_lines": excluded,
            }
        )
    inventory_digest = sha256(
        dumps(
            inventory,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return (
        report_digest,
        inventory_digest,
        len(inventory),
        sum(cast(int, entry["statements"]) for entry in inventory),
        sum(cast(int, entry["excluded_lines"]) for entry in inventory),
    )


def _sha256_string(value: object, label: str) -> str:
    digest = _nonempty_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise AcceptanceVerificationError(
            f"{label} must be lowercase hexadecimal"
        )
    return digest


def _contains_none(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_none(item) for item in value)
    return False


def _record_ids(
    raw: object,
    keys: set[str],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise AcceptanceVerificationError(
            f"{label} inventory must be non-empty"
        )
    values: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise AcceptanceVerificationError(f"{label} must be an object")
        item = cast(dict[str, object], entry)
        _exact_keys(item, keys, label)
        values.append(_nonempty_string(item.get("id"), f"{label} id"))
        _nonempty_string(item.get("description"), f"{label} description")
    _unique(values, f"{label} id")
    return tuple(values)


def _run_probe(
    driver: str,
    sentinel: str,
    node_ids: tuple[str, ...],
    root: Path,
) -> dict[str, object]:
    environment = dict(environ)
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_PLUGINS", None)
    environment.pop("PYTHONPATH", None)
    for key in tuple(environment):
        if key.startswith("COV_CORE_") or key == "COVERAGE_PROCESS_START":
            del environment[key]
    completed = run(
        [executable, "-c", driver, *node_ids],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=_PROCESS_TIMEOUT_SECONDS,
    )
    return _probe_payload(completed, sentinel)


def _probe_payload(
    completed: CompletedProcess[str], sentinel: str
) -> dict[str, object]:
    if type(completed.returncode) is not int or completed.returncode != 0:
        raise AcceptanceVerificationError(
            "acceptance probe process exited with code "
            f"{completed.returncode}:\n{completed.stdout}{completed.stderr}"
        )
    lines = [
        line.removeprefix(sentinel)
        for line in completed.stdout.splitlines()
        if line.startswith(sentinel)
    ]
    if len(lines) != 1:
        raise AcceptanceVerificationError(
            "acceptance probe did not return one result payload"
        )
    try:
        payload = strict_json_loads(lines[0])
    except (StrictJsonError, ValueError) as exc:
        raise AcceptanceVerificationError(
            f"acceptance probe returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AcceptanceVerificationError(
            "acceptance probe payload must be an object"
        )
    result = cast(dict[str, object], payload)
    result["probe_stdout"] = completed.stdout[-4000:]
    result["probe_stderr"] = completed.stderr[-4000:]
    return result


def _collection_node_ids(
    payload: dict[str, object],
    *,
    reject_disallowed_markers: bool = True,
) -> tuple[str, ...]:
    """Return validated node IDs from one collection probe."""
    _exact_keys(
        payload,
        {
            "exit_code",
            "items",
            "deselected",
            "collection_reports",
            "probe_stdout",
            "probe_stderr",
        },
        "collection payload",
    )
    _verify_probe_common(payload)
    if (
        payload.get("exit_code") != 0
        or type(payload.get("exit_code")) is not int
    ):
        raise AcceptanceVerificationError(
            "acceptance collection exited with code"
            f" {payload.get('exit_code')}"
        )
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise AcceptanceVerificationError("collection items must be a list")
    observed: list[str] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "collection item must be an object"
            )
        item = cast(dict[str, object], raw)
        _exact_keys(item, {"nodeid", "markers"}, "collection item")
        node_id = _nonempty_string(item.get("nodeid"), "collected nodeid")
        markers = _string_list_allow_empty(item.get("markers"), "markers")
        disallowed = (
            sorted(set(markers) & _DISALLOWED_MARKERS)
            if reject_disallowed_markers
            else []
        )
        if disallowed:
            raise AcceptanceVerificationError(
                f"{node_id} has disallowed markers: {disallowed}"
            )
        observed.append(node_id)
    collected = tuple(observed)
    _unique(collected, "collected node ID")
    return collected


def _verify_collection(
    expected: tuple[str, ...], payload: dict[str, object]
) -> tuple[str, ...]:
    collected = _collection_node_ids(payload)
    _verify_identical_nodes(expected, collected, "collected")
    return collected


def _verify_execution(
    expected: tuple[str, ...],
    payload: dict[str, object],
    collected: tuple[str, ...],
) -> None:
    _exact_keys(
        payload,
        {
            "exit_code",
            "items",
            "deselected",
            "collection_reports",
            "reports",
            "probe_stdout",
            "probe_stderr",
        },
        "execution payload",
    )
    _verify_probe_common(payload)
    _verify_identical_nodes(expected, collected, "collected")
    raw_items = _string_list(payload.get("items"), "execution items")
    _verify_identical_nodes(collected, raw_items, "executed")
    raw_reports = payload.get("reports")
    if not isinstance(raw_reports, list):
        raise AcceptanceVerificationError("execution reports must be a list")
    by_node: dict[str, list[dict[str, object]]] = {
        node: [] for node in raw_items
    }
    for raw in raw_reports:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "execution report must be an object"
            )
        report = cast(dict[str, object], raw)
        _exact_keys(
            report,
            {"nodeid", "when", "outcome", "wasxfail", "detail"},
            "execution report",
        )
        node_id = report.get("nodeid")
        if not isinstance(node_id, str) or node_id not in by_node:
            raise AcceptanceVerificationError(
                f"unexpected acceptance execution report: {node_id!r}"
            )
        by_node[node_id].append(report)
    for node_id, reports in by_node.items():
        phases = [report.get("when") for report in reports]
        if (
            phases.count("setup") != 1
            or phases.count("call") < 1
            or phases.count("teardown") != 1
            or set(phases) != {"setup", "call", "teardown"}
        ):
            raise AcceptanceVerificationError(
                f"{node_id} was not exactly once fully executed: {phases}"
            )
        for report in reports:
            if not isinstance(report.get("wasxfail"), str):
                raise AcceptanceVerificationError("wasxfail must be a string")
            if report.get("wasxfail"):
                raise AcceptanceVerificationError(
                    f"{node_id} produced an xfail/xpass outcome"
                )
            if report.get("outcome") != "passed":
                raise AcceptanceVerificationError(
                    f"{node_id} {report.get('when')} outcome was "
                    f"{report.get('outcome')}: {report.get('detail')}"
                )
    if (
        payload.get("exit_code") != 0
        or type(payload.get("exit_code")) is not int
    ):
        raise AcceptanceVerificationError(
            f"acceptance execution exited with code {payload.get('exit_code')}"
        )


def _verify_probe_common(payload: dict[str, object]) -> None:
    if not isinstance(payload.get("probe_stdout"), str) or not isinstance(
        payload.get("probe_stderr"), str
    ):
        raise AcceptanceVerificationError("probe diagnostics must be strings")
    deselected = _string_list_allow_empty(
        payload.get("deselected"), "deselected"
    )
    if deselected:
        raise AcceptanceVerificationError(
            f"acceptance nodes were deselected: {deselected}"
        )
    reports = payload.get("collection_reports")
    if not isinstance(reports, list):
        raise AcceptanceVerificationError("collection reports must be a list")
    if reports:
        raise AcceptanceVerificationError(
            f"acceptance collection was skipped or failed: {reports}"
        )


def _verify_identical_nodes(
    expected: tuple[str, ...], observed: tuple[str, ...], label: str
) -> None:
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    duplicates = sorted(
        node_id for node_id in set(observed) if observed.count(node_id) > 1
    )
    if missing or unexpected or duplicates or len(expected) != len(observed):
        raise AcceptanceVerificationError(
            f"acceptance nodes were not exactly {label}: missing={missing},"
            f" unexpected={unexpected}, duplicates={duplicates}"
        )


def _validate_execution_scope(
    manifest_path: Path,
    node_ids: tuple[str, ...],
    root: Path,
) -> None:
    if not root.is_dir():
        raise AcceptanceVerificationError(
            f"acceptance repository root is not a directory: {root}"
        )
    try:
        manifest_path.resolve().relative_to(root)
    except ValueError as exc:
        raise AcceptanceVerificationError(
            "acceptance manifest must be inside the repository root"
        ) from exc
    for node_id in node_ids:
        raw_path = node_id.split("::", 1)[0]
        posix = PurePosixPath(raw_path)
        if posix.is_absolute() or ".." in posix.parts or "\\" in raw_path:
            raise AcceptanceVerificationError(
                f"acceptance node path escapes repository root: {raw_path}"
            )
        if not posix.parts or posix.parts[0] != "tests":
            raise AcceptanceVerificationError(
                f"active acceptance node must be under tests/: {raw_path}"
            )
        path = (root / Path(*posix.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AcceptanceVerificationError(
                f"acceptance node path escapes repository root: {raw_path}"
            ) from exc


def _validate_test_implementation(node_id: str, root: Path) -> None:
    raw_path, *parts = node_id.split("::")
    if not parts:
        raise AcceptanceVerificationError(f"invalid pytest node ID: {node_id}")
    names = [part.split("[", 1)[0] for part in parts]
    function_name = names[-1]
    path = root / Path(*PurePosixPath(raw_path).parts)
    try:
        source = path.read_text(encoding="utf-8")
        tree = parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise AcceptanceVerificationError(
            f"cannot inspect acceptance test {node_id}: {exc}"
        ) from exc
    scope: object = tree
    for class_name in names[:-1]:
        body = getattr(scope, "body", ())
        classes = [
            node
            for node in body
            if isinstance(node, ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise AcceptanceVerificationError(
                f"acceptance test class is missing or ambiguous: {node_id}"
            )
        scope = classes[0]
    functions = [
        node
        for node in getattr(scope, "body", ())
        if isinstance(node, (FunctionDef, AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(functions) != 1:
        raise AcceptanceVerificationError(
            f"acceptance test function is missing or ambiguous: {node_id}"
        )
    function = functions[0]
    body = list(function.body)
    if (
        body
        and isinstance(body[0], Expr)
        and isinstance(body[0].value, Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    meaningful = [
        statement
        for statement in body
        if not _placeholder_statement(statement)
    ]
    if not meaningful:
        raise AcceptanceVerificationError(
            "acceptance test is a placeholder or unconditional pass:"
            f" {node_id}"
        )
    class_scope = scope if isinstance(scope, ClassDef) else None
    aliases = _contextlib_suppress_aliases(
        tree,
        function,
        class_scope=class_scope,
    )
    check_paths = _check_sequence(body, frozenset({False}), aliases)
    successful_paths = check_paths.next_states | check_paths.return_states
    if (
        not successful_paths or False in successful_paths
    ) and not _has_static_nonempty_checked_loop(function, aliases):
        raise AcceptanceVerificationError(
            "acceptance test has a reachable successful path without a"
            f" meaningful check: {node_id}"
        )
    _validate_prohibited_test_constructs(function, node_id, tree)
    segment = "\n".join(
        source.splitlines()[function.lineno - 1 : function.end_lineno]
    )
    if _COVERAGE_EXCLUSION_PATTERN.search(segment):
        raise AcceptanceVerificationError(
            "acceptance test uses a feature-specific coverage exclusion:"
            f" {node_id}"
        )


def _has_static_nonempty_checked_loop(
    function: FunctionDef | AsyncFunctionDef,
    aliases: _SuppressContextAliases,
) -> bool:
    """Return whether a non-empty local literal drives a checked loop."""
    for index, statement in enumerate(function.body):
        if not isinstance(statement, (For, AsyncFor)):
            continue
        iterator_name = (
            statement.iter.id
            if isinstance(statement.iter, Name)
            else (
                statement.iter.args[0].id
                if isinstance(statement.iter, Call)
                and isinstance(statement.iter.func, Name)
                and statement.iter.func.id in {"enumerate", "iter", "reversed"}
                and len(statement.iter.args) == 1
                and not statement.iter.keywords
                and isinstance(statement.iter.args[0], Name)
                else None
            )
        )
        if iterator_name is None or not _name_is_statically_nonempty(
            function.body[:index],
            iterator_name,
        ):
            continue
        paths = _check_sequence(
            statement.body,
            frozenset({False}),
            aliases,
        )
        successful = (
            paths.next_states
            | paths.return_states
            | paths.break_states
            | paths.continue_states
        )
        prefix = _check_sequence(
            function.body[:index],
            frozenset({False}),
            aliases,
        )
        if (
            prefix.next_states
            and False not in prefix.return_states
            and successful
            and False not in successful
        ):
            return True
    return False


def _name_is_statically_nonempty(
    statements: Sequence[AST],
    name: str,
) -> bool:
    """Return whether the latest reachable binding is a non-empty literal."""
    nonempty = False
    for statement in statements:
        binds_name, value = _direct_name_binding(statement, name)
        if binds_name:
            nonempty = value is not None and _statically_nonempty_iter(value)
    return nonempty


def _direct_name_binding(
    statement: AST,
    name: str,
) -> tuple[bool, AST | None]:
    """Return a conservative direct binding made by one statement."""
    if isinstance(statement, Assign):
        values = tuple(
            item_value
            for target in statement.targets
            for target_name, item_value in _assignment_bindings(
                target,
                statement.value,
            )
            if target_name == name
        )
        if values:
            return True, values[-1]
    elif isinstance(statement, AnnAssign):
        values = tuple(
            item_value
            for target_name, item_value in _assignment_bindings(
                statement.target,
                statement.value,
            )
            if target_name == name
        )
        if values:
            return True, values[-1]
    elif isinstance(statement, (FunctionDef, AsyncFunctionDef, ClassDef)):
        return statement.name == name, None
    elif isinstance(statement, Import):
        return (
            any(
                (item.asname or item.name.split(".", 1)[0]) == name
                for item in statement.names
            ),
            None,
        )
    elif isinstance(statement, ImportFrom):
        return (
            any(
                (item.asname or item.name) == name for item in statement.names
            ),
            None,
        )
    if any(
        isinstance(node, Name)
        and node.id == name
        and not isinstance(node.ctx, Load)
        for node in walk(statement)
    ):
        return True, None
    if _statement_may_empty_iterable(statement, name):
        return True, None
    return False, None


def _statement_may_empty_iterable(statement: AST, name: str) -> bool:
    """Return whether a statement may shrink or expose an iterable."""
    if isinstance(
        statement, (Assign, AnnAssign, AugAssign, NamedExpr)
    ) and any(
        isinstance(node, Name)
        and node.id == name
        and isinstance(node.ctx, Load)
        for node in walk(statement)
    ):
        return True
    for node in walk(statement):
        if isinstance(node, Subscript) and not isinstance(node.ctx, Load):
            if any(
                isinstance(child, Name) and child.id == name
                for child in walk(node.value)
            ):
                return True
        if not isinstance(node, Call):
            continue
        references = tuple(
            child
            for child in walk(node)
            if isinstance(child, Name) and child.id == name
        )
        if not references:
            continue
        if (
            isinstance(node.func, Attribute)
            and isinstance(node.func.value, Name)
            and node.func.value.id == name
            and node.func.attr
            in {"append", "extend", "insert", "reverse", "sort"}
            and len(references) == 1
        ):
            continue
        return True
    return False


def _placeholder_statement(statement: AST) -> bool:
    if isinstance(statement, Pass):
        return True
    if isinstance(statement, Expr) and isinstance(statement.value, Constant):
        return statement.value.value in {True, Ellipsis, None}
    if isinstance(statement, Assert) and isinstance(statement.test, Constant):
        return True
    if isinstance(statement, Expr) and isinstance(statement.value, Call):
        call = statement.value
        if (
            isinstance(call.func, Attribute)
            and call.func.attr in {"assertTrue", "assertEqual"}
            and call.args
            and all(isinstance(argument, Constant) for argument in call.args)
        ):
            values = tuple(
                cast(Constant, argument).value for argument in call.args
            )
            if call.func.attr == "assertTrue" and bool(values[0]):
                return True
            if (
                call.func.attr == "assertEqual"
                and len(values) >= 2
                and values[0] == values[1]
            ):
                return True
    return isinstance(statement, Return) and statement.value is None


def _is_check(node: AST) -> bool:
    if isinstance(node, Assert):
        return _expression_is_dynamic(node.test)
    if not isinstance(node, Call):
        return False
    if isinstance(node.func, Attribute):
        if node.func.attr in {"raises", "warns"}:
            return bool(node.args) and any(
                _expression_is_dynamic(argument) for argument in node.args
            )
        if not node.func.attr.startswith("assert"):
            return False
        if (
            node.func.attr
            in {"assertEqual", "assertIs", "assertSequenceEqual"}
            and len(node.args) >= 2
            and _same_expression(node.args[0], node.args[1])
        ):
            return False
        return not node.args or any(
            _expression_is_dynamic(argument) for argument in node.args
        )
    return False


def _node_position(node: AST) -> tuple[int, int]:
    """Return a stable source position for one AST node."""
    return (
        cast(int, getattr(node, "lineno", -1)),
        cast(int, getattr(node, "col_offset", -1)),
    )


def _assignment_bindings(
    target: AST,
    value: AST | None,
) -> tuple[tuple[str, AST | None], ...]:
    """Return simple names and corresponding values from an assignment."""
    if isinstance(target, Name):
        return ((target.id, value),)
    if isinstance(target, Starred):
        starred_value = (
            AstList(elts=value.elts, ctx=Load())
            if isinstance(value, (AstList, AstTuple))
            else None
        )
        return _assignment_bindings(target.value, starred_value)
    if isinstance(target, (AstList, AstTuple)):
        values: list[AST | None] = [None] * len(target.elts)
        if isinstance(value, (AstList, AstTuple)):
            starred = tuple(
                index
                for index, item in enumerate(target.elts)
                if isinstance(item, Starred)
            )
            if not starred and len(value.elts) == len(target.elts):
                values = list(value.elts)
            elif len(starred) == 1:
                starred_index = starred[0]
                trailing = len(target.elts) - starred_index - 1
                if len(value.elts) >= starred_index + trailing:
                    values[:starred_index] = value.elts[:starred_index]
                    if trailing:
                        values[-trailing:] = value.elts[-trailing:]
                    end = len(value.elts) - trailing if trailing else None
                    values[starred_index] = AstList(
                        elts=value.elts[starred_index:end],
                        ctx=Load(),
                    )
        elif value is not None:
            expression_value = cast(AstExpression, value)
            starred = tuple(
                index
                for index, item in enumerate(target.elts)
                if isinstance(item, Starred)
            )
            possible_starred_index: int | None = (
                starred[0] if len(starred) == 1 else None
            )
            for index, item in enumerate(target.elts):
                if isinstance(item, Starred):
                    values[index] = Starred(
                        value=expression_value,
                        ctx=Load(),
                    )
                    continue
                subscript_index = (
                    index
                    if possible_starred_index is None
                    or index < possible_starred_index
                    else index - len(target.elts)
                )
                values[index] = Subscript(
                    value=expression_value,
                    slice=Constant(value=subscript_index),
                    ctx=Load(),
                )
        return tuple(
            binding
            for item, item_value in zip(target.elts, values, strict=True)
            for binding in _assignment_bindings(item, item_value)
        )
    return ()


def _append_target_alias_bindings(
    bindings: list[_AliasBinding],
    target: AST,
    value: AST | None,
    position: tuple[int, int],
    definite: bool,
) -> None:
    """Append target bindings with one control-flow certainty."""
    bindings.extend(
        _AliasBinding(
            name=name,
            position=position,
            value=item_value,
            definite=definite,
        )
        for name, item_value in _assignment_bindings(target, value)
    )


def _collect_expression_alias_bindings(
    expression: AST,
    bindings: list[_AliasBinding],
    definite: bool,
) -> None:
    """Collect walrus bindings while respecting expression short-circuiting."""
    if isinstance(expression, Lambda):
        return
    if isinstance(expression, NamedExpr):
        _collect_expression_alias_bindings(
            expression.value,
            bindings,
            definite,
        )
        _append_target_alias_bindings(
            bindings,
            expression.target,
            expression.value,
            _node_position(expression),
            definite,
        )
        return
    if isinstance(expression, BoolOp):
        next_definite = definite
        for value in expression.values:
            _collect_expression_alias_bindings(
                value,
                bindings,
                next_definite,
            )
            if isinstance(value, Constant):
                short_circuits = (
                    not bool(value.value)
                    if type(expression.op).__name__ == "And"
                    else bool(value.value)
                )
                if short_circuits:
                    break
                continue
            next_definite = False
        return
    if isinstance(expression, IfExp):
        _collect_expression_alias_bindings(
            expression.test,
            bindings,
            definite,
        )
        if isinstance(expression.test, Constant):
            branch = (
                expression.body
                if bool(expression.test.value)
                else expression.orelse
            )
            _collect_expression_alias_bindings(branch, bindings, definite)
            return
        _collect_expression_alias_bindings(
            expression.body,
            bindings,
            False,
        )
        _collect_expression_alias_bindings(
            expression.orelse,
            bindings,
            False,
        )
        return
    if isinstance(expression, Compare):
        _collect_expression_alias_bindings(
            expression.left,
            bindings,
            definite,
        )
        for index, comparator in enumerate(expression.comparators):
            _collect_expression_alias_bindings(
                comparator,
                bindings,
                definite if index == 0 else False,
            )
        return
    nested_definite = (
        False
        if isinstance(
            expression,
            (DictComp, GeneratorExp, ListComp, SetComp),
        )
        else definite
    )
    for child in iter_child_nodes(expression):
        _collect_expression_alias_bindings(
            child,
            bindings,
            nested_definite,
        )


def _iterable_alias_values(expression: AST) -> tuple[AST, ...] | None:
    """Return statically possible values yielded by an iterable."""
    if isinstance(expression, (AstList, AstSet, AstTuple)):
        return tuple(expression.elts)
    if isinstance(expression, BinOp) and type(expression.op).__name__ == "Add":
        left = _iterable_alias_values(expression.left)
        right = _iterable_alias_values(expression.right)
        if left is not None and right is not None:
            return (*left, *right)
    if (
        isinstance(expression, (GeneratorExp, ListComp, SetComp))
        and len(expression.generators) == 1
    ):
        generator = expression.generators[0]
        values = _iterable_alias_values(generator.iter)
        if (
            values is not None
            and isinstance(generator.target, Name)
            and isinstance(expression.elt, Name)
            and expression.elt.id == generator.target.id
        ):
            return values
        return (expression.elt,)
    return None


def _collect_scope_statement_bindings(
    statements: Sequence[AST],
    bindings: list[_AliasBinding],
    definite: bool,
) -> None:
    """Collect may-alias bindings across one sequence of statements."""
    for statement in statements:
        position = _node_position(statement)
        if isinstance(statement, (FunctionDef, AsyncFunctionDef, ClassDef)):
            bindings.append(
                _AliasBinding(
                    name=statement.name,
                    position=position,
                    definite=definite,
                )
            )
            continue
        if isinstance(statement, Import):
            for item in statement.names:
                name = item.asname or item.name.split(".", 1)[0]
                direct_kind = {
                    "builtins": _ALIAS_BUILTINS,
                    "contextlib": _ALIAS_CONTEXTLIB,
                    "pytest": _ALIAS_PYTEST,
                    "tempfile": _ALIAS_TEMPFILE,
                    "unittest": _ALIAS_UNITTEST,
                    "unittest.mock": (
                        _ALIAS_UNITTEST_MOCK
                        if item.asname is not None
                        else _ALIAS_UNITTEST
                    ),
                }.get(item.name)
                bindings.append(
                    _AliasBinding(
                        name=name,
                        position=position,
                        direct_kind=direct_kind,
                        definite=definite,
                    )
                )
            continue
        if isinstance(statement, ImportFrom):
            canonical_module = (
                statement.module if statement.level == 0 else None
            )
            for item in statement.names:
                if item.name == "*":
                    if canonical_module == "contextlib":
                        for name, direct_kind in (
                            ("nullcontext", _ALIAS_NULLCONTEXT),
                            ("suppress", _ALIAS_SUPPRESS),
                        ):
                            bindings.append(
                                _AliasBinding(
                                    name=name,
                                    position=position,
                                    direct_kind=direct_kind,
                                    imported_attribute=name,
                                    definite=definite,
                                )
                            )
                    elif canonical_module == "pytest":
                        for name in _PYTEST_CHECK_CONTEXT_NAMES:
                            bindings.append(
                                _AliasBinding(
                                    name=name,
                                    position=position,
                                    direct_kind=_ALIAS_CHECK_CONTEXT,
                                    imported_attribute=name,
                                    definite=definite,
                                )
                            )
                    elif canonical_module == "tempfile":
                        for name in _TEMPFILE_SAFE_CONTEXT_NAMES:
                            bindings.append(
                                _AliasBinding(
                                    name=name,
                                    position=position,
                                    direct_kind=_ALIAS_SAFE_CONTEXT,
                                    imported_attribute=name,
                                    definite=definite,
                                )
                            )
                    elif canonical_module == "unittest":
                        for name in _UNITTEST_TEST_CASE_NAMES:
                            bindings.append(
                                _AliasBinding(
                                    name=name,
                                    position=position,
                                    direct_kind=_ALIAS_UNITTEST_TEST_CASE,
                                    imported_attribute=name,
                                    definite=definite,
                                )
                            )
                    elif canonical_module == "unittest.mock":
                        bindings.append(
                            _AliasBinding(
                                name="patch",
                                position=position,
                                direct_kind=_ALIAS_UNITTEST_PATCH,
                                imported_attribute="patch",
                                definite=definite,
                            )
                        )
                    continue
                imported_kind: str | None = None
                exception_names: frozenset[str] = frozenset()
                if (
                    canonical_module == "contextlib"
                    and item.name == "suppress"
                ):
                    imported_kind = _ALIAS_SUPPRESS
                elif (
                    canonical_module == "contextlib"
                    and item.name == "nullcontext"
                ):
                    imported_kind = _ALIAS_NULLCONTEXT
                elif (
                    canonical_module == "builtins"
                    and item.name in _KNOWN_BUILTIN_EXCEPTION_NAMES
                ):
                    imported_kind = _ALIAS_EXCEPTIONS
                    exception_names = frozenset((item.name,))
                elif (
                    canonical_module == "pytest"
                    and item.name in _PYTEST_CHECK_CONTEXT_NAMES
                ):
                    imported_kind = _ALIAS_CHECK_CONTEXT
                elif (
                    canonical_module == "tempfile"
                    and item.name in _TEMPFILE_SAFE_CONTEXT_NAMES
                ):
                    imported_kind = _ALIAS_SAFE_CONTEXT
                elif (
                    canonical_module == "unittest"
                    and item.name in _UNITTEST_TEST_CASE_NAMES
                ):
                    imported_kind = _ALIAS_UNITTEST_TEST_CASE
                elif canonical_module == "unittest" and item.name == "mock":
                    imported_kind = _ALIAS_UNITTEST_MOCK
                elif (
                    canonical_module == "unittest.mock"
                    and item.name == "patch"
                ):
                    imported_kind = _ALIAS_UNITTEST_PATCH
                elif canonical_module == "builtins" and item.name == "open":
                    imported_kind = _ALIAS_SAFE_CONTEXT
                bindings.append(
                    _AliasBinding(
                        name=item.asname or item.name,
                        position=position,
                        direct_kind=imported_kind,
                        direct_exception_names=exception_names,
                        imported_attribute=(
                            item.name if imported_kind is not None else None
                        ),
                        definite=definite,
                    )
                )
            continue
        if isinstance(statement, Assign):
            _collect_expression_alias_bindings(
                statement.value,
                bindings,
                definite,
            )
            for target in statement.targets:
                _append_target_alias_bindings(
                    bindings,
                    target,
                    statement.value,
                    position,
                    definite,
                )
            continue
        if isinstance(statement, AnnAssign):
            if statement.value is not None:
                _collect_expression_alias_bindings(
                    statement.value,
                    bindings,
                    definite,
                )
            _append_target_alias_bindings(
                bindings,
                statement.target,
                statement.value,
                position,
                definite,
            )
            continue
        if isinstance(statement, AugAssign):
            _collect_expression_alias_bindings(
                statement.value,
                bindings,
                definite,
            )
            _append_target_alias_bindings(
                bindings,
                statement.target,
                None,
                position,
                definite,
            )
            continue
        if isinstance(statement, Delete):
            for target in statement.targets:
                _append_target_alias_bindings(
                    bindings,
                    target,
                    None,
                    position,
                    definite,
                )
            continue
        if isinstance(statement, If):
            _collect_expression_alias_bindings(
                statement.test,
                bindings,
                definite,
            )
            if isinstance(statement.test, Constant):
                branch = (
                    statement.body
                    if bool(statement.test.value)
                    else statement.orelse
                )
                _collect_scope_statement_bindings(
                    branch,
                    bindings,
                    definite,
                )
                continue
            _collect_scope_statement_bindings(
                statement.body,
                bindings,
                False,
            )
            _collect_scope_statement_bindings(
                statement.orelse,
                bindings,
                False,
            )
            continue
        if isinstance(statement, (For, AsyncFor)):
            _collect_expression_alias_bindings(
                statement.iter,
                bindings,
                definite,
            )
            values = _iterable_alias_values(statement.iter)
            if values is None:
                values = (Starred(value=statement.iter, ctx=Load()),)
            for value in values:
                _append_target_alias_bindings(
                    bindings,
                    statement.target,
                    value,
                    position,
                    False,
                )
            _collect_scope_statement_bindings(
                statement.body,
                bindings,
                False,
            )
            _collect_scope_statement_bindings(
                statement.orelse,
                bindings,
                False,
            )
            continue
        if isinstance(statement, While):
            _collect_expression_alias_bindings(
                statement.test,
                bindings,
                definite,
            )
            if isinstance(statement.test, Constant) and not bool(
                statement.test.value
            ):
                _collect_scope_statement_bindings(
                    statement.orelse,
                    bindings,
                    definite,
                )
                continue
            _collect_scope_statement_bindings(
                statement.body,
                bindings,
                False,
            )
            _collect_scope_statement_bindings(
                statement.orelse,
                bindings,
                False,
            )
            continue
        if isinstance(statement, (With, AsyncWith)):
            for context_item in statement.items:
                _collect_expression_alias_bindings(
                    context_item.context_expr,
                    bindings,
                    definite,
                )
                if context_item.optional_vars is not None:
                    context_value = (
                        context_item.context_expr.args[0]
                        if isinstance(context_item.context_expr, Call)
                        and context_item.context_expr.args
                        else context_item.context_expr
                    )
                    _append_target_alias_bindings(
                        bindings,
                        context_item.optional_vars,
                        context_value,
                        position,
                        False,
                    )
            _collect_scope_statement_bindings(
                statement.body,
                bindings,
                False,
            )
            continue
        if isinstance(statement, Try):
            _collect_scope_statement_bindings(
                statement.body,
                bindings,
                False,
            )
            _collect_scope_statement_bindings(
                statement.orelse,
                bindings,
                False,
            )
            for handler in statement.handlers:
                if handler.name is not None:
                    bindings.append(
                        _AliasBinding(
                            name=handler.name,
                            position=_node_position(handler),
                            definite=False,
                        )
                    )
                _collect_scope_statement_bindings(
                    handler.body,
                    bindings,
                    False,
                )
            _collect_scope_statement_bindings(
                statement.finalbody,
                bindings,
                False,
            )
            continue
        if isinstance(statement, Match):
            _collect_expression_alias_bindings(
                statement.subject,
                bindings,
                definite,
            )
            for case in statement.cases:
                capture_value = Starred(
                    value=statement.subject,
                    ctx=Load(),
                )
                for name in _match_capture_names(case.pattern):
                    bindings.append(
                        _AliasBinding(
                            name=name,
                            position=_node_position(case.pattern),
                            value=capture_value,
                            definite=False,
                        )
                    )
                if case.guard is not None:
                    _collect_expression_alias_bindings(
                        case.guard,
                        bindings,
                        False,
                    )
                _collect_scope_statement_bindings(
                    case.body,
                    bindings,
                    False,
                )
            continue
        _collect_expression_alias_bindings(statement, bindings, definite)


def _match_capture_names(pattern: AST) -> frozenset[str]:
    """Return every name that a structural pattern may capture."""
    names: set[str] = set()
    for node in walk(pattern):
        if isinstance(node, (MatchAs, MatchStar)) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, MatchMapping) and node.rest is not None:
            names.add(node.rest)
    return frozenset(names)


def _scope_alias_bindings(scope: AST) -> tuple[_AliasBinding, ...]:
    """Return control-flow-aware aliases and invalidations in one scope."""
    bindings: list[_AliasBinding] = []
    if isinstance(scope, (FunctionDef, AsyncFunctionDef)):
        arguments = (
            *scope.args.posonlyargs,
            *scope.args.args,
            *scope.args.kwonlyargs,
        )
        for argument in arguments:
            bindings.append(
                _AliasBinding(
                    name=argument.arg,
                    position=(scope.lineno, -1),
                )
            )
        for optional_argument in (scope.args.vararg, scope.args.kwarg):
            if optional_argument is not None:
                bindings.append(
                    _AliasBinding(
                        name=optional_argument.arg,
                        position=(scope.lineno, -1),
                    )
                )
    _collect_scope_statement_bindings(
        cast(Sequence[AST], getattr(scope, "body", ())),
        bindings,
        True,
    )
    return tuple(sorted(bindings, key=lambda binding: binding.position))


def _contextlib_suppress_aliases(
    module_tree: AST,
    function: FunctionDef | AsyncFunctionDef,
    enclosing_functions: Sequence[FunctionDef | AsyncFunctionDef] = (),
    class_scope: ClassDef | None = None,
) -> _SuppressContextAliases:
    """Return lexical scopes used to resolve contextlib.suppress aliases."""
    scopes: tuple[AST, ...] = (
        (module_tree, class_scope, *enclosing_functions, function)
        if class_scope is not None
        else (module_tree, *enclosing_functions, function)
    )
    instance_names = frozenset(
        argument.arg
        for scope in (*enclosing_functions, function)
        for argument in (*scope.args.posonlyargs, *scope.args.args)[:1]
        if argument.arg in {"self", "cls"}
    )
    return _SuppressContextAliases(
        scopes=scopes,
        class_scope_index=1 if class_scope is not None else None,
        instance_names=instance_names,
    )


def _direct_aliases(binding: _AliasBinding) -> frozenset[_ResolvedAlias]:
    """Return the exact abstract value for a direct binding."""
    if binding.direct_kind is None:
        return frozenset()
    return frozenset(
        (
            _ResolvedAlias(
                kind=binding.direct_kind,
                exception_names=binding.direct_exception_names,
            ),
        )
    )


def _unknown_aliases() -> frozenset[_ResolvedAlias]:
    """Return the abstract value for an unproved runtime object."""
    return frozenset((_ResolvedAlias(kind=_ALIAS_UNKNOWN),))


def _resolve_alias_name(
    name: str,
    aliases: _SuppressContextAliases,
    position: tuple[int, int],
    maximum_scope_index: int,
    visiting: frozenset[tuple[int, str, tuple[int, int]]],
    expand_sequences: bool,
    include_class_scope: bool = False,
) -> frozenset[_ResolvedAlias]:
    """Resolve every possible value of one lexical name without cycles."""
    if maximum_scope_index < 0:
        if name == "open":
            return frozenset((_ResolvedAlias(kind=_ALIAS_SAFE_CONTEXT),))
        return (
            frozenset(
                (
                    _ResolvedAlias(
                        kind=_ALIAS_EXCEPTIONS,
                        exception_names=frozenset((name,)),
                    ),
                )
            )
            if name in _KNOWN_BUILTIN_EXCEPTION_NAMES
            else _unknown_aliases()
        )
    scope = aliases.scopes[maximum_scope_index]
    if isinstance(scope, ClassDef) and not include_class_scope:
        return _resolve_alias_name(
            name,
            aliases,
            _ALIAS_END_POSITION,
            maximum_scope_index - 1,
            visiting,
            expand_sequences,
            False,
        )
    named = tuple(
        binding
        for binding in _scope_alias_bindings(scope)
        if binding.name == name
    )
    local_shadow = isinstance(scope, (FunctionDef, AsyncFunctionDef)) and bool(
        named
    )
    if local_shadow:
        possibilities: frozenset[_ResolvedAlias] = frozenset()
    else:
        outer_position = (
            _ALIAS_END_POSITION if maximum_scope_index - 1 == 0 else position
        )
        possibilities = _resolve_alias_name(
            name,
            aliases,
            outer_position,
            maximum_scope_index - 1,
            visiting,
            expand_sequences,
            include_class_scope,
        )
    for binding in named:
        if binding.position >= position:
            continue
        key = (id(scope), name, binding.position)
        if key in visiting:
            continue
        if binding.direct_kind is not None:
            binding_values = _direct_aliases(binding)
            if (
                binding.imported_attribute is not None
                and not _scope_attribute_is_unmodified(
                    binding.imported_attribute,
                    scope,
                    binding.position,
                )
            ):
                binding_values |= _unknown_aliases()
        elif binding.value is None:
            binding_values = _unknown_aliases()
        else:
            binding_values = _resolve_alias_expression(
                binding.value,
                aliases,
                binding.position,
                maximum_scope_index,
                visiting | frozenset((key,)),
                expand_sequences,
                include_class_scope,
            )
            if isinstance(binding.value, (AstDict, AstList, AstSet)):
                binding_values |= _unknown_aliases()
        possibilities = (
            binding_values
            if binding.definite
            else possibilities | binding_values
        )
    return possibilities


def _class_is_unittest_test_case(aliases: _SuppressContextAliases) -> bool:
    """Return whether unittest wins before every unknown direct base."""
    if aliases.class_scope_index is None:
        return False
    class_scope = aliases.scopes[aliases.class_scope_index]
    if not isinstance(class_scope, ClassDef) or not class_scope.bases:
        return False
    values = _resolve_alias_expression(
        class_scope.bases[0],
        aliases,
        _node_position(class_scope),
        aliases.class_scope_index - 1,
    )
    return bool(values) and all(
        value.kind == _ALIAS_UNITTEST_TEST_CASE for value in values
    )


def _scope_runtime_nodes(scope: AST) -> Iterable[AST]:
    """Yield nodes evaluated in one scope without nested scope bodies."""
    pending = list(cast(Sequence[AST], getattr(scope, "body", ())))
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef, Lambda)):
            continue
        pending.extend(iter_child_nodes(node))


def _attribute_is_unmodified(
    name: str,
    aliases: _SuppressContextAliases,
    position: tuple[int, int],
) -> bool:
    """Return whether no lexical path may replace a resolved attribute."""
    for index, scope in enumerate(aliases.scopes):
        limit = (
            position
            if index == len(aliases.scopes) - 1
            else _ALIAS_END_POSITION
        )
        if not _scope_attribute_is_unmodified(name, scope, limit):
            return False
    return True


def _scope_attribute_is_unmodified(
    name: str,
    scope: AST,
    limit: tuple[int, int],
) -> bool:
    """Return whether a scope has no prior mutation for an attribute."""
    return not any(
        _mutation_may_target(_attribute_mutation(node), name)
        and _node_position(node) < limit
        for node in _scope_runtime_nodes(scope)
    )


def _mutation_may_target(
    mutation: _AttributeMutation | None,
    name: str,
) -> bool:
    """Return whether one mutation may replace an attribute name."""
    return mutation is not None and (
        mutation.names is None or name in mutation.names
    )


def _attribute_mutation(node: AST) -> _AttributeMutation | None:
    """Return attribute names mutated by one syntax node."""
    if isinstance(node, Attribute) and not isinstance(node.ctx, Load):
        return _AttributeMutation(
            names=(
                None if node.attr == "__dict__" else frozenset((node.attr,))
            ),
            owners=(node.value,),
        )
    if (
        isinstance(node, Subscript)
        and not isinstance(node.ctx, Load)
        and _is_attribute_namespace(node.value)
    ):
        owner = _attribute_namespace_owner(node.value)
        return _mutation_from_name(
            node.slice,
            owners=(owner,) if owner is not None else None,
        )
    if isinstance(node, Call):
        return _call_attribute_mutation(node)
    return None


def _call_attribute_mutation(call: Call) -> _AttributeMutation | None:
    """Return attribute names mutated by one recognized runtime call."""
    if isinstance(call.func, Name) and call.func.id in {"delattr", "setattr"}:
        return (
            _mutation_from_name(call.args[1], owners=(call.args[0],))
            if len(call.args) >= 2
            else _AttributeMutation(names=None, owners=None)
        )
    if isinstance(call.func, Name) and call.func.id == "patch":
        return (
            _mutation_from_name(call.args[0], dotted=True, owners=None)
            if call.args
            else _AttributeMutation(names=None, owners=None)
        )
    if not isinstance(call.func, Attribute):
        return None
    method = call.func.attr
    if method in {"__delattr__", "__setattr__"}:
        object_method = (
            isinstance(call.func.value, Name)
            and call.func.value.id == "object"
        )
        index = 1 if object_method else 0
        owner = (
            call.args[0] if object_method and call.args else call.func.value
        )
        return (
            _mutation_from_name(call.args[index], owners=(owner,))
            if len(call.args) > index
            else _AttributeMutation(names=None, owners=(owner,))
        )
    if method in {"delattr", "setattr"}:
        if len(call.args) >= 3:
            return _mutation_from_name(
                call.args[1],
                owners=(call.args[0],),
            )
        if len(call.args) >= 2 and isinstance(call.args[0], Constant):
            return _mutation_from_name(
                call.args[0],
                dotted=True,
                owners=None,
            )
        return (
            _mutation_from_name(
                call.args[1],
                owners=(call.args[0],),
            )
            if len(call.args) >= 2
            else _AttributeMutation(names=None, owners=None)
        )
    if method == "object":
        return (
            _mutation_from_name(
                call.args[1],
                owners=(call.args[0],),
            )
            if len(call.args) >= 2
            else _AttributeMutation(names=None, owners=None)
        )
    if method == "multiple":
        if any(keyword.arg is None for keyword in call.keywords):
            return _AttributeMutation(names=None, owners=None)
        return _AttributeMutation(
            names=frozenset(
                cast(str, keyword.arg) for keyword in call.keywords
            ),
            owners=(call.args[0],) if call.args else None,
        )
    if not _is_attribute_namespace(call.func.value):
        return None
    namespace_owner = _attribute_namespace_owner(call.func.value)
    owners = (namespace_owner,) if namespace_owner is not None else None
    if method in {"__delitem__", "__setitem__", "pop", "setdefault"}:
        return (
            _mutation_from_name(call.args[0], owners=owners)
            if call.args
            else _AttributeMutation(names=None, owners=owners)
        )
    if method in {"clear", "popitem"}:
        return _AttributeMutation(names=None, owners=owners)
    if method != "update":
        return None
    names = {
        keyword.arg for keyword in call.keywords if keyword.arg is not None
    }
    if any(keyword.arg is None for keyword in call.keywords):
        return _AttributeMutation(names=None, owners=owners)
    for argument in call.args:
        argument_names = _mapping_attribute_names(argument)
        if argument_names is None:
            return _AttributeMutation(names=None, owners=owners)
        names.update(argument_names)
    return _AttributeMutation(names=frozenset(names), owners=owners)


def _mutation_from_name(
    expression: AST,
    dotted: bool = False,
    owners: tuple[AST, ...] | None = None,
) -> _AttributeMutation:
    """Return an exact attribute name or a dynamic mutation target."""
    if isinstance(expression, Constant) and isinstance(expression.value, str):
        name = (
            expression.value.rsplit(".", 1)[-1] if dotted else expression.value
        )
        return _AttributeMutation(names=frozenset((name,)), owners=owners)
    return _AttributeMutation(names=None, owners=owners)


def _mapping_attribute_names(expression: AST) -> frozenset[str] | None:
    """Return exact string keys from a literal attribute mapping."""
    if not isinstance(expression, AstDict):
        return None
    names: set[str] = set()
    for key in expression.keys:
        if (
            key is None
            or not isinstance(key, Constant)
            or not isinstance(key.value, str)
        ):
            return None
        names.add(key.value)
    return frozenset(names)


def _is_attribute_namespace(expression: AST) -> bool:
    """Return whether an expression denotes an object's attribute mapping."""
    if isinstance(expression, Attribute):
        return expression.attr == "__dict__"
    return isinstance(expression, Call) and (
        isinstance(expression.func, Name)
        and expression.func.id in {"globals", "locals", "vars"}
        or isinstance(expression.func, Attribute)
        and expression.func.attr in {"globals", "locals", "vars"}
    )


def _attribute_namespace_owner(expression: AST) -> AST | None:
    """Return the object whose attribute mapping an expression denotes."""
    if isinstance(expression, Attribute) and expression.attr == "__dict__":
        return expression.value
    if (
        isinstance(expression, Call)
        and isinstance(expression.func, Name)
        and expression.func.id == "vars"
        and len(expression.args) == 1
    ):
        return expression.args[0]
    return None


def _class_inherits_unmodified_unittest_attribute(
    name: str,
    aliases: _SuppressContextAliases,
    position: tuple[int, int],
) -> bool:
    """Return whether a standard TestCase attribute is not overridden."""
    if not _class_is_unittest_test_case(aliases):
        return False
    class_scope_index = aliases.class_scope_index
    assert class_scope_index is not None
    if any(
        binding.name == name
        for binding in _scope_alias_bindings(aliases.scopes[class_scope_index])
    ):
        return False
    return _attribute_is_unmodified(name, aliases, position)


def _resolve_alias_expression(
    expression: AST,
    aliases: _SuppressContextAliases,
    position: tuple[int, int] | None = None,
    maximum_scope_index: int | None = None,
    visiting: frozenset[tuple[int, str, tuple[int, int]]] = frozenset(),
    expand_sequences: bool = False,
    include_class_scope: bool = False,
) -> frozenset[_ResolvedAlias]:
    """Resolve every possible module, callable, or exception alias."""
    resolved_position = position or _node_position(expression)
    scope_index = (
        len(aliases.scopes) - 1
        if maximum_scope_index is None
        else maximum_scope_index
    )
    if isinstance(expression, Name):
        return _resolve_alias_name(
            expression.id,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            expand_sequences,
            include_class_scope,
        )
    if isinstance(expression, NamedExpr):
        return _resolve_alias_expression(
            expression.value,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            expand_sequences,
            include_class_scope,
        )
    if isinstance(expression, Attribute):
        if (
            isinstance(expression.value, Name)
            and expression.value.id in aliases.instance_names
            and aliases.class_scope_index is not None
        ):
            if not _attribute_is_unmodified(
                expression.attr,
                aliases,
                resolved_position,
            ):
                return _unknown_aliases()
            if (
                expression.attr in _UNITTEST_CHECK_CONTEXT_NAMES
                and _class_inherits_unmodified_unittest_attribute(
                    expression.attr,
                    aliases,
                    resolved_position,
                )
            ):
                return frozenset((_ResolvedAlias(kind=_ALIAS_CHECK_CONTEXT),))
            if (
                expression.attr in _UNITTEST_SAFE_CONTEXT_NAMES
                and _class_inherits_unmodified_unittest_attribute(
                    expression.attr,
                    aliases,
                    resolved_position,
                )
            ):
                return frozenset((_ResolvedAlias(kind=_ALIAS_SAFE_CONTEXT),))
            return _resolve_alias_name(
                expression.attr,
                aliases,
                _ALIAS_END_POSITION,
                aliases.class_scope_index,
                visiting,
                expand_sequences,
                True,
            )
        owners = _resolve_alias_expression(
            expression.value,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            expand_sequences,
            include_class_scope,
        )
        resolved: set[_ResolvedAlias] = set()
        attribute_unmodified = _attribute_is_unmodified(
            expression.attr,
            aliases,
            resolved_position,
        )
        for owner in owners:
            if not attribute_unmodified:
                resolved.add(_ResolvedAlias(kind=_ALIAS_UNKNOWN))
                continue
            if (
                owner.kind == _ALIAS_CONTEXTLIB
                and expression.attr == "suppress"
            ):
                resolved.add(_ResolvedAlias(kind=_ALIAS_SUPPRESS))
            elif (
                owner.kind == _ALIAS_CONTEXTLIB
                and expression.attr == "nullcontext"
            ):
                resolved.add(_ResolvedAlias(kind=_ALIAS_NULLCONTEXT))
            elif (
                owner.kind == _ALIAS_PYTEST
                and expression.attr in _PYTEST_CHECK_CONTEXT_NAMES
            ):
                resolved.add(_ResolvedAlias(kind=_ALIAS_CHECK_CONTEXT))
            elif (
                owner.kind == _ALIAS_TEMPFILE
                and expression.attr in _TEMPFILE_SAFE_CONTEXT_NAMES
            ):
                resolved.add(_ResolvedAlias(kind=_ALIAS_SAFE_CONTEXT))
            elif (
                owner.kind == _ALIAS_UNITTEST
                and expression.attr in _UNITTEST_TEST_CASE_NAMES
            ):
                resolved.add(_ResolvedAlias(kind=_ALIAS_UNITTEST_TEST_CASE))
            elif owner.kind == _ALIAS_UNITTEST and expression.attr == "mock":
                resolved.add(_ResolvedAlias(kind=_ALIAS_UNITTEST_MOCK))
            elif (
                owner.kind == _ALIAS_UNITTEST_MOCK
                and expression.attr == "patch"
            ):
                resolved.add(_ResolvedAlias(kind=_ALIAS_UNITTEST_PATCH))
            elif owner.kind == _ALIAS_UNITTEST_PATCH and expression.attr in {
                "dict",
                "multiple",
                "object",
            }:
                resolved.add(_ResolvedAlias(kind=_ALIAS_SAFE_CONTEXT))
            elif (
                owner.kind == _ALIAS_BUILTINS
                and expression.attr in _KNOWN_BUILTIN_EXCEPTION_NAMES
            ):
                resolved.add(
                    _ResolvedAlias(
                        kind=_ALIAS_EXCEPTIONS,
                        exception_names=frozenset((expression.attr,)),
                    )
                )
            else:
                resolved.add(_ResolvedAlias(kind=_ALIAS_UNKNOWN))
        return frozenset(resolved)
    if isinstance(expression, Starred):
        return _resolve_alias_expression(
            expression.value,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            True,
            include_class_scope,
        )
    if isinstance(expression, IfExp):
        if isinstance(expression.test, Constant):
            branch = (
                expression.body
                if bool(expression.test.value)
                else expression.orelse
            )
            return _resolve_alias_expression(
                branch,
                aliases,
                resolved_position,
                scope_index,
                visiting,
                expand_sequences,
                include_class_scope,
            )
        return _resolve_alias_expression(
            expression.body,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            expand_sequences,
            include_class_scope,
        ) | _resolve_alias_expression(
            expression.orelse,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            expand_sequences,
            include_class_scope,
        )
    if isinstance(expression, Subscript):
        values = _iterable_alias_values(expression.value)
        if (
            values is not None
            and isinstance(expression.slice, Constant)
            and isinstance(expression.slice.value, int)
        ):
            try:
                selected = values[expression.slice.value]
            except IndexError:
                return _unknown_aliases()
            return _resolve_alias_expression(
                selected,
                aliases,
                resolved_position,
                scope_index,
                visiting,
                expand_sequences,
                include_class_scope,
            )
        return _resolve_alias_expression(
            expression.value,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            True,
            include_class_scope,
        )
    if expand_sequences and isinstance(
        expression,
        (AstList, AstSet, AstTuple),
    ):
        return frozenset().union(
            *(
                _resolve_alias_expression(
                    item,
                    aliases,
                    resolved_position,
                    scope_index,
                    visiting,
                    True,
                    include_class_scope,
                )
                for item in expression.elts
            )
        )
    if (
        expand_sequences
        and isinstance(expression, BinOp)
        and type(expression.op).__name__ == "Add"
    ):
        return _resolve_alias_expression(
            expression.left,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            True,
            include_class_scope,
        ) | _resolve_alias_expression(
            expression.right,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            True,
            include_class_scope,
        )
    if expand_sequences and isinstance(
        expression,
        (GeneratorExp, ListComp, SetComp),
    ):
        if (
            len(expression.generators) == 1
            and isinstance(expression.generators[0].target, Name)
            and isinstance(expression.elt, Name)
            and expression.elt.id == expression.generators[0].target.id
        ):
            return _resolve_alias_expression(
                expression.generators[0].iter,
                aliases,
                resolved_position,
                scope_index,
                visiting,
                True,
                include_class_scope,
            )
        values = _iterable_alias_values(expression) or (expression.elt,)
        return frozenset().union(
            *(
                _resolve_alias_expression(
                    value,
                    aliases,
                    resolved_position,
                    scope_index,
                    visiting,
                    True,
                    include_class_scope,
                )
                for value in values
            )
        )
    if isinstance(expression, Call):
        targets = _resolve_alias_expression(
            expression.func,
            aliases,
            resolved_position,
            scope_index,
            visiting,
            expand_sequences,
            include_class_scope,
        )
        if targets and all(
            target.kind
            in {
                _ALIAS_NULLCONTEXT,
                _ALIAS_SAFE_CONTEXT,
                _ALIAS_UNITTEST_PATCH,
            }
            for target in targets
        ):
            return frozenset(
                (_ResolvedAlias(kind=_ALIAS_SAFE_CONTEXT_INSTANCE),)
            )
        return _unknown_aliases()
    return _unknown_aliases()


def _is_assertion_safe_context(
    expression: AST,
    aliases: _SuppressContextAliases,
) -> bool:
    """Return whether a context cannot hide an enclosed failed check."""
    if not isinstance(expression, Call):
        targets = _resolve_alias_expression(expression, aliases)
        return bool(targets) and all(
            target.kind == _ALIAS_SAFE_CONTEXT_INSTANCE for target in targets
        )
    targets = _resolve_alias_expression(expression.func, aliases)
    if not targets:
        return False
    for target in targets:
        if target.kind in {
            _ALIAS_NULLCONTEXT,
            _ALIAS_SAFE_CONTEXT,
            _ALIAS_UNITTEST_PATCH,
        }:
            continue
        if target.kind == _ALIAS_CHECK_CONTEXT:
            if not expression.args:
                return False
            continue
        if target.kind != _ALIAS_SUPPRESS:
            return False
        if expression.keywords or not all(
            _is_proven_unrelated_exception(argument, aliases)
            for argument in expression.args
        ):
            return False
    return True


def _is_proven_unrelated_exception(
    expression: AST,
    aliases: _SuppressContextAliases,
) -> bool:
    """Return whether an argument cannot match a failed assertion."""
    resolved = _resolve_alias_expression(expression, aliases)
    if not resolved:
        return (
            isinstance(expression, Starred)
            and _iterable_alias_values(expression.value) == ()
        )
    return all(
        value.kind == _ALIAS_EXCEPTIONS
        and bool(value.exception_names)
        and not value.exception_names & _CHECK_FAILURE_EXCEPTION_NAMES
        for value in resolved
    )


def _is_recognized_check_context(
    expression: AST,
    aliases: _SuppressContextAliases,
) -> bool:
    """Return whether a context is always a real pytest/unittest check."""
    if not isinstance(expression, Call) or not expression.args:
        return False
    targets = _resolve_alias_expression(expression.func, aliases)
    return bool(targets) and all(
        target.kind == _ALIAS_CHECK_CONTEXT for target in targets
    )


def _validate_prohibited_test_constructs(
    tree: AST,
    node_id: str,
    module_tree: AST | None = None,
) -> None:
    aliases: dict[str, str] = {}
    for statement in getattr(module_tree, "body", ()):
        if not isinstance(statement, ImportFrom):
            continue
        for item in statement.names:
            imported = item.name.rsplit(".", 1)[-1]
            if imported in _PROHIBITED_TEST_SYMBOLS:
                aliases[item.asname or imported] = imported
    for node in walk(tree):
        prohibited: str | None = None
        if isinstance(node, ImportFrom):
            prohibited = next(
                (
                    item.name.rsplit(".", 1)[-1]
                    for item in node.names
                    if item.name.rsplit(".", 1)[-1] in _PROHIBITED_TEST_SYMBOLS
                ),
                None,
            )
        elif (
            isinstance(node, Name)
            and isinstance(node.ctx, Load)
            and (node.id in _PROHIBITED_TEST_SYMBOLS or node.id in aliases)
        ):
            prohibited = aliases.get(node.id, node.id)
        elif (
            isinstance(node, Attribute)
            and node.attr in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.attr
        elif (
            isinstance(node, Subscript)
            and isinstance(node.slice, Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.slice.value
        elif (
            isinstance(node, Call)
            and isinstance(node.func, Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.args[1].value
        elif (
            isinstance(node, Call)
            and isinstance(node.func, Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.args[0].value
        if prohibited is None:
            continue
        category = (
            "pytest control"
            if prohibited in _PROHIBITED_TEST_CONTROLS
            else "execution trick"
        )
        raise AcceptanceVerificationError(
            f"acceptance test uses a prohibited {category} ({prohibited}):"
            f" {node_id}"
        )


def _check_sequence(
    statements: Sequence[AST],
    initial_states: frozenset[bool],
    aliases: _SuppressContextAliases,
    checks_suppressed: bool = False,
) -> _CheckPaths:
    next_states = set(initial_states)
    return_states: set[bool] = set()
    break_states: set[bool] = set()
    continue_states: set[bool] = set()
    failed_check_states: set[bool] = set()
    for statement in statements:
        if not next_states:
            break
        statement_paths = _merge_check_paths(
            _check_statement(
                statement,
                state,
                aliases,
                checks_suppressed,
            )
            for state in next_states
        )
        next_states = set(statement_paths.next_states)
        return_states.update(statement_paths.return_states)
        break_states.update(statement_paths.break_states)
        continue_states.update(statement_paths.continue_states)
        failed_check_states.update(statement_paths.failed_check_states)
    return _CheckPaths(
        next_states=frozenset(next_states),
        return_states=frozenset(return_states),
        break_states=frozenset(break_states),
        continue_states=frozenset(continue_states),
        failed_check_states=frozenset(failed_check_states),
    )


def _check_statement(
    statement: AST,
    checked: bool,
    aliases: _SuppressContextAliases,
    checks_suppressed: bool,
) -> _CheckPaths:
    if isinstance(statement, (FunctionDef, AsyncFunctionDef, ClassDef)):
        return _CheckPaths(next_states=frozenset({checked}))
    if isinstance(statement, Assert):
        if checks_suppressed:
            return _CheckPaths(next_states=frozenset({checked}))
        if isinstance(statement.test, Constant):
            if bool(statement.test.value):
                return _CheckPaths(next_states=frozenset({checked}))
            return _CheckPaths()
        meaningful = _is_check(statement)
        state = checked or meaningful
        return _CheckPaths(
            next_states=frozenset({state}),
            failed_check_states=(
                frozenset({checked}) if meaningful else frozenset()
            ),
        )
    if isinstance(statement, Return):
        return _CheckPaths(return_states=frozenset({checked}))
    if isinstance(statement, Raise):
        if checks_suppressed:
            return _CheckPaths(next_states=frozenset({checked}))
        return _CheckPaths(failed_check_states=frozenset({checked}))
    if isinstance(statement, Break):
        return _CheckPaths(break_states=frozenset({checked}))
    if isinstance(statement, Continue):
        return _CheckPaths(continue_states=frozenset({checked}))
    if isinstance(statement, If):
        branches: tuple[Sequence[AST], ...]
        if isinstance(statement.test, Constant):
            branches = (
                (
                    statement.body
                    if bool(statement.test.value)
                    else statement.orelse
                ),
            )
        else:
            branches = (statement.body, statement.orelse)
        return _merge_check_paths(
            _check_sequence(
                branch,
                frozenset({checked}),
                aliases,
                checks_suppressed,
            )
            for branch in branches
        )
    if isinstance(statement, (For, AsyncFor)):
        body = _check_sequence(
            statement.body,
            frozenset({checked}),
            aliases,
            checks_suppressed,
        )
        natural_states = body.next_states | body.continue_states
        if not _statically_nonempty_iter(statement.iter):
            natural_states |= frozenset({checked})
        orelse = _check_sequence(
            statement.orelse,
            natural_states,
            aliases,
            checks_suppressed,
        )
        return _CheckPaths(
            next_states=orelse.next_states | body.break_states,
            return_states=body.return_states | orelse.return_states,
            break_states=orelse.break_states,
            continue_states=orelse.continue_states,
            failed_check_states=(
                body.failed_check_states | orelse.failed_check_states
            ),
        )
    if isinstance(statement, While):
        if isinstance(statement.test, Constant) and not bool(
            statement.test.value
        ):
            return _check_sequence(
                statement.orelse,
                frozenset({checked}),
                aliases,
                checks_suppressed,
            )
        body = _check_sequence(
            statement.body,
            frozenset({checked}),
            aliases,
            checks_suppressed,
        )
        if isinstance(statement.test, Constant):
            natural_states = frozenset()
        else:
            natural_states = body.next_states | body.continue_states
            natural_states |= frozenset({checked})
        orelse = _check_sequence(
            statement.orelse,
            natural_states,
            aliases,
            checks_suppressed,
        )
        return _CheckPaths(
            next_states=orelse.next_states | body.break_states,
            return_states=body.return_states | orelse.return_states,
            break_states=orelse.break_states,
            continue_states=orelse.continue_states,
            failed_check_states=(
                body.failed_check_states | orelse.failed_check_states
            ),
        )
    if isinstance(statement, (With, AsyncWith)):
        suppressing = checks_suppressed or any(
            not _is_assertion_safe_context(item.context_expr, aliases)
            for item in statement.items
        )
        recognized_check = not suppressing and any(
            _is_recognized_check_context(item.context_expr, aliases)
            for item in statement.items
        )
        body = _check_sequence(
            statement.body,
            frozenset({checked or recognized_check}),
            aliases,
            suppressing,
        )
        if not recognized_check:
            return body
        return _merge_check_paths(
            (
                body,
                _CheckPaths(failed_check_states=frozenset({checked})),
            )
        )
    if isinstance(statement, Try):
        body = _check_sequence(
            statement.body,
            frozenset({checked}),
            aliases,
            checks_suppressed,
        )
        normal = _check_sequence(
            statement.orelse,
            body.next_states,
            aliases,
            checks_suppressed,
        )
        handlers = _merge_check_paths(
            _check_sequence(
                handler.body,
                frozenset({checked}),
                aliases,
                checks_suppressed,
            )
            for handler in statement.handlers
        )
        combined = _merge_check_paths(
            (
                _CheckPaths(
                    next_states=normal.next_states,
                    return_states=body.return_states | normal.return_states,
                    break_states=body.break_states | normal.break_states,
                    continue_states=(
                        body.continue_states | normal.continue_states
                    ),
                    failed_check_states=(
                        body.failed_check_states | normal.failed_check_states
                    ),
                ),
                handlers,
            )
        )
        return _apply_finally(
            combined,
            statement.finalbody,
            aliases,
            checks_suppressed,
        )
    if isinstance(statement, Match):
        paths = [
            _check_sequence(
                case.body,
                frozenset({checked}),
                aliases,
                checks_suppressed,
            )
            for case in statement.cases
        ]
        if not any(
            isinstance(case.pattern, MatchAs)
            and case.pattern.pattern is None
            and case.guard is None
            for case in statement.cases
        ):
            paths.append(_CheckPaths(next_states=frozenset({checked})))
        return _merge_check_paths(paths)
    meaningful = not checks_suppressed and _statement_executes_check(statement)
    return _CheckPaths(
        next_states=frozenset({checked or meaningful}),
        failed_check_states=(
            frozenset({checked}) if meaningful else frozenset()
        ),
    )


def _statement_executes_check(statement: AST) -> bool:
    if isinstance(statement, Expr):
        return _is_check(statement.value)
    return _is_check(statement)


def _merge_check_paths(paths: Iterable[_CheckPaths]) -> _CheckPaths:
    next_states: set[bool] = set()
    return_states: set[bool] = set()
    break_states: set[bool] = set()
    continue_states: set[bool] = set()
    failed_check_states: set[bool] = set()
    for path in paths:
        next_states.update(path.next_states)
        return_states.update(path.return_states)
        break_states.update(path.break_states)
        continue_states.update(path.continue_states)
        failed_check_states.update(path.failed_check_states)
    return _CheckPaths(
        next_states=frozenset(next_states),
        return_states=frozenset(return_states),
        break_states=frozenset(break_states),
        continue_states=frozenset(continue_states),
        failed_check_states=frozenset(failed_check_states),
    )


def _apply_finally(
    paths: _CheckPaths,
    statements: Sequence[AST],
    aliases: _SuppressContextAliases,
    checks_suppressed: bool,
) -> _CheckPaths:
    if not statements:
        return paths
    transformed: list[_CheckPaths] = []
    for outcome, states in (
        ("next", paths.next_states),
        ("return", paths.return_states),
        ("break", paths.break_states),
        ("continue", paths.continue_states),
        ("failed_check", paths.failed_check_states),
    ):
        for state in states:
            final = _check_sequence(
                statements,
                frozenset({state}),
                aliases,
                checks_suppressed,
            )
            if outcome == "next":
                preserved = _CheckPaths(next_states=final.next_states)
            elif outcome == "return":
                preserved = _CheckPaths(return_states=final.next_states)
            elif outcome == "break":
                preserved = _CheckPaths(break_states=final.next_states)
            elif outcome == "continue":
                preserved = _CheckPaths(continue_states=final.next_states)
            else:
                preserved = _CheckPaths(failed_check_states=final.next_states)
            transformed.extend(
                (
                    preserved,
                    _CheckPaths(
                        return_states=final.return_states,
                        break_states=final.break_states,
                        continue_states=final.continue_states,
                        failed_check_states=final.failed_check_states,
                    ),
                )
            )
    return _merge_check_paths(transformed)


def _expression_is_dynamic(expression: AST) -> bool:
    if (
        isinstance(expression, Compare)
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], (Eq, Is))
        and len(expression.comparators) == 1
        and _same_expression(expression.left, expression.comparators[0])
    ):
        return False
    return any(
        isinstance(node, (Name, Call, Attribute, Subscript))
        for node in walk(expression)
    )


def _same_expression(left: AST, right: AST) -> bool:
    return dump(left, include_attributes=False) == dump(
        right,
        include_attributes=False,
    )


def _statically_nonempty_iter(expression: AST) -> bool:
    if (
        isinstance(expression, Call)
        and isinstance(expression.func, Name)
        and expression.func.id in {"enumerate", "iter", "reversed"}
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return _statically_nonempty_iter(expression.args[0])
    if isinstance(expression, (AstList, AstSet, AstTuple)):
        return bool(expression.elts)
    if isinstance(expression, AstDict):
        return bool(expression.keys)
    return (
        isinstance(expression, Constant)
        and isinstance(expression.value, (str, tuple, frozenset))
        and bool(expression.value)
    )


def _type_fixture_path(raw: str, root: Path) -> Path:
    posix = PurePosixPath(raw)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or "\\" in raw
        or len(posix.parts) < 3
        or posix.parts[:2] != ("tests", "input_type_contracts")
        or not raw.endswith(".py")
    ):
        raise AcceptanceVerificationError(
            f"type fixture path is outside its tracked directory: {raw}"
        )
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to((root / "tests" / "input_type_contracts").resolve())
    except ValueError as exc:
        raise AcceptanceVerificationError(
            f"type fixture path escapes its tracked directory: {raw}"
        ) from exc
    return path


def _verify_digest(
    value: object,
    raw_digest: object,
    expected_digest: str,
    label: str,
) -> None:
    digest = _nonempty_string(raw_digest, f"{label} SHA-256")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise AcceptanceVerificationError(
            f"{label} SHA-256 must be lowercase hexadecimal"
        )
    canonical = dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    calculated = sha256(canonical).hexdigest()
    if digest != calculated or digest != expected_digest:
        raise AcceptanceVerificationError(
            f"{label} digest mismatch: declared={digest},"
            f" calculated={calculated}"
        )


def _exact_keys(
    payload: dict[str, object], expected: set[str], label: str
) -> None:
    if set(payload) != expected:
        raise AcceptanceVerificationError(
            f"{label} has invalid keys: expected={sorted(expected)}, "
            f"observed={sorted(payload)}"
        )


def _phase(value: object, label: str) -> int:
    if type(value) is not int or value < _MIN_PHASE or value > _MAX_PHASE:
        raise AcceptanceVerificationError(
            f"{label} must be an integer from {_MIN_PHASE} to {_MAX_PHASE}"
        )
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AcceptanceVerificationError(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_number(value: object, label: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise AcceptanceVerificationError(f"{label} must be positive")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceVerificationError(
            f"{label} must be a non-empty string"
        )
    return value


def _unambiguous_outcome(
    value: str,
    label: str,
    key: tuple[str, str],
) -> None:
    normalized = f" {value.lower()} "
    if " or " in normalized or " unless " in normalized or "|" in value:
        raise AcceptanceVerificationError(
            f"failure {label} is ambiguous for {key}: {value}"
        )


def _string_list(value: object, label: str) -> tuple[str, ...]:
    result = _string_list_allow_empty(value, label)
    if not result:
        raise AcceptanceVerificationError(f"{label} must not be empty")
    return result


def _string_list_allow_empty(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AcceptanceVerificationError(f"{label} must be a string list")
    return tuple(cast(list[str], value))


def _node_id(value: object) -> str:
    node_id = _nonempty_string(value, "pytest node ID")
    raw_path, separator, test_name = node_id.partition("::")
    if not separator or not raw_path.endswith(".py") or not test_name:
        raise AcceptanceVerificationError(f"invalid pytest node ID: {node_id}")
    posix = PurePosixPath(raw_path)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or "\\" in raw_path
        or not posix.parts
        or posix.parts[0] != "tests"
    ):
        raise AcceptanceVerificationError(
            f"pytest node path must be a safe tracked test path: {node_id}"
        )
    return node_id


def _unique(values: Iterable[str], label: str) -> None:
    items = tuple(values)
    duplicates = sorted(item for item in set(items) if items.count(item) > 1)
    if duplicates:
        raise AcceptanceVerificationError(f"duplicate {label}: {duplicates}")


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Collect and execute every active structured-input acceptance "
            "node without skips, xfails, or deselection."
        )
    )
    parser.add_argument("--through-phase", required=True, type=int)
    parser.add_argument(
        "--manifest", type=Path, default=default_manifest_path()
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="execute only the exact current behavioral node inventory",
    )
    return parser.parse_args()


def main() -> int:
    """Run acceptance verification from the command line."""
    args = _parse_args()
    try:
        if args.runtime_only:
            if args.through_phase != 4:
                raise AcceptanceVerificationError(
                    "--runtime-only requires --through-phase 4"
                )
            manifest = verify_current_runtime(
                args.manifest,
                repo_root=args.repo_root,
            )
        else:
            manifest = verify_acceptance(
                args.manifest,
                repo_root=args.repo_root,
                through_phase=args.through_phase,
            )
    except (AcceptanceVerificationError, TimeoutExpired) as exc:
        print(f"structured-input acceptance failed: {exc}", file=stderr)
        return 1
    if args.runtime_only:
        active_count = len(_current_runtime_node_ids(manifest.nodes))
        instance_count = len(_current_runtime_instance_ids(manifest))
    else:
        active_count = len(manifest.active_nodes(args.through_phase))
        instance_count = len(
            manifest.active_pytest_instances(args.through_phase)
        )
    print(
        "structured-input acceptance passed: "
        f"through_phase={args.through_phase} nodes={active_count}"
        f" instances={instance_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
