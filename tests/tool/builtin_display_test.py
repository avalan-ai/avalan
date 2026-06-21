from datetime import UTC, datetime
from json import dumps
from pathlib import Path
from typing import Any, cast
from unittest import skipIf
from uuid import uuid4

from avalan.entities import (
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolManagerSettings,
)
from avalan.memory.manager import MemoryManager
from avalan.memory.permanent import Memory, MemoryType, PermanentMemoryStore
from avalan.tool import ToolSet
from avalan.tool.browser import (
    HAS_BROWSER_DEPENDENCIES,
    BrowserTool,
    BrowserToolSettings,
)
from avalan.tool.builtin_display import (
    _joined_limited_strings,
    project_ast_grep_tool_display,
    project_browser_open_tool_display,
    project_calculator_tool_display,
    project_code_run_tool_display,
    project_graph_tool_display,
    project_mcp_call_tool_display,
    project_memory_tool_display,
    project_search_tool_display,
)
from avalan.tool.code import CodeToolSet
from avalan.tool.display import ToolDisplayProjection
from avalan.tool.graph import GraphToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.math import MathToolSet
from avalan.tool.mcp import McpToolSet
from avalan.tool.memory import MemoryToolSet
from avalan.tool.search_engine import SearchEngineTool

REPO_ROOT = Path(__file__).resolve().parents[2]

_CALL_INTENTS: dict[str, tuple[dict[str, Any], str]] = {
    "math.calculator": ({"expression": "2 + 2"}, "calculate"),
    "mcp.call": (
        {
            "uri": "https://mcp.example.test",
            "name": "remote.search",
            "arguments": {"query": "status"},
        },
        "call",
    ),
    "code.run": ({"code": "def run(): return 1"}, "run"),
    "code.search.ast.grep": (
        {"pattern": "print($A)", "lang": "py", "paths": ["src"]},
        "search",
    ),
    "search": ({"query": "weather", "engine": "google"}, "search"),
    "memory.message.read": ({"search": "name"}, "search"),
    "memory.read": (
        {"namespace": "docs", "search": "architecture"},
        "search",
    ),
    "memory.list": ({"namespace": "docs"}, "list"),
    "memory.stores": ({}, "list"),
    "graph.pie": (
        {"labels": ["A", "B"], "values": [1, 2]},
        "render",
    ),
    "graph.bar": (
        {"categories": ["A"], "values": [1]},
        "render",
    ),
    "graph.line": (
        {"x_labels": ["Q1"], "values": [1]},
        "render",
    ),
    "graph.scatter": (
        {"x": [1], "y": [2]},
        "render",
    ),
    "graph.histogram": (
        {"values": [1, 2, 3], "bins": 3},
        "render",
    ),
}


def test_remaining_builtin_descriptors_expose_call_projection() -> None:
    manager = _manager()

    for name, (arguments, action) in _CALL_INTENTS.items():
        descriptor = manager.describe_tool(name)

        assert descriptor is not None
        assert descriptor.display_projector is not None
        projection = descriptor.project_display(
            ToolCall(id=f"{name}-1", name=name, arguments=arguments)
        )

        assert isinstance(projection, ToolDisplayProjection)
        assert projection.label == name
        assert projection.action == action


@skipIf(not HAS_BROWSER_DEPENDENCIES, "browser dependencies not installed")
def test_browser_descriptor_exposes_call_projection_when_available() -> None:
    manager = ToolManager.create_instance(
        available_toolsets=[
            ToolSet(
                namespace="browser",
                tools=[BrowserTool(BrowserToolSettings(), object())],
            )
        ],
        settings=ToolManagerSettings(),
    )
    descriptor = manager.describe_tool("browser.open")

    assert descriptor is not None
    assert descriptor.display_projector is not None
    projection = descriptor.project_display(
        ToolCall(
            id="browser-open-1",
            name="browser.open",
            arguments={"url": "https://example.test/page"},
        )
    )

    assert isinstance(projection, ToolDisplayProjection)
    assert projection.action == "open"
    assert projection.label == "browser.open"


def test_browser_projection_does_not_require_browser_runtime() -> None:
    projection = project_browser_open_tool_display(
        call=ToolCall(
            id="browser-open-1",
            name="browser.open",
            arguments={
                "url": (
                    "https://user:password@example.test/page"
                    "?token=secret#fragment"
                )
            },
        ),
        settings=BrowserToolSettings(engine="webkit", search=True),
    )
    payload = _payload_text(projection)

    assert projection.action == "open"
    assert projection.redacted
    assert "password" not in payload
    assert "secret" not in payload
    assert "token" not in payload
    assert "https://example.test/page" in payload


def test_browser_projection_handles_empty_and_malformed_urls() -> None:
    missing_projection = project_browser_open_tool_display(
        call=ToolCall(id="browser-open-1", name="browser.open")
    )
    malformed_projection = project_browser_open_tool_display(
        call=ToolCall(
            id="browser-open-2",
            name="browser.open",
            arguments={"url": "http://[::1"},
        )
    )
    bad_port_projection = project_browser_open_tool_display(
        call=ToolCall(
            id="browser-open-3",
            name="browser.open",
            arguments={"url": "https://example.test:abc/page?token=secret"},
        )
    )
    relative_query_projection = project_browser_open_tool_display(
        call=ToolCall(
            id="browser-open-4",
            name="browser.open",
            arguments={"url": "docs/page?token=secret#fragment"},
        )
    )
    relative_projection = project_browser_open_tool_display(
        call=ToolCall(
            id="browser-open-5",
            name="browser.open",
            arguments={"url": "docs/page"},
        )
    )

    assert missing_projection.target == "URL"
    assert not missing_projection.redacted
    assert malformed_projection.redacted
    assert malformed_projection.target == "[redacted]"
    assert bad_port_projection.redacted
    assert bad_port_projection.target == "https://example.test/page"
    assert relative_query_projection.redacted
    assert relative_query_projection.target == "docs/page"
    assert relative_projection.target == "docs/page"
    assert not relative_projection.redacted


def test_protocol_relative_urls_do_not_leak_credentials() -> None:
    browser_projection = project_browser_open_tool_display(
        call=ToolCall(
            id="browser-open-1",
            name="browser.open",
            arguments={
                "url": "//user:hunter2@example.test/page?token=secret#frag",
            },
        ),
    )
    mcp_projection = project_mcp_call_tool_display(
        call=ToolCall(
            id="mcp-call-1",
            name="mcp.call",
            arguments={
                "uri": "//user:hunter2@mcp.example.test/rpc?token=secret",
                "name": "remote.search",
            },
        ),
    )
    ipv6_projection = project_browser_open_tool_display(
        call=ToolCall(
            id="browser-open-2",
            name="browser.open",
            arguments={
                "url": "http://user:hunter2@[::1]:8080/page?token=secret",
            },
        ),
    )

    for projection, safe_url in (
        (browser_projection, "//example.test/page"),
        (mcp_projection, "//mcp.example.test/rpc"),
        (ipv6_projection, "http://[::1]:8080/page"),
    ):
        payload = _payload_text(projection)
        assert projection.redacted
        assert "hunter2" not in payload
        assert "token=secret" not in payload
        assert safe_url in payload


def test_browser_terminal_projections_redact_url_targets() -> None:
    call = ToolCall(
        id="browser-open-1",
        name="browser.open",
        arguments={
            "url": "https://user:hunter2@example.test/page?code=abc#frag",
        },
    )
    projections = [
        project_browser_open_tool_display(
            call=call,
            outcome=ToolCallResult(
                id="browser-result-1",
                name=call.name,
                arguments=call.arguments,
                call=call,
                result="browser output",
            ),
        ),
        project_browser_open_tool_display(
            call=call,
            outcome=ToolCallError(
                id="browser-error-1",
                name=call.name,
                arguments=call.arguments,
                call=call,
                error=RuntimeError("failed"),
                message="failed",
            ),
        ),
        project_browser_open_tool_display(
            call=call,
            outcome=ToolCallDiagnostic(
                id="browser-diag-1",
                requested_name=call.name,
                canonical_name=call.name,
                code=ToolCallDiagnosticCode.USER_REJECTED,
                stage=ToolCallDiagnosticStage.CONFIRM,
                message="not approved",
            ),
        ),
    ]

    for projection in projections:
        payload = _payload_text(projection)
        assert projection.redacted
        assert "hunter2" not in payload
        assert "code=abc" not in payload
        assert "#frag" not in payload
        assert "https://example.test/page" in payload


def test_code_call_projection_never_includes_source_body() -> None:
    secret = "plain-hunter2"
    projection = project_code_run_tool_display(
        call=ToolCall(
            id="code-run-1",
            name="code.run",
            arguments={
                "code": (
                    "def run():\n"
                    f"    api_key = '{secret}'\n"
                    "    return api_key\n"
                )
            },
        )
    )
    payload = _payload_text(projection)
    code_chars = projection.metrics["code_chars"]

    assert isinstance(code_chars, int)
    assert code_chars > 0
    assert secret not in payload
    assert "api_key" not in payload
    assert "def run" not in payload


def test_calculator_terminal_projection_includes_result_preview() -> None:
    call = ToolCall(
        id="calculator-1",
        name="math.calculator",
        arguments={"expression": "2 + 2"},
    )

    projection = project_calculator_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="calculator-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result=4,
        ),
    )

    assert projection.status == "completed"
    assert projection.target == "2 + 2"
    assert _detail_value(projection, "result") == "4"
    assert projection.preview is not None
    assert projection.preview.content == "4"


def test_ast_grep_terminal_projection_keeps_rewrite_action() -> None:
    call = ToolCall(
        id="ast-grep-1",
        name="code.search.ast.grep",
        arguments={
            "pattern": "print($A)",
            "rewrite": "logger.info($A)",
            "paths": ["src"],
        },
    )

    projection = project_ast_grep_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="ast-grep-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result=["src/a.py", "src/b.py"],
        ),
    )

    assert projection.action == "rewrite"
    assert projection.status == "completed"
    assert projection.metrics["items"] == 2


def test_ast_grep_path_projection_handles_edge_inputs() -> None:
    no_paths_projection = project_ast_grep_tool_display(
        call=ToolCall(
            id="ast-grep-1",
            name="code.search.ast.grep",
            arguments={"pattern": "print($A)"},
        )
    )
    string_path_projection = project_ast_grep_tool_display(
        call=ToolCall(
            id="ast-grep-2",
            name="code.search.ast.grep",
            arguments={"pattern": "print($A)", "paths": "src"},
        )
    )
    mapping_path_projection = project_ast_grep_tool_display(
        call=ToolCall(
            id="ast-grep-3",
            name="code.search.ast.grep",
            arguments={"pattern": "print($A)", "paths": {"src": True}},
        )
    )
    limited_path_projection = project_ast_grep_tool_display(
        call=ToolCall(
            id="ast-grep-4",
            name="code.search.ast.grep",
            arguments={
                "pattern": "print($A)",
                "paths": [
                    "src0",
                    7,
                    "src1",
                    "src2",
                    "src3",
                    "src4",
                    "src5",
                    "src6",
                    "src7",
                ],
            },
        )
    )

    assert no_paths_projection.scope == "workspace"
    assert string_path_projection.scope == "src"
    assert mapping_path_projection.scope == "workspace"
    assert (
        limited_path_projection.scope
        == "src0, src1, src2, src3, src4, src5, src6, ..."
    )


def test_mcp_projection_redacts_uri_credentials_and_argument_keys() -> None:
    projection = project_mcp_call_tool_display(
        call=ToolCall(
            id="mcp-call-1",
            name="mcp.call",
            arguments={
                "uri": (
                    "https://user:password@mcp.example.test/rpc?token=secret"
                ),
                "name": "remote.search",
                "arguments": {
                    "api_key": "secret-value",
                    "query": "weather",
                },
            },
        )
    )
    payload = _payload_text(projection)

    assert projection.redacted
    assert "password" not in payload
    assert "secret-value" not in payload
    assert "api_key" not in payload
    assert "https://mcp.example.test/rpc" in payload


def test_mcp_projection_limits_argument_keys() -> None:
    projection = project_mcp_call_tool_display(
        call=ToolCall(
            id="mcp-call-1",
            name="mcp.call",
            arguments={
                "uri": "https://mcp.example.test/rpc",
                "name": "remote.search",
                "arguments": {f"key_{index}": index for index in range(9)},
            },
        )
    )

    assert _detail_value(projection, "argument_count") == 9
    assert (
        _detail_value(projection, "argument_keys")
        == "key_0, key_1, key_2, key_3, key_4, key_5, key_6, key_7, ..."
    )


def test_mcp_result_projection_counts_mapping_results() -> None:
    call = ToolCall(
        id="mcp-call-1",
        name="mcp.call",
        arguments={"name": "remote.info"},
    )
    projection = project_mcp_call_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="mcp-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result={"first": 1, "second": 2},
        ),
    )

    assert projection.status == "completed"
    assert projection.target == "remote.info"
    assert projection.metrics["items"] == 2


def test_query_like_results_are_summarized_without_output() -> None:
    secret_result = "QUERY_RESULT_SECRET"
    search_call = ToolCall(
        id="search-1",
        name="search",
        arguments={"query": "latest release", "engine": "google"},
    )
    code_call = ToolCall(
        id="code-run-1",
        name="code.run",
        arguments={"code": "def run(): return 'ok'"},
    )

    search_projection = project_search_tool_display(
        call=search_call,
        outcome=ToolCallResult(
            id="search-result-1",
            name=search_call.name,
            arguments=search_call.arguments,
            call=search_call,
            result=secret_result,
        ),
    )
    code_projection = project_code_run_tool_display(
        call=code_call,
        outcome=ToolCallResult(
            id="code-result-1",
            name=code_call.name,
            arguments=code_call.arguments,
            call=code_call,
            result=secret_result,
        ),
    )

    for projection in (search_projection, code_projection):
        payload = _payload_text(projection)
        assert projection.status == "completed"
        assert projection.metrics["text_chars"] == len(secret_result)
        assert secret_result not in payload


def test_graph_projection_excludes_input_values_and_data_uri() -> None:
    call = ToolCall(
        id="graph-bar-1",
        name="graph.bar",
        arguments={
            "categories": ["private-segment"],
            "values": [12345.0],
            "title": "Revenue",
        },
    )
    call_projection = project_graph_tool_display(call=call)
    result_projection = project_graph_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="graph-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result={
                "chart_type": "bar",
                "format": "png",
                "mime_type": "image/png",
                "encoding": "base64_data_uri",
                "data_uri": "data:image/png;base64,SECRET_GRAPH_DATA",
                "title": "Revenue",
                "width": 6.4,
                "height": 4.8,
                "dpi": 100,
                "series": ["values"],
                "points": 1,
            },
        ),
    )

    call_payload = _payload_text(call_projection)
    result_payload = _payload_text(result_projection)

    assert call_projection.metrics["points"] == 1
    assert "private-segment" not in call_payload
    assert "12345" not in call_payload
    assert "SECRET_GRAPH_DATA" not in result_payload
    assert "data_uri" not in result_payload
    assert result_projection.metrics["points"] == 1


def test_graph_projection_counts_series_mapping() -> None:
    line_projection = project_graph_tool_display(
        call=ToolCall(
            id="graph-line-1",
            name="graph.line",
            arguments={
                "x_labels": ["Q1", "Q2"],
                "series": {"actual": [1, 2], "forecast": [3, 4]},
                "width": True,
                "height": False,
            },
        )
    )
    unknown_projection = project_graph_tool_display(
        call=ToolCall(
            id="graph-radar-1",
            name="graph.radar",
            arguments={"values": [1, 2, 3]},
        )
    )

    assert line_projection.metrics["series"] == 2
    assert line_projection.metrics["points"] == 4
    assert _detail_value(line_projection, "width") is None
    assert _detail_value(line_projection, "height") is None
    assert unknown_projection.target == "radar chart"
    assert unknown_projection.metrics["points"] == 0


def test_graph_terminal_projection_falls_back_for_non_mapping_result() -> None:
    call = ToolCall(
        id="graph-line-1",
        name="graph.line",
        arguments={"title": "Revenue"},
    )

    projection = project_graph_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="graph-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result=["series-a", "series-b"],
        ),
    )

    assert projection.status == "completed"
    assert projection.outcome == "result"
    assert projection.target == "Revenue"
    assert projection.metrics["items"] == 2


def test_graph_result_redacts_file_and_ignores_bool_dimensions() -> None:
    call = ToolCall(
        id="graph-line-1",
        name="graph.line",
        arguments={"title": "Revenue"},
    )
    projection = project_graph_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="graph-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result={
                "chart_type": "line",
                "format": "png",
                "width": True,
                "height": False,
                "file": "/private/charts/revenue.png",
            },
        ),
    )
    payload = _payload_text(projection)

    assert projection.redacted
    assert _detail_value(projection, "width") is None
    assert _detail_value(projection, "height") is None
    assert _detail_value(projection, "file") == "[redacted]"
    assert "/private/charts/revenue.png" not in payload


def test_memory_terminal_projection_uses_typed_counts_without_data() -> None:
    memory = Memory(
        id=uuid4(),
        model_id="model",
        type=MemoryType.RAW,
        participant_id=uuid4(),
        namespace="docs",
        identifier="secret-identifier",
        data="SECRET_MEMORY_DATA",
        partitions=3,
        symbols=None,
        created_at=datetime(2026, 6, 20, tzinfo=UTC),
        title="secret title",
        description="secret description",
    )
    call = ToolCall(
        id="memory-list-1",
        name="memory.list",
        arguments={"namespace": "docs"},
    )

    projection = project_memory_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="memory-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result=cast(Any, [memory]),
        ),
    )
    payload = _payload_text(projection)

    assert projection.metrics["memories"] == 1
    assert projection.metrics["partitions"] == 3
    assert "SECRET_MEMORY_DATA" not in payload
    assert "secret-identifier" not in payload
    assert "secret title" not in payload
    assert "secret description" not in payload


def test_message_memory_terminal_projection_reports_match_status() -> None:
    call = ToolCall(
        id="message-read-1",
        name="memory.message.read",
        arguments={"search": "needle"},
    )
    found_projection = project_memory_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="message-read-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result="remembered answer",
        ),
    )
    missing_projection = project_memory_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="message-read-result-2",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result="NOT_FOUND",
        ),
    )

    assert found_projection.outcome == "result"
    assert found_projection.metrics["matches"] == 1
    assert found_projection.metrics["text_chars"] == len("remembered answer")
    assert missing_projection.outcome == "not_found"
    assert missing_projection.metrics["matches"] == 0
    assert missing_projection.metrics["text_chars"] == 0


def test_memory_search_terminal_projection_counts_partition_text() -> None:
    call = ToolCall(
        id="memory-read-1",
        name="memory.read",
        arguments={"namespace": "docs", "search": "needle"},
    )
    projection = project_memory_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="memory-read-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result=cast(Any, ["alpha", {"ignored": True}, "beta"]),
        ),
    )

    assert projection.metrics["matches"] == 3
    assert projection.metrics["text_chars"] == len("alpha") + len("beta")


def test_memory_terminal_projection_handles_error_and_diagnostic() -> None:
    call = ToolCall(
        id="memory-list-1",
        name="memory.list",
        arguments={"namespace": "docs"},
    )
    error_projection = project_memory_tool_display(
        call=call,
        outcome=ToolCallError(
            id="memory-error-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            error=RuntimeError("failed"),
            message="failed",
        ),
    )
    diagnostic_projection = project_memory_tool_display(
        call=call,
        outcome=ToolCallDiagnostic(
            id="memory-diag-1",
            requested_name=call.name,
            canonical_name=call.name,
            code=ToolCallDiagnosticCode.USER_REJECTED,
            stage=ToolCallDiagnosticStage.CONFIRM,
            message="not approved",
        ),
    )

    assert error_projection.action == "list"
    assert error_projection.status == "error"
    assert error_projection.outcome == "RuntimeError"
    assert diagnostic_projection.action == "skip"
    assert diagnostic_projection.severity == "warning"


def test_unknown_memory_operation_uses_generic_summary() -> None:
    projection = project_memory_tool_display(
        call=ToolCall(
            id="memory-prune-1",
            name="memory.prune",
            arguments={"namespace": "docs"},
        )
    )

    assert projection.summary == "Use memory tool."


def test_memory_store_projection_lists_namespaces_only() -> None:
    call = ToolCall(id="memory-stores-1", name="memory.stores")
    projection = project_memory_tool_display(
        call=call,
        outcome=ToolCallResult(
            id="memory-stores-result-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            result=cast(
                Any,
                [
                    PermanentMemoryStore(
                        namespace="docs",
                        description="SECRET_STORE_DESCRIPTION",
                    )
                ],
            ),
        ),
    )
    payload = _payload_text(projection)

    assert projection.metrics["stores"] == 1
    assert "docs" in payload
    assert "SECRET_STORE_DESCRIPTION" not in payload


def test_terminal_error_projection_omits_error_message_payload() -> None:
    call = ToolCall(
        id="code-run-1",
        name="code.run",
        arguments={"code": "def run(): return 1"},
    )
    projection = project_code_run_tool_display(
        call=call,
        outcome=ToolCallError(
            id="code-error-1",
            name=call.name,
            arguments=call.arguments,
            call=call,
            error={
                "type": "RuntimeError",
                "message": "password=hunter2",
            },
            message="password=hunter2",
        ),
    )
    payload = _payload_text(projection)

    assert projection.status == "error"
    assert projection.outcome == "RuntimeError"
    assert "hunter2" not in payload
    assert "password" not in payload


def test_terminal_diagnostic_projection_is_bounded_metadata() -> None:
    call = ToolCall(
        id="mcp-call-1",
        name="mcp.call",
        arguments={"uri": "https://mcp.example.test", "name": "remote"},
    )
    projection = project_mcp_call_tool_display(
        call=call,
        outcome=ToolCallDiagnostic(
            id="diag-1",
            requested_name="mcp.call",
            canonical_name="mcp.call",
            code=ToolCallDiagnosticCode.USER_REJECTED,
            stage=ToolCallDiagnosticStage.CONFIRM,
            message="Contains SECRET_DIAGNOSTIC_TEXT.",
        ),
    )
    payload = _payload_text(projection)

    assert projection.action == "skip"
    assert projection.outcome == "tool_call.user_rejected"
    assert "SECRET_DIAGNOSTIC_TEXT" not in payload


def test_projection_coverage_excludes_youtube() -> None:
    names = set(_CALL_INTENTS)

    assert "youtube" not in names
    assert not any(name.startswith("youtube.") for name in names)
    assert not (REPO_ROOT / "src/avalan/tool/youtube.py").exists()


def test_joined_limited_strings_ignores_scalar_inputs() -> None:
    assert _joined_limited_strings("abc") is None
    assert _joined_limited_strings(42) is None


def _manager() -> ToolManager:
    return ToolManager.create_instance(
        available_toolsets=[
            MathToolSet(namespace="math"),
            McpToolSet(),
            CodeToolSet(namespace="code"),
            GraphToolSet(namespace="graph"),
            MemoryToolSet(cast(MemoryManager, object()), namespace="memory"),
            ToolSet(tools=[SearchEngineTool()]),
        ],
        settings=ToolManagerSettings(),
    )


def _payload_text(projection: ToolDisplayProjection) -> str:
    return dumps(projection.to_payload(), sort_keys=True)


def _detail_value(projection: ToolDisplayProjection, label: str) -> Any:
    for detail in projection.details:
        if detail.label == label:
            return detail.value
    raise AssertionError(f"Missing detail {label}")
