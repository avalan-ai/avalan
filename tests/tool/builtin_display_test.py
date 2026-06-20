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
    project_browser_open_tool_display,
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
