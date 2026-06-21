from ast import Import, ImportFrom
from ast import parse as parse_python
from ast import walk as walk_python
from collections.abc import Iterable, Iterator
from pathlib import Path
from re import findall
from typing import cast

from avalan.entities import (
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
)
from avalan.memory.manager import MemoryManager
from avalan.tool import ToolSet
from avalan.tool import display as tool_display
from avalan.tool.code import CodeToolSet
from avalan.tool.display import (
    MAX_DISPLAY_DETAIL_VALUE_LENGTH,
    MAX_DISPLAY_DETAILS,
    MAX_DISPLAY_METRICS,
    MAX_DISPLAY_PREVIEW_LENGTH,
    REDACTED_DISPLAY_VALUE,
    TOOL_DISPLAY_PROJECTION_METADATA_KEY,
    ToolDisplayDetail,
    ToolDisplayPreview,
    ToolDisplayProjection,
    fallback_tool_call_display_projection,
    fallback_tool_outcome_display_projection,
    is_json_safe_display_value,
    is_sensitive_display_value,
    sanitize_display_label,
    sanitize_display_value,
    tool_call_display_projection_from_metadata,
    tool_display_projection_from_metadata,
    tool_display_projection_metadata,
    tool_outcome_display_projection_from_metadata,
)
from avalan.tool.graph import GraphToolSet
from avalan.tool.math import MathToolSet
from avalan.tool.mcp import McpToolSet
from avalan.tool.memory import MemoryToolSet
from avalan.tool.search_engine import SearchEngineTool
from avalan.tool.shell.toolset import ShellToolSet

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_BUILT_IN_TOOL_NAMES = {
    "browser.open",
    "code.run",
    "code.search.ast.grep",
    "database.count",
    "database.inspect",
    "database.keys",
    "database.kill",
    "database.locks",
    "database.plan",
    "database.relationships",
    "database.run",
    "database.sample",
    "database.size",
    "database.tables",
    "database.tasks",
    "graph.bar",
    "graph.histogram",
    "graph.line",
    "graph.pie",
    "graph.scatter",
    "math.calculator",
    "mcp.call",
    "memory.list",
    "memory.message.read",
    "memory.read",
    "memory.stores",
    "search",
    "shell.awk",
    "shell.cat",
    "shell.file",
    "shell.find",
    "shell.head",
    "shell.jq",
    "shell.ls",
    "shell.pdfinfo",
    "shell.pdftoppm",
    "shell.pdftotext",
    "shell.rg",
    "shell.sed",
    "shell.tail",
    "shell.tesseract",
    "shell.wc",
}


def test_valid_projection_round_trips_through_metadata() -> None:
    projection = ToolDisplayProjection(
        action="read",
        label="shell.cat",
        target="src/avalan/tool/display.py",
        scope="workspace",
        summary="Read display projection source.",
        status="running",
        outcome="stdout",
        severity="info",
        progress=0.5,
        details=(
            ToolDisplayDetail(
                label="path", value="src/avalan/tool/display.py"
            ),
            ToolDisplayDetail(label="lines", value=120),
        ),
        metrics={"bytes": 4096},
        preview=ToolDisplayPreview(
            label="stdout",
            content="from avalan.tool.display import ToolDisplayProjection",
            media_type="text/plain",
        ),
    )

    metadata = tool_display_projection_metadata(projection)
    restored = tool_display_projection_from_metadata(metadata)

    assert TOOL_DISPLAY_PROJECTION_METADATA_KEY in metadata
    assert restored == projection
    assert is_json_safe_display_value(metadata)


def test_detail_labels_and_values_are_preserved_after_sanitization() -> None:
    projection = ToolDisplayProjection(
        action="search",
        label="shell.rg",
        target="src",
        details=(
            ToolDisplayDetail(label="pattern", value="ToolCall"),
            ToolDisplayDetail(label="fixed_strings", value=True),
        ),
    )

    details = tuple(projection.details)

    assert details[0].label == "pattern"
    assert details[0].value == "ToolCall"
    assert details[1].label == "fixed_strings"
    assert details[1].value is True
    assert not projection.redacted
    assert not projection.truncated


def test_sensitive_labels_and_values_are_redacted() -> None:
    projection = ToolDisplayProjection(
        action="call",
        label="third.party",
        details=(
            ToolDisplayDetail(label="api_key", value="abc123"),
            ToolDisplayDetail(label="notes", value="password=hunter2"),
            ToolDisplayDetail(
                label="nested",
                value=sanitize_display_value(
                    "nested",
                    {"token": "secret-value"},
                ),
            ),
        ),
    )

    details = tuple(projection.details)

    assert details[0].label == REDACTED_DISPLAY_VALUE
    assert details[0].value == REDACTED_DISPLAY_VALUE
    assert details[1].value == REDACTED_DISPLAY_VALUE
    assert details[2].value == REDACTED_DISPLAY_VALUE
    assert projection.redacted


def test_control_obfuscated_sensitive_labels_are_redacted() -> None:
    projection = ToolDisplayProjection(
        action="call",
        label="third.party",
        details=(
            ToolDisplayDetail(label="to\x1b[31mken", value="abc123"),
            ToolDisplayDetail(label="api[red]_key", value="abc123"),
            ToolDisplayDetail(label="pass-wd", value="abc123"),
            ToolDisplayDetail(label="bear.er", value="abc123"),
            ToolDisplayDetail(label="cre-dential", value="abc123"),
        ),
    )

    details = tuple(projection.details)

    for detail in details:
        assert detail.label == REDACTED_DISPLAY_VALUE
        assert detail.value == REDACTED_DISPLAY_VALUE


def test_overlong_text_is_truncated() -> None:
    overlong = "x" * (MAX_DISPLAY_DETAIL_VALUE_LENGTH + 40)
    detail = ToolDisplayDetail(label="output", value=overlong)

    assert isinstance(detail.value, str)
    assert len(detail.value) <= MAX_DISPLAY_DETAIL_VALUE_LENGTH
    assert detail.value.endswith("...")
    assert detail.truncated


def test_redaction_and_truncation_markers_serialize() -> None:
    projection = ToolDisplayProjection(
        action="call",
        label="custom",
        details=(
            ToolDisplayDetail(label="password", value="secret"),
            ToolDisplayDetail(
                label="output",
                value="x" * (MAX_DISPLAY_DETAIL_VALUE_LENGTH + 1),
            ),
        ),
        preview=ToolDisplayPreview(
            content="x" * 1001,
            label="stdout",
            media_type="text/plain",
        ),
    )
    payload = projection.to_payload()

    assert payload["redacted"] is True
    assert payload["truncated"] is True
    assert projection.preview is not None
    assert projection.preview.to_payload()["truncated"] is True
    assert tuple(projection.details)[0].to_payload()["redacted"] is True
    assert tuple(projection.details)[1].to_payload()["truncated"] is True


def test_sensitive_preview_label_redacts_content() -> None:
    projection = ToolDisplayProjection(
        action="finish",
        preview=ToolDisplayPreview(label="api_key", content="abc123"),
    )

    assert projection.preview is not None
    assert projection.preview.label == REDACTED_DISPLAY_VALUE
    assert projection.preview.content == REDACTED_DISPLAY_VALUE
    assert projection.preview.redacted
    assert projection.to_payload()["redacted"] is True


def test_scalar_helpers_cover_safe_and_sensitive_values() -> None:
    assert sanitize_display_label(" api_key ") == REDACTED_DISPLAY_VALUE
    assert sanitize_display_value("count", None) is None
    assert sanitize_display_value("count", 1.5) == 1.5
    assert sanitize_display_value("count", float("nan")) == "nan"
    assert is_sensitive_display_value(["token"])
    assert not is_sensitive_display_value([[[[[["token"]]]]]])
    assert is_json_safe_display_value(["ok", {"count": 1}])
    assert not is_json_safe_display_value(object())


def test_projection_bounds_trim_details_and_metrics() -> None:
    projection = ToolDisplayProjection(
        action="measure",
        label="custom",
        details=tuple(
            ToolDisplayDetail(label=f"detail_{index}", value=index)
            for index in range(MAX_DISPLAY_DETAILS + 1)
        ),
        metrics={
            f"metric_{index}": index
            for index in range(MAX_DISPLAY_METRICS + 1)
        },
    )

    assert len(projection.details) == MAX_DISPLAY_DETAILS
    assert len(projection.metrics) == MAX_DISPLAY_METRICS
    assert projection.truncated


def test_projection_payload_stops_parsing_after_retained_bounds() -> None:
    payload = {
        "action": "measure",
        "details": [
            {"label": f"detail_{index}", "value": index}
            for index in range(MAX_DISPLAY_DETAILS)
        ]
        + [{"malformed": object()}],
        "metrics": {
            **{
                f"metric_{index}": index
                for index in range(MAX_DISPLAY_METRICS)
            },
            "malformed": object(),
        },
    }

    projection = ToolDisplayProjection.from_payload(payload)

    assert projection is not None
    assert len(projection.details) == MAX_DISPLAY_DETAILS
    assert len(projection.metrics) == MAX_DISPLAY_METRICS
    assert projection.truncated
    assert projection.to_payload()["truncated"] is True


def test_fallback_preview_uses_bounded_shallow_text() -> None:
    call = ToolCall(id="call-1", name="custom.tool")
    result = ToolCallResult(
        id="result-1",
        name=call.name,
        arguments={},
        call=call,
        result={
            "custom": object(),
            "items": ["x" * MAX_DISPLAY_PREVIEW_LENGTH]
            * (MAX_DISPLAY_DETAILS + 2),
        },
    )

    projection = fallback_tool_outcome_display_projection(result)

    assert projection.preview is not None
    assert len(projection.preview.content) <= MAX_DISPLAY_PREVIEW_LENGTH
    assert "object" in projection.preview.content


def test_sensitive_scan_is_bounded_to_retained_items() -> None:
    values: dict[str, object] = {
        f"key_{index}": index for index in range(MAX_DISPLAY_DETAILS)
    }
    values["token"] = "secret-value"

    detail = ToolDisplayDetail(
        label="payload",
        value=sanitize_display_value("payload", values),
    )

    assert detail.value != REDACTED_DISPLAY_VALUE
    assert not detail.redacted


def test_sensitive_scan_finds_nested_retained_values() -> None:
    mapping_detail = ToolDisplayDetail(
        label="payload",
        value=sanitize_display_value(
            "payload",
            {"nested": {"token": "secret-value"}},
        ),
    )
    sequence_detail = ToolDisplayDetail(
        label="payload",
        value=sanitize_display_value(
            "payload",
            list(range(MAX_DISPLAY_DETAILS)) + ["token"],
        ),
    )

    assert mapping_detail.value == REDACTED_DISPLAY_VALUE
    assert sequence_detail.value != REDACTED_DISPLAY_VALUE


def test_sensitive_scan_stops_iterating_after_retained_items() -> None:
    class CountingMapping(dict[str, object]):
        iterations = 0

        def items(self) -> Iterator[tuple[str, object]]:
            for item in super().items():
                self.iterations += 1
                yield item

    values = CountingMapping(
        {f"key_{index}": index for index in range(MAX_DISPLAY_DETAILS + 20)}
    )

    detail = ToolDisplayDetail(
        label="payload",
        value=sanitize_display_value("payload", values),
    )

    assert detail.value != REDACTED_DISPLAY_VALUE
    assert values.iterations <= (MAX_DISPLAY_DETAILS + 1) * 2


def test_sensitive_metric_label_redacts_value() -> None:
    projection = ToolDisplayProjection(
        action="measure",
        metrics={"pass-wd": "abc123"},
    )

    assert projection.metrics == {
        REDACTED_DISPLAY_VALUE: REDACTED_DISPLAY_VALUE
    }
    assert projection.redacted


def test_bounded_text_defensive_branches_are_covered() -> None:
    oversized_mapping = {
        f"key_{index}": index for index in range(MAX_DISPLAY_DETAILS + 1)
    }

    assert tool_display._bounded_display_text_part("secret", 3, 0) == "..."
    assert tool_display._bounded_display_text_part(object(), 10, 3) == "object"
    assert "..." in tool_display._bounded_display_text(
        oversized_mapping,
        MAX_DISPLAY_PREVIEW_LENGTH,
    )
    assert tool_display._bounded_display_text({"name": "x" * 80}, 4)


def test_bounded_text_sampling_retains_tail_after_long_whitespace() -> None:
    detail = ToolDisplayDetail(
        label="output",
        value="alpha" + (" " * 5000) + "omega",
    )

    assert detail.value == "alpha omega"
    assert detail.truncated


def test_sensitive_tail_in_bounded_text_sample_is_redacted() -> None:
    detail = ToolDisplayDetail(
        label="output",
        value="alpha" + (" " * 5000) + "password=hunter2",
    )

    assert detail.value == REDACTED_DISPLAY_VALUE
    assert detail.redacted


def test_empty_optional_projection_text_is_omitted() -> None:
    projection = ToolDisplayProjection(action="call", label="   ")

    assert projection.label is None
    assert "label" not in projection.to_payload()


def test_projection_payload_rejects_malformed_values() -> None:
    invalid_payloads: tuple[object, ...] = (
        [],
        {"action": 1},
        {"action": "call", "label": 1},
        {"action": "call", "progress": True},
        {"action": "call", "details": "bad"},
        {"action": "call", "details": ["bad"]},
        {"action": "call", "details": [{"label": "x"}]},
        {"action": "call", "metrics": []},
        {"action": "call", "metrics": {1: "bad"}},
        {"action": "call", "metrics": {"bad": []}},
        {"action": "call", "preview": "bad"},
        {"action": "call", "preview": {"content": 1}},
        {"action": "call", "redacted": "yes"},
        {"action": "call", "progress": 2},
    )

    for payload in invalid_payloads:
        assert ToolDisplayProjection.from_payload(payload) is None


def test_detail_and_preview_payload_reject_malformed_values() -> None:
    invalid_details: tuple[object, ...] = (
        [],
        {},
        {"label": 1, "value": "x"},
        {"label": "x"},
        {"label": "x", "value": []},
        {"label": "x", "value": "x", "redacted": "yes"},
    )
    invalid_previews: tuple[object, ...] = (
        [],
        {},
        {"content": 1},
        {"content": "x", "label": 1},
        {"content": "x", "media_type": 1},
        {"content": "x", "truncated": "yes"},
    )

    for payload in invalid_details:
        assert ToolDisplayDetail.from_payload(payload) is None
    for payload in invalid_previews:
        assert ToolDisplayPreview.from_payload(payload) is None


def test_metadata_reader_handles_missing_payloads() -> None:
    assert tool_display_projection_from_metadata(None) is None
    assert tool_display_projection_from_metadata({}) is None
    assert (
        tool_display_projection_from_metadata(
            {TOOL_DISPLAY_PROJECTION_METADATA_KEY: []}
        )
        is None
    )


def test_unknown_third_party_tool_call_uses_generic_fallback() -> None:
    """Unknown third-party tools get a generic display projection."""
    call = ToolCall(
        id="call-1",
        name="third.party.fetch",
        arguments={"query": "status", "token": "secret-value"},
    )

    projection = fallback_tool_call_display_projection(call)

    assert projection.action == "call"
    assert projection.label == "third.party.fetch"
    assert projection.target == "third.party.fetch"
    assert projection.summary == "Call third.party.fetch."
    assert tuple(projection.details)[0].label == "query"
    assert tuple(projection.details)[1].label == REDACTED_DISPLAY_VALUE
    assert projection.redacted


def test_tool_call_fallback_handles_empty_and_bounded_arguments() -> None:
    empty = fallback_tool_call_display_projection(
        ToolCall(id="call-1", name="", arguments=None),
    )
    bounded = fallback_tool_call_display_projection(
        ToolCall(
            id="call-2",
            name="third.party.fetch",
            arguments={
                f"arg_{index}": index
                for index in range(MAX_DISPLAY_DETAILS + 1)
            },
        ),
    )

    assert empty.label == "tool"
    assert empty.details == ()
    assert len(bounded.details) == MAX_DISPLAY_DETAILS


def test_malformed_projection_metadata_falls_back_to_tool_call() -> None:
    call = ToolCall(
        id="call-1",
        name="third.party.fetch",
        arguments={"query": "status"},
    )
    metadata = {
        TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
            "action": ["not", "a", "string"],
        },
    }

    projection = tool_call_display_projection_from_metadata(call, metadata)

    assert tool_display_projection_from_metadata(metadata) is None
    assert projection == fallback_tool_call_display_projection(call)


def test_outcome_fallbacks_cover_result_error_and_diagnostic() -> None:
    call = ToolCall(
        id="call-1",
        name="third.party.fetch",
        arguments={"query": "status"},
    )
    result = ToolCallResult(
        id="result-1",
        name=call.name,
        arguments=call.arguments,
        call=call,
        result={"status": "ok"},
    )
    error = ToolCallError(
        id="error-1",
        name=call.name,
        arguments=call.arguments,
        call=call,
        error={"type": "RemoteError", "message": "failed"},
        message="Remote call failed.",
    )
    diagnostic = ToolCallDiagnostic(
        id="diag-1",
        requested_name="third.party.fetch",
        code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
        stage=ToolCallDiagnosticStage.RESOLVE,
        message="Tool is unknown.",
    )

    result_projection = fallback_tool_outcome_display_projection(result)
    error_projection = fallback_tool_outcome_display_projection(error)
    diagnostic_projection = fallback_tool_outcome_display_projection(
        diagnostic,
    )

    assert result_projection.status == "completed"
    assert result_projection.outcome == "result"
    assert result_projection.preview is not None
    assert error_projection.status == "error"
    assert error_projection.severity == "error"
    assert tuple(error_projection.details)[0].value == "RemoteError"
    assert diagnostic_projection.action == "skip"
    assert diagnostic_projection.outcome == "tool.unknown"


def test_outcome_result_without_value_omits_preview() -> None:
    call = ToolCall(id="call-1", name="")
    outcome = ToolCallResult(
        id="result-1",
        name=call.name,
        arguments={},
        call=call,
        result=None,
    )

    projection = fallback_tool_outcome_display_projection(outcome)

    assert projection.label == "tool"
    assert projection.preview is None


def test_valid_outcome_metadata_is_used_before_fallback() -> None:
    call = ToolCall(id="call-1", name="third.party.fetch")
    outcome = ToolCallResult(
        id="result-1",
        name=call.name,
        arguments={},
        call=call,
        result="ok",
    )
    metadata = ToolDisplayProjection(
        action="finish",
        label="custom",
        status="completed",
    ).to_metadata()

    projection = tool_outcome_display_projection_from_metadata(
        outcome,
        metadata,
    )

    assert projection.label == "custom"


def test_builtin_tool_inventory_is_recorded_with_youtube_excluded() -> None:
    inventory = set(EXPECTED_BUILT_IN_TOOL_NAMES)
    discovered = (
        _cheap_toolset_names()
        | _source_assigned_tool_names("database")
        | _source_assigned_tool_names("browser")
    )

    assert "youtube" not in inventory
    assert not any(name.startswith("youtube.") for name in inventory)
    assert not (REPO_ROOT / "src/avalan/tool/youtube.py").exists()
    assert discovered == inventory


def test_projection_metadata_is_not_part_of_provider_tool_schemas() -> None:
    schemas = MathToolSet(namespace="math").json_schemas()

    assert schemas is not None
    assert TOOL_DISPLAY_PROJECTION_METADATA_KEY not in str(schemas)


def test_display_helpers_stay_independent_of_rendering_and_tool_modules() -> (
    None
):
    display_path = REPO_ROOT / "src/avalan/tool/display.py"
    imported_modules = _imported_modules(display_path)

    assert not _has_module_segment(imported_modules, "cli")
    assert not _has_module_segment(imported_modules, "database")
    assert not _has_module_segment(imported_modules, "shell")


def test_theme_modules_do_not_import_tool_specific_modules() -> None:
    theme_dir = REPO_ROOT / "src/avalan/cli/theme"
    allowed_modules = {"tool.display"}

    for path in theme_dir.glob("*.py"):
        imported_modules = _imported_modules(path)
        unexpected_modules = {
            module
            for module in imported_modules
            if "tool" in module.split(".") and module not in allowed_modules
        }

        assert not unexpected_modules
        assert not _has_module_segment(imported_modules, "database")
        assert not _has_module_segment(imported_modules, "shell")


def _cheap_toolset_names() -> set[str]:
    names = {
        SearchEngineTool().__name__,
    }
    for toolset in (
        MathToolSet(namespace="math"),
        McpToolSet(),
        CodeToolSet(namespace="code"),
        GraphToolSet(namespace="graph"),
        MemoryToolSet(cast(MemoryManager, object()), namespace="memory"),
        ShellToolSet(),
    ):
        names.update(_toolset_names(toolset))
    return names


def _toolset_names(
    toolset: ToolSet,
    prefix: str | None = None,
) -> set[str]:
    namespace = (
        f"{prefix}.{toolset.namespace}"
        if prefix and toolset.namespace
        else prefix or toolset.namespace
    )
    tool_prefix = f"{namespace}." if namespace else ""
    names: set[str] = set()
    for tool in toolset.tools:
        if isinstance(tool, ToolSet):
            names.update(_toolset_names(tool, namespace))
            continue
        tool_name = getattr(tool, "__name__", tool.__class__.__name__)
        names.add(f"{tool_prefix}{tool_name}")
    return names


def _source_assigned_tool_names(namespace: str) -> set[str]:
    source_dir = REPO_ROOT / "src/avalan/tool" / namespace
    if source_dir.is_dir():
        paths: Iterable[Path] = source_dir.glob("*.py")
    else:
        paths = (REPO_ROOT / "src/avalan/tool" / f"{namespace}.py",)
    names = {
        f"{namespace}.{name}"
        for path in paths
        for name in findall(r"self\.__name__ = \"([^\"]+)\"", path.read_text())
    }
    return names


def _imported_modules(path: Path) -> set[str]:
    tree = parse_python(path.read_text())
    modules: set[str] = set()
    for node in walk_python(tree):
        if isinstance(node, ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _has_module_segment(modules: set[str], segment: str) -> bool:
    return any(segment in module.split(".") for module in modules)
