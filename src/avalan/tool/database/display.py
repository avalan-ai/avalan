from ...entities import (
    ToolCall,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallOutcome,
    ToolCallResult,
)
from ..display import (
    REDACTED_DISPLAY_VALUE,
    ToolDisplayDetail,
    ToolDisplayPreview,
    ToolDisplayProjection,
    is_sensitive_display_value,
    truncate_display_text,
)
from . import (
    DatabaseLock,
    DatabaseTask,
    QueryPlan,
    Table,
    TableKey,
    TableRelationship,
    TableSize,
)
from .settings import DatabaseToolSettings

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from re import DOTALL, IGNORECASE
from re import compile as compile_pattern
from typing import Any, TypeAlias, TypeGuard, TypeVar
from urllib.parse import unquote, urlsplit

_SQL_DISPLAY_LIMIT = 320
_SQL_LITERAL_LIMIT = 48
_LIST_DISPLAY_LIMIT = 8
_PREVIEW_ITEM_LIMIT = 5
_SQL_SUMMARY_VERBS = {
    "ALTER",
    "ANALYZE",
    "BEGIN",
    "CALL",
    "COMMIT",
    "COPY",
    "CREATE",
    "DELETE",
    "DESC",
    "DESCRIBE",
    "DROP",
    "EXEC",
    "EXECUTE",
    "EXPLAIN",
    "GRANT",
    "INSERT",
    "LISTEN",
    "MERGE",
    "NOTIFY",
    "REINDEX",
    "RESET",
    "REVOKE",
    "ROLLBACK",
    "SELECT",
    "SET",
    "SHOW",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
    "WITH",
}

DatabaseDisplayScalar: TypeAlias = None | bool | int | float | str
_T = TypeVar("_T")

_ACTIONS = {
    "count": "count",
    "inspect": "inspect",
    "keys": "inspect",
    "relationships": "inspect",
    "plan": "explain",
    "run": "query",
    "sample": "sample",
    "size": "measure",
    "tables": "list",
    "tasks": "list",
    "kill": "cancel",
    "locks": "inspect",
}
_SUMMARIES = {
    "count": "Count table rows.",
    "inspect": "Inspect table metadata.",
    "keys": "List table keys.",
    "relationships": "List table relationships.",
    "plan": "Explain SQL execution.",
    "run": "Run SQL statement.",
    "sample": "Sample table rows.",
    "size": "Summarize table storage.",
    "tables": "List database tables.",
    "tasks": "List database tasks.",
    "kill": "Cancel a database task.",
    "locks": "List database locks.",
}
_SENSITIVE_NAME = (
    r"[0-9A-Za-z_]*(?:api[_-]?key|authorization|bearer|credential|"
    r"password|passwd|private[_-]?key|secret|session[_-]?key|token)"
    r"[0-9A-Za-z_]*"
)
_SENSITIVE_IDENTIFIER = (
    rf'(?:"(?:{_SENSITIVE_NAME})"|`(?:{_SENSITIVE_NAME})`|'
    rf"\[(?:{_SENSITIVE_NAME})\]|\b(?:{_SENSITIVE_NAME})\b)"
)
_SENSITIVE_COMMENT_PATTERN = compile_pattern(
    rf"(--[^\r\n]*(?:{_SENSITIVE_NAME})[^\r\n]*"
    rf"|#[^\r\n]*(?:{_SENSITIVE_NAME})[^\r\n]*"
    rf"|/\*[\s\S]*?(?:{_SENSITIVE_NAME})[\s\S]*?\*/)",
    IGNORECASE,
)
_SENSITIVE_IDENTIFIER_CONTEXT_PATTERN = compile_pattern(
    _SENSITIVE_IDENTIFIER,
    IGNORECASE,
)
_SQL_LITERAL_PATTERN = compile_pattern(
    r"N?'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
    DOTALL,
)
_SQL_VERB_PATTERN = compile_pattern(r"^\s*([A-Za-z]+)")
_WHITESPACE_PATTERN = compile_pattern(r"\s+")


@dataclass(frozen=True, kw_only=True, slots=True)
class _DisplayText:
    value: str
    redacted: bool = False
    truncated: bool = False


def project_database_tool_display(
    *,
    call: ToolCall,
    settings: DatabaseToolSettings,
    dialect: str | None,
    outcome: ToolCallOutcome | None = None,
    sample_default_count: int | None = None,
) -> ToolDisplayProjection | None:
    assert isinstance(call, ToolCall)
    assert isinstance(settings, DatabaseToolSettings)
    operation = _operation_name(call.name)
    if outcome is not None:
        return _project_outcome(
            call=call,
            outcome=outcome,
            operation=operation,
            settings=settings,
            dialect=dialect,
        )
    return _project_call(
        call=call,
        operation=operation,
        settings=settings,
        dialect=dialect,
        sample_default_count=sample_default_count,
    )


def _project_call(
    *,
    call: ToolCall,
    operation: str,
    settings: DatabaseToolSettings,
    dialect: str | None,
    sample_default_count: int | None,
) -> ToolDisplayProjection:
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    target = _call_target(operation, arguments)
    sql = _call_sql(operation, arguments)
    details = [
        _detail("operation", operation),
        *_settings_details(settings, dialect),
        *_call_details(operation, arguments, sample_default_count),
    ]
    preview = None
    redacted = target.redacted
    truncated = target.truncated
    if sql is not None:
        preview = ToolDisplayPreview(
            content=sql.value,
            label="sql",
            redacted=sql.redacted,
            truncated=sql.truncated,
        )
        details.append(
            _detail(
                "sql",
                sql.value,
                redacted=sql.redacted,
                truncated=sql.truncated,
            )
        )
        command = _sql_command(sql.value)
        if command:
            details.append(_detail("sql_command", command))
        redacted = redacted or sql.redacted
        truncated = truncated or sql.truncated

    return ToolDisplayProjection(
        action=_ACTIONS.get(operation, "inspect"),
        label=call.name,
        target=target.value,
        scope="database",
        summary=_SUMMARIES.get(operation, "Use database tool."),
        severity="warning" if operation == "kill" else None,
        details=tuple(details),
        metrics=_call_metrics(operation, arguments, sample_default_count),
        preview=preview,
        redacted=redacted,
        truncated=truncated,
    )


def _project_outcome(
    *,
    call: ToolCall,
    outcome: ToolCallOutcome,
    operation: str,
    settings: DatabaseToolSettings,
    dialect: str | None,
) -> ToolDisplayProjection | None:
    if isinstance(outcome, ToolCallResult):
        return _project_result(
            call=call,
            result=outcome.result,
            operation=operation,
            settings=settings,
            dialect=dialect,
        )
    if isinstance(outcome, ToolCallError):
        return _project_error(
            call=call,
            error=outcome,
            operation=operation,
            settings=settings,
            dialect=dialect,
        )
    if isinstance(outcome, ToolCallDiagnostic):
        return _project_diagnostic(
            call=call,
            diagnostic=outcome,
            operation=operation,
            settings=settings,
            dialect=dialect,
        )
    return None


def _project_result(
    *,
    call: ToolCall,
    result: object,
    operation: str,
    settings: DatabaseToolSettings,
    dialect: str | None,
) -> ToolDisplayProjection:
    target = _result_target(operation, result, call)
    summary = _result_summary(operation, result)
    severity = _result_severity(operation, result)
    preview = _result_preview(operation, result)
    return ToolDisplayProjection(
        action=_ACTIONS.get(operation, "inspect"),
        label=call.name,
        target=target.value,
        scope="database",
        summary=summary,
        status="completed",
        outcome=_result_outcome(operation, result),
        severity=severity,
        details=tuple(
            [
                _detail("operation", operation),
                *_settings_details(settings, dialect),
                *_result_details(operation, result, call),
            ]
        ),
        metrics=_result_metrics(operation, result),
        preview=preview,
        redacted=target.redacted or (preview.redacted if preview else False),
        truncated=(
            target.truncated or (preview.truncated if preview else False)
        ),
    )


def _project_error(
    *,
    call: ToolCall,
    error: ToolCallError,
    operation: str,
    settings: DatabaseToolSettings,
    dialect: str | None,
) -> ToolDisplayProjection:
    target = _call_target(
        operation,
        call.arguments if isinstance(call.arguments, dict) else {},
    )
    return ToolDisplayProjection(
        action=_ACTIONS.get(operation, "inspect"),
        label=call.name,
        target=target.value,
        scope="database",
        summary="Database operation failed.",
        status="error",
        outcome=error.error_type,
        severity="error",
        details=tuple(
            [
                _detail("operation", operation),
                *_settings_details(settings, dialect),
                _detail("error_type", error.error_type),
            ]
        ),
        redacted=target.redacted,
        truncated=target.truncated,
    )


def _project_diagnostic(
    *,
    call: ToolCall,
    diagnostic: ToolCallDiagnostic,
    operation: str,
    settings: DatabaseToolSettings,
    dialect: str | None,
) -> ToolDisplayProjection:
    target = _call_target(
        operation,
        call.arguments if isinstance(call.arguments, dict) else {},
    )
    return ToolDisplayProjection(
        action="skip",
        label=call.name,
        target=target.value,
        scope="database",
        summary="Database operation was not executed.",
        status=diagnostic.status.value,
        outcome=diagnostic.code.value,
        severity="warning",
        details=tuple(
            [
                _detail("operation", operation),
                *_settings_details(settings, dialect),
                _detail("stage", diagnostic.stage.value),
            ]
        ),
        redacted=target.redacted,
        truncated=target.truncated,
    )


def _operation_name(name: str) -> str:
    if "." not in name:
        return name
    return name.rsplit(".", 1)[1]


def _settings_details(
    settings: DatabaseToolSettings, dialect: str | None
) -> tuple[ToolDisplayDetail, ...]:
    details = [
        _detail("read_only", settings.read_only),
    ]
    if dialect:
        details.insert(0, _detail("dialect", dialect))
    database_name = _database_name_from_dsn(settings.dsn)
    if database_name:
        details.insert(1 if dialect else 0, _detail("database", database_name))
    return tuple(details)


def _call_details(
    operation: str,
    arguments: Mapping[str, object],
    sample_default_count: int | None,
) -> tuple[ToolDisplayDetail, ...]:
    details: list[ToolDisplayDetail] = []
    schema = _string_argument(arguments, "schema")
    if schema:
        details.append(_detail("schema", schema))
    table = _table_argument(operation, arguments)
    if table:
        details.append(_detail("table", table))
    if operation == "inspect":
        tables = _table_names(arguments)
        if tables:
            details.append(_detail("tables", _join_limited(tables)))
    if operation == "sample":
        count = _positive_int_argument(arguments, "count")
        effective_count = count if count is not None else sample_default_count
        if effective_count is not None:
            details.append(_detail("limit", effective_count))
        columns = _string_sequence_argument(arguments, "columns")
        if columns:
            details.append(_detail("columns", _join_limited(columns)))
        conditions = _string_argument(arguments, "conditions")
        if conditions:
            safe_conditions = _safe_sql_text(conditions)
            details.append(
                _detail(
                    "conditions",
                    safe_conditions.value,
                    redacted=safe_conditions.redacted,
                    truncated=safe_conditions.truncated,
                )
            )
        order = arguments.get("order")
        if isinstance(order, Mapping) and order:
            details.append(_detail("order", _mapping_summary(order)))
    if operation == "tasks":
        running_for = _positive_int_argument(arguments, "running_for")
        if running_for is not None:
            details.append(_detail("running_for_seconds", running_for))
    if operation == "kill":
        task_id = _string_argument(arguments, "task_id")
        if task_id:
            details.append(_detail("task_id", task_id))
        details.append(_detail("intent", "cancel database task"))
    return tuple(details)


def _call_metrics(
    operation: str,
    arguments: Mapping[str, object],
    sample_default_count: int | None,
) -> dict[str, DatabaseDisplayScalar]:
    metrics: dict[str, DatabaseDisplayScalar] = {}
    if operation == "sample":
        count = _positive_int_argument(arguments, "count")
        effective_count = count if count is not None else sample_default_count
        if effective_count is not None:
            metrics["limit"] = effective_count
    if operation == "inspect":
        tables = _table_names(arguments)
        if tables:
            metrics["tables"] = len(tables)
    return metrics


def _call_target(
    operation: str, arguments: Mapping[str, object]
) -> _DisplayText:
    if operation in {"count", "keys", "relationships", "sample", "size"}:
        return _safe_target(_qualified_table_target(arguments))
    if operation == "inspect":
        tables = _table_names(arguments)
        schema = _string_argument(arguments, "schema")
        if tables:
            qualified = [
                _qualify_table(schema=schema, table_name=table)
                for table in tables
            ]
            return _safe_target(_join_limited(qualified))
        return _DisplayText(value="tables")
    if operation in {"plan", "run"}:
        sql = _call_sql(operation, arguments)
        return (
            _DisplayText(value="SQL statement")
            if sql
            else _DisplayText(value="SQL")
        )
    if operation == "tables":
        return _DisplayText(value="tables")
    if operation == "tasks":
        return _DisplayText(value="tasks")
    if operation == "kill":
        task_id = _string_argument(arguments, "task_id")
        return _safe_target(f"task {task_id}" if task_id else "task")
    if operation == "locks":
        return _DisplayText(value="locks")
    return _DisplayText(value="database")


def _call_sql(
    operation: str, arguments: Mapping[str, object]
) -> _DisplayText | None:
    if operation not in {"plan", "run"}:
        return None
    sql = _string_argument(arguments, "sql")
    if sql is None:
        return None
    return _safe_sql_text(sql)


def _result_target(
    operation: str,
    result: object,
    call: ToolCall,
) -> _DisplayText:
    if isinstance(result, TableSize):
        return _safe_target(result.name)
    if isinstance(result, QueryPlan):
        return _call_target(
            operation,
            call.arguments if isinstance(call.arguments, dict) else {},
        )
    if operation == "inspect" and _is_table_sequence(result):
        return _safe_target(result[0].name if len(result) == 1 else "tables")
    if isinstance(result, Sequence) and not isinstance(
        result, str | bytes | bytearray
    ):
        first = result[0] if result else None
        if isinstance(first, Table):
            return _safe_target(first.name if len(result) == 1 else "tables")
        return _call_target(
            operation,
            call.arguments if isinstance(call.arguments, dict) else {},
        )
    return _call_target(
        operation,
        call.arguments if isinstance(call.arguments, dict) else {},
    )


def _result_summary(operation: str, result: object) -> str:
    if operation == "kill":
        return (
            "Database task cancellation request was accepted."
            if result is True
            else "Database task cancellation did not complete."
        )
    if operation == "locks" and _blocking_lock_count(result):
        return "Found blocking database locks."
    return _SUMMARIES.get(operation, "Database operation completed.")


def _result_outcome(operation: str, result: object) -> str:
    if operation == "kill":
        return "cancel_requested" if result is True else "not_cancelled"
    if operation == "locks" and _blocking_lock_count(result):
        return "blocking"
    return "result"


def _result_severity(operation: str, result: object) -> str | None:
    if operation == "kill" and result is True:
        return "warning"
    if operation == "locks" and _blocking_lock_count(result):
        return "warning"
    return None


def _result_details(
    operation: str, result: object, call: ToolCall
) -> tuple[ToolDisplayDetail, ...]:
    details: list[ToolDisplayDetail] = []
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    if operation in {"plan", "run"}:
        sql = _call_sql(operation, arguments)
        if sql is not None:
            details.append(
                _detail(
                    "sql",
                    sql.value,
                    redacted=sql.redacted,
                    truncated=sql.truncated,
                )
            )
            command = _sql_command(sql.value)
            if command:
                details.append(_detail("sql_command", command))
    if operation == "inspect":
        tables = (
            _result_table_names(result)
            if _is_table_sequence(result)
            else _table_names(arguments)
        )
        if tables:
            details.append(_detail("tables", _join_limited(tables)))
    if operation == "plan" and isinstance(result, QueryPlan):
        details.append(_detail("plan_dialect", result.dialect))
        details.append(_detail("steps", len(result.steps)))
    if operation == "size" and isinstance(result, TableSize):
        details.append(_detail("table", result.name))
        if result.schema:
            details.append(_detail("schema", result.schema))
        for metric in result.metrics[:_LIST_DISPLAY_LIMIT]:
            value = metric.human_readable or metric.bytes
            details.append(_detail(f"{metric.category}_size", value))
    if operation == "tables" and isinstance(result, Mapping):
        details.append(_detail("schemas", len(result)))
        details.append(_detail("tables", _table_mapping_count(result)))
    if operation == "tasks" and _is_database_task_sequence(result):
        details.append(_detail("tasks", len(result)))
    if operation == "locks" and _is_database_lock_sequence(result):
        details.append(_detail("locks", len(result)))
        blocking = _blocking_lock_count(result)
        if blocking:
            details.append(_detail("blocking_locks", blocking))
    return tuple(details)


def _result_table_names(result: object) -> tuple[str, ...]:
    if not _is_table_sequence(result):
        return ()
    return tuple(table.name for table in result if table.name)


def _database_name_from_dsn(dsn: str) -> str | None:
    assert isinstance(dsn, str)
    source = dsn.strip()
    if not source:
        return None

    if "://" in source:
        parsed = urlsplit(source)
        path = unquote(parsed.path or "").strip("/")
        if not path:
            return None
        if "/" in path:
            return path.rsplit("/", 1)[-1] or None
        return path

    path = source.split("?", 1)[0].rstrip("/")
    if "/" not in path:
        return None
    return unquote(path.rsplit("/", 1)[-1]) or None


def _result_metrics(
    operation: str, result: object
) -> dict[str, DatabaseDisplayScalar]:
    metrics: dict[str, DatabaseDisplayScalar] = {}
    if operation == "count" and isinstance(result, int):
        metrics["rows"] = result
    elif operation in {"run", "sample"}:
        count = _sequence_count(result)
        if count is not None:
            metrics["rows"] = count
    elif operation == "inspect" and _is_table_sequence(result):
        metrics["tables"] = len(result)
        metrics["columns"] = sum(len(table.columns) for table in result)
        metrics["foreign_keys"] = sum(
            len(table.foreign_keys) for table in result
        )
    elif operation == "keys" and _is_table_key_sequence(result):
        metrics["keys"] = len(result)
    elif operation == "relationships" and _is_relationship_sequence(result):
        metrics["relationships"] = len(result)
        metrics["incoming"] = sum(
            1
            for relationship in result
            if relationship.direction == "incoming"
        )
        metrics["outgoing"] = sum(
            1
            for relationship in result
            if relationship.direction == "outgoing"
        )
    elif operation == "plan" and isinstance(result, QueryPlan):
        metrics["steps"] = len(result.steps)
    elif operation == "size" and isinstance(result, TableSize):
        for metric in result.metrics:
            if metric.bytes is not None:
                metrics[f"{metric.category}_bytes"] = metric.bytes
    elif operation == "tables" and isinstance(result, Mapping):
        metrics["schemas"] = len(result)
        metrics["tables"] = _table_mapping_count(result)
    elif operation == "tasks" and _is_database_task_sequence(result):
        metrics["tasks"] = len(result)
    elif operation == "locks" and _is_database_lock_sequence(result):
        metrics["locks"] = len(result)
        metrics["blocking_locks"] = _blocking_lock_count(result)
    return metrics


def _result_preview(
    operation: str, result: object
) -> ToolDisplayPreview | None:
    if operation == "tasks" and _is_database_task_sequence(result):
        previews = tuple(
            _task_preview(task) for task in result[:_PREVIEW_ITEM_LIMIT]
        )
        content = "\n".join(preview.value for preview in previews)
        if content:
            return ToolDisplayPreview(
                content=content,
                label="tasks",
                redacted=any(preview.redacted for preview in previews),
                truncated=any(preview.truncated for preview in previews),
            )
        return None
    if operation == "locks" and _is_database_lock_sequence(result):
        previews = tuple(
            _lock_preview(lock) for lock in result[:_PREVIEW_ITEM_LIMIT]
        )
        content = "\n".join(preview.value for preview in previews)
        if content:
            return ToolDisplayPreview(
                content=content,
                label="locks",
                redacted=any(preview.redacted for preview in previews),
                truncated=any(preview.truncated for preview in previews),
            )
        return None
    return None


def _task_preview(task: DatabaseTask) -> _DisplayText:
    parts = [f"id={task.id}"]
    redacted = False
    truncated = False
    if task.state:
        parts.append(f"state={task.state}")
    if task.duration is not None:
        parts.append(f"duration={task.duration}s")
    if task.query:
        query = _safe_sql_text(task.query)
        parts.append(f"query={query.value}")
        redacted = redacted or query.redacted
        truncated = truncated or query.truncated
    value = " ".join(parts)
    if len(value) > _SQL_DISPLAY_LIMIT:
        truncated = True
        value = truncate_display_text(value, _SQL_DISPLAY_LIMIT)
    return _DisplayText(value=value, redacted=redacted, truncated=truncated)


def _lock_preview(lock: DatabaseLock) -> _DisplayText:
    parts: list[str] = []
    redacted = False
    truncated = False
    if lock.pid:
        parts.append(f"pid={lock.pid}")
    if lock.lock_target:
        parts.append(f"target={lock.lock_target}")
    if lock.mode:
        parts.append(f"mode={lock.mode}")
    if lock.granted is not None:
        parts.append(f"granted={lock.granted}")
    if lock.blocking:
        blocking = ",".join(lock.blocking[:_LIST_DISPLAY_LIMIT])
        parts.append(f"blocking={blocking}")
    if lock.query:
        query = _safe_sql_text(lock.query)
        parts.append(f"query={query.value}")
        redacted = redacted or query.redacted
        truncated = truncated or query.truncated
    value = " ".join(parts)
    if len(value) > _SQL_DISPLAY_LIMIT:
        truncated = True
        value = truncate_display_text(value, _SQL_DISPLAY_LIMIT)
    return _DisplayText(value=value, redacted=redacted, truncated=truncated)


def _table_argument(
    operation: str, arguments: Mapping[str, object]
) -> str | None:
    if operation in {"count", "keys", "relationships", "sample", "size"}:
        return _qualified_table_target(arguments)
    return None


def _qualified_table_target(arguments: Mapping[str, object]) -> str | None:
    table_name = _string_argument(arguments, "table_name")
    if not table_name:
        return None
    schema = _string_argument(arguments, "schema")
    return _qualify_table(schema=schema, table_name=table_name)


def _qualify_table(*, schema: str | None, table_name: str) -> str:
    if schema and "." not in table_name:
        return f"{schema}.{table_name}"
    return table_name


def _table_names(arguments: Mapping[str, object]) -> tuple[str, ...]:
    value = arguments.get("table_names")
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(
        value, bytes | bytearray
    ):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _safe_sql_text(sql: str) -> _DisplayText:
    assert isinstance(sql, str)
    raw = sql.strip()
    redacted = False
    truncated = False
    if not raw:
        return _DisplayText(value="")

    normalized = _WHITESPACE_PATTERN.sub(" ", raw)
    if _has_sensitive_sql_context(raw):
        return _DisplayText(
            value=_sensitive_sql_summary(normalized),
            redacted=True,
            truncated=len(normalized) > _SQL_DISPLAY_LIMIT,
        )

    def bound_literal(match: Any) -> str:
        nonlocal redacted, truncated
        literal = str(match.group(0))
        prefix = "N" if literal.startswith("N'") else ""
        quoted = literal[1:] if prefix else literal
        quote = quoted[0]
        content = quoted[1:-1]
        if is_sensitive_display_value(content):
            redacted = True
            return f"{prefix}{quote}{REDACTED_DISPLAY_VALUE}{quote}"
        if len(content) <= _SQL_LITERAL_LIMIT:
            return literal
        truncated = True
        return (
            f"{prefix}{quote}"
            f"{truncate_display_text(content, _SQL_LITERAL_LIMIT)}"
            f"{quote}"
        )

    normalized = _SQL_LITERAL_PATTERN.sub(bound_literal, normalized)
    if len(normalized) > _SQL_DISPLAY_LIMIT:
        truncated = True
        normalized = truncate_display_text(normalized, _SQL_DISPLAY_LIMIT)
    return _DisplayText(
        value=normalized, redacted=redacted, truncated=truncated
    )


def _sql_command(sql: str) -> str | None:
    match = _SQL_VERB_PATTERN.search(sql)
    if not match:
        return None
    return match.group(1).upper()


def _has_sensitive_sql_context(sql: str) -> bool:
    return bool(
        _SENSITIVE_COMMENT_PATTERN.search(sql)
        or _SENSITIVE_IDENTIFIER_CONTEXT_PATTERN.search(sql)
    )


def _sensitive_sql_summary(sql: str) -> str:
    match = _SQL_VERB_PATTERN.search(sql)
    if not match:
        return REDACTED_DISPLAY_VALUE
    verb = match.group(1).upper()
    if verb not in _SQL_SUMMARY_VERBS:
        return REDACTED_DISPLAY_VALUE
    return f"{verb} {REDACTED_DISPLAY_VALUE}"


def _safe_target(value: str | None) -> _DisplayText:
    if not value:
        return _DisplayText(value="database")
    truncated = len(value) > _SQL_DISPLAY_LIMIT
    return _DisplayText(
        value=truncate_display_text(value, _SQL_DISPLAY_LIMIT),
        truncated=truncated,
    )


def _detail(
    label: str,
    value: DatabaseDisplayScalar,
    *,
    redacted: bool = False,
    truncated: bool = False,
) -> ToolDisplayDetail:
    return ToolDisplayDetail(
        label=label,
        value=value,
        redacted=redacted,
        truncated=truncated,
    )


def _string_argument(arguments: Mapping[str, object], name: str) -> str | None:
    value = arguments.get(name)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _positive_int_argument(
    arguments: Mapping[str, object], name: str
) -> int | None:
    value = arguments.get(name)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _string_sequence_argument(
    arguments: Mapping[str, object], name: str
) -> tuple[str, ...]:
    value = arguments.get(name)
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence) and not isinstance(
        value, bytes | bytearray
    ):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


def _join_limited(values: Sequence[str]) -> str:
    items = list(values[:_LIST_DISPLAY_LIMIT])
    suffix = ", ..." if len(values) > _LIST_DISPLAY_LIMIT else ""
    return ", ".join(items) + suffix


def _mapping_summary(value: Mapping[object, object]) -> str:
    items = []
    for index, (key, item) in enumerate(value.items()):
        if index >= _LIST_DISPLAY_LIMIT:
            items.append("...")
            break
        items.append(f"{key}: {item}")
    return ", ".join(items)


def _sequence_count(value: object) -> int | None:
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return len(value)
    return None


def _table_mapping_count(value: Mapping[object, object]) -> int:
    count = 0
    for tables in value.values():
        if isinstance(tables, Sequence) and not isinstance(
            tables, str | bytes | bytearray
        ):
            count += len(tables)
    return count


def _blocking_lock_count(value: object) -> int:
    if not _is_database_lock_sequence(value):
        return 0
    return sum(1 for lock in value if lock.blocking or lock.granted is False)


def _is_table_sequence(value: object) -> TypeGuard[Sequence[Table]]:
    return _is_sequence_of(value, Table)


def _is_table_key_sequence(value: object) -> TypeGuard[Sequence[TableKey]]:
    return _is_sequence_of(value, TableKey)


def _is_relationship_sequence(
    value: object,
) -> TypeGuard[Sequence[TableRelationship]]:
    return _is_sequence_of(value, TableRelationship)


def _is_database_task_sequence(
    value: object,
) -> TypeGuard[Sequence[DatabaseTask]]:
    return _is_sequence_of(value, DatabaseTask)


def _is_database_lock_sequence(
    value: object,
) -> TypeGuard[Sequence[DatabaseLock]]:
    return _is_sequence_of(value, DatabaseLock)


def _is_sequence_of(
    value: object, expected_type: type[_T]
) -> TypeGuard[Sequence[_T]]:
    if not isinstance(value, Sequence) or isinstance(
        value, str | bytes | bytearray
    ):
        return False
    return all(isinstance(item, expected_type) for item in value)
