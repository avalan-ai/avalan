from types import SimpleNamespace
from typing import Any, cast

from pytest import MonkeyPatch

from avalan.entities import (
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    ToolManagerSettings,
)
from avalan.tool import ToolSet
from avalan.tool.database import (
    DatabaseLock,
    DatabaseTask,
    DatabaseTool,
    DatabaseToolSettings,
    QueryPlan,
    Table,
    TableKey,
    TableRelationship,
    TableSize,
    TableSizeMetric,
)
from avalan.tool.database import display as database_display
from avalan.tool.database.count import DatabaseCountTool
from avalan.tool.database.inspect import DatabaseInspectTool
from avalan.tool.database.keys import DatabaseKeysTool
from avalan.tool.database.kill import DatabaseKillTool
from avalan.tool.database.locks import DatabaseLocksTool
from avalan.tool.database.plan import DatabasePlanTool
from avalan.tool.database.relationships import DatabaseRelationshipsTool
from avalan.tool.database.run import DatabaseRunTool
from avalan.tool.database.sample import DatabaseSampleTool
from avalan.tool.database.size import DatabaseSizeTool
from avalan.tool.database.tables import DatabaseTablesTool
from avalan.tool.database.tasks import DatabaseTasksTool
from avalan.tool.display import ToolDisplayDetail, ToolDisplayProjection
from avalan.tool.manager import ToolManager

_DSN = "postgresql+asyncpg://db_user:super-secret-password@example.com/app"


def test_database_descriptors_expose_call_projection() -> None:
    manager = _manager()

    for name, arguments in _call_arguments().items():
        descriptor = manager.describe_tool(name)

        assert descriptor is not None
        assert descriptor.display_projector is not None
        projection = descriptor.project_display(
            ToolCall(id=f"{name}-1", name=name, arguments=arguments)
        )

        assert isinstance(projection, ToolDisplayProjection)
        assert projection.scope == "database"
        assert projection.label == name
        assert _detail_value(projection, "operation") == name.rsplit(".", 1)[1]
        assert _detail_value(projection, "database") == "app"
        assert "super-secret-password" not in _projection_text(projection)
        assert "db_user" not in _projection_text(projection)


def test_table_targeted_calls_show_schema_and_table() -> None:
    manager = _manager()
    cases: dict[str, dict[str, Any]] = {
        "database.count": {"table_name": "analytics.orders"},
        "database.inspect": {
            "schema": "analytics",
            "table_names": ["orders"],
        },
        "database.keys": {"schema": "analytics", "table_name": "orders"},
        "database.relationships": {
            "schema": "analytics",
            "table_name": "orders",
        },
        "database.sample": {"table_name": "analytics.orders", "count": 2},
        "database.size": {"table_name": "analytics.orders"},
    }

    for name, arguments in cases.items():
        projection = _project_call(manager, name, arguments)

        assert projection.target == "analytics.orders"
        detail_value = _detail_value(projection, "table")
        if detail_value is None:
            detail_value = _detail_value(projection, "tables")
        assert "orders" in str(detail_value)


def test_query_calls_show_bounded_redacted_sql_summary() -> None:
    manager = _manager()
    long_value = "x" * 600
    sql = (
        "SELECT id, name FROM users WHERE password_hash = 'hunter2' "
        "AND api_token = 'token-secret' "
        "AND note = 'bearer abc' "
        f"AND description = '{long_value}'"
    )

    for name in ("database.run", "database.plan"):
        projection = _project_call(manager, name, {"sql": sql})

        assert projection.preview is not None
        assert projection.preview.content == "SELECT [redacted]"
        assert "hunter2" not in projection.preview.content
        assert "token-secret" not in projection.preview.content
        assert "bearer abc" not in projection.preview.content
        assert long_value not in projection.preview.content
        assert "[redacted]" in projection.preview.content
        assert projection.redacted
        assert projection.truncated
        assert len(projection.preview.content) <= 320
        assert _detail_value(projection, "sql_command") == "SELECT"


def test_query_calls_redact_sensitive_sql_edge_cases() -> None:
    manager = _manager()
    cases = (
        ("-- password hunter2\nSELECT 1", "hunter2"),
        ("SELECT * FROM users WHERE \"password\" = 'hunter2'", "hunter2"),
        ("SELECT * FROM users WHERE token = 123456", "123456"),
        ("SELECT * FROM users WHERE secret = $$abc123$$", "abc123"),
        ("SELECT * FROM users WHERE api_key BETWEEN 100 AND 200", "100"),
        ("SELECT * FROM users WHERE password = E'hunter2'", "hunter2"),
        ("SELECT * FROM users WHERE secret = $tag$abc123$tag$", "abc123"),
        (
            "INSERT INTO users (api_token, note) VALUES ('abc123', 'safe')",
            "abc123",
        ),
        ("SELECT 1 # password hunter2", "hunter2"),
        ("hunter2 -- password marker", "hunter2"),
        ("abc123 /* token marker */ SELECT 1", "abc123"),
    )

    for sql, leaked_value in cases:
        with_projection = _project_call(manager, "database.run", {"sql": sql})

        assert with_projection.preview is not None
        assert leaked_value not in with_projection.preview.content
        assert "[redacted]" in with_projection.preview.content
        assert with_projection.redacted


def test_operational_calls_show_task_and_lock_intent() -> None:
    manager = _manager()

    kill = _project_call(manager, "database.kill", {"task_id": "12345"})
    tasks = _project_call(manager, "database.tasks", {"running_for": 30})
    locks = _project_call(manager, "database.locks", {})

    assert kill.action == "cancel"
    assert kill.severity == "warning"
    assert kill.target == "task 12345"
    assert _detail_value(kill, "intent") == "cancel database task"
    assert tasks.target == "tasks"
    assert _detail_value(tasks, "running_for_seconds") == 30
    assert locks.target == "locks"


def test_unusual_call_arguments_use_safe_fallbacks() -> None:
    manager = _manager()

    inspect_without_tables = _project_call(
        manager, "database.inspect", {"table_names": 123}
    )
    inspect_string_table = _project_call(
        manager,
        "database.inspect",
        {"schema": "analytics", "table_names": "orders"},
    )
    count_without_table = _project_call(manager, "database.count", {})
    run_without_sql = _project_call(manager, "database.run", {})
    sample_string_columns = _project_call(
        manager,
        "database.sample",
        {"table_name": "users", "columns": "id"},
    )
    tasks_bool_filter = _project_call(
        manager, "database.tasks", {"running_for": True}
    )
    tasks_negative_filter = _project_call(
        manager, "database.tasks", {"running_for": -1}
    )

    assert inspect_without_tables.target == "tables"
    assert inspect_string_table.target == "analytics.orders"
    assert _detail_value(inspect_string_table, "tables") == "orders"
    assert count_without_table.target == "database"
    assert run_without_sql.target == "SQL"
    assert run_without_sql.preview is None
    assert _detail_value(sample_string_columns, "columns") == "id"
    assert _detail_value(sample_string_columns, "limit") == 10
    assert _detail_value(tasks_bool_filter, "running_for_seconds") is None
    assert _detail_value(tasks_negative_filter, "running_for_seconds") is None


def test_sample_call_details_summarize_filters_and_ordering() -> None:
    manager = _manager()
    long_literal = "x" * 96
    order = {f"col_{index}": "desc" for index in range(10)}

    projection = _project_call(
        manager,
        "database.sample",
        {
            "table_name": "events",
            "count": -5,
            "columns": ["id", "", 7, "email"],
            "conditions": f"status = 'active' AND note = '{long_literal}'",
            "order": order,
        },
    )
    conditions_detail = _detail(projection, "conditions")

    assert _detail_value(projection, "limit") == 10
    assert _detail_value(projection, "columns") == "id, email"
    assert conditions_detail is not None
    assert conditions_detail.truncated
    assert long_literal not in str(conditions_detail.value)
    assert (
        _detail_value(projection, "order")
        == "col_0: desc, col_1: desc, col_2: desc, col_3: desc, "
        "col_4: desc, col_5: desc, col_6: desc, col_7: desc, ..."
    )
    assert projection.metrics["limit"] == 10


def test_direct_projector_handles_unknown_operations_and_outcomes() -> None:
    unknown = _project_direct_call("unknown", {})
    unsupported = database_display.project_database_tool_display(
        call=ToolCall(
            id="database-run-unsupported",
            name="database.run",
            arguments={"sql": "SELECT 1"},
        ),
        settings=_settings(),
        dialect="postgresql",
        outcome=cast(Any, object()),
    )

    assert unknown.action == "inspect"
    assert unknown.target == "database"
    assert unknown.summary == "Use database tool."
    assert unsupported is None


def test_sql_literal_sanitization_bounds_literals(
    monkeypatch: MonkeyPatch,
) -> None:
    def never_sensitive_context(sql: str) -> bool:
        assert sql
        return False

    monkeypatch.setattr(
        database_display,
        "_has_sensitive_sql_context",
        never_sensitive_context,
    )
    long_literal = "value-" * 20

    projection = _project_direct_call(
        "database.run",
        {"sql": f"SELECT 'active', N'credential value', '{long_literal}'"},
    )

    assert projection.preview is not None
    assert "'active'" in projection.preview.content
    assert "N'[redacted]'" in projection.preview.content
    assert "credential value" not in projection.preview.content
    assert long_literal not in projection.preview.content
    assert projection.preview.redacted
    assert projection.preview.truncated


def test_long_sql_without_sensitive_context_is_truncated() -> None:
    manager = _manager()
    sql = "SELECT " + ", ".join(f"column_{index}" for index in range(80))

    projection = _project_call(manager, "database.run", {"sql": sql})

    assert projection.preview is not None
    assert projection.preview.truncated
    assert len(projection.preview.content) <= 320
    assert sql not in projection.preview.content


def test_typed_terminal_results_include_bounded_counts() -> None:
    manager = _manager()

    cases: list[tuple[str, dict[str, Any], object, str, int]] = [
        ("database.count", {"table_name": "users"}, 7, "rows", 7),
        (
            "database.inspect",
            {"table_names": ["users"]},
            [
                Table(
                    name="users",
                    columns={"id": "INTEGER", "name": "TEXT"},
                    foreign_keys=[],
                )
            ],
            "tables",
            1,
        ),
        (
            "database.keys",
            {"table_name": "users"},
            [TableKey(type="primary", name="users_pkey", columns=("id",))],
            "keys",
            1,
        ),
        (
            "database.relationships",
            {"table_name": "books"},
            [
                TableRelationship(
                    direction="outgoing",
                    local_columns=("author_id",),
                    related_table="authors",
                    related_columns=("id",),
                    constraint_name="books_author_id_fkey",
                )
            ],
            "relationships",
            1,
        ),
        (
            "database.plan",
            {"sql": "SELECT * FROM users"},
            QueryPlan(dialect="postgresql", steps=[{"Plan": "Seq Scan"}]),
            "steps",
            1,
        ),
        (
            "database.run",
            {"sql": "SELECT id FROM users"},
            [{"id": 1}, {"id": 2}],
            "rows",
            2,
        ),
        (
            "database.sample",
            {"table_name": "users", "count": 2},
            [{"id": 1}, {"id": 2}],
            "rows",
            2,
        ),
        (
            "database.size",
            {"table_name": "users"},
            TableSize(
                name="users",
                schema=None,
                metrics=(
                    TableSizeMetric(
                        category="total",
                        bytes=2048,
                        human_readable="2.0 KiB",
                    ),
                ),
            ),
            "total_bytes",
            2048,
        ),
        (
            "database.tables",
            {},
            {"public": ["users", "books"]},
            "tables",
            2,
        ),
        (
            "database.tasks",
            {},
            [
                DatabaseTask(
                    id="10",
                    user="db_user",
                    state="active",
                    query="SELECT * FROM users WHERE token = 'secret'",
                    duration=42,
                )
            ],
            "tasks",
            1,
        ),
    ]

    for name, arguments, result, metric, expected in cases:
        projection = _project_result(manager, name, arguments, result)

        assert projection.status == "completed"
        assert projection.metrics[metric] == expected
        assert "secret" not in _projection_text(projection)


def test_terminal_results_keep_database_request_context() -> None:
    manager = _manager()

    query = _project_result(
        manager,
        "database.run",
        {"sql": "SELECT id FROM users"},
        [{"id": 1}, {"id": 2}],
    )
    inspected = _project_result(
        manager,
        "database.inspect",
        {"table_names": ["users", "orders"]},
        [
            Table(name="users", columns={"id": "INTEGER"}, foreign_keys=[]),
            Table(name="orders", columns={"id": "INTEGER"}, foreign_keys=[]),
        ],
    )

    assert _detail_value(query, "database") == "app"
    assert _detail_value(query, "sql") == "SELECT id FROM users"
    assert _detail_value(query, "sql_command") == "SELECT"
    assert query.metrics["rows"] == 2
    assert _detail_value(inspected, "database") == "app"
    assert _detail_value(inspected, "tables") == "users, orders"
    assert inspected.metrics["tables"] == 2


def test_inspect_terminal_results_can_supply_table_context() -> None:
    manager = _manager()

    inferred = _project_result(
        manager,
        "database.inspect",
        {},
        [
            Table(name="users", columns={"id": "INTEGER"}, foreign_keys=[]),
            Table(name="orders", columns={"id": "INTEGER"}, foreign_keys=[]),
        ],
    )
    invalid = _project_result(manager, "database.inspect", {}, "not tables")

    assert _detail_value(inferred, "tables") == "users, orders"
    assert _detail_value(invalid, "tables") is None


def test_database_name_from_dsn_handles_edge_cases() -> None:
    assert database_display._database_name_from_dsn(" ") is None
    assert (
        database_display._database_name_from_dsn("postgresql://host") is None
    )
    assert (
        database_display._database_name_from_dsn(
            "sqlite:////tmp/checkmate/app.db"
        )
        == "app.db"
    )
    assert database_display._database_name_from_dsn("memory") is None
    assert (
        database_display._database_name_from_dsn(
            "postgresql://user@host/checkmate39?sslmode=require"
        )
        == "checkmate39"
    )
    assert (
        database_display._database_name_from_dsn(
            "user@host/checkmate%2039?sslmode=require"
        )
        == "checkmate 39"
    )


def test_size_results_include_schema_detail() -> None:
    manager = _manager()

    projection = _project_result(
        manager,
        "database.size",
        {"table_name": "analytics.events"},
        TableSize(
            name="events",
            schema="analytics",
            metrics=(
                TableSizeMetric(
                    category="data",
                    bytes=4096,
                    human_readable=None,
                ),
            ),
        ),
    )

    assert projection.target == "events"
    assert _detail_value(projection, "schema") == "analytics"
    assert _detail_value(projection, "data_size") == 4096
    assert projection.metrics["data_bytes"] == 4096


def test_kill_and_blocking_locks_use_elevated_terminal_severity() -> None:
    manager = _manager()

    kill = _project_result(
        manager, "database.kill", {"task_id": "12345"}, True
    )
    locks = _project_result(
        manager,
        "database.locks",
        {},
        [
            DatabaseLock(
                pid="10",
                user="db_user",
                lock_type="relation",
                lock_target="public.users",
                mode="AccessExclusiveLock",
                granted=False,
                blocking=("20",),
                state="active",
                query="UPDATE users SET token = 'secret'",
            )
        ],
    )

    assert kill.outcome == "cancel_requested"
    assert kill.severity == "warning"
    assert "request was accepted" in (kill.summary or "")
    assert locks.outcome == "blocking"
    assert locks.severity == "warning"
    assert locks.metrics["blocking_locks"] == 1
    assert locks.preview is not None
    assert locks.preview.redacted
    assert locks.redacted
    assert "secret" not in locks.preview.content


def test_task_preview_carries_sql_redaction_flags() -> None:
    manager = _manager()

    projection = _project_result(
        manager,
        "database.tasks",
        {},
        [
            DatabaseTask(
                id="10",
                user="db_user",
                state="active",
                query="SELECT * FROM users WHERE token = 'secret'",
                duration=42,
            )
        ],
    )

    assert projection.preview is not None
    assert projection.preview.redacted
    assert projection.redacted
    assert "secret" not in projection.preview.content


def test_empty_terminal_previews_are_omitted() -> None:
    manager = _manager()

    tasks = _project_result(manager, "database.tasks", {}, [])
    locks = _project_result(manager, "database.locks", {}, [])
    not_locks = _project_result(manager, "database.locks", {}, "not locks")
    non_sequence_rows = _project_result(
        manager, "database.run", {"sql": "SELECT 1"}, {"id": 1}
    )

    assert tasks.preview is None
    assert tasks.metrics["tasks"] == 0
    assert locks.preview is None
    assert locks.metrics["locks"] == 0
    assert locks.metrics["blocking_locks"] == 0
    assert not_locks.outcome == "result"
    assert not_locks.metrics == {}
    assert "rows" not in non_sequence_rows.metrics


def test_task_and_lock_previews_are_bounded() -> None:
    manager = _manager()
    long_value = "x" * 500

    task_projection = _project_result(
        manager,
        "database.tasks",
        {},
        [
            DatabaseTask(
                id=f"task-{long_value}",
                user=None,
                state=None,
                query=" ",
                duration=None,
            )
        ],
    )
    lock_projection = _project_result(
        manager,
        "database.locks",
        {},
        [
            DatabaseLock(
                pid="10",
                user=None,
                lock_type="relation",
                lock_target=f"public.{long_value}",
                mode="AccessExclusiveLock",
                granted=True,
            )
        ],
    )

    assert task_projection.preview is not None
    assert task_projection.preview.truncated
    assert task_projection.truncated
    assert len(task_projection.preview.content) <= 320
    assert lock_projection.preview is not None
    assert lock_projection.preview.truncated
    assert lock_projection.truncated
    assert len(lock_projection.preview.content) <= 320


def test_error_and_diagnostic_terminal_projections_are_safe() -> None:
    manager = _manager()
    call = ToolCall(
        id="call-1",
        name="database.run",
        arguments={"sql": "SELECT * FROM users"},
    )
    error = ToolCallError(
        id="error-1",
        name=call.name,
        arguments=call.arguments,
        call=call,
        error=RuntimeError(
            "failed for postgresql://db_user:super-secret-password@host/app"
        ),
        message="connection failed",
    )
    diagnostic = ToolCallDiagnostic(
        id="diag-1",
        requested_name=call.name,
        code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        stage=ToolCallDiagnosticStage.VALIDATE,
        message="SQL argument is invalid.",
    )

    error_projection = _project_outcome(manager, call, error)
    diagnostic_projection = _project_outcome(manager, call, diagnostic)

    assert error_projection.status == "error"
    assert error_projection.severity == "error"
    assert diagnostic_projection.action == "skip"
    assert diagnostic_projection.outcome == "tool_call.arguments_invalid"
    combined = _projection_text(error_projection) + _projection_text(
        diagnostic_projection
    )
    assert "super-secret-password" not in combined
    assert "db_user" not in combined


def _project_call(
    manager: ToolManager, name: str, arguments: dict[str, Any]
) -> ToolDisplayProjection:
    descriptor = manager.describe_tool(name)
    assert descriptor is not None
    projection = descriptor.project_display(
        ToolCall(id=f"{name}-1", name=name, arguments=arguments)
    )
    assert isinstance(projection, ToolDisplayProjection)
    return projection


def _project_result(
    manager: ToolManager,
    name: str,
    arguments: dict[str, Any],
    result: object,
) -> ToolDisplayProjection:
    call = ToolCall(id=f"{name}-1", name=name, arguments=arguments)
    outcome = ToolCallResult(
        id=f"{name}-result",
        name=name,
        arguments=arguments,
        call=call,
        result=cast(Any, result),
    )
    return _project_outcome(manager, call, outcome)


def _project_outcome(
    manager: ToolManager,
    call: ToolCall,
    outcome: object,
) -> ToolDisplayProjection:
    descriptor = manager.describe_tool(call.name)
    assert descriptor is not None
    projection = descriptor.project_display(call, cast(Any, outcome))
    assert isinstance(projection, ToolDisplayProjection)
    return projection


def _project_direct_call(
    name: str,
    arguments: dict[str, Any],
) -> ToolDisplayProjection:
    projection = database_display.project_database_tool_display(
        call=ToolCall(id=f"{name}-direct", name=name, arguments=arguments),
        settings=_settings(),
        dialect="postgresql",
    )
    assert isinstance(projection, ToolDisplayProjection)
    return projection


def _manager() -> ToolManager:
    settings = _settings()
    return ToolManager.create_instance(
        available_toolsets=[
            ToolSet(namespace="database", tools=_tools(settings))
        ],
        settings=ToolManagerSettings(),
    )


def _settings() -> DatabaseToolSettings:
    return DatabaseToolSettings(dsn=_DSN, read_only=True)


def _tools(settings: DatabaseToolSettings) -> list[DatabaseTool]:
    engine = cast(Any, _engine())
    return [
        DatabaseCountTool(engine, settings),
        DatabaseInspectTool(engine, settings),
        DatabaseKeysTool(engine, settings),
        DatabaseRelationshipsTool(engine, settings),
        DatabasePlanTool(engine, settings),
        DatabaseRunTool(engine, settings),
        DatabaseSampleTool(engine, settings),
        DatabaseSizeTool(engine, settings),
        DatabaseTablesTool(engine, settings),
        DatabaseTasksTool(engine, settings),
        DatabaseKillTool(engine, settings),
        DatabaseLocksTool(engine, settings),
    ]


def _engine() -> SimpleNamespace:
    return SimpleNamespace(
        sync_engine=SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    )


def _call_arguments() -> dict[str, dict[str, Any]]:
    return {
        "database.count": {"table_name": "public.users"},
        "database.inspect": {
            "schema": "public",
            "table_names": ["users", "books"],
        },
        "database.keys": {"schema": "public", "table_name": "users"},
        "database.relationships": {
            "schema": "public",
            "table_name": "books",
        },
        "database.plan": {"sql": "SELECT * FROM users"},
        "database.run": {"sql": "SELECT * FROM users"},
        "database.sample": {"table_name": "public.users", "count": 3},
        "database.size": {"table_name": "public.users"},
        "database.tables": {},
        "database.tasks": {"running_for": 10},
        "database.kill": {"task_id": "123"},
        "database.locks": {},
    }


def _detail_value(
    projection: ToolDisplayProjection, label: str
) -> object | None:
    detail = _detail(projection, label)
    return detail.value if detail else None


def _detail(
    projection: ToolDisplayProjection, label: str
) -> ToolDisplayDetail | None:
    for detail in projection.details:
        if detail.label == label:
            return detail
    return None


def _projection_text(projection: ToolDisplayProjection) -> str:
    return repr(projection.to_payload())
