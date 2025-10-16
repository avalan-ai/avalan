from ...compat import override
from .. import Tool

from abc import ABC
from asyncio import sleep
from dataclasses import dataclass, field
from re import compile as regex_compile
from typing import Any, Literal, final

from sqlalchemy import event
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlglot import exp, parse, parse_one


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ForeignKey:
    field: str
    ref_table: str
    ref_field: str


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class Table:
    name: str
    columns: dict[str, str]
    foreign_keys: list[ForeignKey]


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class TableKey:
    type: Literal["primary", "unique"]
    name: str | None
    columns: tuple[str, ...]


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class DatabaseTask:
    id: str
    user: str | None
    state: str | None
    query: str | None
    duration: int | None


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class DatabaseLock:
    pid: str | None
    user: str | None
    lock_type: str | None
    lock_target: str | None
    mode: str | None
    granted: bool | None
    blocking: tuple[str, ...] = ()
    state: str | None = None
    query: str | None = None


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class DatabaseToolSettings:
    dsn: str
    delay_secs: float | None = None
    identifier_case: Literal["preserve", "lower", "upper"] = "preserve"
    read_only: bool = True
    allowed_commands: list[str] | None = field(
        default_factory=lambda: ["select"],
    )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class QueryPlan:
    dialect: str
    steps: list[dict[str, Any]]


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class TableRelationship:
    direction: Literal["incoming", "outgoing"]
    local_columns: tuple[str, ...]
    related_table: str
    related_columns: tuple[str, ...]
    constraint_name: str | None


class IdentifierCaseNormalizer:
    __slots__ = ("_mode", "_token_pattern")

    def __init__(self, mode: Literal["preserve", "lower", "upper"]) -> None:
        self._mode = mode
        self._token_pattern = regex_compile(
            r"[0-9A-Za-z_]+(?:\.[0-9A-Za-z_]+)?"
        )

    def normalize(self, identifier: str) -> str:
        if self._mode == "lower":
            return identifier.lower()
        if self._mode == "upper":
            return identifier.upper()
        return identifier

    def normalize_token(self, identifier: str) -> str:
        if "." not in identifier:
            return self.normalize(identifier)
        schema, table = identifier.split(".", 1)
        table_normalized = self.normalize(table)
        return f"{schema}.{table_normalized}"

    def iter_tokens(self, sql: str) -> list[tuple[str, int, int]]:
        return [
            (match.group(0), match.start(), match.end())
            for match in self._token_pattern.finditer(sql)
        ]


class DatabaseTool(Tool, ABC):
    _engine: AsyncEngine
    _settings: DatabaseToolSettings
    _normalizer: IdentifierCaseNormalizer | None
    _table_cache: dict[str | None, dict[str, str]]

    def __init__(
        self,
        engine: AsyncEngine,
        settings: DatabaseToolSettings,
        *,
        normalizer: IdentifierCaseNormalizer | None = None,
        table_cache: dict[str | None, dict[str, str]] | None = None,
    ) -> None:
        self._engine = engine
        self._settings = settings
        if settings.identifier_case == "preserve":
            self._normalizer = None
        else:
            self._normalizer = normalizer or IdentifierCaseNormalizer(
                settings.identifier_case
            )
        self._table_cache = table_cache if table_cache is not None else {}
        super().__init__()

    async def _sleep_if_configured(self) -> None:
        delay = self._settings.delay_secs
        if delay:
            await sleep(delay)

    @staticmethod
    def _inspect_connection(connection: Connection) -> Inspector:
        return inspect(connection)

    @staticmethod
    def _create_engine(dsn: str, **kwargs: Any) -> AsyncEngine:
        return create_async_engine(dsn, **kwargs)

    def _register_table_names(
        self, schema: str | None, table_names: list[str]
    ) -> None:
        if self._normalizer is None:
            return
        cache = self._table_cache.setdefault(schema, {})
        for name in table_names:
            cache[self._normalizer.normalize(name)] = name

    def _denormalize_table_name(
        self,
        connection: Connection,
        schema: str | None,
        table_name: str,
    ) -> str:
        if self._normalizer is None:
            return table_name

        cache = self._table_cache.get(schema)
        normalized = self._normalizer.normalize(table_name)

        if cache is None or normalized not in cache:
            inspector = inspect(connection)
            actual_tables = inspector.get_table_names(schema=schema)
            cache = {
                self._normalizer.normalize(name): name
                for name in actual_tables
            }
            self._table_cache[schema] = cache

        return cache.get(normalized, table_name)

    def _normalize_table_for_output(self, table_name: str) -> str:
        if self._normalizer is None:
            return table_name
        return self._normalizer.normalize(table_name)

    def _apply_identifier_case(self, connection: Connection, sql: str) -> str:
        if self._normalizer is None:
            return sql

        inspector = inspect(connection)
        _, schemas = self._schemas(connection, inspector)

        for schema in schemas:
            if schema in self._table_cache:
                continue
            actual_tables = inspector.get_table_names(schema=schema)
            self._register_table_names(schema, actual_tables)

        replacements: dict[str, str] = {}
        for schema, table_map in self._table_cache.items():
            for normalized, actual in table_map.items():
                replacements[normalized] = actual
                if schema is not None:
                    replacements[f"{schema}.{normalized}"] = (
                        f"{schema}.{actual}"
                    )

        if not replacements:
            return sql

        dialect = self._sqlglot_dialect_name(self._engine)
        try:
            tree = parse_one(sql, read=dialect) if dialect else parse_one(sql)
        except Exception:
            return self._rewrite_sql_with_tokens(sql, replacements)

        def normalize_table(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Table):
                ident = node.this
                if isinstance(ident, exp.Identifier) and not ident.quoted:
                    name = ident.this
                    schema_ident = node.args.get("db")
                    schema = (
                        schema_ident.this
                        if isinstance(schema_ident, exp.Identifier)
                        else None
                    )
                    key = self._normalizer.normalize(name)
                    lookup = f"{schema}.{key}" if schema else key
                    actual = replacements.get(lookup) or replacements.get(key)
                    if actual:
                        if schema and "." in actual:
                            _, actual_name = actual.split(".", 1)
                        else:
                            actual_name = actual
                        node.set(
                            "this",
                            exp.Identifier(this=actual_name, quoted=False),
                        )
            return node

        tree = tree.transform(normalize_table)
        return tree.sql(dialect=dialect) if dialect else tree.sql()

    def _rewrite_sql_with_tokens(
        self, sql: str, replacements: dict[str, str]
    ) -> str:
        if self._normalizer is None:
            return sql

        tokens = self._normalizer.iter_tokens(sql)
        if not tokens:
            return sql

        rewritten: list[str] = []
        cursor = 0

        for token, start, end in tokens:
            if start < cursor:
                continue
            rewritten.append(sql[cursor:start])

            if self._token_is_quoted(sql, start, end):
                rewritten.append(token)
            else:
                lookup = self._normalizer.normalize_token(token)
                replacement = replacements.get(lookup)
                rewritten.append(replacement or token)

            cursor = end

        rewritten.append(sql[cursor:])
        return "".join(rewritten)

    @staticmethod
    def _token_is_quoted(sql: str, start: int, end: int) -> bool:
        if start > 0 and sql[start - 1] in {'"', "'", "`"}:
            return True
        if end < len(sql) and sql[end] in {'"', "'", "`"}:
            return True
        return False

    def _normalize_sql(self, sql: str) -> str:
        dialect = self._sqlglot_dialect_name(self._engine)
        try:
            trees = parse(sql, read=dialect) if dialect else parse(sql)
            count = len([t for t in trees if t is not None])
            if count > 1:
                raise PermissionError(
                    "Multiple SQL statements are not permitted in a single"
                    " execution.",
                )
        except PermissionError:
            raise
        except Exception:
            pass
        return sql.strip()

    def _prepare_sql_for_execution(self, sql: str) -> str:
        normalized_sql = self._normalize_sql(sql)
        if self._settings.allowed_commands is None:
            return normalized_sql
        dialect = self._sqlglot_dialect_name(self._engine)
        self._ensure_sql_command_allowed(
            normalized_sql, self._settings.allowed_commands, dialect
        )
        return normalized_sql

    @staticmethod
    def _schemas(
        connection: Connection, inspector: Inspector
    ) -> tuple[str | None, list[str | None]]:
        default_schema = inspector.default_schema_name
        dialect = connection.dialect.name

        if dialect == "postgresql":
            sys = {"information_schema", "pg_catalog"}
            schemas = [
                s
                for s in inspector.get_schema_names()
                if s not in sys and not (s or "").startswith("pg_")
            ]
            if default_schema and default_schema not in schemas:
                schemas.append(default_schema)
            return default_schema, schemas

        all_schemas = inspector.get_schema_names() or (
            [default_schema] if default_schema is not None else [None]
        )

        sys_filters = {
            "mysql": {
                "information_schema",
                "performance_schema",
                "mysql",
                "sys",
            },
            "mariadb": {
                "information_schema",
                "performance_schema",
                "mysql",
                "sys",
            },
            "mssql": {"INFORMATION_SCHEMA", "sys"},
            "oracle": {"SYS", "SYSTEM"},
            "sqlite": set(),
        }
        sys = sys_filters.get(dialect, set())
        schemas = [s for s in all_schemas if s not in sys]

        if not schemas:
            schemas = (
                [default_schema] if default_schema is not None else [None]
            )

        seen: set[str | None] = set()
        uniq: list[str | None] = []
        for s in schemas:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        if default_schema not in seen:
            uniq.append(default_schema)

        return default_schema, uniq

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: BaseException | None,
    ) -> bool:
        return await super().__aexit__(exc_type, exc_value, traceback)

    @staticmethod
    def _sqlglot_dialect_name(engine: AsyncEngine) -> str | None:
        sync_engine = getattr(engine, "sync_engine", None)
        if sync_engine is None:
            return None
        name = getattr(sync_engine.dialect, "name", None)
        if name == "postgresql":
            return "postgres"
        if name == "mariadb":
            return "mysql"
        return name

    @staticmethod
    def _ensure_sql_command_allowed(
        statement: str, allowed: list[str], dialect_name: str | None = None
    ) -> None:
        normalized_allowed = {command.lower() for command in allowed}
        if not normalized_allowed:
            raise PermissionError(
                "No SQL commands are permitted by the current configuration.",
            )

        try:
            expr = (
                parse_one(statement, read=dialect_name)
                if dialect_name
                else parse_one(statement)
            )
        except Exception:
            raise PermissionError(
                "SQL could not be parsed to enforce allowed commands.",
            )

        if expr is None:
            return

        key = (expr.key or "").lower()
        if key == "with":
            inner = getattr(expr, "this", None)
            if inner is not None and getattr(inner, "key", None):
                key = inner.key.lower()

        if not key:
            return
        if key not in normalized_allowed:
            allowed_display = ", ".join(sorted(normalized_allowed))
            raise PermissionError(
                f"SQL command '{key.upper()}' is not permitted."
                f" Allowed commands: {allowed_display}.",
            )

    @staticmethod
    def _configure_read_only_engine(
        engine: AsyncEngine, read_only: bool
    ) -> None:
        if not read_only:
            return

        sync_engine = getattr(engine, "sync_engine", None)
        if sync_engine is None:
            return

        sg_name = DatabaseTool._sqlglot_dialect_name(engine)
        statements_by_sqlglot: dict[str, tuple[str, ...]] = {
            "sqlite": ("PRAGMA query_only = ON",),
            "postgres": (
                "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
            ),
            "mysql": ("SET SESSION TRANSACTION READ ONLY",),
            "oracle": ("ALTER SESSION SET READ ONLY = TRUE",),
        }
        statements = statements_by_sqlglot.get(sg_name or "")

        if statements is None:
            dialect_name = getattr(sync_engine.dialect, "name", "")
            statements_by_sa: dict[str, tuple[str, ...]] = {
                "sqlite": ("PRAGMA query_only = ON",),
                "postgresql": (
                    "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
                ),
                "mysql": ("SET SESSION TRANSACTION READ ONLY",),
                "mariadb": ("SET SESSION TRANSACTION READ ONLY",),
                "oracle": ("ALTER SESSION SET READ ONLY = TRUE",),
            }
            statements = statements_by_sa.get(dialect_name)

        if statements is None:
            return

        @event.listens_for(sync_engine, "connect")
        def _set_read_only(dbapi_connection, _connection_record):  # type: ignore[arg-type]
            cursor = dbapi_connection.cursor()
            try:
                for statement in statements:
                    cursor.execute(statement)
            finally:
                cursor.close()


# ruff: noqa: E402

from .count import DatabaseCountTool  # noqa: F401
from .inspect import DatabaseInspectTool  # noqa: F401
from .keys import DatabaseKeysTool  # noqa: F401
from .kill import DatabaseKillTool  # noqa: F401
from .locks import DatabaseLocksTool  # noqa: F401
from .plan import DatabasePlanTool  # noqa: F401
from .relationships import DatabaseRelationshipsTool  # noqa: F401
from .run import DatabaseRunTool  # noqa: F401
from .tables import DatabaseTablesTool  # noqa: F401
from .tasks import DatabaseTasksTool  # noqa: F401
from .toolset import DatabaseToolSet  # noqa: F401

# Preserve the SQLAlchemy inspect callable for tests that patch this attribute.
inspect = sqlalchemy_inspect
