from argparse import ArgumentParser
from importlib import import_module
from importlib.util import find_spec
from os import environ
from re import fullmatch
from subprocess import run
from sys import executable, stderr
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--admin-dsn",
        default=environ.get("AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN"),
    )
    parser.add_argument(
        "--database-prefix",
        default=environ.get(
            "AVALAN_TASK_TEST_POSTGRESQL_DATABASE_PREFIX",
            "avalan_task_test",
        ),
    )
    args, pytest_args = parser.parse_known_args()
    admin_dsn = args.admin_dsn
    if not admin_dsn:
        raise SystemExit("AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN is not set")
    _require_runtime_modules()
    database_name = _database_name(args.database_prefix)
    test_dsn = _database_dsn(admin_dsn, database_name)
    pytest_args = tuple(pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    exit_code = 1
    try:
        _create_database(admin_dsn, database_name)
        child_env = environ.copy()
        child_env["AVALAN_TASK_TEST_POSTGRESQL_DSN"] = test_dsn
        completed = run(
            (executable, "-m", "pytest", *pytest_args),
            check=False,
            env=child_env,
        )
        exit_code = completed.returncode
    finally:
        try:
            _drop_database(admin_dsn, database_name)
        except Exception as exc:
            print(
                "Unable to drop PostgreSQL test database "
                f"{database_name}: {exc.__class__.__name__}",
                file=stderr,
            )
            if exit_code == 0:
                exit_code = 1
    return exit_code


def _database_name(prefix: str) -> str:
    if not fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,40}", prefix):
        raise SystemExit("database prefix must be a PostgreSQL identifier")
    name = f"{prefix}_{uuid4().hex}"
    if len(name) > 63:
        raise SystemExit("database prefix is too long")
    return name


def _database_dsn(admin_dsn: str, database_name: str) -> str:
    parts = urlsplit(admin_dsn)
    if not parts.scheme or not parts.netloc:
        raise SystemExit("admin DSN must be a URL-style PostgreSQL DSN")
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        "/" + quote(database_name, safe=""),
        parts.query,
        parts.fragment,
    ))


def _create_database(admin_dsn: str, database_name: str) -> None:
    psycopg, sql = _psycopg_modules()
    try:
        with psycopg.connect(
            _psycopg_dsn(admin_dsn),
            autocommit=True,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(database_name)
                    )
                )
    except Exception as exc:
        raise SystemExit(
            "Unable to create PostgreSQL test database "
            f"{database_name}: {exc.__class__.__name__}"
        ) from None


def _drop_database(admin_dsn: str, database_name: str) -> None:
    psycopg, sql = _psycopg_modules()
    with psycopg.connect(
        _psycopg_dsn(admin_dsn), autocommit=True
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_terminate_backend(pid) "
                "FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            cursor.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    sql.Identifier(database_name)
                )
            )


def _psycopg_modules() -> tuple[object, object]:
    if find_spec("psycopg") is None:
        raise SystemExit(
            "psycopg is required for PostgreSQL test database setup"
        )
    return import_module("psycopg"), import_module("psycopg.sql")


def _require_runtime_modules() -> None:
    missing = [
        module
        for module in ("alembic", "psycopg", "psycopg_pool", "sqlalchemy")
        if find_spec(module) is None
    ]
    if missing:
        raise SystemExit(
            "PostgreSQL task e2e tests require: " + ", ".join(missing)
        )


def _psycopg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return "postgresql://" + dsn.removeprefix("postgresql+psycopg://")
    return dsn


if __name__ == "__main__":
    raise SystemExit(main())
