from argparse import ArgumentParser
from importlib import import_module
from importlib.util import find_spec
from os import environ
from re import fullmatch
from secrets import token_urlsafe
from socket import AF_INET, SOCK_STREAM, socket
from subprocess import CalledProcessError, run
from sys import executable, stderr
from time import monotonic, sleep
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
    parser.add_argument(
        "--docker",
        action="store_true",
        default=_truthy(environ.get("AVALAN_TASK_TEST_POSTGRESQL_DOCKER")),
    )
    parser.add_argument(
        "--docker-image",
        default=environ.get(
            "AVALAN_TASK_TEST_POSTGRESQL_DOCKER_IMAGE",
            "postgres:16-alpine",
        ),
    )
    parser.add_argument(
        "--docker-timeout-seconds",
        default=float(
            environ.get(
                "AVALAN_TASK_TEST_POSTGRESQL_DOCKER_TIMEOUT_SECONDS",
                "60",
            )
        ),
        type=float,
    )
    args, pytest_args = parser.parse_known_args()
    pytest_args = _pytest_args(pytest_args)
    if args.docker:
        return _run_with_docker(
            args.database_prefix,
            pytest_args,
            image=args.docker_image,
            timeout_seconds=args.docker_timeout_seconds,
        )
    admin_dsn = args.admin_dsn
    if not admin_dsn:
        raise SystemExit("AVALAN_TASK_TEST_POSTGRESQL_ADMIN_DSN is not set")
    return _run_with_admin_dsn(
        admin_dsn,
        args.database_prefix,
        pytest_args,
    )


def _run_with_admin_dsn(
    admin_dsn: str,
    database_prefix: str,
    pytest_args: tuple[str, ...],
) -> int:
    _require_runtime_modules()
    database_name = _database_name(database_prefix)
    test_dsn = _database_dsn(admin_dsn, database_name)
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


def _run_with_docker(
    database_prefix: str,
    pytest_args: tuple[str, ...],
    *,
    image: str,
    timeout_seconds: float,
) -> int:
    _require_runtime_modules()
    container_name = _docker_container_name()
    password = token_urlsafe(24)
    port = _free_tcp_port()
    admin_dsn = _docker_admin_dsn(port, password)
    started = False
    try:
        _start_docker_postgres(
            image=image,
            name=container_name,
            password=password,
            port=port,
        )
        started = True
        _wait_for_database(admin_dsn, timeout_seconds)
        return _run_with_admin_dsn(admin_dsn, database_prefix, pytest_args)
    finally:
        if started:
            _stop_docker_container(container_name)


def _pytest_args(args: list[str]) -> tuple[str, ...]:
    values = tuple(args)
    if values and values[0] == "--":
        return values[1:]
    return values


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


def _docker_admin_dsn(port: int, password: str) -> str:
    return (
        "postgresql://postgres:"
        f"{quote(password, safe='')}@127.0.0.1:{port}/postgres"
    )


def _docker_container_name() -> str:
    return f"avalan-task-pgsql-test-{uuid4().hex}"


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


def _free_tcp_port() -> int:
    with socket(AF_INET, SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    assert isinstance(port, int)
    return port


def _psycopg_modules() -> tuple[object, object]:
    if find_spec("psycopg") is None:
        raise SystemExit(
            "psycopg is required for PostgreSQL test database setup"
        )
    return import_module("psycopg"), import_module("psycopg.sql")


def _run_docker(command: tuple[str, ...]) -> str:
    try:
        completed = run(
            command,
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Docker is required for PostgreSQL task e2e tests"
        ) from exc
    except CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = "Docker command failed"
        if detail:
            message += f": {detail}"
        raise SystemExit(message) from None
    return completed.stdout.strip()


def _start_docker_postgres(
    *,
    image: str,
    name: str,
    password: str,
    port: int,
) -> None:
    _run_docker((
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--env",
        "POSTGRES_USER=postgres",
        "--env",
        "POSTGRES_DB=postgres",
        "--env",
        f"POSTGRES_PASSWORD={password}",
        "--publish",
        f"127.0.0.1:{port}:5432",
        image,
    ))


def _stop_docker_container(name: str) -> None:
    try:
        _run_docker(("docker", "rm", "--force", "--volumes", name))
    except SystemExit as exc:
        print(
            f"Unable to stop PostgreSQL Docker container {name}: {exc}",
            file=stderr,
        )


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


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _wait_for_database(dsn: str, timeout_seconds: float) -> None:
    psycopg, _ = _psycopg_modules()
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    while monotonic() < deadline:
        try:
            with psycopg.connect(
                _psycopg_dsn(dsn),
                autocommit=True,
                connect_timeout=2,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return
        except Exception as exc:
            last_error = exc
            sleep(0.5)
    detail = ""
    if last_error is not None:
        detail = f": {last_error.__class__.__name__}"
    raise SystemExit(
        "PostgreSQL Docker container did not become ready" + detail
    )


def _psycopg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return "postgresql://" + dsn.removeprefix("postgresql+psycopg://")
    return dsn


if __name__ == "__main__":
    raise SystemExit(main())
