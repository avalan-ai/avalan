from os import environ
from uuid import uuid4


class FakeAlembicConfig:
    def __init__(self) -> None:
        self.options: dict[str, str] = {}
        self.attributes: dict[str, object] = {}

    def set_main_option(self, name: str, value: str) -> None:
        self.options[name] = value


class FakeAlembicConfigModule:
    def __init__(self) -> None:
        self.configs: list[FakeAlembicConfig] = []

    def Config(self) -> FakeAlembicConfig:
        config = FakeAlembicConfig()
        self.configs.append(config)
        return config


class FakeAlembicCommandModule:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = (
            []
        )

    def upgrade(self, config: object, revision: str) -> None:
        self.calls.append(("upgrade", (config, revision), {}))

    def current(self, config: object, **kwargs: object) -> None:
        self.calls.append(("current", (config,), kwargs))

    def stamp(self, config: object, revision: str) -> None:
        self.calls.append(("stamp", (config, revision), {}))


class FakeAlembicModules:
    def __init__(self) -> None:
        self.config = FakeAlembicConfigModule()
        self.command = FakeAlembicCommandModule()

    def module_finder(self, module: str) -> object | None:
        if module in {"alembic", "sqlalchemy"}:
            return object()
        return None

    def module_importer(self, module: str) -> object:
        if module == "alembic.config":
            return self.config
        if module == "alembic.command":
            return self.command
        raise AssertionError(f"unexpected module import: {module}")


class FakeAlembicEnvironmentConfig:
    config_ini_section = "alembic"

    def __init__(
        self,
        *,
        options: dict[str, str] | None = None,
        attributes: dict[str, object] | None = None,
    ) -> None:
        self.options = options or {}
        self.attributes = attributes or {}

    def get_main_option(self, name: str) -> str | None:
        return self.options.get(name)

    def get_section(
        self,
        name: str,
        default: dict[str, object],
    ) -> dict[str, object]:
        return {"sqlalchemy.url": "postgresql://localhost/avalan"}


class FakeTransaction:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeAlembicEnvironmentContext:
    def __init__(
        self,
        *,
        offline: bool,
        config: FakeAlembicEnvironmentConfig,
    ) -> None:
        self.config = config
        self.offline = offline
        self.configure_kwargs: dict[str, object] | None = None
        self.ran_migrations = False

    def is_offline_mode(self) -> bool:
        return self.offline

    def configure(self, **kwargs: object) -> None:
        self.configure_kwargs = kwargs

    def begin_transaction(self) -> FakeTransaction:
        return FakeTransaction()

    def run_migrations(self) -> None:
        self.ran_migrations = True


class FakeConnectionContext:
    def __init__(self, connection: "FakeSqlalchemyConnection") -> None:
        self.connection = connection

    def __enter__(self) -> "FakeSqlalchemyConnection":
        return self.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeSqlalchemyConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[object, object | None]] = []

    def begin(self) -> FakeTransaction:
        return FakeTransaction()

    def execute(
        self,
        statement: object,
        parameters: object | None = None,
    ) -> None:
        self.executed.append((statement, parameters))


class FakeSqlalchemyConnectable:
    def __init__(self, connection: FakeSqlalchemyConnection) -> None:
        self.connection = connection

    def connect(self) -> FakeConnectionContext:
        return FakeConnectionContext(self.connection)


class FakeAlembicBind:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def exec_driver_sql(self, statement: str) -> None:
        self.statements.append(statement)


class FakeRevisionOp:
    def __init__(self) -> None:
        self.bind = FakeAlembicBind()

    def get_bind(self) -> FakeAlembicBind:
        return self.bind


def isolated_task_pgsql_schema(prefix: str = "avalan_task_test") -> str:
    return f"{prefix}_{uuid4().hex}"


def real_task_pgsql_dsn() -> str | None:
    return environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")


def task_pgsql_psycopg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return "postgresql://" + dsn.removeprefix("postgresql+psycopg://")
    return dsn


def unexpected_import(module: str) -> object:
    raise AssertionError(f"unexpected module import: {module}")
