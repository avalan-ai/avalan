from unittest import TestCase
from unittest.mock import patch

from avalan.tool.database import DatabaseTool, DatabaseToolSettings
from avalan.tool.database.settings import resolve_database_dsn


class DatabaseToolSettingsTestCase(TestCase):
    def test_resolve_database_dsn_reads_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {"AVALAN_DATABASE_DSN": "sqlite:///env.sqlite"},
            clear=True,
        ):
            self.assertEqual(
                resolve_database_dsn("env:AVALAN_DATABASE_DSN"),
                "sqlite:///env.sqlite",
            )

    def test_resolve_database_dsn_rejects_missing_or_empty_env(self) -> None:
        cases = (
            ("AVALAN_MISSING_DATABASE_DSN", {}),
            ("AVALAN_EMPTY_DATABASE_DSN", {"AVALAN_EMPTY_DATABASE_DSN": ""}),
        )
        for env_name, env in cases:
            with self.subTest(env_name=env_name):
                with (
                    patch.dict("os.environ", env, clear=True),
                    self.assertRaises(AssertionError),
                ):
                    resolve_database_dsn(f"env:{env_name}")

    def test_settings_resolve_env_dsn(self) -> None:
        with patch.dict(
            "os.environ",
            {"AVALAN_DATABASE_DSN": "sqlite:///env.sqlite"},
            clear=True,
        ):
            settings = DatabaseToolSettings(dsn="env:AVALAN_DATABASE_DSN")

        self.assertEqual(settings.dsn, "sqlite:///env.sqlite")

    def test_settings_reject_invalid_env_name(self) -> None:
        with self.assertRaises(AssertionError):
            DatabaseToolSettings(dsn="env:INVALID-NAME")

    def test_settings_normalize_bare_string_allowed_commands(self) -> None:
        settings = DatabaseToolSettings(
            dsn="sqlite://",
            allowed_commands="select",  # type: ignore[arg-type]
        )

        self.assertEqual(settings.allowed_commands, ["select"])

    def test_settings_allow_disabled_command_filter(self) -> None:
        settings = DatabaseToolSettings(
            dsn="sqlite://",
            allowed_commands=None,
        )

        self.assertIsNone(settings.allowed_commands)

    def test_settings_reject_empty_allowed_commands(self) -> None:
        with self.assertRaises(AssertionError):
            DatabaseToolSettings(dsn="sqlite://", allowed_commands=[])

    def test_create_engine_resolves_env_dsn_at_runtime(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"AVALAN_DATABASE_DSN": "sqlite:///env.sqlite"},
                clear=True,
            ),
            patch("avalan.tool.database.create_async_engine") as create_engine,
        ):
            engine = DatabaseTool._create_engine(
                "env:AVALAN_DATABASE_DSN",
                pool_pre_ping=True,
            )

        self.assertIs(engine, create_engine.return_value)
        create_engine.assert_called_once_with(
            "sqlite:///env.sqlite",
            pool_pre_ping=True,
        )
