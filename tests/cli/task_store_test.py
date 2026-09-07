from argparse import ArgumentParser, Namespace
from unittest import TestCase
from unittest.mock import patch

from avalan.cli.task_store import (
    TaskStoreConfiguration,
    TaskStoreConfigurationError,
    add_task_store_arguments,
    task_store_configuration,
)


class TaskStoreConfigurationTest(TestCase):
    def test_shared_options_and_environment_have_one_contract(self) -> None:
        parser = ArgumentParser()
        add_task_store_arguments(parser)
        with patch.dict(
            "os.environ",
            {
                "AVALAN_TASK_STORE_DSN": "postgresql://host/db",
                "AVALAN_TASK_STORE_SCHEMA": "tenant",
                "AVALAN_TASK_PGSQL_DSN": "ignored",
            },
            clear=True,
        ):
            self.assertEqual(
                task_store_configuration(parser.parse_args([])).require_dsn(),
                "postgresql://host/db",
            )
            explicit = task_store_configuration(
                parser.parse_args(
                    [
                        "--store-dsn",
                        "postgresql://other/db",
                        "--store-schema",
                        "other",
                    ]
                )
            )
            self.assertEqual(explicit.schema, "other")
            self.assertNotIn("postgresql", repr(explicit))
            self.assertEqual(explicit.dsn, "postgresql://other/db")
        with patch.dict(
            "os.environ", {"AVALAN_TASK_PGSQL_DSN": "ignored"}, clear=True
        ):
            self.assertIsNone(
                task_store_configuration(
                    Namespace(dsn="ignored", schema="ignored")
                ).dsn
            )
            with self.assertRaises(TaskStoreConfigurationError):
                task_store_configuration(Namespace()).require_dsn()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--dsn", "obsolete"])

    def test_invalid_fields_are_safe(self) -> None:
        for value in (None, " "):
            with patch.dict("os.environ", {}, clear=True):
                self.assertIsNone(
                    task_store_configuration(Namespace(store_dsn=value)).dsn
                )
        for args in (
            Namespace(store_dsn=3),
            Namespace(store_schema="bad;schema"),
        ):
            with self.assertRaises(TaskStoreConfigurationError):
                task_store_configuration(args)
        with self.assertRaises(TaskStoreConfigurationError):
            TaskStoreConfiguration(dsn="")
