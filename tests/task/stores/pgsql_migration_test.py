from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from sys import modules
from types import SimpleNamespace
from typing import cast
from unittest import TestCase, main

from pgsql_harness import (
    FakeAlembicConfig,
    FakeAlembicEnvironmentConfig,
    FakeAlembicEnvironmentContext,
    FakeAlembicModules,
    FakeRevisionOp,
    FakeSqlalchemyConnectable,
    FakeSqlalchemyConnection,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    unexpected_import,
)
from pytest import importorskip

from avalan.task import TaskArtifactPurpose, TaskAttemptState, TaskRunState
from avalan.task.stores import (
    TASK_PGSQL_ALEMBIC_VERSION_TABLE,
    TASK_PGSQL_HEAD_REVISION,
    PgsqlTaskMigrationError,
    PgsqlTaskMigrationSettings,
    task_pgsql_alembic_config,
    task_pgsql_check,
    task_pgsql_claim_token_predicate,
    task_pgsql_current,
    task_pgsql_schema_statements,
    task_pgsql_script_location,
    task_pgsql_stamp,
    task_pgsql_state_predicate,
    task_pgsql_upgrade,
)


class PgsqlMigrationSchemaTest(TestCase):
    def test_task_schema_has_expected_tables_and_constraints(self) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for table_name in (
            "task_definitions",
            "task_runs",
            "task_run_transitions",
            "task_attempts",
            "task_attempt_transitions",
            "task_artifacts",
            "task_artifact_bytes",
            "task_idempotency_keys",
            "task_queue_items",
            "task_events",
            "task_usage_records",
            "task_run_rollups",
        ):
            self.assertIn(f'"{table_name}"', schema)

        self.assertNotIn("task_schema_migrations", schema)
        self.assertIn('"uq_task_definitions_identity"', schema)
        self.assertIn('"uq_task_attempts_run_order"', schema)
        self.assertIn('"uq_task_attempts_one_active_per_run"', schema)
        self.assertIn('"uq_task_idempotency_keys_identity"', schema)
        self.assertIn('"uq_task_events_run_sequence"', schema)
        self.assertIn('"uq_task_usage_records_run_sequence"', schema)
        self.assertIn('"ck_task_usage_records_sequence_positive"', schema)
        self.assertIn(
            "WHERE \"state\" NOT IN ('succeeded', 'failed', 'abandoned')",
            schema,
        )
        self.assertIn('"task_reject_terminal_run_state_change"', schema)
        self.assertIn('"tr_task_runs_terminal_state"', schema)

    def test_schema_covers_current_state_and_artifact_vocabularies(
        self,
    ) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for state in TaskRunState:
            self.assertIn(f"'{state.value}'", schema)
        for state in TaskAttemptState:
            self.assertIn(f"'{state.value}'", schema)
        for purpose in TaskArtifactPurpose:
            self.assertIn(f"'{purpose.value}'", schema)

    def test_schema_covers_idempotency_and_byte_storage_contracts(
        self,
    ) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for column_name in (
            "owner_scope_hash",
            "window_hash",
            "input_hash",
            "file_hash",
            "custom_hash",
            "ciphertext",
            "encryption_key_id",
            "retention_days",
            "retention_deadline_at",
        ):
            self.assertIn(f'"{column_name}"', schema)
        self.assertIn('"fk_task_idempotency_keys__task_runs"', schema)
        self.assertIn("'input_hash'", schema)
        self.assertIn("'input_and_files_hash'", schema)
        self.assertIn("'custom'", schema)
        self.assertIn('"ck_task_artifact_bytes_retention_positive"', schema)
        self.assertIn(
            '"ck_task_artifact_bytes_retention_deadline_after_created"',
            schema,
        )
        self.assertIn(
            '"ck_task_artifact_bytes_encryption_metadata_shape"',
            schema,
        )
        self.assertIn('"ix_task_artifact_bytes_retention_deadline"', schema)
        self.assertIn('"ix_task_artifact_bytes_active_artifact"', schema)

    def test_schema_constrains_jsonb_snapshot_shapes(self) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for constraint_name in (
            "ck_task_definitions_definition_shape",
            "ck_task_runs_request_shape",
            "ck_task_runs_claim_shape",
            "ck_task_runs_result_shape",
            "ck_task_attempts_context_shape",
            "ck_task_attempts_result_shape",
            "ck_task_artifacts_ref_shape",
            "ck_task_artifacts_retention_shape",
            "ck_task_idempotency_keys_owner_scope_shape",
            "ck_task_queue_items_metadata_shape",
            "ck_task_usage_records_metadata_shape",
            "ck_task_run_rollups_metadata_shape",
        ):
            self.assertIn(f'"{constraint_name}"', schema)

        self.assertNotIn("raw_payload", schema)

    def test_schema_orders_usage_records_by_typed_sequence(self) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        self.assertIn('"sequence" BIGINT NOT NULL', schema)
        self.assertIn('"ix_task_usage_records_by_run_sequence"', schema)
        self.assertIn(
            'ON "task_usage_records" ("run_id", "sequence")',
            schema,
        )

    def test_schema_keeps_all_usage_counters_typed(self) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for column_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
            "cache_creation_input_tokens",
            "reasoning_tokens",
        ):
            self.assertIn(f'"{column_name}"', schema)
        for constraint_name in (
            "ck_task_usage_records_cache_creation_tokens_non_negative",
            "ck_task_usage_records_reasoning_tokens_non_negative",
            "ck_task_run_rollups_cache_creation_tokens_non_negative",
            "ck_task_run_rollups_reasoning_tokens_non_negative",
        ):
            self.assertIn(f'"{constraint_name}"', schema)

    def test_schema_keeps_provider_usage_counters_independent(self) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        self.assertNotIn("cached_not_above_prompt", schema)
        self.assertNotIn('"cached_tokens" <= "prompt_tokens"', schema)

    def test_schema_keeps_queue_and_lease_hot_fields_typed(self) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for column_name in (
            "queue_name",
            "worker_id",
            "claim_token",
            "heartbeat_at",
            "lease_expires_at",
        ):
            self.assertIn(f'"{column_name}"', schema)
        for constraint_name in (
            "ck_task_queue_items_claimed_fields",
            "ck_task_queue_items_lease_expires_after_claim",
            "ck_task_queue_items_heartbeat_after_claim",
            "uq_task_queue_items_one_active_per_run",
        ):
            self.assertIn(f'"{constraint_name}"', schema)
        for index_name in (
            "ix_task_runs_by_queue_state_updated",
            "ix_task_queue_items_claimable",
            "ix_task_queue_items_lease_expiry",
            "ix_task_queue_items_retry_sweep",
        ):
            self.assertIn(f'"{index_name}"', schema)
        self.assertIn('"queue_name",\n        "state"', schema)
        self.assertIn('"queue_name",\n        "lease_expires_at"', schema)

    def test_state_and_claim_predicates_are_parameterized(self) -> None:
        state_sql, state_params = task_pgsql_state_predicate(
            "state",
            {TaskRunState.QUEUED, TaskRunState.CLAIMED},
            table_alias="r",
        )
        no_claim_sql, no_claim_params = task_pgsql_claim_token_predicate(
            "claim_token",
            None,
            table_alias="r",
        )
        claim_sql, claim_params = task_pgsql_claim_token_predicate(
            "claim_token",
            "claim-secret",
        )

        self.assertEqual(state_sql, '"r"."state" IN (%s, %s)')
        self.assertEqual(set(state_params), {"queued", "claimed"})
        self.assertEqual(no_claim_sql, '"r"."claim_token" IS NULL')
        self.assertEqual(no_claim_params, ())
        self.assertEqual(claim_sql, '"claim_token" = %s')
        self.assertEqual(claim_params, ("claim-secret",))
        self.assertNotIn("claim-secret", claim_sql)

    def test_state_and_claim_predicates_reject_unsafe_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            task_pgsql_state_predicate("state;drop", {TaskRunState.QUEUED})
        with self.assertRaises(AssertionError):
            task_pgsql_state_predicate("state", set())
        with self.assertRaises(AssertionError):
            task_pgsql_state_predicate(
                "state",
                cast(set[TaskRunState], {"queued"}),
            )
        with self.assertRaises(AssertionError):
            task_pgsql_claim_token_predicate("claim", "")
        with self.assertRaises(AssertionError):
            task_pgsql_claim_token_predicate(
                "claim",
                "token",
                table_alias="bad-alias",
            )


class PgsqlMigrationHelperTest(TestCase):
    def test_builds_alembic_config_with_schema_and_metadata(self) -> None:
        modules = FakeAlembicModules()
        settings = PgsqlTaskMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            schema="task_schema",
            module_finder=modules.module_finder,
            module_importer=modules.module_importer,
            attributes={"connection_name": "test"},
        )

        config = cast(
            FakeAlembicConfig,
            task_pgsql_alembic_config(settings),
        )

        self.assertEqual(
            config.options["script_location"],
            task_pgsql_script_location(),
        )
        self.assertEqual(
            config.options["version_table"],
            TASK_PGSQL_ALEMBIC_VERSION_TABLE,
        )
        self.assertEqual(config.options["task_schema"], "task_schema")
        self.assertEqual(
            config.options["version_table_schema"],
            "task_schema",
        )
        self.assertEqual(config.attributes["connection_name"], "test")

    def test_missing_dependencies_raise_stable_diagnostic(self) -> None:
        settings = PgsqlTaskMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=lambda module: None,
            module_importer=unexpected_import,
        )

        with self.assertRaisesRegex(
            PgsqlTaskMigrationError,
            "dependency.task_pgsql_migrations_missing",
        ):
            task_pgsql_alembic_config(settings)

    def test_helpers_dispatch_to_alembic_commands(self) -> None:
        modules = FakeAlembicModules()
        settings = PgsqlTaskMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=modules.module_finder,
            module_importer=modules.module_importer,
        )

        task_pgsql_upgrade(settings)
        task_pgsql_current(settings, verbose=True)
        task_pgsql_check(settings)
        task_pgsql_stamp(settings, revision=TASK_PGSQL_HEAD_REVISION)

        self.assertEqual(
            [
                (name, args[1:] if len(args) > 1 else (), kwargs)
                for name, args, kwargs in modules.command.calls
            ],
            [
                ("upgrade", ("head",), {}),
                ("current", (), {"verbose": True}),
                ("current", (), {"check_heads": True}),
                ("stamp", (TASK_PGSQL_HEAD_REVISION,), {}),
            ],
        )

    def test_invalid_helper_settings_fail_fast(self) -> None:
        modules = FakeAlembicModules()

        for factory in (
            lambda: PgsqlTaskMigrationSettings(url=""),
            lambda: PgsqlTaskMigrationSettings(
                url="postgresql+psycopg://localhost/avalan",
                schema="bad-name",
            ),
            lambda: PgsqlTaskMigrationSettings(
                url="postgresql+psycopg://localhost/avalan",
                version_table="1bad",
            ),
        ):
            with self.assertRaises(AssertionError):
                factory()

        settings = PgsqlTaskMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=modules.module_finder,
            module_importer=modules.module_importer,
        )
        with self.assertRaises(AssertionError):
            task_pgsql_upgrade(settings, revision="head;drop")

    def test_real_postgresql_migration_lifecycle_when_configured(
        self,
    ) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("sqlalchemy")
        schema = isolated_task_pgsql_schema()

        settings = PgsqlTaskMigrationSettings(
            url=dsn,
            schema=schema,
        )
        task_pgsql_upgrade(settings)
        task_pgsql_current(settings, verbose=True)
        task_pgsql_check(settings)
        task_pgsql_stamp(settings, revision=TASK_PGSQL_HEAD_REVISION)
        task_pgsql_check(settings)

    def test_real_postgresql_schema_isolation_when_configured(self) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("sqlalchemy")
        schemas = (
            isolated_task_pgsql_schema(),
            isolated_task_pgsql_schema(),
        )

        for schema in schemas:
            with self.subTest(schema=schema):
                settings = PgsqlTaskMigrationSettings(url=dsn, schema=schema)
                task_pgsql_upgrade(settings)
                task_pgsql_check(settings)

    def test_real_postgresql_concurrent_migrations_when_configured(
        self,
    ) -> None:
        dsn = real_task_pgsql_dsn()
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("sqlalchemy")
        settings = PgsqlTaskMigrationSettings(
            url=dsn,
            schema=isolated_task_pgsql_schema(),
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(task_pgsql_upgrade, settings) for _ in range(2)
            )
            for future in futures:
                future.result()

        task_pgsql_check(settings)


class PgsqlMigrationEnvironmentTest(TestCase):
    def test_offline_environment_uses_default_version_table(self) -> None:
        context = FakeAlembicEnvironmentContext(
            offline=True,
            config=FakeAlembicEnvironmentConfig(),
        )

        self._import_env(context=context)

        self.assertTrue(context.ran_migrations)
        assert context.configure_kwargs is not None
        self.assertEqual(
            context.configure_kwargs["version_table"],
            TASK_PGSQL_ALEMBIC_VERSION_TABLE,
        )
        self.assertEqual(
            context.configure_kwargs["version_table_schema"], None
        )

    def test_online_environment_prepares_schema_and_lock(self) -> None:
        connection = FakeSqlalchemyConnection()
        context = FakeAlembicEnvironmentContext(
            offline=False,
            config=FakeAlembicEnvironmentConfig(
                options={
                    "task_advisory_lock_id": "42",
                    "version_table": "custom_version_table",
                },
                attributes={"task_schema": "task_schema"},
            ),
        )

        self._import_env(context=context, connection=connection)

        self.assertTrue(context.ran_migrations)
        assert context.configure_kwargs is not None
        self.assertTrue(context.configure_kwargs["include_schemas"])
        self.assertEqual(
            context.configure_kwargs["version_table"],
            "custom_version_table",
        )
        self.assertEqual(
            connection.executed,
            [
                ('CREATE SCHEMA IF NOT EXISTS "task_schema"', None),
                ('SET search_path TO "task_schema"', None),
                (
                    "SELECT pg_advisory_xact_lock(:lock_id)",
                    {"lock_id": 42},
                ),
            ],
        )

    def _import_env(
        self,
        *,
        context: FakeAlembicEnvironmentContext,
        connection: FakeSqlalchemyConnection | None = None,
    ) -> None:
        module_name = "avalan.task.stores.pgsql_migrations.env"
        old_alembic = modules.get("alembic")
        old_sqlalchemy = modules.get("sqlalchemy")
        old_env = modules.pop(module_name, None)
        fake_connection = connection or FakeSqlalchemyConnection()
        modules["alembic"] = SimpleNamespace(context=context)
        modules["sqlalchemy"] = SimpleNamespace(
            engine_from_config=lambda *args, **kwargs: (
                FakeSqlalchemyConnectable(
                    fake_connection,
                )
            ),
            pool=SimpleNamespace(NullPool=object),
            text=lambda value: value,
        )
        try:
            import_module(module_name)
        finally:
            modules.pop(module_name, None)
            if old_env is not None:
                modules[module_name] = old_env
            if old_alembic is None:
                modules.pop("alembic", None)
            else:
                modules["alembic"] = old_alembic
            if old_sqlalchemy is None:
                modules.pop("sqlalchemy", None)
            else:
                modules["sqlalchemy"] = old_sqlalchemy


class PgsqlMigrationRevisionTest(TestCase):
    def test_revision_upgrade_executes_schema_statements(self) -> None:
        revision_module = import_module(
            "avalan.task.stores.pgsql_migrations.versions."
            "v20260530_0001_task_schema"
        )
        fake_op = FakeRevisionOp()
        old_alembic = modules.get("alembic")
        modules["alembic"] = SimpleNamespace(op=fake_op)
        try:
            revision_module.upgrade()
        finally:
            if old_alembic is None:
                modules.pop("alembic", None)
            else:
                modules["alembic"] = old_alembic

        self.assertEqual(
            fake_op.bind.statements,
            list(task_pgsql_schema_statements()),
        )

    def test_revision_downgrade_is_forward_only(self) -> None:
        revision_module = import_module(
            "avalan.task.stores.pgsql_migrations.versions."
            "v20260530_0001_task_schema"
        )

        with self.assertRaises(NotImplementedError):
            revision_module.downgrade()


if __name__ == "__main__":
    main()
