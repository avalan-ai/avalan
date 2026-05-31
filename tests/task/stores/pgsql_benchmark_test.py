from os import environ
from unittest import TestCase, main

from pgsql_harness import isolated_task_pgsql_schema
from pytest import importorskip

from avalan.task.stores import PgsqlTaskMigrationSettings, task_pgsql_upgrade
from avalan.task.stores.pgsql_benchmark import (
    TaskPgsqlBenchmarkCase,
    TaskPgsqlBenchmarkOperation,
    TaskPgsqlBenchmarkSettings,
    task_pgsql_benchmark_cases,
    task_pgsql_benchmark_metadata,
    task_pgsql_explain_statement,
    task_pgsql_plan_issues,
)


class PgsqlBenchmarkCaseTest(TestCase):
    def test_cases_cover_operational_paths_with_safe_parameters(self) -> None:
        cases = task_pgsql_benchmark_cases()

        self.assertEqual(
            {case.operation for case in cases},
            set(TaskPgsqlBenchmarkOperation),
        )
        for benchmark_case in cases:
            with self.subTest(operation=benchmark_case.operation.value):
                self.assertNotIn(";", benchmark_case.statement)
                self.assertTrue(benchmark_case.expected_indexes)
                self.assertIn(
                    "EXPLAIN (FORMAT TEXT)",
                    task_pgsql_explain_statement(benchmark_case),
                )
                self.assertNotIn(
                    "benchmark-claim-token",
                    benchmark_case.statement,
                )
                self.assertTrue(benchmark_case.parameter_map)

        claim_case = self._case(TaskPgsqlBenchmarkOperation.CLAIM)
        self.assertIn("FOR UPDATE SKIP LOCKED", claim_case.statement)

    def test_benchmark_metadata_records_profile_and_thresholds(self) -> None:
        settings = TaskPgsqlBenchmarkSettings(
            worker_count=4,
            run_count=100,
            queue_count=2,
            pool_size=8,
            postgresql_version="16.3",
            thresholds={TaskPgsqlBenchmarkOperation.CLAIM: 25.0},
        )

        metadata = task_pgsql_benchmark_metadata(settings)

        self.assertEqual(metadata["worker_count"], 4)
        self.assertEqual(metadata["run_count"], 100)
        self.assertEqual(metadata["queue_count"], 2)
        self.assertEqual(metadata["pool_size"], 8)
        self.assertEqual(metadata["postgresql_version"], "16.3")
        self.assertEqual(metadata["thresholds"], {"claim": 25.0})

    def test_explain_statement_can_enable_analyze_explicitly(self) -> None:
        benchmark_case = self._case(TaskPgsqlBenchmarkOperation.RUN_CREATION)

        self.assertTrue(
            task_pgsql_explain_statement(
                benchmark_case,
                analyze=True,
            ).startswith("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)")
        )

    def test_plan_issues_detect_missing_indexes_and_scan_regressions(
        self,
    ) -> None:
        benchmark_case = self._case(TaskPgsqlBenchmarkOperation.CLAIM)

        self.assertEqual(
            task_pgsql_plan_issues(
                benchmark_case,
                (
                    (
                        "Index Scan using ix_task_queue_items_claimable "
                        "on task_queue_items"
                    ),
                ),
            ),
            (),
        )
        self.assertEqual(
            task_pgsql_plan_issues(
                benchmark_case,
                ("Seq Scan on task_queue_items",),
            ),
            (
                "plan.missing_index.ix_task_queue_items_claimable",
                "plan.unbounded_scan",
            ),
        )

    def test_plan_issues_detect_claims_without_skip_locked(self) -> None:
        benchmark_case = TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.CLAIM,
            statement='SELECT "queue_item_id" FROM "task_queue_items"',
            expected_indexes=("ix_task_queue_items_claimable",),
        )

        self.assertIn(
            "plan.duplicate_claim_risk",
            task_pgsql_plan_issues(benchmark_case, ()),
        )

    def test_invalid_benchmark_inputs_fail_fast(self) -> None:
        with self.assertRaises(AssertionError):
            TaskPgsqlBenchmarkSettings(
                worker_count=0,
                run_count=1,
                queue_count=1,
                pool_size=1,
                postgresql_version="16",
            )
        with self.assertRaises(AssertionError):
            TaskPgsqlBenchmarkSettings(
                worker_count=1,
                run_count=1,
                queue_count=1,
                pool_size=1,
                postgresql_version="",
            )
        with self.assertRaises(AssertionError):
            TaskPgsqlBenchmarkCase(
                operation=TaskPgsqlBenchmarkOperation.CLAIM,
                statement='SELECT "queue_item_id";',
            )
        with self.assertRaises(AssertionError):
            TaskPgsqlBenchmarkCase(
                operation=TaskPgsqlBenchmarkOperation.CLAIM,
                statement='SELECT "queue_item_id"',
                parameters=(("bad-name", "value"),),
            )

    def _case(
        self,
        operation: TaskPgsqlBenchmarkOperation,
    ) -> TaskPgsqlBenchmarkCase:
        for benchmark_case in task_pgsql_benchmark_cases():
            if benchmark_case.operation == operation:
                return benchmark_case
        raise AssertionError(f"missing benchmark case: {operation.value}")


class PgsqlBenchmarkIntegrationTest(TestCase):
    def test_explain_real_postgresql_when_configured(self) -> None:
        dsn = environ.get("AVALAN_TASK_BENCHMARK_POSTGRESQL_DSN")
        if not dsn:
            self.skipTest("AVALAN_TASK_BENCHMARK_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        sqlalchemy = importorskip("sqlalchemy")
        schema = isolated_task_pgsql_schema("avalan_task_benchmark")
        settings = PgsqlTaskMigrationSettings(url=dsn, schema=schema)

        task_pgsql_upgrade(settings)
        engine = sqlalchemy.create_engine(dsn)
        try:
            with engine.begin() as connection:
                connection.execute(
                    sqlalchemy.text(f'SET search_path TO "{schema}"')
                )
                version = str(
                    connection.execute(
                        sqlalchemy.text("SHOW server_version")
                    ).scalar_one()
                )
                metadata = task_pgsql_benchmark_metadata(
                    TaskPgsqlBenchmarkSettings(
                        worker_count=1,
                        run_count=1,
                        queue_count=1,
                        pool_size=1,
                        postgresql_version=version,
                    )
                )

                self.assertIn("postgresql_version", metadata)
                for benchmark_case in task_pgsql_benchmark_cases():
                    with self.subTest(
                        operation=benchmark_case.operation.value
                    ):
                        rows = connection.execute(
                            sqlalchemy.text(
                                task_pgsql_explain_statement(benchmark_case)
                            ),
                            benchmark_case.parameter_map,
                        ).all()
                        self.assertTrue(rows)
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
