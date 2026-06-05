from os import environ
from unittest import TestCase, main

from pgsql_harness import isolated_task_pgsql_schema
from pytest import importorskip

from avalan.task.stores import PgsqlTaskMigrationSettings, task_pgsql_upgrade
from avalan.task.stores.pgsql_benchmark import (
    TaskPgsqlBenchmarkCase,
    TaskPgsqlBenchmarkOperation,
    TaskPgsqlBenchmarkSettings,
    TaskPgsqlEventVolumeProfile,
    TaskPgsqlQueueLoadProfile,
    task_pgsql_benchmark_cases,
    task_pgsql_benchmark_metadata,
    task_pgsql_event_volume_issues,
    task_pgsql_event_volume_metadata,
    task_pgsql_explain_statement,
    task_pgsql_plan_issues,
    task_pgsql_queue_load_issues,
    task_pgsql_queue_load_metadata,
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

    def test_queue_load_metadata_records_operational_profile(self) -> None:
        profile = TaskPgsqlQueueLoadProfile(
            worker_count=4,
            run_count=100,
            queue_count=2,
            lease_seconds=30,
            max_attempts=3,
            abandon_limit=25,
            pool_size=8,
            postgresql_version="16.3",
            retry_delay_seconds=5,
            min_claims_per_second=20.0,
        )

        metadata = task_pgsql_queue_load_metadata(
            profile,
            elapsed_seconds=4.0,
        )

        self.assertEqual(metadata["worker_count"], 4)
        self.assertEqual(metadata["run_count"], 100)
        self.assertEqual(metadata["queue_count"], 2)
        self.assertEqual(metadata["lease_seconds"], 30)
        self.assertEqual(metadata["max_attempts"], 3)
        self.assertEqual(metadata["abandon_limit"], 25)
        self.assertEqual(metadata["pool_size"], 8)
        self.assertEqual(metadata["postgresql_version"], "16.3")
        self.assertEqual(metadata["retry_delay_seconds"], 5)
        self.assertEqual(metadata["min_claims_per_second"], 20.0)
        self.assertEqual(metadata["claims_per_second"], 25.0)

    def test_queue_load_metadata_allows_unthresholded_profiles(self) -> None:
        profile = TaskPgsqlQueueLoadProfile(
            worker_count=1,
            run_count=1,
            queue_count=1,
            lease_seconds=30,
            max_attempts=1,
            abandon_limit=1,
            pool_size=1,
            postgresql_version="16.3",
        )

        metadata = task_pgsql_queue_load_metadata(profile)

        self.assertNotIn("min_claims_per_second", metadata)
        self.assertNotIn("claims_per_second", metadata)

    def test_event_volume_metadata_records_rollout_profile(self) -> None:
        profile = TaskPgsqlEventVolumeProfile(
            run_count=10_000,
            max_events_per_run=250,
            retention_days=30,
            max_unpartitioned_event_rows=5_000_000,
            postgresql_version="16.3",
        )

        metadata = task_pgsql_event_volume_metadata(profile)

        self.assertEqual(metadata["run_count"], 10_000)
        self.assertEqual(metadata["max_events_per_run"], 250)
        self.assertEqual(metadata["expected_event_rows"], 2_500_000)
        self.assertEqual(metadata["retention_days"], 30)
        self.assertEqual(metadata["max_unpartitioned_event_rows"], 5_000_000)
        self.assertFalse(metadata["partitioning_enabled"])
        self.assertEqual(metadata["postgresql_version"], "16.3")

    def test_queue_load_issues_accept_clean_measurement(self) -> None:
        profile = TaskPgsqlQueueLoadProfile(
            worker_count=2,
            run_count=3,
            queue_count=1,
            lease_seconds=30,
            max_attempts=2,
            abandon_limit=2,
            pool_size=4,
            postgresql_version="16.3",
            min_claims_per_second=1.0,
        )

        issues = task_pgsql_queue_load_issues(
            profile,
            claimed_run_ids=("run-1", "run-2", "run-3"),
            attempt_count=3,
            stale_token_commits=0,
            reaped_claims=2,
            elapsed_seconds=1.0,
        )

        self.assertEqual(issues, ())

    def test_queue_load_issues_detect_delivery_regressions(self) -> None:
        profile = TaskPgsqlQueueLoadProfile(
            worker_count=2,
            run_count=4,
            queue_count=2,
            lease_seconds=30,
            max_attempts=2,
            abandon_limit=1,
            pool_size=4,
            postgresql_version="16.3",
            min_claims_per_second=10.0,
        )

        issues = task_pgsql_queue_load_issues(
            profile,
            claimed_run_ids=("run-1", "run-1", "run-2"),
            attempt_count=1,
            stale_token_commits=1,
            reaped_claims=2,
            elapsed_seconds=1.0,
        )

        self.assertEqual(
            issues,
            (
                "queue.duplicate_claim",
                "queue.claims_missing",
                "queue.lost_attempt",
                "queue.stale_token_commit",
                "queue.reaper_unbounded",
                "queue.claim_throughput_low",
            ),
        )

    def test_queue_load_issues_detect_extra_attempts(self) -> None:
        profile = TaskPgsqlQueueLoadProfile(
            worker_count=2,
            run_count=3,
            queue_count=1,
            lease_seconds=30,
            max_attempts=2,
            abandon_limit=1,
            pool_size=4,
            postgresql_version="16.3",
        )

        issues = task_pgsql_queue_load_issues(
            profile,
            claimed_run_ids=("run-1", "run-2", "run-3"),
            attempt_count=4,
            stale_token_commits=0,
            reaped_claims=0,
            elapsed_seconds=1.0,
        )

        self.assertEqual(issues, ("queue.extra_attempt",))

    def test_queue_load_issues_detect_extra_claims(self) -> None:
        profile = TaskPgsqlQueueLoadProfile(
            worker_count=2,
            run_count=3,
            queue_count=1,
            lease_seconds=30,
            max_attempts=2,
            abandon_limit=1,
            pool_size=4,
            postgresql_version="16.3",
        )

        issues = task_pgsql_queue_load_issues(
            profile,
            claimed_run_ids=("run-1", "run-2", "run-3", "run-4"),
            attempt_count=4,
            stale_token_commits=0,
            reaped_claims=0,
            elapsed_seconds=1.0,
        )

        self.assertEqual(issues, ("queue.claims_extra",))

    def test_queue_load_issues_detect_duplicate_extra_claims(self) -> None:
        profile = TaskPgsqlQueueLoadProfile(
            worker_count=2,
            run_count=3,
            queue_count=1,
            lease_seconds=30,
            max_attempts=2,
            abandon_limit=1,
            pool_size=4,
            postgresql_version="16.3",
        )

        issues = task_pgsql_queue_load_issues(
            profile,
            claimed_run_ids=("run-1", "run-2", "run-3", "run-3"),
            attempt_count=3,
            stale_token_commits=0,
            reaped_claims=0,
            elapsed_seconds=1.0,
        )

        self.assertEqual(
            issues,
            ("queue.duplicate_claim", "queue.claims_extra"),
        )

    def test_event_volume_issues_accept_indexed_unpartitioned_profile(
        self,
    ) -> None:
        profile = TaskPgsqlEventVolumeProfile(
            run_count=100,
            max_events_per_run=100,
            retention_days=14,
            max_unpartitioned_event_rows=100_000,
            postgresql_version="16.3",
        )
        index_plan = (
            "Index Scan using ix_task_events_by_run_sequence on task_events",
        )

        issues = task_pgsql_event_volume_issues(
            profile,
            append_plan_lines=index_plan,
            fetch_plan_lines=index_plan,
        )

        self.assertEqual(issues, ())

    def test_event_volume_issues_detect_partition_and_plan_regressions(
        self,
    ) -> None:
        profile = TaskPgsqlEventVolumeProfile(
            run_count=2_000,
            max_events_per_run=1_000,
            retention_days=90,
            max_unpartitioned_event_rows=1_000_000,
            postgresql_version="16.3",
        )

        issues = task_pgsql_event_volume_issues(
            profile,
            append_plan_lines=("Seq Scan on task_events",),
            fetch_plan_lines=(),
        )

        self.assertEqual(
            issues,
            (
                "event.partitioning_required",
                "event.append_missing_index",
                "event.append_unbounded_scan",
                "event.fetch_missing_index",
            ),
        )

    def test_event_volume_issues_allow_explicit_partitioned_profile(
        self,
    ) -> None:
        profile = TaskPgsqlEventVolumeProfile(
            run_count=2_000,
            max_events_per_run=1_000,
            retention_days=90,
            max_unpartitioned_event_rows=1_000_000,
            postgresql_version="16.3",
            partitioning_enabled=True,
        )
        index_plan = (
            "Index Scan using ix_task_events_by_run_sequence on task_events",
        )

        issues = task_pgsql_event_volume_issues(
            profile,
            append_plan_lines=index_plan,
            fetch_plan_lines=index_plan,
        )

        self.assertEqual(issues, ())

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
        with self.assertRaises(AssertionError):
            TaskPgsqlQueueLoadProfile(
                worker_count=1,
                run_count=1,
                queue_count=1,
                lease_seconds=30,
                max_attempts=1,
                abandon_limit=2,
                pool_size=1,
                postgresql_version="16",
            )
        profile = TaskPgsqlQueueLoadProfile(
            worker_count=1,
            run_count=1,
            queue_count=1,
            lease_seconds=30,
            max_attempts=1,
            abandon_limit=1,
            pool_size=1,
            postgresql_version="16",
        )
        with self.assertRaises(AssertionError):
            task_pgsql_queue_load_metadata(
                profile,
                elapsed_seconds=0.0,
            )
        with self.assertRaises(AssertionError):
            task_pgsql_queue_load_issues(
                profile,
                claimed_run_ids="run-1",
                attempt_count=1,
                stale_token_commits=0,
                reaped_claims=0,
                elapsed_seconds=1.0,
            )
        with self.assertRaises(AssertionError):
            TaskPgsqlEventVolumeProfile(
                run_count=0,
                max_events_per_run=1,
                retention_days=1,
                max_unpartitioned_event_rows=1,
                postgresql_version="16",
            )
        event_profile = TaskPgsqlEventVolumeProfile(
            run_count=1,
            max_events_per_run=1,
            retention_days=1,
            max_unpartitioned_event_rows=1,
            postgresql_version="16",
        )
        with self.assertRaises(AssertionError):
            task_pgsql_event_volume_issues(
                event_profile,
                append_plan_lines="Seq Scan on task_events",
                fetch_plan_lines=(),
            )
        with self.assertRaises(AssertionError):
            task_pgsql_queue_load_issues(
                profile,
                claimed_run_ids=("run-1",),
                attempt_count=-1,
                stale_token_commits=0,
                reaped_claims=0,
                elapsed_seconds=1.0,
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
