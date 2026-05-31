from unittest import TestCase, main

from avalan.task import (
    FeatureGateCategory,
    FeatureGateCheckLocation,
    FeatureGateDiagnostic,
    FeatureGateSeverity,
    TaskFeature,
    feature_available,
    feature_diagnostic,
    feature_spec,
    gate_check_locations,
    require_feature,
    require_features,
)


class FeatureGateTest(TestCase):
    def test_dependency_gate_returns_stable_jsonschema_diagnostic(
        self,
    ) -> None:
        diagnostics = require_feature(
            TaskFeature.JSON_SCHEMA,
            module_finder=self._missing_module,
        )

        self.assertEqual(
            diagnostics,
            (
                FeatureGateDiagnostic(
                    code="dependency.jsonschema_missing",
                    path="input.schema",
                    category=FeatureGateCategory.DEPENDENCY,
                    severity=FeatureGateSeverity.ERROR,
                    message="JSON Schema validation requires the task extra.",
                    hint=(
                        "Install avalan[task] to validate object, array, or "
                        "json contracts."
                    ),
                ),
            ),
        )
        self.assertEqual(
            diagnostics[0].as_dict(),
            {
                "code": "dependency.jsonschema_missing",
                "path": "input.schema",
                "category": "dependency",
                "severity": "error",
                "message": "JSON Schema validation requires the task extra.",
                "hint": (
                    "Install avalan[task] to validate object, array, or json "
                    "contracts."
                ),
            },
        )

    def test_dependency_gate_allows_present_module(self) -> None:
        self.assertTrue(
            feature_available(
                TaskFeature.JSON_SCHEMA,
                module_finder=self._present_module,
            )
        )
        self.assertEqual(
            require_feature(
                TaskFeature.JSON_SCHEMA,
                module_finder=self._present_module,
            ),
            (),
        )

    def test_dependency_gate_swallows_import_discovery_errors(self) -> None:
        diagnostics = require_feature(
            TaskFeature.OPENTELEMETRY,
            module_finder=self._raising_module_finder,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].code, "dependency.task_otel_missing")
        self.assertEqual(diagnostics[0].path, "observability.sinks")

    def test_dependency_gate_hides_discovery_exception_details(self) -> None:
        diagnostics = require_feature(
            TaskFeature.POSTGRESQL,
            module_finder=self._leaking_module_finder,
        )
        rendered = " ".join(diagnostics[0].as_dict().values())

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].code, "dependency.task_pgsql_missing")
        self.assertNotIn("/Users/person/private", rendered)
        self.assertNotIn("raw secret", rendered)

    def test_migration_dependency_gate_requires_alembic_and_sqlalchemy(
        self,
    ) -> None:
        diagnostics = require_feature(
            TaskFeature.POSTGRESQL_MIGRATIONS,
            module_finder=self._missing_module,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].as_dict(),
            {
                "code": "dependency.task_pgsql_migrations_missing",
                "path": "store.postgresql.migrations",
                "category": "dependency",
                "severity": "error",
                "message": (
                    "PostgreSQL task migrations require Alembic and "
                    "SQLAlchemy."
                ),
                "hint": (
                    "Install the task migration dependencies before running "
                    "PostgreSQL schema migrations."
                ),
            },
        )

        seen_modules: list[str] = []

        def one_missing_module(module: str) -> object | None:
            seen_modules.append(module)
            if module == "sqlalchemy":
                return None
            return object()

        self.assertEqual(
            require_feature(
                TaskFeature.POSTGRESQL_MIGRATIONS,
                module_finder=one_missing_module,
            )[0].code,
            "dependency.task_pgsql_migrations_missing",
        )
        self.assertEqual(seen_modules, ["alembic", "sqlalchemy"])

    def test_configuration_gate_requires_explicit_enablement(self) -> None:
        self.assertFalse(feature_available(TaskFeature.RAW_STORAGE))
        self.assertEqual(
            require_feature(TaskFeature.RAW_STORAGE)[0].as_dict(),
            {
                "code": "feature.raw_storage_disabled",
                "path": "artifacts.storage",
                "category": "configuration",
                "severity": "error",
                "message": "Raw artifact byte storage is disabled.",
                "hint": (
                    "Enable raw storage only with encryption keys and a "
                    "retention policy configured."
                ),
            },
        )
        self.assertEqual(
            require_feature(TaskFeature.RAW_STORAGE, enabled=True),
            (),
        )

    def test_require_features_aggregates_in_order(self) -> None:
        diagnostics = require_features(
            (
                TaskFeature.POSTGRESQL,
                TaskFeature.REMOTE_URL_FILE_INPUTS,
                TaskFeature.FLOW_BACKED_TASKS,
            ),
            enabled_features=(TaskFeature.REMOTE_URL_FILE_INPUTS,),
            module_finder=self._missing_module,
        )

        self.assertEqual(
            [diagnostic.code for diagnostic in diagnostics],
            [
                "dependency.task_pgsql_missing",
                "feature.flow_backed_tasks_disabled",
            ],
        )
        self.assertEqual(
            [diagnostic.category for diagnostic in diagnostics],
            [
                FeatureGateCategory.DEPENDENCY,
                FeatureGateCategory.UNSUPPORTED,
            ],
        )

    def test_diagnostics_cover_initial_task_feature_set(self) -> None:
        expected_codes = {
            TaskFeature.DOCUMENT_CONVERSION: (
                "dependency.task_documents_missing"
            ),
            TaskFeature.FLOW_BACKED_TASKS: (
                "feature.flow_backed_tasks_disabled"
            ),
            TaskFeature.JSON_SCHEMA: "dependency.jsonschema_missing",
            TaskFeature.OPENTELEMETRY: "dependency.task_otel_missing",
            TaskFeature.POSTGRESQL: "dependency.task_pgsql_missing",
            TaskFeature.POSTGRESQL_MIGRATIONS: (
                "dependency.task_pgsql_migrations_missing"
            ),
            TaskFeature.PROMETHEUS: "dependency.task_prometheus_missing",
            TaskFeature.RAW_STORAGE: "feature.raw_storage_disabled",
            TaskFeature.REMOTE_URL_FILE_INPUTS: (
                "feature.remote_url_file_inputs_disabled"
            ),
            TaskFeature.TASK_CLI: "feature.task_cli_disabled",
        }

        self.assertEqual(set(TaskFeature), set(expected_codes))
        for feature, code in expected_codes.items():
            spec = feature_spec(feature)
            diagnostic = feature_diagnostic(feature)
            self.assertEqual(spec.code, code)
            self.assertEqual(diagnostic.code, code)
            self.assertNotIn("traceback", diagnostic.message.lower())
            self.assertNotIn("exception", diagnostic.hint.lower())

    def test_diagnostic_path_can_be_overridden(self) -> None:
        diagnostic = feature_diagnostic(
            TaskFeature.JSON_SCHEMA,
            path="output.schema",
        )

        self.assertEqual(diagnostic.path, "output.schema")

    def test_gate_check_locations_record_cli_and_sdk_ownership(self) -> None:
        self.assertEqual(
            gate_check_locations(TaskFeature.TASK_CLI),
            (FeatureGateCheckLocation.CLI,),
        )
        self.assertEqual(
            gate_check_locations(TaskFeature.JSON_SCHEMA),
            (FeatureGateCheckLocation.SDK,),
        )
        self.assertEqual(
            gate_check_locations(TaskFeature.POSTGRESQL_MIGRATIONS),
            (
                FeatureGateCheckLocation.CLI,
                FeatureGateCheckLocation.SDK,
            ),
        )
        self.assertEqual(
            FeatureGateCheckLocation.CLI.value,
            "avalan.cli.commands.task",
        )
        self.assertEqual(FeatureGateCheckLocation.SDK.value, "avalan.task")

    @staticmethod
    def _missing_module(module: str) -> object | None:
        return None

    @staticmethod
    def _present_module(module: str) -> object:
        return object()

    @staticmethod
    def _raising_module_finder(module: str) -> object | None:
        raise ModuleNotFoundError(module)

    @staticmethod
    def _leaking_module_finder(module: str) -> object | None:
        raise ValueError("/Users/person/private module has raw secret")


if __name__ == "__main__":
    main()
