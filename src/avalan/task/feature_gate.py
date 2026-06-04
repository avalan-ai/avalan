from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec

TASK_CLI_CHECK_MODULE = "avalan.cli.commands.task"
TASK_SDK_CHECK_MODULE = "avalan.task"
TASK_WORKER_CHECK_MODULE = "avalan.task.worker"

ModuleFinder = Callable[[str], object | None]


class FeatureGateCategory(StrEnum):
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    UNSUPPORTED = "unsupported"


class FeatureGateCheckLocation(StrEnum):
    CLI = TASK_CLI_CHECK_MODULE
    SDK = TASK_SDK_CHECK_MODULE
    WORKER = TASK_WORKER_CHECK_MODULE


class FeatureGateSeverity(StrEnum):
    ERROR = "error"


class TaskFeature(StrEnum):
    DOCUMENT_CONVERSION = "document_conversion"
    FLOW_BACKED_TASKS = "flow_backed_tasks"
    JSON_SCHEMA = "json_schema"
    OPENTELEMETRY = "opentelemetry"
    PDF_IMAGE_CONVERSION = "pdf_image_conversion"
    POSTGRESQL = "postgresql"
    POSTGRESQL_MIGRATIONS = "postgresql_migrations"
    PROMETHEUS = "prometheus"
    QUEUE_WORKERS = "queue_workers"
    RAW_STORAGE = "raw_storage"
    REMOTE_URL_FILE_INPUTS = "remote_url_file_inputs"
    TASK_CLI = "task_cli"


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureGateDiagnostic:
    code: str
    path: str
    category: FeatureGateCategory
    severity: FeatureGateSeverity
    message: str
    hint: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "hint": self.hint,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureGateSpec:
    feature: TaskFeature
    code: str
    path: str
    category: FeatureGateCategory
    message: str
    hint: str
    modules: tuple[str, ...] = ()
    check_locations: tuple[FeatureGateCheckLocation, ...] = (
        FeatureGateCheckLocation.SDK,
    )


_FEATURE_GATE_SPECS = {
    TaskFeature.DOCUMENT_CONVERSION: FeatureGateSpec(
        feature=TaskFeature.DOCUMENT_CONVERSION,
        code="dependency.task_documents_missing",
        path="input.files",
        category=FeatureGateCategory.DEPENDENCY,
        message="Document conversion requires the task-documents extra.",
        hint="Install avalan[task-documents] to convert document inputs.",
        modules=("markdownify",),
    ),
    TaskFeature.FLOW_BACKED_TASKS: FeatureGateSpec(
        feature=TaskFeature.FLOW_BACKED_TASKS,
        code="feature.flow_backed_tasks_disabled",
        path="target.flow",
        category=FeatureGateCategory.UNSUPPORTED,
        message="Flow-backed task targets are not enabled.",
        hint=(
            "Use agent or model targets until flow runtime task contract "
            "support is enabled."
        ),
    ),
    TaskFeature.JSON_SCHEMA: FeatureGateSpec(
        feature=TaskFeature.JSON_SCHEMA,
        code="dependency.jsonschema_missing",
        path="input.schema",
        category=FeatureGateCategory.DEPENDENCY,
        message="JSON Schema validation requires the task extra.",
        hint=(
            "Install avalan[task] to validate object, array, or json "
            "contracts."
        ),
        modules=("jsonschema",),
    ),
    TaskFeature.OPENTELEMETRY: FeatureGateSpec(
        feature=TaskFeature.OPENTELEMETRY,
        code="dependency.task_otel_missing",
        path="observability.sinks",
        category=FeatureGateCategory.DEPENDENCY,
        message=(
            "OpenTelemetry task observability requires the task-otel extra."
        ),
        hint="Install avalan[task-otel] before enabling OpenTelemetry sinks.",
        modules=("opentelemetry.sdk",),
    ),
    TaskFeature.PDF_IMAGE_CONVERSION: FeatureGateSpec(
        feature=TaskFeature.PDF_IMAGE_CONVERSION,
        code="dependency.task_pdf_images_missing",
        path="input.file_conversions.pdf_image",
        category=FeatureGateCategory.DEPENDENCY,
        message="PDF image conversion requires the task-pdf-images extra.",
        hint="Install avalan[task-pdf-images] to rasterize PDF inputs.",
        modules=("pypdfium2",),
    ),
    TaskFeature.POSTGRESQL: FeatureGateSpec(
        feature=TaskFeature.POSTGRESQL,
        code="dependency.task_pgsql_missing",
        path="store.postgresql",
        category=FeatureGateCategory.DEPENDENCY,
        message="PostgreSQL task storage requires the task-pgsql extra.",
        hint="Install avalan[task-pgsql] and configure a durable task store.",
        modules=("psycopg", "psycopg_pool"),
    ),
    TaskFeature.POSTGRESQL_MIGRATIONS: FeatureGateSpec(
        feature=TaskFeature.POSTGRESQL_MIGRATIONS,
        code="dependency.task_pgsql_migrations_missing",
        path="store.postgresql.migrations",
        category=FeatureGateCategory.DEPENDENCY,
        message="PostgreSQL task migrations require Alembic and SQLAlchemy.",
        hint=(
            "Install the task migration dependencies before running "
            "PostgreSQL schema migrations."
        ),
        modules=("alembic", "sqlalchemy"),
        check_locations=(
            FeatureGateCheckLocation.CLI,
            FeatureGateCheckLocation.SDK,
        ),
    ),
    TaskFeature.PROMETHEUS: FeatureGateSpec(
        feature=TaskFeature.PROMETHEUS,
        code="dependency.task_prometheus_missing",
        path="observability.sinks",
        category=FeatureGateCategory.DEPENDENCY,
        message="Prometheus task metrics require the task-prometheus extra.",
        hint=(
            "Install avalan[task-prometheus] before enabling Prometheus sinks."
        ),
        modules=("prometheus_client",),
    ),
    TaskFeature.QUEUE_WORKERS: FeatureGateSpec(
        feature=TaskFeature.QUEUE_WORKERS,
        code="dependency.task_worker_pgsql_missing",
        path="worker.store.postgresql",
        category=FeatureGateCategory.DEPENDENCY,
        message="Task queue workers require the task-pgsql extra.",
        hint="Install avalan[task-pgsql] before starting task workers.",
        modules=("psycopg", "psycopg_pool"),
        check_locations=(
            FeatureGateCheckLocation.CLI,
            FeatureGateCheckLocation.WORKER,
        ),
    ),
    TaskFeature.RAW_STORAGE: FeatureGateSpec(
        feature=TaskFeature.RAW_STORAGE,
        code="feature.raw_storage_disabled",
        path="artifacts.storage",
        category=FeatureGateCategory.CONFIGURATION,
        message="Raw artifact byte storage is disabled.",
        hint=(
            "Enable raw storage only with encryption keys and a retention "
            "policy configured."
        ),
    ),
    TaskFeature.REMOTE_URL_FILE_INPUTS: FeatureGateSpec(
        feature=TaskFeature.REMOTE_URL_FILE_INPUTS,
        code="feature.remote_url_file_inputs_disabled",
        path="input.files",
        category=FeatureGateCategory.CONFIGURATION,
        message="Remote URL file inputs are disabled.",
        hint=(
            "Enable remote URL inputs only after SSRF controls, redirects, "
            "and size limits are configured."
        ),
    ),
    TaskFeature.TASK_CLI: FeatureGateSpec(
        feature=TaskFeature.TASK_CLI,
        code="feature.task_cli_disabled",
        path="cli.task",
        category=FeatureGateCategory.CONFIGURATION,
        message="The task CLI is disabled.",
        hint=(
            "Enable the task CLI after task store and privacy policy "
            "configuration are available."
        ),
        check_locations=(FeatureGateCheckLocation.CLI,),
    ),
}


def feature_spec(feature: TaskFeature) -> FeatureGateSpec:
    assert feature in _FEATURE_GATE_SPECS
    return _FEATURE_GATE_SPECS[feature]


def feature_diagnostic(
    feature: TaskFeature, *, path: str | None = None
) -> FeatureGateDiagnostic:
    spec = feature_spec(feature)
    return FeatureGateDiagnostic(
        code=spec.code,
        path=path or spec.path,
        category=spec.category,
        severity=FeatureGateSeverity.ERROR,
        message=spec.message,
        hint=spec.hint,
    )


def gate_check_locations(
    feature: TaskFeature,
) -> tuple[FeatureGateCheckLocation, ...]:
    return feature_spec(feature).check_locations


def feature_available(
    feature: TaskFeature,
    *,
    enabled: bool = False,
    module_finder: ModuleFinder = find_spec,
) -> bool:
    spec = feature_spec(feature)
    if not spec.modules:
        return enabled

    return all(
        _module_available(module, module_finder) for module in spec.modules
    )


def require_feature(
    feature: TaskFeature,
    *,
    enabled: bool = False,
    path: str | None = None,
    module_finder: ModuleFinder = find_spec,
) -> tuple[FeatureGateDiagnostic, ...]:
    if feature_available(
        feature, enabled=enabled, module_finder=module_finder
    ):
        return ()
    return (feature_diagnostic(feature, path=path),)


def require_features(
    features: Iterable[TaskFeature],
    *,
    enabled_features: Iterable[TaskFeature] = (),
    module_finder: ModuleFinder = find_spec,
) -> tuple[FeatureGateDiagnostic, ...]:
    enabled_feature_set = set(enabled_features)
    diagnostics: list[FeatureGateDiagnostic] = []
    for feature in features:
        diagnostics.extend(
            require_feature(
                feature,
                enabled=feature in enabled_feature_set,
                module_finder=module_finder,
            )
        )
    return tuple(diagnostics)


def _module_available(module: str, module_finder: ModuleFinder) -> bool:
    try:
        return module_finder(module) is not None
    except (ImportError, ValueError):
        return False
