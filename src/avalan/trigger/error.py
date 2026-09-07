from enum import StrEnum


class TriggerErrorCode(StrEnum):
    INVALID_CONFIG = "trigger.invalid_config"
    UNSUPPORTED_VERSION = "trigger.unsupported_version"
    INVALID_BINDING = "trigger.invalid_binding"
    INVALID_SCHEDULE = "trigger.invalid_schedule"
    UNKNOWN_TIMEZONE = "trigger.unknown_timezone"
    IMPOSSIBLE_SCHEDULE = "trigger.impossible_schedule"
    SEARCH_BUDGET_EXHAUSTED = "trigger.search_budget_exhausted"
    DATETIME_OVERFLOW = "trigger.datetime_overflow"
    PAST_SCHEDULE = "trigger.past_schedule"
    QUEUE_REQUIRED = "trigger.queue_required"
    CAPABILITY_UNAVAILABLE = "trigger.capability_unavailable"
    DEPLOYMENT_MISMATCH = "trigger.deployment_mismatch"
    ARTIFACT_NOT_DURABLE = "trigger.artifact_not_durable"
    PROVIDER_REFERENCE_EXPIRED = "trigger.provider_reference_expired"
    CONFLICT = "trigger.conflict"
    STORE_INCOMPATIBLE = "trigger.store_incompatible"
    SCHEMA_MISMATCH = "trigger.schema_mismatch"
    ADMISSION_RETRYABLE = "trigger.admission_retryable"
    ADMISSION_EXHAUSTED = "trigger.admission_exhausted"
    COMMIT_UNKNOWN = "trigger.commit_unknown"
    SHUTDOWN_TIMEOUT = "trigger.shutdown_timeout"


class TriggerError(ValueError):
    """Report a safe trigger diagnostic without echoing configuration."""

    def __init__(self, code: TriggerErrorCode, path: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code.value}: {path}")
