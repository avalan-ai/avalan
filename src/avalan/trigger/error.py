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


class TriggerError(ValueError):
    """Report a safe trigger diagnostic without echoing configuration."""

    def __init__(self, code: TriggerErrorCode, path: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code.value}: {path}")
