from .definition import (
    AtTrigger,
    CronTrigger,
    IntervalTrigger,
    JsonValue,
    RecurringPolicy,
    TriggerConfiguration,
    utc_text,
)

from collections.abc import Mapping
from hashlib import sha256
from json import dumps


def plain_value(value: object) -> JsonValue:
    """Convert immutable input snapshots into JSON containers."""
    if isinstance(value, Mapping):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain_value(item) for item in value]
    assert value is None or isinstance(value, str | bool | int | float)
    return value


def canonical_configuration(
    configuration: TriggerConfiguration,
    *,
    task_definition_id: str,
    execution_deployment_id: str,
) -> str:
    """Encode semantic identity using host-resolved execution identities."""
    assert isinstance(configuration, TriggerConfiguration)
    assert isinstance(task_definition_id, str) and task_definition_id
    assert isinstance(execution_deployment_id, str) and execution_deployment_id
    spec = configuration.schedule
    schedule: dict[str, JsonValue]
    match spec:
        case CronTrigger():
            schedule = {
                "type": "cron",
                "expression": spec.expression,
                "timezone": spec.timezone,
            }
        case IntervalTrigger():
            schedule = {
                "type": "interval",
                "every_seconds": spec.every_seconds,
                "start_at": utc_text(spec.start_at) if spec.start_at else None,
            }
        case AtTrigger():
            schedule = {"type": "at", "at": utc_text(spec.at)}
    policy: dict[str, JsonValue] = {
        "misfire_grace_seconds": configuration.policy.misfire_grace_seconds
    }
    if isinstance(configuration.policy, RecurringPolicy):
        policy.update(
            misfire=configuration.policy.misfire.value,
            overlap=configuration.policy.overlap.value,
        )
    payload = {
        "schema_version": 1,
        "schedule_semantics_version": 1,
        "task_definition_id": task_definition_id,
        "execution_deployment_id": execution_deployment_id,
        "schedule": schedule,
        "input": {
            "value": plain_value(configuration.input.value),
            "bindings": [
                {"path": binding.path, "source": binding.source.value}
                for binding in sorted(
                    configuration.input.bindings, key=lambda item: item.path
                )
            ],
        },
        "policy": policy,
    }
    return dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def configuration_hash(
    configuration: TriggerConfiguration,
    *,
    task_definition_id: str,
    execution_deployment_id: str,
) -> str:
    """Hash canonical semantic configuration without operational state."""
    value = canonical_configuration(
        configuration,
        task_definition_id=task_definition_id,
        execution_deployment_id=execution_deployment_id,
    )
    return sha256(value.encode("utf-8")).hexdigest()
