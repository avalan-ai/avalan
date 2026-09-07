"""Verify the scheduled task command's mounted invocation before dispatch."""

from .deployment import ExecutionDeploymentError
from .deployment_closure import ExecutionClosure
from .execution_codec import _plain, context_from_payload, context_to_payload
from .store import (
    TaskExecutionContext,
    TaskSnapshotValue,
    freeze_snapshot_value,
)

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from json import JSONDecodeError, dumps, loads
from pathlib import Path

CONTAINER_INVOCATION_PATH = "/run/avalan/task-invocation.json"
CONTAINER_DEPLOYMENT_ROOT = "/workspace"
CONTAINER_INVOCATION_PROTOCOL = "avalan.task.container_invocation.v1"


@dataclass(frozen=True, slots=True)
class ContainerTaskInvocation:
    """Carry the admitted public context and frozen executable input."""

    context: TaskExecutionContext
    input_value: TaskSnapshotValue


def container_invocation_bytes(
    context: TaskExecutionContext,
    *,
    input_value: object = None,
    file_delivery: bool = False,
) -> bytes:
    """Serialize invocation identity without claim credentials or metadata."""
    assert isinstance(file_delivery, bool)
    assert context.trigger is not None and context.deployment is not None
    public = TaskExecutionContext(
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        attempt_number=context.attempt_number,
        trigger=context.trigger,
        deployment=context.deployment,
    )
    return dumps(
        {
            "format": CONTAINER_INVOCATION_PROTOCOL,
            "deployment_id": context.deployment.execution_deployment_id,
            "context": context_to_payload(public),
            "file_delivery": file_delivery,
            "input_value": _plain(freeze_snapshot_value(input_value)),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def verify_container_invocation(
    *,
    invocation_path: Path = Path(CONTAINER_INVOCATION_PATH),
    application_root: Path = Path(CONTAINER_DEPLOYMENT_ROOT),
) -> ContainerTaskInvocation:
    """Load the command protocol and verify mounted bytes before dispatch.

    The host admits only image digests explicitly trusted to call this
    boundary before their existing task command executes a target.
    """
    try:
        raw = loads(invocation_path.read_bytes())
    except (OSError, UnicodeError, JSONDecodeError):
        raise ExecutionDeploymentError("container.invocation") from None
    if (
        not isinstance(raw, Mapping)
        or set(raw)
        != {
            "format",
            "deployment_id",
            "context",
            "input_value",
            "file_delivery",
        }
        or raw["format"] != CONTAINER_INVOCATION_PROTOCOL
        or not isinstance(raw["file_delivery"], bool)
    ):
        raise ExecutionDeploymentError("container.protocol")
    context = context_from_payload(raw["context"])
    manifest = context.deployment
    if (
        manifest is None
        or context.trigger is None
        or context.claim is not None
        or context.metadata
        or raw["deployment_id"] != manifest.execution_deployment_id
    ):
        raise ExecutionDeploymentError("container.identity")
    if manifest.runtime_version != version("avalan"):
        raise ExecutionDeploymentError("container.runtime_version")
    closure = ExecutionClosure(
        application_root, Path(__file__).resolve().parents[1]
    )
    required = closure.collect(
        manifest.task_ref, file_delivery=raw["file_delivery"]
    )
    declared = {(item.root, item.path): item for item in manifest.files}
    if any(declared.get((item.root, item.path)) != item for item in required):
        raise ExecutionDeploymentError("container.closure")
    for item in manifest.files:
        closure.read(item.root, item.path)
        if closure.files[(item.root, item.path)] != item:
            raise ExecutionDeploymentError("container.file")
    return ContainerTaskInvocation(
        context, freeze_snapshot_value(raw["input_value"])
    )
