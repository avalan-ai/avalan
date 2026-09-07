"""Construct the task worker from the retained scheduled CLI host."""

from ..agent.loader import OrchestratorLoader
from ..model.hubs.huggingface import HuggingfaceHub
from ..pgsql import PgsqlDatabase
from ..task.deployment import ExecutionDeploymentError
from ..task.privacy import HmacProvider
from ..task.queues.pgsql import PgsqlTaskQueue
from ..task.stores.pgsql import PgsqlTaskStore
from ..task.worker import TaskWorker, TaskWorkerShutdown
from .trigger_host import TriggerCliHost
from .trigger_runtime import encrypted_input_store

from argparse import Namespace
from contextlib import AsyncExitStack
from logging import Logger, getLogger
from pathlib import Path
from typing import cast
from uuid import uuid4


async def deployment_worker(
    args: Namespace,
    *,
    database: PgsqlDatabase,
    stack: AsyncExitStack,
    hub: object | None,
    logger: Logger | None,
    shutdown: TaskWorkerShutdown,
    hmac_provider: HmacProvider | None,
) -> TaskWorker:
    """Load the same verified files and encrypted bytes used by apply."""
    if hub is None:
        raise ExecutionDeploymentError("cli.hub_required")
    cipher, artifacts = encrypted_input_store(args, database)
    loader = OrchestratorLoader(
        hub=cast(HuggingfaceHub, hub),
        logger=logger or getLogger("avalan.task"),
        participant_id=uuid4(),
        stack=stack,
    )
    host = TriggerCliHost(Path(args.deployment_root), loader)
    await host.restore()
    bindings = tuple(host.bindings.values())
    if not bindings:
        raise ExecutionDeploymentError("cli.catalog_empty")
    binding = bindings[0]
    return TaskWorker(
        PgsqlTaskStore(database),
        PgsqlTaskQueue(database),
        target=binding.target,
        execution_deployments=host.catalog,
        encryption_provider=cipher,
        raw_storage_allowed=True,
        artifact_store=artifacts,
        execution_roots=tuple(item.application_root for item in bindings),
        definition_base=binding.application_root,
        hmac_provider=hmac_provider,
        worker_id=args.worker_id,
        queue_name=args.queue or "default",
        lease_seconds=args.lease_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        shutdown=shutdown,
    )
