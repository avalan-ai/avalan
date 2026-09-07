"""Derive deployment requirements from resolved task runtime contracts."""

from .definition import TaskDefinition
from .deployment import ExecutionDeploymentError
from .skills import task_definition_with_skills_identity

from collections.abc import Mapping
from hashlib import sha256
from json import dumps
from pathlib import Path


async def deployment_skill_manifest_id(
    definition: TaskDefinition, *, definition_base: Path
) -> str | None:
    """Resolve the current skill registry and hash its canonical identity."""
    resolved = await task_definition_with_skills_identity(
        definition, schema_base_path=definition_base
    )
    identity = resolved.skills_identity
    if identity is None:
        return None
    return sha256(
        dumps(
            _plain(identity),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def deployment_container_images(definition: TaskDefinition) -> tuple[str, ...]:
    """Return every enabled task container's immutable image reference."""
    images: set[str] = set()
    for settings in (
        definition.container.attempt,
        definition.container.worker_envelope,
    ):
        if settings is None or not settings.enabled:
            continue
        if settings.profile is None:
            raise ExecutionDeploymentError("container.profile")
        image = settings.profile.image
        assert image.digest is not None
        repository = image.reference.split("@", 1)[0]
        # Tags do not contribute mutable identity; the digest is mandatory.
        if repository.rfind(":") > repository.rfind("/"):
            repository = repository.rsplit(":", 1)[0]
        images.add(repository + "@" + image.digest)
    return tuple(sorted(images))


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value
