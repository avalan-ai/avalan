"""Describe host-owned immutable execution files and runtime requirements."""

from ..types import assert_non_empty_string

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from json import dumps
from pathlib import PurePosixPath
from re import fullmatch


class ExecutionDeploymentError(ValueError):
    """Report deployment mismatch without exposing paths or file content."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__("task.deployment_mismatch: " + path)


class DeploymentFileRoot(StrEnum):
    APPLICATION = "application"
    RUNTIME = "runtime"


def deployment_path(value: str) -> str:
    """Require an unambiguous root-relative POSIX file name."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).as_posix() != value
    ):
        raise ExecutionDeploymentError("file.path")
    return value


def deployment_digest(value: str) -> str:
    if not isinstance(value, str) or fullmatch("[0-9a-f]{64}", value) is None:
        raise ExecutionDeploymentError("sha256")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DeploymentFile:
    root: DeploymentFileRoot
    path: str
    sha256: str

    def __post_init__(self) -> None:
        assert isinstance(self.root, DeploymentFileRoot)
        deployment_path(self.path)
        deployment_digest(self.sha256)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeploymentRuntimeOption:
    """Bind one host-allowlisted non-secret runtime option."""

    name: str
    value: str | int | bool

    def __post_init__(self) -> None:
        assert_non_empty_string(self.name, "runtime_option.name")
        assert type(self.value) in {str, int, bool}


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionDeployment:
    """Seal execution closure independently of mutable trigger revisions."""

    task_ref: str
    task_hash: str
    files: tuple[DeploymentFile, ...]
    runtime_version: str
    required_tools: tuple[str, ...] = ()
    runtime_options: tuple[DeploymentRuntimeOption, ...] = ()
    skill_manifest_id: str | None = None
    container_image_digests: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ExecutionDeploymentError("schema_version")
        deployment_path(self.task_ref)
        deployment_digest(self.task_hash)
        assert_non_empty_string(self.runtime_version, "runtime_version")
        assert isinstance(self.files, tuple) and self.files
        assert all(isinstance(value, DeploymentFile) for value in self.files)
        keys = tuple((value.root.value, value.path) for value in self.files)
        if keys != tuple(sorted(set(keys))):
            raise ExecutionDeploymentError("files")
        if (DeploymentFileRoot.APPLICATION.value, self.task_ref) not in keys:
            raise ExecutionDeploymentError("task_ref")
        assert isinstance(self.required_tools, tuple)
        for tool in self.required_tools:
            assert_non_empty_string(tool, "required_tool")
        if self.required_tools != tuple(sorted(set(self.required_tools))):
            raise ExecutionDeploymentError("required_tools")
        assert isinstance(self.runtime_options, tuple)
        assert all(
            isinstance(value, DeploymentRuntimeOption)
            for value in self.runtime_options
        )
        names = tuple(value.name for value in self.runtime_options)
        if names != tuple(sorted(set(names))):
            raise ExecutionDeploymentError("runtime_options")
        if self.skill_manifest_id is not None:
            deployment_digest(self.skill_manifest_id)
        if (
            not isinstance(self.container_image_digests, tuple)
            or any(
                not isinstance(value, str)
                or fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value) is None
                for value in self.container_image_digests
            )
            or self.container_image_digests
            != tuple(sorted(set(self.container_image_digests)))
        ):
            raise ExecutionDeploymentError("container_image_digests")

    @property
    def execution_deployment_id(self) -> str:
        return sha256(
            dumps(
                self.payload(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_ref": self.task_ref,
            "task_hash": self.task_hash,
            "runtime_version": self.runtime_version,
            "files": [
                {
                    "root": value.root.value,
                    "path": value.path,
                    "sha256": value.sha256,
                }
                for value in self.files
            ],
            "required_tools": list(self.required_tools),
            "runtime_options": [
                {"name": value.name, "value": value.value}
                for value in self.runtime_options
            ],
            "skill_manifest_id": self.skill_manifest_id,
            "container_image_digests": list(self.container_image_digests),
        }


def execution_deployment_from_payload(value: object) -> ExecutionDeployment:
    """Decode the complete schema-one manifest without legacy fallbacks."""
    payload = _fields(
        value,
        {
            "schema_version",
            "task_ref",
            "task_hash",
            "runtime_version",
            "files",
            "required_tools",
            "runtime_options",
            "skill_manifest_id",
            "container_image_digests",
        },
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
    ):
        raise ExecutionDeploymentError("schema_version")
    files = payload["files"]
    tools = payload["required_tools"]
    options = payload["runtime_options"]
    images = payload["container_image_digests"]
    if (
        not isinstance(files, list)
        or not isinstance(tools, list)
        or not isinstance(options, list)
        or not isinstance(images, list)
    ):
        raise ExecutionDeploymentError("manifest.arrays")
    decoded_files: list[DeploymentFile] = []
    for raw in files:
        item = _fields(raw, {"root", "path", "sha256"})
        try:
            root = DeploymentFileRoot(_string(item["root"]))
        except ValueError:
            raise ExecutionDeploymentError("file.root") from None
        decoded_files.append(
            DeploymentFile(
                root=root,
                path=_string(item["path"]),
                sha256=_string(item["sha256"]),
            )
        )
    decoded_options: list[DeploymentRuntimeOption] = []
    for raw in options:
        item = _fields(raw, {"name", "value"})
        option = item["value"]
        if type(option) not in {str, int, bool}:
            raise ExecutionDeploymentError("runtime_option.value")
        assert isinstance(option, str | int | bool)
        decoded_options.append(
            DeploymentRuntimeOption(name=_string(item["name"]), value=option)
        )
    return ExecutionDeployment(
        task_ref=_string(payload["task_ref"]),
        task_hash=_string(payload["task_hash"]),
        files=tuple(decoded_files),
        runtime_version=_string(payload["runtime_version"]),
        required_tools=tuple(_string(tool) for tool in tools),
        runtime_options=tuple(decoded_options),
        skill_manifest_id=_nullable_string(payload["skill_manifest_id"]),
        container_image_digests=tuple(_string(value) for value in images),
    )


def _fields(value: object, names: set[str]) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != names:
        raise ExecutionDeploymentError("manifest.fields")
    return {name: value[name] for name in names}


def _string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionDeploymentError("manifest.string")
    return value


def _nullable_string(value: object) -> str | None:
    return None if value is None else _string(value)
