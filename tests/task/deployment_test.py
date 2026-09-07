from copy import deepcopy
from dataclasses import replace
from unittest import TestCase

from avalan.task.deployment import (
    DeploymentFile,
    DeploymentFileRoot,
    DeploymentRuntimeOption,
    ExecutionDeployment,
    ExecutionDeploymentError,
    deployment_digest,
    deployment_path,
    execution_deployment_from_payload,
)


def deployment() -> ExecutionDeployment:
    return ExecutionDeployment(
        task_ref="task.toml",
        task_hash="a" * 64,
        files=(
            DeploymentFile(
                root=DeploymentFileRoot.APPLICATION,
                path="task.toml",
                sha256="b" * 64,
            ),
        ),
        runtime_version="1.0",
        required_tools=("search",),
        runtime_options=(
            DeploymentRuntimeOption(name="concurrency_limit", value=1),
        ),
        skill_manifest_id="c" * 64,
        container_image_digests=("registry/image@sha256:" + "d" * 64,),
    )


class ExecutionDeploymentTest(TestCase):
    def test_manifest_round_trip_and_identity(self) -> None:
        value = deployment()
        self.assertEqual(
            execution_deployment_from_payload(value.payload()), value
        )
        self.assertEqual(len(value.execution_deployment_id), 64)
        for changed in (
            replace(value, runtime_version="2"),
            replace(value, task_hash="0" * 64),
            replace(value, required_tools=()),
            replace(value, runtime_options=()),
            replace(value, skill_manifest_id=None),
            replace(value, container_image_digests=()),
        ):
            self.assertNotEqual(
                changed.execution_deployment_id, value.execution_deployment_id
            )
            self.assertEqual(
                execution_deployment_from_payload(changed.payload()), changed
            )

    def test_manifest_rejects_ambiguous_identity(self) -> None:
        value = deployment()
        for fields in (
            {"schema_version": True},
            {"schema_version": 2},
            {"files": value.files * 2},
            {"task_ref": "other.toml"},
            {"required_tools": ("z", "a")},
            {"runtime_options": value.runtime_options * 2},
            {"container_image_digests": ("image:latest",)},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaises(ExecutionDeploymentError),
            ):
                replace(value, **fields)
        for path in (
            "",
            "/root",
            "../file",
            "a/../file",
            "a//file",
            "a/./file",
            "a\\file",
            "a\x00file",
            "a/",
        ):
            with (
                self.subTest(path=path),
                self.assertRaises(ExecutionDeploymentError),
            ):
                deployment_path(path)
        for digest in ("", "A" * 64, "a" * 63, "g" * 64):
            with self.assertRaises(ExecutionDeploymentError):
                deployment_digest(digest)

    def test_closed_decoder_rejects_bad_types_and_versions(self) -> None:
        payload = deployment().payload()
        invalid: list[object] = [None, {}, {**payload, "extra": 1}]
        for key, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("files", ()),
            ("required_tools", ()),
            ("runtime_options", ()),
            (
                "files",
                [{"root": "host", "path": "task.toml", "sha256": "b" * 64}],
            ),
            ("files", [{}]),
            ("runtime_options", [{"name": "secret", "value": []}]),
            ("runtime_options", [{"name": "", "value": "value"}]),
            ("task_ref", 1),
            ("runtime_version", " "),
            ("skill_manifest_id", False),
        ):
            invalid.append({**deepcopy(payload), key: value})
        for invalid_value in invalid:
            with (
                self.subTest(value=invalid_value),
                self.assertRaises(ExecutionDeploymentError),
            ):
                execution_deployment_from_payload(invalid_value)
