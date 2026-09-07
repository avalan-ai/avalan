"""Verify the mounted command protocol before any target dispatch."""

from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from json import dumps, loads
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from avalan.task.container_invocation import (
    container_invocation_bytes,
    verify_container_invocation,
)
from avalan.task.deployment import (
    DeploymentFile,
    DeploymentFileRoot,
    ExecutionDeployment,
    ExecutionDeploymentError,
)
from avalan.task.deployment_closure import ExecutionClosure
from avalan.task.provenance import TriggerInvocationContext
from avalan.task.store import TaskExecutionContext


class ContainerInvocationTest(TestCase):
    def test_closed_protocol_verifies_actual_mount_and_rejects_changes(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Path(__file__).resolve().parents[2] / "src/avalan"
            (root / "task.toml").write_text(
                '[execution]\ntype="agent"\nref="agent.toml"'
            )
            (root / "agent.toml").write_text(
                '[agent]\ninstructions="retained"\n[tool]\nenable=[]'
            )
            manifest = ExecutionDeployment(
                task_ref="task.toml",
                task_hash="a" * 64,
                files=ExecutionClosure(root, runtime).collect("task.toml"),
                runtime_version=version("avalan"),
            )
            now = datetime(2026, 9, 7, tzinfo=UTC)
            trigger = TriggerInvocationContext(
                trigger_id="trigger",
                trigger_revision=1,
                occurrence_id="occurrence",
                scheduled_at=now,
                dispatched_at=now,
            )
            context = TaskExecutionContext(
                run_id="run",
                attempt_id="attempt",
                attempt_number=2,
                trigger=trigger,
                deployment=manifest,
                metadata={"private": "never transport"},
            )
            path = root / "invocation.json"
            original = container_invocation_bytes(
                context, input_value={"frozen": [1, True]}
            )
            path.write_bytes(original)
            verified = verify_container_invocation(
                invocation_path=path, application_root=root
            )
            self.assertEqual(verified.context.trigger, trigger)
            self.assertEqual(verified.context.deployment, manifest)
            self.assertEqual(dict(verified.context.metadata), {})
            self.assertEqual(verified.input_value, {"frozen": (1, True)})
            self.assertNotIn(b"private", original)
            for raw in (
                {},
                {**loads(original), "file_delivery": "yes"},
                {"format": "unsupported", "context": {}, "deployment_id": "x"},
            ):
                path.write_text(dumps(raw))
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "container.protocol"
                ):
                    verify_container_invocation(
                        invocation_path=path, application_root=root
                    )
            for content in (b"invalid", b"\xff"):
                path.write_bytes(content)
                with self.assertRaisesRegex(
                    ExecutionDeploymentError, "container.invocation"
                ):
                    verify_container_invocation(
                        invocation_path=path, application_root=root
                    )
            path.unlink()
            with self.assertRaisesRegex(
                ExecutionDeploymentError, "container.invocation"
            ):
                verify_container_invocation(
                    invocation_path=path, application_root=root
                )
            changed = loads(original)
            changed["deployment_id"] = "b" * 64
            path.write_text(dumps(changed))
            with self.assertRaisesRegex(
                ExecutionDeploymentError, "container.identity"
            ):
                verify_container_invocation(
                    invocation_path=path, application_root=root
                )
            path.write_bytes(
                container_invocation_bytes(
                    replace(
                        context,
                        deployment=replace(manifest, runtime_version="wrong"),
                    )
                )
            )
            with self.assertRaisesRegex(
                ExecutionDeploymentError, "container.runtime_version"
            ):
                verify_container_invocation(
                    invocation_path=path, application_root=root
                )
            path.write_bytes(original)
            (root / "agent.toml").write_text('[agent]\ninstructions="changed"')
            with self.assertRaisesRegex(
                ExecutionDeploymentError, "container.closure"
            ):
                verify_container_invocation(
                    invocation_path=path, application_root=root
                )
            (root / "agent.toml").write_text(
                '[agent]\ninstructions="retained"\n[tool]\nenable=[]'
            )
            (root / "extra.txt").write_text("changed")
            extra = DeploymentFile(
                root=DeploymentFileRoot.APPLICATION,
                path="extra.txt",
                sha256="b" * 64,
            )
            expanded = replace(
                manifest,
                files=tuple(
                    sorted(
                        (*manifest.files, extra),
                        key=lambda item: (item.root, item.path),
                    )
                ),
            )
            path.write_bytes(
                container_invocation_bytes(
                    replace(context, deployment=expanded)
                )
            )
            with self.assertRaisesRegex(
                ExecutionDeploymentError, "container.file"
            ):
                verify_container_invocation(
                    invocation_path=path, application_root=root
                )

    def test_auxiliary_file_delivery_requires_the_application_template(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Path(__file__).resolve().parents[2] / "src/avalan"
            (root / "task.toml").write_text(
                '[input]\ntype="string"\n[execution]\ntype="agent"\nref="agent.toml"'
            )
            (root / "agent.toml").write_text(
                '[agent]\nuser_template="agent.md"'
            )
            manifest = ExecutionDeployment(
                task_ref="task.toml",
                task_hash="a" * 64,
                files=ExecutionClosure(root, runtime).collect("task.toml"),
                runtime_version=version("avalan"),
            )
            now = datetime(2026, 9, 7, tzinfo=UTC)
            context = TaskExecutionContext(
                run_id="run",
                attempt_id="attempt",
                attempt_number=1,
                deployment=manifest,
                trigger=TriggerInvocationContext(
                    trigger_id="trigger",
                    trigger_revision=1,
                    occurrence_id="occurrence",
                    scheduled_at=now,
                    dispatched_at=now,
                ),
            )
            path = root / "invocation.json"
            path.write_bytes(
                container_invocation_bytes(context, file_delivery=True)
            )
            with self.assertRaisesRegex(
                ExecutionDeploymentError, "file.unavailable"
            ):
                verify_container_invocation(
                    invocation_path=path, application_root=root
                )
            (root / "agent.md").write_text("application file prompt")
            with self.assertRaisesRegex(
                ExecutionDeploymentError, "container.closure"
            ):
                verify_container_invocation(
                    invocation_path=path, application_root=root
                )
            complete = replace(
                manifest,
                files=ExecutionClosure(root, runtime).collect(
                    "task.toml", file_delivery=True
                ),
            )
            path.write_bytes(
                container_invocation_bytes(
                    replace(context, deployment=complete), file_delivery=True
                )
            )
            self.assertEqual(
                verify_container_invocation(
                    invocation_path=path, application_root=root
                ).context.deployment,
                complete,
            )
