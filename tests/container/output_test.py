from collections.abc import Iterator
from contextlib import contextmanager
from os import chmod, mkfifo, stat_result
from os import open as os_open
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import cast
from unittest import TestCase, main
from unittest.mock import patch

from avalan.container import (
    ContainerArchiveEntry,
    ContainerArchiveEntryType,
    ContainerAuditEventType,
    ContainerExecutionScope,
    ContainerOutputArtifact,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerOutputDiagnostic,
    ContainerOutputDiagnosticCode,
    ContainerOutputMediaPolicy,
    ContainerOutputPolicy,
    ContainerOutputValidationResult,
    ContainerPartialOutput,
    ContainerPartialOutputMode,
    ContainerPartialOutputPolicy,
    ContainerPartialOutputReason,
    ContainerResultStatus,
    output_contracts_from_policy,
    validate_archive_entries,
    validate_copied_outputs,
    validate_output_stream,
)


class ContainerOutputTest(TestCase):
    def test_output_contracts_and_bounded_streams(self) -> None:
        contracts = output_contracts_from_policy(
            ContainerOutputPolicy(
                max_stdout_bytes=16,
                max_stderr_bytes=8,
                max_artifact_bytes=128,
                allow_artifacts=True,
            )
        )
        stdout = contracts[0]
        stderr = contracts[1]
        accepted = validate_output_stream(b"hello", stdout)
        denied = validate_output_stream(b"too long!", stderr)
        disabled = validate_output_stream(
            b"hidden",
            ContainerOutputContract(
                contract_type="stdout",
                max_bytes=16,
                enabled=False,
            ),
        )

        self.assertEqual(
            [contract.to_dict()["contract_type"] for contract in contracts],
            [
                "stdout",
                "stderr",
                "generated_file",
                "task_artifact",
                "runtime_envelope_artifact",
            ],
        )
        self.assertEqual(
            accepted.decision,
            ContainerOutputDecisionType.ACCEPT,
        )
        self.assertEqual(accepted.total_bytes, 5)
        self.assertEqual(denied.decision, ContainerOutputDecisionType.REJECT)
        self.assertEqual(
            denied.diagnostics[0].code,
            ContainerOutputDiagnosticCode.TOO_LARGE,
        )
        self.assertEqual(
            disabled.diagnostics[0].to_dict()["code"],
            "container.output.contract_disabled",
        )

    def test_accepts_generated_files_and_task_artifacts(self) -> None:
        with _workspace() as root:
            _write(root, "nested/report.txt", b"summary")
            _write(root, "metadata.json", b'{"ok": true}')
            generated = validate_copied_outputs(
                str(root),
                _artifact_contract(
                    ContainerOutputContractType.GENERATED_FILE,
                    max_bytes=128,
                    max_files=4,
                ),
            )
            task = validate_copied_outputs(
                str(root),
                _artifact_contract(
                    ContainerOutputContractType.TASK_ARTIFACT,
                    max_bytes=128,
                    max_files=4,
                ),
            )

            self.assertEqual(
                generated.decision,
                ContainerOutputDecisionType.ACCEPT,
            )
            self.assertEqual(generated.file_count, 2)
            self.assertEqual(generated.artifacts[0].path, "metadata.json")
            self.assertTrue(
                generated.artifacts[0].digest.startswith("sha256:")
            )
            self.assertEqual(
                task.artifacts[1].artifact_type,
                ContainerOutputContractType.TASK_ARTIFACT,
            )
            self.assertEqual(
                generated.to_execution_result().status,
                ContainerResultStatus.COMPLETED,
            )
            self.assertEqual(
                generated.artifacts[0].media_type,
                "application/json",
            )
            self.assertEqual(generated.to_dict()["decision"], "accept")

    def test_archive_accepts_safe_runtime_envelope_artifact(self) -> None:
        contract = _artifact_contract(
            ContainerOutputContractType.RUNTIME_ENVELOPE_ARTIFACT,
            max_bytes=128,
            max_files=2,
            allowed_uids=(1000,),
            allowed_gids=(1000,),
        )
        result = validate_archive_entries(
            (
                ContainerArchiveEntry(
                    path="logs",
                    entry_type=ContainerArchiveEntryType.DIRECTORY,
                    mode=0o700,
                    uid=1000,
                    gid=1000,
                ),
                ContainerArchiveEntry(
                    path="logs//result.txt",
                    entry_type="file",
                    size_bytes=12,
                    mode=0o600,
                    uid=1000,
                    gid=1000,
                    signature=b"result",
                ),
            ),
            contract,
        )

        self.assertEqual(result.decision, ContainerOutputDecisionType.ACCEPT)
        self.assertEqual(result.artifacts[0].path, "logs/result.txt")
        self.assertEqual(
            result.artifacts[0].artifact_type,
            ContainerOutputContractType.RUNTIME_ENVELOPE_ARTIFACT,
        )
        self.assertEqual(
            result.artifacts[0].to_dict()["quarantined"],
            False,
        )

    def test_partial_output_allow_deny_and_quarantine_modes(self) -> None:
        policy = ContainerPartialOutputPolicy(
            timeout=ContainerPartialOutputMode.ALLOW,
            cancellation=ContainerPartialOutputMode.DENY,
            runtime_failure=ContainerPartialOutputMode.QUARANTINE,
        )
        allowed = validate_output_stream(
            b"partial",
            ContainerOutputContract(
                contract_type=ContainerOutputContractType.STDOUT,
                max_bytes=64,
            ),
            partial_output=ContainerPartialOutput(reason="timeout"),
            partial_policy=policy,
        )
        denied = validate_output_stream(
            b"partial",
            ContainerOutputContract(
                contract_type=ContainerOutputContractType.STDOUT,
                max_bytes=64,
            ),
            partial_output=ContainerPartialOutput(
                reason=ContainerPartialOutputReason.CANCELLATION,
            ),
            partial_policy=policy,
        )
        with _workspace() as root:
            _write(root, "partial.txt", b"partial")
            copied_denied = validate_copied_outputs(
                str(root),
                _artifact_contract(
                    ContainerOutputContractType.TASK_ARTIFACT,
                    max_bytes=64,
                ),
                partial_output=ContainerPartialOutput(
                    reason=ContainerPartialOutputReason.CANCELLATION,
                ),
                partial_policy=policy,
            )
        archive_denied = validate_archive_entries(
            (
                ContainerArchiveEntry(
                    path="partial.txt",
                    entry_type="file",
                    size_bytes=7,
                    signature=b"partial",
                ),
            ),
            _artifact_contract(
                ContainerOutputContractType.TASK_ARTIFACT,
                max_bytes=64,
            ),
            partial_output=ContainerPartialOutput(
                reason=ContainerPartialOutputReason.CANCELLATION,
            ),
            partial_policy=policy,
        )

        self.assertEqual(allowed.decision, ContainerOutputDecisionType.ACCEPT)
        self.assertIsNotNone(allowed.partial_output)
        partial_output = cast(ContainerPartialOutput, allowed.partial_output)
        self.assertEqual(partial_output.to_dict()["reason"], "timeout")
        self.assertEqual(denied.decision, ContainerOutputDecisionType.REJECT)
        self.assertEqual(copied_denied.file_count, 0)
        self.assertEqual(archive_denied.file_count, 0)
        self.assertEqual(
            denied.diagnostics[0].code,
            ContainerOutputDiagnosticCode.PARTIAL_OUTPUT_DENIED,
        )
        self.assertEqual(policy.to_dict()["runtime_failure"], "quarantine")

    def test_copied_outputs_reject_symlink_special_limits_and_media(
        self,
    ) -> None:
        with _workspace() as root:
            outside = _write(root.parent, "outside.txt", b"outside")
            _write(root, "a-unsafe.bin", b"bin")
            _write(root, "b-program.txt", b"MZ")
            _write(root, "one.txt", b"1")
            _write(root, "two.txt", b"2")
            _write(root, "z-too-large.txt", b"12345")
            (root / "link.txt").symlink_to(outside)
            fifo = root / "pipe"
            mkfifo(fifo)
            unsafe_permissions = _write(root, "unsafe-permissions.txt", b"x")
            chmod(unsafe_permissions, 0o666)
            result = validate_copied_outputs(
                str(root),
                _artifact_contract(
                    ContainerOutputContractType.GENERATED_FILE,
                    max_bytes=64,
                    max_files=4,
                    per_file_bytes=4,
                ),
            )
            codes = _codes(result)

            self.assertEqual(
                result.decision,
                ContainerOutputDecisionType.REJECT,
            )
            self.assertIn(
                ContainerOutputDiagnosticCode.SYMLINK_ESCAPE,
                codes,
            )
            self.assertIn(ContainerOutputDiagnosticCode.SPECIAL_FILE, codes)
            self.assertIn(ContainerOutputDiagnosticCode.TOO_LARGE, codes)
            self.assertIn(ContainerOutputDiagnosticCode.TOO_MANY_FILES, codes)
            self.assertIn(ContainerOutputDiagnosticCode.UNSAFE_MEDIA, codes)
            self.assertIn(
                ContainerOutputDiagnosticCode.UNSAFE_SIGNATURE,
                codes,
            )
            self.assertIn(
                ContainerOutputDiagnosticCode.UNSAFE_PERMISSIONS,
                codes,
            )

    def test_copied_output_skips_unsafe_file_without_artifact_or_open(
        self,
    ) -> None:
        with _workspace() as root:
            unsafe_permissions = _write(root, "unsafe.txt", b"unsafe")
            chmod(unsafe_permissions, 0o666)
            with patch(
                "avalan.container.output.open_fd",
                side_effect=AssertionError("unsafe output opened"),
            ):
                result = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                    ),
                )

            self.assertEqual(
                result.decision,
                ContainerOutputDecisionType.REJECT,
            )
            self.assertEqual(result.artifacts, ())
            self.assertIn(
                ContainerOutputDiagnosticCode.UNSAFE_PERMISSIONS,
                _codes(result),
            )

    def test_copied_output_rejects_unsafe_ownership(self) -> None:
        with _workspace() as root:
            _write(root, "owned.txt", b"owned")
            result = validate_copied_outputs(
                str(root),
                _artifact_contract(
                    ContainerOutputContractType.TASK_ARTIFACT,
                    max_bytes=64,
                    allowed_uids=(999999,),
                ),
            )

            self.assertEqual(result.artifacts, ())
            self.assertIn(
                ContainerOutputDiagnosticCode.UNSAFE_OWNERSHIP,
                _codes(result),
            )

    def test_copied_output_rejects_race_after_lstat(self) -> None:
        with _workspace() as root:
            artifact = _write(root, "artifact.txt", b"artifact")
            swapped = _write(root.parent, "swapped.txt", b"swapped")

            def open_swapped(path: str, flags: int) -> int:
                return os_open(str(swapped), flags)

            def changed_stat(file_descriptor: int) -> stat_result:
                return _stat_with(artifact.lstat(), size=file_descriptor + 9)

            with patch(
                "avalan.container.output.open_fd",
                side_effect=open_swapped,
            ):
                result = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                    ),
                )

            self.assertEqual(
                result.decision,
                ContainerOutputDecisionType.REJECT,
            )
            self.assertEqual(result.artifacts, ())
            self.assertEqual(
                result.diagnostics[0].code,
                ContainerOutputDiagnosticCode.RACE_DETECTED,
            )
            with patch(
                "avalan.container.output.open_fd",
                side_effect=OSError("changed"),
            ):
                open_error = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                    ),
                )
            with patch(
                "avalan.container.output.fstat",
                side_effect=(artifact.lstat(), changed_stat(0)),
            ):
                changed_after_read = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                    ),
                )
            with patch(
                "avalan.container.output.fstat",
                return_value=_stat_with(
                    artifact.lstat(),
                    mode=artifact.lstat().st_mode | 0o022,
                ),
            ):
                changed_mode = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                    ),
                )
            with patch(
                "avalan.container.output.fstat",
                return_value=_stat_with(
                    artifact.lstat(),
                    uid=artifact.lstat().st_uid + 1,
                ),
            ):
                changed_owner = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                    ),
                )

            self.assertEqual(
                open_error.diagnostics[0].code,
                ContainerOutputDiagnosticCode.RACE_DETECTED,
            )
            self.assertEqual(
                changed_after_read.diagnostics[0].code,
                ContainerOutputDiagnosticCode.RACE_DETECTED,
            )
            self.assertEqual(
                changed_mode.diagnostics[0].code,
                ContainerOutputDiagnosticCode.RACE_DETECTED,
            )
            self.assertEqual(
                changed_owner.diagnostics[0].code,
                ContainerOutputDiagnosticCode.RACE_DETECTED,
            )

    def test_copied_output_rejects_short_or_growing_fd_read(self) -> None:
        with _workspace() as root:
            _write(root, "artifact.txt", b"artifact")
            with patch(
                "avalan.container.output.read_fd",
                side_effect=(b"art", b""),
            ):
                short_read = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                    ),
                )
            with patch(
                "avalan.container.output.read_fd",
                side_effect=(b"artifact", b"x"),
            ):
                growing_read = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                    ),
                )

            self.assertEqual(
                short_read.diagnostics[0].code,
                ContainerOutputDiagnosticCode.RACE_DETECTED,
            )
            self.assertEqual(
                growing_read.diagnostics[0].code,
                ContainerOutputDiagnosticCode.RACE_DETECTED,
            )

    def test_copied_output_rejects_oversized_files_before_open(self) -> None:
        with _workspace() as root:
            _write(root, "oversized.txt", b"12345")
            with patch(
                "avalan.container.output.open_fd",
                side_effect=AssertionError("oversized output opened"),
            ):
                result = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=64,
                        per_file_bytes=4,
                    ),
                )

            self.assertEqual(
                result.decision,
                ContainerOutputDecisionType.REJECT,
            )
            self.assertEqual(result.artifacts, ())
            self.assertIn(
                ContainerOutputDiagnosticCode.TOO_LARGE,
                _codes(result),
            )

    def test_copied_output_rejects_total_limits_before_second_open(
        self,
    ) -> None:
        with _workspace() as root:
            _write(root, "one.txt", b"123")
            _write(root, "two.txt", b"456")
            opened_paths: list[str] = []

            def limited_open(path: str, flags: int) -> int:
                opened_paths.append(Path(path).name)
                if Path(path).name == "two.txt":
                    raise AssertionError("over-limit output opened")
                return os_open(path, flags)

            with patch(
                "avalan.container.output.open_fd",
                side_effect=limited_open,
            ):
                result = validate_copied_outputs(
                    str(root),
                    _artifact_contract(
                        ContainerOutputContractType.TASK_ARTIFACT,
                        max_bytes=4,
                        max_files=1,
                    ),
                )

            self.assertEqual(opened_paths, ["one.txt"])
            self.assertEqual(
                result.decision,
                ContainerOutputDecisionType.REJECT,
            )
            self.assertIn(
                ContainerOutputDiagnosticCode.TOO_MANY_FILES,
                _codes(result),
            )
            self.assertIn(
                ContainerOutputDiagnosticCode.TOO_LARGE,
                _codes(result),
            )

    def test_archive_rejects_unsafe_entries(self) -> None:
        result = validate_archive_entries(
            (
                ContainerArchiveEntry(
                    path="/absolute.txt",
                    entry_type="file",
                    size_bytes=1,
                    signature=b"a",
                ),
                ContainerArchiveEntry(
                    path="../escape.txt",
                    entry_type="file",
                    size_bytes=1,
                    signature=b"a",
                ),
                ContainerArchiveEntry(
                    path="link.txt",
                    entry_type=ContainerArchiveEntryType.SYMLINK,
                ),
                ContainerArchiveEntry(
                    path="hard.txt",
                    entry_type=ContainerArchiveEntryType.HARDLINK,
                ),
                ContainerArchiveEntry(
                    path="device",
                    entry_type=ContainerArchiveEntryType.CHARACTER_DEVICE,
                ),
                ContainerArchiveEntry(
                    path="fifo",
                    entry_type=ContainerArchiveEntryType.FIFO,
                ),
                ContainerArchiveEntry(
                    path="socket",
                    entry_type=ContainerArchiveEntryType.SOCKET,
                ),
                ContainerArchiveEntry(
                    path="block",
                    entry_type=ContainerArchiveEntryType.BLOCK_DEVICE,
                ),
                ContainerArchiveEntry(
                    path="Report.txt",
                    entry_type="file",
                    size_bytes=1,
                    mode=0o777,
                    uid=0,
                    gid=0,
                    signature=b"a",
                ),
                ContainerArchiveEntry(
                    path="report.TXT",
                    entry_type="file",
                    size_bytes=10,
                    media_type="application/x-msdownload",
                    signature=b"\x7fELF",
                ),
                ContainerArchiveEntry(
                    path="./REPORT.txt",
                    entry_type="file",
                    size_bytes=1,
                    signature=b"a",
                ),
                ContainerArchiveEntry(
                    path="dir/file.txt",
                    entry_type="file",
                    size_bytes=1,
                    signature=b"a",
                ),
                ContainerArchiveEntry(
                    path="dir//FILE.txt",
                    entry_type="file",
                    size_bytes=1,
                    signature=b"a",
                ),
            ),
            _artifact_contract(
                ContainerOutputContractType.TASK_ARTIFACT,
                max_bytes=8,
                max_files=2,
                per_file_bytes=4,
                allowed_uids=(1000,),
                allowed_gids=(1000,),
            ),
        )
        codes = _codes(result)

        self.assertEqual(result.decision, ContainerOutputDecisionType.REJECT)
        self.assertIn(ContainerOutputDiagnosticCode.ABSOLUTE_PATH, codes)
        self.assertIn(ContainerOutputDiagnosticCode.TRAVERSAL, codes)
        self.assertIn(ContainerOutputDiagnosticCode.SYMLINK_ESCAPE, codes)
        self.assertIn(ContainerOutputDiagnosticCode.HARDLINK, codes)
        self.assertIn(ContainerOutputDiagnosticCode.SPECIAL_FILE, codes)
        self.assertIn(ContainerOutputDiagnosticCode.CASE_COLLISION, codes)
        self.assertIn(ContainerOutputDiagnosticCode.UNSAFE_OWNERSHIP, codes)
        self.assertIn(
            ContainerOutputDiagnosticCode.UNSAFE_PERMISSIONS,
            codes,
        )
        self.assertIn(ContainerOutputDiagnosticCode.TOO_LARGE, codes)
        self.assertIn(ContainerOutputDiagnosticCode.TOO_MANY_FILES, codes)
        self.assertIn(ContainerOutputDiagnosticCode.UNSAFE_MEDIA, codes)
        self.assertIn(ContainerOutputDiagnosticCode.UNSAFE_SIGNATURE, codes)
        self.assertEqual(
            ContainerArchiveEntry(
                path="sample.txt",
                entry_type="file",
            ).to_dict()["entry_type"],
            "file",
        )

    def test_archive_case_collisions_suppress_colliding_artifacts(
        self,
    ) -> None:
        result = validate_archive_entries(
            (
                ContainerArchiveEntry(
                    path="report.txt",
                    entry_type="file",
                    size_bytes=6,
                    signature=b"report",
                ),
                ContainerArchiveEntry(
                    path="./REPORT.txt",
                    entry_type="file",
                    size_bytes=6,
                    signature=b"report",
                ),
                ContainerArchiveEntry(
                    path="summary.txt",
                    entry_type="file",
                    size_bytes=7,
                    signature=b"summary",
                ),
            ),
            _artifact_contract(
                ContainerOutputContractType.TASK_ARTIFACT,
                max_bytes=64,
                max_files=3,
            ),
        )

        self.assertEqual(result.decision, ContainerOutputDecisionType.REJECT)
        self.assertEqual(
            [artifact.path for artifact in result.artifacts],
            ["summary.txt"],
        )
        self.assertIn(
            ContainerOutputDiagnosticCode.CASE_COLLISION,
            _codes(result),
        )

    def test_disabled_artifact_contract_rejects_copy(self) -> None:
        with _workspace() as root:
            _write(root, "artifact.txt", b"artifact")
            disabled_contract = ContainerOutputContract(
                contract_type=ContainerOutputContractType.GENERATED_FILE,
                max_bytes=64,
                enabled=False,
            )
            result = validate_copied_outputs(
                str(root),
                disabled_contract,
            )
            archive = validate_archive_entries(
                (
                    ContainerArchiveEntry(
                        path="artifact.txt",
                        entry_type="file",
                        size_bytes=8,
                        signature=b"artifact",
                    ),
                ),
                disabled_contract,
            )

            self.assertEqual(
                result.decision,
                ContainerOutputDecisionType.REJECT,
            )
            self.assertEqual(
                archive.decision,
                ContainerOutputDecisionType.REJECT,
            )
            self.assertEqual(
                result.diagnostics[0].code,
                ContainerOutputDiagnosticCode.CONTRACT_DISABLED,
            )
            self.assertEqual(result.artifacts, ())
            self.assertEqual(result.file_count, 0)
            self.assertEqual(archive.artifacts, ())
            self.assertEqual(archive.file_count, 0)

            with patch(
                "avalan.container.output.open_fd",
                side_effect=AssertionError("disabled contract read"),
            ):
                self.assertEqual(
                    validate_copied_outputs(
                        str(root),
                        disabled_contract,
                    ).decision,
                    ContainerOutputDecisionType.REJECT,
                )

    def test_fake_e2e_output_copy_rejection_quarantine_result_and_audit(
        self,
    ) -> None:
        with _workspace() as root:
            _write(root, "artifact.txt", b"artifact")
            accepted = validate_copied_outputs(
                str(root),
                _artifact_contract(
                    ContainerOutputContractType.TASK_ARTIFACT,
                    max_bytes=64,
                ),
            )
            rejected = validate_archive_entries(
                (
                    ContainerArchiveEntry(
                        path="../escape.txt",
                        entry_type="file",
                        size_bytes=1,
                    ),
                ),
                _artifact_contract(
                    ContainerOutputContractType.TASK_ARTIFACT,
                    max_bytes=64,
                ),
            )
            quarantined = validate_copied_outputs(
                str(root),
                _artifact_contract(
                    ContainerOutputContractType.TASK_ARTIFACT,
                    max_bytes=64,
                ),
                partial_output=ContainerPartialOutput(
                    reason=ContainerPartialOutputReason.RUNTIME_FAILURE,
                ),
            )

            self.assertEqual(
                accepted.to_execution_result().status,
                ContainerResultStatus.COMPLETED,
            )
            self.assertEqual(
                rejected.to_execution_result().status,
                ContainerResultStatus.DENIED,
            )
            self.assertEqual(
                quarantined.to_execution_result().status,
                ContainerResultStatus.FAILED,
            )
            self.assertTrue(quarantined.artifacts[0].quarantined)
            self.assertEqual(
                rejected.audit_events(
                    scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                    profile_name="phase6",
                    policy_version="phase6",
                )[1].event_type,
                ContainerAuditEventType.DENIAL,
            )
            self.assertEqual(
                quarantined.audit_events(
                    scope="shell_container_execution",
                    profile_name="phase6",
                    policy_version="phase6",
                )[1].event_type,
                ContainerAuditEventType.FAILURE,
            )
            self.assertEqual(
                accepted.audit_events(
                    scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
                    profile_name="phase6",
                    policy_version="phase6",
                )[0].event_type,
                ContainerAuditEventType.OUTPUT_COPY,
            )

    def test_value_objects_reject_invalid_values(self) -> None:
        with self.assertRaises(AssertionError):
            ContainerOutputContract(contract_type="stdout", max_bytes=0)
        with self.assertRaises(AssertionError):
            ContainerOutputMediaPolicy(denied_signatures=(b"",))
        with self.assertRaises(AssertionError):
            ContainerArchiveEntry(path="bad", entry_type="file", mode=0o10000)
        with self.assertRaises(AssertionError):
            ContainerArchiveEntry(path="bad", entry_type="file", uid=-1)
        with self.assertRaises(AssertionError):
            ContainerOutputArtifact(
                artifact_type="stdout",
                path="stream",
                size_bytes=1,
                media_type="text/plain",
                digest=_digest("stream"),
            )
        with self.assertRaises(AssertionError):
            ContainerOutputArtifact(
                artifact_type="generated_file",
                path="../escape",
                size_bytes=1,
                media_type="text/plain",
                digest=_digest("escape"),
            )
        diagnostic = ContainerOutputDiagnostic(
            code="container.output.too_large",
            path="artifact.txt",
            message="too large",
        )
        self.assertEqual(
            diagnostic.to_dict()["code"],
            "container.output.too_large",
        )


@contextmanager
def _workspace() -> Iterator[Path]:
    with TemporaryDirectory(dir=str(Path(gettempdir()).resolve())) as root:
        yield Path(root)


def _artifact_contract(
    contract_type: ContainerOutputContractType,
    *,
    max_bytes: int,
    max_files: int = 8,
    per_file_bytes: int | None = None,
    allowed_uids: tuple[int, ...] | None = None,
    allowed_gids: tuple[int, ...] | None = None,
) -> ContainerOutputContract:
    return ContainerOutputContract(
        contract_type=contract_type,
        max_bytes=max_bytes,
        max_files=max_files,
        per_file_bytes=per_file_bytes,
        media_policy=ContainerOutputMediaPolicy(
            allowed_media_types=(
                "application/json",
                "text/plain",
            ),
        ),
        allowed_uids=allowed_uids,
        allowed_gids=allowed_gids,
    )


def _write(root: Path, relative: str, content: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _stat_with(
    source: stat_result,
    *,
    mode: int | None = None,
    uid: int | None = None,
    gid: int | None = None,
    size: int | None = None,
) -> stat_result:
    return stat_result(
        (
            source.st_mode if mode is None else mode,
            source.st_ino,
            source.st_dev,
            source.st_nlink,
            source.st_uid if uid is None else uid,
            source.st_gid if gid is None else gid,
            source.st_size if size is None else size,
            source.st_atime,
            source.st_mtime,
            source.st_ctime,
        )
    )


def _codes(
    result: ContainerOutputValidationResult,
) -> set[ContainerOutputDiagnosticCode]:
    return {
        cast(ContainerOutputDiagnosticCode, diagnostic.code)
        for diagnostic in result.diagnostics
    }


def _digest(value: str) -> str:
    return (
        "sha256:0" * (64 - len(value.encode("utf-8").hex()))
        + value.encode("utf-8").hex()
    )


if __name__ == "__main__":
    main()
