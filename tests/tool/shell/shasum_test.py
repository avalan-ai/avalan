from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.tool.shell import (
    ExecutionPolicy,
    PathOperand,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellCommandStepRequest,
    ShellCompositionRequest,
    ShellExecutionErrorCode,
    ShellPolicyDenied,
    ShellToolSettings,
)
from avalan.tool.shell.entities import ShellPathKind


class ShasumPolicyTest(IsolatedAsyncioTestCase):
    async def test_builds_only_allowlisted_algorithm_argv(self) -> None:
        algorithms = ("1", "224", "256", "384", "512", "512224", "512256")
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "-first.bin").write_bytes(b"\x00first")
            (root / "second file.txt").write_text(
                "second\n",
                encoding="utf-8",
            )
            for algorithm in algorithms:
                with self.subTest(algorithm=algorithm):
                    resolver = _RecordingResolver()
                    policy = ExecutionPolicy(
                        settings=ShellToolSettings(workspace_root=str(root)),
                        resolver=resolver,
                    )

                    spec = await policy.normalize(
                        _request(
                            {"algorithm": algorithm},
                            paths=("-first.bin", "second file.txt"),
                        )
                    )

                    self.assertEqual(
                        spec.argv,
                        (
                            "shasum",
                            "-a",
                            algorithm,
                            "--",
                            "./-first.bin",
                            "second file.txt",
                        ),
                    )
                    self.assertEqual(spec.display_argv, spec.argv)
                    self.assertEqual(
                        spec.executable,
                        "/trusted/bin/shasum",
                    )
                    self.assertEqual(resolver.commands, ["shasum"])

    async def test_default_algorithm_preserves_native_sha1(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.bin").write_bytes(b"\x00visible")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
                resolver=_RecordingResolver(),
            )

            spec = await policy.normalize(_request({}, paths=("visible.bin",)))

        self.assertEqual(
            spec.argv,
            ("shasum", "-a", "1", "--", "visible.bin"),
        )
        self.assertEqual(spec.stdout_media_type, "text/plain")

    async def test_nested_cwd_paths_remain_cwd_relative_and_option_safe(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "-first.bin").write_bytes(b"\x00first")
            (nested / "second file.bin").write_bytes(b"\x00second")
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
                resolver=_RecordingResolver(),
            )

            spec = await policy.normalize(
                _request(
                    {"algorithm": "256"},
                    paths=("-first.bin", "second file.bin"),
                    cwd="nested",
                )
            )

        self.assertEqual(spec.cwd, str(nested.resolve()))
        self.assertEqual(spec.display_cwd, "nested")
        self.assertEqual(
            spec.argv,
            (
                "shasum",
                "-a",
                "256",
                "--",
                "./-first.bin",
                "second file.bin",
            ),
        )
        self.assertEqual(spec.display_argv, spec.argv)
        self.assertEqual(spec.runtime_dependencies, ())

    async def test_rejects_unstructured_modes_and_algorithms_before_resolve(
        self,
    ) -> None:
        cases = (
            ({"algorithm": "sha256"}, "algorithm is unsupported"),
            ({"algorithm": 256}, "algorithm is unsupported"),
            ({"algorithm": True}, "algorithm is unsupported"),
            ({"check": True}, "unknown shasum option: check"),
            ({"binary": True}, "unknown shasum option: binary"),
            ({"argv": ["-c"]}, "unknown shasum option: argv"),
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.bin").write_bytes(b"visible")
            for options, message in cases:
                with self.subTest(options=options):
                    resolver = _RecordingResolver()
                    policy = ExecutionPolicy(
                        settings=ShellToolSettings(workspace_root=str(root)),
                        resolver=resolver,
                    )

                    with self.assertRaisesRegex(
                        ShellPolicyDenied,
                        message,
                    ) as raised:
                        await policy.normalize(
                            _request(options, paths=("visible.bin",))
                        )

                    self.assertIs(
                        raised.exception.error_code,
                        ShellExecutionErrorCode.INVALID_OPTION,
                    )
                    self.assertEqual(resolver.commands, [])

    async def test_rejects_missing_paths_and_wrong_path_kinds(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("visible\n", encoding="utf-8")
            cases = (
                (_request({"algorithm": "256"}), "requires at least one path"),
                (
                    _request(
                        {"algorithm": "256"},
                        path_kind="text_file",
                        paths=("visible.txt",),
                    ),
                    "unsupported shasum path kind",
                ),
            )
            for request, message in cases:
                with self.subTest(message=message):
                    resolver = _RecordingResolver()
                    policy = ExecutionPolicy(
                        settings=ShellToolSettings(workspace_root=str(root)),
                        resolver=resolver,
                    )

                    with self.assertRaisesRegex(
                        ShellPolicyDenied,
                        message,
                    ) as raised:
                        await policy.normalize(request)

                    self.assertIs(
                        raised.exception.error_code,
                        ShellExecutionErrorCode.INVALID_OPTION,
                    )
                    self.assertEqual(resolver.commands, [])

    async def test_rejects_directories_and_sensitive_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "directory").mkdir()
            (root / "credentials").write_text("secret\n", encoding="utf-8")
            cases = (
                (
                    "directory",
                    "input path must be a regular file",
                    ShellExecutionErrorCode.DENIED_PATH,
                ),
                (
                    "credentials",
                    "path is denied",
                    ShellExecutionErrorCode.SENSITIVE_PATH,
                ),
            )
            for path, message, error_code in cases:
                with self.subTest(path=path):
                    resolver = _RecordingResolver()
                    policy = ExecutionPolicy(
                        settings=ShellToolSettings(workspace_root=str(root)),
                        resolver=resolver,
                    )

                    with self.assertRaisesRegex(
                        ShellPolicyDenied,
                        message,
                    ) as raised:
                        await policy.normalize(
                            _request({"algorithm": "256"}, paths=(path,))
                        )

                    self.assertIs(
                        raised.exception.error_code,
                        error_code,
                    )
                    self.assertEqual(resolver.commands, [])

    async def test_rejects_stdin_before_resolve(self) -> None:
        resolver = _RecordingResolver()
        policy = ExecutionPolicy(
            settings=ShellToolSettings(),
            resolver=resolver,
        )

        with self.assertRaisesRegex(
            ShellPolicyDenied,
            "stdin is disabled",
        ) as raised:
            await policy.normalize(
                _request(
                    {"algorithm": "256"},
                    paths=("pyproject.toml",),
                    stdin=b"forged",
                )
            )

        self.assertIs(
            raised.exception.error_code,
            ShellExecutionErrorCode.STDIN_DENIED,
        )
        self.assertEqual(resolver.commands, [])

    async def test_structured_composition_hashes_explicit_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.bin").write_bytes(b"\x00visible")
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_pipelines=True,
            )
            policy = ExecutionPolicy(
                settings=settings,
                resolver=_RecordingResolver(),
            )

            spec = await policy.normalize_composition(
                ShellCompositionRequest(
                    mode="serial",
                    steps=(
                        ShellCommandStepRequest(
                            id="hash",
                            command="shasum",
                            options={"algorithm": "512"},
                            paths=("visible.bin",),
                        ),
                    ),
                )
            )

        self.assertEqual(
            spec.steps[0].spec.argv,
            ("shasum", "-a", "512", "--", "visible.bin"),
        )


def _request(
    options: dict[str, object],
    *,
    paths: tuple[str, ...] = (),
    path_kind: ShellPathKind = "file",
    stdin: bytes | None = None,
    cwd: str | None = None,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.shasum",
        command="shasum",
        options=options,
        paths=tuple(
            PathOperand(
                name=f"path_{index}",
                path=path,
                kind=path_kind,
                access="read",
            )
            for index, path in enumerate(paths)
        ),
        cwd=cwd,
        stdin=stdin,
    )


class _RecordingResolver:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        self.commands.append(command.logical_id)
        return "/trusted/bin/shasum"


if __name__ == "__main__":
    main()
