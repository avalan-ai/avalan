from pathlib import Path
from unittest import IsolatedAsyncioTestCase, main

from avalan.tool.shell.entities import (
    ExecutionSpec,
    ShellCommandStepRequest,
    ShellCompositionRequest,
    ShellExecutionErrorCode,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellStreamRef,
)
from avalan.tool.shell.policy import ExecutionPolicy
from avalan.tool.shell.registry import ShellCommandDefinition
from avalan.tool.shell.settings import ShellToolSettings

_FIXTURE_ROOT = Path("tests/tool/shell/fixtures")


class ExecutionCompositionPolicyTest(IsolatedAsyncioTestCase):
    async def test_pipeline_normalizes_cat_sed_wc_and_composition_caps(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        executor = _RecordingExecutor()
        settings = _settings(
            default_timeout_seconds=2.0,
            max_timeout_seconds=5.0,
            max_stdout_bytes=10,
            max_stderr_bytes=20,
            max_pipeline_bytes=40,
            max_intermediate_bytes=30,
        )
        policy = ExecutionPolicy(settings=settings, resolver=resolver)

        spec = await policy.normalize_composition(
            ShellCompositionRequest(
                steps=(
                    _step(
                        "read",
                        "cat",
                        paths=("filesystem/visible.txt",),
                    ),
                    _step(
                        "select",
                        "sed",
                        options={"line_ranges": ("1",)},
                    ),
                    _step(
                        "count",
                        "wc",
                        options={
                            "lines": True,
                            "words": False,
                            "count_bytes": False,
                        },
                    ),
                ),
                timeout_seconds=99.0,
                max_stdout_bytes=99,
                max_stderr_bytes=99,
                max_intermediate_bytes=99,
            )
        )

        self.assertEqual(spec.mode, "pipeline")
        self.assertEqual(spec.timeout_seconds, 5.0)
        self.assertEqual(spec.max_stdout_bytes, 40)
        self.assertEqual(spec.max_stderr_bytes, 20)
        self.assertEqual(spec.max_intermediate_bytes, 30)
        self.assertEqual(resolver.calls, ("cat", "sed", "wc"))
        self.assertEqual(executor.calls, ())
        self.assertEqual(
            tuple(step.id for step in spec.steps),
            ("read", "select", "count"),
        )
        self.assertIsNone(spec.steps[0].stdin_from)
        self.assertEqual(
            spec.steps[1].stdin_from,
            ShellStreamRef(step_id="read", stream="stdout"),
        )
        self.assertEqual(
            spec.steps[2].stdin_from,
            ShellStreamRef(step_id="select", stream="stdout"),
        )
        self.assertEqual(spec.steps[0].spec.stdout_media_type, "text/plain")
        self.assertEqual(spec.steps[0].spec.output_kind, ShellOutputKind.TEXT)
        self.assertEqual(spec.steps[0].spec.max_stdout_bytes, 10)
        self.assertEqual(spec.steps[1].spec.argv, ("sed", "-n", "-e", "1p"))
        self.assertEqual(spec.steps[2].spec.argv, ("wc", "-l"))

    async def test_pipeline_accepts_declared_linear_ref_for_rg_to_wc(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)

        spec = await policy.normalize_composition(
            ShellCompositionRequest(
                steps=(
                    _step(
                        "search",
                        "rg",
                        options={"pattern": "visible"},
                        paths=("filesystem",),
                    ),
                    _step(
                        "count",
                        "wc",
                        stdin_from=ShellStreamRef(
                            step_id="search",
                            stream="stdout",
                        ),
                    ),
                )
            )
        )

        self.assertEqual(spec.mode, "pipeline")
        self.assertEqual(resolver.calls, ("rg", "wc"))
        self.assertEqual(
            spec.steps[1].stdin_from,
            ShellStreamRef(step_id="search", stream="stdout"),
        )
        self.assertEqual(spec.steps[0].spec.command, "rg")
        self.assertEqual(spec.steps[1].spec.argv, ("wc", "-l"))

    async def test_serial_routes_declared_cat_json_to_jq(self) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)

        spec = await policy.normalize_composition(
            ShellCompositionRequest(
                mode="serial",
                steps=(
                    _step("read", "cat", paths=("json/valid.json",)),
                    _step(
                        "filter",
                        "jq",
                        options={"filter": "."},
                        stdin_from=ShellStreamRef(
                            step_id="read",
                            stream="stdout",
                        ),
                    ),
                ),
            )
        )

        self.assertEqual(spec.mode, "serial")
        self.assertEqual(resolver.calls, ("cat", "jq"))
        self.assertIsNone(spec.steps[0].stdin_from)
        self.assertEqual(
            spec.steps[1].stdin_from,
            ShellStreamRef(step_id="read", stream="stdout"),
        )
        self.assertEqual(
            spec.steps[0].spec.stdout_media_type, "application/json"
        )
        self.assertEqual(spec.steps[0].spec.output_kind, ShellOutputKind.JSON)
        self.assertEqual(spec.steps[1].spec.argv, ("jq", "--", "."))

    async def test_parallel_normalizes_independent_steps(self) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)

        spec = await policy.normalize_composition(
            ShellCompositionRequest(
                mode="parallel",
                steps=(
                    _step("read", "cat", paths=("filesystem/visible.txt",)),
                    _step(
                        "search",
                        "rg",
                        options={"pattern": "visible"},
                        paths=("filesystem",),
                    ),
                ),
            )
        )

        self.assertEqual(spec.mode, "parallel")
        self.assertEqual(resolver.calls, ("cat", "rg"))
        self.assertEqual(
            tuple(step.stdin_from for step in spec.steps), (None, None)
        )

    async def test_composition_preserves_non_local_backend_metadata(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(
            settings=_settings(execution_mode="sandbox"),
            resolver=resolver,
        )

        spec = await policy.normalize_composition(
            ShellCompositionRequest(
                steps=(
                    _step(
                        "search",
                        "rg",
                        options={"pattern": "visible"},
                        paths=("filesystem",),
                    ),
                )
            )
        )

        self.assertEqual(spec.steps[0].spec.backend, "sandbox")
        self.assertNotIn(
            "local_host_approval",
            spec.steps[0].spec.metadata,
        )

    async def test_denies_disabled_pipeline_before_resolver(self) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(resolver=resolver)

        await self._assert_denied(
            ShellCompositionRequest(
                steps=(
                    _step("read", "cat", paths=("filesystem/visible.txt",)),
                )
            ),
            ShellExecutionErrorCode.POLICY_DENIED,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ())

    async def test_denies_invalid_mode_empty_steps_duplicates_and_stage_cap(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(
            settings=_settings(max_pipeline_stages=1),
            resolver=resolver,
        )
        read = _step("read", "cat", paths=("filesystem/visible.txt",))
        count = _step("count", "wc")
        duplicate = _step("read", "rg", options={"pattern": "visible"})
        invalid_mode = ShellCompositionRequest(steps=(read,))
        empty_steps = ShellCompositionRequest(steps=(read,))
        duplicate_steps = ShellCompositionRequest(steps=(read,))
        too_many_steps = ShellCompositionRequest(steps=(read, count))
        object.__setattr__(invalid_mode, "mode", "batch")
        object.__setattr__(empty_steps, "steps", ())
        object.__setattr__(duplicate_steps, "steps", (read, duplicate))

        for request in (
            invalid_mode,
            empty_steps,
            duplicate_steps,
            too_many_steps,
        ):
            with self.subTest(request=request):
                await self._assert_denied(
                    request,
                    ShellExecutionErrorCode.INVALID_OPTION,
                    policy=policy,
                )

        self.assertEqual(resolver.calls, ())

    async def test_denies_invalid_refs_before_resolver(self) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)
        read = _step("read", "cat", paths=("filesystem/visible.txt",))
        select = _step(
            "select",
            "sed",
            options={"line_ranges": ("1",)},
            stdin_from=ShellStreamRef(step_id="read", stream="stdout"),
        )
        count = _step(
            "count",
            "wc",
            stdin_from=ShellStreamRef(step_id="read", stream="stdout"),
        )
        pipeline_first_ref = ShellCompositionRequest(
            steps=(
                _step(
                    "read",
                    "cat",
                    paths=("filesystem/visible.txt",),
                    stdin_from=ShellStreamRef(
                        step_id="count",
                        stream="stdout",
                    ),
                ),
                _step("count", "wc"),
            )
        )
        pipeline_wrong_ref = ShellCompositionRequest(
            steps=(read, select, count)
        )
        serial_later_ref = ShellCompositionRequest(
            mode="serial",
            steps=(
                _step(
                    "read",
                    "cat",
                    stdin_from=ShellStreamRef(
                        step_id="count",
                        stream="stdout",
                    ),
                ),
                _step("count", "wc"),
            ),
        )
        serial_unknown_ref = ShellCompositionRequest(
            mode="serial",
            steps=(read, _step("count", "wc")),
        )
        object.__setattr__(
            serial_unknown_ref.steps[1],
            "stdin_from",
            ShellStreamRef(step_id="missing", stream="stdout"),
        )
        parallel_ref = ShellCompositionRequest(
            mode="parallel",
            steps=(read, count),
        )

        for request in (
            pipeline_first_ref,
            pipeline_wrong_ref,
            serial_later_ref,
            serial_unknown_ref,
            parallel_ref,
        ):
            with self.subTest(request=request):
                await self._assert_denied(
                    request,
                    ShellExecutionErrorCode.INVALID_OPTION,
                    policy=policy,
                )

        self.assertEqual(resolver.calls, ())

    async def test_denies_unknown_command_before_resolver(self) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)

        await self._assert_denied(
            ShellCompositionRequest(
                steps=(_step("unknown", "wat", paths=("input.txt",)),)
            ),
            ShellExecutionErrorCode.DENIED_COMMAND,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ())

    async def test_denies_disallowed_known_command_before_resolver(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(
            settings=_settings(allowed_commands=("rg",)),
            resolver=resolver,
        )

        await self._assert_denied(
            ShellCompositionRequest(
                steps=(
                    _step("read", "cat", paths=("filesystem/visible.txt",)),
                )
            ),
            ShellExecutionErrorCode.DENIED_COMMAND,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ())

    async def test_denies_corrupted_serial_stream_ref_before_resolver(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)
        stdin_from = ShellStreamRef(step_id="read", stream="stdout")
        object.__setattr__(stdin_from, "stream", "stderr")

        await self._assert_denied(
            ShellCompositionRequest(
                mode="serial",
                steps=(
                    _step("read", "cat", paths=("filesystem/visible.txt",)),
                    _step("count", "wc", stdin_from=stdin_from),
                ),
            ),
            ShellExecutionErrorCode.INVALID_OPTION,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ())

    async def test_denies_step_cwd_policy_violation_before_resolver(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)

        await self._assert_denied(
            ShellCompositionRequest(
                steps=(
                    _step(
                        "search",
                        "rg",
                        options={"pattern": "visible"},
                        cwd="../outside",
                    ),
                )
            ),
            ShellExecutionErrorCode.TRAVERSAL,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ())

    async def test_denies_unsupported_consumer_without_later_resolution(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)

        await self._assert_denied(
            ShellCompositionRequest(
                steps=(
                    _step("read", "cat", paths=("filesystem/visible.txt",)),
                    _step("again", "cat"),
                    _step("count", "wc"),
                )
            ),
            ShellExecutionErrorCode.DENIED_COMMAND,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ("cat",))

    async def test_denies_media_mismatch_without_consumer_resolution(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(settings=_settings(), resolver=resolver)

        await self._assert_denied(
            ShellCompositionRequest(
                steps=(
                    _step("read", "cat", paths=("filesystem/visible.txt",)),
                    _step("filter", "jq", options={"filter": "."}),
                )
            ),
            ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ("cat",))

    async def test_denies_path_policy_violations_before_resolver(self) -> None:
        cases = (
            (
                _step("traverse", "cat", paths=("../secret.txt",)),
                ShellExecutionErrorCode.TRAVERSAL,
            ),
            (
                _step("hidden", "cat", paths=(".hidden.txt",)),
                ShellExecutionErrorCode.HIDDEN_PATH,
            ),
            (
                _step("sensitive", "cat", paths=("credentials",)),
                ShellExecutionErrorCode.SENSITIVE_PATH,
            ),
        )

        for step, error_code in cases:
            with self.subTest(error_code=error_code):
                resolver = _CountingResolver("/usr/bin/tool")
                policy = ExecutionPolicy(
                    settings=_settings(),
                    resolver=resolver,
                )
                await self._assert_denied(
                    ShellCompositionRequest(steps=(step,)),
                    error_code,
                    policy=policy,
                )
                self.assertEqual(resolver.calls, ())

    async def test_denies_shell_evaluation_and_budgets_before_resolver(
        self,
    ) -> None:
        cases = (
            (
                ShellCompositionRequest(
                    steps=(
                        _step(
                            "search",
                            "rg",
                            options={"pattern": "visible", "shell": True},
                        ),
                    )
                ),
                _settings(),
                ShellExecutionErrorCode.SHELL_DENIED,
            ),
            (
                ShellCompositionRequest(
                    steps=(
                        _step(
                            "search",
                            "rg",
                            options={"pattern": "visible"},
                        ),
                    ),
                    max_stdout_bytes=0,
                ),
                _settings(),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                ShellCompositionRequest(
                    steps=(
                        _step(
                            "search",
                            "rg",
                            options={"pattern": "visible"},
                        ),
                    ),
                    max_intermediate_bytes=0,
                ),
                _settings(),
                ShellExecutionErrorCode.INVALID_OPTION,
            ),
            (
                ShellCompositionRequest(
                    steps=(
                        _step(
                            "search",
                            "rg",
                            options={"pattern": "visible"},
                        ),
                    )
                ),
                _settings(max_argument_bytes=4),
                ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            ),
        )

        for request, settings, error_code in cases:
            with self.subTest(error_code=error_code):
                resolver = _CountingResolver("/usr/bin/tool")
                policy = ExecutionPolicy(settings=settings, resolver=resolver)
                await self._assert_denied(
                    request,
                    error_code,
                    policy=policy,
                )
                self.assertEqual(resolver.calls, ())

    async def test_denies_generated_output_command_in_composition(
        self,
    ) -> None:
        resolver = _CountingResolver("/usr/bin/tool")
        policy = ExecutionPolicy(
            settings=_settings(allow_media_tools=True),
            resolver=resolver,
        )

        await self._assert_denied(
            ShellCompositionRequest(
                steps=(
                    _step(
                        "raster",
                        "pdftoppm",
                        options={
                            "first_page": 1,
                            "last_page": 1,
                            "format": "png",
                        },
                        paths=("media/small.pdf",),
                    ),
                )
            ),
            ShellExecutionErrorCode.DENIED_COMMAND,
            policy=policy,
        )
        self.assertEqual(resolver.calls, ("pdftoppm",))

    async def _assert_denied(
        self,
        request: ShellCompositionRequest,
        error_code: ShellExecutionErrorCode,
        *,
        policy: ExecutionPolicy,
    ) -> None:
        with self.assertRaises(ShellPolicyDenied) as context:
            await policy.normalize_composition(request)
        self.assertEqual(context.exception.error_code, error_code)


class _CountingResolver:
    def __init__(self, result: str | None) -> None:
        self._result = result
        self._calls: list[str] = []

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        self._calls.append(command.logical_id)
        return self._result


class _RecordingExecutor:
    def __init__(self) -> None:
        self._calls: list[ExecutionSpec] = []

    @property
    def calls(self) -> tuple[ExecutionSpec, ...]:
        return tuple(self._calls)

    async def execute(self, spec: ExecutionSpec) -> None:
        self._calls.append(spec)


def _settings(**kwargs: object) -> ShellToolSettings:
    return ShellToolSettings(
        allow_pipelines=True,
        workspace_root=str(_FIXTURE_ROOT),
        **kwargs,
    )


def _step(
    step_id: str,
    command: str,
    *,
    options: dict[str, object] | None = None,
    paths: tuple[str, ...] = (),
    cwd: str | None = None,
    stdin_from: ShellStreamRef | None = None,
) -> ShellCommandStepRequest:
    return ShellCommandStepRequest(
        id=step_id,
        command=command,
        options={} if options is None else options,
        paths=paths,
        cwd=cwd,
        stdin_from=stdin_from,
    )


if __name__ == "__main__":
    main()
