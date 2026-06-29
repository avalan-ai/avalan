from dataclasses import FrozenInstanceError
from typing import Any, cast, get_args
from unittest import TestCase, main

from avalan.tool.shell.entities import (
    DEFAULT_MAX_PIPELINE_STAGES,
    ExecutionSpec,
    ShellCommandStepRequest,
    ShellCompositionMode,
    ShellCompositionRequest,
    ShellCompositionResult,
    ShellCompositionSpec,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellExecutionStepSpec,
    ShellOutputKind,
    ShellStreamRef,
)
from avalan.tool.shell.policy import ExecutionPolicy


def _create_execution_spec(command: str = "cat") -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend="local",
        tool_name=f"shell.{command}",
        command=command,
        executable=f"/usr/bin/{command}",
        argv=(command,),
        display_argv=(command,),
        cwd="/workspace",
        display_cwd=".",
        env={},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=1.0,
        max_stdout_bytes=10,
        max_stderr_bytes=10,
    )


def _create_step_result(step_id: str = "read") -> ShellExecutionStepResult:
    return ShellExecutionStepResult(
        id=step_id,
        command="cat",
        status=ShellExecutionStatus.COMPLETED,
        exit_code=0,
        stdout="contents",
        stderr="",
        stdout_bytes=8,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        duration_ms=1,
    )


def _create_command_steps(
    count: int,
) -> tuple[ShellCommandStepRequest, ...]:
    return tuple(
        ShellCommandStepRequest(id=f"step-{index}", command="cat")
        for index in range(count)
    )


def _create_execution_steps(
    count: int,
) -> tuple[ShellExecutionStepSpec, ...]:
    return tuple(
        ShellExecutionStepSpec(
            id=f"step-{index}",
            spec=_create_execution_spec("cat"),
        )
        for index in range(count)
    )


class ShellCompositionEntitiesTest(TestCase):
    def test_composition_mode_values_are_locked(self) -> None:
        self.assertEqual(
            get_args(ShellCompositionMode),
            ("pipeline", "serial", "parallel"),
        )
        for mode in get_args(ShellCompositionMode):
            with self.subTest(mode=mode):
                request = ShellCompositionRequest(
                    mode=mode,
                    steps=(
                        ShellCommandStepRequest(
                            id=f"{mode}-step",
                            command="cat",
                        ),
                    ),
                )

                self.assertEqual(request.mode, mode)

    def test_stream_ref_accepts_only_stdout(self) -> None:
        stream_ref = ShellStreamRef(step_id="read", stream="stdout")

        self.assertEqual(stream_ref.step_id, "read")
        self.assertEqual(stream_ref.stream, "stdout")

        with self.assertRaises(AssertionError):
            ShellStreamRef(step_id="", stream="stdout")
        with self.assertRaises(AssertionError):
            ShellStreamRef(
                step_id="read",
                stream=cast(Any, "stderr"),
            )

    def test_command_step_request_defaults_and_copies_options(self) -> None:
        options: dict[str, object] = {"number": True}
        step = ShellCommandStepRequest(
            id="read",
            command="cat",
            options=options,
            paths=("input.txt",),
            cwd=".",
        )
        minimal_step = ShellCommandStepRequest(id="count", command="wc")

        options["number"] = False

        self.assertEqual(step.options, {"number": True})
        self.assertEqual(step.paths, ("input.txt",))
        self.assertEqual(step.cwd, ".")
        self.assertIsNone(step.stdin_from)
        self.assertEqual(minimal_step.options, {})
        self.assertEqual(minimal_step.paths, ())
        self.assertIsNone(minimal_step.cwd)

    def test_command_step_request_requires_typed_stream_refs(self) -> None:
        valid: dict[str, object] = {"id": "count", "command": "wc"}
        invalid_kwargs: tuple[dict[str, object], ...] = (
            {"id": ""},
            {"command": ""},
            {"options": []},
            {"paths": ["input.txt"]},
            {"paths": ("",)},
            {"paths": (object(),)},
            {"cwd": ""},
            {"stdin_from": "read.stdout"},
            {"stdin_from": {"step_id": "read", "stream": "stdout"}},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                step_kwargs = dict(valid)
                step_kwargs.update(kwargs)
                with self.assertRaises(AssertionError):
                    ShellCommandStepRequest(**cast(Any, step_kwargs))

    def test_composition_request_accepts_known_refs_and_zero_caps(
        self,
    ) -> None:
        read = ShellCommandStepRequest(id="read", command="cat")
        count = ShellCommandStepRequest(
            id="count",
            command="wc",
            stdin_from=ShellStreamRef(step_id="read", stream="stdout"),
        )

        request = ShellCompositionRequest(
            steps=(read, count),
            timeout_seconds=0.1,
            max_stdout_bytes=0,
            max_stderr_bytes=0,
            max_intermediate_bytes=0,
        )

        self.assertEqual(request.mode, "pipeline")
        self.assertEqual(request.steps, (read, count))
        self.assertEqual(request.max_intermediate_bytes, 0)

    def test_composition_request_rejects_invalid_modes_steps_refs_and_caps(
        self,
    ) -> None:
        read = ShellCommandStepRequest(id="read", command="cat")
        duplicate = ShellCommandStepRequest(id="read", command="wc")
        missing_ref = ShellCommandStepRequest(
            id="count",
            command="wc",
            stdin_from=ShellStreamRef(step_id="missing", stream="stdout"),
        )
        valid: dict[str, object] = {"steps": (read,)}
        invalid_kwargs: tuple[dict[str, object], ...] = (
            {"mode": "batch"},
            {"steps": []},
            {"steps": ()},
            {"steps": (object(),)},
            {"steps": (read, duplicate)},
            {"steps": (read, missing_ref)},
            {"timeout_seconds": 0},
            {"timeout_seconds": True},
            {"max_stdout_bytes": -1},
            {"max_stdout_bytes": True},
            {"max_stderr_bytes": -1},
            {"max_stderr_bytes": True},
            {"max_intermediate_bytes": -1},
            {"max_intermediate_bytes": True},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                request_kwargs = dict(valid)
                request_kwargs.update(kwargs)
                with self.assertRaises(AssertionError):
                    ShellCompositionRequest(**cast(Any, request_kwargs))

    def test_composition_request_stage_count_validation_uses_default_cap(
        self,
    ) -> None:
        boundary = ShellCompositionRequest(
            steps=_create_command_steps(DEFAULT_MAX_PIPELINE_STAGES),
        )
        oversized = ShellCompositionRequest(
            steps=_create_command_steps(DEFAULT_MAX_PIPELINE_STAGES + 1),
        )

        boundary.validate_stage_count()
        with self.assertRaises(AssertionError):
            oversized.validate_stage_count()

    def test_composition_request_stage_count_validation_uses_policy_cap(
        self,
    ) -> None:
        boundary = ShellCompositionRequest(steps=_create_command_steps(2))
        oversized = ShellCompositionRequest(steps=_create_command_steps(3))

        boundary.validate_stage_count(max_pipeline_stages=2)
        with self.assertRaises(AssertionError):
            oversized.validate_stage_count(max_pipeline_stages=2)
        for cap in (0, True):
            with self.subTest(cap=cap):
                with self.assertRaises(AssertionError):
                    boundary.validate_stage_count(
                        max_pipeline_stages=cast(Any, cap),
                    )

    def test_execution_step_spec_and_composition_spec_accept_known_refs(
        self,
    ) -> None:
        read = ShellExecutionStepSpec(
            id="read",
            spec=_create_execution_spec("cat"),
        )
        count = ShellExecutionStepSpec(
            id="count",
            spec=_create_execution_spec("wc"),
            stdin_from=ShellStreamRef(step_id="read", stream="stdout"),
        )

        spec = ShellCompositionSpec(
            mode="serial",
            steps=(read, count),
            timeout_seconds=0.1,
            max_stdout_bytes=0,
            max_stderr_bytes=0,
            max_intermediate_bytes=0,
        )

        self.assertEqual(spec.mode, "serial")
        self.assertEqual(spec.steps, (read, count))
        self.assertEqual(spec.max_stdout_bytes, 0)

    def test_execution_step_spec_rejects_invalid_fields(self) -> None:
        spec = _create_execution_spec()

        with self.assertRaises(AssertionError):
            ShellExecutionStepSpec(id="", spec=spec)
        with self.assertRaises(AssertionError):
            ShellExecutionStepSpec(
                id="read",
                spec=cast(Any, object()),
            )
        with self.assertRaises(AssertionError):
            ShellExecutionStepSpec(
                id="read",
                spec=spec,
                stdin_from=cast(Any, "read.stdout"),
            )

    def test_composition_spec_rejects_invalid_modes_steps_refs_and_caps(
        self,
    ) -> None:
        read = ShellExecutionStepSpec(
            id="read",
            spec=_create_execution_spec("cat"),
        )
        duplicate = ShellExecutionStepSpec(
            id="read",
            spec=_create_execution_spec("wc"),
        )
        missing_ref = ShellExecutionStepSpec(
            id="count",
            spec=_create_execution_spec("wc"),
            stdin_from=ShellStreamRef(step_id="missing", stream="stdout"),
        )
        valid: dict[str, object] = {
            "mode": "pipeline",
            "steps": (read,),
            "timeout_seconds": 1.0,
            "max_stdout_bytes": 1,
            "max_stderr_bytes": 1,
            "max_intermediate_bytes": 1,
        }
        invalid_kwargs: tuple[dict[str, object], ...] = (
            {"mode": "batch"},
            {"steps": []},
            {"steps": ()},
            {"steps": (object(),)},
            {"steps": (read, duplicate)},
            {"steps": (read, missing_ref)},
            {"timeout_seconds": 0},
            {"timeout_seconds": True},
            {"max_stdout_bytes": -1},
            {"max_stdout_bytes": True},
            {"max_stderr_bytes": -1},
            {"max_stderr_bytes": True},
            {"max_intermediate_bytes": -1},
            {"max_intermediate_bytes": True},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                spec_kwargs = dict(valid)
                spec_kwargs.update(kwargs)
                with self.assertRaises(AssertionError):
                    ShellCompositionSpec(**cast(Any, spec_kwargs))

    def test_composition_spec_stage_count_validation_uses_default_cap(
        self,
    ) -> None:
        boundary = ShellCompositionSpec(
            mode="pipeline",
            steps=_create_execution_steps(DEFAULT_MAX_PIPELINE_STAGES),
            timeout_seconds=1.0,
            max_stdout_bytes=1,
            max_stderr_bytes=1,
            max_intermediate_bytes=1,
        )
        oversized = ShellCompositionSpec(
            mode="pipeline",
            steps=_create_execution_steps(DEFAULT_MAX_PIPELINE_STAGES + 1),
            timeout_seconds=1.0,
            max_stdout_bytes=1,
            max_stderr_bytes=1,
            max_intermediate_bytes=1,
        )

        boundary.validate_stage_count()
        with self.assertRaises(AssertionError):
            oversized.validate_stage_count()

    def test_composition_spec_stage_count_validation_uses_policy_cap(
        self,
    ) -> None:
        boundary = ShellCompositionSpec(
            mode="pipeline",
            steps=_create_execution_steps(2),
            timeout_seconds=1.0,
            max_stdout_bytes=1,
            max_stderr_bytes=1,
            max_intermediate_bytes=1,
        )
        oversized = ShellCompositionSpec(
            mode="pipeline",
            steps=_create_execution_steps(3),
            timeout_seconds=1.0,
            max_stdout_bytes=1,
            max_stderr_bytes=1,
            max_intermediate_bytes=1,
        )

        boundary.validate_stage_count(max_pipeline_stages=2)
        with self.assertRaises(AssertionError):
            oversized.validate_stage_count(max_pipeline_stages=2)
        for cap in (0, True):
            with self.subTest(cap=cap):
                with self.assertRaises(AssertionError):
                    boundary.validate_stage_count(
                        max_pipeline_stages=cast(Any, cap),
                    )

    def test_step_result_and_composition_result_copy_metadata(self) -> None:
        step_metadata: dict[str, object] = {"stage": 1}
        step = ShellExecutionStepResult(
            id="read",
            command="cat",
            status=ShellExecutionStatus.NONZERO_EXIT,
            exit_code=1,
            stdout="",
            stderr="failed",
            stdout_bytes=0,
            stderr_bytes=6,
            stdout_truncated=False,
            stderr_truncated=False,
            duration_ms=2,
            error_code=ShellExecutionErrorCode.NONZERO_EXIT,
            error_message="command failed",
            metadata=step_metadata,
        )
        result_metadata: dict[str, object] = {"duration_source": "clock"}
        result = ShellCompositionResult(
            mode="pipeline",
            status=ShellExecutionStatus.NONZERO_EXIT,
            stdout="",
            stderr="failed",
            steps=(step,),
            stdout_bytes=0,
            stderr_bytes=6,
            stderr_truncated=True,
            duration_ms=3,
            error_code=ShellExecutionErrorCode.NONZERO_EXIT,
            error_message="composition failed",
            metadata=result_metadata,
        )

        step_metadata["stage"] = 2
        result_metadata["duration_source"] = "changed"

        self.assertEqual(step.metadata, {"stage": 1})
        self.assertEqual(result.metadata, {"duration_source": "clock"})
        self.assertEqual(result.steps, (step,))

    def test_step_result_rejects_invalid_fields(self) -> None:
        valid: dict[str, object] = {
            "id": "read",
            "command": "cat",
            "status": ShellExecutionStatus.COMPLETED,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "stdout_truncated": False,
            "stderr_truncated": False,
            "duration_ms": 0,
        }
        invalid_values: dict[str, object] = {
            "id": "",
            "command": "",
            "status": "completed",
            "exit_code": True,
            "stdout": b"",
            "stderr": b"",
            "stdout_bytes": -1,
            "stderr_bytes": True,
            "stdout_truncated": 1,
            "stderr_truncated": 0,
            "duration_ms": -1,
            "error_code": "timeout",
            "error_message": "",
            "metadata": [],
        }
        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                kwargs = dict(valid)
                kwargs[field_name] = value
                with self.assertRaises(AssertionError):
                    ShellExecutionStepResult(**cast(Any, kwargs))

    def test_composition_result_rejects_invalid_fields(self) -> None:
        step = _create_step_result()
        duplicate = _create_step_result()
        valid: dict[str, object] = {
            "mode": "pipeline",
            "status": ShellExecutionStatus.COMPLETED,
            "stdout": "",
            "stderr": "",
            "steps": (step,),
        }
        invalid_kwargs: tuple[dict[str, object], ...] = (
            {"mode": "batch"},
            {"status": "completed"},
            {"stdout": b""},
            {"stderr": b""},
            {"steps": []},
            {"steps": ()},
            {"steps": (object(),)},
            {"steps": (step, duplicate)},
            {"stdout_bytes": -1},
            {"stdout_bytes": True},
            {"stderr_bytes": -1},
            {"stderr_bytes": True},
            {"stdout_truncated": 1},
            {"stderr_truncated": 0},
            {"timed_out": 1},
            {"cancelled": 0},
            {"duration_ms": -1},
            {"error_code": "timeout"},
            {"error_message": ""},
            {"metadata": []},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                result_kwargs = dict(valid)
                result_kwargs.update(kwargs)
                with self.assertRaises(AssertionError):
                    ShellCompositionResult(**cast(Any, result_kwargs))

    def test_composition_entities_are_frozen_and_slotted(self) -> None:
        ref = ShellStreamRef(step_id="read", stream="stdout")
        request_step = ShellCommandStepRequest(id="read", command="cat")
        request = ShellCompositionRequest(steps=(request_step,))
        spec_step = ShellExecutionStepSpec(
            id="read",
            spec=_create_execution_spec(),
        )
        spec = ShellCompositionSpec(
            mode="pipeline",
            steps=(spec_step,),
            timeout_seconds=1.0,
            max_stdout_bytes=1,
            max_stderr_bytes=1,
            max_intermediate_bytes=1,
        )
        step_result = _create_step_result()
        result = ShellCompositionResult(
            mode="pipeline",
            status=ShellExecutionStatus.COMPLETED,
            stdout="contents",
            stderr="",
            steps=(step_result,),
        )

        for instance, field_name in (
            (ref, "step_id"),
            (request_step, "id"),
            (request, "mode"),
            (spec_step, "id"),
            (spec, "mode"),
            (step_result, "id"),
            (result, "mode"),
        ):
            with self.subTest(instance=type(instance).__name__):
                self.assertFalse(hasattr(instance, "__dict__"))
                with self.assertRaises(FrozenInstanceError):
                    value = getattr(instance, field_name)
                    setattr(instance, field_name, value)


if __name__ == "__main__":
    main()
