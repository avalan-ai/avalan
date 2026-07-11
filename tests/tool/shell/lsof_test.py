from collections.abc import Awaitable, Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import ToolCallContext, ToolExecutionStreamEvent
from avalan.tool.shell import (
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    LsofTool,
    PathOperand,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellCommandStepRequest,
    ShellCompositionRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellToolSettings,
)
from avalan.tool.shell.lsof import (
    LSOF_DEFAULT_LIMIT,
    LSOF_MAX_LIMIT,
    LSOF_MAX_PID,
    REDACTED_LSOF_STDERR,
    LsofOutput,
    open_file_rows_stdout,
    redacted_stderr,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
PRIVATE_VALUE = "private-lsof-value"
_MISSING = object()


class LsofPolicyTest(IsolatedAsyncioTestCase):
    async def test_normalizes_exact_fixed_argv_and_metadata(self) -> None:
        expected = (
            "lsof",
            "-n",
            "-P",
            "-w",
            "-b",
            "-a",
            "-p",
            "42",
            "-F0pftaP",
        )
        for limit in (LSOF_DEFAULT_LIMIT, LSOF_MAX_LIMIT):
            with self.subTest(limit=limit):
                resolver = _RecordingResolver("/trusted/bin/lsof")
                request = (
                    _request(42)
                    if limit == LSOF_DEFAULT_LIMIT
                    else _request(42, limit=limit)
                )

                spec = await _policy(resolver=resolver).normalize(request)

                self.assertEqual(spec.executable, "/trusted/bin/lsof")
                self.assertEqual(spec.argv, expected)
                self.assertEqual(spec.display_argv, expected)
                self.assertNotIn("exit_code_statuses", spec.metadata)
                self.assertEqual(spec.metadata["_lsof_requested_pid"], 42)
                self.assertEqual(spec.metadata["_lsof_limit"], limit)
                self.assertEqual(resolver.commands, ["lsof"])

    async def test_rejects_invalid_pid_and_limit_before_resolution(
        self,
    ) -> None:
        invalid_requests = (
            _request(None),
            _request(True),
            _request(0),
            _request(-1),
            _request(LSOF_MAX_PID + 1),
            _request("42"),
            _request(42.0),
            _request(42, limit=None),
            _request(42, limit=True),
            _request(42, limit=0),
            _request(42, limit=-1),
            _request(42, limit=LSOF_MAX_LIMIT + 1),
            _request(42, limit="64"),
            _request(42, limit=64.0),
        )
        for request in invalid_requests:
            with self.subTest(options=request.options):
                resolver = _RecordingResolver("/trusted/bin/lsof")

                with self.assertRaises(ShellPolicyDenied) as raised:
                    await _policy(resolver=resolver).normalize(request)

                self.assertIs(
                    raised.exception.error_code,
                    ShellExecutionErrorCode.INVALID_OPTION,
                )
                self.assertEqual(resolver.commands, [])

    async def test_accepts_maximum_pid(self) -> None:
        spec = await _policy().normalize(_request(LSOF_MAX_PID))

        self.assertEqual(
            spec.argv,
            (
                "lsof",
                "-n",
                "-P",
                "-w",
                "-b",
                "-a",
                "-p",
                str(LSOF_MAX_PID),
                "-F0pftaP",
            ),
        )
        self.assertEqual(
            spec.metadata["_lsof_requested_pid"],
            LSOF_MAX_PID,
        )

    async def test_rejects_unknown_options_and_paths_before_resolution(
        self,
    ) -> None:
        requests = (
            ShellCommandRequest(
                tool_name="shell.lsof",
                command="lsof",
                options={"pid": 42, "limit": 64, "all": True},
                paths=(),
                cwd=None,
            ),
            ShellCommandRequest(
                tool_name="shell.lsof",
                command="lsof",
                options={"pid": 42, "limit": 64},
                paths=(
                    PathOperand(
                        name="input",
                        path="filesystem/visible.txt",
                        kind="text_file",
                        access="read",
                    ),
                ),
                cwd=None,
            ),
        )
        for request in requests:
            with self.subTest(request=request):
                resolver = _RecordingResolver("/trusted/bin/lsof")

                with self.assertRaises(ShellPolicyDenied):
                    await _policy(resolver=resolver).normalize(request)

                self.assertEqual(resolver.commands, [])

    async def test_process_gate_denies_before_resolution(self) -> None:
        resolver = _RecordingResolver("/trusted/bin/lsof")
        settings = ShellToolSettings(workspace_root=str(FIXTURE_ROOT))

        with self.assertRaises(ShellPolicyDenied) as raised:
            await ExecutionPolicy(
                settings=settings,
                resolver=resolver,
            ).normalize(_request(42))

        self.assertIs(
            raised.exception.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )
        self.assertEqual(resolver.commands, [])

    async def test_lsof_is_denied_in_every_composition_mode(self) -> None:
        for mode in ("serial", "parallel", "pipeline"):
            with self.subTest(mode=mode):
                resolver = _RecordingResolver("/trusted/bin/lsof")
                settings = ShellToolSettings(
                    workspace_root=str(FIXTURE_ROOT),
                    allow_pipelines=True,
                    allow_process_tools=True,
                )
                request = ShellCompositionRequest(
                    mode=mode,
                    steps=(
                        ShellCommandStepRequest(
                            id="open-files",
                            command="lsof",
                            options={"pid": 42, "limit": 64},
                        ),
                    ),
                )

                with self.assertRaises(ShellPolicyDenied) as raised:
                    await ExecutionPolicy(
                        settings=settings,
                        resolver=resolver,
                    ).normalize_composition(request)

                self.assertIs(
                    raised.exception.error_code,
                    ShellExecutionErrorCode.DENIED_COMMAND,
                )
                self.assertEqual(resolver.commands, [])


class LsofOutputTest(TestCase):
    def test_parses_darwin_and_linux_records_into_fixed_tsv(self) -> None:
        raw = (
            _record("p42")
            + _record("f0", "ar", "tCHR")
            + _record("f3", "tREG", "aw")
            + _record("f4", "tIPv4", "PTCP", "au")
            + _record("f5", "tIPv6", "PUDP")
        )

        parsed = open_file_rows_stdout(
            raw,
            requested_pid=42,
            limit=LSOF_DEFAULT_LIMIT,
        )

        self.assertIsInstance(parsed, LsofOutput)
        self.assertEqual(
            parsed.stdout,
            "42\t0\tr\tcharacter\t-\n"
            "42\t3\tw\tregular\t-\n"
            "42\t4\tu\tipv4\ttcp\n"
            "42\t5\t-\tipv6\tudp\n",
        )
        self.assertFalse(parsed.row_limit_truncated)
        self.assertFalse(parsed.byte_limit_truncated)
        self.assertFalse(parsed.malformed)

    def test_skips_pseudo_descriptors_and_normalizes_blank_fields(
        self,
    ) -> None:
        raw = (
            _record("p42")
            + _record("fcwd", "a ", "tDIR")
            + _record("ftxt", "tREG")
            + _record("ferr")
            + _record("f6", "a", "tPIPE")
            + _record("f7", "a ", "tREG")
        )

        parsed = open_file_rows_stdout(raw, requested_pid=42, limit=64)

        self.assertEqual(
            parsed.stdout,
            "42\t6\t-\tpipe\t-\n42\t7\t-\tregular\t-\n",
        )
        self.assertFalse(parsed.row_limit_truncated)
        self.assertFalse(parsed.byte_limit_truncated)
        self.assertFalse(parsed.malformed)

    def test_numeric_descriptor_without_type_maps_to_other(self) -> None:
        parsed = open_file_rows_stdout(
            _record("p42") + _record("f3", "ar"),
            requested_pid=42,
            limit=64,
        )

        self.assertEqual(parsed.stdout, "42\t3\tr\tother\t-\n")
        self.assertFalse(parsed.row_limit_truncated)
        self.assertFalse(parsed.byte_limit_truncated)
        self.assertFalse(parsed.malformed)

        unsafe = open_file_rows_stdout(
            _record("p42") + _record("f3", "ar", "t/private/type"),
            requested_pid=42,
            limit=64,
        )
        self.assertEqual(unsafe.stdout, "")
        self.assertFalse(unsafe.row_limit_truncated)
        self.assertFalse(unsafe.byte_limit_truncated)
        self.assertTrue(unsafe.malformed)

    def test_normalizes_finite_type_and_protocol_vocabulary(self) -> None:
        type_cases = (
            ("REG", "regular"),
            ("DIR", "directory"),
            ("CHR", "character"),
            ("BLK", "block"),
            ("FIFO", "pipe"),
            ("PIPE", "pipe"),
            ("IPv4", "ipv4"),
            ("IPv6", "ipv6"),
            ("unix", "unix_socket"),
            ("UNIX", "unix_socket"),
            ("SOCK", "socket"),
            ("KQUEUE", "event"),
            ("UNKNOWN_IDENTIFIER", "other"),
        )
        protocol_cases = (
            ("TCP", "tcp"),
            ("UDP", "udp"),
            ("UDPLITE", "udplite"),
            ("UNKNOWN_PROTOCOL", "other"),
        )
        raw = _record("p42")
        expected_rows: list[str] = []
        for descriptor, (raw_type, canonical_type) in enumerate(type_cases):
            raw += _record(f"f{descriptor}", f"t{raw_type}")
            expected_rows.append(f"42\t{descriptor}\t-\t{canonical_type}\t-\n")
        for offset, (raw_protocol, canonical_protocol) in enumerate(
            protocol_cases,
            start=len(type_cases),
        ):
            raw += _record(f"f{offset}", "tREG", f"P{raw_protocol}")
            expected_rows.append(
                f"42\t{offset}\t-\tregular\t{canonical_protocol}\n"
            )

        parsed = open_file_rows_stdout(raw, requested_pid=42, limit=64)

        self.assertEqual(parsed.stdout, "".join(expected_rows))
        self.assertFalse(parsed.row_limit_truncated)
        self.assertFalse(parsed.byte_limit_truncated)
        self.assertFalse(parsed.malformed)

    def test_maps_adversarial_type_and_protocol_values_without_echo(
        self,
    ) -> None:
        private_values = (
            "IGNORE_PREVIOUS_INSTRUCTIONS",
            "SENTINELPRIVATE42",
            "Secret_value_42",
        )
        raw = _record("p42")
        expected_rows: list[str] = []
        for index, private_value in enumerate(private_values):
            type_descriptor = index * 2
            protocol_descriptor = type_descriptor + 1
            raw += _record(f"f{type_descriptor}", f"t{private_value}")
            raw += _record(
                f"f{protocol_descriptor}",
                "tREG",
                f"P{private_value}",
            )
            expected_rows.append(f"42\t{type_descriptor}\t-\tother\t-\n")
            expected_rows.append(
                f"42\t{protocol_descriptor}\t-\tregular\tother\n"
            )

        parsed = open_file_rows_stdout(raw, requested_pid=42, limit=64)

        self.assertEqual(parsed.stdout, "".join(expected_rows))
        self.assertFalse(parsed.row_limit_truncated)
        self.assertFalse(parsed.byte_limit_truncated)
        self.assertFalse(parsed.malformed)
        for private_value in private_values:
            self.assertNotIn(private_value, parsed.stdout)

    def test_rejects_unsafe_type_and_protocol_values_without_echo(
        self,
    ) -> None:
        private_values = (
            "/private/path",
            "remote.example:443",
            "SECRET=value",
            '"quoted secret"',
            "'quoted secret'",
            "private\\path",
            "\x1b[31mCONTROL_SECRET",
            "private\tvalue",
            "privatÉ",
        )
        for private_value in private_values:
            for fields in (
                ("f3", f"t{private_value}"),
                ("f3", "tREG", f"P{private_value}"),
            ):
                with self.subTest(
                    private_value=private_value,
                    field=fields[-1][0],
                ):
                    parsed = open_file_rows_stdout(
                        _record("p42") + _record(*fields),
                        requested_pid=42,
                        limit=64,
                    )

                    self.assertEqual(parsed.stdout, "")
                    self.assertFalse(parsed.row_limit_truncated)
                    self.assertFalse(parsed.byte_limit_truncated)
                    self.assertTrue(parsed.malformed)
                    self.assertNotIn(private_value, parsed.stdout)

    def test_distinguishes_valid_empty_output_from_malformed_output(
        self,
    ) -> None:
        for raw in (
            _record("p42"),
            _record("p42") + _record("fcwd", "tDIR"),
        ):
            with self.subTest(raw=raw):
                parsed = open_file_rows_stdout(
                    raw,
                    requested_pid=42,
                    limit=64,
                )

                self.assertEqual(parsed.stdout, "")
                self.assertFalse(parsed.row_limit_truncated)
                self.assertFalse(parsed.byte_limit_truncated)
                self.assertFalse(parsed.malformed)

        malformed = open_file_rows_stdout(
            _record("p42") + _record("f3", "n/private/path", "tREG"),
            requested_pid=42,
            limit=64,
        )
        self.assertEqual(malformed.stdout, "")
        self.assertFalse(malformed.row_limit_truncated)
        self.assertFalse(malformed.byte_limit_truncated)
        self.assertTrue(malformed.malformed)

    def test_rejects_wrong_extra_and_forged_process_records(self) -> None:
        invalid_values = (
            _record("p43") + _record("f3", "tREG"),
            _record("p42") + _record("p43") + _record("f3", "tREG"),
            _record("p42") + _record("p42") + _record("f3", "tREG"),
            _record("p042") + _record("f3", "tREG"),
            _record("p0") + _record("f3", "tREG"),
            _record(f"p{LSOF_MAX_PID + 1}") + _record("f3", "tREG"),
        )
        for raw in invalid_values:
            with self.subTest(raw=raw):
                self._assert_malformed(raw)

    def test_rejects_duplicate_unknown_and_missing_fields(self) -> None:
        invalid_values = (
            _record("f3", "tREG"),
            _record("p42") + _record("tREG"),
            _record("p42") + _record("f3", "", "tREG"),
            _record("p42") + _record("f3", "f4", "tREG"),
            _record("p42") + _record("f3", "tREG", "tDIR"),
            _record("p42") + _record("f3", "ar", "aw", "tREG"),
            _record("p42") + _record("f3", "tIPv4", "PTCP", "PUDP"),
            _record("p42") + _record("f3", "tREG") + _record("f3", "tREG"),
            _record("p42") + _record("f3", "tREG", "cprivate"),
            _record("p42") + _record("f3", "tREG", "n/private"),
        )
        for raw in invalid_values:
            with self.subTest(raw=raw):
                self._assert_malformed(raw)

    def test_rejects_noncanonical_descriptors_access_and_oversized_tokens(
        self,
    ) -> None:
        invalid_fields = (
            ("f03", "tREG"),
            (f"f{2**31}", "tREG"),
            (f"f{'9' * 65}", "tREG"),
            ("f3\x1b", "tREG"),
            ("f3", "aexecute", "tREG"),
            ("f3", "a\x1b", "tREG"),
            ("f3", f"t{'X' * 65}"),
            ("f3", "tIPv4", f"P{'X' * 65}"),
        )
        for fields in invalid_fields:
            with self.subTest(fields=fields):
                self._assert_malformed(_record("p42") + _record(*fields))

    def test_rejects_malformed_framing_but_discards_raw_truncated_tail(
        self,
    ) -> None:
        complete = _record("p42") + _record("f3", "ar", "tREG")
        incomplete = "f4\0aw\0tIPv4"
        invalid_values = (
            "",
            "p42\nf3\ntREG\n",
            "p42\0f3\0tREG\0",
            complete + incomplete,
            complete + "private trailing text",
            _record("p42") + "\n",
        )
        for raw in invalid_values:
            with self.subTest(raw=raw):
                self._assert_malformed(raw)

        parsed = open_file_rows_stdout(
            complete + incomplete,
            requested_pid=42,
            limit=64,
            stdout_truncated=True,
        )
        self.assertEqual(parsed.stdout, "42\t3\tr\tregular\t-\n")
        self.assertFalse(parsed.row_limit_truncated)
        self.assertFalse(parsed.byte_limit_truncated)
        self.assertFalse(parsed.malformed)

        no_complete_record = open_file_rows_stdout(
            "p42\0",
            requested_pid=42,
            limit=64,
            stdout_truncated=True,
        )
        self.assertEqual(no_complete_record.stdout, "")
        self.assertFalse(no_complete_record.row_limit_truncated)
        self.assertFalse(no_complete_record.byte_limit_truncated)
        self.assertFalse(no_complete_record.malformed)

    def test_enforces_logical_limit_without_counting_pseudo_descriptors(
        self,
    ) -> None:
        raw = (
            _record("p42")
            + _record("fcwd", "tDIR")
            + _record("f0", "tCHR")
            + _record("f1", "tPIPE")
            + _record("f2", "tREG")
        )

        limited = open_file_rows_stdout(raw, requested_pid=42, limit=2)
        exact = open_file_rows_stdout(raw, requested_pid=42, limit=3)

        self.assertEqual(
            limited.stdout,
            "42\t0\t-\tcharacter\t-\n42\t1\t-\tpipe\t-\n",
        )
        self.assertTrue(limited.row_limit_truncated)
        self.assertFalse(limited.byte_limit_truncated)
        self.assertFalse(limited.malformed)
        self.assertEqual(
            exact.stdout,
            limited.stdout + "42\t2\t-\tregular\t-\n",
        )
        self.assertFalse(exact.row_limit_truncated)
        self.assertFalse(exact.byte_limit_truncated)
        self.assertFalse(exact.malformed)

    def test_caps_canonical_stdout_at_whole_rows_after_expansion(
        self,
    ) -> None:
        raw = (
            _record(f"p{LSOF_MAX_PID}")
            + _record("f0", "tREG")
            + _record("f1", "tREG")
        )
        raw_bytes = len(raw.encode("ascii"))
        first_row = f"{LSOF_MAX_PID}\t0\t-\tregular\t-\n"

        parsed = open_file_rows_stdout(
            raw,
            requested_pid=LSOF_MAX_PID,
            limit=64,
            max_stdout_bytes=raw_bytes,
        )

        self.assertEqual(raw_bytes, 31)
        self.assertEqual(parsed.stdout, first_row)
        self.assertLessEqual(len(parsed.stdout.encode("ascii")), raw_bytes)
        self.assertFalse(parsed.row_limit_truncated)
        self.assertTrue(parsed.byte_limit_truncated)
        self.assertFalse(parsed.malformed)

        empty = open_file_rows_stdout(
            raw,
            requested_pid=LSOF_MAX_PID,
            limit=64,
            max_stdout_bytes=0,
        )
        self.assertEqual(empty.stdout, "")
        self.assertFalse(empty.row_limit_truncated)
        self.assertTrue(empty.byte_limit_truncated)
        self.assertFalse(empty.malformed)

    def test_continues_validation_after_presentation_truncation(self) -> None:
        byte_prefix = (
            _record(f"p{LSOF_MAX_PID}")
            + _record("f0", "tREG")
            + _record("f1", "tREG")
        )
        cases = (
            (
                "row",
                _record("p42")
                + _record("f0", "tREG")
                + _record("f1", "tREG")
                + _record("f0", "tREG"),
                42,
                1,
                None,
                True,
                False,
            ),
            (
                "byte",
                byte_prefix + _record("f0", "tREG"),
                LSOF_MAX_PID,
                64,
                len(byte_prefix.encode("ascii")),
                False,
                True,
            ),
        )
        for (
            cap,
            raw,
            requested_pid,
            limit,
            max_stdout_bytes,
            row_limit_truncated,
            byte_limit_truncated,
        ) in cases:
            with self.subTest(cap=cap):
                parsed = open_file_rows_stdout(
                    raw,
                    requested_pid=requested_pid,
                    limit=limit,
                    max_stdout_bytes=max_stdout_bytes,
                )

                self.assertEqual(parsed.stdout, "")
                self.assertIs(
                    parsed.row_limit_truncated,
                    row_limit_truncated,
                )
                self.assertIs(
                    parsed.byte_limit_truncated,
                    byte_limit_truncated,
                )
                self.assertTrue(parsed.malformed)

    def test_helper_contracts_and_stderr_redaction_are_strict(self) -> None:
        parsed = open_file_rows_stdout(
            _record("p42"),
            requested_pid=42,
            limit=64,
        )
        with self.assertRaises(FrozenInstanceError):
            parsed.stdout = "changed"  # type: ignore[misc]
        self.assertEqual(redacted_stderr(""), "")
        self.assertEqual(
            redacted_stderr(f"diagnostic {PRIVATE_VALUE}"),
            REDACTED_LSOF_STDERR,
        )
        with self.assertRaises(AssertionError):
            redacted_stderr(object())  # type: ignore[arg-type]
        invalid_calls = (
            (object(), 42, 64, False),
            ("", True, 64, False),
            ("", 42, 0, False),
            ("", 42, 64, 1),
        )
        for value, pid, limit, truncated in invalid_calls:
            with self.subTest(
                value=value,
                pid=pid,
                limit=limit,
                truncated=truncated,
            ):
                with self.assertRaises(AssertionError):
                    open_file_rows_stdout(
                        value,  # type: ignore[arg-type]
                        requested_pid=pid,  # type: ignore[arg-type]
                        limit=limit,
                        stdout_truncated=truncated,  # type: ignore[arg-type]
                    )
        for max_stdout_bytes in (True, -1, 1.0, "64"):
            with self.subTest(max_stdout_bytes=max_stdout_bytes):
                with self.assertRaises(AssertionError):
                    open_file_rows_stdout(
                        _record("p42"),
                        requested_pid=42,
                        max_stdout_bytes=max_stdout_bytes,  # type: ignore[arg-type]
                    )

    def _assert_malformed(self, raw: str) -> None:
        parsed = open_file_rows_stdout(
            raw,
            requested_pid=42,
            limit=64,
        )
        self.assertEqual(parsed.stdout, "")
        self.assertFalse(parsed.row_limit_truncated)
        self.assertFalse(parsed.byte_limit_truncated)
        self.assertTrue(parsed.malformed)


class LsofToolTest(IsolatedAsyncioTestCase):
    async def test_schema_is_structured_bounded_and_non_streaming(
        self,
    ) -> None:
        tool = _tool(_RecordingExecutor(_valid_raw()))

        schema = tool.json_schema()["function"]["parameters"]
        properties = schema["properties"]

        self.assertEqual(schema["required"], ["pid"])
        self.assertEqual(
            set(properties),
            {
                "pid",
                "limit",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(properties["pid"]["minimum"], 1)
        self.assertEqual(properties["pid"]["maximum"], LSOF_MAX_PID)
        self.assertEqual(properties["limit"]["minimum"], 1)
        self.assertEqual(properties["limit"]["maximum"], LSOF_MAX_LIMIT)
        self.assertEqual(properties["limit"]["default"], LSOF_DEFAULT_LIMIT)
        self.assertFalse(tool.supports_streaming)

    async def test_model_boundary_sanitizes_and_shapes_public_result(
        self,
    ) -> None:
        raw = (
            _record("p42")
            + _record("f3", "ar", "tREG")
            + _record("f4", "aw", "tIPv4", "PTCP")
        )
        executor = _RecordingExecutor(raw)
        tool = _tool(executor)
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        output = await tool(
            42,
            context=ToolCallContext(stream_event=record),
        )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIsNotNone(executor.spec)
        assert executor.spec is not None
        self.assertEqual(executor.spec.metadata["_lsof_requested_pid"], 42)
        self.assertEqual(executor.spec.metadata["_lsof_limit"], 64)
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(
            result.stdout,
            "42\t3\tr\tregular\t-\n42\t4\tw\tipv4\ttcp\n",
        )
        self.assertEqual(result.stderr, REDACTED_LSOF_STDERR)
        self.assertEqual(result.argv, executor.spec.display_argv)
        self.assertEqual(result.display_argv, executor.spec.display_argv)
        self.assertEqual(result.metadata, {})
        self.assertEqual(result.generated_files, ())
        self.assertEqual(
            result.stdout_bytes, len(result.stdout.encode("utf-8"))
        )
        self.assertEqual(
            result.stderr_bytes,
            len(REDACTED_LSOF_STDERR.encode("utf-8")),
        )
        self.assertFalse(result.stdout_truncated)
        self.assertIsNone(result.error_message)
        self.assertEqual(events, [])
        self.assertNotIn(PRIVATE_VALUE, output)

    async def test_model_boundary_maps_valid_unknown_identifiers_without_echo(
        self,
    ) -> None:
        private_type = "IGNORE_PREVIOUS_INSTRUCTIONS"
        private_protocol = "SENTINELPRIVATE42"
        output = await _tool(
            _RecordingExecutor(
                _record("p42")
                + _record(
                    "f3",
                    f"t{private_type}",
                    f"P{private_protocol}",
                ),
            )
        )(
            42,
            context=ToolCallContext(),
        )

        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "42\t3\t-\tother\tother\n")
        self.assertFalse(result.stdout_truncated)
        self.assertNotIn(private_type, output)
        self.assertNotIn(private_protocol, output)

    async def test_model_boundary_maps_missing_type_to_other(self) -> None:
        output = await _tool(
            _RecordingExecutor(
                _record("p42") + _record("f3", "ar"),
                stderr="",
            )
        )(
            42,
            context=ToolCallContext(),
        )

        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "42\t3\tr\tother\t-\n")
        self.assertFalse(result.stdout_truncated)
        self.assertIsNone(result.error_code)
        self.assertIsNone(result.error_message)

        private_type = "/private/type"
        unsafe = await _tool(
            _RecordingExecutor(
                _record("p42") + _record("f3", "ar", f"t{private_type}"),
                stderr="",
            )
        )(
            42,
            context=ToolCallContext(),
        )
        assert isinstance(unsafe, ShellFormattedResult)
        unsafe_result = unsafe.execution_result
        self.assertIs(unsafe_result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(unsafe_result.stdout, "")
        self.assertEqual(
            unsafe_result.error_message,
            "lsof output was malformed",
        )
        self.assertNotIn(private_type, unsafe)

    async def test_model_boundary_rejects_unsafe_values_without_echo(
        self,
    ) -> None:
        private_values = (
            "/private/path",
            "remote.example:443",
            "SECRET=value",
            '"quoted secret"',
            "'quoted secret'",
            "private\\path",
            "\x1b[31mCONTROL_SECRET",
            "private\tvalue",
        )
        for private_value in private_values:
            for fields in (
                ("f3", f"t{private_value}"),
                ("f3", "tREG", f"P{private_value}"),
            ):
                with self.subTest(
                    private_value=private_value,
                    field=fields[-1][0],
                ):
                    output = await _tool(
                        _RecordingExecutor(
                            _record("p42") + _record(*fields),
                        )
                    )(
                        42,
                        context=ToolCallContext(),
                    )

                    assert isinstance(output, ShellFormattedResult)
                    result = output.execution_result
                    self.assertIs(
                        result.status,
                        ShellExecutionStatus.TOOL_ERROR,
                    )
                    self.assertIs(
                        result.error_code,
                        ShellExecutionErrorCode.TOOL_ERROR,
                    )
                    self.assertEqual(
                        result.error_message,
                        "lsof output was malformed",
                    )
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(result.stdout_bytes, 0)
                    self.assertNotIn(private_value, output)

    async def test_public_stdout_cap_keeps_whole_rows_after_expansion(
        self,
    ) -> None:
        raw = (
            _record(f"p{LSOF_MAX_PID}")
            + _record("f0", "tREG")
            + _record("f1", "tREG")
        )
        max_stdout_bytes = len(raw.encode("ascii"))
        executor = _RecordingExecutor(raw, stderr="")

        output = await _tool(executor)(
            LSOF_MAX_PID,
            max_stdout_bytes=max_stdout_bytes,
            context=ToolCallContext(),
        )

        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIsNotNone(executor.spec)
        assert executor.spec is not None
        self.assertEqual(executor.spec.max_stdout_bytes, max_stdout_bytes)
        self.assertEqual(
            result.stdout,
            f"{LSOF_MAX_PID}\t0\t-\tregular\t-\n",
        )
        self.assertLessEqual(result.stdout_bytes, max_stdout_bytes)
        self.assertTrue(result.stdout.endswith("\n"))
        self.assertTrue(result.stdout_truncated)
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)

    async def test_public_result_preserves_truncation_on_late_malformed_row(
        self,
    ) -> None:
        byte_prefix = (
            _record(f"p{LSOF_MAX_PID}")
            + _record("f0", "tREG")
            + _record("f1", "tREG")
        )
        cases = (
            (
                "row",
                _record("p42")
                + _record("f0", "tREG")
                + _record("f1", "tREG")
                + _record("f0", "tREG"),
                42,
                1,
                None,
            ),
            (
                "byte",
                byte_prefix + _record("f0", "tREG"),
                LSOF_MAX_PID,
                64,
                len(byte_prefix.encode("ascii")),
            ),
        )
        for cap, raw, pid, limit, max_stdout_bytes in cases:
            with self.subTest(cap=cap):
                output = await _tool(_RecordingExecutor(raw, stderr=""))(
                    pid,
                    limit=limit,
                    max_stdout_bytes=max_stdout_bytes,
                    context=ToolCallContext(),
                )

                assert isinstance(output, ShellFormattedResult)
                result = output.execution_result
                self.assertIs(result.status, ShellExecutionStatus.TOOL_ERROR)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stdout_bytes, 0)
                self.assertTrue(result.stdout_truncated)
                self.assertEqual(
                    result.error_message,
                    "lsof output was malformed",
                )

    async def test_public_result_combines_raw_and_logical_truncation(
        self,
    ) -> None:
        complete = (
            _record("p42") + _record("f3", "tREG") + _record("f4", "tPIPE")
        )
        cases = (
            (complete, False, 1, "42\t3\t-\tregular\t-\n"),
            (
                complete + "f5\0tIPv4",
                True,
                64,
                "42\t3\t-\tregular\t-\n42\t4\t-\tpipe\t-\n",
            ),
        )
        for raw, raw_truncated, limit, expected_stdout in cases:
            with self.subTest(
                raw_truncated=raw_truncated,
                limit=limit,
            ):
                output = await _tool(
                    _RecordingExecutor(
                        raw,
                        stdout_truncated=raw_truncated,
                    )
                )(
                    42,
                    limit=limit,
                    context=ToolCallContext(),
                )

                assert isinstance(output, ShellFormattedResult)
                result = output.execution_result
                self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
                self.assertEqual(result.stdout, expected_stdout)
                self.assertTrue(result.stdout_truncated)
                self.assertEqual(
                    result.stdout_bytes,
                    len(expected_stdout.encode("utf-8")),
                )

    async def test_completed_malformed_output_becomes_generic_tool_error(
        self,
    ) -> None:
        output = await _tool(
            _RecordingExecutor(
                _record("p42") + _record("f3", "tREG", "n/private"),
            )
        )(
            42,
            context=ToolCallContext(),
        )

        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertIs(result.error_code, ShellExecutionErrorCode.TOOL_ERROR)
        self.assertEqual(result.error_message, "lsof output was malformed")
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stdout_bytes, 0)
        self.assertEqual(result.stderr, REDACTED_LSOF_STDERR)
        self.assertEqual(result.metadata, {})
        self.assertEqual(result.generated_files, ())
        self.assertNotIn(PRIVATE_VALUE, output)

    async def test_nonzero_error_is_generic_even_when_private_error_is_absent(
        self,
    ) -> None:
        for include_error_message in (True, False):
            with self.subTest(include_error_message=include_error_message):
                output = await _tool(
                    _RecordingExecutor(
                        _valid_raw(),
                        status=ShellExecutionStatus.NONZERO_EXIT,
                        include_error_message=include_error_message,
                    )
                )(
                    42,
                    context=ToolCallContext(),
                )

                assert isinstance(output, ShellFormattedResult)
                result = output.execution_result
                self.assertIs(
                    result.status,
                    ShellExecutionStatus.NONZERO_EXIT,
                )
                self.assertEqual(result.error_message, "lsof exited non-zero")
                self.assertNotIn(PRIVATE_VALUE, output)

    async def test_exit_one_with_output_or_truncation_is_not_no_matches(
        self,
    ) -> None:
        cases = (
            ("", f"diagnostic {PRIVATE_VALUE}", False, False),
            ("p42", "", False, False),
            ("", "", True, False),
            ("", "", False, True),
        )
        for stdout, stderr, stdout_truncated, stderr_truncated in cases:
            with self.subTest(
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            ):
                output = await _tool(
                    _RecordingExecutor(
                        stdout,
                        status=ShellExecutionStatus.NONZERO_EXIT,
                        exit_code=1,
                        stderr=stderr,
                        stdout_truncated=stdout_truncated,
                        stderr_truncated=stderr_truncated,
                    )
                )(
                    42,
                    context=ToolCallContext(),
                )

                assert isinstance(output, ShellFormattedResult)
                result = output.execution_result
                self.assertIs(
                    result.status,
                    ShellExecutionStatus.NONZERO_EXIT,
                )
                self.assertIs(
                    result.error_code,
                    ShellExecutionErrorCode.NONZERO_EXIT,
                )
                self.assertEqual(result.error_message, "lsof exited non-zero")

    async def test_process_gate_returns_generic_public_denial(self) -> None:
        settings = ShellToolSettings(workspace_root=str(FIXTURE_ROOT))
        resolver = _RecordingResolver("/trusted/bin/lsof")
        executor = _RecordingExecutor(_valid_raw())
        tool = LsofTool(
            settings=settings,
            policy=ExecutionPolicy(settings=settings, resolver=resolver),
            executor=executor,
        )

        output = await tool(42, context=ToolCallContext())

        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertIs(
            result.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )
        self.assertEqual(result.error_message, "lsof was denied by policy")
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.metadata, {})
        self.assertIsNone(executor.spec)
        self.assertEqual(resolver.commands, [])

    async def test_strict_empty_exit_one_maps_to_no_matches(self) -> None:
        output = await _tool(
            _RecordingExecutor(
                "",
                status=ShellExecutionStatus.NONZERO_EXIT,
                stderr="",
                exit_code=1,
            )
        )(
            42,
            context=ToolCallContext(),
        )

        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.NO_MATCHES)
        self.assertEqual(result.exit_code, 1)
        self.assertIs(
            result.error_code,
            ShellExecutionErrorCode.NO_MATCHES,
        )
        self.assertIsNone(result.error_message)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.metadata, {})


def _request(
    pid: object,
    *,
    limit: object = _MISSING,
) -> ShellCommandRequest:
    options: dict[str, object] = {"pid": pid}
    if limit is not _MISSING:
        options["limit"] = limit
    return ShellCommandRequest(
        tool_name="shell.lsof",
        command="lsof",
        options=options,
        paths=(),
        cwd=None,
    )


def _policy(
    *,
    resolver: "_RecordingResolver | None" = None,
) -> ExecutionPolicy:
    settings = ShellToolSettings(
        workspace_root=str(FIXTURE_ROOT),
        allow_process_tools=True,
    )
    return ExecutionPolicy(
        settings=settings,
        resolver=resolver or _RecordingResolver("/trusted/bin/lsof"),
    )


def _tool(executor: "_RecordingExecutor") -> LsofTool:
    settings = ShellToolSettings(
        workspace_root=str(FIXTURE_ROOT),
        allow_process_tools=True,
    )
    return LsofTool(
        settings=settings,
        policy=ExecutionPolicy(
            settings=settings,
            resolver=_RecordingResolver("/trusted/bin/lsof"),
        ),
        executor=executor,
    )


def _record(*fields: str) -> str:
    return "\0".join(fields) + "\0\n"


def _valid_raw() -> str:
    return _record("p42") + _record("f3", "ar", "tREG")


class _RecordingResolver:
    def __init__(self, executable: str | None) -> None:
        self.executable = executable
        self.commands: list[str] = []

    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        self.commands.append(command.logical_id)
        return self.executable


class _RecordingExecutor:
    def __init__(
        self,
        stdout: str,
        *,
        status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
        stdout_truncated: bool = False,
        include_error_message: bool = True,
        stderr: str = f"diagnostic {PRIVATE_VALUE}",
        exit_code: int | None = None,
        stderr_truncated: bool = False,
    ) -> None:
        self.stdout = stdout
        self.status = status
        self.stdout_truncated = stdout_truncated
        self.include_error_message = include_error_message
        self.stderr = stderr
        self.exit_code = exit_code
        self.stderr_truncated = stderr_truncated
        self.spec: ExecutionSpec | None = None

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        self.spec = spec
        return ExecutionResult(
            backend="custom",
            tool_name="custom.lsof",
            command="lsof",
            argv=spec.argv,
            display_argv=spec.argv,
            cwd=spec.cwd,
            display_cwd=spec.display_cwd,
            status=self.status,
            exit_code=(
                self.exit_code
                if self.exit_code is not None
                else (
                    0 if self.status is ShellExecutionStatus.COMPLETED else 2
                )
            ),
            stdout=self.stdout,
            stderr=self.stderr,
            stdout_media_type=spec.stdout_media_type,
            output_kind=ShellOutputKind.TEXT,
            generated_files=(
                GeneratedFile(
                    display_path="private.txt",
                    media_type="text/plain",
                    suffix="txt",
                    bytes=7,
                    metadata={"private": PRIVATE_VALUE},
                ),
            ),
            stdout_bytes=len(self.stdout.encode("utf-8")),
            stderr_bytes=len(self.stderr.encode("utf-8")),
            stdout_truncated=self.stdout_truncated,
            stderr_truncated=self.stderr_truncated,
            duration_ms=1,
            error_code=(
                None
                if self.status is ShellExecutionStatus.COMPLETED
                else ShellExecutionErrorCode.NONZERO_EXIT
            ),
            error_message=(
                None
                if self.status is ShellExecutionStatus.COMPLETED
                or not self.include_error_message
                else f"private error {PRIVATE_VALUE}"
            ),
            metadata={"private": PRIVATE_VALUE},
        )


if __name__ == "__main__":
    main()
