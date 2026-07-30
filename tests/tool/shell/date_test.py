from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.tool.shell import (
    ExecutionPolicy,
    PathOperand,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellPolicyDenied,
    ShellToolSettings,
)
from avalan.tool.shell.commands.date import _custom_format_option


class DatePolicyTest(IsolatedAsyncioTestCase):
    def test_empty_custom_format_guard_is_defensive(self) -> None:
        with self.assertRaisesRegex(
            ShellPolicyDenied,
            "custom_format must not be empty",
        ) as raised:
            _custom_format_option({"custom_format": ""})

        self.assertIs(
            raised.exception.error_code,
            ShellExecutionErrorCode.INVALID_OPTION,
        )

    async def test_builds_only_fixed_portable_argv(self) -> None:
        cases = (
            ({}, ("date",)),
            ({"format": "default"}, ("date",)),
            ({"custom_format": None}, ("date",)),
            ({"utc": True}, ("date", "-u")),
            ({"format": "date"}, ("date", "+%Y-%m-%d")),
            ({"format": "time"}, ("date", "+%H:%M:%S")),
            (
                {"format": "iso8601"},
                ("date", "+%Y-%m-%dT%H:%M:%S%z"),
            ),
            ({"format": "unix"}, ("date", "+%s")),
            (
                {"utc": True, "format": "iso8601"},
                ("date", "-u", "+%Y-%m-%dT%H:%M:%S%z"),
            ),
        )
        for options, expected in cases:
            with self.subTest(options=options):
                resolver = _RecordingResolver()
                policy = ExecutionPolicy(
                    settings=ShellToolSettings(),
                    resolver=resolver,
                )

                spec = await policy.normalize(_request(options))

                self.assertEqual(spec.argv, expected)
                self.assertEqual(spec.display_argv, expected)
                self.assertEqual(spec.executable, "/trusted/bin/date")
                self.assertEqual(resolver.commands, ["date"])

    async def test_builds_validated_custom_format_argv(self) -> None:
        all_directives = (
            "%%|%C|%d|%D|%e|%F|%g|%G|%H|%I|%j|%m|%M|%R|%s|%S|%T|"
            "%u|%U|%V|%w|%W|%y|%Y|%z"
        )
        cases = (
            (
                {"custom_format": all_directives},
                ("date", f"+{all_directives}"),
            ),
            (
                {
                    "format": "default",
                    "custom_format": "stamp=%Y-%m-%d %H:%M:%S %z",
                },
                ("date", "+stamp=%Y-%m-%d %H:%M:%S %z"),
            ),
            (
                {
                    "utc": True,
                    "custom_format": "--date=%Y %% [UTC]",
                },
                ("date", "-u", "+--date=%Y %% [UTC]"),
            ),
            (
                {"custom_format": "x" * 128},
                ("date", f"+{'x' * 128}"),
            ),
            (
                {"custom_format": " "},
                ("date", "+ "),
            ),
        )
        for options, expected in cases:
            with self.subTest(options=options):
                resolver = _RecordingResolver()
                policy = ExecutionPolicy(
                    settings=ShellToolSettings(),
                    resolver=resolver,
                )

                spec = await policy.normalize(_request(options))

                self.assertEqual(spec.argv, expected)
                self.assertEqual(spec.display_argv, expected)
                self.assertEqual(resolver.commands, ["date"])

    async def test_rejects_unknown_options_types_and_formats_before_resolve(
        self,
    ) -> None:
        cases = (
            (
                {"unknown": True},
                "unknown date option: unknown",
            ),
            (
                {"utc": "true"},
                "utc must be a boolean",
            ),
            (
                {"format": "rfc3339"},
                "format is unsupported",
            ),
            (
                {"format": 1},
                "format is unsupported",
            ),
        )
        for options, message in cases:
            with self.subTest(options=options):
                resolver = _RecordingResolver()
                policy = ExecutionPolicy(
                    settings=ShellToolSettings(),
                    resolver=resolver,
                )

                with self.assertRaisesRegex(
                    ShellPolicyDenied,
                    message,
                ) as raised:
                    await policy.normalize(_request(options))

                self.assertIs(
                    raised.exception.error_code,
                    ShellExecutionErrorCode.INVALID_OPTION,
                )
                self.assertEqual(resolver.commands, [])

    async def test_rejects_invalid_custom_formats_before_resolve(self) -> None:
        invalid_option_cases = (
            ({"custom_format": 1}, "must be a string"),
            ({"custom_format": True}, "must be a string"),
            ({"custom_format": b"%Y"}, "unsupported type"),
            ({"custom_format": ""}, "must not be empty"),
            ({"custom_format": "\t"}, "only printable ASCII"),
            ({"custom_format": "\r"}, "only printable ASCII"),
            ({"custom_format": "\n"}, "only printable ASCII"),
            ({"custom_format": "\x00"}, "only printable ASCII"),
            ({"custom_format": "\x1f"}, "only printable ASCII"),
            ({"custom_format": "\x7f"}, "only printable ASCII"),
            ({"custom_format": "café %Y"}, "only printable ASCII"),
            ({"custom_format": "clock 🕰"}, "only printable ASCII"),
            ({"custom_format": "value=%"}, "dangling %"),
            ({"custom_format": "%a"}, "unsupported directive"),
            ({"custom_format": "%n"}, "unsupported directive"),
            ({"custom_format": "%t"}, "unsupported directive"),
            ({"custom_format": "%Z"}, "unsupported directive"),
            ({"custom_format": "%-d"}, "unsupported directive"),
            ({"custom_format": "%_d"}, "unsupported directive"),
            ({"custom_format": "%0d"}, "unsupported directive"),
            ({"custom_format": "%^B"}, "unsupported directive"),
            ({"custom_format": "%#m"}, "unsupported directive"),
            ({"custom_format": "%3Y"}, "unsupported directive"),
            ({"custom_format": "%Ec"}, "unsupported directive"),
            ({"custom_format": "%Od"}, "unsupported directive"),
            ({"custom_format": "%N"}, "unsupported directive"),
            ({"custom_format": "%:z"}, "unsupported directive"),
            ({"custom_format": "%+%Y"}, "unsupported directive"),
            (
                {"format": "date", "custom_format": "%Y"},
                "requires format=default",
            ),
        )
        cases = tuple(
            (
                options,
                message,
                ShellExecutionErrorCode.INVALID_OPTION,
            )
            for options, message in invalid_option_cases
        ) + (
            (
                {"custom_format": "x" * 129},
                "too large",
                ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            ),
        )
        for options, message, error_code in cases:
            with self.subTest(options=options):
                resolver = _RecordingResolver()
                policy = ExecutionPolicy(
                    settings=ShellToolSettings(),
                    resolver=resolver,
                )

                with self.assertRaisesRegex(
                    ShellPolicyDenied,
                    message,
                ) as raised:
                    await policy.normalize(_request(options))

                self.assertIs(raised.exception.error_code, error_code)
                self.assertEqual(resolver.commands, [])

    async def test_rejects_paths_before_resolve(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text("visible\n", encoding="utf-8")
            resolver = _RecordingResolver()
            policy = ExecutionPolicy(
                settings=ShellToolSettings(workspace_root=str(root)),
                resolver=resolver,
            )

            with self.assertRaisesRegex(
                ShellPolicyDenied,
                "date does not accept paths",
            ) as raised:
                await policy.normalize(
                    _request(
                        {},
                        paths=(
                            PathOperand(
                                name="input",
                                path="visible.txt",
                                kind="file",
                                access="read",
                            ),
                        ),
                    )
                )

        self.assertIs(
            raised.exception.error_code,
            ShellExecutionErrorCode.INVALID_OPTION,
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
            await policy.normalize(_request({}, stdin=b"forged"))

        self.assertIs(
            raised.exception.error_code,
            ShellExecutionErrorCode.STDIN_DENIED,
        )
        self.assertEqual(resolver.commands, [])


def _request(
    options: dict[str, object],
    *,
    paths: tuple[PathOperand, ...] = (),
    stdin: bytes | None = None,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.date",
        command="date",
        options=options,
        paths=paths,
        cwd=None,
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
        return "/trusted/bin/date"


if __name__ == "__main__":
    main()
