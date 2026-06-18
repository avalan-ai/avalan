from dataclasses import dataclass
from json import loads as load_json
from pathlib import Path
from shlex import split as split_shell
from tomllib import loads as load_toml
from unittest import TestCase, main

REPO_ROOT = Path(__file__).parents[2]
DOC_ROOT = REPO_ROOT / "docs"
README = REPO_ROOT / "README.md"
DS4_DOC = DOC_ROOT / "DS4.md"
RECORDING_DOC = DOC_ROOT / "RECORDING.md"
AGENT_TOOL = DOC_ROOT / "examples" / "agent_tool.toml"
PYPI_SOURCE = DOC_ROOT / "examples" / "pypi_avalan_source.md"
GETTING_STARTED = DOC_ROOT / "tutorials" / "getting_started.ipynb"
TOUCHED_DOCS = (
    README,
    DS4_DOC,
    RECORDING_DOC,
    AGENT_TOOL,
    PYPI_SOURCE,
    GETTING_STARTED,
)

CALCULATOR_PROMPT = (
    "What is (4 + 6) and then that result times 5, divided by 2?"
)
STALE_STATS_GATE = (
    "only become visible in the live renderer when stats are enabled"
)


@dataclass(frozen=True)
class CliInvocation:
    shell: str
    program: str
    args: list[str]


class ThemeRecordingDocsTest(TestCase):
    def test_readme_calculator_examples_cover_fancy_and_basic(self) -> None:
        docs = _read(README)
        normalized = _normalize_space(docs)
        invocations = _cli_invocations(docs)

        math_agent_invocations = [
            invocation
            for invocation in invocations
            if _is_agent_run(invocation.args)
            and _option_values(invocation.args, "--tool")
            == ["math.calculator"]
            and CALCULATOR_PROMPT in invocation.shell
        ]

        fancy_invocations = [
            invocation
            for invocation in math_agent_invocations
            if "--theme" not in invocation.args
            and _option_values(invocation.args, "--backend") == ["mlx"]
            and "--stats" in invocation.args
            and "--display-events" in invocation.args
            and "--display-tools" in invocation.args
        ]
        basic_invocations = [
            invocation
            for invocation in math_agent_invocations
            if _option_values(invocation.args, "--theme") == ["basic"]
            and _option_values(invocation.args, "--backend") == ["mlx"]
            and "--display-tools" in invocation.args
            and "--stats" not in invocation.args
        ]

        self.assertTrue(fancy_invocations)
        self.assertTrue(basic_invocations)
        self.assertIn(
            "The local example below is the default Fancy/live-panel "
            "invocation.",
            normalized,
        )
        self.assertIn("Fancy is the default theme", normalized)
        self.assertIn(
            "Both the default Fancy example and the Basic example validate "
            "the result `25`.",
            normalized,
        )

    def test_ds4_docs_examples_parse_as_ds4_commands(self) -> None:
        docs = _read(DS4_DOC)
        self.assertNotIn("--tool-format dsml", docs)

        invocations = _cli_invocations(docs)
        ds4_invocations = [
            invocation
            for invocation in invocations
            if _is_agent_run(invocation.args) or _is_model_run(invocation.args)
        ]
        agent_invocations = [
            invocation
            for invocation in ds4_invocations
            if _is_agent_run(invocation.args)
        ]

        self.assertTrue(ds4_invocations)
        self.assertTrue(agent_invocations)
        for invocation in ds4_invocations:
            with self.subTest(command=invocation.shell):
                self.assertTrue(_uses_ds4_backend(invocation.args))

        for invocation in agent_invocations:
            with self.subTest(command=invocation.shell):
                self.assertEqual(
                    _option_values(invocation.args, "--tool"),
                    ["math.calculator"],
                )

    def test_recording_docs_cover_owner_and_coalesced_frames(self) -> None:
        docs = _read(RECORDING_DOC)
        normalized = _normalize_space(docs)
        invocations = _cli_invocations(docs)
        agent_invocations = [
            invocation
            for invocation in invocations
            if _is_agent_run(invocation.args)
        ]

        self.assertTrue(agent_invocations)
        self.assertTrue(
            any(
                invocation.program == ".venv/bin/avalan"
                for invocation in agent_invocations
            )
        )
        for invocation in agent_invocations:
            with self.subTest(command=invocation.shell):
                self.assertEqual(
                    _option_values(invocation.args, "--tool"),
                    ["math.calculator"],
                )
                self.assertTrue(_uses_ds4_backend(invocation.args))
                self.assertNotIn("--tool-format", invocation.args)

        for expected in (
            "`--stats` shows token generation statistics",
            "`--display-events` shows non-tool stream events",
            "`--display-tools` shows tool lifecycle details and results",
            "Tool and event diagnostics do not require `--stats`.",
            "one live owner",
            "render the active terminal view for all roles",
            "recording saves after the owner render",
            "coalesce to the latest live frame",
            "lossless canonical/public response surfaces remain intact",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, normalized)

    def test_touched_docs_do_not_gate_tools_or_events_on_stats(self) -> None:
        for path in (README, DS4_DOC, RECORDING_DOC):
            with self.subTest(path=path):
                docs = _read(path).lower()
                self.assertNotIn(STALE_STATS_GATE, docs)
                self.assertNotIn("only become visible", docs)

    def test_touched_docs_use_current_tool_names_and_formats(self) -> None:
        stale_snippets = (
            '--tool "calculator"',
            "--tool 'calculator'",
            "--tool calculator",
            "--tool-format dsml",
        )
        for path in TOUCHED_DOCS:
            docs = _read(path)
            for snippet in stale_snippets:
                with self.subTest(path=path, snippet=snippet):
                    self.assertNotIn(snippet, docs)

    def test_agent_tool_example_uses_namespaced_calculator(self) -> None:
        docs = _read(AGENT_TOOL)
        config = load_toml(docs)

        self.assertEqual(config["tool"]["enable"], ["math.calculator"])
        self.assertIn(CALCULATOR_PROMPT, docs)
        self.assertIn("The calculator result is 25.", docs)
        self.assertNotIn('--tool "calculator"', docs)
        self.assertNotIn("ephemereal", docs)

    def test_getting_started_tool_snippet_keeps_display_flags(self) -> None:
        docs = _notebook_markdown(GETTING_STARTED)
        invocations = _cli_invocations(docs)
        tool_invocations = [
            invocation
            for invocation in invocations
            if invocation.program == "poetry run avalan"
            and _is_agent_run(invocation.args)
            and _option_values(invocation.args, "--tool")
            == ["math.calculator"]
        ]
        display_invocations = [
            invocation
            for invocation in tool_invocations
            if "--display-events" in invocation.args
            and "--display-tools" in invocation.args
        ]

        self.assertGreaterEqual(len(tool_invocations), 2)
        self.assertEqual(len(display_invocations), 1)
        invocation = display_invocations[0]
        self.assertNotIn("--quiet", invocation.args)

    def test_pypi_source_calculator_command_uses_namespaced_tool(self) -> None:
        docs = _read(PYPI_SOURCE)
        invocations = [
            invocation
            for fence in _plain_fences(docs)
            if "avalan agent run" in fence
            for invocation in _cli_invocations_from_fence(fence)
        ]
        calculator_invocations = [
            invocation
            for invocation in invocations
            if _is_agent_run(invocation.args)
            and CALCULATOR_PROMPT in invocation.shell
        ]

        self.assertEqual(len(calculator_invocations), 1)
        invocation = calculator_invocations[0]
        self.assertEqual(invocation.program, "avalan")
        self.assertEqual(
            _option_values(invocation.args, "--tool"),
            ["math.calculator"],
        )
        self.assertIn("--memory-recent", invocation.args)
        self.assertIn("--stats", invocation.args)
        self.assertIn("--conversation", invocation.args)
        self.assertNotIn("--tool-format", invocation.args)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _notebook_markdown(path: Path) -> str:
    notebook = load_json(_read(path))
    cells = notebook["cells"]
    return "\n".join(
        "".join(cell["source"])
        for cell in cells
        if cell.get("cell_type") == "markdown"
    )


def _cli_invocations(docs: str) -> list[CliInvocation]:
    invocations: list[CliInvocation] = []
    for fence in _shell_fences(docs):
        invocations.extend(_cli_invocations_from_fence(fence))
    return invocations


def _shell_fences(docs: str) -> list[str]:
    return _fences(docs, {"bash", "sh"})


def _plain_fences(docs: str) -> list[str]:
    return _fences(docs, {""})


def _fences(docs: str, languages: set[str]) -> list[str]:
    blocks: list[str] = []
    in_fence = False
    fence_language = ""
    current: list[str] = []

    for line in docs.splitlines():
        if line.startswith("```") and not in_fence:
            in_fence = True
            fence_language = line[3:].strip()
            current = []
            continue
        if line == "```" and in_fence:
            if fence_language in languages:
                blocks.append("\n".join(current))
            in_fence = False
            fence_language = ""
            current = []
            continue
        if in_fence:
            current.append(line)

    return blocks


def _cli_invocations_from_fence(fence: str) -> list[CliInvocation]:
    invocations: list[CliInvocation] = []
    for command in _joined_shell_lines(fence):
        invocations.extend(_extract_cli_invocations(command))
    return invocations


def _joined_shell_lines(fence: str) -> list[str]:
    commands: list[str] = []
    current = ""

    for raw_line in fence.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        current = f"{current} {line}".strip() if current else line
        if continued or not _is_complete_shell_command(current):
            continue
        commands.append(current)
        current = ""

    if current:
        commands.append(current.strip())

    return commands


def _extract_cli_invocations(command: str) -> list[CliInvocation]:
    tokens = _split_command(command)
    invocations: list[CliInvocation] = []

    for embedded_command in _ttyrec_embedded_commands(tokens):
        invocations.extend(_extract_cli_invocations(embedded_command))

    index = 0
    while index < len(tokens):
        if tokens[index : index + 3] == ["poetry", "run", "avalan"]:
            invocations.append(
                CliInvocation(
                    shell=command,
                    program="poetry run avalan",
                    args=_command_args(tokens[index + 3 :]),
                )
            )
            index += 3
            continue

        token = tokens[index]
        if _is_avalan_token(token):
            invocations.append(
                CliInvocation(
                    shell=command,
                    program=token,
                    args=_command_args(tokens[index + 1 :]),
                )
            )
        index += 1

    return invocations


def _split_command(command: str) -> list[str]:
    try:
        return split_shell(command, comments=True)
    except ValueError as exc:
        raise AssertionError(
            f"Failed to parse shell command {command!r}: {exc}"
        ) from exc


def _is_complete_shell_command(command: str) -> bool:
    try:
        split_shell(command, comments=True)
    except ValueError:
        return False
    return True


def _ttyrec_embedded_commands(tokens: list[str]) -> list[str]:
    embedded: list[str] = []
    for index, token in enumerate(tokens):
        if Path(token).name != "ttyrec":
            continue
        try:
            exec_index = tokens.index("-e", index + 1)
        except ValueError:
            continue
        if exec_index + 1 < len(tokens):
            embedded.append(tokens[exec_index + 1])
    return embedded


def _is_avalan_token(token: str) -> bool:
    return Path(token).name in {"avalan", "avl"}


def _command_args(tokens: list[str]) -> list[str]:
    args: list[str] = []
    for token in tokens:
        if token in {"|", ";", "&&", "||", ")"}:
            break
        args.append(token)
    return args


def _is_agent_run(args: list[str]) -> bool:
    return args[:2] == ["agent", "run"]


def _is_model_run(args: list[str]) -> bool:
    return args[:2] == ["model", "run"]


def _option_values(args: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, arg in enumerate(args):
        if arg == option and index + 1 < len(args):
            values.append(args[index + 1])
        elif arg.startswith(option + "="):
            values.append(arg.split("=", 1)[1])
    return values


def _uses_ds4_backend(args: list[str]) -> bool:
    if "backend=ds4" in " ".join(args):
        return True
    return _option_values(args, "--backend") == ["ds4"]


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    main()
