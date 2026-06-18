import re
import shlex
import sys
from argparse import ArgumentParser, _SubParsersAction
from io import StringIO
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

from avalan.cli.__main__ import CLI

DOC_ROOT = Path(__file__).parents[2] / "docs"
CLI_DOC = DOC_ROOT / "CLI.md"
THEME_USAGE_FRAGMENT = "[--locale LOCALE] [--theme {fancy,basic}]"
THEME_OPTION_FRAGMENT = "--theme {fancy,basic}"
THEME_HELP_FRAGMENT = "Theme to use (default is fancy)"
DISPLAY_EVENTS_DESCRIPTION = (
    "Show non-tool stream events when an orchestrator or agent is involved."
)
DISPLAY_TOOLS_DESCRIPTION = (
    "Show tool lifecycle details for agent or orchestrator runs."
)
STATS_DESCRIPTION = "Show token generation statistics for streaming output"
STALE_DISPLAY_TOOLS_DESCRIPTION = (
    "If --display-events is specified and there's an orchestrator / agent "
    "involved, show the events panel."
)


class CliDocsTest(TestCase):
    def setUp(self) -> None:
        with patch.object(sys, "argv", ["avalan"]):
            self.parser = CLI._create_parser(
                "cpu", "/cache", "/locale", "en_US"
            )

    def test_global_theme_is_documented_in_help_blocks(self) -> None:
        docs = _read_docs()
        help_blocks = _plain_help_blocks(docs)
        self.assertTrue(help_blocks)

        for block in help_blocks:
            if (
                "--cache-dir CACHE_DIR" not in block
                or "--locale LOCALE" not in block
            ):
                continue
            with self.subTest(prog=_prog_from_help(block)):
                normalized = _normalize_space(block)
                self.assertIn(THEME_USAGE_FRAGMENT, normalized)
                self.assertIn(THEME_OPTION_FRAGMENT, block)
                self.assertIn(THEME_HELP_FRAGMENT, block)

    def test_representative_help_blocks_match_parser_theme_help(
        self,
    ) -> None:
        docs_by_prog = _help_blocks_by_prog(_plain_help_blocks(_read_docs()))

        for prog in (
            "avalan",
            "avalan agent run",
            "avalan model run",
            "avalan flow run",
        ):
            with self.subTest(prog=prog):
                docs_help = docs_by_prog[prog]
                parser_help = _find_parser(self.parser, prog).format_help()

                for fragment in (
                    THEME_USAGE_FRAGMENT,
                    THEME_OPTION_FRAGMENT,
                    THEME_HELP_FRAGMENT,
                ):
                    self.assertIn(
                        _normalize_space(fragment),
                        _normalize_space(docs_help),
                    )
                    self.assertIn(
                        _normalize_space(fragment),
                        _normalize_space(parser_help),
                    )

    def test_agent_run_docs_cover_current_tool_and_backend_help(self) -> None:
        docs_by_prog = _help_blocks_by_prog(_plain_help_blocks(_read_docs()))
        docs_help = _normalize_space(docs_by_prog["avalan agent run"])
        parser_help = _normalize_space(
            _find_parser(self.parser, "avalan agent run").format_help()
        )

        expected_fragments = (
            "[--display-events] [--stats]",
            "--tool-format {json,react,bracket,openai,harmony,dsml}",
            "--tool-recovery-format",
            "--ds4-ctx DS4_CTX",
            "--tool-graph-file TOOL_GRAPH_FILE",
            "--tool-shell-workspace-root TOOL_SHELL_WORKSPACE_ROOT",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                normalized = _normalize_space(fragment)
                self.assertIn(normalized, parser_help)
                self.assertIn(normalized, docs_help)

    def test_agent_and_model_run_docs_cover_current_parser_options(
        self,
    ) -> None:
        docs_by_prog = _help_blocks_by_prog(_plain_help_blocks(_read_docs()))
        parser = _docs_default_parser()

        for prog in ("avalan agent run", "avalan model run"):
            with self.subTest(prog=prog):
                docs_metadata = _help_metadata(docs_by_prog[prog])
                parser_metadata = _help_metadata(
                    _find_parser(parser, prog).format_help()
                )

                self.assertEqual(set(docs_metadata), set(parser_metadata))

    def test_agent_run_docs_match_current_parser_help_metadata(self) -> None:
        docs_by_prog = _help_blocks_by_prog(_plain_help_blocks(_read_docs()))
        parser = _docs_default_parser()

        docs_metadata = _help_metadata(docs_by_prog["avalan agent run"])
        parser_metadata = _help_metadata(
            _find_parser(parser, "avalan agent run").format_help()
        )

        self.assertEqual(docs_metadata, parser_metadata)

    def test_display_tool_help_uses_lifecycle_wording(self) -> None:
        docs = _read_docs()
        self.assertNotIn(STALE_DISPLAY_TOOLS_DESCRIPTION, docs)

        docs_by_prog = _help_blocks_by_prog(_plain_help_blocks(docs))
        for prog in ("avalan agent run", "avalan model run"):
            with self.subTest(prog=prog):
                block = docs_by_prog[prog]
                display_tools = _option_description(block, "--display-tools")

                self.assertIn(DISPLAY_TOOLS_DESCRIPTION, display_tools)
                self.assertNotIn("--display-events", display_tools)
                self.assertNotIn("--stats", display_tools)
                self.assertNotIn("events panel", display_tools)
                self.assertIn(
                    DISPLAY_EVENTS_DESCRIPTION,
                    _option_description(block, "--display-events"),
                )
                self.assertIn(
                    STATS_DESCRIPTION,
                    _option_description(block, "--stats"),
                )

    def test_bash_snippets_parse_as_cli_commands(self) -> None:
        commands = list(_snippet_commands(_bash_fences(_read_docs())))
        self.assertTrue(commands)
        self.assertTrue(
            any(
                command.startswith("avalan model run ") for command in commands
            )
        )
        self.assertIn("avalan model run MODEL --theme basic", commands)
        self.assertTrue(
            any(command.startswith("avalan flow run ") for command in commands)
        )
        self.assertTrue(
            any(
                command.startswith("poetry run avalan ")
                for command in commands
            )
        )

        for command in commands:
            with self.subTest(command=command):
                args = _parser_args(command)
                assert args is not None
                stderr = StringIO()
                stdout = StringIO()
                with (
                    patch.object(sys, "stderr", stderr),
                    patch.object(sys, "stdout", stdout),
                ):
                    try:
                        self.parser.parse_args(args)
                    except SystemExit as exc:
                        if exc.code != 0:
                            self.fail(
                                f"Failed to parse {command!r}: "
                                f"{stderr.getvalue()}"
                            )


def _read_docs() -> str:
    return CLI_DOC.read_text(encoding="utf-8")


def _docs_default_parser() -> ArgumentParser:
    with patch.object(sys, "argv", ["avalan"]):
        return CLI._create_parser(
            "cpu",
            "/root/.cache/huggingface/hub",
            "/workspace/avalan/locale",
            "en_US",
        )


def _plain_help_blocks(docs: str) -> list[str]:
    blocks: list[str] = []
    for block in _fenced_blocks(docs, ""):
        if block.startswith("usage:"):
            blocks.append(block)
    return blocks


def _bash_fences(docs: str) -> list[str]:
    return _fenced_blocks(docs, "bash")


def _fenced_blocks(docs: str, language: str) -> list[str]:
    blocks: list[str] = []
    in_fence = False
    fence_language = ""
    current: list[str] = []
    for line in docs.splitlines():
        if line.startswith("```") and not in_fence:
            in_fence = True
            fence_language = line[3:]
            current = []
            continue
        if line == "```" and in_fence:
            if fence_language == language:
                blocks.append("\n".join(current))
            in_fence = False
            fence_language = ""
            current = []
            continue
        if in_fence:
            current.append(line)
    return blocks


def _snippet_commands(fences: list[str]) -> list[str]:
    commands: list[str] = []
    for fence in fences:
        for line in _joined_bash_lines(fence):
            args = _parser_args(line)
            if args is not None:
                commands.append(line)
    return commands


def _joined_bash_lines(fence: str) -> list[str]:
    commands: list[str] = []
    current = ""
    for raw_line in fence.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("\\"):
            current += line[:-1].rstrip() + " "
            continue
        commands.append((current + line).strip())
        current = ""
    if current:
        commands.append(current.strip())
    return commands


def _parser_args(command: str) -> list[str] | None:
    tokens = shlex.split(command)
    if not tokens:
        return None
    if tokens[:3] == ["poetry", "run", "avalan"]:
        return tokens[3:]
    if tokens[0] in {"avalan", "avl"}:
        return tokens[1:]
    return None


def _help_blocks_by_prog(blocks: list[str]) -> dict[str, str]:
    return {_prog_from_help(block): block for block in blocks}


def _prog_from_help(block: str) -> str:
    first_line = block.splitlines()[0]
    tokens = first_line.removeprefix("usage: ").split()
    prog_tokens: list[str] = []
    for token in tokens:
        if token.startswith("[") or token.startswith("{") or token == "...":
            break
        prog_tokens.append(token)
    return " ".join(prog_tokens)


def _find_parser(parser: ArgumentParser, prog: str) -> ArgumentParser:
    stack = [parser]
    while stack:
        candidate = stack.pop()
        if candidate.prog == prog:
            return candidate
        for action in candidate._actions:
            if isinstance(action, _SubParsersAction):
                stack.extend(action.choices.values())
    raise AssertionError(f"Parser {prog!r} was not found.")


def _option_description(block: str, option: str) -> str:
    lines = block.splitlines()
    pattern = re.compile(rf"^\s{{2}}{re.escape(option)}(?:\s|$)")
    for index, line in enumerate(lines):
        if not pattern.match(line):
            continue
        description = [line.split(option, 1)[1].strip()]
        for continuation in lines[index + 1 :]:
            if not continuation.startswith("                        "):
                break
            description.append(continuation.strip())
        return _normalize_space(" ".join(description))
    raise AssertionError(f"Option {option!r} was not found.")


def _help_metadata(block: str) -> dict[str, tuple[str, str]]:
    metadata: dict[str, tuple[str, str]] = {}
    active = False
    keys: tuple[str, ...] = ()
    invocation_parts: list[str] = []
    description_parts: list[str] = []

    for line in block.splitlines():
        if line in {"positional arguments:", "options:"}:
            _store_help_entry(
                metadata, keys, invocation_parts, description_parts
            )
            active = True
            keys = ()
            invocation_parts = []
            description_parts = []
            continue
        if _is_help_section_header(line):
            _store_help_entry(
                metadata, keys, invocation_parts, description_parts
            )
            active = False
            keys = ()
            invocation_parts = []
            description_parts = []
            continue
        if not active:
            continue
        if line.startswith("  ") and not line.startswith(
            "                        "
        ):
            _store_help_entry(
                metadata, keys, invocation_parts, description_parts
            )
            invocation, description = _split_help_entry_line(line[2:])
            keys = _help_entry_keys(invocation)
            invocation_parts = [invocation]
            description_parts = [description] if description else []
            continue
        if keys and line.startswith("                        "):
            description_parts.append(line.strip())
            continue
        if keys and line and not line.startswith(" "):
            invocation_parts.append(line.strip())

    _store_help_entry(metadata, keys, invocation_parts, description_parts)
    return metadata


def _store_help_entry(
    metadata: dict[str, tuple[str, str]],
    keys: tuple[str, ...],
    invocation_parts: list[str],
    description_parts: list[str],
) -> None:
    if not keys:
        return
    invocation = _normalize_space(" ".join(invocation_parts))
    description = _normalize_space(" ".join(description_parts))
    for key in keys:
        metadata[key] = (invocation, description)


def _split_help_entry_line(line: str) -> tuple[str, str]:
    parts = re.split(r"\s{2,}", line.strip(), maxsplit=1)
    description = parts[1] if len(parts) == 2 else ""
    return parts[0], description


def _help_entry_keys(invocation: str) -> tuple[str, ...]:
    if invocation.startswith("-"):
        keys = [
            token
            for token in invocation.replace(",", " ").split()
            if token.startswith("-")
        ]
        return tuple(keys)
    return (invocation.split()[0],)


def _is_help_section_header(line: str) -> bool:
    return bool(line) and not line.startswith(" ") and line.endswith(":")


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    main()
