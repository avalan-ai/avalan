from pathlib import Path
from re import DOTALL, IGNORECASE, Pattern, compile
from unittest import TestCase, main

ROOT = Path(__file__).parents[2]
DOC_ROOT = ROOT / "docs"
EXAMPLE_ROOT = DOC_ROOT / "examples"
PLAYGROUND_EXAMPLE_ROOT = EXAMPLE_ROOT / "playground"
README = ROOT / "README.md"

SDK_STREAMING_EXAMPLES = (
    EXAMPLE_ROOT / "text_generation.py",
    EXAMPLE_ROOT / "text_generation_anthropic.py",
    EXAMPLE_ROOT / "text_generation_google.py",
    EXAMPLE_ROOT / "text_generation_groq.py",
    EXAMPLE_ROOT / "text_generation_mlxlm.py",
    EXAMPLE_ROOT / "text_generation_ollama.py",
    EXAMPLE_ROOT / "text_generation_openai.py",
    EXAMPLE_ROOT / "text_generation_vllm.py",
)
PROTOCOL_CHUNK_EXAMPLES = (
    EXAMPLE_ROOT / "openai_client_agent_messi.py",
    EXAMPLE_ROOT / "openai_client_agent_tool.py",
)
HISTORICAL_PYPI_SNAPSHOT = EXAMPLE_ROOT / "pypi_avalan_source.md"
SKIPPED_ACTIVE_EXAMPLE_NAMES = {
    "openai_client_agent_messi.py",
    "openai_client_agent_tool.py",
    "pypi_avalan_source.md",
    "token_classification.py",
}
SKIPPED_ACTIVE_EXAMPLE_DIRS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
FORBIDDEN_STREAMING_PATTERNS: dict[str, Pattern[str]] = {
    "hosted TextGenerationModel URI": compile(
        r"\bTextGenerationModel\(\s*[\"']ai://"
    ),
    "token loop": compile(r"\basync\s+for\s+token\b"),
    "legacy token attribute": compile(r"\btoken\.token\b"),
    "non-avalan streaming flag": compile(r"\bstream=True\b"),
    "token streaming phrase": compile(
        r"\b(?:token(?: and tool)? streaming|"
        r"streaming tokens|streams tokens)\b",
        flags=IGNORECASE,
    ),
    "legacy stream entity": compile(
        r"\b(?:ReasoningToken|StreamToken|TokenDetail|ToolCallToken)\b"
    ),
    "legacy token import": compile(
        r"from avalan\.entities import [^\n]*\bToken\b"
    ),
    "legacy multiline token import": compile(
        r"from avalan\.entities import \([^)]+\bToken\b",
        flags=DOTALL,
    ),
    "legacy adapter recommendation": compile(
        r"\b(?:enable|keep|prefer|recommend|rely on|use)\b"
        r"[^\n.]{0,80}\blegacy[- ]adapter\b",
        flags=IGNORECASE,
    ),
    "shim recommendation": compile(
        r"\b(?:enable|keep|prefer|recommend|rely on|use)\b"
        r"[^\n.]{0,80}\b(?:compatibility|current|ingestion)[- ]shim\b",
        flags=IGNORECASE,
    ),
}
REQUIRED_SDK_STREAMING_SNIPPETS = {
    "missing canonical stream import": (
        "from avalan.model.stream import CanonicalStreamItem, StreamItemKind"
    ),
    "missing canonical item loop": "async for item in await lm(",
    "missing canonical item assertion": (
        "assert isinstance(item, CanonicalStreamItem)"
    ),
    "missing answer delta kind check": (
        "item.kind is StreamItemKind.ANSWER_DELTA"
    ),
    "missing text delta check": "item.text_delta is not None",
    "missing text delta output": "print(item.text_delta",
    "missing explicit async generator setting": "use_async_generator=True",
}


class StreamingDocsTest(TestCase):
    def test_sdk_streaming_examples_consume_canonical_items(self) -> None:
        for path in SDK_STREAMING_EXAMPLES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertSetEqual(
                    _sdk_streaming_example_violations(text), set()
                )

    def test_readme_sdk_examples_and_breaking_note_are_canonical(
        self,
    ) -> None:
        text = README.read_text(encoding="utf-8")

        self.assertGreaterEqual(
            text.count(
                "from avalan.model.stream import "
                "CanonicalStreamItem, StreamItemKind"
            ),
            3,
        )
        self.assertGreaterEqual(
            text.count("async for item in await model("), 3
        )
        self.assertGreaterEqual(
            text.count("item.kind is StreamItemKind.ANSWER_DELTA"),
            3,
        )
        self.assertGreaterEqual(text.count("item.text_delta is not None"), 3)
        self.assertNotIn('TextGenerationModel("ai://', text)
        self.assertGreaterEqual(text.count("OpenAIModel("), 2)
        self.assertIn("use_async_generator=False", text)
        self.assertGreaterEqual(text.count("use_async_generator=True"), 3)
        self.assertIn(
            "legacy streaming items are rejected by Avalan-owned runtime APIs "
            "instead of converted",
            text,
        )

    def test_active_docs_do_not_recommend_legacy_streaming_shapes(
        self,
    ) -> None:
        for path in _active_streaming_doc_paths():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertSetEqual(_streaming_pattern_violations(text), set())

    def test_streaming_guard_detects_stale_sdk_fixtures(self) -> None:
        fixtures = {
            "hosted TextGenerationModel URI": (
                'TextGenerationModel("ai://env:OPENAI_API_KEY@openai/gpt-4o")',
                {"hosted TextGenerationModel URI"},
            ),
            "non-avalan streaming flag": (
                "client.chat.completions.create(stream=True)",
                {"non-avalan streaming flag"},
            ),
            "token loop": (
                (
                    "async for token in await lm('hello'):\n"
                    "    print(token.token)"
                ),
                {"token loop", "legacy token attribute"},
            ),
            "missing explicit async generator setting": (
                (
                    "from avalan.model.stream import "
                    "CanonicalStreamItem, StreamItemKind\n\n"
                    "async for item in await lm(\n"
                    "    'hello',\n"
                    "    settings=GenerationSettings(max_new_tokens=8),\n"
                    "):\n"
                    "    assert isinstance(item, CanonicalStreamItem)\n"
                    "    if (\n"
                    "        item.kind is StreamItemKind.ANSWER_DELTA\n"
                    "        and item.text_delta is not None\n"
                    "    ):\n"
                    "        print(item.text_delta, end='', flush=True)\n"
                ),
                {"missing explicit async generator setting"},
            ),
        }

        for name, (text, expected) in fixtures.items():
            with self.subTest(name=name):
                self.assertTrue(
                    expected.issubset(_sdk_streaming_example_violations(text))
                )

    def test_protocol_chunk_examples_are_exempt_from_active_guard(
        self,
    ) -> None:
        active_paths = set(_active_streaming_doc_paths())
        for path in PROTOCOL_CHUNK_EXAMPLES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn(path, active_paths)
                self.assertIn(
                    "non-avalan streaming flag",
                    _streaming_pattern_violations(text),
                )
                self.assertIn("async for chunk in stream:", text)
                self.assertIn("chunk.choices[0].delta.content", text)

    def test_historical_pypi_snapshot_is_exempt_from_active_guard(
        self,
    ) -> None:
        text = HISTORICAL_PYPI_SNAPSHOT.read_text(encoding="utf-8")

        self.assertNotIn(
            HISTORICAL_PYPI_SNAPSHOT, set(_active_streaming_doc_paths())
        )
        self.assertIn("token loop", _streaming_pattern_violations(text))

    def test_playground_examples_are_exempt_from_active_guard(self) -> None:
        playground_example = PLAYGROUND_EXAMPLE_ROOT / "agent_team.py"
        generated_dependency = (
            PLAYGROUND_EXAMPLE_ROOT
            / "real_estate"
            / "client"
            / ".venv"
            / "lib"
            / "python3.11"
            / "site-packages"
            / "httpx"
            / "_client.py"
        )
        active_paths = set(_active_streaming_doc_paths())

        self.assertNotIn(playground_example, active_paths)
        self.assertNotIn(generated_dependency, active_paths)
        self.assertIn(
            "token loop",
            _streaming_pattern_violations(
                playground_example.read_text(encoding="utf-8")
            ),
        )

    def test_protocol_examples_remain_openai_chunk_consumers(self) -> None:
        for path in PROTOCOL_CHUNK_EXAMPLES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIn("async for chunk in stream:", text)
                self.assertIn("chunk.choices[0].delta.content", text)
                self.assertNotIn("CanonicalStreamItem", text)
                self.assertNotIn("StreamItemKind", text)

    def test_serving_docs_describe_canonical_protocol_projection(
        self,
    ) -> None:
        text = " ".join(README.read_text(encoding="utf-8").split())
        for phrase in (
            (
                "serving surfaces project that stream to OpenAI-compatible SSE"
                " chunks/events, MCP notifications/resources, or A2A"
                " task/artifact events"
            ),
            (
                "Chat completions and Responses requests project Avalan's "
                "canonical stream to protocol-specific chunks or SSE events"
            ),
            (
                "MCP server projects canonical stream items to MCP progress "
                "notifications, resources, and final tool results"
            ),
            (
                "A2A surface projects canonical stream items to A2A task, "
                "status, artifact, tool-call, and intermediate-output events"
            ),
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(" ".join(phrase.split()), text)


def _active_streaming_doc_paths() -> tuple[Path, ...]:
    docs = (README,) + tuple(sorted(path for path in DOC_ROOT.glob("*.md")))
    examples = tuple(
        sorted(
            path
            for path in EXAMPLE_ROOT.rglob("*")
            if _is_active_streaming_example_path(path)
        )
    )
    return docs + examples


def _is_active_streaming_example_path(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix in {".md", ".py"}
        and path.name not in SKIPPED_ACTIVE_EXAMPLE_NAMES
        and not path.is_relative_to(PLAYGROUND_EXAMPLE_ROOT)
        and SKIPPED_ACTIVE_EXAMPLE_DIRS.isdisjoint(path.parts)
    )


def _streaming_pattern_violations(text: str) -> set[str]:
    return {
        name
        for name, pattern in FORBIDDEN_STREAMING_PATTERNS.items()
        if pattern.search(text)
    }


def _sdk_streaming_example_violations(text: str) -> set[str]:
    violations = _streaming_pattern_violations(text)
    for name, snippet in REQUIRED_SDK_STREAMING_SNIPPETS.items():
        if snippet not in text:
            violations.add(name)
    return violations


if __name__ == "__main__":
    main()
