import unittest
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from logging import Logger, getLogger
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np

from avalan.cli.download import create_live_tqdm_class, tqdm_rich_progress
from avalan.cli.theme import Theme, TokenRenderFrame, TokenRenderState
from avalan.cli.theme.fancy import FancyTheme
from avalan.entities import (
    HubCache,
    HubCacheDeletion,
    ImageEntity,
    Model,
    ModelConfig,
    SearchMatch,
    SentenceTransformerModelConfig,
    Similarity,
    TextPartition,
    Token,
    TokenizerConfig,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
    User,
)
from avalan.event import Event, EventType


def _gettext(message: str) -> str:
    return message


def _ngettext(singular: str, plural: str, n: int) -> str:
    return singular if n == 1 else plural


def _theme() -> Theme:
    return Theme(_gettext, _ngettext)


def _model() -> Model:
    now = datetime(2024, 1, 1)
    return Model(
        id="model-id",
        parameters=None,
        parameter_types=None,
        inference=None,
        library_name=None,
        license=None,
        pipeline_tag=None,
        tags=[],
        architectures=None,
        model_type=None,
        auto_model=None,
        processor=None,
        gated=False,
        private=False,
        disabled=False,
        last_downloads=0,
        downloads=0,
        likes=0,
        ranking=None,
        author="author",
        created_at=now,
        updated_at=now,
    )


def _tool_call() -> ToolCall:
    return ToolCall(
        id="call-12345678",
        name="calc[bold]\x1b[31m",
        arguments={
            "x": 1,
            "api_key": "secret-value",
            "nested": {"password": "hidden"},
        },
    )


def _diagnostic() -> ToolCallDiagnostic:
    return ToolCallDiagnostic(
        id="diagnostic-123",
        call_id="call-12345678",
        requested_name="calc",
        canonical_name="calculator",
        code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
        stage=ToolCallDiagnosticStage.VALIDATE,
        message="bad [red]\x1b[31margument",
        details={
            "token": "raw-token",
            "limit": 5,
            "items": list(range(8)),
        },
    )


@dataclass
class BrokenData:
    value: str

    def __getattribute__(self, name: str) -> object:
        if name == "value":
            raise RuntimeError("cannot read")
        return super().__getattribute__(name)


class BrokenString:
    def __str__(self) -> str:
        raise RuntimeError("cannot stringify")


@dataclass
class ManyFields:
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int
    g: int


class BrokenMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("cannot iterate")

    def __len__(self) -> int:
        return 1

    def __str__(self) -> str:
        return "broken mapping"


class ThemePropertyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = _theme()

    def test_default_properties(self) -> None:
        self.assertEqual(
            self.theme.icons,
            {"agent_output": "", "user_input": ""},
        )
        self.assertEqual(self.theme.quantity_data, [])
        self.assertEqual(self.theme.spinners, {})
        self.assertEqual(self.theme.stylers, {})
        self.assertEqual(self.theme.styles, {})


class ThemeFlowProgressTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = _theme()

    def test_flow_run_progress_messages_cover_node_events(self) -> None:
        cases = {
            "flow_node_started": "Running node_a (attempt 2).",
            "flow_node_retrying": "Retrying node_a after attempt 2.",
            "flow_node_completed": "Finished node_a.",
            "flow_node_failed": "node_a failed.",
            "flow_node_skipped": "Skipped node_a.",
            "flow_node_paused": "Paused node_a.",
            "flow_node_resumed": "Resumed node_a.",
            "flow_node_cancelled": "Cancelled node_a.",
        }

        for event_type, expected in cases.items():
            with self.subTest(event_type=event_type):
                self.assertEqual(
                    self.theme.flow_run_progress_message(
                        event_type,
                        node="node_a",
                        attempt=2,
                    ),
                    expected,
                )

    def test_flow_run_progress_messages_cover_flow_events(self) -> None:
        cases = {
            "flow_started": "Flow run started.",
            "flow_completed": "Flow run completed.",
            "flow_cancelled": "Flow run cancelled.",
            "unknown": "Flow run is active.",
        }

        for event_type, expected in cases.items():
            with self.subTest(event_type=event_type):
                self.assertEqual(
                    self.theme.flow_run_progress_message(event_type),
                    expected,
                )

    def test_flow_run_progress_message_omits_first_attempt(self) -> None:
        self.assertEqual(
            self.theme.flow_run_progress_message(
                "flow_node_started",
                node="node_a",
                attempt=1,
            ),
            "Running node_a.",
        )

    def test_flow_run_progress_plain_renderable(self) -> None:
        renderable = self.theme.flow_run_progress(
            "flowchart LR\n  node_a\n",
            node_states={"node_a": "running"},
            active_nodes=("node_a",),
            message="Running node_a.",
            console_width=80,
            flow_stats={"__total__": {"elapsed_ms": 1}},
        )

        self.assertEqual(
            renderable,
            "Running node_a.\n\nflowchart LR\n  node_a\n",
        )


class ThemeConcreteDefaultsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = _theme()

    def test_theme_instantiates_directly(self) -> None:
        self.assertIsInstance(self.theme, Theme)

    def test_command_facing_defaults_do_not_raise(self) -> None:
        cache = HubCache(
            model_id="cached/model",
            path="/cache/cached/model",
            size_on_disk=1,
            revisions=[],
            files={},
            total_files=0,
            total_revisions=0,
        )
        cache_deletion = HubCacheDeletion(
            model_id="cached/model",
            revisions=[],
            deletable_size_on_disk=1.0,
            deletable_blobs=[],
            deletable_refs=[],
            deletable_repos=[],
            deletable_snapshots=[],
        )
        participant_id = UUID(int=0)
        token = Token(token="hello")
        tokenizer_config = TokenizerConfig(
            name_or_path="tokenizer",
            tokens=[],
            special_tokens=[],
            tokenizer_model_max_length=128,
        )
        text_partition = TextPartition(
            data="partition",
            total_tokens=1,
            embeddings=np.array([0.0]),
        )

        renderables = [
            self.theme.action("n", "d", "a", "id", "lib", True, False),
            self.theme.agent(object(), models=[_model(), "other/model"]),
            self.theme.ask_access_token(),
            self.theme.ask_delete_paths(),
            self.theme.ask_login_to_hub(),
            self.theme.ask_secret_password("k"),
            self.theme.ask_override_secret("k"),
            self.theme.bye(),
            self.theme.cache_delete(cache_deletion, True),
            self.theme.cache_delete(None, False),
            self.theme.cache_list("/c", [cache]),
            self.theme.download_access_denied("m", "u"),
            self.theme.download_start("m"),
            self.theme.download_finished("m", "/p"),
            self.theme.logging_in("domain"),
            self.theme.memory_embeddings(
                "text",
                np.array([0.0]),
                total_tokens=1,
                minv=0.0,
                maxv=0.0,
                meanv=0.0,
                stdv=0.0,
                normv=0.0,
            ),
            self.theme.memory_embeddings_comparison(
                {
                    "a": Similarity(
                        cosine_distance=0.0,
                        inner_product=0.0,
                        l1_distance=0.0,
                        l2_distance=0.0,
                        pearson=0.0,
                    )
                },
                "a",
            ),
            self.theme.memory_embeddings_search(
                [
                    SearchMatch(
                        query="query",
                        match="match",
                        l2_distance=0.0,
                    )
                ]
            ),
            self.theme.memory_partitions(
                [text_partition],
                display_partitions=1,
            ),
            self.theme.model(_model()),
            self.theme.model_display(None, None),
            self.theme.recent_messages(participant_id, object(), []),
            self.theme.saved_tokenizer_files("/d", 0),
            self.theme.search_message_matches(participant_id, object(), []),
            self.theme.memory_search_matches(participant_id, "ns", []),
            self.theme.tokenizer_config(tokenizer_config),
            self.theme.tokenizer_tokens([token], None, None),
            self.theme.display_image_entities(
                [ImageEntity(label="dog"), ImageEntity(label="cat")],
                True,
            ),
            self.theme.display_image_entity(ImageEntity(label="cat")),
            self.theme.display_audio_labels({"label": 0.5}),
            self.theme.display_image_labels(["cat", "dog"]),
            self.theme.display_token_labels([{"hello": "GREETING"}]),
            self.theme.welcome(
                "https://example.test",
                "avalan",
                "1.0.0",
                "MIT",
                User(name="user"),
            ),
            self.theme.token_frames(
                TokenRenderState(model_id="m"),
                console_width=80,
                logger=getLogger(__name__),
            ),
        ]

        for renderable in renderables:
            with self.subTest(renderable=renderable):
                self.assertIsNotNone(renderable)

    def test_events_default_is_non_crashing_compatibility_noop(self) -> None:
        self.assertIsNone(self.theme.events([]))

    def test_download_progress_returns_live_tqdm_template(self) -> None:
        progress = self.theme.download_progress()
        LiveTqdm = create_live_tqdm_class(progress)

        self.assertIsInstance(progress, tuple)
        self.assertTrue(issubclass(LiveTqdm, tqdm_rich_progress))


class ThemeCommonDisplayDomainsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = _theme()

    def test_cache_delete_summarizes_empty_pending_and_deleted_states(
        self,
    ) -> None:
        empty_deletion = HubCacheDeletion(
            model_id="model",
            revisions=[],
            deletable_size_on_disk=0.0,
            deletable_blobs=[],
            deletable_refs=[],
            deletable_repos=[],
            deletable_snapshots=[],
        )
        deletion = HubCacheDeletion(
            model_id="model[red]",
            revisions=["abcdef123"],
            deletable_size_on_disk=1024.0,
            deletable_blobs=["blob"],
            deletable_refs=[],
            deletable_repos=["repo"],
            deletable_snapshots=[],
        )

        self.assertEqual(
            self.theme.cache_delete(empty_deletion),
            "Nothing found for deletion. No action taken.",
        )
        pending = self.theme.cache_delete(deletion)
        deleted = self.theme.cache_delete(deletion, deleted=True)

        self.assertIn("will be freed", pending)
        self.assertIn("BLOBs: 1", pending)
        self.assertIn("repositories: 1", pending)
        self.assertIn(r"model\[red]", pending)
        self.assertIn("were freed", deleted)

    def test_cache_list_filters_and_escapes_model_rows(self) -> None:
        cache = HubCache(
            model_id="cached/[red]",
            path="/cache/[path]",
            size_on_disk=2048,
            revisions=["abcdef123"],
            files={"abcdef123": []},
            total_files=2,
            total_revisions=1,
        )

        renderable = self.theme.cache_list(
            "/cache",
            [cache],
            display_models=["cached/[red]"],
        )

        self.assertIn("Cache: /cache", renderable)
        self.assertIn(r"cached/\[red]", renderable)
        self.assertIn(r"/cache/\[path]", renderable)
        self.assertIn("Revisions: abcdef", renderable)

    def test_model_summary_and_configuration_defaults(self) -> None:
        model = replace(
            _model(),
            parameters=1_500,
            library_name="transformers",
            license="MIT",
            pipeline_tag="text-generation",
            model_type="llama",
            tags=["chat", "[unsafe]"],
        )
        tokenizer_config = TokenizerConfig(
            name_or_path="tok[red]",
            tokens=["a"],
            special_tokens=["<s>"],
            tokenizer_model_max_length=2048,
            fast=True,
        )
        model_config = SimpleNamespace(
            model_type="causal-lm",
            vocab_size=32_000,
            hidden_size=4096,
        )
        transformer_config = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=None,
            bos_token=None,
            decoder_start_token_id=None,
            eos_token_id=None,
            eos_token=None,
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type=None,
            max_position_embeddings=10,
            model_type="sentence-transformer",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=None,
            pad_token=None,
            prefix=None,
            sep_token_id=None,
            sep_token=None,
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        sentence_config = SentenceTransformerModelConfig(
            backend="torch",
            similarity_function="cosine",
            truncate_dimension=None,
            transformer_model_config=transformer_config,
        )

        model_text = self.theme.model(
            model,
            can_access=False,
            expand=True,
        )
        display_text = self.theme.model_display(
            model_config,
            tokenizer_config,
            is_runnable=True,
        )
        sentence_text = self.theme.model_display(sentence_config, None)

        self.assertIn("Parameters: 1.5 thousand", model_text)
        self.assertIn("Access: no", model_text)
        self.assertIn("Library: transformers", model_text)
        self.assertIn(r"\[unsafe]", model_text)
        self.assertIn("Model type: causal-lm", display_text)
        self.assertIn("Vocabulary: 32,000", display_text)
        self.assertIn(r"Tokenizer: tok\[red]", display_text)
        self.assertIn("Runnable: yes", display_text)
        self.assertIn("Backend: torch", sentence_text)
        self.assertIn(
            "Transformer model type: sentence-transformer", sentence_text
        )
        self.assertIn("Similarity: cosine", sentence_text)

    def test_model_display_handles_missing_optional_fields(self) -> None:
        self.assertEqual(
            self.theme.model_display(None, None),
            "Model type: unknown",
        )

    def test_tokenizer_config_and_tokens_include_details(self) -> None:
        current = Token(id=1, token="[current]", probability=0.25)
        renderable = self.theme.tokenizer_config(
            TokenizerConfig(
                name_or_path="tokenizer",
                tokens=None,
                special_tokens=None,
                tokenizer_model_max_length=128,
            )
        )
        tokens = self.theme.tokenizer_tokens(
            [Token(token="plain"), current],
            added_tokens=["[added]"],
            special_tokens=["[special]"],
            display_details=True,
            current_dtoken=current,
            dtokens_selected=[current],
        )

        self.assertIn("Tokens: 0", renderable)
        self.assertIn("Fast: no", renderable)
        self.assertIn(r"Added tokens: \[added]", tokens)
        self.assertIn(r"Special tokens: \[special]", tokens)
        self.assertIn(r"* \[current] (id 1, p 0.25)", tokens)
        self.assertEqual(
            self.theme.tokenizer_tokens([], None, None),
            "No tokens.",
        )

    def test_memory_and_media_outputs_escape_labels_and_handle_empty(
        self,
    ) -> None:
        partition = TextPartition(
            data="[partition]",
            total_tokens=3,
            embeddings=np.array([0.0]),
        )
        comparison = self.theme.memory_embeddings_comparison(
            {
                "[item]": Similarity(
                    cosine_distance=0.1,
                    inner_product=0.2,
                    l1_distance=0.3,
                    l2_distance=0.4,
                    pearson=0.5,
                )
            },
            "[item]",
        )
        embedding = self.theme.memory_embeddings(
            "abcdefghijklmnopqrstuvwxyz",
            np.array([0.0]),
            total_tokens=5,
            minv=0.0,
            maxv=1.0,
            meanv=0.5,
            stdv=0.25,
            normv=1.5,
            input_string_peek=8,
            partition=2,
            total_partitions=3,
        )
        tiny = self.theme.memory_embeddings(
            "abcdef",
            np.array([0.0]),
            total_tokens=1,
            minv=0.0,
            maxv=0.0,
            meanv=0.0,
            stdv=0.0,
            normv=0.0,
            input_string_peek=3,
            show_stats=False,
        )

        self.assertIn(r"Most similar: \[item]", comparison)
        self.assertIn("Partition 2 of 3", embedding)
        self.assertIn("abcde...", embedding)
        self.assertIn("Input: abc", tiny)
        self.assertEqual(
            self.theme.memory_embeddings_search([]), "No matches."
        )
        self.assertIn(
            r"\[partition]",
            self.theme.memory_partitions([partition], display_partitions=1),
        )
        self.assertEqual(
            self.theme.memory_partitions([], display_partitions=1),
            "No partitions.",
        )
        self.assertIn(
            r"cat\[x] (0.9) box [1, 2]",
            self.theme.display_image_entities(
                [ImageEntity(label="cat[x]", score=0.9, box=[1, 2])],
                False,
            ),
        )
        self.assertEqual(
            self.theme.display_audio_labels({"dog[red]": 0.25}),
            r"dog\[red]: 0.25",
        )
        self.assertEqual(
            self.theme.display_image_labels(["cat[red]"]),
            r"cat\[red]",
        )
        self.assertEqual(
            self.theme.display_token_labels([{"tok[red]": "LBL[blue]"}]),
            r"tok\[red]: LBL\[blue]",
        )

    def test_message_and_tokenizer_status_text_pluralizes(self) -> None:
        participant_id = UUID(int=1)

        self.assertIn(
            "1 recent message",
            self.theme.recent_messages(participant_id, object(), [object()]),
        )
        self.assertIn(
            "2 message matches",
            self.theme.search_message_matches(
                participant_id,
                object(),
                [object(), object()],
            ),
        )
        self.assertIn(
            "0 memory matches",
            self.theme.memory_search_matches(participant_id, "ns[red]", []),
        )
        self.assertEqual(
            self.theme.saved_tokenizer_files("/tmp/[tok]", 1),
            r"Saved 1 tokenizer file to /tmp/\[tok].",
        )
        self.assertEqual(
            self.theme.saved_tokenizer_files(2),
            "Saved tokenizer files: 2",
        )


class ThemeCommonEventFormattingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = _theme()

    def test_events_common_helper_formats_tool_lifecycle(self) -> None:
        call = _tool_call()
        result = ToolCallResult(
            id="result-123",
            name=call.name,
            call=call,
            result={
                "ok": True,
                "authorization": "Bearer secret",
                "long": "x" * 400,
            },
        )
        events = [
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": call},
            ),
            Event(
                type=EventType.TOOL_MODEL_RUN,
                payload={"model_id": "react[model]", "messages": [1, 2]},
            ),
            Event(
                type=EventType.TOOL_MODEL_RESPONSE,
                payload={"model_id": "react[model]"},
            ),
            Event(
                type=EventType.TOOL_PROCESS,
                payload=[call],
            ),
            Event(
                type=EventType.TOOL_RESULT,
                payload={"result": result},
                elapsed=0.25,
            ),
        ]

        renderable = self.theme.events(events)

        assert renderable is not None
        self.assertIn(r"calc\[bold]", renderable)
        self.assertIn("<redacted>", renderable)
        self.assertNotIn("secret-value", renderable)
        self.assertNotIn("Bearer secret", renderable)
        self.assertIn(r"react\[model]", renderable)
        self.assertIn("0.250s", renderable)
        self.assertLess(len(renderable), 900)
        self.assertNotIn("\x1b", renderable)
        self.assertNotIn("calc[bold]", renderable)

    def test_events_common_helper_formats_diagnostics_and_errors(self) -> None:
        call = _tool_call()
        diagnostic = _diagnostic()
        error = ToolCallError(
            id="error-123",
            name=call.name,
            call=call,
            error=RuntimeError("boom"),
            message="failure [red]",
        )
        direct = self.theme._tool_diagnostic_from_payload(
            {"diagnostic": diagnostic}
        )
        from_result = self.theme._tool_diagnostic_from_payload(
            {"result": diagnostic}
        )
        from_list = self.theme._tool_diagnostic_from_payload(
            {"diagnostics": [object(), diagnostic]}
        )

        renderable = self.theme.events(
            [
                Event(
                    type=EventType.TOOL_DIAGNOSTIC,
                    payload={"call": call, "diagnostic": diagnostic},
                ),
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": diagnostic},
                ),
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": error},
                ),
            ]
        )

        self.assertIs(direct, diagnostic)
        self.assertIs(from_result, diagnostic)
        self.assertIs(from_list, diagnostic)
        self.assertIsNone(self.theme._tool_diagnostic_from_payload("bad"))
        self.assertIsNone(self.theme._tool_diagnostic_from_payload({}))
        assert renderable is not None
        self.assertIn("tool_call.arguments_invalid", renderable)
        self.assertIn("call #call-123", renderable)
        self.assertNotIn("call #diagnost", renderable)
        self.assertIn("limit", renderable)
        self.assertIn("<redacted>", renderable)
        self.assertIn("failure", renderable)
        self.assertNotIn("failure [red]", renderable)
        self.assertNotIn("raw-token", renderable)
        self.assertNotIn("\x1b", renderable)

    def test_events_common_helper_filters_and_limits(self) -> None:
        events = [
            Event(type=EventType.TOOL_DETECT, payload={"call": "hidden"}),
            Event(type=EventType.TOKEN_GENERATED, payload={"token": "hidden"}),
            Event(type=EventType.START, payload={"visible": True}),
            Event(type=EventType.TOOL_PROCESS, payload=[]),
            Event(type=EventType.END, payload=None),
        ]

        self.assertIsNone(self.theme.events([]))
        self.assertIsNone(self.theme.events(events, events_limit=0))
        self.assertEqual(
            self.theme.events(
                events,
                include_tokens=False,
                include_tool_detect=False,
                include_tools=False,
                events_limit=1,
            ),
            "event <end>",
        )
        self.assertEqual(
            self.theme.events(
                events,
                include_non_tools=False,
                events_limit=1,
            ),
            "Executing 0 tools: none.",
        )

    def test_events_common_helper_bounds_tool_process_call_names(self) -> None:
        events = [
            Event(
                type=EventType.TOOL_PROCESS,
                payload=[
                    ToolCall(
                        id=f"call-{index}",
                        name=f"tool-{index}-" + "x" * 150,
                    )
                    for index in range(10)
                ],
            )
        ]

        renderable = self.theme.events(events)

        assert renderable is not None
        self.assertIn("Executing 10 tools:", renderable)
        self.assertIn("tool-0-", renderable)
        self.assertIn("4 more", renderable)
        self.assertNotIn("tool-6-", renderable)
        self.assertLess(len(renderable), 900)

    def test_events_common_helper_limits_before_formatting(self) -> None:
        events = [
            Event(type=EventType.TOOL_PROCESS, payload=[_tool_call()]),
            Event(type=EventType.END, payload=None),
        ]

        with patch.object(
            self.theme,
            "_format_tool_event",
            wraps=self.theme._format_tool_event,
        ) as format_tool_event:
            renderable = self.theme.events(events, events_limit=1)

        self.assertEqual(renderable, "event <end>")
        format_tool_event.assert_not_called()

    def test_events_common_helper_falls_back_for_malformed_payloads(
        self,
    ) -> None:
        generic = Event(
            type=EventType.TOOL_EXECUTE,
            payload={"missing": "call"},
            elapsed=1.5,
        )
        no_arguments = Event(
            type=EventType.TOOL_EXECUTE,
            payload={"call": ToolCall(id="call-2", name="noop")},
        )

        renderable = self.theme.events([generic])
        no_argument_renderable = self.theme.events([no_arguments])

        self.assertEqual(
            renderable,
            '1.500s <tool_execute>: {"missing": "call"}',
        )
        self.assertIn("with 0 arguments", no_argument_renderable)
        self.assertEqual(self.theme._payload_size(None), 0)
        self.assertEqual(self.theme._payload_size(object()), 1)
        self.assertIsNone(
            self.theme._format_tool_model_run_event({"model_id": "m"})
        )
        self.assertIsNone(self.theme._format_tool_model_response_event({}))
        self.assertIsNone(self.theme._format_tool_process_event("bad"))
        self.assertIsNone(
            self.theme._format_tool_result_event(
                Event(type=EventType.TOOL_RESULT, payload={})
            )
        )

    def test_events_common_helper_safely_summarizes_hostile_payloads(
        self,
    ) -> None:
        cyclic: dict[str, object] = {"token": "hidden"}
        cyclic["self"] = cyclic
        hostile_key = BrokenString()
        payload = {
            "payload": cyclic,
            hostile_key: BrokenString(),
            "bytes": b"abc",
            "deep": [[[[[["too-deep"]]]]]],
            "many": list(range(8)),
            "broken": BrokenData("value"),
            "markup": "[red]\x1b[31msecret",
        }

        renderable = self.theme.events(
            [Event(type=EventType.START, payload=payload)]
        )

        assert renderable is not None
        self.assertIn("<cycle>", renderable)
        self.assertIn("<redacted>", renderable)
        self.assertIn("<bytes 3>", renderable)
        self.assertIn("truncated", renderable)
        self.assertIn("<list>", renderable)
        self.assertIn("<unreadable value>", renderable)
        self.assertIn("<unrepresentable BrokenString>", renderable)
        self.assertNotIn("\x1b", renderable)
        self.assertNotIn("[red]", renderable)
        self.assertLess(len(renderable), 500)

    def test_events_common_helper_redacts_control_split_sensitive_keys(
        self,
    ) -> None:
        renderable = self.theme.events(
            [
                Event(
                    type=EventType.TOOL_EXECUTE,
                    payload={
                        "call": ToolCall(
                            id="call-1",
                            name="calc",
                            arguments={
                                "api\x1b[31m_key": "secret-api-key",
                                "pass\x1b(Bword": "secret-password",
                                "to\x9b31mken": "secret-token",
                                "authorization": "Bearer secret",
                                "safe": "visible",
                            },
                        )
                    },
                )
            ]
        )

        assert renderable is not None
        self.assertIn("<redacted>", renderable)
        self.assertIn("visible", renderable)
        self.assertNotIn("secret-api-key", renderable)
        self.assertNotIn("secret-password", renderable)
        self.assertNotIn("secret-token", renderable)
        self.assertNotIn("Bearer secret", renderable)

    def test_events_common_helper_strips_control_protocol_strings(
        self,
    ) -> None:
        renderable = self.theme.events(
            [
                Event(
                    type=EventType.START,
                    payload={
                        "osc_link": (
                            "\x1b]8;;https://secret.example\x07click\x1b]8;;\x07"
                        ),
                        "osc_embedded_escape": (
                            "\x1b]8;;https://secret.example\x1b[31mhidden"
                            "\x07visible"
                        ),
                        "osc_title": "\x1b]0;secret-title\x07visible",
                        "dcs": "\x1bPsecret-protocol\x1b\\visible",
                        "dcs_c1": "\x90secret-protocol\x9cvisible",
                        "pm_c1": "\x9esecret-protocol\x9cvisible",
                        "apc_c1": "\x9fsecret-protocol\x9cvisible",
                        "sos_c1": "\x98secret-protocol\x9cvisible",
                        "dcs_c1_bel": (
                            "\x90prefix\x07secret-protocol\x9cvisible"
                        ),
                        "pm_c1_bel": (
                            "\x9eprefix\x07secret-protocol\x9cvisible"
                        ),
                        "apc_c1_bel": (
                            "\x9fprefix\x07secret-protocol\x9cvisible"
                        ),
                        "sos_c1_bel": (
                            "\x98prefix\x07secret-protocol\x9cvisible"
                        ),
                        "charset": "\x1b(Bvisible",
                        "dec_screen": "\x1b#8visible",
                        "utf8_select": "\x1b%Gvisible",
                        "reset": "\x1bcvisible",
                    },
                )
            ]
        )

        assert renderable is not None
        self.assertIn("click", renderable)
        self.assertIn("visible", renderable)
        self.assertNotIn("secret.example", renderable)
        self.assertNotIn("hidden", renderable)
        self.assertNotIn("secret-title", renderable)
        self.assertNotIn("secret-protocol", renderable)
        self.assertNotIn("prefix", renderable)
        self.assertNotIn("]8;;", renderable)
        self.assertNotIn("]0;", renderable)
        self.assertNotIn("(B", renderable)
        self.assertNotIn("#8", renderable)
        self.assertNotIn("%G", renderable)

    def test_events_common_helper_contains_nested_mapping_failures(
        self,
    ) -> None:
        renderable = self.theme.events(
            [
                Event(
                    type=EventType.START,
                    payload={
                        "safe": BrokenMapping(),
                        "password": "must-not-leak",
                    },
                )
            ]
        )

        assert renderable is not None
        self.assertIn("<unreadable BrokenMapping>", renderable)
        self.assertIn("<redacted>", renderable)
        self.assertNotIn("must-not-leak", renderable)

    def test_events_common_helper_keeps_fallback_summary_redacted(
        self,
    ) -> None:
        with patch(
            "avalan.cli.display_safety.dumps", side_effect=TypeError("bad")
        ):
            renderable = self.theme.events(
                [
                    Event(
                        type=EventType.START,
                        payload={
                            "password": "must-not-leak",
                            "markup": "[red]visible",
                        },
                    )
                ]
            )

        assert renderable is not None
        self.assertIn("<redacted>", renderable)
        self.assertIn("visible", renderable)
        self.assertNotIn("must-not-leak", renderable)
        self.assertIn(r"\[red]visible", renderable)
        self.assertNotIn("\x1b", renderable)

    def test_events_common_helper_covers_summary_edge_values(self) -> None:
        renderable = self.theme.events(
            [
                Event(
                    type=EventType.START,
                    payload={
                        "when": datetime(2024, 1, 1),
                        "id": UUID(int=2),
                        "fields": ManyFields(1, 2, 3, 4, 5, 6, 7),
                    },
                ),
                Event(type=EventType.END, payload=BrokenMapping()),
            ]
        )

        assert renderable is not None
        self.assertIn("2024-01-01", renderable)
        self.assertIn("00000000-0000-0000-0000-000000000002", renderable)
        self.assertIn('"fields": {"...": "truncated"', renderable)
        self.assertIn("<unreadable BrokenMapping>", renderable)
        self.assertIsNone(
            self.theme._format_tool_event(
                Event(type=EventType.START), "tool_unknown"
            )
        )


class ThemeTokensTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.theme = _theme()

    async def test_tokens_default_defers_stream_presentation(self) -> None:
        frames = [
            frame
            async for frame in self.theme.tokens(
                TokenRenderState(
                    model_id="m",
                    reasoning_text_tokens=("thinking",),
                    tool_text_tokens=("tool",),
                    answer_text_tokens=("answer", " text"),
                ),
                console_width=80,
                logger=getLogger(__name__),
            )
        ]

        self.assertEqual(frames, [])

    async def test_tokens_default_without_answer_text_yields_no_frames(
        self,
    ) -> None:
        frames = [
            frame
            async for frame in self.theme.tokens(
                TokenRenderState(model_id="m"),
                console_width=80,
                logger=getLogger(__name__),
            )
        ]

        self.assertEqual(frames, [])

    async def test_tokens_delegates_to_token_frames(self) -> None:
        class YieldingTheme(Theme):
            def token_frames(
                self,
                state: TokenRenderState,
                *,
                console_width: int,
                logger: Logger,
                maximum_frames: int | None = None,
                logits_count: int | None = None,
                tool_events_limit: int | None = None,
                think_height: int = 6,
                think_padding: int = 1,
                tool_height: int = 6,
                tool_padding: int = 1,
                height: int = 12,
                padding: int = 1,
                wrap_padding: int = 4,
                limit_think_height: bool = True,
                limit_tool_height: bool = True,
                limit_answer_height: bool = False,
                start_thinking: bool = False,
            ) -> tuple[TokenRenderFrame, ...]:
                _ = (
                    state,
                    console_width,
                    logger,
                    maximum_frames,
                    logits_count,
                    tool_events_limit,
                    think_height,
                    think_padding,
                    tool_height,
                    tool_padding,
                    height,
                    padding,
                    wrap_padding,
                    limit_think_height,
                    limit_tool_height,
                    limit_answer_height,
                    start_thinking,
                )
                return ((None, "frame"),)

        theme = YieldingTheme(_gettext, _ngettext)

        frames = [
            frame
            async for frame in theme.tokens(
                TokenRenderState(model_id="m"),
                console_width=80,
                logger=getLogger(__name__),
            )
        ]

        self.assertEqual(frames, [(None, "frame")])


class ThemeMiscMethodsTestCase(unittest.TestCase):
    def test_get_styles_and_spinner_and_call(self) -> None:
        theme = _theme()
        styles = theme.get_styles()
        model = _model()

        self.assertIn("id", styles)
        self.assertEqual(theme.get_spinner("thinking"), None)
        self.assertIn("Model: model-id", theme(model))
        self.assertEqual(theme("x"), "x")
        self.assertFalse(theme.default_display_tools)
        self.assertFalse(theme.prefix_stream_answers)
        self.assertEqual(
            theme.tool_status_icon("result"),
            ":white_check_mark:",
        )
        self.assertEqual(theme.tool_status_icon("cancelled"), ":warning:")
        self.assertEqual(
            theme.tool_status_icon("unknown"),
            ":information_source:",
        )
        self.assertEqual(theme.tool_status_style("error"), "red")
        self.assertEqual(theme.tool_status_style("cancelled"), "yellow")
        self.assertEqual(theme.tool_status_style("unknown"), "cyan")
        self.assertEqual(
            theme.tool_status_style(
                "result",
                success_style="spring_green3",
            ),
            "spring_green3",
        )
        self.assertEqual(theme.precise_elapsed_text(None), None)
        self.assertEqual(theme.precise_elapsed_text(0.0), "0 microseconds")


class ThemeFormatTestCase(unittest.TestCase):
    def test_custom_format_and_spinner(self) -> None:
        theme = Theme(
            _gettext,
            _ngettext,
            icons={"id": ":robot:", "likes": ":heart:"},
            quantity_data=["likes"],
        )

        self.assertIsNone(theme.get_spinner("thinking"))
        self.assertEqual(theme._f("id", "test"), ":robot: [id]test[/id]")
        self.assertEqual(theme._f("unknown", "value"), "value")
        likes_formatted = theme._f("likes", 1000)
        self.assertIn("thousand", likes_formatted)


class FancyThemeStabilityTestCase(unittest.TestCase):
    def test_flow_run_progress_message_keeps_flow_defaults(self) -> None:
        theme = FancyTheme(_gettext, _ngettext)

        cases = {
            "flow_started": "Flow run started.",
            "flow_completed": "Flow run completed.",
            "unknown": "Flow run is active.",
        }

        for event_type, expected in cases.items():
            with self.subTest(event_type=event_type):
                self.assertEqual(
                    theme.flow_run_progress_message(event_type),
                    expected,
                )
