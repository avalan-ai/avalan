import unittest
from datetime import datetime
from logging import getLogger
from uuid import UUID

import numpy as np

from avalan.cli.download import create_live_tqdm_class, tqdm_rich_progress
from avalan.cli.theme import Theme, TokenRenderState
from avalan.cli.theme.fancy import FancyTheme
from avalan.entities import (
    HubCache,
    HubCacheDeletion,
    ImageEntity,
    Model,
    SearchMatch,
    Similarity,
    TextPartition,
    Token,
    TokenizerConfig,
    User,
)


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


class ThemeMiscMethodsTestCase(unittest.TestCase):
    def test_get_styles_and_spinner_and_call(self) -> None:
        theme = _theme()
        styles = theme.get_styles()
        model = _model()

        self.assertIn("id", styles)
        self.assertEqual(theme.get_spinner("thinking"), None)
        self.assertEqual(theme(model), "model-id")
        self.assertEqual(theme("x"), "x")


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
