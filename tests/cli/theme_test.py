import unittest
from datetime import datetime
from types import SimpleNamespace
import numpy as np

from avalan.cli.theme import Theme
from avalan.entities import Model


class DummyTheme(Theme):
    def action(self, *args, **kwargs):
        raise NotImplementedError()

    def agent(self, *args, **kwargs):
        raise NotImplementedError()

    def ask_access_token(self) -> str:
        raise NotImplementedError()

    def ask_delete_paths(self) -> str:
        raise NotImplementedError()

    def ask_login_to_hub(self) -> str:
        raise NotImplementedError()

    def ask_secret_password(self, key: str) -> str:
        raise NotImplementedError()

    def ask_override_secret(self, key: str) -> str:
        raise NotImplementedError()

    def bye(self):
        raise NotImplementedError()

    def cache_delete(self, cache_deletion, deleted: bool = False):
        raise NotImplementedError()

    def cache_list(
        self,
        cache_dir: str,
        cached_models,
        display_models=None,
        show_summary: bool = False,
    ):
        raise NotImplementedError()

    def download_access_denied(self, model_id: str, model_url: str):
        raise NotImplementedError()

    def download_start(self, model_id: str):
        raise NotImplementedError()

    def download_progress(self):
        raise NotImplementedError()

    def download_finished(self, model_id: str, path: str):
        raise NotImplementedError()

    def logging_in(self, domain: str) -> str:
        raise NotImplementedError()

    def memory_embeddings(
        self,
        input_string: str,
        embeddings,
        *args,
        total_tokens: int,
        minv: float,
        maxv: float,
        meanv: float,
        stdv: float,
        normv: float,
        embedding_peek: int = 3,
        horizontal: bool = True,
        input_string_peek: int = 40,
        show_stats: bool = True,
        partition=None,
        total_partitions=None,
    ):
        raise NotImplementedError()

    def memory_embeddings_comparison(self, similarities, most_similar: str):
        raise NotImplementedError()

    def memory_embeddings_search(
        self, matches, *args, match_preview_length: int = 300
    ):
        raise NotImplementedError()

    def memory_partitions(self, partitions, *args, display_partitions: int):
        raise NotImplementedError()

    def model(self, model: Model, *args, **kwargs):
        raise NotImplementedError()

    def model_display(self, model_config, tokenizer_config, *args, **kwargs):
        raise NotImplementedError()

    def recent_messages(self, participant_id, agent, messages):
        raise NotImplementedError()

    def saved_tokenizer_files(self, directory_path: str, total_files: int):
        raise NotImplementedError()

    def search_message_matches(self, participant_id, agent, messages):
        raise NotImplementedError()

    def memory_search_matches(self, participant_id, namespace: str, memories):
        raise NotImplementedError()

    def tokenizer_config(self, config):
        raise NotImplementedError()

    def tokenizer_tokens(
        self,
        dtokens,
        added_tokens=None,
        special_tokens=None,
        current_dtoken=None,
        dtokens_selected=None,
    ):
        raise NotImplementedError()

    async def tokens(self, *args, **kwargs):
        raise NotImplementedError()

    def welcome(self, url: str, name: str, version: str, license: str, user):
        raise NotImplementedError()


class CallableTheme(DummyTheme):
    def model(self, model: Model, *args, **kwargs):
        return f"model:{model.id}"


class ThemePropertyTestCase(unittest.TestCase):
    def setUp(self):
        self.theme = DummyTheme(lambda s: s, lambda s, p, n: s)

    def test_default_properties(self):
        self.assertEqual(self.theme.icons, {})
        self.assertEqual(self.theme.quantity_data, [])
        self.assertEqual(self.theme.spinners, {})
        self.assertEqual(self.theme.stylers, {})
        self.assertEqual(self.theme.styles, {})


class ThemeAbstractMethodsTestCase(unittest.TestCase):
    def setUp(self):
        self.theme = DummyTheme(lambda s: s, lambda s, p, n: s)

    def test_all_methods_raise(self):
        methods = [
            lambda: self.theme.action("n", "d", "a", "id", "lib", True, False),
            lambda: self.theme.agent(
                SimpleNamespace(), models=[], cans_access=None
            ),
            self.theme.ask_access_token,
            self.theme.ask_delete_paths,
            self.theme.ask_login_to_hub,
            lambda: self.theme.ask_secret_password("k"),
            lambda: self.theme.ask_override_secret("k"),
            self.theme.bye,
            lambda: self.theme.cache_delete(None, False),
            lambda: self.theme.cache_list("/c", []),
            lambda: self.theme.download_access_denied("m", "u"),
            lambda: self.theme.download_start("m"),
            self.theme.download_progress,
            lambda: self.theme.download_finished("m", "/p"),
            lambda: self.theme.logging_in("domain"),
            lambda: self.theme.memory_embeddings(
                "text",
                np.array([0.0]),
                total_tokens=0,
                minv=0.0,
                maxv=0.0,
                meanv=0.0,
                stdv=0.0,
                normv=0.0,
            ),
            lambda: self.theme.memory_embeddings_comparison({}, "m"),
            lambda: self.theme.memory_embeddings_search([]),
            lambda: self.theme.memory_partitions([], display_partitions=0),
            lambda: self.theme.model(SimpleNamespace()),
            lambda: self.theme.model_display(None, None),
            lambda: self.theme.recent_messages("id", SimpleNamespace(), []),
            lambda: self.theme.saved_tokenizer_files("/d", 0),
            lambda: self.theme.search_message_matches(
                "id", SimpleNamespace(), []
            ),
            lambda: self.theme.memory_search_matches("id", "ns", []),
            lambda: self.theme.tokenizer_config(None),
            lambda: self.theme.tokenizer_tokens([]),
            lambda: self.theme.welcome("u", "n", "v", "lic", None),
        ]
        for call in methods:
            with self.assertRaises(NotImplementedError):
                result = call()
                if hasattr(result, "__await__"):
                    # async method tokens returns awaitable
                    self.theme.loop.run_until_complete(result)

        async def run_tokens():
            await self.theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                text_tokens=[],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                ttft=0.0,
                ttnt=0.0,
                ellapsed=0.0,
                console_width=80,
                logger=SimpleNamespace(),
            )

        with self.assertRaises(NotImplementedError):
            import asyncio

            asyncio.run(run_tokens())


class ThemeMiscMethodsTestCase(unittest.TestCase):
    def test_get_styles_and_spinner_and_call(self):
        theme = CallableTheme(lambda s: s, lambda s, p, n: s)
        styles = theme.get_styles()
        self.assertIn("id", styles)
        self.assertEqual(theme.get_spinner("thinking"), None)
        model = Model(
            id="i",
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
            author="a",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.assertEqual(theme(model), "model:i")
        self.assertEqual(theme("x"), "x")


class ThemeFormatTestCase(unittest.TestCase):
    def test_custom_format_and_spinner(self):
        theme = CallableTheme(
            lambda s: s,
            lambda s, p, n: s,
            icons={"id": ":robot:", "likes": ":heart:"},
            quantity_data=["likes"],
        )

        self.assertIsNone(theme.get_spinner("thinking"))
        self.assertEqual(theme._f("id", "test"), ":robot: [id]test[/id]")
        likes_formatted = theme._f("likes", 1000)
        self.assertIn("thousand", likes_formatted)


class ThemeBaseMethodsCoverageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = CallableTheme(lambda s: s, lambda s, p, n: s)

    def test_base_methods_raise(self):
        calls = [
            lambda: Theme.action(
                self.theme, "n", "d", "a", "id", "lib", True, False
            ),
            lambda: Theme.agent(
                self.theme, SimpleNamespace(), models=[], cans_access=None
            ),
            lambda: Theme.ask_access_token(self.theme),
            lambda: Theme.ask_delete_paths(self.theme),
            lambda: Theme.ask_login_to_hub(self.theme),
            lambda: Theme.ask_secret_password(self.theme, "k"),
            lambda: Theme.ask_override_secret(self.theme, "k"),
            lambda: Theme.bye(self.theme),
            lambda: Theme.cache_delete(self.theme, None, False),
            lambda: Theme.cache_list(self.theme, "/c", []),
            lambda: Theme.download_access_denied(self.theme, "m", "u"),
            lambda: Theme.download_start(self.theme, "m"),
            lambda: Theme.download_progress(self.theme),
            lambda: Theme.download_finished(self.theme, "m", "/p"),
            lambda: Theme.logging_in(self.theme, "domain"),
            lambda: Theme.memory_embeddings(
                self.theme,
                "text",
                np.array([0.0]),
                total_tokens=0,
                minv=0.0,
                maxv=0.0,
                meanv=0.0,
                stdv=0.0,
                normv=0.0,
            ),
            lambda: Theme.memory_embeddings_comparison(self.theme, {}, "m"),
            lambda: Theme.memory_embeddings_search(self.theme, []),
            lambda: Theme.memory_partitions(
                self.theme, [], display_partitions=0
            ),
            lambda: Theme.model(self.theme, SimpleNamespace()),
            lambda: Theme.model_display(self.theme, None, None),
            lambda: Theme.recent_messages(
                self.theme, "id", SimpleNamespace(), []
            ),
            lambda: Theme.saved_tokenizer_files("/d", 0),
            lambda: Theme.search_message_matches(
                self.theme, "id", SimpleNamespace(), []
            ),
            lambda: Theme.memory_search_matches(self.theme, "id", "ns", []),
            lambda: Theme.tokenizer_config(self.theme, None),
            lambda: Theme.tokenizer_tokens(self.theme, [], None, None),
            lambda: Theme.welcome(self.theme, "u", "n", "v", "lic", None),
        ]

        for call in calls:
            with self.assertRaises(NotImplementedError):
                call()

        async def run_tokens():
            await Theme.tokens(
                self.theme,
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                text_tokens=[],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                ttft=0.0,
                ttnt=0.0,
                ellapsed=0.0,
                console_width=80,
                logger=SimpleNamespace(),
            )

        with self.assertRaises(NotImplementedError):
            import asyncio

            asyncio.run(run_tokens())
