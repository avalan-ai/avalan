from datetime import datetime
from types import SimpleNamespace
from uuid import UUID
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from rich.spinner import Spinner
from rich.text import Text
from rich.table import Table
from rich import box
import numpy as np
from numpy.linalg import norm

from avalan.entities import (
    EngineMessage,
    EngineMessageScored,
    HubCache,
    HubCacheFile,
    HubCacheDeletion,
    Message,
    ModelConfig,
    SearchMatch,
    SentenceTransformerModelConfig,
    Similarity,
    Token,
    TokenDetail,
    TokenizerConfig,
    ImageEntity,
    User,
)
from avalan.memory.permanent import Memory, MemoryType
from avalan.memory.partitioner.text import TextPartition

from avalan.cli.theme.fancy import FancyTheme
from avalan.event import Event, EventType


class FancyThemeTokensTestCase(IsolatedAsyncioTestCase):
    async def test_tool_running_spinner_text(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        spinner = Spinner("dots", text="[cyan]run[/cyan]", style="cyan")
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            gen = theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                thinking_text_tokens=[],
                tool_text_tokens=[],
                answer_text_tokens=["a"],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=[],
                tool_event_calls=[
                    Event(type=EventType.TOOL_PROCESS, payload=[])
                ],
                tool_event_results=[],
                tool_running_spinner=spinner,
                ttft=0.0,
                ttnt=0.0,
                elapsed=1.0,
                console_width=80,
                logger=MagicMock(),
            )
            frame = await gen.__anext__()
        self.assertTrue(
            any(
                getattr(r, "renderable", None) is spinner
                for r in frame[1].renderables
            )
        )

    async def test_tool_text_tokens_panel(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            gen = theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                thinking_text_tokens=[],
                tool_text_tokens=["tool"],
                answer_text_tokens=["answer"],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.0,
                ttnt=0.0,
                elapsed=1.0,
                console_width=80,
                logger=MagicMock(),
            )
            _, frame = await gen.__anext__()
        self.assertEqual(len(frame.renderables), 2)
        self.assertIn("tool", frame.renderables[0].renderable)


class FancyThemeTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_quantity_data(self):
        self.assertEqual(self.theme.quantity_data, ["likes"])

    def test_agent(self):
        memory = SimpleNamespace(
            has_recent_message=True,
            has_permanent_message=True,
            permanent_message=SimpleNamespace(
                session_id=UUID("11111111-1111-1111-1111-111111111111"),
                has_session=True,
            ),
        )
        agent = SimpleNamespace(id=UUID(int=0), name="agent", memory=memory)
        model = SimpleNamespace(
            id="m",
            parameters=1,
            parameter_types=["p"],
            inference=None,
            library_name=None,
            pipeline_tag=None,
            tags=[],
            architectures=None,
            model_type=None,
            license=None,
            gated=False,
            private=False,
            disabled=False,
            downloads=1,
            likes=2,
            ranking=1,
            author="a",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        panel = self.theme.agent(agent, models=[model], can_access=True)
        self.assertTrue(panel.title)

    def test_ask_methods(self):
        self.assertEqual(
            self.theme.ask_access_token(),
            "Enter your Huggingface access token",
        )
        self.assertEqual(
            self.theme.ask_delete_paths(), "Delete selected paths?"
        )
        self.assertEqual(
            self.theme.ask_login_to_hub(), "Login to huggingface?"
        )
        self.assertEqual(
            self.theme.ask_secret_password("k"), "Enter secret for k"
        )
        self.assertEqual(
            self.theme.ask_override_secret("k"), "Secret k exists, override?"
        )

    def test_cache_methods(self):
        deletion = HubCacheDeletion(
            model_id="m",
            revisions=["r"],
            deletable_size_on_disk=1,
            deletable_blobs=["b"],
            deletable_refs=[],
            deletable_repos=[],
            deletable_snapshots=[],
        )
        result = self.theme.cache_delete(deletion)
        self.assertTrue(
            hasattr(result, "renderables") or hasattr(result, "text")
        )

        cache = HubCache(
            model_id="m",
            path="/p",
            size_on_disk=1,
            revisions=["r"],
            files={},
            total_files=0,
            total_revisions=1,
        )
        table = self.theme.cache_list("/c", [cache], show_summary=True)
        self.assertEqual(len(table.rows), 1)

    def test_download_methods(self):
        self.theme.download_progress()
        self.theme.download_start("m")
        self.theme.download_finished("m", "/path")
        self.theme.download_access_denied("m", "url")

    def test_memory_embeddings(self):
        data = np.arange(6, dtype=float)
        grp = self.theme.memory_embeddings(
            "text",
            data,
            total_tokens=1,
            minv=float(data.min()),
            maxv=float(data.max()),
            meanv=float(data.mean()),
            stdv=float(data.std()),
            normv=float(norm(data)),
        )
        self.assertTrue(grp.renderables)

        grp = self.theme.memory_embeddings(
            "text",
            data,
            total_tokens=1,
            minv=float(data.min()),
            maxv=float(data.max()),
            meanv=float(data.mean()),
            stdv=float(data.std()),
            normv=float(norm(data)),
            embedding_peek=2,
            show_stats=False,
        )
        self.assertTrue(grp.renderables)

    def test_memory_embeddings_comparison(self):
        sim = Similarity(
            cosine_distance=0.0,
            inner_product=0.0,
            l1_distance=0.0,
            l2_distance=0.0,
            pearson=0.0,
        )
        res = self.theme.memory_embeddings_comparison({"t": sim}, "t")
        self.assertEqual(len(res.renderable.rows), 1)

    def test_memory_embeddings_search(self):
        match = SearchMatch(query="q", match="m", l2_distance=0.1)
        res = self.theme.memory_embeddings_search([match])
        self.assertEqual(len(res.renderable.rows), 1)

    def test_memory_partitions(self):
        part = TextPartition(
            data="t", total_tokens=1, embeddings=np.array([1])
        )
        group = self.theme.memory_partitions([part] * 3, display_partitions=2)
        self.assertEqual(len(group.renderables), 3)

    def test_model(self):
        model = SimpleNamespace(
            id="m",
            parameters=1,
            parameter_types=["p"],
            inference=None,
            library_name=None,
            pipeline_tag=None,
            tags=[],
            architectures=None,
            model_type=None,
            license=None,
            gated=False,
            private=False,
            disabled=False,
            downloads=1,
            likes=2,
            ranking=1,
            author="a",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        panel = self.theme.model(model, can_access=True)
        self.assertIn("m", panel.title)

    def test_model_display(self):
        cfg = ModelConfig(
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
            model_type="t",
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
        tok_cfg = TokenizerConfig(
            name_or_path="t",
            tokens=["a"],
            special_tokens=["b"],
            tokenizer_model_max_length=10,
            fast=True,
        )
        group = self.theme.model_display(cfg, tok_cfg)
        self.assertEqual(len(group.renderables), 2)

    def test_recent_messages(self):
        msg = EngineMessage(
            agent_id=UUID(int=0),
            model_id="m",
            message=Message(role="user", content="hi"),
        )
        agent = SimpleNamespace(name="n")
        group = self.theme.recent_messages(str(UUID(int=1)), agent, [msg])
        self.assertEqual(len(group.renderables), 1)

    def test_saved_tokenizer_files(self):
        pad = self.theme.saved_tokenizer_files("/d", 2)
        self.assertIn("2 tokenizer files", pad.renderable)

    def test_search_message_matches(self):
        msg = EngineMessageScored(
            agent_id=UUID(int=0),
            model_id="m",
            message=Message(role="user", content="hi"),
            score=0.1,
        )
        agent = SimpleNamespace(name="n")
        group = self.theme.search_message_matches(
            str(UUID(int=1)), agent, [msg]
        )
        self.assertEqual(len(group.renderables), 1)

    def test_memory_search_matches(self):
        mem = Memory(
            id=UUID(int=0),
            model_id="m",
            type=MemoryType.RAW,
            participant_id=str(UUID(int=1)),
            namespace="ns",
            identifier="id",
            data="d",
            partitions=1,
            symbols={},
            created_at=datetime.now(),
        )
        group = self.theme.memory_search_matches(str(UUID(int=1)), "ns", [mem])
        self.assertEqual(len(group.renderables), 1)

    def test_tokenizer_config(self):
        cfg = TokenizerConfig(
            name_or_path="t",
            tokens=["a"],
            special_tokens=["b"],
            tokenizer_model_max_length=10,
            fast=True,
        )
        panel = self.theme.tokenizer_config(cfg)
        self.assertTrue(panel.renderable.rows)

    def test_tokenizer_tokens(self):
        t1 = Token(id=1, token="a")
        t2 = Token(id=2, token="b")
        panel = self.theme.tokenizer_tokens(
            [t1, t2], ["a"], ["s"], current_dtoken=t1
        )
        self.assertEqual(len(panel.renderable.renderable.renderables), 2)

    async def test_tokens_thinking(self):
        t = Token(id=1, token="a")
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            gen = self.theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=1,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=lambda x: True,
                thinking_text_tokens=["x\n"],
                tool_text_tokens=[],
                answer_text_tokens=["y"],
                tokens=[t],
                input_token_count=0,
                total_tokens=1,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.0,
                ttnt=0.0,
                elapsed=1.0,
                console_width=40,
                logger=MagicMock(),
                maximum_frames=1,
            )
            frame = await gen.__anext__()
        self.assertTrue(frame[1].renderables)

    async def test_tokens_multiple_frames(self):
        alt1 = Token(id=2, token="b", probability=0.6)
        alt2 = Token(id=3, token="c", probability=0.4)
        dtoken = TokenDetail(
            id=1, token="a", probability=0.8, tokens=[alt1, alt2]
        )

        with (
            patch(
                "avalan.cli.theme.fancy._lf",
                lambda i: list(filter(None, i or [])),
            ),
            patch(
                "avalan.cli.theme.fancy._j",
                lambda sep, items: sep.join(str(x) for x in items if x),
            ),
        ):
            gen = self.theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=1,
                display_probabilities=True,
                pick=2,
                focus_on_token_when=lambda x: True,
                thinking_text_tokens=["x\n"],
                tool_text_tokens=[],
                answer_text_tokens=["y"],
                tokens=[dtoken],
                input_token_count=0,
                total_tokens=1,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.1,
                ttnt=0.1,
                elapsed=1.0,
                console_width=40,
                logger=MagicMock(),
                maximum_frames=2,
            )
            frame1 = await gen.__anext__()
            frame2 = await gen.__anext__()
            with self.assertRaises(StopAsyncIteration):
                await gen.__anext__()

        self.assertTrue(frame1[1].renderables)
        self.assertTrue(frame2[1].renderables)

    async def test_tokens_early_return(self):
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            gen = self.theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                thinking_text_tokens=[],
                tool_text_tokens=[],
                answer_text_tokens=["x\n"],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.0,
                ttnt=0.0,
                elapsed=1.0,
                console_width=40,
                logger=MagicMock(),
            )
            token, frame = await gen.__anext__()
            with self.assertRaises(StopAsyncIteration):
                await gen.__anext__()

        self.assertIsNone(token)
        self.assertTrue(frame.renderables)

    async def test_tokens_pick_first_full_batch(self):
        alt = Token(id=2, token="b", probability=0.5)
        dtoken = TokenDetail(id=1, token="a", probability=0.6, tokens=[alt])
        with (
            patch(
                "avalan.cli.theme.fancy._lf",
                lambda i: list(filter(None, i or [])),
            ),
            patch(
                "avalan.cli.theme.fancy._j",
                lambda sep, items: sep.join(str(x) for x in items if x),
            ),
        ):
            gen = self.theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=1,
                display_probabilities=True,
                pick=1,
                focus_on_token_when=lambda x: True,
                thinking_text_tokens=[],
                tool_text_tokens=[],
                answer_text_tokens=["x\n"],
                tokens=[dtoken],
                input_token_count=0,
                total_tokens=1,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.0,
                ttnt=0.0,
                elapsed=1.0,
                console_width=40,
                logger=MagicMock(),
                maximum_frames=1,
            )
            frame = await gen.__anext__()
        self.assertTrue(frame[1].renderables)

    async def test_tokens_wrap_long_lines(self):
        long1 = "Word " * 20
        long2 = "Another word " * 20
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            gen = self.theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                thinking_text_tokens=[],
                tool_text_tokens=[],
                answer_text_tokens=[f"{long1}\n", long2],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.0,
                ttnt=0.0,
                elapsed=1.0,
                console_width=40,
                logger=MagicMock(),
            )
            _, frame = await gen.__anext__()

        self.assertEqual(len(frame.renderables), 1)
        text = frame.renderables[0].renderable
        self.assertIn("Word Word", text)
        self.assertIn("Another word", text)
        self.assertGreaterEqual(text.count("\n"), 1)

    async def test_tokens_thinking_wrap_long_lines(self):
        think_line = "Reasoning " * 10
        answer_line = "Answer " * 10
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            gen = self.theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                thinking_text_tokens=[f"{think_line}\n", f"{think_line}\n"],
                tool_text_tokens=[],
                answer_text_tokens=[f"{answer_line}\n", answer_line],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.0,
                ttnt=0.0,
                elapsed=1.0,
                console_width=40,
                logger=MagicMock(),
            )
            _, frame = await gen.__anext__()

        self.assertEqual(len(frame.renderables), 2)
        think_text = frame.renderables[0].renderable
        answer_text = frame.renderables[1].renderable
        self.assertIn("Reasoning", think_text)
        self.assertIn("Answer", answer_text)
        self.assertGreaterEqual(answer_text.count("\n"), 1)

    async def test_tokens_thinking_uses_full_height(self):
        lines = [f"line{i}\n" for i in range(4)]
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            gen = self.theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                thinking_text_tokens=lines,
                tool_text_tokens=[],
                answer_text_tokens=[],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.0,
                ttnt=0.0,
                elapsed=1.0,
                console_width=40,
                logger=MagicMock(),
            )
            _, frame = await gen.__anext__()

        self.assertEqual(len(frame.renderables), 1)
        think_text = frame.renderables[0].renderable
        self.assertIn("line0", think_text)
        self.assertGreaterEqual(think_text.count("\n"), 3)


class FancyThemeAdditionalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_bye(self):
        self.assertEqual(self.theme.bye(), ":vulcan_salute: bye :)")

    def test_action(self):
        panel = self.theme.action(
            "task",
            "desc",
            "author",
            "m",
            "lib",
            highlight=True,
            finished=True,
        )
        self.assertEqual(panel.box, box.DOUBLE)
        self.assertIn(
            "[green]desc[/green]", panel.renderable.renderable.renderables[0]
        )

    def test_cache_delete_deleted_true(self):
        deletion = HubCacheDeletion(
            model_id="m",
            revisions=["r"],
            deletable_size_on_disk=1,
            deletable_blobs=["b"],
            deletable_refs=[],
            deletable_repos=[],
            deletable_snapshots=[],
        )
        result = self.theme.cache_delete(deletion, deleted=True)
        self.assertTrue(result.renderables)

    def test_cache_list_display_models(self):
        cache = HubCache(
            model_id="m",
            path="/p",
            size_on_disk=1,
            revisions=["r"],
            files={"r": []},
            total_files=0,
            total_revisions=1,
        )
        group = self.theme.cache_list(
            "/c",
            [cache],
            display_models=["m"],
            show_summary=False,
        )
        self.assertEqual(len(group.renderables), 1)
        self.assertEqual(group.renderables[0].renderable.title, cache.model_id)

    def test_logging_in(self):
        self.assertEqual(
            self.theme.logging_in("hf"),
            "Logging in to hf...",
        )

    def test_memory_embeddings_orientation(self):
        data = np.arange(6, dtype=float)
        group = self.theme.memory_embeddings(
            "text",
            data,
            total_tokens=1,
            minv=float(data.min()),
            maxv=float(data.max()),
            meanv=float(data.mean()),
            stdv=float(data.std()),
            normv=float(norm(data)),
            embedding_peek=2,
            horizontal=False,
            show_stats=False,
        )
        table = group.renderables[0].renderable
        self.assertEqual(len(table.columns), 2)

        group = self.theme.memory_embeddings(
            "text",
            data,
            total_tokens=1,
            minv=float(data.min()),
            maxv=float(data.max()),
            meanv=float(data.mean()),
            stdv=float(data.std()),
            normv=float(norm(data)),
            embedding_peek=2,
            horizontal=True,
            show_stats=False,
        )
        table = group.renderables[0].renderable
        self.assertEqual(len(table.columns), 5)

    def test_model_display_sentence_transformer(self):
        cfg = ModelConfig(
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
            model_type="t",
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
        st_cfg = SentenceTransformerModelConfig(
            backend="torch",
            similarity_function="cosine",
            truncate_dimension=None,
            transformer_model_config=cfg,
        )
        tok_cfg = TokenizerConfig(
            name_or_path="t",
            tokens=["a"],
            special_tokens=["b"],
            tokenizer_model_max_length=10,
            fast=True,
        )
        group = self.theme.model_display(st_cfg, tok_cfg)
        self.assertEqual(len(group.renderables), 2)

    def test_welcome(self):
        user = User(name="u", access_token_name="tok")
        pad = self.theme.welcome("http://u", "avalan", "1.0", "MIT", user)
        text = str(pad.renderable.renderable)
        self.assertIn("avalan", text)
        self.assertIn("1.0", text)
        self.assertIn("tok", text)


class FancyThemeMoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_cache_delete_none(self):
        result = self.theme.cache_delete(None)
        self.assertIsInstance(result, Text)
        self.assertIn("Nothing found", result.plain)

    def test_cache_list_multiple_revision_files(self):
        now = datetime.now()
        file1 = HubCacheFile(
            name="f1",
            path="/f1",
            size_on_disk=1,
            last_accessed=now,
            last_modified=now,
        )
        file2 = HubCacheFile(
            name="f2",
            path="/f2",
            size_on_disk=1,
            last_accessed=now,
            last_modified=now,
        )
        cache = HubCache(
            model_id="m",
            path="/p",
            size_on_disk=2,
            revisions=["r1", "r2"],
            files={"r1": [file1, file2], "r2": []},
            total_files=2,
            total_revisions=2,
        )
        group = self.theme.cache_list(
            "/c",
            [cache],
            display_models=["m"],
            show_summary=False,
        )
        table = group.renderables[0].renderable
        self.assertEqual(table.row_count, 2)
        self.assertIn("[bright_black]", table.columns[0]._cells[1])

    def test_memory_partitions_many(self):
        part = TextPartition(
            data="t", total_tokens=1, embeddings=np.array([1])
        )
        group = self.theme.memory_partitions([part] * 5, display_partitions=3)
        self.assertEqual(len(group.renderables), 4)

    def test_sentence_transformer_model_config(self):
        cfg = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=1,
            bos_token="<s>",
            decoder_start_token_id=None,
            eos_token_id=2,
            eos_token="</s>",
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type="ce",
            max_position_embeddings=10,
            model_type="t",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=0,
            pad_token="<pad>",
            prefix="pre",
            sep_token_id=3,
            sep_token="<sep>",
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        st_cfg = SentenceTransformerModelConfig(
            backend="torch",
            similarity_function="cosine",
            truncate_dimension=128,
            transformer_model_config=cfg,
        )
        align = self.theme._sentence_transformer_model_config(
            st_cfg, is_runnable=True, summary=False
        )
        table = align.renderable
        headers = table.columns[0]._cells
        self.assertIn("Truncate dimension", headers)

    def test_display_image_entities(self):
        align = self.theme.display_image_entities(
            [
                ImageEntity(label="cat", score=0.5, box=[0.0, 1.0, 2.0, 3.0]),
                ImageEntity(label="dog", score=0.9, box=None),
            ],
            sort=True,
        )
        table = align.renderable
        self.assertEqual(table.row_count, 2)
        # dog should be first due to higher score
        self.assertEqual(table.columns[0]._cells[0], "dog")
        self.assertEqual(table.columns[0]._cells[1], "cat")
        self.assertEqual(table.columns[1]._cells[0], "[score]0.90[/score]")
        self.assertEqual(table.columns[1]._cells[1], "[score]0.50[/score]")

    def test_display_image_entity(self):
        align = self.theme.display_image_entity(ImageEntity(label="cat"))
        table = align.renderable
        self.assertEqual(table.row_count, 1)
        self.assertEqual(table.columns[0]._cells[0], "cat")

    def test_display_image_labels(self):
        align = self.theme.display_image_labels(["cat", "dog"])
        table = align.renderable
        self.assertEqual(table.row_count, 2)
        self.assertEqual(table.columns[0]._cells[0], "cat")
        self.assertEqual(table.columns[0]._cells[1], "dog")

    def test_display_audio_labels(self):
        align = self.theme.display_audio_labels({"dog": 0.9, "cat": 0.5})
        table = align.renderable
        self.assertEqual(table.row_count, 2)
        self.assertEqual(table.columns[0]._cells[0], "dog")
        self.assertEqual(table.columns[0]._cells[1], "cat")
        self.assertEqual(table.columns[1]._cells[0], "[score]0.90[/score]")
        self.assertEqual(table.columns[1]._cells[1], "[score]0.50[/score]")

    def test_display_token_labels(self):
        align = self.theme.display_token_labels([{"tok": "LBL"}])
        table = align.renderable
        self.assertEqual(table.row_count, 1)
        self.assertEqual(table.columns[0]._cells[0], "tok")
        self.assertEqual(table.columns[1]._cells[0], "LBL")

    def test_fill_model_config_table(self):
        cfg = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=1,
            bos_token="<s>",
            decoder_start_token_id=None,
            eos_token_id=2,
            eos_token="</s>",
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type="ce",
            max_position_embeddings=10,
            model_type="t",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=0,
            pad_token="<pad>",
            prefix="pre",
            sep_token_id=3,
            sep_token="<sep>",
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        table = Table(show_lines=True)
        table.add_column()
        table.add_column()
        filled = self.theme._fill_model_config_table(
            cfg, table, is_runnable=True, summary=False
        )
        cells = filled.columns[0]._cells
        self.assertIn("Runs on this instance", cells)
        self.assertIn("Architectures", cells)
        self.assertIn("Start of stream token", cells)

    def test_fill_model_config_table_pad_token(self):
        cfg = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=1,
            bos_token="<s>",
            decoder_start_token_id=None,
            eos_token_id=2,
            eos_token="</s>",
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type="ce",
            max_position_embeddings=10,
            model_type="t",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=4,
            pad_token="<pad>",
            prefix=None,
            sep_token_id=None,
            sep_token=None,
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        table = Table(show_lines=True)
        table.add_column()
        table.add_column()
        filled = self.theme._fill_model_config_table(
            cfg, table, is_runnable=True, summary=False
        )
        cells = filled.columns[0]._cells
        self.assertIn("Padding token", cells)

    def test_tokenizer_config_tokens(self):
        cfg = TokenizerConfig(
            name_or_path="t",
            tokens=["a"],
            special_tokens=["b"],
            tokenizer_model_max_length=10,
            fast=True,
        )
        panel = self.theme.tokenizer_config(cfg)
        headers = panel.renderable.columns[0]._cells
        self.assertIn("Added tokens", headers)

    def test_tokens_table_multiple(self):
        t1 = Token(id=1, token="a", probability=0.1)
        t2 = Token(id=2, token="b", probability=0.2)
        table = self.theme._tokens_table([t1, t2], t1, t2)
        self.assertEqual(table.row_count, 2)
        self.assertIn("[cyan]", table.columns[1]._cells[1])

    def test_parameter_count_none(self):
        self.assertEqual(self.theme._parameter_count(None), "N/A")

    def test_symmetric_indices(self):
        self.assertEqual(
            FancyTheme._symmetric_indices([0.1, 0.5, 0.2, 0.4]),
            [2, 0, 1, 3],
        )

    def test_percentage(self):
        self.assertEqual(FancyTheme._percentage(0.5), "50%")
        self.assertEqual(FancyTheme._percentage(0.123), "12.3%")
        self.assertEqual(FancyTheme._percentage(1), "100%")


class FancyThemeEventsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_no_events(self):
        self.assertIsNone(self.theme.events([]))
        self.assertIsNone(self.theme.events([], events_limit=0))

    def test_single_event(self):
        event = Event(type=EventType.START)
        panel = self.theme.events([event])
        self.assertEqual(panel.height, 4)
        self.assertIn("<start>", str(panel.renderable))

        panel = self.theme.events([event], events_limit=1)
        self.assertEqual(panel.height, 3)
        self.assertIn("<start>", str(panel.renderable))

    def test_multiple_events_with_limit(self):
        e1 = Event(type=EventType.START)
        e2 = Event(type=EventType.END)

        panel = self.theme.events([e1, e2])
        text = str(panel.renderable)
        self.assertEqual(panel.height, 4)
        self.assertIn("<start>", text)
        self.assertIn("<end>", text)

        panel = self.theme.events([e1, e2], events_limit=1)
        text = str(panel.renderable)
        self.assertEqual(panel.height, 3)
        self.assertNotIn("<start>", text)
        self.assertIn("<end>", text)

        panel = self.theme.events([e1, e2], events_limit=2)
        text = str(panel.renderable)
        self.assertEqual(panel.height, 4)
        self.assertIn("<start>", text)
        self.assertIn("<end>", text)
