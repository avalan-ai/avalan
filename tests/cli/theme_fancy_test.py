from datetime import datetime
from types import SimpleNamespace
from uuid import UUID
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from rich.spinner import Spinner
from rich import box
import numpy as np
from numpy.linalg import norm

from avalan.entities import (
    EngineMessage,
    EngineMessageScored,
    HubCache,
    HubCacheDeletion,
    Message,
    ModelConfig,
    SearchMatch,
    SentenceTransformerModelConfig,
    Similarity,
    Token,
    TokenizerConfig,
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
                text_tokens=["a"],
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
                ellapsed=1.0,
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


class FancyThemeTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)

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
            self.theme.ask_access_token(), "Enter your Huggingface access token"
        )
        self.assertEqual(
            self.theme.ask_delete_paths(), "Delete selected paths?"
        )
        self.assertEqual(self.theme.ask_login_to_hub(), "Login to huggingface?")
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
        part = TextPartition(data="t", total_tokens=1, embeddings=np.array([1]))
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
                text_tokens=["<think>", "x", "</think>", "y"],
                tokens=[t],
                input_token_count=0,
                total_tokens=1,
                tool_events=None,
                tool_event_calls=None,
                tool_event_results=None,
                tool_running_spinner=None,
                ttft=0.0,
                ttnt=0.0,
                ellapsed=1.0,
                console_width=40,
                logger=MagicMock(),
                maximum_frames=1,
            )
            frame = await gen.__anext__()
        self.assertTrue(frame[1].renderables)


class FancyThemeAdditionalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)

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
