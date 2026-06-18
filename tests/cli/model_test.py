import asyncio
from argparse import Namespace
from base64 import b64encode
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from gc import collect
from io import StringIO
from logging import getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event as ThreadEvent
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

from async_helpers import run_async
from rich.console import Console

from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator import OrchestratorResponse
from avalan.cli.commands import (
    cache as cache_cmds,
)
from avalan.cli.commands import (
    get_model_settings,
)
from avalan.cli.commands import (
    model as model_cmds,
)
from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.theme import TokenRenderState
from avalan.cli.theme.basic import BasicTheme
from avalan.cli.theme.fancy import FancyTheme
from avalan.cli.theme.stream_presenter import (
    CliStreamAnswerTextChunk,
    CliStreamPresenterRequest,
    LegacyThemeStreamPresenter,
)
from avalan.entities import (
    GenerationCacheStrategy,
    GenerationSettings,
    ImageEntity,
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
    Modality,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallToken,
    TransformerEngineSettings,
)
from avalan.event import Event, EventObservabilityPayload, EventType
from avalan.event.manager import EventManager
from avalan.model.call import ModelCallContext
from avalan.model.manager import ModelManager as RealModelManager
from avalan.model.nlp.text.generation import TextGenerationModel
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamPerformanceBudget,
    StreamProviderEvent,
    StreamTerminalOutcome,
    StreamValidationError,
    project_canonical_stream_item,
    stream_channel_for_kind,
)
from avalan.tool.parser import ToolCallParser


def _text_generation_input(
    prompt: str | None,
    input_file: list[str] | None,
) -> str | Message | None:
    return run_async(model_cmds._text_generation_input(prompt, input_file))


@dataclass(frozen=True)
class _CanonicalAnswerDeltaFixture:
    text: str
    metadata: dict[str, Any]


def _canonical_answer_delta(
    text: str, **metadata: Any
) -> _CanonicalAnswerDeltaFixture:
    return _CanonicalAnswerDeltaFixture(text=text, metadata=dict(metadata))


def _canonical_answer_delta_parts(
    token: str | _CanonicalAnswerDeltaFixture,
) -> tuple[str, dict[str, Any]]:
    if isinstance(token, _CanonicalAnswerDeltaFixture):
        return token.text, token.metadata
    return token, {}


def _canonical_answer_stream_items(
    *tokens: str | _CanonicalAnswerDeltaFixture,
    stream_session_id: str = "stream",
    run_id: str = "run",
    turn_id: str = "turn",
) -> tuple[CanonicalStreamItem, ...]:
    items = [
        CanonicalStreamItem(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
    ]
    sequence = 1
    for token in tokens:
        text, metadata = _canonical_answer_delta_parts(token)
        items.append(
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text,
                metadata=metadata,
            )
        )
        sequence += 1
    items.extend(
        [
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=sequence + 1,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={"output_tokens": len(tokens)},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]
    )
    return tuple(items)


def _canonical_reasoning_answer_stream_items(
    *,
    reasoning: tuple[str, ...],
    answer: tuple[str | _CanonicalAnswerDeltaFixture, ...],
) -> tuple[CanonicalStreamItem, ...]:
    items = [
        CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
    ]
    sequence = 1
    for delta in reasoning:
        items.append(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta=delta,
            )
        )
        sequence += 1
    if reasoning:
        items.append(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            )
        )
        sequence += 1
    for token in answer:
        text, metadata = _canonical_answer_delta_parts(token)
        items.append(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text,
                metadata=metadata,
            )
        )
        sequence += 1
    if answer:
        items.append(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
        )
        sequence += 1
    items.append(
        CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={"output_tokens": len(answer)},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
    )
    return tuple(items)


def _canonical_tool_call_answer_stream_items(
    tool_text: str,
    *answer: str | _CanonicalAnswerDeltaFixture,
) -> tuple[CanonicalStreamItem, ...]:
    tool_call_id = "tool-call"
    correlation = StreamItemCorrelation(tool_call_id=tool_call_id)
    items: list[CanonicalStreamItem] = [
        CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        ),
        CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=correlation,
            text_delta=tool_text,
        ),
        CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=2,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            correlation=correlation,
            data={"name": "tool", "arguments": {}},
        ),
        CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=3,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=correlation,
        ),
    ]
    sequence = 4
    for token in answer:
        text, metadata = _canonical_answer_delta_parts(token)
        items.append(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text,
                metadata=metadata,
            )
        )
        sequence += 1
    if answer:
        items.append(
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
        )
        sequence += 1
    items.append(
        CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={"output_tokens": len(answer)},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
    )
    return tuple(items)


def _disable_mlx_model_import(test_case):
    mlx_model_patch = patch(
        "avalan.model.modalities.text._get_mlx_model", return_value=None
    )
    mlx_model_patch.start()
    test_case.addCleanup(mlx_model_patch.stop)


def _legacy_adapter_theme(token_side_effect: object) -> object:
    class LegacyAdapterTheme:
        def __init__(self) -> None:
            self.token_frames = MagicMock(side_effect=token_side_effect)

        def stream_presenter(
            self,
            logger: Any,
            *,
            event_stats: Any | None = None,
        ) -> LegacyThemeStreamPresenter:
            return LegacyThemeStreamPresenter(
                self,
                logger,
                event_stats=event_stats,
            )

        def tokens(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("tokens should not be called")

    return LegacyAdapterTheme()


class CliModelTestCase(TestCase):
    def setUp(self):
        self.console = MagicMock()
        self.theme = MagicMock()
        self.theme.ask_secret_password.side_effect = lambda k: f"ask-{k}"
        self.logger = MagicMock()
        self.hub = MagicMock()

    def test_get_model_settings(self):
        engine_uri = MagicMock()
        args = Namespace(
            skip_display_reasoning_time=False,
            attention="flash",
            base_url="http://localhost:9001/v1",
            device="cpu",
            disable_loading_progress_bar=True,
            sentence_transformer=True,
            loader_class="auto",
            backend="transformers",
            low_cpu_mem_usage=True,
            quiet=False,
            revision="rev",
            special_token=["<s>"],
            tokenizer="tok",
            token=["t"],
            trust_remote_code=True,
            weight_type="fp16",
            tool_events=2,
            output_hidden_states=False,
        )

        result = get_model_settings(args, self.hub, self.logger, engine_uri)
        expected = {
            "engine_uri": engine_uri,
            "base_url": "http://localhost:9001/v1",
            "attention": "flash",
            "output_hidden_states": False,
            "device": "cpu",
            "parallel": None,
            "disable_loading_progress_bar": True,
            "modality": Modality.EMBEDDING,
            "loader_class": "auto",
            "backend": "transformers",
            "low_cpu_mem_usage": True,
            "quiet": False,
            "revision": "rev",
            "base_model_id": None,
            "checkpoint": None,
            "special_tokens": ["<s>"],
            "tokenizer": "tok",
            "tokens": ["t"],
            "subfolder": None,
            "tokenizer_subfolder": None,
            "refiner_model_id": None,
            "upsampler_model_id": None,
            "trust_remote_code": True,
            "weight_type": "fp16",
        }
        self.assertEqual(result, expected)

    def test_text_generation_input_includes_local_files(self) -> None:
        with NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(b"%PDF-1.7")
            tmp.flush()

            result = _text_generation_input("Summarize", [tmp.name])

        self.assertIsInstance(result, Message)
        self.assertEqual(result.role, MessageRole.USER)
        assert isinstance(result.content, list)
        self.assertEqual(
            result.content[0],
            MessageContentText(type="text", text="Summarize"),
        )
        self.assertEqual(
            result.content[1],
            MessageContentFile(
                type="file",
                file={
                    "file_data": b64encode(b"%PDF-1.7").decode("ascii"),
                    "filename": Path(tmp.name).name,
                    "mime_type": "application/pdf",
                },
            ),
        )

    def test_text_generation_input_uses_files_without_prompt(self) -> None:
        with NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(b"doc")
            tmp.flush()

            result = _text_generation_input(None, [tmp.name])

        self.assertIsInstance(result, Message)
        self.assertEqual(result.role, MessageRole.USER)
        assert isinstance(result.content, list)
        self.assertEqual(len(result.content), 1)
        self.assertEqual(
            result.content[0],
            MessageContentFile(
                type="file",
                file={
                    "file_data": b64encode(b"doc").decode("ascii"),
                    "filename": Path(tmp.name).name,
                    "mime_type": "application/pdf",
                },
            ),
        )

    def test_text_generation_input_rejects_missing_file(self) -> None:
        missing = Path("tests/__missing_input__.pdf").resolve()
        self.assertFalse(missing.exists())

        with self.assertRaisesRegex(
            AssertionError, f"Input file not found: {missing}"
        ):
            _text_generation_input("Summarize", [str(missing)])

    def test_text_generation_input_without_files_returns_input(self) -> None:
        self.assertEqual(_text_generation_input("Hello", None), "Hello")
        self.assertIsNone(_text_generation_input(None, []))

    def test_supports_optional_stdin_only_for_encoder_decoder(self) -> None:
        self.assertTrue(
            model_cmds._supports_optional_stdin(
                Modality.VISION_ENCODER_DECODER
            )
        )
        self.assertFalse(
            model_cmds._supports_optional_stdin(Modality.TEXT_GENERATION)
        )
        self.assertFalse(
            model_cmds._supports_optional_stdin(
                Modality.VISION_IMAGE_CLASSIFICATION
            )
        )

    def test_stream_projection_helpers_use_canonical_adapter(self) -> None:
        reasoning = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta="plan",
        )
        answer = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )
        tool_call = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id="tool-call"),
            text_delta="tool",
        )
        done = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=4,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )

        reasoning_projection = model_cmds._stream_projection(reasoning)
        answer_projection = model_cmds._stream_projection(answer)
        tool_projection = model_cmds._stream_projection(tool_call)
        done_projection = model_cmds._stream_projection(done)

        self.assertEqual(model_cmds._stream_text(reasoning_projection), "plan")
        self.assertEqual(
            model_cmds._stream_text(answer_projection),
            "answer",
        )
        self.assertEqual(model_cmds._stream_text(tool_projection), "tool")
        self.assertIsNone(model_cmds._stream_text(done_projection))
        self.assertTrue(
            model_cmds._is_reasoning_stream_item(reasoning_projection)
        )
        self.assertFalse(model_cmds._is_reasoning_stream_item(tool_projection))
        self.assertTrue(model_cmds._is_tool_stream_item(tool_projection))
        self.assertFalse(model_cmds._is_tool_stream_item(answer_projection))

    def test_stream_projection_helpers_legacy_rejection_invalid_text_input(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported CLI stream item",
        ):
            model_cmds._stream_projection(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            model_cmds._stream_text(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            model_cmds._stream_text(Token(token="raw"))  # type: ignore[arg-type]
        self.assertFalse(
            model_cmds._is_reasoning_stream_item(ReasoningToken(token="raw"))
        )
        self.assertFalse(
            model_cmds._is_tool_stream_item(ToolCallToken(token="raw"))
        )
        self.assertFalse(model_cmds._is_reasoning_stream_item(object()))
        self.assertFalse(model_cmds._is_tool_stream_item(object()))

    def test_model_install_secret_creates_secret(self):
        args = Namespace(skip_display_reasoning_time=False, model="m")
        engine_uri = SimpleNamespace(
            vendor="openai", password="pw", user="secret"
        )
        secrets = MagicMock()
        secrets.read.return_value = None
        with (
            patch.object(
                model_cmds.ModelManager, "parse_uri", return_value=engine_uri
            ),
            patch.object(
                model_cmds, "KeyringSecrets", return_value=secrets
            ) as ks,
            patch.object(model_cmds.Prompt, "ask", return_value="val") as ask,
            patch.object(model_cmds, "cache_download") as cache_download,
            patch.object(model_cmds, "confirm") as confirm,
        ):
            model_cmds.model_install(args, self.console, self.theme, self.hub)

        ks.assert_called_once_with()
        secrets.read.assert_called_once_with("pw")
        ask.assert_called_once_with("ask-pw")
        secrets.write.assert_called_once_with("pw", "val")
        confirm.assert_not_called()
        cache_download.assert_called_once_with(
            args, self.console, self.theme, self.hub
        )

    def test_model_install_passes_download_options(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="m",
            skip_hub_access_check=False,
            workers=7,
            local_dir="/ld",
            local_dir_symlinks=True,
        )
        engine_uri = SimpleNamespace(vendor=None, password=None, user=None)
        self.theme._ = lambda s: s
        self.theme.download_progress.return_value = ("tpl",)
        self.hub.cache_dir = "/cache"
        self.hub.model.return_value = "model"
        self.hub.download.return_value = "/path"
        with (
            patch.object(
                model_cmds.ModelManager, "parse_uri", return_value=engine_uri
            ),
            patch.object(
                cache_cmds, "create_live_tqdm_class", return_value="C"
            ),
        ):
            model_cmds.model_install(args, self.console, self.theme, self.hub)

        self.hub.download.assert_called_once_with(
            "m",
            tqdm_class="C",
            workers=7,
            local_dir="/ld",
            local_dir_use_symlinks=True,
        )

    def test_model_install_secret_no_override_when_declined(self):
        args = Namespace(skip_display_reasoning_time=False, model="m")
        engine_uri = SimpleNamespace(
            vendor="openai", password="pw", user="secret"
        )
        secrets = MagicMock()
        secrets.read.return_value = "tok"
        with (
            patch.object(
                model_cmds.ModelManager, "parse_uri", return_value=engine_uri
            ),
            patch.object(model_cmds, "KeyringSecrets", return_value=secrets),
            patch.object(model_cmds, "cache_download") as cache_download,
            patch.object(model_cmds, "confirm", return_value=False),
            patch.object(model_cmds.Prompt, "ask") as ask,
        ):
            model_cmds.model_install(args, self.console, self.theme, self.hub)

        ask.assert_not_called()
        secrets.write.assert_not_called()
        cache_download.assert_called_once_with(
            args, self.console, self.theme, self.hub
        )

    def test_model_uninstall_secret(self):
        args = Namespace(skip_display_reasoning_time=False, model="m")
        engine_uri = SimpleNamespace(
            vendor="openai", password="pw", user="secret"
        )
        secrets = MagicMock()
        with (
            patch.object(
                model_cmds.ModelManager, "parse_uri", return_value=engine_uri
            ),
            patch.object(
                model_cmds, "KeyringSecrets", return_value=secrets
            ) as ks,
            patch.object(model_cmds, "cache_delete") as cache_delete,
        ):
            model_cmds.model_uninstall(
                args, self.console, self.theme, self.hub
            )

        ks.assert_called_once_with()
        secrets.delete.assert_called_once_with("pw")
        cache_delete.assert_called_once_with(
            args, self.console, self.theme, self.hub, is_full_deletion=True
        )

    def test_model_uninstall_without_secret_skips_keyring(self):
        args = Namespace(skip_display_reasoning_time=False, model="m")
        engine_uri = SimpleNamespace(vendor=None, password=None, user=None)

        with (
            patch.object(
                model_cmds.ModelManager, "parse_uri", return_value=engine_uri
            ),
            patch.object(model_cmds, "KeyringSecrets") as keyring_secrets,
            patch.object(model_cmds, "cache_delete") as cache_delete,
        ):
            model_cmds.model_uninstall(
                args, self.console, self.theme, self.hub
            )

        keyring_secrets.assert_not_called()
        cache_delete.assert_called_once_with(
            args, self.console, self.theme, self.hub, is_full_deletion=True
        )

    def test_model_display_uses_provided_model(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            skip_hub_access_check=False,
            summary=False,
            load=False,
        )
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        engine_uri = SimpleNamespace(is_local=False)
        manager.parse_uri.return_value = engine_uri
        model = SimpleNamespace(config="cfg", tokenizer_config="tok_cfg")
        self.hub.can_access.return_value = True
        self.hub.model.return_value = "hub_model"
        with (
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch.object(model_cmds, "get_model_settings", return_value={}),
        ):
            model_cmds.model_display(
                args,
                self.console,
                self.theme,
                self.hub,
                self.logger,
                model=model,
            )

        manager.parse_uri.assert_called_once_with("id")
        self.hub.can_access.assert_called_once_with("id")
        self.hub.model.assert_called_once_with("id")
        manager.load.assert_not_called()
        self.theme.model.assert_called_once_with(
            "hub_model",
            can_access=True,
            expand=True,
            summary=False,
        )
        self.theme.model_display.assert_called_once_with(
            model.config,
            model.tokenizer_config,
            is_runnable=True,
            summary=False,
        )
        self.console.print.assert_called()

    def test_model_display_loads_model(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            skip_hub_access_check=False,
            summary=False,
            load=True,
        )
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        engine_uri = SimpleNamespace(is_local=False)
        manager.parse_uri.return_value = engine_uri
        lm = MagicMock()
        lm.config = "cfg"
        lm.tokenizer_config = "tok"
        lm.is_runnable.return_value = True
        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False
        manager.load.return_value = load_cm
        self.hub.can_access.return_value = True
        self.hub.model.return_value = "hub_model"
        with (
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch.object(model_cmds, "get_model_settings", return_value={}),
        ):
            model_cmds.model_display(
                args,
                self.console,
                self.theme,
                self.hub,
                self.logger,
                load=True,
            )

        manager.load.assert_called_once()
        self.console.print.assert_called()

    def test_model_install_secret_override(self):
        args = Namespace(skip_display_reasoning_time=False, model="m")
        engine_uri = SimpleNamespace(
            vendor="openai", password="pw", user="secret"
        )
        secrets = MagicMock()
        secrets.read.return_value = "tok"
        with (
            patch.object(
                model_cmds.ModelManager, "parse_uri", return_value=engine_uri
            ),
            patch.object(
                model_cmds, "KeyringSecrets", return_value=secrets
            ) as ks,
            patch.object(model_cmds.Prompt, "ask", return_value="new") as ask,
            patch.object(model_cmds, "cache_download"),
            patch.object(model_cmds, "confirm", return_value=True) as confirm,
        ):
            model_cmds.model_install(args, self.console, self.theme, self.hub)

        ks.assert_called_once_with()
        confirm.assert_called_once()
        ask.assert_called_once_with("ask-pw")
        secrets.write.assert_called_once_with("pw", "new")


class CliTokenGenerationTestCase(IsolatedAsyncioTestCase):
    def _display_config(
        self,
        *,
        display_events: bool = False,
        display_tools: bool = False,
        display_tools_events: int | None = 2,
        interactive: bool = True,
        record: bool = False,
        stats: bool = False,
    ) -> CliStreamDisplayConfig:
        return CliStreamDisplayConfig(
            quiet=False,
            stats=stats,
            display_tools=display_tools,
            display_events=display_events,
            display_tools_events=display_tools_events,
            record=record,
            interactive=interactive,
            refresh_per_second=2,
            answer_height=12,
            answer_height_expand=False,
            display_tokens=0,
            display_pause=0,
            display_probabilities=False,
            display_probabilities_maximum=0.8,
            display_probabilities_sample_minimum=0.1,
            display_time_to_n_token=None,
            display_reasoning_time=True,
        )

    def _empty_response(self) -> object:
        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    await asyncio.sleep(0)
                    if False:
                        yield ""

                return gen()

        return Response()

    def _answer_response(self, text: str = "answer") -> object:
        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    yield text

                return gen()

        return Response()

    def _theme(self) -> object:
        def fake_token_frames(*_args: object, **_kwargs: object):
            return ((None, "frame"),)

        return self._legacy_theme(fake_token_frames)

    def _legacy_theme(self, token_side_effect: object) -> object:
        return _legacy_adapter_theme(token_side_effect)

    def _coordinator_observer_theme(
        self,
        observed: list[object | None],
        coordinator_container: dict[str, object | None],
    ) -> object:
        class RecordingPresenter:
            async def present(self, request: CliStreamPresenterRequest):
                observed.append(coordinator_container.get("coordinator"))
                if request.snapshot.answer_text:
                    yield CliStreamAnswerTextChunk(
                        text=request.snapshot.answer_text
                    )

        class RecordingTheme:
            def stream_presenter(
                self,
                logger: object,
                *,
                event_stats: object | None = None,
            ) -> RecordingPresenter:
                _ = logger, event_stats
                return RecordingPresenter()

            def tokens(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("tokens should not be called")

        return RecordingTheme()

    def _event_manager(
        self,
        *events: Event,
        started: asyncio.Event | None = None,
        error: BaseException | None = None,
    ) -> object:
        class EventManager:
            async def listen(self, *, stop_signal):
                if started is not None:
                    started.set()
                await asyncio.sleep(0)
                if error is not None:
                    raise error
                for event in events:
                    yield event
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        return EventManager()

    def _live_update_renderables(self, live: MagicMock) -> list[object]:
        renderables: list[object] = []
        for update_call in live.update.call_args_list:
            renderable = update_call.args[0]
            if isinstance(renderable, model_cmds.Group):
                renderables.extend(renderable.renderables)
            else:
                renderables.append(renderable)
        return renderables

    def test_stream_presenter_context_normalizes_flags(
        self,
    ):
        class Response:
            input_token_count = 0
            can_think = True
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

        response = Response()
        context = model_cmds._stream_presenter_context(
            Namespace(start_thinking=True, model="args-model"),
            MagicMock(width=72),
            SimpleNamespace(input_token_count=9),
            SimpleNamespace(
                model_id="",
                tokenizer_config=SimpleNamespace(
                    tokens="answer",
                    special_tokens={"<eos>": 2},
                ),
            ),
            "input",
            response,
            display_config=self._display_config(),
            dtokens_pick=1,
        )

        self.assertTrue(response.is_thinking)
        self.assertEqual(context.model_id, "args-model")
        self.assertEqual(context.console_width, 72)
        self.assertEqual(context.input_token_count, 9)
        self.assertEqual(context.tokenizer_tokens, ("answer",))
        self.assertEqual(context.tokenizer_special_tokens, ("<eos>",))
        self.assertTrue(context.start_thinking)

    def test_stream_tokenizer_tuple_edge_cases(self):
        self.assertIsNone(model_cmds._stream_tokenizer_tuple(None))
        self.assertIsNone(model_cmds._stream_tokenizer_tuple(7))

    async def test_token_generation_uses_theme_stream_presenter(self) -> None:
        class RecordingPresenter:
            def __init__(self) -> None:
                self.requests: list[CliStreamPresenterRequest] = []

            async def present(self, request: CliStreamPresenterRequest):
                self.requests.append(request)
                if request.snapshot.answer_text:
                    yield CliStreamAnswerTextChunk(
                        text=request.snapshot.answer_text
                    )

        class RecordingTheme:
            def __init__(self, presenter: RecordingPresenter) -> None:
                self.presenter = presenter
                self.stream_presenter_calls: list[dict[str, object]] = []
                self.tokens_called = False

            def stream_presenter(
                self,
                logger: object,
                *,
                event_stats: object | None = None,
            ) -> RecordingPresenter:
                self.stream_presenter_calls.append(
                    {
                        "logger": logger,
                        "event_stats": event_stats,
                    }
                )
                return self.presenter

            def tokens(self, *_args: object, **_kwargs: object) -> object:
                self.tokens_called = True
                raise AssertionError("tokens should not be called")

        presenter = RecordingPresenter()
        theme = RecordingTheme(presenter)
        console = MagicMock()
        display_config = self._display_config(interactive=False)

        await model_cmds.token_generation(
            args=Namespace(skip_display_reasoning_time=False),
            console=console,
            theme=theme,  # type: ignore[arg-type]
            logger=getLogger(__name__),
            orchestrator=None,
            event_stats=None,
            lm=SimpleNamespace(model_id="m", tokenizer_config=None),
            input_string="i",
            response=self._answer_response("answer"),
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
            tool_events_limit=2,
            refresh_per_second=2,
            display_config=display_config,
        )

        self.assertFalse(theme.tokens_called)
        self.assertEqual(len(theme.stream_presenter_calls), 1)
        self.assertEqual(len(presenter.requests), 1)
        request = presenter.requests[0]
        self.assertIs(request.display_config, display_config)
        self.assertEqual(request.context.model_id, "m")
        self.assertEqual(request.mode, "answer")
        self.assertEqual(request.snapshot.answer_text, "answer")
        console.print.assert_called_once_with("answer", end="")

    async def test_token_generation_fancy_uses_snapshot_presenter(
        self,
    ) -> None:
        items = [
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    for item in items:
                        yield item

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        console = MagicMock()
        console.width = 100
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = FancyTheme(lambda message: message, lambda s, p, n: s)
        display_config = self._display_config(stats=True)

        with (
            patch("avalan.cli.stream_coordinator.Live", return_value=live),
            patch.object(
                FancyTheme,
                "tokens",
                side_effect=AssertionError("legacy tokens should not run"),
            ),
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=display_config,
            )

        rendered = Console(file=StringIO(), record=True, width=120)
        for update_call in live.update.call_args_list:
            rendered.print(update_call.args[0])
        output = rendered.export_text()
        self.assertIn("m response", output)
        self.assertIn("answer", output)
        self.assertIn("1 token in", output)

    async def test_print_plain_stdout_response_filters_channels(
        self,
    ):
        async def gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=1,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="answer",
                ),
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=2,
                    kind=StreamItemKind.ANSWER_DONE,
                    channel=StreamChannel.ANSWER,
                ),
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=3,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ):
                yield item

        console = MagicMock()
        with patch(
            "avalan.cli.commands.model.sleep",
            new_callable=AsyncMock,
        ) as sleep_patch:
            await model_cmds._print_plain_stdout_response(console, gen())

        console.print.assert_called_once_with("answer", end="")
        sleep_patch.assert_awaited_once_with(0)

    async def test_plain_stdout_projections_validates_public_stream(
        self,
    ):
        class Response:
            def consumer_projections(self, **_kwargs: object):
                async def gen():
                    for item in (
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="answer",
                        ),
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DONE,
                            channel=StreamChannel.ANSWER,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=3,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            usage={},
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                    ):
                        yield project_canonical_stream_item(item)

                return gen()

        projections = [
            projection
            async for projection in model_cmds._plain_stdout_projections(
                Response()
            )
        ]

        self.assertEqual(
            [projection.sequence for projection in projections], [0, 1, 2, 3]
        )
        self.assertEqual(projections[1].text_delta, "answer")

    async def test_token_generation_no_stats(self):
        async def gen():
            for item in _canonical_answer_stream_items("a", "b"):
                yield item

        args = Namespace(
            skip_display_reasoning_time=False,
        )
        console = MagicMock()
        await model_cmds.token_generation(
            args=args,
            console=console,
            theme=MagicMock(),
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=MagicMock(),
            input_string="i",
            response=gen(),
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
            tool_events_limit=2,
            refresh_per_second=2,
        )
        console.print.assert_has_calls([call("a", end=""), call("b", end="")])

    async def test_token_generation_no_stats_with_empty_stream(self):
        async def gen():
            if False:
                yield "unreachable"

        args = Namespace(
            skip_display_reasoning_time=False,
        )
        console = MagicMock()
        await model_cmds.token_generation(
            args=args,
            console=console,
            theme=MagicMock(),
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=MagicMock(),
            input_string="i",
            response=gen(),
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
            tool_events_limit=2,
            refresh_per_second=2,
        )

        console.print.assert_not_called()

    async def test_token_generation_no_stats_with_canonical_items(self):
        async def gen():
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        await model_cmds.token_generation(
            args=args,
            console=console,
            theme=MagicMock(),
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=MagicMock(),
            input_string="i",
            response=gen(),
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
            tool_events_limit=2,
            refresh_per_second=2,
        )

        console.print.assert_called_once_with("answer", end="")

    async def test_token_generation_no_stats_prints_empty_answer_delta(self):
        async def gen():
            yield StreamConsumerProjection(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
                correlation=StreamItemCorrelation(),
            )
            yield StreamConsumerProjection(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
                text_delta="",
            )
            yield StreamConsumerProjection(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
                text_delta="answer",
            )
            yield StreamConsumerProjection(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
            )
            yield StreamConsumerProjection(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                correlation=StreamItemCorrelation(),
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        console = MagicMock()
        await model_cmds.token_generation(
            args=Namespace(skip_display_reasoning_time=False),
            console=console,
            theme=MagicMock(),
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=MagicMock(),
            input_string="i",
            response=gen(),
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
            tool_events_limit=2,
            refresh_per_second=2,
        )

        console.print.assert_has_calls(
            [
                call("", end=""),
                call("answer", end=""),
            ]
        )

    async def test_token_generation_no_stats_prints_only_answer_channel(self):
        tool_call_id = "call-1"

        def item(
            sequence: int,
            kind: StreamItemKind,
            channel: StreamChannel,
            *,
            text_delta: str | None = None,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=sequence,
                kind=kind,
                channel=channel,
                correlation=(
                    StreamItemCorrelation(tool_call_id=tool_call_id)
                    if channel is StreamChannel.TOOL_CALL
                    else StreamItemCorrelation()
                ),
                text_delta=text_delta,
                usage=(
                    {} if kind is StreamItemKind.STREAM_COMPLETED else None
                ),
                terminal_outcome=(
                    StreamTerminalOutcome.COMPLETED
                    if kind is StreamItemKind.STREAM_COMPLETED
                    else None
                ),
            )

        async def gen():
            yield item(0, StreamItemKind.STREAM_STARTED, StreamChannel.CONTROL)
            yield project_canonical_stream_item(
                item(
                    1,
                    StreamItemKind.REASONING_DELTA,
                    StreamChannel.REASONING,
                    text_delta="reasoning",
                )
            )
            yield item(
                2,
                StreamItemKind.REASONING_DONE,
                StreamChannel.REASONING,
            )
            yield project_canonical_stream_item(
                item(
                    3,
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamChannel.TOOL_CALL,
                    text_delta="tool",
                )
            )
            yield item(
                4,
                StreamItemKind.TOOL_CALL_READY,
                StreamChannel.TOOL_CALL,
            )
            yield project_canonical_stream_item(
                item(
                    5,
                    StreamItemKind.TOOL_CALL_DONE,
                    StreamChannel.TOOL_CALL,
                )
            )
            yield item(
                6,
                StreamItemKind.ANSWER_DELTA,
                StreamChannel.ANSWER,
                text_delta="answer ",
            )
            yield project_canonical_stream_item(
                item(
                    7,
                    StreamItemKind.ANSWER_DELTA,
                    StreamChannel.ANSWER,
                    text_delta="text",
                )
            )
            yield item(8, StreamItemKind.ANSWER_DONE, StreamChannel.ANSWER)
            yield project_canonical_stream_item(
                item(
                    9,
                    StreamItemKind.STREAM_COMPLETED,
                    StreamChannel.CONTROL,
                )
            )

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        await model_cmds.token_generation(
            args=args,
            console=console,
            theme=MagicMock(),
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=MagicMock(),
            input_string="i",
            response=gen(),
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
            tool_events_limit=2,
            refresh_per_second=2,
        )

        self.assertEqual(
            console.print.call_args_list,
            [call("answer ", end=""), call("text", end="")],
        )

    async def test_token_generation_no_stats_display_tools_renders_live(
        self,
    ) -> None:
        event_reduced = asyncio.Event()
        tool_call = {
            "id": "tool-1",
            "name": "search",
            "arguments": {"query": "weather"},
        }

        class EventManager:
            async def listen(self, *, stop_signal):
                yield Event(
                    type=EventType.TOOL_PROCESS,
                    payload=[tool_call],
                )
                event_reduced.set()
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[str]:
                async def gen() -> AsyncIterator[str]:
                    await event_reduced.wait()
                    yield "answer"

                return gen()

        async def fail_tokens(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("tokens should not be called")

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = self._legacy_theme(fail_tokens)

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_tools=True),
            )

        console.print.assert_called_once_with("answer", end="")
        theme.tokens.assert_not_called()
        renderables = self._live_update_renderables(live)
        self.assertTrue(
            any(
                "tool event tool_process: search" in str(renderable)
                for renderable in renderables
            )
        )

    async def test_token_generation_no_stats_display_events_renders_live(
        self,
    ) -> None:
        event_reduced = asyncio.Event()

        class EventManager:
            async def listen(self, *, stop_signal):
                yield Event(type=EventType.START, payload={"value": "ok"})
                event_reduced.set()
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[str]:
                async def gen() -> AsyncIterator[str]:
                    await event_reduced.wait()
                    yield "answer"

                return gen()

        async def fail_tokens(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("tokens should not be called")

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = self._legacy_theme(fail_tokens)

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_events=True),
            )

        console.print.assert_called_once_with("answer", end="")
        theme.tokens.assert_not_called()
        renderables = self._live_update_renderables(live)
        self.assertTrue(
            any(
                str(renderable).startswith("event start:")
                for renderable in renderables
            )
        )

    async def test_token_generation_basic_no_stats_tools_keep_stdout(
        self,
    ) -> None:
        event_reduced = asyncio.Event()
        tool_call = {
            "id": "tool-1",
            "name": "search",
            "arguments": {"query": "weather"},
        }

        class EventManager:
            async def listen(self, *, stop_signal):
                yield Event(
                    type=EventType.TOOL_PROCESS,
                    payload=[tool_call],
                )
                event_reduced.set()
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[str]:
                async def gen() -> AsyncIterator[str]:
                    await event_reduced.wait()
                    yield "answer"

                return gen()

        class LazyTokenizerLm:
            model_id = "m"

            @property
            def tokenizer_config(self) -> object:
                raise AssertionError("tokenizer config should be opt-in")

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = BasicTheme(lambda message: message, lambda s, p, n: s)

        with (
            patch("avalan.cli.stream_coordinator.Live", return_value=live),
            patch.object(
                BasicTheme,
                "tokens",
                side_effect=AssertionError("tokens should not be called"),
            ) as tokens,
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=LazyTokenizerLm(),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_tools=True),
            )

        tokens.assert_not_called()
        self.assertEqual(
            console.print.call_args_list,
            [call("answer", end=""), call("\n", end="")],
        )
        renderables = self._live_update_renderables(live)
        self.assertTrue(
            any(
                "tool event tool_process: search" in str(renderable)
                for renderable in renderables
            )
        )

    async def test_token_generation_basic_no_stats_events_keep_stdout(
        self,
    ) -> None:
        event_reduced = asyncio.Event()

        class EventManager:
            async def listen(self, *, stop_signal):
                yield Event(type=EventType.START, payload={"value": "ok"})
                event_reduced.set()
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[str]:
                async def gen() -> AsyncIterator[str]:
                    await event_reduced.wait()
                    yield "answer"

                return gen()

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = BasicTheme(lambda message: message, lambda s, p, n: s)

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_events=True),
            )

        self.assertEqual(
            console.print.call_args_list,
            [call("answer", end=""), call("\n", end="")],
        )
        renderables = self._live_update_renderables(live)
        self.assertTrue(
            any(
                str(renderable).startswith("event start:")
                for renderable in renderables
            )
        )

    async def test_token_generation_basic_default_answer_newline_no_live(
        self,
    ) -> None:
        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        theme = BasicTheme(lambda message: message, lambda s, p, n: s)

        with (
            patch("avalan.cli.stream_coordinator.Live") as live,
            patch.object(
                BasicTheme,
                "tokens",
                side_effect=AssertionError("tokens should not be called"),
            ) as tokens,
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=self._answer_response("answer"),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(),
            )

        tokens.assert_not_called()
        live.assert_not_called()
        self.assertEqual(
            console.print.call_args_list,
            [call("answer", end=""), call("\n", end="")],
        )

    async def test_token_generation_basic_non_interactive_uses_stderr(
        self,
    ) -> None:
        event_reduced = asyncio.Event()
        tool_call = {
            "id": "tool-1",
            "name": "search",
            "arguments": {"query": "weather"},
        }

        class EventManager:
            async def listen(self, *, stop_signal):
                yield Event(type=EventType.START, payload={"value": "ok"})
                yield Event(
                    type=EventType.TOOL_PROCESS,
                    payload=[tool_call],
                )
                event_reduced.set()
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[str]:
                async def gen() -> AsyncIterator[str]:
                    await event_reduced.wait()
                    yield "answer"

                return gen()

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        diagnostic_console = MagicMock()
        theme = BasicTheme(lambda message: message, lambda s, p, n: s)

        with (
            patch("avalan.cli.stream_coordinator.Live") as live,
            patch(
                "avalan.cli.stream_coordinator.Console",
                return_value=diagnostic_console,
            ) as stderr_console,
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(
                    display_events=True,
                    display_tools=True,
                    interactive=False,
                    record=True,
                    stats=True,
                ),
            )

        live.assert_not_called()
        stderr_console.assert_called_once_with(
            stderr=True,
            force_terminal=False,
        )
        self.assertEqual(
            console.print.call_args_list,
            [call("answer", end=""), call("\n", end="")],
        )
        diagnostic_text = "\n".join(
            str(args[0])
            for args, _kwargs in diagnostic_console.print.call_args_list
        )
        self.assertIn("event start", diagnostic_text)
        self.assertIn("tool event tool_process: search", diagnostic_text)
        self.assertIn("tokens", diagnostic_text)

    async def test_token_generation_basic_non_interactive_stdout_is_plain(
        self,
    ) -> None:
        protocol_text = (
            '<tool_call>{"name": "calc", "arguments": {}}</tool_call>'
            "<|channel|>analysis<|message|>hidden<|end|>"
            "Plain 25"
        )
        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        diagnostic_console = MagicMock()
        theme = BasicTheme(lambda message: message, lambda s, p, n: s)

        with (
            patch("avalan.cli.stream_coordinator.Live") as live,
            patch(
                "avalan.cli.stream_coordinator.Console",
                return_value=diagnostic_console,
            ),
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=self._answer_response(protocol_text),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(
                    display_tools=True,
                    interactive=False,
                    stats=True,
                ),
            )

        live.assert_not_called()
        stdout_text = "".join(
            str(args[0]) for args, _kwargs in console.print.call_args_list
        )
        self.assertEqual(stdout_text, "Plain 25\n")
        for fragment in (
            "\x1b",
            "[",
            "Spinner",
            "<tool_call",
            "arguments",
            "<|channel|>",
            "hidden",
            "tool ",
            "tokens ",
        ):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, stdout_text)
        diagnostic_text = "\n".join(
            str(args[0])
            for args, _kwargs in diagnostic_console.print.call_args_list
        )
        self.assertIn("tokens", diagnostic_text)

    async def test_token_generation_basic_combined_tools_events_are_distinct(
        self,
    ) -> None:
        event_reduced = asyncio.Event()
        tool_call_id = "call-1"
        tool_call = {
            "id": tool_call_id,
            "name": "calc",
            "arguments": {"expression": "(4 + 6) * 5 / 2"},
        }

        def item(
            sequence: int,
            kind: StreamItemKind,
            *,
            text_delta: str | None = None,
            data: dict[str, object] | None = None,
            usage: dict[str, object] | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
        ) -> CanonicalStreamItem:
            channel = stream_channel_for_kind(kind)
            return CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=kind,
                channel=channel,
                correlation=(
                    StreamItemCorrelation(tool_call_id=tool_call_id)
                    if channel
                    in (StreamChannel.TOOL_CALL, StreamChannel.TOOL_EXECUTION)
                    else StreamItemCorrelation()
                ),
                text_delta=text_delta,
                data=data,
                usage=usage,
                terminal_outcome=terminal_outcome,
            )

        class EventManager:
            async def listen(self, *, stop_signal):
                yield Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={"token": "hidden"},
                )
                yield Event(type=EventType.TOOL_PROCESS, payload=[tool_call])
                event_reduced.set()
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    await event_reduced.wait()
                    yield item(0, StreamItemKind.STREAM_STARTED)
                    yield item(
                        1,
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        text_delta='{"expression": "(4 + 6) * 5 / 2"}',
                    )
                    yield item(
                        2,
                        StreamItemKind.TOOL_CALL_READY,
                        data={"name": "calc"},
                    )
                    yield item(3, StreamItemKind.TOOL_CALL_DONE)
                    yield item(
                        4,
                        StreamItemKind.TOOL_EXECUTION_STARTED,
                        data={"name": "calc"},
                    )
                    yield item(
                        5,
                        StreamItemKind.TOOL_EXECUTION_COMPLETED,
                        data={"name": "calc", "result": 25},
                    )
                    yield item(
                        6,
                        StreamItemKind.ANSWER_DELTA,
                        text_delta="25",
                    )
                    yield item(7, StreamItemKind.ANSWER_DONE)
                    yield item(
                        8,
                        StreamItemKind.STREAM_COMPLETED,
                        usage={
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    )

                return gen()

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = BasicTheme(lambda message: message, lambda s, p, n: s)

        with (
            patch("avalan.cli.stream_coordinator.Live", return_value=live),
            patch.object(
                BasicTheme,
                "tokens",
                side_effect=AssertionError("tokens should not be called"),
            ) as tokens,
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(
                    display_events=True,
                    display_tools=True,
                ),
            )

        tokens.assert_not_called()
        self.assertEqual(
            console.print.call_args_list,
            [call("25", end=""), call("\n", end="")],
        )
        self.assertGreater(live.update.call_count, 0)
        render_console = Console(file=StringIO(), record=True, width=160)
        for update_call in live.update.call_args_list:
            render_console.print(update_call.args[0])
        all_live_text = render_console.export_text()

        final_console = Console(file=StringIO(), record=True, width=160)
        final_console.print(live.update.call_args_list[-1].args[0])
        final_text = final_console.export_text()

        self.assertNotIn("token_generated", all_live_text)
        self.assertEqual(final_text.count("tool calc completed"), 1)
        self.assertEqual(final_text.count("tool calc result:"), 1)
        self.assertNotIn("tool event tool_process", final_text)
        self.assertNotIn("event tool_process", final_text)

    async def test_token_generation_display_events_uses_response_projection(
        self,
    ) -> None:
        def item(
            sequence: int,
            kind: StreamItemKind,
            *,
            text_delta: str | None = None,
            data: dict[str, object] | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
            usage: dict[str, object] | None = None,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=kind,
                channel=stream_channel_for_kind(kind),
                text_delta=text_delta,
                data=data,
                terminal_outcome=terminal_outcome,
                usage=usage,
            )

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def consumer_projections(
                self,
                **_kwargs: object,
            ) -> AsyncIterator[StreamConsumerProjection]:
                async def gen() -> AsyncIterator[StreamConsumerProjection]:
                    for stream_item in (
                        item(0, StreamItemKind.STREAM_STARTED),
                        item(
                            1,
                            StreamItemKind.FLOW_EVENT,
                            data={"stage": "projection"},
                        ),
                        item(
                            2,
                            StreamItemKind.ANSWER_DELTA,
                            text_delta="answer",
                        ),
                        item(3, StreamItemKind.ANSWER_DONE),
                        item(
                            4,
                            StreamItemKind.STREAM_COMPLETED,
                            usage={},
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                        ),
                    ):
                        yield project_canonical_stream_item(stream_item)

                return gen()

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = BasicTheme(lambda message: message, lambda s, p, n: s)

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_events=True),
            )

        self.assertEqual(
            console.print.call_args_list,
            [call("answer", end=""), call("\n", end="")],
        )
        renderables = self._live_update_renderables(live)
        rendered_text = "\n".join(
            str(renderable) for renderable in renderables
        )
        self.assertIn("event flow.event:", rendered_text)
        self.assertIn("projection", rendered_text)

    async def test_token_generation_no_stats_rejects_late_projection(self):
        async def gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=1,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
                CanonicalStreamItem(
                    stream_session_id="s",
                    run_id="r",
                    turn_id="t",
                    sequence=2,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="late",
                ),
            ):
                yield project_canonical_stream_item(item)

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()
        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=gen(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
            )
        console.print.assert_not_called()

    async def test_token_generation_display_events_without_orchestrator_plain(
        self,
    ):
        args = Namespace(
            quiet=False,
            stats=False,
            display_events=True,
            display_tools=False,
            display_tools_events=2,
            record=False,
            skip_display_reasoning_time=False,
        )

        console = MagicMock()
        with patch("avalan.cli.stream_coordinator.Live") as live:
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=self._answer_response("answer"),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_events=True),
            )

        console.print.assert_called_once_with("answer", end="")
        live.assert_not_called()

    async def test_token_generation_display_tools_without_stats_starts_events(
        self,
    ):
        args = Namespace(
            quiet=False,
            stats=False,
            display_events=False,
            display_tools=True,
            display_tools_events=2,
            record=False,
            skip_display_reasoning_time=False,
        )
        event_started = asyncio.Event()
        orchestrator = SimpleNamespace(
            event_manager=self._event_manager(
                Event(type=EventType.TOOL_PROCESS, payload=[]),
                started=event_started,
            )
        )

        async def fail_tokens(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("tokens should not be called")

        coordinator_container: dict[str, object | None] = {}
        console = MagicMock()
        theme = self._legacy_theme(fail_tokens)
        await model_cmds.token_generation(
            args=args,
            console=console,
            theme=theme,
            logger=MagicMock(),
            orchestrator=orchestrator,
            event_stats=None,
            lm=MagicMock(),
            input_string="i",
            response=self._answer_response("answer"),
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
            tool_events_limit=2,
            refresh_per_second=2,
            coordinator_container=coordinator_container,
            display_config=self._display_config(display_tools=True),
        )

        console.print.assert_called_once_with("answer", end="")
        theme.tokens.assert_not_called()
        self.assertTrue(event_started.is_set())
        self.assertIsNone(coordinator_container["coordinator"])

    async def test_token_generation_display_tools_plain_error_cleans_live(
        self,
    ):
        args = Namespace(
            quiet=False,
            stats=False,
            display_events=False,
            display_tools=True,
            display_tools_events=2,
            record=False,
            skip_display_reasoning_time=False,
        )
        coordinator_container: dict[str, object | None] = {}

        class FailingResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    raise ValueError("plain failed")
                    yield ""

                return gen()

        with self.assertRaisesRegex(ValueError, "plain failed"):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(),
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(
                    event_manager=self._event_manager()
                ),
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=FailingResponse(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                coordinator_container=coordinator_container,
                display_config=self._display_config(display_tools=True),
            )

        self.assertIsNone(coordinator_container["coordinator"])

    async def test_token_generation_display_tools_raises_event_error(self):
        args = Namespace(
            quiet=False,
            stats=False,
            display_events=False,
            display_tools=True,
            display_tools_events=2,
            record=False,
            skip_display_reasoning_time=False,
        )
        with self.assertRaisesRegex(RuntimeError, "event failed"):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(),
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(
                    event_manager=self._event_manager(
                        error=RuntimeError("event failed")
                    )
                ),
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=self._empty_response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_tools=True),
            )

    async def test_token_generation_quiet_all_flags_stays_answer_clean(self):
        async def gen():
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="tool"),
                text_delta="tool",
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="tool"),
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=3,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=StreamItemCorrelation(tool_call_id="tool"),
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=4,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=5,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=6,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        args = Namespace(
            quiet=True,
            stats=True,
            display_events=True,
            display_tools=True,
            display_tools_events=3,
            record=True,
            display_tokens=4,
            display_pause=100,
            display_probabilities=True,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            display_time_to_n_token=2,
            skip_display_reasoning_time=False,
        )
        console = MagicMock()

        with (
            patch("avalan.cli.stream_coordinator.Live") as live,
            patch.object(
                model_cmds,
                "_event_stream",
                new_callable=AsyncMock,
            ) as event_stream,
            patch.object(
                model_cmds,
                "_token_stream",
                new_callable=AsyncMock,
            ) as token_stream,
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(event_manager=MagicMock()),
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=gen(),
                display_tokens=4,
                dtokens_pick=4,
                with_stats=True,
                tool_events_limit=3,
                refresh_per_second=2,
            )

        console.print.assert_called_once_with("answer", end="")
        live.assert_not_called()
        event_stream.assert_not_called()
        token_stream.assert_not_called()

    async def test_token_generation_non_interactive_record_stays_answer_clean(
        self,
    ) -> None:
        args = Namespace(
            quiet=False,
            stats=True,
            display_events=True,
            display_tools=True,
            display_tools_events=3,
            record=True,
            display_tokens=4,
            display_pause=100,
            display_probabilities=True,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            display_time_to_n_token=2,
            skip_display_reasoning_time=False,
        )
        console = MagicMock()

        with patch("avalan.cli.stream_coordinator.Live") as live:
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(event_manager=MagicMock()),
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=self._answer_response("answer"),
                display_tokens=4,
                dtokens_pick=4,
                with_stats=True,
                tool_events_limit=3,
                refresh_per_second=2,
                display_config=self._display_config(
                    display_events=True,
                    display_tools=True,
                    display_tools_events=3,
                    interactive=False,
                    record=True,
                    stats=True,
                ),
            )

        console.print.assert_called_once_with("answer", end="")
        live.assert_not_called()
        console.save_svg.assert_not_called()

    async def test_token_generation_mixed_live_frames_record_once_owner(
        self,
    ) -> None:
        tool_call_id = "call-1"

        def item(
            sequence: int,
            kind: StreamItemKind,
            *,
            text_delta: str | None = None,
            data: dict[str, object] | None = None,
            usage: dict[str, object] | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
            tool_call: bool = False,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=kind,
                channel=stream_channel_for_kind(kind),
                correlation=(
                    StreamItemCorrelation(tool_call_id=tool_call_id)
                    if tool_call
                    else StreamItemCorrelation()
                ),
                text_delta=text_delta,
                data=data,
                usage=usage,
                terminal_outcome=terminal_outcome,
            )

        items = [
            item(0, StreamItemKind.STREAM_STARTED),
            item(
                1,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                text_delta='{"x": 2}',
                tool_call=True,
            ),
            item(
                2,
                StreamItemKind.TOOL_CALL_READY,
                data={"name": "calc"},
                tool_call=True,
            ),
            item(3, StreamItemKind.TOOL_CALL_DONE, tool_call=True),
            item(
                4,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                data={"name": "calc"},
                tool_call=True,
            ),
            item(
                5,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                text_delta="4",
                data={"name": "calc"},
                tool_call=True,
            ),
            item(
                6,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                data={"name": "calc", "result": 4},
                tool_call=True,
            ),
            item(
                7,
                StreamItemKind.FLOW_EVENT,
                data={"node": "math"},
            ),
            item(8, StreamItemKind.ANSWER_DELTA, text_delta="done"),
            item(9, StreamItemKind.ANSWER_DONE),
            item(
                10,
                StreamItemKind.STREAM_COMPLETED,
                usage={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class MixedResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    for stream_item in items:
                        yield stream_item
                        await asyncio.sleep(0)

                return gen()

        token_frames: list[dict[str, object]] = []

        def fake_token_frames(
            state: object,
            **kwargs: object,
        ) -> tuple[tuple[None, object], ...]:
            token_frames.append({"state": state, **kwargs})
            return ((None, f"stream-frame-{len(token_frames)}"),)

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=True,
        )
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = self._legacy_theme(fake_token_frames)
        display_config = self._display_config(
            display_events=True,
            display_tools=True,
            record=True,
            stats=True,
        )

        with patch(
            "avalan.cli.stream_coordinator.Live", return_value=live
        ) as live_patch:
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=MixedResponse(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=display_config,
            )

        live_patch.assert_called_once_with(
            None,
            console=console,
            refresh_per_second=2,
            screen=True,
        )
        live.__enter__.assert_called_once_with()
        live.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(console.save_svg.call_count, live.update.call_count)
        self.assertGreater(live.update.call_count, 0)

        renderables = self._live_update_renderables(live)
        rendered_text = "\n".join(
            str(renderable) for renderable in renderables
        )
        self.assertIn("completed tool calc", rendered_text)
        self.assertIn("tool result calc", rendered_text)
        self.assertIn("event flow.event", rendered_text)
        self.assertIn("usage stream.completed", rendered_text)
        self.assertTrue(
            any(
                str(renderable).startswith("stream-frame-")
                for renderable in renderables
            )
        )

    async def test_token_generation_event_heavy_stream_is_gated_final(
        self,
    ) -> None:
        tool_call_id = "call-1"

        def item(
            sequence: int,
            kind: StreamItemKind,
            *,
            text_delta: str | None = None,
            data: dict[str, object] | None = None,
            usage: dict[str, object] | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
            tool_call: bool = False,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=kind,
                channel=stream_channel_for_kind(kind),
                correlation=(
                    StreamItemCorrelation(tool_call_id=tool_call_id)
                    if tool_call
                    else StreamItemCorrelation()
                ),
                text_delta=text_delta,
                data=data,
                usage=usage,
                terminal_outcome=terminal_outcome,
            )

        items = [
            item(0, StreamItemKind.STREAM_STARTED),
            item(
                1,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                text_delta='{"x": 2}',
                tool_call=True,
            ),
            item(
                2,
                StreamItemKind.TOOL_CALL_READY,
                data={"name": "calc"},
                tool_call=True,
            ),
            item(3, StreamItemKind.TOOL_CALL_DONE, tool_call=True),
            item(
                4,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                data={"name": "calc"},
                tool_call=True,
            ),
            item(
                5,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                text_delta="4",
                data={"name": "calc"},
                tool_call=True,
            ),
            item(
                6,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                data={"name": "calc", "result": 4},
                tool_call=True,
            ),
            item(7, StreamItemKind.FLOW_EVENT, data={"node": "math"}),
            item(8, StreamItemKind.ANSWER_DELTA, text_delta="done"),
            item(9, StreamItemKind.ANSWER_DONE),
            item(
                10,
                StreamItemKind.STREAM_COMPLETED,
                usage={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]
        consumed_sequences: list[int] = []

        class MixedResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    for stream_item in items:
                        consumed_sequences.append(stream_item.sequence)
                        yield stream_item
                        await asyncio.sleep(0)

                return gen()

        class EventManager:
            async def listen(self, *, stop_signal):
                for index in range(25):
                    yield Event(
                        type=EventType.START,
                        payload={"index": index},
                    )
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        def fake_token_frames(
            state: TokenRenderState, **_kwargs: object
        ) -> tuple[tuple[None, str], ...]:
            answer = "".join(state.answer_text_tokens)
            return ((None, f"stream-frame-{answer or 'pending'}"),)

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=True,
        )
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = self._legacy_theme(fake_token_frames)
        display_config = self._display_config(
            display_events=True,
            display_tools=True,
            record=True,
            stats=True,
        )

        with (
            patch(
                "avalan.cli.stream_coordinator.perf_counter",
                return_value=0.0,
            ),
            patch("avalan.cli.stream_coordinator.Live", return_value=live),
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(
                    event_manager=EventManager(),
                ),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=MixedResponse(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=display_config,
            )

        self.assertEqual(consumed_sequences, list(range(len(items))))
        self.assertLessEqual(live.update.call_count, 2)
        self.assertEqual(console.save_svg.call_count, live.update.call_count)
        self.assertGreater(live.update.call_count, 0)

        final_renderable = live.update.call_args_list[-1].args[0]
        final_renderables = (
            list(cast(model_cmds.Group, final_renderable).renderables)
            if isinstance(final_renderable, model_cmds.Group)
            else [final_renderable]
        )
        final_text = "\n".join(
            str(renderable) for renderable in final_renderables
        )
        self.assertIn("stream-frame-done", final_text)
        self.assertIn("completed tool calc", final_text)
        self.assertIn("tool result calc", final_text)
        self.assertIn("event flow.event", final_text)
        self.assertIn("usage stream.completed", final_text)

    async def test_token_generation_ignores_token_generated_event(
        self,
    ) -> None:
        event_reduced = asyncio.Event()

        def item(
            sequence: int,
            kind: StreamItemKind,
            *,
            text_delta: str | None = None,
            usage: dict[str, object] | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=kind,
                channel=stream_channel_for_kind(kind),
                text_delta=text_delta,
                usage=usage,
                terminal_outcome=terminal_outcome,
            )

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    await event_reduced.wait()
                    yield item(0, StreamItemKind.STREAM_STARTED)
                    yield item(
                        1,
                        StreamItemKind.ANSWER_DELTA,
                        text_delta="done",
                    )
                    yield item(2, StreamItemKind.ANSWER_DONE)
                    yield item(
                        3,
                        StreamItemKind.STREAM_COMPLETED,
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    )

                return gen()

        class EventManager:
            async def listen(self, *, stop_signal):
                yield Event(type=EventType.TOKEN_GENERATED)
                event_reduced.set()
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=Namespace(skip_display_reasoning_time=False),
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
                display_config=self._display_config(display_events=True),
            )

        self.assertTrue(event_reduced.is_set())

    async def test_token_generation_ignores_post_terminal_side_event(
        self,
    ) -> None:
        terminal_processed = asyncio.Event()

        def item(
            sequence: int,
            kind: StreamItemKind,
            *,
            text_delta: str | None = None,
            usage: dict[str, object] | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=kind,
                channel=stream_channel_for_kind(kind),
                text_delta=text_delta,
                usage=usage,
                terminal_outcome=terminal_outcome,
            )

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    yield item(0, StreamItemKind.STREAM_STARTED)
                    yield item(
                        1,
                        StreamItemKind.ANSWER_DELTA,
                        text_delta="done",
                    )
                    yield item(2, StreamItemKind.ANSWER_DONE)
                    yield item(
                        3,
                        StreamItemKind.STREAM_COMPLETED,
                        usage={
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    )
                    terminal_processed.set()
                    await asyncio.sleep(0)

                return gen()

        class EventManager:
            async def listen(self, *, stop_signal):
                _ = stop_signal
                await terminal_processed.wait()
                yield Event(
                    type=EventType.START,
                    payload={"post": "terminal"},
                )
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        def fake_token_frames(
            state: TokenRenderState, **_kwargs: object
        ) -> tuple[tuple[None, str], ...]:
            answer = "".join(state.answer_text_tokens)
            return ((None, f"stream-frame-{answer or 'pending'}"),)

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=True,
        )
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = self._legacy_theme(fake_token_frames)

        with (
            patch(
                "avalan.cli.stream_coordinator.perf_counter",
                return_value=0.0,
            ),
            patch("avalan.cli.stream_coordinator.Live", return_value=live),
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(
                    display_events=True,
                    record=True,
                    stats=True,
                ),
            )

        self.assertEqual(live.update.call_count, 2)
        self.assertEqual(console.save_svg.call_count, 2)
        final_renderable = live.update.call_args_list[-1].args[0]
        final_renderables = (
            list(cast(model_cmds.Group, final_renderable).renderables)
            if isinstance(final_renderable, model_cmds.Group)
            else [final_renderable]
        )
        final_text = "\n".join(
            str(renderable) for renderable in final_renderables
        )
        self.assertIn("stream-frame-done", final_text)
        self.assertIn("usage stream.completed", final_text)
        self.assertNotIn("event start", final_text)
        self.assertNotIn("terminal", final_text)

    async def test_token_generation_ignores_event_before_terminal_lock(
        self,
    ) -> None:
        terminal_read = asyncio.Event()
        event_yielded = asyncio.Event()

        def item(
            sequence: int,
            kind: StreamItemKind,
            *,
            text_delta: str | None = None,
            usage: dict[str, object] | None = None,
            terminal_outcome: StreamTerminalOutcome | None = None,
        ) -> CanonicalStreamItem:
            return CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=sequence,
                kind=kind,
                channel=stream_channel_for_kind(kind),
                text_delta=text_delta,
                usage=usage,
                terminal_outcome=terminal_outcome,
            )

        class RaceLock:
            yielded_to_event = False

            def __init__(self) -> None:
                self._lock = asyncio.Lock()

            async def __aenter__(self) -> "RaceLock":
                if terminal_read.is_set() and not RaceLock.yielded_to_event:
                    RaceLock.yielded_to_event = True
                    await asyncio.sleep(0)
                await self._lock.acquire()
                return self

            async def __aexit__(self, *_exc_info: object) -> bool:
                self._lock.release()
                return False

        async def fake_stream(
            *_args: object,
            **_kwargs: object,
        ) -> AsyncIterator[StreamConsumerProjection]:
            for stream_item in (
                item(0, StreamItemKind.STREAM_STARTED),
                item(1, StreamItemKind.ANSWER_DELTA, text_delta="done"),
                item(2, StreamItemKind.ANSWER_DONE),
            ):
                yield project_canonical_stream_item(stream_item)
            terminal_read.set()
            yield project_canonical_stream_item(
                item(
                    3,
                    StreamItemKind.STREAM_COMPLETED,
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                )
            )

        class EventManager:
            async def listen(self, *, stop_signal):
                await terminal_read.wait()
                event_yielded.set()
                yield Event(
                    type=EventType.START,
                    payload={"race": "before-terminal-lock"},
                )
                while not stop_signal.is_set():
                    await asyncio.sleep(0)

        def fake_token_frames(
            state: TokenRenderState, **_kwargs: object
        ) -> tuple[tuple[None, str], ...]:
            answer = "".join(state.answer_text_tokens)
            return ((None, f"stream-frame-{answer or 'pending'}"),)

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=True,
        )
        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        theme = self._legacy_theme(fake_token_frames)

        with (
            patch.object(model_cmds, "Lock", RaceLock),
            patch.object(
                model_cmds, "_stream_render_projections", fake_stream
            ),
            patch(
                "avalan.cli.stream_coordinator.perf_counter",
                return_value=0.0,
            ),
            patch("avalan.cli.stream_coordinator.Live", return_value=live),
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=getLogger(__name__),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=self._empty_response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(
                    display_events=True,
                    record=True,
                    stats=True,
                ),
            )

        self.assertTrue(event_yielded.is_set())
        self.assertTrue(RaceLock.yielded_to_event)
        self.assertGreater(live.update.call_count, 0)
        final_renderable = live.update.call_args_list[-1].args[0]
        final_renderables = (
            list(cast(model_cmds.Group, final_renderable).renderables)
            if isinstance(final_renderable, model_cmds.Group)
            else [final_renderable]
        )
        final_text = "\n".join(
            str(renderable) for renderable in final_renderables
        )
        self.assertIn("stream-frame-done", final_text)
        self.assertIn("usage stream.completed", final_text)
        self.assertNotIn("event start", final_text)
        self.assertNotIn("before-terminal-lock", final_text)

    async def test_stream_render_projections_projects_canonical_items(
        self,
    ):
        async def gen():
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        observed = [
            item
            async for item in model_cmds._stream_render_projections(
                gen(),
                stream_session_id="fallback-stream",
                run_id="fallback-run",
                turn_id="fallback-turn",
            )
        ]

        self.assertEqual(
            [item.sequence for item in observed],
            [0, 1, 2, 3],
        )
        self.assertEqual(observed[0].stream_session_id, "stream")
        self.assertEqual(observed[1].text_delta, "answer")
        self.assertIs(
            observed[3].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    async def test_stream_render_projections_projects_consumer_projections(
        self,
    ):
        items = (
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="plan",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        async def gen():
            yield project_canonical_stream_item(items[0])
            yield project_canonical_stream_item(items[1])
            yield project_canonical_stream_item(items[2])
            yield project_canonical_stream_item(items[3])

        observed = [
            item
            async for item in model_cmds._stream_render_projections(
                gen(),
                stream_session_id="fallback-stream",
                run_id="fallback-run",
                turn_id="fallback-turn",
            )
        ]

        self.assertEqual(
            [item.sequence for item in observed],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            {item.stream_session_id for item in observed}, {"stream"}
        )
        self.assertEqual(observed[1].text_delta, "plan")
        self.assertIs(
            observed[3].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    async def test_render_projections_accepts_mixed_semantic_stream(self):
        async def gen():
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield project_canonical_stream_item(
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=1,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="answer",
                )
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        observed = [
            item
            async for item in model_cmds._stream_render_projections(
                gen(),
                stream_session_id="fallback-stream",
                run_id="fallback-run",
                turn_id="fallback-turn",
            )
        ]

        self.assertEqual([item.sequence for item in observed], [0, 1, 2, 3])
        self.assertEqual(observed[1].text_delta, "answer")

    async def test_stream_render_projections_rejects_late_projection(self):
        async def gen():
            for item in (
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=0,
                    kind=StreamItemKind.STREAM_STARTED,
                    channel=StreamChannel.CONTROL,
                ),
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=1,
                    kind=StreamItemKind.STREAM_COMPLETED,
                    channel=StreamChannel.CONTROL,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=2,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="late",
                ),
            ):
                yield project_canonical_stream_item(item)

        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            [
                item
                async for item in model_cmds._stream_render_projections(
                    gen(),
                    stream_session_id="fallback-stream",
                    run_id="fallback-run",
                    turn_id="fallback-turn",
                )
            ]

    async def test_render_projections_overhead_within_budget(self):
        count = 1000
        items = (
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            *(
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=sequence,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta="x",
                )
                for sequence in range(1, count + 1)
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=count + 1,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=count + 2,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={"output_tokens": count},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )
        budget = StreamPerformanceBudget()

        async def gen():
            for item in items:
                yield item

        started = perf_counter()
        observed = [
            item
            async for item in model_cmds._stream_render_projections(
                gen(),
                stream_session_id="fallback-stream",
                run_id="fallback-run",
                turn_id="fallback-turn",
            )
        ]
        elapsed_us = (perf_counter() - started) * 1_000_000

        self.assertEqual(len(observed), count + 3)
        self.assertLessEqual(
            elapsed_us / len(observed),
            budget.per_item_overhead_us,
        )

    async def test_projection_display_token_uses_canonical_metadata(self):
        stream_delta = _canonical_answer_delta("answer", token_id=1)

        async def gen():
            for item in _canonical_answer_stream_items(stream_delta):
                yield item

        observed = [
            item
            async for item in model_cmds._stream_render_projections(
                gen(),
                stream_session_id="fallback-stream",
                run_id="fallback-run",
                turn_id="fallback-turn",
            )
        ]

        self.assertEqual(len(observed), 4)
        self.assertEqual(observed[1].text_delta, "answer")
        display_token = model_cmds._projection_display_token(observed[1])
        assert display_token is not None
        self.assertEqual(display_token.token, "answer")
        self.assertEqual(display_token.id, 1)
        self.assertEqual(display_token.sequence, observed[1].sequence)

    async def test_projection_display_token_rejects_non_projection(
        self,
    ):
        with self.assertRaises(AssertionError):
            model_cmds._projection_display_token(object())  # type: ignore[arg-type]

    async def test_projection_display_token_normalizes_missing_text(self):
        projection = object.__new__(StreamConsumerProjection)
        object.__setattr__(projection, "kind", StreamItemKind.ANSWER_DELTA)
        object.__setattr__(projection, "text_delta", None)
        object.__setattr__(projection, "metadata", {"token_id": 1})
        object.__setattr__(projection, "sequence", 1)
        object.__setattr__(projection, "channel", StreamChannel.ANSWER)

        display_token = model_cmds._projection_display_token(projection)

        assert display_token is not None
        self.assertEqual(display_token.token, "")
        self.assertEqual(display_token.id, 1)

    def test_projection_display_token_candidates_filter_metadata(self):
        candidates = model_cmds._projection_display_token_candidates(
            [
                object(),
                {},
                {"token": 1},
                {
                    "token": "a",
                    "token_id": 3,
                    "probability": "bad",
                },
                {
                    "token": "b",
                    "token_id": "bad",
                    "probability": 0.5,
                },
            ]
        )

        self.assertEqual(
            candidates,
            (
                model_cmds.TokenRenderDisplayTokenCandidate(
                    token="a", id=3, probability=None
                ),
                model_cmds.TokenRenderDisplayTokenCandidate(
                    token="b", id=None, probability=0.5
                ),
            ),
        )

    async def test_render_projections_rejects_legacy_mixed_stream(self):
        async def gen():
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield "legacy"

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported CLI stream item",
        ):
            [
                item
                async for item in model_cmds._stream_render_projections(
                    gen(),
                    stream_session_id="fallback-stream",
                    run_id="fallback-run",
                    turn_id="fallback-turn",
                )
            ]

    async def test_render_projections_rejects_unsupported_item(self):
        async def gen():
            yield object()

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported CLI stream item",
        ):
            [
                item
                async for item in model_cmds._stream_render_projections(
                    gen(),
                    stream_session_id="fallback-stream",
                    run_id="fallback-run",
                    turn_id="fallback-turn",
                )
            ]

    async def test_stream_render_projections_rejects_legacy_event_item(self):
        async def gen():
            yield Event(type=EventType.START)

        with self.assertRaisesRegex(
            StreamValidationError,
            "unsupported CLI stream item",
        ):
            [
                item
                async for item in model_cmds._stream_render_projections(
                    gen(),
                    stream_session_id="fallback-stream",
                    run_id="fallback-run",
                    turn_id="fallback-turn",
                )
            ]

    async def test_stream_render_projections_legacy_rejection_first_item(
        self,
    ):
        canonical_item = CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )

        legacy_first_items = (
            "legacy",
            Token(token="legacy"),
            TokenDetail(token="legacy", id=1),
            ReasoningToken(token="legacy"),
            ToolCallToken(
                token="{}",
                call=ToolCall(
                    id="legacy-call",
                    name="legacy",
                    arguments={},
                ),
            ),
            Event(type=EventType.START),
        )

        for legacy_first_item in legacy_first_items:
            for item in (
                canonical_item,
                project_canonical_stream_item(canonical_item),
            ):
                with self.subTest(
                    legacy_item_type=type(legacy_first_item).__name__,
                    item_type=type(item).__name__,
                ):

                    async def gen():
                        yield legacy_first_item
                        yield item

                    with self.assertRaisesRegex(
                        StreamValidationError,
                        "unsupported CLI stream item",
                    ):
                        stream = model_cmds._stream_render_projections(
                            gen(),
                            stream_session_id="fallback-stream",
                            run_id="fallback-run",
                            turn_id="fallback-turn",
                        )
                        [render_item async for render_item in stream]

    async def test_render_projections_prefers_consumer_projections(self):
        class Response:
            def __init__(self) -> None:
                self.raw_iterated = False
                self.projection_requested = False

            def __aiter__(self):
                self.raw_iterated = True

                async def gen():
                    yield object()

                return gen()

            def consumer_projections(
                self,
                *,
                stream_session_id: str,
                run_id: str,
                turn_id: str,
            ):
                self.projection_requested = True

                async def gen():
                    for item in (
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="projected",
                        ),
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DONE,
                            channel=StreamChannel.ANSWER,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=3,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            usage={},
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                        ),
                    ):
                        yield project_canonical_stream_item(item)

                return gen()

        response = Response()

        items = [
            item
            async for item in model_cmds._stream_render_projections(
                response,
                stream_session_id="fallback-stream",
                run_id="fallback-run",
                turn_id="fallback-turn",
            )
        ]

        self.assertFalse(response.raw_iterated)
        self.assertTrue(response.projection_requested)
        self.assertEqual(items[1].text_delta, "projected")

    async def test_render_projections_rejects_bad_api_item(self):
        class Response:
            def __aiter__(self):
                async def gen():
                    yield object()

                return gen()

            def consumer_projections(self, **_kwargs):
                async def gen():
                    yield "bad"

                return gen()

        with self.assertRaisesRegex(
            StreamValidationError,
            "consumer projection stream item must be StreamConsumerProjection",
        ):
            [
                item
                async for item in model_cmds._stream_render_projections(
                    Response(),
                    stream_session_id="fallback-stream",
                    run_id="fallback-run",
                    turn_id="fallback-turn",
                )
            ]

    def test_stream_event_type_accepts_string_event_type(self) -> None:
        self.assertEqual(
            model_cmds._stream_event_type(Event(type="custom.event")),
            "custom.event",
        )

    async def test_plain_stdout_legacy_rejection_first_item(
        self,
    ):
        legacy_first_items = (
            "legacy",
            Token(token="legacy"),
            TokenDetail(token="legacy", id=1),
            ReasoningToken(token="legacy"),
            ToolCallToken(
                token="{}",
                call=ToolCall(
                    id="legacy-call",
                    name="legacy",
                    arguments={},
                ),
            ),
            Event(type=EventType.START),
        )

        for legacy_first_item in legacy_first_items:
            with self.subTest(
                legacy_item_type=type(legacy_first_item).__name__
            ):

                async def gen():
                    yield legacy_first_item
                    yield CanonicalStreamItem(
                        stream_session_id="stream",
                        run_id="run",
                        turn_id="turn",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    )

                with self.assertRaisesRegex(
                    StreamValidationError,
                    "unsupported CLI stream item",
                ):
                    [
                        projection
                        async for projection in (
                            model_cmds._plain_stdout_projections(gen())
                        )
                    ]

    async def test_token_generation_no_stats_uses_response_projection(self):
        async def gen():
            for item in _canonical_tool_call_answer_stream_items(
                '{"x":1}', "public"
            ):
                yield item

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
        )
        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()

        await model_cmds.token_generation(
            args=args,
            console=console,
            theme=MagicMock(),
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=MagicMock(),
            input_string="i",
            response=response,
            display_tokens=0,
            dtokens_pick=0,
            with_stats=False,
            tool_events_limit=2,
            refresh_per_second=2,
        )

        console.print.assert_called_once_with("public", end="")

    async def test_token_generation_no_stats_rejects_bad_response_projection(
        self,
    ):
        class Response:
            def consumer_projections(self, **_kwargs):
                async def gen():
                    for item in (
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=0,
                            kind=StreamItemKind.STREAM_STARTED,
                            channel=StreamChannel.CONTROL,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=1,
                            kind=StreamItemKind.STREAM_COMPLETED,
                            channel=StreamChannel.CONTROL,
                            usage={},
                            terminal_outcome=StreamTerminalOutcome.COMPLETED,
                        ),
                        CanonicalStreamItem(
                            stream_session_id="stream",
                            run_id="run",
                            turn_id="turn",
                            sequence=2,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            text_delta="late",
                        ),
                    ):
                        yield project_canonical_stream_item(item)

                return gen()

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()

        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        console.print.assert_not_called()

    async def test_plain_stdout_rejects_non_projection_response_item(self):
        class Response:
            def consumer_projections(self, **_kwargs):
                async def gen():
                    yield object()

                return gen()

        with self.assertRaisesRegex(
            StreamValidationError,
            "consumer projection stream item must be StreamConsumerProjection",
        ):
            [
                projection
                async for projection in model_cmds._plain_stdout_projections(
                    Response()
                )
            ]

    async def test_token_generation_no_stats_rejects_late_canonical_content(
        self,
    ):
        async def gen():
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=1,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yield CanonicalStreamItem(
                stream_session_id="s",
                run_id="r",
                turn_id="t",
                sequence=2,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="late",
            )

        args = Namespace(skip_display_reasoning_time=False)
        console = MagicMock()

        with self.assertRaises(StreamValidationError):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=gen(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        console.print.assert_not_called()

    async def test_token_generation_rejects_legacy_event_stream_item(self):
        async def gen():
            yield Event(type=EventType.TOOL_EXECUTE, payload={})

        args = Namespace(
            skip_display_reasoning_time=False,
        )
        console = MagicMock()
        with self.assertRaises(StreamValidationError):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=MagicMock(),
                input_string="i",
                response=gen(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=False,
                tool_events_limit=2,
                refresh_per_second=2,
            )
        console.print.assert_not_called()

    async def test_token_generation_raises_event_stream_error(self):
        args = Namespace(
            display_events=True,
            display_tools=False,
            record=False,
            skip_display_reasoning_time=False,
        )

        with self.assertRaisesRegex(ValueError, "event stream failed"):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(
                    event_manager=self._event_manager(
                        error=ValueError("event stream failed")
                    )
                ),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=self._empty_response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

    async def test_token_generation_event_failure_closes_response(self):
        args = Namespace(
            display_events=True,
            display_tools=False,
            record=False,
            skip_display_reasoning_time=False,
        )
        never = asyncio.Event()

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False
            close_count = 0
            iterating = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            async def aclose(self) -> None:
                if self.iterating:
                    raise AssertionError("response closed during iteration")
                self.close_count += 1
                if self.close_count > 1:
                    raise AssertionError("response closed more than once")

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.iterating = True
                try:
                    await never.wait()
                finally:
                    self.iterating = False
                return "answer"

        class EventManager:
            async def listen(self, *, stop_signal):
                _ = stop_signal
                raise ValueError("event stream failed")
                yield Event(type=EventType.START, payload={})

        response = Response()
        with self.assertRaisesRegex(ValueError, "event stream failed"):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=response,
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_events=True),
            )

        self.assertEqual(response.close_count, 1)

    async def test_token_generation_startup_cancel_cleans_supervised_tasks(
        self,
    ):
        args = Namespace(skip_display_reasoning_time=False)
        coordinator_container: dict[str, object | None] = {}
        response_closed = asyncio.Event()
        event_closed = asyncio.Event()
        never = asyncio.Event()

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    try:
                        await never.wait()
                        yield "answer"
                    finally:
                        response_closed.set()

                return gen()

        class EventManager:
            async def listen(self, *, stop_signal):
                _ = stop_signal
                try:
                    await never.wait()
                    if False:
                        yield Event(type=EventType.AGENT, payload={})
                finally:
                    event_closed.set()

        async def cancel_startup(delay: float) -> None:
            self.assertEqual(delay, 0)
            await asyncio.sleep(0)
            raise asyncio.CancelledError()

        with patch(
            "avalan.cli.commands.model.sleep",
            new=AsyncMock(side_effect=cancel_startup),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await model_cmds.token_generation(
                    args=args,
                    console=MagicMock(width=80),
                    theme=self._theme(),
                    logger=MagicMock(),
                    orchestrator=SimpleNamespace(event_manager=EventManager()),
                    event_stats=None,
                    lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                    input_string="i",
                    response=Response(),
                    display_tokens=0,
                    dtokens_pick=0,
                    with_stats=True,
                    tool_events_limit=2,
                    refresh_per_second=2,
                    coordinator_container=coordinator_container,
                    display_config=self._display_config(display_events=True),
                )

        self.assertIsNone(coordinator_container["coordinator"])
        self.assertTrue(response_closed.is_set())
        self.assertTrue(event_closed.is_set())

    async def test_token_generation_startup_cancel_consumes_done_failures(
        self,
    ):
        args = Namespace(skip_display_reasoning_time=False)
        loop = asyncio.get_running_loop()
        exception_contexts: list[dict[str, object]] = []
        old_handler = loop.get_exception_handler()

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    raise ValueError("response failed")
                    yield "answer"

                return gen()

        class EventManager:
            async def listen(self, *, stop_signal):
                _ = stop_signal
                raise RuntimeError("event failed")
                yield Event(type=EventType.AGENT, payload={})

        async def cancel_startup(delay: float) -> None:
            self.assertEqual(delay, 0)
            await asyncio.sleep(0)
            raise asyncio.CancelledError()

        loop.set_exception_handler(
            lambda _loop, context: exception_contexts.append(context)
        )
        try:
            with patch(
                "avalan.cli.commands.model.sleep",
                new=AsyncMock(side_effect=cancel_startup),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await model_cmds.token_generation(
                        args=args,
                        console=MagicMock(width=80),
                        theme=self._theme(),
                        logger=MagicMock(),
                        orchestrator=SimpleNamespace(
                            event_manager=EventManager()
                        ),
                        event_stats=None,
                        lm=SimpleNamespace(
                            model_id="m",
                            tokenizer_config=None,
                        ),
                        input_string="i",
                        response=Response(),
                        display_tokens=0,
                        dtokens_pick=0,
                        with_stats=True,
                        tool_events_limit=2,
                        refresh_per_second=2,
                        display_config=self._display_config(
                            display_events=True
                        ),
                    )
            collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(old_handler)

        self.assertEqual(exception_contexts, [])

    async def test_token_generation_failure_stops_cooperative_event_listener(
        self,
    ):
        args = Namespace(skip_display_reasoning_time=False)
        event_stopped = asyncio.Event()

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    raise ValueError("response failed")
                    yield "answer"

                return gen()

        class EventManager:
            async def listen(self, *, stop_signal):
                try:
                    await asyncio.Event().wait()
                    if False:
                        yield Event(type=EventType.AGENT, payload={})
                except asyncio.CancelledError:
                    while not stop_signal.is_set():
                        await asyncio.sleep(0)
                    event_stopped.set()
                    return

        with self.assertRaisesRegex(ValueError, "response failed"):
            await asyncio.wait_for(
                model_cmds.token_generation(
                    args=args,
                    console=MagicMock(width=80),
                    theme=self._theme(),
                    logger=MagicMock(),
                    orchestrator=SimpleNamespace(event_manager=EventManager()),
                    event_stats=None,
                    lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                    input_string="i",
                    response=Response(),
                    display_tokens=0,
                    dtokens_pick=0,
                    with_stats=True,
                    tool_events_limit=2,
                    refresh_per_second=2,
                    display_config=self._display_config(display_events=True),
                ),
                timeout=1,
            )

        self.assertTrue(event_stopped.is_set())

    async def test_token_generation_response_done_raises_event_cleanup_error(
        self,
    ):
        args = Namespace(skip_display_reasoning_time=False)

        class EventManager:
            async def listen(self, *, stop_signal):
                _ = stop_signal
                try:
                    await asyncio.Event().wait()
                    if False:
                        yield Event(type=EventType.AGENT, payload={})
                except asyncio.CancelledError:
                    raise RuntimeError("event cleanup failed")

        with self.assertRaisesRegex(RuntimeError, "event cleanup failed"):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=self._answer_response("answer"),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_events=True),
            )

    async def test_token_generation_groups_simultaneous_stream_failures(self):
        args = Namespace(skip_display_reasoning_time=False)

        class Response:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    raise ValueError("response failed")
                    yield "answer"

                return gen()

        class EventManager:
            async def listen(self, *, stop_signal):
                _ = stop_signal
                raise RuntimeError("event failed")
                yield Event(type=EventType.AGENT, payload={})

        with self.assertRaises(BaseExceptionGroup) as raised:
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="i",
                response=Response(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
                display_config=self._display_config(display_events=True),
            )

        failure_messages = {str(exc) for exc in raised.exception.exceptions}
        self.assertEqual(
            failure_messages,
            {"event failed", "response failed"},
        )

    async def test_token_generation_with_stats_uses_canonical_projection(
        self,
    ):
        items = [
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="plan",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=4,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=5,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in items:
                        yield item

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        captured: list[dict[str, object]] = []

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[tuple[None, str], ...]:
            _ = kw
            captured.append(
                {
                    "thinking_text_tokens": list(state.reasoning_text_tokens),
                    "answer_text_tokens": list(state.answer_text_tokens),
                    "total_tokens": state.total_tokens,
                    "ttft": state.ttft,
                }
            )
            return ((None, "frame"),)

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)
        live = MagicMock()
        lm = SimpleNamespace(
            model_id="m",
            tokenizer_config=None,
            input_token_count=MagicMock(return_value=1),
        )

        await model_cmds._token_stream(
            args=args,
            console=MagicMock(width=80),
            live=live,
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=lm,
            input_string="i",
            response=Resp(),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        self.assertEqual(
            captured,
            [
                {
                    "thinking_text_tokens": ["plan"],
                    "answer_text_tokens": ["answer"],
                    "total_tokens": 2,
                    "ttft": captured[-1]["ttft"],
                },
            ],
        )
        self.assertIsNotNone(captured[-1]["ttft"])

    async def test_token_generation_consumes_stream_while_render_is_slow(
        self,
    ) -> None:
        render_started = asyncio.Event()
        render_release = ThreadEvent()
        answer_consumed = asyncio.Event()
        consumed: list[str] = []

        async def token_gen() -> AsyncIterator[CanonicalStreamItem]:
            for item in _canonical_answer_stream_items("A", "B"):
                if item.kind is StreamItemKind.ANSWER_DELTA:
                    assert item.text_delta is not None
                    consumed.append(item.text_delta)
                yield item
                if item.text_delta == "A":
                    await asyncio.wait_for(render_started.wait(), timeout=1.0)
                if item.text_delta == "B":
                    answer_consumed.set()
                await asyncio.sleep(0)

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                return token_gen()

        def fake_token_frames(
            state: TokenRenderState, **_kwargs: object
        ) -> tuple[tuple[None, str], ...]:
            return ((None, "frame-" + "".join(state.answer_text_tokens)),)

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            display_events=True,
            display_tools=True,
            record=False,
        )
        live = MagicMock()
        live_cm = MagicMock()
        live_cm.__enter__.return_value = live
        live_cm.__exit__.return_value = False
        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)
        theme.events.return_value = None
        rendered: list[str] = []
        live_container: dict[str, object | None] = {}
        loop = asyncio.get_running_loop()

        def slow_render(*call_args: object) -> None:
            frame = cast(str, call_args[3])
            group = cast(Any, call_args[4])
            group_index = cast(int | None, call_args[5])
            rendered.append(frame)
            loop.call_soon_threadsafe(render_started.set)
            render_release.wait(2.0)
            if group is not None and group_index is not None:
                group.renderables[group_index] = frame

        with (
            patch.object(model_cmds, "Live", return_value=live_cm),
            patch.object(model_cmds, "_render_frame", side_effect=slow_render),
        ):
            task = asyncio.create_task(
                model_cmds.token_generation(
                    args=args,
                    console=MagicMock(width=80),
                    theme=theme,
                    logger=MagicMock(),
                    orchestrator=SimpleNamespace(
                        event_manager=EventManager(), input_token_count=1
                    ),
                    event_stats=None,
                    lm=SimpleNamespace(
                        model_id="m",
                        tokenizer_config=None,
                        input_token_count=MagicMock(return_value=1),
                    ),
                    input_string="prompt",
                    response=Resp(),
                    display_tokens=0,
                    dtokens_pick=0,
                    refresh_per_second=1000,
                    tool_events_limit=None,
                    with_stats=True,
                    live_container=live_container,
                )
            )
            await asyncio.wait_for(render_started.wait(), timeout=1.0)
            await asyncio.wait_for(answer_consumed.wait(), timeout=1.0)

            self.assertEqual(consumed, ["A", "B"])
            self.assertFalse(task.done())

            render_release.set()
            await task

        self.assertEqual(rendered[-1], "frame-AB")
        self.assertEqual(live_container.get("live"), None)

    async def test_token_stream_skips_tokenizer_config_without_display_tokens(
        self,
    ):
        items = [
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in items:
                        yield item

                return gen()

        class LazyTokenConfigLm:
            model_id = "m"

            @property
            def tokenizer_config(self) -> object:
                raise AssertionError("tokenizer config should be opt-in")

            def input_token_count(self, value: object) -> int:
                _ = value
                return 1

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        captured: list[tuple[object, object, object]] = []

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[tuple[None, str], ...]:
            _ = kw
            captured.append(
                (
                    state.added_tokens,
                    state.special_tokens,
                    state.display_tokens or None,
                )
            )
            return ((None, "frame"),)

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)

        await model_cmds._token_stream(
            args=args,
            console=MagicMock(width=80),
            live=MagicMock(),
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=LazyTokenConfigLm(),
            input_string="i",
            response=Resp(),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        self.assertEqual(captured, [(None, None, None)])

    async def test_token_stream_reuses_unchanged_token_snapshots(self):
        items = [
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.REASONING_DELTA,
                channel=StreamChannel.REASONING,
                text_delta="plan",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.REASONING_DONE,
                channel=StreamChannel.REASONING,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="A",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=4,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="B",
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=5,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            ),
            CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=6,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        ]

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in items:
                        yield item

                return gen()

        captured: list[TokenRenderState] = []

        class CapturingFrameBuilder:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def mark_dirty(self, state: TokenRenderState) -> None:
                captured.append(state)

            async def close(self) -> None:
                pass

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        with (
            patch.object(
                model_cmds,
                "_LatestTokenFrameBuilder",
                CapturingFrameBuilder,
            ),
            patch.object(model_cmds, "_projection_display_token") as display,
        ):
            await model_cmds._token_stream(
                args=args,
                console=MagicMock(width=80),
                live=MagicMock(),
                group=None,
                tokens_group_index=None,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(
                    model_id="m",
                    tokenizer_config=None,
                    input_token_count=lambda value: 1,
                ),
                input_string="i",
                response=Resp(),
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=2,
                stop_signal=None,
                tool_events_limit=2,
                with_stats=True,
            )

        self.assertEqual(len(captured), 3)
        self.assertEqual(captured[0].reasoning_text_tokens, ("plan",))
        self.assertEqual(captured[1].answer_text_tokens, ("A",))
        self.assertEqual(captured[2].answer_text_tokens, ("A", "B"))
        self.assertIs(
            captured[0].reasoning_text_tokens,
            captured[1].reasoning_text_tokens,
        )
        self.assertIs(
            captured[1].reasoning_text_tokens,
            captured[2].reasoning_text_tokens,
        )
        self.assertIsNot(
            captured[1].answer_text_tokens,
            captured[2].answer_text_tokens,
        )
        display.assert_not_called()

    async def test_token_stream_passes_tokenizer_config_for_display_tokens(
        self,
    ):
        stream_delta = _canonical_answer_delta("answer", token_id=7)
        tokenizer_tokens = {"answer": 7}
        tokenizer_special_tokens = {"<eos>": 2}

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in _canonical_answer_stream_items(stream_delta):
                        yield item

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        captured: list[tuple[object, object, object]] = []

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[tuple[None, str], ...]:
            _ = kw
            captured.append(
                (
                    state.added_tokens,
                    state.special_tokens,
                    list(state.display_tokens),
                )
            )
            return ((None, "frame"),)

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)
        lm = SimpleNamespace(
            model_id="m",
            tokenizer_config=SimpleNamespace(
                tokens=tokenizer_tokens,
                special_tokens=tokenizer_special_tokens,
            ),
            input_token_count=lambda value: 1,
        )

        await model_cmds._token_stream(
            args=args,
            console=MagicMock(width=80),
            live=MagicMock(),
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=lm,
            input_string="i",
            response=Resp(),
            display_tokens=1,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        self.assertEqual(
            captured,
            [
                (
                    tuple(tokenizer_tokens),
                    tuple(tokenizer_special_tokens),
                    [
                        model_cmds.TokenRenderDisplayToken(
                            sequence=1,
                            kind=StreamItemKind.ANSWER_DELTA,
                            channel=StreamChannel.ANSWER,
                            token="answer",
                            id=7,
                        )
                    ],
                )
            ],
        )

    async def test_token_stream_focus_predicate_uses_display_metadata(self):
        stream_delta = _canonical_answer_delta(
            "answer",
            token_id=1,
            probability=0.9,
            tokens=[
                {
                    "token": "alternate",
                    "token_id": 2,
                    "probability": 0.8,
                }
            ],
        )

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in _canonical_answer_stream_items(stream_delta):
                        yield item

                return gen()

        async def capture_state(
            display_probabilities: bool,
        ) -> TokenRenderState:
            args = Namespace(
                skip_display_reasoning_time=False,
                display_time_to_n_token=1,
                display_pause=0,
                start_thinking=False,
                display_probabilities=display_probabilities,
                display_probabilities_maximum=0.5,
                display_probabilities_sample_minimum=0.5,
                record=False,
            )
            captured: list[TokenRenderState] = []

            def fake_token_frames(
                state: TokenRenderState, **kw: object
            ) -> tuple[tuple[None, str], ...]:
                _ = kw
                captured.append(state)
                return ((None, "frame"),)

            theme = MagicMock()
            theme.token_frames = MagicMock(side_effect=fake_token_frames)
            await model_cmds._token_stream(
                args=args,
                console=MagicMock(width=80),
                live=MagicMock(),
                group=None,
                tokens_group_index=None,
                theme=theme,
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(
                    model_id="m",
                    tokenizer_config=None,
                    input_token_count=lambda value: 1,
                ),
                input_string="i",
                response=Resp(),
                display_tokens=1,
                dtokens_pick=1,
                refresh_per_second=2,
                stop_signal=None,
                tool_events_limit=2,
                with_stats=True,
            )
            return captured[-1]

        enabled_state = await capture_state(True)
        enabled_predicate = enabled_state.focus_on_token_when
        assert enabled_predicate is not None
        self.assertTrue(enabled_predicate(enabled_state.display_tokens[-1]))
        self.assertTrue(
            enabled_predicate(
                model_cmds.TokenRenderDisplayToken(
                    sequence=2,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    token="low",
                    probability=0.1,
                )
            )
        )

        disabled_state = await capture_state(False)
        disabled_predicate = disabled_state.focus_on_token_when
        assert disabled_predicate is not None
        self.assertFalse(disabled_predicate(enabled_state.display_tokens[-1]))

    async def test_token_stream_skips_non_answer_text_projection(self):
        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    yield CanonicalStreamItem(
                        stream_session_id="stream",
                        run_id="run",
                        turn_id="turn",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="stream",
                        run_id="run",
                        turn_id="turn",
                        sequence=1,
                        kind=StreamItemKind.STREAM_DIAGNOSTIC,
                        channel=StreamChannel.CONTROL,
                        text_delta="diagnostic",
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="stream",
                        run_id="run",
                        turn_id="turn",
                        sequence=2,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="answer",
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="stream",
                        run_id="run",
                        turn_id="turn",
                        sequence=3,
                        kind=StreamItemKind.ANSWER_DONE,
                        channel=StreamChannel.ANSWER,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="stream",
                        run_id="run",
                        turn_id="turn",
                        sequence=4,
                        kind=StreamItemKind.STREAM_COMPLETED,
                        channel=StreamChannel.CONTROL,
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    )

                return gen()

        captured: list[TokenRenderState] = []

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[tuple[None, str], ...]:
            _ = kw
            captured.append(state)
            return ((None, "frame"),)

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)
        await model_cmds._token_stream(
            args=Namespace(
                skip_display_reasoning_time=False,
                display_time_to_n_token=1,
                display_pause=0,
                start_thinking=False,
                display_probabilities=False,
                display_probabilities_maximum=0.0,
                display_probabilities_sample_minimum=0.0,
                record=False,
            ),
            console=MagicMock(width=80),
            live=MagicMock(),
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=SimpleNamespace(
                model_id="m",
                tokenizer_config=None,
                input_token_count=lambda value: 1,
            ),
            input_string="i",
            response=Resp(),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        self.assertEqual(captured[-1].answer_text_tokens, ("answer",))
        self.assertEqual(captured[-1].total_tokens, 1)

    async def test_token_stream_reuses_tokenizer_config_per_stream(self):
        first_delta = _canonical_answer_delta("one", token_id=7)
        second_delta = _canonical_answer_delta("two", token_id=8)

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in _canonical_answer_stream_items(
                        first_delta, second_delta
                    ):
                        yield item

                return gen()

        class CountingLm:
            model_id = "m"

            def __init__(self) -> None:
                self.config_reads = 0

            @property
            def tokenizer_config(self) -> object:
                self.config_reads += 1
                return SimpleNamespace(
                    tokens=[f"token-{self.config_reads}"],
                    special_tokens=[f"special-{self.config_reads}"],
                )

            def input_token_count(self, value: object) -> int:
                _ = value
                return 1

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        captured: list[tuple[object, object]] = []

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[tuple[None, str], ...]:
            _ = kw
            captured.append((state.added_tokens, state.special_tokens))
            return ((None, "frame"),)

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)
        lm = CountingLm()

        for _ in range(2):
            await model_cmds._token_stream(
                args=args,
                console=MagicMock(width=80),
                live=MagicMock(),
                group=None,
                tokens_group_index=None,
                theme=theme,
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="i",
                response=Resp(),
                display_tokens=1,
                dtokens_pick=0,
                refresh_per_second=2,
                stop_signal=None,
                tool_events_limit=2,
                with_stats=True,
            )

        self.assertEqual(lm.config_reads, 2)
        self.assertEqual(
            captured,
            [
                (("token-1",), ("special-1",)),
                (("token-2",), ("special-2",)),
            ],
        )

    async def test_token_stream_consumes_sync_theme_frames(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        yielded: list[str] = []

        def fake_frames(*_: object, **__: object):
            yielded.append("first")
            yielded.append("second")
            return ((None, "frame1"), (None, "frame2"))

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)

        await model_cmds._token_stream(
            args=args,
            console=MagicMock(width=80),
            live=MagicMock(),
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=SimpleNamespace(
                model_id="m",
                tokenizer_config=None,
                input_token_count=lambda value: 1,
            ),
            input_string="i",
            response=Resp(),
            display_tokens=1,
            dtokens_pick=1,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        self.assertEqual(yielded, ["first", "second"])

    async def test_token_stream_sets_stop_signal_on_pause_cancel(
        self,
    ):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        yielded: list[str] = []

        def fake_frames(*_: object, **__: object):
            yielded.append("first")
            yielded.append("second")
            return (
                (Token(id=1, token="A"), "frame1"),
                (None, "frame2"),
            )

        async def cancellable_sleep(delay: float) -> None:
            if delay == 0.01:
                raise asyncio.CancelledError()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=10,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)
        stop_signal = asyncio.Event()

        with patch(
            "avalan.cli.commands.model.sleep",
            new=AsyncMock(side_effect=cancellable_sleep),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await model_cmds._token_stream(
                    args=args,
                    console=MagicMock(width=80),
                    live=MagicMock(),
                    group=None,
                    tokens_group_index=None,
                    theme=theme,
                    logger=MagicMock(),
                    orchestrator=None,
                    event_stats=None,
                    lm=SimpleNamespace(
                        model_id="m",
                        tokenizer_config=None,
                        input_token_count=lambda value: 1,
                    ),
                    input_string="i",
                    response=Resp(),
                    display_tokens=1,
                    dtokens_pick=1,
                    refresh_per_second=2,
                    stop_signal=stop_signal,
                    tool_events_limit=2,
                    with_stats=True,
                )

        self.assertEqual(yielded, ["first", "second"])
        self.assertTrue(stop_signal.is_set())

    async def test_token_stream_accepts_empty_theme_frame_stream(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        def fake_frames(*_: object, **__: object) -> tuple[object, ...]:
            return ()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)
        live = MagicMock()

        await model_cmds._token_stream(
            args=args,
            console=MagicMock(width=80),
            live=live,
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=SimpleNamespace(
                model_id="m",
                tokenizer_config=None,
                input_token_count=lambda value: 1,
            ),
            input_string="i",
            response=Resp(),
            display_tokens=1,
            dtokens_pick=1,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        live.update.assert_not_called()

    async def test_token_stream_consumes_probability_theme_frames(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        yielded: list[str] = []

        def fake_frames(*_: object, **__: object):
            yielded.append("first")
            yielded.append("second")
            return (
                (Token(id=1, token="A"), "frame1"),
                (None, "frame2"),
            )

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=1,
            start_thinking=False,
            display_probabilities=True,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)

        with patch(
            "avalan.cli.commands.model.sleep",
            new_callable=AsyncMock,
        ) as sleep_patch:
            await model_cmds._token_stream(
                args=args,
                console=MagicMock(width=80),
                live=MagicMock(),
                group=None,
                tokens_group_index=None,
                theme=theme,
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(
                    model_id="m",
                    tokenizer_config=None,
                    input_token_count=lambda value: 1,
                ),
                input_string="i",
                response=Resp(),
                display_tokens=1,
                dtokens_pick=1,
                refresh_per_second=2,
                stop_signal=None,
                tool_events_limit=2,
                with_stats=True,
            )

        self.assertEqual(yielded, ["first", "second"])
        self.assertIn(call(0.001), sleep_patch.await_args_list)
        self.assertIn(call(0.1), sleep_patch.await_args_list)
        self.assertIn(call(0), sleep_patch.await_args_list)

    async def test_token_generation_timing_pause(self):
        frame_token = model_cmds.TokenRenderDisplayToken(
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            token="a",
            id=0,
        )
        stream_token = _canonical_answer_delta("a", token_id=0)

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def g():
                    for item in _canonical_answer_stream_items(
                        stream_token, stream_token
                    ):
                        yield item

                return g()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=1,
            start_thinking=False,
            display_probabilities=True,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()

        def fake_token_frames(*_: object, **__: object):
            return ((frame_token, "frame1"), (None, "frame2"))

        theme = self._legacy_theme(fake_token_frames)

        live = MagicMock()
        console = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        lm = SimpleNamespace(model_id="m", tokenizer_config=None)

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="i",
                response=Resp(),
                display_tokens=1,
                dtokens_pick=1,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        theme.token_frames.assert_called()
        renderables = self._live_update_renderables(live)
        self.assertNotIn("frame1", renderables)
        self.assertIn("frame2", renderables)

    async def test_token_generation_ttnt_metric(self):
        stream_delta = _canonical_answer_delta("a", token_id=0)

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def g():
                    for item in _canonical_answer_stream_items(stream_delta):
                        yield item

                return g()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()

        def fake_token_frames(*_: object, **__: object):
            return ((None, "frame"),)

        theme = self._legacy_theme(fake_token_frames)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        lm = SimpleNamespace(model_id="m", tokenizer_config=None)

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="i",
                response=Resp(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        theme.token_frames.assert_called_once()
        self.assertIn("frame", self._live_update_renderables(live))

    async def test_token_generation_stats_do_not_count_control_items(self):
        control = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        answer = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="a",
        )
        answer_done = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=2,
            kind=StreamItemKind.ANSWER_DONE,
            channel=StreamChannel.ANSWER,
        )
        terminal = CanonicalStreamItem(
            stream_session_id="s",
            run_id="r",
            turn_id="t",
            sequence=3,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def g():
                    yield control
                    yield answer
                    yield answer_done
                    yield terminal

                return g()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()
        captured: list[dict[str, float | int | None]] = []

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[tuple[None, str], ...]:
            _ = kw
            captured.append(
                {
                    "total_tokens": state.total_tokens,
                    "ttft": state.ttft,
                    "ttnt": state.ttnt,
                }
            )
            return ((None, "frame"),)

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)
        live = MagicMock()
        lm = SimpleNamespace(
            model_id="m",
            tokenizer_config=None,
            input_token_count=lambda s: 1,
        )

        await model_cmds._token_stream(
            args=args,
            console=console,
            live=live,
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=logger,
            orchestrator=None,
            event_stats=None,
            lm=lm,
            input_string="i",
            response=Resp(),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        self.assertEqual([c["total_tokens"] for c in captured], [1])
        self.assertIsNotNone(captured[0]["ttft"])
        self.assertIsNotNone(captured[0]["ttnt"])

    async def test_token_generation_tool_call_tokens_do_not_count_as_output(
        self,
    ):
        class Resp:
            input_token_count = 0
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def g():
                    for item in _canonical_tool_call_answer_stream_items(
                        "TOOL"
                    ):
                        yield item

                return g()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()
        captured: list[dict[str, Any]] = []

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[tuple[None, str], ...]:
            _ = kw
            captured.append(
                {
                    "tool_text_tokens": list(state.tool_text_tokens),
                    "tokens": (
                        list(state.display_tokens)
                        if state.display_tokens
                        else None
                    ),
                    "input_token_count": state.input_token_count,
                    "total_tokens": state.total_tokens,
                    "ttft": state.ttft,
                    "ttnt": state.ttnt,
                    "tool_token_count": state.tool_token_count,
                }
            )
            return ((None, "frame"),)

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)
        live = MagicMock()
        lm = SimpleNamespace(
            model_id="m",
            tokenizer_config=None,
            input_token_count=MagicMock(side_effect=[12, 2]),
        )
        orchestrator = SimpleNamespace(input_token_count=None)

        await model_cmds._token_stream(
            args=args,
            console=console,
            live=live,
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=logger,
            orchestrator=orchestrator,
            event_stats=None,
            lm=lm,
            input_string="i",
            response=Resp(),
            display_tokens=1,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        self.assertEqual(
            captured,
            [
                {
                    "tool_text_tokens": ["TOOL"],
                    "tokens": None,
                    "input_token_count": 12,
                    "total_tokens": 0,
                    "ttft": None,
                    "ttnt": None,
                    "tool_token_count": 2,
                }
            ],
        )
        lm.input_token_count.assert_has_calls([call("i"), call("TOOL")])

    async def test_token_generation_empty_tool_call_token_counts_zero(self):
        class Resp:
            input_token_count = 0
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def g():
                    for item in _canonical_tool_call_answer_stream_items(""):
                        yield item

                return g()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()
        captured: list[dict[str, object]] = []

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[tuple[None, str], ...]:
            _ = kw
            captured.append(
                {
                    "tool_text_tokens": list(state.tool_text_tokens),
                    "tool_token_count": state.tool_token_count,
                }
            )
            return ((None, "frame"),)

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_token_frames)
        live = MagicMock()
        lm = SimpleNamespace(
            model_id="m",
            tokenizer_config=None,
            input_token_count=MagicMock(return_value=12),
        )

        await model_cmds._token_stream(
            args=args,
            console=console,
            live=live,
            group=None,
            tokens_group_index=None,
            theme=theme,
            logger=logger,
            orchestrator=None,
            event_stats=None,
            lm=lm,
            input_string="i",
            response=Resp(),
            display_tokens=1,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=2,
            with_stats=True,
        )

        self.assertEqual(
            captured,
            [{"tool_text_tokens": [""], "tool_token_count": 0}],
        )
        lm.input_token_count.assert_called_once_with("i")

    async def test_token_generation_rejects_model_response_event_item(self):
        async def inner_gen():
            for item in _canonical_answer_stream_items("x"):
                yield item

        inner_response = model_cmds.TextGenerationResponse(
            lambda: inner_gen(),
            inputs={"input_ids": [[1, 2, 3]]},
            logger=getLogger(),
            use_async_generator=True,
        )
        event = Event(
            type=EventType.TOOL_MODEL_RESPONSE,
            payload={"response": inner_response},
        )

        class Resp:
            input_token_count = 10
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def g():
                    yield event

                return g()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()

        theme = MagicMock()
        live = MagicMock()
        lm = SimpleNamespace(
            model_id="m",
            tokenizer_config=None,
            input_token_count=MagicMock(return_value=1),
        )

        with self.assertRaises(StreamValidationError):
            await model_cmds._token_stream(
                args=args,
                console=console,
                live=live,
                group=None,
                tokens_group_index=None,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="i",
                response=Resp(),
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=2,
                stop_signal=None,
                tool_events_limit=2,
                with_stats=True,
            )

        theme.token_frames.assert_not_called()

    async def test_token_generation_coordinator_container_without_orchestrator(
        self,
    ):
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )

        coordinator_container: dict[str, object | None] = {}
        observed: list[object | None] = []

        theme = self._coordinator_observer_theme(
            observed,
            coordinator_container,
        )

        await model_cmds.token_generation(
            args=args,
            console=MagicMock(width=80),
            theme=theme,
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=SimpleNamespace(model_id="m", tokenizer_config=None),
            input_string="prompt",
            response=self._answer_response("answer"),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=1,
            tool_events_limit=0,
            coordinator_container=coordinator_container,
        )

        self.assertTrue(observed)
        self.assertIsNotNone(observed[0])
        self.assertIsNone(coordinator_container.get("coordinator"))

    async def test_token_generation_coordinator_container_with_orchestrator(
        self,
    ):
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=False,
            record=False,
        )

        coordinator_container: dict[str, object | None] = {}
        observed: list[object | None] = []

        theme = self._coordinator_observer_theme(
            observed,
            coordinator_container,
        )

        await model_cmds.token_generation(
            args=args,
            console=MagicMock(width=80),
            theme=theme,
            logger=MagicMock(),
            orchestrator=SimpleNamespace(
                event_manager=self._event_manager(
                    Event(type=EventType.START, payload={})
                )
            ),
            event_stats=None,
            lm=SimpleNamespace(model_id="m", tokenizer_config=None),
            input_string="prompt",
            response=self._answer_response("answer"),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=1,
            tool_events_limit=0,
            coordinator_container=coordinator_container,
        )

        self.assertTrue(observed)
        self.assertIsNotNone(observed[0])
        self.assertIsNone(coordinator_container.get("coordinator"))

    async def test_token_generation_coordinator_container_cleared_on_interrupt(
        self,
    ):
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )
        coordinator_container: dict[str, object | None] = {}
        response_closed = asyncio.Event()

        class InterruptResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            async def aclose(self) -> None:
                response_closed.set()

            def __aiter__(self):
                async def gen():
                    raise KeyboardInterrupt()
                    yield ""

                return gen()

        with self.assertRaises(KeyboardInterrupt):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(),
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=MagicMock(),
                input_string="prompt",
                response=InterruptResponse(),
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
                coordinator_container=coordinator_container,
            )

        self.assertIsNone(coordinator_container.get("coordinator"))
        self.assertTrue(response_closed.is_set())

    async def test_token_generation_no_event_validation_closes_once(
        self,
    ) -> None:
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )

        class InvalidResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False
            close_count = 0

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            async def aclose(self) -> None:
                self.close_count += 1
                if self.close_count > 1:
                    raise AssertionError("response closed more than once")

            def __aiter__(self):
                return self

            async def __anext__(self):
                items = (
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    ),
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=1,
                        kind=StreamItemKind.STREAM_COMPLETED,
                        channel=StreamChannel.CONTROL,
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=2,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="late",
                    ),
                )
                index = getattr(self, "index", 0)
                if index >= len(items):
                    raise StopAsyncIteration
                self.index = index + 1
                return items[index]

        response = InvalidResponse()
        with self.assertRaises(StreamValidationError):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=response,
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
            )

        self.assertEqual(response.close_count, 1)

    async def test_token_generation_projection_adapter_owns_response_close(
        self,
    ) -> None:
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )

        class ProjectionAdapter:
            def __init__(self, response: "ProjectionResponse") -> None:
                self._response = response
                self._index = 0

            def __aiter__(self) -> "ProjectionAdapter":
                return self

            async def __anext__(self) -> StreamConsumerProjection:
                items = (
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    ),
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=1,
                        kind=StreamItemKind.STREAM_COMPLETED,
                        channel=StreamChannel.CONTROL,
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=2,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="late",
                    ),
                )
                if self._index >= len(items):
                    raise StopAsyncIteration
                item = items[self._index]
                self._index += 1
                return project_canonical_stream_item(item)

            async def aclose(self) -> None:
                self._response.adapter_close_count += 1
                await self._response.aclose()
                raise RuntimeError("adapter close failed")

        class ProjectionResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def __init__(self) -> None:
                self.adapter_close_count = 0
                self.close_count = 0

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                raise AssertionError(
                    "raw response iterator should not be used"
                )

            def consumer_projections(self, **_kwargs: object):
                return ProjectionAdapter(self)

            async def aclose(self) -> None:
                self.close_count += 1
                if self.close_count > 1:
                    raise AssertionError("response closed more than once")

        response = ProjectionResponse()
        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=response,
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
            )

        self.assertEqual(response.adapter_close_count, 1)
        self.assertEqual(response.close_count, 1)

    async def test_token_generation_invalid_projection_setup_closes_response(
        self,
    ) -> None:
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )

        class RaisingProjectionResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def __init__(self) -> None:
                self.close_count = 0

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                raise AssertionError(
                    "raw response iterator should not be used"
                )

            def consumer_projections(self, **_kwargs: object):
                raise RuntimeError("projection setup failed")

            async def aclose(self) -> None:
                self.close_count += 1

        class NonAsyncProjectionResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def __init__(self) -> None:
                self.close_count = 0

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                raise AssertionError(
                    "raw response iterator should not be used"
                )

            def consumer_projections(self, **_kwargs: object) -> object:
                return object()

            async def aclose(self) -> None:
                self.close_count += 1

        class BrokenAdapter:
            def __aiter__(self) -> "BrokenAdapter":
                raise RuntimeError("projection iterator failed")

            async def __anext__(self) -> StreamConsumerProjection:
                raise StopAsyncIteration

        class BrokenAiterProjectionResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def __init__(self) -> None:
                self.close_count = 0

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                raise AssertionError(
                    "raw response iterator should not be used"
                )

            def consumer_projections(self, **_kwargs: object) -> BrokenAdapter:
                return BrokenAdapter()

            async def aclose(self) -> None:
                self.close_count += 1

        raising_response = RaisingProjectionResponse()
        with self.assertRaisesRegex(RuntimeError, "projection setup failed"):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=raising_response,
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
            )
        self.assertEqual(raising_response.close_count, 1)

        non_async_response = NonAsyncProjectionResponse()
        with self.assertRaises(AssertionError):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=non_async_response,
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
            )
        self.assertEqual(non_async_response.close_count, 1)

        broken_aiter_response = BrokenAiterProjectionResponse()
        with self.assertRaisesRegex(
            RuntimeError,
            "projection iterator failed",
        ):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=broken_aiter_response,
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
            )
        self.assertEqual(broken_aiter_response.close_count, 1)

    async def test_token_generation_success_surfaces_response_close_error(
        self,
    ) -> None:
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )

        class ClosingResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=1,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="done",
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=2,
                        kind=StreamItemKind.ANSWER_DONE,
                        channel=StreamChannel.ANSWER,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=3,
                        kind=StreamItemKind.STREAM_COMPLETED,
                        channel=StreamChannel.CONTROL,
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    )

                return gen()

            async def aclose(self) -> None:
                raise RuntimeError("response close failed")

        with self.assertRaisesRegex(RuntimeError, "response close failed"):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=ClosingResponse(),
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
                display_config=self._display_config(interactive=False),
            )

    async def test_token_generation_direct_close_error_preserves_validation(
        self,
    ) -> None:
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )

        class InvalidResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def __init__(self) -> None:
                self.close_count = 0

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
                async def gen() -> AsyncIterator[CanonicalStreamItem]:
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=1,
                        kind=StreamItemKind.STREAM_COMPLETED,
                        channel=StreamChannel.CONTROL,
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=2,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="late",
                    )

                return gen()

            async def aclose(self) -> None:
                self.close_count += 1
                raise RuntimeError("response close failed")

        response = InvalidResponse()
        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=response,
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
            )

        self.assertEqual(response.close_count, 1)

    async def test_token_generation_validation_error_closes_response(
        self,
    ) -> None:
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=False,
            record=False,
        )
        coordinator_container: dict[str, object | None] = {}
        event_closed = asyncio.Event()
        observed_stop_signal: asyncio.Event | None = None

        class InvalidResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False
            close_count = 0

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            async def aclose(self) -> None:
                self.close_count += 1
                if self.close_count > 1:
                    raise AssertionError("response closed more than once")

            def __aiter__(self):
                return self

            async def __anext__(self):
                items = (
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    ),
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=1,
                        kind=StreamItemKind.STREAM_COMPLETED,
                        channel=StreamChannel.CONTROL,
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    ),
                    CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=2,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="late",
                    ),
                )
                index = getattr(self, "index", 0)
                if index >= len(items):
                    raise StopAsyncIteration
                self.index = index + 1
                return items[index]

        class EventManager:
            async def listen(self, *, stop_signal):
                nonlocal observed_stop_signal
                observed_stop_signal = stop_signal
                try:
                    while not stop_signal.is_set():
                        await asyncio.sleep(0)
                    if False:
                        yield Event(type=EventType.AGENT, payload={})
                finally:
                    event_closed.set()

        response = InvalidResponse()
        with self.assertRaises(StreamValidationError):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=SimpleNamespace(event_manager=EventManager()),
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=response,
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
                coordinator_container=coordinator_container,
                display_config=self._display_config(display_events=True),
            )

        self.assertEqual(response.close_count, 1)
        self.assertTrue(event_closed.is_set())
        self.assertIsNotNone(observed_stop_signal)
        assert observed_stop_signal is not None
        self.assertTrue(observed_stop_signal.is_set())
        self.assertIsNone(coordinator_container.get("coordinator"))

    async def test_token_generation_cancelled_error_closes_response(
        self,
    ) -> None:
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )
        coordinator_container: dict[str, object | None] = {}

        class CancelledResponse:
            input_token_count = 1
            can_think = False
            is_thinking = False
            close_count = 0

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            async def aclose(self) -> None:
                self.close_count += 1
                if self.close_count > 1:
                    raise AssertionError("response closed more than once")

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError()

        response = CancelledResponse()
        with self.assertRaises(asyncio.CancelledError):
            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="prompt",
                response=response,
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=1,
                tool_events_limit=0,
                coordinator_container=coordinator_container,
            )

        self.assertEqual(response.close_count, 1)
        self.assertIsNone(coordinator_container.get("coordinator"))

    async def test_token_generation_with_stats(self):
        frame_token = model_cmds.TokenRenderDisplayToken(
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            token="a",
            id=0,
            probability=0.4,
        )
        stream_token = _canonical_answer_delta(
            "a", token_id=0, probability=0.4
        )

        class Resp:
            def __init__(self, deltas):
                self._deltas = deltas
                self.input_token_count = 1
                self.can_think = False
                self.is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in _canonical_answer_stream_items(*self._deltas):
                        yield item

                return gen()

        response = Resp([stream_token])

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=True,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()

        captured: dict[str, list] = {}

        def fake_token_frames(
            state: TokenRenderState, **kw: object
        ) -> tuple[
            tuple[model_cmds.TokenRenderDisplayToken | None, str], ...
        ]:
            _ = kw
            captured["text_tokens"] = list(state.reasoning_text_tokens) + list(
                state.answer_text_tokens
            )
            captured["input_token_count"] = state.input_token_count
            return ((frame_token, "frame1"), (None, "frame2"))

        theme = self._legacy_theme(fake_token_frames)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=MagicMock()
        )

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="i",
                response=response,
                display_tokens=1,
                dtokens_pick=1,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        theme.token_frames.assert_called_once()
        self.assertEqual(captured["text_tokens"], ["a"])
        self.assertEqual(captured["input_token_count"], 1)
        renderables = self._live_update_renderables(live)
        self.assertNotIn("frame1", renderables)
        self.assertIn("frame2", renderables)
        lm.input_token_count.assert_not_called()

    async def test_token_generation_input_count_fallback(self):
        stream_delta = _canonical_answer_delta("a", token_id=0)

        class Resp:
            def __init__(self, deltas, count):
                self._deltas = deltas
                self.input_token_count = count
                self.can_think = False
                self.is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in _canonical_answer_stream_items(*self._deltas):
                        yield item

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            display_events=False,
            display_tools=False,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()

        def gen_frame(*a: object, **k: object):
            _ = a, k
            return ((None, "frame"),)

        theme = self._legacy_theme(gen_frame)

        lm = MagicMock()
        lm.model_id = "m"
        lm.tokenizer_config = None
        lm.input_token_count = MagicMock(return_value=33)

        # Response provides count, orchestrator should be ignored
        response = Resp([stream_delta], count=5)
        orchestrator = SimpleNamespace(input_token_count=7, event_manager=None)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=orchestrator,
                event_stats=None,
                lm=lm,
                input_string="i",
                response=response,
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        state = theme.token_frames.call_args.args[0]
        self.assertEqual(state.input_token_count, 5)
        lm.input_token_count.assert_not_called()

        # Response has zero count, fall back to orchestrator
        response_zero = Resp([stream_delta], count=0)
        theme.token_frames.reset_mock()

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=orchestrator,
                event_stats=None,
                lm=lm,
                input_string="i",
                response=response_zero,
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        state = theme.token_frames.call_args.args[0]
        self.assertEqual(state.input_token_count, 7)

        # Response zero and orchestrator none -> use lm.input_token_count
        theme.token_frames.reset_mock()
        lm.input_token_count.reset_mock()

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="text",
                response=response_zero,
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        state = theme.token_frames.call_args.args[0]
        self.assertEqual(state.input_token_count, 33)
        lm.input_token_count.assert_called_once_with("text")

    async def test_token_generation_rejects_tool_event_stream_item(self):
        events = [
            Event(type=EventType.TOOL_EXECUTE),
            Token(id=1, token="a"),
        ]

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in events:
                        yield item

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
            display_answer_height_expand=False,
            display_answer_height=12,
        )
        console = MagicMock()
        console.width = 80
        theme = MagicMock()
        theme.get_spinner.return_value = "dots"
        theme._n = lambda s, p, n: s if n == 1 else p
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        lm = SimpleNamespace(
            model_id="m",
            tokenizer_config=None,
            input_token_count=MagicMock(return_value=1),
        )

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            with self.assertRaises(StreamValidationError):
                await model_cmds.token_generation(
                    args=args,
                    console=console,
                    theme=theme,
                    logger=MagicMock(),
                    orchestrator=None,
                    event_stats=None,
                    lm=lm,
                    input_string="text",
                    response=Resp(),
                    display_tokens=0,
                    dtokens_pick=0,
                    with_stats=True,
                    tool_events_limit=2,
                    refresh_per_second=2,
                )

        theme.token_frames.assert_not_called()

    async def test_token_generation_display_options_combinations(self):
        combos = [
            (False, False, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ]

        for display_events, display_tools, expected_event_started in combos:
            args = Namespace(
                skip_display_reasoning_time=False,
                display_time_to_n_token=None,
                display_pause=0,
                start_thinking=False,
                display_probabilities=False,
                display_probabilities_maximum=0.0,
                display_probabilities_sample_minimum=0.0,
                display_events=display_events,
                display_tools=display_tools,
                record=False,
            )

            event_started = asyncio.Event()
            orchestrator = SimpleNamespace(
                event_manager=self._event_manager(
                    Event(type=EventType.START, payload={}),
                    started=event_started,
                )
            )
            lm = SimpleNamespace(model_id="m", tokenizer_config=None)

            await model_cmds.token_generation(
                args=args,
                console=MagicMock(width=80),
                theme=self._theme(),
                logger=MagicMock(),
                orchestrator=orchestrator,
                event_stats=None,
                lm=lm,
                input_string="i",
                response=self._answer_response("answer"),
                display_tokens=0,
                dtokens_pick=0,
                tool_events_limit=None,
                refresh_per_second=2,
                with_stats=True,
            )

            self.assertEqual(event_started.is_set(), expected_event_started)


class CliModelRunTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        _disable_mlx_model_import(self)

    async def test_returns_when_no_input(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=False,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = MagicMock()
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(
                model_cmds, "ModelManager", return_value=manager
            ) as mm_patch,
            patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ) as gms_patch,
            patch.object(model_cmds, "get_input", return_value=None),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_GENERATION,
        )
        lm.assert_not_called()
        tg_patch.assert_not_called()
        hub.can_access.assert_called_once_with("id")
        hub.model.assert_called_once_with("id")
        theme.model.assert_called_once_with(
            "hub_model", can_access=True, summary=True
        )
        console.print.assert_called_once()

    async def test_run_local_ds4_model_skips_hub_summary(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="ai://local/../pyds4/.local/ds4/ds4flash.gguf",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=False,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            backend="ds4",
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        hub = MagicMock()
        logger = MagicMock()

        engine_uri = SimpleNamespace(
            model_id="../pyds4/.local/ds4/ds4flash.gguf",
            is_local=True,
            params={},
        )
        lm = MagicMock()
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ),
            patch.object(model_cmds, "get_input", return_value=None),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        hub.can_access.assert_not_called()
        hub.model.assert_not_called()
        theme.model.assert_not_called()
        console.print.assert_not_called()
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_GENERATION,
        )

    async def test_run_local_model(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=2,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=False,
            do_sample=True,
            enable_gradient_calculation=True,
            min_p=0.1,
            repetition_penalty=1.1,
            temperature=0.5,
            top_k=5,
            top_p=0.9,
            use_cache=False,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="resp")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.TEXT_GENERATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_GENERATION,
        )

        lm.assert_awaited_once()
        call_kwargs = lm.await_args.kwargs
        self.assertEqual(call_kwargs["system_prompt"], None)
        self.assertEqual(call_kwargs["manual_sampling"], 0)
        self.assertEqual(call_kwargs["pick"], 0)
        self.assertFalse(call_kwargs["skip_special_tokens"])
        settings = call_kwargs["settings"]
        self.assertIsInstance(settings, model_cmds.GenerationSettings)
        self.assertEqual(settings.max_new_tokens, args.max_new_tokens)

        tg_patch.assert_awaited_once()
        tg_kwargs = tg_patch.await_args.kwargs
        self.assertEqual(tg_kwargs["input_string"], "hi")
        self.assertEqual(tg_kwargs["response"], "resp")

    async def test_run_local_model_non_interactive_uses_stderr_diagnostics(
        self,
    ):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=2,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=False,
            do_sample=True,
            enable_gradient_calculation=True,
            min_p=0.1,
            repetition_penalty=1.1,
            temperature=0.5,
            top_k=5,
            top_p=0.9,
            use_cache=False,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=True,
            display_tools=True,
            display_tools_events=2,
            record=True,
        )
        console = MagicMock()
        console.is_terminal = False
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="resp")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ),
            patch.object(
                model_cmds,
                "get_input",
                wraps=model_cmds.get_input,
            ) as get_input_patch,
            patch("avalan.cli.has_input", return_value=True),
            patch("avalan.cli.stdin", StringIO("hi\n")),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        hub.can_access.assert_not_called()
        hub.model.assert_not_called()
        theme.model.assert_not_called()
        console.print.assert_not_called()
        get_input_patch.assert_called_once()
        self.assertTrue(get_input_patch.call_args.kwargs["is_quiet"])
        self.assertTrue(get_input_patch.call_args.kwargs["echo_stdin"])
        lm.assert_awaited_once()
        tg_patch.assert_awaited_once()
        tg_kwargs = tg_patch.await_args.kwargs
        display_config = tg_kwargs["display_config"]
        self.assertTrue(display_config.answer_stdout_only)
        self.assertEqual(display_config.diagnostic_channel, "stderr")
        self.assertTrue(display_config.show_tools)
        self.assertTrue(display_config.show_events)
        self.assertFalse(tg_kwargs["with_stats"])
        self.assertEqual(tg_kwargs["input_string"], "hi")

    async def test_run_remote_model(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=False)
        lm = AsyncMock(return_value="resp")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(
                model_cmds, "ModelManager", return_value=manager
            ) as mm_patch,
            patch(
                "avalan.cli.commands.model.ModelManager."
                "get_operation_from_arguments",
                new=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ) as gms_patch,
            patch.object(model_cmds, "get_input", return_value="hi"),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_GENERATION,
        )
        lm.assert_awaited_once()
        tg_patch.assert_awaited_once()

    async def test_run_remote_model_quiet_uses_answer_clean_stream(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=True,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=True,
            display_tools=True,
            display_tools_events=0,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=False)
        lm = AsyncMock(return_value="resp")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch(
                "avalan.cli.commands.model.ModelManager."
                "get_operation_from_arguments",
                new=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ),
            patch.object(model_cmds, "get_input", return_value="hi"),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        theme.model.assert_not_called()
        console.print.assert_not_called()
        lm.assert_awaited_once()
        tg_patch.assert_awaited_once()
        self.assertFalse(tg_patch.await_args.kwargs["with_stats"])
        self.assertEqual(tg_patch.await_args.kwargs["tool_events_limit"], 0)

    async def test_run_remote_model_with_input_file_and_no_prompt(self):
        with NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(b"%PDF-1.7")
            tmp.flush()

            args = Namespace(
                skip_display_reasoning_time=False,
                model="id",
                device="cpu",
                max_new_tokens=1,
                quiet=False,
                skip_hub_access_check=False,
                no_repl=True,
                do_sample=False,
                enable_gradient_calculation=False,
                min_p=None,
                repetition_penalty=1.0,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                use_cache=True,
                stop_on_keyword=None,
                system=None,
                skip_special_tokens=False,
                display_tokens=0,
                tool_events=2,
                display_events=False,
                display_tools=False,
                display_tools_events=2,
                input_file=[tmp.name],
            )
            console = MagicMock()
            theme = MagicMock()
            theme._ = lambda s: s
            theme.icons = {"user_input": ">"}
            theme.model.return_value = "panel"
            hub = MagicMock()
            hub.can_access.return_value = True
            hub.model.return_value = "hub_model"
            logger = MagicMock()

            engine_uri = SimpleNamespace(model_id="id", is_local=False)
            lm = AsyncMock(return_value="resp")
            lm.config = MagicMock()
            lm.config.__repr__ = lambda self=None: "cfg"

            load_cm = MagicMock()
            load_cm.__enter__.return_value = lm
            load_cm.__exit__.return_value = False

            manager = RealModelManager(hub, logger)
            manager.parse_uri = MagicMock(return_value=engine_uri)
            manager.load = MagicMock(return_value=load_cm)

            with (
                patch.object(model_cmds, "ModelManager", return_value=manager),
                patch(
                    "avalan.cli.commands.model.ModelManager."
                    "get_operation_from_arguments",
                    new=RealModelManager.get_operation_from_arguments,
                ),
                patch.object(
                    model_cmds,
                    "get_model_settings",
                    return_value={
                        "engine_uri": engine_uri,
                        "modality": Modality.TEXT_GENERATION,
                    },
                ),
                patch.object(model_cmds, "get_input", return_value=None),
                patch.object(
                    model_cmds, "token_generation", new_callable=AsyncMock
                ) as tg_patch,
            ):
                await model_cmds.model_run(
                    args, console, theme, hub, 5, logger
                )

        lm.assert_awaited_once()
        request_input = lm.await_args.args[0]
        self.assertIsInstance(request_input, Message)
        self.assertEqual(request_input.role, MessageRole.USER)
        assert isinstance(request_input.content, list)
        self.assertEqual(
            request_input.content,
            [
                MessageContentFile(
                    type="file",
                    file={
                        "file_data": b64encode(b"%PDF-1.7").decode("ascii"),
                        "filename": Path(tmp.name).name,
                        "mime_type": "application/pdf",
                    },
                )
            ],
        )
        tg_patch.assert_awaited_once()
        self.assertEqual(tg_patch.await_args.kwargs["input_string"], "")

    async def test_run_rejects_input_file_for_non_text_modality(self):
        with NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(b"%PDF-1.7")
            tmp.flush()

            args = Namespace(
                model="id",
                device="cpu",
                max_new_tokens=1,
                quiet=True,
                input_file=[tmp.name],
                skip_hub_access_check=False,
            )
            console = MagicMock()
            theme = MagicMock()
            theme._ = lambda s: s
            theme.icons = {}
            hub = MagicMock()
            logger = MagicMock()
            engine_uri = SimpleNamespace(model_id="id", is_local=False)
            load_cm = MagicMock()
            load_cm.__enter__.return_value = MagicMock(config=MagicMock())
            load_cm.__exit__.return_value = False

            with (
                patch.object(model_cmds, "ModelManager") as manager_cls,
                patch.object(
                    model_cmds,
                    "get_model_settings",
                    return_value={
                        "engine_uri": engine_uri,
                        "modality": Modality.EMBEDDING,
                    },
                ),
            ):
                manager = manager_cls.return_value.__enter__.return_value
                manager.parse_uri.return_value = engine_uri
                manager.load.return_value = load_cm
                manager_cls.get_operation_from_arguments.return_value = (
                    SimpleNamespace(
                        modality=Modality.EMBEDDING, requires_input=False
                    )
                )

                with self.assertRaisesRegex(
                    AssertionError,
                    "--input-file is only supported for text generation",
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

    async def test_model_run_use_cache_cli(self):
        for use_cache in (True, False):
            args = Namespace(
                skip_display_reasoning_time=False,
                model="id",
                device="cpu",
                max_new_tokens=1,
                quiet=False,
                skip_hub_access_check=False,
                no_repl=True,
                do_sample=False,
                enable_gradient_calculation=False,
                min_p=None,
                repetition_penalty=1.0,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                use_cache=use_cache,
                cache_strategy=None,
                stop_on_keyword=None,
                system=None,
                skip_special_tokens=False,
                display_tokens=0,
                tool_events=2,
                display_events=False,
                display_tools=False,
                display_tools_events=2,
            )
            console = MagicMock()
            theme = MagicMock()
            theme._ = lambda s: s
            theme.icons = {"user_input": ">"}
            theme.model.return_value = "panel"
            hub = MagicMock()
            hub.can_access.return_value = True
            hub.model.return_value = "hub_model"
            logger = MagicMock()

            engine_uri = SimpleNamespace(model_id="id", is_local=True)
            lm = AsyncMock(return_value="resp")
            lm.config = MagicMock()
            lm.config.__repr__ = lambda self=None: "cfg"

            load_cm = MagicMock()
            load_cm.__enter__.return_value = lm
            load_cm.__exit__.return_value = False

            manager = RealModelManager(hub, logger)
            manager.parse_uri = MagicMock(return_value=engine_uri)
            manager.load = MagicMock(return_value=load_cm)

            with (
                patch.object(model_cmds, "ModelManager", return_value=manager),
                patch.object(
                    model_cmds.ModelManager,
                    "get_operation_from_arguments",
                    side_effect=RealModelManager.get_operation_from_arguments,
                ),
                patch.object(
                    model_cmds,
                    "get_model_settings",
                    return_value={
                        "engine_uri": engine_uri,
                        "modality": Modality.TEXT_GENERATION,
                    },
                ),
                patch.object(model_cmds, "get_input", return_value="hi"),
                patch.object(
                    model_cmds, "token_generation", new_callable=AsyncMock
                ),
            ):
                await model_cmds.model_run(
                    args, console, theme, hub, 5, logger
                )

            settings = lm.await_args.kwargs["settings"]
            self.assertEqual(settings.use_cache, use_cache)

    async def test_model_run_cache_strategy_cli(self):
        strategies = [None] + [s.value for s in GenerationCacheStrategy]
        for strat in strategies:
            args = Namespace(
                skip_display_reasoning_time=False,
                model="id",
                device="cpu",
                max_new_tokens=1,
                quiet=False,
                skip_hub_access_check=False,
                no_repl=True,
                do_sample=False,
                enable_gradient_calculation=False,
                min_p=None,
                repetition_penalty=1.0,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                use_cache=True,
                cache_strategy=strat,
                stop_on_keyword=None,
                system=None,
                skip_special_tokens=False,
                display_tokens=0,
                tool_events=2,
                display_events=False,
                display_tools=False,
                display_tools_events=2,
            )
            console = MagicMock()
            theme = MagicMock()
            theme._ = lambda s: s
            theme.icons = {"user_input": ">"}
            theme.model.return_value = "panel"
            hub = MagicMock()
            hub.can_access.return_value = True
            hub.model.return_value = "hub_model"
            logger = MagicMock()

            engine_uri = SimpleNamespace(model_id="id", is_local=True)
            lm = AsyncMock(return_value="resp")
            lm.config = MagicMock()
            lm.config.__repr__ = lambda self=None: "cfg"

            load_cm = MagicMock()
            load_cm.__enter__.return_value = lm
            load_cm.__exit__.return_value = False

            manager = RealModelManager(hub, logger)
            manager.parse_uri = MagicMock(return_value=engine_uri)
            manager.load = MagicMock(return_value=load_cm)

            with (
                patch.object(model_cmds, "ModelManager", return_value=manager),
                patch.object(
                    model_cmds.ModelManager,
                    "get_operation_from_arguments",
                    side_effect=RealModelManager.get_operation_from_arguments,
                ),
                patch.object(
                    model_cmds,
                    "get_model_settings",
                    return_value={
                        "engine_uri": engine_uri,
                        "modality": Modality.TEXT_GENERATION,
                    },
                ),
                patch.object(model_cmds, "get_input", return_value="hi"),
                patch.object(
                    model_cmds, "token_generation", new_callable=AsyncMock
                ),
            ):
                await model_cmds.model_run(
                    args, console, theme, hub, 5, logger
                )

            settings = lm.await_args.kwargs["settings"]
            if strat is None:
                self.assertIsNone(settings.cache_strategy)
            else:
                self.assertEqual(settings.cache_strategy, strat)

    async def test_model_run_reasoning_effort_cli(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            cache_strategy=None,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            start_thinking=False,
            chat_disable_thinking=False,
            no_reasoning=False,
            reasoning_tag=None,
            reasoning_effort="xhigh",
            reasoning_max_new_tokens=None,
            reasoning_stop_on_max_new_tokens=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = MagicMock()
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)
        captured: dict[str, GenerationSettings] = {}

        async def manager_call(self, model_task):
            operation = model_task.operation
            captured["settings"] = operation.generation_settings
            return TextGenerationResponse(
                lambda: "resp",
                logger=getLogger(),
                use_async_generator=False,
                generation_settings=operation.generation_settings,
                settings=operation.generation_settings,
            )

        with (
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(RealModelManager, "__call__", manager_call),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ),
            patch.object(model_cmds, "get_input", return_value="hi"),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        settings = captured["settings"]
        self.assertEqual(
            settings.reasoning, ReasoningSettings(effort=ReasoningEffort.XHIGH)
        )

    async def test_model_run_sets_output_hidden_states(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            output_hidden_states=True,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        hub = MagicMock()
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = AsyncMock(
            config=MagicMock(__repr__=lambda self=None: "cfg")
        )
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(
                model_cmds,
                "ModelManager",
                return_value=manager,
            ) as mm,
            patch(
                "avalan.cli.commands.model.ModelManager.get_operation_from_arguments",
                new=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                    "output_hidden_states": True,
                },
            ) as gms,
            patch.object(model_cmds, "get_input", return_value="hi"),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        mm.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms.assert_called_once_with(args, hub, logger, engine_uri)
        self.assertTrue(manager.load.call_args.kwargs["output_hidden_states"])

    async def test_model_run_chat_disable_thinking(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            chat_disable_thinking=True,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        hub = MagicMock()
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        settings = TransformerEngineSettings(
            auto_load_model=False, auto_load_tokenizer=False
        )
        lm = TextGenerationModel("id", settings)
        lm._model = MagicMock()
        lm._tokenizer = MagicMock()
        lm._tokenizer.eos_token_id = 99
        tok_mock = MagicMock(return_value={"input_ids": [[1]]})
        lm._tokenize_input = tok_mock
        lm._string_output = MagicMock()

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(
                model_cmds, "ModelManager", return_value=manager
            ) as mm,
            patch(
                "avalan.cli.commands.model.ModelManager.get_operation_from_arguments",
                new=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ) as gms,
            patch.object(model_cmds, "get_input", return_value="hi"),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        mm.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms.assert_called_once_with(args, hub, logger, engine_uri)
        tok_kwargs = tok_mock.call_args.kwargs
        self.assertFalse(
            tok_kwargs["chat_template_settings"]["enable_thinking"]
        )

    async def test_run_audio_text_to_speech(self):
        base_args = dict(
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="out.wav",
            audio_sampling_rate=16_000,
        )

        for ref_path, ref_text in (
            (None, None),
            ("ref.wav", None),
            (None, "hello"),
            ("ref.wav", "hello"),
        ):
            with self.subTest(
                reference_path=ref_path, reference_text=ref_text
            ):
                args = Namespace(
                    skip_display_reasoning_time=False,
                    **base_args,
                    audio_reference_path=ref_path,
                    audio_reference_text=ref_text,
                )
                console = MagicMock()
                theme = MagicMock()
                theme._ = lambda s: s
                theme.icons = {"user_input": ">"}
                theme.model.return_value = "panel"
                hub = MagicMock()
                hub.can_access.return_value = True
                hub.model.return_value = "hub_model"
                logger = MagicMock()

                engine_uri = SimpleNamespace(model_id="id", is_local=True)
                lm = MagicMock()
                lm.config = MagicMock()
                lm.config.__repr__ = lambda self=None: "cfg"

                load_cm = MagicMock()
                load_cm.__enter__.return_value = lm
                load_cm.__exit__.return_value = False

                manager = AsyncMock()
                manager._logger = logger
                manager.__enter__.return_value = manager
                manager.__exit__.return_value = False
                manager.parse_uri = MagicMock(return_value=engine_uri)
                manager.load = MagicMock(return_value=load_cm)
                manager.return_value = "gen.wav"

                with patch.object(
                    model_cmds, "ModelManager", return_value=manager
                ) as mm_patch:
                    with patch.object(
                        model_cmds.ModelManager,
                        "get_operation_from_arguments",
                        side_effect=RealModelManager.get_operation_from_arguments,
                    ):
                        with (
                            patch.object(
                                model_cmds,
                                "get_model_settings",
                                return_value={
                                    "engine_uri": engine_uri,
                                    "modality": Modality.AUDIO_TEXT_TO_SPEECH,
                                },
                            ) as gms_patch,
                            patch.object(
                                model_cmds, "get_input", return_value="hi"
                            ),
                            patch.object(
                                model_cmds,
                                "token_generation",
                                new_callable=AsyncMock,
                            ) as tg_patch,
                        ):
                            await model_cmds.model_run(
                                args, console, theme, hub, 5, logger
                            )

                mm_patch.assert_called_once_with(hub, logger)
                manager.parse_uri.assert_called_once_with("id")
                gms_patch.assert_called_once_with(
                    args, hub, logger, engine_uri
                )
                manager.load.assert_called_once_with(
                    engine_uri=engine_uri,
                    modality=Modality.AUDIO_TEXT_TO_SPEECH,
                )
                manager.assert_awaited_once()
                task = manager.await_args_list[0].args[0]
                operation = task.operation
                audio_params = operation.parameters["audio"]
                self.assertIs(task.model, lm)
                self.assertEqual(operation.input, "hi")
                self.assertEqual(audio_params.path, "out.wav")
                self.assertEqual(audio_params.reference_path, ref_path)
                self.assertEqual(audio_params.reference_text, ref_text)
                self.assertEqual(audio_params.sampling_rate, 16_000)
                lm.assert_not_called()
                tg_patch.assert_not_called()
                self.assertEqual(
                    console.print.call_args.args[0],
                    "Audio generated in gen.wav",
                )

    async def test_run_audio_generation(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=2,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="out.wav",
            audio_sampling_rate=44_100,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = MagicMock()
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)
        manager.return_value = "gen.wav"

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.AUDIO_GENERATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.AUDIO_GENERATION,
        )
        manager.assert_awaited_once()
        task = manager.await_args_list[0].args[0]
        operation = task.operation
        audio_params = operation.parameters["audio"]
        self.assertIs(task.model, lm)
        self.assertEqual(operation.input, "hi")
        self.assertEqual(audio_params.path, "out.wav")
        self.assertEqual(audio_params.sampling_rate, 44_100)
        lm.assert_not_called()
        tg_patch.assert_not_called()
        self.assertEqual(
            console.print.call_args.args[0],
            "Audio generated in gen.wav",
        )

    async def test_run_audio_speech_recognition(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="in.wav",
            audio_sampling_rate=16_000,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = MagicMock()
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)
        manager.return_value = "transcript"

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.AUDIO_SPEECH_RECOGNITION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.AUDIO_SPEECH_RECOGNITION,
        )
        manager.assert_awaited_once()
        task = manager.await_args_list[0].args[0]
        operation = task.operation
        audio_params = operation.parameters["audio"]
        self.assertIs(task.model, lm)
        self.assertEqual(audio_params.path, "in.wav")
        self.assertEqual(audio_params.sampling_rate, 16_000)
        lm.assert_not_called()
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "transcript")

    async def test_run_audio_classification(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="audio.wav",
            audio_sampling_rate=16_000,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        theme.display_audio_labels.return_value = "table"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = MagicMock()
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)
        manager.return_value = {"ok": 1.0}

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.AUDIO_CLASSIFICATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.AUDIO_CLASSIFICATION,
        )
        manager.assert_awaited_once()
        task = manager.await_args_list[0].args[0]
        operation = task.operation
        self.assertIs(task.model, lm)
        self.assertEqual(operation.parameters["audio"].path, "audio.wav")
        self.assertEqual(operation.parameters["audio"].sampling_rate, 16_000)
        theme.display_audio_labels.assert_called_once_with(
            manager.return_value
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args_list[-1].args[0], "table")

    async def test_run_text_question_answering(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system="sys",
            developer="dev",
            text_context="ctx",
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="answer")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.TEXT_QUESTION_ANSWERING,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_QUESTION_ANSWERING,
        )
        lm.assert_awaited_once_with(
            "hi",
            context="ctx",
            system_prompt="sys",
            developer_prompt="dev",
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "answer")

    async def test_run_text_token_classification(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system="sys",
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        theme.display_token_labels.return_value = "table"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value={"t": "L"})
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.TEXT_TOKEN_CLASSIFICATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_TOKEN_CLASSIFICATION,
        )
        lm.assert_awaited_once_with(
            "hi",
            labeled_only=False,
            system_prompt="sys",
            developer_prompt=None,
        )
        tg_patch.assert_not_called()
        theme.display_token_labels.assert_called_once_with([lm.return_value])
        self.assertEqual(console.print.call_args.args[0], "table")

    async def test_run_text_token_classification_labeled_only(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system="sys",
            text_labeled_only=True,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        theme.display_token_labels.return_value = "table"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value={"t": "L"})
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.TEXT_TOKEN_CLASSIFICATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_TOKEN_CLASSIFICATION,
        )
        lm.assert_awaited_once_with(
            "hi",
            labeled_only=True,
            system_prompt="sys",
            developer_prompt=None,
        )
        tg_patch.assert_not_called()
        theme.display_token_labels.assert_called_once_with([lm.return_value])
        self.assertEqual(console.print.call_args.args[0], "table")

    async def test_run_text_sequence_classification(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="lbl")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.TEXT_SEQUENCE_CLASSIFICATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_SEQUENCE_CLASSIFICATION,
        )
        lm.assert_awaited_once_with("hi")
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "lbl")

    async def test_run_text_sequence_to_sequence(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=["stop"],
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="summary")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"
        lm.tokenizer = MagicMock()

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.TEXT_SEQUENCE_TO_SEQUENCE,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_SEQUENCE_TO_SEQUENCE,
        )
        lm.assert_awaited_once()
        kw = lm.await_args.kwargs
        self.assertIsInstance(kw["settings"], model_cmds.GenerationSettings)
        self.assertEqual(kw["settings"].max_new_tokens, args.max_new_tokens)
        self.assertIsNotNone(kw["stopping_criterias"])
        self.assertIsInstance(
            kw["stopping_criterias"][0], model_cmds.KeywordStoppingCriteria
        )
        self.assertEqual(kw["stopping_criterias"][0]._keywords, ["stop"])
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "summary")

    async def test_run_text_translation(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=["stop"],
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            text_from_lang="en_XX",
            text_to_lang="fr_XX",
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="summary")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"
        lm.tokenizer = MagicMock()

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.TEXT_TRANSLATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.TEXT_TRANSLATION,
        )
        lm.assert_awaited_once()
        kw = lm.await_args.kwargs
        self.assertIsInstance(kw["settings"], model_cmds.GenerationSettings)
        self.assertEqual(kw["settings"].max_new_tokens, args.max_new_tokens)
        self.assertEqual(kw["source_language"], args.text_from_lang)
        self.assertEqual(kw["destination_language"], args.text_to_lang)
        self.assertEqual(kw["skip_special_tokens"], args.skip_special_tokens)
        self.assertIsNotNone(kw["stopping_criterias"])
        self.assertIsInstance(
            kw["stopping_criterias"][0], model_cmds.KeywordStoppingCriteria
        )
        self.assertEqual(kw["stopping_criterias"][0]._keywords, ["stop"])
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "summary")

    async def test_run_vision_object_detection(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="img.png",
            vision_threshold=0.5,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        theme.display_image_entities.return_value = "table"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(
            return_value=[
                ImageEntity(label="lbl", score=0.9, box=[0, 1, 2, 3])
            ]
        )
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_OBJECT_DETECTION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_OBJECT_DETECTION,
        )
        lm.assert_awaited_once_with("img.png", threshold=0.5)
        theme.display_image_entities.assert_called_once_with(
            lm.return_value, sort=True
        )
        tg_patch.assert_not_called()
        self.assertEqual(
            console.print.call_args.args[0],
            theme.display_image_entities.return_value,
        )

    async def test_run_vision_semantic_segmentation(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="img.png",
            vision_threshold=0.5,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        theme.display_image_labels.return_value = "table"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value=["lbl1", "lbl2"])
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_SEMANTIC_SEGMENTATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_SEMANTIC_SEGMENTATION,
        )
        lm.assert_awaited_once_with("img.png")
        theme.display_image_labels.assert_called_once_with(lm.return_value)
        tg_patch.assert_not_called()
        self.assertEqual(
            console.print.call_args.args[0],
            theme.display_image_labels.return_value,
        )

    async def test_run_vision_image_classification(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="img.png",
            vision_threshold=0.5,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        theme.display_image_entity.return_value = "table"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value=ImageEntity(label="cat"))
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_IMAGE_CLASSIFICATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_IMAGE_CLASSIFICATION,
        )
        lm.assert_awaited_once_with("img.png")
        theme.display_image_entity.assert_called_once_with(lm.return_value)
        tg_patch.assert_not_called()
        self.assertEqual(
            console.print.call_args.args[0],
            theme.display_image_entity.return_value,
        )

    async def test_run_vision_image_to_text(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="img.png",
            vision_threshold=0.5,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="caption")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_IMAGE_TO_TEXT,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_IMAGE_TO_TEXT,
        )
        lm.assert_awaited_once()
        self.assertEqual(lm.await_args.args, ("img.png",))
        kw = lm.await_args.kwargs
        self.assertIsInstance(kw["settings"], model_cmds.GenerationSettings)
        self.assertEqual(kw["settings"].max_new_tokens, args.max_new_tokens)
        self.assertEqual(kw["skip_special_tokens"], False)
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "caption")

    async def test_run_vision_encoder_decoder(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="img.png",
            vision_threshold=0.5,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="caption")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_ENCODER_DECODER,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "has_input", return_value=False),
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_ENCODER_DECODER,
        )
        lm.assert_awaited_once_with(
            "img.png",
            prompt=None,
            skip_special_tokens=False,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "caption")

    async def test_run_vision_encoder_decoder_reads_optional_piped_stdin(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="img.png",
            vision_threshold=0.5,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="caption")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_ENCODER_DECODER,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "has_input", return_value=True),
                    patch.object(
                        model_cmds, "get_input", return_value="hi"
                    ) as get_input_patch,
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_ENCODER_DECODER,
        )
        get_input_patch.assert_called_once()
        lm.assert_awaited_once_with(
            "img.png",
            prompt="hi",
            skip_special_tokens=False,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "caption")

    async def test_run_vision_image_text_to_text(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system="sys",
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="img.png",
            vision_threshold=0.5,
            vision_width=42,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="txt")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_IMAGE_TEXT_TO_TEXT,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_IMAGE_TEXT_TO_TEXT,
        )
        lm.assert_awaited_once_with(
            "img.png",
            "hi",
            system_prompt="sys",
            developer_prompt=None,
            settings=ANY,
            width=42,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "txt")

    async def test_run_vision_text_to_image(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="out.png",
            vision_color_model="RGB",
            vision_image_format="PNG",
            vision_high_noise_frac=0.9,
            vision_steps=50,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="out.png")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_TEXT_TO_IMAGE,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_TEXT_TO_IMAGE,
        )
        lm.assert_awaited_once_with(
            "hi",
            "out.png",
            color_model="RGB",
            high_noise_frac=0.9,
            image_format="PNG",
            n_steps=50,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "out.png")

    async def test_run_vision_text_to_animation(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="out.gif",
            vision_steps=4,
            vision_timestep_spacing="trailing",
            vision_beta_schedule="linear",
            vision_guidance_scale=1.0,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="out.gif")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_TEXT_TO_ANIMATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_TEXT_TO_ANIMATION,
        )
        lm.assert_awaited_once_with(
            "hi",
            "out.gif",
            beta_schedule="linear",
            guidance_scale=1.0,
            steps=4,
            timestep_spacing="trailing",
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "out.gif")

    async def test_run_vision_text_to_animation_custom_options(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="anim.gif",
            vision_steps=8,
            vision_timestep_spacing="leading",
            vision_beta_schedule="scaled_linear",
            vision_guidance_scale=2.5,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="anim.gif")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_TEXT_TO_ANIMATION,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_TEXT_TO_ANIMATION,
        )
        lm.assert_awaited_once_with(
            "hi",
            "anim.gif",
            beta_schedule="scaled_linear",
            guidance_scale=2.5,
            steps=8,
            timestep_spacing="leading",
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "anim.gif")

    async def test_run_text_to_video(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="out.mp4",
            vision_reference_path="ref.png",
            vision_negative_prompt=None,
            vision_height=None,
            vision_downscale=2 / 3,
            vision_frames=96,
            vision_denoise_strength=0.4,
            vision_inference_steps=10,
            vision_decode_timestep=0.05,
            vision_noise_scale=0.025,
            vision_fps=24,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="out.mp4")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_TEXT_TO_VIDEO,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_TEXT_TO_VIDEO,
        )
        lm.assert_awaited_once_with(
            "hi",
            path="out.mp4",
            reference_path="ref.png",
            negative_prompt="",
            downscale=2 / 3,
            frames=96,
            denoise_strength=0.4,
            inference_steps=10,
            decode_timestep=0.05,
            noise_scale=0.025,
            fps=24,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "out.mp4")

    async def test_run_text_to_video_custom_options(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="custom.mp4",
            vision_reference_path="ref.png",
            vision_negative_prompt="neg",
            vision_height=256,
            vision_downscale=0.5,
            vision_frames=50,
            vision_denoise_strength=0.2,
            vision_inference_steps=5,
            vision_decode_timestep=0.1,
            vision_noise_scale=0.01,
            vision_fps=12,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="custom.mp4")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_TEXT_TO_VIDEO,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_TEXT_TO_VIDEO,
        )
        lm.assert_awaited_once_with(
            "hi",
            path="custom.mp4",
            reference_path="ref.png",
            negative_prompt="neg",
            height=256,
            downscale=0.5,
            frames=50,
            denoise_strength=0.2,
            inference_steps=5,
            decode_timestep=0.1,
            noise_scale=0.01,
            fps=12,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "custom.mp4")

    async def test_run_text_to_video_width_only(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="wide.mp4",
            vision_reference_path="ref.png",
            vision_negative_prompt=None,
            vision_height=None,
            vision_width=320,
            vision_downscale=2 / 3,
            vision_frames=96,
            vision_denoise_strength=0.4,
            vision_inference_steps=10,
            vision_decode_timestep=0.05,
            vision_noise_scale=0.025,
            vision_fps=24,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="wide.mp4")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_TEXT_TO_VIDEO,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_TEXT_TO_VIDEO,
        )
        lm.assert_awaited_once_with(
            "hi",
            path="wide.mp4",
            reference_path="ref.png",
            negative_prompt="",
            width=320,
            downscale=2 / 3,
            frames=96,
            denoise_strength=0.4,
            inference_steps=10,
            decode_timestep=0.05,
            noise_scale=0.025,
            fps=24,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "wide.mp4")

    async def test_run_text_to_video_steps_only(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            path="steps.mp4",
            vision_reference_path="ref.png",
            vision_negative_prompt=None,
            vision_height=None,
            vision_downscale=2 / 3,
            vision_frames=96,
            vision_denoise_strength=0.4,
            vision_steps=7,
            vision_inference_steps=10,
            vision_decode_timestep=0.05,
            vision_noise_scale=0.025,
            vision_fps=24,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value="steps.mp4")
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(model_task):
            return await RealModelManager.__call__(manager, model_task)

        manager.side_effect = call_side_effect

        with patch.object(
            model_cmds, "ModelManager", return_value=manager
        ) as mm_patch:
            with patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ):
                with (
                    patch.object(
                        model_cmds,
                        "get_model_settings",
                        return_value={
                            "engine_uri": engine_uri,
                            "modality": Modality.VISION_TEXT_TO_VIDEO,
                        },
                    ) as gms_patch,
                    patch.object(model_cmds, "get_input", return_value="hi"),
                    patch.object(
                        model_cmds, "token_generation", new_callable=AsyncMock
                    ) as tg_patch,
                ):
                    await model_cmds.model_run(
                        args, console, theme, hub, 5, logger
                    )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.VISION_TEXT_TO_VIDEO,
        )
        lm.assert_awaited_once_with(
            "hi",
            path="steps.mp4",
            reference_path="ref.png",
            negative_prompt="",
            downscale=2 / 3,
            frames=96,
            denoise_strength=0.4,
            steps=7,
            inference_steps=10,
            decode_timestep=0.05,
            noise_scale=0.025,
            fps=24,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "steps.mp4")

    async def test_run_invalid_modality_raises(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = MagicMock()
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = AsyncMock(side_effect=NotImplementedError())
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(
                model_cmds, "ModelManager", return_value=manager
            ) as mm_patch,
            patch(
                "avalan.cli.commands.model.ModelManager.get_operation_from_arguments",
                new=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.EMBEDDING,
                },
            ) as gms_patch,
            patch.object(model_cmds, "get_input", return_value="hi"),
        ):
            with self.assertRaises(NotImplementedError):
                await model_cmds.model_run(
                    args, console, theme, hub, 5, logger
                )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(args, hub, logger, engine_uri)
        manager.load.assert_called_once_with(
            engine_uri=engine_uri,
            modality=Modality.EMBEDDING,
        )
        manager.assert_awaited_once()
        lm.assert_not_called()

    async def test_run_unknown_modality(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=True,
            skip_hub_access_check=False,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=0,
            display_events=False,
            display_tools=False,
            display_tools_events=0,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = MagicMock()
        load_cm.__exit__.return_value = False

        manager = AsyncMock()
        manager._logger = logger
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(
                model_cmds, "ModelManager", return_value=manager
            ) as mm_patch,
            patch(
                "avalan.cli.commands.model.ModelManager.get_operation_from_arguments",
                new=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={"engine_uri": engine_uri, "modality": "bad"},
            ),
        ):
            with self.assertRaises(NotImplementedError):
                await model_cmds.model_run(
                    args, console, theme, hub, 5, logger
                )

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        manager.load.assert_called_once_with(
            engine_uri=engine_uri, modality="bad"
        )


class CliModelSearchTestCase(IsolatedAsyncioTestCase):
    async def test_model_search(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            filter=["f"],
            search=["q"],
            library=["lib"],
            author="a",
            gated=True,
            open=False,
            language=["en"],
            name=["n"],
            task=["t"],
            tag=["tag"],
            limit=2,
        )

        console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm

        theme = MagicMock()
        theme._ = lambda s: s
        theme.get_spinner.return_value = "sp"
        theme.model.side_effect = lambda m, **kw: (
            f"{m.id}-{kw.get('can_access')}"
        )

        model1 = SimpleNamespace(id="m1")
        model2 = SimpleNamespace(id="m2")
        hub = MagicMock()
        hub.models.return_value = [model1, model2]
        hub.can_access.side_effect = lambda mid: mid == "m1"

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        updates: list[tuple] = []
        live.update.side_effect = lambda g: updates.append(g)

        groups: list[tuple] = []

        def fake_group(*items):
            groups.append(items)
            return items

        async def to_thread_stub(fn, *a, **kw):
            return fn()

        with (
            patch.object(model_cmds, "Live", return_value=live),
            patch.object(model_cmds, "Group", side_effect=fake_group),
            patch.object(model_cmds, "to_thread", side_effect=to_thread_stub),
        ):
            await model_cmds.model_search(args, console, theme, hub, 5)

        console.status.assert_called_once_with(
            "Loading models...",
            spinner=theme.get_spinner.return_value,
            refresh_per_second=5,
        )
        hub.models.assert_called_once_with(
            filter=["f"],
            search=["q"],
            library=["lib"],
            author="a",
            gated=True,
            language=["en"],
            name=["n"],
            task=["t"],
            tags=["tag"],
            limit=2,
        )
        hub.can_access.assert_has_calls(
            [call("m1"), call("m2")], any_order=True
        )
        # Initial render without access info
        self.assertIn(("m1-None", "m2-None"), groups)
        # Final update includes access results
        self.assertIn(("m1-True", "m2-False"), updates)
        self.assertEqual(live.update.call_count, 2)

    async def test_model_search_open(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            filter=None,
            search=None,
            library=None,
            author=None,
            gated=False,
            open=True,
            language=None,
            name=None,
            task=None,
            tag=None,
            limit=1,
        )

        console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm

        theme = MagicMock()
        theme._ = lambda s: s
        theme.get_spinner.return_value = "sp"
        theme.model.side_effect = lambda m, **kw: m.id

        model = SimpleNamespace(id="m")
        hub = MagicMock()
        hub.models.return_value = [model]
        hub.can_access.return_value = True

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        with (
            patch.object(model_cmds, "Live", return_value=live),
            patch.object(model_cmds, "Group", side_effect=lambda *i: i),
            patch.object(
                model_cmds, "to_thread", side_effect=lambda f, *a, **k: f()
            ),
        ):
            await model_cmds.model_search(args, console, theme, hub, 5)

        hub.models.assert_called_once_with(
            filter=None,
            search=None,
            library=None,
            author=None,
            gated=False,
            language=None,
            name=None,
            task=None,
            tags=None,
            limit=1,
        )


class CliModelInternalTestCase(IsolatedAsyncioTestCase):
    async def test_latest_token_frame_builder_ignores_spurious_dirty(self):
        console = MagicMock()
        console.width = 80
        theme = MagicMock()
        frame_renderer = MagicMock()
        builder = model_cmds._LatestTokenFrameBuilder(
            Namespace(record=False),
            console,
            theme,
            MagicMock(),
            frame_renderer,
            refresh_per_second=1000,
            display_pause=0,
            frame_minimum_pause_ms=0,
            tool_events_limit=None,
            height=12,
            limit_answer_height=False,
            start_thinking=False,
        )

        builder._dirty.set()
        await asyncio.sleep(0)
        await builder.close()

        theme.token_frames.assert_not_called()
        frame_renderer.mark_dirty.assert_not_called()

    async def test_latest_token_frame_builder_skips_empty_frames(self):
        console = MagicMock()
        console.width = 80
        theme = MagicMock()
        theme.token_frames.return_value = ()
        frame_renderer = MagicMock()
        builder = model_cmds._LatestTokenFrameBuilder(
            Namespace(record=False),
            console,
            theme,
            MagicMock(),
            frame_renderer,
            refresh_per_second=1000,
            display_pause=0,
            frame_minimum_pause_ms=0,
            tool_events_limit=None,
            height=12,
            limit_answer_height=False,
            start_thinking=False,
        )

        await builder._build_and_render(
            TokenRenderState(model_id="m"), version=0
        )
        await builder.close()

        theme.token_frames.assert_called_once()
        frame_renderer.mark_dirty.assert_not_called()

    def test_canonical_event_payload_rejects_invalid_payloads(self):
        self.assertIsNone(
            model_cmds._canonical_event_payload(Event(type=EventType.START))
        )
        self.assertIsNone(
            model_cmds._canonical_event_payload(
                Event(
                    type=EventType.START,
                    payload={
                        "stream_session_id": "stream-1",
                        "run_id": "run-1",
                        "turn_id": "turn-1",
                        "channel": "control",
                    },
                )
            )
        )
        self.assertIsNone(
            model_cmds._canonical_event_payload(
                Event(
                    type=EventType.START,
                    payload={
                        "stream_session_id": "stream-1",
                        "run_id": "run-1",
                        "turn_id": "turn-1",
                        "kind": "stream.started",
                        "channel": 1,
                    },
                )
            )
        )

    async def test_event_stream_updates_and_stops(self):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        events = MagicMock(name="events")
        tools = MagicMock(name="tools")
        group = SimpleNamespace(
            renderables=[events, tools, MagicMock(name="tokens")]
        )
        theme = MagicMock()
        theme.events.side_effect = [None, "panel"]
        live = MagicMock()
        console = MagicMock()
        stop_signal = asyncio.Event()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=True,
            record=False,
        )
        task = asyncio.create_task(
            model_cmds._event_stream(
                args,
                console,
                live,
                group,
                0,
                1,
                orchestrator,
                theme,
                stop_signal=stop_signal,
            )
        )
        await orchestrator.event_manager.trigger(Event(type=EventType.START))
        await orchestrator.event_manager.trigger(Event(type=EventType.END))
        await asyncio.sleep(0)
        stop_signal.set()
        await task

        self.assertEqual(group.renderables[0], "panel")
        self.assertEqual(theme.events.call_count, 2)
        live.refresh.assert_called()

    async def test_event_stream_returns_when_no_event_manager_or_options(self):
        orchestrator = SimpleNamespace(event_manager=None)
        live = MagicMock()
        console = MagicMock()
        group = SimpleNamespace(renderables=[MagicMock(), MagicMock()])
        theme = MagicMock()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=True,
            record=False,
        )
        await model_cmds._event_stream(
            args,
            console,
            live,
            group,
            0,
            1,
            orchestrator,
            theme,
            stop_signal=asyncio.Event(),
        )

        theme.events.assert_not_called()
        live.refresh.assert_not_called()

        orchestrator = SimpleNamespace(event_manager=MagicMock())
        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=False,
            record=False,
        )
        await model_cmds._event_stream(
            args,
            console,
            live,
            group,
            0,
            1,
            orchestrator,
            theme,
            stop_signal=asyncio.Event(),
        )
        orchestrator.event_manager.listen.assert_not_called()
        theme.events.assert_not_called()
        live.refresh.assert_not_called()

    async def test_event_stream_events_only(self):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        events = MagicMock(name="events")
        tools = MagicMock(name="tools")
        group = SimpleNamespace(renderables=[events, tools])
        theme = MagicMock()
        theme.events.return_value = "panel"
        live = MagicMock()
        stop_signal = asyncio.Event()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=False,
            record=False,
        )
        console = MagicMock()
        task = asyncio.create_task(
            model_cmds._event_stream(
                args,
                console,
                live,
                group,
                0,
                1,
                orchestrator,
                theme,
                stop_signal=stop_signal,
            )
        )
        await orchestrator.event_manager.trigger(Event(type=EventType.START))
        await orchestrator.event_manager.trigger(Event(type=EventType.END))
        await asyncio.sleep(0)
        stop_signal.set()
        await task

        self.assertEqual(group.renderables[0], "panel")
        self.assertEqual(theme.events.call_count, 2)
        live.refresh.assert_called()

    async def test_event_stream_tools_only(self):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        events = MagicMock(name="events")
        tools = MagicMock(name="tools")
        group = SimpleNamespace(renderables=[events, tools])
        theme = MagicMock()
        theme.events.return_value = "panel"
        live = MagicMock()
        stop_signal = asyncio.Event()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=True,
            record=False,
        )
        console = MagicMock()
        task = asyncio.create_task(
            model_cmds._event_stream(
                args,
                console,
                live,
                group,
                0,
                1,
                orchestrator,
                theme,
                stop_signal=stop_signal,
            )
        )
        await orchestrator.event_manager.trigger(
            Event(type=EventType.TOOL_MODEL_RUN)
        )
        await orchestrator.event_manager.trigger(
            Event(type=EventType.TOOL_RESULT)
        )
        await asyncio.sleep(0)
        stop_signal.set()
        await task

        self.assertEqual(group.renderables[1], "panel")
        self.assertEqual(theme.events.call_count, 2)
        live.refresh.assert_called()

    async def test_event_stream_combined_tools_and_events_do_not_duplicate(
        self,
    ):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        group = SimpleNamespace(renderables=[MagicMock(), MagicMock()])
        theme = MagicMock()
        theme.events.side_effect = lambda *_args, **kwargs: (
            None if kwargs["include_tools"] else "panel"
        )
        live = MagicMock()
        stop_signal = asyncio.Event()
        console = MagicMock()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=True,
            display_tools_events=2,
            record=False,
        )
        task = asyncio.create_task(
            model_cmds._event_stream(
                args,
                console,
                live,
                group,
                0,
                1,
                orchestrator,
                theme,
                stop_signal=stop_signal,
            )
        )
        await orchestrator.event_manager.trigger(
            Event(type=EventType.TOOL_RESULT)
        )
        await orchestrator.event_manager.trigger(Event(type=EventType.START))
        await asyncio.sleep(0)
        stop_signal.set()
        await task

        include_tools = [
            call.kwargs["include_tools"]
            for call in theme.events.call_args_list
        ]
        include_non_tools = [
            call.kwargs["include_non_tools"]
            for call in theme.events.call_args_list
        ]
        self.assertEqual(include_tools, [True, False])
        self.assertEqual(include_non_tools, [False, True])

    async def test_event_stream_tool_event_limit_zero_keeps_non_tool_events(
        self,
    ):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        group = SimpleNamespace(renderables=[MagicMock(), MagicMock()])
        theme = MagicMock()
        theme.events.side_effect = lambda *_args, **kwargs: (
            None if kwargs["include_tools"] else "panel"
        )
        live = MagicMock()
        stop_signal = asyncio.Event()
        console = MagicMock()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=True,
            display_tools_events=0,
            record=False,
        )
        task = asyncio.create_task(
            model_cmds._event_stream(
                args,
                console,
                live,
                group,
                0,
                1,
                orchestrator,
                theme,
                stop_signal=stop_signal,
            )
        )
        await orchestrator.event_manager.trigger(
            Event(type=EventType.TOOL_RESULT)
        )
        await orchestrator.event_manager.trigger(Event(type=EventType.START))
        await asyncio.sleep(0)
        stop_signal.set()
        await task

        self.assertEqual(theme.events.call_count, 2)
        tool_call, event_call = theme.events.call_args_list
        self.assertEqual(tool_call.kwargs["events_limit"], 0)
        self.assertTrue(tool_call.kwargs["include_tools"])
        self.assertFalse(tool_call.kwargs["include_non_tools"])
        self.assertEqual(event_call.kwargs["events_limit"], 4)
        self.assertFalse(event_call.kwargs["include_tools"])
        self.assertTrue(event_call.kwargs["include_non_tools"])
        self.assertEqual(group.renderables[0], "panel")
        self.assertNotEqual(group.renderables[1], "panel")

    async def test_event_stream_routes_canonical_tool_payload_to_tools_panel(
        self,
    ):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        group = SimpleNamespace(
            renderables=[MagicMock(name="events"), MagicMock(name="tools")]
        )
        theme = MagicMock()
        theme.events.return_value = "tool-panel"
        live = MagicMock()
        stop_signal = asyncio.Event()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=True,
            record=False,
        )
        console = MagicMock()
        task = asyncio.create_task(
            model_cmds._event_stream(
                args,
                console,
                live,
                group,
                0,
                1,
                orchestrator,
                theme,
                stop_signal=stop_signal,
            )
        )
        await orchestrator.event_manager.trigger(
            Event.from_observability_payload(
                type=EventType.START,
                observability_payload=EventObservabilityPayload.canonical_stream(
                    {
                        "stream_session_id": "stream-1",
                        "run_id": "run-1",
                        "turn_id": "turn-1",
                        "sequence": 1,
                        "kind": "tool_call.ready",
                        "channel": "tool_call",
                        "visibility": "public",
                    }
                ),
            )
        )
        await asyncio.sleep(0)
        stop_signal.set()
        await task

        self.assertEqual(group.renderables[1], "tool-panel")
        theme.events.assert_called_once()
        self.assertTrue(theme.events.call_args.kwargs["tool_view"])

    async def test_event_stream_keeps_canonical_control_off_tools_panel(
        self,
    ):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        group = SimpleNamespace(
            renderables=[MagicMock(name="events"), MagicMock(name="tools")]
        )
        theme = MagicMock()
        theme.events.return_value = "panel"
        live = MagicMock()
        stop_signal = asyncio.Event()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=False,
            display_tools=True,
            record=False,
        )
        console = MagicMock()
        task = asyncio.create_task(
            model_cmds._event_stream(
                args,
                console,
                live,
                group,
                0,
                1,
                orchestrator,
                theme,
                stop_signal=stop_signal,
            )
        )
        await orchestrator.event_manager.trigger(
            Event.from_observability_payload(
                type=EventType.TOOL_DETECT,
                observability_payload=EventObservabilityPayload.canonical_stream(
                    {
                        "stream_session_id": "stream-1",
                        "run_id": "run-1",
                        "turn_id": "turn-1",
                        "sequence": 1,
                        "kind": "stream.diagnostic",
                        "channel": "control",
                        "visibility": "diagnostic",
                    }
                ),
            )
        )
        await asyncio.sleep(0)
        stop_signal.set()
        await task

        theme.events.assert_not_called()
        live.refresh.assert_not_called()

    async def test_event_stream_splits_canonical_diagnostics_and_tools(
        self,
    ) -> None:
        orchestrator = SimpleNamespace(event_manager=EventManager())
        group = SimpleNamespace(
            renderables=["events-slot", "tools-slot", "tokens-slot"]
        )
        theme = MagicMock()
        theme.events.side_effect = lambda _history, **kwargs: (
            "tools-panel" if kwargs["tool_view"] else "events-panel"
        )
        live = MagicMock()
        stop_signal = asyncio.Event()
        console = MagicMock()
        render_calls: list[tuple[int | None, object]] = []

        async def immediate_to_thread(
            func: Any, *args: object, **kwargs: object
        ) -> None:
            func(*args, **kwargs)

        def render_frame(*call_args: object) -> None:
            frame = call_args[3]
            target_group = cast(Any, call_args[4])
            group_index = cast(int | None, call_args[5])
            render_calls.append((group_index, frame))
            if target_group is not None and group_index is not None:
                target_group.renderables[group_index] = frame
            live.refresh()

        def canonical_event(
            event_type: EventType,
            kind: str,
            channel: str,
            *,
            sequence: int,
            visibility: str = "public",
        ) -> Event:
            return Event.from_observability_payload(
                type=event_type,
                observability_payload=EventObservabilityPayload.canonical_stream(
                    {
                        "stream_session_id": "stream-1",
                        "run_id": "run-1",
                        "turn_id": "turn-1",
                        "sequence": sequence,
                        "kind": kind,
                        "channel": channel,
                        "visibility": visibility,
                    }
                ),
            )

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=True,
            record=False,
        )

        with (
            patch.object(
                model_cmds, "to_thread", side_effect=immediate_to_thread
            ),
            patch.object(
                model_cmds, "_render_frame", side_effect=render_frame
            ),
        ):
            task = asyncio.create_task(
                model_cmds._event_stream(
                    args,
                    console,
                    live,
                    group,
                    0,
                    1,
                    orchestrator,
                    theme,
                    stop_signal=stop_signal,
                    refresh_per_second=1000,
                )
            )
            await orchestrator.event_manager.trigger(
                canonical_event(
                    EventType.TOOL_DETECT,
                    "stream.diagnostic",
                    "control",
                    sequence=1,
                    visibility="diagnostic",
                )
            )
            await orchestrator.event_manager.trigger(
                canonical_event(
                    EventType.START,
                    "tool.execution.started",
                    "control",
                    sequence=2,
                )
            )
            stop_signal.set()
            await task

        self.assertEqual(group.renderables[0], "events-panel")
        self.assertEqual(group.renderables[1], "tools-panel")
        self.assertCountEqual(
            render_calls,
            [(0, "events-panel"), (1, "tools-panel")],
        )
        first_call, second_call = theme.events.call_args_list
        self.assertFalse(first_call.kwargs["tool_view"])
        self.assertFalse(first_call.kwargs["include_tools"])
        self.assertTrue(first_call.kwargs["include_non_tools"])
        self.assertTrue(second_call.kwargs["tool_view"])
        self.assertTrue(second_call.kwargs["include_tools"])
        self.assertFalse(second_call.kwargs["include_non_tools"])

    async def test_event_stream_consumes_events_while_rendering_is_slow(
        self,
    ):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        group = SimpleNamespace(
            renderables=[MagicMock(name="events"), MagicMock(name="tools")]
        )
        theme = MagicMock()
        theme.events.side_effect = lambda history, **_: f"panel-{len(history)}"
        live = MagicMock()
        stop_signal = asyncio.Event()
        render_started = asyncio.Event()
        release_render = ThreadEvent()
        rendered: list[object] = []
        loop = asyncio.get_running_loop()

        def slow_render(*call_args: object) -> None:
            frame = call_args[3]
            target_group = cast(Any, call_args[4])
            group_index = cast(int | None, call_args[5])
            rendered.append(frame)
            if target_group is not None and group_index is not None:
                target_group.renderables[group_index] = frame
            loop.call_soon_threadsafe(render_started.set)
            release_render.wait(1.0)

        async def wait_for_theme_calls(total: int) -> None:
            for _ in range(100):
                if theme.events.call_count >= total:
                    return
                await asyncio.sleep(0.01)
            self.fail(f"theme.events was called {theme.events.call_count}x")

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=False,
            record=False,
        )
        console = MagicMock()

        with patch.object(
            model_cmds, "_render_frame", side_effect=slow_render
        ):
            task = asyncio.create_task(
                model_cmds._event_stream(
                    args,
                    console,
                    live,
                    group,
                    0,
                    1,
                    orchestrator,
                    theme,
                    stop_signal=stop_signal,
                    refresh_per_second=60,
                )
            )
            await orchestrator.event_manager.trigger(
                Event(type=EventType.START)
            )
            await asyncio.wait_for(render_started.wait(), timeout=1.0)

            await orchestrator.event_manager.trigger(Event(type=EventType.END))
            await orchestrator.event_manager.trigger(
                Event(type=EventType.MODEL_EXECUTE_AFTER)
            )
            await wait_for_theme_calls(3)

            stop_signal.set()
            release_render.set()
            await task

        self.assertEqual(theme.events.call_count, 3)
        self.assertEqual(rendered[-1], "panel-3")
        self.assertEqual(group.renderables[0], "panel-3")

    async def test_event_stream_skip_unselected_tool(self):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        group = SimpleNamespace(renderables=[MagicMock(), MagicMock()])
        theme = MagicMock()
        live = MagicMock()
        stop_signal = asyncio.Event()
        console = MagicMock()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_events=True,
            display_tools=False,
            record=False,
        )
        task = asyncio.create_task(
            model_cmds._event_stream(
                args,
                console,
                live,
                group,
                0,
                1,
                orchestrator,
                theme,
                stop_signal=stop_signal,
            )
        )
        await orchestrator.event_manager.trigger(
            Event(type=EventType.TOOL_RESULT)
        )
        await asyncio.sleep(0)
        stop_signal.set()
        await task

        theme.events.assert_not_called()
        live.refresh.assert_not_called()

    async def test_token_stream_extra_frames_and_stop(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        def fake_frames(*_: object, **__: object):
            return (
                (Token(id=1, token="A"), "frame1"),
                (None, "frame2"),
                (None, "frame3"),
            )

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=True,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        live = MagicMock()
        logger = MagicMock()
        stop_signal = asyncio.Event()

        class CaptureList(list):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.calls: list = []

            def __setitem__(self, index: int, value):
                self.calls.append(value)
                super().__setitem__(index, value)

        group = SimpleNamespace(renderables=CaptureList([None]))

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=lambda s: 1
        )

        await model_cmds._token_stream(
            live=live,
            group=group,
            tokens_group_index=0,
            args=args,
            console=console,
            theme=theme,
            logger=logger,
            orchestrator=None,
            event_stats=None,
            lm=lm,
            input_string="hi",
            response=Resp(),
            display_tokens=1,
            dtokens_pick=1,
            refresh_per_second=2,
            stop_signal=stop_signal,
            tool_events_limit=None,
            with_stats=True,
        )

        self.assertTrue(stop_signal.is_set())
        self.assertEqual(group.renderables.calls, ["frame3"])
        live.refresh.assert_called()
        theme.token_frames.assert_called_once()

    async def test_token_stream_sets_stop_signal_when_cancelled(self):
        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        stop_signal = asyncio.Event()

        with self.assertRaises(asyncio.CancelledError):
            await model_cmds._token_stream(
                live=MagicMock(),
                group=SimpleNamespace(renderables=[None]),
                tokens_group_index=0,
                args=args,
                console=MagicMock(width=80),
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(
                    model_id="m",
                    tokenizer_config=None,
                    input_token_count=lambda s: 1,
                ),
                input_string="hi",
                response=Resp(),
                display_tokens=1,
                dtokens_pick=1,
                refresh_per_second=2,
                stop_signal=stop_signal,
                tool_events_limit=None,
                with_stats=True,
            )

        self.assertTrue(stop_signal.is_set())

    async def test_token_generation_events_stop_when_token_stream_cancelled(
        self,
    ):
        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            display_events=True,
            display_tools=True,
            display_tools_events=None,
            record=False,
        )

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        orchestrator = SimpleNamespace(
            event_manager=EventManager(), input_token_count=1
        )

        with patch.object(model_cmds, "Live", return_value=live):
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(
                    model_cmds.token_generation(
                        args=args,
                        console=MagicMock(width=80),
                        theme=MagicMock(),
                        logger=MagicMock(),
                        orchestrator=orchestrator,
                        event_stats=None,
                        lm=SimpleNamespace(
                            model_id="m",
                            tokenizer_config=None,
                            input_token_count=lambda s: 1,
                        ),
                        input_string="hi",
                        response=Resp(),
                        display_tokens=1,
                        dtokens_pick=1,
                        refresh_per_second=2,
                        tool_events_limit=None,
                        with_stats=True,
                    ),
                    timeout=1,
                )

    async def test_token_stream_pause_no_probabilities(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        def fake_frames(*_: object, **__: object):
            return ((None, "frame1"),)

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=10,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        live = MagicMock()
        logger = MagicMock()
        stop_signal = asyncio.Event()

        class CaptureList(list):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)

            def __setitem__(self, index: int, value):
                super().__setitem__(index, value)

        group = SimpleNamespace(renderables=CaptureList([None]))

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=lambda s: 1
        )

        with patch("avalan.cli.commands.model.sleep", new=AsyncMock()) as slp:
            await model_cmds._token_stream(
                live=live,
                group=group,
                tokens_group_index=0,
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="hi",
                response=Resp(),
                display_tokens=1,
                dtokens_pick=1,
                refresh_per_second=2,
                stop_signal=stop_signal,
                tool_events_limit=None,
                with_stats=True,
            )

        self.assertTrue(stop_signal.is_set())
        slp.assert_called()

    async def test_token_stream_rejects_non_iterable_theme_frames(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        def bad_frames(*_args: object, **_kwargs: object) -> object:
            return object()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        stop_signal = asyncio.Event()
        group = SimpleNamespace(renderables=[None])
        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=bad_frames)
        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=lambda _: 1
        )

        with self.assertRaises(TypeError):
            await model_cmds._token_stream(
                live=MagicMock(),
                group=group,
                tokens_group_index=0,
                args=args,
                console=console,
                theme=theme,
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="hi",
                response=Resp(),
                display_tokens=1,
                dtokens_pick=0,
                refresh_per_second=2,
                stop_signal=stop_signal,
                tool_events_limit=None,
                with_stats=True,
            )

        self.assertTrue(stop_signal.is_set())

    async def test_token_stream_skips_stale_slow_theme_frame_build(self):
        build_started = ThreadEvent()
        release_build = ThreadEvent()
        answer_consumed = asyncio.Event()
        consumed: list[str] = []

        async def token_gen():
            items = _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1),
                _canonical_answer_delta("B", token_id=2),
            )
            for item in items:
                if item.kind is StreamItemKind.ANSWER_DELTA:
                    assert item.text_delta is not None
                    consumed.append(item.text_delta)
                yield item
                if item.text_delta == "A":
                    self.assertTrue(
                        await asyncio.to_thread(build_started.wait, 1)
                    )
                if item.text_delta == "B":
                    answer_consumed.set()
                await asyncio.sleep(0)

        async def stale_token_gen():
            async for item in token_gen():
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return stale_token_gen()

        built_answers: list[str] = []

        def fake_frames(state: TokenRenderState, **_kwargs: object):
            answer = "".join(state.answer_text_tokens)
            built_answers.append(answer)
            if answer == "A":
                build_started.set()
                release_build.wait(timeout=2)
            return ((None, f"frame-{answer}"),)

        rendered: list[str] = []

        def slow_render(*call_args):
            rendered.append(call_args[3])

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)
        stop_signal = asyncio.Event()

        with patch.object(
            model_cmds, "_render_frame", side_effect=slow_render
        ):
            task = asyncio.create_task(
                model_cmds._token_stream(
                    live=MagicMock(),
                    group=SimpleNamespace(renderables=[None]),
                    tokens_group_index=0,
                    args=args,
                    console=MagicMock(width=80),
                    theme=theme,
                    logger=MagicMock(),
                    orchestrator=None,
                    event_stats=None,
                    lm=SimpleNamespace(
                        model_id="m",
                        tokenizer_config=None,
                        input_token_count=lambda s: 1,
                    ),
                    input_string="hi",
                    response=Resp(),
                    display_tokens=1,
                    dtokens_pick=0,
                    refresh_per_second=1000,
                    stop_signal=stop_signal,
                    tool_events_limit=None,
                    with_stats=True,
                )
            )
            await asyncio.wait_for(answer_consumed.wait(), timeout=1)
            self.assertEqual(consumed, ["A", "B"])
            self.assertFalse(task.done())
            release_build.set()
            await task

        self.assertTrue(stop_signal.is_set())
        self.assertIn("A", built_answers)
        self.assertEqual(built_answers[-1], "AB")
        self.assertEqual(rendered, ["frame-AB"])


class CliRecordOptionTestCase(IsolatedAsyncioTestCase):
    async def test_token_generation_record_enables_screen(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            record=True,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            display_events=False,
            display_tools=False,
        )

        console = MagicMock()
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    yield "answer"

                return gen()

        async def fake_tokens(**kwargs):
            _ = kwargs
            yield (None, "frame")

        lm = SimpleNamespace(model_id="m", tokenizer_config=None)
        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=fake_tokens)

        with patch(
            "avalan.cli.stream_coordinator.Live", return_value=live
        ) as live_patch:
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="text",
                response=Resp(),
                display_tokens=0,
                dtokens_pick=0,
                tool_events_limit=None,
                refresh_per_second=3,
                with_stats=True,
            )

        live_patch.assert_called_once_with(
            None,
            refresh_per_second=3,
            screen=True,
            console=console,
        )
        live.update.assert_called()
        console.save_svg.assert_called_once()


class CliRenderFrameTestCase(IsolatedAsyncioTestCase):
    async def test_frame_rate_renderer_renders_latest_dirty_frame(self):
        args = Namespace(skip_display_reasoning_time=False, record=False)
        console = MagicMock()
        live = MagicMock()
        group = SimpleNamespace(renderables=[None])
        rendered: list[str] = []

        def capture_render(*call_args):
            rendered.append(call_args[3])

        with patch.object(
            model_cmds, "_render_frame", side_effect=capture_render
        ):
            renderer = model_cmds._FrameRateRenderer(
                args,
                console,
                live,
                group,
                0,
                refresh_per_second=10,
            )
            renderer.mark_dirty("first")
            renderer.mark_dirty("latest")
            await renderer.close()

        self.assertEqual(rendered, ["latest"])

    async def test_frame_rate_renderer_coalesces_fancy_theme_frames(self):
        async def fancy_frame(answer: str) -> object:
            frames = theme.token_frames(
                TokenRenderState(
                    model_id="m",
                    answer_text_tokens=(answer,),
                    input_token_count=1,
                    total_tokens=1,
                    ttft=0.0,
                    elapsed=1.0,
                ),
                console_width=80,
                logger=MagicMock(),
                maximum_frames=1,
            )
            return frames[0][1]

        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        first_frame = await fancy_frame("first")
        latest_frame = await fancy_frame("latest")
        args = Namespace(skip_display_reasoning_time=False, record=False)
        rendered: list[object] = []

        def capture_render(*call_args: object) -> None:
            rendered.append(call_args[3])

        with patch.object(
            model_cmds, "_render_frame", side_effect=capture_render
        ):
            renderer = model_cmds._FrameRateRenderer(
                args,
                MagicMock(),
                MagicMock(),
                None,
                None,
                refresh_per_second=10,
            )
            renderer.mark_dirty(first_frame)
            renderer.mark_dirty(latest_frame)
            await renderer.close()

        self.assertEqual(len(rendered), 1)
        self.assertIs(rendered[0], latest_frame)

    async def test_frame_rate_renderer_uses_render_lock(self):
        args = Namespace(skip_display_reasoning_time=False, record=False)
        rendered: list[object] = []

        def capture_render(*call_args: object) -> None:
            rendered.append(call_args[3])

        with patch.object(
            model_cmds, "_render_frame", side_effect=capture_render
        ):
            renderer = model_cmds._FrameRateRenderer(
                args,
                MagicMock(),
                MagicMock(),
                None,
                None,
                refresh_per_second=10,
                render_lock=asyncio.Lock(),
            )
            renderer.mark_dirty("locked")
            await renderer.close()

        self.assertEqual(rendered, ["locked"])

    async def test_frame_rate_renderer_closes_without_dirty_frame(self):
        args = Namespace(skip_display_reasoning_time=False, record=False)

        with patch.object(model_cmds, "_render_frame") as render_frame:
            renderer = model_cmds._FrameRateRenderer(
                args,
                MagicMock(),
                MagicMock(),
                None,
                None,
                refresh_per_second=10,
            )
            await renderer.close()

        render_frame.assert_not_called()

    async def test_frame_rate_renderer_ignores_spurious_dirty_wake(self):
        args = Namespace(skip_display_reasoning_time=False, record=False)

        with patch.object(model_cmds, "_render_frame") as render_frame:
            renderer = model_cmds._FrameRateRenderer(
                args,
                MagicMock(),
                MagicMock(),
                None,
                None,
                refresh_per_second=10,
            )
            renderer._dirty.set()
            await asyncio.sleep(0)
            await renderer.close()

        render_frame.assert_not_called()

    def test_frame_rate_renderer_rejects_invalid_refresh_rate(self):
        args = Namespace(skip_display_reasoning_time=False, record=False)

        with self.assertRaises(AssertionError):
            model_cmds._FrameRateRenderer(
                args,
                MagicMock(),
                MagicMock(),
                None,
                None,
                refresh_per_second=0,
            )

    def test_render_frame_rejects_recording_bypass(self):
        args = Namespace(skip_display_reasoning_time=False, record=True)
        console = MagicMock()
        live = MagicMock()

        with self.assertRaises(AssertionError):
            model_cmds._render_frame(args, console, live, "frame")

        console.save_svg.assert_not_called()
        live.update.assert_not_called()

    def test_render_frame_no_record(self):
        args = Namespace(skip_display_reasoning_time=False, record=False)
        console = MagicMock()
        live = MagicMock()

        model_cmds._render_frame(args, console, live, "frame")

        console.save_svg.assert_not_called()
        live.update.assert_called_once_with("frame")

    async def test_token_stream_second_frames_pause(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        async def fake_frames(*_, **__):
            yield (Token(id=1, token="A"), "frame1")
            yield (Token(id=2, token="B"), "frame2")

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=1,
            display_pause=10,
            start_thinking=False,
            display_probabilities=True,
            display_probabilities_maximum=1.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        live = MagicMock()
        logger = MagicMock()
        stop_signal = asyncio.Event()

        class CaptureList(list):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)

            def __setitem__(self, index: int, value):
                super().__setitem__(index, value)

        group = SimpleNamespace(renderables=CaptureList([None]))

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=lambda s: 1
        )

        with patch("avalan.cli.commands.model.sleep", new=AsyncMock()) as slp:
            await model_cmds._token_stream(
                live=live,
                group=group,
                tokens_group_index=0,
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="hi",
                response=Resp(),
                display_tokens=1,
                dtokens_pick=1,
                refresh_per_second=2,
                stop_signal=stop_signal,
                tool_events_limit=None,
                with_stats=True,
            )

        self.assertTrue(stop_signal.is_set())
        slp.assert_called()

    async def test_token_stream_start_thinking_orchestrator_response(self):
        async def gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        response = TextGenerationResponse(
            lambda: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
        )

        engine = SimpleNamespace(model_id="m", tokenizer=MagicMock())
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = MagicMock()
        input_message = Message(role=MessageRole.USER, content="hi")
        context = ModelCallContext(
            specification=operation.specification,
            input=input_message,
            engine_args={},
        )
        orch_response = OrchestratorResponse(
            input_message,
            response,
            agent,
            operation,
            {},
            context,
        )

        orchestrator = SimpleNamespace(event_manager=None, input_token_count=1)

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=True,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        live = MagicMock()
        logger = MagicMock()
        stop_signal = asyncio.Event()
        group = SimpleNamespace(renderables=[None])

        async def fake_frames(*_, **__):
            yield (None, "frame")

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=lambda s: 1
        )

        self.assertTrue(orch_response.can_think)
        self.assertFalse(orch_response.is_thinking)

        await model_cmds._token_stream(
            live=live,
            group=group,
            tokens_group_index=0,
            args=args,
            console=console,
            theme=theme,
            logger=logger,
            orchestrator=orchestrator,
            event_stats=None,
            lm=lm,
            input_string="hi",
            response=orch_response,
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=stop_signal,
            tool_events_limit=None,
            with_stats=True,
        )

        self.assertTrue(stop_signal.is_set())
        self.assertTrue(orch_response.is_thinking)
        theme.token_frames.assert_called_once()

    async def test_token_stream_answer_height_expand_option(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        async def fake_frames(*_, **__):
            yield (None, "frame")

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
            display_answer_height_expand=True,
        )

        console = MagicMock()
        console.width = 80
        live = MagicMock()
        logger = MagicMock()

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=lambda s: 1
        )

        await model_cmds._token_stream(
            live=live,
            group=None,
            tokens_group_index=None,
            args=args,
            console=console,
            theme=theme,
            logger=logger,
            orchestrator=None,
            event_stats=None,
            lm=lm,
            input_string="hi",
            response=Resp(),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=None,
            with_stats=True,
        )

        self.assertFalse(
            theme.token_frames.call_args.kwargs["limit_answer_height"]
        )

    async def test_token_stream_answer_height_option(self):
        async def token_gen():
            for item in _canonical_answer_stream_items(
                _canonical_answer_delta("A", token_id=1)
            ):
                yield item

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                return token_gen()

        async def fake_frames(*_, **__):
            yield (None, "frame")

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
            display_answer_height=20,
        )

        console = MagicMock()
        console.width = 80
        live = MagicMock()
        logger = MagicMock()

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_frames)

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=lambda s: 1
        )

        await model_cmds._token_stream(
            live=live,
            group=None,
            tokens_group_index=None,
            args=args,
            console=console,
            theme=theme,
            logger=logger,
            orchestrator=None,
            event_stats=None,
            lm=lm,
            input_string="hi",
            response=Resp(),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=None,
            tool_events_limit=None,
            with_stats=True,
        )

        self.assertTrue(
            theme.token_frames.call_args.kwargs["limit_answer_height"]
        )
        self.assertEqual(theme.token_frames.call_args.kwargs["height"], 20)


class CliReasoningTokenTestCase(IsolatedAsyncioTestCase):
    async def test_reasoning_token_tracked(self):
        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in _canonical_reasoning_answer_stream_items(
                        reasoning=("A",),
                        answer=("B",),
                    ):
                        yield item

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()

        captured: list[dict[str, list[str]]] = []

        async def fake_tokens(*p, **kw):
            state = p[0]
            captured.append(
                {
                    "thinking": list(state.reasoning_text_tokens),
                    "answer": list(state.answer_text_tokens),
                }
            )
            yield (None, "frame")

        theme = _legacy_adapter_theme(fake_tokens)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        with patch("avalan.cli.stream_coordinator.Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(model_id="m", tokenizer_config=None),
                input_string="text",
                response=Resp(),
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        self.assertEqual(captured[-1]["thinking"], ["A"])
        self.assertEqual(captured[-1]["answer"], ["B"])


class CliToolCallTokenTestCase(IsolatedAsyncioTestCase):
    async def test_tool_call_token_tracked(self):
        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    for item in _canonical_tool_call_answer_stream_items(
                        "TOOL", _canonical_answer_delta("A", token_id=1)
                    ):
                        yield item

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()
        stop_signal = asyncio.Event()

        captured: list[list[str]] = []

        async def fake_tokens(*p, **kw):
            state = p[0]
            captured.append(list(state.tool_text_tokens))
            yield (None, "frame")

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_tokens)

        live = MagicMock()

        group = SimpleNamespace(renderables=[None])

        await model_cmds._token_stream(
            live=live,
            group=group,
            tokens_group_index=0,
            args=args,
            console=console,
            theme=theme,
            logger=logger,
            orchestrator=None,
            event_stats=None,
            lm=SimpleNamespace(
                model_id="m",
                tokenizer_config=None,
                input_token_count=lambda s: 1,
            ),
            input_string="text",
            response=Resp(),
            display_tokens=1,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=stop_signal,
            tool_events_limit=None,
            with_stats=True,
        )

        self.assertTrue(stop_signal.is_set())
        self.assertEqual(captured[-1], ["TOOL"])

    async def test_token_stream_tracks_canonical_items(self):
        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=0,
                        kind=StreamItemKind.STREAM_STARTED,
                        channel=StreamChannel.CONTROL,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=1,
                        kind=StreamItemKind.REASONING_DELTA,
                        channel=StreamChannel.REASONING,
                        text_delta="PLAN",
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=2,
                        kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        channel=StreamChannel.TOOL_CALL,
                        correlation=StreamItemCorrelation(
                            tool_call_id="call-1"
                        ),
                        text_delta="TOOL",
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=3,
                        kind=StreamItemKind.TOOL_CALL_READY,
                        channel=StreamChannel.TOOL_CALL,
                        correlation=StreamItemCorrelation(
                            tool_call_id="call-1"
                        ),
                        data={"name": "tool", "arguments": {}},
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=4,
                        kind=StreamItemKind.TOOL_CALL_DONE,
                        channel=StreamChannel.TOOL_CALL,
                        correlation=StreamItemCorrelation(
                            tool_call_id="call-1"
                        ),
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=5,
                        kind=StreamItemKind.REASONING_DONE,
                        channel=StreamChannel.REASONING,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=6,
                        kind=StreamItemKind.ANSWER_DELTA,
                        channel=StreamChannel.ANSWER,
                        text_delta="ANSWER",
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=7,
                        kind=StreamItemKind.ANSWER_DONE,
                        channel=StreamChannel.ANSWER,
                    )
                    yield CanonicalStreamItem(
                        stream_session_id="s",
                        run_id="r",
                        turn_id="t",
                        sequence=8,
                        kind=StreamItemKind.STREAM_COMPLETED,
                        channel=StreamChannel.CONTROL,
                        usage={},
                        terminal_outcome=StreamTerminalOutcome.COMPLETED,
                    )

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        captured: list[dict[str, list[str]]] = []

        async def fake_tokens(*p, **kw):
            state = p[0]
            captured.append(
                {
                    "thinking": list(state.reasoning_text_tokens),
                    "tool": list(state.tool_text_tokens),
                    "answer": list(state.answer_text_tokens),
                }
            )
            yield (None, "frame")

        theme = MagicMock()
        theme.token_frames = MagicMock(side_effect=fake_tokens)
        stop_signal = asyncio.Event()

        await model_cmds._token_stream(
            live=MagicMock(),
            group=SimpleNamespace(renderables=[None]),
            tokens_group_index=0,
            args=args,
            console=MagicMock(width=80),
            theme=theme,
            logger=MagicMock(),
            orchestrator=None,
            event_stats=None,
            lm=SimpleNamespace(
                model_id="m",
                tokenizer_config=None,
                input_token_count=lambda s: 1,
            ),
            input_string="text",
            response=Resp(),
            display_tokens=0,
            dtokens_pick=0,
            refresh_per_second=2,
            stop_signal=stop_signal,
            tool_events_limit=None,
            with_stats=True,
        )

        self.assertTrue(stop_signal.is_set())
        self.assertEqual(captured[-1]["thinking"], ["PLAN"])
        self.assertEqual(captured[-1]["tool"], ["TOOL"])
        self.assertEqual(captured[-1]["answer"], ["ANSWER"])

    async def test_tool_result_event_with_none_result_is_rejected(self):
        call_obj = ToolCall(id="call-none", name="tool", arguments={})

        class Resp:
            input_token_count = 1
            can_think = False
            is_thinking = False

            def set_thinking(self, value: bool) -> None:
                self.is_thinking = value

            def __aiter__(self):
                async def gen():
                    yield Event(
                        type=EventType.TOOL_PROCESS,
                        payload=[call_obj],
                    )
                    yield Event(
                        type=EventType.TOOL_RESULT,
                        payload={"call": call_obj, "result": None},
                    )
                    for item in _canonical_answer_stream_items("A"):
                        yield item

                return gen()

        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=False,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()
        stop_signal = asyncio.Event()

        theme = MagicMock()
        theme.get_spinner.return_value = "dots"
        theme._n.side_effect = lambda singular, plural, count: singular

        live = MagicMock()
        group = SimpleNamespace(renderables=[None])

        with self.assertRaises(StreamValidationError):
            await model_cmds._token_stream(
                live=live,
                group=group,
                tokens_group_index=0,
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=SimpleNamespace(
                    model_id="m",
                    tokenizer_config=None,
                    input_token_count=lambda s: 1,
                ),
                input_string="text",
                response=Resp(),
                display_tokens=0,
                dtokens_pick=0,
                refresh_per_second=2,
                stop_signal=stop_signal,
                tool_events_limit=None,
                with_stats=True,
            )

        theme.token_frames.assert_not_called()


class CliModelMixedTokensTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        _disable_mlx_model_import(self)

    async def test_model_run_mixed_tokens(self):
        async def complex_generator():
            rp = ReasoningParser(
                reasoning_settings=ReasoningSettings(),
                logger=getLogger(),
            )
            tm = MagicMock()
            tm.is_potential_tool_call.return_value = True
            tm.get_calls.return_value = None
            base_parser = ToolCallParser()
            tm.tool_call_status.side_effect = base_parser.tool_call_status
            tp = ToolCallResponseParser(tm, None)
            sequence = [
                "X",
                "<think>",
                "ra",
                "rb",
                "</think>",
                "Y",
                "<tool_call>",
                "foo",
                "bar",
                "</tool_call>",
                "Z",
            ]
            for s in sequence:
                items = await rp.push(s)
                for item in items:
                    parsed = (
                        await tp.push(item)
                        if isinstance(item, str)
                        else [item]
                    )
                    for p in parsed:
                        assert isinstance(p, StreamProviderEvent)
                        yield p

        settings = GenerationSettings(
            reasoning=ReasoningSettings(enabled=False)
        )
        response = TextGenerationResponse(
            lambda **_: "",
            logger=getLogger(),
            use_async_generator=False,
            generation_settings=settings,
            settings=settings,
        )

        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=False,
            skip_hub_access_check=True,
            no_repl=True,
            do_sample=False,
            enable_gradient_calculation=False,
            min_p=None,
            repetition_penalty=1.0,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=2,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
        )

        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">"}
        theme.model.return_value = "panel"
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = "hub_model"
        logger = MagicMock()

        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        lm = AsyncMock(return_value=response)
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"
        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        with (
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ),
            patch.object(model_cmds, "get_input", return_value="hi"),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)

        tg_patch.assert_awaited_once()
        tokens = []
        async for t in complex_generator():
            tokens.append(t)

        self.assertFalse(any(isinstance(t, ReasoningToken) for t in tokens))
        reasoning_deltas = [
            t
            for t in tokens
            if (
                isinstance(t, StreamProviderEvent)
                and t.kind is StreamItemKind.REASONING_DELTA
            )
        ]
        self.assertEqual(
            [event.text_delta for event in reasoning_deltas],
            ["<think>", "ra", "rb", "</think>"],
        )
        self.assertTrue(
            any(
                isinstance(t, StreamProviderEvent)
                and t.kind is StreamItemKind.REASONING_DONE
                for t in tokens
            )
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, ToolCallToken)]),
            0,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, TokenDetail)]),
            0,
        )
        self.assertEqual(len([t for t in tokens if type(t) is Token]), 0)
        self.assertEqual(len([t for t in tokens if isinstance(t, str)]), 0)
        answer_events = [
            t
            for t in tokens
            if (
                isinstance(t, StreamProviderEvent)
                and t.kind is StreamItemKind.ANSWER_DELTA
            )
        ]
        self.assertEqual(
            [event.text_delta for event in answer_events],
            ["X", "Y", "Z"],
        )


if __name__ == "__main__":
    main()
