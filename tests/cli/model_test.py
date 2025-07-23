from avalan.cli.commands import get_model_settings
from avalan.cli.commands import model as model_cmds
from avalan.entities import (
    Modality,
    ImageEntity,
    Token,
    TokenDetail,
    ToolCallToken,
)
from avalan.event.manager import EventManager
from avalan.event import Event, EventType
from types import SimpleNamespace
from argparse import Namespace
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch, call, ANY
from avalan.model.manager import ModelManager as RealModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.parsers.tool import ToolCallParser
import asyncio
from unittest import IsolatedAsyncioTestCase, main, TestCase


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
            attention="flash",
            base_url="http://localhost:9001/v1",
            device="cpu",
            disable_loading_progress_bar=True,
            sentence_transformer=True,
            loader_class="auto",
            low_cpu_mem_usage=True,
            quiet=False,
            revision="rev",
            special_token=["<s>"],
            tokenizer="tok",
            token=["t"],
            trust_remote_code=True,
            weight_type="fp16",
            tool_events=2,
        )

        result = get_model_settings(args, self.hub, self.logger, engine_uri)
        expected = {
            "engine_uri": engine_uri,
            "base_url": "http://localhost:9001/v1",
            "attention": "flash",
            "device": "cpu",
            "parallel": None,
            "disable_loading_progress_bar": True,
            "modality": Modality.EMBEDDING,
            "loader_class": "auto",
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

    def test_model_install_secret_creates_secret(self):
        args = Namespace(model="m")
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

    def test_model_uninstall_secret(self):
        args = Namespace(model="m")
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

    def test_model_display_uses_provided_model(self):
        args = Namespace(
            model="id", skip_hub_access_check=False, summary=False, load=False
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
            model="id", skip_hub_access_check=False, summary=False, load=True
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
        args = Namespace(model="m")
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
    async def test_token_generation_no_stats(self):
        async def gen():
            for t in ["a", "b"]:
                yield t

        args = Namespace()
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

    async def test_token_generation_no_stats_with_event(self):
        async def gen():
            yield model_cmds.Event(
                type=model_cmds.EventType.TOOL_EXECUTE, payload={}
            )
            yield "t"

        args = Namespace()
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
        console.print.assert_called_once_with("t", end="")

    async def test_token_generation_timing_pause(self):
        token = model_cmds.Token(id=0, token="a")

        class Resp:
            input_token_count = 1

            def __aiter__(self):
                async def g():
                    for _ in range(2):
                        yield token

                return g()

        args = Namespace(
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

        async def fake_tokens(*p, **kw):
            yield (token, "frame1")
            yield (None, "frame2")

        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=fake_tokens)

        live = MagicMock()
        console = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        lm = SimpleNamespace(model_id="m", tokenizer_config=None)

        with patch.object(model_cmds, "Live", return_value=live):
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

        theme.tokens.assert_called()
        live.update.assert_any_call("frame1")
        live.update.assert_any_call("frame2")

    async def test_token_generation_ttnt_metric(self):
        token = model_cmds.Token(id=0, token="a")

        class Resp:
            input_token_count = 1

            def __aiter__(self):
                async def g():
                    yield token

                return g()

        args = Namespace(
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

        async def fake_tokens(*p, **kw):
            yield (None, "frame")

        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=fake_tokens)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        lm = SimpleNamespace(model_id="m", tokenizer_config=None)

        with patch.object(model_cmds, "Live", return_value=live):
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

        theme.tokens.assert_called_once()
        live.update.assert_called_once_with("frame")

    async def test_token_generation_with_stats(self):
        token = model_cmds.Token(id=0, token="a", probability=0.4)

        class Resp:
            def __init__(self, toks):
                self._toks = toks
                self.input_token_count = 1

            def __aiter__(self):
                async def gen():
                    for t in self._toks:
                        yield t

                return gen()

        response = Resp([token])

        args = Namespace(
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

        async def fake_tokens(*p, **kw):
            captured["text_tokens"] = list(p[7]) + list(p[9])
            captured["input_token_count"] = p[11]
            yield (token, "frame1")
            yield (None, "frame2")

        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=fake_tokens)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=MagicMock()
        )

        with patch.object(model_cmds, "Live", return_value=live):
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

        theme.tokens.assert_called_once()
        self.assertEqual(captured["text_tokens"], ["a"])
        self.assertEqual(captured["input_token_count"], 1)
        live.update.assert_has_calls([call("frame1"), call("frame2")])
        lm.input_token_count.assert_not_called()

    async def test_token_generation_input_count_fallback(self):
        token = model_cmds.Token(id=0, token="a")

        class Resp:
            def __init__(self, toks, count):
                self._toks = toks
                self.input_token_count = count

            def __aiter__(self):
                async def gen():
                    for t in self._toks:
                        yield t

                return gen()

        args = Namespace(
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

        def gen_frame(*a, **k):
            async def g():
                yield (None, "frame")

            return g()

        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=gen_frame)

        lm = MagicMock()
        lm.model_id = "m"
        lm.tokenizer_config = None
        lm.input_token_count = MagicMock(return_value=33)

        # Response provides count, orchestrator should be ignored
        response = Resp([token], count=5)
        orchestrator = SimpleNamespace(input_token_count=7, event_manager=None)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        with patch.object(model_cmds, "Live", return_value=live):
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

        self.assertEqual(theme.tokens.call_args[0][11], 5)
        lm.input_token_count.assert_not_called()

        # Response has zero count, fall back to orchestrator
        response_zero = Resp([token], count=0)
        theme.tokens.reset_mock()

        with patch.object(model_cmds, "Live", return_value=live):
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

        self.assertEqual(theme.tokens.call_args[0][11], 7)

        # Response zero and orchestrator none -> use lm.input_token_count
        theme.tokens.reset_mock()
        lm.input_token_count.reset_mock()

        with patch.object(model_cmds, "Live", return_value=live):
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

        self.assertEqual(theme.tokens.call_args[0][11], 33)
        lm.input_token_count.assert_called_once_with("text")

    async def test_token_generation_tool_events(self):
        token_a = model_cmds.Token(id=0, token="a")
        token_b = model_cmds.Token(id=1, token="b")
        token_c = model_cmds.Token(id=2, token="c")

        call = SimpleNamespace(id="c1", name="calc")

        async def inner_gen():
            yield "x"

        inner_response = model_cmds.TextGenerationResponse(
            lambda: inner_gen(),
            inputs={"input_ids": [[1, 2, 3]]},
            use_async_generator=True,
        )

        events = [
            model_cmds.Event(
                type=model_cmds.EventType.TOOL_EXECUTE,
                payload=[call],
            ),
            token_a,
            model_cmds.Event(
                type=model_cmds.EventType.TOOL_RESULT,
                payload={"call": call},
            ),
            token_b,
            model_cmds.Event(
                type=model_cmds.EventType.TOOL_MODEL_RESPONSE,
                payload={"response": inner_response},
            ),
            token_c,
        ]

        class Resp:
            def __init__(self, items, count):
                self._items = items
                self.input_token_count = count

            def __aiter__(self):
                async def gen():
                    for i in self._items:
                        yield i

                return gen()

        response = Resp(events, 1)

        args = Namespace(
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

        call_args: list[dict] = []

        async def fake_tokens(*p, **kw):
            call_args.append(
                {
                    "text_tokens": list(p[7]) + list(p[9]),
                    "tokens": list(p[10]) if p[10] else None,
                    "tool_events": list(p[13]),
                    "tool_event_calls": list(p[14]),
                    "tool_event_results": list(p[15]),
                    "spinner": p[16],
                    "input_token_count": p[11],
                }
            )
            yield (None, "frame")

        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=fake_tokens)
        theme.get_spinner.return_value = "dots"
        theme._n = lambda s, p, n: s if n == 1 else p

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        lm = SimpleNamespace(
            model_id="m",
            tokenizer_config=None,
            input_token_count=MagicMock(),
        )

        with patch.object(model_cmds, "Live", return_value=live):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="text",
                response=response,
                display_tokens=0,
                dtokens_pick=0,
                with_stats=True,
                tool_events_limit=2,
                refresh_per_second=2,
            )

        self.assertEqual(len(call_args), len(events))

        first = call_args[0]
        self.assertEqual(first["text_tokens"], [])
        self.assertIsNone(first["tokens"])
        self.assertEqual(
            first["tool_events"][0].type, model_cmds.EventType.TOOL_EXECUTE
        )
        self.assertEqual(
            first["tool_event_calls"][0].type,
            model_cmds.EventType.TOOL_EXECUTE,
        )
        self.assertIsNotNone(first["spinner"])

        third = call_args[2]
        self.assertEqual(
            third["tool_event_results"][0].type,
            model_cmds.EventType.TOOL_RESULT,
        )
        self.assertIsNone(third["spinner"])

        fifth = call_args[4]
        self.assertEqual(fifth["text_tokens"], [])
        self.assertIsNone(fifth["tokens"])
        self.assertEqual(
            fifth["input_token_count"], inner_response.input_token_count
        )

    async def test_token_generation_display_options_combinations(self):
        combos = [
            (False, False, 0),
            (True, False, 1),
            (False, True, 1),
            (True, True, 1),
        ]

        for display_events, display_tools, expected_event_calls in combos:
            args = Namespace(
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

            console = MagicMock()
            live = MagicMock()
            live.__enter__.return_value = live
            live.__exit__.return_value = False

            orchestrator = SimpleNamespace(event_manager=MagicMock())
            lm = SimpleNamespace(model_id="m", tokenizer_config=None)

            with (
                patch.object(model_cmds, "Live", return_value=live),
                patch.object(
                    model_cmds, "_token_stream", new=AsyncMock()
                ) as ts_patch,
                patch.object(
                    model_cmds, "_event_stream", new=AsyncMock()
                ) as es_patch,
            ):
                await model_cmds.token_generation(
                    args=args,
                    console=console,
                    theme=MagicMock(),
                    logger=MagicMock(),
                    orchestrator=orchestrator,
                    event_stats=None,
                    lm=lm,
                    input_string="i",
                    response=MagicMock(),
                    display_tokens=0,
                    dtokens_pick=0,
                    tool_events_limit=None,
                    refresh_per_second=2,
                    with_stats=True,
                )

            ts_patch.assert_awaited_once()
            if expected_event_calls:
                es_patch.assert_awaited_once()
            else:
                es_patch.assert_not_called()


class CliModelRunTestCase(IsolatedAsyncioTestCase):
    async def test_returns_when_no_input(self):
        args = Namespace(
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

    async def test_run_local_model(self):
        args = Namespace(
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

    async def test_run_remote_model(self):
        args = Namespace(
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
                lm.assert_not_called()
                tg_patch.assert_not_called()
                self.assertEqual(
                    console.print.call_args.args[0],
                    "Audio generated in gen.wav",
                )

    async def test_run_audio_generation(self):
        args = Namespace(
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
        lm.assert_not_called()
        tg_patch.assert_not_called()
        self.assertEqual(
            console.print.call_args.args[0],
            "Audio generated in gen.wav",
        )

    async def test_run_audio_speech_recognition(self):
        args = Namespace(
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
        lm.assert_not_called()
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "transcript")

    async def test_run_audio_classification(self):
        args = Namespace(
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
        operation = manager.await_args_list[0].args[3]
        self.assertEqual(operation.parameters["audio"].path, "audio.wav")
        self.assertEqual(operation.parameters["audio"].sampling_rate, 16_000)
        theme.display_audio_labels.assert_called_once_with(
            manager.return_value
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args_list[-1].args[0], "table")

    async def test_run_text_question_answering(self):
        args = Namespace(
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
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "answer")

    async def test_run_text_token_classification(self):
        args = Namespace(
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        )
        tg_patch.assert_not_called()
        theme.display_token_labels.assert_called_once_with([lm.return_value])
        self.assertEqual(console.print.call_args.args[0], "table")

    async def test_run_text_token_classification_labeled_only(self):
        args = Namespace(
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        )
        tg_patch.assert_not_called()
        theme.display_token_labels.assert_called_once_with([lm.return_value])
        self.assertEqual(console.print.call_args.args[0], "table")

    async def test_run_text_sequence_classification(self):
        args = Namespace(
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        lm.assert_awaited_once_with("img.png", skip_special_tokens=False)
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "caption")

    async def test_run_vision_encoder_decoder(self):
        args = Namespace(
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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

    async def test_run_vision_image_text_to_text(self):
        args = Namespace(
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
            settings=ANY,
            width=42,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "txt")

    async def test_run_vision_text_to_image(self):
        args = Namespace(
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
            vision_reference_path=None,
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
            "out.mp4",
            reference_path=None,
            negative_prompt=None,
            height=None,
            downscale=2 / 3,
            frames=96,
            denoise_strength=0.4,
            inference_steps=10,
            decode_timestep=0.05,
            noise_scale=0.025,
            frames_per_second=24,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "out.mp4")

    async def test_run_text_to_video_custom_options(self):
        args = Namespace(
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
            "custom.mp4",
            reference_path="ref.png",
            negative_prompt="neg",
            height=256,
            downscale=0.5,
            frames=50,
            denoise_strength=0.2,
            inference_steps=5,
            decode_timestep=0.1,
            noise_scale=0.01,
            frames_per_second=12,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "custom.mp4")

    async def test_run_text_to_video_width_only(self):
        args = Namespace(
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
            vision_reference_path=None,
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
            "wide.mp4",
            reference_path=None,
            negative_prompt=None,
            height=None,
            width=320,
            downscale=2 / 3,
            frames=96,
            denoise_strength=0.4,
            inference_steps=10,
            decode_timestep=0.05,
            noise_scale=0.025,
            frames_per_second=24,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "wide.mp4")

    async def test_run_text_to_video_steps_only(self):
        args = Namespace(
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
            vision_reference_path=None,
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
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def call_side_effect(engine_uri, modality, model, operation):
            return await RealModelManager.__call__(
                manager, engine_uri, modality, model, operation
            )

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
            "steps.mp4",
            reference_path=None,
            negative_prompt=None,
            height=None,
            downscale=2 / 3,
            frames=96,
            denoise_strength=0.4,
            steps=7,
            inference_steps=10,
            decode_timestep=0.05,
            noise_scale=0.025,
            frames_per_second=24,
        )
        tg_patch.assert_not_called()
        self.assertEqual(console.print.call_args.args[0], "steps.mp4")

    async def test_run_invalid_modality_raises(self):
        args = Namespace(
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
        args = Namespace(filter="f", search="q", limit=2)

        console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm

        theme = MagicMock()
        theme._ = lambda s: s
        theme.get_spinner.return_value = "sp"
        theme.model.side_effect = (
            lambda m, **kw: f"{m.id}-{kw.get('can_access')}"
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
        hub.models.assert_called_once_with(filter="f", search="q", limit=2)
        hub.can_access.assert_has_calls(
            [call("m1"), call("m2")], any_order=True
        )
        # Initial render without access info
        self.assertIn(("m1-None", "m2-None"), groups)
        # Final update includes access results
        self.assertIn(("m1-True", "m2-False"), updates)
        self.assertEqual(live.update.call_count, 2)


class CliModelInternalTestCase(IsolatedAsyncioTestCase):
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

        args = Namespace(display_events=True, display_tools=True, record=False)
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

        args = Namespace(display_events=True, display_tools=True, record=False)
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
            display_events=False, display_tools=False, record=False
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
            display_events=True, display_tools=False, record=False
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
            display_events=False, display_tools=True, record=False
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

    async def test_event_stream_skip_unselected_tool(self):
        orchestrator = SimpleNamespace(event_manager=EventManager())
        group = SimpleNamespace(renderables=[MagicMock(), MagicMock()])
        theme = MagicMock()
        live = MagicMock()
        stop_signal = asyncio.Event()
        console = MagicMock()

        args = Namespace(
            display_events=True, display_tools=False, record=False
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
            yield model_cmds.Token(id=1, token="A")

        class Resp:
            input_token_count = 1

            def __aiter__(self):
                return token_gen()

        async def fake_frames(*_, **__):
            yield (model_cmds.Token(id=1, token="A"), "frame1")
            yield (None, "frame2")
            yield (None, "frame3")

        args = Namespace(
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
        theme.tokens = MagicMock(side_effect=fake_frames)

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
        self.assertIn("frame1", group.renderables.calls)
        self.assertIn("frame2", group.renderables.calls)
        self.assertIn("frame3", group.renderables.calls)
        live.refresh.assert_called()
        theme.tokens.assert_called_once()

    async def test_token_stream_pause_no_probabilities(self):
        async def token_gen():
            yield model_cmds.Token(id=1, token="A")

        class Resp:
            input_token_count = 1

            def __aiter__(self):
                return token_gen()

        async def fake_frames(*_, **__):
            yield (None, "frame1")

        args = Namespace(
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
        theme.tokens = MagicMock(side_effect=fake_frames)

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


class CliRecordOptionTestCase(IsolatedAsyncioTestCase):
    async def test_token_generation_record_enables_screen(self):
        args = Namespace(
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

        lm = SimpleNamespace(model_id="m", tokenizer_config=None)

        with (
            patch.object(model_cmds, "Live", return_value=live) as live_patch,
            patch.object(
                model_cmds, "_token_stream", new=AsyncMock()
            ) as ts_patch,
        ):
            await model_cmds.token_generation(
                args=args,
                console=console,
                theme=MagicMock(),
                logger=MagicMock(),
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="text",
                response=MagicMock(),
                display_tokens=0,
                dtokens_pick=0,
                tool_events_limit=None,
                refresh_per_second=3,
                with_stats=True,
            )

        live_patch.assert_called_once_with(refresh_per_second=3, screen=True)
        ts_patch.assert_awaited_once()


class CliRenderFrameTestCase(IsolatedAsyncioTestCase):
    def test_render_frame_saves_svg_when_recording(self):
        args = Namespace(record=True)
        console = MagicMock()
        live = MagicMock()
        dt_value = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)

        with patch.object(model_cmds, "datetime") as dt_patch:
            dt_patch.now.return_value = dt_value
            model_cmds._render_frame(args, console, live, "frame")

        expected = "avalan-screenshot-20240102030405-123.svg"
        console.save_svg.assert_called_once_with(expected, clear=True)
        live.update.assert_called_once_with("frame")

    def test_render_frame_no_record(self):
        args = Namespace(record=False)
        console = MagicMock()
        live = MagicMock()

        model_cmds._render_frame(args, console, live, "frame")

        console.save_svg.assert_not_called()
        live.update.assert_called_once_with("frame")

    async def test_token_stream_second_frames_pause(self):
        async def token_gen():
            yield model_cmds.Token(id=1, token="A")

        class Resp:
            input_token_count = 1

            def __aiter__(self):
                return token_gen()

        async def fake_frames(*_, **__):
            yield (model_cmds.Token(id=1, token="A"), "frame1")
            yield (model_cmds.Token(id=2, token="B"), "frame2")

        args = Namespace(
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
        theme.tokens = MagicMock(side_effect=fake_frames)

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


class CliReasoningTokenTestCase(IsolatedAsyncioTestCase):
    async def test_reasoning_token_tracked(self):
        class Resp:
            input_token_count = 1

            def __aiter__(self):
                async def gen():
                    yield model_cmds.ReasoningToken(token="A")
                    yield "B"

                return gen()

        args = Namespace(
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
            captured.append({"thinking": list(p[7]), "answer": list(p[9])})
            yield (None, "frame")

        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=fake_tokens)

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        with patch.object(model_cmds, "Live", return_value=live):
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

            def __aiter__(self):
                async def gen():
                    yield model_cmds.ToolCallToken(token="TOOL")
                    yield model_cmds.Token(id=1, token="A")

                return gen()

        args = Namespace(
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
            captured.append(list(p[8]))
            yield (None, "frame")

        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=fake_tokens)

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
        self.assertEqual(captured, [["TOOL"], ["TOOL"]])


class CliModelMixedTokensTestCase(IsolatedAsyncioTestCase):
    async def test_model_run_mixed_tokens(self):
        async def complex_generator():
            rp = ReasoningParser()
            tm = MagicMock()
            tm.is_potential_tool_call.return_value = True
            tm.get_calls.return_value = None
            tp = ToolCallParser(tm, None)
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
                        if isinstance(p, str):
                            if p == "</think>":
                                yield TokenDetail(
                                    id=3, token=p, probability=0.5
                                )
                            elif p in {"X", "Y"}:
                                yield Token(id=1, token=p)
                            elif p == "<think>" or p == "Z":
                                yield p
                        elif isinstance(p, ToolCallToken):
                            if p.token == "</tool_call>":
                                yield TokenDetail(
                                    id=4, token=p.token, probability=0.5
                                )
                            else:
                                yield p
                        else:
                            yield p

        response = TextGenerationResponse(
            lambda: complex_generator(), use_async_generator=True
        )

        args = Namespace(
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
        resp_obj = tg_patch.await_args.kwargs["response"]
        tokens = []
        async for t in resp_obj:
            tokens.append(t)

        self.assertEqual(
            len(
                [t for t in tokens if isinstance(t, model_cmds.ReasoningToken)]
            ),
            4,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, ToolCallToken)]),
            3,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, TokenDetail)]),
            1,
        )
        self.assertGreaterEqual(
            len([t for t in tokens if type(t) is Token]),
            2,
        )
        self.assertEqual(len([t for t in tokens if isinstance(t, str)]), 1)


if __name__ == "__main__":
    main()
