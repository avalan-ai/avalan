from avalan.cli.commands import model as model_cmds
from avalan.cli.commands import get_model_settings
from types import SimpleNamespace
from argparse import Namespace
from unittest.mock import MagicMock, AsyncMock, patch, call
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
            tool_events=2
        )

        result = get_model_settings(args, self.hub, self.logger, engine_uri)
        expected = {
            "engine_uri": engine_uri,
            "attention": "flash",
            "device": "cpu",
            "disable_loading_progress_bar": True,
            "is_sentence_transformer": True,
            "loader_class": "auto",
            "low_cpu_mem_usage": True,
            "quiet": False,
            "revision": "rev",
            "special_tokens": ["<s>"],
            "tokenizer": "tok",
            "tokens": ["t"],
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
            model_cmds.model_uninstall(args, self.console, self.theme, self.hub)

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
        manager.parse_uri.return_value = "uri"
        model = SimpleNamespace(config="cfg", tokenizer_config="tok_cfg")
        self.hub.can_access.return_value = True
        self.hub.model.return_value = "hub_model"
        with patch.object(model_cmds, "ModelManager", return_value=manager):
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
            summary=False,
        )


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
            tool_events_limit=2
        )
        console.print.assert_has_calls([call("a", end=""), call("b", end="")])

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
        )

        console = MagicMock()
        console.width = 80
        logger = MagicMock()

        captured: dict[str, list] = {}

        async def fake_tokens(*p, **kw):
            captured["text_tokens"] = list(p[7])
            captured["input_token_count"] = p[9]
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
                tool_events_limit=2
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
        orchestrator = SimpleNamespace(input_token_count=7)

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
                tool_events_limit=2
            )

        self.assertEqual(theme.tokens.call_args[0][9], 5)
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
                tool_events_limit=2
            )

        self.assertEqual(theme.tokens.call_args[0][9], 7)

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
                tool_events_limit=2
            )

        self.assertEqual(theme.tokens.call_args[0][9], 33)
        lm.input_token_count.assert_called_once_with("text")


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

        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri.return_value = engine_uri
        manager.load.return_value = load_cm

        with (
            patch.object(
                model_cmds, "ModelManager", return_value=manager
            ) as mm_patch,
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={"engine_uri": engine_uri},
            ) as gms_patch,
            patch.object(model_cmds, "get_input", return_value=None),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await model_cmds.model_run(args, console, theme, hub, logger)

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(
            args, hub, logger, engine_uri, is_sentence_transformer=False
        )
        manager.load.assert_called_once_with(engine_uri=engine_uri)
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
            tool_events=2
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
        lm = AsyncMock()
        lm.config = MagicMock()
        lm.config.__repr__ = lambda self=None: "cfg"
        lm.return_value = "resp"

        load_cm = MagicMock()
        load_cm.__enter__.return_value = lm
        load_cm.__exit__.return_value = False

        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri.return_value = engine_uri
        manager.load.return_value = load_cm

        with (
            patch.object(
                model_cmds, "ModelManager", return_value=manager
            ) as mm_patch,
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={"engine_uri": engine_uri},
            ) as gms_patch,
            patch.object(model_cmds, "get_input", return_value="hi"),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await model_cmds.model_run(args, console, theme, hub, logger)

        mm_patch.assert_called_once_with(hub, logger)
        manager.parse_uri.assert_called_once_with("id")
        gms_patch.assert_called_once_with(
            args, hub, logger, engine_uri, is_sentence_transformer=False
        )
        manager.load.assert_called_once_with(engine_uri=engine_uri)

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


if __name__ == "__main__":
    main()
