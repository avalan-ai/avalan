from argparse import Namespace
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

from rich.console import Group

from avalan.cli.commands import model as model_cmds
from avalan.entities import Modality


class CliModelInstallTestCase(TestCase):
    def test_model_install_secret_no_override(self):
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
            patch.object(model_cmds.Prompt, "ask") as ask,
            patch.object(model_cmds, "cache_download") as cache_download,
            patch.object(model_cmds, "confirm", return_value=False) as confirm,
        ):
            model_cmds.model_install(
                args, MagicMock(), MagicMock(), MagicMock()
            )
        confirm.assert_called_once()
        ask.assert_not_called()
        secrets.write.assert_not_called()
        cache_download.assert_called_once()


class CliModelSearchTestCase(IsolatedAsyncioTestCase):
    async def test_model_search_updates_access(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            limit=2,
            filter=None,
            search=None,
            library=None,
            author=None,
            gated=False,
            open=False,
            language=None,
            name=None,
            task=None,
            tag=None,
        )
        console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = status_cm
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm

        theme = MagicMock()
        theme.model.side_effect = (
            lambda m, **kw: f"model-{m.id}-{kw.get('can_access')}"
        )
        hub = MagicMock()
        hub.models.return_value = [
            SimpleNamespace(id="a"),
            SimpleNamespace(id="b"),
        ]
        hub.can_access.side_effect = lambda mid: mid == "a"

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False

        async def fake_to_thread(fn, *a, **kw):
            return fn()

        with (
            patch.object(model_cmds, "Live", return_value=live),
            patch.object(model_cmds, "to_thread", fake_to_thread),
        ):
            await model_cmds.model_search(args, console, theme, hub, 10)

        hub.models.assert_called_once()

        rendered = [
            str(r)
            for c in live.update.call_args_list
            for r in c.args[0].renderables
        ]
        self.assertIn("model-a-True", rendered)
        self.assertIn("model-b-False", rendered)


class CliRenderFrameTestCase(TestCase):
    def test_render_frame_group_and_record(self):
        args = Namespace(record=True)
        console = MagicMock()
        live = MagicMock()
        group = Group("x", "y")
        model_cmds._render_frame(args, console, live, "frame", group, 1)
        self.assertEqual(group.renderables[1], "frame")
        live.refresh.assert_called_once()
        console.save_svg.assert_called_once()


class CliModelRunUnsupportedTestCase(IsolatedAsyncioTestCase):
    async def test_model_run_unsupported_modality(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=1,
            quiet=True,
            skip_hub_access_check=False,
            no_repl=True,
            display_tokens=0,
            tool_events=0,
            display_events=False,
            display_tools=False,
            display_tools_events=0,
        )
        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = MagicMock()
        load_cm.__exit__.return_value = False
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri.return_value = engine_uri
        manager.load.return_value = load_cm

        async def fake_call(*_a, **_kw):
            return "out"

        manager.side_effect = fake_call

        operation = SimpleNamespace(
            modality=Modality.EMBEDDING,
            requires_input=False,
            input="",
            parameters={"text": SimpleNamespace(pick_tokens=0)},
        )

        with (
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                return_value=operation,
            ),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.EMBEDDING,
                },
            ),
            patch.object(model_cmds, "has_input", return_value=False),
        ):
            with self.assertRaises(NotImplementedError):
                await model_cmds.model_run(
                    args, MagicMock(), MagicMock(), MagicMock(), 1, MagicMock()
                )
