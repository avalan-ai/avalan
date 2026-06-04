import asyncio
from logging import Logger
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent import Specification
from avalan.entities import (
    Backend,
    EngineUri,
    GenerationSettings,
    Modality,
    Operation,
    OperationParameters,
    OperationTextParameters,
    TransformerEngineSettings,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.model import manager as model_manager
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.hubs.huggingface import HuggingfaceHub
from avalan.model.manager import ModelManager


class ModelManagerExtraTestCase(TestCase):
    def setUp(self):
        self.hub = MagicMock(spec=HuggingfaceHub)
        self.hub.cache_dir = "cache"
        self.logger = MagicMock(spec=Logger)

    def test_parse_uri_invalid_scheme(self):
        manager = ModelManager(self.hub, self.logger)
        with self.assertRaises(ValueError):
            manager.parse_uri("http://openai/gpt-4o")

    def test_parse_uri_params(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://openai/gpt-4o?temperature=0.6&max_new_tokens=8192&backend=mlx"
        )
        self.assertEqual(uri.params["temperature"], 0.6)
        self.assertEqual(uri.params["max_new_tokens"], 8192)
        self.assertEqual(uri.params["backend"], "mlx")

    def test_parse_uri_local_paths(self):
        manager = ModelManager(self.hub, self.logger)

        relative_uri = manager.parse_uri(
            "ai://local/../pyds4/.local/ds4/ds4flash.gguf"
        )
        absolute_uri = manager.parse_uri(
            "ai://local//Users/mariano/Code/ai/pyds4/.local/ds4/ds4flash.gguf"
        )
        encoded_absolute_uri = manager.parse_uri(
            "ai://local/%2FUsers/mariano/Code/ai/some%20dir/model.gguf"
        )

        self.assertEqual(
            relative_uri.model_id, "../pyds4/.local/ds4/ds4flash.gguf"
        )
        self.assertEqual(
            absolute_uri.model_id,
            "/Users/mariano/Code/ai/pyds4/.local/ds4/ds4flash.gguf",
        )
        self.assertEqual(
            encoded_absolute_uri.model_id,
            "/Users/mariano/Code/ai/some dir/model.gguf",
        )

    def test_get_engine_settings_user_password_no_secret(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host="openai",
            port=None,
            user="token",
            password="pass",
            vendor="openai",
            model_id="gpt",
            params={},
        )
        settings = manager.get_engine_settings(uri)
        self.assertIsNone(settings.access_token)

    def test_load_passes_arguments(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            result = manager.load(
                uri,
                base_url="url",
                quiet=True,
                attention="sd",
                trust_remote_code=True,
                backend="mlxlm",
            )
        args = get_mock.call_args.args[1]
        self.assertTrue(args["disable_loading_progress_bar"])
        self.assertEqual(args["base_url"], "url")
        self.assertEqual(args["attention"], "sd")
        self.assertEqual(args["backend"], "mlxlm")
        self.assertTrue(args["trust_remote_code"])
        load_mock.assert_called_once_with(
            uri, get_mock.return_value, Modality.TEXT_GENERATION
        )
        self.assertEqual(result, "model")

    def test_load_backend_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={"backend": "mlx"},
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)
        args = get_mock.call_args.args[1]
        self.assertEqual(args["backend"], Backend.MLXLM)

    def test_get_engine_settings_uses_azure_api_version_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host="openai",
            port=None,
            user=None,
            password=None,
            vendor="openai",
            model_id="deployment",
            params={"azure_api_version": "2025-04-01-preview"},
        )

        settings = manager.get_engine_settings(uri, {})

        self.assertEqual(settings.azure_api_version, "2025-04-01-preview")

    def test_get_engine_settings_rejects_invalid_azure_api_version(self):
        manager = ModelManager(self.hub, self.logger)
        uri = EngineUri(
            host="openai",
            port=None,
            user=None,
            password=None,
            vendor="openai",
            model_id="deployment",
            params={"azure_api_version": 20250401},
        )

        with self.assertRaises(AssertionError):
            manager.get_engine_settings(uri, {})

    def test_backend_ds4_value(self):
        self.assertEqual(Backend("ds4"), Backend.DS4)

    def test_load_ds4_backend_config_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4&ds4_ctx=4096"
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)

        args = get_mock.call_args.args[1]
        self.assertEqual(args["backend"], Backend.DS4)
        self.assertEqual(args["backend_config"], {"ctx_size": 4096})
        self.assertNotIn("ctx_size", args)
        self.assertNotIn("ds4_ctx", args)

    def test_get_engine_settings_uses_ds4_backend_config_from_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4&ds4_ctx=4096"
            "&ds4_native_backend=metal&ds4_native_log=false"
        )

        settings = manager.get_engine_settings(
            uri, settings={"backend": Backend.TRANSFORMERS}
        )

        self.assertEqual(settings.backend, Backend.DS4)
        self.assertEqual(
            settings.backend_config,
            {
                "ctx_size": 4096,
                "native_backend": "metal",
                "native_log": False,
            },
        )

    def test_get_engine_settings_ds4_explicit_config_overrides_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4&ds4_ctx=2048"
        )

        settings = manager.get_engine_settings(
            uri,
            settings={
                "backend_config": {
                    "ctx_size": 4096,
                    "native_backend": "metal",
                },
            },
        )

        self.assertEqual(settings.backend, Backend.DS4)
        self.assertEqual(
            settings.backend_config,
            {"ctx_size": 4096, "native_backend": "metal"},
        )

    def test_load_ds4_explicit_config_overrides_uri(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4"
            "&ds4_ctx=2048&ds4_native_backend=metal"
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(
                uri,
                backend_config={
                    "ctx_size": 4096,
                    "native_backend": "cuda",
                },
            )

        args = get_mock.call_args.args[1]
        self.assertEqual(
            args["backend_config"],
            {"ctx_size": 4096, "native_backend": "cuda"},
        )

    def test_load_ds4_uri_directional_steering_config(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4"
            "&ds4_directional_steering_file=steer.bin"
            "&ds4_directional_steering_attn=0.5"
            "&ds4_directional_steering_ffn=-0.25"
        )
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)

        args = get_mock.call_args.args[1]
        self.assertEqual(
            args["backend_config"],
            {
                "directional_steering_file": "steer.bin",
                "directional_steering_attn": 0.5,
                "directional_steering_ffn": -0.25,
            },
        )

    def test_load_unknown_ds4_uri_key_raises(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri(
            "ai://local/./model.gguf?backend=ds4&ds4_unknown=1"
        )

        with self.assertRaisesRegex(
            ValueError, "Unknown DS4 configuration key 'ds4_unknown'"
        ):
            manager.load(uri)

    def test_load_invalid_ds4_uri_values_raise(self):
        manager = ModelManager(self.hub, self.logger)
        cases = {
            "ds4_ctx=0": "a positive integer",
            "ds4_mtp_draft_tokens=-1": "a non-negative integer",
            "ds4_kv_disk_space_mb=-1": "a non-negative integer",
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                uri = manager.parse_uri(
                    f"ai://local/./model.gguf?backend=ds4&{query}"
                )
                with self.assertRaisesRegex(ValueError, expected):
                    manager.load(uri)

    def test_ds4_backend_config_from_mapping_rejects_invalid_values(self):
        cases = (
            ({"ds4_mtp_margin": "bad"}, "a number"),
            ({"ds4_mtp_margin": -0.1}, "a non-negative number"),
            ({"ds4_mtp_path": ""}, "a non-empty string"),
            ({"ds4_quality": 1}, "a boolean"),
            (
                {"ds4_native_backend": "rocm"},
                "one of auto, cpu, cuda, metal",
            ),
        )

        for mapping, expected in cases:
            with self.subTest(mapping=mapping):
                with self.assertRaisesRegex(ValueError, expected):
                    ModelManager.ds4_backend_config_from_mapping(mapping)

    def test_private_ds4_config_helpers_cover_defensive_paths(self):
        self.assertEqual(
            model_manager._normalize_ds4_backend_config(
                {"ds4_unknown": 1, "ds4_ctx": 2048},
                reject_unknown=False,
                allow_normalized_keys=False,
            ),
            {"ctx_size": 2048},
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unknown DS4 backend configuration key 'unsupported'",
        ):
            model_manager._validate_ds4_config_value(
                "ds4_unsupported",
                "unsupported",
                1,
            )

    def test_load_non_ds4_ignores_ds4_config(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://local/model?backend=mlx&ds4_ctx=4096")
        with (
            patch.object(manager, "get_engine_settings") as get_mock,
            patch.object(manager, "load_engine") as load_mock,
        ):
            get_mock.return_value = TransformerEngineSettings()
            load_mock.return_value = "model"
            manager.load(uri)

        args = get_mock.call_args.args[1]
        self.assertEqual(args["backend"], Backend.MLXLM)
        self.assertIsNone(args["backend_config"])

    def test_load_engine_invalid_modality(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://tok@openai/gpt-4o")
        settings = TransformerEngineSettings()
        with self.assertRaises(NotImplementedError):
            manager.load_engine(uri, settings, "invalid")  # type: ignore[arg-type]

    def test_load_invalid_modality(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://tok@openai/gpt-4o")
        with self.assertRaises(NotImplementedError):
            manager.load(uri, modality="invalid")  # type: ignore[arg-type]

    def test_load_output_hidden_states(self):
        uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        manager = ModelManager(self.hub, self.logger)
        for modality in (Modality.TEXT_GENERATION, Modality.EMBEDDING):
            for value in (True, False, None):
                with self.subTest(modality=modality, value=value):
                    with (
                        patch.object(manager, "get_engine_settings") as ges,
                        patch.object(manager, "load_engine") as le,
                    ):
                        manager._stack.enter_context = MagicMock()
                        ges.return_value = TransformerEngineSettings()
                        le.return_value = "model"
                        result = manager.load(
                            uri, modality=modality, output_hidden_states=value
                        )
                    args = ges.call_args.args[1]
                    if value is None:
                        self.assertNotIn("output_hidden_states", args)
                    else:
                        self.assertEqual(args["output_hidden_states"], value)
                    le.assert_called_once_with(uri, ges.return_value, modality)
                    self.assertEqual(result, "model")

    def test_parse_uri_boolean_params(self):
        manager = ModelManager(self.hub, self.logger)
        uri = manager.parse_uri("ai://openai/gpt-4o?stream=true&debug=false")
        self.assertTrue(uri.params["stream"])
        self.assertFalse(uri.params["debug"])

    def test_async_context_manager(self):
        manager = ModelManager(self.hub, self.logger)

        async def run():
            async with manager as mm:
                self.assertIs(mm, manager)

        asyncio.run(run())


class ModelManagerEventDispatchTestCase(IsolatedAsyncioTestCase):
    async def test_triggers_events_and_closes_stack(self) -> None:
        hub = MagicMock(spec=HuggingfaceHub)
        hub.cache_dir = "cache"
        logger = MagicMock(spec=Logger)
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        manager = ModelManager(hub, logger, event_manager=event_manager)
        manager._stack.aclose = AsyncMock()  # type: ignore[assignment]

        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="model",
            params={},
        )
        operation = Operation(
            generation_settings=GenerationSettings(),
            input="prompt",
            modality=Modality.TEXT_GENERATION,
            parameters=OperationParameters(
                text=OperationTextParameters(
                    system_prompt="system",
                    developer_prompt="dev",
                )
            ),
        )
        expected = object()

        async def handler(
            engine_uri_arg: EngineUri,
            model_arg: object,
            operation_arg: Operation,
            tool_arg: object,
        ) -> object:
            self.assertIs(engine_uri_arg, engine_uri)
            self.assertIs(operation_arg, operation)
            self.assertIsNone(tool_arg)
            return expected

        model = object()
        context = ModelCallContext(
            specification=Specification(role=None, goal=None),
            input=operation.input,
        )
        task = ModelCall(
            engine_uri=engine_uri,
            model=model,
            operation=operation,
            tool=None,
            context=context,
        )
        with patch(
            "avalan.model.manager.ModalityRegistry.get",
            return_value=handler,
        ):
            result = await manager(task)

        self.assertIs(result, expected)
        self.assertEqual(event_manager.trigger.await_count, 2)
        before_event = event_manager.trigger.await_args_list[0].args[0]
        self.assertEqual(
            before_event.type, EventType.MODEL_MANAGER_CALL_BEFORE
        )
        self.assertEqual(
            before_event.payload["modality"], Modality.TEXT_GENERATION
        )
        self.assertIs(before_event.payload["task"], task)
        self.assertIs(before_event.payload["context"], context)
        after_event = event_manager.trigger.await_args_list[1].args[0]
        self.assertEqual(after_event.type, EventType.MODEL_MANAGER_CALL_AFTER)
        self.assertIs(after_event.payload["result"], expected)
        self.assertIs(after_event.payload["task"], task)
        self.assertIs(after_event.payload["context"], context)
        self.assertIsNotNone(after_event.started)
        self.assertIsNotNone(after_event.finished)
        self.assertIsNotNone(after_event.elapsed)

        manager.__exit__(None, None, None)
        await asyncio.sleep(0)
        manager._stack.aclose.assert_awaited_once()

    async def test_exit_with_running_loop_closes_on_interrupt(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        manager = ModelManager(hub, logger)
        manager._stack = AsyncMock()

        manager.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)
        await asyncio.sleep(0)

        manager._stack.aclose.assert_awaited_once()

    async def test_aexit_awaits_pending_close_on_interrupt(self):
        hub = MagicMock(spec=HuggingfaceHub)
        logger = MagicMock(spec=Logger)
        manager = ModelManager(hub, logger)
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        async def close_task() -> None:
            close_started.set()
            await release_close.wait()

        pending = asyncio.create_task(close_task())
        manager._pending_exit_task = pending

        exit_task = asyncio.create_task(
            manager.__aexit__(KeyboardInterrupt, KeyboardInterrupt(), None)
        )

        await close_started.wait()
        await asyncio.sleep(0)
        self.assertFalse(exit_task.done())

        release_close.set()
        result = await exit_task

        self.assertFalse(result)
        self.assertFalse(pending.cancelled())
        self.assertIsNone(manager._pending_exit_task)
