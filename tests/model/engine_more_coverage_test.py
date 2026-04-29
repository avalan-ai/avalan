import asyncio
import importlib.machinery
import logging
import sys
import types
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.entities import EngineSettings, WeightType
from avalan.model.engine import Engine


class MinimalEngine(Engine):
    async def __call__(self, input, **kwargs):
        return "out"

    def _load_model(self):
        raise NotImplementedError


class WeightAndDeviceTestCase(TestCase):
    def test_weight(self) -> None:
        self.assertIs(Engine.weight("f32"), Engine._WEIGHTS["f32"])
        self.assertEqual(
            Engine.weight(cast(WeightType, "unknown")),
            "auto",
        )

    def test_get_default_device_cuda(self) -> None:
        with patch("avalan.model.engine.cuda.is_available", return_value=True):
            self.assertEqual(Engine.get_default_device(), "cuda")

    def test_get_default_device_mps(self) -> None:
        with (
            patch("avalan.model.engine.cuda.is_available", return_value=False),
            patch("avalan.model.engine.mps.is_available", return_value=True),
        ):
            self.assertEqual(Engine.get_default_device(), "mps")

    def test_get_default_device_cpu(self) -> None:
        with (
            patch("avalan.model.engine.cuda.is_available", return_value=False),
            patch("avalan.model.engine.mps.is_available", return_value=False),
        ):
            self.assertEqual(Engine.get_default_device(), "cpu")

    def test_has_module_handles_module_not_found(self) -> None:
        with patch(
            "avalan.model.engine.find_spec",
            side_effect=ModuleNotFoundError("missing parent package"),
        ):
            self.assertFalse(Engine._has_module("mlx.nn"))


class UsesTokenizerPropertyTestCase(TestCase):
    def test_default_property(self) -> None:
        engine = MinimalEngine(
            "id",
            EngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        self.assertFalse(engine.uses_tokenizer)


class PipelineMetadataTestCase(TestCase):
    def test_pipeline_with_non_callable_parameters_skips_metadata(
        self,
    ) -> None:
        class DummyPipeline:
            parameters: dict[str, object] = {}

            def eval(self) -> None:
                pass

        class DummyEngine(Engine):
            def __init__(self) -> None:
                self.fake_model = DummyPipeline()
                super().__init__(
                    "id",
                    EngineSettings(
                        auto_load_model=True,
                        auto_load_tokenizer=False,
                        enable_eval=False,
                    ),
                )

            async def __call__(self, input, **kwargs):
                return "out"

            def _load_model(self):
                return self.fake_model

        with patch("avalan.model.engine.DiffusionPipeline", DummyPipeline):
            engine = DummyEngine()

        self.assertTrue(engine._loaded_model)
        self.assertIsNone(engine.parameter_count)
        self.assertIsNone(engine.parameter_types)


class ContextLevelErrorTestCase(TestCase):
    def test_enter_exit_level_error(self) -> None:
        engine = MinimalEngine(
            "id",
            EngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        engine._exit_stack = AsyncMock()
        engine._transformers_logging_logger = MagicMock(level=logging.ERROR)
        engine._transformers_logging_level = logging.ERROR
        with patch(
            "avalan.model.engine.transformers_logging.set_verbosity_error"
        ) as sve:
            engine.__enter__()
            sve.assert_not_called()
        with patch.object(
            engine._transformers_logging_logger, "setLevel"
        ) as sl:
            engine.__exit__(None, None, None)
            sl.assert_not_called()
        engine._exit_stack.aclose.assert_awaited_once()


class ContextAsyncExitTestCase(IsolatedAsyncioTestCase):
    async def test_exit_with_running_loop(self) -> None:
        engine = MinimalEngine(
            "id",
            EngineSettings(auto_load_model=False, auto_load_tokenizer=False),
        )
        engine._exit_stack = AsyncMock()
        engine.__exit__(None, None, None)
        await asyncio.sleep(0)
        engine._exit_stack.aclose.assert_awaited_once()


class MlxLoadTestCase(TestCase):
    def test_mlx_branch(self) -> None:
        module_nn = types.ModuleType("mlx.nn")
        module_nn.__spec__ = importlib.machinery.ModuleSpec("mlx.nn", None)

        class Module:
            pass

        Module.__module__ = "mlx.nn.layers.base"
        module_nn.Module = Module
        module_mlx = types.ModuleType("mlx")
        module_mlx.nn = module_nn
        module_mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", None)

        def fake_find_spec(name: str):
            if name == "mlx":
                return module_mlx.__spec__
            if name == "mlx.nn":
                return module_nn.__spec__
            return None

        sys_modules = {"mlx": module_mlx, "mlx.nn": module_nn}
        with patch.dict(sys.modules, sys_modules):
            with patch(
                "avalan.model.engine.find_spec", side_effect=fake_find_spec
            ):

                class DummyModel(module_nn.Module):
                    def parameters(self):
                        return []

                    def eval(self):
                        pass

                    def resize_token_embeddings(self, *_):
                        pass

                    def state_dict(self):
                        return {}

                class DummyEngine(Engine):
                    def __init__(self) -> None:
                        self.fake_model = DummyModel()
                        super().__init__(
                            "id",
                            EngineSettings(
                                auto_load_model=True, auto_load_tokenizer=False
                            ),
                        )

                    async def __call__(self, input, **kwargs):
                        return "out"

                    def _load_model(self):
                        return self.fake_model

                engine = DummyEngine()
                self.assertTrue(engine._loaded_model)

    def test_missing_mlx_nn_namespace_is_ignored(self) -> None:
        module_mlx = types.ModuleType("mlx")
        module_mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", None)

        def fake_find_spec(name: str):
            return module_mlx.__spec__ if name == "mlx" else None

        class DummyPipeline:
            def eval(self) -> None:
                pass

            def parameters(self) -> list[object]:
                return []

            def resize_token_embeddings(self, *_: object) -> None:
                pass

        class DummyEngine(Engine):
            def __init__(self) -> None:
                self.fake_model = DummyPipeline()
                super().__init__(
                    "id",
                    EngineSettings(
                        auto_load_model=True, auto_load_tokenizer=False
                    ),
                )

            async def __call__(self, input, **kwargs):
                return "out"

            def _load_model(self):
                return self.fake_model

        with patch.dict(sys.modules, {"mlx": module_mlx}):
            with patch(
                "avalan.model.engine.find_spec", side_effect=fake_find_spec
            ):
                with patch(
                    "avalan.model.engine.DiffusionPipeline", DummyPipeline
                ):
                    engine = DummyEngine()
                    self.assertTrue(engine._loaded_model)

    def test_missing_mlx_parent_package_is_ignored(self) -> None:
        def fake_find_spec(name: str):
            if name == "mlx.nn":
                raise ModuleNotFoundError("No module named 'mlx'")
            return None

        class DummyPipeline:
            def eval(self) -> None:
                pass

            def parameters(self) -> list[object]:
                return []

            def resize_token_embeddings(self, *_: object) -> None:
                pass

        class DummyEngine(Engine):
            def __init__(self) -> None:
                self.fake_model = DummyPipeline()
                super().__init__(
                    "id",
                    EngineSettings(
                        auto_load_model=True, auto_load_tokenizer=False
                    ),
                )

            async def __call__(self, input, **kwargs):
                return "out"

            def _load_model(self):
                return self.fake_model

        with patch(
            "avalan.model.engine.find_spec", side_effect=fake_find_spec
        ):
            with patch("avalan.model.engine.DiffusionPipeline", DummyPipeline):
                engine = DummyEngine()
                self.assertTrue(engine._loaded_model)
