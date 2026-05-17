import asyncio
import json
import os
import threading
from asyncio import run
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from logging import Logger
from math import exp
from pathlib import Path
from threading import Event as ThreadEvent
from threading import get_ident
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import avalan.backends.ds4_native.availability as availability
import avalan.model.nlp.text.ds4 as ds4_module
from avalan.backends.ds4_native import Backend as Ds4NativeBackend
from avalan.backends.ds4_native import Engine as Ds4CompatEngine
from avalan.backends.ds4_native import (
    EngineOptions,
    SamplingOptions,
    ThinkMode,
)
from avalan.backends.ds4_native.errors import (
    Ds4BackendUnavailable,
    Ds4ContextError,
    Ds4GenerationError,
    Ds4InvalidModel,
    Ds4LoadError,
)
from avalan.backends.ds4_native.metadata import (
    DS4_API_COMMIT,
    DS4_API_VERSION,
    DS4_REQUIRED_C_SYMBOLS,
)
from avalan.entities import (
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageContentImage,
    MessageContentText,
    MessageRole,
    MessageToolCall,
    ReasoningEffort,
    ReasoningSettings,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolCallToken,
    ToolFormat,
    ToolManagerSettings,
    TransformerEngineSettings,
)
from avalan.model.nlp.text.ds4 import (
    _CPU_WARNING,
    Ds4Model,
    Ds4Worker,
    _Ds4DiskKvCache,
    _Ds4GenerationPlan,
)
from avalan.tool.dsml import DsmlParseResult, DsmlPromptMessage
from avalan.tool.manager import ToolManager
from avalan.tool.math import MathToolSet


class NativeBackend(StrEnum):
    METAL = "metal"
    CUDA = "cuda"
    CPU = "cpu"


class NativeThinkMode(StrEnum):
    NONE = "none"
    HIGH = "high"
    MAX = "max"


@dataclass(frozen=True, slots=True)
class FakeNativeOptions:
    model_path: str
    backend: NativeBackend
    mtp_path: str | None = None
    n_threads: int = 0
    mtp_draft_tokens: int = 0
    mtp_margin: float = 0.0
    directional_steering_file: str | None = None
    directional_steering_attn: float = 0.0
    directional_steering_ffn: float = 0.0
    warm_weights: bool = False
    quality: bool = False
    native_log: bool = True


@dataclass(frozen=True, slots=True)
class FakeNativeSamplingOptions:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int | None = None


class FakeNativeSession:
    def __init__(self, engine: "FakeNativeEngine", ctx_size: int) -> None:
        self.argmax_calls = 0
        self.close_calls = 0
        self.ctx = ctx_size
        self.engine = engine
        self.eval_calls: list[int] = []
        self.invalidate_calls = 0
        self.load_payload_calls = 0
        self.load_snapshot_calls = 0
        self.loaded_payloads: list[bytes] = []
        self.loaded_snapshots: list[bytes] = []
        self.pos = 0
        self.sample_options: list[FakeNativeSamplingOptions] = []
        self.save_payload_calls = 0
        self.save_snapshot_calls = 0
        self.token_logprob_calls: list[int] = []
        self.tokens: list[int] = []
        self.top_logprobs_calls: list[int] = []

    def close(self) -> None:
        self.engine.thread_ids.append(get_ident())
        self.close_calls += 1

    def sync(self, prompt_tokens: list[int]) -> None:
        self.engine.thread_ids.append(get_ident())
        if self.engine.sync_error is not None:
            raise self.engine.sync_error
        self.engine.session_sync_calls.append(tuple(prompt_tokens))
        self.engine.operation_log.append(("sync", tuple(prompt_tokens)))
        self.tokens = list(prompt_tokens)
        self.pos = len(prompt_tokens)

    def eval(self, token_id: int) -> None:
        self.engine.thread_ids.append(get_ident())
        self.engine.operation_log.append(("eval", token_id))
        if self.engine.fail_on_eval:
            raise RuntimeError("eval failed")
        self.eval_calls.append(token_id)
        self.tokens.append(token_id)
        self.pos += 1

    def argmax(self) -> int:
        self.engine.thread_ids.append(get_ident())
        self.argmax_calls += 1
        self.engine.operation_log.append(("argmax", None))
        if (
            self.engine.block_argmax_after_calls is not None
            and self.argmax_calls > self.engine.block_argmax_after_calls
        ):
            self.engine.argmax_block_started.set()
            self.engine.argmax_release.wait(timeout=2.0)
        if self.engine.argmax_script:
            return self.engine.argmax_script.pop(0)
        return self.engine.eos_token_id

    def sample(self, options: FakeNativeSamplingOptions) -> int:
        self.engine.thread_ids.append(get_ident())
        self.engine.operation_log.append(("sample", options))
        self.sample_options.append(options)
        if self.engine.sample_script:
            return self.engine.sample_script.pop(0)
        return self.engine.eos_token_id

    def top_logprobs(self, top_k: int) -> list[tuple[int, float]]:
        self.engine.thread_ids.append(get_ident())
        self.engine.operation_log.append(("top_logprobs", top_k))
        self.top_logprobs_calls.append(top_k)
        return self.engine.top_logprobs[:top_k]

    def token_logprob(self, token_id: int) -> float:
        self.engine.thread_ids.append(get_ident())
        self.engine.operation_log.append(("token_logprob", token_id))
        self.token_logprob_calls.append(token_id)
        return self.engine.token_logprobs.get(token_id, -20.0)

    def invalidate(self) -> None:
        self.engine.thread_ids.append(get_ident())
        self.invalidate_calls += 1
        self.engine.invalidate_event.set()

    def save_snapshot(self) -> bytes:
        self.engine.thread_ids.append(get_ident())
        self.save_snapshot_calls += 1
        token_bytes = ",".join(str(token_id) for token_id in self.tokens)
        return f"ctx={self.ctx};tokens={token_bytes}".encode()

    def load_snapshot(self, snapshot: bytes) -> None:
        self.engine.thread_ids.append(get_ident())
        self.load_snapshot_calls += 1
        self.loaded_snapshots.append(snapshot)
        if snapshot == b"corrupt":
            raise RuntimeError("corrupt snapshot")
        prefix = f"ctx={self.ctx};tokens=".encode()
        if not snapshot.startswith(prefix):
            raise RuntimeError("snapshot context mismatch")
        token_text = snapshot[len(prefix) :].decode()
        self.tokens = (
            [int(token_id) for token_id in token_text.split(",")]
            if token_text
            else []
        )
        self.pos = len(self.tokens)
        self.engine.snapshot_loaded_event.set()

    def save_payload(self) -> bytes:
        self.engine.thread_ids.append(get_ident())
        self.save_payload_calls += 1
        return b"payload:" + self.save_snapshot()

    def load_payload(self, payload: bytes) -> None:
        self.engine.thread_ids.append(get_ident())
        self.load_payload_calls += 1
        self.loaded_payloads.append(payload)
        if not payload.startswith(b"payload:"):
            raise RuntimeError("corrupt payload")
        self.load_snapshot(payload.removeprefix(b"payload:"))


class FakeNativeEngine:
    instances: list["FakeNativeEngine"] = []

    def __init__(self, options: FakeNativeOptions) -> None:
        self.argmax_script: list[int] = []
        self.argmax_block_started = ThreadEvent()
        self.argmax_release = ThreadEvent()
        self.block_argmax_after_calls: int | None = None
        self.close_calls = 0
        self.eos_token_id = 100001
        self.fail_token_text_ids: set[int] = set()
        self.fail_on_eval = False
        self.invalidate_event = ThreadEvent()
        self.operation_log: list[tuple[str, object]] = []
        self.options = options
        self.sample_script: list[int] = []
        self.session_sync_calls: list[tuple[int, ...]] = []
        self.sessions: list[FakeNativeSession] = []
        self.snapshot_loaded_event = ThreadEvent()
        self.sync_error: BaseException | None = None
        self.thread_ids: list[int] = []
        self.token_logprobs: dict[int, float] = {}
        self.token_text_map: dict[int, bytes] = {}
        self.tokenization_calls: list[tuple[str, object, object]] = []
        self.tokenize_rendered_chat_calls: list[str] = []
        self.top_logprobs: list[tuple[int, float]] = []
        self.instances.append(self)

    def close(self) -> None:
        self.thread_ids.append(get_ident())
        self.close_calls += 1

    def create_session(self, ctx_size: int) -> FakeNativeSession:
        self.thread_ids.append(get_ident())
        session = FakeNativeSession(self, ctx_size)
        self.sessions.append(session)
        return session

    def token_text(self, token_id: int) -> bytes:
        self.thread_ids.append(get_ident())
        self.operation_log.append(("token_text", token_id))
        if token_id in self.fail_token_text_ids:
            raise RuntimeError("token text failed")
        return self.token_text_map.get(token_id, f"<{token_id}>".encode())

    def chat_begin(self) -> list[int]:
        self.thread_ids.append(get_ident())
        self.tokenization_calls.append(("chat_begin", None, None))
        return [1]

    def chat_append_message(
        self, tokens: list[int], role: str, content: str
    ) -> None:
        self.thread_ids.append(get_ident())
        self.tokenization_calls.append((role, content, tuple(tokens)))
        role_token = {"system": 10, "user": 11, "assistant": 12}[role]
        tokens.extend([role_token, len(content)])

    def chat_append_assistant_prefix(
        self, tokens: list[int], think_mode: NativeThinkMode
    ) -> None:
        self.thread_ids.append(get_ident())
        self.tokenization_calls.append(
            (
                "assistant_prefix",
                think_mode,
                tuple(tokens),
            )
        )
        prefix_token = {
            NativeThinkMode.NONE: 20,
            NativeThinkMode.HIGH: 21,
            NativeThinkMode.MAX: 22,
        }[think_mode]
        tokens.append(prefix_token)

    def encode_chat_prompt(
        self,
        system: str | None,
        prompt: str,
        think_mode: NativeThinkMode,
    ) -> list[int]:
        self.thread_ids.append(get_ident())
        self.tokenization_calls.append(
            (
                "encode_chat_prompt",
                (system, prompt),
                think_mode,
            )
        )
        prefix_token = {
            NativeThinkMode.NONE: 20,
            NativeThinkMode.HIGH: 21,
            NativeThinkMode.MAX: 22,
        }[think_mode]
        return [30, len(system or ""), len(prompt), prefix_token]

    def tokenize_rendered_chat(self, text: str) -> list[int]:
        self.thread_ids.append(get_ident())
        self.tokenize_rendered_chat_calls.append(text)
        return [40, len(text)]


FakeNativeEngine.__module__ = "pyds4"


def _fake_async_engine_type(
    native_engine_type: type[Any],
    *,
    logprobs: bool = False,
    payloads: bool = False,
    snapshots: bool = False,
) -> type[Any]:
    class FakeAsyncSession:
        session_closed: bool

        def __init__(
            self,
            owner: "FakeAsyncEngine",
            native: FakeNativeSession,
        ) -> None:
            self._native = native
            self._owner = owner
            self.session_closed = False

        async def sync(self, prompt_tokens: list[int]) -> None:
            await self._owner._call(lambda: self._native.sync(prompt_tokens))

        async def eval(self, token_id: int) -> None:
            await self._owner._call(lambda: self._native.eval(token_id))

        async def argmax(self) -> int:
            return cast(int, await self._owner._call(self._native.argmax))

        async def sample(self, options: FakeNativeSamplingOptions) -> int:
            return cast(
                int,
                await self._owner._call(lambda: self._native.sample(options)),
            )

        async def next_token(
            self,
            options: FakeNativeSamplingOptions | None = None,
            *,
            advance: bool = True,
            decode: bool = False,
            stop_on_eos: bool = True,
            exclude_token_id: int | None = None,
        ) -> SimpleNamespace:
            def step() -> SimpleNamespace:
                if options is not None and exclude_token_id is not None:
                    raise ValueError(
                        "exclude_token_id cannot be used with sampling "
                        "options."
                    )
                if options is not None:
                    token_id = self._native.sample(options)
                elif exclude_token_id is not None:
                    token_id = self._native.argmax()
                else:
                    token_id = self._native.argmax()

                is_eos = token_id == self._owner._native.eos_token_id
                should_advance = advance and not (stop_on_eos and is_eos)
                if should_advance:
                    try:
                        self._native.eval(token_id)
                    except RuntimeError as error:
                        raise Ds4GenerationError(
                            f"eval failed: {error}"
                        ) from error
                token_bytes = (
                    self._token_text(token_id)
                    if decode and not (stop_on_eos and is_eos)
                    else None
                )
                return SimpleNamespace(
                    token_id=token_id,
                    is_eos=is_eos,
                    advanced=should_advance,
                    token_bytes=token_bytes,
                )

            return cast(SimpleNamespace, await self._owner._call(step))

        def _token_text(self, token_id: int) -> bytes:
            try:
                return cast(bytes, self._owner._native.token_text(token_id))
            except RuntimeError as error:
                raise Ds4GenerationError(
                    f"token_text failed: {error}"
                ) from error

        async def invalidate(self) -> None:
            await self._owner._call(self._native.invalidate)

        async def aclose(self) -> None:
            if self.session_closed:
                return
            self.session_closed = True
            await self._owner._call(self._native.close)

    if logprobs:

        async def top_logprobs(
            self: Any, top_k: int
        ) -> list[tuple[int, float]]:
            return cast(
                list[tuple[int, float]],
                await self._owner._call(
                    lambda: self._native.top_logprobs(top_k)
                ),
            )

        async def token_logprob(self: Any, token_id: int) -> float:
            return cast(
                float,
                await self._owner._call(
                    lambda: self._native.token_logprob(token_id)
                ),
            )

        FakeAsyncSession.top_logprobs = top_logprobs
        FakeAsyncSession.token_logprob = token_logprob

    if snapshots:

        async def save_snapshot(self: Any) -> bytes:
            return cast(
                bytes, await self._owner._call(self._native.save_snapshot)
            )

        async def load_snapshot(self: Any, snapshot: bytes) -> None:
            await self._owner._call(
                lambda: self._native.load_snapshot(snapshot)
            )

        FakeAsyncSession.save_snapshot = save_snapshot
        FakeAsyncSession.load_snapshot = load_snapshot

    if payloads:

        async def save_payload(self: Any) -> bytes:
            return cast(
                bytes, await self._owner._call(self._native.save_payload)
            )

        async def load_payload(self: Any, payload: bytes) -> None:
            await self._owner._call(lambda: self._native.load_payload(payload))

        FakeAsyncSession.save_payload = save_payload
        FakeAsyncSession.load_payload = load_payload

    class FakeAsyncEngine:
        engine_closed: bool

        def __init__(self, options: FakeNativeOptions) -> None:
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._native = native_engine_type(options)
            self.engine_closed = False

        @classmethod
        async def open(cls, options: FakeNativeOptions) -> "FakeAsyncEngine":
            return cls(options)

        @property
        def eos_token_id(self) -> int:
            return self._native.eos_token_id

        async def _call(self, func: Callable[[], object]) -> object:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, func)

        async def create_session(self, ctx_size: int) -> FakeAsyncSession:
            native_session = await self._call(
                lambda: self._native.create_session(ctx_size)
            )
            return FakeAsyncSession(
                self, cast(FakeNativeSession, native_session)
            )

        async def token_text(self, token_id: int) -> bytes:
            return cast(
                bytes,
                await self._call(lambda: self._native.token_text(token_id)),
            )

        async def chat_begin(self) -> list[int]:
            return cast(list[int], await self._call(self._native.chat_begin))

        async def chat_append_message(
            self, tokens: list[int], role: str, content: str
        ) -> None:
            await self._call(
                lambda: self._native.chat_append_message(tokens, role, content)
            )

        async def chat_append_assistant_prefix(
            self, tokens: list[int], think_mode: NativeThinkMode
        ) -> None:
            await self._call(
                lambda: self._native.chat_append_assistant_prefix(
                    tokens, think_mode
                )
            )

        async def encode_chat_prompt(
            self,
            system: str | None,
            prompt: str,
            think_mode: NativeThinkMode,
        ) -> list[int]:
            return cast(
                list[int],
                await self._call(
                    lambda: self._native.encode_chat_prompt(
                        system, prompt, think_mode
                    )
                ),
            )

        async def tokenize_rendered_chat(self, text: str) -> list[int]:
            return cast(
                list[int],
                await self._call(
                    lambda: self._native.tokenize_rendered_chat(text)
                ),
            )

        async def aclose(self) -> None:
            if self.engine_closed:
                return
            self.engine_closed = True
            try:
                await self._call(self._native.close)
            finally:
                self._executor.shutdown(wait=True)

    FakeAsyncEngine.__module__ = "pyds4"
    return FakeAsyncEngine


def _fake_binding(**overrides: object) -> SimpleNamespace:
    native_engine_type = cast(
        type[Any], overrides.get("Engine", FakeNativeEngine)
    )
    values: dict[str, object] = {
        "__ds4_commit__": DS4_API_COMMIT,
        "__ds4_symbols__": DS4_REQUIRED_C_SYMBOLS,
        "AsyncEngine": overrides.get(
            "AsyncEngine", _fake_async_engine_type(native_engine_type)
        ),
        "Backend": NativeBackend,
        "Engine": native_engine_type,
        "EngineOptions": FakeNativeOptions,
        "SamplingOptions": FakeNativeSamplingOptions,
        "ThinkMode": NativeThinkMode,
        "is_backend_available": (
            lambda backend: backend in {"metal", "cuda", "cpu"}
        ),
        "think_mode_for_context": lambda mode, ctx_size: mode,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_capabilities(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "available_backends": ("metal", "cuda", "cpu"),
        "backend": "metal",
        "ds4_api_version": DS4_API_VERSION,
        "ds4_commit": DS4_API_COMMIT,
        "logprobs": False,
        "mtp": True,
        "payloads": False,
        "progress": True,
        "required_symbols": DS4_REQUIRED_C_SYMBOLS,
        "snapshots": False,
        "speculative_eval": False,
        "top_logprobs": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_binding(monkeypatch: pytest.MonkeyPatch, binding: object) -> None:
    monkeypatch.setattr(availability, "import_module", lambda _: binding)


def _model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    return model_path


def _worker_options(model_path: Path) -> EngineOptions:
    return EngineOptions(
        model_path=str(model_path), backend=Ds4NativeBackend.METAL
    )


def _latest_fake_engine() -> FakeNativeEngine:
    return FakeNativeEngine.instances[-1]


def test_ds4_model_is_available_with_fake_safe_pyds4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())

    assert Ds4Model.is_available() is True
    assert FakeNativeEngine.instances == []


def test_ds4_model_is_unavailable_when_binding_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_: str) -> object:
        raise ModuleNotFoundError("No module named 'pyds4'")

    monkeypatch.setattr(availability, "import_module", fail_import)

    assert Ds4Model.is_available() is False


def test_ds4_model_unsafe_import_is_unavailable_and_load_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding(__ds4_import_safe__=False))
    model_path = _model_file(tmp_path)

    assert Ds4Model.is_available() is False
    with pytest.raises(Ds4BackendUnavailable, match=r"avalan\[ds4\]"):
        Ds4Model(str(model_path))


def test_ds4_model_construction_does_not_load_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_tokenizer(*_: object, **__: object) -> object:
        raise AssertionError("DS4 construction must not load a tokenizer")

    _install_binding(monkeypatch, _fake_binding())
    monkeypatch.setattr(
        Ds4Model, "_load_tokenizer_with_tokens", fail_tokenizer
    )
    model_path = _model_file(tmp_path)

    model = Ds4Model(
        str(model_path),
        TransformerEngineSettings(backend_config={"native_backend": "metal"}),
    )

    assert model.uses_tokenizer is False
    assert model.tokenizer is None
    model.close()


def test_ds4_model_load_passes_normalized_engine_options_to_pyds4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    mtp_path = tmp_path / "mtp.gguf"
    mtp_path.write_bytes(b"mtp")
    steering_path = tmp_path / "steer.bin"
    steering_path.write_bytes(b"steer")

    model = Ds4Model(
        str(model_path),
        TransformerEngineSettings(
            device="cuda",
            backend_config={
                "native_backend": "cuda",
                "mtp_path": str(mtp_path),
                "mtp_draft_tokens": 2,
                "mtp_margin": 0.25,
                "directional_steering_file": str(steering_path),
                "directional_steering_attn": 0.5,
                "directional_steering_ffn": -0.25,
                "warm_weights": True,
                "quality": True,
                "native_log": False,
            },
        ),
    )

    native = _latest_fake_engine()
    assert native is FakeNativeEngine.instances[0]
    assert native.options == FakeNativeOptions(
        model_path=str(model_path),
        backend=NativeBackend.CUDA,
        mtp_path=str(mtp_path),
        mtp_draft_tokens=2,
        mtp_margin=0.25,
        directional_steering_file=str(steering_path),
        directional_steering_attn=0.5,
        directional_steering_ffn=-0.25,
        warm_weights=True,
        quality=True,
        native_log=False,
    )
    model.close()


def test_ds4_model_load_accepts_complete_pyds4_capabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(
        monkeypatch,
        _fake_binding(
            capabilities=lambda: _fake_capabilities(
                logprobs=True,
                payloads=True,
                snapshots=True,
                speculative_eval=True,
                top_logprobs=True,
            )
        ),
    )

    model = Ds4Model(str(_model_file(tmp_path)))

    assert len(FakeNativeEngine.instances) == 1
    model.close()


def test_ds4_model_can_suppress_native_engine_open_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    base_engine = _fake_async_engine_type(FakeNativeEngine)

    class NoisyAsyncEngine(base_engine):
        @classmethod
        async def open(cls, options: FakeNativeOptions) -> "NoisyAsyncEngine":
            if options.native_log:
                os.write(2, b"native debug line\n")
            return cls(options)

    NoisyAsyncEngine.__module__ = "pyds4"
    _install_binding(
        monkeypatch,
        _fake_binding(AsyncEngine=NoisyAsyncEngine),
    )

    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(backend_config={"native_log": False}),
    )

    model.close()
    captured = capfd.readouterr()
    assert "native debug line" not in captured.err


def test_ds4_model_suppresses_native_engine_open_stderr_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    base_engine = _fake_async_engine_type(FakeNativeEngine)

    class NoisyAsyncEngine(base_engine):
        @classmethod
        async def open(cls, options: FakeNativeOptions) -> "NoisyAsyncEngine":
            if options.native_log:
                os.write(2, b"native debug line\n")
            return cls(options)

    NoisyAsyncEngine.__module__ = "pyds4"
    _install_binding(
        monkeypatch,
        _fake_binding(AsyncEngine=NoisyAsyncEngine),
    )

    model = Ds4Model(str(_model_file(tmp_path)))

    model.close()
    captured = capfd.readouterr()
    assert "native debug line" not in captured.err


def test_ds4_model_suppresses_native_engine_open_stderr_for_older_pyds4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    @dataclass(frozen=True, slots=True)
    class OldNativeOptions:
        model_path: str
        backend: NativeBackend

    base_engine = _fake_async_engine_type(FakeNativeEngine)

    class NoisyAsyncEngine(base_engine):
        @classmethod
        async def open(cls, options: OldNativeOptions) -> "NoisyAsyncEngine":
            os.write(2, b"native debug line\n")
            return cls(
                FakeNativeOptions(
                    model_path=options.model_path,
                    backend=options.backend,
                )
            )

    NoisyAsyncEngine.__module__ = "pyds4"
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=NoisyAsyncEngine,
            EngineOptions=OldNativeOptions,
        ),
    )

    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(backend_config={"native_log": False}),
    )

    model.close()
    captured = capfd.readouterr()
    assert "native debug line" not in captured.err


def test_ds4_model_replays_native_engine_open_stderr_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    base_engine = _fake_async_engine_type(FakeNativeEngine)

    class NoisyAsyncEngine(base_engine):
        @classmethod
        async def open(cls, options: FakeNativeOptions) -> "NoisyAsyncEngine":
            os.write(2, b"native debug line\n")
            return cls(options)

    NoisyAsyncEngine.__module__ = "pyds4"
    _install_binding(
        monkeypatch,
        _fake_binding(AsyncEngine=NoisyAsyncEngine),
    )

    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(backend_config={"native_log": True}),
    )

    model.close()
    captured = capfd.readouterr()
    assert "native debug line" in captured.err


def test_ds4_model_uses_async_engine_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class SyncEngineShouldNotOpen:
        def __init__(self, _: FakeNativeOptions) -> None:
            raise AssertionError("sync DS4 Engine must not be opened")

    _install_binding(
        monkeypatch,
        _fake_binding(
            Engine=SyncEngineShouldNotOpen,
            AsyncEngine=_fake_async_engine_type(FakeNativeEngine),
        ),
    )
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.token_text_map = {101: b"A"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "A"
    assert fake.sessions[0].eval_calls == [101]
    model.close()


def test_ds4_model_accepts_fake_pyds4_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False),
    )
    native = FakeNativeEngine(
        FakeNativeOptions("native.gguf", NativeBackend.METAL)
    )

    assert model._accepts_loaded_model(native) is True
    assert model._accepts_loaded_model(object()) is False


def test_ds4_model_cpu_backend_emits_warning(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)

    with caplog.at_level("WARNING"):
        model = Ds4Model(
            str(model_path),
            TransformerEngineSettings(
                backend_config={"native_backend": "cpu"}
            ),
        )

    assert _CPU_WARNING in caplog.text
    model.close()


def test_ds4_model_missing_model_path_raises_before_native_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())

    with pytest.raises(Ds4InvalidModel, match="does not exist"):
        Ds4Model(str(tmp_path / "missing.gguf"))

    assert FakeNativeEngine.instances == []


def test_ds4_model_missing_mtp_path_raises_before_native_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)

    with pytest.raises(Ds4InvalidModel, match="DS4 MTP path"):
        Ds4Model(
            str(model_path),
            TransformerEngineSettings(
                backend_config={"mtp_path": str(tmp_path / "missing-mtp.gguf")}
            ),
        )

    assert FakeNativeEngine.instances == []


def test_ds4_model_missing_steering_path_raises_before_native_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)

    with pytest.raises(Ds4InvalidModel, match="directional steering file"):
        Ds4Model(
            str(model_path),
            TransformerEngineSettings(
                backend_config={
                    "directional_steering_file": str(
                        tmp_path / "missing-steer.bin"
                    )
                }
            ),
        )

    assert FakeNativeEngine.instances == []


def test_ds4_model_steering_coefficients_require_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)

    with pytest.raises(Ds4InvalidModel, match="directional_steering_file"):
        Ds4Model(
            str(model_path),
            TransformerEngineSettings(
                backend_config={"directional_steering_attn": 0.25}
            ),
        )

    assert FakeNativeEngine.instances == []


@pytest.mark.parametrize("key", ("mtp_draft_tokens", "mtp_margin"))
def test_ds4_model_invalid_mtp_values_raise_before_native_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)

    with pytest.raises(ValueError, match=key):
        Ds4Model(
            str(model_path),
            TransformerEngineSettings(backend_config={key: -1}),
        )

    assert FakeNativeEngine.instances == []


def test_ds4_model_native_load_failure_propagates_with_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingNativeEngine:
        def __init__(self, _: FakeNativeOptions) -> None:
            raise RuntimeError("native open failed")

    FailingNativeEngine.__module__ = "pyds4"
    _install_binding(monkeypatch, _fake_binding(Engine=FailingNativeEngine))
    model_path = _model_file(tmp_path)

    with pytest.raises(Ds4LoadError, match="native open failed"):
        Ds4Model(
            str(model_path),
            TransformerEngineSettings(
                backend_config={"native_backend": "metal"}
            ),
        )
    assert not any(
        thread.name == "ds4-async-compat" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_ds4_plain_string_input_renders_as_single_user_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    model = Ds4Model(str(model_path))

    response = run(model("hello"))

    fake = _latest_fake_engine()
    assert fake.tokenization_calls == [
        (
            "encode_chat_prompt",
            (None, "hello"),
            NativeThinkMode.NONE,
        )
    ]
    assert response.input_token_count == 4
    model.close()


def test_ds4_system_and_developer_prompts_are_merged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    model = Ds4Model(str(model_path))

    response = run(
        model(
            "hello",
            system_prompt="system text",
            developer_prompt="developer text",
        )
    )

    fake = _latest_fake_engine()
    assert fake.tokenization_calls == [
        (
            "encode_chat_prompt",
            (
                "system text\n\nDeveloper instructions:\ndeveloper text",
                "hello",
            ),
            NativeThinkMode.NONE,
        )
    ]
    assert response.input_token_count == 4
    model.close()


def test_ds4_multi_turn_input_preserves_user_assistant_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    model = Ds4Model(str(model_path))
    messages = [
        Message(role=MessageRole.USER, content="one"),
        Message(role=MessageRole.ASSISTANT, content="two"),
        Message(role=MessageRole.USER, content="three"),
    ]

    response = run(model(messages))

    fake = _latest_fake_engine()
    assert fake.tokenization_calls == [
        ("chat_begin", None, None),
        ("user", "one", (1,)),
        ("assistant", "two", (1, 11, 3)),
        ("user", "three", (1, 11, 3, 12, 3)),
        (
            "assistant_prefix",
            NativeThinkMode.NONE,
            (1, 11, 3, 12, 3, 11, 5),
        ),
    ]
    assert response.input_token_count == 8
    model.close()


def test_ds4_input_token_count_counts_rendered_token_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    model = Ds4Model(str(model_path))

    count = model.input_token_count("hello")

    assert count == 4
    model.close()


def test_ds4_tool_manager_renders_native_dsml_tool_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    model = Ds4Model(str(model_path))
    manager = ToolManager.create_instance(
        available_toolsets=[MathToolSet(namespace="math")]
    )

    response = run(
        model(
            "hello",
            tool=manager,
            settings=GenerationSettings(max_new_tokens=0),
        )
    )

    assert run(response.to_str()) == ""
    fake = _latest_fake_engine()
    rendered = fake.tokenize_rendered_chat_calls[-1]
    assert "## Tools" in rendered
    assert '"name":"math.calculator"' in rendered
    assert "<｜User｜>hello<｜Assistant｜></think>" in rendered
    assert response.input_token_count == 2
    model.close()


def test_ds4_tool_role_history_renders_dsml_tool_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    model = Ds4Model(str(model_path))

    response = run(
        model(
            [
                Message(role=MessageRole.USER, content="calculate"),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        MessageToolCall(
                            id="call_1",
                            name="math.calculator",
                            arguments={"expression": "2 + 2"},
                        )
                    ],
                ),
                Message(role=MessageRole.TOOL, content="4"),
            ],
            settings=GenerationSettings(max_new_tokens=0),
        )
    )

    assert run(response.to_str()) == ""
    rendered = _latest_fake_engine().tokenize_rendered_chat_calls[-1]
    assert '<｜DSML｜invoke name="math.calculator">' in rendered
    assert (
        '<｜DSML｜parameter name="expression" string="true">2 + 2'
        "</｜DSML｜parameter>"
        in rendered
    )
    assert "<tool_result>4</tool_result>" in rendered
    model.close()


def test_ds4_empty_prompt_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    model = Ds4Model(str(model_path))

    with pytest.raises(ValueError, match="non-empty text input"):
        run(model(""))

    model.close()


def test_ds4_unsupported_input_shape_raises_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model_path = _model_file(tmp_path)
    model = Ds4Model(str(model_path))

    with pytest.raises(ValueError):
        run(model(cast(Any, 123)))

    model.close()


def test_ds4_temperature_zero_uses_argmax(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102, 103]
    fake.token_text_map = {101: b"A", 102: b"B", 103: b"C"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                do_sample=True,
                max_new_tokens=2,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "AB"
    session = fake.sessions[0]
    assert session.argmax_calls == 2
    assert session.sample_options == []
    assert session.eval_calls == [101, 102]
    model.close()


def test_ds4_nonzero_temperature_sampling_uses_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.sample_script = [201]
    fake.token_text_map = {201: b"S"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                do_sample=True,
                max_new_tokens=1,
                temperature=0.7,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "S"
    session = fake.sessions[0]
    assert session.argmax_calls == 0
    assert session.sample_options == [
        FakeNativeSamplingOptions(
            temperature=0.7,
            top_k=50,
            top_p=1.0,
            min_p=0.0,
            seed=None,
        )
    ]
    model.close()


def test_ds4_none_sampling_values_map_to_ds4_defaults_and_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(backend_config={"seed": 7}),
    )
    fake = _latest_fake_engine()
    fake.sample_script = [301]
    fake.token_text_map = {301: b"D"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                do_sample=True,
                max_new_tokens=1,
                min_p=None,
                temperature=0.5,
                top_k=None,
                top_p=None,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "D"
    assert fake.sessions[0].sample_options == [
        FakeNativeSamplingOptions(
            temperature=0.5,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            seed=7,
        )
    ]
    model.close()


def test_ds4_max_new_tokens_limits_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102, 103]
    fake.token_text_map = {101: b"A", 102: b"B", 103: b"C"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=2,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "AB"
    assert fake.sessions[0].eval_calls == [101, 102]
    model.close()


def test_ds4_max_length_fallback_accounts_for_prompt_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102, 103]
    fake.token_text_map = {101: b"A", 102: b"B", 103: b"C"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_length=6,
                max_new_tokens=None,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert response.input_token_count == 4
    assert run(response.to_str()) == "AB"
    assert fake.sessions[0].eval_calls == [101, 102]
    model.close()


def test_ds4_stop_string_terminates_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102, 103]
    fake.token_text_map = {101: b"hello ", 102: b"STOP", 103: b"after"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=3,
                stop_strings="STOP",
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "hello "
    assert fake.sessions[0].eval_calls == [101, 102]
    model.close()


def test_ds4_stop_string_spanning_token_boundaries_is_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102, 103]
    fake.token_text_map = {101: b"hello E", 102: b"ND after", 103: b"more"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=3,
                stop_strings="END",
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "hello "
    assert fake.sessions[0].eval_calls == [101, 102]
    model.close()


def test_ds4_eos_stops_generation_without_emitting_or_eval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, fake.eos_token_id, 102]
    fake.token_text_map = {101: b"A", fake.eos_token_id: b"EOS", 102: b"B"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=3,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "A"
    assert fake.sessions[0].eval_calls == [101]
    assert ("token_text", fake.eos_token_id) not in fake.operation_log
    model.close()


def test_ds4_invalid_utf8_token_bytes_are_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.token_text_map = {101: b"A\xffB"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "A\ufffdB"
    assert fake.sessions[0].eval_calls == [101]
    model.close()


def test_ds4_generation_stream_returns_plain_strings_without_token_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> list[object]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101]
        fake.token_text_map = {101: b"A"}
        response = await model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                output_scores=True,
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks: list[object] = [chunk async for chunk in response]
        model.close()
        return chunks

    chunks = run(run_case())

    assert chunks == ["A"]
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert not any(isinstance(chunk, TokenDetail) for chunk in chunks)


def test_ds4_generation_stream_parses_dsml_tool_call_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> list[object]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        dsml = (
            "I will calculate.\n\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="math.calculator">\n'
            '<｜DSML｜parameter name="expression" string="true">'
            "2 + 2"
            "</｜DSML｜parameter>\n"
            '<｜DSML｜parameter name="precision" string="false">'
            "2"
            "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        fake.argmax_script = [101, 102]
        fake.token_text_map = {
            101: dsml.encode(),
            102: b" ignored",
        }
        manager = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")]
        )
        response = await model(
            "hello",
            tool=manager,
            settings=GenerationSettings(
                max_new_tokens=1,
                reasoning=ReasoningSettings(enabled=False),
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks: list[object] = [chunk async for chunk in response]
        model.close()
        return chunks

    chunks = run(run_case())

    assert chunks[0] == "I will calculate."
    tool_deltas = [
        chunk.token
        for chunk in chunks
        if isinstance(chunk, ToolCallToken) and chunk.call is None
    ]
    assert tool_deltas == ["2 + 2", "2"]
    final_calls = [
        chunk
        for chunk in chunks
        if isinstance(chunk, ToolCallToken) and chunk.call is not None
    ]
    assert len(final_calls) == 1
    assert final_calls[0].token == ""
    assert final_calls[0].call is not None
    assert final_calls[0].call.name == "math.calculator"
    assert final_calls[0].call.arguments == {
        "expression": "2 + 2",
        "precision": 2,
    }
    assert (
        "".join(
            str(chunk)
            for chunk in chunks
            if not isinstance(chunk, ToolCallToken)
        )
        == "I will calculate."
    )


def test_ds4_generation_stream_emits_split_dsml_argument_deltas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> list[object]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101, 102, 103, 104, 105]
        fake.token_text_map = {
            101: (
                b"I will calculate.\n\n"
                b"<\xef\xbd\x9cDSML\xef\xbd\x9ctool_calls>\n"
                b"<\xef\xbd\x9cDSML\xef\xbd\x9cinvoke "
                b'name="math.calculator">\n'
                b"<\xef\xbd\x9cDSML\xef\xbd\x9cparameter "
                b'name="expression" string="true">'
            ),
            102: b"2 + ",
            103: b"2",
            104: (
                b"</\xef\xbd\x9cDSML\xef\xbd\x9cparameter>\n"
                b"</\xef\xbd\x9cDSML\xef\xbd\x9cinvoke>\n"
            ),
            105: b"</\xef\xbd\x9cDSML\xef\xbd\x9ctool_calls>",
        }
        manager = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")]
        )
        response = await model(
            "hello",
            tool=manager,
            settings=GenerationSettings(
                max_new_tokens=5,
                reasoning=ReasoningSettings(enabled=False),
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks: list[object] = [chunk async for chunk in response]
        model.close()
        return chunks

    chunks = run(run_case())

    assert [
        chunk.token
        for chunk in chunks
        if isinstance(chunk, ToolCallToken) and chunk.call is None
    ] == ["2 + 2"]
    assert (
        "".join(
            str(chunk)
            for chunk in chunks
            if not isinstance(chunk, ToolCallToken)
        )
        == "I will calculate."
    )
    final_calls = [
        chunk.call
        for chunk in chunks
        if isinstance(chunk, ToolCallToken) and chunk.call is not None
    ]
    assert len(final_calls) == 1
    assert final_calls[0] is not None
    assert final_calls[0].arguments == {"expression": "2 + 2"}


def test_ds4_tool_mode_streams_plain_content_incrementally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> tuple[object, list[int]]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101, 102, 103]
        fake.token_text_map = {101: b"A", 102: b"B", 103: b"C"}
        manager = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")]
        )
        response = await model(
            "hello",
            tool=manager,
            settings=GenerationSettings(
                max_new_tokens=3,
                reasoning=ReasoningSettings(enabled=False),
                temperature=0.0,
                use_async_generator=True,
            ),
        )

        iterator = aiter(response)
        first = await anext(iterator)
        eval_calls = list(fake.sessions[0].eval_calls)
        _ = [chunk async for chunk in iterator]
        model.close()
        return first, eval_calls

    first, eval_calls = run(run_case())

    assert first == "A"
    assert eval_calls == [101]


def test_ds4_tool_mode_holds_only_potential_dsml_start_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> list[object]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101, 102, 103, 104]
        fake.token_text_map = {
            101: b"Answer",
            102: b"\n\n<",
            103: b"not a tool call",
            104: b".",
        }
        manager = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")]
        )
        response = await model(
            "hello",
            tool=manager,
            settings=GenerationSettings(
                max_new_tokens=4,
                reasoning=ReasoningSettings(enabled=False),
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks = [chunk async for chunk in response]
        model.close()
        return chunks

    chunks = run(run_case())

    assert chunks == ["Answer", "\n\n<not a tool call", "."]


def test_ds4_generated_tool_call_round_trips_through_tool_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> ToolCallResult:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101]
        dsml = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="math.calculator">\n'
            '<｜DSML｜parameter name="expression" string="true">'
            "2 + 2"
            "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        fake.token_text_map = {101: dsml.encode()}
        manager = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")]
        )
        response = await model(
            "calculate",
            tool=manager,
            settings=GenerationSettings(
                max_new_tokens=1,
                reasoning=ReasoningSettings(enabled=False),
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks = [chunk async for chunk in response]
        calls = [
            chunk.call
            for chunk in chunks
            if isinstance(chunk, ToolCallToken) and chunk.call is not None
        ]
        assert len(calls) == 1
        result = await manager(calls[0], ToolCallContext(calls=calls))
        model.close()
        assert isinstance(result, ToolCallResult)
        return result

    result = run(run_case())

    assert result.name == "math.calculator"
    assert result.arguments == {"expression": "2 + 2"}
    assert result.result == "4"


def test_ds4_tool_prompt_keeps_dsml_when_manager_uses_other_tool_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    manager = ToolManager.create_instance(
        available_toolsets=[MathToolSet(namespace="math")],
        settings=ToolManagerSettings(tool_format=ToolFormat.JSON),
    )

    response = run(
        model(
            "hello",
            tool=manager,
            settings=GenerationSettings(max_new_tokens=0),
        )
    )

    assert run(response.to_str()) == ""
    assert manager.tool_format is ToolFormat.JSON
    rendered = _latest_fake_engine().tokenize_rendered_chat_calls[-1]
    assert "<｜DSML｜tool_calls>" in rendered
    assert '"tool"' not in rendered
    assert '"name":"math.calculator"' in rendered
    model.close()


def test_ds4_tool_history_replays_exact_sampled_dsml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> str:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        raw_dsml = (
            "\n\n<DSML｜tool_calls>\n"
            '<DSML｜invoke name="math.calculator">\n'
            '<DSML｜parameter name="precision" string="false">'
            "2"
            "</DSML｜parameter>\n"
            '<DSML｜parameter name="expression" string="true">'
            "2 + 2"
            "</DSML｜parameter>\n"
            "</DSML｜invoke>\n"
            "</DSML｜tool_calls>"
        )
        fake.argmax_script = [101]
        fake.token_text_map = {101: raw_dsml.encode()}
        manager = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")]
        )

        first = await model(
            "hello",
            tool=manager,
            settings=GenerationSettings(
                max_new_tokens=1,
                reasoning=ReasoningSettings(enabled=False),
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks = [chunk async for chunk in first]
        call_tokens = [
            chunk
            for chunk in chunks
            if isinstance(chunk, ToolCallToken) and chunk.call is not None
        ]
        assert len(call_tokens) == 1
        call = call_tokens[0].call
        assert call is not None

        await model(
            [
                Message(role=MessageRole.USER, content="hello"),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        MessageToolCall(
                            id=str(call.id),
                            name=call.name,
                            arguments=call.arguments or {},
                        )
                    ],
                ),
                Message(role=MessageRole.TOOL, content="4"),
            ],
            tool=manager,
            settings=GenerationSettings(max_new_tokens=0),
        )
        rendered = fake.tokenize_rendered_chat_calls[-1]
        model.close()
        return rendered

    rendered = run(run_case())

    assert "<DSML｜tool_calls>" in rendered
    assistant_history = rendered.split("<｜Assistant｜>", 1)[1]
    assert "<｜DSML｜tool_calls>" not in assistant_history
    assert '<DSML｜parameter name="precision" string="false">2' in rendered


def test_ds4_tool_history_unknown_replay_id_uses_canonical_dsml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    manager = ToolManager.create_instance(
        available_toolsets=[MathToolSet(namespace="math")]
    )

    run(
        model(
            [
                Message(role=MessageRole.USER, content="calculate"),
                Message(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        MessageToolCall(
                            id="missing_ds4_tool_id",
                            name="math.calculator",
                            arguments={"expression": "2 + 2"},
                        )
                    ],
                ),
                Message(role=MessageRole.TOOL, content="4"),
            ],
            tool=manager,
            settings=GenerationSettings(max_new_tokens=0),
        )
    )

    rendered = _latest_fake_engine().tokenize_rendered_chat_calls[-1]
    assert "<DSML｜tool_calls>" not in rendered
    assert '<｜DSML｜invoke name="math.calculator">' in rendered
    model.close()


def test_ds4_generation_stream_rejects_malformed_dsml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> None:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101]
        fake.token_text_map = {
            101: b"<\xef\xbd\x9cDSML\xef\xbd\x9ctool_calls>",
        }
        manager = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")]
        )
        response = await model(
            "hello",
            tool=manager,
            settings=GenerationSettings(
                max_new_tokens=1,
                reasoning=ReasoningSettings(enabled=False),
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        with pytest.raises(Ds4GenerationError, match="malformed DSML"):
            _ = [chunk async for chunk in response]
        model.close()

    run(run_case())


def test_ds4_generation_stream_flushes_unsafe_non_dsml_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def generate_text_chunks(
        self: Ds4Worker,
        session: object,
        generation_plan: _Ds4GenerationPlan,
    ):
        yield "<"

    monkeypatch.setattr(
        Ds4Worker, "_generate_text_chunks", generate_text_chunks
    )
    worker = object.__new__(Ds4Worker)
    plan = _Ds4GenerationPlan(
        max_new_tokens=1,
        sampling_options=SamplingOptions(),
        stop_strings=(),
        use_sampling=False,
        parse_dsml_tools=True,
    )

    async def run_case() -> list[object]:
        return [
            chunk
            async for chunk in worker._generate_dsml_tool_chunks(
                object(), plan
            )
        ]

    assert run(run_case()) == ["<"]


def test_ds4_generation_stream_returns_token_details_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, logprobs=True
            )
        ),
    )

    async def run_case() -> tuple[list[object], FakeNativeEngine]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101]
        fake.token_logprobs = {101: -0.1}
        fake.token_text_map = {101: b"A", 102: b"B"}
        fake.top_logprobs = [(101, -0.1), (102, -1.5)]
        response = await model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=True,
            ),
            manual_sampling=True,
            pick=2,
        )
        chunks: list[object] = [chunk async for chunk in response]
        model.close()
        return chunks, fake

    chunks, fake = run(run_case())

    assert len(chunks) == 1
    detail = chunks[0]
    assert isinstance(detail, TokenDetail)
    assert detail.id == 101
    assert detail.token == "A"
    assert detail.step == 0
    assert detail.probability == pytest.approx(exp(-0.1))
    assert detail.probability_distribution == "log_softmax"
    assert detail.tokens == [
        Token(id=101, token="A", probability=exp(-0.1)),
        Token(id=102, token="B", probability=exp(-1.5)),
    ]
    assert fake.sessions[0].top_logprobs_calls == [2]
    assert fake.sessions[0].token_logprob_calls == [101]
    assert fake.sessions[0].eval_calls == [101]


def test_ds4_token_details_reject_invalid_top_k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))

    with pytest.raises(ValueError, match="top_logprobs"):
        run(
            model(
                "hello",
                settings=GenerationSettings(max_new_tokens=1),
                manual_sampling=True,
                pick=-1,
            )
        )
    model.close()


def test_ds4_token_details_require_native_logprob_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=True,
            ),
            manual_sampling=True,
            pick=2,
        )
    )

    with pytest.raises(NotImplementedError, match="token logprobs"):
        run(response.to_str())
    model.close()


def test_ds4_missing_logprob_capability_only_blocks_token_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, logprobs=True
            ),
            capabilities=lambda: _fake_capabilities(
                logprobs=False,
                top_logprobs=False,
            ),
        ),
    )
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.token_text_map = {101: b"A"}

    plain = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )
    assert run(plain.to_str()) == "A"

    detailed = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
            manual_sampling=True,
            pick=1,
        )
    )
    with pytest.raises(NotImplementedError, match="token logprobs"):
        run(detailed.to_str())
    assert fake.sessions[0].token_logprob_calls == []
    model.close()


def test_ds4_eval_call_order_matches_verified_session_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.token_text_map = {101: b"A"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "A"
    assert fake.operation_log == [
        ("sync", (30, 0, 5, 20)),
        ("argmax", None),
        ("eval", 101),
        ("token_text", 101),
    ]
    model.close()


def test_ds4_empty_token_text_does_not_corrupt_output_or_counters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102]
    fake.token_text_map = {101: b"", 102: b"A"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=2,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "A"
    assert response.output_token_count == 1
    assert fake.sessions[0].eval_calls == [101, 102]
    model.close()


def test_ds4_token_text_failure_propagates_and_invalidates_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.fail_token_text_ids = {101}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    with pytest.raises(Ds4GenerationError, match="token_text failed"):
        run(response.to_str())

    assert fake.sessions[0].invalidate_calls == 1
    assert len(fake.sessions) == 2
    model.close()


def test_ds4_token_text_failure_restores_snapshot_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, snapshots=True
            )
        ),
    )
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.fail_token_text_ids = {101}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    with pytest.raises(Ds4GenerationError, match="token_text failed"):
        run(response.to_str())

    assert fake.sessions[0].save_snapshot_calls == 1
    assert fake.sessions[0].load_snapshot_calls == 1
    assert fake.sessions[0].invalidate_calls == 0
    assert fake.sessions[0].tokens == [30, 0, 5, 20]
    assert len(fake.sessions) == 1
    model.close()


def test_ds4_missing_snapshot_capability_rebuilds_on_generation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, snapshots=True
            ),
            capabilities=lambda: _fake_capabilities(snapshots=False),
        ),
    )
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.fail_token_text_ids = {101}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    with pytest.raises(Ds4GenerationError, match="token_text failed"):
        run(response.to_str())

    assert fake.sessions[0].save_snapshot_calls == 0
    assert fake.sessions[0].load_snapshot_calls == 0
    assert fake.sessions[0].invalidate_calls == 1
    assert len(fake.sessions) == 2
    model.close()


def test_ds4_disk_kv_cache_hit_restores_matching_token_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, payloads=True
            )
        ),
    )
    cache_dir = tmp_path / "kv"
    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(
            backend_config={
                "kv_disk_dir": str(cache_dir),
                "kv_disk_space_mb": 1,
            }
        ),
    )
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102]
    fake.token_text_map = {101: b"A", 102: b"B"}

    first = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )
    second = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(first.to_str()) == "A"
    assert run(second.to_str()) == "B"
    assert fake.session_sync_calls == [(30, 0, 5, 20)]
    assert fake.sessions[0].save_payload_calls == 1
    assert fake.sessions[0].load_payload_calls == 1
    assert fake.sessions[0].tokens == [30, 0, 5, 20, 102]
    assert list(cache_dir.glob("*.payload"))
    model.close()


def test_ds4_disk_kv_cache_misses_for_different_prefix_or_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, payloads=True
            )
        ),
    )
    cache_dir = tmp_path / "kv"
    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(
            backend_config={
                "ctx_size": 16,
                "kv_disk_dir": str(cache_dir),
                "kv_disk_space_mb": 1,
            }
        ),
    )
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102]

    first = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )
    second = run(
        model(
            "different",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )

    assert run(first.to_str()) == "<101>"
    assert run(second.to_str()) == "<102>"
    assert fake.session_sync_calls == [(30, 0, 5, 20), (30, 0, 9, 20)]
    assert fake.sessions[0].load_payload_calls == 0
    model.close()

    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(
            backend_config={
                "ctx_size": 32,
                "kv_disk_dir": str(cache_dir),
                "kv_disk_space_mb": 1,
            }
        ),
    )
    fake = _latest_fake_engine()
    fake.argmax_script = [103]
    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "<103>"
    assert fake.session_sync_calls == [(30, 0, 5, 20)]
    assert fake.sessions[0].load_payload_calls == 0
    model.close()


def test_ds4_disk_kv_cache_corrupt_payload_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, payloads=True
            )
        ),
    )
    cache_dir = tmp_path / "kv"
    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(
            backend_config={
                "kv_disk_dir": str(cache_dir),
                "kv_disk_space_mb": 1,
            }
        ),
    )
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102]

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )
    assert run(response.to_str()) == "<101>"
    payload_path = next(cache_dir.glob("*.payload"))
    payload_path.write_bytes(b"corrupt")

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "<102>"
    assert fake.session_sync_calls == [(30, 0, 5, 20), (30, 0, 5, 20)]
    assert fake.sessions[0].load_payload_calls == 1
    model.close()


def test_ds4_disk_kv_cache_write_failure_falls_back_to_live_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, payloads=True
            )
        ),
    )
    cache_file = tmp_path / "not-a-directory"
    cache_file.write_text("file", encoding="utf-8")
    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(
            backend_config={
                "kv_disk_dir": str(cache_file),
                "kv_disk_space_mb": 1,
            }
        ),
    )
    fake = _latest_fake_engine()
    fake.argmax_script = [101]

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "<101>"
    assert fake.session_sync_calls == [(30, 0, 5, 20)]
    assert fake.sessions[0].save_payload_calls == 1
    assert cache_file.read_text(encoding="utf-8") == "file"
    model.close()


def test_ds4_disk_kv_cache_disabled_leaves_no_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, payloads=True
            )
        ),
    )
    cache_dir = tmp_path / "kv"
    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(
            backend_config={
                "kv_disk_dir": str(cache_dir),
                "kv_disk_space_mb": 0,
            }
        ),
    )
    fake = _latest_fake_engine()
    fake.argmax_script = [101]

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "<101>"
    assert fake.sessions[0].save_payload_calls == 0
    assert not cache_dir.exists()
    model.close()


def test_ds4_missing_payload_capability_disables_disk_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, payloads=True
            ),
            capabilities=lambda: _fake_capabilities(payloads=False),
        ),
    )
    cache_dir = tmp_path / "kv"
    model = Ds4Model(
        str(_model_file(tmp_path)),
        TransformerEngineSettings(
            backend_config={
                "kv_disk_dir": str(cache_dir),
                "kv_disk_space_mb": 1,
            }
        ),
    )
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102]

    first = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )
    second = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                use_async_generator=False,
            ),
        )
    )

    assert run(first.to_str()) == "<101>"
    assert run(second.to_str()) == "<102>"
    assert fake.session_sync_calls == [
        (30, 0, 5, 20),
        (30, 0, 5, 20),
    ]
    assert fake.sessions[0].save_payload_calls == 0
    assert fake.sessions[0].load_payload_calls == 0
    assert not cache_dir.exists()
    model.close()


def test_ds4_disk_kv_cache_budget_evicts_least_useful_entries(
    tmp_path: Path,
) -> None:
    logger = cast(Logger, MagicMock(spec=Logger))
    cache = _Ds4DiskKvCache(tmp_path / "kv", 700, logger, "namespace")
    session = SimpleNamespace(save_payload=lambda: b"x" * 100)

    run(cache.store(session, [1], 16))
    first_payload = next((tmp_path / "kv").glob("*.payload"))
    run(cache.store(session, [2], 16))

    payloads = tuple((tmp_path / "kv").glob("*.payload"))
    assert len(payloads) == 1
    assert first_payload not in payloads


def test_ds4_streaming_response_yields_fake_token_text_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> tuple[list[str], FakeNativeEngine, int]:
        main_thread_id = get_ident()
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101, 102]
        fake.token_text_map = {101: b"A", 102: b"B"}

        response = await model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=2,
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks = [cast(str, chunk) async for chunk in response]
        model.close()
        return chunks, fake, main_thread_id

    chunks, fake, main_thread_id = run(run_case())

    assert chunks == ["A", "B"]
    assert fake.sessions[0].eval_calls == [101, 102]
    assert fake.thread_ids
    assert all(thread_id != main_thread_id for thread_id in fake.thread_ids)


def test_ds4_non_streaming_response_returns_complete_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102]
    fake.token_text_map = {101: b"A", 102: b"B"}

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=2,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    assert run(response.to_str()) == "AB"
    assert fake.sessions[0].eval_calls == [101, 102]
    model.close()


def test_ds4_worker_serializes_concurrent_streaming_generations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> tuple[list[list[str]], FakeNativeEngine]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101, 102]
        fake.token_text_map = {101: b"A", 102: b"B"}

        async def collect(prompt: str) -> list[str]:
            response = await model(
                prompt,
                settings=GenerationSettings(
                    max_new_tokens=1,
                    temperature=0.0,
                    use_async_generator=True,
                ),
            )
            return [cast(str, chunk) async for chunk in response]

        results = await asyncio.gather(collect("one"), collect("two"))
        model.close()
        return list(results), fake

    results, fake = run(run_case())

    assert results == [["A"], ["B"]]
    assert len(fake.session_sync_calls) == 2
    assert len(fake.sessions) == 1
    assert fake.sessions[0].eval_calls == [101, 102]


def test_ds4_worker_exception_surfaces_when_stream_is_consumed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> FakeNativeEngine:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101]
        fake.fail_on_eval = True
        fake.token_text_map = {101: b"A"}
        response = await model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=True,
            ),
        )

        with pytest.raises(Ds4GenerationError, match="eval failed"):
            async for _ in response:
                pass

        model.close()
        return fake

    fake = run(run_case())

    assert fake.sessions[0].invalidate_calls == 1
    assert len(fake.sessions) == 2


def test_ds4_context_overflow_surfaces_as_context_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.sync_error = RuntimeError("prompt exceeds context")

    response = run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=1,
                temperature=0.0,
                use_async_generator=False,
            ),
        )
    )

    with pytest.raises(Ds4ContextError, match="prompt exceeds context"):
        run(response.to_str())

    assert fake.sessions[0].invalidate_calls == 1
    assert len(fake.sessions) == 2
    model.close()


def test_ds4_model_context_exit_closes_worker_session_and_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    worker = cast(Ds4Worker, model.model)
    fake = _latest_fake_engine()

    with model:
        assert worker.is_alive is True

    assert worker.closed is True
    assert worker.is_alive is False
    assert fake.sessions[0].close_calls == 1
    assert fake.close_calls == 1


def test_ds4_stream_cancellation_invalidates_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> tuple[list[str], FakeNativeEngine]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101, 102]
        fake.block_argmax_after_calls = 1
        fake.token_text_map = {101: b"A", 102: b"B"}
        response = await model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=2,
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks: list[str] = []

        async def consume() -> None:
            async for chunk in response:
                chunks.append(cast(str, chunk))

        task = asyncio.create_task(consume())
        while not chunks:
            await asyncio.sleep(0.01)

        assert await asyncio.to_thread(fake.argmax_block_started.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        fake.argmax_release.set()
        assert await asyncio.to_thread(fake.invalidate_event.wait, 1.0)
        for _ in range(100):
            if len(fake.sessions) == 2:
                break
            await asyncio.sleep(0.01)
        model.close()
        return chunks, fake

    chunks, fake = run(run_case())

    assert chunks == ["A"]
    assert fake.sessions[0].invalidate_calls == 1
    assert len(fake.sessions) == 2


def test_ds4_stream_cancellation_restores_snapshot_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(
            AsyncEngine=_fake_async_engine_type(
                FakeNativeEngine, snapshots=True
            )
        ),
    )

    async def run_case() -> tuple[list[str], FakeNativeEngine]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101, 102]
        fake.block_argmax_after_calls = 1
        fake.token_text_map = {101: b"A", 102: b"B"}
        response = await model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=2,
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks: list[str] = []

        async def consume() -> None:
            async for chunk in response:
                chunks.append(cast(str, chunk))

        task = asyncio.create_task(consume())
        while not chunks:
            await asyncio.sleep(0.01)

        assert await asyncio.to_thread(fake.argmax_block_started.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        fake.argmax_release.set()
        assert await asyncio.to_thread(fake.snapshot_loaded_event.wait, 1.0)
        model.close()
        return chunks, fake

    chunks, fake = run(run_case())

    assert chunks == ["A"]
    assert fake.sessions[0].save_snapshot_calls == 1
    assert fake.sessions[0].load_snapshot_calls == 1
    assert fake.sessions[0].invalidate_calls == 0
    assert fake.sessions[0].tokens == [30, 0, 5, 20]
    assert len(fake.sessions) == 1


def test_ds4_stream_queue_backpressure_does_not_deadlock_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())

    async def run_case() -> list[str]:
        model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101, 102, 103, 104, 105]
        fake.token_text_map = {
            101: b"A",
            102: b"B",
            103: b"C",
            104: b"D",
            105: b"E",
        }
        response = await model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=5,
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks: list[str] = []
        async for chunk in response:
            chunks.append(cast(str, chunk))
            await asyncio.sleep(0.01)
        model.close()
        return chunks

    chunks = run(asyncio.wait_for(run_case(), timeout=1.0))

    assert chunks == ["A", "B", "C", "D", "E"]


def test_ds4_reasoning_settings_map_to_think_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))

    run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=0,
                reasoning=ReasoningSettings(enabled=False),
            ),
        )
    )
    run(
        model(
            "hello",
            settings=GenerationSettings(
                max_new_tokens=0,
                reasoning=ReasoningSettings(effort=ReasoningEffort.MAX),
            ),
        )
    )

    fake = _latest_fake_engine()
    assert fake.tokenization_calls[-2:] == [
        (
            "encode_chat_prompt",
            (None, "hello"),
            NativeThinkMode.NONE,
        ),
        (
            "encode_chat_prompt",
            (None, "hello"),
            NativeThinkMode.MAX,
        ),
    ]
    model.close()


def test_ds4_negative_length_is_rejected_before_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()

    with pytest.raises(ValueError, match="max_new_tokens"):
        run(model("hello", settings=GenerationSettings(max_new_tokens=-1)))

    assert fake.session_sync_calls == []
    model.close()


def test_ds4_invalid_sampling_values_are_rejected_before_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()

    with pytest.raises(ValueError, match="top_p"):
        run(
            model(
                "hello",
                settings=GenerationSettings(
                    do_sample=True,
                    max_new_tokens=1,
                    temperature=0.5,
                    top_p=1.5,
                ),
            )
        )

    assert fake.session_sync_calls == []
    model.close()


def test_ds4_unsupported_beam_search_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))

    with pytest.raises(NotImplementedError, match="beam search"):
        run(model("hello", settings=GenerationSettings(num_beams=2)))

    model.close()


def test_ds4_unsupported_multi_return_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))

    with pytest.raises(NotImplementedError, match="multiple return sequences"):
        run(
            model(
                "hello",
                settings=GenerationSettings(num_return_sequences=2),
            )
        )

    model.close()


def test_ds4_worker_start_and_close_are_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )

    assert worker.closed is True
    assert worker.is_alive is False

    worker.start()
    worker.start()

    assert worker.closed is False
    assert worker.is_alive is True
    assert len(FakeNativeEngine.instances) == 1

    worker.close()
    worker.close()

    assert worker.closed is True
    assert worker.is_alive is False


def test_ds4_worker_stream_rejects_closed_worker(tmp_path: Path) -> None:
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )
    plan = _Ds4GenerationPlan(
        max_new_tokens=1,
        sampling_options=SamplingOptions(),
        stop_strings=(),
        use_sampling=False,
    )

    async def consume() -> None:
        async for _ in worker.stream([1], plan):
            pass

    with pytest.raises(Ds4LoadError, match="closed"):
        run(consume())


def test_ds4_worker_requires_async_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding(AsyncEngine=None))

    with pytest.raises(Ds4LoadError, match="AsyncEngine"):
        Ds4Model(str(_model_file(tmp_path)))


def test_ds4_worker_supports_constructor_only_async_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_engine = _fake_async_engine_type(FakeNativeEngine)

    class ConstructorOnlyAsyncEngine(base_engine):
        entered = False
        open = None

        async def __aenter__(self) -> "ConstructorOnlyAsyncEngine":
            type(self).entered = True
            return self

    ConstructorOnlyAsyncEngine.__module__ = "pyds4"
    _install_binding(
        monkeypatch, _fake_binding(AsyncEngine=ConstructorOnlyAsyncEngine)
    )

    model = Ds4Model(str(_model_file(tmp_path)))

    assert ConstructorOnlyAsyncEngine.entered is True
    model.close()


def test_ds4_worker_constructor_open_failure_is_mapped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingAsyncEngine:
        open = None

        def __init__(self, _: FakeNativeOptions) -> None:
            raise RuntimeError("constructor open failed")

    FailingAsyncEngine.__module__ = "pyds4"
    _install_binding(
        monkeypatch, _fake_binding(AsyncEngine=FailingAsyncEngine)
    )

    with pytest.raises(Ds4LoadError, match="constructor open failed"):
        Ds4Model(str(_model_file(tmp_path)))


def test_ds4_worker_generate_string_uses_shared_generation_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101, 102]
    fake.token_text_map = {101: b"A", 102: b"B"}
    plan = _Ds4GenerationPlan(
        max_new_tokens=2,
        sampling_options=SamplingOptions(),
        stop_strings=(),
        use_sampling=False,
    )

    assert model._ds4_worker().generate_string([1], plan) == "AB"
    model.close()


def test_ds4_worker_token_text_fetches_and_normalizes_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    worker = model._ds4_worker()
    fake = _latest_fake_engine()
    fake.token_text_map = {101: b"A"}

    async def run_case() -> tuple[str, str, str]:
        engine = worker._require_engine()
        return (
            await worker._token_text(engine, SimpleNamespace(token_id=101)),
            await worker._token_text(
                engine, SimpleNamespace(token_bytes=bytearray(b"B"))
            ),
            await worker._token_text(
                engine, SimpleNamespace(token_bytes=memoryview(b"C"))
            ),
        )

    assert run(run_case()) == ("A", "B", "C")
    model.close()


@pytest.mark.parametrize(
    "step, message",
    [
        (SimpleNamespace(token_id=True), "token id"),
        (SimpleNamespace(token_bytes="bad"), "must be bytes"),
    ],
)
def test_ds4_worker_token_text_rejects_invalid_steps(
    tmp_path: Path, step: SimpleNamespace, message: str
) -> None:
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )

    with pytest.raises(Ds4GenerationError, match=message):
        run(worker._token_text(object(), step))


def test_ds4_worker_uses_binding_fallbacks_without_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(
        monkeypatch,
        _fake_binding(SamplingOptions=None, think_mode_for_context=None),
    )
    model = Ds4Model(str(_model_file(tmp_path)))
    worker = model._ds4_worker()
    options = SamplingOptions(temperature=0.5, top_k=4)

    assert worker._effective_think_mode(ThinkMode.HIGH) == NativeThinkMode.HIGH
    assert worker._native_sampling_options(options) is options
    model.close()


def test_ds4_worker_call_async_reports_missing_and_native_errors() -> None:
    class FailingTarget:
        def fail(self) -> None:
            raise RuntimeError("native failure")

    with pytest.raises(Ds4LoadError, match="missing"):
        run(Ds4Worker._call_async(object(), "missing"))

    with pytest.raises(Ds4GenerationError, match="native failure"):
        run(Ds4Worker._call_async(FailingTarget(), "fail"))


def test_ds4_worker_run_sync_propagates_threaded_failure() -> None:
    async def fail() -> object:
        raise RuntimeError("threaded failure")

    async def run_case() -> None:
        with pytest.raises(RuntimeError, match="threaded failure"):
            Ds4Worker._run_sync(fail)

    run(run_case())


def test_ds4_worker_token_list_validation() -> None:
    assert Ds4Worker._token_list((1, 2, 3)) == [1, 2, 3]

    with pytest.raises(Ds4GenerationError, match="token IDs"):
        Ds4Worker._token_list([1, True])


def test_ds4_worker_require_methods_fail_before_start(tmp_path: Path) -> None:
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )

    with pytest.raises(Ds4LoadError, match="binding"):
        worker._require_binding()
    with pytest.raises(Ds4LoadError, match="engine"):
        worker._require_engine()
    with pytest.raises(Ds4GenerationError, match="session"):
        worker._require_session()


def test_ds4_model_support_flags_and_async_native_acceptance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class NativeAsyncEngine:
        pass

    NativeAsyncEngine.__module__ = "external"
    _install_binding(
        monkeypatch,
        _fake_binding(AsyncEngine=NativeAsyncEngine),
    )
    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False),
    )

    assert model.supports_sample_generation is True
    assert model.supports_token_streaming is True
    assert model._accepts_loaded_model(NativeAsyncEngine()) is True


def test_ds4_model_acceptance_falls_back_when_binding_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_import(_: str) -> object:
        raise ModuleNotFoundError("No module named 'pyds4'")

    monkeypatch.setattr(availability, "import_module", fail_import)
    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False),
    )

    assert model._accepts_loaded_model(object()) is False


def test_ds4_prompt_messages_merge_system_developer_and_text_content(
    tmp_path: Path,
) -> None:
    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False),
    )
    content = [
        MessageContentImage(type="image_url", image_url={"url": "x"}),
        MessageContentText(type="text", text="Hello"),
        MessageContentFile(type="file", file={"filename": "x.txt"}),
    ]

    system_content, messages = model._ds4_prompt_messages(
        [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(
                role=MessageRole.DEVELOPER,
                content=MessageContentText(type="text", text="Developer"),
            ),
            Message(role=MessageRole.USER, content=content),
        ],
        None,
        None,
    )

    assert system_content == "System\n\nDeveloper instructions:\nDeveloper"
    assert len(messages) == 1
    assert messages[0].role is MessageRole.USER
    assert messages[0].content == "Hello"


def test_ds4_prompt_messages_reject_invalid_input_and_role(
    tmp_path: Path,
) -> None:
    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False),
    )

    with pytest.raises(ValueError):
        model._ds4_prompt_messages(cast(Any, 1), None, None)
    with pytest.raises(ValueError, match="critic"):
        model._ds4_prompt_messages(
            [Message(role=cast(MessageRole, "critic"), content="Hello")],
            None,
            None,
        )


def test_ds4_message_text_handles_empty_and_non_text_content() -> None:
    assert Ds4Model._message_text(None) == ""
    assert (
        Ds4Model._message_text(
            MessageContentImage(type="image_url", image_url={"url": "x"})
        )
        == ""
    )
    assert (
        Ds4Model._message_text(
            MessageContentFile(type="file", file={"filename": "x.txt"})
        )
        == ""
    )
    assert Ds4Model._message_text(123) == "123"


def test_ds4_generation_plan_defaults_and_validation(
    tmp_path: Path,
) -> None:
    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(
            auto_load_model=False,
            backend_config={"seed": None},
        ),
    )

    plan = model._generation_plan(
        GenerationSettings(
            max_length=None,
            temperature=None,
            top_k=None,
            top_p=None,
            min_p=None,
            stop_strings=[],
        ),
        prompt_length=3,
    )

    assert plan.max_new_tokens == 20
    assert plan.stop_strings == ()
    assert plan.sampling_options == SamplingOptions()

    with pytest.raises(ValueError, match="stop_strings"):
        model._generation_plan(
            GenerationSettings(stop_strings=[""]),
            prompt_length=3,
        )
    with pytest.raises(NotImplementedError, match="beam groups"):
        model._generation_plan(
            GenerationSettings(num_beam_groups=2),
            prompt_length=3,
        )


def test_ds4_native_backend_auto_selection_and_rejection(
    tmp_path: Path,
) -> None:
    cuda_model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False, device="cuda:0"),
    )
    mps_model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False, device="mps"),
    )
    default_model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False, device="cpu"),
    )

    assert cuda_model._native_backend({}) is Ds4NativeBackend.CUDA
    assert mps_model._native_backend({}) is Ds4NativeBackend.METAL
    assert default_model._native_backend({}) is Ds4NativeBackend.METAL

    with pytest.raises(Ds4BackendUnavailable, match="must be a string"):
        default_model._native_backend({"native_backend": 1})
    with pytest.raises(Ds4BackendUnavailable, match="Unsupported"):
        default_model._native_backend({"native_backend": "vulkan"})


def test_ds4_model_path_and_config_validation(tmp_path: Path) -> None:
    with pytest.raises(Ds4InvalidModel, match="directory"):
        Ds4Model._validated_file_path(str(tmp_path), "DS4 model path")
    with pytest.raises(Ds4InvalidModel, match="non-empty"):
        Ds4Model._optional_file_path("", "DS4 MTP path")
    with pytest.raises(ValueError, match="ctx_size"):
        Ds4Model(
            str(tmp_path / "not-loaded.gguf"),
            TransformerEngineSettings(
                auto_load_model=False,
                backend_config={"ctx_size": True},
            ),
        )._context_size()
    with pytest.raises(ValueError, match="n_threads"):
        Ds4Model._int_config({"n_threads": True}, "n_threads", 0)
    with pytest.raises(ValueError, match="mtp_margin"):
        Ds4Model._float_config({"mtp_margin": True}, "mtp_margin", 0.0)
    with pytest.raises(ValueError, match="quality"):
        Ds4Model._bool_config({"quality": "yes"}, "quality", False)
    with pytest.raises(ValueError, match="directional_steering_file"):
        Ds4Model._optional_string_config(
            {"directional_steering_file": ""},
            "directional_steering_file",
        )


def test_ds4_worker_flushes_pending_stop_buffer_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.token_text_map = {101: b"A"}
    plan = _Ds4GenerationPlan(
        max_new_tokens=1,
        sampling_options=SamplingOptions(),
        stop_strings=("XYZ",),
        use_sampling=False,
    )

    assert model._ds4_worker().generate_string([1], plan) == "A"
    model.close()


def test_ds4_worker_schedule_reset_ignores_missing_event_loop(
    tmp_path: Path,
) -> None:
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )

    worker._schedule_reset(invalidate=True)

    assert worker._reset_tasks == set()


def test_ds4_worker_close_helpers_cover_sync_and_failed_resources(
    tmp_path: Path,
) -> None:
    logger = MagicMock(spec=Logger)
    worker = Ds4Worker(_worker_options(_model_file(tmp_path)), 16, logger)

    class CloseReturnsAwaitable:
        awaited = False

        def close(self) -> object:
            async def finish() -> None:
                self.awaited = True

            return finish()

    class BadClose:
        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    resource = CloseReturnsAwaitable()
    run(worker._close_async_resource(resource))
    assert resource.awaited is True

    worker._session = BadClose()
    worker._engine = BadClose()
    run(worker._close_resources())
    logger.warning.assert_called_once()


def test_ds4_worker_call_async_returns_plain_value() -> None:
    class PlainTarget:
        def value(self) -> int:
            return 7

    assert run(Ds4Worker._call_async(PlainTarget(), "value")) == 7


def test_ds4_model_close_accepts_compat_engine(tmp_path: Path) -> None:
    closed: list[bool] = []

    class FakeCompatEngine(Ds4CompatEngine):
        def close(self) -> None:
            closed.append(True)

    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False),
    )
    model._model = object.__new__(FakeCompatEngine)

    model.close()

    assert closed == [True]


def test_ds4_model_unloaded_worker_and_sync_tool_errors(
    tmp_path: Path,
) -> None:
    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False),
    )

    with pytest.raises(Ds4LoadError, match="worker"):
        model._ds4_worker()
    with pytest.raises(Ds4LoadError, match="worker"):
        model._render_prompt_tokens(
            "hello",
            None,
            None,
            GenerationSettings(),
            tool=MagicMock(spec=ToolManager),
        )


def test_ds4_reasoning_helpers_cover_none_and_high_effort() -> None:
    assert (
        Ds4Model._think_mode(
            GenerationSettings(
                reasoning=ReasoningSettings(effort=ReasoningEffort.NONE)
            )
        )
        is ThinkMode.NONE
    )
    assert (
        Ds4Model._think_mode(
            GenerationSettings(
                reasoning=ReasoningSettings(effort=ReasoningEffort.HIGH)
            )
        )
        is ThinkMode.HIGH
    )


def test_ds4_generation_string_and_extra_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    fake = _latest_fake_engine()
    fake.argmax_script = [101]
    fake.token_text_map = {101: b"G"}
    plan = _Ds4GenerationPlan(
        max_new_tokens=1,
        sampling_options=SamplingOptions(),
        stop_strings=(),
        use_sampling=False,
    )

    assert model._generation_string([1], plan) == "G"
    assert (
        Ds4Model._merge_system_parts([], ["Developer"])
        == "Developer instructions:\nDeveloper"
    )
    with pytest.raises(ValueError, match="temperature"):
        Ds4Model._non_negative_float("temperature", -0.1, 0.0)
    model.close()


def test_ds4_disk_kv_cache_handles_payload_edge_cases(
    tmp_path: Path,
) -> None:
    logger = MagicMock(spec=Logger)
    cache = _Ds4DiskKvCache(tmp_path / "kv", 1024, logger, "namespace")

    assert run(cache.restore(SimpleNamespace(), [1], 16)) is False
    run(cache.store(SimpleNamespace(), [1], 16))

    class SaveFails:
        def save_payload(self) -> bytes:
            raise RuntimeError("save failed")

    run(cache.store(SaveFails(), [1], 16))
    assert logger.warning.called

    cache_dir = tmp_path / "kv"
    cache_dir.mkdir()
    entry = cache._entry_path([1], 16)
    entry.metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "key": entry.key,
                "namespace": "namespace",
                "ctx_size": 32,
                "token_sha256": entry.token_digest,
                "payload_file": entry.payload_path.name,
            }
        ),
        encoding="utf-8",
    )
    entry.payload_path.write_bytes(b"payload")
    session = SimpleNamespace(load_payload=lambda payload: None)

    assert run(cache.restore(session, [1], 16)) is False
    assert not entry.metadata_path.exists()
    assert not entry.payload_path.exists()

    entry = cache._entry_path([2], 16)
    entry.metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "key": entry.key,
                "namespace": "namespace",
                "ctx_size": 16,
                "token_sha256": entry.token_digest,
                "payload_file": entry.payload_path.name,
            }
        ),
        encoding="utf-8",
    )
    entry.payload_path.mkdir()

    assert run(cache.restore(session, [2], 16)) is False


def test_ds4_disk_kv_cache_metadata_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logger = MagicMock(spec=Logger)
    cache = _Ds4DiskKvCache(tmp_path / "kv", 1024, logger, "namespace")
    entry = cache._entry_path([1], 16)
    entry.metadata_path.parent.mkdir(parents=True)

    def fail_write_text(self: Path, text: str, encoding: str) -> int:
        _ = self, text, encoding
        raise OSError("metadata write failed")

    monkeypatch.setattr(Path, "write_text", fail_write_text)
    cache._record_hit({"hit_count": "bad"}, entry)
    logger.warning.assert_called()

    def fail_glob(self: Path, pattern: str) -> tuple[Path, ...]:
        _ = self, pattern
        raise OSError("glob failed")

    monkeypatch.setattr(Path, "glob", fail_glob)
    assert cache._cache_entries() == []

    def fail_stat(self: Path) -> object:
        _ = self
        raise OSError("stat failed")

    monkeypatch.setattr(Path, "stat", fail_stat)
    assert cache._path_size(entry.metadata_path) == 0

    def fail_unlink(self: Path, missing_ok: bool = False) -> None:
        _ = self, missing_ok
        raise OSError("unlink failed")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    cache._delete_entry(entry)
    assert logger.warning.call_count >= 3


def test_ds4_disk_kv_cache_skips_invalid_metadata_entries(
    tmp_path: Path,
) -> None:
    logger = MagicMock(spec=Logger)
    cache_dir = tmp_path / "kv"
    cache_dir.mkdir()
    (cache_dir / "invalid.json").write_text("{", encoding="utf-8")
    (cache_dir / "missing-fields.json").write_text(
        json.dumps({"key": "only-key"}),
        encoding="utf-8",
    )
    cache = _Ds4DiskKvCache(cache_dir, 1024, logger, "namespace")

    assert cache._cache_entries() == []


def test_ds4_worker_dsml_replay_no_match_and_eviction_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )

    assert worker.exact_dsml_for_tool_calls(()) is None
    assert (
        worker.exact_dsml_for_tool_calls(
            (
                MessageToolCall(
                    id="",
                    name="math.calculator",
                    arguments={},
                ),
            )
        )
        is None
    )

    worker._tool_dsml_replay["a"] = "one"
    worker._tool_dsml_replay["b"] = "two"
    assert (
        worker.exact_dsml_for_tool_calls(
            (
                MessageToolCall(id="a", name="a", arguments={}),
                MessageToolCall(id="b", name="b", arguments={}),
            )
        )
        is None
    )

    monkeypatch.setattr(ds4_module, "_DS4_TOOL_REPLAY_MAX_ENTRIES", 1)
    worker._tool_dsml_replay.clear()
    worker._remember_dsml_tool_replay(
        DsmlParseResult(
            "",
            (
                ToolCall(id="old", name="tool.old", arguments={}),
                ToolCall(id="new", name="tool.new", arguments={}),
            ),
            None,
            "<tool_calls/>",
        )
    )

    assert worker._tool_dsml_replay == {"new": "<tool_calls/>"}


def test_ds4_worker_aclose_snapshot_and_pending_reset_edges(
    tmp_path: Path,
) -> None:
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )

    run(worker.aclose())
    assert run(worker._load_snapshot(object(), b"snapshot")) is False
    run(worker._store_disk_cache(object(), [1]))

    async def run_case() -> bool:
        pending_worker = Ds4Worker(
            _worker_options(_model_file(tmp_path)),
            16,
            MagicMock(spec=Logger),
        )
        task = asyncio.create_task(asyncio.sleep(0.01))
        pending_worker._reset_tasks.add(task)
        await pending_worker._close_resources()
        return task.done()

    assert run(run_case()) is True


def test_ds4_worker_render_prompt_and_plain_dsml_content_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_binding(monkeypatch, _fake_binding())
    model = Ds4Model(str(_model_file(tmp_path)))
    worker = model._ds4_worker()

    tokens = run(
        worker.render_prompt_tokens_async(
            "system",
            [
                DsmlPromptMessage(role=MessageRole.USER, content="hello"),
                DsmlPromptMessage(role=MessageRole.ASSISTANT, content="hi"),
            ],
            ThinkMode.NONE,
        )
    )
    assert tokens == [1, 10, 6, 11, 5, 12, 2, 20]
    model.close()

    async def run_plain_content() -> list[object]:
        plain_model = Ds4Model(str(_model_file(tmp_path)))
        fake = _latest_fake_engine()
        fake.argmax_script = [101]
        fake.token_text_map = {101: b"plain content"}
        manager = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")]
        )
        response = await plain_model(
            "hello",
            tool=manager,
            settings=GenerationSettings(
                max_new_tokens=1,
                reasoning=ReasoningSettings(enabled=False),
                temperature=0.0,
                use_async_generator=True,
            ),
        )
        chunks = [chunk async for chunk in response]
        plain_model.close()
        return chunks

    assert run(run_plain_content()) == ["plain content"]


def test_ds4_worker_generation_error_adds_reset_failure_note(
    tmp_path: Path,
) -> None:
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )
    worker._closed = False
    worker._session = SimpleNamespace(
        sync=lambda tokens: (_ for _ in ()).throw(RuntimeError("sync failed"))
    )
    worker._engine = object()

    async def fail_reset(
        *, invalidate: bool, snapshot: bytes | None = None
    ) -> BaseException:
        _ = invalidate, snapshot
        return RuntimeError("reset failed")

    worker._reset_session = fail_reset  # type: ignore[method-assign]
    plan = _Ds4GenerationPlan(
        max_new_tokens=1,
        sampling_options=SamplingOptions(),
        stop_strings=(),
        use_sampling=False,
    )

    async def consume() -> None:
        async for _ in worker.stream([1], plan):
            pass

    with pytest.raises(Ds4GenerationError) as exc_info:
        run(consume())

    notes = getattr(exc_info.value, "__notes__", ())
    assert any(
        "reset after generation failure failed" in note for note in notes
    )


def test_ds4_worker_open_reraises_known_open_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class OpenRaisesInvalidModel:
        @classmethod
        async def open(cls, _: FakeNativeOptions) -> object:
            raise Ds4InvalidModel("open invalid")

    OpenRaisesInvalidModel.__module__ = "pyds4"
    _install_binding(
        monkeypatch, _fake_binding(AsyncEngine=OpenRaisesInvalidModel)
    )
    with pytest.raises(Ds4InvalidModel, match="open invalid"):
        Ds4Model(str(_model_file(tmp_path)))

    class ConstructorRaisesInvalidModel:
        open = None

        def __init__(self, _: FakeNativeOptions) -> None:
            raise Ds4InvalidModel("constructor invalid")

    ConstructorRaisesInvalidModel.__module__ = "pyds4"
    _install_binding(
        monkeypatch,
        _fake_binding(AsyncEngine=ConstructorRaisesInvalidModel),
    )
    with pytest.raises(Ds4InvalidModel, match="constructor invalid"):
        Ds4Model(str(_model_file(tmp_path)))


def test_ds4_worker_logprob_probability_and_bytes_edge_helpers(
    tmp_path: Path,
) -> None:
    worker = Ds4Worker(
        _worker_options(_model_file(tmp_path)), 16, MagicMock(spec=Logger)
    )
    worker._binding = _fake_binding()

    class EosSession:
        def token_logprob(self, token_id: int) -> float:
            _ = token_id
            return 0.0

        def argmax(self) -> int:
            return 9

    step = run(
        worker._detailed_generation_step(
            SimpleNamespace(eos_token_id=9),
            EosSession(),
            _Ds4GenerationPlan(
                max_new_tokens=1,
                sampling_options=SamplingOptions(),
                stop_strings=(),
                use_sampling=False,
            ),
            0,
        )
    )
    assert step.is_eos is True

    class SampleSession:
        def sample(self, options: object) -> int:
            assert isinstance(options, FakeNativeSamplingOptions)
            return 7

    sampled = run(
        worker._select_token(
            SampleSession(),
            _Ds4GenerationPlan(
                max_new_tokens=1,
                sampling_options=SamplingOptions(temperature=0.5),
                stop_strings=(),
                use_sampling=True,
            ),
        )
    )
    assert sampled == 7

    with pytest.raises(NotImplementedError, match="top_logprobs"):
        Ds4Worker._require_logprob_support(
            SimpleNamespace(token_logprob=lambda token_id: 0.0),
            1,
        )

    class AwaitableEosEngine:
        def eos_token_id(self) -> object:
            async def value() -> int:
                return 11

            return value()

    assert run(worker._eos_token_id(AwaitableEosEngine())) == 11
    assert run(worker._token_probabilities(object(), object(), 0)) == []
    with pytest.raises(NotImplementedError, match="top_logprobs"):
        run(worker._token_probabilities(object(), SimpleNamespace(), 1))
    with pytest.raises(Ds4GenerationError, match="top_logprobs"):
        run(
            worker._token_probabilities(
                object(),
                SimpleNamespace(top_logprobs=lambda top_k: "bad"),
                1,
            )
        )

    alternative = worker._token_probability(
        {"id": 5, "probability": 0.25}, "scores"
    )
    assert (
        run(
            worker._chosen_token_probability(
                SimpleNamespace(), 5, [alternative]
            )
        )
        == 0.25
    )
    assert (
        run(
            worker._chosen_token_probability(
                SimpleNamespace(), 6, [alternative]
            )
        )
        is None
    )
    with pytest.raises(NotImplementedError, match="token_logprob"):
        run(worker._chosen_token_probability(SimpleNamespace(), 6, []))

    assert (
        worker._token_probability(
            {"token_id": 1, "probability": 0.5, "logprob": -20.0},
            "scores",
        ).probability
        == 0.5
    )
    assert (
        worker._token_probability((2, "ignored", -0.2), "scores").token_id == 2
    )
    assert (
        worker._token_probability(
            SimpleNamespace(id=3, probability=0.75), "scores"
        ).probability
        == 0.75
    )
    with pytest.raises(Ds4GenerationError, match="token id"):
        worker._token_probability((1,), "scores")
    with pytest.raises(Ds4GenerationError, match="token id"):
        worker._token_id(True, "sample")
    with pytest.raises(Ds4GenerationError, match="probabilities"):
        worker._probability(2, "scores")
    with pytest.raises(Ds4GenerationError, match="log probabilities"):
        worker._probability_from_logprob("bad", "scores")

    NativeContextError = type("Ds4ContextError", (Exception,), {})

    class NativeContextTarget:
        def fail(self) -> None:
            raise NativeContextError("native context")

    with pytest.raises(Ds4ContextError, match="native context"):
        run(Ds4Worker._call_async(NativeContextTarget(), "fail"))

    assert Ds4Worker._bytes_value(bytearray(b"A"), "bytes") == b"A"
    assert Ds4Worker._bytes_value(memoryview(b"B"), "bytes") == b"B"
    with pytest.raises(Ds4GenerationError, match="must return bytes"):
        Ds4Worker._bytes_value("bad", "bytes")


def test_ds4_worker_reset_session_warns_and_returns_create_error(
    tmp_path: Path,
) -> None:
    logger = MagicMock(spec=Logger)
    worker = Ds4Worker(_worker_options(_model_file(tmp_path)), 16, logger)
    worker._closed = False

    class SnapshotFailSession:
        async def load_snapshot(self, snapshot: bytes) -> None:
            _ = snapshot
            raise RuntimeError("snapshot failed")

        async def invalidate(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    class CreateFailEngine:
        async def create_session(self, ctx_size: int) -> object:
            _ = ctx_size
            raise RuntimeError("create failed")

    worker._session = SnapshotFailSession()
    worker._engine = CreateFailEngine()

    error = run(worker._reset_session(invalidate=True, snapshot=b"bad"))

    assert isinstance(error, Ds4GenerationError)
    assert "create failed" in str(error)
    logger.warning.assert_called_once()


def test_ds4_model_sync_dsml_render_and_validation_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = Ds4Model(
        str(tmp_path / "not-loaded.gguf"),
        TransformerEngineSettings(auto_load_model=False),
    )

    with pytest.raises(ValueError, match="kv_disk_dir"):
        model._disk_cache_config({"kv_disk_dir": ""})
    with pytest.raises(ValueError):
        model._ds4_prompt_messages(cast(Any, [object()]), None, None)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            Ds4Model,
            "_message_role",
            staticmethod(lambda message: cast(MessageRole, "critic")),
        )
        with pytest.raises(ValueError, match="Unsupported DS4 message role"):
            model._ds4_prompt_messages(
                [Message(role=MessageRole.USER, content="hello")],
                None,
                None,
            )

    assert model._uses_dsml_tools(cast(Any, [object()]), None) is False

    special_path = tmp_path / "special"
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(Path, "exists", lambda _: True)
        patcher.setattr(Path, "is_dir", lambda _: False)
        patcher.setattr(Path, "is_file", lambda _: False)
        with pytest.raises(Ds4InvalidModel, match="regular file"):
            Ds4Model._validated_file_path(str(special_path), "DS4 model path")

    assert Ds4Model._optional_string_config({}, "key") is None
    assert Ds4Model._optional_string_config({"key": "value"}, "key") == "value"

    _install_binding(monkeypatch, _fake_binding())
    loaded_model = Ds4Model(str(_model_file(tmp_path)))
    manager = ToolManager.create_instance(
        available_toolsets=[MathToolSet(namespace="math")]
    )
    tokens = loaded_model._render_prompt_tokens(
        "hello",
        None,
        None,
        GenerationSettings(max_new_tokens=0),
        tool=manager,
    )

    assert tokens[0] == 40
    loaded_model.close()


def test_ds4_model_top_logprobs_manual_sampling_edges() -> None:
    assert Ds4Model._top_logprobs(True, None) == 0
    assert Ds4Model._top_logprobs(3, None) == 3
    with pytest.raises(ValueError, match="top_logprobs"):
        Ds4Model._top_logprobs(-1, None)
