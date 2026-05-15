from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import avalan.backends.ds4_native.availability as availability
from avalan.backends.ds4_native.engine import Engine
from avalan.backends.ds4_native.errors import (
    Ds4BackendUnavailable,
    Ds4ContextError,
    Ds4GenerationError,
    Ds4LoadError,
)
from avalan.backends.ds4_native.errors import (
    Ds4InvalidModel as AvalanDs4InvalidModel,
)
from avalan.backends.ds4_native.metadata import (
    DS4_API_COMMIT,
    DS4_REQUIRED_C_SYMBOLS,
)
from avalan.backends.ds4_native.types import (
    Backend,
    EngineOptions,
    SamplingOptions,
    ThinkMode,
)


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


@dataclass(frozen=True, slots=True)
class FakeNativeSamplingOptions:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int | None = None


class FakeNativeSession:
    def __init__(self, ctx_size: int) -> None:
        self.argmax_calls = 0
        self.close_calls = 0
        self.ctx = ctx_size
        self.invalidated = False
        self.pos = 0
        self.sample_options: FakeNativeSamplingOptions | None = None
        self.tokens: list[int] = []

    def close(self) -> None:
        self.close_calls += 1

    def sync(self, prompt_tokens: list[int]) -> None:
        self.tokens = list(prompt_tokens)
        self.pos = len(prompt_tokens)

    def eval(self, token_id: int) -> None:
        if token_id == 666:
            raise RuntimeError("eval failed")
        self.tokens.append(token_id)
        self.pos += 1

    def argmax(self) -> int:
        self.argmax_calls += 1
        return 40

    def argmax_excluding(self, token_id: int) -> int:
        return 41 if token_id == 40 else 40

    def sample(self, options: FakeNativeSamplingOptions) -> int:
        self.sample_options = options
        if options.seed == 13:
            raise RuntimeError("sample failed")
        return 42

    def rewind(self, pos: int) -> None:
        self.tokens = self.tokens[:pos]
        self.pos = pos

    def invalidate(self) -> None:
        self.invalidated = True
        self.tokens = []
        self.pos = 0

    def save_snapshot(self) -> bytes:
        token_bytes = ",".join(str(token_id) for token_id in self.tokens)
        return f"ctx={self.ctx};tokens={token_bytes}".encode()

    def load_snapshot(self, snapshot: bytes) -> None:
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


class FakeNativeEngine:
    instances: list["FakeNativeEngine"] = []

    def __init__(self, options: FakeNativeOptions) -> None:
        self.close_calls = 0
        self.eos_token_id = 100001
        self.has_mtp = True
        self.mtp_draft_tokens = 4
        self.options = options
        self.routed_quant_bits = 2
        self.sessions: list[FakeNativeSession] = []
        self.tokenization_calls: list[tuple[str, object, object]] = []
        self.instances.append(self)

    def close(self) -> None:
        self.close_calls += 1

    def create_session(self, ctx_size: int) -> FakeNativeSession:
        session = FakeNativeSession(ctx_size)
        self.sessions.append(session)
        return session

    def token_text(self, token_id: int) -> bytes:
        if token_id == 42:
            return b"answer"
        raise RuntimeError("invalid token")

    def tokenize_text(self, text: str) -> list[int]:
        self.tokenization_calls.append(("tokenize_text", text, None))
        return [100, len(text)]

    def tokenize_rendered_chat(self, text: str) -> list[int]:
        self.tokenization_calls.append(("tokenize_rendered_chat", text, None))
        return [101, len(text)]

    def chat_begin(self) -> list[int]:
        self.tokenization_calls.append(("chat_begin", None, None))
        return [1]

    def chat_append_message(
        self, tokens: list[int], role: str, content: str
    ) -> None:
        self.tokenization_calls.append((role, content, tuple(tokens)))
        role_token = {"system": 10, "user": 11, "assistant": 12}[role]
        tokens.extend([role_token, len(content)])

    def chat_append_assistant_prefix(
        self, tokens: list[int], think_mode: NativeThinkMode
    ) -> None:
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
        self.tokenization_calls.append((system or "", prompt, think_mode))
        prefix_token = {
            NativeThinkMode.NONE: 20,
            NativeThinkMode.HIGH: 21,
            NativeThinkMode.MAX: 22,
        }[think_mode]
        return [30, len(system or ""), len(prompt), prefix_token]


class FakeSession:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _fake_binding(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "__ds4_commit__": DS4_API_COMMIT,
        "__ds4_symbols__": DS4_REQUIRED_C_SYMBOLS,
        "Backend": NativeBackend,
        "Engine": FakeNativeEngine,
        "EngineOptions": FakeNativeOptions,
        "SamplingOptions": FakeNativeSamplingOptions,
        "ThinkMode": NativeThinkMode,
        "is_backend_available": lambda backend: backend == "metal",
        "think_mode_for_context": lambda mode, ctx_size: (
            NativeThinkMode.HIGH
            if mode == NativeThinkMode.MAX and ctx_size < 8192
            else mode
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_binding(monkeypatch: pytest.MonkeyPatch, binding: object) -> None:
    monkeypatch.setattr(availability, "import_module", lambda _: binding)


def test_ds4_engine_api_types_are_available() -> None:
    assert Backend.METAL.value == "metal"
    assert Backend.CUDA.value == "cuda"
    assert Backend.CPU.value == "cpu"
    assert ThinkMode.NONE.value == "none"
    assert ThinkMode.HIGH.value == "high"
    assert ThinkMode.MAX.value == "max"


def test_engine_opens_with_valid_options_through_fake_native_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    FakeNativeEngine.instances.clear()
    _install_binding(monkeypatch, _fake_binding())

    engine = Engine(
        EngineOptions(
            model_path=str(model_path),
            backend=Backend.METAL,
            mtp_draft_tokens=2,
            warm_weights=True,
        )
    )

    native = cast(FakeNativeEngine, engine.native)
    assert native is FakeNativeEngine.instances[0]
    assert native.options.model_path == str(model_path)
    assert native.options.backend is NativeBackend.METAL
    assert native.options.mtp_draft_tokens == 2
    assert native.options.warm_weights is True
    assert engine.options.model_path == str(model_path)
    engine.close()


def test_engine_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)

    engine.close()
    engine.close()

    assert native.close_calls == 1
    assert engine.closed is True


def test_engine_context_manager_closes_on_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())

    with Engine(EngineOptions(model_path=str(model_path))) as engine:
        native = cast(FakeNativeEngine, engine.native)

    assert native.close_calls == 1
    assert engine.closed is True


def test_engine_metadata_properties_use_fake_native_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))

    assert engine.routed_quant_bits == 2
    assert engine.has_mtp is True
    assert engine.mtp_draft_tokens == 4
    assert engine.eos_token_id == 100001
    engine.close()


def test_engine_tokenizes_text_with_native_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)

    assert engine.tokenize_text("hello") == [100, 5]
    assert engine.tokenize_rendered_chat("<|User|>hello") == [101, 13]
    assert native.tokenization_calls == [
        ("tokenize_text", "hello", None),
        ("tokenize_rendered_chat", "<|User|>hello", None),
    ]
    engine.close()


def test_engine_token_text_returns_bytes_without_decoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))

    assert engine.token_text(42) == b"answer"
    engine.close()


def test_single_turn_chat_prompt_encodes_with_think_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)

    tokens = engine.encode_chat_prompt("system", "hello", ThinkMode.HIGH)

    assert tokens == [30, 6, 5, 21]
    assert native.tokenization_calls == [
        ("system", "hello", NativeThinkMode.HIGH)
    ]
    engine.close()


def test_multi_turn_chat_prompt_appends_messages_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)

    tokens = engine.chat_begin()
    engine.chat_append_message(tokens, "system", "s")
    engine.chat_append_message(tokens, "user", "hello")
    engine.chat_append_message(tokens, "assistant", "hi")
    engine.chat_append_assistant_prefix(tokens, ThinkMode.MAX)

    assert tokens == [1, 10, 1, 11, 5, 12, 2, 22]
    assert native.tokenization_calls == [
        ("chat_begin", None, None),
        ("system", "s", (1,)),
        ("user", "hello", (1, 10, 1)),
        ("assistant", "hi", (1, 10, 1, 11, 5)),
        ("assistant_prefix", NativeThinkMode.MAX, (1, 10, 1, 11, 5, 12, 2)),
    ]
    engine.close()


def test_assistant_prefix_reflects_all_thinking_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))

    tokens: list[int] = []
    engine.chat_append_assistant_prefix(tokens, ThinkMode.NONE)
    engine.chat_append_assistant_prefix(tokens, ThinkMode.HIGH)
    engine.chat_append_assistant_prefix(tokens, ThinkMode.MAX)

    assert tokens == [20, 21, 22]
    engine.close()


def test_think_mode_for_context_uses_native_downgrade_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))

    assert engine.think_mode_for_context(ThinkMode.MAX, 4096) is ThinkMode.HIGH
    assert engine.think_mode_for_context(ThinkMode.MAX, 8192) is ThinkMode.MAX
    engine.close()


def test_engine_creates_session_with_context_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)

    with engine.create_session(8) as session:
        native_session = cast(FakeNativeSession, session.native)
        assert session.ctx == 8
        assert session.pos == 0
        assert session.tokens == ()

    assert native.sessions == [native_session]
    assert native_session.close_calls == 1
    assert session.closed is True
    engine.close()


def test_session_sync_accepts_prompt_tokens_and_updates_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)

    session.sync([1, 2, 3])

    assert session.pos == 3
    assert session.tokens == (1, 2, 3)
    engine.close()


def test_session_greedy_generation_calls_argmax(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)
    native_session = cast(FakeNativeSession, session.native)

    assert session.argmax() == 40

    assert native_session.argmax_calls == 1
    engine.close()


def test_session_argmax_excluding_uses_native_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)

    assert session.argmax_excluding(40) == 41
    assert session.argmax_excluding(99) == 40
    engine.close()


def test_session_sampling_maps_options_to_native_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)
    native_session = cast(FakeNativeSession, session.native)

    token_id = session.sample(
        SamplingOptions(
            temperature=0.7,
            top_k=12,
            top_p=0.9,
            min_p=0.1,
            seed=99,
        )
    )

    assert token_id == 42
    assert native_session.sample_options == FakeNativeSamplingOptions(
        temperature=0.7,
        top_k=12,
        top_p=0.9,
        min_p=0.1,
        seed=99,
    )
    engine.close()


def test_session_eval_advances_state_after_candidate_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)
    session.sync([1, 2])
    token_id = session.argmax()

    session.eval(token_id)

    assert session.pos == 3
    assert session.tokens == (1, 2, 40)
    engine.close()


def test_session_rewind_and_invalidate_reset_native_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)
    native_session = cast(FakeNativeSession, session.native)
    session.sync([1, 2, 3])

    session.rewind(1)
    assert session.pos == 1
    assert session.tokens == (1,)

    session.invalidate()
    assert native_session.invalidated is True
    assert session.pos == 0
    assert session.tokens == ()
    engine.close()


def test_session_snapshot_save_and_load_restore_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)
    session.sync([1, 2])

    snapshot = session.save_snapshot()
    session.eval(40)
    assert session.pos == 3
    assert session.tokens == (1, 2, 40)

    session.load_snapshot(snapshot)

    assert session.pos == 2
    assert session.tokens == (1, 2)
    engine.close()


def test_session_snapshot_rejects_mismatched_or_corrupt_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)

    with pytest.raises(Ds4GenerationError, match="context mismatch"):
        session.load_snapshot(b"ctx=16;tokens=1,2")
    with pytest.raises(Ds4GenerationError, match="corrupt snapshot"):
        session.load_snapshot(b"corrupt")

    engine.close()


def test_missing_model_path_raises_before_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "missing.gguf"

    def fail_import(_: str) -> object:
        raise AssertionError("missing path must be rejected before import")

    monkeypatch.setattr(availability, "import_module", fail_import)

    with pytest.raises(AvalanDs4InvalidModel, match="does not exist"):
        Engine(EngineOptions(model_path=str(model_path)))


def test_directory_model_path_raises_before_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_import(_: str) -> object:
        raise AssertionError("directory path must be rejected before import")

    monkeypatch.setattr(availability, "import_module", fail_import)

    with pytest.raises(AvalanDs4InvalidModel, match="must be a file"):
        Engine(EngineOptions(model_path=str(tmp_path)))


def test_native_load_failure_raises_ds4_load_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingEngine:
        def __init__(self, _: FakeNativeOptions) -> None:
            raise RuntimeError("native open failed")

    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding(Engine=FailingEngine))

    with pytest.raises(Ds4LoadError, match="native open failed"):
        Engine(EngineOptions(model_path=str(model_path)))


def test_native_invalid_model_failure_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Ds4InvalidModel(Exception):
        pass

    class FailingEngine:
        def __init__(self, _: FakeNativeOptions) -> None:
            raise Ds4InvalidModel("unsupported GGUF")

    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding(Engine=FailingEngine))

    with pytest.raises(
        AvalanDs4InvalidModel,
        match="unsupported GGUF",
    ):
        Engine(EngineOptions(model_path=str(model_path)))


def test_unsupported_backend_raises_before_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")

    def fail_import(_: str) -> object:
        raise AssertionError("unsupported backend must be rejected first")

    monkeypatch.setattr(availability, "import_module", fail_import)

    with pytest.raises(Ds4BackendUnavailable, match="Unsupported"):
        Engine(
            EngineOptions(
                model_path=str(model_path),
                backend=cast(Backend, "vulkan"),
            )
        )


@pytest.mark.parametrize("ctx_size", [0, -1])
def test_non_positive_session_context_raises_context_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ctx_size: int
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))

    with pytest.raises(Ds4ContextError, match="positive integer"):
        engine.create_session(ctx_size)
    engine.close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (SamplingOptions(top_k=-1), "top_k"),
        (SamplingOptions(top_p=-0.1), "top_p"),
        (SamplingOptions(top_p=1.1), "top_p"),
        (SamplingOptions(min_p=-0.1), "min_p"),
        (SamplingOptions(min_p=1.1), "min_p"),
        (SamplingOptions(temperature=-0.1), "temperature"),
    ],
)
def test_sampling_options_reject_invalid_ranges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: SamplingOptions,
    message: str,
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)

    with pytest.raises(ValueError, match=message):
        session.sample(options)
    engine.close()


def test_native_eval_error_maps_to_generation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)

    with pytest.raises(Ds4GenerationError, match="eval failed"):
        session.eval(666)
    engine.close()


def test_native_sample_error_maps_to_generation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)

    with pytest.raises(Ds4GenerationError, match="sample failed"):
        session.sample(SamplingOptions(seed=13))
    engine.close()


def test_session_calls_after_close_raise_closed_session_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)

    session.close()

    with pytest.raises(Ds4GenerationError, match="session is closed"):
        session.argmax()
    engine.close()


def test_engine_close_closes_tracked_live_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)
    session = FakeSession()

    engine._track_session(session)
    engine.close()

    assert session.close_calls == 1
    assert native.close_calls == 1


def test_chat_append_message_rejects_unsupported_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))

    with pytest.raises(ValueError, match="Unsupported DS4 chat role"):
        engine.chat_append_message([], "tool", "result")
    engine.close()


def test_invalid_token_id_maps_native_error_to_generation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))

    with pytest.raises(Ds4GenerationError, match="invalid token"):
        engine.token_text(100_000)
    engine.close()


def test_missing_think_mode_helper_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding(think_mode_for_context=None))
    engine = Engine(EngineOptions(model_path=str(model_path)))

    with pytest.raises(Ds4GenerationError, match="think_mode_for_context"):
        engine.think_mode_for_context(ThinkMode.MAX, 4096)
    engine.close()
