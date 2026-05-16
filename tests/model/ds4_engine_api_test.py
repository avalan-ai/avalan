from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import avalan.backends.ds4_native.availability as availability
from avalan.backends.ds4_native.engine import Engine, Session
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
class MinimalNativeOptions:
    model_path: str
    backend: NativeBackend


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

    def top_logprobs(self, top_k: int) -> list[tuple[int, float]]:
        return [(40, -0.25), (41, -1.25), (42, -2.25)][:top_k]

    def token_logprob(self, token_id: int) -> float:
        return -0.25 if token_id == 40 else -2.25

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


def test_engine_defaults_work_with_minimal_native_engine_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    FakeNativeEngine.instances.clear()
    _install_binding(
        monkeypatch, _fake_binding(EngineOptions=MinimalNativeOptions)
    )

    engine = Engine(EngineOptions(model_path=str(model_path)))

    native = cast(FakeNativeEngine, engine.native)
    assert native.options == MinimalNativeOptions(
        model_path=str(model_path), backend=NativeBackend.METAL
    )
    engine.close()


def test_engine_requested_mtp_options_require_binding_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(
        monkeypatch, _fake_binding(EngineOptions=MinimalNativeOptions)
    )

    with pytest.raises(Ds4BackendUnavailable, match="mtp_path"):
        Engine(
            EngineOptions(
                model_path=str(model_path),
                mtp_path=str(tmp_path / "mtp.gguf"),
            )
        )


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


def test_session_top_logprobs_returns_token_logprob_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)

    assert session.top_logprobs(2) == ((40, -0.25), (41, -1.25))
    assert session.token_logprob(40) == -0.25
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


def test_engine_rejects_binding_without_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding(Engine=None))

    with pytest.raises(Ds4LoadError, match="does not expose Engine"):
        Engine(EngineOptions(model_path=str(model_path)))


def test_engine_preserves_native_open_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingEngine:
        def __init__(self, _: FakeNativeOptions) -> None:
            raise Ds4LoadError("load failed")

    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding(Engine=FailingEngine))

    with pytest.raises(Ds4LoadError, match="load failed"):
        Engine(EngineOptions(model_path=str(model_path)))


def test_engine_maps_native_backend_unavailable_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    Ds4NativeBackendUnavailable = type(
        "Ds4BackendUnavailable", (Exception,), {}
    )

    class FailingEngine:
        def __init__(self, _: FakeNativeOptions) -> None:
            raise Ds4NativeBackendUnavailable("backend missing")

    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding(Engine=FailingEngine))

    with pytest.raises(Ds4BackendUnavailable, match="backend missing"):
        Engine(EngineOptions(model_path=str(model_path)))


def test_token_text_accepts_bytes_like_values_and_rejects_other_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)

    native.token_text = lambda _: bytearray(b"A")  # type: ignore[method-assign]
    assert engine.token_text(1) == b"A"
    native.token_text = lambda _: memoryview(b"B")  # type: ignore[method-assign]
    assert engine.token_text(1) == b"B"
    native.token_text = lambda _: "bad"  # type: ignore[method-assign]
    with pytest.raises(Ds4GenerationError, match="must return bytes"):
        engine.token_text(1)
    engine.close()


def test_think_mode_helper_errors_are_mapped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_helper(_: object, __: int) -> object:
        raise RuntimeError("helper failed")

    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(
        monkeypatch, _fake_binding(think_mode_for_context=fail_helper)
    )
    engine = Engine(EngineOptions(model_path=str(model_path)))

    with pytest.raises(Ds4GenerationError, match="helper failed"):
        engine.think_mode_for_context(ThinkMode.MAX, 4096)
    engine.close()


@pytest.mark.parametrize(
    ("creator", "error_type", "match"),
    [
        (lambda _: Ds4ContextError("bad context"), Ds4ContextError, "bad"),
        (lambda _: RuntimeError("boom"), Ds4ContextError, "boom"),
    ],
)
def test_create_session_maps_native_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    creator: object,
    error_type: type[Exception],
    match: str,
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)

    def fail_create(_: int) -> object:
        raise cast(Exception, cast(object, creator)(None))

    native.create_session = fail_create  # type: ignore[method-assign]
    with pytest.raises(error_type, match=match):
        engine.create_session(8)
    engine.close()


def test_create_session_rejects_none_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)
    native.create_session = lambda _: None  # type: ignore[method-assign]

    with pytest.raises(Ds4ContextError, match="returned no session"):
        engine.create_session(8)
    engine.close()


def test_engine_private_open_and_binding_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))

    with pytest.raises(Ds4LoadError, match="does not expose missing"):
        engine._generation_method("missing")
    native = cast(FakeNativeEngine, engine.native)
    native.fail = lambda: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        Ds4GenerationError("native generation")
    )
    with pytest.raises(Ds4GenerationError, match="native generation"):
        engine._call_generation("fail")

    engine._binding = None
    with pytest.raises(Ds4LoadError, match="binding is unavailable"):
        engine._binding_module()

    engine.close()
    with pytest.raises(Ds4LoadError, match="engine is closed"):
        engine.native


def test_metadata_value_validation_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)

    native.routed_quant_bits = lambda: 3  # type: ignore[method-assign]
    assert engine.routed_quant_bits == 3
    native.routed_quant_bits = None  # type: ignore[assignment]
    with pytest.raises(Ds4LoadError, match="metadata"):
        _ = engine.routed_quant_bits
    native.routed_quant_bits = True  # type: ignore[assignment]
    with pytest.raises(Ds4LoadError, match="must be an integer"):
        _ = engine.routed_quant_bits
    native.has_mtp = 1  # type: ignore[assignment]
    with pytest.raises(Ds4LoadError, match="must be a boolean"):
        _ = engine.has_mtp
    engine.close()


def test_engine_static_validation_helpers_cover_edges(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        Engine._coerce_think_mode(cast(ThinkMode, "unsupported"))
    assert Engine._token_list((1, 2), "tokens") == [1, 2]
    with pytest.raises(Ds4GenerationError, match="token ids"):
        Engine._token_list([1, True], "tokens")
    with pytest.raises(ValueError, match="token buffers"):
        Engine._validate_token_buffer([True])

    tokens = [1]
    Engine._replace_tokens_if_returned(tokens, (2, 3), "replace")
    assert tokens == [2, 3]

    assert Engine._think_mode_from_native(ThinkMode.HIGH) is ThinkMode.HIGH
    with pytest.raises(Ds4GenerationError, match="unsupported thinking mode"):
        Engine._think_mode_from_native(1)
    with pytest.raises(Ds4GenerationError, match="unsupported thinking mode"):
        Engine._think_mode_from_native("unsupported")

    normalized = Engine._normalize_options(
        EngineOptions(model_path=str(tmp_path / "model.gguf"), backend="metal")
    )
    assert normalized.backend is Backend.METAL

    with pytest.raises(AvalanDs4InvalidModel, match="required"):
        Engine._validate_model_path(EngineOptions(model_path=""))

    special_path = tmp_path / "special"
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(Path, "exists", lambda _: True)
        patcher.setattr(Path, "is_dir", lambda _: False)
        patcher.setattr(Path, "is_file", lambda _: False)
        with pytest.raises(AvalanDs4InvalidModel, match="regular file"):
            Engine._validate_model_path(
                EngineOptions(model_path=str(special_path))
            )


def test_native_engine_option_introspection_edges() -> None:
    options = EngineOptions(model_path="model.gguf")
    assert Engine._native_engine_options(SimpleNamespace(), options) is options

    class MissingBackendOptions:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

    with pytest.raises(Ds4LoadError, match="backend"):
        Engine._native_engine_options(
            SimpleNamespace(EngineOptions=MissingBackendOptions), options
        )

    class KwargsOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    assert Engine._native_engine_option_keys(vars) is None
    assert Engine._native_engine_option_keys(KwargsOptions) is None
    assert Engine._engine_option_is_requested("model_path", None) is True


def test_native_enum_value_fallbacks() -> None:
    assert (
        Engine._native_enum_value(SimpleNamespace(), "Backend", Backend.METAL)
        == "metal"
    )

    class BrokenNativeBackend:
        METAL = "native-metal"

        def __init__(self, value: str) -> None:
            raise ValueError(value)

    assert (
        Engine._native_enum_value(
            SimpleNamespace(Backend=BrokenNativeBackend),
            "Backend",
            Backend.METAL,
        )
        == "native-metal"
    )


def test_create_native_session_binding_fallbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    native = cast(FakeNativeEngine, engine.native)
    native.create_session = None  # type: ignore[method-assign]
    binding = engine._binding_module()
    setattr(binding, "Session", lambda native_engine, ctx_size: "session")

    assert engine._create_native_session(native, 8) == "session"

    setattr(binding, "Session", None)
    with pytest.raises(Ds4LoadError, match="create_session"):
        engine._create_native_session(native, 8)
    engine.close()


def test_context_error_mapping_helpers() -> None:
    NativeDs4ContextError = type("Ds4ContextError", (Exception,), {})

    with pytest.raises(Ds4ContextError, match="native context"):
        Engine._raise_mapped_context_error(
            NativeDs4ContextError("native context"), "sync"
        )
    with pytest.raises(Ds4ContextError, match="DS4 sync failed: boom"):
        Engine._raise_mapped_context_error(RuntimeError("boom"), "sync")


def test_session_value_and_result_validation_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)
    native_session = cast(FakeNativeSession, session.native)

    native_session.tokens = (1, 2)  # type: ignore[assignment]
    assert session.tokens == (1, 2)
    native_session.tokens = "bad"  # type: ignore[assignment]
    with pytest.raises(Ds4GenerationError, match="list or tuple"):
        _ = session.tokens
    native_session.tokens = [True]  # type: ignore[list-item]
    with pytest.raises(Ds4GenerationError, match="integer token IDs"):
        _ = session.tokens

    with pytest.raises(Ds4ContextError, match="non-negative"):
        session.rewind(-1)

    session.close()
    session.close()
    with pytest.raises(Ds4LoadError, match="does not expose missing"):
        new_session = engine.create_session(8)
        new_session._session_method("missing")

    engine.close()


def test_session_callable_values_and_missing_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "ds4flash.gguf"
    model_path.write_bytes(b"gguf")
    _install_binding(monkeypatch, _fake_binding())
    engine = Engine(EngineOptions(model_path=str(model_path)))
    session = engine.create_session(8)
    native_session = cast(FakeNativeSession, session.native)

    def fail_pos() -> object:
        raise RuntimeError("pos failed")

    native_session.pos = fail_pos  # type: ignore[assignment]
    with pytest.raises(Ds4GenerationError, match="pos failed"):
        _ = session.pos
    native_session.pos = None  # type: ignore[assignment]
    with pytest.raises(Ds4LoadError, match="unavailable"):
        _ = session.pos
    native_session.pos = True  # type: ignore[assignment]
    with pytest.raises(Ds4GenerationError, match="must be an integer"):
        _ = session.pos
    engine.close()


def test_session_static_result_helpers_cover_edges() -> None:
    with pytest.raises(Ds4GenerationError, match="one token id"):
        Session._token_result("bad", "sample")
    with pytest.raises(ValueError, match="top_k"):
        Session._validate_top_k(-1)
    with pytest.raises(Ds4GenerationError, match="token log probabilities"):
        Session._top_logprobs_result("bad", "top_logprobs")

    assert Session._token_score({"id": 1, "logprob": -0.1}, "scores") == (
        1,
        -0.1,
    )
    assert Session._token_score((2, "ignored", -0.2), "scores") == (2, -0.2)
    assert Session._token_score(
        SimpleNamespace(token_id=3, logprob=-0.3), "scores"
    ) == (3, -0.3)
    with pytest.raises(Ds4GenerationError, match="token ids"):
        Session._token_score((1,), "scores")
    with pytest.raises(Ds4GenerationError, match="token ids"):
        Session._token_score(
            SimpleNamespace(token_id=True, logprob=-0.1), "scores"
        )
    with pytest.raises(Ds4GenerationError, match="numeric"):
        Session._logprob_result(True, "logprob")

    assert Session._snapshot_bytes(bytearray(b"A"), "snapshot") == b"A"
    assert Session._snapshot_bytes(memoryview(b"B"), "snapshot") == b"B"
    with pytest.raises(Ds4GenerationError, match="must return bytes"):
        Session._snapshot_bytes("bad", "snapshot")

    with pytest.raises(ValueError, match="seed"):
        Session._normalize_sampling_options(
            SamplingOptions(seed=cast(int, True))
        )
    options = SamplingOptions()
    assert (
        Session._native_sampling_options(
            SimpleNamespace(SamplingOptions=None), options
        )
        is options
    )
