from asyncio import run
from collections.abc import Awaitable
from logging import WARNING
from pathlib import Path
from typing import cast

import pytest
from _ds4_integration import (
    DS4_SMOKE_PROMPT,
    ds4_settings,
    forbid_hugging_face_tokenizer,
    greedy_settings,
    require_ds4_available,
    require_ds4_model_path,
    stream_chunks,
)

from avalan.backends.ds4_native.errors import (
    Ds4BackendUnavailable,
    Ds4ContextError,
    Ds4InvalidModel,
)
from avalan.model.nlp.text.ds4 import _CPU_WARNING, Ds4Model

_LABEL = "DS4 real integration tests"
_REUSE_PROMPT = "Name one primary color."
_SHORT_PROMPT = "Answer with one short word."


def _require_real_model_path() -> Path:
    path = require_ds4_model_path(_LABEL)
    require_ds4_available(_LABEL)
    return path


async def _session_value(model: Ds4Model, name: str) -> object:
    session = model._ds4_worker()._require_session()
    value = getattr(session, name)
    if isinstance(value, Awaitable):
        return await value
    return value


async def _session_tokens(model: Ds4Model) -> list[int]:
    value = await _session_value(model, "tokens")
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and all(
        isinstance(token_id, int) for token_id in value
    ):
        return cast(list[int], value)
    pytest.fail("DS4 integration session did not expose token IDs.")


def test_ds4_real_model_loads_and_counts_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_hugging_face_tokenizer(monkeypatch, _LABEL)
    model_path = _require_real_model_path()

    with Ds4Model(str(model_path), ds4_settings()) as model:
        assert model.input_token_count(_SHORT_PROMPT) > 0
        assert model.tokenizer is None


def test_ds4_real_model_greedy_generation_streams_and_returns_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_hugging_face_tokenizer(monkeypatch, _LABEL)
    model_path = _require_real_model_path()

    async def scenario() -> tuple[list[str], str, int]:
        with Ds4Model(str(model_path), ds4_settings()) as model:
            chunks = await stream_chunks(
                model, DS4_SMOKE_PROMPT, max_new_tokens=4
            )
            response = await model(
                DS4_SMOKE_PROMPT,
                settings=greedy_settings(
                    max_new_tokens=4,
                    use_async_generator=False,
                ),
            )
            return chunks, await response.to_str(), response.input_token_count

    chunks, text, input_token_count = run(scenario())

    assert chunks
    assert "".join(chunks).strip()
    assert text.strip()
    assert input_token_count > 0


def test_ds4_real_model_second_call_syncs_prompt_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_hugging_face_tokenizer(monkeypatch, _LABEL)
    model_path = _require_real_model_path()

    async def scenario() -> tuple[bool, list[int], list[int], list[int]]:
        with Ds4Model(str(model_path), ds4_settings()) as model:
            settings = greedy_settings(
                max_new_tokens=1,
                use_async_generator=False,
            )
            prompt_tokens = await model._render_prompt_tokens_async(
                _REUSE_PROMPT,
                None,
                None,
                settings,
                tool=None,
            )
            session_before = model._ds4_worker()._require_session()

            first_response = await model(_REUSE_PROMPT, settings=settings)
            await first_response.to_str()
            first_tokens = await _session_tokens(model)
            session_after_first = model._ds4_worker()._require_session()

            second_response = await model(_REUSE_PROMPT, settings=settings)
            await second_response.to_str()
            second_tokens = await _session_tokens(model)
            session_after_second = model._ds4_worker()._require_session()

            return (
                session_before is session_after_first is session_after_second,
                prompt_tokens,
                first_tokens,
                second_tokens,
            )

    reused_session, prompt_tokens, first_tokens, second_tokens = run(
        scenario()
    )

    assert reused_session is True
    assert prompt_tokens
    assert first_tokens[: len(prompt_tokens)] == prompt_tokens
    assert second_tokens[: len(prompt_tokens)] == prompt_tokens


def test_ds4_real_model_invalid_path_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    forbid_hugging_face_tokenizer(monkeypatch, _LABEL)
    _require_real_model_path()

    with pytest.raises(Ds4InvalidModel, match="does not exist"):
        Ds4Model(str(tmp_path / "missing.gguf"), ds4_settings())


def test_ds4_real_model_cpu_backend_warns_debug_reference(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_hugging_face_tokenizer(monkeypatch, _LABEL)
    model_path = _require_real_model_path()
    model: Ds4Model | None = None

    with caplog.at_level(WARNING):
        try:
            model = Ds4Model(
                str(model_path),
                ds4_settings(native_backend="cpu"),
            )
        except Ds4BackendUnavailable as error:
            if "cpu" not in str(error).lower():
                raise
        finally:
            if model is not None:
                model.close()

    assert _CPU_WARNING in caplog.text


def test_ds4_real_model_context_too_small_raises_context_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_hugging_face_tokenizer(monkeypatch, _LABEL)
    model_path = _require_real_model_path()

    async def scenario() -> None:
        model: Ds4Model | None = None
        try:
            model = Ds4Model(str(model_path), ds4_settings(ctx_size=1))
            response = await model(
                _SHORT_PROMPT,
                settings=greedy_settings(max_new_tokens=1),
            )
            await response.to_str()
        finally:
            if model is not None:
                model.close()

    with pytest.raises(Ds4ContextError, match="context|ctx_size"):
        run(scenario())
