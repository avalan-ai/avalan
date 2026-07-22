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
from avalan.entities import Message, MessageRole, MessageToolCall
from avalan.model.capability import ModelCapabilityCatalog
from avalan.model.nlp.text.ds4 import _CPU_WARNING, Ds4Model
from avalan.tool.dsml import DsmlTools
from avalan.tool.manager import ToolManager
from avalan.tool.math import MathToolSet

_LABEL = "DS4 real integration tests"
_REUSE_PROMPT = "Name one primary color."
_SHORT_PROMPT = "Answer with one short word."


def _model_capability(manager: ToolManager) -> ModelCapabilityCatalog:
    return ModelCapabilityCatalog.create(
        manager.export_model_capability_seed()
    )


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
                capability=None,
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


def test_ds4_real_model_tokenizes_exact_dsml_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_hugging_face_tokenizer(monkeypatch, _LABEL)
    model_path = _require_real_model_path()

    async def scenario() -> tuple[str, list[int]]:
        with Ds4Model(str(model_path), ds4_settings()) as model:
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
            parsed = DsmlTools.parse_generated_message(raw_dsml)
            assert parsed is not None
            assert len(parsed.calls) == 1
            call = parsed.calls[0]
            worker = model._ds4_worker()
            worker._remember_dsml_tool_replay(parsed)

            captured: list[str] = []
            original_tokenize = worker.tokenize_rendered_chat_async

            async def capture_rendered(text: str) -> list[int]:
                captured.append(text)
                return await original_tokenize(text)

            monkeypatch.setattr(
                worker, "tokenize_rendered_chat_async", capture_rendered
            )
            manager = ToolManager.create_instance(
                available_toolsets=[MathToolSet(namespace="math")]
            )
            tokens = await model._render_prompt_tokens_async(
                [
                    Message(role=MessageRole.USER, content="calculate"),
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
                None,
                None,
                greedy_settings(max_new_tokens=1),
                capability=_model_capability(manager),
            )
            return captured[-1], tokens

    rendered, tokens = run(scenario())

    assert tokens
    assert "<DSML｜tool_calls>" in rendered
    assistant_history = rendered.split("<｜Assistant｜>", 1)[1]
    assert "<｜DSML｜tool_calls>" not in assistant_history
    assert '<DSML｜parameter name="precision" string="false">2' in rendered


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
