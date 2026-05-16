from argparse import Namespace
from asyncio import run
from logging import getLogger
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _ds4_integration import DS4_SMOKE_PROMPT as _PROMPT
from _ds4_integration import ds4_backend_config as _backend_config
from _ds4_integration import (
    forbid_hugging_face_tokenizer as _forbid_tokenizer,
)
from _ds4_integration import (
    greedy_settings,
    require_ds4_available,
    require_ds4_model_path,
)
from _ds4_integration import stream_chunks as _stream_chunks

from avalan.cli.commands import model as model_cmds
from avalan.entities import TransformerEngineSettings
from avalan.model.nlp.text.ds4 import Ds4Model

_PHASE3_LABEL = "DS4 phase 3 smoke tests"


def _require_ds4_model_path() -> Path:
    return require_ds4_model_path(_PHASE3_LABEL)


def _require_ds4_available() -> None:
    require_ds4_available(_PHASE3_LABEL)


def _forbid_hugging_face_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_tokenizer(monkeypatch, "DS4 phase 3 smoke")


def test_ds4_phase3_real_model_streaming_response_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_hugging_face_tokenizer(monkeypatch)
    model_path = _require_ds4_model_path()
    _require_ds4_available()

    with Ds4Model(
        str(model_path),
        TransformerEngineSettings(backend_config=_backend_config()),
    ) as model:
        chunks = run(_stream_chunks(model))

        assert chunks
        assert "".join(chunks).strip()
        assert model.tokenizer is None


def test_ds4_phase3_real_model_non_streaming_response_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_hugging_face_tokenizer(monkeypatch)
    model_path = _require_ds4_model_path()
    _require_ds4_available()

    with Ds4Model(
        str(model_path),
        TransformerEngineSettings(backend_config=_backend_config()),
    ) as model:
        response = run(
            model(
                _PROMPT,
                settings=greedy_settings(
                    max_new_tokens=8,
                    use_async_generator=False,
                ),
            )
        )
        text = run(response.to_str())

        assert text.strip()
        assert response.input_token_count > 0
        assert model.tokenizer is None


def _model_run_args(model_path: Path) -> Namespace:
    backend_config = _backend_config()
    return Namespace(
        attention=None,
        backend="ds4",
        base_url=None,
        cache_strategy=None,
        chat_disable_thinking=True,
        developer=None,
        device="cpu",
        disable_loading_progress_bar=True,
        display_answer_height=12,
        display_answer_height_expand=False,
        display_events=False,
        display_pause=None,
        display_probabilities=False,
        display_probabilities_maximum=0.8,
        display_probabilities_sample_minimum=0.1,
        display_time_to_n_token=None,
        display_tokens=0,
        display_tools=False,
        display_tools_events=0,
        do_sample=False,
        ds4_ctx=backend_config["ctx_size"],
        ds4_mtp=None,
        ds4_mtp_draft=None,
        ds4_mtp_margin=None,
        ds4_native_backend=backend_config["native_backend"],
        ds4_quality=None,
        ds4_warm_weights=None,
        enable_gradient_calculation=False,
        input_file=None,
        loader_class="auto",
        low_cpu_mem_usage=False,
        max_new_tokens=8,
        min_p=None,
        model=str(model_path),
        no_reasoning=True,
        no_repl=True,
        output_hidden_states=False,
        parallel=None,
        quiet=True,
        reasoning_effort=None,
        reasoning_max_new_tokens=None,
        reasoning_stop_on_max_new_tokens=False,
        reasoning_tag=None,
        record=False,
        repetition_penalty=1.0,
        revision=None,
        sentence_transformer=False,
        skip_display_reasoning_time=False,
        skip_hub_access_check=True,
        skip_special_tokens=False,
        special_token=None,
        start_thinking=False,
        stop_on_keyword=None,
        subfolder=None,
        system=None,
        text_max_length=None,
        text_num_beams=1,
        token=None,
        tokenizer=None,
        tokenizer_subfolder=None,
        top_k=None,
        top_p=None,
        trust_remote_code=False,
        use_cache=True,
        weight_type="auto",
        temperature=0.0,
    )


def test_ds4_phase3_model_run_cli_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _forbid_hugging_face_tokenizer(monkeypatch)
    model_path = _require_ds4_model_path()
    _require_ds4_available()
    args = _model_run_args(model_path)
    console = MagicMock()
    theme = MagicMock()
    theme.icons = {"user_input": ">"}
    hub = MagicMock()
    hub.cache_dir = str(tmp_path)

    monkeypatch.setattr(model_cmds, "get_input", lambda *_, **__: _PROMPT)

    run(model_cmds.model_run(args, console, theme, hub, 5, getLogger()))

    output = "".join(
        str(call.args[0]) for call in console.print.call_args_list if call.args
    )
    assert output.strip()
