from os import environ
from pathlib import Path

import pytest

from avalan.entities import GenerationSettings, TransformerEngineSettings
from avalan.model.nlp.text.ds4 import Ds4Model
from avalan.model.transformer import TransformerModel

DS4_SMOKE_PROMPT = "Write a short greeting."


def require_ds4_model_path(label: str) -> Path:
    model_path = environ.get("AVALAN_DS4_MODEL")
    if not model_path:
        pytest.skip(f"Set AVALAN_DS4_MODEL to run {label}.")

    path = Path(model_path).expanduser()
    if not path.is_file():
        pytest.skip(f"AVALAN_DS4_MODEL does not point to a file: {path}.")
    return path


def require_ds4_available(label: str) -> None:
    if not Ds4Model.is_available():
        pytest.skip(f"Install avalan[ds4] to run {label}.")


def ds4_backend_config(
    *,
    ctx_size: int | None = None,
    native_backend: str | None = None,
) -> dict[str, object]:
    if ctx_size is None:
        ctx_value = environ.get("AVALAN_DS4_CTX", "4096")
        try:
            ctx_size = int(ctx_value)
        except ValueError as error:
            raise ValueError("AVALAN_DS4_CTX must be an integer.") from error

    return {
        "ctx_size": ctx_size,
        "native_backend": (
            native_backend
            if native_backend is not None
            else environ.get("AVALAN_DS4_BACKEND", "metal")
        ),
    }


def ds4_settings(
    *,
    ctx_size: int | None = None,
    native_backend: str | None = None,
) -> TransformerEngineSettings:
    return TransformerEngineSettings(
        backend_config=ds4_backend_config(
            ctx_size=ctx_size,
            native_backend=native_backend,
        )
    )


def forbid_hugging_face_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
) -> None:
    def fail_tokenizer(*_: object, **__: object) -> object:
        raise AssertionError(f"{label} must not load a tokenizer.")

    monkeypatch.setattr(
        TransformerModel, "_load_tokenizer_with_tokens", fail_tokenizer
    )


def greedy_settings(
    *,
    max_new_tokens: int = 8,
    use_async_generator: bool = True,
) -> GenerationSettings:
    return GenerationSettings(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_async_generator=use_async_generator,
    )


async def stream_chunks(
    model: Ds4Model,
    prompt: str = DS4_SMOKE_PROMPT,
    *,
    max_new_tokens: int = 8,
) -> list[str]:
    response = await model(
        prompt,
        settings=greedy_settings(
            max_new_tokens=max_new_tokens,
            use_async_generator=True,
        ),
    )
    chunks: list[str] = []
    async for chunk in response:
        if chunk:
            chunks.append(str(chunk))
    return chunks
