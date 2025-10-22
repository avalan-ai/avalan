from argparse import Namespace
from asyncio import run
from contextlib import AsyncExitStack
from logging import Logger
from sys import modules
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

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
from avalan.model.criteria import KeywordStoppingCriteria
from avalan.model.modalities.text import (
    TextGenerationModality,
    TextQuestionAnsweringModality,
    TextSequenceClassificationModality,
    TextSequenceToSequenceModality,
    TextTokenClassificationModality,
    TextTranslationModality,
    _get_mlx_model,
    _stopping_criteria,
)


class DummyTokenizer:
    def decode(self, token_id: int, skip_special_tokens: bool = False) -> str:
        return "token"


class DummyModel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.tokenizer = DummyTokenizer()

    async def __call__(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append((args, kwargs))
        return "result"


class RecordingLoader:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture()
def local_engine_uri() -> EngineUri:
    return EngineUri(
        host=None,
        port=None,
        user=None,
        password=None,
        vendor=None,
        model_id="local",
        params={},
    )


@pytest.fixture()
def remote_engine_uri() -> EngineUri:
    return EngineUri(
        host="api",
        port=443,
        user=None,
        password=None,
        vendor="openai",
        model_id="remote",
        params={},
    )


def make_operation(
    *,
    modality: Modality,
    text_params: OperationTextParameters | None = None,
    settings: GenerationSettings | None = None,
    parameters: OperationParameters | None = ...,  # type: ignore[assignment]
) -> Operation:
    final_settings = settings or GenerationSettings()
    if parameters is ...:
        parameters = (
            OperationParameters(text=text_params)
            if text_params is not None
            else OperationParameters()
        )
    return Operation(
        generation_settings=final_settings,
        input="prompt",
        modality=modality,
        parameters=parameters,
        requires_input=True,
    )


def test_stopping_criteria_without_keywords() -> None:
    operation = make_operation(
        modality=Modality.TEXT_GENERATION,
        text_params=OperationTextParameters(),
    )
    assert _stopping_criteria(operation, DummyModel()) is None


def test_stopping_criteria_with_keywords() -> None:
    operation = make_operation(
        modality=Modality.TEXT_GENERATION,
        text_params=OperationTextParameters(stop_on_keywords=["DONE"]),
    )
    result = _stopping_criteria(operation, DummyModel())
    assert isinstance(result, KeywordStoppingCriteria)


def test_get_mlx_model_no_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    _get_mlx_model.cache_clear()
    monkeypatch.setattr(
        "avalan.model.modalities.text.find_spec", lambda name: None
    )
    assert _get_mlx_model() is None


def test_get_mlx_model_module_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _get_mlx_model.cache_clear()
    monkeypatch.setattr(
        "avalan.model.modalities.text.find_spec", lambda name: object()
    )

    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("mlx_lm"):
            raise ModuleNotFoundError
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(modules, "avalan.model.nlp.text.mlxlm", raising=False)
    monkeypatch.delitem(modules, "mlx_lm", raising=False)
    monkeypatch.delitem(modules, "mlx_lm.sample_utils", raising=False)
    assert _get_mlx_model() is None


def test_get_mlx_model_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _get_mlx_model.cache_clear()
    monkeypatch.setattr(
        "avalan.model.modalities.text.find_spec", lambda name: object()
    )
    module_name = "avalan.model.nlp.text.mlxlm"
    stub = ModuleType(module_name)
    stub.MlxLmModel = RecordingLoader
    monkeypatch.setitem(modules, module_name, stub)
    assert _get_mlx_model() is RecordingLoader


def test_text_generation_load_engine_mlx(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_mlx_model",
        lambda: RecordingLoader,
    )
    settings = TransformerEngineSettings(backend=Backend.MLXLM)
    logger = MagicMock(spec=Logger)
    exit_stack = AsyncExitStack()
    loader = TextGenerationModality().load_engine(
        local_engine_uri,
        settings,
        logger,
        exit_stack,
    )
    assert isinstance(loader, RecordingLoader)
    assert loader.kwargs == {
        "model_id": "local",
        "settings": settings,
        "logger": logger,
    }


def test_text_generation_load_engine_mlx_missing_dependency(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_mlx_model", lambda: None
    )
    settings = TransformerEngineSettings(backend=Backend.MLXLM)
    logger = MagicMock(spec=Logger)
    exit_stack = AsyncExitStack()

    with pytest.raises(ModuleNotFoundError, match=r"avalan\[mlx\]"):
        TextGenerationModality().load_engine(
            local_engine_uri,
            settings,
            logger,
            exit_stack,
        )


def test_text_generation_load_engine_vllm(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "avalan.model.nlp.text.vllm"
    stub = ModuleType(module_name)
    stub.VllmModel = RecordingLoader
    monkeypatch.setitem(modules, module_name, stub)
    settings = TransformerEngineSettings(backend=Backend.VLLM)
    logger = MagicMock(spec=Logger)
    loader = TextGenerationModality().load_engine(
        local_engine_uri,
        settings,
        logger,
        AsyncExitStack(),
    )
    assert isinstance(loader, RecordingLoader)
    assert loader.kwargs["settings"] == settings


def test_text_generation_load_engine_default(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text.TextGenerationModel",
        RecordingLoader,
    )
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)
    loader = TextGenerationModality().load_engine(
        local_engine_uri,
        settings,
        logger,
        AsyncExitStack(),
    )
    assert isinstance(loader, RecordingLoader)
    assert loader.kwargs["logger"] is logger


def test_text_generation_get_operation_from_arguments() -> None:
    args = Namespace(
        display_tokens=5,
        stop_on_keyword=["STOP"],
        quiet=True,
        skip_special_tokens=False,
        system="sys",
        developer="dev",
    )
    operation = TextGenerationModality().get_operation_from_arguments(
        args,
        "question",
        GenerationSettings(),
    )
    text_params = operation.parameters["text"]
    assert text_params.manual_sampling == 5
    assert text_params.pick_tokens == 10
    assert text_params.skip_special_tokens is True
    assert text_params.stop_on_keywords == ["STOP"]
    assert text_params.system_prompt == "sys"
    assert text_params.developer_prompt == "dev"
    assert operation.input == "question"


def test_text_generation_call_local_uses_manual_sampling(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_mlx_model", lambda: None
    )
    modality = TextGenerationModality()
    model = DummyModel()
    text_params = OperationTextParameters(
        manual_sampling=True,
        pick_tokens=3,
        stop_on_keywords=["DONE"],
        system_prompt="sys",
        developer_prompt="dev",
        skip_special_tokens=False,
    )
    operation = make_operation(
        modality=Modality.TEXT_GENERATION,
        text_params=text_params,
    )
    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
            tool=None,
        )
    )
    assert result == "result"
    assert len(model.calls) == 1
    args, kwargs = model.calls[0]
    assert args == ("prompt",)
    assert isinstance(kwargs["stopping_criterias"][0], KeywordStoppingCriteria)
    assert kwargs["manual_sampling"] is True
    assert kwargs["pick"] == 3
    assert kwargs["system_prompt"] == "sys"
    assert kwargs["developer_prompt"] == "dev"


def test_text_generation_call_mlx_branch(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyMlxModel(DummyModel):
        pass

    monkeypatch.setattr(
        "avalan.model.modalities.text._get_mlx_model",
        lambda: DummyMlxModel,
    )
    modality = TextGenerationModality()
    model = DummyMlxModel()
    text_params = OperationTextParameters(
        system_prompt="sys",
        developer_prompt="dev",
    )
    operation = make_operation(
        modality=Modality.TEXT_GENERATION,
        text_params=text_params,
    )
    tool = object()
    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
            tool=tool,
        )
    )
    assert result == "result"
    assert len(model.calls) == 1
    _, kwargs = model.calls[0]
    assert set(kwargs) == {
        "system_prompt",
        "developer_prompt",
        "settings",
        "tool",
    }
    assert kwargs["tool"] is tool


def test_text_question_answering_load_engine_local(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text.QuestionAnsweringModel",
        RecordingLoader,
    )
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)
    loader = TextQuestionAnsweringModality().load_engine(
        local_engine_uri,
        settings,
        logger,
        AsyncExitStack(),
    )
    assert isinstance(loader, RecordingLoader)
    assert loader.kwargs["model_id"] == "local"


def test_text_question_answering_load_engine_remote(
    remote_engine_uri: EngineUri,
) -> None:
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)
    with pytest.raises(NotImplementedError):
        TextQuestionAnsweringModality().load_engine(
            remote_engine_uri,
            settings,
            logger,
            AsyncExitStack(),
        )


def test_text_question_answering_get_operation_from_arguments() -> None:
    args = Namespace(
        text_context="context",
        system="sys",
        developer="dev",
    )
    operation = TextQuestionAnsweringModality().get_operation_from_arguments(
        args,
        "question",
        GenerationSettings(),
    )
    text_params = operation.parameters["text"]
    assert text_params.context == "context"
    assert text_params.system_prompt == "sys"
    assert text_params.developer_prompt == "dev"


def test_text_question_answering_call_invokes_model(
    local_engine_uri: EngineUri,
) -> None:
    modality = TextQuestionAnsweringModality()
    model = DummyModel()
    text_params = OperationTextParameters(
        context="context",
        system_prompt="sys",
        developer_prompt="dev",
    )
    operation = make_operation(
        modality=Modality.TEXT_QUESTION_ANSWERING,
        text_params=text_params,
    )
    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
        )
    )
    assert result == "result"
    assert model.calls[0][1]["context"] == "context"
    assert model.calls[0][1]["system_prompt"] == "sys"
    assert model.calls[0][1]["developer_prompt"] == "dev"


def test_text_sequence_classification_load_engine_local(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text.SequenceClassificationModel",
        RecordingLoader,
    )
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)
    loader = TextSequenceClassificationModality().load_engine(
        local_engine_uri,
        settings,
        logger,
        AsyncExitStack(),
    )
    assert isinstance(loader, RecordingLoader)
    assert loader.kwargs["settings"] == settings


def test_text_sequence_classification_get_operation_from_arguments() -> None:
    operation = (
        TextSequenceClassificationModality().get_operation_from_arguments(
            Namespace(),
            "text",
            GenerationSettings(),
        )
    )
    assert operation.parameters is None
    assert operation.input == "text"


def test_text_sequence_classification_call(
    local_engine_uri: EngineUri,
) -> None:
    modality = TextSequenceClassificationModality()
    model = DummyModel()
    operation = make_operation(
        modality=Modality.TEXT_SEQUENCE_CLASSIFICATION,
        text_params=None,
        parameters=None,
    )
    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
        )
    )
    assert result == "result"
    assert model.calls[0][0] == ("prompt",)


def test_text_sequence_to_sequence_load_engine_local(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text.SequenceToSequenceModel",
        RecordingLoader,
    )
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)
    loader = TextSequenceToSequenceModality().load_engine(
        local_engine_uri,
        settings,
        logger,
        AsyncExitStack(),
    )
    assert isinstance(loader, RecordingLoader)
    assert loader.kwargs["logger"] is logger


def test_text_sequence_to_sequence_get_operation_from_arguments() -> None:
    args = Namespace(stop_on_keyword=["STOP"])
    operation = TextSequenceToSequenceModality().get_operation_from_arguments(
        args,
        "input",
        GenerationSettings(),
    )
    text_params = operation.parameters["text"]
    assert text_params.stop_on_keywords == ["STOP"]


def test_text_sequence_to_sequence_call(
    local_engine_uri: EngineUri,
) -> None:
    modality = TextSequenceToSequenceModality()
    model = DummyModel()
    text_params = OperationTextParameters(stop_on_keywords=["STOP"])
    operation = make_operation(
        modality=Modality.TEXT_SEQUENCE_TO_SEQUENCE,
        text_params=text_params,
    )
    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
        )
    )
    assert result == "result"
    assert isinstance(
        model.calls[0][1]["stopping_criterias"][0], KeywordStoppingCriteria
    )


def test_text_token_classification_load_engine_local(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text.TokenClassificationModel",
        RecordingLoader,
    )
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)
    loader = TextTokenClassificationModality().load_engine(
        local_engine_uri,
        settings,
        logger,
        AsyncExitStack(),
    )
    assert isinstance(loader, RecordingLoader)
    assert loader.kwargs["model_id"] == "local"


def test_text_token_classification_get_operation_from_arguments() -> None:
    args = Namespace(
        text_labeled_only=True,
        system="sys",
        developer="dev",
    )
    operation = TextTokenClassificationModality().get_operation_from_arguments(
        args,
        "input",
        GenerationSettings(),
    )
    text_params = operation.parameters["text"]
    assert text_params.labeled_only is True
    assert text_params.system_prompt == "sys"
    assert text_params.developer_prompt == "dev"


def test_text_token_classification_call(
    local_engine_uri: EngineUri,
) -> None:
    modality = TextTokenClassificationModality()
    model = DummyModel()
    text_params = OperationTextParameters(
        labeled_only=None,
        system_prompt="sys",
        developer_prompt="dev",
    )
    operation = make_operation(
        modality=Modality.TEXT_TOKEN_CLASSIFICATION,
        text_params=text_params,
    )
    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
        )
    )
    assert result == "result"
    kwargs = model.calls[0][1]
    assert kwargs["labeled_only"] is False
    assert kwargs["system_prompt"] == "sys"
    assert kwargs["developer_prompt"] == "dev"


def test_text_translation_load_engine_local(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text.TranslationModel",
        RecordingLoader,
    )
    settings = TransformerEngineSettings()
    logger = MagicMock(spec=Logger)
    loader = TextTranslationModality().load_engine(
        local_engine_uri,
        settings,
        logger,
        AsyncExitStack(),
    )
    assert isinstance(loader, RecordingLoader)
    assert loader.kwargs["settings"] == settings


def test_text_translation_get_operation_from_arguments() -> None:
    args = Namespace(
        text_to_lang="fr",
        text_from_lang="en",
        stop_on_keyword=["STOP"],
        skip_special_tokens=True,
    )
    operation = TextTranslationModality().get_operation_from_arguments(
        args,
        "input",
        GenerationSettings(),
    )
    text_params = operation.parameters["text"]
    assert text_params.language_destination == "fr"
    assert text_params.language_source == "en"
    assert text_params.stop_on_keywords == ["STOP"]
    assert text_params.skip_special_tokens is True


def test_text_translation_call(
    local_engine_uri: EngineUri,
) -> None:
    modality = TextTranslationModality()
    model = DummyModel()
    text_params = OperationTextParameters(
        language_destination="fr",
        language_source="en",
        stop_on_keywords=["STOP"],
        skip_special_tokens=True,
    )
    operation = make_operation(
        modality=Modality.TEXT_TRANSLATION,
        text_params=text_params,
    )
    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
        )
    )
    assert result == "result"
    kwargs = model.calls[0][1]
    assert kwargs["source_language"] == "en"
    assert kwargs["destination_language"] == "fr"
    assert kwargs["skip_special_tokens"] is True
    assert isinstance(kwargs["stopping_criterias"][0], KeywordStoppingCriteria)
