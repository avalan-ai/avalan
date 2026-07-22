from argparse import Namespace
from asyncio import run
from contextlib import AsyncExitStack
from logging import Logger
from sys import modules
from types import ModuleType
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from avalan.backends.ds4_native import Ds4BackendUnavailable
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
from avalan.model.modalities import text as text_module
from avalan.model.modalities.text import (
    TextGenerationModality,
    TextQuestionAnsweringModality,
    TextSequenceClassificationModality,
    TextSequenceToSequenceModality,
    TextTokenClassificationModality,
    TextTranslationModality,
    _get_ds4_model,
    _get_mlx_model,
    _normalize_backend,
    _resolve_model_class,
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

    @classmethod
    def is_available(cls) -> bool:
        return True


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


def test_normalize_backend_preserves_unknown_string() -> None:
    assert _normalize_backend("custom") == "custom"


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

    monkeypatch.setitem(modules, "avalan.model.nlp.text.mlxlm", None)
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


def test_get_mlx_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _get_mlx_model.cache_clear()
    monkeypatch.setattr(
        "avalan.model.modalities.text.find_spec", lambda name: object()
    )

    class UnavailableLoader:
        @classmethod
        def is_available(cls) -> bool:
            return False

    module_name = "avalan.model.nlp.text.mlxlm"
    stub = ModuleType(module_name)
    stub.MlxLmModel = UnavailableLoader
    monkeypatch.setitem(modules, module_name, stub)
    assert _get_mlx_model() is None


@pytest.mark.parametrize(
    "message",
    ("missing", "The binding marked itself unsafe."),
)
def test_get_ds4_model_binding_unavailable(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    _get_ds4_model.cache_clear()

    def fail_import() -> None:
        raise Ds4BackendUnavailable(message)

    monkeypatch.setattr(
        "avalan.model.modalities.text.import_compatible_binding",
        fail_import,
    )

    assert _get_ds4_model() is None


def test_get_ds4_model_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_ds4_model.cache_clear()
    monkeypatch.setattr(
        "avalan.model.modalities.text.import_compatible_binding",
        lambda: object(),
    )
    monkeypatch.setitem(modules, "avalan.model.nlp.text.ds4", None)

    assert _get_ds4_model() is None


def test_get_ds4_model_missing_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_ds4_model.cache_clear()
    monkeypatch.setattr(
        "avalan.model.modalities.text.import_compatible_binding",
        lambda: object(),
    )
    module_name = "avalan.model.nlp.text.ds4"
    stub = ModuleType(module_name)
    monkeypatch.setitem(modules, module_name, stub)

    assert _get_ds4_model() is None


def test_get_ds4_model_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _get_ds4_model.cache_clear()
    monkeypatch.setattr(
        "avalan.model.modalities.text.import_compatible_binding",
        lambda: object(),
    )
    module_name = "avalan.model.nlp.text.ds4"
    stub = ModuleType(module_name)
    stub.Ds4Model = RecordingLoader
    monkeypatch.setitem(modules, module_name, stub)

    assert _get_ds4_model() is RecordingLoader


def test_get_ds4_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _get_ds4_model.cache_clear()
    monkeypatch.setattr(
        "avalan.model.modalities.text.import_compatible_binding",
        lambda: object(),
    )

    class UnavailableLoader:
        @classmethod
        def is_available(cls) -> bool:
            return False

    module_name = "avalan.model.nlp.text.ds4"
    stub = ModuleType(module_name)
    stub.Ds4Model = UnavailableLoader
    monkeypatch.setitem(modules, module_name, stub)

    assert _get_ds4_model() is None


def test_resolve_model_class_imports_and_caches_any_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "tests.model.stub_generation_model"

    class ImportedModel:
        pass

    stub = ModuleType(module_name)
    stub.ImportedModel = ImportedModel
    monkeypatch.setitem(modules, module_name, stub)
    monkeypatch.setattr(text_module, "TextGenerationModel", Any)

    result = _resolve_model_class(
        "TextGenerationModel", module_name, "ImportedModel"
    )

    assert result is ImportedModel
    assert text_module.TextGenerationModel is ImportedModel


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


def test_text_generation_load_engine_ds4(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_ds4_model",
        lambda: RecordingLoader,
    )
    settings = TransformerEngineSettings(backend=Backend.DS4)
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


def test_text_generation_load_engine_ds4_string_backend(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_ds4_model",
        lambda: RecordingLoader,
    )
    settings = TransformerEngineSettings(
        backend=cast(Backend, Backend.DS4.value)
    )
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


def test_text_generation_load_engine_ds4_missing_dependency(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_ds4_model", lambda: None
    )
    settings = TransformerEngineSettings(backend=Backend.DS4)
    logger = MagicMock(spec=Logger)
    exit_stack = AsyncExitStack()

    with pytest.raises(ModuleNotFoundError, match=r"avalan\[ds4\]"):
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
        instructions="provider",
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
    assert text_params.instructions == "provider"
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
            capability=None,
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
    capability = object()
    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
            capability=capability,
        )
    )
    assert result == "result"
    assert len(model.calls) == 1
    _, kwargs = model.calls[0]
    assert set(kwargs) == {
        "system_prompt",
        "developer_prompt",
        "settings",
        "capability",
    }
    assert kwargs["capability"] is capability


def test_text_generation_call_ds4_branch_avoids_tokenizer_kwargs(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyDs4Model:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        @property
        def tokenizer(self) -> DummyTokenizer:
            raise AssertionError("DS4 call path should not use a tokenizer")

        async def __call__(self, *args: Any, **kwargs: Any) -> str:
            self.calls.append((args, kwargs))
            return "result"

    monkeypatch.setattr(
        "avalan.model.modalities.text._get_mlx_model",
        lambda: None,
    )
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_ds4_model",
        lambda: DummyDs4Model,
    )
    modality = TextGenerationModality()
    model = DummyDs4Model()
    text_params = OperationTextParameters(
        manual_sampling=True,
        pick_tokens=3,
        stop_on_keywords=["DONE"],
        system_prompt="sys",
        developer_prompt="dev",
        skip_special_tokens=True,
    )
    operation = make_operation(
        modality=Modality.TEXT_GENERATION,
        text_params=text_params,
    )
    capability = object()

    result = run(
        modality(
            local_engine_uri,
            model,
            operation,
            capability=capability,
        )
    )

    assert result == "result"
    assert len(model.calls) == 1
    args, kwargs = model.calls[0]
    assert args == ("prompt",)
    assert set(kwargs) == {
        "system_prompt",
        "developer_prompt",
        "settings",
        "manual_sampling",
        "pick",
        "capability",
    }
    assert kwargs["system_prompt"] == "sys"
    assert kwargs["developer_prompt"] == "dev"
    assert kwargs["settings"] == operation.generation_settings
    assert kwargs["manual_sampling"] is True
    assert kwargs["pick"] == 3
    assert kwargs["capability"] is capability


def test_text_generation_call_ds4_branch_forwards_instructions(
    local_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyDs4Model(DummyModel):
        pass

    monkeypatch.setattr(
        "avalan.model.modalities.text._get_mlx_model",
        lambda: None,
    )
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_ds4_model",
        lambda: DummyDs4Model,
    )
    modality = TextGenerationModality()
    model = DummyDs4Model()
    operation = make_operation(
        modality=Modality.TEXT_GENERATION,
        text_params=OperationTextParameters(instructions="provider"),
    )

    result = run(modality(local_engine_uri, model, operation))

    assert result == "result"
    assert len(model.calls) == 1
    _, kwargs = model.calls[0]
    assert kwargs["instructions"] == "provider"


def test_text_generation_call_remote_does_not_probe_mlx(
    remote_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_mlx_lookup() -> type[Any] | None:
        raise AssertionError("remote vendor call should not load mlx")

    def fail_ds4_lookup() -> type[Any] | None:
        raise AssertionError("remote vendor call should not load ds4")

    monkeypatch.setattr(
        "avalan.model.modalities.text._get_mlx_model",
        fail_mlx_lookup,
    )
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_ds4_model",
        fail_ds4_lookup,
    )
    modality = TextGenerationModality()
    model = DummyModel()
    operation = make_operation(
        modality=Modality.TEXT_GENERATION,
        text_params=OperationTextParameters(
            system_prompt="sys",
            developer_prompt="dev",
        ),
    )
    capability = object()

    result = run(
        modality(
            remote_engine_uri,
            model,
            operation,
            capability=capability,
        )
    )

    assert result == "result"
    assert len(model.calls) == 1
    _, kwargs = model.calls[0]
    assert set(kwargs) == {
        "system_prompt",
        "developer_prompt",
        "settings",
        "capability",
    }
    assert kwargs["capability"] is capability


def test_text_generation_call_remote_forwards_instructions(
    remote_engine_uri: EngineUri, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_mlx_model",
        lambda: None,
    )
    monkeypatch.setattr(
        "avalan.model.modalities.text._get_ds4_model",
        lambda: None,
    )
    modality = TextGenerationModality()
    model = DummyModel()
    operation = make_operation(
        modality=Modality.TEXT_GENERATION,
        text_params=OperationTextParameters(instructions="provider"),
    )

    result = run(modality(remote_engine_uri, model, operation))

    assert result == "result"
    assert len(model.calls) == 1
    _, kwargs = model.calls[0]
    assert kwargs["instructions"] == "provider"


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
