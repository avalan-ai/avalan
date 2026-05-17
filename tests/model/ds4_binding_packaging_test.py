from pathlib import Path
from types import SimpleNamespace

import pytest

import avalan.backends.ds4_native.availability as availability
from avalan.backends.ds4_native.availability import (
    binding_metadata,
    import_compatible_binding,
)
from avalan.backends.ds4_native.errors import (
    Ds4ApiVersionError,
    Ds4BackendUnavailable,
)
from avalan.backends.ds4_native.metadata import (
    DS4_API_COMMIT,
    DS4_API_VERSION,
    DS4_BINDING_IMPORT_NAME,
    DS4_BINDING_VERSION_FALLBACK,
    DS4_REQUIRED_C_SYMBOLS,
)


def _fake_binding(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "__ds4_commit__": DS4_API_COMMIT,
        "__ds4_symbols__": DS4_REQUIRED_C_SYMBOLS,
        "__version__": "0.1.0",
        "__ds4_native_backend__": "metal",
        "is_backend_available": lambda backend: backend == "metal",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_import_compatible_binding_uses_pyds4_without_opening_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def engine_open(*_: object, **__: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("availability checks must not open models")

    binding = _fake_binding(Engine=engine_open)

    def fake_import_module(module_name: str) -> object:
        assert module_name == DS4_BINDING_IMPORT_NAME
        return binding

    monkeypatch.setattr(availability, "import_module", fake_import_module)

    assert import_compatible_binding() is binding
    assert opened is False


def test_import_compatible_binding_accepts_installed_local_pyds4() -> None:
    pyds4 = pytest.importorskip("pyds4")

    binding = import_compatible_binding()
    metadata = binding_metadata(binding)
    capabilities = binding.capabilities()

    assert binding is pyds4
    assert Path(binding.__file__).resolve().name == "__init__.py"
    assert metadata.ds4_commit == DS4_API_COMMIT
    assert metadata.ds4_api_version == DS4_API_VERSION
    assert capabilities.ds4_commit == DS4_API_COMMIT
    assert set(DS4_REQUIRED_C_SYMBOLS) <= set(capabilities.required_symbols)


def test_binding_metadata_reports_stable_fallback_fields() -> None:
    metadata = binding_metadata(_fake_binding())

    assert metadata.module_name == DS4_BINDING_IMPORT_NAME
    assert metadata.binding_version == "0.1.0"
    assert metadata.ds4_commit == DS4_API_COMMIT
    assert metadata.ds4_api_version == DS4_API_VERSION
    assert metadata.native_backend_name == "metal"


def test_binding_metadata_falls_back_without_binding_version() -> None:
    metadata = binding_metadata(
        _fake_binding(__version__=None, __ds4_native_backend__=""),
        native_backend_name="cpu",
    )

    assert metadata.binding_version == DS4_BINDING_VERSION_FALLBACK
    assert metadata.native_backend_name == "cpu"


def test_import_compatible_binding_rejects_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_: str) -> object:
        raise ModuleNotFoundError("No module named 'pyds4'")

    monkeypatch.setattr(availability, "import_module", fail_import)

    with pytest.raises(
        Ds4BackendUnavailable,
        match=r"DS4 binding 'pyds4' is unavailable.*Install avalan\[ds4\]",
    ):
        import_compatible_binding()


def test_import_compatible_binding_rejects_unsafe_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _fake_binding(__ds4_import_safe__=False)
    monkeypatch.setattr(availability, "import_module", lambda _: binding)

    with pytest.raises(
        Ds4BackendUnavailable,
        match=r"unsafe.*Install avalan\[ds4\]",
    ):
        import_compatible_binding()


def test_import_compatible_binding_rejects_platform_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _fake_binding(is_backend_available=lambda backend: False)
    monkeypatch.setattr(availability, "import_module", lambda _: binding)

    with pytest.raises(
        Ds4BackendUnavailable,
        match=r"DS4 native backend 'metal'.*Install avalan\[ds4\]",
    ):
        import_compatible_binding(backend="metal")


def test_require_compatible_binding_rejects_invalid_api_version() -> None:
    with pytest.raises(Ds4ApiVersionError, match="__ds4_api_version__"):
        availability.require_compatible_binding(
            _fake_binding(__ds4_api_version__="1")
        )
