"""Load checked-in reasoning-summary scripts for direct unit testing."""

import sys
from hashlib import sha256
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any, cast

_MISSING_MODULE = object()
_LOADER_PATH = Path(__file__).resolve()
_LOADER_CONTENT_SHA256 = sha256(_LOADER_PATH.read_bytes()).hexdigest()
_HELPER_PATH = (
    _LOADER_PATH.parents[1] / "scripts" / "reasoning_summary_json.py"
).resolve()
_HELPER_CONTENT_SHA256 = sha256(_HELPER_PATH.read_bytes()).hexdigest()
_PATH_IDENTITY = f"{_LOADER_PATH}\0{_HELPER_PATH}"
_PROCESS_STATE_KEY = (
    "_avalan_reasoning_summary_loader_state_"
    f"{sha256(_PATH_IDENTITY.encode('utf-8')).hexdigest()}_"
    f"{_LOADER_CONTENT_SHA256}"
)


def _new_process_state() -> dict[str, Any]:
    """Return an uninitialized process-shared loader registry."""
    return {
        "lock": RLock(),
        "loader_path": str(_LOADER_PATH),
        "loader_content_sha256": _LOADER_CONTENT_SHA256,
        "script_load_sequence": 0,
        "helper": None,
        "helper_path": None,
        "helper_content_sha256": None,
        "helper_module_name": None,
    }


def _load_process_state() -> dict[str, Any]:
    """Return one registry shared by every physical loader alias."""
    state = cast(
        dict[str, Any],
        sys.__dict__.setdefault(_PROCESS_STATE_KEY, _new_process_state()),
    )
    lock = state.get("lock")
    assert lock is not None
    with lock:
        assert state.get("loader_path") == str(_LOADER_PATH)
        assert state.get("loader_content_sha256") == _LOADER_CONTENT_SHA256
        assert type(state.get("script_load_sequence")) is int
    return state


_PROCESS_STATE = _load_process_state()
_PROCESS_LOCK = cast(Any, _PROCESS_STATE["lock"])


def _module_registry() -> dict[str, Any]:
    """Return ``sys.modules`` with its runtime-permitted ``None`` values."""
    return cast(dict[str, Any], sys.modules)


def _module_entry(name: str) -> tuple[bool, object]:
    """Capture one module entry without conflating missing and ``None``."""
    registry = _module_registry()
    if name in registry:
        return True, registry[name]
    return False, _MISSING_MODULE


def _restore_module_entry(
    name: str,
    previous: tuple[bool, object],
) -> None:
    """Restore one module entry exactly to its captured state."""
    existed, value = previous
    registry = _module_registry()
    if existed:
        registry[name] = value
    else:
        registry.pop(name, None)


def _fresh_module_name(base_name: str) -> str:
    """Return an unused module name without replacing an existing entry."""
    registry = _module_registry()
    candidate = base_name
    collision_sequence = 0
    while candidate in registry:
        collision_sequence += 1
        candidate = f"{base_name}_{collision_sequence}"
    return candidate


def _execute_module(spec: ModuleSpec, module: ModuleType) -> None:
    """Execute a module through its checked import specification."""
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _load_reasoning_summary_json() -> ModuleType:
    """Load and retain the physically verified shared JSON helper."""
    path = _HELPER_PATH
    content_sha256 = sha256(path.read_bytes()).hexdigest()
    assert content_sha256 == _HELPER_CONTENT_SHA256
    with _PROCESS_LOCK:
        existing = _PROCESS_STATE.get("helper")
        if existing is not None:
            assert isinstance(existing, ModuleType)
            assert _PROCESS_STATE.get("helper_path") == str(path)
            assert (
                _PROCESS_STATE.get("helper_content_sha256") == content_sha256
            )
            existing_path = getattr(existing, "__file__", None)
            assert existing_path is not None
            assert Path(existing_path).resolve() == path
            return existing

        path_digest = sha256(str(path).encode("utf-8")).hexdigest()[:16]
        module_name = _fresh_module_name(
            "_avalan_reasoning_summary_json_"
            f"{path_digest}_{content_sha256[:16]}"
        )
        spec = spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        registry = _module_registry()
        registry[module_name] = module
        try:
            _execute_module(spec, module)
        except BaseException:
            registry.pop(module_name, None)
            raise
        loaded_path = getattr(module, "__file__", None)
        assert loaded_path is not None
        assert Path(loaded_path).resolve() == path
        assert sha256(path.read_bytes()).hexdigest() == content_sha256
        _PROCESS_STATE["helper"] = module
        _PROCESS_STATE["helper_path"] = str(path)
        _PROCESS_STATE["helper_content_sha256"] = content_sha256
        _PROCESS_STATE["helper_module_name"] = module_name
        return module


_REASONING_SUMMARY_JSON = _load_reasoning_summary_json()
StrictJsonError = cast(
    type[ValueError],
    cast(Any, _REASONING_SUMMARY_JSON).StrictJsonError,
)
DuplicateJsonObjectNameError = cast(
    type[ValueError],
    cast(Any, _REASONING_SUMMARY_JSON).DuplicateJsonObjectNameError,
)
NonFiniteJsonNumberError = cast(
    type[ValueError],
    cast(Any, _REASONING_SUMMARY_JSON).NonFiniteJsonNumberError,
)


def strict_json_loads(source: str) -> object:
    """Parse JSON through the shared duplicate-rejecting helper."""
    return cast(Any, _REASONING_SUMMARY_JSON).strict_json_loads(source)


def canonical_json_pointer(path: tuple[str | int, ...]) -> str:
    """Return a canonical pointer through the shared JSON helper."""
    return cast(
        str, cast(Any, _REASONING_SUMMARY_JSON).canonical_json_pointer(path)
    )


def typed_json_path(
    path: tuple[str | int, ...],
) -> tuple[tuple[str, str | int], ...]:
    """Return a typed path identity through the shared JSON helper."""
    return cast(
        tuple[tuple[str, str | int], ...],
        cast(Any, _REASONING_SUMMARY_JSON).typed_json_path(path),
    )


def json_mapping_entries(
    value: object,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str, tuple[str | int, ...], tuple[str, ...]], ...]:
    """Return mapping entries through the shared JSON helper."""
    return cast(
        tuple[tuple[str, tuple[str | int, ...], tuple[str, ...]], ...],
        cast(Any, _REASONING_SUMMARY_JSON).json_mapping_entries(value, path),
    )


def load_reasoning_summary_script(name: str) -> ModuleType:
    """Load one repository script without making ``scripts`` a package."""
    assert isinstance(name, str)
    assert name in {
        "benchmark_reasoning_summary",
        "verify_reasoning_summary_acceptance",
    }

    path = (_LOADER_PATH.parents[1] / "scripts" / f"{name}.py").resolve()
    path_digest = sha256(str(path).encode("utf-8")).hexdigest()[:16]
    content_sha256 = sha256(path.read_bytes()).hexdigest()
    with _PROCESS_LOCK:
        assert _PROCESS_STATE.get("helper") is _REASONING_SUMMARY_JSON
        assert _PROCESS_STATE.get("helper_path") == str(
            Path(cast(str, _REASONING_SUMMARY_JSON.__file__)).resolve()
        )
        assert sha256(
            Path(cast(str, _REASONING_SUMMARY_JSON.__file__)).read_bytes()
        ).hexdigest() == _PROCESS_STATE.get("helper_content_sha256")

        registry = _module_registry()
        while True:
            sequence = cast(int, _PROCESS_STATE["script_load_sequence"]) + 1
            _PROCESS_STATE["script_load_sequence"] = sequence
            module_name = (
                f"_avalan_test_{name}_{path_digest}_"
                f"{content_sha256[:16]}_{sequence}"
            )
            if module_name not in registry:
                break

        spec = spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        script_entry = _module_entry(module_name)
        assert not script_entry[0]
        helper_module_name = cast(str, _PROCESS_STATE["helper_module_name"])
        private_helper_entry = _module_entry(helper_module_name)
        public_helper_entry = _module_entry("reasoning_summary_json")
        registry[module_name] = module
        registry[helper_module_name] = _REASONING_SUMMARY_JSON
        registry["reasoning_summary_json"] = _REASONING_SUMMARY_JSON
        try:
            _execute_module(spec, module)
        except BaseException:
            _restore_module_entry(module_name, script_entry)
            raise
        finally:
            _restore_module_entry(helper_module_name, private_helper_entry)
            _restore_module_entry(
                "reasoning_summary_json", public_helper_entry
            )
        return module
