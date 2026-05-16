from .errors import Ds4ApiVersionError, Ds4BackendUnavailable
from .metadata import (
    DS4_API_COMMIT,
    DS4_API_VERSION,
    DS4_BACKEND_NAME,
    DS4_BINDING_IMPORT_NAME,
    DS4_BINDING_VERSION_FALLBACK,
    DS4_REQUIRED_C_SYMBOLS,
    DS4_SUPPORTED_NATIVE_BACKENDS,
)

from collections.abc import Callable, Collection
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

_UNAVAILABLE_SUFFIX = (
    "Install avalan[ds4] and ensure DS4 supports this platform. "
    "Supported production targets: macOS arm64 + Metal, Linux + CUDA. "
    "CPU mode is debug/reference only."
)


@dataclass(frozen=True, slots=True)
class Ds4BindingMetadata:
    """Report the stable metadata Avalan relies on for DS4 bindings."""

    module_name: str
    binding_version: str
    ds4_commit: str
    ds4_api_version: int | None
    native_backend_name: str


def _binding_symbols(binding: object) -> set[str]:
    symbols = getattr(binding, "__ds4_symbols__", None)
    if symbols is None:
        return {
            symbol
            for symbol in DS4_REQUIRED_C_SYMBOLS
            if hasattr(binding, symbol)
        }
    if isinstance(symbols, Collection) and not isinstance(
        symbols, (bytes, str)
    ):
        return {str(symbol) for symbol in symbols}
    raise Ds4ApiVersionError(
        "DS4 binding metadata __ds4_symbols__ must be a collection of "
        "public C symbol names."
    )


def _binding_version(binding: object) -> str:
    version = getattr(binding, "__version__", None)
    if version is None:
        version = getattr(binding, "__ds4_binding_version__", None)
    return (
        version if isinstance(version, str) else DS4_BINDING_VERSION_FALLBACK
    )


def _binding_api_version(binding: object) -> int | None:
    version = getattr(binding, "__ds4_api_version__", DS4_API_VERSION)
    if isinstance(version, int) or version is None:
        return version
    raise Ds4ApiVersionError(
        "DS4 binding metadata __ds4_api_version__ must be an integer or None."
    )


def _binding_native_backend_name(
    binding: object, fallback_backend_name: str
) -> str:
    native_backend_name = getattr(binding, "__ds4_native_backend__", None)
    if isinstance(native_backend_name, str) and native_backend_name:
        return native_backend_name
    return fallback_backend_name


def _unavailable_message(module_name: str, details: str = "") -> str:
    detail_text = f" {details}" if details else ""
    return (
        f"DS4 binding {module_name!r} is unavailable.{detail_text} "
        f"{_UNAVAILABLE_SUFFIX}"
    )


def _require_safe_import(binding: object, module_name: str) -> None:
    if getattr(binding, "__ds4_import_safe__", True) is False:
        raise Ds4BackendUnavailable(
            _unavailable_message(
                module_name, "The binding marked itself unsafe."
            )
        )


def require_compatible_binding(binding: object) -> None:
    """Validate that a DS4 binding matches Avalan's pinned API surface."""
    binding_commit = getattr(binding, "__ds4_commit__", None)
    if binding_commit != DS4_API_COMMIT:
        raise Ds4ApiVersionError(
            "DS4 binding API mismatch: expected DS4 C API commit "
            f"{DS4_API_COMMIT}, got {binding_commit!r}."
        )

    symbols = _binding_symbols(binding)
    missing = tuple(
        symbol for symbol in DS4_REQUIRED_C_SYMBOLS if symbol not in symbols
    )
    if missing:
        missing_symbols = ", ".join(missing)
        raise Ds4ApiVersionError(
            "DS4 binding API mismatch: missing required public C symbols "
            f"{missing_symbols}."
        )

    binding_api_version = _binding_api_version(binding)
    if binding_api_version != DS4_API_VERSION:
        raise Ds4ApiVersionError(
            "DS4 binding API mismatch: expected DS4 C API version "
            f"{DS4_API_VERSION!r}, got {binding_api_version!r}."
        )


def _backend_unavailable_reason(binding: object, backend: str) -> str:
    reason = getattr(binding, "backend_unavailable_reason", None)
    if not callable(reason):
        return ""
    reason_fn = cast(Callable[[str], object], reason)
    value = reason_fn(backend)
    return value if isinstance(value, str) else ""


def require_backend_available(binding: object, backend: str) -> None:
    """Validate that a DS4 native backend is supported by the binding."""
    assert backend, "A DS4 native backend name is required."
    if backend not in DS4_SUPPORTED_NATIVE_BACKENDS:
        supported = ", ".join(DS4_SUPPORTED_NATIVE_BACKENDS)
        raise Ds4BackendUnavailable(
            f"Unsupported DS4 native backend {backend!r}. "
            f"Supported native backends: {supported}."
        )

    available = getattr(binding, "is_backend_available", None)
    if callable(available):
        available_fn = cast(Callable[[str], object], available)
        if bool(available_fn(backend)):
            return
        reason = _backend_unavailable_reason(binding, backend)
        details = f" {reason}" if reason else ""
        raise Ds4BackendUnavailable(
            f"DS4 native backend {backend!r} is unavailable on this "
            f"platform.{details} {_UNAVAILABLE_SUFFIX}"
        )

    available_backends = getattr(binding, "__ds4_available_backends__", None)
    if available_backends is None:
        return
    if (
        isinstance(available_backends, Collection)
        and not isinstance(available_backends, (bytes, str))
        and backend in {str(item) for item in available_backends}
    ):
        return
    raise Ds4BackendUnavailable(
        f"DS4 native backend {backend!r} is unavailable on this platform. "
        f"{_UNAVAILABLE_SUFFIX}"
    )


def require_compatible_startup(binding: object, backend: str) -> None:
    """Validate DS4 binding compatibility and backend availability."""
    require_compatible_binding(binding)
    require_backend_available(binding, backend)


def binding_metadata(
    binding: object,
    *,
    module_name: str = DS4_BINDING_IMPORT_NAME,
    native_backend_name: str = DS4_BACKEND_NAME,
) -> Ds4BindingMetadata:
    """Return stable metadata for a compatible DS4 binding."""
    require_compatible_binding(binding)
    return Ds4BindingMetadata(
        module_name=module_name,
        binding_version=_binding_version(binding),
        ds4_commit=DS4_API_COMMIT,
        ds4_api_version=DS4_API_VERSION,
        native_backend_name=_binding_native_backend_name(
            binding, native_backend_name
        ),
    )


def import_compatible_binding(
    module_name: str = DS4_BINDING_IMPORT_NAME,
    *,
    backend: str | None = None,
) -> Any:
    """Import and validate a DS4 binding module without opening a model."""
    try:
        binding = import_module(module_name)
    except Exception as error:
        raise Ds4BackendUnavailable(
            _unavailable_message(module_name, str(error))
        ) from error
    _require_safe_import(binding, module_name)
    require_compatible_binding(binding)
    if backend is not None:
        require_backend_available(binding, backend)
    return binding
