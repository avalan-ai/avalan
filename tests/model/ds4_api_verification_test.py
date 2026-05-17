from os import environ
from pathlib import Path
from subprocess import run
from types import SimpleNamespace

import pytest

from avalan.backends.ds4_native.availability import (
    require_backend_available,
    require_compatible_binding,
    require_compatible_startup,
)
from avalan.backends.ds4_native.errors import (
    Ds4ApiVersionError,
    Ds4BackendUnavailable,
)
from avalan.backends.ds4_native.metadata import (
    DS4_API_COMMIT,
    DS4_API_VERSION,
    DS4_ARGMAX_ADVANCES_SESSION,
    DS4_BACKEND_NAME,
    DS4_BINDING_IMPORT_NAME,
    DS4_PUBLIC_C_SYMBOLS,
    DS4_REQUIRED_C_SYMBOLS,
    DS4_SAMPLE_ADVANCES_SESSION,
    DS4_SESSION_EVAL_ADVANCES_SESSION,
    DS4_SESSION_THREAD_POLICY,
    DS4_SUPPORTED_NATIVE_BACKENDS,
)


def _fake_binding(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "capabilities": lambda: _fake_capabilities(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_capabilities(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "available_backends": ("metal",),
        "backend": "metal",
        "ds4_api_version": DS4_API_VERSION,
        "ds4_commit": DS4_API_COMMIT,
        "logprobs": True,
        "mtp": True,
        "payloads": True,
        "progress": True,
        "required_symbols": DS4_REQUIRED_C_SYMBOLS,
        "snapshots": True,
        "speculative_eval": True,
        "top_logprobs": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ds4_api_metadata_locks_verified_commit_and_symbols() -> None:
    assert DS4_BACKEND_NAME == "ds4"
    assert DS4_BINDING_IMPORT_NAME == "pyds4"
    assert DS4_API_COMMIT == "8809b90a1e3247389d7652b565ab6772e036f1ea"
    assert DS4_API_VERSION is None
    assert DS4_SUPPORTED_NATIVE_BACKENDS == ("metal", "cuda", "cpu")
    assert DS4_SAMPLE_ADVANCES_SESSION is False
    assert DS4_ARGMAX_ADVANCES_SESSION is False
    assert DS4_SESSION_EVAL_ADVANCES_SESSION is True
    assert DS4_SESSION_THREAD_POLICY == "single_worker_session_owner"

    expected = {
        "ds4_engine_open",
        "ds4_session_sync",
        "ds4_session_sample",
        "ds4_session_argmax",
        "ds4_session_eval",
        "ds4_session_top_logprobs",
        "ds4_session_save_snapshot",
        "ds4_session_save_payload",
        "ds4_think_mode_for_context",
        "ds4_engine_mtp_draft_tokens",
    }
    assert expected <= set(DS4_PUBLIC_C_SYMBOLS)
    assert set(DS4_REQUIRED_C_SYMBOLS) <= set(DS4_PUBLIC_C_SYMBOLS)


def test_require_compatible_startup_accepts_pyds4_capabilities() -> None:
    require_compatible_startup(_fake_binding(), "metal")


def test_require_compatible_startup_rejects_missing_capabilities() -> None:
    with pytest.raises(Ds4ApiVersionError, match="capabilities"):
        require_compatible_startup(SimpleNamespace(), "metal")


def test_require_compatible_binding_rejects_missing_capability_symbol() -> (
    None
):
    symbols = tuple(
        symbol
        for symbol in DS4_REQUIRED_C_SYMBOLS
        if symbol != "ds4_session_eval"
    )

    with pytest.raises(Ds4ApiVersionError, match="ds4_session_eval"):
        require_compatible_binding(
            _fake_binding(
                capabilities=lambda: _fake_capabilities(
                    required_symbols=symbols
                )
            )
        )


def test_require_backend_available_uses_pyds4_capabilities() -> None:
    binding = _fake_binding(
        capabilities=lambda: _fake_capabilities(available_backends=("cuda",)),
    )

    with pytest.raises(
        Ds4BackendUnavailable, match="unavailable on this platform"
    ):
        require_backend_available(binding, "metal")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "capabilities": lambda: _fake_capabilities(
                    required_symbols="ds4_engine_open"
                )
            },
            "required_symbols",
        ),
        (
            {
                "capabilities": lambda: _fake_capabilities(
                    ds4_api_version=0 if DS4_API_VERSION != 0 else 1
                )
            },
            "expected DS4 C API version",
        ),
    ],
)
def test_require_compatible_binding_rejects_metadata_mismatches(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(Ds4ApiVersionError, match=match):
        require_compatible_binding(_fake_binding(**overrides))


def test_require_compatible_binding_rejects_wrong_commit() -> None:
    with pytest.raises(Ds4ApiVersionError, match="expected DS4 C API commit"):
        require_compatible_binding(
            _fake_binding(
                capabilities=lambda: _fake_capabilities(ds4_commit="old")
            )
        )


def test_require_backend_available_rejects_unsupported_backend() -> None:
    with pytest.raises(Ds4BackendUnavailable, match="Unsupported"):
        require_backend_available(_fake_binding(), "vulkan")


def test_require_backend_available_reports_platform_unavailable() -> None:
    binding = _fake_binding(
        capabilities=lambda: _fake_capabilities(available_backends=("cuda",)),
        backend_unavailable_reason=lambda backend: "Metal runtime missing.",
    )

    with pytest.raises(
        Ds4BackendUnavailable,
        match="macOS arm64 \\+ Metal, Linux \\+ CUDA",
    ):
        require_backend_available(binding, "metal")


def test_require_backend_available_rejects_missing_capabilities() -> None:
    with pytest.raises(Ds4ApiVersionError, match="capabilities"):
        require_backend_available(SimpleNamespace(), "metal")


def test_require_backend_available_rejects_malformed_available_backends() -> (
    None
):
    binding = _fake_binding(
        capabilities=lambda: _fake_capabilities(available_backends="metal")
    )

    with pytest.raises(Ds4ApiVersionError, match="available_backends"):
        require_backend_available(binding, "metal")


def test_require_backend_available_rejects_unavailable_backend() -> None:
    binding = _fake_binding(
        capabilities=lambda: _fake_capabilities(available_backends=("cuda",)),
    )

    with pytest.raises(
        Ds4BackendUnavailable, match="unavailable on this platform"
    ):
        require_backend_available(binding, "metal")


def test_local_ds4_checkout_matches_verified_header_symbols() -> None:
    source_dir = environ.get("AVALAN_DS4_SOURCE_DIR")
    if not source_dir:
        pytest.skip("Set AVALAN_DS4_SOURCE_DIR to smoke-test a DS4 checkout.")

    source_path = Path(source_dir)
    header_path = source_path / "ds4.h"
    assert header_path.is_file()
    header = header_path.read_text(encoding="utf-8")
    missing = [
        symbol for symbol in DS4_REQUIRED_C_SYMBOLS if symbol not in header
    ]
    assert missing == []

    if (source_path / ".git").exists():
        completed = run(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        assert completed.stdout.strip() == DS4_API_COMMIT
