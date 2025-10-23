"""Helpers for checking optional A2A protocol support."""

from importlib.util import find_spec


def is_a2a_supported() -> bool:
    """Return True when the optional a2a-sdk dependency is installed."""

    return find_spec("a2a") is not None


def ensure_a2a_support() -> None:
    """Raise an informative error when the a2a-sdk dependency is missing."""

    if not is_a2a_supported():
        raise ImportError(
            "A2A support requires the optional 'a2a-sdk' package. "
            "Install it with `pip install a2a-sdk`."
        )
