"""Utilities for exposing Avalan via the A2A protocol."""

from .availability import ensure_a2a_support

ensure_a2a_support()

from .router import router, well_known_router  # noqa: F401
