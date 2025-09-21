from typing import Final

SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_message(data: str | bytes, *, event: str | None = None) -> str:
    """Format a server-sent event payload as a string."""
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = data

    lines = text.splitlines()
    if not lines:
        lines = [""]

    parts: list[str] = []
    if event is not None:
        parts.append(f"event: {event}")
    parts.extend(f"data: {line}" for line in lines)
    return "\n".join(parts) + "\n\n"


def sse_bytes(data: str | bytes, *, event: str | None = None) -> bytes:
    """Format a server-sent event payload as bytes."""
    return sse_message(data, event=event).encode("utf-8")


def sse_headers() -> dict[str, str]:
    """Return default headers for server-sent event responses."""
    return dict(SSE_HEADERS)
