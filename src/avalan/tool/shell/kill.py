from os import getpid, getppid

REDACTED_KILL_STDERR = "kill diagnostic redacted"


def redacted_stderr(value: str) -> str:
    """Return a generic kill diagnostic."""
    assert isinstance(value, str), "value must be a string"
    return REDACTED_KILL_STDERR if value else ""


def is_protected_pid(value: int) -> bool:
    """Return whether a process identifier is protected from signaling."""
    assert isinstance(value, int) and not isinstance(
        value, bool
    ), "value must be an integer"
    return value in {1, getpid(), getppid()}
