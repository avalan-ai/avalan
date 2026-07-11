PGREP_MAX_PID = 2**31 - 1
REDACTED_PGREP_PATTERN = "[redacted]"
REDACTED_PGREP_STDERR = "pgrep diagnostic redacted"


def redacted_stderr(value: str) -> str:
    """Return a generic pgrep diagnostic."""
    assert isinstance(value, str), "value must be a string"
    return REDACTED_PGREP_STDERR if value else ""


def pid_only_stdout(
    value: str,
    *,
    stdout_truncated: bool = False,
) -> str:
    """Return only complete canonical process identifier lines."""
    assert isinstance(value, str), "value must be a string"
    assert isinstance(
        stdout_truncated, bool
    ), "stdout_truncated must be a boolean"
    if stdout_truncated and value and not value.endswith("\n"):
        final_newline = value.rfind("\n")
        value = value[: final_newline + 1] if final_newline >= 0 else ""
    pids = tuple(line for line in value.splitlines() if _is_pid_line(line))
    if not pids:
        return ""
    suffix = "\n" if value.endswith("\n") else ""
    return "\n".join(pids) + suffix


def _is_pid_line(value: str) -> bool:
    if not value.isascii() or not value.isdigit() or value.startswith("0"):
        return False
    if len(value) > len(str(PGREP_MAX_PID)):
        return False
    return int(value) <= PGREP_MAX_PID
