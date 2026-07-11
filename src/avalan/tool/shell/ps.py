from re import compile as compile_pattern
from unicodedata import category as unicode_category

PS_MAX_PID = 2**31 - 1
REDACTED_PS_STDERR = "ps diagnostic redacted"

_PS_ROW_PATTERN = compile_pattern(
    r"^\s*([1-9][0-9]*)\s+([0-9]+)\s+([^\s]+)\s+([^\s]+)\s+(.+?)\s*$"
)
_PS_STATE_PATTERN = compile_pattern(r"^[A-Za-z?+<LNsl]+$")
_PS_ELAPSED_PATTERN = compile_pattern(
    r"^(?:(?:[0-9]+-)?[0-9]{1,2}:)?[0-9]{2}:[0-9]{2}$"
)


def redacted_stderr(value: str) -> str:
    """Return a generic ps diagnostic."""
    assert isinstance(value, str), "value must be a string"
    return REDACTED_PS_STDERR if value else ""


def process_rows_stdout(
    value: str,
    *,
    requested_pids: tuple[int, ...],
    stdout_truncated: bool = False,
) -> str:
    """Return only complete canonical process inspection rows."""
    assert isinstance(value, str), "value must be a string"
    assert isinstance(
        stdout_truncated, bool
    ), "stdout_truncated must be a boolean"
    assert isinstance(requested_pids, tuple), "requested_pids must be a tuple"
    assert all(
        isinstance(pid, int) and not isinstance(pid, bool)
        for pid in requested_pids
    ), "requested_pids must contain integers"
    if len(requested_pids) != 1:
        return ""
    if stdout_truncated and value and not value.endswith("\n"):
        final_newline = value.rfind("\n")
        value = value[: final_newline + 1] if final_newline >= 0 else ""
    allowed_pids = set(requested_pids)
    seen_pids: set[int] = set()
    rows: list[str] = []
    for line in value.splitlines():
        normalized = _canonical_process_row(line)
        if normalized is None:
            if line:
                return ""
            continue
        pid, row = normalized
        if pid not in allowed_pids or pid in seen_pids:
            return ""
        seen_pids.add(pid)
        rows.append(row)
    if not rows:
        return ""
    suffix = "\n" if value.endswith("\n") else ""
    return "\n".join(rows) + suffix


def _canonical_process_row(value: str) -> tuple[int, str] | None:
    if not value.isascii() or any(
        unicode_category(character).startswith("C") for character in value
    ):
        return None
    matched = _PS_ROW_PATTERN.fullmatch(value)
    if matched is None:
        return None
    pid, parent_pid, state, elapsed, command = matched.groups()
    if not _canonical_pid(pid) or not _canonical_parent_pid(parent_pid):
        return None
    if _PS_STATE_PATTERN.fullmatch(state) is None:
        return None
    if not _canonical_elapsed(elapsed):
        return None
    if not command or len(command.encode("ascii")) > 4096:
        return None
    return int(pid), f"{pid} {parent_pid} {state} {elapsed} {command}"


def _canonical_pid(value: str) -> bool:
    return (
        value.isdigit()
        and not value.startswith("0")
        and int(value) <= PS_MAX_PID
    )


def _canonical_parent_pid(value: str) -> bool:
    return (
        value.isdigit()
        and (value == "0" or not value.startswith("0"))
        and int(value) <= PS_MAX_PID
    )


def _canonical_elapsed(value: str) -> bool:
    if _PS_ELAPSED_PATTERN.fullmatch(value) is None:
        return False
    clock = value.rsplit("-", maxsplit=1)[-1].split(":")
    if len(clock) == 3 and int(clock[0]) > 23:
        return False
    return int(clock[-2]) <= 59 and int(clock[-1]) <= 59
