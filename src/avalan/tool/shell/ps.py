from re import compile as compile_pattern
from typing import Literal, TypeAlias
from unicodedata import category as unicode_category

PS_MAX_PID = 2**31 - 1
PS_VIEWS = ("summary", "resources")
REDACTED_PS_STDERR = "ps diagnostic redacted"
PsView: TypeAlias = Literal["summary", "resources"]

_PS_ROW_PATTERN = compile_pattern(
    r"^\s*([1-9][0-9]*)\s+([0-9]+)\s+([^\s]+)\s+([^\s]+)\s+(.+?)\s*$"
)
_PS_RESOURCE_ROW_PATTERN = compile_pattern(
    r"^\s*([1-9][0-9]*)\s+([^\s]+)\s+([^\s]+)\s+"
    r"([0-9]+)\s+([0-9]+)\s+([^\s]+)\s+([^\s]+)\s*$"
)
_PS_STATE_PATTERN = compile_pattern(r"^[A-Za-z?+<LNsl]+$")
_PS_ELAPSED_PATTERN = compile_pattern(
    r"^(?:(?:[0-9]+-)?[0-9]{1,2}:)?[0-9]{2}:[0-9]{2}$"
)
_PS_PERCENTAGE_PATTERN = compile_pattern(r"^(?:0|[1-9][0-9]*)\.[0-9]$")
_PS_DARWIN_CPU_TIME_PATTERN = compile_pattern(
    r"^((?:0|[1-9][0-9]*)):([0-9]{2})\.[0-9]{2}$"
)
_PS_PROCPS_CPU_TIME_PATTERN = compile_pattern(
    r"^(?:((?:0|[1-9][0-9]*))-)?([0-9]{2}):([0-9]{2}):([0-9]{2})$"
)
_PS_MAX_RESOURCE_KIB = 2**63 - 1


def redacted_stderr(value: str) -> str:
    """Return a generic ps diagnostic."""
    assert isinstance(value, str), "value must be a string"
    return REDACTED_PS_STDERR if value else ""


def process_rows_stdout(
    value: str,
    *,
    requested_pids: tuple[int, ...],
    view: PsView = "summary",
    stdout_truncated: bool = False,
) -> str:
    """Return only complete canonical process inspection rows."""
    assert isinstance(value, str), "value must be a string"
    assert isinstance(
        stdout_truncated, bool
    ), "stdout_truncated must be a boolean"
    assert isinstance(requested_pids, tuple), "requested_pids must be a tuple"
    assert view in PS_VIEWS, "view must be supported"
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
        normalized = (
            _canonical_process_row(line)
            if view == "summary"
            else _canonical_resource_row(line)
        )
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


def _canonical_resource_row(value: str) -> tuple[int, str] | None:
    if not value.isascii() or any(
        unicode_category(character).startswith("C") for character in value
    ):
        return None
    matched = _PS_RESOURCE_ROW_PATTERN.fullmatch(value)
    if matched is None:
        return None
    pid, cpu, memory, rss, virtual_size, cpu_time, nice = matched.groups()
    if not _canonical_pid(pid):
        return None
    if not _canonical_percentage(cpu):
        return None
    if not _canonical_percentage(memory):
        return None
    if not _canonical_resource_kib(rss) or not _canonical_resource_kib(
        virtual_size
    ):
        return None
    if not _canonical_cpu_time(cpu_time) or not _canonical_nice(nice):
        return None
    row = f"{pid} {cpu} {memory} {rss} {virtual_size} {cpu_time} {nice}"
    return int(pid), row


def _canonical_pid(value: str) -> bool:
    return (
        value.isdigit()
        and len(value) <= len(str(PS_MAX_PID))
        and not value.startswith("0")
        and int(value) <= PS_MAX_PID
    )


def _canonical_parent_pid(value: str) -> bool:
    return (
        value.isdigit()
        and len(value) <= len(str(PS_MAX_PID))
        and (value == "0" or not value.startswith("0"))
        and int(value) <= PS_MAX_PID
    )


def _canonical_elapsed(value: str) -> bool:
    if len(value) > 32 or _PS_ELAPSED_PATTERN.fullmatch(value) is None:
        return False
    clock = value.rsplit("-", maxsplit=1)[-1].split(":")
    if len(clock) == 3 and int(clock[0]) > 23:
        return False
    return int(clock[-2]) <= 59 and int(clock[-1]) <= 59


def _canonical_percentage(value: str) -> bool:
    return (
        len(value) <= 12
        and _PS_PERCENTAGE_PATTERN.fullmatch(value) is not None
    )


def _canonical_resource_kib(value: str) -> bool:
    return (
        value.isdigit()
        and len(value) <= len(str(_PS_MAX_RESOURCE_KIB))
        and (value == "0" or not value.startswith("0"))
        and int(value) <= _PS_MAX_RESOURCE_KIB
    )


def _canonical_cpu_time(value: str) -> bool:
    if len(value) > 32:
        return False
    darwin_match = _PS_DARWIN_CPU_TIME_PATTERN.fullmatch(value)
    if darwin_match is not None:
        minutes, seconds = darwin_match.groups()
        return len(minutes) <= 10 and int(seconds) <= 59
    procps_match = _PS_PROCPS_CPU_TIME_PATTERN.fullmatch(value)
    if procps_match is None:
        return False
    days, hours, minutes, seconds = procps_match.groups()
    if days is not None and len(days) > 10:
        return False
    return int(hours) <= 23 and int(minutes) <= 59 and int(seconds) <= 59


def _canonical_nice(value: str) -> bool:
    if len(value) > 3:
        return False
    if value.startswith("-"):
        magnitude = value[1:]
        if not magnitude or magnitude.startswith("0"):
            return False
    else:
        magnitude = value
        if magnitude != "0" and magnitude.startswith("0"):
            return False
    return magnitude.isdigit() and -20 <= int(value) <= 20
