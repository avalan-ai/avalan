from dataclasses import dataclass

LSOF_MAX_PID = 2**31 - 1
LSOF_DEFAULT_LIMIT = 64
LSOF_MAX_LIMIT = 256
REDACTED_LSOF_STDERR = "lsof diagnostic redacted"
_LSOF_MAX_TOKEN_BYTES = 64
_LSOF_ALLOWED_FIELD_IDS = frozenset(("P", "a", "f", "p", "t"))
_LSOF_ACCESS_MODES = frozenset(("r", "u", "w"))
_LSOF_RECORD_TERMINATOR = "\x00\n"
_LSOF_FILE_TYPES = {
    "REG": "regular",
    "DIR": "directory",
    "CHR": "character",
    "BLK": "block",
    "FIFO": "pipe",
    "PIPE": "pipe",
    "IPv4": "ipv4",
    "IPv6": "ipv6",
    "unix": "unix_socket",
    "UNIX": "unix_socket",
    "SOCK": "socket",
    "KQUEUE": "event",
}
_LSOF_PROTOCOLS = {
    "TCP": "tcp",
    "UDP": "udp",
    "UDPLITE": "udplite",
}


@dataclass(frozen=True, slots=True)
class LsofOutput:
    stdout: str
    row_limit_truncated: bool
    byte_limit_truncated: bool
    malformed: bool


def redacted_stderr(value: str) -> str:
    """Return a generic lsof diagnostic."""
    assert isinstance(value, str), "value must be a string"
    return REDACTED_LSOF_STDERR if value else ""


def open_file_rows_stdout(
    value: str,
    *,
    requested_pid: int,
    limit: int = LSOF_DEFAULT_LIMIT,
    max_stdout_bytes: int | None = None,
    stdout_truncated: bool = False,
) -> LsofOutput:
    """Return complete canonical numeric file descriptor rows."""
    assert isinstance(value, str), "value must be a string"
    assert isinstance(requested_pid, int) and not isinstance(
        requested_pid, bool
    ), "requested_pid must be an integer"
    assert 1 <= requested_pid <= LSOF_MAX_PID, "requested_pid is invalid"
    assert isinstance(limit, int) and not isinstance(
        limit, bool
    ), "limit must be an integer"
    assert 1 <= limit <= LSOF_MAX_LIMIT, "limit is invalid"
    assert max_stdout_bytes is None or (
        isinstance(max_stdout_bytes, int)
        and not isinstance(max_stdout_bytes, bool)
        and max_stdout_bytes >= 0
    ), "max_stdout_bytes must be a non-negative integer or None"
    assert isinstance(
        stdout_truncated, bool
    ), "stdout_truncated must be a boolean"

    complete, incomplete_tail = _complete_lsof_output(
        value,
        stdout_truncated=stdout_truncated,
    )
    if complete is None:
        return LsofOutput(
            stdout="",
            row_limit_truncated=False,
            byte_limit_truncated=False,
            malformed=True,
        )
    if not complete:
        return LsofOutput(
            stdout="",
            row_limit_truncated=False,
            byte_limit_truncated=False,
            malformed=not incomplete_tail,
        )

    records = complete.split(_LSOF_RECORD_TERMINATOR)
    assert records[-1] == ""
    records.pop()
    if not _valid_process_record(records[0], requested_pid=requested_pid):
        return LsofOutput(
            stdout="",
            row_limit_truncated=False,
            byte_limit_truncated=False,
            malformed=True,
        )

    rows: list[str] = []
    seen_file_descriptors: set[int] = set()
    numeric_record_count = 0
    stdout_bytes = 0
    byte_limit_truncated = False
    for record in records[1:]:
        parsed = _file_record(record)
        if parsed is None:
            return LsofOutput(
                stdout="",
                row_limit_truncated=numeric_record_count > limit,
                byte_limit_truncated=byte_limit_truncated,
                malformed=True,
            )
        file_descriptor, access, file_type, protocol = parsed
        if file_descriptor is None:
            continue
        if file_descriptor in seen_file_descriptors:
            return LsofOutput(
                stdout="",
                row_limit_truncated=numeric_record_count > limit,
                byte_limit_truncated=byte_limit_truncated,
                malformed=True,
            )
        seen_file_descriptors.add(file_descriptor)
        numeric_record_count += 1
        if numeric_record_count > limit or byte_limit_truncated:
            continue
        row = (
            "\t".join(
                (
                    str(requested_pid),
                    str(file_descriptor),
                    access,
                    file_type,
                    protocol,
                )
            )
            + "\n"
        )
        row_bytes = len(row.encode("ascii"))
        if (
            max_stdout_bytes is not None
            and stdout_bytes + row_bytes > max_stdout_bytes
        ):
            byte_limit_truncated = True
            continue
        rows.append(row)
        stdout_bytes += row_bytes

    return LsofOutput(
        stdout="".join(rows),
        row_limit_truncated=numeric_record_count > limit,
        byte_limit_truncated=byte_limit_truncated,
        malformed=False,
    )


def _complete_lsof_output(
    value: str,
    *,
    stdout_truncated: bool,
) -> tuple[str | None, bool]:
    if not stdout_truncated:
        if not value.endswith(_LSOF_RECORD_TERMINATOR):
            return None, False
        return value, False
    final_record = value.rfind(_LSOF_RECORD_TERMINATOR)
    if final_record < 0:
        return "", True
    end = final_record + len(_LSOF_RECORD_TERMINATOR)
    return value[:end], end != len(value)


def _valid_process_record(record: str, *, requested_pid: int) -> bool:
    fields = record.split("\x00")
    return len(fields) == 1 and fields[0] == f"p{requested_pid}"


def _file_record(
    record: str,
) -> tuple[int | None, str, str, str] | None:
    raw_fields = record.split("\x00")
    if not raw_fields or not raw_fields[0].startswith("f"):
        return None
    fields: dict[str, str] = {}
    for raw_field in raw_fields:
        if not raw_field:
            return None
        field_id = raw_field[0]
        field_value = raw_field[1:]
        if (
            field_id not in _LSOF_ALLOWED_FIELD_IDS
            or field_id == "p"
            or field_id in fields
        ):
            return None
        fields[field_id] = field_value

    raw_file_descriptor = fields["f"]
    if raw_file_descriptor.isdigit():
        if not _canonical_non_negative_int(raw_file_descriptor):
            return None
        file_descriptor: int | None = int(raw_file_descriptor)
    else:
        if not _safe_identifier(raw_file_descriptor):
            return None
        file_descriptor = None

    access = fields.get("a")
    if access is None or access in {"", " "}:
        canonical_access = "-"
    elif access in _LSOF_ACCESS_MODES:
        canonical_access = access
    else:
        return None

    file_type = fields.get("t")
    if file_type is None:
        canonical_file_type = "other"
    else:
        normalized_file_type = _canonical_file_type(file_type)
        if normalized_file_type is None:
            return None
        canonical_file_type = normalized_file_type

    protocol = fields.get("P")
    if protocol is None:
        canonical_protocol = "-"
    else:
        normalized_protocol = _canonical_protocol(protocol)
        if normalized_protocol is None:
            return None
        canonical_protocol = normalized_protocol
    return (
        file_descriptor,
        canonical_access,
        canonical_file_type,
        canonical_protocol,
    )


def _canonical_non_negative_int(value: str) -> bool:
    return (
        value.isascii()
        and value.isdigit()
        and len(value) <= len(str(LSOF_MAX_PID))
        and (value == "0" or not value.startswith("0"))
        and int(value) <= LSOF_MAX_PID
    )


def _canonical_file_type(value: str) -> str | None:
    if not _safe_identifier(value):
        return None
    return _LSOF_FILE_TYPES.get(value, "other")


def _canonical_protocol(value: str) -> str | None:
    if not _safe_identifier(value):
        return None
    return _LSOF_PROTOCOLS.get(value, "other")


def _safe_identifier(value: str) -> bool:
    return (
        bool(value)
        and value.isascii()
        and len(value.encode("ascii")) <= _LSOF_MAX_TOKEN_BYTES
        and value[0].isalpha()
        and all(
            character.isalpha() or character.isdigit() or character == "_"
            for character in value
        )
    )
