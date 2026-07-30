DATE_FORMATS = ("default", "date", "time", "iso8601", "unix")
DATE_CUSTOM_FORMAT_DIRECTIVES = (
    "%",
    "C",
    "d",
    "D",
    "e",
    "F",
    "g",
    "G",
    "H",
    "I",
    "j",
    "m",
    "M",
    "R",
    "s",
    "S",
    "T",
    "u",
    "U",
    "V",
    "w",
    "W",
    "y",
    "Y",
    "z",
)
DATE_CUSTOM_FORMAT_MAX_BYTES = 128
DATE_CUSTOM_FORMAT_PATTERN = (
    rf"^(?:[ -$&-~]|%[{''.join(DATE_CUSTOM_FORMAT_DIRECTIVES)}])+"
    r"(?![\s\S])"
)
