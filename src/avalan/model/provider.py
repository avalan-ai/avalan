from collections.abc import Mapping
from enum import StrEnum


class ProviderFamily(StrEnum):
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    GOOGLE = "google"
    HUGGING_FACE = "hugging_face"
    LOCAL = "local"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    OTHER = "other"


_LEGACY_PROVIDER_OPTION_URI_KEYS = frozenset({"azure_api_version"})
_PROVIDER_OPTION_URI_PREFIX = "provider_"


def provider_family_value(value: ProviderFamily | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, ProviderFamily):
        return value.value
    assert isinstance(value, str), "provider_family must be a string"
    assert value.strip(), "provider_family must not be empty"
    return value


def provider_options_from_uri_params(
    params: Mapping[str, str | int | float | bool],
) -> dict[str, object] | None:
    options: dict[str, object] = {}
    for key, value in params.items():
        option_key = _provider_option_key(key)
        if option_key is None:
            continue
        options[option_key] = value
    return options or None


def provider_string_option(
    options: Mapping[str, object] | None,
    key: str,
) -> str | None:
    assert isinstance(key, str) and key.strip()
    if options is None:
        return None
    value = options.get(key)
    if value is None:
        return None
    assert isinstance(value, str) and value.strip()
    return value


def _provider_option_key(key: str) -> str | None:
    if key in _LEGACY_PROVIDER_OPTION_URI_KEYS:
        return key
    if not key.startswith(_PROVIDER_OPTION_URI_PREFIX):
        return None
    option_key = key.removeprefix(_PROVIDER_OPTION_URI_PREFIX)
    assert option_key, "provider option URI key must not be empty"
    return option_key
