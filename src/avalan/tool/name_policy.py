from ..entities import ToolNamePolicyMode, ToolNamePolicySettings

from base64 import urlsafe_b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from re import compile as compile_regex
from re import escape as escape_regex

_DEFAULT_SETTINGS = ToolNamePolicySettings()
_OPENAI_SAFE_PATTERN = compile_regex(r"^[A-Za-z0-9_-]{1,64}$")
_LOCAL_SAFE_PATTERN = compile_regex(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_SANITIZE_PATTERN = compile_regex(r"[^A-Za-z0-9_]+")
_LOCAL_PROVIDER_FAMILIES = frozenset(
    {
        "local",
        "ollama",
    }
)


@dataclass(frozen=True, slots=True)
class ToolNamePolicy:
    settings: ToolNamePolicySettings = field(
        default_factory=ToolNamePolicySettings
    )
    provider_family: Enum | str | None = None
    _canonical_to_provider: dict[str, str] = field(default_factory=dict)
    _provider_to_canonical: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert isinstance(self.settings, ToolNamePolicySettings)
        self._validate_prefix()
        self._validate_replacement()
        if self._provider_family_value() is not None:
            assert self._provider_family_value()
        for provider_name in self.settings.map.values():
            self._assert_provider_name(provider_name)

    @classmethod
    def default(cls) -> "ToolNamePolicy":
        return cls(settings=_DEFAULT_SETTINGS)

    def for_provider(
        self, provider_family: Enum | str | None
    ) -> "ToolNamePolicy":
        return ToolNamePolicy(
            settings=self.settings,
            provider_family=provider_family or self._provider_family_value(),
            _canonical_to_provider=dict(self._canonical_to_provider),
            _provider_to_canonical=dict(self._provider_to_canonical),
        )

    def bind(self, canonical_names: Iterable[str]) -> "ToolNamePolicy":
        canonical_to_provider: dict[str, str] = {}
        provider_to_canonical: dict[str, str] = {}
        for canonical_name in canonical_names:
            assert isinstance(canonical_name, str)
            assert canonical_name.strip(), "tool name must not be empty"
            provider_name = self._provider_name_for(canonical_name)
            self._assert_provider_name(provider_name)
            previous = provider_to_canonical.get(provider_name)
            if previous is not None and previous != canonical_name:
                raise AssertionError(
                    "tool provider name collision: "
                    f"{previous!r} and {canonical_name!r} both map to "
                    f"{provider_name!r}"
                )
            canonical_to_provider[canonical_name] = provider_name
            provider_to_canonical[provider_name] = canonical_name
        return ToolNamePolicy(
            settings=self.settings,
            provider_family=self.provider_family,
            _canonical_to_provider=canonical_to_provider,
            _provider_to_canonical=provider_to_canonical,
        )

    def provider_name(self, canonical_name: str) -> str:
        assert isinstance(canonical_name, str)
        assert canonical_name.strip(), "tool name must not be empty"
        if canonical_name in self._canonical_to_provider:
            provider_name = self._canonical_to_provider[canonical_name]
            self._assert_provider_name(provider_name)
            return provider_name
        provider_name = self._provider_name_for(canonical_name)
        self._assert_provider_name(provider_name)
        return provider_name

    def canonical_name(self, provider_name: str) -> str:
        assert isinstance(provider_name, str)
        assert provider_name.strip(), "provider tool name must not be empty"
        self._assert_provider_name(provider_name)
        if provider_name in self._provider_to_canonical:
            return self._provider_to_canonical[provider_name]
        if self.settings.mode is ToolNamePolicyMode.ENCODED:
            return self.decode_encoded(
                provider_name, prefix=self.settings.prefix
            )
        return provider_name

    def validate_provider_names(self) -> None:
        for provider_name in self._canonical_to_provider.values():
            self._assert_provider_name(provider_name)

    @staticmethod
    def encode_encoded(tool_name: str, *, prefix: str = "avl_") -> str:
        ToolNamePolicy._assert_non_empty(tool_name, "tool name")
        ToolNamePolicy._assert_non_empty(prefix, "tool name prefix")
        if _OPENAI_SAFE_PATTERN.fullmatch(
            tool_name
        ) and not tool_name.startswith(prefix):
            return tool_name
        encoded = urlsafe_b64encode(tool_name.encode()).decode().rstrip("=")
        return f"{prefix}{encoded}"

    @staticmethod
    def decode_encoded(tool_name: str, *, prefix: str = "avl_") -> str:
        ToolNamePolicy._assert_non_empty(tool_name, "provider tool name")
        ToolNamePolicy._assert_non_empty(prefix, "tool name prefix")
        assert _OPENAI_SAFE_PATTERN.fullmatch(
            tool_name
        ), "provider tool name is invalid"

        if not tool_name.startswith(prefix):
            return tool_name

        payload = tool_name[len(prefix) :]
        assert payload, "provider tool name is missing encoded content"
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = urlsafe_b64decode(f"{payload}{padding}").decode()
        except (BinasciiError, UnicodeDecodeError) as exc:
            raise AssertionError("provider tool name is malformed") from exc
        assert decoded.strip(), "decoded tool name must not be empty"
        assert (
            ToolNamePolicy.encode_encoded(decoded, prefix=prefix) == tool_name
        ), "provider tool name is malformed"
        return decoded

    def _provider_name_for(self, canonical_name: str) -> str:
        match self.settings.mode:
            case ToolNamePolicyMode.ENCODED:
                mapped_name = self.settings.map.get(canonical_name)
                if mapped_name is not None:
                    return mapped_name
                return self.encode_encoded(
                    canonical_name,
                    prefix=self.settings.prefix,
                )
            case ToolNamePolicyMode.MAPPED | ToolNamePolicyMode.RAW:
                mapped_name = self.settings.map.get(canonical_name)
                return (
                    mapped_name if mapped_name is not None else canonical_name
                )
            case ToolNamePolicyMode.SANITIZED:
                mapped_name = self.settings.map.get(canonical_name)
                if mapped_name is not None:
                    return mapped_name
                return self._sanitize(canonical_name)
            case _:
                raise AssertionError(
                    f"unsupported ToolNamePolicyMode: {self.settings.mode!r}"
                )

    def _sanitize(self, canonical_name: str) -> str:
        replacement = self.settings.replacement
        provider_name = _SANITIZE_PATTERN.sub(replacement, canonical_name)
        if self.settings.collapse_replacement:
            replacement_pattern = f"{escape_regex(replacement)}+"
            provider_name = compile_regex(replacement_pattern).sub(
                replacement, provider_name
            )
        assert provider_name, "sanitized tool name must not be empty"
        return provider_name

    def _assert_provider_name(self, provider_name: str) -> None:
        self._assert_non_empty(provider_name, "provider tool name")
        pattern = self._provider_pattern()
        assert pattern.fullmatch(provider_name), (
            "provider tool name is invalid for "
            f"{self._provider_family_label()}: {provider_name!r}"
        )

    def _provider_pattern(self) -> Pattern[str]:
        provider_family = self._provider_family_value()
        if provider_family in _LOCAL_PROVIDER_FAMILIES or (
            provider_family is None
            and self.settings.mode is ToolNamePolicyMode.RAW
        ):
            return _LOCAL_SAFE_PATTERN
        return _OPENAI_SAFE_PATTERN

    def _provider_family_label(self) -> str:
        return self._provider_family_value() or "openai-compatible providers"

    def _provider_family_value(self) -> str | None:
        value = self.provider_family or self.settings.provider_family
        if value is None:
            return None
        if isinstance(value, Enum):
            value = value.value
        assert isinstance(value, str), "provider_family must be a string"
        assert value.strip(), "provider_family must not be empty"
        return value

    def _validate_prefix(self) -> None:
        prefix = self.settings.prefix
        assert _OPENAI_SAFE_PATTERN.fullmatch(
            prefix
        ), "tool name policy prefix must be provider-safe"

    def _validate_replacement(self) -> None:
        replacement = self.settings.replacement
        assert _OPENAI_SAFE_PATTERN.fullmatch(
            replacement
        ), "tool name policy replacement must be provider-safe"

    @staticmethod
    def _assert_non_empty(value: str, label: str) -> None:
        assert isinstance(value, str), f"{label} must be a string"
        assert value.strip(), f"{label} must not be empty"
