from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import pytest

from avalan.model import (
    FileDeliveryDiagnostic,
    FileDeliveryLimit,
    FileDeliveryMode,
    FileDeliveryProfile,
    LocalFileDeliveryProfile,
    resolve_file_delivery_profile,
)


def test_hosted_provider_profiles_allow_expected_delivery_modes() -> None:
    openai = resolve_file_delivery_profile("ai://env:KEY@openai/gpt-4o")
    anthropic = resolve_file_delivery_profile("ai://env:KEY@anthropic/claude")
    google = resolve_file_delivery_profile("ai://env:KEY@google/gemini")

    assert openai.name == "openai"
    assert openai.supports_delivery_mode(FileDeliveryMode.PROVIDER_FILE_ID)
    assert openai.supports_delivery_mode(FileDeliveryMode.HOSTED_URL)
    assert openai.supports_delivery_mode(FileDeliveryMode.INLINE_BYTES)
    assert openai.supports_delivery_mode(FileDeliveryMode.INLINE_TEXT)
    assert openai.accepts_mime_type("application/pdf")
    assert openai.accepts_source_kind("local_path")
    assert not openai.allows_object_store_uri("gs://bucket/object")
    assert anthropic.name == "anthropic"
    assert anthropic.has_native_file_delivery
    assert google.supports_delivery_mode(FileDeliveryMode.OBJECT_STORE_URI)
    assert google.allows_object_store_uri("gs://bucket/object")
    assert not google.allows_object_store_uri("s3://bucket/object")
    assert {limit.source for limit in google.limits} == {
        "converter",
        "model.context",
        "provider.google",
        "task.artifact",
        "task.limits",
    }


def test_bedrock_profile_uses_text_and_s3_delivery_only() -> None:
    profile = resolve_file_delivery_profile(
        "ai://env:KEY@bedrock/us.anthropic.claude"
    )

    assert profile.name == "bedrock"
    assert not profile.supports_delivery_mode(
        FileDeliveryMode.PROVIDER_FILE_ID
    )
    assert not profile.supports_delivery_mode(FileDeliveryMode.HOSTED_URL)
    assert profile.supports_delivery_mode(FileDeliveryMode.OBJECT_STORE_URI)
    assert profile.allows_object_store_uri("s3://bucket/key")
    assert not profile.allows_object_store_uri("gs://bucket/key")
    assert profile.accepts_mime_type("text/plain")
    assert profile.accepts_mime_type("application/json")
    assert not profile.accepts_mime_type("application/pdf")


def test_local_profiles_distinguish_text_and_multimodal_delivery() -> None:
    text_profile = resolve_file_delivery_profile("ai://local/model")
    multimodal_profile = resolve_file_delivery_profile(
        "ai://local/model",
        local_profile=LocalFileDeliveryProfile.MULTIMODAL,
    )

    assert text_profile.name == "local_text"
    assert not text_profile.has_native_file_delivery
    assert text_profile.supports_delivery_mode(FileDeliveryMode.INLINE_TEXT)
    assert not text_profile.supports_delivery_mode(
        FileDeliveryMode.INLINE_BYTES
    )
    assert text_profile.accepts_mime_type("application/markdown")
    assert not text_profile.accepts_mime_type("image/png")
    assert multimodal_profile.name == "local_multimodal"
    assert multimodal_profile.has_native_file_delivery
    assert multimodal_profile.supports_delivery_mode(
        FileDeliveryMode.INLINE_BYTES
    )
    assert multimodal_profile.accepts_mime_type("image/png")
    assert multimodal_profile.accepts_mime_type("audio/wav")


@pytest.mark.parametrize(
    "uri",
    [
        None,
        "",
        "openai/gpt-4o",
        "http://example.test/model",
        "ai://env:KEY@unknown/model",
    ],
)
def test_unknown_profiles_fail_closed_with_stable_diagnostics(
    uri: str | None,
) -> None:
    profile = resolve_file_delivery_profile(uri)

    assert profile.name == "unknown"
    assert profile.delivery_modes == frozenset({FileDeliveryMode.REJECT})
    assert not profile.has_native_file_delivery
    assert not profile.accepts_mime_type("text/plain")
    assert not profile.accepts_source_kind("local_path")
    assert profile.diagnostics == (
        FileDeliveryDiagnostic(
            code="model.file_delivery.unknown_profile",
            message="Model file delivery capabilities are unknown.",
            hint=(
                "Configure an explicit provider or backend file delivery "
                "profile before sending file inputs."
            ),
        ),
    )


def test_profile_metadata_is_immutable() -> None:
    source = {"provider": "fake"}
    profile = FileDeliveryProfile(
        name="custom",
        delivery_modes=frozenset({FileDeliveryMode.REJECT}),
        metadata=source,
    )
    source["provider"] = "changed"

    assert profile.metadata == MappingProxyType({"provider": "fake"})
    with pytest.raises(TypeError):
        cast(dict[str, object], profile.metadata)["provider"] = "changed"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FileDeliveryLimit(name="", source="source"),
        lambda: FileDeliveryLimit(name="limit", source=""),
        lambda: FileDeliveryLimit(name="limit", source="source", max_bytes=0),
        lambda: FileDeliveryLimit(
            name="limit",
            source="source",
            max_count=True,
        ),
        lambda: FileDeliveryDiagnostic(
            code="", message="message", hint="hint"
        ),
        lambda: FileDeliveryDiagnostic(code="code", message="", hint="hint"),
        lambda: FileDeliveryDiagnostic(
            code="code",
            message="message",
            hint="",
        ),
        lambda: FileDeliveryProfile(
            name="",
            delivery_modes=frozenset({FileDeliveryMode.REJECT}),
        ),
        lambda: FileDeliveryProfile(
            name="bad",
            delivery_modes=cast(
                frozenset[FileDeliveryMode],
                frozenset({"reject"}),
            ),
        ),
        lambda: FileDeliveryProfile(
            name="bad",
            delivery_modes=frozenset({FileDeliveryMode.REJECT}),
            accepted_mime_types=("",),
        ),
        lambda: FileDeliveryProfile(
            name="bad",
            delivery_modes=frozenset({FileDeliveryMode.REJECT}),
            source_kinds=frozenset({""}),
        ),
        lambda: FileDeliveryProfile(
            name="bad",
            delivery_modes=frozenset({FileDeliveryMode.REJECT}),
            object_store_uri_schemes=("",),
        ),
        lambda: FileDeliveryProfile(
            name="bad",
            delivery_modes=frozenset({FileDeliveryMode.REJECT}),
            limits=cast(tuple[FileDeliveryLimit, ...], ("limit",)),
        ),
        lambda: FileDeliveryProfile(
            name="bad",
            delivery_modes=frozenset({FileDeliveryMode.REJECT}),
            diagnostics=cast(
                tuple[FileDeliveryDiagnostic, ...],
                ("diagnostic",),
            ),
        ),
    ],
)
def test_profile_entities_reject_invalid_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(AssertionError):
        factory()


def test_profile_methods_validate_arguments() -> None:
    profile = resolve_file_delivery_profile("ai://env:KEY@openai/gpt-4o")

    with pytest.raises(AssertionError):
        profile.supports_delivery_mode(cast(FileDeliveryMode, "inline_bytes"))
    with pytest.raises(AssertionError):
        profile.accepts_source_kind("")
    with pytest.raises(AssertionError):
        profile.allows_object_store_uri("")
    with pytest.raises(AssertionError):
        resolve_file_delivery_profile(
            "ai://local/model",
            local_profile=cast(LocalFileDeliveryProfile, "text"),
        )
