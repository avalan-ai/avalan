from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import pytest

from avalan.model import (
    FileDeliveryDecision,
    FileDeliveryDiagnostic,
    FileDeliveryLimit,
    FileDeliveryMode,
    FileDeliveryProfile,
    FileDeliveryRequest,
    LocalFileDeliveryProfile,
    plan_file_delivery,
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
    assert not openai.accepts_mime_type(None)
    assert openai.accepts_source_kind("local_path")
    assert openai.has_reference_delivery
    assert not openai.allows_object_store_uri("gs://bucket/object")
    assert anthropic.name == "anthropic"
    assert anthropic.has_native_file_delivery
    assert google.supports_delivery_mode(FileDeliveryMode.OBJECT_STORE_URI)
    assert google.allows_object_store_uri("gs://bucket/object")
    assert not google.allows_object_store_uri("s3://bucket/object")
    assert google.inline_byte_limit == FileDeliveryLimit(
        name="inline_file_bytes",
        source="provider.google",
    )
    assert google.inline_text_limit == FileDeliveryLimit(
        name="inline_text_bytes",
        source="provider.google",
    )
    assert google.file_count_limit == FileDeliveryLimit(
        name="file_count",
        source="provider.google",
    )
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
    assert not text_profile.has_reference_delivery
    assert text_profile.supports_file_delivery
    assert text_profile.requires_conversion_for_file_blocks
    assert text_profile.supports_delivery_mode(FileDeliveryMode.INLINE_TEXT)
    assert not text_profile.supports_delivery_mode(
        FileDeliveryMode.INLINE_BYTES
    )
    assert text_profile.accepts_mime_type("application/markdown")
    assert not text_profile.accepts_mime_type("image/png")
    assert multimodal_profile.name == "local_multimodal"
    assert multimodal_profile.has_native_file_delivery
    assert not multimodal_profile.requires_conversion_for_file_blocks
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
    assert not profile.supports_file_delivery
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


def test_request_metadata_is_immutable() -> None:
    source = {"provider_file_id": "file-1"}
    request = FileDeliveryRequest(metadata=source)
    source["provider_file_id"] = "changed"

    assert request.metadata == MappingProxyType({"provider_file_id": "file-1"})
    with pytest.raises(TypeError):
        cast(dict[str, object], request.metadata)["provider_file_id"] = "x"


def test_plan_delivery_prefers_provider_references() -> None:
    openai = resolve_file_delivery_profile("ai://env:KEY@openai/gpt-4o")
    google = resolve_file_delivery_profile("ai://env:KEY@google/gemini")

    id_decision = plan_file_delivery(
        openai,
        FileDeliveryRequest(
            mime_type="application/pdf",
            metadata={
                "provider_file_id": "file-1",
                "provider_file_url": "https://example.test/file",
            },
        ),
    )
    url_decision = openai.plan_delivery(
        FileDeliveryRequest(
            metadata={"provider_file_url": "https://example.test/file"}
        )
    )
    uri_decision = google.plan_delivery(
        FileDeliveryRequest(metadata={"provider_uri": "gs://bucket/object"})
    )

    assert id_decision == FileDeliveryDecision(
        mode=FileDeliveryMode.PROVIDER_FILE_ID,
        reference="file-1",
    )
    assert not id_decision.needs_artifact_read
    assert url_decision == FileDeliveryDecision(
        mode=FileDeliveryMode.HOSTED_URL,
        reference="https://example.test/file",
    )
    assert uri_decision == FileDeliveryDecision(
        mode=FileDeliveryMode.OBJECT_STORE_URI,
        reference="gs://bucket/object",
    )


def test_plan_delivery_uses_typed_provider_reference_metadata() -> None:
    openai = resolve_file_delivery_profile("ai://env:KEY@openai/gpt-4o")
    google = resolve_file_delivery_profile("ai://env:KEY@google/gemini")

    id_decision = openai.plan_delivery(
        FileDeliveryRequest(
            metadata={
                "provider_reference": {
                    "kind": "provider_file_id",
                    "provider": "openai",
                    "reference": "file-1",
                    "owner_scope": "tenant-a",
                }
            }
        )
    )
    uri_decision = google.plan_delivery(
        FileDeliveryRequest(
            metadata={
                "provider_reference": {
                    "kind": "object_store_uri",
                    "provider": "google",
                    "reference": "gs://bucket/object",
                }
            }
        )
    )
    url_decision = openai.plan_delivery(
        FileDeliveryRequest(
            metadata={
                "provider_reference": {
                    "kind": "hosted_url",
                    "provider": "openai",
                    "reference": "https://example.test/file",
                }
            }
        )
    )

    assert id_decision == FileDeliveryDecision(
        mode=FileDeliveryMode.PROVIDER_FILE_ID,
        reference="file-1",
    )
    assert uri_decision == FileDeliveryDecision(
        mode=FileDeliveryMode.OBJECT_STORE_URI,
        reference="gs://bucket/object",
    )
    assert url_decision == FileDeliveryDecision(
        mode=FileDeliveryMode.HOSTED_URL,
        reference="https://example.test/file",
    )


def test_plan_delivery_selects_inline_bytes_and_text() -> None:
    openai = resolve_file_delivery_profile("ai://env:KEY@openai/gpt-4o")
    local_text = resolve_file_delivery_profile("ai://local/model")
    multimodal = resolve_file_delivery_profile(
        "ai://local/model",
        local_profile=LocalFileDeliveryProfile.MULTIMODAL,
    )

    bytes_decision = openai.plan_delivery(
        FileDeliveryRequest(
            mime_type="application/pdf",
            size_bytes=12,
            has_artifact=True,
        )
    )
    text_decision = local_text.plan_delivery(
        FileDeliveryRequest(
            mime_type="text/plain",
            size_bytes=12,
            has_artifact=True,
        )
    )
    image_decision = multimodal.plan_delivery(
        FileDeliveryRequest(
            mime_type="image/png",
            size_bytes=12,
            has_artifact=True,
        )
    )

    assert bytes_decision.mode == FileDeliveryMode.INLINE_BYTES
    assert bytes_decision.needs_artifact_read
    assert text_decision.mode == FileDeliveryMode.INLINE_TEXT
    assert text_decision.needs_artifact_read
    assert image_decision.mode == FileDeliveryMode.INLINE_BYTES


def test_fake_profiles_cover_delivery_shapes() -> None:
    id_only = FileDeliveryProfile(
        name="id_only",
        delivery_modes=frozenset({FileDeliveryMode.PROVIDER_FILE_ID}),
    )
    inline_only = FileDeliveryProfile(
        name="inline_only",
        delivery_modes=frozenset(
            {
                FileDeliveryMode.INLINE_BYTES,
                FileDeliveryMode.INLINE_TEXT,
            }
        ),
        accepted_mime_types=("image/*",),
    )
    object_store_only = FileDeliveryProfile(
        name="object_store_only",
        delivery_modes=frozenset({FileDeliveryMode.OBJECT_STORE_URI}),
        object_store_uri_schemes=("s3",),
    )
    text_only = FileDeliveryProfile(
        name="text_only",
        delivery_modes=frozenset({FileDeliveryMode.INLINE_TEXT}),
        accepted_mime_types=("text/plain",),
    )
    local_none = FileDeliveryProfile(
        name="local_none",
        delivery_modes=frozenset({FileDeliveryMode.REJECT}),
        accepted_mime_types=(),
    )
    unknown_provider = resolve_file_delivery_profile(
        "ai://env:KEY@unknown/model"
    )

    assert (
        id_only.plan_delivery(
            FileDeliveryRequest(metadata={"provider_file_id": "file-1"})
        ).mode
        == FileDeliveryMode.PROVIDER_FILE_ID
    )
    assert (
        inline_only.plan_delivery(
            FileDeliveryRequest(
                mime_type="image/png",
                has_artifact=True,
            )
        ).mode
        == FileDeliveryMode.INLINE_BYTES
    )
    assert (
        object_store_only.plan_delivery(
            FileDeliveryRequest(metadata={"provider_uri": "s3://bucket/key"})
        ).mode
        == FileDeliveryMode.OBJECT_STORE_URI
    )
    assert (
        text_only.plan_delivery(
            FileDeliveryRequest(
                mime_type="text/plain",
                has_artifact=True,
            )
        ).mode
        == FileDeliveryMode.INLINE_TEXT
    )
    assert (
        local_none.plan_delivery(FileDeliveryRequest()).mode
        == FileDeliveryMode.REJECT
    )
    assert unknown_provider.plan_delivery(FileDeliveryRequest()).mode == (
        FileDeliveryMode.REJECT
    )


def test_plan_delivery_rejects_unsupported_profiles_and_inputs() -> None:
    unknown = resolve_file_delivery_profile("ai://env:KEY@unknown/model")
    bedrock = resolve_file_delivery_profile(
        "ai://env:KEY@bedrock/us.anthropic.claude"
    )
    local_text = resolve_file_delivery_profile("ai://local/model")

    unsupported = unknown.plan_delivery(FileDeliveryRequest())
    unsupported_mime = bedrock.plan_delivery(
        FileDeliveryRequest(
            mime_type="application/pdf",
            metadata={"provider_uri": "s3://bucket/object"},
        )
    )
    unsupported_uri = bedrock.plan_delivery(
        FileDeliveryRequest(
            mime_type="text/plain",
            metadata={"provider_uri": "gs://bucket/object"},
        )
    )
    no_mode = local_text.plan_delivery(
        FileDeliveryRequest(mime_type="application/pdf", has_artifact=True)
    )
    no_delivery_mode = local_text.plan_delivery(FileDeliveryRequest())
    provider_mismatch = bedrock.plan_delivery(
        FileDeliveryRequest(
            metadata={
                "provider_reference": {
                    "kind": "object_store_uri",
                    "provider": "google",
                    "reference": "s3://bucket/object",
                }
            }
        )
    )
    unsupported_reference = bedrock.plan_delivery(
        FileDeliveryRequest(
            metadata={
                "provider_reference": {
                    "kind": "provider_file_id",
                    "provider": "bedrock",
                    "reference": "file-1",
                }
            }
        )
    )
    unsupported_reference_uri = bedrock.plan_delivery(
        FileDeliveryRequest(
            metadata={
                "provider_reference": {
                    "kind": "object_store_uri",
                    "provider": "bedrock",
                    "reference": "gs://bucket/object",
                }
            }
        )
    )
    invalid_reference_metadata = bedrock.plan_delivery(
        FileDeliveryRequest(
            metadata={
                "provider_reference": {
                    "kind": "object_store_uri",
                    "provider": "bedrock",
                },
                "provider_uri": "s3://bucket/object",
            }
        )
    )

    assert unsupported.mode == FileDeliveryMode.REJECT
    assert unsupported.diagnostic is not None
    assert unsupported.diagnostic.code == "model.file_delivery.unsupported"
    assert unsupported_mime.diagnostic is not None
    assert (
        unsupported_mime.diagnostic.code
        == "model.file_delivery.unsupported_mime_type"
    )
    assert unsupported_uri.diagnostic is not None
    assert (
        unsupported_uri.diagnostic.code
        == "model.file_delivery.unsupported_object_store_uri"
    )
    assert no_mode.diagnostic is not None
    assert (
        no_mode.diagnostic.code == "model.file_delivery.unsupported_mime_type"
    )
    assert no_delivery_mode.diagnostic is not None
    assert (
        no_delivery_mode.diagnostic.code
        == "model.file_delivery.no_supported_delivery_mode"
    )
    assert provider_mismatch.diagnostic is not None
    assert (
        provider_mismatch.diagnostic.code
        == "model.file_delivery.provider_mismatch"
    )
    assert unsupported_reference.diagnostic is not None
    assert (
        unsupported_reference.diagnostic.code
        == "model.file_delivery.unsupported_provider_reference"
    )
    assert unsupported_reference_uri.diagnostic is not None
    assert (
        unsupported_reference_uri.diagnostic.code
        == "model.file_delivery.unsupported_object_store_uri"
    )
    assert invalid_reference_metadata.mode == FileDeliveryMode.OBJECT_STORE_URI


def test_plan_delivery_rejects_inline_limit_excess() -> None:
    text_profile = FileDeliveryProfile(
        name="tiny",
        delivery_modes=frozenset({FileDeliveryMode.INLINE_TEXT}),
        accepted_mime_types=("text/plain",),
        inline_text_limit=FileDeliveryLimit(
            name="tiny_text",
            source="test",
            max_bytes=4,
        ),
    )
    bytes_profile = FileDeliveryProfile(
        name="tiny_bytes",
        delivery_modes=frozenset(
            {
                FileDeliveryMode.INLINE_BYTES,
                FileDeliveryMode.INLINE_TEXT,
            }
        ),
        accepted_mime_types=("image/*",),
        inline_byte_limit=FileDeliveryLimit(
            name="tiny_bytes",
            source="test",
            max_bytes=4,
        ),
    )

    text_decision = text_profile.plan_delivery(
        FileDeliveryRequest(
            mime_type="text/plain",
            size_bytes=5,
            has_artifact=True,
        )
    )
    bytes_decision = bytes_profile.plan_delivery(
        FileDeliveryRequest(
            mime_type="image/png",
            size_bytes=5,
            has_artifact=True,
        )
    )

    assert text_decision.mode == FileDeliveryMode.REJECT
    assert text_decision.diagnostic is not None
    assert (
        text_decision.diagnostic.code
        == "model.file_delivery.inline_limit_exceeded"
    )
    assert "tiny_text" in text_decision.diagnostic.hint
    assert bytes_decision.mode == FileDeliveryMode.REJECT
    assert bytes_decision.diagnostic is not None
    assert (
        bytes_decision.diagnostic.code
        == "model.file_delivery.inline_limit_exceeded"
    )
    assert "tiny_bytes" in bytes_decision.diagnostic.hint


def test_profile_file_count_limits() -> None:
    unlimited = FileDeliveryProfile(
        name="unlimited",
        delivery_modes=frozenset({FileDeliveryMode.INLINE_TEXT}),
    )
    deferred = FileDeliveryProfile(
        name="deferred",
        delivery_modes=frozenset({FileDeliveryMode.INLINE_TEXT}),
        file_count_limit=FileDeliveryLimit(name="files", source="test"),
    )
    limited = FileDeliveryProfile(
        name="limited",
        delivery_modes=frozenset({FileDeliveryMode.INLINE_TEXT}),
        file_count_limit=FileDeliveryLimit(
            name="files",
            source="test",
            max_count=1,
        ),
    )

    assert unlimited.accepts_file_count(100)
    assert deferred.accepts_file_count(100)
    assert limited.accepts_file_count(1)
    assert not limited.accepts_file_count(2)


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
        lambda: FileDeliveryProfile(
            name="bad",
            delivery_modes=frozenset({FileDeliveryMode.REJECT}),
            inline_byte_limit=cast(FileDeliveryLimit, "limit"),
        ),
        lambda: FileDeliveryProfile(
            name="bad",
            delivery_modes=frozenset({FileDeliveryMode.REJECT}),
            requires_conversion_for_file_blocks=cast(bool, "yes"),
        ),
        lambda: FileDeliveryRequest(mime_type=""),
        lambda: FileDeliveryRequest(size_bytes=-1),
        lambda: FileDeliveryRequest(has_artifact=cast(bool, "yes")),
        lambda: FileDeliveryRequest(metadata=cast(MappingProxyType, {1: "x"})),
        lambda: FileDeliveryDecision(
            mode=cast(FileDeliveryMode, "reject"),
        ),
        lambda: FileDeliveryDecision(
            mode=FileDeliveryMode.REJECT,
            reference="",
        ),
        lambda: FileDeliveryDecision(
            mode=FileDeliveryMode.REJECT,
            diagnostic=cast(FileDeliveryDiagnostic, "diagnostic"),
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
        profile.accepts_file_count(-1)
    with pytest.raises(AssertionError):
        profile.accepts_file_count(cast(int, True))
    with pytest.raises(AssertionError):
        plan_file_delivery(
            cast(FileDeliveryProfile, object()), FileDeliveryRequest()
        )
    with pytest.raises(AssertionError):
        plan_file_delivery(profile, cast(FileDeliveryRequest, object()))
    with pytest.raises(AssertionError):
        resolve_file_delivery_profile(
            "ai://local/model",
            local_profile=cast(LocalFileDeliveryProfile, "text"),
        )
