from avalan.task import (
    IdempotencyMode,
    TaskDefinition,
    TaskExecutionTarget,
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskInputContract,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskOutputContract,
    TaskProviderReferenceKind,
    TaskRunPolicy,
)


def build_large_direct_definition() -> TaskDefinition:
    """Return a direct file task definition."""
    return TaskDefinition(
        task=TaskMetadata(
            name="large_direct_file",
            version="1",
            description="Read a direct document input with bounded file delivery.",
            labels=("example", "file"),
        ),
        input=TaskInputContract.file(
            conversions=("text", "markdown"),
            mime_types=("text/plain", "text/markdown", "application/pdf"),
            description="Document descriptor.",
        ),
        output=TaskOutputContract.text(description="Document answer."),
        execution=TaskExecutionTarget.agent("agents/document_reader.toml"),
        run=TaskRunPolicy.direct(
            timeout_seconds=180,
            idempotency=IdempotencyMode.INPUT_AND_FILES_HASH,
        ),
        limits=TaskLimitsPolicy(
            file_count=1,
            file_bytes=1048576,
            output_bytes=8192,
            total_tokens=3000,
        ),
    )


def local_document_descriptor(path: str) -> TaskFileDescriptor:
    """Return a local document descriptor with a conversion request."""
    return TaskFileDescriptor.local_path(
        path,
        role="source",
        mime_type="application/pdf",
        size_bytes=2048,
        conversions=(TaskFileConversionRequest(name="text"),),
    )


def provider_file_id_descriptor(file_id: str) -> TaskFileDescriptor:
    """Return a provider file-id descriptor."""
    return TaskFileDescriptor.provider_reference_descriptor(
        file_id,
        kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
        provider="openai",
        owner_scope="tenant-a",
        mime_type="application/pdf",
        size_bucket="small",
        identity_hmac="hmac:file",
        role="source",
    )


def hosted_url_descriptor(url: str) -> TaskFileDescriptor:
    """Return a provider-hosted URL descriptor."""
    return TaskFileDescriptor.provider_reference_descriptor(
        url,
        kind=TaskProviderReferenceKind.HOSTED_URL,
        provider="anthropic",
        mime_type="application/pdf",
        size_bucket="small",
        identity_hmac="hmac:url",
        role="source",
    )


def object_store_descriptor(uri: str) -> TaskFileDescriptor:
    """Return an object-store URI descriptor."""
    return TaskFileDescriptor.provider_reference_descriptor(
        uri,
        kind=TaskProviderReferenceKind.OBJECT_STORE_URI,
        provider="google",
        mime_type="application/pdf",
        size_bucket="small",
        identity_hmac="hmac:object",
        role="source",
    )


if __name__ == "__main__":
    print(build_large_direct_definition())
