"""Reject untyped storage-axis values."""

from avalan.conversation import StoragePolicy

INVALID_POLICY = StoragePolicy(
    local="durable",
    upstream="stored",
    provider_storage_disclosed=True,
)
