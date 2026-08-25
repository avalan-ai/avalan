"""Reject an unsealed string where a pinned worker image is required."""

from avalan.patch.container_target import ContainerPatchImage

image: ContainerPatchImage = "python:3.11-slim-bookworm"
