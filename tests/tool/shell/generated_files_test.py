"""Verify content-specific image paths across shell execution backends."""

from .image_fixtures import VALID_JPEG_BYTES, valid_png_bytes

from base64 import b64decode
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from avalan.container import (
    ContainerOutputArtifact,
    ContainerOutputContractType,
)
from avalan.sandbox.backend import SandboxOutputArtifact
from avalan.tool.shell.container import (
    _generated_file as container_generated_file,
)
from avalan.tool.shell.entities import (
    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY,
    GeneratedOutputPlan,
)
from avalan.tool.shell.executor import _collect_generated_files
from avalan.tool.shell.sandbox import (
    _generated_files as sandbox_generated_files,
)
from avalan.tool.shell.settings import ShellToolSettings


class GeneratedImageBackendsTest(IsolatedAsyncioTestCase):
    async def test_backends_share_content_specific_image_paths(self) -> None:
        cases = (
            (".png", "image/png", valid_png_bytes(width=2, height=3), (2, 3)),
            (".jpg", "image/jpeg", VALID_JPEG_BYTES, (16, 16)),
        )
        for inline_limit in (0, 1024):
            for suffix, media_type, pixels, dimensions in cases:
                with (
                    self.subTest(inline_limit=inline_limit, suffix=suffix),
                    TemporaryDirectory() as temporary_directory,
                ):
                    root = Path(temporary_directory)
                    settings = ShellToolSettings(workspace_root=str(root))
                    output = root / "outputs"
                    output.mkdir(mode=0o700)
                    filename = f"page-1{suffix}"
                    (output / filename).write_bytes(pixels)
                    plan = GeneratedOutputPlan(
                        prefix_name="page",
                        display_prefix="GENERATED_PREFIX",
                        allowed_suffixes=(suffix,),
                        suffix_media_types={suffix: media_type},
                        max_files=1,
                        max_file_bytes=1024,
                        max_total_bytes=1024,
                        max_inline_bytes=inline_limit,
                    )
                    digest = sha256(pixels).hexdigest()
                    local = await _collect_generated_files(
                        plan, output / "page", 64, settings=settings
                    )
                    sandbox = await sandbox_generated_files(
                        (
                            SandboxOutputArtifact(
                                path=filename, content=pixels
                            ),
                        ),
                        plan,
                        settings=settings,
                    )
                    container = container_generated_file(
                        ContainerOutputArtifact(
                            artifact_type=ContainerOutputContractType.GENERATED_FILE,
                            path=filename,
                            size_bytes=len(pixels),
                            media_type=media_type,
                            digest=f"sha256:{digest}",
                            signature=pixels,
                            content=pixels,
                        ),
                        plan,
                    )

                    for generated in (*local, *sandbox, container):
                        self.assertEqual(
                            generated.display_path,
                            f"GENERATED_PREFIX-1-{digest}{suffix}",
                        )
                        self.assertEqual(generated.sha256, digest)
                        self.assertEqual(generated.bytes, len(pixels))
                        self.assertEqual(
                            (generated.width, generated.height), dimensions
                        )
                        if generated.transient_content is not None:
                            self.assertEqual(
                                generated.transient_content, pixels
                            )
                        elif generated.content_base64 is not None:
                            self.assertEqual(
                                b64decode(generated.content_base64), pixels
                            )
                        else:
                            source = generated.metadata[
                                GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY
                            ]
                            assert isinstance(source, str)
                            self.assertEqual(Path(source).read_bytes(), pixels)
