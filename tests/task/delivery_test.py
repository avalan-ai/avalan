from io import BytesIO
from unittest import IsolatedAsyncioTestCase, main

from avalan.model import (
    FileDeliveryLimit,
    FileDeliveryMode,
    FileDeliveryProfile,
)
from avalan.task import (
    TaskArtifactPolicy,
    TaskArtifactRef,
    TaskArtifactStat,
    TaskDefinition,
    TaskExecutionTarget,
    TaskFileDeliveryPlan,
    TaskInputContract,
    TaskInputFile,
    TaskLimitsPolicy,
    TaskMetadata,
    TaskOutputContract,
    TaskProviderReference,
    TaskProviderReferenceKind,
    plan_task_file_delivery,
)
from avalan.task import delivery as delivery_module
from avalan.task.artifact import ArtifactStoreError


class RecordingArtifactStore:
    def __init__(
        self,
        *,
        size_bytes: int = 4,
        stat_error: bool = False,
    ) -> None:
        self.size_bytes = size_bytes
        self.stat_error = stat_error
        self.open_count = 0
        self.stat_count = 0

    async def open(self, ref: TaskArtifactRef) -> BytesIO:
        self.open_count += 1
        return BytesIO(b"x" * self.size_bytes)

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        self.stat_count += 1
        if self.stat_error:
            raise ArtifactStoreError("private backend failure")
        return TaskArtifactStat(
            ref=ref,
            size_bytes=self.size_bytes,
            sha256="0" * 64,
        )


class TaskDeliveryPlannerTest(IsolatedAsyncioTestCase):
    async def test_provider_reference_wins_over_artifact_inline(self) -> None:
        store = RecordingArtifactStore()
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="provider:file",
                artifact_ref=_artifact_ref(size_bytes=4),
                provider_reference=TaskProviderReference(
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    provider="test",
                    reference="file-test",
                ),
                media_type="text/plain",
                size_bytes=4,
            ),
            profile=_profile(
                FileDeliveryMode.PROVIDER_FILE_ID,
                FileDeliveryMode.INLINE_BYTES,
                FileDeliveryMode.INLINE_TEXT,
            ),
            artifact_store=store,
        )

        self.assertIsInstance(plan, TaskFileDeliveryPlan)
        self.assertEqual(plan.decision.mode, FileDeliveryMode.PROVIDER_FILE_ID)
        self.assertEqual(plan.decision.reference, "file-test")
        self.assertEqual(store.stat_count, 0)
        self.assertEqual(store.open_count, 0)

    async def test_legacy_reference_metadata_is_not_executable(self) -> None:
        legacy_plan = await plan_task_file_delivery(
            _definition(artifact_max_bytes=None),
            TaskInputFile(
                logical_path="provider:file",
                media_type="application/pdf",
                metadata={"provider_file_id": "file-private"},
            ),
            profile=_profile(FileDeliveryMode.PROVIDER_FILE_ID),
        )
        metadata_plan = await plan_task_file_delivery(
            _definition(artifact_max_bytes=None),
            TaskInputFile(
                logical_path="provider:file",
                media_type="application/pdf",
                metadata={
                    "provider_reference": {
                        "kind": "provider_file_id",
                        "provider": "test",
                        "reference": "file-private",
                    }
                },
            ),
            profile=_profile(FileDeliveryMode.PROVIDER_FILE_ID),
        )

        self.assertEqual(legacy_plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertIsNotNone(legacy_plan.decision.diagnostic)
        assert legacy_plan.decision.diagnostic is not None
        self.assertEqual(
            legacy_plan.decision.diagnostic.code,
            "task.file_delivery.rejected",
        )
        self.assertNotIn("file-private", str(legacy_plan.decision))
        self.assertEqual(metadata_plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertIsNotNone(metadata_plan.decision.diagnostic)
        assert metadata_plan.decision.diagnostic is not None
        self.assertEqual(
            metadata_plan.decision.diagnostic.code,
            "task.file_delivery.rejected",
        )
        self.assertNotIn("file-private", str(metadata_plan.decision))

    async def test_typed_provider_reference_supplies_mime_type(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(artifact_max_bytes=None),
            TaskInputFile(
                logical_path="provider:file",
                provider_reference=TaskProviderReference(
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    provider="test",
                    reference="file-test",
                    mime_type="application/pdf",
                ),
            ),
            profile=_profile(FileDeliveryMode.PROVIDER_FILE_ID),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.PROVIDER_FILE_ID)
        self.assertEqual(plan.decision.reference, "file-test")

    async def test_profile_source_kind_rejects_provider_reference(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(artifact_max_bytes=None),
            TaskInputFile(
                logical_path="provider:file",
                provider_reference=TaskProviderReference(
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    provider="test",
                    reference="file-private",
                    mime_type="application/pdf",
                ),
            ),
            profile=_profile(
                FileDeliveryMode.PROVIDER_FILE_ID,
                source_kinds=frozenset({"artifact"}),
            ),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertIsNotNone(plan.decision.diagnostic)
        assert plan.decision.diagnostic is not None
        self.assertEqual(
            plan.decision.diagnostic.code,
            "model.file_delivery.unsupported_source_kind",
        )
        self.assertNotIn("file-private", str(plan.decision))

    async def test_reference_metadata_does_not_require_artifact_stat(
        self,
    ) -> None:
        store = RecordingArtifactStore()
        plan = await plan_task_file_delivery(
            _definition(artifact_max_bytes=None),
            TaskInputFile(
                logical_path="provider:file",
                artifact_ref=_artifact_ref(),
                media_type="application/pdf",
                provider_reference=TaskProviderReference(
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    provider="test",
                    reference="file-test",
                ),
            ),
            profile=_profile(FileDeliveryMode.PROVIDER_FILE_ID),
            artifact_store=store,
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.PROVIDER_FILE_ID)
        self.assertEqual(store.stat_count, 0)

    async def test_task_limit_rejects_before_artifact_open(self) -> None:
        store = RecordingArtifactStore()
        plan = await plan_task_file_delivery(
            _definition(limits=TaskLimitsPolicy(file_bytes=3)),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="application/octet-stream",
                size_bytes=4,
            ),
            profile=_profile(FileDeliveryMode.INLINE_BYTES),
            artifact_store=store,
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.limit_exceeded",
        )
        self.assertIn("task_file_bytes", plan.decision.diagnostic.hint)
        self.assertIn("size_bucket=1B-1KB", plan.decision.diagnostic.hint)
        self.assertNotIn("4", plan.decision.diagnostic.hint)
        self.assertEqual(store.open_count, 0)

    async def test_provider_inline_limit_accounts_for_base64_size(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="application/octet-stream",
                size_bytes=4,
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_BYTES,
                inline_byte_limit=FileDeliveryLimit(
                    name="inline_file_bytes",
                    source="provider.test",
                    max_bytes=5,
                ),
            ),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.limit_exceeded",
        )
        self.assertIn("inline_file_bytes", plan.decision.diagnostic.hint)

    async def test_inline_bytes_without_provider_limit_succeeds(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="application/octet-stream",
                size_bytes=4,
            ),
            profile=_profile(FileDeliveryMode.INLINE_BYTES),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.INLINE_BYTES)

    async def test_inline_image_delivery_uses_flattened_dimensions(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:page",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="image/png",
                size_bytes=4,
                metadata={
                    "dimensions": {
                        "width_pixels": 512,
                        "height_pixels": 512,
                    }
                },
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_IMAGE,
                accepted_mime_types=("image/*",),
                vision_token_limit=FileDeliveryLimit(
                    name="vision_tokens",
                    source="provider.test",
                    max_tokens=85,
                ),
            ),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.INLINE_IMAGE)

    async def test_inline_image_delivery_rejects_unknown_vision_cost(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:page",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="image/png",
                size_bytes=4,
                metadata={
                    "dimensions": {
                        "width_pixels": 512,
                        "height_pixels": "private",
                    }
                },
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_IMAGE,
                accepted_mime_types=("image/*",),
                vision_token_limit=FileDeliveryLimit(
                    name="vision_tokens",
                    source="provider.test",
                    max_tokens=85,
                ),
            ),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "model.file_delivery.unknown_vision_tokens",
        )
        self.assertNotIn("private", str(plan.decision))

    async def test_inline_image_requires_estimate_without_limit(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:page",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="image/png",
                size_bytes=4,
                metadata={"pixels": "private"},
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_IMAGE,
                FileDeliveryMode.INLINE_TEXT,
                accepted_mime_types=("image/*",),
            ),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "model.file_delivery.unknown_vision_tokens",
        )
        self.assertIn("vision_tokens", plan.decision.diagnostic.hint)
        self.assertNotIn("private", str(plan.decision))

    async def test_unknown_inline_byte_size_rejects_with_limit_name(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(artifact_max_bytes=None),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(),
                media_type="application/octet-stream",
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_BYTES,
                inline_byte_limit=FileDeliveryLimit(
                    name="inline_file_bytes",
                    source="provider.test",
                    max_bytes=5,
                ),
            ),
            artifact_store=RecordingArtifactStore(stat_error=True),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.unknown_size",
        )
        self.assertIn("inline_file_bytes", plan.decision.diagnostic.hint)

    async def test_provider_limit_falls_back_to_inline_text(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="text/plain",
                size_bytes=4,
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_BYTES,
                FileDeliveryMode.INLINE_TEXT,
                inline_byte_limit=FileDeliveryLimit(
                    name="inline_file_bytes",
                    source="provider.test",
                    max_bytes=5,
                ),
                inline_text_limit=FileDeliveryLimit(
                    name="inline_text_bytes",
                    source="provider.test",
                    max_bytes=8,
                ),
            ),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.INLINE_TEXT)

    async def test_inline_text_limit_rejects_without_fallback(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="text/plain",
                size_bytes=4,
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_TEXT,
                inline_text_limit=FileDeliveryLimit(
                    name="inline_text_bytes",
                    source="provider.test",
                    max_bytes=3,
                ),
            ),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.limit_exceeded",
        )
        self.assertIn("inline_text_bytes", plan.decision.diagnostic.hint)

    async def test_unknown_size_uses_stat_before_limited_inline(self) -> None:
        store = RecordingArtifactStore(size_bytes=6)
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(),
                media_type="text/plain",
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_TEXT,
                inline_text_limit=FileDeliveryLimit(
                    name="inline_text_bytes",
                    source="provider.test",
                    max_bytes=8,
                ),
            ),
            artifact_store=store,
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.INLINE_TEXT)
        self.assertEqual(plan.size_bytes, 6)
        self.assertEqual(store.stat_count, 1)
        self.assertEqual(store.open_count, 0)

    async def test_unknown_size_rejects_when_stat_fails(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(artifact_max_bytes=None),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(),
                media_type="text/plain",
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_TEXT,
                inline_text_limit=FileDeliveryLimit(
                    name="inline_text_bytes",
                    source="provider.test",
                    max_bytes=8,
                ),
            ),
            artifact_store=RecordingArtifactStore(stat_error=True),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.unknown_size",
        )
        self.assertNotIn("private backend failure", str(plan.decision))

    async def test_missing_stat_rejects_unknown_task_limited_size(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(
                limits=TaskLimitsPolicy(file_bytes=16),
                artifact_max_bytes=None,
            ),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(),
                media_type="text/plain",
            ),
            profile=_profile(FileDeliveryMode.INLINE_TEXT),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.unknown_size",
        )
        self.assertIn("task_file_bytes", plan.decision.diagnostic.hint)

    async def test_missing_artifact_store_rejects_artifact_delivery(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="text/plain",
                size_bytes=4,
            ),
            profile=_profile(FileDeliveryMode.INLINE_TEXT),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.missing_artifact_store",
        )

    async def test_unsupported_mime_rejects_without_conversion(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="application/pdf",
                size_bytes=4,
            ),
            profile=_profile(
                FileDeliveryMode.INLINE_TEXT,
                accepted_mime_types=("text/*",),
            ),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.unsupported_mime",
        )

    async def test_artifact_file_without_mime_rejects_safely(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                size_bytes=4,
            ),
            profile=_profile(FileDeliveryMode.INLINE_TEXT),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.unsupported_mime",
        )

    async def test_declared_conversion_falls_back_after_mime_mismatch(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(
                input_contract=TaskInputContract.file(
                    conversions=("text",),
                    mime_types=("application/pdf",),
                )
            ),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=4),
                media_type="application/pdf",
                size_bytes=4,
            ),
            profile=_profile(
                FileDeliveryMode.CONVERTED_ARTIFACT,
                accepted_mime_types=("text/*",),
            ),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(
            plan.decision.mode,
            FileDeliveryMode.CONVERTED_ARTIFACT,
        )

    async def test_retrieval_fallback_requires_context_budget(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(limits=TaskLimitsPolicy(total_tokens=16)),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=16),
                media_type="text/plain",
                size_bytes=16,
            ),
            profile=_profile(FileDeliveryMode.RETRIEVAL_CONTEXT),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(
            plan.decision.mode,
            FileDeliveryMode.RETRIEVAL_CONTEXT,
        )

    async def test_map_reduce_fallback_covers_no_retrieval_budget(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=16),
                media_type="text/plain",
                size_bytes=16,
            ),
            profile=_profile(FileDeliveryMode.MAP_REDUCE_CONTEXT),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(
            plan.decision.mode,
            FileDeliveryMode.MAP_REDUCE_CONTEXT,
        )

    async def test_rejects_when_no_mode_applies(self) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=16),
                media_type="text/plain",
                size_bytes=16,
            ),
            profile=_profile(FileDeliveryMode.REJECT),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.unsupported",
        )

    async def test_supported_profile_rejects_when_no_candidate_applies(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:input",
                artifact_ref=_artifact_ref(size_bytes=16),
                media_type="text/plain",
                size_bytes=16,
            ),
            profile=_profile(FileDeliveryMode.PROVIDER_FILE_ID),
            artifact_store=RecordingArtifactStore(),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.rejected",
        )

    async def test_non_artifact_without_reference_reaches_generic_reject(
        self,
    ) -> None:
        plan = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(logical_path="inline:missing"),
            profile=_profile(FileDeliveryMode.INLINE_TEXT),
        )

        self.assertEqual(plan.decision.mode, FileDeliveryMode.REJECT)
        self.assertEqual(
            plan.decision.diagnostic.code,
            "task.file_delivery.rejected",
        )

    async def test_size_buckets_cover_empty_and_large_files(self) -> None:
        empty = await plan_task_file_delivery(
            _definition(),
            TaskInputFile(
                logical_path="artifact:empty",
                artifact_ref=_artifact_ref(size_bytes=0),
                media_type="text/plain",
                size_bytes=0,
            ),
            profile=_profile(FileDeliveryMode.INLINE_TEXT),
            artifact_store=RecordingArtifactStore(size_bytes=0),
        )
        large = await plan_task_file_delivery(
            _definition(artifact_max_bytes=None),
            TaskInputFile(
                logical_path="provider:file",
                media_type="application/pdf",
                size_bytes=(100 * 1024 * 1024) + 1,
                provider_reference=TaskProviderReference(
                    kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                    provider="test",
                    reference="file-test",
                ),
            ),
            profile=_profile(FileDeliveryMode.PROVIDER_FILE_ID),
        )

        self.assertEqual(empty.size_bucket, "0B")
        self.assertEqual(large.size_bucket, "100MB+")

    def test_mime_type_matching_covers_exact_and_negative_branches(
        self,
    ) -> None:
        self.assertTrue(
            delivery_module._mime_type_matches("image/png", "image/png")
        )
        self.assertFalse(
            delivery_module._mime_type_matches("application/pdf", "image/*")
        )
        self.assertFalse(
            delivery_module._mime_type_matches("application/pdf", "text/plain")
        )


def _definition(
    *,
    input_contract: TaskInputContract | None = None,
    limits: TaskLimitsPolicy | None = None,
    artifact_max_bytes: int | None = 64,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="delivery", version="1.0"),
        input=input_contract or TaskInputContract.file(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        artifact=TaskArtifactPolicy(max_bytes=artifact_max_bytes),
        limits=limits or TaskLimitsPolicy(),
    )


def _artifact_ref(
    *,
    size_bytes: int | None = None,
) -> TaskArtifactRef:
    return TaskArtifactRef(
        artifact_id="artifact-1",
        store="local",
        storage_key="ar/artifact-1",
        size_bytes=size_bytes,
    )


def _profile(
    *modes: FileDeliveryMode,
    accepted_mime_types: tuple[str, ...] = ("*/*",),
    inline_byte_limit: FileDeliveryLimit | None = None,
    inline_image_limit: FileDeliveryLimit | None = None,
    inline_text_limit: FileDeliveryLimit | None = None,
    source_kinds: frozenset[str] | None = None,
    vision_token_limit: FileDeliveryLimit | None = None,
) -> FileDeliveryProfile:
    if source_kinds is not None:
        return FileDeliveryProfile(
            name="test",
            delivery_modes=frozenset(modes),
            accepted_mime_types=accepted_mime_types,
            inline_byte_limit=inline_byte_limit,
            inline_image_limit=inline_image_limit,
            inline_text_limit=inline_text_limit,
            source_kinds=source_kinds,
            vision_token_limit=vision_token_limit,
        )
    return FileDeliveryProfile(
        name="test",
        delivery_modes=frozenset(modes),
        accepted_mime_types=accepted_mime_types,
        inline_byte_limit=inline_byte_limit,
        inline_image_limit=inline_image_limit,
        inline_text_limit=inline_text_limit,
        vision_token_limit=vision_token_limit,
    )


if __name__ == "__main__":
    main()
