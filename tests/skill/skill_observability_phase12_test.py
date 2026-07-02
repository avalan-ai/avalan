from collections.abc import Mapping
from datetime import datetime
from json import dumps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch
from uuid import uuid4

from avalan.entities import ToolCall, ToolCallContext
from avalan.event import Event, EventObservabilityPayload, EventType
from avalan.event.manager import (
    EventManager,
    EventManagerMode,
    EventSubscriberClass,
)
from avalan.skill import (
    SkillConfiguredSource,
    SkillObservabilitySettings,
    SkillPrivacySettings,
    SkillReadLimits,
    SkillRegistry,
    SkillResourceReader,
    SkillSourceAuthorityKind,
    SkillSourceConfig,
    SkillStatus,
    TrustedSkillSettings,
    WorkspaceSkillSourceAuthority,
    build_skill_registry,
    resolve_skill_sources,
)
from avalan.skill.observability import (
    SKILL_AUDIT_MAX_PAYLOAD_BYTES,
    SkillAuditDeliveryError,
    SkillEventPublisher,
    _compact_value,
    _force_payload_bound,
    assert_skill_event_publisher,
    skill_audit_authority_value,
    skill_audit_context_fields,
    skill_audit_events_enabled,
    skill_audit_hash_prefix,
    skill_audit_payload_data,
)
from avalan.skill.registry import _registry_source_authority_value
from avalan.task import (
    ObservabilitySinkHealth,
    PrivacySanitizer,
    TaskClient,
    TaskDefinition,
    TaskDirectTarget,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskIdempotencyIdentity,
    TaskInputContract,
    TaskInputType,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservedEvent,
    TaskOutputContract,
    TaskQueueArtifact,
    TaskQueueSubmission,
    TaskRunPolicy,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationContext,
    TaskValidationIssue,
    UsageSource,
    UsageTotals,
)
from avalan.task.canonical import canonical_json
from avalan.task.definition import TaskTargetType
from avalan.task.queue import TaskQueue
from avalan.task.runner import DirectTaskRunner
from avalan.task.skills import (
    build_task_skill_registry,
    task_skill_audit_event_publisher,
    task_skills_identity,
)
from avalan.task.stores import InMemoryTaskStore
from avalan.tool.skills import CheckSkillTool, MatchSkillsTool, ReadSkillTool

_BODY_MARKER = "phase12-skill-body-marker"
_HOST_PATH_MARKER = "/Users/mariano/.ssh/id_rsa"


class SkillObservabilityPhase12Test(IsolatedAsyncioTestCase):
    async def test_registry_events_are_ordered_correlated_and_redacted(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                body=f"{_BODY_MARKER}\n{_HOST_PATH_MARKER}\n",
            )
            event_manager = EventManager(mode=EventManagerMode.TEST)
            operation_id = "skill-registry-build:phase12"
            settings = _settings(root)

            resolution = await resolve_skill_sources(
                (_config(root),),
                settings=settings,
                event_manager=event_manager,
                audit_operation_id=operation_id,
            )
            registry = await build_skill_registry(
                resolution,
                settings=settings,
                event_manager=event_manager,
                audit_operation_id=operation_id,
            )

            self.assertEqual(registry.usable_metadata[0].skill_id, "pdf")
            types = _event_types(event_manager)
            start = types.index(EventType.SKILL_REGISTRY_BUILD_STARTED)
            registered = types.index(EventType.SKILL_REGISTERED)
            completed = types.index(EventType.SKILL_REGISTRY_BUILD_COMPLETED)
            self.assertLess(start, registered)
            self.assertLess(registered, completed)
            self.assertIn(EventType.SKILL_SOURCE_ACCEPTED, types)

            payloads = _event_payloads(event_manager)
            operation_ids = {
                payload.get("operation_id")
                for payload in payloads
                if payload.get("operation_id") is not None
            }
            self.assertEqual(operation_ids, {operation_id})
            for payload in payloads:
                self.assertLessEqual(
                    len(dumps(payload, sort_keys=True).encode("utf-8")),
                    SKILL_AUDIT_MAX_PAYLOAD_BYTES,
                )
            encoded = dumps(payloads, sort_keys=True)
            self.assertNotIn(_BODY_MARKER, encoded)
            self.assertNotIn(_HOST_PATH_MARKER, encoded)
            self.assertNotIn(str(root), encoded)
            self.assertNotIn("content_sha256", encoded)

            registered_payload = _payload_for(
                event_manager, EventType.SKILL_REGISTERED
            )
            hash_prefix = registered_payload["hash_prefix"]
            self.assertIsInstance(hash_prefix, str)
            hash_prefix_text = cast(str, hash_prefix)
            self.assertEqual(len(hash_prefix_text), 16)
            self.assertTrue(
                all(
                    character in "0123456789abcdef"
                    for character in hash_prefix_text
                )
            )
            self.assertEqual(registered_payload["source_label"], "workspace")
            self.assertEqual(
                registered_payload["source_authority"], "workspace"
            )

            snapshot = task_skills_identity(
                settings,
                registry=registry,
                enabled_tools=("skills.read",),
                target_type=TaskTargetType.AGENT,
            )
            snapshot_text = dumps(snapshot, sort_keys=True)
            self.assertNotIn(_BODY_MARKER, snapshot_text)
            self.assertNotIn(_HOST_PATH_MARKER, snapshot_text)
            self.assertNotIn(str(root), snapshot_text)

    async def test_event_publisher_accepts_dynamic_duck_trigger(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
            )
            publisher = _DynamicSkillEventPublisher()
            typed_publisher = cast(SkillEventPublisher, publisher)

            assert_skill_event_publisher(typed_publisher)
            resolution = await resolve_skill_sources(
                (_config(root),),
                settings=_settings(root),
                event_manager=typed_publisher,
            )
            registry = await build_skill_registry(
                resolution,
                settings=_settings(root),
                event_manager=typed_publisher,
            )

            self.assertEqual(registry.usable_metadata[0].skill_id, "pdf")
            self.assertIn(
                EventType.SKILL_SOURCE_ACCEPTED,
                publisher.event_types,
            )
            self.assertIn(
                EventType.SKILL_REGISTRY_BUILD_COMPLETED,
                publisher.event_types,
            )

    async def test_registry_emits_disabled_malformed_duplicate_and_shadowed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "valid" / "SKILL.md",
                name="valid",
                description="Valid guidance.",
            )
            _write_skill(
                root / "disabled" / "SKILL.md",
                name="disabled",
                description="Disabled guidance.",
                enabled=False,
            )
            _write_text(root / "malformed" / "SKILL.md", "not a manifest")
            _write_skill(
                root / "one" / "SKILL.md",
                name="duplicate",
                description="Duplicate one.",
            )
            _write_skill(
                root / "two" / "SKILL.md",
                name="duplicate",
                description="Duplicate two.",
            )
            event_manager = EventManager(mode=EventManagerMode.TEST)
            settings = _settings(root)

            resolution = await resolve_skill_sources(
                (_config(root),),
                settings=settings,
                event_manager=event_manager,
                audit_operation_id="skill-registry-build:classes",
            )
            await build_skill_registry(
                resolution,
                settings=settings,
                event_manager=event_manager,
                audit_operation_id="skill-registry-build:classes",
            )

            types = _event_types(event_manager)
            self.assertIn(EventType.SKILL_REGISTERED, types)
            self.assertIn(EventType.SKILL_DISABLED, types)
            self.assertIn(EventType.SKILL_MALFORMED, types)
            self.assertEqual(types.count(EventType.SKILL_DUPLICATE), 2)
            self.assertEqual(types.count(EventType.SKILL_SHADOWED), 1)

    async def test_registry_failure_events_and_authority_fallback(
        self,
    ) -> None:
        event_manager = EventManager(mode=EventManagerMode.TEST)
        with patch(
            "avalan.skill.registry.parse_skill_manifests",
            side_effect=RuntimeError("registry failed"),
        ):
            with self.assertRaises(RuntimeError):
                await build_skill_registry((), event_manager=event_manager)

        self.assertIn(
            EventType.SKILL_REGISTRY_BUILD_FAILED,
            _event_types(event_manager),
        )
        self.assertEqual(
            _payload_for(
                event_manager,
                EventType.SKILL_REGISTRY_BUILD_FAILED,
            )["status"],
            SkillStatus.BLOCKED.value,
        )

        settings = TrustedSkillSettings(
            observability=SkillObservabilitySettings(audit_fail_closed=True)
        )
        fail_closed_manager = EventManager(mode=EventManagerMode.TEST)

        def fail(_: Event) -> None:
            raise RuntimeError("critical registry delivery failed")

        fail_closed_manager.add_listener(
            fail,
            [EventType.SKILL_REGISTRY_BUILD_COMPLETED],
            subscriber_class=EventSubscriberClass.CRITICAL,
        )
        with self.assertRaises(SkillAuditDeliveryError):
            await build_skill_registry(
                (),
                settings=settings,
                event_manager=fail_closed_manager,
            )

        self.assertIsNone(_registry_source_authority_value({}, "missing"))

    async def test_resolver_generates_audit_operation_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
            )
            event_manager = EventManager(mode=EventManagerMode.TEST)

            await resolve_skill_sources(
                (_config(root),),
                settings=_settings(root),
                event_manager=event_manager,
            )

        operation_id = _payload_for(
            event_manager,
            EventType.SKILL_SOURCE_ACCEPTED,
        )["operation_id"]
        self.assertIsInstance(operation_id, str)
        self.assertRegex(
            cast(str, operation_id),
            r"^(skill-source-resolve:[a-f0-9]{16}|id-[a-f0-9]{16})$",
        )

    async def test_match_events_cover_candidates_empty_and_ambiguous(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "basic" / "SKILL.md",
                name="pdf-basic",
                description="Basic PDF guidance.",
            )
            _write_skill(
                root / "advanced" / "SKILL.md",
                name="pdf-advanced",
                description="Advanced PDF guidance.",
            )
            registry = await _registry(root)
            event_manager = EventManager(mode=EventManagerMode.TEST)
            tool = MatchSkillsTool(registry, event_manager=event_manager)
            agent_id = uuid4()
            session_id = uuid4()
            context = ToolCallContext(
                agent_id=agent_id,
                session_id=session_id,
                calls=(
                    [
                        ToolCall(
                            id="match-call",
                            name="skills.match",
                            arguments={"query": "pdf"},
                        )
                    ]
                ),
            )

            await tool(context, query="pdf-basic")
            await tool(context, query="missing")
            await tool(context, query="pdf")

            types = _event_types(event_manager)
            self.assertEqual(
                types,
                [
                    EventType.SKILL_MATCH_QUERY_EVALUATED,
                    EventType.SKILL_MATCH_CANDIDATES_RETURNED,
                    EventType.SKILL_MATCH_QUERY_EVALUATED,
                    EventType.SKILL_MATCH_EMPTY,
                    EventType.SKILL_MATCH_QUERY_EVALUATED,
                    EventType.SKILL_MATCH_AMBIGUOUS,
                ],
            )
            for payload in _event_payloads(event_manager):
                self.assertEqual(payload["agent_id"], str(agent_id))
                self.assertEqual(payload["session_id"], str(session_id))
                self.assertEqual(payload["tool_call_id"], "match-call")
                self.assertNotIn("query", payload)

    async def test_read_and_check_events_cover_resource_outcomes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "allowed" / "SKILL.md",
                name="allowed",
                description="Allowed guidance.",
                body="Use this guidance for a long enough read window.\n",
            )
            _write_text(root / "malformed" / "SKILL.md", "not a manifest")

            allowed_manager = EventManager(mode=EventManagerMode.TEST)
            registry = await _registry(root)
            read_context = ToolCallContext(
                calls=(
                    [
                        ToolCall(
                            id="read-call",
                            name="skills.read",
                            arguments={"skill": "allowed"},
                        )
                    ]
                )
            )
            await ReadSkillTool(
                registry,
                SkillResourceReader(),
                event_manager=allowed_manager,
            )(read_context, skill="allowed")
            await ReadSkillTool(
                registry,
                SkillResourceReader(
                    read_limits=SkillReadLimits(max_bytes_per_read=32)
                ),
                event_manager=allowed_manager,
            )(read_context, skill="allowed")
            await ReadSkillTool(
                registry,
                SkillResourceReader(),
                event_manager=allowed_manager,
            )(read_context, skill="malformed")

            stale_manager = EventManager(mode=EventManagerMode.TEST)
            stale_registry = await _registry(root)
            _write_skill(
                root / "allowed" / "SKILL.md",
                name="allowed",
                description="Allowed guidance changed.",
            )
            await ReadSkillTool(
                stale_registry,
                SkillResourceReader(),
                event_manager=stale_manager,
            )(read_context, skill="allowed")
            check_context = ToolCallContext(
                calls=(
                    [
                        ToolCall(
                            id="check-call",
                            name="skills.check",
                            arguments={"skill": "allowed"},
                        )
                    ]
                )
            )
            await CheckSkillTool(
                stale_registry,
                SkillResourceReader(),
                event_manager=stale_manager,
            )(check_context, skill="allowed")

            denied_manager = EventManager(mode=EventManagerMode.TEST)
            denied_registry = await _registry(
                root,
                settings=_settings(root, allowed_skill_ids=("other",)),
            )
            await ReadSkillTool(
                denied_registry,
                SkillResourceReader(),
                event_manager=denied_manager,
            )(read_context, skill="allowed")

            deleted_manager = EventManager(mode=EventManagerMode.TEST)
            deleted_registry = await _registry(root)
            (root / "allowed" / "SKILL.md").unlink()
            await ReadSkillTool(
                deleted_registry,
                SkillResourceReader(),
                event_manager=deleted_manager,
            )(read_context, skill="allowed")

            read_types = (
                _event_types(allowed_manager)
                + _event_types(stale_manager)
                + _event_types(deleted_manager)
                + _event_types(denied_manager)
            )
            self.assertIn(EventType.SKILL_READ_ALLOWED, read_types)
            self.assertIn(EventType.SKILL_READ_TRUNCATED, read_types)
            self.assertIn(EventType.SKILL_READ_BLOCKED, read_types)
            self.assertIn(EventType.SKILL_READ_STALE, read_types)
            self.assertIn(EventType.SKILL_READ_DELETED, read_types)
            self.assertIn(EventType.SKILL_READ_DENIED, read_types)
            self.assertIn(
                EventType.SKILL_CHECK_DIAGNOSTICS_PRODUCED,
                _event_types(stale_manager),
            )

            encoded = dumps(
                _event_payloads(allowed_manager)
                + _event_payloads(stale_manager)
                + _event_payloads(deleted_manager)
                + _event_payloads(denied_manager),
                sort_keys=True,
            )
            self.assertNotIn(_BODY_MARKER, encoded)
            self.assertNotIn(str(root), encoded)

    async def test_fail_closed_audit_delivery_blocks_registry_exposure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
            )
            settings = _settings(
                root,
                observability=SkillObservabilitySettings(
                    audit_fail_closed=True
                ),
            )
            event_manager = EventManager(mode=EventManagerMode.TEST)

            def fail(_: Event) -> None:
                raise RuntimeError(f"raw {_HOST_PATH_MARKER}")

            event_manager.add_listener(
                fail,
                [EventType.SKILL_REGISTRY_BUILD_STARTED],
                subscriber_class=EventSubscriberClass.CRITICAL,
            )
            resolution = await resolve_skill_sources(
                (_config(root),),
                settings=settings,
            )

            with self.assertRaises(SkillAuditDeliveryError) as error:
                await build_skill_registry(
                    resolution,
                    settings=settings,
                    event_manager=event_manager,
                )

            self.assertNotIn(_HOST_PATH_MARKER, str(error.exception))

            open_settings = _settings(root)
            open_registry = await build_skill_registry(
                resolution,
                settings=open_settings,
                event_manager=event_manager,
            )
            self.assertEqual(open_registry.usable_metadata[0].skill_id, "pdf")

    async def test_audit_events_respect_hidden_source_label_privacy(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_label = "private-source"
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                body="Hidden source labels must not leak in audits.\n",
            )
            settings = _settings(
                root,
                source_label=source_label,
                privacy=SkillPrivacySettings(include_source_labels=False),
            )
            event_manager = EventManager(mode=EventManagerMode.TEST)

            resolution = await resolve_skill_sources(
                (_config(root, source_label=source_label),),
                settings=settings,
                event_manager=event_manager,
                audit_operation_id="skill-registry-build:hidden-labels",
            )
            registry = await build_skill_registry(
                resolution,
                settings=settings,
                event_manager=event_manager,
                audit_operation_id="skill-registry-build:hidden-labels",
            )
            await ReadSkillTool(
                registry,
                SkillResourceReader(),
                event_manager=event_manager,
            )(ToolCallContext(), skill="pdf")

            payloads = _event_payloads(event_manager)
            encoded = dumps(payloads, sort_keys=True)
            self.assertNotIn(source_label, encoded)
            for payload in payloads:
                self.assertNotIn("source_label", payload)

    async def test_task_skill_identity_snapshots_hide_source_labels(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_label = "id_rsa"
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
                body="Task metadata must not expose hidden labels.\n",
            )
            settings = _settings(
                root,
                source_label=source_label,
                privacy=SkillPrivacySettings(include_source_labels=False),
            )
            registry = await build_task_skill_registry(settings)

            snapshot = task_skills_identity(
                settings,
                registry=registry,
                enabled_tools=("skills.read",),
                target_type=TaskTargetType.TOOL,
            )
            self.assertEqual(snapshot["source_labels"], ())

            canonical = await canonical_json(_task_definition(settings))
            queue = _RecordingQueue(InMemoryTaskStore())
            client = TaskClient(
                queue.store,
                target=_NoopTaskTargetRunner(),
                queue=cast(TaskQueue, queue),
                hmac_provider=_StaticHmacProvider(),
            )
            submission = await client.enqueue(
                _task_definition(
                    settings,
                    execution=TaskExecutionTarget.agent("agent"),
                    input_contract=TaskInputContract(
                        type=TaskInputType.STRING,
                        required=False,
                    ),
                    run=TaskRunPolicy.queued("default"),
                )
            )

            encoded = str(
                {
                    "snapshot": snapshot,
                    "canonical": canonical,
                    "request_metadata": queue.requests[0].metadata,
                    "submitted_request_metadata": (
                        submission.run.request.metadata
                    ),
                }
            )
            self.assertNotIn(source_label, encoded)
            self.assertNotIn(str(root), encoded)
            self.assertIn("source_fingerprint", encoded)

    async def test_denied_read_and_check_redact_sensitive_handles(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
            )
            hidden_registry = await _registry(
                root,
                settings=_settings(
                    root,
                    privacy=(
                        SkillPrivacySettings(include_source_labels=False)
                    ),
                ),
            )
            denied_registry = await _registry(
                root,
                settings=_settings(root, allowed_skill_ids=("other",)),
            )
            event_manager = EventManager(mode=EventManagerMode.TEST)
            read_context = ToolCallContext(
                calls=(
                    [
                        ToolCall(
                            id="read-denied",
                            name="skills.read",
                            arguments={"resource_id": "id_rsa"},
                        )
                    ]
                )
            )
            check_context = ToolCallContext(
                calls=(
                    [
                        ToolCall(
                            id="check-denied",
                            name="skills.check",
                            arguments={"resource_id": "id_rsa"},
                        )
                    ]
                )
            )

            await ReadSkillTool(
                hidden_registry,
                SkillResourceReader(),
                event_manager=event_manager,
            )(
                read_context,
                skill="pdf",
                resource_id="id_rsa",
                source_label="workspace",
            )
            await CheckSkillTool(
                hidden_registry,
                SkillResourceReader(),
                event_manager=event_manager,
            )(
                check_context,
                skill="pdf",
                resource_id="id_rsa",
                source_label="workspace",
            )
            await ReadSkillTool(
                denied_registry,
                SkillResourceReader(),
                event_manager=event_manager,
            )(read_context, skill="pdf", resource_id="id_rsa")
            await ReadSkillTool(
                denied_registry,
                SkillResourceReader(),
                event_manager=event_manager,
            )(read_context, skill="id_rsa")
            await CheckSkillTool(
                denied_registry,
                SkillResourceReader(),
                event_manager=event_manager,
            )(check_context, skill="id_rsa")

            encoded = dumps(_event_payloads(event_manager), sort_keys=True)
            self.assertNotIn("id_rsa", encoded)
            self.assertNotIn("source_label", encoded)
            for payload in _event_payloads(event_manager):
                resource_id = payload.get("resource_id")
                if resource_id is not None:
                    self.assertNotEqual(resource_id, "id_rsa")

    async def test_audit_payload_size_is_bounded_after_compaction(
        self,
    ) -> None:
        oversized = "a" * 12_000
        payload = skill_audit_payload_data(
            EventType.SKILL_READ_ALLOWED,
            {
                "agent_id": oversized,
                "session_id": oversized,
                "tool_call_id": oversized,
                "operation_id": oversized,
                "registry_version": oversized,
                "source_authority": oversized,
                "source_label": oversized,
                "skill_id": oversized,
                "resource_id": oversized,
                "status": oversized,
                "diagnostic_code": oversized,
                "extra_payload": [oversized] * 32,
            },
        )

        self.assertLessEqual(
            len(dumps(payload, sort_keys=True).encode("utf-8")),
            SKILL_AUDIT_MAX_PAYLOAD_BYTES,
        )
        self.assertIs(payload["payload_truncated"], True)

    async def test_task_registry_fail_closed_blocks_registry_exposure(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
            )
            settings = _settings(
                root,
                observability=SkillObservabilitySettings(
                    audit_fail_closed=True
                ),
            )
            event_manager = EventManager(mode=EventManagerMode.TEST)

            def fail(_: Event) -> None:
                raise RuntimeError(f"raw {_HOST_PATH_MARKER}")

            event_manager.add_listener(
                fail,
                [EventType.SKILL_REGISTRY_BUILD_STARTED],
                subscriber_class=EventSubscriberClass.CRITICAL,
            )

            with self.assertRaises(SkillAuditDeliveryError) as error:
                await build_task_skill_registry(
                    settings,
                    event_manager=event_manager,
                )

            self.assertNotIn(_HOST_PATH_MARKER, str(error.exception))

    async def test_task_registry_fail_closed_without_publisher_is_noop(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
            )
            settings = _settings(
                root,
                observability=SkillObservabilitySettings(
                    audit_fail_closed=True
                ),
            )

            self.assertIsNone(
                task_skill_audit_event_publisher(sanitizer=PrivacySanitizer())
            )
            registry = await build_task_skill_registry(settings)

            self.assertEqual(registry.usable_metadata[0].skill_id, "pdf")

    async def test_task_skill_audit_publisher_awaits_trace_and_sink(
        self,
    ) -> None:
        traced: list[TaskObservedEvent] = []
        sink = _TaskAuditSink()

        async def trace(event: TaskObservedEvent) -> None:
            traced.append(event)

        publisher = task_skill_audit_event_publisher(
            sanitizer=PrivacySanitizer(),
            trace_event_observer=trace,
            observability_sink=sink,
        )
        assert publisher is not None

        await publisher.trigger(
            Event(
                type=EventType.SKILL_READ_ALLOWED,
                payload={"status": "ok"},
            )
        )

        self.assertEqual(len(traced), 1)
        self.assertEqual(len(sink.events), 1)
        self.assertEqual(traced[0].event_type, "skill_read_allowed")

    async def test_audit_helper_privacy_and_compaction_edges(self) -> None:
        event = Event.from_observability_payload(
            type=EventType.SKILL_READ_ALLOWED,
            observability_payload=EventObservabilityPayload.canonical_stream(
                {"status": "ok"}
            ),
        )
        event_manager = EventManager(mode=EventManagerMode.TEST)
        assert_skill_event_publisher(event_manager)
        await event_manager.trigger(event)
        self.assertEqual(event_manager.history[0].type, event.type)
        self.assertTrue(skill_audit_events_enabled(None))
        self.assertIsNone(skill_audit_authority_value(None))
        self.assertEqual(
            skill_audit_authority_value(SkillSourceAuthorityKind.WORKSPACE),
            "workspace",
        )
        self.assertIsNone(skill_audit_hash_prefix(None))
        self.assertIsNone(skill_audit_hash_prefix("not-a-hash"))

        context_fields = skill_audit_context_fields(
            ToolCallContext(
                calls=(
                    [
                        ToolCall(
                            id=None,
                            name="skills.read",
                            arguments={},
                        ),
                        ToolCall(
                            id="other",
                            name="skills.match",
                            arguments={},
                        ),
                    ]
                )
            ),
            tool_name="skills.read",
        )
        self.assertNotIn("tool_call_id", context_fields)

        hidden_settings = TrustedSkillSettings(
            privacy=SkillPrivacySettings(
                include_source_labels=False,
                include_authority=False,
            )
        )
        audit_uuid = uuid4()
        hidden_payload = skill_audit_payload_data(
            EventType.SKILL_SOURCE_ACCEPTED,
            {
                "agent_id": uuid4(),
                "source_labels": ("private-source",),
                "source_id": "source:private-source",
                "source_authority": "workspace",
                "skill_id": "id_rsa",
                "skill_ids": ("pdf", "id_rsa"),
                "resource_id": 3,
                "resource_ids": ("id_rsa",),
                "status": SkillStatus.OK,
                "metadata": {"path": _HOST_PATH_MARKER, "uuid": audit_uuid},
                "opaque": object(),
            },
            settings=hidden_settings,
        )
        hidden_text = dumps(hidden_payload, sort_keys=True)
        self.assertNotIn("source_label", hidden_text)
        self.assertNotIn("source_authority", hidden_text)
        self.assertNotIn("private-source", hidden_text)
        self.assertNotIn("id_rsa", hidden_text)
        self.assertNotIn(_HOST_PATH_MARKER, hidden_text)
        self.assertIn("redacted", hidden_text)
        self.assertIn(str(audit_uuid), hidden_text)
        skill_ids = cast(list[object], hidden_payload["skill_ids"])
        self.assertEqual(skill_ids[0], "pdf")
        self.assertTrue(str(hidden_payload["skill_id"]).startswith("skill-"))
        self.assertTrue(str(skill_ids[1]).startswith("skill-"))

        identifier_payload = skill_audit_payload_data(
            EventType.SKILL_MATCH_QUERY_EVALUATED,
            {
                "agent_id": "build:.ssh",
                "session_id": "session:.config",
                "tool_call_id": "call:.env",
                "operation_id": "id_rsa",
            },
        )
        identifier_text = dumps(identifier_payload, sort_keys=True)
        self.assertNotIn("id_rsa", identifier_text)
        self.assertNotIn("build:.ssh", identifier_text)
        self.assertNotIn("session:.config", identifier_text)
        self.assertNotIn("call:.env", identifier_text)
        self.assertTrue(str(identifier_payload["agent_id"]).startswith("id-"))
        self.assertTrue(
            str(identifier_payload["session_id"]).startswith("id-")
        )
        self.assertTrue(
            str(identifier_payload["tool_call_id"]).startswith("id-")
        )
        self.assertTrue(
            str(identifier_payload["operation_id"]).startswith("id-")
        )

        invalid_identifier_payload = skill_audit_payload_data(
            EventType.SKILL_MATCH_QUERY_EVALUATED,
            {
                "agent_id": "bad/slash",
                "session_id": "build..id",
            },
        )
        invalid_identifier_text = dumps(
            invalid_identifier_payload,
            sort_keys=True,
        )
        self.assertNotIn("bad/slash", invalid_identifier_text)
        self.assertNotIn("build..id", invalid_identifier_text)
        self.assertTrue(
            str(invalid_identifier_payload["agent_id"]).startswith("id-")
        )
        self.assertTrue(
            str(invalid_identifier_payload["session_id"]).startswith("id-")
        )

        visible_payload = skill_audit_payload_data(
            EventType.SKILL_SOURCE_ACCEPTED,
            {
                "source_label": "id_rsa",
                "source_labels": "not-a-list",
                "source_id": "source:id_rsa",
                "skill_id": "PDF Skill",
            },
        )
        self.assertTrue(
            str(visible_payload["source_label"]).startswith("source-")
        )
        self.assertEqual(visible_payload["source_labels"], [])
        self.assertTrue(
            str(visible_payload["source_id"]).startswith("source:source-")
        )
        self.assertEqual(visible_payload["skill_id"], "pdf-skill")

        oversized = "a" * 12_000
        drop_payload = skill_audit_payload_data(
            EventType.SKILL_CHECK_DIAGNOSTICS_PRODUCED,
            {
                "diagnostic_codes": [oversized] * 32,
                "resource_ids": [oversized] * 32,
                "status": "ok",
            },
        )
        self.assertIs(drop_payload["payload_truncated"], True)
        self.assertNotIn("diagnostic_codes", drop_payload)
        self.assertLessEqual(
            len(dumps(drop_payload, sort_keys=True).encode("utf-8")),
            SKILL_AUDIT_MAX_PAYLOAD_BYTES,
        )

        compact_list: list[object] = [oversized] * 32
        forced_payload = _force_payload_bound(
            {
                "schema": "skills.audit.v1",
                "event_type": "skill_read_allowed",
                "policy_version": "skills.settings.phase9",
                "payload_truncated": True,
                "status": compact_list,
                "diagnostic_code": compact_list,
                "hash_prefix": compact_list,
            }
        )
        self.assertNotIn("diagnostic_code", forced_payload)
        self.assertIn("hash_prefix", forced_payload)
        self.assertLessEqual(
            len(dumps(forced_payload, sort_keys=True).encode("utf-8")),
            SKILL_AUDIT_MAX_PAYLOAD_BYTES,
        )

        loop_payload = skill_audit_payload_data(
            EventType.SKILL_READ_ALLOWED,
            {
                "agent_id": compact_list,
                "session_id": compact_list,
                "tool_call_id": compact_list,
                "operation_id": compact_list,
                "registry_version": compact_list,
                "source_authority": compact_list,
                "skill_id": compact_list,
                "status": "ok",
                "extra_payload": compact_list,
            },
        )
        self.assertIs(loop_payload["payload_truncated"], True)
        self.assertLessEqual(
            len(dumps(loop_payload, sort_keys=True).encode("utf-8")),
            SKILL_AUDIT_MAX_PAYLOAD_BYTES,
        )

        minimal_payload = skill_audit_payload_data(
            EventType.SKILL_READ_ALLOWED,
            {
                "agent_id": compact_list,
                "status": compact_list,
                "size_bytes": compact_list,
                "start_byte": compact_list,
                "end_byte": compact_list,
                "extra_payload": compact_list,
            },
        )
        self.assertIs(minimal_payload["payload_truncated"], True)
        self.assertLessEqual(
            len(dumps(minimal_payload, sort_keys=True).encode("utf-8")),
            SKILL_AUDIT_MAX_PAYLOAD_BYTES,
        )

        fallback_payload = skill_audit_payload_data(
            EventType.SKILL_READ_ALLOWED,
            {
                "agent_id": compact_list,
                "status": {
                    f"entry_{index}": compact_list for index in range(8)
                },
                "extra_payload": compact_list,
            },
        )
        self.assertEqual(
            fallback_payload["event_type"], "skills.audit.truncated"
        )
        self.assertLessEqual(
            len(dumps(fallback_payload, sort_keys=True).encode("utf-8")),
            SKILL_AUDIT_MAX_PAYLOAD_BYTES,
        )
        self.assertEqual(_compact_value(object()), "redacted")

    async def test_task_client_and_runner_deliver_skill_audit_fail_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_skill(
                root / "pdf" / "SKILL.md",
                name="pdf",
                description="PDF rendering guidance.",
            )
            settings = _settings(
                root,
                observability=SkillObservabilitySettings(
                    audit_fail_closed=True
                ),
            )
            observed: list[object] = []
            target = cast(TaskDirectTarget, _noop_task_target)
            client = TaskClient(
                InMemoryTaskStore(),
                target=target,
                event_observer=observed.append,
            )

            _ = await client.validate(_task_definition(settings))

            event_types = [
                event.event_type
                for event in observed
                if hasattr(event, "event_type")
            ]
            self.assertIn("skill_registry_build_started", event_types)
            self.assertIn("skill_registry_build_completed", event_types)

            def fail(_: object) -> None:
                raise RuntimeError(f"raw {_HOST_PATH_MARKER}")

            failing_client = TaskClient(
                InMemoryTaskStore(),
                target=target,
                event_observer=fail,
            )
            with self.assertRaises(SkillAuditDeliveryError) as client_error:
                await failing_client.validate(_task_definition(settings))
            self.assertNotIn(_HOST_PATH_MARKER, str(client_error.exception))

            runner = DirectTaskRunner(
                InMemoryTaskStore(),
                target=target,
                event_observer=fail,
            )
            with self.assertRaises(SkillAuditDeliveryError) as runner_error:
                await runner.run(_task_definition(settings))
            self.assertNotIn(_HOST_PATH_MARKER, str(runner_error.exception))

            queued_client = TaskClient(
                InMemoryTaskStore(),
                target=target,
                queue=cast(TaskQueue, object()),
                event_observer=fail,
            )
            with self.assertRaises(SkillAuditDeliveryError) as enqueue_error:
                await queued_client.enqueue(
                    _task_definition(
                        settings,
                        run=TaskRunPolicy.queued("default"),
                    )
                )
            self.assertNotIn(_HOST_PATH_MARKER, str(enqueue_error.exception))


async def _registry(
    root: Path,
    *,
    settings: TrustedSkillSettings | None = None,
) -> SkillRegistry:
    if settings is None:
        settings = _settings(root)
    resolution = await resolve_skill_sources(
        (_config(root),),
        settings=settings,
    )
    return await build_skill_registry(resolution, settings=settings)


def _settings(
    root: Path,
    *,
    allowed_skill_ids: tuple[str, ...] = (),
    observability: SkillObservabilitySettings | None = None,
    privacy: SkillPrivacySettings | None = None,
    source_label: str = "workspace",
) -> TrustedSkillSettings:
    return TrustedSkillSettings(
        sources=(
            SkillSourceConfig(
                label=source_label,
                authority=WorkspaceSkillSourceAuthority(),
                root_path=root,
            ),
        ),
        allowed_skill_ids=allowed_skill_ids,
        privacy=privacy if privacy is not None else SkillPrivacySettings(),
        observability=(
            observability
            if observability is not None
            else SkillObservabilitySettings()
        ),
    )


def _config(
    root: Path,
    *,
    source_label: str = "workspace",
) -> SkillConfiguredSource:
    return SkillConfiguredSource(
        label=source_label,
        authority=WorkspaceSkillSourceAuthority(),
        root_path=root,
    )


def _task_definition(
    settings: TrustedSkillSettings,
    *,
    execution: TaskExecutionTarget | None = None,
    input_contract: TaskInputContract | None = None,
    run: TaskRunPolicy | None = None,
) -> TaskDefinition:
    assert isinstance(settings, TrustedSkillSettings)
    assert execution is None or isinstance(execution, TaskExecutionTarget)
    assert input_contract is None or isinstance(
        input_contract, TaskInputContract
    )
    return TaskDefinition(
        task=TaskMetadata(name="skills", version="1"),
        input=input_contract or TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=execution or TaskExecutionTarget.tool("skills.read"),
        skills=settings,
        run=run or TaskRunPolicy.direct(),
    )


async def _noop_task_target(_: TaskTargetContext) -> object:
    return "ok"


class _NoopTaskTargetRunner(TaskTargetRunner):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        _ = definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> str:
        _ = context
        return "ok"


class _StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"test-secret",
        )


class _TaskAuditSink:
    def __init__(self) -> None:
        self.events: list[TaskObservedEvent] = []

    async def record_event(self, event: TaskObservedEvent) -> None:
        self.events.append(event)

    async def record_usage(
        self,
        *,
        run_id: str,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        _ = run_id, source, totals, attempt_id, metadata

    def health(self) -> ObservabilitySinkHealth:
        return ObservabilitySinkHealth(
            name="task-audit-test",
            event_count=len(self.events),
        )


class _DynamicSkillEventPublisher:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def __getattr__(self, name: str) -> object:
        if name == "trigger":
            return self._trigger
        raise AttributeError(name)

    async def _trigger(self, event: Event) -> None:
        self.events.append(event)

    @property
    def event_types(self) -> list[EventType]:
        return [event.type for event in self.events]


class _RecordingQueue:
    def __init__(self, store: InMemoryTaskStore) -> None:
        self.store = store
        self.requests: list[TaskExecutionRequest] = []

    async def enqueue_run(
        self,
        request: TaskExecutionRequest,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        idempotency: TaskIdempotencyIdentity | None = None,
        idempotency_expires_at: datetime | None = None,
        artifacts: tuple[TaskQueueArtifact, ...] = (),
        run_metadata: Mapping[str, object] | None = None,
        queue_metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSubmission:
        _ = (
            queue_name,
            priority,
            available_at,
            idempotency,
            idempotency_expires_at,
            artifacts,
            queue_metadata,
        )
        self.requests.append(request)
        run = await self.store.create_run(request, metadata=run_metadata)
        return TaskQueueSubmission(run=run, created=True)


def _write_skill(
    path: Path,
    *,
    name: str,
    description: str,
    enabled: bool = True,
    body: str = "",
) -> None:
    enabled_line = "" if enabled else "enabled: false\n"
    _write_text(
        path,
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{enabled_line}"
        "---\n"
        f"{body}",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _event_types(event_manager: EventManager) -> list[EventType]:
    return [event.type for event in event_manager.history]


def _event_payloads(event_manager: EventManager) -> list[dict[str, object]]:
    return [dict(event.observability.data) for event in event_manager.history]


def _payload_for(
    event_manager: EventManager,
    event_type: EventType,
) -> dict[str, object]:
    for event in event_manager.history:
        if event.type is event_type:
            return dict(event.observability.data)
    raise AssertionError(f"missing event {event_type.value}")


if __name__ == "__main__":
    main()
