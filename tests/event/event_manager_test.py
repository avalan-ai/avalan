import asyncio
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import (
    TOOL_TYPES,
    Event,
    EventObservabilityPayload,
    EventPayloadKind,
    EventStats,
    EventType,
)
from avalan.event.manager import (
    EventDeliveryConfig,
    EventDeliveryPolicy,
    EventHistoryConfig,
    EventListenConfig,
    EventManager,
    EventManagerDefaults,
    EventManagerMode,
    EventSubscriberClass,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    stream_observability_payload,
)


class EventManagerTestCase(IsolatedAsyncioTestCase):
    def test_event_can_wrap_canonical_observability_payload(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="session-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="large text omitted from observability",
            metadata={"source": "test"},
        )
        payload = EventObservabilityPayload.canonical_stream(
            stream_observability_payload(item)
        )

        event = Event.from_observability_payload(
            type=EventType.TOKEN_GENERATED,
            observability_payload=payload,
        )

        self.assertIs(event.observability, payload)
        self.assertIs(event.for_history(), event)
        self.assertEqual(event.payload, payload.data)
        self.assertEqual(
            event.observability.to_dict(),
            {
                "kind": EventPayloadKind.CANONICAL_STREAM.value,
                "data": {
                    "stream_session_id": "session-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "sequence": 0,
                    "kind": StreamItemKind.ANSWER_DELTA.value,
                    "channel": StreamChannel.ANSWER.value,
                    "visibility": "public",
                    "summary": {
                        "text_delta_length": 37,
                        "metadata_keys": ["source"],
                    },
                },
            },
        )

    async def test_trigger_stream_item_publishes_canonical_projection(
        self,
    ) -> None:
        manager = EventManager(history_config=EventHistoryConfig(max_events=4))
        called: list[Event] = []

        async def listener(event: Event) -> None:
            called.append(event)

        manager.add_observability_listener(
            listener,
            [EventType.TOKEN_GENERATED],
            include_token_events=True,
        )
        item = CanonicalStreamItem(
            stream_session_id="session-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="hello",
        )

        await manager.trigger_stream_item(item)
        await asyncio.sleep(0)

        self.assertEqual(len(called), 1)
        event = called[0]
        self.assertIs(event.type, EventType.TOKEN_GENERATED)
        self.assertIs(
            event.observability.kind,
            EventPayloadKind.CANONICAL_STREAM,
        )
        self.assertEqual(
            event.payload,
            {
                "stream_session_id": "session-1",
                "run_id": "run-1",
                "turn_id": "turn-1",
                "sequence": 0,
                "kind": StreamItemKind.ANSWER_DELTA.value,
                "channel": StreamChannel.ANSWER.value,
                "visibility": "public",
                "summary": {"text_delta_length": 5},
            },
        )
        self.assertEqual(manager.history, [event])
        subscriber = manager._subscriber_index[listener]
        self.assertIs(
            subscriber.config.policy,
            EventDeliveryPolicy.DROP,
        )
        self.assertEqual(subscriber.config.queue_limit, 32)

    async def test_trigger_stream_item_skips_default_token_without_subscriber(
        self,
    ) -> None:
        manager = EventManager()
        item = CanonicalStreamItem(
            stream_session_id="session-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="hello",
        )

        await manager.trigger_stream_item(item)

        self.assertEqual(manager.history, [])
        self.assertEqual(manager.stats.published, 0)

    async def test_trigger_stream_item_rejects_invalid_inputs(self) -> None:
        manager = EventManager()
        item = CanonicalStreamItem(
            stream_session_id="session-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=0,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="hello",
        )

        with self.assertRaises(AssertionError):
            await manager.trigger_stream_item(cast(Any, object()))
        with self.assertRaises(AssertionError):
            await manager.trigger_stream_item(
                item,
                event_type=cast(Any, "token_generated"),
            )

    def test_listener_policy_helpers_use_bounded_defaults(self) -> None:
        manager = EventManager()

        def ui_listener(event: Event) -> None:
            _ = event

        def observability_listener(event: Event) -> None:
            _ = event

        manager.add_ui_listener(ui_listener)
        manager.add_observability_listener(
            observability_listener,
            [EventType.TOKEN_GENERATED],
            include_token_events=True,
        )

        ui_subscriber = manager._subscriber_index[ui_listener]
        observability_subscriber = manager._subscriber_index[
            observability_listener
        ]
        self.assertIs(
            ui_subscriber.config.policy,
            EventDeliveryPolicy.COALESCE,
        )
        self.assertEqual(ui_subscriber.config.queue_limit, 64)
        self.assertIs(
            observability_subscriber.config.policy,
            EventDeliveryPolicy.DROP,
        )
        self.assertEqual(observability_subscriber.config.queue_limit, 32)
        self.assertTrue(manager.should_emit(EventType.TOKEN_GENERATED))

        with self.assertRaises(AssertionError):
            manager.add_ui_listener(
                ui_listener,
                include_token_events=cast(Any, "yes"),
            )
        with self.assertRaises(AssertionError):
            manager.add_observability_listener(
                observability_listener,
                include_token_events=cast(Any, "yes"),
            )

    def test_legacy_event_observability_is_summary_only(self) -> None:
        event = Event(
            type=EventType.START,
            payload={"response": object(), "context": "secret"},
            started=1,
            finished=2,
            elapsed=1,
        )

        payload = event.observability

        self.assertIs(payload.kind, EventPayloadKind.TEMPORARY_LEGACY)
        self.assertEqual(payload.owner, "event-listener-facade")
        self.assertEqual(
            payload.data,
            {
                "event_type": EventType.START.value,
                "started": 1.0,
                "finished": 2.0,
                "elapsed": 1.0,
                "payload_summary": {
                    "type": "dict",
                    "keys": ["context", "response"],
                    "size": 2,
                },
            },
        )
        self.assertNotIn("response", payload.to_dict()["data"])

    def test_legacy_event_observability_summarizes_sequences(self) -> None:
        event = Event(type=EventType.START, payload=["a", "b"])

        self.assertEqual(
            event.observability.data["payload_summary"],
            {"type": "list", "size": 2},
        )

    def test_legacy_event_observability_bounds_mapping_keys(self) -> None:
        event = Event(
            type=EventType.START,
            payload={f"k{i:02d}": i for i in range(20)},
        )

        summary = cast(
            dict[str, object], event.observability.data["payload_summary"]
        )

        self.assertEqual(summary["type"], "dict")
        self.assertEqual(
            summary["keys"],
            [f"k{i:02d}" for i in range(16)],
        )
        self.assertEqual(summary["size"], 20)
        self.assertIs(summary["keys_truncated"], True)
        self.assertNotIn("k19", repr(summary["keys"]))

    def test_legacy_event_observability_bounds_mapping_key_length(
        self,
    ) -> None:
        key = "k" * 200
        event = Event(type=EventType.START, payload={key: "value"})

        summary = cast(
            dict[str, object], event.observability.data["payload_summary"]
        )

        self.assertEqual(summary["keys"], ["k" * 125 + "..."])
        self.assertEqual(summary["size"], 1)
        self.assertIs(summary["keys_truncated"], True)
        self.assertNotIn(key, repr(summary["keys"]))

    async def test_history_stores_legacy_observability_projection(
        self,
    ) -> None:
        manager = EventManager(
            history_config=EventHistoryConfig(max_bytes=512)
        )
        event = Event(
            type=EventType.START,
            payload={"response": "x" * 10_000},
        )

        await manager.trigger(event)

        self.assertEqual(len(manager.history), 1)
        history_event = manager.history[0]
        self.assertIsNone(history_event.payload)
        self.assertIs(
            history_event.observability.kind,
            EventPayloadKind.TEMPORARY_LEGACY,
        )
        self.assertEqual(
            history_event.observability.data["payload_summary"],
            {"type": "dict", "keys": ["response"], "size": 1},
        )

    async def test_history_drops_payload_when_observability_is_summary(
        self,
    ) -> None:
        manager = EventManager(
            history_config=EventHistoryConfig(max_bytes=512)
        )
        event = Event(
            type=EventType.TOKEN_GENERATED,
            payload={"token": "x" * 10_000},
            observability_payload=EventObservabilityPayload.temporary_legacy(
                {
                    "event_type": EventType.TOKEN_GENERATED.value,
                    "token_length": 10_000,
                },
                owner="token-event-listener-facade",
                removal_condition="Listeners consume canonical projections.",
            ),
        )

        manager.add_listener(lambda event: None, [EventType.TOKEN_GENERATED])
        await manager.trigger(event)

        self.assertEqual(len(manager.history), 1)
        history_event = manager.history[0]
        self.assertIsNone(history_event.payload)
        self.assertEqual(
            history_event.observability.data,
            {
                "event_type": EventType.TOKEN_GENERATED.value,
                "token_length": 10_000,
            },
        )
        self.assertNotIn("x" * 100, repr(history_event))

    def test_observability_payload_rejects_invalid_definitions(self) -> None:
        with self.assertRaises(AssertionError):
            EventObservabilityPayload(
                kind=cast(Any, "canonical_stream"),
                data={"event_type": "start"},
            )
        with self.assertRaises(AssertionError):
            EventObservabilityPayload.canonical_stream({})
        with self.assertRaises(AssertionError):
            EventObservabilityPayload.canonical_stream({cast(Any, ""): "bad"})
        with self.assertRaises(AssertionError):
            EventObservabilityPayload.temporary_legacy(
                {"event_type": "start"},
                owner="",
                removal_condition="done",
            )
        with self.assertRaises(AssertionError):
            EventObservabilityPayload.temporary_legacy(
                {"event_type": "start"},
                owner="owner",
                removal_condition="",
            )
        with self.assertRaises(AssertionError):
            EventObservabilityPayload(
                kind=EventPayloadKind.CANONICAL_STREAM,
                data={"event_type": "start"},
                owner="owner",
            )
        with self.assertRaises(AssertionError):
            Event(type=EventType.START, started=-1)
        invalid_event_times = (
            lambda: Event(type=EventType.START, started=cast(Any, True)),
            lambda: Event(type=EventType.START, finished=cast(Any, True)),
            lambda: Event(type=EventType.START, elapsed=cast(Any, True)),
        )
        for build_event in invalid_event_times:
            with self.subTest(build_event=build_event):
                with self.assertRaises(AssertionError):
                    build_event()
        with self.assertRaises(AssertionError):
            Event(
                type=EventType.START,
                observability_payload=cast(Any, object()),
            )

    def test_tool_diagnostic_event_is_tool_event(self) -> None:
        event = Event(type=EventType.TOOL_DIAGNOSTIC)

        self.assertIs(event.type, EventType.TOOL_DIAGNOSTIC)
        self.assertIn(EventType.TOOL_DIAGNOSTIC, TOOL_TYPES)
        self.assertEqual(EventType.TOOL_DIAGNOSTIC.value, "tool_diagnostic")

    def test_tool_progress_event_is_tool_event(self) -> None:
        event = Event(
            type=EventType.TOOL_PROGRESS,
            payload={
                "call_id": "call1",
                "progress": {"current": 1, "total": 2},
            },
        )

        self.assertIs(event.type, EventType.TOOL_PROGRESS)
        self.assertIn(EventType.TOOL_PROGRESS, TOOL_TYPES)
        self.assertEqual(EventType.TOOL_PROGRESS.value, "tool_progress")
        assert event.payload is not None
        self.assertEqual(event.payload["call_id"], "call1")
        self.assertEqual(
            event.payload["progress"],
            {"current": 1, "total": 2},
        )

    async def test_trigger_and_history(self) -> None:
        manager = EventManager(history_length=2)
        called: list[tuple[str, EventType]] = []

        async def a_listener(event: Event) -> None:
            called.append(("a", event.type))

        def s_listener(event: Event) -> None:
            called.append(("s", event.type))

        manager.add_listener(a_listener, [EventType.START])
        manager.add_listener(s_listener)

        evt1 = Event(type=EventType.START)
        evt2 = Event(type=EventType.END)
        await manager.trigger(evt1)
        await manager.trigger(evt2)

        self.assertIn(("a", EventType.START), called)
        self.assertIn(("s", EventType.START), called)
        self.assertIn(("s", EventType.END), called)
        self.assertEqual(len(manager.history), 2)

        evt3 = Event(type=EventType.START)
        await manager.trigger(evt3)
        self.assertEqual(manager.history, [evt2, evt3])
        self.assertEqual(manager.stats.published, 3)
        self.assertEqual(manager.stats.delivered, 5)
        self.assertEqual(manager.stats.triggers[EventType.START], 2)
        self.assertEqual(manager.stats.triggers[EventType.END], 1)

    async def test_disabled_history_still_delivers_events(self) -> None:
        manager = EventManager(
            history_config=EventHistoryConfig(enabled=False)
        )
        evt = Event(type=EventType.START)

        await manager.trigger(evt)

        self.assertEqual(manager.history, [])
        gen = manager.listen(stop_signal=None)
        self.assertIs(await gen.__anext__(), evt)
        self.assertEqual(manager.stats.published, 1)
        self.assertEqual(manager.stats.delivered, 1)
        self.assertEqual(manager.stats.queue_depth, 0)

    async def test_disabled_listen_queue_does_not_retain_events(
        self,
    ) -> None:
        manager = EventManager(listen_config=EventListenConfig(enabled=False))

        await manager.trigger(Event(type=EventType.START))

        self.assertEqual(manager.stats.published, 1)
        self.assertEqual(manager.stats.queue_depth, 0)
        gen = manager.listen(stop_signal=None, timeout=0.01)
        with self.assertRaises(StopAsyncIteration):
            await gen.__anext__()

    async def test_listen_queue_drops_when_full(self) -> None:
        manager = EventManager(
            listen_config=EventListenConfig(
                queue_limit=1,
                policy=EventDeliveryPolicy.DROP,
            )
        )
        first = Event(type=EventType.START, payload={"n": 1})
        second = Event(type=EventType.END, payload={"n": 2})

        await manager.trigger(first)
        await manager.trigger(second)

        gen = manager.listen(stop_signal=None, timeout=0.01)
        self.assertIs(await gen.__anext__(), first)
        with self.assertRaises(StopAsyncIteration):
            await gen.__anext__()
        self.assertEqual(manager.stats.dropped, 1)

    async def test_listen_queue_coalesces_when_full(self) -> None:
        manager = EventManager(
            listen_config=EventListenConfig(
                queue_limit=1,
                policy=EventDeliveryPolicy.COALESCE,
            )
        )
        first = Event(type=EventType.START, payload={"n": 1})
        second = Event(type=EventType.END, payload={"n": 2})

        await manager.trigger(first)
        await manager.trigger(second)

        gen = manager.listen(stop_signal=None, timeout=0.01)
        self.assertIs(await gen.__anext__(), second)
        with self.assertRaises(StopAsyncIteration):
            await gen.__anext__()
        self.assertEqual(manager.stats.coalesced, 1)

    async def test_history_evicts_oversized_events_by_byte_limit(
        self,
    ) -> None:
        manager = EventManager(history_config=EventHistoryConfig(max_bytes=1))

        await manager.trigger(
            Event(type=EventType.START, payload={"text": "too large"})
        )

        self.assertEqual(manager.history, [])

    async def test_history_expires_events_by_ttl(self) -> None:
        manager = EventManager(
            history_config=EventHistoryConfig(ttl_seconds=0)
        )

        await manager.trigger(Event(type=EventType.START))

        self.assertEqual(manager.history, [])

    def test_history_config_rejects_invalid_settings(self) -> None:
        with self.assertRaises(AssertionError):
            EventHistoryConfig(enabled=cast(Any, "yes"))
        with self.assertRaises(AssertionError):
            EventHistoryConfig(max_events=-1)
        with self.assertRaises(AssertionError):
            EventHistoryConfig(max_events=cast(Any, True))
        with self.assertRaises(AssertionError):
            EventHistoryConfig(max_bytes=-1)
        with self.assertRaises(AssertionError):
            EventHistoryConfig(max_bytes=cast(Any, False))
        with self.assertRaises(AssertionError):
            EventHistoryConfig(ttl_seconds=-1)
        with self.assertRaises(AssertionError):
            EventHistoryConfig(ttl_seconds=cast(Any, True))
        with self.assertRaises(AssertionError):
            EventHistoryConfig()
        with self.assertRaises(AssertionError):
            EventManager(
                history_length=1,
                history_config=EventHistoryConfig(max_events=1),
            )

    def test_history_config_is_exposed(self) -> None:
        config = EventHistoryConfig(max_events=3)
        manager = EventManager(history_config=config)

        self.assertIs(manager.history_config, config)

    async def test_manager_instances_do_not_share_event_state(self) -> None:
        first = EventManager(
            history_config=EventHistoryConfig(max_events=2),
            listen_config=EventListenConfig(queue_limit=2),
        )
        second = EventManager(
            history_config=EventHistoryConfig(max_events=2),
            listen_config=EventListenConfig(queue_limit=2),
        )
        called: list[EventType] = []

        def listener(event: Event) -> None:
            called.append(event.type)

        first.add_listener(
            listener,
            [EventType.START, EventType.TOKEN_GENERATED],
        )
        first_start = Event(type=EventType.START)
        first_end = Event(type=EventType.END)

        await first.trigger(first_start)
        await first.trigger(first_end)

        self.assertEqual(called, [EventType.START])
        self.assertTrue(first.should_emit(EventType.TOKEN_GENERATED))
        self.assertFalse(second.should_emit(EventType.TOKEN_GENERATED))
        self.assertEqual(first.history, [first_start, first_end])
        self.assertEqual(second.history, [])
        self.assertEqual(first.stats.total_triggers, 2)
        self.assertEqual(first.stats.triggers[EventType.START], 1)
        self.assertEqual(first.stats.triggers[EventType.END], 1)
        self.assertEqual(second.stats.total_triggers, 0)
        self.assertEqual(second.stats.triggers, {})
        self.assertEqual(first._delivery_queue.qsize(), 2)
        self.assertEqual(second._delivery_queue.qsize(), 0)

        second_end = Event(type=EventType.END)
        await second.trigger(second_end)

        self.assertEqual(first.history, [first_start, first_end])
        self.assertEqual(second.history, [second_end])
        self.assertEqual(first.stats.triggers[EventType.START], 1)
        self.assertEqual(second.stats.triggers[EventType.END], 1)
        self.assertNotIn(EventType.START, second.stats.triggers)

    async def test_listener_less_bursts_keep_history_and_queue_bounded(
        self,
    ) -> None:
        manager = EventManager(
            history_config=EventHistoryConfig(max_events=3),
            listen_config=EventListenConfig(
                queue_limit=2,
                policy=EventDeliveryPolicy.DROP,
            ),
        )

        def event_for_index(index: int) -> Event:
            return Event.from_observability_payload(
                type=EventType.START,
                observability_payload=(
                    EventObservabilityPayload.temporary_legacy(
                        {
                            "event_type": EventType.START.value,
                            "index": index,
                        },
                        owner="event-manager-test",
                        removal_condition="Test fixture only.",
                    )
                ),
            )

        for index in range(25):
            await manager.trigger(event_for_index(index))

        self.assertEqual(len(manager.history), 3)
        self.assertEqual(
            [
                cast(dict[str, object], event.payload)["index"]
                for event in manager.history
            ],
            [22, 23, 24],
        )
        self.assertEqual(manager._delivery_queue.qsize(), 2)
        self.assertEqual(manager.stats.published, 25)
        self.assertEqual(manager.stats.dropped, 23)
        self.assertEqual(manager.stats.queue_depth, 2)
        self.assertEqual(manager.stats.max_queue_depth, 2)

    def test_sdk_mode_uses_bounded_lossless_defaults(self) -> None:
        manager = EventManager()

        self.assertEqual(manager.history_config.max_events, 512)
        self.assertTrue(manager.listen_config.enabled)
        self.assertEqual(manager.listen_config.queue_limit, 512)
        self.assertIs(
            manager.default_delivery_config.policy,
            EventDeliveryPolicy.BLOCK,
        )
        self.assertEqual(manager.default_delivery_config.queue_limit, 1)
        self.assertTrue(manager.collect_stats)
        self.assertFalse(manager.enrich_token_ids)

    def test_server_mode_disables_history_and_drops_observability(
        self,
    ) -> None:
        manager = EventManager(mode=EventManagerMode.SERVER)

        self.assertFalse(manager.history_config.enabled)
        self.assertIs(
            manager.default_delivery_config.policy,
            EventDeliveryPolicy.DROP,
        )
        self.assertEqual(manager.default_delivery_config.queue_limit, 32)
        self.assertFalse(manager.listen_config.enabled)
        self.assertFalse(manager.collect_stats)

    async def test_server_mode_stats_are_explicit_opt_in(self) -> None:
        default_manager = EventManager(mode=EventManagerMode.SERVER)
        stats_manager = EventManager(
            mode=EventManagerMode.SERVER,
            collect_stats=True,
        )

        await default_manager.trigger(Event(type=EventType.START))
        await stats_manager.trigger(Event(type=EventType.START))

        self.assertEqual(default_manager.stats.published, 0)
        self.assertEqual(default_manager.stats.total_triggers, 0)
        self.assertEqual(stats_manager.stats.published, 1)
        self.assertEqual(stats_manager.stats.triggers[EventType.START], 1)

    async def test_disabled_stats_still_delivers_to_subscribers(self) -> None:
        manager = EventManager(
            mode=EventManagerMode.SERVER,
            default_delivery_config=EventDeliveryConfig(),
        )
        called: list[Event] = []

        async def listener(event: Event) -> None:
            called.append(event)

        manager.add_listener(listener, [EventType.START])

        event = Event(type=EventType.START)
        await manager.trigger(event)

        self.assertEqual(called, [event])
        self.assertEqual(manager.stats.published, 0)
        self.assertEqual(manager.stats.delivered, 0)

    async def test_disabled_stats_do_not_track_dropped_listen_events(
        self,
    ) -> None:
        manager = EventManager(
            collect_stats=False,
            listen_config=EventListenConfig(queue_limit=1),
        )

        await manager.trigger(Event(type=EventType.START))
        await manager.trigger(Event(type=EventType.END))

        self.assertEqual(manager._delivery_queue.qsize(), 1)
        self.assertEqual(manager.stats.dropped, 0)
        self.assertEqual(manager.stats.queue_depth, 0)

    def test_cli_mode_keeps_bounded_history_and_coalesces_ui_events(
        self,
    ) -> None:
        manager = EventManager(mode=EventManagerMode.CLI)

        self.assertEqual(manager.history_config.max_events, 256)
        self.assertIs(
            manager.default_delivery_config.policy,
            EventDeliveryPolicy.COALESCE,
        )
        self.assertEqual(manager.default_delivery_config.queue_limit, 64)
        self.assertEqual(manager.listen_config.queue_limit, 256)
        self.assertIs(
            manager.listen_config.policy,
            EventDeliveryPolicy.COALESCE,
        )

    def test_test_mode_keeps_deterministic_bounded_history(self) -> None:
        manager = EventManager(mode=EventManagerMode.TEST)

        self.assertEqual(manager.history_config.max_events, 1024)
        self.assertIs(
            manager.default_delivery_config.policy,
            EventDeliveryPolicy.BLOCK,
        )
        self.assertEqual(manager.listen_config.queue_limit, 1024)

    def test_mode_defaults_can_be_inspected_without_manager(self) -> None:
        defaults = EventManager.defaults_for_mode(EventManagerMode.SERVER)

        self.assertIsInstance(defaults, EventManagerDefaults)
        self.assertFalse(defaults.history_config.enabled)
        self.assertIs(
            defaults.delivery_config.policy,
            EventDeliveryPolicy.DROP,
        )
        self.assertFalse(defaults.listen_config.enabled)

    def test_subscriber_class_defaults_are_bounded(self) -> None:
        lossless = EventManager.default_delivery_config_for_subscriber_class(
            EventSubscriberClass.LOSSLESS
        )
        ui = EventManager.default_delivery_config_for_subscriber_class(
            EventSubscriberClass.UI
        )
        critical = EventManager.default_delivery_config_for_subscriber_class(
            EventSubscriberClass.CRITICAL
        )
        observability = (
            EventManager.default_delivery_config_for_subscriber_class(
                EventSubscriberClass.OBSERVABILITY
            )
        )

        self.assertIs(lossless.policy, EventDeliveryPolicy.BLOCK)
        self.assertEqual(lossless.queue_limit, 1)
        self.assertIs(ui.policy, EventDeliveryPolicy.COALESCE)
        self.assertEqual(ui.queue_limit, 64)
        self.assertTrue(critical.critical)
        self.assertEqual(critical.timeout, 1.0)
        self.assertIs(observability.policy, EventDeliveryPolicy.DROP)
        self.assertEqual(observability.queue_limit, 32)

    def test_explicit_configs_override_mode_defaults(self) -> None:
        history_config = EventHistoryConfig(max_events=2)
        delivery_config = EventDeliveryConfig(
            policy=EventDeliveryPolicy.FAIL_CLOSED,
            queue_limit=3,
        )
        listen_config = EventListenConfig(queue_limit=5)
        manager = EventManager(
            mode=EventManagerMode.SERVER,
            history_config=history_config,
            default_delivery_config=delivery_config,
            listen_config=listen_config,
            enrich_token_ids=True,
        )

        self.assertIs(manager.history_config, history_config)
        self.assertIs(manager.default_delivery_config, delivery_config)
        self.assertIs(manager.listen_config, listen_config)
        self.assertTrue(manager.enrich_token_ids)

    def test_history_length_overrides_mode_history_default(self) -> None:
        manager = EventManager(
            history_length=2,
            mode=EventManagerMode.SERVER,
        )

        self.assertTrue(manager.history_config.enabled)
        self.assertEqual(manager.history_config.max_events, 2)

    def test_manager_defaults_reject_invalid_settings(self) -> None:
        with self.assertRaises(AssertionError):
            EventManager(mode=cast(Any, "server"))
        with self.assertRaises(AssertionError):
            EventManager.defaults_for_mode(cast(Any, "sdk"))
        with self.assertRaises(AssertionError):
            EventManager.default_delivery_config_for_subscriber_class(
                cast(Any, "ui")
            )
        with self.assertRaises(AssertionError):
            EventManager(history_config=cast(Any, object()))
        with self.assertRaises(AssertionError):
            EventManager(default_delivery_config=cast(Any, object()))
        with self.assertRaises(AssertionError):
            EventManager(listen_config=cast(Any, object()))
        with self.assertRaises(AssertionError):
            EventManagerDefaults(
                history_config=cast(Any, object()),
                delivery_config=EventDeliveryConfig(),
                listen_config=EventListenConfig(),
            )
        with self.assertRaises(AssertionError):
            EventManagerDefaults(
                history_config=EventHistoryConfig(max_events=1),
                delivery_config=cast(Any, object()),
                listen_config=EventListenConfig(),
            )
        with self.assertRaises(AssertionError):
            EventManagerDefaults(
                history_config=EventHistoryConfig(max_events=1),
                delivery_config=EventDeliveryConfig(),
                listen_config=cast(Any, object()),
            )
        with self.assertRaises(AssertionError):
            EventManagerDefaults(
                history_config=EventHistoryConfig(max_events=1),
                delivery_config=EventDeliveryConfig(),
                listen_config=EventListenConfig(),
                collect_stats=cast(Any, "yes"),
            )
        with self.assertRaises(AssertionError):
            EventManagerDefaults(
                history_config=EventHistoryConfig(max_events=1),
                delivery_config=EventDeliveryConfig(),
                listen_config=EventListenConfig(),
                enrich_token_ids=cast(Any, "yes"),
            )
        with self.assertRaises(AssertionError):
            EventManager(collect_stats=cast(Any, "yes"))

    def test_listen_config_rejects_invalid_settings(self) -> None:
        with self.assertRaises(AssertionError):
            EventListenConfig(enabled=cast(Any, "yes"))
        with self.assertRaises(AssertionError):
            EventListenConfig(queue_limit=0)
        with self.assertRaises(AssertionError):
            EventListenConfig(queue_limit=cast(Any, True))
        with self.assertRaises(AssertionError):
            EventListenConfig(policy=cast(Any, "drop"))
        with self.assertRaises(AssertionError):
            EventListenConfig(policy=EventDeliveryPolicy.BLOCK)

    def test_event_stats_instances_are_isolated(self) -> None:
        first = EventStats()
        second = EventStats()

        first.record_published(EventType.START, queue_depth=2)
        first.record_delivered(queue_depth=1)
        first.record_dropped(2, queue_depth=0)
        first.record_coalesced(3)
        first.record_failed(4)
        first.record_listener_lag(0.5)
        first.record_listener_lag(0.25)
        first.record_listener_lag(0.75)
        first.record_listener_lag(0.5)
        first.record_critical_wait_time(1)
        first.record_critical_wait_time(0.5)
        first.record_critical_wait_time(1.5)
        first.record_critical_wait_time(1)

        first_snapshot = first.snapshot()
        second_snapshot = second.snapshot()
        self.assertEqual(first_snapshot.total_triggers, 1)
        self.assertEqual(first_snapshot.published, 1)
        self.assertEqual(first_snapshot.delivered, 1)
        self.assertEqual(first_snapshot.dropped, 2)
        self.assertEqual(first_snapshot.coalesced, 3)
        self.assertEqual(first_snapshot.failed, 4)
        self.assertEqual(first_snapshot.queue_depth, 0)
        self.assertEqual(first_snapshot.max_queue_depth, 2)
        self.assertEqual(first_snapshot.listener_lag, 0.75)
        self.assertEqual(first_snapshot.critical_wait_time, 1.5)
        self.assertEqual(first_snapshot.triggers[EventType.START], 1)
        self.assertEqual(second_snapshot.total_triggers, 0)
        self.assertEqual(second_snapshot.triggers, {})
        self.assertEqual(second_snapshot.max_queue_depth, 0)

    def test_event_stats_rejects_bool_numeric_values(self) -> None:
        stats = EventStats()

        with self.assertRaises(AssertionError):
            stats.record_published(
                EventType.START,
                queue_depth=cast(Any, True),
            )
        with self.assertRaises(AssertionError):
            stats.record_delivered(queue_depth=cast(Any, True))
        with self.assertRaises(AssertionError):
            stats.record_dropped(cast(Any, True))
        with self.assertRaises(AssertionError):
            stats.record_dropped(queue_depth=cast(Any, False))
        with self.assertRaises(AssertionError):
            stats.record_queue_depth(cast(Any, True))
        with self.assertRaises(AssertionError):
            stats.record_coalesced(cast(Any, True))
        with self.assertRaises(AssertionError):
            stats.record_failed(cast(Any, True))
        with self.assertRaises(AssertionError):
            stats.record_listener_lag(cast(Any, False))
        with self.assertRaises(AssertionError):
            stats.record_critical_wait_time(cast(Any, True))

    def test_event_stats_snapshot_triggers_are_immutable(self) -> None:
        stats = EventStats()
        stats.record_trigger(EventType.START)
        snapshot = stats.snapshot()

        with self.assertRaises(TypeError):
            cast(Any, snapshot.triggers)[EventType.END] = 1

    async def test_listen(self) -> None:
        manager = EventManager()
        evt = Event(type=EventType.START)
        await manager.trigger(evt)
        gen = manager.listen(stop_signal=None)
        self.assertIs(await gen.__anext__(), evt)

        async def get_next() -> Event:
            return await gen.__anext__()

        task = asyncio.create_task(get_next())
        await asyncio.sleep(0)
        evt2 = Event(type=EventType.END)
        await manager.trigger(evt2)
        self.assertIs(await task, evt2)

    async def test_listen_stop_signal(self) -> None:
        manager = EventManager()
        stop = asyncio.Event()
        events: list[Event] = []

        async def iterate() -> None:
            async for event in manager.listen(stop_signal=stop, timeout=0.01):
                events.append(event)

        task = asyncio.create_task(iterate())
        await asyncio.sleep(0.02)
        self.assertFalse(task.done())
        stop.set()
        await task
        self.assertEqual(events, [])

    async def test_add_listener_reuses_matching_delivery_config(
        self,
    ) -> None:
        manager = EventManager()
        count = 0

        def listener(event: Event) -> None:
            nonlocal count
            count += 1

        manager.add_listener(listener, [EventType.START])
        manager.add_listener(listener, [EventType.START])
        manager.add_listener(
            listener,
            [EventType.END],
            delivery_config=EventDeliveryConfig(),
        )
        await manager.trigger(Event(type=EventType.START))
        await manager.trigger(Event(type=EventType.END))
        self.assertEqual(count, 2)

    async def test_add_listener_rejects_delivery_config_change(
        self,
    ) -> None:
        manager = EventManager()
        called: list[EventType] = []

        def listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(listener, [EventType.START])

        with self.assertRaises(AssertionError):
            manager.add_ui_listener(listener, [EventType.END])
        with self.assertRaises(AssertionError):
            manager.add_listener(
                listener,
                [EventType.END],
                delivery_config=EventDeliveryConfig(
                    policy=EventDeliveryPolicy.DROP
                ),
            )

        await manager.trigger(Event(type=EventType.END))
        await manager.trigger(Event(type=EventType.START))
        self.assertEqual(called, [EventType.START])

    async def test_remove_listener(self) -> None:
        manager = EventManager()
        called: list[EventType] = []

        def listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(listener)
        await manager.trigger(Event(type=EventType.START))
        manager.remove_listener(listener)
        await manager.trigger(Event(type=EventType.END))
        self.assertEqual(called, [EventType.START])

    async def test_remove_listener_during_trigger_skips_closed_listener(
        self,
    ) -> None:
        manager = EventManager()
        called: list[str] = []

        def second_listener(event: Event) -> None:
            _ = event
            called.append("second")

        def first_listener(event: Event) -> None:
            _ = event
            called.append("first")
            manager.remove_listener(second_listener, [EventType.START])

        manager.add_listener(first_listener, [EventType.START])
        manager.add_listener(second_listener, [EventType.START])

        await manager.trigger(Event(type=EventType.START))

        self.assertEqual(called, ["first"])

    def test_remove_unknown_listener_is_noop(self) -> None:
        manager = EventManager()

        def listener(event: Event) -> None:
            _ = event

        manager.remove_listener(listener, [EventType.START])

        self.assertTrue(manager.should_emit(EventType.START))

    async def test_empty_listener_event_types_do_not_subscribe(
        self,
    ) -> None:
        manager = EventManager()
        called: list[EventType] = []

        def listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(listener, [])
        manager.add_ui_listener(listener, [])
        manager.add_observability_listener(listener, [])

        await manager.trigger(Event(type=EventType.START))

        self.assertNotIn(listener, manager._subscriber_index)
        self.assertEqual(called, [])
        self.assertEqual(manager.stats.delivered, 0)

    async def test_remove_listener_empty_event_types_is_noop(self) -> None:
        manager = EventManager()
        called: list[EventType] = []

        def listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(listener, [EventType.START])
        manager.remove_listener(listener, [])

        await manager.trigger(Event(type=EventType.START))

        self.assertIn(listener, manager._subscriber_index)
        self.assertEqual(called, [EventType.START])
        self.assertEqual(manager.stats.delivered, 1)

    def test_add_listener_rejects_invalid_event_inputs(self) -> None:
        manager = EventManager()

        def listener(event: Event) -> None:
            _ = event

        with self.assertRaises(AssertionError):
            manager.add_listener(cast(Any, object()), [EventType.START])
        with self.assertRaises(AssertionError):
            manager.add_listener(
                listener,
                include_token_events=cast(Any, "yes"),
            )
        with self.assertRaises(AssertionError):
            manager.add_listener(
                listener,
                [EventType.START],
                delivery_config=cast(Any, object()),
            )
        with self.assertRaises(AssertionError):
            manager.add_listener(listener, [cast(Any, "start")])
        with self.assertRaises(AssertionError):
            manager.add_listener(
                listener,
                subscriber_class=cast(Any, "ui"),
            )
        with self.assertRaises(AssertionError):
            manager.add_listener(
                listener,
                delivery_config=EventDeliveryConfig(),
                subscriber_class=EventSubscriberClass.UI,
            )
        with self.assertRaises(AssertionError):
            manager.remove_listener(listener, [cast(Any, "start")])

    async def test_drop_policy_drops_when_subscriber_queue_is_full(
        self,
    ) -> None:
        manager = EventManager()
        called: list[Event] = []

        async def listener(event: Event) -> None:
            called.append(event)

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=1,
            ),
        )
        first = Event(type=EventType.START, payload={"n": 1})
        second = Event(type=EventType.START, payload={"n": 2})

        await manager.trigger(first)
        await manager.trigger(second)
        await asyncio.sleep(0)

        self.assertEqual(called, [first])
        self.assertEqual(manager.stats.delivered, 1)
        self.assertEqual(manager.stats.dropped, 1)

    async def test_coalesce_policy_keeps_latest_event(self) -> None:
        manager = EventManager()
        called: list[Event] = []

        async def listener(event: Event) -> None:
            called.append(event)

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.COALESCE,
                queue_limit=1,
            ),
        )
        first = Event(type=EventType.START, payload={"n": 1})
        second = Event(type=EventType.START, payload={"n": 2})
        third = Event(type=EventType.START, payload={"n": 3})

        await manager.trigger(first)
        await manager.trigger(second)
        await manager.trigger(third)
        await asyncio.sleep(0)

        self.assertEqual(called, [third])
        self.assertEqual(manager.stats.delivered, 1)
        self.assertEqual(manager.stats.coalesced, 2)

    async def test_fail_closed_policy_removes_overflowing_subscriber(
        self,
    ) -> None:
        manager = EventManager()
        called: list[Event] = []

        async def listener(event: Event) -> None:
            called.append(event)

        manager.add_listener(
            listener,
            [EventType.TOKEN_GENERATED],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.FAIL_CLOSED,
                queue_limit=1,
            ),
        )

        self.assertTrue(manager.should_emit(EventType.TOKEN_GENERATED))
        await manager.trigger(Event(type=EventType.TOKEN_GENERATED))
        await manager.trigger(Event(type=EventType.TOKEN_GENERATED))
        await manager.trigger(Event(type=EventType.TOKEN_GENERATED))
        await asyncio.sleep(0)

        self.assertEqual(called, [])
        self.assertEqual(manager.stats.delivered, 0)
        self.assertEqual(manager.stats.failed, 1)
        self.assertFalse(manager.should_emit(EventType.TOKEN_GENERATED))

    async def test_stream_item_subscriber_overflow_fails_closed(self) -> None:
        manager = EventManager(listen_config=EventListenConfig(enabled=False))
        called: list[Event] = []

        async def listener(event: Event) -> None:
            called.append(event)

        manager.add_listener(
            listener,
            [EventType.TOKEN_GENERATED],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.FAIL_CLOSED,
                queue_limit=1,
            ),
            include_token_events=True,
        )

        for sequence in range(3):
            await manager.trigger_stream_item(
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=sequence,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta=str(sequence),
                )
            )
        await asyncio.sleep(0)

        self.assertEqual(called, [])
        self.assertEqual(manager.stats.delivered, 0)
        self.assertEqual(manager.stats.failed, 1)
        self.assertFalse(manager.should_emit(EventType.TOKEN_GENERATED))

    async def test_coalescing_ui_stream_subscriber_keeps_lossless_items(
        self,
    ) -> None:
        manager = EventManager(listen_config=EventListenConfig(enabled=False))
        ui_started = asyncio.Event()
        release_ui = asyncio.Event()
        ui_sequences: list[int] = []
        lossless_sequences: list[int] = []
        total_items = 72

        def event_sequence(event: Event) -> int:
            payload = event.payload
            assert isinstance(payload, dict)
            sequence = payload["sequence"]
            assert isinstance(sequence, int)
            return sequence

        async def ui_listener(event: Event) -> None:
            ui_sequences.append(event_sequence(event))
            ui_started.set()
            await release_ui.wait()

        async def lossless_listener(event: Event) -> None:
            lossless_sequences.append(event_sequence(event))

        manager.add_ui_listener(
            ui_listener,
            [EventType.TOKEN_GENERATED],
            include_token_events=True,
        )
        manager.add_listener(
            lossless_listener,
            [EventType.TOKEN_GENERATED],
            include_token_events=True,
            subscriber_class=EventSubscriberClass.LOSSLESS,
        )

        for sequence in range(total_items):
            await manager.trigger_stream_item(
                CanonicalStreamItem(
                    stream_session_id="stream",
                    run_id="run",
                    turn_id="turn",
                    sequence=sequence,
                    kind=StreamItemKind.ANSWER_DELTA,
                    channel=StreamChannel.ANSWER,
                    text_delta=str(sequence),
                )
            )
            if sequence == 0:
                await asyncio.wait_for(ui_started.wait(), timeout=0.1)

        self.assertEqual(lossless_sequences, list(range(total_items)))
        self.assertGreater(manager.stats.coalesced, 0)

        ui_subscriber = manager._subscriber_index[ui_listener]
        assert ui_subscriber.task is not None
        release_ui.set()
        await asyncio.wait_for(ui_subscriber.task, timeout=0.1)

        self.assertEqual(ui_sequences[0], 0)
        self.assertEqual(ui_sequences[-1], total_items - 1)
        self.assertLess(len(ui_sequences), total_items)
        self.assertEqual(lossless_sequences, list(range(total_items)))
        self.assertEqual(manager.stats.dropped, 0)

    async def test_non_critical_token_listener_does_not_block_generation(
        self,
    ) -> None:
        manager = EventManager()
        started = asyncio.Event()
        release = asyncio.Event()
        called: list[EventType] = []

        async def listener(event: Event) -> None:
            started.set()
            await release.wait()
            called.append(event.type)

        manager.add_listener(
            listener,
            [EventType.TOKEN_GENERATED],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=1,
            ),
        )

        await manager.trigger(Event(type=EventType.TOKEN_GENERATED))
        await asyncio.wait_for(started.wait(), timeout=0.1)
        self.assertEqual(called, [])
        release.set()
        await asyncio.sleep(0)
        self.assertEqual(called, [EventType.TOKEN_GENERATED])

    async def test_lossy_policy_continues_after_listener_cancellation(
        self,
    ) -> None:
        for policy in (
            EventDeliveryPolicy.DROP,
            EventDeliveryPolicy.COALESCE,
        ):
            with self.subTest(policy=policy):
                manager = EventManager()
                called: list[EventType] = []
                attempts = 0

                async def listener(event: Event) -> None:
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise asyncio.CancelledError()
                    called.append(event.type)

                manager.add_listener(
                    listener,
                    [EventType.START],
                    delivery_config=EventDeliveryConfig(
                        policy=policy,
                        queue_limit=2,
                    ),
                )

                await manager.trigger(Event(type=EventType.START))
                await manager.trigger(Event(type=EventType.START))
                await asyncio.sleep(0)

                self.assertEqual(attempts, 2)
                self.assertEqual(called, [EventType.START])
                self.assertEqual(manager.stats.delivered, 1)
                self.assertEqual(manager.stats.failed, 1)

    async def test_lossy_policy_refreshes_queue_depth_after_failure(
        self,
    ) -> None:
        for policy in (
            EventDeliveryPolicy.DROP,
            EventDeliveryPolicy.COALESCE,
        ):
            with self.subTest(policy=policy):
                manager = EventManager(
                    listen_config=EventListenConfig(enabled=False)
                )

                async def listener(event: Event) -> None:
                    _ = event
                    raise RuntimeError("failed")

                manager.add_listener(
                    listener,
                    [EventType.START],
                    delivery_config=EventDeliveryConfig(
                        policy=policy,
                        queue_limit=1,
                    ),
                )

                await manager.trigger(Event(type=EventType.START))
                subscriber = manager._subscriber_index[listener]
                assert subscriber.task is not None
                await asyncio.wait_for(subscriber.task, timeout=0.1)

                self.assertTrue(subscriber.queue.empty())
                self.assertEqual(manager.stats.failed, 1)
                self.assertEqual(manager.stats.delivered, 0)
                self.assertEqual(manager.stats.queue_depth, 0)
                self.assertEqual(manager.stats.max_queue_depth, 1)

    async def test_fail_closed_policy_records_listener_exception(self) -> None:
        manager = EventManager()

        async def listener(event: Event) -> None:
            _ = event
            raise RuntimeError("failed")

        manager.add_listener(
            listener,
            [EventType.TOKEN_GENERATED],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.FAIL_CLOSED,
                queue_limit=1,
            ),
        )

        await manager.trigger(Event(type=EventType.TOKEN_GENERATED))
        await asyncio.sleep(0)

        self.assertEqual(manager.stats.failed, 1)
        self.assertEqual(manager.stats.delivered, 0)
        self.assertFalse(manager.should_emit(EventType.TOKEN_GENERATED))

    async def test_fail_closed_policy_records_listener_cancellation(
        self,
    ) -> None:
        manager = EventManager()

        async def listener(event: Event) -> None:
            _ = event
            raise asyncio.CancelledError()

        manager.add_listener(
            listener,
            [EventType.TOKEN_GENERATED],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.FAIL_CLOSED,
                queue_limit=1,
            ),
        )

        await manager.trigger(Event(type=EventType.TOKEN_GENERATED))
        await asyncio.sleep(0)

        self.assertEqual(manager.stats.failed, 1)
        self.assertEqual(manager.stats.delivered, 0)
        self.assertFalse(manager.should_emit(EventType.TOKEN_GENERATED))

    async def test_direct_token_trigger_requires_subscriber(self) -> None:
        manager = EventManager()

        await manager.trigger(
            Event(type=EventType.TOKEN_GENERATED, payload={"token": "a"})
        )

        self.assertEqual(manager.history, [])
        self.assertEqual(manager.stats.published, 0)

    async def test_removing_listener_cancels_pending_delivery_task(
        self,
    ) -> None:
        manager = EventManager()
        started = asyncio.Event()
        release = asyncio.Event()

        async def listener(event: Event) -> None:
            _ = event
            started.set()
            await release.wait()

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=1,
            ),
        )
        await manager.trigger(Event(type=EventType.START))
        await asyncio.wait_for(started.wait(), timeout=0.1)
        subscriber = manager._subscriber_index[listener]
        assert subscriber.task is not None

        manager.remove_listener(listener, [EventType.START])
        await asyncio.sleep(0)

        self.assertTrue(subscriber.task.cancelled())
        self.assertEqual(manager.stats.failed, 0)
        release.set()

    async def test_queue_stats_keep_high_water_after_delivery_drains(
        self,
    ) -> None:
        manager = EventManager(listen_config=EventListenConfig(enabled=False))
        started = asyncio.Event()
        release = asyncio.Event()
        seen: list[int] = []

        async def listener(event: Event) -> None:
            payload = cast(dict[str, int], event.payload)
            seen.append(payload["n"])
            started.set()
            await release.wait()

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=2,
            ),
        )
        await manager.trigger(Event(type=EventType.START, payload={"n": 1}))
        await asyncio.wait_for(started.wait(), timeout=0.1)
        await manager.trigger(Event(type=EventType.START, payload={"n": 2}))
        subscriber = manager._subscriber_index[listener]
        assert subscriber.task is not None

        self.assertEqual(manager.stats.queue_depth, 1)
        self.assertEqual(manager.stats.max_queue_depth, 1)

        release.set()
        await asyncio.wait_for(subscriber.task, timeout=0.1)

        self.assertEqual(seen, [1, 2])
        self.assertEqual(manager.stats.queue_depth, 0)
        self.assertEqual(manager.stats.max_queue_depth, 1)

    async def test_queue_stats_keep_listen_depth_during_delivery(
        self,
    ) -> None:
        manager = EventManager(listen_config=EventListenConfig(queue_limit=4))
        started = asyncio.Event()
        release = asyncio.Event()

        async def listener(event: Event) -> None:
            _ = event
            started.set()
            await release.wait()

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=2,
            ),
        )
        await manager.trigger(Event(type=EventType.START, payload={"n": 1}))
        await asyncio.wait_for(started.wait(), timeout=0.1)
        await manager.trigger(Event(type=EventType.START, payload={"n": 2}))
        subscriber = manager._subscriber_index[listener]
        assert subscriber.task is not None

        self.assertEqual(manager.stats.queue_depth, 2)
        self.assertEqual(manager.stats.max_queue_depth, 2)

        release.set()
        await asyncio.wait_for(subscriber.task, timeout=0.1)

        self.assertEqual(manager.stats.queue_depth, 2)
        gen = manager.listen(stop_signal=None, timeout=0.01)
        self.assertEqual((await gen.__anext__()).payload, {"n": 1})
        self.assertEqual(manager.stats.queue_depth, 1)
        self.assertEqual((await gen.__anext__()).payload, {"n": 2})
        self.assertEqual(manager.stats.queue_depth, 0)
        with self.assertRaises(StopAsyncIteration):
            await gen.__anext__()
        self.assertEqual(manager.stats.max_queue_depth, 2)

    async def test_remove_listener_drains_pending_queue_depth(self) -> None:
        manager = EventManager(listen_config=EventListenConfig(enabled=False))
        started = asyncio.Event()
        release = asyncio.Event()

        async def listener(event: Event) -> None:
            _ = event
            started.set()
            await release.wait()

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=2,
            ),
        )
        await manager.trigger(Event(type=EventType.START, payload={"n": 1}))
        await asyncio.wait_for(started.wait(), timeout=0.1)
        await manager.trigger(Event(type=EventType.START, payload={"n": 2}))
        subscriber = manager._subscriber_index[listener]
        assert subscriber.task is not None

        self.assertEqual(subscriber.queue.qsize(), 1)
        self.assertEqual(manager.stats.queue_depth, 1)

        manager.remove_listener(listener, [EventType.START])
        await asyncio.sleep(0)

        self.assertTrue(subscriber.queue.empty())
        self.assertEqual(manager.stats.queue_depth, 0)
        self.assertEqual(manager.stats.max_queue_depth, 1)
        self.assertEqual(manager.stats.dropped, 1)
        self.assertTrue(subscriber.task.cancelled())
        release.set()

    async def test_aclose_cancels_subscribers_and_drains_queues(self) -> None:
        manager = EventManager(listen_config=EventListenConfig(queue_limit=4))
        started = asyncio.Event()
        release = asyncio.Event()

        async def listener(event: Event) -> None:
            _ = event
            started.set()
            await release.wait()

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=2,
            ),
        )
        await manager.trigger(Event(type=EventType.START, payload={"n": 1}))
        await asyncio.wait_for(started.wait(), timeout=0.1)
        await manager.trigger(Event(type=EventType.START, payload={"n": 2}))
        subscriber = manager._subscriber_index[listener]
        assert subscriber.task is not None
        self.assertEqual(subscriber.queue.qsize(), 1)
        self.assertEqual(manager._delivery_queue.qsize(), 2)

        await manager.aclose()

        self.assertTrue(subscriber.closed)
        self.assertTrue(subscriber.queue.empty())
        self.assertTrue(manager._delivery_queue.empty())
        self.assertEqual(manager._subscriber_index, {})
        self.assertEqual(manager._subscribers, {})
        self.assertEqual(manager.stats.queue_depth, 0)
        self.assertEqual(manager.stats.max_queue_depth, 2)
        self.assertEqual(manager.stats.dropped, 3)
        self.assertTrue(subscriber.task.cancelled())
        release.set()

    async def test_aclose_rejects_late_listeners_and_publications(
        self,
    ) -> None:
        manager = EventManager()

        await manager.trigger(Event(type=EventType.START))
        await manager.aclose()

        self.assertTrue(manager.closed)
        self.assertFalse(manager.should_emit(EventType.START))
        self.assertEqual(manager.history, [])
        self.assertEqual(manager._history_bytes, 0)
        with self.assertRaises(AssertionError):
            manager.add_listener(lambda event: None, [EventType.START])

        await manager.trigger(Event(type=EventType.END))
        await manager.trigger_stream_item(
            CanonicalStreamItem(
                stream_session_id="session-1",
                run_id="run-1",
                turn_id="turn-1",
                sequence=0,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="hello",
            )
        )
        gen = manager.listen(stop_signal=None, timeout=0.01)
        with self.assertRaises(StopAsyncIteration):
            await gen.__anext__()

        self.assertEqual(manager.history, [])
        self.assertEqual(manager._history_bytes, 0)
        self.assertEqual(manager.stats.published, 1)

    async def test_aclose_from_subscriber_worker_does_not_self_await(
        self,
    ) -> None:
        manager = EventManager()
        closed = asyncio.Event()

        async def listener(event: Event) -> None:
            _ = event
            await manager.aclose()
            closed.set()

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                policy=EventDeliveryPolicy.DROP,
                queue_limit=1,
            ),
        )
        subscriber = manager._subscriber_index[listener]

        await manager.trigger(Event(type=EventType.START))
        await asyncio.wait_for(closed.wait(), timeout=0.1)
        assert subscriber.task is not None
        await asyncio.wait_for(subscriber.task, timeout=0.1)

        self.assertTrue(manager.closed)
        self.assertTrue(subscriber.closed)
        self.assertEqual(manager._subscriber_index, {})
        self.assertEqual(manager._subscribers, {})
        self.assertEqual(manager.stats.failed, 0)
        self.assertEqual(manager.stats.delivered, 1)

    async def test_aclose_is_idempotent(self) -> None:
        manager = EventManager()

        await manager.aclose()
        await manager.aclose()

        self.assertTrue(manager.closed)
        self.assertEqual(manager._subscriber_index, {})
        self.assertEqual(manager._subscribers, {})
        self.assertEqual(manager.stats.dropped, 0)

    async def test_block_policy_records_failed_listener_and_continues(
        self,
    ) -> None:
        manager = EventManager()
        called: list[EventType] = []

        async def listener(event: Event) -> None:
            _ = event
            raise RuntimeError("failed")

        async def later_listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(listener, [EventType.START])
        manager.add_listener(later_listener, [EventType.START])

        await manager.trigger(Event(type=EventType.START))

        self.assertEqual(manager.stats.failed, 1)
        self.assertEqual(manager.stats.delivered, 1)
        self.assertEqual(called, [EventType.START])

    async def test_block_policy_awaits_future_listener(self) -> None:
        manager = EventManager()
        pending = asyncio.get_running_loop().create_future()

        def listener(event: Event) -> asyncio.Future[None]:
            _ = event
            return pending

        manager.add_listener(listener, [EventType.START])

        task = asyncio.create_task(
            manager.trigger(Event(type=EventType.START))
        )
        await asyncio.sleep(0)
        self.assertFalse(task.done())

        pending.set_result(None)
        await asyncio.wait_for(task, timeout=0.1)

        self.assertEqual(manager.stats.failed, 0)
        self.assertEqual(manager.stats.delivered, 1)

    async def test_block_policy_records_failed_future_and_continues(
        self,
    ) -> None:
        manager = EventManager()
        failed = asyncio.get_running_loop().create_future()
        failed.set_exception(RuntimeError("failed"))
        called: list[EventType] = []

        def listener(event: Event) -> asyncio.Future[None]:
            _ = event
            return failed

        async def later_listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(listener, [EventType.START])
        manager.add_listener(later_listener, [EventType.START])

        await manager.trigger(Event(type=EventType.START))

        self.assertEqual(manager.stats.failed, 1)
        self.assertEqual(manager.stats.delivered, 1)
        self.assertEqual(called, [EventType.START])

    async def test_block_policy_records_cancelled_listener_and_continues(
        self,
    ) -> None:
        manager = EventManager()
        called: list[EventType] = []

        async def listener(event: Event) -> None:
            _ = event
            raise asyncio.CancelledError()

        async def later_listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(listener, [EventType.START])
        manager.add_listener(later_listener, [EventType.START])

        await manager.trigger(Event(type=EventType.START))

        self.assertEqual(manager.stats.failed, 1)
        self.assertEqual(manager.stats.delivered, 1)
        self.assertEqual(called, [EventType.START])

    async def test_block_policy_propagates_trigger_task_cancellation(
        self,
    ) -> None:
        manager = EventManager()
        started = asyncio.Event()
        called: list[EventType] = []

        async def listener(event: Event) -> None:
            _ = event
            started.set()
            await asyncio.sleep(1)

        async def later_listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(listener, [EventType.START])
        manager.add_listener(later_listener, [EventType.START])

        task = asyncio.create_task(
            manager.trigger(Event(type=EventType.START))
        )
        await asyncio.wait_for(started.wait(), timeout=0.1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(manager.stats.failed, 0)
        self.assertEqual(manager.stats.delivered, 0)
        self.assertEqual(called, [])

    async def test_critical_block_policy_records_wait_time(self) -> None:
        manager = EventManager()

        async def listener(event: Event) -> None:
            _ = event
            await asyncio.sleep(0)

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                critical=True,
                timeout=0.1,
            ),
        )

        await manager.trigger(Event(type=EventType.START))

        self.assertGreaterEqual(manager.stats.critical_wait_time, 0)
        self.assertEqual(manager.stats.delivered, 1)

    async def test_critical_block_policy_records_timeout_failure(
        self,
    ) -> None:
        manager = EventManager()
        called: list[EventType] = []

        async def listener(event: Event) -> None:
            _ = event
            await asyncio.sleep(1)

        async def later_listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                critical=True,
                timeout=0,
            ),
        )
        manager.add_listener(later_listener, [EventType.START])

        with self.assertRaises(asyncio.TimeoutError):
            await manager.trigger(Event(type=EventType.START))

        self.assertEqual(manager.stats.failed, 1)
        self.assertEqual(manager.stats.delivered, 1)
        self.assertGreaterEqual(manager.stats.critical_wait_time, 0)
        self.assertEqual(called, [EventType.START])

    async def test_critical_block_policy_cancelled_listener_raises_late(
        self,
    ) -> None:
        manager = EventManager()
        called: list[EventType] = []

        async def listener(event: Event) -> None:
            _ = event
            raise asyncio.CancelledError()

        async def later_listener(event: Event) -> None:
            called.append(event.type)

        manager.add_listener(
            listener,
            [EventType.START],
            delivery_config=EventDeliveryConfig(
                critical=True,
                timeout=0.1,
            ),
        )
        manager.add_listener(later_listener, [EventType.START])

        with self.assertRaises(asyncio.CancelledError):
            await manager.trigger(Event(type=EventType.START))

        self.assertEqual(manager.stats.failed, 1)
        self.assertEqual(manager.stats.delivered, 1)
        self.assertEqual(called, [EventType.START])

    def test_delivery_config_rejects_invalid_settings(self) -> None:
        with self.assertRaises(AssertionError):
            EventDeliveryConfig(policy=cast(Any, "drop"))
        with self.assertRaises(AssertionError):
            EventDeliveryConfig(queue_limit=0)
        with self.assertRaises(AssertionError):
            EventDeliveryConfig(queue_limit=cast(Any, True))
        with self.assertRaises(AssertionError):
            EventDeliveryConfig(timeout=-1)
        with self.assertRaises(AssertionError):
            EventDeliveryConfig(timeout=cast(Any, False))
        with self.assertRaises(AssertionError):
            EventDeliveryConfig(critical=cast(Any, "yes"))
        with self.assertRaises(AssertionError):
            EventDeliveryConfig(critical=True)
        for policy in (
            EventDeliveryPolicy.DROP,
            EventDeliveryPolicy.COALESCE,
            EventDeliveryPolicy.FAIL_CLOSED,
        ):
            with self.subTest(policy=policy):
                with self.assertRaises(AssertionError):
                    EventDeliveryConfig(policy=policy, critical=True)

    def test_token_events_require_subscriber(self) -> None:
        manager = EventManager()

        self.assertFalse(manager.should_emit(EventType.TOKEN_GENERATED))
        self.assertTrue(manager.should_emit(EventType.START))

        def listener(event: Event) -> None:
            _ = event

        manager.add_listener(listener)
        self.assertFalse(manager.should_emit(EventType.TOKEN_GENERATED))
        manager.remove_listener(listener)

        manager.add_listener(listener, [EventType.TOKEN_GENERATED])
        self.assertTrue(manager.should_emit(EventType.TOKEN_GENERATED))
        manager.remove_listener(listener, [EventType.TOKEN_GENERATED])
        self.assertFalse(manager.should_emit(EventType.TOKEN_GENERATED))

    def test_token_events_can_be_included_for_broad_listener(self) -> None:
        manager = EventManager()

        def listener(event: Event) -> None:
            _ = event

        manager.add_listener(listener, include_token_events=True)

        self.assertTrue(manager.should_emit(EventType.TOKEN_GENERATED))

    def test_token_id_enrichment_is_explicit(self) -> None:
        self.assertFalse(EventManager().enrich_token_ids)
        self.assertTrue(EventManager(enrich_token_ids=True).enrich_token_ids)

    async def test_listen_without_stop_signal(self) -> None:
        manager = EventManager()
        events: list[Event] = []

        async def iterate() -> None:
            async for event in manager.listen(stop_signal=None, timeout=0.01):
                events.append(event)

        task = asyncio.create_task(iterate())
        await asyncio.wait_for(task, timeout=0.1)
        self.assertTrue(task.done())
        self.assertEqual(events, [])


if __name__ == "__main__":
    main()
