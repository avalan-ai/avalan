from avalan.event import Event, EventType
from avalan.event.manager import EventManager
from unittest import IsolatedAsyncioTestCase, main


class EventManagerTestCase(IsolatedAsyncioTestCase):
    async def test_trigger(self):
        manager = EventManager()
        called: list[tuple[str, EventType]] = []

        async def a_listener(event: Event):
            called.append(("a", event.type))

        def s_listener(event: Event):
            called.append(("s", event.type))

        manager.add_listener(a_listener, [EventType.START])
        manager.add_listener(s_listener)

        await manager.trigger(Event(type=EventType.START))
        await manager.trigger(Event(type=EventType.END))

        self.assertIn(("a", EventType.START), called)
        self.assertIn(("s", EventType.START), called)
        self.assertIn(("s", EventType.END), called)
        self.assertEqual(len([c for c in called if c[1] == EventType.START]), 2)


if __name__ == "__main__":
    main()
