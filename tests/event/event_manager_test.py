import asyncio
from avalan.event import Event, EventType
from avalan.event.manager import EventManager
from unittest import IsolatedAsyncioTestCase, main


class EventManagerTestCase(IsolatedAsyncioTestCase):
    async def test_trigger_and_history(self):
        manager = EventManager(history_length=2)
        called: list[tuple[str, EventType]] = []

        async def a_listener(event: Event):
            called.append(("a", event.type))

        def s_listener(event: Event):
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

    async def test_listen(self):
        manager = EventManager()
        evt = Event(type=EventType.START)
        await manager.trigger(evt)
        gen = manager.listen()
        self.assertIs(await gen.__anext__(), evt)

        async def get_next():
            return await gen.__anext__()

        task = asyncio.create_task(get_next())
        await asyncio.sleep(0)
        evt2 = Event(type=EventType.END)
        await manager.trigger(evt2)
        self.assertIs(await task, evt2)


if __name__ == "__main__":
    main()
