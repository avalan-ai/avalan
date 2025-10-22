import asyncio
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import Event, EventType
from avalan.event.manager import EventManager


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
        gen = manager.listen(stop_signal=None)
        self.assertIs(await gen.__anext__(), evt)

        async def get_next():
            return await gen.__anext__()

        task = asyncio.create_task(get_next())
        await asyncio.sleep(0)
        evt2 = Event(type=EventType.END)
        await manager.trigger(evt2)
        self.assertIs(await task, evt2)

    async def test_listen_stop_signal(self):
        manager = EventManager()
        stop = asyncio.Event()
        events: list[Event] = []

        async def iterate():
            async for event in manager.listen(stop_signal=stop, timeout=0.01):
                events.append(event)

        task = asyncio.create_task(iterate())
        await asyncio.sleep(0.02)
        self.assertFalse(task.done())
        stop.set()
        await task
        self.assertEqual(events, [])

    async def test_add_listener_once(self):
        manager = EventManager()
        count = 0

        def listener(event: Event):
            nonlocal count
            count += 1

        manager.add_listener(listener, [EventType.START])
        manager.add_listener(listener, [EventType.START])
        await manager.trigger(Event(type=EventType.START))
        self.assertEqual(count, 1)

    async def test_remove_listener(self):
        manager = EventManager()
        called: list[EventType] = []

        def listener(event: Event):
            called.append(event.type)

        manager.add_listener(listener)
        await manager.trigger(Event(type=EventType.START))
        manager.remove_listener(listener)
        await manager.trigger(Event(type=EventType.END))
        self.assertEqual(called, [EventType.START])

    async def test_listen_without_stop_signal(self):
        manager = EventManager()
        events: list[Event] = []

        async def iterate():
            async for event in manager.listen(stop_signal=None, timeout=0.01):
                events.append(event)

        task = asyncio.create_task(iterate())
        await asyncio.wait_for(task, timeout=0.1)
        self.assertTrue(task.done())
        self.assertEqual(events, [])


if __name__ == "__main__":
    main()
