from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.filters import Partitioner


class DummyPartitioner(Partitioner):
    pass


DummyPartitioner.__abstractmethods__ = set()


class PartitionerPropertyTestCase(TestCase):
    def test_sentence_model_raises(self):
        partitioner = DummyPartitioner()
        with self.assertRaises(NotImplementedError):
            _ = partitioner.sentence_model


class PartitionerCallTestCase(IsolatedAsyncioTestCase):
    async def test_call_raises(self):
        partitioner = DummyPartitioner()
        with self.assertRaises(NotImplementedError):
            await partitioner("text")
