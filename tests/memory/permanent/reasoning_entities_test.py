from datetime import datetime
from unittest import TestCase
from uuid import uuid4
from numpy import array

from avalan.memory.permanent import (
    Entity,
    Hyperedge,
    HyperedgeEntity,
    HyperedgeMemory,
)


class ReasoningEntitiesTestCase(TestCase):
    def test_hyperedge(self):
        now = datetime.now()
        hyperedge = Hyperedge(
            id=uuid4(),
            relation="rel",
            surface_text="A rel B",
            embedding=array([0.0, 1.0]),
            symbols={"foo": "bar"},
            created_at=now,
        )
        self.assertEqual(hyperedge.relation, "rel")
        self.assertEqual(hyperedge.symbols["foo"], "bar")
        self.assertEqual(hyperedge.embedding.shape, (2,))
        self.assertEqual(hyperedge.created_at, now)

    def test_hyperedge_memory(self):
        memory_id = uuid4()
        hyperedge_id = uuid4()
        hyperedge_memory = HyperedgeMemory(
            hyperedge_id=hyperedge_id,
            memory_id=memory_id,
            char_start=1,
            char_end=5,
        )
        self.assertEqual(hyperedge_memory.memory_id, memory_id)
        self.assertEqual(hyperedge_memory.char_start, 1)

    def test_entity_and_hyperedge_entity(self):
        now = datetime.now()
        participant_id = uuid4()
        entity = Entity(
            id=uuid4(),
            name="Alice",
            type="person",
            embedding=array([1.0, 0.0]),
            participant_id=participant_id,
            namespace="ns",
            created_at=now,
        )
        self.assertEqual(entity.name, "Alice")
        hyperedge_id = uuid4()
        hyperedge_entity = HyperedgeEntity(
            hyperedge_id=hyperedge_id,
            entity_id=entity.id,
            role_idx=1,
            role_label="subject",
        )
        self.assertEqual(hyperedge_entity.role_idx, 1)
        with self.assertRaises(TypeError):
            HyperedgeEntity(hyperedge_id=uuid4(), entity_id=uuid4())
