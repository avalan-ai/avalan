from datetime import datetime, timezone
from logging import Logger
from typing import Any
from uuid import UUID, uuid4

from elasticsearch import AsyncElasticsearch
import numpy as np

from ....memory.partitioner.text import TextPartition
from ....memory.permanent import (
    Memory,
    MemoryType,
    PermanentMemory,
    PermanentMemoryPartition,
    VectorFunction,
)
from . import ElasticsearchMemory, to_thread  # noqa: F401


class ElasticsearchRawMemory(ElasticsearchMemory, PermanentMemory):
    _index: str
    _client: Any
    _logger: Logger

    def __init__(
        self,
        index: str,
        *,
        client: Any,
        logger: Logger,
    ) -> None:
        ElasticsearchMemory.__init__(
            self, index=index, client=client, logger=logger
        )
        PermanentMemory.__init__(self, sentence_model=None)

    @classmethod
    async def create_instance(
        cls,
        index: str,
        *,
        logger: Logger,
        es_client: Any | None = None,
    ) -> "ElasticsearchRawMemory":
        if es_client is None:
            es_client = AsyncElasticsearch()
        memory = cls(index=index, client=es_client, logger=logger)
        return memory

    async def append_with_partitions(
        self,
        namespace: str,
        participant_id: UUID,
        *,
        memory_type: MemoryType,
        data: str,
        identifier: str,
        partitions: list[TextPartition],
        symbols: dict | None = None,
        model_id: str | None = None,
        title: str | None = None,
        description: str | None = None,
    ) -> None:
        assert (
            namespace and participant_id and data and identifier and partitions
        )
        now_utc = datetime.now(timezone.utc)
        entry, partition_rows = self._build_memory_with_partitions(
            namespace,
            participant_id,
            memory_type,
            data,
            identifier,
            partitions,
            created_at=now_utc,
            symbols=symbols,
            model_id=model_id,
            memory_id=uuid4(),
            title=title,
            description=description,
        )
        await self._index_document(
            index=self._index,
            id=str(entry.id),
            document={
                "id": str(entry.id),
                "model_id": entry.model_id,
                "type": str(entry.type),
                "participant_id": str(entry.participant_id),
                "namespace": entry.namespace,
                "identifier": entry.identifier,
                "data": entry.data,
                "partitions": entry.partitions,
                "symbols": entry.symbols,
                "created_at": entry.created_at.isoformat(),
                "title": entry.title,
                "description": entry.description,
            },
        )
        for row in partition_rows:
            await self._index_vector(
                index=self._index,
                id=f"{row.memory_id}:{row.partition}",
                vector=row.embedding.tolist(),
                metadata={
                    "memory_id": str(row.memory_id),
                    "participant_id": str(row.participant_id),
                    "namespace": namespace,
                    "partition": row.partition,
                    "data": row.data,
                    "embedding": row.embedding.tolist(),
                    "created_at": row.created_at.isoformat(),
                },
            )

    async def search_memories(
        self,
        *,
        search_partitions: list[TextPartition],
        participant_id: UUID,
        namespace: str,
        function: VectorFunction,
        limit: int | None = None,
    ) -> list[PermanentMemoryPartition]:
        assert participant_id and namespace and search_partitions
        query = search_partitions[0].embeddings.tolist()
        response = await self._query_vector(
            index=self._index,
            query_vector=query,
            top_k=limit or 10,
            function=str(function),
            filter={
                "memory_id": "*",
                "participant_id": str(participant_id),
                "namespace": namespace,
            },
        )
        results: list[PermanentMemoryPartition] = []
        for item in response.get("Items", []):
            metadata = item.get("Metadata") or {}
            mem_id = metadata.get("memory_id")
            partition_index = metadata.get("partition")
            data = metadata.get("data")
            participant = metadata.get("participant_id")
            created_at = metadata.get("created_at")
            embedding = metadata.get("embedding") or item.get("Vector")
            if (
                not mem_id
                or partition_index is None
                or data is None
                or participant is None
                or embedding is None
                or created_at is None
            ):
                continue
            results.append(
                PermanentMemoryPartition(
                    participant_id=UUID(participant),
                    memory_id=UUID(mem_id),
                    partition=int(partition_index),
                    data=data,
                    embedding=np.array(embedding, dtype=float),
                    created_at=datetime.fromisoformat(created_at),
                )
            )
        return results

    async def list_memories(
        self,
        *,
        participant_id: UUID,
        namespace: str,
    ) -> list[Memory]:
        assert participant_id and namespace
        response = await self._call_client(
            self._client.search,
            index=self._index,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"participant_id": str(participant_id)}},
                            {"term": {"namespace": namespace}},
                        ]
                    }
                },
                "sort": [{"created_at": {"order": "desc"}}],
                "size": 1000,
            },
        )
        hits = response.get("hits", {}).get("hits", []) if response else []
        memories: list[Memory] = []
        for hit in hits:
            source = hit.get("_source")
            if not source:
                continue
            memories.append(
                Memory(
                    id=UUID(source["id"]),
                    model_id=source.get("model_id"),
                    type=MemoryType(source["type"]),
                    participant_id=UUID(source["participant_id"]),
                    namespace=source["namespace"],
                    identifier=source["identifier"],
                    data=source["data"],
                    partitions=source["partitions"],
                    symbols=source.get("symbols"),
                    created_at=datetime.fromisoformat(source["created_at"]),
                    title=source.get("title"),
                    description=source.get("description"),
                )
            )
        return memories

    async def search(
        self, query: str
    ) -> list[PermanentMemoryPartition] | None:
        raise NotImplementedError()
