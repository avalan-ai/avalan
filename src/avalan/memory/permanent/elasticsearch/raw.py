from datetime import datetime, timezone
from logging import Logger
from typing import Any
from uuid import UUID, uuid4

from elasticsearch import AsyncElasticsearch

from ....memory.partitioner.text import TextPartition
from ....memory.permanent import (
    Memory,
    MemoryType,
    PermanentMemory,
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
    ) -> list[Memory]:
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
        results: list[Memory] = []
        for item in response.get("Items", []):
            mem_id = item.get("Metadata", {}).get("memory_id")
            if not mem_id:
                continue
            obj = await self._get_document(index=self._index, id=mem_id)
            meta = obj.get("_source") if obj else None
            if not meta:
                continue
            results.append(
                Memory(
                    id=UUID(meta["id"]),
                    model_id=meta["model_id"],
                    type=MemoryType(meta["type"]),
                    participant_id=UUID(meta["participant_id"]),
                    namespace=meta["namespace"],
                    identifier=meta["identifier"],
                    data=meta["data"],
                    partitions=meta["partitions"],
                    symbols=meta["symbols"],
                    created_at=datetime.fromisoformat(meta["created_at"]),
                )
            )
        return results

    async def search(self, query: str) -> list[Memory] | None:
        raise NotImplementedError()
