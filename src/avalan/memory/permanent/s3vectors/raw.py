from ....deploy.aws import AsyncClient
from ....memory.partitioner.text import TextPartition
from ....memory.permanent import (
    Memory,
    MemoryType,
    PermanentMemory,
    VectorFunction,
)
from . import S3VectorsMemory
from asyncio import to_thread  # noqa: F401
from datetime import datetime, timezone
from json import dumps, loads
from logging import Logger
from typing import Any
from uuid import UUID, uuid4

from boto3 import client as boto_client


class S3VectorsRawMemory(S3VectorsMemory, PermanentMemory):
    _bucket: str
    _collection: str
    _client: Any
    _logger: Logger

    def __init__(
        self,
        bucket: str,
        collection: str,
        *,
        client: Any,
        logger: Logger,
    ) -> None:
        S3VectorsMemory.__init__(
            self,
            bucket=bucket,
            collection=collection,
            client=client,
            logger=logger,
        )
        PermanentMemory.__init__(self, sentence_model=None)

    @classmethod
    async def create_instance(
        cls,
        bucket: str,
        collection: str,
        *,
        logger: Logger,
        aws_client: Any | None = None,
    ) -> "S3VectorsRawMemory":
        if aws_client is None:
            aws_client = boto_client("s3vectors")
        memory = cls(
            bucket=bucket,
            collection=collection,
            client=AsyncClient(aws_client),
            logger=logger,
        )
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
        await self._put_object(
            Bucket=self._bucket,
            Key=f"{self._collection}/{entry.id}.json",
            Body=dumps(
                {
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
                }
            ).encode(),
        )
        for row in partition_rows:
            await self._put_vector(
                Bucket=self._bucket,
                Collection=self._collection,
                Id=f"{row.memory_id}:{row.partition}",
                Vector=row.embedding.tolist(),
                Metadata={
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
            Bucket=self._bucket,
            Collection=self._collection,
            QueryVector=query,
            TopK=limit or 10,
            Function=str(function),
            Filter={
                "memory_id": "*",
                "participant_id": str(participant_id),
                "namespace": namespace,
            },
        )
        results = []
        for item in response.get("Items", []):
            mem_id = item.get("Metadata", {}).get("memory_id")
            if not mem_id:
                continue
            obj = await self._get_object(
                Bucket=self._bucket,
                Key=f"{self._collection}/{mem_id}.json",
            )
            meta = loads(obj["Body"].read().decode())
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
