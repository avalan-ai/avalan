from ....deploy.aws import AsyncClient
from ....memory.partitioner.text import TextPartition
from ....memory.permanent import (
    Memory,
    MemoryType,
    PermanentMemory,
    PermanentMemoryPartition,
    VectorFunction,
)
from . import S3VectorsMemory

from asyncio import to_thread  # noqa: F401
from datetime import datetime, timezone
from json import dumps, loads
from logging import Logger
from typing import Any
from uuid import UUID, uuid4

import numpy as np
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
                    "title": entry.title,
                    "description": entry.description,
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
            self._client.list_objects_v2,
            Bucket=self._bucket,
            Prefix=f"{self._collection}/",
        )
        objects = response.get("Contents", []) if response else []
        memories: list[Memory] = []
        for obj in objects:
            key = obj.get("Key")
            if not key:
                continue
            document = await self._get_object(Bucket=self._bucket, Key=key)
            body = document.get("Body") if document else None
            if not body:
                continue
            metadata = loads(body.read().decode())
            if (
                metadata.get("participant_id") != str(participant_id)
                or metadata.get("namespace") != namespace
            ):
                continue
            memories.append(
                Memory(
                    id=UUID(metadata["id"]),
                    model_id=metadata.get("model_id"),
                    type=MemoryType(metadata["type"]),
                    participant_id=UUID(metadata["participant_id"]),
                    namespace=metadata["namespace"],
                    identifier=metadata["identifier"],
                    data=metadata["data"],
                    partitions=metadata["partitions"],
                    symbols=metadata.get("symbols"),
                    created_at=datetime.fromisoformat(metadata["created_at"]),
                    title=metadata.get("title"),
                    description=metadata.get("description"),
                )
            )
        return memories

    async def search(
        self, query: str
    ) -> list[PermanentMemoryPartition] | None:
        raise NotImplementedError()
