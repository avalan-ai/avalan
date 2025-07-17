from datetime import datetime, timezone
from logging import Logger
from typing import Any
from uuid import UUID, uuid4

from elasticsearch import AsyncElasticsearch

from ....entities import (
    EngineMessage,
    EngineMessageScored,
    Message,
    MessageRole,
)
from ....memory.partitioner.text import TextPartition
from ....memory.permanent import (
    PermanentMessage,
    PermanentMessageMemory,
    PermanentMessagePartition,
    VectorFunction,
)
from . import ElasticsearchMemory


class ElasticsearchMessageMemory(ElasticsearchMemory, PermanentMessageMemory):
    def __init__(
        self,
        index: str,
        *,
        client: Any,
        logger: Logger,
        sentence_model: Any | None = None,
    ) -> None:
        ElasticsearchMemory.__init__(
            self, index=index, client=client, logger=logger
        )
        PermanentMessageMemory.__init__(self, sentence_model=sentence_model)  # type: ignore[arg-type]

    @classmethod
    async def create_instance(
        cls,
        index: str,
        *,
        logger: Logger,
        es_client: Any | None = None,
        sentence_model: Any | None = None,
    ) -> "ElasticsearchMessageMemory":
        if es_client is None:
            es_client = AsyncElasticsearch()
        return cls(
            index=index,
            client=es_client,
            logger=logger,
            sentence_model=sentence_model,
        )

    async def create_session(
        self, *, agent_id: UUID, participant_id: UUID
    ) -> UUID:
        return uuid4()

    async def continue_session_and_get_id(
        self,
        *,
        agent_id: UUID,
        participant_id: UUID,
        session_id: UUID,
    ) -> UUID:
        return session_id

    async def append_with_partitions(
        self,
        engine_message: EngineMessage,
        *,
        partitions: list[TextPartition],
    ) -> None:
        assert engine_message and partitions
        now_utc = datetime.now(timezone.utc)
        message_id = uuid4()
        message = PermanentMessage(
            id=message_id,
            agent_id=engine_message.agent_id,
            model_id=engine_message.model_id,
            session_id=self._session_id,
            author=engine_message.message.role,
            data=engine_message.message.content,
            partitions=len(partitions),
            created_at=now_utc,
        )
        key = (
            f"{self._index}/{message.session_id}/{message.id}.json"
            if message.session_id
            else f"{self._index}/{message.id}.json"
        )
        await self._index_document(
            index=self._index,
            id=key,
            document={
                "id": str(message.id),
                "agent_id": str(message.agent_id),
                "model_id": message.model_id,
                "session_id": (
                    str(message.session_id) if message.session_id else None
                ),
                "author": str(message.author),
                "data": message.data,
                "partitions": message.partitions,
                "created_at": message.created_at.isoformat(),
            },
        )
        for idx, part in enumerate(partitions):
            row = PermanentMessagePartition(
                agent_id=message.agent_id,
                session_id=message.session_id,
                message_id=message.id,
                partition=idx + 1,
                data=part.data,
                embedding=part.embeddings,
                created_at=now_utc,
            )
            await self._index_vector(
                index=self._index,
                id=f"{row.message_id}:{row.partition}",
                vector=row.embedding.tolist(),
                metadata={
                    "message_id": str(row.message_id),
                    "agent_id": str(row.agent_id),
                    "session_id": (
                        str(row.session_id) if row.session_id else None
                    ),
                },
            )

    async def get_recent_messages(
        self,
        session_id: UUID,
        participant_id: UUID,
        *,
        limit: int | None = None,
    ) -> list[EngineMessage]:
        response = await self._call_client(
            self._client.search,
            index=self._index,
            body={
                "query": {"term": {"session_id": str(session_id)}},
                "sort": [{"created_at": {"order": "desc"}}],
                "size": limit or 10,
            },
        )
        hits = response.get("hits", {}).get("hits", []) if response else []
        messages: list[EngineMessage] = []
        for hit in hits:
            meta = hit.get("_source")
            if not meta:
                continue
            messages.append(
                EngineMessage(
                    agent_id=UUID(meta["agent_id"]),
                    model_id=meta["model_id"],
                    message=Message(
                        role=MessageRole(meta["author"]),
                        content=meta["data"],
                    ),
                )
            )
        return messages

    async def search_messages(
        self,
        *,
        agent_id: UUID,
        function: VectorFunction,
        limit: int | None = None,
        participant_id: UUID,
        search_partitions: list[TextPartition],
        search_user_messages: bool,
        session_id: UUID | None,
        exclude_session_id: UUID | None,
    ) -> list[EngineMessageScored]:
        assert agent_id and participant_id and search_partitions
        query = search_partitions[0].embeddings.tolist()
        filt = {
            "message_id": "*",
            "agent_id": str(agent_id),
            "participant_id": str(participant_id),
        }
        if session_id:
            filt["session_id"] = str(session_id)
        response = await self._query_vector(
            index=self._index,
            query_vector=query,
            top_k=limit or 10,
            function=str(function),
            filter=filt,
        )
        results: list[EngineMessageScored] = []
        for item in response.get("Items", []):
            msg_id = item.get("Metadata", {}).get("message_id")
            if not msg_id:
                continue
            key = (
                f"{self._index}/{session_id}/{msg_id}.json"
                if session_id
                else f"{self._index}/{msg_id}.json"
            )
            obj = await self._get_document(index=self._index, id=key)
            meta = obj.get("_source") if obj else None
            if not meta:
                continue
            results.append(
                EngineMessageScored(
                    agent_id=UUID(meta["agent_id"]),
                    model_id=meta["model_id"],
                    message=Message(
                        role=MessageRole(meta["author"]), content=meta["data"]
                    ),
                    score=item.get("Score", 0.0),
                )
            )
        return results
