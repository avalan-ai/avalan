from datetime import datetime, timezone
from json import dumps, loads
from logging import Logger
from typing import Any
from uuid import UUID, uuid4

from boto3 import client as boto_client
from ....deploy.aws import AsyncClient

from ....entities import (
    EngineMessage,
    EngineMessageScored,
    Message,
    MessageRole,
)
from ....memory.partitioner.text import TextPartition
from ....memory.permanent import (
    PermanentMessageMemory,
    VectorFunction,
)
from . import S3VectorsMemory


class S3VectorsMessageMemory(S3VectorsMemory, PermanentMessageMemory):
    def __init__(
        self,
        bucket: str,
        collection: str,
        *,
        client: Any,
        logger: Logger,
        sentence_model: Any | None = None,
    ) -> None:
        S3VectorsMemory.__init__(
            self,
            bucket=bucket,
            collection=collection,
            client=client,
            logger=logger,
        )
        PermanentMessageMemory.__init__(self, sentence_model=sentence_model)  # type: ignore[arg-type]

    @classmethod
    async def create_instance(
        cls,
        bucket: str,
        collection: str,
        *,
        logger: Logger,
        aws_client: Any | None = None,
        sentence_model: Any | None = None,
    ) -> "S3VectorsMessageMemory":
        if aws_client is None:
            aws_client = boto_client("s3vectors")
        return cls(
            bucket=bucket,
            collection=collection,
            client=AsyncClient(aws_client),
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
        message, message_partitions = self._build_message_with_partitions(
            engine_message,
            self._session_id,
            partitions,
            created_at=now_utc,
            message_id=uuid4(),
        )
        key = (
            f"{self._collection}/{message.session_id}/{message.id}.json"
            if message.session_id
            else f"{self._collection}/{message.id}.json"
        )
        await self._put_object(
            Bucket=self._bucket,
            Key=key,
            Body=dumps(
                {
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
                }
            ).encode(),
        )
        for row in message_partitions:
            await self._put_vector(
                Bucket=self._bucket,
                Collection=self._collection,
                Id=f"{row.message_id}:{row.partition}",
                Vector=row.embedding.tolist(),
                Metadata={
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
        prefix = f"{self._collection}/{session_id}/"
        response = await self._call_client(
            self._client.list_objects_v2, Bucket=self._bucket, Prefix=prefix
        )
        objects = response.get("Contents", []) if response else []
        objs = sorted(
            objects, key=lambda o: o.get("LastModified"), reverse=True
        )
        messages: list[EngineMessage] = []
        for obj in objs[: limit or len(objs)]:
            data = await self._get_object(Bucket=self._bucket, Key=obj["Key"])
            meta = loads(data["Body"].read().decode())
            messages.append(
                EngineMessage(
                    agent_id=UUID(meta["agent_id"]),
                    model_id=meta["model_id"],
                    message=Message(
                        role=MessageRole(meta["author"]), content=meta["data"]
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
            Bucket=self._bucket,
            Collection=self._collection,
            QueryVector=query,
            TopK=limit or 10,
            Function=str(function),
            Filter=filt,
        )
        results: list[EngineMessageScored] = []
        for item in response.get("Items", []):
            msg_id = item.get("Metadata", {}).get("message_id")
            if not msg_id:
                continue
            key = (
                f"{self._collection}/{session_id}/{msg_id}.json"
                if session_id
                else f"{self._collection}/{msg_id}.json"
            )
            obj = await self._get_object(Bucket=self._bucket, Key=key)
            meta = loads(obj["Body"].read().decode())
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
