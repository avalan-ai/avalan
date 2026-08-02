"""Verify encrypted conversation semantics across real PostgreSQL restarts."""

from collections.abc import Callable
from os import environ
from uuid import uuid4

import pytest
from phase2_fixtures import (
    NOW,
    authority,
    binding,
    coordinator,
    empty_stateless_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation
from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        _DSN is None,
        reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run the restart proof on asyncio only."""
    return "asyncio"


def _runtime(
    lane_binding: conversation.ProviderLaneBinding,
    result: conversation.ProviderResult,
) -> conversation.ConversationLaneRuntime:
    return conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=(result,)
        ),
    )


def _resolver(
    scope: conversation.AuthorityScope,
    keys: tuple[conversation.ConversationDataKey, ...],
) -> conversation.InMemoryConversationKeyResolver:
    return conversation.InMemoryConversationKeyResolver(
        {conversation.authority_digest(scope): keys}
    )


def _store(
    dsn: str,
    schema: str,
    resolver: conversation.InMemoryConversationKeyResolver,
) -> conversation.PgsqlConversationStore:
    return conversation.PgsqlConversationStore.from_settings(
        conversation.PgsqlConversationStoreSettings(
            dsn=dsn,
            schema=schema,
            pool_minimum=1,
            pool_maximum=2,
        ),
        key_resolver=resolver,
        cipher=conversation.AesGcmConversationCipher(),
        clock=conversation.DeterministicFakeClock(NOW),
    )


async def _drop_schema(dsn: str, schema: str) -> None:
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


async def test_normative_durable_contract(
    record_property: Callable[[str, object], None],
) -> None:
    """Match memory semantics after restart, rotation, and tamper checks."""
    record_property("conversation_acceptance_evidence", "database")
    assert _DSN is not None
    schema = f"avalan_conversation_e2e_{uuid4().hex}"
    task_pgsql_upgrade(PgsqlTaskMigrationSettings(url=_DSN, schema=schema))
    scope = authority()
    assert scope.principal_id == conversation.AuthorityPrincipalId(
        "principal-phase2"
    )
    first_key = conversation.ConversationDataKey(
        key_id="conversation-key-1",
        revision=1,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"1" * 32,
    )
    resolver = _resolver(scope, (first_key,))
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    provider_result = conversation.fake_provider_result(plan, turn=1)
    run = request(
        scope=scope,
        identity=root_identity("pgsql-restart"),
        advance=conversation.FirstTurnAdvance(),
        key="pgsql-restart-key",
        response_suffix="pgsql-restart",
    )
    first = _store(_DSN, schema, resolver)
    try:
        await first.open()
        readiness = await first.readiness(scope)
        assert readiness.key_id == first_key.key_id
        durable_receipt = await coordinator(
            store=first,
            scope=scope,
            runtimes=(_runtime(lane_binding, provider_result),),
        ).execute(run)
    finally:
        await first.close()

    restarted = _store(_DSN, schema, resolver)
    try:
        await restarted.open()
        restored_checkpoint = await restarted.load(
            durable_receipt.checkpoint.identity.checkpoint_id,
            scope,
        )
        assert run.public_response_id is not None
        restored_result = await restarted.retrieve(
            run.public_response_id,
            scope,
        )
        assert restored_checkpoint == durable_receipt.checkpoint
        assert restored_result == durable_receipt.result

        memory_receipt = await coordinator(
            store=conversation.InMemoryConversationStore(),
            scope=scope,
            runtimes=(_runtime(lane_binding, provider_result),),
        ).execute(run)
        assert memory_receipt.checkpoint == durable_receipt.checkpoint
        assert memory_receipt.result == durable_receipt.result

        second_key = conversation.ConversationDataKey(
            key_id="conversation-key-2",
            revision=2,
            status=conversation.ConversationKeyStatus.CURRENT,
            key_bytes=b"2" * 32,
        )
        grace_key = conversation.ConversationDataKey(
            key_id=first_key.key_id,
            revision=first_key.revision,
            status=conversation.ConversationKeyStatus.GRACE,
            key_bytes=first_key.key_bytes,
        )
        scope_digest = conversation.authority_digest(scope)
        await resolver.replace_keys(
            scope_digest,
            (grace_key, second_key),
        )
        rotation = await restarted.rotate_keys(scope, limit=10)
        assert rotation.examined == rotation.reencrypted == 2
        assert (
            await restarted.load(
                durable_receipt.checkpoint.identity.checkpoint_id,
                scope,
            )
            == durable_receipt.checkpoint
        )
        await restarted.retire_key(
            scope,
            key_id=grace_key.key_id,
            revision=grace_key.revision,
            at=durable_receipt.checkpoint.timestamps.committed_at,
        )
        retired_key = conversation.ConversationDataKey(
            key_id=grace_key.key_id,
            revision=grace_key.revision,
            status=conversation.ConversationKeyStatus.RETIRED,
            key_bytes=grace_key.key_bytes,
        )
        await resolver.replace_keys(scope_digest, (retired_key, second_key))

        database = restarted.database
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT ciphertext FROM conversation_encrypted_payloads"
                )
                rows = await cursor.fetchall()
                assert rows
                assert all(
                    b"visible-pgsql-restart" not in bytes(row["ciphertext"])
                    for row in rows
                )
                await cursor.execute("""
                    UPDATE conversation_encrypted_payloads
                    SET ciphertext = set_byte(
                        ciphertext,
                        0,
                        (get_byte(ciphertext, 0) + 1) % 256
                    )
                    WHERE payload_kind = 'checkpoint'
                    """)
        with pytest.raises(conversation.ConversationCryptoAuthenticationError):
            await restarted.load(
                durable_receipt.checkpoint.identity.checkpoint_id,
                scope,
            )
    finally:
        await restarted.close()
        await _drop_schema(_DSN, schema)
