# Memory PostgreSQL Storage

This document records the operational shape for permanent memory
PostgreSQL schemas. It applies to message memory, raw memory,
embedding partitions, and reasoning graph tables.

## Migration Framework

Alembic is the authoritative schema migration framework for memory
PostgreSQL schemas. Revisions are hand-authored PostgreSQL migrations
stored under `src/avalan/memory/permanent/pgsql_migrations/`.

Runtime memory modules do not apply schema changes on hot request paths.
Operators should use the programmatic migration helpers to apply,
inspect, validate, or stamp schema state.

The helper API is exposed from
`avalan.memory.permanent.pgsql_migrations` as `memory_pgsql_upgrade`,
`memory_pgsql_current`, `memory_pgsql_check`, and `memory_pgsql_stamp`.

## Revision Layout

Memory revisions live in a memory-owned Alembic script location with a
separate revision history from task storage. The default version table is
`avalan_memory_alembic_version`.

Task migrations do not import memory modules and do not require pgvector.
Memory migrations own the `vector`, `ltree`, and `pg_trgm` extension
requirements used by permanent memory tables and indexes.

## Extension Requirements

Permanent memory requires these PostgreSQL extensions:

- `vector` for embedding columns and IVFFLAT indexes.
- `ltree` for namespace tree generated columns and GiST indexes.
- `pg_trgm` for trigram lookup on reasoning graph entity names.

The migration revisions issue `CREATE EXTENSION IF NOT EXISTS` before
objects that depend on each extension. `pg_trgm` is declared before the
`gin_trgm_ops` index is created.

The runtime memory connection still verifies the `vector` extension and
registers pgvector codecs with psycopg. Composite type registration for
`message_author_type` continues to happen through the existing connection
configuration path.

## Managed Providers

Hosted PostgreSQL providers differ in who may create extensions. Some
allow the database owner to run `CREATE EXTENSION`; others require a
provider administrator, a privileged maintenance role, or an allowlisted
extension catalog.

For AWS RDS/Aurora, Azure Database for PostgreSQL, Cloud SQL, Supabase,
and Neon, confirm that `vector`, `ltree`, and `pg_trgm` are available in
the target database before running migrations. If extension creation is
restricted, pre-provision the extensions in the target database or run
the migrations with the provider-approved role.

When migrations run in an isolated schema, the Alembic environment sets
the search path to the target schema and `public` so pre-provisioned
extensions remain visible.

## Testing

Set `AVALAN_MEMORY_TEST_POSTGRESQL_DSN` to run the env-gated PostgreSQL
memory migration test. The test creates an isolated schema and upgrades
the memory migration history to the current head.
