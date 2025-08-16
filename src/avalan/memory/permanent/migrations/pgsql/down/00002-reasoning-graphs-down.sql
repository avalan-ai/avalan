DROP VIEW IF EXISTS "v_live_hyperedges";

DROP INDEX IF EXISTS "ix_hyperedge_entities_by_entity";
DROP TABLE IF EXISTS "hyperedge_entities";

DROP INDEX IF EXISTS "ix_entities_scope";
DROP INDEX IF EXISTS "ix_entities_name_trgm";
DROP INDEX IF EXISTS "ix_entities_embedding_ivfflat";
ALTER TABLE IF EXISTS "entities" DROP CONSTRAINT IF EXISTS "uq_entities_scope_name";
DROP TABLE IF EXISTS "entities";

DROP INDEX IF EXISTS "ix_hyperedges_memories_by_memory";
DROP TABLE IF EXISTS "hyperedges_memories";

DROP INDEX IF EXISTS "ix_hyperedges_relation_lc";
DROP INDEX IF EXISTS "ix_hyperedges_by_created_at";
DROP INDEX IF EXISTS "ix_hyperedges_embedding_ivfflat";
DROP TABLE IF EXISTS "hyperedges";
