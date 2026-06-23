# Memories

Avalan memory gives agents controlled context beyond the current prompt.
Memory can be session-scoped conversation history, permanent retrieval over
stored text, or a tool surface that lets an agent search prior messages and
permanent stores during work.

Use memory when context should be explicit, inspectable, and scoped by
namespace instead of silently appended to every prompt.

## Memory Types

Avalan has three practical memory surfaces:

- Recent memory keeps the current session transcript available to the agent.
- Permanent message memory persists conversation history across sessions.
- Permanent document memory stores and retrieves indexed knowledge across runs.

Recent memory is session state. Permanent memory is application state.

## Recent Memory

Recent memory is useful for chat, iterative editing, or multi-turn support
sessions.

```toml
[memory]
recent = true
```

When a conversation needs durable facts, put those facts in permanent memory
instead of relying on the current session transcript.

When continuing a durable session, Avalan can load recent messages back from
permanent message memory:

```sh
echo "Continue from the prior support conversation." \
  | avalan agent run \
    --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
    --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
    --session "83f897e0-f824-4935-bbbd-e0dc93b2b488" \
    --memory-recent \
    --memory-permanent-message "postgresql://user:pass@localhost:5432/avalan" \
    --load-recent-messages-limit 20
```

Use `--skip-load-recent-messages` when you want to continue a durable session
without replaying older messages into recent memory.

## Permanent Memory

Permanent memory stores content under namespaces and retrieves relevant
chunks for later prompts.

```toml
[memory]
recent = true

[memory.permanent]
"support.docs" = "postgresql://user:pass@localhost:5432/avalan,Support docs"
```

Permanent memory is useful for:

- Customer preferences.
- Product documentation.
- Prior decisions.
- Knowledge-base retrieval.
- Summaries of completed tasks.
- Cross-session project context.

The shared interface is `MemoryStore[T]`; concrete backends live under
[src/avalan/memory/permanent](../src/avalan/memory/permanent).

## Namespaces

Permanent memory retrieval is scoped by participant plus namespace. Use both
deliberately:

- Per tenant for SaaS applications.
- Per user for personal assistants.
- Per project for engineering workflows.
- Per domain for shared knowledge bases.

Avoid mixing unrelated knowledge in one namespace. Retrieval quality and
security both improve when memory is scoped clearly. Current memory retrieval
matches the configured namespace exactly.

## Document Indexing

Index documents before an agent can retrieve them through the memory tools:

```sh
avalan memory document index \
    --participant "user-123" \
    --dsn "postgresql://user:pass@localhost:5432/avalan" \
    --namespace "support.docs" \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "./handbook.md"
```

The source can be a local file path or URL. PDF inputs are supported by the
document loader when the needed document dependencies are installed.

Use metadata flags to make stored entries easier to inspect later:

```sh
avalan memory document index \
    --participant "user-123" \
    --dsn "postgresql://user:pass@localhost:5432/avalan" \
    --namespace "papers.arxiv" \
    --title "Generalizable End-to-End Tool-Use RL with Synthetic CodeGym" \
    --description "Paper indexed for retrieval examples." \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "https://arxiv.org/pdf/2509.17325"
```

## Chunking and Retrieval

Permanent memory usually stores chunks, not whole documents. Chunking
strategy controls what gets embedded or indexed and what the model sees
later.

Good chunks are:

- Small enough to retrieve precisely.
- Large enough to preserve meaning.
- Tagged with useful metadata.
- Stored in the namespace where they will be queried.

Chunking strategies live under
[src/avalan/memory/partitioner](../src/avalan/memory/partitioner).

The common CLI controls are:

- `--partition-max-tokens` for the maximum size of each partition.
- `--partition-overlap` for overlap between nearby partitions.
- `--partition-window` for the token window used while splitting.
- `--partitioner text|code` to choose a partitioning strategy.
- `--language` when using the code partitioner.

## Backends

Avalan includes permanent memory implementations for PostgreSQL with
pgvector, Elasticsearch-compatible stores, and AWS S3 Vectors. The documented
CLI and agent TOML workflows instantiate PostgreSQL stores today;
Elasticsearch-compatible stores and S3 Vectors are available for SDK or
application-level integrations.

PostgreSQL is the most common durable application backend because it keeps
metadata, namespaces, and retrieval close to normal application operations.

For PostgreSQL setup, see [MEMORY_POSTGRESQL.md](MEMORY_POSTGRESQL.md).

## Message Memory

Permanent message memory stores conversation history by agent, participant,
and session. Configure it on an agent:

```toml
[memory]
recent = true
permanent_message = "postgresql://user:pass@localhost:5432/avalan"
```

Then run with stable identities so later runs can search the same history:

```sh
echo "Hi, my name is Leo." \
  | avalan agent run \
      --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
      --id "f4fd12f4-25ea-4c81-9514-d31fb4c48128" \
      --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
      --session "83f897e0-f824-4935-bbbd-e0dc93b2b488" \
      --memory-recent \
      --memory-permanent-message "postgresql://user:pass@localhost:5432/avalan"
```

Search permanent message memory directly with `avalan agent message search`:

```sh
printf "%s\n" "What did the user say about retention?" \
  | avalan agent message search \
      --engine-uri "ai://env:OPENAI_API_KEY@openai/gpt-4o" \
      --id "f4fd12f4-25ea-4c81-9514-d31fb4c48128" \
      --participant "c67d6ec7-b6ea-40db-bf1a-6de6f9e0bb58" \
      --session "83f897e0-f824-4935-bbbd-e0dc93b2b488" \
      --memory-permanent-message "postgresql://user:pass@localhost:5432/avalan" \
      --function l2_distance \
      --limit 5
```

## Memory as Tools

Agents can use memory through tools. This is useful when the model should
decide what to retrieve during a run.

Examples:

- Search memory before answering a support question.
- Retrieve prior decisions before proposing a plan.
- List available memory stores before choosing a namespace.

Durable writes are handled by application code, message persistence, or
commands such as `avalan memory document index`; the current memory tool set
is read/list oriented.

```toml
[tool]
enable = ["memory"]

[memory]
recent = true

[memory.engine]
model_id = "sentence-transformers/all-MiniLM-L6-v2"
max_tokens = 500
overlap_size = 125
window_size = 250

[memory.permanent]
"support.docs" = "postgresql://user:pass@localhost:5432/avalan,Support docs"
```

## Memory Search from the CLI

Search indexed PostgreSQL memory by piping a query to `avalan memory search`:

```sh
printf "%s\n" "How does Avalan handle task file delivery?" \
  | avalan memory search \
      "sentence-transformers/all-MiniLM-L6-v2" \
      --dsn "postgresql://user:pass@localhost:5432/avalan" \
      --participant "user-123" \
      --namespace "support.docs" \
      --function l2_distance \
      --limit 5
```

Use the same embedding model for indexing and retrieval. The common PostgreSQL
path uses pgvector distance functions such as `l2_distance`,
`cosine_distance`, and `inner_product`.

## Security and Privacy

Memory can contain private or regulated data. Treat it as application data,
not as harmless prompt context.

Recommended practices:

- Separate namespaces by tenant or user.
- Store source metadata and retention information.
- Avoid writing secrets to memory.
- Redact sensitive fields before indexing.
- Log retrieval source ids, not only generated answers.
- Make durable writes explicit in application code or indexing jobs.

## Choosing Memory Scope

- Use no memory for stateless calls and deterministic extraction.
- Use recent memory for one conversation.
- Use permanent memory for knowledge that should survive a run.
- Use task storage for durable task inputs, outputs, and status.
- Use a database tool when the agent needs authoritative live records.

Memory helps the model recall context, but it should not replace application
state or source-of-truth systems.

## Related Documentation

- [AGENTS.md](AGENTS.md) - Configure memory on agents.
- [TOOLS.md](TOOLS.md) - Memory tools, retrieval, and confirmation.
- [TASKS.md](TASKS.md) - Durable execution and output storage.
- [MEMORY_POSTGRESQL.md](MEMORY_POSTGRESQL.md) - PostgreSQL memory setup.
- [CLI.md](CLI.md) - Memory CLI reference.
