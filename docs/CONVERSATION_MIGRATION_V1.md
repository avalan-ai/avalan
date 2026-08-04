# Conversation continuity migration v1

This guide migrates existing Avalan applications to the version 1 continuity
contract. Migration is opt-in. Existing one-shot behavior remains the default,
and the legacy CLI conversation loop does not gain provider continuation.

## Compatibility rules

- Keep mode explicit: `off`, `stateless`, or `stored`.
- Keep parent types explicit and mode-specific.
- Preserve complete ordered provider output for stateless replay. Visible text
  alone is not continuity.
- Keep public Avalan response IDs separate from private upstream IDs.
- Treat caller-held envelopes as session state and bind them to authenticated
  authority.
- Preserve tool call/result adjacency and idempotency fences.
- Use an explicit reset when continuity cannot be preserved. Never silently
  degrade to visible transcript replay.
- Keep CLI, Flow, MCP, and A2A continuity disabled until each surface has an
  explicit supported contract.

## One-shot direct calls

Before migration, applications call a model once and may retain visible output
themselves. Keep that behavior with `OneShotConversationSettings()` or the
existing non-continuation model API. No checkpoint or provider state is created.

To opt into stateless continuity:

1. Build a trusted `DirectConversationRuntime` with exact authority, lane,
   retention, and store.
2. Create with `StatelessConversationSettings()`.
3. Retain the returned `StatelessConversationHandle` as application state.
4. Continue with `StatelessParent(handle=...)`.
5. Add an explicit branch intent for forks; do not overwrite the parent.

To opt into stored continuity, complete provider-retention disclosure and
deployment storage approval first. Use
`StoredConversationSettings(provider_storage_disclosed=True)` and retain the
`StoredConversationHandle`. A request must never select stored mode merely by
supplying an ID.

If the application cannot retain a typed handle, stay one-shot or use a served
public contract. Do not serialize `repr(handle)` or copy internal fields into an
ad hoc token.

## Tool replay loops

Legacy tool loops often retain messages and visible function results but omit
provider reasoning or provider item identity. That representation is not enough
for v1 stateless continuity.

Migrate at the provider-item boundary:

1. Normalize every provider item with stable order, caller, phase, call ID, and
   completion state.
2. Keep tool calls adjacent to their outputs and reject missing, duplicate, or
   cross-lane results.
3. Commit internal provider/tool segments exactly once.
4. Publish only the public projection while retaining the full private ledger.
5. On retry, resolve the idempotency fence before repeating either provider or
   tool effects.

Old transcripts without complete item state cannot be upgraded by guessing.
Finish them as legacy one-shot history, or reset into a new v1 root. Record the
reset so callers know opaque continuity was lost.

## Served Responses clients

Existing `/responses` one-shot clients remain valid because `store` defaults to
`false`. Unknown fields now fail validation instead of being ignored.

Choose one migration:

### Canonical stateless replay

Keep the original input array, append every item from `response.output` in
order, append the next user item, and send the complete array with
`store=false`. Preserve assistant `phase`, encrypted reasoning, compaction, and
tool items. This path is caller-managed and creates no stored parent.

### Stored public-ID continuation

Set `store=true` on the first call. Continue with `store=true` and
`previous_response_id` equal to the immediately preceding public Avalan
response ID. Configure durable local storage for restart continuity and accept
the disclosed provider retention policy. Never pass a provider upstream ID.

### Caller-held agent envelope

For supported agent or multi-lane stateless responses, keep the terminal Avalan
continuation envelope in protected session state. Send it back in the version 1
conversation extension with `mode=caller_held` and `store=false`. Do not put the
envelope in a URL, log, source file, or client-visible analytics event.

Clients that used arbitrary `extra_body` continuation fields must migrate to
the strict documented schema. Unsupported fields are rejected; Avalan does not
accept and discard them.

### Lifecycle operations

Use `GET /responses/{public_response_id}` and
`DELETE /responses/{public_response_id}` only for stored resources. Delete is a
local tombstone plus independently reconciled upstream effect. A caller-held
envelope is not a retrieval ID.

## Structured input

Legacy attached prompts may pause only inside one live process. Durable v1
structured input persists an internal suspension checkpoint, request identity,
authority, lane topology, execution definition, and exactly-once resolution
state.

Migrate by selecting one handling mode before the run:

- `attached` keeps the interaction on the connected controller and has explicit
  disconnect/cancellation behavior.
- `detached` returns an input-required public result and resumes through the
  durable controller API.
- `unavailable` rejects requests that require input.

A model-visible question is not a resume token. Resolve by the typed request and
continuation references under the same authority. Repeated answers return the
documented idempotent result or conflict; they never advance twice. If legacy
state lacks a durable continuation reference, restart the logical task at a new
root rather than reconstructing private execution from its question text.

## Agent and multi-agent applications

Assign one deterministic provider lane to each parent/child agent and model
slot. Keep child private state in the child lane and return only its public
result to the parent. Durable resume restores topology, lane ownership, provider
binding, execution segment, tool adjacency, and structured-input state before
advancing.

Applications that previously flattened all agent messages into one history
must reset at the migration boundary unless they can prove exact lane ownership
for every retained provider item. Do not merge encrypted reasoning across
models or agents.

## CLI users

`avalan agent ... --conversation` remains a repeated visible-message loop. It
does not enable provider conversation continuation, caller-held envelopes,
stored response lifecycle, branching, compaction, or durable restart.

For automation that needs v1 continuity, migrate to the direct SDK or served
Responses API. Keep existing CLI scripts unchanged for one-shot/repeated-message
behavior. Do not scrape CLI output for a handle. A future CLI continuity surface
requires its own explicit activation and migration contract.

## Versioned durable deployment

Deploy readers before writers:

1. Back up encrypted state and separately verify key recovery.
2. Deploy N+1 readers that accept the documented N/N+1 window for every state
   surface.
3. Run schema migration and readiness checks.
4. Switch writers to N+1 only after all readers and historical resolvers are
   ready.
5. Rehearse rollback with N and N+1 state, including retrieve, continue,
   resolve, delete, expiry, named heads, and outbox work.
6. Atomically activate only reviewed provider rows.

After rollback, old state is either resolvable within the compatibility window
or deterministically unavailable. Delete remains allowed. Visible transcript
fallback and automatic mode conversion remain forbidden.

## Migration outcomes

| Existing path | v1 path | Continuity outcome |
| --- | --- | --- |
| one-shot model call | one-shot settings | unchanged; no continuity state |
| complete provider-item loop | stateless typed handle or canonical replay | preserved when every ordered item is retained |
| visible-message-only tool loop | new stateless root | explicit reset; opaque continuity cannot be proven |
| served one-shot | `/responses`, default `store=false` | unchanged |
| server-retained response IDs | `store=true` plus public immediate parent | preserved after ID and retention validation |
| arbitrary continuation token | version 1 caller-held envelope | reset unless produced by the trusted codec and authority |
| process-local input pause | durable structured-input integration | preserved only with a committed suspension reference |
| flattened multi-agent history | isolated deterministic lanes | reset unless ownership can be proven exactly |
| CLI repeated-message loop | unchanged CLI or SDK/served migration | CLI semantics unchanged; provider continuity not implied |

Run both deterministic examples after migration:

```bash
poetry run python docs/examples/conversation_continuity_sdk.py
poetry run python docs/examples/conversation_responses_local.py
```

Then follow the storage and rollback gates in
[Conversation operations](CONVERSATION_OPERATIONS.md) and the envelope rules in
[Conversation security](CONVERSATION_SECURITY.md).
