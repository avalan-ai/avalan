# Conversation continuity

Avalan conversation continuity preserves the exact provider and tool state
needed to continue a run without making that private state part of the public
answer. It is an explicit contract: choose a mode, retain the returned typed
handle, and use a continuation operation. Avalan never changes modes because a
field happened to be present.

The public continuity surfaces are the direct model SDK, the agent SDK, and
served Responses. Provider-backed production dispatch remains disabled until
an exact, reviewed provider row has current deterministic and authorized live
evidence. The examples in this guide use repository-owned deterministic local
providers; they are not evidence that a hosted provider is active.

## Choose a mode

| Mode | Provider state | Avalan state | Continuation input |
| --- | --- | --- | --- |
| `off` | none | none | a new one-shot request |
| `stateless` | not stored for continuation | ordered provider items in a checkpoint or caller-held envelope | a `StatelessParent` or complete replay |
| `stored` | stored after explicit disclosure | an immediate private upstream parent plus public mapping | a `StoredParent` or public `previous_response_id` |

`OneShotConversationSettings` preserves one-shot behavior. A stateless turn
uses `StatelessConversationSettings`; a stored turn requires
`StoredConversationSettings(provider_storage_disclosed=True)`. Parent types
are mode-specific, so a stored handle cannot be passed as a stateless parent.
The unsupported `convert` operation fails closed. Use `reset` when changing
mode is allowed to discard opaque continuity.

## Direct client

Construct `DirectConversationClient` with a trusted
`DirectConversationRuntime`. The runtime owns authority, the exact provider
lane binding, storage, retention, and the run-scoped coordinator. Application
input does not select those authorities.

### Handles and immutable branches

Every committed turn returns a `DirectConversationResult` containing visible
`output`, bounded `usage`, effective `reasoning`, and a mode-specific `handle`.
Treat a handle as an immutable parent:

- `continue_conversation` creates a child on the parent's branch.
- `branch` requires `ConversationBranchIntent` and a different branch ID.
- Both operations leave the parent checkpoint unchanged.
- `NamedHeadParent` adds a compare-and-swap head ID and expected revision.
  Exactly one concurrent writer can advance a revision. A stale writer gets
  `conversation_conflict`; it is never silently rebased.
- Applications create or restore named heads through their trusted storage and
  migration boundary. Request data cannot invent storage authority.

Do not derive a parent from output text or a provider ID. Keep the typed handle
returned by the committed operation.

### Reasoning context

`reasoning_context` accepts `auto`, `current_turn`, or `all_turns`. The result
records requested and provider-reported effective values separately. Avalan
does not claim a requested context became effective unless the provider
reported it. Stateless continuation replays every ordered output item needed by
the selected profile, including encrypted reasoning and assistant phase. Code
must not extract, print, edit, summarize, or reorder opaque items.

### Inline and standalone compaction

`InlineCompaction(compact_threshold=...)` is part of a create or continue
request. It is not an implicit mode switch. It can be dispatched only when the
exact lane profile proves inline compaction for the selected mode, transport,
and reasoning context.

`client.compact(StandaloneCompactRequest(parent=...))` returns a
`StandaloneCompactResult`. Its canonical context may contain retained items in
addition to a compaction item. Do not prune or edit it. The result is not a
normal continuation parent until the caller explicitly chooses one operation:

- `commit_compact(result)` commits it on the existing branch.
- `fork_compact(result, branch_id)` commits it on a new branch.

Stored-parent standalone compaction is unsupported. Reset or use an explicitly
defined stateless path; do not relabel a stored handle.

### Streaming and terminal access

Passing `stream=True` returns `DirectConversationStream`. It yields visible
`DirectConversationOutputDelta` values followed by exactly one
`DirectConversationStreamTerminal` after the checkpoint and publication
commit. `stream.committed_handle` and `stream.terminal` raise
`ConversationHandleUnavailableError` before that successful terminal state.
On failure, cancellation, or early close, there is no usable continuation
handle. Do not retry after visible output without an explicit reconciliation
decision.

### Reset

`ConversationResetIntent` names the current typed parent and target mode.
`client.reset` resolves the parent under the current authority, starts a fresh
root, and records that opaque continuity was lost. Stored reset also requires
`provider_storage_disclosed=True`. Reset is the safe response to an intentional
semantic break; it is not a fallback for binding drift or malformed state.

### Typed errors

Catch `ConversationError` for domain failures and inspect its content-safe
`code` and `boundary`. Error messages never include prompts, tool arguments,
opaque provider state, envelopes, keys, or upstream identifiers.

| Code | Meaning |
| --- | --- |
| `conversation_validation_failed` | malformed or contradictory input |
| `conversation_capability_unsupported` | exact requested behavior is inactive or unsupported |
| `conversation_binding_drift` | stored state no longer matches the exact lane binding |
| `conversation_conflict` | compare-and-swap, branch, or idempotency conflict |
| `conversation_integrity_failed` | checkpoint or state integrity failed |
| `conversation_expired` | state passed its effective lifetime |
| `conversation_deleted` | state is tombstoned or deleted |
| `conversation_storage_failed` | local storage operation failed |
| `conversation_dispatch_ambiguous` | dispatch may have occurred and needs reconciliation |
| `conversation_state_commit_failed` | provider output could not be atomically committed |
| `conversation_publication_failed` | outward publication failed |
| `conversation_authorization_failed` | authority cannot resolve the requested state |
| `conversation_limit_exceeded` | a configured state or replay bound was exceeded |
| `conversation_codec_failed` | encoded state is malformed or unsupported |
| `conversation_transition_invalid` | lifecycle transition is not allowed |

Durable deployments can also return
`conversation_key_missing`, `conversation_key_retired`,
`conversation_key_compromised`, `conversation_key_policy_invalid`,
`conversation_crypto_authentication_failed`,
`conversation_feature_unavailable`, and
`conversation_migration_required`.

### Deterministic SDK example

[conversation_continuity_sdk.py](examples/conversation_continuity_sdk.py)
executes create, streaming terminal access, continue, branch, standalone
compact, compact commit, and reset using only a process-local synthetic
provider:

```bash
poetry run python docs/examples/conversation_continuity_sdk.py
```

It prints visible results and structural booleans only. It never prints the
provider ledger or opaque compaction value.

## Served Responses API

Avalan serves these strict endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/responses` | create, continue, or stream a response |
| `POST` | `/responses/compact` | standalone stateless compaction |
| `GET` | `/responses/{response_id}` | retrieve a stored public resource |
| `DELETE` | `/responses/{response_id}` | tombstone and reconcile deletion |

Unknown fields are rejected. The exact generated schema and examples are
tracked in `tests/fixtures/conversation/served_responses_openapi.phase10.json`.
`store` defaults to `false`, which prevents stored continuation; it does not by
itself create caller-held state.

### Stored continuation

Set `store=true` on the first request and disclose provider storage in the
deployment policy. Continue with the immediate public Avalan response ID in
`previous_response_id` and `store=true`. Public IDs start with the Avalan
namespace and are not provider IDs. Avalan retains upstream IDs only inside
private encrypted lane state. Retrieval and deletion accept public IDs only.

Stored retention has independent local and upstream axes. The effective
lifetime is the shortest applicable configured limit. A deployment without a
durable store can still serve one-shot and allowed caller-held stateless
requests, but must reject durable stored continuation before dispatch.

### Canonical stateless replay

For plain stateless continuation, keep the original inputs, append every output
item in order, then append the next user item and call `/responses` again with
`store=false`. Preserve reasoning, compaction, tool adjacency, and assistant
phase. Never reduce replay to visible text.

### Caller-held continuation envelopes

An agent or multi-lane stateless response can return a terminal-only Avalan
continuation extension. Continue with:

```json
{
  "model": "configured-model",
  "input": "Continue the task.",
  "store": false,
  "extensions": {
    "avalan": {
      "version": "1",
      "conversation": {
        "version": "1",
        "mode": "caller_held",
        "continuation_envelope": "<value returned by the prior terminal>",
        "operation": "continue"
      }
    }
  }
}
```

The placeholder above must be replaced in memory from the prior terminal; do
not copy a real envelope into source code, logs, URLs, screenshots, analytics,
or issue trackers. `branch` additionally requires `branch_id`. `named_head`
requires `head_id` and `expected_head_revision`; revision zero may create a new
head without an envelope. Caller-held mode and `store=true` are mutually
exclusive.

### Compact, retrieve, and delete

`POST /responses/compact` accepts either complete, fully tagged canonical
input or an authorized caller-held envelope. It rejects
`previous_response_id`, partial items, references, and broken tool pairs. Treat
the entire returned `output` as the canonical next input window.

`GET` returns the committed public response. `DELETE` first makes the local
resource unavailable, then reports whether upstream deletion is reconciled or
pending outbox reconciliation. Repeating delete is idempotent only within the
documented authority and lifecycle policy; callers must not infer existence
across authorities from error wording.

### Streaming

`stream=true` returns server-sent events. Consume through a terminal
`response.completed`, `response.failed`, or `response.incomplete` event and the
`[DONE]` marker. Continuation state is terminal-only. If the connection ends
after visible deltas but before a committed terminal, do not manufacture a
parent or replay the visible fragment as if it were committed.

### OpenAI-shaped errors

HTTP failures use:

```json
{
  "error": {
    "message": "conversation input is invalid",
    "type": "invalid_request_error",
    "code": "conversation_validation_failed",
    "param": null
  }
}
```

Validation, unsupported capability, conflict, authorization, limits, and
commit failures map to stable content-safe codes. Clients must branch on
`error.code`, not parse `message`.

### Deterministic served example

[conversation_responses_local.py](examples/conversation_responses_local.py)
executes canonical replay, standalone compact, stored chaining,
retrieve/delete, and terminal streaming through a process-local HTTP transport:

```bash
poetry run python docs/examples/conversation_responses_local.py
```

The host uses the reserved `.invalid` domain and the transport never opens a
socket. Replace the transport with the application's authenticated Avalan base
URL only after its exact capability configuration is approved.

## Agent, tool, and structured-input boundaries

Agent continuity owns private provider/tool execution segments and produces one
outward checkpoint per completed turn. Each parent or child agent/model slot
has a deterministic provider lane. A child lane retains its own provider state;
the parent receives only the child's public result. Private reasoning and tool
state are never merged across lanes.

Tool calls and tool results remain adjacent and exactly-once within their lane.
An ambiguous provider dispatch or tool effect is fenced for reconciliation,
not automatically repeated. Structured input suspends at a durable internal
checkpoint, records the continuation ID, and resumes once under the same
authority and execution definition. A fresh worker can resume from durable
state; a client cannot fabricate a continuation from the displayed question.

The agent SDK and served Responses surfaces are activated at the framework
contract level. Conversation continuity through CLI, Flow, MCP, and A2A is
intentionally deferred and fails with `conversation_capability_unsupported`.
MCP items inside a supported provider ledger do not activate an MCP transport
continuation surface. The existing CLI `--conversation` option is a visible
repeated-message loop and does not enable provider continuation.

## Configuration reference

| Boundary | Required configuration | Fail-closed rule |
| --- | --- | --- |
| Direct runtime | trusted authority, exact lane, coordinator, store, retention | request data cannot replace runtime authority |
| Stateless SDK | stateless settings and optional stateless parent | stored parent and implicit conversion rejected |
| Stored SDK | stored settings plus explicit provider-storage disclosure | omission or false disclosure rejected |
| Inline compact | positive threshold and exact capability row | unsupported cross-product rejected before dispatch |
| Standalone compact | stateless parent and complete canonical provider output | stored parent, pruning, or partial output rejected |
| Served stored | `store=true`, durable local policy where restart is required | `previous_response_id` with `store=false` rejected |
| Served caller-held | `store=false`, versioned extension, authenticated envelope | stored mixing or unversioned state rejected |
| Streaming | separately proven streaming transport | non-stream evidence cannot activate stream dispatch |
| Native provider | exact endpoint/API/SDK/model revision and active manifest row | generic-compatible identity remains incapable |

Configuration precedence is explicit SDK/request intent inside a trusted host
profile, then trusted deployment configuration. Environment discovery supplies
operator-owned values only; request payloads never select credentials, storage
keys, tenant scope, provider endpoint, or activation authority.

## Capability and evidence status

Evidence was refreshed on **2026-08-03**. The tracked evidence records OpenAI
conversation-state, reasoning-context, compaction, create/compact/retrieve/
delete references and Microsoft native Azure Responses documentation, together
with the installed typed `openai` SDK surface. It permits conformance testing;
it does not grant activation authority.

| Provider lane | Exact API form | SDK range | Candidate coverage | Production status |
| --- | --- | --- | --- | --- |
| Native OpenAI | `https://api.openai.com/v1`, `openai_responses_v1`, `openapi-2.3.0` | `>=2.42.0,<3.0.0` | exact `gpt-5.6-sol` model identity retrieved; inference matrix blocked before its first case by account credit quota | inactive; no completed live capability receipt |
| Native Azure OpenAI | `https://{resource}.openai.azure.com/openai/v1`, `azure-openai-v1-preview` | `>=2.42.0,<3.0.0` | six exact deployment/revision candidates evaluated; `gpt-5.6-terra` revision `2026-07-09` completed all eight cases | inactive; exact live proof exists but the post-live harness delta is pending review |
| Generic OpenAI-compatible | no approved native identity | none | no cross-product approved | incapable; reject before dispatch |
| Deterministic synthetic | repository-owned local test profile | repository runtime | contract, examples, negative and E2E tests | test-only; never production advertised |

The atomic activation fixture is
`tests/fixtures/conversation/activation_manifest.phase12.json`. Its state is
`inactive`, both production flags are `false`, `active_production_rows` is
empty. This is an inactive pending-review decision, sealed by its
content-digest review signature; that digest is not authenticated signer proof.
The mapping is deliberately non-vacuous: it records one evaluated native
OpenAI profile, six evaluated Azure candidates, one complete Azure live
profile, the incomplete cross-provider `CONV-E2E-015` state, and exact
deterministic fail-closed nodes. Do not advertise any row until a later
reviewed replacement atomically names every active row and its required proof.

Redacted authorized native-provider outcomes are in
`tests/fixtures/conversation/live_conformance_results.phase12.json`. The exact
native OpenAI model retrieve confirmed `gpt-5.6-sol`. The first typed
store-free harness request and one bounded diagnostic retry were rejected
before inference with HTTP 429 `credit_balance_exhausted` /
`insufficient_quota`. No project or organization selector was configured, the
safe error identified no project or account, and no stored response,
compaction item, or opaque-content receipt was created. The exact Azure
deployments were `gpt-5`, `gpt-5-mini`, `gpt-5.4-mini`, `gpt-5-nano`,
`gpt-5.6-terra`, and `gpt-5.6-sol` at their recorded deployment revisions.
The exact `gpt-5.6-terra` revision completed all eight required cases with 11
logical operations and 11 HTTP requests: seven create-or-stream, one compact,
one retrieve, and two delete requests, with zero retries, zero unexpected
requests, zero path-class mismatches, and completed protected cleanup and client
close. Those counts come from the tracked CLI transport's actual logical
operation boundaries and content-free HTTP request hook; they are included in
the receipt and its structural digest rather than copied from an external
wrapper. The live harness and traceability delta remains pending independent
review, and native OpenAI remains quota-blocked, so every production row stays
inactive. Partial, rejected, and superseded observations for the other Azure
profiles are not capability proof. The separate inactive preflight fixture
remains available for the opt-in redacted harness; it cannot grant activation.

Provider evidence is in
`tests/fixtures/conversation/provider_evidence.phase12.json`. Current official
sources include:

- [OpenAI conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
- [OpenAI reasoning](https://developers.openai.com/api/docs/guides/reasoning)
- [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction)
- [OpenAI Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create)
- [Azure OpenAI Responses](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses)
- [Azure OpenAI Responses REST reference](https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/responses)

OpenAI documents complete ordered output replay for stateless reasoning,
`previous_response_id` for stored chaining, 30-day default response-object
storage unless `store=false`, reasoning-context selection, and canonical
standalone compact output. Conversation objects have different retention
semantics from response objects. Azure documents the `/openai/v1` native form
and deployment-name request field. Live proof separately requires
`response.model` to match the reviewed Azure model identity while the exact
deployment revision remains a separate configuration pin. These are provider
facts, not an Avalan promise that every model, deployment, region, or
cross-product is active.

## Related guides

- [Conversation security](CONVERSATION_SECURITY.md)
- [Conversation operations](CONVERSATION_OPERATIONS.md)
- [Conversation migration v1](CONVERSATION_MIGRATION_V1.md)
