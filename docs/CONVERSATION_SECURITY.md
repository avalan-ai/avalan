# Conversation continuity security

Conversation state can contain prompts, provider reasoning state, tool calls,
tool results, structured-input suspension data, and child-agent lane state.
Keep public text, public response IDs, local checkpoint IDs, upstream provider
IDs, and caller-held envelopes as separate data classes. Possession of one
must not grant access to another.

## Caller-held envelopes are session state

An Avalan continuation envelope is authenticated ciphertext, not a bearer
authorization mechanism. Treat it with the same care as a session cookie:

- Send it only over authenticated TLS. Reject plaintext HTTP at every
  network-exposed boundary.
- Carry it in the versioned request body field. Never put it in a URL, query
  string, path, redirect, referrer, command-line argument, or form field that
  may be retained by infrastructure.
- Never log, print, trace, meter, index, cache-key, screenshot, or attach it to
  an issue. Redact request bodies before application, proxy, WAF, APM, and
  exception logging.
- Keep it out of browser history, local storage, analytics, crash dumps, and
  clipboard-driven support workflows. Prefer an in-memory or platform secret
  container with the shortest practical lifetime.
- Return it only with a successfully committed terminal response. Deltas,
  preambles, failures, disconnects, and provisional IDs carry no usable
  continuation authority.
- Enforce size, decoded-item, depth, lane-count, tool-pair, and replay bounds
  before provider dispatch.

Encryption prevents disclosure to a party without the key. It does not make an
envelope safe to publish, prove that the caller is authorized, or prevent replay
by an authorized party. Every open operation must bind authenticated tenant,
principal, agent, endpoint, codec version, key revision, and expected branch or
head operation.

## Authority and identifier separation

Network requests derive authority from authenticated server context. A payload
cannot supply or override tenant, principal, endpoint, agent, storage key,
provider credential, or provider endpoint. Fixed local single-user authority is
valid only when explicitly configured and cannot be network exposed.

Use only public Avalan response IDs in served retrieve, continue, and delete
routes. Do not expose or accept upstream provider response IDs. Checkpoint IDs
and lane IDs are internal unless an explicitly versioned response contract
projects a safe alias. Return the same content-safe unavailable response for
missing, expired, deleted, and cross-authority state where distinguishing them
would create an existence oracle.

## TTL and retention

Configure local checkpoint, envelope, and known upstream lifetimes
independently. The effective lifetime is the minimum applicable value. Enforce
expiry before decrypting large payloads, resolving a parent, dispatching a
provider, or running a tool. Sweeping expired state is cleanup, not the
authorization decision.

`store=false` controls response storage for a request; it does not authorize
unbounded caller-held retention. Provider response retention and provider
conversation-object retention may differ. Keep provider documentation and the
exact active profile under review, and shorten local retention if the upstream
lifetime becomes uncertain.

Deletion first denies local access. Upstream deletion is an independently
reconciled effect. During an upstream outage, retain only the encrypted
deletion intent and content-safe identifiers needed to retry; do not restore
the public response while reconciliation is pending.

## Replay and branch policy

Bind idempotency keys to the normalized semantic request, authority, operation,
and parent. Reusing a key with different content is a conflict. A safe retry may
return an already committed result; it must not duplicate provider or tool
effects.

Ordinary continuation advances one immutable parent. Explicit branches require
a new branch ID. Named heads require an expected revision and compare-and-swap
advance. Reject stale revisions and cross-branch parents. Never infer branch
intent from a changed prompt, an envelope alone, or a public response ID.

An ambiguous dispatch is fenced. Reconcile it using provider evidence and the
durable idempotency record; never retry automatically merely because no public
response was observed. After any visible streaming output, retry only under an
explicit reconciliation policy.

## Key lifecycle

Use separate versioned keys for durable checkpoint encryption and caller-held
envelopes. Keep key IDs non-secret but content-safe. Store key material in the
deployment secret manager, not configuration files, database rows, fixtures,
logs, or process arguments.

The normal rotation sequence is:

1. Provision a new active write key and retain the old key as read-only.
2. Deploy resolvers that can read the previous and current revisions.
3. Switch new writes atomically to the new revision.
4. Re-encrypt or naturally expire retained state according to policy.
5. Verify no live state needs the old revision, then retire it.

Retirement is not compromise. A retired key may remain available for an
approved compatibility window, while a compromised key must be disabled for
both reads and writes immediately.

## Compromise response

If an envelope or key may be compromised:

1. Stop new dispatch for the affected profile and key revision.
2. Mark the key compromised so opens fail before state resolution or provider
   dispatch.
3. Rotate unaffected write authority and revoke affected caller sessions.
4. Bound impact by tenant, principal, agent, endpoint, key revision, issued
   time, and state lifetime without recording envelope values.
5. Tombstone affected public mappings and queue authorized provider deletion
   where disclosure and policy permit it.
6. Preserve content-safe audit evidence, reconcile ambiguous effects, and
   notify the responsible security operator.
7. Restore dispatch only through a newly reviewed activation manifest and
   completed rollback/forward rehearsal.

Do not attempt to decode suspected state in support tools or paste it into a
new environment. Do not weaken authority binding to recover an inaccessible
conversation. Reset is safer than accepting unverifiable continuity.

## Logging and observability

Allowed observations include operation, outcome, failure boundary, public
response state, safe lane alias, item kind, count, byte bounds, key ID/revision,
provider profile revision, and integrity digests. Prompt text, model output,
tool arguments/results, encrypted reasoning, compaction content, upstream IDs,
envelopes, credentials, and key material are forbidden.

Redaction must apply to values, exception strings, `repr`, structured logs,
metrics labels, traces, SSE diagnostics, and dead-letter records. Tests use a
canary to prove forbidden values are absent from every observation channel.
Hashing a low-entropy secret or opaque value does not automatically make it
safe telemetry.

Live conformance converts SDK and cleanup failures into cause-free typed
errors. Failed upstream cleanup identifiers remain only in a bounded protected
retry channel whose representation and public receipt expose no identifier,
URL, body, credential, or opaque state.

## Deployment checklist

- TLS and authenticated authority are mandatory for network exposure.
- Request-body logging is disabled or field-redacted at every hop.
- Envelope, checkpoint, provider, and public identifiers have distinct types.
- Local, envelope, and upstream TTL policies are explicit.
- Replay, branch, named-head, and idempotency conflicts fail before dispatch.
- Key resolvers support the documented compatibility window only.
- Deletion outages leave resources locally unavailable.
- Backup data is encrypted and restore rehearsals revalidate authority,
  migration version, expiry, tombstones, and activation capability.
- Generic-compatible providers and unreviewed native cross-products remain
  incapable.
- Incident responders know the stop-dispatch, compromise, rotate, reconcile,
  delete, and reviewed-reactivation sequence.

See [Conversation operations](CONVERSATION_OPERATIONS.md) for procedures and
[Conversation continuity](CONVERSATIONS.md) for the public contracts.
