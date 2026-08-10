# Conversation continuity operations

This runbook covers durable storage, key and retention policy, reconciliation,
provider capability changes, backup/restore, atomic activation, and rollback.
The current native-provider activation fixture is intentionally inactive. The
procedures below are rehearsal and release gates; they do not authorize hosted
provider calls or production activation.

## Roles and change control

Assign these roles before a rollout:

- The release operator owns the deployment, compare-and-swap activation, and
  rollback decision.
- The storage operator owns schema migration, backups, restore validation,
  retention, sweeper capacity, and outbox health.
- The key operator owns checkpoint and envelope keys, rotation, retirement,
  and compromise response.
- The provider owner records exact endpoint, API form, SDK, model/deployment
  revision, transport, and live conformance evidence.
- Architecture, security, and test/traceability reviewers approve the immutable
  activation content. Reviewers do not supply runtime credentials.

Use separate principals for migration, application traffic, worker effects,
and read-only health checks where the deployment platform supports them.

## Durable storage migration

The PostgreSQL implementation is currently guarded as test-only. Do not expose
production stored continuation until a release deliberately removes that guard
and completes this runbook. The tracked head revision is
`20260809_0001_patch_durable`; the application compatibility version is `2` and the current
schema read/write window is version `1`.

Before a deployment:

1. Back up the database and encryption-key metadata under the same recovery
   point objective. Never export key material into the database backup.
2. Stop or drain writers if the migration is not proven online-compatible.
3. Install the required server and PostgreSQL extras.
4. Run the repository migration command with the deployment's secret-injected
   DSN:

   ```bash
   python3 -m pip install -U "avalan[task-pgsql,server]"
   avalan task pgsql migrate head
   ```

5. Open the store with `check_schema_on_open=True`. Require readiness to report
   the expected schema, application version, checkpoint codec, current write
   key ID, and key revision.
6. Run an isolated create/restart/retrieve/delete cycle before admitting
   traffic.

Never place a DSN or password directly in a command, unit file, manifest, or
shell history. Use the platform's secret injection and verify the resolved
database host/name without printing credentials.

### N/N+1 compatibility

Each checkpoint, execution reference, outbox record, idempotency record,
envelope, and tombstone has an explicit readable/writable revision. During a
rolling deployment, N and N+1 readers must accept the planned window while only
the selected writer revision emits new state. Visible transcript reconstruction
is never a fallback for an unreadable opaque revision.

Before advancing the writer:

- prove old and new readers can read retained N state;
- prove the new reader can read N+1 state;
- define whether rollback leaves N+1 state resolvable or deterministically
  unavailable;
- keep delete permitted for old state even if continuation is unavailable;
- preserve exact execution-definition and provider-binding resolvers for every
  unexpired checkpoint.

## Keys

Readiness requires exactly one active checkpoint write key and one active
envelope write key for each authority scope. Older revisions may be read-only
only within the documented resolver window.

Rotate by provisioning, dual-reading, switching writes, rewrapping or expiring,
and then retiring. Monitor rewrap work independently from response traffic.
Never delete an old key while unexpired state, backups, outbox work, or rollback
snapshots require it.

On compromise, stop dispatch for affected revisions, mark the key compromised,
rotate unaffected writes, revoke sessions, tombstone affected public mappings,
and follow the incident sequence in
[Conversation security](CONVERSATION_SECURITY.md). Compromised keys are not
available for a compatibility read window.

## Retention and sweeping

Set local checkpoint TTL, envelope TTL, and known upstream TTL independently.
The effective lifetime is the minimum applicable value. Document the reason
for every TTL and the legal/business owner.

Run the sweeper continuously with bounded batches. The first pass transitions
expired active state into its cleanup lifecycle; physical deletion happens only
when references, leases, idempotency fences, named heads, outbox work, and
retention policy permit it. Track content-free counts and lag, not state values.

Pause activation or reduce admission when:

- sweeper health is not running or reports a failure;
- outbox lag exceeds the configured maximum;
- capacity limits reject new cleanup work;
- storage schema/readiness differs across replicas;
- key rewrap cannot keep pace with expiry or retirement.

## Outbox and reconciliation

Provider publication, deletion, rewrap, and ambiguous dispatch are effects
outside the local database transaction. The durable outbox makes their intent
recoverable without claiming the effect already happened.

Workers must claim with an owner lease, execute under bounded timeout, and
acknowledge only after authoritative success. A lost lease is not success.
Release transient failures for bounded retry; quarantine permanent binding,
authorization, or integrity failures for operator review.

For ambiguous provider dispatch:

1. Fence the idempotency record and block automatic redispatch.
2. Resolve through the exact historical provider adapter and binding.
3. Record `confirmed_not_dispatched`, `confirmed_dispatched`, or the remaining
   ambiguous disposition.
4. Commit or release the local result exactly once.
5. Keep the public response unavailable until settlement is authoritative.

For deletion, tombstone locally first. If upstream deletion is unavailable,
return a content-safe pending-reconciliation status and keep the resource
locally inaccessible. Do not restore it to make retries easier.

## Provider capability rollout and retirement

An activation row is an exact cross-product, not a provider-wide switch. Pin:

- native provider family and normalized endpoint;
- endpoint/API form and provider API revision;
- installed and minimum/maximum supported SDK versions;
- model or Azure deployment and its configuration revision;
- streaming or non-streaming transport;
- stored or stateless mode;
- requested/effective reasoning context;
- none, inline, or standalone compaction;
- retrieve/delete support;
- deterministic wire, public E2E, current documentation, authorized live, and
  independent review proof IDs.

Generic-compatible providers remain incapable unless a separate exact native
profile and evidence set is reviewed. Never widen one proven row into an
untested cross-product.

To retire a row, stop advertising and resolving it for new requests first.
Keep the historical resolver and deletion capability only for manifests that
were actually active and whose compatibility evidence remains unexpired.
Loading a candidate never creates historical authority. After the compatibility
window and reconciliation queues reach zero, retire the resolver and its keys.
Record an explicit reset path for users whose old state is no longer
continuable.

## Provider and SDK drift

Re-run evidence inspection on every SDK, provider API, endpoint form, model/
deployment revision, or relevant documentation change. Diff typed create,
stream, compact, retrieve, delete, item, reasoning, and context-management
signatures. Untyped request bodies and `extra_body` are not substitutes for a
missing minimum-supported typed field.

If drift is detected:

1. Mark the affected rows incapable before dispatch.
2. Preserve historical resolvers for already committed state when safe.
3. Run deterministic drift fixtures and the opt-in redacted live matrix.
4. Obtain architecture, security, and test/traceability review.
5. Publish a new immutable manifest revision; never mutate the active content
   in place.

## Backup and restore

Back up encrypted database state, schema metadata, activation snapshots, and
content-safe key IDs/revisions. Back up key material separately through the
secret-management recovery process. Ensure the restore operator needs both
authorized channels.

Restore into an isolated network first. Before traffic:

1. Verify backup integrity and expected schema/application versions.
2. Restore the matching key revisions without logging key material.
3. Keep provider dispatch disabled.
4. Run checkpoint, tombstone, idempotency, named-head, outbox, expiry, and
   authority reconciliation.
5. Prove deleted state stays unavailable and pending deletion resumes.
6. Prove N/N+1 readers and historical provider resolvers cover all unexpired
   state.
7. Compare the loaded activation digest with the expected reviewed digest.
8. Rehearse rollback, then enable only the exact reviewed rows.

A database restore must not roll public state behind an already completed
provider effect without reconciliation. Treat that mismatch as ambiguous, not
as permission to repeat the effect.

## Atomic activation

Activation uses `AsyncActivationRegistry`. Loading a reviewed manifest is
dormant. `apply` uses the registry generation as a compare-and-swap token and
switches the complete manifest atomically. Partial field or endpoint rollout is
not supported.

### Preconditions

- The candidate manifest has an active row for every intended cross-product
  and no other row.
- Every row has deterministic wire/public E2E/current-doc/authorized-live proof
  and unexpired evidence.
- Runtime SDK and exact bindings match the manifest.
- The deployment pins each reviewed manifest content digest and all required
  reviews are complete. The `review_signature` field is that content digest;
  it is not authenticated signer proof.
- Storage migration, keys, workers, outbox lag, resolver coverage, and loaded
  manifest digest are ready.
- A prior activation snapshot and rollback operator are available.

The current tracked fixture fails the activation preconditions by design. It
is an inactive pending-review decision with no production rows and one exact
Azure live-proof ID. Its linked result set records six evaluated Azure
candidates and one authorized native OpenAI candidate. The exact
`gpt-5.6-terra` revision `2026-07-09` completed all eight cases in 11 logical
operations and 11 HTTP requests, with SDK retries configured to zero, zero
observed retries, exact content-free request-category counts, and completed
protected cleanup and client close. The tracked harness generates those counts
from actual operation boundaries and an HTTP request hook, binds them into the
structural digest, and rejects mismatches, unknown paths, retries, or cleanup
drift. The harness and traceability delta used for that run is pending
independent review. The exact native OpenAI
`gpt-5.6-sol` model identity was retrievable, but two store-free generation
attempts were rejected before inference by account credit quota. No project or
organization selector was configured and the safe 429 metadata identified no
project or account. The cross-provider matrix therefore remains incomplete
with zero active rows. The review signature seals a no-activation decision
rather than authorizing an empty matrix by vacuity; it is a content digest, not
authenticated signer proof.

### Rehearsal commands

Run these from the repository root in an isolated test environment:

```bash
poetry run pytest -q tests/conversation/activation_test.py::test_registry_starts_dormant_and_load_does_not_activate
poetry run pytest -q tests/conversation/activation_test.py::test_generation_cas_makes_concurrent_apply_atomic
poetry run pytest -q tests/conversation/activation_test.py::test_rollback_restores_prior_manifest_or_dormant_state
poetry run pytest -q tests/conversation/security_e2e_test.py::test_phase11_pgsql_migration_restart_and_rollback
```

The first proves load is dormant, the second proves one compare-and-swap
winner, the third restores an exact prior snapshot or dormant state, and the
fourth proves durable N/N+1 restart and rollback behavior.

### Apply procedure

1. Record the pre-change `ActivationSnapshot` and generation.
2. Validate and load the immutable candidate. Confirm loading did not advertise
   or dispatch it.
3. Compare health and loaded digests on every replica.
4. Apply the candidate digest with the expected generation once.
5. Confirm all replicas observe the new complete manifest and no partial row.
6. Run pre-dispatch negative probes, then deterministic acceptance. Run hosted
   probes only with separate explicit authority, credentials, and cost
   acknowledgement.
7. Monitor conflict, drift, dispatch, commit, deletion, outbox, sweeper, and
   resolver metrics using content-safe labels.

New provider dispatch linearizes when `resolve` returns the active evidence row
under the registry lock. Revocation prevents later resolutions; a call that
already resolved may complete. The registry does not drain or cancel in-flight
SDK work, so operators must separately drain callers when that stronger
incident semantic is required.

## Rollback

The release operator triggers rollback for integrity or binding drift, SDK/
provider incompatibility, unauthorized capability advertisement, sustained
commit/publication failures, unresolved ambiguous dispatch growth, deletion or
outbox backlog above policy, key compromise, or readiness disagreement across
replicas.

Rollback is an atomic registry compare-and-swap to the recorded prior snapshot
or dormant state. It stops new dispatch; it does not erase committed state.
Keep N/N+1 readers and historical provider resolvers for the published
compatibility window.

After rollback:

- already committed public responses remain retrievable only when their policy,
  authority, migration revision, and historical resolver allow it;
- continuation either resolves through the documented compatibility window or
  fails with `conversation_migration_required`/
  `conversation_capability_unsupported` before dispatch;
- visible transcript replay never replaces unreadable opaque state;
- delete remains accepted, tombstones locally first, and reconciles upstream
  through the historical resolver;
- caller-held envelopes from unsupported revisions are rejected; users receive
  an explicit reset path rather than implicit state loss;
- outbox, sweeper, rewrap, and deletion workers continue unless the incident
  requires a separately documented stop.

Do not delete old manifests, resolvers, database revisions, or keys until the
post-rollback compatibility window, backups, and all reconciliation queues are
closed.

## Deletion outage checklist

1. Confirm local tombstones are committing and retrieval is denied.
2. Confirm the deletion outbox is durable and leases are progressing.
3. Reduce or stop admission before bounded capacity is exhausted.
4. Verify retries use the exact historical binding and do not reveal upstream
   IDs.
5. Escalate to the provider owner without copying request state or envelopes.
6. Reconcile, acknowledge, sweep, and verify source state remains unavailable.
7. Record content-safe counts, duration, affected profile revisions, and final
   disposition.

The deterministic retry behavior is exercised with:

```bash
poetry run pytest -q tests/conversation/phase6_lifecycle_test.py::test_reconciler_retries_failure_without_restoring_local_access
poetry run pytest -q tests/conversation/pgsql_conformance_test.py::test_pgsql_atomic_idempotency_and_outbox_recovery
```

See [Conversation continuity](CONVERSATIONS.md) for the public API and
[Conversation security](CONVERSATION_SECURITY.md) for envelope and incident
handling.
