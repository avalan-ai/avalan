# Patch tools

Patch.edit and patch.apply are two-phase semantic mutation tools. They are not
a shell wrapper, a Git wrapper, or a general workspace-write capability. The
production-source activation manifest records a capability inventory, not an
advertisement. It is deliberately incomplete, so no profile advertises a patch
tool or receives authority. An operator needs a verifier-issued receipt bound
to the exact durable owner, fence, and epoch before any profile can be used.

This page describes the Version 1 contract and operational boundaries. It does
not grant authority, turn on a context, or substitute for target, policy, or
approval checks.

## Selection is not authority

Selecting patch.edit, patch.apply, or patch.* chooses a candidate tool name. It
does not grant mutation capability. patch.* selection is independent of shell.*
in both directions, and there are no shell aliases for patch.

Before a selected tool can appear, the exact context must prove containment,
filesystem metadata handling, a target protocol, coordination and fencing,
policy, a plan-review approval path, persistence, a surface, a provider codec,
and a version. The requested operation then needs its own capabilities:
update, create, delete, move, read_for_mutation, and
observe_mutation_preconditions are separately checked. Approval is a third
gate: a trusted broker grants approval for one complete sealed plan; generic
"approve all" and raw-tool confirmation never approve a patch.

The model cannot set a workspace, current directory, backend, capabilities,
approval, overwrite behavior, policy, widening limits, disclosure, validation,
or a container profile. A denied or incomplete profile is incapable and is not
advertised; no local, host, or provider fallback is allowed.

## Choosing a tool

Use patch.edit for small or medium exact changes in one existing text file. It
cannot create, delete, move, change final-newline state, or change metadata.
Use patch.apply for a multi-file change, a create/delete/move, a move with an
update, or an explicit final-newline transition. If an exact match fails,
re-read or search the file and make a new exact request. Do not enlarge the
matching rule, guess surrounding text, switch to fuzzy matching, or retry the
same stale plan.

## Exact Version 1 schemas

Both JSON function schemas are closed: each object has
"additionalProperties": false, and duplicate object member names are rejected
while decoding.

The patch.edit parameter schema is:

~~~json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": {"type": "string"},
    "edits": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "old_text": {"type": "string", "minLength": 1},
          "new_text": {"type": "string"}
        },
        "required": ["old_text", "new_text"]
      }
    }
  },
  "required": ["path", "edits"]
}
~~~

The patch.apply parameter schema is:

~~~json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {"patch": {"type": "string"}},
  "required": ["patch"]
}
~~~

The terminal model-result schema is closed. It has exactly kind (always
"patch_result"), status, mutation_state, lineage_state,
requested_effect_occurred, artifact_state, commit_set_exact,
workspace_changed, postcondition, lifecycle (always "request_completed"), and
code (a string or null). A pending operation is not a terminal tool result.

### Executable request examples

The following canonical JSON inputs are exercised by the tracked Patch
documentation test through the Version 1 parser.

<!-- patch-example: valid-edit -->
~~~json
{"path":"notes/today.txt","edits":[{"old_text":"draft\n","new_text":"final\n"}]}
~~~

<!-- patch-example: invalid-edit-empty-old-text -->
~~~json
{"path":"notes/today.txt","edits":[{"old_text":"","new_text":"final\n"}]}
~~~

The first input is valid. The second is invalid because old_text is empty. An
object with an extra workspace member, an empty edits array, a non-string value,
or duplicate member names is invalid as well.

<!-- patch-example: valid-apply -->
~~~json
{"patch":"*** Begin Patch v1\n*** Update File: notes/today.txt\n@@\n-draft\n+final\n*** End Patch"}
~~~

<!-- patch-example: invalid-apply-version -->
~~~json
{"patch":"*** Begin Patch\n*** Update File: notes/today.txt\n@@\n-draft\n+final\n*** End Patch"}
~~~

The valid document starts with exactly "*** Begin Patch v1", has one or more
file declarations, and ends with exactly "*** End Patch". The invalid document
omits the Version 1 marker and must be rejected before planning or mutation.

## Patch.apply Version 1 language

The document uses uniform LF or uniform CRLF records. There are no blank
records, comments, indentation, trailing spaces on controls, quoted paths, or
escape forms. The only control records are:

~~~text
*** Begin Patch v1
*** Add File: PATH
*** Update File: PATH
*** Move to: PATH
*** Delete File: PATH
@@
@@ LABEL
 CONTEXT
-REMOVE
+ADD
\ No newline at end of file
*** End of File
*** End Patch
~~~

Add creates a regular text file at an absent destination with an existing
parent. Update applies exact contextual hunks. Delete removes an existing
regular text file. Move has an absent destination and may include an update.
Evaluation is ordered over a virtual workspace, but all hunks in an update
resolve against the same input snapshot and apply simultaneously. Conflicting
chains, cycles, duplicate producers, destination collisions, overwrite, missing
parents, directory operations, cross-filesystem moves, and no-effect plans are
rejected before mutation.

## Paths, text, and metadata

Paths are relative logical slash-separated paths. Absolute, drive, UNC, home,
URI, environment-expanded, alternate-stream, traversal, repeated-separator,
empty-component, overlong, control, terminal-escape, and bidirectional-control
paths are rejected. Links, hardlinks, reparse points, devices, FIFOs, sockets,
directories, special files, and unauthorized mount transitions are rejected.
Targets use rooted no-follow operations rather than a path-string containment
check.

Patch text and source files must be strict UTF-8, optionally with one leading
UTF-8 BOM for an existing file. NUL, unpaired surrogates, bare CR, mixed
newlines, invalid UTF-8, and unsupported representations are rejected. Text
uses uniform LF, uniform CRLF, or no line separator. Unchanged bytes, BOM,
final-newline state, mode, and every required security-relevant metadata field
must be preserved; a target that cannot prove preservation is incapable.

Hidden and sensitive paths, including repository internals, require explicit
policy and disclosure authority. Denial occurs before existence, type, content,
hash, size, or match disclosure whenever possible. The tool never follows a
link or implicitly creates a parent directory.

Matching is exact only: no regex, fuzzy, similarity, AST, whitespace-folded,
case-folded, or Unicode-normalized matching exists. A patch.edit replacement
must occur exactly once in the original snapshot; missing, ambiguous, duplicate,
nested, or overlapping ranges reject the whole call.

## Contexts and container limits

Local, sandbox, and container are distinct context selections. A patch runs and
is later read only in its selected context's filesystem view. Every active
profile needs the same semantic contract; an unsupported or incomplete context
does not degrade to another context.

Container patching needs a narrow, context-owned mutation authority. It does
not make the ordinary container workspace writable and does not make shell or
code execution a writer. A read-only container workspace stays incapable until
there is a conforming narrow target with containment, target-native identity,
metadata preservation, durable fencing, and a persistent mutation lease. Host
path fallback, host-path disclosure, ambient network, and ambient secrets are
not substitutes.

## Review, approval, and outcomes

Planning creates an immutable complete operation manifest and deterministic full
diff before any workspace write. The privileged approver sees the complete
untruncated resolved manifest and diff. Model, SDK, server, display, event,
audit, and telemetry outputs are independently redacted and may explicitly omit
diff content; a truncated or hidden-tail view never authorizes commit.

The structured terminal result distinguishes committed, partial, indeterminate,
stale, denied, and diagnostic outcomes through its status, stage/code, mutation
state, requested-effect occurrence, artifact state, workspace-change fact,
postcondition, and retryability. Matching exactness and commit-set exactness
are distinct. A diagnostic is an association with a completed patch, not
evidence that the patch succeeded or failed.

Precommit cancellation, denial, timeout, parse, policy, match, and stale-plan
failures make zero workspace writes. After commit_started, cancellation only
records the signal: the journal must settle to one terminal result, or the
operation remains settlement_pending until the original worker settles or is
provably fenced. Partial and indeterminate outcomes are never automatically
reapplied, rolled back, or treated as success.

## Pending, retry, and idempotency

PatchPending is a nonterminal host envelope, not mutation truth or a model tool
result. It carries a schema version, pending operation ID, request ID,
correlation ID, and settlement_pending lifecycle. The original model turn, flow
branch, dependent task, and later mutation stay suspended. Only the same
authenticated principal, execution scope, route, and request identity may
inspect or await/resume it; delivery of the one terminal result uses the
original correlation.

Every retryable route reserves an authenticated retransmission key and
canonical request digest before planning. Reusing the same key and digest
returns or reconciles recorded truth; reusing it with a different digest fails.
It does not run a second mutation. A transport timeout during commit is
reconciled, never blindly retried. Staging artifacts are target-owned and
journaled independently from requested effects. Foreign writers are outside
the patch coordinator boundary; revalidation detects defined before-state
changes and produces stale or journal-derived truth rather than automatic
rebase.

## Configuration and operations

An operator configuration is a reviewed, atomic profile key containing exactly
the context, platform, filesystem, target implementation, target protocol,
policy, approval broker, persistence, surface, provider codec, and Version 1
identifier. Each component is pinned; changing any component produces another
profile rather than broadening an existing one. The production-source manifest
generates and freeze-checks the public schemas, tool inventory, protocol
descriptions, and capability profiles from tracked source symbols, never from
ignored specifications.

The kill switch is safe deactivation of an exact active profile. It immediately
prevents new advertisements and new operation bindings. It does not revoke,
rewrite, downgrade, or transfer the owner, epoch, lease, journal, or
reconciliation responsibility of an in-flight, partial, or pending operation.
Those operations retain their original profile and reconcile normally.

The approval broker is trusted and plan-bound. Data retention is minimal, bounded,
audience-controlled, encrypted when persisted, and cleaned by policy; no raw
patch, source content, diff, temporary path, approval grant, or credential
belongs in generic public events. Pending reconciliation is owned by the
durable coordinator and target evidence, not by a request retry. Capability
diagnostics are privacy-safe and state only that a profile is unavailable; they
do not reveal hidden paths or missing authority.

Platform receipts are profile-specific. macOS Seatbelt, Linux Bubblewrap, local
filesystem, and container claims must each be independently proven; a receipt
for one does not activate another. During an incident: deactivate the exact
profile, preserve the durable journal and fences, stop new work, and use the
owner-bound reconciliation path. Do not issue recovery writes, automatic
rollback, automatic rebase, or broad retries under patch authority.

## Explicit non-features

Patch does not compose shell commands or participate in shell pipelines. It
does not run Git commands, stage or commit repositories, call formatters, tests,
interpreters, language servers, hooks, plugins, or workspace code. Diagnostic
execution is separate: it needs independent trusted execution authority and
approval, and it cannot change patch mutation truth.

Patch provides no binary/non-UTF-8 changes, directory operations, implicit
parents, link mutation, overwrite, requested owner/ACL/xattr changes, general
executable-file mutation, model-selected backend/policy/limits, ACID multi-file
transactions, generic rollback, undo, automatic rebase, automatic retry, or
foreign-writer no-lost-update guarantee. Git, shell composition, validation,
rollback, and undo are intentionally outside this feature.
