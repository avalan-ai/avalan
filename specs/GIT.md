# Avalan Git Shell Wrapper Specification

**Draft:** v0.2
**Date:** 2026-07-03
**Status:** Phase 0 contract locked
**Audience:** Avalan maintainers, shell tool implementers, agent authors,
CLI users, flow authors, and runtime operators

---

## 1. Purpose

Add Git command wrappers inside the existing shell toolset. The public
contract is shell-owned: tools use flat names such as `shell.git_status`,
and settings live under `[tool.shell.git]`.

The wrappers follow the existing shell execution profile:

```text
typed shell Git tool arguments
-> ShellGitCommandRequest
-> Git-aware shell policy normalization
-> ExecutionSpec
-> CommandExecutor.execute(...)
-> ExecutionResult
-> shell Git formatter/display projection
```

No public shell Git tool may accept a raw Git command, raw argument list,
Git aliases, shell snippets, passthrough subcommands, or user-selected
global options. Policy owns final argument-vector construction.

The default useful profile is read-only. "Read-only" means the wrapper does
not intentionally mutate repository state. It does not mean output is safe
to disclose: Git history, patches, stashes, author metadata, file names,
remote names, token-bearing URLs, and deleted content may contain secrets.

---

## 2. Locked Public Contract

### 2.1 Namespace

Git commands live in the existing shell toolset. Public tool names are flat
`shell.git_*` names, for example `shell.git_status` and `shell.git_diff`.

The implementation must not add a separate public Git namespace, a nested
`shell.git.*` namespace, or a generic shell Git dispatcher. Public examples,
agent TOML, CLI docs, flow docs, formatter docs, and tests must use the
flat `shell.git_*` names.

### 2.2 Settings

All Git-specific settings live under `[tool.shell.git]`. They may be
represented internally by a shell-owned Git settings object or by typed
fields on `ShellToolSettings`, but the public configuration path is fixed.

### 2.3 Default Posture

Git tools are absent unless explicitly enabled through the normal tool
enablement mechanisms. Enabling the shell namespace with `shell` or
`shell.*` may expose only shell Git tools authorized by both Git capability
settings and Git command allowlists.

When Git is enabled and no mutation capability is configured, the effective
Git capability set is `["read"]`. This exposes read-only behavior only for
explicitly enabled read tools. `worktree`, `history`, and `remote` are never
implied by `read`, by each other, or by shell wildcard enablement.

### 2.4 Public Configuration Examples

Read-only profile:

```toml
[tool]
enable = [
  "shell.git_status",
  "shell.git_diff",
  "shell.git_log",
]

[tool.shell.git]
workspace_root = "."
cwd = "."
capabilities = ["read"]
allowed_commands = [
  "status",
  "diff",
  "show",
  "log",
  "ls-files",
  "rev-parse",
  "branch",
  "tag",
  "describe",
  "blame",
  "grep",
  "stash-list",
  "stash-show",
]
default_timeout_seconds = 10.0
max_timeout_seconds = 60.0
max_stdout_bytes = 65536
max_stderr_bytes = 32768
max_diff_bytes = 131072
max_log_count = 50
max_grep_matches = 1000
max_pathspecs = 64
allow_external_diff = false
allow_textconv = false
allow_optional_locks = false
allow_submodules = false
allow_bare_repositories = false
allow_linked_worktrees = false
allow_alternates = false
allow_remote_credentials = false
allow_submodule_update = false
allowed_remote_protocols = ["https"]
allowed_remote_hosts = []
redact_remote_urls = true
```

Remote-enabled profile:

```toml
[tool]
enable = [
  "shell.git_fetch",
  "shell.git_push",
]

[tool.shell.git]
capabilities = ["read", "remote"]
allowed_commands = ["status", "fetch", "push", "remote"]
allowed_remote_protocols = ["https"]
allowed_remote_hosts = ["github.com"]
allow_remote_credentials = true
redact_remote_urls = true
```

---

## 3. Capability Model

The capability vocabulary is fixed.

| Capability | Default | Scope |
| --- | --- | --- |
| `read` | Enabled when Git is explicitly enabled | Non-network repository inspection. |
| `worktree` | Disabled | Local working-tree and index mutations. |
| `history` | Disabled | Local commits, refs, history rewrites, and destructive repository mutations. |
| `remote` | Disabled | Network, credentials, remote management, clone, fetch, pull, push, and submodule update. |

Tool exposure and execution require both public tool enablement and the
capability required by the command or mode. `allowed_commands` is an
additional allowlist and cannot grant capabilities.

`remote` is a separate boundary. It is never implied by `worktree` or
`history`, and `remote` alone cannot permit local worktree or history
mutation.

---

## 4. Command Surface

Default exposure values in the table mean:

- `Read if enabled`: visible and executable only after tool enablement, with
  the default `read` capability and matching `allowed_commands`.
- `Capability gated`: absent and denied unless the named non-read
  capability, command allowlist, and concrete tool enablement all match.
- `Remote-policy gated`: absent and denied unless `remote` capability,
  remote command allowlist, protocol allowlist, host allowlist, credential
  policy, and concrete tool enablement all match.

| Tool | Git subcommand or mode | Capability | Default exposure | Phase | Test area |
| --- | --- | --- | --- | --- | --- |
| `shell.git_status` | status metadata | `read` | Read if enabled | 3 | Status policy, optional-lock tests, smoke tests |
| `shell.git_rev_parse` | approved repository and HEAD facts | `read` | Read if enabled | 3 | Revision policy, fact enum schema, boundary tests |
| `shell.git_branch` | current/list branches only | `read` | Read if enabled | 3 | Branch read schema, mutation denial, formatter tests |
| `shell.git_tag` | list/show tags only | `read` | Read if enabled | 3 | Tag read schema, mutation denial, revision tests |
| `shell.git_describe` | bounded describe forms | `read` | Read if enabled | 3 | Ref validation, limit tests, smoke tests |
| `shell.git_ls_files` | tracked/safe enum listing modes | `read` | Read if enabled | 3 | Pathspec policy, output caps, smoke tests |
| `shell.git_log` | bounded fixed-format history | `read` | Read if enabled | 3 | Count caps, format denial, pathspec tests |
| `shell.git_diff` | worktree/staged/range/stat/name-only | `read` | Read if enabled | 4 | External-process denial, output caps, pathspec tests |
| `shell.git_show` | commit or tag summaries and bounded details | `read` | Read if enabled | 4 | Object-read denial, revision policy, output caps |
| `shell.git_blame` | one repo-relative file, bounded line range | `read` | Read if enabled | 4 | File path policy, range caps, textconv denial |
| `shell.git_grep` | bounded repository search | `read` | Read if enabled | 4 | Pattern policy, match caps, pager/network denial |
| `shell.git_stash_list` | bounded stash listing | `read` | Read if enabled | 4 | Stash listing caps, reflog denial |
| `shell.git_stash_show` | bounded stash stat/patch display | `read` | Read if enabled | 4 | External-process denial, output caps |
| `shell.git_add` | add repo-relative paths | `worktree` | Capability gated | 5 | Path mutation policy, dry-run/preview where supported, audit tests |
| `shell.git_restore` | constrained restore forms | `worktree` | Capability gated | 5 | Path/ref validation, mutation audit |
| `shell.git_checkout` | constrained path or branch checkout forms | `worktree` | Capability gated | 5 | Ref/path disambiguation, unsafe checkout denial |
| `shell.git_switch` | constrained branch switching | `worktree` | Capability gated | 5 | Ref validation, branch-mode schema |
| `shell.git_reset` | path/index reset modes | `worktree` | Capability gated | 5 | Mode gating, path validation, audit tests |
| `shell.git_rm` | remove repo-relative paths | `worktree` | Capability gated | 5 | Path mutation policy, outside-worktree denial |
| `shell.git_mv` | move repo-relative paths | `worktree` | Capability gated | 5 | Source/destination policy, overwrite denial |
| `shell.git_stash_push` | bounded stash creation | `worktree` | Capability gated | 5 | Message/path policy, audit tests |
| `shell.git_stash_apply` | bounded stash apply | `worktree` | Capability gated | 5 | Stash ref validation, conflict/audit tests |
| `shell.git_commit` | commit from existing index | `history` | Capability gated | 6 | Message policy, editor/signing/hook denial, audit tests |
| `shell.git_branch_create` | create branch | `history` | Capability gated | 6 | Ref creation schema, unsafe ref denial |
| `shell.git_branch_delete` | delete branch | `history` | Capability gated | 6 | Destructive confirmation policy, audit tests |
| `shell.git_branch_rename` | rename branch | `history` | Capability gated | 6 | Ref validation, collision denial |
| `shell.git_tag_create` | create tag | `history` | Capability gated | 6 | Tag/ref policy, signing denial |
| `shell.git_tag_delete` | delete tag | `history` | Capability gated | 6 | Destructive confirmation policy, audit tests |
| `shell.git_merge` | constrained merge | `history` | Capability gated | 6 | Merge-tool/hook denial, conflict reporting |
| `shell.git_rebase` | constrained rebase | `history` | Capability gated | 6 | Exec/editor denial, destructive-mode tests |
| `shell.git_cherry_pick` | constrained cherry-pick | `history` | Capability gated | 6 | Revision validation, conflict reporting |
| `shell.git_revert` | constrained revert | `history` | Capability gated | 6 | Revision validation, audit tests |
| `shell.git_reset` | ref-moving reset modes | `history` | Capability gated | 6 | Hard/destructive gating, audit tests |
| `shell.git_clean` | constrained clean | `history` | Capability gated | 6 | Dry-run policy, destructive confirmation |
| `shell.git_stash_pop` | bounded stash pop | `history` | Capability gated | 6 | Stash ref validation, destructive audit |
| `shell.git_stash_drop` | bounded stash drop | `history` | Capability gated | 6 | Stash ref validation, destructive audit |
| `shell.git_fetch` | bounded fetch | `remote` | Remote-policy gated | 7 | Protocol/host/refspec/credential tests |
| `shell.git_pull` | bounded pull | `remote` | Remote-policy gated | 7 | Remote plus local mutation gating, conflict tests |
| `shell.git_push` | bounded push | `remote` | Remote-policy gated | 7 | Force/mirror denial, credential redaction, audit tests |
| `shell.git_clone` | clone into approved workspace path | `remote` | Remote-policy gated | 7 | Destination policy, timeout cleanup, network audit |
| `shell.git_remote_list` | approved remote entry listing | `remote` | Remote-policy gated | 7 | URL redaction, config scope tests |
| `shell.git_remote_add` | add approved remote entry | `remote` | Remote-policy gated | 7 | URL validation, credential denial, audit tests |
| `shell.git_remote_set_url` | update approved remote URL | `remote` | Remote-policy gated | 7 | URL validation, redaction, audit tests |
| `shell.git_remote_remove` | remove approved remote entry | `remote` | Remote-policy gated | 7 | Name validation, audit tests |
| `shell.git_remote_rename` | rename approved remote entry | `remote` | Remote-policy gated | 7 | Name validation, collision denial |
| `shell.git_submodule_update` | gated submodule update | `remote` | Remote-policy gated | 7 | Submodule flag, recursion denial, protocol/host tests |

---

## 5. Threat Model

The repository is hostile input. The following may be attacker-controlled:

- Worktree files and file names.
- Repository configuration, attributes, hooks, filters, textconv drivers,
  fsmonitor settings, and pager/diff settings.
- Refs, tags, branches, stash entries, commit messages, author metadata, and
  object content.
- Submodules, linked worktrees, repository indirection, common directories,
  alternates, and nested repositories.
- Command output, including paths, remotes, patch contents, historical
  secrets, token-bearing URLs, and diagnostics.

Security requirements:

- Deny by default.
- Fail closed for unknown commands, unknown modes, unknown options,
  abbreviated options, unsupported ref forms, unsupported pathspecs, and
  unsupported capability transitions.
- Deny user-controlled Git global options and configuration overrides.
- Ensure Git aliases cannot influence command dispatch.
- Disable interactive behavior.
- Keep policy diagnostics and logs from leaking environment values,
  credential material, token-bearing remotes, or full sensitive command
  output.

---

## 6. Repository And Path Policy

The wrapper must resolve an effective `cwd` inside `workspace_root`, discover
the repository from there, and require the resolved repository root to remain
inside `workspace_root`. It must not silently climb to a repository outside
the workspace.

Default repository restrictions:

- Reject non-repositories with `repo_not_found`.
- Reject bare repositories with `bare_repo_denied`.
- Reject repository indirection or common directories that resolve outside
  the workspace with `repo_boundary_denied`.
- Reject alternates outside the workspace with `alternate_denied`.
- Reject submodule recursion by default with `submodule_denied`.
- Reject linked worktrees by default with `repo_boundary_denied` unless a
  future trusted setting proves every common directory and worktree path
  stays inside allowed roots.

Pathspec requirements:

- Accept repo-relative paths only.
- Reject absolute paths, parent traversal, NUL bytes, control characters,
  pathspec magic, expansion forms, and option-looking pathspecs unless the
  command proves they are safely passed as path data after policy separators.
- Keep hidden and sensitive path denylist behavior aligned with the shell
  tool policy unless a trusted setting explicitly narrows or widens it.
- Validate refs and revisions separately from paths. Object path syntax must
  not become a hidden file-read or object-read API.

For mutating commands, path validation must be repeated close to execution
and must avoid following symlinks outside the worktree for wrapper-side
filesystem checks.

---

## 7. Environment And Process Policy

Git execution must use the existing shell executor profile or a Git-specific
wrapper around it:

- Async subprocess execution through `CommandExecutor`.
- No synchronous subprocess APIs in runtime code.
- No shell command evaluation.
- Stdin disabled unless a future typed command explicitly needs bounded
  input.
- Bounded stdout, stderr, runtime, command length, argument count, and
  argument bytes.
- Deterministic display argv with secret-aware redaction.

The child environment must be scrubbed by default. Remove or override
untrusted values including:

- Unowned `GIT_*` variables.
- `HOME` and `XDG_*`.
- Pagers and editor variables.
- External diff variables.
- SSH, askpass, credential-helper, and prompt-related variables.

Set safe wrapper-owned values where supported:

- Disable optional locks for read-only commands.
- Disable terminal prompts.
- Disable pagers, color, editors, and prompts.
- Use isolated `HOME` and XDG directories.
- Disable system/global configuration or point it at controlled empty
  configuration.
- Use a deterministic locale.

Repository-local configuration is still hostile. Commands that can trigger
external behavior must force safe flags or be denied. Output-producing
commands such as `shell.git_diff`, `shell.git_show`, `shell.git_log`,
`shell.git_stash_show`, and `shell.git_blame` must deny external diff,
textconv, output files, pager-opening modes, binary patch output, and custom
formats unless a future phase explicitly supports a safe typed form with
tests.

For mutation, hooks, filters, signing, merge tools, rebase exec hooks,
editors, and prompts must be disabled or the command must remain out of
scope.

---

## 8. Denial Matrix

| Input or threat | Default decision | Stable error code | Phase | Test area |
| --- | --- | --- | --- | --- |
| Unknown command or unsupported internal subcommand | Deny before execution | `command_disabled` | 1, 2 | Tool registration and policy dispatch |
| Disabled command in `allowed_commands` | Deny before exposure and execution | `command_disabled` | 1 | Settings and enablement |
| Unknown option, unsupported mode, or unsupported value | Deny before execution | `invalid_option` | 2-7 | Per-command schema and policy |
| Abbreviated or ambiguous option | Deny before execution | `invalid_option` | 2-7 | Option validation |
| User-controlled global option or configuration override | Deny before execution | `invalid_option` | 2-7 | Escape-hatch denial |
| Unsafe ref or revision form | Deny before execution | `revision_denied` | 2-7 | Ref/revision validation |
| Missing revision | Return stable not-found result | `revision_not_found` | 2-7 | Revision lookup |
| Ambiguous revision | Return stable ambiguity result | `ambiguous_revision` | 2-7 | Revision lookup |
| Unsafe path or pathspec | Deny before execution | `pathspec_denied` | 2-7 | Path validation |
| Disabled capability | Deny before exposure and execution | `capability_required` | 1-7 | Capability matrix |
| Non-repository effective cwd | Return stable repository failure | `repo_not_found` | 2 | Repository discovery |
| Repository boundary escapes workspace | Deny before execution | `repo_boundary_denied` | 2 | Repository boundary |
| Bare repository | Deny by default | `bare_repo_denied` | 2 | Repository form |
| Linked worktree outside trusted policy | Deny by default | `repo_boundary_denied` | 2 | Repository form |
| Submodule recursion or ungated submodule update | Deny by default | `submodule_denied` | 2, 7 | Repository form and remote policy |
| Alternates outside workspace | Deny by default | `alternate_denied` | 2 | Repository form |
| Hostile repository configuration, attributes, filters, or fsmonitor | Deny behavior or neutralize configuration | `unsafe_git_config` | 2-7 | Hostile repo tests |
| External diff, textconv, pager, editor, hook, signing, merge tool, or prompt attempt | Deny or force safe mode | `external_process_denied` | 2, 4-7 | Process hardening |
| Optional-lock-free read command cannot be enforced | Deny read command | `optional_lock_denied` | 2, 3 | Status/read policy |
| Credential-bearing URL when credentials are disabled | Deny before network | `credential_denied` | 7 | Credential policy |
| Inherited credential helper, askpass, SSH command, or prompt behavior | Deny or scrub before execution | `credential_denied` | 2, 7 | Credential policy and environment |
| Network command without `remote` capability | Deny before exposure and execution | `capability_required` | 1, 7 | Capability matrix |
| Remote protocol not allowlisted | Deny before network | `remote_protocol_denied` | 7 | Remote policy |
| Remote host not allowlisted | Deny before network | `remote_host_denied` | 7 | Remote policy |
| Output cap reached | Return bounded result with truncation metadata | `output_truncated` | 2-7 | Output limits |
| Timeout reached | Cancel execution and report bounded result | `timeout` | 2-7 | Timeout/cancellation |
| Allowed process exits nonzero | Return stable nonzero result | `nonzero_exit` | 2-7 | Formatter and diagnostics |
| Local Git executable unavailable | Return invocation-time unavailable result | `command_unavailable` | 2 | Invocation behavior |

---

## 9. Status, Error, And Result Contract

Tool calls return formatted strings for model compatibility, backed by
structured internal results. Expected policy denials return stable formatted
policy-denied results, not raw exceptions.

Stable result `status` values:

- `success`
- `policy_denied`
- `command_unavailable`
- `failed`
- `timeout`
- `cancelled`

Stable `error_code` values:

- `capability_required`
- `command_disabled`
- `repo_not_found`
- `repo_boundary_denied`
- `bare_repo_denied`
- `submodule_denied`
- `alternate_denied`
- `pathspec_denied`
- `revision_denied`
- `revision_not_found`
- `ambiguous_revision`
- `invalid_option`
- `unsafe_git_config`
- `external_process_denied`
- `optional_lock_denied`
- `credential_denied`
- `remote_protocol_denied`
- `remote_host_denied`
- `output_truncated`
- `timeout`
- `nonzero_exit`
- `command_unavailable`

Internal result fields are fixed:

- Logical tool name.
- Git subcommand.
- Redacted display argv.
- Effective cwd.
- Resolved repo root.
- Capability required.
- Capability used.
- Execution mode: local, sandbox, or container when inherited from shell
  execution settings.
- Exit code.
- Status.
- Error code.
- Stdout snippet subject to caps.
- Stderr snippet subject to caps.
- Stdout byte count.
- Stderr byte count.
- Stdout truncation flag.
- Stderr truncation flag.
- Timeout flag.
- Cancellation flag.
- Duration.
- Audit metadata.

Audit metadata must not include raw secrets. Remote audit entries include
command type, remote host, protocol, redacted URL, selected refs, network
policy, credential mode, and whether remote state may be mutated.

Denial messages should be actionable. Example:

```text
shell.git_commit requires capability history; configured capabilities: read.
```

---

## 10. Read-Only Command Constraints

| Tool | Allowed read forms | Required denials and limits |
| --- | --- | --- |
| `shell.git_status` | Stable parse-friendly status with branch information. | Optional-lock-free mode; no refresh/write behavior; bounded untracked reporting; no submodule recursion by default. |
| `shell.git_diff` | Explicit modes such as worktree, staged, range, stat, and name-only. | External diff, textconv, output files, no-index diff, binary patch output, external drivers, and unbounded patch output. |
| `shell.git_show` | Commit or tag summaries and bounded patch/details through enum formats. | Broad blob/object reads, object path syntax, arbitrary formats, textconv, external diff, and output files. |
| `shell.git_log` | Bounded commit history with fixed output formats and required max count. | Arbitrary formats, reflog walking, unbounded counts, textconv, external diff, and pathspec magic. |
| `shell.git_ls_files` | Tracked files and optional safe enum modes for modified/deleted/untracked. | Submodule recursion, pathspec magic, and unbounded file lists. |
| `shell.git_rev_parse` | Specific safe facts only, such as current HEAD, short HEAD, current branch, and repository-root facts already proven inside the workspace. | Arbitrary rev parsing, gitdir discovery, environment inspection, path expansion, local environment variable reporting, and object database inspection. |
| `shell.git_branch` | Current branch and listing modes such as local branch names, with remote-name display only if the deployment accepts that disclosure. | Create, delete, rename, set-upstream, edit-description, and any write form. |
| `shell.git_tag` | List and show modes only. | Create, delete, sign, verify, and arbitrary object display beyond `shell.git_show` constraints. |
| `shell.git_describe` | Bounded describe forms for approved refs. | Dirty state mutation, arbitrary candidate expansion beyond configured limits, and unsafe ref forms. |
| `shell.git_blame` | Single repo-relative file with optional bounded line range. | External contents, unbounded ranges, textconv, and submodule recursion. |
| `shell.git_grep` | Bounded search, preferably fixed-string by default, over repo-relative paths. | Advanced regex engines unless typed and bounded, pager opening, no-index search, submodule recursion, pathspec magic, and unbounded matches. |
| `shell.git_stash_list` | Bounded stash listing. | Reflog-like broad traversal unless explicitly added later. |
| `shell.git_stash_show` | Bounded stash patch/stat display. | External diff, textconv, apply/pop/drop behavior, and unbounded patch output. |

---

## 11. Mutating And Remote Constraints

### 11.1 Worktree Capability

The `worktree` capability covers local working-tree and index changes.
It does not permit commits, ref changes, destructive history changes, or
network access.

Required constraints:

- No interactive patch mode by default.
- No arbitrary checkout of unsafe refs or paths.
- No hook, filter, pager, editor, prompt, signing, or network behavior.
- Strong pathspec validation and repository-boundary checks.
- Prefer preview or dry-run fields where Git supports them.
- Stable audit records for every attempted mutation.

### 11.2 History Capability

The `history` capability covers local history, ref, and destructive
repository changes. It requires a stronger opt-in than `worktree`.

Required constraints:

- Deny interactive flags, editors, prompts, custom merge/diff tools,
  signing, hooks, and rebase exec behavior by default.
- Destructive operations such as hard reset, clean, branch/tag deletion,
  rebase, merge, and stash pop/drop need explicit command-level support,
  focused tests, and preferably dry-run or preview behavior.
- Use existing Avalan confirmation or human-review mechanisms where
  available for destructive operations.

### 11.3 Remote Capability

The `remote` capability covers network and credential behavior. It is
disabled unless trusted configuration explicitly enables it.

Required constraints:

- Require protocol and host allowlists. Empty allowlists deny network
  access.
- Deny prompts, askpass programs, inherited credential helpers, and
  arbitrary SSH command behavior unless supplied through trusted runtime
  configuration.
- Redact token-bearing URLs, credentials, remote names where configured, and
  credential-helper diagnostics in command display, output, errors, logs,
  and audit records.
- Keep credential use explicit. `allow_remote_credentials=false` denies
  credential-bearing URLs and credential helper injection.
- Deny broad refspecs by default. Allow only typed, bounded refspec forms
  such as selected branch, selected tag, or current branch.
- Deny force push, mirror push, prune, tags-all behavior, recursive
  submodule behavior, and custom upload/download helper paths unless
  explicitly supported by command-level settings and tests.
- `shell.git_clone` may run without an existing repository at `cwd`, but it
  must create or populate only a policy-approved path under
  `workspace_root` and must not overwrite existing paths unless a typed
  overwrite mode is explicitly enabled.
- `shell.git_remote_*` tools must not expose general repository
  configuration. They may read or edit only approved remote entries, with
  URL redaction in all observable output.
- `shell.git_submodule_update` requires both `remote` and
  `allow_submodule_update=true`; it must not recurse by default and must
  enforce the same protocol, host, path, credential, and output policies as
  other remote tools.
- Hooks, including local pre-push hooks, are disabled for remote commands
  unless a future trusted setting provides a reviewed hook profile.

---

## 12. Acceptance Criteria Mapping

| Acceptance criterion | Phase | Test area |
| --- | --- | --- |
| Read-only Git tools are absent unless explicitly enabled. | 1 | ToolManager registration, concrete enablement, `shell`, and `shell.*` wildcard exposure |
| With only `read` capability, no mutating command or mutating mode is exposed or executable. | 1, 5-7 | Capability matrix, schema exposure, execution denial |
| Shell wildcard enablement does not bypass capability gates. | 1 | `shell` and `shell.*` enablement tests with each capability set |
| Unsupported commands, unsupported options, abbreviated options, arbitrary flags, global Git options, aliases, and raw argument forms fail closed. | 2-7 | Schema validation, policy dispatch, hostile config tests |
| Read-only status uses optional-lock-free execution and does not refresh or write the index. | 2, 3 | Status policy, environment, real-repository smoke tests |
| Diff, show, log, stash show, and blame do not execute external diff or textconv behavior. | 2, 4 | External-process hardening, hostile attributes/config tests |
| Tests cover hostile repository configuration, attributes, external diff, textconv, pager, fsmonitor, hooks, signing, filters, credential prompts, and editor attempts. | 2-7 | Hostile repository and environment policy tests |
| Tests cover path traversal, absolute paths, NUL bytes, control characters, pathspec magic, option injection through filenames, hidden/sensitive paths, submodules, linked worktrees, bare repos, and alternates outside the workspace. | 2-7 | Path policy and repository form tests |
| Tests verify config access, ungated remote inspection, broad object inspection, reflog access, arbitrary object-path reads, and network commands are denied unless `remote` and required remote policy are enabled. | 2-4, 7 | Denied command surface, revision/object policy, remote policy |
| Tests enforce output and timeout limits for large logs, diffs, grep results, blame output, and stash output. | 2-4 | Output cap, timeout, truncation metadata |
| Tests verify worktree capabilities cannot access history operations, and history capabilities cannot access remote operations. | 5-7 | Capability separation matrix |
| Tests verify remote operations require `remote`, protocol allowlists, host allowlists, prompt denial, URL redaction, credential policy, bounded refspecs, and audit metadata. | 7 | Remote policy, credential policy, audit tests |
| Tests verify `remote` does not bypass read, worktree, or history command gates; `remote` alone must not permit commit, reset, or branch deletion. | 7 | Capability separation matrix |
| Tests verify clone cannot write outside `workspace_root`, cannot overwrite existing paths without an explicit typed mode, and cannot leave partial output unreported after timeout or cancellation. | 7 | Clone destination, timeout, cancellation cleanup |
| Tests verify submodule update requires both `remote` and `allow_submodule_update=true`, with no recursion by default. | 7 | Submodule policy and remote policy |
| Real-subprocess smoke tests cover the supported read-only commands in temporary repositories. | 3, 4 | Isolated Git smoke tests |
| Guardrail tests assert Git command modules do not directly spawn processes, use shell evaluation, or bypass the policy/executor boundary. | 2-8 | Static guardrails and runtime process tests |
| Denials are auditable without logging secrets from environment variables, remotes, paths, patches, or command output. | 2-8 | Formatter, logging, event metadata, audit redaction |

---

## 13. Agent And Operator Guidance

Agents may use read-only Git tools for diagnosis, but they must still treat
results as sensitive repository data.

Recommended workflow:

1. Inspect with the narrowest read-only shell Git tool that answers the
   question.
2. Keep pathspecs scoped and explicit.
3. Before mutation, inspect status and relevant diffs.
4. Use `worktree` tools only for local file/index changes.
5. Use `history` tools only when the task explicitly requires commits,
   ref changes, history transforms, or destructive repository operations.
6. Use `remote` tools only when the task explicitly requires network access,
   and inspect the configured host, protocol, and credential policy first.

Operators must treat formatted tool output, snippets, logs, audit metadata,
and task projections as potentially sensitive even when the command is
read-only.

---

## 14. Documentation Requirements

User-facing documentation is Phase 8 work and must include:

- How to enable read-only shell Git tools under `[tool.shell.git]`.
- How to enable `worktree`, `history`, and `remote` capabilities separately.
- A command table listing capability, supported modes, denied forms, output
  caps, default exposure, implementation phase, and mutation risk.
- Examples for common read-only workflows using only `shell.git_*` tool
  names.
- Clear warnings that read-only output may expose sensitive repository data.
- Notes about requiring a local Git executable or configured shell execution
  backend.
- Failure-mode examples for missing Git, non-repository cwd, capability
  denial, path denial, revision denial, denied remote host/protocol, denied
  credentials, timeout, and output truncation.

---

## 15. Remaining Decisions Before Later Phases

The following decisions are intentionally not settled by Phase 0:

- Whether mutating commands require Avalan confirmation or human-in-the-loop
  review by default, or only for destructive `history` operations.
- Whether remote commands require Avalan confirmation or human-in-the-loop
  review by default, especially push, pull, clone, remote URL changes, and
  submodule update.
- Whether any deployment needs safe remote branch-name listing in read-only
  mode, given that remote names can reveal private project details.
- Whether a future content-read capability should allow controlled object
  path reads, or whether that remains out of scope with broad object
  database inspection.
