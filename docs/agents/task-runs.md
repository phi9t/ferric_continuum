# Task Runs, Contracts, and Recovery

This document defines the execution model for Kata-backed engineering work.
Kata remains the durable issue system. Git stores versioned task context. The
private run journal stores detailed execution state.

## Core Model

Keep these four concepts separate:

- **Engineering Task**: durable intent tracked by one Kata issue.
- **Task Contract**: committed, machine-readable declaration of mode,
  authoritative context, outputs, gates, and recovery policy.
- **Task Run**: one execution of a contract in one dedicated Git worktree.
- **Agent Attempt**: one agent session holding the writer lease for that run.

A fresh Task Run starts from a clean dedicated worktree. The worktree may become
dirty during normal implementation. If an Agent Attempt dies, a replacement
attempt recovers the same run and same dirty worktree after proving ownership
and snapshotting state.

The relationship is:

```text
Kata engineering task
  -> committed task contract and specs/plans
    -> frozen Task Run in a dedicated clean worktree
      -> Agent Attempt 1
      -> Agent Attempt 2 after verified recovery
```

## Task Contracts

Each Kata engineering task should point to exactly one committed contract
manifest:

```text
Execution contract:
- Contract: .scratch/contracts/kata-2742.toml
```

The contract is the small interface for execution. It should be structured,
stable, and parseable. Long mutable ticket prose is context, not the contract.

Example:

```toml
schema_version = 1
kata_ref = "2742"
mode = "delivery"

[[context]]
role = "design"
path = ".scratch/example-effort/spec.org"

[[context]]
role = "task_plan"
path = ".scratch/example-effort/issues/03-implement-task.org"

[[context]]
role = "command_contract"
path = "docs/agents/agentic-engineering.md"

[outputs]
required = [
  "native capsule launcher",
  "capsule contamination receipt",
  "batch Emacs receipt",
]

[verification]
commands = [
  "mise run macos-emacs:capsule-probe",
  "mise run macos-emacs:batch",
  "scripts/agentic/check-fast",
]

[authorization]
gates = []

[recovery]
allow_automatic_resume = true
checkpoint_policy = [
  "after-red",
  "after-green",
  "after-commit",
  "after-review",
  "before-external-side-effect",
  "on-failure",
  "before-handoff",
]
```

Small tasks still get small contracts. A remediation can reference the issue
body, the review finding, and one verification command. A scout can reference a
short committed brief instead of a full implementation plan.

## Work Modes

The contract declares exactly one mode:

- **delivery**: objective, design when applicable, plan, outputs, and
  verification. Completion means implemented behavior, tests, commits, and
  reviews.
- **remediation**: exact defect or review provenance, reproduction, and affected
  contract. Completion means reproduction, red/green regression, correction
  commit, and exact review closure.
- **scout**: bounded questions, permitted evidence sources, and stopping rule.
  Completion means findings, evidence anchors, uncertainties, and recommended
  follow-up tickets. A scout need not implement.
- **experiment**: hypothesis, fixed protocol, evaluation rule, and attempt
  budget. Completion means an attempt ledger and a supported, rejected, or
  inconclusive disposition.

## Admission

`task-start` is an admission gate for a new Task Run. It must verify:

- the Kata issue is open and is an engineering task;
- the ticket contains exactly one contract pointer;
- the contract exists as a Git blob at the proposed context commit;
- every referenced context document exists at that commit;
- no authoritative context file is only untracked or only in the worktree;
- the worktree is a dedicated linked worktree;
- the worktree and index are clean;
- no Git operation is already in progress;
- branch and issue ownership are compatible;
- no other live Task Run owns the worktree;
- the declared mode has all required fields;
- dependency tickets required for admission are closed or have prerequisite
  receipts.

Admission reads context through Git objects, such as `git show <sha>:<path>`,
instead of trusting mutable filesystem bytes.

The admission snapshot records at least:

```json
{
  "task_run_id": "run-2742-...",
  "kata_ref": "2742",
  "kata_revision_observed": 24,
  "contract_digest": "...",
  "context_commit": "...",
  "task_boundary": "...",
  "context_documents": [
    {
      "role": "task_plan",
      "path": "docs/...",
      "blob_id": "...",
      "sha256": "..."
    }
  ],
  "worktree_identity": "...",
  "initial_state": "clean"
}
```

Kata comments and operational metadata may increment issue revisions. Record
the exact observed revision for audit, but compute the contract digest over the
title, body, contract pointer, dependency relationships, and referenced Git
blobs. Ignore comments and `work.*` metadata when deciding whether the execution
contract changed. Require `task-rebind` only when that digest changes.

## Runtime State

Kata uses open and closed as durable issue states. Execution state lives under
`work.*` metadata:

- `work.run_id`
- `work.state`: `ready`, `active`, `interrupted`, `recovering`,
  `needs-human`, or `completed`
- `work.branch`
- `work.worktree_id`
- `work.start_sha`
- `work.context_digest`
- `work.attempt_id`
- `work.session_provider`
- `work.session_id`
- `work.last_checkpoint`
- `work.heartbeat_at`
- `work.attention`
- `work.attention_msg`

The full event journal does not belong in Kata metadata values.

## Run Journal

Each Task Run has a private append-only journal under the repository Git common
directory so all linked worktrees can find it:

```text
<git-common-dir>/devx/runs/<run-id>/
|-- manifest.json
|-- events.jsonl
|-- checkpoints/
|   |-- 0001-start.json
|   |-- 0002-red.json
|   |-- 0003-green.json
|   `-- 0004-interrupted.json
|-- recovery/
|   `-- <attempt-id>/
|       |-- state.json
|       |-- tracked.patch
|       |-- index
|       |-- untracked.tar
|       `-- manifest.sha256
`-- trajectories/
    `-- <attempt-id>.json
```

Directories are mode `0700`. Files are mode `0600`.

The journal records commands and exit codes, test results, commits, context
rebinds, writer lease changes, Git HEAD and index identities, dirty-state
digests, trajectory locators and hashes, checkpoint summaries, recovery
decisions, blockers, and authorization decisions.

## Checkpoints

Checkpoints combine mechanically captured state with a short semantic handoff:

```json
{
  "phase": "green",
  "head_sha": "...",
  "dirty_state_digest": "...",
  "tests": [
    {
      "command": "...",
      "exit_code": 0
    }
  ],
  "summary": "Capsule environment builder passes hostile PATH tests.",
  "next_action": "Add the mise wrapper and contamination receipt.",
  "known_failures": [],
  "open_questions": []
}
```

Mandatory checkpoint moments:

- after observing red;
- after reaching green;
- after a commit;
- after review;
- before an authenticated provider call or other external side effect;
- after an unexpected failure;
- before a deliberate handoff;
- at normal task completion.

If an agent crashes before writing a semantic checkpoint, recovery uses the
trajectory to reconstruct the gap.

## Trajectory Binding

At attempt admission, record the current agent-session identity:

```json
{
  "provider": "traecli",
  "thread_id": "...",
  "session_id": "...",
  "source_kind": "rollout-jsonl",
  "start_offset": 12345,
  "start_turn": "...",
  "local_locator": "...",
  "source_identity": {
    "device": 123,
    "inode": 456
  }
}
```

At every checkpoint, record the last observed byte offset or turn ordinal and a
hash-chain value. Recovery can then identify which trajectory segment belongs
to the attempt.

Trajectory access is an adapter interface:

- identify the current session;
- locate the durable transcript;
- verify transcript identity;
- read a bounded event range;
- extract deterministic execution facts.

Trae is the first adapter. Other agents may provide equivalent adapters. If no
adapter exists, explicit checkpoints are still portable, but recovery
confidence is lower.

## Kata Projection

Raw trajectories must not be copied into Kata. They can contain prompts, source
fragments, command output, filesystem paths, credentials, or provider
responses.

Kata receives current-state metadata and lifecycle comments only.

Example metadata:

```text
work.run_id=run-2742-...
work.state=interrupted
work.last_checkpoint=0004
work.session_provider=traecli
work.session_id=<opaque ID>
work.attention=ok
work.attention_msg="automatic recovery eligible"
```

Example lifecycle comment:

```text
RUN INTERRUPTED v1

Run: run-2742-...
Attempt: attempt-1
Context: <digest>
HEAD: <sha>
Checkpoint: 0004
Dirty state: 7 tracked, 2 untracked, no conflicts
Last verified result: capsule unit suite passed
Current intent: implementing mise launcher
Next action: inspect wrapper RED test and continue
Raw trajectory: locally available, sha256=<digest>
Recovery: eligible for automatic resume
```

Write lifecycle comments only for run admission, contract rebind,
interruption, recovery, needs-human transition, and completion.

## Dirty-Worktree Recovery

Recovery starts read-only:

1. Resolve the open Task Run from Kata metadata.
2. Locate the recorded dedicated worktree.
3. Prove the previous attempt is no longer active using session status, process
   identity, lease nonce, heartbeat, and host identity.
4. Verify branch, Git common directory, task boundary, context digest, and
   current HEAD.
5. Inventory the dirty tree with `git status --porcelain=v2 -z`.
6. Detect staged, unstaged, untracked, conflicted, ignored, submodule, and
   in-progress Git-operation state.
7. Create an immutable private pre-recovery snapshot: binary patch for tracked
   changes, exact index copy, no-follow archive of regular untracked files,
   path metadata, modes, sizes, content hashes, and `manifest.sha256`.
8. Resolve the previous attempt's trajectory.
9. Extract deterministic facts: patches applied, commands, exit codes, test
   runs, commits, and failures.
10. Build a sanitized recovery memo anchored to checkpoints and trajectory
    event offsets.
11. Compare observed changes with the task contract and expected outputs.
12. Admit a new Agent Attempt and continue in the same worktree.

Automatic recovery is allowed when the previous writer is provably gone, the
contract is unchanged, the worktree belongs to the same Task Run, no concurrent
writer is ambiguous, source state can be snapshotted without loss, there is no
unexplained destructive Git operation, and dirty state is structurally readable.

A missed heartbeat does not prove the writer is gone. Before reusing the same
writable worktree, require enforceable exclusion of the previous writer: for
example, supervisor-confirmed shutdown of the relevant process group and
remaining mutators, or revoked write access through an implemented isolation
mechanism. A journal lease token does not fence arbitrary direct filesystem
writes.

A filesystem snapshot is not a complete execution snapshot. Classify source
changes, index state, declared untracked inputs, reproducible caches, ignored
outputs, running processes, remote jobs, notebooks, databases, and credentials
separately. Do not claim lossless recovery merely because a patch and
untracked-file archive exist. The Git common-dir journal protects against
ordinary worktree removal, not host loss or backup failure.

For remote launches and other consequential operations, record a stable logical
operation identity before dispatch and retain the resulting receipt. Lost
acknowledgement means outcome unknown; reconcile the operation before retrying
it. Use idempotency keys where the remote system supports them.

A restored checkpoint is historical evidence, not a fresh permission grant.
Before resuming, revalidate current authorization, cancellation, remaining
budgets, contract binding, and dependency validity. Long-running jobs need a
durable job identity and lifecycle record; replacing the agent that launched the
job must not implicitly kill, restart, or orphan it.

Move to `needs-human` when two writers may still be active, the contract
changed, branch or worktree identity changed, a merge/rebase/cherry-pick/bisect
is in progress without recorded intent, untracked state has unsupported file
types or exceeds the snapshot budget, trajectory and filesystem materially
disagree, an external side effect may be half-completed or unreconciled, or the
recovery snapshot cannot be created.

Missing raw trajectory alone does not block recovery. It lowers confidence, but
the committed contract, dirty-state snapshot, Git history, and portable Kata
checkpoint may still be sufficient.

## Rebinding

When the human changes scope mid-run:

1. `task-checkpoint`
2. commit the revised spec or plan
3. edit the Kata contract when necessary
4. `task-rebind <ref> --reason "..."`

`task-rebind` records previous and new contract digests, previous and new
document blobs, the human instruction or authorization, current dirty-state
digest, which prior outputs remain valid, and which gates must rerun. It does
not rewrite `work.start_sha`, discard work, or silently reinterpret prior
evidence.

## Command Interface

Put implementation behind one deep module and keep scripts as wrappers:

```text
scripts/agentic/task context <ref>
scripts/agentic/task validate <ref>
scripts/agentic/task start <ref>
scripts/agentic/task checkpoint <ref> ...
scripts/agentic/task status <ref>
scripts/agentic/task resume <ref>
scripts/agentic/task rebind <ref> ...
scripts/agentic/task finish <ref> ...
```

Existing commands such as `scripts/agentic/task-start`,
`scripts/agentic/task-finish`, and `scripts/agentic/new-worktree` should remain
valid wrappers when they exist.

The narrow interface is:

- resolve contract;
- admit run;
- record checkpoint;
- recover run;
- rebind contract;
- finish run.

Git inspection, trajectory adapters, snapshots, and Kata projection stay behind
that interface.

## Finish Gate

`task-finish` additionally requires:

- all intended work committed;
- clean worktree and index;
- no Git operation in progress;
- current context digest matches the latest admitted binding;
- all task-owned commits have valid trailers;
- all required reviews pass;
- mode-specific outputs exist;
- mode-specific verification succeeds;
- latest agent attempt is checkpointed;
- no unresolved recovery ambiguity;
- private run journal and portable Kata summary reconcile;
- final HEAD and commit set remain unchanged through closure.

The resulting architecture is:

```text
Kata body
  -> committed task-contract manifest
    -> committed design/spec/plan
      -> frozen context snapshot
        -> dedicated Task Run worktree
          -> one or more recoverable Agent Attempts
            -> private detailed journal
            -> sanitized portable Kata lifecycle
```
