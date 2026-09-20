# Evidence-Bound Agentic Engineering

Organize agentic engineering around durable tasks that accumulate verifiable
results. Do not organize it around conversations, agent personalities, or the
number of agents running.

The execution model is defined in `docs/agents/task-runs.md`: an Engineering
Task records intent, a Task Contract defines execution terms, a Task Run owns a
dedicated worktree, and replaceable Agent Attempts execute or recover that run.
This document defines the working method that sits on top of that model.

The loop is:

```text
Frame an outcome
  -> resolve consequential uncertainty
  -> contract a bounded task
  -> execute in evidence-producing increments
  -> independently verify a fixed candidate
  -> authorize promotion
  -> improve the method from observed failures
```

This is a local, environment-first workflow. Remote execution is an extension.
Human authority covers merge, push, deployment, publication, security-policy
changes, external effects, and destructive cleanup.

## Optimization Target

Optimize for reliable engineering outcomes per unit of human attention and
total cost, subject to correctness, maintainability, and authority constraints.

Progress means different things by mode:

- implementation progress: behavior becomes demonstrably correct;
- scout progress: a consequential unknown becomes a supported answer or a
  precisely bounded unknown;
- experiment progress: evidence discriminates between hypotheses;
- recovery progress: trustworthy control is restored without losing work or
  repeating an uncertain effect.

Large diffs, long sessions, and many parallel agents are not progress by
themselves.

## Roles

Roles are responsibilities. They are not necessarily separate model agents.

- **Human engineering lead**: chooses outcomes, resolves consequential
  tradeoffs, approves scope and promotion.
- **Planning/scouting agent**: inspects reality, identifies uncertainty,
  proposes slices and acceptance criteria. It cannot silently turn
  recommendations into authorized scope.
- **Implementing agent**: produces a candidate within the contract. It cannot
  declare its own evidence sufficient by assertion.
- **Verifier/reviewer**: tests claims and challenges the candidate. It does not
  modify the candidate while reviewing it.
- **Task runtime and command adapters**: capture state, execute checks, enforce
  implemented gates, and record lifecycle events.
- **Human-authorized integration path**: accepts a fixed candidate into a target
  branch or external environment. It does not infer authorization from passing
  tests.

Use these three terms when discussing evidence:

- **Candidate**: the exact artifact being evaluated. During development this may
  be a captured dirty-tree snapshot; at final review it should normally be a
  clean commit or an identified commit range.
- **Evidence record**: an observation about a candidate under stated
  conditions. It identifies the command or evaluator, inputs, environment,
  result, and retained output.
- **Promotion**: an authorized transition into accepted or externally visible
  state. Completing a run, merging a branch, pushing, and deploying are
  separate transitions.

An agent produces a candidate. Verification produces evidence. An authorized
decision accepts the outcome. None substitutes for another.

## Work Modes

Choose the work mode before choosing the workflow. The modes are defined in
`docs/agents/task-runs.md` and have different completion conditions:

- **delivery** asks whether specified behavior can be provided.
- **remediation** asks whether a specific defect can be explained and corrected.
- **scout** asks what must be learned before committing to an approach.
- **experiment** asks whether an intervention satisfies a predefined evaluation
  rule.

A scout may discover an attractive implementation. That does not authorize
implementing it. An experiment may reject the hypothesis and still complete
successfully.

Scale ceremony with risk and uncertainty. Increase upfront design when a wrong
decision is expensive to reverse, affects other work, or cannot be detected
cheaply. Otherwise prefer a bounded probe and early evidence.

## Outcome Packet

Before execution, frame the outcome rather than the requested edit. A practical
packet answers:

```text
Outcome:
Why this matters:
Mode:
Non-goals:
Authoritative context:
Acceptance claims and how each will be checked:
Allowed change surfaces:
Allowed external effects:
Required environment and dependencies:
Time, compute, retry, and delegation budgets:
Escalation conditions:
Required final artifacts:
Completion and promotion rules:
```

Agents may draft this packet. The responsible human approves consequential
choices. Routine tasks can use previously approved policies.

## Uncertainty Ledger

Before substantial implementation, inspect the actual code path, existing
commands, tests, environment, and relevant failure evidence.

Record consequential uncertainty as a compact ledger:

```text
Question:
Current evidence:
Consequence if wrong:
Next discriminating action:
Status: observed | inferred | unknown
```

The objective is not complete understanding. It is enough understanding to
choose the next safe, informative step.

## Slicing

Design and slice work around independently checkable outcomes. Prefer vertical
slices that expose behavior through the real interface.

Each slice needs:

- an acceptance claim;
- an owner;
- bounded inputs and outputs;
- an integration point;
- required evidence;
- dependencies expressed as interface versions or prerequisite receipts.

A change of tactic within authorized scope is ordinary execution. A change to
authoritative design, acceptance criteria, output boundaries, or permissions is
a contract change and uses `task-rebind`.

## Baseline

The first execution step asks:

```text
Can this environment evaluate this task, and what was already failing before we
changed anything?
```

Record baseline failures separately from regressions. A missing prerequisite is
not a successful check. Environment-specific qualification is not a hermetic
repository test.

For ML systems work, the environment description may include accelerator,
driver/runtime, dependency locks, launch configuration, data identity, and
shared-resource conditions. Include only facts that can materially change the
claim being evaluated.

## Increment Loop

Execute through short evidence-producing increments:

```text
Observe
  -> state an expectation
  -> make a bounded change
  -> run a discriminating check
  -> interpret the result
  -> checkpoint
```

For a defect, begin with a reproduction. For a deterministic feature, use
behavior-first tests. For a performance change, establish a baseline and
measurement protocol. For a scout, state the question and stopping rule. For an
experiment, state the hypothesis and evaluation rule before inspecting final
results.

When a check fails, classify the failure before editing again:

- implementation defect;
- incorrect assumption;
- environment problem;
- inadequate test;
- uncertain external outcome.

Stop repeating a tactic after two attempts that produce materially the same
failure without new evidence. The next action must change the hypothesis,
improve instrumentation, or request a decision.

## Acceptance Map

Verification challenges claims. It does not merely execute commands.

Every Task Contract should include an acceptance map:

```text
Claim:
Evidence:
Candidate:
Environment:
Reuse rule:
Invalidation rule:
```

Examples:

- ordinary execution works -> behavioral test through the public command or
  interface;
- interrupted work is preserved -> crash-injection test plus state comparison;
- an uncertain launch is not blindly repeated -> lost-acknowledgement scenario
  plus remote operation reconciliation;
- review covers intended requirements -> recorded task context, candidate
  range, and reviewer output;
- result applies to final candidate -> candidate-bound receipt and unchanged
  verification inputs.

Use increasing verification scope. During editing, run targeted checks. At
coherent checkpoints, run the relevant subsystem suite. At candidate sealing,
run required integration and qualification gates.

Reuse expensive evidence only when the inputs relevant to that claim remain
unchanged and the reuse rule is explicit. Otherwise invalidate it.

For higher assurance, evaluate a separately materialized fixed candidate rather
than a worktree that remains writable throughout testing.

## Independent Review

Give reviewers the contract, exact candidate, relevant evidence, and unresolved
questions. A reviewer challenges specification coverage, failure behavior,
maintainability, and evidence sufficiency.

The candidate cannot silently redefine the policy that accepts it. Changes to
acceptance rules, required checks, or authorization policy require explicit
review and authorization.

Every review finding needs a disposition:

- corrected with evidence;
- rejected with justification;
- deferred by authorized decision.

Changing the reviewed candidate requires renewed review at the appropriate
scope. A second model is another source of scrutiny, not a guarantee of
correctness.

## Completion and Promotion

`task-finish` means the run produced its contracted work package and the
required evidence passed. It does not mean merged, pushed, or deployed.

For delivery, the completed package can be ready for human-authorized
integration. For an experiment, it can contain a rejected hypothesis. Keep
scientific disposition and promotion status separate from execution completion.

After integration changes a candidate through conflict resolution, rebasing, or
combination with other work, rerun the checks needed for that new candidate.

## Delegation

Default to one implementing writer with optional read-only scouting or review in
parallel. Use parallel implementation only when contracts and interfaces make
the work independently executable.

A delegation must identify:

- question or deliverable;
- frozen inputs;
- permitted actions;
- required evidence;
- budget;
- stopping rule;
- parent task that consumes the result.

Read-only helpers may inspect the same candidate. Any helper that edits,
formats, generates source, or fixes tests is a writer and needs its own Task Run
and worktree.

A human editing the active worktree is also a writer. Pause or transfer the
writer role instead of treating human intervention as exempt from concurrency
rules.

Child tasks consume parent-authorized budget. Agent replacement does not reset
the budget. Recursive delegation cannot create new authority or spending
capacity.

## Recovery Method

Recovery is engineering work, not session restoration. The base rule is:

```text
A new Task Run starts clean. A replacement Agent Attempt may recover the same
dirty worktree.
```

Strengthen the recovery model from `docs/agents/task-runs.md` with these rules:

- A missed heartbeat does not prove the writer is gone. Before reusing the same
  writable worktree, require enforceable exclusion of the old writer, such as
  supervisor-confirmed process-group shutdown or revoked write access.
- A journal lease token does not fence arbitrary direct filesystem writes.
- A filesystem snapshot is not a complete execution snapshot. Classify source
  changes, index state, declared untracked inputs, reproducible caches, ignored
  outputs, running processes, remote jobs, notebooks, databases, and
  credentials separately.
- Do not claim lossless recovery merely because a patch and untracked-file
  archive exist.
- The Git common-dir journal protects against ordinary worktree removal, not
  host loss or backup failure.
- For remote launches and other consequential operations, record a stable
  logical operation identity before dispatch and retain the resulting receipt.
- Lost acknowledgement means outcome unknown. Reconcile before retrying. Use
  idempotency keys where the remote system supports them.
- A restored checkpoint is historical evidence, not a fresh permission grant.
  Revalidate authorization, cancellation, remaining budgets, contract binding,
  and dependency validity before resuming.
- Long-running jobs need durable job identity and lifecycle records. Agent
  replacement must not implicitly kill, restart, or orphan them.

## Experiments

For experiments and ML systems work, separate three questions:

- Does the implementation execute correctly?
- Does the experiment support the hypothesis?
- Is the result worth adopting?

Before an experiment, fix the baseline, intervention, workload/data identity,
metric, practical acceptance threshold, repetition or seed policy, resource
budget, and stopping rule. Separate exploratory measurements from final
evaluation. Record protocol changes as new versions.

Keep the complete attempt ledger: unsuccessful variants, exclusions, reruns, and
inconclusive outcomes belong in the record.

Harness-improvement tasks use this form:

```text
Observed failure pattern:
Suspected cause:
Proposed intervention:
Predicted effect:
Bounded evaluation:
Reviewed adoption:
```

A harness-improvement task must not weaken its own evaluator or authorize its
own promotion. Keep held-out evaluation where meaningful, and preserve a
reversible path to the previous harness version.

## Human Supervision

The primary interface for supervision should be a task-and-evidence explorer.
Raw conversation is a drill-down view.

Navigation follows:

```text
Task -> Run -> Attempt -> Checkpoint -> Event -> Artifact
```

The default task view should show current outcome, latest binding, next
acceptance claim, latest verified result, active jobs, remaining budget,
unresolved uncertainty, and required decision.

Marked events or artifacts should become an evidence packet with stable
identities, candidate/context references, bounded excerpts, and a specific
question for the reviewing agent.

Escalations ask for decisions, not status acknowledgement. Include the decision
needed, why it exceeds the current contract, evidence, options, recommendation,
and what is paused.

At handoff, require a checkpoint, active-job inventory, and an unambiguous next
action.

Periodically review repeat failures, unnecessary interventions, verification
cost, stale evidence, integration delays, and recovery outcomes. Promote
recurring lessons into small reviewed changes to the repository, tools, or
methodology.

Measure accepted outcomes and decision quality. Do not reward token volume,
lines changed, agent count, or ticket closure alone.

## Adoption Sequence

Adopt the method before building a large orchestration system:

1. **Disciplined local workflow**: small contracts, explicit modes, one-writer
   worktrees, baseline checks, candidate-bound receipts, independent review, and
   human-authorized integration. Lifecycle steps may remain manual.
2. **Automated continuity and gates**: admission, checkpoints, recovery
   snapshots, evidence invalidation, and completion checks. Qualify by
   intentionally interrupting execution at meaningful boundaries.
3. **Distributed delegation**: remote job reconciliation, durable artifact
   retention, hierarchical budgets, cancellation handling, and stronger writer
   exclusion once the local method works.

A first end-to-end qualification should start a bounded engineering task,
produce a partial candidate, launch verification, interrupt the agent, recover
with a fresh attempt, reconcile the job, finish the evidence package, and
present the unchanged candidate for human-authorized integration.

## Governing Principle

Authorize outcomes and boundaries. Let agents choose tactics within them.
Require evidence for claims. Preserve state across failures. Keep promotion
explicit.
