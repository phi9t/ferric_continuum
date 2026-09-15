--------------------------- MODULE StepTxn ---------------------------
\* Abstract step-transaction model (Layer 0 of the distributed-training TLA+
\* track). A training step is a distributed transaction over a versioned device
\* mesh. Each rank walks a small phase machine; the group takes exactly one
\* global commit/abort decision; committed model versions increase
\* monotonically; and no optimizer state mutates after an abort. Collectives,
\* layout, and pipeline scheduling are deliberately abstracted away here -- they
\* refine this contract in later layers (issues 04/06) and this spec is the
\* refinement target for the first runtime trace bridge (issue 08).
\*
\* Phases (per rank):
\*   "idle"          -- before the step starts
\*   "computing"     -- local forward/backward, may observe an AMP/loss-scale
\*                      overflow (a nondeterministic local input)
\*   "synchronizing" -- gradients reduced across the group (abstract collective)
\*   "prepared"      -- rank has voted; ready for the global decision
\*   "committing"    -- global decision was commit; rank applies optimizer step
\*   "committed"     -- optimizer step applied, version bumped
\*   "aborting"      -- global decision was abort; rank discards the step
\*   "aborted"       -- step discarded, no mutation
EXTENDS Naturals, FiniteSets

CONSTANTS
  Ranks,          \* set of participating ranks (one logical group)
  JobGen,         \* job generation (constant across the step)
  Epoch,          \* membership epoch (constant across the step)
  LogicalStep,    \* logical step index being attempted
  AccumWindow,    \* gradient-accumulation window size
  Attempt,        \* attempt number for this logical step
  InitVersion     \* committed model version before this step

VARIABLES
  phase,          \* [Ranks -> phase string]
  overflow,       \* [Ranks -> BOOLEAN]  local AMP/loss-scale overflow observed
  decision,       \* "none" | "commit" | "abort"  -- one global decision
  version,        \* committed model version (monotone)
  optim_mutated   \* BOOLEAN -- has optimizer state been mutated this step

vars == <<phase, overflow, decision, version, optim_mutated>>

Phases == {"idle", "computing", "synchronizing", "prepared",
           "committing", "committed", "aborting", "aborted"}

\* A transaction id is the tuple that names this step attempt. It is constant
\* for the whole behavior; recorded here so trace validation can bind to it.
TxnId == <<JobGen, Epoch, LogicalStep, AccumWindow, Attempt>>

AllInPhase(p) == \A r \in Ranks : phase[r] = p
AnyInPhase(p)  == \E r \in Ranks : phase[r] = p

\* Local compute: a rank moves idle -> computing, nondeterministically
\* observing an overflow. The overflow flag is a local input, not a decision.
Compute(r) ==
  /\ phase[r] = "idle"
  /\ \E ov \in BOOLEAN :
       /\ phase' = [phase EXCEPT ![r] = "computing"]
       /\ overflow' = [overflow EXCEPT ![r] = ov]
  /\ UNCHANGED <<decision, version, optim_mutated>>

\* Cross-group synchronization (abstract collective): a rank moves
\* computing -> synchronizing once it has finished local compute.
Sync(r) ==
  /\ phase[r] = "computing"
  /\ phase' = [phase EXCEPT ![r] = "synchronizing"]
  /\ UNCHANGED <<overflow, decision, version, optim_mutated>>

\* Prepare (vote): a rank moves synchronizing -> prepared. Voting requires all
\* ranks to have reached at least synchronizing so the group vote is consistent
\* (no rank votes before the reduction is group-wide observable).
Prepare(r) ==
  /\ phase[r] = "synchronizing"
  /\ \A q \in Ranks : phase[q] \in {"synchronizing", "prepared"}
  /\ phase' = [phase EXCEPT ![r] = "prepared"]
  /\ UNCHANGED <<overflow, decision, version, optim_mutated>>

\* The single global commit/abort decision. Taken once, only when every rank is
\* prepared. Commit iff no rank observed an overflow; otherwise abort. This is
\* the "one global commit/abort decision" requirement.
Decide ==
  /\ decision = "none"
  /\ AllInPhase("prepared")
  /\ decision' = IF \E r \in Ranks : overflow[r] THEN "abort" ELSE "commit"
  /\ UNCHANGED <<phase, overflow, version, optim_mutated>>

\* Commit path: each prepared rank applies the optimizer step. The first rank to
\* enter "committing" bumps the committed version exactly once and marks the
\* optimizer mutated. Ranks then settle into "committed".
Commit(r) ==
  /\ decision = "commit"
  /\ phase[r] = "prepared"
  /\ phase' = [phase EXCEPT ![r] = "committing"]
  /\ version' = IF ~optim_mutated THEN version + 1 ELSE version
  /\ optim_mutated' = TRUE
  /\ UNCHANGED <<overflow, decision>>

Committed(r) ==
  /\ decision = "commit"
  /\ phase[r] = "committing"
  /\ phase' = [phase EXCEPT ![r] = "committed"]
  /\ UNCHANGED <<overflow, decision, version, optim_mutated>>

\* Abort path: each prepared rank discards the step. No optimizer mutation and
\* no version bump are allowed on this path.
Abort(r) ==
  /\ decision = "abort"
  /\ phase[r] = "prepared"
  /\ phase' = [phase EXCEPT ![r] = "aborting"]
  /\ UNCHANGED <<overflow, decision, version, optim_mutated>>

Aborted(r) ==
  /\ decision = "abort"
  /\ phase[r] = "aborting"
  /\ phase' = [phase EXCEPT ![r] = "aborted"]
  /\ UNCHANGED <<overflow, decision, version, optim_mutated>>

Resolved ==
  \/ AllInPhase("committed")
  \/ AllInPhase("aborted")

\* Stuttering recovery action allowed only once the transaction has resolved.
ResolvedStutter ==
  /\ Resolved
  /\ UNCHANGED vars

Init ==
  /\ phase = [r \in Ranks |-> "idle"]
  /\ overflow = [r \in Ranks |-> FALSE]
  /\ decision = "none"
  /\ version = InitVersion
  /\ optim_mutated = FALSE

Next ==
  \/ \E r \in Ranks : Compute(r)
  \/ \E r \in Ranks : Sync(r)
  \/ \E r \in Ranks : Prepare(r)
  \/ Decide
  \/ \E r \in Ranks : Commit(r)
  \/ \E r \in Ranks : Committed(r)
  \/ \E r \in Ranks : Abort(r)
  \/ \E r \in Ranks : Aborted(r)
  \/ ResolvedStutter

Spec ==
  /\ Init
  /\ [][Next]_vars
  /\ \A r \in Ranks : WF_vars(Compute(r))
  /\ \A r \in Ranks : WF_vars(Sync(r))
  /\ \A r \in Ranks : WF_vars(Prepare(r))
  /\ WF_vars(Decide)
  /\ \A r \in Ranks : WF_vars(Commit(r))
  /\ \A r \in Ranks : WF_vars(Committed(r))
  /\ \A r \in Ranks : WF_vars(Abort(r))
  /\ \A r \in Ranks : WF_vars(Aborted(r))

TypeOK ==
  /\ phase \in [Ranks -> Phases]
  /\ overflow \in [Ranks -> BOOLEAN]
  /\ decision \in {"none", "commit", "abort"}
  /\ version \in Nat
  /\ optim_mutated \in BOOLEAN

\* Exactly one global commit/abort decision governs every rank: no rank is on a
\* commit-flavored phase while the group decided abort, or vice versa.
OneGlobalDecision ==
  /\ (\E r \in Ranks : phase[r] \in {"committing", "committed"})
       => decision = "commit"
  /\ (\E r \in Ranks : phase[r] \in {"aborting", "aborted"})
       => decision = "abort"

\* No optimizer mutation on the abort path.
NoMutationAfterAbort ==
  (decision = "abort") => optim_mutated = FALSE

\* Committed model version never exceeds one bump past the initial version, and
\* only a commit decision may raise it. (Monotone increase within one step.)
VersionMonotone ==
  /\ version >= InitVersion
  /\ version <= InitVersion + 1
  /\ (version > InitVersion) => decision = "commit"

\* A rank only commits/aborts from a prepared-or-later state reached through the
\* full phase pipeline: nobody skips synchronization straight into commit.
NoSkipSync ==
  \A r \in Ranks :
    phase[r] \in {"committing", "committed", "aborting", "aborted"}
      => decision \in {"commit", "abort"}

\* Liveness: every posted step eventually resolves to committed or aborted.
StepResolves == <>Resolved

Inv ==
  /\ TypeOK
  /\ OneGlobalDecision
  /\ NoMutationAfterAbort
  /\ VersionMonotone
  /\ NoSkipSync

=============================================================================
