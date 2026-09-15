--------------------------- MODULE StepTxnBad ---------------------------
\* A deliberately buggy variant of StepTxn that violates NoMutationAfterAbort:
\* the AbortMutating action lets a rank on the abort path still mutate optimizer
\* state and bump the committed version. TLC must surface the Inv violation with
\* a counterexample trace, mirroring the MeshPlanBad contract.
\*
\* This is a full re-statement of the StepTxn state machine (rather than an
\* INSTANCE override) because the bug lives inside a transition, and TLA+
\* INSTANCE cannot replace a single action of the parent module.
EXTENDS Naturals, FiniteSets

BadRanks == {"r0", "r1"}
InitVersion == 7

VARIABLES phase, overflow, decision, version, optim_mutated

vars == <<phase, overflow, decision, version, optim_mutated>>

Phases == {"idle", "computing", "synchronizing", "prepared",
           "committing", "committed", "aborting", "aborted"}

AllInPhase(p) == \A r \in BadRanks : phase[r] = p

Compute(r) ==
  /\ phase[r] = "idle"
  /\ \E ov \in BOOLEAN :
       /\ phase' = [phase EXCEPT ![r] = "computing"]
       /\ overflow' = [overflow EXCEPT ![r] = ov]
  /\ UNCHANGED <<decision, version, optim_mutated>>

Sync(r) ==
  /\ phase[r] = "computing"
  /\ phase' = [phase EXCEPT ![r] = "synchronizing"]
  /\ UNCHANGED <<overflow, decision, version, optim_mutated>>

Prepare(r) ==
  /\ phase[r] = "synchronizing"
  /\ \A q \in BadRanks : phase[q] \in {"synchronizing", "prepared"}
  /\ phase' = [phase EXCEPT ![r] = "prepared"]
  /\ UNCHANGED <<overflow, decision, version, optim_mutated>>

Decide ==
  /\ decision = "none"
  /\ AllInPhase("prepared")
  /\ decision' = IF \E r \in BadRanks : overflow[r] THEN "abort" ELSE "commit"
  /\ UNCHANGED <<phase, overflow, version, optim_mutated>>

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

\* BUG: this abort action mutates optimizer state and bumps the version, which
\* violates NoMutationAfterAbort and VersionMonotone. A correct implementation
\* discards the step without touching optimizer/version state.
AbortMutating(r) ==
  /\ decision = "abort"
  /\ phase[r] = "prepared"
  /\ phase' = [phase EXCEPT ![r] = "aborting"]
  /\ version' = version + 1
  /\ optim_mutated' = TRUE
  /\ UNCHANGED <<overflow, decision>>

Aborted(r) ==
  /\ decision = "abort"
  /\ phase[r] = "aborting"
  /\ phase' = [phase EXCEPT ![r] = "aborted"]
  /\ UNCHANGED <<overflow, decision, version, optim_mutated>>

Resolved ==
  \/ AllInPhase("committed")
  \/ AllInPhase("aborted")

ResolvedStutter ==
  /\ Resolved
  /\ UNCHANGED vars

Init ==
  /\ phase = [r \in BadRanks |-> "idle"]
  /\ overflow = [r \in BadRanks |-> FALSE]
  /\ decision = "none"
  /\ version = InitVersion
  /\ optim_mutated = FALSE

Next ==
  \/ \E r \in BadRanks : Compute(r)
  \/ \E r \in BadRanks : Sync(r)
  \/ \E r \in BadRanks : Prepare(r)
  \/ Decide
  \/ \E r \in BadRanks : Commit(r)
  \/ \E r \in BadRanks : Committed(r)
  \/ \E r \in BadRanks : AbortMutating(r)
  \/ \E r \in BadRanks : Aborted(r)
  \/ ResolvedStutter

Spec ==
  /\ Init
  /\ [][Next]_vars

TypeOK ==
  /\ phase \in [BadRanks -> Phases]
  /\ overflow \in [BadRanks -> BOOLEAN]
  /\ decision \in {"none", "commit", "abort"}
  /\ version \in Nat
  /\ optim_mutated \in BOOLEAN

NoMutationAfterAbort ==
  (decision = "abort") => optim_mutated = FALSE

VersionMonotone ==
  /\ version >= InitVersion
  /\ version <= InitVersion + 1
  /\ (version > InitVersion) => decision = "commit"

Inv ==
  /\ TypeOK
  /\ NoMutationAfterAbort
  /\ VersionMonotone

=============================================================================
