#!/usr/bin/env python3
"""Translate a Ferric ``tnsr.mesh_sim_trace.v0`` trace into a TLA+ behavior that
is checked to *refine* the abstract step-transaction contract ``StepTxn.tla``.

This is the first runtime trace bridge for the distributed-training TLA+ track
(``.scratch/distributed-training-tla/issues/08``). The framing is inherited from
arXiv:2602.22631 (TorchLean): a recorded runtime artifact is only trustworthy if
it *refines* a checked abstract spec. A trace linter with no spec is out of
scope. Here the abstract spec is ``StepTxn.tla`` and the runtime artifact is a
recorded ``MeshTrace`` (one training step) emitted by the graduated DTensor mesh
simulation (``ferric_continuum/tnsr/src/dtensor/trace.rs``).

Refinement strategy
-------------------
The trace is a *linear* observation of one step, so the refinement question is
trace inclusion: is the observed transition sequence a prefix-consistent
behavior of ``StepTxn``? We answer it by generating a small self-contained TLA+
module (``<Stem>.tla``) that:

  * fixes ``StepTxn``'s constants to a one-rank group (the trace is a single
    logical step; the abstract phase pipeline for one rank captures the
    commit/abort ordering we observe),
  * pins ``overflow`` to whether the trace contained a Failure event (so a
    Failure forces the ``abort`` decision, exactly as ``StepTxn.Decide`` does),
  * drives the state machine along the observed phase sequence via a step
    counter ``k`` and a fixed ``ObservedPhase`` function, and
  * asserts ``StepTxn.Inv`` as the invariant plus ``TraceConsumed`` (the whole
    observed sequence is realized).

If the observed sequence is *not* a legal StepTxn behavior (e.g. the trace shows
an optimizer commit after a Failure), the generated Next relation deadlocks
before consuming the trace and ``TraceConsumed`` is violated -- TLC surfaces the
named invariant with a counterexample.

Variant-subset decision (observed vs. ignored)
-----------------------------------------------
The ``MeshTraceEvent`` schema has four variants. The first refinement observes a
strict subset and ignores the rest, with rationale:

  observed:
    * ``collective``   -> the abstract cross-group synchronization
                          (``StepTxn.Sync``/``Prepare``). Any collective means
                          the step reached the ``synchronizing`` phase.
    * phase ``optimizer`` (on any observed event) -> the commit path
                          (``Commit``/``Committed``). The optimizer phase only
                          runs on a committed step.
    * ``failure``      -> the abort path (``Abort``/``Aborted``); a recorded
                          failure sets ``overflow`` so ``Decide`` must abort.
  ignored:
    * ``layout_transition`` -> pure data-placement bookkeeping; it refines the
                          layout/provenance contract (issue 04), not the
                          step-transaction phase machine. Observing it here would
                          couple two independent abstractions.
    * ``injection``    -> a fault *stimulus*, not an observed transition; its
                          *effect* shows up as a ``failure`` event, which we do
                          observe. Counting the stimulus separately would
                          double-count the abort cause.

The mapping is deliberately conservative: a step that reaches ``optimizer``
without any preceding ``failure`` is a legal commit; a step with a ``failure``
that still reaches ``optimizer`` is the canonical bad trace.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCHEMA = "tnsr.mesh_sim_trace.v0"

# The abstract phase sequence StepTxn walks for a committed step and for an
# aborted step, over a single logical rank. These are the *target* behaviors we
# check the observed trace against.
_COMMIT_PHASES = (
    "computing",
    "synchronizing",
    "prepared",
    "committing",
    "committed",
)
_ABORT_PHASES = (
    "computing",
    "synchronizing",
    "prepared",
    "aborting",
    "aborted",
)


class TraceError(ValueError):
    """Raised when a trace cannot be translated into a StepTxn behavior."""


def classify_trace(trace: dict) -> dict:
    """Reduce a mesh_sim_trace.v0 trace to the StepTxn-relevant observations.

    Returns a dict with:
      * ``saw_collective`` -- any observed collective (reached synchronizing)
      * ``saw_failure``    -- any observed failure (abort cause)
      * ``saw_optimizer``  -- any observed optimizer-phase event (commit path)
      * ``decision``       -- "abort" if a failure was seen else "commit"
      * ``phases``         -- the *observed* abstract phase sequence to replay
      * ``commit_after_failure`` -- the illegal case (optimizer despite failure)
    """
    if trace.get("schema") != SCHEMA:
        raise TraceError(f"expected schema {SCHEMA!r}, got {trace.get('schema')!r}")
    events = trace.get("events")
    if not isinstance(events, list):
        raise TraceError("trace has no 'events' array")

    saw_collective = False
    saw_failure = False
    saw_optimizer = False
    for ev in events:
        kind = ev.get("kind")
        phase = ev.get("phase")
        if kind == "collective":
            saw_collective = True
        elif kind == "failure":
            saw_failure = True
        # Ignored variants (documented above): layout_transition, injection.
        if phase == "optimizer":
            saw_optimizer = True

    # A Failure forces the abort decision (mirrors StepTxn.Decide: abort iff any
    # rank observed an overflow). The optimizer phase is the commit signal.
    #
    # The *observed* phase sequence is what the runtime actually did:
    #   - a failure drives the abort path (aborting -> aborted);
    #   - an optimizer phase drives the commit path (committing -> committed).
    # StepTxn only permits ONE of these. When the trace observed BOTH (a
    # failure AND an optimizer commit), we still replay the commit path it
    # recorded so TLC catches the contradiction: overflow forces `Decide` to
    # "abort", so the observed `committing` phase is unreachable and the
    # behavior deadlocks before consuming the trace (TraceConsumed fails).
    commit_after_failure = saw_failure and saw_optimizer
    decision = "abort" if saw_failure else "commit"
    if saw_optimizer:
        phases = _COMMIT_PHASES
    elif saw_failure:
        phases = _ABORT_PHASES
    else:
        # No terminal signal observed: fall back to the commit path so a bare
        # collective-only trace still resolves.
        phases = _COMMIT_PHASES
    return {
        "saw_collective": saw_collective,
        "saw_failure": saw_failure,
        "saw_optimizer": saw_optimizer,
        "decision": decision,
        "phases": phases,
        # The canonical illegal trace: an abort cause (failure) yet the runtime
        # still ran the optimizer/commit path. StepTxn forbids this.
        "commit_after_failure": commit_after_failure,
    }


def _tla_phase_seq(phases: Sequence[str]) -> str:
    quoted = ", ".join(f'"{p}"' for p in phases)
    return f"<<{quoted}>>"


def render_behavior_module(stem: str, obs: dict) -> str:
    """Render a self-contained TLA+ module that drives StepTxn along the
    observed phase sequence and asserts StepTxn.Inv plus TraceConsumed.

    The module walks one logical rank ``r0`` through the observed phase list. It
    binds ``overflow[r0]`` to whether a failure was observed, so ``Decide`` picks
    the same decision the trace implies. If the observed sequence is illegal
    (e.g. it reaches a committing phase after a failure forced abort), the guard
    on the commit action is unsatisfiable and the behavior deadlocks before
    ``k`` reaches the end -- ``TraceConsumed`` then fails.
    """
    overflow_init = "TRUE" if obs["saw_failure"] else "FALSE"
    phase_seq = _tla_phase_seq(obs["phases"])
    n = len(obs["phases"])
    return f"""--------------------------- MODULE {stem} ---------------------------
\\* GENERATED by formal/distributed_training/trace_bridge.py -- do not edit.
\\* Trace-refinement harness: drives StepTxn along one observed mesh_sim_trace.v0
\\* step and checks the observation is a legal StepTxn behavior (issue 08).
\\*
\\* observed decision : {obs["decision"]}
\\* saw_collective    : {obs["saw_collective"]}
\\* saw_failure       : {obs["saw_failure"]}
\\* saw_optimizer     : {obs["saw_optimizer"]}
EXTENDS Naturals, Sequences, FiniteSets

TraceRanks == {{"r0"}}
ObservedPhases == {phase_seq}
ObservedLen == {n}
ObservedOverflow == {overflow_init}

VARIABLES phase, overflow, decision, version, optim_mutated, k

txnVars == <<phase, overflow, decision, version, optim_mutated>>
vars == <<phase, overflow, decision, version, optim_mutated, k>>

S == INSTANCE StepTxn WITH
  Ranks <- TraceRanks,
  JobGen <- 1,
  Epoch <- 1,
  LogicalStep <- 1,
  AccumWindow <- 1,
  Attempt <- 0,
  InitVersion <- 0,
  phase <- phase,
  overflow <- overflow,
  decision <- decision,
  version <- version,
  optim_mutated <- optim_mutated

\\* The next observed phase the trace expects rank r0 to enter.
NextObservedPhase == ObservedPhases[k]

\\* Compute for the trace bridge PINS the overflow flag to what the trace
\\* observed (ObservedOverflow) rather than re-picking it nondeterministically
\\* like StepTxn.Compute. A recorded trace is a concrete observation: a Failure
\\* event means overflow=TRUE (abort), its absence means overflow=FALSE (commit).
\\* Binding it here is what makes Decide take the decision the trace implies.
ComputeObserved(r) ==
  /\\ phase[r] = "idle"
  /\\ phase' = [phase EXCEPT ![r] = "computing"]
  /\\ overflow' = [overflow EXCEPT ![r] = ObservedOverflow]
  /\\ UNCHANGED <<decision, version, optim_mutated>>

\\* Advance one StepTxn action, but only if it lands on the phase the trace
\\* observed next. Sync/Prepare/Commit/Committed/Abort/Aborted are the underlying
\\* StepTxn transitions; ComputeObserved is the overflow-pinned local step. When
\\* the observed phase is unreachable under the abstract contract (e.g. a
\\* "committing" phase after overflow forced the abort decision), none of the
\\* disjuncts is enabled and the behavior deadlocks before k passes ObservedLen
\\* -- TraceConsumed then fails.
StepToObserved ==
  /\\ k <= ObservedLen
  /\\ k' = k + 1
  /\\ LET target == NextObservedPhase IN
       \\/ (target = "computing"     /\\ ComputeObserved("r0"))
       \\/ (target = "synchronizing" /\\ S!Sync("r0"))
       \\/ (target = "prepared"      /\\ S!Prepare("r0"))
       \\/ (target = "committing"    /\\ S!Commit("r0"))
       \\/ (target = "committed"     /\\ S!Committed("r0"))
       \\/ (target = "aborting"      /\\ S!Abort("r0"))
       \\/ (target = "aborted"       /\\ S!Aborted("r0"))

\\* The single global decision, taken once all ranks are prepared. This is the
\\* only action that mutates `decision`; k is untouched (Decide is not part of
\\* the observed phase list).
DecideStep ==
  /\\ S!Decide
  /\\ UNCHANGED k

Init ==
  /\\ phase = [r \\in TraceRanks |-> "idle"]
  /\\ overflow = [r \\in TraceRanks |-> ObservedOverflow]
  /\\ decision = "none"
  /\\ version = 0
  /\\ optim_mutated = FALSE
  /\\ k = 1

\\* A terminal stutter so the behavior never deadlocks: once every observed
\\* transition has been consumed (or the trace got stuck), the system idles.
\\* This keeps the *refinement* check (TraceConsumed) as an INVARIANT rather
\\* than relying on TLC's deadlock detector, so an illegal trace surfaces as a
\\* named invariant violation with a counterexample -- matching the StepTxn/
\\* MeshPlan good/bad contract exactly.
Stutter == UNCHANGED vars

Next ==
  \\/ StepToObserved
  \\/ DecideStep
  \\/ Stutter

Spec == Init /\\ [][Next]_vars

\\* Refinement obligations:
\\*   Inv           -- the observed behavior never breaks the abstract contract.
\\*   TraceConsumed -- the whole observed sequence is realizable and, once the
\\*                    observed transitions run out, the step has resolved. An
\\*                    illegal trace (e.g. optimizer/commit after a Failure forced
\\*                    the abort decision) gets stuck: Commit is disabled under an
\\*                    abort decision, so k stalls below ObservedLen with no
\\*                    resolution -- TraceStuck flips and TraceConsumed is
\\*                    violated with a counterexample.
Inv == S!Inv
\\* The trace stalled if we can no longer advance (StepToObserved disabled) yet
\\* have not consumed the whole observed sequence.
CanAdvance ==
  /\\ k <= ObservedLen
  /\\ LET target == ObservedPhases[k] IN
       \\/ (target = "computing"     /\\ ENABLED ComputeObserved("r0"))
       \\/ (target = "synchronizing" /\\ ENABLED S!Sync("r0"))
       \\/ (target = "prepared"      /\\ ENABLED S!Prepare("r0"))
       \\/ (target = "committing"    /\\ ENABLED S!Commit("r0"))
       \\/ (target = "committed"     /\\ ENABLED S!Committed("r0"))
       \\/ (target = "aborting"      /\\ ENABLED S!Abort("r0"))
       \\/ (target = "aborted"       /\\ ENABLED S!Aborted("r0"))
TraceConsumed ==
  \\/ CanAdvance                 \\* still making progress, obligation not yet due
  \\/ ENABLED DecideStep         \\* waiting on the single global decision
  \\/ (k > ObservedLen /\\ S!Resolved)  \\* consumed the whole trace, resolved
TypeOK == /\\ S!TypeOK /\\ k \\in 1..(ObservedLen + 1)

=============================================================================
"""


def _bad_config_note(obs: dict) -> str:
    if obs["commit_after_failure"]:
        return (
            "\\* This trace recorded an optimizer/commit phase after a Failure, "
            "which StepTxn forbids."
        )
    return ""


def render_config(stem: str, obs: dict) -> str:
    """Render the TLC .cfg. Both good and bad use the same shape: check Inv (the
    abstract StepTxn contract) and TraceConsumed (the trace-refinement
    obligation) as state invariants. A legal trace keeps both true in every
    reachable state; an illegal trace violates TraceConsumed with a
    counterexample."""
    note = _bad_config_note(obs)
    prefix = f"{note}\n" if note else ""
    return f"""{prefix}SPECIFICATION Spec

INVARIANT Inv
INVARIANT TraceConsumed
"""


def translate(trace_path: Path, out_dir: Path, stem: str) -> dict:
    trace = json.loads(trace_path.read_text())
    obs = classify_trace(trace)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.tla").write_text(render_behavior_module(stem, obs))
    (out_dir / f"{stem}.cfg").write_text(render_config(stem, obs))
    return obs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="mesh_sim_trace.v0 JSON path")
    parser.add_argument("--out-dir", required=True, help="output dir for <stem>.tla/.cfg")
    parser.add_argument("--stem", required=True, help="generated module stem (e.g. TraceGood)")
    args = parser.parse_args(argv)

    try:
        obs = translate(Path(args.trace), Path(args.out_dir), args.stem)
    except TraceError as exc:
        print(f"trace error: {exc}", file=sys.stderr)
        return 2
    print(
        f"translated {args.trace} -> {args.stem}: decision={obs['decision']} "
        f"collective={obs['saw_collective']} failure={obs['saw_failure']} "
        f"optimizer={obs['saw_optimizer']} commit_after_failure={obs['commit_after_failure']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
