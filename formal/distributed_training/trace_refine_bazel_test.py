#!/usr/bin/env python3
"""Hermetic Bazel driver for the mesh-trace -> StepTxn refinement check (issue 08).

Given a good and a bad recorded ``tnsr.mesh_sim_trace.v0`` trace (data deps), this
driver:

  1. translates each trace into a self-contained TLA+ behavior module that drives
     ``StepTxn.tla`` along the observed step (via ``trace_bridge.py``),
  2. runs each generated module under the pinned ``@tla2tools//jar`` with Bazel's
     own JDK (the same hermetic scaffolding as ``tlc_bazel_test.py``), and
  3. asserts the refinement contract: the good trace is a legal StepTxn behavior
     (``result == "success"``) and the bad trace violates ``TraceConsumed``
     (``result == "invariant_violation"``) with a counterexample.

The generated ``.tla`` and the shared ``StepTxn.tla`` base must sit in the same
directory (StepTxn is INSTANCE'd), so the driver stages them into a temp dir
next to a copy of ``StepTxn.tla`` from the runfiles.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from python.runfiles import runfiles

from formal.distributed_training import trace_bridge
from tools import tla_check


def _rlocation(rf: runfiles.Runfiles, path: str) -> Path:
    resolved = rf.Rlocation(path)
    if not resolved or not Path(resolved).exists():
        raise SystemExit(f"runfiles path not found: {path} -> {resolved}")
    return Path(resolved)


def _resolve_java(rf: runfiles.Runfiles, java_arg: str) -> Path:
    java = Path(java_arg)
    if java.is_file():
        return java
    for candidate in (java_arg, java_arg.removeprefix("external/")):
        resolved = rf.Rlocation(candidate)
        if resolved and Path(resolved).is_file():
            return Path(resolved)
    raise SystemExit(f"could not resolve $(JAVA): {java_arg}")


def _find(rf: runfiles.Runfiles, rel: str) -> Path:
    resolved = rf.Rlocation(f"_main/formal/distributed_training/{rel}")
    if resolved and Path(resolved).is_file():
        return Path(resolved)
    here = Path(__file__).resolve().parent
    if (here / rel).is_file():
        return here / rel
    raise SystemExit(f"could not locate {rel} in runfiles")


def _check_one(checker, tla: Path, cfg: Path, out_dir: Path):
    exit_code = tla_check.run_checker(checker, tla, cfg, out_dir)
    evidence = tla_check.json.loads((out_dir / "tla-check.json").read_text())
    return exit_code, evidence


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("java_bin", help="Bazel $(JAVA) launcher path")
    parser.add_argument("jar", help="Path to tla2tools.jar")
    parser.add_argument("--good", required=True, help="good trace JSON basename")
    parser.add_argument("--bad", required=True, help="bad trace JSON basename")
    args = parser.parse_args(argv[1:])

    rf = runfiles.Create()
    if rf is None:
        raise SystemExit("runfiles not available")

    java_bin = _resolve_java(rf, args.java_bin)
    jar = Path(args.jar)
    if not jar.is_file():
        jar = _rlocation(rf, args.jar)

    step_txn = _find(rf, "StepTxn.tla")
    good_trace = _find(rf, args.good)
    bad_trace = _find(rf, args.bad)

    checker = tla_check.discover_checker(jar_arg=str(jar), java_arg=str(java_bin))
    if checker.kind != "tlc":
        raise SystemExit(f"expected a tlc checker, got: {checker.kind} ({checker.message})")

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy(step_txn, stage / "StepTxn.tla")

        good_obs = trace_bridge.translate(good_trace, stage, "TraceGood")
        bad_obs = trace_bridge.translate(bad_trace, stage, "TraceBad")
        print(f"[bridge] good: {good_obs}")
        print(f"[bridge] bad:  {bad_obs}")

        # Sanity: the bad fixture must be the canonical illegal case so the
        # check is meaningful (never present a trivially-passing bad as PASS).
        if not bad_obs["commit_after_failure"]:
            failures.append(
                "bad trace fixture is not the illegal commit-after-failure case; "
                "the refinement check would be vacuous"
            )

        good_exit, good = _check_one(
            checker, stage / "TraceGood.tla", stage / "TraceGood.cfg", stage / "good_out"
        )
        print(f"[good] exit={good_exit} result={good['result']} ok={good['ok']}")
        if not (good_exit == 0 and good["ok"] and good["result"] == "success"):
            failures.append("good trace did not refine StepTxn cleanly")
            print((stage / "good_out" / "stdout.log").read_text(), file=sys.stderr)

        bad_exit, bad = _check_one(
            checker, stage / "TraceBad.tla", stage / "TraceBad.cfg", stage / "bad_out"
        )
        print(f"[bad]  exit={bad_exit} result={bad['result']} ok={bad['ok']}")
        if bad["result"] != "invariant_violation" or bad["ok"]:
            failures.append("bad trace did not surface the expected TraceConsumed violation")
            print((stage / "bad_out" / "stdout.log").read_text(), file=sys.stderr)
        if "counterexample" not in bad:
            failures.append("bad trace evidence is missing a counterexample path")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("PASS: good trace refines StepTxn; bad trace violates TraceConsumed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
