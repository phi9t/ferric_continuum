#!/usr/bin/env python3
"""Hermetic Bazel driver for the MeshPlan TLC model checks.

This wraps ``tools/tla_check.py`` so both the good and bad distributed-training
mesh models are checked with Bazel's own JDK against the pinned
``@tla2tools//jar``. It is intentionally free of any host ``java`` / TLA
assumptions:

  * the tla2tools jar arrives as a data dep and is located via the runfiles
    library,
  * the Java launcher is Bazel's ``$(JAVA)`` make-variable (the current Java
    runtime toolchain), passed in as a runfiles-relative path and resolved
    through the same runfiles library,
  * the ``.tla`` / ``.cfg`` fixtures arrive as data deps in the same runfiles
    directory (``MeshPlan.tla`` must sit next to the Good/Bad modules because
    they ``INSTANCE MeshPlan``).

Contract:
  * ``MeshPlanGood`` must complete with no error (``result == "success"``).
  * ``MeshPlanBad`` must violate ``Inv`` (``result == "invariant_violation"``);
    TLC exits nonzero for this, which is the *expected* outcome, so we assert on
    the classified result rather than the raw exit code.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from python.runfiles import runfiles  # rules_python runfiles library

# tla_check ships as a py_library on the same PYTHONPATH (see BUILD.bazel).
from tools import tla_check


def _rlocation(rf: runfiles.Runfiles, path: str) -> Path:
    resolved = rf.Rlocation(path)
    if not resolved or not Path(resolved).exists():
        raise SystemExit(f"runfiles path not found: {path} -> {resolved}")
    return Path(resolved)


def _resolve_java(rf: runfiles.Runfiles, java_arg: str) -> Path:
    """Resolve Bazel's $(JAVA) make-var to a real launcher path.

    $(JAVA) is emitted as a runfiles-relative path such as
    ``external/rules_java++.../bin/java``. Map it into this test's runfiles.
    """
    java = Path(java_arg)
    if java.is_file():
        return java
    # Bazel prefixes external runfiles with the workspace name; the runfiles
    # library expects a repo-qualified key. Try common shapes.
    for candidate in (java_arg, java_arg.removeprefix("external/")):
        resolved = rf.Rlocation(candidate)
        if resolved and Path(resolved).is_file():
            return Path(resolved)
    raise SystemExit(f"could not resolve $(JAVA): {java_arg}")


def _fixture_dir(rf: runfiles.Runfiles) -> Path:
    good = rf.Rlocation("_main/formal/distributed_training/MeshPlanGood.tla")
    if good and Path(good).is_file():
        return Path(good).parent
    # Fallback: co-located with this script in the runfiles tree.
    here = Path(__file__).resolve().parent
    if (here / "MeshPlanGood.tla").is_file():
        return here
    raise SystemExit("MeshPlan fixtures not found in runfiles")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: tlc_bazel_test.py <java-bin> <tla2tools.jar>")
    rf = runfiles.Create()
    if rf is None:
        raise SystemExit("runfiles not available")

    java_bin = _resolve_java(rf, argv[1])
    jar = Path(argv[2])
    if not jar.is_file():
        jar = _rlocation(rf, argv[2])
    fixtures = _fixture_dir(rf)

    checker = tla_check.discover_checker(jar_arg=str(jar), java_arg=str(java_bin))
    if checker.kind != "tlc":
        raise SystemExit(f"expected a tlc checker, got: {checker.kind} ({checker.message})")

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        out_root = Path(tmp)

        good_exit = tla_check.run_checker(
            checker,
            fixtures / "MeshPlanGood.tla",
            fixtures / "MeshPlanGood.cfg",
            out_root / "good",
        )
        good = tla_check.json.loads((out_root / "good" / "tla-check.json").read_text())
        print(f"[good] exit={good_exit} result={good['result']} ok={good['ok']}")
        if not (good_exit == 0 and good["ok"] and good["result"] == "success"):
            failures.append("MeshPlanGood did not complete cleanly")
            print((out_root / "good" / "stdout.log").read_text(), file=sys.stderr)

        bad_exit = tla_check.run_checker(
            checker,
            fixtures / "MeshPlanBad.tla",
            fixtures / "MeshPlanBad.cfg",
            out_root / "bad",
        )
        bad = tla_check.json.loads((out_root / "bad" / "tla-check.json").read_text())
        print(f"[bad]  exit={bad_exit} result={bad['result']} ok={bad['ok']}")
        # The bad model is *expected* to break Inv; TLC exits nonzero (12) for
        # that, so success here means "found the invariant violation".
        if bad["result"] != "invariant_violation" or bad["ok"]:
            failures.append("MeshPlanBad did not surface the expected invariant violation")
            print((out_root / "bad" / "stdout.log").read_text(), file=sys.stderr)
        if "counterexample" not in bad:
            failures.append("MeshPlanBad evidence is missing a counterexample path")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("PASS: MeshPlanGood clean, MeshPlanBad invariant violation detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
