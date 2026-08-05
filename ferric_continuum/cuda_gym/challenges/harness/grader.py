"""Grader for CUDA gym challenges (Python standard library only).

A challenge ships two binaries built from the same problem contract:

  * ``student``   — links ``student.cu`` (TODO stubs; fails until filled in)
  * ``reference`` — links ``reference.cu`` (complete solution)

Each binary reads a single case description as JSON on argv[1] and prints the
result as JSON on stdout:

    in : {"case": {...}, "seed": 123}         # emitted per-case by this grader
    out: {"status": 0, "shape": [m, n], "data": [...], "elapsed_ms": 1.23}

The grader loads ``cases.json``, runs the student and reference for each case,
compares element-wise within the case tolerances, prints per-case wall times,
and exits non-zero if any case fails. It deliberately avoids numpy/pytest so it
runs with only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CaseResult:
    name: str
    passed: bool
    detail: str
    student_ms: float
    reference_ms: float


def _run_binary(binary: str, case: dict, seed: int) -> dict:
    """Runs a challenge binary on one case, returning its parsed JSON output."""
    payload = json.dumps({"case": case, "seed": seed})
    proc = subprocess.run(
        [binary, payload],
        capture_output=True,
        text=True,
        timeout=case.get("timeout_s", 60),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{binary} exited {proc.returncode}\nstderr:\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"{binary} produced non-JSON output:\n{proc.stdout}\n{exc}"
        )


def _allclose(got: list, want: list, rtol: float, atol: float) -> tuple[bool, str]:
    if len(got) != len(want):
        return False, f"length mismatch: got {len(got)} want {len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
        if math.isnan(g) or math.isinf(g):
            return False, f"non-finite value {g} at index {i}"
        if abs(g - w) > atol + rtol * abs(w):
            return False, (
                f"mismatch at index {i}: got {g} want {w} "
                f"(tol {atol + rtol * abs(w):.3e})"
            )
    return True, "ok"


def grade_case(student: str, reference: str, case: dict) -> CaseResult:
    name = case.get("name", "case")
    seed = int(case.get("seed", 0))
    rtol = float(case.get("rtol", 1e-4))
    atol = float(case.get("atol", 1e-5))

    try:
        ref_out = _run_binary(reference, case, seed)
    except RuntimeError as exc:
        return CaseResult(name, False, f"reference crashed: {exc}", 0.0, 0.0)

    if ref_out.get("status", -1) != 0:
        return CaseResult(
            name, False,
            f"reference returned status {ref_out.get('status')}",
            0.0, float(ref_out.get("elapsed_ms", 0.0)),
        )

    try:
        stu_out = _run_binary(student, case, seed)
    except RuntimeError as exc:
        return CaseResult(name, False, f"student crashed: {exc}", 0.0,
                          float(ref_out.get("elapsed_ms", 0.0)))

    if stu_out.get("status", -1) != 0:
        return CaseResult(name, False,
                          f"student returned status {stu_out.get('status')}",
                          float(stu_out.get("elapsed_ms", 0.0)),
                          float(ref_out.get("elapsed_ms", 0.0)))

    if stu_out.get("shape") != ref_out.get("shape"):
        return CaseResult(
            name, False,
            f"shape mismatch: got {stu_out.get('shape')} "
            f"want {ref_out.get('shape')}",
            float(stu_out.get("elapsed_ms", 0.0)),
            float(ref_out.get("elapsed_ms", 0.0)),
        )

    passed, detail = _allclose(stu_out.get("data", []), ref_out.get("data", []),
                               rtol, atol)

    # Optional soft performance budget: reported, not enforced by default.
    budget = case.get("time_budget_ms")
    if passed and budget is not None and stu_out.get("elapsed_ms", 0.0) > budget:
        detail = (f"correct but over time budget: "
                  f"{stu_out['elapsed_ms']:.3f}ms > {budget}ms")

    return CaseResult(name, passed, detail,
                      float(stu_out.get("elapsed_ms", 0.0)),
                      float(ref_out.get("elapsed_ms", 0.0)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a CUDA gym challenge.")
    parser.add_argument("--cases", required=True, help="path to cases.json")
    parser.add_argument("--student", required=True, help="student binary")
    parser.add_argument("--reference", required=True, help="reference binary")
    args = parser.parse_args(argv)

    with open(args.cases, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    cases = spec["cases"] if isinstance(spec, dict) else spec

    results = [grade_case(args.student, args.reference, c) for c in cases]

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        all_passed = all_passed and r.passed
        print(f"[{status}] {r.name}: {r.detail} "
              f"(student {r.student_ms:.3f}ms, reference {r.reference_ms:.3f}ms)")

    if not all_passed:
        print("Challenge FAILED: fill in student.cu until all cases pass.")
        return 1
    print("Challenge PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
