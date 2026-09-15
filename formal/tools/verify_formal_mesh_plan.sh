#!/usr/bin/env bash
# Verify the distributed-training TLA+ formal models with a hermetic, Bazel-run
# TLC (tla2tools.jar + Bazel's own JDK). No host `java` or `TLA_TOOLS_JAR` is
# required.
#
#   Level 1  runner unit tests   -> //tests/tla_check:tla_check_test
#                                   (pure-Python: checker discovery, output
#                                   classification, evidence recording).
#   Level 2  hermetic TLC check  -> //formal/distributed_training:mesh_plan_tlc_test
#                                   under --config=formal: MeshPlanGood must
#                                   complete cleanly and MeshPlanBad must surface
#                                   the expected invariant (BlockedCycle)
#                                   violation. This fetches the pinned
#                                   @tla2tools//jar (needs network at fetch time)
#                                   and runs it with remotejdk_17.
#   Level 3  hermetic TLC check  -> //formal/distributed_training:step_txn_tlc_test
#                                   under --config=formal: StepTxnGood must
#                                   satisfy Inv + StepResolves and StepTxnBad
#                                   must surface the NoMutationAfterAbort /
#                                   VersionMonotone violation (the Layer-0
#                                   step-transaction contract and refinement
#                                   target for the trace bridge).
#
# The TLC level SKIPs honestly (never PASS, never FAIL) when the jar cannot be
# fetched (e.g. no network in a sandbox) rather than masking a real failure.
#
# Env:
#   BAZEL    bazel binary (default: bazel; version pinned by .bazelversion)
#   OUT_DIR  scratch dir (default: /tmp/dsv41_formal_compat)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BAZEL="${BAZEL:-bazel}"
OUT_DIR="${OUT_DIR:-/tmp/dsv41_formal_compat}"
mkdir -p "$OUT_DIR"

declare -A RESULT
ORDER=()
record() { RESULT["$1"]="$2"; ORDER+=("$1"); }
hr() { echo "------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# Preflight: can Bazel fetch the pinned tla2tools jar? If not (offline sandbox),
# the hermetic TLC level is an honest SKIP rather than a FAIL.
# ---------------------------------------------------------------------------
echo "==> Preflight: fetch pinned @tla2tools//jar"
JAR_OK=1
JAR_REASON=""
if ! "$BAZEL" build @tla2tools//jar >"$OUT_DIR/fetch.log" 2>&1; then
  JAR_OK=0
  JAR_REASON="could not fetch @tla2tools//jar (see $OUT_DIR/fetch.log)"
  echo "    $JAR_REASON"
  echo "    Level 2 hermetic TLC check will be recorded as SKIP."
fi

# ---------------------------------------------------------------------------
# Level 1: runner unit tests (pure Python, no jar/JDK needed).
# ---------------------------------------------------------------------------
echo "==> Level 1: tla_check runner unit tests"
if "$BAZEL" test //tests/tla_check:tla_check_test; then
  record "L1-runner-tests" PASS
else
  record "L1-runner-tests" FAIL
fi

# ---------------------------------------------------------------------------
# Level 2: hermetic TLC model check (good clean + bad invariant violation).
# ---------------------------------------------------------------------------
echo "==> Level 2: hermetic TLC MeshPlan check"
if [ "$JAR_OK" -eq 0 ]; then
  echo "    SKIP: $JAR_REASON."
  record "L2-hermetic-tlc" SKIP
elif "$BAZEL" test --config=formal \
    //formal/distributed_training:mesh_plan_tlc_test; then
  record "L2-hermetic-tlc" PASS
else
  record "L2-hermetic-tlc" FAIL
fi

# ---------------------------------------------------------------------------
# Level 3: hermetic TLC step-transaction check (good clean + bad violation).
# ---------------------------------------------------------------------------
echo "==> Level 3: hermetic TLC StepTxn check"
if [ "$JAR_OK" -eq 0 ]; then
  echo "    SKIP: $JAR_REASON."
  record "L3-step-txn-tlc" SKIP
elif "$BAZEL" test --config=formal \
    //formal/distributed_training:step_txn_tlc_test; then
  record "L3-step-txn-tlc" PASS
else
  record "L3-step-txn-tlc" FAIL
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "Distributed-training formal (TLA+/TLC) verifier summary"
hr
fail=0
skip=0
for k in "${ORDER[@]}"; do
  v="${RESULT[$k]}"
  printf "  %-24s %s\n" "$k" "$v"
  case "$v" in
    FAIL) fail=1 ;;
    SKIP) skip=1 ;;
  esac
done
hr
if [ "$fail" -ne 0 ]; then
  echo "RESULT: FAIL (one or more levels failed)"
  exit 1
fi
if [ "$skip" -ne 0 ]; then
  echo "RESULT: PASS-WITH-SKIPS (skipped levels are NOT counted as pass)"
  exit 0
fi
echo "RESULT: PASS (all levels)"
exit 0
