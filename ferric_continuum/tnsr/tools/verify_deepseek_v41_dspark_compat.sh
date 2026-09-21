#!/usr/bin/env bash
# Verify tnsr's Rust DeepSeek V4.1 *DSpark* speculative head against the ladder
# of evidence levels. This is the Level 1-6 verifier for the DSpark path
# (Wave 3).
#
#   Level 1  source claims        -> deepseek_v41_verify_sources.py
#                                     (incl. Transformer.forward_spec / MTP /
#                                      Markov / confidence head claims)
#   Level 2  dspark config parse   -> deepseek_v41_config_tests (Bazel)
#                                     (n_mtp_layers / dspark_block_size /
#                                      target layers / dspark MoE geometry)
#   Level 3  dspark op fixtures     -> deepseek_v41_math_tests   (Bazel)
#                                     (markov / confidence / draft-input /
#                                      main_proj_norm / topk / draft-loop)
#   Level 4  dspark head + loader   -> deepseek_v41_model_tests  (Bazel)
#            forward_spec parity        (tiny_dspark_head_forward_spec_matches_*
#                                        + mtp.* loader cases)
#   Level 6  tiny forward_spec       -> run the Rust DSpark fixture dumper and
#            Rust-vs-reference           compare output_ids/logits/confidence
#                                        against the recomputed Python reference
#   Level 6* real-checkpoint parity  -> SKIP unless DEEPSEEK_V41_MODEL_DIR /
#                                        MODEL_DIR points at real weights
#
# Why no `deepseek_v41_infer --dump-logits` DSpark path?  The CLI intentionally
# REJECTS `--dspark` / `--speculative` / `--mtp` (exit 2): a text checkpoint has
# n_mtp=0 so there is nothing to draft, and DSpark decode is out of the text CLI
# scope.  Level 6 therefore uses a dedicated Rust fixture dumper that executes
# `DeepSeekV41DsparkHead::forward_spec` over `dspark_tiny_model_fixture.json`
# and compares that Rust JSON directly with the Python reference.
#
# This verifier NEVER presents a SKIP as a PASS. It also runs negative-control
# self-tests first (P0): the logits comparator must FAIL a perturbed row, FAIL a
# moved DSpark output id, FAIL an out-of-tolerance confidence, and ERROR a
# non-finite / missing field; the source verifier must FAIL a missing claim.
# Levels needing the reference interpreter (6) SKIP honestly when the CPU venv
# is absent, rather than failing or silently passing.
#
# Env:
#   DEEPSEEK_V41_MODEL_DIR / MODEL_DIR  real checkpoint dir (optional)
#   HF_PYTHON   python with torch+numpy+safetensors
#               (default: main-checkout .venv-hf; worktrees lack it)
#   OUT_DIR     scratch dir for JSON dumps (default: /tmp/dsv41_dspark_compat)
#   BAZEL       bazel binary (default: bazel; version pinned by .bazelversion)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# The CPU venv lives in the MAIN checkout, not in worktrees. Default to it.
MAIN_VENV="${HOME}/workspace/ferric_continuum/.venv-hf/bin/python"
HF_PYTHON="${HF_PYTHON:-$MAIN_VENV}"
OUT_DIR="${OUT_DIR:-/tmp/dsv41_dspark_compat}"
BAZEL="${BAZEL:-bazel}"
TOOLS="$REPO_ROOT/ferric_continuum/tnsr/tools"
MODEL_DIR="${DEEPSEEK_V41_MODEL_DIR:-${MODEL_DIR:-}}"
FIXTURE="$REPO_ROOT/ferric_continuum/tnsr/testdata/deepseek_v41/dspark_tiny_model_fixture.json"

mkdir -p "$OUT_DIR"

# Per-level result accounting. Values: PASS / SKIP / FAIL.
declare -A RESULT
ORDER=()
record() { RESULT["$1"]="$2"; ORDER+=("$1"); }

hr() { echo "------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# Preflight: is HF_PYTHON a usable reference interpreter? The Level-6
# reference-consistency step needs numpy (and imports the fixture generator).
# If the CPU venv is missing we record that level as an honest SKIP (never FAIL,
# never a silent PASS) so a bare checkout without the venv does not masquerade
# as verified.
# ---------------------------------------------------------------------------
HF_OK=1
HF_REASON=""
if [ ! -x "$HF_PYTHON" ] && ! command -v "$HF_PYTHON" >/dev/null 2>&1; then
  HF_OK=0
  HF_REASON="HF_PYTHON '$HF_PYTHON' is not an executable interpreter"
elif ! "$HF_PYTHON" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
PY
then
  HF_OK=0
  HF_REASON="HF_PYTHON '$HF_PYTHON' lacks numpy"
fi
if [ "$HF_OK" -eq 0 ]; then
  echo "==> Preflight: reference interpreter unavailable ($HF_REASON)"
  echo "    Level 6 tiny parity will be recorded as SKIP."
fi

# ---------------------------------------------------------------------------
# Preflight self-check: prove the logits comparator actually bites, including
# the DSpark output_ids/confidence gates (moved id FAILs, out-of-tol confidence
# FAILs, missing confidence ERRORs). The comparator imports numpy, so run it
# through the same interpreter validated for reference work.
# ---------------------------------------------------------------------------
echo "==> Preflight: comparator negative control (self-test, incl. dspark)"
if [ "$HF_OK" -eq 0 ]; then
  echo "    SKIP: no numpy-capable interpreter ($HF_REASON); cannot run comparator self-test."
  record "P0-comparator-selftest" SKIP
elif "$HF_PYTHON" "$TOOLS/deepseek_v41_compare_logits.py" --self-test >/dev/null 2>&1; then
  record "P0-comparator-selftest" PASS
else
  echo "    FAIL: comparator self-test did not behave (dspark gate not biting)."
  record "P0-comparator-selftest" FAIL
fi

# Likewise prove the source verifier fails a deliberately-missing claim.
echo "==> Preflight: source-verifier negative control (self-test)"
if python3 "$TOOLS/deepseek_v41_verify_sources.py" --self-test-negative >/dev/null 2>&1; then
  record "P0-sources-selftest" PASS
else
  echo "    FAIL: source-verifier self-test did not fail a missing claim."
  record "P0-sources-selftest" FAIL
fi

# ---------------------------------------------------------------------------
# Level 1: source claims (text + dspark forward_spec / MTP / Markov / conf)
# ---------------------------------------------------------------------------
echo "==> Level 1: source claims"
if "$TOOLS/verify_deepseek_v41_sources.sh"; then
  record "L1-sources" PASS
else
  record "L1-sources" FAIL
fi

# ---------------------------------------------------------------------------
# Levels 2-4: config / op / model Bazel tests (cover dspark config activation,
# dspark op fixtures, and the forward_spec + mtp.* loader model cases)
# ---------------------------------------------------------------------------
echo "==> Level 2: dspark config parse (deepseek_v41_config_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_config_tests; then
  record "L2-config" PASS
else
  record "L2-config" FAIL
fi

echo "==> Level 3: dspark op fixtures (deepseek_v41_math_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_math_tests; then
  record "L3-ops" PASS
else
  record "L3-ops" FAIL
fi

echo "==> Level 4: dspark head + mtp.* loader (deepseek_v41_model_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_model_tests; then
  record "L4-model+loader" PASS
else
  record "L4-model+loader" FAIL
fi

# ---------------------------------------------------------------------------
# Level 6: tiny forward_spec Rust-vs-reference parity. The Python reference
# recomputes the DSpark forward_spec (output_ids / logits / confidence) from
# the shared `generate_dspark_model` helper. The Rust side is produced by a
# dedicated Bazel binary that executes `DeepSeekV41DsparkHead::forward_spec`.
# ---------------------------------------------------------------------------
echo "==> Level 6: tiny forward_spec Rust-vs-reference parity"
if [ "$HF_OK" -eq 0 ]; then
  echo "    SKIP: no reference interpreter ($HF_REASON); cannot compute reference."
  record "L6-tiny-parity" SKIP
elif [ ! -f "$FIXTURE" ]; then
  echo "    FAIL: tracked dspark_tiny_model_fixture.json missing at $FIXTURE"
  record "L6-tiny-parity" FAIL
else
  REF_JSON="$OUT_DIR/dspark_ref.json"
  TNSR_JSON="$OUT_DIR/dspark_tnsr.json"
  tiny_ok=1

  echo "==> recomputing DSpark reference (output_ids/logits/confidence)"
  if ! "$HF_PYTHON" "$TOOLS/deepseek_v41_reference.py" \
      --repo-root "$REPO_ROOT" --dspark --out "$REF_JSON"; then
    tiny_ok=0
  fi

  if [ "$tiny_ok" -eq 1 ]; then
    echo "==> running Rust DSpark fixture dumper"
    if ! "$BAZEL" build //ferric_continuum/tnsr:deepseek_v41_dspark_fixture_dump; then
      echo "    ERROR: could not build Rust DSpark fixture dumper."
      tiny_ok=0
    elif ! "$REPO_ROOT/bazel-bin/ferric_continuum/tnsr/deepseek_v41_dspark_fixture_dump" \
        "$FIXTURE" >"$TNSR_JSON"; then
      echo "    ERROR: could not run Rust DSpark fixture dumper."
      tiny_ok=0
    fi
  fi

  if [ "$tiny_ok" -eq 1 ]; then
    echo "==> compare (tiny dspark: logits + output_ids + confidence)"
    # exits 0=PASS, 1=numeric/id/confidence FAIL, 2=harness/malformed ERROR.
    "$HF_PYTHON" "$TOOLS/deepseek_v41_compare_logits.py" \
        --dspark "$TNSR_JSON" "$REF_JSON"
    cmp_rc=$?
    if [ "$cmp_rc" -eq 2 ]; then
      echo "    ERROR: comparator reported a malformed/non-finite input."
      tiny_ok=0
    elif [ "$cmp_rc" -ne 0 ]; then
      tiny_ok=0
    fi
  fi

  if [ "$tiny_ok" -eq 1 ]; then
    record "L6-tiny-parity" PASS
  else
    record "L6-tiny-parity" FAIL
  fi
fi

# ---------------------------------------------------------------------------
# Level 6*: real-checkpoint DSpark parity (SKIP: no faithful CPU reference and
# no real weights). A real DSpark decode needs the upstream GPU kernels and the
# released 510GB checkpoint; both are out of scope on this CPU host.
# ---------------------------------------------------------------------------
echo "==> Level 6*: real-checkpoint DSpark parity"
if [ -z "$MODEL_DIR" ]; then
  echo "    SKIP: no DEEPSEEK_V41_MODEL_DIR / MODEL_DIR set (no local weights)."
  record "L6-real-parity" SKIP
elif [ ! -d "$MODEL_DIR" ]; then
  echo "    SKIP: MODEL_DIR '$MODEL_DIR' does not exist."
  record "L6-real-parity" SKIP
else
  echo "    SKIP: no faithful CPU reference for real DSpark parity"
  echo "          (upstream forward_spec needs GPU-only kernels; out of scope)."
  record "L6-real-parity" SKIP
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "DeepSeek V4.1 DSpark verifier summary"
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
