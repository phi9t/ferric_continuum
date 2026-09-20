#!/usr/bin/env bash
# Verify tnsr's Rust DeepSeek V4.1 text-only forward against the ladder of
# evidence levels. This is the Level 1-6 verifier for the text-only path.
#
#   Level 1  source claims        -> deepseek_v41_verify_sources.py
#   Level 2  config/index parse    -> deepseek_v41_config_tests (Bazel)
#   Level 3  op fixtures           -> deepseek_v41_math_tests   (Bazel)
#   Level 4  layer/block/model     -> deepseek_v41_model_tests  (Bazel)
#   Level 5  loader round-trip     -> covered by model_tests loader cases
#   Level 6  tiny logits parity    -> emit tiny checkpoint, dump Rust logits,
#                                      recompute reference logits, compare
#   Level 6* real-checkpoint parity -> SKIP unless DEEPSEEK_V41_MODEL_DIR /
#                                       MODEL_DIR points at real weights
#
# This verifier NEVER presents a SKIP as a PASS. It also runs negative-control
# self-tests first (P0): the logits comparator must FAIL a perturbed row and
# ERROR a non-finite one, and the source verifier must FAIL a missing claim.
# Levels needing the reference interpreter (6) SKIP honestly when the CPU venv
# is absent, rather than failing or silently passing. Real-checkpoint parity is
# skipped when no local weights are configured, and reported as SKIP.
#
# Env:
#   DEEPSEEK_V41_MODEL_DIR / MODEL_DIR  real checkpoint dir (optional)
#   HF_PYTHON   python with torch+numpy+safetensors
#               (default: main-checkout .venv-hf; worktrees lack it)
#   OUT_DIR     scratch dir for JSON dumps
#               (default: $TMPDIR/deepseek-v41-text-compat)
#   BAZEL       bazel binary (default: bazel; version pinned by .bazelversion)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# The CPU venv lives in the MAIN checkout, not in worktrees. Default to it.
MAIN_VENV="${HOME}/workspace/ferric_continuum/.venv-hf/bin/python"
HF_PYTHON="${HF_PYTHON:-$MAIN_VENV}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/deepseek-v41-text-compat}"
BAZEL="${BAZEL:-bazel}"
TOOLS="$REPO_ROOT/ferric_continuum/tnsr/tools"
MODEL_DIR="${DEEPSEEK_V41_MODEL_DIR:-${MODEL_DIR:-}}"
INFER_BIN="$REPO_ROOT/bazel-bin/ferric_continuum/tnsr/deepseek_v41_infer"

mkdir -p "$OUT_DIR"

# Per-level result accounting. Values: PASS / SKIP / FAIL.
declare -A RESULT
ORDER=()
record() { RESULT["$1"]="$2"; ORDER+=("$1"); }

hr() { echo "------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# Preflight: is HF_PYTHON a usable reference interpreter? The Level-6 parity
# step needs torch+numpy+safetensors. If the CPU venv is missing we record that
# level as an honest SKIP (never FAIL, never a silent PASS) so a bare checkout
# without the venv does not masquerade as verified.
# ---------------------------------------------------------------------------
HF_OK=1
HF_REASON=""
if [ ! -x "$HF_PYTHON" ] && ! command -v "$HF_PYTHON" >/dev/null 2>&1; then
  HF_OK=0
  HF_REASON="HF_PYTHON '$HF_PYTHON' is not an executable interpreter"
elif ! "$HF_PYTHON" - <<'PY' >/dev/null 2>&1
import numpy, torch, safetensors  # noqa: F401
PY
then
  HF_OK=0
  HF_REASON="HF_PYTHON '$HF_PYTHON' lacks numpy/torch/safetensors"
fi
if [ "$HF_OK" -eq 0 ]; then
  echo "==> Preflight: reference interpreter unavailable ($HF_REASON)"
  echo "    Level 6 tiny parity will be recorded as SKIP."
fi

# ---------------------------------------------------------------------------
# Preflight self-check: prove the logits comparator actually bites (identical
# PASSes, perturbed FAILs, non-finite ERRORs). A parity gate that cannot fail
# is worthless; this negative control guards against that. The comparator
# imports numpy, so run it through the same interpreter validated for reference
# work.
# ---------------------------------------------------------------------------
echo "==> Preflight: comparator negative control (self-test)"
if [ "$HF_OK" -eq 0 ]; then
  echo "    SKIP: no numpy-capable interpreter ($HF_REASON); cannot run comparator self-test."
  record "P0-comparator-selftest" SKIP
elif "$HF_PYTHON" "$TOOLS/deepseek_v41_compare_logits.py" --self-test >/dev/null 2>&1; then
  record "P0-comparator-selftest" PASS
else
  echo "    FAIL: comparator self-test did not behave (perturbed row not rejected)."
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
# Level 1: source claims
# ---------------------------------------------------------------------------
echo "==> Level 1: source claims"
if "$TOOLS/verify_deepseek_v41_sources.sh"; then
  record "L1-sources" PASS
else
  record "L1-sources" FAIL
fi

# ---------------------------------------------------------------------------
# Levels 2-5: config / op / layer / block / loader Bazel tests
# ---------------------------------------------------------------------------
echo "==> Level 2: config/index parse (deepseek_v41_config_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_config_tests; then
  record "L2-config" PASS
else
  record "L2-config" FAIL
fi

echo "==> Level 3: op fixtures (deepseek_v41_math_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_math_tests; then
  record "L3-ops" PASS
else
  record "L3-ops" FAIL
fi

echo "==> Levels 4-5: layer/block/model + loader (deepseek_v41_model_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_model_tests; then
  record "L4-model+loader" PASS
else
  record "L4-model+loader" FAIL
fi

# ---------------------------------------------------------------------------
# Level 6: tiny logits parity (upstream-executed reference on a tiny model)
# ---------------------------------------------------------------------------
echo "==> Level 6: tiny logits parity"
if [ "$HF_OK" -eq 0 ]; then
  echo "    SKIP: no reference interpreter ($HF_REASON); cannot compute reference."
  record "L6-tiny-parity" SKIP
elif ! "$BAZEL" build -c opt \
    --@rules_rust//rust/settings:extra_rustc_flags=-Copt-level=3 \
    //ferric_continuum/tnsr:deepseek_v41_infer; then
  echo "==> building deepseek_v41_infer (opt)"
  record "L6-tiny-parity" FAIL
else
  TINY_CKPT="$OUT_DIR/tiny_ckpt"
  REF_JSON="$OUT_DIR/tiny_ref.json"
  RUST_JSON="$OUT_DIR/tiny_rust.json"
  rm -rf "$TINY_CKPT"

  tiny_ok=1
  echo "==> emitting tiny checkpoint + reference logits"
  if ! "$HF_PYTHON" "$TOOLS/deepseek_v41_reference.py" \
      --repo-root "$REPO_ROOT" \
      --emit-tiny-checkpoint "$TINY_CKPT" \
      --token-ids 1,3 \
      --out "$REF_JSON"; then
    tiny_ok=0
  fi

  if [ "$tiny_ok" -eq 1 ]; then
    echo "==> tnsr dump (tiny)"
    if ! "$INFER_BIN" --model-dir "$TINY_CKPT" --token-ids 1,3 \
        --text-only --dump-logits "$RUST_JSON"; then
      tiny_ok=0
    fi
  fi

  if [ "$tiny_ok" -eq 1 ]; then
    echo "==> compare (tiny)"
    # The comparator exits 0=PASS, 1=numeric FAIL, 2=harness/malformed ERROR.
    # Only exit 0 counts as parity; both 1 and 2 fail this level.
    "$HF_PYTHON" "$TOOLS/deepseek_v41_compare_logits.py" \
        "$RUST_JSON" "$REF_JSON"
    cmp_rc=$?
    if [ "$cmp_rc" -eq 2 ]; then
      echo "    ERROR: comparator reported a malformed/non-finite logits input."
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
# Level 6*: real-checkpoint parity (SKIP: no faithful CPU reference exists)
# ---------------------------------------------------------------------------
echo "==> Level 6*: real-checkpoint parity"
if [ -z "$MODEL_DIR" ]; then
  echo "    SKIP: no DEEPSEEK_V41_MODEL_DIR / MODEL_DIR set (no local weights)."
  record "L6-real-parity" SKIP
elif [ ! -d "$MODEL_DIR" ]; then
  echo "    SKIP: MODEL_DIR '$MODEL_DIR' does not exist."
  record "L6-real-parity" SKIP
else
  # A real checkpoint is present, but there is no faithful upstream reference
  # on this CPU host (the tiny reference only reproduces the helper-based tiny
  # forward, and upstream model.py needs GPU-only kernels). We can still prove
  # the Rust loader consumes the checkpoint and produces finite logits, but
  # this is a smoke check, not numeric parity -> reported as SKIP, never PASS.
  REAL_RUST="$OUT_DIR/real_rust.json"
  echo "==> tnsr dump (real, --token-ids 1,2,3) [smoke only]"
  if [ ! -x "$INFER_BIN" ]; then
    "$BAZEL" build -c opt \
      --@rules_rust//rust/settings:extra_rustc_flags=-Copt-level=3 \
      //ferric_continuum/tnsr:deepseek_v41_infer || true
  fi
  if [ -x "$INFER_BIN" ] && "$INFER_BIN" --model-dir "$MODEL_DIR" --token-ids 1,2,3 \
      --text-only --dump-logits "$REAL_RUST"; then
    echo "    Rust loader consumed the checkpoint and dumped logits."
    echo "    SKIP: no faithful CPU reference for numeric parity (out of scope)."
    record "L6-real-parity" SKIP
  else
    echo "    Rust loader FAILED to consume the real checkpoint."
    record "L6-real-parity" FAIL
  fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "DeepSeek V4.1 text-only verifier summary"
hr
fail=0
skip=0
for k in "${ORDER[@]}"; do
  v="${RESULT[$k]}"
  printf "  %-20s %s\n" "$k" "$v"
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
