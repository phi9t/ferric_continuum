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
# This verifier NEVER presents a SKIP as a PASS. Real-checkpoint parity is
# skipped when no local weights are configured, and reported as SKIP.
#
# Env:
#   DEEPSEEK_V41_MODEL_DIR / MODEL_DIR  real checkpoint dir (optional)
#   HF_PYTHON   python with torch+numpy+safetensors
#               (default: main-checkout .venv-hf; worktrees lack it)
#   OUT_DIR     scratch dir for JSON dumps (default: /tmp/dsv41_text_compat)
#   BAZEL       bazel binary (default: ~/.local/bin/bazel-9.2.0)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# The CPU venv lives in the MAIN checkout, not in worktrees. Default to it.
MAIN_VENV="/data02/home/philip.yang/workspace/ferric_continuum/.venv-hf/bin/python"
HF_PYTHON="${HF_PYTHON:-$MAIN_VENV}"
OUT_DIR="${OUT_DIR:-/tmp/dsv41_text_compat}"
BAZEL="${BAZEL:-$HOME/.local/bin/bazel-9.2.0}"
TOOLS="$REPO_ROOT/ferric_continuum/tnsr/tools"
MODEL_DIR="${DEEPSEEK_V41_MODEL_DIR:-${MODEL_DIR:-}}"

mkdir -p "$OUT_DIR"

# Per-level result accounting. Values: PASS / SKIP / FAIL.
declare -A RESULT
ORDER=()
record() { RESULT["$1"]="$2"; ORDER+=("$1"); }

hr() { echo "------------------------------------------------------------"; }

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
echo "==> building deepseek_v41_infer (opt)"
if ! "$BAZEL" build -c opt \
    --@rules_rust//rust/settings:extra_rustc_flags=-Copt-level=3 \
    //ferric_continuum/tnsr:deepseek_v41_infer; then
  record "L6-tiny-parity" FAIL
else
  INFER_BIN="$REPO_ROOT/bazel-bin/ferric_continuum/tnsr/deepseek_v41_infer"
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
    if ! "$HF_PYTHON" "$TOOLS/deepseek_v41_compare_logits.py" \
        "$RUST_JSON" "$REF_JSON"; then
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
  if "$INFER_BIN" --model-dir "$MODEL_DIR" --token-ids 1,2,3 \
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
