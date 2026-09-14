#!/usr/bin/env bash
# Verify tnsr's Rust DeepSeek V4.1 *performance/device* seam (Wave 4) against the
# ladder of evidence levels.
#
#   Level 1  source claims          -> deepseek_v41_verify_sources.py
#   Level 2  config/index parse      -> deepseek_v41_config_tests (Bazel)
#   Level 3  op fixtures             -> deepseek_v41_math_tests   (Bazel)
#   Level 4  layer/block/model       -> deepseek_v41_model_tests  (Bazel)
#   Level 5  cost accounting         -> lib_tests (Bazel): the symbolic
#                                       params/FLOP/byte model in cost.rs
#   Level 6  CUDA forward parity     -> deepseek_v41_cuda_forward_tests under
#                                       --config=cuda: the DeepSeek forward must
#                                       agree between the forced-CPU host path and
#                                       the default GPU path (max_abs_diff < 1e-4).
#                                       SKIP (never PASS, never FAIL) on a
#                                       CPU-only host with no CUDA runtime.
#   Level 6* real-checkpoint parity  -> SKIP unless DEEPSEEK_V41_MODEL_DIR /
#                                       MODEL_DIR points at real weights (native
#                                       FP8/FP4 kernels + 510GB weights are out of
#                                       Wave-4 scope).
#
# This verifier NEVER presents a SKIP as a PASS. It runs a comparator
# negative-control self-test first (P0). The CUDA level SKIPs honestly when no
# GPU/CUDA runtime is present rather than failing or silently passing.
#
# Env:
#   DEEPSEEK_V41_MODEL_DIR / MODEL_DIR  real checkpoint dir (optional)
#   FERRIC_TNSR_CUDA  set to 0 to force the CUDA level to SKIP (e.g. on a host
#                     without a GPU); auto-detected from nvidia-smi otherwise
#   OUT_DIR     scratch dir (default: /tmp/dsv41_perf_compat)
#   BAZEL       bazel binary (default: bazel; version pinned by .bazelversion)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-/tmp/dsv41_perf_compat}"
BAZEL="${BAZEL:-bazel}"
TOOLS="$REPO_ROOT/ferric_continuum/tnsr/tools"
MODEL_DIR="${DEEPSEEK_V41_MODEL_DIR:-${MODEL_DIR:-}}"

mkdir -p "$OUT_DIR"

# Per-level result accounting. Values: PASS / SKIP / FAIL.
declare -A RESULT
ORDER=()
record() { RESULT["$1"]="$2"; ORDER+=("$1"); }

hr() { echo "------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# CUDA preflight: only run the GPU forward-parity level when a CUDA runtime and
# a GPU are actually present. A missing GPU is an honest SKIP, never a PASS.
# ---------------------------------------------------------------------------
CUDA_OK=1
CUDA_REASON=""
if [ "${FERRIC_TNSR_CUDA:-1}" = "0" ]; then
  CUDA_OK=0
  CUDA_REASON="FERRIC_TNSR_CUDA=0 forces CPU-only"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_OK=0
  CUDA_REASON="no nvidia-smi (no GPU/CUDA runtime)"
elif ! nvidia-smi -L >/dev/null 2>&1; then
  CUDA_OK=0
  CUDA_REASON="nvidia-smi found no GPUs"
fi
if [ "$CUDA_OK" -eq 0 ]; then
  echo "==> Preflight: CUDA unavailable ($CUDA_REASON)"
  echo "    Level 6 CUDA forward parity will be recorded as SKIP."
fi

# ---------------------------------------------------------------------------
# P0: comparator negative control (self-test). numpy-only; runs with system
# python. Proves the parity comparator actually bites before we trust any level.
# ---------------------------------------------------------------------------
echo "==> Preflight: comparator negative control (self-test)"
if python3 "$TOOLS/deepseek_v41_compare_logits.py" --self-test >/dev/null 2>&1; then
  record "P0-comparator-selftest" PASS
else
  echo "    FAIL: comparator self-test did not behave (mutation not biting)."
  record "P0-comparator-selftest" FAIL
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
# Levels 2-4: config / op / model Bazel tests
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

echo "==> Level 4: layer/block/model (deepseek_v41_model_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_model_tests; then
  record "L4-model" PASS
else
  record "L4-model" FAIL
fi

# ---------------------------------------------------------------------------
# Level 5: cost accounting. The symbolic params/FLOP/byte model in cost.rs is
# unit-tested inside the lib_tests crate (block_params / block_forward /
# model_forward hand-checks + the release-shape magnitude/sparsity guard).
# ---------------------------------------------------------------------------
echo "==> Level 5: cost accounting (lib_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:lib_tests; then
  record "L5-cost-accounting" PASS
else
  record "L5-cost-accounting" FAIL
fi

# ---------------------------------------------------------------------------
# Level 6: CUDA forward-agreement parity. The DeepSeek V4.1 forward must produce
# the same logits on the GPU (cuda_ffi gemm/softmax) as on the forced-CPU host
# path. Gated by --config=cuda and the requires-gpu tag; SKIP on a CPU host.
# ---------------------------------------------------------------------------
echo "==> Level 6: CUDA forward parity (deepseek_v41_cuda_forward_tests)"
if [ "$CUDA_OK" -eq 0 ]; then
  echo "    SKIP: $CUDA_REASON."
  record "L6-cuda-forward-parity" SKIP
elif "$BAZEL" test --config=cuda \
    //ferric_continuum/tnsr:deepseek_v41_cuda_forward_tests; then
  record "L6-cuda-forward-parity" PASS
else
  record "L6-cuda-forward-parity" FAIL
fi

# ---------------------------------------------------------------------------
# Level 6*: real-checkpoint parity. Native FP8/FP4 matmul kernels and the
# released 510GB checkpoint are out of Wave-4 scope, so this is always a SKIP
# (documented, never presented as PASS).
# ---------------------------------------------------------------------------
echo "==> Level 6*: real-checkpoint perf parity"
if [ -z "$MODEL_DIR" ]; then
  echo "    SKIP: no DEEPSEEK_V41_MODEL_DIR / MODEL_DIR set (no local weights)."
  record "L6-real-parity" SKIP
elif [ ! -d "$MODEL_DIR" ]; then
  echo "    SKIP: MODEL_DIR '$MODEL_DIR' does not exist."
  record "L6-real-parity" SKIP
else
  echo "    SKIP: native FP8/FP4 kernels + TP serving parity are out of Wave-4"
  echo "          scope; no faithful real-weight perf reference on this host."
  record "L6-real-parity" SKIP
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "DeepSeek V4.1 performance/device verifier summary"
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
