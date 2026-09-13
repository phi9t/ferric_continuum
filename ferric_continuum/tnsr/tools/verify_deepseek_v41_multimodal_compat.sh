#!/usr/bin/env bash
# Verify tnsr's Rust DeepSeek V4.1 *multimodal* forward against the ladder of
# evidence levels. This is the Level 1-6 verifier for the vision path (Wave 2).
#
#   Level 1  source claims        -> deepseek_v41_verify_sources.py
#                                     (incl. vision.py / image_processor.py /
#                                      model.py merge+bias_vl claims)
#   Level 2  vision config parse   -> deepseek_v41_config_tests (Bazel)
#   Level 3  vision op fixtures     -> deepseek_v41_math_tests   (Bazel)
#   Level 4  ViT/Aligner/merge/     -> deepseek_v41_model_tests  (Bazel)
#            image-mask routing        (vision + grid + merge cases)
#   Level 5  vl-prompt encoding      -> regenerate prompt_vl_fixture via upstream
#                                        image_processor and diff against tracked
#   Level 6  tiny MM logits parity   -> emit tiny vision checkpoint, dump Rust
#                                        multimodal logits, recompute faithful
#                                        upstream ViT/Aligner reference, compare
#   Level 6* real-checkpoint parity  -> SKIP unless DEEPSEEK_V41_MODEL_DIR /
#                                        MODEL_DIR points at real weights
#   B200     GPU tiny parity         -> PASS if a CUDA torch runtime is present,
#                                        else SKIP (this CPU host reports SKIP)
#
# This verifier NEVER presents a SKIP as a PASS.
#
# Env:
#   DEEPSEEK_V41_MODEL_DIR / MODEL_DIR  real checkpoint dir (optional)
#   HF_PYTHON   python with torch+numpy+safetensors
#               (default: main-checkout .venv-hf; worktrees lack it)
#   OUT_DIR     scratch dir for JSON dumps (default: /tmp/dsv41_mm_compat)
#   BAZEL       bazel binary (default: ~/.local/bin/bazel-9.2.0)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# The CPU venv lives in the MAIN checkout, not in worktrees. Default to it.
MAIN_VENV="/data02/home/philip.yang/workspace/ferric_continuum/.venv-hf/bin/python"
HF_PYTHON="${HF_PYTHON:-$MAIN_VENV}"
OUT_DIR="${OUT_DIR:-/tmp/dsv41_mm_compat}"
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
# Level 1: source claims (text + vision + image-processor + merge/bias_vl)
# ---------------------------------------------------------------------------
echo "==> Level 1: source claims"
if "$TOOLS/verify_deepseek_v41_sources.sh"; then
  record "L1-sources" PASS
else
  record "L1-sources" FAIL
fi

# ---------------------------------------------------------------------------
# Levels 2-4: config / op / model Bazel tests (cover vision config, vision op
# fixtures, and ViT/Aligner/merge/image-mask routing model cases)
# ---------------------------------------------------------------------------
echo "==> Level 2: vision config parse (deepseek_v41_config_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_config_tests; then
  record "L2-config" PASS
else
  record "L2-config" FAIL
fi

echo "==> Level 3: vision op fixtures (deepseek_v41_math_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_math_tests; then
  record "L3-ops" PASS
else
  record "L3-ops" FAIL
fi

echo "==> Level 4: ViT/Aligner/merge/image-mask (deepseek_v41_model_tests)"
if "$BAZEL" test //ferric_continuum/tnsr:deepseek_v41_model_tests; then
  record "L4-model" PASS
else
  record "L4-model" FAIL
fi

# ---------------------------------------------------------------------------
# Level 5: vl-prompt encoding parity (regenerate via upstream image_processor
# and diff against the tracked fixture).
# ---------------------------------------------------------------------------
echo "==> Level 5: vl-prompt encoding parity"
VL_FIXTURE="$REPO_ROOT/ferric_continuum/tnsr/testdata/deepseek_v41/prompt_vl_fixture.json"
if [ ! -f "$VL_FIXTURE" ]; then
  echo "    FAIL: tracked prompt_vl_fixture.json missing at $VL_FIXTURE"
  record "L5-vl-prompt" FAIL
else
  VL_BEFORE="$OUT_DIR/vl_before.json"
  cp "$VL_FIXTURE" "$VL_BEFORE"
  if "$HF_PYTHON" "$TOOLS/deepseek_v41_fixture_gen.py" \
      --repo-root "$REPO_ROOT" --family vl-prompt \
      && diff -q "$VL_BEFORE" "$VL_FIXTURE" >/dev/null; then
    echo "    regenerated vl-prompt fixture matches tracked bytes."
    record "L5-vl-prompt" PASS
  else
    echo "    FAIL: regenerated vl-prompt fixture differs from tracked bytes."
    # Restore the tracked fixture so the tree is not left dirty on failure.
    cp "$VL_BEFORE" "$VL_FIXTURE"
    record "L5-vl-prompt" FAIL
  fi
fi

# ---------------------------------------------------------------------------
# Level 6: tiny multimodal logits parity (faithful upstream ViT/Aligner ref)
# ---------------------------------------------------------------------------
echo "==> Level 6: tiny multimodal logits parity"
echo "==> building deepseek_v41_infer (opt)"
INFER_BIN="$REPO_ROOT/bazel-bin/ferric_continuum/tnsr/deepseek_v41_infer"
if ! "$BAZEL" build -c opt \
    --@rules_rust//rust/settings:extra_rustc_flags=-Copt-level=3 \
    //ferric_continuum/tnsr:deepseek_v41_infer; then
  record "L6-tiny-mm-parity" FAIL
else
  MM_CKPT="$OUT_DIR/tiny_mm_ckpt"
  MM_PATCHES="$OUT_DIR/tiny_mm_patches.json"
  MM_REF="$OUT_DIR/tiny_mm_ref.json"
  MM_RUST="$OUT_DIR/tiny_mm_rust.json"
  MM_IDS="3,4,4,4,4,4,4,4,4,2"
  MM_TYPES="-1,0,1,1,2,1,1,2,3,-1"
  rm -rf "$MM_CKPT"

  mm_ok=1
  echo "==> emitting tiny MM checkpoint + patches + reference logits"
  if ! "$HF_PYTHON" "$TOOLS/deepseek_v41_reference.py" \
      --repo-root "$REPO_ROOT" \
      --emit-tiny-mm-checkpoint "$MM_CKPT" \
      --emit-image-patches "$MM_PATCHES" \
      --multimodal \
      --out "$MM_REF"; then
    mm_ok=0
  fi

  if [ "$mm_ok" -eq 1 ]; then
    echo "==> tnsr dump (tiny multimodal)"
    if ! "$INFER_BIN" --model-dir "$MM_CKPT" --multimodal \
        --token-ids "$MM_IDS" --token-types "$MM_TYPES" \
        --image-patches "$MM_PATCHES" --max-new-tokens 0 \
        --dump-logits "$MM_RUST"; then
      mm_ok=0
    fi
  fi

  if [ "$mm_ok" -eq 1 ]; then
    echo "==> compare (tiny multimodal)"
    if ! "$HF_PYTHON" "$TOOLS/deepseek_v41_compare_logits.py" \
        "$MM_RUST" "$MM_REF"; then
      mm_ok=0
    fi
  fi

  if [ "$mm_ok" -eq 1 ]; then
    record "L6-tiny-mm-parity" PASS
  else
    record "L6-tiny-mm-parity" FAIL
  fi
fi

# ---------------------------------------------------------------------------
# B200: GPU tiny multimodal parity. The tiny reference runs identical math on
# CPU or GPU; we only distinguish provenance by the detected torch device. On a
# CUDA host we re-run the reference under torch.cuda and re-compare; on this CPU
# host (torch cpu-only) there is nothing to run, so this is an honest SKIP.
# ---------------------------------------------------------------------------
echo "==> B200: GPU tiny multimodal parity"
GPU_DEVICE="$("$HF_PYTHON" - <<'PY'
try:
    import torch
    print(f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu-no-torch")
PY
)"
if [[ "$GPU_DEVICE" == cuda:* ]]; then
  echo "    detected $GPU_DEVICE; re-running tiny MM reference on GPU device"
  GPU_REF="$OUT_DIR/tiny_mm_ref_gpu.json"
  if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$HF_PYTHON" "$TOOLS/deepseek_v41_reference.py" \
        --repo-root "$REPO_ROOT" --multimodal --out "$GPU_REF" \
      && "$HF_PYTHON" "$TOOLS/deepseek_v41_compare_logits.py" \
        "$OUT_DIR/tiny_mm_rust.json" "$GPU_REF"; then
    record "B200-gpu-parity" PASS
  else
    record "B200-gpu-parity" FAIL
  fi
else
  echo "    SKIP: no CUDA torch runtime on this host (device=$GPU_DEVICE)."
  record "B200-gpu-parity" SKIP
fi

# ---------------------------------------------------------------------------
# Level 6*: real-checkpoint parity (SKIP: no faithful CPU reference exists)
# ---------------------------------------------------------------------------
echo "==> Level 6*: real-checkpoint multimodal parity"
if [ -z "$MODEL_DIR" ]; then
  echo "    SKIP: no DEEPSEEK_V41_MODEL_DIR / MODEL_DIR set (no local weights)."
  record "L6-real-parity" SKIP
elif [ ! -d "$MODEL_DIR" ]; then
  echo "    SKIP: MODEL_DIR '$MODEL_DIR' does not exist."
  record "L6-real-parity" SKIP
else
  echo "    SKIP: no faithful CPU reference for real multimodal parity"
  echo "          (upstream model.py needs GPU-only kernels; out of scope)."
  record "L6-real-parity" SKIP
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "DeepSeek V4.1 multimodal verifier summary"
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
