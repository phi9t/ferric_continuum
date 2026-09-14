#!/usr/bin/env bash
# Verify tnsr's Rust Qwen3 forward is numerically compatible with upstream
# HuggingFace `Qwen3ForCausalLM`, by comparing full-vocabulary last-position
# logits for a set of prompts.
#
#   1. build the `qwen3_infer` binary (opt),
#   2. for each prompt: dump tnsr logits -> tnsr_logits_<i>.json,
#   3. run the HF reference     -> hf_logits_<i>.json,
#   4. compare each pair,
#   5. if token-ids differ, re-run the tnsr dump with HF's ids and re-compare,
#      isolating any tnsr-BPE approximation from a model-math mismatch.
#
# Env:
#   MODEL_DIR   local Qwen3 checkpoint (default: ~/models/qwen3-0.6b)
#   HF_PYTHON   python with torch+transformers (default: .venv-hf/bin/python)
#   OUT_DIR     scratch dir for JSON dumps (default: /tmp/tnsr_hf_compat)
#   BAZEL       bazel binary (default: bazel; version pinned by .bazelversion)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3-0.6b}"
HF_PYTHON="${HF_PYTHON:-$REPO_ROOT/.venv-hf/bin/python}"
OUT_DIR="${OUT_DIR:-/tmp/tnsr_hf_compat}"
BAZEL="${BAZEL:-bazel}"
TOOLS="$REPO_ROOT/ferric_continuum/tnsr/tools"

mkdir -p "$OUT_DIR"

PROMPTS=(
  "The capital of France is"
  "Water is made of hydrogen and"
  "Once upon a time, there was a"
)

echo "==> building qwen3_infer (opt)"
"$BAZEL" build -c opt \
  --@rules_rust//rust/settings:extra_rustc_flags=-Copt-level=3 \
  //ferric_continuum/tnsr:qwen3_infer

INFER_BIN="$REPO_ROOT/bazel-bin/ferric_continuum/tnsr/qwen3_infer"

fail=0
for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  tnsr_json="$OUT_DIR/tnsr_logits_$i.json"
  hf_json="$OUT_DIR/hf_logits_$i.json"

  echo
  echo "############################################################"
  echo "# prompt[$i]: $prompt"
  echo "############################################################"

  echo "==> tnsr dump"
  "$INFER_BIN" --model-dir "$MODEL_DIR" --prompt "$prompt" \
    --max-new-tokens 0 --dump-logits "$tnsr_json"

  echo "==> HF reference"
  "$HF_PYTHON" "$TOOLS/hf_reference.py" --model-dir "$MODEL_DIR" \
    --prompt "$prompt" --out "$hf_json"

  echo "==> compare"
  if ! python3 "$TOOLS/compare_logits.py" "$tnsr_json" "$hf_json"; then
    # If tokenizations differ, re-run tnsr with HF's ids to test the math.
    hf_ids="$(python3 -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1]))['token_ids'])))" "$hf_json")"
    tnsr_ids="$(python3 -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1]))['token_ids'])))" "$tnsr_json")"
    if [ "$hf_ids" != "$tnsr_ids" ]; then
      echo "==> tokenizations differ; re-running tnsr with HF ids"
      tnsr_json2="$OUT_DIR/tnsr_logits_${i}_hfids.json"
      "$INFER_BIN" --model-dir "$MODEL_DIR" --prompt "$prompt" \
        --token-ids "$hf_ids" --max-new-tokens 0 --dump-logits "$tnsr_json2"
      echo "==> compare (HF ids fed into tnsr)"
      if ! python3 "$TOOLS/compare_logits.py" "$tnsr_json2" "$hf_json"; then
        fail=1
      fi
    else
      fail=1
    fi
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "ALL PROMPTS PASS"
else
  echo "ONE OR MORE PROMPTS FAILED"
fi
exit "$fail"
