#!/usr/bin/env bash
# Verify DeepSeek V4.1 source claims against mirrored upstream text files.
#
# This is the Level 1 verifier for the DeepSeek V4.1 text implementation
# ladder. It does not download or load model weights.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python3 "$REPO_ROOT/ferric_continuum/tnsr/tools/deepseek_v41_verify_sources.py" \
  --repo-root "$REPO_ROOT" \
  --manifest "$REPO_ROOT/ferric_continuum/tnsr/testdata/deepseek_v41/source_manifest.json"
