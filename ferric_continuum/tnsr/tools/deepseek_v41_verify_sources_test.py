#!/usr/bin/env python3
"""Bazel test for the DeepSeek V4.1 source-claim manifest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from python.runfiles import runfiles


REQUIRED_CLAIMS = {
    "report-ced-decoder-global-kv",
    "report-csa2-sharing-semantics",
    "report-v4-csa-vs-v41-csa2",
    "report-swa-every-layer-bounded-replay",
    "report-hierarchical-indexer",
    "model-indexer-v32-origin-and-v41-ops",
    "model-swa-ring-cache",
    "engram-hash-mask-and-compressed-map",
    "model-engram-table-lookup",
    "model-dspark-forward-spec-start-pos",
    "model-dspark-attention-topk-main-x",
    "report-dspark-mtp-training-contrast",
}


def _rlocation(rf: runfiles.Runfiles, path: str) -> Path:
    for candidate in (f"_main/{path}", f"ferric_continuum/{path}", path):
        resolved = rf.Rlocation(candidate)
        if resolved and Path(resolved).exists():
            return Path(resolved)
    raise AssertionError(f"runfiles path not found: {path}")


def _load_verifier(verifier_path: Path):
    spec = importlib.util.spec_from_file_location("deepseek_v41_verify_sources", verifier_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import verifier: {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repo_root_from_runfiles(rf: runfiles.Runfiles) -> Path:
    readme_path = _rlocation(
        rf, "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/README.md"
    )
    for parent in readme_path.parents:
        if (
            parent / "ferric_continuum/tnsr/testdata/deepseek_v41/source_manifest.json"
        ).exists() and (
            parent / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/README.md"
        ).exists():
            return parent
    raise AssertionError(f"could not derive repository root from runfiles path: {readme_path}")


def main() -> int:
    rf = runfiles.Create()
    if rf is None:
        raise AssertionError("runfiles not available")

    verifier_path = _rlocation(rf, "ferric_continuum/tnsr/tools/deepseek_v41_verify_sources.py")
    manifest_path = _rlocation(rf, "ferric_continuum/tnsr/testdata/deepseek_v41/source_manifest.json")
    repo_root = _repo_root_from_runfiles(rf)

    verifier = _load_verifier(verifier_path)
    claims = verifier.load_manifest(manifest_path)
    claim_ids = {claim.claim_id for claim in claims}
    missing_claim_ids = sorted(REQUIRED_CLAIMS - claim_ids)
    if missing_claim_ids:
        raise AssertionError(f"manifest missing faithful-coverage claims: {missing_claim_ids}")

    checked, errors = verifier.verify_claims(repo_root, manifest_path)
    if errors:
        raise AssertionError("source manifest verification failed:\n" + "\n".join(errors))
    if not REQUIRED_CLAIMS.issubset(set(checked)):
        raise AssertionError("required claims were present but not checked")

    print(f"PASS: checked {len(checked)} DeepSeek V4.1 source claim(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
