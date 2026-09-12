#!/usr/bin/env python3
"""Compare tnsr and upstream DeepSeek V4.1 last-position logits.

Both inputs are JSON files with the schema::

    {"token_ids": [...], "prompt": "...", "vocab_size": V,
     "model_type": "deepseek_v41_text", "logits": [f32; V], ...}

(one produced by ``deepseek_v41_infer --dump-logits``, the other by
``deepseek_v41_reference.py``).  Reports:

* whether the two token-id sequences match (flags a tokenizer divergence),
* shape equality,
* ``max_abs_diff``, ``mean_abs_diff``, cosine similarity over the [V] vectors,
* top-1 argmax agreement and top-5 id overlap,
* a PASS/FAIL verdict against tolerances.

numpy-only, so it runs even in environments without torch/transformers.
"""

import argparse
import json
import sys

import numpy as np

# f32-CPU (tnsr) vs f32-eager (upstream) on logits of O(10) should agree
# closely.  These mirror the Qwen3 comparator thresholds.
COSINE_MIN = 0.999
TOP5_OVERLAP_MIN = 4  # of 5


def load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    d["logits"] = np.asarray(d["logits"], dtype=np.float64)
    return d


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "tnsr_json", help="tnsr logits JSON (deepseek_v41_infer --dump-logits)"
    )
    p.add_argument(
        "reference_json", help="upstream logits JSON (deepseek_v41_reference.py)"
    )
    args = p.parse_args()

    t = load(args.tnsr_json)
    r = load(args.reference_json)

    tl = t["logits"]
    rl = r["logits"]

    print("=" * 60)
    print(f"tnsr      : {args.tnsr_json}")
    print(f"reference : {args.reference_json}")
    print(f"model_type: {t.get('model_type')} / {r.get('model_type')}")
    if r.get("reference_backend") is not None:
        print(f"backend   : {r.get('reference_backend')}")
    if r.get("kernel_patch") is not None:
        print(f"kernel_patch: {r.get('kernel_patch')}")
    print("-" * 60)

    ids_match = t["token_ids"] == r["token_ids"]
    print(f"token_ids match : {ids_match}")
    if not ids_match:
        print(f"  tnsr ids : {t['token_ids']}")
        print(f"  ref  ids : {r['token_ids']}")
        ref_ids = ",".join(str(i) for i in r["token_ids"])
        print(
            "  NOTE: tokenizations differ; re-run the tnsr dump with "
            f"--token-ids {ref_ids} to isolate tokenizer vs model math."
        )

    if tl.shape != rl.shape:
        print(f"FAIL: vocab size mismatch (tnsr={tl.shape[0]}, ref={rl.shape[0]})")
        return 1

    diff = np.abs(tl - rl)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    cos = cosine(tl, rl)

    t_top1 = int(np.argmax(tl))
    r_top1 = int(np.argmax(rl))
    top1_match = t_top1 == r_top1

    t_top5 = set(np.argsort(tl)[-5:].tolist())
    r_top5 = set(np.argsort(rl)[-5:].tolist())
    top5_overlap = len(t_top5 & r_top5)

    print(f"max_abs_diff    : {max_abs:.6g}")
    print(f"mean_abs_diff   : {mean_abs:.6g}")
    print(f"cosine_sim      : {cos:.8f}")
    print(f"top1 (tnsr/ref) : {t_top1} / {r_top1}  -> match={top1_match}")
    print(f"top5 overlap    : {top5_overlap}/5")

    # Verdict: model math is what we assert on.  A tokenizer mismatch is
    # reported but does not itself fail the run (the driver re-runs with
    # reference ids to test the math directly).
    math_ok = top1_match and cos >= COSINE_MIN and top5_overlap >= TOP5_OVERLAP_MIN
    verdict = "PASS" if math_ok else "FAIL"
    print("-" * 60)
    print(
        f"{verdict}  (cosine>={COSINE_MIN}, top1 match, "
        f"top5 overlap>={TOP5_OVERLAP_MIN})"
    )
    print("=" * 60)
    return 0 if math_ok else 1


if __name__ == "__main__":
    sys.exit(main())
