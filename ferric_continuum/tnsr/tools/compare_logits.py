#!/usr/bin/env python3
"""Compare tnsr and HuggingFace last-position logits for Qwen3.

Both inputs are JSON files with the schema::

    {"token_ids": [...], "prompt": "...", "vocab_size": V, "logits": [f32; V]}

(one produced by ``qwen3_infer --dump-logits``, the other by
``hf_reference.py``).  Reports:

* whether the two token-id sequences match (flags a tokenizer divergence),
* ``max_abs_diff``, ``mean_abs_diff``, cosine similarity over the [V] vectors,
* top-1 argmax agreement and top-5 id overlap,
* a PASS/FAIL verdict against tolerances.

numpy-only, so it runs even in environments without torch/transformers.
"""

import argparse
import json
import sys

import numpy as np

# f32-CPU (tnsr) vs f32-eager (HF) on logits of O(10) should agree closely.
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
    p.add_argument("tnsr_json", help="tnsr logits JSON (qwen3_infer --dump-logits)")
    p.add_argument("hf_json", help="HF logits JSON (hf_reference.py)")
    args = p.parse_args()

    t = load(args.tnsr_json)
    h = load(args.hf_json)

    tl = t["logits"]
    hl = h["logits"]

    print("=" * 60)
    print(f"tnsr : {args.tnsr_json}")
    print(f"HF   : {args.hf_json}")
    print("-" * 60)

    ids_match = t["token_ids"] == h["token_ids"]
    print(f"token_ids match : {ids_match}")
    if not ids_match:
        print(f"  tnsr ids : {t['token_ids']}")
        print(f"  HF   ids : {h['token_ids']}")
        print(
            "  NOTE: tokenizations differ; re-run the tnsr dump with "
            "--token-ids <HF ids> to isolate tokenizer vs model math."
        )

    if tl.shape != hl.shape:
        print(
            f"FAIL: vocab size mismatch (tnsr={tl.shape[0]}, HF={hl.shape[0]})"
        )
        return 1

    diff = np.abs(tl - hl)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    cos = cosine(tl, hl)

    t_top1 = int(np.argmax(tl))
    h_top1 = int(np.argmax(hl))
    top1_match = t_top1 == h_top1

    t_top5 = set(np.argsort(tl)[-5:].tolist())
    h_top5 = set(np.argsort(hl)[-5:].tolist())
    top5_overlap = len(t_top5 & h_top5)

    print(f"max_abs_diff    : {max_abs:.6g}")
    print(f"mean_abs_diff   : {mean_abs:.6g}")
    print(f"cosine_sim      : {cos:.8f}")
    print(f"top1 (tnsr/HF)  : {t_top1} / {h_top1}  -> match={top1_match}")
    print(f"top5 overlap    : {top5_overlap}/5")

    # Verdict: model math is what we assert on.  A tokenizer mismatch is
    # reported but does not itself fail the run (the driver re-runs with HF
    # ids to test the math directly).
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
    raise SystemExit(main())
