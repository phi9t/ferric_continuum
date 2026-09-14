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

Exit codes are distinct so a driver can tell a *harness* problem (bad JSON,
missing keys, non-finite logits, vocab mismatch) apart from a genuine *numeric*
divergence of the model math:

* ``0``  PASS: math within tolerance.
* ``1``  FAIL: math diverged (cosine/top-1/top-5 below threshold).
* ``2``  ERROR: malformed input (bad JSON, missing/short logits, NaN/inf,
         vocab-size mismatch).  A parity gate must never read this as PASS.
"""

import argparse
import json
import sys

import numpy as np

# f32-CPU (tnsr) vs f32-eager (upstream) on logits of O(10) should agree
# closely.  These mirror the Qwen3 comparator thresholds.
COSINE_MIN = 0.999
TOP5_OVERLAP_MIN = 4  # of 5

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

REQUIRED_KEYS = ("token_ids", "logits")


class LogitsError(ValueError):
    """A structural problem with a logits JSON file (not a numeric divergence)."""


def load(path: str) -> dict:
    """Load and structurally validate a logits JSON file.

    Raises ``LogitsError`` (mapped to exit code 2 by the caller) on any problem
    that would make a numeric comparison meaningless: unreadable/malformed
    JSON, missing keys, a non-list/empty ``logits`` array, a ``vocab_size`` that
    disagrees with the array length, or any non-finite entry.  This keeps a
    silently-corrupt reference from being scored as a PASS.
    """
    try:
        with open(path) as f:
            d = json.load(f)
    except FileNotFoundError as exc:
        raise LogitsError(f"{path}: file not found") from exc
    except json.JSONDecodeError as exc:
        raise LogitsError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(d, dict):
        raise LogitsError(f"{path}: top-level JSON must be an object")

    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise LogitsError(f"{path}: missing required key(s): {', '.join(missing)}")

    raw_logits = d["logits"]
    if not isinstance(raw_logits, list) or not raw_logits:
        raise LogitsError(f"{path}: 'logits' must be a non-empty list")

    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.ndim != 1:
        raise LogitsError(f"{path}: 'logits' must be a flat 1-D row, got ndim={logits.ndim}")
    if not np.all(np.isfinite(logits)):
        n_bad = int(np.count_nonzero(~np.isfinite(logits)))
        raise LogitsError(f"{path}: 'logits' has {n_bad} non-finite (NaN/inf) value(s)")

    vocab = d.get("vocab_size")
    if vocab is not None:
        if not isinstance(vocab, int) or vocab <= 0:
            raise LogitsError(f"{path}: 'vocab_size' must be a positive int, got {vocab!r}")
        if vocab != logits.shape[0]:
            raise LogitsError(
                f"{path}: 'vocab_size'={vocab} disagrees with len(logits)={logits.shape[0]}"
            )

    if not isinstance(d["token_ids"], list) or not d["token_ids"]:
        raise LogitsError(f"{path}: 'token_ids' must be a non-empty list")

    d["logits"] = logits
    return d


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare(t: dict, r: dict) -> tuple[bool, dict]:
    """Return ``(math_ok, metrics)`` for two validated logits dicts.

    Raises ``LogitsError`` on a vocab-size mismatch between the two rows (an
    apples-to-oranges comparison, not a numeric divergence).
    """
    tl = t["logits"]
    rl = r["logits"]
    if tl.shape != rl.shape:
        raise LogitsError(
            f"vocab size mismatch (tnsr={tl.shape[0]}, ref={rl.shape[0]})"
        )

    diff = np.abs(tl - rl)
    t_top5 = set(np.argsort(tl)[-5:].tolist())
    r_top5 = set(np.argsort(rl)[-5:].tolist())
    metrics = {
        "ids_match": t["token_ids"] == r["token_ids"],
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cosine": cosine(tl, rl),
        "t_top1": int(np.argmax(tl)),
        "r_top1": int(np.argmax(rl)),
        "top5_overlap": len(t_top5 & r_top5),
    }
    metrics["top1_match"] = metrics["t_top1"] == metrics["r_top1"]
    math_ok = (
        metrics["top1_match"]
        and metrics["cosine"] >= COSINE_MIN
        and metrics["top5_overlap"] >= TOP5_OVERLAP_MIN
    )
    return math_ok, metrics


def run_compare(tnsr_json: str, reference_json: str) -> int:
    try:
        t = load(tnsr_json)
        r = load(reference_json)
    except LogitsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print("=" * 60)
    print(f"tnsr      : {tnsr_json}")
    print(f"reference : {reference_json}")
    print(f"model_type: {t.get('model_type')} / {r.get('model_type')}")
    if r.get("reference_backend") is not None:
        print(f"backend   : {r.get('reference_backend')}")
    if r.get("kernel_patch") is not None:
        print(f"kernel_patch: {r.get('kernel_patch')}")
    print("-" * 60)

    try:
        math_ok, m = compare(t, r)
    except LogitsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"token_ids match : {m['ids_match']}")
    if not m["ids_match"]:
        print(f"  tnsr ids : {t['token_ids']}")
        print(f"  ref  ids : {r['token_ids']}")
        ref_ids = ",".join(str(i) for i in r["token_ids"])
        print(
            "  NOTE: tokenizations differ; re-run the tnsr dump with "
            f"--token-ids {ref_ids} to isolate tokenizer vs model math."
        )

    print(f"max_abs_diff    : {m['max_abs']:.6g}")
    print(f"mean_abs_diff   : {m['mean_abs']:.6g}")
    print(f"cosine_sim      : {m['cosine']:.8f}")
    print(f"top1 (tnsr/ref) : {m['t_top1']} / {m['r_top1']}  -> match={m['top1_match']}")
    print(f"top5 overlap    : {m['top5_overlap']}/5")

    # Verdict: model math is what we assert on.  A tokenizer mismatch is
    # reported but does not itself fail the run (the driver re-runs with
    # reference ids to test the math directly).
    verdict = "PASS" if math_ok else "FAIL"
    print("-" * 60)
    print(
        f"{verdict}  (cosine>={COSINE_MIN}, top1 match, "
        f"top5 overlap>={TOP5_OVERLAP_MIN})"
    )
    print("=" * 60)
    return EXIT_PASS if math_ok else EXIT_FAIL


def run_self_test() -> int:
    """Negative + positive control: prove the comparator actually bites.

    Builds a synthetic reference row, then checks that (a) an identical row
    PASSes, (b) a perturbed row whose argmax moved FAILs, and (c) a non-finite
    row is rejected as ERROR.  Returns 0 only if all three controls behave.
    """
    import tempfile

    vocab = 8
    base = [float(i) for i in range(vocab)]  # argmax = 7

    def write(tmp: str, name: str, logits: list, ids=None) -> str:
        path = f"{tmp}/{name}"
        payload = {
            "token_ids": ids if ids is not None else [1, 2, 3],
            "prompt": "",
            "vocab_size": vocab,
            "model_type": "deepseek_v41_selftest",
            "logits": logits,
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    with tempfile.TemporaryDirectory(prefix="dsv41_compare_selftest_") as tmp:
        ref = write(tmp, "ref.json", base)

        # (a) identical -> PASS
        same = write(tmp, "same.json", base)
        if run_compare(same, ref) != EXIT_PASS:
            print("SELF-TEST FAIL: identical logits did not PASS", file=sys.stderr)
            return 1

        # (b) argmax moved to index 0 by a large margin -> FAIL
        moved = list(base)
        moved[0] = 100.0
        bad = write(tmp, "moved.json", moved)
        if run_compare(bad, ref) != EXIT_FAIL:
            print("SELF-TEST FAIL: perturbed logits did not FAIL", file=sys.stderr)
            return 1

        # (c) non-finite -> ERROR (exit 2), never PASS/FAIL
        nan_row = list(base)
        nan_row[3] = float("inf")
        nanp = write(tmp, "nan.json", nan_row)
        if run_compare(nanp, ref) != EXIT_ERROR:
            print("SELF-TEST FAIL: non-finite logits were not rejected", file=sys.stderr)
            return 1

    print("SELF-TEST PASS: identical=PASS, perturbed=FAIL, non-finite=ERROR")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "tnsr_json", nargs="?", help="tnsr logits JSON (deepseek_v41_infer --dump-logits)"
    )
    p.add_argument(
        "reference_json", nargs="?", help="upstream logits JSON (deepseek_v41_reference.py)"
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run positive+negative controls proving the comparator bites, then exit",
    )
    args = p.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.tnsr_json or not args.reference_json:
        p.error("tnsr_json and reference_json are required unless --self-test is given")

    return run_compare(args.tnsr_json, args.reference_json)


if __name__ == "__main__":
    sys.exit(main())
