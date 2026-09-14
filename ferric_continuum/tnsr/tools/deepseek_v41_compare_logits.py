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

The ``--dspark`` mode additionally asserts greedy ``output_ids`` equality and
compares the per-position ``confidence`` vector (max/mean abs diff against a
tolerance), on top of the shared logits comparison.
"""

import argparse
import json
import sys

import numpy as np

# f32-CPU (tnsr) vs f32-eager (upstream) on logits of O(10) should agree
# closely.  These mirror the Qwen3 comparator thresholds.
COSINE_MIN = 0.999
TOP5_OVERLAP_MIN = 4  # of 5
# DSpark confidence is a small scalar per draft position; the fixture pins an
# absolute tolerance of 2e-5, so allow a hair of slack for the JSON round-trip.
CONFIDENCE_ABS_TOL = 1e-4

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

REQUIRED_KEYS = ("token_ids", "logits")


class LogitsError(ValueError):
    """A structural problem with a logits JSON file (not a numeric divergence)."""


def load(path: str, dspark: bool = False) -> dict:
    """Load and structurally validate a logits JSON file.

    Raises ``LogitsError`` (mapped to exit code 2 by the caller) on any problem
    that would make a numeric comparison meaningless: unreadable/malformed
    JSON, missing keys, a non-list/empty ``logits`` array, a ``vocab_size`` that
    disagrees with the array length, or any non-finite entry.  This keeps a
    silently-corrupt reference from being scored as a PASS.

    In ``dspark`` mode the ``logits`` array is a flattened ``[block, vocab]``
    block rather than a single ``[vocab]`` row, so the ``vocab_size`` guard
    requires ``len(logits) % vocab_size == 0`` instead of exact equality.
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
        if dspark:
            if logits.shape[0] % vocab != 0:
                raise LogitsError(
                    f"{path}: len(logits)={logits.shape[0]} is not a multiple of "
                    f"'vocab_size'={vocab} (expected a flat [block, vocab] block)"
                )
        elif vocab != logits.shape[0]:
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


def _dspark_extra(t: dict, r: dict) -> tuple[bool, dict]:
    """Compare the DSpark-only ``output_ids`` and ``confidence`` fields.

    Returns ``(dspark_ok, metrics)``.  ``output_ids`` must match *exactly*
    (greedy temperature-0 draft ids are a discrete decision, not a tolerance
    band); ``confidence`` is compared as max/mean absolute difference against
    ``CONFIDENCE_ABS_TOL``.  Raises ``LogitsError`` if a required field is
    missing or malformed so a truncated reference cannot slip through as PASS.
    """
    metrics: dict = {}
    for name, obj in (("tnsr", t), ("reference", r)):
        if "output_ids" not in obj or not isinstance(obj["output_ids"], list):
            raise LogitsError(f"{name}: dspark comparison needs a list 'output_ids'")
        if "confidence" not in obj or not isinstance(obj["confidence"], list):
            raise LogitsError(f"{name}: dspark comparison needs a list 'confidence'")

    metrics["output_ids_match"] = t["output_ids"] == r["output_ids"]

    tc = np.asarray(t["confidence"], dtype=np.float64)
    rc = np.asarray(r["confidence"], dtype=np.float64)
    if tc.shape != rc.shape:
        raise LogitsError(
            f"confidence length mismatch (tnsr={tc.shape[0]}, ref={rc.shape[0]})"
        )
    if not (np.all(np.isfinite(tc)) and np.all(np.isfinite(rc))):
        raise LogitsError("confidence has non-finite (NaN/inf) value(s)")
    cdiff = np.abs(tc - rc)
    metrics["conf_max_abs"] = float(cdiff.max()) if cdiff.size else 0.0
    metrics["conf_mean_abs"] = float(cdiff.mean()) if cdiff.size else 0.0
    metrics["conf_within_tol"] = metrics["conf_max_abs"] <= CONFIDENCE_ABS_TOL

    dspark_ok = metrics["output_ids_match"] and metrics["conf_within_tol"]
    return dspark_ok, metrics


def run_compare(tnsr_json: str, reference_json: str, dspark: bool = False) -> int:
    try:
        t = load(tnsr_json, dspark=dspark)
        r = load(reference_json, dspark=dspark)
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

    dspark_ok = True
    dm: dict = {}
    if dspark:
        try:
            dspark_ok, dm = _dspark_extra(t, r)
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

    if dspark:
        print("-" * 60)
        print(f"output_ids match: {dm['output_ids_match']}")
        if not dm["output_ids_match"]:
            print(f"  tnsr output_ids : {t['output_ids']}")
            print(f"  ref  output_ids : {r['output_ids']}")
        print(f"confidence max  : {dm['conf_max_abs']:.6g}")
        print(f"confidence mean : {dm['conf_mean_abs']:.6g}")
        print(
            f"confidence tol  : {dm['conf_within_tol']} "
            f"(abs<= {CONFIDENCE_ABS_TOL:g})"
        )

    # Verdict: model math is what we assert on.  A tokenizer mismatch is
    # reported but does not itself fail the run (the driver re-runs with
    # reference ids to test the math directly).
    ok = math_ok and dspark_ok
    verdict = "PASS" if ok else "FAIL"
    print("-" * 60)
    gate = f"cosine>={COSINE_MIN}, top1 match, top5 overlap>={TOP5_OVERLAP_MIN}"
    if dspark:
        gate += f", output_ids exact, confidence abs<= {CONFIDENCE_ABS_TOL:g}"
    print(f"{verdict}  ({gate})")
    print("=" * 60)
    return EXIT_PASS if ok else EXIT_FAIL


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
    return _dspark_self_test()


def _dspark_self_test() -> int:
    """DSpark controls: prove output_ids and confidence gates actually bite.

    Builds a synthetic DSpark reference (logits + output_ids + confidence) and
    checks that (a) an identical row PASSes, (b) a moved output_id FAILs even
    when logits are unchanged, (c) a confidence perturbation past tolerance
    FAILs, and (d) a missing confidence field is rejected as ERROR.
    """
    import tempfile

    vocab = 8
    base = [float(i) for i in range(vocab)]  # argmax = 7

    def write(tmp: str, name: str, *, logits, output_ids, confidence, drop=None) -> str:
        path = f"{tmp}/{name}"
        payload = {
            "token_ids": [1],
            "prompt": "",
            "vocab_size": vocab,
            "model_type": "deepseek_v41_dspark_selftest",
            "logits": logits,
            "output_ids": output_ids,
            "confidence": confidence,
        }
        if drop is not None:
            payload.pop(drop)
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    with tempfile.TemporaryDirectory(prefix="dsv41_compare_dspark_selftest_") as tmp:
        ref = write(tmp, "ref.json", logits=base, output_ids=[1, 3, 3, 0], confidence=[0.1, 0.2, 0.3])

        # (a) identical -> PASS
        same = write(tmp, "same.json", logits=base, output_ids=[1, 3, 3, 0], confidence=[0.1, 0.2, 0.3])
        if run_compare(same, ref, dspark=True) != EXIT_PASS:
            print("SELF-TEST FAIL: identical dspark row did not PASS", file=sys.stderr)
            return 1

        # (b) output_id moved (logits unchanged) -> FAIL
        moved = write(tmp, "moved.json", logits=base, output_ids=[1, 3, 2, 0], confidence=[0.1, 0.2, 0.3])
        if run_compare(moved, ref, dspark=True) != EXIT_FAIL:
            print("SELF-TEST FAIL: moved output_id did not FAIL", file=sys.stderr)
            return 1

        # (c) confidence past tolerance -> FAIL
        conf_bad = write(tmp, "conf.json", logits=base, output_ids=[1, 3, 3, 0], confidence=[0.1, 0.2, 0.5])
        if run_compare(conf_bad, ref, dspark=True) != EXIT_FAIL:
            print("SELF-TEST FAIL: perturbed confidence did not FAIL", file=sys.stderr)
            return 1

        # (d) missing confidence -> ERROR
        no_conf = write(tmp, "noconf.json", logits=base, output_ids=[1, 3, 3, 0], confidence=[], drop="confidence")
        if run_compare(no_conf, ref, dspark=True) != EXIT_ERROR:
            print("SELF-TEST FAIL: missing confidence was not rejected", file=sys.stderr)
            return 1

    print("SELF-TEST PASS: dspark identical=PASS, moved-id=FAIL, conf=FAIL, missing=ERROR")
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
    p.add_argument(
        "--dspark",
        action="store_true",
        help="also assert greedy output_ids equality and compare confidence vectors",
    )
    args = p.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.tnsr_json or not args.reference_json:
        p.error("tnsr_json and reference_json are required unless --self-test is given")

    return run_compare(args.tnsr_json, args.reference_json, dspark=args.dspark)


if __name__ == "__main__":
    sys.exit(main())
