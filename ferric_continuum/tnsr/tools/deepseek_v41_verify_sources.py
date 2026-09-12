#!/usr/bin/env python3
"""Verify DeepSeek V4.1 source claims against mirrored upstream text files."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    claim_id: str
    source_path: str
    all_phrases: tuple[str, ...]
    why: str


class ManifestError(ValueError):
    pass


def repo_path(repo_root: pathlib.Path, path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.is_absolute():
        raise ManifestError(f"manifest path must be relative: {path_text}")
    return repo_root / path


def load_manifest(path: pathlib.Path) -> list[Claim]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"missing manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid manifest JSON: {path}: {exc}") from exc

    claims = raw.get("claims")
    if not isinstance(claims, list):
        raise ManifestError("manifest must contain a claims list")

    parsed: list[Claim] = []
    seen: set[str] = set()
    for index, entry in enumerate(claims):
        if not isinstance(entry, dict):
            raise ManifestError(f"claim[{index}] must be an object")

        claim_id = entry.get("id")
        source_path = entry.get("source_path")
        expected = entry.get("expected")
        why = entry.get("why")
        if not isinstance(claim_id, str) or not claim_id:
            raise ManifestError(f"claim[{index}] has no string id")
        if claim_id in seen:
            raise ManifestError(f"duplicate claim id: {claim_id}")
        seen.add(claim_id)
        if not isinstance(source_path, str) or not source_path:
            raise ManifestError(f"{claim_id}: missing source_path")
        if not isinstance(why, str) or not why:
            raise ManifestError(f"{claim_id}: missing why")
        if not isinstance(expected, dict):
            raise ManifestError(f"{claim_id}: expected must be an object")

        phrases = expected.get("all_phrases")
        if not isinstance(phrases, list) or not phrases:
            raise ManifestError(f"{claim_id}: expected.all_phrases must be a non-empty list")
        bad_phrase = next((phrase for phrase in phrases if not isinstance(phrase, str) or not phrase), None)
        if bad_phrase is not None:
            raise ManifestError(f"{claim_id}: expected phrases must be non-empty strings")

        parsed.append(
            Claim(
                claim_id=claim_id,
                source_path=source_path,
                all_phrases=tuple(phrases),
                why=why,
            )
        )

    return parsed


def verify_claims(repo_root: pathlib.Path, manifest_path: pathlib.Path) -> tuple[list[str], list[str]]:
    repo_root = repo_root.resolve()
    claims = load_manifest(manifest_path)
    checked: list[str] = []
    errors: list[str] = []
    source_cache: dict[pathlib.Path, str] = {}

    for claim in claims:
        source = repo_path(repo_root, claim.source_path)
        if not source.exists():
            errors.append(f"{claim.claim_id}: missing source path: {source}")
            continue

        if source not in source_cache:
            source_cache[source] = source.read_text(encoding="utf-8")
        text = source_cache[source]

        missing = [phrase for phrase in claim.all_phrases if phrase not in text]
        if missing:
            missing_list = ", ".join(repr(phrase) for phrase in missing)
            errors.append(f"{claim.claim_id}: missing expected phrase(s) in {claim.source_path}: {missing_list}")
            continue

        checked.append(claim.claim_id)

    return checked, errors


def report_verification(checked: list[str], errors: list[str]) -> int:
    if errors:
        print("FAIL: DeepSeek V4.1 source manifest verification failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"PASS: checked {len(checked)} DeepSeek V4.1 source claim(s)")
    for claim_id in checked:
        print(f"- {claim_id}")
    return 0


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="deepseek_v41_source_selftest_") as tmp:
        root = pathlib.Path(tmp)
        source = root / "source.txt"
        manifest = root / "manifest.json"
        source.write_text("present upstream claim\n", encoding="utf-8")
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "claims": [
                        {
                            "id": "selftest-missing-claim",
                            "source_path": "source.txt",
                            "expected": {"all_phrases": ["claim that is not present"]},
                            "why": "Proves a missing claim returns nonzero and names the claim id.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        checked, errors = verify_claims(root, manifest)
        status = report_verification(checked, errors)
        if status == 0:
            print("SELF-TEST FAIL: missing claim returned success", file=sys.stderr)
            return 1
        if not errors or "selftest-missing-claim" not in errors[0]:
            print("SELF-TEST FAIL: missing-claim error did not name claim id", file=sys.stderr)
            for error in errors:
                print(error, file=sys.stderr)
            return 1

    print("SELF-TEST PASS: missing claim fails and names selftest-missing-claim")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("ferric_continuum/tnsr/testdata/deepseek_v41/source_manifest.json"),
    )
    parser.add_argument(
        "--self-test-negative",
        action="store_true",
        help="run a negative self-test that proves missing claims fail clearly",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test_negative:
        return run_self_test()

    repo_root = args.repo_root.resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = repo_root / manifest

    try:
        checked, errors = verify_claims(repo_root, manifest)
    except ManifestError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    return report_verification(checked, errors)


if __name__ == "__main__":
    raise SystemExit(main())
