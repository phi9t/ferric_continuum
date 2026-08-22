#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Checker:
    kind: str
    command_prefix: list[str]
    jar: Path | None = None
    message: str = ""


def discover_checker(
    *,
    jar_arg: str | None,
    environ: Mapping[str, str] | None = None,
    path: Iterable[str] | None = None,
) -> Checker:
    env = dict(os.environ if environ is None else environ)
    search_path = list(path) if path is not None else None
    jar_value = jar_arg or env.get("TLA_TOOLS_JAR")
    if jar_value:
        jar = Path(jar_value).expanduser().resolve()
        if jar.is_file():
            return Checker(
                kind="tlc",
                jar=jar,
                command_prefix=["java", "-cp", str(jar), "tlc2.TLC"],
            )
        return Checker(
            kind="missing",
            command_prefix=[],
            message=f"TLA tools jar not found: {jar}",
        )

    tlc = shutil.which("tlc", path=os.pathsep.join(search_path) if search_path is not None else None)
    if tlc:
        return Checker(kind="tlc", command_prefix=[tlc])

    return Checker(
        kind="missing",
        command_prefix=[],
        message="No TLA+ checker found; pass --tla-tools-jar or set TLA_TOOLS_JAR.",
    )


def _read_version(command_prefix: Sequence[str]) -> str | None:
    if not command_prefix:
        return None
    try:
        completed = subprocess.run(
            [*command_prefix, "-version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout + completed.stderr).strip()
    return text.splitlines()[0] if text else None


def _write_evidence(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def summarize_tlc_output(text: str) -> str:
    if "Model checking completed. No error has been found." in text:
        return "success"
    if "Error: Invariant" in text and "is violated" in text:
        return "invariant_violation"
    if "Error: Temporal properties were violated." in text:
        return "temporal_violation"
    return "checker_error"


def build_argv(checker: Checker, model: Path, config: Path) -> list[str]:
    return [*checker.command_prefix, str(model), "-config", str(config)]


def build_execution_argv(checker: Checker, model: Path, config: Path, out_dir: Path) -> list[str]:
    return [
        *checker.command_prefix,
        "-metadir",
        str(out_dir / "states"),
        str(model),
        "-config",
        str(config),
    ]


def run_checker(checker: Checker, model: Path, config: Path, out_dir: Path) -> int:
    command = build_execution_argv(checker, model, config, out_dir)
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(model.parent),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    result = summarize_tlc_output(completed.stdout + completed.stderr)
    ok = completed.returncode == 0 and result == "success"
    evidence = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": str(model),
        "config": str(config),
        "checker": checker.kind,
        "checker_version": _read_version(checker.command_prefix),
        "argv": command,
        "executed": True,
        "ok": ok,
        "exit_code": completed.returncode,
        "result": result,
        "message": checker.message,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    if result != "success":
        evidence["counterexample"] = str(stdout_path)
    _write_evidence(out_dir / "tla-check.json", evidence)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a TLA+ model check and record evidence.")
    parser.add_argument("--model", required=True, help="Path to the .tla module.")
    parser.add_argument("--config", required=True, help="Path to the TLC .cfg file.")
    parser.add_argument("--tla-tools-jar", help="Path to tla2tools.jar.")
    parser.add_argument("--out-dir", default=".ferric/tla", help="Directory for tla-check.json/stdout/stderr.")
    parser.add_argument("--dry-run", action="store_true", help="Record the checker command without executing it.")
    args = parser.parse_args(argv)

    model = Path(args.model).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    evidence_path = out_dir / "tla-check.json"
    checker = discover_checker(jar_arg=args.tla_tools_jar)

    base = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": str(model),
        "config": str(config),
        "checker": checker.kind,
        "checker_version": _read_version(checker.command_prefix),
        "executed": False,
        "ok": False,
        "message": checker.message,
    }

    for path, label in [(model, "model"), (config, "config")]:
        if not path.is_file():
            _write_evidence(evidence_path, {**base, "message": f"{label} not found: {path}"})
            return 2

    command = build_execution_argv(checker, model, config, out_dir)
    if args.dry_run:
        _write_evidence(evidence_path, {**base, "argv": command, "executed": False, "ok": True})
        return 0

    if checker.kind == "missing":
        _write_evidence(evidence_path, base)
        print(checker.message, file=sys.stderr)
        return 2

    return run_checker(checker, model, config, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
