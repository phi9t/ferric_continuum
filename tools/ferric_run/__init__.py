"""Ferric local rootfs runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tools.ferric_run.bwrap import BwrapPlan, build_bwrap_argv, plan_to_json_dict
from tools.ferric_run import config as ferric_config
from tools.ferric_run.preflight import report_to_json_dict, resolve_bwrap, run_preflight
from tools.ferric_run.records import create_run_dir, new_run_id, write_json, write_text

__version__ = "0.1.0"

DEFAULT_PROFILE = Path("tools/ferric_run/profiles/gpu-kernel-study.json")
MISSING_BWRAP = Path("<missing-bwrap>")


def _normalize_command_tail(command_tail: list[str]) -> list[str]:
    if command_tail and command_tail[0] == "--":
        return command_tail[1:]
    return command_tail


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ferric commands in the local rootfs.")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--rootfs")
    parser.add_argument("--bwrap")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    command_argv = _normalize_command_tail(list(args.command))
    if not command_argv:
        parser.error("command is required after --")

    host_cwd = Path.cwd()
    repo_root = ferric_config.find_repo_root(Path.cwd())
    profile_path = args.profile or repo_root / DEFAULT_PROFILE
    config = ferric_config.resolve_config(
        repo_root=repo_root,
        profile_path=profile_path,
        rootfs_arg=args.rootfs,
        bwrap_arg=args.bwrap,
    )
    resolved_bwrap = resolve_bwrap(config.bwrap)
    if resolved_bwrap is not None and resolved_bwrap != config.bwrap:
        config = ferric_config.resolve_config(
            repo_root=repo_root,
            profile_path=profile_path,
            rootfs_arg=str(config.rootfs),
            bwrap_arg=str(resolved_bwrap),
        )

    run_id = new_run_id()
    run_dir = create_run_dir(config.repo_root, run_id)
    report = run_preflight(config, cuda_requested=args.cuda)
    primary_repo = config.workspace.repos[config.workspace.primary_repo]
    plan = BwrapPlan(
        run_id=run_id,
        rootfs=config.rootfs,
        bwrap=resolved_bwrap or MISSING_BWRAP,
        target_repo=config.workspace.primary_repo,
        cwd=primary_repo.sandbox,
        mounts=report.mounts,
        environment={"HOME": "/home/ferric", "USER": "ferric"},
        command_argv=command_argv,
        cuda_requested=args.cuda,
    )

    write_json(
        run_dir / "command.json",
        {
            "run_id": run_id,
            "raw_argv": raw_argv,
            "command_argv": command_argv,
            "profile_path": str(config.profile_path),
            "rootfs": str(config.rootfs),
            "bwrap": str(resolved_bwrap) if resolved_bwrap is not None else None,
            "cuda": {"requested": args.cuda},
            "dry_run": args.dry_run,
            "host_cwd": str(host_cwd),
        },
    )
    write_json(run_dir / "preflight.json", report_to_json_dict(report))
    write_json(run_dir / "bwrap-plan.json", plan_to_json_dict(plan))
    exit_code = 0 if report.ok else 1
    executed = False
    duration_ms = 0
    stdout = ""
    stderr = ""

    if report.ok and not args.dry_run:
        started = time.monotonic()
        completed = subprocess.run(
            build_bwrap_argv(plan),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
        executed = True

    write_text(run_dir / "stdout.log", stdout)
    write_text(run_dir / "stderr.log", stderr)
    write_json(
        run_dir / "result.json",
        {
            "run_id": run_id,
            "executed": executed,
            "ok": report.ok and exit_code == 0,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
        },
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
