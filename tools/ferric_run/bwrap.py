from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class MountSpec:
    name: str
    host: Path
    sandbox: str
    mode: Literal["ro", "rw"]
    required: bool
    present: bool


@dataclass(frozen=True)
class BwrapPlan:
    run_id: str
    rootfs: Path
    bwrap: Path
    target_repo: str
    cwd: str
    mounts: list[MountSpec]
    environment: dict[str, str]
    command_argv: list[str]
    cuda_requested: bool


def build_bwrap_argv(plan: BwrapPlan) -> list[str]:
    argv = [
        str(plan.bwrap),
        "--die-with-parent",
        "--unshare-all",
        "--share-net",
        "--clearenv",
        "--ro-bind",
        str(plan.rootfs),
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind-try",
        "/sys",
        "/sys",
    ]
    for key in sorted(plan.environment):
        argv.extend(["--setenv", key, plan.environment[key]])
    for mount in sorted((m for m in plan.mounts if m.present), key=lambda m: m.sandbox):
        argv.extend(["--bind" if mount.mode == "rw" else "--ro-bind", str(mount.host), mount.sandbox])
    argv.extend(["--chdir", plan.cwd, "--"])
    argv.extend(plan.command_argv)
    return argv


def argv_sha256(argv: list[str]) -> str:
    payload = json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_to_json_dict(plan: BwrapPlan) -> dict[str, object]:
    argv = build_bwrap_argv(plan)
    return {
        "run_id": plan.run_id,
        "rootfs": str(plan.rootfs),
        "bwrap": str(plan.bwrap),
        "target_repo": plan.target_repo,
        "cwd": plan.cwd,
        "environment": dict(sorted(plan.environment.items())),
        "command_argv": list(plan.command_argv),
        "cuda": {"requested": plan.cuda_requested},
        "mounts": [
            {
                "name": mount.name,
                "host": str(mount.host),
                "sandbox": mount.sandbox,
                "mode": mount.mode,
                "required": mount.required,
                "present": mount.present,
            }
            for mount in sorted(plan.mounts, key=lambda m: m.sandbox)
        ],
        "bwrap_argv": argv,
        "bwrap_argv_sha256": argv_sha256(argv),
    }
