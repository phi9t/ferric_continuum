from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from tools.ferric_run.bwrap import MountSpec
from tools.ferric_run.config import FerricRunConfig


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    message: str


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    checks: list[PreflightCheck]
    warnings: list[str]
    mounts: list[MountSpec]


def resolve_bwrap(configured: Path | None, environ: Mapping[str, str] = os.environ) -> Path | None:
    if configured is not None:
        return configured
    env_value = environ.get("FERRIC_BWRAP")
    if env_value:
        return Path(env_value).expanduser().resolve()
    found = shutil.which("bwrap")
    return Path(found).resolve() if found else None


def evaluate_cuda_signals(*, device_paths: Sequence[Path], library_roots: Sequence[Path]) -> tuple[bool, list[str]]:
    messages: list[str] = []
    if not any(path.exists() for path in device_paths):
        messages.append("CUDA requested but no /dev/nvidia* device path was found")
    if not any(path.exists() for path in library_roots):
        messages.append("CUDA requested but no likely NVIDIA/CUDA driver library root was found")
    return (not messages, messages)


def _discover_cuda_devices() -> list[Path]:
    return list(Path("/dev").glob("nvidia*"))


def _candidate_cuda_library_roots() -> list[Path]:
    return [
        Path("/run/nvidia-driver"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/cuda/lib64"),
    ]


def rootfs_path_for_sandbox_target(rootfs: Path, sandbox: str) -> Path:
    sandbox_path = Path(sandbox)
    if not sandbox_path.is_absolute():
        raise ValueError(f"sandbox target must be absolute: {sandbox}")
    relative_parts = sandbox_path.parts[1:]
    if any(part in {"", ".", ".."} for part in relative_parts):
        raise ValueError(f"sandbox target must not contain relative components: {sandbox}")
    return rootfs.joinpath(*relative_parts)


def run_preflight(config: FerricRunConfig, *, cuda_requested: bool) -> PreflightReport:
    checks: list[PreflightCheck] = []
    warnings: list[str] = []
    mounts: list[MountSpec] = []

    rootfs_ok = config.rootfs.is_dir()
    checks.append(PreflightCheck("rootfs", rootfs_ok, f"rootfs: {config.rootfs}"))
    bwrap_ok = config.bwrap is not None and config.bwrap.is_file() and os.access(config.bwrap, os.X_OK)
    checks.append(PreflightCheck("bwrap", bwrap_ok, f"bwrap: {config.bwrap}"))

    for repo in config.workspace.repos.values():
        host_present = repo.host.exists()
        sandbox_target = rootfs_path_for_sandbox_target(config.rootfs, repo.sandbox)
        sandbox_target_present = rootfs_ok and sandbox_target.exists()
        present = host_present and sandbox_target_present
        mounts.append(MountSpec(f"workspace:{repo.name}", repo.host, repo.sandbox, repo.mode, repo.required, present))
        if repo.required:
            checks.append(PreflightCheck(f"repo:{repo.name}", host_present, f"required repo {repo.name}: {repo.host}"))
            checks.append(
                PreflightCheck(
                    f"mountpoint:{repo.name}",
                    sandbox_target_present,
                    f"required repo {repo.name} sandbox target {repo.sandbox}: {sandbox_target}",
                )
            )
        elif not host_present:
            warnings.append(f"optional repo {repo.name} is missing at {repo.host}; mount omitted")
        elif not sandbox_target_present:
            warnings.append(
                f"optional repo {repo.name} sandbox target {repo.sandbox} is missing in rootfs at "
                f"{sandbox_target}; mount omitted"
            )

    if cuda_requested:
        cuda_ok, messages = evaluate_cuda_signals(
            device_paths=_discover_cuda_devices(),
            library_roots=_candidate_cuda_library_roots(),
        )
        checks.append(PreflightCheck("cuda", cuda_ok, "; ".join(messages) if messages else "CUDA host signals present"))

    return PreflightReport(ok=all(check.ok for check in checks), checks=checks, warnings=warnings, mounts=mounts)


def report_to_json_dict(report: PreflightReport) -> dict[str, object]:
    return {
        "ok": report.ok,
        "checks": [{"name": check.name, "ok": check.ok, "message": check.message} for check in report.checks],
        "warnings": list(report.warnings),
        "mounts": [
            {
                "name": mount.name,
                "host": str(mount.host),
                "sandbox": mount.sandbox,
                "mode": mount.mode,
                "required": mount.required,
                "present": mount.present,
            }
            for mount in report.mounts
        ],
    }
