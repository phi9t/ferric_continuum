from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping


@dataclass(frozen=True)
class RepoMount:
    name: str
    host: Path
    sandbox: str
    mode: Literal["ro", "rw"]
    required: bool


@dataclass(frozen=True)
class WorkspaceProfile:
    primary_repo: str
    repos: dict[str, RepoMount]


@dataclass(frozen=True)
class FerricRunConfig:
    repo_root: Path
    profile_path: Path
    rootfs: Path
    bwrap: Path | None
    workspace: WorkspaceProfile


def find_repo_root(start: Path) -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return Path(out).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return start.resolve()


def resolve_repo_host(value: str, *, repo_root: Path) -> Path:
    if value == "repo://self":
        return repo_root.resolve()
    return Path(value).expanduser().resolve()


def _require_workspace(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    workspace = data.get("workspace")
    if not isinstance(workspace, dict):
        raise ValueError("profile missing object field `workspace`")
    return workspace


def load_profile(path: Path, *, repo_root: Path) -> WorkspaceProfile:
    data = json.loads(path.read_text())
    workspace = _require_workspace(data)
    primary = workspace.get("primary_repo")
    repos_data = workspace.get("repos")
    if not isinstance(primary, str) or not primary:
        raise ValueError("workspace.primary_repo must be a non-empty string")
    if not isinstance(repos_data, dict) or not repos_data:
        raise ValueError("workspace.repos must be a non-empty object")

    repos: dict[str, RepoMount] = {}
    for name, raw in repos_data.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ValueError("workspace.repos entries must be objects keyed by name")
        host_raw = raw.get("host")
        sandbox = raw.get("sandbox")
        mode = raw.get("mode", "ro")
        required = raw.get("required", False)
        if not isinstance(host_raw, str) or not host_raw:
            raise ValueError(f"repo `{name}` missing string host")
        if not isinstance(sandbox, str) or not sandbox.startswith("/workspace/"):
            raise ValueError(f"repo `{name}` sandbox must start with /workspace/")
        if mode not in {"ro", "rw"}:
            raise ValueError(f"repo `{name}` mode must be ro or rw")
        if not isinstance(required, bool):
            raise ValueError(f"repo `{name}` required must be boolean")
        repos[name] = RepoMount(
            name=name,
            host=resolve_repo_host(host_raw, repo_root=repo_root),
            sandbox=sandbox,
            mode=mode,
            required=required,
        )
    if primary not in repos:
        raise ValueError(f"primary_repo `{primary}` is not listed in workspace.repos")
    return WorkspaceProfile(primary_repo=primary, repos=repos)


def resolve_config(
    *,
    repo_root: Path,
    profile_path: Path,
    rootfs_arg: str | None,
    bwrap_arg: str | None,
    environ: Mapping[str, str] = os.environ,
) -> FerricRunConfig:
    rootfs_value = rootfs_arg or environ.get("FERRIC_ROOTFS")
    if not rootfs_value:
        raise ValueError("rootfs is required; pass --rootfs or set FERRIC_ROOTFS")
    bwrap_value = bwrap_arg or environ.get("FERRIC_BWRAP")
    return FerricRunConfig(
        repo_root=repo_root.resolve(),
        profile_path=profile_path.resolve(),
        rootfs=Path(rootfs_value).expanduser().resolve(),
        bwrap=Path(bwrap_value).expanduser().resolve() if bwrap_value else None,
        workspace=load_profile(profile_path, repo_root=repo_root),
    )
