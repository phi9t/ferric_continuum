#!/usr/bin/env python3
"""Validate or bootstrap local Ultron integration branches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO


class MainlineError(Exception):
    """An unsafe or invalid mainline request."""


class GitInvocationError(MainlineError):
    """Git could not complete an inspection or execution call."""


class BootstrapExecutionError(MainlineError):
    """Bootstrap execution stopped after completing a repository prefix."""

    def __init__(self, message: str, completed: tuple[str, ...]):
        super().__init__(message)
        self.completed = completed


@dataclass(frozen=True)
class RepositoryContract:
    name: str
    root: Path
    upstream_mainline: str
    ultron_mainline: str
    bootstrap_commit: str

    @property
    def integration_worktree(self) -> Path:
        return self.root / ".worktrees" / "ultron-mainline"


MANIFEST_VERSION = "2026.08.15.2"
CANONICAL_REPOSITORIES = (
    RepositoryContract(
        "ultron", Path("ultron"), "main", "ultron/mainline",
        "3c7e68147ec06869085777fa3d37798e48177219"
    ),
    RepositoryContract(
        "torchtitan", Path("torchtitan"), "main", "ultron/mainline",
        "d9d58bf2d3fe5f0f4434cf9aa943a867d8c551f3"
    ),
    RepositoryContract(
        "monarch", Path("monarch"), "main", "ultron/mainline",
        "2d9a3e102853f3bb2e465e46bd6b0aff1f9d33cf"
    ),
    RepositoryContract(
        "megatronlm", Path("megatronlm"), "main", "ultron/mainline",
        "122311a7e7a5c8f1861aad67961de6495ab82b39"
    ),
    RepositoryContract(
        "nccl", Path("nccl"), "master", "ultron/mainline",
        "1c1cb435012474385edfd32476499d2cfe97cdb7"
    ),
    RepositoryContract(
        "ferric_continuum",
        Path("ferric_continuum"),
        "main",
        "ultron/mainline",
        "d2a2c33fa8eb4ef84d93ff8f8e2a41d0e6afa323",
    ),
    RepositoryContract(
        "tnsr", Path("tnsr"), "master", "ultron/mainline",
        "7c49c1912b6abced917854212371c441f8cde02e"
    ),
)


@dataclass(frozen=True)
class BootstrapAction:
    repository: str
    root: Path
    source_ref: str
    source_commit: str
    destination_ref: str
    destination_path: Path
    create_ref: bool
    create_worktree: bool

    @property
    def name(self) -> str:
        return self.repository

    @property
    def expected_common_directory(self) -> Path:
        return (self.root / ".git").resolve()


@dataclass(frozen=True)
class PreflightResult:
    action: BootstrapAction
    source_tip: str
    reason: str | None = None
    operation: str = "create"


@dataclass(frozen=True)
class IntegrationInspection:
    branch_presence: subprocess.CompletedProcess[str]
    registration_paths: tuple[Path, ...]
    target_has_symlink: bool
    target_is_directory: bool


def _inspect_integration_state(
    action: BootstrapAction, git_runner: Callable
) -> IntegrationInspection:
    registrations = _call_git(
        git_runner,
        ("worktree", "list", "--porcelain"),
        cwd=action.root,
    )
    _require_success(registrations)
    branch_presence = _call_git(
        git_runner,
        (
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{action.destination_ref}",
        ),
        cwd=action.root,
    )
    if branch_presence.returncode not in {0, 1}:
        _require_success(branch_presence)
    return IntegrationInspection(
        branch_presence=branch_presence,
        registration_paths=tuple(
            _integration_registration_paths(
                registrations.stdout, action.destination_ref
            )
        ),
        target_has_symlink=_path_contains_symlink(
            action.root, action.destination_path
        ),
        target_is_directory=action.destination_path.is_dir(),
    )


def run_git(arguments: tuple[str, ...], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run Git without a shell through the tool's declared test seam."""

    return subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )


def _call_git(
    git_runner: Callable,
    arguments: tuple[str, ...],
    *,
    cwd: Path,
    phase: str = "inspection",
) -> subprocess.CompletedProcess[str]:
    try:
        return git_runner(arguments, cwd=cwd)
    except subprocess.TimeoutExpired as error:
        raise GitInvocationError(
            f"Git {phase} failed (timeout after 15 seconds)"
        ) from error
    except OSError as error:
        raise GitInvocationError(
            f"Git {phase} failed (OSError: {error})"
        ) from error


def _path_contains_symlink(root: Path, target: Path) -> bool:
    if root.is_symlink():
        return True
    current = root
    for part in target.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _resolved_common_directory(
    result: subprocess.CompletedProcess[str], cwd: Path
) -> Path | None:
    if result.returncode or not result.stdout.strip():
        return None
    value = Path(result.stdout.strip())
    return (value if value.is_absolute() else cwd / value).resolve()


def _integration_registration_paths(output: str, destination_ref: str) -> list[Path]:
    paths = []
    current_path = None
    branch_line = f"branch refs/heads/{destination_ref}"
    for line in output.splitlines():
        if line.startswith("worktree "):
            current_path = Path(line.removeprefix("worktree ")).resolve()
        elif line == branch_line and current_path is not None:
            paths.append(current_path)
    return paths


def default_workspace_root(script: Path, *, git_runner: Callable) -> Path:
    """Resolve the canonical workspace through the script checkout's Git identity."""

    source_worktree = script.absolute().parents[1]
    try:
        common = git_runner(
            ("rev-parse", "--git-common-dir"), cwd=source_worktree
        )
    except subprocess.TimeoutExpired as error:
        raise MainlineError("cannot discover workspace: Git inspection timed out") from error
    except OSError as error:
        raise MainlineError(f"cannot discover workspace: {error}") from error
    if common.returncode or not common.stdout.strip():
        raise MainlineError(
            "cannot discover workspace: script is not in a canonical Git worktree"
        )
    common_value = Path(common.stdout.strip())
    declared_common = (
        common_value if common_value.is_absolute() else source_worktree / common_value
    )
    if _path_contains_symlink(Path(declared_common.anchor), declared_common):
        raise MainlineError(
            "cannot discover workspace: Git common directory is a symlink"
        )
    common_directory = declared_common.resolve()
    if common_directory.name != ".git":
        raise MainlineError(
            "cannot discover workspace: script is not in a canonical Git worktree"
        )
    repository = common_directory.parent
    canonical_names = {contract.name for contract in CANONICAL_REPOSITORIES}
    if repository.name not in canonical_names or repository.is_symlink():
        raise MainlineError(
            "cannot discover workspace: Git common directory is not canonical"
        )
    return repository.parent


def _actions_from_contracts(
    workspace_root: Path,
    contracts: tuple[RepositoryContract, ...],
) -> list[BootstrapAction]:
    return [
        BootstrapAction(
            repository=contract.name,
            root=(
                contract.root
                if contract.root.is_absolute()
                else workspace_root / contract.name
            ),
            source_ref=contract.upstream_mainline,
            source_commit=contract.bootstrap_commit,
            destination_ref=contract.ultron_mainline,
            destination_path=(
                contract.integration_worktree
                if contract.root.is_absolute()
                else workspace_root / contract.name / ".worktrees" / "ultron-mainline"
            ),
            create_ref=True,
            create_worktree=True,
        )
        for contract in contracts
    ]


def load_contracts(
    manifest_path: Path, workspace_root: Path
) -> tuple[RepositoryContract, ...]:
    """Load and fully validate the repository contract."""

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MainlineError(f"cannot read manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise MainlineError("manifest root must be an object")
    if manifest.get("version") != MANIFEST_VERSION:
        raise MainlineError(f"manifest version must be {MANIFEST_VERSION}")
    if manifest.get("source_repository") != "ultron":
        raise MainlineError("manifest source_repository must be ultron")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        raise MainlineError("manifest repositories must be a non-empty list")
    allowed_fields = {
        "name",
        "upstream_mainline",
        "ultron_mainline",
        "ultron_mainline_bootstrap",
        "instruction_files",
        "install_local_tracker",
    }
    contracts = []
    names = set()
    for index, entry in enumerate(repositories):
        if not isinstance(entry, dict) or set(entry) != allowed_fields:
            raise MainlineError(
                f"manifest repository {index} has unsupported or missing fields"
            )
        name = entry["name"]
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or name in {".", ".."}
        ):
            raise MainlineError(f"manifest repository {index} has unsafe name")
        if name in names:
            raise MainlineError(f"manifest has duplicate repository {name}")
        names.add(name)
        upstream = entry["upstream_mainline"]
        if upstream not in {"main", "master"}:
            raise MainlineError(f"{name}: upstream_mainline must be main or master")
        ultron = entry["ultron_mainline"]
        if ultron != "ultron/mainline":
            raise MainlineError(f"{name}: ultron_mainline must be ultron/mainline")
        bootstrap = entry["ultron_mainline_bootstrap"]
        if not isinstance(bootstrap, str) or len(bootstrap) != 40:
            raise MainlineError(f"{name}: bootstrap commit must be a 40-hex SHA")
        try:
            int(bootstrap, 16)
        except ValueError as error:
            raise MainlineError(
                f"{name}: bootstrap commit must be a 40-hex SHA"
            ) from error
        instruction_files = entry["instruction_files"]
        if not isinstance(instruction_files, list) or not all(
            isinstance(value, str) and value for value in instruction_files
        ):
            raise MainlineError(f"{name}: instruction_files must be strings")
        if not isinstance(entry["install_local_tracker"], bool):
            raise MainlineError(f"{name}: install_local_tracker must be boolean")
        contracts.append(
            RepositoryContract(
                name=name,
                root=workspace_root / name,
                upstream_mainline=upstream,
                ultron_mainline=ultron,
                bootstrap_commit=bootstrap,
            )
        )
    return tuple(contracts)


def _manifest_actions(
    workspace_root: Path,
    manifest_path: Path,
    *,
    expected_contracts: tuple[RepositoryContract, ...] = CANONICAL_REPOSITORIES,
) -> list[BootstrapAction]:
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise MainlineError("manifest root must be an object")
    if manifest.get("version") != MANIFEST_VERSION:
        raise MainlineError(f"manifest version must be {MANIFEST_VERSION}")
    if manifest.get("source_repository") != "ultron":
        raise MainlineError("manifest source_repository must be ultron")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, list):
        raise MainlineError("manifest repositories must be a list")
    if len(repositories) != len(expected_contracts):
        raise MainlineError("manifest repository list does not match canonical snapshot")

    allowed_fields = {
        "name",
        "upstream_mainline",
        "ultron_mainline",
        "ultron_mainline_bootstrap",
        "instruction_files",
        "install_local_tracker",
    }
    required_fields = allowed_fields
    for index, (entry, expected) in enumerate(
        zip(repositories, expected_contracts, strict=True)
    ):
        if not isinstance(entry, dict):
            raise MainlineError(f"manifest repository {index} must be an object")
        if set(entry) != required_fields:
            raise MainlineError(
                f"manifest repository {index} has unsupported or missing fields"
            )
        instruction_files = entry["instruction_files"]
        if not isinstance(instruction_files, list) or not all(
            isinstance(value, str) and value for value in instruction_files
        ):
            raise MainlineError(
                f"manifest repository {index} instruction_files must be strings"
            )
        if not isinstance(entry["install_local_tracker"], bool):
            raise MainlineError(
                f"manifest repository {index} install_local_tracker must be boolean"
            )
        actual_identity = (
            entry["name"],
            entry["upstream_mainline"],
            entry["ultron_mainline"],
            entry["ultron_mainline_bootstrap"],
        )
        expected_identity = (
            expected.name,
            expected.upstream_mainline,
            expected.ultron_mainline,
            expected.bootstrap_commit,
        )
        if actual_identity != expected_identity:
            raise MainlineError(
                f"manifest repository {index} does not match canonical snapshot"
            )
    return _actions_from_contracts(workspace_root, expected_contracts)


def _inspect_bootstrap_actions(
    actions: list[BootstrapAction], *, git_runner: Callable
) -> list[PreflightResult]:
    results = []
    for action in actions:
        common = _call_git(
            git_runner,
            ("rev-parse", "--git-common-dir"), cwd=action.root
        )
        reason = None
        if common.returncode:
            if "not a git repository" in common.stderr.lower():
                reason = "repository path is not a Git repository"
            else:
                reason = (
                    f"Git inspection failed (exit {common.returncode}): "
                    f"{common.stderr.strip()}"
                )
        else:
            common_value = Path(common.stdout.strip())
            resolved_common = (
                common_value
                if common_value.is_absolute()
                else action.root / common_value
            ).resolve()
            expected_common = action.expected_common_directory
            if resolved_common != expected_common:
                reason = (
                    "wrong Git identity "
                    f"(expected {expected_common}, got {resolved_common})"
                )
            toplevel = _call_git(
                git_runner,
                ("rev-parse", "--show-toplevel"), cwd=action.root
            )
            _require_success(toplevel)
            if (
                reason is None
                and (
                    Path(toplevel.stdout.strip()).resolve()
                    != action.root.resolve()
                )
            ):
                reason = "repository path is not a Git repository"
        source = _call_git(
            git_runner,
            ("rev-parse", "--verify", f"{action.source_ref}^{{commit}}"),
            cwd=action.root,
        )
        _require_success(source)
        source_tip = source.stdout.strip()
        if reason is None and source_tip != action.source_commit:
            reason = (
                "source tip changed "
                f"(expected {action.source_commit}, got {source_tip or 'unresolved'})"
            )
        operation = "create"
        destination_ref = f"refs/heads/{action.destination_ref}"
        integration = _inspect_integration_state(action, git_runner)
        destination_presence = integration.branch_presence
        if destination_presence.returncode == 0:
            destination = _call_git(
                git_runner,
                ("rev-parse", "--verify", f"{destination_ref}^{{commit}}"),
                cwd=action.root,
            )
            _require_success(destination)
        else:
            destination = destination_presence
            if reason is None and destination_presence.returncode != 1:
                reason = (
                    f"Git inspection failed (exit {destination_presence.returncode}): "
                    f"{destination_presence.stderr.strip()}"
                )
        target_has_symlink = integration.target_has_symlink
        if reason is None and target_has_symlink:
            reason = "integration target contains a symlink"
        if (
            reason is None
            and action.destination_path.exists()
            and (destination.returncode or not action.destination_path.is_dir())
        ):
            reason = "integration target is occupied"
        destination_parts = action.destination_ref.split("/")
        namespace_conflict = False
        for length in range(1, len(destination_parts)):
            parent_ref = "/".join(destination_parts[:length])
            parent = _call_git(
                git_runner,
                ("show-ref", "--verify", "--quiet", f"refs/heads/{parent_ref}"),
                cwd=action.root,
            )
            if parent.returncode == 0:
                namespace_conflict = True
                break
            if parent.returncode != 1:
                if reason is None:
                    reason = (
                        f"Git inspection failed (exit {parent.returncode}): "
                        f"{parent.stderr.strip()}"
                    )
                break
        if reason is None and destination.returncode and namespace_conflict:
            reason = "destination namespace conflict"
        if not destination.returncode:
            registration_paths = list(integration.registration_paths)
            if (
                reason is None
                and not action.destination_path.is_dir()
                and registration_paths
                and (
                registration_paths != [action.destination_path.resolve()]
                )
            ):
                reason = (
                    "integration branch is registered at unexpected worktree"
                    if len(registration_paths) == 1
                    else "integration branch has "
                    f"{len(registration_paths)} worktree registrations"
                )
            upstream = _call_git(
                git_runner,
                (
                    "for-each-ref",
                    "--format=%(upstream)",
                    f"refs/heads/{action.destination_ref}",
                ),
                cwd=action.root,
            )
            _require_success(upstream)
            if reason is None and upstream.stdout.strip():
                reason = "integration branch has configured upstream"
            if destination.stdout.strip() != action.source_commit:
                ancestry = _call_git(
                    git_runner,
                    (
                        "merge-base",
                        "--is-ancestor",
                        action.source_commit,
                        action.destination_ref,
                    ),
                    cwd=action.root,
                )
                if ancestry.returncode not in {0, 1}:
                    _require_success(ancestry)
                if reason is None:
                    reason = (
                        "integration branch advanced; use --check"
                        if ancestry.returncode == 0
                        else "integration history is rewritten or unrelated"
                    )
            if not target_has_symlink and action.destination_path.is_dir():
                target_common = _call_git(
                    git_runner,
                    ("rev-parse", "--git-common-dir"),
                    cwd=action.destination_path,
                )
                _require_success(target_common)
                if reason is None and (
                    _resolved_common_directory(
                        target_common, action.destination_path
                    )
                    != action.expected_common_directory
                ):
                    reason = "integration target has wrong Git identity"
                current = _call_git(
                    git_runner,
                    ("branch", "--show-current"), cwd=action.destination_path
                )
                _require_success(current)
                if reason is None and current.stdout.strip() != action.destination_ref:
                    reason = "integration target is registered to the wrong branch"
                status = _call_git(
                    git_runner,
                    ("status", "--porcelain"), cwd=action.destination_path
                )
                _require_success(status)
                if reason is None and status.stdout:
                    reason = "integration worktree is dirty"
                if reason is None and (
                    registration_paths != [action.destination_path.resolve()]
                ):
                    reason = (
                        "integration branch has "
                        f"{len(registration_paths)} worktree registrations"
                    )
        if (
            not destination.returncode
            and destination.stdout.strip() == action.source_commit
            and action.destination_path.is_dir()
        ):
            operation = "unchanged"
        elif (
            not destination.returncode
            and destination.stdout.strip() == action.source_commit
        ):
            operation = "attach"
        results.append(
            PreflightResult(
                action=action,
                source_tip=source_tip,
                reason=reason,
                operation=operation,
            )
        )
    return results


def _plan_bootstrap_results(
    *,
    workspace_root: Path,
    manifest_path: Path,
    git_runner: Callable,
    expected_contracts: tuple[RepositoryContract, ...] = CANONICAL_REPOSITORIES,
) -> list[PreflightResult]:
    return _inspect_bootstrap_actions(
        _manifest_actions(
            workspace_root,
            manifest_path,
            expected_contracts=expected_contracts,
        ),
        git_runner=git_runner,
    )


def _require_success(
    result: subprocess.CompletedProcess[str], *, phase: str = "inspection"
) -> None:
    if result.returncode:
        detail = result.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise GitInvocationError(
            f"Git {phase} failed (exit {result.returncode}){suffix}"
        )


def _check_reason(action: BootstrapAction, git_runner: Callable) -> str | None:
    integration = _inspect_integration_state(action, git_runner)
    destination = integration.branch_presence
    if integration.target_has_symlink:
        return "integration target contains a symlink"
    if not integration.target_is_directory:
        if integration.registration_paths:
            return "integration branch is registered at unexpected worktree"
        return "integration worktree is missing"
    if destination.returncode == 1:
        return "integration branch is missing"

    reason = None
    target_common = _call_git(
        git_runner,
        ("rev-parse", "--git-common-dir"),
        cwd=action.destination_path,
    )
    _require_success(target_common)
    if (
        _resolved_common_directory(target_common, action.destination_path)
        != action.expected_common_directory
    ):
        reason = "integration target has wrong Git identity"

    descendant = _call_git(
        git_runner,
        (
            "merge-base",
            "--is-ancestor",
            action.source_commit,
            action.destination_ref,
        ),
        cwd=action.root,
    )
    if descendant.returncode == 1 and reason is None:
        reason = "integration history is rewritten or unrelated"
    elif descendant.returncode not in {0, 1}:
        _require_success(descendant)

    current = _call_git(
        git_runner,
        ("branch", "--show-current"),
        cwd=action.destination_path,
    )
    _require_success(current)
    if reason is None and current.stdout.strip() != action.destination_ref:
        reason = "integration worktree is invalid"

    status = _call_git(
        git_runner,
        ("status", "--porcelain"),
        cwd=action.destination_path,
    )
    _require_success(status)
    if reason is None and status.stdout:
        reason = "integration worktree is dirty"

    upstream = _call_git(
        git_runner,
        (
            "for-each-ref",
            "--format=%(upstream)",
            f"refs/heads/{action.destination_ref}",
        ),
        cwd=action.root,
    )
    _require_success(upstream)
    if reason is None and upstream.stdout.strip():
        reason = "integration branch has configured upstream"

    registration_paths = list(integration.registration_paths)
    if reason is None and registration_paths != [action.destination_path.resolve()]:
        reason = (
            "integration branch has "
            f"{len(registration_paths)} worktree registrations"
        )
    return reason


def check_contracts(
    contracts: tuple[RepositoryContract, ...]
) -> tuple[str, ...]:
    """Return diagnostics; an empty tuple means all established contracts pass."""

    diagnostics = []
    for action in _actions_from_contracts(Path("."), contracts):
        try:
            reason = _check_reason(action, run_git)
        except GitInvocationError as error:
            reason = str(error)
        if reason:
            diagnostics.append(f"{action.name}: {reason}")
    return tuple(diagnostics)


def check_contract(
    actions: list[BootstrapAction], *, output: TextIO, git_runner: Callable
) -> int:
    drift_failures = 0
    invocation_failures = 0
    for action in actions:
        try:
            reason = _check_reason(action, git_runner)
        except GitInvocationError as error:
            invocation_failures += 1
            print(
                f"ERROR repository={action.name} "
                f"source={action.source_ref}@{action.source_commit} "
                f"destination={action.destination_ref} "
                f"path={action.destination_path} status=error reason={error}",
                file=output,
            )
            continue
        if reason:
            drift_failures += 1
            print(
                f"DRIFT repository={action.name} "
                f"source={action.source_ref}@{action.source_commit} "
                f"destination={action.destination_ref} "
                f"path={action.destination_path} status=invalid reason={reason}",
                file=output,
            )
            continue
        print(
            f"OK repository={action.name} "
            f"source={action.source_ref}@{action.source_commit} "
            f"destination={action.destination_ref} "
            f"path={action.destination_path} status=valid",
            file=output,
        )
    if invocation_failures:
        print(f"CHECK ERROR: {invocation_failures} failure(s)", file=output)
        return 2
    if drift_failures:
        print(f"CHECK FAILED: {drift_failures} repository(s)", file=output)
        return 1
    print(f"CHECK OK: {len(actions)} repositories", file=output)
    return 0


def _planned_action(result: PreflightResult) -> BootstrapAction:
    action = result.action
    return BootstrapAction(
        repository=action.repository,
        root=action.root,
        source_ref=action.source_ref,
        source_commit=action.source_commit,
        destination_ref=action.destination_ref,
        destination_path=action.destination_path,
        create_ref=result.operation == "create",
        create_worktree=result.operation != "unchanged",
    )


def _bootstrap_arguments(action: BootstrapAction) -> tuple[str, ...] | None:
    if not action.create_worktree:
        return None
    if action.create_ref:
        return (
            "worktree",
            "add",
            "-b",
            action.destination_ref,
            str(action.destination_path),
            action.source_commit,
        )
    return (
        "worktree",
        "add",
        str(action.destination_path),
        action.destination_ref,
    )


def _execute_bootstrap_results(
    results: list[PreflightResult],
    *,
    output: TextIO,
    git_runner: Callable,
) -> int:
    completed: list[str] = []
    for preflight in results:
        action = _planned_action(preflight)
        arguments = _bootstrap_arguments(action)
        if arguments is None:
            completed.append(action.name)
            print(
                f"OK repository={action.name} "
                f"source={action.source_ref}@{preflight.source_tip} "
                f"destination={action.destination_ref} "
                f"path={action.destination_path} status=unchanged",
                file=output,
            )
            continue
        try:
            result = _call_git(
                git_runner,
                arguments,
                cwd=action.root,
                phase="execution",
            )
            failure_reason = None
            if result.returncode:
                detail = result.stderr.strip()
                suffix = f": {detail}" if detail else ""
                failure_reason = (
                    f"Git execution failed (exit {result.returncode}){suffix}"
                )
        except GitInvocationError as error:
            failure_reason = str(error)
        if failure_reason:
            raise BootstrapExecutionError(
                failure_reason,
                tuple(completed),
            )
        completed.append(action.name)
        print(
            f"CREATED repository={action.name} "
            f"source={action.source_ref}@{action.source_commit} "
            f"destination={action.destination_ref} "
            f"path={action.destination_path} status=created",
            file=output,
        )
    print(f"BOOTSTRAP OK: {len(completed)} repositories", file=output)
    return 0


def plan_bootstrap(
    contracts: tuple[RepositoryContract, ...]
) -> tuple[BootstrapAction, ...]:
    """Preflight every repository without writing and return the exact plan."""

    inspected = _inspect_bootstrap_actions(
        _actions_from_contracts(Path("."), contracts),
        git_runner=run_git,
    )
    failures = [
        f"{result.action.name}: {result.reason}"
        for result in inspected
        if result.reason
    ]
    if failures:
        raise MainlineError("; ".join(failures))
    return tuple(_planned_action(result) for result in inspected)


def execute_bootstrap(
    actions: tuple[BootstrapAction, ...],
    *,
    run_git: Callable[
        [tuple[str, ...], Path], subprocess.CompletedProcess[str]
    ],
) -> tuple[str, ...]:
    """Execute preflighted actions in order and return completed repositories."""

    completed = []
    for action in actions:
        arguments = _bootstrap_arguments(action)
        if arguments is None:
            completed.append(action.repository)
            continue
        try:
            result = run_git(arguments, action.root)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BootstrapExecutionError(
                f"{action.repository}: Git execution failed: {error}",
                tuple(completed),
            ) from error
        if result.returncode:
            detail = result.stderr.strip()
            raise BootstrapExecutionError(
                f"{action.repository}: Git execution failed (exit "
                f"{result.returncode}): {detail}",
                tuple(completed),
            )
        completed.append(action.repository)
    return tuple(completed)


def _emit_setup_failure(
    mode: str,
    actions: list[BootstrapAction],
    *,
    output: TextIO,
    error: BaseException,
) -> int:
    for action in actions:
        print(
            f"ERROR repository={action.name} "
            f"source={action.source_ref}@{action.source_commit} "
            f"destination={action.destination_ref} "
            f"path={action.destination_path} status=error "
            f"reason=manifest setup failed: {error}",
            file=output,
        )
    if mode == "bootstrap":
        print("BOOTSTRAP ERROR: 1 failure(s); completed=[]", file=output)
    else:
        print("CHECK ERROR: 1 setup failure(s)", file=output)
    return 2


def _run_with_contracts(
    *,
    mode: str,
    workspace_root: Path,
    manifest_path: Path,
    output: TextIO,
    expected_contracts: tuple[RepositoryContract, ...],
    git_runner: Callable | None = None,
) -> int:
    if mode not in {"bootstrap", "check"}:
        raise ValueError(f"unsupported mode: {mode}")
    selected_runner = git_runner or run_git
    resolved_workspace = workspace_root.resolve()
    try:
        actions = _manifest_actions(
            resolved_workspace,
            manifest_path,
            expected_contracts=expected_contracts,
        )
    except (MainlineError, OSError, UnicodeError, json.JSONDecodeError) as error:
        return _emit_setup_failure(
            mode,
            _actions_from_contracts(resolved_workspace, expected_contracts),
            output=output,
            error=error,
        )
    if mode == "check":
        return check_contract(
            actions,
            output=output,
            git_runner=selected_runner,
        )
    try:
        preflight = _plan_bootstrap_results(
            workspace_root=resolved_workspace,
            manifest_path=manifest_path,
            git_runner=selected_runner,
            expected_contracts=expected_contracts,
        )
    except GitInvocationError as error:
        for action in actions:
            print(
                f"ERROR repository={action.name} "
                f"source={action.source_ref}@{action.source_commit} "
                f"destination={action.destination_ref} "
                f"path={action.destination_path} status=blocked reason={error}",
                file=output,
            )
        print("BOOTSTRAP ERROR: 1 failure(s); completed=[]", file=output)
        return 2
    failures = [result for result in preflight if result.reason]
    if failures:
        for result in preflight:
            action = result.action
            reason = result.reason or "preflight aborted"
            print(
                f"ERROR repository={action.name} "
                f"source={action.source_ref}@{result.source_tip or action.source_commit} "
                f"destination={action.destination_ref} "
                f"path={action.destination_path} status=blocked reason={reason}",
                file=output,
            )
        print(
            f"BOOTSTRAP ERROR: {len(failures)} failure(s); completed=[]",
            file=output,
        )
        return 2
    for result in preflight:
        action = result.action
        print(
            f"PLAN repository={action.name} "
            f"source={action.source_ref}@{result.source_tip} "
            f"destination={action.destination_ref} "
            f"path={action.destination_path} status=ready",
            file=output,
        )
    print(f"PREFLIGHT OK: {len(preflight)} repositories", file=output)
    try:
        return _execute_bootstrap_results(
            preflight,
            output=output,
            git_runner=selected_runner,
        )
    except BootstrapExecutionError as error:
        failed_index = len(error.completed)
        failed = preflight[failed_index]
        action = failed.action
        print(
            f"ERROR repository={action.name} "
            f"source={action.source_ref}@{failed.source_tip} "
            f"destination={action.destination_ref} "
            f"path={action.destination_path} status=failed reason={error}",
            file=output,
        )
        for pending in preflight[failed_index + 1 :]:
            pending_action = pending.action
            print(
                f"ERROR repository={pending_action.name} "
                f"source={pending_action.source_ref}@{pending.source_tip} "
                f"destination={pending_action.destination_ref} "
                f"path={pending_action.destination_path} status=not-attempted "
                "reason=execution stopped",
                file=output,
            )
        print(
            "BOOTSTRAP ERROR: 1 execution failure(s); "
            f"completed={list(error.completed)!r}",
            file=output,
        )
        return 2


def run(
    mode: Literal["check", "bootstrap"],
    *,
    manifest_path: Path,
    workspace_root: Path,
    output: TextIO,
) -> int:
    """Run the exact immutable production integration-line contract."""

    return _run_with_contracts(
        mode=mode,
        workspace_root=workspace_root,
        manifest_path=manifest_path,
        output=output,
        expected_contracts=CANONICAL_REPOSITORIES,
    )


def main(argv: list[str] | None = None) -> int:
    script = Path(__file__).resolve()
    default_source = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate contracts")
    mode.add_argument(
        "--bootstrap", action="store_true", help="create local integration lines"
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="parent directory containing the canonical repositories",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_source / "docs/agents/agentic-workflow-repos.json",
        help="dual-mainline repository manifest",
    )
    arguments = parser.parse_args(argv)
    workspace_root = arguments.workspace_root
    if workspace_root is None:
        try:
            workspace_root = default_workspace_root(script, git_runner=run_git)
        except MainlineError as error:
            parser.error(str(error))
    return run(
        mode="bootstrap" if arguments.bootstrap else "check",
        workspace_root=workspace_root,
        manifest_path=arguments.manifest,
        output=sys.stdout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
