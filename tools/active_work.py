#!/usr/bin/env python3
"""Local active-work projection over Kata and observational tools."""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import posixpath
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path, PurePosixPath
from typing import NamedTuple, TextIO

SUPPORTED_MANIFEST_VERSION = "2026.08.15.1"
CANONICAL_REPOSITORIES = {
    "ultron": ("main", "ultron"),
    "torchtitan": ("main", "torchtitan"),
    "monarch": ("main", "monarch"),
    "megatronlm": ("main", "megatronlm"),
    "nccl": ("master", "nccl"),
    "ferric_continuum": ("main", "ferric-continuum"),
    "tnsr": ("master", "tnsr"),
}


class ActiveWorkError(Exception):
    """An invalid or unsafe active-work operation."""


class RepositoryConfig(NamedTuple):
    name: str
    root: Path
    mainline: str
    kata_project: str
    legacy_tracker: str


class WorkspaceConfig(NamedTuple):
    version: str
    source_repository: str
    root: Path
    repositories: tuple[RepositoryConfig, ...]


class IssueRecord(NamedTuple):
    ref: str
    project: str
    title: str
    closed: bool
    owner: str | None
    blocked: bool = False
    parked: bool = False
    attention: str | None = None
    has_implementation: bool = False
    branch: str | None = None
    session_links: tuple[tuple[str, str], ...] = ()
    scheduled_on: datetime | None = None
    legacy_path: str | None = None
    legacy_hash: str | None = None
    launcher_pid: int | None = None
    launcher_host: str | None = None


class Observation(NamedTuple):
    live: bool = False
    last_seen: datetime | None = None
    branch_valid: bool | None = None
    review_pending: bool = False


class SessionRecord(NamedTuple):
    id: str
    agent: str
    project: str
    cwd: Path
    updated_at: datetime
    active: bool


class ReviewRecord(NamedTuple):
    id: str
    branch: str
    status: str
    pending: bool
    findings: int


class ProjectionItem(NamedTuple):
    ref: str
    project: str
    state: str
    owner: str | None
    title: str


class Projection(NamedTuple):
    items: tuple[ProjectionItem, ...]
    untracked_sessions: tuple[SessionRecord, ...]
    diagnostics: tuple[str, ...]


class LegacyIssue(NamedTuple):
    repository: str
    kata_project: str
    source_path: str
    title: str
    status: str
    blocked_by: tuple[str, ...]
    parent: str | None
    content_hash: str
    idempotency_key: str
    body: str
    source_format: str
    owner: str | None


def _json_object(text: str, source: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ActiveWorkError(f"invalid {source} JSON: {error}") from error
    if not isinstance(value, dict):
        raise ActiveWorkError(f"{source} JSON must be an object")
    return value


def _load_repository_map(path: Path | None, workspace_root: Path) -> dict[str, Path] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ActiveWorkError(f"cannot read repository map {path}: {error}") from error
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and name and isinstance(value, str) and value
        for name, value in payload.items()
    ):
        raise ActiveWorkError("repository map must be an object of path strings")
    return {
        name: (Path(value) if Path(value).is_absolute() else workspace_root / value)
        for name, value in payload.items()
    }


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActiveWorkError(f"{label} must be a non-empty string")
    return value


def parse_kata_issues(text: str) -> tuple[IssueRecord, ...]:
    """Parse the stable issue subset consumed by the projection."""

    envelope = _json_object(text, "Kata")
    if envelope.get("kata_api_version") != 1:
        raise ActiveWorkError("unsupported Kata API version")
    raw_issues = envelope.get("issues")
    if not isinstance(raw_issues, list):
        raise ActiveWorkError("Kata issues must be an array")
    issues: list[IssueRecord] = []
    for raw in raw_issues:
        if not isinstance(raw, dict):
            raise ActiveWorkError("Kata issue must be an object")
        short_id = _required_text(raw.get("short_id"), "Kata issue short_id")
        project = _required_text(raw.get("project_name"), "Kata issue project_name")
        qualified_id = _required_text(
            raw.get("qualified_id"), "Kata issue qualified_id"
        )
        if qualified_id != f"{project}#{short_id}":
            raise ActiveWorkError("Kata issue qualified_id is inconsistent")
        title = _required_text(raw.get("title"), "Kata issue title")
        status = raw.get("status")
        if status not in {"open", "closed"}:
            raise ActiveWorkError("Kata issue status must be open or closed")
        owner = raw.get("owner")
        if owner is not None and (not isinstance(owner, str) or not owner):
            raise ActiveWorkError("Kata issue owner must be null or non-empty")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ActiveWorkError("Kata issue metadata must be an object")
        attention = metadata.get("work.attention")
        if attention is not None and attention not in {"ok", "needs-human", "stuck"}:
            raise ActiveWorkError("Kata work.attention is unsupported")
        blocked_by = raw.get("blocked_by", [])
        if not isinstance(blocked_by, list):
            raise ActiveWorkError("Kata issue blocked_by must be an array")
        blocker_states: list[bool] = []
        for blocker in blocked_by:
            if isinstance(blocker, dict):
                blocker_status = blocker.get("status")
                if blocker_status not in {"open", "closed"}:
                    raise ActiveWorkError(
                        "Kata blocker status must be open or closed"
                    )
                blocker_states.append(blocker_status == "open")
            elif isinstance(blocker, str) and blocker:
                blocker_states.append(True)
            else:
                raise ActiveWorkError("Kata blocker must be an object or ref")
        blocked = raw.get("blocked")
        if blocked is None:
            blocked = any(blocker_states)
        if not isinstance(blocked, bool):
            raise ActiveWorkError("Kata issue blocked must be boolean")
        someday = metadata.get("someday", False)
        if not isinstance(someday, bool):
            raise ActiveWorkError("Kata someday metadata must be boolean")
        branch = metadata.get("work.branch")
        if branch is not None and (not isinstance(branch, str) or not branch):
            raise ActiveWorkError("Kata work.branch must be null or non-empty")
        scheduled_on = raw.get("scheduled_on")
        scheduled_at: datetime | None = None
        if scheduled_on is not None:
            scheduled_text = _required_text(scheduled_on, "Kata scheduled_on")
            try:
                scheduled_at = datetime.combine(
                    date.fromisoformat(scheduled_text), time.min, tzinfo=UTC
                )
            except ValueError as error:
                raise ActiveWorkError("Kata scheduled_on must be YYYY-MM-DD") from error
        legacy_path = metadata.get("legacy.path")
        legacy_hash = metadata.get("legacy.sha256")
        for value, label in (
            (legacy_path, "legacy.path"),
            (legacy_hash, "legacy.sha256"),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ActiveWorkError(f"Kata {label} must be null or non-empty")
        launcher_pid_value = metadata.get("work.launcher_pid")
        launcher_pid: int | None = None
        if launcher_pid_value is not None:
            try:
                launcher_pid = int(launcher_pid_value)
            except (TypeError, ValueError) as error:
                raise ActiveWorkError(
                    "Kata work.launcher_pid must be an integer"
                ) from error
            if launcher_pid <= 0:
                raise ActiveWorkError("Kata work.launcher_pid must be positive")
        launcher_host = metadata.get("work.launcher_host")
        if launcher_host is not None and (
            not isinstance(launcher_host, str) or not launcher_host
        ):
            raise ActiveWorkError("Kata work.launcher_host must be non-empty")
        issues.append(
            IssueRecord(
                ref=qualified_id,
                project=project,
                title=title,
                closed=status == "closed",
                owner=owner,
                blocked=blocked,
                parked=someday,
                attention=attention if attention != "ok" else None,
                has_implementation=bool(metadata.get("work.implementation")),
                branch=branch,
                session_links=tuple(
                    sorted(
                        (key.removeprefix("work.session."), value)
                        for key, value in metadata.items()
                        if key.startswith("work.session.")
                        and isinstance(value, str)
                        and value
                    )
                ),
                scheduled_on=scheduled_at,
                legacy_path=legacy_path,
                legacy_hash=legacy_hash,
                launcher_pid=launcher_pid,
                launcher_host=launcher_host,
            )
        )
    return tuple(issues)


def _parse_utc(value: object, label: str) -> datetime:
    text = _required_text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ActiveWorkError(f"{label} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ActiveWorkError(f"{label} must include a timezone")
    return parsed


def parse_agentsview_sessions(
    text: str, *, assume_active: bool = False
) -> tuple[SessionRecord, ...]:
    envelope = _json_object(text, "AgentsView")
    raw_sessions = envelope.get("sessions")
    if not isinstance(raw_sessions, list):
        raise ActiveWorkError("AgentsView sessions must be an array")
    total = envelope.get("total")
    cursor = envelope.get("next_cursor")
    if not isinstance(total, int) or (cursor is not None and not isinstance(cursor, str)):
        raise ActiveWorkError("AgentsView pagination fields are malformed")
    sessions: list[SessionRecord] = []
    for raw in raw_sessions:
        if not isinstance(raw, dict):
            raise ActiveWorkError("AgentsView session must be an object")
        updated_at = raw.get("ended_at") or raw.get("started_at")
        termination = _required_text(
            raw.get("termination_status"), "AgentsView session termination_status"
        )
        sessions.append(
            SessionRecord(
                id=_required_text(raw.get("id"), "AgentsView session id"),
                agent=_required_text(raw.get("agent"), "AgentsView session agent"),
                project=_required_text(
                    raw.get("project"), "AgentsView session project"
                ),
                cwd=Path(_required_text(raw.get("cwd"), "AgentsView session cwd")),
                updated_at=_parse_utc(updated_at, "AgentsView session ended_at"),
                active=assume_active
                and termination not in {"completed", "failed", "cancelled"},
            )
        )
    return tuple(sessions)


def parse_roborev_reviews(text: str) -> tuple[ReviewRecord, ...]:
    try:
        raw_reviews = json.loads(text)
    except json.JSONDecodeError as error:
        raise ActiveWorkError(f"invalid RoboRev JSON: {error}") from error
    if raw_reviews is None:
        return ()
    if not isinstance(raw_reviews, list):
        raise ActiveWorkError("RoboRev JSON must be null or an array")
    reviews: list[ReviewRecord] = []
    for raw in raw_reviews:
        if not isinstance(raw, dict):
            raise ActiveWorkError("RoboRev review must be an object")
        review_id = raw.get("id")
        if not isinstance(review_id, (str, int)):
            raise ActiveWorkError("RoboRev review id must be a string or integer")
        branch = _required_text(raw.get("branch"), "RoboRev review branch")
        status = _required_text(raw.get("status"), "RoboRev review status")
        closed = raw.get("closed")
        findings = raw.get("findings")
        if not isinstance(closed, bool) or not isinstance(findings, int):
            raise ActiveWorkError("RoboRev closed/findings fields are malformed")
        reviews.append(
            ReviewRecord(
                id=str(review_id),
                branch=branch,
                status=status,
                pending=not closed,
                findings=findings,
            )
        )
    return tuple(reviews)


def _repository_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ActiveWorkError("repository name must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
        raise ActiveWorkError(f"repository points outside the workspace: {value!r}")
    return path.parts[0]


def load_workspace_config(
    manifest_path: Path,
    workspace_root: Path,
    *,
    repository_roots: dict[str, Path] | None = None,
    enforce_canonical_topology: bool = False,
) -> WorkspaceConfig:
    """Load and validate the active-work portion of the canonical manifest."""

    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ActiveWorkError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ActiveWorkError("manifest root must be an object")
    version = payload.get("version")
    source_repository = payload.get("source_repository")
    entries = payload.get("repositories")
    if version != SUPPORTED_MANIFEST_VERSION:
        raise ActiveWorkError(
            f"unsupported manifest version: expected {SUPPORTED_MANIFEST_VERSION}"
        )
    if not isinstance(source_repository, str) or not source_repository:
        raise ActiveWorkError("source_repository must be a non-empty string")
    if not isinstance(entries, list) or not entries:
        raise ActiveWorkError("repositories must be a non-empty list")

    root = workspace_root.resolve()
    if not root.is_dir():
        raise ActiveWorkError(f"workspace root is missing: {root}")
    repositories: list[RepositoryConfig] = []
    names: set[str] = set()
    projects: set[str] = set()
    resolved_roots: dict[Path, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ActiveWorkError("every repository entry must be an object")
        name = _repository_name(entry.get("name"))
        if name in names:
            raise ActiveWorkError(f"duplicate repository: {name}")
        names.add(name)
        if repository_roots is not None and name not in repository_roots:
            raise ActiveWorkError(f"repository map is missing: {name}")
        requested_root = (
            repository_roots[name] if repository_roots is not None else root / name
        )
        repository_root = requested_root.resolve()
        if not repository_root.is_relative_to(root):
            raise ActiveWorkError(f"repository points outside the workspace: {name}")
        if not repository_root.is_dir() or repository_root.is_symlink():
            raise ActiveWorkError(f"repository is missing or unsafe: {repository_root}")
        previous_name = resolved_roots.get(repository_root)
        if previous_name is not None:
            raise ActiveWorkError(
                f"repositories {previous_name} and {name} share one resolved root"
            )
        resolved_roots[repository_root] = name
        git_marker = repository_root / ".git"
        if not git_marker.exists():
            raise ActiveWorkError(f"{name}: repository root is not a Git checkout")
        completed = subprocess.run(
                ("git", "-C", str(repository_root), "rev-parse", "--git-common-dir"),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
        )
        if completed.returncode != 0:
            raise ActiveWorkError(f"{name}: cannot validate Git repository identity")
        common = Path(completed.stdout.strip())
        if not common.is_absolute():
            common = repository_root / common
        expected_common = root / name / ".git"
        if common.resolve() != expected_common.resolve():
            raise ActiveWorkError(f"{name}: repository identity does not match manifest")
        mainline = entry.get("mainline")
        kata_project = entry.get("kata_project")
        legacy_tracker = entry.get("legacy_tracker")
        if mainline not in {"main", "master"}:
            raise ActiveWorkError(f"{name}: mainline must be main or master")
        if not isinstance(kata_project, str) or not kata_project:
            raise ActiveWorkError(f"{name}: kata_project must be non-empty")
        if kata_project in projects:
            raise ActiveWorkError(f"duplicate Kata project: {kata_project}")
        projects.add(kata_project)
        if legacy_tracker not in {"markdown", "org", "none"}:
            raise ActiveWorkError(
                f"{name}: legacy_tracker must be markdown, org, or none"
            )
        repositories.append(
            RepositoryConfig(
                name=name,
                root=repository_root,
                mainline=mainline,
                kata_project=kata_project,
                legacy_tracker=legacy_tracker,
            )
        )
    if source_repository not in names:
        raise ActiveWorkError("source_repository is not a managed repository")
    if repository_roots is not None:
        unexpected = sorted(set(repository_roots) - names)
        if unexpected:
            raise ActiveWorkError(f"unexpected repository map entries: {unexpected}")
    if enforce_canonical_topology:
        observed = {
            repository.name: (repository.mainline, repository.kata_project)
            for repository in repositories
        }
        if observed != CANONICAL_REPOSITORIES:
            raise ActiveWorkError("manifest does not declare the canonical seven repositories")
    return WorkspaceConfig(
        version=version,
        source_repository=source_repository,
        root=root,
        repositories=tuple(repositories),
    )


def derive_state(
    issue: IssueRecord,
    observation: Observation,
    now: datetime,
    recent_grace: timedelta,
) -> str | None:
    """Project authoritative and observed facts into one display-only state."""

    if issue.closed:
        return None
    if issue.attention in {"needs-human", "stuck"}:
        return issue.attention
    if issue.blocked:
        return "blocked"
    if issue.parked or (
        issue.scheduled_on is not None and issue.scheduled_on > now
    ):
        return "parked"
    if observation.review_pending or (
        issue.has_implementation and observation.branch_valid is True
    ):
        return "review"
    if issue.owner and observation.live:
        return "active"
    if observation.last_seen is not None and now - observation.last_seen <= recent_grace:
        return "recent"
    if issue.owner:
        return "orphaned"
    return "ready"


def _merge_session(observation: Observation, session: SessionRecord) -> Observation:
    last_seen = observation.last_seen
    if last_seen is None or session.updated_at > last_seen:
        last_seen = session.updated_at
    return Observation(
        live=observation.live or session.active,
        last_seen=last_seen,
        branch_valid=observation.branch_valid,
        review_pending=observation.review_pending,
    )


def _session_key(session: SessionRecord) -> str:
    prefix = f"{session.agent}:"
    return session.id if session.id.startswith(prefix) else f"{prefix}{session.id}"


def build_projection(
    *,
    issues: tuple[IssueRecord, ...],
    sessions: tuple[SessionRecord, ...],
    session_links: dict[str, str],
    observations: dict[str, Observation],
    in_scope_projects: set[str],
    now: datetime,
    recent_grace: timedelta,
) -> Projection:
    """Join durable issues to explicitly linked observational evidence."""

    issue_by_ref = {issue.ref: issue for issue in issues}
    session_by_key = {_session_key(session): session for session in sessions}
    combined = dict(observations)
    diagnostics: list[str] = []
    for session_key, issue_ref in sorted(session_links.items()):
        issue = issue_by_ref.get(issue_ref)
        if issue is None:
            diagnostics.append(f"session {session_key} links to unknown issue {issue_ref}")
            continue
        session = session_by_key.get(session_key)
        if session is None:
            diagnostics.append(f"linked session {session_key} is not observed")
            continue
        combined[issue_ref] = _merge_session(
            combined.get(issue_ref, Observation()), session
        )

    items: list[ProjectionItem] = []
    for issue in issues:
        state = derive_state(
            issue,
            combined.get(issue.ref, Observation()),
            now,
            recent_grace,
        )
        if state is None:
            continue
        items.append(
            ProjectionItem(
                ref=issue.ref,
                project=issue.project,
                state=state,
                owner=issue.owner,
                title=issue.title,
            )
        )
    untracked = tuple(
        session
        for session in sessions
        if session.project in in_scope_projects
        and _session_key(session) not in session_links
    )
    return Projection(
        items=tuple(sorted(items, key=lambda item: (item.project, item.ref))),
        untracked_sessions=untracked,
        diagnostics=tuple(diagnostics),
    )


def render_projection(projection: Projection, output_format: str) -> str:
    if output_format == "json":
        payload = {
            "schema_version": 1,
            "items": [item._asdict() for item in projection.items],
            "untracked_sessions": [
                {
                    **session._asdict(),
                    "cwd": str(session.cwd),
                    "updated_at": session.updated_at.isoformat(),
                }
                for session in projection.untracked_sessions
            ],
            "diagnostics": list(projection.diagnostics),
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_format == "markdown":
        lines = [
            "| Issue | State | Owner | Title |",
            "| --- | --- | --- | --- |",
        ]
        lines.extend(
            f"| {item.ref} | {item.state} | {item.owner or '-'} | {item.title} |"
            for item in projection.items
        )
        if projection.untracked_sessions:
            lines.extend(["", "## Untracked sessions", ""])
            lines.extend(
                f"- `{_session_key(session)}` ({session.project})"
                for session in projection.untracked_sessions
            )
        if projection.diagnostics:
            lines.extend(["", "## Diagnostics", ""])
            lines.extend(f"- {message}" for message in projection.diagnostics)
        return "\n".join(lines) + "\n"
    if output_format == "human":
        lines = [
            f"{item.state:12} {item.ref:16} {item.owner or '-':12} {item.title}"
            for item in projection.items
        ]
        lines.extend(
            f"untracked    {_session_key(session)} ({session.project})"
            for session in projection.untracked_sessions
        )
        lines.extend(f"diagnostic   {message}" for message in projection.diagnostics)
        return "\n".join(lines) + ("\n" if lines else "")
    raise ActiveWorkError(f"unsupported output format: {output_format}")


def _run_process(
    arguments: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: float | None = 15,
) -> int:
    executable = _resolve_executable(arguments[0], environment)
    command = (executable, *arguments[1:])
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise ActiveWorkError(
            f"{arguments[0]} timed out after {timeout} seconds"
        ) from error
    except OSError as error:
        raise ActiveWorkError(f"cannot run {arguments[0]}: {error}") from error
    return completed.returncode


def _resolve_executable(name: str, environment: dict[str, str]) -> str:
    resolved = shutil.which(name, path=environment.get("PATH"))
    if resolved:
        return resolved
    local = Path.home() / ".local/bin" / name
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    raise ActiveWorkError(f"required executable is unavailable: {name}")


def _capture_process(
    arguments: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: float = 15,
) -> str:
    executable = _resolve_executable(arguments[0], environment)
    try:
        completed = subprocess.run(
            (executable, *arguments[1:]),
            cwd=cwd,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise ActiveWorkError(
            f"{arguments[0]} timed out after {timeout} seconds"
        ) from error
    except OSError as error:
        raise ActiveWorkError(f"cannot run {arguments[0]}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ActiveWorkError(
            f"{arguments[0]} exited {completed.returncode}: {detail}"
        )
    return completed.stdout


def adopt_session(
    *,
    issue_ref: str,
    agent: str,
    session_id: str,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> int:
    """Attach an explicitly selected session to an existing Kata issue."""

    _required_text(issue_ref, "issue ref")
    _required_text(session_id, "session id")
    if not re.fullmatch(r"[a-z0-9_-]+", agent):
        raise ActiveWorkError("agent must contain only lowercase letters, digits, _ or -")
    process_environment = dict(environment or os.environ)
    existing = parse_kata_issues(
        _capture_process(
            ("kata", "list", "--all", "--status", "all", "--json"),
            cwd=cwd,
            environment=process_environment,
        )
    )
    for issue in existing:
        if issue.ref == issue_ref:
            continue
        if (agent, session_id) in issue.session_links:
            status = _run_process(
                ("kata", "meta", "unset", issue.ref, f"work.session.{agent}"),
                cwd=cwd,
                environment=process_environment,
            )
            if status != 0:
                return status
    return _run_process(
        (
            "kata",
            "meta",
            "set",
            issue_ref,
            f"work.session.{agent}",
            session_id,
            "--agent",
        ),
        cwd=cwd,
        environment=process_environment,
    )


def run_tracked_session(
    *,
    issue_ref: str,
    command: tuple[str, ...],
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> int:
    """Run one command between Kata attention lifecycle hooks."""

    _required_text(issue_ref, "issue ref")
    if not command or not all(isinstance(part, str) and part for part in command):
        raise ActiveWorkError("tracked command must contain non-empty arguments")
    child_environment = dict(environment or os.environ)
    child_environment["KATA_REF"] = issue_ref
    start_status = _run_process(
        ("kata", "attention-hook", "start"),
        cwd=cwd,
        environment=child_environment,
    )
    if start_status != 0:
        return start_status
    for key, value in (
        ("work.launcher_pid", str(os.getpid())),
        ("work.launcher_host", socket.gethostname()),
    ):
        metadata_status = _run_process(
            ("kata", "meta", "set", issue_ref, key, value, "--agent"),
            cwd=cwd,
            environment=child_environment,
        )
        if metadata_status != 0:
            _run_process(
                ("kata", "attention-hook", "end"),
                cwd=cwd,
                environment=child_environment,
            )
            return metadata_status
    child_status: int | None = None
    try:
        child_status = _run_process(
            command,
            cwd=cwd,
            environment=child_environment,
            timeout=None,
        )
    finally:
        try:
            for key in ("work.launcher_pid", "work.launcher_host"):
                _run_process(
                    ("kata", "meta", "unset", issue_ref, key, "--agent"),
                    cwd=cwd,
                    environment=child_environment,
                )
        finally:
            end_status = _run_process(
                ("kata", "attention-hook", "end"),
                cwd=cwd,
                environment=child_environment,
            )
    assert child_status is not None
    return child_status if child_status != 0 else end_status


def validate_start_context(
    *,
    issue_ref: str,
    repository: RepositoryConfig,
    environment: dict[str, str],
) -> None:
    """Require an open, claimed issue in the selected Git worktree."""

    if not issue_ref.startswith(f"{repository.kata_project}#"):
        raise ActiveWorkError(
            f"issue {issue_ref} does not belong to {repository.kata_project}"
        )
    try:
        git_root = subprocess.run(
            ("git", "-C", str(repository.root), "rev-parse", "--show-toplevel"),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ActiveWorkError(f"invalid Git worktree {repository.root}: {error}") from error
    if Path(git_root).resolve() != repository.root.resolve():
        raise ActiveWorkError(f"Git worktree root mismatch: {git_root}")
    if not (repository.root / ".git").is_file():
        raise ActiveWorkError("start requires an isolated linked Git worktree")
    envelope = _json_object(
        _capture_process(
            ("kata", "show", issue_ref, "--json"),
            cwd=repository.root,
            environment=environment,
        ),
        "Kata show",
    )
    if envelope.get("kata_api_version") != 1 or not isinstance(
        envelope.get("issue"), dict
    ):
        raise ActiveWorkError("malformed Kata show response")
    issue = envelope["issue"]
    if issue.get("status") != "open":
        raise ActiveWorkError(f"issue is not open: {issue_ref}")
    if not isinstance(issue.get("owner"), str) or not issue["owner"]:
        raise ActiveWorkError(f"issue must be claimed before start: {issue_ref}")
    metadata = issue.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ActiveWorkError("Kata show metadata must be an object")
    expected_branch = metadata.get("work.branch")
    if not isinstance(expected_branch, str) or not expected_branch:
        raise ActiveWorkError("issue must declare work.branch before start")
    try:
        current_branch = subprocess.run(
            ("git", "-C", str(repository.root), "branch", "--show-current"),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ActiveWorkError(f"cannot inspect worktree branch: {error}") from error
    if current_branch != expected_branch:
        raise ActiveWorkError(
            f"issue branch {expected_branch} does not match {current_branch}"
        )


_RESOLVED_LEGACY_STATUSES = frozenset(
    {"resolved", "done", "closed", "cancelled", "canceled"}
)


def _legacy_field(text: str, field: str, tracker: str) -> str | None:
    if tracker == "org":
        prefix = f"#+{field.upper()}:"
    else:
        prefix = f"{field.title()}:"
    for line in text.splitlines():
        if line.upper().startswith(prefix.upper()):
            return line[len(prefix) :].strip()
    return None


def _legacy_title(text: str, tracker: str, path: Path) -> str:
    if tracker == "org":
        title = _legacy_field(text, "title", tracker)
    else:
        title = next(
            (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
            None,
        )
    if not title:
        raise ActiveWorkError(f"legacy issue is missing title: {path}")
    return title


def _assert_plain_file(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ActiveWorkError(f"legacy issue path contains a symlink: {current}")
    if not path.is_file():
        raise ActiveWorkError(f"legacy issue is not a regular file: {path}")


def collect_legacy_issues(
    config: WorkspaceConfig, *, include_resolved: bool = False
) -> tuple[LegacyIssue, ...]:
    """Read unresolved legacy issues without mutating their source files."""

    collected: list[LegacyIssue] = []
    for repository in config.repositories:
        if repository.legacy_tracker == "none":
            continue
        suffix = ".org" if repository.legacy_tracker == "org" else ".md"
        paths = [
            *repository.root.glob(f".scratch/*/issues/*{suffix}"),
            *repository.root.glob(f".scratch/*/spec{suffix}"),
        ]
        for path in sorted(paths):
            _assert_plain_file(repository.root, path)
            text = path.read_text()
            status = _legacy_field(text, "status", repository.legacy_tracker)
            if not status:
                raise ActiveWorkError(f"legacy issue is missing status: {path}")
            if status.lower() in _RESOLVED_LEGACY_STATUSES and not include_resolved:
                continue
            blocked_text = _legacy_field(
                text, "blocked_by", repository.legacy_tracker
            )
            if blocked_text is None:
                blocked_text = _legacy_field(
                    text, "blocked by", repository.legacy_tracker
                )
            blocked_by = tuple(
                value.strip()
                for value in (blocked_text or "").split(",")
                if value.strip().lower() not in {"", "-", "none", "n/a"}
            )
            parent = _legacy_field(text, "parent", repository.legacy_tracker)
            owner = _legacy_field(text, "owner", repository.legacy_tracker)
            if owner is None:
                owner = _legacy_field(text, "assignee", repository.legacy_tracker)
            relative = path.relative_to(repository.root).as_posix()
            collected.append(
                LegacyIssue(
                    repository=repository.name,
                    kata_project=repository.kata_project,
                    source_path=relative,
                    title=_legacy_title(text, repository.legacy_tracker, path),
                    status=status,
                    blocked_by=blocked_by,
                    parent=parent or None,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    idempotency_key=f"legacy-v1:{repository.name}:{relative}",
                    body=text,
                    source_format=repository.legacy_tracker,
                    owner=(
                        owner
                        or (
                            "legacy-unattributed"
                            if status.lower() == "claimed"
                            else None
                        )
                    ),
                )
            )
    return tuple(
        sorted(collected, key=lambda issue: (issue.repository, issue.source_path))
    )


def _created_issue_ref(text: str, project: str) -> str:
    envelope = _json_object(text, "Kata create")
    if envelope.get("kata_api_version") != 1:
        raise ActiveWorkError("unsupported Kata create API version")
    issue = envelope.get("issue")
    if not isinstance(issue, dict):
        raise ActiveWorkError("Kata create response is missing issue")
    qualified_id = issue.get("qualified_id")
    if isinstance(qualified_id, str) and qualified_id:
        return qualified_id
    short_id = _required_text(issue.get("short_id"), "Kata created issue short_id")
    return f"{project}#{short_id}"


def _resolve_legacy_dependency(
    issue: LegacyIssue,
    token: str,
    refs_by_source: dict[tuple[str, str], str],
    repository_aliases: dict[str, str],
) -> str | None:
    if "#" in token:
        return token
    target_repository = issue.repository
    target_token = token
    if ":" in token:
        prefix, remainder = token.split(":", 1)
        mapped_repository = repository_aliases.get(prefix)
        if mapped_repository is not None:
            target_repository = mapped_repository
            target_token = remainder
    directory = PurePosixPath(issue.source_path).parent
    candidate = (
        posixpath.normpath(target_token)
        if target_token.startswith(".scratch/")
        else posixpath.normpath(f"{directory.as_posix()}/{target_token}")
    )
    direct = refs_by_source.get((target_repository, candidate))
    if direct is not None:
        return direct
    normalized = PurePosixPath(candidate).stem
    candidate_directory = PurePosixPath(candidate).parent
    matches = [
        ref
        for (repository, source_path), ref in refs_by_source.items()
        if repository == target_repository
        and PurePosixPath(source_path).parent == candidate_directory
        and (
            PurePosixPath(source_path).stem == normalized
            or PurePosixPath(source_path).stem.startswith(f"{normalized}-")
        )
    ]
    return matches[0] if len(matches) == 1 else None


def apply_legacy_issues(
    issues: tuple[LegacyIssue, ...],
    *,
    repositories: dict[str, RepositoryConfig],
    environment: dict[str, str] | None = None,
    existing_refs_by_source: dict[tuple[str, str], str] | None = None,
) -> tuple[str, ...]:
    """Import legacy records, then add edges after all stable refs exist."""

    process_environment = dict(environment or os.environ)
    repository_aliases = {
        alias: repository.name
        for repository in repositories.values()
        for alias in (repository.name, repository.kata_project)
    }
    refs_by_source: dict[tuple[str, str], str] = dict(existing_refs_by_source or {})
    diagnostics: list[str] = []
    for issue in issues:
        repository = repositories.get(issue.repository)
        if repository is None:
            raise ActiveWorkError(f"unknown legacy repository: {issue.repository}")
        source_key = (issue.kata_project, issue.source_path)
        existing_ref = refs_by_source.get(source_key)
        body = (
            f"Imported from `{issue.source_path}`. The source remains read-only.\n\n"
            f"{issue.body}"
        )
        create_arguments = [
                "kata",
                "--workspace",
                str(repository.root),
                "--project",
                issue.kata_project,
                "create",
                issue.title,
                "--body",
                body,
                "--idempotency-key",
                issue.idempotency_key,
                "--label",
                f"legacy-status:{issue.status}",
                "--meta",
                f"legacy.path={issue.source_path}",
                "--meta",
                f"legacy.status={issue.status}",
                "--meta",
                f"legacy.sha256={issue.content_hash}",
                "--meta",
                f"legacy.format={issue.source_format}",
                "--json",
        ]
        if issue.owner:
            create_arguments[-1:-1] = ["--owner", issue.owner]
        if existing_ref is None:
            response = _capture_process(
                tuple(create_arguments),
                cwd=repository.root,
                environment=process_environment,
            )
            existing_ref = _created_issue_ref(response, issue.kata_project)
        refs_by_source[(issue.repository, issue.source_path)] = existing_ref
        refs_by_source[source_key] = existing_ref
        _capture_process(
            (
                "kata",
                "--workspace",
                str(repository.root),
                "meta",
                "set",
                refs_by_source[(issue.repository, issue.source_path)],
                "legacy.format",
                issue.source_format,
                "--json",
            ),
            cwd=repository.root,
            environment=process_environment,
        )

    for issue in issues:
        repository = repositories[issue.repository]
        issue_ref = refs_by_source[(issue.repository, issue.source_path)]
        if issue.owner:
            _capture_process(
                (
                    "kata",
                    "--workspace",
                    str(repository.root),
                    "edit",
                    issue_ref,
                    "--owner",
                    issue.owner,
                    "--json",
                ),
                cwd=repository.root,
                environment=process_environment,
            )
        for blocker in issue.blocked_by:
            blocker_ref = _resolve_legacy_dependency(
                issue, blocker, refs_by_source, repository_aliases
            )
            if blocker_ref is None:
                diagnostics.append(
                    f"{issue.source_path}: unresolved blocker {blocker!r}"
                )
                continue
            _capture_process(
                (
                    "kata",
                    "--workspace",
                    str(repository.root),
                    "edit",
                    issue_ref,
                    "--blocked-by",
                    blocker_ref,
                    "--json",
                ),
                cwd=repository.root,
                environment=process_environment,
            )
        if issue.parent:
            parent_ref = _resolve_legacy_dependency(
                issue, issue.parent, refs_by_source, repository_aliases
            )
            if parent_ref is None:
                diagnostics.append(
                    f"{issue.source_path}: unresolved parent {issue.parent!r}"
                )
            else:
                _capture_process(
                    (
                        "kata",
                        "--workspace",
                        str(repository.root),
                        "edit",
                        issue_ref,
                        "--parent",
                        parent_ref,
                        "--json",
                    ),
                    cwd=repository.root,
                    environment=process_environment,
                )
    return tuple(diagnostics)


def _binding_text(project: str) -> str:
    return f'version = 1\n\n[project]\nname = "{project}"\n'


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _known_kata_projects(
    source_root: Path, environment: dict[str, str]
) -> set[str]:
    text = _capture_process(
        ("kata", "projects", "list", "--json"),
        cwd=source_root,
        environment=environment,
    )
    envelope = _json_object(text, "Kata projects")
    if envelope.get("kata_api_version") != 1:
        raise ActiveWorkError("unsupported Kata projects API version")
    projects = envelope.get("projects")
    if not isinstance(projects, list):
        raise ActiveWorkError("Kata projects must be an array")
    names: set[str] = set()
    for project in projects:
        if not isinstance(project, dict):
            raise ActiveWorkError("Kata project must be an object")
        names.add(_required_text(project.get("name"), "Kata project name"))
    return names


def configure_workspace(
    config: WorkspaceConfig,
    *,
    apply: bool,
    environment: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Check or apply managed bindings and local Kata project registration."""

    process_environment = dict(environment or os.environ)
    source = next(
        repository
        for repository in config.repositories
        if repository.name == config.source_repository
    )
    known_projects = _known_kata_projects(source.root, process_environment)
    drift: list[str] = []
    for repository in config.repositories:
        binding = repository.root / ".kata.toml"
        expected = _binding_text(repository.kata_project)
        if not binding.is_file() or binding.is_symlink() or binding.read_text() != expected:
            drift.append(f"{repository.name}/.kata.toml")
            if apply:
                if binding.is_symlink() or (binding.exists() and not binding.is_file()):
                    raise ActiveWorkError(f"unsafe Kata binding: {binding}")
                _atomic_text(binding, expected)
        if repository.kata_project not in known_projects:
            drift.append(f"kata-project:{repository.kata_project}")
            if apply:
                status = _run_process(
                    (
                        "kata",
                        "projects",
                        "create",
                        repository.kata_project,
                        "--json",
                    ),
                    cwd=source.root,
                    environment=process_environment,
                )
                if status != 0:
                    raise ActiveWorkError(
                        f"cannot create Kata project {repository.kata_project}"
                    )
                known_projects.add(repository.kata_project)
    return tuple(drift)


def doctor_workspace(
    config: WorkspaceConfig,
    *,
    environment: dict[str, str] | None = None,
) -> tuple[tuple[str, ...], bool]:
    """Report required and optional local control-plane dependencies."""

    process_environment = dict(environment or os.environ)
    lines: list[str] = []
    healthy = True
    available: set[str] = set()
    for tool in ("kata", "agentsview", "roborev"):
        try:
            path = _resolve_executable(tool, process_environment)
        except ActiveWorkError:
            lines.append(f"MISSING required {tool}")
            healthy = False
        else:
            lines.append(f"OK required {tool} {path}")
            available.add(tool)
    for tool in ("kenn-forge", "ghosthub"):
        try:
            path = _resolve_executable(tool, process_environment)
        except ActiveWorkError:
            lines.append(f"OPTIONAL {tool} unavailable")
        else:
            lines.append(f"OK optional {tool} {path}")
    for repository in config.repositories:
        binding = repository.root / ".kata.toml"
        expected = _binding_text(repository.kata_project)
        if binding.is_file() and not binding.is_symlink() and binding.read_text() == expected:
            lines.append(f"OK binding {repository.name}/.kata.toml")
        else:
            lines.append(f"MISSING binding {repository.name}/.kata.toml")
            healthy = False
    source = next(
        repository
        for repository in config.repositories
        if repository.name == config.source_repository
    )
    schema_checks = {
        "kata": lambda: _known_kata_projects(source.root, process_environment),
        "agentsview": lambda: parse_agentsview_sessions(
            _capture_process(
                (
                    "agentsview",
                    "session",
                    "list",
                    "--json",
                    "--active",
                    "--limit",
                    "1",
                ),
                cwd=source.root,
                environment=process_environment,
            ),
            assume_active=True,
        ),
        "roborev": lambda: parse_roborev_reviews(
            _capture_process(
                ("roborev", "list", "--json", "--limit", "1"),
                cwd=source.root,
                environment=process_environment,
            )
        ),
    }
    for tool, check in schema_checks.items():
        if tool not in available:
            continue
        try:
            check()
        except ActiveWorkError as error:
            lines.append(f"FAIL schema {tool}: {error}")
            healthy = False
        else:
            lines.append(f"OK schema {tool}")
    return tuple(lines), healthy


def _git_branch_valid(repository: RepositoryConfig, branch: str) -> bool | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository.root), "worktree", "list", "--porcelain"),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    records = completed.stdout.split("\n\n")
    for record in records:
        fields = dict(
            line.split(" ", 1)
            for line in record.splitlines()
            if " " in line
        )
        path = fields.get("worktree")
        if fields.get("branch") == f"refs/heads/{branch}" and path:
            return Path(path).is_dir()
    return False


def _launcher_live(issue: IssueRecord) -> bool:
    if issue.launcher_pid is None or issue.launcher_host != socket.gethostname():
        return False
    try:
        os.kill(issue.launcher_pid, 0)
    except (OSError, PermissionError):
        return False
    return True


def _legacy_reconciliation_diagnostics(
    config: WorkspaceConfig,
    issues: tuple[IssueRecord, ...],
) -> tuple[str, ...]:
    imported = {
        (issue.project, issue.legacy_path): issue
        for issue in issues
        if issue.legacy_path is not None
    }
    diagnostics: list[str] = []
    main_repositories = tuple(
        repository._replace(root=config.root / repository.name)
        for repository in config.repositories
    )
    main_config = config._replace(repositories=main_repositories)
    all_sources: dict[tuple[str, str], LegacyIssue] = {}
    unresolved_sources: dict[tuple[str, str], LegacyIssue] = {}
    for source_config in (main_config, config):
        for legacy in collect_legacy_issues(source_config, include_resolved=True):
            all_sources[(legacy.kata_project, legacy.source_path)] = legacy
        for legacy in collect_legacy_issues(source_config):
            unresolved_sources[(legacy.kata_project, legacy.source_path)] = legacy
    for key, legacy in unresolved_sources.items():
        issue = imported.get((legacy.kata_project, legacy.source_path))
        if issue is None:
            diagnostics.append(
                f"legacy source is not imported: {legacy.repository}/{legacy.source_path}"
            )
    for key, issue in imported.items():
        legacy = all_sources.get(key)
        if legacy is None:
            diagnostics.append(
                f"legacy source is missing: {key[0]}/{key[1]} ({issue.ref})"
            )
        elif issue.legacy_hash != legacy.content_hash:
            diagnostics.append(
                f"legacy source hash drift: {legacy.repository}/{legacy.source_path}"
            )
    return tuple(diagnostics)


def _pending_legacy_issues(
    config: WorkspaceConfig,
    issues: tuple[LegacyIssue, ...],
    *,
    environment: dict[str, str],
) -> tuple[LegacyIssue, ...]:
    source = next(
        repository
        for repository in config.repositories
        if repository.name == config.source_repository
    )
    imported = parse_kata_issues(
        _capture_process(
            (
                "kata",
                "--workspace",
                str(source.root),
                "list",
                "--all",
                "--status",
                "all",
                "--json",
            ),
            cwd=source.root,
            environment=environment,
        )
    )
    hashes = {
        (issue.project, issue.legacy_path): issue.legacy_hash
        for issue in imported
        if issue.legacy_path is not None
    }
    return tuple(
        issue
        for issue in issues
        if hashes.get((issue.kata_project, issue.source_path)) != issue.content_hash
    )


def _existing_legacy_refs(
    config: WorkspaceConfig,
    *,
    environment: dict[str, str],
) -> dict[tuple[str, str], str]:
    source = next(
        repository
        for repository in config.repositories
        if repository.name == config.source_repository
    )
    issues = parse_kata_issues(
        _capture_process(
            (
                "kata",
                "--workspace",
                str(source.root),
                "list",
                "--all",
                "--status",
                "all",
                "--json",
            ),
            cwd=source.root,
            environment=environment,
        )
    )
    return {
        (issue.project, issue.legacy_path): issue.ref
        for issue in issues
        if issue.legacy_path is not None
    }


def _collect_status(
    config: WorkspaceConfig,
    *,
    environment: dict[str, str],
    now: datetime,
    reconcile: bool = False,
) -> Projection:
    source = next(
        repository
        for repository in config.repositories
        if repository.name == config.source_repository
    )
    issues = parse_kata_issues(
        _capture_process(
            (
                "kata",
                "--workspace",
                str(source.root),
                "list",
                "--all",
                "--status",
                "all",
                "--json",
            ),
            cwd=source.root,
            environment=environment,
        )
    )
    adapter_diagnostics: list[str] = []
    try:
        active_sessions = parse_agentsview_sessions(
            _capture_process(
                (
                "agentsview",
                "session",
                "list",
                "--json",
                "--active",
                "--limit",
                "500",
                "--include-children",
                "--include-one-shot",
                "--include-automated",
                ),
                cwd=source.root,
                environment=environment,
            ),
            assume_active=False,
        )
        recent_sessions = parse_agentsview_sessions(
            _capture_process(
                (
                    "agentsview",
                    "session",
                    "list",
                    "--json",
                    "--since",
                    "1h",
                    "--limit",
                    "500",
                    "--include-children",
                    "--include-one-shot",
                    "--include-automated",
                ),
                cwd=source.root,
                environment=environment,
            ),
            assume_active=False,
        )
        sessions_by_key = {_session_key(session): session for session in recent_sessions}
        sessions_by_key.update(
            {_session_key(session): session for session in active_sessions}
        )
        sessions = tuple(sessions_by_key.values())
    except ActiveWorkError as error:
        sessions = ()
        adapter_diagnostics.append(f"AgentsView unavailable: {error}")
    session_links: dict[str, str] = {}
    observations: dict[str, Observation] = {}
    repository_by_project = {
        repository.kata_project: repository for repository in config.repositories
    }
    diagnostics: list[str] = []
    for issue in issues:
        if issue.closed:
            continue
        if _launcher_live(issue):
            observations[issue.ref] = Observation(live=True)
        for agent, session_id in issue.session_links:
            key = session_id if session_id.startswith(f"{agent}:") else f"{agent}:{session_id}"
            session_links[key] = issue.ref
        if issue.branch:
            repository = repository_by_project.get(issue.project)
            branch_valid = (
                _git_branch_valid(repository, issue.branch)
                if repository is not None
                else False
            )
            current = observations.get(issue.ref, Observation())
            observations[issue.ref] = current._replace(branch_valid=branch_valid)
            if branch_valid is None:
                diagnostics.append(
                    f"Git evidence unavailable for {issue.ref}:{issue.branch}"
                )
            elif not branch_valid:
                diagnostics.append(f"{issue.ref}: invalid branch {issue.branch}")

    issues_by_branch = {
        (issue.project, issue.branch): issue.ref
        for issue in issues
        if issue.branch is not None
    }
    for repository in config.repositories:
        try:
            reviews = parse_roborev_reviews(
                _capture_process(
                    (
                    "roborev",
                    "list",
                    "--repo",
                    str(repository.root),
                    "--open",
                    "--json",
                    ),
                    cwd=repository.root,
                    environment=environment,
                )
            )
        except ActiveWorkError as error:
            adapter_diagnostics.append(
                f"RoboRev unavailable for {repository.name}: {error}"
            )
            continue
        for review in reviews:
            issue_ref = issues_by_branch.get(
                (repository.kata_project, review.branch)
            )
            if issue_ref is None:
                diagnostics.append(
                    f"untracked RoboRev review {review.id} on "
                    f"{repository.name}:{review.branch}"
                )
                continue
            current = observations.get(issue_ref, Observation())
            observations[issue_ref] = current._replace(
                review_pending=current.review_pending or review.pending
            )
            if review.findings:
                diagnostics.append(
                    f"RoboRev review {review.id} has {review.findings} finding(s) "
                    f"for {issue_ref}"
                )
    projection = build_projection(
        issues=issues,
        sessions=sessions,
        session_links=session_links,
        observations=observations,
        in_scope_projects={
            value
            for repository in config.repositories
            for value in (repository.name, repository.kata_project)
        },
        now=now,
        recent_grace=timedelta(hours=1),
    )
    reconciliation = (
        _legacy_reconciliation_diagnostics(config, issues) if reconcile else ()
    )
    return projection._replace(
        diagnostics=(
            projection.diagnostics
            + tuple(diagnostics)
            + tuple(adapter_diagnostics)
            + reconciliation
        )
    )


def _default_workspace_root(repository_root: Path) -> Path:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository_root), "rev-parse", "--git-common-dir"),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        common = Path(completed.stdout.strip())
        if not common.is_absolute():
            common = repository_root / common
        return common.resolve().parent.parent
    except (OSError, subprocess.SubprocessError):
        return repository_root.parent


def _discover_matching_worktrees(
    manifest_path: Path,
    workspace_root: Path,
    repository_root: Path,
) -> dict[str, Path] | None:
    """Use the current linked-worktree branch across all managed repositories."""

    if not (repository_root / ".git").is_file():
        return None
    try:
        branch = subprocess.run(
            ("git", "-C", str(repository_root), "branch", "--show-current"),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        ).stdout.strip()
        manifest = _json_object(manifest_path.read_text(), "manifest")
        entries = manifest.get("repositories")
        if not branch or not isinstance(entries, list):
            return None
        discovered: dict[str, Path] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                return None
            name = _repository_name(entry.get("name"))
            main_root = workspace_root / name
            listing = subprocess.run(
                ("git", "-C", str(main_root), "worktree", "list", "--porcelain"),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            ).stdout
            matches: list[Path] = []
            for record in listing.split("\n\n"):
                fields = dict(
                    line.split(" ", 1)
                    for line in record.splitlines()
                    if " " in line
                )
                if fields.get("branch") == f"refs/heads/{branch}" and fields.get(
                    "worktree"
                ):
                    matches.append(Path(fields["worktree"]))
            if len(matches) > 1:
                raise ActiveWorkError(
                    f"{name}: multiple worktrees use branch {branch}"
                )
            discovered[name] = matches[0] if matches else main_root
        return discovered
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ActiveWorkError(f"cannot discover matching worktrees: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "docs/agents/agentic-workflow-repos.json",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=_default_workspace_root(repository_root),
    )
    parser.add_argument(
        "--repository-map",
        type=Path,
        help="JSON object mapping repository names to repo-local worktrees",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    configure = subparsers.add_parser("configure")
    configure_mode = configure.add_mutually_exclusive_group(required=True)
    configure_mode.add_argument("--check", action="store_true")
    configure_mode.add_argument("--apply", action="store_true")
    subparsers.add_parser("doctor")
    import_legacy = subparsers.add_parser("import-legacy")
    import_mode = import_legacy.add_mutually_exclusive_group(required=True)
    import_mode.add_argument("--check", action="store_true")
    import_mode.add_argument("--apply", action="store_true")
    adopt = subparsers.add_parser("adopt-session")
    adopt.add_argument("--issue", required=True)
    adopt.add_argument("--agent", required=True)
    adopt.add_argument("--session", required=True)
    adopt.add_argument("--repository")
    start = subparsers.add_parser("start")
    start.add_argument("--issue", required=True)
    start.add_argument("--repository")
    start.add_argument("child_command", nargs=argparse.REMAINDER)
    status = subparsers.add_parser("status")
    status.add_argument("--format", choices=("human", "json", "markdown"), default="human")
    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("--format", choices=("human", "json", "markdown"), default="human")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    output: TextIO = sys.stdout,
    error: TextIO = sys.stderr,
    environment: dict[str, str] | None = None,
    now: datetime | None = None,
    enforce_canonical_topology: bool = True,
) -> int:
    arguments = _parser().parse_args(argv)
    process_environment = dict(environment or os.environ)
    try:
        repository_root = Path(__file__).resolve().parents[1]
        repository_roots = _load_repository_map(
            arguments.repository_map, arguments.workspace_root.resolve()
        )
        if repository_roots is None:
            repository_roots = _discover_matching_worktrees(
                arguments.manifest,
                arguments.workspace_root.resolve(),
                repository_root,
            )
        config = load_workspace_config(
            arguments.manifest,
            arguments.workspace_root,
            repository_roots=repository_roots,
            enforce_canonical_topology=enforce_canonical_topology,
        )
        repository_by_name = {
            repository.name: repository for repository in config.repositories
        }
        source = repository_by_name[config.source_repository]
        if arguments.command == "configure":
            drift = configure_workspace(
                config, apply=arguments.apply, environment=process_environment
            )
            for item in drift:
                print(("APPLY " if arguments.apply else "DRIFT ") + item, file=output)
            if arguments.apply:
                print(f"configured {len(drift)} item(s)", file=output)
                return 0
            return 1 if drift else 0
        if arguments.command == "doctor":
            lines, healthy = doctor_workspace(config, environment=process_environment)
            for line in lines:
                print(line, file=output)
            return 0 if healthy else 1
        if arguments.command == "import-legacy":
            issues = collect_legacy_issues(config)
            if arguments.check:
                pending = _pending_legacy_issues(
                    config, issues, environment=process_environment
                )
                for issue in pending:
                    print(
                        f"PLAN {issue.kata_project} {issue.source_path} {issue.title}",
                        file=output,
                    )
                return 1 if pending else 0
            diagnostics = apply_legacy_issues(
                issues,
                repositories=repository_by_name,
                environment=process_environment,
                existing_refs_by_source=_existing_legacy_refs(
                    config, environment=process_environment
                ),
            )
            print(f"imported {len(issues)} legacy issue(s)", file=output)
            for diagnostic in diagnostics:
                print(f"DIAGNOSTIC {diagnostic}", file=output)
            return 0
        if arguments.command in {"adopt-session", "start"}:
            repository = source
            if arguments.repository:
                repository = repository_by_name.get(arguments.repository)
                if repository is None:
                    raise ActiveWorkError(
                        f"unknown repository: {arguments.repository}"
                    )
            if arguments.command == "adopt-session":
                return adopt_session(
                    issue_ref=arguments.issue,
                    agent=arguments.agent,
                    session_id=arguments.session,
                    cwd=repository.root,
                    environment=process_environment,
                )
            validate_start_context(
                issue_ref=arguments.issue,
                repository=repository,
                environment=process_environment,
            )
            command = tuple(arguments.child_command)
            if command[:1] == ("--",):
                command = command[1:]
            return run_tracked_session(
                issue_ref=arguments.issue,
                command=command,
                cwd=repository.root,
                environment=process_environment,
            )
        projection = _collect_status(
            config,
            environment=process_environment,
            now=now or datetime.now().astimezone(),
            reconcile=arguments.command == "reconcile",
        )
        print(render_projection(projection, arguments.format), end="", file=output)
        return 0
    except ActiveWorkError as failure:
        print(f"ERROR {failure}", file=error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
