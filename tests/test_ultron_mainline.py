import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tools import ultron_mainline


def _repository_root(test_file=Path(__file__)):
    resolved_test_file = Path(test_file).resolve()
    tool_marker = Path("tools/ultron_mainline.py")
    manifest_marker = Path("docs/agents/agentic-workflow-repos.json")
    for ancestor in resolved_test_file.parents:
        if (ancestor / tool_marker).is_file() and (
            ancestor / manifest_marker
        ).is_file():
            return ancestor
    raise AssertionError(
        f"could not find repository root from {resolved_test_file}; expected an "
        f"ancestor containing {tool_marker} and {manifest_marker}"
    )


class UltronMainlineTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.workspace = Path(self.tempdir.name)
        self.repositories = ("ultron", "peer")
        self.bootstrap_commits = {}
        for name in self.repositories:
            repository = self.workspace / name
            repository.mkdir()
            self.git("init", "-b", "main", cwd=repository)
            self.git("config", "user.name", "Test User", cwd=repository)
            self.git("config", "user.email", "test@example.com", cwd=repository)
            (repository / "README.md").write_text(f"{name}\n")
            self.git("add", "README.md", cwd=repository)
            self.git("commit", "-m", "bootstrap", cwd=repository)
            self.bootstrap_commits[name] = self.git(
                "rev-parse", "HEAD", cwd=repository
            ).stdout.strip()

        self.contracts = tuple(
            ultron_mainline.RepositoryContract(
                name,
                self.workspace / name,
                "main",
                "ultron/mainline",
                self.bootstrap_commits[name],
            )
            for name in self.repositories
        )

        self.manifest = self.workspace / "manifest.json"
        self.write_manifest()

    def git(self, *arguments, cwd, check=True):
        return subprocess.run(
            ("git", *arguments),
            cwd=cwd,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write_manifest(self):
        self.manifest.write_text(
            json.dumps(
                {
                    "version": "2026.08.15.2",
                    "source_repository": "ultron",
                    "managed_policy_files": ["CONSTITUTION.md"],
                    "local_tracker_files": [],
                    "repositories": [
                        {
                            "name": name,
                            "upstream_mainline": "main",
                            "ultron_mainline": "ultron/mainline",
                            "ultron_mainline_bootstrap": self.bootstrap_commits[name],
                            "instruction_files": ["AGENTS.md"],
                            "install_local_tracker": False,
                        }
                        for name in self.repositories
                    ],
                }
            )
        )

    def test_repository_root_discovery_supports_python_tests_placement(self):
        repository = self.workspace / "installed-repository"
        installed_test = repository / "python/tests/test_ultron_mainline.py"
        tool = repository / "tools/ultron_mainline.py"
        manifest = repository / "docs/agents/agentic-workflow-repos.json"
        installed_test.parent.mkdir(parents=True)
        tool.parent.mkdir(parents=True)
        manifest.parent.mkdir(parents=True)
        installed_test.write_text("# synchronized test placeholder\n")
        tool.write_text("# synchronized tool placeholder\n")
        manifest.write_text("{}\n")

        self.assertEqual(_repository_root(installed_test), repository)

        with mock.patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(
                AssertionError,
                "could not find repository root.*tools/ultron_mainline.py.*"
                "docs/agents/agentic-workflow-repos.json",
            ):
                _repository_root(installed_test)

    def invoke(self, mode, **overrides):
        output = io.StringIO()
        status = ultron_mainline._run_with_contracts(
            mode=mode,
            workspace_root=self.workspace,
            manifest_path=self.manifest,
            output=output,
            expected_contracts=self.contracts,
            git_runner=overrides.get("git_runner"),
        )
        return status, output.getvalue()

    def advance_upstream(self, name):
        repository = self.workspace / name
        (repository / "advance.txt").write_text("advanced\n")
        self.git("add", "advance.txt", cwd=repository)
        self.git("commit", "-m", "advance upstream", cwd=repository)

    def snapshot_refs_and_worktrees(self):
        return {
            name: (
                self.git(
                    "for-each-ref", "--format=%(refname):%(objectname)",
                    cwd=self.workspace / name,
                ).stdout,
                self.git(
                    "worktree", "list", "--porcelain", cwd=self.workspace / name
                ).stdout,
            )
            for name in self.repositories
        }

    def commit_in_integration_worktree(self, name="ultron"):
        worktree = self.workspace / name / ".worktrees/ultron-mainline"
        (worktree / "integration.txt").write_text("integrated\n")
        self.git("add", "integration.txt", cwd=worktree)
        self.git("commit", "-m", "human integration", cwd=worktree)

    def create_integration_worktree(self, name, path=None):
        repository = self.workspace / name
        target = path or repository / ".worktrees/ultron-mainline"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.git(
            "worktree",
            "add",
            "-b",
            "ultron/mainline",
            str(target),
            self.bootstrap_commits[name],
            cwd=repository,
        )
        return target

    def install_wrong_identity_at_exact_target(self, name):
        repository = self.workspace / name
        actual = repository / ".worktrees/actual-integration"
        self.create_integration_worktree(name, actual)
        target = repository / ".worktrees/ultron-mainline"
        target.mkdir()
        self.git("init", "-b", "ultron/mainline", cwd=target)
        self.git("config", "user.name", "Impostor", cwd=target)
        self.git("config", "user.email", "impostor@example.com", cwd=target)
        (target / "README.md").write_text("impostor\n")
        self.git("add", "README.md", cwd=target)
        self.git("commit", "-m", "impostor", cwd=target)
        return target

    def test_bootstrap_creates_local_branches_and_exact_repo_local_worktrees(self):
        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 0, output)
        for name in self.repositories:
            repository = self.workspace / name
            worktree = repository / ".worktrees/ultron-mainline"
            self.assertEqual(
                self.git("rev-parse", "ultron/mainline", cwd=repository).stdout.strip(),
                self.bootstrap_commits[name],
            )
            self.assertEqual(
                self.git("branch", "--show-current", cwd=worktree).stdout.strip(),
                "ultron/mainline",
            )
            self.assertEqual(
                self.git(
                    "for-each-ref",
                    "--format=%(upstream)",
                    "refs/heads/ultron/mainline",
                    cwd=repository,
                ).stdout.strip(),
                "",
            )

    def test_bootstrap_rejects_branch_registered_elsewhere_when_target_absent(self):
        repository = self.workspace / "ultron"
        elsewhere = repository / ".worktrees/elsewhere"
        elsewhere.parent.mkdir()
        self.git(
            "worktree",
            "add",
            "-b",
            "ultron/mainline",
            str(elsewhere),
            self.bootstrap_commits["ultron"],
            cwd=repository,
        )

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("registered at unexpected worktree", output)
        self.assertFalse((repository / ".worktrees/ultron-mainline").exists())

    def test_public_interfaces_match_the_approved_contract(self):
        contract = ultron_mainline.RepositoryContract(
            name="ultron",
            root=self.workspace / "ultron",
            upstream_mainline="main",
            ultron_mainline="ultron/mainline",
            bootstrap_commit=self.bootstrap_commits["ultron"],
        )
        self.assertEqual(
            contract.integration_worktree,
            self.workspace / "ultron/.worktrees/ultron-mainline",
        )
        loaded = ultron_mainline.load_contracts(self.manifest, self.workspace)
        self.assertEqual(loaded[0], contract)
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        self.assertEqual(ultron_mainline.check_contracts(loaded), ())

    def test_changed_source_aborts_all_writes_and_categorizes_every_repository(self):
        self.advance_upstream("peer")

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        lines = output.splitlines()
        repository_lines = [line for line in lines if line.startswith("ERROR repository=")]
        self.assertEqual(len(repository_lines), len(self.repositories), output)
        for name, line in zip(self.repositories, repository_lines, strict=True):
            self.assertIn(f"repository={name}", line)
            self.assertIn("source=main@", line)
            self.assertIn("destination=ultron/mainline", line)
            self.assertIn(
                f"path={self.workspace / name / '.worktrees/ultron-mainline'}", line
            )
            self.assertIn("status=blocked", line)
            self.assertNotEqual(
                self.git(
                    "show-ref", "--verify", "--quiet", "refs/heads/ultron/mainline",
                    cwd=self.workspace / name,
                    check=False,
                ).returncode,
                0,
            )
        self.assertIn("reason=source tip changed", repository_lines[1])
        self.assertEqual(lines[-1], "BOOTSTRAP ERROR: 1 failure(s); completed=[]")

    def test_second_bootstrap_is_idempotent_before_advancement(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        before = self.snapshot_refs_and_worktrees()

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 0, output)
        self.assertEqual(self.snapshot_refs_and_worktrees(), before)
        self.assertEqual(output.count("status=unchanged"), len(self.repositories))

    def test_check_accepts_a_human_advanced_descendant(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        self.commit_in_integration_worktree()

        status, output = self.invoke("check")

        self.assertEqual(status, 0, output)
        self.assertEqual(output.count("OK repository="), len(self.repositories))
        self.assertTrue(output.endswith("CHECK OK: 2 repositories\n"), output)

    def test_check_rejects_dirty_integration_worktree(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        dirty = self.workspace / "ultron" / ".worktrees/ultron-mainline/dirty.txt"
        dirty.write_text("dirty\n")

        status, output = self.invoke("check")

        self.assertEqual(status, 1, output)
        self.assertIn("repository=ultron", output)
        self.assertIn("reason=integration worktree is dirty", output)
        self.assertTrue(output.endswith("CHECK FAILED: 1 repository(s)\n"), output)

    def test_bootstrap_rejects_destination_namespace_conflict_before_writes(self):
        self.git("branch", "ultron", cwd=self.workspace / "peer")

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=destination namespace conflict", output)
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )
        self.assertEqual(output.count("ERROR repository="), len(self.repositories))

    def test_bootstrap_rejects_occupied_integration_target_before_writes(self):
        occupied = self.workspace / "peer/.worktrees/ultron-mainline"
        occupied.mkdir(parents=True)
        (occupied / "keep.txt").write_text("keep\n")

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration target is occupied", output)
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )
        self.assertEqual((occupied / "keep.txt").read_text(), "keep\n")

    def test_bootstrap_rejects_occupied_file_with_existing_branch_before_writes(self):
        peer = self.workspace / "peer"
        self.git(
            "branch", "ultron/mainline", self.bootstrap_commits["peer"], cwd=peer
        )
        occupied = peer / ".worktrees/ultron-mainline"
        occupied.parent.mkdir()
        occupied.write_text("keep\n")

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration target is occupied", output)
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )
        self.assertEqual(occupied.read_text(), "keep\n")

    def test_bootstrap_rejects_symlink_integration_target_before_writes(self):
        actual = self.workspace / "peer/actual-target"
        actual.mkdir()
        target = self.workspace / "peer/.worktrees/ultron-mainline"
        target.parent.mkdir()
        target.symlink_to(actual, target_is_directory=True)

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration target contains a symlink", output)
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )

    def test_bootstrap_rejects_symlink_integration_ancestor_before_writes(self):
        actual = self.workspace / "peer/actual-worktrees"
        actual.mkdir()
        ancestor = self.workspace / "peer/.worktrees"
        ancestor.symlink_to(actual, target_is_directory=True)

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration target contains a symlink", output)
        self.assertFalse((actual / "ultron-mainline").exists())
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )

    def test_check_rejects_configured_upstream(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        repository = self.workspace / "ultron"
        self.git(
            "config", "branch.ultron/mainline.remote", ".", cwd=repository
        )
        self.git(
            "config",
            "branch.ultron/mainline.merge",
            "refs/heads/main",
            cwd=repository,
        )

        status, output = self.invoke("check")

        self.assertEqual(status, 1, output)
        self.assertIn("reason=integration branch has configured upstream", output)

    def test_check_rejects_rewritten_or_unrelated_history(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        repository = self.workspace / "ultron"
        tree = self.git("rev-parse", "main^{tree}", cwd=repository).stdout.strip()
        unrelated = self.git(
            "commit-tree", tree, "-m", "unrelated root", cwd=repository
        ).stdout.strip()
        self.git(
            "update-ref", "refs/heads/ultron/mainline", unrelated, cwd=repository
        )

        status, output = self.invoke("check")

        self.assertEqual(status, 1, output)
        self.assertIn("reason=integration history is rewritten or unrelated", output)

    def test_check_rejects_duplicate_worktree_registration(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        repository = self.workspace / "ultron"
        duplicate = repository / ".worktrees/duplicate"
        self.git(
            "worktree",
            "add",
            "--detach",
            str(duplicate),
            self.bootstrap_commits["ultron"],
            cwd=repository,
        )
        gitdir = Path(
            (duplicate / ".git").read_text().removeprefix("gitdir: ").strip()
        )
        (gitdir / "HEAD").write_text("ref: refs/heads/ultron/mainline\n")

        status, output = self.invoke("check")

        self.assertEqual(status, 1, output)
        self.assertIn("reason=integration branch has 2 worktree registrations", output)

    def test_partial_execution_reports_completed_subset_and_resumes_idempotently(self):
        def fail_peer_creation(arguments, *, cwd):
            if arguments[:3] == ("worktree", "add", "-b") and cwd.name == "peer":
                return subprocess.CompletedProcess(
                    arguments, 9, stdout="", stderr="injected execution failure"
                )
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke("bootstrap", git_runner=fail_peer_creation)

        self.assertEqual(status, 2, output)
        self.assertIn("completed=['ultron']", output)
        self.assertTrue(
            (self.workspace / "ultron/.worktrees/ultron-mainline").is_dir()
        )
        self.assertFalse((self.workspace / "peer/.worktrees/ultron-mainline").exists())

        resume_status, resume_output = self.invoke("bootstrap")
        self.assertEqual(resume_status, 0, resume_output)
        self.assertIn("repository=ultron", resume_output)
        self.assertIn("status=unchanged", resume_output)
        self.assertTrue((self.workspace / "peer/.worktrees/ultron-mainline").is_dir())

    def test_execution_timeout_reports_failed_pending_and_completed_subset(self):
        later = self.workspace / "later"
        later.mkdir()
        self.git("init", "-b", "main", cwd=later)
        self.git("config", "user.name", "Test User", cwd=later)
        self.git("config", "user.email", "test@example.com", cwd=later)
        (later / "README.md").write_text("later\n")
        self.git("add", "README.md", cwd=later)
        self.git("commit", "-m", "bootstrap", cwd=later)
        later_commit = self.git("rev-parse", "HEAD", cwd=later).stdout.strip()
        manifest = json.loads(self.manifest.read_text())
        later_entry = dict(manifest["repositories"][1])
        later_entry.update(
            {"name": "later", "ultron_mainline_bootstrap": later_commit}
        )
        manifest["repositories"].append(later_entry)
        self.manifest.write_text(json.dumps(manifest))
        self.contracts = (
            *self.contracts,
            ultron_mainline.RepositoryContract(
                "later",
                later,
                "main",
                "ultron/mainline",
                later_commit,
            ),
        )

        def timeout_peer_creation(arguments, *, cwd):
            if arguments[:3] == ("worktree", "add", "-b") and cwd.name == "peer":
                raise subprocess.TimeoutExpired(cmd="git worktree add", timeout=15)
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke(
            "bootstrap", git_runner=timeout_peer_creation
        )

        self.assertEqual(status, 2, output)
        self.assertIn("ERROR repository=peer", output)
        self.assertIn("status=failed", output)
        self.assertIn("Git execution failed (timeout after 15 seconds)", output)
        self.assertIn("ERROR repository=later", output)
        self.assertIn("status=not-attempted reason=execution stopped", output)
        self.assertTrue(output.endswith("completed=['ultron']\n"), output)
        self.assertTrue(
            (self.workspace / "ultron/.worktrees/ultron-mainline").is_dir()
        )
        self.assertFalse((self.workspace / "peer/.worktrees/ultron-mainline").exists())
        self.assertFalse((self.workspace / "later/.worktrees/ultron-mainline").exists())

    def test_bootstrap_categorizes_git_inspection_nonzero_oserror_and_timeout(self):
        cases = [
            (
                "nonzero",
                lambda arguments: subprocess.CompletedProcess(
                    arguments, 7, stdout="", stderr="injected nonzero"
                ),
                "Git inspection failed (exit 7)",
            ),
            ("oserror", lambda arguments: OSError("injected os error"), "Git inspection failed (OSError"),
            (
                "timeout",
                lambda arguments: subprocess.TimeoutExpired(arguments, 15),
                "Git inspection failed (timeout",
            ),
        ]
        for name, injected, expected in cases:
            with self.subTest(name=name):
                def fail_inspection(arguments, *, cwd):
                    if arguments == ("rev-parse", "--git-common-dir") and cwd.name == "peer":
                        outcome = injected(arguments)
                        if isinstance(outcome, BaseException):
                            raise outcome
                        return outcome
                    return subprocess.run(
                        ("git", *arguments),
                        cwd=cwd,
                        check=False,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=15,
                    )

                status, output = self.invoke("bootstrap", git_runner=fail_inspection)
                self.assertEqual(status, 2, output)
                self.assertIn(expected, output)
                self.assertEqual(
                    output.count("ERROR repository="), len(self.repositories), output
                )
                self.assertTrue(
                    output.endswith("BOOTSTRAP ERROR: 1 failure(s); completed=[]\n"),
                    output,
                )

    def test_bootstrap_rejects_existing_branch_with_configured_upstream(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        repository = self.workspace / "peer"
        self.git(
            "config", "branch.ultron/mainline.remote", ".", cwd=repository
        )
        self.git(
            "config",
            "branch.ultron/mainline.merge",
            "refs/heads/main",
            cwd=repository,
        )

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration branch has configured upstream", output)
        self.assertEqual(output.count("ERROR repository="), len(self.repositories))

    def test_bootstrap_resumes_when_exact_ref_exists_without_worktree(self):
        peer = self.workspace / "peer"
        self.git(
            "branch",
            "ultron/mainline",
            self.bootstrap_commits["peer"],
            cwd=peer,
        )

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 0, output)
        self.assertIn("repository=peer", output)
        self.assertIn("status=created", output)
        self.assertEqual(
            self.git(
                "branch",
                "--show-current",
                cwd=peer / ".worktrees/ultron-mainline",
            ).stdout.strip(),
            "ultron/mainline",
        )

    def test_bootstrap_rejects_existing_unrelated_branch_before_writes(self):
        peer = self.workspace / "peer"
        tree = self.git("rev-parse", "main^{tree}", cwd=peer).stdout.strip()
        unrelated = self.git(
            "commit-tree", tree, "-m", "unrelated root", cwd=peer
        ).stdout.strip()
        self.git("branch", "ultron/mainline", unrelated, cwd=peer)

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration history is rewritten or unrelated", output)
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )

    def test_bootstrap_rejects_advanced_descendant_and_directs_to_check(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        self.commit_in_integration_worktree("peer")

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration branch advanced; use --check", output)

    def test_bootstrap_rejects_target_registered_to_the_wrong_branch(self):
        peer = self.workspace / "peer"
        self.git(
            "branch", "ultron/mainline", self.bootstrap_commits["peer"], cwd=peer
        )
        target = peer / ".worktrees/ultron-mainline"
        target.parent.mkdir()
        self.git("worktree", "add", "-b", "other", str(target), cwd=peer)

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration target is registered to the wrong branch", output)
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )

    def test_bootstrap_rejects_duplicate_worktree_registration(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        repository = self.workspace / "peer"
        duplicate = repository / ".worktrees/duplicate"
        self.git(
            "worktree",
            "add",
            "--detach",
            str(duplicate),
            self.bootstrap_commits["peer"],
            cwd=repository,
        )
        gitdir = Path(
            (duplicate / ".git").read_text().removeprefix("gitdir: ").strip()
        )
        (gitdir / "HEAD").write_text("ref: refs/heads/ultron/mainline\n")

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration branch has 2 worktree registrations", output)

    def test_bootstrap_rejects_dirty_existing_integration_worktree(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)
        dirty = self.workspace / "peer/.worktrees/ultron-mainline/dirty.txt"
        dirty.write_text("dirty\n")

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration worktree is dirty", output)

    def test_check_rejects_symlink_integration_target(self):
        self.create_integration_worktree("ultron")
        peer = self.workspace / "peer"
        actual = peer / "actual-target"
        self.create_integration_worktree("peer", actual)
        target = peer / ".worktrees/ultron-mainline"
        target.parent.mkdir()
        target.symlink_to(actual, target_is_directory=True)

        status, output = self.invoke("check")

        self.assertEqual(status, 1, output)
        self.assertIn("reason=integration target contains a symlink", output)

    def test_check_rejects_symlink_integration_ancestor(self):
        self.create_integration_worktree("ultron")
        peer = self.workspace / "peer"
        actual_parent = peer / "actual-worktrees"
        actual = actual_parent / "ultron-mainline"
        actual_parent.mkdir()
        self.create_integration_worktree("peer", actual)
        (peer / ".worktrees").symlink_to(actual_parent, target_is_directory=True)

        status, output = self.invoke("check")

        self.assertEqual(status, 1, output)
        self.assertIn("reason=integration target contains a symlink", output)

    def test_check_categorizes_missing_integration_worktrees(self):
        status, output = self.invoke("check")

        self.assertEqual(status, 1, output)
        self.assertEqual(
            output.count("reason=integration worktree is missing"),
            len(self.repositories),
        )
        self.assertTrue(output.endswith("CHECK FAILED: 2 repository(s)\n"), output)

    def test_cli_rejects_repository_map_surface(self):
        repository_map = self.workspace / "repository-map.json"
        repository_map.write_text(
            json.dumps(
                {name: str(self.workspace / name) for name in self.repositories}
            )
        )

        with self.assertRaises(SystemExit) as raised:
            ultron_mainline.main(
                [
                    "--bootstrap",
                    "--workspace-root",
                    str(self.workspace),
                    "--manifest",
                    str(self.manifest),
                    "--repository-map",
                    str(repository_map),
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        for name in self.repositories:
            self.assertFalse(
                (self.workspace / name / ".worktrees/ultron-mainline").exists()
            )

    def test_linked_worktree_script_discovers_canonical_workspace_parent(self):
        source = self.workspace / "ultron"
        linked = source / ".worktrees/ticket"
        linked.parent.mkdir()
        self.git("worktree", "add", "-b", "ticket", str(linked), cwd=source)
        script = linked / "tools/ultron_mainline.py"
        script.parent.mkdir()
        script.write_text("# synchronized tool placeholder\n")

        workspace = ultron_mainline.default_workspace_root(
            script, git_runner=ultron_mainline.run_git
        )

        self.assertEqual(workspace, self.workspace)

    def test_default_workspace_rejects_real_symlink_git_common_directory(self):
        repository = self.workspace / "monarch"
        repository.mkdir()
        self.git("init", "-b", "main", cwd=repository)
        self.git("config", "user.name", "Test User", cwd=repository)
        self.git("config", "user.email", "test@example.com", cwd=repository)
        (repository / "README.md").write_text("monarch\n")
        self.git("add", "README.md", cwd=repository)
        self.git("commit", "-m", "bootstrap", cwd=repository)
        script = repository / "tools/ultron_mainline.py"
        script.parent.mkdir()
        script.write_text("# synchronized tool placeholder\n")
        actual_git_directory = self.workspace / "monarch-actual-git"
        (repository / ".git").rename(actual_git_directory)
        (repository / ".git").symlink_to(
            actual_git_directory, target_is_directory=True
        )

        functioning = self.git(
            "rev-parse", "--is-inside-work-tree", cwd=repository
        )
        self.assertEqual(functioning.stdout.strip(), "true")
        with self.assertRaisesRegex(
            ultron_mainline.MainlineError,
            "Git common directory is a symlink",
        ):
            ultron_mainline.default_workspace_root(
                script, git_runner=ultron_mainline.run_git
            )

    def test_linked_worktree_cli_without_workspace_argument_uses_common_dir(self):
        linked = self.workspace / "linked-script"
        self.git(
            "worktree",
            "add",
            "-b",
            "linked-script",
            str(linked),
            cwd=self.workspace / "ultron",
        )
        script = linked / "tools/ultron_mainline.py"
        script.parent.mkdir()
        repository_root = _repository_root()
        source_script = repository_root / "tools/ultron_mainline.py"
        script.write_text(source_script.read_text())
        production_manifest = repository_root / (
            "docs/agents/agentic-workflow-repos.json"
        )
        before = self.git(
            "for-each-ref",
            "--format=%(refname):%(objectname)",
            cwd=self.workspace / "ultron",
        ).stdout

        result = subprocess.run(
            (
                "python",
                str(script),
                "--check",
                "--manifest",
                str(production_manifest),
            ),
            cwd=linked,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )

        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(
            f"path={self.workspace / 'torchtitan/.worktrees/ultron-mainline'}",
            result.stdout,
        )
        self.assertNotIn(f"path={linked / 'torchtitan'}", result.stdout)
        self.assertEqual(
            self.git(
                "for-each-ref",
                "--format=%(refname):%(objectname)",
                cwd=self.workspace / "ultron",
            ).stdout,
            before,
        )

    def test_sibling_linked_copy_parses_explicit_and_discovers_default_workspace(self):
        sibling = self.workspace / "torchtitan"
        sibling.mkdir()
        self.git("init", "-b", "main", cwd=sibling)
        self.git("config", "user.name", "Test User", cwd=sibling)
        self.git("config", "user.email", "test@example.com", cwd=sibling)
        (sibling / "README.md").write_text("torchtitan\n")
        self.git("add", "README.md", cwd=sibling)
        self.git("commit", "-m", "bootstrap", cwd=sibling)
        linked = sibling / ".worktrees/synchronized-copy"
        linked.parent.mkdir()
        self.git(
            "worktree",
            "add",
            "-b",
            "synchronized-copy",
            str(linked),
            cwd=sibling,
        )
        script = linked / "tools/ultron_mainline.py"
        script.parent.mkdir()
        repository_root = _repository_root()
        source_script = repository_root / "tools/ultron_mainline.py"
        script.write_text(source_script.read_text())
        production_manifest = repository_root / (
            "docs/agents/agentic-workflow-repos.json"
        )

        spec = importlib.util.spec_from_file_location(
            "synchronized_sibling_ultron_mainline", script
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        synchronized_copy = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = synchronized_copy
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(synchronized_copy)
        real_runner = synchronized_copy.run_git

        def reject_eager_default(arguments, *, cwd):
            if arguments == ("rev-parse", "--git-common-dir") and cwd == linked:
                raise OSError("explicit workspace must skip default discovery")
            return real_runner(arguments, cwd=cwd)

        synchronized_copy.run_git = reject_eager_default
        explicit_output = io.StringIO()
        with redirect_stdout(explicit_output):
            explicit_status = synchronized_copy.main(
                [
                    "--check",
                    "--workspace-root",
                    str(self.workspace),
                    "--manifest",
                    str(production_manifest),
                ]
            )
        discovered = subprocess.run(
            (
                "python",
                str(script),
                "--check",
                "--manifest",
                str(production_manifest),
            ),
            cwd=linked,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )

        self.assertEqual(explicit_status, 2, explicit_output.getvalue())
        self.assertIn("ERROR repository=", explicit_output.getvalue())
        self.assertIn("CHECK ERROR:", explicit_output.getvalue())
        self.assertEqual(discovered.returncode, 2, discovered.stdout + discovered.stderr)
        self.assertNotIn("cannot discover workspace", discovered.stderr)
        self.assertNotIn("Traceback", discovered.stderr)
        self.assertIn("ERROR repository=", discovered.stdout)
        self.assertIn("CHECK ERROR:", discovered.stdout)
        for rendered in (explicit_output.getvalue(), discovered.stdout):
            self.assertIn(
                f"path={self.workspace / 'monarch/.worktrees/ultron-mainline'}",
                rendered,
            )

    def test_all_plan_lines_and_preflight_count_precede_first_write(self):
        output = io.StringIO()
        before_writes = []

        def observe_first_write(arguments, *, cwd):
            if arguments[:2] == ("worktree", "add"):
                before_writes.append(output.getvalue().splitlines())
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status = ultron_mainline._run_with_contracts(
            mode="bootstrap",
            workspace_root=self.workspace,
            manifest_path=self.manifest,
            output=output,
            expected_contracts=self.contracts,
            git_runner=observe_first_write,
        )

        self.assertEqual(status, 0, output.getvalue())
        self.assertTrue(before_writes)
        expected_plans = [
            line
            for line in before_writes[0]
            if line.startswith("PLAN repository=")
        ]
        self.assertEqual(len(expected_plans), len(self.repositories))
        self.assertEqual(
            before_writes[0][-1],
            f"PREFLIGHT OK: {len(self.repositories)} repositories",
        )
        for line in expected_plans:
            self.assertIn("source=main@", line)
            self.assertIn("destination=ultron/mainline", line)
            self.assertIn("path=", line)
            self.assertIn("status=ready", line)

    def test_later_preflight_oserror_is_categorized_before_any_write(self):
        peer = self.workspace / "peer"
        self.git(
            "branch", "ultron/mainline", self.bootstrap_commits["peer"], cwd=peer
        )

        def fail_upstream_inspection(arguments, *, cwd):
            if arguments[0] == "for-each-ref" and cwd.name == "peer":
                raise OSError("injected later preflight failure")
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke(
            "bootstrap", git_runner=fail_upstream_inspection
        )

        self.assertEqual(status, 2, output)
        self.assertIn("Git inspection failed (OSError", output)
        self.assertEqual(output.count("ERROR repository="), len(self.repositories))
        self.assertTrue(output.endswith("completed=[]\n"), output)
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )

    def test_later_preflight_nonzero_is_git_error_not_semantic_drift(self):
        peer = self.workspace / "peer"
        self.git(
            "branch", "ultron/mainline", self.bootstrap_commits["peer"], cwd=peer
        )

        def fail_upstream_inspection(arguments, *, cwd):
            if arguments[0] == "for-each-ref" and cwd.name == "peer":
                return subprocess.CompletedProcess(
                    arguments, 7, stdout="", stderr="injected later nonzero"
                )
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke(
            "bootstrap", git_runner=fail_upstream_inspection
        )

        self.assertEqual(status, 2, output)
        self.assertIn("Git inspection failed (exit 7): injected later nonzero", output)
        self.assertNotIn("reason=integration branch has configured upstream", output)
        self.assertEqual(output.count("ERROR repository="), len(self.repositories))
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )

    def test_check_timeout_is_one_error_record_and_continues(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)

        def timeout_peer_status(arguments, *, cwd):
            if arguments[:2] == ("status", "--porcelain") and cwd.parts[-3:] == (
                "peer",
                ".worktrees",
                "ultron-mainline",
            ):
                raise subprocess.TimeoutExpired(cmd="git status", timeout=15)
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke("check", git_runner=timeout_peer_status)

        self.assertEqual(status, 2, output)
        records = [
            line
            for line in output.splitlines()
            if line.startswith(("OK repository=", "ERROR repository="))
        ]
        self.assertEqual(len(records), len(self.repositories), output)
        self.assertIn("OK repository=ultron", records[0])
        self.assertIn("ERROR repository=peer", records[1])
        self.assertIn("Git inspection failed (timeout after 15 seconds)", records[1])
        self.assertTrue(output.endswith("CHECK ERROR: 1 failure(s)\n"), output)

    def test_check_nonzero_is_error_not_drift(self):
        self.assertEqual(self.invoke("bootstrap")[0], 0)

        def fail_peer_status(arguments, *, cwd):
            if arguments[:2] == ("status", "--porcelain") and cwd.parts[-3:] == (
                "peer",
                ".worktrees",
                "ultron-mainline",
            ):
                return subprocess.CompletedProcess(
                    arguments, 7, stdout="", stderr="injected check nonzero"
                )
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke("check", git_runner=fail_peer_status)

        self.assertEqual(status, 2, output)
        self.assertIn("ERROR repository=peer", output)
        self.assertIn("Git inspection failed (exit 7): injected check nonzero", output)
        self.assertNotIn("DRIFT repository=peer", output)
        self.assertTrue(output.endswith("CHECK ERROR: 1 failure(s)\n"), output)

    def test_check_rejects_wrong_identity_at_exact_integration_target(self):
        self.create_integration_worktree("ultron")
        self.install_wrong_identity_at_exact_target("peer")

        status, output = self.invoke("check")

        self.assertEqual(status, 1, output)
        self.assertIn("reason=integration target has wrong Git identity", output)

    def test_bootstrap_rejects_wrong_identity_at_exact_target_before_writes(self):
        self.install_wrong_identity_at_exact_target("peer")

        status, output = self.invoke("bootstrap")

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration target has wrong Git identity", output)
        self.assertFalse(
            (self.workspace / "ultron/.worktrees/ultron-mainline").exists()
        )

    def test_destination_ref_inspection_error_never_becomes_safe_absence(self):
        def fail_destination_lookup(arguments, *, cwd):
            if (
                arguments
                == (
                    "show-ref",
                    "--verify",
                    "--quiet",
                    "refs/heads/ultron/mainline",
                )
                and cwd.name == "peer"
            ):
                return subprocess.CompletedProcess(
                    arguments, 7, stdout="", stderr="permission denied"
                )
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke(
            "bootstrap", git_runner=fail_destination_lookup
        )

        self.assertEqual(status, 2, output)
        self.assertIn("reason=Git inspection failed (exit 7)", output)
        self.assertEqual(output.count("ERROR repository="), len(self.repositories))
        for name in self.repositories:
            self.assertFalse(
                (self.workspace / name / ".worktrees/ultron-mainline").exists()
            )

    def test_namespace_inspection_error_never_becomes_safe_absence(self):
        def fail_namespace_lookup(arguments, *, cwd):
            if (
                arguments
                == ("show-ref", "--verify", "--quiet", "refs/heads/ultron")
                and cwd.name == "peer"
            ):
                return subprocess.CompletedProcess(
                    arguments, 9, stdout="", stderr="namespace unreadable"
                )
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke(
            "bootstrap", git_runner=fail_namespace_lookup
        )

        self.assertEqual(status, 2, output)
        self.assertIn("reason=Git inspection failed (exit 9)", output)
        self.assertEqual(output.count("ERROR repository="), len(self.repositories))
        for name in self.repositories:
            self.assertFalse(
                (self.workspace / name / ".worktrees/ultron-mainline").exists()
            )

    def test_preflight_never_invokes_git_through_rejected_symlink_ancestor(self):
        peer = self.workspace / "peer"
        self.git(
            "branch", "ultron/mainline", self.bootstrap_commits["peer"], cwd=peer
        )
        actual = peer / "actual-worktrees"
        actual.mkdir()
        (actual / "ultron-mainline").mkdir()
        (peer / ".worktrees").symlink_to(actual, target_is_directory=True)
        rejected_target = peer / ".worktrees/ultron-mainline"
        forbidden_calls = []

        def record_target_calls(arguments, *, cwd):
            if cwd == rejected_target:
                forbidden_calls.append(arguments)
                return subprocess.CompletedProcess(
                    arguments, 99, stdout="", stderr="forbidden symlink traversal"
                )
            return subprocess.run(
                ("git", *arguments),
                cwd=cwd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )

        status, output = self.invoke("bootstrap", git_runner=record_target_calls)

        self.assertEqual(status, 2, output)
        self.assertIn("reason=integration target contains a symlink", output)
        self.assertEqual(forbidden_calls, [])
        self.assertTrue((actual / "ultron-mainline").is_dir())

    def test_check_invalid_manifest_uses_complete_immutable_fallback(self):
        invalid_manifest = self.workspace / "invalid-manifest.json"
        invalid_manifest.write_text("{")
        output = io.StringIO()

        status = ultron_mainline._run_with_contracts(
            mode="check",
            workspace_root=self.workspace,
            manifest_path=invalid_manifest,
            output=output,
            expected_contracts=ultron_mainline.CANONICAL_REPOSITORIES,
            git_runner=lambda arguments, *, cwd: self.fail(
                f"Git must not run after manifest failure: {arguments} in {cwd}"
            ),
        )

        rendered = output.getvalue()
        expected_names = (
            "ultron",
            "torchtitan",
            "monarch",
            "megatronlm",
            "nccl",
            "ferric_continuum",
            "tnsr",
        )
        self.assertEqual(status, 2, rendered)
        records = [
            line
            for line in rendered.splitlines()
            if line.startswith("ERROR repository=")
        ]
        self.assertEqual(len(records), len(expected_names), rendered)
        for name, record in zip(expected_names, records, strict=True):
            self.assertIn(f"repository={name}", record)
            self.assertRegex(record, r"source=(main|master)@[0-9a-f]{40}")
            self.assertIn("destination=ultron/mainline", record)
            self.assertIn(
                f"path={self.workspace / name / '.worktrees/ultron-mainline'}",
                record,
            )
            self.assertIn("status=error reason=manifest setup failed", record)
        self.assertTrue(
            rendered.endswith("CHECK ERROR: 1 setup failure(s)\n"), rendered
        )

    def test_public_contract_rejects_every_inexact_manifest_identity(self):
        production_manifest = _repository_root() / (
            "docs/agents/agentic-workflow-repos.json"
        )
        original = json.loads(production_manifest.read_text())
        cases = {}
        wrong_version = json.loads(json.dumps(original))
        wrong_version["version"] = "2026.08.15.1"
        cases["version"] = wrong_version
        reordered = json.loads(json.dumps(original))
        reordered["repositories"][0], reordered["repositories"][1] = (
            reordered["repositories"][1],
            reordered["repositories"][0],
        )
        cases["order"] = reordered
        wrong_upstream = json.loads(json.dumps(original))
        wrong_upstream["repositories"][1]["upstream_mainline"] = "master"
        cases["upstream"] = wrong_upstream
        wrong_destination = json.loads(json.dumps(original))
        wrong_destination["repositories"][1]["ultron_mainline"] = "other/mainline"
        cases["destination"] = wrong_destination
        wrong_sha_type = json.loads(json.dumps(original))
        wrong_sha_type["repositories"][1]["ultron_mainline_bootstrap"] = 7
        cases["sha type"] = wrong_sha_type
        legacy = json.loads(json.dumps(original))
        legacy["repositories"][1]["mainline"] = "main"
        cases["legacy identity"] = legacy
        path_name = json.loads(json.dumps(original))
        path_name["repositories"][1]["name"] = "../outside"
        cases["path name"] = path_name
        extra = json.loads(json.dumps(original))
        extra["repositories"].append(dict(extra["repositories"][-1]))
        cases["extra repository"] = extra

        for label, manifest in cases.items():
            with self.subTest(label=label):
                candidate = self.workspace / f"{label.replace(' ', '-')}.json"
                candidate.write_text(json.dumps(manifest))
                output = io.StringIO()
                status = ultron_mainline._run_with_contracts(
                    mode="check",
                    workspace_root=self.workspace,
                    manifest_path=candidate,
                    output=output,
                    expected_contracts=ultron_mainline.CANONICAL_REPOSITORIES,
                    git_runner=lambda arguments, *, cwd: self.fail(
                        f"Git must not run for invalid manifest: {arguments} in {cwd}"
                    ),
                )
                rendered = output.getvalue()
                self.assertEqual(status, 2, rendered)
                self.assertEqual(rendered.count("ERROR repository="), 7, rendered)
                self.assertTrue(
                    rendered.endswith("CHECK ERROR: 1 setup failure(s)\n"),
                    rendered,
                )

    def test_production_manifest_constructs_only_canonical_paths(self):
        repository_root = _repository_root()
        actions = ultron_mainline._manifest_actions(
            self.workspace,
            repository_root / "docs/agents/agentic-workflow-repos.json",
        )

        self.assertEqual(
            [action.name for action in actions],
            [contract.name for contract in ultron_mainline.CANONICAL_REPOSITORIES],
        )
        for action in actions:
            self.assertEqual(action.root, self.workspace / action.name)
            self.assertEqual(action.destination_ref, "ultron/mainline")
            self.assertEqual(
                action.destination_path,
                self.workspace / action.name / ".worktrees/ultron-mainline",
            )


if __name__ == "__main__":
    unittest.main()
