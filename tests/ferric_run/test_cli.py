import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import ferric_run


def make_rootfs_mountpoint(rootfs: Path, sandbox: str) -> None:
    (rootfs / sandbox.removeprefix("/")).mkdir(parents=True)


class CliTests(unittest.TestCase):
    def test_dry_run_writes_run_record_without_executing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            rootfs = root / "rootfs"
            bwrap = root / "bwrap"
            ferric = root / "ferric"
            rootfs.mkdir()
            make_rootfs_mountpoint(rootfs, "/workspace/ferric_continuum")
            ferric.mkdir()
            bwrap.write_text("#!/bin/sh\nexit 0\n")
            bwrap.chmod(0o755)
            profile.write_text(json.dumps({
                "workspace": {
                    "primary_repo": "ferric_continuum",
                    "repos": {
                        "ferric_continuum": {
                            "host": "repo://self",
                            "sandbox": "/workspace/ferric_continuum",
                            "mode": "rw",
                            "required": True,
                        },
                        "missing_peer": {
                            "host": str(root / "missing"),
                            "sandbox": "/workspace/missing_peer",
                            "mode": "ro",
                            "required": False,
                        },
                    },
                },
            }))

            with mock.patch("tools.ferric_run.config.find_repo_root", return_value=ferric):
                exit_code = ferric_run.main([
                    "--profile", str(profile),
                    "--rootfs", str(rootfs),
                    "--bwrap", str(bwrap),
                    "--dry-run",
                    "--",
                    "true",
                ])

            self.assertEqual(exit_code, 0)
            runs = list((ferric / ".ferric" / "runs").iterdir())
            self.assertEqual(len(runs), 1)
            run_dir = runs[0]
            self.assertTrue((run_dir / "command.json").exists())
            self.assertTrue((run_dir / "preflight.json").exists())
            self.assertTrue((run_dir / "bwrap-plan.json").exists())
            result = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(result["executed"], False)
            self.assertEqual(result["ok"], True)

            command = json.loads((run_dir / "command.json").read_text())
            self.assertEqual(command["raw_argv"], [
                "--profile", str(profile),
                "--rootfs", str(rootfs),
                "--bwrap", str(bwrap),
                "--dry-run",
                "--",
                "true",
            ])
            self.assertEqual(command["command_argv"], ["true"])
            self.assertEqual(command["profile_path"], str(profile.resolve()))
            self.assertEqual(command["rootfs"], str(rootfs.resolve()))
            self.assertEqual(command["bwrap"], str(bwrap.resolve()))
            self.assertEqual(command["cuda"], {"requested": False})
            self.assertEqual(command["dry_run"], True)
            self.assertEqual(command["host_cwd"], str(Path.cwd()))
            self.assertEqual(command["run_id"], result["run_id"])

    def test_cuda_dry_run_records_intent_in_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            rootfs = root / "rootfs"
            bwrap = root / "bwrap"
            ferric = root / "ferric"
            cuda_device = root / "dev" / "nvidia0"
            cuda_library_root = root / "nvidia-driver"
            rootfs.mkdir()
            make_rootfs_mountpoint(rootfs, "/workspace/ferric_continuum")
            ferric.mkdir()
            cuda_device.parent.mkdir()
            cuda_device.touch()
            cuda_library_root.mkdir()
            bwrap.write_text("#!/bin/sh\nexit 0\n")
            bwrap.chmod(0o755)
            profile.write_text(json.dumps({
                "workspace": {
                    "primary_repo": "ferric_continuum",
                    "repos": {
                        "ferric_continuum": {
                            "host": "repo://self",
                            "sandbox": "/workspace/ferric_continuum",
                            "mode": "rw",
                            "required": True,
                        },
                    },
                },
            }))

            with (
                mock.patch("tools.ferric_run.config.find_repo_root", return_value=ferric),
                mock.patch("tools.ferric_run.preflight._discover_cuda_devices", return_value=[cuda_device]),
                mock.patch("tools.ferric_run.preflight._candidate_cuda_library_roots", return_value=[cuda_library_root]),
            ):
                exit_code = ferric_run.main([
                    "--profile", str(profile),
                    "--rootfs", str(rootfs),
                    "--bwrap", str(bwrap),
                    "--cuda",
                    "--dry-run",
                    "--",
                    "true",
                ])

            self.assertEqual(exit_code, 0)
            [run_dir] = list((ferric / ".ferric" / "runs").iterdir())
            command = json.loads((run_dir / "command.json").read_text())
            preflight = json.loads((run_dir / "preflight.json").read_text())
            bwrap_plan = json.loads((run_dir / "bwrap-plan.json").read_text())
            cuda_checks = [check for check in preflight["checks"] if check["name"] == "cuda"]
            self.assertEqual(command["cuda"], {"requested": True})
            self.assertEqual(len(cuda_checks), 1)
            self.assertTrue(cuda_checks[0]["ok"])
            self.assertEqual(bwrap_plan["cuda"], {"requested": True})

    def test_non_dry_run_executes_bwrap_and_records_subprocess_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            rootfs = root / "rootfs"
            bwrap = root / "bwrap"
            ferric = root / "ferric"
            rootfs.mkdir()
            make_rootfs_mountpoint(rootfs, "/workspace/ferric_continuum")
            ferric.mkdir()
            bwrap.write_text("#!/bin/sh\necho \"fake stdout\"\necho \"fake stderr\" >&2\nexit 7\n")
            bwrap.chmod(0o755)
            profile.write_text(json.dumps({
                "workspace": {
                    "primary_repo": "ferric_continuum",
                    "repos": {
                        "ferric_continuum": {
                            "host": "repo://self",
                            "sandbox": "/workspace/ferric_continuum",
                            "mode": "rw",
                            "required": True,
                        },
                    },
                },
            }))

            with mock.patch("tools.ferric_run.config.find_repo_root", return_value=ferric):
                exit_code = ferric_run.main([
                    "--profile", str(profile),
                    "--rootfs", str(rootfs),
                    "--bwrap", str(bwrap),
                    "--",
                    "ignored",
                ])

            self.assertEqual(exit_code, 7)
            [run_dir] = list((ferric / ".ferric" / "runs").iterdir())
            self.assertEqual((run_dir / "stdout.log").read_text(), "fake stdout\n")
            self.assertEqual((run_dir / "stderr.log").read_text(), "fake stderr\n")
            result = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(result["executed"], True)
            self.assertEqual(result["ok"], False)
            self.assertEqual(result["exit_code"], 7)
            self.assertIn("duration_ms", result)

    def test_profile_defaults_to_repo_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            profile_dir = repo / "tools" / "ferric_run" / "profiles"
            profile = profile_dir / "gpu-kernel-study.json"
            rootfs = root / "rootfs"
            bwrap = root / "bwrap"
            profile_dir.mkdir(parents=True)
            rootfs.mkdir()
            make_rootfs_mountpoint(rootfs, "/workspace/ferric_continuum")
            bwrap.write_text("#!/bin/sh\nexit 0\n")
            bwrap.chmod(0o755)
            profile.write_text(json.dumps({
                "workspace": {
                    "primary_repo": "ferric_continuum",
                    "repos": {
                        "ferric_continuum": {
                            "host": "repo://self",
                            "sandbox": "/workspace/ferric_continuum",
                            "mode": "rw",
                            "required": True,
                        },
                    },
                },
            }))

            with mock.patch("tools.ferric_run.config.find_repo_root", return_value=repo):
                exit_code = ferric_run.main([
                    "--rootfs", str(rootfs),
                    "--bwrap", str(bwrap),
                    "--dry-run",
                    "--",
                    "true",
                ])

            self.assertEqual(exit_code, 0)
            [run_dir] = list((repo / ".ferric" / "runs").iterdir())
            command = json.loads((run_dir / "command.json").read_text())
            self.assertEqual(command["profile_path"], str(profile.resolve()))

    def test_executable_wrapper_delegates_to_package_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            profile_dir = repo / "tools" / "ferric_run" / "profiles"
            profile = profile_dir / "gpu-kernel-study.json"
            rootfs = root / "rootfs"
            bwrap = root / "bwrap"
            profile_dir.mkdir(parents=True)
            rootfs.mkdir()
            make_rootfs_mountpoint(rootfs, "/workspace/ferric_continuum")
            bwrap.write_text("#!/bin/sh\nexit 0\n")
            bwrap.chmod(0o755)
            profile.write_text(json.dumps({
                "workspace": {
                    "primary_repo": "ferric_continuum",
                    "repos": {
                        "ferric_continuum": {
                            "host": "repo://self",
                            "sandbox": "/workspace/ferric_continuum",
                            "mode": "rw",
                            "required": True,
                        },
                    },
                },
            }))

            result = subprocess.run(
                [
                    "python3",
                    str(Path.cwd() / "tools" / "ferric_run.py"),
                    "--profile", str(profile),
                    "--rootfs", str(rootfs),
                    "--bwrap", str(bwrap),
                    "--dry-run",
                    "--",
                    "true",
                ],
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            [run_dir] = list((repo / ".ferric" / "runs").iterdir())
            self.assertTrue((run_dir / "result.json").exists())

    def test_dry_run_writes_run_record_when_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            rootfs = root / "missing-rootfs"
            bwrap = root / "bwrap"
            ferric = root / "ferric"
            ferric.mkdir()
            bwrap.write_text("#!/bin/sh\nexit 0\n")
            bwrap.chmod(0o755)
            profile.write_text(json.dumps({
                "workspace": {
                    "primary_repo": "ferric_continuum",
                    "repos": {
                        "ferric_continuum": {
                            "host": "repo://self",
                            "sandbox": "/workspace/ferric_continuum",
                            "mode": "rw",
                            "required": True,
                        },
                    },
                },
            }))

            with mock.patch("tools.ferric_run.config.find_repo_root", return_value=ferric):
                exit_code = ferric_run.main([
                    "--profile", str(profile),
                    "--rootfs", str(rootfs),
                    "--bwrap", str(bwrap),
                    "--dry-run",
                    "--",
                    "true",
                ])

            self.assertEqual(exit_code, 1)
            runs = list((ferric / ".ferric" / "runs").iterdir())
            self.assertEqual(len(runs), 1)
            run_dir = runs[0]
            self.assertTrue((run_dir / "command.json").exists())
            self.assertTrue((run_dir / "preflight.json").exists())
            self.assertTrue((run_dir / "bwrap-plan.json").exists())
            self.assertTrue((run_dir / "stdout.log").exists())
            self.assertTrue((run_dir / "stderr.log").exists())
            result = json.loads((run_dir / "result.json").read_text())
            self.assertEqual(result["executed"], False)
            self.assertEqual(result["ok"], False)
            self.assertEqual(result["exit_code"], 1)

    def test_dry_run_records_missing_bwrap_without_dot_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            rootfs = root / "rootfs"
            ferric = root / "ferric"
            rootfs.mkdir()
            make_rootfs_mountpoint(rootfs, "/workspace/ferric_continuum")
            ferric.mkdir()
            profile.write_text(json.dumps({
                "workspace": {
                    "primary_repo": "ferric_continuum",
                    "repos": {
                        "ferric_continuum": {
                            "host": "repo://self",
                            "sandbox": "/workspace/ferric_continuum",
                            "mode": "rw",
                            "required": True,
                        },
                    },
                },
            }))

            with (
                mock.patch("tools.ferric_run.config.find_repo_root", return_value=ferric),
                mock.patch("tools.ferric_run.preflight.shutil.which", return_value=None),
            ):
                exit_code = ferric_run.main([
                    "--profile", str(profile),
                    "--rootfs", str(rootfs),
                    "--dry-run",
                    "--",
                    "true",
                ])

            self.assertEqual(exit_code, 1)
            [run_dir] = list((ferric / ".ferric" / "runs").iterdir())
            command = json.loads((run_dir / "command.json").read_text())
            preflight = json.loads((run_dir / "preflight.json").read_text())
            bwrap_plan = json.loads((run_dir / "bwrap-plan.json").read_text())
            bwrap_checks = [check for check in preflight["checks"] if check["name"] == "bwrap"]
            self.assertEqual(len(bwrap_checks), 1)
            self.assertFalse(bwrap_checks[0]["ok"])
            self.assertIsNone(command["bwrap"])
            self.assertNotEqual(bwrap_plan["bwrap"], ".")
            self.assertNotEqual(bwrap_plan["bwrap_argv"][0], ".")


if __name__ == "__main__":
    unittest.main()
