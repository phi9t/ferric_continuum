import tempfile
import unittest
from pathlib import Path

from tools.ferric_run.config import FerricRunConfig, RepoMount, WorkspaceProfile
from tools.ferric_run.preflight import evaluate_cuda_signals, rootfs_path_for_sandbox_target, run_preflight


def make_config(
    root: Path,
    *,
    rootfs_exists: bool = True,
    ferric_mountpoint_exists: bool = True,
    peer_host_exists: bool = False,
    peer_mountpoint_exists: bool = False,
    peer_required: bool = False,
) -> FerricRunConfig:
    rootfs = root / "rootfs"
    bwrap = root / "bwrap"
    ferric = root / "ferric"
    peer = root / "peer"
    if rootfs_exists:
        rootfs.mkdir()
        if ferric_mountpoint_exists:
            (rootfs / "workspace" / "ferric_continuum").mkdir(parents=True)
        if peer_mountpoint_exists:
            (rootfs / "workspace" / "missing_peer").mkdir(parents=True)
    ferric.mkdir()
    if peer_host_exists:
        peer.mkdir()
    bwrap.write_text("#!/bin/sh\nexit 0\n")
    bwrap.chmod(0o755)
    repos = {
        "ferric_continuum": RepoMount("ferric_continuum", ferric, "/workspace/ferric_continuum", "rw", True),
        "missing_peer": RepoMount("missing_peer", peer, "/workspace/missing_peer", "ro", peer_required),
    }
    return FerricRunConfig(
        repo_root=ferric,
        profile_path=root / "profile.json",
        rootfs=rootfs,
        bwrap=bwrap,
        workspace=WorkspaceProfile("ferric_continuum", repos),
    )


class PreflightTests(unittest.TestCase):
    def test_sandbox_target_maps_inside_rootfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            rootfs = Path(tmp) / "rootfs"

            self.assertEqual(
                rootfs_path_for_sandbox_target(rootfs, "/workspace/ferric_continuum"),
                rootfs / "workspace" / "ferric_continuum",
            )

    def test_missing_rootfs_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_preflight(make_config(Path(tmp), rootfs_exists=False), cuda_requested=False)
            self.assertFalse(report.ok)
            self.assertTrue(any(check.name == "rootfs" for check in report.checks))

    def test_missing_required_repo_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_preflight(make_config(Path(tmp), peer_required=True), cuda_requested=False)
            self.assertFalse(report.ok)
            self.assertTrue(any("missing_peer" in check.message for check in report.checks))

    def test_missing_required_sandbox_mountpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_preflight(
                make_config(Path(tmp), ferric_mountpoint_exists=False),
                cuda_requested=False,
            )
            self.assertFalse(report.ok)
            self.assertTrue(
                any(
                    check.name == "mountpoint:ferric_continuum"
                    and "/workspace/ferric_continuum" in check.message
                    for check in report.checks
                )
            )
            self.assertFalse(next(m for m in report.mounts if m.name == "workspace:ferric_continuum").present)

    def test_missing_optional_repo_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_preflight(make_config(Path(tmp)), cuda_requested=False)
            self.assertTrue(report.ok)
            self.assertTrue(report.warnings)
            self.assertFalse(next(m for m in report.mounts if m.name == "workspace:missing_peer").present)

    def test_missing_optional_sandbox_mountpoint_warns_and_omits_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_preflight(
                make_config(Path(tmp), peer_host_exists=True, peer_mountpoint_exists=False),
                cuda_requested=False,
            )
            self.assertTrue(report.ok)
            self.assertTrue(any("/workspace/missing_peer" in warning for warning in report.warnings))
            self.assertFalse(next(m for m in report.mounts if m.name == "workspace:missing_peer").present)

    def test_cpu_preflight_does_not_require_cuda_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_preflight(make_config(Path(tmp)), cuda_requested=False)
            self.assertTrue(report.ok)
            self.assertFalse(any(check.name == "cuda" for check in report.checks))

    def test_cuda_signal_evaluation_reports_missing_devices_and_libraries(self):
        ok, messages = evaluate_cuda_signals(device_paths=[], library_roots=[])
        self.assertFalse(ok)
        self.assertTrue(any("device" in message for message in messages))
        self.assertTrue(any("library" in message for message in messages))

    def test_cuda_signal_evaluation_accepts_device_and_library_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            device = root / "nvidia0"
            lib_root = root / "lib"
            device.touch()
            lib_root.mkdir()
            ok, messages = evaluate_cuda_signals(device_paths=[device], library_roots=[lib_root])
            self.assertTrue(ok)
            self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
