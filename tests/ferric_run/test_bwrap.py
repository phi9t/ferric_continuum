import tempfile
import unittest
from pathlib import Path

from tools.ferric_run.bwrap import (
    BwrapPlan,
    MountSpec,
    build_bwrap_argv,
    plan_to_json_dict,
)


class BwrapPlanTests(unittest.TestCase):
    def test_build_bwrap_argv_orders_workspace_mounts_and_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = BwrapPlan(
                run_id="20260821T000000000000Z",
                rootfs=root / "rootfs",
                bwrap=Path("/usr/bin/bwrap"),
                target_repo="ferric_continuum",
                cwd="/workspace/ferric_continuum",
                mounts=[
                    MountSpec("workspace:modular", root / "modular", "/workspace/modular", "ro", False, True),
                    MountSpec("workspace:ferric_continuum", root / "ferric", "/workspace/ferric_continuum", "rw", True, True),
                ],
                environment={"HOME": "/home/ferric", "USER": "ferric"},
                command_argv=["true"],
                cuda_requested=False,
            )

            argv = build_bwrap_argv(plan)

            self.assertEqual(argv[0], "/usr/bin/bwrap")
            ferric_idx = argv.index("/workspace/ferric_continuum")
            modular_idx = argv.index("/workspace/modular")
            self.assertLess(ferric_idx, modular_idx)
            self.assertIn("--bind", argv)
            self.assertIn("--ro-bind", argv)
            self.assertEqual(argv[-2:], ["--", "true"])

    def test_plan_json_contains_hash_and_mount_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = BwrapPlan(
                run_id="20260821T000000000000Z",
                rootfs=root / "rootfs",
                bwrap=Path("/usr/bin/bwrap"),
                target_repo="ferric_continuum",
                cwd="/workspace/ferric_continuum",
                mounts=[MountSpec("workspace:missing", root / "missing", "/workspace/missing", "ro", False, False)],
                environment={},
                command_argv=["true"],
                cuda_requested=True,
            )

            data = plan_to_json_dict(plan)

            self.assertEqual(data["cuda"]["requested"], True)
            self.assertIn("bwrap_argv_sha256", data)
            self.assertEqual(data["mounts"][0]["present"], False)

    def test_cuda_flag_does_not_add_projection_mounts_to_bwrap_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {
                "run_id": "20260821T000000000000Z",
                "rootfs": root / "rootfs",
                "bwrap": Path("/usr/bin/bwrap"),
                "target_repo": "ferric_continuum",
                "cwd": "/workspace/ferric_continuum",
                "mounts": [
                    MountSpec(
                        "workspace:ferric_continuum",
                        root / "ferric",
                        "/workspace/ferric_continuum",
                        "rw",
                        True,
                        True,
                    ),
                ],
                "environment": {"HOME": "/home/ferric", "USER": "ferric"},
                "command_argv": ["true"],
            }
            cpu_plan = BwrapPlan(cuda_requested=False, **common)
            cuda_plan = BwrapPlan(cuda_requested=True, **common)

            self.assertEqual(build_bwrap_argv(cuda_plan), build_bwrap_argv(cpu_plan))


if __name__ == "__main__":
    unittest.main()
