import json
import tempfile
import unittest
from pathlib import Path

from tools.ferric_run.config import load_profile, resolve_config, resolve_repo_host


class ConfigTests(unittest.TestCase):
    def test_load_profile_defaults_peer_repo_to_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
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
                        "modular": {
                            "host": str(root / "modular"),
                            "sandbox": "/workspace/modular",
                            "required": False,
                        },
                    },
                },
            }))

            loaded = load_profile(profile, repo_root=root)

            self.assertEqual(loaded.primary_repo, "ferric_continuum")
            self.assertEqual(loaded.repos["ferric_continuum"].mode, "rw")
            self.assertEqual(loaded.repos["modular"].mode, "ro")
            self.assertFalse(loaded.repos["modular"].required)

    def test_repo_self_resolves_to_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(resolve_repo_host("repo://self", repo_root=root), root)

    def test_resolve_config_prefers_cli_over_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.json"
            cli_rootfs = root / "cli-rootfs"
            env_rootfs = root / "env-rootfs"
            profile.write_text(json.dumps({
                "workspace": {
                    "primary_repo": "ferric_continuum",
                    "repos": {
                        "ferric_continuum": {
                            "host": "repo://self",
                            "sandbox": "/workspace/ferric_continuum",
                            "mode": "rw",
                            "required": True,
                        }
                    },
                },
            }))

            cfg = resolve_config(
                repo_root=root,
                profile_path=profile,
                rootfs_arg=str(cli_rootfs),
                bwrap_arg="/custom/bwrap",
                environ={"FERRIC_ROOTFS": str(env_rootfs), "FERRIC_BWRAP": "/env/bwrap"},
            )

            self.assertEqual(cfg.rootfs, cli_rootfs)
            self.assertEqual(cfg.bwrap, Path("/custom/bwrap"))


if __name__ == "__main__":
    unittest.main()
