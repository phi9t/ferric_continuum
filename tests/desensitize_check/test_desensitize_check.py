"""Unit tests for tools/desensitize_check.py.

Mirrors the tests/tla_check convention (stdlib unittest, no third-party deps).
Each detector gets a positive case (must fire) and a negative/allowlisted case
(must NOT fire); --fix idempotence and the built-in self-test are also covered.
"""

import unittest

from tools import desensitize_check as dc


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.allow = dc.Allowlist.default()

    def find(self, text, path="f.txt"):
        return dc.scan_text(path, text, self.allow)

    def cats(self, text):
        return sorted({f.category for f in self.find(text)})

    # -- home abspaths -------------------------------------------------------
    def test_home_abspath_data02_fires(self):
        self.assertIn("home_abspath", self.cats("cd /mnt/home/local.user/workspace"))

    def test_home_abspath_plain_home_fires(self):
        self.assertIn("home_abspath", self.cats("see /home/local.user/models"))

    def test_home_ferric_sandbox_is_allowlisted(self):
        self.assertEqual([], self.find('HOME=/home/ferric USER=ferric'))

    def test_home_env_var_reference_is_clean(self):
        self.assertEqual([], self.find("${HOME}/workspace/tnsr and $HOME/.local/bin"))

    # -- bare usernames ------------------------------------------------------
    def test_bare_username_fires(self):
        self.assertIn("username", self.cats("author: local.user did this"))

    def test_public_handle_phi9t_allowlisted(self):
        self.assertEqual([], self.find("github.com/phi9t/ferric_continuum"))

    # -- MAC addresses -------------------------------------------------------
    def test_mac_address_fires(self):
        self.assertIn("mac_address", self.cats("nic 00:1b:44:11:3a:b7 up"))

    def test_mac_not_confused_with_time(self):
        self.assertEqual([], self.find("elapsed 12:30:45 seconds"))

    # -- routable IPv4 -------------------------------------------------------
    def test_public_ipv4_fires(self):
        self.assertIn("ipv4", self.cats("connect 8.8.8.8 now"))

    def test_loopback_allowlisted(self):
        self.assertEqual([], self.find("bind 127.0.0.1:8080"))

    def test_private_ranges_allowlisted(self):
        self.assertEqual([], self.find("10.0.0.5 192.168.1.1 172.16.0.9 0.0.0.0"))

    def test_version_like_quad_allowlisted(self):
        # x.y.z.w in an obvious version context should not be an IP finding.
        self.assertEqual([], self.find("bazel 9.2.0.0 and proto v1.2.3.4"))

    def test_pep440_specifier_quad_allowlisted(self):
        self.assertEqual([], self.find("foo==1.0.0.0 and bar ~= 2.0.0.0"))

    def test_version_like_path_segment_allowlisted(self):
        # A dotted-quad that is a URL/path segment is a version, not an IP.
        self.assertEqual([], self.find("https://bcr.bazel.build/modules/x/1.3.1.2/MODULE.bazel"))

    # -- personal emails -----------------------------------------------------
    def test_personal_email_fires(self):
        self.assertIn("email", self.cats("contact someone@example.com"))

    def test_allowlisted_emails_clean(self):
        self.assertEqual(
            [],
            self.find(
                "Co-authored-by: TRAE CLI <noreply@bytedance.com> "
                "git@github.com noreply@github.com support@github.com"
            ),
        )

    def test_github_masked_noreply_email_clean(self):
        self.assertEqual([], self.find("108351695+phi9t@users.noreply.github.com"))

    # -- multiple categories on one line ------------------------------------
    def test_multiple_findings_same_line(self):
        cats = self.cats("owner local.user used /mnt/home/bob mailed x@example.com from 8.8.8.8")
        self.assertEqual(cats, ["email", "home_abspath", "ipv4", "username"])


class FixTests(unittest.TestCase):
    def setUp(self):
        self.allow = dc.Allowlist.default()

    def test_fix_bazel_launcher_path_to_plain_bazel(self):
        src = "/mnt/home/local.user/.local/bin/bazel-9.2.0 build //x"
        fixed, changed = dc.fix_text(src, self.allow)
        self.assertTrue(changed)
        self.assertEqual(fixed, "bazel build //x")

    def test_fix_home_abspath_to_home_var(self):
        src = "path /mnt/home/local.user/workspace/tnsr end"
        fixed, changed = dc.fix_text(src, self.allow)
        self.assertTrue(changed)
        self.assertEqual(fixed, "path ${HOME}/workspace/tnsr end")

    def test_fix_plain_home_abspath(self):
        src = "/home/local.user/models/qwen3"
        fixed, _ = dc.fix_text(src, self.allow)
        self.assertEqual(fixed, "${HOME}/models/qwen3")

    def test_fix_is_idempotent(self):
        src = "/mnt/home/local.user/.local/bin/bazel-9.2.0 test //y"
        once, _ = dc.fix_text(src, self.allow)
        twice, changed2 = dc.fix_text(once, self.allow)
        self.assertEqual(once, twice)
        self.assertFalse(changed2)

    def test_fixed_text_has_no_findings(self):
        src = "run /mnt/home/local.user/.local/bin/bazel-9.2.0 in /home/local.user/x"
        fixed, _ = dc.fix_text(src, self.allow)
        self.assertEqual([], dc.scan_text("f", fixed, self.allow))

    def test_fix_leaves_ferric_sandbox_alone(self):
        src = "HOME=/home/ferric"
        fixed, changed = dc.fix_text(src, self.allow)
        self.assertFalse(changed)
        self.assertEqual(fixed, src)


class SelfTestTests(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertEqual(0, dc.run_self_test())


class SelfExemptionTests(unittest.TestCase):
    def test_checker_sources_are_skipped(self):
        from pathlib import Path

        self.assertTrue(dc._skip(Path("tools/desensitize_check.py")))
        self.assertTrue(dc._skip(Path("tests/desensitize_check/test_desensitize_check.py")))
        self.assertTrue(dc._skip(Path(".scratch/desensitize-repo/spec.org")))

    def test_ordinary_files_are_not_skipped(self):
        from pathlib import Path

        self.assertFalse(dc._skip(Path("ferric_continuum/tnsr/README.md")))


class VendorScopeTests(unittest.TestCase):
    def setUp(self):
        self.allow = dc.Allowlist.default()

    def test_vendor_email_and_ip_allowlisted_by_path(self):
        text = "contact service@deepseek.com from 8.8.8.8"
        vendor = dc.scan_text("x/third_party/deepseek/README.md", text, self.allow)
        self.assertEqual([], vendor)

    def test_vendor_still_flags_our_home_abspath(self):
        text = "built at /mnt/home/local.user/x"
        vendor = dc.scan_text("x/third_party/deepseek/notes.md", text, self.allow)
        self.assertEqual(["home_abspath"], sorted({f.category for f in vendor}))


class AllowlistFileTests(unittest.TestCase):
    def test_extra_literal_allowlist_silences_finding(self):
        allow = dc.Allowlist.default()
        self.assertIn("ipv4", {f.category for f in dc.scan_text("f", "9.9.9.9", allow)})
        allow.add_line("9.9.9.9")
        self.assertEqual([], dc.scan_text("f", "9.9.9.9", allow))

    def test_regex_allowlist_line(self):
        allow = dc.Allowlist.default()
        allow.add_line("re:203\\.0\\.113\\.\\d+")
        self.assertEqual([], dc.scan_text("f", "203.0.113.7", allow))


if __name__ == "__main__":
    unittest.main()
