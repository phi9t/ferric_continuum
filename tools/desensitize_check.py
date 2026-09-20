#!/usr/bin/env python3
"""Desensitization checker: fail on PII / machine-identifiable data in tracked files.

Detects, per line:
  - home_abspath : optional storage prefix plus /home/<user> (except /home/ferric)
  - username     : a configured real username
  - mac_address  : xx:xx:xx:xx:xx:xx
  - ipv4         : routable IPv4 (excludes loopback/private/version-like/allowed)
  - email        : personal email (excludes allowlisted project emails)

Design goals (match tools/tla_check.py): stdlib only, small and testable.

Usage:
  desensitize_check.py [PATH ...]      scan tracked files (or given paths); exit 1 on findings
  desensitize_check.py --fix [PATH..]  apply safe substitutions in place
  desensitize_check.py --self-test     run built-in negative controls; exit 0 if the
                                        detectors bite exactly where expected

Exit codes: 0 clean, 1 findings, 2 harness/self-test error.

The allowlist lives in tools/desensitize_allowlist.txt (one token per line;
lines beginning `re:` are treated as regexes; `#` comments and blanks ignored).
Tokens keep intended-public identity (phi9t, noreply@bytedance.com, ...) from
tripping the scan.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
ALLOWLIST_PATH = HERE / "desensitize_allowlist.txt"

# --- detector patterns ------------------------------------------------------
# Home abspaths: an optional storage prefix (e.g. /data02) then /home/<user>.
# <user> is a login-name token; we capture it so the sandbox user can be spared.
HOME_ABSPATH_RE = re.compile(r"(?:/[a-zA-Z0-9_.-]+)?/home/([a-zA-Z0-9][a-zA-Z0-9._-]*)")
# The pinned Bazel launcher path -> plain `bazel` (version pinned by .bazelversion).
BAZEL_LAUNCHER_RE = re.compile(
    r"(?:/[a-zA-Z0-9_.-]+)?/home/[a-zA-Z0-9][a-zA-Z0-9._-]*/\.local/bin/bazel(?:-[0-9.]+)?"
)
MAC_RE = re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:])")
IPV4_RE = re.compile(r"(?<![\w.])((?:\d{1,3}\.){3}\d{1,3})(?![\w.])")
# A dotted-quad that is part of a longer dotted-numeric run (e.g. 1.2.3.4.5 or a
# version like 9.2.0.0) is not an IP; guard by requiring no adjacent `.<digit>`.
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# The sandbox container user is fabricated, not a real person.
SANDBOX_USERS = {"ferric"}

# Seed deny-list of real usernames matched as bare tokens (outside abspaths).
# Extra project-local usernames can be supplied without committing them:
#   DESENSITIZE_USERNAMES="name.one,name.two" python3 tools/desensitize_check.py
REAL_USERNAMES = sorted(
    {
        u.strip()
        for u in ("local.user," + os.environ.get("DESENSITIZE_USERNAMES", "")).split(",")
        if u.strip() and u.strip() not in SANDBOX_USERS
    }
)
USERNAME_RES = [re.compile(r"(?<![\w.@/-])" + re.escape(u) + r"(?![\w.-])") for u in REAL_USERNAMES]

SKIP_DIRS = {".git", ".worktrees", "node_modules", "__pycache__"}
SKIP_DIR_PREFIXES = ("bazel-",)

# The checker's own sources, unit tests, and spec deliberately embed PII-shaped
# fixtures (known-bad strings the detectors must fire on). Scanning them would
# be self-defeating, so they are exempt by path. Keep this list tight.
SELF_EXEMPT_SUFFIXES = (
    "tools/desensitize_check.py",
    "tools/desensitize_allowlist.txt",
    "tests/desensitize_check/test_desensitize_check.py",
    ".scratch/desensitize-repo/spec.org",
)

# Vendored upstream artifacts are imported verbatim; their contents (including
# the upstream vendor's own public contact emails) are not our PII to rewrite.
# We still scan them for OUR home abspaths/usernames via the detectors, but the
# email/ipv4 categories are treated as vendor-owned and allowlisted there.
VENDOR_PATH_MARKERS = ("/third_party/",)
VENDOR_ALLOWED_CATEGORIES = {"email", "ipv4", "mac_address"}


@dataclass
class Allowlist:
    literals: set = field(default_factory=set)
    regexes: list = field(default_factory=list)

    @classmethod
    def default(cls) -> "Allowlist":
        allow = cls()
        # Intended-public identity + benign values, always allowlisted.
        for tok in ("phi9t", "noreply@bytedance.com", "git@github.com"):
            allow.literals.add(tok)
        if ALLOWLIST_PATH.exists():
            for line in ALLOWLIST_PATH.read_text().splitlines():
                allow.add_line(line)
        return allow

    def add_line(self, line: str) -> None:
        line = line.strip()
        if not line or line.startswith("#"):
            return
        if line.startswith("re:"):
            self.regexes.append(re.compile(line[3:]))
        else:
            self.literals.add(line)

    def allows(self, token: str) -> bool:
        if token in self.literals:
            return True
        return any(r.search(token) for r in self.regexes)


@dataclass
class Finding:
    path: str
    line: int
    col: int
    category: str
    token: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.category}: {self.token}"


def _is_version_like(match: re.Match, text: str) -> bool:
    """True if a dotted-quad is really a version string, not an IP.

    Two signals: (a) it is embedded in a longer dotted-numeric run
    (e.g. 1.2.3.4.5), or (b) it is preceded by a version keyword/marker
    (e.g. `bazel 9.2.0.0`, `v1.2.3.4`, `version 1.2.3.4`)."""
    start, end = match.span(1)
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if before == "." or after == ".":
        return True
    prefix = text[:start].rstrip()
    if prefix.endswith("v") or prefix.endswith("V"):
        return True
    if re.search(r"(?i)\b(version|ver|bazel|proto|v|release|rev|tag)\s*$", prefix):
        return True
    # A dotted-quad that is a path/URL segment (…/1.3.1.2/…) is a version, not
    # an IP: it is delimited by slashes rather than whitespace/punctuation.
    raw_before = text[start - 1] if start > 0 else ""
    raw_after = text[end] if end < len(text) else ""
    if raw_before == "/" and raw_after == "/":
        return True
    return False


def _ipv4_is_routable(ip: str) -> bool:
    try:
        octets = [int(o) for o in ip.split(".")]
    except ValueError:
        return False
    if len(octets) != 4 or any(o > 255 for o in octets):
        return False  # not a valid IPv4 at all
    a, b = octets[0], octets[1]
    if a == 127 or a == 0:
        return False  # loopback / unspecified
    if a == 10:
        return False  # 10.0.0.0/8
    if a == 192 and b == 168:
        return False  # 192.168.0.0/16
    if a == 172 and 16 <= b <= 31:
        return False  # 172.16.0.0/12
    if a == 169 and b == 254:
        return False  # link-local
    return True


def scan_text(path: str, text: str, allow: Allowlist) -> List[Finding]:
    findings: List[Finding] = []
    is_vendor = any(marker in path.replace(os.sep, "/") for marker in VENDOR_PATH_MARKERS)
    for lineno, line in enumerate(text.splitlines(), start=1):
        # home abspaths (skip the sandbox user)
        for m in HOME_ABSPATH_RE.finditer(line):
            user = m.group(1)
            if user in SANDBOX_USERS:
                continue
            if allow.allows(m.group(0)):
                continue
            findings.append(Finding(path, lineno, m.start() + 1, "home_abspath", m.group(0)))
        # bare usernames (outside abspaths, which are already caught above)
        for ure in USERNAME_RES:
            for m in ure.finditer(line):
                # avoid double-counting when inside a /home/<user> abspath
                if line[max(0, m.start() - 6):m.start()].endswith("/home/"):
                    continue
                if allow.allows(m.group(0)):
                    continue
                findings.append(Finding(path, lineno, m.start() + 1, "username", m.group(0)))
        # MAC addresses
        for m in MAC_RE.finditer(line):
            if allow.allows(m.group(0)):
                continue
            findings.append(Finding(path, lineno, m.start() + 1, "mac_address", m.group(0)))
        # routable IPv4
        for m in IPV4_RE.finditer(line):
            ip = m.group(1)
            if _is_version_like(m, line):
                continue
            if not _ipv4_is_routable(ip):
                continue
            if allow.allows(ip):
                continue
            findings.append(Finding(path, lineno, m.start(1) + 1, "ipv4", ip))
        # personal emails
        for m in EMAIL_RE.finditer(line):
            if allow.allows(m.group(0)):
                continue
            findings.append(Finding(path, lineno, m.start() + 1, "email", m.group(0)))
    if is_vendor:
        findings = [f for f in findings if f.category not in VENDOR_ALLOWED_CATEGORIES]
    return findings


def fix_text(text: str, allow: Allowlist) -> Tuple[str, bool]:
    """Apply safe, deterministic substitutions. Returns (new_text, changed)."""
    original = text
    # 1) pinned Bazel launcher path -> plain `bazel` (version via .bazelversion).
    text = BAZEL_LAUNCHER_RE.sub("bazel", text)

    # 2) remaining home abspaths -> ${HOME}/... (spare the sandbox user).
    def repl_home(m: re.Match) -> str:
        if m.group(1) in SANDBOX_USERS:
            return m.group(0)
        if allow.allows(m.group(0)):
            return m.group(0)
        return "${HOME}"

    text = HOME_ABSPATH_RE.sub(repl_home, text)
    return text, (text != original)


def iter_target_files(paths: List[str]) -> List[Path]:
    if paths:
        out = []
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                out.extend(x for x in pp.rglob("*") if x.is_file())
            elif pp.is_file():
                out.append(pp)
        return out
    # default: tracked files via git
    try:
        res = subprocess.run(
            ["git", "ls-files", "-z"], capture_output=True, text=True, check=True
        )
        names = [n for n in res.stdout.split("\0") if n]
        return [Path(n) for n in names]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [x for x in Path(".").rglob("*") if x.is_file()]


def _skip(path: Path) -> bool:
    parts = path.parts
    for part in parts:
        if part in SKIP_DIRS or part.startswith(SKIP_DIR_PREFIXES):
            return True
    norm = str(path).replace(os.sep, "/")
    if any(norm.endswith(suffix) for suffix in SELF_EXEMPT_SUFFIXES):
        return True
    return False


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None  # binary
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def run_self_test() -> int:
    """Built-in negative controls: detectors must fire on bad input and stay
    silent on allowlisted input. Returns 0 on success, nonzero on misbehavior."""
    allow = Allowlist.default()
    checks = [
        # (text, expected_category_present, description)
        ("/mnt/home/local.user/x", "home_abspath", "prefixed home abspath fires"),
        ("/home/local.user/x", "home_abspath", "plain home abspath fires"),
        ("author local.user", "username", "bare username fires"),
        ("nic 00:1b:44:11:3a:b7", "mac_address", "mac fires"),
        ("connect 8.8.8.8", "ipv4", "public ipv4 fires"),
        ("mail someone@example.com", "email", "personal email fires"),
    ]
    negatives = [
        ("HOME=/home/ferric", "home_abspath", "sandbox user spared"),
        ("github.com/phi9t/x", "username", "public handle spared"),
        ("bind 127.0.0.1", "ipv4", "loopback spared"),
        ("10.0.0.1 192.168.0.1", "ipv4", "private spared"),
        ("bazel 9.2.0.0", "ipv4", "version-like spared"),
        ("noreply@bytedance.com", "email", "project email spared"),
        ("${HOME}/workspace/x", "home_abspath", "home var spared"),
    ]
    ok = True
    for text, cat, desc in checks:
        cats = {f.category for f in scan_text("selftest", text, allow)}
        if cat not in cats:
            print(f"SELFTEST FAIL: {desc}: expected {cat}, got {sorted(cats)}", file=sys.stderr)
            ok = False
    for text, cat, desc in negatives:
        cats = {f.category for f in scan_text("selftest", text, allow)}
        if cat in cats:
            print(f"SELFTEST FAIL: {desc}: {cat} should not fire on {text!r}", file=sys.stderr)
            ok = False
    # --fix must clear a compound bad line and be idempotent.
    src = "run /mnt/home/local.user/.local/bin/bazel-9.2.0 in /home/local.user/x"
    fixed, changed = fix_text(src, allow)
    if not changed or scan_text("selftest", fixed, allow):
        print(f"SELFTEST FAIL: fix did not clean {src!r} -> {fixed!r}", file=sys.stderr)
        ok = False
    twice, changed2 = fix_text(fixed, allow)
    if changed2 or twice != fixed:
        print("SELFTEST FAIL: fix is not idempotent", file=sys.stderr)
        ok = False
    if ok:
        print("desensitize self-test: OK")
        return 0
    return 2


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="files/dirs to scan (default: tracked files)")
    ap.add_argument("--fix", action="store_true", help="apply safe substitutions in place")
    ap.add_argument("--self-test", action="store_true", help="run built-in negative controls")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    allow = Allowlist.default()
    files = [p for p in iter_target_files(args.paths) if not _skip(p)]

    all_findings: List[Finding] = []
    fixed_count = 0
    for path in files:
        text = _read_text(path)
        if text is None:
            continue
        if args.fix:
            new_text, changed = fix_text(text, allow)
            if changed:
                path.write_text(new_text)
                fixed_count += 1
                text = new_text
        all_findings.extend(scan_text(str(path), text, allow))

    if args.fix:
        print(f"desensitize --fix: rewrote {fixed_count} file(s)")

    if all_findings:
        for f in all_findings:
            print(str(f))
        print(
            f"\ndesensitize: {len(all_findings)} finding(s). "
            f"Run `python3 {Path(__file__).name} --fix` for auto-fixable ones "
            f"or add a reviewed line to {ALLOWLIST_PATH.name}.",
            file=sys.stderr,
        )
        return 1

    print(f"desensitize: clean ({len(files)} file(s) scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
