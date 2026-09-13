#!/usr/bin/env bash
# Provide a real libtinfo.so.5 on Ubuntu 24.04 CI runners.
#
# The hermetic LLVM 18.1.8 toolchain binaries (clang, etc.) are linked against
# libtinfo.so.5 and require its versioned symbol NCURSES_TINFO_5.0.19991023.
# Ubuntu 24.04 dropped libtinfo5, and a symlink to libtinfo.so.6 does NOT work
# because .so.6 does not export that versioned symbol. Install the genuine
# libtinfo5 package (from Ubuntu jammy-security) so the symbol resolves, then
# refresh the loader cache before Bazel runs.
#
# Emits `LD_LIBRARY_PATH=...` to stdout for the caller to append to $GITHUB_ENV.
set -euo pipefail

# If a working libtinfo.so.5 with the required symbol is already present, reuse
# it. Otherwise fetch and install the jammy libtinfo5 .deb.
LIBDIR="/usr/lib/x86_64-linux-gnu"
have_symbol() {
  local so="$1"
  [ -e "$so" ] || return 1
  readelf -a "$so" 2>/dev/null | grep -q 'NCURSES_TINFO_5.0.19991023'
}

if ! have_symbol "${LIBDIR}/libtinfo.so.5"; then
  deb="libtinfo5_6.3-2ubuntu0.3_amd64.deb"
  sha256="4df4288404108f1a156d014e8764a064e977e34e6d44931ab60451694c03c90d"
  url="http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/${deb}"
  tmp="$(mktemp -d)"
  echo "Fetching ${deb} ..." >&2
  curl -fsSL --retry 3 -o "${tmp}/${deb}" "${url}"
  echo "${sha256}  ${tmp}/${deb}" | sha256sum -c - >&2
  sudo dpkg -i "${tmp}/${deb}" >&2
  rm -rf "${tmp}"
fi

sudo ldconfig

if ! have_symbol "${LIBDIR}/libtinfo.so.5"; then
  echo "::error::libtinfo.so.5 with NCURSES_TINFO_5.0.19991023 still unavailable" >&2
  exit 1
fi
echo "libtinfo.so.5 ready with required NCURSES_TINFO_5.0 symbol" >&2

echo "LD_LIBRARY_PATH=${LIBDIR}:${LD_LIBRARY_PATH:-}"
