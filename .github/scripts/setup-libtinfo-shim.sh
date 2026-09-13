#!/usr/bin/env bash
# Provide a libtinfo.so.5 compat shim on Ubuntu 24.04 CI runners.
#
# The hermetic LLVM 18.1.8 toolchain binaries (clang, etc.) are linked against
# libtinfo.so.5, which Ubuntu 24.04 dropped. Point a stable .so.5 symlink at the
# installed libtinfo.so.6 and refresh the loader cache so Bazel sandboxed
# actions resolve it. Resolve the .so.6 path from the filesystem (dpkg + glob)
# rather than parsing `ldconfig -p`, whose line format is brittle.
#
# Emits `LD_LIBRARY_PATH=...` to stdout for the caller to append to $GITHUB_ENV.
set -euo pipefail

tinfo6="$(dpkg -L libtinfo6 2>/dev/null | grep -m1 'libtinfo\.so\.6$' || true)"
if [ -z "${tinfo6}" ]; then
  tinfo6="$(ls /usr/lib/x86_64-linux-gnu/libtinfo.so.6* 2>/dev/null | head -n1 || true)"
fi
if [ -z "${tinfo6}" ] || [ ! -e "${tinfo6}" ]; then
  echo "::error::libtinfo.so.6 not found; cannot create .so.5 compat symlink" >&2
  exit 1
fi
echo "Using libtinfo.so.6 at: ${tinfo6}" >&2

sudo ln -sf "${tinfo6}" /usr/lib/x86_64-linux-gnu/libtinfo.so.5
sudo ldconfig

if [ ! -e /usr/lib/x86_64-linux-gnu/libtinfo.so.5 ]; then
  echo "::error::libtinfo.so.5 compat symlink missing after ldconfig" >&2
  exit 1
fi

echo "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
