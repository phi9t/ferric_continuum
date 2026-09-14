#!/bin/bash
# Ferric Continuum - install repo-local git hooks
# Symlinks (or copies) the tracked hooks under scripts/git-hooks/ into
# .git/hooks so every clone can opt in with one command. Idempotent.

set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOOK_SRC_DIR="${SCRIPT_DIR}/git-hooks"

GIT_DIR="$(cd "${PROJECT_ROOT}" && git rev-parse --git-dir)"
HOOK_DST_DIR="${GIT_DIR}/hooks"
mkdir -p "${HOOK_DST_DIR}"

for src in "${HOOK_SRC_DIR}"/*; do
  [[ -f "${src}" ]] || continue
  name="$(basename "${src}")"
  dst="${HOOK_DST_DIR}/${name}"
  cp "${src}" "${dst}"
  chmod +x "${dst}"
  echo "installed hook: ${name} -> ${dst}"
done

echo "git hooks installed. Pre-commit now runs the desensitization checker on staged files."
