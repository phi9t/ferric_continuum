#!/bin/bash
# Ferric Continuum - scan commit messages and identities for PII/MII.
#
# Usage:
#   scripts/check-commit-scrub.sh --message-file .git/COMMIT_EDITMSG
#   scripts/check-commit-scrub.sh --range origin/ultron/mainline..HEAD
#
# The file-content scrubber lives in tools/desensitize_check.py. This wrapper
# feeds commit metadata through the same detectors so hooks and PR CI cover the
# text Git stores outside the worktree.

set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKER="${PROJECT_ROOT}/tools/desensitize_check.py"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/check-commit-scrub.sh --message-file PATH
  scripts/check-commit-scrub.sh --range BASE..HEAD
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

if ! command -v python3 &> /dev/null; then
  echo "commit-scrub: python3 not found; cannot scan commit metadata." >&2
  exit 2
fi

if [[ ! -f "${CHECKER}" ]]; then
  echo "commit-scrub: checker not found at ${CHECKER}" >&2
  exit 2
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

case "$1" in
  --message-file)
    msg="$2"
    if [[ ! -f "${msg}" ]]; then
      echo "commit-scrub: message file not found: ${msg}" >&2
      exit 2
    fi
    cp "${msg}" "${tmpdir}/commit-message.txt"
    python3 "${CHECKER}" "${tmpdir}/commit-message.txt"
    ;;
  --range)
    range="$2"
    if [[ "${range}" != *".."* ]]; then
      echo "commit-scrub: range must use BASE..HEAD syntax" >&2
      exit 2
    fi
    base="${range%%..*}"
    head="${range#*..}"
    if [[ -n "${base}" ]] && ! git rev-parse --verify --quiet "${base}^{commit}" >/dev/null; then
      echo "commit-scrub: invalid range base in ${range}" >&2
      exit 2
    fi
    if ! git rev-parse --verify --quiet "${head}^{commit}" >/dev/null; then
      echo "commit-scrub: invalid range head in ${range}" >&2
      exit 2
    fi
    git log --format='%H%n%an <%ae>%n%cn <%ce>%n%B%n---END-COMMIT---' "${range}" \
      > "${tmpdir}/commit-range.txt"
    python3 "${CHECKER}" "${tmpdir}/commit-range.txt"
    ;;
  *)
    usage
    exit 2
    ;;
esac
