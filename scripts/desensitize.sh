#!/bin/bash
# Ferric Continuum - Desensitization wrapper
# Thin entrypoint around tools/desensitize_check.py so CI, hooks, and humans
# share one command. Fails on PII / machine-identifiable data (home abspaths,
# real usernames, MAC addresses, routable IPv4, personal emails).

set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKER="${PROJECT_ROOT}/tools/desensitize_check.py"

if ! command -v python3 &> /dev/null; then
  echo "Error: python3 not found; the desensitization checker needs it." >&2
  exit 2
fi

# Run the built-in negative-control self-test first so a broken checker fails
# loudly instead of silently passing every scan.
python3 "${CHECKER}" --self-test

# Forward remaining args (e.g. --fix, explicit paths); default is a full scan.
exec python3 "${CHECKER}" "$@"
