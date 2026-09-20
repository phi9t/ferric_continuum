#!/bin/bash
# Smoke-test scripts/check-commit-scrub.sh against real Git commit metadata.

set -eu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_REPO="$(mktemp -d)"
trap 'rm -rf "${TMP_REPO}"' EXIT

git -C "${TMP_REPO}" init -q
git -C "${TMP_REPO}" config user.name "Ferric Automation"
git -C "${TMP_REPO}" config user.email "noreply@bytedance.com"
export GIT_AUTHOR_NAME="Ferric Automation"
export GIT_AUTHOR_EMAIL="noreply@bytedance.com"
export GIT_COMMITTER_NAME="Ferric Automation"
export GIT_COMMITTER_EMAIL="noreply@bytedance.com"
bad_path="/""home/local.user/model"

printf 'ok\n' > "${TMP_REPO}/file.txt"
git -C "${TMP_REPO}" add file.txt
tree="$(git -C "${TMP_REPO}" write-tree)"
base="$(git -C "${TMP_REPO}" commit-tree "${tree}" -m "Initial clean commit")"
git -C "${TMP_REPO}" update-ref refs/heads/main "${base}"
git -C "${TMP_REPO}" symbolic-ref HEAD refs/heads/main
base="$(git -C "${TMP_REPO}" rev-parse HEAD)"

printf 'ok2\n' >> "${TMP_REPO}/file.txt"
git -C "${TMP_REPO}" add file.txt
tree="$(git -C "${TMP_REPO}" write-tree)"
head="$(git -C "${TMP_REPO}" commit-tree "${tree}" -p "${base}" -m "Second clean commit")"
git -C "${TMP_REPO}" update-ref refs/heads/main "${head}"
head="$(git -C "${TMP_REPO}" rev-parse HEAD)"
(
  cd "${TMP_REPO}"
  "${REPO_ROOT}/scripts/check-commit-scrub.sh" --range "${base}..${head}"
)

printf 'bad\n' >> "${TMP_REPO}/file.txt"
git -C "${TMP_REPO}" add file.txt
tree="$(git -C "${TMP_REPO}" write-tree)"
bad_head="$(git -C "${TMP_REPO}" commit-tree "${tree}" -p "${head}" -m "Bad path ${bad_path}")"
git -C "${TMP_REPO}" update-ref refs/heads/main "${bad_head}"
bad_head="$(git -C "${TMP_REPO}" rev-parse HEAD)"
if (
  cd "${TMP_REPO}"
  "${REPO_ROOT}/scripts/check-commit-scrub.sh" --range "${head}..${bad_head}"
); then
  echo "expected commit-scrub to reject sensitive commit metadata" >&2
  exit 1
fi

msg="$(mktemp)"
printf 'Clean message\n' > "${msg}"
"${REPO_ROOT}/scripts/check-commit-scrub.sh" --message-file "${msg}"
printf 'Bad message %s\n' "${bad_path}" > "${msg}"
if "${REPO_ROOT}/scripts/check-commit-scrub.sh" --message-file "${msg}"; then
  echo "expected commit-scrub to reject sensitive message file" >&2
  exit 1
fi
