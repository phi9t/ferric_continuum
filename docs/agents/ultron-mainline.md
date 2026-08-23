# Ultron Mainline Operations

Read this guide whenever Ultron-related work chooses or validates a base branch.

## Dual mainlines

Each canonical repository keeps two distinct lines:

- `upstream_mainline` is the repository's native `main` or `master` and remains upstream-facing.
- `ultron_mainline` is the local-only `ultron/mainline` integration line for Ultron-related work.
- `ultron_mainline_bootstrap` is the immutable approved commit used only for initial creation.

The integration branch has no upstream. It is never implicitly published, rebased, reset, force-updated, or rewritten. Agents do not edit, commit on, advance, or publish it.

## Before Ultron-related work

Run the read-only contract check from Ultron:

```bash
python tools/ultron_mainline.py --check
```

The integration worktree is exactly `<repo>/.worktrees/ultron-mainline`. A ticket branch starts from the exact current `ultron/mainline` tip and uses a separate repo-local path such as `<repo>/.worktrees/<ticket-name>`. Verify the ticket branch's merge-base against that captured tip before implementation.

Read-only diagnosis may inspect refs, Git common directories, worktree registrations, branch ancestry, upstream configuration, and worktree status. It must leave primary checkouts, feature branches, dirty files, and unrelated worktrees unchanged.

## Repository-native verification

The canonical contract test is `tests/test_ultron_mainline.py`, but the managed
manifest declares its installation path per repository. Monarch runs the copy
at `python/tests/test_ultron_mainline.py`. Ferric Continuum owns the root
`tests/test_ultron_mainline.py` copy and synchronized tool through the managed
`//tests:ultron_mainline_test` Bazel target. Other repositories use the root
`tests/` destination with their native Python runner.

Run the command for the current repository from its ticket worktree root:

- Ultron, TorchTitan, NCCL, and tnsr:

  ```bash
  python -B -m pytest -q -p no:cacheprovider tests/test_ultron_mainline.py
  ```

- Megatron-LM (temporary state must remain in the repository-local worktree):

  ```bash
  (
    set -eu
    local_created=0
    test ! -L local
    if test ! -e local; then
      mkdir local
      local_created=1
    fi
    test -d local
    test ! -L local/ultron-mainline-test-tmp
    test ! -e local/ultron-mainline-test-tmp
    mkdir local/ultron-mainline-test-tmp
    test_status=0
    TMPDIR="$PWD/local/ultron-mainline-test-tmp" python -B -m pytest -q -p no:cacheprovider tests/test_ultron_mainline.py || test_status=$?
    test -z "$(find local/ultron-mainline-test-tmp -mindepth 1 -print -quit)"
    rmdir local/ultron-mainline-test-tmp
    if test "$local_created" -eq 1; then
      rmdir local
    fi
    exit "$test_status"
  )
  ```

- Monarch (the repository's mandatory hermetic gateway):

  ```bash
  scripts/run python -B -c 'import runpy; runpy.run_path("python/tests/test_ultron_mainline.py", run_name="__main__")'
  ```

- Ferric Continuum:

  ```bash
  bazel test //tests:ultron_mainline_test
  ```

## Bootstrap

An approved operator may bootstrap with:

```bash
python tools/ultron_mainline.py --bootstrap
```

Bootstrap resolves every canonical repository before writing. Any missing, extra, changed, unsafe, dirty, divergent, or wrongly registered repository stops the entire preflight and reports one categorized line per repository. Independent repository creation cannot be atomic; if execution later fails, successful repositories remain intact and the exact completed subset is reported. Correct the external cause and rerun to resume idempotently. Bootstrap never rolls work back.

## Human integration boundary

Only an explicitly human-authorized, non-rewriting operation may advance `ultron/mainline`, integrate a reviewed ticket, or synchronize an upstream change. Fetching, merging, rebasing, resetting, force-updating, pushing, remote branch creation, and changing publication status are outside this bootstrap/check tool.

After human integration, agents use `--check` to accept a clean descendant of the approved bootstrap commit. They do not rerun bootstrap to advance an established line and never work directly in the integration worktree.
