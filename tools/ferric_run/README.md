# Ferric Run

`tools/ferric_run` is the Ferric-local GPU-kernel study runner. It records a
run, preflights a caller-supplied `bwrap` rootfs, builds the exact generic
`bwrap` command line, and optionally executes the command inside the sandbox.

The runner is influenced by Vaso-style local study harnesses: make the runtime
contract explicit, keep peer repository access declarative, and leave an audit
trail for every attempt. The ownership is Ferric's. Profiles, record layout, and
workspace defaults live in this repository and should evolve with Ferric's CUDA
kernel study needs.

## Scope

The v1 runner is intentionally small:

- It requires an existing rootfs path through `--rootfs` or `FERRIC_ROOTFS`.
- It does not build, sync, or repair rootfs contents.
- It executes only generic command tails after `--`.
- It mounts Ferric as the writable primary repo at
  `/workspace/ferric_continuum`.
- It mounts profile-declared peer repos read-only when present.
- It records CUDA intent when `--cuda` is passed, but does not project CUDA
  devices or driver libraries into the sandbox yet.

## Examples

Use an explicit rootfs and dry run to inspect the run record without executing
inside the sandbox:

```sh
python3 tools/ferric_run.py \
  --rootfs /path/to/rootfs \
  --dry-run \
  -- \
  true
```

Run a CPU command inside the rootfs:

```sh
python3 tools/ferric_run.py \
  --rootfs /path/to/rootfs \
  -- \
  /bin/bash -lc 'python3 --version'
```

Record CUDA intent and require host CUDA signals during preflight:

```sh
python3 tools/ferric_run.py \
  --rootfs /path/to/rootfs \
  --cuda \
  --dry-run \
  -- \
  true
```

Pass `--bwrap /path/to/bwrap` or set `FERRIC_BWRAP` when `bwrap` is not on
`PATH`.

## Run Records

Each invocation creates a generated run directory under:

```text
.ferric/runs/<run-id>/
```

The directory contains:

- `command.json`: host-side invocation details, normalized command argv, rootfs
  path, resolved profile path, dry-run flag, and CUDA request intent.
- `preflight.json`: rootfs, `bwrap`, workspace mount, optional peer, and CUDA
  signal checks.
- `bwrap-plan.json`: the complete planned sandbox contract, including mounts,
  environment, command argv, CUDA intent, full `bwrap` argv, and argv hash.
- `stdout.log` and `stderr.log`: captured subprocess streams for executed runs,
  empty for dry runs or failed preflight.
- `result.json`: execution flag, final exit code, duration, and overall result.

`.ferric/` is generated state and is ignored by git.

## Profiles and Peer Repos

The default profile is:

```text
tools/ferric_run/profiles/gpu-kernel-study.json
```

That profile makes Ferric the primary writable workspace at
`/workspace/ferric_continuum`. Peer repositories are declared by the profile and
default to read-only mounts. Missing optional peers are omitted with a preflight
warning; missing required repos fail preflight.

The default GPU-kernel study profile includes `modular` as an optional read-only
peer, alongside other local research checkouts. Profile entries are the only
source of peer mounts.

## CUDA Boundary

`--cuda` is an intent and preflight flag in v1. When present, the runner records
CUDA intent in `command.json` and `bwrap-plan.json`, then checks for host CUDA
signals in `preflight.json`:

- at least one existing `/dev/nvidia*` device path
- at least one likely NVIDIA or CUDA driver library root

The v1 runner does not add CUDA device mounts, driver-library mounts,
`LD_LIBRARY_PATH`, `PATH`, or other CUDA projection behavior to the sandbox.
Those mounts are a follow-up once the rootfs/device projection contract is
designed and tested.
