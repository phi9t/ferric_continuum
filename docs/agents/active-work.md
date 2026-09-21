<!-- Active Work Control Plane v2026.08.15.1. Canonical source: ultron/docs/agents/active-work.md. Sibling copies are managed by Ultron. -->
# Active work

Kata is the durable system of record for engineering intent. Agent sessions,
Git worktrees, reviews, and pull requests are evidence about work; they do not
create or complete work by themselves.

## Start or resume work

1. Run `kata quickstart`, then search before creating an issue.
2. Select one unblocked issue with `kata next --unowned --agent` or inspect the
   queue with `kata ready --unowned --agent`.
3. Claim the issue before implementation. A failed claim means another actor
   owns it; select another issue.
4. Carry the qualified Kata ref in `KATA_REF`. Link an existing session only
   through `tools/active_work.py adopt-session`; an unlinked session remains
   observational and must not create an issue automatically.
5. For cross-repository work, use an Ultron parent issue and native linked child
   issues in affected repository projects.

## Pause, hand off, or finish

- Record decisions, partial attempts, and remaining work in Kata before a
  pause, compaction, or handoff.
- Set `work.attention` to `needs-human` or `stuck` with a substantive
  `work.attention_msg` when intervention is required. Structural dependency
  blocking uses native `blocked_by` links instead.
- Close only after fresh verification. Supply a substantive message and typed
  evidence such as tests, reviewed paths, commits, or pull requests.
- Leave incomplete work open. Session disappearance, a clean branch, passing
  tests, or a RoboRev result never closes an issue automatically.

## Human supervision

Run `python tools/active_work.py status` for the combined local view and
`python tools/active_work.py reconcile` for stale or inconsistent evidence.
The script is an Ultron-managed copy in every repository, including repo-local
worktrees; do not edit sibling copies directly.
Derived states such as `active`, `recent`, `orphaned`, `review`, and
`untracked-session` are display-only. Resolve diagnostics through an explicit
human or agent action; the projection does not assign, close, repair, commit,
push, open pull requests, or merge.

Federation, remote listeners, and shared services require a separate approved
deployment design. The default topology is one local Kata daemon with one
project per managed repository.
