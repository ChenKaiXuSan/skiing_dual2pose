# Branch roles and cleanup inventory

Reviewed on 2026-10-31. The agreed arrangement keeps both existing branches and preserves `master` as the old code baseline.

## Branch roles

| Branch | Role | Handling |
| --- | --- | --- |
| `master` | Preserved pre-migration baseline; GitHub default branch | Keep at `f3ab2bc8d7f43097495e62c504458fa9c34195b0` for this cleanup. No merge, reset, rename, or deletion. |
| `per-frame-canonicalization` | Active code for per-frame training, evaluation, baseline adaptations, and dependency-based experiment orchestration | Continue current work here. Record the exact code commit in each experiment's artifacts. |

The default GitHub landing page shows `master`; select `per-frame-canonicalization` to inspect the current implementation. The repository is private at the time of review. Neither the default branch nor visibility is changed by this cleanup.

## Relationship at review time

Before this documentation update:

- `master`: `f3ab2bc8d7f43097495e62c504458fa9c34195b0`.
- `per-frame-canonicalization`: `07f51ec99a7ad079faa34f94d33d986d7c12d302`.
- The active branch contains all commits from `master`, plus 12 commits. There are no commits unique to `master` relative to the active branch.
- Local branch tips match the remote tips; the tracked working tree is clean.
- GitHub has exactly these two branches, no tags, and no open pull requests. Neither branch reports branch protection enabled. Preservation of `master` is the agreed workflow, not a server-enforced protection rule.

This is an inventory of history and branch roles, not a fresh validation of every code change or a decision to merge.

## Protocol and artifact boundaries

Current experiments use independent per-frame body-coordinate construction for inputs and references, direct output scoring, and the common-13 primary joint set. Position errors are reported in mm; acceleration errors retain their stated sampled-frame time basis. Older experiments must retain their original protocol and provenance.

Record the exact commit alongside the experiment configuration, checkpoint, evaluation indices, and result artifacts. A branch name alone is insufficient because its tip changes. A documentation-only commit does not require rerunning completed experiments or relabeling their code provenance.

The repository ignores `paper/`, `reports/`, `logs/`, and `ckpt/`. Current manuscript edits and author photos therefore do not appear in a code-branch diff or get uploaded by pushing this branch. Preserve these workspace artifacts separately. Branch cleanup does not remove them, change existing experiment outputs, or interrupt running jobs.

## Cleanup candidates

| Item | Finding | Decision |
| --- | --- | --- |
| Remote branches | Only the preserved baseline and active branch exist | Keep both; no remote branch deletion candidates. |
| Local branches | Same two branches as the remote | Keep both. |
| Remote-tracking references | `git remote prune origin --dry-run` reports no stale references | Nothing to prune. |
| Registered detached worktree at `ee701406dcb06ffcd7cdcefac66586fb031f14b0` (`code_ee70140`) | Clean worktree; commit is already an ancestor of the active branch | Candidate for later removal only after confirming no other task uses its directory. Retained during this cleanup. |
| Ignored manuscript and experiment artifacts | Kept outside code commits | Preserve; do not use `git clean -fdx` as branch cleanup. |

Future branch removal should first check unique commits, open pull requests, registered worktrees, and job dependencies. This review does not authorize merging the active branch into `master`, changing repository visibility, or deleting the retained worktree.
