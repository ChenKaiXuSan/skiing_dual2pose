# Branch roles and cleanup inventory

Reviewed on 2026-10-31. The agreed arrangement keeps both existing branches and preserves `master` as the old code baseline.

## Branch roles

| Branch | Role | Handling |
| --- | --- | --- |
| `master` | Preserved pre-migration baseline; GitHub default branch | Keep at `f3ab2bc8d7f43097495e62c504458fa9c34195b0` for this cleanup. No merge, reset, rename, or deletion. |
| `per-frame-canonicalization` | Active code for per-frame training, evaluation, baseline adaptations, and dependency-based experiment orchestration | Continue current work here. Record the exact code commit in each experiment's artifacts. |

The default GitHub landing page shows `master`; select `per-frame-canonicalization` to inspect the current implementation. The repository is private at the time of review. Neither the default branch nor visibility is changed by this cleanup.

## Initial review snapshot

Before this documentation update:

- `master`: `f3ab2bc8d7f43097495e62c504458fa9c34195b0`.
- `per-frame-canonicalization`: `07f51ec99a7ad079faa34f94d33d986d7c12d302`.
- The active branch contains all commits from `master`, plus 12 commits. There are no commits unique to `master` relative to the active branch.
- Local branch tips match the remote tips; the tracked working tree is clean.
- GitHub has exactly these two branches, no tags, and no open pull requests. Neither branch reports branch protection enabled. Preservation of `master` is the agreed workflow, not a server-enforced protection rule.

This is an inventory of history and branch roles, not a fresh validation of every code change or a decision to merge.

## Follow-up review: 2026-10-31

Before this follow-up documentation commit, the active local and remote tips both
pointed to `772b256480c09dc5517089a59cb891a148d71e34`, containing all of `master`
plus 13 commits. The local and remote `master` tips remained at the baseline
listed above. GitHub still reports a private repository with `master` as the
default branch. Remote branch enumeration and dry-run pruning found no extra
branches or stale references; worktree dry-run pruning found nothing to remove.

Unlike the initial review, the working tree now contains uncommitted recovery
and performance work. The following groups describe candidate future commits;
they are preserved in place and are not included in this documentation cleanup.

| Candidate group | Files / scope | Integration boundary |
| --- | --- | --- |
| Geometry input relocation | `dual2pose/experiments/run_geometry_baselines.py`, `tests/test_run_geometry_baselines.py` | Resolve archived input paths while preserving manifest provenance checks. |
| Verified Unity cache and performance diagnosis | `dual2pose/train_unity.py`, `dual2pose/dataloader/unity_cached_data.py`, `pegasus/runtime/diagnose_unity_performance.py`, `tests/test_unity_cached_data.py` | Keep the optional storage path separate from changes to pose or training protocols. |
| Pegasus successor recovery | `pegasus/runtime/dispatch.py`, `pegasus/runtime/recover.py`, `tests/test_pegasus_dispatch.py`, `tests/test_pegasus_recover.py` | Review completed-task imports, immutable evidence, and dependency release together. |
| Experiment records | `docs/plans/2026-10-31-pegasus-recovery.md`, `docs/experiments/pegasus_recovery_20261031.md`, `docs/experiments/metapose_unity_diagnosis_20261031.md` | Preserve observations and their timestamps; do not treat them as a fresh live-job check. |

Existing recovery notes describe a deployed snapshot with recorded source
hashes. This branch inventory does not independently validate its running jobs
or repeat its tests. Do not attribute uncommitted or deployed source changes to
the branch HEAD alone. Before integrating these groups, inspect their final
diffs and validation evidence; preserve active workers and prior outputs.

The retained detached worktree remains a conditional cleanup candidate, not a
deletion target. No branch merge, history rewrite, visibility change, worktree
removal, or experiment-file cleanup is part of this follow-up.

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
