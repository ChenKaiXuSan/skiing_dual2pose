# Branch policy

## Current state — 2026-11-03

`master` is the only retained development branch locally, on GitHub and in the Pegasus development checkout. It is also the GitHub default branch. The per-frame method was fast-forwarded into master at `39a0ad6664e8d5f87b57215542deb62ce9980c45`; subsequent documentation commits remain on master.

The merged `per-frame-canonicalization` and temporary `codex/release-ready-20261103` branches are retired. Historical Pegasus backup branch tips are preserved as archive tags before branch removal. Detached worktrees are preserved; they are not additional branches.

## Preserved snapshots

| Archive tag | Commit | Meaning |
| --- | --- | --- |
| `archive/master-before-per-frame-20261103` | `f3ab2bc8d7f43097495e62c504458fa9c34195b0` | Master before per-frame integration |
| `archive/per-frame-before-release-20261103` | `3d2e05cc2536ae81b7aa5f233804f7d8ad25121a` | Per-frame implementation before release packaging |

Tags preserve prior commits without retaining development branches. No history is rewritten and no force push is needed. Backup tags also preserve historical files; branch deletion is not a publication scrub.

## Experiment and manuscript protection

Use independent per-frame coordinates, common-13 primary scoring and mm for position errors. Keep the sampled-frame time basis for acceleration. Record each run's exact source hashes, checkpoint, data split and configuration; a moving master branch is not a substitute for immutable experiment provenance.

Changing branch names and documentation does not alter frozen sources, results or manuscripts and does not require rerunning experiments. Never run `git clean -fdx` as branch cleanup. Keep detached worktrees and ignored artifacts unless separately authorized for removal.

Dated runbooks mentioning the former per-frame branch or preserved old master describe their observation dates, not the current branch policy. Earlier branch inventories remain available in Git history. Current publication requirements are in [release readiness](reproducibility/release_readiness.md); the repository remains private.
