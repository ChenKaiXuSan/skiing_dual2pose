# Public release readiness — 2026-11-01

Status: **internal preparation; not ready to publish as a complete reproduction**.
The development tree and all existing experiment paths are preserved.

## Completed preparation

- Added a documentation index, dated reproduction inventory and Table 2 audit.
- Corrected README claims about ignored versus tracked manuscripts, fixed-fold
  reporting and release completeness.
- Aligned the README badge and package license metadata with the existing root
  Apache-2.0 license; this does not change third-party licensing.
- Removed the placeholder citation with an incomplete author list.
- Prepared a separate working-tree source candidate with file hashes and explicit
  exclusions. It is an internal review archive, not a tag, commit or publication.

## Items to finish before public release

| Item | Current finding | Needed before release |
| --- | --- | --- |
| Scientific validity | Unity geometry repaired; 18/18 successor tasks and artifact/source verification complete | Preserve versioned evidence; do not reuse pre-repair caches or generalize single-run findings |
| Consolidated source | Frontend recovery, MetaPose stability and geometry fixes are grouped on the active branch | Use the consolidated branch for new work and original frozen hashes for past runs |
| Public data contract | Full DAG relies on workspace-only manifests, prepared inputs and weights | Publish shareable split/index/schema manifests and obtain/download instructions; validate on a clean clone |
| Environment portability | Runtime environments depend on an account-home Python installation | Provide portable environment creation and test CUDA/framework compatibility from scratch |
| Historical manuscripts | 256 tracked files under `paper/`, about 66.8 MB, despite ignore rules | Review contents and full Git history before changing visibility; a clean export alone does not clean history |
| Unity assets | 1,295 tracked files under `unity/`, about 1.53 GB | Review asset redistribution rights; separate optional assets from core pose code |
| Third-party notices | Patched SAM3D tree and gated weights have separate provenance/terms | Verify applicable notices and permitted redistribution; do not imply root Apache-2.0 covers them |
| CI / packaging | Legacy workflows reference an old template/Python 3.7; `MANIFEST.in` contains template paths | Replace and validate a supported clean-clone CI/build before publishing packages |
| Numeric evidence | Corrected geometry and MetaPose timings adopted; unrelated paper rows retain original runs | Keep explicit row/checkpoint provenance; distinguish C07 control and mixed timing hardware |
| Backup | Local text summaries complete; selected Pegasus binaries are not fully mirrored | Back up the missing artifact list and preserve predecessor plans, caches, external assets and source snapshots |

Counts above are the current tracked-file snapshot, not Git-history sizes. No
repository visibility, default branch, master tip, Git history or remote content
was changed by this preparation. Do not run `git clean -fdx` or delete historical
plans to make the repository appear clean.

## Candidate archive boundaries

The workspace archive under `reports/public_release_preparation_20261101/`
contains current core source, tests, configuration, Pegasus scripts, selected
documentation, patched frontend source and Unity C# capture scripts. Its manifest
records the working-tree state and hashes, including relevant uncommitted source.
It omits `.git`, manuscripts, results, datasets, checkpoints, external downloaded
repositories/environments, Unity binary assets, editor settings and obsolete CI.

This archive is for review only. It does not contain the data/weights or complete
portable DAG needed to reproduce every experiment, and third-party notices still
need review. Validate source export separately from environment and full-pipeline
reproduction. The source candidate archive is not a public release. Pushing development commits to the existing repository does not change its visibility or publish a reproduction package.

## Consolidation validation boundary

The latest source check adds a regression for Unity metadata exported before the camera aspect is configured: focal lengths use the requested capture dimensions. This affects future captures; completed experiments used repaired legacy intrinsics and are not rewritten. Unity Editor compilation has not been performed. Existing geometry caches do not hash every source calibration/estimated-2D file; use fresh output directories when those inputs change. CI/package modernization and a clean-clone full pipeline check remain open.
