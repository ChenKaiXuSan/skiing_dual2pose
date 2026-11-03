# Public-readiness audit — 2026-11-03

**Decision: the core source is a release candidate; the existing GitHub repository is not cleared for a direct switch to public visibility.** `master` is the only retained development branch. The consolidated per-frame method was integrated at `39a0ad6`; removing the former branches does not remove files or history reachable from master and archive tags.

## Verified engineering preparation

- Hydra configuration/model YAML and versioned base-chain recipes are packaged. Archived recipe precedence and provenance hashes are tested.
- GitHub CPU tests and distribution/source-export workflows passed for the integrated `39a0ad6` code: 694 tests passed, 6 skipped, and 24 subtests passed. A source-only export passed 654 tests with 20 skipped and 24 subtests passed. These are dated observations, not a claim about an arbitrary later commit.
- CI now targets pushes to master and pull requests. The README, installation guide, data/weight acquisition contract, final numerical CSVs and checkpoint registry identify the current method and final results.
- `tools/export_source.py` exports allowlisted tracked files (or a prior export manifest), rejects traversal, skips symlinks, excludes untracked files, supplies SHA-256 hashes and refuses overwrite. It includes two project-authored Unity capture scripts but excludes Unity meshes/textures/animations, manuscripts, vendored frontend code, Git history, results, data and weights.
- Root core-code licensing, author contact, manuscript title and data-on-request terms are documented. No publication DOI is invented.

## Items blocking publication of the existing repository

The inventory below was measured at integrated commit `39a0ad6`; this branch/documentation cleanup does not remove these files.

| Item | Verified evidence | Needed before publishing the existing repository |
| --- | --- | --- |
| Historical manuscripts | 256 tracked files under `paper/`, also reachable in Git history | Decide which manuscript/evidence material may be made public; exclude uncleared content from the public repository, including its history |
| Unity project assets | 1,295 tracked files under `unity/`, including character FBX animations and textures; no license/notice/copyright-named files found there | Establish redistribution terms for the actual assets or exclude them from public source and history |
| Vendored frontend | 68 tracked files under `SAM3Dbody/`; upstream provenance exists, but no license/notice/copyright-named files are tracked there | Verify the applicable upstream license and required notices, or exclude vendored frontend code and obtain it separately |
| Historical material beyond core source | Four tracked `.vscode` files, machine-specific paths and old development records remain | Review what is intended for disclosure; deleting a branch or adding ignore rules does not remove tracked or historical content |

This is a file/provenance finding, not a conclusion that any asset is legally prohibited. The root Apache-2.0 license alone does not establish permission for these materials. Archive tags also retain earlier history.

## Credential scan scope

A local read-only pattern scan enumerated 3,206 reachable Git blobs and scanned 2,479 text blobs for recognizable private-key headers and GitHub, Hugging Face, AWS access-ID, Slack and Google API-key formats. It found no matches. Binary content and 167 blobs larger than 2 MB were excluded. This limited scan does not establish the absence of arbitrary passwords, credentials in images, or all sensitive information. Recheck the exact intended public artifact before publication. Detailed inventory is retained privately in `reports/master_only_20261103/public_inventory.json`.

## Public source and reproduction boundaries

The prepared history-free export is the preferred basis for a separately approved public repository with a single master branch. Keep this development repository and its experiment archive private. This avoids publishing historical manuscripts and unreviewed third-party assets without rewriting or discarding the research history. No new repository, public visibility change or history rewrite is performed by this cleanup.

Datasets, checkpoints and permitted derived evaluation material remain available upon request as documented in [data and run requirements](data_and_runs.md); raw author-collected videos are not shared. External code and weights require their own acquisition terms. Reproducing the exact final tables also requires the frozen sources, selected weights, fixed data partitions and successor receipts; preparing the bundled base DAG alone does not reproduce all later corrections. CPU/package checks do not assert a fresh full-GPU replication.

Manuscripts, frozen experiment code and existing results are preserved by branch cleanup. Check the hosted workflow status for the actual final commit before calling that commit CI-passing.
