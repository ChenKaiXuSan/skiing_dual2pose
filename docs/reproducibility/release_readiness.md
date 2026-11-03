# Release readiness — 2026-11-03

The current change repairs the core distribution and clean-checkout plan preparation. `master` may be fast-forwarded to the validated per-frame branch while preserving both branches and backup refs. This does not make the repository public.

## Corrected for the consolidated source

- Hydra configuration/model YAML is packaged in wheel and sdist; configuration loading is exercised outside the checkout.
- Base-chain recipes are versioned with provenance hashes, with archived recipes retaining precedence. Preparation no longer requires private log manifests and does not submit work.
- CPU CI targets the supported Python 3.11 environment and current test/package entrypoints, including the active branches.
- Final numerical CSVs and checkpoint registry replace the obsolete main/C07 mixed-checkpoint explanation. Legacy observations remain clearly dated.
- A portable core environment, data/weight acquisition contract and source-export command are documented.
- The source export excludes paper, history, Unity assets, vendored frontend code, logs, weights and datasets. The root license does not relicense excluded third-party materials.

## Still separate from code integration

- Public visibility: the existing repository still includes old manuscript/Unity assets and their history. Review permissions and history before making that repository public, or use the clean source export for a separately approved public repository.
- Frontends, datasets and weights: acquire them under their applicable terms. The author has approved the listed derived data and checkpoints being supplied upon request; no raw personal videos are included.
- Exact final-result reproduction: use immutable frozen sources, selected weights, data partitions and successor receipts described in data_and_runs.md. A successful base DAG preparation or CPU test is not a fresh replication of all GPU experiments.
- GitHub CI: local validation and remote workflow results are distinct; inspect the workflow run for the final commit before declaring hosted CI green.

The internal source candidate from 2026-11-01 is historical and superseded by a new allowlisted export. No historical run, checkpoint or manuscript is replaced by this code integration.
