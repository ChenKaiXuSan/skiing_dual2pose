# Pegasus runtime and branch preservation plan

> Historical snapshot: branch names and preservation instructions below apply to the recorded operation. Since 2026-11-03, only `master` is retained; see `docs/branch_management.md` for the current policy.

Scope: preserve the current per-frame branch on GitHub, organize this project's Pegasus paths, and prove training/evaluation work in fresh output directories. The user has authorized execution. No master update, force push, manuscript edit, result replacement, or full experimental rerun.

- [x] Inventory remote queue/account/environment and local branch divergence. Preserve before-state hashes.
- [x] Retain repository/data locations. Add project-specific runtime directories for environments, fresh runs, setup logs and an index of migration archives/conflicts; do not move unrelated projects or historical files.
- [x] Add pinned PyTorch and isolated TensorFlow environment specifications, portable path setup, and small real-data train/checkpoint/evaluation checks. Use scheduled GPU allocation only.
- [x] Run focused local tests; commit the current authorized code changes and setup files on per-frame-canonicalization; fast-forward push only that branch. Verify master hashes unchanged.
- [x] Synchronize only committed code into Pegasus, preserving differing existing copies. Rebuild runtime in new prefixes and log versions.
- [x] Submit a bounded GPU smoke job. Validate Unity, Ski, Pocket3 and MetaPose training/evaluation where dependencies are available; distinguish cached-pose pipeline checks from RGB frontend coverage.
- [x] Verify completed job artifacts, errors, paper fingerprints and branch identities. Document any environment or scheduling blockers honestly.

Decisions: keep published experiment snapshots untouched; use explicit runtime environment variables and fresh manifests, not rewrites of historical manifests. Store new setup artifacts under logs locally and a project-specific runtime directory remotely. Keep paper and generated review artifacts outside the Git commit.

Scheduler correction: the user explicitly specified account HP260146 and queue gen_S. Only use this combination for subsequent compute jobs. The failed gpu/gpu_S probes and prior SKIING diagnostic are recorded in setup logs.

Completion evidence: docs/pegasus_runtime_20261030.md. GPU job 195112.nqsv passed all five diagnostic steps; Unity/Ski checkpoints record one optimizer step and per_frame mode, and 1,289 manuscript files matched their pre-setup SHA256 values. Full scientific reruns and RGB frontend checks remain outside this setup scope.
