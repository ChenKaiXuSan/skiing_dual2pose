# Pegasus runtime and branch preservation plan

Scope: preserve the current per-frame branch on GitHub, organize this project's Pegasus paths, and prove training/evaluation work in fresh output directories. The user has authorized execution. No master update, force push, manuscript edit, result replacement, or full experimental rerun.

- [ ] Inventory remote queue/account/environment and local branch divergence. Preserve before-state hashes.
- [ ] Retain repository/data locations. Add project-specific runtime directories for environments, fresh runs, setup logs and an index of migration archives/conflicts; do not move unrelated projects or historical files.
- [ ] Add pinned PyTorch and isolated TensorFlow environment specifications, portable path setup, and small real-data train/checkpoint/evaluation checks. Use scheduled GPU allocation only.
- [ ] Run focused local tests; commit the current authorized code changes and setup files on per-frame-canonicalization; fast-forward push only that branch. Verify master hashes unchanged.
- [ ] Synchronize only committed code into Pegasus, preserving differing existing copies. Rebuild runtime in new prefixes and log versions.
- [ ] Submit a bounded GPU smoke job. Validate Unity, Ski, Pocket3 and MetaPose training/evaluation where dependencies are available; distinguish cached-pose pipeline checks from RGB frontend coverage.
- [ ] Verify completed job artifacts, errors, paper fingerprints and branch identities. Document any environment or scheduling blockers honestly.

Decisions: keep published experiment snapshots untouched; use explicit runtime environment variables and fresh manifests, not rewrites of historical manifests. Store new setup artifacts under logs locally and a project-specific runtime directory remotely. Keep paper and generated review artifacts outside the Git commit.

Scheduler correction: the user explicitly specified account HP260146 and queue gen_S. Only use this combination for subsequent compute jobs. The failed gpu/gpu_S probes and prior SKIING diagnostic are recorded in setup logs.
