# Per-frame experiment dependency scheduler

User approved automatic execution of all remaining experiments, dependency ordering, two-GPU utilization and per-task verification. This extends the approved `docs/per_frame_rerun_plan_20261028.md`; no new study scope or permissions are needed.

Design: preserve running main/core launchers and reserve their GPUs until they exit. Observe and validate their artifacts. A persistent tmux scheduler owns a DAG of remaining tasks, runs one task per available GPU plus bounded CPU preparation, validates subprocess outcomes and scientific artifact contracts, then releases dependencies. Failures block only descendants. Status and error histories persist atomically; no destructive retry or overwriting research output. Timing jobs require both GPUs otherwise idle to avoid shared-load contamination. No external notifications.

Ruling: existing dirty worktree and running jobs must be preserved in place. No git checkout/commit or active process termination. Ready tasks are prioritized ahead of waiting dependencies. Epoch completion uses optimizer steps/loop counters and CSV history, not only Lightning's ambiguous terminal epoch field.

Shared family interface: `build_jobs(context)->list[dict]`, `validate(job,context)->dict` in family modules. Context keys: repo, root, python, main_root, core_root, raw_cache (absolute strings). Job keys: id, deps (ids), commands (list of argv lists), env (dict), resource (`gpu`, `cpu`, `exclusive_gpu`), family, validation (dict). Output directories are beneath context root. Commands/environment may use `{unity_checkpoint}` / `{ski_checkpoint}` and `{checkpoint:JOB_ID}` resolved from verified upstream metadata. Worker runs commands and validation, writes a completion record only after checks; scheduler does not equate PID exit with success. Family builders must enumerate all authorized experiments and report unavailable assets precisely.

- [x] Training family: geometry and residual2, gate3, frontend7, Pocket12+4 plus frozen eval; manifests and validators.
- [x] Evaluation family: remaining frozen Unity/Ski robustness matrices, sync calibration, N-view, C07/RGB, frontend32 cells; validators.
- [x] External/report family: STRIDE2, MetaPose2, PA/main comparisons, profile/qualitative tasks where executable; validators.
- [x] Engine: external-job adoption, DAG/resource scheduling, atomic state, artifact gate, failure isolation and restart safety; tests first.
- [x] Integrate coverage against approved plan, smoke/dry-run, tests and independent review.
- [x] Start durable scheduler; verify reservations and at least one task dispatch/verification, publish manifest/status/session paths.

## Verification and deployment

- Full suite:512passed,1optional-dependency skip,24subtests. Independent read-only integration review found no high-severity blocker. CLI/schema tests and actual frozen-code CPU workers verified.
- Built179DAGnodes:2main+7existingcore+33training-family+116evaluation-family+21external/report. Scientific fit budget36=2main+6existingcorefits+28remaining; nodes also include preparation, evaluation, reporting.
- Frozen source package/configs plus source hashes; reuse immutable raw caches through declared paths. Checkpoint tokens are resolved only from verified metadata and hashes are rechecked before dependent execution.
- Deployment root: `/home/kaixu_chen/skiing/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/per_frame_common13_v1/auto_20261028T061053Z`; session `perframe_auto_20261028`;10second polling. Original main andcore reservations retained.
- Actual runtime verified frontend_assets,RGBassets andfrontendasset workers completed; no failures or blocked tasks at initial check. Both physicalGPUs remained occupied by their original training jobs.
- Ruling: shared-GPU concurrency is not introduced; one model task per physicalGPU avoids VRAM contention. GPU utilization may dip during I/O. Exclusive timing tasks wait for bothGPUs and CPUprep to be free. The longest independent STRIDE Unity TTA receives priority to reduce the schedule tail.
- Recovery: scheduler may restart via the same run command; worker PID start-time tokens and completion records prevent duplicate dispatch. Failed tasks retain outputs and block only descendants. No destructive automatic retry.
