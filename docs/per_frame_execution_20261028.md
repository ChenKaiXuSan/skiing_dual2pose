# Execution ledger — plan: docs/per_frame_rerun_plan_20261028.md

> Historical snapshot: branch names and preservation instructions below apply to the recorded operation. Since 2026-11-03, only `master` is retained; see `docs/branch_management.md` for the current policy.

User authorized all method/Pocket3 repairs, verification, then training on 2026-10-28.

- Workspace: existing dirty `per-frame-canonicalization` branch; preserve prior user/session edits, do not move to clean worktree and lose them. No commit requested.
- Ruling: independent file ownership uses dispatching-parallel-agents workflow; parent integrates shared contracts and final validation. No per-task reviewer fanout.
- Ruling: Ski retains paper splits, fixed final checkpoint; no new validation split or sixfold study.
- Ruling: historical result files/checkpoints remain untouched. Active methods migrate; tagged metadata must describe real transformations.
- Ruling: formal training cannot start until integration tests, per-method smoke checks and final review pass. Launch main Unity/Ski first, preserve output isolation and active GPU jobs.

## Tasks

- [x] Pocket3 mapping, reliability masking, split validation, common main flow.
- [x] Local baselines, geometry, gate/C07 per-frame and common13 selection.
- [x] STRIDE, DeciWatch, MetaPose per-frame coordinate adapters and metadata.
- [x] Ski fixed selection, synchronization descriptor, angle statistics, front-end orchestration.
- [x] Full suite, runtime smoke/data preflight, independent review.
- [x] Launch approved main training and verify active computation, Ski epoch progress and artifact paths. Unity first full validation remains pending.

## Interface coordination

- Pocket task owns trainer reliability masks. Parent must not edit that file concurrently.
- Local task owns `canonical_batch`, switches active default to per_frame; external task uses same shared API.
- Parent owns execution configs, Ski entrypoint, synchronization/angle modules and run scheduling.

## Verified integration (2026-10-28)

- Final scoped regression: **475 passed, 1 skipped, 24 subtests passed**, with optional DeciWatch repo enabled. One skip is an optional external dependency. Bare repository-wide pytest discovers upstream cache scripts with import-time argparse; use `python -m pytest tests -q -p no:cacheprovider`.
- Independent review: 43 focused tests passed; no P0/P1 primary-training blocker. Fixed all reported P2 items: current-provenance lookup in fusion profiling, explicit current-run external profiling, and stale Unity eval checkpoint fallback.
- Unity: real raw train cache, batch4096, finite loss/backward, optimizer update and exact save/load prediction roundtrip; peak allocated GPU memory 41,241,134,592 bytes. Actual data-module CLI 1 train/val/test batch completed with best-checkpoint reload; original split coverage train193320/val128880/test64440.
- Ski: actual CLI2epochs, one train batch each, one final-test batch; no validation; final epoch1 checkpoint loaded. Fixed Lightning/PyTorch `weights_only` compatibility exposed by this end-to-end check. Paper split unchanged.
- Local adapters: synthetic native15/13 TCN/SmoothNet/DLT train-save-load-score; gate variants, geometry and direct-score translation regression passed.
- Frontend adaptation: head-only/full update and exact save/load on real raw train windows passed. This checks adaptation mechanics, not accuracy on alternate frontend inputs.
- External: actual DeciWatch GPU forward/update and cache regeneration; official STRIDE full30step TTA on one real train window; MetaPose30 frozen native Ski predictions hash-verified and rescored. These are preflight artifacts, not newly completed paper experiments.
- Pocket3 real-data smoke: native15/native13 finite gradients, one update, exact checkpoint-score roundtrip; one30-frame window and390 reliable common13 points. Fresh availability:17 full-record nonoverlapping windows; forward train/val/test42/3/5, reverse40/3/2. Historic windows/counts must not be copied into new results. Evidence under `logs/per_frame_common13_v1/pocket3/real_data_smoke/20261028T053313.360143Z/`.

## Authorized formal launch

- Root: `/home/kaixu_chen/skiing/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/per_frame_common13_v1/main_20261028T053836Z`
- tmux: `perframe_unity_20261028` (physical GPU0), `perframe_ski_20261028` (physical GPU1).
- Unity: fold0, seed42,100epochs,batch4096,16workers, common13 validation selection, best checkpoint final test.
- Ski: paper fixed train/test, seed42,100epochs,batch4,4workers, predeclared last epoch99 final test.
- Pre-launch GPU check: no compute processes;15MiB/272MiB allocated on49,140MiB devices. Old results/checkpoints unchanged.
- Root contains source hashes, code snapshot, dirty-tree patch, verification logs, launch_manifest.json and independent dataset status JSON. A zero process return code is followed by checkpoint/metric artifact verification.
- Downstream baselines/ablations/Pocket/adaptation are not launched yet; they follow the approved dependency order after primary checkpoints are frozen.

- Launch observation: {"unity": {"csv_rows": 0, "last_logged_epoch": null, "last_logged_step": null, "traceback_or_oom": false}, "ski": {"csv_rows": 116, "last_logged_epoch": "47", "last_logged_step": "1727", "traceback_or_oom": false}}; Unity GPU at100% utilization,47,583MiB reserved in the observed training interval. Formal runs are still active; no final research result is claimed.

## Progress audit and GPU1 continuation (2026-10-28)

- Ski finished100epochs/3600steps and30test sequences (900frames/11700common13points). Checkpoint `epoch=100` is Lightning on_train_end state; full logged train epochs0..99 and loop completed100 verified. Corrected launcher false alarm; retained original error in status history. Exact reloaded/cache score 0.12508279747901654 agrees with CLI0.1250828803 within1e-6; canonical average 0.19347613975572853.
- Unity validated 5 epochs; latest val MPJPE 0.06763572245836258. Median recent full epoch+validation 158.3sec; ETA is an extrapolation, not guaranteed. Final test still pending.
- GPU1 continuation queue `/home/kaixu_chen/skiing/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/per_frame_common13_v1/core_20261028T055152Z`, tmux `perframe_core_gpu1_20261028`: Ski reference verified, then TCN/SmoothNet bothdatasets100epochs each, DeciWatch bothdatasets70epochs seed4321. Raw cache/index hashes checked; fresh outputs; stop on process/artifact failure. Independent baselines overlap Unity primary training; Unity-checkpoint-dependent studies remain pending.

## Full dependency automation (2026-10-28, client timezone Asia/Shanghai)

User authorized allremaining experiments automatically, acrossbothGPUs, with result/error gates. Implemented and tested as `docs/superpowers/plans/2026-10-28-dependency-gpu-scheduler.md`.

- Workflow: `/home/kaixu_chen/skiing/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/per_frame_common13_v1/auto_20261028T061053Z`
- Persistent session: `perframe_auto_20261028`
- Source,manifest,context,state,events andper-tasklogs/completionchecks stored beneathworkflowroot. Read `state.json` job states as authoritative; `events.jsonl` records dispatch/completion.
- Existing main/core jobs are adopted without termination or duplicate execution. Verified existingSki results remain available. GPU0/1 reserved whiletheir original launchers run; readyDAG tasks automaticallytake releaseddevices.
- Full512tests passed,1optional skip; independentreview passed. FirstCPU prep workers actually executed and verified successfully whilebothGPU training jobs remained active.
- Failure policy: logfullerror, preserveartifacts, blockonlydependenttasks andcontinueindependentwork. Resourceprobe failure preventsnewGPUdispatch. Timingjobs exclusive; neitherresearchscope norfinalfixedbudgets changed.

## Post-commit correctness review (2026-10-28)

- Reviewed `ff04cf0` against the approved per-frame/common13 contract, including independent review of training, Ski selection, Pocket3 mapping/masking/partitions, and external adapters. No additional confirmed primary-training/scoring defect was found within this scope.
- Fixed standalone PA/table entrypoints still requiring retired ordinary `mlp` results. Regression tests reproduced `FileNotFoundError` with otherwise complete approved results before the fix; the renderer now accepts 11 local methods per dataset and reports dynamic counts. The active frozen workflow already filtered `mlp`; its effective PA list, including order, equals the corrected list.
- Fixed DeciWatch GPU-test eligibility: setting `DECIWATCH_REPO` while hiding CUDA previously caused two environment failures. These tests now explicitly require CUDA. This does not replace GPU adapter verification.
- Fresh CPU suite with the optional DeciWatch checkout configured: **511 passed, 4 skipped, 5 warnings, 24 subtests passed**. Three skips require GPU access; one requires the isolated MetaPose environment. Ten distinct scheduled Hydra entrypoints accepted their configurations after checkpoint token resolution.
- Actual Pocket3 cohort audit: forward train/val/test windows42/3/5 and reverse40/3/2; paired right-camera frames do not overlap across blocks. Retained input/GT comparison transforms have matching handedness. Perfect native13 GT converted to the comparison frame agrees within1.79e-7 coordinate units; reliable common13 test points are1950/780.
- Active source snapshots and experiment outputs remain unchanged. Repairs affect standalone postprocessing and test eligibility, not training, model selection, or predictions; no rerun is required for these repairs. Full outstanding GPU experiments still require their normal artifact checks before completion claims.

## Unity completion gate recovery (2026-10-28 10:22 UTC)

- Unity fit completed all 100 epochs / 4,700 optimizer steps and tested validation-best epoch66 on all64,440 windows: primary common13 MPJPE0.0508277379. The installed Lightning callback refreshes `save_last=True` only when top-k changes, leaving historical `last.ckpt` at epoch72 / step3431. This caused the launcher's terminal-file assertion and scheduler gate to fail despite complete training/test logs. The earlier code review did not catch this framework behavior.
- Preserved every existing checkpoint byte and the original errors. The historical Unity `last.ckpt` is **not** a final-epoch resume checkpoint. Downstream work uses verified best checkpoint `66-0.0287-0.0639.ckpt`, SHA256 `9d16594510439169cbf3c10df9dd412b8aebb665936ae691766aeaebfc0a3218`.
- Corrected Unity completion validation to check all train/val epochs, exact per-epoch steps, finite scores, fit completion, final test step, full test coverage, and best-checkpoint selection/weights independently. Unity and frontend entrypoints now explicitly save terminal state after fit, before any test reload. New regression reproduces stale-last rejection and rejects missing epochs, wrong steps, incorrect selection, and incomplete coverage. Full CPU suite:516 passed,4 skipped,24 subtests passed; independent live-artifact verification passed.
- Recovery audit: `logs/per_frame_common13_v1/auto_20261028T061053Z/recovery_20261028T102205Z/`. Backed up scheduler state, source hashes, changed source files, and Unity status; deployed only the scheduler verifier and two terminal-save entrypoints into the active snapshot, with old/new hashes recorded. Restarted only the scheduler; detached STRIDE and synchronization workers survived. Released143 dependency-blocked tasks. No retraining or output overwrite was needed.

## Reporting-unit decision (2026-10-28)

- User explicitly requested all future position-error results be reported in **mm**, consistently across Unity, Ski-PosePTZ, Pocket3, methods and ablations. This supersedes earlier reporting text that left Unity/Ski errors in unnamed dataset units.
- Convert metre-valued MPJPE, PA-MPJPE and other length errors by multiplying by 1000 exactly once. Existing `*_mm` values are already converted; existing `*_m` and raw Unity/Ski length-error values need conversion when reporting. Preserve raw artifacts and document the source unit; do not infer units from numerical magnitude alone.
- Unity reporting adopts the standard scene convention `1 Unity unit = 1 m`. Male/female FBX importer metadata has `globalScale: 1`, `useFileScale: 1`; joint export writes `Transform.position` directly. Unity documents that FBX `fileScale` converts file units to metres: https://docs.unity3d.com/2022.3/Documentation/ScriptReference/ModelImporter-fileScale.html . This is the synthetic scene's metric convention, not a measurement of a real person.
- Ski official local README declares 3D labels in metres; pseudo-GT export preserves the reference-camera scale. Train seq103/subj0 and test seq405/subj5 re-export checks matched existing frames and coordinates within 1.2e-7 and 1.8e-7 respectively. Continue labelling Ski scores as errors against a constructed reference.
- This decision changes reporting units only; no retraining, output recanonicalization, fitted alignment, loss rescaling or checkpoint-selection change is implied. Existing running snapshots and result files have not been migrated by this documentation update.
- Angles, ratios, percentages, runtimes and training losses retain their appropriate units. Acceleration errors require an explicit time basis (`mm/frame^2` or `mm/s^2`); converting frame differences to seconds additionally requires the correct frame interval.

## Evaluation export recovery (2026-10-29, Asia/Shanghai)

- Fixed the Unity gate-visualization import to `dual2pose.map_config` in the branch and the active workflow source snapshot. Full-test canaries then exposed missing per-frame protocol columns in the masking and temporal-offset CSV writers; both headers now retain `canonical_mode`, `evaluation_protocol`, `output_alignment`, and `reduction`. Model predictions, metrics, checkpoints, splits, and alignment remain unchanged.
- Validation: 40 focused regression checks passed. Real masking and temporal-offset canaries each completed all 64,440 test windows and passed the workflow artifact validators. Their numerical JSON results exactly match the original partial results.
- Preserved 778 existing completed-artifact files with SHA-256 verification. Archived all 52 original failures and both intermediate CSV-failure attempts before retrying. After the two canaries passed, requeued the remaining 50 failures and released 19 blocked dependents. Existing running jobs survived scheduler-only restarts.
- Current persistent scheduler session: `perframe_map_import_remaining_20261029`. The scheduler continues both GPUs automatically, while retaining dependency checks, artifact validation, and exclusive timing/profile measurements. Counts in recovery records are snapshots; use the workflow `state.json` for live progress.
- Recovery evidence: `logs/per_frame_common13_v1/auto_20261028T061053Z/recovery_20261029T014505Z_map_import/`. No completed training or evaluation result was overwritten.

## Completion polling race recovery (2026-10-29, Asia/Shanghai)

- The scheduler had incorrectly marked `eval_mask_left_distal_0p25` as disappeared despite its successful completion artifact, blocking the merge and 18 RGB evaluation cells. Root cause: checking completion-file existence before process liveness allowed a publication/exit race during one poll.
- Changed polling to check liveness once before inspecting completion. A deterministic regression reproduced the old failure; all 41 focused checks passed after repair. Patched both branch and frozen workflow snapshot.
- Revalidated the existing full-test artifact and checkpoint hash, reconciled success without recomputing results, and released 19 downstream jobs. Preserved all 159 prior complete job records and verified 1,709 protected files unchanged. All 52 originally failed masking/offset tasks are now validated successes.
- Current scheduler session: `perframe_completion_recovery_20261029`. Evidence: `logs/per_frame_common13_v1/auto_20261028T061053Z/recovery_20261029T034359Z_completion_race/`.
