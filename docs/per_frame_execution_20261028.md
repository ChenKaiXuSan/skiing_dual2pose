# Execution ledger — plan: docs/per_frame_rerun_plan_20261028.md

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
