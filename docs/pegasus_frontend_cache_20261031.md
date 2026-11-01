# Pegasus frontend adaptation cache recovery, 2026-10-31


Status update, 2026-11-01: all seven replacement trainings and their dependent evaluations completed; the containing experiment chain reached 220/220. The rollout observations below are historical. See [reproduction inventory](reproducibility/README.md) for coverage and preservation boundaries.
This recovery preserves the per-frame/common13 protocol, source Unity checkpoint, seed 42, batch 4096, workers 16, 20 adaptation epochs, learning rate 1e-4, optimizer/scheduler policy, source mixture, and validation checkpoint selection. It changes storage only. No paper edits or overwrites of completed experiment outputs are part of this recovery.

Runtime base: `/work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime` (physical prefix `/work/1/HP260146/chenkaixu`). Account/queue: HP260146 / gen_S.

## Diagnosis and validation

The original `FrontEndPoseDataset` first loads full Unity GT and both SAM3D streams, then replaces the SAM3D streams with frontend predictions. Every sample also reopens and decodes frontend prediction files. Balanced mixed adaptation repeats the base population for four sources.

Independent diagnostic 199948.nqsv checked 24 real train/val items across three frontends. All batch dictionaries matched exactly when substituting verified raw Unity storage. Median original sample cost: 122 numpy file loads, 46.33ms cold, 37.32ms warm. Verified raw Unity base plus original frontend loading:2 file loads and 1.503ms.

Frontend caches store exact raw `FrontEndManifest.load_pose` outputs before batch canonicalization. Packing decodes each unique source stream once, preserves repeated frame IDs and joint selection, and refuses existing output paths. Runtime checks source/implementation hashes, full array hashes, shapes, dtypes and finite values. Mixed source order and inherited loader shuffle/drop_last/worker policy are unchanged. Adaptation cache setup accesses train/val only.

Three independent build jobs 199964/199965/199966 created all six caches under `cache/frontend_aligned_20261031_v1`. Each split checked 24 evenly spaced examples against the original frontend loader; 144/144 matched bitwise. Each frontend has 193320 train items and 128880 validation items.

GPU diagnostic 199972.nqsv completed all seven variants on an isolated H100: two real training batches, two validation batches, finite losses/gradients, and best/last checkpoint saves. Configurations retained batch 4096 and workers 16. It invoked the actual adaptation entrypoint with bounded Trainer limits. These pilot outputs are explicitly not publication results; their shortened cosine schedule does not demonstrate full-budget trajectory equivalence. Measured cached item medians for the first two variants were approximately 0.047ms. Full-epoch timing must come from the formal replacement runs.

Local focused checks: 91 passed. The final deployed source passed 70 cache/entrypoint/recovery/dispatcher tests before successor preparation. A main-entrypoint NameError introduced during refactoring was caught by independent review, reproduced by a regression test, and fixed before deployment.

## Isolated successor

Active plan: `plans/frontend_recovery_20261031_v2`.
Frozen deployment stage: `recovery_staging/frontend_20261031_v3`.
Cache implementation and adaptation entrypoint bytes are identical to the GPU-tested first stage. Later stage changes affect recovery output ownership only.

The successor preserves 184 completed tasks, including the completed100-epoch Unity model, and replaces seven frontend training roots. Their 28 dependent evaluations and final coverage task retain their dependencies. Fresh formal outputs use `results/frontend/training/` below the successor plan. The old seven partial runs remain below `plans/recovery_20261031_v2`.

Selected Unity checkpoint SHA256: `f1d1ac529960e605d931aa8664722db00a6f7c2e935e895192c1b3cc66b548d4`.

Recovery now verifies multi-generation receipt/log/artifact provenance. Legacy predecessor omissions required two bounded repairs:

- Three missing Ski reference links are imported directly from the original execution only after all four original records match the completed reference collector's receipt-pinned copies. The evidence and exact source/destination mapping are recorded. No predecessor output is modified.
- Main comparison imports include JSON, CSV and Markdown companions, without importing unrelated siblings in the shared external-results directory.

The unpublished partial `plans/frontend_recovery_20261031_v1` and failed preparation logs are preserved. A read-only validation of all 184 imported jobs identified the table companions as the sole remaining omission after reference repair; the other183 passed.

Diagnostics and deployment records: `diagnostics/frontend_cache_20261031_v1`. Local snapshots: `logs/pegasus_chain_20261030/frontend_*.json`. The original retained slow Unity job197033.nqsv is separate and must not be cancelled as part of frontend recovery.

## Handoff status

Successor preparation completed and all 184 imported jobs revalidated. The old dispatcher acknowledged submission-only STOP; the new controller runs in remote tmux `canonfuse-frontend-20261031`, max 64 in-flight requests, 30-second checks. It checks worker receipts, artifacts and staged scheduler errors before releasing dependent work.

All seven formal jobs were accepted (199987–199993.nqsv), each on a separate node. After the first three replacements completed full epochs without errors, the seven old requests 198739–198745.nqsv were deleted, with successful scheduler responses recorded in `superseded_frontend_cancellation.json`. Their output directories remain intact. Original Unity197033.nqsv remains running with its separate read-only monitor.

All six single-frontend runs reproduced their original first-epoch train loss and validation common13 MPJPE exactly (absolute differences 0). Mixed completed its first full train/validation epoch without errors; the slow predecessor had not completed an epoch for comparison. TensorBoard wall times show approximately 19 seconds/epoch for heads-only and 29–31 seconds/epoch for full single-source adaptation. The approximately 800x sample-read improvement is not an 800x whole-training claim.

At the first complete rollout check, current epochs were VideoPose3D18/12, PoseFormer18/8, MotionBERT11/6 (heads/full), and mixed1, all out of 20. Seven replacement nodes plus the separately retained original Unity node were RUN; no failed/unknown/blocked jobs or new fatal logs. The remaining 28 evaluations and final coverage are still dependent work, not yet claimed complete.


## Observed dependency release

VideoPose3D heads-only199987.nqsv and PoseFormer heads-only199989.nqsv finished all 20 epochs. The dispatcher verified their receipts/checkpoints and staged logs, then automatically submitted 8 dependent evaluations 200016–200023.nqsv. The snapshot had 186/220 complete, 0 failed/unknown/blocked, five replacement training jobs RUN and eight downstream evaluations submitted. This confirms actual continuation, not only a prepared dependency graph. The rest of the chain remains in progress.

Evidence snapshots include `frontend_live_status.json`, `frontend_nodes.json`, `frontend_epoch_timing.json`, `frontend_pilot_completion.json`, `frontend_recovery_preflight_v3.json` and `frontend_cancellation.json` under local `logs/pegasus_chain_20261030/`.
