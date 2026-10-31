# Pegasus recovery status, 2026-10-31

The full successor DAG is running under HP260146 / gen_S in
`/work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime/plans/recovery_20261031_v2`.
Its frozen source is the deployment-time copy of the `per-frame-canonicalization` worktree; source hashes are recorded. Later branch commits do not update that deployed copy. Existing results and paper files were not modified. Master remains `f3ab2bc8d7f43097495e62c504458fa9c34195b0`. Running-job statements below describe the recovery-launch snapshot, not a live status query.

The successor imported67 completed tasks with receipt, artifact, supporting-file and log checks. All pending outputs are fresh. The predecessor controller stopped new submissions only. Original Unity job197033.nqsv continues untouched and has a separate read-only completion monitor at `dispatcher/retained_worker.json`. The successor's main Unity job198626.nqsv starts from seed42 and completes the original100epochs/batch4096/T30/workers16 budget; all later dependencies use this fresh full-budget result.

## Completed repair and observed continuation

- Geometry manifest `index` and `source_index` now use input-only relocation. Original manifests are unmodified and their SHA-256 provenance remains checked.
- Both geometry reruns completed: Unity198627.nqsv (~65s), Ski198628.nqsv (~32s).
- Ski DLT residual198644.nqsv automatically followed its parent and completed (~24s).
- Unity DLT residual198645.nqsv automatically followed its parent and was running at the last snapshot.
- The dispatcher uses30s polling and at most64 simultaneous user requests. It validates task receipts and final scheduler logs before releasing dependents. At the snapshot there were no new failed/unknown tasks or fatal log signatures.

## Unity performance and equivalence

Isolated diagnostic198623.nqsv completed on H100. Full cache hashes, index provenance and metadata order passed;24 sampled real items across train/val/test matched every tensor, frame ID and metadata field bitwise. Cached native15-joint poses remain before canonicalization. Inherited DataLoader shuffle/drop_last/worker seeding and trainer per-frame canonicalization are unchanged.

Original warm sample reads took29.74ms median (cold107.19ms), versus0.05794ms from rawcache. A bounded65536-item cache loader benchmark took1.178s(train)/1.065s(val). Synthetic GPU compute estimate was22.35s/epoch.

Actual new full epochs0,1,2 completed with28s checkpoint spacing, versus the predecessor's ~39.8minutes/epoch. All six corresponding train/loss_epoch and val/mpjpe values matched the predecessor exactly. This supports a storage bottleneck diagnosis and unchanged early optimization trajectory; it does not promise bitwise equality of the entire future training run.

At this initial rate the100-epoch main Unity fit is roughly47minutes, plus startup/test/validation. This is not an ETA for the full220-task graph.

## MetaPose

See `metapose_unity_diagnosis_20261031.md`. New learned depth correlates negatively with input depth from stage1 onward, despite lower validation2D loss. Input manifests, coordinate/evaluation controls and checkpoint selection checks pass. Preserve the existing successful result; no test-driven checkpoint change, depth flip, or silent recipe retuning was performed.

## Verification and evidence

- Local targeted regression suite:86 passed. Pegasus targeted suite:58 passed. CPU-only test environments report a nonfatal NVML warning; actual allocated H100 training succeeds.
- Real-data cache equality and isolated GPU benchmark passed; initial full-training history matches the predecessor exactly.
- Deployment archive SHA-256: `6664248b239ea3ce90c3fd40ded68388194c7d6b3a55ec8185edf63e5f398c1d`.
- Local live evidence: `logs/pegasus_chain_20261030/{recovery_metrics.json,recovery_launch.json,unity_diagnostic_final.json,metapose_trainval_comparison.json}`.
- Remote control: tmux `canonfuse-recovery-20261031` on pegasus03; successor `dispatcher/state.json`, `events.jsonl`, `console.log`, and `retained_worker.json`.
- Failed preparation v1 evidence remains under runtime/recovery_staging/20261031_v1. It refused an unrecognized legitimate STRIDE input alias before any task submission. v2 only accepts leaf aliases into verified outputs owned by the same completed job or its completed transitive dependencies.

## Pre-commit review follow-up

A local regression reproduced an output-ownership omission when recovering
after `core_ski_reference` completes but before its downstream reference
collector runs. Recovery must preserve all four reference records
(`left_canonical`, `right_canonical`, `canonical_avg`, and `canonfuse3d`),
not only the learned fusion record. The reviewed recovery code now imports
and hashes all four, with a regression test covering their availability in
the successor. This change has not been deployed to the running successor.

Recovery currently supports a predecessor with its own completed-task logs
and output ownership. Using an already recovered successor as the source of
another recovery is unsupported: inherited tasks lack local scheduler log
records and link outside that source root. Such attempts fail validation;
recursive provenance import requires a separate implementation before use.
