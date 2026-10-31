# Pegasus recovery and Unity storage diagnosis

Constraints: HP260146 / gen_S; retain per-frame common13 protocol, original budgets/seeds, position metrics in mm; no changes to paper, master, prior outputs or running job 197033.nqsv.

1. Fix archived geometry manifest `index` and `source_index` resolution using the existing input-only path mapper; preserve immutable manifest bytes and hashes. Test direct and relocated inputs, plus tampered provenance rejection.
2. Diagnose Unity in a separately allocated node. Compare original and raw-cache items bitwise on all splits, validate full cache hashes/index/order, measure configured DataLoader throughput and isolated GPU compute. Raw cache contains native poses before canonicalization; batching and model preprocessing remain unchanged.
3. If checks pass, launch a fresh same-seed full-budget Unity run in a new successor namespace. Keep the old running job untouched. Its current ~40-minute epochs cannot complete100epochs within the24-hour queue limit. No unsupported checkpoint continuation or partial-budget final result.
4. Build successor DAG with fresh pending outputs and copied immutable receipts for dependency-closed, validated completed jobs. Recheck predecessor receipt/artifact hashes and staged logs. Share completed outputs only through leaf links. Stop predecessor submissions only when the successor is ready; do not cancel its active worker.
5. Rerun failed geometry jobs, then automatically release DLT residual and other dependents. Dispatcher validates receipts, artifact hashes and staged stderr before release; up to64 active requests, subject to scheduler capacity. Preserve failure evidence.
6. MetaPose Unity comparison: identical input manifests and evaluation controls; correct validation checkpoint selection. Final validation2D MSE improves despite worse3D. Fixed train/val inference identifies learned depth-direction drift, not a coordinate conversion defect. Retain the existing successful result; do not silently select a better test result or retune the recipe as part of failure recovery. Record separate diagnosis.
7. Verify fresh run startup, progress/epoch duration, geometry results, controller health and dependency release before reporting restart complete. Keep measured timing separate from projections.

## Verified evidence

- Independent Pegasus diagnostic job198623.nqsv completed on H100 in150seconds, without touching197033.nqsv. All24 real samples matched bitwise across train/val/test; all raw-cache file hashes and fullmetadataorder validated.
- Median original warm item read29.74ms, cold107.19ms; verified cache0.05794ms. Configured16worker DataLoader read65536 cached trainitems in1.178s and valitems in1.065s (bounded throughput includingworkerstartup/shutdown, not wholeepoch timing).
- Synthetic shape-matched H100 training step0.3768s, validation0.1451s;47train+32val compute-only estimate22.35s/epoch. Real Lightning epoch time remains to be measured after restart.
- Recovery import review additionally checks supporting-file hashes and successor symlink destinations, not only original checkpoint-only receipts. Imported completed geometry recreates validated immutableinput aliases before pending residualjobs.
