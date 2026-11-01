# Unity geometry repair and additive rerun — 2026-11-01

The Unity focal-length/viewport mismatch identified in the Table 2 follow-up
has been corrected in the active Python projection interface and both Unity C#
capture scripts. Existing camera JSON files are read without modification.
Unversioned exports must match the known legacy focal-length ratio; unknown
formats fail closed. New exports carry `unity_viewport_w_minus_1_v2` and are not
corrected twice. Occlusion camera perturbations use the same intrinsic helper.
Unity geometry caches without the new convention marker are rejected by the
training/evaluation loader.

## Evidence before cluster training

- Reproduced five failing projection/format tests and one failing old-cache
  rejection test before implementing the repair.
- Real training-only probe: 16 camera pairs, five frames per pair. Mean
  reprojection error 0.00004693 px; mean GT-2D triangulation error 0.00044320 mm.
- Full local suite: 686 passed, six skipped, 24 subtests; GPU/TF-specific skips
  remain explicit. No Unity Editor compilation was performed on this host.
- Each cluster run first repeats the training-only geometry check and executes
  an actual CUDA forward/backward operation before releasing downstream jobs.

## Rerun contract

Use account `HP260146`, queue `gen_S`, a new frozen source and new output paths.
The plan is named `unity_geometry_v2_20261101T062040Z` in the runtime `plans/`
directory. It contains 18 dependency-controlled tasks, one scientific training
configuration and a maximum of eight concurrent scheduler requests. Submission
and current status are recorded separately in the runtime dispatcher state.

1. Train-only interface/CUDA gate.
2. Independent train, val and test geometry exports with byte/shape/coverage
   validation. Cached SAM3D estimates and original input ordering are reused.
3. DLT and reprojection-gated DLT complete-test evaluation.
4. DLT-residual: 100 epochs, seed 42, unchanged batch/loss/optimizer and validation
   minimum selection; the corrected geometry inputs are the intended change.
5. Common-13 and native-15 direct/PA/acceleration rescoring for all three methods.
6. Seven independent C07 conditions, followed by coverage-checked merge: 63
   geometry cells plus the unchanged pose-control evaluations, 105 displayed
   cells in total. Each condition has its own directory and completion receipt.
7. Stable MetaPose Unity latency in both solver-only and adapter-inclusive
   scopes, with TF32 disabled, deterministic operations and saved-prediction
   reproduction checks before timing.
8. Final dependency/artifact validation and mm result export.

The dispatcher checks result hashes and task logs before releasing successors;
failed or unknown dependencies block dependent jobs. A queued plan is not a
completed experiment. Outputs and source snapshots from prior runs are retained.
No manuscript edits, master changes or public release are part of this rerun.

Local preparation, failure/pass logs, source archive hash and deployment records
are in `reports/unity_geometry_fix_20261101/`. Earlier source-review archives
remain dated snapshots containing the pre-repair implementation.


## Completed result and integration record

All 18 tasks completed on 2026-11-01; 86 result artifacts and 332 frozen-source hashes passed the final audit. The three Unity geometric methods retain per-frame/common-13 direct scoring, with PA reported separately:

| Method | MPJPE (mm) | PA-MPJPE (mm) | Acceleration (mm/sampled-frame²) |
| --- | ---: | ---: | ---: |
| DLT | 114.96 | 64.47 | 122.28 |
| Reprojection-gated DLT | 103.55 | 63.30 | 88.42 |
| DLT-residual | 45.44 | 33.44 | 41.88 |

DLT-residual completed 100 epochs, seed 42, and selected epoch 96 (zero-based 95) by validation common-13 MPJPE. Seven C07 conditions completed 77 distinct computations and 105 displayed cells, including camera-independent control reuse. Native-15 auxiliary scoring also completed. New stable MetaPose Unity timing is 2.915 ms for the solver and 10.070 ms including adapters per 30-frame input on H100 PCIe, using 20 warmups and 100 repeats. It is not a hardware-matched ranking against older A6000 measurements.

The paper was updated in a separately authorized synchronization; that operation did not replace unrelated benchmark runs. Source/output evidence is under this plan's `final/`, `results/`, receipts and `source_hashes.json`. Local completion evidence is in `reports/unity_geometry_fix_20261101/status_20261101T1054Z/`.

A later pre-commit review found that future Unity metadata export could run before `cam.aspect` is initialized. Both C# exporters now derive the focal length from `captureWidth / captureHeight`, which is the aspect explicitly used for rendering. A source-level numerical regression covers mismatched initial camera aspect in both exporters; Unity Editor execution is still unverified. This consolidation-only adjustment does not change the Python correction or the existing input data used by the completed run. Its original frozen C# files are retained as evidence.

Geometry cache provenance distinguishes projection versions but does not hash every source camera/estimated-2D file. If these inputs change within the same dataset path, generate a fresh geometry cache in a new output directory. Do not relabel existing results as a new run.

Consolidation checks: 688 tests passed, six skipped and 24 subtests passed. The additional capture-aspect regression first failed for both exporters and passed after the correction. Independent review found no remaining blocking issue within this change scope.
