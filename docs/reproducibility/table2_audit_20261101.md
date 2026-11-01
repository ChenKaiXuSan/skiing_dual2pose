# Table 2 implementation audit — 2026-11-01

## Current disposition

This document preserves the pre-repair audit below. The identified Unity issue has since been corrected and all 18 tasks in `unity_geometry_v2_20261101T062040Z` completed, including 63 geometric C07 cells and both MetaPose timing scopes. The [repair record](../experiments/unity_geometry_repair_20261101.md) supersedes the pending-action statements below. Archived outputs and their source snapshots remain unchanged.

## Original finding: Unity camera intrinsics

Both Unity capture scripts set `fy = H / (2*tan(vertical_fov/2))`, then incorrectly
set `fx = fy * W/H` in `WriteCameraIntrinsics`. Unity's
[Camera.fieldOfView is vertical](https://docs.unity3d.com/cn/6000.0/ScriptReference/Camera-fieldOfView.html);
for the inspected render cameras, the aspect ratio is already accounted for in
the projection. The extra horizontal aspect multiplier is wrong. In addition,
2D annotations use viewport coordinates multiplied by `W-1` and `H-1`.

Affected code:

- `unity/{female,male}/Assets/Scripts/OneCameraCaptureFrame.cs`,
  `WriteCameraIntrinsics` and the `WorldToViewportPoint` annotation export.
- `dual2pose/eval/main_baseline_geometry.py`, `unity_projection_matrix`, trusts
  the exported focal lengths.
- `dual2pose/eval/evaluate_occluded_geometry.py`, `camera_matrices`, constructs
  another intrinsic matrix directly from the same fields. A future repair must
  keep that matrix consistent with the projection used for camera perturbations.

A diagnostic confined to 16 fixed training camera pairs, five frames per pair,
compared known world-space GT joints projected into each camera with the actual
exported 2D GT. No validation/test data or model scores were used to tune a fix.

| Training-only diagnostic | Existing intrinsics | Derived annotation-consistent intrinsics |
| --- | ---: | ---: |
| Mean Unity reprojection error, px | 70.703827 | 0.00004693 |
| Maximum Unity reprojection error, px | 553.287325 | 0.00015126 |
| Mean triangulated GT-2D error, mm | 186.893778 | 0.00044320 |
| Maximum triangulated GT-2D error, mm | 902.169973 | 0.00566961 |

For this exporter, the independent diagnostic derives
`fx_effective = stored_fy * (W-1)/W` and
`fy_effective = stored_fy * (H-1)/H`, leaving the principal point unchanged.
This is derived from the renderer/annotation conventions, not fitted to GT.
Ski's corresponding training check has mean reprojection error 0.00011571 px.
These are diagnostic geometry errors, **not new Table 2 baseline scores**.
The production code and saved data have not been changed by this audit.

## Reruns required by the original audit (now completed)

| Scope | Required action |
| --- | --- |
| Unity DLT and reprojection-gated DLT | Correct the versioned geometry interface, regenerate inputs and rerun complete evaluation |
| Unity DLT-residual | Regenerate train/val/test geometry, train one new configuration, select with the existing validation rule, then evaluate |
| Unity native-15 auxiliary results | Regenerate the corresponding three geometry rows from the corrected run |
| C07 calibration / image occlusion | Rerun 63 geometric cells: three methods × seven conditions × three angles, using the corrected K consistently |
| Ski geometry | No mandatory rerun established by this finding |
| Pose-only methods, external adaptations and RGB frontend predictions | No mandatory rerun established by this finding |

The C07 matrix displays 105 cells, but the pose-only rows reuse angle-independent
results. The 63 geometry cells are the affected subset, not 63 trainings.
Reuse detector predictions and unchanged pose inputs; retain all old caches and
results under their original identities. Future corrected inputs must have new
provenance and result directories. This audit does not start the reruns.

The previously reported Unity calibrated-baseline ranking, including
DLT-residual versus CanonFuse3D, is not validated until these reruns finish.
Earlier synthetic-camera tests tested the triangulator, not the actual Unity
exporter. This finding supersedes earlier broad statements that all geometry
implementations were verified.

## Other methods and protocol boundaries

The review traced the loaders, training runners, evaluator paths, saved metadata
and manuscript baseline descriptions. No further concrete implementation error
was established for the methods below; this is not a guarantee of bug-free code.

| Methods | Contract checked / qualification |
| --- | --- |
| Left, right, canonical average, aligned average, quality weighting | Per-frame construction, common-13 scoring, input-only alignment/weights; no output alignment to GT for direct MPJPE |
| TCN, SmoothNet | Stated residual/input architecture, loss, budget and selection; Unity validation selection, fixed Ski budget |
| CanonFuse3D | Per-frame loader, direct output scoring, saved checkpoint; prior Sim3 fixes do not by themselves mandate retraining based on the previous train-only impact check |
| DeciWatch | Official model/loss with documented 31-token adaptation, repeated final frame and observed-mask handling; not an unmodified original benchmark protocol |
| STRIDE | Official prior with per-window reset and input-only test-time adaptation; not the original mesh demo or an additional 3D-supervised training run |
| MetaPose | Official learned solver with adapted observations, coordinate adapter and stability recipe; not an unchanged original-paper reproduction |

Unity canonicalization uses native 15 landmarks before selecting common-13;
Ski uses its 13 landmarks with face-sign correction disabled. PA-MPJPE fits a
separate per-frame proper similarity transform; primary MPJPE scores model output
directly. Position metrics are mm and acceleration is mm/sampled-frame².
Ski uses a fixed training budget because its configured validation stream is not
an independent held-out set. These are fixed-partition single runs, not a
multi-seed/cross-fold significance study.

MetaPose's training-only uncertainty fit uses more training frames than its
small optimization pilot; describe those two budgets separately. No test leakage
or mandatory retraining was established from that documentation distinction.

## Validation and publication differences at the original audit

- Main test suite: 678 passed, 6 skipped, 24 subtests. The isolated official
  MetaPose CPU suite passed three tests. GPU-only coverage is not claimed.
- Current-paper provenance: 28 rows checked against their archived sources;
  saved numerical consistency does not resolve the new intrinsics issue.
- The paper intentionally contains mostly older numbers; the Pegasus rerun is
  not byte-identical to those independent trainings. Do not silently mix runs.
- New MetaPose Unity efficiency needs two timing scopes. The Pocket3 laboratory
  qualitative panel uses a different frame/window than the manuscript and needs
  regeneration if the new run is adopted; it does not require retraining.

The manuscript, frozen experiment sources, saved results and master branch were
not changed. Detailed probes and verification logs are in the workspace report
`reports/table2_followup_audit_20261101/`.
