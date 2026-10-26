# Calibrated geometry implementation audit

Date: 2026-09-06

## Scope and provenance

`dual2pose/eval/main_baseline_geometry.py` implements the additional-input
geometry references for the IVC baseline extension. Its Unity camera matrix,
linear DLT, projection, and reprojection gate are derived from these functions
in the read-only Array submission reference
`scripts/unity/evaluate_large_unity_benchmark.py`:

- `camera_projection_matrix`
- `triangulate_point` / `triangulate_sequence`
- `project_points` / `reprojection_residuals`
- `reprojection_gated_triangulate_sequence`

The local code batches those operations, supports per-frame PTZ calibration,
retains invalid points explicitly, and adds a deterministic temporal fill for
reprojection-gated outputs. The DLT-residual MLP inputs are a controlled local
baseline. They are not an implementation of an official Iskakov model.

## Data and coordinate audit

Unity estimated 2D comes from the exact SAM3D files selected by the IVC cache
frame IDs. The joint order is the repository's 15-joint
`FILTERED_KPTS_MAPPING`: left/right eye, left/right upper arm, left/right lower
arm, left/right thigh, left/right calf, left/right foot, right/left hand, neck.
Static camera calibration is loaded from each person's camera directory. The
projection matrix exactly follows the Array implementation:

`P = K @ diag(1, -1, -1, 1)[:3, :4] @ t_world_cam_4x4`.

Unity DLT therefore has the same metric world coordinate source as the Unity
target before the controller independently canonicalizes each stream. No GT
2D or fitted alignment is used.

Ski-PosePTZ estimated 2D is read from SAM3D NPZ files for the cached frame IDs,
then filtered from the common 15-joint order to the target's 13 joints by
selecting indices 2 through 14. `labels.h5` provides per-frame normalized
intrinsics, camera position, and `R_cam_2_world`. SAM3D detections are in the
256 x 256 cropped-image pixel frame, so the normalized intrinsics are scaled
by `diag(256, 256, 1)`. The documented calibration gives

`P_t = K_px @ [R_cam_2_world.T | -R_cam_2_world.T @ camera_position]`.

The Ski target is not calibrated-world ground truth. It was constructed by
fitting per-camera similarities from H36M 3D on the first 80 calibration frames
to one selected camera and median-fusing the aligned views. Those fitted
transforms were not exported. Consequently there is no GT-free exact mapping
from raw calibrated-world DLT into the pseudo-GT's pre-canonical camera-local
frame. The exporter records Ski DLT in calibrated metric world coordinates;
the controller compares it only after the same independently applied archived
body canonicalization and must label this limitation. It must not fit a test
similarity to the pseudo-GT.

## Exact index and failure policy

The exporter matches every cache record back to the source index by identity:
Unity uses `(person, action, cam1, cam2)` and Ski uses
`(subject, sequence, cam1, cam2)`. It then loads estimated 2D and calibration
for exactly the cached `frames.npy` IDs. Missing estimated 2D or calibration is
an error; the exporter never regenerates a different frame intersection.

Plain DLT leaves nonfinite observations or unsolved homogeneous points as NaN
and writes `dlt_valid.npy`. Robust DLT accepts points whose maximum two-view
reprojection error is at most 25 pixels. Within each sample and joint track,
rejected finite points are linearly interpolated from accepted frames, using
the nearest accepted point outside the accepted time range. If a whole joint
track is rejected but ungated DLT is finite, robust DLT falls back to ungated
DLT and records `robust_fallback.npy`. Truly nonfinite DLT remains NaN and false
in `robust_valid.npy`; it is never silently discarded or filled from target
data.

Each split exports:

- `dlt.npy`, `robust_dlt.npy`: `(N, 30, J, 3)` float32
- `dlt_valid.npy`, `robust_valid.npy`: `(N, 30, J)` boolean
- `gate_accepted.npy`, `reprojection_error_px.npy`
- `robust_interpolated.npy`, `robust_fallback.npy`
- `manifest.json` with shapes, counts, coverage, source/index hashes, geometry
  provenance, and the Ski coordinate limitation

The manifest also records the exporter source SHA256 and the SHA256 of every
output array. A completed export is reused only when dataset, split, shape, gate
threshold, input hashes, exporter hash, and all output hashes match. Stale
partial files or incomplete final file sets raise unless `--overwrite` is
explicit.

## Residual learning interface

`build_dlt_residual_features` has no target argument. For each joint it returns
the complete per-frame DLT pose, normalized estimated 2D from both cameras,
reprojection residual, DLT validity, and joint one-hot identity. Nonfinite
feature values are zero-filled alongside an explicit validity mask. Targets
may only be supplied later by the controller for the authorized train/validation
loss; no target or test-fitted transform enters feature construction.

## Verification evidence

Focused synthetic tests cover calibrated sequence recovery, invalid estimated
2D retention, reprojection rejection/interpolation/all-rejected fallback,
full-pose residual features, and UTF-8-BOM Unity camera JSON. Command:

`/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m unittest tests.test_main_baseline_geometry -v`

An actual Ski test cache smoke export covered all 30 ordered samples and
30 x 13 points per sample. DLT and robust validity were both 1.0; gate
acceptance was 0.9752136752 and interpolation coverage was 0.0247863248. An
actual Unity cached sample loaded as `(30, 15, 2)` estimated 2D with
`(30, 3, 4)` projection matrices and produced `(30, 15, 3)` DLT with complete
validity; its gate acceptance was 0.9644444444.
