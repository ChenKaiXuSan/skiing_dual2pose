# Ski-PosePTZ geometry coordinate audit

Date: 2026-09-06

## Verdict

The original calibrated DLT export is geometrically valid but is represented
in the Ski-PosePTZ calibrated world frame. The IVC Ski target is a constructed,
root-relative pose reference. Applying the archived first-frame
canonicalization directly to world DLT retains the skier's global trajectory,
whereas that trajectory is absent from the target. This representation
mismatch produced the diagnostic 8.1975 MPJPE value and must not be reported as
a DLT baseline result.

For Ski only, the defensible GT-free pose conversion is to subtract the
predicted pelvis independently in every frame before applying the unchanged
archived canonicalization:

```python
pelvis = 0.5 * (prediction[..., 4, :] + prediction[..., 5, :])
prediction_pelvis_relative = prediction - pelvis[..., None, :]
```

Indices 4 and 5 are the left/right thighs in the repository's common 13-joint
order. Their midpoint implements the same pelvis convention used by SAM3D and
the Ski H36M annotations. This conversion uses only the DLT prediction from
estimated 2D and supplied calibration. It introduces no target value, fitted
alignment, scale estimate, or test-selected parameter. Raw calibrated-world
DLT must remain preserved as a diagnostic artifact with separate provenance.

Unity must not receive this conversion because its target retains the Unity
world trajectory before first-frame canonicalization.

## Evidence from the actual arrays

For the complete 30-sample Ski test cache:

- DLT predicted-hip-center norm percentiles `(min, p25, p50, p75, p95, max)`
  were `(5.4368, 6.1240, 6.7050, 7.6970, 8.8182, 10.9206)`.
- Target hip-center norm percentiles were
  `(0.00134, 0.00170, 0.00218, 0.00385, 0.00458, 0.00476)`.
- Median consecutive-frame hip displacement was `0.6464` for world DLT and
  `0.000212` for the constructed target.
- Mean pelvis-relative joint radius was `0.4933` for DLT and `0.5126` for the
  target. Median bone-length ratios were mostly `0.78` to `1.04`, apart from
  one shoulder-side ratio of `1.39`. The evidence supports removing global
  translation and does not support fitting scale.
- Per-frame predicted-pelvis subtraction followed by the unchanged archived
  batch canonicalization gave diagnostic test MPJPE `0.398409`, compared with
  `8.1975` when world translation was retained. This diagnostic is used to
  validate the representation change; final reported values must come from
  the controller's preserved, hashed evaluation pipeline.

Reproduce these checks:

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python - <<'PY'
import numpy as np

base = "logs/ivc_mmsports_extension/main_baselines/20260906"
dlt = np.load(f"{base}/geometry/ski/test/dlt.npy", mmap_mode="r")
target = np.load(f"{base}/cache/ski/test/target.npy", mmap_mode="r")

dlt_pelvis = 0.5 * (dlt[:, :, 4] + dlt[:, :, 5])
target_pelvis = 0.5 * (target[:, :, 4] + target[:, :, 5])
percentiles = [0, 25, 50, 75, 95, 100]
print("DLT pelvis norm:", np.percentile(np.linalg.norm(dlt_pelvis, axis=-1), percentiles))
print("target pelvis norm:", np.percentile(np.linalg.norm(target_pelvis, axis=-1), percentiles))
print("DLT pelvis step:", np.percentile(np.linalg.norm(np.diff(dlt_pelvis, axis=1), axis=-1), [0, 50, 95, 100]))
print("target pelvis step:", np.percentile(np.linalg.norm(np.diff(target_pelvis, axis=1), axis=-1), [0, 50, 95, 100]))
print("DLT pelvis-relative radius:", np.linalg.norm(dlt - dlt_pelvis[:, :, None], axis=-1).mean())
print("target pelvis-relative radius:", np.linalg.norm(target - target_pelvis[:, :, None], axis=-1).mean())
PY
```

The archived-canonical diagnostic can be reproduced with:

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python - <<'PY'
import numpy as np
import torch
from dual2pose.trainer.canonicalize import canonicalize_pose_torch

base = "logs/ivc_mmsports_extension/main_baselines/20260906"
dlt = np.asarray(np.load(f"{base}/geometry/ski/test/dlt.npy", mmap_mode="r"))
target = np.asarray(np.load(f"{base}/cache/ski/test/target.npy", mmap_mode="r"))
dlt = dlt - 0.5 * (dlt[:, :, 4:5] + dlt[:, :, 5:6])
errors = []
for start in range(0, len(dlt), 4):
    pred_canonical, _ = canonicalize_pose_torch(
        torch.from_numpy(dlt[start:start + 4].copy()),
        left_hip=4, right_hip=5, neck=12,
    )
    target_canonical, _ = canonicalize_pose_torch(
        torch.from_numpy(target[start:start + 4].copy()),
        left_hip=4, right_hip=5, neck=12,
    )
    errors.append(torch.linalg.norm(pred_canonical - target_canonical, dim=-1).reshape(-1).numpy())
errors = np.concatenate(errors)
print("mean/median/p95:", errors.mean(), np.median(errors), np.percentile(errors, 95))
PY
```

## Root-relative annotation invariant

The dataset's `load_h5_example.py` names H36M joint 0 `hip`, joint 1
`right_up_leg`, and joint 4 `left_up_leg`. Across every original training and
test HDF5 annotation, joint 0 and the midpoint of joints 1 and 4 are exactly
zero. Thus the supplied 3D poses contain body pose relative to the pelvis and
not global skier position.

The six-view pseudo-GT exporter loads those HDF5 3D arrays directly, fits one
fixed per-camera similarity on calibration frames, and median-fuses aligned
camera-local poses. It does not reconstruct or add a world trajectory. The
result remains a constructed camera-local/root-relative reference, with small
fixed offsets introduced by the alignment and median operation.

Reproduce the HDF5 invariant and inspect representative pseudo-GT exports:

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python - <<'PY'
import h5py
import numpy as np

root = "/home/kaixu_chen/skiing/data/Ski-PosePTZ-CameraDataset-png"
for split in ("train", "test"):
    with h5py.File(f"{root}/data/{split}/labels.h5", "r") as handle:
        pose = np.asarray(handle["3D"]).reshape(-1, 17, 3)
    hip = np.linalg.norm(pose[:, 0], axis=-1)
    hip_mid = np.linalg.norm(0.5 * (pose[:, 1] + pose[:, 4]), axis=-1)
    print(split, "joint0 max/median", hip.max(), np.median(hip))
    print(split, "joint1-4 midpoint max/median", hip_mid.max(), np.median(hip_mid))

for relative in (
    "pseudo_gt_exports/train/seq103_subj0/seq103_subj0_cams0-1-2-3-4-5_pseudo_gt.npz",
    "pseudo_gt_exports/test/seq405_subj5/seq405_subj5_cams0-1-2-3-4-5_pseudo_gt.npz",
):
    pose = np.load(f"{root}/{relative}", allow_pickle=True)["poses"]
    hip_mid = np.linalg.norm(0.5 * (pose[:, 1] + pose[:, 4]), axis=-1)
    print(relative, "constructed-reference hip midpoint median/max", np.median(hip_mid), hip_mid.max())
PY
```

## Calibration convention verification

The following was a protocol-only verification using supplied Ski GT 2D and
3D labels. GT labels were not used to generate experiment predictions, fit a
transform, select a threshold, or construct residual-model features.

The documented convention is:

```python
K_px = np.diag([256.0, 256.0, 1.0]) @ K_normalized
R_world_to_camera = R_cam_2_world.T
t_world_to_camera = -R_world_to_camera @ camera_position
P = K_px @ np.column_stack([R_world_to_camera, t_world_to_camera])
```

On 40 frames from subject 5, sequence 405, cameras 0 and 1, triangulating the
supplied GT 2D, converting the result to camera 0, and subtracting H36M root
joint 0 reproduced the supplied camera-local/root-relative 3D coordinates with
mean numerical residual `5.31e-8` and 95th-percentile residual `1.04e-7` in the
stored coordinate units. The alternative non-transposed rotation produced a
mean residual `0.582`. The verified convention therefore requires both the
256-pixel intrinsic scaling and the transpose of `R_cam_2_world`. These values
only establish internal calibration/annotation consistency; they are not a
paper accuracy result for the constructed pseudo-GT benchmark.

Reproduce the protocol check:

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python - <<'PY'
import h5py
import numpy as np
from dual2pose.eval.main_baseline_geometry import ski_projection_matrix, triangulate_sequence

path = "/home/kaixu_chen/skiing/data/Ski-PosePTZ-CameraDataset-png/data/test/labels.h5"
with h5py.File(path, "r") as handle:
    data = {key: np.asarray(handle[key]) for key in (
        "subj", "seq", "cam", "frame", "2D", "3D",
        "cam_intrinsic", "cam_position", "R_cam_2_world",
    )}
lookup = {
    (int(data["subj"][i]), int(data["seq"][i]), int(data["cam"][i]), int(data["frame"][i])): i
    for i in range(len(data["frame"]))
}
errors = []
wrong_rotation_errors = []
frames = sorted(key[3] for key in lookup if key[:3] == (5, 405, 0))[:40]
for frame in frames:
    i0 = lookup[(5, 405, 0, frame)]
    i1 = lookup[(5, 405, 1, frame)]
    p0 = ski_projection_matrix(data["cam_intrinsic"][i0], data["cam_position"][i0], data["R_cam_2_world"][i0])
    p1 = ski_projection_matrix(data["cam_intrinsic"][i1], data["cam_position"][i1], data["R_cam_2_world"][i1])
    uv0 = data["2D"][i0].reshape(17, 2) * 256.0
    uv1 = data["2D"][i1].reshape(17, 2) * 256.0
    world = triangulate_sequence(p0[None], p1[None], uv0[None], uv1[None]).prediction[0]
    center = data["cam_position"][i0]
    rotation = data["R_cam_2_world"][i0]
    local = (rotation.T @ (world - center).T).T
    local -= local[0]
    target = data["3D"][i0].reshape(17, 3)
    target -= target[0]
    errors.append(np.linalg.norm(local - target, axis=-1))
    wrong = (rotation @ (world - center).T).T
    wrong -= wrong[0]
    wrong_rotation_errors.append(np.linalg.norm(wrong - target, axis=-1))

errors = np.concatenate(errors)
wrong_rotation_errors = np.concatenate(wrong_rotation_errors)
print("verified residual mean/median/p95/max:", errors.mean(), np.median(errors), np.percentile(errors, 95), errors.max())
print("non-transposed rotation residual mean/median:", wrong_rotation_errors.mean(), np.median(wrong_rotation_errors))
PY
```

## Reporting rule

The Ski geometry rows must be described as estimated-2D plus supplied
calibration, converted to a per-frame predicted-pelvis-relative pose before
the archived canonicalization. The manuscript must state that the Ski target
is a constructed reference. Raw-world DLT diagnostics and the invalid 8.1975
representation-mismatch value must not appear as method accuracy. The final
DLT, robust DLT, and DLT-residual MLP values must be recomputed from the
preserved full train/test indices after this fixed GT-free conversion.
