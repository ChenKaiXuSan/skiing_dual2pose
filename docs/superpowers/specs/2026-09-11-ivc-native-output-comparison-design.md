# IVC Native-Output Comparison Design

**Date:** 2026-09-11
**Status:** Approved for implementation on 2026-09-11
**Scope:** Add one independent table immediately after the existing main comparison. Preserve the existing canonical-space table and all archived results.

## Purpose

The existing main table measures every method after the archived CanonFuse3D body-coordinate transform. It therefore measures downstream fusion in the canonical output space, not the quality of native front-end or external-method outputs.

The new table will answer a separate question: how accurate are native outputs and pre-canonicalization combinations when only CanonFuse3D is allowed to use its method-internal canonicalization?

## Non-negotiable protocol rules

1. SAM3D, MotionBERT, PoseFormer, VideoPose3D, STRIDE, DeciWatch, MetaPose, DLT, and the pre-canonicalization controls must not call `canonicalize_pose_torch` or reuse a cached canonical tensor.
2. CanonFuse3D retains canonicalization because it is part of the proposed method, not an evaluation-time advantage added to competitors.
3. Test labels may be opened only after predictions and their checksums are fixed. Test results must not select configurations, checkpoints, alignments, or reporting subsets.
4. Existing canonical-space results and tables are immutable. The new outputs use a new run root and separate provenance manifests.
5. Every method is evaluated on the complete fixed test split: 64,440 Unity camera-pair windows and 30 Ski-PTZ-Pose windows, each with 30 frames, on the common 13-joint subset.

## Common MPJPE coordinate protocol

Ordinary MPJPE is reported, but native predictions cannot be subtracted from a reference expressed in a different camera gauge. Coordinate conversion is therefore separated from body canonicalization:

- Express each native prediction and its reference in the method's declared output frame using supplied camera transforms or the method's documented inverse observation transform.
- Subtract the predicted and reference pelvis independently in every frame. This is translation-only root alignment.
- Do not construct body axes from hips, neck, eyes, or shoulders.
- Do not fit prediction-to-reference rotation or scale for MPJPE.
- Preserve the native predicted scale. Unit conversions documented by an official implementation are allowed and recorded.

For single-view front ends and temporal refiners, score each physical view separately against the reference expressed in that camera frame, then pool all joint-frame errors across the two view roles. `View mean` denotes this pooled score, not an arithmetic average of two 3-D poses.

For MetaPose, use its restored left-camera output. For DLT methods, use their calibrated reconstruction frame before transforming prediction and reference together into the declared reporting frame. For CanonFuse3D, retain the model's internal preprocessing and map the prediction back through the inverse transform computed from the left input only; no target-derived transform is permitted. A round-trip test must demonstrate that the same inverse reconstructs the left native input before the CanonFuse3D result is considered comparable.

If a method cannot be placed in the common frame without target-derived fitting, its MPJPE cell must be marked unavailable rather than filled with a protocol-specific number.

## Metrics

The table reports, for Unity and Ski-PTZ-Pose:

- **MPJPE:** Euclidean error after the translation-only root protocol above; no scale or rotation fit.
- **PA-MPJPE:** a separate per-frame proper similarity alignment with translation, nonnegative uniform scale, and proper rotation; reflection is forbidden.
- **Acceleration error:** second-order difference error computed from the root-relative native trajectories before the PA fit.

All metrics are pooled over all evaluated joint-frame points. PA-MPJPE does not alter MPJPE or acceleration.

## Table rows

### Native monocular front ends

1. SAM3D (native, view mean)
2. MotionBERT (native, view mean; ground-truth 2-D input disclosed)
3. PoseFormer (native, view mean; ground-truth 2-D input disclosed)
4. VideoPose3D (native, view mean; ground-truth 2-D input disclosed)

### Pre-canonicalization controls

5. Left native
6. Right native
7. Unaligned native average
8. Native sequence-aligned average
9. Native aligned quality-weighted fusion
10. Native aligned average + SmoothNet

`Unaligned native average` is retained only as a diagnostic showing the consequence of averaging camera-coordinate poses. It is not eligible for boldface or a best-method claim. Alignment in rows 8--10 uses only the two predictions and never the reference.

The existing canonical-space MLP and TCN are not reused on native inputs. Adding them would require a separately approved native-coordinate-space training target and retraining protocol.

### Published external pose-processing methods

11. STRIDE (native per-view refinement; pooled view score)
12. DeciWatch (native per-view sparse recovery; pooled view score)
13. MetaPose (native multiview output)

STRIDE and DeciWatch must not receive the canonical average used by the current adapters. Their official architectures and frozen, predeclared configurations are retained, with only joint/window boundary adaptations documented. MetaPose retains its own official observation normalization; that normalization is not CanonFuse3D canonicalization.

### Calibrated geometric methods

14. Calibrated DLT
15. Reprojection-gated DLT
16. DLT-residual MLP

These rows use estimated 2-D joints and camera calibration and remain a separately labeled information group.

### Proposed method

17. CanonFuse3D (ours; internal Canon)

## Table layout and interpretation

The table is inserted immediately after `tab:native_reference` as a separate `table*`:

| Method | Canon | Unity MPJPE | Unity PA-MPJPE | Unity Accel. | Ski MPJPE | Ski PA-MPJPE | Ski Accel. |
|---|---:|---:|---:|---:|---:|---:|---:|

The caption will call this a native-output and pre-canonicalization comparison. Boldface, if used, is restricted to methods sharing the same input and output-coordinate group. The text must not describe the table as an overall ranking because the monocular lifters, pose refiners, calibrated methods, and CanonFuse3D have different input requirements.

## Implementation artifacts

- A new read-only native-output evaluator under `dual2pose/eval/`.
- Focused tests for: no competitor canonicalizer calls; coordinate round trips; root-only MPJPE; PA reflection rejection; left/right role pooling; full-split denominators; and target-open-after-prediction ordering.
- A new immutable result root under `logs/ivc_mmsports_extension/` with per-method JSON, prediction checksums, configuration hashes, and progress files.
- A new LaTeX table file under `paper/ivc_draft_20260821/tables/`, included immediately after the current main table.
- A manuscript paragraph that distinguishes native-output accuracy from the existing canonical-space fusion comparison.

## Acceptance criteria

1. All 17 rows are measured or explicitly marked unavailable with a documented protocol reason; no placeholder number is permitted.
2. Automated checks confirm that no non-CanonFuse row invokes the CanonFuse canonicalizer.
3. MPJPE uses translation-only root alignment in a declared common frame; PA-MPJPE is computed independently.
4. Existing main-table JSON and LaTeX values remain byte-for-byte unchanged.
5. The compiled PDF places the new table after the current main table and contains the protocol note in readable form.
6. The result manifest records exact inputs, code hashes, checkpoints, joint mappings, units, sample counts, and aggregation denominators.
