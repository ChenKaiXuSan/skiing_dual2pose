# IVC MetaPose adapted implementation plan

**Status:** Complete and archived on 2026-09-13 for both Unity and Ski-PTZ-Pose.

> **For agentic workers:** Use superpowers:executing-plans inline, task by task. This continues the author's approved full-experiment request. No delegation, commit, push, or manuscript-number replacement.

**Goal:** train and evaluate a learned, domain-retrained MetaPose common-13 adaptation on the complete archived Unity/Ski split.

**Architecture:** a target-free native-input exporter, a separately guarded 2D-supervision exporter, the unmodified official TensorFlow S1 and learned-stage modules, and a downstream frozen-prediction evaluator.

**Tech Stack:** NumPy and the existing PyTorch evaluator; isolated TF 2.15.1, TFP 0.23.0, Python 3.11. Official Google Research commit `08a8d6736475776f42ffac23b2c13111a28e5795`.

**Spec:** `2026-09-10-ivc-external-baselines.md`, sections 2, 4 and 8; the concrete adaptation contract below resolves the outstanding MetaPose gates.

## Global constraints and frozen adaptation contract

- Existing cache counts: Unity train/val/test 193320/128880/64440; Ski train/test 145/30. Each window has 30 frames. Complete evaluation, including partial batches; finite failures stop the run, never substitute the average or drop examples.
- Seed 42, three sequential learned stages, Adam learning rate 0.0001, float64, SELU, official main MLP `[512,512,'ccat',512,512,'ccat',512]`, pose embedding 512, pose MLP `[256,128]`. Common-13 output, two cameras, four mixture components. No borrowed 17-joint checkpoint, fake joints, bone-length branch, GT initialization, or GT heatmaps. No H36M-index-specific pose standardization.
- Full-run timeout is 24 hours per dataset (not the skill's 30-minute diagnostic default), with roughly 30-second progress output. Training completion automatically invokes the independent PyTorch evaluator. No automatic failure retry or budget expansion.
- Unity: 10 epochs per stage, frame batch 1024, choose each stage's lowest full-validation 2D MSE checkpoint; freeze previous stages. Ski: 300 epochs per stage, frame batch 128, retain the fixed final epoch; no test-based selection. These are budgeted adaptations, not the original paper's training sweep. Single seed must be disclosed.
- Train using only 2D reprojection MSE (official best configuration's forward loss). Ground-truth 3D and supplied calibration do not enter initialization, training, or inference. Training/validation GT 2D is a separate label API and is never a likelihood center. Validation labels are not used to fit uncertainty.
- Per frame/view fit `u = a * X_xy + b` in least squares from native predictions alone. Let `o=min(u)` and `d=max(range(u_x),range(u_y))`; normalize estimated 2D as `(u-o)/d` and all three 3D axes with `q=a/d`, translation `t=[(b-o)/d,0]`. Reject nonfinite/degenerate/nonpositive fits. This is an input-estimated weak camera, not supplied calibration.
- Point uncertainty is a train-only four-scale isotropic zero-mean mixture per joint, fitted by 30 EM steps to normalized estimated-minus-GT-2D residuals. Deterministically sample at most 200000 training frame pairs (seed 42); include both views. Variance floor 1e-6; component centers remain the estimated points. This is point-derived uncertainty, not original detector heatmaps.
- Unity GT 2D uses existing `filter_unity_kpts(..., '2d', gender)` with its required source-index +1 offset. Ski GT 2D uses existing common-13 mapping and normalized image coordinates multiplied by 256, matching https://github.com/HowieMa/TransFusion-Pose/blob/main/data/preprocess_skipose.py . Only `2D/subj/seq/cam/frame` HDF5 fields are read by the training-label exporter.
- S1 uses the official `initial_epi_estimate`, including its first-view convention. Recover each learned output's left-camera 3D using its estimated camera, invert that input's `q,t`, then restore the native left prediction's per-frame pelvis (unobservable global translation; hips 4/5). No GT root/scale is used.
- Compatibility detail: TF 2.15 cannot graph the original S1 `tf.range` plus Python-list loop. The adapter unrolls its exactly two-view case using the original `align_aba` and `reparam`; packing is checked against the original eager function. Official learned modules and third-party source remain unchanged.
- For evaluation, apply the native left-stream batch canonical transform: Unity native15, batch256, then select13; Ski native13, batch4. GT is independently canonicalized only downstream. MPJPE/PA-MPJPE/acceleration retain archived units.
- Report `MetaPose (adapted, 2D-supervised, additional estimated 2D)` and S1 separately. All existing paper cells remain unchanged.

## Task 1 — adapters and isolated supervision

Files: `dual2pose/eval/metapose_inputs.py`, `dual2pose/eval/metapose_supervision.py`, `tests/test_metapose_inputs.py`.

- [x] RED: `python -m unittest tests.test_metapose_inputs -v` initially failed on missing new modules.
- [x] Implement `normalize_observations(xyz, uv) -> (xyz, uv, q, t, origin, side)`, `restore_left_gauge(projected, q, t, native_left)`, `fit_point_uncertainty(residuals)`, `point_mixtures(uv, uncertainty)`. Test synthetic round trip, input-derived pelvis, positive mixtures, estimated-only centers and invalid-input rejection.
- [x] Export inputs after cache/index SHA checks and exact frame matching. Separate label API refuses test before opening annotations. Refuse output overwrite, write complete manifests only after flush.
- [x] GREEN: run adapter and pre-existing external-baseline tests.

## Task 2 — official learned-stage training and prediction

Files: `dual2pose/experiments/prepare_metapose.py`, `dual2pose/experiments/run_metapose.py`, `tests/test_metapose_model.py`.

- [x] Use official S1, `InvariantStageModel` and `DeepInverseSolver`; record source revision and local code hashes.
- [x] Verify S1 projection parity, finite gradients, decreasing real training loss, strict checkpoint reload and unchanged frozen stages.
- [x] Cache frame-level initialization and normalized observations; construct mixtures by batch. Train all frame pairs with full Unity validation.
- [x] Save selected stage weights, losses, configuration, split/uncertainty hashes and environment. Infer only after weights freeze; save S1 and learned raw-left common13 predictions with full coverage.

## Task 3 — full evaluation and handoff

File: `dual2pose/eval/evaluate_metapose.py`; artifacts under `logs/ivc_mmsports_extension/external_baselines/20260911/`.

- [x] Check native canonical transforms and partial batches against archived implementation.
- [x] Run complete Ski first, then Unity with frozen contract; durable stdout/PID and completion/error records.
- [x] Verify hashes/finite coverage, use `PAMetricAccumulator`, and verify archived average control before admitting results.
- [x] Record exact status and run paths; never label running or diagnostic-only work as complete.

## Preflight evidence

2026-09-11: both full STRIDE jobs are complete (64440 Unity / 30 Ski). Existing paper values remain unchanged. The released two-camera H36M checkpoint has finite stages 0–2 and nonfinite stages 3–6; the official one-stage model passes strict finite forward checks. This common13 experiment does not load that checkpoint. Independent GPU environment: `/tmp/ivc-metapose-20260911-7P6c7g/env`; TF 2.15.1 recognizes GPU1. Source remains outside the repository.

## Historical execution checkpoint — 11:11 JST

- Ski full run **complete and independently rescored**: `logs/ivc_mmsports_extension/external_baselines/20260911/metapose_ski_full_seed42/metrics.json`. All 30 windows / 900 frames are finite, all three final-epoch stage hashes verified; average control matches exactly. MPJPE 0.4252570622, PA-MPJPE 0.1425203865, acceleration 0.0634478443. Mixed/negative findings retained.
- Unity full process PID 459395 active on GPU0, still initializing full training inputs before validation initialization and learning. Do not mark the aggregate training/evaluation checklist complete until its final metrics exist and pass artifact checks.
- Detailed commands, method deviations, final Ski table and live status instructions are in `docs/ivc_external_baselines_20260911.md`.
- No manuscript editing or push; all 28 original numerical result-evidence hashes unchanged.

## Final completion checkpoint — 2026-09-13

- Unity reached `complete_evaluated` with 64,440 windows and 1,933,200 frames. Learned MetaPose MPJPE/PA-MPJPE/acceleration error are 0.2654556570/0.1089323300/0.0399751340.
- Ski reached `complete_evaluated` with 30 windows and 900 frames. Learned MetaPose values are 0.4252570622/0.1425203865/0.0634478443.
- Both saved-prediction evaluations have complete point counts, no degenerate PA frames, and zero delta from the archived canonical-average controls. The final rows are integrated into the manuscript with the additional-estimated-2D input distinction.
- Compact copies of final metrics, configuration, progress, selected-stage and report files are archived under `paper/ivc_draft_20260821/evidence/external_baselines/`.
