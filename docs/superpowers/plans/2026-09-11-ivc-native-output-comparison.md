# IVC Native-Output Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second, independent IVC comparison table that measures native front-end and external-method outputs before CanonFuse3D canonicalization, while reporting ordinary MPJPE, PA-MPJPE, and acceleration error under an auditable no-target-fit protocol.

**Architecture:** Use a two-stage, immutable pipeline. Stage A exports target-free native predictions plus coordinate-frame manifests and freezes their checksums. Stage B opens test references, converts prediction and reference together through only documented camera or observation transforms, applies translation-only pelvis alignment for MPJPE, and aggregates every joint-frame point. CanonFuse3D is isolated in the only module allowed to invoke the body canonicalizer; all other rows are implemented in modules that are statically and dynamically checked to remain Canon-free.

**Tech Stack:** Python 3, NumPy, PyTorch, `unittest`, existing dual2pose loaders and PA-MPJPE implementation, JSON/NPY manifests, LaTeX, `latexmk`, `pdftotext`.

**Spec:** `docs/superpowers/specs/2026-09-11-ivc-native-output-comparison-design.md`

## Global Constraints

- Preserve `paper/ivc_draft_20260821/tables/native_reference.tex` and all existing main-table JSON files byte-for-byte.
- Do not edit the currently uncommitted STRIDE, DeciWatch, MetaPose, or main-baseline adapter files. Add separate native-output modules so this work does not overwrite the canonical-space experiments already in progress.
- Do not select a checkpoint, alignment, threshold, subset, or presentation rule from test results. Any train/validation selection must be frozen in the prediction manifest before test targets are opened.
- Do not run full GPU jobs until `nvidia-smi` and the process table show that the selected device is available. This shared worktree may also contain another session's active jobs.
- Store new artifacts only below `logs/ivc_mmsports_extension/native_output_comparison/20260911/`. Result files are create-once: reruns use a new run ID rather than overwrite an existing manifest or score.
- Use the fixed common-13 subset, 30-frame windows, all 64,440 Unity test windows, and all 30 Ski-PTZ-Pose test windows.
- A missing common coordinate transform is a scientific result: emit `status: unavailable` and a precise reason. Never replace it with a target-fitted number.
- Do not commit from this shared side-conversation worktree. Record the exact changed-file list and verification output for the main thread to integrate.

---

## Task 1: Freeze the approved protocol and immutable result contract

**Files:**

- Modify: `docs/superpowers/specs/2026-09-11-ivc-native-output-comparison-design.md`
- Create: `dual2pose/eval/native_output_manifest.py`
- Test: `tests/test_native_output_manifest.py`

- [x] **Step 1: Mark the written design approved**

Change only the status line to:

```markdown
**Status:** Approved for implementation on 2026-09-11
```

- [x] **Step 2: Write failing tests for the two-stage manifest**

The tests must require:

```python
from dual2pose.eval.native_output_manifest import (
    PredictionManifest,
    freeze_prediction_manifest,
    open_scoring_manifest,
)

class NativeOutputManifestTest(unittest.TestCase):
    def test_scoring_requires_frozen_prediction_checksum(self):
        with self.assertRaisesRegex(ValueError, "prediction manifest is not frozen"):
            open_scoring_manifest(self.unfrozen_path, self.target_path)

    def test_freeze_rejects_target_fields(self):
        manifest = PredictionManifest(
            method="sam3d_left_native",
            dataset="unity",
            coordinate_frame="left_camera_m",
            prediction_path="predictions.npy",
            sample_count=64440,
            frame_count=30,
            joint_count=13,
            unit="m",
            source_checkpoint_sha256="1" * 64,
            config_sha256="2" * 64,
            code_sha256="3" * 64,
            sample_index_sha256="4" * 64,
            target_path="forbidden.npy",
        )
        with self.assertRaisesRegex(ValueError, "target"):
            freeze_prediction_manifest(manifest, self.output)

    def test_existing_frozen_manifest_is_not_overwritten(self):
        freeze_prediction_manifest(self.valid_manifest, self.output)
        with self.assertRaises(FileExistsError):
            freeze_prediction_manifest(self.valid_manifest, self.output)
```

- [x] **Step 3: Run the focused test and verify it fails**

Run:

```bash
conda run -n dual2pose python -m unittest tests.test_native_output_manifest -v
```

Expected: import failure because `native_output_manifest.py` does not yet exist.

- [x] **Step 4: Implement the manifest contract**

Implement frozen dataclasses with these required fields:

```python
@dataclass(frozen=True)
class PredictionManifest:
    method: str
    dataset: str
    coordinate_frame: str
    prediction_path: str
    sample_count: int
    frame_count: int
    joint_count: int
    unit: str
    source_checkpoint_sha256: str
    config_sha256: str
    code_sha256: str
    sample_index_sha256: str
    prediction_sha256: str | None = None
    target_path: str | None = None

@dataclass(frozen=True)
class ScoringManifest:
    prediction_manifest_sha256: str
    target_path: str
    target_sha256: str
    coordinate_transform: dict[str, object]
    expected_point_count: int
    expected_acceleration_point_count: int
```

`freeze_prediction_manifest()` must compute the prediction checksum, reject target-bearing fields, write with exclusive creation mode, and set `frozen: true`. `open_scoring_manifest()` must validate the frozen checksum before hashing or opening the target.

- [x] **Step 5: Make the focused tests pass**

Run the Task 1 command again. Expected: all manifest tests pass.

---

## Task 2: Implement metric semantics independently of coordinate conversion

**Files:**

- Create: `dual2pose/eval/native_output_protocol.py`
- Test: `tests/test_native_output_protocol.py`
- Reuse: `dual2pose/eval/pa_mpjpe.py`

- [x] **Step 1: Write failing tests for root-only MPJPE**

Cover these exact invariants:

```python
def pelvis(pose: np.ndarray, left_hip: int = 4, right_hip: int = 5) -> np.ndarray:
    return 0.5 * (pose[..., left_hip, :] + pose[..., right_hip, :])

class NativeOutputProtocolTest(unittest.TestCase):
    def test_translation_is_removed_but_rotation_and_scale_are_not(self):
        translated = self.target + np.array([8.0, -3.0, 2.0])
        self.assertAlmostEqual(native_mpjpe(translated, self.target), 0.0)
        self.assertGreater(native_mpjpe(self.target @ self.rotation, self.target), 0.0)
        self.assertGreater(native_mpjpe(2.0 * self.target, self.target), 0.0)

    def test_pa_does_not_change_mpjpe_or_acceleration(self):
        first = metric_result(self.prediction, self.target)
        second = metric_result(self.prediction, self.target, compute_pa=False)
        self.assertEqual(first["mpjpe"], second["mpjpe"])
        self.assertEqual(first["acceleration_error"], second["acceleration_error"])

    def test_acceleration_never_crosses_sequence_boundaries(self):
        accumulator = NativeMetricAccumulator()
        accumulator.update(self.sequence_a, self.target_a)
        accumulator.update(self.sequence_b, self.target_b)
        self.assertEqual(accumulator.result()["acceleration_point_count"], 2 * 28 * 13)

    def test_reflection_is_not_accepted_by_pa(self):
        reflected = self.target.copy()
        reflected[..., 0] *= -1
        self.assertGreater(metric_result(reflected, self.target)["pa_mpjpe"], 0.0)
```

- [x] **Step 2: Run the focused test and verify it fails**

```bash
conda run -n dual2pose python -m unittest tests.test_native_output_protocol -v
```

- [x] **Step 3: Implement the metric module**

The public API is:

```python
def root_center(pose: np.ndarray, *, left_hip: int = 4, right_hip: int = 5) -> np.ndarray:
    root = 0.5 * (pose[..., left_hip, :] + pose[..., right_hip, :])
    return np.asarray(pose, dtype=np.float64) - root[..., None, :]

class NativeMetricAccumulator:
    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Accumulate float64 distance sums without crossing sequence boundaries."""

    def result(self) -> dict[str, float | int]:
        """Return sums divided by exact point denominators."""

def metric_result(
    prediction: np.ndarray, target: np.ndarray, *, compute_pa: bool = True
) -> dict[str, float | int]:
    accumulator = NativeMetricAccumulator(compute_pa=compute_pa)
    accumulator.update(prediction, target)
    return accumulator.result()
```

Implementation rules:

- Accept only finite matching `(N,T,13,3)` arrays in metres.
- Root-center prediction and target independently for MPJPE and acceleration.
- Compute acceleration inside each sequence as `x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]`.
- Call existing `pa_joint_errors()` on root-centred arrays; do not feed its aligned output into either other metric.
- Accumulate float64 sums and exact integer denominators.
- For `N` sequences, require `point_count == N*30*13` and `acceleration_point_count == N*28*13`.

- [x] **Step 4: Run protocol and existing PA regression tests**

```bash
conda run -n dual2pose python -m unittest \
  tests.test_native_output_protocol \
  tests.test_pa_mpjpe -v
```

Expected: all tests pass and the existing reflection-free PA behavior is unchanged.

---

## Task 3: Declare and validate native coordinate frames

**Files:**

- Create: `dual2pose/eval/native_output_frames.py`
- Test: `tests/test_native_output_frames.py`
- Read without modifying: `dual2pose/eval/main_baseline_geometry.py`

- [x] **Step 1: Write failing frame-conversion tests**

Require a conversion to transform prediction and target with the same supplied transform, prohibit target-derived fits, and return an explicit unavailable decision:

```python
@dataclass(frozen=True)
class FrameDecision:
    status: Literal["comparable", "unavailable"]
    reporting_frame: str
    reason: str | None

class NativeOutputFramesTest(unittest.TestCase):
    def test_rigid_camera_transform_is_applied_to_both_arrays(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([2.0, 3.0, 4.0])
        pred, target = transform_pair(self.prediction, self.target, rotation=rotation, translation=translation)
        np.testing.assert_allclose(pred, self.prediction @ rotation.T + translation)
        np.testing.assert_allclose(target, self.target @ rotation.T + translation)

    def test_unit_conversion_changes_prediction_and_target_together(self):
        pred, target = transform_pair(self.prediction, self.target, rotation=np.eye(3), translation=np.zeros(3), scale=0.001)
        np.testing.assert_allclose(pred, self.prediction * 0.001)
        np.testing.assert_allclose(target, self.target * 0.001)

    def test_transform_api_has_no_target_fit_mode(self):
        parameters = set(inspect.signature(transform_pair).parameters)
        self.assertTrue(parameters.isdisjoint({"fit", "procrustes", "target_rotation", "target_scale"}))
    def test_ski_dlt_is_unavailable_without_reference_world_mapping(self):
        decision = decide_common_frame(
            method_frame="calibrated_world_m",
            target_frame="constructed_camera_local_m",
            supplied_transform=None,
        )
        self.assertEqual(decision.status, "unavailable")
        self.assertIn("no supplied frame mapping", decision.reason)
```

- [x] **Step 2: Run the focused test and verify it fails**

```bash
conda run -n dual2pose python -m unittest tests.test_native_output_frames -v
```

- [x] **Step 3: Implement frame declarations and conversions**

Provide only explicit transforms:

```python
def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if scale <= 0 or not np.isclose(np.linalg.det(rotation), 1.0):
        raise ValueError("transform must use positive scale and proper rotation")
    return scale * (np.asarray(points, dtype=np.float64) @ rotation.T) + translation

def transform_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        transform_points(prediction, rotation, translation, scale),
        transform_points(target, rotation, translation, scale),
    )

def decide_common_frame(
    *, method_frame: str, target_frame: str, supplied_transform: Mapping[str, object] | None
) -> FrameDecision:
    if method_frame == target_frame:
        return FrameDecision("comparable", method_frame, None)
    if supplied_transform is None:
        return FrameDecision("unavailable", method_frame, "no supplied frame mapping")
    return FrameDecision("comparable", str(supplied_transform["reporting_frame"]), None)
```

Do not expose Procrustes, landmark, hip-axis, neck-axis, target rotation, or target scale arguments in this module. Reject improper rotations and non-positive scales. The evaluator may use this module only for documented camera/world transforms and unit conversions.

- [x] **Step 4: Audit dataset frame metadata before any full experiment**

Run a read-only metadata audit that writes only under the new result root:

```bash
conda run -n dual2pose python -m dual2pose.eval.native_output_frames \
  --audit-datasets \
  --output logs/ivc_mmsports_extension/native_output_comparison/20260911/frame_audit.json
```

Expected decisions:

- Unity SAM3D/lifter per-view outputs: comparable in their declared camera frames when the loader supplies the matching target transform.
- Unity DLT: comparable in calibrated Unity world frame.
- Ski DLT: unavailable unless an independently supplied calibration-to-reference mapping exists; the existing constructed camera-local reference is not enough.
- Any unexpected `comparable` decision must include the exact transform source and checksum.

- [x] **Step 5: Run the frame tests again**

Expected: all frame tests pass and `frame_audit.json` contains no fitted transform.

---

## Task 4: Implement native front-end rows and pre-Canon controls

**Files:**

- Create: `dual2pose/eval/native_output_methods.py`
- Create: `dual2pose/experiments/export_native_frontend_predictions.py`
- Test: `tests/test_native_output_methods.py`
- Test: `tests/test_native_output_no_canon.py`
- Reuse without changing canonical behavior: `dual2pose/eval/frontend_lifters.py`

- [x] **Step 1: Write failing behavioral tests for all six controls**

Tests must cover:

```python
class NativeOutputMethodsTest(unittest.TestCase):
    def test_unaligned_average_is_literal_average(self):
        np.testing.assert_allclose(unaligned_average(self.left, self.right), 0.5 * (self.left + self.right))

    def test_sequence_alignment_uses_predictions_only(self):
        signature = inspect.signature(align_right_to_left_sequence)
        self.assertNotIn("target", signature.parameters)

    def test_quality_weights_depend_only_on_inputs(self):
        self.assertNotIn("target", inspect.signature(aligned_quality_weighted).parameters)
        first_pose, first_weights = aligned_quality_weighted(self.left, self.right)
        second_pose, second_weights = aligned_quality_weighted(self.left.copy(), self.right.copy())
        np.testing.assert_array_equal(first_pose, second_pose)
        np.testing.assert_array_equal(first_weights, second_weights)
        np.testing.assert_allclose(first_weights.sum(axis=-1), 1.0)

    def test_view_pooling_uses_all_point_errors_not_mean_pose(self):
        result = pooled_view_metrics(self.left_one, self.right_three, self.left_gt_one, self.right_gt_three)
        separate = [metric_result(self.left_one, self.left_gt_one), metric_result(self.right_three, self.right_gt_three)]
        expected = sum(item["mpjpe"] * item["point_count"] for item in separate) / sum(item["point_count"] for item in separate)
        self.assertAlmostEqual(result["mpjpe"], expected)
        self.assertNotAlmostEqual(result["mpjpe"], 0.5 * sum(item["mpjpe"] for item in separate))

    def test_common13_and_thirty_frame_shapes_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "30 frames and 13 joints"):
            unaligned_average(np.zeros((1, 29, 13, 3)), np.zeros((1, 29, 13, 3)))
```

The pooled-view test must deliberately use unequal per-view sample counts and verify a pooled numerator/denominator, not `(left_mean + right_mean)/2`.

- [x] **Step 2: Write a failing source and runtime guard against Canon**

`tests/test_native_output_no_canon.py` must scan every non-ours native module and monkeypatch the canonicalizer to raise:

```python
FORBIDDEN_MODULES = (
    "dual2pose.eval.native_output_methods",
    "dual2pose.experiments.export_native_frontend_predictions",
    "dual2pose.experiments.run_native_external_predictions",
)

def test_no_competitor_module_mentions_canonicalizer(self):
    for module_name in FORBIDDEN_MODULES:
        source = inspect.getsource(importlib.import_module(module_name))
        self.assertNotIn("canonicalize_pose_torch", source)
        self.assertNotIn("canonical_batch", source)
```

- [x] **Step 3: Run both focused suites and verify they fail**

```bash
conda run -n dual2pose python -m unittest \
  tests.test_native_output_methods \
  tests.test_native_output_no_canon -v
```

- [x] **Step 4: Implement the pre-Canon methods**

Implement:

```python
def unaligned_average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    validate_native_pair(left, right)
    return 0.5 * (left + right)

def align_right_to_left_sequence(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    validate_native_pair(left, right)
    transform = estimate_prediction_only_sequence_transform(source=right, destination=left)
    return apply_sequence_transform(right, transform), transform

def aligned_average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    aligned_right, _ = align_right_to_left_sequence(left, right)
    return 0.5 * (left + aligned_right)

def aligned_quality_weighted(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aligned_right, transform = align_right_to_left_sequence(left, right)
    weights = prediction_only_reliability(left, aligned_right, transform)
    return weights[..., :1] * left + weights[..., 1:] * aligned_right, weights

def apply_smoothnet_native(sequence: np.ndarray, checkpoint: Path) -> np.ndarray:
    metadata = load_checkpoint_metadata(checkpoint)
    require_native_coordinate_contract(metadata)
    return run_frozen_smoothnet(sequence, checkpoint)

def pooled_view_metrics(left_pred: np.ndarray, right_pred: np.ndarray, left_gt: np.ndarray, right_gt: np.ndarray) -> dict[str, float | int]:
    accumulator = NativeMetricAccumulator()
    accumulator.update(left_pred, left_gt)
    accumulator.update(right_pred, right_gt)
    return accumulator.result()
```

The alignment estimator may use only left and right predictions. Preserve predicted scale unless the predeclared alignment method intrinsically estimates a prediction-to-prediction similarity; record that transform in the prediction manifest. Do not introduce a target argument.

For SmoothNet, first verify from its checkpoint metadata that its input/output coordinate target is native and compatible. If the existing checkpoint was trained only on canonical averages, emit `status: unavailable` for the native SmoothNet row. Do not silently apply that checkpoint to a different coordinate distribution and do not retrain without a separately frozen train/validation protocol.

- [x] **Step 5: Export native front-end predictions without targets**

The exporter must create per-view arrays and one frozen manifest for each of:

- SAM3D
- MotionBERT using the predeclared ground-truth 2-D input
- PoseFormer using the predeclared ground-truth 2-D input
- VideoPose3D using the predeclared ground-truth 2-D input

Run a non-test smoke export first:

```bash
conda run -n dual2pose python -m dual2pose.experiments.export_native_frontend_predictions \
  --dataset unity --split validation --limit 2 \
  --methods sam3d motionbert poseformer videopose3d \
  --output-root logs/ivc_mmsports_extension/native_output_comparison/20260911/smoke_frontends
```

Expected: raw `(N,30,13,3)` per-view prediction arrays, target-free manifests, explicit units/frame names, and no canonical cache path.

- [x] **Step 6: Make method and no-Canon tests pass**

Run the Task 4 tests again. Expected: all pass.

---

## Task 5: Isolate CanonFuse3D's internal Canon and validate output recovery

**Files:**

- Create: `dual2pose/eval/native_output_canonfuse.py`
- Test: `tests/test_native_output_canonfuse.py`
- Read without modifying: `dual2pose/models/crossview_fusion.py`
- Read without modifying: `dual2pose/trainer/canonicalize.py`

- [x] **Step 1: Write failing round-trip and leakage tests**

```python
class NativeOutputCanonFuseTest(unittest.TestCase):
    def test_inverse_left_transform_round_trips_left_input(self):
        canonical, transform = canonicalize_left_with_transform(self.left)
        recovered = inverse_left_transform(canonical, transform)
        np.testing.assert_allclose(recovered, self.left, atol=1e-6)

    def test_transform_depends_only_on_inputs(self):
        _, first = prepare_canonfuse_inputs(self.left, self.right)
        _, second = prepare_canonfuse_inputs(self.left, self.right)
        self.assertEqual(first.sha256(), second.sha256())

    def test_public_prediction_api_has_no_target_argument(self):
        self.assertNotIn("target", inspect.signature(predict_native_canonfuse).parameters)
```

- [x] **Step 2: Run the focused test and verify it fails**

```bash
conda run -n dual2pose python -m unittest tests.test_native_output_canonfuse -v
```

- [x] **Step 3: Implement the only Canon-allowed adapter**

Public API:

```python
@dataclass(frozen=True)
class LeftInputTransform:
    pelvis: np.ndarray
    linear: np.ndarray

def canonicalize_left_with_transform(left: np.ndarray) -> tuple[np.ndarray, LeftInputTransform]:
    canonical, raw = canonicalize_pose_torch(torch.from_numpy(left), left_hip=4, right_hip=5, neck=12)
    return canonical.cpu().numpy(), LeftInputTransform(raw["pelvis"].cpu().numpy(), raw["R"].cpu().numpy())

def inverse_left_transform(canonical: np.ndarray, transform: LeftInputTransform) -> np.ndarray:
    flat = np.asarray(canonical, dtype=np.float64).reshape(-1, 3)
    recovered = np.linalg.solve(transform.linear.T, flat.T).T
    return recovered.reshape(canonical.shape) + transform.pelvis

def predict_native_canonfuse(left: np.ndarray, right: np.ndarray, checkpoint: Path) -> tuple[np.ndarray, PredictionManifest]:
    model = load_frozen_canonfuse(checkpoint)
    model_inputs, transform = prepare_canonfuse_inputs(left, right)
    canonical_prediction = run_canonfuse(model, model_inputs)
    native_prediction = inverse_left_transform(canonical_prediction, transform)
    require_left_input_round_trip(left, transform)
    return native_prediction, build_target_free_manifest(native_prediction, checkpoint, transform)
```

The inverse must be derived from the left input transform only. It must use a true matrix solve for the stored linear transform and prove the round trip on every exported batch. Do not recompute a transform from the prediction or target. Because the current canonicalizer flattens `(B,T)` and selects `pelvis[0]` and `R[0]` in `first_frame` mode, the adapter must canonicalize one sequence at a time (or enforce `B=1`) and store one transform per sample; a transform shared across different sequences must fail the round-trip gate.

- [x] **Step 4: Add an admission gate for the learned output gauge**

Because the network was supervised in independently canonicalized target space and fuses left/right canonical gauges, a left-input inverse is not automatically scientifically valid. Add a validation-only admission record with these fields:

```json
{
  "round_trip_max_abs_m": 0.0,
  "output_gauge_declared": "left_input_canonical",
  "gauge_supported_by_model_contract": false,
  "native_mpjpe_admitted": false,
  "reason": "model output gauge is not guaranteed to equal the left-input canonical gauge"
}
```

Set `native_mpjpe_admitted` true only if the architecture/training contract—not a test-target fit—establishes that the output is in the exact left-input canonical gauge. If not, report CanonFuse3D native MPJPE and acceleration as unavailable. PA-MPJPE may be reported only if the model-space transform is proven to be a proper similarity; the current body-axis transform must not be assumed to preserve PA geometry.

- [x] **Step 5: Run round-trip tests and emit the admission record**

Expected: numerical round trip passes. The admission record truthfully states whether a common native frame is justified.

---

## Task 6: Export native STRIDE, DeciWatch, MetaPose, and DLT predictions

**Files:**

- Create: `dual2pose/experiments/run_native_external_predictions.py`
- Test: `tests/test_native_external_predictions.py`
- Reuse read-only outputs where provenance permits: current STRIDE, DeciWatch, MetaPose, and DLT experiment roots

- [x] **Step 1: Write failing adapter-contract tests**

Require:

```python
class NativeExternalPredictionsTest(unittest.TestCase):
    def test_stride_receives_one_native_view_at_a_time(self):
        runner = unittest.mock.Mock(side_effect=lambda pose: pose)
        export_two_view_refiner("stride", self.left, self.right, runner)
        self.assertEqual(runner.call_count, 2)
        np.testing.assert_array_equal(runner.call_args_list[0].args[0], self.left)
        np.testing.assert_array_equal(runner.call_args_list[1].args[0], self.right)

    def test_deciwatch_receives_one_native_view_at_a_time(self):
        runner = unittest.mock.Mock(side_effect=lambda pose: pose)
        export_two_view_refiner("deciwatch", self.left, self.right, runner)
        self.assertEqual(runner.call_count, 2)
        self.assertIsNot(runner.call_args_list[0].args[0], runner.call_args_list[1].args[0])

    def test_metapose_export_is_raw_restored_left_camera_output(self):
        prediction, manifest = export_external("metapose", self.metapose_raw, self.config)
        np.testing.assert_array_equal(prediction, self.metapose_raw)
        self.assertEqual(manifest.coordinate_frame, "left_camera_m")

    def test_dlt_export_preserves_calibrated_world_frame(self):
        _, manifest = export_external("dlt", self.dlt_raw, self.config)
        self.assertEqual(manifest.coordinate_frame, "calibrated_world_m")

    def test_export_stage_never_opens_target(self):
        with unittest.mock.patch("dual2pose.experiments.run_native_external_predictions.load_target", side_effect=AssertionError("target opened")):
            export_external("dlt", self.dlt_raw, self.config)

    def test_test_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "limit is forbidden for test"):
            validate_export_request(split="test", limit=1)
```

- [x] **Step 2: Run the focused test and verify it fails**

```bash
conda run -n dual2pose python -m unittest tests.test_native_external_predictions -v
```

- [x] **Step 3: Implement target-free external exports**

Rules by row:

- STRIDE: run the frozen official architecture/configuration independently on left-native and right-native sequences; pool later at metric level.
- DeciWatch: apply the frozen official sparse-sampling/recovery protocol independently to each native view. Name outputs `official`, never `recovered` without defining the input observation ratio and checkpoint in the manifest.
- MetaPose: save the restored raw left-camera prediction before any CanonFuse body transform. Reuse an existing raw prediction only if its exact checkpoint, input normalization, sample ordering, and checksum can be proven.
- DLT/gated DLT/DLT-residual: export calibrated-world arrays before body canonicalization. Reuse existing raw arrays only if they are present and their source hashes match.

The runner must reject canonical input labels such as `canonical_average`, `left_canonical`, or `gt_canonical`.

- [x] **Step 4: Run non-test smoke exports**

```bash
conda run -n dual2pose python -m dual2pose.experiments.run_native_external_predictions \
  --dataset unity --split validation --limit 2 \
  --methods stride deciwatch metapose dlt gated_dlt dlt_residual \
  --output-root logs/ivc_mmsports_extension/native_output_comparison/20260911/smoke_external
```

Expected: frozen prediction manifests with no target fields and no canonical inputs.

- [x] **Step 5: Make external and no-Canon tests pass**

```bash
conda run -n dual2pose python -m unittest \
  tests.test_native_external_predictions \
  tests.test_native_output_no_canon -v
```

---

## Task 7: Score all 17 rows with complete-coverage gates

**Files:**

- Create: `dual2pose/eval/evaluate_native_output_comparison.py`
- Test: `tests/test_evaluate_native_output_comparison.py`

- [x] **Step 1: Write failing controller tests**

Cover the evaluation state machine:

```python
class EvaluateNativeOutputComparisonTest(unittest.TestCase):
    def test_target_is_opened_only_after_prediction_manifest_verifies(self):
        target_loader = unittest.mock.Mock(side_effect=AssertionError("target opened early"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            score_frozen_prediction(self.bad_manifest, target_loader=target_loader)
        target_loader.assert_not_called()

    def test_unity_requires_64440_sequences(self):
        with self.assertRaisesRegex(ValueError, "64440"):
            validate_coverage(dataset="unity", sample_count=64439, frame_count=30, joint_count=13)

    def test_ski_requires_30_sequences(self):
        with self.assertRaisesRegex(ValueError, "30"):
            validate_coverage(dataset="ski", sample_count=29, frame_count=30, joint_count=13)

    def test_denominators_are_exact(self):
        result = metric_result(np.zeros((2, 30, 13, 3)), np.zeros((2, 30, 13, 3)))
        self.assertEqual(result["point_count"], 2 * 30 * 13)
        self.assertEqual(result["acceleration_point_count"], 2 * 28 * 13)

    def test_unavailable_row_has_reason_and_no_numeric_metric(self):
        row = unavailable_result("no supplied frame mapping")
        self.assertEqual(row["status"], "unavailable")
        self.assertTrue(row["reason"])
        self.assertIsNone(row["mpjpe"])
        self.assertIsNone(row["pa_mpjpe"])
        self.assertIsNone(row["acceleration_error"])

    def test_existing_result_is_never_overwritten(self):
        write_result_exclusive(self.output, self.report)
        with self.assertRaises(FileExistsError):
            write_result_exclusive(self.output, self.report)

    def test_all_seventeen_method_ids_are_emitted(self):
        self.assertEqual(tuple(build_empty_report()["methods"]), METHOD_IDS)
```

Required method IDs:

```python
METHOD_IDS = (
    "sam3d_native_view_pool",
    "motionbert_native_view_pool",
    "poseformer_native_view_pool",
    "videopose3d_native_view_pool",
    "left_native",
    "right_native",
    "unaligned_native_average",
    "native_sequence_aligned_average",
    "native_aligned_quality_weighted",
    "native_aligned_average_smoothnet",
    "stride_native_view_pool",
    "deciwatch_native_view_pool",
    "metapose_native",
    "calibrated_dlt_native",
    "reprojection_gated_dlt_native",
    "dlt_residual_mlp_native",
    "canonfuse3d_internal_canon",
)
```

- [x] **Step 2: Run the focused test and verify it fails**

```bash
conda run -n dual2pose python -m unittest tests.test_evaluate_native_output_comparison -v
```

- [x] **Step 3: Implement the evaluator**

The CLI must require a frozen prediction root and a separate target descriptor:

```bash
python -m dual2pose.eval.evaluate_native_output_comparison \
  --dataset {unity,ski} \
  --prediction-root PATH \
  --target-descriptor PATH \
  --output PATH.json
```

For each row, write either:

```json
{
  "status": "measured",
  "coordinate_frame": "left_camera_m",
  "mpjpe": 0.0,
  "pa_mpjpe": 0.0,
  "acceleration_error": 0.0,
  "sample_count": 64440,
  "point_count": 25131600,
  "acceleration_point_count": 23456160
}
```

or:

```json
{
  "status": "unavailable",
  "reason": "no supplied frame mapping from calibrated_world_m to constructed_camera_local_m",
  "mpjpe": null,
  "pa_mpjpe": null,
  "acceleration_error": null
}
```

The numeric zeros above express schema types only; tests must use known nonzero fixtures. The evaluator must reject incomplete sample indices, repeated samples, missing frames, non-finite arrays, unexpected joints, wrong units, and manifest checksum mismatches.

- [x] **Step 4: Run controller and protocol tests**

```bash
conda run -n dual2pose python -m unittest \
  tests.test_evaluate_native_output_comparison \
  tests.test_native_output_manifest \
  tests.test_native_output_protocol \
  tests.test_native_output_frames -v
```

Expected: all tests pass.

---

## Task 8: Render the independent table and integrate the manuscript text

**Files:**

- Create: `dual2pose/eval/render_native_output_table.py`
- Create: `paper/ivc_draft_20260821/tables/native_output_comparison.tex`
- Modify: `paper/ivc_draft_20260821/main.tex`
- Modify: `paper/ivc_draft_20260821/revision_log.md`
- Test: `tests/test_native_output_table.py`
- Test: `tests/test_ivc_journal_visuals.py`

- [x] **Step 1: Freeze a checksum of the existing main table**

Before editing:

```bash
sha256sum paper/ivc_draft_20260821/tables/native_reference.tex
```

Save the hash in the native-output run manifest and in the renderer test fixture. Do not regenerate `native_reference.tex`.

- [x] **Step 2: Write failing renderer and manuscript-placement tests**

Require:

```python
class NativeOutputTableTest(unittest.TestCase):
    def test_contains_all_seventeen_rows_in_declared_order(self):
        tex = render_native_output_table(self.fixture_report)
        offsets = [tex.index(label) for label in EXPECTED_ROW_LABELS]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(len(offsets), 17)

    def test_has_mpjpe_pa_and_acceleration_for_both_datasets(self):
        tex = render_native_output_table(self.fixture_report)
        self.assertEqual(tex.count("MPJPE"), 4)
        self.assertGreaterEqual(tex.count("Accel."), 2)
        self.assertIn("Unity", tex)
        self.assertIn("Ski-PTZ-Pose", tex)

    def test_unavailable_cells_render_as_double_dash(self):
        tex = render_native_output_table(self.report_with_unavailable)
        self.assertIn("--", tex)
        self.assertNotIn("nan", tex.lower())

    def test_diagnostic_unaligned_average_is_never_bold(self):
        row = next(line for line in render_native_output_table(self.fixture_report).splitlines() if "Unaligned native average" in line)
        self.assertNotIn("\\textbf", row)

    def test_table_is_input_immediately_after_native_reference(self):
        main = self.main_tex.read_text(encoding="utf-8")
        expected = "\\input{tables/native_reference.tex}\n\\input{tables/native_output_comparison.tex}"
        self.assertIn(expected, main)

    def test_existing_native_reference_hash_is_unchanged(self):
        digest = hashlib.sha256(self.native_reference.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_NATIVE_REFERENCE_SHA256)
```

- [x] **Step 3: Run the focused test and verify it fails**

```bash
conda run -n dual2pose python -m unittest tests.test_native_output_table -v
```

- [x] **Step 4: Implement the renderer**

Generate a standalone `table*` with columns:

```latex
Method & Canon & \multicolumn{3}{c}{Unity} & \multicolumn{3}{c}{Ski-PTZ-Pose} \\
& & MPJPE & PA-MPJPE & Accel. & MPJPE & PA-MPJPE & Accel. \\
```

Group rows as Native monocular front ends, Pre-Canon controls, Published external methods, Calibrated geometry, and Proposed method. Render unavailable metrics as `--` with a caption/footnote stating that no target-independent common frame was available. Only apply bold within explicitly comparable input/output-coordinate groups; do not bold a global winner.

- [x] **Step 5: Insert the table after the existing main table**

In `paper/ivc_draft_20260821/main.tex`, add:

```latex
\input{tables/native_reference.tex}
\input{tables/native_output_comparison.tex}
```

Then add one Results paragraph explaining:

- the existing main table is a canonical-space downstream fusion comparison;
- the new table evaluates native/pre-Canon outputs;
- ordinary MPJPE uses only translation-only pelvis alignment in a declared common frame;
- PA-MPJPE is separate;
- rows with different inputs are not an overall architecture ranking.

Replace the existing claim that the Unity right stream is intrinsically weaker. The current large left/right canonical MPJPE gap must instead be described as a coordinate/canonicalization effect unless the native table independently supports a front-end quality gap.

- [x] **Step 6: Render from fixture scores and make table tests pass**

```bash
conda run -n dual2pose python -m unittest \
  tests.test_native_output_table \
  tests.test_ivc_journal_visuals -v
```

Expected: placement, all columns, grouping, unavailable notation, and immutable old-table hash pass.

---

## Task 9: Run smoke validation, then full target-free prediction exports

**Files:**

- Write only: `logs/ivc_mmsports_extension/native_output_comparison/20260911/`
- Update checkboxes in this plan as each job completes

- [x] **Step 1: Run the complete unit-test gate**

```bash
conda run -n dual2pose python -m unittest \
  tests.test_native_output_manifest \
  tests.test_native_output_protocol \
  tests.test_native_output_frames \
  tests.test_native_output_methods \
  tests.test_native_output_no_canon \
  tests.test_native_output_canonfuse \
  tests.test_native_external_predictions \
  tests.test_evaluate_native_output_comparison \
  tests.test_native_output_table -v
```

Expected: zero failures before any full test export starts.

- [x] **Step 2: Check shared compute state**

```bash
nvidia-smi
ps -eo pid,ppid,etime,cmd
```

Record selected GPU UUID, free memory, process owner, and start time in `compute_manifest.json`. If a GPU is occupied by another experiment, choose another free device or stop here; do not terminate another process.

- [x] **Step 3: Run validation smoke scoring end to end**

Use exported validation predictions, validation references, and fixture-independent real metrics. Confirm every measured result is finite, denominators match the validation index, and every unavailable row carries a reason.

- [ ] **Step 4: Export full Unity test predictions without opening targets**

```bash
CUDA_VISIBLE_DEVICES=SELECTED_GPU conda run -n dual2pose \
  python -m dual2pose.experiments.export_native_frontend_predictions \
  --dataset unity --split test \
  --methods sam3d motionbert poseformer videopose3d canonfuse3d \
  --output-root logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/predictions

CUDA_VISIBLE_DEVICES=SELECTED_GPU conda run -n dual2pose \
  python -m dual2pose.experiments.run_native_external_predictions \
  --dataset unity --split test \
  --methods stride deciwatch metapose dlt gated_dlt dlt_residual \
  --output-root logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/predictions
```

Replace `SELECTED_GPU` with the explicitly recorded device number; do not use an unresolved shell variable. The exporter must also build the six pre-Canon controls from frozen source predictions and freeze every checksum.

- [x] **Step 5: Export full Ski test predictions without opening targets**

Run the corresponding two commands with `--dataset ski` and the Ski output root. Do not reuse Unity configurations unless the external method's frozen protocol explicitly declares them cross-dataset compatible.

- [ ] **Step 6: Audit frozen prediction coverage before scoring**

```bash
conda run -n dual2pose python -m dual2pose.eval.native_output_manifest \
  --verify-root logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/predictions \
  --expected-samples 64440 --expected-frames 30 --expected-joints 13

conda run -n dual2pose python -m dual2pose.eval.native_output_manifest \
  --verify-root logs/ivc_mmsports_extension/native_output_comparison/20260911/ski/predictions \
  --expected-samples 30 --expected-frames 30 --expected-joints 13
```

Expected: all available rows frozen with unique complete indices; unavailable rows have no fabricated prediction file.

---

## Task 10: Open targets, score, render, compile, and audit the final artifact

**Files:**

- Create: `logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/scores.json`
- Create: `logs/ivc_mmsports_extension/native_output_comparison/20260911/ski/scores.json`
- Generate: `paper/ivc_draft_20260821/tables/native_output_comparison.tex`
- Generate: `paper/ivc_draft_20260821/main.pdf`

- [ ] **Step 1: Score full Unity and Ski exactly once**

```bash
conda run -n dual2pose python -m dual2pose.eval.evaluate_native_output_comparison \
  --dataset unity \
  --prediction-root logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/predictions \
  --target-descriptor logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/target_descriptor.json \
  --output logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/scores.json

conda run -n dual2pose python -m dual2pose.eval.evaluate_native_output_comparison \
  --dataset ski \
  --prediction-root logs/ivc_mmsports_extension/native_output_comparison/20260911/ski/predictions \
  --target-descriptor logs/ivc_mmsports_extension/native_output_comparison/20260911/ski/target_descriptor.json \
  --output logs/ivc_mmsports_extension/native_output_comparison/20260911/ski/scores.json
```

Before these commands, create each target descriptor from the fixed dataset index and record its checksum. Do not alter any prediction artifact after this step.

- [ ] **Step 2: Run the result-integrity audit**

Verify:

- Unity measured rows: `sample_count == 64440`, `point_count == 25,131,600`, `acceleration_point_count == 23,456,160` for one pose per sample.
- Ski measured rows: `sample_count == 30`, `point_count == 11,700`, `acceleration_point_count == 10,920` for one pose per sample.
- Pooled two-view rows explicitly record twice the point denominators if each physical view is treated as a separate prediction.
- No result is NaN/Inf.
- Every metric unit is metres.
- Every unavailable cell has a target-independent protocol reason.
- No source or manifest path contains a canonical tensor for non-ours rows.

- [ ] **Step 3: Render the final table from the two score files**

```bash
conda run -n dual2pose python -m dual2pose.eval.render_native_output_table \
  --unity logs/ivc_mmsports_extension/native_output_comparison/20260911/unity/scores.json \
  --ski logs/ivc_mmsports_extension/native_output_comparison/20260911/ski/scores.json \
  --output paper/ivc_draft_20260821/tables/native_output_comparison.tex
```

- [ ] **Step 4: Re-run all focused and manuscript tests**

Run the Task 9 test gate plus:

```bash
conda run -n dual2pose python -m unittest tests.test_ivc_journal_visuals -v
```

Expected: zero failures.

- [ ] **Step 5: Compile and inspect the real PDF**

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftotext main.pdf /tmp/ivc_native_output_main.txt
rg -n "Native-output|pre-canonicalization|PA-MPJPE|unavailable" /tmp/ivc_native_output_main.txt
```

Run `latexmk` from `paper/ivc_draft_20260821/`. Inspect the rendered table page visually, confirming it follows the existing main table, fits the page, contains all groups, and has readable metric headings/notes.

- [ ] **Step 6: Verify the old main table is unchanged**

```bash
sha256sum paper/ivc_draft_20260821/tables/native_reference.tex
git diff -- paper/ivc_draft_20260821/tables/native_reference.tex
```

Expected: the checksum matches Task 8 and the diff is empty.

- [ ] **Step 7: Produce the final experiment report**

Report:

- all 17 Unity and Ski rows with measured values or explicit unavailable reasons;
- which methods used GT 2-D, calibration, sparse observations, or method-internal Canon;
- exact checkpoints/configs, prediction hashes, coordinate frames, units, and denominators;
- any row excluded from cross-group best-method claims;
- the compiled PDF path and page containing the new table;
- the full verification command output and any unresolved limitation.

Do not claim that CanonFuse3D outperforms a row whose coordinate protocol, input requirements, or metric cell is not directly comparable.

---

## Completion Gate

Implementation is complete only when all of the following are true:

- [ ] The new table exists immediately after the current main table and contains all 17 declared rows.
- [ ] Ordinary MPJPE, PA-MPJPE, and acceleration are present for both datasets when scientifically computable.
- [ ] All non-ours paths pass static and runtime no-Canon checks.
- [ ] Target files were opened only after frozen prediction checksums were verified.
- [ ] Full-split counts and pooled denominators are exact.
- [ ] CanonFuse3D's native-frame admission decision is evidence-based, not assumed.
- [ ] Existing canonical-space table and result artifacts remain unchanged.
- [ ] The PDF compiles, its rendered page is inspected, and the manuscript wording matches the actual protocol/results.
