# Independent initial review: IVC main baselines

Date: 2026-09-06

Scope: read-only review of `dual2pose/models/main_baselines.py`,
`dual2pose/eval/main_baseline_data.py`,
`dual2pose/experiments/run_main_baselines.py`, their focused tests, and the
implementation plan. No implementation or test file was changed and no broad
suite or experiment was rerun.

## Actionable findings

### High: resume compatibility permits mixed training protocols

`run_main_baselines.py:113-122` records the microbatch size, loss, runner hash,
optimizer settings, and training-drop policy, but the resume gate at
`run_main_baselines.py:124-127` compares only dataset, method, seed, epochs,
training batch size, and model-source hash. A checkpoint produced by a changed
runner or loss therefore resumes silently. This was especially hazardous while
the SmoothNet acceleration weight was being corrected: a pre-correction
checkpoint would have been accepted under the corrected objective. The code
then overwrites `config.json` at `run_main_baselines.py:135`, making the run look
uniform even though earlier epochs used a different protocol.

Required correction: reject resume unless all result-affecting configuration
and provenance fields match, including loss, `runner_sha256`, microbatch size,
optimizer hyperparameters, training-drop policy, and cache/index/file hashes.
Preserve every invocation or otherwise record the protocol transition rather
than replacing the earlier config. No current run was resumed, so this finding
does not invalidate completed or active first-invocation runs.

### Medium: cache hashes are recorded but not verified by training/evaluation

The exporter writes file hashes to each manifest at
`main_baseline_data.py:169-176`, and its reuse path verifies them at
`main_baseline_data.py:112-119`. Training and evaluation do not invoke that
reuse path. Their direct loader at `main_baseline_data.py:180-187` reads the
three arrays and checks only the first-dimension count. A corrupted, replaced,
or swapped `.npy` with the same length is accepted, while result provenance
continues to report the stale manifest digest. Shape, dtype, and finiteness are
also not validated at load time.

Required correction: verify manifest hashes and exact expected array
shape/dtype before a run uses a cache, then retain the existing runtime
finiteness check for predictions and targets. No corruption was observed in the
current complete Unity or Ski caches; this is an admission/provenance hazard,
not evidence that current arrays are wrong.

### Medium: the quality heuristic rewards a collapsed view

`main_baselines.py:100-111` clamps zero body scale to epsilon, then scores only
temporal bone-length variation and acceleration. A constant collapsed/all-zero
pose consequently has cost zero. Against a valid moving view it receives more
than half the fusion weight through `softmax(-cost)` at
`main_baselines.py:131-134`; against a valid static view it receives equal
weight. Thus a plausible catastrophic pose-estimator failure is treated as the
cleaner view. The degeneracy test at `tests/test_main_baselines.py:80-92` checks
only collapsed-versus-identical-collapsed input and cannot detect this behavior.

Required correction: add an internal degenerate-skeleton rejection/penalty that
uses only the approved within-view 3-D cues, with explicit behavior when both
views are degenerate, and add collapsed-versus-valid static and moving tests.
Recompute the heuristic rows after the correction.

## Verified checks

- **Reference reproduction:**
  `reference_reproduction_unity.json` reproduces all four authoritative Unity
  batch-256/common-13 reference rows within `2.3e-7` MPJPE; the Ski comparison
  is within `4.3e-7`. The older Unity batch-4096 masking report is a different
  batch-anchor protocol and is not the main-table reference.
- **Checkpoint policy:** the archived Unity `last.ckpt` and validation-selected
  `89-0.0837-0.1590.ckpt` have the same epoch and tensor-identical model state,
  so using `last.ckpt` does not create a checkpoint-selection asymmetry here.
  Ski uses the fixed final checkpoint as planned.
- **No test selection leakage:** test arrays are first loaded only after the
  fixed Ski budget or the Unity validation selection at
  `run_main_baselines.py:170-175`. Test metrics do not affect training or
  checkpoint choice.
- **Partial test batches and batch anchors:** evaluation uses
  `drop_last=False` slicing at `run_main_baselines.py:55-56`; the tests at
  `tests/test_main_baseline_data.py:9-13` retain Unity's final 184 samples and
  Ski's final 2. `canonical_batch` independently applies the archived
  whole-batch first-frame anchor to left, right, and target at
  `main_baseline_data.py:53-57`, covered by
  `tests/test_main_baseline_data.py:18-34`.
- **Training `drop_last=True`:** this is an intentional match to the archived
  Unity and Ski training loaders. Every index remains eligible under seeded
  epoch shuffling; histories must report 192,512 Unity and 144 Ski training
  samples per epoch. The plan wording should distinguish training eligibility
  from complete validation/test coverage.
- **Metric arithmetic:** `MetricAccumulator` sums double-precision pointwise
  distances and divides by point counts at `run_main_baselines.py:20-35`, so
  unequal final batches are sample/point weighted and acceleration never crosses
  sequence boundaries. Focused tests cover partial weighting, a known quadratic
  second difference, and nonfinite rejection.
- **SmoothNet:** `main_baselines.py:224-286` matches the official temporal-only
  layer order and H36M-FCN widths while explicitly adapting the window to 30;
  `tests/test_main_baselines.py:171-183` verifies forward parity with the
  transcribed official topology. The runner now uses position L1 plus 0.1
  acceleration-error L1 at `run_main_baselines.py:122,151-154`, matching the
  recorded official objective. The correction was made before initial
  SmoothNet training, so no current checkpoint mixes loss versions.


## Scoped fix disposition

Re-review date: 2026-09-06. No broad suite or experiment was rerun.

### Approved fixes

- **Resume compatibility: approved.** run_main_baselines.py:15 captures the runner hash at import; lines 105-111 compare every config key except command, and line 139 preserves the initial config. The regression at tests/test_main_baseline_hardening.py:42-49 covers changed loss and microbatch. No learned run resumed, so completed results remain valid.
- **Cache integrity: approved.** main_baseline_data.py:180-198 verifies every manifest-listed hash and exact pose shape, float32 dtype, and finiteness. Tests at tests/test_main_baseline_hardening.py:20-33 cover same-shape tampering and wrong shape with an updated digest.
- **Collapsed-view heuristic: approved.** main_baselines.py:131-142 masks a collapsed stream when the other has extent and keeps both-collapsed inputs finite. The bidirectional regression is at tests/test_main_baseline_hardening.py:35-40. Pre-guard heuristic results were preserved separately; use the corrected recomputation in the tables.

Supplied focused evidence reports the four new tests failing before the fixes and 20/20 passing afterward across the four targeted test modules.

### Remaining blocking finding for table admission

**Medium:** render_main_baseline_tables.py:28-34 checks subset presence, sequence count, and finite headline metrics, but not the metric denominators already stored in every result. A wrong frame/joint evaluation could therefore enter the manuscript with the correct sample_count. Require exact point_count / acceleration_point_count pairs: Unity all15 28,998,000 / 27,064,800; Unity common13 25,131,600 / 23,456,160; Ski common13 11,700 / 10,920. Current inspected result schemas have correct counters; this is an admission-guard gap, not evidence that current metrics are wrong.
