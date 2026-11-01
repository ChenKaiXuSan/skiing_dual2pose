# MetaPose Unity stability and depth detection implementation plan

> Execution: inline with test-driven-development and executing-plans. User authorized implementation, train-only verification, then Pegasus retraining/evaluation; no further approval gate is needed for those actions.

**Goal:** Add a separate, auditable Unity stable runner, verify with training data only, then run pilot -> full fit -> frozen export -> per-frame evaluation in a new Pegasus directory.
**Architecture:** Preserve the pinned official model and the 2D MSE objective. Reuse identity-stage initialization, Adam1e-5, clip norm1 and deterministic TF operations with TF32 explicitly disabled. Add streaming depth/geometry checks relative to the fixed S1 prediction, using only training inputs. No GT3D regularizer, output reflection, or test-based choice.
**Tech:** TensorFlow2.15.1, NumPy, existing Torch evaluator, Pegasus HP260146/gen_S.
**Spec:** User request and `reports/metapose_unity_audit_20261031/README.md`.

## Constraints and design decisions

- Add new files in the current per-frame branch; do not touch unrelated dirty files, old runners, paper or old runs. Deploy a new immutable source snapshot. No commit/push required by this request.
- Official modules, initialization unrolling and forward 2D MSE remain; this is a stability adaptation, not an exact original-paper reproduction. Original GMMs came from heatmaps; ours are point-centered uncertainty mixtures. Document this distinction.
- Depth check: center joints separately for each view/frame; compare projected Z against fixed S1 Z. Exclude reference depth std below 1% of reference 3D joint RMS (uninformative planar views). Detect cosine <=0 or std ratio outside [0.1,10]; abort if bad informative view-frames exceed1%. Fail closed if no informative views. Radius and positive-scale checks remain. These are input-consistency guards, not ground-truth accuracy guarantees.
- No new depth loss or hard constraint on inference. This avoids silently changing the official 2D training objective. Failed guard stops the run and prevents export, not silent seed retries or test selection.
- Pilot: fixed seed42, 32768 sampled training frames, all3 stages x3 epochs; no val/test arguments/data, require final train MSE below S1, all health gates and strict reload pass.
- Formal: seed42,3x10 epochs,batch1024,all train/val frames. All epochs must pass full training-only health checks; then select lowest complete val 2D MSE per stage as before. Export requires full history, hashes, config and guard checks. Test opened only by separate export after fit frozen.
- All output directories exclusive-create. Existing stage best weights may be updated only inside the new run, final artifacts sealed by hashes. Reports carry source hashes and method label.

## Review focus

1. Reflection with unchanged 2D/radius must fail depth gate.
2. Constant depth/NaN/near-planar reference must not silently pass.
3. Streaming chunking must not change decisions or coverage.
4. Pilot/test-data role leakage or incomplete training must block export.
5. Failed pilot/fit must prevent later jobs, leaving old results untouched.

## Tasks

- [x] Add failing depth/geometry tests, then implement independent streaming guard module.
- [x] Add failing pilot/full export validation tests, then implement Unity train/pilot/export runner.
- [x] Run focused protocol tests and TensorFlow checks; train-only Pegasus pilot, no full job until pass.
- [x] Review complete change and immutable deployment; run full train -> export -> eval with failure gating, inspect receipts/logs.

## Alternatives

- Copy Ski stable runner verbatim: rejected because Ski has no validation, small full-memory arrays,300 fixed epochs and no depth test.
- Add depth regularization immediately: deferred; it changes the objective and needs its own declared methodology. Initial implementation preserves 2D MSE and detects failures.
- Current choice: numerical stabilization plus explicit train-only rejection. If this fails, report the failure and investigate train-only evidence rather than launch unconditional full training.

## Execution rulings

- GPU preflight on H100 exposed a float32 batch-vs-eager initializer tolerance failure (max0.00032347 vs atol0.00002). Disable TF32 explicitly in this new runner and rerun the unchanged test; do not loosen the tolerance or alter the official source.
- Seal validation manifest and verify a fresh reconstructed selected stack on fixed training inputs.

## Completion record — 2026-11-01

The stable Unity run `metapose_unity_stable_20261031T222817_v3` completed training, export and evaluation. The follow-up `unity_geometry_v2_20261101T062040Z` measured both timing scopes. These checkmarks record completed tasks, not unmodified official-protocol equivalence. Original run snapshots and artifacts remain authoritative. The current consolidation suite passed 688 tests with six explicit skips and 24 subtests; Unity Editor execution was not performed.
