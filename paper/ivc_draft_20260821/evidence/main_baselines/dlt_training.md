# DLT reference and residual-MLP training report

Date: 2026-09-06

## Scope and implementation

This run adds the three approved additional-input rows for the IVC main-table
extension: linear DLT, reprojection-gated robust DLT, and a learned DLT
residual MLP.  The implementation is isolated in:

- `dual2pose/experiments/run_geometry_baselines.py`
- `dual2pose/models/dlt_residual_baseline.py`
- `tests/test_run_geometry_baselines.py`

The MLP is framewise and consumes the complete pose at once.  Its input is
`flatten(canonical DLT) (3J) + log1p(reprojection_error_px / 25) (J) +
DLT validity (J)`.  It has two 256-wide ReLU hidden layers, dropout 0.1, and
a 3J output residual over canonical DLT.  The residual head is initialized to
zero, so the untrained model is exactly the canonical-DLT identity.  Neither
features nor preprocessing accept target coordinates.

Training uses AdamW, learning rate 0.001, weight decay 0.01, 100 epochs, and
seed 42.  The loss is L1 position plus 0.01 times the predicted acceleration
norm.  There is no test-time clipping, fitted transform, hyperparameter
selection, or ground-truth feature input.

Unity uses training batch 4096, microbatch 1024, shuffle seeded once per epoch,
and drop-last coverage 192512 of 193320 samples per epoch.  Every epoch
evaluates all 128880 validation samples in batches of 4096; the checkpoint
with minimum all15 validation MPJPE is selected.  Test evaluation uses all
64440 samples in batches of 256 and reports all15 and common13.

Ski uses training batch 4, drop-last coverage 144 of the 145 native-loader
samples per epoch, no validation split, and the fixed epoch-99 checkpoint.
Test evaluation uses all 30 samples in batches of 4 and reports common13.
For both datasets, each complete shuffled/evaluation batch is formed first,
then DLT and target are independently canonicalized by
`canonical_batch(dlt, dlt, target, dataset=...)`; microbatching happens only
after this step.

## Ski root-reference correction

The first Ski diagnostic treated calibrated-world DLT as though it shared the
target's translation reference.  It produced implausible MPJPE values of
8.197519 for DLT and 8.195732 for robust DLT.  Source inspection established
that the Ski labels are root-relative: original hip joint 0 and the midpoint
of original joints 1 and 4 are exactly zero throughout train and test.  Raw
calibrated-world DLT retains the skier trajectory.

The final Ski protocol therefore subtracts, independently in every predicted
frame, the predicted pelvis midpoint of common13 joints 4 and 5 before the
archived canonicalization.  This is a prediction-only translation.  It does
not change scale, use target data, estimate a test transform, or tune a
threshold.  Every final Ski result records:

- `ski_prediction_root_centered: true`
- `reference_root: midpoint_common13_joints_4_5_per_frame`

The superseded world-coordinate outputs remain preserved under
`results/ski/ski_world_coordinate_diagnostic/`; their training state is
preserved under `training/ski/ski_world_coordinate_diagnostic/dlt_mlp/`.
The corrected Ski training is separately versioned as
`training/ski/dlt_mlp_root_relative/`.

## Final results

| Dataset | Method | Joint set | MPJPE | Acceleration error | Samples |
|---|---|---:|---:|---:|---:|
| Unity | DLT | all15 | 0.198891 | 0.042314 | 64440 |
| Unity | DLT | common13 | 0.198315 | 0.037557 | 64440 |
| Unity | Robust DLT | all15 | 0.210878 | 0.033454 | 64440 |
| Unity | Robust DLT | common13 | 0.208759 | 0.029507 | 64440 |
| Unity | DLT residual MLP | all15 | 0.099574 | 0.045552 | 64440 |
| Unity | DLT residual MLP | common13 | 0.096377 | 0.040277 | 64440 |
| Ski | DLT | common13 | 0.398409 | 0.081605 | 30 |
| Ski | Robust DLT | common13 | 0.399647 | 0.084609 | 30 |
| Ski | DLT residual MLP | common13 | 0.164267 | 0.059798 | 30 |

Unity selected epoch 71, whose validation all15 MPJPE was 0.1378299226.
The selected checkpoint SHA256 is
`f58f5777f42b441939ae9c3638c4a26a2a83641567495d6685de6a5ff8c7dd03`.
Ski selected the fixed epoch 99 checkpoint with SHA256
`1d1a7eeb03f8a1e7fcc476a75f3c19a537f67ff2c0b8db416b58067e41ab720c`.

## Provenance and verification

Before every run, the controller verified each cache and geometry manifest,
every recorded input/output SHA256, exact shape, dtype, finite content, and
DLT/robust validity coverage.  All train, validation, and test geometry points
were valid; no point or sample was silently discarded.

Unity was trained with runner/model SHA256
`53597bf8c409715645ae13fa53699161d91fadfb875b51140d791d427842c691` /
`62dfd4c35514b032fee26779abd6ceb818b13fe88526e195e7904d84ba192c60`.
Those exact sources are frozen under
`source_at_training/dual2pose/{experiments,models}/`.

Corrected Ski was trained with runner/model SHA256
`5a044831da863dbfecf62869b86bbf8ed31a509e35a89d72416e133542d5f4c5` /
`77f8b57ab2d8f0608f2af9a4cb26b68bd7cbb46b27292334a053feddd009ae32`.
Those sources are frozen under
`source_at_training/ski_root_relative/dual2pose/{experiments,models}/`.

TDD covered exact zero-head identity, literal feature layout, synthetic
residual learning, nonfinite/negative rejection, point-weighted accumulation,
within-sequence acceleration, input checksum/finite validation, completed
result preservation, and Ski translation-trajectory invariance.  The final
focused suite passed 9/9 tests.  A real four-sample corrected Ski batch passed
forward, backward, and optimizer update with a finite loss of 0.21819447.

Exact commands:

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m unittest tests.test_run_geometry_baselines -v
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m py_compile dual2pose/models/dlt_residual_baseline.py dual2pose/experiments/run_geometry_baselines.py
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -u -m dual2pose.experiments.run_geometry_baselines --root logs/ivc_mmsports_extension/main_baselines/20260906 --dataset ski --mode smoke --device cpu --smoke-samples 4
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -u -m dual2pose.experiments.run_geometry_baselines --root logs/ivc_mmsports_extension/main_baselines/20260906 --dataset unity --mode evaluate --device cuda:0
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -u -m dual2pose.experiments.run_geometry_baselines --root logs/ivc_mmsports_extension/main_baselines/20260906 --dataset unity --mode train --device cuda:0 --microbatch 1024
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -u -m dual2pose.experiments.run_geometry_baselines --root logs/ivc_mmsports_extension/main_baselines/20260906 --dataset ski --mode evaluate --device cpu
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -u -m dual2pose.experiments.run_geometry_baselines --root logs/ivc_mmsports_extension/main_baselines/20260906 --dataset ski --mode train --device cpu
```

The final machine-readable outputs are
`results/{unity,ski}/{dlt,robust_dlt,dlt_mlp}.json`.  A read-only final audit
confirmed all six method names, finite metrics, exact test sample/point
denominators, 100 contiguous training epochs, per-epoch train/validation
coverage, selection rules, checkpoint hashes, source hashes, Ski root flags,
and existence of the superseded diagnostic archive.
