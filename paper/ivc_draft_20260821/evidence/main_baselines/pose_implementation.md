# Pose-only main baseline implementation

Date: 2026-09-06

## Scope and interface

Implemented `dual2pose/models/main_baselines.py` and focused tests in
`tests/test_main_baselines.py`. No archived result, checkpoint, raw-data,
training-controller, geometry, or manuscript file was changed.

Public interfaces:

- `aligned_average(left, right)`
- `quality_weighted_fusion(left, right)`
- `build_baseline(name, num_joints, time_window=30)`, where `name` is `mlp`,
  `tcn`, or `smoothnet`

All inputs and outputs use `(B, T, J, 3)`. Learned-model forward methods accept
exactly `left` and `right`; the module has no target, confidence, image, 2-D,
or calibration input.

## Implemented methods

### Aligned average

The method estimates one repository `Sim(3)` transform per batch item using
all `T * J` right/left correspondences, maps the complete right sequence into
the left coordinate frame, and takes the equal average. It calls the existing
`dual2pose.models.sim3.align_points_sim3` implementation. Its regularized
source-variance path keeps constant and all-zero sequences finite.

### 3-D quality-weighted fusion

Each view receives one sequence-level cost composed only of:

1. mean second-difference magnitude, normalized by the sequence's median bone
   length; and
2. mean absolute temporal bone-length deviation from the per-bone temporal
   median, normalized by the same body scale.

The two fusion weights are `softmax(-cost)` and sum to one. This calculation
does not use the other view when scoring a view, and does not use ground truth
or external confidence.

The 15-joint graph exactly follows `FILTER_SKELETON_CONNECTIONS` in
`dual2pose/map_config.py`. The 13-joint graph is that graph after removing the
two eyes and reindexing the common subset:

- left arm: neck-shoulder-elbow-hand
- right arm: neck-shoulder-elbow-hand
- trunk/legs: neck-left/right hip, then hip-knee-foot on each side

Only the documented 13- and 15-joint orders are accepted by this function.

### Residual cross-view MLP

For each frame, both complete poses are flattened and concatenated (`6J`
features). Two 256-wide ReLU layers with dropout 0.1 predict a `3J` residual
over the equal view average. Joint coordinates are therefore processed jointly,
not with a shared per-joint MLP.

### Temporal convolution fusion

Both complete poses are concatenated as `6J` channels. Three same-length
temporal convolutions use kernel size 5; the first two have 128 channels, ReLU,
and dropout 0.1, and the final convolution predicts `3J` channels. The output
is a residual over the equal view average.

### SmoothNet

Primary sources inspected on 2026-09-06:

- official model: https://github.com/cure-lab/SmoothNet/blob/c03e93e/lib/models/smoothnet.py
- official H36M-FCN 3-D configuration: https://github.com/cure-lab/SmoothNet/blob/c03e93e/configs/h36m_fcn_3D.yaml
- official loss: https://github.com/cure-lab/SmoothNet/blob/c03e93e/lib/core/loss.py

This is a faithful port of the official temporal-only topology, applied to the
canonical equal average after conversion from `(B,T,J,3)` to `(B,3J,T)`. It
retains the H36M-FCN 3-D widths: temporal encoder `T -> 512`, five residual
blocks `512 -> 128 -> 512`, LeakyReLU slopes 0.1/0.2, in-place dropout 0.25,
and decoder `512 -> T`. The explicit IVC adaptation is `T=30` for both encoder
input and decoder output. There is no extra residual connection around the
complete SmoothNet because the official network predicts the refined sequence
directly.

The controller should train this row with the official motion-aware loss:

`L = 1.0 * mean_L1(position, target) + 0.1 * mean_L1(second_difference(position), second_difference(target))`.

The MLP and TCN have no architecture-specific loss in this implementation and
can use the controller's common supervised position objective. Dropout remains
active during training and is disabled by normal PyTorch `eval()` behavior.

## Resource estimates

Exact trainable parameter counts and FP32 storage estimates:

| Model | J=13 params | J=15 params | J=15 FP32 parameters | J=15 parameters + gradients + AdamW moments |
|---|---:|---:|---:|---:|
| MLP | 96,039 | 100,653 | 0.403 MB | 1.610 MB |
| TCN | 157,095 | 168,621 | 0.674 MB | 2.698 MB |
| SmoothNet | 689,822 | 689,822 | 2.759 MB | 11.037 MB |

Using PyTorch saved-tensor hooks on CPU with FP32, `T=30`, and `J=15`, the
measured autograd-saved tensors scale by approximately 0.201 MB/sample for MLP,
0.108 MB/sample for TCN, and 1.923 MB/sample for SmoothNet. Linear extrapolation
to a 4,096-sequence microbatch is 0.821 GB, 0.444 GB, and 7.877 GB respectively.
These figures exclude CUDA workspace, allocator overhead, dataloader tensors,
and other controller state. SmoothNet may therefore need a smaller microbatch
with gradient accumulation to preserve the approved effective batch size.

## TDD and verification evidence

Environment:

- interpreter: `/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python`
- PyTorch: `2.11.0+cu126`

RED command, before implementation:

```text
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m unittest tests.test_main_baselines -v
```

Expected failure observed: `ModuleNotFoundError: No module named
'dual2pose.models.main_baselines'`; 1 loader error, exit code 1.

GREEN command after implementation: the same command completed with 8 tests,
0 failures/errors, exit code 0 in 0.652 s. Covered behaviors:

- synthetic sequence-level similarity-transform recovery;
- finite all-zero Sim(3) degeneracy;
- reduced weight for a view with temporal jitter and changing bone lengths;
- unchanged identical static and all-zero 13/15-joint poses;
- output shapes, finite outputs, nonzero gradients to both views, and finite
  parameter gradients for MLP, TCN, and SmoothNet;
- deterministic evaluation-mode outputs with dropout disabled;
- rejection of target arguments, wrong temporal shape, and unknown names; and
- numerical forward parity with an independently transcribed official
  SmoothNet reference using identical weights.

Compile check:

```text
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m py_compile \
  dual2pose/models/main_baselines.py tests/test_main_baselines.py
```

Result: exit code 0 with no output.

## Integration note

The controller can import all three interfaces now. `dlt_mlp` is deliberately
absent because calibrated DLT features and their validity contract belong to
the separate geometry/controller task.
