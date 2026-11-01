# Documentation index

## Current protocol and audit

- [Unity geometry repair and additive rerun](experiments/unity_geometry_repair_20261101.md)

- [Reproduction inventory and material layout](reproducibility/README.md)
- [Table 2 implementation audit and required reruns, 2026-11-01](reproducibility/table2_audit_20261101.md)
- [Public release readiness](reproducibility/release_readiness.md)
- [Branch roles and historical cleanup observations](branch_management.md)
- [Per-frame experiment plan](per_frame_rerun_plan_20261028.md)

The current audit takes precedence over older claims of completeness or geometry
correctness. Dated runbooks describe the snapshots they inspected; a completed
task or a passing synthetic test is not proof that the scientific protocol is
correct.

## Source layout

| Directory | Role |
| --- | --- |
| `dual2pose/models`, `trainer`, `dataloader` | Models, canonicalization, loaders and training |
| `dual2pose/experiments`, `eval` | Baseline adaptations, evaluation and evidence generation |
| `configs`, `cross_validation` | Configuration and split construction |
| `pegasus/runtime` | Cluster environment, frozen run plans and dependency orchestration |
| `tests` | Regression tests; external frameworks and GPU checks have separate requirements |
| `SAM3Dbody` | Patched external frontend and provenance |
| `unity` | Capture code plus separately reviewable project assets |

Manuscripts, experiment outputs, data and checkpoints have separate preservation
requirements. They are not moved or deleted as part of documentation cleanup.
