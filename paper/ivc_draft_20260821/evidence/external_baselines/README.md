# External-baseline evidence archive

Status: complete and verified on 2026-09-13.

This package freezes the final admitted external-baseline evidence used by the IVC manuscript. It contains eight compact result files and fourteen provenance files copied byte-for-byte from the experiment log tree.

## Admitted canonical-space matrix

| Method | Unity MPJPE / PA-MPJPE / Accel. | Ski MPJPE / PA-MPJPE / Accel. |
|---|---:|---:|
| Avg. + STRIDE (adapted) | 0.2799 / 0.1174 / 0.0316 | 0.3483 / 0.1861 / 0.0387 |
| DeciWatch (official-protocol adaptation) | 0.1134 / 0.0695 / 0.0132 | 0.2239 / 0.1549 / 0.0168 |
| MetaPose (adapted) | 0.2655 / 0.1089 / 0.0400 | 0.4253 / 0.1425 / 0.0634 |

All Unity values cover 64,440 test windows; all Ski-PTZ-Pose values cover 30 test windows. The method budgets differ and remain separated in the manuscript: STRIDE refines the dense canonical average, DeciWatch recovers 30 frames from four observed source frames, and MetaPose additionally uses estimated 2D joints.

The native-output score files are archived separately. Unity contains 13 measured and four protocol-unavailable methods. All 17 Ski native-space methods are unavailable because there is no independently supplied target mapping into those declared output frames; these are protocol exclusions, not unfinished experiments.

## Multi-seed cross-fold decision

The proposed uncertainty matrix was two legacy action folds times three seeds (13, 42, and 73), for six independent 100-epoch CanonFuse3D trainings. It was intended to quantify training and fold sensitivity, not to add a baseline or an external comparison.

That matrix is retired from the current submission scope. The only artifacts are two fold-1/seed-13 launch-smoke logs; both stop during data loading on missing SAM3D frame files before any epoch, checkpoint, or result. Accordingly, this manuscript makes no multi-seed mean, standard-deviation, confidence-interval, or cross-fold performance claim.

## Verification

From the repository root, run:

    python3 paper/ivc_draft_20260821/evidence/external_baselines/verify_archive.py

The verifier checks archive/source SHA-256 identity, all six exact result triplets and coverage counts, zero-control deltas where recorded, MetaPose completion states, canonical and native manuscript table rows, citation keys, protocol-separation language, LaTeX diagnostics, table pages, and the retired multi-seed status. It regenerates manifest.json and verification.json.
