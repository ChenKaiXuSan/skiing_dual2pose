# Pegasus runtime verification — 2026-10-30

> Historical snapshot: branch names and preservation instructions below apply to the recorded operation. Since 2026-11-03, only `master` is retained; see `docs/branch_management.md` for the current policy.

Account **HP260146**, NQSV queue **gen_S**. GPU smoke job **195112.nqsv** completed successfully on an NVIDIA H100 PCIe. Scheduler elapsed time was 241 seconds. The diagnostic ran against code commit `ba98aad9362596abc55b68392372e3ccc76da119` on `per-frame-canonicalization`.

## Workspace

Under `/work/HP260146/chenkaixu/`:

- `Skiing_Canonical_DualView_3D_Pose_PyTorch/`: current branch, original paper and historical results.
- `Skiing_Canonical_DualView_3D_Pose_PyTorch_data/`: datasets with their original internal structure.
- `Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime/`: isolated environments, fresh outputs, setup logs and migration archives.

The original `migration_packages`, `migration_reports`, and `migration_full` addresses remain compatibility symlinks to runtime `archives/`. Existing unrelated projects were not reorganized. Differing code from before synchronization is preserved under runtime `setup/code_before_ba98aad/`; Git also retains `refs/migration-backup/before-runtime-20261030`.

## Verified scope

| Check | Evidence |
|---|---|
| PyTorch CUDA | 2.11.0+cu126; actual CUDA backward and synchronize passed |
| Unity | One training, validation and test batch; saved last/best checkpoints and test metrics |
| Ski | One training and test batch; saved final checkpoint and test metrics |
| Pocket3 | Both temporal directions; frozen, fine-tuned and scratch evaluations; one training epoch; four new weights files |
| MetaPose | TensorFlow 2.15.1 GPU; four Ski training frames and three updates; finite losses; parameters changed; HDF5 reload reproduced inference |

Both Python 3.11 environments passed `pip check`. Resolved packages are recorded in runtime `setup/torch.freeze.txt` and `setup/metapose.freeze.txt`. Existing shared conda environments were not modified.

TensorFlow logged duplicate cuDNN/cuFFT/cuBLAS registration messages at startup; GPU execution, training and reload assertions nevertheless passed. Lightning also reported low-worker performance warnings for deliberately small diagnostics. No smoke subprocess failed. Independent artifact checks confirmed finite metrics and checkpoint tensors, one optimizer step and `per_frame` metadata in both Unity/Ski checkpoints. All 1,289 manuscript files matched their pre-setup SHA256 values.

Local validation: the protocol/workflow/MetaPose test group passed **67 tests**, with **3 skipped** official-TensorFlow tests in the local environment. Runtime/Pocket3/protocol checks passed **43 tests**. These groups overlap and should not be summed as unique tests.

## Evidence and usage

Remote diagnostic output:

```text
/work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime/runs/smoke_20261030T021538Z_0:195112.nqsv/
```

This directory contains `status.json`, per-step logs, checkpoints and evaluation output. Additional evidence is under runtime `setup/`: `code_sync.json`, `artifact_verification.json`, `smoke.stdout`, `smoke.stderr`, and package freezes. Local copies of verification records are under `logs/pegasus_setup_20261030/` (ignored by Git).

To submit a new diagnostic with unique output and scheduler logs:

```bash
cd /work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch
bash pegasus/runtime/submit_smoke.sh
```

Source `pegasus/runtime/env.sh` for standalone commands; use the exported interpreter and new output directories. Do not resume workstation-specific state manifests without preparing new manifests with the relocated paths and provenance. The runtime setup does not rewrite archived manifests.

## Limits

These checks establish that the tested cached-pose training/evaluation entrypoints run in the migrated environment. They do not establish full-run convergence or paper-level reproduction. MetaPose's check covers training and checkpoint inference, not full held-out 3D rescoring. RGB/SAM3D inference and complete STRIDE/DeciWatch training were not run. Diagnostic outputs are explicitly marked ineligible for publication. No full experiment rerun was launched by this setup task.

Only `per-frame-canonicalization` is pushed. The local and GitHub `master` reference remains `f3ab2bc8d7f43097495e62c504458fa9c34195b0`. Manuscript files and historical experiment outputs are excluded from the setup writes.
