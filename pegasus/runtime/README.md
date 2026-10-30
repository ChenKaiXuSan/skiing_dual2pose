# Pegasus per-frame runtime

Project account: **HP260146**. Queue: **gen_S** (confirmed by the user). Do not substitute the historical SSR/SKIING accounts or the gpu queue.

```text
/work/HP260146/chenkaixu/
├── Skiing_Canonical_DualView_3D_Pose_PyTorch/       # current branch, paper, historical logs
├── Skiing_Canonical_DualView_3D_Pose_PyTorch_data/  # original dataset structure
└── Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime/
    ├── envs/{torch,metapose}/
    ├── runs/                                    # new run outputs only
    ├── setup/                                   # install, scheduler and verification logs
    ├── cache/
    └── archives/{packages,reports,full}/          # migrated audit material; original paths remain aliases
```

Source `pegasus/runtime/env.sh` before running commands. PyTorch 2.11.0+cu126 and TensorFlow 2.15.1 live in separate Python 3.11 virtual environments. `install.sh torch` and `install.sh metapose` install pinned direct dependencies; resolved versions are saved under runtime/setup. The interpreter comes from the existing Python 3.11 installation and does not modify that environment. OpenCV has one headless 4.x installation, rather than conflicting 4.x/5.x packages.

```bash
cd /work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch
source pegasus/runtime/env.sh
qsub -A HP260146 -q gen_S \
  -o "$DUAL2POSE_RUNTIME/setup/smoke.stdout" \
  -e "$DUAL2POSE_RUNTIME/setup/smoke.stderr" pegasus/runtime/smoke.pbs
```

The job creates a unique runtime/runs directory and refuses an existing output path. It checks CUDA backward execution, Unity and Ski training plus test entrypoints with one batch, Pocket3 training/validation/test with a one-epoch budget, and MetaPose training-frame updates plus HDF5 reload/inference. Each command has its own log and status. These are runtime diagnostics, **not scientific results**; one batch/epoch does not establish reproduction accuracy. RGB-to-SAM3D inference and every external baseline's full training are not covered by this smoke job.

Fresh production commands must also set explicit new output directories. Preserve archived manifests and their original path/hash provenance; do not restart old workstation-specific workflow state files. Pocket3 supports `DUAL2POSE_POCKET3_ROOT` or `--dataset-root`. Hydra datasets use the Unity/Ski variables in env.sh and map archived index paths without modifying the index JSON.

Do not edit or regenerate manuscript files as part of runtime setup. Do not overwrite historical results. Old migration archives and conflict copies remain available; inspect the migration conflict report before choosing a newer paper version.
