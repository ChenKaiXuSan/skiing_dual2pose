# Reproduction inventory — 2026-11-01

This is a dated, read-only audit of the completed Pegasus run, not a claim that
a fresh public clone reproduces every conclusion. The [original geometry audit](table2_audit_20261101.md) is retained; its required reruns are now complete, as recorded in the [repair report](../experiments/unity_geometry_repair_20261101.md).

## What is complete

The live check at 2026-11-01 05:23 UTC found 220/220 plan jobs with complete
receipts, no missing receipts, and a complete dispatcher. The latest stable
Unity MetaPose run is an additional run, retained separately from those 220 jobs.

- 508 expected result artifacts exist and match their SHA-256 receipts.
- Three frozen source manifests match all 139, 143 and 144 listed files;
  the latest MetaPose run's five explicitly pinned source files also match.
- All 25 arrays across Unity train/val/test and Ski train/test raw pose caches
  match their manifests.
- Eight selected main/TCN/SmoothNet/DLT-residual checkpoints match their recorded
  hashes. Latest MetaPose stage weights are covered by the artifact inventory.
- No broken symlinks were found in the inspected runtime plan trees.
- Unity, Ski and Pocket3 data roots, external repositories and environment
  records are present. This check did not reread every RGB file in the datasets.
- No jobs were listed by `qstat` at that observation.

The original migration and later run receipts remain the source of truth for
files outside this scoped recheck. No new full GPU training or evaluation was
executed by this audit.

Local storage has 776 embedded text evidence artifacts with verified source
hashes (CRLF is restored where transport normalized line endings), plus the
latest consolidated tables. It is **not a complete binary replica**: of 49
selected binary artifacts, 47 were not found by size/SHA-256 under local
`logs/` and `reports/`, even allowing renamed copies. These 47 total 933,001,599
bytes, about 0.87 GiB. This is a lower-bound backup list, not the size of the
entire experiment or a whole-disk absence claim.

## Pegasus layout and dependencies

Under the project allocation, keep these siblings together:

```text
<project>/
├── Skiing_Canonical_DualView_3D_Pose_PyTorch/          # development code and archived inputs
├── Skiing_Canonical_DualView_3D_Pose_PyTorch_data/     # Unity, Ski, Pocket3 data
└── Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime/
    ├── plans/                                      # frozen source, configs, receipts, outputs
    ├── envs/                                       # Torch and isolated TensorFlow environments
    └── setup/                                      # package freezes and migration/runtime records
```

The successful chain spans `full_chain_20261030_final_v2`,
`recovery_20261031_v2` and `frontend_recovery_20261031_v2`; the latest Unity
MetaPose run is `metapose_unity_stable_20261031T222817_v3`. Successor plans reuse
predecessor artifacts, so keep their dependencies and symlinks together. Do not
delete older plan directories simply because a successor is complete.

The two environment launchers currently resolve to a Python 3.11 installation
in the Pegasus account's home directory. They work independently of the local
workstation, but copying the project directory alone is **not** a relocatable
environment backup. Preserve package freezes and recreate the interpreters on a
new host. `pegasus/runtime/requirements-main.txt` and
`requirements-metapose.txt` describe separate dependency sets; they must not be
merged into one environment. GPU/CUDA compatibility must be checked on the target.

Verified external source revisions:

| Repository | Revision |
| --- | --- |
| STRIDE | `84200692c8a445972b59afefee687c552aabe082` |
| DeciWatch | `5afe2d67005441e2785a8d374d31a91f046a0da6` |
| google-research / MetaPose | `08a8d6736475776f42ffac23b2c13111a28e5795` |

External model weights and frontend assets are separate from these repositories.
`SAM3Dbody/UPSTREAM.md` records the locally patched frontend. Dataset permissions
and gated weights must be handled separately in any public release.

## Reproducing records versus validating conclusions

Use each run's frozen source, configuration, split/index manifest, selected
checkpoint, dependency receipts and environment record to reproduce its saved
numbers. A development branch name alone does not identify these inputs.
The development branch now consolidates the frontend cache/recovery, MetaPose stability, and corrected Unity geometry changes. A commit alone still does not identify a past run's external weights, caches, dependency receipts or frozen source snapshot. Synchronization records preserve the previous Pegasus development commit and any displaced files; they do not rewrite historical run provenance.

The corrected Unity geometry plan `unity_geometry_v2_20261101T062040Z` completed 18/18 tasks. Its 86 artifacts and 332 source hashes passed verification. Both new MetaPose Unity timing scopes are available (2.915 ms learned solver; 10.070 ms including adapters, on H100 PCIe). These are method-specific timings and must not be ranked directly against timings measured on RTX A6000.

The current local/Pegasus Access manuscript was synchronized separately: the new Unity geometry rows, matched corruption/calibration matrix and MetaPose timing were adopted. Other benchmark numbers intentionally retain their original runs. The C07 fusion control uses an independently retrained checkpoint (clean 50.967 mm), while the primary table retains 50.827 mm. Run identifiers and hashes, rather than matching rounded numbers, establish provenance.

The earlier 47-file local binary-backup gap above has not been re-audited by code synchronization. A full public reproduction still needs portable data/weight acquisition, environment setup, licensing review and clean-clone validation; see [release readiness](release_readiness.md).

## Local audit artifacts

The workspace-only evidence is under
`reports/table2_followup_audit_20261101/`: `README.md`, train-only geometry probes,
test logs, paper provenance, live Pegasus inventories, and `local_replica.json`.
The latest numerical tables remain under `reports/pegasus_final_audit_20261101/`;
their geometry-validity interpretation is superseded by this audit.
