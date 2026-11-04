# Post-Estimation Fusion of Uncalibrated Dual-View 3D Pose Streams

Post-estimation fusion of uncalibrated dual-view 3D pose streams, with per-frame body coordinates, joint-wise reliability gating and a residual correction.

This repository is public. `master` is the sole maintained branch and contains the consolidated per-frame method. The former `per-frame-canonicalization` and temporary integration branches have been retired after merging. Pre-integration snapshots remain available through archive tags. Each experiment must record its exact frozen code and checkpoint hashes rather than relying on a moving branch name.

## Final experimental protocol

- Canonicalize each input stream and target independently for every frame. The orthonormal body frame preserves scale and bone lengths while removing global orientation and translation.
- Score the predicted output directly against the canonical target on the common 13 joints. Do not fit, recenter or recanonicalize predictions for the primary metric.
- Report position errors in **mm**. PA-MPJPE is an auxiliary alignment metric; acceleration is **mm per sampled-frame squared**.
- Main Unity and matched C07 evaluations use the same selected fusion and DLT+MLP weights; differing observation cohorts explain their different clean scores.
- Independently trained ablations and adapted models retain their own checkpoint identities. The final Unity/Ski comparisons are fixed-partition, single-training-run results, not multi-seed generalization estimates.

| Proposed method | Unity | Ski-PTZ-Pose |
| --- | ---: | ---: |
| MPJPE (mm) | 50.91 | 124.33 |
| PA-MPJPE (mm) | 41.07 | 91.49 |

The final numerical and checkpoint registry is preserved separately from the public code checkout and is available upon reasonable request. It records full-precision results and provenance. Historical manuscript outputs in Git history are not current result sources.

## Install and check

Python 3.11 is the tested release environment. Start in a new environment:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# CPU checks; use a compatible torch/torchvision CUDA build for GPU training.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r tests/requirements.txt
python -m pytest -q
python -m build
```

The wheel and source distribution include the Hydra YAML and model configuration groups. Optional raw-image frontends and the TensorFlow MetaPose environment are installed separately. Some legacy manuscript integration tests depend on separately archived paper files; the public checkout does not include those files.

## Data and training

The code consumes prepared 3D pose streams; it does not include private datasets or gated model weights. The author has confirmed that the Unity dataset, selected checkpoints, derived laboratory poses/reference/synchronization information and evaluation outputs are available upon reasonable request. Raw author-collected videos are not shared. Contact: chenkaixusan@gmail.com.

Set the dataset roots before running:

```bash
export DUAL2POSE_UNITY_ROOT=/your/data/skiing_unity_dataset
export DUAL2POSE_SKI_ROOT=/your/data/Ski-PosePTZ-CameraDataset-png
export DUAL2POSE_POCKET3_ROOT=/your/data/skiing_pocket3_20261012
export DUAL2POSE_RAW_ROOT=/your/data
python -m dual2pose.train_unity
python -m dual2pose.train_ski_poseptz --help
python -m dual2pose.pocket3.evaluate --help
```

Use the supplied fixed split/index metadata; generating a different split does not reproduce the paper. Unity supports two action-disjoint folds, with the reported result using fold 0. Preserve the recorded Unity evaluation batch size (256). Active loaders reject untagged and first-frame checkpoints instead of treating them as per-frame results.

Full reproduction also requires the separately preserved data contracts, external baseline revisions, final run manifests and immutable experiment archives. Request these materials together with the data and checkpoints. The bundled Pegasus recipes allow preparing the original base plan without private log manifests; they do not replace later recovery, stable MetaPose, geometry correction and unified reevaluation records. Preparing a plan never submits GPU jobs.

## Repository layout

- `dual2pose/`: model, canonicalization, training, evaluation and baseline adapters.
- `configs/`: packaged Hydra configuration and model groups.
- `cross_validation/`: index generation and split support.
- `pegasus/`: NQSV orchestration, verified receipts and versioned base recipes.
- `tests/`: CPU regression and distribution tests.
- `tools/export_source.py`: allowlisted, history-free core source exporter.

Manuscripts (`paper/`) and supporting documentation (`docs/`) are maintained separately and are no longer tracked in the current branch. Existing local copies are preserved; a fresh clone does not include these directories.

## Release boundary

```bash
python tools/export_source.py --output /absolute/new/dual-view-pose-source.tar.gz
```

The exporter refuses existing output paths and includes a SHA-256 manifest. It excludes Git history, manuscripts, Unity assets, vendored SAM3D frontend code, private results, datasets and weights. This produces a core source archive, not a complete experiment archive.

The public repository retains its existing Git history, including historical manuscripts and documentation. Untracking `paper/` and `docs/` removes them from the current checkout, not from earlier commits or archive tags. Unity assets and vendored frontend code remain in the repository and retain their applicable third-party terms.

No fresh full-GPU replication is implied by the CPU tests or packaging checks. Reproduce published numbers using the exact recorded inputs, selected checkpoints, configurations and frozen source hashes.

## License and citation

This project's core code uses [Apache-2.0](LICENSE). External software, model weights, datasets and Unity assets retain their own terms; see [third-party boundaries](THIRD_PARTY.md). The root license does not authorize redistributing all materials in the historical repository.

The manuscript, *Post-Estimation Fusion of Uncalibrated Dual-View 3D Pose Streams*, has been submitted to IEEE Access. Publication identifiers will be added when available; submission does not imply acceptance or publication.
