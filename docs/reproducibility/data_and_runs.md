# Data, weights and run provenance

The final result registry is [final_20261103](final_20261103/README.md). Position errors are mm, common-13 direct per-frame scoring is primary, and PA alignment is auxiliary. Preserve raw coordinate units and convert only in reporting. Do not recanonicalize model outputs before the primary metric.

## Inputs supplied separately

The corresponding author (chenkaixusan@gmail.com) has confirmed availability upon reasonable request of the Unity data, selected weights, hand-held per-frame pose streams, external-camera 3D reference, synchronization metadata and evaluation outputs. Author-collected videos are not shared. Obtain Ski-PTZ-Pose through the dataset provider. Request the exact immutable run archive and split/index manifests together with the data: a fresh random split does not reproduce the paper.

| Asset | Required interface and verification |
| --- | --- |
| Unity | Both action-disjoint fold indices and metadata consumed by `dual2pose.training_protocol`; the reported run uses fold 0; raw pose/image/calibration inputs and capture metadata. Keep the recorded evaluator batch size (256). |
| Ski | Official train/test partition, native 13-joint reference and prepared monocular poses. Preserve the final-epoch selection rule. |
| Pocket | Chronological window split, left/right frame alignment, 3D reference and reliability masks; preserve within-session versus frozen/short-adaptation cohorts. |
| Model weights | Match SHA-256 in `final_20261103/checkpoint_registry.json`; per-frame protocol tags are required. Untagged legacy weights must not be relabelled. |
| Cached inputs | Include cache manifests, shapes, source hashes and data-root mapping; do not reuse pre-repair Unity calibration caches. |
| External baselines | STRIDE `84200692c8a445972b59afefee687c552aabe082`; DeciWatch `5afe2d67005441e2785a8d374d31a91f046a0da6`; google-research/MetaPose `08a8d6736475776f42ffac23b2c13111a28e5795`; acquire their external weights separately. |

These declarations describe what to obtain; the source package does not silently download private data or gated weights.

## Base plan versus final runs

`python -m pegasus.runtime.chain prepare --repo <checkout> --runtime <runtime> --plan <new-plan>` creates the original 38-training-configuration base DAG without submitting jobs. The three versioned seed manifests under `pegasus/runtime/manifests/` permit preparation without private `logs/`; their hashes are checked. Supplied original archived manifests retain precedence for historical replay. Each run writes into a new directory and freezes source hashes.

The base plan is not the entire final experiment provenance. Final results also use recovery/frontend cache successors, stable Unity MetaPose, corrected Unity geometry (`unity_geometry_v2_20261101T062040Z`) and the unified reevaluation. Use the exact frozen sources, configuration, receipt dependencies and selected checkpoint hashes from those archives. The stable MetaPose training/export driver and versioned geometry evaluators are present in the development tree, but a single base-plan invocation must not be described as reproducing every final row.

Main Unity and C07 use identical selected fusion and DLT+MLP weights; their clean observation cohorts differ. Separate ablations and adapters intentionally train separate models. All reported H100 timings retain their method-specific scopes.

## Public source boundary

`python tools/export_source.py --output /absolute/new/canonfuse3d-source.tar.gz` exports core source, tests, configuration and current release documentation with SHA-256 hashes. It excludes Git history, manuscripts, Unity assets, vendored SAM3D, results, data and weights. Existing outputs are never overwritten. Review `THIRD_PARTY.md` before publishing; making the original repository public exposes historical branches/files too.
