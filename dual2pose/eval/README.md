# dual2pose.eval

Evaluation code. Active CanonFuse3D evaluation uses the fixed per-frame body
frame (see `dual2pose/trainer/train_crossview_fusion.py`). Archived baseline
and native-output analysis explicitly opt into the manuscript's batch-referenced
first-frame construction (`canonicalize_pose_torch(mode="first_frame")`);
the shared function now defaults to `per_frame`. The manuscript runs that
train models (`dual2pose/experiments/run_main_baselines.py`,
`run_gate_ablation.py`) are not the training protocol of this repository.

## Protocol evaluation

| Module | Role |
|---|---|
| `eval_unity.py` | `Trainer.test` of a per-frame checkpoint on the Unity test fold |
| `eval_ski_poseptz.py` | the same on the Ski-PosePTZ reference |
| `eval_unity_nview.py` | per-frame checkpoint evaluation with pairwise N-view composition |
| `eval_unity_temporal_alignment.py` | per-frame checkpoint evaluation with temporal corrections |
| `pa_mpjpe.py` | reflection-free PA-MPJPE used everywhere |
| `multiseed_metrics.py` | per-action summary written by `train_unity.py` |

The hand-held dual-Pocket pipeline lives in `dual2pose/pocket3/`, real-world
inference in `dual2pose/realworld/`.

Pocket3, direct real-world inference, N-view and temporal-alignment loaders share
the protocol guard in `dual2pose/training_protocol.py`. These current loaders
reject untagged and first-frame weights; historical results require their
archived code and checkpoint protocol. Pocket3 averages use the same per-frame
inputs as the learned model.

Active primary scores use `common13_direct_per_frame_v1` from `pose_metrics.py`:
select common-13 and compare the output directly with per-frame canonical GT.
Do not canonicalize or fit a transform to the output for the primary score.
Validation and checkpoint selection follow this contract. Aligned/recanonicalized
scores and explicitly requested `all15` front-end scores are auxiliary.

## Extension evaluation and archived analyses

The extension evaluator entry points using the current trainer (masking, view
angle, sampling rate, temporal offset, image occlusion and front-end comparison)
also report common-13 primary scores. Their saved manuscript results are historical
and have not been regenerated. The explicitly archived baseline/native-output
analyses retain their own first-frame protocol.

| Group | Modules |
|---|---|
| Main table and pose-only / geometric baselines | `main_baseline_data.py`, `main_baseline_geometry.py`, `evaluate_main_baseline_pa.py`, `render_main_baseline_tables.py`, `evaluate_gate_interventions.py` |
| Native-output audit | `native_output_protocol.py`, `native_output_frames.py`, `native_output_methods.py`, `native_output_manifest.py`, `native_output_canonfuse.py`, `evaluate_native_output_comparison.py`, `render_native_output_table.py` |
| External baselines (STRIDE, MetaPose, DeciWatch, efficiency) | `external_baseline_inputs.py`, `stride_external.py`, `metapose_inputs.py`, `metapose_supervision.py`, `evaluate_metapose.py`, `profile_external_efficiency.py`, `profile_fusion_efficiency.py` |
| View angle and sampling rate | `extension_experiment_utils.py`, `eval_unity_view_angle.py`, `view_angle_significance.py`, `run_unity_view_angle_significance.py`, `eval_unity_sampling_rate.py` |
| Masking and image occlusion | `eval_unity_masking.py`, `summarize_unity_masking.py`, `plot_alpha_vs_mask.py`, `eval_ski_poseptz_masking.py`, `image_occlusion.py`, `eval_unity_image_occlusion.py`, `run_unity_image_occlusion_frontend.py`, `run_matched_occlusion_frontend.py`, `evaluate_occluded_geometry.py` |
| N-view helpers and archived acceptance | `nview_protocol.py`, `nview_loading.py`, `nview_acceptance.py` |
| Temporal helpers and offset | `temporal_alignment.py`, `calibrate_temporal_alignment.py`, `eval_unity_temporal_offset.py` |
| Pose front ends and adaptation | `frontend_lifters.py`, `frontend_manifest.py`, `frontend_comparison.py`, `export_unity_frontend_predictions.py`, `eval_unity_frontend_generalization.py`, `run_unity_frontend_suite.py`, `summarize_frontend_adaptation.py`, `summarize_multiseed_crossfold.py` |
| Figures and tables | `journal_figure_style.py`, `render_e4_e5_artifacts.py` |
