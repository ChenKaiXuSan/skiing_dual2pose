# dual2pose.eval

Evaluation code. The first group belongs to the fixed training protocol
(per-frame body frame, see `dual2pose/trainer/train_crossview_fusion.py`); the
"manuscript suite" reproduces the IVC/MMSports manuscript and keeps that
manuscript's batch-referenced first-frame canonicalization on purpose
(`canonicalize_pose_torch(mode="first_frame")`). The manuscript runs that
train models (`dual2pose/experiments/run_main_baselines.py`,
`run_gate_ablation.py`) are not the training protocol of this repository.

## Protocol evaluation

| Module | Role |
|---|---|
| `eval_unity.py` | `Trainer.test` of a per-frame checkpoint on the Unity test fold |
| `eval_ski_poseptz.py` | the same on the Ski-PosePTZ reference |
| `pa_mpjpe.py` | reflection-free PA-MPJPE used everywhere |
| `multiseed_metrics.py` | per-action summary written by `train_unity.py` |

The hand-held dual-Pocket pipeline lives in `dual2pose/pocket3/`, real-world
inference in `dual2pose/realworld/`.

## Manuscript suite (archived)

| Group | Modules |
|---|---|
| Main table and pose-only / geometric baselines | `main_baseline_data.py`, `main_baseline_geometry.py`, `evaluate_main_baseline_pa.py`, `render_main_baseline_tables.py`, `evaluate_gate_interventions.py` |
| Native-output audit | `native_output_protocol.py`, `native_output_frames.py`, `native_output_methods.py`, `native_output_manifest.py`, `native_output_canonfuse.py`, `evaluate_native_output_comparison.py`, `render_native_output_table.py` |
| External baselines (STRIDE, MetaPose, DeciWatch, efficiency) | `external_baseline_inputs.py`, `stride_external.py`, `metapose_inputs.py`, `metapose_supervision.py`, `evaluate_metapose.py`, `profile_external_efficiency.py`, `profile_fusion_efficiency.py` |
| View angle and sampling rate | `extension_experiment_utils.py`, `eval_unity_view_angle.py`, `view_angle_significance.py`, `run_unity_view_angle_significance.py`, `eval_unity_sampling_rate.py` |
| Masking and image occlusion | `eval_unity_masking.py`, `summarize_unity_masking.py`, `plot_alpha_vs_mask.py`, `eval_ski_poseptz_masking.py`, `image_occlusion.py`, `eval_unity_image_occlusion.py`, `run_unity_image_occlusion_frontend.py`, `run_matched_occlusion_frontend.py`, `evaluate_occluded_geometry.py` |
| N-view | `nview_protocol.py`, `nview_loading.py`, `nview_acceptance.py`, `eval_unity_nview.py` |
| Temporal alignment and offset | `temporal_alignment.py`, `calibrate_temporal_alignment.py`, `eval_unity_temporal_alignment.py`, `eval_unity_temporal_offset.py` |
| Pose front ends and adaptation | `frontend_lifters.py`, `frontend_manifest.py`, `frontend_comparison.py`, `export_unity_frontend_predictions.py`, `eval_unity_frontend_generalization.py`, `run_unity_frontend_suite.py`, `summarize_frontend_adaptation.py`, `summarize_multiseed_crossfold.py` |
| Figures and tables | `journal_figure_style.py`, `render_e4_e5_artifacts.py` |
