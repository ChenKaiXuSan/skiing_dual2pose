# dual2pose.experiments

Archived runners of the IVC/MMSports manuscript: external baselines
(DeciWatch, STRIDE, MetaPose, geometric DLT variants), the main-table training
of the pose-only baselines, the gate/output-residual ablation, the front-end
adaptation matrix and the 3-seed x 2-fold matrix. They reproduce the
manuscript's batch-referenced first-frame protocol through
`dual2pose/eval/main_baseline_data.py` and are not part of the fixed training
protocol; the training entry point is `dual2pose/train_unity.py`.
