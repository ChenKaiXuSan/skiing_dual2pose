# IVC extension evidence manifest

- Checkpoint SHA-256: `869a2217f8676c0ada75ed3c9a3c82a9b8efbb105749f6ffb8bef71e9172f50f`
- Fold: `0`
- Seed: `42`
- Complete test camera-pair sequences: `64440`
- Unity evaluation batch size: `256`; the executable first-frame transform is
  batch-referenced, so this is part of the numerical protocol.
- Perturbation metrics use all 15 joints in dataset coordinate units after the batch-referenced transform.
- Front-end metrics use the shared common-13 subset in dataset coordinate units after the batch-referenced transform.

- Ski-PTZ-Pose checkpoint SHA-256: `2b985d4f8b5e1f083fa549ae1cfa524c7d48a8fa2452fbcb136f2f514d1ed78d` (`last.ckpt`, epoch 97).
- Ski-PTZ-Pose fixed-index SHA-256: `967a17ad2edfa8f56801d9a7cdd9c683dc73d0072660298ff65294445b41ba02`.
- Ski-PTZ-Pose protocol: `30` fixed test entries, common-13 joints, 30 frames, batch size 4, `drop_last=False`, dataset coordinate units after the batch-referenced transform.
- The Ski index contains 150 train, 0 validation, and 30 test entries. Its archived absolute root is rewritten in memory from `/home/kaixu_chen/data/skiing/Ski-PosePTZ-CameraDataset-png` to `/home/kaixu_chen/skiing/data/Ski-PosePTZ-CameraDataset-png`; the JSON itself is unchanged.

## Source artifacts

- `logs/ivc_mmsports_extension/frontend_generalization/frontend_generalization_comparison.csv` — SHA-256 `338af6c0b3ac642fd06100585d092e87650be614d7b8438907433708db4b0a67`; packaged copy `evidence/results/frontend_generalization__frontend_generalization_comparison.csv`.
- `logs/ivc_mmsports_extension/temporal_offset/temporal_offset_summary_last.csv` — SHA-256 `32e70010153172b68989c9c76f997317551c90cd0677c22fb655331e2145a1a9`; packaged copy `evidence/results/temporal_offset__temporal_offset_summary_last.csv`.
- `logs/ivc_mmsports_extension/sampling_rate/sampling_rate_summary_last.csv` — SHA-256 `e59bd780d3d0356feb6a4b33dbf3d18e7a99e698171a955a142cf7ee72b5cd1a`; packaged copy `evidence/results/sampling_rate__sampling_rate_summary_last.csv`.
- `logs/ivc_mmsports_extension/view_angle/view_angle_summary_last.csv` — SHA-256 `7767e89c42202498a902d77030acf97ddb1210bb3b920c16a2767fddb3b781cd`; packaged copy `evidence/results/view_angle__view_angle_summary_last.csv`.
- `logs/ivc_mmsports_extension/view_angle/view_angle_per_pair_last.csv` — SHA-256 `0bc565852de9979064466b7d457265ae11576843926e0635ae942197b9ead851`; packaged copy `evidence/results/view_angle__view_angle_per_pair_last.csv`.
- `logs/ivc_mmsports_extension/view_angle/view_angle_significance_last.csv` — SHA-256 `295fdb69d8b80e5dfb17ef085170d24c8b01210dbf11c000e3de95752f4e7f5f`; packaged copy `evidence/results/view_angle__view_angle_significance_last.csv`.
- `logs/ivc_mmsports_extension/view_angle/view_angle_pairwise_contrasts_last.csv` — SHA-256 `409c92028ed37381da1d4780f0b8b18b16047d7962fc889d112f4d8490a8c6c1`; packaged copy `evidence/results/view_angle__view_angle_pairwise_contrasts_last.csv`.
- `logs/ivc_mmsports_extension/view_angle/view_angle_statistics_last.json` — SHA-256 `4cb04e4e909d862c4d804ec3239fa9b5de5ab63c0c3d6ed561fe1529cb539315`; packaged copy `evidence/results/view_angle__view_angle_statistics_last.json`.
- `logs/ivc_mmsports_extension/masking/left/occlusion_summary_last.csv` — SHA-256 `2dc60cbb953ec8885416069e81517bb8675f10c50a82c80c1fbeadec104f1381`; packaged copy `evidence/results/masking__left__occlusion_summary_last.csv`.
- `logs/ivc_mmsports_extension/masking/right/occlusion_summary_last.csv` — SHA-256 `10410ad9522a5e664b11fb693866df48faafaff508536cbe70d2e433c296cfa9`; packaged copy `evidence/results/masking__right__occlusion_summary_last.csv`.
- `logs/ivc_mmsports_extension/masking/both/occlusion_summary_last.csv` — SHA-256 `350dddacbd5cd4f89288c00fd4567c2c6bf9247fb391a542abbab2891019c74b`; packaged copy `evidence/results/masking__both__occlusion_summary_last.csv`.

- `logs/ivc_mmsports_extension/ski_poseptz_journal/summary/baseline_comparison_last.csv` — SHA-256 `c355bca919b1d867d0fc94d4568ba22af8cedb46cce42badaae2bd57b3f788d1`; packaged copy `evidence/results/ski_poseptz_journal__baseline_comparison_last.csv`.
- `logs/ivc_mmsports_extension/ski_poseptz_journal/summary/method_comparison_last.csv` — SHA-256 `f22e1466759ec7415b6067c2330fbecf36bfba1247473cedf6ecd0eb550845f0`; packaged copy `evidence/results/ski_poseptz_journal__method_comparison_last.csv`.
- `logs/ivc_mmsports_extension/ski_poseptz_journal/summary/journal_evaluation_last.json` — SHA-256 `5173b8f1f1ecffa4bf7b46134512e394ddd7e68c139628d044b0679fd8430f52`; packaged copy `evidence/results/ski_poseptz_journal__journal_evaluation_last.json`.
- `logs/ivc_mmsports_extension/ski_poseptz_journal/summary/report_last.txt` — SHA-256 `66cda187b3c56519b98e5c340203fd529db13677db0ea52ee314281aeaeca07a`; packaged copy `evidence/results/ski_poseptz_journal__report_last.txt`.

- `logs/ivc_p1/temporal_alignment/temporal_alignment_summary.csv` — SHA-256 `a491376c392224de1ee9cf9dbe3f5916851773c69785ec7e46a4cab9d06fcb56`; packaged copy `evidence/results/temporal_alignment__temporal_alignment_summary.csv`.
- `logs/ivc_p1/frontend_adaptation/frontend_adaptation_matrix.csv` — SHA-256 `1f73aa7edede1af19083b1c304af0eae0a84d819c69c3a483840462d3626ee48`; packaged copy `evidence/results/frontend_adaptation__frontend_adaptation_matrix.csv`.


## N-view accuracy--cost provenance

- `logs/ivc_p1/nview/nview_summary.csv` — SHA-256 `4d51d234259cd843efc572015d4d8ccf40ce1de66be149514277a311d99db32c`; packaged copy `evidence/results/nview__nview_summary.csv`.
- `logs/ivc_p1/nview/nview_efficiency.csv` — SHA-256 `0fbb61a0ae4319e4a3fac922550e75db789f534318cdd09475ef0c2ca30f18be`; packaged copy `evidence/results/nview__nview_efficiency.csv`.
- `logs/ivc_p1/nview/nview_per_group.csv` — SHA-256 `8c42ee885f94a39f0f87bbc49e4a74a46bf7937a41cfaaf3a18c397d7df815e2`; packaged copy `evidence/results/nview__nview_per_group.csv`.
- `logs/ivc_p1/nview/nview_group_manifest.json` — SHA-256 `01df788600c87cb971a78be1f070c129dedcb86abbdb84f92948edc1c5a4887b`; packaged copy `evidence/results/nview__nview_group_manifest.json`.
- `logs/ivc_p1/nview/nview_provenance.json` — SHA-256 `cb6dd240b017bca0e8df5c13d1fbced80916872436ee144e15cd8e5a173b34d9`; packaged copy `evidence/results/nview__nview_provenance.json`.
- Timing protocol: 10 four-view warm-up compositions; 44 accepted 30-frame groups; synchronized serial all-pairs execution on one NVIDIA RTX A6000; PyTorch 2.11.0+cu126 and CUDA 12.6; monocular front-end inference, data loading, and oracle selection excluded.

## Upstream estimator provenance

- `motionbert`: repository commit `705d3a95354db8bdb696b3492e47a3b5537174ff`, estimator checkpoint SHA-256 `d80af32396c60cf66fa5afb7ef7f7c869ae0851afd3d91a75d55e76c5a62cb23`, prediction manifest SHA-256 `84b8b9a0e4d81dfec862c58197945b3209115a2fc49818f1ac07bbe900f9528a`.
- `poseformer`: repository commit `a908b29adca354b2e0acd36606229ca19b6a5e3c`, estimator checkpoint SHA-256 `fa0ef7ce52fb927b2ec2cb86fe46dd897807923c47ee756dee1706ae91d162fe`, prediction manifest SHA-256 `e7e898163b4eb124728acf7a0d49cafcf7ed75fad4f8a749b93e2c429d2595f3`.
- `videopose3d`: repository commit `1afb1ca0f1237776518469876342fc8669d3f6a9`, estimator checkpoint SHA-256 `b3865c7eff38ff182b8a269259436728487e802edcdb65a3eb4f843341e2d692`, prediction manifest SHA-256 `bc435db3c30761f21c1654e155498f6be9661ad7db984dc1965ea5afefa07cab`.

## Masking grid

- Rows: `45` (3 view modes x 3 patterns x 5 ratios).
- Corruption: 35% zero replacement and 65% Gaussian perturbation with standard deviation 0.08.

## Image-level occlusion through SAM3D

- Complete 18-cell grid: left/right/both views, random/distal/temporal patterns, ratios 0.5/1.0; 64,440 sequences per cell, all 15 joints, batch size 256, fold 0, seed 42, and the frozen Unity checkpoint listed above.
- Six completed prediction sets contain 720 camera streams and 17,089 unique source frames each, totaling 4,320 stream-condition outputs and 102,534 inference frame instances across conditions. Source images are reused across conditions.
- Fifteen cells have positive fusion gains; random ratio 1.0 has negative gains for left, right, and both views. These counts describe conditions from one frozen checkpoint, not independent training repeats or significance tests.
- Failed SAM3D detections are retained as zero poses. The table reports the mean detection-failure indicator over pair-aligned sequence frames in affected views, pooling left and right for both-view conditions. This is distinct from the unique-source-frame failure fraction and from the fused-pose error-threshold failure rate.
- Matching image- and pose-level ratios denote protocol labels, not equal physical occlusion severity. Ratio 1.0 does not mean that every image pixel is occluded. MPJPE is in dataset coordinate units after the archived batch-referenced transform, without conversion to millimeters.
- The independently re-evaluated `both_random_r1p00` summary row matches the main evaluation exactly. This check covers one cell; the other 17 cells were not independently repeated.

- `logs/ivc_mmsports_extension/image_occlusion/image_occlusion_summary_last.csv` — SHA-256 `0744e1121ec61f8379ef72d76dec11fb51203bc6f9328d6a2b406c5c4cac2e28`; packaged copy `evidence/results/image_occlusion__image_occlusion_summary_last.csv`.
- `logs/ivc_mmsports_extension/image_occlusion/image_occlusion_summary_last.json` — SHA-256 `2b707b8e0d51d8414f87b9463aceaaf824aa8200fdfa9eecc10e8ed6ad9764b7`; packaged copy `evidence/results/image_occlusion__image_occlusion_summary_last.json`.
- `logs/ivc_mmsports_extension/image_occlusion/image_vs_pose_occlusion_last.csv` — SHA-256 `6c31aae0827f1864d118d035e31fddb458d582a7240fcb163ed5df59b0d2afcb`; packaged copy `evidence/results/image_occlusion__image_vs_pose_occlusion_last.csv`.
- `logs/ivc_mmsports_extension/image_occlusion/reproducibility/both_random_r1p00_20260905T170432/image_occlusion_summary_last.csv` — SHA-256 `40e4594c8189a016a0b3845b0bf5d0945ed294baa1de1919153cb971f933675d`; packaged copy `evidence/results/image_occlusion__both_random_r1p00_recheck.csv`.
- `evidence/image_occlusion_frontend_provenance.json` — SHA-256 `1268df24e946d7da2071324c6e2e997cac20002b210e4402d51a51f9e7554351`; records all six frontend and frame-manifest hashes, mask protocols, completeness counts, model hashes, and the evaluation command.

## Expanded main Tables 2 and 3 (2026-09-06)

- Authoritative new run: `logs/ivc_mmsports_extension/main_baselines/20260906/`. The archived reference-only files listed above remain preserved historical evidence; the expanded main tables use this new complete recomputation.
- Twelve methods per dataset: left/right, canonical average, aligned average, 3D quality weighting, MLP, TCN, average plus SmoothNet, CanonFuse3D, DLT, reprojection-gated DLT, and DLT-residual MLP. The three calibrated rows are identified as additional-input references.
- Unity test coverage is 64,440 sequences, 28,998,000 all-15 joint-frame points and 25,131,600 common-13 points; acceleration denominators are 27,064,800 and 23,456,160. Ski test coverage is 30 sequences, 11,700 joint-frame points and 10,920 acceleration points. No final test batch is discarded.
- Eight learned-model runs completed epochs 0--99. Unity validation-selected epochs are MLP 65, TCN 20, SmoothNet 89, and DLT-residual MLP 71. All Ski selections use epoch 99. Ski has 145 usable native-loader training entries; five empty camera-pair entries are documented in the packaged coverage report.
- Ski geometry is prediction-pelvis-centered per frame before the unchanged archived canonicalization. This follows the source annotation convention, introduces no target-fitted transform, and is enforced by table admission checks. Raw-world diagnostics are preserved outside the final result set.
- `evidence/main_baselines_manifest.json` identifies all 24 source result hashes, the table renderer, generated tables, and packaged JSON. `evidence/main_baselines_verification.json` now records the PA-MPJPE extension: 46 passing tests, 108 checked table cells, 26 rehashed test input/geometry files, 10 checkpoint hashes, legacy-regeneration check, final PDF hash, and table pages 25/26. The earlier expansion verification is preserved in the PA run manuscript backup.
- `evidence/main_baselines/manifest.json` records SHA-256 values for compact reference-reproduction, training-coverage, geometry-coordinate, implementation, independent-review, and test reports. Both journal reference MPJPE reproductions differ by less than 5e-7.
- New baseline results are single-seed clean-index measurements. The main text reports the lower MPJPE of MLP and lower acceleration error of SmoothNet, and does not infer robustness superiority over these methods from the earlier averaging-based stress studies.

- `evidence/results/main_baselines__results.json` — SHA-256 `691b670ae5f8c4781d11c88b7810cc74d4eba3abd2b626b6badaf77ddabebdf8`.
- `evidence/results/main_baselines__summary.csv` — SHA-256 `c2655247285877ab6a827037d3990ff51cdc05eaa4aabf6764fa8e0a0379d1cf`.

## PA-MPJPE extension of the main tables

- Current enriched records: `logs/ivc_mmsports_extension/main_baselines/20260906/pa_mpjpe/results/`. The 24 parent result files remain unchanged; enriched records link their original hashes, selected checkpoints, input manifests, evaluator source versions, and re-evaluated original-metric differences.
- Added 36 PA-MPJPE values: 12 methods times two Unity joint subsets plus 12 Ski rows. Every frame is included. Each Unity subset has 1,933,200 evaluated frames; Ski has 900. PA joint-frame denominators match the MPJPE denominators listed above, with no degenerate prediction frame observed.
- Alignment convention: independently per frame and joint subset, float64 least-squares proper rotation, one nonnegative scale, and translation; no reflection. Alignment is solely for PA-MPJPE reporting, with the unaligned MPJPE and acceleration definitions retained. Formula parity against the VideoPose3D official code is recorded in the packaged `official_parity.json` with URL and downloaded-source SHA-256.
- All 72 earlier rounded metric cells are unchanged; the largest unrounded recomputation difference is 3.95714e-8. All selected checkpoints are unchanged and no training is repeated.
- `evidence/main_baselines_pa_verification.json` records all 108 table-cell checks, 46 passing tests, 10 checkpoint hashes, 26 test cache/geometry hashes, and final PDF/table-page checks. `evidence/main_baselines/pa_mpjpe/manifest.json` hashes the compact evaluation and verification reports. The PA-evaluation PDF snapshot had 52 pages, with Tables 2 and 3 on pages 25 and 26. The current layout is recorded in the consolidation verification below.

## Current manuscript consolidation and supplementary material

- Current delivery: `main.pdf` (46 pages; nine tables and nine figures) and `supplementary.pdf` (seven pages; five tables and two figures). The baseline/PA verification files describe their earlier PDF snapshot and point to the current `evidence/manuscript_consolidation_verification.json` for layout hashes and numbering.
- Original Figure 10 remains in the main article as Figure 8, and original Table 11 remains as Table 8. Original Table 14 is Supplementary Table S5; original Figure 8 is Supplementary Figure S2. The supplementary allocation preserves all moved displays and their numerical values.
- Main Table 5 combines angle-bin performance and camera-pair bootstrap intervals. Supplementary Table S1 retains full paired-test statistics. The consolidated generator records all four input CSV hashes and six combined rows in `evidence/consolidated_artifacts.json`.
- Main Figure 4 plots relative temporal/rate error, using each method's own zero-perturbation denominator. Its 21 plotted points are saved in `figures/extension/temporal_sampling.json` and match the original complete seven-condition source sweeps. The older extension generator uses the same implementation.
- All 12 pre-existing standalone table files, including both expanded baseline tables with PA-MPJPE, remain byte-for-byte unchanged. All 24 packaged main-baseline result source hashes match. No new training, inference or checkpoint selection was performed for this consolidation.
- Verification covers 50 focused tests, the six combined rows, all 21 relative-curve points, nine external links to the supplementary PDF, correct main/supplementary allocation, preserved negative findings, and compilation without undefined references or overfull boxes. Visual checks cover main pages 29, 30, 34 and 37 and supplementary pages 1, 2, 5 and 7.
- Backups: `logs/ivc_mmsports_extension/manuscript_consolidation/20260906/before/`. The packaged audit manifest is `evidence/manuscript_consolidation/manifest.json`.

- `main.pdf` — SHA-256 `66b8c7ca68eef7e8181152d53a02a12d3c18068e547569608d603712f53c1fcc`.
- `supplementary.pdf` — SHA-256 `caf14f5c5cf5385289ecad11f3e0d5913b6548c0871daf1cb7a0646eb8dd1db0`.

## External-baseline archive (2026-09-13)

- `evidence/external_baselines/` freezes eight final result JSON files and fourteen provenance JSON files for STRIDE, MetaPose, DeciWatch, and the native-output audit. Every archived copy is SHA-256 identical to its experiment-log source.
- The admitted canonical-space matrix is complete for three methods on both datasets: 64,440 Unity test windows and 30 Ski-PTZ-Pose test windows per method. The verifier checks all six exact metric triplets, coverage, finite values, PA degeneracy counts, control deltas, and MetaPose completion records.
- The verifier also checks 36 displayed metric cells across the canonical and native external-method rows, their BibTeX keys, the manuscript's prohibition on cross-protocol ranking, and the current table pages. Its generated reports are `evidence/external_baselines/manifest.json` and `evidence/external_baselines/verification.json`.
- Unity native-space evidence has 13 measured and four unavailable method records. All 17 Ski native-space records are protocol-unavailable because no independent target mapping exists in the declared output frames; these are not unfinished runs.
- The proposed multi-seed cross-fold matrix is retired from the current submission scope. It would have required six full 100-epoch trainings (two folds times three seeds). The two retained fold-1/seed-13 launch-smoke logs both stop during data loading before any epoch, checkpoint, or result. No aggregate statistic, confidence interval, or cross-fold claim is admitted.

## Author-confirmation items

- Confirm final MMSports publication metadata (venue, DOI, and pages).
- Confirm CRediT roles, competing-interest declaration, code/data release URLs, ethics wording, and generative-AI disclosure.
