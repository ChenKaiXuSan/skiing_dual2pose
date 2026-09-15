# IVC published external baselines: plan and bounded integration

**Final status (2026-09-13):** Superseded by the completed full-experiment record in `docs/ivc_external_baselines_20260911.md`. STRIDE, MetaPose, and the corrected official-protocol DeciWatch adaptation now have admitted full Unity and Ski-PTZ-Pose results. The older unchecked Phase D items and running snapshots below are historical checkpoints, not pending submission work. Compact evidence is archived under `paper/ivc_draft_20260821/evidence/external_baselines/`.

**Date:** 2026-09-10. **Authority:** author requested table citations, an external-experiment plan, code integration and small-scale validation. This does not authorize a full training sweep, submission, or push.

**Goal:** augment the existing component/architecture comparisons with attributable published methods without changing any existing result or treating a local variant as an original-method reproduction.

**Architecture:** target-free readers of the archived raw-pose cache; optional estimated-2D reader independent of the calibrated geometry loader; adapters loading user-supplied official source checkouts; fresh, separate diagnostic outputs. Keep the active IVC checkout and existing model/evaluator unchanged. No third-party code or weights are vendored.

**Stack:** existing NumPy/PyTorch `dual2pose` environment for input checks and STRIDE backbone checks. MetaPose requires a separate TensorFlow environment; do not replace dependencies in `dual2pose`.

## Material passport

- Origin mode: code experiment plan plus implementation diagnostics, not a completed comparison.
- Sources: official papers, repositories, current IVC source code and archived cache manifests.
- Verification status: **UNVERIFIED for research performance** until complete training/evaluation and provenance checks below. A shape/finite-value pass does not change this status.
- Raw data remain local. No GT poses, GT 2D or supplied camera calibration may enter the external inference/adaptation API.

## 1. Questions and comparison groups

Does learned two-stream fusion improve on published temporal refinement under the same native front end? How does it compare with an uncalibrated multi-view method when the latter also receives estimated 2D observations?

| ID | Method and authoritative source | Proposed IVC condition | Status / admission condition |
|---|---|---|---|
| E7a | [STRIDE, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Lal_STRIDE_Single-Video_Based_Temporally_Continuous_Occlusion-Robust_3D_Pose_Estimation_WACV_2025_paper.html), [official code](https://github.com/take2rohit/STRIDE) | Refine the canonical average with single-video test-time adaptation; compare with average, Avg. + SmoothNet and CanonFuse3D. Label **Avg. + STRIDE (adapted)** after validation. | Priority 1. Need an explicit 17-joint input policy, official motion prior, and verified adaptation objective. A 13-joint randomly initialized backbone is only an interface diagnostic. |
| E7b | [MetaPose, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Usman_MetaPose_Fast_3D_Pose_From_Multiple_Views_Without_3D_Supervision_CVPR_2022_paper.html), [official code](https://github.com/google-research/google-research/tree/master/metapose) | Two native monocular 3D streams plus synchronized estimated 2D observations; initialize unknown cameras using predictions, not calibration. | Priority 2. Separate **additional estimated 2D, no supplied calibration** group. Need uncertainty/coordinate preparation and learned-stage checkpoint or training. Official Ski-Pose data/checkpoints are not released. |
| E7c | [DeciWatch, ECCV 2022](https://arxiv.org/abs/2203.08713), [official code](https://github.com/cure-lab/DeciWatch) | Published temporal recovery alternative if STRIDE's released assets cannot support a defensible adaptation. | Backup only. Predeclare sampling rate, recovery window, boundary policy and dense/sparse input budget before execution; do not present it as dense-input smoothing. |

Retain the existing pose-only group and calibrated-DLT group. Do not classify MetaPose as calibrated just because it estimates cameras. Do not mix the three input budgets in a single unqualified ranking. Existing TCN/DLT combinations remain local baselines with component citations, not additional published fusion systems.

## 2. Fixed evaluation contract

- Native SAM3D predictions, existing action-disjoint Unity fold_00 and fixed Ski split. Unity train/val/test = 193,320 / 128,880 / 64,440; Ski usable train/test = 145 / 30 (no validation split). No full-test iteration in the initial smoke stage.
- Preserve 30-frame windows and exact cached frame identities. An extended context requires separate, declared evaluation; do not concatenate unrelated camera pairs into a video.
- Preserve the archived batch-anchored canonicalization: independent left/right transforms, batch size 256 for Unity and 4 for Ski. Canonicalize native 15-point Unity streams before selecting the common 13. Do not center every sample/frame independently under the same protocol label.
- Report common-13 MPJPE, per-frame PA-MPJPE and acceleration error with the existing evaluator and coordinate/unit conventions. Unity all-15 scores are optional only with an actual valid 15-point output, never fabricated eyes as evaluated predictions.
- Ground truth belongs exclusively in the downstream evaluator or permitted training loss. Test-time optimization must receive predictions/estimated observations only. Do not use GT 2D from the earlier frontend-distribution experiment for these native-front-end comparisons.
- Use Unity validation for choices. Ski has no validation set: freeze externally selected settings and a fixed training budget, or predeclare a new train-only subdivision as a separate protocol. Never copy upstream `train,test` early-stopping selection literally.
- Include all 64,440 / 30 test sequences and final partial batches. Report failures and finite-prediction coverage explicitly; do not silently drop difficult examples or substitute the average when an external method fails.
- Seed 42 for integration; for a full trainable comparison, target three fixed seeds (42, 43, 44) if resources permit. Label single-seed results if only one is run. Bootstrap/aggregate at the original motion/subject grouping, not independently over correlated camera-pair windows.
- Preserve prior numbers and negative findings. New results enter a new evidence package/table revision only after complete verification.

## 3. STRIDE-specific gates

The official `lib/data/dataset_motion_3d_stride_infer.py` constructs H36M-17 pseudo-labels from predicted mesh vertices with SMPL(-X) conversion and a joint regressor. The default non-synthetic inference path supplies **3D xyz**, despite the local variable name `motion_2d`. Its `stride/stride_inference.py` fine-tunes a pretrained DSTformer on the same video, using pseudo-label position/scale/velocity and limb losses. This is not merely a frozen temporal filter.

- Record official source commit and checkpoint SHA-256; load strictly, without dropping positional embeddings or silently borrowing an arbitrary MotionBERT checkpoint.
- Current 13/15-joint caches do not provide the complete H36M-17 skeleton. Neck is index 9 in the existing official-lifter mapping, not the Ski annotation's different mapping. Do not invert a many-to-fewer joint selection or insert zeros and claim checkpoint compatibility.
- Preferred next check: inspect whether native predicted meshes or independently specified full joint predictions permit the required regressor. If not, explicitly propose a 13-joint retrained adaptation and its training budget before running it. Such an adaptation is not the original checkpoint reproduction.
- Predeclare averaging/alignment convention, root/scale conversion, 30-frame versus 243-frame handling, flip augmentation and every adaptation-loss coefficient from the chosen official configuration.
- Reset model **and optimizer** from the same prior for every independent video/camera-pair sample. Never carry adapted weights between held-out examples. Tune the number of steps on validation only; no GT-based stop criterion.
- Measure initialization/loading, adaptation and forward latency separately, and total per-sequence time/peak memory on matched hardware. A random-weight forward timing is not published STRIDE latency.

## 4. MetaPose-specific gates

The [official README](https://github.com/google-research/google-research/tree/master/metapose) releases Human3.6M assets but explicitly withholds Ski-Pose data/checkpoints for licensing reasons. The released code is TensorFlow-based. Its generic `initial_epi_estimate` function can be checked separately from its learned refinement stage.

- Read matched native `pred_keypoints_2d` / `kpt2d_*.npy` only. Source keys and frame IDs must match the 3D cache. The calibrated loader is **not** reusable as the inference entry point because it also reads calibration.
- Keep raw per-camera 3D for initial pose/camera estimation. Do not reuse a GT-calibrated DLT output as uncalibrated initialization.
- Export pixel 2D observations with their explicit coordinate convention. Resolve image normalization and 3D-to-2D scale/translation consistently before using the objective. Do not treat pixel xy and metric xyz as already co-normalized.
- The probabilistic solver expects four-component per-joint mixtures (`cams x joints x 4 x 4`: weight, mean-x, mean-y, variance). Native point predictions are not detector heatmaps. If approximating uncertainty from point detections, specify and validate the approximation on training/validation and label it as an adaptation.
- Stage-1 initialization, iterative optimization (S1+IR), and the learned MetaPose stage must be reported separately. Running only S1 or a generic bundle optimizer does not establish a learned MetaPose result.
- Disable GT initialization, GT heatmaps and GT bone lengths. Never enable `fake_gt_init` or `gt_heatmaps`; bone-length branches require independent non-test information and separate disclosure.
- Record TensorFlow/Keras/TFP versions. An isolated compatibility environment may differ from the original TF 2.8 release and must be labeled accordingly.

## 5. Implementation and acceptance tasks

Execute inline in the existing checkout, with a review checkpoint after each phase. No commit/push or long GPU job is part of this first stage.

### Phase A — citations and preservation

- [x] Add component/method references to the current main common-13 table and supplemental all-15/common-13 table, retaining all cells, bolding and commented rows.
- [x] Update future table-generator labels and add an attribution-boundary regression test.
- [x] Rebuild both PDFs, inspect the actual rendered tables, compare all numeric cells and verify unchanged result-file hashes.
- Do not run the old two-table renderer over the current author-edited merged main table: its layout predates the September 7 consolidation.

### Phase B — target-free input adapter (this execution)

- [x] Tests first: create `tests/test_external_baseline_inputs.py` for label-free loading, exact sample/frame order, independent archived canonicalization before joint selection, invalid shapes/frames, and refusals of test data or output overwrite in smoke mode.
- [x] Create `dual2pose/eval/external_baseline_inputs.py`: `load_pose_probe(cache_dir, count=2)`, `load_estimated_2d(probe, dataset_root)`. Reuse cache provenance and source-index matching, but never open `target.npy`, labels HDF5 or calibrated projection matrices.
- [x] Verify the input files and source index against their manifest hashes; save cache/index/selected-input provenance. Read a protocol batch from memory-mapped arrays, not the full multi-GB arrays into RAM (whole-file checksums are still verified).
- [x] Preserve raw per-view common-13 inputs separately from canonical left/right/average. Do not synthesize an H36M-17 export.

### Phase C — official-source smoke (this execution)

- [x] Create `dual2pose/experiments/smoke_external_baselines.py`: `--cache-dir`, `--dataset-root`, `--output-dir`, optional `--stride-repo`, optional `--metapose-repo` and `--metapose-python`; CPU-only, maximum four non-test sequences, output directory must be new.
- [x] On two Unity validation and two Ski training windows, check input sizes, joint mapping, exact paired 2D frames and finite values; save input NPZ plus report JSON, without targets or accuracy metrics.
- [x] Run the official STRIDE DSTformer forward with an explicitly **random-initialized, common-13 architecture adaptation**; record this limitation in the file name/report and never call it a pretrained STRIDE result. Check output shape, finiteness and input immutability.
- [x] Attempt MetaPose official stage-1 initialization in an isolated compatible environment. Record dependency failure if unavailable; if successful, report only S1 initialization shapes/finite values, not learned refinement performance.
- [x] Record commands, environment, source commits, durations, hashes, sample IDs, component statuses and remaining blockers. Run focused tests again after live probes.

### Phase D — future work; NOT authorized as a full job in this execution

- [ ] Resolve STRIDE full-skeleton/prior/TTT gates and MetaPose 2D uncertainty/normalization/learned-stage training gates; review the adaptation descriptions and compute budget with the author.
- [ ] Add strict checkpoint loading, independent-sequence reset tests, real adaptation-objective checks, bounded training/validation overfit tests, and explicit failure-coverage handling.
- [ ] Execute the full split/seed matrix into a new run directory and evaluate with the existing metric accumulators, including total adaptation cost.
- [ ] Only then add measured rows, citations, input-group notes and matched claims to the paper; compile and visually inspect again.

## 6. Commands and outputs

Initial source checkouts are outside the tracked repository; supply their real paths explicitly. Keep dependency installation separate from the existing `dual2pose` environment.

```bash
conda run -n dual2pose python -m unittest tests.test_external_baseline_inputs tests.test_main_baseline_pa_tables -v
conda run -n dual2pose python -m dual2pose.experiments.smoke_external_baselines \
  --cache-dir logs/ivc_mmsports_extension/main_baselines/20260906/cache/unity/val \
  --dataset-root /home/kaixu_chen/skiing/data/skiing_unity_dataset \
  --output-dir logs/ivc_mmsports_extension/external_baselines/20260910/unity_val_probe \
  --stride-repo /path/to/official/STRIDE
```

The commands are now implemented. Use a **new** output path for each repeat (the example output above already exists). The analogous Ski probe uses `cache/ski/train` and the Ski dataset root. Artifacts are `inputs.npz`, optional explicitly named component outputs, and `report.json` with `publication_eligible: false`; no MPJPE table is produced by this diagnostic tool.

## 7. Evidence record

First actual execution completed on 2026-09-10:

| Non-test subset | Windows / frames | STRIDE official backbone, random/common13 | MetaPose official S1 only | Full method accuracy |
|---|---|---|---|---|
| Unity val | 2 x 30, first archived batch anchor (256) | Finite `(2,30,13,3)` forward; no adaptation | Finite pose `(2,30,13,3)`, rotations `(2,30,2,3,3)`, scales `(2,30,2)`, shifts `(2,30,2,3)` | **Not evaluated** |
| Ski train | 2 x 30, first archived batch anchor (4) | Same checks passed | Same checks passed | **Not evaluated** |

Evidence paths (repository-relative):

- `logs/ivc_mmsports_extension/external_baselines/20260910/unity_val_probe/report.json`
- `logs/ivc_mmsports_extension/external_baselines/20260910/ski_train_probe/report.json`
- Each directory also contains `inputs.npz`, `stride_untrained_common13_diagnostic.npz`, `metapose_stage1_only.npz`, `metapose_stage1.json`, and the MetaPose worker log. Reports include artifact/input hashes, exact selected sample identities, source provenance and full invocation.

Source revisions:

- STRIDE `84200692c8a445972b59afefee687c552aabe082`.
- Google Research / MetaPose `08a8d6736475776f42ffac23b2c13111a28e5795` (only the `metapose` directory checked out).
- Working checkouts are `/tmp/ivc-table-citations-20260910-hc8JTg/stride_source` and `/tmp/ivc-table-citations-20260910-hc8JTg/metapose_source`; these temporary paths are not durable installation locations. Recreate source checkouts at the recorded commits if cleaned.

Runtime: Python 3.11.15, existing PyTorch 2.11.0+cu126 run explicitly on CPU. MetaPose uses a separate temporary virtualenv at `/tmp/ivc-table-citations-20260910-hc8JTg/metapose_env`, with `tensorflow-cpu==2.15.1`, `tensorflow-probability==0.23.0`, Keras 2.15.0, NumPy 1.26.4. This is a compatibility smoke environment, **not** the original TF 2.8 environment and not a complete learned-training installation. [TFP's official 0.23 release notes](https://github.com/tensorflow/probability/releases/tag/v0.23.0) identify compatibility with TF 2.15.

To recreate the isolated dependencies and invoke the MetaPose part alongside a probe:

```bash
python -m venv /path/to/new/metapose-smoke-env
/path/to/new/metapose-smoke-env/bin/python -m pip install tensorflow-cpu==2.15.1 tensorflow-probability==0.23.0
# Add these arguments to the smoke command, with a fresh output directory:
# --metapose-repo /path/to/google-research --metapose-python /path/to/new/metapose-smoke-env/bin/python
```

No checkpoint was downloaded or loaded for either component; no training, test-time adaptation, learned MetaPose refinement, or accuracy evaluation was performed. The 2D positions were paired/exported successfully but **not consumed by MetaPose S1**, which takes only raw per-view 3D. Uncertainty and coordinate preparation remain open for the subsequent stage.

Table preservation: both TeX tables retain all 144 metric cells including the commented MLP rows; the 28 numerical result evidence files retain their pre-edit SHA-256. Main PDF remains 38 pages and supplement 15 pages; actual table pages 20 and 2 were visually inspected, with resolved citations and no overfull boxes or undefined references. Backups are under `/tmp/ivc-table-citations-20260910-hc8JTg/`.

Final regression verification: **55 tests passed**, including all 13 new external-input/diagnostic contract tests, all five PA-table tests, and existing cache, pose-baseline, geometry, metric-accumulation and frontend-mapping tests. The new tests were observed to fail before their corresponding modules were implemented. Exact command:

```bash
conda run -n dual2pose python -m unittest \
  tests.test_external_baseline_inputs tests.test_main_baseline_pa_tables \
  tests.test_main_baseline_data tests.test_main_baselines \
  tests.test_main_baseline_hardening tests.test_run_main_baselines \
  tests.test_main_baseline_geometry tests.test_frontend_lifters -v
```

Phases A--C are complete at the stated **component-diagnostic** scope. Phase D remains open; no full external-method comparison is complete.

## 8. Full-run authorization and execution amendment (2026-09-10)

The author subsequently requested: determine the setup, start full experiments,
and report final results. This authorizes full local computation; the earlier
small-scale-only authority above is historical. It does not authorize a push or
changing the existing paper numbers. Execute inline using executing-plans.

### STRIDE adapted contract

Retain native common-13 predictions exactly. Supplement H36M slots 0/7/8/10
with actual MHR-127 bones `root` / `c_spine1` / `c_spine3` / `c_head_null`
(indices 1/35/37/126), respectively. These are a declared semantic adaptation,
not the official SMPL mesh regressor. The original 13 surface keypoints are
not replaced by internal bones. Apply each archived native-15/13 batch transform
to all 17 points, then average the two streams. Use metric xyz without a new
scale normalization. Feed this average to the official 17-joint prior; root-center
only the pseudo-target, as upstream does. Restore the input average's own root
to the root-relative output; do not use any GT alignment outside PA evaluation.
Omit the upstream demo's extra mean-of-joints-2/3 translation (these are not hips
in H36M-17). Keep 30-frame windows; no padding to 243 or cross-example context.

Freeze the released configuration before test evaluation: 30 updates, AdamW,
lr 0.0002, decay 0.99 per update, weight decay 0.01, position/scale/velocity/limb
variance coefficients 1/0.5/20/200. Other loss coefficients are zero. Use upstream
random horizontal flip in training and flip averaging at inference. Reset the
model, Adam state, learning rate and seed 42 independently per window. Single
adaptation seed; this is not a three-seed domain-trained comparison.

### Implementation tasks

- [x] Add `tests/test_stride_external.py`: actual-joint mapping preserves every
  common-13 coordinate; reject missing/nonfinite bones; canonicalize additional
  joints using the native anchor; repeated adaptation after an unrelated window
  reproduces the first output, including optimizer reset; reject nonfinite prior.
- [x] Implement `dual2pose/eval/stride_external.py`: `assemble_h36m17(common,bones)`,
  `canonical_average17(left,right,bones_left,bones_right,dataset)`, and
  `IndependentStride(model, loss_module, flip_fn, config, seed).refine(x)`.
  Load official model parameters strictly with no positional-embedding resizing.
- [x] Implement `dual2pose/experiments/run_stride_external.py`: target-free input
  preparation with exact source/frame/cache checks, separate downstream evaluator,
  new output directories, saved predictions, full sample coverage, failure report,
  per-window timing and input/checkpoint/code provenance. Never overwrite a run.
- [x] Run the new tests and existing 55-test regression suite. Run real 30-update
  preflight on non-test windows, measure runtime, and check adaptation/reset.
- [ ] If preflight passes, execute all Ski test windows and all Unity test windows
  with the frozen setup. Report incomplete/failed status until full coverage passes.

### MetaPose checkpoint finding

The official cam2 TensorFlow checkpoint contains seven model objects. Strict model
loading needs legacy Adam under TF 2.15. A direct checkpoint-tensor audit found
nonfinite weights in later stages, explaining the seven-stage synthetic-forward
failure. The README's one-stage loading example must not be dismissed as incomplete
solely because the archive stores seven objects. A finite, documented stage choice
or domain retraining remains necessary, together with the unresolved estimated-point
uncertainty and 3D/2D coordinate adapter. No MetaPose full run is admitted yet.

### Actual execution checkpoint, 17:34 JST

- 63 tests passed: the preceding 55 plus eight STRIDE mapping/reset/preparation
  tests. Tests were run and failed before the corresponding new modules existed.
- True official-weight, 30-update preflights completed on two Ski train and two
  Unity val windows. Stable adaptation plus inference/reset is about 0.78 seconds
  per window on GPU1 (RTX A6000). No test tuning followed these preflights.
- Ski full test **completed**, 30/30 windows, 900 frames, 11,700 scored points:
  MPJPE 0.34827979103566553, PA-MPJPE 0.1860614763567546, acceleration error
  0.03867885560383655 (archived native units). Saved-prediction re-evaluation
  exactly reproduced all reported metrics. Average-control differences from old
  metrics are below 1.7e-9; all old numerical evidence files are unchanged.
- Unity full test **running**, PID 4124427, GPU1, started 17:32 JST, 24-hour
  maximum. All 17,089 required raw prediction frames passed preparation and all
  64,440 input windows were prepared. Initial 25 windows completed; no final
  Unity metric exists at this checkpoint. Expected total about 14 hours at the
  measured speed, subject to shared-machine load.
- MetaPose's 0-based stages 0--2 have finite weights; stages 3--6 each contain
  20 nonfinite model tensors (2,989,885 parameters). The README one-stage model
  loads with all expected model objects matched and gives finite synthetic
  `x_opt (1,71)` and `fwd (1,68)`. Seven-stage execution failed. This is a
  checkpoint/preflight finding, not a MetaPose accuracy result.
- Full-run and preflight artifacts are under the existing external-baseline
  log root. See `docs/ivc_external_baselines_20260910.md` for exact paths and
  comparison limitations. The manuscript's tables/PDFs are not changed here.
