# IVC external-baseline execution record — 2026-09-11

## Material passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: local code implementation, full experiment, artifact verification.
- Verification status: **COMPLETE AND ARCHIVED** for STRIDE, MetaPose, and the corrected official-protocol DeciWatch adaptation on both Unity and Ski-PTZ-Pose. Do not treat launch, a finite smoke forward, or training loss as a final 3D result.
- This record supersedes the **status snapshot** in `ivc_external_baselines_20260910.md`, not its historical evidence.
- Existing 28 manuscript numerical evidence files were rehashed: all unchanged. No paper numbers, PDFs, existing STRIDE outputs, or main environment dependencies were replaced. No commit/push.

## Final archived matrix — 2026-09-13

This final status supersedes the timestamped running snapshots retained below. Every admitted external method has a complete Unity result over 64,440 windows and a complete Ski-PTZ-Pose result over 30 windows.

| Method | Unity MPJPE / PA-MPJPE / Accel. | Ski MPJPE / PA-MPJPE / Accel. | Admitted protocol |
|---|---:|---:|---|
| Avg. + STRIDE | 0.2799 / 0.1174 / 0.0316 | 0.3483 / 0.1861 / 0.0387 | Released STRIDE model adapted as a temporal post-processor of the dense canonical average |
| DeciWatch | 0.1134 / 0.0695 / 0.0132 | 0.2239 / 0.1549 / 0.0168 | Corrected official-protocol adaptation, four unique observed source frames in each 30-frame window |
| MetaPose | 0.2655 / 0.1089 / 0.0400 | 0.4253 / 0.1425 / 0.0634 | Common-13 domain adaptation using native 3D plus additional estimated 2D, without supplied calibration |

Compact result and provenance copies, a SHA-256 manifest, and a table-consistency verifier are under `paper/ivc_draft_20260821/evidence/external_baselines/`. The canonical and native-space manuscript presentations remain separate; unavailable Ski native-space cells are protocol exclusions rather than unfinished runs.

## MetaPose adaptation

Design and frozen budgets: [implementation plan](superpowers/plans/2026-09-11-ivc-metapose.md).

This is **MetaPose (adapted, common13, domain-retrained using 2D supervision)**, not an exact reproduction of the published Ski-Pose result or the released Human3.6M checkpoint. It uses the official permutation-equivariant stage network and deep inverse solver. Three stage-wise MLPs, seed 42, float64, original forward 2D MSE objective, no 3D-supervision or bone-length branch. Native SAM3D 3D plus synchronized estimated 2D are inputs; no supplied camera calibration.

Input differences that must accompany a paper row:

- Four-scale point-centered uncertainty fitted on training residuals replaces the original fitted detector heatmap mixtures. No GT-centered likelihood and no validation/test uncertainty fitting.
- Common13 joints replace Human3.6M17; no fabricated missing joints. The official architecture is re-created at the new dimensionality and trained from scratch.
- Input-estimated weak projection brings pixel 2D and metric 3D into consistent coordinates. Outputs are returned through their estimated left camera, with the native left prediction's pelvis restored, then transformed by the archived native-left batch canonicalizer.
- The TF 2.15 S1 batching wrapper unrolls the official two-view calculation. Unit tests compare its packed state against the original eager initializer; third-party source and learned modules remain unchanged.
- Unity uses validation 2D MSE for checkpoint selection. Ski uses fixed final epochs, because the archived protocol has no validation split. Test data are opened by the trainer only after all stage checkpoints are frozen. 3D labels enter only the separate downstream evaluator.

## Launched full jobs

Snapshot at launch; inspect each `progress.json` / `failure.json` / `metrics.json` for current state.

| Dataset | GPU / PID | Train / val / test windows | Budget | Launch (JST) |
|---|---|---|---|---|
| Ski | 1 / 458506 | 145 / none / 30 | 3 stages × 300 epochs, batch 128 | 10:57:54 |
| Unity | 0 / 459395 | 193320 / 128880 / 64440 | 3 stages × 10 epochs, batch 1024 | 10:58:45 |

At 11:11 JST: Ski is `complete_evaluated` and its process has exited. Unity remains active, with 5,332,992 / 5,799,600 training frames initialized; validation initialization and the learned stages follow automatically. This is a timestamped snapshot, not a claim that Unity is already training or has final accuracy.

All windows retain exactly 30 archived frames. The full data exports are complete and SHA-verified for all five splits. All native prediction scale fits passed; no frames were dropped or substituted.

Each job is a detached local process, with a 24-hour maximum and no automatic retry. The training command automatically invokes the separate existing-PyTorch-environment evaluator after full frozen prediction export. The final `metrics.json` is written only after complete coverage and the archived-average control check pass. Process completion does not itself send a chat notification.

`report.json` is the immutable **prediction-export snapshot** used by the evaluator's checksum. After scoring, its older `complete_predictions_not_yet_scored` field is not rewritten; the final state is in `progress.json` (`complete_evaluated`) together with `metrics.json`.

Artifacts under `logs/ivc_mmsports_extension/external_baselines/20260911/`:

- `metapose_data/{ski,unity}/{train,val,test}/`: relevant splits only; arrays, source hashes, progress and complete export manifests. Test exports have no label array.
- `metapose_{ski,unity}_full_seed42.launch.json`: exact command, PID, GPU and timestamp.
- `metapose_{ski,unity}_full_seed42.stdout.log`: detached process output.
- `metapose_{ski,unity}_full_seed42/`: config, environment package freeze, uncertainty, S1 states, smoke check, epoch JSONL, selected-stage hashes, final predictions and metrics when complete.

Independent environment: `/tmp/ivc-metapose-20260911-7P6c7g/env`. Source checkout: `/tmp/ivc-table-citations-20260910-hc8JTg/metapose_source`, commit `08a8d6736475776f42ffac23b2c13111a28e5795`. Temporary assets are not durable installations; each run records versions, commit and file hashes for reconstruction.

## Checks performed

- RED→GREEN adapter, official-model and evaluator tests.
- Official S1 packing parity, finite learned output/gradients, synthetic loss decrease, exact strict weight reload and unchanged frozen-stage weights: passed in the isolated TF environment.
- Ski real-data 40-step gate: normalized 2D MSE 2.0157953945 → 0.0883023872; strict reload identical. These diagnostic weights are discarded before the seed-42 full fit.
- No-GT test-input fixture and HDF5-only-2D/identity fixture pass. End-to-end saved-prediction evaluation verifies the final partial batch (5 Ski windows → 150 frames / 1950 points).
- `pip check`: no broken requirements. Relevant main-environment tests pass; the TensorFlow test is intentionally skipped there and runs in isolation.
- Original 28 result-evidence hashes rechecked, all unchanged.

Final relevant regression: 74 tests executed in the main environment, 73 passed and the TF-only integration test skipped there. That integration test separately passed in the isolated GPU environment. No tests failed; no dependency inconsistencies. This test count includes native input isolation, training-only HDF5 fields, saved-weight equivalence and final partial-batch scoring.

## Completed MetaPose Ski result — seed 42

All three stages retained their predeclared final epoch 300, without test-based selection. Training 2D MSE at those epochs: stage 1 = 0.0012473467, stage 2 = 0.0007779689, stage 3 = 0.0010220533. Stage 2 had a transient loss increase around epochs 250–275 before recovery; the full epoch history is preserved. These training losses are not test performance.

Full evaluation: **30 / 30 windows, 900 / 900 frames, 11700 joint-frame points**, no nonfinite predictions or degenerate PA frames. Re-reading saved S1 and learned predictions and recomputing all metrics produced exactly the same values. All three frozen checkpoint hashes were verified. Average-control differences against the already completed STRIDE evaluation are exactly zero for all three metrics.

| Ski method | MPJPE | PA-MPJPE | Acceleration error |
|---|---:|---:|---:|
| Archived canonical average control | 0.3487475095 | 0.1877044295 | 0.0391832679 |
| MetaPose S1 only (initialization diagnostic) | 0.3645459285 | 0.1824899599 | 0.0657312651 |
| MetaPose learned, adapted, seed 42 | 0.4252570622 | 0.1425203865 | 0.0634478443 |

Interpretation is mixed: learned MetaPose improves PA-MPJPE over both S1 and the average, but worsens MPJPE relative to both, and worsens acceleration relative to the average. This does **not** establish an overall improvement. The result is not removed, repaired using test GT, or substituted with an earlier checkpoint. It is a single-seed adapted-method result, not the published original-method result.

Total train/init/predict/evaluate elapsed: 752.30 seconds (about 12.54 minutes). The recorded prediction-loop duration 6.96 seconds includes both S1 and learned-output export and tracing overhead; it is not a clean model-only latency benchmark. The existing paper remains untouched; no new rows have been inserted into the manuscript yet.

## Completed STRIDE results

Both full runs are complete, including 64440 Unity and 30 Ski windows. Read-only new-method values in archived native units:

| Dataset | MPJPE | PA-MPJPE | Acceleration error |
|---|---:|---:|---:|
| Unity, Avg. + STRIDE adapted | 0.2799222416 | 0.1173699139 | 0.0315757297 |
| Ski, Avg. + STRIDE adapted | 0.3482797910 | 0.1860614764 | 0.0386788556 |

Detailed STRIDE adaptation and additional-point input caveats remain in the September 10 record. Acceleration is unscaled second-difference error, not physical m/s².

## DeciWatch execution (superseded diagnostic)

The seed-42 results in this subsection are retained only as historical diagnostics. They are not admissible as the formal external baseline because the runner used a custom 30-frame forward, supervised the denoising branch on all frames, omitted Adam AMSGrad, trained Unity for only 10 epochs, and overwrote its validation-best checkpoint at the final epoch. The corrected predeclared rerun below supersedes these values.

The official checkout was available at `/tmp/ivc-deciwatch-20260911-uw729J/source`, commit `5afe2d67005441e2785a8d374d31a91f046a0da6`, but no official checkpoint was present locally. The first Ski launch stopped before training because a `Path` value was written directly to JSON; that failed directory and log remain as evidence. The corrected retry completed without changing the protocol.

Frozen adaptation: common-13 canonical average input, 30-frame windows, observed indices 0/10/20 (exactly 10% of frames), official DeciWatch Transformer architecture (hidden 128, five encoder/five decoder layers, four heads), seed 42, Adam 1e-3 with 0.95 epoch decay, `denoise_L1 + recovered_L1`, Unity 10 epochs with validation-L1 selection and Ski 70 fixed epochs. The 30-frame boundary uses endpoint-hold linear interpolation for the decoder seed; this is explicitly an adaptation because the official implementation expects `(window-1) % interval == 0`.

Ski is complete: 30/30 windows, 900/900 frames, all finite. Saved recovered and denoised arrays were independently rescored and matched the recorded metrics exactly. The default DeciWatch result is the recovered branch:

| Ski method | MPJPE | PA-MPJPE | Acceleration error |
|---|---:|---:|---:|
| Archived canonical average | 0.3487475095 | 0.1877044295 | 0.0391832679 |
| DeciWatch recovered (adapted, seed 42) | 0.2402832201 | 0.1596764269 | 0.0183041295 |
| DeciWatch denoised diagnostic | 0.2245183332 | 0.1327338301 | 0.1364055069 |

The recovered branch improves all three metrics over the average on this Ski split. The denoised branch has lower position errors but much worse acceleration, so it should not replace the recovered output in the main comparison. These are adapted domain-retrained results, not original pretrained DeciWatch results.

Unity DeciWatch is complete on physical GPU1 (PID 508320); MetaPose Unity remains on GPU0 (PID 459395). Unity used validation-L1 selection and selected epoch 10. The saved outputs were independently rescored using the required Unity canonicalization batch size 256; all three metric groups matched the recorded values exactly. Live artifacts are `deciwatch_{ski,unity}_full_seed42*` under the dated external-baseline directory.

Unity full evaluation: 64,440 / 64,440 windows, 1,933,200 / 1,933,200 frames, 25,131,600 joint-frame points, all finite.

| Unity method | MPJPE | PA-MPJPE | Acceleration error |
|---|---:|---:|---:|
| Archived canonical average | 0.2792121322 | 0.1162364325 | 0.0322140439 |
| DeciWatch recovered (adapted, seed 42) | 0.1161717831 | 0.0690493053 | 0.0125427262 |
| DeciWatch denoised diagnostic | 0.1084922657 | 0.0656675518 | 0.0271837981 |

The recovered branch improves all three metrics over the average on Unity. The denoised branch has lower MPJPE and PA-MPJPE and still lower acceleration error than the average, but it is retained as a separate diagnostic output rather than silently replacing the recovered branch.

## Corrected DeciWatch official-protocol adaptation — seed 4321

The formal rerun was frozen before test evaluation and supersedes the diagnostic seed-42 runs above. It follows the closest official `config_h36m_fcn_3D.yaml` settings wherever the archived IVC data allow.

Protocol differences and fixed controls:

- Source: official DeciWatch commit `5afe2d67005441e2785a8d374d31a91f046a0da6`; no released H36M checkpoint was available locally, so both datasets are domain-retrained from scratch.
- Model/training: common-13 input dimension 39, hidden 128, five encoder/five decoder blocks, four heads, pre-norm, dropout 0.1, batch 512, 70 epochs, seed 4321, Adam 1e-3 with `amsgrad=True`, and exponential LR decay 0.95.
- Loss: official sampled-frame denoising L1 plus all-frame recovery L1, both with weight 1. The superseded run incorrectly supervised denoising on all frames.
- Boundary: repeat archived frame 29 once to form 31 tokens, run the unmodified official forward at token indices 0/10/20/30, then crop predictions to the original 30 frames. This reads four unique source frames per archived window (4/30 = 13.3%), so it is not labelled as the original 10%/101-frame setting.
- Compatibility: the official integer attention masks are converted to boolean for PyTorch 2.11; mask values and observed positions are unchanged.
- Selection: Unity uses the minimum validation official loss and selects epoch 5. Ski has no archived validation split and therefore uses the predeclared fixed epoch 70. Test labels are opened only after predictions and hashes are fixed.

| Dataset | Windows / frames | MPJPE | PA-MPJPE | Acceleration error | Selected epoch |
|---|---:|---:|---:|---:|---:|
| Unity | 64,440 / 1,933,200 | 0.1133649794 | 0.0694621634 | 0.0132084328 | 5 |
| Ski | 30 / 900 | 0.2239018530 | 0.1548995967 | 0.0167880537 | 70 |

Both full outputs are finite, contain every expected window and frame, and have zero degenerate PA frames. In an independent second scoring pass, every metric and count matched the recorded JSON exactly; prediction hashes also matched `report.json`. The archived-average controls differed from their prior reference metrics by exactly zero for MPJPE, PA-MPJPE, and acceleration error.

| Dataset | CanonFuse3D MPJPE / PA / Accel. | DeciWatch MPJPE / PA / Accel. | Honest comparison |
|---|---:|---:|---|
| Unity | 0.1551 / 0.0637 / 0.0297 | 0.1134 / 0.0695 / 0.0132 | DeciWatch is better on MPJPE and acceleration; CanonFuse3D is better on PA-MPJPE. |
| Ski | 0.1794 / 0.1269 / 0.0311 | 0.2239 / 0.1549 / 0.0168 | CanonFuse3D is better on MPJPE and PA-MPJPE; DeciWatch is better on acceleration. |

The result is mixed rather than an overall win for either method. DeciWatch is a sparse temporal recovery method applied after the canonical two-view average, whereas CanonFuse3D is a dense two-view pose-space fusion method; any manuscript presentation must keep these task and input-budget differences explicit.

Artifacts:

- Unity: `logs/ivc_mmsports_extension/external_baselines/20260911/deciwatch_unity_official_protocol_full_seed4321/`
- Ski: `logs/ivc_mmsports_extension/external_baselines/20260911/deciwatch_ski_official_protocol_full_seed4321/`

Prediction SHA-256: Unity `13c2d3e9fc1ba2a731b2b74198d36ae14ec5c1f277bf35f2878c0d1adb9d0ff8`; Ski `ee745023f010194ab2437f5e4684d52d163da8d9082bb7d909b3bea461da354a`.

## Why DeciWatch was initially a backup

[DeciWatch](https://arxiv.org/abs/2203.08713) targets efficient sequence estimation by processing sparse sampled frames and recovering a dense pose sequence. It can improve smoothness/accuracy too; backup status is not a claim that it is weak. The current experiment supplies dense per-frame 3D and focuses on fusion/refinement. A DeciWatch experiment therefore needs an explicit sampling ratio, sequence recovery window and input/compute budget. STRIDE is the closer temporal-refinement comparison, while MetaPose adds the distinct uncalibrated multi-view comparison. DeciWatch was also reserved as a fallback if STRIDE's released assets could not be adapted defensibly.
