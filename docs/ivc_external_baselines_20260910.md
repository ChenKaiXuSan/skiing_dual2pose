# IVC external-baseline execution record

**Newer status:** see [2026-09-11 execution record](ivc_external_baselines_20260911.md). The running/not-launched states below are the original September 10 snapshot, not current status.

## Material Passport

- Origin Skill: experiment-agent (academic-research-suite)
- Origin Mode: run
- Origin Date: 2026-09-10
- Verification Status: UNVERIFIED for the complete external comparison;
  Ski saved-artifact metrics have been independently recomputed in this session.
- Version Label: external_run_v1
- Snapshot: approximately 17:34 JST. Unity status below is a historical snapshot;
  consult its live `report.json` before reporting completion.

## Status

| Experiment | Actual state | Coverage / remaining work |
|---|---|---|
| Avg. + STRIDE adapted, Ski test | Completed, process exit 0 | 30/30 windows, 900/900 frames, no nonfinite outputs |
| Avg. + STRIDE adapted, Unity test | Running on GPU1, PID 4124427 | All 64,440 inputs prepared; initial windows completed; final metrics not yet available |
| MetaPose adapted | Not launched as a full experiment | Point-uncertainty/coordinate adaptation and learned-stage training/evaluation setup remain unresolved |
| DeciWatch | Backup, not launched | No result |

## Completed Ski result

All numbers below use the archived paper units: MPJPE/PA-MPJPE in metres;
acceleration error is the unscaled second-difference error, not a physical m/s²
measurement. Only the STRIDE row is new. Other rows are read-only archived results.

| Method | MPJPE | PA-MPJPE | Acceleration error |
|---|---:|---:|---:|
| Canonical average | 0.3487475112 | 0.1877044307 | 0.0391832689 |
| Avg. + STRIDE (adapted, seed 42) | 0.3482797910 | 0.1860614764 | 0.0386788556 |
| Avg. + SmoothNet | 0.3227119607 | 0.1924098998 | 0.0151236094 |
| CanonFuse3D | 0.1793983718 | 0.1268888936 | 0.0310755188 |

The new row is one frozen adaptation seed, not a three-seed training mean or a
statistical-significance claim. Its small improvement over the average does not
establish superiority over all external methods. Full Unity evidence is pending.

### Protocol and attribution boundaries

- Official STRIDE 17-joint pretrained DSTformer, all model keys strictly matched;
  official loss implementations and flip routine. 30 updates independently per
  30-frame window; fresh model weights, optimizer, learning rate and seed each time.
- MHR-127 bone indices 1/35/37/126 supplement root/spine/thorax/head slots.
  Existing common-13 surface predictions are preserved. This is a semantic
  adaptation, **not** the original SMPL(-X)-to-H36M mesh-regressor reproduction.
- STRIDE receives four additional predicted skeletal points from the same native
  front end. All scores use the same common 13 joints, but input joint cardinality
  is not identical to the 13-point fusion models; disclose this in any paper row.
- The native 15/13-joint batch-anchored transform is applied before joint mapping
  and averaging. The fixed protocol batch sizes remain Unity 256 and Ski 4.
- Input coordinates remain metric canonical xyz; only pseudo-targets are made
  root-relative in the TTT loss. Output restores the input's own root trajectory.
  No GT, GT 2D, supplied camera calibration or test-selected transform enters TTT.
- Final PA alignment is evaluation-only. It does not alter MPJPE or acceleration.
- No hyperparameter selection used Ski test scores. The Unity run uses exactly
  the same settings frozen before the first Ski test evaluation.

### Verification

Eight new tests plus the existing 55 passed. Re-evaluating saved Ski predictions
reproduced `metrics.json` exactly. The recomputed average control differs from the
old average by only 1.622e-9 / 1.243e-9 / 1.014e-9 in the three metrics. The original
28 manuscript result-evidence files retain their earlier SHA-256 hashes. Existing
paper tables, results and PDFs were not replaced by this execution.

Ski elapsed pre-evaluation runtime: 26.56 seconds including preparation. Mean
post-warmup reset+adaptation+forward time: 0.7785 seconds/window. Peak allocated
CUDA tensor memory: 1122.05 MiB on an RTX A6000. This is framework-allocated peak,
not total board memory. The result report records the exact config and file hashes.

## Artifacts and live monitoring

All paths below are relative to the repository root:

- Completed Ski:
  `logs/ivc_mmsports_extension/external_baselines/20260910/stride_ski_test_full_seed42/`.
  `report.json`, `metrics.json`, `predictions_common13.npy`, `inputs17.npy`,
  `input_provenance.json`, `per_window.jsonl`.
- Running Unity:
  `logs/ivc_mmsports_extension/external_baselines/20260910/stride_unity_test_full_seed42/`.
  `report.json` contains status/completed_count. Final `metrics.json` is created
  only after all 64,440 predictions have been saved and checked.
- Unity stdout:
  `logs/ivc_mmsports_extension/external_baselines/20260910/stride_unity_test_full_seed42.stdout.log`.
- Exact launch command, PID and timestamp:
  `logs/ivc_mmsports_extension/external_baselines/20260910/stride_unity_test_full_seed42.launch.json`.
- True-TTT non-test preflights:
  `stride_ski_train_ttt_preflight/` and `stride_unity_val_ttt_preflight/` under
  the same dated external-baseline root.

Unity was started as an independent local process at 17:32 JST. It saves progress
every 25 windows and saves timing/loss history after every window. A declared
24-hour budget stops the loop with failed/timeout evidence; there is no automatic
restart, dropped-example fallback or synthetic final score. At 0.78 seconds/window
the initial estimate is approximately 14 hours, plus preparation and final metric
calculation. This is not a promise of a specific completion time.

## Official assets and MetaPose finding

STRIDE source commit: `84200692c8a445972b59afefee687c552aabe082`.
Checkpoint SHA-256:
`702a9eba19b95f3c8c08c89c6593c8486dba3c3dedc5e28f917cba23c00bfeee`.
Source/config origin: [official STRIDE repository](https://github.com/take2rohit/STRIDE).
The working checkout and downloaded prior are external temporary assets; their
absolute paths and hashes are recorded in every run report.

MetaPose source commit: `08a8d6736475776f42ffac23b2c13111a28e5795`.
The cam2 checkpoint was retrieved from the
[official asset archive](https://storage.googleapis.com/gresearch/metapose/metapose.tar)
using verified tar-entry offsets, without downloading H36M/Ski pose records.
TensorFlow's checkpoint reader successfully read the tensors.

| Zero-based learned stage | Model tensors | Tensors containing NaN/Inf |
|---|---:|---:|
| 0 | 20 | 0 |
| 1 | 20 | 0 |
| 2 | 20 | 0 |
| 3 | 20 | 20 |
| 4 | 20 | 20 |
| 5 | 20 | 20 |
| 6 | 20 | 20 |

Each affected stage contains 2,989,885 nonfinite model parameters; optimizer slots
are excluded from this count. Full seven-stage synthetic forward failed. Using
the [official README](https://github.com/google-research/google-research/tree/master/metapose)
one-stage setting, all expected model objects matched and synthetic outputs
`x_opt (1,71)` / `fwd (1,68)` were finite. Thus it is incorrect to infer that all
seven stored objects must be used for a valid baseline merely from their presence.
No zero-filling, NaN replacement or silent stage dropping was used.

Checkpoint data SHA-256:
`02905b913ed2bd746bf9c673dd24ff0fd55dfba0a38388c17d0d1f3c6a88e806`.
Checkpoint index SHA-256:
`fdca04a7608c950f4554629f30b1dd63c55df601121b722a1932e63ffe672ce1`.

The separate MetaPose compatibility environment now uses TF CPU 2.15.1, TFP
0.23.0, TFDS 4.9.3, tensorflow-metadata 1.14.0, protobuf 3.20.3,
googleapis-common-protos 1.62.0, Keras 2.15.0 and NumPy 1.26.4; `pip check` passed.
Legacy Adam resolves the old-checkpoint optimizer incompatibility. The project
PyTorch environment was not changed. These are dependency/checkpoint checks, not
a MetaPose result on the skiing datasets.
