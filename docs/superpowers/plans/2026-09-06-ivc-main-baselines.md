# IVC main baseline expansion implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent model and geometry tasks; the controller owns data preparation, execution and manuscript integration.

**Goal:** Recompute the approved expanded baselines on complete IVC Unity and Ski-PosePTZ indices and update main Tables 2 and 3 with verified results.

**Architecture:** Keep archived checkpoints, data and results immutable. Add an isolated baseline package, timestamped experiment output, raw-pose cache, deterministic evaluation and provenance. Independent model/geometry implementations feed the controller's training and reporting pipeline.

**Tech Stack:** Existing PyTorch environment, numpy, original dataloaders and canonicalization, LaTeX.

**Spec:** User-approved baseline list in this conversation (2026-09-06): left/right/average, aligned average, 3D quality weighting, MLP, TCN, average + SmoothNet; additional-input DLT, robust DLT, DLT-residual MLP.

## Global constraints
- Preserve all pre-existing logs, checkpoints, manuscript changes and raw data. Work in the active user-named checkout; new experiment modules and timestamped outputs isolate this work without moving the user's active manuscript.
- Unity: full fold_00, train 193320 / val 128880 / test 64440, 30 frames; 15-joint training, report all15 and common13, test batch 256, drop_last=False.
- Ski: indexed train 150 (native loader usable145; documented five filtered camera0 pairs) / val0 / test30, 30 frames, 13 joints, batch 4, drop_last=False. Fixed training budget and final checkpoint; never select on test.
- Canonicalize left, right and target independently using the unmodified archived batch-first-frame implementation. Preserve test order. Raw caches allow training canonicalization after seeded shuffling.
- Training preserves archived shuffle=True/drop_last=True (eligible193320Unity/145Ski; processed192512/144 per epoch). Validation and test retain every final partial batch.
- AdamW lr .001, weight decay .01, 100 epochs, seed 42. Unity train batch 4096, choose validation MPJPE checkpoint; Ski train batch 4, final epoch. Any architecture-specific loss or optimization variation is recorded before test evaluation.
- Direct group uses 3D only. Geometry reference uses estimated 2D and supplied calibration, with its own clearly documented coordinate conversion and validity coverage. Never substitute GT 2D or test-fitted transformations.
- SmoothNet must follow the official architecture/source with explicit window adaptation and attribution; do not relabel a custom temporal MLP as SmoothNet.
- Record full run commands, input/index/checkpoint/code hashes, sample counts, training history and metrics. No placeholder result rows.

## Task 1: Raw data cache and exact reference reproduction (controller)
- [x] Implement `dual2pose/eval/main_baseline_data.py`: raw left/right/target float32 arrays (N,30,J,3), ordered indices and provenance; preserve source dataset semantics.
- [x] Add tests for final partial batches and stream-independent batch canonicalization.
- [x] Re-evaluate archived CanonFuse checkpoints and existing single-view/average rows against archived metrics before admitting new results.

## Task 2: Pose-only baseline methods (independent implementer)
- [x] Create `dual2pose/models/main_baselines.py`, tests `tests/test_main_baselines.py`.
- [x] Interface `aligned_average(left,right)`, `quality_weighted_fusion(left,right)`, `build_baseline(name, num_joints, time_window=30)`; all consume canonical (B,T,J,3), output same shape, no target/confidence/calibration arguments.
- [x] Aligned average uses sequence-level right-to-left Sim3 then equal weighting; quality weighting uses only temporal acceleration and within-sequence bone-length inconsistency.
- [x] Implement residual cross-view MLP, temporal convolution fusion and verified official SmoothNet structure on canonical average.
- [x] Test rigid/similarity synthetic recovery, finite degenerate inputs, deterministic output shape, actual gradients, and SmoothNet source parity.

## Task 3: Calibrated geometry reference (independent implementer)
- [x] Audit Array geometry functions and exact IVC native 2D/calibration data; document frame/joint/coordinate mapping before computation.
- [x] Create `dual2pose/eval/main_baseline_geometry.py` and focused synthetic projection tests. Export indexed DLT and reprojection-gated DLT sequences in a stated reference suitable for the controller.
- [x] Supply DLT residual learning inputs without target leakage; ensure no silently discarded IVC test samples or oracle 2D.

## Task 4: Reproducible training and evaluation (controller)
- [x] Create `dual2pose/experiments/run_main_baselines.py` with cache/train/eval CLI, resume-safe outputs, heartbeat, histories, model hashes and exact MPJPE/acceleration accumulation.
- [x] Run meaningful integration tests on synthetic training and an actual small batch before full jobs.
- [x] Execute full requested matrix, preferentially on idle GPU1; use remaining GPU0 capacity only without disturbing its existing process.
- [x] Audit counts, finite metrics, held-out partition use and existing-result reproduction; independently review new code and address findings.

## Task 5: Main tables and paper integration (controller)
- [x] Generate data-backed `native_reference.tex` and `ski_poseptz_reference.tex`, with direct and additional-input groups and MPJPE/acceleration.
- [x] Update main text, baseline protocol, result interpretation and evidence manifest to match actual measurements; preserve unfavourable results.
- [x] Compile manuscript PDF and inspect rendered Tables 2/3 and references; record verification and report final values and limitations.

## Completion evidence

All five tasks are complete. Results and verification are in `logs/ivc_mmsports_extension/main_baselines/20260906/verification.json`; final independent review is in `final_review.md`. The main paper contains 12 methods per dataset, with 37 focused tests passing and all 72 displayed metric cells verified. PDF Tables 2 and 3 are on pages 25 and 26. Single-seed and constructed-reference limitations are stated in the manuscript.

## 2026-09-10 follow-on: published external methods

The completion evidence above describes the September 6 artifact, before the author's subsequent table consolidation. It does not establish STRIDE or MetaPose results. The current main comparison is merged Table 3; citations now identify the sources of alignment, DLT, temporal-convolution background and SmoothNet without relabeling local combinations as published systems.

See [the external-baseline plan](2026-09-10-ivc-external-baselines.md) for the initial integration design. As of September 13, STRIDE, MetaPose, and the corrected official-protocol DeciWatch adaptation have completed full Unity and Ski evaluations; final status and protocol details are in [the execution record](../../ivc_external_baselines_20260911.md). All previous result files are preserved.
