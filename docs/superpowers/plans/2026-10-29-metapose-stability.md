# MetaPose stability implementation plan

> **For agentic workers:** Use superpowers:executing-plans. User authorized implementation and rerun; no further approval or integration commit requested.

**Goal:** Reject finite-but-collapsed MetaPose runs, stabilize Ski training, and run a fresh Ski fit/evaluation without modifying active runs or the manuscript.
**Architecture:** Backward-compatible optimizer controls in the wrapper; pure NumPy health checks; separate train-only and export commands; a new frozen-source run directory. No test data is accepted by the training/pilot command. Existing scheduler/source remains untouched.
**Tech Stack:** Python, NumPy, TensorFlow 2.15 isolated environment, CPU pytest, tmux.
**Spec:** User instruction and logs/diagnostics/metapose_ski_20261029/README.md, including confirmed scale collapse.

## Constraints and decisions
- Preserve existing paper/results/current task manifests and sources. No GPU jobs until existing GPU 1 jobs are finished. Do not interrupt either GPU.
- Per-frame/common13 scoring unchanged. Ski train/test split unchanged. Train-only stability checks do not load GT 3D or test arrays.
- Start with Adam 1e-5 and global gradient clipping at 1.0. These are engineering choices, not test-selected hyperparameters. Log pre-clipping gradient norms.
- Keep 3 stages, 300 epochs each, batch128, seed42 and fixed final-epoch selection for final fit. No best-test or best-training checkpoint substitution.
- Health uses projected pose radius relative to normalized input radius (0.1 to 10), at most 1% bad frames, positive camera scales with median >=1e-4, and final full-training 2D MSE <= initial S1 MSE and <=5x the best full-training MSE. The bounds identify gross degeneration and are fixed before any new test read.
- Full training-set loss and geometry checked each epoch, persisted before errors; final stage gate must pass before the next stage/export. A failed fit stays failed with artifacts preserved. No silent retries.
- Training-only pilot uses a deterministic subset of training frames, all 3 stages; short real-data checks plus full-run quality gates cannot guarantee convergence of all future runs.
- New source copy and independent wait launcher cooperate by waiting for completion of all existing ordinary GPU1-compatible jobs; GPU0 Unity stays active. Check actual device process/VRAM state before start. If existing exclusive efficiency becomes ready, wait until it finishes.

## Review focus
- Finite constant predictions, localized degenerate frames, NaN/Inf and bad reference shape must fail health checks.
- Train-only command cannot open test data; pilot cannot be exported as final run.
- Missing/tampered stage/checkpoint/health metadata prevents export.
- Existing output directory must be rejected, even after a previous failure.
- Existing GPU jobs and frozen code must not be edited or killed.

## Steps
- [x] Add red tests for health checks and quality-gated export; implement and run green.
- [x] Add optimizer controls and a real TF clipping/freeze/reload regression; run CPU tests.
- [x] Implement train-only fit / frozen prediction export with audit artifacts and no overwrite.
- [x] Run train-only CPU pilot; revise only from training evidence if necessary and record all attempts.
- [x] Freeze source/config and run independent code review. Verify protected originals.
- [x] Launch resource-safe fresh full Ski train -> train quality gate -> test export -> per-frame evaluation -> result validation in new directory; report actual state.

## Execution ledger
- Ruling: user explicitly authorized this repair plan; proceed without additional approval. A separate frozen code copy provides runtime isolation while branch changes remain reviewable; no git worktree/commit required.
- Training-only pilot v1: lr1e-5/global clip1 with original random residual heads failed at stage1 epoch11 (3.125% small-radius frames). Preserved, never exported. This validates the new failure gate.
- Ruling: initialize only residual output heads to zero, retaining nonzero hidden/pose embeddings. Evidence: v1 initially disrupted S1 and drove scale down; identity initialization exactly preserves previous solution on stage insertion. This is an explicitly disclosed training adaptation, not an official-code bug fix. Risk: training behavior differs from the original recipe; do not label it exact official reproduction.
- Pilot v2: 128 train frames, 3x120 epochs, train MSE 0.040273 ->0.014933 ->0.008019 ->0.005258, no geometry failures; no test input opened. Full-training-coverage CPU pilot also scheduled.
- Final reviewer: no clipping/freeze/identity/coordinate/leakage bugs found. Two important gate gaps fixed: quality must match hashed final history; export enforces current adapter code hashes. Three regression tests observed fail then pass.
- Resource launcher adds GPU-only train-data pilot before full fit, waits for all old ordinary GPU jobs, checks live GPU processes twice, and yields to old exclusive profile when Unity is finished. No changes to old scheduler/source/manifest.
- Ruling: preserve full CPU pilot as numerical training evidence even though later changes only strengthened verifier/logging checks; frozen final implementation also runs a fresh GPU train-only pilot before full fitting. No test-based parameter selection.

- Completed: full-coverage CPU pilot 4350 frames x3x30 epochs, all stage gates passed; MSE0.041599->0.003368. No test inputs opened.
- Final tests:542 passed/6 skipped/5 warnings/24 subtests; TF3 passed. Protection3864 unchanged. Persistent frozen-source rerun launched and verified waiting_for_gpu; full training not yet started.
