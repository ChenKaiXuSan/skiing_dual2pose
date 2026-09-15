# IVC DeciWatch adapted implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. This plan executes the author's explicit request to run DeciWatch. It does not authorize changing manuscript numbers, committing, or pushing.

**Goal:** run a reproducible DeciWatch-style sparse-sample temporal recovery baseline on the complete IVC Unity and Ski test splits.

**Architecture:** use the official DeciWatch Transformer architecture and sample-denoise-recover loss, with a small adapter for the archived 30-frame windows. Train from scratch on each dataset because no official checkpoint is present locally and the released checkpoints target different estimators/datasets.

**Tech Stack:** PyTorch 2.11.0+cu126 in the existing `dual2pose` environment; official DeciWatch checkout `/tmp/ivc-deciwatch-20260911-uw729J/source`, commit `5afe2d67005441e2785a8d374d31a91f046a0da6`.

## Superseded seed-42 protocol

- Input: the archived native left/right SAM3D 3D streams, canonicalized independently with the existing batch anchors (Unity batch 256, Ski batch 4), then averaged over views. Only common-13 joints enter DeciWatch. The target-free input path never opens `target.npy`.
- Window: exact archived 30-frame windows, no cross-window context. `sample_interval=10`; observed frames are indices 0, 10, 20 (3/30 = 10%). Frames 21–29 use the official recovery network, with interpolation only as the network's internal decoder seed. The adapter removes the upstream assertion that `(L-1) % interval == 0` and uses endpoint-hold linear interpolation for the decoder's sparse seed; this is the declared 30-frame boundary adaptation, not the original 31/101-frame setting.
- Model: official `DeciWatch` / `Transformer` modules, common-13 input dimension 39, encoder/decoder hidden dimension 128, 5 encoder and 5 decoder layers, 4 heads, pre-norm, dropout 0.1, linear interpolation, Transformer recovery mode. No random-weight test-only result is admitted.
- Training: dataset-specific from-scratch seed 42, Adam lr 1e-3 with official exponential decay 0.95 per epoch, loss `denoise_L1 + recover_L1` (`lambda=1`, `w_denoise=1`), all frames supervised by archived training targets after the same canonicalization. Unity 10 epochs; Ski 70 epochs. No test data or test metrics for model selection. Unity has validation and selects lowest validation recovery L1; Ski has no archived validation, so the fixed final epoch is used.
- Evaluation: all 64,440 Unity and 30 Ski windows, including partial batches. Save both recovered and denoised outputs. Report MPJPE, PA-MPJPE and unscaled second-difference acceleration with the existing evaluator. Recompute the archived average control in the same pass and require exact agreement before accepting a result.
- Label: `DeciWatch (adapted, 10% sparse temporal recovery, seed 42)`. This is not the original pretrained DeciWatch checkpoint result and does not claim the original paper's efficiency numbers; report the 30-frame boundary policy and training budget.

## Superseded seed-42 tasks

- [x] Confirm official source commit and absence of local DeciWatch checkpoints.
- [x] Add tested canonical-average input loader and 30-frame DeciWatch adapter.
- [x] Run GPU smoke: finite output, observed-frame mask, endpoint policy and one-step finite training.
- [x] Train Ski and Unity full splits without interrupting the running MetaPose Unity job on GPU0 (DeciWatch used GPU1).
- [x] Evaluate and independently rescore saved outputs; verify complete coverage, hashes and average-control equality.
- [x] Record results in `docs/ivc_external_baselines_20260911.md`; no paper rows were added.

## Corrected official-protocol adaptation

- Input: the same archived target-free common-13 canonical two-view average; test targets remain closed until predictions are fixed.
- Boundary: repeat frame 29 once to obtain the official-compatible 31-token window, sample tokens 0/10/20/30, run the unmodified official forward, and crop to 30 frames. This exposes four unique archived source frames (13.3%), which is disclosed explicitly.
- Model: official H36M-FCN-3D architecture settings, commit `5afe2d67005441e2785a8d374d31a91f046a0da6`; only the integer-to-bool attention-mask compatibility cast is added for PyTorch 2.11.
- Training: seed 4321, batch 512, 70 epochs for both datasets, Adam 1e-3 with AMSGrad, and exponential decay 0.95.
- Loss: official denoising L1 on sampled tokens plus recovery L1 on all 31 tokens, weights 1 and 1.
- Selection: minimum Unity validation official loss; fixed epoch 70 for Ski because no validation split exists.
- Evaluation: all 64,440 Unity and 30 Ski windows, existing MPJPE/PA-MPJPE/unscaled acceleration evaluator, exact average-control equality, saved-output re-score, and hash verification.
- Label: `DeciWatch (official-protocol adaptation, common13, 4/30 observed source frames, seed 4321)`; not an original pretrained-checkpoint result.

## Corrected rerun tasks

- [x] Add RED-to-GREEN tests for padding, official masked loss, AMSGrad, checkpoint selection, and official GPU forward.
- [x] Pass a real-data one-epoch Ski smoke including complete evaluation and zero control deltas.
- [x] Run full 70-epoch Ski and Unity jobs without using test metrics for training or selection.
- [x] Independently re-score saved recovered outputs; verify all metric/count deltas are zero and hashes match.
- [x] Record corrected results while retaining and labelling the superseded seed-42 diagnostic artifacts.

## Evidence boundary

The official repository presents DeciWatch as sparse temporal sampling plus dense recovery. Because the archived windows contain 30 rather than `N*Q+1` frames, the corrected boundary adapter supplies four unique source frames (13.3%) after repeat-last padding; this difference must accompany every reported result. It is a temporal post-processing comparison over the existing two-view average, not a new calibrated multi-view method.
