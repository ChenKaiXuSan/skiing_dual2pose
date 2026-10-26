# Final independent review of the IVC baseline expansion

Date: 2026-09-06

## Verdict

No remaining material correctness blocker was identified in the reviewed baseline implementation, measured result records, or updated main-table interpretation. The expanded Tables 2 and 3 can use the final 12 methods per dataset, subject to the controller completing its separate PDF/render and evidence-package checks.

This was a bounded read-only audit; only this report was written. No training, broad test suite, archived experiment, or checkpoint was changed.

## Directly checked

- Read the raw-cache exporter/loader, pose baseline models and runner, geometry export and runner, DLT-residual model, table renderer, baseline protocol, and main manuscript Results/Discussion/Conclusion passages.
- Raw poses remain pre-canonicalization cache inputs. Both training runners form the complete seeded shuffled batch before canonicalization, then split microbatches. Evaluation retains the partial final batch and uses the declared dataset-specific batch size. The intentional archived batch-anchor and training drop-last policies are accurately disclosed.
- Learned pose baselines consume only left/right pose tensors. DLT-residual features contain only canonical DLT, log-scaled reprojection residuals, and validity. Geometry extraction reads estimated SAM3D 2D coordinates and supplied calibration; GT is not an input to its prediction path.
- Unity checkpoint selection uses validation all-15 MPJPE; Ski uses the fixed final epoch. Both learned runners first load test inputs after this selection/budget. The corrected Ski geometry model was trained separately with the same prediction-only pelvis conversion used in evaluation.
- The Ski coordinate correction subtracts the predicted midpoint of common-13 joints 4 and 5 per frame. The source audit establishes the root-relative annotation convention and internally verifies calibration; no target-fitted scale or similarity is introduced. The raw world-coordinate diagnostic is preserved separately and excluded from the final rows.
- Reprojection-gated DLT uses the stated 25-pixel maximum two-view residual, temporal interpolation, nearest endpoint extension, and ungated fallback for wholly rejected finite tracks. Strict geometry loading rejects incomplete/invalid evaluation coverage.
- Metric accumulation sums pointwise distances in double precision and divides by actual point counts; acceleration remains within sequence. The renderer now enforces all sample, joint-frame, and acceleration denominators, resolving the last table-admission finding in initial_review.md. It also requires the corrected Ski geometry flag.
- Read all 24 final result JSONs. Every recorded model/runner source hash resolved to a current source file or preserved source_at_training snapshot. Rehashed every selected checkpoint referenced by the results; all matched their recorded SHA-256 values.
- Read all eight final learned-model histories: each contains exactly epochs 0--99, with 192,512 Unity or 144 Ski training entries per epoch. All four Unity selected epochs equal the minimum validation all-15 MPJPE epoch: MLP 65, TCN 20, SmoothNet 89, and DLT-residual MLP 71. All four Ski methods select epoch 99.
- SmoothNet architecture and loss match the locally documented official-source adaptation (commit c03e93e), with the 30-frame window and training budget explicitly stated. The original and corrected source snapshots explain later hardening without misattributing the code used to train a checkpoint.
- The manuscript states that the MLP has lower clean-input MPJPE than CanonFuse3D on both datasets, SmoothNet has the lowest pose-only acceleration error, Ski TCN also has lower MPJPE, and calibrated methods use additional inputs. It limits robustness comparisons to the methods actually evaluated under perturbation. No unsupported best-overall claim was found in the updated passages.

## Existing validation evidence inspected

The prior focused reports record 20 passing pose/cache/runner tests after hardening, six geometry tests, and nine final DLT-controller tests including root-trajectory invariance and a real corrected Ski forward/backward/update smoke test. These tests were not rerun during this audit. The earlier review also documents reference-row reproduction against the authoritative full-test frontend evaluation and tensor identity of the archived Unity last and validation-selected checkpoints.

## Scope limits

This review did not rerun full model inference, rehash the large raw caches, independently fetch the official SmoothNet repository, or inspect the newly compiled PDF. Those checks are outside this bounded final code/result review or are being completed by the controller. The reported comparisons remain single-seed, protocol-specific measurements, and Ski measures agreement with a constructed reference; the manuscript now says so.
