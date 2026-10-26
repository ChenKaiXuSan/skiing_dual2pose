# 2026-09-16 gate and efficiency integration

This compact evidence supplements, and does not replace, the original external-baseline archive.

## Contents

- `gate_training/{full,fixed_gate,no_output_residual}.json`: exact completed-test records, checkpoint and input hashes, and training configuration. The original 100-epoch histories and weights remain under `logs/ivc_review_revision_20260916/gate_training_full/`.
- `gate_{unity,ski}/`: byte-identical full-split frozen diagnostics for five conditions, eight gate/residual interventions, and the canonical-average control. Per-condition metrics, streaming gate summaries, and the first two fixed examples are retained; full per-frame candidate tensors were not separately dumped.
- `external_efficiency/{stride,deciwatch,metapose}.json`: every measured latency, timing boundary, parameter bytes, input-slice/checkpoint hashes, and archived-prediction parity. New execution source: `dual2pose/eval/profile_external_efficiency.py`.
- `plot_gate_diagnostics.py`: generates the two manuscript figures from this evidence without sample selection.
- `verify_integrated_results.py`: verifies 123 newly typeset numerical cells, sample coverage, and byte identity of both pre-existing benchmark table files.

## Reproduction

Run the plot and verification scripts with the existing `dual2pose` Python environment. The external profiler runs from the repository root, with `--method stride|deciwatch|metapose --output <new JSON path>`; it refuses to overwrite existing output. Select physical GPU 0 using `CUDA_VISIBLE_DEVICES=0`. Run MetaPose in `.cache/external_baselines/metapose_env/bin/python`, separately from the PyTorch environment.

Official source versions used:

- STRIDE: `84200692c8a445972b59afefee687c552aabe082`, persisted in `.cache/external_baselines/STRIDE`; prior in `ckpt/external/stride_latest_epoch.bin`.
- DeciWatch: `5afe2d67005441e2785a8d374d31a91f046a0da6`, restored to `.cache/external_baselines/DeciWatch`.
- MetaPose: google-research `08a8d6736475776f42ffac23b2c13111a28e5795`, sparse `metapose/` checkout restored to `.cache/external_baselines/google-research`.

The previous `/tmp` source/environment paths no longer existed after reboot. The new MetaPose environment uses TF 2.15.1, TFP 0.23.0, NumPy 1.26.4, and CUDA dependencies supplied by `tensorflow[and-cuda]`; the main training environment was not modified. Exact environment packages are in `metapose_environment.txt`. GPU 0 was otherwise idle during timing; another user's GPU 1 job was not modified.

## Interpretation boundaries

New retraining is Unity-only, one seed and one fixed partition. The full model's new MPJPE is not substituted into the original benchmark. Frozen interventions are not retrained architecture ablations. Raw-pose zeroing also affects canonicalization/alignment; it is not equivalent to matched RGB occlusion. Negative results are retained.

The old gate-region table used representative conference numbers (overall 0.497), incorrectly labeled as the current full split. Following the request to integrate the current results, it is corrected to measured 0.6804 with the proper filtered-joint mapping. This is a documented correction, not a change to the accuracy benchmark.

External latency is a method-specific audit, not a fair common-input ranking or end-to-end RGB runtime. STRIDE optimization is included in the full-call result; MetaPose retains FP64. External full FLOP counts and matched-occlusion geometric results are not claimed. C07 remains unexecuted.

## Revision mechanics

Targeted LaTeX patches preserve untouched paragraphs and the two benchmark tables. The academic-paper skill's Markdown HTML block markers are not inserted into LaTeX. `apply_patch` worked for new files but intermittently failed reading existing files with the sandbox's `bwrap` loopback error; exact-context `patch --fuzz=0` was used for those edits. No whole-manuscript regeneration, commit, or push was performed.
