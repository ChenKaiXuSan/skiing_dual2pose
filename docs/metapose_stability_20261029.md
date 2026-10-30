# MetaPose Ski stability repair — 2026-10-29

This is **MetaPose (adapted)** using pinned official model modules, with a repository-specific training recipe. It is not an exact reproduction of the official experimental protocol. No manuscript files were changed.

## Recipe and protocol

The official model architecture, forward projection, common13 mapping, native-to-input-frame conversion and per-frame direct scoring remain unchanged. The new recipe uses Adam 1e-5, global gradient norm clipping at 1, and zero-initialized residual output heads (camera-delta columns and final pose-delta head only). Other hidden weights keep their original initialization. Deterministic TensorFlow operations are enabled.

Final training remains seed42, 3 stages x 300 epochs, batch128, all 4350 training frame entries, fixed final epoch per stage. No best-test or best-training checkpoint substitution is allowed. Existing train/test identities are unchanged. The train command has no test input argument and loads no 3D labels. New training guards stop a degenerate run before prediction export/evaluation.

Each epoch records full-training 2D loss, gradient norms before/after clipping, projected-pose radius relative to input pose, and learned camera scales. Frozen thresholds are in `dual2pose/eval/metapose_stability.py`. Each final stage must pass geometry, initialization-loss and loss-rebound checks. Export validates checkpoint/history/config/frame-index/code hashes and independently checks the final training quality recorded in the history.

The existing legacy wrapper defaults are preserved for compatibility; its new optimizer and initialization controls are opt-in. This repair runs through `dual2pose.experiments.run_metapose_stable`, and the active Unity process still uses its original frozen source. Future repaired Ski runs should use this new entry point rather than the old unguarded loop.

## Training-only verification

- First pilot (original random residual heads, reduced LR/clipping): failed geometry at stage1 epoch11. Kept in `pilot_cpu_v1`; never exported.
- Identity-initialized pilot: 128 training frame entries, 3 x120 epochs; training MSE 0.040273 ->0.014933 ->0.008019 ->0.005258; all final geometry checks passed.
- Full-coverage CPU pilot: all 4350 training frame entries, 3 x30 epochs; training MSE 0.041599 ->0.006835 ->0.004252 ->0.003368; all final geometry checks passed, zero bad-radius frames. This is a short train-only pilot, not the full 3 x300 final fit.
- Applying the new geometry check to saved **training** projections of the previous run rejects stages2/3 (100% bad frames) and accepts the initialization/stage1.
- Final CPU suite: 542 passed, 6 skipped, 5 warnings, 24 subtests passed. Isolated TensorFlow real-model suite: 3 passed, including clipping, exact identity stage insertion, learning, previous-stage freeze and strict reload.
- Independent review identified two provenance/gate gaps. Both fixed with failing-then-passing regression tests: quality/history agreement and current adapter code hashes. No remaining reviewed blocker. Resource-wait logic separately covered by queued-work, active-GPU and exclusive-profile tests.
- Short pilots support the changes but do not establish full-budget convergence or test accuracy. Full-run gates remain mandatory.

## Independent rerun

Root: `logs/per_frame_common13_v1/metapose_stable_20261029T052359Z`

Tmux: `metapose-stable-20261029T052359Z`

The persistent launcher waits for all previous ordinary GPU jobs to finish, then confirms GPU1 has no compute process and at least 40GB free twice. If Unity has finished, the old exclusive efficiency profile has priority. It does not modify the old scheduler/manifest/source or terminate any process.

Pipeline: fresh GPU train-only pilot -> fresh full-budget Ski training -> training quality/provenance gate -> frozen test prediction export -> unchanged per-frame common13 evaluation -> validated mm summary. Any failure stops this pipeline and leaves its evidence in the new directory; no silent retry or overwrite.

Status is `repair_state.json`; eventual results are `predictions/evaluation/metrics.json` and `result_summary.json`. The latter reports mm. The new run does not overwrite the old completion table; its results must be reconciled explicitly when complete.

At launch: waiting for GPU1; existing Unity training and short evaluation remained active. Protection check: 3864 paper/old-result/frozen-source/completed-Ski files unchanged. This check intentionally excludes actively growing Unity logs.

## Commands

Attach: `tmux attach -t metapose-stable-20261029T052359Z`

New fit CLI: `python -m dual2pose.experiments.run_metapose_stable train --train-dir TRAIN --output-dir NEW_TRAIN --metapose-repo OFFICIAL --require-gpu`

Frozen export CLI: `python -m dual2pose.experiments.run_metapose_stable export --train-run NEW_TRAIN --test-dir TEST --output-dir NEW_PREDICTIONS`

The training/export commands require the isolated TensorFlow environment. Evaluation uses the main dual2pose environment.
