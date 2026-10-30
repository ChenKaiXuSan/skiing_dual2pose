# Pegasus full-chain preparation

User authorization: synchronize the full inventory, configure all experiment paths/dependencies, check external baselines, and permit multiple allocated nodes. Do not start full reruns, edit the manuscript, or overwrite results.

Design: preserve the archived scientific recipes and inputs; create a new NQSV job DAG with one task per allocated node and explicit dependencies. A preparation command writes plans only. Per-job execution requires a separately submitted scheduler job and successful upstream artifact validation. All output belongs to a new runtime/plans namespace. Opt-in input-path relocation leaves original manifests and their content hashes unchanged. Keep PyTorch and TensorFlow environments separate.

- [x] Synchronize inventory into the repository and verify SHA256.
- [x] Add an opt-in source-path resolver and test path boundaries, unchanged originals, missing targets and hash preservation.
- [x] Generate a new multi-node DAG from the approved recipes, including main/core training and stable Ski MetaPose, with immutable inputs and fresh outputs; validate dependencies and all 38 training configurations.
- [x] Provide safe per-job NQSV submission/execution and prepare-only commands. Test no-overwrite, unmet dependency rejection and job selection.
- [x] Run bounded GPU checks of DeciWatch, STRIDE and MetaPose; check frontend assets and CLI/import dependencies.
- [x] Reconcile source, config and package versions; document evidence and any remaining limitations. No full DAG submission.

Review focus: nested JSON command arguments; original metadata containing absolute paths; environment-specific Python/CUDA libraries; jobs sharing outputs or using historical learned weights; GPU visibility restricted by NQSV. Missing scientific inputs fail rather than being silently substituted.
