# Pegasus automatic dispatch implementation plan

User authorization: start the approved full Pegasus rerun, automatically advance dependencies across multiple nodes, inspect every task, preserve existing results and paper.

Design: retain the immutable 220-job full_chain_20261030_final_v2 plan. Add a stdlib dispatcher outside its frozen source. Run one locked dispatcher in remote tmux; poll NQSV every 30 seconds with at most 64 in-flight requests. Reuse the verified per-job submission and execution entrypoints. Jobs release dependencies only after successful receipt, artifact hashes and log review. Failed tasks retain their output; unrelated branches continue. Unknown scheduler/submission state never causes duplicate execution. Confirm actual initial running jobs and a subsequent dependency submission before reporting launch.

- [x] Add tests for dependency release, failure propagation, capacity, unknown scheduler state, missing receipts, log checks and duplicate exclusion.
- [x] Implement `pegasus/runtime/dispatch.py` with atomic control status, flock, append-only events and bounded qsub retries. No automatic rerun of started/failed tasks.
- [ ] Run focused tests and independent code review before deployment.
- [ ] Synchronize controller and record checksum/version; launch remote tmux under HP260146/gen_S.
- [ ] Observe initial jobs and automatic dependency advance; record run path, queue state and errors. Do not edit paper or historical results.
