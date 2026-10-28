# Per-frame protocol fixes

The user authorized fixing the reviewed defects and keeping the active pipelines on per-frame canonicalization. Work in the current checkout, preserving pre-existing edits and experiment outputs.

## Design

All active pose fusion paths and their comparison baselines use the same per-frame representation. Old manuscript analysis remains explicitly marked as archived; legacy weights must never enter a current-protocol model silently. Exported Pocket3 weights include their protocol and configuration. The cosine scheduler advances once per optimizer step.

## Implementation

- [x] Add regressions for moving-camera baseline parity, checkpoint rejection and save/load round trips, and the scheduler's full decay.
- [x] Make the shared canonicalization default per-frame and make archived first-frame calls explicit. Remove the old reference choice from the active Pocket3 interface and update its report metadata.
- [x] Guard direct CanonFuse3D checkpoint loaders and preserve saved configuration in Pocket3 exports/imports.
- [x] Set the scheduler interval to step. Validate with a short CPU Lightning training run.
- [x] Run the complete tests in the existing dual2pose environment; repair portable test fixtures for the four camera-pair tests instead of relying on a machine-specific dataset.
- [x] Review changes and document verification; do not launch research experiments or overwrite saved results.

## Review focus

- Different view motion must not create a false gain over canonical averaging.
- Batch composition must not select a different coordinate frame.
- Missing or first-frame checkpoint tags must be rejected by active loaders.
- Pocket3 saved model configuration and weights must round-trip exactly.
- Scheduler progression must agree with optimizer updates, including gradient accumulation.

## Verification

The initial regressions reproduced the baseline, checkpoint and scheduler defects.
Follow-up regressions reproduced and fixed mismatched saved face-sign settings and
fresh-process loading of a Lightning checkpoint containing a dynamic Hydra log path.
Pocket3 now shares model canonicalization settings across baselines, targets and
metrics, and rejects incompatible comparison settings or skeleton indices.

Final command (existing dual2pose environment, CPU test workloads):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m pytest tests -q -p no:cacheprovider --durations=5
```

Result: 412 passed, 3 skipped, 22 subtests passed. Four warnings came from the
short CPU Lightning smoke test (GPU available but unused, no validation loader,
worker count and an upstream deprecated API). The skipped optional tests require
external artifacts/checkouts. Independent read-only review confirmed the saved
canonicalization-settings finding was fixed; its three focused checks passed.
No research training run was launched, and existing checkpoints/results were not modified.
