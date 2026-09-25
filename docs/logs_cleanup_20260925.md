# Log cleanup record — 2026-09-25

Deleted untracked, gitignored intermediate artifacts under `logs/`. Every path below was checked against the IVC manuscript package (`paper/ivc_draft_20260821/`), `docs/`, `dual2pose/` and `tests/` and had no exact-path reference.

## Deleted

| Path | Size | Last modified | Files |
|---|---|---|---|
| `logs/_OUTPUT` | 52K | 2026-09-11 | 6 |
| `logs/alpha_vs_mask_plots` | 1.2M | 2026-06-24 | 8 |
| `logs/ivc_p1/smoke` | 8.0K | 2026-08-21 | 0 |
| `logs/ivc_p1/smoke_fixed` | 32K | 2026-08-21 | 4 |
| `logs/ivc_p1/smoke_retry` | 392K | 2026-08-21 | 4 |
| `logs/ivc_p1/temporal_alignment_pre_localfix_20260821_1830` | 168M | 2026-08-21 | 4 |
| `logs/ivc_review_revision_20260916/gate_val_smoke` | 1.9M | 2026-09-16 | 7 |
| `logs/ivc_review_revision_20260916/gate_training_smoke` | 16M | 2026-09-16 | 16 |
| `logs/ivc_review_revision_20260916/occluded_geometry_smoke` | 260K | 2026-09-16 | 21 |
| `logs/ivc_mmsports_extension/masking_smoke` | 204K | 2026-08-21 | 7 |

Total: 187M

## Kept deliberately

| Path | Why |
|---|---|
| `logs/ivc_p1/multiseed_smoke`, `logs/ivc_p1/multiseed_smoke_retry` | `paper/ivc_draft_20260821/evidence/external_baselines/verify_archive.py` reads `fold_1/seed_13/train_unity.log` from both |
| `logs/ivc_review_revision_20260916/occluded_geometry_smoke_v2` | recorded in `evidence/c07_matched_occlusion_20260917/runner_status.json` (the v1 run it superseded was deleted) |
| `logs/ivc_mmsports_extension/view_angle_bs256_recheck` | an independent batch-size-256 recheck of the view-angle analysis, not a smoke run; undocumented, so left for the authors to decide |
| `logs/train_unity`, `logs/train_ski_poseptz` | the frozen checkpoints every IVC experiment builds on |
