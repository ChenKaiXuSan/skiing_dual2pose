# Result-preserving layout revision — 2026-09-17

## Implemented scope

- Removed the former Table 2 (joint convention) from the manuscript and consolidated all required joint-set definitions in Experimental Protocol prose. The old table source remains archived, but is not included.
- Reduced the former Table 4 to Unity only. It is now Table 3, on page 22. All 16 method rows, citations, numerical values and Unity-unavailable entries are retained. The prose explains why Ski native-space results are unavailable; the canonical-space Ski benchmark remains intact in the current Table 2, page 21.
- Replaced Figure 11, page 36, with fresh inference from fixed midpoints of previously unshown clips: pro_2 / 1231 and run_4 / 180. No test-quality selection, RGB cropping, synthetic images, synthesized head landmarks or per-method display rescaling. Four pose panels show left, right, canonical average and frozen Unity-trained fusion.
- Preserved quantitative Results sections, including unfavorable cases, C07, efficiency, front-end adaptation and N-view findings. No model training or benchmark rerun.
- Old figure and pre-edit sources remain recoverable in pre_revision/. The old generator still targets the archived realworld_qualitative.pdf; the manuscript now references the distinct realworld_qualitative_refresh.pdf, so regeneration of old assets cannot silently replace it.

## Reproduction and checks

- Run scripts/generate_realworld_refresh.py in the dual2pose environment. This uses the existing archived checkpoint on CPU; manifest records all 120 NPZ source hashes, checkpoint hash, windows and output hashes. Prediction arrays are preserved in realworld_refresh_predictions.npz.
- scripts/test_realworld_refresh.py checks fixed shared-index midpoint selection, axis-only display transformation and global enclosing limits.
- dual2pose.eval.render_native_output_table --unity-only generates the Unity-only table from archived scores; default two-dataset rendering remains supported.
- Focused tests: tests/test_native_output_table.py, tests/test_native_output_unity_only.py, tests/test_ivc_journal_visuals.py (24 tests); three additional realworld refresh tests.
- review_revision_20260916/verify_integrated_results.py validates 123 numeric cells, the unchanged canonical-table SHA-256, and all 16 native Unity rows against the hash-validated pre-revision table.
- external_baselines/verify_archive.py still validates archived external method scores, native unavailability and retired multi-seed/cross-fold scope.
- Main PDF: 46 to 45 pages, 12 to 11 tables, still 11 figures. Supplement remains 25 pages. Table numbering after removal shifts down one; historical C07 integration notes retain the numbering of that earlier snapshot (current C07 tables are 7 and 8).

## Further presentation recommendations — not implemented

Preserve quantitative content; do not assume that converting a table to a figure saves space.

| Current table | Recommendation | Reason / preservation rule |
|---|---|---|
| 2–3: canonical and native benchmarks | Keep tables | Readers need exact three-metric values, citations and distinct input protocols. Too many methods for a compact bar chart. |
| 4: gate/output-residual retraining | Keep table | Four rows and three metrics compactly expose the acceleration trade-off. |
| 5: region-wise gate statistics | Consider integrating values into the gate-analysis figure or prose | Avoid a separate table if the existing figure already explains the pattern; retain all statistics and their descriptive scope. |
| 6: view angle with bootstrap intervals | Optional error-bar plot | Trends and intervals suit a plot, but exact counts/values should remain in the supplement; avoid duplicating the existing angle plot. |
| 7: matched occlusion method comparison | Keep table initially | All methods, conditions and unfavorable cases remain directly auditable. |
| 8: calibration sensitivity | Best candidate for small-multiple line plots | Plot the actual 0/1/3-degree samples for every condition, with unchanged pose-only references as horizontal lines. Show all seven conditions, including severe occlusion where averaging wins; connecting lines do not imply measured intermediate angles. Keep exact numerical table in supplementary material. |
| 9–10: frozen transfer and adaptation | Table 10 is a good annotated-heatmap candidate | Show every existing MPJPE value with a shared color scale and identify row/column meaning; retain worsened transfer cells. A heatmap improves scanability but may take more space. |
| 11: N-view accuracy/cost | Keep compact table | Latency, throughput, memory and accuracy are different quantities; one scatter plot would discard some of them. |

Priority: Table 8 trend presentation, then Table 10 annotated heatmap. No unapproved conversions or experiment removals were applied.
