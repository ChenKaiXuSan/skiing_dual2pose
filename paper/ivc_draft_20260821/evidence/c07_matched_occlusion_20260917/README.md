# C07 matched-occlusion manuscript integration — 2026-09-17

## Scope and provenance

Responds to the supplied PDF annotation C07: compare pose-only and geometry-assisted methods under matched corruption and changing geometric-information quality. This is not a formal IVC acceptance decision. The seven conditions and 0/1/3-degree perturbations are the study's protocol choices.

Original outputs: `logs/ivc_review_revision_20260916/occluded_geometry_full/`.
The `results/` JSON files, frontend manifests and runner status are compact copies; large paired caches remain at their manifest paths. `pre_revision/` preserves the main and supplementary sources before C07 integration. No original benchmark or E5 table is replaced.

## Manuscript locations and size

| Location | Added material | PDF location |
|---|---|---|
| Main Section 5.7 | Matched occlusion and calibration sensitivity; approximately 432 prose words excluding headings/tables | pp.30–32 |
| Main Table 8 | Seven conditions, five methods, supplied calibration | p.31 |
| Main Table 9 | All conditions, pose-only references and DLT-residual at 0/1/3 degrees | p.32 |
| Supplementary S14 | Four subsections; approximately 945 prose words excluding headings/tables | pp.19–25 |
| Tables S11–S13 | Full five-method, three-metric matrices at the three angles | pp.21–23 |
| Tables S14–S15 | Detection/reference coverage and output degeneracy | p.24 |

Main PDF: 43 → 46 pages. Supplementary PDF: 19 → 25 pages. The total increase is nine pages under the current layouts, including float movement. The new section spans do not imply nine full pages of new prose. Protocol, joint conventions, Discussion and Conclusion received additional targeted consistency edits. No new figure or bibliographic entry was added; classical DLT cites the existing Hartley–Zisserman entry, and gating/residual controls are identified as study-specific.

## Evidence and interpretation

- Three frontend sets, each 720 streams and 17,089 unique source images; same selected person's paired model-projected 2D and 3D observations. Ground-truth 2D only locates patches.
- Seven conditions, five methods and three angles: 77 independently computed metric cells, 105 displayed cells after reuse of camera-independent methods. Each cell retains 64,440 windows, 1,933,200 frames and 25,131,600 common-13 points.
- Same archived checkpoints, batch size 256, sample order, all-15 canonicalization followed by common-13 scoring. No new training, test-set tuning, multi-seed or cross-fold inference.
- At 0 and 1 degrees, DLT-residual has the lowest MPJPE in six of seven conditions. At 3 degrees, CanonFuse3D leads MPJPE/PA-MPJPE for clean and ratio-0.5 conditions; averaging leads both for ratio-1.0 conditions. All negative cases remain visible.
- Missing detections, invalid triangulation, zero fallback and batch-anchor collapse remain in denominators. Low acceleration on collapsed poses is not successful motion recovery. Calibration-only invariance of pose-only methods follows from unchanged inputs, not a new physical-camera experiment.

## Verification performed

- 455 displayed numerical cells compared directly with archived JSON: 35 + 35 main cells, 315 complete-matrix cells, 35 coverage and 35 degeneracy cells.
- All nine evaluation JSON copies match their original bytes; all 2160 frontend cache hashes match manifests. Evaluation/frontend source and camera-file hashes match recorded provenance.
- Existing `../review_revision_20260916/verify_integrated_results.py`: 123 numeric cells pass; both benchmark table hashes unchanged.
- Existing `../external_baselines/verify_archive.py`: 36 external numeric cells pass; native unavailable rows and retired multi-seed scope preserved.
- Both PDFs compile with latexmk. Final LaTeX logs have no undefined references, multiply defined labels, Overfull boxes or LaTeX errors. Existing author-confirmation placeholders and the historical dataset-entry missing-year bibliography warning are outside this revision.
- Poppler rendering and text extraction used for page checks; main Tables 8/9 and supplementary matrices/coverage inspected. Full-matrix table font increased for readability. Native image-view sandbox failure was bypassed with read-only loading of the rendered PNG files.

Paper and logs remain ignored by Git. No commit/push or experiment-code modification was performed for this integration.
