# Evidence-preserving prose compression — 2026-09-17

Implemented the author's approved compression direction: remove repeated explanation, especially in Discussion and standard method modules, while preserving experiments, numbers, citations and scientific limitations. No font-size or margin reduction was used.

## Outcome

| Section | Before | After | Reduction |
|---|---:|---:|---:|
| Introduction | 653 | 495 | 158 |
| Related Work | 963 | 691 | 272 |
| Proposed Method | 1,255 | 905 | 350 |
| Experimental Protocol | 1,150 | 1,006 | 144 |
| Results | 3,544 | 3,384 | 160 |
| Discussion | 1,020 | 609 | 411 |
| Conclusion | 196 | 143 | 53 |
| Total | 8,781 | 7,233 | 1,548 |

Approximate prose count excludes captions, tables, mathematics, reference keys, abstract, declarations and bibliography. The reduction is 17.63%; Results alone is reduced by 4.51%.

- Main PDF: 46 → 42 pages, retaining 10 tables and 12 figures.
- Supplementary PDF: unchanged at 26 pages.
- Display-math blocks: 26 → 20 through algebraically equivalent grouping, without dropping method operations.
- Front matter, abstract, author information, disclosure and final declarations are unchanged.

## Scientific boundaries preserved

The manuscript retains fixed-partition/single-seed scope, separate canonical/native comparisons, adapted external-method attribution, additional-input budgets, constructed Ski references, lack of independent real-video 3D ground truth, batch/reference dependence, calibrated-geometry advantages, severe-occlusion failures, and limits of gate and uncertainty interpretation. Unsupported qualitative stability language was removed from the fixed-frame illustration discussion.

## Verification and provenance

- `pre_revision/` preserves the pre-compression main source/PDF and C07/P1 section sources.
- `protected_before.json` records 52 protected artifacts; all hashes remain unchanged, including tables, figures, supplementary source, included sections and bibliography.
- All 28 main-source citation keys and all section headings remain present.
- Embedded figure/table blocks and all decimal-result occurrences in the Results source remain unchanged.
- Integrated-result verification passed for 123 numerical cells; external archive verification passed for 36 cells.
- The 24 focused tests passed: `tests/test_native_output_table.py`, `tests/test_native_output_unity_only.py`, and `tests/test_ivc_journal_visuals.py`.
- Final main PDF compiled without LaTeX errors, undefined references or overfull boxes. Rendered final pages 10 and 36 were visually checked for formula and prose layout.
- `verification.json` contains final counts and source/PDF hashes.

Academic Research Suite guidance was applied as scoped native-LaTeX edits, with evidence preservation rather than a whole-manuscript rewrite. This pass did not change experiment code or rerun experiments, and did not commit or push. Paper/log artifacts remain ignored. Existing author-confirmation declarations are outside this editing scope; these checks are not a submission-readiness certification.
