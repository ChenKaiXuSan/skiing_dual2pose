# IVC manuscript revision and conference-relationship log

## 2026-09-17: Evidence-preserving prose compression

- Implemented the approved compression direction using scoped native-LaTeX edits. Main PDF is now 42 pages (previously 46); supplementary remains 26 pages. Approximate prose count is 7,233 versus 8,781 words, a 17.63% reduction, without changing margins or font size.
- Concentrated cuts in Discussion, standard-module explanations, and repeated Introduction/Related Work framing. Results loses only 160 words (4.51%); all experiments, headings, citations and decimal-result occurrences are preserved.
- Grouped algebraically equivalent method expressions, reducing display-math blocks from 26 to 20 while retaining all operations, dimensions and scientific qualifications.
- All 52 protected artifact hashes, 28 main citation keys, embedded table/figure blocks, front matter and declarations remain unchanged. The main paper retains 10 tables and 12 figures.
- Numerical verification passed for 123 integrated cells and 36 external cells; 24 focused tests passed. Final PDF has no LaTeX errors, undefined references or overfull boxes; rendered method/discussion pages were inspected.
- Pre-edit sources/PDF, detailed section counts, preservation checks and final source/PDF hashes are archived under evidence/prose_compression_20260917/. No experiment reruns or commit/push in this prose-editing pass.


## 2026-09-17: Compact native table and calibration line figure

- Removed the redundant Canon column and single-dataset Unity header from Table 3; retained its caption's dataset/protocol identity, all 16 method rows, metric values and citations. Disabled automatic enlargement of the narrower table.
- Replaced main Table 8 by Figure 9 with all seven input conditions, shared axes and the measured 0/1/3-degree samples. Pose-only methods are horizontal references, not newly repeated evaluations. Validated all 35 original numerical cells against archived JSON.
- Preserved the exact original calibration table as Supplementary Table S16 (p.25). Original qualitative Figure 11 is now Figure 12. Main PDF is 46 pages; supplement is 26. This presentation change does not by itself shorten the paper.
- Added reproducible figure code/tests and source-to-figure hashes. Pre-edit sources and main PDF are retained under evidence/compact_layout_20260917/pre_revision/.
- Assessed prose compression separately, without broad rewriting: approximately 8,800 current words to an editorial target of 7,250–7,620, prioritizing Discussion repetition and standard-module explanations while preserving Results and scientific limitations. Details: evidence/compact_layout_20260917/compression_assessment.md.


## 2026-09-17: Result-preserving table simplification and new real-video figure

- Incorporated the former joint-convention Table 2 into protocol prose. The former native-output Table 4 now shows Unity only (current Table 3); all 16 rows, Unity values and citations are preserved. The canonical benchmark remains byte-identical.
- Replaced Figure 11 with fresh CPU inference on fixed shared-frame midpoints of pro_2 (1231) and run_4 (180), using the archived Unity checkpoint and 30-frame windows. Four pose panels use common display limits and only actual body joints; full RGB frames are retained. Predictions, source hashes and selection rules are archived in evidence/layout_revision_20260917/.
- Limited qualitative interpretation to observable output differences and deployment feasibility without independent 3D ground truth. The old conference images remain archived but are not included in the new figure.
- Main PDF now has 45 pages, 11 tables and 11 figures (previously 46 pages and 12 tables). Current canonical/native tables occupy pages 21/22; Figure 11 is on page 36. Quantitative Results content, C07, efficiency and negative findings are preserved.
- Added a backwards-compatible Unity-only table rendering option and targeted tests. Further table-to-figure recommendations are recorded in the layout evidence README; no extra conversions have been applied.


## 2026-09-17: C07 matched occlusion and calibration results integrated

- Added Section 5.7 (pp.30–32), Table 8 (p.31: five-method matched occlusion MPJPE) and Table 9 (p.32: calibration sensitivity). Added Supplementary S14 (pp.19–25) with Tables S11–S15: all three metric matrices, detection/reference coverage, and output degeneracy.
- Responds to supplied review annotation C07, not a new formal IVC decision. Seven conditions and the 0/1/3-degree protocol are the study's implementation choices.
- Reports accurate-calibration benefits of DLT-residual, its retained 1-degree advantage, the conditional 3-degree crossover, and severe-occlusion failure relative to averaging. Discloses same-person model-projected 2D, additional input budgets, frozen weights, all failures, and coupled batch-reference degeneration.
- Updated joint-set conventions, experimental protocol, Discussion and Conclusion. Original benchmark and original E5 tables are retained; the new clean control is not substituted into old results.
- Two PDFs compiled: main 43 to 46 pages, supplement 19 to 25 pages. Seven new tables contain 455 checked numeric cells; 77 independent evaluations expand to 105 displayed method/condition/angle cells, not independent training replicates.
- Archive/source JSON identity, 2160 frontend file hashes, source/camera hashes and existing 123/36-cell verifiers passed. No undefined references, Overfull boxes or LaTeX errors; rendered new pages inspected. Existing author-confirmation placeholders remain outside this experiment-integration task.
- Used evidence-bounded academic revision with scoped native-LaTeX patches; retained pre-revision sources in evidence/c07_matched_occlusion_20260917/pre_revision/. No experiment-code edits or commit/push in this integration.

## Historical stage records


## 2026-09-16: C07 implemented and full experiment launched

- Added matched RGB-to-SAM3D paired 2D/3D export and frozen five-method geometry comparison: seven conditions, right-camera local-y 0/1/3-degree perturbations, original batch/order/common13 protocol.
- Forty-six regression tests and a real validation-only smoke passed. Read-only review findings on auxiliary-weight and input-manifest provenance were fixed; missing detections/triangulations remain in the cohort with explicit fallback/anchor counts.
- Full GPU 0 frontend-to-evaluation workflow launched separately under logs/ivc_review_revision_20260916/occluded_geometry_full/ (runner PID 2268603). Full results and manuscript integration remain pending; no new accuracy claims made.
- Original E5 and both benchmark tables preserved; no changes to manuscript numerical tables, no commit/push. Paper/logs remain ignored.

### Earlier integration snapshot

## 2026-09-16: Completed gate and external-efficiency integration

- Added Table 5 (p.23): controlled Unity retraining, three learned 100-epoch runs plus a deterministic fourth cell. New full-model values do not replace the benchmark.
- Corrected the disclosed gate-region provenance mismatch in Table 6 (p.24), from inherited representative 0.497 to current full-split 0.6804, with explicit SD semantics.
- Added Figure 4 (p.25), Supplementary Table S9 and Figure S3 (p.17), retaining every frozen stress condition and negative outcome. These coupled raw-zeroing diagnostics are not isolated gate or RGB-occlusion proof.
- Added cited external-efficiency Table S10 (p.18): original checkpoint hashes, actual precision, STRIDE optimization and MetaPose initialization/adapters retained. No common-input speed ranking or complete external FLOP claim.
- Targeted academic revision preserved both old benchmark files. Both PDFs compiled (43 / 19 pages), new pages visually checked, 123 new numerical cells and 36 old external cells verified. Relevant tests pass with the restored dependency paths.
- C07 remains unexecuted. No commit/push; paper and logs remain ignored. See evidence/review_revision_20260916/README.md.

### Earlier stage record


## 2026-09-16: Current-draft annotation revision, first implementation stage

- Reworked abstract/introduction around the incremental value of a second monocular pose stream; grouped monocular related work by input interface and added verified RSB-Pose and Dual-Diffusion references.
- Corrected the characterization of MetaPose, separated local baseline and external-adaptation training recipes, distinguished canonical/native scoring, and qualified conclusions using existing severe-occlusion and metric-dependent comparison outcomes.
- Preserved both existing benchmark tables and all their values; archive verification still passes.
- Added matched local fusion efficiency in Supplementary Table S8, explicitly excluding uncounted arithmetic and keeping the original N-view timing unchanged.
- Completed frozen gate interventions on both full test indices; started matched three-variant Unity training after objective parity and full-batch smoke tests. The fourth 2x2 cell is deterministic, not trainable.
- Found a material unresolved provenance conflict: the old gate-region rows match the conference source but do not match the current full-split checkpoint's alpha distribution (0.497 versus 0.6804 overall). Requested author direction before changing those numbers; this table is not verified.
- See revision_execution_20260916.md for evidence, scope limits, process/output locations, and remaining C06/C07/C08 work.

## 2026-09-13: Archived the completed external benchmark matrix

- Froze byte-identical compact result and provenance copies for STRIDE, MetaPose, the corrected official-protocol DeciWatch adaptation, and the independent native-output audit under `evidence/external_baselines/`.
- Verified all six canonical external method/dataset cells at complete coverage: 64,440 Unity windows and 30 Ski-PTZ-Pose windows per method. The archive verifier checks SHA-256 source identity, exact unrounded metrics, finite values, PA coverage, zero control deltas where recorded, and MetaPose completion states.
- Checked all 36 external-method metric cells displayed across Tables 3 and 4, together with the STRIDE, DeciWatch, and MetaPose BibTeX keys and the explicit prohibition on ranking across canonical and native protocols. Tables remain on pages 21 and 22.
- Retired the optional multi-seed cross-fold plan from the current submission scope. Its intended matrix was two folds times seeds 13/42/73 (six 100-epoch trainings); the two retained fold-1/seed-13 launch-smoke logs stop during data loading before any epoch or checkpoint, no formal result was admitted, and the manuscript makes no multi-seed or cross-fold statistical claim.
- All 72 native-output, manuscript-table, PA-table, and journal-visual regression tests pass. The current PDF is up to date and its LaTeX log has no undefined references, overfull boxes, or LaTeX errors.

## Accepted preliminary version

- The CanonFuse3D preliminary paper was accepted for oral presentation and publication at the 9th International Workshop on Multimedia Content Analysis in Sports (MMSports 2026), co-located with ACM Multimedia 2026.
- The accepted paper is treated as a prior peer-reviewed publication even though its ACM Digital Library record, DOI, ISBN, and page range are not yet public.
- The provisional bibliography uses the title in the local camera-ready source. The author must confirm that title against the ACM TAPS proof because the decision-letter title differs.

## Architecture retained

- The CanonFuse3D architecture, human-centric canonicalization, reliability-aware gating, and training objective are unchanged.
- Preliminary numerical comparison and feature-ablation tables are not reproduced as journal evidence. Reused qualitative examples remain explicitly cited to the preliminary paper.

## Current quantitative basis

- Unity accuracy and every controlled extension use the complete 64,440-sequence held-out record.
- Ski-PTZ-Pose accuracy is regenerated from the fixed 30-entry common-13 test index using the predeclared epoch-97 last.ckpt; checkpoint, index, and result hashes are archived.
- The two datasets use separate checkpoints and reference regimes, and their errors are not pooled.

## Additional journal evidence

1. Complete-split upstream-estimator transfer using SAM3D, VideoPose3D, PoseFormer, and MotionBERT.
2. Right-view temporal offsets from minus 2 to plus 2 frames, including half-frame conditions.
3. Right-view sampling-rate mismatch from minus 2 to plus 2 percent.
4. Relative camera-angle bins spanning 0 to 180 degrees, with camera-pair cluster bootstrap intervals, paired tests, and across-bin effect-size reporting.
5. A 45-condition missing-pose grid crossing three affected-view modes, three corruption patterns, and five ratios.
6. Frozen pairwise N-view composition for one to four views, with synchronized serial latency, throughput, peak-memory, and accuracy--cost reporting on an RTX A6000.
7. Validation-calibrated automatic temporal alignment across 33 test cells, including batch-order invariance and reported failure directions.
8. Seven front-end adaptation runs evaluated in a complete 8-by-4 test matrix with native-SAM3D retention checks.
9. An 18-condition image-level occlusion study propagated through SAM3D and the frozen fusion model, including detection failures, negative fusion gains, and a comparison with direct pose-stream corruption.

## Manuscript reframing

- Replaced the conference-centered title with a journal title focused on robustness and generalization.
- Retained a concise title-page note disclosing the accepted preliminary version without using the note to summarize the journal extensions.
- Rewrote the abstract as a standalone summary of the complete journal study.
- Condensed the abstract from 245 to 215 words, foregrounding reliability-aware pose-space fusion while retaining the two primary accuracy results and the main robustness, transfer, and view-scaling boundaries.
- Recast the single Introduction disclosure paragraph around the journal study's operating-range question and its distinct empirical findings.
- Removed conference-history wording from the Method and Discussion while preserving the exact executable preprocessing description and its limitation.
- Renamed conference-centered Results headings so that the manuscript reads independently.
- Updated the cover letter with an itemized eight-part difference statement and a commitment to supply the accepted conference manuscript.
- Replaced planning labels such as P1 and MMSports anchor in reader-visible results text.

## Related Work and reference verification

- Expanded the active Related Work from approximately 463 to 812 words and reorganized it into three focused subsections: monocular and sports estimation; calibrated and calibration-free multi-view estimation; and pose-space refinement, reliability, and generalization.
- Added six directly relevant publications covering an IVC multi-view survey, SportsCap, AdaFuse, STRIDE, PoseAug, and AdaptPose, and restored the existing Ski-Pose citation to the active discussion.
- Verified titles, author lists, venues, years, pagination or article numbers, and DOIs against publisher, Crossref, CVF, ECVA, or author-paper records. Publisher DOI metadata is used where CVF open-access pagination differs.
- Corrected publisher pagination for seven existing IEEE records, added the missing Rhodin et al. and MVGFormer DOIs, and replaced the truncated SAM 3D Body author list with the full author list from the accepted paper.
- Checked each added citation against the source abstract or paper text so that the cited work supports the specific statement made in Related Work.

## Journal-specific visual redesign

- Replaced the conference overview and feature-construction figures with new vector diagrams that present the complete study scope and end-to-end pose-space pipeline without conference-versus-journal labels.
- Removed the duplicated conference masking plot in favor of the complete held-out-split robustness figure.
- Recomposed the two previously reported qualitative outputs into a new 2-by-3 journal panel with explicit source disclosure.
- Regenerated all six quantitative extension figures with one colorblind-safe style, paired line/marker encodings, and automated text-overflow checks for diagram boxes.
- Archived figure provenance in `evidence/figure_provenance.md`; the manuscript no longer directly includes any file under `figures/original/`.

## Results narrative and evidence hierarchy

- Replaced the separate `Core Evaluation` and `Generalization and Robustness Analysis` blocks with one `Results` section organized by scientific question.
- Promoted the provenance-controlled 64,440-sequence Unity evaluation to the primary result and generated `tables/native_reference.tex` for its all-15 and common-13 joint summaries.
- Replaced the preliminary combined comparison with two journal tables: the complete-split Unity reference and the independently rerun fixed-index Ski-PTZ-Pose result.
- Retained the complete-split reliability-gate analysis, removed the preliminary feature-ablation number table, and grouped camera angle, timing/rate drift, and missing-pose evidence under capture perturbations.
- Placed frozen front-end transfer immediately before the complete adaptation matrix, followed by N-view scaling, the synchronization diagnostic, and qualitative evidence.
- Removed the empty baseline-masking pointer subsection and reader-visible conference-process wording from the protocol and results.
- Added float barriers between result groups and before Discussion so result tables and figures cannot be deferred into Discussion, Conclusion, or References.
- Anchored the qualitative figure below its subsection heading and constrained the N-view table to the text width, eliminating the reader-visible order defect and the N-view overfull box.
- Added a generated N-view efficiency table and discussion showing that two views capture the largest accuracy gain at 2.02 ms, while three and four views require 2.98-fold and 5.96-fold serial latency for smaller additional improvements.

## Reproducibility and method corrections

- Corrected the active temporal kernel from 3 to 5.
- Corrected the four regularization weights to 0.05, 0.02, 0.005, and 0.01.
- Corrected the reconstruction description to use the raw canonical inter-view delta.
- Recorded batch size 4096, a 100-epoch maximum, and action-disjoint split counts of 193,320 training, 128,880 validation, and 64,440 test camera-pair sequences.
- Declared normalized coordinate units and the common-13 subset used only for cross-estimator comparison.
- Added checkpoint, split, sample-count, seed, corruption, and upstream-estimator provenance.
- Added the Ski-PTZ-Pose epoch, batch size, empty-validation limitation, fixed test count, common-13 convention, runtime path rewrite, and SHA-256 provenance.

## 2026-09-06: Completed image-level occlusion integration

- Added the Results subsection “Image-level occlusion through the pose front end” after the missing-pose study, with all 18 conditions in the result table and a three-panel image/pose intervention comparison figure.
- Reported the 15 positive-gain conditions and all three random ratio-1.0 failures. The bilateral failure has MPJPE 0.6311 versus 0.4914 for canonical averaging, a 28.4% disadvantage; severe single-view interventions also produce negative gains.
- Distinguished protocol ratio from physical image coverage and defined SAM3D failure rates over pair-aligned frame occurrences in the affected views. Detection failures are reported as associated observations, not a demonstrated cause of the full fusion degradation.
- Updated the abstract, Introduction, Discussion, Conclusion, highlights, cover letter, and reproduction instructions. The alternative-estimator lifting comparison still requires image-detection validation; artificial occlusion of the native SAM3D frontend is now completed evidence.
- Archived four result files and a compact six-condition provenance record with SHA-256 values. The independent representative-cell recheck matches its primary summary row exactly; this is not a repeat of the full 18-cell grid.
- Adopted the shared journal figure style with separate markers for masking patterns and filled/solid versus open/dashed encodings for image/pose interventions. The E4 significance table remains byte-for-byte unchanged.
- Verified all 18 table rows against the source CSV, all four copied-source hashes, and the representative recheck. Both existing E4/E5 artifact tests pass. The rebuilt PDF has 47 pages with no undefined references or overfull boxes; rendered pages 2, 29--31, and 42--43 were visually inspected. E5 Results are on page 29, Table 9 on page 30, Figure 7 on page 31, the updated discussion on page 42, and the conclusion on page 43. Details are recorded in `evidence/image_occlusion_integration_verification.json`.

## 2026-09-06: Completed main Tables 2 and 3 baseline expansion

- Recomputed all 24 method/dataset records: 12 rows per table, comprising nine pose-only methods including CanonFuse3D and three separately identified estimated-2D/calibration references. Added aligned averaging, 3D quality weighting, cross-view MLP, TCN, average plus SmoothNet, DLT, reprojection-gated DLT, and DLT-residual MLP.
- Completed eight new learned-model runs of 100 epochs. Unity selection uses complete validation all-15 MPJPE; Ski uses the predeclared final epoch, with no test-based selection. All results retain the complete 64,440 Unity and 30 Ski test sequences and final partial batches. The Ski native loader admits 145 of the 150 indexed training entries; excluded entries are recorded.
- Reproduced the archived CanonFuse3D journal MPJPE values within 5e-7, preserving the original checkpoints, batch-referenced canonicalization, and ordered test indices. Original logs and checkpoints remain unchanged; pre-expansion manuscript artifacts are backed up under the new run directory.
- Corrected the Ski geometry coordinate convention by subtracting the predicted pelvis per frame before the archived canonicalization, consistent with the source root-relative annotations. This prediction-only conversion uses no target-fitted transform. Preserved the superseded raw-world diagnostic and retrained the Ski DLT-residual model from scratch with the corrected convention.
- Updated the experimental protocol, reference-construction description, main Results, abstract, Discussion, Conclusion, cover letter, and reproduction instructions. The MLP has lower clean-input MPJPE than CanonFuse3D on both datasets; SmoothNet has the lowest pose-only acceleration error. The paper explicitly limits robustness conclusions to the baselines actually tested under perturbation.
- Packaged all 24 source result records, the 36-row metric CSV, source/checkpoint hashes, and compact audit reports. Verified all 72 displayed metric cells, rehashed 65 raw/geometry cache files and the selected checkpoints, and confirmed that legacy table regeneration preserves the expanded tables.
- All 37 focused tests pass; independent final review found no material remaining correctness blocker. The rebuilt 49-page PDF has Table 2 on page 25 and Table 3 on page 26, both visually inspected, with no undefined references/citations or overfull boxes. See `evidence/main_baselines_verification.json` and `evidence/main_baselines/`.

## 2026-09-06: Added PA-MPJPE to both expanded main tables

- Re-evaluated all 12 methods on each complete dataset test index using the existing selected checkpoints. Added separate PA-MPJPE columns for both Unity joint subsets and for Ski common-13, yielding 36 new values and 108 displayed metric cells in total.
- PA-MPJPE uses a per-frame least-squares similarity fit with translation, uniform scale and proper rotation, with no reflection. Each joint subset is fitted independently. The metric alignment is applied only for PA-MPJPE; original prediction preprocessing, checkpoint selection, MPJPE and acceleration error are unaffected. The protocol retains degenerate frames at the reference centroid; no degenerate frames occurred in these evaluations.
- Preserved all original results and backed up the preceding paper/evidence package. The maximum difference in re-evaluated original metrics is 3.96e-8, and every pre-existing four-decimal table value is unchanged.
- Updated the metric definition, geometry-preprocessing clarification, main-table PA comparisons, reproduction commands, and evidence package. CanonFuse3D common-13 PA-MPJPE is 0.0637 on Unity and 0.1269 on Ski; MLP has lower values of 0.0561 and 0.1045.
- The 46 focused tests pass; the metric matches the official VideoPose3D formula on independent nondegenerate fixtures. Verified 108 table cells, 10 checkpoint hashes, 26 test cache/geometry files, and preservation of the PA columns by the older artifact generator.
- Recompiled the 52-page PDF and visually checked Tables 2 and 3 on pages 25 and 26. The Unity seven-column table fits the text width, with no overfull boxes or undefined citations/references. Current verification is in `evidence/main_baselines_pa_verification.json`.

## 2026-09-06: Consolidated extension displays and added a standalone supplement

- Followed the author's explicit choices to retain the automatic-synchronization figure and absolute frontend-adaptation table in the main article. Original Figure 10 is now Figure 8; original Table 11 is now Table 8. Original Table 14 is Supplementary Table S5, and original Figure 8 is Supplementary Figure S2.
- Combined camera-angle errors, gains, camera-pair counts, and bootstrap intervals into main Table 5. Detailed paired-test statistics and the old absolute-error plot are in the supplement. The combined table checks agreement between the summary and camera-pair inference sources.
- Moved representative temporal/sampling and missing-pose tables, the complete N-view numerical table, and synchronization diagnostics into the supplement. Retained the full missing-pose curves, image-occlusion table/comparison figure, frozen-transfer table, N-view plot/efficiency table, and all relevant failure findings in the main text.
- Converted the temporal figure to percentage change from each curve's own unperturbed MPJPE, preserving all seven offset and seven rate conditions. Saved the plotted values and verified all 21 points. Panel labels and axis titles were checked in the actual rendered page.
- Added supplementary.tex/PDF with five tables and two figures, S numbering, and working external references from the main article. The main article now has nine tables and nine figures; the old-to-new mapping is in README.md.
- Preserved all 12 original standalone table files byte-for-byte, including the full PA-MPJPE main tables. Every packaged main-baseline source result hash is unchanged. The previous paper, figures, generator and evidence documents are backed up in the consolidation run directory.
- All 50 focused tests pass. Verified six merged angle rows, 21 transformed plot points, nine external PDF links, document allocation, preserved negative findings, and compatibility with the older figure generator. Both documents compile without undefined references/citations or overfull boxes. The main PDF is reduced from 52 to 46 pages; the supplement has seven pages. Main pages 29/30/34/37 and supplementary pages 1/2/5/7 were visually inspected. Current hashes and locations are in evidence/manuscript_consolidation_verification.json.

## Items requiring author confirmation

See author_confirmation_checklist.md. No DOI, ISBN, page range, ethics statement, conflict declaration, authorship role, or availability promise has been invented.

## 2026-09-06: Clarified four remaining conference-review text items

- Added Takahashi et al. (CVPRW 2018) to Related Work and the bibliography after checking the original paper. Distinguished its 2D-joint-based estimation of camera extrinsics, time offsets, and 3D pose from the approximately synchronized 3D-input refinement setting; no comparison result was invented.
- Defined the output coordinate convention directly in the 240-word abstract and clarified the fixed batch anchor in the method. The description does not relabel the executable transform as per-frame root-relative normalization.
- Specified the sequence-level least-squares similarity objective, unweighted SVD solution, proper-rotation constraint, uniform scale, and variance safeguard. Documented missing-input encoding, corruption before canonicalization, equal participation in alignment, and the absence of input-visibility weighting in the losses. Corrected the masking noise-unit description to raw input coordinate units.
- Added a separate motion-blur discussion connecting exposure-related front-end errors to temporal features, view disagreement, and anchor landmarks. These are explicitly expected mechanisms; a blur experiment is still absent.
- Applied seven bounded patch operations through an anchored working copy with a lossless LaTeX adapter. Preserved 148 of 152 original blocks byte-for-byte and all 29 protected table, figure, and included-section source artifacts. The supplemental PDF is byte-identical.
- Recompiled the main PDF to 47 pages with nine figures and nine tables, no undefined references/citations, and no overfull boxes. Visually checked pages 1, 7, 10, 11, 15, and 41, including the final independent paragraph boundaries. The original PDFs and sources are backed up under logs/ivc_mmsports_extension/reviewer_text_revision/20260906/before/.
- Current locations: output definition on page 1 and pages 9-10; Takahashi discussion on page 7; alignment and missing-input rules on pages 10-11 and 15; masking noise units on page 31; motion-blur discussion on page 41. See evidence/reviewer_text_revision_verification.json.


## 2026-09-06 — Author annotation revision

Refocused the title and main narrative on calibration-free post-estimation dual-view 3D pose refinement via human-centric fusion. Condensed Abstract, Introduction, method prose, protocol, and Conclusion; moved reproducibility details to Supplementary S7–S8; separated the three evaluation metrics; redrew the system overview and added a five-cue feature-concatenation figure. Removed the unpublished conference self-reference at the author's explicit request, retaining overlap disclosure. Verified the 22 remaining cited resources against official sources and corrected bibliographic metadata. Applied upright prose typography and vector fonts. Results content and numerical artifacts were retained apart from the authorized self-reference caption sentence.

The complete annotation mapping, reference audit, source backup, patch report, and visual checks are in `logs/ivc_mmsports_extension/author_annotations/20260906/`.

## 2026-09-07: Author review revision (title, narrative, figures, tables)

- Retitled the manuscript to "Calibration-Free Post-Estimation Fusion of Dual-View 3D Human Pose Streams: Robustness, Front-End Generalization, and Multi-View Extension" to separate it from the MMSports title and to foreground the journal-specific contributions; the title-page note still discloses the preliminary version.
- Rewrote the abstract to start from multi-view capture and introduce two views as the smallest deployable configuration, removed the coordinate-convention sentence, summarized the three datasets in one sentence, and condensed the robustness summary. Keywords now include post-estimation pose fusion and front-end generalization and omit temporal refinement.
- Rewrote the Introduction to start from monocular 3D pose estimation with citations, motivate multi-view and then dual-view capture explicitly, reference Figure 1 where the framework is described, and restate the contributions.
- Related Work: renamed 2.1 to monocular estimation only, condensed the sports paragraph to the works closest to this study, and reduced the Takahashi et al. discussion to three sentences without a difference statement.
- Method: merged the previous architecture figure and feature-construction figure into one figure placed directly after the overview; checked every formula in 3.2--3.8 against the executable model (two-layer encoder, per-direction attention parameters, depthwise-then-pointwise temporal convolution, two-layer gate and residual heads) and wrote out all five loss terms with their weights in 3.9.
- Protocol: Table 1 now states dataset scale (sequences, views, frames, camera-pair counts); a new joint-convention table lists all 15 joints and explains why Unity is evaluated on 15 joints and Ski-PTZ-Pose on the common 13; PA-MPJPE is defined by an explicit per-frame similarity fit; the controlled-evaluation list became a paragraph titled Robustness and Generalization Protocols.
- Results: merged the Unity and Ski-PTZ-Pose comparisons into one common-13 table without the footnote; moved the all-15 Unity table to Supplementary Table S1; commented out the cross-view MLP row of the main comparison at the author's request while keeping the TCN, SmoothNet, non-learned, and calibrated rows; the accuracy claims in the abstract, results, discussion, and conclusion were restated to match the retained rows; promoted the gate analysis and the four robustness studies to subsections; added a 2-by-3 image-occlusion example figure and moved the 18-condition occlusion table to the supplement; reframed the front-end section around adaptation-based generalization with gains relative to each front end's canonical average.
- Discussion and Conclusion updated for the revised comparison scope. Cover letter, highlights, README, and figure provenance updated; the journal-visuals tests were updated for the new title and section order and pass together with the E4/E5 and consolidation tests.
- The commented-out MLP row remains in the table source and in evidence/results/main_baselines__results.json and logs/ivc_mmsports_extension/main_baselines/20260906/; no experiment output was deleted. Source backups of the edited files are in backup_20260907/.

## 2026-09-10: Baseline source attribution without changing results

- Added Umeyama (TPAMI 1991) for the SVD similarity estimator, Hartley and Zisserman (second edition, 2004) for classical DLT, and Bai et al. (2018 technical report) for temporal-convolution background. Bibliographic identities and DOI metadata were checked against the original paper, author/publisher pages, and arXiv.
- Kept the existing SmoothNet (ECCV 2022) citation and clarified that its use after canonical averaging is an adaptation of a published refinement method.
- Explicitly distinguished the local non-dilated TCN, 3D quality heuristic, reprojection-gated DLT, and DLT-residual MLP from reproductions of published systems. Mirrored these attribution boundaries in the supplementary implementation record.
- Limited source edits to main.tex, supplementary.tex, references.bib, and this log; retained every result-table source, numerical evidence file, model, and evaluation protocol. No external-method experiment was added or claimed.
- Pre-edit sources and PDFs are backed up under /tmp/ivc-baseline-citations-20260910-ozmesG/.
- Verification: all 43 files under tables/ and evidence/results/ retain their pre-edit SHA-256 digests. Both PDFs compile successfully (main: 38 pages; supplement: 15 pages), with no undefined references/citations or overfull boxes. New bibliography entries resolve to [20], [24], and [25]; the pre-existing missing-year warning for the Ski-Pose dataset web entry remains. Visually checked main pages 17 and 38 and supplementary pages 11 and 12.

## 2026-09-10: Table citations and external-method integration start

- Added Umeyama, Bai et al., SmoothNet and Hartley--Zisserman references to the appropriate rows in main Table 3 and Supplementary Table S1; the supplement now resolves its own bibliography. Captions explicitly limit the local alignment/DLT/TCN citations to components or architectural background.
- Updated the baseline-table generator labels and attribution regression test, without using its older two-table layout to overwrite the current merged table. Preserved all numeric cells, bolding, input groups and the commented MLP row.
- Added the STRIDE/MetaPose external-comparison plan (DeciWatch backup) and a target-free adapter/CPU diagnostic entry point. On two Unity validation and two Ski training windows, the official STRIDE backbone's random-weight common-13 forward and MetaPose's official stage-1 initialization returned finite outputs with the required shapes. These are component checks, not complete external-method reproductions or new paper results.
- No existing model, training output or numerical result was changed; no new accuracy numbers or placeholder rows were inserted. The plan records the remaining STRIDE skeleton/prior/TTT and MetaPose uncertainty/coordinates/learned-stage gates. The isolated MetaPose compatibility environment leaves the existing conda environment untouched.
- Verification: 144 source metric cells (including commented rows) are unchanged; all 28 files under evidence/results retain their pre-edit hashes. Both PDFs compile successfully (main 38 pages, supplement 15), with resolved citations/references and no overfull boxes; main page 20 and supplementary page 2 were visually checked. Backups: /tmp/ivc-table-citations-20260910-hc8JTg/. Diagnostic evidence: logs/ivc_mmsports_extension/external_baselines/20260910/.
- All 55 focused regression tests pass, including the 13 new input/diagnostic contract tests and five PA-table tests. Full external-method training and evaluation remain pending; this verification establishes only the bounded integration stage.
## 2026-09-11: Added an independent native-output comparison protocol

- Preserved tables/native_reference.tex byte-for-byte (pre-edit SHA-256 ade0b6d43034d7fc60fa88f651fc758f2b184a1892868d83dd9b267c8eca13e6) and inserted a new standalone table immediately after it.
- Separated the existing canonical-space downstream-fusion comparison from a 17-row audit of front-end, pre-Canon, external, geometric, and proposed outputs. Every competing method bypasses Canon; only CanonFuse3D is marked as using its internal body-coordinate transform.
- Added target-free frozen prediction manifests, exact complete-coverage scoring gates, root-translation-only MPJPE, independent reflection-free PA-MPJPE, and sequence-bounded acceleration error.
- Withheld values when no independently supplied common coordinate frame or admissible learned-output gauge exists. In particular, no target-fitted conversion is used to make a row numerically available.
- Added the DeciWatch ECCV 2022 bibliographic record from the official ECVA and Springer publication metadata. Full result generation and final table population are recorded separately after completion.

## 2026-09-13: Split the primary benchmark by canonicalization protocol

- Replaced the previous panel-style presentation by two standalone tables: Table 3 contains only results evaluated after the shared body-coordinate normalization, while Table 4 contains only native/pre-Canon results. CanonFuse3D is therefore present only in Table 3 and the former internal-Canon empty row was removed from Table 4.
- Added the completed STRIDE, DeciWatch, and MetaPose adaptations to the canonical table with their method citations and explicit input/task-budget groupings. Added their Unity native-space results to the no-Canon table; Ski native-space values remain withheld because the constructed annotation reference has no independently supplied mapping into those output frames.
- Updated the protocol and Results text to state the official-protocol DeciWatch window adaptation, MetaPose's additional estimated-2D input, all reported external-method values, and the rule that methods must not be ranked across the two normalization protocols.
- Regenerated Table 4 directly from the frozen 2026-09-11 Unity and Ski score files. A combined 72 native-output, manuscript, and table regression tests pass.
- Rebuilt the 40-page main PDF. Tables 3 and 4 occupy separate pages 21 and 22; citations and references resolve, and the final log contains no overfull boxes, undefined citations, or undefined references.

## 2026-09-18: Visio figures and funding statement

- Replaced Fig. 1 and Fig. 2 with the author-drawn Visio versions (`CanonFuse3D_Formula_Final_A01.svg`, `CanonFuse3D_Formula_Final_A02.svg`), converted to vector PDF with headless Chromium, raster content downsampled to 200 ppi on the 22-inch canvas, and cropped to content (`study_overview_visio.pdf`, 231 KB; `method_pipeline_visio.pdf`, 284 KB). Fig. 1 caption rewritten to describe panels A--E and the qualitative source of the embedded example. The matplotlib figures and their generator are retained but no longer included.
- Added a Funding section before Acknowledgments: JSPS KAKENHI Grant Numbers 24K02978 and 25K03146. The JP-prefix convention for KAKENHI acknowledgements is left for the authors to confirm.
- Updated the journal-visuals test and figure provenance for the new file names. Main PDF: 42 pages, no warnings.

## 2026-09-18: Directory cleanup and funding prefix

- Funding statement now reads "Japan Society for the Promotion of Science (JSPS) KAKENHI, Grant Numbers JP24K02978 and JP25K03146" (JP prefix added; wording adjusted to avoid an overfull line).
- Moved everything not needed to compile, submit, or reproduce the manuscript into `../ivc_draft_20260821_archive_20260918/`: `main_diff.*`, `main_marked.*`, `backup_20260907/`, the two 2026-09-16 review-response working notes, `figures/original/` (conference figures), the superseded matplotlib `study_overview.pdf`, `method_pipeline.pdf`, `feature_details.pdf`, and `realworld_qualitative.pdf`, PNG duplicates of PDF figures, and the unused `sections/main_baseline_protocol.tex`, `sections/p1_protocol.tex`, `tables/joint_convention.tex`, `tables/ski_poseptz_reference.tex`, `tables/view_angle.tex`. LaTeX auxiliary files were deleted.
- `generate_journal_figures.py` and `tests/test_ivc_journal_visuals.py` no longer require the archived conference panels. Both documents recompile from a clean state without warnings; 46 paper-related tests pass. README rewritten to describe the cleaned layout.

## 2026-09-18: Table 1 transposed

- Table 1 now lists attributes as rows (type, cameras, sequences, model input, 3D reference, split, role) and the three data sources as columns, replacing the prose-heavy column layout that occupied half a page. The previous version is archived as `../ivc_draft_20260821_archive_20260918/tables/dataset_summary_v1_columns.tex`. Main PDF: 41 pages.
- Table 1 further trimmed at the author's request: removed the Type and Input rows (redundant with the column headers and Section 4.1), shortened every cell, and reduced the caption to the split definition. Five rows remain (cameras, sequences, reference, split, role). Previous transposed version archived as `tables/dataset_summary_v2_transposed.tex` in the archive directory.

## 2026-09-18: Conference version cited in the text

- Removed the title-page note. The accepted MMSports 2026 paper is now cited in a short disclosure paragraph at the end of the Introduction (architecture retained; no result reused; all quantitative values regenerated), with the bibliography entry restored as an accepted, to-appear proceedings paper. This supersedes the 2026-09-06 removal of the self-reference and satisfies the Elsevier requirement to cite the primary conference source. DOI and pages remain to be added.
- Same day, at the author's request: the Introduction disclosure paragraph was removed again. The manuscript now only cites the accepted MMSports paper in plain form, once in the Introduction and once in the Method overview; the full reused/new-material account lives in cover_letter.md, which was rewritten to cover all experiments added through 2026-09-17 (native-output audit, gate ablation, published-method adaptations, matched occlusion and calibration sensitivity, efficiency audits).
- Added one positioning sentence after the method description in the Introduction ("This article builds on the architecture of [16] and focuses on its robustness, front-end generalization, and extension to additional views."), so that the relation to the cited conference paper is stated without a disclosure paragraph.

## 2026-09-18: Submission package and author-list change

- Removed Nan Luo from the author list at the corresponding author's request (author block, `pdfauthor`, and the CRediT placeholder now refer to six authors). The `imiss` affiliation is kept for Qingyu Wang. Both documents recompile without warnings; main PDF 42 pages.
- Split the former "Ethics and generative-AI disclosure" section into an "Ethics statement" and a separate "Declaration of Generative AI and AI-assisted technologies in the writing process" section carrying the Elsevier template sentence, placed before the reference list as the journal requires; the authors still have to fill in or delete it.
- Added the mandatory graphical abstract (`graphical_abstract.pdf`/`.png`, 13 x 5 cm, 1535 x 590 px) generated by the new `scripts/generate_graphical_abstract.py` from the Fig. 1 skeleton panels, in Liberation Sans (Arial-metric).
- Renamed `highlights.txt` to `Highlights.txt` (journal asks for "Highlights" in the file name; content unchanged, 74-81 characters per bullet).
- Corrected the `wan2025rsb` bibliography entry (RSB-Pose) to include the fourth author, Yiming Bao.
- Cover letter: filled in the corresponding-author signature block and fixed the list conjunction; MMSports DOI still pending.
- Added `suggested_reviewers.md` (five reviewers, institutional e-mails, OpenAlex conflict check) and rewrote `author_confirmation_checklist.md` as the full pre-submission checklist, including the unresolved 7500-word limit question.

- Contribution 1 reworded from "a human-centric framework ..." to "a complete formulation of the human-centric fusion framework ..., together with a controlled ablation of its reliability gate and residual correction", so that it no longer claims the architecture of the cited conference paper as new.

## 2026-09-18: Declarations and addresses drafted

- Title page: full postal addresses added for all four affiliations (CCS 305-8577; Degree Programs in Systems and Information Engineering 305-8573; CiNet/NICT, Suita 565-0871; NTT CS Labs, Atsugi 243-0198, correcting the earlier "Tokyo"). Addresses verified against the institutions' web pages; wording for NICT and NTT flagged for the respective authors.
- Replaced the "author confirmation required" placeholders with drafted text: competing-interest statement (none), CRediT roles for six authors (funding roles taken from the KAKEN records of JP25K03146 and JP24K02978), data and code availability (public GitHub repository; Unity data and checkpoint to be deposited with a DOI upon publication; author-collected videos not shared), ethics statement (consenting adult volunteers, no institutional review), and the Elsevier generative-AI declaration (Claude and OpenAI Codex for readability and language). Each block carries a `% CONFIRM` comment in the source.
- Added `coauthor_confirmation_request.md`, a ready-to-send message asking each co-author to confirm the five items, plus the corresponding-author-only tasks (ORCID candidate 0009-0003-8698-8899, declaration .docx, MMSports PDF).
- Main PDF: 43 pages, no warnings.


## 2026-09-18: Compression round 1 (relocation and de-duplication only)

- Results: protocol descriptions repeated from Section 4.5 and the supplement were removed from Sections 5.2, 5.5, 5.6, 5.7, and 5.8 and replaced by pointers; the pose-corruption mixture, both-view sampling, and temporal-ratio definitions were moved into Supplementary Section S10 so no protocol detail was lost. Per-subsection caveat sentences were reduced to one each; the removed caveats (single seed and fixed partition, conditional bootstrap intervals, reference regimes, batch-shared body reference, uncalibrated gate, timing tolerance, artificial patches and one front end, non-causal detection association, ground-truth 2D tracks for alternative front ends, smoothness versus reconstruction) now form one Limitations paragraph at the end of the Discussion. The motion-blur speculation paragraph was deleted.
- Captions of all 22 main-text figures and tables were shortened to describe what is shown; the gate-region table note was trimmed. No numerical value, table row, or figure file changed; the caption-only edit of tables/native_reference.tex changes its byte hash, and the pinned hash in tests/test_native_output_table.py was updated accordingly.
- Word counts: Results 3,520 to 2,830, Discussion 605 to 483, captions 959 to 682 (about 1,090 words in total). Main PDF 42 to 40 pages; supplement unchanged at 26. All 27 paper-related tests pass. A latexdiff record of every change is in `../ivc_draft_20260821_archive_20260918/before_compression_20260918/round1_diff.pdf`, with the pre-compression sources alongside it.
- Pre-submission audit (2026-09-18): added the missing in-text reference to Table~\ref{tab:c07_occlusion_mpjpe}; removed four never-cited bibliography entries (DJI Osmo, S4, Mamba, Venkat et al.). All cross-references in both documents resolve, all cited keys exist, all figure files exist, all fonts are embedded, supplement numbering S1--S16 / S1--S3 is consecutive, and every checked number in the text matches its table and evidence file. Items still requiring author confirmation are marked by `% CONFIRM` comments in main.tex.

## 2026-09-18: Title changed

- New title: "CanonFuse3D: Reliability-Gated Fusion of Uncalibrated Dual-View 3D Human Pose Streams for Fast-Moving Athletes". The method name is retained; the subtitle now states the gating mechanism, the post-estimation input, and the application, and no longer enumerates study areas. Updated in main.tex (title and PDF metadata), supplementary.tex, cover_letter.md, and the journal-visuals test. Earlier snapshots under evidence/*/pre_revision keep the previous title as historical record.

## 2026-10-26: Desk rejection and transfer to PLOS ONE

- The 2026-09-26 submission to Image and Vision Computing was desk-rejected on 2026-10-26. This directory is kept unchanged as the record of the submitted version.
- A new submission package for PLOS ONE was started in `../plos_one_20261026/` from these sources; see its README for the conversion checklist.
