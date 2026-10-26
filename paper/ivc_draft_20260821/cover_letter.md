26 September 2026

Dear Editor,

Please consider the enclosed manuscript, “CanonFuse3D: Reliability-Gated Fusion of Uncalibrated Dual-View 3D Human Pose Streams for Fast-Moving Athletes,” for publication in Image and Vision Computing.

The manuscript presents CanonFuse3D, a human-centric framework that fuses two independently estimated monocular 3D pose streams without camera calibration. Controlled experiments establish its robustness to capture perturbations, its generalization to alternative upstream pose estimators through lightweight adaptation, and its extension to additional cameras. It fits the scope of Image and Vision Computing through its focus on multi-view pose estimation, learned data fusion, temporal robustness, and motion analysis in sports imagery.

A preliminary version of CanonFuse3D has been accepted for oral presentation and publication at the 9th International Workshop on Multimedia Content Analysis in Sports (MMSports 2026), co-located with ACM Multimedia 2026. The proceedings article has not yet been published; the accepted manuscript is supplied with this submission for editorial assessment of the overlap, and it is cited in the journal manuscript as an accepted, to-appear proceedings paper (DOI and page numbers will be added when assigned). The journal manuscript does not restate this relationship in its text; the complete account of reused and new material is given here.

Material retained from the accepted paper:

- the CanonFuse3D architecture (human-centric canonicalization, sequence-level alignment, geometric-motion features, bidirectional cross-view attention, temporal refinement, reliability-gated fusion with residual correction) and its training objective;
- the two quantitative datasets (Unity synthetic and Ski-PTZ-Pose) and the problem formulation.

Material not reused: no numerical table, figure, or qualitative example from the accepted paper appears in the journal manuscript. Every quantitative value is regenerated under the journal protocols (complete 64,440-sequence Unity record; fixed 30-sequence Ski-PTZ-Pose index with archived checkpoint and index hashes), and the qualitative examples are new clips.

Material added in the journal manuscript:

1. expanded main comparisons on both datasets with position, Procrustes-aligned, and acceleration errors, covering single views, canonical, aligned, and quality-weighted averaging, a trained temporal convolutional baseline, SmoothNet, adaptations of the published STRIDE, DeciWatch, and MetaPose methods, and a separately identified calibrated triangulation group;
2. an independent native-output comparison on Unity that evaluates every method before the shared body-coordinate normalization;
3. a controlled gate and output-residual ablation with three retrained configurations, together with frozen-checkpoint gate diagnostics and interventions on both datasets;
4. camera-angle stratification over six azimuth-separation bins with cluster-aware confidence intervals, significance tests, and effect sizes;
5. temporal-offset robustness, including sub-frame conditions, and sampling-rate mismatch analysis;
6. a 45-condition missing-pose analysis covering affected views, corruption patterns, and ratios;
7. an 18-condition RGB-occlusion study propagated through the SAM3D front end, retaining failed detections and reporting both positive and negative gains alongside direct pose-stream corruption;
8. a matched RGB-occlusion comparison in which the same corrupted frames yield paired 2D and 3D estimates for five methods under seven occlusion conditions and three levels of supplied-calibration error;
9. frozen transfer across SAM3D, VideoPose3D, PoseFormer, and MotionBERT, followed by seven leakage-controlled adaptation runs evaluated in a complete 8-by-4 front-end matrix with native-front-end retention checks; and
10. frozen pairwise composition for one to four views with synchronized accuracy--cost measurements, matched fusion-module efficiency audits for the local and published-method baselines, and a validation-calibrated automatic temporal-alignment diagnostic.

These experiments establish findings not available in the accepted paper: CanonFuse3D outperforms every non-learned fusion baseline on both datasets in position, Procrustes-aligned, and acceleration error, while a trained temporal convolutional baseline is competitive in position error and SmoothNet attains the lowest acceleration error; frozen transfer is negative for two front ends and is recovered by lightweight adaptation at a measured cost in native-front-end retention; dense bilateral random corruption and severe random image occlusion define the failure regime; calibrated geometric baselines win only with accurate calibration and lose their advantage under small extrinsic errors; pairwise composition improves accuracy with additional views at a measured latency cost that motivates the dual-view default; and automatic synchronization is directionally asymmetric. The journal submission therefore changes the evidential scope and conclusions without claiming a new model architecture.

All authors have approved the submitted version. The work described has not been published previously except as the accepted MMSports 2026 preliminary version disclosed above, it is not under consideration for publication elsewhere, and if accepted it will not be published elsewhere in the same form. The declarations of competing interest, author contributions, and data availability are included in the manuscript, and the publication metadata of the conference version will be completed once the ACM proceedings record becomes public.

Sincerely,

Kaixu Chen  
on behalf of all authors  
Center for Computational Sciences, University of Tsukuba  
1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577, Japan  
chenkaixusan@gmail.com
