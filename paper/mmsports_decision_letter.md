MMsportsのコメント
Reviewer 1（Rating: 8 / Clear Accept）
全体評価：非常に肯定的。
本研究は、事前較正が困難な実環境スポーツ映像に対して、2視点の単眼3次元姿勢推定結果をcanonical pose space上で融合する実用的な手法であると高く評価された。Body-centric canonicalization、cross-view attention、temporal refinement等の設計も妥当であり、遮蔽を想定したmasking実験や、実データの限界を明示している点も評価された。
主な指摘：
    • 現在はSAM 3Dへの依存が大きいため、VideoPose3D、PoseFormer、MotionBERT等、異なる前段姿勢推定器に対する汎化性能を検証するとよい。 
    • Reliability gateの重み が、実際の姿勢推定誤差と相関する「信頼度」なのか、単なる適応的な融合重みなのかを分析する必要がある。 
    • 現在は2視点映像の完全同期を仮定しているため、わずかな時間ずれに対する感度・頑健性を評価すると、実用性がさらに高まる。

Reviewer 2（Rating: 5 / Marginally Below Acceptance）
全体評価：手法の実用性と空間精度の改善は評価する一方、完全同期の仮定を主要な問題として指摘。
特に「calibration-freeな実環境スポーツ」を対象としながら、カメラ間の完全なframe/sub-frame synchronizationを前提としている点について、実環境では非現実的であると強く批判された。
主な指摘：
    • 高速スポーツでは数ms～数frameの時間ずれでも大きな位置誤差につながるため、時間同期誤差に対するrobustness評価が必要。 
    • Unityデータを用いて、±0.5、±1、±2 frame等のtemporal offsetを人工的に付与し、MPJPEやAcceleration Errorの変化を評価すべき。 
    • Takahashi et al.（CVSports 2018）など、unsynchronizedかつuncalibratedな多視点カメラを扱う先行研究との比較・議論が不足している。 
    • 現在のcross-view attentionとscalar reliability gateは2視点に特化しており、3台以上のカメラへの拡張性について議論が必要。

Reviewer 3（Rating: 6 / Marginally Above Acceptance）
全体評価：問題設定と基本的なアプローチは妥当であるが、手法の記述と実験がまだ十分ではない。
特に、実装に必要な細部が省略されているため、現状では再現性が十分ではないと指摘された。
主な指摘：
    • Missing jointsの処理、cross-view alignmentの方法、誤差関数、許容するscale/rotation等、手法の詳細を明確化する必要がある。 
    • Abstractで、本研究の3D poseがworld-space poseではなく、root-relative / canonical 3D poseであることを明確に定義すべき。 
    • 完全同期だけでなく、異なるカメラ間のsampling rate差についても検討が必要。 
    • 高速スポーツを対象としているため、motion blurの影響について議論すべき。 
    • Fig. 3等について、数値を提示するだけでなく、結果の意味をより詳細に考察する必要がある。 
    • 現在は主にleft/rightの反対視点という理想的な条件で評価しているため、視点間角度が変化した場合の性能を評価すべき。 
    • 手法自体はスキーに限定されないため、他のmulti-view sports datasetでの評価も望まれる。

結論
3名とも研究課題と基本的な融合アプローチ自体は概ね評価している。特に共通して指摘された最大の課題は、カメラ間の完全な時間同期を仮定している点である。今後の拡張では、①時間ずれへの頑健性、②異なる姿勢推定器・視点配置への汎化、③3視点以上への拡張、④reliability gateの解析、⑤手法記述と実験考察の充実、が主要な改善点となる。最終判定は Accept (Oral) であった。

次に追加実験
    1. 時間ずれに対する頑健性評価
3名のReviewerが共通して指摘した最重要課題である。Unityデータ上で ±0.5、±1、±2、±3 frame などの時間ずれを人工的に付与し、MPJPE、PA-MPJPE、Acceleration Error等の変化を評価する。さらに、時間ずれを自動補正するTemporal Alignment機構を導入し、その有効性を検証する。
    2. カメラ視点角度に対する頑健性評価
現在は左右の反対視点に近い条件が中心であるため、30°～180°程度まで視点間角度を変化させ、多視点融合の効果と限界を評価する。
    3. 異なる3D姿勢推定器への汎化評価
SAM 3Dだけでなく、MotionBERTやPoseFormer等の異なる前段モデルから得た3D姿勢を入力し、CanonFuse3Dによる改善量を比較する。これにより、特定の前段モデルに依存しないことを示す。
    4. Reliability Gateの解析
融合重み (\alpha) と実際の関節誤差との相関を解析し、(\alpha) が信頼度として機能しているかを検証する。相関が弱い場合は、「Reliability Gate」ではなく「Adaptive Fusion Gate」として位置付け直す。
    5. 遮蔽・欠損関節に対する追加評価
既存のmasking実験を拡張し、欠損率を10～50%程度まで変化させた性能低下を評価する。また、欠損関節の入力表現やloss処理について明確化する。
    6. Motion Blurの影響評価
高速スポーツを想定し、異なる強度のmotion blurを加えた映像から3D姿勢を推定し、前段推定誤差がCanonFuse3Dにどの程度影響するかを評価する。
    7. 既存の非同期・未較正手法との比較
Takahashiら（CVSports 2018）など、unsynchronizedかつuncalibratedな多視点姿勢推定手法との関連を整理し、可能であれば比較実験を追加する。
    8. 3視点以上への拡張性評価
現在のDual-view構成を3視点以上へ拡張し、Unity上で2、3、4視点の場合の性能を比較する。これにより、N-viewへの拡張可能性を検証する。    


以下是原文
CanonFuse3D: Calibration-Free Two-View 3D Pose Refinement via Body-Centric Canonicalization for Skiing Videos
Official Reviewby Reviewer WyH107 Aug 2026, 22:46 (modified: 11 Aug 2026, 15:18)Program Chairs, Reviewers, AuthorsRevisions
Review:
This paper addresses the challenge of 3D human pose estimation in high-speed sports (specifically skiing) without requiring pre-calibrated multi-camera setups. The authors propose CanonFuse3D, a post-estimation refinement framework that takes two independent, synchronized, uncalibrated 3D pose streams (e.g., predicted via monocular estimators like SAM 3D) and refines them directly in pose space.

Strength:

Practical and Relevant Problem: Traditional 3D multi-view triangulation often fails in real-world sports filming due to dynamic, uncalibrated, or moving cameras. Focusing on post-estimation refinement in canonical pose space is a highly practical approach for outdoor sports analytics.
Methodological Soundness: The design choices—combining body-centric canonicalization, pairwise similarity alignment, cross-view attention, and temporal depthwise convolutions—are well-motivated and work together effectively to handle view-dependent ambiguities.
Rigorous Robustness Testing: The masking experiments (random joint, distal joint, and temporal maskings) provide valuable insights into how gracefully the model degrades under severe occlusion or tracking failure.
Clarity and Scientific Honesty: The authors are explicit about the dataset limitations, particularly acknowledging that the Ski-PTZ-Pose metrics evaluate consistency with a fused reference rather than an absolute ground-truth MoCap.
Weakness:

Dependency on Front-End Estimators (Cascaded Errors): Since this is a post-estimation refinement framework, its output is heavily bounded by the monocular front-end (SAM 3D). As noted in Section 5.4, if both monocular streams suffer from identical severe failure modes, the refinement model cannot recover the correct pose. It would strengthen the paper to evaluate performance across different front-end monocular estimators (e.g., VideoPose3D, PoseFormer, or MotionBERT) to demonstrate estimator-agnostic generalization.
Uncertainty Calibration vs. Learned Gating Weight (alpha): The paper explicitly notes that the gating weight ‭alpha is a learned fusion signal rather than a calibrated probability uncertainty. However, in Section 3.8, the authors refer to it as a "reliability gate". The paper would benefit from a deeper analysis of whether ‭alpha‬ actually correlates with true error or joint distance to ground truth, or if it simply acts as an adaptive smoothing weight.
Temporal Synchronization Assumption: The current pipeline assumes strictly synchronized multi-view frames. In real-world handheld or PTZ skiing capture, frame-level temporal offsets (asynchrony) frequently occur. Discussing or evaluating how sensitive the bidirectional cross-view attention is to slight frame misalignments would increase practical value.
The paper is well-structured, addresses a clear real-world gap in uncalibrated sports pose estimation, and presents solid experimental validation. Addressing the points regarding front-end generalization and gating weight analysis will make the paper even stronger.

Rating: 8: Top 50% of accepted papers, clear accept
Confidence: 5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature
Paper Decision
Decisionby Program Chairs06 Aug 2026, 18:39 (modified: 07 Aug 2026, 06:16)Program Chairs, Reviewers, AuthorsRevisions
Decision: Accept (Oral)
While the proposed post-processing pose fusion is practical, the framework relies on unrealistic camera synchronization and lacks sensitivity analysis to temporal shifts.
Official Reviewby Reviewer pcem03 Aug 2026, 22:43 (modified: 11 Aug 2026, 15:18)Program Chairs, Reviewers, AuthorsRevisions
Review:
While the paper presents a practical and lightweight post-processing framework to refine 3D poses from uncalibrated cameras, the core problem formulation relies on unrealistic premises (perfect frame/sub-frame synchronization), fails to evaluate sensitivity to temporal shifts despite using synthetic data, and represents a conceptual regression compared to existing literature in the same field (e.g., CVSports 2018).

Summary of the Paper: The paper introduces CanonFuse3D, a calibration-free dual-view 3D pose refinement framework designed for sports motion analysis. Operating entirely in the 3D pose space without access to raw images or camera parameters, the method converts monocular 3D pose streams into a body-centric canonical space. It utilizes bidirectional cross-view attention, a temporal refiner, and a reliability-gated fusion mechanism to improve spatial accuracy and temporal smoothness.

Strengths:

Practical and Elegant Task Formulation: Addressing 3D pose refinement as a post-estimation process--fusing independent single-view outputs directly in the 3D pose space--is a highly intuitive and practical approach for real-world sports analysis.
Well-Designed Fusion Pipeline: The combination of body-centric canonicalization, cross-view attention, and reliability-gated fusion effectively handles view-dependent occlusions and depth ambiguities under ideal conditions.
Solid Evaluation on Spatial Accuracy: The experiments on both synthetic (Unity) and real-world (Ski-PTZ-Pose) datasets demonstrate noticeable improvements in MPJPE and acceleration error compared to baseline monocular estimators and simple averaging.
Weaknesses:

Contradictory and Unrealistic Premise of Perfect Synchronization: The paper emphasizes a "calibration-free" setup for unconstrained outdoor sports. However, assuming perfect frame-level and sub-frame synchronization across unconstrained cameras is highly contradictory. In real-world sports recordings, temporal offsets (both integer frame lags and sub-frame shutter timing phase shifts) are inevitable. In high-speed motions like skiing (e.g., 20 m/s), a sub-frame delay of just 10 ms corresponds to a ~20 cm spatial displacement. Because the framework relies on frame-wise cross-view attention and velocity features, even minor temporal desynchronization could severely distort geometric alignment and reliability estimation.
Missing Sensitivity Analysis in Synthetic (Unity) Experiments: Given that the authors utilize a synthetic dataset (Unity), introducing temporal offsets (+/- 0.5, +/- 1, +/- 2 frames) for evaluation would have been straightforward. The lack of any sensitivity analysis regarding temporal shifts raises significant concerns about the robustness and trustworthiness of the proposed framework.
Comparison with Existing Literature and Conceptual Regression: The paper misses crucial related literature in the multi-view sports pose estimation domain. Specifically, prior work such as Takahashi et al. ("Human Pose As Calibration Pattern: 3D Human Pose Estimation with Multiple Unsynchronized and Uncalibrated Cameras", CVPRW CVSports 2018) addressed both uncalibrated and unsynchronized camera setups simultaneously by leveraging human motion. Compared to such prior literature, assuming perfectly synchronized video streams while claiming a "calibration-free practical setup" feels like a conceptual step backward in terms of problem formulation.
Limited Scalability to N >= 3 Cameras: The architecture (specifically the scalar reliability gate alpha and bidirectional cross-view attention) is tightly coupled to a dual-view setup (N=2). Unlike traditional multi-view approaches where adding more cameras enhances constraints and accuracy, this formulation cannot naturally scale to arbitrary camera counts without major architectural modifications.
Suggestions for Future Revision:

Thorough Sensitivity Analysis on Temporal Offsets: To prove the practical utility of the framework, the manuscript should include a quantitative evaluation (e.g., using the Unity dataset) showing how MPJPE and Acceleration Error degrade when temporal offsets (e.g., +/- 0.5, +/- 1, +/- 2 frames) or sub-frame phase shifts are present.
Clarification on Multi-View Scalability: The authors should explicitly discuss how the proposed body-centric canonicalization and reliability-gated fusion could be generalized to setups with 3 or more uncalibrated cameras.
Discussion on Unsynchronized Baselines: The paper would benefit significantly from citing and discussing prior works that handle unsynchronized setups (e.g., Takahashi et al., CVPRW 2018) to clearly contextualize the trade-offs and limitations of the proposed approach.
Rating: 5: Marginally below acceptance threshold
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
The paper proposes a method to merge two 3D pose sequences, which have been independently estimated from two temporally and clockwise synchronized, but uncalibrated cameras. This is motivated by the scenario where coaches sometimes spontaneously use additional cameras to record sports actions like skiing. Unfortunately, the paper is a little bit premature. The overall approach seems reasonable, but the description of the core process repeatedly omits small details, making it difficult to understand exactly how something is done and thus rendering the entire work somewhat irreproducible.
Official Reviewby Reviewer pnc123 Jul 2026, 15:33 (modified: 11 Aug 2026, 15:18)Program Chairs, Reviewers, AuthorsRevisions
Review:
The paper proposes a method to merge two 3D pose sequences, which have been independently estimated from two temporally and clockwise synchronized, but uncalibrated cameras. This is motivated by the scenario where coaches sometimes spontaneously use additional cameras to record sports actions like skiing.

Strengths:

The problem statement is very good if temporal and clockwise synchronization is added. Simply assuming that the 3D sequences are synchronized (temporally and sample-rate-wise) weakens the problem statement.
The overall approach seems reasonable, but the description of the core process repeatedly omits small details, making it difficult to understand exactly how something is done and thus rendering the entire work somewhat irreproducible. For example, the authors do not explain how they handle missing joints. How are they encoded in their framework? Or how are pose-level cross-view sequences aligned precisely? What is your error function, and which transformations are allowed during alignment? Can they scale or rotate independently for each frame?
Weaknesses:

The authors should clearly define what they understand under 3D pose estimation in the abstract. The precise meaning of the term 3D pose estimation is not clearly defined upfront. The authors should clearly define that they are talking about root-relative 3D pose estimation. It is neither located in a 3D world space nor does it preserve the relative 3D orientation from frame to frame anymore. They call it the canonical pose space.
The paper assumes that videos can be easily synchronized by some external signal and never discusses how to achieve that easily. It also does not address how to synchronize the slightly different sample rates of cameras. Two different cameras never agree exactly on the sampling rate. Especially for fast outdoor motion, which the paper wants to address (such as skiing), this becomes an issue.
The authors focus on fast outdoor motion but fail to discuss the impact of blur on the estimated 3D poses.
In section 5, the values in the figures are not sufficiently discussed (e.g. figure 3). They only describe which values were computed without providing precise parameters. Almost nothing is said about the resulting values, except for vague statements like „Can be seen table xyz.“
The authors claim to propose a dual-view 3D pose refinement framework, but all their training and testing involve an opposite-view 3D pose refinement framework. They assume a left and right view, which is a best-case scenario. They do not analyze what happens if the viewpoints of the input sequences become less opposite. Is there still a benefit or would it degrade?
While the problem statement is precise, the overall text is difficult to read and comprehend. Please streamline your explanations to make it easier to understand.
The proposed framework is not specific to skiing. Why haven’t they evaluated their work on at least one other sports dataset that includes multiview recordings and (slightly noisy) 3D ground truth to test their approach more thoroughly?
Rating: 6: Marginally above acceptance threshold
Confidence: 5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature