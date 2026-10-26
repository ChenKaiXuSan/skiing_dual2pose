# IVC 投稿推荐审稿人（Suggested Reviewers）— 已定稿

更新：2026-09-18

## 规则（来自 IVC Guide for Authors）

- 提供姓名 + 机构邮箱；指南只说 "several"，Editorial Manager 通常最少 3 人，本次填 5 人。
- 不能是近 3 年（2023-09 之后）与任一作者合作/共同署名的人，不能是同事，不能是 IVC 编委。
- 建议来自不同国家/地区；7 位作者都在日本，日本籍只放 1 人。
- 编辑不一定采用推荐名单。

## 核查记录（2026-09-18，OpenAlex）

- 正向：7 位作者 2023 年以来的 220 位合作者中，没有任何候选人，也没有来自名大 / 上交 / Porto / Linköping / TAU 的人。
- 反向：5 位候选人 2023 年以来的论文里，没有 7 位作者中任何一位，也没有筑波 / NICT / NTT 的共同作者，**唯一例外**见 Fujii 备注。
- 邮箱均取自各自官方页面（researchmap / 上交自动化系 / INESC TEC / 个人主页 / TAU 官方 profile）。

## 正式名单（5 人）

| # | 姓名 | 职务与单位 | 国家 | 机构邮箱 | 推荐理由（可直接填入 EM 的 reason 栏） |
|---|---|---|---|---|---|
| 1 | Keisuke Fujii（藤井慶輔） | Associate Professor, Graduate School of Informatics, Nagoya University | 日本 | fujii@i.nagoya-u.ac.jp | Expert in sports motion analysis and 3D pose benchmarks for athletes (AthletePose3D, AthleticsPose); directly relevant to evaluating monocular 3D pose estimators in sports scenes. |
| 2 | Xu Zhao（赵旭） | Professor, School of Automation and Intelligent Sensing (Dept. of Automation), Shanghai Jiao Tong University | 中国 | zhaoxu@sjtu.edu.cn | Leads work on binocular/dual-view 3D human pose estimation with occlusion handling (RSB-Pose, IEEE TIP 2025; Dual-Diffusion, NeurIPS 2024), the closest problem setting to this manuscript. |
| 3 | Hélder P. Oliveira | Assistant Professor, Faculty of Sciences, University of Porto; Senior Researcher, INESC TEC | 葡萄牙 | helder.f.oliveira@inesctec.pt | Co-author of the markerless multi-view 3D human pose estimation survey published in Image and Vision Computing (2025); broad view of multi-view fusion methods. |
| 4 | Bastian Wandt | Assistant Professor, Computer Vision Laboratory, Linköping University | 瑞典 | bastian.wandt@liu.se | Works on 3D human pose estimation, cross-dataset adaptation (AdaptPose, CVPR 2022) and uncertainty-aware pose lifting; expertise in generalization of pose estimators across datasets. |
| 5 | Raja Giryes | Professor, School of Electrical Engineering, Tel Aviv University | 以色列 | raja@tauex.tau.ac.il | Co-author of FLEX (ECCV 2022), extrinsic-parameter-free multi-view 3D human motion reconstruction; expertise in calibration-free multi-view human pose. |

### 备注

- **Fujii**：未与 7 位作者合作，但 2024 年与筑波大学亀田能成（Kameda）、内田育真共同发表 TeamTrack 和足球战术动作两篇论文；Kameda 与 Kitahara 同属筑波视觉组。按 IVC 规则不构成冲突，但他本人可能因此回避。已决定保留。
- **Zhao**：参考文献 `wan2025rsb` 作者列表缺 Yiming Bao（arXiv/TIP 版为 Wan, Chen, Bao, Zhao），投稿前核对并补上。
- Wandt 是 Assistant Professor（非 Associate）；Oliveira 主职是 FCUP Assistant Professor。

## 替补（如上述任何人被编辑或本人排除）

| 姓名 | 单位 | 国家 | 相关工作 | 备注 |
|---|---|---|---|---|
| Norimichi Ukita（浮田宗伯） | 豊田工業大学 | 日本 | 人体姿态、多视角、体育跟踪 | 顶替日本名额 |
| Qiang Xu | CUHK | 香港 | SmoothNet、DeciWatch（本文基线） | |
| Diego Thomas | 九州大学 | 日本 | 多视角人体 3D 重建 | |
| Chunyu Wang | Microsoft Research Asia | 中国 | AdaFuse、MVGFormer | 工业界 |
| Chen Chen | University of Central Florida | 美国 | PoseFormer（本文前端之一） | |
| Amit K. Roy-Chowdhury | UC Riverside | 美国 | STRIDE (WACV 2025) | |
| Helge Rhodin | Bielefeld University | 德国 | Ski-PTZ 数据集作者 | 先核对与 Kitahara/Kimura 的滑雪合作 |

## 待办

- [ ] 投稿当天在 EM 里逐条填入；如系统要求"opposed reviewers"可留空
- [ ] 修正 `references.bib` 中 `wan2025rsb` 的作者列表
- [ ] （可选）Related Work 补引 AthletePose3D / AthleticsPose
