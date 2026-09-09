# IVC E5 Manuscript Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. Execute inline; this plan does not request delegation.

**Goal:** 将已完成的图像遮挡追加实验整合到 IVC 正文、图表、讨论、投稿材料和证据包，并生成内容一致的 PDF。

**Architecture:** 保留现有论文结构，在姿态遮挡结果后增加图像遮挡结果小节，沿用冻结模型和已完成的 18 组结果。图表由现有生成器输出，论文证据目录保存带校验值的结果快照，最终以编译后的 PDF 核对内容与排版。

**Tech Stack:** LaTeX/latexmk、Python 标准库、Matplotlib、pdftotext、pdftoppm。

**Spec:** 本轮用户要求“你帮我规划一下”，承接 E4 已整合、E5 结果尚未接入正文的检查结论；实验背景见 `docs/superpowers/plans/2026-09-03-ivc-e4-e5-additional-experiments.md`。本文件只交付整合计划，复选框表示后续实施工作。

## Global Constraints

- 论文范围为 `paper/ivc_draft_20260821/`；方法、已有主结果、E4 统计结论及其他追加实验的实验口径保持一致。
- 数据来源为 `logs/ivc_mmsports_extension/image_occlusion/`，使用已完成结果；保留原始输出，不启动新训练或 GPU 评估。
- 冻结模型 SHA-256 为 `869a2217f8676c0ada75ed3c9a3c82a9b8efbb105749f6ffb8bef71e9172f50f`；fold 0、seed 42、15 joints、batch size 256、64,440 条测试序列。保持现有批次参考坐标变换口径。
- MPJPE 使用数据集坐标单位，不改写为毫米。单个条件含 17,089 个独立源帧；六条件共 102,534 帧次推理，不能称为 102,534 张互不重复图像。
- 18 组结果包含全部三组负增益；15/18 是描述性统计，不据此声称统计显著或独立重复实验的一致性。
- 图像遮挡与姿态遮挡的 ratio 是各自协议参数，相同数值不表示相同物理遮挡强度；ratio 1.0 也不表示整张图像被完全遮住。
- SAM3D 检测失败样本按既有规则保留为带失败标记的零姿态。检测失败率与融合误差超过阈值的失败率分开表述。
- 只完成这次结果整合所需的局部修改；不更改训练、推理、指标计算、数据划分或会议版论文，也不改动无关工作区文件。

## 已确认状态与关键结果

截至 2026-09-06 上午的现场检查：六组前端都完成 720/720 个相机序列；18 组评估、图表生成、代表条件复跑正常结束，代表条件的 CSV 行与主评估完全一致。正文已有 E5 方法及测试矩阵，但未引用 E5 表和图。现有 `main.pdf` 为 2026-09-04 版本。

| 证据 | 写入论文的事实 | 限定 |
|---|---|---|
| 完整 18 组汇总 | 15 组融合 MPJPE 低于 canonical averaging | 同一冻结模型的条件比较 |
| Random, ratio 0.5 | Left/Right/Both 增益为 31.9% / 34.3% / 20.9% | 图像遮挡协议 |
| Distal / Temporal, 两个 ratio | 十二组均为正增益 | 不泛化到任意真实遮挡 |
| Random, ratio 1.0 | Left/Right/Both 增益为 -18.0% / -9.2% / -28.4% | 单侧严重遮挡也可能产生负增益 |
| Both / Random / ratio 1.0 | Fused MPJPE 0.6311，canonical average 0.4914 | 对应 SAM3D 检测失败率 36.36%，为伴随现象，不作因果归因 |
| 图像与姿态遮挡对照 | 两种输入干预产生不同误差行为 | 不据相同 ratio 排定两种干预的真实严重程度 |

## 文件与职责

下列路径除生成器外均相对于 `paper/ivc_draft_20260821/`。

| 文件 | 计划操作 |
|---|---|
| `main.tex` | 新增 E5 结果小节、引用图表，调整实验引导段、摘要、引言贡献、讨论和结论 |
| `tables/image_occlusion_summary.tex` | 由生成器重建，保留 18 行并补齐单位、增益和失败率说明 |
| `figures/extension/image_occlusion_robustness.pdf` | 由生成器重建，修正协议参数轴名并核对可读性 |
| `dual2pose/eval/render_e4_e5_artifacts.py`（仓库根目录） | 仅修改图表标签、表注及必要的展示样式 |
| `evidence/results/image_occlusion__*` | 新增最终汇总、图像/姿态对照和代表条件复跑的证据副本 |
| `evidence/image_occlusion_frontend_provenance.json` | 新增六条件来源、完整性、协议、模型与清单哈希的紧凑记录 |
| `evidence/evidence_manifest.md` | 记录新证据副本、来源路径、SHA-256、协议及复跑范围 |
| `evidence/figure_provenance.md` | 增加 E5 图表的输入、生成器及其支持的正文结论 |
| `README.md`、`revision_log.md` | 同步实验范围、再生成命令及这次更新记录 |
| `cover_letter.md`、`highlights.txt` | 与正文实际增加的图像遮挡证据保持一致 |
| `main.pdf` | 用更新后正文重新编译并逐页检查新增内容 |

## Task 1: 固定证据来源与报告口径

**输入：** 已完成的日志目录与最终汇总。
**产出：** 论文目录中的可追溯证据副本和清单。

- [x] 保存以下源文件的副本，按现有目录使用 `image_occlusion__` 前缀：`image_occlusion_summary_last.csv`、`image_occlusion_summary_last.json`、`image_vs_pose_occlusion_last.csv`。
- [x] 将 `reproducibility/both_random_r1p00_20260905T170432/image_occlusion_summary_last.csv` 保存为 `evidence/results/image_occlusion__both_random_r1p00_recheck.csv`。
- [x] 为每个副本计算 SHA-256，与源文件按字节核对后写入 `evidence_manifest.md`。
- [x] 从六个 `frontend_manifest.json` 和 `mask_protocol.json` 提取 condition、complete、expected/completed streams/frames、检测失败数、seed、命令、模型哈希及清单哈希，写入紧凑来源记录。保留原始清单路径；无需将姿态数组复制进论文目录。
- [x] 核对 18 个唯一条件、每行 64,440 个样本、同一冻结 checkpoint，以及主 CSV 和复跑 CSV 的对应行一致性。
- [x] 在来源说明中写明：表内 SAM3D 失败率按照评估相机配对和序列帧对齐后，在受干预视图上聚合。它不是六组独立源图像的简单失败比例，也不是融合 MPJPE 阈值失败率。

**完成条件：** 读者可以从表内数字追溯到结果副本和源日志；复跑证据被准确限定为一个代表条件。

## Task 2: 准备可直接引用的图表

**输入：** Task 1 的 E5 汇总和既有 E4 统计源文件。
**产出：** 一张 18 行结果表、一张三面板图像/姿态干预对照图。

- [x] 在生成器 `_render_e5_table` 的表注中补充：15 joints、数据集坐标单位、增益定义 `100 × (canonical average − fused) / canonical average`，及受干预视图的检测失败率口径。
- [x] 在 `_render_comparison_figure` 中将横轴 `Severity ratio` 改为 `Protocol ratio`。保留 Left/Right/Both 面板、全部三种模式、两个 ratio、图像/姿态两种干预；核对线型和标记是否能区分条件。
- [x] 为 `main.tex` 中新增图注写明：图比较相同协议标签下的 fused MPJPE；相同 ratio 不代表相同物理强度。图中没有 canonical-average 曲线，因此相对基线增益结论应引用结果表。
- [x] 从仓库根目录运行以下既有生成器命令；记录生成前后的 E4 表内容，确认 E4 统计数字保持一致。

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -m dual2pose.eval.render_e4_e5_artifacts \
  --e4-significance logs/ivc_mmsports_extension/view_angle/view_angle_significance_last.csv \
  --e4-statistics logs/ivc_mmsports_extension/view_angle/view_angle_statistics_last.json \
  --e5-summary logs/ivc_mmsports_extension/image_occlusion/image_occlusion_summary_last.csv \
  --comparison logs/ivc_mmsports_extension/image_occlusion/image_vs_pose_occlusion_last.csv \
  --table-root paper/ivc_draft_20260821/tables \
  --figure paper/ivc_draft_20260821/figures/extension/image_occlusion_robustness.pdf
```

- [x] 对照 CSV 核对表格 18 行的舍入值和三组负号，渲染图像检查图例、坐标、线条和裁切。展示文案改动通过实际生成和视觉核对验证，无需新增仅检查字符串的单元测试。

**完成条件：** 表格完整保留负结果；图表的标签和读法与实际干预协议一致。

## Task 3: 在 Results 增加 E5 结果

**输入：** Task 1 的结果、Task 2 的图表。
**产出：** 位于现有 `Missing-pose evidence` 之后、`Front-end generalization and adaptation` 之前的结果小节。

- [x] 更新 `Controlled capture perturbations` 的引导段，使其覆盖姿态干预以及经 SAM3D 传播的图像干预。
- [x] 新增 `\subsubsection{Image-level occlusion through the pose front end}`，正文目标约 250–350 个英文词，按下面三段安排。

| 段落 | 内容 | 引用证据 |
|---|---|---|
| 1：完整结果 | 总体 15/18 正增益；random 0.5 三组增益；distal 和 temporal 在所测条件均有优势 | `tab:image_occlusion` |
| 2：失败边界 | random 1.0 三组均退化，双侧 0.6311 对 0.4914；报告伴随的检测失败，不断言其单独造成全部退化 | `tab:image_occlusion` 与原始汇总 |
| 3：干预层次 | 图像遮挡先影响 SAM3D，姿态遮挡直接改输入；说明两者提供互补压力测试，不将 ratio 当成等强度尺度 | `fig:image_occlusion_extension` |

- [x] 插入以下表图引用与图注。

```latex
\input{tables/image_occlusion_summary.tex}
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/extension/image_occlusion_robustness.pdf}
\caption{Fused MPJPE under image-level occlusion propagated through SAM3D and direct pose-stream corruption, for left-, right-, and both-view interventions. Ratios are protocol parameters; matching values do not imply equal physical occlusion severity. Errors are reported in dataset coordinate units.}
\label{fig:image_occlusion_extension}
\end{figure}
```

- [x] 图放在该小节内，并保留结果组末尾的 `\FloatBarrier`。不在新小节中重复现有完整遮挡生成方法。
- [x] 检查新增内容明确保留严重单侧随机图像遮挡的失败结果，避免沿用仅讨论“双侧失败”的概括。

**完成条件：** 完整 18 组结果、失败条件及其解释均能从 Results 中找到，表图引用实际生效。

## Task 4: 同步论文主张与投稿材料

**输入：** Task 3 的最终结果表述。
**产出：** 摘要、引言、讨论、结论及投稿材料的一致表述。

- [x] 摘要：替换现有过于宽泛的随机腐败边界句，用一句话同时表达多数图像遮挡条件下的正增益和最严重随机条件下的退化；通过替换控制长度，保留两套数据的主要准确率结果。
- [x] 引言贡献：在已有鲁棒性贡献中加入“RGB occlusion propagated through SAM3D”，不提出新模型架构贡献。
- [x] Discussion：新增或合并约 80–120 个英文词，解释图像干预使实验覆盖前端检测失败，但仍是 Unity 上人工遮挡、单一冻结 SAM3D 前端。干净互补视图存在也不能保证严重受损条件下一定获益。
- [x] 区分两处现有限制：对 VideoPose3D/PoseFormer/MotionBERT 的跨前端比较仍基于 Unity 真值 2D，完整图像检测比较仍待验证；对原生 SAM3D，人工图像遮挡已完成，未来工作应收窄为真实遮挡、更广泛图像退化和外部数据。
- [x] Conclusion：增加一个简洁的图像遮挡结果句；将姿态遮挡和图像遮挡的失败条件分别限定，避免泛化为“只有双侧损坏才失败”。
- [x] 更新 cover letter 的角度统计描述和图像遮挡新增项，并同步 README/revision log 的实验列表。保留已接受会议版的现有披露与作者确认事项。
- [x] 将 highlights 的已有鲁棒性条目明确到图像遮挡，维持原有五条结构；确认不出现正文没有支持的新主张。

**完成条件：** 各部分对已完成实验、负结果及剩余限制的描述一致。

## Task 5: 编译、检查实际 PDF 与交付

**输入：** 完成更新的论文与证据文件。
**产出：** 最新 `main.pdf` 和可追溯的简短修改记录。

- [x] 在论文目录执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

- [x] 检查构建日志中的未定义引用、缺失文件和新增 overfull box；只处理本次引入的问题。确认编译时间晚于本次正文和图表更新。
- [x] 使用 `pdftotext -layout main.pdf /tmp/ivc-e5-integration-main.txt`，搜索新增小节、图表标题、`0.6311`、`0.4914` 及负增益；核对图表编号和正文引用。
- [x] 使用 `pdftoppm` 渲染新增小节、结果表、对照图和讨论所在页，并打开渲染图检查表格宽度、可读性、浮动顺序、分页和图注。以实际 PDF 页码记录新增内容的位置。
- [x] 复核证据副本与清单中的 SHA-256 一致；检查更新后的 Results、Discussion、Conclusion 都保留三组负结果及干预口径。
- [x] 查看最终 diff，确认原始实验输出、模型和无关文件未被改动；在 `revision_log.md` 记录整合日期、内容、编译检查与代表条件复跑的证据范围。

**交付标准：** E5 全部结果进入正文和 PDF；图表、叙述、投稿材料及证据快照一致；模型局限表达准确。计划执行完成不等于已开展新的统计显著性检验或全面投稿验收。

## 执行顺序与工作量

按 Task 1 → 2 → 3 → 4 → 5 顺序执行。核心产物是一个新增结果小节、一张完整结果表、一张对照图及配套的局部论述更新。现有结果足够支持此次整合，所需验证以数据核对、CPU 出图、LaTeX 编译及 PDF 检查为主。

## 执行完成记录（2026-09-06）

本计划已按用户后续授权执行。最终 PDF 共 47 页；E5 结果在第 29 页，完整表在第 30 页，对照图在第 31 页，讨论与结论在第 42–43 页。18 行数值、四份来源哈希、单条件复跑一致性及两项现有图表测试均已核对。已检查最终 PDF 的实际渲染页，无未定义引用或 overfull box。验证记录见 `paper/ivc_draft_20260821/evidence/image_occlusion_integration_verification.json`。
