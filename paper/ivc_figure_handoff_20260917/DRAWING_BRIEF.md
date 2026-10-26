---
title: IVC 流程图绘制任务书
subtitle: CanonFuse3D · 图一总览 / 图二机制 · 2026-09-17
lang: zh-CN
---

# 0. 三数据集扩展使用说明

本素材包已扩展为 Unity、Ski-PTZ-Pose、自采双视角视频各 4 组，共 12 组。先打开 `ASSET_GALLERY.html` 选样；每组素材位于 `multi_samples/数据集/编号/`，样本表见 `MULTI_SAMPLE_INDEX.md`（另有 Word/PDF）。原单样本素材继续保留。

以下第 1–7 节的流程、布局和公式仍适用；其中固定 Unity sample 0 / frame 10 的数值和图注仅适用于原版。换用新增样本时请同步修改数据集名、帧号和参考类型，不得沿用原样本指标。

- Unity：4 个动作，预先指定 30/60/120/180° 视角间隔，15 关节，有 avatar GT。
- Ski-PTZ-Pose：seq 405 和 seq 412 各两组视角，共 4 组；属于同一受试者，不代表四个独立被试。使用 13 个共同关节，参考标为 `Constructed reference`，不标独立实测 GT。
- 自采：pro_1、pro_2、run_3、run_4，分别两段室外与两段室内，15 关节，无 GT，输出仅作定性展示。固定使用 Unity 检查点；没有挑选模型或帧以追求更好结果。
- 新增样本包含真实注意力热图（当前帧、各头平均）。原版 `attention_connections` 仍只是连线示意。热图不要混用；权重色标固定 0–1。
- 新增数据均用现有固定模型在 CPU 上重新推理。Unity/Ski 保留原测试批次人体参考；自采采用单窗口首帧参考。所有导出都记录检查点和源文件。

对于图一，仍优先选一组样本贯穿输入和输出。若要并列三数据集，以三行展示并分别标出 `Unity GT`、`Constructed reference`、`No GT`。图二机制小图尽量来自同一组选定样本，不要跨数据集拼接成一条貌似连续的推理流程。

# 1. 交付目标

绘制两张英文论文插图。图一让读者立即看懂“输入是什么、如何处理、输出是什么”；图二解释“五路信息怎样构造、怎样决定融合结果”。不需要把每个模块都画满。随包的图像是可拆分素材，最终版式由绘图者完成。

统一用灰蓝表示左视角、灰绿表示右视角，暖橙表示融合输出，深灰虚线表示 GT。视角 L/R 表示数据流，不表示人体左侧/右侧。两张图沿用同一配色。

| 对象 | 建议颜色 | 辅助区分 |
|---|---|---|
| View L / 左候选 | `#587C9E` 灰蓝 | 实线、L 标签 |
| View R / 右候选 | `#5C9488` 灰绿 | 虚线或 R 标签 |
| Fused output | `#D18B47` 暖橙 | 较粗实线 |
| Unity GT | `#575D66` 深灰 | 细虚线 |
| Difference / correction arrows | `#A06BA5` 淡紫 | 带箭头的细线 |

白底、简洁边框、少量浅色底块即可。建议按约 180 mm 宽设计；这是排版建议，不是已核对的期刊尺寸规定。缩到论文最终尺寸后，主标签建议不小于 8 pt。交付可编辑源文件、矢量 PDF/SVG 和 300 dpi PNG；另附保留可编辑文字的版本。

# 2. 图一：五步系统总览

**英文图名：System overview of CanonFuse3D**

沿用现有 `reference/study_overview.pdf` 的五步信息层级，从左到右排列。推荐横向五列，输入与输出略宽；不要在这张图展开五个特征公式。

| 顺序 | 框内英文主标签 | 要画的内容 | 可用素材 |
|---|---|---|---|
| A | Two monocular 3D pose streams | 两个 3D 输入，标 L、R；旁边或上方加同帧 RGB 小窗 | `raw_left`、`raw_right`、两张 `rgb_*` |
| B | Human-centric coordinates + pose alignment | 朝向/位置差异经坐标表示和对齐后更可比较；仍保留局部差异 | `alignment_before_after`，可选 `body_reference` |
| C | Cross-view + temporal features | 简洁的双流特征条和时间符号，图二再展开 | 图二五路模块的缩略图，不重复细节 |
| D | Joint-frame gate + residual correction | 一个小门控符号和加号即可 | 可缩略使用 `gate_heatmap`，无需完整坐标标签 |
| E | Refined canonical 3D pose sequence | 一组输出骨架，旁边选加同帧 GT 对照 | `fused`、`output_comparison` |

主箭头：A → B → C → D → E。

RGB 属于上游单目估计器的输入。推荐在 A 左上方用小括号区画 `Synchronized views → Monocular 3D estimation`，箭头只连接到两条 3D 输入；CanonFuse3D 的边界从 3D pose streams 开始。若空间不足，直接并排放 RGB 与 3D 骨架，标 `Upstream illustration`。不要把 RGB 箭头接到特征融合模块。

图下保留一句：**Approximately synchronized 3D pose inputs; no camera parameters at the fusion stage.**

输出局部放大是可选项。如果放大手腕或脚踝，左右对照必须用相同投影、范围和比例，三维误差见 `data/local_errors.csv`。不得为突出改善而移动关节点。当前输出对照有真实 Unity GT；不要把 GT 延伸到真实户外视频，也不要称为物理世界轨迹。

# 3. 图二：特征构造与融合机制

**英文图名：Feature construction and reliability-gated fusion**

建议三行：第一行为输入和两种坐标处理；第二行为五路特征；第三行为编码、跨视角注意力、时间细化和门控/残差输出。保留原图的八步逻辑，但重点把五路特征和末端融合画清楚。

## 第一行：步骤 1–3

`1 Input pose streams → 2 Human-centric canonicalization → 3 Sequence-level alignment`

- 输入：Pᴸ、Pᴿ，每条数据形状 `T × J × 3`。
- 人体坐标：用髋部中点、髋轴和颈部方向定义参考；用 `body_reference`。这是首帧参考示例，不是每一帧都把骨盆强制放回原点。
- 序列对齐：左右两条方向都存在，`R → L` 和 `L → R`。每个方向对整段序列拟合一次相似变换；用 `alignment_before_after` 展示 R → L，旁注 `Both directions` 即可。

精度边界：不要画出保证正交的三轴角标；当前人体轴不保证两两正交。不要写 `scale normalization`，人体坐标阶段不做骨长归一化。后续序列对齐才允许旋转、平移和统一尺度变化。对齐是姿态空间的辅助变换，不是标定相机外参；对齐后两组骨架也不应完全重合。

## 第二行：步骤 4，五路特征

只完整展开左流，标 `Left stream shown; swap L and R for the right stream`。五个并列小格，每格一个小图、一条短英文名和 `3 ch`；汇聚到 `Concatenate → 15 ch per joint and frame`。这五路是特征组成，不是五个各自训练的神经网络。

| 路 | 英文短标签 | 内容与符号 | 怎么画 | 素材 |
|---|---|---|---|---|
| 1 | Own-view pose | canonical Pᴸ | 一组灰蓝骨架 | `left` |
| 2 | Aligned other view | Pᴿ→ᴸ | 一组灰绿骨架；强调已变换到左流姿态空间 | `right_to_left` |
| 3 | Cross-view difference | Dᴸ = Pᴿ→ᴸ − canonical Pᴸ | 两骨架少量对应关节之间的差值箭头 | `cue_difference` |
| 4 | Temporal velocity | Vᴸₜ = canonical Pᴸₜ − canonical Pᴸₜ₋₁ | 前后两个姿态、手腕/脚踝运动箭头 | `cue_velocity` |
| 5 | Alignment rotation | rᴿ→ᴸ | 两组基方向或旋转箭头，加 `broadcast` | `cue_rotation` |

每路均为 3 维，所以 `3 + 3 + 3 + 3 + 3 = 15`。符号的标准 LaTeX 版本见 `captions_and_formulas.tex`。

速度是重采样窗口内一阶差分，首位置置零，未除以时间间隔；不能标 m/s 或当成原始视频匀速采样时间。旋转特征是对齐旋转的三维 rotation vector，不是 3 × 3 矩阵输入、角速度或相机姿态；在同一序列内广播到所有帧与关节。

## 第三行：步骤 5–8

| 顺序 | 英文框名 | 内部内容 | 绘图重点 |
|---|---|---|---|
| 5 | Shared joint encoder | `15 → 128`；MLP、LayerNorm、GELU | 左右各一条流，编码器共享参数 |
| 6 | Bidirectional cross-view attention | L attends to R；R attends to L | 同一帧内跨视角关节交互；两个方向的注意力参数不共享 |
| 7 | Temporal refinement | `Depthwise temporal conv, k = 5` | 时间轴上的局部特征细化；保留残差连接 |
| 8 | Gate + residual correction | Z 分别进入 gate head 与 residual head | 热图、候选骨架混合、输出修正三部分的关系要明确 |

注意力可用 `attention_connections` 作为示意：少量跨骨架连线，允许对应关节和相关关节。这里没有导出实测注意力权重，所有连线应保持统一样式；如使用不同深浅，只能明确写 `Illustrative`，不能写成模型测得的关注程度。注意力不是只连同名关节，也不是跨全部 T 帧的注意力。

时间小图可用 `temporal_poses` 的三个姿态说明连续运动，但卷积核是 **5**；最好在框内另加五格特征窗口，标 `t−2, t−1, t, t+1, t+2`。不能因为画了三个人就把模块标为 k=3，也不能声称严格因果在线处理。

# 4. 末端融合：必须画对的箭头和公式

左右时间特征先拼接 `Z = [H̄ᴸ, H̄ᴿ]`，再分到两个预测头：

- `Gate head + Sigmoid → α`：每帧、每关节一个标量。
- `Residual head → ΔP`：每帧、每关节一个三维修正。

另画一条来自坐标处理结果的骨架旁路，构成两组基础候选：

`Bᴸ = ½(canonical Pᴸ + Pᴿ→ᴸ)`

`Bᴿ = ½(canonical Pᴿ + Pᴸ→ᴿ)`

之后是：

`Pbase = α Bᴸ + (1−α) Bᴿ`

`Pᶠ = Pbase + ΔP`

可以用两个乘法圆点、一处求和与最后一个加号表达。Bᴸ、Bᴿ 与 α 必须有明确箭头进入加权节点；ΔP 接到最后一个加号。不能画成“热图经过残差网络变成骨架”。

门控权重作用于 **Bᴸ 与 Bᴿ**，它们分别已经包含原流及对齐后的对侧流；不能改成 `α Pᴸ + (1−α) Pᴿ`。两者具有各自人体参考，网络学习的是姿态空间映射；不能画一个世界坐标大框声称所有流共享真实物理坐标。

热图使用 `gate_heatmap` 或 `data/gate_alpha.csv`。横轴是窗口位置 1–30，纵轴是 15 个关节；色标 0–1，写 `Weight of Bᴸ (α)`。α=1 表示仅取 Bᴸ，α=0 表示仅取 Bᴿ，不代表仅用一台相机的原始姿态。它是学习到的混合权重，不是经校准的置信概率。

差值 D 与输出修正 ΔP 是不同对象：D 在五路特征中，来自两个姿态的差；ΔP 是网络预测的输出修正。对应素材分别为 `cue_difference` 和 `residual_correction`。

# 5. 素材真实性及取舍

主素材固定为同一个 Unity 样本，窗口位置 15 / 原始帧 10。不是按改善幅度挑出来的最佳帧。RGB 原图、输入、候选与输出可以作为同一样本的对应示例使用，但骨架插图是便于观察的斜投影，不能直接叠到 RGB 人物上。

| 类别 | 本包状态 | 允许表达 |
|---|---|---|
| 同帧双视角 RGB | 两张未修改原图 | 同一动作的两个观察方向；本例相差 10° |
| 输入、对齐、特征、输出骨架 | 真实缓存与归档组件复原 | 方法输入输出与中间表示 |
| GT | Unity avatar GT，经原协议独立人体参考变换 | 该 Unity 样本的 canonical-space 对照 |
| 门控热图 | 真值导出，范围约 0.459–0.597 | 实际权重在帧/关节维度的分布 |
| 注意力连线 | 机制示意 | 网络可在对应及相关关节之间交换信息 |
| 图中效果改善 | 单帧例子 | 可以展示此帧对照；不能推及所有样本或总体百分比 |

该帧全 15 关节平均误差：canonical average 0.09211，CanonFuse3D 0.04920，单位为 canonical coordinate units。不要未经作者核对把它改写成 mm、原始世界坐标精度或论文总体结果。局部关节并非保证全部改善；放大前查 `local_errors.csv`，若选某个改善的部位，应将其标为局部示例。

这里没有重新运行网络推理：输出按归档的真实 α、Bᴸ、Bᴿ 和 ΔP 复原。原始输入的 canonicalization 依原评估首批 256 个样本计算，随后重新计算 Sim3；两类候选与归档对照的最大差异分别是 3.58e-7 和 4.77e-7。

不要通过拉大误差箭头、移动关节或把热图色阶压缩到本例最小/最大值来制造更强改善。若最终图追求更明显的旋转或大视角差异，可额外画明确标注 `Schematic` 的图标；不得替换本包实测数据而保留“真实示例”的标注。

# 6. 可直接用的英文图注

**Figure 1.** System overview of CanonFuse3D. Two approximately synchronized monocular 3D pose streams are expressed in human-centric coordinates and aligned in pose space. Cross-view and temporal features drive joint-frame gating with residual correction to produce a refined canonical pose sequence. RGB thumbnails illustrate the upstream observations; the fusion stage uses only 3D pose streams and requires no camera parameters. Pose illustrations and the output comparison show one fixed Unity sample, with canonicalized avatar ground truth used only for the comparison.

**Figure 2.** Feature construction and reliability-gated fusion. For each joint and frame, the own-view canonical pose, aligned opposite-view pose, cross-view difference, temporal first difference, and sequence-level rotation vector are concatenated into a 15-dimensional feature. A shared joint encoder, bidirectional cross-view attention, and temporal refinement produce features for joint-frame gating and residual correction. The final output combines the two base candidates using α and 1−α and adds the predicted correction ΔP. The gate heatmap is measured on a fixed Unity sample; attention connections are schematic. The right stream uses the same feature construction with the view roles reversed.

如果最终未放 RGB 或 GT 对照，删去图一图注对应句；如果未放注意力连线，也删去图二对应示意说明。不能保留不存在面板的说明。

# 7. 最终交付检查

1. 图一仍能一眼读出五步；图二五路明确汇成 15 维。
2. RGB 只连到上游估计器；GT 只用于对照，不作为推理输入。
3. 两个 Sim3 方向、左右特征流、候选骨架旁路均无歧义。
4. 注意力在每帧关节维度进行；时间卷积 k=5。
5. 门控权重对应 Bᴸ/Bᴿ，输出残差接在混合骨架之后。
6. 没有错误的相机标定、世界轨迹、骨长归一化或概率置信度标注。
7. 图像/骨架样本与帧一致；叠加的 GT 和预测使用同一显示投影与范围。
8. SVG/PDF 字体正常、无截断，缩到论文尺寸后可读，色彩以外仍能区分左右流。
