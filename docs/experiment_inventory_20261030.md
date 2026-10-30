# 当前完整实验清单（2026-10-30，北京时间）

本清单统一 IEEE Access 主文、补充材料、逐帧重跑主队列、补跑与 MetaPose 恢复记录。论文组织保持不变；本文不修改论文、不启动重跑、不覆盖结果。旧计划中的“36 次训练”不再代表最终范围。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate（实验台账及产物核对，不是本次重新训练）
- Verification Status: ANALYZED；读取实际状态、最终覆盖记录，复核最终记录引用的 237 个产物 SHA256，缺失/不一致均为 0。
- Version Label: experiment_inventory_20261030
- 当前代码基线：`701e052d040450556c35c4aff708ab3492d33676`，分支 `per-frame-canonicalization`。历史实验使用各自冻结源码；当前 HEAD 不等于所有实验的执行源码。
- 权威总表：`logs/per_frame_common13_v1/metapose_unity_recovery_20261029T122908Z/final/main_comparison_mm.csv`。

## 统一协议

- 主协议 `common13_direct_per_frame_v1`：输入与 GT 按 frame 建立身体坐标；规范空间模型输出直接与规范 GT 评分，不再按预测或 GT 拟合主指标。
- MetaPose 等原生坐标输出使用预先声明的输入侧坐标转换；不把所有方法强制改成同一训练目标。
- 主指标 common-13 MPJPE，长度误差报告 mm；PA-MPJPE、Unity all-15 为明确标注的辅助结果。加速度为 mm/采样帧²，不等同世界轨迹加速度。
- Unity 使用独立 val 选模；Ski 保持论文 train/test 划分和预声明固定最终轮次；Pocket3 正式域内训练用块内 val 选模，20 轮补充适配按固定预算。外部方法保留各自既定选模契约。
- Unity 主表完整测试 64,440 窗；Ski 30 序列/窗口。Pocket3 当前方向一 train/val/test=42/3/5，方向二=40/3/2；不能继承旧稿 37 全窗/18 测试窗/540 帧。
- 除 Pocket3 明确的 3 seeds 外，不宣称主比较是多种子或跨折结论；MetaPose 标为 adapted，不能称完全复现官方实验协议。

## A. 正式训练与适配：38 套配置，均有完成结果

| 编号 | 内容 | 规模 | 预算或选模要点 | 结果根目录简称 |
|---|---|---:|---|---|
| T01 | CanonFuse3D 主模型 | Unity、Ski 各 1，共 2 | 各 100 epochs；Unity val-best，Ski 固定最后一轮 | MAIN |
| T02 | TCN、SmoothNet | 2 方法 × 2 数据集 = 4 | 各 100 epochs；按数据集选模契约 | CORE |
| T03 | DLT-residual | 2 数据集 = 2 | 几何辅助方法，输入条件单独说明 | AUTO/main_baselines |
| T04 | DeciWatch adapted | 2 数据集 = 2 | 各 70 epochs、seed4321；4 帧恢复 30 帧 | CORE/deciwatch |
| T05 | 门控/残差消融 | full、fixed_gate、no_output_residual = 3 | Unity 各 100 epochs；full 不与 T01 混作同一次训练 | AUTO/gates |
| T06 | 前端适配 | 3 前端 × heads-only/full + mixed-full = 7 | 20 epochs；VideoPose3D、PoseFormer、MotionBERT | AUTO/frontend/training，详见任务 CSV |
| T07 | Pocket3 正式域内训练 | 2 方向 × Unity 初始化/随机初始化 × 3 seeds = 12 | 各 300 epochs 预算；块内 val 选模 | AUTO/pocket |
| T08 | Pocket3 补充短适配 | 2 方向 × heads-only/full = 4 | 固定 20 epochs | AUTO/pocket |
| T09 | MetaPose adapted | Unity、Ski 各 1，共 2 | Unity 3×10，val 2D MSE 选各阶段；Ski 稳定版 3×300，训练期退化门禁、固定最终轮次 | SUPP/metapose/unity；STABLE；REC |
| 合计 | 仅计当前正式配置 | **38** | 36 原计划 + 2 MetaPose；不计失败尝试、pilot、环境检查 | |

另外：**STRIDE 两数据集各一套逐窗口测试时优化（TTA）**，每窗 30 次更新，已完成。它不是两次新的离线训练。DLT、reprojection-gated DLT、单视角和平均/质量加权方法不需要训练，但必须有新协议评测。

MetaPose Unity 的训练在原补跑中完成，之后因 GPU 故障中断推理；REC 复用该轮已选权重完成推理和评测。Ski 使用独立稳定重训版，旧退化版本只作历史记录，不进入最终表。

## B. 全部评价与论文证据

“完成”表示当前轮对应结果已产出并有验证记录，不表示旧论文数字已经更新。各行包含共享控制项，不应相加为独立任务数。

| 编号 | 实验与覆盖 | 论文位置（按主题/标签，避免表号漂移） | 当前权威来源 | 状态 |
|---|---|---|---|---|
| E01 | 14 方法 × Unity/Ski = 28 行，MPJPE/PA/Accel | 主比较 `tab:native_reference` | REC/final/main_comparison_mm.csv | 完成 |
| E02 | Unity native-15 辅助比较 | 补充 all-15 | AUTO/main_baselines_pa/results/unity | 完成；仅适用具备 15 点输出的方法 |
| E03 | 三种门控/残差结构重训对照 | `tab:gate_retraining` | AUTO/gates | 完成 |
| E04 | 五种冻结门控干预；关节、区域、误差关联、固定窗口图；Unity + Ski | `tab:gate_region_summary`、`fig:gate_diagnostics_full`、补充门控 | AUTO/evaluations/eval_gate_interventions；SUPP/ski_gate、gate_figures；遮挡 alpha 用 SUPP 版本 | 完成 |
| E05 | 6 摄像机夹角分箱、配对 bootstrap/统计 | `tab:view_angle` 及补充 | AUTO/evaluations/eval_angles、eval_angle_significance | 完成 |
| E06 | 7 时间偏移：−2、−1、−0.5、0、0.5、1、2 | `fig:temporal_sampling` 及补充 | SUPP/evaluations/eval_offset_* | 完成；优先补跑输出 |
| E07 | 7 采样率误差：−2%、−1%、−0.5%、0、0.5%、1%、2% | 同上 | AUTO/evaluations/eval_sampling | 完成 |
| E08 | 姿态缺失：3 视角 × 3 模式 × 5 比例 = 45 条件 | `fig:masking_extension` 及补充 | SUPP/evaluations/eval_mask_* | 完成；含新 alpha 导出 |
| E09 | RGB 遮挡：3 视角 × 3 模式 × 2 比例 = 18 条件 | `fig:image_occlusion_extension` 及补充 | AUTO/evaluations/eval_rgb_* | 完成；复用已核验 raw 前端预测，重做融合评分 |
| E10 | 匹配遮挡与标定 C07：7 条件，3 几何方法×3角度及2姿态方法；77实际计算格/105展示格 | `tab:c07_occlusion_mpjpe`、`fig:c07_calibration` 及补充 coverage/degeneracy | AUTO/evaluations/eval_c07 | 完成 |
| E11 | 前端泛化：冻结参考+7适配模型，共8×4前端=32格 | `tab:frontend_adaptation` 及补充 | AUTO/evaluations/eval_frontend_* | 完成 |
| E12 | N=1/2/3/4，10组合、44匹配组；bootstrap及2/3/4视角成本 | `tab:nview_efficiency`、`fig:nview_scaling` 及补充 | AUTO/evaluations/eval_nview | 完成 |
| E13 | 自动同步：val阈值标定；11偏移×uncorrected/automatic/oracle=33格 | `fig:temporal_alignment` 及补充 | AUTO/evaluations/sync | 完成 |
| E14 | Pocket3：Unity/Ski冻结；12正式训练；4短适配；gate-only；共同mask/分块 | `tab:pocket3` 及补充 | AUTO/pocket | 完成；必须采用本轮新cohort |
| E15 | Pocket3 reference时移：−4至4，9设置×8方法；固定有效点交集 | Pocket3补充 reference sensitivity | SUPP/pocket_shift | 完成；17基准窗、5926共同关节帧点 |
| E16 | 户外/室内ski及Pocket3定性；逐帧表示、mm坐标刻度、等比例坐标轴 | `fig:real_world_comparison` | SUPP/realworld；外部方法例图 AUTO/external/qualitative | 完成；尚未替换论文旧图 |
| E17 | 本地融合与外部方法效率；MetaPose更新后效率 | 补充 local/external efficiency | AUTO/external/efficiency；REC/efficiency.json | 完成；不同输入/任务预算不能作统一公平速度排名 |

E01 的 14 种方法：left/right canonical、canonical average、aligned average、3D quality weighting、TCN、SmoothNet、CanonFuse3D、DLT、reprojection-gated DLT、DLT-residual、DeciWatch adapted、STRIDE adapted、MetaPose adapted。表中 `dlt_mlp` 对应 DLT-residual，不是已退出论文范围的普通 cross-view MLP。

## C. 状态计数与结果优先级

- 主自动队列 AUTO：**179/179 complete**，包含准备、训练、评测、汇总和效率任务，不能称为179次训练。
- 原补跑 SUPP：62任务；58 complete、1 failed、3 blocked 是保留的历史状态。不是当前尚缺4项。
- 恢复队列 REC：**7/7 complete，退出码均0**；最终 coverage 校验 **61项结果任务 = 56项既有补跑结果 + 5项MetaPose训练/评分/效率项**，替代原final_coverage等6项。
- 上述队列互有重复与替代，不能把179、62、61、7直接相加称独立实验总数。任务历史 CSV 共248行（179+62+7），保留原状态及当前有效来源。
- 取数顺序：**REC 最终表/覆盖记录 → Ski STABLE及REC独立重评分 → SUPP补跑产物 → AUTO产物**。原始Ski退化MetaPose及旧原生预测重评分结果，不替代新正式训练结果。
- 本次重新核对 REC coverage 引用的 **237个产物内容哈希**：缺失0、变化0。主队列179项的完整验证以原审计记录为依据，本次没有重新执行全量训练/评分验证器。
- 已有正式实验的已知缺项：**0（按上述已批准范围和完成记录）**。这是实验台账结论，不等同于对全部代码或论文文字作无条件正确性保证。

## D. 仍待完成的工作与依赖

| 工作 | 当前状态 | 依赖/处理 |
|---|---|---|
| 将新结果映射到Access全部表图、补充材料、摘要/讨论 | 尚未合入；本次不改论文 | 使用B清单及当前权威来源，之后按用户授权更新 |
| 修改论文旧first-frame描述、Pocket3旧样本数、MetaPose配方及旧结果论断 | 待论文更新时处理 | 不仅换数字；要同步协议、覆盖量、训练选择与限制 |
| 在Pegasus再次完整跑38套配置及全部评测 | 未启动，不是补齐缺失结果的必需步骤 | 若要跨机器完整复跑，先生成迁移后的新manifest/输出目录，保持原实验冻结源码和依赖版本记录 |
| Pegasus主流程运行验证 | 已完成独立短测试 | HP260146/gen_S；Unity/Ski/Pocket3训练评测、MetaPose训练及加载；不计入38套正式结果 |
| Pegasus外部baseline与前端缓存验证 | 三种外部方法短检查及前端路径检查通过 | 详见 `pegasus_full_chain_20261030.md`；不等同全预算训练或RGB前端重新推理 |
| 全链路在Pegasus的重跑命令与依赖调度 | 已配置220项新DAG，支持多节点独立提交；未启动 | 新输出、冻结源码、依赖与结果验证；目前提供ready/submit-one，不含自动调度守护进程 |

若再次全量复跑，依赖顺序：数据/缓存与版本核验 → Unity/Ski主模型及独立baseline并行 → 门控、几何残差、前端适配、Pocket3、MetaPose/STRIDE → 冻结鲁棒性/同步/N-view → 独占GPU效率测量 → 覆盖/哈希/有限值验收 → 生成新汇总。负结果同样保留。

本次未新增多种子Unity/Ski主比较、Ski六折、普通cross-view MLP或新的MetaPose配方；这些均属于新增研究范围。

## E. 路径与证据索引

以下简称均相对项目仓库：

| 简称 | 路径 |
|---|---|
| MAIN | `logs/per_frame_common13_v1/main_20261028T053836Z` |
| CORE | `logs/per_frame_common13_v1/core_20261028T055152Z` |
| AUTO | `logs/per_frame_common13_v1/auto_20261028T061053Z` |
| SUPP | `logs/per_frame_common13_v1/completion_20261029T042127Z` |
| STABLE | `logs/per_frame_common13_v1/metapose_stable_20261029T052359Z` |
| REC | `logs/per_frame_common13_v1/metapose_unity_recovery_20261029T122908Z` |

- 逐任务状态、依赖、validation规格、来源：`docs/experiment_inventory_tasks_20261030.csv`。
- 当前台账核验：`logs/experiment_inventory_20261030/verification.json`。
- 主队列验收：`AUTO/final_audit_20261029T041144Z/audit_summary.json`、`job_validation.json`。
- 补跑最终验收：`REC/final/coverage.json`、`verification.json`。
- Pegasus运行说明：`docs/pegasus_runtime_20261030.md`。

本地仓库根为 `/home/kaixu_chen/skiing/Skiing_Canonical_DualView_3D_Pose_PyTorch/`；Pegasus仓库根为 `/work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch/`，相对结构一致。归档JSON/CSV中的旧绝对来源保留溯源用途；引用时可按仓库根定位，执行时另建新manifest。
