# IVC 批注修订执行记录（2026-09-16）

## 当前交付（2026-09-17，C07 全量及合稿完成）

C07 全量前端和评估均正常结束（returncode=0）。三套前端各 720 流、17,089 图像；77 个独立评估单元，每单元 64,440 窗口。所有条件和失败样本保留，不重训、不修改原 benchmark。

新增主文 5.7 节及表 8/9（pp.30–32）；补充 S14 及表 S11–S15（pp.19–25）。同步修订关节协议、讨论与结论。两份 PDF 编译通过，主稿 46 页、补充 25 页；新增页目视检查，完整矩阵表增大字号。455 个新增数值、2160 个前端文件哈希、源代码/标定文件哈希、既有 123 数值及 36 外部数值检查通过。原 canonical/native 表字节不变。

本轮 C01–C08 的计划内修订已交付；作者声明仍需投稿前确认。无 commit/push，paper/ 和 logs/ 继续不追踪。证据与说明见 evidence/c07_matched_occlusion_20260917/README.md。

## 以下为历史执行快照（不能作为当前待办）


## 当前状态（C07 实现及全量启动更新）

已完成的门控消融、冻结诊断、外部效率审计已整合进主稿和补充材料。原 canonical/native benchmark 两表字节不变。主稿 43 页，补充材料 19 页。C07 已完成代码接入、46 项回归和真实验证集 smoke，并启动 GPU 0 全量前端→评估流水线（runner PID 2268603）。全量结果尚未完成，尚未写入正文；不得宣称全部 C01–C08 实验完成。

- 主稿表 5（第 23 页）：四格门控/输出残差消融，其中三格同协议重新训练，第四格为确定性对照。
- 主稿表 6（第 24 页）、图 4（第 25 页）：修正后的全量门控统计和分布；旧 0.497 改为实测 0.6804。
- 补充表 S9、图 S3（第 17 页）：Unity/Ski 各五条件完整冻结干预及前两个固定 Unity 示例。
- 补充表 S10、S13 节（第 18–19 页）：三种外部方法参数、权重字节、实测延迟及逐方法边界；表中均有论文引用。
- 新证据在 evidence/review_revision_20260916/；核验脚本通过 123 个新增数值、完整覆盖及原两表哈希检查。
- 最终指定恢复的 DeciWatch 源码路径后，64 项联合回归全部通过、无跳过；MetaPose 独立环境集成测试 1 项通过。TF 有插件注册、弃用和 tracing 警告，无测试失败；pip check 通过。
- 两份 PDF 已编译并目视检查新增页；没有未定义引用、Overfull hbox 或 LaTeX error。旧 BibTeX ski_poseptz_dataset 缺 year 警告未在本轮修复。

### 本轮实验进度

本记录是分阶段执行结果，不表示全部修订计划完成。用户已批准在当前 IVC 分支和目录执行。未 commit/push；paper/、logs/ 仍被忽略，两个原有 .orig 文件保留。

| 项目 | 实际进展 | 剩余工作 |
|---|---|---|
| C01–C05 | 摘要/引言主线、单目分类、双目文献、任务边界已修订；主稿编译通过 | 合稿后最后通读 |
| 关联文字问题 | 内部/外部训练协议、canonical/native 指标边界、失败条件、bootstrap 与训练波动区别已修订 | 保持随新结果同步 |
| C06 冻结诊断 | 全量诊断、固定索引图、区域统计纠正和完整干预表均已入稿 | 已披露归一化/对齐耦合，不声称纯门控可靠性证明 |
| C06 重新训练 | 三组各 100 epochs 与完整选模/测试完成，四格结果已入主稿表 5 | 仍限单 seed、固定 Unity 划分 |
| C08 效率 | 局部表 S8 + 外部表 S10 已完成；正文有实测摘要和引用 | 未声称外部完整 FLOPs 或 RGB 端到端速度 |
| C07 受损几何比较 | 同源 2D+3D、五方法、七条件、0°/1°/3° 已接通；46 测试与真实 smoke 通过；全量已启动 | 等待完整前端与 77 独立单元覆盖核验，再整理入稿 |

## C07 运行记录

- 协议：docs/ivc_c07_matched_occlusion_20260916.md；全量输出 logs/ivc_review_revision_20260916/occluded_geometry_full/。
- 最终验证：occluded_geometry_smoke_v2/，固定首个 val pair、三个输入缓存，共 180 张条件化图像；7 条件、105 展示单元（77 独立）均通过完整性检查。不是论文准确率结果。
- 真实前端验证成功退出；辅助权重改为显式绑定原本已缓存的相同 ViTDet/MoGe 文件，没有切换模型。完整 stderr/console 已保留，包含原模型加载及第三方弃用提示。
- 只读代码审查提出的辅助权重/输入文件来源哈希缺口已修正，损坏或不完整的续跑结果会拒绝。
- 新 2D 明确为 SAM3D 投影到原图的预测关键点，不是 GT，也不假称独立 2D 检测器。检测/几何失败不删样本；零填充和 batch-anchor 耦合单列披露。
- GPU 1 的 kexun 作业未干预；全量 runner 每阶段 48h 上限，失败停止、不自动重试。尚无新的全量结果或胜负结论。

## 正文与引用

- 保留 44.5% / 48.6% 对 canonical average 的既有提升数字及两套独立 benchmark 表。
- 不再将 MetaPose 概括为替代而非利用单目估计器。明确其利用单目 3D 初始化，但仍使用 2D 观测和相机投影相关优化。
- 新增 RSB-Pose（IEEE TIP 34, 60–72, 2025；DOI 10.1109/TIP.2024.3490401）与 Dual-Diffusion（NeurIPS 37, 78079–78103, 2024；DOI 10.52202/079017-2482）。
- RSB-Pose 正式书目信息由 DOI 注册元数据核实；方法描述对照作者论文 https://arxiv.org/html/2311.14242v2 。Dual-Diffusion 对照官方 proceedings 页面及 BibTeX。
- Zotero 本地 API 未运行；没有启用服务或修改文献库，使用一手公开来源与本地 references.bib。
- 原源文件恢复快照：/tmp/ivc-review-before-20260916-Pz9vW6/（临时位置，非长期归档）。

## 门控：已测结果与边界

以下均为原 checkpoint 的推理干预，**不是重新训练消融**。common-13 MPJPE：

| 条件 | Unity 原门控+残差 | Unity 固定 0.5+残差 | Ski 原门控+残差 | Ski 固定 0.5+残差 |
|---|---:|---:|---:|---:|
| clean | 0.155069 | 0.210386 | 0.179398 | 0.184206 |
| 左侧 raw pose 全零 | 0.747376 | 0.748227 | 0.340014 | 0.344438 |
| 右侧 raw pose 全零 | 0.557821 | 0.569513 | 0.288344 | 0.289312 |
| 前半左损坏、后半右损坏 | 0.676524 | 0.693908 | 0.478189 | 0.484695 |
| 前半右损坏、后半左损坏 | 0.560598 | 0.567271 | 0.460667 | 0.465644 |

解释限制：

- alpha=1/0 是选择实际融合候选 base_l/base_r，不等于删除另一原始视角；两个候选各含另一视角对齐后的信息。
- 完整 raw-pose 全零和窗口切换也会破坏 batch anchor 和 Sim3 输入，因此是联合流水线极端压力测试，不是排除了归一化影响的门控可靠性实验。不能由此宣称模型能自动恢复丢失视角。
- 输出残差开关 disable_output_residual 与原 disable_residual（清零输入差分特征）分离；旧默认路径和 checkpoint key 兼容。
- 固定 0.5 且无输出残差是确定性对照，根本没有可训练输出路径。正式训练矩阵为 full、fixed_gate、no_output_residual 三组；第四格直接评估。
- 正式三组使用 seed 42、Unity 原 train/val 索引、训练与验证 batch 4096、100 epochs、相同原融合目标和 AdamW 配置，以完整验证集 all-15 MPJPE 选模，最后 test batch 256。完整模型也重新训练，以提供同协议控制；不替换原主表 checkpoint。
- 训练 smoke 产物不可用作论文结果。

冻结记录：logs/ivc_review_revision_20260916/gate_{unity,ski}_full/。
正式训练：logs/ivc_review_revision_20260916/gate_training_full/。
启动日志：logs/ivc_review_revision_20260916/gate_training_full.launch2.log。
启动 PID：2099918，GPU 0；每个配置的 progress.json/history.jsonl/metrics.json 是实际进度/结果依据，PID 本身不是完成证据。
初次 shell 后台启动未存活，保留其空 stdout.log，已改为独立进程会话启动；确认完成 epoch 0 和完整验证集后才报告正式运行。GPU 1 属于其他用户的活跃作业，未干预。

## 已纠正的旧门控表冲突

当前 main.tex 的门控区域表原样沿用了 paper/mmsports_reference/main_gate_table.tex:606–608 的 Unity 三行，但旧会议表标注为 representative evaluation settings，当前 IVC 表却称完整 held-out split。

同一归档 checkpoint 在本次完整 Unity 索引上复现主表误差，同时得到全关节 alpha 均值 0.6804228781，而稿件原值为 0.497。既有 temporal_offset/right_offset_p0 的 alpha CSV 数值整体也在 0.68 左右。当前没有证据支持把旧 0.497 当作现协议的全量统计。

使用正式 FILTERED_KPTS_MAPPING（不是旧 CSV 的可疑关节名顺序）分组，对逐关节全样本/时间均值计算总体标准差：

| 区域 | 0-based filtered indices | 本次 mean ± across-joint SD |
|---|---|---|
| 上肢 | 2,3,4,5,12,13 | 0.687928 ± 0.013231 |
| 下肢 | 6,7,8,9,10,11 | 0.668626 ± 0.010300 |
| 全部 | 0–14 | 0.680423 ± 0.014417 |

这与逐 joint-frame 标准差 0.112202 是不同统计量。旧 CSV 的 _alpha_joint_names() 从 Unity 源骨架索引排序取名，不能直接按那些标签重建 filtered 区域。

此前单独报告了这处冲突。用户随后要求将当前内容全部整理入稿，本轮据已披露的完整测试结果纠正区域表及“接近平衡”的解释，并保留本节审计来源。两张主 benchmark 数字不变。

## 效率证据

最终局部效率记录：logs/ivc_review_revision_20260916/local_efficiency_v2.json。
初版 local_efficiency.json 的 profiler FLOPs 漏计 Conv1d，已通过新增失败测试定位并改为算子派发计数；初版计数不入稿。

| 方法 | 参数数 | 纯权重 bytes | 矩阵/卷积 MACs | 中位延迟 ms |
|---|---:|---:|---:|---:|
| CanonFuse3D | 235524 | 942096 | 123979680 | 1.763 |
| local TCN | 168621 | 674484 | 5049600 | 0.114 |
| Avg. + SmoothNet | 689822 | 2759288 | 30873600 | 0.202 |

FP32，TF32 关闭，B=1/T=30/J=15，20 次预热，100 次计时，GPU 0 无其他计算进程时测量。已报告未计入的 SVD、归一化、非线性及元素运算，不能将此 MAC 数视作端到端完整运算量。计时含内部 Sim3，不含输入 canonicalization/RGB 前端，与原 N-view 计时边界不同。

## 已完成验证

- 54 项新旧联合回归通过；新测试覆盖候选融合、残差开关、旧 checkpoint 加载、默认输出重建、扰动不修改输入、尾批计数、原训练目标逐值一致、参数/卷积计数。
- 旧 HEAD 模型与新默认模型加载同一归档 checkpoint，独立固定输入上的预测和 alpha 逐位一致。
- 干净全量 Unity 两项原指标精确复现；Ski 在允许浮点容差内复现。
- 外部归档核验器通过，36 个外部方法展示数值核对通过。
- 两张主 benchmark 表 SHA-256 保持不变；表 3、4 仍为第 21、22 页。
- 已编译主稿/补充材料；无 undefined references、Overfull hbox 或 LaTeX error；已查看首页与新增效率表的实际渲染页。
- 三组最终训练消融及外部效率现已完成并入稿；无受损几何结果，不将本记录视为完整计划验收。
