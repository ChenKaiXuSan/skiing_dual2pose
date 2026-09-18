# C07: 同源 RGB 遮挡与几何质量比较

## 完成与入稿（2026-09-17）

全量前端与评估均已完成，三套缓存共 2160 流，77 个独立评估单元，每单元 64,440 窗口。主文 5.7/表 8–9 和补充 S14/表 S11–S15 已入稿，PDF 已重新编译（46/25 页）。

准确标定下 DLT-residual 在 6/7 条件有最低 MPJPE；1° 下仍是 6/7。3° 下 CanonFuse3D 在干净和三种 ratio-0.5 条件的 MPJPE/PA-MPJPE 最低，三种 ratio-1.0 条件则是平均基线最低。严重受损结果包含前端缺失、零值 fallback 和 batch-reference 退化耦合，不归因于纯网络失败。

新增 455 表格数值、2160 缓存哈希及代码/标定来源核验通过。完整结果及局限见 paper/ivc_draft_20260821/evidence/c07_matched_occlusion_20260917/README.md。以下保留预先确定的协议。


## 范围与协议（全量运行前固定）

本实验补充既有 E5，不替换其 18 个条件，也不替换主 benchmark。原 E5 已执行 RGB 遮挡 → SAM3D → CanonFuse3D / canonical average；新增工作是同次推理保存同一选中人物的预测 2D 和 3D，再接入几何基线。

- Unity fold_00，原完整测试索引 64,440 个窗口，T=30；保持原始顺序、重复采样帧和 batch=256。验证集首个固定 pair 仅用于接口检查，不作为论文结果。
- 输入条件：clean，以及 random ratio=0.5/1.0 × 左/右/双侧，共 7 种。ratio 是关节选中概率，不是图像像素遮挡比例。
- 沿用 E5 的 seed-42 BLAKE2b mask、方块尺度、填充值和原图。GT 2D 仅定位遮挡块，不进入几何重建。
- SAM3D 每张图一次推理，按原最大 bbox 规则选择人物，成对保存 15-joint 3D 与原图像素 2D。2D 是 SAM3D 投影关键点，不宣称独立检测器或人工真值。
- clean / random_0p5 / random_1 各导出一套完整相机流；左右组合复用对应缓存，不为每个 pair 重复推理。
- CanonFuse3D、canonical average、DLT、25-pixel 重投影筛选 DLT、DLT-residual MLP；两个学习模型均使用原归档 checkpoint，不重训，不调测试阈值。
- 右相机局部 +y 轴外参旋转扰动 0°/1°/3°，左相机与右相机中心不变。用 K Q K^-1 P 构造；这些角度是人工敏感性测试，不是实测标定误差。
- 原 all-15 canonicalization 后切 common-13；MPJPE、PA-MPJPE、acceleration error。无相机输入的两种方法在不同角度复用相同结果。共 77 个独立 method/condition/angle 指标单元，展开显示为 105 个，不虚报独立运行次数。

## 失败处理及解释边界

- 检测失败保留原 E5 零 3D，同时保存 NaN 2D 和 false validity；不把 (0,0) 当作缺失 2D，也不删除窗口。
- DLT 缺失/无效点在 canonicalization 前填零，显式报告数量；这只是用于全索引评分的 fallback，不是成功恢复。
- 保留原 robust DLT 语义：只对有限但超阈值的 DLT 点做时间插值；全轨迹超阈值时回退到原有限 DLT。缺失 2D 导致的 NaN 不由该旧插值器补齐。
- DLT-residual 接收零填充预测、有效性 mask、有限重投影误差特征。无效误差填零并保留 false mask，不传入 GT。
- 不改变 batch-anchor canonicalization。首样本首帧检测/几何缺失可能影响整个 batch，另报缺失 anchor 与受影响窗口数；不能把此实验解释为排除了归一化影响的纯三角测量鲁棒性证明。
- 成功/失败在唯一源图像和重复 pair-frame 位置两种分母下分别统计；所有方法最终保留相同窗口数。
- 新 clean 和遮挡输入均重新推理，不能自动声称它们与历史 E5/主表预测逐位相同。旧数字保持原档，不用旧 clean DLT 拼接新遮挡表。

## 实现与核验

- `dual2pose/eval/run_matched_occlusion_frontend.py`：成对导出、严格缓存校验、无覆盖原子发布；锁定本地 SAM3D/MHR/ViTDet/MoGe 权重及架构/代码哈希。
- `dual2pose/eval/evaluate_occluded_geometry.py`：帧/样本清单与旧缓存一致检查、全部 7 条件和角度、失败统计、77 独立单元、原模型 checkpoint 校验。
- `dual2pose/experiments/run_occluded_geometry.py`：必须先通过验证集 smoke，再顺序执行全量前端和评估；独立日志、互斥锁、失败即停止、不自动重试、每阶段硬超时 48 小时。
- `tests/test_occluded_geometry.py`：同源 mask、2D 坐标不二次缩放、失败语义、旧缓存保护、重复帧、零角度恒等、相机中心、缺失几何、矩阵和损坏结果恢复校验。
- 独立只读审查指出的两项溯源缺口已修正：辅助权重哈希、真实输入 manifest 哈希均绑定断点恢复。

## 产物位置与执行入口

最终 smoke：`logs/ivc_review_revision_20260916/occluded_geometry_smoke_v2/`。
全量：`logs/ivc_review_revision_20260916/occluded_geometry_full/`。

```bash
/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python -u -m dual2pose.experiments.run_occluded_geometry \
  --smoke logs/ivc_review_revision_20260916/occluded_geometry_smoke_v2 \
  --output logs/ivc_review_revision_20260916/occluded_geometry_full \
  --gpu 0 --timeout-hours 48
```

进度以 `runner_status.json`、`frontend/progress.json`、三个前端 manifest、`evaluation/progress.json` 和七个结果 JSON 为准。PID 存活不是完成证据。GPU 1 为其他用户活跃任务，本实验不使用、不干预。

只有完整覆盖、有限值和全部产物核验通过后才将新数字写入论文。`paper/` 与 `logs/` 继续不追踪；本轮不 commit/push。
