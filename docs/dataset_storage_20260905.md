# IVC 论文数据集体积统计

统计日期：2026-09-05，日本时间。论文范围：`paper/ivc_draft_20260821/main.tex`。

单位为十进制 GB（1 GB = 1,000,000,000 字节）。表中大小是本机文件系统实际磁盘占用，采用 `du -B1` 统计，包含目录及小文件的块开销；不是压缩包大小。基础数据与完整目录是两种包含关系，不应相加。

| 数据来源 | 本机规模与论文用途 | 基础图像/视频和标注 | 现有完整相关目录 |
|---|---|---:|---:|
| Unity 合成滑雪数据 | 2 个角色、24 个动作、每动作 180 个视角，共 4,320 个相机序列；训练、测试和追加实验 | 29.77 GB | 215.81 GB |
| Ski-PTZ-Pose | 本机 10,197 张 PNG；固定索引为训练 150 条、测试 30 条；真实数据定量实验 | 0.93 GB | 5.32 GB |
| 作者自采双视角视频 | 6 组录制、12 个视频；仅定性展示 | 2.19 GB | 86.01 GB |
| 合计 | | **32.89 GB** | **307.14 GB（286.05 GiB）** |

基础数据合计精确值：32,889,212,928 字节。完整目录合计精确值：307,143,266,304 字节。

## 目录及体积明细

所有数据路径以下述目录为根：`/home/kaixu_chen/skiing/data`。

Unity：`skiing_unity_dataset/`，完整目录 215.8103 GB。

| 内容 | 路径或范围 | 磁盘占用 |
|---|---|---:|
| RGB 帧 | `data_pole_ski/{female,male}/Anim_*/frames/` | 27.8210 GB |
| 2D 标注，包含人物、雪板及雪杖 | `data_pole_ski/{female,male}/Anim_*/kpt2d/` | 1.9349 GB |
| 3D 标注 | `data_pole_ski/{female,male}/Anim_*/kpt3d/` | 0.0107 GB |
| 动作元信息 | `data_pole_ski/{female,male}/Anim_*/meta/` | 0.0006 GB |
| 原始数据配套可视化 | `data_pole_ski/{female,male}/Anim_*/viz/` | 83.3323 GB |
| 原生 SAM3D 完整推理档案 | `sam3d_body_results/inference/` | 52.0019 GB |
| 原生 SAM3D 姿态导出 | `sam3d_body_results/modalities_from_sam3d/` | 1.2696 GB |
| SAM3D 可视化 | `sam3d_body_results/visualization/` | 20.2132 GB |
| SAM3 结果及其可视化 | `sam3_results/` | 26.3141 GB |
| 已有两个划分索引 | `index_mapping/` | 2.8836 GB |

`fold_00.json` 的文件大小为 1.4565 GB，含 193,320 条训练、128,880 条验证、64,440 条测试记录。当前 Unity dataloader 按索引读取 `modalities_from_sam3d/` 中的姿态数组；完整 SAM3D 推理档案与姿态数组不能重复视为必需训练输入。E5 图像遮挡实验另需 RGB 帧和投影 2D 标注。

Ski-PTZ-Pose：`Ski-PosePTZ-CameraDataset-png/`，完整目录 5.3224 GB。

- 图像及 HDF5 标注：`data/`，0.9334 GB。
- SAM3D 推理：`sam3d_body_results/inference/`，3.2941 GB。
- SAM3D 可视化：`sam3d_body_results/visualization/`，1.0913 GB。
- 构造参考姿态：`pseudo_gt_exports/`，约 0.27 MB；索引约 0.21 MB。
- 根目录旁的 ZIP 是额外归档副本，未计入以上合计。

自采视频：

- `side_raw/`：2.1886 GB。六组为 `pro_1`、`pro_2`、`run_3`、`run_4`、`run_5`、`run_6`，每组两个视频。
- `sam3d_body_results/person/`：83.8220 GB，包含逐帧结果和额外整段归档。
- 抽查逐帧 NPZ 可见 `frame`、`pred_vertices`、`pred_keypoints_3d` 等字段，体积包含图像与人体网格，不能等同于纯姿态数据量。
- 当前定性推理程序读取每组的 `left/`、`right/` 逐帧 NPZ；论文数据示例图也直接读取其中的图像。

## 实验派生数据与统计边界

下面内容位于代码仓库，未计入上面的数据目录合计：

- 三种替代前端的测试预测：约 22.8 MB，`logs/eval_unity_frontend_generalization/predictions/`。
- 前端适配预测：约 176.3 MB，`logs/ivc_p1/frontend_adaptation/predictions/`。
- E5 遮挡推理目录：统计时约 287.4 MB，`logs/ivc_mmsports_extension/image_occlusion/inference/`；实验运行中，数值会增加。
- 本仓库 `ckpt/`：约 3.82 GB，属于模型权重等依赖，未计入数据集大小；训练日志下的 checkpoint 另计。

32.89 GB 描述基础图像、视频和标注，不包含现成复现所需的姿态导出、划分索引、模型权重及实验输出。307.14 GB 描述当前完整相关数据目录，包含可重生成的可视化和中间结果；它也不是最小复现包大小。本次只统计，没有打包、清理或改写任何数据。

论文将 Ski-PTZ-Pose 原数据集描述为约 20K 张图像；本机实际发现 10,197 张 PNG。这里报告本机版本的占用，没有把论文中的原数据集规模当作本地文件数，也未核实本机版本与原始发布版本的完整性关系。

## 依据

- 论文数据来源：`paper/ivc_draft_20260821/main.tex`。
- 当前路径及加载开关：`configs/dual2pose.yaml`。
- Unity 实际数据路径：活动 `fold_00.json` 与 `dual2pose/dataloader/unity_dataset_dual_view.py`。
- Ski 实际路径：`ski_pose_ptz_index_mapping.json` 与 `dual2pose/dataloader/ski_poseptz_dataset_dual_view.py`。
- 自采视频输入与图像来源：`eval_realworld_inference.py`、`paper/ivc_draft_20260821/scripts/generate_journal_figures.py`。
- 磁盘占用：逐目录 `du -B1`；文件数量与有效负载大小通过只读遍历交叉检查。
