# Pegasus 全链路准备与多节点运行

使用账号 HP260146、队列 gen_S。完整实验范围见 `experiment_inventory_20261030.md`，历史任务来源见同名 tasks CSV。本次只准备和诊断；不提交全量重跑，不修改论文，不覆盖历史结果。

## 运行目录

根目录 `/work/HP260146/chenkaixu/` 下保留仓库、同名 `_data`、同名 `_runtime` 三个目录。源数据布局不变。环境入口为仓库的 `pegasus/runtime/env.sh`：PyTorch 与 MetaPose TensorFlow 分开使用 Python 3.11 环境。

最终准备计划：

`/work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime/plans/full_chain_20261030_final_v2/`

该目录包含 `plan.json`、冻结的 `source/` 与 `source_hashes.json`、各任务 `pbs/`、新输出 `main/`、`core/`、`results/`、`supplement/`，以及 `tasks/`、`submissions/`、`receipts/`。准备阶段后三者为空。v1/v2/v3 setup 与早期 plans 仅为配置诊断快照，保留作为审核记录，不作为后续正式入口。

## 依赖与并行

- 220 个调度任务，包括 38 个正式训练/适配配置，以及准备、评测、汇总任务；不能称为 220 次训练。
- 独立任务可各占一个节点同时提交。主模型权重通过验证后，依赖它的适配与评测才可提交。
- STRIDE 两数据集属于逐窗口 TTA；Unity 的 64,440 窗按最多 2,048 窗分为 32 份，每窗仍执行原来的 30 次更新。合并前验证范围无遗漏、无重叠和输入/权重哈希一致，随后对完整 cohort 评分。
- gen_S 实测单任务时限 24 小时，用户组最大排队请求 200。220 项不能一次性全部排队；按依赖释放任务。分片预计预算仍受实际节点速度影响，超时会失败而不会发布完整结果。
- 当前提供 `ready` 和 `submit-one`，不含后台自动调度守护进程，也没有全量提交命令。后续获准启动时可按 ready 列表并行提交；本次不启动。

只读查看准备状态：

```bash
cd /work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch
source pegasus/runtime/env.sh
plan="$DUAL2POSE_RUNTIME/plans/full_chain_20261030_final_v2"
"$DUAL2POSE_PYTHON" -m pegasus.runtime.chain ready --plan "$plan"
```

获准运行后，单任务入口为 `chain submit-one --plan "$plan" --job JOB_ID`。成功提交和正在执行的任务不重复提交；明确 qsub 拒绝会移入 `submission_failures/`，可重试。若提交期间连接/进程中断，保留提交锁，必须先核对 qstat，防止重复作业。训练已失败的任务保留日志和结果，不自动覆盖重试；需建立新的计划/输出空间。

## 路径与完整性

旧 manifest 中的工作站路径通过只读 `DUAL2POSE_INPUT_PATH_MAP` 解析，不改写原文件及其哈希。计划命令的 Python、数据、缓存、baseline 源码和结果路径都使用 Pegasus 路径。无映射或缺失输入会报错，不静默替代。

每个节点执行前验证冻结源码、上游产物与 checkpoint。完成回执原子发布且不覆盖已有回执。52 项遮挡/时间偏移的 alpha CSV、正确关节标签和两类图均须验收。最终汇总重新验证全部任务的结果、训练历史与选模记录，然后输出 mm 主表；失败或缺失不能标为完整。

旧结果仍保留在仓库 logs，论文仍保留原目录。新训练不会写入这些历史结果目录。直接重新运行老的工作站 manifest 不受本入口保护，不能拿它恢复本计划。

## 诊断证据与边界

远端 setup/chain_20261030_v1：

| 检查 | NQSV 任务 | 结果 |
|---|---|---|
| DeciWatch Ski 1 epoch 训练及测试 | 196890.nqsv | 通过 |
| STRIDE 1 个训练窗口、完整 30 次更新 | 196891.nqsv | 通过 |
| MetaPose 训练帧更新、权重重载及既有 Ski 完整重评分 | 196892.nqsv | 通过 |

MetaPose Ski 重评分与既有结果数值完全一致：MPJPE 355.2573 mm，PA-MPJPE 91.3513 mm。这里只验证迁移一致性，不据此重新选择模型。

setup/chain_20261030_v2 的路径诊断任务 196904.nqsv 通过：31 个入口可导入；5 个 raw cache 索引哈希一致；15 份前端/RGB manifest 路径与 C07 的 2,160 个流路径可读取。官方 DeciWatch、STRIDE、google-research checkout 均无改动。196915.nqsv 对一个训练窗口的 STRIDE 串行/分片比较最大绝对差为 0。

本地针对路径、调度保护、baseline 适配及协议的测试为 116 passed、2 subtests passed；有本地 GPU 故障引起的 NVML warning。审核发现的并发回执、提交失败重试、最终结果漏验证已修复，并单独复核通过。

以上是运行兼容性和局部数值一致性证据，不代表在 Pegasus 已完成 38 套正式实验。原始 RGB 到各前端的重新推理不在此次诊断范围；本轮沿用已迁移、已核验的 raw 前端预测，重新训练/评分融合与比较方法。

## 后续启动授权

用户随后授权全量重跑。自动调度配置和日志入口见 `pegasus_rerun_launch_20261030.md`；前述“未启动”描述保留为准备阶段记录。
