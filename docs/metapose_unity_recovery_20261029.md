# MetaPose Unity 推理恢复与最终合并

用户授权：GPU 1 从已有权重继续推理和评测，后续依赖自动执行；纳入 Ski 稳定版，不覆盖既有结果、不改论文。

## 原始状态

- 原队列：`logs/per_frame_common13_v1/completion_20261029T042127Z`。
- Unity 已完成 3×10 轮训练；选中的各阶段 epoch 为 9、10、5，依据完整验证集 2D MSE。
- GPU 0 在测试数据初始化阶段故障，退出码 -6；训练权重未丢失。
- 原 `test_x_init.npy` 为未完成缓存，不作为有效输入复用。
- Ski 稳定版：`logs/per_frame_common13_v1/metapose_stable_20261029T052359Z`。

## 恢复执行

- 独立目录：`logs/per_frame_common13_v1/metapose_unity_recovery_20261029T122908Z`。
- tmux：`metapose-unity-recovery-20261029T122908Z`。
- GPU 使用物理 UUID `GPU-c6477171-5e81-2fb4-5367-6fc5a674444f`（物理 GPU 1）。进程内部显示 GPU 0 是可见设备重编号。
- `source/` 为独立源码快照：原队列冻结的 Unity 模型和评测实现，只加入恢复入口与恢复队列。原源码快照不修改。
- 恢复入口：`dual2pose/experiments/resume_metapose.py`；队列：`dual2pose/experiments/run_metapose_completion_recovery.py`。
- `state.json` 保存各步骤状态与退出码；各步骤单独日志；任何步骤失败都会阻塞后续步骤。

执行顺序：GPU 真实权重预检 → Unity 完整测试初始化与推理 → Unity per-frame/common13 评分 → GPU 效率测试 → Ski 稳定版训练溯源检查 → Ski 独立重新评分 → 所有结果验证与合并。

## 检查与保护

- 14 项恢复、评分和完成检查单元测试通过。
- 预检只读取固定的前 64 个训练帧，不读取 GT、不据测试结果选择权重；分批预测最大绝对差为 `3.175237850427948e-14`。
- 完整 Unity 测试覆盖应为 64,440 个窗口、1,933,200 个帧条目（窗口内条目，不宣称全部互不重复）、25,131,600 个关节帧点。
- Ski 重新评分应为 30 个窗口、900 个帧条目；应复现稳定版已有结果。
- 三阶段权重、原训练历史、原模型代码均校验 SHA-256；生成的预测记录完整形状、有限值覆盖和 SHA-256。
- 新目录保存 `protected_before.json`：论文全文件内容哈希及原有实验文件保护记录；大数组采用大小/mtime 检查，并对关键权重、预测另行核验内容哈希。
- 原始队列的 failed/blocked 状态保留为历史记录，恢复结果由新目录 `final/coverage.json` 证明；不将旧失败状态改成成功。
- 本次论文不作任何更改，也不更新旧的保护基线。

## 汇总口径

最终表为 `final/main_comparison_mm.csv`，全部长度误差换算为 mm，加速度误差为 mm/frame²。
仅替换 Unity、Ski 两行 MetaPose：Unity 使用原验证集选模的训练权重；Ski 使用单独的稳定训练版本。
两者都标为 MetaPose adapted，并分别记录训练配方、原训练路径和评测来源，不混称为同一训练配方或官方原始协议。
效率测试使用固定输入，分别给出 learned solver 与 adapters 总耗时，不据此声称所有方法具有同输入的公平速度排名。

实时进度读取恢复目录的 `state.json` 与 `unity/progress.json`；只有最终检查通过才可宣称本轮收尾完成。

## 完成记录

2026-10-29 20:37:52（北京时间）完成全部 7 个步骤，退出码均为 0。
最终检查通过 61 项结果任务（56 项既有补跑结果 + 5 项 MetaPose 训练/评分/效率结果），然后完成最终汇总步骤。
对旧队列的失败状态不作改写；新恢复目录 `final/coverage.json` 与 `final/verification.json` 为本次收尾依据。

| 数据集 | 方法 | MPJPE (mm) | PA-MPJPE (mm) | 加速度误差 (mm/frame²) |
|---|---|---:|---:|---:|
| Unity | CanonFuse3D | 50.8272 | 40.9639 | 16.6831 |
| Unity | MetaPose adapted，已有权重恢复推理 | 212.0243 | 105.4423 | 44.6958 |
| Ski | CanonFuse3D | 125.0828 | 92.5269 | 34.6306 |
| Ski | MetaPose adapted，稳定训练版 | 355.2573 | 91.3513 | 73.1054 |

完整 28 行方法对比见恢复目录 `final/main_comparison_mm.csv`。
Ski 稳定版独立重评分复现原结果；两个数据集的 canonical-average 控制指标均与已有对比表一致。
MetaPose Unity 精度仍差于 canonical average；Ski 稳定版虽有较低 PA-MPJPE，直接 MPJPE 仍高，不能声称整体优于 CanonFuse3D。

MetaPose 在 GPU 1（RTX A6000）上每批 30 帧的耗时中位数为：solver 8.751 ms，包含适配器 14.054 ms；预热 20 次、测量 100 次。
固定窗口的独立预测复现最大绝对差为 `2.384185791015625e-07`（原始输出单位）。
恢复日志没有 Traceback、CUDA 启动失败、显存不足或 NaN；保留了 TensorFlow 的重复库注册、未安装 TensorRT、重追踪及 SVD 转换提示，这些未造成失败。
论文内容与既有结果保护检查均无差异；本次没有重新训练 Unity 或覆盖其原始输出。
