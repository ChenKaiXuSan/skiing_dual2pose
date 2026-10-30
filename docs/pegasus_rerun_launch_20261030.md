# Pegasus 全量重跑启动配置（2026-10-30）

用户已授权开始完整重跑并自动按依赖接续、多节点并行；论文和既有结果必须保留。本次沿用此前核验的 220 任务、38 训练/适配配置，无协议或训练预算改动。

- 计划：`/work/HP260146/chenkaixu/Skiing_Canonical_DualView_3D_Pose_PyTorch_runtime/plans/full_chain_20261030_final_v2`
- 冻结实验代码：`0cb26de` 时的计划 source；自动调度器单独部署，不能改动该冻结目录。
- 账号/队列：HP260146 / gen_S；每任务独立请求一个节点，最多64个在途请求，每30秒轮询。实际运行并发由NQSV分配。
- 远端 tmux：`canonfuse-full-20261030`。本机 SSH 断开不影响远端 tmux 和已提交的 NQSV 作业；登录节点重启仍需恢复控制器。
- 控制状态：计划下 `dispatcher/state.json`；完整事件：`dispatcher/events.jsonl`；控制器日志：`dispatcher/console.log`。
- 任务输出和错误：`tasks/JOB/run.log`、`submissions/JOB/{stdout,stderr}.log`；正式验收回执：`receipts/JOB.json`。
- 新结果只进入该计划的 `main/core/results/supplement/final`；历史 logs、已有结果、论文均不覆盖。

自动接续要求：worker验收成功、产物哈希通过、请求退出队列、至少30秒日志落盘宽限且stdout/stderr存在并检查后，才释放下游。失败节点阻塞其后代，其他独立分支继续。失败任务不覆盖重跑。qstat失败时不推断任务丢失、不新提交；提交状态不明保留占位，后续明确回执可自动恢复。明确qsub拒绝采用120秒递增、最多900秒的退避重试，不因暂时额度不足放弃实验。

控制器使用flock防止重复运行。恢复时使用原计划和控制状态，禁止删除submission目录或已有任务目录。若需只停止后续提交，可创建计划下 `dispatcher/STOP`，已经排队/运行的任务不会因此取消。

实现验证包含依赖自动推进、重复排除、并发容量、未知提交恢复、队列查询失败、结果哈希变化、迟到错误日志及单控制器锁等场景。状态文件中的 `complete` 才代表通过验收，`validating` 仍在等待最终日志；`unknown` 要持续等待/核对，不算成功。
