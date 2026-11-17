# 当前可训练与部署的任务

本页区分训练入口、完整仿真运行和技能质量验收。2026-11-17 使用当前代码重新运行下列五个保留模型，均从参考首帧开始、全程由神经网络控制，不自动重置。数据、模型和运行记录是本机产物，未随 Git 分发。机器可读结果与来源摘要见 [复核记录](tasks/readiness-baseline.json)。

| 任务／模型 | 完整 CPU MuJoCo 运行 | 整段回报 | 平面 RMSE / m | 关节 RMSE / rad | 当前结论 |
| --- | --- | --- | --- | --- | --- |
| 基础行走参考 | 261.30 s | 0.6504 | 5.408 | 0.2933 | 可运行的行走演示，路径误差明显 |
| 锚点反馈行走参考 | 261.30 s | 0.8215 | 0.495 | 0.2963 | 当前行走参考演示中路径误差更小 |
| 跳跃参考 | 244.42 s | 0.8035 | 0.211 | 0.1350 | 严格验收未通过：最差一秒末端误差 0.25495 m，大于 0.25 m |
| 舞蹈参考 | 131.46 s | 0.8116 | 0.190 | 0.1379 | 严格验收未通过：末端误差及倒置姿态关节误差 |
| 格斗参考 | 244.86 s | 0.6097 | 0.743 | 0.2046 | 严格验收未通过：回报及最差窗口的姿态／末端误差 |

这些模型都已有训练记录、检查点和导出包，可以继续研究和仿真演示。完整运行只证明没有触发原高度／倾斜终止条件；跳跃、舞蹈、格斗仍按 `g1-skills-v2-full-motion` 的所有条件判定。行走不套用为动态技能设计的该判据，也不因此称为已通过同一严格门槛。所有结果限于指定训练参考与初始化，不是新动作泛化或论文 benchmark 成绩。

实机通信、控制接管和硬件验证尚未完成；现有产物的目标为 CPU MuJoCo 仿真。速度指令行走等其他控制接口不在已验证清单内。

## 本机模型与复现入口

推荐行走演示使用 `exports/g1-anchor-best-return/`，可续训检查点为 `runs/anchor-final-20261106/training-checkpoint.pt`。基础版位于 `exports/g1-best/`，对应 `runs/final/training-checkpoint.pt`。

```bash
./supermimic deploy --bundle exports/g1-anchor-best-return \
  --motion data/npz/g1/walk1_subject1.npz --output runs/my-anchor-walk
```

跳跃、舞蹈、格斗当前复验的模型位置与部署记录在复核 JSON 中逐项列出；这些目录是未通过严格门槛的实验产物，不以新的“已验收技能”名称发布。各任务使用自己的完整参考文件，训练时应显式选择片段，例如：

```bash
./supermimic train --profile g1 --motions data/npz/validation \
  --clips jumps1_subject1 --num-envs 32 --hours 0.25 \
  --output runs/my-jump-training
```

这是从零训练入口，运行时长不保证达到表内质量。其他两个片段名为 `dance1_subject1`、`fight1_subject2`。共享机器执行训练、评测和批量操作时，应按协作约定持有重任务锁、检查整机资源并保留至少 `max(12 GiB, 总内存的 20%)` 可用内存；本次复验实际保留 12.5 GiB，规则不代表其他运行中会话已自动限流。

训练默认每回合最多 10 秒，并按回报选择参考起点。需要学习更长的连续片段时，可显式设置 `--episode-length-seconds 30 --sampling-mode start`，从首帧训练最长 30 秒；原高度／倾斜终止条件仍生效。`uniform` 均匀选择参考帧，`return` 使用回报采样；难点窗口优先采样要求 `return`。严格续训继承检查点的时长与采样方式，显式参数可以覆盖；`--transfer` 使用新任务的默认值。

`--root-orientation-tracking-weight` 可单独调整训练中的根姿态辅助损失，`0` 表示关闭该辅助项。这些参数只影响训练，不修改部署时长、仿真控制器或技能验收阈值。

长回合的采样回报日志仍受固定容量约束。日志写满时结算连续时间块，提供 bootstrap 回调时沿用该估值，否则省略无法可靠估计的尾部；当前标准训练入口采用后者。这个结算只更新采样统计，不重置机器人或参考时钟，避免将早期奖励与覆盖到最后一格的晚期奖励拼成错误时间序列。

## 检查点是否能严格续训

插值或校准生成的新权重可能保留旧 Adam 状态供审计。若检查点明确声明 actor、critic 或世界模型优化器状态失效／未验证，严格续训会拒绝该状态，并提示 `--transfer`。迁移仅继承网络，重新建立优化器、采样器、计数和新种子的随机状态，不声称是不中断续训。

未含这些可选声明的旧检查点保持原读取兼容性。有效新优化器链中的历史父级失效记录不会使新链失效。导出与推理只使用网络，不要求保留的优化器可续训；因此“能部署”和“能严格恢复训练”是分别检查的能力。

识别的声明为 `native_policy_improvement.policy_optimizer_resume_valid`、`native_policy_improvement.critic_optimizer_resume_valid`、`native_dynamics_calibration.optimizer_resume_valid` 和 `native_dynamics_calibration.optimizer_correspondence_not_proved`。这些检查防止忽略已知失效声明；没有声明并不意味着已经独立证明了所有优化器张量与网络的对应关系。
