# 当前可训练与部署的任务

本页区分训练入口、完整仿真运行和技能质量验收。当前可交接的模型包括已通过固定姿态条件的 120 秒站立保持、261.30 秒行走参考演示，以及三个未验收的动态技能预览。它们的推理文件、参考数据和证据在本机 `exports/g1-tasks-20261117/`，详情见该目录的 `README.md` 与 `catalog.json`。

下表是 2026-11-17 对五个原有保留模型的完整复验基线，均从参考首帧开始、全程由神经网络控制，不自动重置。数据、模型和运行记录是本机产物，未随 Git 分发。来源摘要见 [复核记录](tasks/readiness-baseline.json)，本轮新增站立的结果见下方及[验收证据](tasks/static-hold-20261117.json)。

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

`--root-height-tracking-weight` 和 `--root-orientation-tracking-weight` 可单独调整训练中的根高度与根姿态辅助损失，`0` 表示关闭对应辅助项。这些参数只影响训练，不修改部署时长、仿真控制器或技能验收阈值。

长回合的采样回报日志仍受固定容量约束。日志写满时结算连续时间块，提供 bootstrap 回调时沿用该估值，否则省略无法可靠估计的尾部；当前标准训练入口采用后者。这个结算只更新采样统计，不重置机器人或参考时钟，避免将早期奖励与覆盖到最后一格的晚期奖励拼成错误时间序列。

## 检查点是否能严格续训

插值或校准生成的新权重可能保留旧 Adam 状态供审计。若检查点明确声明 actor、critic 或世界模型优化器状态失效／未验证，严格续训会拒绝该状态，并提示 `--transfer`。迁移仅继承网络，重新建立优化器、采样器、计数和新种子的随机状态，不声称是不中断续训。

未含这些可选声明的旧检查点保持原读取兼容性。有效新优化器链中的历史父级失效记录不会使新链失效。导出与推理只使用网络，不要求保留的优化器可续训；因此“能部署”和“能严格恢复训练”是分别检查的能力。

识别的声明为 `native_policy_improvement.policy_optimizer_resume_valid`、`native_policy_improvement.critic_optimizer_resume_valid`、`native_dynamics_calibration.optimizer_resume_valid` 和 `native_dynamics_calibration.optimizer_correspondence_not_proved`。这些检查防止忽略已知失效声明；没有声明并不意味着已经独立证明了所有优化器张量与网络的对应关系。

标准训练器的每个 Adam 优化器使用一个参数组。原生标定实验中采用其他分组的检查点，需要通过 `--transfer` 建立新优化器；严格续训会在配置阶段提示不兼容。当前舞蹈和格斗保留检查点的世界模型属于这种情况；跳跃保留检查点已通过真实 CLI 续训、导出和整段回放。

舞蹈和搏击也已实际验证 `--transfer`：分别完成 54 次和 48 次更新并导出新模型。舞蹈新模型完成整段回放；搏击新模型在 11,164 / 12,243 步终止，未采用。交接包使用原有保留预览，迁移后仍需完整评测再选择部署版本。

## 本轮新增：静态姿态保持

`stand30-walk-first-frame.npz` 将行走首帧固定为 30 秒参考，所有参考速度为零。使用标准训练入口迁移并续训后，五个保留模型中两版通过了训练前制定的姿态与稳定性条件。第 28,000 次迭代模型的完整 1,500 步回放中，平面误差为 0.0220 m、关节误差为 0.0495 rad、根高度误差为 0.0112 m、根姿态误差为 0.0395 rad、关节速度 RMS 为 0.1159 rad/s，非脚部触地为零。

该模型的训练检查点为 `runs/task-readiness-20261117/stand-full30-pose-decoder-h16/training/ckpt_0028000.pt`，已验证推理包为同名 `-evaluations/ckpt_0028000/export/` 目录。后续检查点有位置漂移，部署时使用已经验证的版本。

同一模型随后通过完整 120 秒、6,000 步试验：平面误差 0.02354 m、关节误差 0.08068 rad、根高度误差 0.02471 m、根姿态误差 0.03452 rad、关节速度 RMS 0.05800 rad/s，非脚部触地为零。最初 30 秒的状态、动作及参考帧与前一次回放逐位一致。另一次 600 秒对照虽完成运行，但高度误差 0.0918 m 超标；该版的已验收时长为 120 秒。

本机可直接使用交接目录：

```bash
./supermimic deploy --bundle exports/g1-tasks-20261117/stand-hold/bundle \
  --motion exports/g1-tasks-20261117/stand-hold/reference.npz --output runs/my-static-hold
.venv/bin/python -I scripts/assess_static_hold.py \
  --bundle exports/g1-tasks-20261117/stand-hold/bundle \
  --motion exports/g1-tasks-20261117/stand-hold/reference.npz \
  --run runs/my-static-hold --output runs/my-static-hold-assessment.json
```

`scripts/assess_static_hold.py` 会重新计算真实轨迹的误差，并核对静态参考、文件指纹与完整参考时钟。固定的 `g1-static-pose-v1` 条件为：完成至少 30 秒；平面、关节、根高度和根姿态 RMSE 分别不超过 0.1 m、0.25 rad、0.05 m、0.15 rad；关节速度 RMS 不超过 0.5 rad/s；非脚部触地为零。该结果限于这一个参考姿态，和跳跃／舞蹈／格斗的动态技能判据分别报告。
