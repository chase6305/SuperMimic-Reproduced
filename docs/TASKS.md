# 当前可训练与部署的任务

本页区分训练入口、完整仿真运行和技能质量验收。当前可交接的模型包括已通过 600 秒交付复验和 1,800 秒延长耐久验证的固定姿态保持、新通过验收的首 120 秒跳跃参考（另保留 30 秒版本），以及 261.30 秒行走参考演示。原始完整跳跃、舞蹈和格斗仍是未验收预览。原五类任务的推理文件、参考数据和证据在本机 `exports/g1-tasks-20261117/`；新增跳跃位于 `exports/g1-tasks-20261117b/`。数据与权重未随 Git 分发。

| 任务 | 当前训练能力 | 当前仿真交付范围 |
| --- | --- | --- |
| 固定姿态保持 | 已验证严格续训；沿用原 120 秒训练参考 | 600 秒通过验收，同模型通过 1,800 秒延长耐久验证 |
| 首 120 秒跳跃参考（另保留 30 秒版） | 已验证严格续训；沿用原 244.42 秒训练参考，换参考使用 `--transfer` | 两个时长分别通过原有 23 项跳跃条件，并独立复验复制后的推理包 |
| 行走参考 | 有训练检查点与标准训练入口 | 261.30 秒完整参考演示；不是速度指令接口 |
| 原始完整跳跃参考 | 可训练，候选需重新进行全程评测 | 244.42 秒预览，尚未通过全部严格条件 |
| 舞蹈／格斗参考 | 已实际验证 `--transfer`；原保留优化器分组不兼容严格续训 | 原参考预览，尚未通过全部严格条件 |

表中部署均指指定参考和初始化下的 CPU MuJoCo 仿真；实机部署尚未验证。各任务的文件路径、训练参考和验收证据如下。

## 新增：120 秒跳跃参考

按评测前预先声明的范围，将原始跳跃参考的前 6,001 帧另行定义为完整 120 秒任务，不重采样或改变时钟。选定的支撑脚高度权重 200 模型在完整 6,000 个控制步中通过原有 23 项跳跃条件：整段回报 0.78271，平面 RMSE 0.21736 m，关节 RMSE 0.13526 rad，最差一秒末端误差 0.23544 m；83 个参考腾空事件中匹配 73 个，时序精确率 0.84680，腾空时长比 0.74837。

复制后的推理包与参考已独立重跑完整 120 秒，状态、动作、参考帧和物理接触全部逐位一致。前 5,990 个控制步与同一模型的原始完整轨迹逐位一致，末尾 10 步受参考前视钳位影响，未要求与全长轨迹一致。[120 秒证据](tasks/jump120-20261117.json)保留全部阈值、四项预先声明的延长验证及选型依据。另一模型的 60 秒任务因腾空时长比 0.59904 小于 0.6 而失败，未放宽门槛；一个时长的通过不能自动证明任意首段都通过。

```bash
./supermimic deploy --bundle exports/g1-tasks-20261117b/jump120/bundle \
  --motion exports/g1-tasks-20261117b/jump120/reference.npz --output runs/my-jump120
.venv/bin/python -I scripts/assess_skill.py \
  --bundle exports/g1-tasks-20261117b/jump120/bundle \
  --motion exports/g1-tasks-20261117b/jump120/reference.npz \
  --run runs/my-jump120 --skill jump --output runs/my-jump120-assessment.json
```

模型训练检查点为 `runs/task-optimization-20261117b/jump-pose100-stance200-resume/training/ckpt_latest.pt`，SHA-256 为 `a4db166b41d4361cd121fae3941c8ce9327d454df24d6a4154b5e8d79dccc746`。严格续训使用原训练数据 `data/npz/validation/jumps1_subject1.npz`；改用 120 秒参考训练时使用 `--transfer`。同一模型在原始 244.42 秒任务上的腾空时序精确率为 0.69172，仍低于 0.8，因此原始完整任务尚未验收。所有结果均限于指定参考、初始化和 CPU MuJoCo 仿真。

完整 120 秒保存轨迹的视频在本机 `exports/g1-tasks-20261117b/previews/jump120-full.mp4`；视频通过 CPU 软件渲染生成，用于查看已验证的动作，物理验收以上述轨迹及接触记录为准。

## 保留：30 秒跳跃参考

从原始跳跃参考保留首 1,501 帧，另行定义完整 30 秒任务，不重采样或改变时钟。使用在原始完整参考上训练的第 5,000 次迭代模型，它在 1,500 个控制步中通过 `g1-skills-v2-full-motion` 的 23 项跳跃条件：整段回报 0.80969，关节 RMSE 0.12488 rad，最差一秒末端误差 0.16819 m；29 个参考腾空事件中匹配 25 个，时序精确率 0.99024，腾空时长比 0.60831。时长比距 0.6 门槛较近，扰动与其他起点尚需另行验证。

复制后的推理包与参考文件已独立重跑完整 30 秒，状态、动作、参考帧和物理接触逐位一致。[验收证据](tasks/jump30-20261117.json)保留阈值、指纹及完整原任务的失败结果。同一模型在原始 244.42 秒任务上虽通过空间误差条件，腾空时序精确率仍只有 0.71142，低于 0.8；短任务通过不代表完整任务通过。

```bash
./supermimic deploy --bundle exports/g1-tasks-20261117b/jump30/bundle \
  --motion exports/g1-tasks-20261117b/jump30/reference.npz --output runs/my-jump30
.venv/bin/python -I scripts/assess_skill.py \
  --bundle exports/g1-tasks-20261117b/jump30/bundle \
  --motion exports/g1-tasks-20261117b/jump30/reference.npz \
  --run runs/my-jump30 --skill jump --output runs/my-jump30-assessment.json
```

可续训检查点为 `runs/task-optimization-20261117b/jump-full-pose100-decoder-h16/training/ckpt_0005000.pt`，SHA-256 为 `c511d5bffe21965e3ffda9680d636d4558d708b00f8c5b1795e329cb6f136494`。严格续训沿用原训练数据 `data/npz/validation/jumps1_subject1.npz`；改用 30 秒参考训练时使用 `--transfer`。所有部署与验收命令仍需遵守下述共享机器资源约定。

## 原完整参考的复验基线

下表是 2026-11-17 对五个原有保留模型的完整复验基线，均从参考首帧开始、全程由神经网络控制，不自动重置。来源摘要见 [复核记录](tasks/readiness-baseline.json)，本轮新增站立的结果见下方、[120 秒证据](tasks/static-hold-20261117.json)及 [600 秒证据](tasks/static-hold-600s-20261117.json)。

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

训练种子须为 `0..2**32 - 1` 范围内的整数，与环境的 NumPy 随机数接口一致；超范围值在命令行探测资源、创建输出目录或启动 worker 前拒绝；直接调用训练配置接口时，也会在读取检查点和分配模拟器前拒绝。[种子校验证据](performance/training-seed-validation.json)包含失败启动记录和边界测试。

训练默认每回合最多 10 秒，并按回报选择参考起点。需要学习更长的连续片段时，可显式设置 `--episode-length-seconds 30 --sampling-mode start`，从首帧训练最长 30 秒；原高度／倾斜终止条件仍生效。`uniform` 均匀选择参考帧，`return` 使用回报采样；难点窗口优先采样要求 `return`。严格续训继承检查点的时长与采样方式，显式参数可以覆盖；`--transfer` 使用新任务的默认值。

`--root-height-tracking-weight` 和 `--root-orientation-tracking-weight` 可单独调整训练中的根高度与根姿态辅助损失，`0` 表示关闭对应辅助项。这些参数只影响训练，不修改部署时长、仿真控制器或技能验收阈值。

`--foot-support-height-weight` 可独立调整参考支撑期的脚底碰撞几何高度辅助项，`0` 表示关闭。该项与根高度跟踪不同；严格续训继承保存的值，显式参数在目标预设及恢复配置后生效。

`--reset-noise-scale` 在 `0..1` 范围内缩放原有复位姿态、速度和关节扰动，并在 `--training-mode` 之后生效。正值开启复位扰动，`0` 关闭；观测、动作和物理参数随机化保持各自设置。因此 `--training-mode deterministic --reset-noise-scale 0.4` 可单独训练初始状态变化：根位置 X/Y 范围为 ±2 cm，横滚／俯仰为 ±0.04 rad。配置和导出元数据记录该比例，续训继承，旧检查点沿用比例 `1`，没有改变原有扰动范围。

长回合的采样回报日志仍受固定容量约束。日志写满时结算连续时间块，提供 bootstrap 回调时沿用该估值，否则省略无法可靠估计的尾部；当前标准训练入口采用后者。这个结算只更新采样统计，不重置机器人或参考时钟，避免将早期奖励与覆盖到最后一格的晚期奖励拼成错误时间序列。

离线物理轨迹可用 [RecordedStartReplay](OFFLINE_REPLAY.md) 按需组装策略历史，区分干净网络输出和实际执行扰动，并保留真实暖启动上下文。

## 检查点是否能严格续训

续训或迁移的计划配置会保留原锚点反馈开关，和已继承的前视帧数、目标维数保持一致，模拟器启动前即可读取正确的策略输入协议。[配置恢复回归证据](performance/goal-feedback-configuration.json)记录修复前的两个失败案例及修复后的验证。

插值或校准生成的新权重可能保留旧 Adam 状态供审计。若检查点明确声明 actor、critic 或世界模型优化器状态失效／未验证，严格续训会拒绝该状态，并提示 `--transfer`。迁移仅继承网络，重新建立优化器、采样器、计数和新种子的随机状态，不声称是不中断续训。

未含这些可选声明的旧检查点保持原读取兼容性。有效新优化器链中的历史父级失效记录不会使新链失效。导出与推理只使用网络，不要求保留的优化器可续训；因此“能部署”和“能严格恢复训练”是分别检查的能力。

识别的声明为 `native_policy_improvement.policy_optimizer_resume_valid`、`native_policy_improvement.critic_optimizer_resume_valid`、`native_dynamics_calibration.optimizer_resume_valid` 和 `native_dynamics_calibration.optimizer_correspondence_not_proved`。这些检查防止忽略已知失效声明；没有声明并不意味着已经独立证明了所有优化器张量与网络的对应关系。

标准训练器的每个 Adam 优化器使用一个参数组。原生标定实验中采用其他分组的检查点，需要通过 `--transfer` 建立新优化器；严格续训会在配置阶段提示不兼容。当前舞蹈和格斗保留检查点的世界模型属于这种情况；跳跃保留检查点已通过真实 CLI 续训、导出和整段回放。

舞蹈和搏击也已实际验证 `--transfer`：分别完成 54 次和 48 次更新并导出新模型。舞蹈新模型完成整段回放；搏击新模型在 11,164 / 12,243 步终止，未采用。交接包使用原有保留预览，迁移后仍需完整评测再选择部署版本。

## 本轮新增：静态姿态保持

`stand30-walk-first-frame.npz` 将行走首帧固定为 30 秒参考，所有参考速度为零。使用标准训练入口迁移并续训后，五个保留模型中两版通过了训练前制定的姿态与稳定性条件。第 28,000 次迭代模型的完整 1,500 步回放中，平面误差为 0.0220 m、关节误差为 0.0495 rad、根高度误差为 0.0112 m、根姿态误差为 0.0395 rad、关节速度 RMS 为 0.1159 rad/s，非脚部触地为零。

该模型的训练检查点为 `runs/task-readiness-20261117/stand-full30-pose-decoder-h16/training/ckpt_0028000.pt`，已验证推理包为同名 `-evaluations/ckpt_0028000/export/` 目录。后续检查点有位置漂移，部署时使用已经验证的版本。

同一模型随后通过完整 120 秒、6,000 步试验：平面误差 0.02354 m、关节误差 0.08068 rad、根高度误差 0.02471 m、根姿态误差 0.03452 rad、关节速度 RMS 0.05800 rad/s，非脚部触地为零。最初 30 秒的状态、动作及参考帧与前一次回放逐位一致。另一次 600 秒对照虽完成运行，但高度误差 0.0918 m 超标；该版的已验收时长为 120 秒。

针对这一高度漂移，以 120 秒静态参考迁移训练，将高度损失权重从 10 调到 100；该阶段最佳模型在完整 600 秒中的平面误差为 0.10900 m，仍超过 0.1 m 门槛。随后从该检查点严格续训，仅将平面损失权重从 100 调到 200。第 12,000 次迭代模型通过完整 600 秒、30,000 步验收：平面 RMSE 0.03015 m、关节 RMSE 0.06657 rad、根高度 RMSE 0.01295 m、根姿态 RMSE 0.04535 rad、关节速度 RMS 0.02114 rad/s，非脚部触地为零。后续第 16,000、20,000 次及最终检查点均未通过，因此交付已验证的第 12,000 次模型。

同一模型又通过完整 1,800 秒、90,000 步延长耐久诊断：平面 RMSE 0.03009 m、关节 RMSE 0.06638 rad、根高度 RMSE 0.01301 m、根姿态 RMSE 0.04443 rad、关节速度 RMS 0.01221 rad/s，非脚部触地为零。前 600 秒的状态、动作、参考帧及接触记录与已验收轨迹逐位一致。它使用同一固定姿态与初始化，没有新增训练或放宽条件，不能据此推断扰动恢复或其他姿态的表现。

600 秒版训练检查点为 `runs/task-readiness-20261117/stand-height100-xy200-resume/training/ckpt_0012000.pt`，SHA-256 为 `6984a9c7ea58c96834349941e986e3b28f6bddba706448c9c3f690e8f248a5f8`。其训练参考为 `data/npz/probes/stand120-walk-first-frame.npz`；600 秒文件仅用于更长的部署测试，严格续训应继续使用原 120 秒训练文件。修改训练数据时使用 `--transfer` 并重新评测。

继续训练时可使用以下入口，损失权重与 120 秒起点采样协议从检查点继承，学习率和本轮计算预算显式设置：

```bash
./supermimic train --profile g1 \
  --checkpoint runs/task-readiness-20261117/stand-height100-xy200-resume/training/ckpt_0012000.pt \
  --motions data/npz/probes --clips stand120-walk-first-frame \
  --num-envs 32 --wm-batches-per-update 2 --wm-batch-size 128 \
  --policy-batch-size 32 --replay-capacity 512 --learning-rate 0.0000005 \
  --training-mode deterministic --compile-policy --compile-world-model \
  --iterations 100000 --hours 0.25 --output runs/my-static-continuation
```

后续训练的质量可能回退，需导出候选并重新运行完整部署与验收，再决定采用哪个模型。

本机可直接使用 600 秒交接目录；旧版 `stand-hold/` 仍保留 120 秒证据。旧版的原始训练参考是 `data/npz/probes/stand30-walk-first-frame.npz`，严格续训使用 30 秒文件；各条目的 `source.strict_resume_training_motion` 记录训练与部署参考的区别：

```bash
./supermimic deploy --bundle exports/g1-tasks-20261117/stand-hold-600/bundle \
  --motion exports/g1-tasks-20261117/stand-hold-600/reference.npz --output runs/my-static-hold
.venv/bin/python -I scripts/assess_static_hold.py \
  --bundle exports/g1-tasks-20261117/stand-hold-600/bundle \
  --motion exports/g1-tasks-20261117/stand-hold-600/reference.npz \
  --run runs/my-static-hold --output runs/my-static-hold-assessment.json
```

`scripts/assess_static_hold.py` 会重新计算真实轨迹的误差，并核对静态参考、文件指纹与完整参考时钟。固定的 `g1-static-pose-v1` 条件为：完成至少 30 秒；平面、关节、根高度和根姿态 RMSE 分别不超过 0.1 m、0.25 rad、0.05 m、0.15 rad；关节速度 RMS 不超过 0.5 rad/s；非脚部触地为零。该结果限于这一个参考姿态，和跳跃／舞蹈／格斗的动态技能判据分别报告。

复现 1,800 秒运行时，使用同一个 `stand-hold-600/bundle`，将 `--motion` 改为 `exports/g1-tasks-20261117/stand-hold-600/extended-1800/reference.npz`；部署与评测都使用该参考，并指定新的输出目录。`extended-1800/` 已附完整运行、评测和预先制定的诊断计划。

公开评测器也复验了原行走模型的静态对照：它完成 30 秒，但关节误差 0.50005 rad 超过 0.25 rad，因此仍判定失败。通过站立验收需要姿态跟踪与稳定性同时满足，单纯站住或完成时长不够。

前一阶段跳跃预览的最差一秒末端误差从基线 0.25495 m 降到 0.25409 m，仍高于 0.25 m，原交接包保留未验收标记。后续模型虽通过全长空间误差条件，腾空时序仍未通过；新增首 120 秒任务的验收范围见上方。其余候选、真实 CLI 训练结果和资源约束见 [优化记录](tasks/optimization-20261117.json)。世界模型的预测误差改善未转化为动态技能验收通过，未采用相关候选。

新增四小时阶段的完整原参考候选、离线历史内存对照及腾空诊断见 [后续优化记录](tasks/optimization-20261117b.json)。该记录明确区分原始完整任务和首段任务的验收结果。

前一阶段累计完成 35 次原始完整动态参考候选评测，均未通过，未替换交接版本。舞蹈增强姿态约束后的三个候选分别在 897–2,151 步终止；格斗候选也未改善原有质量。已验收的首 120 秒跳跃与固定姿态版本保留，后续试验继续独立记录。

北京时间 22:25 开始的后续阶段见 [新一轮优化记录](tasks/optimization-20261117c.json)。动作蒸馏使用实际历史并重新进行完整原参考评测，训练拟合误差与实际技能验收分别记录；研究性辅助目标生成的候选需用 `--transfer` 开始标准训练。

本轮对 180／200／220 秒首段的九项评测均未通过，随后另行声明的 150 秒三项后续评测也均未通过，已验收范围仍为 120 秒。两阶段全部结果保留在新一轮记录中。独立残差支路的零修正版本在完整跳跃／舞蹈上与原轨迹逐位一致，但八个训练变体仍未通过，未引入新的公开网络结构；[数值对照证据](performance/neural-residual-controls.json)记录这一实验限制。
