# SuperMimic-Reproduced · Unitree G1

只维护 SuperMimic 的独立复现和 G1 29 自由度训练／部署链路。参考本地 EmbodiedForge 的验证经验，算法底座来自固定版本 `CuboiLeo/SuperMimic-Reproduced@d9be385`；本项目不依赖 EmbodiedForge 的 Python 包、绝对路径或已有训练结果。来源和原始文件摘要见 [third_party/UPSTREAM.json](third_party/UPSTREAM.json)，MIT 许可证保留在 [LICENSE](LICENSE)。

核心流程：真实 MuJoCo Warp 仿真采样 → 学习世界模型 → 通过想象轨迹做 BPTT 策略更新 → 确定性评测 → 导出 CPU 策略与 G1 控制契约 → CPU MuJoCo 部署。未使用官方模型权重，缩小配置不代表论文规模复现；仿真部署不代表已上实机。

## 交付范围与限制

当前交付为 G1 训练、恢复、评测、导出与 CPU MuJoCo 部署的代码基线。默认配置沿用固定上游的算法接口；额外位姿反馈、跟踪辅助项和局部参数微调均为显式启用的实验选项，不代表论文默认算法或已通过技能验收。

本机早期两组从零训练的模型均连续执行了 261.3 秒行走参考；最高整段归一化回报约 0.6504，关节 RMSE 约 0.2933 rad，平面 RMSE 约 5.408 m。详细历史指标和来源见 [结果报告](docs/RESULTS.md)。这些结果仅覆盖训练动作，路径误差明显；连续执行不等于高精度模仿。

跳跃、舞蹈、格斗的严格技能验收尚未通过，后续技能优化与实验编排在独立 PR 中审阅，不作为本基线合并的完成条件。未交付官方权重、全量论文 benchmark 或实机控制。数据、权重、`runs/` 与 `exports/` 为未入库的本地产物，克隆后需要自行准备数据并训练或提供可信检查点。

## 安装和检查

训练环境：Linux、Python 3.12、NVIDIA CUDA；依赖版本及 mjlab 提交由 `uv.lock` 固定。

```bash
uv sync --locked --group dev
./supermimic doctor
```

`./supermimic` 自动使用本仓库 `.venv` 并清理继承的 Python/ROS 路径。安装后也可使用 `supermimic` 命令。

## 准备动作

使用自己有权使用的 G1 动作。CSV 无表头，每帧为根位置 3 维、根四元数 `xyzw` 4 维、29 个关节角；转换后刚体四元数为 `wxyz`，控制频率为 50 Hz。

```bash
.venv/bin/python -I scripts/data/csv_to_npz_batch.py \
  --input-dir /path/to/g1-csv --output-dir data/npz/converted \
  --input-fps 30 --workers 2

./supermimic prepare --source data/npz/converted \
  --clips walk1_subject1 --output data/npz/g1
```

已有 NPZ 可直接作为 `prepare --source`。准备过程检查帧率、29 关节顺序、30 个刚体、数组形状、有限数值和四元数归一化，复制并保存 SHA-256。不会自动下载数据。LAFAN1 重定向数据、机器人资产和模型分别遵守其来源许可；代码许可证不替代这些许可。

CSV 在解析前限制为 128 MiB，重采样后每条最多 250000 帧；NPZ 解压大小上限为 512 MiB。更大的动作应先切片。转换默认 2 个、最多 4 个 CPU 工作进程，按每进程 4 GiB 加父进程 0.5 GiB 做资源准入和运行监控；工作进程使用单线程并关闭 CUDA。紧凑库打包同样在导入 SDK 前做资源检查。

## 训练与续训

先运行闭环小配置：

```bash
./supermimic train --profile smoke --num-envs 8 \
  --iterations 60 --hours 0.05 --output runs/smoke
```

开始独立 G1 训练：

```bash
./supermimic train --profile g1 --num-envs 16 \
  --hours 1 --output runs/g1-walk

./supermimic train --profile g1 --num-envs 16 \
  --checkpoint runs/g1-walk/training/ckpt_latest.pt \
  --hours 1 --output runs/g1-walk-resume
```

`--iterations` 是累计迭代目标，`--hours` 限制本轮训练循环；进程监督另给初始化留 180 秒。新运行目录必须不存在，避免覆盖历史结果。续训恢复模型、优化器、采样器和随机状态，重新填充未保存的 replay buffer；不宣称与不中断训练完全等价。只加载可信的本地 checkpoint。

`--checkpoint-interval 2000` 可增大编号检查点的保存间隔，减少长训练的存储开销；省略时保持原500步，结束时仍保存最终检查点。续训若需要其他间隔须显式指定，该选项不改变优化器或按迭代计算的学习率。

| 配置 | 世界模型 | 策略编码器／解码器 | 想象长度 | 默认环境数 |
| --- | --- | --- | --- | --- |
| `smoke` | 128,128 | 128,64 | 8 | 8 |
| `g1` | 512,512,512 | 512,256,128 | 32 | 32 |
| `paper-width` | 2048,2048,2048 | 2048,1024,512,256 | 32 | 32 |

`--policy-horizon 8` 可显式调整每次策略更新的想象控制步数（1–64）；严格续训默认继承保存值，新数据迁移使用所选配置的默认值，显式参数始终优先。原动作时长、控制周期和完整物理验收不变。

`--policy-update-scope action-head` 可只微调原动作输出层，`decoder` 只更新解码器，默认 `all` 保持原全部参数更新。冻结层仍传递想象轨迹的输入导数，完整 Adam 参数组及冻结层历史状态保留；严格续训继承范围，迁移默认 `all`，显式参数优先。该选项是减少微调漂移的实验，目前没有完整技能通过证据。

`paper-width` 仅保留论文网络宽度，批量和并行环境仍缩小。默认关闭 Torch 编译，避免编译阶段占用大量资源。`sm-train` 和 `python -m supermimic.scripts.train` 也进入同一受控训练入口。

本机探测中，`--compile-policy` 的完整 G1 小网络训练约从 2.3 提升到 5.8 迭代／秒；首次编译需额外时间，此速度不保证适用于其他机器。完整网络宽度也已在相同显存上限内运行，训练质量仍需用连续动作评测判断。

已实测的较大采样配置如下，仍使用默认 2304 MiB 显存上限：

```bash
./supermimic train --profile paper-width --num-envs 128 \
  --policy-batch-size 48 --wm-batch-size 512 --replay-capacity 1000 \
  --compile-policy --compile-world-model --wm-batches-per-update 10 \
  --hours 0.25 --seed 42 --output runs/paper-width-g1
```

上述完整宽度配置的采样峰值显存约 1646 MiB；这不是其他机器上的内存保证。更改批量、环境数与回放容量后，会在创建环境前重新估算预算。想象终止分支与 critic 拟合的数值稳定修复见 [实现约定](docs/REPRODUCTION.md)。

## 显存与内存保护

独立技能验收脚本 `scripts/assess_skill.py` 支持 `--max-rss-gib`（0.25–2 GiB，默认 2），可选择更严格的进程内存上限；系统仍保留 8 GiB 可用内存和 20 GiB 磁盘。预算不足时会推迟或停止验收，评分逻辑和门槛保持不变。

默认本次进程显存预算 2304 MiB、内存预算 6 GiB；为其他任务保留 4096 MiB 全局空闲显存、8 GiB 可用内存和 20 GiB 磁盘空间。启动前要求预算与余量同时满足；运行中每 2 秒检查整个工作进程树，显存统计包含 Torch、Warp 与 MuJoCo 的分配。Torch 另设置分配器预算，给非 Torch 分配留出空间。

```bash
./supermimic train --profile g1 --num-envs 16 --hours 1 \
  --max-gpu-mib 2304 --reserve-gpu-mib 4096 --max-rss-gib 6
```

阈值触发时只停止本次子进程，并请求在当前迭代完成后保存检查点；超过 30 秒退出宽限才强制结束。采样监测不能绝对阻止采样间隔内的瞬时峰值，因此应保守设置预算。运行目录记录真实退出状态 `run.json`、执行结果 `result.json`、日志 `worker.log`、资源序列 `resources.jsonl`、源码快照及训练配置。检查点原子替换，只保留 latest 和最近 3 个编号文件。

训练配置还会在创建回放和物理环境前估算模型、优化器、激活、回放和参考数据的显存需求；这是保守启发式，不是峰值保证。CPU 操作在导入 SDK 前检查资源：导出／部署的进程树 RSS 上限为 2 GiB，准备／打包／回放为 4.5 GiB，同样保留 8 GiB 可用内存和 20 GiB 磁盘；资源记录位于 `<output>.resources.json`。CPU 操作关闭 CUDA 可见设备。此前 169 次成功导出／部署的采样峰值分别约 1.54／1.45 GiB；本轮据此收紧预算，较大输入仍可能被资源保护中止。

## 评测、导出与部署

```bash
./supermimic eval --checkpoint runs/g1-walk/training/ckpt_latest.pt \
  --num-envs 1 --output runs/eval-g1

./supermimic play --checkpoint runs/g1-walk/training/ckpt_latest.pt \
  --viewer viser

./supermimic export --checkpoint runs/g1-walk/training/ckpt_latest.pt \
  --output exports/g1

./supermimic deploy --bundle exports/g1 \
  --motion data/npz/g1/walk1_subject1.npz --output runs/deploy-g1
```

导出包包含 `policy.ts`、`model.mjb`、`contract.json`。契约明确 29 关节顺序、动作尺度、默认姿态、PD 参数、四元数约定、10 帧 term-major 历史和 680 维目标；编译模型包含地面、碰撞及执行器，与训练场景同源。导出比较非零输入与不同 batch 的动作数值。CPU MuJoCo 部署不分配 CUDA 显存，加载前检查文件摘要；可加 `--viewer` 打开本机窗口。

部署从动作首帧初始化，持续记录真实物理轨迹；高度／倾斜终止时立即停止，不以反复重置掩盖失败。`result.json` 报告是否完成整条动作，`trajectory.npz` 保存状态、动作和参考帧。实机通信、控制接管与硬件验证尚未完成。

GPU 评测与 CPU 部署均以 `frames - 1` 个控制覆盖参考区间，报告动作覆盖率及失败后补零的整段回报；执行部分的平均回报另行保留。CPU 与 Warp 物理、奖励采样时钟存在差异，分别报告结果。

## 验证

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
uv lock --check --offline
uv build
```

本次基线 CPU 验证为 **157 项通过、24 项跳过**（另有 12 个 subtests 通过）。跳过项受 CUDA、动作数据或显式编译开关限制。

CPU 测试覆盖 G1 FK、状态表示、训练／部署历史和目标特征一致性、PD／执行器映射、动作数据检查及资源监督。依赖真实 CUDA 和动作数据的上游集成测试会在条件不足时跳过。CPU CI 使用同一锁文件和上述测试入口，不需要本机数据或训练权重。CUDA 集成与编译测试须在具备相应数据和 GPU 的环境另行运行；CPU CI 通过不代表技能验收通过。

保存轨迹后可用 CPU 软件渲染生成回放，避免占用训练显存：

```bash
./supermimic render --bundle exports/g1 \
  --trajectory runs/deploy-g1/trajectory.npz \
  --output runs/deploy-g1/preview.mp4 --seconds 10
```

需要系统的 Mesa EGL 库；已验证 `llvmpipe` 渲染。可选 `train --compile-policy` 在相同预算内编译想象步骤，首次编译时间也计入训练循环预算。详细输入、时钟和指标定义见 [实现约定](docs/REPRODUCTION.md)。

运行记录可汇总为 JSON，汇总过程不加载 checkpoint：

```bash
.venv/bin/python -I scripts/summarize_runs.py --output runs/summary.json
```

## 后续技能优化（独立审阅）

未通过技能验收的候选、资源记录和后续方案见 [专项记录](docs/SKILLS.md)、[能力探测](docs/CAPABILITIES.md)、[实验进展](docs/EXPERIMENT_PROGRESS.md) 和 [历史工作记录](docs/WORKLOG.md)。`scripts/optimize_g1.py` 为待单独验收的分段优化编排入口；本 PR 不采用新模型或修改默认训练配置。
