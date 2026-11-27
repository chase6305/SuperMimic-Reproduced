# G1 仿真演示

GIF 来自 Unitree G1 策略在 CPU MuJoCo 中生成的保存轨迹。渲染逐帧读取实际状态，使用跟随相机和平面背景。时间标记使用原轨迹时钟，循环在片段结束后回到起点。

| GIF | 原轨迹片段（秒） | 对应任务与结果 |
| --- | --- | --- |
| [行走](walk.gif) | 245–250 | 优化模型执行完整 261.30 秒参考，平面 RMSE 为 0.19607 m。[行走优化报告](WALK_OPTIMIZATION.md) |
| [跳跃](jump.gif) | 6–11 | 原跳跃参考的首 120 秒作为单独任务，通过全部 23 项检查。[120 秒验收证据](https://github.com/chase6305/SuperMimic-Reproduced/blob/44e8c2ff97c86bf7d3fc4fe5e5247a807692e8d9/docs/tasks/jump120-20261117.json) |
| [固定姿态保持](stand.gif) | 594–599 | 复位扰动训练版在默认初始化下完成 600 秒复验，平面 RMSE 为 0.00411 m。[600 秒验收证据](https://github.com/chase6305/SuperMimic-Reproduced/blob/44e8c2ff97c86bf7d3fc4fe5e5247a807692e8d9/docs/tasks/static-reset-600s-20261117c.json) |
| [舞蹈预览](dance.gif) | 105–110 | 131.46 秒完整参考的翻转片段，严格技能验收未通过。[任务证据](https://github.com/chase6305/SuperMimic-Reproduced/blob/44e8c2ff97c86bf7d3fc4fe5e5247a807692e8d9/docs/tasks/readiness-baseline.json) |
| [格斗预览](fight.gif) | 20–25 | 244.86 秒完整参考的片段，严格技能验收未通过。[任务证据](https://github.com/chase6305/SuperMimic-Reproduced/blob/44e8c2ff97c86bf7d3fc4fe5e5247a807692e8d9/docs/tasks/readiness-baseline.json) |

完整 244.42 秒跳跃参考要求 12,221 步和至少 80% 的腾空时间精度。一版跑完全段并通过其余检查，腾空精度为 71.28%。另一版跑完全段后达到 72.93%，最差 1 秒末端位置 RMSE 为 0.340 m，限值为 0.25 m。表中的验收和动画使用独立的首 120 秒任务。

## 渲染验证

GIF 的离屏缓冲按目标画幅分配，阴影贴图上限为 2048 像素。五段 GIF 在 Mesa llvmpipe 下用相同输入重复渲染，两次的时钟、轨迹摘要和 GIF 字节一致。行走在最新 CPU 部署中重新生成轨迹后，也得到与 README 相同的 GIF。

五段 GIF 均为 384 × 288、50 帧、5 秒循环，合计 3.14 MiB。最初五段同轨迹样片从 128 色试制参数切换到当前 64 色编码参数后，体积减少约 32%。MP4 对照的首帧与原实现逐像素一致。另测了 12 秒、25 FPS 的跳跃 GIF，输出 300 帧，采样进程树 RSS 峰值约为 733 MiB。

对同一段行走轨迹交替渲染三对，5 秒 GIF 的中位用时由 6.70 秒降到 4.67 秒，采样峰值内存中位数由 708 MiB 降到 352 MiB。60 秒、640 × 480、50 FPS 的边界复验生成了 3,000 帧，每帧 20 毫秒，采样峰值内存约 434 MiB。该次渲染用时约 254 秒。

## 再生成

准备可信的导出包和对应部署轨迹。下例使用行走演示的本机源文件，其他片段使用表中的起点和各自输入。历史片段的输入路径相对于生成它们的工作区。使用独立工作树时可传入对应的绝对路径。

```bash
./supermimic render --bundle runs/opt-7h/walk-final/mixed-compact-tempo-child-075/bundle \
  --trajectory runs/opt-7h/walk-final/mixed-compact-tempo-child-075/deploy/trajectory.npz \
  --start-seconds 245 --seconds 5 --width 384 --fps 10 \
  --output runs/my-g1-demo/walk.gif
```

| 演示 | `--bundle` | `--trajectory` |
| --- | --- | --- |
| 行走 | `runs/opt-7h/walk-final/mixed-compact-tempo-child-075/bundle` | `runs/opt-7h/walk-final/mixed-compact-tempo-child-075/deploy/trajectory.npz` |
| 跳跃 | `exports/g1-tasks-20261117b/jump120/bundle` | `exports/g1-tasks-20261117b/jump120/evidence/trajectory.npz` |
| 固定姿态保持 | `exports/g1-tasks-20261117c/stand-hold-600-reset040/bundle` | `exports/g1-tasks-20261117c/stand-hold-600-reset040/evidence/trajectory.npz` |
| 舞蹈 | `exports/g1-tasks-20261117/dance-preview/bundle` | `exports/g1-tasks-20261117/dance-preview/evidence/trajectory.npz` |
| 格斗 | `exports/g1-tasks-20261117/fight-preview/bundle` | `exports/g1-tasks-20261117/fight-preview/evidence/trajectory.npz` |

[演示指纹](manifest.json)记录渲染参数以及 GIF、轨迹、部署结果、契约和检查点的 SHA-256。

动作参考来源于 Ubisoft La Forge 的 LAFAN1 动作集及 Unitree 重定向版本。机器人和场景资产来自 Unitree 与固定版本的 mjlab。来源和许可见 [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)、[Unitree 重定向版](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)和[机器人资产许可](../../src/supermimic/licenses/Unitree-BSD-3-Clause.txt)。
