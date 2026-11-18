# 固定姿态初始条件验证

`scripts/assess_static_initialization.py` 对预先声明的每个初始条件运行完整参考，再从保存的原生积分状态独立回放实际命令。使用已有的 `g1-static-pose-v1` 判据，不改变神经网络、PD 参数、物理模型或高度／倾斜终止条件。

该命令适用于已经准备好的静态参考与 CPU MuJoCo 推理包。动态参考会被拒绝。每组参考须为 30 至 600 秒，最多 64 组，合计不超过 3,600 秒仿真时间；不会自动把长参考截短，也不会在失败后重置。

## 运行

先检查整机内存、负载和相关进程。沿用项目的跨对话排他锁，计算库保持单线程；资源忙时稍后重试，不绕过锁。以下路径使用本机已有交付文件，数据和权重未随 Git 分发。

```bash
flock -n -E 75 /home/ubuntu/.cache/codex-heavy-work.lock \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 CMAKE_BUILD_PARALLEL_LEVEL=2 CTEST_PARALLEL_LEVEL=1 \
      MAX_JOBS=2 RAYON_NUM_THREADS=2 \
  nice -n 10 timeout --signal=INT --kill-after=30s 1200s \
  .venv/bin/python -I scripts/assess_static_initialization.py \
    --bundle exports/g1-tasks-20261117c/stand-hold-600-reset040/bundle \
    --motion data/npz/probes/stand60-initialization-20261117c.npz \
    --plan experiments/static-initialization/selected-20261117c.json \
    --output runs/my-static-initialization
```

输出目录必须不存在。进程保持 CPU 推理，默认限制进程树内存为 2 GiB，并为系统保留至少 12.5 GiB 或整机内存的 20%。锁退出码 75 表示暂缓。质量不通过的条件保留在结果中；输入或独立回放验证出错则停止，并保存部分结果。

## 条件文件

```json
{
  "schema": "g1-static-initialization-v1",
  "cases": [
    {"id": "baseline", "perturbation": {}},
    {
      "id": "combined-example",
      "perturbation": {
        "root_translation_m": [0.01, -0.01, 0],
        "root_world_rotation_vector_rad": [0.02, 0, 0],
        "root_linear_velocity_delta_m_s": [0, 0.025, 0],
        "joint_position_noise_std_rad": 0.005,
        "seed": 2026111805
      }
    }
  ]
}
```

首组必须是零扰动 `baseline`，其余 ID 不重复且只能包含字母、数字、连字符和下划线。未知字段、非有限数、布尔数值和越界种子会被拒绝。

| 字段 | 坐标与含义 | 输入范围 |
| --- | --- | --- |
| `root_translation_m` | 根位置的世界坐标增量 | 各轴 ±0.1 m |
| `root_world_rotation_vector_rad` | 世界坐标旋转向量，左乘原姿态 | 各轴 ±0.3 rad |
| `root_linear_velocity_delta_m_s` | 根线速度的世界坐标增量 | 各轴 ±1 m/s |
| `root_body_angular_velocity_delta_rad_s` | 根角速度的本体坐标增量 | 各轴 ±2 rad/s |
| `joint_position_noise_std_rad` | 独立高斯关节位置噪声，按契约初始软限位裁剪 | 0 至 0.1 rad |
| `seed` | 局部 PCG64 随机数发生器种子，不改变全局 NumPy RNG | 整数 0 至 2³²−1 |

上述范围是协议输入边界，不是模型恢复能力或验收阈值。旋转向量三个分量的合成角度可能大于单轴上限。所有增量仅在首帧施加，推理历史仍按契约重复第一条实际观测初始化。

## 证据

输出先固定条件文件、推理包和参考副本，之后各组运行只读取这些副本。每组保存：

- `result.json`、`trajectory.npz`：实际神经网络命令、状态、参考帧、每个物理子步的已求解接触力及原终止原因。
- `initialization.npz`：`mjSTATE_INTEGRATION` 完整状态，包括求解器热启动状态；只保存位置和速度不足以保证复现。
- `actuator-forces.npz`：每个物理子步的实际执行器力。
- `assessment.json`：重新从状态与接触计算的固定质量条件。
- `replay.json`：从声明和参考重建首帧，确认完整初始状态逐位一致；另建物理数据实例，从保存的积分状态执行记录的命令，逐位比较状态、接触记录及执行器力。

顶层 `result.json` 保留所有条件的结果和输入指纹。独立回放校验的是已记录命令对应的物理过程，不进行第二次策略推理，也不单独证明技能质量。某组 60 秒通过，仅支持该条件和该时长；不能据此宣称带扰动的 600／1,800 秒、任意扰动恢复或实机验证。

`selected-20261117c.json` 对应上轮用于选型的九组条件。`heldout-20261118.json` 在本轮运行前固定，含基线和十九组新条件；它们的结果应与原九组结果分别报告。
