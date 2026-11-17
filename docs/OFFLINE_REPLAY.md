# 从已记录的物理轨迹构建策略起点

`RecordedStartReplay` 为离线策略训练按需组装 `StartBatch`。它保存每个状态及其前一个干净动作，采样时才读取所需历史，避免为全部控制步重复保存十帧历史。世界模型拟合仍使用实际执行动作；两类动作在有执行噪声时不同。

零执行噪声的标准部署轨迹中，`actions` 就是干净网络输出。例如：

```python
import numpy as np
import torch
from supermimic.data.recorded_replay import RecordedEpisode, RecordedStartReplay

with np.load("runs/my-jump/deploy/trajectory.npz", allow_pickle=False) as data:
    episode = RecordedEpisode(
        states=torch.from_numpy(data["states"]),
        clean_actions=torch.from_numpy(data["actions"]),
        frames=torch.from_numpy(data["reference_frame"]),
        xi=torch.tensor([1.0] + [0.0] * 32),
    )
replay = RecordedStartReplay([episode], history=10, device="cpu")
batch = replay.sample_starts(32, 10)
```

有执行扰动的分支必须提供独立记录的干净网络输出，例如 `policy_actions`；不能把执行噪声后的命令作为 `clean_actions`。`xi` 也应使用该次运行的真实上下文，上例仅适用于默认质量且无编码器偏置的确定性运行。

完整回合的开头会重复首状态，前序动作为零。若从真实轨迹中途分叉，`RecordedEpisode.initial_states` 与 `initial_actions` 必须同时提供：各有十行，按时间从旧到新排列；前者最后一行等于分支首状态，后者每行是对应状态之前的干净动作，最后一行是分叉前的最后一个干净动作。分支后续历史使用分支自己的输出，且不会越过其他回合。

`starts(indices, history)` 可取指定的控制起点，适合有记录的优先采样；`sample_starts` 提供均匀采样和真正回合起点的混合。只有没有显式暖启动历史的回合才属于重置起点；全是暖启动分支时，沿用普通回放的均匀采样回退。

浮点数据统一存为 float32。恒定的物理上下文按回合共享，用控制起点索引确定所属回合；小回合采用逐步存储更省空间时，保留逐步存法，字节预算包含回合边界索引。默认保留张量预算为 128 MiB，超过时会在值扫描和设备分配前拒绝；该预算不包括输入、临时转换或模型内存。输入文件指纹、接触证据及物理有效性由调用流程先核对，运行训练与批量检查仍遵守共享机器的锁和资源余量约定。

本机真实数据对照覆盖 25 个回合的 15,293 个控制起点，全部 `StartBatch` 字段逐位一致。保留的历史上下文张量由 61,294,476 字节降至 6,455,388 字节，减少 89.47%；其中回合上下文共享又比前一版 8,470,572 字节减少 23.79%。这些数字不是整个进程或整卡的内存降幅，也不表示采样速度提升。固定随机种子后，一次策略更新及 96 次 critic Adam 步的采样索引、网络、优化器和随机状态也全部逐位一致，详见 [物理数据与 GPU 对照证据](performance/recorded-history.json)。
