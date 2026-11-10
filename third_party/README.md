# 来源

`UPSTREAM.json` 记录导入时的上游文件摘要，不要求修改后的工作树继续匹配这些摘要。上游为 MIT 许可证的 `CuboiLeo/SuperMimic-Reproduced`，固定提交 `d9be385b3eeb13a638fb33fb200d3bfd9be9839f`；本轮从用户指定的 EmbodiedForge 本地缓存读取该提交的 Git blob，没有复制其虚拟环境、缓存、其他算法或历史报告。

本仓库继续保留原世界模型、BPTT 策略和 G1 环境实现，并修改了回放终止标签、运动学批量计算、续训回放补充、资源预算、原子检查点及运行记录。新增独立 CLI、数据检查、策略导出和 CPU MuJoCo 部署。实验中对照使用的 EmbodiedForge 本地训练模型单独标记来源，不称为本轮从零训练成果。

mjlab 固定为 `8bb2bc9e187797689ec82c78c35019afd2fd0376`，其 Apache-2.0 代码与 Unitree BSD-3-Clause 机器人资产通过依赖包提供。导出包携带对应许可证；数据与权重的使用范围仍以各自来源为准。
