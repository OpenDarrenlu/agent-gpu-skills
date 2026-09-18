# PipeFusion：如何用PCIe互联GPU低成本并行推理扩散模型

**原文链接**: https://zhuanlan.zhihu.com/p/699612077

**下载时间**: 2026-06-05 22:38:32

---

作者丨方佳瑞 来源丨https://zhuanlan.zhihu.com/p/699612077 编辑丨GiantPandaCV

今年二月Sora横空出世，我们正在跑步进入视频生成时代。同时，Sora巨大的部署难题也引爆了长序列的DiT推理方法研究热潮。面对这个问题，我的团队最近在这这方面做了一个非常有趣的工作叫PipeFusion。

PipeFusion可以显著降低DiT模型并行推理的带宽需求，能在PCIe互联的GPU上更有性价比地部署DiT并行推理。

论文地址：https://arxiv.org/abs/2405.14430
代码开源：https://github.com/PipeFusion/PipeFusion

## Diffusion Model推理原理和特性

扩散模型的训练过程：给一个图片，经过很多步骤，每一步加噪声最后变成一个全是噪音的图片。训练过程就是预测一个Noise Predictor的监督学习任务。

扩散模型的推理过程：给一个噪声，通过Noise Predictor来通过多次去噪，最后变成一个有意义的图片。每一次去噪就是一个Diffusion Step。

扩散模型推理和LLM的一个很大的差异点：扩散模型是重复计算很多相同的Diffusion Step，而且连续的Diffusion Step之间输入数据和激活状态之间存在的高度相似性。我们称之为 **输入时间冗余（Input Temporal Redundancy）**。

## Diffusion Model并行方法对比

1. **张量并行（TP）**: 通信量最大，同步AllReduce，参数和激活都减少到1/N
2. **序列并行（SP）**: Ulysses和Ring方式，通信复杂，参数内存需求不变
3. **DistriFusion**：Displaced Patch Parallelism，使用前一个timestep的stale K,V与当前步骤的fresh K,V结合
4. **PipeFusion**：Displaced Patch Pipeline Parallelism，通过流水线方式组织异步通信和计算

## PipeFusion核心思想

PipeFusion将输入图像分割成M个不重叠的Patch，并将DiT网络均匀分成N个阶段，每个阶段由不同的计算设备顺序处理。每个设备以流水线方式处理其分配阶段的一个Patch。

利用输入时间冗余性质，设备无需等待接收当前流水线步骤的全空间形状KV Activation即可开始其阶段的计算，它使用前一步骤的stale activations代替fresh activations为当前步骤所用。

PipeFusion通信上仅传输不同stage之间的hidden states，而不是像DistriFusion一样传递每层的K,V。因此PipeFusion通信量没有L项，极大减少了通信开销。

## 实验效果

使用pixart-alpha模型，在8xL20(PCIe)和4xA100(PCIe)上生成1024px到8192px图像：
- PipeFusion基本都获得了最低的延迟表现
- 在8xL20上，1024px和4096px分别达到了其它最佳方法1.47倍和1.31倍的加速比
- 相较于单GPU Baseline，PipeFusion分别实现了2.46倍和4.3倍的加速比
- DistriFusion和SP在4092px分辨率生成任务OOM了