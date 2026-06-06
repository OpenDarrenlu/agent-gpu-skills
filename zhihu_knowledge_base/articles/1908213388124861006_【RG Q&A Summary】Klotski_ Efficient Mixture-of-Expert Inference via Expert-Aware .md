# 【RG Q&A Summary】Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/1908213388124861006

---

这里是中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。 Reading Group 的主页地址：ADSL Reading Group bilibili 链接：USTC-NHPCC的个人空间
Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline

Klotski: 基于专家感知型多批次流水线的高效MoE模型推理

作者：Zhiyuan Fang1, Yuegui Huang1, Zicong Hong2, Yufeng Lyu3, Wuhui Chen1, Yue Yu4, Fan Yu3, Zibin Zheng1

1 中山大学, 2 香港科技大学, 3 华为技术有限公司, 4 鹏城实验室

MoE (Mixture-of-Experts) 模型凭借其稀疏结构，使得语言模型可以扩展至万亿级参数，同时避免了计算成本的大幅增长。然而，MoE 模型庞大的参数规模给推理带来了挑战，尤其是 GPU 内存增长速度难以匹配参数量的增长。尽管卸载技术能够减少 GPU 内存需求，但由于 MoE 模型计算与 I/O 负载高度不均衡，推理过程中往往会出现大量 GPU bubble，影响系统吞吐。

为此，本文提出 Klotski，一款专为 MoE 设计的推理引擎。Klotski 通过构建专家感知的多批次流水线，有效消除推理过程中的 GPU bubble，大幅提升资源受限环境下的推理吞吐。其核心策略是在多个批次之间共享权重，从而延长计算时间，使其完全覆盖下一层的加载时间。然而，与密集模型不同，多批次计算会增加输入 token 数量，从而激活更多专家，导致 I/O 开销增大，可能引入额外的 GPU bubble。为此，Klotski 设计了一种 MoE 适配的多批次推理调度策略，仅预取高频使用的热门专家，并利用这些专家的计算时间隐藏其他专家的加载开销，以减少层内 bubble。此外，Klotski 还会测量硬件能力，并根据存储资源及计算与 I/O 速度的差异，自动搜索最优推理配置。实验结果表明，与现有方法相比，Klotski 在吞吐-延迟权衡方面表现更优，吞吐量最高可提升85.12×。

Q&A

Q1：Slides 背景部分使用实测数据说明 PCIe 带宽无法直接满足专家预取需求，其中展示了最多激活专家数与可预取专家数的 gap，这二者是如何计算的？

A1：此处使用 DeepSeek-V2-Lite 模型测试，其单个 token 每层激活的专家数为 k=6。那么，最多激活专家数即为 batch size * k * #layers，而可预取专家考虑的是理想情况，计算方式为 decode step time / PCIe bandwidth / expert size。

Q2：文章提出的方法需要将多个 batch 的 token 进行重排，这需要知晓每个 batch 的专家使用情况，如何做到？

A2：对于同一层内的某一模块，多 batch 流水线会串行完成所有 batches 的计算，再开始计算下个模块。具体来说，先计算完所有 batches 的 attention，再计算完 gate，才会开始 experts 计算。因为 gate 均已被计算过，所以进入 MoE 层时即可知晓全部 batches 的专家使用情况。

Q3：基于问题 2 的回答，这样的串行计算模式，需要同时保留多个 batches 的 KVCache，造成内存压力，这一问题如何解决？

A3：KVCache 也会被卸载至 CPU 内存或磁盘，可缓解一定内存压力。即便如此，本文方法的性能确受到 KVCache 大小影响，因此输入的文本长度不宜过长，总 batch 数也应尽量小。

Q4：虽然本文方法更适合 offline 推理，但如果有对 SLOs 不敏感的 online 场景，是不是也可以使用该方法？

A4：是的。

Q5：多卡会给本文的场景带来什么影响？

A5：本文使用多 batch 策略的根本原因是 PCIe 带宽无法满足 overlap 需求。多卡情况下，每张 GPU 各自独立地与 CPU 内存连接，因此系统总体 PCIe 带宽更高。此情况下，若使用 EP，那么单 batch 的策略可能更为合适，例如 slides 背景部分提到的 fMoE 工作考虑的便是这种场景。

Q6：现有的MoE Offloading工作均依赖于专家预测算法，例如跨层预测，但据观察，在 DeepSeek-V3 上，这种算法可能失效，如何看待这一问题？

A6：笔者目前的确没有测试过 DeepSeek-V3 上的预测效果，但个人认为可以考虑使用 ProMoE 提出的训练预测器思路，以提高预测准确度。
