# 【RG Q&A Summary】[OSDI'24] Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/16742425732

---

这里是中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

大模型推理通常分为两个阶段：Prefill阶段和Decode阶段。提升这两个阶段的吞吐量和延迟是优化系统性能的关键。当今的LLM推理系统普遍采用批处理（batching）策略以提高吞吐量，但这往往会带来吞吐量与延迟之间的Trade-off。此外，由于显存限制，LLM推理系统通常采用流水线并行（pipeline parallelism）来支持多卡任务，这种并行策略可能引发流水线气泡问题，从而对模型推理的延迟产生一定影响。 本文引入了 Chunked Prefills 技术，将 Prefill 请求 分割为大小相近的小块，并采用 Stall-Free Scheduling，使得新请求可以在不暂停正在进行的 Decode 任务的情况下被添加到批处理中。无阻塞调度解锁了通过大批量提升吞吐量的机会，同时最小化了批处理对延迟的影响。此外，由于采用了固定的Token budget，不同迭代之间的负载相对均衡，减少了流水线气泡的出现。通过这些技术，Sarathi 在多种模型和硬件配置下显著提升了推理性能，尤其是在尾延迟约束下。以 Mistral-7B 在单个A100 GPU 上为例，Sarathi 实现了2.6倍的Capacity（保证SLO的最高RPS）提升；在 Yi-34B 上，使用两个 A100 GPU 可获得3.7倍的提升。使用 流水线并行 在 Falcon-180B 上，Sarathi 提高了5.6倍的Capacity。

Q&A

Q1：本文方法在分离场景下还有意义吗？
A1：在分离场景下P、D任务之间没有干扰，但可能存在硬件利用不足的情况。

Q2：本文方法和Sliding window的优劣对比？
A2：Silding window是模型本身自带的技术，而并非推理系统技术。silding window对于读取数据量的影响相较于本文方法可能较小。

Q3：Capacity的定义？
A3：Capacity是指在满足p99 TBT要求下所能达到的最大的RPS，和goodput不同。goodput则是指在满足给定的slo下能够处理的最大RPS

Q4：为什么会产生两类Bubble？
A4：Bubble的产生原因是两个相邻流水线阶段消耗时间的不同。两类中，第一种是Prefill之间的耗时不同导致的，而第二种是Prefill和Decode之间的耗时不同导致的。

Q5：在多轮对话中是否可以复用前轮的KV-cache？如果复用的话，该轮Prefill是memory-bound还是compute-bound？
A5：在本文方法中应该是重新进行了Prefill计算。如果进行KV-cache复用的话，该情况可能不是一个memory-bound的Prefill问题。本文的分类方法可能并不适用。
