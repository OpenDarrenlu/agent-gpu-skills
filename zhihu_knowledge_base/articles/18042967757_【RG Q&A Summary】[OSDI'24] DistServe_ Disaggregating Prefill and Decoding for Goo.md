# 【RG Q&A Summary】[OSDI'24] DistServe: Disaggregating Prefill and Decoding for Goodput-optimized La...

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/18042967757

---

这里是中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
DistServe: 分解预填充和解码，以实现吞吐量优化的大型语言模型服务

作者：Yinmin Zhong1, Shengyu Liu1, Junda Chen2, Jianbo Hu1, Yibo Zhu3, Xuanzhe Liu1, Xin Jin1, Hao Zhang2
1 北京大学, 2 UC San Diego, 3 StepFun

DistServe架构图

随着大型语言模型（LLM）如GPT-4、Bard和LLaMA的广泛应用，它们正在引领生成式人工智能的重大变革。这些模型不仅重塑了现有的互联网服务，如搜索引擎和个人助手，还推动了如通用聊天机器人和编程助手等全新应用的诞生。然而，这些进步也带来了一个重要的挑战：处理一次完整的LLM查询的时间比传统的搜索查询要长得多。在许多应用中，尤其是对响应时间有严格要求的服务中，延迟是至关重要的。

LLM服务通常分为两个阶段：预填充（Prefill）阶段和解码（Decoding）阶段。预填充阶段负责处理用户的提示输入，生成响应的第一个词（Token）；解码阶段则根据之前生成的词，逐步生成后续的词，直到生成结束标记。这种双阶段的处理方式使得LLM服务的延迟有两个关键指标：生成第一个输出词的计算时间（Time to first token，TTFT）和生成之后每个输出词的计算时间（Time per output token，TPOT）。不同的应用对这两个指标有不同的需求，例如实时聊天机器人更关注低TTFT，而文档摘要则更强调低TPOT。

现有的LLM服务系统通常将预填充和解码阶段合并在同一GPU上处理，通过批处理（Batching）来提高系统的吞吐量。然而，这种做法也带来了显著的问题：首先，预填充和解码阶段之间的强干扰会导致性能下降。预填充步骤通常比解码步骤更耗时，这使得解码阶段的延迟增加；其次，将两个阶段放在同一GPU上会导致资源分配和并行策略无法根据每个阶段的需求独立优化，进一步影响了系统的整体性能。

为了解决这些问题，本文提出了DistServe，一个通过解耦预填充和解码阶段来优化LLM服务性能的系统。DistServe将预填充和解码阶段分配到不同的GPU上，从而消除了这两个阶段之间的干扰。同时，DistServe根据每个阶段的TTFT和TPOT要求，分别优化资源分配和并行策略，以最大化每个GPU的吞吐量。此外，DistServe还根据集群带宽优化预填充和解码阶段的计算位置，以最小化两阶段之间的通信开销。

通过大量的实验评估，DistServe在多个流行的LLM和应用场景下表现出了显著的优势。与现有的先进系统相比，DistServe在满足延迟约束的前提下，可以服务更多的请求或提供更严格的服务水平目标（SLO），同时保持较高的资源利用效率。

本文的贡献包括：
1）识别了现有LLM服务系统中预填充和解码阶段干扰及资源耦合的问题，并提出了解耦两阶段的方案；
2）设计了一种新颖的模型实例调度算法，自动选择最佳的预填充和解码实例部署方案；
3）进行了全面的实验评估，验证了DistServe在实际工作负载中的优越表现。

Q&A

Q1：混部中chunked prefill已经非常优秀，可以同时降低TTFT和TPOT。PD分离相比它在性能上有哪些优势？
A1：Chunked prefill等混部方法优化的是Inference Engine内部的Batching，而PD分离则将优化范围扩展到Prefill-only和Decoding-only两种Inference Engine之间的请求分发和KV Cache调度等。这种方法在分布式环境中可能带来更多的优化空间。

Q2：作者在Placement问题中提到如何将任务放置到物理集群上，这与解决通信开销挑战有何关联？
A2：作者在此提出的Placement问题，是为了引出如何应对“机间通信带宽较低、机内互连带宽较高”的集群环境。DistServe会尽量将同一请求的Prefill实例和Decoding实例放置在同一台机器上，从而充分利用机内互连带宽传输KV Cache，减少因KV Cache传输导致的通信开销。

Q3：作者在文中从Batching和并行策略两个角度展开讨论，文中更多强调了并行策略，Batching优化具体体现在哪些方面？
A3：在batching优化方面，作者提出，解码阶段希望有更大的Batch size。PD分离后，更多的Prefill处理结束后可以发送到一个Decoding实例，进而为解码阶段积累更大的Batch，从而提升性能。

Q4：在PD分离的大场景下，还有哪些值得研究的问题？
A4：与混部不同，PD分离需要为预填充和解码阶段各自分配完整的模型副本，这可能导致GPU内存中的冗余存储。因此，消除这种模型冗余将是未来一个值得深入研究的方向。
