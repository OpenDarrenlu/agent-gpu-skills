# [大模型推理系统] Deepseek inference 并行方式思考

**作者**: JoeNomad​​新南威尔士大学 信息技术硕士

**原文链接**: https://zhuanlan.zhihu.com/p/26639126208

---

​
目录
收起
Update
切分变化
卡数变化
前言
Prologue
TL,DR
MAIN
方法调研
是否存在更优的解法
Epilogue
Update
本文更新记录		
2.27	初版	
3.3	follow deepseek新发的内容做一些跟进	
		

周末deepseek官方账号发了他们的infra架构的一部分信息，其中跟本文相关的部分有所变动。切分方式退化成DP+EP，并且卡数变成了144，以此为依据，我们再来推理盘一盘这个逻辑。

Cites：DeepSeek：DeepSeek-V3 / R1 推理系统概览

切分变化


首先，并行策略是一个约束问题，并行策略和SLO是强相关的，没有最优的切分方法，只有在SLO约束下的最优解。当然，如果面向吞吐，是有全局最优的，但是在serving场景下，抛开SLO谈吞吐是毫无意义的。
先说为什么TP和SP没有了，我依旧秉持我在第一版的推论，SP是为了抵消kvcache的replica而存在的，TP则是为了dense部分的latency。那么原先的系统对于单条request的latency来说更低，现在变化了意味着说，我适当牺牲用户的体验(即SLO)，换取更大的吞吐。

卡数变化

Q1: 为什么Decoding变成了144张卡？

这个数字非常刺眼，肯定不是一个拍脑袋想出来的数值，他的道理是什么呢？碰巧我之前有看过deepseek集群架构那篇论文，我发现有些数字能对的上，我大胆地推测一波。

Deepseek发在SC的那篇papar

是的，我们可以看到，这个交换机的port是40个，和一般的高性能组网的64port的交换机不一样，在spine-leaf这样的集群架构下，有一半的port要连上层交换机，这个配置每个leaf最多接入20个nodes，其中还有一些storage node，那么这就能对上了，每个leaf 18个计算节点 + 2 个存储节点正好是144张卡，这样每个gpu都在同一个leaf下，他们的通信是最快的。

Q2: prefill的变化是什么

prefill从DP8变到DP32, 以我的经验，这对性能影响很大，而且DP32我觉得更难做prefill的负载均衡，DP域接EP域的时候要做一个allgather，如果负载差的比较多会有很大的气泡，prefill阶段attn和token数是二次关系，相比decode来说负载不均衡的影响更严重。当然，ds目前流量这么大，负载均衡会相对好做一些，不过还是respect，非常非常solid的工作。

P.S. 最终版的Deepseek逻辑和sglang的DP attn一样了，只用关注EP高性能实现了， 复现难度大大降低，哦豁！

前言

最近一段时间被高密度的开源盛宴席卷，这两天又品味了一下deepseek V3 paper里面的inference细节，有些地方初看就漏掉了，再深入看的时候觉得有点confusing，于是推理了一波，在此记录。

Prologue

先在这里小结一下原文的3.4节，特别的，后文中都以MLA attention作为讨论基础，计算逻辑采用absorb的形式

Prefill instance

并行方式: 32GPU, 4TP + 4SP + 8DP + 32EP
做了MoE expert Redundancy, 令每路EP上都会多存一些expert ，相当于router的时候有>1的路径可以选择
动态冗余机制，会根据负载动态地换redundancy experts，interval = 10mins

Decode instance

并行方式: 320GPU, 4TP + 4SP + 80 DP +320 EP
MoE阶段256个expert均分在256张卡上，剩下64张卡每张卡上放shared expert + redundancy experts
每个expert最多放256个token，对于Decode来说是memorybound，分配少的sm对latency的decay不大，用2-micro batch和attn做overlap

这里令我比较confuse的点是，在此之前，我理解SP和TP不会放在同一张卡上做，所以最终用到的GPU数是SP * TP * DP * PP，但明显从论文中看并非如此。

TL,DR

核心是MLA，推理的阶段是MQA，所有head共用同一个kv，TP按head切，那么只做tp会造成kvcache的replica, decoding instance需要大batch来提高MFU，SP叠加在TP后抵消replica的问题。同样的，DP也可以规避kvcache replica的问题，在SGlang中也有dp attn这个特性用dp取代tp抵消kvcache replica。

MAIN
方法调研

对目前SP的方法做了一些调研，核心还是online softmax，可以理解为flash attention的多卡版

Ring KV attention
这和训练的方法一样，将kv切成多分，分别放在不同的卡上，每次通过传KV块迭代的计算出全局softmax


这里借用loongserve里面的图解

Hydragon Decoding
这里是说，SP在attention计算前拿到的是全量的Q(在decoding阶段Q的length是1)，每个rank上的kv是常驻的，每个rank并行地计算local attention之后再allreduce进行聚合，我们来详细拆解一下这个方法中每个步骤。

先回顾一下Attn公式和online softmax的因式分解

o_i = \sum_{j=1}^{m} \frac{\exp\left( \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_i \right)}{\sum_{l=1}^{m} \exp\left( \frac{q_i \cdot k_l}{\sqrt{d_k}} - m_i \right)} v_j

y = e^{x - m_{global}} = e^{x - m_{local} + m_{local} - m_{global}} = e^{x - m_{local} } * e^{ m_{local} - m_{global}}

Step 1:

首先每个rank上先做local的attention，同时拿到lse和局部max，假设SP = 4
O_{i} = Attn(Q_{i}) \ \ \ Q_{i} ∈ \{Q_{1} , Q_{2} , Q_{3} , Q_{4}\}\\ lse = log(O_{denominator}) \\ max_{local} = max(Qi)

Step 2:

allreduce拿到全局的max，把local O的分子分母都做online的变换

denominator = exp(lse - log(\exp(m_{global}))) \\ factor = exp(m_{local} - m_{global}) \\ numerator = \exp\left( \frac{qk}{\sqrt{d_k}} - m_{local} \right) * factor

Step 3:

对分子分母都做allreduce，再做除法，拿到全局的attention结果。

结合Deepseek的模型结构画了一个示意图

1DPx4TPx4SP tree decoding

Ring Query

这个出自meta的论文，原文中给了个计算公式来说明用Ring KV和Ring Query的边界条件，其实就是求了通信量的比值。方法其实也非常容易理解，直接给原文的图。

是否存在更优的解法

其实单看这两种方法其实差不多，他们在算attention的时候通信相差无几。但是结合decode layer来看这个问题就不同了。如果用ring Q的方法，可以更好的结合前面linear的TP，可以将hydragon的方法中那个allgather省去，并且少1/2的activation，但是问题是后面多了一个all2all和local accumulate kernel，这些是串行的无法overlap。

先叠个甲，以下是一些不成熟的，未经验证过的想法，如果有问题还请轻喷。

对Ring Q的方法稍作改动，我愿称之为zegzag ring Q，ring的过程中每个iteration将自己rank的Q和O都发往下一个rank，下一个rank的会计算local O并和recv的O迭代的更新，在最后一个iteration计算完的时候就已经拿到完整的attention了。此时每个rank上拿到的是不同head的attention结果，叠加上col parallel的v up proj和row parallel的o proj，只需要在最后做一次allreduce。如果把最后v,o的权重重排一下，那么最后一个rank甚至不需要传回去，节约尾部的通信，中间如果能够overlap的话相当于没有通信开销。过程如下图

做了一些非常粗的理论计算

以H800为例：
Nvlink 400GB/s Bi
BF16: 1513TFLOPS

compute latency = 4bhsc / compute_flopS
transfer latency = 2bhc / (tp_size * Bandwidth)
Overlap .s.t compute_latency > transfer latency
2s(tp_size * Bandwidth) > compute_flopS
--> 400G * tp_size * 2s > 1513 * 1e3
--> s > 472

这里只算了dense，没算softmax，实际应该会小不少，结论是在prefix超过一个相对不太大的阈值时能overlap

Epilogue

SGlang DP attention:

SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs | LMSYS Org

Reference:

https://arxiv.org/abs/2408.14158

朱小霖：ring attention + flash attention：超长上下文之路

手抓饼熊：大模型推理序列并行

LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism

Hydragen: High-Throughput LLM Inference with Shared Prefixes

Context Parallelism for Scalable Million-Token Inference
