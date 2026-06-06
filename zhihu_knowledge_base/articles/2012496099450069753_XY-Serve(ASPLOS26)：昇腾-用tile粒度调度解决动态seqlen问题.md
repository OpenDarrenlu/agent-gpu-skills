# XY-Serve(ASPLOS26)：昇腾-用tile粒度调度解决动态seqlen问题

**作者**: Arsmart​上海交通大学 计算机科学技术博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/2012496099450069753

---

XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads

优点：方法可以落地到多种真实场景，工程细节想的比较充分；对动态seqlen解决的较好

缺点：文章内容组织的比较乱…

后续研究可能：运行时调度开销会不会比较大？听说昇腾的AIcpu功能强大，如何协调运行时调度的软件需求与硬件能力？

1. 为什么动态性是核心矛盾

图 1：论文把真实 serving 里的动态 workload 拆成几类典型场景。

各种seqlen动态性的来源（输入不一的长度；前缀命中；推测解码的verify；异形mask；chunked prefill与decoding的结合）
实测数据证实（可惜只验证了：输入不一的长度；推测解码的verify）

图 2：真实 workload 中，输入/输出长度分布和 speculation length 都高度动态。

这两张图合起来，前文更适合概括成五类动态场景，而不是只举三个例子：

输入 / 输出长度不均：同一 batch 里可能同时有很短和很长的请求。
前缀命中：APC 会让有效输入长度变成“原始 prompt 长度 - 命中前缀长度”，长度波动进一步放大。
推测解码的 Verify：decode 不再总是 1 token 一步，而是变成动态 specLen 的 verify。
异形 / 不规则 mask：尤其是 tree-based speculation，会把原本规则的 causal mask 变成局部 irregular mask。
Chunked Prefill 与 Decode / Verify 混合：长 prefill 被切 chunk 以后，会和 decode / verify 交错进入同一个调度 budget。

如果再往算子层看，其实还能再抽象出一个更底层的共同结果：dynamic seqlen / dynamic shape。上面五种场景最后都会体现在 tokenNum、kvLen、mask 形状和 stage 组合上；也正因为如此，这篇论文的重点不是为每个场景单独写一套 kernel，而是把这些变化统一投影到 token 和 tile 两层抽象上。

2. Design

先把动态请求整理成 token chunk，再把 token chunk 变成 tile，最后统一执行：

这张图里，真正的 tile 调度层就是 Decomposition -> Reordering。其中 RR 策略就在 Reordering 里：Attention 侧先按 tileSize × kvLen 近似计算量排序，再用 symmetric round-robin 分发到各个 AI Core。只是到了执行层，也不是完全不看语义，因为 Meta-Attention 还要继续处理 mask 和 token-wise KV reuse。

介绍关键词：

APC = Automatic Prefix Caching，意思是“请求进来时，先检查它的前缀有没有已经在 K/V cache 里；命中的前缀不再重算”。
“block 内部的部分命中”意思是：系统按 block 管 K/V cache，但前缀命中的结束位置未必刚好落在 block 边界上，所以会出现“一个 block 里前半段命中、后半段不命中”的情况。
2.1 APC + Token-wise KV Reuse + Copy-on-Write（在线）
这是最前面的入口层设计，先处理前缀命中。
APC（Automatic Prefix Caching）先匹配已有前缀，把命中的部分直接从输入里扣掉，只把未命中的 token 送进后续调度。
这里的难点是：系统物理上仍按 block 管 K/V cache，但前缀命中长度往往不是 block size 的整数倍。
一个具体例子：假设每个 block 有 16 个 token，请求前 30 个 token 命中旧前缀。那么第 1 个 block 的 16 个 token 可以整块复用；第 2 个 block 里只有前 14 个 token 命中，最后 2 个 token 不命中。这就叫“block 内部的部分命中”。
如果只支持整 block 复用，那么这第 2 个 block 就得整块放弃，连前面那 14 个其实已经命中的 token 也要跟着重算。
Copy-on-Write 在这里的意思就是：新建一个 block，把这个旧 block 里仍然有效的那一段 K/V 复制过来，再把后面新产生的 token 接进去。于是系统表面上仍按 block 管理，效果上却能做到接近 token 级复用。
它主要吸收的是 prefix reuse 带来的“有效输入长度不确定”和“历史 kvLen 任意变化”。
2.2 Token-wise Scheduling（在线）
这是把动态请求统一成 token chunk 的关键一步。
APC 之后剩下的 prefill token，以及之前遗留的 decode / verify token，会一起进入调度队列。
调度器按固定 budget length 取 token 组成 chunk，所以一个 chunk 里可以同时混有 Prefill / Decode / Verify。
超长 prefill 会自动切 chunk；同时系统会给 decode 和 speculative token 预留 slot，避免被 prefill 挤压。
它主要吸收的是长度不均、chunked prefill 和 P/D/V 混合执行这几类动态性。
2.3 Dynamic Task Decomposition（在线）
到这里，系统开始从 token 视角切到 tile 视角。
对 Attention，不管 token 来自 Prefill、Decode 还是 Verify，都会先进入 Token-Table，记录 stageID、tokenNum、kvLen、tileSize 等元数据，然后统一逻辑分解成 tile。
对 Linear，不同阶段的 token 会先拼成统一张量，再把动态 GEMM 形状映射到有限的基础 primitive。
这一步只改任务抽象，不改物理数据布局；也就是说，底层存储保持原样，但上层 kernel 看到的是统一的 tile 任务接口。
2.4 Task Reordering（Attention 在线；Linear 离线 + 在线查表）
这一步就是纯粹的 tile 调度 / 负载均衡层。
但 Attention 和 Linear 的做法并不一样。
对 Attention，论文把每个 tile 的负载近似看成 tileSize × kvLen，先按面积从大到小排序，再用 symmetric round-robin 分配到各个 AI Core。这个 RR 是 Attention 侧明确写出来的在线策略。
对 Linear，因为需要支持的 shape 集合小得多，所以论文走的是另一条路：先离线 profiling，找出每种 shape 最合适的跨核分配方式；运行时再根据当前 shape 取出对应的 Task-Table。
这里的 shape lookup 不是查 block size，而是查当前 Linear 算子的 shape，以及它对应的任务分配表。block size 更接近 K/V cache 的存储管理概念，不是这里 lookup 的核心对象。
Attention 当然也要根据当前 workload 调整，但它不是靠“少数 shape 的离线查表”来做，而是通过在线 dynamic tiling、在线 reordering，以及后面的 Meta-Attention 自适应处理 kvLen、mask 和 pipeline。
它不改变计算语义，只负责把已经切出来的 tile 更均匀地铺到硬件上，减少 core 闲置和长尾。
2.5 Meta-Attention（离线 kernel 设计 + 在线执行）
这是 Attention 侧真正的统一执行层，也是全文最核心的设计。
它不再为 Prefill / Decode / Verify 分别写 kernel，而是把它们统一归约成 GEMM -> Softmax -> GEMM 的 Attention primitive。
在 tile 已经切好并重排完之后，Meta-Attention 继续消化那些“仅靠 tile 调度还解决不了”的语义问题：
第一，支持 token-wise KV reuse，也就是能够读取前面 APC + CoW 形成的那种任意历史 kvLen、可能离散的 K/V 视图。
第二，最小化 speculative mask。真正特殊处理的只是不规则 mask 中 specLen × specLen 的那一小块，尤其适合 tree-based speculation。
第三，根据 workload 特征在 2 / 3 / 4-stage pipeline 之间切换，让不同长度、不同阶段的 tile 都能高效执行。
2.6 SmoothGEMM（离线优化 + 在线虚拟填充）
这是 Linear 侧与 Meta-Attention 对应的执行层设计。
Linear 的动态性主要来自 M 维，也就是当前 chunk 里的 token 数，因此本质上也是 dynamic seqlen / dynamic shape 问题。
SmoothGEMM 不去追求“任意 shape 通吃”的万能 GEMM，而是只优化少量固定 shape，再通过 virtual padding + selective read/write 把运行时 shape 映射进去。
所以你可以把它理解成：Meta-Attention 统一处理动态 Attention，SmoothGEMM 统一处理动态 Linear。
