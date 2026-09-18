# llm decode加速器架构详解（一）Cerebras Wafer-Scale Engine

**作者**: aiiiiii爱丁堡大学博士在读/微架构设计/计算机体系结构/MLsys

**原文链接**: https://zhuanlan.zhihu.com/p/2026781196298757013

---

1·工艺

Wafer-scale computing 的关键并不在于“把芯片做得更大”，而在于如何在现有半导体制造约束下，把“本来不应该成为一颗芯片的整片晶圆”真正做成一颗可工作的芯片。其第一道约束来自光刻 reticle limit：现代曝光系统单次可曝光面积约为 26 mm × 33 mm，因此 wafer-scale 设计必须采用跨 reticle 的 field stitching，而不是传统单曝光场成芯片的方式。其第二道约束来自良率：面积一旦扩大到整片晶圆，制造缺陷将不再是偶发问题，而是必然事件，因此体系结构必须原生支持细粒度冗余与故障绕行。其第三道约束则来自封装与系统：如此巨大的芯片必须面对远高于常规处理器的供电、测试和散热压力。也正因如此，wafer-scale 从来不是单纯的“更大 die”，而是一种制造、架构与封装协同优化的结果。

基于此芯片工艺，Cerebras 并不是只是在做一代代更大的芯片，而是在逐步把单块超大晶圆芯片扩展成系统的完整计算平台。WSE 从 CS-1/CS-2 时代开始，首先被拿来做的是大模型预训练、继续预训练和微调。但现在随着大模型的爆火，Cerebras 当前官方 inference 页面强调的是 production-scale inference、极高 tokens/s，以及即时响应体验；2024 年发布 Cerebras Inference 时，也重点宣传了 Llama 3.1 8B 和 70B 的高输出速度，把它包装成“world’s fastest AI inference”

根据公开数据，我先比较了几款产品的单位面积的计算吞吐量和sram容量，体现wafer工艺的强大

选这些baseline是因为只有这些的面积数据是公开的

当然，这些面积优势也和它的架构设计有关。GPU 往往要花大量面积在 HBM 控制器、PHY、片外高速 I/O、大层级 cache、复杂控制和通用执行支持上；而 WSE 的设计目标更集中，更多面积可以直接拿去放计算单元 + 本地 SRAM + 简单规则互连。所以它看起来不是“晶体管密度更高”，而是“真正分配给有效 AI 数据通路的面积比例更高”。当然，代价是很多复杂性被转移成了编程模型与数据流组织，并且热、供电、封装和测试都更难，经常有小计算单元失效等等问题

2·性能分析

对 wafer-scale 这类超大单片来说，外部 I/O、封装、供电和散热的扩展都更困难，因而它不适合继续沿着传统处理器那种“不断堆片外带宽”的路线扩张。更自然的方向，是把更多数据复用留在片上，用更高的片上 SRAM 带宽、片上 fabric 带宽，以及匹配的数据流和调度，来减少对片外带宽的依赖。因此，传统由片外 DRAM 带宽主导的 compute-bound / memory-bound 二分法被明显弱化了；由于芯片面积、片上 SRAM 容量和片上带宽都被极大放大，系统瓶颈更多转移为 compute 能否被充分喂满，以及数据在片上和跨系统之间如何高效通信，因此其主要权衡更接近 compute-bound 与 communication-bound 之间的权衡。

各产品参数

从参数上看也符合以上论述观点。WSE-1 到 WSE-3 的演进已经从是在整片晶圆级面积上持续堆高计算资源、片上 SRAM 和片上互连能力。WSE-1 已经提供 40 万个 AI core 和 18 GB 片上 SRAM；到 WSE-2 增长到 85 万个 core 和 40 GB SRAM；WSE-3 则达到 90 万个 core、44 GB 片上 SRAM，以及 125 PFLOPS 峰值算力。与此同时，Cerebras 官方长期强调其片上带宽和片上 fabric 带宽也处于极端高水平：WSE-1 为 9 PB/s 片上内存带宽和 100 Pb/s fabric，WSE-2 提升到 20 PB/s 和 220 Pb/s，WSE-3 官方当前产品页给出的量级则约为 21 PB/s 和 214 Pb/s。

因为其巨大的片上内存，他们设计一种名为weight streaming 的方法，把参数容量与片上存储容量解耦，以此和一般的GPU/TPU调度区别开来达到更灵活的计算-访存调度：在传统 GPU 上，算力和显存通常是绑死的：你有多少 GPU，才有多少总显存；模型只要比单卡显存大一点点，就不得不加更多 GPU，并引入模型并行、流水并行之类的复杂分布式机制。Cerebras 官方专门拿这个做对比：如果 GPU 有 80 GB 显存，而模型需要 82 GB，你就得上更多 GPU；但在 weight streaming 架构里，参数存在外部 MemoryX，内存容量可以独立扩展，不必因为“想多一点参数容量”就被迫同步增加一堆计算芯片。因此计算放在 WSE 上，参数放在外部 MemoryX 里，两者可以分别扩。

关于weight streaming，软件调度方法上GPU也会有类似的调度，那么体现不同的地方就是硬件设计MemoryX 上。关于片外存储，Cerebras 在官方博客里把 MemoryX 描述为“dedicated, external memory device”，并明确说它“uses flash and DRAM”，但没有给出 DRAM 代际或颗粒/通道配置。MemoryX 提供的是 central weight storage，把模型参数存放在独立内存系统里，再高效地流式送到 CS-2/CS-3；CS-3 还可配置高达 1,200 TB 的外部内存容量。从官方文档为数不多的信息可以推断出，MemoryX 更像一个分层存储 + 调度控制器系统。 Flash 更像是高容量的权重持久化层。DRAM 很可能承担近端缓冲 / staging / 重排 / 流式队列的角色，用来把即将送入 WSE 的 layer weights 提前准备好。




为了彻底实现参数容量与片上存储容量解耦，还需要介绍一下SwarmX。相比于NVLink 这种更通用的高带宽 GPU/CPU 互连，SwarmX相当于是专门为 Cerebras 的 weight streaming 训练而设计的 broadcast/reduce fabric。SwarmX 高度特化主要处理层权重向下广播，以及梯度向上归约。NVLink 面向的是更通用的 GPU 远端内存访问、张量并行流量、激活/参数/KV/collective 等多种 GPU 之间的数据交换。

我在这里第一个想到的是专用互联缺乏灵活性，moe网络映射可能不太行，但忽然想到他们应该都是在单片上推理的，又顺带研究了一下他们的片上互联。虽然他们的片上互联灵活性不是无限 all-to-all，但如果能把 MoE 组织成局部化 dispatch 问题，片上互联还是可以解决的，学术界和工业界已经有很多类似的方法了。

3·微架构设计

借一下EPCC的图

最后看计算微架构方面：
单个 WSE-3 core 的组成：

48 kB SRAM per core
512 B local cache per core
8-way SIMD for 16b data (FP/BF16)
16-way SIMD for 8b data (Fixed/INT8)

每个 die 内部是 2D mesh fabric；互连能 跨 reticle/die boundary 延伸且保持 full performance；硬件内建 redundancy 用来绕开失效位置；软件看到的是一个统一的 2D mesh。WSE-3 不是传统 GPU/TPU 那种少量大核 + 共享 cache/hbm 的组织，而是极大量小核、每核私有 SRAM、片上数据流调度统一。一开始我觉得比较奇怪，因为我觉得设置更大的脉动阵列式的加速器，以增强数据复用性，使得可以更加利用片上丰富的存储资源。

后面猜了几个原因，第一个是Cerebras 很在意非线性、小算子和稀疏，不想把体系绑死在矩阵核上，Cerebras 官方说一开始就是想做一个适合 fine-grained, dynamic sparsity in neural networks 的 core。学术界也有用它来做stencil等科学计算。第二个是散热，WSE-3 是一块 46,255 mm²、约 4 trillion transistors 的 wafer-scale 芯片，单系统功率密度非常高。官方专门强调，engine block 之所以要把电力“straight into the face of the wafer”，就是因为传统封装方式达不到所需功率密度；同样地，传统风冷也很难在这么大的单片硅面积上实现足够均匀、足够低热阻的散热。用脉动阵列这样的高密度计算单元很容易造成热点的聚集。第三是良率，之前说片上很容易有坏的地方，阵列做得过大、单体太集中、缺陷绕不过去，容错和重映射通常会更麻烦。

最后展示实际推理场景，虽然亚马逊已经尝试只把wse当作decode加速器了（还是太富了，归不得最近股价猛涨），但目前还是prefill-decode一起来算，体现端到端加速，不然prefill的kv cache搬运这些还要时间。根据wafer-llm，GEMM 对应 prefill，GEMV 对应 decode（顺便吐槽一句，这么贵的芯片一次只跑1batch还是太浪费了，好像理解前面亚马逊多batch decode推理了）GEMM两维都切，使用compute-shift loop一边移动一边算。GEMV：一维切分 + 一维复制，L 维沿 x-axis 复制，local compute + allreduce 用片上高通信带宽来掩盖片外带宽不足，缓解memory-bound问题。
