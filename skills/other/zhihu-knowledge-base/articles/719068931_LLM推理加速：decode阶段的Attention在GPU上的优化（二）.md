# LLM推理加速：decode阶段的Attention在GPU上的优化（二）

**作者**: djy123456​阿里巴巴 员工

**原文链接**: https://zhuanlan.zhihu.com/p/719068931

---

背景

随着大语言模型的广泛应用，如何构建低成本高性能的推理服务，越来越成为业界关注的方向。RTP-LLM是阿里巴巴智能引擎团队推出的大模型推理加速引擎，已被广泛应用于阿里内部，积累了一定的实践经验，我们曾在7月26日的文章中，分析了当前MMHA在GPU上的计算：




整个计算过程接近两个MatMul的级联，其中，MHA的实现相当于batch size = B * H 的Batch GEMV，是一个memory bound的问题；而GQA的实现相当于Batch size = B的Batch GEMM，有概率会走入compute bound。如之前的介绍，SoftMax虽然计算量较小，但却限制了MMHA计算的任务划分。在长seq的场景下，我们会采用flash decoding的思路做S上的切分以提升占用；而在中等seq（1k-2k）下，切分S的收益可能不足以覆盖被新增buffer读写的开销，此时的memory load效率又比较低，且我们的平台承接的大量LLM任务的seq都分布在这个区间。

因此，在这篇文章里，我们想以MMHA在A10上的执行为例，简单探讨下在这个区间内，MMHA的计算可以怎么继续调优。

先来看看初始的性能吧：A10, B=1, H=32, D=128, S=1024, 执行时间=58.40us




常量化

当前kernel的瓶颈是memory bound，最重要的修改当然应该是memory load；在修改之前，我们先整理一下代码。除了精简部分代码，去掉一些已经被迭代而不再使用的feature外，我们把计算相对复杂的Rotary Embedding也放到模板参数里展开。我们的推理框架RTP-LLM当前支持多种不同的rotary embedding，从基础的llama，到稍有变化的glm、linear、dynamic ntk, 再到复杂的yarn和时髦的llama3：虽然实际计算中只会走进一种实现，但多样的分支在编译后的代码里体现出较高的寄存器占用和由此降低的occpancy。因此，我们判断将ROPE展开是有潜在收益的。

然而由于MMHA本身也需要根据传入的各种数据类型和head_size生成大量instance，把ROPE展开又把实例数翻了几倍。为此我们把编译的so也做了拆分，避免了符号表溢出的问题。

看看简单的展开带来的性能收益：




再看看这时候的stall情况，大比例的stall long scoreboard还在提示我们memory load一定是这个kernel的瓶颈：




具体的，几乎都是stall在HADD2, 这里的HADD2是将从global中load的KV Cache做计算操作，合理推断stall在这里是因为数据还没有取到。因此优化的关键还是提高load效率。




cp.async

针对最关键的memory bound问题，我们还是先从提高load效率入手。如此前的分析，MMHA基础版本的load效率其实不低。在由计算特点决定的任务分配下，每个thread group按照每个thread 16 Bytes进行连续的256Bytes的load，且每个thread都通过寄存器缓存了部分数据，从而一定程度的overlap数据加载与计算。如果希望进一步提高load效率，最好是让数据缓存更加提前。

但由于MMHA的计算相对复杂，寄存器用量也必须严格控制，提前用寄存器缓存会因加大寄存器用量而影响占用。因此，我们改用shared memory来完成数据的提前缓存。同时，在Ampere及以后，从global到shared memory的数据load效率增强（cp.async）也进一步的提升了load效率。

当然这种类似GEMM中multi stage的优化方式会受到shared memory容量的限制而影响stage深度；此外，shared memory还需要用来存QK dot和output的部分结果：K Cache和V Cache可以复用smem buffer；而V Cache与QK dot的结果不能复用。因此，提前load进smem的优化在中等长度的S上是有显著收益的；随着S变大，需要配合S的切分。

load策略改变带来的提升是非常明显的：




相比之下，stall long scoreboard又缓解了一些：




现在warp stall在ISETP了，仔细分析可以得到，这里表现出来的stall应该还是stall在memcpy async之后的wait。




KV Cache Block指针缓存

一般来说，如果stall在async memcpy，在合并访存且stage深度固定的基础上，除了更好的分配任务以发射足够多的load指令外，我们其实也很难做更多的优化：毕竟load还是主要靠带宽。但MMHA的实现还是略有不同，这里主要的差异是非连续的KV Cache。

在之前的分析中，我们仅介绍了KV Cache连续的存储；然而实际的实现中，我们采用的是类似vLLM的PageAttention的存储方式，这是由于完全连续的KV Cache无法满足实际服务的需求。在连续的KV Cache中，K Cache和V Cache存储的laout都是(B, H_kv, S, D)。这种存储方式要么在一开始就必须按照最长的seq来分配buffer，从而很大概率造成显存的浪费；要么需要在生成过程中随着不断增长的seq分配新的buffer，并将原来的KV Cache拷贝到新的buffer上，而这又会带来延时的明显增长和潜在的显存碎片。PageAttention将KV Cache分成固定大小的Block Cache，每个Block Cache包含固定长度的KV Cache，记这个长度为num_tokens, 那么每个Block的layout依然可以表达为(B, H_kv, s, D); 其中s的取值范围是[0 : num_tokens]。

在这种存储方式下，我们在load KV Cache时，需要先计算对应的KV Cache在那个block内，再计算seq对应在block的地址；这种二次寻址的方式load效率是不如连续KV Cache的。我们调整了计算时循环的顺序，减少了重复KV Cache Block指针的重复load，并且提前load了KV Cache Block指针，这也有助于kernel性能的提升：




其他

最后，我们在优化过程中还穿插了一些调优小技巧:

cache hint

PTX ISA提供了一系列指令可以相对精细的控制cache行为，如常用的.ca(在各level cache)和.cg(bypass L1，仅cache在L2)，根据load的数据是否会被多次用到来决定哪一种策略。进一步的，在MMHA计算中，KV Cache的每个数实际上仅访问一次，除了bypass L1外，我们可以将evict_first作为cache hint进一步提高load效率。

forceinline

forceinline强制编译器内联函数，会更有助于编译器完成指令重排，并进一步的优化指令，常见的如将FMUL+FADD优化成FFMA等。

这些小trick单独的提升都比较有限，因此本文就不将它们作为一个优化点详细展开。

最后，我们可以看到的是，经过一系列的优化探索，同样的kernel，执行时间从58.4us降低到41.64us；memory 效率也提升到69.5%（当然这是一个统计值）。

展望

在这篇文章里我们仅以A10为例，介绍了特定seq下MMHA的优化可以怎么展开。然而，在不同的卡上，不同的Seq下，或者是MHA和GQA的区别下，kernel的性能瓶颈都是不太一样的，相应的，优化策略也应该有差异，比如GQA应该改变任务划分，A100更应该从提高占用的角度出发，H100可能需要想想怎么利用好Hopper的新feature等等。

我们的优化尝试会不断进行，在未来，我们将继续探索和分享更多关于LLM的优化策略和实践经验，并在RTP-LLM项目中分享给大家，欢迎共建交流。
