# 大模型训练：Megatron-Kwai中的内存优化

**作者**: Lin Zhang​香港科技大学 计算机科学技术博士

**原文链接**: https://zhuanlan.zhihu.com/p/710296768

---

在上一篇文章中，我们介绍了大模型训练中的通信优化技术。

Lin Zhang：大模型训练：Megatron-Core中的通信优化
289 赞同 · 19 评论 文章

除了通信优化，对于大模型训练来说，内存优化也是非常重要的问题。在这篇文章里，我们会结合快手发表在今年ATC上的工作（Megatron-Kwai[1]），简单介绍一下内存优化技术。

背景

首先，根据生命周期的长短，我们将内存开销分为：长内存、中内存、和短内存。

长内存在训练任务开始以后就不会消失，包括模型参数、模型梯度、和优化器参数（例如Adam中的一阶矩估计和二阶矩估计）。中内存会跨越不同阶段，例如由前向传递得到的激活值会保存到后向传递用于计算梯度。短内存只跨越几个算子，或者仅仅在算子计算过程中使用（workspace内存）。

在PyTorch框架中，以上内存开销可以通过memory_allocated查看[2]。除此以外，我们还有CUDA context、NCCL buffer等开销。在这篇文章中，我们主要分析长内存开销和激活值开销。

LLaMA模型

以LLaMA模型为例，大模型结构包含一个embedding词表，L个transformer层，以及一个分类头。对于每一层transformer layer，我们统计所有线性层的参数，可以得到模型参数为：

𝑃
=
(
2
+
2
𝑔
𝑎
+
3
𝐻
ℎ
)
ℎ
2
 [3]，其中， 
2
ℎ
2
 对应query和output层， 
(
2
𝑔
/
𝑎
)
ℎ
2
 对应key和value层（考虑grouped query attention）， 
3
𝐻
ℎ
 对应FFN中的参数。整个LLaMA的模型参数约为 
𝐿
𝑃
+
2
ℎ
𝑉
 [4]。

为了能放下大模型，我们采用两种模型并行策略：张量并行和流水线并行。假设张量并行的大小为t，流水线并行的大小为p，每个设备上的模型参数为：

𝑃
(
𝑡
,
𝑝
)
=
(
𝐿
𝑝
⋅
𝑃
+
ℎ
𝑉
⋅
1
𝑟
𝑝
𝑝
=
0
+
ℎ
𝑉
⋅
1
𝑟
𝑝
𝑝
=
𝑝
−
1
)
/
𝑡
 .

长内存开销

对于大模型训练，我们一般采用混合精度的方式，其中模型参数为BF16格式，梯度累加后的格式为FP32，主模型参数（main weight）和优化器状态参数为FP32格式。

于是，我们可以得到长内存的开销为 18 P(t, p) . 此外，我们可以在数据并行维度（包括文本并行）采用ZeRO-1算法，将长内存的开销进一步降低至 (6+\frac{12}{cd})P(t,p)[5] .

激活值开销

对于激活值开销，假设激活值的格式为BF16，每层transformer layer，我们需要存储两个RMSNorm层的输入（ 4bsh[6]），Attention模块以及FFN模块的输入（ 4bsh ），Flash-Attention的输入和输出（4bsh+4\frac{g}{a}bsh ），SwiGLU的输入和输出（ 8bsH ），总共 M_a=(12+\frac{4g}{a}+\frac{8H}{h})bsh 的激活值开销。

考虑到序列并行（SP）和文本并行（CP），激活值开销为 \frac{M_a}{tc} . 此外，对于Interleaved-1F1B流水线并行而言，激活值峰值出现在第一个流水线设备上，其开销为 (vp+p-1)l \cdot\frac{M_a}{tc}[7] .

内存优化技术

针对激活值开销，在Megatron-Kwai中，主要介绍了两种内存优化手段：重计算和CPU卸载。

首先，重计算技术在前向传递的时候不保存激活值，而在反向传递前重新进行一次前向计算，是一种用时间换空间的手段。在Megatron-LM中，支持全部的重计算和层级的选择性重计算，也就是说，我们可以对所有的transformer layers或者部分的transformer layers进行重计算。

Megatron-Kwai采用算子级的选择性重计算。也就是说，我们可以只对transformer layer中计算轻量的算子进行重计算，这包括两个RMSNorm层，激活函数中的SiLU和Mul。通过以上选择性重计算，我们只需要增加很小的计算开销，就可以将激活值内存降低至 (8+\frac{4g}{a}+\frac{4H}{h})bsh 。在LLaMA-2中，这意味着减少44%的激活值开销。

其次，我们可以使用CPU卸载的方法继续降低激活值内存。在流水线并行中，我们将每个model chunk中的激活值卸载到CPU。在反向传递阶段，我们提前一步将CPU上的激活值加载到GPU中。

在这个过程中，激活值的加载/卸载可以和前向/反向计算overlap。Megatron-Kwai允许调整卸载的比例（offload_ratio），选择将部分激活值卸载到CPU上。如果数据卸载的时间小于计算的时间，我们可以将这部分开销全部隐藏。

总结

对于内存优化，主要的手段包括并行技术、重计算、以及CPU卸载。一方面，我们使用张量并行和流水线并行来降低模型参数对应的长内存开销。另一方面，我们使用序列并行和文本并行来降低模型输入对应的激活值开销。最后，合理使用重计算和CPU卸载技术，可以用很小的成本来降低激活值内存，这有助于我们选择更加高效的并行配置来训练大模型。

参考
^Accelerating the Training of Large Language Models using Efficient Activation Rematerialization and Optimal Hybrid Parallelism,  https://www.usenix.org/conference/atc24/presentation/yuan
^PyTorch还有active内存和reserved内存的概念，这和内存复用机制相关，我们先忽略。
^h为hidden size，H为intermediate size，a是attention head number，g是kv head number
^L为模型层数，V为词表大小
^d为data parallel size，c为context parallel size
^b为batch size，s为sequence length
^v为model chunk number，l为transformer layer number per-chunk
