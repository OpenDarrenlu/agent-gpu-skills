# 大模型训练：Megatron-Core中的通信优化

**作者**: Lin Zhang​香港科技大学 计算机科学技术博士

**原文链接**: https://zhuanlan.zhihu.com/p/694877232

---

​
目录
收起
Megatron-LM和3D并行
Megatron-Core和通信优化
总结

提到大模型的训练框架，相信大家对Megatron-LM应该都比较熟悉。知乎上有不少文章介绍Megatron-LM对应的三篇论文，或者是源码实现，这些内容对于初学者的帮助很大。

然而，和之前的文章不同。这篇文章的目的不是科普扫盲，也不是源码解析，而是从研究者的视角，简单地聊一聊Megatron-Core中的通信优化设计。

分布式训练中的通信优化是我博士期间的老本行。在去年的一个学术会议上，我和一个学弟聊到了如何在Megatron-LM上做通信优化。尽管这样的想法十分常见，当时的代码库中却没有考虑这些。

我们当时觉得可能是英伟达的集群带宽太高，并不太在意通信优化。没想到过了大概一年，这些想法已经全部在Megatron-Core中实现[1]。有趣的是，MegaScale同期也发表了论文，提到了类似的做法[2]。

总的来说，在框架层面，通信优化能做的事情不多，真正有效的手段也就那么几样（overlapping, tensor fusion, tensor partitioning）。以至于通信优化本身的工作，发论文老是会被审稿人批评创新性不足。

言归正传，让我们先简单回顾一下3D并行的基础知识。

Megatron-LM和3D并行

Megatron-LM是一个为transformer大模型设计的分布式训练框架。除了传统数据并行的维度，Megatron-LM实现了流水线并行和张量并行，用于切分大模型的层数和权重，解决内存墙的问题。

在Megatron-LM的三篇文章中，第一篇介绍了张量并行的技术[3]，对于transformer模型而言，主要是attention中的注意力头，和ffn中的隐藏层可以并行处理，在具体实现上则是设计了column-wise线性层和row-wise线性层的抽象。对于张量并行而言，每次前向传递和反向传递的过程中，attention和ffn都需要all-reduce通信。此外，张量并行一般还会对embedding和cross entropy进行切分。

第二篇文章介绍了3D并行的技术[4]，也就是通过数据并行+张量并行+流水线并行，训练千亿乃至万亿规模的大模型。文章分析了3D并行的最佳切分方式，其中最重要的一点，就是我们需要将通信开销最大的张量并行限制在单个节点内。另外，在流水线并行的部分，文章提出了interleaved 1F1B，采取交错式的模型切分方案，来降低流水线并行中存在的气泡开销（bubble）。

当然，对于更大规模的训练来说，流水线气泡的问题依旧存在。简单来说，由于训练过程中的global batch size受限，随着数据并行和流水线并行的规模持续扩大，我们无法保证micro-batch数量远大于流水线并行。对于这个问题，从硬件的角度，我们可以通过设计超节点增大张量并行的规模；或者从算法的角度，我们可以使用例如LAMB优化器来增大global batch size。

第三篇文章，针对张量并行中激活值内存浪费的问题，提出了序列并行的技术[5]。也就是说，我们可以将输入按照序列的维度进行切分，在需要进行相应的attention或者ffn计算的时候再重新聚合。在具体的实现上，我们可以将all-reduce拆分成all-gather和reduce-scatter，实现all-gather+column-wise线性层、以及row-wise线性层+reduce-scatter的抽象。其中，all-gather+线性层的反向传递可以进行通信优化，这点我们之后再展开。

值得注意的是，这篇文章详细分析了GPT训练中激活值内存的分布。当然，这和当前Llama模型的情况有些不同，主要包括Llama中没有使用dropout，以及SwiGLU比GeLU需要更多的激活值开销。同时，对于Huggingface中的模型实现，非融合算子版本的layernorm/rmsnorm，以及cross entropy的实际内存开销也会大于文章中的理论分析。最后，文章里提到对self-attention采取选择性重计算的方案，当前基本上已经被flash-attention所取代。

Megatron-Core和通信优化

Megatron-Core中的通信优化包括数据并行、张量并行、和流水线并行。

首先是数据并行，DeepSpeed中的ZeRO系列可以在数据并行的维度上对模型、梯度、和优化器参数进行切分[6]。其中，ZeRO-1将原本数据并行中的all-reduce梯度操作切分成reduce-scatter梯度+all-gather参数，这样做的好处是优化器更新可以在切分后的参数量上进行，从而减少了内存开销。

Megatron-Core支持ZeRO-1形式的数据并行，即在DDP中实现reduce-scatter反向传递得到的梯度，在distributed optimizer中实现all-gather优化器更新后的模型参数。一般来说，降低数据并行的通信开销有两个常用的手段。首先，我们可以通过梯度累加，比如说流水线并行中的micro-batching，来降低数据并行中通信的比例，这一点对ZeRO-1依旧适用。

此外，我们可以将通信和计算进行隐藏。和传统的DDP相比，ZeRO-1允许将reduce-scatter和反向传递进行隐藏，将all-gather和前向传递进行隐藏。同时，为了提高通信效率，我们需要将小参数进行合并（即tensor fusion[7]）。我之前有篇论文做的就是这方面的工作，其中通信优化部分的实现方式和现在Megatron-Core里的实现基本一样。当然，类似的技巧对于ZeRO-3来说依旧适用，例如PyTorch中的FSDP也实现了类似的通信隐藏[8]。

其次是张量并行，前面提到，Megatron-LM对于all-gather+线性层的反向传递进行了通信优化。为了节省内存开销，Megatron-LM只存了all-gather之前的输入，所以在反向传递阶段，我们需要all-gather保存的输入用来计算权重的梯度，另外我们还需要对于计算得到的输入的梯度进行reduce-scatter。于是，我们可以将all-gather和计算输入的梯度进行隐藏，然后将reduce-scatter和计算权重的梯度进行隐藏。这种通信和计算无依赖关系的隐藏，又叫做bulk overlap。

除了以上例子，Megatron-LM并没有对其他操作进行通信优化，包括前向传递中的all-gather+矩阵乘，和矩阵乘+reduce-scatter，因为这两个操作中的计算和通信存在依赖关系，无法直接进行隐藏。针对这种情况，我们可以使用tensor partitioning的技术，将一个大的矩阵乘法和集合通信操作，拆分成一系列小的矩阵乘法和集合通信操作，然后对更加细粒度的计算和通信进行流水线式的隐藏。当然，将一个tensor切分得太小，反而会影响实际性能，一般来说切成4份是比较常见的配置。

除了直接切分张量以外，我们还可以将集合通信操作拆分成一系列的p2p通信操作，例如all-gather操作可以拆分成ring-based send/recv通信[9]，其中拆分后的通信和计算同样可以进行隐藏。

具体实现上，Megatron-Core调用了Transformer Engine中的线性层，支持bulk overlap通信隐藏，以及张量切分或者p2p切分方式的通信隐藏。同时，为了降低通信和计算之间存在的干扰，TE使用userbuffer进行张量并行的进程间通信。

最后是流水线并行，流水线并行中需要用到大量的send/recv操作，实现起来非常繁琐。为此，Megatron-LM设计了一系列的p2p通信接口，用来打包send-next, recv-prev, send-prev, recv-next操作，防止p2p通信因为执行顺序不同导致的死锁问题。

Megatron-Core支持1F1B和interleaved 1F1B这两种流水线并行方案，并针对interleaved 1F1B进行了通信隐藏优化。一方面，因为interleaved 1F1B在大模型训练中更为常用，同时其通信开销要远远大于普通的1F1B方案。另一方面，对于1F1B而言，哪怕使用异步的send/recv操作，其实也没有太多的通信优化空间[10]。而对于interleaved 1F1B来说，在steady阶段，我们可以将forward-send-forward-recv通信和反向传递的计算隐藏，然后将backward-send-backward-recv通信和前向传递的计算隐藏。

总结

对于大模型训练来说，集群的有效算力 = 单卡的有效算力 x 集群规模 x 线性度 x 可靠性。其中，Megatron-Core将3D并行中的通信和计算进行隐藏，也就是尽可能提高大模型训练的线性度。

在当前大模型结构逐步收敛的背景下，在训练框架的层面上，大模型系统优化（包括通信优化）并没有剩下太多空间，其中针对MoE的优化显然成了大家（包括Megatron团队）今年的研究重点。

参考
^Megatron-Core,  https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#megatron-core
^MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs,  https://arxiv.org/abs/2402.15627
^Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism,  https://arxiv.org/abs/1909.08053
^Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM,  https://arxiv.org/abs/2104.04473
^Reducing Activation Recomputation in Large Transformer Models,  https://arxiv.org/abs/2205.05198
^ZeRO: Memory Optimizations Toward Training Trillion Parameter Models,  https://arxiv.org/abs/1910.02054
^Horovod: fast and easy distributed deep learning in TensorFlow,  https://arxiv.org/abs/1802.05799
^PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel,  https://arxiv.org/abs/2304.11277
^Scaling Vision Transformers to 22 Billion Parameters, Figure 3,  https://arxiv.org/abs/2302.05442
^On Optimizing the Communication of Model Parallelism, Figure 4,  https://arxiv.org/abs/2211.05322
