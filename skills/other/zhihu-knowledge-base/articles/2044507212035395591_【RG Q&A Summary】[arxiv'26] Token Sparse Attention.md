# 【RG Q&A Summary】[arxiv'26] Token Sparse Attention

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/2044507212035395591

---

这篇文章来自中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
词元级稀疏注意力：基于交错式词元选择的高效长上下文推理

作者：Dongwon Jo1, Beomseok Kang1, Jiwon Song1, Jae-Joon Kim1

1 Department of Electrical and Computer Engineering, Seoul National University, Seoul, South Korea

大语言模型在处理长上下文（如 128K 以上）时，自注意力机制的计算复杂度随序列长度呈平方增长，导致推理延迟显著增加。现有稀疏注意力方法（如 StreamingLLM、H2O、FlexPrefill、Minference 等）通过固定策略或学习到的模块丢弃部分 token，但这些方法往往存在两个局限：一是永久性丢弃 token，导致信息无法被后续层重新利用；二是对各层和各注意力头采用统一的稀疏策略，缺乏动态适应性，难以兼顾精度与效率。

本文提出 Token Sparse Attention，一种可逆的 token 级稀疏化机制。核心思想是：在每一注意力头中，通过轻量级的评分函数（如基于 QK 内积的近似）评估所有 token 的重要性，选择重要性最高的 top-k token 组成压缩后的 Q、K、V 子集，在该子集上进行标准注意力计算；然后通过解压缩（将子集注意力结果映射回原始序列的对应位置，缺失位置补零）恢复完整序列的输出。这种“压缩-解压缩”设计允许每个头动态选择不同数量的 token，并且不同层可以依据层间表示漂移指标选择是否进行稀疏化，从而实现 head 级和 layer 级自适应。

实验在 128K 上下文长度的长文本任务（如文档理解、多跳问答）上进行评估。Token Sparse Attention 在保证精度下降小于 1% 的前提下，实现了高达 ×3.23 的注意力加速。当与 FlexPrefill 等方法组合时，可将加速比从 ×2.44 提升至 ×2.76，且额外计算开销低于总注意力延迟的 11%。该方法兼容 FlashAttention 等密集注意力实现，展现出优越的精度-速度权衡。

本文的主要贡献包括：

1）提出压缩-解压缩的 token 级稀疏注意力机制：通过可逆的动态 token 选择，避免永久性信息丢失，且每个注意力头独立决策，支持 head 级和 layer 级自适应稀疏化。

2）引入动态 token 覆盖率与稀疏层选择策略：基于累计覆盖阈值自适应调整每层保留的 token 数量，并利用层间表示漂移指标识别适合稀疏化的稳定层，在保证精度的同时最大化计算效率。

3）展示与其他稀疏方法互补的通用加速能力：Token Sparse Attention 可叠加于现有方法（如 FlexPrefill）之上，进一步提升加速比（如从 ×2.44 到 ×2.76），且额外开销极小，验证了其作为通用加速模块的实用性和有效性。

Q&A

Q：\tau全局设定后，每个layer都是相同的，那么动态性从何而来呢？

A：虽然全局\tau是一个固定的覆盖率阈值，但动态性并非来自\tau本身的变化，而是来源于每层与每个注意力头所面临的token重要性分布不同，以及稀疏化层的选择策略。Dynamic Token Coverage会先计算该层所有token的聚合重要性分布（跨头求和并归一化），然后以\tau为阈值确定最小的一组低重要性 token，其累积重要性刚好达到\tau。由于不同层的token重要性分布差异巨大，即使\tau固定，每层实际需要丢弃的token数量和保留的token数量都是动态自适应的。

Q：\tau选出的token数量，不同head相同吗，head是如何选出各自的token的？

A：在 Token Sparse Attention 中，\tau 选出的 token 数量（即 k_keep）对所有 attention head 都是相同的，这是通过层级覆盖率阈值 \tau 统一确定的（Algorithm 1 第9–11行）：先跨头聚合得到层级别重要性分布，然后找到最小的一组低重要性 token 使得其累积质量恰好达到 \tau，从而确定需丢弃的 token 数量 k_sparse，进而得到每个 head 应保留的 token 数量 k_keep = L - k_sparse。但每个 head 独立地选择自己保留的 token 集合：每个 head 根据自身计算的 token 重要性分数 s_h（通过少量 recent queries 与所有 keys 近似注意力后 pooling 得到），选取分数最高的 k_keep 个 token 作为该 head 的注意力子集。因此，尽管保留数量相同，不同 head 选出的具体 token 索引可以不同。
