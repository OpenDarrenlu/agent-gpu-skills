# 【RG Q&A Summary】[arXiv] RetroInfer:A Vector-Storage Approach for Scalable Long-Context LLM Inference

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/1934337496696725728

---

这篇文章来自中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
RetroInfer：一种用于可扩展长上下文大语言模型推理的向量存储方法

作者：Yaoqi Chen, Jinkai Zhang, Baotong Lu, Qianxi Zhang, Chengruidong Zhang, Jingjia Luo, Di Liu, Huiqiang Jiang, Qi Chen, Jing Liu, Bailu Ding, Xiao Yan, Jiawei Jiang, Chen Chen, Mingxing Zhang, Yuqing Yang, Fan Yang, Mao Yang







随着大语言模型（LLMs）上下文长度的不断增长，推理效率正面临严峻挑战，主要受限于GPU的内存和带宽。我们提出RetroInfer，一个全新的系统，将键值（KV）缓存重新构想为一个向量存储系统，利用注意力机制中的稀疏性来加速长上下文LLM推理。

该系统的核心是wave index，一种感知注意力的向量索引（Attention-aWare VEctor index），通过三分式注意力近似（tripartite attention approximation）、有精度界限的注意力估计（accuracy-bounded attention estimation）和分段聚类（segmented clustering）等技术，实现关键token的高效、准确检索。

与之配套的是wave buffer，它负责KV缓存的管理，同时在GPU和CPU之间协调计算与数据传输的重叠，从而维持高吞吐量。

与以往基于稀疏性的推理方法相比，这些方法通常在token选择与硬件协同上存在困难，而RetroInfer则在不牺牲精度的前提下提供了稳健的性能表现。

在长上下文基准测试中，RetroInfer在GPU内存限制内相较于完整注意力机制实现了最高4.5倍的加速，而在KV缓存扩展到CPU内存的情况下，相比稀疏注意力方法最多加速达10.5倍，且保持了与全注意力机制相当的精度。

Q&A

Q1：ANNS 在 KV Cache 中的使用是什么，是在 prefill 阶段把 key 向量算出来，对应 build 1个 ANNS 索引是吗？

A1：是的，其实本质上 KV Cache 的检索问题都属于 ANNS，即给定一个查询向量，在数据点中选出最近的向量。例如 MagicPIG 中用到的 LSH 也是 ANNS 领域经典的算法。

Q2：KV Cache 的搜索和 ANNS 的关系真的很大吗？ANNS 中的 update 问题、索引空间开销等，我觉得在 KV Cache 中不需要这么复杂的技术？

A2：有道理，其实本文最后也是选用了一个很简单的聚类方法构建索引，没有用到复杂的索引，例如图索引。

Q3：这篇文章有引用 ClusterKV 的 paper 吗？这篇文章也是做 clustering 的，这篇文章的聚类应该不是比较创新的部分？

A3：没有引用，并且聚类算法用到 KV Cache 中确实不是新鲜事，例如 SqueezedAttention 也是利用聚类算法。这篇文章比较新颖的地方，个人感觉还是在 Estimation zone 上，用低成本的计算利用起 un-retrieved 部分的 token，对精度做一个提高。

Q4：这篇文章中的 segment 是怎么划分的？

A4：就是在顺序的 token 序列上进行 segment 划分，比如 “Hello World”，Hello 是第一个 token，World 是第二个 token，这样得到 token 序列，按照这个顺序划分 segment。

Q5：实验部分，没有一个 baseline 是做聚类的？这里是不够公平的，其他的系统都是没有通过聚类方法做一种体积（信息）的压缩？

A5：是的。

Q6：Quest 的 offload 实现我感觉是没有必要的，这样反而把人家搞差了。

A6：是的。

Q7：论文里利用了 ANNS 聚类的方法，本应该重点对比一下索引构建开销的，文章对 decoding 阶段的测试做的非常详细，而关于 prefilling index building 的实验只有这么一点，这是不合理的

A7：对的，而且 prefilling 的测试选择的都是时间非常长的，达到分钟级别，我怀疑基础时间长会使得额外开销显得比较少。
