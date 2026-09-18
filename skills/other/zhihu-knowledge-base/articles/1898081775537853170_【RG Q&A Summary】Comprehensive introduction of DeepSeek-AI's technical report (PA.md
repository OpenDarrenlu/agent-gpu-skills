# 【RG Q&A Summary】Comprehensive introduction of DeepSeek-AI's technical report (PART ⅠI)

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/1898081775537853170

---

这里是中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。 Reading Group 的主页地址：ADSL Reading Group bilibili 链接：USTC-NHPCC的个人空间

本次分享有关 DeepSeek-V3 文章的内容，分为RL和3fs两个部分

作者：DeepSeek-AI

RL
Summary

后训练（Post-training）正变得越来越重要，通过Post-training可以实现定制化任务、对齐、以及Test-time scaling等。DeepSeek团队使用强化学习作为Post-training的主要手段。DeepSeek-R1技术报告中讲到了两个模型的训练，DeepSeek-R1-Zero和DeepSeek-R1。

R1-Zero使用GRPO进行训练，GRPO相比于PPO最大的改进是去掉了Value模型，从而达到了节省训练时显存和计算的目的。训练时的奖励设计包括回答的准确性以及格式是否符合要求两部分。不使用Model-based奖励来避免Reward Hacking。作为只使用强化学习进行训练的模型，R1-Zero涌现出了令人惊叹的Test-time scaling能力，在数学、编程等任务上取得了不亚于甚至好于OpenAI-o1的好成绩。

R1则是由初始模型DeepSeek-V3-Base经过4个阶段的训练得到。首先是冷启动阶段，该阶段使用DeepSeek-R1-Zero的输出对模型进行微调，来改善模型输出的可读性，同时避免强化学习初期的不稳定阶段。然后是强化学习阶段，该阶段与R1-Zero的训练过程相同，目的是增强模型在数学、代码等方面的推理能力。之后又是一个微调阶段，使用了推理和非推理数据，来加强模型在写作、角色扮演等任务上的表现。最后是另一个强化学习阶段，使用了推理数据+Rule-based奖励以及非推理数据+Model-based奖励的组合，来进一步改善模型的推理能力。

Q&A

Q1：怎么样理解有Reward和Value这样的两个东西，这两个东西的含义有什么不同？

A1：Value是将来获得的所有Reward的期望，因为RL希望能获得的总和Reward最高，所以要看能获得的总的Reward。以贪吃蛇游戏为例，可能朝某个方向走，马上可以吃到分（获得Reward），但之后就没有路走了，Value就会很低。

Q2：PPO里，clip的比例大概是个什么情况？

A2：比例和epsilon的值有关，我们测下来大概在20-30%左右。

Q3：PPO里的ratio function对training有什么作用？是不是用来加速training的？

A3：ratio是看当前的回答相比于ref model的回答有什么改变，就是一个不断改善模型回答的过程。同时有这个部分后续才可以算gradient来更新模型。

Q4：RL和蒸馏是否是适用于不同范围的？

A4：对，根据DeepSeek做的蒸馏和RL对比，对小模型蒸馏效果会是更好的。后续也有别的工作去探索对什么样的模型做RL才有效果，得出的结论是应该要1.5B及以上的模型。

3fs
Summary

3FS是一个专为AI工作负载设计的分布式文件系统，采用元数据与数据分离的架构，通过面向读友好的链式复制(CRAQ)机制和负载均衡策略，在180个存储节点上实现了6.6TiB/s的读取吞吐量。其主要的设计目标是支持大规模的AI训练和推理任务，尤其是对大文件的高效读取。

Q&A

Q1：CRAQ相比传统主从复制有什么优势？

A1：CRAQ的主要优势在于读取友好性：(1) 可以从链上任意节点读取数据，避免访问主节点；(2) Leader 节点故障时仍可提供读取服务；(3) 能充分发挥SSD和网络的聚合带宽。但代价是写入必须同步传播整个链。

Q2：平衡链表如何保证故障时的负载均衡？

A2：通过精心设计链表中存储目标的排列顺序，使得当某个SSD故障时，其负载会被均匀分散到剩余所有SSD上。文档中的例子显示，6个SSD中1个故障时，剩余SSD的负载仅增加20%，而非简单方案下的50%。

Q3：3FS为什么使用小端字节序存储inode ID？这对负载均衡有什么帮助？

A3：自然分布特性：小端存储使得连续分配的 inode ID 在键空间中是随机分布的，避免了基于 key 顺序的 FoundationDB 热点问题。这种设计使得元数据操作尽可能均匀分布在FoundationDB的所有分片上，不会因为 inode ID 的连续性（小 inode ID 通常在目录树的上层）导致某些分片过载。
