# 【RG Q&A Summary】[arXiv] Down with the Hierarchy: The 'H' in HNSW Stands for "Hubs"

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/1915539843318002513

---

这篇文章来自中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
Down with the Hierarchy: The 'H' in HNSW Stands for "Hubs"

作者：Blaise Munyampirwa, Vihan Lakshman, Benjamin Coleman

这篇文章是2025年的arXiv文章，它关注到了目前得到广泛应用的用于向量检索的图索引算法HNSW。随着LLM的广泛应用，目前向量检索所用到的向量已经发生了改变，从早期的几十维的向量变为了现在百维甚至千维的向量。作者发现HNSW被提出的时候，向量检索所用到的向量基本都是一百维左右，而现在HNSW会被用于千维的向量，这种从低维到高维的变化可能会对HNSW的性能产生影响。

作者认为在高维情况下，HNSW的hierarchical structure会失去作用。因此作者先进行了benchmark实验来对比在高维情况下HNSW的性能与NSW的性能。为了去除目前向量数据库中引入的工业级优化的影响，作者使用的HNSW代码是随着HNSW论文一起发出的开源代码https://github.com/nmslib/hnswlib，NSW也是在此基础上修改得到的。结果发现在高维情况下，HNSW的latency与NSW的latency相差无几，但是相对于NSW，HNSW会引入额外40%的内存开销。

针对这一现象，作者认为是在高维情况下，NSW构建后就会出现hub nodes和highway导致的。这里的hub nodes的意思是连接良好且遍历次数非常多的顶点，而hub nodes之间的边就称为highway，hub nodes和非hub nodes之间的边称为feeders。作者发现在高维情况下，NSW的搜索过程通常是entry point先通过feeders快速到达hub nodes，然后通过highway快速横跨整个图找到离nearest neighbor最近的hub node，然后通过feeder到达nearest neighbor。在这种情况下HNSW引入的hierarchical structure只是在重复放置highway而已，不需要hierarchical structure也可以在NSW中快速横跨整个图找到nearest neighbor。







为了验证这一猜想，作者将其拆分为了3个假设：1. 高维情况下图中存在一些顶点它们的访问频率比其它顶点多，称它们为hub nodes；2. hub nodes总是倾向于互相连接；3. 在搜索的早期阶段，总是会先遍历到hub nodes。作者进行了大量实验证明这3个假设在高维情况下是成立的，因此作者认为上述猜想成立，在高维情况下应该深入思考高维带来的影响改进索引。

Q&A

Q1：NSW在搜索时如何发现highway?

A1：NSW在搜索时是贪心搜索，搜索时它会遍历当前访问到的顶点的所有邻居，选择和query最近的那个邻居作为下一个访问的顶点。因此NSW无法辨别highway，这也是HNSW的改进思路。

Q2：highway如何定义？

A2：在NSW的定义上应该是那些由于NSW的构建过程而出现的距离比较远、在正常的按距离大小顺序插入不会出现的那些边，这些边可以快速地横跨整个图，对一些情况的搜索带来便利。但是论文中没有明确定义，论文是在后续的猜想中才定义highway，但是这种情况下定义出来的highway难以发现其特殊性质，讨论认为还需要进一步更明确的定义。

Q3：NSW的搜索过程如何停止？

A3：NSW用于向量搜索，一般会寻找topk个结果，直到找到足够的candidate才结束，PPT演示只展示了一部分。

Q4：通过highway一定能让搜索获益吗？

A4：目前认为对于某一特定情况不一定保证获益，但是平均来说通过highway是能让搜索变快的。

Q5：对于HNSW和NSW在高维和低维情况下的差异会不会是因为数据集的不同导致的？

A5：回看了数据集，发现论文选用的数据集在高维和低维情况下用于构建图的顶点数确实没有随着维度上升而增加，那么确实有可能性能的差异是由于高维情况下的数据太稀疏了导致的。论文没有考虑这一点。

Q6：Hub nodes和维度的关系是什么？直觉上hub nodes无论什么维度都有可能出现。

A6：论文的实验认为维度越高这些hub nodes出现的概率越高。

Q7：目前实际的production用到的维度是多少？感觉96维就差不多了？

A7：目前按照实际的使用，维度从几十到几百维不等，有些业务是会用到上千维的维度的。
