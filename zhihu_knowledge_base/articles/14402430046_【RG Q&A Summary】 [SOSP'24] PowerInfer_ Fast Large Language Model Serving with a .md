# 【RG Q&A Summary】 [SOSP'24] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/14402430046

---

这里是中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
PowerInfer（面向消费级GPU的快速大型语言模型推理系统）

作者：Yixin Song, Zeyu Mi, Haotong Xie and Haibo Chen

总览图

本文介绍了 PowerInfer，一种在配备单张消费级 GPU 的个人电脑上运行的大语言模型（LLM）高效推理引擎。PowerInfer 的核心设计理念是利用 LLM 推理中固有的高局部性特性，这种特性表现为神经元激活的幂律分布：少量“热神经元”在不同输入中始终被激活，而大多数“冷神经元”则随具体输入而变化。

基于这一观察，PowerInfer 设计了一种 GPU-CPU 混合推理引擎：将热神经元预加载到 GPU 以实现快速访问，而冷神经元则在 CPU 上计算，从而显著降低 GPU 内存需求和 CPU-GPU 数据传输开销。此外，PowerInfer 集成了自适应预测器和神经元感知的稀疏算子，进一步优化了神经元激活效率和计算稀疏性。

评估结果显示，PowerInfer 在单张 NVIDIA RTX 4090 GPU 上对比 llama.cpp 性能提升高达 11.69 倍，同时保持模型精度。针对 OPT-30B 模型，PowerInfer 的性能接近高端服务器级 A100 GPU，达到了其生成速度的 82%。

Q&A

Q：它判断，比如说一个模模型，就哪些参数是热的，哪些是冷的，它大概要花多长时间，就相比于整个推理过程？
A：他其实就是会拿一些现有的一些数据集去 offline 的去给他把这个数据集遍历一遍，这个开销很小，因为一个模型你只需要搞一次，然后之后就一直去部署就行了。

Q：为什么会用2080p 这种机器？
A：文章就是面向消费级显卡，这样可以让更多的人受益。2080 Ti人手都有，A100就不一定了。

Q：GNN里面sparsity会很高，但是LLM里面会有那么多sparsity高的场景嘛？
A：文章里面涵盖了40%~98%各种sparsity的场景，当然sparsity低的场景收益相应也会低。
