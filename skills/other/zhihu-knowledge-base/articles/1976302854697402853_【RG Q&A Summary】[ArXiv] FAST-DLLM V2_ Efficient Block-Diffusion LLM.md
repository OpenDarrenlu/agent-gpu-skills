# 【RG Q&A Summary】[ArXiv] FAST-DLLM V2: Efficient Block-Diffusion LLM

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/1976302854697402853

---

这篇文章来自中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。
Reading Group 的主页地址：ADSL Reading Group
bilibili 链接：USTC-NHPCC的个人空间
FAST-DLLM V2：高效块扩散大语言模型

作者：Chengyue Wu, The University of Hong Kong and NVIDIA; Hao Zhang, NVIDIA; Shuchen Xue, NVIDIA; Shizhe Diao, NVIDIA; Yonggan Fu, NVIDIA; Zhijian Liu, NVIDIA; Pavlo Molchanov, NVIDIA; Ping Luo, The University of Hong Kong; Song Han, NVIDIA and MIT; Enze Xie, NVIDIA

FAST-DLLM 架构图
内容总结：

​ 基于自回归（Auto-Regressive）的大语言模型（LLM）通过对下一个token的预测进行建模，生成流畅、连贯的文本。它们在大多数自然语言任务中所展现的卓越性能使其成为LLM系统中的主流范式。然而，AR型的LLM存在固有的低效缺陷：由于token严格从左到右顺序逐个生成，它们无法在解码过程中同时生成多个token，从而完全发挥并行效率。

​ 对此，基于扩散的大语言模型（dLLM）提供了一个在推理速度上更具潜力的替代方案。通过允许多个token共同预测，dLLM原则上可以实现更高的解码并行性。不过，当前开源的dLLM模型在实践中同样存在缺陷：双向注意力的结构使它们无法有效使用KV Cache进行缓存，在保证生成质量的前提下，当前开源的dLLM模型在速度上仍无法超越AR模型。

​ 为了解决这个问题，块扩散（Block-Diffusion）逐渐成为开源dLLM模型中的新主流架构。块扩散是一种半自回归半扩散的方法：在每个区块内以扩散的方式并行生成token，在区块之间以自回归的方式从左到右解码。区块内的并行解码能提升推理速度，而区块之间自回归式的架构为KV Cache的缓存复用再次提供了土壤。

​ Fast-dLLM v2这篇文章推出了一个高效的块扩散大语言模型，在训练和推理上做了对应的协同优化。

​ 在训练上，这项工作将预训练的AR模型微调到块扩散的框架中。通过块扩散机制与互补注意力掩码的结合，设计AR友好性质的注意力掩码结构，实现按块的双向上下文建模，在保留原始AR模型训练目标和预测性能的同时，提升微调的数据效率。

​ 在推理上，这项工作设计了一种分层缓存机制：一个跨块存储上下文表示的块级缓存，以及一个支持块内进行高效并行解码的子块缓存。这种设计可以在有效跨区块重用上下文的同时，加速每个区块内的token生成速度，是在纯自回归与纯扩散之间的一种trade-off。

​ 该工作在7B参数的模型上进行了全面的大规模实验。结果表明，相较于标准AR模型，Fast-dLLM v2在保证生成质量的同时实现了2.5倍的推理加速。

Q&A

Q：dLLM的inference还是需要做多轮的predict和remask的操作，才能得到比较好的效果。如果把模型训练得很好，模型能力很强，那是可以少做几轮的迭代就可以达到一个比较好的效果吗？

A：理论上是的，但是当前的开源模型比如llada，它最好的效果还是一次只生成一个token。在现在的情况下，主要是通过training和inference作协同优化来达到一个比较好的效果。

Q：如果文本已有信息很少，mask ratio很高，那模型训练的搜索空间会很大，训练难度是否会很高？

A：从已有的paper来看，纯扩散模型的训练难度确实很高，会像以前bert模型一样数据利用率较低。现有的dLLM模型和bert模型的区别在于训练时的mask ratio是可变化的，而且dLLM的优势主要体现在它的推理速度会比已有自回归型的LLM要快很多。

Q：dLLM并行生成token的解码有点类似投机推理，它在推理速度上的效果是ok的，但是要怎么保证准确率呢？感觉上它能一下达到一个不错的效果，但是想要更加细化就比较难？

A：现在开源dLLM模型的accuracy也没有说很高，闭源模型也不知道是如何做的，也没有说能碾压AR型的LLM。dLLM的并行解码在原理上是对独立概率的刻画，相较于文本预测真正的联合概率分布，是不可避免要做速度和准确率之间的trade-off。现实中的应用场景有时候并不需要强逻辑性和高准确率，不是every token predict，因此我觉得dLLM绝对是有应用场景的。
