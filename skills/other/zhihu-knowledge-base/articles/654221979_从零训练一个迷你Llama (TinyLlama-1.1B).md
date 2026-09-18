# 从零训练一个迷你Llama (TinyLlama-1.1B)

**作者**: Peiyuan007UCSD CS PhD student

**原文链接**: https://zhuanlan.zhihu.com/p/654221979

---

介绍


TinyLlama 项目旨在3万亿tokens上进行预训练，最终构建一个拥有11亿参数的，基于Llama架构的语言模型。经过精心优化，我们仅需16块A100-40G GPU，预计90天完成训练。训练已于2023-09-01开始。我们采用了和Llama2相同的分词器。这意味着TinyLlama可以在许多基于Llama的开源项目中即插即用。此外，TinyLlama只有1.1B的参数，体积小巧，适用于需要限制计算和内存占用的多种应用。


我们github repo链接如下：
TinyLlama-1.1B


潜在应用场景


小型但强大的语言模型对许多应用都很有用。以下是一些潜在的场景：


帮助对大型模型进行speculative decoding。
在边缘装置上运行，比如离线的实时机器翻译 (TinyLlama的4比特量化版本的模型权重只需要550MB的内存)。
在游戏中实现实时对话生成(因为还得给游戏本身留显存所以模型要小)。

此外，我们的代码可以给初学者做一个入门预训练的简洁参考。如果你要训练50亿以下参数的语言模型, 你其实不需要Megatron-LM。


我们的训练速度很快


我们的代码库支持以下特性：


multi-gpu and multi-node distributed training with FSDP.
flash attention 2.
fused layernorm.
fused swiglu.
fused cross entropy loss .
fused rotary positional embedding.

有了这些优化, 我们可以达到24k tokens/秒/A100的训练速度，也就是56%的MFU（在A100-80G上的MFU会更高）。这个速度可以让你可以在8个A100上用32小时训练一个chinchilla-optimial的模型(11亿参数，220亿token)。这些优化也大大减少了显存占用, 我们可以把11亿参数的模型塞入40GB的GPU里面还能同时维持16k tokens的per-gpu batch size。只需要把batch size改小一点， 你就可以在RTX 3090/4090上面训练TinyLlama。 下面是我们的代码库与Pythia和MPT的训练速度的比较。


Model	A100 GPU hours on 300 tokens
TinyLlama-1.1B	3456
Pythia-1.0B	4830
MPT-1.3B	7920


我们的repo提供了完整的预训练代码，后续也会更新SFT所需要的代码、数据、训练参数配置等，用户可以根据自己的需求调整。
