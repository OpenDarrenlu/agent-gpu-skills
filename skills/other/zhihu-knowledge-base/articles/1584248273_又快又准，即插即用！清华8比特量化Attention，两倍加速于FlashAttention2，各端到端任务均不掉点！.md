# 又快又准，即插即用！清华8比特量化Attention，两倍加速于FlashAttention2，各端到端任务均不掉点！

**作者**: Tete​清华大学  计算机科学与技术博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/1584248273

---

大模型中，线性层的低比特量化（例如INT8, INT4）已经逐步落地；对于注意力模块，目前几乎各个模型都还在用高精度（例如FP16或FP32）的注意力运算进行训练和推理。然而，随着大型模型需要处理的序列长度不断增加，Attention（注意力运算）的时间开销逐渐成为网络优化的主要瓶颈。

为了提高注意力运算的效率，清华大学陈键飞团队提出了8Bit的Attention（SageAttention）。实现了2倍以及2.7倍相比于FlashAttention2和xformers的即插即用的推理加速，且在视频、图像、文本生成等大模型上均没有端到端的精度损失。

论文标题：SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration

论文链接：

开源代码：

即插即用举例

SageAttention 可以一行代码轻松替换掉 torch 中当前最优的 Attention 接口（scaled_dot_product_attention），实现即插即用的推理加速。

具体来说，SageAttention 的使用非常方便，使用 pip install sageattention 后，只需要在模型的推理脚本前加入以下三行代码即可：

from sageattention import sageattn
import torch.nn.functional as F

F.scaled_dot_product_attention = sageattn

效果上，以开源视频生成模型CogvideoX为例，使用SageAttention可以端到端加速35%，且生成的视频无损：

使用全精度Attention:

00:06
全精度Attention

使用SageAttention:

00:06

接下来，将从背景与挑战，技术方案，以及实验效果介绍SageAttention。

背景

随着大模型需要处理的序列长度越来越长（比如Llama3.1支持128K的序列长度），Attention的速度优化变得越来越重要。下图展示了一个标准的Transformer模型中各运算随着序列长度变化的时间占比：

挑战

为了方便指代注意力元算中包含的矩阵，我们先回顾一下注意力的计算公式：

将神经网络中各运算的数值类型从高比特量化至低比特是一种有效提升计算和访存效率的方法。然而，研究团队发现直接将注意力运算中的Q, K, P, V从FP16量化为INT8或者FP8后将会导致在几乎所有模型和任务上都会得到极差的结果，例如，在Unidiffuser文生图模型中，会得到一张完全模糊的图像；在Llama2-7B进行四选一选择题任务上得到25.5%的准确率。

经过仔细分析后，研究团队发现主要是两个原因导致了量化注意力的不准确：

（1）大多视频、图像生成模型中，矩阵K表现出了极强的通道维度的异常值分布，直接使用INT8或者FP8数据类型对其进行量化会导致巨大的误差。

（2）在所有模型中，对矩阵P, V进行量化不能保证一个模型中所有层的精度。下表展示了对P, V量化后，Llama2-7B和Unidiffuser模型所有层中，最差情况的层对应的量化注意力的准确度，（该准确度为量化注意力相比全精度注意力的误差），可以发现不管对P, V矩阵进行何种8Bit （INT8，E4M3，E5M2）量化，总有些层的准确率非常差，导致了端到端效果的下降。

技术方案

为了解决上述的两个关键问题，研究团队提出了对应的解决办法。

（1）对K进行平滑处理。SageAttention采用了一个简单但非常实用的方法来消除矩阵K的异常值：K = K – mean(K) 其中mean(K) 是沿着通道维度求平均值。这个简单的做法不仅不会影响注意力计算的正确性 Softmax(QK^T) = Softmax(Q(K-mean(K))^T) ；且对整个Attention速度的影响只有0.2%；同时还保证了量化后的注意力运算的精度：

（2）对Q, K进行分块INT8量化。对于矩阵Q, K，SageAttention采用了以FlashAttention的分块大小为粒度的INT8量化。这是因为：1. 对Q, K矩阵进行INT8量化相比于进行FP8量化，注意力的精度更高。2. 在一些常用卡上，比如RTX4090，INT8矩阵乘法（INT32为累加器）的速度是FP8（FP32为累加器）的两倍 [1]。

（3）对P, V采用FP16数据类型的矩阵乘法累加器。对于矩阵P, V，SageAttention采用了保留P, V为FP16的类型，但进行矩阵乘法时采用FP16数据类型的累加器。这是因为：1. PV矩阵乘法的数值范围始终在FP16的表示范围内，且经过大量实验验证，FP16作为累加器的数据类型不会带来任何精度损失（见下表）。2. 在一些常用卡上，比如RTX4090，以FP16为累加器数据类型的矩阵乘法的速度是FP32作为累加器的两倍 [1]。

SageAttention的流程图及算法如下所示：

实验效果

SageAttention实现了底层的GPU Kernel，在算子速度以及各个模型的端到端精度上都有十分不错的表现。

具体来说，算子速度相比于FlashAttention2 和 xformers 有 2.1 以及 2.7 倍的加速。以下4张图展示了在RTX4090上，不同的序列长度下SageAttention的各种Kernel与其他方法的速度比较。

以下4张图展示了在RTX3090上，不同的序列长度下SageAttention的各种Kernel与其他方法的速度比较。

下表展示了在RTX4090上，各模型中的注意力模块中SageAttention相比于使用模型原始的注意力的加速比。

真实任务的精度上，下表展示了SageAttention在视频、图像、文本生成等大模型上均没有端到端的精度损失：

欢迎使用SageAttention~: https://github.com/thu-ml/SageAttention




参考文献：

[1] https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
