# 清华稀疏Attention，无需训练加速一切模型！(SpargeAttn详细解读，4-7倍加速于FlashAttention的秘密）

**作者**: Tete​清华大学  计算机科学与技术博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/1890065552170545501

---

在当今各类大语言模型以及视频模型中，长序列场景越来越普遍，而Attention的计算复杂度随着序列长度呈平方增长，成为长序列任务下的主要计算瓶颈。此前，清华大学陈键飞团队提出的即插即用量化的SageAttention系列工作实现了3倍加速于 FlashAttention，且在各类大模型上均保持了端到端的精度。

为了进一步加速Attention，清华大学陈键飞团队进一步提出了无需训练的即插即用的稀疏注意力SpargeAttn，可以加速任意模型。实现了4-7倍相比稠密FlashAttention的推理加速，且在语言，视频、图像生成等大模型上均保持了精度无损。

论文标题：SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference

论文链接：

开源代码：

（注：SpargeAttn是建立在 SageAttention 基础上的工作）

SpargeAttention 支持了即插即用的API，只需指定topk即可；也支持了自定义稀疏mask的即插即用API，非常易于使用:

接下来，将从前言，挑战，方法，以及实验效果四个方面介绍SpargeAttn。




前言

随着大模型需要处理的序列长度越来越长，Attention的速度优化变得越来越重要。这是因为相比于网络中其它操作的O(N)的时间复杂度，Attention的时间复杂度是O(N^2)。为了方便指代注意力运算中的矩阵，先回顾一下注意力的计算公式：

尽管 Attention 的计算复杂度为 O (N^2)，但幸运的是 Attention 具备很好的稀疏性质，即 P 矩阵的很多值都接近 0。如何利用这种稀疏性来节省计算就成为了 attention 加速的一个重要方向。大多数现有的工作都集中在利用 P 矩阵在语言模型中表现出来的固定的稀疏形状（如滑动窗口）来节省计算，或是需要重新训练模型，比如 DeepSeek 的 NSA 以及 Kimi 的 MoBA。此外，现有稀疏 Attention 通常需要较大的上下文窗口（如 64K~1M）才能有明显加速。SpargeAttn 的目标是开发一个无需训练、对各种模型（语言 / 视频 / 图像）通用、精度无损、对中等长度的上下文（如 4-32K）也有加速效果的注意力算子。也就是对任意输入都可以实时进行算子级别的稀疏加速。

不同的模型表现出不同的稀疏模式
实现通用的，无需训练的稀疏Attention有哪些挑战？
挑战1-通用性：Attention虽然具备稀疏性质，但是其稀疏形状在不同的模型甚至同一模型的不同层中都是不同的，体现出很强的动态性。如图1所示，前两种模型分别为视频模型和图像生成模型，这两个模型中的Attention的稀疏形状相比语言模型更加没有规律。设计一种各种模型通用的稀疏Attention是困难的。
挑战2-可用性：对于各种Attention的输入，很难同时实现准确且高效的稀疏Attention。这是因为准确性要求了完全精确地预测P中的稀疏区域，高效性则要求了此预测的时间开销极短。在一个极短的时间内完全精准地预测P的稀疏形状是困难的。




预备知识-稀疏的 FlashAttention

FlashAttention是一种为了节省GPU显存读写的一种Attention实现方法。想要实现一种标准Attention的高效计算方法离不开FlashAttention。为了便于理解后续的算法，此处先定义一下稀疏的FlashAttention的计算流程。

首先，先回顾FlashAttention的计算流程：

其中 Q_i， K_j^\top， V_j 对应了针对 Q，K，V 矩阵的分块。 \widetilde \sigma 是一个 Online Softmax 的运算，对应的计算是：

针对公式（1）中的计算流程，FlashAttention将Q矩阵的分块对应的计算分布到GPU不同的处理器中，将K矩阵的分块对应的计算在一个处理器中进行循环迭代，最终可得到Attention的输出结果。




了解完FlashAttention，稀疏Attention就变得直观。首先，可以定义一个稀疏掩码 M_g ，并令：

就可以省略稀疏掩码对应的Q与K的分块，P与V的分块之间的矩阵乘法运算。

其次，还可以进一步定义一个稀疏掩码 M_{pv} ，并令：

这样还可以进一步省略P与V之间的分块矩阵乘法运算。

但问题是，该省略哪些Q与K，P与V的分块之间的矩阵乘法运算可以不影响Attention的计算结果呢？

方法

为了解决上述的两个挑战，研究团队提出了对应的解决办法。

（1）研究团队提出了一种各模型通用的快速的对P矩阵稀疏部分进行预测的算法。该方法选择性地对Q, K矩阵进行压缩并预测P矩阵，接着使用TopCdf操作省略P中稀疏部分对应的QK^T与PV的矩阵乘法。

（2）研究团队提出了在 GPU Warp级别上的稀疏Online Softmax算法，该算法通过利用Online Softmax中全局最大值与局部最大值之间的差异，进一步省略了一些PV的矩阵乘法计算。

（3）可选的，针对视频和图像模型，研究团队充分利用图像以及视频中的Token局部相似性质，使用希尔伯特重排的方法对Attention前的Token进行重新排列，进一步提高稀疏度。

（4）最后，研究团队将这种稀疏方法与基于量化的SageAttention融合到一起，进一步加速Attention。

SpargeAttn的算法流程图
选择性块压缩以及稀疏预测
不同模型表现出相近Token之间的相似性

尽管不同模型的P矩阵的稀疏形状各不相同，但研究团队观察到各种模型表现出一个共同特征：在Q矩阵和K矩阵中，大多数距离较近的Token显示出高度相似性（如图）。

研究团队提出：

对Q,K矩阵按照FlashAttention的分块大小进行分块，并计算每一块之间的余弦相似度。

2. 针对由高度相似的Token组成的块，使用平均压缩压缩为单一Token。

3. 将压缩过的Q, K进行矩阵乘法和Softmax运算，得到压缩后的P矩阵。

4. 对压缩后的P矩阵的每一行进行TopCdf运算，对于TopCdf之外的值对应的QK^T和PV矩阵乘法进行省略运算。

需要注意的是这里共有两个关键点：

对Q, K矩阵中的Token块的压缩是选择性的，即只压缩相似度高的，而相似度低的Token块对应的Attention运算不可省略。这是因为对相似度不高的Token块进行压缩会损失信息，可能导致最终忽略一些重要的矩阵运算。

2. 研究团队使用了TopCdf来根据Softmax中的值执行动态稀疏，而不是使用Top-K来省略固定数量的矩阵乘法。这是因为Top-K可能会将一些包含较多信息量的矩阵乘法忽略掉。




具体来说，上述的运算流程如下所示：

TopCdf可以理解为在每一行Softmax归一化的概率分布中，选择其中最大的概率值的累计分布值达到τ的对应的块进行计算, 其他块则省略。TopCdf可以实现自适应稀疏度同时保证精度。TopCdf使用代码表示如下:

最后, 因为相似度低的块对应的运算不能忽略, 所以研究团队将稀疏掩码 中对应的值设置为1:

最后，只需在FlashAttention的计算过程中，将稀疏掩码 的值为0对应的矩阵乘法进行省略即可：

稀疏Online Softmax

研究团队还发现了一种简洁有效的进一步省略PV矩阵乘法的方法。具体来说，在FlashAttention的分块的PV矩阵乘法中，如果分块的P矩阵中的所有值都足够小的话，则可以省略该分块对应的PV矩阵乘法。而根据FlashAttention的计算流程，P矩阵的分块的值为：

\widetilde P_{ij} = exp(S_{ij} - m_{ij})

根据预备知识中对于FlashAttention的讲解，可以直接写出：

可以发现，如果:

那么 P 和 V 之间的分块矩阵乘法则可以省略。证明如下：

这无疑是一种非常巧妙的做法，只需在FlashAttention的计算过程中加一行判断即可进一步省略P和V的矩阵乘法。




使用SageAttention

为了进一步加速稀疏Attention，研究团队还将SpargeAttn与此前第一个做到即插即用，同时各模型不掉点的低比特量化Attention（SageAttention）结合起来。注意这里的结合是非常直接的，因为稀疏与量化本身就是正交的。

综合上述的稀疏算法以及与SageAttention结合，SpargeAttn的算法流程如下所示：

Token的希尔伯特重排

研究团队还提出一种可选的，可以进一步提高图像和视频模型中Attention稀疏度的方法。具体来说，研究团队发现图像和视频模型中相邻的像素很可能是相似的。对应到模型的Token（对应一个图像patch）中则是相邻的Token以及跨过固定长度序列的Token之间具备较高的相似度，因为这些Token对应到图像中是相邻的。为了利用这一特点，研究团队提出了希尔伯特重排，即将相邻的Token在Attention前排列在一起，在Attention后再恢复原始排列，目的是增加相邻Token之间的相似性，从而提升SpargeAttn的稀疏度。

实验效果

总的来说，SpargeAttn在视频、图像、文本生成等大模型均可以实现无需训练的加速效果，同时保证了各任务上的端到端的精度。下表展示了SpargeAttn在各模型上的稀疏度，Attention速度，以及各任务上的端到端精度：（注：此论文中的所有实验都是基于SageAttention实现，目前Github仓库中已有基于SageAttention2的实现，进一步提供了30%的加速（https://github.com/thu-ml/SpargeAttn）

下图展示了SpargeAttn的速度，可以发现在RTX4090上，SpargeAttn在60%稀疏度的情况下可以达到900TOPS的速度，几乎是A100理论峰值性能的3倍。

研究团队还发现序列越长，Attention稀疏度越高：

值得一提的是，此前的稀疏Attention工作很多无法实际使用的原因之一是稀疏预测部分的Overhead较大，而SpargeAttn团队还将稀疏预测部分的代码进行了极致优化，将Overhead压缩到了几乎在各种长度的序列下都可以忽略的地步：

下表展示了对于各模型的端到端的加速效果：（注：此论文中的所有实验都是基于SageAttention实现，目前Github仓库中已有基于SageAttention2的实现，进一步提供了30%的加速（https://github.com/thu-ml/SpargeAttn）

下图是一些在语言、图像和视频生成模型中的可视化对比的示例：

欢迎大家去Github仓库使用SpargeAttention，并引用该文章:
