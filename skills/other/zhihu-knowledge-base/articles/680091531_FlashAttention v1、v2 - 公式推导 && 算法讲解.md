# FlashAttention v1、v2 - 公式推导 && 算法讲解

**作者**: Alan 小分享​香港科技大学 资讯科技硕士

**原文链接**: https://zhuanlan.zhihu.com/p/680091531

---

​
目录
收起
1、设计思路 && 公式推导
1.1、Attention 标准实现
1.2、第一步：优化 Softmax 计算过程（Online Softmax）
1.3、第二步：把 O = PV 的计算考虑进来
1.4、怎么直观一点理解这些公式呢？
2、Forward 流程
2.1、FlashAttention v1
2.2、FlashAttention v2
3、Backward 流程
3.1、softmax 求导公式
3.2、backward 标准实现
3.3、FlashAttention v1
3.4、FlashAttention v2
4、总结
Reference

为了提高大模型中 Attention 层的计算速度，Tri Dao 在 2022 年 5 月提出了 FlashAttention 算法（即 V1），计算速度相比于标准实现提高了 2 - 4 倍（不同的 sequence length 会不一样）。这个算法主要针对的是训练场景～

论文链接：

FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
arxiv.org/abs/2205.14135v2

然后在 2023 年 7 月推出了改进版本 FlashAttentionV2，进一步将速度提高为原来的 2 倍。主要优化点是调换了外层和内层循环的顺序，并且将针对 Q 的循环也改为了使用多个 thread block 来计算。

原来是只针对 batch 和 head 维度进行了并发，即每个 sample 的每个 head 分配一个 thread block（共 batch_size * head_num 个 thread block）。现在则是将针对 Q 的循环也进行了并发，即总的 thread block 增加了，从而提高 GPU 的利用率。

FlashAttention V2 论文：

另外，Tri Dao 在 2023 年 10 月推出了针对推理场景的版本 FlashDecoding：

基本原理其实差不多，先好好理解下 FlashAttention，然后 FlashDecoding 也很容易理解了。

1、设计思路 && 公式推导
1.1、Attention 标准实现

先看看标准的 Attention 实现：

这里最主要的问题是第 1 - 3 步中，每一轮都要从 HBM（即 global memory）读数据，计算完后再将结果写回去，效率就很低。




那么有没有什么办法，可以减少读写次数呢？

那当然是有的。下面我们一步步拆解～～




背景：

1、GPU 中计算时，一般是把数据一块一块（Tiling）地加载到 shared_memory（即 on-chip SRAM），然后再计算。（下文中 Br 和 Bc 是为了设定每一块数据的大小而设置的参数）

2、safe softmax 公式：（即计算 exp 时需要减去每一行的最大值 m）

\text{safe softmax}(s_i) = \frac{e^{s_i - m}}{\sum_{j=1}^{n} e^{s_j - m}} （其中 s_i 表示某一行中第 i 个元素）




我们先看看为了计算最终结果 O 中的 Br 行，具体需要经历哪些过程。然后再想办法优化～～

加载 Q 中的 Br 行数据到（即大小为 Br * d）的矩阵，记为 Q_i ;
遍历K^T ，每次加载 Bc 列（即大小为 d * Bc），记为 K_{j}^{T} ，然后计算 S_{i,j} = Q_iK_{j}^{T} （大小为 Br * Bc）
然后得到了 S 中的 Br 行数据；
为了计算 softmax，我们需要对 S 中的这 Br 行数据遍历 3 遍，
第一遍是计算每一行的最大值 m，
第二遍计算每一行的 sumexp，即 \sum_{j=1}^{n} e^{s_j - m} ，
第三遍才是计算 safe softmax 的值；
然后我们就得到了 P 的 Br 行数据；
将 P 中的 Br 行 数据，与 V 进行矩阵相乘（也是逐块数据进行加载和计算），就得到了 O 中 的 Br 行数据，done！




可以清晰感受到，为了计算 P（即 softmax 结果），我们进行了多轮的数据遍历；

那么有没有可能压缩成一轮遍历呢？

即我们在遍历 K_{j}^{T} 的时候，就把 P_{i,j} 算好，然后乘以 V_j （V 中第 j 块数据），再把每一次的结果（ j = 1 到 N）累加起来，就得到了 O 中对应位置的结果。

当然是可以的，这就是 FlashAttention 中做的事情！

下面我们再看看怎么做～～




1.2、第一步：优化 Softmax 计算过程（Online Softmax）

先来看看怎么优化 softmax 的计算，即 P = sofamax(S) 这一步～～

前面我们说过，为了对某一行数据计算 softmax，需要对这一行数据遍历三次。

来看看具体的过程：

记我们需要计算的这一行为 s，记 s_i 为其中的第 i 个元素，

记 m_i 为第 1 到 i 个元素的最大值，记 l_i 为第 1 到 i 个位置的 sumexp，

记 p_i 为第 i 个位置的 softmax 值，

计算过程：

接下来就可以思考下怎么把第 1、2 轮遍历，融合成一轮呢？

计算 m_i 的时候，只用到了 m_{i-1} 和 s_i ，并没有用到位置大于 i 的元素的信息；

但是计算 l_i 的时候，用到了 m_N ，即所有元素的最大值；




于是，我们可以考虑，构造 l_i 的替身 \tilde{l}_i = \sum_{j =1}^{i}{e^{s_j - m_i}} ，这时只需要用到截止 i 位置的信息。

且 \tilde{l}_N 等于 l_N。

于是，只需要找到 \tilde{l}_i 和 \tilde{l}_{i-1} 的迭代关系，就可以融合成一轮遍历了。

推导过程：

\begin{align*} \tilde{l}_i &= \sum_{j=1}^{i} e^{s_j - m_i} \\ &= \left( \sum_{j=1}^{i-1} e^{s_j - m_i} \right) + e^{s_i - m_i} \\ &= \left( \sum_{j=1}^{i-1} e^{s_j - m_{i-1}} \right) e^{m_{i-1} - m_i} + e^{s_i - m_i} \\ &= \tilde{l}_{i-1} e^{m_{i-1} - m_i} + e^{s_i - m_i} \end{align*}




现在 softmax 的计算可以优化成：（这个就 Online Softmax）

到这里后，就不再能压缩成一个 for 循环了。




1.3、第二步：把 O = PV 的计算考虑进来

虽然计算 P = softmax(S) 这一步只能压缩为两个 for 循环，但是可以把 O = PV 的计算考虑进来，就可以进一步压缩成一个 for 循环了！

先一步步来～～～

为了方便描述，假设我们现在的目标是计算 O 中第 k 行的结果，结合上面的 online softmax 算法，当前计算 attention 的流程如下：

（注：O 中每一行的计算都是独立的。

比如为了计算 O 中第 k 行，需要：

取 Q 中第 k 行，与整个 K^T 相乘，得到 S 中第 k 行；
进行 softmax，得到 P 中第 k 行；
然后与整个 V 相乘，就得到了 O 中第 k 行；

具体过程：

现在我们的目标是将两个 for 循环，压缩成一个 ！！

先将 p_i 的公式代入 o_i ，得到

（注： o_i 是一个行向量， p_i 是单一一个标量）

o_i = o_{i-1} + p_i V[i,:] = \sum_{j=1}^i p_j V[j,:] = \sum_{j=1}^i( \frac{exp(s_j - m_N)}{\tilde{l}_N} V[j,:])

可以看到，这里依赖第一个 for 循环的结果 m_N 和 \tilde{l}_N ，

于是，我们参考 1.2 中的做法，构造 o_i 的替身

\tilde{o}_i = \sum_{j=1}^i( \frac{exp(s_j - m_i)}{\tilde{l}_i} V[j,:]) ，

其中 ，m_i 和 \tilde{l}_i 代表的是截止第 i 轮循环，我们得到的最大值，以及对应的 sumexp。

现在我们可以知道 \tilde{o}_N 等于 o_N。

于是，只需要找到 \tilde{o}_i 和 \tilde{o}_{i-1} 的迭代关系，就可以融合成一轮遍历了。

推导过程：

\begin{align*} \tilde{o}_i &= \sum_{j=1}^{i} \frac{e^{s_j - m_i}}{\tilde{l}_i} V[j,:] \\ &= \left( \sum_{j=1}^{i-1} \frac{e^{s_j - m_i}}{\tilde{l}_i} V[j,:] \right) + \frac{e^{s_i - m_i}}{\tilde{l}_i} V[i,:] \\ &= \left( \sum_{j=1}^{i-1} \frac{e^{s_j - m_{i-1}}}{\tilde{l}_{i-1}} \frac{\tilde{l}_{i-1}}{\tilde{l}_i} V[j,:] \right) + \frac{e^{s_i - m_i}}{\tilde{l}_i} V[i,:] \\ &= \left( \sum_{j=1}^{i-1} \frac{e^{s_j - m_{i-1}}}{\tilde{l}_{i-1}} V[j,:] \right) \frac{\tilde{l}_{i-1} e^{m_{i-1} - m_i}}{\tilde{l}_i} + \frac{e^{s_i - m_i}}{\tilde{l}_i} V[i,:] \\ &= \tilde{o}_{i-1} \frac{\tilde{l}_{i-1} e^{m_{i-1} - m_i}}{\tilde{l}_i} + \frac{e^{s_i - m_i}}{\tilde{l}_i} V[i,:] \end{align*}




于是就有了 FlashAttention 中的算法：





实际计算时，我们一般是进行分块计算，就像 1.1 中的图那样子～～




1.4、怎么直观一点理解这些公式呢？

观察下我们做了替身的两项：\tilde{l}_i = \sum_{j =1}^{i}{e^{s_j - m_i}} 和 \tilde{o}_i = \sum_{j=1}^i( \frac{exp(s_j - m_i)}{\tilde{l}_i} V[j,:]) ，

都是前面多项的想加，然后都用到了截止第 i 项的信息（比如第 1 到 i 个位置的最大值 m_i ）；、




比如进一步看看 \tilde{l}_i，在计算第 i 轮循环时，我们已经有的值是 m_i 和 \tilde{l}_{i-1} = \sum_{j =1}^{i-1}{e^{s_j - m_{i-1}}}，

注意 \tilde{l}_{i-1} 中每一项的指数部分都是是 m_{i-1} 进行计算，但是 \tilde{l}_i 中每一项都是用 m_i 计算，

那么我们可以对 \tilde{l}_{i-1} 进行 rescale，即乘以 e^{m_{i-1} - m_i} ，就解决了这个问题，然后加上最新的一项 e^{s_i - m_i} ，

就得到了 \tilde{l}_i～～




\tilde{o}_i 也是类似的原理，对 \tilde{o}_{i-1} 进行 rescale，然后加上最新的一项，就完事了！




2、Forward 流程
2.1、FlashAttention v1

FlashAttention v1 就是按照上面的公式来计算。

具体操作时，用了分块计算，并且外层是对 K^T 进行循环，内层对 Q 进行循环；（v2 中就把这个循环的顺序调换了，更方便并行，本来每一行的计算就是独立的）

下标有点凌乱，总结了几个理解技巧：

所有以ij作为下标的，都表示只针对当前分块的计算；
所有以i作为下标的，都表示截止到前一个分块（即第 1 到第 i - 1 个分块）的计算结果；
所有以new为上标的，都表示引入当前分块做更新后的结果；

比如

\tilde{m}_{ij} 表示只对 S_{ij} 这个分块做 rowmax，

m_i 表示从 S_{1,j} 到 S_{i-1,j} 这 i - 1 个分块的 rowmax，

m_i^{new} 则是结合了 m_i 和 \tilde{m}_{ij} 后的结果。




另外，公式从直观上看，和 1.3 有一点点区别的地方就是，

第 10 行是先对当前分块计算 rowmax、sumexp（即 \tilde{l}_{ij} )

所以第 11 行使用 \tilde{l}_{ij} 时需要 rescale 一下，使得每一项的指数部分用的是 m_i^{new}。




2.2、FlashAttention v2

v2 最主要的修改是循环顺序变为：外层是对 Q 逐块遍历，内层是对 K^T 进行逐块遍历。

流程如下图：




外层的每轮遍历之间是互相独立的，就可以分别分配一个 thread block 去执行，以便于提高 GPU 利用率。




另外还有两个小 trick：

第二点说的是保存必要数据，留在 backward pass 的时候用。




具体算法：




3、Backward 流程
3.1、softmax 求导公式

这里唯一麻烦点的是 softmax 求导，所以这里介绍一下～～

设

\begin{cases} y = \text{softmax}(z) \\ L = f(y) \end{cases} ，最终目标是想求 \frac{\partial L}{\partial z} ，这里我们简单假设 L 是标量，y 和 z 都是向量；




（1）求 \frac{\partial y}{\partial z}

先来看看 y = softmax(z) 的求导：（直接贴了 GPT 的回答，清晰明了）：

所以， y 关于 z 求导的结果是一个雅可比矩阵（Jacobian matrix），它的 i 行 j 列元素是 y_i 关于 z_j 的偏导数，

假设 y 的长度为 3，展开后就是这样子：

\begin{align*} \frac{\partial y}{\partial z} &= \text{diag}(\mathbf{y}) - \mathbf{y}^T \mathbf{y} \\ &= \begin{bmatrix} y_1 & 0 & 0 \\ 0 & y_2 & 0 \\ 0 & 0 & y_3 \end{bmatrix} - \begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix} \begin{bmatrix} y_1 & y_2 & y_3 \end{bmatrix} \\ &= \begin{bmatrix} y_1 - y_1^2 & -y_1 y_2 & -y_1 y_3 \\ -y_2 y_1 & y_2 - y_2^2 & -y_2 y_3 \\ -y_3 y_1 & -y_3 y_2 & y_3 - y_3^2 \end{bmatrix} \end{align*}

（2）求 \frac{\partial L}{\partial z}

先看看 L 对 z_j 的偏导数：\frac{\partial L}{\partial z_j} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z_j} = \sum_{i=1}^{n} \frac{\partial L}{\partial y_i} \frac{\partial y_i}{\partial z_j}

记 \frac{\partial L}{\partial y} = [dy_1, dy_2, dy_3] ，

那么就有：

\begin{align*} \frac{\partial L}{\partial \mathbf{z}} &= \frac{\partial L}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{z}} = \mathbf{dy}(\text{diag}(\mathbf{y}) - \mathbf{y} \mathbf{y}^T) \\ &= [\text{dy}_1 \ \text{dy}_2 \ \text{dy}_3] \left( \begin{bmatrix} y_1 & 0 & 0 \\ 0 & y_2 & 0 \\ 0 & 0 & y_3 \end{bmatrix} - \begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix} \begin{bmatrix} y_1 & y_2 & y_3 \end{bmatrix} \right) \\ &= [\text{dy}_1 \ \text{dy}_2 \ \text{dy}_3] \begin{bmatrix} y_1 - y_1^2 & -y_1 y_2 & -y_1 y_3 \\ -y_2 y_1 & y_2 - y_2^2 & -y_2 y_3 \\ -y_3 y_1 & -y_3 y_2 & y_3 - y_3^2 \end{bmatrix} \end{align*}




3.2、backward 标准实现

算法：

FlashAttention v1 附录 B.4 的图




3.3、FlashAttention v1

（这部分内容在论文附录 B.4 中）

主要有两个改动：

1、forward 过程中，没有保存 S 和 P 这两个中间结果，而是保存了 m（row-wise max）和 l （row-wise sum of exponentials in the softmax），用于 backward 阶段中重新计算出 S 和 P；

这么做可以节省很多显存空间～～

（S 直接用 Q、K 计算，Q 和 K 是每一层的输入，本来就是存在显存上；

P 则是用 S 和 m、l 重算）




2、对 dS 进行了公式进行了简化：

完整的算法：




3.4、FlashAttention v2

相对于 v1 ，有 2 个小改动：

1、forward 阶段不是保存 m（row-wise max）和 l （row-wise sum of exponentials in the softmax），而是保存 L（row-wise logsumexp），也是用于 backward 阶段重新计算 S 和 P；

2、把中间值 D 的计算提前了，变成一个大的矩阵计算，效率更高；




算法：（两点改动都圈出来了；下面的算法省略了 mask 和 dropout 步骤）




4、总结

FlashAttention 算是近两年来在大模型训练方面影响力最大的工作之一了，值得好好学习下。




求求点个赞呀！




Reference

[1] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

[2] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

[3] From Online Softmax to FlashAttention

[4] 猛猿：图解大模型计算加速系列：FlashAttention V1，从硬件到计算逻辑

[5] https://twitter.com/fvsmassa/status/1580229170629849089
