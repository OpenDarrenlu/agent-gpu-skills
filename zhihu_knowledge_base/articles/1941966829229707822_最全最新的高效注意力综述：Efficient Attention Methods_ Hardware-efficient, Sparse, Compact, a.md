# 最全最新的高效注意力综述：Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention

**作者**: Tete​清华大学  计算机科学与技术博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/1941966829229707822

---

高效注意力方法综述：硬件高效、稀疏、压缩与线性注意力

A Survey of Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention

作者：Jintao Zhang 等

单位：清华大学、UC Berkeley、MIT

原文链接：

论文PDF：

BibTeX引用：

@article{zhang2025efficient,

title={A Survey of Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention},

author={Zhang, Jintao and Su, Rundong and Liu, Chunyu and Wei, Jia and Wang, Ziteng and Zhang, Pengle and Wang, Haoxu and Jiang, Huiqiang and Huang, Haofeng and Xiang, Chendong and Xi, Haocheng and Yang, Shuo and Li, Xingyang and Hu, Yuezhou and Fu, Tianyu and Zhao, Tianchen and Zhang, Yicheng and Jiang, Youhe and Chen, Chang and Jiang, Kai and Chen, Huayu and Zhao, Min and Xu, Xiaoming and Zhu, Jun and Chen, Jianfei}, year={2025}

}




概览

在现代 Transformer 中，注意力（Attention）是唯一的 O(N²) 复杂度的操作，其余模块均为 O(N)。随着语言、视频等生成式模型的序列长度不断增加，提升注意力效率变得尤为关键。近期大量工作从四个方向改进注意力计算效率：

1. 硬件高效注意力：利用 GPU 特性优化实现；

2. 压缩注意力：用权值共享或低秩分解压缩 KV 缓存；

3. 稀疏注意力：跳过不重要的计算；

4. 线性注意力：重排计算顺序，将复杂度降至 O(N)。

本文对以上方法进行了系统梳理：




Hardware-efficient Attention (硬件高效注意力)

在现代GPU上，一个操作的速度会受到计算（使其受限于计算）或内存数据传输（使其受限于内存）的限制。硬件高效的注意力方法通过优化计算执行方式和数据在GPU内存层次结构中的移动直接针对这些瓶颈。

与LLM推理中的两个阶段（prefilling和decoding）相对应，硬件高效注意力可以分为两类：

Prefilling 方法，其灵感来源于 FlashAttention，将 Q、K 和 V 分割成块 Q_i, K_i, V_i 。它们通过以下方式迭代地计算每个输出块 O_i ​：

\begin{aligned} \hat{\mathbf{Q}}, \hat{\mathbf{K}}, \hat{\mathbf{V}} &= \Psi(\mathbf{Q}), \Psi(\mathbf{K}), \Theta(\mathbf{V}). \\ \mathbf{S} = \hat{\mathbf{Q}} \hat{\mathbf{K}}^\top, \quad \hat{\mathbf{P}}& = \Theta (\mathrm{softmax}(\mathbf{S})), \quad \mathbf{O} = \hat{\mathbf{P}} \hat{\mathbf{V}}, \end{aligned}\\

其中 \Psi(\cdot)\text{和 } \Theta(\cdot) 是用于加速计算的预处理函数，例如 SageAttention 中的量化函数。

Decoding 方法：同样将 K \text{ 和 } V 分割成块，但它们的输入 q 是一个向量，因此输出向量 o 的计算方式如下：

\begin{aligned} \hat{\mathbf{K}}, \hat{\mathbf{V}} &= \Psi(\mathbf{K}), \Theta(\mathbf{V}).\\ \mathbf{s} = \mathbf{q} \hat{\mathbf{K}}^\top, \quad \mathbf{p} &= \mathrm{softmax}(\mathbf{s}), \quad \mathbf{o} = \mathbf{p} \hat{\mathbf{V}}. \end{aligned}\\ 其中 \Psi(\cdot)\text{和 } \Theta(\cdot) 是KV 缓存的预处理函数。

我们在论文表2中总结了这些硬件高效的方法。 \Psi(\cdot)\text{和 } \Theta(\cdot) 的类型指的是不同的预处理函数，例如将 KV 缓存分割到 GPU 的 SMs（流式多处理器）上，或者将其重新分配为高效的格式（如 PagedAttention）以提高 I/O 速度。

Compact Attention (压缩注意力)

压缩注意力(Compact attention) 旨在降低大语言模型（LLM）推理过程中 KV 缓存的内存消耗。在多头注意力（MHA）中，我们精确地存储计算时所使用的全分辨率 KV 矩阵，这导致 KV 缓存大小迅速增长。紧凑注意力方法将用于存储的 KV 从用于计算的 KV 中解耦，通过存储压缩后的 KV 状态，并在计算时再将其扩展。与 MHA 相比，这种方法显著减小了存储 KV 的大小，降低了内存使用，同时保持了计算 KV 的大小以防止显著的性能下降。

其通用公式可以表示如下：

\begin{align} q, \mathcal{K}_c, \mathcal{V}_c &= \text{proj}_\mathcal{Q}(x), \text{proj}_{\mathcal{K}_c}(X), \text{proj}_{\mathcal{V}_c}(X). \\ \mathcal{K}, \mathcal{V} &= \text{expand}_\mathcal{K}({\mathcal{K}_c}), \text{expand}_\mathcal{V}({\mathcal{V}_c}). \\ o &= \text{MHA}(q, \mathcal{K}, \mathcal{V}). \end{align}\\ 其中 \mathcal{K} = [K^{(1)}, \dots, K^{(h)}] \in \mathbb{R}^{N \times D}表示 h 个注意力头的键矩阵的拼接，其中K^{(i)} \in \mathbb{R}^{N \times d}代表第 i 个头的键矩阵，且 D = h d。相同的表示法也适用于 q 和 \mathcal{V}。此处的 x \in \mathbb{R}^{D_m}是当前 token 的隐藏状态，X \in \mathbb{R}^{n \times D_m} 是上下文 tokens 的隐藏状态矩阵。\text{proj}(\cdot) \text{ 和 } \text{expand}(\cdot) 分别表示投影函数和扩展函数，而 \text{MHA}(\cdot) 表示多头注意力操作。

我们在论文表3中总结了紧凑注意力方法中每个 token 的 KV 缓存大小、注意力的总参数量以及扩展函数的类型。

Sparse Attention (稀疏注意力)

注意力图 P = \mathrm{Softmax}(QK^\top / \sqrt{d})表现出固有的稀疏性，因为 softmax 操作通常会产生许多趋近于零的值。稀疏注意力方法利用这种稀疏性，通过两个步骤来加速注意力计算。首先，它构建一个稀疏掩码 M ，用以决定是计算还是跳过注意力图 P 中的特定元素。其次，它仅对稀疏掩码 M 对应的部分进行注意力计算。

\begin{align} P &= \mathrm{Softmax}(M + QK^\top / \sqrt{d}). \\ O &= PV. \end{align}\\

其中 M 是一个 N \times N 的矩阵，其元素为 0 或 -\infty。M_{i,j} = 0 指定了注意力分数 Q_iK_j^T 及其对应的输出 P_{i,j}V_j 都需要计算，而 M_{i,j} = -\infty 则表示这些计算应该被跳过。

根据稀疏掩码的生成方式，稀疏注意力方法可分为两大类：

基于模式的方法 (Pattern-based method) 依赖于从经验观察中得出的预定义稀疏模式，其中 M 中-\infty元素的位置遵循固定的几何形状（例如，滑动窗口形状）。
动态稀疏注意力 (Dynamic sparse attention) 在运行时基于一些输入依赖的函数自适应地计算稀疏掩码 M （例如，对于一个阈值 \tau ，如果 \mathrm{pool}(Q_i) \mathrm{pool}(K_j^T) < \tau ，则 M_{i,j} = -\infty，其中 \mathrm{pool}(\cdot) 可以是对 tokens 进行的均值池化）。

我们在论文表4中，根据稀疏注意力方法的稀疏掩码 M （基于模式或动态）是否能减少 KV 缓存存储、是否需要训练模型，以及对语言模型和diffusion Transformer 的适用性进行了总结。

Linear Attention (线性注意力)

线性注意力方法 (Linear attention methods) 通过用核函数 (kernel function) 替换 softmax 函数，将计算复杂度从 O(N^2) 降低到 O(N) 。这允许对矩阵乘法进行重排序，从而避免了对N \times N注意力矩阵的显式计算。对于自回归任务，这些方法可以采用循环的方式进行表述，使用一个在每一步都会更新的固定大小的状态。这使得它们在对超长序列进行推理时非常高效。

Naive Formulation\begin{aligned} H_t &= H_{t-1} + \phi(k_t)^\top v_t \\ o_t &= \phi(q_t)H_t \end{aligned}\\ 不同的计算形式(Computation Forms)

论文中图3展示了线性注意力的三种计算形式。

线性并行形式 (Linear Parallel Form): 该形式通过先计算\phi(K)^T V来计算输出 O，使用的公式为 O=\phi(Q)(\phi(K)^\top V)，其计算复杂度降低至O(Nd^2)。它对于非自回归任务的训练和推理非常高效，因为整个序列是同时处理的。对于自回归训练，则使用掩码（mask）来强制实现因果关系：O = (\phi(Q) \phi(K)^\top \odot M)V。

循环形式 (Recurrent Form)：该形式引入了一个循环更新的固定大小的状态 H_t ：H_t = H_{t-1} + \phi(k_t)^\top v_t。然后，输出通过o_t=\phi(q_t)H_t来计算。在计算时，它首先计算 \phi(k_t)^Tv_t ，然后更新隐藏状态 H_t ，最后计算 o_t=\phi(q_t)H_t。

分块形式 (Chunk-wise Form): 该形式是为自回归训练设计的一种混合解决方案，解决了先前形式的问题。它将序列划分为固定大小的块（chunks），并采用双重策略：在每个块内部，注意力以二次方的并行形式计算，以最大化并行度。因果关系通过在块之间传递循环状态来保持。如图所示，每个块的最终注意力输出是两个不同部分的总和：

块内注意力 (Intra-chunk Attention)：该部分对当前块内的查询、键和值矩阵使用标准的、并行的带掩码自注意力来计算。它捕捉了块内部的局部依赖关系。
块间注意力 (Inter-chunk Attention)：该部分融合了来自所有先前块的历史信息。它是通过将当前块的查询与前一个块传递过来的隐藏状态相结合来计算的。




线性注意力的四种类别

为了使固定大小的隐藏状态 H_t 能够动态地保留最相关的信息，通常会引入了遗忘门（forget gate）和选择门（select gate）。于是，H_t的更新可以公式化为：

H_t = G_f^{(t)} \odot H_{t-1} + G_s^{(t)} \odot \phi(k_t)^\top v_t \\

G_f^{(t)}作为遗忘门，决定保留多少历史信息H_{t-1}；G_s^{(t)}作为选择门，决定保留多少当前信息。其计算过程如论文中图4所示。

线性注意力方法可以根据其隐藏状态的更新方法进行分类。前三种类别依赖于对 H_t 的直接计算：

(1) 朴素线性注意力 (Naive Linear Attention)：不带门控的线性注意力，即G_f^{(t)}和G_s^{(t)}都固定为 \mathbf{1}^\top \mathbf{1}。论文中表5展示了一些典型的朴素线性注意力方法。

(2) 带遗忘门的线性注意力 (Linear Attention with a Forget Gate)：仅选择门G_s^{(t)}固定为 \mathbf{1}^\top \mathbf{1}，而遗忘门G_f^{(t)}是预定义或输入依赖的。论文表6展示了一些典型的带遗忘门的线性注意力方法。表中显示的所有复杂度均为训练阶段的复杂度。

(3) 同时带遗忘门和选择门的线性注意力 (Linear Attention with both Forget and Select Gates)：遗忘门 G_f^{(t)}和选择门G_s^{(t)}都是预定义或输入依赖的，而不是固定为\mathbf{1}^\top \mathbf{1}。论文中表7展示了一些典型的带遗忘门和选择门的线性注意力方法。表中显示的所有复杂度均为训练阶段的复杂度。

(4) Test-Time Training

测试时训练（Test-Time Training, TTT）将隐藏状态H_t视为一组可学习的参数，也称为“快速权重”（fast weights）。TTT 在训练和推理过程中都通过梯度下降来持续更新隐藏状态，如论文中图5所示。这种隐藏状态的更新过程与前三种线性注意力方法不同，因此在本文中我们将其归类为线性注意力的第四种类别。

对于完整论文，请参阅我们的论文: A Survey of Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention




如需引用，请使用以下 BibTeX：

@article{zhang2025efficient,

title={A Survey of Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention},

author={Zhang, Jintao and Su, Rundong and Liu, Chunyu and Wei, Jia and Wang, Ziteng and Zhang, Pengle and Wang, Haoxu and Jiang, Huiqiang and Huang, Haofeng and Xiang, Chendong and Xi, Haocheng and Yang, Shuo and Li, Xingyang and Hu, Yuezhou and Fu, Tianyu and Zhao, Tianchen and Zhang, Yicheng and Jiang, Youhe and Chen, Chang and Jiang, Kai and Chen, Huayu and Zhao, Min and Xu, Xiaoming and Zhu, Jun and Chen, Jianfei}, year={2025}

}
