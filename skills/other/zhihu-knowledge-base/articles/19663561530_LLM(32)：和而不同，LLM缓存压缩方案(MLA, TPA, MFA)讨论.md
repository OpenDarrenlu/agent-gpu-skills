# LLM(32)：和而不同，LLM缓存压缩方案(MLA, TPA, MFA)讨论

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/19663561530

---

​
目录
收起
一、从 MHA 到 MLA
1.1 MHA, MQA 与 GQA
1.2 理解 MLA
1.3 FlashMLA 的高效实现
二、Tensor Product Attention (TPA)
2.1 TPA 基本过程
2.2 TPA 兼容 RoPE
三、Multi-matrix Factorization Attention (MFA)
3.1 MFA 计算过程
3.2 反思 MFA
参考资料

众所周知，在 LLM 推理过程中，KV cache 是最基础、应用最广泛的优化方案，其核心思想即“以空间换时间”。这样就会导致另一个问题：即随着序列长度的增加，KV cache 占用的显存会显著增加，进而制约序列长度的增加和 MFU 的提高。为此也出现了一些优化方法，笔者将其归类如下：

有损压缩派：即不改变模型结构，而通过有损的方式减少 KV cache，具体实现有如下方式：
量化稀疏：即对 KV cache 进行 8 bits，4 bits 甚至 2 bits 量化以减少显存占用，典型方案如 KIVI，QServe
窗口优化：即不计算 dense attention, 只保留部分 cache 进行存储和计算，典型方案如 H2O，StreamingLLM, FastKV
缓存卸载派：即将 GPU 上的缓存 offload 到 CPU 上，典型方案如 OffloadedCache， Prefix Caching
信息共享派：即改变多头机制中 Q, K, V 的一一对应关系，而采用 Q 与 KV 多对一的方式以共享 KV，这也是当前主流预训练模型所采用的方式, 典型方案如 GQA, MQA
分解映射派：即在 attention 机制的计算框架下，通过低秩分解、因子分解等方法仅保留少量的缓存信息，然后通过计算（近似）等价变换为 KV , 典型方案如 MLA, TPA, MFA，本篇将主要讨论该方法
线性改造派：即部分或完全放弃 softmax attention, 而将 attention 机制进行线性化处理，以使缓存量不随序列长度增加而显著增加，典型方案如 RWKV, RetNet, Mamba, MiniMax-01

从 transformer 的角度出发，以上方式的改造的激进程度一次递增，在笔者之前的文章中已比较详细讨论过有损压缩派、信息共享派和线性改造派的内容，在此不予赘述。本篇的内容将主要围绕分解映射派展开。

一、从 MHA 到 MLA

DeepSeek-v2 中提出的 MLA（Multi-head Latent Attention）是基于 transformer attention 结构的一次重要创新，也启发了后来很多重要的工作。本节将从标准 attention 出发，试图深入讨论其设计思路与实现过程。

1.1 MHA, MQA 与 GQA

我们都知道 SDPA (Scaled Dot-Product Attention) 的计算过程 \operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{Softmax}\left(\frac{\mathbf{Q K}^{\top}}{\sqrt{d_k}}\right) \mathbf{V}\\ 其中 \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d_k} , n 为序列长度， d_k 为隐藏维度。

在实际实现时，最常见的是 MHA（Multi-Head Attention）即多头注意力，具体过程如下：

假设计算 \mathbf{Q}, \mathbf{K}, \mathbf{V} 的权重为 \boldsymbol{W}^Q, \boldsymbol{W}^K, \boldsymbol{W}^V \in \mathbb{R}^{d \times d} ，头数为 h ，有 d=d_h \times h ，对于第 i 个头，其权重为 \boldsymbol{W}_i^Q, \boldsymbol{W}_i^K, \boldsymbol{W}_i^V \in \mathbb{R}^{d \times d_h} , 由此可以计算得到 \begin{gathered} \mathbf{Q}_i=\mathbf{X} \boldsymbol{W}_i^Q, \quad \mathbf{K}_i=\mathbf{X} \boldsymbol{W}_i^K, \quad \mathbf{V}_i=\mathbf{X} \boldsymbol{W}_i^V \\ \operatorname{head}_i=\operatorname{Attention}\left(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i\right) \\ \operatorname{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{Concat}\left(\operatorname{head}_1, \ldots, \operatorname{head}_h\right) \boldsymbol{W}^O \end{gathered}\\ 其中输入 \mathbf{X} \in \mathbb{R}^{B \times T \times d} ，则 \mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i \in \mathbb{R}^{B \times T \times d_h} , 在推理过程中，当 token by token 生成到第 t 步时，需要缓存的内容即 \mathbf{K}_i^{\leq t}, \mathbf{V}_i^{\leq t} ~~~~for~ i \in [h] \\

这样每个头都保存了自己的一份 KV ，那么是否可以只保存一份 KV，然后多个Q共享呢？这就是 MQA (Multi-Query Attention)，其计算过程相应变为 \begin{gathered} \mathbf{Q}_i=\mathbf{X} \boldsymbol{W}_i^Q, \quad \mathbf{K}_{shared}=\mathbf{X} \boldsymbol{W}_{shared}^K, \quad \mathbf{V}_{shared}=\mathbf{X} \boldsymbol{W}_{shared}^V \\ \operatorname{head}_i=\operatorname{Attention}\left(\mathbf{Q}_i, \mathbf{K}_{shared}, \mathbf{V}_{shared}\right) \\ \operatorname{MQA}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{Concat}\left(\operatorname{head}_1, \ldots, \operatorname{head}_h\right) \boldsymbol{W}^O \end{gathered}\\ 通过这种方式，MQA直接将 KV Cache 减少到了原来的 1/h ，但是效果也有所损失。因此就有了介于 MHA 和 MQA 之间的 GQA（Grouped-Query Attention）, 其计算过程可描述为 \begin{gathered} \mathbf{Q}_i=\mathbf{X} \boldsymbol{W}_i^Q, \quad \mathbf{K}_{g(i)}=\mathbf{X} \boldsymbol{W}_{g(i)}^K, \quad \mathbf{V}_{g(i)}=\mathbf{X} \boldsymbol{W}_{g(i)}^V \\ \operatorname{head}_i=\operatorname{Attention}\left(\mathbf{Q}_i, \mathbf{K}_{g(i)}, \mathbf{V}_{g(i)}\right) \\ \operatorname{GQA}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{Concat}\left(\operatorname{head}_1, \ldots, \operatorname{head}_h\right) \boldsymbol{W}^O \end{gathered}\\ 即将所有Head分为 g 个组（ g 可以整除 h ）当 g=h 时就是MHA， g=1 时就是MQA，当 1<g<h 时，它只将KV Cache压缩到 g/h ，压缩率不如MQA，但同时也提供了更大的自由度，效果上更有保证。

1.2 理解 MLA

在 MQA 与 GQA 中，可以看到对 \mathbf{K}_i, \mathbf{V}_i \in \mathbb{R}^{B \times T \times d_h} 的缓存优化都是在发生在 head 维度，而开篇提到的窗口优化派则是在 T 的维度进行优化，那么是否可以在 d 的维度进行优化呢？本质上 MLA（Multi-head Latent Attention）就是这么做的，接下来我们研究下其具体做法。

1）Q向量

在DeepSeek-V2中，Q向量也采用了低秩压缩的方式。首先，将输入向量投影到一个 d_{cq} ( DeepSeek-V2中该值为1536, d_{cq} \ll d )维的低维空间： \mathbf{C}^Q =\mathbf{X} \boldsymbol{W}^{D Q} \in \mathbb{R}^{B \times T \times d_{cq}}, \quad\left(\boldsymbol{W}^{D Q} \in \mathbb{R}^{d \times d_{cq}}\right) \\ 然后，将其投影到 \mathbb{R}^{H \times d_h} 的多头向量空间上（其中 H=128 是heads数， d_h=128 ），得到了Q向量的第一部分： \operatorname{Concat}\left(\mathbf{Q}_1^C, \mathbf{Q}_2^C, \ldots, \mathbf{Q}_h^C\right)=Q^C =\mathbf{C}^Q \mathbf{W}^{UQ} \in \mathbb{R}^{B \times L \times H \times d_h}, (\mathbf{W}^{UQ} \in \mathbb{R}^{d_{cq} \times d})\\ 再将其投影到 \mathbb{R}^{H \times d_h^R} ( d_h^R=64 )上并使用RoPE嵌入位置信息，得到Q向量的第二部分： \operatorname{Concat}\left(\mathbf{Q}_1^R, \mathbf{Q}_2^R, \ldots, \mathbf{Q}_h^R\right)=\mathbf{Q}^R =\operatorname{RoPE}\left(\mathbf{C}^Q \boldsymbol{W}^{Q R}\right) \in \mathbb{R}^{B \times L \times H \times d_h^R}, \quad\left(\boldsymbol{W}^{Q R} \in \mathbb{R}^{d_{cq}\times d_h^R \times H}\right) \\ 将两部分拼接的到最终的Q向量： \mathbf{Q} =\operatorname{Concat}\left(\mathbf{Q}^C, \mathbf{Q}^R\right) \in \mathbb{R}^{B \times L \times H \times (d_{h}+d_{h}^R)} \\ 2）KV 向量

计算KV向量时，首先需要将输入向量投影为 d_{ckv} (512)维的联合压缩表示： \mathbf{C}^{K V} =\mathbf{X} \boldsymbol{W}^{D K V} \in \mathbb{R}^{B \times L \times d_{ckv}}, \quad\left(\boldsymbol{W}^{D K V} \in \mathbb{R}^{d \times d_{ckv}}\right) \\ 与Q向量的计算过程类似，K向量的第一部分是将 C^{KV} 通过投影解压缩到 \mathbb{R}^{H \times d_h} 的多头向量空间： \operatorname{Concat}\left(\mathbf{K}_1^C, \mathbf{K}_2^C, \ldots, \mathbf{K}_h^C\right)=\mathbf{K}^C =\mathbf{C}^{K V} \boldsymbol{W}^{U K} \in \mathbb{R}^{B \times L \times H \times d_h}, \quad\left(\boldsymbol{W}^{U K} \in \mathbb{R}^{d_{ckv} \times d}\right) \\
