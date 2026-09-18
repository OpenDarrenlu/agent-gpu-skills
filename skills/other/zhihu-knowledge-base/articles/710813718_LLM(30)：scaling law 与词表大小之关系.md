# LLM(30)：scaling law 与词表大小之关系

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/710813718

---

​
目录
收起
一、词表大小对模型训练得影响要素及分析
1.1 词表大小与数据压缩比
1.2 词表大小与损失函数
二、最佳词表大小的估算
2.1 实验法
2.2 导数求极值法
2.3 拟合损失函数
2.4 三种方法的比较与当前模型比较
参考资料

在之前的文章中，笔者比较简略讨论过 scaling law 与词表大小的关系，当时的主要结论如下：

词表大小决定了信息熵
H
(
P
)
的上界。词表的增大，会导致 loss 的增大。
当模型不变时，适当增加词表会提高编码率及模型性能，但当词表过大时，会导致训练不充分，反而降低模型性能

以上的分析比较定性，缺少理论和实验依据。因此本篇将借助 Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies 的工作进一步深入讨论这一问题。

一、词表大小对模型训练得影响要素及分析
1.1 词表大小与数据压缩比

词表大小的最直接影响就是分词效率，即数据压缩比，如果字符数(characters)为 H ，token 数为 D ，则压缩率(compression ratio) 为 CR = D/H \\ 下表展示了部分 BPE 分词的词表大小与不同语言的压缩率的关系。

关于词表和压缩率的关系，可以拟合成如下公式： f(V)=a \log ^2(V)+b \log (V)+c\\

其中整体误差较小的系数组合是 =0.0064, =−0.1581, =1.2047 。

需要说明的是：

该拟合过程只使用单一数据集，仅调整词表大小查看生成的 token 数，不涉及多语言及多数据集，因此压缩程度比实际训练的分词器要高；
不同类型的分词器的压缩效率不同，如上图所示，BPE 和 Unigram 分词器的效率接近，且明显高于 word-base 分词器；
压缩率随着词表大小的增加先快速减小，进而逐渐平缓，这是因为当词汇量足够大时，训练语料库中的单词已经可以被词汇表有效地覆盖。
1.2 词表大小与损失函数

词表大小对损失函数的影响并不那么直接，对此笔者在引文[1]的 1.1~1.3 节进行过比较充分的讨论，主要结论如下：

词表大小决定了信息熵的上限，即 H(P)\leq log(V) ，词表越大，信息熵的期望越大；
词表越大，计算出来的概率分布就越分散，即 D_{K L}(P \| Q) 越大，进而导致损失变大。

为了使结果在同一评价标准下，需要对损失函数进行修正，

标准的损失函数的定义为 \mathcal{L}=-\frac{1}{T} \sum_{i=1}^T \log p\left(w_i \mid w_{1: i-1}, V\right)\\ 修正后的损失函数为 \mathcal{L}_u=-\frac{1}{T} \sum_{i=1}^T \log \frac{p\left(w_i \mid w_{1: i-1}, V\right)}{p\left(w_i \mid V\right)}\\ 其中 p\left(w_i \mid V\right) 表示单词 _ 在分词后语料库中的频率，分词器的词表大小为 V 。需要说明的是：

对于给定模型，归一化损失\mathcal{L}_u 在不同词汇量的情况下保持一致， \mathcal{L}_u 的区别来自语言模型本身的能力 (证明过程见引文[4] 2.3节)
与 \mathcal{L} 相比， \mathcal{L}_u 的值小得多，并且可能是负的
与BPC相比，BPC表示语料库中原始的每字符语言模型损失， \mathcal{L}_u 相当于通过每个字符的频率归一化的每字符语言模型损失
二、最佳词表大小的估算

当模型参数量为 N ，token 数为 D 时，总的计算 FLOPs 可以估算为 C \approx 6 N D \approx 6\left(N_{\mathrm{nv}}+V d\right) H f(V) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ (1) \\ 其中 N=N_{\mathrm{nv}}+N_{\mathrm{v}} ， N_{\mathrm{nv}} 表示非词表参数， N_{\mathrm{v}} 表示词表参数， D=H f(V) 。

即然词表大小对训练过程有明显影响，那么该如何确定最佳的词表大小呢？这里提供了 3 种方法。

2.1 实验法

即延续 scaling law 的论文的思路通过大量实验来发现其中的关联关系，具体实验设置如下：

共设置了 7 组模型， N_{\mathrm{nv}} 从 33M 到 1.13B。每组中词表大小从 4K 到 96K，研究在相同 FLOPs 下的结果。

为每个 FLOPs 选择最小 \mathcal{L}_u 的数据点，该点是对 （N_{\mathrm{nv}}， N_{\mathrm{v}}， H） 的计算最优分配，根据scaling law 的结论，可以假设 N_{\mathrm{nv}}=k_1 C^{\alpha_1}, N_{\mathrm{v}}=k_2 C^{\alpha_2} , H=k_3 C^{\alpha_3} 进行拟合

拟合后的结果显示： N_{\mathrm{nv}}=0.08 * C^{0.50} ， N_{\mathrm{v}}=0.20 * C^{0.42} ，H=6.42 * C^{0.50} 据此可以得到以下几点结论：

LLMs 是数据饥饿型的。相比于增加模型参数，增加数据效率更高；
N_{\mathrm{v}} 与 FLOPs 也呈幂律关系。随着模型变得越来越计算密集，更大的词表增强了模型理解更多样化文本的能力，因此词汇表的大小对 scaling 至关重要；
N_{\mathrm{v}} 的缩放速度慢于 N_{\mathrm{nv}}
2.2 导数求极值法

在此之前，当我们考虑到计算预算的时候，意识里就会想到模型参数和训练 token 数，那么词表大小除了很小程度影响模型参数外，二者是否还有更多深刻的联系呢？上一节通过实验的方法，得到了二者之间的幂律关系，这种关系表示了二者之间相对变化的关系，那么该如何找到最佳词表大小呢？本节将通过导数的方法研究二者的关系。

对（1）式求导，可得 \begin{aligned} \frac{\partial C}{\partial V}= & \frac{\partial}{\partial V}\left[6\left(N_{n v}+d V\right) H(f(V))\right] \\ = & \frac{\partial}{\partial V}\left[6\left(N_{n v}+d V\right) H\left(a(\log (V))^2+b \log (V)+c\right)\right] \\ = & 6 H\left[\left(N_{n v}+d V\right) \frac{d}{d V}\left(a(\log (V))^2+b \log (V)+c\right)\right. \\ & \left.\quad+\left(a(\log (V))^2+b \log (V)+c\right) \frac{d}{d V}\left(N_{n v}+d V\right)\right] \\ & =6 H\left[\left(N_{n v}+V d\right) \frac{2 a \log (V)+b}{V}+\left(a(\log (V))^2+b \log (V)+c\right) d\right] \end{aligned}\\ 那么 \frac{\partial C}{\partial V}=0 时，C 取最小值，使用数值搜索找到最优值点如下：

值得注意的是，在最佳词汇表大小 时，该模型在给定预算下花费了最大数量的训练字符。这恰恰说明，在最优词表的情况下，相同的预算可以训练更多数据。

需要说明的是，词表大小本质上由 \mathcal{L}_u 决定，而非由计算预算 C 决定。而在计算分配最优时，损失\mathcal{L}_u与计算预算 C呈现幂律关系（即 scaling law），同时计算预算 C 与词表大小也呈现幂律关系（即方法1），那么则可以建立一个计算预算 C 与词表大小的关系。

具体而言，针对不同的非词汇参数 N_{nv} 获得一组导数最优词汇参数 N_v ，表示为 \{(N_{nv}^i,N_v^i)|i=1,⋯, \} 。然后使用幂律函数 N_v∝N_{nv}^\gamma 拟合 N_{nv} 和 N_v 之间的关系。这导致缩放方程： N_v/N_v^0=(N_{nv}/N_{nv}^0)^\gamma 其中 N_{nv}^0 是相对较小的模型（例如，33M），而 N_v^0 是具有相同 FLOPs 预算的足够训练字符的搜索到的最佳词汇表参数。 通过将从导数获得的 值与小模型上的经验解相结合，可以估计任何大模型的最佳词汇量，如下： N_{\mathrm{v}}^{\mathrm{opt}}=N_{\mathrm{v}}^0 *\left(\frac{N_{\mathrm{nv}}}{N_{\mathrm{nv}}^0}\right)^\gamma\\ 其中 \gamma = 0.83 是一个较好的经验值。

2.3 拟合损失函数

在给定非词表参数 N_{nv} 、词表参数 N_v 和训练字符量 H 的情况下，直接预测损失。然后，最佳的词表配置可以通过找到最小的损失点\mathcal{L}_u相对于词表来预测。此时损失函数可表示为： \mathcal{L}_u=-E+\frac{A_1}{N_{n v}^{\alpha_1}}+\frac{A_2}{N_v^{\alpha_2}}+\frac{B}{[H f(V)]^\beta} ~~~~~~~~~~~~~~~~~~~~~~~(2)\\ 其中 E, A_1, A_2, B, \alpha_1, \alpha_2, \beta 是学习参数。

可以对（2）进行求导，并搜索最优值，即 \begin{aligned} \frac{\partial \mathcal{L}_u}{\partial V} & =\frac{\partial}{\partial V}\left(\frac{A_2}{(V d)^{\alpha_2}}\right)+\frac{\partial}{\partial V}\left(\frac{B}{\left(\frac{C}{6\left(N_{n v}+V d\right)}\right)^\beta}\right) \\ & =-\alpha_2 \frac{A_2 d}{(V d)^{\alpha_2+1}}+\beta \frac{B \frac{C d}{6\left(N_{n v}+V d\right)^2}}{\left(\frac{F}{6\left(N_{n v}+V d\right)}\right)^{\beta+1}} . \end{aligned}\\其代码实现如下：

def dl_dv(V, Nnv, d, F):
    term1 = 0  # Derivative of -E
    term2 = 0  # Derivative of A1/[Nnv]^alpha1
    term3 = -alpha2 * A2 * d / (V * d) ** (alpha2 + 1)
    u = F / (6 * (Nnv + V * d))
    du_dV = F * d / (6 * (Nnv + V * d) ** 2)
    term4 = beta * B * du_dV / (u ** (beta + 1))
    
    return term1 + term2 + term3 + term4 

也可以对（2）式进行最优化变换，即 \begin{gathered} \min _{a_1, a_2, b, e, \alpha_1, \alpha_2} \operatorname{Huber}_\delta\left(-\exp (e)+\exp \left(a_1-\alpha_1 * \log \left(N_{n v}\right)+\exp \left(a_2-\alpha_2 * \log \left(N_v\right)\right.\right.\right. \\ \left.+\exp (b-\beta * \log ([H f(V)])), \quad \mathcal{L}_u\right) \end{gathered} \\ 其中 A_1=\exp \left(a_1\right), A_2=\exp \left(a_2\right), B=\exp (b), E=\exp (e) ，其代码实现如下：

def LSE(params, Nnv, V, d, H):
    # fit a, b, e, alpha, beta, where A = exp(a), B = exp(b), E = exp(e)
    a1, a2, b, e, alpha2, beta = params
    alpha1 = beta
    term1_1 = a1 - alpha1 * np.log(Nnv)
    term1_2 = a2 - alpha2 * np.log(V*d)
    term2 = b - beta * np.log(H_to_D(H,V))
    term3 = e
    return (np.exp(term1_1) + np.exp(term1_2) + np.exp(term2) - np.exp(term3))

最终得到的是\mathcal{L}_u与 V 的关系曲线如下：

2.4 三种方法的比较与当前模型比较

以上的三种方法得到了类似的结论，即最优词表大小与计算预算相关，且是可以计算的。以下是三种方法计算得到的最优词表大小值：

根据这一结论可以反观当前的主流开源模型，除了Gemma2-9B之外，所有模型分配的词表参数都小于预测的最佳词表参数。

最后研究最佳词表大小如何随着不同数量的训练数据而变化的趋势。只改变数据量，但保持非词表参数固定。词表大小的选择有8⁢K、10⁢K、16⁢K、24⁢K、32⁢K 和48⁢K。以 N_{nv}=302⁢M 为例，当可用数据是瓶颈时，最佳词表大小根据经验减小（防止过度拟合）。相反，当训练大量数据时，最优词表大小增加。

左图说明了所有词表选择中的最佳词表大小如何随着训练数据而变化。非词表参数是固定的N_{nv}=302⁢M。左图中的每个单元格表示给定一定FLOP预算的损失，以进行公平评估，颜色强度指示损失值。带标记的黑线表示每个FLOPs 预算的最佳词汇量，它基本上随着训练数据数量的增加而增加。右图为给定一定的 FLOPs 预算，不同词表大小的训练 token 数略有不同。为了保持 FLOPs 的一致性，具有较大词表大小的模型在较少的 tokens 上进行训练。




参考资料

[1] LLM（廿六）：从信息论的角度解释 scaling law - 知乎 (zhihu.com)

[2] https://arxiv.org/pdf/2407.13623

[3] https://arxiv.org/pdf/2401.12246

[4] arXiv reCAPTCHA

且莫辞沉醉，听取阳关彻。念故人，千里至此共明月。 —— 寇准《踏莎行·寒草烟光阔》
