# [EMNLP'23] Simple Hardware-Efficient PCFGs with Independent Left and Right Productions

**作者**: 张宇月之暗面

**原文链接**: https://zhuanlan.zhihu.com/p/663319816

---

TL;DR

EMNLP23的一篇工作，优化之前的Low-rank PCFGs.
简单来说，作者们提出的SimplePCFG将Production 
𝐴
→
𝐵
𝐶
分解为左右独立的 
𝐵
←
𝐴
 和 
𝐴
→
𝐶
（左右不共享，see Footnote2）.
在SimplePCFG的基础上，作者提出了FlashInside，相比naive impl.，Speed和Memory effiency都有几十倍的提升.
SimplePCFG在Unsupervised Parsing (SOTA) & Language Modeling上取得了很好的结果.[1]

Simple Hardware-Efficient PCFGs with Independent Left and Right Productions
arxiv.org/abs/2310.14997/
SimplePCFG

设计了如下的递归公式，可以看到的是 
𝜋
𝐴
→
𝐵
𝐶
被分解成了 
𝜋
𝐵
←
𝐴
 以及 
𝜋
𝐴
→
𝐶
，相应的，就可以和各自左右子span的inside probabilities结合了，即 
𝛽
𝑖
𝑘
 以及
𝛽
𝑘
𝑗
，最后得到 
𝛽
𝑖
𝑗
.
通过这样子的展开，我们可以cache孩子span的结果，避免重复计算.

FlashInside

这里包含了非常多优化PCFG的trick，通用性很强.

Span-level Parallelism:

对于宽度为 
𝑤
的span，我们可以并行计算 
𝛽
𝑖
(
𝑖
+
𝑤
)
,
𝛽
(
𝑖
+
1
)
(
𝑖
+
𝑤
+
1
)
,
𝛽
(
𝑖
+
2
)
(
𝑖
+
𝑤
+
2
)
等等. 然后继续自底向上增长宽度到 
𝑤
+
1
,
𝑤
+
1
,
…
,
𝑁
.
这个算是well-known了[2].

logeinsumexp
 Trick:

我们知道为了避免溢出，会广泛使用 
logsumexp
 这个trick，即 
exp
 时减掉一个最大值，然后在 
log
 外面加上这个最大值 
𝑥
⋆

但是很遗憾上式 
𝑎
𝑖
𝑘
+
𝑏
𝑘
𝑗
 这个操作是非常糟糕的，为了后面的反向传播，这里会留下一个 
𝑁
×
𝑁
 的中间结果， 
𝑁
 通常来说很大，这对于memory footprint很不利.
作者们利用了一个所谓的 \texttt{logeinsumexp} trick[3]来解决这个问题，motivation很简单：

不希望留下上面那么大的中间结果；
我们希望尽可能用multiplication算子来搞定，因为他们被高度优化过；
既如此那何不直接计算 \mathrm{exp}(a_{ik}) \cdot \mathrm{exp}(b_{kj})呢？只要能搞定数值稳定性问题；

解决办法确实存在！首先处理 a_{ik} 和 b_{kj} 保证他们不会溢出不就好了，式子如下（下面两个式子下标写作 {ik} 和 {kj} 应该更合适，其中 \mathbf{L},\mathbf{R}\in \mathbb{R}^{\mathcal{N}\times\mathcal{N}} 是Production weights）.

后面我们直接带入新的 a_{ik} 和 b_{kj} 到式(1)，其他减掉最大值的trick和 \texttt{logsumexp} 类似（also see this blog for gradient derivations and cuda impls）.
如果忽略掉那些为了数值上稳定的操作，最大的区别在于和a_{ik}以及 b_{kj} 相关的+\rightarrow\exp转变成了 \exp\rightarrow\times ，丢弃掉了中间变量存储（后面backprop的时候再recompute），因此非常memory-efficient.

Kernel Fusion:

算法里面element-wise操作很多，比较memory-bounded，因此作者一口气用kernel把他们全都fuse掉了

Recomputation:

仍然是memory concerns，作者丢弃掉了很多中间结果，在back propagation的时候重计算了（a.k.a. gradient checkpointing）.
这里和 \texttt{logeinsumexp} 其实是一体两面.

结果

下表是这些trick的ablations，可以看到Nonterminals 512时， \texttt{logeinsumexp} 在memory方便相比naive impl.的提升十分惊人，速度也提升了5倍左右.
Kernel Fusion在 \texttt{logeinsumexp} 基础上进一步带来了1倍的加速，同时只有1/3的memory占用.

其他见论文，略.

总结

非常cool的FlashInside，对PCFGs极致的优化，效率比TorchStruct & SynJax这两个baseline要优越很多.
Learnt a lot from it（ @sonta 和 @彼德V 的工作总是很好玩的～）.

参考
^Wei Liu and Songlin Yang and Yoon Kim and Kewei Tu. 2023. Simple Hardware-Efficient PCFGs with Independent Left and Right Productions. In Findings of EMNLP. https://arxiv.org/abs/2310.14997
^Youngmin Yi and Chao-Yue Lai and Slav Petrov and Kurt Keutzer. 2011. Efficient Parallel CKY Parsing on GPUs. In Proceedings of ICPT. https://aclanthology.org/W11-2921
^Robert Peharz and Steven Lang and Antonio Vergari and Karl Stelzner and Alejandro Molina and Martin Trapp and Guy Van Den Broeck and Kristian Kersting and Zoubin Ghahramani. 2020. Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic Circuits. In Proceedings of ICML. https://proceedings.mlr.press/v119/peharz20a.html
