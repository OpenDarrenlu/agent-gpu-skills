# Speculative Decoding: 总结、分析、展望（二）Tree Attention 升级之路

**作者**: 笑渐不闻声渐悄​中国科学技术大学  信息与通信工程博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/1908249002421511962

---

原文撰写于 2025 年 6 月
0. Background

书接上文，这篇文章想写一些自己关于 tree attention 的理解。我对于 tree attention 的态度经历了三个阶段的变化：

刚入门 speculative decoding 时，读完 medusa，感觉 tree attention 真是太吊了。作为一个从 graph 领域转到 LLM 的小白来说，重新引入 tree，甚至是 graph 的概念，这真是太 cool 了；
仔细阅读 medusa 和 eagle 的代码之后，感觉 tree attention 的要求实在是有些高。大部分使用 tree attention 的代码都会重写一个 modeling_llama.py 文件，所以我一开始会认为 tree attention 这么复杂，而且只在 batch size 非常小的时候 work，有什么必要呢？
经历多轮审稿、交流之后，我认识到在 spec 应用的场景下，tree attention 是一项很关键的技术。

对于这个子领域而言，我认为是还有很多的深挖的空间的。如果说 draft model 的能力决定了 spec 最终效果的上限的话，那 tree attention 就是决定其均值及下限的关键技术。（这里我会将 adaptive draft length 技术也归于 tree attention 的构造，因为其等价于分叉数为 1 的 tree attention）。另外，tree 的发展是暗含高 batch 场景下 spec 的发展的。（why？）

1. summary

在 decoding 的过程中，draft model 生成 draft token 的过程是链式的，即 a -> b -> C；但是这样做有个显著的问题：由于 token 的序列性以及我们要保证输出的无损性，如果某个 draft token 没有被接受，那它之后的 draft token 都要被丢弃。例如，b 没有被接受，那 C 也会自然的被抛弃。为了解决这个问题，一个很自然的想法就是，对某个位置进行猜测时，我们同时使用多个 draft token 来猜测。例如，a, b, C 都是对第一个位置的猜测。为了实现相同的 parallel verification，我们要对序列 a, b, C 生成一个独特的 attention mask. 对于上面这个例子，其 attention mask 就是一个边长为 3 的单位阵。

从算法层面来说，构造一个 draft sequence，其中某些 token 是对同一个位置的猜测，这个过程等价于构造一棵 draft tree。同样使用上面这个例子，a, b, C 都是对第一个位置的猜测的情况下，draft tree 就是一棵 1-层 3-叉 树。

至此，我们会很自然地问一个问题，这棵 draft tree 应该长什么样？我们应该怎么合理规划这棵 draft tree，以实现更高的 MAT?

2. analysis

事实上，非常多的工作本质上都在回答这个问题，即如何规划 draft tree。但到目前仍然没有一篇文章系统性的研究并回答，这棵 draft tree 到底该长什么样。因此，这里我就抛砖引玉，说一些我自己的分析。如果感兴趣的话，欢迎联系我合作。

draft tree 的容量 D 该如何确定？

到目前为止，我没有看到有 spec 的工作专门研究这个问题。但实际上，这个问题是一个很重要，但经常被忽略的问题。主流的工作都把这个超参数 D 当成一个可以调整的超参数、或者干脆就统一设置为 64，但实际上这种做法是受限的：论文里跑的评测集都是 input length 较小，batchsize=1，上下文接近，输入输出有较高重合度，且大家基本都使用 A100-SXM-80G 来跑实验，与实际应用中的场景差距很大。

在我的实验中，我发现这个超参数 D 对结果的影响还挺大的。定义平均接受长度 MAT 与D的函数 \tau(D) , draft model 生成 D 个 draft token 以及 target model 验证 D 个 draft token 的时间为 v(D)，则确定 draft tree 的容量 D 这个问题可以形式化为一个优化问题：

D=\arg\max\limits_{D} \frac{\tau(D)}{v(D)}\\ 我们直觉上可以认为 \tau(D) 是一个单调不增函数，v(D) 是一个单调不减函数。很遗憾，我们没有足够的信息来描述\tau(D) ，因为其与 draft tree 的具体构造算法相关，我们可以通过蒙特卡洛方法来近似\tau(D)函数。 v(D) 理论上是可以通过 GPU 的具体参数、input length、batch size 来描述的，不过本人缺少对 GPU 底层参数的认知，目前还无法具体描述这个函数。OPT-tree 给的图就很好的刻画了 v(D) 函数中的 verification 部分：

OPT-tree的 v(D) 函数图

当然这个图还不完善，没有考虑 batch size 和 context length。通过分别确定 \tau(D) 和 v(D) 的函数关系，我们可以很好的回答一棵树的容量该如何确定，而不是简单的选取一个常数如 64。对于如何确定 draft tree 的容量 D 这个问题，我的一个思路是 Speculative Decoding: 从 Padding 的视角重新理解投机解码, 从 padding 的角度来分析当前的系统距离 compute-bound 还有多远。

draft tree 的层数 L 该如何确定？

同样的，目前我还很少见到有研究 draft tree 的层数 L 的工作，目前的有关 draft tree 的文章，基本都会设定一个给定层数 depth，并且让 draft tree 生长到指定的层数。但可以预见的是：随着 draft model 能力的提高，预测短距 token 的准确率上升，draft tree 的容量会逐渐向深层倾斜。而层数 L ，决定了 spec 的理论加速上限。

另一方面，spec 是一个很依赖输入的算法。一个例子是，大部分 spec 算法在 code/math 输入上能够取得最大的加速比，而在其他任务上收益相对低。这个性质意味着 optimal draft length 会随着 context 变化而发生很大的变化，可能这一轮的第一个 token 就很困难，也有可能这一轮本可以连续猜对 20 个 token。而与之对应的是，遇到困难 token 时，draft tree 应当收缩到浅层，变得宽而浅；当遇到简单 token 时，draft tree 应当延申，变得窄而深。

现有工作表明，draft model 输出 token 的 confidence (即 pre-norm probability)与最终被接受的可能性呈明显正相关性。那一个很自然的想法就是，当 draft confidence 较高时，收缩这一层 draft tree，少选一些 candidates；直到路径 confidence 累计之和达到一个给定阈值，停止生长 draft tree。本质上，动态调整 draft tree 的层数 L ，就等价于 adaptive draft length, 会有明显的收益。

draft tree 的层宽 K 该如何确定？

目前 EAGLE-2/3 等 dynamic draft tree 方法，都是根据当前 draft confidence 来选取 top-K 个 candidate，导致 draft tree 的每一层都是固定宽度。但实际上，就像之前分析的那样，固定的层宽并不适配所有的情况。举例：输入简单的句子 "the capital of france is"时，draft model 本身就已经足够 confident, 完全没必要把宽度拉高；而输入困难的句子 "quantum computing can"时，draft model 本身没有足够的预测能力来判断语境，那此时应该把当前层的宽度拉高来对抗不确定性，而不是选 top-k 个 draft tokens 来继续生长 draft tree.

这么做有什么好处呢？rethinking 一下 EAGLE 的 draft tree 的生长过程：

draft model 迭代 L 次，得到一棵 宽度为 K , 深度为 L 的 draft tree;
然后再做一次全局 pruning, 按照 path score 剪枝到只剩 D 个节点；

本质上是一个先生成后剪枝的过程。那一个很自然的想法是，借鉴决策树的思想，我们为什么不把这个过程做成一个边生成边剪枝的过程呢？这样的话可以少生成很多的冗余节点。继续思考，以边生成边剪枝作为流程，draft tree 的生长过程可以被修改为：

给定总预算 D ; D_0=1
在第 i 层，draft model 迭代一次，得到 L_i 个节点；(动态宽度)
更新前 i 层的总结点数 D_i = D_{i-1} + L_i ;
当 D_i 达到总结点数 D 时停止生成。

这样，我们只需要考虑第 i 层的节点数L_i。当上下文较为困难时，升高L_i，而上下文简单时，降低L_i。我们就自然得到了一棵可以随上下文语境而动态变化的 draft tree.

对于 draft tree 的某一层，如何确定L_i ？

其实这里的思路很简单，上述的一切弊端都是 top-k 这个 operator 带来的。top-k 对应的这些弊端在 sampling 领域已经被研究了很多了，常见的做法就是替换为 top-p. 那也就是说，我们把 top-k operator 替换为 top-p, 就解决了？

No, No, No. 换成 top-p, 就会出现某一层保留的节点数上千... 也就是我们常说的长尾问题。这个 operator 的核心目的，是从 draft model 不够可靠的 distribution 里面，找到相对更可靠的一部分。top-k 找的数量固定，不合适；top-p 找的数量存在长尾问题，也不合适。一个最直观的想法就是，用 min-p! 即保留当前 distribution 里最大概率 p 乘以一个阈值 \mu 的所有 token.

写到这里，包了那么大的饺子，也是时候掏出这碟醋了：

具体细节大家可以参考一下 paper 的详细描述，有一些复杂。直接上结果：

可以看到，全面优于 EAGLE-3. 值得注意的是，整个方法流程，完全 training-free, 我们所做的，只是修改了 draft tree 的生成过程，让 draft model 真正学会“审时度势”，“量力而行”。

细心的你或许会注意到，很多时候 TALON 的 MAT 并不会高于 EAGLE-3，但总体加速比反而超过。这里我想从偏理论方面给出一个解释。我们知道，speculative decoding 的核心就是 draft model 的准确率和 overhead。那我们可以定义两个指标分别进行量化：

MAT (mean accepted tokens) \tau : 即一轮 speculative decoding 中，能够被接受的 token 数；
draft efficiency \delta : 即一轮 speculative decoding 中，draft model forward 的次数。

那很直观，最理想的情况下，我们应该有 \tau=\delta (暂不考虑 target model 给的 bonus token), 即每次 draft forward, 都会产出 1 个可接受的 draft token. 在此基础上，我们希望 \tau 越大越好。实验中，我们绘制了 TALON 和 EAGLE-3 的这个曲线图：

可以看到，无论实际的 \tau 是多少， EAGLE-3 都要运行固定次数的 draft model forward. 而 TALON 则是在 approaching oracle: 在困难的场景下 ( \tau 比较小)减少 draft model forward 次数 (量力而行)，在简单场景下 ( \tau 比较大) 增大 draft model forward 次数 (追求更高收益)。橙色曲线和蓝色曲线和 oracle 曲线的面积之差，就是 TALON 的收益来源。

值得注意的是，这个思想同样可以引入各种场景下的 speculative decoding，读者们可以自行脑补。

关于 tree attention 的升级之路就先到这，下一期：make retrieval-based speculative decoding great again!
