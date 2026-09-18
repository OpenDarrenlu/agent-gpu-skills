# Speculative Decoding: 投机采样只能复用Hidden States吗？

**作者**: 笑渐不闻声渐悄​中国科学技术大学  信息与通信工程博士在读

**原文链接**: https://zhuanlan.zhihu.com/p/2036027467606012834

---

写在前言: 主播又来给大家带来投机采样领域的最新进展了！这篇文章会简述最近一段时间我在投机采样领域的“失败”经验与探索，并且结合Gemma 4与最新的DFlash 给出一些analysis. 本文纯手写，AI 含量 0%，可以放心阅读。由于内容较多，可能有一些地方不够清晰，欢迎大家在评论区交流。

When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?
arxiv.org/pdf/2604.26412
Intuition: 复用 hidden states vs 复用 kv cache

先问读者们一个问题：投机采样领域广为人知的几个方案：EAGLE-3 / MTP 都是在复用 hidden states. 但是为什么是hidden states？

早在去年5月份，我就在文章Speculative Decoding: 总结、分析、展望 中提出了一个模糊的观点：把复用的特征从hidden states层转移到kv cache层可以缓解长距衰减。受限于资源问题 (我一度只有2块GPU可用)，一直没有去探索这个问题。今年1月份我重新思考了这个观点，并总结到了 Speculative Decoding: 长距衰减问题不仅源于训推不一致，于是在Qwen C端开启了复用KV Cache的艰难探索之路。

复用KVCache的想法最早是由CunXiao哥 @真实哥 的Glide with a Cape提出, 后续的followup工作还有鹏辉哥的 LongSpec。这两篇paper的时间都早于我对复用kvcache的探索，给了我很多启发，solute!

这里只简要说明一下我认为复用 KV Cache可以缓解长距衰减的Intuition，具体分析可以check我们的paper.

设想 target model 在推理 x_1, x_2, x_3 , 并把 x_3 对应的 hidden states h_3 传给了 draft model 去做复用。我们知道 h_3 是前3个token的kvcache的加权聚合，那自然就会有一种可能性：target model 给 x_2 的 value v_2 分配的 attention weight 非常小 \alpha_2 = 0.01 , 这表示对于当前的 next-token prediction 来说，target model 认为 x_2 的信息不重要。draft model 获取了 h_3 之后推理下一个token。这也就是说：draft model 预测下一个 token时，继承了 target model 上一轮的 attention bias. 由于 transformers的特性，相邻 token 之间的attention weights变化通常比较小，所以对于短距离的第1个draft token效果会很好. Qwen 3.5 的 MTP在第一个位置的接受率可以训到 0.85.

但是随着预测距离的变长：来自 target model 的 attention bias 还是有益的吗？如果预测第3个token需要 x_2 的信息，draft model 能否从 h_3 中提取 x_2 的信息呢？很可惜，这是一个非常困难的事情—— h_3 几乎不包含 x_2 ，又怎么提取呢?

从上述这个例子我们可以看到，复用 hidden states 会携带 target model 在当前position的attention bias; 这种 attention bias 对于短距离的预测有帮助，但是对于长距离的预测反而有害。

其实从这个角度也可以理解EAGLE-3为什么要用多层 hidden states: 因为不同层的hidden states通常带有不同的 attention bias; 最后一层的 h_3 或许不含 x_2 的信息，中间层/浅层的 h_3 很可能就包含了。

那换个角度想想看，如果我们复用kvcache，是不是就不存在这种attention bias了呢？当draft model 复用了target model 的kvcache，它只要产出一个质量足够好的 query, 就可以得到和 target model 完全相同的attn output.

复用 KVCache 栽的第一个跟斗

复用kvcache 和复用 hidden states的一个最大差别就是：复用的target kvcache 需要直接inject到draft model的attention计算过程之中，而不是concat在input embedding之后。基于此，我们可以设计一个最简单版本的模型：

如图所示，我们抽取 4 层 target kvcache，然后直接 inject 到draft model的attention过程中，相当于把self- attention改为了cross-attention。这里 4层kvcache的injection方案有两种：

把4层target kvcache concat起来，用一个 linear 来降维到 1层；
把4层target kvcache 在 heads这一维度拼起来，然后把draft model的num kv heads 翻4倍。

结果很不理想，这里方案 1的效果要优于方案 2，但都远逊于 EAGLE3 baseline.

我们进一步做了很多实验，包括修正 linear 的 rope项 (4层kvcache通过 fc mixing 之后rope会被破坏，解决方案时重新apply rope); 修正rope的index (inject kvcache的数量会比 hidden states 数量多1，因为 hidden states 和 input embedding 对齐需要 offset 1)，发现都没有足够好的效果.

难道是kvcache的信息质量没用？我们补充了不 inject target kv (对应上图的No target info), 完全由 draft model 自己生成kv，相当于一个独立的单层 llm作为draft model，效果又远逊于 inject target kv.

经过一通分析之后，我们找到了第一个关键原因：

injection 范式下，query estimate 的难度比想象中的高。如果 draft model 想采用 injection 范式，则 draft model必须scale到多层。因为当 draft model 仅为单层是，query 是 input embedding 的线性变化。也就是说，draft model 生成的 query 是上下文无关的，这极大损失了 draft model 的 query质量，无法正确的 re-attention 到target kvcache. 这个分析验证了一个重要性质：injection-kv 范式的方法必须要 scale 到多层。所以我们会看到，DFlash 的模型是多层的，Gemma 4的 MTP 也是多层的。

那很自然的想法就是，我们来 scale draft layer 看看结果:

可以看到 2 层的模型效果相比于 1 层发生了一次跃升，但后续的 scaling 层数提升就变小了。但 2 层的 KV-only reuse 模型也只相当于 1 层的 hidden-only reuse (EAGLE)，那我们多付出的这一层开销，谁给我们补？

没办法，只能继续去思考解决方法。我们想到，为什么 EAGLE 不是必须得 2 层呢？因为 EAGLE 的输入本身就有 hidden states 这一信息，天然就是 context-dependent的. 那我们如果在复用 KV 的同时，再把 hidden states也输入进来，是不是就不用 scale 到多层了呢？这里又犯难了，两份信息，该如何恰当的融合？

从 KV-only Reuse 到 Hybrid-Reuse

如果直接简单的加起来，由于训练过后的 EAGLE ckpt已经学会用 hidden states信息，但没有学会用 KV信息，那模型大概率会直接忽略 KV 的信息. 经过一些intuition和拍脑袋，我们想出了用 gated delta rule 来将 KV 的信息更新到 hidden states信息里的这么一种结构 hybrid-reuse: (吹一手 yuhao @沐风rs , 画图的神!)


简单描述一下就是：在原有 EAGLE-3 的 hidden states 通路 (Self-Attention) 的基础上，增加一条 Cross-Attention 分路，并通过 gated delta rule 来将 cross-attention 分支更新到 self-attention分支. 具体公式参考论文。话不多说，直接上结果:


这里可以看到，hybrid 结构随机初始化或是从已有的 EAGLE-3 ckpt 初始化，都取得了比 EAGLE-3 train from scatch 更好的效果，且长程衰减更少。为了验证这种设计的有效性，我们还额外做了一组 cross-only 实验：即没有 self-attention分路，只走 cross-attention 分路。结果显而易见的差，基本持平或劣于 EAGLE-3 train from scratch. 到这里，第一次取得了 positive results，也是整个实验周期里唯一一次取得的positive results.

拿着这个design, 我去 Qwen 3.5 MTP上去做了 post-training (事实上，这才是我整个工作周期的主要内容). 可能大家对这个会比较感兴趣，但由于这里的实验结果比较混乱，没有整理出来，所以就没有放出来。总而言之，这一套 design 在 Qwen 3.5 MTP 的post-training上没有取得预期的效果。于是又回过头来 rethinking 这一套 design 为什么在 EAGLE-3 上 work 了. 然后我发现，整个实验有很多经不起推敲的点需要补上：

training data 没有从 target model regenerate, 而是直接用的 ShareGPT 过了 3 个 epoch.
数据量太少了，原版 EAGLE-3 用了 ShareGPT + UltraChat, 大约是纯 ShareGPT的 4 倍数据量;
没有跑 end-to-end 的实际推理，没有 profile 出真实推理场景的 MAT;
没有跑对比的 EAGLE-3 train from ckpt 实验;

于是按照上述几个 concern:

scaling training data 到ShareGPT + UltraChat.
用 target model regenerate 整个 training data;
补充 huggingface 格式的推理代码；
增加对比实验 EAGLE-3 train from ckpt

然后发现结果就悲剧了:

这里 Hybrid train from scratch 的结果疑似有 bug，就没有放上来. Huggingface 的推理代码具体 tok / s也没有放出来, 因为 huggingface 的推理速度本来就不是很可信，只有 MAT 数字比较可信。

复盘

仔细回想起来，整个复用 KV 的理论存在两个方面的问题：

query estimation 的难度被低估了;
draft model architecture里面的 KV proj 的梯度是一个很大的问题;

query estimation 的难度自不用说. target model 的 query 来自多层 transformer layer的聚合，并不是简单的一层 layer 就能拟合的；这个问题对于 EAGLE 也成立，本质上是 draft model 是单层结构，参数量小，capacity 受限。

最要紧的是 draft model 本身 KV proj 的梯度问题。这个有点难以理解：不是都复用了 target model 的 KV cache 了吗，为什么还需要 draft model 的 KV proj 呢？

因为在推理过程中，draft model 并不是只推 1 个 token；当 draft token 自回归的往后推理时，新出现的 token 没有 target model 的 kvcache，这种时候 draft model 必须得用自身的 KV proj 去生成一份 KV cache.

而在 draft model 训练时，这一部分 KV proj 的梯度尤为稀疏：整个 prefix 的 KVcache 都来自于 target model; 只有做 TTT 时产生的新 token 的 KV cache 来自 draft model 本身. max length=4096，TTT=7 时，这个比例回来到 4096: 7, 可以看到 KV proj 的梯度是尤为稀疏的.

即使把梯度放大 50 倍，仍然不会有很显著的效果。本质原因仍然是 KV梯度的稀疏性。

基于KV 梯度的稀疏性这个问题，会引出另外一个问题——cross-attn 分支信号过差，gate 门控难以打开：

可以看到，训练初始，gate 会被迅速压到趋近于 0，因为 cross-attn 的贡献非常少，优化器认为砍掉 cross-attn 分支是梯度下降最快的方向。而由于 gate 被关闭，流经 cross-attn 的梯度会更少，形成一个难以优化的负循环。直到训练末期，gate 也只能打开到 0.1左右。

Summary 一下，复用 KVcache 这条路，在现有的 TTT 路线下很难训出来，难度主要集中在如何充分训练 draft model 上.
从KVShot 的视角来看 DFlash 和 Gemma 4 MTP

首先是一个很有意思的问题：DFlash 为什么可以 work?

因为上面的两个问题 DFlash 都巧妙的解决了。

Query Estimation 能力不够 -> DFlash 增加到 5 层甚至是 8 层；

KV 的梯度太稀疏了 -> block-wise training:

一次训练，mask token 的梯度很充分，不存在 TTT 这里的梯度稀疏问题。

说实话，DFlash 正好解决了这两个问题，让我觉得他们可能也踩过我这里踩的坑，最终的解决方案才是现在的 diffusion-style (存疑，感觉更像 BERT) block-wise drafting.

最后，来蹭一下最近很火的Gemma 4的 MTP 模块:

首先，Gemma 4 的 MTP 模块有一个非常显眼的东西 —— KV Sharing. 说白话，就是 draft model 复用 target model 的 kvcache. (有点熟悉) 但是 Gemma 4这里有一点地方我不太理解：draft model 完全复用 target model 的 kvcache，而没有自己产生新的 draft token 的kvcache. 所以对于 draft model 推理的第 2, 3, 4个 token，所使用的target kvcache 是完全一样的，这很反直觉。我推测是通过最后一层global decoder 来实现区分的，可能只有global 层会生成新的 token 的kvcache.

其次，Gemma 4 的 MTP 会把target model 的最后一层 hidden states 和 draft model 的 input embedding 拼起来作为输入. (更熟悉了) 但是这里 gemma 4的 MTP 魔改了一版升维降维，我感觉这个意义不是很大 :)

然后，Gemma 4的 MTP 拥有多层 decoder. (更更熟悉了)

最后，Gemma 4 的 MTP 有一个所谓的 E2B / E4B. motivation 和 FR-Spec 完全一样，就是整个vocabulary 256k 个 token存在明显的长尾效应: 一小部分 token 的出现频率会主导整个 decoding 过程。因此，完全可以对 vocab 做剪枝。FR-Spec 是直接用 t2d 和 d2t 两个 mapping 来完成映射，Gemma 4这里的 E2B 就是先聚类再预测。说更直白一点：

本来 LM head 的 shape 是 d\times V , 通过聚类算法把整个vocab 分成 r 类，把整个矩阵运算进行降维, 从而减少计算量的效果.


最后，内容写的有点随意，不足之处还请多多包涵。大家有问题的欢迎直接找我交流～
