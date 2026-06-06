# Sliding Tile Attention: 高效的Attn稀疏&加速方法, 视频生成快3倍

**作者**: Peiyuan007UCSD CS PhD student

**原文链接**: https://zhuanlan.zhihu.com/p/24951760568

---

来知乎宣传一下我们最近的工作：
Fast Video Generation with Sliding Tile Attention

主要是对Video DiT 的attention做稀疏， 感觉一些思考和Deepseek NSA & Kimi MoBA 都很像： sparse attn不只是sparse就行了， 还得保证sparse 的pattern: 1. 符合数据本生的pattern. 2. 在GPU上有高效实现 (就是得符合flash attn的block boundary)
这个blog会有点长， 但是我个人觉得都是干货， 还请看官耐心看完。

Abstract

现在视频生成模型的速度简直是慢的离谱——Hunyuan Video用一张H100， 加上FlashAttention3，加上torch compile, 生成一段5秒的视频居然要花16分钟。Sliding Tile Attention（STA)把时间砍到了5分钟，而且画质基本没损失，还不需要额外训练。Kernel层面来说，STA是一种sparse attention, 因为有sparsity, 速度能比FlashAttention-2 full attn快了2.8到17倍，比FlashAttention-3快了1.6到10倍。把STA用到HunyuanVideo能把inference提速1.8倍， 再加上TeaCache，我们的方案能加速Hunyuan Video 2.98倍，如果再加上微调，速度还能进一步提升！

为啥视频模型这么慢


现在Video DiT 处理视频非常简单除暴： 把vae latent flatten成一条sequence然后直接喂一个non-causal的full attention. 但是视频模型的token数量极其庞大， 我感觉之后很有可能会比最长的LLM还要长——以HunyuanVideo为例，仅仅生成一段5秒的720p视频片段，就会产生115K个token。随着分辨率提高或时长增加，这个问题会变得更加严重：假设视频的shape是（L，L, L)，即使L稍微增加一点，token数量也会呈立方级爆炸式增长。再加上注意力机制的计算复杂度是平方级的，attention的计算量就直接爆炸了。图一（a)画了一下DiT的flops占比。



图一

很容易想到的一点就是我们应该让attention稀疏一点-- 毕竟视频这个模态天然就是很稀疏的，很多之前CV里面的工作其实explore了ViT/DiT的sparse attn，但是很多都是降Flops不降latency, 比如图一(b)的NATTEN和CLEAR.


3D Locality in Video DiT

其实，Video DiT里面的每个 Token 都倾向于关于相邻的 Token （Locality），如下图所示：

图2（左）是在 HunyuanVideo 的注意力分数，发现了一个显著的 3D Attention Locality 模式：Query（绿色的点）倾向于关注 Spatial 和 Temporal 上邻近的 Key。这一点也是很好理解的，越近的 token 关联越强。
为了定量分析这一现象，我们计算了 Attention Recall ——即局部窗口内的注意力分数占总注意力分数的比例。如图2（中）所示，一个很小的局部窗口（仅占总空间的15.52%）就占据了70%的总注意力分数。
图2（右）体现这个现象在不同的 prompt 都是始终存在的 (std 很小），也就是 data-independent。





那其实问题就很简单了 -- locality不就是CNN的intuition吗？ 对于attn和video DiT来说，设计一个3D sliding window attention (SWA), 问题就解决了？ 其实图一里面的NATTEN就是一种SWA, 但是他的问题就是跑的太慢了. 其实SWA只在1D的时候有高效实现（比如mistral的1D SWA), 在2D/3D里面SWA就是很慢， 根本原因是2D/3D SWA和Flash Attention不兼容， 我下面会具体分析为啥他们不兼容。

为啥2D/3D SWA这么慢？

为了理解为什么SWA和FlashAttention（FA）不兼容，我们首先需要回顾FA的逐块计算模式。FA并不是逐个处理token，而是将输入序列分割成block——通常是（128, 64）的大小。为了简化讨论，我们假设这些block是正方形的 (B, B)。FA将一个block中的Q、K和V加载到GPU的SRAM中，执行所有必要的计算，并只将输出矩阵O写回HBM，从而避免了存储中间值（mask, attn score ..）。如图3所示，每个block可以理解为FA的基本计算单元。为什么这很重要？首先，这避免了生成大型中间张量，从而节省了memory 读写。其次，GPU是为矩阵乘法设计的，它们不擅长处理标量甚至向量；它们擅长的是matrix mul (一次计算算一整个block)，而不是逐个token处理。


图三

在FlashAttention中实现2D/3D SWA， 其实就是要定义SWA在attention中的mask, 根据block和mask的关系，我们可以把 FA block 分为三种类型：

Dense blocks：保留所有注意力分数（高效 ✅），
Empty blocks：屏蔽所有值（可以完全跳过 ✅），
Mixed block：保留部分分数，同时屏蔽其他部分（效率极低 ❌）。

Mixed block会引入远大于Dense block的开销，原因如下：

计算浪费：由于块是最小的计算单元，FA必须先算整个block,再用mask把不要的value去掉
GPU不友好的mask操作：FA需要计算每个的block所对应的mask, 而mask取决于用户定义的sparse attn pattern，也区别于block在attention map中的位置。更G是，mask无法预先计算——整个mask的size是O*2 complexity。FlexAttention论文有提到计算一个简单的causal attention mask 会增加15%的latency. 其实causal mask都算好的， 因为他只需要计算在对角线上的mixed block的mask。 在3D SWA中 (图三（a))，mixed block无处不在，mask计算能超过计算block本身的成本！这就是为什么2D/3D SWA本质上对GPU不友好——它产生了太多的Mixed Block！

为了缓解这个问题，Tiled NATTEN通过重新排序输入来增加Dense Block的数量（图3(b)）。然而，仍有相当一部block是mixed block，这使得SWA在GPU上本质上效率低下。

下午展示了为啥2D的SWA会生成一个非常奇怪的attention map。






Sliding Tile Attention

Sliding Tile Attention (STA) 的intuition其实很简单：GPU喜欢一次算一个block, 而不是一个token, 导致SWA不高效的根源是它在滑动窗口的时候一次是滑动一个token, 导致每个query需要attendkey value group都不一样。STA只引入了对SWA的一个非常小的修改： STA每次滑动一个tile。

具体来说, (注意以下的notation是针对3D， 也就是video的， 但是为了图好画， 所有的图都画得是2D, 不要混淆了）
1. 一个尺寸为 (L, L, L)的视频会先被分割成很多个大小为(T, T, T)的tile。假设 Flash Attention 的block大小为 (B, B)，T 应满足B = T*3。

2. 在把3D的（L, L, L)flatten成一个1D sequence输入给attention kernel的时候， 一个tile 内的token会有连续的index。并且STA的window side 也需要是（T，T, T）的整数倍。

3. 注意力窗口以（T，T, T) 为单位逐块移动。对于每个local window，中心的query tile (不是query token), 会attend整个window的KV.

这样搞， Mixed Block就不见了。

STA可以用FlexAttention实现， 我还写了一个ThunderKittens,用上了Asynchrounous data/compute warpgroups, 这样compute warpgroup就完全不需要管哪些block应该算， 哪些block要skip了（都归data warpgroup管） 这样速度还能更快。



后续实验有点懒得写了，写到现在也有点长了。 先拖更， 有时间补上。。。。 感兴趣的读者可以移步Fast Video Generation with Sliding Tile Attention ，里面有如何把STA apply到VideoDIt以及一些炫酷的视频。
