# Scaled Dot Product Attention (SDPA) 在 CPU 上的 性能优化

**作者**: MingfeiPyTorch CPU Perf Maintainer

**原文链接**: https://zhuanlan.zhihu.com/p/647907549

---

​
目录
收起
Previous Work
SDPA 优化
概述
Lazy Softmax
在 KV 上做 Blocking
在 Q 上做 Blocking
Float16 和 BFloat16 的实现
Causal Mask
一些问题

PyTorch 2.0 的主要 feature 是 compile，一起 release 的还有一个很重要的 feature 是 SDPA: Scaled Dot Product Attention 的优化。这个东西使用在 Transformer 的 MHA: multi-head attention 里面的。一共包含三个算法：

Math: 把原始实现从 Python 挪到了 C++
Efficient Attention
Flash Attention

后两种算法是无损加速，不同于使用 low rank 或者 sparse 的方式，从数学上来说计算没有发生变化，所以不影响精度。

SDPA 主要是为了解决 LLM 中的两方面痛点：

memory footprint: attn 的尺寸是 {B, H, T, T}。和 T 是 O(n2) 的关系，随着 sequence 变长，memory 开销太大；
performance speedup: 针对 attn 的 pointwise 操作都是 memory bandwidth bound，速度太慢了

目前的版本中，后两种算法都只支持 CUDA device。CPU 的版本目前差不多都完事了，争取下个版本放到 master 里面，主要的 Pull Request 是 #104238， #104239， #103826， #104693。有兴趣的可以看一下。

这篇文章主要是介绍一下 SDPA 的优化算法，目的不在于宣传性能。一般来讲，我们在 Pull Request 里面会列举具体的性能数据，对于重要的 feature 会在 Meta 和 Intel 的官方渠道宣传。

注：文中的 CPU 指 Xeon。

Previous Work

其实 1.3 版本就出现了 nn.MultiheadAttention 的优化，具体应用的 API 是 HuggingFace Optimum 的 BetterTransformer。思路其实很简单，把 gemm 之间的 pointwise 统统 fuse 起来，过程如图 Fig-1:

Fig-1: Implementation of nn.MultiheadAttention

在这个 workflow 中，最大的收益来自于对 attn 操作的 fusion，因为 QKV 尺寸是和 T * K 成正比，而 attn 是和 T * T 成正比。这里的 K 是每个 head 上的 feature size，T 是 sequence length，一般来讲 T 会 比 K 大很多。具体来说：

原始实现对于 masked softmax 的处理一共需要 4 reads + 5 writes:

对于 mask 的处理会非常繁琐：需要 4 次操作: ones, tril, not, masked_fill。共需要 3 reads + 4 writes。softmax 由于需要保障数值稳定性，需要 4 个 steps 完成，不过这 4 步只有 1 read + 1 write，原因在于 transformer 里面是在 lastdim 上做 softmax，正常情况下数据 parallel 的方式保障 L1 cache hit，所以只有 1 read + 1 write。

做了 fuse 之后，masked_softmax 一共需要 1 read + 1 write。强调一下 attn 这是个很大的 tensor，所以主要的性能收益来自这个地方。但即使只有 1次读和1次写，还是不够快，另外这个算法解决不了内存开销太大的问题。为了解决这些问题， SDPA 应运而生了，不管是 efficient attention 还是 flash attention，核心都是如何通过 blocking （或者叫 tiling）避免直接分配一块 {B, H, T, T} 这么大的 attn。通过让数据停留在 cache 上面，达到对 pointwise 操作的加速。

SDPA 优化
概述

这里着重介绍算法的演化过程：很多时候我们看到了一个最终形态的 kernel 不理解为什么会写成这样。所以从简单入手，一步一步介绍演化的过程。这里从 efficient attention 的最初形态开始，到 flash attention 2 结束，在经过 fully optimized 之后这两种算法本质上没有区别的。

整个 scaled dot product attention 的原始过程可用 Fig-2 来表示，对于每一个 {B, H} 的 slice：

Fig-2: scaled dot product attention

这里，把 V 看作一个 {v0, v1, ..., } 的向量会比较好理解。另外，我们认为这里 attn 还是做了实际的内存分配。

整个过程可以分解为 3 步：

Step-1 是一个 vec-vec 的 DP；
Step-2 是针对 attn 每一行元素的 pointwise；
Step-3 是一个 vec-mat 的 GEMV。
Lazy Softmax

引入 lazy softmax 可以避免为 attn 实际分配内存，在每个 thread 保留一些 momentum 信息即可: m* 记录当前的 max value； s* 记录 sum value；v* 记录 out 中每一行的累计值。那么，可以很容易地算出来每个 thread 需要的额外内存只有：1 + 1 + Kv （Kv 是 V 每个 head 的 feature size)。

Fig-3: lazy softmax

为了保障数值计算的稳定性，计算过程如 Fig-3 所示。从性能角度出发我们更关心计算的性质，与原始形态计算量实际上发生了退化，不过好在不需要分配 {B, H, T, T} 这么大一个 tensor 了：

Step-1 还是一个 vec-vec 的 DP；
Step-2 是针对 attn 每一行元素的 pointwise （变成 scalar 操作，无法向量化）；
Step-3 是一个 scalar-vec 的乘法。

但是，这种实现依旧很原始，性能并不好，这个 kernel 大概会比原版还慢十几倍。主要原因有两点：

对于每一个 q_i，都需要遍历整个 K，才能完成 attn 中一行的计算；
s_i 需要和 v_i 相乘并累加到 o_i 中，这个过程中同样对于 V 有重复访问，并且要多次写入 O;

按模型中实际尺寸来算，KV是不可能被 cache 命中的，所以就是在不停地刷内存带宽，肯定快不了。

在 KV 上做 Blocking

首先让我们在 KV 上做 blocking，即每一个 iteration 计算 q_i 和 一个 K block 和 V block，这么做是为了减少对 O 的写入次数，KV block 的数量就是减少写入次数的倍数。

Fig-4: Blocking on KV

注意，这个时候计算的性质已经发生了变化，每一步的计算量被放大了 NB 倍：

Step-1 是一个 vec-mat 的 GEMV；
Step-2 是针对 attn 每一行元素的 row pointwise；
Step-3 是一个 vec-mat 的 GEMV。

我们也需要一个额外的 s_i 来记录 qk 的内积结果，那么每个 thread 的额外内存变为：1 + 1 + NB + Kv

另外 step-2 变成了 row pointwise，那么我们又可以做向量化了。

不过这样还是不能解决对 KV 的重复访问。

在 Q 上做 Blocking

然后让我们在 Q 上做 blocking，即每一个 iteration 计算 一个 Q block 和 一个 KV block，这么做是为了减少对 KV 的读取次数，Q block 的数量就是减少读取次数的倍数。

Fig-5: Blocking on Q

让我们再来分析一下计算的性质，每一步的计算量被再次放大了 MB 倍：

Step-1 是一个 mat-mat 的 GEMM；
Step-2 是针对 attn 每个切块的 block pointwise；
Step-3 是一个 mat-mat 的 GEMM。

每个 thread 的额外内存变为：MB * (1 + 1 + NB + Kv)，扩大了 MB 倍。不过我们还是可以通过计算保障这个 buffer 被 L2 命中（L1 大小是 32KB，L2 是 1MB，这个 buffer 大小可以设置 L2 的 25%）。

至此，我们完成了对 SDPA 基本形态的推导，从 efficient 算法入手，可以得到数学上和 flash2 完全一致的过程：

Fig-6: flash2
Float16 和 BFloat16 的实现

Fig-5 的过程只需稍加修改即可实现对 float16 或 bfloat16 的支持，基本原则是用 float32 来做 accumulation。当然在 intel xeon 上得益于 AMX 的硬件加速，code 中使用了 MKL 中的 cblas_gemm_bf16bf16f32 函数，即 A(bf16) x B(bf16) = C(fp32)：

Fig-7: Bfloat16
Causal Mask

SDPA 对于 Causal mask 的处理是在 s_i 这个 buffer 里面加 mask，配合上 blocking，可以额外省掉上三角的 GEMM，所以在 causal mask 的情况下 SDPA 能拿到更大的加速比：

Fig-8: Causal Mask

实际中因为配合了 blocking，所以中间的那条线应该是个阶梯状的，阶梯上面的 GEMM 会被省略掉。

一些问题

首先最显著的一个问题就是 load imbalance, 我们依赖在 B-H-MB （batch-head-q_block) 这三个维度上做 parallel，但每一个 q block 对应访问的 kv block 数量是不一样的，可能会导致 load imbalance:

Fig-9: load imbalance

这个问题其实很好解决，因为我们预先就可以算出每个 q block 对应几个 kv block。

还有一个比较难处理的问题是每个 thread memory 访问不均衡的问题。比如我们有 10 个 q block，但每个 thread 只能计算 8 个，那么 T0 只会访问一组 KV (都来自 Head_0)；而 T2 会访问两组 KV (来自于 Head_0 和 Head_1)。这个图片比较难画，我就没画...

另外还有一个让 amx 和 avx512 并行的问题，也就是如何让 GEMM 和 pointwise 并行起来 ...

当然，这些问题都会慢慢处理掉。
