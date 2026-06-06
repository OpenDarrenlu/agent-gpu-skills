# Weight Only Quantization 的性能优化

**作者**: MingfeiPyTorch CPU Perf Maintainer

**原文链接**: https://zhuanlan.zhihu.com/p/687844000

---

​
目录
收起
WOQ 背景介绍
Weight Prepacking
Micro Kernel Design
性能数据
结语

过去一年 Generative AI 可以说是博足了眼球，如何提高部署阶段的性能成了各大公司工程团队的必修课。各种各样的技术手段层出不穷，比如用来解决大 sequence length 情况下 SDPA (scaled dot product attention) 的 flash attention, 还有优化 kv cache 的 paged attention 等等。 在 Text Generation 领域，WOQ (weight only quantization) 是一个提高性能的常规手段。

PyTorch 团队去年开启了一个新的 blog 系列，主题是通过 pure，native PyTorch 解决 Generative AI 中的性能问题。这个系列中的第二篇介绍了一个模版类型的项目 GPTFast，教大家怎么通过 torch.compile 加速 LLM inference，灵感部分来源于另一个著名开源 LLM 加速项目 Llama.cpp。

这个项目中一个非常重要的 feature 就是 WOQ，项目发布之初只添加了 CUDA 的支持；最近添加 CPU 的支持，会随 PyTorch 2.3 一并发布。

之前的文章中介绍过 SDPA 的优化，有兴趣的同学可以看一下：




Mingfei：Scaled Dot Product Attention (SDPA) 在 CPU 上的 性能优化
136 赞同 · 18 评论 文章

这篇文章介绍 WOQ kernel 的实现细节，侧重技术讨论，并非广告贴。读者需要对 SIMD 性能调优有一定基础。目前主要的 kernel code 都已经 upstream 到 PyTorch master，在 /aten/src/ATen/native/cpu 下面的 int4mm_kernel.cpp 和 int8mm_kernel.cpp 这两个文件。

WOQ 背景介绍

大家都知道在 LLM 进入 auto regressive decoding 阶段之后主要的性能瓶颈是 memory bandwidth，原理在于 当 batch_size = 1 的时候， GEMM 操作会退化为 GEMV，activation 占用的内存要远远小于 weight 的内存，所以一个常规手段是对 weight 进行量化，通过减少内存读取开销来加速。对比常规的 W8A8（weight 和 activation 都存成 8 bit），WOQ 在 text generation 模型中更受欢迎，即只对 weight 进行量化，而 activation 保留在更高的数据类型。原因在于对 activation 量化会造成 accuracy 明显下降，而且工程上实现起来比较麻烦，另外性能收益有限。

GPTFast 这个项目中实现了两种 WOQ：W8A16 和 W4A16。从名字上就可以判断，weight 是 4-bit 或者 8-bit，activation 是 16-bit 。（注：Llama2 的模型默认是 float16，但在 GPTFast 中转成了 BFloat16 计算）

Fig-1 Weight Only Quantization Pattern

具体来说，int8 采用了常规的 per channel quantization，而 int4 为了保留更多精度，采用了 groupwise quantization，默认情况下每 32 个元素会记录一个 scale 和 zero。这个思路和 GGUF 完全一致，不同的地方在于 GPTFast 可以通过 torch ATen 的内部接口自定义 weight prepacking format，对性能更加友好；但 GGUF 实现的 format 其实更多，除了 4 bit 和 8 bit 之外还有很多，比如 1.5 bit 等。

主要的性能加速实际上来自于 ATen 里面的这个函数 _weight_int4pack_mm , 上面在 torch.compile 的架构里面套了一个壳子，好让从用户层面看起来都是 compile 做得加速。本质上是已经在各种 torch 派生项目上发扬光大（用烂了）的手段：module dynamic replacement + customized kernel。

Weight Prepacking

weight prepacking 是一种低精度加速中的常规手段，就是指提前将 data format 改成硬件指令执行比较方便的顺序，有时候也叫 reorder。具体选取什么样的 data format 和硬件 ISA 直接相关，这里 AVX512 的 format 选取的是 [N/64, K, 32]:

Fig-2 Weight Prepacking on Int4

首先，我们在 N 维度上做 Blocking，每个 block 为 64。在 block 之间并行，原因是 weight >> activation 的时候要把 weight 切掉，这样每个 thread 上拿到的是 weight / T。选择 64 为 block size 的原因是刚好是 4 x vector length。这里的 4x 是指每个 micro gemm kernel 的 NB 大小 （我们把每个 micro gemm kernel 定义为 MB * NB，这里面 MB 和 NB 大小选择主要和硬件 register 数量有关系，micro kernel 里面的 register 数量不能超过 32，以避免 register spill）。

然后，将最里面的两个维度 transpose，也就是从 [N/64, 64, K] 换成 [N/64, K, 64]。这是为了将 micro gemm kernel 从 NT （non-transpose, transpose）换成 NN （non-transpose, non-transpose），这么换主要原因是为了避免计算 dot product 时候最后一个 vector 到 scalar 的 horizontal reduce。

最后，对于每个行上的 64 个元素，进行 packing。AVX512 选用的 format 和 GGUF 有一点不一样，因为要一个 de-quant 256 bits (64 x 4 bits)，把 Lane2 和 Lane0 压成一行，Lane3 和 Lane1 压成一行，这么做的原因是 x86 上的 shift 指令只能以 Lane （128 bits） 为单位。另外 AVX512 上面比较取巧，用了 _mm512_permutexvar_ps 这个指令，提前定义好了一个 LUT，然后通过 permute 完成 de-quant。这样比直接 dtype convert 再做减法要快一些。AVX2 选用的 format 和 GGUF 完全一致，每次 de-quant 128 bits （32 x 4 bits）。

de-quant 是 WOQ kernel 里面比较重点的一个地方，因为当 batch size =1 的时候，gemm 里面的 M = 1，de-quant 的开销没办法在多行里面被平摊掉。

Micro Kernel Design

目前的计算还是通过比较慢的 float32 完成的，activation 和 weight 都会被转成 float32 然后通过 fused multiply-add （FMA）计算。这里主要限制是 GCC 版本，如果使用更高级别的 ISA 比如 avx512bf16 或者 amx_bf16 需要更高 GCC 版本支持，TODO。

Fig-3 Micro Kernel Design in GEMM

SIMD 上 GEMM 的写法和正常人类的认知刚好是相反的，要将常规的 M-N-K 的循环循序改成 K-M-N，为了适配 FMA。基本思路是对于每一个 block，A 矩阵每次拿一个 scalar，然后 broadcast 成一个 vector （va）; B 矩阵每次拿一行，load 4 x vectors （vb[4]），C 的累加值用 16 个 register （vc[16]）记录；另外还有 8 个 register 放 scale 和 zero 的数据。然后一直沿着 K 方向做 FMA，最后把 vc store 回 memory 中。如果要加什么 Post OP fusion，就在这时候加。比如 add bias，不然的话，bias 会被 broadcast 到 C 的大小，还挺费时间的。

性能数据

在 Llama2-7b 上面目前能拿到的数据大概是 37 tokens /sec，使用的 4th gen Xeon，单 socket。运行参数是 GPTFast README 上的默认值。

Fig-4 Llama2 7B Performance Data

这个数据还不是最优值，还需要继续改进，比如可以像 GGML （Llama.cpp 的算子库）那样对 activation 做 dynamic quant 然后用 int8 计算。

结语

目前主要的 kernel 均已 merge 到 PyTorch master 当中，欢迎大家试用。

最后想说，这个数据很容易被横向比较，我知道完全不够看。但就像篮球运动一样，有些球员就是超级球星，无所不能，有的球员就是角色球员，就是抢篮板和底角三分。我只能考虑在现有条件下怎么做得更好。
