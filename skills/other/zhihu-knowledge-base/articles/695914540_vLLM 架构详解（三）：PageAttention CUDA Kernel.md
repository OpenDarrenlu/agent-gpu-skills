# vLLM 架构详解（三）：PageAttention CUDA Kernel

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/695914540

---

CalebDu：vLLM 架构详解（一）

CalebDu：vLLM 架构详解（二）：内存管理

这一章节将详细讲解一下vLLM的核心 PageAttention CUDA Kernel 的详细实现。【需要对CUDA C/SIMT编程模型有一定的了解】

1. Attention

𝑂
=
𝑠
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
(
𝑠
𝑐
𝑎
𝑙
𝑒
×
𝑄
𝐾
𝑇
)
𝑉
𝑠
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
(
𝑥
)
=
𝑒
𝑥
−
𝑥
𝑚
𝑎
𝑥
∑
𝑒
𝑥
−
𝑥
𝑚
𝑎
𝑥

Attention模块是整个Transformer模型的核心， 在上式之外可能还会有ALiBi（Attention linear bias）/causal mask 等组件。在LLM中，通常采用MXA（MHA/MQA/MGA）的架构，对于每个Attention head 在prefill phase 
、
、
𝑄
∈
𝑅
𝑠
𝑒
𝑞
𝑙
𝑒
𝑛
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
、
𝐾
∈
𝑅
𝑠
𝑒
𝑞
𝑙
𝑒
𝑛
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
、
𝑉
∈
𝑅
𝑠
𝑒
𝑞
𝑙
𝑒
𝑛
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
 是compute bound，直接调用flash attention或xformer的实现， decode phase 
、
、
𝑄
∈
𝑅
1
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
、
𝐾
∈
𝑅
𝑠
𝑒
𝑞
𝑙
𝑒
𝑛
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
、
𝑉
∈
𝑅
𝑠
𝑒
𝑞
𝑙
𝑒
𝑛
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
 是memory bound，调用自定义的Page Attention 算子。

Page Attention的主要思想是建立一个全局K/V Cache Tensor，模仿OS中的内存页管理，将每个request 的KV Cache 通过Page Table建立映射关系，避免了按Max Seqlen 创建KV Cache 造成的内存浪费和 Concat 每次新的K V值导致的内存碎片。

2. CUDA Kernel

【在vLLM 的Page Attention实现中，由于是memory bound的场景，在Kernel中只使用了CUDA Core进行计算】

Page Attention 分为V1和V2两个实现，V2 对应为Flash Decoding 的实现，在SeqLen特别大、Batch偏小的场景下，通过把SeqLen按一定的长度在不同的CTA上进行计算来提高GPU 的Occupancy，增加计算资源的利用率。

Page Attention 以grid（n_seq，n_head, n_partition(v1=1, v2=seq/2048)）,cta(128[4warp],1,1)的配置launch Kernel。V1中每个CTA负责计算 
、
、
𝑄
∈
𝑅
1
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
、
𝐾
∈
𝑅
𝑠
𝑒
𝑞
𝑙
𝑒
𝑛
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
、
𝑉
∈
𝑅
𝑠
𝑒
𝑞
𝑙
𝑒
𝑛
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
 的Attention。V2中每个CTA负责计算 
、
、
𝑄
∈
𝑅
1
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
、
𝐾
∈
𝑅
2048
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
、
𝑉
∈
𝑅
2048
×
ℎ
𝑒
𝑎
𝑑
_
𝑠
𝑖
𝑧
𝑒
 的部分SeqLen Attention结果之后，再进行全局SeqLen reduce。Page Attention的参数输入K Cache 的shape为【n_block, n_head, head_size/x, block_size, x】其中 
𝑥
=
16
𝑠
𝑖
𝑧
𝑒
𝑜
𝑓
(
𝑠
𝑐
𝑎
𝑙
𝑎
𝑟
_
𝑡
)
 ,V Cache 的shape为【n_block, n_head, head_size, block_size】

本文以float16（16bit）、Block_size=8、head_size=128为例（紫），通过不同的颜色表示不同的线程层次（warp灰、thread group 蓝、thread 棕）和内存层次（DRAM 绿、SRAM 橘、Register 红）。

Page Attention 的第一部分为 \mathbf{P} =\mathbf{Q}\mathbf{K}^T

其中每个warp负责计算一个block内的计算 [1, head\_size] \cdot[head\_size, block\_size] = [1, block\_size] ，128线程共4warp 循环的计算每个partition的start_block 到end_block。 在warp层级之下，定义thread_group 来计算一个block内的每一个entry的计算 [1,head\_size] \cdot[head\_size, 1] = [1, 1] ，thread group内每个线程需要计算head_size 内的n_ele_per_thread个元素。如果 block\_size > warp\_size ，则每个group需要循环的n_token_per_group次计算block内的entry。head_size个特征以16B为单位拆分成两部分，每一个thread group内的线程合并访问这16B的数据,每个thread 访问vec 个元素【有点不太理解这里的操作，刚开始以为拆成16B是为了每个线程用int4来向量化访存，后来发现不是】。

先把当前CTA 对应的Q head按[thread_group_size, n_vec_per_thread, vec]的layout把 head_size个元素从DRAM载入到SRAM内，用于 \mathbf{Q}\mathbf{K}^T 的计算，如下图的d2s橘黄色部分。

QK^T

在每个 \mathbf{Q}\mathbf{K}^T 的核心循环内，每个thread_group 需要把Cache K对应block内的entry 载入到group内每个线程的k_reg 内[n_vec_per_thread, vec]，如下图的d2r部分。再把SRAM 中的Q 每个thread对应的部分用于计算qk向量点积，如图 k\_reg[0,1, 8, 9...120， 121] \cdot q\_smem[0, 1,8,9...120,121] 得到每个thread group内的partial qk结果， 【vLLM内对不同数据类型和不同的Vec长度的add、mul、fma做了模版特化】。之后还需要通过warp shuffle 对thread_group内线程的partial qk 结果进行reduce，得到一个entry 完整的qk结果再计算scale、alibi，需要更新循环中每个thread_group local max qk ，并把对应qk 存储到长度为partition 的 SRAM logits【注意mask】，用于后续的softmax计算。

QK^T

Page Attention 的第二部分为 SoftMax计算

在 \mathbf{Q}\mathbf{K}^T 的计算循环内，每个thread group 内的第一个thread 维护了本group内local max qk，需要对同一个warp内所有thread group 进行warp shuffle， 得到warp内的local max qk（block内的max qk）。由于warp之间不能直接通信，需要借助SRAM reduce_smem 再次warp shuffle，从而得到global max qk，并通过warp shuffle broadcast 给CTA内全部的thread。

CTA内的所有线程按照seqlen 并行的计算各自负责的 logit[i]=e^{logit[i]-qk_{max}} ,并计算local exp\_sum = \sum e^{logit[i]-qk_{max}} 。在调用block_reduce 得到global exp_sum, 再并行的更新 logits[i] = \frac{logits[i]}{exp\_sum} ,得到softmax 的计算 L ，并存储qk_max 和exp_sum。

softmax
Page Attention 的第三部分为 LV 计算

对于最后的 \mathbf{L}\mathbf{V} 计算，同样的一个warp负责一个block内的计算，每个thread计算block_size 中的v_vec个元素，n_vec_per_row 个thread 覆盖block_size 的一行，一个warp 一轮迭代可以计算head_size 内n_row_per_iter 行，每个thread需要负责 head_size中的n_row_per_thread 行, 存储在accs[n_row_per_thread]，如下图的棕色部分。

在 \mathbf{L}\mathbf{V} 的核心循环内，每个thread 从Cache V和logits读取对应位置的v_reg和logit 计算向量点积[需注意seqlen边界]，得到局部的acc并更新到对应行的accs[i]上，如图红色。

核心循环结束后，每n_vec_per_row个thread 的accs 是block 同一行内的local result，需要再进行warp shuffle得到block 内的完整结果， 从warp 的视角（每n_vec_per_row中第一个thread的n_row_per_thread个结果reduce，组成head_size）看如灰色部分。

LV dot

由于cta 共128 thread=4warp，以4 为步长访问start_block to end_block, 需要对warp 间的accs 进行reduce才能得到global result，如下图，借助SRAM实现树形规约，把warp0 reduce得到global result store 回DRAM。

result reduce
PageAttention V2

V2是Flash Decoding 在Page Attention的实现，主要目的是在cache len 特别长、n_seq 小、 grid小的场景中，提高GPU的Occupancy，避免计算资源的浪费。v2 reduce kernel比较简单，就是reduce每个partition local max qk， 再 logit[i]\times e^{local\_max\_qk - global\_qk\_max} 更新logit，并计算logits / global exp sum。flash decoding 的相关文章比较多，这里就不详细展开了。
