# 如何用 Triton-TLE 写出比 FlashInfer 更快的 TopK Selector

**作者**: SunnyCase​DL Compiler

**原文链接**: https://zhuanlan.zhihu.com/p/2025245870753429243

---

0. TL;DR

TopK Selector 是 DeepSeek Sparse Attention (DSA) 中的关键算子之一，随着上下文长度的不断提升，它的优化价值逐渐体现出来。

本文分析了 TopK Selector 在 TileLang、TRT-LLM 中的实现方法，利用 Triton-TLE：

复刻了 TileLang 算法，在 H800（下同） 与 TileLang 接近/略快；
复刻了 TRT-LLM 算法，达到 TRT-LLM 的 85%~97%；
最后利用 TLE 编写的 DSMEM 版本在 batch=1 场景下相比 TRT-LLM 最高达到 2.5x 加速，相比 FlashInfer 最高达到 1.5x 加速。
1. TopK Selector 是什么

上下文长度是大模型能力的一个重要指标，随着上下文越来越长，Attention 
𝑂
(
𝑁
2
)
 的计算和访存代价越来越高。于是类似 DeepSeek Sparse Attention (DSA) 等结构被提出来限制参与计算的 KVCache 数量。

DSA 的计算流程可以简单拆成三步：

拿当前的 query 和所有历史 token 算出一行 logits
从这一行 logits 里挑出 topk 对应的 token 位置（索引）
只对这 k 个位置的 KV 做后续的 attention 计算

其中第2步就是 TopK Selector，随着上下文长度的增加，它在总延迟的占比越来越高。

对于 128K 的上下文，即使是 batch=1 的请求，FlashInfer 在 H800 上也要花 45us，这意味着 60 层的模型会带来 2.7ms 的 TPOT 延迟增加。

2. TopK Selector 和普通 TopK 的区别

TopK Selector 跟 torch.topk 有3点不同：

只需要索引，不需要值
通用 topk 通常返回排好序的值和索引，而这里只要尽快知道“哪几个位置要被后续使用”。
不需要全排序
只要能确定哪些元素属于前 k 名就够了，不需要把前 k 名内部也排得明明白白。
输入经常是长序列
DSA 阶段的序列长度可以很大，选择器的压力不在功能，而在延迟。尤其在 batch=1 这种小批量下，单次选择器的耗时直接加在端到端时延上。
3. GPU 上的常规做法：radix selection

GPU 上做 topk，radix selection 是常见方案。原因也直接：

GPU 很擅长并行计数和前缀和
radix selection 不要求全排序
它可以一轮一轮把候选范围压缩下来

基本流程是：

把浮点数映射成可以直接比较的无符号整数 key
从高位到低位，按 bit 段分桶
统计当前 bit 段的直方图
找到第 k 个元素落在哪个桶
只保留这个桶里的候选，进入下一轮
最后得到阈值，再收集所有大于阈值（以及部分等于阈值）的元素

伪代码：

# 返回第k大值（k从1开始）
def radix_select_kth_largest(A, k):
    C = A
    rank = k - 1                      # 0-based
    exp = highest_power_of_10(max(A)) # 例如 839 -> 100

    while exp > 0:
        count[0..9] = 0
        for x in C:
            d = (x // exp) % 10
            count[d] += 1

        acc = 0
        chosen = 0
        for d from 9 downto 0:        # 找“第rank个最大”落在哪个桶
            if rank < acc + count[d]:
                chosen = d
                rank = rank - acc
                break
            acc += count[d]

        C = [x in C where ((x // exp) % 10) == chosen]
        exp = exp // 10

    return C[0]


def radix_topk(A, k, need_sorted=true):
    T = radix_select_kth_largest(A, k)

    out = [x in A where x > T]
    need = k - len(out)

    for x in A:
        if x == T and need > 0:
            out.append(x)
            need -= 1

    if need_sorted:
        sort out in descending order
    return out

逻辑上没问题，但性能上有一个隐患：如果每一轮都要遍历整个序列，总开销仍然不小。序列越长，这个问题越突出。

4. TileLang 和 TRT-LLM 的改进思路
4.1 TileLang 的做法

代码：https://github.com/tile-ai/tilelang/blob/v0.1.8/examples/deepseek_v32/topk_selector.py

核心想法是降低访存代价：

第 0 轮 8 bit 初筛，把候选索引留在片上（shared memory）
后续 4 轮逐步缩小候选集
4.2 TRT-LLM prefill 的做法

代码：https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc10/cpp/tensorrt_llm/kernels/indexerTopK.cu

TRT-LLM prefill 虽然也是四段式筛选，但每段用的位宽不同：

第 1 步：用 fp16 的高位做一次粗筛
第 2 步：用 uint32 key 的高 11 位继续细化
第 3 步：用下一段 11 位再筛一轮
第 4 步：用最后 10 位确定阈值

如果某一轮留下的桶已经很小（<=4096）就直接早停，在小范围里做 final sort 收尾。

另外 TRT-LLM 访问 GMEM 时还采用了向量化指令，这个对性能影响很大。

5. 为什么 Triton 直接复刻 TileLang、TRT-LLM 版本很难

问题不出在算法上，出在实现这套算法所需的“表达能力”上。

片上状态很多
TRT-LLM 的选择器在 kernel 内部维护了大量中间数据：直方图、每一步的阈值、当前已找到的 topk 计数、输出索引、final sort 时的临时缓冲等等。这些状态如果不能在片上稳定存放，性能会明显掉下来。

共享内存的访问模式比较复杂
实现过程中反复出现这样的操作：

在 shared memory 上申请缓冲
用不同的索引视图访问同一块缓冲
在 shared memory 上做原子更新
在不同阶段把同一块缓冲当不同语义的数据结构用

Triton 没有暴露 shared memory，只能在 global 做，访存代价很高。

batch=1 场景需要跨 block 协作
单 block 版本处理一般情况够用，但对 batch=1 的长序列不够：

行数太少，batch 维度提供不了足够并行度
单行太长，一个 block 扫描会成为瓶颈

自然的想法是让多个 block 一起处理同一行。这马上就要求：

多 block 分工
局部结果汇总
跨 block 的同步和远程访问

Triton 并没有操作 cluster 的能力。

6. TLE 提供了什么能力

TLE 是在 Triton 基础上的语言扩展，它主要补齐了下面这些能力：

tle.gpu.alloc：显式申请片上缓冲
tle.gpu.local_ptr：在片上缓冲上构造指针视图，避免手写地址
tle.remote：访问其他 block 的片上缓冲
tle.device_mesh：定义 block cluster 的组织方式
tle.distributed_barrier：在 cluster 内做有范围的同步

TLE 可以让这类依赖 shared memory、cluster 的实现能够自然地写出来。

TLE 的详细说明：https://github.com/flagos-ai/FlagTree/wiki/TLE

7. 用 TLE 复刻 TRT-LLM 选择器

代码：https://github.com/flagos-ai/FlagTree/blob/f9a8d23602a65ec5c1af3b117e1faa46fe6f63b7/python/tutorials/tle/deepseek_v32/01-topk_selector.py#L658

片上数据结构
先在 shared memory 里放好：

histogram
各步阈值
输出索引
最终计数
final bucket 的临时索引和值缓冲

这样后续的 histogram 更新、候选写入和 final sort 都可以在片上完成闭环。

TLE 分配 shared memory 的方法：

@triton.jit
def tle_topk_selector_kernel(...):
    ...
    HIST_SIZE: tl.constexpr = 4096

    s_histogram = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )




用 local_ptr 访问共享内存
tle.gpu.local_ptr 让代码可以自然地表达：

对 shared memory 中的 histogram 做 load / store / atomic_add
把候选索引和值写入 final bucket buffer
在 final sort 阶段重新读这些候选

TLE 访问 shared memory 的方法：

@triton.jit
def tle_topk_selector_kernel(...):
    ...
    flush_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for flush_chunk in tl.static_range(flush_chunks):
        pos = flush_chunk * BLOCK_SIZE + lane
        mask = pos < TOPK
        out_vals = tl.load(tle.gpu.local_ptr(s_out_indices, (pos, )), mask=mask, other=-1)
        tl.store(out_row + pos * stride_outn, out_vals, mask=mask)
8. 为什么还要做 TLE distributed + DSMEM

单 block 版本已经能复刻 TRT-LLM 的大部分思路，但在 batch=1 长序列下还有优化空间。

问题在于：一行太长，单 block 的并行度不够。

所以需要把一行拆给多个 block 共同处理。

9. 用 TLE distributed + DSMEM 优化 batch=1

代码：https://github.com/flagos-ai/FlagTree/blob/f9a8d23602a65ec5c1af3b117e1faa46fe6f63b7/python/tutorials/tle/deepseek_v32/01-topk_selector.py#L3055

具体做法是：

用 device_mesh 定义一个 block cluster
每个 block 只负责一部分 tile
每个 block 先算自己的局部 histogram
通过 remote 把局部 histogram 汇总到 rank0 的 shared memory
rank0 找到当前轮的 threshold bucket
用 distributed_barrier 保证顺序
所有 block 按新的 threshold 继续筛选
基于 local_ptr 的直方图统计

这一步的收益在于把单行的工作量分散开了：

单 block 版本受限于单行扫描的串行度
cluster 版本把单行扫描拆到多个 block
汇总仍然在片上完成，不需要退到 global memory
10. 性能结果

环境：单卡 NVIDIA H800。序列长度覆盖到 512K。

provider：Triton、TRT-LLM prefill、TRT-LLM prefill-1024T、FlashInfer、TileLang、TLE (ours)。batch=1 额外包含 TLE cluster (ours)。

注1：TRT-LLM 默认使用的 num_threads=512，我们发现使用 1024 时在 H800 性能更高，于是增加了 TRT-LLM 1024T 的测试。

注2：TileLang 算法在 seq_len>=262144 的测试中会出现候选集溢出的情况，导致结果错误，表格中记为 N/A。

10.1 TileLang 与 TLE 复刻版对比

注：TLE-TileLang指的是用TLE复刻的TileLang算法

TLE vs TileLang
10.2 batch=1
Latency of batch=1




seq_len	topk	Triton (ms)	TRT-LLM prefill (ms)	TRT-LLM prefill-1024T (ms)	FlashInfer (ms)	TLE (ours) (ms)	TLE cluster (ours) (ms)	TileLang (ms)
8192	256	0.044672	0.011456	0.010400	0.013312	0.010848	0.018048	0.016416
32768	1024	0.141888	0.025184	0.018592	0.022400	0.021952	0.022656	0.034880
131072	2048	0.565440	0.075456	0.048880	0.044544	0.052368	0.029984	0.126576
262144	2048	1.116624	0.129600	0.079360	0.048160	0.090480	0.038448	N/A
524288	2048	2.172256	0.237504	0.139712	0.048064	0.164864	0.054832	N/A
10.3 batch=64
Latency of batch=64




seq_len	topk	Triton (ms)	TRT-LLM prefill (ms)	TRT-LLM prefill-1024T (ms)	FlashInfer (ms)	TLE (ours) (ms)	TileLang (ms)
4096	128	0.031040	0.010336	0.009840	0.012832	0.010144	0.015040
8192	256	0.046656	0.012640	0.011488	0.014512	0.012304	0.018496
32768	1024	0.144480	0.026912	0.020416	0.025376	0.024000	0.037792
131072	2048	0.601968	0.092256	0.061152	0.067392	0.063040	0.152448
262144	2048	1.251968	0.173760	0.106656	0.126032	0.112032	N/A
524288	2048	2.412192	0.311104	0.183168	0.195776	0.198592	N/A
11. 总结
TopK Selector 的优化关键是向量化访存和快速缩小候选范围。
Triton 直接复刻 TRT-LLM 的障碍主要在片上状态管理和跨 block 协作表达上；TLE 补上了这层能力，而 TLE distributed + DSMEM 进一步解决了 batch=1 长序列下的并行度问题。
未来的优化思路：
结合 TileLang 把候选集留在 shared memory
利用多个 block/cluster 处理一行
