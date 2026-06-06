# vLLM Triton Merge Attention States Kernel详解

**原文链接**: http://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247527742&idx=1&sn=78b89e3a19497bd335706e4866647706

**下载时间**: 2026-06-05 22:16:50

---

作者丨DefTruth
来源丨https://zhuanlan.zhihu.com/p/1904937907703243110
编辑丨GiantPandaLLM

### 0x00 前言

本文介绍vLLM中Triton Merge Attention States Kernel的实现，与 pytorch原生实现相比，该Triton kernel最高可实现 3-5 倍以上的算子加速。

### 0x01 Merge Attention States 简介

Merge Attention States在FlashInfer论文中2.2 Attention Composition小节中出现，然后在vLLM的Triton MLA实现中也被使用到。

Attention的计算是可以分块的。Block-Parallel Transformer (BPT)表明，对于相同的query以及不同的key/value，Attention Output(O)可以通过同时保留每个块的O及其缩放比例LSE来进行组合。

在decode阶段，我们通常面临的是query很小，比如1，但是key和value很长，seqlen长度。因此，对于长序列，可以考虑对key/value先分块，每个块各自计算自己的Attention结果，记录块对应的LSE，最后通过缩放比例来合并。这就是所谓的 "Merge Attention States"。这种用法，在Chunked-Prefill、Prefix-Cache和Split-KV的场景都会有意义。

### 0x02 PyTorch实现

PyTorch实现使用safe softmax常规操作，先减去最大值，然后计算exp和scale值，最后对结果校准得到最终Attention输出。

### 0x03 Triton 基础算子

vLLM中提供了一个基于Triton实现的kernel，完整代码链接：https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py

Triton kernel做的事情和PyTorch实现的一样，但是将所有的操作都fused到一个kernel中，online判断inf值（寄存器）而不是修改global memory中的值，性能一般来说会更高。

vLLM里边的实现，给merge_attn_states_kernel，分配(num_tokens, num_query_heads)个thread block，每个block处理当前head的所有值。

### 0x04 Triton 算子分析

当num_tokens、num_query_heads很大，而head_size很小（比如32）时，就会导致thread block数过大，每个block处理的数据量又过少，计算密度很小。而且，这种情况下，Triton也不一定能生成高效的kernel。

可以通过指定TRITON_CACHE_DIR环境变量，把Triton生成的中间IR文件给保存下来，进行分析。关注PTX文件，看是否生成向量化ld/st指令。

### 0x05 NCU Profile分析

通过ncu抓实际跑的PTX和SASS。对比memory throughput: 45.67(Triton kernel) -> 60.57 (CUDA kernel)

### 0x06 性能评估

与 pytorch原生实现相比，Triton kernel最高可实现 3-5 倍以上的算子加速。

### 0x07 总结

本文介绍了vLLM中merge_attn_states triton算子的实现，内容包括：Merge Attention States 简介、PyTorch实现、Triton 基础算子、Triton 算子分析、NCU 分析、性能评估。

代码在：https://github.com/xlite-dev/LeetCUDA/tree/main/kernels/openai-triton/merge-attn-states