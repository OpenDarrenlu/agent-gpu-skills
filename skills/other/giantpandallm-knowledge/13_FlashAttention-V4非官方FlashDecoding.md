# 【FlashAttention-V4，非官方】FlashDecoding++

**原文链接**: http://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247519603&idx=1&sn=2572a1e9d42cf21bc581a61c8173c31c

**下载时间**: 2026-06-05 22:38:32

---

作者丨Austin 来源丨https://zhuanlan.zhihu.com/p/665595287 编辑丨GiantPandaCV

## 1. Introduction

为了提高softmax并行性，之前方法（FlashAttention、FlashDecoding）将计算过程拆分，各自计算partial softmax结果，最后需要通过同步操作来更新partial softmax结果。

本文在A100 GPU上分析了输入长度为1024的情况，这种同步partial softmax更新操作占Llama2-7B推理的注意力计算的18.8%。

## 三个挑战

1. **同步partial softmax更新代价高**：FlashAttention每次计算partial softmax结果都会更新之前的结果，而FlashDecoding是在最后统一更新
2. **解码阶段Flat GEMM计算资源未充分利用**：batch size较小时变成GEMV，cublas和cutlass会填充zeros执行更大batchsize的GEMM，导致计算利用率不足50%
3. **动态输入和固定硬件配置影响性能**：batch size较小时是memory-bounded，较大时是compute-bounded

## 优化方法

针对这3个问题，本文分别提出了对应优化方法，构成了FlashDecoding++。