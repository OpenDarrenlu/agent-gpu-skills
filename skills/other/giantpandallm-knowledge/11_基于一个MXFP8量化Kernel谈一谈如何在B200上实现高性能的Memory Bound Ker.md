# 基于一个MXFP8量化Kernel谈一谈如何在B200上实现高性能的Memory Bound Kernel

**原文链接**: https://mp.weixin.qq.com/s/rJBw4B4e7ObQ4CFP-8fXjA

**下载时间**: 2026-06-05 22:38:32

---

作者丨Anonymous 来源丨https://zhuanlan.zhihu.com/p/1975964435991527542 编辑丨GiantPandaLLM

## 背景

近期，笔者向SGLang社区提交了基于CUTLASS的MXFP8 Blockscaled Grouped GEMM实现。这个PR不仅包含Grouped GEMM，还包含了一个量化Kernel，用于将多组fp16/bf16的输入矩阵转换为fp8(e4m3)的输入矩阵，同时还会输出MXFP8 Blockscaled量化所需的Scale Factor。

## 核心问题

这个量化Kernel是一个典型的Memory Bound Kernel。由于目标架构是Blackwell B200，而B200具备 **7.7TB/s** 的HBM3e显存带宽（接近于H200的2倍），因此如何有效利用显存带宽就成为了这个量化Kernel所面临的最主要的问题。

## 优化方法

本文按照从整体到局部的思路，介绍这些优化技术：
1. 向量化加载/存储（Vectorized Load/Store）
2. 合并访存事务（Coalesced Memory Access）
3. 减少指令开销
4. 利用Blackwell架构新特性

这些优化技术具有一定的普适性，可以用于优化其它Memory Bound Kernel，甚至是应用于其它的目标架构当中。