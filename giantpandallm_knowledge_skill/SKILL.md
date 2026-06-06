---
name: giantpandallm-knowledge
type: knowledge
description: >
  GiantPandaLLM 公众号文章知识库。涵盖 CUDA 优化、Triton 编程、CUTLASS/CuTe、
  vLLM/SGLang 推理优化、多模态模型、端侧 AI 部署、Agent 驱动开发等方向的技术文章合集。
  支持增量更新，可添加更多公众号文章。
---

# GiantPandaLLM 公众号文章知识库

> **来源**: GiantPandaLLM 公众号
> **整理时间**: 2026-06-05
> **文章总数**: 20 篇
> **更新方式**: 增量更新（可添加其他公众号文章）

## 知识库简介

本 Skill 收录了 GiantPandaLLM 公众号的高质量技术文章，主要聚焦于：
- **GPU 内核优化** (CUDA / Triton / CUTLASS)
- **LLM/VLM 推理加速** (vLLM / SGLang / FlashAttention / 量化)
- **多模态模型** (VLM / 扩散模型 / 端侧部署)
- **深度学习框架优化** (PyTorch / 编译优化 / 分布式训练)
- **Agent 驱动开发** (Codex / Humanize / KernelPilot)

## 文章目录

### 🔥 CUDA / GPU 内核优化

| # | 文章标题 | 核心主题 | 文件 |
|---|---------|---------|------|
| 04 | CUDA优化: 让向量求和变得非常快 | CUDA 归约优化、warp 同步、向量化加载、H100 性能调优 | [04_CUDA优化](articles/04_CUDA优化%20让向量求和变得非常快.md) |
| 09 | CUTLASS CuTe GEMM细节分析（三） | Swizzle 模板参数、共享内存 Layout、Bank Conflict 消除 | [09_CUTLASS](articles/09_CUTLASS%20CuTe%20GEMM细节分析三SwizzleB%20M%20S模板参数的取值.md) |
| 11 | 基于MXFP8量化Kernel谈B200 Memory Bound优化 | Blackwell B200、MXFP8、量化Kernel、显存带宽优化 | [11_MXFP8](articles/11_基于一个MXFP8量化Kernel谈一谈如何在B200上实现高性能的Memory%20Bound%20Ker.md) |
| 12 | Tensor-001 矩阵乘法分块乘法概述 | GEMM 内积/外积形式、矩阵计算基础、CUTLASS 前置 | [12_GEMM](articles/12_Tensor-001%20矩阵乘法分块乘法概述.md) |
| 13 | 【FlashAttention-V4非官方】FlashDecoding++ | FlashAttention 优化、partial softmax、解码阶段GEMM优化 | [13_FlashDecoding](articles/13_FlashAttention-V4非官方FlashDecoding.md) |

### 🚀 Triton 编程入门与进阶

| # | 文章标题 | 核心主题 | 文件 |
|---|---------|---------|------|
| 08 | Triton极简入门: Triton Vector Add | Triton 基础语法、Block-wise 编程、PTX 分析 | [08_Triton入门](articles/08_Triton极简入门%20Triton%20Vector%20Add.md) |
| 07 | Triton Fused Softmax Kernel详解 | Fused Softmax、多级流水线、PTX 分析、性能优化 | [07_TritonSoftmax](articles/07_Triton编程基础%20Triton%20Fused%20Softmax%20Kernel详解%20从Python源码.md) |
| 05 | vLLM Triton Merge Attention States Kernel详解 | Merge Attention States、Triton 算子开发、NCU Profile | [05_vLLMTriton](articles/05_vLLM%20Triton%20Merge%20Attention%20States%20Kernel详解.md) |

### 🤖 LLM / 多模态模型

| # | 文章标题 | 核心主题 | 文件 |
|---|---------|---------|------|
| 01 | 轻量化视觉语言模型实战：TinyMind（90M） | 端侧 VLM、MiniMind、MobileCLIP、模型压缩、Android 部署 | [01_TinyMind](articles/01_轻量化视觉语言模型实战TinyMind90M从训练到端侧部署的完整旅程.md) |
| 06 | 万字长文图解Qwen2.5-VL实现细节 | Qwen2.5-VL、Window Attention、动态帧率采样、多模态 RoPE | [06_Qwen2.5-VL](articles/06_万字长文图解Qwen25-VL实现细节.md) |
| 16 | 非常简洁的图像复原新方法：退化分类预训练 | ICLR2025、图像复原、退化分类、All-in-one 训练 | [16_图像复原](articles/16_GiantPandaLLM%20%20非常简洁的图像复原新方法退化分类预训练.md) |

### ⚡ 推理优化 / 框架加速

| # | 文章标题 | 核心主题 | 文件 |
|---|---------|---------|------|
| 03 | Presenting Flux Fast: 让 Flux 在 H100 上疾速飞驰 | torch.compile、Flash Attention v3、FP8 量化、CUDAGraphs | [03_FluxFast](articles/03_博客翻译Presenting%20Flux%20Fast%20让%20Flux%20在%20H100%20上疾速飞驰.md) |
| 10 | PipeFusion：PCIe互联GPU低成本并行推理扩散模型 | DiT 并行推理、Input Temporal Redundancy、流水线并行 | [10_PipeFusion](articles/10_PipeFusion如何用PCIe互联GPU低成本并行推理扩散模型.md) |
| 14 | SGLang｜SGLang Diffusion | 扩散模型推理引擎、图像/视频生成、性能加速 5.9x | [14_SGLangDiffusion](articles/14_SGLangSGLang%20Diffusion.md) |
| 15 | 万字详解《超大规模操作手册：在GPU集群上训练》Part1 | DP/TP/PP/SP 并行、3D并行、训练配置优化 | [15_大规模训练](articles/15_万字详解超大规模操作手册在GPU集群上训练Part1基础概念DPTP.md) |
| 20 | 在SGLang中使用reasoning模型 | Qwen3 reasoning、结构化输出、Pydantic、SGLang API | [20_SGLangReasoning](articles/20_GiantPandaLLM%20%20在SGLang中使用reasoning模型.md) |

### 🤖 Agent / 自动化开发

| # | 文章标题 | 核心主题 | 文件 |
|---|---------|---------|------|
| 17 | Humanize带来的Codex使用范式变化，解锁Agent优化kernel上限 | Humanize、KernelPilot、AVO、Codex /goal、自动kernel优化 | [17_Humanize](articles/17_Humanize带来的Codex使用范式变化解锁Agent优化kernel上限.md) |
| 18 | 面向SGLang的自动驾驶开发：远程连接、CUDA Crash排查 | SGLang SKILL、远程开发、CUDA Debug、Benchmark、Profile | [18_SGLangAuto](articles/18_面向SGLang的自动驾驶开发远程连接CUDA%20Crash排查自动benchmark与Profile.md) |
| 19 | 记录下SGLang开发，优化，debug的技巧之大SKILL时代已来临 | Codex + GPT5.4、Agent编程、SKILL驱动开发、kernel优化 | [19_SKILL时代](articles/19_记录下SGLang开发优化debug的技巧之大SKILL时代已来临.md) |

### 📢 运营/其他

| # | 文章标题 | 核心主题 | 文件 |
|---|---------|---------|------|
| 02 | GiantPandaCV 2年半从0到35000+粉丝的经验 | 技术公众号运营、内容策略、粉丝增长 | [02_运营经验](articles/02_GiantPandaCV%202年半从0到35000粉丝的经验.md) |

---

## 核心知识点速查

### CUDA 优化技巧

| 技术点 | 说明 | 来源文章 |
|-------|------|---------|
| Two Pass 归约 | 先按 block 归约，再归约 block 结果 | #04 |
| Warp 级同步 | 使用 `__syncwarp()` 替代 `__syncthreads()` | #04 |
| 算术强度优化 | 每个线程处理多个元素，提升计算密度 | #04 |
| 向量化加载 | 使用 `int4` 类型提升访存带宽 | #04 |
| atomicAdd | One Pass 归约，减少 kernel 启动次数 | #04 |
| B200 带宽优化 | Blackwell 7.7TB/s HBM3e 带宽利用 | #11 |
| FlashDecoding++ | 消除 partial softmax 同步、Flat GEMM 优化 | #13 |

### Triton 编程要点

| 技术点 | 说明 | 来源文章 |
|-------|------|---------|
| Block-wise 编程 | Triton 编程粒度是 Block 而非 Thread | #08 |
| program_id | 使用 `tl.program_id()` 获取当前 block ID | #08 |
| 向量化访存 | Triton 自动生成 `ld.global.v4.b32` 等指令 | #08 |
| PTX 分析 | 设置 `TRITON_CACHE_DIR` 查看生成代码 | #07, #08 |
| Fused Kernel | 将多个操作融合到单个 kernel 减少访存 | #07 |
| num_stages | 控制多级流水线，实现 `cp.async` 异步拷贝 | #07 |
| Merge Attention | 分块 Attention 结果合并，用于 Chunked-Prefill | #05 |

### vLLM / SGLang 推理优化

| 技术点 | 说明 | 来源文章 |
|-------|------|---------|
| PagedAttention | KV-cache 分页管理，减少内存碎片 | #03, #15 |
| torch.compile | `fullgraph=True` + `max-autotune` 模式 | #03 |
| Flash Attention v3 | 最新版 Flash Attention 加速 | #03 |
| FP8 量化 | 降低显存占用提升吞吐 | #03, #11 |
| CUDAGraphs | 消除 CPU-GPU 同步开销 | #03 |
| PipeFusion | DiT 流水线并行，利用 Input Temporal Redundancy | #10 |
| SGLang Diffusion | 扩散模型推理引擎，比 Diffusers 快 5.9x | #14 |
| Prefix Caching | RadixAttention 基数树复用 KV Cache | #15 |
| Expert Parallel | EPMoE、Group GEMM、All2All 优化 | #20 |

### CUTLASS / CuTe / GEMM

| 技术点 | 说明 | 来源文章 |
|-------|------|---------|
| Swizzle<B,M,S> | 共享内存重排，消除 Bank Conflict | #09 |
| ldmatrix | PTX 指令与共享内存 Bank 结构 | #09 |
| 逻辑 Layout | `(8, 32):(32, 1)` 等共享内存布局 | #09 |
| GEMM 内积/外积 | 矩阵乘法的基本算法形式 | #12 |
| MXFP8 Blockscaled | Blackwell 混合精度量化 GEMM | #11 |

### 端侧 AI / VLM

| 技术点 | 说明 | 来源文章 |
|-------|------|---------|
| MiniMind | 25.8M 参数轻量语言模型 | #01 |
| MobileCLIP | 11.4M 参数轻量视觉编码器 | #01 |
| 两阶段训练 | Stage 1: 训练 Projection；Stage 2: 端到端微调 | #01 |
| ONNX 导出 | 模型转换与移动端部署 | #01 |
| Qwen2.5-VL | Window Attention、动态帧率采样 | #06 |
| 多模态 RoPE | 视觉-文本统一位置编码 | #06 |
| 退化分类预训练 | 图像复原中的退化感知判别 | #16 |

### Agent / 自动化开发

| 技术点 | 说明 | 来源文章 |
|-------|------|---------|
| Humanize | 外部审查门禁 + 状态机，防止 Agent 过早停止 | #17 |
| Codex /goal | 原生长目标，让长任务进入官方运行时 | #17 |
| KernelPilot | 基于 Humanize 的 kernel 专用工作流 | #17 |
| AVO | Agentic Variation Operators，自主进化搜索 | #17 |
| SKILL 驱动开发 | 将开发经验抽取为 SKILL 复用 | #18, #19 |
| Auto Benchmark | 自动 benchmark 与 profile 分析 | #18 |
| CUDA Crash Debug | 自动排查 CUDA 崩溃 | #18 |

---

## 使用指南

### 如何查询知识

本 Skill 为**知识库型 Skill**，主要用于：
1. **技术问题解答** - 查询特定技术点的实现细节
2. **方案参考** - 寻找类似场景的优化方案
3. **代码参考** - 获取 kernel 实现思路和关键代码片段
4. **性能调优** - 了解优化技巧和最佳实践
5. **Agent 开发** - 了解如何用 Codex/Claude Code 自动化开发

### 增量更新方法

要添加新的公众号文章：
1. 下载新文章为 Markdown 格式，放入 `articles/` 目录
2. 按编号命名（如 `21_文章标题.md`）
3. 更新本 SKILL.md 的目录表格和知识点速查
4. 更新文章总数统计

---

## 文章原始链接

| # | 原文链接 |
|---|---------|
| 01 | https://mp.weixin.qq.com/s/8KtMrg2DShP1GhJn-wYKiw |
| 02 | https://mp.weixin.qq.com/s/OrdhchUNGVi0b7ZgsYfIIg |
| 03 | https://mp.weixin.qq.com/s/KRKqZdcTjfbAmhIPYDXwTQ |
| 04 | http://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247527109&idx=1&sn=1741588c322191b095f2ed734d20e61a |
| 05 | http://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247527742&idx=1&sn=78b89e3a19497bd335706e4866647706 |
| 06 | http://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247527847&idx=1&sn=1819180dba8b628a99a81fd57ca8dd5f |
| 07 | https://mp.weixin.qq.com/s/X-tpwHgwSmthxLzLpYzW7A |
| 08 | http://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247527533&idx=1&sn=6a186f514e42e543c0d3ff3567ff7656 |
| 09 | https://mp.weixin.qq.com/s/Vktpz0uV2yqjgI8c689eBQ |
| 10 | https://zhuanlan.zhihu.com/p/699612077 |
| 11 | https://mp.weixin.qq.com/s/rJBw4B4e7ObQ4CFP-8fXjA |
| 12 | https://mp.weixin.qq.com/s/21ztNgVr7sUNu1ajRJfOog |
| 13 | http://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247519603&idx=1&sn=2572a1e9d42cf21bc581a61c8173c31c |
| 14 | https://mp.weixin.qq.com/s/lnCp1FXz71s_yhe0IS0jzQ |
| 15 | https://zhuanlan.zhihu.com/p/25783337136 |
| 16 | https://blog.csdn.net/csdn_xmj/article/details/146249416 |
| 17 | https://www.163.com/dy/article/KT0K0IRK05568W0A.html |
| 18 | https://finance.sina.com.cn/wm/2026-04-01/doc-inhtaktw2132411.shtml |
| 19 | https://k.sina.com.cn/article_5952915720_162d2490806703jsuk.html |
| 20 | https://blog.csdn.net/csdn_xmj/article/details/147929348 |

---

*本 Skill 由 Kimi Work 自动整理生成，仅供学习研究使用。*
