# 研究维度5：GPU架构差异对性能建模的影响

## 维度概述

本研究深入调查NVIDIA三代GPU架构（Ampere SM80、Hopper SM90、Blackwell SM100）的微架构差异及其对性能建模方法论的影响。重点关注Blackwell相对于Hopper的关键架构变化，包括统一INT32/FP32执行单元、第五代Tensor Core（tcgen05.mma）、Tensor Memory（TMEM）、L1/SMEM减半与统一L2缓存增大、以及全新的异步流水线模型。研究涵盖微架构参数对比、指令流水线差异、内存层次结构变化，以及这些变化如何影响性能建模公式的构建与校准。

---

## 核心发现

### 1. Blackwell SM100核心架构变化

**Claim**: Blackwell引入第五代Tensor Core，使用warp级单线程指令`tcgen05.mma`替代Hopper的warp-group同步`wgmma`，实现2.9-11.6倍更低的单指令延迟（~11周期），并减少调度器停滞18-23%。[^40^][^54^]
**Source**: Microbenchmarking NVIDIA's Blackwell Architecture
**URL**: https://arxiv.org/html/2512.02189v1
**Date**: 2025-12-01
**Excerpt**: "Blackwell achieves 2.9-11.6x lower single-instruction latency than Hopper. Crucially, this latency remains nearly constant across tile sizes (11.0-11.4 cycles), whereas Hopper scales linearly with tile width. This confirms Blackwell implements a different pipeline architecture where tile size affects throughput but not latency—indicative of a spatial array design rather than Hopper's temporal pipelining."
**Context**: 微基准测试论文，系统表征Blackwell架构
**Confidence**: high

**Claim**: Blackwell每个SM集成256KB专用TMEM（512列x128通道x32位），提供16 TB/s读取带宽和8 TB/s写入带宽，与共享内存带宽叠加不竞争，将缓存未命中延迟降低58%（从Hopper的~1000周期降至~420周期）。[^40^][^39^]
**Source**: Microbenchmarking NVIDIA's Blackwell Architecture
**URL**: https://arxiv.org/html/2512.02189v1
**Date**: 2025-12-01
**Excerpt**: "TMEM achieves 420 clock cycles for end-to-end memory access in cache-miss scenarios, representing a 58% reduction compared to Hopper's 1000-cycle global memory latency...TMEM provides 16 TB/s read bandwidth and 8 TB/s write bandwidth per SM, and this bandwidth operates additively with L1/SMEM bandwidth rather than competing for the same resources."
**Context**: Blackwell微架构详细分析
**Confidence**: high

**Claim**: Blackwell引入统一INT32/FP32执行单元（64-lane标量ALU），通过时间复用替代Hopper的独立流水线（128 FP32 + 64 INT32 ALU），将混合工作负载延迟降低近一半。纯FP32/INT32真延迟保持4周期不变，但Blackwell混合1:1序列真延迟为15.96周期（vs Hopper 31.62周期）。[^41^][^16^]
**Source**: Unified INT32/FP32 Execution Unit in Blackwell / Blackwell vs Hopper Deep Dive
**URL**: https://intuitionlabs.ai/articles/blackwell-vs-hopper-gpu-architecture-comparison
**Date**: 2026-02-28
**Excerpt**: "Mixed-type instruction chains on Blackwell increase true latency by a factor of 4-6x (to 16-26 cycles), while Hopper suffers an 8-11x blow-up (32-44 cycles). Thus, Blackwell substantially reduces the performance penalty for mixed FP32/INT32 workloads."
**Context**: 微架构对比分析
**Confidence**: high

**Claim**: Blackwell将L1/SMEM从Hopper的256KB减半至128KB（GB203消费级），但通过统一65MB单体式L2（per die，总计126MB双die）补偿。L2延迟从Hopper分区的~273周期增至~358周期，但更大容量减少了高并发下的容量未命中。[^11^][^16^]
**Source**: Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks
**URL**: https://arxiv.org/html/2507.10789v2
**Date**: 2025-07-21
**Excerpt**: "GH100 features up to 256 KB of combined L1/shared memory per SM, whereas GB203 reduces this to 128 KB/SM...the GB203 employs a monolithic L2 cache shared by all GPCs...GB203 exhibits a fixed latency of approximately 358 cycles, while the GH100 achieves a lower latency of around 273 cycles."
**Context**: GB203 (RTX 5080) vs GH100 (H100)微基准对比
**Confidence**: high

### 2. Hopper SM90架构特征

**Claim**: Hopper架构包含132个SM，每个SM有256KB统一L1/SMEM（最大228KB可配置为共享内存）、TMA单元、4个处理块（各含CUDA Core和第4代Tensor Core），所有SM共享50MB L2缓存。[^37^][^202^]
**Source**: AsyncSparse: Accelerating Sparse Matrix-Matrix Multiplication on Asynchronous GPU Architectures
**URL**: https://arxiv.org/html/2604.17834v1
**Date**: 2026-04-20
**Excerpt**: "Each of 132 SMs contains a TMA unit, a 256 KB shared memory, and 4 processing blocks with CUDA cores and a 4th generation Tensor Core. All SMs share a 50 MB L2 cache backed by 80 GB high-bandwidth global memory (HBM3)."
**Context**: Hopper架构概述图与描述
**Confidence**: high

**Claim**: Hopper引入warp-group级(128线程)`wgmma.mma_async`指令，支持从共享内存直接读取操作数、通过显式fence/commit/wait协议异步执行。微基准显示wgmma在N>=64时达到Hopper 989 TFLOPS峰值的96%以上，而遗留mma.sync仅~63%。[^37^][^154^]
**Source**: AsyncSparse / Dissecting NVIDIA Hopper Architecture
**URL**: https://arxiv.org/html/2604.17834v1 / https://arxiv.org/html/2501.12084v2
**Date**: 2026-04-20 / 2025-01-21
**Excerpt**: "wgmma achieves over 96% of Hopper's 989 TFLOP/s BF16 peak when N>=64, whereas the legacy mma.sync path reaches only ~63% of peak."
**Context**: WGMMA性能表征
**Confidence**: high

**Claim**: Hopper的TMA（Tensor Memory Accelerator）由单一线程发起`cp.async.bulk.tensor`指令，硬件处理所有地址计算、swizzle和传输，释放其他线程用于计算，消除每个线程30-40个寄存器的地址运算开销。TMA吞吐在tile大小超过4KB时饱和H100的3.35 TB/s HBM3带宽。[^37^]
**Source**: AsyncSparse paper
**URL**: https://arxiv.org/html/2604.17834v1
**Date**: 2026-04-20
**Excerpt**: "A single thread issues cp.async.bulk.tensor with tile coordinates, and TMA handles all address computation, swizzle, and transfers up to 5D tensors between global and shared memory...eliminates 30-40 registers per thread previously consumed by address arithmetic. Its throughput saturates the H100's 3.35 TB/s HBM3 bandwidth when the tile size exceeds 4 KB."
**Context**: TMA硬件加速数据传输
**Confidence**: high

**Claim**: Hopper的Distributed Shared Memory（DSM）允许cluster内（最多16个CTA）线程块访问彼此的共享内存，跨SM延迟181-213周期，比通过全局内存传输快5.28倍。[^65^]
**Source**: Dissecting the NVIDIA Hopper Architecture through Microbenchmarking
**URL**: https://arxiv.org/html/2501.12084v1
**Date**: 2025-01-21
**Excerpt**: "When the cluster size increases to two, the access latency between SMs is 181 cycles, a 32% reduction compared to L2 cache...utilizing DSM can reduce this latency by 5.28x, which is close to the 7x reduction reported in NVIDIA's official documentation."
**Context**: DSM微基准测试
**Confidence**: high

### 3. Ampere SM80架构特征

**Claim**: Ampere架构(A100)有108个SM，每个SM有192KB统一L1/SMEM（最大162KB可配置为共享内存），引入`cp.async`异步全局到共享内存拷贝、第3代Tensor Core支持TF32/BF16、2:4结构化稀疏性，L2缓存40MB。[^210^][^55^]
**Source**: JAX Scaling Book / CUTLASS Ampere文档
**URL**: https://jax-ml.github.io/scaling-book/gpus/
**Date**: 2025-06-23
**Excerpt**: "A100: 108 SMs, 192kB SMEM capacity/SM, 40MB L2, 80GB HBM...introduced cp.async for asynchronous shared memory copies, TF32 tensor cores, 2:4 structured sparsity."
**Context**: GPU规格总结
**Confidence**: high

### 4. 架构差异对性能建模的影响

**Claim**: 朴素的Roofline模型在现代GPU上产生>95%误差，因为它(1)数据手册峰值高估可持续吞吐1.5-2倍；(2)忽略串行流水线阶段（Blackwell上TMA→TMEM→Tensor Core阶段增加延迟）；(3)使用单一带宽数值，错过多级缓存层次。Stage-centric分析模型在Blackwell上实现1.31% MAE。[^34^][^155^]
**Source**: Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures
**URL**: https://arxiv.org/html/2605.04178v1
**Date**: 2026-05-05
**Excerpt**: "naive roofline baselines showing >95% error...Our models are interpretable, parameterized by measured hardware values, and accurate to within 1-5% MAE, compared to over 95% error for naive roofline."
**Context**: 跨架构分析性能建模论文
**Confidence**: high

**Claim**: Blackwell的显式阶段(TMA→TMEM→Tensor Core→Sync)允许使用可测量延迟和带宽的阶段中心建模；而Roofline的单个max()无法表示它们的串行化。MI300A的重叠是隐式和占用率驱动的。这些结构差异需要架构特定的模型项，而非通用Roofline中的参数替换。[^34^]
**Source**: Microbenchmark-Driven Analytical Performance Modeling
**URL**: https://arxiv.org/html/2605.04178v1
**Date**: 2026-05-05
**Excerpt**: "Blackwell's explicit stages (TMA→TMEM→Tensor Core→Sync) allow stage-centric modeling with measurable latencies and bandwidths; roofline's single max() cannot represent their serialization...These structural differences require architecture-specific model terms, not just parameter substitution in a generic roofline."
**Context**: 建模方法论分析
**Confidence**: high

**Claim**: FlashAttention-4揭示Blackwell存在"非对称硬件扩展"问题：Tensor Core FP16/BF16吞吐比Hopper翻倍(2.25 PFLOPS vs 1 PFLOPS)，但共享内存带宽(128 bytes/clock/SM)和指数单元吞吐(16 ops/clock/SM)保持不变。在Blackwell上，典型注意力工作负载的共享内存流量和指数操作主导执行时间，超过MMA计算25-60%。[^38^]
**Source**: FlashAttention-4: Algorithm and Kernel Pipelining Co-Design
**URL**: https://arxiv.org/html/2603.05451v1
**Date**: 2026-03-05
**Excerpt**: "Although Blackwell B200 doubles the tensor core throughput compared to Hopper H100 (2.25 PFLOPS vs. 1 PFLOPS for FP16/BF16), other functional units (shared memory bandwidth, exponential units, and integer/floating point ALUs) scale more slowly or remain unchanged. As a result, non-MMA resources emerge as bottlenecks...shared memory traffic and exponential operations now dominate execution time, exceeding MMA compute by 25-60%."
**Context**: FlashAttention-4论文中的硬件瓶颈分析
**Confidence**: high

**Claim**: 通过更新分析模型的硬件参数文件（无需重新推导公式），可以将Blackwell模型应用于Hopper H200。模型框架由架构如何累积结果决定：专用TMEM（Blackwell阶段模型）vs VGPR累加器（CDNA波前模型）。对于未来的GPU（如Rubin、CDNA4），预期仅参数更新即可。[^34^][^155^]
**Source**: Microbenchmark-Driven Analytical Performance Modeling
**URL**: https://arxiv.org/html/2605.04178v1
**Date**: 2026-05-05
**Excerpt**: "By updating parameters of our analytical models, they can be applied to H200 and MI250X...For future GPUs within these families (Rubin, CDNA4), we expect parameter-only updates to suffice; a fundamentally new accumulation mechanism would require one new stage term."
**Context**: 模型可移植性分析
**Confidence**: high

---

## 技术方法论详解

### 1. Stage-Centric分析性能模型（Blackwell）

针对Blackwell架构的阶段中心分析模型由Jarmusch等人(2026)提出，模型公式如下：

**执行时间公式**（Hong-Kim框架）：
```
T_exec = max(T_compute, T_memory) + T_overhead
```

**Blackwell具体阶段分解**：
```
T_step = max(T_compute, T_io^eff) + T_sync + O_misc

其中：
- T_compute = FLOPs / (Sustained_TensorCore_TFLOPS)
- T_io^eff = (1-α)(T_tma + T_decomp) + T_sync
- T_tma = bytes(tile) / B_TMA + L_TMA_setup
- T_sync = mbarrier_wait + pipeline_bubble
- O_misc = TMEM管理开销 + pipeline bubbles
```

**稳态流水线步骤**：
```
T_step_pipelined = max(T_tma, T_decomp, T_compute, T_sync) + ε
```

**关键模型参数**（Blackwell B200）：

| 参数 | 数值 | 来源 |
|------|------|------|
| TMEM Read BW | 16 TB/s/SM | 微基准 |
| TMEM Write BW | 8 TB/s/SM | 微基准 |
| TMEM Latency | 420 cycles | 微基准 |
| tcgen05.mma Latency | 11.0-12.6 cycles | 微基准 |
| FP16 Tensor Throughput | 1929 TFLOPS (96.5% peak) | 微基准 |
| FP8 Tensor Throughput | 3851 TFLOPS (96.3% peak) | 微基准 |
| FP4 Tensor Throughput | 7702 TFLOPS (96.3% peak) | 微基准 |
| HBM Sustained BW | 6.8-7.1 TB/s | 微基准 |
| L1/SMEM per SM | 128KB (GB203) / 256KB (SM100) | 数据手册 |
| L2 Cache | 65MB per die / 126MB total | 数据手册 |
| NV-HBI BW | 10 TB/s | 数据手册 |

### 2. Tensor Core指令流水线对比

**各代Tensor Core指令特征**：

| 特征 | Ampere (SM80) | Hopper (SM90) | Blackwell (SM100) |
|------|--------------|---------------|-------------------|
| 指令 | `mma.sync` | `wgmma.mma_async` | `tcgen05.mma` |
| 执行粒度 | Warp (32 threads) | Warp-group (128 threads) | Warp (32 threads), 单线程发起 |
| A/B操作数位置 | Registers | SMEM | SMEM (A可来自TMEM) |
| C/D累加器位置 | Registers | Registers | TMEM |
| 同步模型 | 同步 | 异步 (fence/commit/wait) | 完全异步 |
| 延迟特性 | 依赖tile大小 | 线性随tile宽度增长 | 恒定~11周期，与tile无关 |
| 流水线类型 | 时间流水线 | 时间流水线 | 空间阵列 |
| BF16峰值 | 312 TFLOPS | 989 TFLOPS | 1929-2250 TFLOPS |
| 新精度 | TF32, BF16 | FP8 | FP4, FP6 |

**Blackwell tcgen05.mma延迟-吞吐特性**：

| Input (A/B) | Accum (C/D) | Shape | Latency (cycles) | Throughput (TFLOPS) |
|-------------|-------------|-------|------------------|---------------------|
| FP16 | FP16 | m64n8k16 | 11.2 | 964.8 |
| FP16 | FP32 | m64n8k16 | 11.5 | 482.4 |
| FP8 | FP16 | m64n8k16 | 11.8 | 1925.3 |
| FP8 | FP32 | m64n8k16 | 12.1 | 1912.8 |
| FP6 | FP16 | m64n8k16 | 12.3 | 2567.2 |
| FP4 | FP16 | m64n8k16 | 12.6 | 3850.1 |
| INT8 | INT32 | m64n8k16 | 11.9 | 3928.5 |

**关键洞察**：尽管吞吐从FP16的965 TFLOPS到FP4的3850 TFLOPS（4x差异），延迟仅从11.2到12.6周期（1.12x差异），确认吞吐扩展通过更宽数据路径而非更深流水线实现。

### 3. 内存层次结构对比

**三代架构内存参数对比**：

| 参数 | Ampere (A100) | Hopper (H100) | Blackwell (B200) |
|------|--------------|---------------|-------------------|
| SM数量 | 108 | 132 | 148 (74x2 dies) |
| 寄存器文件/SM | 256 KB | 256 KB | 256 KB + 256KB TMEM |
| L1+SMEM/SM | 192 KB (SMEM max 162KB) | 256 KB (SMEM max 228KB) | 128/256 KB (SMEM max 99/228KB) |
| L2 Cache | 40 MB | 50 MB (2x25MB分区) | 126 MB (2x63MB单体式) |
| HBM | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e |
| HBM Bandwidth | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| L2 Latency | ~200 cycles | ~273 cycles | ~358 cycles |
| Shared Mem BW | ~19 TB/s | ~33 TB/s | ~33 TB/s (additive with TMEM) |
| NVLink BW | 600 GB/s | 900 GB/s | 1800 GB/s |

### 4. 统一INT32/FP32执行单元技术细节

**Blackwell统一ALU特性**：
- 64-lane标量ALU簇，每个周期可执行FP32 FMA或INT32 MAD，但不能同时执行两者
- 纯FP32/INT32真延迟：4周期（与Hopper相同）
- 完成延迟：INT32 16.97周期，FP32 7.97周期（Blackwell）
- 混合1:1工作负载真延迟：Blackwell 15.96周期 vs Hopper 31.62周期
- 混合2:1工作负载真延迟：Blackwell 26.28周期 vs Hopper 43.54周期

**吞吐公式**：
```
T_fp32 = (N_fp32_units * f_clk) / L_fp32_completion
其中 N_fp32_units = 64, f_clk = 2.2 GHz, L_fp32_completion = 8 cycles
```

### 5. 性能建模的可移植性原则

**模型可移植性层级**：

| 架构演进 | 模型调整 | 示例 |
|----------|---------|------|
| 同族参数更新 | 仅更新参数文件 | B200→H200, MI300A→MI250X |
| 同族新增功能 | 添加新阶段项 | TMEM→新增TMEM阶段项 |
| 跨族结构差异 | 切换模型框架 | Blackwell(stage) vs CDNA(wavefront) |
| 全新累加机制 | 重新推导公式 | VGPR累加器→TMEM专用存储 |

**MAE阈值判断**：
- MAE < 15% 参数更新后：框架足够
- MAE > 30%：需要结构变化

---

## 架构适配性分析

### 1. 编程模型与指令集差异

**三代架构PTX指令演进**：

| 功能 | Ampere (sm80) | Hopper (sm90) | Blackwell (sm100) |
|------|--------------|---------------|-------------------|
| 加载 | `cp.async` | `cp.async.bulk.tensor` (TMA) | `cp.async.bulk.tensor` (TMA) |
| 计算 | `mma.sync` | `wgmma.mma_async` | `tcgen05.mma` |
| 同步 | `__syncthreads()` | `mbarrier` + `wgmma.fence` | `mbarrier` + TMEM管理 |
| 数据移动 | `ldmatrix`→registers | SMEM→TC direct | SMEM/TMEM↔TC |
| 累加器 | 隐式在registers | 隐式在registers | 显式在TMEM |
| 专用指令 | - | `setmaxnreg`, DSM | `tcgen05.alloc/dealloc/ld/st/cp` |

### 2. 不同架构的Kernel适配策略

**vLLM/CUTLASS架构分发模式**：

```cpp
int version_num = get_sm_version_num();
if (version_num >= 120)      cutlass_sm120();  // RTX Blackwell
else if (version_num >= 100) cutlass_sm100();  // Blackwell DC
else if (version_num >= 90)  cutlass_sm90();   // Hopper
else if (version_num == 89)  cutlass_sm89();   // Ada Lovelace
else if (version_num >= 80)  cutlass_sm80();   // Ampere
```

**架构特定优化策略**：

| 架构 | 关键优化 | 典型性能提升 |
|------|---------|------------|
| SM80 (Ampere) | `cp.async`多级流水线, `ldmatrix`, SMEM swizzle | 5-15% |
| SM90 (Hopper) | Warp specialization, TMA, WGMMA异步流水线 | 20-40% |
| SM100 (Blackwell) | TMEM累加器, tcgen05.mma, 2-CTA模式, 低精度FP4/6 | 30-60% |

### 3. 非对称硬件扩展的影响

**Blackwell资源扩展比例**（相对Hopper）：

| 资源 | Blackwell vs Hopper | 影响 |
|------|---------------------|------|
| Tensor Core (BF16) | 2.0x (8192→16384 ops/clock/SM) | 不再是唯一瓶颈 |
| Shared Memory BW | 1.0x (128 bytes/clock/SM) | 成为新瓶颈 |
| Exponential Unit | 1.0x→2.0x (16→32 ops/clock, B300) | forward pass瓶颈 |
| FP32 ALU | 0.5x (unified 64 vs 128) | 混合工作负载影响 |
| TMEM | 全新 (0→256KB/SM) | 减少SMEM压力 |

**FlashAttention-4的适配策略**：
1. 重新设计流水线，利用完全异步MMA操作和更大tile size
2. 软件模拟指数函数（多项式逼近），使用FMA单元替代MUFU
3. 利用TMEM存储中间结果，减少共享内存流量
4. 2-CTA MMA模式减少原子操作和共享内存竞争

### 4. 各代warp调度器差异

| 特性 | Hopper | Blackwell (GB203) |
|------|--------|---------------------|
| 最大活跃warp | 64 warps/SM | 64 warps/SM |
| 调度器数量 | 4 per SM | 4 per SM |
| 最大ILP饱和 | ILP=5, 29 warps | ILP=6, 25 warps |
| 完成延迟(FP16 MMA) | 1.65625 cycles | 1.21094 cycles |
| 混合工作负载延迟 | 高(31-44 cycles) | 中(16-26 cycles) |
| 调度器优化目标 | 批量并发，深度缓冲 | 低精度高ILP，干净控制流 |

---

## 工具与资源

### 微基准测试工具

| 工具/项目 | 用途 | URL |
|-----------|------|-----|
| Blackwell Microbenchmark Suite | 全面表征B200 TMEM/TC/DE/内存 | arxiv.org/abs/2512.02189 |
| Hopper Microbenchmark Suite | H100/H800指令延迟吞吐 | arxiv.org/abs/2501.12084 |
| MT4G | GPU拓扑和参数自动发现 | arxiv.org/abs/2511.05958 |
| Sim-FA | FlashAttention流水线模拟器 | arxiv.org/abs/2605.00555 |
| pyptx | Python PTX DSL for Hopper/Blackwell | github.com/patrick-toulme/pyptx |

### 开源项目参考

| 项目 | 架构支持 | 关键特性 |
|------|---------|---------|
| CUTLASS 3.x | SM80/SM90/SM100 | Warp specialization, TMA, tcgen05 |
| FlashAttention-3 | SM90 (Hopper) | WGMMA异步流水线 |
| FlashAttention-4 | SM100 (Blackwell) | TMEM, 2-CTA, 非对称瓶颈缓解 |
| ThunderKittens | SM90/SM100 | AI kernel DSL |
| TileLang | SM70/SM80/SM89/SM90 | 自动布局推断 |

### 关键性能计数器

| 计数器 | 用途 |
|--------|------|
| `tensor__pipe_tensor_cycles_elapsed` | Tensor Core利用率 |
| `sm__warps_launched` | Occupancy |
| `l1tex__t_sectors_pipe_lsu_mem_global` | 全局内存流量 |
| `sm__pipe_tensor_cycles_elapsed` | Tensor Core活跃周期 |
| `smsp__warp_issue_stalled_barrier` | mbarrier等待停滞 |

---

## 关键引用列表

[^34^] Jarmusch et al., "Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures," arXiv:2605.04178, May 2026.

[^37^] Zheng et al., "AsyncSparse: Accelerating Sparse Matrix-Matrix Multiplication on Asynchronous GPU Architectures," arXiv:2604.17834, April 2026.

[^38^] Zadouri et al., "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling," arXiv:2603.05451, March 2026.

[^39^] "Blackwell GPU Architecture," EmergentMind, Dec 2025. https://www.emergentmind.com/topics/blackwell-gpu-architecture

[^40^] Jarmusch et al., "Microbenchmarking NVIDIA's Blackwell Architecture: An In-depth Architectural Analysis," arXiv:2512.02189, Dec 2025.

[^41^] "Unified INT32/FP32 Execution Unit in Blackwell," EmergentMind, Nov 2025. https://www.emergentmind.com/topics/unified-int32-fp32-execution-unit

[^16^] "Blackwell vs Hopper: A Deep Dive GPU Architecture Comparison," IntuitionLabs, Feb 2026. https://intuitionlabs.ai/articles/blackwell-vs-hopper-gpu-architecture-comparison

[^11^] Jarmusch et al., "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks," arXiv:2507.10789, July 2025.

[^65^] "Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis," arXiv:2501.12084, Jan 2025.

[^210^] "How to Think About GPUs," JAX Scaling Book, June 2025. https://jax-ml.github.io/scaling-book/gpus/

[^54^] CUTLASS Documentation, "Tensor Core Programming," 2026. https://mintlify.com/NVIDIA/cutlass/concepts/tensor-cores

[^55^] CUTLASS Documentation, "Ampere Architecture (SM80/86)," 2026.

[^56^] "tcgen05 for dummies," gau-nernst.github.io, Dec 2025.

[^58^] SemiAnalysis, "In-Depth Breakdown: Full Details of the Blackwell Architecture," April 2026.

[^154^] "Dissecting the NVIDIA Hopper Architecture through Microbenchmarking," arXiv:2501.12084v2, Sep 2025.

[^155^] Jarmusch & Chandrasekaran, "Microbenchmark-Driven Analytical Performance Modeling," arXiv:2605.04178v1, May 2026.

[^199^] "Sim-FA: A GPGPU Simulator Framework for Fine-Grained FlashAttention Pipeline Analysis," arXiv:2605.00555, May 2026.

[^200^] "OpenAI Triton Kernel Development on GPU Cloud," Spheron, May 2026.

[^202^] "GPU Architecture & Compute Fundamentals," NCP-AII Guide, April 2026.

[^203^] "Matching CUDA arch and CUDA gencode," arnon.dk, April 2026.

[^204^] "Advanced CUTLASS Architecture and Kernel Optimization," Oboe Study Guide, March 2026.

[^205^] "vLLM对不同GPU SM架构的适配机制深度剖析," March 2026.

[^206^] "CUDA Programming Complete Guide," youngju.dev, March 2026.

---

## 待深入区域

1. **Blackwell数据中心级(SM100) vs 消费级(GB203/SM120)差异**: 现有研究主要基于B200/GB203，SM100的256KB TMEM和SMEM配置与GB203的128KB存在差异，需要更多SM100-specific数据。

2. **Rubin架构预期影响**: 下一代Rubin架构（预计H2 2026）可能引入的进一步变化（如tcgen08、更大的TMEM）对当前模型的影响。

3. **FP4/FP6数值稳定性建模**: 性能模型中如何量化低精度格式的数值误差与性能trade-off。

4. **多die NV-HBI延迟建模**: 跨die访问的NUMA效应对性能模型的影响，目前数据有限。

5. **2-CTA MMA模式的性能模型扩展**: 如何形式化TPC范围内协同CTA的性能模型。

6. **功耗建模的架构差异**: 不同架构在功耗/性能trade-off上的差异，特别是在低精度模式下。

7. **实际工作负载（LLM推理/训练）的架构特定优化策略**: 如何将微架构模型转化为端到端的性能优化建议。

8. ** occupancy限制的新因素**: Blackwell中TMEM分配、统一ALU争用等对occupancy的影响需要更深入分析。

---

*报告生成时间: 2025年*
*搜索查询次数: 25+*
*覆盖架构: Ampere SM80, Hopper SM90, Blackwell SM100/GB203*
