---
name: persistent-kernel-utilization
description: |
  Persistent Kernel 的 SM/Memory 利用率极致优化指南。系统化解决五大利用率杀手：
  Wave Quantization 尾效、负载不均、Occupancy 不足、Pipeline 空泡、Epilogue-Mainloop 未重叠。
  覆盖 Stream-K、Hybrid Stream-K、CLC 动态调度、Warp Specialization、Multistage Pipeline、
  TMA Multicast、TMEM Double Buffering、Register Rebalancing (setmaxnreg)、Hilbert-Curve Scheduling
  等优化技术。适用场景：persistent kernel 利用率低、SM 空泡严重、TC 利用率不达标、
  需要极致榨干 GPU 性能的场景。
triggers:
  - "persistent kernel 利用率"
  - "SM 利用率"
  - "TC 利用率"
  - "occupancy"
  - "wave quantization"
  - "Stream-K"
  - "warp specialization"
  - "multistage pipeline"
  - "TMA multicast"
  - "setmaxnreg"
  - "Hilbert curve"
  - "epilogue mainloop 重叠"
  - "极致优化"
  - "GPU 性能榨干"
---

# Persistent Kernel 的 SM/Memory 利用率问题与极致优化指南

> 基于 Hopper/Blackwell 架构、CUTLASS 实践和 NCU 分析经验

---

## 一、Persistent Kernel 会导致 SM/Memory 利用率降低吗？

**答案是：设计不当的 Persistent Kernel 确实会降低利用率，但这不是 Persistent 模式本身的问题，而是具体实现中的陷阱。**

Persistent Kernel 的核心收益是**消除 CTA launch 开销**和**重叠 Epilogue-Mainloop**。但如果以下任何一个环节没做好，利用率反而会比 Non-persistent 更差：

### 1.1 利用率降低的五大根因

| 根因 | 表现 | 影响 |
|------|------|------|
| **Wave Quantization / 尾效** | 最后一波 wave 只有部分 SM 有工作 | SM 利用率骤降 |
| **Load Imbalance（负载不均）** | 静态调度下某些 SM 的 tile 工作量更大 | 部分 SM 提前空闲 |
| **Occupancy 不足** | 每个 SM 只跑少量 warp，warp slot 未填满 | SM 内部并行度浪费 |
| **Pipeline 未填满 / 同步间隙** | TMA load 和 MMA compute 之间有空泡 | Tensor/Memory Pipe 交替空闲 |
| **Epilogue-Mainloop 未重叠** | 当前 tile 写回时，下一个 tile 未开始加载 | Persistent 的核心收益丧失 |

### 1.2 尾效（Wave Quantization）的量化

这是**最常见**的利用率杀手：

```
# 理论计算
num_sms = 148          # B200
cluster_size = 2         # 2-CTA cluster
max_concurrent_clusters = num_sms / cluster_size = 74

total_tiles = (M/bM) * (N/bN)  # 总输出 tile 数
waves = ceil(total_tiles / max_concurrent_clusters)
full_waves = floor(total_tiles / max_concurrent_clusters)
tail_clusters = total_tiles % max_concurrent_clusters

# 尾效损失比例
tail_waste = (max_concurrent_clusters - tail_clusters) / max_concurrent_clusters

# 例子：total_tiles = 75, max_concurrent = 74
# waves = 2, tail = 1
# 第 2 波只有 1 个 cluster 工作，73 个 cluster 的空闲 SM 被浪费
# tail_waste = 73/74 ≈ 98.6% 的 SM 在第 2 波空闲！
```

> **关键洞察**：尾效的相对影响随 workload 规模增大而减小。小规模 workload（< 10 waves）下尾效是主要瓶颈；大规模 workload 下，尾效被摊薄。

### 1.3 负载不均的量化

静态 Persistent 调度中：

```
# 静态分配：SM i 处理 tile i, i+num_sms, i+2*num_sms...
# 如果 tile 工作量不同（如 grouped GEMM 中不同 problem 的 K 不同）

# 用 NCU 验证
sm_cycles_max = ncu_metric("sm__cycles_active.max")
sm_cycles_min = ncu_metric("sm__cycles_active.min")
imbalance_ratio = sm_cycles_max / sm_cycles_min

if imbalance_ratio > 1.5:
    → 严重负载不均，SM 利用率被拖垮
```

### 1.4 Occupancy 不足的陷阱

Persistent Kernel 通常每个 SM 只 launch **1 个 block**（或 1 个 cluster），如果 block 设计不当：

```
# SM 资源上限（Hopper/Blackwell）
max_warps_per_sm = 64        # 64 warps = 2048 threads
max_threads_per_sm = 2048

# 常见错误：block 只有 4 warps (128 threads)
block_warps = 4
occupancy = block_warps / max_warps_per_sm = 4/64 = 6.25%

# 即使 persistent，SM 内部 93.75% 的 warp slot 空闲！
```

**正确的 Persistent Kernel 设计**：
- 使用 **Warp Specialization**，一个 block 内包含多个 warp group
- Producer warps (TMA load) + Consumer warps (MMA compute) + Epilogue warps
- 目标：block 内 warp 数接近 SM 上限（如 8-16 warps）

---

## 二、如何极致提高 SM/Memory 利用率

### 2.1 调度层：消灭尾效和负载不均

#### A. Stream-K（终极尾效解决方案）

Stream-K 的核心思想：**让 Persistent CTA 处理 fractional tiles**，而不是整 tile。

```
传统 Data-Parallel：
  SM 0: tile 0, 4, 8...    SM 1: tile 1, 5, 9...
  → 尾效：最后几个 tile 只有部分 SM 工作

Stream-K：
  所有 SM 协作处理 K 维度的切片
  每个 SM 处理一个 K-split 的片段，通过 atomic 累加
  → 无尾效！所有 SM 同时开始、同时结束
```

**代价**：
- 需要 atomic accumulation（GMEM 上的 partial sum reduction）
- 增加同步开销
- 适合 **小 workload**（tile 数 < 10×SM 数）或 **尾效严重** 的场景

#### B. Hybrid Stream-K（折中方案）

> Colfax Research 推荐：大部分 tile 用 Data-Parallel（L2 locality 好），只剩尾效部分用 Stream-K。

```
Phase 1 (Stream-K): 处理 1-2 个 full waves + 所有 partial tiles
  → 所有 SM 同时完成

Phase 2 (Data-Parallel): 剩余 tile 数是 SM 数的整数倍
  → 无尾效，且 L2 locality 好
```

#### C. Dynamic Persistent / CLC（Blackwell）

对于**负载不均**或**SM 被抢占**的场景：
- CLC 允许运行中的 cluster 偷取未 launch cluster 的工作
- 自动平衡 workload，无需 atomic
- 但均衡 workload 下可能因 L2 locality 下降而性能倒退

#### D. Tile Size 调优（减少尾效相对影响）

```
# 策略：减小 tile size → 增加 tile 数量 → 尾效占比降低
# 但 tile 不能太小，否则：
#   - TMA 启动开销占比增大
#   - MMA 的 arithmetic intensity 下降
#   - Epilogue 占比增大

# 经验法则：
#   - 目标 tile 数 > 10 × max_concurrent_clusters
#   - 或调整 tile size 使 total_tiles % max_concurrent ≈ 0
```

#### E. CTA Swizzling / Hilbert-Curve Scheduling

改善 L2 cache locality，间接提高 memory 利用率：

```
线性调度：tile (0,0) → (0,1) → (0,2) ...
  → 相邻 tile 可能访问不连续的 GMEM，L2 miss 高

Hilbert/Z-curve 调度：
  → 空间局部性更好，L2 hit rate 提升
  → 实测可提升 1-3% 性能（Aleksa Gordić 的优化路径中最后一步）
```

### 2.2 Pipeline 层：填满流水线，消灭空泡

#### A. Warp Specialization（必备）

```
# 理想 warp 分工（Blackwell warp-specialized GEMM）
Warp 0:     MMA (consumer)        → 跑 tcgen05.mma / wgmma
Warp 1:     Scheduler (CLC)       → 发起 try_cancel
Warp 2:     Mainloop Load (producer) → TMA load A/B
Warp 3:     Epilogue Load         → TMA load C (for epilogue fusion)
Warp 4-7:   Epilogue              → TMA store D, elementwise ops

# 关键：producer 和 consumer 是不同 warp，通过 mbarrier 同步
# 这样 LSU (TMA) 和 Tensor Pipe (MMA) 可以并行工作
```

#### B. Multistage Pipeline（掩盖 TMA 延迟）

```
# 核心思想：提前加载未来多个 stage 的数据
# Stage 0: TMA load tile N+0    → MMA compute tile N-2
# Stage 1: TMA load tile N+1    → MMA compute tile N-1
# Stage 2: TMA load tile N+2    → MMA compute tile N

# 典型配置：3-5 stage
#   - stage 太少：TMA 延迟暴露，MMA 等待
#   - stage 太多：SMEM 超限，occupancy 下降

# SMEM 预算计算：
smem_per_stage = smem_A + smem_B + smem_C_buffer
total_smem = num_stages * smem_per_stage + smem_epilogue
# 必须 < SM 的 SMEM 上限（Hopper: 228KB with opt-in）
```

#### C. TMA Multicast（Cluster 内数据共享）

```
# 同一 cluster 内的多个 CTA 共享 A/B operand
# 例如 2×2 cluster：
#   - 同一行的 2 个 CTA 共享 A tile
#   - 同一列的 2 个 CTA 共享 B tile
# 
# 效果：GMEM → SMEM 的带宽需求降低 2-4 倍
# 条件：cluster 内 CTA 必须 co-scheduled 在同一 GPC
```

#### D. TMA Async Store（Epilogue 重叠）

```
# 传统：MMA 完成 → 等所有 warp 同步 → 写回 GMEM → 开始下一个 tile
# Async Store：
#   MMA 完成 → 结果先写回 SMEM → TMA async copy SMEM→GMEM
#   同时，producer warp 已经开始加载下一个 tile 的 A/B

# PTX: cp.async.bulk.tensor (TMA store)
# 或者 Blackwell 的 tcgen05.cp (TMEM→GMEM)
```

#### E. TMEM Double Buffering（Blackwell）

```
# Blackwell 的 accumulator 在 TMEM 中
# Double buffer：
#   Buffer 0: MMA 写入当前 tile 的 accumulator
#   Buffer 1: Epilogue 读取上一个 tile 的 accumulator
# 
# 效果：MMA 和 Epilogue 在 TMEM 上 overlap
# 这是 Blackwell 相比 Hopper 的重要优势
```

### 2.3 微架构层：榨干每个 cycle

#### A. Register Rebalancing（`setmaxnreg`）

```cuda
// Hopper/Blackwell PTX: 动态调整 warp group 的 register 预算
// Producer warp-group (TMA load) 不需要很多寄存器
// Consumer warp-group (MMA) 需要大量寄存器存 accumulator

asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(256));  // consumer
// ... MMA compute ...
asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(128));  // producer
```

**效果**：把 producer 省下的 register 给 consumer，提高 MMA 的 tile size 或 accumulator 精度。

#### B. Skip Redundant Accumulator Initialization

```
# 传统：每个 tile 开始前 zeroing accumulator registers
# 优化：
#   Tile 0: MMA 指令用 Zero accumulator mode  →  C = A @ B
#   Tile 1+: MMA 指令用 Accumulate mode       →  C = A @ B + C
# 
# 节省：每个 tile 省去一次 zeroing 的指令开销
```

#### C. Bypass L1/L2 on Store（避免 Cache 污染）

```cuda
// 方法 1: write-through，绕过 L1/L2
__stwt(dst, value);  // store with write-through

// 方法 2: async store via TMA（推荐）
// 数据先写 SMEM，然后 TMA async copy 到 GMEM
// 这样 compute 和 store 在 pipeline 层面 overlap
```

#### D. Faster Barriers（PTX 级优化）

```
# 传统：__syncthreads() 或 cluster.sync()
# 优化：
#   - 用 mbarrier 替代 full barrier（只同步需要的 warp）
#   - 用 arrive/wait 分离同步点（split barrier）
#   - 用 targeted barrier（只同步数据依赖的 CTA，而非整个 cluster）

# 例如 Blackwell 的 umma_arrive_multicast：
#   只通知共享同一 operand tile 的 CTA，而非整个 cluster
```

### 2.4 Occupancy 层：填满 SM 的 Warp Slot

#### A. 最大化 Block 内的 Warp 数

```
# 目标：每个 SM 的 warp slot 尽量满
# Hopper/Blackwell: 64 warps per SM

# 设计：
#   - 1 cluster = 2 CTA (Blackwell 2-SM UMMA)
#   - 每个 CTA = 8 warps (256 threads)
#   - 1 cluster = 16 warps
#   - 如果 SM 只跑 1 cluster: occupancy = 16/64 = 25%
#   
#   这看起来低，但 warp specialization 下：
#   - 4 warps 做 MMA (consumer)
#   - 1 warp 做 TMA load (producer)
#   - 1 warp 做 scheduler
#   - 4 warps 做 epilogue
#   - 这些 warp 是**同时活跃**的，通过异步指令 overlap

# 关键：不是 occupancy 越高越好，而是**有效 warp 的并行度**要高
```

#### B. 控制 Register Pressure

```
# 如果 register 使用过多：
#   - 编译器 spill 到 local memory → LSU 压力增大
#   - 或 occupancy 被硬件限制（register file 是 per-SM 资源）

# 策略：
#   - 用 __launch_bounds__(max_threads, min_blocks) 指导编译器
#   - 用 setmaxnreg 动态调整（Hopper+）
#   - 减少不必要的 live variable
```

#### C. Shared Memory 优化

```
# SMEM 是 occupancy 的硬限制之一
# Hopper: 默认 100KB per SM (限制 1 block/SM if block uses >50KB)
#         opt-in 可到 228KB

# 策略：
#   - 用 cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)
#   - 减少 bank conflict（swizzle layout）
#   - 用 async copy (TMA) 减少 SMEM 中转
```

---

## 三、优化路径：从 Baseline 到 SOTA

以下数据来自 Aleksa Gordić 对 Hopper H100 上 GEMM 的优化实践，展示了每一步的收益：

| 优化步骤 | 性能 (TFLOP/s) | 提升 | 核心解决的问题 |
|---------|---------------|------|--------------|
| Baseline (warp-tiling) | 32 | — | — |
| + Tensor Cores + TMA | 317 | 9.9× | 用 TC 替代 CUDA Core |
| + 增大 output tile size | 423 | +33% | 提高 arithmetic intensity |
| + Pipeline (TMA↔TC overlap) | 498 | +18% | **消灭 LSU/TC 空泡** |
| + Tile growth (2 consumer WGs) | 610 | +22% | 提高 SM 内并行度 |
| + **Persistent Kernel** | 660 | +8% | **重叠 Epilogue-Mainloop** |
| + Faster PTX barriers | 704 | +7% | 减少同步开销 |
| + Clusters + TMA Multicast | 734 | +4% | 减少 GMEM 带宽 |
| + Micro-optimizations | 747 | +2% | 指令级优化 |
| + TMA Async Stores | 758 | +1% | Epilogue 完全重叠 |
| + Hilbert-curve scheduling | 764 | +1% | L2 locality |

**关键洞察**：
1. **Pipeline 设计**（第 3 步）是**最大单步提升**（+18%），直接解决 LSU/TC 空泡
2. **Persistent** 本身只带来 +8%，但如果前面 pipeline 没做好，persistent 的收益会被 epilogue  stall 吃掉
3. 后面的优化都是**1-4% 的边际收益**，需要前面的大框架正确

---

## 四、NCU 验证 Checklist

用以下指标验证你的 persistent kernel 是否达到了极致利用率：

### 4.1 SM 利用率

```bash
ncu --metrics \
  sm__cycles_active.avg,sm__cycles_active.min,sm__cycles_active.max,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  sm__inst_issued.avg.per_cycle_active
```

| 指标 | 目标 | 诊断 |
|------|------|------|
| `sm__cycles_active.max/min` | < 1.2 | 负载均衡 |
| `sm__warps_active.avg.pct` | > 50% | occupancy 充足 |
| `sm__inst_issued.avg.per_cycle_active` (IPC) | 2-4 | warp scheduler 饱和 |

### 4.2 Tensor/Memory Pipe 利用率

```bash
ncu --metrics \
  sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
```

| 模式 | 含义 |
|------|------|
| TC 高 + LSU 低 | MMA 密集，但可能 TMA 跟不上 |
| TC 低 + LSU 高 | Memory bound，TC 空泡 |
| TC 和 LSU 交替高 | ✅ 良好掩盖 |
| TC 和 LSU 同时高 | ⚠️ 可能 stage 数不足，pipeline 未填满 |

### 4.3 Stall 分解

```bash
ncu --metrics \
  smsp__warp_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active,\
  smsp__warp_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active,\
  smsp__warp_issue_stalled_barrier.avg.pct_of_peak_sustained_active,\
  smsp__warp_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active
```

| Stall 类型 | 目标 | 根因 |
|-----------|------|------|
| Long Scoreboard | < 20% | TMA/GMEM 延迟暴露 → 增加 stage |
| Short Scoreboard | < 10% | SMEM/barrier 等待 → 优化同步粒度 |
| Barrier | < 5% | 过度同步 → 用 targeted barrier |
| Math Pipe Throttle | < 10% | TC 背压 → 正常，但过高说明 warp 不足 |

### 4.4 PM Sampling 时间线

```bash
ncu --section PmSampling -o report_pm ./kernel
```

**观察**：
- 末尾是否有骤降？→ 尾效
- 是否有周期性锯齿？→ Pipeline stage 切换间隙
- LSU 和 TC 是否交替填满？→ Pipeline 掩盖效果

---

## 五、一句话总结

> **Persistent Kernel 本身不会降低利用率，但设计不当会。极致利用率的公式是：**
> 
> **Stream-K/CLC 消灭尾效 → Warp Specialization 填满 Warp Slot → Multistage Pipeline 消灭 LSU/TC 空泡 → TMA Multicast+Async Store 重叠 Epilogue → 微架构优化榨干最后 5%**

每一步的收益递减，但前面的步骤是后面步骤的基础。如果 pipeline 没做好，persistent 模式下的 epilogue-mainloop 重叠就是空谈。
