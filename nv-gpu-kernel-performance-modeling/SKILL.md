---
name: nv-gpu-kernel-performance-modeling
description: >
  NVIDIA GPU kernel performance modeling skill. Use for analytical performance
  modeling, latency prediction, roofline-style reasoning, kernel decomposition,
  stage-centric pipeline analysis, producer-consumer pipelines, warp-specialized
  pipelines, multistage pipelines, instruction pipeline demand modeling, tile size,
  stage count, register allocation, occupancy/latency tradeoffs, and cross-architecture
  prediction for A100/H100/B200, Ampere sm_80, Hopper sm_90, Blackwell sm_100.
  中文触发词：GPU 性能建模、kernel 延迟预测、理论性能、流水线建模、访存/计算重叠、
  跨架构迁移、A100/H100/B200 性能估计、tile/stage/register 参数选择。
---

# NVIDIA GPU 算子性能建模 Skill

## 目录

1. [核心概念与术语](#1-核心概念与术语)
2. [方法论框架：五步流程](#2-方法论框架五步流程)
3. [Step 1: Kernel 分析与分解](#3-step-1-kernel-分析与分解)
4. [Step 2: 硬件映射与流水线识别](#4-step-2-硬件映射与流水线识别)
5. [Step 3: 流水线排布设计](#5-step-3-流水线排布设计)
6. [Step 4: 延迟计算与掩盖分析](#6-step-4-延迟计算与掩盖分析)
7. [Step 5: 跨架构适配](#7-step-5-跨架构适配)
8. [数学公式集](#8-数学公式集)
9. [架构参数参考表](#9-架构参数参考表)
10. [工具链](#10-工具链)
11. [实践案例](#11-实践案例)

---

## 1. 核心概念与术语

### 1.1 GPU 执行模型

| 术语 | 定义 |
|------|------|
| **SM** (Streaming Multiprocessor) | GPU的核心计算单元，包含warp调度器、执行单元、寄存器文件、共享内存 |
| **Warp** | 32个线程组成的调度单位，同一warp内的线程执行相同指令(SIMT) |
| **Warp Group** | 128个线程(4个warp)，Hopper/Blackwell中Tensor Core的调度单位 |
| **CTA** (Cooperative Thread Array) | 协作线程阵列，即Thread Block，可共享SMEM和同步 |
| **Kernel** | 在GPU上执行的函数，启动时生成CTA网格 |
| **Occupancy** | 每个SM上active warp数量与最大warp数量的比值 |

### 1.2 指令流水线类型

| 流水线 | 功能 | 关键指令 | 相关架构 |
|--------|------|----------|----------|
| **Tensor Core** | 矩阵乘法加速 | `mma.sync`, `wgmma.mma_async`, `tcgen05.mma` | All |
| **FMA** | 浮点乘加 | `fma`, `fmul`, `fadd` | All |
| **XU** | 特殊函数(指数、对数等) | `ex2`, `lg2`, `sin`, `cos` | All |
| **ALU** | 整数/位运算 | `mad`, `shl`, `and`, `csel` | All |
| **LSU** (Load-Store) | 全局/共享内存访问 | `ld`, `st`, `ldg`, `sts`, `cp.async` | All |
| **TMA** | 张量内存异步拷贝 | `cp.async.bulk.tensor` | Hopper+ |

### 1.3 流水线排布类型

| 类型 | 描述 | 适用架构 | 代表实现 |
|------|------|----------|----------|
| **Multistage** | 所有warp同时担任producer和consumer，通过async指令重叠 | Ampere+ | FlashAttention-2, Ampere GEMM |
| **Warp-Specialized** | Producer warps(访存)和Consumer warps(计算)分离 | Hopper+ | CUTLASS Hopper GEMM, FA3 |
| **Persistent** | CTA驻留SM上，反复处理多个tile | Hopper+ | Ping-Pong GEMM, FA3 |

### 1.4 内存层次

```
Register File (per SM)
    ↕ (1 TB/s+)
Shared Memory / L1 Cache (per SM)
    ↕ (~10 TB/s aggregated L2 BW)
L2 Cache (unified across all SMs)
    ↕ (HBM BW: 2-8 TB/s)
HBM (Global Memory)
```

### 1.5 关键性能指标

| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **Arithmetic Intensity (AI)** | 每字节访存对应的计算操作数 | FLOPs / Bytes (HBM) |
| **Theoretical Cycles** | 某流水线处理其workload所需的最少周期 | Demand / Throughput_per_cycle |
| **Execution Efficiency** | 实测延迟与理论延迟的比值 | Theoretical_Cycles / Measured_Cycles |
| **Wave Quantization** | 由于任务数不是SM数的整数倍导致的效率损失 | ceil(Num_Tasks / Num_SMs) |
| **Tile Quantization** | 由于矩阵维度不是tile size整数倍导致的额外工作 | ceil(Dim / TileSize) * TileSize |

---

## 2. 方法论框架：五步流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU Kernel 性能建模流程                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Kernel 分析                                                │
│    ├── 读取 kernel 源码/PTX/SASS                                    │
│    ├── 识别计算模式 (GEMM/Attention/Elementwise/Reduction)          │
│    ├── 提取 tile 划分策略和循环结构                                  │
│    └── 确定任务定义 (CTA vs Persistent work packet)                 │
│                              ↓                                      │
│  Step 2: 硬件映射                                                   │
│    ├── 识别使用的指令流水线 (Tensor/FMA/XU/LSU/TMA)                 │
│    ├── 确定内存访问模式 (GMEM→SMEM→Register)                       │
│    └── 分析数据依赖和同步点                                         │
│                              ↓                                      │
│  Step 3: 流水线排布设计                                             │
│    ├── 选择流水线类型 (Multistage/Warp-Specialized)                 │
│    ├── 确定 stage count (pipeline深度)                              │
│    ├── 设计 tile size (影响arithmetic intensity)                    │
│    └── 规划 register 分配 (producer vs consumer)                    │
│                              ↓                                      │
│  Step 4: 延迟计算与掩盖分析                                         │
│    ├── 计算每个流水线的 Demand 和 Theoretical Cycles                │
│    ├── 分析流水线间掩盖 (TMA∥WGMMA, cp.async∥mma)                  │
│    ├── 计算 wave/tile quantization 开销                              │
│    └── 预测总体延迟 (分析模型 + MLP校准)                            │
│                              ↓                                      │
│  Step 5: 跨架构适配                                                 │
│    ├── 更新架构参数 (SM数、带宽、吞吐、延迟)                        │
│    ├── 调整流水线设计 (TMA→TMEM, wgmma→tcgen05)                    │
│    └── 重新计算延迟预测                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step 1: Kernel 分析与分解

### 3.1 识别 Kernel 类型

| Kernel 类型 | 特征 | 典型实现 |
|-------------|------|----------|
| **GEMM** | 三重循环 over M,N,K | cuBLAS, CUTLASS, triton.matmul |
| **Attention** | Q@K^T → softmax → @V | FlashAttention-2/3/4, FlashInfer |
| **Elementwise** | 逐元素操作，无reduction | add, mul, activation, layernorm |
| **Reduction** | 跨线程聚合 | sum, max, softmax normalization |
| **Fused** | 多种操作合并 | Fused MoE, Fused Attention+FFN |

### 3.2 提取计算流程

**方法**: 通过源代码或PTX/SASS分析提取 kernel 的计算流程图。

**对于开源 kernel**:
```python
# 示例：从FlashAttention-2提取计算流程
def analyze_flashattention2(seq_len, head_dim, tile_q=128, tile_kv=128):
    """
    识别FlashAttention-2的计算流程：
    1. 外层循环: 遍历Q的tile (seq_len / tile_q iterations)
    2. 内层循环: 遍历KV的tile (seq_len / tile_kv iterations)
    3. 每个iter: 
       - GMEM→SMEM: 加载Q_tile, K_tile, V_tile (TMA/cp.async)
       - 计算: S = Q_tile @ K_tile^T (WGMMA/mma)
       - 计算: online softmax (XU/ALU)
       - 计算: O += P @ V_tile (WGMMA/mma)
       - 同步: mbarrier等待
    """
    num_q_tiles = (seq_len + tile_q - 1) // tile_q
    num_kv_tiles = (seq_len + tile_kv - 1) // tile_kv
    
    operations = []
    for q_tile in range(num_q_tiles):
        for kv_tile in range(num_kv_tiles):
            operations.extend([
                ("TMA_LOAD", "Q_tile", tile_q * head_dim * 2),    # BF16=2B
                ("TMA_LOAD", "K_tile", tile_kv * head_dim * 2),
                ("TMA_LOAD", "V_tile", tile_kv * head_dim * 2),
                ("MMA", "Q@K^T", tile_q * tile_kv * head_dim * 2),  # MAC=2 FLOPs
                ("XU", "online_softmax", tile_q * tile_kv * 8),       # exp等
                ("MMA", "P@V", tile_q * head_dim * tile_kv * 2),
                ("SYNC", "mbarrier", 0),
            ])
    return operations
```

**对于闭源 kernel (如cuBLAS)**:
- 使用 PyTorch Profiler 捕获 kernel launch 参数
- 通过 kernel name 推断 tile size (如 `cutlass_tensorop_s16816gemm_256x128_...` 中 256x128 是 tile)
- 通过 profiling 数据反向推断任务划分策略

### 3.3 任务定义与分解

**映射函数 F**: 将 kernel 输入参数 X 和硬件规格 S 映射为任务集合 T。

```
T = F(X, S) = {τ₁, τ₂, ..., τₜ}

其中每个任务 τᵢ 由维度参数向量 dᵢ 表征。
```

**两种任务范式**:

| 范式 | 任务定义 | 适用 Kernel |
|------|----------|-------------|
| **传统模型** | CTA / Thread Block | FlashAttention-2, RMSNorm, Elementwise |
| **Persistent** | 从全局work queue获取的计算包 | FlashInfer FA3, Ping-Pong GEMM |

### 3.4 计算工作负载量化

对于每个任务，计算各流水线的 **Demand**:

```
D_math = MACs_count            # 乘累加操作数
D_mem  = Bytes_transferred      # 传输字节数 (GMEM↔SMEM)
D_xu   = Special_ops_count     # 特殊函数调用数
D_alu  = Integer_ops_count     # 整数/位操作数
```

---

## 4. Step 2: 硬件映射与流水线识别

### 4.1 指令到流水线映射

通过 SASS 分析确定每条指令映射到哪个硬件流水线:

```
SASS指令 → 流水线映射:
  mma.sync.*, wgmma.mma_async.*, tcgen05.mma.*  → Tensor Core
  fma.rn.*, fmul.*, fadd.*                       → FMA
  ex2.approx.*, lg2.approx.*                      → XU
  mad.lo.*, shl.*, and.*                          → ALU
  ld.global.*, st.global.*, cp.async.*            → LSU / TMA
  cp.async.bulk.tensor.*                          → TMA
```

### 4.2 内存访问路径分析

```
GMEM → [TMA/cp.async/ld.global] → SMEM/L1
SMEM → [ld.shared] → Registers
Registers → [mma/wgmma] → Registers (累加器)
Registers → [st.shared] → SMEM
SMEM → [cp.async.bulk] → GMEM

Blackwell特有:
SMEM → [tcgen05.mma] → TMEM (累加器直接写入TMEM)
GMEM → [TMA] → TMEM (直接到TMEM)
```

### 4.3 数据依赖与同步点

| 同步机制 | 适用场景 | 开销 |
|----------|----------|------|
| `__syncthreads()` | CTA内所有线程同步 | ~10-20 cycles |
| `mbarrier` (Hopper+) | Producer-Consumer异步同步 | 硬件管理，低开销 |
| `fence.proxy.async` | TMA事务完成信号 | 异步，非阻塞 |
| `wgmma.commit_group` | WGMMA提交组同步 | 异步流水线管理 |

---

## 5. Step 3: 流水线排布设计

### 5.1 选择流水线类型

**决策树**:
```
是否有Tensor Core操作?
  ├── 否 → Simple Sequential / ILP优化
  └── 是 → 架构支持TMA?
            ├── 否(Ampere) → Multistage Pipeline
            │                  └── cp.async重叠加载与mma
            └── 是(Hopper+) → 寄存器压力大?
                              ├── 是 → Warp-Specialized
                              │          ├── Producer: TMA load (少寄存器)
                              │          └── Consumer: MMA compute (多寄存器)
                              └── 否 → 可考虑Multistage或Warp-Specialized
```

### 5.2 Stage Count 设计

**Pipeline depth** 决定可以隐藏多少延迟:

```
所需最小stage数 ≥ Memory_latency / Compute_time_per_tile

典型值:
- Ampere GEMM: 2-4 stages
- Hopper GEMM: 2-6 stages (warp-specialized)
- Hopper Attention: 1-2 stages (FA2 style)
- Blackwell GEMM: 2 stages + TMEM double buffering
```

**Stage count 与 shared memory 的权衡**:
```
Total_SMEM = Stage_count × (SMEM_per_operand_A + SMEM_per_operand_B + SMEM_per_output_C)

约束: Total_SMEM ≤ SMEM_capacity_per_SM × Occupancy_target

例如 H100: 228KB SMEM, target occupancy = 1 CTA/SM
→ 最大Stage_count = 228KB / (128×128×2B × 2 operands) ≈ 3-4
```

### 5.3 Tile Size 设计

Tile size 影响 arithmetic intensity:

```
AI_GEMM = (2 × M_tile × N_tile × K_tile) / (2 × (M_tile × K_tile + N_tile × K_tile) × element_size)
        = (M_tile × N_tile × K_tile) / ((M_tile + N_tile) × K_tile × element_size)
        = M_tile × N_tile / ((M_tile + N_tile) × element_size)

对于方形 tile M_tile = N_tile = T:
AI = T / (2 × element_size)

目标: AI > Ridge_point (H100 BF16 ≈ 295, B200 BF16 ≈ 281)
→ T > 2 × element_size × Ridge_point
→ BF16: T > 2 × 2 × 295 ≈ 1180

但受限于 SMEM 和寄存器容量!
```

**典型 Tile Size 配置**:

| 架构 | GEMM Tile | Attention Tile Q | Attention Tile KV | SMEM/CTA |
|------|-----------|------------------|-------------------|----------|
| A100 | 128×256×64 | 128 | 128 | ~64KB |
| H100 | 256×128×64 | 64-128 | 64-128 | ~80KB |
| B200 | 128×256×16 (UMMA) | 64-128 | 64-128 | ~64KB |

### 5.4 Register 分配 (Warp-Specialized)

**Hopper 典型配置**:
```
1 Producer WG + 2 Consumer WGs = 384 threads/SM

Producer WG (128 threads):  24-40 registers/thread × 128 =  3-5K registers
Consumer WG 1 (128 threads): 224-240 registers/thread × 128 = 29-31K registers  
Consumer WG 2 (128 threads): 224-240 registers/thread × 128 = 29-31K registers
总计: ~61-67K registers ≤ 65536/SM ✓

典型值: 40/232/232 或 24/240/240
```

**Blackwell 调整**:
- TMEM 存储累加器，consumer register需求减少
- 可考虑 2 个 Producer WG (双缓冲 TMA) + 1 Consumer WG

---

## 6. Step 4: 延迟计算与掩盖分析

### 6.1 核心公式

#### 6.1.1 单个流水线的理论周期

```
Cycles_pipeline = Demand_pipeline / Throughput_per_cycle_per_SM
```

#### 6.1.2 整个 Kernel 的理论延迟 (基础模型)

```
# Hong-Kim 基础框架
T_kernel = max(T_compute, T_memory) + T_overhead

其中:
T_compute = max(Cycles_math_pipeline, Cycles_fma_pipeline, ...)
T_memory  = Bytes_total / Effective_memory_bandwidth
T_overhead = Launch_latency + Sync_costs + Tile_quantization + Wave_quantization
```

#### 6.1.3 Stage-Centric 模型 (Blackwell/Hopper 精确模型)

```
# 稳态流水线步骤 (每个tile的处理时间)
T_step_pipelined = max(T_tma, T_decompress, T_compute, T_sync) + ε

其中:
T_tma      = bytes(tile) / B_TMA + L_TMA_setup
T_compute  = FLOPs(tile) / Sustained_tensor_TFLOPS
T_sync     = mbarrier_wait + pipeline_bubble

# 总时间
T_total = Num_tiles × T_step_pipelined + T_fill + T_drain
T_fill  = (Stage_count - 1) × T_step_pipelined   # 流水线填充
T_drain = (Stage_count - 1) × T_step_pipelined   # 流水线排空
```

#### 6.1.4 PipeWeave 多维度特征

```
# 为每个关键流水线计算 demand 和 theoretical cycles
Feature_vector = [
    # Math pipeline demands
    D_tensor,       # Tensor Core operations count
    D_fma,          # FMA operations count  
    D_xu,           # XU operations count
    
    # Memory pipeline demands
    D_gmem_read,    # GMEM read bytes
    D_gmem_write,   # GMEM write bytes
    D_smem_access,  # SMEM access bytes
    
    # Theoretical cycles per pipeline
    Cycles_tensor = D_tensor / (Tensor_throughput × Num_SMs_occupied)
    Cycles_fma    = D_fma    / (FMA_throughput × Num_SMs_occupied)
    Cycles_mem    = D_gmem_read / (GMEM_bandwidth / Num_active_SMs)
    
    # GPU architecture parameters
    Num_SMs,
    SM_clock_MHz,
    Tensor_throughput_per_SM,
    GMEM_bandwidth_GBps,
    ...
]

# 最终预测: 轻量MLP(Feature_vector) → Predicted_Latency
# MLP通常为3层: Input(12-d) → Hidden(64) → Hidden(32) → Output(1)
```

### 6.2 掩盖分析

#### 6.2.1 访存-计算重叠

```
# 理想情况: TMA和Tensor Core完全重叠
T_eff = max(T_tma, T_tensor)

# 实际重叠率取决于:
# 1. Pipeline stage count (越深重叠越好)
# 2. Memory/compute balance (不平衡会导致一方等待)
# 3. Async execution效率

重叠效率 η = max(T_tma, T_tensor) / (T_tma + T_tensor)  [理想=0.5，即完全重叠]
```

#### 6.2.2 具体掩盖场景

| 场景 | 掩盖类型 | 效率 | 条件 |
|------|----------|------|------|
| TMA load ∥ WGMMA compute | 访存-计算 | 90-95% | Stage count ≥ 2, SMEM足够 |
| cp.async ∥ mma.sync | 访存-计算 | 70-80% | Ampere, async pipeline |
| WGMMA ∥ epilogue (pingpong) | 计算-计算 | 80-90% | Ping-pong schedule |
| TMA load A ∥ TMA load B | 访存-访存 | 50-60% | Dual TMA path |
| tcgen05.mma ∥ TMEM store | 计算-访存 | 85-95% | Blackwell async |

### 6.3 Quantization 开销

#### 6.3.1 Wave Quantization

```
Num_waves    = ceil(Num_CTA / (Num_SMs × CTA_per_SM))
Last_wave_util = (Num_CTA % (Num_SMs × CTA_per_SM)) / (Num_SMs × CTA_per_SM)
Wave_penalty = 1 - (1 - Last_wave_util) / Num_waves
```

#### 6.3.2 Tile Quantization

```
Padded_M = ceil(M / M_tile) × M_tile
Padded_N = ceil(N / N_tile) × N_tile  
Padded_K = ceil(K / K_tile) × K_tile

Tile_penalty = (Padded_M × Padded_N × Padded_K) / (M × N × K) - 1
```

---

## 7. Step 5: 跨架构适配

### 7.1 架构参数更新流程

```python
ARCH_PARAMS = {
    "Ampere_A100": {
        "SM_count": 108,
        "SMEM_per_SM": 192 * 1024,      # 192 KB
        "L2_cache": 40 * 1024 * 1024,    # 40 MB
        "HBM_bw": 2039,                  # GB/s
        "FP16_TC_peak": 312,             # TFLOPS
        "tensor_instr": "mma.sync",
        "async_copy": "cp.async",
        "has_tma": False,
        "has_wgmma": False,
    },
    "Hopper_H100": {
        "SM_count": 132,
        "SMEM_per_SM": 256 * 1024,      # 256 KB (max 228KB config)
        "L2_cache": 50 * 1024 * 1024,    # 50 MB
        "HBM_bw": 3350,                  # GB/s
        "FP16_TC_peak": 989,             # TFLOPS
        "tensor_instr": "wgmma.mma_async",
        "async_copy": "cp.async.bulk.tensor",
        "has_tma": True,
        "has_wgmma": True,
    },
    "Blackwell_B200": {
        "SM_count": 148,
        "SMEM_per_SM": 128 * 1024,      # 128 KB (GB203 consumer)
        "TMEM_per_SM": 256 * 1024,       # 256 KB TMEM
        "L2_cache": 65 * 1024 * 1024,    # 65 MB per die
        "HBM_bw": 8000,                  # GB/s
        "FP16_TC_peak": 2250,            # TFLOPS
        "tensor_instr": "tcgen05.mma",
        "async_copy": "cp.async.bulk.tensor",
        "has_tma": True,
        "has_tmem": True,
        "accumulator_location": "TMEM",
    },
}
```

### 7.2 流水线设计调整

| 设计要素 | Ampere | Hopper | Blackwell |
|----------|--------|--------|-----------|
| Tensor Core指令 | `mma.sync` | `wgmma.mma_async` | `tcgen05.mma` |
| 执行粒度 | Warp (32T) | Warp-group (128T) | Warp (32T), 单线程发起 |
| 累加器位置 | Registers | Registers | TMEM |
| 同步模型 | 同步 | 异步(fence/commit/wait) | 完全异步 |
| 延迟特性 | 依赖tile大小 | 线性随tile宽度增长 | 恒定~11周期 |
| 流水线类型 | 时间流水线 | 时间流水线 | 空间阵列 |
| 最佳pipeline | Multistage | Warp-specialized | Warp-specialized + TMEM DB |
| Stage count | 2-4 | 2-6 | 2 + TMEM double buffer |

### 7.3 Blackwell 特殊考量

#### 7.3.1 TMEM (Tensor Memory)

```
TMEM特性:
- 容量: 256KB per SM
- 读带宽: 16 TB/s per SM (与SMEM带宽叠加!)
- 写带宽: 8 TB/s per SM
- 延迟: ~420 cycles (cache miss场景, 比GMEM快58%)
- 用途: 存储MMA累加器 (替代register file)

建模影响:
- 累加器不再占用register file → consumer register需求减少
- TMEM读写是独立的带宽通道 → 可以与SMEM访问并行
- 需要额外的TMEM管理开销
```

#### 7.3.2 非对称硬件扩展

```
Blackwell问题: Tensor Core FP16/BF16吞吐翻倍(2.25 PFLOPS vs 1 PFLOPS)，
但 SMEM带宽(128B/cycle/SM) 和指数单元吞吐(16 ops/cycle/SM) 不变。

结果: 在注意力工作负载中，SMEM流量和指数操作可能超过MMA计算25-60%。

建模修正:
T_step = max(T_tensor, T_smem, T_xu, T_tma)  # 不仅仅是compute vs memory!
```

#### 7.3.3 统一INT32/FP32

```
变化: 独立流水线(128 FP32 + 64 INT32) → 统一单元(64-lane，每周期INT32或FP32)

影响:
- 纯FP32或纯INT32: 吞吐减半(128→64 ops/cycle)
- 混合1:1: 延迟从32周期→16周期 (大幅改善)
- 建模需要知道指令mix ratio
```

---

## 8. 数学公式集

### 8.1 基础延迟公式

```
# 1. 计算绑定kernel
T_compute = FLOPs / (Tensor_peak_TFLOPS × 10^12 × Utilization)

# 2. 内存绑定kernel  
T_memory = Bytes / (HBM_bw_GBps × 10^9 × Utilization)

# 3. 混合kernel (简单max)
T_kernel_simple = max(T_compute, T_memory)

# 4. 精确kernel (stage-centric)
T_kernel_precise = Σ_i max(T_stage_i_compute, T_stage_i_memory) + T_overhead

# 5. 考虑occupancy
Effective_SM_usage = min(Num_CTA, Num_SMs × Max_CTA_per_SM)
T_with_occupancy = T_kernel × (Num_CTA / Effective_SM_usage) [近似]
```

### 8.2 流水线 Demand 计算

```
# GEMM Demand
D_tensor = 2 × M_tile × N_tile × K_tile  # MACs (每CTA每step)
D_mem_read  = (M_tile × K_tile + N_tile × K_tile) × element_size  # A,B
D_mem_write = M_tile × N_tile × element_size  # C

# Attention Demand (FlashAttention style)
D_tensor_qk = 2 × Q_tile × KV_tile × head_dim    # Q@K^T
D_tensor_pv = 2 × Q_tile × head_dim × KV_tile    # P@V
D_xu_softmax = Q_tile × KV_tile × 3               # exp, sum, div
D_mem_q = Q_tile × head_dim × element_size
D_mem_kv = 2 × KV_tile × head_dim × element_size  # K+V
```

### 8.3 Theoretical Cycles 计算

```
# Per-SM theoretical cycles for each pipeline
Cycles_tensor_SM = D_tensor / Tensor_ops_per_cycle_per_SM
Cycles_fma_SM    = D_fma    / FMA_ops_per_cycle_per_SM
Cycles_xu_SM     = D_xu     / XU_ops_per_cycle_per_SM
Cycles_mem_SM    = D_mem    / (GMEM_bw_per_SM / element_size)

# Per-CTA cycles (accounting for parallel execution)
Cycles_CTA = max(Cycles_tensor_SM, Cycles_mem_SM)  # with perfect overlap

# Total kernel cycles
Total_cycles = Num_CTA_wave0 × Cycles_CTA 
             + Num_waves × Cycles_CTA × (1 - overlap_efficiency)
             + Pipeline_fill_drain_cycles
```

### 8.4 MLP 校准模型 (PipeWeave style)

```python
# 轻量MLP用于捕获分析模型无法表达的非线性交互
import torch.nn as nn

class PerformanceEstimator(nn.Module):
    def __init__(self, input_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predicted latency (log scale)
        )
    
    def forward(self, features):
        return self.net(features)

# 输入特征 = [D_tensor, D_fma, D_xu, D_mem, 
#             Cycles_tensor, Cycles_fma, Cycles_mem,
#             Num_SMs, SM_clock, Tensor_peak, GMEM_bw, L2_size]
```

---

## 9. 架构参数参考表

### 9.1 Ampere (SM80)

| 参数 | A100 SXM | A100 PCIe | RTX 3090 |
|------|----------|-----------|----------|
| SMs | 108 | 108 | 82 |
| FP32 CUDA Cores/SM | 64 | 64 | 64 |
| Tensor Core Gen | 3rd | 3rd | 3rd |
| FP16/BF16 TC Peak | 312 TF | 312 TF | 142 TF |
| TF32 TC Peak | 156 TF | 156 TF | 71 TF |
| SMEM/SM | 192 KB | 192 KB | 164 KB |
| L2 Cache | 40 MB | 40 MB | 6 MB |
| HBM BW | 2039 GB/s | 1935 GB/s | 936 GB/s |
| TMA | No | No | No |

### 9.2 Hopper (SM90)

| 参数 | H100 SXM | H100 PCIe | H200 SXM |
|------|----------|-----------|----------|
| SMs | 132 | 132 | 132 |
| FP32 CUDA Cores/SM | 128 | 128 | 128 |
| Tensor Core Gen | 4th | 4th | 4th |
| FP16/BF16 TC Peak | 989 TF | 989 TF | 989 TF |
| FP8 TC Peak | 1979 TF | 1979 TF | 1979 TF |
| SMEM/SM | 256 KB | 256 KB | 256 KB |
| L2 Cache | 50 MB | 50 MB | 50 MB |
| HBM BW | 3350 GB/s | 2000 GB/s | 4800 GB/s |
| TMA | Yes | Yes | Yes |

### 9.3 Blackwell (SM100)

| 参数 | B200 | GB200 | GB203 (RTX 5080) |
|------|------|-------|------------------|
| SMs | 148 | 2×168 | 84 |
| FP32 CUDA Cores/SM | 128 (unified) | 128 (unified) | 128 (unified) |
| Tensor Core Gen | 5th | 5th | 5th |
| FP16/BF16 TC Peak | ~2250 TF | ~4500 TF | ~480 TF |
| FP8 TC Peak | ~4500 TF | ~9000 TF | ~960 TF |
| FP4 TC Peak | ~9000 TF | ~18000 TF | ~1920 TF |
| SMEM/SM | 256 KB | 256 KB | 128 KB |
| TMEM/SM | 256 KB | 256 KB | ? |
| L2 Cache | 126 MB | 126 MB | 65 MB |
| HBM BW | 8000 GB/s | 8000 GB/s | GDDR7 |
| TMA | Yes | Yes | Yes |
| TMEM | Yes | Yes | Yes |

### 9.4 指令延迟参考 (cycles)

| 指令 | Ampere | Hopper | Blackwell |
|------|--------|--------|-----------|
| FP32 FMA | 4 | 4 | 4 |
| FP64 FMA | 2 (per 2 cycles) | 2 (per 2 cycles) | 8 |
| INT32 | 4 | 4 | 4 |
| Mixed INT32/FP32 | N/A (independent) | 32-44 | 16-26 |
| `mma.sync` (tile) | 32-256 | N/A | N/A |
| `wgmma.mma_async` (m64n8k16) | N/A | 16-64 (linear) | N/A |
| `tcgen05.mma` (m64n8k16) | N/A | N/A | ~11 (constant) |
| `ex2.approx` | 12 | 12 | 12 |
| GMEM load (hit L2) | 100-200 | 100-200 | 100-200 |
| GMEM load (miss L2) | 400-800 | 400-800 | ~420 (TMEM) |
| SMEM load | 19-26 | 19-26 | 19-26 |

---

## 10. 工具链

### 10.1 性能分析工具

| 工具 | 用途 | 关键功能 |
|------|------|----------|
| **Nsight Compute** | Kernel级分析 | SOL, 流水线利用率, Occupancy, 内存分析 |
| **Nsight Systems** | 系统级分析 | 时间线, API trace, 多stream分析 |
| **nvdisasm** | SASS反汇编 | 指令分析, 控制码解析, 流水线映射 |
| **cuobjdump** | 提取SASS | `-sass` 选项提取编译后指令 |
| **PyTorch Profiler** | Kernel trace | Kernel launch参数, 执行时间 |
| **Triton Proton** | Triton kernel分析 | 与Nsight Compute集成 |

### 10.2 开源性能建模工具

| 工具 | 类型 | 精度 | 架构支持 |
|------|------|------|----------|
| **PipeWeave** | 分析+ML混合 | 6.1% MAE | Ampere/Hopper/Blackwell |
| **GPGPU-Sim/Accel-Sim** | 周期精确模拟 | 5-10%误差 | 可配置 |
| **LLMCompass** | 分析+模拟混合 | 4.1%误差 | LLM专用 |
| **Calculon** | 分析模型 | ~1ms/设计点 | 分布式训练 |
| **TVM/AutoTVM** | 学习成本模型 | 相对排名 | 多后端 |

### 10.3 微基准测试工具

| 工具 | 用途 |
|------|------|
| **nvbench** | NVIDIA官方微基准框架 |
| **gpu-microbench** | 社区延迟/吞吐测量工具 |
| **bandwidthTest** | CUDA SDK内存带宽测试 |
| **deviceQuery** | CUDA设备信息查询 |

### 10.4 关键NVIDIA文档

- CUTLASS 3.x Documentation: https://docs.nvidia.com/cutlass/
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PTX ISA Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
- Nsight Compute Docs: https://docs.nvidia.com/nsight-compute/

---

## 11. 实践案例

### 11.1 案例: GEMM 性能建模 (H100)

```python
# 问题: 预测 8192×8192×8192 BF16 GEMM 在H100上的延迟

def model_gemm_h100(M, N, K, dtype_bytes=2):
    # H100参数
    NUM_SM = 132
    SM_CLOCK_GHZ = 1.98  # 基础频率
    FP16_PEAK_TFLOPS = 989
    HBM_BW_GBPS = 3350
    SMEM_PER_SM = 228 * 1024  # 可配置为SMEM的最大值
    
    # Tile配置 (CUTLASS典型值)
    TILE_M = 256
    TILE_N = 128
    TILE_K = 64
    
    # CTA count
    NUM_CTA_M = (M + TILE_M - 1) // TILE_M
    NUM_CTA_N = (N + TILE_N - 1) // TILE_N
    NUM_CTA_K = (K + TILE_K - 1) // TILE_K  # K方向循环
    TOTAL_CTA = NUM_CTA_M * NUM_CTA_N
    
    # 每个CTA的工作负载
    macs_per_cta = 2 * TILE_M * TILE_N * TILE_K  # 每次K tile
    macs_total = macs_per_cta * NUM_CTA_K * TOTAL_CTA
    
    # 每个CTA的访存量
    gmem_read_per_k = (TILE_M * TILE_K + TILE_N * TILE_K) * dtype_bytes
    gmem_write = TILE_M * TILE_N * dtype_bytes  # 只写一次
    
    # 计算cycles
    tensor_ops_per_cycle_per_sm = 512 * 2  # 512 per cycle per SM × 2 (BF16 dual-issue)
    cycles_compute_per_cta_per_k = macs_per_cta / (tensor_ops_per_cycle_per_sm * NUM_SM)
    
    # 内存cycles (假设HBM带宽均摊)
    bytes_per_cycle = (HBM_BW_GBPS * 1e9) / (SM_CLOCK_GHZ * 1e9)
    cycles_mem_per_k_per_cta = gmem_read_per_k / (bytes_per_cycle * NUM_SM)
    
    # Wave分析
    cta_per_wave = NUM_SM  # 假设1 CTA/SM
    num_waves = (TOTAL_CTA + cta_per_wave - 1) // cta_per_wave
    
    # 每wave时间 (理想重叠)
    cycles_per_wave = NUM_CTA_K * max(cycles_compute_per_cta_per_k, cycles_mem_per_k_per_cta)
    
    # 总cycles
    total_cycles = num_waves * cycles_per_wave
    
    # 转换为时间
    total_ms = total_cycles / (SM_CLOCK_GHZ * 1e6)
    
    # 考虑量化惩罚
    tile_penalty = (NUM_CTA_M * TILE_M * NUM_CTA_N * TILE_N) / (M * N) - 1
    wave_penalty = 1 - (TOTAL_CTA % NUM_SM) / NUM_SM if (TOTAL_CTA % NUM_SM) != 0 else 0
    
    estimated_ms = total_ms * (1 + tile_penalty * 0.1 + wave_penalty * 0.15)
    
    return {
        "estimated_ms": estimated_ms,
        "achieved_tflops": (2 * M * N * K / 1e12) / (estimated_ms / 1000),
        "utilization": ((2 * M * N * K / 1e12) / (estimated_ms / 1000)) / FP16_PEAK_TFLOPS * 100,
        "bottleneck": "compute" if cycles_compute_per_cta_per_k > cycles_mem_per_k_per_cta else "memory",
    }

# 运行
result = model_gemm_h100(8192, 8192, 8192)
# 预期: ~8-10ms (实测cuBLAS约6-8ms)
```

### 11.2 案例: FlashAttention-2 流水线分析

```python
def analyze_fa2_pipeline(seq_len=8192, head_dim=128, headdim_v=128):
    """
    FlashAttention-2流水线分析:
    
    计算流程 (per CTA, per KV tile):
    1. TMA_LOAD Q_tile (GMEM→SMEM)         → TMA pipeline
    2. TMA_LOAD K_tile (GMEM→SMEM)         → TMA pipeline  
    3. WGMMA: S = Q@K^T                    → Tensor pipeline
    4. online_softmax(S)                   → XU + ALU pipeline
    5. TMA_LOAD V_tile (GMEM→SMEM)         → TMA pipeline
    6. WGMMA: O += P@V                     → Tensor pipeline
    7. mbarrier sync                       → Sync overhead
    
    流水线排布: Warp-Specialized (Hopper)
    - Producer warps: 步骤1,2,5 (TMA load)
    - Consumer warps: 步骤3,4,6 (WGMMA + softmax)
    - Producer和Consumer通过mbarrier异步同步
    
    掩盖情况:
    - TMA load Q/K for step i+1 ∥ WGMMA Q@K^T for step i
    - TMA load V for step i+1 ∥ WGMMA P@V for step i  
    - 理想重叠率: 90-95%
    """
    # ... 详细分析代码
    pass
```

### 11.3 案例: Blackwell 适配检查清单

```python
BLACKWELL_MIGRATION_CHECKLIST = {
    "Tensor Core": {
        "更新指令": "wgmma.mma_async → tcgen05.mma",
        "调整粒度": "Warp-group(128T) → Warp(32T), 单线程发起",
        "累加器位置": "Registers → TMEM",
        "延迟模型": "线性增长 → 恒定~11周期",
    },
    "Memory": {
        "新增TMEM": "累加器存TMEM(256KB/SM, 16TB/s读)",
        "SMEM减少": "256KB → 128KB (consumer级)",
        "调整stage count": "可能减少(寄存器需求降低)",
    },
    "Compute": {
        "统一ALU": "独立INT32+FP32 → 统一(混合延迟减半)",
        "非对称瓶颈": "Tensor翻倍但SMEM/XU不变 → 新瓶颈",
    },
    "Pipeline": {
        "更新pipeline类": "PipelineTmaUmma for Blackwell",
        "TMEM double buffering": "新增 accumulator pipeline",
        "2-CTA MMA": "考虑leader/follower CTA同步",
    },
    "Optimization": {
        "精度选择": "新增FP4/FP6选项",
        "tile size调整": "UMMA最大128×256×16",
        "register重新分配": "consumer需求减少",
    },
}
```

---

## 12. 快速参考卡片

### 12.1 建模流程速查

```
1. 识别kernel类型 → 确定task定义(CTA/persistent)
2. 计算各pipeline demand (Tensor/FMA/XU/Memory)
3. 计算theoretical cycles per pipeline
4. 选择流水线类型 (multistage/warp-specialized)
5. 确定stage count和tile size
6. 计算重叠后的effective cycles
7. 添加quantization和overhead
8. (可选) MLP校准
9. 跨架构: 更新参数表, 调整pipeline设计
```

### 12.2 常见瓶颈诊断

| 现象 | 可能原因 | 解决方法 |
|------|----------|----------|
| 利用率 < 50% | Wave quantization, tile太小 | 增大tile, 调整grid size |
| Memory bound | AI < Ridge point | 增大tile, 数据复用 |
| Compute bound | AI >> Ridge point | 优化指令mix, 减少XU/ALU |
| Occupancy低 | 寄存器/SMEM过多 | 减少stage count, 调整register分配 |
| Sync开销大 | 过多__syncthreads | 使用async pipeline, warp-specialized |
| TMA等待 | Producer不能领先consumer | 增加producer warp比例 |

---

*本Skill基于PipeWeave、CUTLASS、Nsight Compute等工具和多篇学术论文(2024-2026)的研究成果综合整理。持续更新以适配新架构。*
