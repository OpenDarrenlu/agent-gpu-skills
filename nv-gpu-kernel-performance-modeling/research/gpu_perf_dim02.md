# 经典GPU分析性能模型：深入研究报告

## 维度概述

在机器学习(ML)方法流行之前，GPU性能建模领域已经建立了丰富的经典分析模型体系。这些模型奠定了GPU性能分析的理论基础，为理解GPU执行机制、识别性能瓶颈以及指导优化策略提供了关键工具。本报告深入分析7个核心经典模型/方法论，涵盖从2008年到2025年的关键技术演进。

**研究范围**：
1. Hong & Kim (2009) MWP-CWP模型
2. Sim et al. (2012) GPUPerf改进框架
3. Kothapalli et al. (2009) PRAM-based模型
4. BSP-based分析模型
5. 微基准测试方法论
6. Little's Law在GPU延迟隐藏中的应用
7. AMALI (2025) 基于指令trace的分析模型

---

## 核心发现

### 发现1：Hong & Kim MWP-CWP模型——最广泛引用的GPU分析性能模型基础

```
Claim: Hong & Kim (2009)提出了GPU架构上最经典的分析性能模型，通过Memory Warp Parallelism (MWP)和Computation Warp Parallelism (CWP)两个核心指标来估计GPU kernel执行时间，在微基准测试上达到5.4%的平均误差，在GPU计算应用上达到13.3%的几何平均误差 [^54^] [^56^]
Source: ISCA 2009 / ACM SIGARCH
URL: https://dl.acm.org/doi/10.1145/1555754.1555775
Date: 2009-06
Excerpt: "The key component of our model is estimating the number of parallel memory requests (we call this the memory warp parallelism) by considering the number of running threads and memory bandwidth. Based on the degree of memory warp parallelism, the model estimates the cost of memory requests, thereby estimating the overall execution time of a program."
Context: 该模型针对NVIDIA Tesla架构，基于PTX汇编指令分析
Confidence: high
```

**模型核心概念**：
- **MWP (Memory Warp Parallelism)**：在一个SM上能够同时访问内存的最大warp数量 [^185^]
- **CWP (Computation Warp Parallelism)**：在一个warp等待内存数据返回期间，SM能够完成执行的其他warp数量加1 [^185^]

**执行时间估算逻辑** [^186^] [^174^]：
- 若 **MWP > CWP**：程序为计算密集型，计算可以隐藏内存延迟
- 若 **CWP > MWP**：程序为内存密集型，内存访问成为瓶颈
- 若 **MWP = CWP = N**：warp数量不足，程序受限于并行度

### 发现2：Sim et al. GPUPerf——对MWP-CWP模型的系统性改进

```
Claim: Sim et al. (2012)提出了GPUPerf框架，在Hong & Kim模型的基础上加入了算术指令延迟、ILP/TLP/MLP的显式建模、缓存效应和SFU指令处理，使模型能够更精确地预测各种优化技术的效果 [^65^] [^184^]
Source: PPoPP 2012
URL: https://jaewoong.org/pubs/sim_ppopp12.pdf
Date: 2012-02
Excerpt: "We model serialization overheads... The base time accounts for the number of operations and degree of parallelism... ITILP models the possibility of inter-thread instruction-level parallelism in GPGPUs."
Context: 针对Fermi架构的改进，增加了binary-level分析
Confidence: high
```

**相对于MWP-CWP的改进** [^65^]：
1. **缓存效应建模**：通过AMAT (Average Memory Access Time)计算缓存影响
2. **SFU指令建模**：特殊功能单元(超越函数、平方根)的独立执行管道
3. **ILP/TLP显式建模**：引入ITILP (Inter-Thread ILP)概念
4. **MLP显式建模**：通过TTMLP (Total Thread MLP)量化内存级并行
5. **Binary-level分析**：不仅使用PTX，还利用二进制信息和硬件性能计数器

**关键公式** [^65^]：
```
T_mem = (#mem_insts × #total_warps / #active_SMs) / (TTMLP) × AMAT
TTMLP = min(MLP × MWP_Q, MWP_parallel_limit)
MWP_Q = min(max(1, CWP-1), MWP)
AMAT = avg_DRAM_lat × miss_ratio + int_lat
avg_DRAM_lat = DRAM_lat + (avg_trans_warp - 1) × Δ
T_comp = W_parallel + W_serial
W_parallel = (#total_insts / #active_SMs) / ITILP × w_inst_lat
ITILP = min(ILP × N, ITILP_max)
ITILP_max = (ave_inst_lat) / (w_inst_lat × SIMD_width / warp_size)
```

### 发现3：Kothapalli et al.——融合BSP/PRAM/QRQW的GPU性能预测模型

```
Claim: Kothapalli et al. (2009)提出了一个融合三种并行计算模型(BSP、PRAM、QRQW)的GPU性能预测模型，针对CUDA GPGPU平台进行执行时间估算，在矩阵乘法等基准测试上达到约12-31%的误差 [^64^] [^66^]
Source: HiPC 2009
URL: https://ieeexplore.ieee.org/document/5433179
Date: 2009-12
Excerpt: "We present a performance prediction model for the CUDA GPGPU platform... Our model integrates BSP, PRAM, and QRQW models to capture different aspects of GPU execution."
Context: 三种经典并行计算模型的首次融合尝试
Confidence: high
```

**模型组成** [^66^] [^67^]：
- **BSP (Bulk Synchronous Parallel)**：Valiant (1990)提出的粗粒度并行模型，考虑计算和通信阶段
- **PRAM (Parallel Random Access Machine)**：理论并行计算模型，包含EREW/CREW/CRCW变体
- **QRQW (Queue-Read Queue-Write)**：Gibbons et al. (1998)提出的考虑内存竞争队列的模型

**核心计算** [^64^]：
- 计算周期 N_comp 和内存访问周期 N_memory 分别测量
- 采用两种调度策略：(1) 考虑所有指令延迟；(2) 只考虑内存和计算指令延迟的最大值
- 评估结果：矩阵乘法31.25%误差，List Ranking 12.50%误差

### 发现4：BSP-based GPU分析模型——简洁实用的执行时间预测

```
Claim: BSP模型及其变体被广泛应用于GPU执行时间预测。相比MWP-CWP的复杂性，BSP-based模型提供了一个简单直观的替代方案，仅需一个额外调整参数即可达到约5%的平均预测误差（矩阵计算任务） [^10^] [^62^]
Source: Journal of Parallel and Distributed Computing / HiPC 2009
URL: https://www.sciencedirect.com/science/article/abs/pii/S0743731522001903
Date: 2022/2024
Excerpt: "A simple and intuitive BSP-based model is proposed. It relies on the number of arithmetic and memory access operations performed by the GPU, with additional information on cache usage obtained from profiling data."
Context: 作为MWP-CWP的轻量级替代
Confidence: high
```

### 发现5：微基准测试方法论——GPU硬件特性测量的基石

```
Claim: 以Wong et al. (2010) ISPASS论文为代表的微基准测试方法，通过精细设计的CUDA kernel揭示GPU微架构细节，为所有分析模型提供关键硬件参数。Mei & Chu (2017)进一步提出了P-chase方法揭示GPU缓存层次结构特性 [^135^] [^140^]
Source: ISPASS 2010 / IEEE TPDS 2017
URL: https://doi.org/10.1109/ISPASS.2010.5452013 / https://doi.org/10.1109/TPDS.2016.2549523
Date: 2010 / 2017
Excerpt: "Demystifying GPU microarchitecture through microbenchmarking" / "We propose a novel fine-grained microbenchmarking approach... to expose the previously unknown characteristics of their memory hierarchies."
Context: 微基准测试是所有分析模型的基础数据来源
Confidence: high
```

### 发现6：Little's Law在GPU延迟隐藏中的核心应用

```
Claim: Volkov (2016)在其博士论文中系统性地应用Little's Law解释GPU延迟隐藏机制，提出了核心关系式 RequiredParallelism = Throughput × Latency，这一原理成为理解GPU warp调度、occupancy优化和内存延迟隐藏的理论基础 [^58^] [^15^] [^146^]
Source: UC Berkeley PhD Thesis / GPU tutorials
URL: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html
Date: 2016-08
Excerpt: "Volkov (2016) compares multiple modeling methods on multiple GPUs... and proposes a simple model for GPU performance based on Little's law, which is also used in operational analysis" [^58^]; "RequiredParallelism = Throughput × Latency" [^15^]
Context: 核心理论基础，被后续所有模型引用
Confidence: high
```

**核心原理** [^15^] [^146^]：
```
RequiredParallelism = Throughput × Latency

在GPU语境下：
- Required Parallelism = 需要活跃的warp数量
- Throughput = SM的指令发射速率
- Latency = 全局内存访问延迟

示例：如果SM每周期可完成1条指令，内存操作需要200周期
则需要至少200条独立指令来完全隐藏延迟
由于单个warp只能提供有限的ILP，GPU依赖大量warp提供MLP
```

### 发现7：AMALI——面向现代GPU和LLM推理的先进分析模型

```
Claim: AMALI (2025 ISCA)是针对现代GPU和LLM推理应用的最新分析模型，通过指令修饰符和吞吐量建模Tensor Core、引入constant cache和instruction cache分析、设计多warp模型反映LLM推理特性，将MAPE从127.56% (GCoM)降低到23.59% [^60^] [^61^]
Source: ISCA 2025
URL: https://dl.acm.org/doi/10.1145/3695053.3731064
Date: 2025-06
Excerpt: "We develop an instruction modifier and throughput based tensor core model by accurately capturing the math pipe throttle stalls... We propose analytical models for constant cache and instruction cache... We design a multi-warp model by leveraging warp instruction number distribution."
Context: 面向A100 GPU验证，针对LLM推理应用
Confidence: high
```

### 发现8：GCoM (2022)——现代GPU的详细核心分析模型

```
Claim: GCoM (GPU Core Model) 2022提供了一个面向现代GPU的详细分析模型，是AMALI之前最先进的GPU分析模型，但在处理Tensor Core和现代缓存层次时存在明显不足 [^272^] [^171^]
Source: ISCA 2022
URL: https://dl.acm.org/doi/10.1145/3470496.3527384
Date: 2022
Excerpt: "GCoM: a detailed GPU core model for accurate analytical modeling of modern GPUs"
Context: 被AMALI引用为state-of-the-art baseline
Confidence: high
```

### 发现9：PPT-GPU——可扩展的trace-driven性能建模工具

```
Claim: PPT-GPU (2019-2021)是一个混合的、可扩展的GPU性能预测工具，使用预收集的内存和指令trace来准确捕获kernel的动态行为，MAPE < 16%，相关系数 > 0.98，比cycle-accurate模拟器快数个数量级 [^170^] [^171^]
Source: SC 2021
URL: https://dl.acm.org/doi/10.1145/3458817.3476221
Date: 2021
Excerpt: "PPT-GPU achieves scalability through a hybrid high-level modeling approach where some computations are extrapolated and multiple parts of the model are parallelized."
Context: Trace-driven方法的代表作
Confidence: high
```

### 发现10：现代微基准驱动分析模型的最新进展 (2025-2026)

```
Claim: 最新的微基准驱动分析模型（2025-2026）展示了经典方法论在现代GPU上的持续有效性。Blackwell B200模型达到1.31% MAE，AMD MI300A达到~0.09% MAE，而naive roofline baseline超过95%误差。关键发现：Hong-Kim框架 (Texec = max(Tcompute, Tmemory) + Toverhead) 仍是现代模型的基础 [^34^]
Source: arXiv 2026 (Microbenchmark-Driven Analytical Performance Modeling)
URL: https://arxiv.org/html/2605.04178v1
Date: 2026-05
Excerpt: "We adopt the Hong-Kim framework: execution time is the maximum of compute and memory time plus overhead... naive roofline baselines exceed 95% error on the same kernels."
Context: 覆盖Blackwell、Hopper、CDNA3、CDNA2架构
Confidence: high
```

---

## 技术方法论详解

### 1. Hong & Kim (2009) MWP-CWP 模型详细公式

**核心参数定义** [^186^] [^184^]：

| 参数 | 含义 |
|------|------|
| Mem_cycles | 每warp的内存等待周期 |
| Comp_cycles | 每warp的计算周期 |
| N | 每SM运行的warp数量 |
| Freq | SM处理器时钟频率 |
| Mem_insts | 每warp的内存指令数 |
| #Rep | 每个SM需要重复计算的次数 |
| BW_per_warp | 每warp的带宽需求 |
| Mem_peak_bw | DRAM到GPU核心的峰值带宽 |
| transaction_size | DRAM请求的事务大小 |

**MWP计算** [^184^]：
```
MWP = min(MWP_peak_bw, MWP_DRAM_lat, N)
MWP_peak_bw = mem_peak_bandwidth / (BW_per_warp × #active_SMs)
BW_per_warp = freq × transaction_size / avg_DRAM_lat
MWP_DRAM_lat = avg_DRAM_lat / Δ
```

其中Δ是连续内存事务的出发延迟(departure delay)。

**CWP计算** [^184^]：
```
CWP = min(CWP_full, N)
CWP_full = (mem_cycles + comp_cycles) / comp_cycles
comp_cycles = (#insts × avg_inst_lat) / ITILP
mem_cycles = (#mem_insts × AMAT) / MLP
```

**执行时间估算** [^186^]：
```
if (MWP == N) and (CWP == N):
    cycles = (Mem_cycles + Comp_cycles + #Mem_insts × (MWP-1)) × #Rep
elif (CWP >= MWP) or (Comp_cycles > Mem_cycles):
    cycles = (Mem_cycles × N/MWP + Comp_cycles/#Mem_insts × (MWP-1)) × #Rep
else:
    cycles = Mem_L + Comp_cycles × N × #Rep
```

### 2. Sim et al. (2012) GPUPerf 详细公式

**内存时间模型** [^65^]：
```
T_mem = (#mem_insts × #total_warps / #active_SMs) / TTMLP × AMAT
AMAT = avg_DRAM_lat × miss_ratio + int_lat
avg_DRAM_lat = DRAM_lat + (avg_trans_warp - 1) × Δ
```

**TTMLP (Total Thread Memory-Level Parallelism)** [^65^]：
```
TTMLP = min(MLP × MWP_Q, MWP_parallel_limit)
MWP_Q = min(max(1, CWP-1), MWP)
```

**计算时间模型** [^65^]：
```
T_comp = W_parallel + W_serial
W_parallel = (#total_insts / #active_SMs) / ITILP × w_inst_lat
ITILP = min(ILP × N, ITILP_max)
ITILP_max = (ave_inst_lat) / (w_inst_lat × SIMD_width / warp_size)
```

**总执行时间** [^65^]：
```
Total_Time = max(T_comp, T_mem)  (改进后使用max或proposed equation)
```

### 3. Little's Law在GPU延迟隐藏中的数学表达

**基本形式** [^15^] [^146^]：
```
RequiredParallelism = Throughput × Latency

GPU warp调度语境：
waves_required = memory_latency / (compute_cycles_per_warp / warps_per_SM)

Occupancy计算：
Occupancy (%) = Active_Warps_per_SM / Maximum_Warps_per_SM × 100

内存带宽饱和需求：
bytes_in_flight = DRAM_bandwidth × DRAM_latency
                = (360 GB/s) × (800 cycles / 2175 MHz)
                ≈ 132 bytes in flight per cycle
```

**Volkov模型** [^58^]：
- 仅考虑两种极端情况：latency-bound和throughput-bound
- 对于中间状态的应用，模型精度有限
- 通过微基准测试收集实际硬件参数

### 4. Kothapalli et al. BSP+PRAM+QRQW融合模型

**模型组件** [^66^]：
- **BSP分量**：模型划分为supersteps，每个superstep包含本地计算和全局通信
- **PRAM分量**：理想化的并行随机存取，提供理论上限
- **QRQW分量**：考虑内存竞争的队列式访问，更贴近实际硬件

**执行时间估算** [^64^]：
```
T_total = T_compute + T_memory + T_sync
T_compute = f(instruction_count, pipeline_depth, ILP)
T_memory = g(memory_requests, bandwidth, coalescing_efficiency)
```

### 5. 微基准测试方法论

**Wong et al. (2010) 核心方法** [^135^]：
- 使用CUDA kernel精确控制指令序列
- 通过 `clock()` 函数测量指令延迟
- 利用依赖链消除ILP影响
- 系统测量各类指令(算术、内存、控制流)的延迟和吞吐量

**Mei & Chu (2017) P-chase方法** [^140^]：
```
// 精细指针追踪微基准
for (i = 0; i < iterations; i++) {
    start = clock();
    j = array[j];  // 关键内存访问
    // 依赖语句确保访问完成
    sink = j;
    end = clock();
    latency[i] = end - start - overhead;
}
```

关键创新：
1. **两阶段方法**：阶段1确定缓存大小和行大小；阶段2确定关联性和地址映射
2. **依赖注入**：通过数据依赖确保 `clock()` 在内存访问完成后执行
3. **ILP消除**：精心设计避免指令级并行干扰测量

**测量的关键参数** [^34^] [^140^]：
- 各类指令延迟和吞吐量
- L1/L2缓存大小、行大小、关联性、替换策略
- 共享内存bank配置和吞吐量
- 全局内存带宽和延迟
- Tensor Core延迟和吞吐量（现代GPU）

### 6. AMALI模型方法论

**三大创新** [^60^]：

(1) **Tensor Core模型**：
```
T_tensor = f(instruction_modifier, throughput, math_pipe_stall)
```
通过准确捕获math pipe throttle stalls来增强架构建模

(2) **Constant Cache & Instruction Cache模型**：
```
T_cache = f(kernel_launch_latency, cache_hit_rate, access_pattern)
```
通过微基准测量CUDA kernel启动延迟

(3) **Multi-warp模型**：
```
T_total = sum_over_warps(T_warp × instruction_distribution)
```
利用warp指令数分布反映LLM推理特性

---

## 架构适配性分析

### 各模型/方法的GPU架构适用性

| 模型/方法 | Tesla | Fermi | Kepler | Maxwell | Pascal | Volta | Ampere | Hopper | Blackwell | 跨架构适用性 |
|-----------|:-----:|:-----:|:------:|:-------:|:------:|:-----:|:------:|:------:|:---------:|:----------:|
| Hong & Kim (2009) MWP-CWP | ✓ | ○ | ○ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 低 |
| Sim et al. (2012) GPUPerf | ✗ | ✓ | ○ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 低 |
| Kothapalli (2009) BSP+PRAM | ✓ | ✓ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | 中 |
| Volkov (2016) Little's Law | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **高** |
| Mei & Chu (2017) 微基准 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 中 |
| GCoM (2022) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ○ | ✗ | 中 |
| AMALI (2025) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ○ | ✗ | 中 |
| 微基准驱动模型 (2026) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | **高** |

**关键结论**：
1. **Hong-Kim框架** (Texec = max(Tcompute, Tmemory) + Toverhead) 具有最强的架构通用性 [^34^]
2. **微基准测试方法** 是跨架构适配的关键基础——每个模型系数都对应一个可测量的微基准 [^34^]
3. **Little's Law原理** 是所有架构中延迟隐藏的根本理论基础 [^58^]
4. **现代GPU特有的组件** (Tensor Core、TMEM、TMA、异步拷贝) 需要架构特定的模型扩展 [^34^] [^60^]

### 不同架构的关键建模差异

**Blackwell B200特有挑战** [^34^]：
- TMEM (256KB/SM) 作为累加器存储，有可测量带宽
- TMA异步批量拷贝
- 第5代Tensor Core
- 2-SM协作执行模式
- naive roofline误差 >95%

**Hopper H200适配** [^34^]：
- wgmma异步warp group矩阵乘法
- TMA增强
- 相同的Hong-Kim框架，仅参数更新

**CDNA3 MI300A特有挑战** [^34^]：
- Infinity Cache层次结构
- VGPR寄存器压力
- Occupancy驱动的tile选择
- 统一物理内存

---

## 工具与资源

### 开源工具/代码

| 工具 | 描述 | 链接/来源 |
|------|------|----------|
| PPT-GPU | 可扩展的trace-driven性能建模工具 | SC 2021 [^170^] |
| GPGPU-Sim | Cycle-accurate GPU模拟器 | 广泛用于验证 |
| Accel-Sim | 可扩展的验证GPU模拟框架 | ISCA 2020 [^272^] |
| MT4G | GPU计算和内存拓扑自动发现工具 | 2025 [^173^] |
| CuAssembler | 非官方CUDA汇编器，用于精确微基准 | [^251^] |

### 关键论文列表

1. **Hong & Kim (2009)** - "An Analytical Model for a GPU Architecture with Memory-level and Thread-level Parallelism Awareness" - ISCA 2009 [^54^]
2. **Hong & Kim (2010)** - "An Integrated GPU Power and Performance Model" - ISCA 2010 [^56^]
3. **Sim et al. (2012)** - "A Performance Analysis Framework for Identifying Potential Benefits in GPGPU Applications" - PPoPP 2012 [^65^]
4. **Kothapalli et al. (2009)** - "A Performance Prediction Model for the CUDA GPGPU Platform" - HiPC 2009 [^66^]
5. **Wong et al. (2010)** - "Demystifying GPU Microarchitecture through Microbenchmarking" - ISPASS 2010 [^135^]
6. **Mei & Chu (2017)** - "Dissecting GPU Memory Hierarchy through Microbenchmarking" - IEEE TPDS [^140^]
7. **Volkov (2016)** - "Understanding Latency Hiding on GPUs" - UC Berkeley PhD Thesis [^58^]
8. **Zhang & Owens (2011)** - "A Quantitative Performance Analysis Model for GPU Architectures" - HPCA 2011 [^74^]
9. **Baghsorkhi et al. (2010)** - "An Adaptive Performance Modeling Tool for GPU Architectures" - PPoPP 2010 [^175^]
10. **Lee et al. (2022)** - "GCoM: A Detailed GPU Core Model" - ISCA 2022 [^272^]
11. **Cao et al. (2025)** - "AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs" - ISCA 2025 [^60^]
12. **Arafa et al. (2021)** - "PPT-GPU: Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs" - SC 2021 [^170^]
13. **Jia et al. (2012)** - "Stargazer: Automated Regression-based GPU Design Space Exploration" - ISPASS 2012 [^269^]
14. **Wu et al. (2015)** - "GPGPU Performance and Power Estimation Using Machine Learning" - HPCA 2015 [^277^]
15. **Konstantinidis & Cotronis (2017)** - "A Quantitative Roofline Model for GPU Kernel Performance Estimation" - JPDC [^5^]

---

## 关键引用列表

[^54^] S. Hong and H. Kim, "An analytical model for a GPU architecture with memory-level and thread-level parallelism awareness," in Proceedings of the 36th annual international symposium on Computer architecture, 2009, pp. 152-163. https://dl.acm.org/doi/10.1145/1555754.1555775

[^56^] S. Hong and H. Kim, "An integrated GPU power and performance model," in Proceedings of the 37th annual international symposium on Computer architecture, 2010, pp. 280-289.

[^65^] J. Sim, A. Dasgupta, H. Kim, and R. Vuduc, "A performance analysis framework for identifying potential benefits in GPGPU applications," in ACM SIGPLAN symposium on Principles and Practice of Parallel Programming, 2012, pp. 11-22. https://jaewoong.org/pubs/sim_ppopp12.pdf

[^66^] K. Kothapalli et al., "A performance prediction model for the CUDA GPGPU platform," in 2009 International Conference on High Performance Computing (HiPC), 2009, pp. 463-472.

[^135^] H. Wong et al., "Demystifying GPU microarchitecture through microbenchmarking," in 2010 IEEE International Symposium on Performance Analysis of Systems & Software (ISPASS), 2010, pp. 235-246.

[^140^] X. Mei and X. Chu, "Dissecting GPU Memory Hierarchy through Microbenchmarking," IEEE Transactions on Parallel and Distributed Systems, vol. 28, no. 1, pp. 72-86, 2017.

[^58^] V. Volkov, "Understanding Latency Hiding on GPUs," PhD thesis, EECS Department, University of California, Berkeley, 2016. https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html

[^60^] S. Cao et al., "AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs," in Proceedings of ISCA, 2025. https://dl.acm.org/doi/10.1145/3695053.3731064

[^34^] "Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures," arXiv, 2026. https://arxiv.org/html/2605.04178v1

[^170^] Y. Arafa et al., "Hybrid, scalable, trace-driven performance modeling of GPGPUs," in SC21: International Conference for High Performance Computing, Networking, Storage and Analysis, 2021. https://dl.acm.org/doi/10.1145/3458817.3476221

[^272^] J. Lee et al., "GCoM: A Detailed GPU Core Model for Accurate Analytical Modeling of Modern GPUs," in Proceedings of the 49th Annual ISCA, 2022, pp. 424-436.

[^175^] S. S. Baghsorkhi et al., "An adaptive performance modeling tool for GPU architectures," in PPoPP 2010, pp. 105-114.

[^15^] "Latency Hiding Pipelines - GPU Kernel Execution Mechanics," oboe.com, 2026. https://oboe.com/learn/gpu-kernel-execution-mechanics-1bezgf/latency-hiding-pipelines-4

[^146^] "Lab 5: Matrix Multiply - Improved Scheduling," Accelerated Computing Academy, 2025. https://accelerated-computing.academy/fall25/labs/lab5/

[^269^] W. Jia, K. A. Shaw, and M. Martonosi, "Stargazer: Automated regression-based GPU design space exploration," in ISPASS 2012, pp. 2-13.

[^277^] G. Wu et al., "GPGPU performance and power estimation using machine learning," in HPCA 2015, pp. 564-576.

[^173^] "A Tool for Reliable Auto-Discovery of NVIDIA and AMD GPU Compute and Memory Topologies," arXiv, 2025. https://arxiv.org/html/2511.05958v1

[^251^] "Cycle accurate benchmarking of CUDA kernels," etasnadi.com, 2025. https://etasnadi.com/2025/12/nanobenchmarking-cycle-accurate-benchmarking-of-cuda-kernels/

[^10^] "Evaluating execution time predictions on GPU kernels using an analytical model and machine learning techniques," Journal of Parallel and Distributed Computing, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0743731522001903

[^62^] "An Empirical Evaluation of GPGPU Performance Models," 2018. https://link.springer.com/chapter/10.1007/978-3-319-14325-5_15

[^64^] "Multilevel interference-aware scheduling on modern GPUs," Northeastern University. https://repository.library.northeastern.edu/files/neu:m044dz31f/fulltext.pdf

[^67^] "A parallel program model for execution time estimation," 2022. https://en.num-meth.ru/index.php/journal/article/view/1177

[^185^] S. Hong and H. Kim, "Memory-level and Thread-level Parallelism Aware GPU Performance Analytical Model," Georgia Tech Technical Report. https://sites.cc.gatech.edu/fac/hyesoon/hong_report09.pdf

[^186^] "GPU Performance Analytical Model," US Patent 8643656. https://patentimages.storage.googleapis.com/7a/11/28/ae507b1159f350/US8643656.pdf

[^184^] "A Performance Analysis Framework for Identifying Potential Benefits in GPGPU Applications," ResearchGate. https://www.researchgate.net/publication/221643648_A_Performance_Analysis_Framework_for_Identifying_Potential_Benefits_in_GPGPU_Applications

[^74^] Y. Zhang and J. D. Owens, "A quantitative performance analysis model for GPU architectures," in HPCA 2011, pp. 382-393.

[^171^] "GCoM: A Detailed GPU Core Model," ISCA 2022. https://dl.acm.org/doi/abs/10.1145/3470496.3527384

[^61^] "AMALI: An Analytical Model for Accurately Modeling LLM Inference on Modern GPUs," ISCA 2025. https://dl.acm.org/doi/abs/10.1145/3695053.3731064

---

## 待深入区域

1. **Blackwell架构的新型组件建模**：TMEM、TMA、第5代Tensor Core的精确建模方法仍需更深入的研究 [^34^]
2. **多kernel并发执行的分析模型**：现有模型大多假设单kernel执行，并发kernel的干扰建模仍是开放问题 [^34^]
3. **不规则workload（如图算法）的分析建模**：BFS等不规则访问模式的模型误差显著高于规则workload（40%+ vs <1%）[^34^]
4. **第一性原理workload特征化**：从源代码推导FLOPs/bytes与实际profiler结果可能存在数量级差异，需要更好的编译器-硬件联合分析方法 [^34^]
5. **混合分析-ML方法**：随着GPU调度硬件日益复杂，纯分析方法可能面临局限，需要探索hybrid analytical-ML方法 [^34^]
6. **功耗-性能联合模型**：Hong & Kim (2010)的开创性工作需要扩展到现代GPU的复杂功耗管理场景 [^56^]
7. **跨厂商架构（NVIDIA/AMD/Intel）的统一模型**：现有模型大多针对特定厂商，真正vendor-agnostic的模型仍是挑战 [^173^]
