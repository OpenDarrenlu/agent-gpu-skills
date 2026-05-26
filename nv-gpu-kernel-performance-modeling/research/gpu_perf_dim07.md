# 维度7：Nsight Compute与GPU性能分析工具链 - 深入研究报告

## 维度概述

NVIDIA提供了一整套丰富的GPU性能分析工具链，这些工具直接体现了GPU性能建模的最佳实践。核心工具包括：**Nsight Compute**（内核级深度分析）、**Nsight Systems**（系统级性能分析）、**CUPTI**（CUDA Profiling Tools Interface，编程化性能数据收集）、**NPP**（NVIDIA Performance Primitives，高性能原语库）以及**Triton Profiler**（Triton内核性能分析）。这些工具共同构成了从系统级概览到指令级微架构分析的完整性能分析层次结构。

本报告基于超过25次独立搜索，涵盖NVIDIA官方文档、学术论文、技术博客和开源项目，系统性地梳理了各工具的使用方法、关键指标含义以及如何用于构建GPU性能模型。

---

## 核心发现

### 发现1：Nsight Compute是GPU内核性能分析的核心工具

**Claim**: Nsight Compute（ncu）是NVIDIA推荐的交互式内核级性能分析工具，取代了已弃用的nvprof/Visual Profiler。它基于PerfWorks框架收集指标，相比nvprof使用的CUPTI Metrics API，提供更细粒度的指标、更结构化的命名约定和更优化的内核重放机制。[^49^][^50^][^51^]

**Source**: NVIDIA官方文档 + 学术论文（LBNL/NERSC）
**URL**: https://docs.nvidia.com/nsight-compute/, https://arxiv.org/pdf/2009.05257
**Date**: 2024-2025
**Excerpt**: "Nsight Compute dives a bit deeper and allows for collection of more detailed performance metrics such as warp issues statistics, instruction pipeline utilization, and memory access pattern...The Nsight profiling toolkit is replacing nvprof as the new performance tool suite for NVIDIA GPU developers."
**Context**: Nsight Compute支持Volta及更新的GPU架构，提供GUI（ncu-ui）和CLI（ncu）两种界面。
**Confidence**: high

### 发现2：Speed of Light（SOL）分析是性能瓶颈分类的核心方法论

**Claim**: Nsight Compute的Speed of Light Throughput（SOL）分析通过比较Compute Throughput（SM利用率）和Memory Throughput（内存利用率）来快速判定内核是compute-bound、memory-bound还是latency-bound。[^54^][^55^][^56^][^57^]

**Source**: NVIDIA Nsight Compute文档 + 技术博客
**URL**: https://lobehub.com/skills/nvidia-skills-perf-nsight-compute-analysis
**Date**: 2025
**Excerpt**: "NVIDIA Nsight Compute (ncu) profiles individual CUDA kernels to determine why they are slow and what to optimize. It measures GPU throughput as a percentage of theoretical peak (Speed of Light / SOL%), enabling systematic bottleneck classification and targeted optimization."
**Context**: SOL分析的具体判定规则如下：
- **Latency bound**: Memory Throughput < 60% AND Compute Throughput < 30%
- **Compute bound**: Compute Throughput > 60% AND Memory Throughput < 60%
- **Memory bound**: Memory Throughput > 60% AND Compute Throughput < 30%
- **Balanced/Optimal**: Both > 60%（可能已达到硬件极限）
**Confidence**: high

### 发现3：Nsight Compute的分层Roofline模型可定位多级存储瓶颈

**Claim**: Nsight Compute集成了与Berkeley Lab合作开发的分层Roofline模型（Hierarchical Roofline），除了传统的DRAM Roofline外，还增加了L1缓存和L2缓存的Roofline，可以精确定位内核在哪一级内存层次遇到瓶颈。[^64^][^65^][^102^]

**Source**: NVIDIA Developer Blog + 学术论文
**URL**: https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/
**Date**: 2024-08-28
**Excerpt**: "The Roofline chart now has support for a hierarchical roofline, which adds rooflines for the L1 and L2 caches in addition to device memory. You can see how close their kernels are to the bandwidth limits of each memory level to determine whether their kernels have bottlenecks related to accessing memory."
**Context**: 收集的关键指标包括：kernel运行时间、各级内存层次（L1/L2/DRAM）的读写字节数、FLOPs。需要收集的NCU指标包括：`smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`等FLOP计数指标，以及`smsp__sass_l1tex_bytes_*`、`lts__t_bytes_*`、`dram__bytes_*`等各级内存字节数指标。
**Confidence**: high

### 发现4：Warp调度分析揭示延迟隐藏效率

**Claim**: Nsight Compute的Scheduler Statistics和Warp State Statistics提供了warp调度层面的深度分析。每个SM子分区（sub-partition）有一个warp scheduler，管理一个warp池（Volta: 16 warps, Turing: 8 warps）。Warp状态分为三种：Stalled（等待中）、Eligible（就绪可发射）、Selected（被选中发射）。[^63^][^182^][^188^][^190^]

**Source**: NVIDIA Nsight Compute Profiling Guide + 技术博客
**URL**: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
**Date**: 2026-03-13
**Excerpt**: "Each scheduler maintains a pool of warps that it can issue instructions for. The upper bound of warps in the pool (Theoretical Warps) is limited by the launch configuration...Active warps that are not stalled (Eligible Warps) are ready to issue their next instruction. From the set of eligible warps, the scheduler selects a single warp from which to issue one or more instructions (Issued Warp)."
**Context**: 关键指标包括：
- **Issue Slot Utilization**: 实际发射指令数/理论最大值，>70%为健康值
- **Active Warps**: 被分配到scheduler的warp数量
- **Eligible Warps**: 就绪可发射的warp数量
- **Issued Warps**: 实际被选中发射的warp数量
- **Skipped Issue Slots**: 没有eligible warp的周期数
**Confidence**: high

### 发现5：Occupancy Calculator量化SM资源利用率

**Claim**: Nsight Compute内置的Occupancy Calculator可以计算给定kernel在特定GPU上的理论multiprocessor occupancy。Occupancy定义为：Active Warps / Maximum Warps per SM，受三个资源限制：寄存器使用量、共享内存使用量和线程块大小。[^60^][^64^][^91^][^96^]

**Source**: NVIDIA Nsight Compute文档 + 技术教程
**URL**: https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
**Date**: 2026-03-13
**Excerpt**: "Occupancy is the ratio of active warps per SM to the theoretical maximum number of active warps. Low occupancy may represent kernels that are too small, unbalanced workloads, or resource contention."
**Context**: Occupancy计算公式：
```
Occupancy = min(Block限制Warp数, 寄存器限制Warp数, 共享内存限制Warp数) / SM最大Warp数
```
Nsight Compute 2021.3+添加了Occupancy Calculator活动，可以模拟调整kernel配置对occupancy的影响。实际使用中需要关注**Achieved Occupancy**（实际值）与**Theoretical Occupancy**（理论值）的差异，大差异通常表示工作负载高度不平衡。
**Confidence**: high

### 发现6：Memory Workload Analysis提供完整的内存子系统画像

**Claim**: Nsight Compute的Memory Workload Analysis通过内存图表、缓存命中率统计和吞吐量指标提供GPU内存子系统的完整性能画像，包括L1/TEX、L2、DRAM各级的访问模式、命中率和瓶颈识别。[^59^][^61^][^63^][^70^]

**Source**: NVIDIA Nsight Compute文档 + CSDN技术博客
**URL**: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
**Date**: 2026-03-13
**Excerpt**: "Memory workload analysis builds a visualization of memory transfer sizes and throughput on the profiled architecture, as well as a guide for improving performance. Heatmaps allow users to intuitively understand potential bottlenecks and under-utilizations in the memory pipeline."
**Context**: 核心指标包括：
- **Memory Throughput**: 实际内存吞吐量（GB/s）
- **Mem Busy**: 内存控制器活跃周期占比
- **Max Bandwidth**: 当前访问模式下内存带宽利用率
- **L1/TEX Hit Rate**: L1/纹理缓存命中率
- **L2 Hit Rate**: L2缓存命中率
- **L2 Compression Success Rate**: L2数据压缩成功率
- **Mem Pipes Busy**: 内存流水线忙碌比例
**Confidence**: high

### 发现7：Nsight Systems提供系统级性能视图

**Claim**: Nsight Systems（nsys）是系统级低开销统计采样分析器，用于可视化应用算法、识别最大优化机会。它追踪CPU和GPU工作负载、CUDA API调用、内存传输和多节点MPI通信，与Nsight Compute形成互补。[^64^][^69^][^70^]

**Source**: NVIDIA官方文档 + NASA HECC技术文章
**URL**: https://developer.nvidia.com/nsight-systems, https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-applications-with-nsight-systems_701.html
**Date**: 2025-2026
**Excerpt**: "Nsight Systems can provide a system-wide visualization of application performance and help users identify issues such as insufficient parallelism on the GPU, unnecessary device-host data transfers, and inefficient kernel synchronization."
**Context**: Nsight Systems的关键能力：
- CPU/GPU时间线关联
- CUDA Runtime/Driver API追踪
- cuBLAS/cuDNN/TensorRT追踪
- GPU Metrics Sampling（PCIe带宽、NVLink、DRAM活动、SM利用率、Tensor Core活动）
- 多节点MPI分析
- Python回溯采样
**Confidence**: high

### 发现8：CUPTI提供编程化性能数据收集能力

**Claim**: CUPTI（CUDA Profiling Tools Interface）是NVIDIA提供的CUDA性能分析工具接口，支持三种核心API：Callback API（API调用追踪）、Activity API（GPU工作负载追踪）、和Profiling API（硬件计数器收集）。从CUDA 12.8开始支持Blackwell架构的Hardware Event System (HES)。[^61^][^62^][^180^][^181^][^187^]

**Source**: NVIDIA CUPTI官方文档
**URL**: https://docs.nvidia.com/cupti/main/main.html
**Date**: 2026-03-16
**Excerpt**: "CUPTI supports collecting many metrics by sampling the GPU's performance monitors (PM) periodically at fixed intervals. The resulting metrics are instanced, with each sample being composed of its value and the (GPU) timestamp when it was collected."
**Context**: CUPTI的三个主要API：
1. **Callback API**: 注册CUDA Runtime/Driver API调用的回调
2. **Activity API**: 追踪GPU活动记录（kernel执行、内存拷贝等）
3. **Profiler API**: 收集硬件性能计数器（Range Profiling和PM Sampling）
4. **PC Sampling API**: 程序计数器采样，识别热点代码和stall原因
**Confidence**: high

### 发现9：NVTX是跨工具性能标注的标准API

**Claim**: NVTX（NVIDIA Tools Extension）是一个跨平台的C-based API，用于在应用程序中标注事件、代码范围和资源。Nsight Systems、Nsight Compute和CUPTI都原生支持NVTX。默认情况下NVTX调用不做任何事情，当从开发者工具启动时，调用会被重定向到工具实现。[^157^][^162^][^163^][^167^]

**Source**: NVIDIA NVTX GitHub + 官方文档
**URL**: https://github.com/NVIDIA/NVTX, https://nvidia.github.io/NVTX/
**Date**: 2026
**Excerpt**: "By default, NVTX API calls do nothing. When you launch a program from a developer tool, NVTX calls in that program are redirected to functions in the tool."
**Context**: NVTX提供的核心标注类型：
- **Markers**: 标注程序执行中的特定时间点
- **Ranges**: 标注两个时间点之间的代码范围（Push/Pop或Start/End）
- **Resource Naming/Tracking**: 为线程、mutex等资源命名
在Nsight Compute中使用`--nvtx --nvtx-include`可以只分析特定NVTX范围内的kernel。
**Confidence**: high

### 发现10：Triton Profiler（Proton）实现细粒度内核性能分析

**Claim**: Triton提供了Proton profiler，可以在用户内核和库内部进行细粒度的性能分析和仪表化。开发者可以在workgroup甚至指令级别instrument特定的通信操作、测量重叠效率和分析性能。与Nsight Compute结合使用（通过`ncu -f --nvtx --nvtx-include`）可以深入分析Triton kernel的硬件级性能。[^54^][^56^][^57^][^95^]

**Source**: TritonForge论文 + Red Hat技术博客 + Iris论文
**URL**: https://arxiv.org/html/2512.09196v1, https://next.redhat.com/2025/11/19/triton-kernel-profiling-with-nvidia-nsight-tools/
**Date**: 2025-12-09
**Excerpt**: "We use NVIDIA Nsight Compute 2025.2.1.0 to collect hardware-level metrics such as occupancy and memory efficiency. NVTX markers isolate the Triton kernel region of interest. Profiling is invoked with: ncu -f --nvtx --nvtx-include <range_label>..."
**Context**: Triton的性能分析流程通常包括：
1. 使用`@triton.autotune`探索最优配置
2. 使用CUDA Events测量kernel执行时间
3. 使用Nsight Compute收集硬件级指标
4. 使用NVTX标记隔离特定的Triton kernel区域
**Confidence**: high

### 发现11：自动化性能诊断工具正在兴起

**Claim**: 基于Nsight Compute输出进行自动化性能诊断的工具正在快速发展。ncu-cli（开源）可以解析NCU profiling数据，应用roofline分析和架构感知启发式方法，输出可操作的优化建议。TritonForge和CudaForge等LLM代理框架将Nsight Compute指标整合到代码生成和优化循环中。[^74^][^54^][^164^]

**Source**: GitHub ncu-cli + TritonForge/CudaForge论文
**URL**: https://github.com/KuangjuX/ncu-cli, https://arxiv.org/html/2512.09196v1
**Date**: 2026-03-06 / 2025-12-09
**Excerpt**: "Parses NCU profiling data, applies roofline analysis and architecture-aware heuristics, and outputs actionable optimization suggestions — in the terminal, JSON, CSV, or Markdown."
**Context**: ncu-cli支持的分析能力：
- Roofline Analysis（Compute/Memory/Latency Bound分类）
- 内存层次诊断（非合并访问、低缓存命中率、bank conflict）
- Occupancy和Launch配置分析
- Warp Stall分析（Long Scoreboard、Barrier、MIO Throttle等）
- 指令混合分析（FP16 Tensor Core利用率、LSU主导）
- 架构特定规则（Ampere cp.async、Hopper TMA、Blackwell FP4/FP6）
**Confidence**: high

---

## 技术方法论详解

### 1. Nsight Compute SOL分析详细流程

Nsight Compute的Speed of Light分析是性能建模的起点，其详细分析流程如下：

**步骤1：收集基础指标**
```bash
ncu --section SpeedOfLight --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis --section Occupancy \
    --section SchedulerStats --section WarpStateStats \
    ./your_application
```

**步骤2：解读SOL Throughput**
| 判定条件 | 瓶颈类型 | 优化方向 |
|---------|---------|---------|
| Memory > 60%, Compute < 30% | Memory-bound | 减少内存流量、融合kernel、提高arithmetic intensity |
| Compute > 60%, Memory < 30% | Compute-bound | 减少指令数、使用Tensor Core、增加ILP |
| Both < 30% | Latency/Occupancy-bound | 增加occupancy、减少寄存器使用、增加每个线程的工作量 |
| Both > 60% | Balanced或已饱和 | 已达硬件极限，考虑算法改进 |

**步骤3：深入分析子指标**
- 在Compute-bound情况下，查看Pipe Utilization图表识别哪个执行单元是瓶颈（FMA、ALU、Tensor Core等）
- 在Memory-bound情况下，查看Memory Workload Analysis确定哪一级内存（L1/L2/DRAM）是瓶颈

### 2. Compute Workload Analysis（计算工作负载分析）

Nsight Compute的Compute Workload Analysis提供了指令流水线利用率的详细视图：

**核心指标**：
- **SM Throughput**: SM整体利用率百分比
- **Active Cycles**: SM活跃周期数
- **Elapse Cycles**: 总经过周期数
- **Instructions Per Cycle (IPC)**: 每周期执行指令数
- **Pipe Utilization**: 各功能单元（FMA、ALU、Tensor Core、LSU等）利用率

**分析要点**：
- Pipe Fma/Alu Cycles Active (%)表示SM实际执行计算的百分比
- Inst Executed Pipe Lsu (%)表示加载/存储单元占用时间，高值说明SM在做大量访存而非计算
- 对于FP16 kernel，关注Tensor Core利用率，低利用率说明没有有效使用Tensor Core加速

### 3. Memory Workload Analysis（内存工作负载分析）

**内存图表解读**：
Nsight Compute的内存图表通过热力图（heatmap）可视化数据传输大小和吞吐量。不同颜色表示各pipeline的利用率等级。

**关键指标详解**：

| 指标名称 | 含义 | 优化建议 |
|---------|------|---------|
| L1/TEX Hit Rate | L1缓存命中率 | >80%为良好，<60%说明访问模式有问题 |
| L2 Hit Rate | L2缓存命中率 | >70%为良好，低值说明working set超过L2容量 |
| Mem Busy | 内存总线忙碌度 | 高值说明内存是瓶颈 |
| Max Bandwidth | 当前模式最大带宽利用率 | 接近100%说明带宽已饱和 |
| Sectors/Req | 每个请求的扇区数 | 高值说明非合并访问 |

**各级缓存分析表**（Nsight Compute Memory Tables提供）：
- L1/TEX Cache表：显示每个访问类型的指令数、命中率、扇区miss数
- L2 Cache表：显示请求数、扇区数、命中率、吞吐量
- L2 Cache Eviction Policies表（GA100+）：显示evict_first/last/normal策略的使用情况

### 4. Occupancy分析方法论

**理论Occupancy计算**：
```
Occupancy = min(LimitByWarps, LimitByRegisters, LimitBySharedMem, LimitByBlocks) / MaxWarpsPerSM
```

其中各限制因素计算：
- **Warp限制**: MaxWarpsPerSM（由架构决定，如48-64）
- **寄存器限制**: RegistersPerSM / (RegistersPerThread * ThreadsPerWarp * WarpsPerBlock)
- **共享内存限制**: SharedMemPerSM / SharedMemPerBlock
- **Block限制**: MaxBlocksPerSM

**Nsight Compute提供的Occupancy指标**：
- Theoretical Occupancy: 理论计算值
- Achieved Occupancy: 实际测量值 = (Active_warps / Active_cycles) / MAX_warps_per_SM
- Waves Per SM: 每个SM上的wave数量，<2说明延迟可能暴露
- Block Limit Registers/SM: 寄存器限制的最大block数
- Block Limit Shared Mem/SM: 共享内存限制的最大block数

**实操建议**[^57^][^96^]：
1. 选择block size为128、256或512（经验值，256是良好起点）
2. 使用`__launch_bounds__`或`-maxrregcount`控制寄存器使用
3. 使用`cudaOccupancyMaxPotentialBlockSize()` API自动确定最优block size
4. 在Nsight Compute中验证Achieved Occupancy与理论值

### 5. Warp State Statistics分析

Warp State Statistics分析warp在执行kernel期间花费周期数的各种状态[^63^][^182^][^190^]：

**Warp状态分类**：
1. **Active Warp**: 被分配到sub-partition的warp（从分配到执行完都在）
2. **Stalled Warp**: 因某种原因无法发射指令的warp
3. **Eligible Warp**: 准备好发射下一条指令的warp（active warp的子集）
4. **Selected Warp**: 当前周期被选中发射指令的warp

**常见Stall原因**（Nsight Compute可检测）[^63^]：

| Stall原因 | 说明 | 优化方向 |
|----------|------|---------|
| Long Scoreboard | 等待L1TEX操作（全局/局部/纹理内存） | 优化内存访问模式、提高缓存命中率 |
| Short Scoreboard | 等待MIO操作（共享内存/特殊指令） | 减少共享内存bank conflict |
| Wait | 等待固定延迟执行依赖 | 增加active warps、循环展开 |
| Barrier | 等待__syncthreads() | 减少同步点、平衡线程工作量 |
| MIO Throttle | MIO pipeline满 | 减少MIO指令密度 |
| Math Pipe Throttle | 数学指令pipeline满 | 平衡数学指令分布 |
| TEX Throttle | L1指令队列满（纹理操作） | 减少纹理获取、改用全局内存 |
| LG Throttle | L1TEX throttle（load/global） | 减少L1TEX pipeline压力 |
| Drain | 等待所有memory指令完成 | 减少memory指令数量 |

**分析决策树**：
1. 如果Issue Slot Utilization低（<70%）→ 查看Warp State Statistics
2. 如果Eligible Warps少 → 查看主要Stall原因
3. 如果Active Warps少 → 查看Occupancy限制因素

### 6. CUPTI编程化数据收集

**Range Profiling API使用流程**[^61^][^62^]：
1. 使用`cuptiRangeProfilerEnable()`启用Range Profiling
2. 使用`cuptiRangeProfilerGetCounterDataSize()`获取缓冲区大小
3. 使用`cuptiRangeProfilerSetConfig()`设置配置
4. 使用`cuptiRangeProfilerStart()/Stop()`定义profiling边界
5. 使用`cuptiRangeProfilerPushRange()/PopRange()`定义用户范围

**PM Sampling API使用流程**：
1. 使用`cuptiPmSamplingEnable()`启用PM采样
2. 配置采样间隔（>=1000ns for GA10x+）
3. 指定要采样的指标（如`gr__cycles_active.avg`）
4. 采集数据后使用`cuptiPmSamplingDecodeData()`解码

**支持的关键指标**：
- `gr__cycles_active.avg`: GPU活跃周期
- `sm__cycles_active.avg`: SM活跃周期
- `sm__warps_launched.sum`: 启动的warp数
- `sm__ctas_launched.sum`: 启动的CTA数

### 7. Triton Kernel性能分析流程

**完整的Triton kernel分析流程**[^54^][^95^]：

1. **初始性能评估**：使用CUDA Events测量kernel执行时间
   ```python
   start_event = torch.cuda.Event(enable_timing=True)
   end_event = torch.cuda.Event(enable_timing=True)
   # warmup iterations
   for _ in range(3):
       kernel(...)
   # timed iterations
   start_event.record()
   for _ in range(5):
       kernel(...)
   end_event.record()
   torch.cuda.synchronize()
   elapsed_ms = start_event.elapsed_time(end_event) / 5
   ```

2. **Nsight Compute硬件级分析**：
   ```bash
   ncu -f --nvtx --nvtx-include "BIG_OP" \
       --section SpeedOfLight \
       --section MemoryWorkloadAnalysis \
       --section Occupancy \
       python your_triton_kernel.py
   ```

3. **关键指标解读**：
   - Compute Throughput > 90%: compute-bound，关注指令优化
   - Memory Throughput > 60%: memory-bound，关注访存模式
   - L1/L2 Hit Rate低: 需要改善数据局部性
   - Occupancy低: 需要调整BLOCK_SIZE或num_warps

4. **Autotune配置优化**：
   ```python
   @triton.autotune(
       configs=[
           triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 
                          'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, 
                          num_stages=3, num_warps=8),
           # ... more configs
       ],
       key=['M', 'N', 'K']
   )
   @triton.jit
   def matmul_kernel(...):
       ...
   ```

---

## 架构适配性分析

### 各GPU架构的工具支持矩阵

| 架构 | Nsight Compute | Nsight Systems | CUPTI HES | 特殊功能 |
|------|---------------|----------------|-----------|---------|
| Maxwell (5.x) | No | No (VP/nvprof) | No | 仅nvprof |
| Pascal (6.x) | No | Yes | No | nvprof+Nsight Systems |
| Volta (7.x) | Yes | Yes | No | Tensor Core支持开始 |
| Turing (8.0-8.6) | Yes | Yes | No | RT Core、独立线程调度 |
| Ampere (8.9) | Yes | Yes | No | 稀疏Tensor Core、GA100 L2 fabric |
| Hopper (9.x) | Yes | Yes | No | TMA、Transformer Engine |
| Blackwell (10.x) | Yes | Yes | Yes | FP4/FP6、HES硬件事件系统 |

### 架构特定的分析注意事项

**Volta/Turing架构**[^63^]：
- 独立线程调度模型，支持per-thread program counter和call stack
- Sub-partition warp pool大小：Volta=16 warps, Turing=8 warps
- 新增scheduler状态采样功能

**Ampere架构**[^63^]：
- GA100新增L2 fabric连接两个L2 partition
- 支持L2 Cache Eviction Policies分析（evict_first/last/normal/demote）
- PM Sampling支持>=1000ns间隔（GA10x+）

**Hopper架构**：
- TMA（Tensor Memory Accelerator）新增内存访问路径
- Warp Specialization增加新的warp状态
- Nsight Compute需2022.3+版本支持

**Blackwell架构**[^187^]：
- 新增Hardware Event System (HES)用于更精确的kernel时间戳
- CUPTI 12.8+支持`cuptiActivityEnableHWTrace()`启用HES
- 新增FP4/FP6数据类型支持

### 性能模型构建中的应用

| 性能模型元素 | 推荐工具 | 关键指标 | 收集方法 |
|-------------|---------|---------|---------|
| 内存带宽上限 | Nsight Compute | dram__bytes_sum, kernel_time | `--section MemoryWorkloadAnalysis` |
| 计算峰值 | Nsight Compute | sm__inst_executed_* | `--section ComputeWorkloadAnalysis` |
| Arithmetic Intensity | Nsight Compute | FLOPs / Bytes transferred | 自定义section或使用roofline |
| Occupancy | Nsight Compute | achieved_occupancy | `--section Occupancy` |
| 延迟隐藏效率 | Nsight Compute | issue_slot_utilization, eligible_warps | `--section SchedulerStats` |
| 缓存命中率 | Nsight Compute | l1tex_hit_rate, l2_hit_rate | `--section MemoryWorkloadAnalysis` |
| 系统级瓶颈 | Nsight Systems | cudaMemcpy时间、cudaLaunchKernel时间 | `nsys profile --trace=cuda` |
| API调用模式 | CUPTI Callback | API调用序列、参数 | Callback API |
| 时间线关联 | CUPTI Activity | Correlation ID关联 | Activity API |

---

## 工具与资源

### 官方工具下载与文档

| 工具 | 文档链接 | 版本 |
|------|---------|------|
| Nsight Compute | https://docs.nvidia.com/nsight-compute/ | 2025.2 / CUDA 12.8 |
| Nsight Systems | https://docs.nvidia.com/nsight-systems/ | 2025.3 |
| CUPTI | https://docs.nvidia.com/cupti/ | CUDA 12.9 |
| NPP/NPP+ | https://docs.nvidia.com/cuda/npp/ | CUDA 12.x |
| NVTX | https://nvidia.github.io/NVTX/ | v3 |

### 开源工具与项目

| 工具 | GitHub/链接 | 功能 |
|------|------------|------|
| ncu-cli | https://github.com/KuangjuX/ncu-cli | 自动化NCU诊断 |
| TritonForge | https://arxiv.org/html/2512.09196v1 | LLM辅助Triton优化 |
| CudaForge | https://github.com/OptimAI-Lab/CudaForge | CUDA Kernel自动优化 |
| Triton Proton | 内置在Triton中 | Triton kernel profiler |
| NVTX v3 | https://github.com/NVIDIA/NVTX | 跨平台标注API |

### 推荐的性能分析命令模板

**基础分析模板**：
```bash
# 1. 首先用Nsight Systems做系统级分析
nsys profile --trace=cuda,nvtx,osrt -o report ./app

# 2. 用Nsight Compute分析特定kernel
ncu --kernel-name regex:your_kernel \
    --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    -o kernel_report ./app

# 3. 使用NVTX范围进行精细化分析
ncu -f --nvtx --nvtx-include "your_range" \
    --section full ./app

# 4. 导出为CSV进行后续分析
ncu --csv --page details -i report.ncu-rep > report.csv
```

**Hierarchical Roofline分析**[^51^][^65^]：
```bash
ncu --metrics \
  sm__sass_thread_inst_executed_op_fadd_pred_on.sum.peak_sustained,\
  sm__sass_thread_inst_executed_op_fmul_pred_on.sum.peak_sustained,\
  sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained,\
  smsp__sass_l1tex_bytes.sum.peak_sustained,\
  lts__t_bytes.sum.peak_sustained,\
  dram__bytes.sum.peak_sustained \
  ./app
```

---

## 关键引用列表

[^49^] Wang, Y., Yang, C., Farrell, S., Kurth, T., & Williams, S. "Hierarchical Roofline Performance Analysis for Deep Learning Applications." arXiv:2009.05257.

[^50^] 同上。

[^51^] 同上。

[^54^] Li, H., et al. "TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization." arXiv:2512.09196, 2025.

[^55^] NVIDIA Skills Marketplace. "perf-nsight-compute-analysis." https://lobehub.com/skills/nvidia-skills-perf-nsight-compute-analysis

[^56^] 同上。

[^57^] dhrv blog. "a primer on profiling - Nsight Compute." https://www.dhrv.org/blog/03/profiling/, 2026-03-08.

[^58^] NVIDIA Profiler User's Guide. https://docs.nvidia.com/cuda/profiler-users-guide/

[^59^] Nsight Compute Profiling Guide. https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

[^60^] Nsight Compute Documentation. https://docs.nvidia.com/nsight-compute/NsightCompute/index.html

[^61^] CUPTI Documentation. https://docs.nvidia.com/cupti/main/main.html

[^62^] CUPTI Tutorial. https://docs.nvidia.com/cupti/tutorial/tutorial.html

[^63^] Nsight Compute Profiling Guide v13.2. https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

[^64^] NVIDIA Developer Blog. "Optimizing GPU Utilization with Nsight Compute 2021.3." https://developer.nvidia.com/blog/optimizing-gpu-utilization-with-nsight-compute-2021-3/

[^65^] NVIDIA Developer Blog. "Accelerating HPC Applications with NVIDIA Nsight Compute Roofline Analysis." https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/

[^67^] NPP Documentation. https://docs.nvidia.com/cuda/npp/introduction.html

[^69^] NASA HECC. "Performance Analysis of Your GPU Applications with Nsight Systems." https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-applications-with-nsight-systems_701.html

[^70^] NVIDIA Nsight Systems. https://developer.nvidia.com/nsight-systems

[^74^] KuangjuX. "ncu-cli: Automated CUDA kernel performance diagnostics." GitHub, 2026. https://github.com/KuangjuX/ncu-cli

[^91^] youngju.dev. "CUDA Programming Fundamentals." https://www.youngju.dev/blog/gpu-cuda/cuda_programming_fundamentals.en

[^93^] Viblo. "NVIDIA Tools Bài 11: Compute - Memory Bound Phần 1." https://viblo.asia/p/nvidia-tools-bai-11-compute-memory-bound-phan-1-2oKLnnyyLQO

[^95^] Red Hat. "Triton Kernel Profiling with NVIDIA Nsight Tools." https://next.redhat.com/2025/11/19/triton-kernel-profiling-with-nvidia-nsight-tools/

[^96^] Taki Blog. "CUDA Occupancy Calculator 实战教学." https://www.taki.com.tw/blog/cuda-occupancy-calculator

[^99^] NVIDIA. "Preparing An Application For Profiling." https://docs.nvidia.com/cuda/profiler-users-guide/

[^102^] ACM PASC23. "Matrix-free SBP-SAT finite difference methods and the multigrid preconditioner on GPUs." https://dl.acm.org/doi/fullHtml/10.1145/3650200.3656614

[^157^] NVIDIA NVTX GitHub. https://github.com/NVIDIA/NVTX

[^159^] NASA HECC. "Performance Analysis of Your GPU CUDA Kernels with Nsight Compute CLI." https://www.nas.nasa.gov/hecc/support/kb/performance-analysis-of-your-gpu-cuda-kernels-with-nsight-compute-cli_706.html

[^161^] NVIDIA Developer Forums. "How to utilize PM sampling?" https://forums.developer.nvidia.com/t/how-to-utilize-pm-sampling/287170

[^163^] NVTX Documentation. https://nvidia.github.io/NVTX/

[^164^] Ouyang et al. "CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization." arXiv:2501.08071.

[^167^] NVIDIA Developer Blog. "NVIDIA Tools Extension API (NVTX): Annotation Tool for Profiling Code in Python and C/C++." https://developer.nvidia.com/blog/nvidia-tools-extension-api-nvtx-annotation-tool-for-profiling-code-in-python-and-c-c/

[^180^] CUPTI Callback API Documentation. https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html

[^181^] CUPTI Usage Documentation. https://docs.nvidia.com/cupti/main/main.html

[^182^] CSDN Blog. "解密CUDA Warp调度：从Stalled到Selected的5种阻塞场景全解析." https://blog.csdn.net/weixin_30555125/article/details/159299560

[^183^] CNBlogs. "NVIDIA Kernel级性能分析工具Nsight Compute入门详解." https://www.cnblogs.com/zhaoweiwei/p/19058528/NsightCompute

[^187^] NVIDIA CUPTI CUDA Toolkit 12.9. https://developer.nvidia.com/cupti-ctk12_9

[^188^] FirstMoonlight Blog. "Warp Scheduler." https://firstmoonlight.github.io/2025-02-08-warp-scheduler/

[^190^] zmurder Blog. "Nsight Compute示例1_总览." https://zmurder.github.io/CUDA/Nsight/Nsight%20Compute%E7%A4%BA%E4%BE%8B1_%E6%80%BB%E8%A7%88/

---

## 待深入区域

1. **Nsight Compute Custom Section开发**：如何利用Section文件和NvRules API开发自定义分析规则和报告模板，实现自动化性能报告生成。

2. **Blackwell HES深度分析**：Hardware Event System提供了更精确的时间戳收集方法，其对性能分析的影响和最佳实践尚需深入研究。

3. **多GPU性能分析协调**：在分布式训练/推理场景中，如何协调多个GPU上的Nsight Systems和Nsight Compute分析，关联跨GPU的性能数据。

4. **Triton Proton与Nsight Compute的深度集成**：如何建立从Triton IR到SASS指令的完整性能映射，实现编译器级优化指导。

5. **PM Sampling实时监控应用**：利用CUPTI PM Sampling API构建实时性能监控系统，用于生产环境中的性能异常检测。

6. **Nsight Compute的Python Report Interface自动化**：利用Python API对`.ncu-rep`文件进行程序化解析和批量分析。

7. **性能分析驱动的自动调优**：结合贝叶斯优化或强化学习，利用Nsight Compute指标作为反馈信号实现kernel自动调优。

8. **NPP Plus与自定义Kernel的性能对比**：NPP/NPP+库提供的优化原语与手写Triton/CUDA kernel在不同场景下的性能差异分析。
