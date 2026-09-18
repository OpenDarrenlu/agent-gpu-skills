# Roofline模型及其扩展 — GPU性能建模深度研究报告

> 研究维度: Roofline模型及其在GPU性能分析中的扩展
> 搜索次数: 24次独立搜索，覆盖学术论文、NVIDIA官方文档、开源项目、技术博客
> 研究范围: 经典Roofline、CARM、ECM模型、PipeWeave、算术强度计算、多维特征空间

---

## 一、维度概述

Roofline模型是GPU性能分析中最基础且最具影响力的性能建模工具。自2009年Williams、Waterman和Patterson在UC Berkeley提出以来，该模型已从简单的二维可视化框架发展为包含多级缓存感知、多维流水线分析、量化感知等复杂扩展的完整方法论体系。近年来，随着GPU架构的演进（Hopper→Blackwell）和LLM工作负载的兴起，Roofline模型在推理性能分析、硬件采购决策和kernel优化方面发挥着越来越重要的作用。

本报告涵盖以下核心方向：
1. 经典Roofline模型的数学基础和GPU适配
2. Cache-Aware Roofline Model (CARM)的多级缓存扩展
3. Roofline Scaling Trajectories的SM级别scaling分析
4. Execution-Cache-Memory (ECM)模型的德国Erlangen方法论
5. PipeWeave的多维度流水线扩展
6. 算术强度(Arithmetic Intensity)的精确计算方法
7. 从2D Roofline到多维特征空间的演进

---

## 二、核心发现

### 2.1 经典Roofline模型的数学基础

Claim: 经典Roofline模型的核心公式为 Performance = min(PeakCompute, AI × Bandwidth)，其中算术强度(AI) = FLOPs / BytesMoved，ridge point = PeakCompute / Bandwidth [^108^][^65^][^70^]
Source: Williams et al., 2009 - Communications of the ACM
URL: https://doi.org/10.1145/1498765.1498785
Date: 2009
Excerpt: "The roofline model serves as a canonical framework for projecting the upper-bound performance of a specific workload. It characterizes the attainable performance P (in FLOPS) as a function of arithmetic intensity I (in FLOPs/Byte)."
Context: 原始论文由Samuel Williams, Andrew Waterman, David Patterson在UC Berkeley发表，发表于Communications of the ACM, Volume 52, Issue 4, pp. 65-76
Confidence: high

Claim: 现代GPU的ridge point（拐点）远高于CPU，这意味着GPU上的工作负载更容易落入memory-bound区域。NVIDIA H100 SXM的FP16 ridge point约为295 FLOP/Byte，而LLM decode阶段的AI仅为1-2 FLOP/Byte [^76^][^155^][^163^]
Source: Hardware-Efficient Attention for Fast Decoding / GMI Cloud GPU Analysis
URL: https://arxiv.org/html/2505.21487v1 / https://www.gmicloud.ai/en/blog/best-gpus-optimized-llm-inference
Date: 2025-05-27 / 2026-04-08
Excerpt: "The arithmetic intensity is far below the dense BF16 roofline of an Nvidia Hopper H100 SXM GPU, ~295 FLOPs per byte (989 TFLOPs / 3.35 TB/s), leaving the tensor cores severely underutilized."
Context: 这一发现对LLM推理优化具有根本性指导意义，解释了为什么decode阶段是memory-bound而prefill阶段是compute-bound
Confidence: high

Claim: naive Roofline模型在现代GPU上的预测误差可超过95%，因为：(1)数据手册峰值高估可达吞吐量(B200持续tensor-core吞吐量1,100-1,400 TFLOPS vs 数据手册2,250)；(2)忽略序列化流水线阶段(Blackwell上TMA→TMEM→Tensor Core阶段)；(3)使用单一带宽数值，忽略MI300A的256 MB Infinity Cache [^34^]
Source: Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures
URL: https://arxiv.org/html/2605.04178v1
Date: 2026-05-05
Excerpt: "The naive bound T_roofline = max(FLOPs/P_peak, bytes/B_HBM) fails for three compounding reasons... These compound multiplicatively to >>90% error."
Context: 该论文通过系统微基准测试为NVIDIA Blackwell (B200)和AMD CDNA3 (MI300A)建立了精确的性能模型，naive roofline基线误差>95%，而新模型在B200上达到1.31% MAE，在MI300A上达到0.09% MAE
Confidence: high

### 2.2 Cache-Aware Roofline Model (CARM)

Claim: CARM由Ilic等人在2013-2014年间提出，扩展了传统Roofline模型以包含多级缓存层次（L1、L2、HBM/DRAM），每一级都有自己的带宽天花板。该工具已集成到Intel Advisor中 [^55^][^123^][^22^]
Source: Cache-aware Roofline Model (CARM) - IEEE CAL 2014 / IEEE Trans. on Computers 2017
URL: https://github.com/champ-hub/carm-roofline / http://ispass.org/ispass2017/slides/ilic_carm.pdf
Date: 2024-2025 (工具持续更新)
Excerpt: "CARM includes micro-benchmarks for assessing key performance characteristics of target hardware, as well as support for dynamic binary instrumentation for extracting application arithmetic intensity and memory usage."
Context: CARM工具目前支持Intel、AMD、ARM和RISC-V CPU以及NVIDIA和AMD GPU。通过微基准测试自动构建各级缓存的带宽天花板，并通过PAPI性能计数器或DynamoRIO/Intel SDE动态二进制插桩进行应用分析
Confidence: high

Claim: 分层Roofline（Hierarchical Roofline）已被集成到NVIDIA Nsight Compute（2020年CUDA 11版本起）和Intel Advisor中。Nsight Compute通过可扩展的section文件接口支持自定义多级缓存Roofline分析 [^51^][^166^][^78^]
Source: NVIDIA Nsight Compute Documentation / NVIDIA Developer Blog
URL: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts / https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/
Date: 2024-2026
Excerpt: "Currently, Nsight Compute does not include support for the Hierarchical Roofline model, but it provides an extensible interface that allows you to create your own implementation. Using the SpeedOfLight_HierarchicalDoubleRooflineChart section file from the GitLab repository, you can create a Hierarchical Roofline chart."
Context: Nsight Compute默认只提供HBM级别的Roofline分析，但可通过自定义section文件实现L1/L2/HBM三级分析。NERSC/NVIDIA提供了示例脚本：https://gitlab.com/NERSC/roofline-on-nvidia-gpus
Confidence: high

Claim: 在分层Roofline中，每个kernel在图上用一组三色圆点表示：蓝色代表L1、红色代表L2、绿色代表HBM/DRAM。如果三个圆点彼此靠近，表示streaming式的数据访问模式和较差的缓存局部性 [^52^]
Source: Hierarchical Roofline Performance Analysis for Deep Learning Applications (Yang et al., 2020)
URL: https://arxiv.org/pdf/2009.05257
Date: 2020-08
Excerpt: "One should expect blue, red, and green circles near the L1, L2, and HBM ceilings respectively to show high memory utilization. Triplets of circles close to each other present a 'streaming' data access pattern and indicate poor cache locality."
Context: 该论文详细描述了基于Nsight Compute的分层Roofline方法论，并将其应用于DeepCAM深度学习基准测试，分析了TensorFlow和PyTorch两种实现
Confidence: high

### 2.3 Roofline Scaling Trajectories

Claim: Roofline Scaling Trajectories由Ibrahim、Williams和Oliker在2019年提出，用于分析GPU编程模型在不同并行度下的性能scaling行为。该方法通过控制active warp数/SM数来追踪性能轨迹 [^162^][^164^]
Source: Performance analysis of GPU programming models using the roofline scaling trajectories
URL: https://arxiv.org (cited in multiple papers referencing Ibrahim et al., 2019)
Date: 2019
Excerpt: "K.Z. Ibrahim, S. Williams, and L. Oliker, 'Performance analysis of GPU programming models using the roofline scaling trajectories,' in International Symposium on Benchmarking, Measuring and Optimization, Springer, 2019, pp.3-19."
Context: 该方法扩展了标准Roofline模型，不仅关注单个kernel的静态位置，还分析性能如何随并发度(scaling)变化，对于理解occupancy、latency hiding和资源竞争有重要价值
Confidence: medium

Claim: GPU的occupancy（occupancy = active warps / max supported warps）直接影响latency hiding效果。根据Little's Law for GPUs：Required Warps = Latency × Throughput。高occupancy是高性能的必要但非充分条件 [^76^]
Source: Advanced GPU Performance Modelling and Analysis (Study Guide)
URL: https://oboe.com/learn/advanced-gpu-performance-modelling-and-analysis-1330yn0/study-guide
Date: 2026-03
Excerpt: "Little's Law for GPUs: Required Warps = Latency × Throughput. This quantifies the concurrency needed to fully hide latency."
Context: Occupancy受register pressure、shared memory使用和thread block大小等因素限制。理解scaling trajectories有助于优化这些参数
Confidence: high

### 2.4 Execution-Cache-Memory (ECM) 模型

Claim: ECM模型由德国Erlangen大学Georg Hager团队开发，将循环kernel的运行时间分解为in-core执行时间(T_core)和数据传输时间(T_data)，假设L1以上的缓存层次传输是串行不重叠的。其数学表达为：T_ECM = max{T_OL, T_nOL + T_L1L2 + T_L2L3 + T_L3Mem} [^157^][^158^][^160^]
Source: A Tool for Analytic Performance Modeling of Loop Kernels / Quantifying performance bottlenecks of stencil computations using the ECM model
URL: https://arxiv.org/pdf/1702.04653 / https://ar5iv.labs.arxiv.org/html/1410.5010
Date: 2014-2017
Excerpt: "The ECM prediction on an Intel core for data in memory is given by: T_ECM,Mem = max{T_OL, T_nOL + T_i,1->L2 + T_i,2->L3 + T_i,3->MEM}. T_OL is the overlapping time for computations and stores, T_nOL is the time for the loads from registers into L1."
Context: ECM模型与Roofline模型的根本区别在于：Roofline假设所有内存层次的数据传输是重叠的（只取最慢的一级），而ECM假设它们是串行累加的。ECM使用紧凑表示法：{T_OL || T_nOL | T_L1L2 | T_L2L3 | T_L3Mem}
Confidence: high

Claim: ECM模型已被成功应用于多种架构分析，包括A64FX、Intel Ivy Bridge、Sandy Bridge等。Kerncraft工具实现了ECM模型的自动化分析，支持通过IACA进行in-core预测和通过cache simulation进行数据传输预测 [^53^][^157^][^167^]
Source: ECM modeling and performance tuning of SpMV and Lattice QCD on A64FX
URL: https://ar5iv.labs.arxiv.org/html/2103.03013
Date: 2021
Excerpt: "We present an architectural analysis of the A64FX used in the Fujitsu FX1000 supercomputer at a level of detail that allows for the construction of Execution-Cache-Memory (ECM) performance models for steady-state loops."
Context: Georg Hager团队（Erlangen National High Performance Computing Center, NHR@FAU）是该模型的主要开发者和推广者。Hager博士因性能建模工作获得了2018年ISC Gauss Award
Confidence: high

Claim: ECM模型的核心假设——Intel x86架构上缓存层次传输的非重叠特性——已被验证相当准确。但对于Power8等架构，缓存层次显示出相当大的重叠，需要调整模型 [^157^][^158^]
Source: Kerncraft: A Tool for Analytic Performance Modeling of Loop Kernels
URL: https://arxiv.org/pdf/1702.04653
Date: 2017
Excerpt: "Depending on the microarchitecture, data transfer times to different memory hierarchy levels may overlap (as in the Roofline model) or they may add up. This latter assumption was shown to fit measurements quite well on x86-based processors; on Power8, for instance, the cache hierarchy shows considerable overlap."
Context: 这解释了为什么ECM模型主要适用于Intel/AMD x86架构，而在GPU等非x86架构上需要谨慎适配。对于GPU，Roofline模型通常是首选，因为GPU的memory subsystem行为与CPU有显著差异
Confidence: high

### 2.5 PipeWeave的多维度扩展

Claim: PipeWeave（论文中也称为SYNPERF）将传统Roofline模型从二维分析扩展为多维分析，不再使用单一计算roof和单一内存roof，而是为每个关键指令流水线计算单独的理论性能上限。关键流水线包括：Tensor Core、FMA、XU（特殊函数单元）、LSU（Load/Store Unit） [^1^]
Source: PipeWeave: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
URL: https://arxiv.org/html/2601.14910v2
Date: 2025-10-27
Excerpt: "To overcome this limitation, PipeWeave expands the Roofline model into a multi-dimensional analysis. Instead of a single compute roof and a single memory roof, our model calculates a separate theoretical performance limit for every key instruction pipeline."
Context: PipeWeave的核心创新在于：(1) 将kernel分解为SM-level任务；(2) 为每个指令流水线计算demand和theoretical cycles；(3) 将这些原始特征提供给MLP学习复杂非线性交互。在11种GPU上训练，seen GPU MAPE 6.0%，unseen GPU MAPE 11.5%
Confidence: high

Claim: PipeWeave在BF16 LLM推理场景下，经典Roofline模型的MAPE在H20上为11%，在H800上高达127%，而PipeWeave在两种硬件上都保持6%左右的MAPE。这是因为H800的巨大计算能力在实践中几乎不可能完全饱和 [^1^]
Source: PipeWeave论文 - Evaluation Section
URL: https://arxiv.org/html/2601.14910v2
Date: 2025
Excerpt: "The Roofline model's MAPE for GEMM kernels between the H20 (11%) and H800 (127%). The H800 features a massive compute capacity that is exceedingly difficult to fully saturate in most practical scenarios."
Context: 这一发现直接证明了传统Roofline模型在高算力GPU上的局限性——理论峰值与实际可达性能之间存在巨大鸿沟，需要更精细的建模方法
Confidence: high

Claim: PipeWeave的pipeline throughput公式为：对于每个pipeline p，理论cycles C_p = N_ops,p / Th_p，其中N_ops,p是该pipeline的总操作数，Th_p是硬件规格中的throughput。GPU-level的cycles为：C_p^GPU = N_ops,p^GPU / (N_SM * Th_p) [^1^]
Source: PipeWeave论文 - Section IV-C
URL: https://arxiv.org/html/2601.14910v2
Date: 2025
Excerpt: "For each pipeline p, the theoretical cycles C_p needed to execute these operations are determined by dividing the total operation count N_ops,p by its corresponding throughput Th_p."
Context: 这一公式化方法是PipeWeave从Roofline模型延伸的核心——不是计算单一AI值，而是为每个流水线分别计算demand和theoretical cycles
Confidence: high

### 2.6 算术强度计算方法

Claim: 算术强度的基本公式为 AI = FLOPs / BytesMoved。对于Nsight Compute，FLOPs = 2 * FMA_count + FADD_count + FMUL_count，其中FMA_count通过smsp__sass_thread_inst_executed_op_ffma_pred_on.sum获取。DRAM bytes = (dram__sectors_read.sum + dram__sectors_write.sum) * 32 bytes/sector [^82^]
Source: Understanding Application Performance with Roofline Modeling (Towards Data Science)
URL: https://towardsdatascience.com/understanding-application-performance-with-roofline-modeling/
Date: 2025-06-20
Excerpt: "FLOPs = 2 * FMA_count + FADD_count + FMUL_count. Total_DRAM_bytes = (dram_sectors_read.sum + dram_sectors_write.sum) * 32. AI = FLOPs / Total_DRAM_Bytes."
Context: 这是使用Nsight Compute进行手动Roofline分析的标准方法论。Nsight Compute的roofline section set自动收集这些指标并生成图表
Confidence: high

Claim: Nsight Compute的roofline metric使用分层命名体系，关键rollup包括：.sum（总操作数）、.sum.per_second（操作数/秒）、.avg.pct_of_peak_sustained_elapsed（包含idle SM cycles的平均吞吐量百分比）、.avg.pct_of_peak_sustained_active（仅active SM cycles的平均吞吐量百分比） [^114^]
Source: NVIDIA Developer Forums - Nsight Compute Roofline chart
URL: https://forums.developer.nvidia.com/t/nsight-compute-roofline-chart/302739
Date: 2024-08-08
Excerpt: "Useful metric rollups are: <metric>.sum is the total operations; <metric>.sum.per_second is the total operations/second; <metric>.avg.pct_of_peak_sustained_elapsed is the average % throughput on each SM (includes idle SM cycles)."
Context: 注意sparsity_on和sparsity_off的区别——对于sm__ops_path_tensor_src_fp16_dst_fp32，peak_sustained基于sparsity_on（2x rate），所以如果只使用sparsity_off，最大pct_of_peak_sustained_elapsed只有50%
Confidence: high

Claim: 对于GEMM操作的算术强度，矩阵乘法(MM)的AI约为(2*N*M*K)/(N*M + M*K + N*K) bytes。当M,N,K较大时，AI ≈ 2*min(N,M,K)*sizeof(dtype)。对于GEMV（矩阵-向量乘），AI ≈ 0.25 FLOP/byte (FP32 dense)，远低于GPU的ridge point [^63^][^75^]
Source: FairyFuse paper / A Survey of NNVMC from Computing Workload Perspective
URL: https://arxiv.org/html/2604.20913v1 / https://arxiv.org/html/2603.18126v2
Date: 2026
Excerpt: "For a GEMV y=Ax with A in R^{n×m}: FP32 dense. Compute: nm FMA = nm FLOP. Data: 4nm bytes. AI = 0.25 FLOP/byte. Ternary (single GEMV, packed 2-bit): AI = 4 OP/byte."
Context: GEMV的低AI是LLM decode阶段成为memory-bound的根本原因。量化可以将AI提高数倍（如INT8 GEMV的AI可达FP32的4-8倍），但仍然远低于现代GPU的ridge point
Confidence: high

Claim: 不同精度下的ridge point不同。对于NVIDIA H100 80GB PCIe，FP16/BF16的HOI（Hardware Operational Intensity）= 1513 TFLOPS / 2.0 TB/s = 756.5 FLOPs/Byte。FP64的ridge point则低得多（约10.15 FLOP/Byte for H100 SXM） [^68^][^69^]
Source: Beyond Accuracy: Unveiling Inefficiency Patterns / Curvature-Aware Optimization
URL: https://arxiv.org/html/2604.05404v2 / https://arxiv.org/html/2604.05230v1
Date: 2026-04
Excerpt: "HOI = Peak FLOPs / Peak Memory Bandwidth. For the H100 80GB PCIe: 1,513 × 10^12 FLOPs/s / 2.0 × 10^12 Bytes/s = 756.5 FLOPs/Byte."
Context: 注意H100 SXM和PCIe版本的带宽差异——SXM为3.35 TB/s，PCIe为2.0 TB/s。同一GPU架构的不同形态因子会导致显著不同的ridge point
Confidence: high

### 2.7 从2D Roofline到多维特征空间

Claim: 传统Roofline模型使用单一AI维度（FLOPs/Byte）和单一性能维度（FLOP/s），无法捕捉现代GPU上的复杂交互。扩展方向包括：(1) 时间维度（Time-Based Roofline）；(2) 精度维度（Multi-precision Roofline）；(3) 能量维度（Energy Roofline）；(4) 流水线维度（Per-pipeline bottleneck analysis） [^165^][^83^]
Source: Time-Based Roofline for Deep Learning Performance Analysis / Mastering the Roofline Model
URL: https://arxiv.org/abs/2009.04598 / https://www.rickyspears.com/technology/mastering-the-roofline-model/
Date: 2020 / 2025
Excerpt: "Time-Based Roofline for Deep Learning Performance Analysis" by Y. Wang, C. Yang, S. Farrel, T. Kurth, and S. Williams, in 2020 IEEE/ACM Deep Learning on Supercomputers Workshop.
Context: Wang等人在2020年提出了Time-Based Roofline，将时间维度引入分析。后续工作进一步扩展了精度维度（FP8/FP16/FP32/FP64各自的ceiling）和能量维度（JOP/s代替FLOP/s）
Confidence: high

Claim: 量化感知的Roofline模型需要为Tensor Core和SIMT core分别设定compute ceiling。例如对于W4A8量化，内存流量减少但SIMT core需要进行dequantization计算，形成两个竞争的性能上限 [^71^]
Source: MixLLM quantization-aware roofline analysis
URL: https://arxiv.org/html/2412.14590v2
Date: 2024-12
Excerpt: "Compute Ceilings: Tensor Core vs. SIMT Dequantization. In the compute-bound regime, performance is dictated by the interplay between the Tensor Cores (handling the GEMM) and the SIMT cores (handling dequantization)."
Context: 这展示了从简单2D Roofline到多维分析的必要性——当存在多个异构计算单元（Tensor Core + SIMT）和多种精度时，单一compute ceiling无法准确描述性能上限
Confidence: high

---

## 三、技术方法论详解

### 3.1 经典Roofline模型公式

**基本公式:**
```
Performance(FLOP/s) = min(P_peak, B × AI)

其中:
  P_peak = 硬件峰值计算性能 (FLOP/s)
  B = 峰值内存带宽 (Byte/s)
  AI = 算术强度 (FLOPs/Byte) = FLOPs_executed / Bytes_moved

Ridge Point (I*) = P_peak / B (FLOPs/Byte)

当 AI < I* 时: memory-bound
当 AI >= I* 时: compute-bound
```

**NVIDIA H100 SXM5 示例:**
```
FP16 Tensor Core: P_peak = 989 TFLOPS (dense), B = 3.35 TB/s
I* = 989 / 3.35 ≈ 295 FLOPs/Byte

FP64: P_peak = 67 TFLOPS, B = 3.35 TB/s
I* = 67 / 3.35 ≈ 20 FLOPs/Byte
```

### 3.2 使用Nsight Compute进行Roofline分析

**命令行收集指标:**
```bash
# 使用内置roofline section set
ncu --set roofline ./application

# 手动收集关键指标
ncu --metrics \
  sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
  sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
  dram__sectors_read.sum,\
  dram__sectors_write.sum,\
  gpu__time_duration.avg \
  ./application
```

**计算公式:**
```
FLOPs = 2 * FMA_count + FADD_count + FMUL_count
Bytes_DRAM = (dram_sectors_read + dram_sectors_write) * 32
AI = FLOPs / Bytes_DRAM
Performance = FLOPs / gpu_time_duration
```

**Nsight Compute Roofline Section定义解析:**
```protobuf
// 来自SpeedOfLight_HierarchicalSingleRooflineChart.section
Rooflines {
  PeakWork {
    // FFMA operations per cycle * SM frequency
    ValueCyclesPerSecondExpression {
      ValuePerCycleMetrics {
        Name: "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2"
      }
      CyclesPerSecondMetric {
        Name: "sm__cycles_elapsed.avg.per_second"
      }
    }
  }
  PeakTraffic {
    // DRAM bytes per cycle * DRAM frequency
    ValueCyclesPerSecondExpression {
      ValuePerCycleMetrics {
        Name: "dram__bytes.sum.peak_sustained"
      }
      CyclesPerSecondMetric {
        Name: "dram__cycles_elapsed.avg.per_second"
      }
    }
  }
}
```

### 3.3 CARM（Cache-Aware Roofline Model）

**多级缓存扩展:**
传统Roofline: Performance = min(P_peak, B_DRAM × AI)

CARM扩展为:
```
Performance = min(P_peak, B_L1 × AI_L1, B_L2 × AI_L2, B_DRAM × AI_DRAM)

其中:
  AI_L1 = FLOPs / Bytes_moved_at_L1
  AI_L2 = FLOPs / Bytes_moved_at_L2
  AI_DRAM = FLOPs / Bytes_moved_at_DRAM
```

每层缓存都有自己的ridge point:
```
I*_L1 = P_peak / B_L1  (通常 > 1000 FLOPs/Byte)
I*_L2 = P_peak / B_L2  (通常 100-500 FLOPs/Byte)
I*_DRAM = P_peak / B_DRAM  (通常 10-300 FLOPs/Byte)
```

### 3.4 ECM模型公式详解

**ECM时间分解:**
```
T_ECM = max{T_OL, T_nOL + T_L1L2 + T_L2L3 + T_L3Mem}

T_OL: 可重叠的计算和store时间 (cycles)
T_nOL: 不可重叠的load到L1时间 (cycles)
T_L1L2: L2到L1的数据传输时间 (cycles)
T_L2L3: L3到L2的数据传输时间 (cycles)
T_L3Mem: 内存到L3的数据传输时间 (cycles)
```

**ECM紧凑表示法:**
```
{T_OL || T_nOL | T_L1L2 | T_L2L3 | T_L3Mem} (per cache line)
```

**多级预测:**
```
T_pred^L1 = max(T_OL, T_nOL)
T_pred^L2 = max(T_OL, T_nOL + T_L1L2)
T_pred^L3 = max(T_OL, T_nOL + T_L1L2 + T_L2L3)
T_pred^Mem = max(T_OL, T_nOL + T_L1L2 + T_L2L3 + T_L3Mem)
```

**与Roofline的关键区别:**
- Roofline: 所有内存层次传输完全重叠 → 只取最慢一级
- ECM: 所有缓存层次传输串行累加 → 更准确地反映x86 CPU行为

### 3.5 PipeWeave的多维分析公式

**Pipeline Demand计算:**
```
对于每个task τ_i和每个math pipeline p:
  N_ops,p = 操作计数 (通过分析kernel的算术表达式和循环迭代空间)
  C_p = N_ops,p / Th_p  (理论cycles)

SM-level聚合:
  N_ops,p^SMj = sum of all tasks on SM j
  C_p^SMj = N_ops,p^SMj / Th_p

GPU-level聚合:
  N_ops,p^GPU = sum over all SMs
  C_p^GPU = N_ops,p^GPU / (N_SM × Th_p)
```

**核心创新——每个pipeline一个"roof":**
不同于Roofline的单一AI维度，PipeWeave将执行效率（理论cycles / 测量latency）对绝对pipeline demand作图，每个pipeline独立显示saturation trend。

### 3.6 不同工作负载的AI计算

**常见操作的算术强度:**

| 操作 | 公式 | 典型AI (FP32) |
|------|------|---------------|
| Dot Product | AI = 2*N / (8*N) = 0.25 | 0.25 FLOPs/Byte |
| GEMV (y=Ax) | AI = 2*n*m / (4*nm + 4m + 4n) ≈ 0.5 | 0.5 FLOPs/Byte |
| GEMM (C=AB) | AI = 2*N*M*K / (4*(NM+MK+NK)) | 几十到几百 |
| LayerNorm | element-wise ops | < 1 FLOPs/Byte |
| Softmax | exp + reduction | 1-5 FLOPs/Byte |
| FlashAttention forward | O(N^2) fused ops | 10-100 FLOPs/Byte |

**LLM推理的AI特征:**
```
Prefill阶段: AI ≈ 50-500 FLOPs/Byte (compute-bound)
Decode阶段 (batch=1): AI ≈ 1-2 FLOPs/Byte (memory-bound)
Decode阶段 (batch=64): AI ≈ 64-128 FLOPs/Byte (接近ridge point)
```

---

## 四、架构适配性分析

### 4.1 不同NVIDIA GPU架构的Roofline参数

| GPU | FP32 TFLOPS | FP16 TC TFLOPS | HBM BW (TB/s) | Ridge Point (FP16) | Architecture |
|-----|-------------|----------------|---------------|-------------------|-------------|
| V100 | 15.7 | 125 (Tensor) | 0.9 | ~139 | Volta |
| A100 80GB SXM | 19.5 | 312 (dense) | 2.0 | ~156 | Ampere |
| H100 SXM5 | 67 | 989 (dense) | 3.35 | ~295 | Hopper |
| H200 SXM5 | 67 | 989 (dense) | 4.8 | ~206 | Hopper |
| B200 | ~? | ~2250 (datasheet) | ~8.0 | ~281 | Blackwell |
| RTX 4090 | 82.6 | ~? | 1.008 | ~82 | Ada Lovelace |

### 4.2 架构特性对Roofline的影响

**Ampere (A100):**
- 第三代Tensor Core支持TF32、BF16、FP16
- 结构化稀疏性（2:4 sparsity）可将峰值翻倍
- HBM2e带宽2 TB/s
- L2缓存40 MB

**Hopper (H100):**
- 第四代Tensor Core，FP8原生支持
- Transformer Engine动态精度管理
- HBM3带宽3.35 TB/s
- Thread Block Cluster支持跨SM协作
- TMA（Tensor Memory Accelerator）异步数据拷贝

**Blackwell (B200):**
- 第五代Tensor Core，FP4/FP6原生支持
- TMEM（Tensor Memory，256 KB per SM）用于accumulator存储
- 完全异步的Tensor Core操作
- 2-SM cooperative执行模式
- 非对称硬件scaling：Tensor Core 2x提升，但shared memory带宽、exp单元等提升有限
- Naive Roofline误差可达>95% [^34^][^38^]

### 4.3 LLM工作负载在不同架构上的表现

**FlashAttention-4在Blackwell上的Roofline分析 [^38^]:**
```
关键发现:
1. B200的Tensor Core吞吐量是H100的2x（2.25 PFLOPS vs 1 PFLOPS FP16/BF16）
2. 但shared memory带宽、exp单元、ALU等scaling更慢或不变
3. 对于典型attention工作负载，shared memory traffic和exp操作现在主导执行时间
4. 非MMA资源成为bottleneck，超过MMA计算时间的25-60%
5. 简单的Roofline分析会严重高估性能
```

**H100 vs H200在LLM推理中的差异 [^155^]:**
```
H100 SXM: compute-bound prefill, 3.35 TB/s bandwidth
H200 SXM: same compute, 4.8 TB/s bandwidth (+43%)

对于decode-bound工作负载：
- H200的额外带宽直接转化为更高的token/s
- 对于prefill-heavy工作负载：性能几乎相同

关键洞察：选择GPU时必须根据workload的瓶颈区域（memory-bound vs compute-bound）来决定
```

---

## 五、工具与资源

### 5.1 NVIDIA Nsight Compute

**功能:**
- 内置Roofline Chart section（HBM级别）
- 支持通过自定义section文件实现Hierarchical Roofline（L1/L2/HBM）
- 提供多个精度级别的roofline（FP64, FP32, FP16, Tensor Core）

**关键命令:**
```bash
# 基础roofline分析
ncu --set roofline ./app

# 收集所有层级带宽
ncu --metrics \
  dram__bytes.sum.per_second,\
  lts__t_bytes.sum.per_second,\
  l1tex__t_bytes.sum.per_second,\
  sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  gpu__time_duration.avg \
  ./app
```

**文档:**
- Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts
- 自定义Section: https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#sections
- NERSC Roofline脚本: https://gitlab.com/NERSC/roofline-on-nvidia-gpus

### 5.2 Empirical Roofline Toolkit (ERT)

**功能:**
- 通过微基准测试经验性确定机器特性
- 测量各级缓存带宽和峰值GFLOP rate
- 支持CPU和GPU（CUDA/OpenMP/MPI）
- 输出JSON格式的roofline参数和Postscript图表

**来源:**
- LBNL/AMCR: https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/empirical-roofline-tool-ert/
- CASS Community: https://cass.community/software/empirical-roofline-tool.html
- 许可证: BSD-3-Clause-LBNL

**ERT对V100的测量结果 [^109^]:**
```
FP64 CUDA Core: 7.7 TFLOP/s
FP32 CUDA Core: 15.2 TFLOP/s
FP16 CUDA Core: 29.2 TFLOP/s
FP16 Tensor Core: 103.7 TFLOP/s
```

### 5.3 CARM Tool

**功能:**
- 跨平台Cache-Aware Roofline Model
- 支持Intel/AMD/ARM/RISC-V CPU和NVIDIA/AMD GPU
- 微基准测试 + 应用分析（PAPI计数器或动态二进制插桩）
- 基于Web的结果可视化（ResultsGUI.py）

**来源:**
- GitHub: https://github.com/champ-hub/carm-roofline
- 重新设计版: https://github.com/xurula/carm-roofline-reimagined

### 5.4 Intel Advisor

**功能:**
- CPU/Memory Roofline Insights
- GPU Roofline Insights（Intel GPU）
- 支持L1/L2/L3/DRAM多级Roofline
- 自动向量化和并行化建议

**关键命令:**
```bash
advisor --collect=survey --profile-gpu --project-dir=./advisor-project ./app
advisor --collect=tripcounts --flop --profile-gpu --project-dir=./advisor-project ./app
advisor --report=roofline --project-dir=./advisor-project --output-format=html
```

**文档:**
- GPU Roofline: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/advisor-roofline.html

### 5.5 可视化脚本

**NERSC Roofline (Python/Matplotlib):**
```python
import numpy as np
import matplotlib.pyplot as plt

# H100 specs
PEAK_FLOPS = 989e12  # FP16 Tensor Core
MEMORY_BW = 3.35e12

# Ridge point
ridge = PEAK_FLOPS / MEMORY_BW

# AI range
ai = np.logspace(-1, 4, 1000)

# Roofline
perf = np.minimum(PEAK_FLOPS, MEMORY_BW * ai)

# Plot
plt.loglog(ai, perf/1e12, 'b-', linewidth=2)
plt.axvline(x=ridge, color='r', linestyle='--', label=f'Ridge: {ridge:.1f}')
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
plt.ylabel('Performance (TFLOPS)')
plt.title('H100 GPU Roofline Model')
plt.legend()
plt.show()
```

**来源:** https://github.com/cyanguwa/nersc-roofline

---

## 六、实际使用案例

### 6.1 LLM推理性能分析

**场景:** 分析Llama-2-70B在H100上的prefill vs decode性能

**方法:**
```
1. 使用Nsight Compute分别profile prefill和decode阶段的kernel
2. 计算每个kernel的AI和achieved performance
3. 在Roofline图上绘制，对比H100 ridge point (295 FLOPs/Byte FP16)
4. 分析结果:
   - Prefill: AI ≈ 100-400, 接近或超过ridge point → compute-bound
   - Decode (batch=1): AI ≈ 1-2, 远低于ridge point → memory-bound
   - Decode (batch=64): AI ≈ 64-128, 接近ridge point → 取决于具体batch size
```

**关键洞察:**
- Decode阶段的AI约为1-2 FLOPs/Byte，远低于H100的ridge point 295
- 这意味着GPU compute units >99%时间处于idle状态，等待数据到达
- 对于batch=1的70B模型，每token step需要约140 GB数据传输
- 在H100的3.35 TB/s带宽下，仅传输就需要~42ms，这是TPOT的下界

### 6.2 DeepCAM深度学习基准分析

**方法:** 使用Nsight Compute的Hierarchical Roofline分析DeepCAM的TensorFlow和PyTorch实现 [^52^]

**发现:**
- 不同训练阶段（forward/backward）具有不同的计算特征
- NVIDIA Automatic Mixed Precision (AMP)包显著改变AI和性能
- Zero-AI kernels（如数据拷贝kernel）严重影响总体FLOP rate
- 需要消除zero-AI kernel以最小化kernel launch latency

### 6.3 跨架构性能移植性分析

**方法:** 使用ERT测量empirical peak，构建normalized Roofline [^79^]

```
Normalization:
  p_norm = p_achieved / p_peak
  a_norm = a_achieved / a_thresh
  a_thresh = p_peak / b_peak
```

**发现:**
- NVIDIA架构在大问题规模下出现异常性能下降
- 不是缓存效率问题（L1/L2 hit rate稳定）
- 也不是occupancy问题（warp occupancy维持~38%）
- 根本原因：coalescing inefficiency来自flattened 1D array indexing
- 此行为在Intel和AMD架构上未出现

### 6.4 量化对Roofline位置的影响

**FP32 vs INT8 vs INT4对GEMV的AI影响:**
```
GEMV y = Ax (A in R^{n×m}):

FP32:   AI = 0.25 FLOPs/Byte
INT8:   AI ≈ 1-2 OPs/Byte (4x improvement)
INT4:   AI ≈ 4-8 OPs/Byte (16-32x improvement)

对于NVIDIA GPU (ridge point FP16: 295, ridge point INT8: ~600):
- 即使INT4的AI=8，仍然远低于GPU ridge point
- 量化帮助但不改变memory-bound的本质（对于batch=1 decode）
```

---

## 七、关键引用列表

1. Williams, S., Waterman, A., and Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Communications of the ACM, 52(4), 65-76. https://doi.org/10.1145/1498765.1498785

2. Ilic, A., Pratas, F., and Sousa, L. (2014). "Cache-aware Roofline Model: Upgrading the loft." IEEE Computer Architecture Letters.

3. Ilic, A., Pratas, F., and Sousa, L. (2017). "Beyond the Roofline: Cache-Aware Roofline Modeling." IEEE Transactions on Computers.

4. Yang, C. et al. (2020). "Hierarchical Roofline Performance Analysis for Deep Learning Applications." IEEE/ACM PMBS 2020. https://arxiv.org/abs/2009.05257

5. Yang, C. et al. (2020). "Time-Based Roofline for Deep Learning Performance Analysis." IEEE/ACM Deep Learning on Supercomputers Workshop.

6. Ding, N. and Williams, S. (2019). "An Instruction Roofline Model for GPUs." IEEE/ACM PMBS 2019.

7. Ibrahim, K.Z., Williams, S., and Oliker, L. (2019). "Performance analysis of GPU programming models using the roofline scaling trajectories." International Symposium on Benchmarking, Measuring and Optimization.

8. Hofmann, J. et al. (2015). "Performance analysis of the Kahan-enhanced scalar product on current multicore processors." https://arxiv.org/abs/1505.02586

9. Stengel, H. et al. (2014). "Quantifying performance bottlenecks of stencil computations using the Execution-Cache-Memory model." https://arxiv.org/abs/1410.5010

10. Alappat, C.L. et al. (2021). "ECM modeling and performance tuning of SpMV and Lattice QCD on A64FX." https://arxiv.org/abs/2103.03013

11. PipeWeave/SYNPERF (2025). "PipeWeave: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction." https://arxiv.org/html/2601.14910v2

12. Jarmusch, A. and Chandrasekaran, S. (2026). "Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures." https://arxiv.org/html/2605.04178v1

13. Dao, T. et al. - FlashAttention-4 (2026). "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling." https://arxiv.org/html/2603.05451v1

14. Wang, H. et al. (2025). "A Systematic Characterization of LLM Inference on GPUs." (Referenced in podcast discussion)

15. NVIDIA Nsight Compute Documentation. https://docs.nvidia.com/nsight-compute/

16. Intel Advisor GPU Roofline. https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/advisor-roofline.html

17. Empirical Roofline Toolkit (ERT). https://amcr.lbl.gov/departments/computer-science-department/ppan/roofline-performance-model/empirical-roofline-tool-ert/

18. CARM Tool. https://github.com/champ-hub/carm-roofline

19. NERSC Roofline on NVIDIA GPUs. https://gitlab.com/NERSC/roofline-on-nvidia-gpus

20. "Accelerating HPC Applications with NVIDIA Nsight Compute Roofline Analysis." NVIDIA Developer Blog, 2024. https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/

---

## 八、待深入区域

1. **Roofline Scaling Trajectories的具体实现**: Ibrahim等人(2019)的工作需要更多细节搜索，特别是如何在现代GPU架构（Hopper/Blackwell）上实现controlled scaling analysis。

2. **Energy-aware Roofline扩展**: 将功耗模型集成到Roofline框架中，形成"Energy Roofline"（JOP/s vs FLOPs/Joule），对数据中心GPU采购和调度有重要意义。

3. **Multi-GPU Roofline分析**: 现有工具主要针对单GPU，跨节点、跨GPU的 Roofline分析方法论仍需发展，特别是要考虑NVLink/InfiniBandwidth等互连带宽。

4. **Sparse computation Roofline**: 结构化稀疏性（2:4 sparsity on Ampere/Hopper）和非结构化稀疏性对AI计算和性能上限的影响需要更精确的建模。

5. **Dynamic Roofline (Runtime adaptation)**: 工作负载特征在运行时会变化（如KV cache增长导致decode AI下降），需要动态 Roofline 分析指导在线调度。

6. **CPU+GPU Heterogeneous Roofline**: 现代LLM serving常混合使用CPU和GPU（如CPU offload、disaggregated inference），需要统一的heterogeneous Roofline框架。

7. **TMA/TMEM/MBARIER等Blackwell新特性的Roofline建模**: Blackwell引入了新的异步执行原语，传统Roofline无法捕捉这些特性，需要新的建模方法（如Jarmusch 2026的stage-centric modeling）。

8. **CARM for GPU的自动化**: 虽然CARM工具已支持GPU，但在Nsight Compute中的集成程度不如CPU侧成熟，需要更自动化的GPU CARM分析流程。

9. **LLM-specific Roofline分析**: 需要更系统化的方法来分析不同LLM架构（Transformer、Mamba、MoE）在不同推理场景（SISO、SILO、LISO、LILO）下的Roofline特征。

10. **Compiler-assisted Roofline analysis**: 利用编译器插桩（而非硬件PMU）来获取AI和performance metrics的方法（如Alappat等人的LLVM IR approach），对新兴平台特别有价值。

---

*报告生成时间: 2025年*
*搜索次数: 24次独立搜索*
*覆盖来源: arXiv论文、NVIDIA官方文档、Intel文档、GitHub开源项目、技术博客*
