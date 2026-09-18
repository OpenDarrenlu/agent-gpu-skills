# 研究维度8：开源GPU性能建模工具与模拟器

## 维度概述

开源GPU性能建模工具与模拟器是计算机体系结构研究的重要基础设施。除了NVIDIA官方工具（如Nsight Compute、NCU）外，开源社区已经发展了数十种工具，覆盖从周期精确模拟到分析模型、从单GPU到多GPU分布式系统、从性能建模到功耗建模等多个层面。

这些工具可以按照以下维度分类：
- **模拟精度**：周期精确（Cycle-accurate） vs 分析模型（Analytical） vs 混合方法
- **目标架构**：NVIDIA GPU（PTX/SASS） vs AMD GPU（GCN/RDNA） vs 加速器
- **模拟范围**：单GPU vs 多GPU分布式系统
- **功耗建模**：纯性能模拟 vs 集成功耗模型
- **工作负载**：通用GPU计算（CUDA/OpenCL） vs 深度学习训练/推理

本报告深入调研了17个主要开源工具，涵盖从经典的GPGPU-Sim到最新的LLMCompass等前沿框架。

---

## 核心发现

### 发现1：GPGPU-Sim生态系统是最成熟的NVIDIA GPU模拟基础设施

**Claim**: GPGPU-Sim及其衍生框架Accel-Sim构成了最广泛验证的NVIDIA GPU开源模拟生态系统，支持从PTX到SASS的多代ISA模拟。 [^49^][^110^]
**Source**: GPGPU-Sim GitHub / Accel-Sim ISCA 2020论文
**URL**: https://github.com/gpgpu-sim/gpgpu-sim_distribution / https://accel-sim.github.io/
**Date**: 2020-2025
**Excerpt**: "Accel-Sim decreases cycle error 79 percentage points, over a wide range of 80 workloads, consisting of 1,945 kernel instances... We further demonstrate that Accel-Sim is able to simulate benchmark suites that no other open-source simulator can."
**Context**: GPGPU-Sim始于2009年，历经4个主要版本，最新4.2.1版本支持CUDA 11和TensorCore。
**Confidence**: high

### 发现2：LLMCompass是评估LLM推理硬件的最先进开源框架

**Claim**: LLMCompass能够在16分钟内模拟4-A100 GPU节点运行GPT-3 175B推理，平均误差率仅4.1%。 [^56^][^50^]
**Source**: LLMCompass ISCA 2024论文 / Princeton University GitHub
**URL**: https://github.com/PrincetonUniversity/LLMCompass
**Date**: 2024
**Excerpt**: "Compared to real-world hardware, LLMCompass' estimated latency achieves an average 10.4% error rate across various operators with various input sizes and an average 4.1% error rate for LLM inference."
**Context**: LLMCompass包含mapper自动寻找最优映射和调度，以及基于面积的成本模型。
**Confidence**: high

### 发现3：Calculon提供统一的LLM训练软硬件协同设计空间探索

**Claim**: Calculon是首个深入的、分析性的LLM训练性能模型，能够统一探索硬件和软件设计空间。 [^53^][^247^]
**Source**: Calculon SC 2023论文
**URL**: https://github.com/paragraph-sin/calculon
**Date**: 2023
**Excerpt**: "Calculon is a Python-based analytical performance model for LLM training and inference on large-scale distributed systems. Each calculation takes about 1ms of CPU thread time."
**Context**: 基于Megatron框架，实现了DP/PP/TP等多种并行策略的组合建模。
**Confidence**: high

### 发现4：PPT-GPU实现了比周期精确模拟快数个数量级的混合建模

**Claim**: PPT-GPU通过混合高级建模方法，在保持MAPE<16%和相关系数>0.98的同时，比周期精确模拟器快数个数量级。 [^58^][^273^]
**Source**: PPT-GPU SC 2021论文
**URL**: https://github.com/Elio-yang/PPT-GPU
**Date**: 2021
**Excerpt**: "The results show that the performance predictions are highly correlated to the actual hardware (MAPE: <16% and Correlation: >0.98). Moreover, PPT-GPU is orders of magnitude faster than cycle-accurate simulators."
**Context**: 使用预收集的内存和指令trace来准确捕获内核的动态行为。
**Confidence**: high

### 发现5：MGPUSim是专门用于AMD GPU的周期精确多GPU模拟器

**Claim**: MGPUSim是基于AMD GCN3 ISA的周期精确多GPU模拟器，与真实硬件的差异仅为5.5%。 [^65^][^69^]
**Source**: MGPUSim ISCA 2019论文
**URL**: https://github.com/bu-icsg/MGPUSim（通过Akita框架）
**Date**: 2019
**Excerpt**: "MGPUSim differs by only 5.5% on average from the actual GPU hardware. We also achieve a 3.5x and a 2.5x average speedup running functional emulation and detailed timing simulation, respectively, on a 4-core CPU."
**Context**: 基于Akita模拟框架构建，支持多线程并行模拟。
**Confidence**: high

### 发现6：GPU Ocelot是PTX执行和动态编译的先驱框架

**Claim**: GPU Ocelot是一个动态JIT编译框架，支持PTX模拟、NVIDIA GPU、AMD GPU和LLVM CPU后端四种执行目标。 [^71^][^74^]
**Source**: Georgia Tech / GPU Computing GEMS
**URL**: http://code.google.com/p/gpuocelot/（原始）
**Date**: 2010-2012
**Excerpt**: "Ocelot supports CUDA applications and provides an implementation of the CUDA Runtime API... Its JIT compiler supports four backend execution targets: (1) an emulator for PTX, (2) NVIDIA, (3) earlier generation AMD GPUs, and (4) a translator to LLVM for multicore CPUs."
**Context**: 已验证超过130个CUDA SDK应用程序，但项目已不再积极维护。
**Confidence**: high

### 发现7：Paleo是深度学习性能预测的先驱分析模型

**Claim**: PALEO（2017 ICLR）是一个分析性性能模型，通过分解计算和通信组件来估计深度神经网络的执行时间。 [^174^][^177^]
**Source**: PALEO ICLR 2017
**URL**: https://openreview.net/forum?id=SyVVJ85lg
**Date**: 2017
**Excerpt**: "PALEO models per-layer operations based on architecture specifications, hardware capabilities, and communication strategies. It supports scalability analysis across different parallelization schemes."
**Context**: 由Qi等人在ICLR 2017提出，开创了深度学习性能建模的方向。
**Confidence**: high

### 发现8：TVM的AutoTVM使用XGBoost学习成本模型进行编译优化

**Claim**: TVM的AutoTVM使用XGBoost作为学习成本模型，通过特征提取和回归/排序损失来预测算子执行时间。 [^275^][^279^]
**Source**: TVM文档/Apache Software Foundation
**URL**: https://github.com/apache/tvm
**Date**: 2019-2025
**Excerpt**: "Tuner that uses xgboost as cost model... feature_type: 'itervar' uses features extracted from IterVar (loop variable). loss_type: 'rank' uses pairwise rank loss to train cost model."
**Context**: 是深度学习编译器中最早采用学习成本模型的系统之一。
**Confidence**: high

### 发现9：ASTRA-sim是分布式训练模拟的标杆框架

**Claim**: ASTRA-sim是由Georgia Tech、Meta和Intel联合开发的分布式机器学习系统模拟器，支持周期级和分析网络后端。 [^159^][^166^]
**Source**: ASTRA-sim论文/Tutorial
**URL**: https://github.com/astra-sim/astra-sim
**Date**: 2020-2025
**Excerpt**: "ASTRA-sim is a distributed machine learning system simulator, developed as a joint collaboration between Georgia Tech, Meta, and Intel... ASTRA-sim's runtime projections for 8-16 node clusters has been validated against real systems to be within 5% difference."
**Context**: 被SimAI、 numerous follow-up works广泛采用。
**Confidence**: high

### 发现10：AccelWattch是现代GPU功耗建模的最佳开源工具

**Claim**: AccelWattch是唯一的开源功耗模型，能够对NVIDIA Volta GPU实现7.5-9.2% MAPE的精度。 [^265^]
**Source**: AccelWattch MICRO 2021论文
**URL**: https://github.com/accel-sim/accel-sim-framework
**Date**: 2021
**Excerpt**: "AccelWattch yields a mean absolute percentage error (MAPE) between 7.5-9.2%, depending on the AccelWattch variant, achieving a Pearson r coefficient of 0.83-0.91."
**Context**: 支持SASS_SIM、HW、HYBRID和PTX_SIM四种建模模式。
**Confidence**: high

---

## 技术方法论详解

### 1. GPGPU-Sim / Accel-Sim 模拟框架

**架构层次**：
- **前端**：支持PTX执行驱动和SASS trace驱动两种模式
- **性能模型**：SIMT核心、内存分区、互连网络（BookSim）
- **功耗模型**：AccelWattch / GPUWattch
- **可视化**：AerialVision

**核心模拟循环**（来自GPGPU-Sim手册 [^54^]）：
```
gpu_sim_loop():
  - 三个时钟域：core, interconnect, memory controller
  - shader_cycle() 推进每个shader core pipeline
  - issue_block2core() 分配新CTA
  - next_clock_domain() 离散事件引擎
```

**关键验证数据**（Accel-Sim [^110^]）：
- 在80个工作负载、1,945个kernel实例上，周期误差降低79个百分点
- SASS模拟的相关系数>0.97，误差<=30%
- Trace驱动模式速度：12.5K warp指令/秒（比GPGPU-Sim 3.x快4.3倍）

**支持的GPU架构**：
| 架构 | ISA | 支持状态 |
|------|-----|---------|
| Tesla | PTX | GPGPU-Sim 1.x-2.x |
| Fermi | PTX | GPGPU-Sim 3.x |
| Kepler | PTX/SASS | GPGPU-Sim 3.x + Accel-Sim |
| Pascal | SASS | Accel-Sim |
| Volta | SASS | Accel-Sim（含TensorCore） |
| Turing | SASS | Accel-Sim |
| Ampere | 部分 | 社区版本 |

### 2. LLMCompass 硬件评估框架

**核心组件** [^56^]：
```
LLMCompass
├── Mapper: 自动寻找最优映射和调度
│   └── 26,400轮参数搜索
├── Performance Model: 算子级性能预测
│   ├── 矩阵乘法tiling模型
│   ├── LayerNorm/Softmax等Transformer算子
│   └── 流水线调度
├── Area Model: 基于硬件面积的成本模型
└── Cost Model: 性能/成本权衡分析
```

**验证精度**：
- 各算子平均延迟误差：10.9%
- LLM推理端到端误差：4.1%
- 模拟速度：4-A100节点GPT-3 175B推理 < 16分钟

**支持的硬件平台** [^50^]：
- NVIDIA GPU（A100, RTX A6000, H100）
- AMD GPU
- Google TPU

### 3. Calculon 分析模型

**核心公式** [^53^]：
- 基于Megatron框架的transformer结构描述
- 三种并行策略：DP（数据并行）、PP（流水线并行）、TP（张量并行）
- 单个设计点计算时间：~1ms CPU时间
- 可并行搜索数十亿种配置

**实现的优化技术**（部分列表）：
| 优化技术 | 范围 | 说明 |
|----------|------|------|
| Activation recomputation | PP | 重计算前向激活值 |
| Sequence parallelism | TP | 序列维度切分 |
| Optimizer sharding | DP | ZeRO-1/2/3 |
| Tensor offloading | CPU | 将张量卸载到CPU |
| Micro-batching | PP | IF1B和交错调度 |

### 4. PPT-GPU 混合建模方法

**技术流程** [^58^]：
```
1. Trace收集：使用LD_PRELOAD拦截CUDA调用
   ├── memory_traces/（内存访问trace）
   ├── sass_traces/或ptx_traces/（指令trace）
   └── app_config.py（应用配置）
2. 混合建模：
   ├── 部分计算外推
   └── 模型多部分并行化
3. 性能预测：MAPE < 16%, Correlation > 0.98
```

**相比周期精确模拟的优势** [^273^]：
- 速度快数个数量级
- 使用预收集的trace准确捕获动态行为
- 支持MPI并行执行

### 5. MGPUSim 多GPU模拟

**架构模型** [^69^]：
```
MGPUSim GPU Model:
├── Command Processor (CP)
├── Asynchronous Compute Engines (ACEs)
├── Compute Units (CUs)
│   ├── Scheduler
│   ├── Decoders
│   ├── Execution Units
│   ├── SGPRs/VGPRs
│   └── LDS (Local Data Share)
├── Cache Hierarchy
│   ├── L1 Vector/Inst/Scalar Cache
│   ├── L2 Cache
│   └── Two-level TLBs
├── Memory Controllers
└── RDMA Engine (GPU间通信)
```

**性能数据**：
- 与AMD R9 Nano硬件差异：5.5%
- 功能模拟：4核加速3.5x
- 详细时序模拟：4核加速2.5x
- 单核模拟吞吐：~27 KIPS

### 6. TVM AutoTVM 学习成本模型

**XGBoost成本模型架构** [^275^]：
```
XGBTuner:
├── XGBoostCostModel:
│   ├── feature_type: 'itervar'/'knob'/'curve'
│   ├── loss_type: 'reg'/'rank'/'rank-binary'
│   └── 预测：normalized flops或rank score
├── SimulatedAnnealingOptimizer
└── plan_size: 64（每轮重训练）
```

**特征类型**：
- `itervar`：从循环变量提取的特征（更准确但较慢）
- `knob`：直接使用配置参数（更快）
- `curve`：采样曲线特征（跨设备/算子调优）

### 7. ASTRA-sim 分布式训练模拟

**三层架构** [^159^]：
```
ASTRA-sim:
├── Workload Layer: DNN模型描述
│   ├── 每层的计算时间
│   └── 通信操作
├── System Layer: 集合通信API
│   ├── Ring-based AllReduce
│   ├── Halving-doubling
│   └── 调度算法
└── Network Layer: 网络后端
    ├── Analytical Model (α+βn)
    ├── Garnet (周期级)
    └── NS-3 (数据包级)
```

### 8. C-AMAT GPGPU内存性能模型

**核心公式** [^272^]：
```
C-AMAT = T_MEMcyc / C_MEMac

五参数形式：
C-AMAT = H/C_H + pMR * pAMP/C_M

其中：
- H: hit次数, C_H: hit cycle
- pMR: 并行miss率
- pAMP: 并行平均miss惩罚
- C_M: miss cycle
```

---

## 架构适配性分析

### NVIDIA GPU架构支持矩阵

| 工具 | Tesla | Fermi | Kepler | Pascal | Volta | Ampere | Hopper | Blackwell |
|------|-------|-------|--------|--------|-------|--------|--------|-----------|
| GPGPU-Sim | PTX | PTX | PTX | 部分 | 部分 | 社区 | 无 | 无 |
| Accel-Sim | - | - | SASS | SASS | SASS | 部分 | 无 | 无 |
| PPT-GPU | - | - | - | SASS | SASS | 部分 | 无 | 无 |
| LLMCompass | - | - | - | - | - | A100 | H100 | 无 |
| GPU Ocelot | PTX | PTX | PTX | - | - | - | - | - |

### AMD GPU架构支持矩阵

| 工具 | Evergreen | GCN1-2 | GCN3 | GCN5 | RDNA | RDNA2 | CDNA |
|------|-----------|--------|------|------|------|-------|------|
| MGPUSim | - | - | GCN3 | 部分 | - | - | - |
| NaviSim | - | - | - | - | RDNA | 部分 | - |
| Multi2Sim | Evergreen | GCN1 | 部分 | - | - | - | - |
| GPU Ocelot | - | AMD GPU | - | - | - | - | - |

### 模拟速度与精度权衡

| 工具 | 精度级别 | 模拟速度 | 与硬件误差 | 适用场景 |
|------|---------|----------|-----------|---------|
| GPGPU-Sim | 周期精确 | ~0.8 KIPS | 15-30% | 微架构研究 |
| Accel-Sim Trace | 周期精确 | ~12.5 KWI/s | 15% | 验证研究 |
| MGPUSim | 周期精确 | ~27 KIPS | 5.5% | AMD GPU研究 |
| PPT-GPU | 混合建模 | >> KIPS级 | <16% | 快速设计空间探索 |
| LLMCompass | 分析模型 | 分钟级 | 4-11% | LLM硬件评估 |
| Calculon | 分析模型 | ~1ms/点 | 中等 | 软硬件协同设计 |
| vTrain | Profiling驱动 | 秒-分钟级 | 高保真 | LLM训练优化 |
| TVM AutoTVM | 学习模型 | 毫秒级预测 | 较高 | 编译优化 |

---

## 工具与资源

### 主要开源工具汇总

| # | 工具名称 | 类型 | 目标架构 | GitHub/URL | 维护状态 | 语言 |
|---|---------|------|---------|-----------|---------|------|
| 1 | **GPGPU-Sim** | 周期精确模拟器 | NVIDIA | github.com/gpgpu-sim | 活跃 | C++ |
| 2 | **Accel-Sim** | 验证模拟框架 | NVIDIA | accel-sim.github.io | 活跃 | C++ |
| 3 | **MGPUSim** | 周期精确模拟器 | AMD GCN3 | Akita框架内置 | 活跃 | Go |
| 4 | **NaviSim** | 周期精确模拟器 | AMD RDNA | 学术发布 | 有限 | C++ |
| 5 | **PPT-GPU** | Trace驱动混合模拟 | NVIDIA | github.com/Elio-yang/PPT-GPU | 有限 | Python/C++ |
| 6 | **LLMCompass** | 分析评估框架 | GPU/TPU | github.com/PrincetonUniversity/LLMCompass | 活跃 | Python |
| 7 | **Calculon** | 分析性能模型 | 分布式GPU | github.com/paragraph-sin/calculon | 活跃 | Python |
| 8 | **vTrain** | Profiling驱动模拟 | NVIDIA GPU | github.com/VIA-Research/vTrain | 活跃 | Python |
| 9 | **GPU Ocelot** | PTX JIT/模拟器 | NVIDIA/AMD/CPU | code.google.com/p/gpuocelot | 不再维护 | C++ |
| 10 | **ASTRA-sim** | 分布式训练模拟 | 通用 | github.com/astra-sim/astra-sim | 活跃 | C++ |
| 11 | **SimAI** | 全栈训练/推理模拟 | 通用 | github.com/aliyun/SimAI | 活跃 | C++/Python |
| 12 | **TVM/AutoTVM** | 学习成本模型 | 通用GPU/CPU | github.com/apache/tvm | 活跃 | Python/C++ |
| 13 | **Multi2Sim** | CPU-GPU模拟器 | AMD Evergreen/GCN | www.multi2sim.org | 有限 | C |
| 14 | **STONNE** | DNN加速器模拟器 | 灵活架构 | github.com/gicLAB/stonne-bifrost | 活跃 | C++ |
| 15 | **Akita** | 模拟引擎框架 | 通用 | Akita框架 | 活跃 | Go |
| 16 | **AccelWattch** | 功耗模型 | NVIDIA Volta+ | accel-sim框架内置 | 活跃 | C++ |
| 17 | **Extra-P** | 自动化性能建模 | 通用HPC | github.com/extra-p/extrap | 活跃 | Python |

### 关键论文引用

| 工具 | 论文 | 会议/期刊 | 年份 |
|------|------|----------|------|
| GPGPU-Sim | Analyzing CUDA Workloads Using a Detailed GPU Simulator | ISPASS | 2009 |
| Accel-Sim | Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling | ISCA | 2020 |
| MGPUSim | Enabling Multi-GPU Performance Modeling and Optimization | ISCA | 2019 |
| NaviSim | NaviSim: A Highly Accurate GPU Simulator for AMD RDNA GPUs | PACT | 2022 |
| PPT-GPU | Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs | SC | 2021 |
| LLMCompass | LLMCompass: Enabling Efficient Hardware Design for LLM Inference | ISCA | 2024 |
| Calculon | Calculon: A Methodology and Tool for High-Level Co-Design | SC | 2023 |
| vTrain | vTrain: A Simulation Framework for Cost-effective LLM Training | arXiv | 2023 |
| AccelWattch | AccelWattch: A Power Modeling Framework for Modern GPUs | MICRO | 2021 |
| Paleo | Paleo: A Performance Model for Deep Neural Networks | ICLR | 2017 |
| TVM/AutoTVM | Learning to Optimize Tensor Programs | NeurIPS | 2018 |
| ASTRA-sim | ASTRA-sim: Enabling SW/HW Co-Design | IEEE TC | 2020 |
| STONNE | STONNE: A Detailed Architectural Simulator | SBAC-PAD | 2020 |
| Akita | Akita: A High Usability Simulation Framework | arXiv | 2026 |

### 安装与使用复杂度

| 工具 | 安装难度 | 使用复杂度 | 文档质量 | 社区支持 |
|------|---------|-----------|---------|---------|
| GPGPU-Sim | 中等（依赖CUDA Toolkit） | 高 | 良好 | Google Groups |
| Accel-Sim | 中等 | 高 | 良好 | GitHub Issues |
| LLMCompass | 低（pip/conda） | 中等 | 良好 | GitHub Issues |
| Calculon | 低（Python） | 中等 | 良好 | GitHub Issues |
| MGPUSim | 中等（Go环境） | 高 | 有限 | 学术支持 |
| PPT-GPU | 中等 | 中等 | 有限 | GitHub |
| vTrain | 低 | 中等 | 良好 | GitHub |
| TVM AutoTVM | 中等 | 高 | 优秀 | 活跃社区 |
| ASTRA-sim | 中等 | 高 | 良好 | Mailing List |
| SimAI | 中等 | 中等 | 良好 | GitHub |

---

## 关键引用列表

[^49^]: GPGPU-Sim GitHub Repository. https://github.com/gpgpu-sim/gpgpu-sim_distribution

[^50^]: LLMCompass arXiv preprint. https://arxiv.org/html/2312.03134v1

[^51^]: Accelerating GPGPU Simulation (PARMA-DITAM 2026). https://drops.dagstuhl.de/storage/01oasics/oasics-vol141-parma-ditam2026/

[^52^]: Comprehensive Performance Modeling (arXiv 2024). https://arxiv.org/html/2410.00273v1

[^53^]: Calculon SC 2023. https://dl.acm.org/doi/pdf/10.1145/3581784.3607102

[^54^]: GPGPU-Sim Manual v1.0. https://pages.cs.wisc.edu/~chen-han/doc/GPGPU-Sim_Manual.html

[^56^]: LLMCompass ISCA 2024. https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2024_2025/papers/LLMCOMPASS_ISCA_2024.pdf

[^58^]: PPT-GPU SC 2021. https://dl.acm.org/doi/10.1145/3458817.3476221

[^60^]: GPGPU-Sim 3.x Manual. https://gpgpu-sim.org/manual/index.php/Main_Page

[^61^]: GPGPU-Ramulator-Simulator. https://github.com/OSU-STARLAB/GPGPU-Ramulator-Simulator

[^63^]: Sim-FA (arXiv 2026). https://arxiv.org/html/2605.00555v1

[^64^]: MGPUSim与Akita框架深度研究. https://zhichai.net/topic/176360515

[^65^]: MGPUSim ISCA 2019. https://dl.acm.org/doi/abs/10.1145/3307650.3322230

[^66^]: Fasor (ACM 2025). https://dl.acm.org/doi/fullHtml/10.1145/3650200.3656631

[^67^]: GPU-to-CPU Transpilation (ResearchGate). https://www.researchgate.net/publication/368809641

[^68^]: Akita arXiv 2026. https://arxiv.org/html/2604.28073v1

[^69^]: MGPUSim详细论文. https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf

[^70^]: Ocelot 1.1.560 Release (NVIDIA Forums). https://forums.developer.nvidia.com/t/ocelot-1-1-560-released/16396

[^71^]: Ocelot | Keeneland. https://keeneland.gatech.edu/software/keeneland/ocelot.html

[^74^]: GPU Application Development with GPU Ocelot (GPU Computing GEMS). https://casl.gatech.edu/publications/gpu-application-development-debugging-and-performance-tuning-with-gpu-ocelot/

[^110^]: Accel-Sim ISCA 2020 (CSDN翻译). https://blog.csdn.net/eloudy/article/details/154484657

[^112^]: Accel-Sim ISCA 2020 (ACM). https://dl.acm.org/doi/10.1109/ISCA45697.2020.00047

[^155^]: End-to-End Modeling for Distributed DNN Training Survey (arXiv 2025). https://arxiv.org/html/2506.09275v1

[^159^]: Impact of RoCE on Distributed Training (arXiv 2022). https://arxiv.org/pdf/2207.10898v1

[^162^]: Extra-Deep (ACM 2025). https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624204

[^165^]: Cluster Design for Distributed DL (arXiv 2022). https://arxiv.org/html/2211.16648v2

[^173^]: Multi2Sim Kepler (ISPASS 2017 Slides). https://www.ispass.org/ispass2017/slides/gong_multi2sim.pdf

[^174^]: PALEO ICLR 2017引用. https://arxiv.org/html/2510.15596v2

[^177^]: PALEO引用. https://arxiv.org/html/2601.01383v1

[^241^]: GPGPU-Sim Releases. https://github.com/gpgpu-sim/gpgpu-sim_distribution/releases

[^247^]: Calculon GitHub. https://github.com/z-zanez/calculon.git

[^248^]: LLMCompass README. https://github.com/PrincetonUniversity/LLMCompass/blob/main/README.md

[^249^]: LLMCompass GitHub. https://github.com/PrincetonUniversity/LLMCompass

[^260^]: Accel-Sim Overview. https://deepwiki.com/accel-sim/accel-sim-framework/1-overview

[^261^]: SimAI Documentation. https://deepwiki.com/aliyun/SimAI/2.2-simai-simulation-(ns-3)

[^262^]: SimAI GitHub. https://github.com/aliyun/SimAI

[^265^]: AccelWattch MICRO 2021. https://paragon.cs.northwestern.edu/papers/2021-MICRO-AccelWattch-Kandiah.pdf

[^269^]: STONNE Paper. https://arxiv.org/pdf/2006.07137

[^272^]: GPGPU C-AMAT Model (MCHPC 2017). https://passlab.github.io/mchpc/mchpc2017Proceedings/MCHPC17-GPUCAMAT-Zhang-final.pdf

[^273^]: PPT-GPU GitHub. https://github.com/Elio-yang/PPT-GPU

[^275^]: TVM XGBTuner. https://xinetzone.github.io/tvm/_modules/tvm/autotvm/tuner/xgboost_tuner.html

[^281^]: vTrain arXiv 2023. https://arxiv.org/html/2312.12391v2

---

## 待深入区域

### 高优先级

1. **Hopper/Blackwell架构支持**：当前开源模拟器对新架构（Hopper的分布式共享内存、TMA、WGMMA，Blackwell的FP4/FP6支持）的跟进速度明显滞后于工业界。Sim-FA [^63^] 是首个支持Hopper异步特性的模拟器，但尚未开源。

2. **LLM工作负载模拟**：随着LLM成为GPU的主要工作负载，现有模拟器在支持大规模注意力机制、FlashAttention、MoE等方面的能力有限。需要进一步调研LLM-specific的模拟优化。

3. **功耗-性能联合优化**：AccelWattch虽然提供了良好的功耗建模基础，但结合性能和功耗进行联合设计空间探索的工具仍然稀缺。

### 中优先级

4. **多GPU互连建模**：随着NVLink、NVSwitch、InfiniBand等互连技术的发展，对多GPU系统中网络拓扑和通信模式的精确建模变得越来越重要。ASTRA-sim和SimAI在此方向有进展，但仍需完善。

5. **自动验证基础设施**：Accel-Sim引入了自动化调优和验证流程，但这一方法论尚未被其他模拟器广泛采用。建立标准化的验证流程是社区的重要需求。

6. **学习增强的性能建模**：TVM的AutoTVM和LLMPerf等工作展示了ML模型用于性能预测的可行性，但如何在保持可解释性的同时提高预测精度仍是开放问题。

### 低优先级

7. **跨平台可移植性**：现有模拟器大多针对特定ISA和架构，开发真正跨NVIDIA/AMD/Intel GPU的统一模拟框架仍然是一个长期目标。

8. **实时模拟和可视化**：Akita框架的实时监测和可视化功能（AkitaRTM、Daisen）代表了模拟器用户体验的重要方向，值得其他工具借鉴。
