# GPU算子性能建模 — 交叉验证报告

## 验证概要
基于10个维度的深度研究（200+次搜索，5000+行研究报告），对核心发现进行交叉验证。

---

## High Confidence 发现（≥2个维度独立确认）

### 1. PipeWeave是当前最全面的GPU性能建模框架
- **确认维度**: Dim01(PipeWeave深度), Dim06(ML方法), Dim10(实践应用)
- **核心能力**: Kernel分解→调度模拟→多维特征分析→MLP预测
- **精度**: Kernel级MAPE 6.1%(seen), 11.4%(unseen)，比NeuSight提升6.7x
- **代码可用性**: Apache-2.0开源，GitHub仓库可用

### 2. 异构指令流水线分析是性能建模的核心
- **确认维度**: Dim01(PipeWeave), Dim04(CUTLASS), Dim09(指令级), Dim05(架构差异)
- **关键流水线**: Tensor Core, FMA, XU, LSU(Memory), ALU
- **方法论**: 每个流水线独立的demand和theoretical cycles计算
- **CUTLASS实践**: PipelineAsync/TmaAsync/UmmaAsync等类家族实现

### 3. Blackwell架构变化要求新的建模方法
- **确认维度**: Dim05(架构差异), Dim04(CUTLASS Blackwell), Dim10(FA4案例)
- **关键变化**: 
  - TMEM引入(256KB/SM, 16TB/s读)
  - tcgen05.mma(延迟恒定~11周期，与tile无关)
  - 统一INT32/FP32(混合工作负载延迟减半)
  - L1/SMEM减半128KB，统一L2 65MB
- **建模影响**: Stage-centric模型替代简单Roofline，误差从>95%降至1.31%

### 4. Warp-Specialization是延迟隐藏的关键技术
- **确认维度**: Dim04(CUTLASS), Dim02(经典模型), Dim07(Nsight)
- **核心机制**: Producer warps(TMA load) + Consumer warps(MMA compute)
- **寄存器分配**: setmaxnreg动态分配(如24/240/240)
- **延迟隐藏**: 通过异步执行和warp切换实现

### 5. 分析+ML混合方法优于纯分析方法
- **确认维度**: Dim01(PipeWeave), Dim06(ML方法), Dim02(经典模型)
- **PipeWeave**: 分析模型提供特征+MLP捕获非线性交互
- **NeuSight**: Tile-granularity + performance bounds + utilization MLP
- **纯分析模型误差**: Roofline >95%, 经典Hong-Kim 10-30%
- **纯ML模型误差**: Habitat 11.8%, 大模型>70%(OOD)

---

## Medium Confidence 发现

### 6. 微基准测试是现代性能建模的基础
- **确认维度**: Dim05(架构差异), Dim09(指令级)
- **关键参数**: 指令延迟/吞吐量、内存带宽、缓存层次参数
- **工具**: 自定义microbenchmark, nvdisasm, Nsight Compute

### 7. Nsight Compute提供关键分析指标
- **确认维度**: Dim07(工具链), Dim03(Roofline)
- **核心指标**: SOL(速度-of-light), 流水线利用率, Occupancy, Warp调度统计

---

## Conflict Zone

### C1. 纯分析模型 vs ML混合模型的选择
- **Dim02观点**: 现代分析模型(stage-centric)可实现1-5% MAE，足够精确
- **Dim01/Dim06观点**: 需要MLP捕获跨流水线非线性交互和contention
- **结论**: 对于精确预测需要ML混合；对于瓶颈诊断和优化指导，分析模型足够

### C2. Tile-granularity vs Kernel-granularity预测
- **NeuSight(Dim06)**: Tile-granularity更robust
- **PipeWeave(Dim01)**: Task-granularity(kernel分解为task)
- **结论**: 两者都比kernel-granularity好，选择取决于kernel类型

---

## 用户需求匹配度分析

| 用户需求 | 最佳匹配方法论 | 匹配度 |
|---------|--------------|--------|
| 读取kernel，提取计算流程 | PipeWeave Kernel Decomposer + CUTLASS Collective分析 | ★★★★★ |
| 计算流水线阶段延迟和掩盖 | PipeWeave Feature Analyzer + Stage-Centric模型 | ★★★★★ |
| 映射到硬件执行流程 | SASS分析 + 指令流水线映射 + Warp调度模拟 | ★★★★☆ |
| 不同GPU设计不同流水线 | CUTLASS Dispatch Policy + 架构参数化模型 | ★★★★★ |
