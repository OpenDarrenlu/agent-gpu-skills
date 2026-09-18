# 维度9：指令级建模与SASS分析 - 深度研究报告

## 维度概述

指令级建模与SASS（Shader Assembly）分析是GPU性能建模中最底层的分析维度。SASS是NVIDIA GPU的原生机器码，是PTX（Parallel Thread Execution）经驱动编译后的最终执行形式。由于SASS是闭源的，且每个架构都有差异，深入理解指令级行为需要结合反汇编工具、微基准测试、学术模拟器和逆向工程。

本维度涵盖以下核心主题：
1. **SASS指令分析**：从CUDA/PTX到SASS的编译轨迹与反编译方法
2. **指令分类**：内存、计算、同步等指令类型的系统分类
3. **流水线映射**：每条指令映射到哪个硬件执行单元
4. **调度逻辑**：Warp调度器如何选择下一个指令
5. **延迟与吞吐量模型**：每条指令的延迟和每个流水线的峰值吞吐
6. **Occupancy计算**：资源限制下的活跃warp数
7. **工具链**：nvdisasm、cuobjdump、DocumentSASS等

---

## 核心发现

### 发现1：SASS是GPU性能建模的最精确表示层

Claim: SASS是NVIDIA GPU执行的原生机器码，相比PTX模拟，SASS跟踪驱动模拟能提供更高精度，因为它捕获了编译器优化、寄存器分配和指令调度等硬件特定细节 [^389^]。
Source: Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling (ISCA 2020)
URL: https://people.ece.ubc.ca/aamodt/publications/papers/accelsim.isca2020.pdf
Date: 2020
Excerpt: "Accel-Sim is the first academic simulation framework to support contemporary CUDA applications, modern SASS machine ISAs and simulate hand-tuned assembly in closed-source GPU binaries."
Context: Accel-Sim通过NVBit生成SASS指令跟踪，然后转换为ISA无关中间表示进行性能模拟
Confidence: high

### 发现2：SASS指令包含丰富的控制码信息

Claim: 每条SASS指令都携带控制码（Control Code），包含等待屏障掩码（wait barrier mask）、读写屏障掩码（read/write barrier mask）、yield标志和stall计数，这些信息用于静态调度指令并防止数据冒险 [^373^]。
Source: CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning (CGO 2025)
URL: https://arxiv.org/html/2501.08071v1
Date: 2025-01-14
Excerpt: "The control code is enclosed by square brackets and is separated into multiple fields by colons... The first field is the wait barrier mask... The second and third fields are read and write barrier masks... The fourth field is the yield flag... Finally, the last field is the stall count."
Context: 控制码格式示例：`[B------:R-:W2:Y:S02] LDG.E R0, [R2.64]`
Confidence: high

### 发现3：每个SM包含4个分区，每个分区有独立的Warp调度器

Claim: 从Kepler架构开始（除GP100外），每个SM包含4个子分区（SMSP，SM Sub-Partition），每个分区有独立的Warp调度器、寄存器文件和执行单元 [^375^]。
Source: Inside the SM: Warps, Partitions, and How GPUs Schedule Work
URL: https://www.alonge.dev/blog/inside-the-sm-warps-partitions-gpu-scheduling
Date: 2026-03-01
Excerpt: "The warp scheduler in each partition selects one ready warp per cycle and dispatches its instruction to the 16 FP32 cores... a single FP32 warp instruction takes 2 cycles to dispatch all 32 threads (16 per cycle)."
Context: Volta/Ampere每分区16 FP32 core，所以32线程warp需2 cycle dispatch完
Confidence: high

### 发现4：Warp调度策略主要是GTO或LRR

Claim: 虽然NVIDIA未公开具体调度算法，但学术界普遍认为GPU采用了Greedy-Then-Oldest (GTO) 或 Loose Round-Robin (LRR) 策略，或二者的结合 [^350^]。
Source: Inter-warp Divergence Aware Execution on GPUs (Northeastern University)
URL: https://repository.library.northeastern.edu/files/neu:cj82nb59m/fulltext.pdf
Date: Unknown
Excerpt: "GTO will keep on giving priority to one warp until it reaches long latency operation. Then it will pick the oldest warp... In case there are multiple warps with the same oldest age, warp with the smaller ID will be picked."
Context: GTO策略倾向于尽快完成一个warp以释放资源，LRR则在所有就绪warp间轮转
Confidence: medium

### 发现5：DocumentSASS项目成功逆向工程了nvdisasm的指令表

Claim: DocumentSASS项目通过拦截nvdisasm的动态数据生成，提取了NVIDIA官方未公开的SASS指令编码和延迟数据，为各架构（Maxwell到Blackwell）提供了完整的指令集参考 [^398^]。
Source: GitHub - 0xD0GF00D/DocumentSASS
URL: https://github.com/0xD0GF00D/DocumentSASS
Date: 2022-07-29
Excerpt: "It turns out that an extensive description of SASS instructions as well as latencies was contained in two specific strings in nvdisasm. Instead of having to write micro-benchmarks to find latencies, one could in theory just consult these files."
Context: 该项目通过拦截memcpy调用从nvdisasm中提取指令描述和延迟表
Confidence: high

### 发现6：指令延迟随架构演进显著降低

Claim: NVIDIA GPU的FP32指令延迟从Fermi的22周期逐步降低到Volta/Ampere/Hopper的4周期，这反映了流水线深度的优化 [^501^]。
Source: CS380 Lecture 10 - GPU Architecture
URL: https://vccvisualization.org/teaching/CS380/CS380_fall2022_lecture_10.pdf
Date: Unknown
Excerpt: "inst.pipe latency(L): 22(Fermi) -> 11(Kepler) -> 9(Maxwell) -> 6(Pascal) -> 4(Volta/Turing/Ampere)"
Context: 延迟降低使得用更少的warp就能隐藏延迟
Confidence: high

### 发现7：Tensor Core指令有专门的SASS编码

Claim: 不同数据类型的Tensor Core操作有对应的SASS指令：HMMA（浮点）、IMMA（整数）、BMMA（二进制）、DMMA（FP64）。Hopper新增wgmma指令编译为GMMA SASS指令 [^31^]。
Source: Benchmarking and Dissecting the Nvidia Hopper GPU Architecture (IEEE TPDS)
URL: https://arxiv.org/html/2402.13499v1
Date: 2024-02-21
Excerpt: "The mma instructions undergo compilation into SASS instructions, with the naming convention following the established patterns: HMMA (for floating-point types), IMMA (for integer types), and BMMA (for binary types)."
Context: Hopper的wgmma指令是warp-group级别，编译为HGMMA/IGMMA/BGMMA/QGMMA等新SASS指令
Confidence: high

### 发现8：Scoreboard机制跟踪指令间依赖

Claim: NVIDIA GPU使用硬件scoreboard机制跟踪指令依赖。每个warp有6个scoreboard barrier（编号0-5），用于标记正在进行的写操作和等待完成的读操作 [^362^]。
Source: GPU Glossary - Scoreboard Stall
URL: https://modal.com/gpu-glossary/perf/scoreboard-stall
Date: Unknown
Excerpt: "A warp has 6 scoreboards which the compiler uses to track data dependencies between instructions... A short scoreboard stall occurs when an instruction is waiting on the result of a variable latency instruction which does not leave the SM... A long scoreboard stall occurs when an instruction is waiting on the result of a memory operation that leaves the SM."
Context: Short scoreboard对应SFU运算、Tensor Core操作、Shared Memory操作；Long scoreboard对应Global Memory访问
Confidence: high

### 发现9：Occupancy由寄存器、共享内存和线程块大小共同限制

Claim: GPU Occupancy是活跃warp数与SM支持的最大warp数之比，受三个资源限制：每线程寄存器数、每块共享内存大小、线程块大小 [^91^]。
Source: CUDA Programming Fundamentals
URL: https://www.youngju.dev/blog/gpu-cuda/cuda_programming_fundamentals.en
Date: 2026-03-01
Excerpt: "Occupancy is determined by three resources: Register usage, Shared Memory usage, Block size. With 64K 32-bit registers per SM, using 128 registers per thread allows only 512 threads (16 warps) on the SM."
Context: 高Occupancy有助于隐藏延迟，但超过50-70%后收益递减
Confidence: high

### 发现10：CuAsmRL展示了SASS级调度的优化潜力

Claim: 通过深度强化学习自动优化SASS指令调度，可以在现有-O3优化基础上进一步提升最多26%的性能，平均提升9% [^384^]。
Source: CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning (CGO 2025)
URL: https://www.cl.cam.ac.uk/~ey204/pubs/2025_CGO.pdf
Date: 2025
Excerpt: "Experiments show that CuAsmRL can further improve the performance of existing specialized CUDA kernels transparently by up to 26%, and on average 9%."
Context: CuAsmRL集成了OpenAI Triton编译器框架，自动发现最优指令重排序
Confidence: high

---

## 技术方法论详解

### 3.1 SASS指令结构与控制码

#### 3.1.1 指令格式

一条典型的SASS指令包含三个部分 [^373^]：
1. **控制码（Control Code）**：包含调度元数据
2. **操作码（Opcode）**：指令操作类型
3. **操作数（Operands）**：寄存器和内存地址

格式示例：
```
[B------:R-:W2:Y:S02] LDG.E R0, [R2.64]
```

#### 3.1.2 控制码字段解析

控制码包含以下字段（以冒号分隔）[^379^]：

| 字段 | 名称 | 描述 |
|------|------|------|
| Bxxxxx | Wait Barrier Mask | 等待的scoreboard barrier位掩码 |
| R# | Read Barrier | 为源操作数GPR设置读barrier |
| W# | Write Barrier | 为目的GPR设置写barrier |
| Y/- | Yield Flag | Y=建议切换到其他warp |
| Sxx | Stall Count | 在下一条指令发射前等待的周期数（0-15）|

在sm_70+架构中，控制码被打包到专用的**调度控制指令**中，位于每组3条真实指令之前 [^379^]。

#### 3.1.3 固定延迟与可变延迟指令

SASS指令分为两类 [^373^]：
- **固定延迟指令**：如IADD3、FFMA等数学运算，执行周期固定
- **可变延迟指令**：如LDG.E（全局内存加载），因内存层次结构（L1、L2、全局内存）而周期不确定

自Kepler架构起，指令执行采用**静态调度**，编译器必须通过控制码中的stall count和barrier来防止数据冒险。

### 3.2 指令分类系统

#### 3.2.1 完整SASS指令分类表

根据多项研究综合整理的SASS指令分类 [^398^][^399^][^402^]：

| 类别 | 指令示例 | 功能描述 | 适用架构 |
|------|----------|----------|----------|
| **整数算术** | IADD3, IMAD, IMUL, LOP3, SHF, PRMT | 整数加/乘/位运算/移位 | >= sm_50 |
| **浮点算术** | FADD, FMUL, FFMA, HADD2, HFMA2 | FP32/FP16运算 | >= sm_50 |
| **双精度浮点** | DADD, DMUL, DFMA | FP64运算 | >= sm_50 |
| **特殊函数** | MUFU.RCP, MUFU.SIN, MUFU.EX2, MUFU.SQRT | 超越函数 | >= sm_50 |
| **Tensor Core** | HMMA, IMMA, DMMA, BMMA, QMMA, OMMA | 矩阵乘加 | >= sm_70 |
| **全局访存** | LDG.E, LDG.E.64, LDG.E.128, STG.E | 全局加载/存储 | >= sm_50 |
| **共享访存** | LDS, LDS.64, LDS.128, STS, STS.64 | Shared Memory访问 | >= sm_50 |
| **本地访存** | LDL, STL | Local Memory访问（寄存器溢出） | >= sm_50 |
| **异步拷贝** | LDGSTS, CP_ASYNC | Global->Shared异步拷贝 | >= sm_80 |
| **原子操作** | ATOMG, ATOMS | 全局/共享原子操作 | >= sm_50 |
| **同步** | BAR, BAR.SYNC, DEPBAR | CTA级屏障/依赖屏障 | >= sm_50 |
| **控制流** | BRA, SSY, BSYNC, EXIT | 分支/同步/退出 | >= sm_50 |
| **Warp通信** | SHFL, SHFL.SYNC, VOTE | Warp内寄存器交换/表决 | >= sm_50 |
| **特殊寄存器** | S2R, CS2R | 读取特殊寄存器 | >= sm_50 |

#### 3.2.2 PTX到SASS映射关系

PTX指令不直接映射到SASS，编译器可能进行指令融合或拆分 [^301^]：

| PTX指令 | SASS指令 | 延迟(周期) | 备注 |
|---------|----------|------------|------|
| add.f32 | FADD | 2 | FP32加法 |
| mad.f32 | FFMA | 2 | 融合乘加 |
| add.f64 | DADD | 4 | FP64加法 |
| mad.f64 | DFMA | 4 | FP64融合乘加 |
| add.u32 | IADD3 | 2 | 无依赖时 |
| add.u32(依赖链) | ADD3/IMAD.IADD | 4 | 有依赖时可能不同映射 |
| mad.lo.u32 | FFMA | 2 | 即使整数也跑浮点流水线 |
| mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 | HMMA.16816.F16 (x2) | 架构相关 | FP16 Tensor Core |

### 3.3 流水线映射与执行单元

#### 3.3.1 SM分区结构

现代NVIDIA GPU（Volta+）的SM被划分为4个分区 [^375^][^494^]：

**每个分区包含的执行单元：**
- 1个Warp调度器 + 1个发射单元
- 16个FP32 CUDA Core（8.x架构，GA100为16，GA10x为16+8+8）
- 8个FP64 CUDA Core（数据中心GPU）
- 16个INT32 Core
- 2个Tensor Core（Volta/Turing/Ampere）
- 8个LD/ST单元（Load/Store Unit）
- 1个SFU（Special Function Unit）
- 16KB寄存器文件
- 1个L0指令缓存

#### 3.3.2 指令到流水线映射

| 指令类型 | 目标流水线/单元 | 每分区每周期吞吐 | 完整warp所需周期 |
|----------|----------------|-------------------|-----------------|
| FFMA, FADD, FMUL | FP32 CUDA Core | 16 ops/cycle | 2 cycles |
| IADD3, LOP3, ISETP | INT32 CUDA Core | 16 ops/cycle | 2 cycles |
| DFMA, DADD | FP64 CUDA Core | 8 ops/cycle | 4 cycles |
| HMMA, IMMA | Tensor Core | 16x8xN per cycle | 1-4 cycles |
| LDG.E, STG.E | LD/ST Unit (LSU) | 16B/cycle | 2+ cycles |
| LDS, STS | LD/ST Unit (LSU) | 128B/cycle | 1-2 cycles |
| MUFU.* | SFU | ~4 ops/cycle | 8 cycles |

#### 3.3.3 指令延迟参考表

综合多项微基准测试研究的延迟数据 [^301^][^346^][^388^]：

| 指令类别 | 代表指令 | sm_70/75延迟 | sm_80/86延迟 | sm_90延迟 |
|----------|----------|-------------|-------------|----------|
| ALU (简单整数) | IADD3, LOP3 | ~4 cycles | ~4 cycles | ~4 cycles |
| ALU (整数乘加) | IMAD | ~4 cycles | ~4 cycles | ~4 cycles |
| FP32 (FMA/ADD/MUL) | FFMA, FADD, FMUL | ~4 cycles | ~4 cycles | ~4 cycles |
| FP16 (packaged) | HFMA2, HADD2 | ~4 cycles | ~4 cycles | ~4 cycles |
| FP64 | DFMA, DADD | ~8 cycles | ~8 cycles | ~8 cycles |
| Shared Memory | LDS, STS | ~28 cycles | ~28 cycles | ~28 cycles |
| Global Memory (L1命中) | LDG.E | ~30-40 cycles | ~30-40 cycles | ~30-40 cycles |
| Global Memory (L2命中) | LDG.E | ~100-200 cycles | ~100-200 cycles | ~100-200 cycles |
| Global Memory (DRAM) | LDG.E | ~400-800 cycles | ~400-800 cycles | ~400-800 cycles |
| SFU | MUFU.SQRT | ~13-48 cycles | ~13-48 cycles | ~13-48 cycles |
| Tensor Core (HMMA) | HMMA.16816 | ~4-8 cycles | ~4-8 cycles | N/A |
| Tensor Core (wgmma) | HGMMA.64x256x16 | N/A | N/A | ~128 cycles |

注意：延迟数据来自不同研究，可能因测量方法不同而有差异。

### 3.4 Warp调度逻辑

#### 3.4.1 调度器层级

GPU有两个调度层级 [^350^][^477^]：
1. **前端调度器（Fetch Scheduler）**：选择下一个取指的warp
2. **后端调度器（Issue Scheduler）**：选择下一个发射指令的warp

#### 3.4.2 调度策略

| 策略 | 描述 | 特点 |
|------|------|------|
| **Strict Round Robin (SRR)** | 严格轮询，每个warp轮流发射 | 公平但可能等待 stalled warp |
| **Loose Round Robin (LRR)** | 轮询但跳过未就绪的warp | 避免不必要等待 |
| **Greedy-Then-Oldest (GTO)** | 持续执行一个warp直到stall，然后选最老的warp | 尽早释放资源，提高吞吐 |
| **Two-Level** | 将warp分为fetch group，组内优先 | 解决同步点所有warp同时stall的问题 |

#### 3.4.3 指令发射约束

一个warp要变为**eligible**（可调度）需满足 [^363^]：
- 指令已取指到指令缓冲区
- 所需流水线（ALU/MEM等）可用
- 所有数据依赖已解决（scoreboard clear）
- 无同步屏障阻塞

### 3.5 Occupancy计算

#### 3.5.1 计算公式

```
Occupancy = Active Warps / Maximum Warps per SM
```

Active Warps = min(因寄存器限制, 因共享内存限制, 因线程块限制, 因warp限制)

#### 3.5.2 资源限制计算

假设：SM有65536个32位寄存器，最多64个warp，最多32个线程块

| 每线程寄存器 | 每SM最大线程 | 最大warp | Occupancy |
|-------------|-------------|---------|-----------|
| 32 | 2048 | 64 | 100% |
| 64 | 1024 | 32 | 50% |
| 128 | 512 | 16 | 25% |
| 255 | 256 | 8 | 12.5% |

#### 3.5.3 各架构关键参数

| 架构 | 每SM寄存器数 | 最大warp | 最大线程块 | 共享内存 |
|------|-------------|---------|-----------|----------|
| Fermi (2.0) | 32768 | 48 | 8 | 48KB |
| Kepler (3.x) | 65536 | 64 | 16 | 48KB |
| Pascal (6.x) | 65536 | 64 | 32 | 64KB |
| Volta (7.0) | 65536 | 64 | 32 | 96KB |
| Turing (7.5) | 65536 | 32 | 16 | 64KB |
| Ampere (8.0) | 65536 | 64 | 32 | 164KB |
| Ampere (8.6) | 65536 | 48 | 16 | 100KB |
| Hopper (9.0) | 65536 | 64 | 32 | 228KB |
| Blackwell (10.0) | 65536 | 64 | 32 | 128KB |

### 3.6 性能建模方法论

#### 3.6.1 GCoM性能模型

GCoM（GPU Cycle Model）是一种分析性估计方法 [^297^]：

1. 获取代表性warp的SASS指令序列
2. 进行区间分析（interval analysis），划分为可连续发射的区间和因数据依赖而stall的区间
3. 通过cache simulation估计平均内存访问延迟
4. 计算每个区间的stall周期
5. 考虑warp间的结构性冲突

总执行周期公式：
```
C_kernel = sum(C_i) / #SM
C_i = sum(C_{i,j}) / #Subcore + S_i
```

其中C_{i,j}是subcore j的执行周期，S_i是subcore间干扰的stall周期。

#### 3.6.2 隐藏延迟的Little's Law

```
L = lambda * W
```

要完全隐藏平均W周期的指令延迟，需要L个活跃warp以最大速率lambda保持流水线满负荷。如果全局内存读取需要200周期，流水线每周期可发射1条指令，则需要200条来自其他warp的指令准备就绪 [^315^]。

---

## 架构适配性分析

### 4.1 Fermi (sm_20/sm_21) - 2代
- 每SM 2个warp调度器，每个调度器每2周期可发射2条指令
- 指令延迟：~22周期（FP32）
- 无Tensor Core
- 4个分区但未完全独立

### 4.2 Kepler (sm_30/sm_35/sm_37) - 3代
- 每SM 4个warp调度器，双发射（每调度器每周期2条指令）
- 指令延迟：~11周期
- 引入了每个调度器对应一个分区的概念
- 无Tensor Core

### 4.3 Maxwell (sm_50/sm_52/sm_53) - 5代
- 每SM 4个warp调度器，单发射
- 指令延迟：~9周期
- 所有执行单元完全分区归属，无共享单元
- 无Tensor Core

### 4.4 Pascal (sm_60/sm_61/sm_62) - 6代
- 每SM 2-4个warp调度器
- 指令延迟：~6周期
- GP100引入第一代Tensor Core

### 4.5 Volta (sm_70/sm_72) - 7代
- 每SM 4个分区，每分区1个调度器
- 指令延迟：~4周期（FP32）
- 引入独立线程调度
- 第二代Tensor Core (HMMA.884)
- 控制码改为打包格式（每3条指令1个控制字）

### 4.6 Turing (sm_75) - 7.5代
- 每SM 4个分区
- 延迟：~4周期
- 第三代Tensor Core (HMMA.1688, HMMA.884)
- 引入RT Core

### 4.7 Ampere (sm_80/sm_86/sm_87) - 8代
- GA100: 每SM 4个分区, 64 FP32 core
- GA10x: 每分区支持32 FP32 ops/cycle或16 FP32+16 INT32
- 延迟：~4周期
- 第三代Tensor Core，支持稀疏性
- 引入异步拷贝LDGSTS

### 4.8 Hopper (sm_90) - 9代
- 每SM 4个分区, 128 FP32 core
- 延迟：~4周期
- 第四代Tensor Core，引入wgmma (warp-group MMA)
- 新增HGMMA/IGMMA/BGMMA/QGMMA SASS指令
- TMA (Tensor Memory Accelerator) 引擎
- INT4 mma退化为IMAD指令序列

### 4.9 Blackwell (sm_100/sm_103/sm_110/sm_120/sm_121) - 10代
- 每SM 4个分区（16个SMSP dispatch slots）
- 统一INT32/FP32单元（sm_120消费级）
- 第五代Tensor Core (tcgen05.mma)
- 新增FP4/FP6支持
- TMEM (Tensor Memory) 256KB per SM
- Scoreboard扩展到255个条目
- 128-bit SASS指令编码

---

## 工具与资源

### 5.1 官方工具

| 工具 | 功能 | 适用场景 |
|------|------|----------|
| **nvdisasm** | cubin文件的SASS反汇编，支持控制流分析 | 反编译cubin为SASS，生成CFG |
| **cuobjdump** | 从host binary提取cubin/PTX/SASS | 快速查看kernel的SASS代码 |
| **Nsight Compute** | 性能分析和SASS/Source关联 | 指令级性能分析，stall原因 |
| **CUDA Binary Utilities** | 完整文档参考 | 各架构指令集查阅 |

**nvdisasm常用命令** [^316^]：
```bash
# 基本反汇编
nvdisasm input.cubin

# 带行号信息
nvdisasm -g input.cubin

# 生成控制流图
nvdisasm -cfg input.cubin | dot -ocfg.png -Tpng

# JSON格式输出
nvdisasm -json input.cubin

# 带指令编码
nvdisasm -hex input.cubin
```

### 5.2 开源社区工具

| 工具 | 支持架构 | 功能 |
|------|----------|------|
| **DocumentSASS** [^398^] | Maxwell-Blackwell | 逆向工程nvdisasm提取指令集和延迟表 |
| **CuAssembler** [^368^] | Turing+ | SASS汇编器，支持修改控制码并重汇编 |
| **TuringAs** [^490^] | Volta/Turing | Volta/Turing的SASS汇编器 |
| **MaxAs** | Maxwell/Pascal | Maxwell/Pascal的SASS汇编器 |
| **KeplerAs** | Kepler | Kepler的SASS汇编器 |
| **NVBit** [^389^] | Volta+ | 动态二进制插桩，生成SASS跟踪 |
| **Accel-Sim** [^389^] | Volta+ | SASS跟踪驱动的周期精确模拟器 |
| **GPGPU-Sim** [^400^] | Fermi-Volta | 开源GPU性能模拟器 |
| **CuAsmRL** [^384^] | Ampere+ | 强化学习SASS调度优化器 |

### 5.3 性能分析指标

Nsight Compute可收集的关键SASS级指标 [^304^]：

| 指标类别 | 具体指标 | 含义 |
|----------|----------|------|
| 指令执行 | inst_executed, inst_issued | 执行/发射的指令数 |
| 浮点指令 | inst_fp32, inst_fp64, inst_fp16 | 各类浮点指令计数 |
| 整数指令 | inst_integer | 整数指令计数 |
| 内存指令 | inst_compute_ld_st, global_load/store | 内存访问指令 |
| warp效率 | warp_execution_efficiency | warp中活跃线程百分比 |
| Stall原因 | stall_long_sb, stall_short_sb, stall_memory throttle | 各类stall周期数 |

---

## 关键引用列表

[^297^] Dataflow-Oriented Classification and Performance Analysis of GPU-Accelerated Homomorphic Encryption, arXiv 2025
[^301^] Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis, arXiv 2022
[^304^] SASS Metrics Collection Tutorial, eunomia.dev, 2025
[^308^] CUDA Binary Utilities, NVIDIA Official Documentation
[^310^] GPU Profiling Under the Hood, eunomia.dev, 2025
[^312^] Time-predictable warp scheduling in a GPU, Hal Science, 2025
[^313^] GPU performance modeling and optimization, TU/e Thesis
[^315^] Instruction Level Parallelism, Advanced GPU Performance Modelling
[^316^] CUDA Binary Utilities 12.9 Documentation
[^317^] GPU Architecture and ASICs, Archie Sengupta Blog
[^325^] Ampere Warp Dispatch Discussion, NVIDIA Developer Forums, 2024
[^346^] Demystifying the Nvidia Ampere Architecture, arXiv 2022
[^350^] Inter-warp Divergence Aware Execution on GPUs, Northeastern University
[^362^] Scoreboard Stall, GPU Glossary, modal.com
[^363^] Understanding GPU performance, ROCm Documentation
[^368^] CuAssembler User Guide, GitHub
[^373^] CuAsmRL: Optimizing GPU SASS Schedules via Deep RL, arXiv 2025
[^375^] Inside the SM: Warps, Partitions, and GPU Scheduling, 2026
[^379^] Scheduler Architecture - PTXAS Reverse Engineering, gh.evko.io
[^384^] CuAsmRL (CGO 2025), Cambridge University
[^389^] Accel-Sim (ISCA 2020), UBC
[^391^] Low Overhead Instruction Latency Characterization, arXiv 2019
[^393^] Instruction Latency Discussion, NVIDIA Forums, 2009
[^398^] DocumentSASS GitHub Repository, 0xD0GF00D
[^399^] SASS Opcode Dictionary, GitHub Gist
[^401^] GainSight with Accel-Sim, Stanford, 2025
[^446^] tcgen05 Tensor Core Codegen, gh.evko.io
[^449^] GPGPU-Sim Configuration Files, DeepWiki
[^457^] FHECore with Accel-Sim, arXiv 2025
[^472^] CuAsmRL Paper Review, Moonlight, 2025
[^479^] NVIDIA Turing Architecture Notes, GitHub
[^487^] Reverse Engineering NVIDIA GPU Microarchitecture, c114.net
[^494^] Nsight Compute Profiling Guide 12.9, NVIDIA
[^496^] Ampere vs Volta GPU, GitHub Gist
[^498^] Warp Execution and Scheduler Discussion, NVIDIA Forums, 2025
[^501^] CS380 GPU Architecture Lecture, 2022
[^502^] CS380 GPU Architecture Lecture 2024

---

## 待深入区域

1. **Blackwell架构的详细延迟数据**：目前公开的黑盒微基准数据较少，特别是tcgen05.mma指令的延迟和吞吐量特性
2. **SASS控制码的精确语义**：stall count的具体计算方式、yield flag在不同上下文中的确切含义仍需更多逆向工程
3. **跨架构调度策略差异**：GTO/LRR等策略在不同架构中的具体实现可能不同
4. **Tensor Memory (TMEM)的精确行为**：Blackwell的TMEM子系统的延迟和带宽特性
5. **SASS级功耗建模**：指令级功耗差异的数据有限
6. **动态并行性(Dynamic Parallelism)的SASS分析**：内核启动内核的SASS级别行为
7. **多内核并发的调度交互**：多个内核同时在同一SM上执行时的warp调度行为
8. **寄存器银行冲突(Register Bank Conflict)**：如何建模和避免
9. **精确的性能模型验证**：需要更多实际应用的验证数据
10. **开源SASS汇编器的完善**：CuAssembler等新架构支持的完善程度
