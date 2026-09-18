# Tensor Core 优化半精度矩阵乘揭秘

**作者**: 2know​东北大学 信号与信息处理硕士

**原文链接**: https://zhuanlan.zhihu.com/p/658306956

---

Tensor Core 优化半精度矩阵乘揭秘
摘要

在深度学习模型训练的过程中半精度矩阵乘(Half-precision matrix multiply) 扮演着十分重要的角色。 Nvidia 的 Tensor Cores 提供了原生的用于计算半精度小矩阵乘法的指令， 基于这种指令，在此基础上开发了通用半精度矩阵乘法(Half-precision General Matrix Multiply, HGEMM)的例行程序，并且可以通过高级 API 来访问。本文，我们将首次揭秘关于 Tensor Cores 如何在 NVIDIA Turning 架构上运行的大量细节，包括指令的使用，所要求的寄存器和数据排布，以及 Tensor Core 操作的吞吐和延迟。我们进一步对 Turning GPUs 的内存系统进行了基准测试并对性能进行了量化分析。我们分析显示，DRAM、L2 cache 以及 shared memory 的带宽成为了 HGEMM 的新的瓶颈，而在此之前认为其性能瓶颈是计算能力。 基于我们最新发现的一些 Tensor Core 的特性，我们对基于 Tensor Core 的基础版本 HGEMM 进行了一些列的优化，包括 Block size 优化，重新设计数据排布，数据预取，以及指令 调度等。经过广泛的评估，结果显示，在 NVIDIA Turning RTX2070 和 T4 上，我们实现的优化的 HGEMM 相较于原生的 cuBLAS 10.1 实现有平均 1.73 和 1.46 倍的性能优势。我 们的代码使用原生硬件的汇编实现。
关键字 GEMM, GPU, Tensor Core, Half-precision

I. 概述

GEMM(General Matrix Multiply, GEMM) 是深度学习的核心构件。例如，全连接层的 Tensor 操作可以直接使用 GEMM。对于卷积神经网络(Convolutional Neural Networks, CNN)来说，卷积操作可以简化为高效的 GEMM。在由多个 cell 组成的 LSTM(Long Short Term Memory)模型中，每个 cell 都要执行多个 GEMM。在最先进的 NLP 模型 BERT 中，最基础的 transformer 单元也需要使用 GEMM 操作。
考虑到 GEMM 的重要性，NVIDIA 2017 年在 Volta V100 GPU 上推出了专用硬件 Tensor Core 来加速其执行。每个 Tensor Core 在一个时钟周期内可以消费两个 4x4 的半精度(FP16)矩阵并计算他们相乘的结果。在 V100, T4 和 RTX2070 这些硬件上，Tensor Core 提供的 FLOPS 是 FP16 单元的 4 倍。同时 Tensor Core 计算结果的精度也比 FP16 计算单元的精度高。虽然 Tensor Core 的性能优势意义深远，但是了解其工作的细节以及吞吐和延迟等信息依然是十分 重要的，而这一部分内容在文献中是缺失的。
本文首先披露 NVIDIA Turning 架构上 Tensor Core 的工作细节。通过广泛的测试研究，我们详细阐述了，用于操作 Tensor Core 的 HMMA.1688 指令所需要的数据排布(data layout)。我们发现，8x8 矩阵是半精度 Tensor Core 可编程的基础单元。这一发现可以有效简化 Tensor Core 的编程。同时，我们设计了基准测试实验测试 Tensor Core 指令的吞吐和延迟。
我们测试测算 Turing GPU 的 DRAM 和 L2 cache 的性能。基于测算的 DRAM 和 L2 cache 的带宽，我们分析了基于 Tensor Core 的 HGEMM 的 roofline 模型。我们的分析结果证明，HGEMM 的性能实际上是受限于 Global Memory 的带宽，而在此之前，我们认为其性能是受限于计算能力。为了减轻 Global Memory 的读写压力，我们分析并增加了 blocking size 来最大化数据复用以减少从 Global Memory 中获取数据。
随后，我们设计基准测试实验以 CPI(clock cycle per instruction) 为指标衡量了 Turing 架构 GPU shared memory 吞吐。我们认为，在 GPU 上测试 shared memory 的 CPI 是第一步工作。CPI 可以帮助我们在 Tensor Core 指令之间插入合适数量的 shared memory 指令。另外，通过分析 shared memory 的 CPI 我们发现，shared memory 的带宽也是一个性能瓶颈。为了消除由有限的 shared memory 带宽带来的性能瓶颈，我们重新设计了 shared memory 中的数据排布，以避免 banck conflict，同时增加 shared memory 的 blocking size 在 shared memory 层级增加计算强度(注，增加计算强度有利于掩盖访存延迟)。
根据已知的硬件特性，我们实现了高度优化的基于 Tensor Core 的 HGEMM。我们在 NVIDIA Turing RTX2070 和 T4 GPU 上评估我们的 HGEMM 实现，并且与 cuBLAS 10.1 对比性能。评估的结果是，我们优化的 HGEMM 在 RTX2070 和 T4 上相比于 cuBLAS 的实现分别有 1.73x 和 1.47x 的加速。对于大矩阵矩阵计算，加速比达到 3x，达到了硬件峰值。
我们总结我们的主要贡献如下:

我们简明扼要的阐述了 Turing GPU 的 Tensor Core 是如何工作的。
我们对 Turing GPU 的 Global memory 指令(LDG) 和 shared memory 指令(LDS, STS) 的 CPI 进行了基准测试。
我们分析使用 Tensor Core 的 HGEMM 的性能，并指出大矩阵 HGEMM 的主要瓶颈是存储带宽，包括 DRAM、L2 cache 以及 shared memory。
我们在 NVIDIA 的 Turing GPU 上基于 Tensor Core 实现了 HGEMM，同时在 RTX2070 和 T4 上评估其性能。在大矩阵上，我们实现的 HGEMM 相比较于 cuBLAS 10.1 上基于 Tensor Core 的 HGEMM 有 3x 的加速。

本文接下来的内容将按照如下方式组织。在第 II 部分我们将描述一些符号和 Tensor Core 编程的基础。我们会在第 III 部分呈现我们的一些工作，在第 IV 部分展示 Turing GPU 上 Tensor Core 的细节。在第 V 部分我们通过基准测试探测 DRAM、L2 Cache 以及 shared memory。第 VI 部分描述设计和优化 HGEMM 的细节，包括选择 blocking size 和指令调度。第 VII 部分评估我们优化的 HGEMM。最后在 第 VIII 部分总结我们的工作。

II. 背景

这一部分主要介绍 GEMM 的概念和我们在工作中使用的一些符号。同时也会介绍我们如何给 Tensor Core 编程。

A. GEMM 和符号

通用矩阵乘法(GEneral Matrix Multiplication, GEMM)的标准形式是 C = \alpha AB + \beta C, 其中 A，B 和 C 分别是 m \times k, k \times n, m \times n 的矩阵，\alpha 和 \beta 是标量常数。本文我们主要关注 \alpha = 1.0f, \beta = 0.0f 的情况，也就是 C = AB 的矩阵乘法。
在此约定，接下来我们使用 m \times n \times k 来代表 C = AB，其中 A，B 和 C 分别是 m \times k, k \times n, m \times n 的矩阵。

B. Tensor Core 编程

在撰写本文的时候，程序员可以通过两种方式访问 Tensor Core。第一种编程 Tensor Core 的方式是在 CUDA C++ 级别使用 WMMA(Warp Matrix Multiply and Accumulation) API。第二种方式是调用来自 NVIDIA 的库函数，例如线性代数库 cuBLAS 或者深度学习库 cuDNN。
当使用 CUDA C++ 级别的 WMMA API 的时候，程序员需要使用 load_matrix_sync 指令加载数据，使用 mma_sync 做计算，最后使用 store_matrix_sync 存储结果。虽然与使用汇编相比，使用 CUDA C++ 需要更少的人力，但是程序员失去了对指令调度的控制。就像我们将要在第 VI-A 部分展示的基于 Tensor Core 的 HGEMM 的性能对内存带宽十分敏感，编译器是很难生成最优的指令调度。正如文献[5]中所讲，一个基础版本的 WMMA HGEMM 仅能达到设备峰值 10% 的性能。即使是高度优化的 CUDA 级别的实现例如 CUTLASS[9] 也只能达到 50% 的设备峰值性能[5]。
另一种使用 Tensor Core 的方式是通过 cuBLAS 这样的库。cuBLAS 中的 HGEMM 被认为是手工汇编编写的，使用 SASS(Streaming ASSembler, SASS 是 CUDA 中对应 GPU 的机器码的硬件指令集)。然而，在 SASS 级别的 Tensor Core 的工作细节是不公开的，这对于我们在 SASS 级别对 Tensor Core 编程是十分就有挑战的。接下来的章节，我们继续介绍 Tensor Core 工作的细节。

III. 相关工作

这一部分我们将介绍逆向 Tensor Core 的相关工作以及 GPU 存储的基准测试和 GEMM 优化。

A. 探测 Tensor Core

文献[10] [11] 中 Jia 等人做了探测 Tensor Core 的工作。他们指出Tensor Core 是通过 HMMA.884 和 HMMA.1688 两条指令控制的。他们也描述了 Tensor Core 使用 HMMA.884 计算 16x16x16 矩阵乘法的时候所需要的数据排布。相比于他们的工作，我们做了如下贡献:

我们指出使用 HMMA.1688 指令对 Tensor Core 编程的基本单元是一个 8x8 矩阵。
我们发现作为基本单元的 8x8 矩阵可以被一个 32-bit 寄存器索引，同时我们给出了 8x8 矩阵在 32-bit 寄存器中数据排布。
我们明确的指出 HMMA.1688 中的 .1688 代表 16x8x8 矩阵相乘。
我们对 HMMA.1688 指令的延迟和吞吐做了基准测试。
我们基于我们的新发现优化基于 Tensor Core 的 HGEMM。
B. GPU 存储基准测试

Mei 和 Chu [12] 介绍了细粒度 P-chase 方法(fine-grained P-chase method) 来测试 GPU 的内存层级。他们的方法可以探测到 cache size 和 latency，并且是使用 CUDA C++ 实现的。我们使用 SASS 来实现我们的基准测试。
SASS 级别的基准测试可以发现一些 CUDA C++ 级别观察不到的模式。例如，在 CUDA C++ 中，可能发射长的 LDG 指令序列(例如连续 128)，因为编译器认为这些指令没有影响，因此可以优化他们。在获取正确的 CPI 值的时候，发射长的 load/store 指令序列是必要的，随后我们将说明这在我们的分析中的重要性。

C. GEMM tuning

关于在 GPU 上优化双精度或者单精度 GEMM，前人做了大量的工作。Lai 等人 [13] 在 Fermi 和 Kepler GPU 上优化了 SGEMM。Zhang 等人 [14] 在 Kepler K20 上优化 SGEMM。Cray [15] 在 Maxwell 上优化 SGEMM。所有的这些工作都优化到接近硬件峰值(超过 85% 的限制)的性能，并且是使用 SASS 写的。当然 HGEMM 和 SGEMM 以及 DGEMM 的优化有很多相似的地方，要求特殊的数据排布，和很高的计算强度来保证 Tensor Core 流水的繁忙。为了解决新出现的问题，我们提出了一个分析模型来选择 blocking size (VI-A 部分)和调度指令(VI-C 部分)。在文献 [16] 中，Li 等人提出一套体系，可以平衡 ILP 和 TLP，目的是对不同尺寸的小矩阵进行批处理。
CUTLASS (CUDA Templates for Linear Algebra Subroutines)[9] 是一套针对线性代数的模板库，尤其对 GEMM。CUTLASS 模板库支持不同的数据类型，包括单精度、双精度、半精度以及整形。它使用了很多不同的优化，例如使用 multi-level blocking 最大化的重用数据和数据预取，以隐藏延迟。CUTLASS 库是使用 CUDA C++ 写的，因此可以方便在不同架构的 GPU 之间移植。但是他们丢失了对指令调度的控制。对基于 Tensor Core 的 HGEMM 来说，CUTLASS 通常不能达到接近硬件峰值的性能。

IV. 详述 Tensor Core

Tensor Core 比 FP16 单元拥有更高的吞吐和计算精度，但是如果不明白潜在的运行机制，将导致用户很难去理解和优化性能。本部分，我们将深度解析 Tensor Core 的运行机制。细节主要包含操作 Tensor Core 的指令，它所需要的数据排布，以及其吞吐和延迟。

A. Tensor Core 指令

Turing GPUs 使用 HMMA 指令来控制 Tensor Core 进行浮点计算。正如前文所述的逆向工程工作[11]中指出, 有两种 HMMA 指令，中缀为 .884 和 .1688。本文主要聚焦在 .1688 版本，因为它更简明。同时，我们首次指出 .1688 中缀代表 16x8x8 矩阵相乘。HMMA.1688 指令还有一类型描述符作为后缀，它表明是使用单精度(.FP32)计算还是半精度(.F16)计算。我们限定本文的讨论是在 .F16 范畴内。
一个典型的 HMMA 指令如下:
HMMA.1688.F16 \quad R0 \quad R2 \quad R6 \quad R4; (1)
它计算的是:
D_{16\times8} = A_{16\times8}B_{8\times8} + C_{16\times8} (2)

其中，D_{16\times8} 可以存储在 64-bit 寄存器，R0 和 R1, A_{16\times8} 存储在 R2 和 R3 中，B_{8\times} 存储在 R6 中，C_{16\times8} 存储在 R4 和 R5 中。当数据类型描述符为 .FP32 的时候，D 和 C 存储在 128-bit 寄存器中。半精度 Tensor Core 的核心机制中的基础构建块(basic building block)是一个 8x8 矩阵。一个 warp(32 threads) 中的一个 32-bit 寄存器可以存储 32x4=128 bytes。一个 8x8 的半精度矩阵也需要占据 128 bytes 的空间。因此一个 8x8 的半精度矩阵相当于一个 warp 中拥有相同索引的一个 32-bit 寄存器。因此，在 HMMA.1688 指令中的寄存器不是常规的线程寄存器(或者标量寄存器)，需要是可以被 warp 可见的寄存器，因为指令需要一个 warp 内的线程合作生成正确的结果。
因此 Tensor Core 的编程模型不在是原始的 CUDA 的编程模型了。在 CUDA 编程模型中，数据是每个线程私有的。如果需要访问其他线程的数据，只能通过 global memory 或者 shared memory，或者使用 shuffle 指令。相反，在 Tensor Core 的 HMMA 指令中，要求同一个 warp 中的线程相互合作，也就是允许不同线程的数据可以绝对的相互访问。

B. 寄存器中的矩阵

知道了一个 "warp 寄存器" 可以存储一个 8x8 的半精度矩阵，那么接下来的一个重要问题就是: 矩阵是如何通过寄存器划分到不同的 lane 中的？
通过观察由编译器编译 load_matrix_sync 和 mma_sync 指令生成的 SASS 代码，我们有如下发现:

一个 8x8 矩阵可以被存储为 8 行主序或者列主序，我们绘画行主序和列主序的数据排布图，如 Fig.1
HMMA.1688.F16 指令要求结果矩阵是行主序的，第一个输入矩阵是行主序，第二个输入矩阵是列主序。例如，在 Eq.(2) 中，D_{16\times8} 和 C_{16\times8} 以及 A_{16\times8} 都是行主序，B_{8\times8} 是列主序。
我们总结了我们的观察结果，展示了矩阵是如何分配到 HMMA.1688.F16 R0 R6 R4 指令的寄存器中的，如 Fig.2。
C. 性能指标

HMMA.1688.F16 指令有两个至关重要的性能指标，叫做延迟(latency)和吞吐(throughput)。我们实现了一个 SASS-level 的基准测试来获取这两个指标，我们使用 CPI(clock cycles per instruction) 来衡量吞吐指标。
具体的，我们通过发射 1000 条 HMMA.1688.F16 指令，然后记录他需要消耗的时钟周期，来计算 HMMA.1688.F16 的 CPI。为了抵消冷启动引起的指令 cache miss，我们再循环中构建了一个长的 HMMA 指令序列，其长度能够适合 L0 指令 cache。 因为 16 x 8 x 8 的矩阵乘法是由 16 个 4 x 4 x 4 的矩阵乘法组成的，每一个处理块(processing block)有 2 个 Tensor Core[17], 因此理论的 CPI 是 16 / 2 = 8. 我们测出的 CPI 是 8.06, 非常接近理论分析。
我们通过在指令序列中插入不定数量的 stall 周期，然后检查是否获取正确输出结果的方法来衡量 HMMA.1688.F16 的延迟。第一次获取半精度矩阵 D_{16\times8} (Eq.(1) 中的 R1)的正确结果，花费了 10 个时钟周期，第二次获取，花费了 14 个时钟周期。HMMA.1688 指令没有寄存器的 bank conflict。寄存器重用也不影响性能。这些指标在 RTX2070 和 T4 上是相同的，因为他们拥有相同的架构和流处理器(Streaming Multiprocessor, SM)。



重要的性能指标我们总结在表 I 中，同时，我们也列举了在 Tensor Core 上观察到关键结果，如下:

半精度 Tensor Core 编程的基础元素是 8x8 矩阵。
8x8 的半精度矩阵被划分给相同 warp 中的不同 thread，它能够被一个寄存器索引。矩阵的数据排布如 Fig.1 所示。
一条 HMMA.1688 指令计算 16x8x8 的矩阵乘法和累加计算。
我们观察到 HMMA.1688 指令没有寄存器的 bank conflict。寄存器重用不影响 HMMA 指令序列的性能。
V. 存储基准测试

因为相比于 CUDA Core，Tensor Core 提供了 4 倍以上的吞吐，性能瓶颈可能从算力转移到存储带宽。然而，关键存储的性能测试仍然是不公开的，例如 L2 cache 的吞吐，shared memory 的吞吐。为了支持随后的分析，我们在 Turning GPU 上对存储系统做了详细的基准测试。我们以 GB/s 来测试 DRAM 和 L2 cache 的吞吐用于使用 Roofline 模型进行性能以选择正确的 blocking size。我们以 CPI 来测试 global 和 shared memory，用于在 HMMA 指令间插入合适数量的访存指令。

A. DRAM and L2 Cache

a) 方法: 我们测试 DRAM 的吞吐通过发射一个具有多个 thread blocks 的 kernel，让每个 thread 加载 512 KB 的数据。为了确保数据是从 DRAM 中加载的，而非 L2 cache, 我们让不同的线程从不同的位置加载数据。使用 cuda event 来记录 kernel 的耗时。
我们测算 L2 cache 的吞吐，让每个 thread 从相同的位置加载 512 KB 的数据，以确保是从 L2 cache 中加载数据而非 DRAM。在 L2 cache 和 DRAM 的吞吐测试中，为了强制 LDG 绕开 L1 cache，我们使用 PTX ld 指令的 .ca flag 以达到目的[19]。
我们通过发射一个只有 32 thread 的 kernel，让 kernel 连续发射上千条 LDG 指令来测算 LDG 指令的 CPI。我们通过在长 LDG 指令序列的开始和结束的位置记录时钟(类似于在 CUDA C++ 中调用 clock())来计算读取数据所需的时钟。为了消除指令 cache 未命中的影响，我们重构了循环中的指令序列，让它足够小(128 条指令)刚好能够适合 L0 指令 cache。
需要指出的是，CPI 的基准测试只能在 SASS 级别测试。如果是使用 CUDA C++ 或者 PTX 编译器将优化掉长 LDG 指令序列，导致其失去作用。虽然我们可以使用 sink 来防止编译器优化掉 LDG 序列，但是指令序列会变得混乱，导致结果不准确。
b) 基准测试结果: 在表 II 中我们以 GB/s 为单位列举了测试出的吞吐。在表 III 中我们以 CPI 为指标列举了不同序列宽度的 LDG 指令吞吐。我们对关于 DRAM 和 L2 cache 的一些重要结果总结如下:

T4 GPU 相比于 RTX2070 有更高的 FLOPs，但是 DRAN 带宽却更小。正如我们将要在 VI-A 和 VII 部分描述的一样，T4 上基于 Tensor Core 的 HGEMM 是受 DRAM 的带宽限制的。
测试显示在 RTX2070 上 DRAM 的带宽可以达到峰值的 85%，在 T4 上可以达到峰值的 75%。
从 SM 的探测结果看，当数据存储在 L2 cache 的时候，LDG.32 和 LDG.64 拥有相同的吞吐(32/CPI_{LDG.32} = 64/CPI_{LDG.64}). LDG.128 的吞吐比 32 和 64 高 5.1%.
B. Shared Memory

a) 方法: 我们测试 shared memory 的 CPI 使用与测试 LDG 的 CPI 相同的方法。也是通过循环发射访问 shared memory 的指令，然后记录所需要的时钟周期。我们通过设置合适的偏移让 shared memory 的访问都是没有 bank 冲突的。 b) 基准测试结果： 我们在表 IV 中展示了 shared memory 指令的 CPI，在表 V 中展示了相对应的指令吞吐(bytes/cycle)。CPI 和吞吐在 RTX2070 和 T4 上是大致相等的。
如下是我们发现的一些重要的 shared memory 的行为:

LDS.128 和 LDS.64 拥有相同的吞吐，都比 LDS.32 高 5.5%。
LDS.64 和 LDS.128 可以达到理论的峰值带宽。对比文献[11]中报告的 58.8 bytes/cycle。本文的工作展示了 SASS 级别的基准测试的强大能力。
STS.128 相比于 STS.64 有 20% 的吞吐优势，相比于 STS.32 有 62.4% 的优势。意味着使用窄类型的指令将付出更昂贵的代价。
V. 矩阵乘法的优化方法

当前 GEMM 优化的发展状况是，为了最大化的复用数据，在文献[9][13][14]中都使用了两级 blocking 的策略。在这一节，我们首先分析 blocking size 是如何影响 HGEMM 的性能的以及如何选择合适的 blocking size。除了 blocking size 之外，指令顺序，shared memory 中的数据排布都会影响 HGEMM 的性能。为此我们深入的探讨如何根据 CPI 值来调度指令以及如何设计 shared memory 中的数据排布。我们评估了这些手段对于性能优化的影响，并且表明这些手段可以显著提升吞吐。我们的结果是在 RTX 2070 上获得的，我们使用 cuda event 来计算运行时间。吞吐数据是通过测量超过 10 次的数据取平均得到的。

A. Blocking size 分析

1) HGEMM 分块: 算法 1 中展示了两级分块算法是如何运行的。为了更简洁，我们丢掉了一些细节，比如数据预取，同步和索引计算等等。我们参考了 CUTLASS 中两级 blocking 的方法。


2) 线程块的尺寸: 线程块位于两级 blocking 的第一层级，在文献中也被称为 shared memory blocking。我们把 m \times n \times k 的矩阵乘划分为多个块。一个线程块计算一个 b_m \times b_n 的分块。因此我们将发射 \lfloor (m + b_m - 1) / b_m \rfloor \times \lfloor (n + b_n - 1)/b_n \rfloor 个线程块。
每个线程块计算 2 \times b_m \times b_n \times b_k 次乘加，需要加载的数据量为 (b_m + b_n)b_k 个半精度元素。因此计算密度为 \frac{2b_mb_nb_k}{2(b_m+b_n)b_k} = \frac{b_mb_n}{b_m+b_n}. 分块的尺寸将影响计算强度。对于半精度典型的 block size (b_m \times b_n) 是 (256 x 256), (256 x 128), (128 x 128), (128 x 64), (64 x 64).
大的 block 可以提升计算密度，但是会降低 GPU 占用率(笔者注，大 block 会导致能够同时 launch 的 thread block 会减少)，会损失性能。为了找到合适的 block size，我们使用前文探测到的带宽数据(Table II)为 RTX 2070 和 T4 GPU 画出 Roofline 模型如 Fig3.



为了对比，我们也在 Fig3 中画出了 FP16 计算单元的 Roofline 线条。当使用 FP16 单元的时候，(128 x 128) 分块足够保持计算单元繁忙。但是对于 Tensor Core，(128 x 128) 会让 DRAM 带宽成为新的瓶颈。
虽然可以通过增加 block size 来增加计算密度，但是我们无法使用大的 block size，比如 (512 x 256)，此时寄存器数量会成为瓶颈。在 Turing 架构 GPU 上，每个 SM 拥有 64 K 的 32-bit 寄存器。512 x 256 的分块占用了所有寄存器，没有给其他数据留下寄存器。因此 (512 x 256) 的分块是无法奏效的。其他的大 block size 比如 (256 x 320) 或许可能奏效，但是我们更希望 block size 是 2 的次幂，因为这对于 GPU 编程是天然的，很容易推论(笔者注，GPU 的线程资源，寄存器资源等等都是以 2 的次幂配置的)。
在本文中我们聚焦于 (256 x 256) 的分块上，对于第三维度 b_k 的大小，受限于 shared memory 的 size。Turing GPU 增加每个 SM 的 shared memory 到 64 KB, 因此 b_k 必须比 64KB / (256 + 256) / 2 = 64。 如果选择 64，那么 64KB 的shared memory将被预取的矩阵 AB 的块数据完全占用，没有给避免 shared memory 的 banck 冲突留下 padding 的空间，这对于性能的影响是非常严重的。最终我们选择 32 作为 b_k 的尺寸。
3. Warp 级别的分块尺寸: 第二级别的分块是在 warp 级别的，在文献中也被叫做寄存器分块。这一层级的问题是如何划分(256, 256)的分块到合适数量的 warp。目前已经存在的一些工作[9][13]是通过试错法来确定合适尺寸的 warp 层级的分块，也有一些是通过 warp 层级的 Roofline 模型按照计算密度来计算 warp 层级分块尺寸。我们提出了一种新的，简单有效的方法在实现之前确定 warp 层级分块尺寸。我们的方法和文献的方法有一下几方面的区别:

我们使用 CPI 作为标准来指导我们选择合适的 warp 层级的分块尺寸。CPI 这一指标在之前的工作中很大程度上被忽视了。
相比于试错法，我们在实现之前给出合适的 warp 层级分块尺寸，大大提升了工程实践的效率。

在 cuBLAS 10.1 中该层级的分块是 (64 x 64), 就是使用 (64 x 64) 的 block size(笔者注，第一层级的分块叫做 thread block 层级，是指每个 thread block 计算多大的数据分块，是对整个矩阵划分 tile；第二层级分块叫做 warp 层级，因为 Tensor Core 编程是 warp 级别的编程，因此这一层级分块是划分一个 warp 计算 thread block 中多大的数据块，也决定了，一个 thread block 中能有多少 warp)。我们将展示 (64 x 64) 不是最好的分块策略，该方式将促使 memory pipe 成为新的瓶颈。而使用(128 x 64) 的分块将解决该问题。
每个 thread block 使用 HMMA 指令完成一个矩阵分块，所需要的时钟周期数量为:
\frac{2b_mb_nb_k}{2 \times 16 \times 8 \times 8 \times 4} \times CPI_{HMMA.1688.F16} （3）
其中，2 x 16 x 8 x 8 是一条 HMMA 指令完成的算量，4 是 每个 SM 拥有 4 个 处理单元。(笔者注，总算量除以 SM 单周期算量峰值便是所需指令数量，乘以 CPI，就得到完成这些算量所需的周期数)。
LDG，STS 和 LDS 指令都占用内存 I/O 的流水。从全局内存加载 (b_m + b_n) \times b_k 数量的数据存储到 shared memory 中(使用 128-bit 指令)需要花费的周期数为:
\frac{2(b_m + b_n)b_k}{32 \times 16} \times (CPI_{LDG.128} + CPI_{STS.128}) (4)
其中，2 是 半精度浮点数的字节数为 2 bytes，16 是 128-bit 是 16 bytes，32 是一个 warp 有 32 个 thread。
从 shared memory 以 32 bit 指令加载数据所需的周期数为：
\frac{b_mb_n}{w_mw_n} \times (\frac{w_m}{8} + \frac{w_n}{8}) \times \frac{b_k}{w_k} \times CPI_{LDS.32} (5)
改变两级分块的大小，并不改变总的计算量，仅仅改变传输的数据量。因此目标是保持 Tensor Core 繁忙，同时保证内存 I/O 流水不会成为性能瓶颈。换句话说，我们需要保证在每次循环中，内存 I/O 所需的指令周期数要少于 Tensor Core 处理数据所需的指令周期数。根据前面的分析和我们在 Table I,III,IV 中收集的 CPI 的值，在 Table VI 中我们列举出对应的 Block size 执行 HMMA 和 内存指令所需的周期数。



根据 Table VI 中的数据，当一级分块大小为 (128 x 128) 的时候，HGEMM 的瓶颈是 Memory IO。当一级分块为 (256 x 128) 的时候，如果 thread block 是 (64 x 64) 瓶颈依然是 Memory IO，如果 thread block 是 (128 x 64) 瓶颈就变为 HMMA 了。最优的选择是一级分块为 (256 x 256), 二级分块为 (128 x 64)。HMMA 所需要的时钟周期数大于内存 IO 所需要的周期数。这组配置是的 HGEMM 的 L2 cache miss 有很强的鲁棒性，并留下了足够的空间用于掩盖延迟。
我们不能设置 warp 级别分块为 (128 x 128), 因为每个线程仅可以使用 256 个寄存器，(128 x 128) 的分块将占用一个 warp 内的所有寄存器。
4)与 cuBLAS 对比: Tab VII 对比了我们的实现和 cuBLAS 10.1 的实现。cuBLAS 使用较小的一级分块(128 x 128),获得的收益是可以有 2个 CTAs(Compute thread arrays) 驻留到 SM 上，这样可以让 warp 以异步的方式执行。异步行为可以让 warp 调度器拥有更多的机会切换 warp 以增加吞吐。这也符合 GPGPU 编程的常规思路。
然而我们在本节开始的分析表明，当 block 的尺寸配置为 128 x 128 x 64 的时候，DRAM 和 shared memory 都将成为性能的瓶颈。值得注意的是 cuBLAS 的 HGEMM 仅仅用了 32 KB 的 shared memory 内存。这表明 cuBLAS 没有使用 padding 来避免 bank conflict。我们认为这样一种简洁的数据布局值得研究。
总结一下，我们从以上分析中学到的一些重要经验如下:

我们应该选择较大的线程分块尺寸。另外 HGEMM 的性能瓶颈是 DRAM 和 L2 cache 的带宽。在线程分块层级，理想的分块是 (256 x 256)。(256 x 256) 是我们能够选择的最大的线程分块，但是 DRAM 和 L2 cache 带宽依然是性能瓶颈。
我们也需要让 warp 级别的分块尽量大，否则 shared memory 的带宽将成为新的瓶颈。(128 x 64) 是我们选择的最优分块大小。
B. 数据预取

与前人的工作一样，我们通过数据预取(有些文献中叫做软流水(software pipelining))来隐藏从 global memory 和 shared memory 中加载数据的延迟，例如我们再当前迭代中加载下一迭代所需要的数据。这种做法会增加寄存器压力，需要消耗更多的寄存器来存储预先加载的数据。在我们的实现中，我们至少有 (64 + 32) x 8 = 768 (笔者注: Tensor Core 计算 + 从 share mem 中加载数据) 个时钟周期来隐藏数据 LDG 的延迟。

C. 指令调度

一旦 block 的尺寸确定，HMMA 指令和内存 IO 指令的数量是固定的。那么接下来的问题就是，如何以合适的顺序排布这些指令(也叫做指令调度问题)。具体来说，我们关心内存 IO 指令如何插入 HMMA 指令序列。目前为止还没有权威性的方法可以确定访存指令之间的空隙。在文献 [14][15]中作者提出一种试错法的策略来寻找访存指令间的合适的空隙。我们则给出一种基于 CPI 的简单有效的原则来确认连续访存指令间空隙的合适大小。
使用 STS.128 指令为例。在 cuBLAS 10.1 的 HGEMM 中，两个 HMMA 指令使用交错连续的 STS.128 指令加载数据。我们认为这样的间隙是不够的。HMMA(#HMMA) 所需要的插入 STS.128 的最少指令数量需要满足:

HMMA \times CPI \geq 4 \times CPI_{STS.128} \quad\quad(6)

其中，4 代表每个 SM 包含 4 个 processing blocks。这一公式的想法来源于，用于计算消耗的时钟周期不应该小于用于 IO 的时钟周期。基于我们计算的 CPI 值，我们至少需要 5 个 HMMA 指令来用于插入 STS.128 指令。
Fig4 展示了使用两条 HMMA 指令插入 STS.128 指令 (STS2) 和使用 5 条 HMMA 指令插入 STS.128 指令 (STS5)在吞吐上的各自的数据。平均情况 STS5 是 STS2 的 1.13 倍，最好的情况达到 1.26 倍。





D. Shared Memory 数据排布

基于在第 VI-A 段落中的讨论，线程块的大小选择 (256x256x32). 我们需要把 A_{256 \times 32}(行主序), B_{32 \times 256}(列主序)存储到 shared memory 中。最简单的方式是在 shared memory 中申请两个 256 x 32 的数组。在 CUDA C++ 中可以写作 A[256][32] 和 B[256]32。在 Fig5 中我们展示了这种简单的存储方式和我们做 padding 的存储方式在吞吐上的差异。如图所示，简单的数据存储方式导致 HGEMM 比我们优化的存储方式慢一倍。


我们提出的数据排布方式是在每隔一行 pad 8 个 半精度元素。例如 A[row][col] 的偏移计算应该是这样的，
row x 32 + row % 2 x 8 + col。这种数据排布方式可以让读写 shared memory 都不会产生 banck 冲突。

VII. 评估

在这一部分，我们将对我们优化的 HGEMM 和 cuBLAS 的 HGEMM 性能进行对比。这一部分所有的结果，测试环境都是 Ubuntu18.04，CUDA 10.1 和 cuBLAS 10.1 。所有的输入和输出都是半精度数据。输入和输出数据是存储在 GPU 内存上的，这也是很多深度学习应用的场景。内存申请的开销是不包含在计时里面的。
我们在两块 NVIDIA 的 Turing GPU 上对比 cuBLAS 和我们自己优化的 HGEMM 性能，两款 GPU 叫做 RTX 2070 和 T4 。在 T4 上我们分别设置内存时钟频率和 GPU 时钟频率为 5001 GHz 和 1590 GHz，另外，RTX 2070 不支持设置时钟频率。我们使用 cuda 的事件机制计算每个 kernel 的耗时。吞吐的数据是通过平均 10次以上测量数据得到的。
考虑的矩阵乘法问题是 C=AB, 其中 A 是行主序，B 是列主序的数据排布。为了让 cuBLAS 使用 Tensor Core 计算，我们设置 cuBLAS 的计算模式为 CUBLAS_TENSOR_OP_MATH 然后调用 cublasGEMMEx() 函数。因为在深度学习应用中一般矩阵都比较大，而且 Tensor core 的目标也是计算大矩阵。因此我们测量采用的矩阵尺寸是从 1024 到 16384，步长为 256. 我们从方阵开始测量。同时，我们通过改变其维度来测量长方形矩阵。我们测试的形状包括:[2WxWxW]、[Wx2WxW]、[WxWx2W]、[4WxWxW]、[Wx4WxW] 和 [WxWx4W]。

A. 方阵的性能比较

a) RTX 2070: Fig6 展示了优化实现和 cuBLAS 实现的性能数据。当矩阵尺寸较小的时候(W<4096)，我们的实现达到了和 cuBLAS 相当的性能。在这种情况下性能容易受到内核启动开销和 thread block 数量较少的影响。随着矩阵尺寸的增加，我们优化实现的 HGEMM 性能也稳步提升直至达到峰值性能。我们实现的最优吞吐是 60.37 TFLOPs，比硬件峰值 (59.7 TFLOPs) 要高。这可能来是计时的噪声和计算吞吐时取整引入的误差的影响导致的。相比之下，随着矩阵尺寸超过 4096，cuBLAS 的性能有所下降。我们观察到，当尺寸超过 12032 的时候，cuBLAS 性能严重下降。我们怀疑这是由于 L2 cache 的分块策略导致 cuBLAS 在这一尺寸上失败了。cuBLAS 的最大吞吐是 52.75 TFLOPs，当矩阵尺寸为 4096 的时候。最大的加速是在矩阵尺寸为 16128 的时候，我们加速了 2.7 倍。相对于 cuBLAS 我们的平均加速为 1.55 倍。



b) T4: Fig7 展示了我们优化的 HGEMM 和 cuBLAS 的 HGEMM 的 TFLOPs 的对比。与 RTX 2070 相似，在矩阵尺寸小于等于 4096 的时候我们的实现与 cuBLAS 相当。随着矩阵尺寸的增加，我们的实现性能稳步提升直到接近 50 TFLOPs。我们最优性能是 49.71 TFLOPs，达到硬件峰值 (65 TFLOPs) 的 76%。RTX 2070 相较于 T4 最大的提升来自于 DRAM 带宽的差异。相比之下，当尺寸大于 4096 的时候，cuBLAS 性能有所下降。它的最优性能为 45.43 TFLOPs，当矩阵尺寸为 2560 的时候。相对于 cuBLAS 最大的加速是在 W=13312 的时候，加速比达到 1.7。在 T4 上相对于 cuBLAS 我们的平均加速为 1.53 倍。
我们也观察到我们的实现在 W 大于 12800 以后，吞吐有下降趋势。我们认为这是由于较低的 L2 cache 命中率和 T4 的 DRAM 带宽较低影响了我们的实现。





B. 长方形矩阵的性能比较

a) RTX 2070: Fig.8 展示了我们的 HGEMM 和 cuBLAS 的 HGEMM 在 RTX 2070 上的性能对比。和方阵的对比有相同的趋势。我们的实现始终能达到接近峰值的吞吐。当形状为 [WxWx4W], W = 14848 的时候，相较于 cuBLAS 我们达到最大加速比为 3.23。长方形矩阵在 RTX 2070 上的平均加速达到 1.77 倍。



b) T4: Fig.9 展示了我们实现的 HGEMM 和 cuBLAS的 HGEMM 在 T4 上的性能对比。和方阵拥有相同的趋势，我们的实现始终可以达到接近峰值的水平。在形状为 [WxWx4W], W = 15360 的时候，相较于 cuBLAS 我们达到最大加速比为 2.17。长方形矩阵在 T4 上的平均加速达到 1.45 倍。





C. 总结

相较于 cuBLAS 10.1 所有的配置，在两款 GPU 设备上的平均加速达到 1.61 倍。虽然 T4 的理论峰值跟高，但是无论是 cuBLAS 10.1 还是我们的实现，在 RTX 2070 上的吞吐均优于 T4。RTX 2070 和 T4 的主要区别是 RTX 2070 的 DRAM 带宽(380GB/s) 比 T4 的带宽(240 GB/s) 高 58%。因此我们认为这是一个强有力的证据，在 T4 上 HGEMM 的性能瓶颈是 DRAM 带宽。







欢迎关注公众号: HPC 漫谈
