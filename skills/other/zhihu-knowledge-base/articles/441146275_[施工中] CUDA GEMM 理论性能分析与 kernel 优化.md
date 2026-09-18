# [施工中] CUDA GEMM 理论性能分析与 kernel 优化

**作者**: 李少侠

**原文链接**: https://zhuanlan.zhihu.com/p/441146275

---

​
目录
收起
0. Benchmark
1. CUDA GEMM 常规实现方案与理论性能分析
1.1 基于 GEMM 定义的朴素实现
1.2 Thread Block Tile: 利用 Shared Memory 减少重复访存
1.3 Warp Tile 与 Thread Tile: 利用寄存器消除 Shared Memory 瓶颈
1.4 Double Buffer: 让 GEMM 流水并行起来
1.5 小结
2. Thread Block Tile 尺寸选择
2.1 等效内存带宽和 L2 Cache 命中率的估算
2.2 L2 Cache 命中率与矩阵规模的关系
2.3 分块大小与 IPC 的关系
2.4 '尾部效应' 的影响
2.5 SplitK

GEMM（General Matrix Multiplication，通用矩阵乘法）是并行计算中经典的计算密集型应用，也是入门计算密集型 CUDA 程序优化非常好的例子，本文从 CUDA GEMM 实现方案的理论性能分析和 kernel 代码优化技巧两个方面分享如何将 GEMM 性能优化到接近设备理论算力。

本文主要有以下三部分：

GEMM 分块方法与性能间的理论关系，即如何根据输入矩阵规模，硬件设备 spec 信息 (内存/Cache带宽，指令 IPC 等) 推导出最佳分块大小；
以 FP32 GEMM (SGEMM) 为例，介绍代码写法层面的优化技巧，以及引导 nvcc 编译器生成更高效的指令的方法；
在 2. 的基础上，针对 Ampere GPU 的一些优化方法，使 kernel 能更充分地发挥 GA100, GA102+(如GA102/104/106) GPU 的 FFMA 性能；

本文中所有的 kernel 代码，编译测试脚本，理论分析中用到的 CUDA microbenchmark 代码都可在下面 github repo (后文简称 "repo") 的 cuda 目录中获得，另外代码为方便大家阅读理解，也添加了很详细的注释。

0. Benchmark

为了提高大家读完本文的动力，先放一下 repo 代码的 benchmark：

测试环境：

Driver 495.44, CUDA 11.5.1

测试结果：

0 号卡 Titan V，FFMA 理论算力 5120 FFMA core * 1455MHz * 2 = 14.90 TFlops :

1 号卡公版频率 RTX3090，理论算力 10496 FFMA core * 1695MHz * 2 = 35.58 TFlops:

公平起见，三者使用相同的分块大小(针对大矩阵的 128x128 或 256x128)，cutlass 版本2.7.0，测试方法为跑 50 轮取最快。对于 RTX3090 上 GEMM 实测性能与硬件算力有较大差距的原因，将在本文 Ampere GPU 优化部分结合 GA102 架构特点再来介绍。关于计算密集型或访存密集型 CUDA 程序的测试方法，以及不同测试方法间结果差异的原因将在本文最后介绍。

因为家里这台服务器只能插两张双槽宽度的卡，拆卸卡也麻烦，没在其他架构 GPU 上测试性能，欢迎手上有 Maxwell/Pascal/Turing GPU 的同学分享测试数据，根据以往经验，repo 中 sgemm.cu 的通用实现在这些 GPU 上就能有不错的性能。

1. CUDA GEMM 常规实现方案与理论性能分析

目前已有很多资料介绍 CUDA/OpenCL GEMM 实现和优化方法，因此对于常规实现方案，本节只做简单介绍。

1.1 基于 GEMM 定义的朴素实现

矩阵乘法定义如下：

输入：矩阵A(M行K列），矩阵B(K行N列)
输出：矩阵C(M行N列)

for i from 0 to M-1:
    for j from 0 to N-1:
        C[i][j] = 0;
        for p from 0 to K-1:
            C[i][j] += A[i][p] * B[p][j]

对于 C 矩阵的每一个元素，都要读取 A 矩阵的一行和 B 矩阵的一列来计算，那么计算完整的 C 矩阵，A B 矩阵都要重复读取多次，所以直接按定义计算效率很低。

不过这样做性能有多低呢？我们来算一下。

首先说明，很多文章在解释这种方案性能差的时候，都是以内存延迟太高作为主要理由，实际上在并行计算中，多过程流水并行是常用的设计方法，延迟只要能被其他过程覆盖就没有问题。所以对 GEMM 的性能分析应该以带宽作为衡量指标。

对于 FP32 数据，如上图所示，一个 warp 一次做 32 次 FFMA，对应 64OP，需读取 A 矩阵 1 个元素和 B 矩阵 32 个元素，共 132byte。C_{[i][j]} 通过寄存器累加，且忽略 C 矩阵写回开销，那么计算访存比为 64OP / 132byte = 0.48。虽然 dram 最小访问单位为一个 memory transaction，但考虑到 L1 cache 的存在也不会影响实际的计算访存比。

通过 repo 中提供的 l2cache_bandwidth.cu 可测得 Titan V L2 cache 带宽约 1.9TB/s，那么最乐观的结果即使 L2 cache 100% 命中，此方案的理论上限也只有 1.9T * 0.48 = 0.912 tflops，远低于 14.9 tflops 的硬件算力。

1.2 Thread Block Tile: 利用 Shared Memory 减少重复访存

利用高速存储器来减少低速存储器的访问是常用的优化手段，所以可以用 shared memory (后文简称 smem) 来减少重复的内存读取。首先把 C 矩阵等分为 M_{tile}*N_{tile}大小的分块 (后文称之为 thread block tile)，每个分块由一个 thread block 计算。之后 FFMA 计算所需的数据全部从 smem 中读取，就消除了一部分重复的 A B 矩阵内存读取。考虑到 smem 容量有限，可以在 K 维上每次读取 K_{tile}大小的分块，直到完整遍历 K 维即可得到 thread block tile 的结果。

利用 smem 优化后，对 M_{tile}*N_{tile}分块，可得：

计算量：M_{tile}\cdot N_{tile}\cdot K\cdot 2

访存量：(M_{tile}+N_{tile})\cdot K\cdot 4Byte

计算访存比：\frac{M_{tile}N_{tile}}{2(M_{tile}+N_{tile})}=\frac{1}{2(\frac{1}{N_{tile}}+\frac{1}{M_{tile}})}

假设M_{tile} N_{tile}都取 64，依上述表达式可得计算访存比为16，代入 Titan V FFMA 算力可得: 14.9tflops / 16 = 931GB/s，即对于 64*64 大小的 thread block tile，平均访存带宽超过 931GB/s 后性能瓶颈就不在内存上了。

根据 repo 中提供的 dram_bandwidth.cu 可测得 Titan V 实际内存只读带宽最高约 520 GB/s，L2 Cache 带宽 1.9TB/s，当 L2 cache 命中率为 30% 时，加权后的平均访存带宽为 520 * 0.7 + 1900 * 0.3 = 934 GB/s，超过了 931GB/s 的拐点。

位于同一行的 thread block tile，读取的 A 矩阵分块相同，位于同一列的 thread block tile 读取相同的 B 矩阵分块，这些重复读取能大大提高 L2 cache 命中率 (通常有 60% 以上的命中率)，所以通过 C 矩阵分块配合 smem 优化，可以很容易地超过内存性能拐点，消除 1.1 节朴素实现方法中的内存读取瓶颈。

另外目前的 GPU FP32 算力与存储系统带宽的比值不像 tensor core 那么悬殊，达到内存性能拐点比较容易。对于 tensor core int8/FP16 就需要一些额外的技巧提高 L2 cache 命中率(80%以上命中率)才能最大化算力，这些在以后介绍 tensor core 优化时再讲。

1.3 Warp Tile 与 Thread Tile: 利用寄存器消除 Shared Memory 瓶颈

1.3.1 Shared Memory 访存瓶颈

对于一个 thread block tile，最直接的计算方法是把 M_{tile}\cdot N_{tile} 个元素平均分配到 thread block 的每个线程上去，每个线程分得 M_{frag}\cdot N_{frag} 个点 (thread tile)。之后每个线程再对各自 thread tile 内的点按照 GEMM 定义计算：

for i from 0 to M_frag-1:
    for j from 0 to N_frag-1:
        for p from 0 to K_tile-1:
            C[i][j] += A_tile[i][p] * B_tile[p][j];

这种实现方法性能如何呢？可以看到上述过程包括两个步骤：

把 A_tile[...] 和 B_tile[...] 从 smem 读到寄存器中；
对寄存器数据做 FFMA 计算。

对最内层的 for p from 0 to K_tile-1 循环，一个 warp 的操作过程如下图所示：

其中黄色部分表示计算 C_tile 的一个点需要读取的 smem 数据，绿色部分表示一个 warp 在每次循环迭代中读取的数据。A_tile 中绿色数据可以利用广播发送给 warp 的 32 个线程，B_tile 中绿色数据为 32 * 4byte = 128byte，没有 bank conflict。那么这一次迭代中共有 32 次 FFMA 和 256byte 的 smem 到寄存器数据传输。GV100 GPU 每个 SM 上 smem 出口带宽为 128byte/cycle，那么 32 次 FFMA 对应 2 cycle 的数据传输。GV100 每 SM 每周期可执行 64 次 FFMA，对应的 smem 数据读取需要 4 cycle，所以此方案中 smem 到寄存器的数据传输会成为瓶颈，只能发挥出每个 SM 1/4 的算力，在 Titan V 上的理论上限为 14.9tflops / 4 = 3.725tflops。

另外也只有在 GV100/GA100 等大核心上 smem 出口带宽才有 128byte/cycle，turing/GA102+等 GPU 上只有 64byte/cycle，所以在 Turing GPU 上只能发挥出 1/8 的算力，在 GA102+ GPU 上只能有 1/16 的硬件算力 (GA102+ 为每 SM 128FFMA/cycle).

需要说明的是，上述分析是基于满流水并行带宽计算的，即所说的 2 cycle / 4 cycle 数据传输并不表示发送 smem 读取请求后经过 2 cycle 或 4 cycle 就能拿到数据，实际上 smem 单次访问延迟有 20~30 cycle。另外单条 FFMA 指令在 volta 上有 4 cycle 延迟，每 SM 的 64 个 FFMA unit 也是分为 4 组做 half warp 调度的。站在带宽角度分析 GEMM 性能时，以满流水状态下的计算或访存带宽为参考依据，就有了类似每 cycle 64 次 FFMA，每 cycle 128Byte 数据传输这些指标。大家不要混淆延迟和带宽。

1.3.2 Thread Tile: 利用寄存器减少 Shared Memory 读取

那么怎么解决 smem 瓶颈呢？再回到刚刚 thread tile 计算过程的伪代码：

for i from 0 to M_frag-1:
    for j from 0 to N_frag-1:
        for p from 0 to K_tile-1:
            C[i][j] += A_tile[i][p] * B_tile[p][j];

其中按照 M-N-K 的循环嵌套顺序实际上是矩阵乘法的向量内积表示形式，A_tile 读取的位置与 i, p 有关，B_tile 读取的位置与 j, p 有关，循环嵌套之下产生了重复的 smem 读取，这也是相对 smem 计算访存比低的原因。而如果改为 K-M-N 的循环嵌套顺序，就变成了矩阵乘法的向量外积表示形式：

for p from 0 to K_tile-1:
    for i from 0 to M_frag-1:
        for j from 0 to N_frag-1:
            C[i][j] += A_tile[i][p] * B_tile[p][j];

再添加一点和存储器相关的细节：

A_frag, B_frag, C_frag: registers
A_tile, B_tile: shared memory

for p from 0 to K_tile-1:
    A_frag[M_frag] <= A_tile[0 to M_frag-1][p]
    B_frag[N_frag] <= B_tile[p][0 to N_frag-1]
    for i from 0 to M_frag-1:
        for j from 0 to N_frag-1:
            C_frag[i][j] += A_frag[i][p] * B_frag[p][j];

相应的 thread tile 处理过程变为下图形式：

可以看到计算一个 thread tile 时，参与计算的 A_tile, B_tile 中的元素只被读取了一次，单线程内消除了向量内积实现中的 smem 重复读取。

但向量外积实现方案中的 A_frag, B_frag, C_frag 需要占用大量的寄存器，假设 M_{frag}, N_{frag} 都为 8，那么 A_frag, B_frag 各需 8 个寄存器 (如果 double buffer 就是 16个)，C_frag 需要 8*8=64 个寄存器，所以此优化的本质还是用寄存器换 smem 访存，即高速存储器换低速存储器。

由于 GPU 通常 FFMA 计算单元很多 (128 或 64 每 SM)，执行访存指令的 LSU (load/store unit) 较少 (32 或 16 每 SM)，访存指令的 IPC 较低，另外 FFMA 与其他指令流水并行状态下，FFMA 要掩盖所有其他指令的延迟，所以除计算访存比外，我们还需要考虑指令调度的开销。针对 smem，即最大化 for p from 0 to K_tile-1 循环中 FFMA 指令与 smem 访存指令 (LDS) 的比值。上述方案中 LDS 指令数量与 M_{frag},N_{frag} 之和成正比 (比例系数 \alpha ，与 LDS 访存宽度有关，如 LDS.32, LDS.64, LDS.128 等)，FFMA 指令数为 M_{frag}\cdot N_{frag} ，那么有：

\begin{equation} \begin{aligned} \frac{FFMA}{LDS}&=\frac{1}{\alpha}\cdot \frac{M_{frag}\cdot N_{frag}}{M_{frag}+N_{frag}}\\\\ &=\frac{1}{\alpha}\cdot \frac{\frac{1}{4}\cdot ((M_{frag}+N_{frag})^{2}-(M_{frag}-N_{frag})^{2})}{M_{frag}+N_{frag}} \end{aligned} \end{equation}

可以得到结论：

M_{frag}, N_{frag}越大，FFMA 与 LDS 指令数比值越高；
M_{frag},N_{frag}之和为常数时， M_{frag},N_{frag}之差越小，FFMA 与 LDS 指令数比值越高；
若 FFMA 与 LDS 指令数比值为常数，M_{frag},N_{frag}之差越小， M_{frag},N_{frag}之和越小；

换句话说：

当设备 FFMA 指令与 LDS 指令 IPC 越悬殊时，需要更大的 M_{frag},N_{frag}实现 FFMA 掩盖 LDS 指令延迟，随之也会消耗更多的寄存器；
当 A_frag 和 B_frag 占用寄存器总量固定时， M_{frag},N_{frag}之差越小，LDS 指令占比越低，越容易被 FFMA 指令延迟掩盖；
若 LDS 指令占比固定，则M_{frag},N_{frag}之差越小，A_frag 和 B_frag 占用寄存器总量越少；

sm_35 及以后的设备单线程最多可用 255 个通用寄存器，thread tile 取 16*16 寄存器不够用，取 8*4 分块 FFMA 总延迟为 32cycle + FFMA_latency，smem 本身有 20~30 cycle 的延迟，global memory 到 smem 的读取以及各种访存地址计算也需要指令，所以 thread tile 取 8*4 不足以用 FFMA 掩盖非 FFMA 指令的延迟 (8*4 以下如 4*4 就更不够了)，所以 SGEMM 中 thread tile 通常取 8*8, 8*16 等数值。注意本段我们分析的是单线程内的延迟覆盖问题，所以用的延迟作为计算指标。

1.3.3 Warp Tile: 最大化相对 Shared Memory 的计算访存比

GPU 硬件上实际调度的单位为 warp，GPU 上的许多开销都与整个 warp 的行为密切相关，所以对于向量外积方案除了单线程内的延迟覆盖问题，我们还要考虑整个 warp 上的计算访存比。

warp tile: 4*8 thread

一个 warp 由 warp_{x}*warp_{y} 个线程组成，可以是 1*32, 2*16 或 4*8，我们把这些线程对应的 thread tile 拼在一起的区域称为 warp tile，尺寸为 M_{warp}*N_{warp} ，如上图所示。

从带宽角度分析，可以看到 smem 访存量与 M_{warp}, N_{warp} 成正比，FFMA 次数等于 M_{warp}\cdot N_{warp} ，显然 warp 摆放为 4*8 thread 时计算访存比最高，1*32 thread 最低。以最糟糕的 1*32 thread 为例，当 thread tile 为 8*8 时，A 矩阵需读取 1*8*4byte = 32byte, Titan V 上至少要 4cycle (代码优化部分再解释为什么不是 1cycle 或者 2cycle，与 broadcast 机制有关)，B 矩阵需读取 32*8*4byte = 1024byte，需要 8cycle。所以即使最糟糕的情况下，向量外积实现方案也能做到 8*8*32=2048 次 FFMA 对应 12cycle 的 smem 读取，平均 1cycle 数据读取对应 170 次 FFMA。带宽分析上是完全满足 Titan V 最高算力需求的 (每 SM 64FFMA/cycle)。

如果是 GP102+, GA102+ 等每 SM 128FFMA/cycle 且 smem 只有 64byte/cycle 的 GPU，1*32 thread 会变为 A 矩阵读取 4cycle, B 矩阵读取 16cycle，平均 8*8*32FFMA / 20cycle = 102 FFMA/cycle，低于设备理论峰值的 128FFMA/cycle。如果改为 4*8 thread 则变成 A,B 矩阵读取均为 4cycle (道理同 1*32 分析中的 Titan V A 矩阵读取 4cycle，后面解释)，平均 8*8*32FFMA / 8cycle = 256FFMA/cycle，可以满足设备理论 FFMA 上限需求，所以一般 warp 都会配置为 4*8 或 8*4 thread。

但文章开头我们说过，以带宽作为参考而忽略高延迟的前提，是延迟能被其他过程覆盖，如无法覆盖则跑不满理论带宽，那么带宽分析的结果也就没有了参考价值。例如此处单个 warp 内一定要等对应的数据从 smem 读到寄存器后才能做 FFMA，如果数据读取总延迟大于 FFMA 总延迟，会导致 FFMA 等待数据读取，那么实际的 FFMA 带宽也就达不到 1cycle 对应 170 次 FFMA 了，除非有较高的 occupancy，有足够多的 warp 填充延迟。

从延迟角度分析，再回到 1*32 thread 的情况，A 矩阵读取延迟为 4cycle + smem_latency，B 矩阵读取延迟为 8cycle + smem_latency，A,B 矩阵读取总延迟为 12cycle + smem_latency，FFMA 总延迟为 8*8cycle + FFMA_latency。考虑到除 smem 读取外还有 global memory 读取，地址计算，循环体的比较/跳转指令，另外在很多 smem 吞吐为 64byte/cycle 的设备上总延迟更高，1*32 thread 在 occupancy 较低时很难做到 FFMA 覆盖其他延迟，所以从延迟的角度，也是选取 4*8 或 8*4 thread 的 warp 更好。

1.4 Double Buffer: 让 GEMM 流水并行起来

Nvidia GPU 上掩盖延迟的方式主要有两种：warp 并行和单 warp 内的指令级并行 (ILP, Instruction Level Parallelism)。warp 并行依赖 occupancy，有足够多的 warp 可调度时，一个 warp 如果因为某些原因无法继续发射指令 (如 barrier, execution dependency 等)，可以发射其他 warp 的指令来填满硬件资源。ILP 则主要靠消除指令发射阻塞，使单个 warp 内的指令序列足够填满硬件资源。

从 1.3.2 和 1.3.3 节的分析中可以看出，要通过向量外积来消除 smem 访存瓶颈，thread tile 至少要 8*8 或更大，那么 A_frag, B_frag, C_frag 至少消耗 8+8+64=80 个寄存器，此外还有从 global memory 读取的中转寄存器，global/shared memory 读写指针，thread block tile 循环变量等等，8*8 thread tile 每线程通常要用到 120~128 个寄存器，在 GV100 GPU 上每 SM 只能有 (64*1024)/(128*32) = 16 个可调度 warp，occupancy 只有 25%。GV100 每 SM 有 4 个调度器，若每个调度器只有 4 个可调度 warp，当指令平均间隔超过 4cycle 后就无法靠 warp 调度掩盖延迟了。考虑到 GEMM 中涉及 smem 读写的过程需要同步 thread block，进一步限制了 warp 调度空间，所以很难靠 warp 并行掩盖延迟。

那么只能想办法提高单 warp 内指令级并行度了。依 1.3 节的 GEMM 实现方案，完整流程如下图所示：

图中黑色字体表示方框的含义和所处的存储器，红色字体分别表示执行的指令 (如 global memory 读取指令 LDG, smem 写入指令 STS)、执行指令的部件、访存指令涉及的存储器。

可以看出 GEMM 实际上由下图所示相互依赖的四步串联而成，每个步骤使用不同的存储器和指令执行部件：

那么很容易想到可以通过 double buffer 和预取的方式实现多个步骤的流水并行：

将用于存储 thread block tile 的 smem 分配两份 (smem[0], smem[1])，存储 A_frag B_frag 的寄存器也分配两份 (reg[0], reg[1])，就能消除了几个步骤的前后依赖，实现 thread block tile 读取，fragment 读取，FFMA 计算之间的流水并行，也减少了一次 thread block 同步。由于 global memory 和 smem 巨大的带宽和延迟差距，实际上save tile[*] to smem[*] 相对 load tile[*] 占比非常小，实现这两个步骤的流水并行会大大增加代码复杂度导致负优化，所以直接串联就好。

1.5 小结

GEMM 实现方案与 GPU 硬件特性密切相关，本节结合 GPU 上不同层次的并行计算部件，描述了多层分块并行的实现策略，并对各种实现方法做了理论性能定量分析：

Thread block tile 配合 smem 解决了内存带宽瓶颈；
向量外积实现方法解决了 smem 访问带宽瓶颈；
Warp tile 和 thread tile 实现了 FFMA 对其他指令的延迟覆盖，并最大化相对 smem 的计算访存比，对于 SGEMM，thread tile 通常取 8*8 或 8*16，warp 取 4*8 或 8*4 thread；
Tile 读取-Fragment读取-FFMA 三级软件流水设计，在向量外积实现使寄存器消耗量巨大导致 occupancy 较低的情况下，通过 warp 内指令并行实现了硬件资源的充分利用。

上述方案伪代码可表示为下面形式：

GEMM: M, N, K

Shared_Memory:  A_tile[2], B_tile[2]
Register:       A_frag[2], B_frag[2], C_frag
Register:       A_ldg_buffer, B_ldg_buffer

// load 1'st tile to shared memory
load_tile(A_ldg_buffer)
load_tile(B_ldg_buffer)
A_tile[0].store_tile(A_ldg_buffer)
B_tile[0].store_tile(B_ldg_buffer)

// double buffer index
tile_load_idx = 0
tile_store_idx = 1

C_frag = {0, 0, ..., 0}

// K-loop
for (k_iter = K/K_tile - 1; k_iter > 0; --k_iter) {
    for (i = 0; i < K_tile; ++i) {
        // store tile to shared memory
        if (i == K_tile-1) {
            A_tile[tile_store_idx].store_tile(A_ldg_buffer)
            B_tile[tile_store_idx].store_tile(B_ldg_buffer)
            tile_store_idx ^= 1
            tile_load_idx ^= 1
        }

        // load next fragment to register
        A_frag[(i+1) % 2].load_fragment(A_tile[tile_load_idx][(i+1) % K_tile])
        B_frag[(i+1) % 2].load_fragment(B_tile[tile_load_idx][(i+1) % K_tile])

        // load tile from global memory
        if (i == 0) {
            load_tile(A_ldg_buffer)
            load_tile(B_ldg_buffer)
        }

        ffma(C_frag, A_frag[i % 2], B_frag[i % 2])
    }
}

// FFMA for the last tile
for (i = 0; i < K_tile; ++i) {
    if (i < K_tile-1) {
        // load next fragment to register
        A_frag[(i+1) % 2].load_fragment(A_tile[tile_load_idx][(i+1) % K_tile])
        B_frag[(i+1) % 2].load_fragment(B_tile[tile_load_idx][(i+1) % K_tile])
    }

    ffma(C_frag, A_frag[i % 2], B_frag[i % 2])
}

// store C_tile to global memory
C_frag.store(...)


K_tile 为常数，for(i =0; i < K_tile;++i)循环体可以被编译器展开，所以代码实际只产生 for(k_iter = K/K_tile -1; k_iter >0;--k_iter) 这一个循环，我们将这个循环称为 K-loop (也有的称之为 main loop)，K-loop 是 GEMM 性能热点，也是优化的主要对象。

2. Thread Block Tile 尺寸选择

在前面基本实现方案的介绍中，我们对各种条件下的理论性能做了简要分析。可以看出，在条件允许的情况下，无论 thread block, warp, 还是 thread 分块通常尺寸越大越容易优化到较高性能。但考虑到矩阵规模是多变的，不可能只用一种分块方案在各种输入规模下都有较好的性能。对于给定的矩阵规模 (M, N, K)，本章我们将分析最佳 thread block tile 尺寸的选择。

为什么不讨论 thread, warp tile 的尺寸选择呢？在前面的分析中可以看出，thread / warp tile 的选择与 SM 上的资源关系密切，如 FFMA, LDS 指令 IPC, smem 带宽，FFMA / LDS 延迟等等。另外对于 SGEMM，warp tile (如 32*64, 64*64) 已经是一个很小的分割单位，所以 thread / warp tile 在一种架构上通常取固定值，通过改变 thread block 的大小和 warp 的摆放来调整 thread block tile 适应输入矩阵规模就足够了 (SplitK, SliceK 单独讨论)，除非遇到 M 或 N 小于 32 或 64 这种极度扁长的矩阵 (更像是矩阵向量乘)。注意，我们所说的 '一种架构' 是指单个 SM 结构完全一致的核心，有些架构代号相同但 SM 结构有差异的 GPU，如 GK210 和 GK110+，GP100 和 GP102+，GA100 和 GA102+还是要视为不同架构进行优化的。

2.1 等效内存带宽和 L2 Cache 命中率的估算

在前面的分析中，thread block tile 主要用于消除内存读取瓶颈。对于大矩阵，需要分块增大到计算访存比超过硬件性能比例才有机会跑满硬件算力，即满足：

sgemm_{fma\_ldg\_ratio}=\frac{1}{2\cdot(\frac{1}{N_{tile}}+\frac{1}{M_{tile}})}\geq\frac{P_{fma}}{P_{ldg}}

其中 P_{fma} 表示 FFMA 算力， P_{ldg} 表示内存与 L2 cache 加权之后的平均访存带宽，所以知道 L2 cache 命中率才能算出跑满硬件算力的最小分块。GPU FP32 算力与内存带宽比值普遍不高，无需特殊的 tile 映射方法提高 L2 cache 命中率，所以我们基于朴素的 tile 映射方法分析 L2 cache 命中率，即 tile_x = blockIdx.x, tile_y = blockIdx.y。

L2 cache 命中率主要依赖同时运行的 thread block 对输入矩阵的重复读取。首先说明 wave 的概念：wave 表示 GPU 上同时执行的 thread block。例如一个 kernel 中 thread block 为 256 线程，每个线程使用了 128 个寄存器，那么在 GV100 上每个 SM 可同时执行 2 个 thread block，GV100 共 80 个 SM，一个 wave 就是 160 个 thread block。

GEMM 一个 wave 对应的 A B C 矩阵区域如下图所示：

一个 wave 大小表示为 wave_{gpu} ，基于上图我们定义：

wave_{x}=\frac{N}{N_{tile}}

wave_{y}=\frac{wave_{gpu}}{wave_{x}}

wave_{rem}=wave_{gpu}\%\ wave_{x}

可以算出一个 wave 对应的访存请求量：

A_{ldg\_request}=wave_{gpu}\cdot M_{tile}\cdot K

B_{ldg\_request}=wave_{gpu}\cdot N_{tile}\cdot K

dram 实际访问量：

A_{dram\_ldg}=(wave_{y}+1)\cdot M_{tile}\cdot K

B_{dram\_ldg}=M\cdot K

可得 L2 cache 命中率约：

L2_{hit\_rate}=1-\frac{A_{dram\_ldg}+B_{dram\_ldg}}{A_{ldg\_request}+B_{ldg\_request}}

L2 cache 加权后平均访存带宽约：

P_{ldg}=P_{L2\_bw}\cdot L2_{hit\_rate}+P_{dram\_bw}\cdot(1-L2_{hit\_rate})

SGEMM 预估性能：

P_{sgemm}= \begin{cases} P_{fma}, & sgemm_{fma\_ldg\_ratio}\geq\frac{P_{fma}}{P_{ldg}},\\ P_{ldg}\cdot sgemm_{fma\_ldg\_ratio}, & sgemm_{fma\_ldg\_ratio}\lt\frac{P_{fma}}{P_{ldg}} \end{cases}

P_{sgemm} 表达式主要有两个作用，对于给定的 M, N, K:

估算出各种分块大小的性能，选择最快的 kernel；
对于给定的 GPU 型号，算出跑满硬件算力所需的最小分块；

注意，上述分析成立有 2 个条件：

大矩阵，即 M, N, K 较大，A, B 矩阵无法完全放进 L2；
Tile 总数超过一个 wave 大小；

由于 L2 cache 的替换策略对命中率有一定影响，上面表达式算出的命中率相比实际命中率会有误差，当A，B矩阵容量较大时 (比如 L2 容量 5 倍以上) 误差较小 (10%以内)，更小的矩阵误差会更大，但这种误差通常不会导致不同分块尺寸间的性能比较结果出错，也就是上面的方法做性能分析是足够准确的。

对于能完全装入 L2 的矩阵，实际 dram 读取量就与 A, B 矩阵内存接近，基于这个假设做理论性能分析就好。

2.2 L2 Cache 命中率与矩阵规模的关系

2.1 节分析了矩阵大小，分块大小，L2 cache 命中率之间的关系。如果我们固定 thread block tile 的大小，只关注 M, N, K 对 L2 命中率的影响，且忽略 2.1中 wave_{rem} 的影响，这个关系就会变得非常简单。本节用这种简化的计算方法估算 L2 命中率，便于大家对 L2 的实际命中率数值有直观感受。

thread block tile 为常数时，occupancy 和一个 wave 覆盖的 C 矩阵面积也是常数，令 wave 覆盖区域高 wave_{m} 宽 wave_{n} ，那么有 wave_{m}\cdot wave_{n} 为常数，且：

wave 访存请求量：

A_{ldg\_request}=wave_{gpu}\cdot M_{tile}\cdot K

B_{ldg\_request}=wave_{gpu}\cdot N_{tile}\cdot K

dram 实际访问量：

A_{dram\_ldg}=wave_{m}\cdot K

B_{dram\_ldg}=wave_{n}\cdot K

L2 cache 命中率：

\begin{equation} \begin{aligned} L2_{hit\_rate}&=1-\frac{A_{dram\_ldg}+B_{dram\_ldg}}{A_{ldg\_request}+B_{ldg\_request}}\\\\ &=1-\frac{wave_{m}+wave_{n}}{(M_{tile}+N_{tile})\cdot wave_{gpu}} \end{aligned} \end{equation}

可以看出，当 wave_{m} 与 wave_{n} 之差越大 (wave 覆盖区域越扁平/细长)，L2 命中率越低。当 N 足够大，使一个 wave 的 thread block 摆成一行时就变成极度扁平的情况。这也解释了用同一种分块大小，当矩阵 N 变大，或 N 极小的情况下 M 变大时 L2 命中率下降的原因。

假设 thread block tile 取 128*128, thread block 为 256 线程，每线程使用 128 个寄存器，在 Titan V 上一个 wave 就是 160 个 thread block。当 N 超过 128*160 = 20480 时变为极度扁平情况，此时 L2 命中率套用上述表达式为 50%。也就是说，对于这个分块方案，无论 M, N, K 怎么变，L2 命中率的下限为 50%。这个理论结果与上述条件下的实测结果相对误差在 4% 左右 (实测 ~52% 命中率)。

经验上讲，采用朴素的 tile 映射方法，大矩阵 L2 命中率通常在 50%~70% 之间，极度扁平/细长的情况在多数应用场景的矩阵尺寸中很少出现。

结合 2.1 和 2.2 节的分析，可能有人已经想到优化 L2 命中率的方法了，这里先卖个关子，在 tensor core GEMM 优化中再讲。

2.3 分块大小与 IPC 的关系

GEMM 计算过程主要包括计算指令 (FFMA) 和访存指令 (LDG, STS, LDS...)，FFMA 由 FFMA unit 执行，访存类指令由 LSU 执行，硬件上两种 unit 的性能比例决定了计算访存指令比例与整体性能的关系。所以为了跑满 FFMA 算力，还应确保 FFMA 与访存指令的比例高于硬件性能比例。

对于 thread block tile，有：

thread block 线程数：

n_{thread}=\frac{M_{tile}\cdot N_{tile}}{M_{frag}\cdot N_{frag}}

32bit LDG 指令数：

n_{ldg}=\frac{(M_{tile}+N_{tile})\cdot K_{tile}}{n_{thread}}

虽然 STS.128 可以一次写入 4 个 float，但也是拆成 4 次 memory transaction 去做的，对 MIO pipe 的占用等同于 4 条 STS.32 指令，所以 STS 按 STS.32 指令数计算：

n_{sts}=n_{ldg}

smem 读取使用 LDS.128 指令，每条指令读 4 个元素，且 warp 内线程通过 'z' 形排列来最大化 broadcast 性能 (每条 LDS.128 只需要 2 次 memory transaction，与硬件有关，后面解释)，等效 LDS 指令数：

n_{lds}=2\cdot(\frac{M_{frag}}{4}+\frac{N_{frag}}{4})\cdot K_{tile}

FFMA 指令数：

n_{ffma}=M_{frag}\cdot N_{frag}\cdot K_{tile}

考虑到 thread block tile 读取由 dram/L2 读取和 smem 写串联而成，且 dram 延迟较高，在 double buffer 流水并行设计中，为防止 FFMA 等待 tile 读取，应尽量让 LDG 在 K-loop 开头发射，STS 在 K-loop 结尾发射：

图中标红部分为访存指令密集区域。假设编译器对访存指令完美排序，即 LDG 读到数据后刚好发射对应的 STS，那么：

标红区域耗时占比：

\beta=1-\frac{LDG_{latency}+STS_{latency}}{n_{ffma}+FFMA_{latency}}\approx1-\frac{LDG_{latency}+STS_{latency}}{n_{ffma}}

标红区域访存指令数：

n^{\prime}_{ls}=n_{ldg}+\beta\cdot n_{lds}

FFMA 指令数：

n^{\prime}_{ffma}=\beta\cdot n_{ffma}

访存与计算指令比：

\begin{equation} \begin{aligned} R_{ls\_ffma}&=\frac{n^{\prime}_{ls}}{n^{\prime}_{ffma}}\\\\ &=\frac{n_{ldg}}{\beta\cdot n_{ffma}}+\frac{n_{lds}}{n_{ffma}}\\\\ &=\frac{\frac{M_{frag}N_{frag}}{M_{tile}}+\frac{M_{frac}N_{frag}}{N_{tile}}}{M_{frag}N_{frag}-\frac{LDG_{latency}+STS_{latency}}{K_{tile}}}+\frac{M_{frag}+N_{frag}}{2\cdot M_{frag}N_{frag}} \end{aligned} \end{equation}

要跑满 FFMA 算力，应满足：

R_{ls\_ffma}\leq\frac{P_{lsu\_ipc}}{P_{ffma\_ipc}}

P_{lsu\_ipc},P_{ffma\_ipc} 表示访存指令和 FFMA 指令 IPC。考虑到指令排布与访存延迟很难完美匹配， R_{ls\_ffma} 应在上述阈值的基础上保留一些裕量，也就是尽量降低访存计算指令比值。从 R_{ls\_ffma} 表达式可以看出，若M_{frag},N_{frag} 固定，M_{tile}, N_{tile} 越大，比值越低。同时 LDG, STS 的延迟是硬件相关的常数， K_{tile} 越大，FFMA 总时间就越长，可用来摆放 LDG 指令的时间窗口也就越大，访存指令占比越低。K_{tile} 过大会导致 smem 用量过多影响 occupancy，一般通过增大 M_{tile},N_{tile} 来降低访存指令占比。

注意，上述表达式成立需满足一个条件：

M_{frag}N_{frag}\gt\frac{LDG_{latency}+STS_{latency}}{K_{tile}}

如果不满足这个条件，直观理解也就意味着一次 LDG+STS 的延迟都无法被 FFMA 总时间覆盖，这种情况下需借助多 warp 切换来填满 FFMA。这也解释了在 DRAM 延迟较高的设备上(比如 GDDR 内存的 GPU)，若 occupancy 过低，即使依 2.1 节分析 tile 大小能满足 FFMA 峰值性能，实测性能也有较大差距的原因。cutlass sgemm 用 cuda11 编译，在 turing/GP102+ 等 DRAM 延迟高的设备上性能差，也有这方面原因，因为寄存器用量过大使 occupancy 太低了 (128*128 tile 在每个 SM 上只能跑一个 thread block)。

一般来讲，tile 减小时 thread block 变小，更容易达到更高的 occupancy，可以降低访存指令数占比对性能的影响，所以对于小 tile, 2.1 节分析的计算访存比对性能的影响更大，2.3 节的主要目的是对于大矩阵乘法，帮助选择合适的 tile 尺寸以跑出硬件算力上限。repo 中提供了dram_latency.cu和smem_latency.cu可用来测试 R_{ls\_ffma} 表达式所需的DRAM 和 smem 的延迟。GDDR 内存延迟一般 450~550 cycle，HBM 内存 350~400 cycle，smem 约 20~30 cycle，SGEMM 对于大矩阵一般选 128*64 或 128*128 的 thread block tile。

2.4 '尾部效应' 的影响

....

2.5 SplitK

...




施工中.....

全文尽快完成，请大家稍安勿躁 >_<

着急的话可以先看看 repo 的代码 ^_^
