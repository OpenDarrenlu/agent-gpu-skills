# Tools(2): Nsight Compute 使用指南

**作者**: 紫气东来​上海交通大学 工学硕士

**原文链接**: https://zhuanlan.zhihu.com/p/715022552

---

​
目录
收起
1. ncu 的安装与profile生成
2. ncu profile 的分析
2.1 Summary
2.2 Details
参考资料
1. ncu 的安装与profile生成

Nsight Compute安装包在 https://developer.nvidia.com/tools-overview/nsight-compute/get-started 可以获得。


Nsight Compute 工具将其测量库插入到应用程序进程中，这允许分析器拦截与CUDA用户模式驱动程序的通信。此外，当检测到内核启动时，库可以从GPU收集所请求的性能指标。然后将结果传输回前端。

可以通过ncu --help来查看ncu命令的参数

General Options:
  -h [ --help ]                         Print this help message.
  -v [ --version ]                      Print the version number.
  --mode arg (=launch-and-attach)       Select the mode of interaction with the target application:
                                          launch-and-attach
                                          (launch and attach for profiling)
                                          launch
                                          (launch and suspend for later attach)
                                          attach
                                          (attach to launched application)
  -p [ --port ] arg (=49152)            Base port for connecting to target application
  --max-connections arg (=64)           Maximum number of ports for connecting to target application
  --config-file arg (=1)                Use config.ncu-cfg config file to set parameters. Searches in the current 
                                        working directory and "$HOME/.config/NVIDIA Corporation" directory.
  --config-file-path arg                Override the default path for config file.

Launch Options:
  --check-exit-code arg (=1)            Check the application exit code and print an error if it is different than 0. 
                                        If set, --replay-mode application will stop after the first pass if the exit 
                                        code is not 0.
  --injection-path-32 arg (=../linux-desktop-glibc_2_11_3-x86)
                                        Override the default path for the 32-bit injection libraries.
  --injection-path-64 arg               Override the default path for the 64-bit injection libraries.
  --preload-library arg                 Prepend a shared library to be loaded by the application before the injection 
                                        libraries.
  --call-stack                          Enable CPU Call Stack collection.
  --nvtx                                Enable NVTX support.
  --support-32bit                       Support profiling processes launched from 32-bit applications.
  --target-processes arg (=all)         Select the processes you want to profile:
                                          application-only
                                          (profile only the application process)
                                          all
                                          (profile the application and its child processes)
  --target-processes-filter arg         Set the comma separated expressions to filter which processes are profiled.
                                          <process name> Set the exact process name to include for profiling.
                                          regex:<expression> Set the regex to include matching process names for 
                                        profiling.
                                            On shells that recognize regular expression symbols as special characters,
                                            the expression needs to be escaped with quotes.
                                          exclude:<process name> Set the exact process name to exclude for profiling.
                                          exclude-tree:<process name> Set the exact process name to exclude
                                            for profiling and further process tracking. None of its child processes
                                            will be profiled, even if they match a positive filter.
                                        The executable name part of the process will be considered in the match.
                                        Processing of filters stops at the first match.
                                        If any positive filter is specified, only processes matching a positive filter 
                                        are profiled.
  --null-stdin                          Launch the application with '/dev/null' as its standard input. This avoids 
                                        applications reading from standard input being stopped by SIGTTIN signals and 
                                        hanging when running as backgrounded processes.

Attach Options:
  --hostname arg                        Set hostname / ip address for connection target.

Common Profile Options:
 ...

在正常情况下，大多数参数并不需要使用，通常使用以下命令即可

ncu --set full -o *** python3 xxx.py

完成后会在服务器上产生一个 ***.ncu-rep 文件, 可以在本地用 Nsight Compute 打开。

2. ncu profile 的分析

上一节介绍了 ncu 生成 profile 的方法，本节将以一个具体案例来介绍如何解读 profile。
在 reference.py 里实现了一个基本的 attention 结构，通过 ncu -o attn_fwd --set full python test_attention.py生成一个名为 attn_fwd.ncu-rep 的文件，生成过程的日志如下所示：

==PROF== Connected to process 3348935 (/usr/bin/python3.10)
==WARNING== Unable to access the following 6 metrics: ctc__rx_bytes_data_user.sum, ctc__rx_bytes_data_user.sum.pct_of_peak_sustained_elapsed, ctc__rx_bytes_data_user.sum.per_second, ctc__tx_bytes_data_user.sum, ctc__tx_bytes_data_user.sum.pct_of_peak_sustained_elapsed, ctc__tx_bytes_data_user.sum.per_second.

==PROF== Profiling "distribution_elementwise_grid..." - 0: 0%....50%....100% - 37 passes
==PROF== Profiling "distribution_elementwise_grid..." - 1: 0%....50%....100% - 37 passes
==PROF== Profiling "distribution_elementwise_grid..." - 2: 0%....50%....100% - 37 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 3: 0%....50%....100% - 37 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 4: 0%....50%....100% - 37 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 5: 0%....50%....100% - 38 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 6: 0%....50%....100% - 38 passes
==PROF== Profiling "elementwise_kernel" - 7: 0%....50%....100% - 38 passes
==PROF== Profiling "elementwise_kernel" - 8: 0%....50%....100% - 38 passes
==PROF== Profiling "Kernel" - 9: 0%....50%....100% - 37 passes
==PROF== Profiling "softmax_warp_forward" - 10: 0%....50%....100% - 37 passes
==PROF== Profiling "vectorized_elementwise_kernel" - 11: 0%....50%....100% - 37 passes
==PROF== Profiling "elementwise_kernel" - 12: 0%....50%....100% - 37 passes
==PROF== Profiling "sm80_xmma_gemm_f32f32_f32f32_..." - 13: 0%....50%....100% - 37 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 14: 0%....50%....100% - 37 passes
==PROF== Profiling "unrolled_elementwise_kernel" - 15: 0%....50%....100% - 38 passes
==PROF== Disconnected from process 3348935
==PROF== Report: /share_data/data-before/zzd/repos/cuda_learning/05_cuda_mode/ncu_profile/attn_fwd.ncu-rep

使用 Nsight Compute 打开这个文件。

2.1 Summary

从第一页看起，该页主要显示的是 summary， 其中序号 0-15 则是依次运算的kernel，其信息包括：

ID: 每个函数的唯一标识符。
Estimated Speedup: 估计的加速比，表示如果优化这个函数可能带来的速度提升。
Function Name: 函数的名称。
Demangled Name: 去掉修饰符的函数名称。
Duration: 函数执行时间（以ns为单位）。
Runtime Improvement: 估计的运行时间提示（以ns为单位），表示如果优化这个函数可能带来的运行时间提升。
Compute Throughput: 计算吞吐量。SM 吞吐量假设在 SMSPs 间负载平衡理想的情况下 （此吞吐量指标表示在所有子单元实例的经过周期内达到的峰值持续率的百分比）。
Memory Throughput: 内存吞吐量。计算内存管道吞吐量 （此吞吐量指标表示在所有子单元实例的经过周期内达到的峰值持续率的百分比）。
Registers: 每个线程使用的寄存器数量。
GridSize：kernel启动的网格大小。
BlockSize：每个Block的线程数。
Cycles：GPC指令周期。GPC：通用处理集群（General Processing Cluster）包含以 TPC（纹理处理集群）形式存在的 SM、纹理和 L1 缓存。 它在芯片上被多次复制。

最上部的Result默认显示的是ID=0的 kernel 运行的部分信息，包括GPU型号及频率。

从该图可以比较快速的得到一些信息：

Memory-bound：Memory Throughput 明显高于 Compute Throughput 的 kernel ，例如 ID=6,11
Compute-bound：Compute Throughput 明显高于 Memory Throughput 的 kernel ，例如 ID=9,13
2.2 Details

上一部分概览了所有 kernel 的信息，本节将对某个具体kernel 进行详细分析，接下来 ID=9 的 kernel 为例进行说明。

(1) GPU Speed Of Light Throughput

该指标可以详细看到 Compute 和 不同层次的 Memory 的实际利用效率的情况，由此可以定位其在 roofline 中的位置。

从这个结果可以看出：

计算吞吐量(75.33%)高于内存吞吐量(43.98%)，表明这可能是一个计算密集型任务。
L2 缓存和 DRAM 吞吐量相对较低，可能存在优化空间。
L1吞吐量与总体内存吞吐量相近，说明主要的内存操作与该部分交互，需要特别说明的是 Shared memory 也统计在内。

除了整体的性能，还可以查看其在 roofline 图中的位置，可见这是一个典型的 Compute-bound 的算子。

(2) Memory Workload Analysis

该指标主要内存资源的使用情况，主要包括通信带宽、内存指令的最大吞吐量。详细的数据表如下：

Memory Throughput: 540.95 Gbyte/s

即每秒在DRAM中访问的字节数。

L1/TEX Hit Rate: 0

每个 sector 的 sector 命中次数 （这个比率指标表示跨所有子单元实例的值，以百分比表示）。

l1tex：一级（L1）/纹理缓存位于GPC内部。 它可以用作定向映射的共享内存和/或在其缓存部分存储全局、本地和纹理数据。

sector：缓存线或设备内存中对齐的32字节内存块。 一个L1或L2缓存线是四个sector，即128字节。 如果标签存在且sector数据在缓存线内，则sector访问被归类为命中。 标签未命中和标签命中但数据未命中都被归类为未命中。

解释一下这里的 Hit Rate 为什么是 0，可能有以下几种解释：

a) 如果加载指令缓存操作是 ld.cg, 该操作直接从 L2 缓存中读取数据，而不会访问 L1 缓存，因此 L1 缓存命中率是 0，依据在这里。

b) Shared memory 的效率更高，并行度较低的情况下主要使用 Shared memory，因此 L1 缓存命中率较低。从目前情况看该解释更合理，那么什么时候应该使用 L1 呢，可以关注这里。

L2 Hit Rate: 89.82%

L2sector查找命中的比例 （这个比率指标表示跨所有子单元实例的值，以百分比表示）。

l2s：二级（L2）缓存切片是二级缓存的一个子分区。 l2s_t 指的是其标签阶段。 l2s_m 指的是其未命中阶段。 l2s_d 指的是其数据阶段。

Mem Busy: 43.98%

缓存和DRAM内部活动的吞吐量（这个吞吐量指标表示在所有子单元实例的经过周期内达到的峰值持续速率的百分比）

Max Bandwidth: 24.66%

SM<->缓存<->DRAM之间互连的吞吐量 （这个吞吐量指标表示在所有子单元实例的经过周期内达到的峰值持续速率的百分比）

L2 Compression Ratio: 0
L2 Compression Success Rate: 0

Memory Chart图分析
改图显示了各级 memory 的连接关系及使用情况，整体情况一目了然。

(3) Compute Workload Analysis

分析完了内存的情况后，接下来分析计算单元的使用情况。即对流式多处理器（SM）的计算资源进行详细分析，包括实际达到的每时钟周期指令数（IPC）以及每个可用流水线的利用率。主要指标包括：

Executed Ipc Elapsed: 3.01 inst/cycle

执行的warp指令数，此计数器指标表示所有子单元实例中每个执行周期的平均操作数。

Executed Ipc Active: 3.16 inst/cycle

执行的warp指令数，此计数器指标表示所有子单元实例中每个活动周期的平均操作数。

Issued Ipc Active: 3.16 inst/cycle

发出的warp指令数，此计数器指标表示所有子单元实例中每个活动周期的平均操作数。与上一项比较可知，在活动周期内发出的指令都被执行。

SM Busy

假设SMSP间理想负载平衡的SM核心指令吞吐量，此吞吐量指标表示在所有子单元实例的活动周期内达到的峰值持续率的百分比。

SMSPs: 每个SM被划分为四个处理块，称为SM子分区。 SM子分区是SM上的主要处理元素。 一个子分区管理固定大小的warp池。

Issue Slots Busy

发出的warp指令数，此计数器指标表示在所有子单元实例的活动周期内达到的峰值持续率的平均百分比。

接下来是一些主要计算单元的利用率情况，先分别介绍一下这些计算单元：

FMA: Fused Multiply Add/Accumulate，融合乘加。FMA流水线处理大多数FP 32算法（FADD、FMUL、FMAD）。它还执行整数乘法运算（IMUL、IMAD）以及整数点积。
ALU: Arithmetic Logic Unit, 算术逻辑单元。ALU负责执行大多数位操作和逻辑指令。它还执行整数指令，不包括IMAD和IMUL。在NVIDIA Ampere架构芯片上，ALU流水线执行快速的FP 32到FP 16转换。
LSU: Load Store Unit, 加载存储单元。LSU流水线向L1 TEX单元发出用于全局、本地和共享内存的加载、存储、原子和归约指令。它还向L1 TEX单元发出特殊的寄存器读取（S2 R）、混洗和CTA级到达/等待屏障指令。
TMA: Tensor Memory Access Unit, 张量存储器访问单元。在全局内存和共享内存之间提供高效的数据传输机制，能够理解和遍历多维数据布局。
ADU: Address Divergence Unit, 地址分支单元。ADU负责分支/跳转的地址发散处理。它还支持恒定加载和块级屏障指令。
CBU：Convergence Barrier Unit，汇聚屏障单元。CBU负责曲速级收敛、屏障和分支指令。
TEX: Texture Unit, 纹理单元。SM纹理流水线将纹理和表面指令转发到L1TEX单元的TEXIN阶段。在FP64或Tensor流水线解耦的GPU上，纹理流水线也会转发这些类型的指令。
Uniform: Uniform Data Path, 统一数据路径。这个标量单元执行所有线程使用相同输入并生成相同输出的指令。
XU: Transcendental and Data Type Conversion Unit, 超越和数据类型转换单元。XU管道负责特殊函数，如sin、cos和倒数平方根。它还负责int到float和float到int类型的转换。

（3）Statistics

a) Scheduler Statistics

度器发出指令活动的总结。每个调度器维护一个可以为其发出指令的warp池。池中warp的上限（理论warp数）受启动配置的限制。在每个周期，每个调度器检查池中分配的warp的状态（活跃warps）。未被停滞的活跃warps（Eligible warps）准备好发出它们的下一条指令。从Eligible warps集合中，调度器选择一个warp来发出一条或多条指令（已发射的warp）。在没有Eligible warps的周期中，发射槽被跳过，不发出任何指令。有许多被跳过的发射槽表明延迟隐藏效果不佳。

Active Warps Per Scheduler[warp]：2.03

累计活跃的线程组数量, 该计数器度量了每个活跃周期中所有子单元实例的平均线程组数量。

Eligible Warps Per Scheduler[warp]：1.42

在每个活跃周期中发出1条指令的周期数。这个计数器指标代表所有子单元实例中每个活跃周期的平均操作数。

Issued Warp Per Scheduler: 0.79

发出1条指令的周期数。这个计数器指标代表了所有子单元实例中操作单元每个活动周期的平均操作数。

No Eligible[%]：21.13

在活跃周期中没有指令被发出的活跃周期的百分比。这个计数器指标表示在所有子单元实例中，达到峰值持续活跃状态期间的平均百分比。

One or More Eligible[%]：78.87

在活跃周期中发出一条指令活跃周期的百分比。这个计数器指标表示在所有子单元实例中，达到峰值持续活跃状态期间的平均百分比。这个指标和No Eligible[%]互补。

每个调度器(scheduler)每个周期都可以发出一条指令, 实际运行中这个内核每 2 个周期才会发出一条指令（Active Warps）。
每个调度器最多可以分配16个线程组(warp),但这个内核平均只分配了2个活跃的线程组。然而,每个周期里平均只有1.42个线程组是可以发出指令的(eligible)。
可以发出指令的线程组(eligible warps)是活跃线程组(active warps)的子集,它们是准备好发出下一条指令的。
每个周期如果没有可以发出指令的线程组,就会导致调度插槽(issue slot)被浪费,没有发出任何指令。
为了提高可以发出指令的线程组数量,需要减少活跃线程组被阻塞的时间。可以查看"Warp State Statistics"和"Source Counters"部分,找出导致线程组被阻塞的主要原因。

b) Warp State Statistics

分析所有 warp 在内核执行期间所花费的周期数。warp state 描述 warp 是否准备好发出下一个指令。每条指令的 warp 周期定义了两条连续指令之间的延迟。该值越高，隐藏此延迟所需的 warp 并行度就越高。对于每个warp state，该图表显示了每个发出的指令在该状态下花费的平均周期数。stall 并不总是影响整体性能，也不是完全可以避免的。

平均而言，该内核的每个warp在等待微调度器选择要发出的warp时会花费0.8个周期。未被选中的warp是指在该周期内没有被调度器选择发出的符合条件的warp。大量未被选中的warp通常意味着有足够多的warps来覆盖warp延迟，并且可以考虑减少活跃warps数量以可能增加缓存一致性和数据局部性。这种停顿类型占总体平均值2.5个周期之间两条指令发出时间31.4%左右。

Warp Stall 的原因使用warp调度器状态采样收集。无论调度程序是否在同一周期内发出指令，它们都会递增。具体每项的解释可通过参考资料[1]了解

c) Source Counters

源度量，包括分支效率和采样的 warp stall 原因。Warp Stall采样指标在内核运行时周期性地进行采样。

（4）Occupancy

占用率(Occupancy)是指每个SM上活跃线程组(warp)的数量与可能的最大活跃线程组数量的比率。另一种看待占用率的方式是,它表示硬件处理线程组的能力中实际被使用的百分比。虽然较高的占用率并不总能带来更高的性能,但是低占用率会降低隐藏延迟的能力,从而导致整体性能下降。在执行过程中,理论占用率和实际达到的占用率之间存在较大差异,通常表示工作负载高度不均衡。占用率反映了GPU资源的利用情况,是评估CUDA程序性能的一个关键指标。过低的占用率会导致性能下降,需要分析并优化造成低占用率的原因。

这个在之前的文章中已经计算过，在此不予赘述

三张图可以看到，块大小对性能的影响，每块共享内存用量对性能的影响。

随着每线程寄存器数量的增加,性能先上升后下降,存在一个最优值。寄存器数量的增加会限制同时运行的线程数,需要权衡利弊。
块大小的变化也会影响性能表现,存在一个最优值。块大小过大会限制并行度,过小则会增加调度开销。
共享内存用量的增加会降低可同时运行的块数,从而影响性能。

以上代码更新在：cuda_learning/05_cuda_mode/ncu_profile at main · ifromeast/cuda_learning (github.com)。

参考资料

[1] https://docs.nvidia.com/nsight-compute/pdf/ProfilingGuide.pdf

[2] CUDA-MODE 第一课课后实战（上）Nsight Compute - 知乎 (zhihu.com)

[3] BBuf：CUDA-MODE 第一课课后实战（下）Nsight Compute

[4] 2. Kernel Profiling Guide

今宵绝胜无人共，卧看星河尽意明。 —— 陈与义《雨晴》
