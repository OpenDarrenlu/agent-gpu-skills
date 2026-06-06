# 关于Tensor Core mma与内存层级 带宽和延迟的一点思考

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/21509666996

---

前言

突发奇想打算整理一下近几代Nvidia 数据中心级/消费级旗舰卡关于Tensor Core(TC)算力和存储之间的配比关系。

GPU	A100	Rtx3090	Rtx4090	H100	Rtx5090	B100
Arch	Ampere
sm_80	Rtx Ampere
sm_86	Rtx Ada
sm_89	Hopper
sm_90	RtxBlackwell
sm_120	Blackwell
sm_100
TFlops
( fp16 TC with fp16 accumulate)	312
(tc gen3)	142.3
(tc gen3)	330.3
(tc gen4)	989.4
(tc gen4)	419
(tc gen5)	1800(tc gen5)(from cudocompute)
DRAM Bandwidth
(GBps)	1555
(HBM2)	936
(GDDR6X)	1008
(GDDR6X)	3352
(HBM3)	1792
(GDDR7)	8000
(HBM3E)
Arithmetic Intensity(fp16 Flop/Byte)	200.6	152	327.7	295.2	233.8	225
SM count	108	82	128	132	170	132(from techpowerup)
Boost Clock
(MHz)	1410	1695	2520	1830	2407	1837(from techpowerup)
FMA/Clock/SM	1024	512	512	2048	512	4096
FMA/Clock/TC
(4TC/SM)	256	128	128	512	128	1024

（注：B100/B200 由于官方没有发布最终的白皮书，不同来源的参数有些出入，按照techpowerup 的spec，1837MHz 的boost clock 对应的2d算力应该是1986TFlops，1800TFlop对应Base Clock 1665 MHz）

如下公式，可以根据gpu 的spec计算出对应历代Tensor Core 的FMA 吞吐。

TFlops = 2\times Clock\times SM\_count\times FMA/Clock/SM \\ FMA/Clock/SM= \frac{TFlops}{2\times Clock\times SM\_count}\\ FMA/Clock/TC = \frac{FMA/Clock/SM}{4}

如上表所示，数据中心计算卡A100,h100,B100 Tensor Core 的FMA吞吐和DRAM 带宽逐代翻倍，消费级旗舰图形卡3090 4090 5090 Tensor Core 的FMA吞吐保持不变仅增加新的数据类型支持(fp8,fp6,fp4等）、不支持最新的TC指令如wgmma和tcgen5 ，且tensor core fp16 with fp32 acc 的算力是折半的，仅通过提高频率和SM 数目来提高整体的算力。

下文中的部分数据引用Benchmarking and Dissecting the Nvidia Hopper GPU Architecture中的benchmark结果。

Global Memory Latency

如下表中的数据，可以看到 近几代Nvidia GPU 各级存储的访问延迟大致是相近的，消费级的产品latency会略高。

hierarchical memory latency
https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/
latency causes memory bus idle

访问Global Memory 的latency 是电信号在总线/集体管中传输的耗时，受制于物理上的限制。latency中memory bus 是空闲，Global Memory latency 大约是500～600cycle，为了隐藏latency 在等待latency的同时发起其他访问GMEM的请求。以下图为例，每个thread 访问2个fp64(16B)的数据触发404ns(569cycle)的latency，A100 可以以1555GBps的带宽在404ns中传输628220B的数据，为了充分隐藏的latency 至少需要39264个thread 同时发起GMEM访问。

Little's Law

L(Needed\ Parallelism)=\lambda(Throughput) W(Latency)

从上述daxpy的例子里，引出Little's Law。为了达到GPU Peak Memory/Compute Throughput， 需要足够大的并行的任务来隐藏计算/访存的延时，至少为 Throughput\times Latency。通常会采用TLP(Thread Level Parallelism)/ILP(Instruction Level Parallelism) 两种并行方式来满足隐藏latency所需的并行度。

https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf
Memory Throughput and TC Throughput Balance
hierarchical memory throughput

本小节讨论存储带宽和TC FMA吞吐之间的平衡，Shared Memory(SMEM)以4B 为单位分为32Bank， 在Bank Conflict Free 的情况下可以提供128B/SM/Clock 的带宽(非旗舰的产品SMEM的带宽会更低，不在讨论范围)。

(注：以下关于TC/ Memory吞吐的分析，基于一个满指令流水并行的条件下，即SMEM 访问、Register 访问、TC mma指令发射等等latency 都被完美隐藏)

Ampere Tensor Core

对于A100 的TC提供256FMA/Clock/TC(1024FMA/Clock/SM) 的fp16 mma 吞吐，1个SM内4个Warp 并行执行基础m16n8k16 的mma op 需要8cycle，从SMEM 读取4个warp 计算m16n8k16 mma 的A B矩阵需要 4\times \frac{(16\times 16+16\times8)\times 2B} {128B/Clock}=24cycle，此时是SMEM 带宽的瓶颈，S2R数据搬运的耗时大于TC计算的耗时。解决SMEM的带宽瓶颈可以通过提高访存计算比的方式（增大warp tile Shape 提高register数据复用），以常见的大CTA Tile m128n128k16 (4warp 128thread)为例，每个warp 负责m64n64k16的tile，计算TC mma latency =\frac{128*128*16}{1024FMA/Clock}=256cycle ，SMEM 搬运数据的latency =4\times \frac{(64\times16\times2)\times2B}{128B/Clock}=128cycle ，转化成Compute Bound。

Hopper Tensor Core

H100 上的4 gen TC 比A100 的3 gen TC吞吐翻倍，达到了512FMA/Clock/TC(2048FMA/Clock/SM)，SMEM的吞吐保持128B/Clock 不变，TC的吞吐的增加使得SMEM 带宽的压力更大。在Hopper为了进一步缓解SMEM 的带宽压力在TC引入了wgmma(warp group mma) 指令,允许一个SM上的4个warp组成warp group 同时对4个TC发起更大尺寸的mma计算，同时wgmma指令允许输入的A来自SMEM/register、B必须来自SMEM，通过4个warp间在SMEM的数据共享，进一步提高计算访存比,缓解SMEM的带宽压力。以wgmma fp16支持的最大mma shape m64n256k16为例，TC mma 计算latency =\frac{64\times256\times16}{2048FMA/Clock}=128cycle ，如果按照warp-level 的mma指令（假设mma在Hopper上可以达到TC的最高吞吐），每个warp独立负责m32n128k16 的tile，4个warp分别从SMEM搬运各自所需的数据的latency =4\times\frac{(32\times16+128\times16)2B}{128B/Clock}=160cycle>128cycle，此时为SMEM 带宽瓶颈。采用wgmma 异步指令，SMEM搬运4个warp共享的全部数据的latency =\frac{(64\times16+256\times16)\times2B}{128B/Clock} = 80cycle ，同时减少了register的使用量。

BlackWell Tensor Core

B100的TC吞吐在H100的基础上进一步翻倍，达到了1024FMA/Clock/TC(4096FMA/Clock/SM)，假设Blackwell SMEM 的吞吐保持128B/Clock, 那么Blackwell在不引入Tensor Memory 缓解SMEM压力的前提下，只能在Hopper 最大的m64n256k16 mma op基础上进一步增大mma op 的shape提高计算访存比，但是过大粒度的mma op 对于实际开发并不友好，tile quantization效应会更显著，导致算力浪费。因此Blackwell引入了Tensor Memory来缓解SMEM的带宽压力，同时在TC gen5中设计了collector提高数据的复用和支持CTA cluster让2个SM 共享B数据一起计算mma，进一步减少内存带宽的瓶颈。（Blackwell ISA只看了一部分，Cutlass 的代码还没看，欢迎补充细节）

总结

用Stephen Jones 在GTC 2021 的presentation中的一句话来总结“Almost nobody really cares about Flops，because we should really be caring about memory bandwidth/latency“。

参考资料

PTX ISA 8.7 documentation

RTX Blackwell whitepaper

Hopper whitepaper

B100 B200 spec from cudocompute
