# 细读论文：Insights into DeepSeek-V3

**作者**: ZhangZVibe Coding忠实拥趸手撸醋打蒜籽

**原文链接**: https://zhuanlan.zhihu.com/p/2042623507931165218

---

这篇文章将为您深入解析 《Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures》 这篇具有里程碑意义的论文。该论文由 DeepSeek 团队发布，详细阐述了如何在 2,048 张 NVIDIA H800 GPU 的有限资源下，通过硬件与模型的深度协同设计（Co-Design），训练出达到 SOTA 性能的 671B 大模型。

现在DeepSeek-V4都有了 为什么要说V3呢？因为当前这个系列是在讲解通信，DS V3时期开源的DeepEP绕开了NCCL，使用NVSHMEM进行了细粒度的RDMA通信控制，这也带动了NCCL 2.28 Device API的革新。这些都是我们后面要讲的。

读过系列中其他文章的同学可以把这篇看作实战中的一些通信经验。

以下是针对该论文核心观点的深度细读：

1. 核心哲学：软硬件协同设计 (Co-Design)

DeepSeek-V3 的成功并非单纯依赖堆砌算力，而是源于对硬件限制的深刻理解。论文指出，当前大模型面临三大挑战：内存容量瓶颈、计算效率限制、互联带宽瓶颈。为了应对这些挑战，DeepSeek 团队没有被动适配硬件，而是根据硬件特性（如 H800 的带宽削减）反向驱动模型架构创新。

设计理念也是围绕着三个点：内存效率，成本效率，推理速度。

2. 内存效率的优化：
量化：bf16->fp8
MLA 架构 (这篇讲的很好deepseek技术解读(1)-彻底理解MLA（Multi-Head Latent Attention）)：设d=h*dh, 也就是hidden size是head的数量乘以dh。把一个4dh的向量up-scale成d，也就是当前token全部head的K和V的cache。RoPE单独存在了一个dh/2的向量cache里，以MQA的形式concat到每一个head的K里。
3. 成本效率：DeepSeek-MoE 与节点局限路由

针对混合专家模型（MoE）带来的巨大跨节点通信压力，论文展示了精妙的策略：

稀疏激活的经济性：DeepSeek-V3 总参数 671B，但每个 Token 仅激活 37B，训练成本仅约 250 GFLOPS/Token，远低于 405B 稠密模型的 2448 GFLOPS。
可在PC上部署：比如ktransformers就可以在用户级GPU上跑到20TPS。
4. 推理加速

DualPipe

如图是一个mbs=20的例子从上往下和从下往上各有10个micro batch的pipeline。为什么要这么做呢？因为上下两个方向的不同micro batch如果在某一个gpu上相遇，而且一个在做forward一个在做backward的话，他们的communication可以被computation完美掩盖。比如下图的batch 4和10以及14和0（这里的原图有个小错误，4 10 之间以及14 0之间不应该有黑色实线，他们是overlapped）

上图分析了DualPipe的bubble。ZB1P是ZeroBubble的方法。注意B代表了整个Backward的时间，W代表了其中weight gradient的计算部分，所以input gradient的部分应该是 B-W。所以ZB1P里每个stage带来的（F+B-2W）应该被拆解为(F)+(B-W)-(W).最后减去一个W其实就是因为weight gradient的计算不在critical path上，所以可以用来填充bubble。除了当前stage外，需要等待（PP-1）个stage让相应的micro batch被完全处理，所以需要乘以它。DualPiple这里因为是双向的，从图中可以看出来需要改成PP/2-1. 另外F&B代表了F和B overlap的情况下的总时长（注意W的部分也就是蓝色的部分是必须参与这个overlap的所以不能被单独拿出来）。除此之外，还会有单独的input gadient 计算B-W在critical path上，以及单独的W用于填充，DualPipe会带来两倍的填充机会（我其实不太确定，只能强行解释一下为什么是-3W）

另外有个缺点就是双向导致的每个GPU需要存储双倍的param，比如GPU0需要存layer0以及layer7的param。论文说因为EP size很大，所以增加的param其实不多。

DeepEP V1

High throughput: 也就是Training或者Prefill的部分。

首先CPU负责notify GPU去根据topk_idx来计算dispatch到不同rank/expert上的token分配，这是多个GPU相互通信一次只会才会知道的。这些信息被记录在host pinned memory上由CPU轮询获得，然后创建tensor，发起computetation/communication kernel。

利用两个micro batch互相overlap计算和通信。

通信任务会被分为不同的channel来进行，每个channel有send和recv两个buffer，各占一个SM。然后dispatch以及combine的路由会根据是在同节点还是不同节点分发到IB或者NVLink。

如果是同节点的memory，会由cuMem创建用IPC共享handle。如果是不同节点的memory，会由NVSHMEM控制。注意我们之前讨论了同一个process内的GPU是可以用GPUDIrect P2P的NCCL P2P_Direct flag绕过intermediate buffer，但是这里DeepEP时一个Process = 一个Buffer = 一个GPU，所以不能绕开。

每个GPU会通信多个rank，所以会有moe_recv_counter(int)以及moe_recv_expert_counter(List[int])这两个东西。moe_recv_expert_counter是每个通信的rank传过来的token数，moe_recv_counter是总共的，这些counter会在notify_dispatch（在dispatch中被调用）中被通信的rank通知填写。每个Buffer会有num_sms/2个channel，每个channel用一个block来send，另一个来recv。这些channel会在runtime被分配给各个需要通信的rank。

Low Latency: 用于decode，最大的好处是不再依赖准确的recv size，所以CPU不需要轮询了。

一句话概括就是RDMA-oriented fixed-capacity buffer。

buffer size不再需要用notify_dispatch来额外通过CPU轮询获得，直接给一个最坏情况的buffer size即可（[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden_size]). 同时这还是个double buffer，用来让两个micro batch互不阻碍。

因为decode的token size足够小，所以直接让NIC来处理data transfer而不会挤爆RDMA的QP数量限制。同时，显示的把一次数据传输（dispatch/combine）分成SEND_PHASE, RECV_PHASE, send会立马issue，而recv无需等待，用一个hook来异步的触发，实现低延迟无等待。这个情况下，0SM被通信用到，如下图。

另外zero copy的combine也是有可能的，我们可以让expert直接把输出数据写入要进行RDMA send 的buffer。

ProfileData

Training

为了验证一下上述方法的有效性，DeepSeek还开源了profile data

F&B

上图验证了forward backward chunk完美 overlap的有效性。

Inference

Prefill
Decode

上面两张图分别解析了DeepEP处理的HT和LL的情况。low latency的情况下需要手动设定async recv hook在哪里触发。这里就就通过在attn中间进行hook的调用，比较巧妙的把shared和attn-0（attn的第一个部分）结合来完美覆盖dispatch， combine也是类似。

DistServe

生产环境中还利用了类似DistServe的PD分离

MTP

一层进行多token预测，第二个token接受率80%～90%，提升TPS 1.8x

5. 算力潜能的释放：FP8 混合精度训练

DeepSeek-V3 是首个大规模应用 FP8 混合精度训练 的开源模型。

细粒度量化：为了解决 NVIDIA Hopper 架构在 FP8 累加精度上的局限性，DeepSeek 实现了weight块级（128x128）和acitvation切片级（1x128）的细粒度量化，并开源了高性能算子库 DeepGEMM。
无损精度：通过高精度累加策略，FP8 训练的相对精度损失控制在 0.25% 以内，但计算速度提升巨大。

限制：

accumulator的精度: FP8FP8FP32的MMA在hopper上实际不是FP32(E8M23)而是FP22(E8M13).这是SageAttention2的发现，不清楚这个和FP8格式是否有关。SageAttention2的解法是先算R=PV, 然后再在FP32的O上累加这个FP22的R。
dequantization overhead：quantization太过细粒度，tile-wise block-wise的quantization会引入过多overheads.

建议：

accumulator精度更高或者可控
block-scaled mma，blackwell已经实现了
6. LogFMT: Communication Compression

目前dispatch是fp8，combine是fp16

LogFMT就是在combine的阶段，对1x128这个tile里的值根据log值进行quantization。如果是LogFMT-8bit那么其实就减半了combine的通信量

建议：

能够把quant/dequant融入到alltoall，这样就不占计算也不占register了
7. 互联设计

因为h800阉割了NVLink带宽（900GB/s成了400GB/s），需要一些技术来增加带宽利用率。同时inter-node用的是IB以及CX7的NIC。

机内互联

并发：

TP：训练的时候就不用了，机内带宽太低了。但是推理的时候还是用来增强TTFT和TPOT
PP：DualPipe上面讲过了
EP：DeepEP上面也讲了

模型和带宽codesign：Node-Limited Routing

inter-node vs. intra-node大概是1:4的带宽（40GB/s:160GB/s)
如果要发送给的gpu在同一个node里，可以先把它们统一发给那个node，再进行节点内的routing。这样会对IB traffic去重
TopK Experts Selection Strategy：实际上DS利用了这一点在训练时限定比如在有8个node的情况下每个token最多发送给4个node，保证能够去重。

NVLink + IB的复杂处理

通常需要额外的SM处理复杂的数据传输。

这里DeepEP Low Latency就绕开了SM，上面已经说过了。

除此之外SM依然在HT场景下有很重的任务：

IB+NVLink数据传输
数据复制：在RDMA buffer 和模型输入输出buffer之间
reduce：alltoall combine
IB+NVLink数据的memory layout
dtype转换

建议：

能把NVLink和IB的通信统一起来，编程会简单一些。有些protocol（Ultra Ethernet Consortium (UEC)， Ultra Accelerator Link (UALink)）已经支持了
具体应该怎么做，比如Unified Network Adaptor，Dedicated Communication Co-Processor, Flexible Forwarding Broadcast and Reduce Mechanisms, Hardware Synchronization Primitives (acq/rel). 这些其实是在NCCL的后续版本的Device API中有相应的改善。

Bandwidth Contention and Latency

举个例子，如果正在从CPU移动KV Cache到GPU，同时需要沟通Network来进行EP，两个操作都需要PCIe，会把bandwidth挤爆。

建议：

应该给不同的场景TP/EP/KVCache不同的NVLink/PCIe优先级，用于动态分配traffic
NIC集成到IO Die上和Compute Die直连，不走PCIe了
CPU-GPU用NVLink连
8. 大规模网络设计

网络拓扑的设计直接影响万卡集群的稳定性与成本

MPFT 两层 Fat-Tree 支撑万卡：DeepSeek 部署了 多平面（Multi-Plane）两层 Fat-Tree 网络，取代了传统的昂贵三层架构，大幅降低了组网成本和通信延迟。

上图就是一个两层的结构每个水平平面都是一个plane，垂直的是一个个node。inter-node的传输发生在plane上，所以GPU0只会发送给另一个机器的GPU0。跨plane的传输会发生在intra-node，用NVLink来传输。这样子，每个平面的链接数目得到了限制，所以两层fat-tree就够了，节约了成本

理想情况：由于每个GPU-NIC对只会连接一个plane，这样的传输效率不够高，如果能够通过多个physical port连接多个plane，就可以做packet spraying，把一个QP的信息散布到多个平面上。这个会带来的一个坏处是同一个QP的信息会通过多个path到达终点，会有乱序的问题，需要NIC来额外解决信息顺序一致性的问题。目前CX-8就支持连接4个plane。这个理想情况也会让数据传输更robust。
流量隔离：每个 GPU 对配对一个独立的网络平面，确保了不同通信模式（如 EP 的 All-to-All 和 DP 的 All-Reduce）互不干扰。其实MPFT是MRFT（multi-rail fat-tree）的子集，区别就是不同Rail之间形成了真正的物理隔离。
PXN：在传统模式下，如果 GPU 0 要通过 NIC 3 发送数据，可能需要跨越复杂的 PCIe 拓扑或 QPI 协议。而 PXN 允许 GPU 0 先通过极高带宽的 NVLink 将数据移动到与目标轨道（或平面）对应的“中间 GPU”（例如 GPU 3），然后再由该 GPU 利用本地 NIC 直接发出。是GPU3->GPU0->NIC0, 而不是GPU3->PCIe-CPU QPI/UPI->NIC.
省钱：FT3-> FT2, 省掉了昂贵的最胖的Core层。
低延迟：层数减少也会降低延迟

实验：对比MPFT和MRFT

说明白一下两者区别：FT2 vs. FT3, Multi Plane vs. Single Plane。MRFT是3层胖树，且rail之间没有物理隔离。

Throughput: EP AlltoAll
Latency: EP AlltoAll

Throughput和Latency都和满血FT3差的不大

9. 低延迟网络

选IB还是RoCE？

IB latency更小，但是更贵，而且交换机只支持64个ports，但是RoCE支持128个。

如何改进RoCE

低延迟Switch：以太网的feature有很多没有用，还拉高延迟。
优化路由机制：比如ECMP的分发策略的ReduceScatter/AllGather带宽就不如Adaptive Routing以及Static Routing。
增强Traffic Isolation以及Congestion Control：virtual output queuing (VOQ)， RTT-based CC (RTTCC)， user-programmable CC (PCC)。

IBGDA（InfiniBand GPUDirect Async）

IBGDA被用来降低latency。绕过CPU直接填充WR（work request）写到RDMA。DeepEP就用到了。

10. 对未来硬件的深刻反思与建议

论文最后对硬件厂商（如 NVIDIA）提出了几点极具前瞻性的建议：

Robustness：比ECC更强的查错机制比如checksum，更强的诊断工具包。
CPU瓶颈：可以用NVLink绕过PCIe。Kernel launch和network processing也需要更多的CPU core。
智能网络：共封装光学（Co-Packaged Optics, CPO）引入硅光子技术（Silicon Photonics）；无损网络与端点驱动的拥塞控制；动态路由标准化如报文喷射（Packet Spraying）拥塞感知路径选择；高效容错协议通过部署自愈协议（Self-healing protocols）、冗余端口和快速故障转移（Failover）技术来增强鲁棒性；动态资源管理支持动态带宽分配和流量优先级排序。
内存语义增强，防止乱序：支持更高效的内存语义通信（如 RAR 机制），消除软件层面的 Fence/Sync 开销，降低 RTT 延迟。packet sequence numbers (PSN)，region-based acquire/release (RAR)
互联融合：建议未来的 NIC 和 GPU 应深度集成，将机内（Scale-up）和机间（Scale-out）网络统一，支持硬件级的包转发与去重。
卸载通信计算：建议引入专用通信协处理器，EP Dispatch容易一些，EP Combine的规约因为不是很规律所以比AllReduce难办。可以吧logFMT压缩和解压集成到硬件里
垂直堆叠显存：提倡使用 3D 堆叠 DRAM（如 SeDRAM）技术，通过垂直集成逻辑层与显存层，突破现有的内存带宽瓶颈。System-on-Wafer (SoW)增强计算密度和内存带宽。
