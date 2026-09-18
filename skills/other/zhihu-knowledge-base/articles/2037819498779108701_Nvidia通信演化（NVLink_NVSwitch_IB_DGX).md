# Nvidia通信演化（NVLink/NVSwitch/IB/DGX)

**作者**: ZhangZVibe Coding忠实拥趸手撸醋打蒜籽

**原文链接**: https://zhuanlan.zhihu.com/p/2037819498779108701

---

在单芯片算力逼近物理极限的今天，大模型训练与推理的成败完全取决于“数据流动的速度”。本文将层层剥离，横跨网卡芯片进化、机柜内物理互联、跨机柜超算网络.

注：大量资料来自网络，未经严格验证，可能会有错误，欢迎指正。

AI 算力中心网络 5 大物理层级全景架构表

大模型分布式算力中心的数据流动，被严密地划分为以下 5 个物理空间层级：




层级	空间范围与介质	核心打通的物理设备	采用的技术与协议	典型单点双向带宽 (Blackwell)	核心设计哲学与技术特征
第 1 层 芯片内部	毫米级 硅基中介层	单个 GPU 内部的计算核心与本地显存	HBM3e / HBM4 片上高密度集成总线	8 TB/s ~ 16 TB/s	近存计算极限。数据直接从高带宽显存送到 Tensor Core 寄存器，延迟几乎为零。
第 2 层 芯片到芯片	厘米级 主板焊接走线	节点内部的 CPU -> GPU	NVLink-C2C (Chip-to-Chip)	900 GB/s	消灭 PCIe 传统边界。Grace 与 Blackwell 内存一致性互联，打破 Device 与 Host 的物理界限。
第 3 层 机柜内部	米级 机柜盲插背板	GPU -> GPU (跨 18 个托盘，共 72 颗 GPU)	NVLink 5.0 协议 + 5000 根被动铜缆 + 机柜内 NVSwitch	1.8 TB/s (每颗 GPU) 整柜交换: 130 TB/s	在机柜内部消灭网卡。72 颗 GPU 在原生协议和地址空间上融合成一体，免去一切网络封装开销。
第 4 层 机柜/POD级	十米级 专用光纤线缆	机柜 -> 机柜 (扩展至 576 颗 GPU 域)	Scale-Out NVLink + 外部专用 NVSwitch 交换机	1.8 TB/s (保持原生协议速率)	NVLink 跨出机柜。在 576 卡范围内不转译 InfiniBand，直接用光纤延长 NVLink 信号，直面张量并行。
第 5 层 数据中心级	百米级 标准集群光纤	集群 -> 集群 (万卡至十万卡宏观网络)	InfiniBand (X800) 或 RoCE (Spectrum-X) + ConnectX 网卡/DPU	200 GB/s (单端口) 可通过多网卡堆叠横向扩展	大包大批发拓扑。超出 576 卡后，由于物理距离限制，网络全面包裹为基于报文的、无损的 RDMA 架构。
GPUDirect: 这是一个技术家族，其核心目标是“去 CPU 化”，让 GPU 与网卡、GPU 与存储、GPU 与 GPU 之间实现最短路径通信。
P2P vs RDMA: P2P 侧重于单机内通过总线（PCIe/NVLink）的直接访问；RDMA 侧重于跨网络节点的直接内存读写。
NVSwitch vs NVLink Network Switch: 前者通常指服务器内部用于连接 8 卡的交换芯片；后者是 NVIDIA 近年推出的独立交换机设备，通过专用光纤将多台服务器的 NVSwitch 连成一个巨大的虚拟 GPU。
第一章 硬件总线与交换芯片的代际演进

要理解为什么 NVIDIA 要在最新的机柜里消灭 PCIe 和传统网络，必须看清物理总线（PCIe）与私有互联（NVLink/NVSwitch）在过去十年间的带宽鸿沟演进。 (注：以下表格所有数据均统一转换为 双向总带宽/总交换容量（Bidirectional Aggregate Bandwidth）；关于GT/s 和GB/s的换算关系请看一些PCIE知识整理——带宽计算；下面PCIe都是16通道全双工的情况; NVLink细节请看NVIDIA NVLink 技术深度解析2025最新版， 一文读懂NVSwitch和NVLink)

1.1 芯片互联总线与 NVSwitch 演进全景表
年份	GPU 核心架构	对应 PCIe 标准及单路（x16）双向带宽	对应 NVLink 版本及单芯片双向带宽	NVSwitch 芯片版本及单芯片总交换容量	架构关键变化与技术突破
2014	Kepler / Maxwell	PCIe Gen 3 ~32 GB/s	尚未诞生	尚未诞生	全员锁死在 PCIe Gen 3，芯片间通信成为严重瓶颈。
2016	Pascal (P100)	PCIe Gen 3 ~32 GB/s	NVLink 1.0 160 GB/s (20GB/s x2 x4)	尚未诞生	NVLink 诞生。首次在 IBM POWER9 平台上实现 CPU-GPU 互联；x86 平台仅 GPU 间点对点互联。
2017	Volta (V100)	PCIe Gen 3 ~32 GB/s	NVLink 2.0 300 GB/s (25GB/s x2 x6)	第一代 NVSwitch	NVSwitch 诞生。8 颗以上 GPU 首次通过网格交换芯片织网，摆脱点对点连线限制。
2020	Ampere (A100)	PCIe Gen 4 ~63 GB/s	NVLink 3.0 600 GB/s (25GB/s x2 x12)	第二代 NVSwitch	彻底进入 x86 时代，CPU-GPU 倒退回 PCIe Gen 4。GPU 间 NVLink 带宽翻倍。
2022	Hopper (H100)	PCIe Gen 5 ~121 GB/s	NVLink 4.0 900 GB/s (25GB/s x2 x18)	第三代 NVSwitch	引入 NVLink-C2C 并搭载于 GH200，使 CPU-GPU 重新跑满 900 GB/s 协议极限；x86 版 DGX 仍走 PCIe。
2024	Blackwell (B200)	PCIe Gen 6 ~242 GB/s	NVLink 5.0 1.8 TB/s (50GB/s x2 x18)	第四代 NVSwitch	机柜即超级计算机。通过第四代 NVSwitch 与 5000 根铜缆，将 72 颗 GPU 融合成单一 1.8 TB/s 的超大域。
2026	Rubin (R100)	PCIe Gen 7 (草案) ~512 GB/s	下一代 NVLink 3.6 TB/s (预计)	第五代 NVSwitch	根据 NVIDIA 路线图，NVLink 带宽再次翻倍，全方面碾压同时代的 PCIe 标准。

不同版本的DGX：https://www.naddod.com/blog/brief-discussion-on-nvidia-nvlink-network?srsltid=AfmBOorQVBzTY1E_7XkQ_9tLpHW1NxOdczqpnR7DDU5EJTZdmNWO3IQ5

DGX System	Number of GPUs	NVLink Version	NVLink Ports per GPU	Per-GPU Bandwidth (Bidirectional)	NVSwitch Generation	Onboard NVSwitches	Ports per NVSwitch	Total Intra-System Bandwidth
DGX-1 (Pascal)	8x P100	NVLink 1.0	4 Ports	160 GB/s	None (Mesh)	0	0	960 GB/s
DGX-1 (Volta)	8x V100	NVLink 2.0	6 Ports	300 GB/s	None (Mesh)	0	0	1.8 TB/s
DGX-2	16x V100	NVLink 2.0	6 Ports	300 GB/s	NVSwitch 1.0	12 chips	18 Ports	4.8 TB/s
DGX A100	8x A100	NVLink 3.0	12 Ports	600 GB/s	NVSwitch 2.0	6 chips	36 Ports	4.8 TB/s
DGX H100 / H200	8x H100	NVLink 4.0	18 Ports	900 GB/s	NVSwitch 3.0	4 chips	64 Ports	7.2 TB/s
DGX B200	8x B200	NVLink 5.0	18 Ports	1,800 GB/s	NVSwitch 4.0	2 chips	72 Ports	14.4 TB/s
DGX GB200	72 x B200	NVLink 5.0	18 Ports	1,800 GB/s	NVSwitch 4.0	9 rack trays, 18 chips	72 ports	129.6TB/s
Vera Rubin NVL72	72x R100	NVLink 6.0	TBD	3,600 GB/s	NVLink 6 Switch	Rack Trays	TBD	260 TB/s

注：可以看到很多nvswitch的port没有被用到

注2:DGX-1没有NVSwitch，所以总bw不是300*8 = 2.4TB/s

注3: NVSwitch如果所有port都用上的bw可以这么算，比如NVSwitch3，64个port每个支持50GB/s的双向bw，总量就是3.2TB/s.

注4: Blackwell时期NVLink C2C不是GPU之间的1.8TB/s的速度而是900GB/s

x86 时代的卡脖子： 在 DGX A100 和传统 DGX H100 时代，由于 Intel 和 AMD 的 x86 CPU 内部没有集成了 NVLink 控制器，CPU 到 GPU 的连接被迫倒退并锁死在 PCIe 槽上（分别为 64 GB/s 和 128 GB/s）。这直接逼得 NVIDIA 掀翻桌子，自研 ARM 架构的 Grace CPU，利用 NVLink-C2C（900 GB/s）实现了 CPU 与 GPU 的原生高带宽内存一致性互联。

第二章 跨节点通信网：从网卡进化到三层网络

当数据流必须跨越物理服务器机柜时，通信方式便从内部总线跨入网络世界。这里存在着三种网络协议（TCP/IP、RoCE、InfiniBand）与三种网卡形态（Basic NIC、SmartNIC、DPU）的交织进化。

2.1 跨节点互联网络（IB vs. RoCE vs. TCP）代际演进表

RoCE、IB和TCP网络的差异对比




特性	InfiniBand (IB)	RoCE v1	RoCE v2	iWARP
协议栈 (传输层)	IB transport protocol	IB transport protocol	IB transport protocol	iWARP* protocol
协议栈 (网络层)	IB network layer	IB network layer	UDP/IP	TCP/IP
协议栈 (链路层)	IB link layer	Ethernet link layer	Ethernet link layer	Ethernet link layer
性能/时延	最好 (<2us)	与 IB 相当 (<5us)	与 IB 相当 (<5us)	稍差 (受TCP影响)
成本	高	低	低	中
稳定性	好	较好	较好	差
交换机要求	专用 IB 交换机	以太网交换机	以太网交换机	以太网交换机
标准定义组织	IBTA	IBTA / IEEE/IETF	IBTA / IEEE/IETF	IEEE/IETF




(注：带宽单位为网络标准的单端口双向全双工比特率)

年份	对应 GPU 架构	InfiniBand (IB) 标准及双向速率	RoCE 以太网标准及双向速率	传统 TCP/IP 常见双向速率	节点间网络核心演进与 AI 背景
2014	Kepler / Maxwell	FDR IB ~112 Gbps	RoCE v1 ~80 Gbps	10 Gbps ~10 Gbps	早期探索期：RoCE v1 因无法跨三层路由很少用于大集群。AI 训练主要靠 FDR InfiniBand。
2016	Pascal	EDR IB ~200 Gbps	RoCE v2 ~200 Gbps	25 Gbps ~25 Gbps	RDMA 爆发期：RoCE v2 诞生，引入 UDP 封装支持三层路由。EDR IB 成为第一代 DGX-1 的标配。
2020	Ampere	HDR IB ~400 Gbps	200 GbE RoCE ~400 Gbps	100 Gbps ~100 Gbps	千卡集群时代：DGX A100 标配 HDR IB。RoCE v2 在互联网巨头（如 Meta）的自研无损以太网中大放异彩。
2022	Hopper	NDR IB (Quantum-2) ~800 Gbps	400 GbE RoCE ~800 Gbps	200 Gbps ~200 Gbps	万卡集群时代：NVIDIA 推出 Spectrum-X 平台，用专属交换机和网卡调优，彻底解决以太网 RoCE 易丢包的痼疾。
2024	Blackwell	X800 IB (Quantum-X800) ~1.6 Tbps	Spectrum-X800 RoCE ~1.6 Tbps	400 Gbps ~400 Gbps	十万卡集群时代：单端口飙升至 800G。NVL72 机柜在内部消灭了 IB，但在机柜之间依然通过 X800 横向扩展。
2.2 网卡形态的政治与技术演进：从 SmartNIC 到 DPU

随着网络速率攀升至 800G/1.6T，如果仅靠服务器 CPU 来解析网络包，CPU 会因过载而“猝死”。网卡经历了三次进化：

传统网卡 (Basic NIC)： 纯粹的传话筒。所有的网络协议栈解析、重传、安全校验全靠主机 CPU，内耗极大。
智能网卡 (SmartNIC - 如 ConnectX 系列)： 引入了硬件卸载引擎。将 RDMA 协议、GPUDirect 技术直接固化在网卡 ASIC 芯片上，实现了网卡直连显存的“零拷贝”奇迹。如果是基于RoCe，我们叫它SmartNIC，如果是基于IB，我们叫它HCA(Host Channel Adapter). ConnectX-8 开始称作SuperNIC。
数据处理器 (DPU - 如 BlueField 系列)： 网卡里被直接塞进了一个多核 ARM CPU + 独立的 Linux 系统。它变成了服务器的“第二大脑”，负责将大模型集群中的安全隔离、网络加密、分布式存储虚拟化（NVMe-oF）这三大杂活 100% 从服务器 CPU 中剥离出来，实现了真正的基础设施层“零内耗（Zero Tax）”。




特性 / 维度	第一代：传统网卡 (Basic NIC)	第二代：智能网卡 (HCA/SmartNIC)	第三代：数据处理器 (DPU)
典型代表芯片	早期的 ConnectX-3 / X-4	ConnectX-5 / X-6 / X-7 / X-8	BlueField-1 / 2 / 3 / 4
核心硬件架构	纯 ASIC 固化硬件流水线	ASIC + 硬件卸载引擎（如 ASIC 化的 RDMA）	ASIC + 强大的多核 ARM CPU + 硬件加速器
可编程性	无（完全固死，无法更改）	极低（仅能通过驱动配置有限的规则）	极高（运行独立的 Linux 操作系统，可自由编程）
CPU 卸载程度	0%（所有的网络协议栈全靠主机 CPU）	50%（卸载了 RDMA、系统校验、基础流控）	100%（网络、存储、安全、虚拟化全部剥离）
在 AI 集群中的角色	简单的网络收发通道	AI 计算平面核心（跑 NCCL/RDMA）	AI 算力工厂的“大管家”（兼顾计算、存储、安全）
核心超能力	把数据发出去	硬件级零拷贝（GPUDirect RDMA）	网络隔离、基础设施层完全独立
第三章 SHARP

NVIDIA 提出的 SHARP (Scalable Hierarchical Aggregation and Reduction Protocol，可扩展分层聚合和规约协议) 技术，核心理念是“网络即计算”(In-Network Computing)。 [1] 传统架构中，网络交换机只负责数据包的“盲目转发”，而 SHARP 技术直接打破了这一物理界限，让数据在通过交换机和网卡（DPU/HCA）的“路上”就把 AI 训练最核心的梯度规约求和（AllReduce/Reduce）给算完了。 [1, 2]

具体实现可以拆解为以下三个层面的协同工作：

3.1. 物理载体：谁来算？怎么算？

在 SHARP 架构中，计算任务被下放到了原本不具备逻辑计算能力的专用网络芯片中：

交换机芯片 (Switch ASIC)： 如 Quantum-2 (InfiniBand) 交换机内部集成了专用的硬件数学计算引擎。这些引擎支持向量算术逻辑单元 (ALU)，能够直接对通过该交换机的高速数据流进行整数和浮点数（包括 INT8, FP16, BF16, FP32, FP64）的 Sum（求和）、Min/Max（极值）等算术操作。
网卡与 DPU (BlueField-3 / ConnectX-7)： 服务器端的网卡或 DPU 负责数据的切片、封装、乱序重排以及控制流的编排。它们利用 NVIDIA DOCA 软件栈 与交换机握手，确保数据流能够以最佳的格式“喂”给交换机的计算引擎。 [1, 3, 4, 5, 6, 7]

3.2 拓扑逻辑：流式聚合树 (Streaming-Aggregation Tree)

SHARP 最精妙的地方在于，它将整个 AI 算力集群（数百台 DGX 服务器）的网络拓扑映射为一棵或多棵逻辑聚合树 (Aggregation Tree)。 [4, 8]

[ 根交换机 (Root Switch) ]  <-- 最终在这里算出最终的总和结果 (并广播下发)
         /           \
 [ 交换机 A ]      [ 交换机 B ]  <-- 在这几台交换机“路过”时完成局部求和
  /        \        /        \
[DPU 1]   [DPU 2] [DPU 3]  [DPU 4] <– 计算节点通过 DPU 注入原始梯度数据    
(GPU 1)   (GPU 2) (GPU 3)  (GPU 4)
传统网络（节点间肉搏）： 4 台服务器要算 AllReduce，必须两两之间把数据包跨网络互相复制、在自己的 GPU/CPU 内部算完、再发给下一个，数据流在网络上来回激荡，不仅带宽减半，延迟也由于数据多次进出显存而极高。
SHARP 网络（沿途截流计算）：
所有参与 AI 训练的计算节点（GPU），在完成反向传播后，通过 DPU 同时且只注入一次 梯度数据到网络中。
当数据流经过第一层 NVIDIA Quantum 交换机 时，交换机不需要等待数据包完全接收，而是采用 Streaming-Aggregation（流式聚合） 机制。
交换机一边把来自多个端口的数据包（如节点 1 和节点 2 的梯度）并行解包，一边将对应的向量数据送入内置的计算芯片直接进行规约求和。
算好的“中间局部和”被重新打包，继续往上一层交换机（树的根节点）发送。 [1, 2, 3, 8, 9, 10]

当数据到达聚合树的根部（Root）时，全集群的梯度总和已经计算完毕。随后根交换机直接将最终结果硬件多播（Multicast）广播回所有节点的 DPU 并送回 GPU，从而一举完成了 AllReduce。 [1, 4, 5, 11]

3.3 核心技术优势（为什么大模型训练极度依赖它？）

根据 NVIDIA 开发者官方技术分享 与 Lambda Labs 的全评测报告，SHARP 带来了颠覆性的硬件红利：

网络流量减半 (Reduce Data in Motion)： 因为数据是在向上传输的过程中“边走边合并”，所以每往上一层交换机，传输的数据量就会呈指数级缩减（如 2 个变 1 个）。相比传统软件算法，整个网络中的总流量直接砍掉了一半。
零延迟抖动 (Eliminate Jitter) 与扁平化延迟： 规约操作全部在数据交换的硬管道中以线速（Line-rate）完成。随着集群规模从 100 台扩大到 1,000 台，其 AllReduce 的延迟几乎是一条平直线（保持极其微小的恒定延迟），彻底解决了大规模网络拥堵和时延抖动问题。
释放宝贵的 GPU 计算资源： 传统的集合通信算法（如 Ring-AllReduce 或树状算法）需要消耗 GPU 大量的串行流处理器（SM）以及主机 CPU 资源来进行内存搬运和代数求和。在引入 SHARP（以及最新的 NCCL 2.27 深度适配）后，GPU 可以完全不用分心，把 100% 的算力全部留给最核心的大模型前向/反向矩阵乘法。 [1, 3, 8, 9, 12, 13]

最新演进：从 IB 到 NVLink

在最新的 Blackwell 世代（如 DGX GB200 NVL72）中，NVIDIA 将这一套“网络计算技术”不仅应用在以太网/InfiniBand 等外网交换机（如 Quantum-X800）上，还深度向下兼容到了机架内部的 NVSwitch 芯片中。这意味着在万卡级超算集群里，无论是机柜内的高速 NVLink，还是机柜间的跨物理网络，都在全面贯彻“数据未到、网络先算”的理念。 [11, 12]

[1] https://developer.nvidia.com [2] https://lambda.ai [3] https://www.youtube.com [4] https://mug.mvapich.cse.ohio-state.edu [5] https://docs.nvidia.com [6] https://resources.nvidia.com [7] https://developer.nvidia.com [8] https://docs.nvidia.com [9] https://link.springer.com [10] https://www.youtube.com [11] https://developer.nvidia.com [12] https://lambda.ai [13] https://developer.nvidia.com [14] https://developer.nvidia.com [15] https://docs.nvidia.com

第四章 产品形态与全景通信层级架构

基于上述底层组件，NVIDIA 衍生出了不同的算力交付形态：单独卖主板的 HGX、卖整机的 DGX、以及直接打包整张机柜的划时代作品 GB200 NVL72。

4.1 HGX vs. DGX vs. NVL72 物理剪影

【HGX 模块】 ─── 纯 GPU 算力主板 (8*GPU + 内部 NVSwitch) -> 卖给戴尔/超微等 OEM

【DGX 服务器】 ── HGX 主板 + 标配 x86 CPU 节点 + 外壳 (机柜间通信强依赖 InfiniBand 网卡)

【NVL72 机柜】 ── 18*Compute Tray + 9*NVSwitch Tray + 5000根铜缆背板 (逻辑上的单体 72-GPU 巨无霸)

4.2 NVL72
Component Category	Exact Count / Spec (NVL72 Standard)	Notes
Racks	1 or 2 Racks	Single-rack is 120kW+; dual-rack splits the footprint.
Compute Nodes (Trays)	18 Trays	Standard 1U liquid-cooled server form factor.
CPUs	36 Grace CPUs	2 CPUs per compute tray.
GPUs	72 Blackwell GPUs	4 GPUs per compute tray.
Switch Trays	9 Trays	Placed centrally in the rack to optimize cable lengths.
NVSwitch Chips	18 ASIC Chips	2 chips per switch tray.
NVLink Cables	5,000 Cables	Completely copper, pre-mapped inside a structural spine.

注：每个switch有72个port，9个switch tray有18个switch，一躬18*72个port。72个gpu每个有18个NVLink port，就正好对上了。

第五章 SuperPOD
SuperPOD 世代	核心节点系统	节点单机 GPU 规格	单节点显存容量	计算网络架构 (Compute Fabric)	存储/管理网络	单个标准计算单元 (SU) 规模
第一代 (Ampere)	DGX A100	8x NVIDIA A100	320GB / 640GB HBM2e	8x HDR InfiniBand (200 Gbps)	2x HDR InfiniBand / 200GbE	140台 DGX A100 (初期) / 后期多采用 32台一组
第二代 (Hopper)	DGX H100	8x NVIDIA H100	640GB HBM3	8x NDR InfiniBand (400 Gbps)	2x NDR InfiniBand / 400GbE	32台 DGX H100 节点 (含 256颗 H100 GPU)
第三代 (Hopper)	DGX H200	8x NVIDIA H200	1,128GB HBM3e	8x NDR InfiniBand (400 Gbps)	2x BlueField-3 DPU / 400GbE	32台 DGX H200 节点 (含 256颗 H200 GPU)
第四代 (Blackwell)	DGX B200	8x NVIDIA B200	1,440GB HBM3e	8x NDR InfiniBand (400 Gbps) / 第五代 NVLink	BlueField-3 DPU / Spectrum-4 800G Ethernet	8台 DGX B200 节点 (液冷集群化演进)
第五代旗舰 (Blackwell 架构超算)	DGX GB200 (NVL72)	72x Tensor Core GPU + 36x Grace CPU	单机架高达 13.5TB Fast Memory	第五代 NVLink (1.8TB/s 双向) + Quantum-X800 IB	Spectrum-X800 Ethernet (800 Gbps)	以机架(Rack)为基本 SU 单位，原生支持万卡级扩展

技术演进要点解析

网络跃升： 从第一代的 200 Gbps (HDR) 演进至目前的 400Gbps (NDR) / 800Gbps (X800 系列)，并深度引入了 NVIDIA BlueField-3 DPU 卸载管理和存储流量。
显存与带宽： 显存从 A100 时代的几百 GB，飙升至 B200 / GB200 世代的 TB 级 HBM3e，GPU 间互连的 NVLink 5.0 带宽已达 1.8TB/s，专门针对万亿参数大模型的训推一体化进行了设计。
统一软件栈： 历代 SuperPOD 均标配 NVIDIA Base Command Manager 运营与编排软件，实现开箱即用的 AI 工厂体验。
