# AI Systems Performance Engineering02--AI系统硬件概述

**作者**: 想飞的石头带着团队在agi大潮里做挖掘机的

**原文链接**: https://zhuanlan.zhihu.com/p/2041492859497493131

---

章节概述

想象将超级计算机的AI硬件压缩到单个机架中。NVIDIA的最新架构正是如此。本章深入探讨NVIDIA如何将CPU和GPU融合为强大的超级芯片，然后用超快互连将数十个连接起来，创建"机架中的AI超级计算机"。我们将探索基本硬件构建块——Grace CPU和Blackwell GPU——并了解它们的紧密集成和巨大内存池如何使AI工程师的工作更轻松。

2.1 CPU和GPU超级芯片
2.1.1 超级芯片概念

NVIDIA扩展AI的方法始于单个组合CPU + GPU超级芯片模块。从Hopper代开始，NVIDIA开始将ARM CPU与一个或多个GPU封装在同一单元中，用高速接口紧密连接。结果是一个表现得像统一计算引擎的单一模块。

演进历程：

Grace Hopper (GH200)：1个Grace CPU + 1个Hopper GPU
Grace Blackwell (GB200)：1个Grace CPU + 2个Blackwell GPU
2.1.2 NVLink-C2C互连

特点：

Grace CPU和Blackwell GPU之间高达~900 GB/s
PCIe Gen5 x16：~64 GB/s每方向
PCIe Gen6 x16：~128 GB/s每方向
NVLink-C2C比典型PCIe快一个数量级
缓存一致性

缓存一致性：

CPU和GPU共享一致的统一内存架构
始终看到相同的值
GPU可直接读写CPU内存中的数据
无需显式复制
2.1.3 内存配置

GB200 Superchip内存：

组件	容量	带宽
Grace CPU LPDDR5X	~480 GB	~500 GB/s
Blackwell GPU HBM3e (×2)	~384 GB	~8 TB/s
总计	~900 GB	-

统一内存优势：

单个超级芯片近1TB快速统一内存
模型可无缝利用CPU内存作为扩展
神经网络层或大型嵌入表可驻留在CPU内存
GPU仍可通过NVLink-C2C访问

性能权衡：

LPDDR5X访问：比HBM低约10×带宽，更高延迟
智能运行时将最频繁使用的数据保留在HBM
CPU LPDDR5X用于溢出或速度关键性较低的数据
2.2 NVIDIA Grace CPU
2.2.1 设计特点

架构：

ARM Neoverse V2 CPU
NVIDIA定制设计
针对带宽和效率优化

规格：

72个CPU核心
高达~500 GB/s LPDDR5X内存带宽
超过100 MB L3缓存
适度时钟速度
2.2.2 角色和功能

主要任务：

处理通用任务
预处理并向GPU提供数据
管理附加的大量内存

优势：

永远不会成为向GPU推送数据的瓶颈
可从存储流式传输数据
执行即时数据转换（如tokenization、数据增强）
通过NVLink-C2C高效向GPU提供数据

CPU-GPU协同：

CPU扩展GPU在GPU较弱领域的能力
随机内存访问
控制密集型代码
GPU加速CPU无法跟上的数值计算
2.3 NVIDIA Blackwell"双die"GPU
2.3.1 多芯片模块(MCM)设计

架构：

不是单芯片
两个GPU die放在单个模块中
超快片上die-to-die互连

原因：

单片die受制造限制
硅芯片尺寸有限制
组合两个物理GPU die可加倍模块的总晶体管预算
2.3.2 规格对比

Blackwell B200 vs Hopper H100：

特性	Hopper H100	Blackwell B200
晶体管数	~800亿	~2080亿
内存	80 GB HBM3	192 (180可用) GB HBM3e
内存带宽	~3.35 TB/s	~8 TB/s
L2缓存	50 MB	126 MB

B200组成：

每GPU die：约1040亿晶体管，96 GB HBM3e
组合模块：约2080亿晶体管，192 GB总内存
8个HBM3e栈（每die 4个）
每栈24 GB（8个3 GB DRAM die垂直堆叠）
2.3.3 NV-HBI互连

特点：

专门的10 TB/s die-to-die互连
让两个GPU die作为单个统一GPU运行
软件层只看到一个GPU
NVIDIA软件和调度确保工作在两个GPU die之间平衡

可用内存说明：

192 GB中只有180 GB可用
原因：ECC、系统固件使用、制造限制
2.4 NVIDIA GPU Tensor Cores和Transformer Engine
2.4.1 Tensor Cores

定义：

GPU每个流多处理器(SM)内的专用单元
可极高速执行矩阵乘法操作

Blackwell增强：

支持更多数值格式
包括极低精度格式（8位和4位浮点）
2.4.2 Transformer Engine (TE)

功能：

自动调整并使用混合精度
关键层使用高精度（FP16或BF16）
不太关键的层使用FP8
自动优化精度平衡以保持模型精度

精度演进：

代际	精度支持	相对加速
Hopper	FP8	2× vs FP16
Blackwell	FP8 + NVFP4	2× vs FP8
2.4.3 FP4优势

计算吞吐量：

整个NVL72机架（72 GPU）FP4理论吞吐量超过1.4 exaFLOPS
单个机架进入世界最快超级计算机领域

内存节省：

FP4比FP8节省一半内存
FP8比FP16节省一半内存
可将更大模型打包到GPU内存
2.4.4 实际性能对比

1.8万亿参数MoE模型推理：

系统	吞吐量	首token延迟
H100系统	~3.4 token/s/GPU	>5秒
Blackwell NVL72	~150 token/s/GPU	~50毫秒

加速原因：

原始FLOPS
更快GPU
低精度(FP4)使用
NVLink互连保持GPU数据供应
2.5 流多处理器、线程和Warp
2.5.1 GPU架构层次

流多处理器(SM)：

GPU的"核心"
每个SM包含：
算术单元（FP32、INT32等）
Tensor Cores用于矩阵数学
加载/存储单元用于内存操作
特殊功能单元（超越数学）




Warp：

固定大小线程组
每个warp恰好32个线程
以锁步执行完全相同的指令
单指令多线程(SIMT)执行模型
2.5.2 延迟隐藏

原理：

SM并发执行许多活跃warp
如果一个warp等待内存获取，另一个warp可运行
高端GPU如Blackwell有数百个SM
每个SM可并发运行数千个线程
单个GPU上数万个活跃线程
2.5.3 内存层次结构

层次（从快到慢）：

寄存器（每线程）
共享内存（每线程块，在每个SM上）
L1缓存（每SM）
L2缓存（GPU上所有SM共享）
HBM内存（片外）

性能原则：

数据需要尽可能保持在层次结构高处
每次操作都访问HBM（即使8 TB/s）也会导致GPU频繁停顿
通过将可重用数据保留在SM本地内存或L2缓存中，GPU可实现巨大吞吐量
2.6 超大规模网络：将多个GPU视为一个
2.6.1 NVL72系统概述

组成：

72个Blackwell GPU
36个Grace CPU
NVLink互连
单个机架中的AI超级计算机

结构：

18个计算节点
每节点2个GB200/GB300 Superchip
每计算节点共4个Blackwell GPU + 2个Grace CPU
2.6.2 NVLink和NVSwitch

NVLink 5规格：

每GPU 18个NVLink 5端口
聚合双向NVLink带宽：1.8 TB/s每GPU
18个NVLink链路 × 100 GB/s双向
是Hopper NVLink 4的两倍

NVSwitch：

专门为NVLink构建的交换芯片
9个交换机托盘
每托盘2个NVSwitch芯片
共18个NVSwitch芯片

网络拓扑：

全交叉连接
每个GPU连接到每个NVSwitch
每个NVSwitch连接到每个GPU
任意GPU对之间单跳路径
聚合双分带宽：~130 TB/s
2.6.3 多GPU编程

编程模型：

一个GPU可通过NVLink直接访问另一个GPU内存
使用点对点和分区全局地址空间(PGAS)模型
NVIDIA SHMEM (NVSHMEM)
全局地址空间，但GPU缓存在GPU间不全局一致

GPUDirect RDMA：

网络接口控制器(NIC)可注册GPU内存
直接执行RDMA到GPU内存
GPU跨节点交换数据
无需CPU参与
NIC直接DMA到GPU内存
无需通过主机RAM暂存
2.6.4 性能对比

NVL72 vs 传统InfiniBand集群：

指标	NVL72	传统IB集群
GPU间带宽	高达1.8 TB/s	20-80 GB/s
延迟	1-2微秒	5-10+微秒
集合开销	百分之几	百分之几十

设计建议：

尽可能将工作负载通信保持在机架内（"机架内"）
利用高速NVLink和NVSwitch硬件
仅在绝对必要时使用较慢的InfiniBand或以太网通信
2.7 NVIDIA SHARP网络内聚合
2.7.1 SHARP功能

定义：

Scalable Hierarchical Aggregation and Reduction Protocol
可扩展层次聚合和归约协议

工作原理：

网络内归约使用集成到NVSwitch ASIC的SHARP引擎
从GPU卸载归约和其他集合操作到交换机硬件
NVSwitch结构组合部分结果
数据无需通过GPU
2.7.2 优势

效率提升：

GPU专注于更复杂的计算
降低集合延迟
减少穿越网络的总体数据量
提高系统效率

扩展性：

即使GPU数量增长也能看到接近线性的性能改进
对训练超大型模型特别关键
每个在集合操作上节省的微秒都可转化为显著的总体加速

历史：

NVIDIA在2019-2020年收购Mellanox时获得
可显著减少集合的延迟和流量
通常改善通信受限训练的扩展效率
2.8 多机架和存储通信
2.8.1 外部连接

网络接口卡：

高速NIC
数据处理单元(DPU)

BlueField-3 DPU功能：

卸载、加速和隔离网络、存储、安全和管理任务
线速数据包处理
RDMA
NVMe over Fabrics (NVMe-oF)
直接在网络、存储和GPU内存之间移动数据
无需CPU参与
2.8.2 网络配置

Quantum-X800 InfiniBand或Spectrum-X800 Ethernet：

每计算节点4个ConnectX-8 800 Gb/s NIC
每节点3.2 Tbit/s
每机架57.6 Tbit/s（18节点）

多机架扩展：

8个NVL72机架共576 GPU
使用NVLink Switch System作为一个NVLink 5域连接
InfiniBand或Ethernet连接该NVLink域到其他域
2.8.3 预集成机架设备

特点：

作为预集成机架"设备"交付
单个机柜中组装所有18个计算节点
所有9个NVSwitch单元
内部NVLink布线
配电系统
冷却系统

部署：

连接机架到设施电源
连接水冷接口
连接InfiniBand电缆到网络
开机即可使用

管理软件：

NVIDIA Base Command Manager集群管理软件
SLURM
Kubernetes
2.9 共封装光学：网络硬件的未来
2.9.1 技术趋势

背景：

网络数据吞吐量攀升至800 Gbit/s、1.6 Tbit/s及更高
NVIDIA开始将硅光子学和共封装光学(CPO)集成到网络硬件

平台：

Quantum-X800 InfiniBand
Spectrum-X800 Ethernet
800 Gb/s端到端连接
网络内计算功能（如SHARP）
2.9.2 CPO优势

技术特点：

光发射器直接集成在交换硅旁边
大幅缩短电路径
实现机架间更高带宽链路
降低功耗
提高整体通信效率

未来影响：

连接数百和数千机架（AI工厂）
单一统一结构
机架间带宽不再是瓶颈
确保网络能跟上GPU的超大规模需求
2.10 计算密度和功率需求
2.10.1 功耗规格

NVL72功耗：

满载时高达~130 kW
比NVIDIA上一代AI机架（50-60 kW）高2×以上
72个尖端GPU + 所有支持硬件

功率分配：

18个计算节点：每节点6 kW，共110 kW
NVSwitch托盘、网络交换机、空气冷却、水冷泵：~20 kW
总计：130 kW
2.10.2 电源设计

冗余：

多个高容量电路
两个完全独立的电源馈线
每个馈线尺寸可承载整个机架负载
一个馈线故障时，剩余电路可支持全部130 kW

电源管理：

专用配电单元(PDU)
仔细监控
电容器或排序避免大电压降
交错GPU升频时钟平滑浪涌
2.10.3 机架重量

规格：

约3000磅（1.3-1.4公吨）
填充硬件和冷却液时
大约是一辆小汽车的重量，但集中在几平方英尺地板上

部署考虑：

架空地板数据中心需检查地板承重
高密度机架放置在加固板上
需要叉车等特殊设备移动
2.11 液冷与风冷
2.11.1 液冷必要性

问题：

130 kW在单个机架中
传统风冷无法处理
72个GPU每个可散热~1200瓦
需要飓风般的气流
极其嘈杂和低效
热空气排气难以处理
2.11.2 NVL72液冷设计

组件：

每个Grace Blackwell Superchip模块有冷板
每个NVSwitch芯片有冷板
冷板：内部有管道的金属板
水基冷却液流过管道带走热量
软管、歧管和泵循环冷却液

连接：

每节点快速断开连接器
可滑入或滑出服务器而不泄漏冷却液
机架有供应和返回连接到外部设施的冷水系统
2.11.3 冷却液分配单元(CDU)

功能：

内置或紧邻机架
热交换器
将热量从机架内部冷却液回路传递到数据中心水回路

温度范围：

设施提供20-30°C冷水
水通过热交换器吸收热量
温水返回冷却器或冷却塔再次冷却
可使用温水冷却（30°C进，45°C出）
蒸发冷却塔无需主动制冷
2.11.4 液冷优势

性能：

GPU和CPU温度远低于风冷
GPU可持续最大时钟而不触及温度限制
运行芯片更冷提高可靠性
较低温度时功耗泄漏更低

温度范围：

GPU温度在负载下保持50-70°C
对如此耗电的设备来说非常出色

流量估算：

约150-200升/分钟
10-12°C水温上升
散热约130 kW
2.12 性能监控和利用率实践
2.12.1 监控工具

NVIDIA Data Center GPU Manager (DCGM)：

跟踪每个GPU的指标
GPU利用率百分比
内存使用
温度
NVLink吞吐量
2.12.2 监控要点

GPU利用率：

理想：训练作业期间接近100%
如果50%利用率：某事物使其空闲一半时间
可能是数据加载瓶颈或同步问题

NVLink使用：

如果NVLink链路频繁饱和
通信可能是瓶颈

BlueField DPU和NIC统计：

确保读取数据时不饱和存储链路
2.12.3 功率监控

重要性：

~130 kW，即使小低效或配置错误也会浪费大量功率和金钱
监控每节点或每GPU功耗

优化策略：

如果不需要每一点性能，可降低GPU时钟
提高效率（性能每瓦）
仍满足吞吐量要求
节省千瓦级功率
数周训练可节省大量成本
2.13 共享和调度
2.13.1 工作负载分区

集群调度器：

SLURM
Kubernetes + NVIDIA插件
可划分GPU子集给不同用户

示例：

8 GPU给一个用户
16 GPU给另一个用户
48 GPU给第三个用户
都在同一机架内
2.13.2 多实例GPU (MIG)

功能：

将单个物理GPU分割成更小的GPU
硬件级分区
每个Blackwell GPU最多7个完全隔离的MIG实例
专用内存和SM

用例：

推理场景
在一个GPU上服务多个模型
安全多租户
DPU作为防火墙和虚拟交换机
隔离不同作业和用户的网络流量
2.13.3 成本管理

资产价值：

NVL72是数百万美元资产
每月消耗数万美元电力
需要做尽可能多的有用工作（goodput）

利用率跟踪：

GPU使用小时 vs 可用小时
如果系统利用率低，合并工作负载或提供给更多团队

计费模型：

内部团队用自己的预算按GPU小时付费
鼓励高效使用
考虑电力和折旧成本
2.14 升级硬件的ROI
2.14.1 性价比分析

案例研究：

当前：100个H100 GPU处理工作负载
升级：50个Blackwell GPU（每个快2×以上）
购买50个而非100个GPU
即使每个Blackwell比H100贵，购买一半可能成本中性或更好
2.14.2 功率节省

对比：

100个H100：~70 kW
50个Blackwell：~50 kW（相同工作）
显著功率节省
一年可节省数万美元
2.14.3 其他节省

减少开销：

更少GPU = 更少服务器维护
更少CPU、RAM、网络开销
进一步节省

ROI时间：

某些情况下1-2年回本
特别是有足够工作让它们24小时忙碌
2.14.4 软性收益

简化：

使用单个强大系统而非多个较小系统
简化系统架构
提高运营效率
降低功耗
减少网络复杂性

竞争优势：

不必因内存限制将模型分割到多个旧GPU
简化软件
减少工程复杂性
拥有最新硬件确保可利用最新软件优化
跟上升级的竞争对手
2.15 NVIDIA路线图展望
2.15.1 Blackwell Ultra和Grace Blackwell Ultra

B300 GPU改进：

内存容量：288 GB（比B200的180 GB多50%）
AI计算性能：1.5×更高
更大的片上加速器（专为注意力操作和NVFP4设计）
推理吞吐量：比B200高45-50%

GB300 NVL72规格：

36个Grace Blackwell Ultra模块
~20.7 TB HBM（72 × 288 GB）
~18 TB DDR（36 × 500 GB）
总计~38 TB快速内存每机架
使用相同NVLink 5代
2.15.2 Vera Rubin Superchip (2026)

命名：

Vera：女性天文学家，其工作提供了暗物质证据
Rubin：GPU架构，Blackwell继任者

规格：

Vera CPU：ARM架构，Grace CPU继任者
Rubin GPU：Blackwell GPU架构继任者
一个Vera CPU + 两个Rubin GPU
TSMC 3nm半导体工艺
更多CPU核心
更快LPDDR6内存（~1 TB/s）
Rubin GPU HBM：~13-14 TB/s
NVLink 6：CPU-to-GPU和GPU-to-GPU链路带宽翻倍

可能扩展：

每机架更多节点
每NVLink域更多机架
超越GB200/GB300 NVL72集群的576 GPU限制
2.15.3 Rubin Ultra和Vera Rubin Ultra (2027)

架构变化：

四die GPU模块
组合两个双die Rubin封装
R300 Rubin Ultra GPU模块：四个GPU die在一个封装上
16个HBM栈，共1 TB HBM内存
双倍B300模块的核心数

Vera Rubin NVL144：

机架中144个die
36个超级芯片模块，每模块4个die

Vera Rubin NVL576：

4× GPU数量
多die封装完整系统

预期性能：

每机架3-4 exaFLOPS计算性能
组合165 TB GPU HBM RAM（288 GB每Rubin GPU × 576 GPU）
2.15.4 Feynman GPU (2028)

预期：

更精细的2nm TSMC工艺节点
HBM5
模块内更多DDR内存
可能从4个die翻倍到8个

推理优化：

推理需求将主导AI工作负载
推理需要比以前非推理模型多数百或数千倍推理时计算
芯片设计可能优化规模推理效率
更多新精度
更多片上内存
片上光学链路进一步提高NVLink吞吐量
2.15.5 持续翻倍模式

NVIDIA翻倍历史：

Blackwell：双GPU die（两die每模块而非一个）
NVLink双向带宽每链路从900 GB/s翻倍到1.8 TB/s
每GPU内存从Blackwell的180 GB增加到Blackwell Ultra的~288 GB
Rubin和Feynman进一步增加计算、内存和带宽

AI工厂愿景：

NVIDIA反复谈论AI工厂
机架是AI模型的生产线
通过合作伙伴提供机架即服务
公司可租用超级计算机的一部分而非自建
每代允许换入新pod以加倍容量、提高性能、降低成本
关键要点总结
1. 集成超级芯片架构
NVIDIA将ARM CPU（Grace）与GPU（Hopper/Blackwell）融合到单个超级芯片
创建统一内存空间
消除CPU和GPU之间手动数据传输的需求
2. 统一内存架构
统一内存架构和一致互连减少编程复杂性
开发者无需担心显式数据移动
加速开发并帮助专注于改进AI算法
3. 超快互连
使用NVLink（包括NVLink-C2C和NVLink 5）和NVSwitch
系统实现极高的机架内带宽和低延迟
GPU通信几乎像是一个大型处理器的部分
对扩展AI训练和推理至关重要
4. 高密度超大规模系统(NVL72)
NVL72机架在紧凑系统中集成72个GPU
组合设计支持大规模模型
高计算性能与巨大统一内存池结合
使传统设置上不切实际的任务成为可能
5. 先进冷却和功率管理
NVL72依赖复杂的液冷和强大配电系统
每机架约130 kW运行
对管理高密度、高性能组件和确保可靠运行至关重要
6. 显著性能和效率提升
相比前代（如Hopper H100），Blackwell GPU提供约2-2.5×计算和内存带宽改进
训练和推理速度显著提升
某些情况下推理快达30×（使用Blackwell FP4 Tensor Cores和Transformer Engine）
通过减少GPU数量潜在节省成本
7. 现代软件栈支持
NVIDIA软件和框架持续演进以充分利用最新硬件
支持最新协同设计系统优化
包括统一内存管理和原生FP8/FP4精度支持
工程师可用最小代码更改利用系统全部性能
8. 面向未来的路线图
NVIDIA开发路线图（包括Blackwell Ultra、Vera Rubin、Vera Rubin Ultra和Feynman）
承诺持续翻倍关键参数如计算吞吐量和内存带宽
轨迹旨在支持未来更大的AI模型和更复杂的工作负载
结论

NVIDIA NVL72系统——及其Grace Blackwell超级芯片、NVLink结构和先进冷却——体现了AI硬件设计的前沿。在本章中，我们看到每个组件都经过协同设计以服务于加速AI工作负载的单一目标。CPU和GPU融合为一个单元以消除数据传输瓶颈并提供巨大的统一内存。数十个GPU用超快网络连接，使它们表现得像一个巨大的GPU，通信延迟最小。内存子系统得到扩展和加速，以满足GPU核心的巨大需求。甚至配电和热管理也被推向新高度以允许这种计算密度。

结果是单个机架提供以前只在多机架超级计算机中看到的性能。NVIDIA采用整个计算栈——芯片、板卡、网络、冷却——并端到端优化，以允许超大规模训练和服务大规模AI模型。

但这种硬件创新带来挑战——需要专用设施、仔细规划功率和冷却、以及复杂软件来充分利用它们。但回报是巨大的。研究人员现在可以以前所未有的规模和复杂性实验模型，无需等待数周或数月获得结果。在旧基础设施上可能需要一个月训练的模型可能在NVL72上几天内完成训练。以前几乎不交互（每查询数秒）的推理任务现在是实时（毫秒级）现实。这为以前不切实际的AI应用打开了大门，如万亿参数交互式AI助手和代理。

NVIDIA的快速路线图表明这只是开始。Grace Blackwell架构将演变为Vera Rubin和Feynman及更远。正如NVIDIA CEO Jensen Huang所描述的："AI正在以光速前进，公司正在竞相建设可扩展以满足推理AI和推理时缩放处理需求的AI工厂。"

NVL72及其继任者是AI工厂的核心。这是将处理大量数据以产生惊人AI能力的重型机械。作为性能工程师，我们站在这些硬件创新的肩膀上。它给了我们巨大的原始能力，我们的角色是通过开发充分利用硬件潜力的软件和算法来利用这种创新。

在下一章中，我们将从硬件转向软件。我们将探索如何优化NVL72等系统上的操作系统、驱动程序和库，以确保这些惊人硬件不会被利用不足。在后续章节中，我们将研究补充软件架构的内存管理和分布式训练/推理算法。

本书的主题是协同设计。正如硬件是为AI协同设计的，我们的软件和方法必须协同设计以利用硬件。现在有了对硬件基础的清晰理解，我们准备好深入研究提高AI系统性能的软件策略。AI超级计算时代已经到来，充分利用它将是一段激动人心的旅程。
