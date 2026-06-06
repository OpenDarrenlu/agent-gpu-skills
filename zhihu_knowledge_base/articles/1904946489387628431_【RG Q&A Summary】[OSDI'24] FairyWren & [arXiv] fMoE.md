# 【RG Q&A Summary】[OSDI'24] FairyWren & [arXiv] fMoE

**作者**: USTC-NHPCC中国科学技术大学-国家高性能计算中心-先进数据系统实验室

**原文链接**: https://zhuanlan.zhihu.com/p/1904946489387628431

---

这里是中国科学技术大学 ADSL 实验室的系统论文阅读小组，我们每学期举办关于系统领域最新论文的阅读分享。本篇文章主要是对讨论过程中问答环节的总结。 Reading Group 的主页地址：ADSL Reading Group bilibili 链接：USTC-NHPCC的个人空间
FairyWren: A Sustainable Cache for Emerging Write-Read-Erase Flash Interfaces

FairyWREN (OSDI’24) 是 Kangaroo (SOSP’21 best paper) 的后续工作。Kangaroo 聚焦于使用闪存缓存海量小文件的场景，设计了基于”日志型闪存缓存“的 KLog，以及基于“组相连闪存缓存”的 KSet，同时获取了两种设计的优势，使得 Kangaroo 同时实现了”DRAM 占用量少“，”减少应用级写放大“与“高闪存空间使用率”的优势。而 FairyWren 在保留 Kangaroo 的组合设计理念的同时，引入WREN (Write Read Erase iNterface) 概念, 并统一缓存驱逐与SSD 垃圾回收流程设计了新的的算法，维持命中率的同时，大大减少了设备级写放大，以及对 FairyWREN 缓存的对象，进行了冷热分离，与大小分离，进一步减少了设备级的写放大和提升了闪存空间利用率，来满足文中提出的“更加低碳环保的目标”。

表1: FairyWREN 与其他闪存缓存设计目标的对比
背景

WREN (Write Read Erase iNterface)

在过去几十年中，闪存介质存储设备广泛普及，逐渐取代传统机械硬盘。闪存介质可以划分成多个存储区域，每个区域仅支持顺序写入，并要求在重新写入同样的物理块之前，需要对区域进行整个的擦除，这里区域被称作擦除单元 (erase unit)。传统闪存设备中，使用盘内闪存转换层 (Flash Translation Layer) 来向主机端暴露可以随机读和更新的块设备接口 (Read W)，对内管理地址映射，以及盘内垃圾回收等功能。而在近年来，业界推出了将 erase 接口暴露出来的新型接口固态硬盘 (e.g. ZNS/FDP SSD)。本文正是借此机会，来优化闪存缓存场景下的碳排放量指标。

FairyWREN 的设计目标是低碳环保，因此需要遵循以下几个设计原则：

尽可能少的 DRAM 使用开销
使用更高密度的闪存缓存
尽可能延长闪存设备的使用寿命

其中，高密度闪存缓存的使用寿命会降低，因此二三两点就落到了如何利用 WREN (Write Read Erase iNterface) 来降低原来闪存设计中的设备级写放大，实现更高的闪存设备使用寿命，并可以借此机会将介质替换为更高密度的闪存。

设计总览

FairyWREN 继承了 Kangaroo 中 ”日志型闪存缓存“，“组相连闪存缓存”的联合设计，并命名为FWLog，FWSet，在此基础上，针对WREN接口，将 FWSet 以日志型的形式存放在 WERN 接口的 ZNS SSD上。同时考虑了缓存对象大小分离，缓存对象冷热分离，以及设计了缓存驱逐与垃圾回收统一算法。来在Kangaroo 基础上，进一步降低了设备级写放大。

核心技术点
图1：nest packing 算法操作

缓存驱逐与垃圾回收统一算法

FWLog 或 FWSets 中任一部分即将用完空闲 EU (erase unit) 空间时，FairyWREN 需要执行 nest packing 操作（如图 1 所示）。FairyWREN 会选择 FWLog 或 FWSets 一个 Victim EU。若两者均已满，则优先选择 FWSets，因为 FWSets 必须预留空间，以接收从 FWLog 中迁移出的对象。

被选中的 EU 会首先被读取到内存中。若 EU。来源为 FWLog，则该 EU 中的每个对象都会通过哈希映射到 Victim Set；若来源为 FWSets，则该 EU 中的每个 Set 本身即为一个 Victim Set。随后，FairyWREN 会对每个 Victim Set 进行如下处理：

找出所有在 FWLog 中映射到该 Victim Set 的对象
构建一个包含这些对象的新Set（可以再次过程中驱逐对象）
将新 Set 追加写入 FWSets 中，
最后，对 Victim EU 执行擦除操作。

这篇文章最主要的亮点即是利用 WREN，将缓存驱逐算法与SSD垃圾回收算法的协同设计，其他设计细节可以参考原文。

fMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving

本文提出了fMoE，一种低延迟并且内存效率高的细粒度MoE侧载推理系统。MoE（Mixture of Experts）模型架构的模型因其庞大的知识容量以及在推理时对每个Prompt只需要激活部分参数，就可以获得较高的生成质量的特点在近年来受到了广泛的关注，但是同时由于其含有巨大的参数量，如果在推理服务时将参数全部加载到GPU中则会对推理系统的GPU内存容量提出较高的需求。现有的推理系统（如BrainStorm、MoE-Infinity等）提出了只在GPU内存中保存对当前输入激活的专家、将其他不活跃的专家放置在CPU内存中的侧载方案。好的侧载方案不仅要支持其系统的运行，还要提升服务的效率，需要提前预测当前输入会激活的对应专家并将其预取（Prefetch）到GPU中，但是现有的方案往往在预取专家时参考的历史数据是粗粒度、损失了细节信息的，导致预取的专家命中率不高，系统难以在推理延迟和GPU内存占用之间取得较好的平衡。

基于对之前系统问题的观察，fMoE提出了一种新的数据结构Expert Map——来细粒度地记录推理历史中每个iteration中模型每个层的专家激活概率分布，用无失真且细粒度的历史数据来指导之后遇见的输入的专家预取。在当前的输入数据与之前存储的Expert Map的模式的匹配上，fMoE采取了使用轨迹信息和语义信息结合的方法保证选取出准确用于指导预取的Expert Map：对于有足够前序层的模型层使用之前层记录的专家概率轨迹与存储的Expert Map进行匹配；对于无足够前序层的模型层使用当前输入的嵌入（embedding）语义信息来匹配合适的Expert Map。fMoE会动态地预取Expert Map中激活概率高的专家，结合高效的缓存管理以及Expert Map的去冗余机制，构建了一套专家命中率高、在推理延迟和GPU之间取得较好平衡的低额外开销的MoE推理系统。

论文中的评估结果显示，fMoE在6张Nvidia RTX3090上部署时，相比于之前的SOTA基准系统（MoE-Infinity、ProMoE等）将推理延迟降低了47%，将专家的命中率提升了36%。

Q&A

Q1: 请问一下前面实验里面，在prefill阶段，也就是TTFT的收益来自于哪个机制？对于稍微长一点的文本会导致prefiil阶段专家几乎全部激活了，然后实验中好像测出来不同的工作都有一定的差异。

A1: 在TTFT上面有提升是和其他offload系统进行对比，为什么收益不一样是因为MoE中存在计算不均衡性，更精确的prefetch不会造成额外的fetch开销，而且prefetch相对冷的expert还能和相对热的计算进行重叠。

Q2: fMoE保存了很多大量的历史数据信息，就是Iteration的概率分布，那这个每一次或者说每一次它都要与历史信息去做一次匹配的话。感觉这个时间开销会比较大，但是在这个图里面又解释说这个相对 inference 的时间它是比较小的，那这个inference时间是包含了prefill还是说只是decoding。

A2: 论文中并没有详细定义这个iteration指的是prefill还是decoding。如果是decode阶段，之所以时间占比开销看起来不明显，可能是因为使用的是消费级显卡RTX3090，其计算性能不够高。

Q2: 既然实验用到了六张卡，像这个并行策略，他是怎么分配的？

A2: 论文中没有明确交代并行策略。
