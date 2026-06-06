# Dynamic Cluster and Scheduler for Blackwell

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/1940858379905001339

---

前言：

本文旨在结合cutlass的源码，学习一下blackwell 架构的GPU的新特性如何在cutlass中应用。本期的主题是Blackwell 新引入的Dynamic Cluster(Perfered CTA cluster) 和Dynamic Scheduler(Cluster Launch Control)。

Dynamic Cluster

Nvidia 在Hopper 架构中开始引入CTA Cluster，通过将CTA 组成cluster实现更大的Tile，并利用TMA 的multicast 机制减少global memory 的访问量和节约L2 Cache的带宽需求。

如上图，对于一个4x4的cluster，对于没有multicast 的场景，A矩阵的global memory访问量为 
𝐶
𝑇
𝐴
_
𝑀
×
𝐶
𝑇
𝐴
_
𝐾
×
16
, 利用multicast，A tile 在M-dim拆分成4份，每一份再multicast 到对应的4个CTA中的smem，A矩阵的global memory访问量为 
𝐶
𝑇
𝐴
_
𝑀
×
𝐶
𝑇
𝐴
_
𝐾
×
4

对于一个1x2的cluster，对于没有multicast 的场景，A矩阵的global memory访问量为
𝐶
𝑇
𝐴
_
𝑀
×
𝐶
𝑇
𝐴
_
𝐾
×
2
 , 利用multicast，A tile在M-dim拆分成2份，每一份再multicast 到对应的2个CTA中的smem，A矩阵的global memory访问量为 
𝐶
𝑇
𝐴
_
𝑀
×
𝐶
𝑇
𝐴
_
𝐾
×
1

tma box shape 拆分为multicast份

可以看出，更大的cluster size 可以更好的减少global memory 的访问量，但是在先前的Nvidia Cute 实战-WarpSpecialization Gemm for Hopper的文章中，提到了在Hopper架构只能支持static cluster,一个GPC 包含18个SM, 如果cluster>2会存在GPC中部分的SM无法launch CTA组成cluster，造成SM资源的浪费，所以在Hopper架构上cluster 往往采用2x1或1x2。为了解决Hopper上Static cluster 的限制，在Blackwell 架构上引入了Dynamic Cluster，既Kernel在launch 的时候可以配置两种Cluster shape，一个大的perferred cluster(如4x4) 和一个小的fallback cluster(如2x1),从而解决static cluster 引起的sm资源浪费。

Programming Blackwell Tensor Cores with CUTLASS GTC 25 2025
launch config

由于kernel中可能同时存在2中不同的cluster，对应在kernel内构造TMA copy 的时候需要包含两种cluster 的情况，根据runtime 获取的cluster shape 选择对应cluster 的TMA copy。

构造两种cluster的tma copy
runtime 选择对应cluster shape的tma copy
Dynamic Scheduler

从Hopper开始Nvidia 推荐使用persistent 的方式来更好的隐藏prologue的延迟，在Hopper上Static Persistent Scheduler推荐launch的CTA 数目=SM的个数，Static Persistent Scheduler 假设当前的GPU所有的SM都说可用的，但是实际的场景中不能保证所有的SM都是独占的，比如deepseek v3中就分配了20个SM给通信。如下图，当一部分SM被其他kernel占用时，第一轮wave没有launch的CTA会造成不均衡的计算负载和无法隐藏的额外epilogue 和prologue开销。

https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_cluster_launch_control.md

为了runtime解决static scheduler造成的负载不均衡的问题，Blackwell 引入了新的clusterlaunchcontrol ptx指令，包含clusterlaunchcontrol.try_cancel 和clusterlaunchcontrol.query_cancel

ptx: clusterlaunchcontrol.try_cancel
ptx: clusterlaunchcontrol.query_cancel

clusterlaunchcontrol.try_cancel 指令的作用是尝试去停止还没启动的CTA cluster，成功了把停止的Cluster 中的第一个CTA id 记录到smem 上的16B 的CLC reponse 中，通过调用clusterlaunchcontrol.query_cancel 从CLC reponse 中动态的解析出下一个CTA id(blockIdx.x, blockIdx.y,blockIdx.z),以及这个CTA id是不是有效的。

Data parallel grid shape

static scheduler 静态launch grid = SM数据，每个SM 负责计算 tiles\_per\_sm = \frac{n\_tiles}{n\_SM}个tile ,dynamic scheduler launch grid 采用传统的data parallel 的方式，根据实时的sm 情况调用clc 指令去停止没有运行的cluster，并获取下一个计算的cta tile id。

dynamic scheduler

用一个比喻来解释dynamic schduler：高铁检票口排队，如果有一个检票口故障了，这个故障的检票口排队的人就会找其他排队最少的检票口继续排队，如果有新的检票口可以开放，其他检票口排队的人可以到新的检票口更快的检票。

需要注意的是，在blackwell 架构引入了scheduler warp 作为producer来获取每一轮循环的tile info，除scheduler warp 外的warp作为consumer 去获取producer 产生的tile info，所以需要额外的clc_pipeline控制。
