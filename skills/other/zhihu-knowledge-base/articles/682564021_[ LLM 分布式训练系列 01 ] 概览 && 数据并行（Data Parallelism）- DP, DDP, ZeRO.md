# [ LLM 分布式训练系列 01 ] 概览 && 数据并行（Data Parallelism）- DP, DDP, ZeRO

**作者**: Alan 小分享​香港科技大学 资讯科技硕士

**原文链接**: https://zhuanlan.zhihu.com/p/682564021

---

​
目录
收起
一、概览
二、DP 和 DDP
2.1、算法
2.2、分析
2.3、仍然存在的问题
三、ZeRO
3.1、直观感受一下
3.2、显存都用在了哪些地方？
3.3、Zero-DP：优化 model stats 所需空间
3.4、分析和对比
3.5、ZeRO-R、ZeRO-Offload 和 ZeRO-Infinity

【本文是 “LLM 分布式训练系列” 的第 1 篇，持续更新中】：

[ LLM 分布式训练系列 01 ] 概览 && 数据并行（Data Parallelism）- DP, DDP, ZeRO

[ LLM 分布式训练系列 02 ] 流水线并行（Pipeline Parallelism）- GPipe

一、概览

最近几年，无论是大语言模型，还是视觉模型，都是朝着“变大 && 数据变多 -> 就可以变强”的思路发展。然后单卡已经无法装下整个模型，所以需要将数据和模型切分到多块 GPU 上进行训练。

在 LLM 分布式训练这个系列，我打算记录一下目前主要的几种并行方法：

流水线并行（Pipeline Parallelism）
数据并行（Data Parallelism）
张量并行（Tensor Parallelism）

实际训练的时候，一般是其中 2 种或者 3 种方法同时用～～

我们先来简单认识下这些方法。（可以先留个印象

下面是这几类并行方法的计算流程图：

简单说下思路：

（1）单 GPU（最原始的方式）

就是直接在单卡上加载整个模型，逐个 batch 跑；

（2）Data Parallelism

把每个 batch 拆分成 n 个 micro-batch，然后发给 n 块 GPU；每块 GPU 做完整的模型的运算，最后得到 n 份梯度，加起来，再更新参数。

这里核心是每块 GPU 都跑完整个模型～～但是模型参数不一定是每块 GPU 都保存一份～～

对于 DP 和 DDP，就是每块 GPU 都把整个模型都加载到自己的显存；
对于 ZeRO，就是模型参数就拆分成 n 份，每块 GPU 存一份，用到的时候，再通过 All-Gather 操作，从其他 GPU 取过来；ZeRO 也是 DeepSpeed （微软推出的分布式训练和推理框架）的核心算法之一；

（3）Pipeline Parallelism

把模型按层切分（就是横着切成多份），每块 GPU 负责一份；数据也是拆分成 n 个 micro-batch，然后依次送进 GPU，流水线执行；

（4）Tensor Parallelism

把每一层切分开（就是竖着切成多份），每块 GPU 负责一份；然后整个 batch 发给每块 GPU，分别负责部分计算，中间按需要会对中间结果进行聚合（通过 All-Reduce 操作）。




关于上面提到的 All-Gather 和 All-Reduce 操作，如果忘记了，可以看看这里：




接下来，我们就来聊聊 Data Parallelism；




二、DP 和 DDP

DP：Data Parallelism；

DDP：Distributed Data Parallelism；

2.1、算法

首先，这两种方法的运行过程都是：

每块 GPU 去加载完整的模型；
把一个 batch 拆分成多个 micro-batch，分别发给 n 块 GPU；
每个 GPU 做完计算后，得到一份梯度；
把所有梯度进行累加，然后用于更新参数；

区别在于第 4 步：

DP 中，会有一个 server 节点（可以是其中一个 GPU 进程，或者单独的一个节点），负责收集所有梯度，累加，然后将结果发给其他 GPU（称为 worker 节点）；每个 worker 再用来更新本地的参数；（当然也可以是 server 更新好参数后，再分发）
DDP 中，则是每个节点都会参与收集梯度、累加、更新参数这个过程，而不是依靠单一节点；
具体来说，就是把需要通讯的数据切成 n 份（n 为 GPU 数量），然后通过 Ring-AllReduce 的方式，使得每块 GPU 都得到累加后的梯度，然后就可以更新本地的参数了。

DP 一个比较经典的实现就是 Parameter Server [1]，感兴趣的同学可以康康～～

2.2、分析

DP 中有一个最明显的问题就是通过单一 server 节点进行梯度聚合，此时 server 的带宽就会称为瓶颈 ；

接着 DDP 就把通讯方式改成了 Ring-AllReduce，解决了通讯负载不均衡的问题；

关于 Ring-AllReduce 的过程和分析，可以看看这篇文章：




来简单看看这两种方法的通讯量～～～（关注的是训练一个 batch 所需要的通讯量）

假设模型参数 W 的数量为 \Psi ，GPU 数量为 N；则需要通讯的梯度的数量也是 \Psi；

（1）DP

每块 GPU 需要发送 \Psi 个梯度数据给 server，然后 server 再把 \Psi 个结果发回去，所以总的通讯量是 2N\Psi；

但是 server 承担了一半的接收和一半的发送任务；

（2）DDP

首先，梯度被切分成 N 份，即每份大小为 \frac{\Psi}{N} ；然后进行 Ring-AllReduce 通信，分为两步：Reduce-Scatter 和 All-Gather；

看看单块 GPU 的情况：

Reduce-Scatter 阶段，需要进行 N - 1 轮通讯，每一轮发送和接收 \frac{\Psi}{N} 个数据，所以通讯量为 (N - 1) \frac{\Psi}{N}；
All-Gather 阶段，过程类似，通讯量也是 (N - 1) \frac{\Psi}{N}；

所有 GPU 的通讯量加起来就是 2(N - 1) \frac{\Psi}{N}N ，即 2(N-1)\Psi ；

（一般也可以近似为 2N\Psi ）




小总结

两者的总通讯量其实很接近，而 DDP 的优点是把通讯负载均衡地分配到每个 GPU 上～～




2.3、仍然存在的问题

还有一个很严重的问题，每个 GPU 都要加载完整模型，那就很难把模型做大了，这可不行～

所以有了后来的 ZeRO；




三、ZeRO

ZeRO [3] 是微软在 2020 年提出的一种数据并行的方法，主要解决了模型参数重复加载的问题，使得可以训练更大模型，且通讯量只是 DP / DDP 的 1.5 倍。

3.1、直观感受一下

ZeRO 和 DP、DDP 相同的地方：每块 GPU 依旧负责整个模型的所有运算；

不同的地方：每一层的参数都拆成 N 份，分别存在 N 块 GPU 上，用到的时候，再通过 All-Gather 操作获取；




举个简单例子，模型现在有三层（La, Lb, Lc），且我们有 3 块 GPU，则将每一层的参数分成 3 份（根据 GPU 数量确定分为多少份）：（例如第一层 La 的参数分为 a0, a1 和 a2）

La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2

然后对于每一层的参数，每块 GPU 保存一份：

GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2

运行时，每个 batch 切分为 3 份 micro-batch：

x0 => GPU0
x1 => GPU1
x2 => GPU2

计算过程：

计算 La 前，每块 GPU 通过 All-Gather 操作，从其他 GPU 获取 La 的完整参数，然后计算，用完以后就可以把多余的参数丢掉（即不是自己负责保存的参数）；
类似地，计算 Lb 和 Lc；
backward 计算时，也是类似的过程；




接下来我们再一步步看细节～～～




3.2、显存都用在了哪些地方？

论文先是给我们分析了显存主要用在了哪些地方，主要分为两类：model states 和 residual stats。

（1）model stats：必须存储的数据

包括：

optimizer stats：比如 Adam 中的 momentum 和 variances；
parameters：指模型参数，主要是 weights 和 bias 等；
gradients：模型参数对应的梯度；

（2）residual stats：运行过程中产生的额外空间占用

包括：

activation：即中间结果；
temporary buffers；
显存碎片；

然后，论文将优化 model stats 占用空间的方法称为 Zero-DP，将优化 residual stats 的方法称为 Zero-R。

那么具体需要使用多少显存呢？

前情提要，为了提高训练速度，训练大模型的时候一般会使用混合精度训练[4] ～

即我们保存一份 fp32（单精度浮点数，存储占 4 bytes）格式的 parameters（主要是 weights 和 bias 等）。计算每一个 batch 前，转换为 fp16，运行 forward 和 backward 过程（中间数据都是用 fp16 格式），得到 fp16 格式的 gradients，然后用于更新 fp32 格式的 parameters。

过程如下图：（把图中的 weights 看成是 parameters 就好）

来自论文 [4]

如果我们的 optimizer 使用 Adam，还需要另外存 fp32 格式的 momentum 和 variance。

论文中以 Adam 为例，分析了显存占用情况：

假设 parameters 数量为 \Psi

fp32 部分，需要存 parameters，momentum，variance，共 3\Psi 个数据，每个占 4 bytes，即 12\Psi bytes；
fp16 部分，需要存 parameters 和 gradient， 共 2\Psi 个数据，每个占 2 bytes，即 4\Psi bytes；

所以总共需要 16\Psi bytes 的空间；

注：论文中称 fp32 的部分为 optimizer stats，称其所占空间为 K\Psi bytes（这个例子中 K = 12）；下文中也会用到这两个记号。




下面的分析中，都是假设模型中使用 Adam 作为 optimizer～～～




3.3、Zero-DP：优化 model stats 所需空间

当我们使用混合精度训练时，model stats 包含的数据可以细分为：

然后就可以按照 3.1 中的思路，对这些数据进行拆分，然后保存到多个 GPU 上。用到的时候，再通过 All-Gather 获取，或者通过 All-Reduce 来更新。

论文中是一步步进行拆分的：

假设有 N 块 GPU～

（1） P_{os} ：只拆分 optimizer stats

即 optimizer stats 拆分为 N 份，每块 GPU 保存一份。

另外，每块 GPU 仍然保存全部的 fp16 格式的 parameters 和 gradients；

计算过程：

每个 GPU 使用本地的 fp16 parameters 计算一份 micro-batch 数据，得到 fp 16 gradients；
通过 All-Reduce 操作，聚合 gradients，并更新各自的 optimizer stats（fp32）；
每个 GPU 用自己的那份 optimizer stats，更新对应的 fp16 parameters（每块 GPU 只更新了其中一部分）
通过 All-Gather 操作，获取所有更新后的 fp16 的 parameters；




现在，每块 GPU 的显存占用，就从 4\Psi + K\Psi bytes，变成 4\Psi + \frac{K\Psi}{N} bytes 了。




（2）P_{os} + P_g ：拆分 optimizer stats 和 gradients

单卡显存占用进一步降低 为 2\Psi + \frac{(2 + K)\Psi}{N} bytes 。




（3）P_{os} + P_g + P_p ：拆分 optimizer stats, gradients, parameters

现在，这些参数的存储位置，就从 DP 中每块 GPU 存全部数据（下图虚线左侧），变成了下图右侧的样子，每块 GPU 存其中一部分：

计算过程：

将当前 batch 切分为 N 份 micro-batch，每个 GPU 拿一份；
forward 阶段中，一层一层地计算；计算每一层前，通过 All-Gather 操作，每块 GPU 都获取到当前层完整的 parameters(fp16)；计算完后，就可以将不属于当前 GPU 负责的 parameters(fp16) 丢弃了；然后计算下一层；
backward 阶段中，也是一层一层计算；计算每一层前，通过 All-Gather 操作 获取 parameters(fp16) ，用于计算梯度；参数也是用完就丢；
接着，对所有 GPU 上的 gradients(fp16) 做一次 Reduce-Scatter 操作，使得每块 GPU 都可以获得自己维护的那部分梯度的累加和；
最后，分别更新自己维护的那份 optimizer stats(fp32)，再用其中 fp32 格式的 parameters 去更新本地的 fp16 的 parameters。

此时，单卡的显存占用，就变为\frac{(4 + K)\Psi}{N} bytes 了～～

即从最初的 16\Psi bytes，变为\frac{16\Psi}{N} bytes。




3.4、分析和对比

2.2 中，我们分析到，对于 \Psi 个数据的通讯，单次 All-Gather 和单次 Reduce-Scatter 操作的通讯量都是 (N - 1) \frac{\Psi}{N}，一般可以近似为 \Psi。那么依次 All-Reduce 操作的通讯量，很自然就是 2\Psi ～

再结合前面的分析和计算过程，可以得到各个方法的显存占用和通讯量如下：

	显存（单卡）	显存例子（来自论文）
K=12， Ψ=7.5B，N=64	单卡通讯量
DDP	(2 + 2 + K) Ψ	120GB	2Ψ
P(os)	(2 + 2 + K / N) Ψ	31.4GB	3Ψ
P(os + g)	(2 + 2 / N + K / N) Ψ	16.6GB	2Ψ
P(os + g + p)	(2 + 2 + K ) Ψ / N	1.9GB	3Ψ




通讯量只变成了 DDP 的 1.5 倍，但是单卡内存只是原来的 1/60 ！！！就说秀不秀吧




3.5、ZeRO-R、ZeRO-Offload 和 ZeRO-Infinity

这部分的思路不复杂，所以这里简略说下～～对细节感兴趣的朋友们可以看看原文；（原文中对应的篇幅也不长）

（1）Zero-R

这是对 residual stats 进行优化的方法的统称；

比如对于 activation（即中间结果），可以通过 checkpointing（或者叫 recomputation） 的方式，节省内存；即算完就丢掉一部分，后面算梯度的时候再重算。

（2）Zero-Offload

思路是可以将一部分数据放到 CPU 内存中～～

比如 optimizer stats 数据量大，但是计算量相对比较低，就可以放到 CPU 中进行存储，算完梯度后，把 fp16 的 gradients 传到 CPU，然后更新。

（3）Zero-Inifinity

也是和 Zero-Offload 类似的思路，找个远程存储来放参数，就可以无限扩展了；（当然通讯时间也会增加不少）。

看到最后的朋友们，求求点个赞吧！




Reference:

[1] https://web.eecs.umich.edu/~mosharaf/Readings/Parameter-Server.pdf

[2] How to derive ring all-reduce’s mathematical property step by step

[3] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

[4] MIXED PRECISION TRAINING

[5] Efficient Training on Multiple GPUs
