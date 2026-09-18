# [深入分析CUTLASS系列] 0x00 cutlass基本认知---为什么要用cutlass

**作者**: JoeNomad​​新南威尔士大学 信息技术硕士

**原文链接**: https://zhuanlan.zhihu.com/p/677616101

---

​
目录
收起
开篇
Prologue
为什么要用Cutlass，它解决了什么问题?
不同的实现路径选择的优缺点是什么
MAIN
密集计算优化问题描述
CUTLASS的实现好在哪里，为什么性能可以超越cudnn？
Epilogue
开篇

大家好，我是joe，作为一个AI编译器工作者，我的一部分工作重心是在解决密集计算的性能问题，有一些个人的经验，也在cutlass开源repo贡献过代码。cutlass作为目前GPU密集计算开源实现非常优秀的工作，广泛的应用在各大AI frame里，这个系列会从high level到low level全面分析cutlass这个工作，本文作为该系列的第一篇，会着重从high level的层面剖析为什么cutlass的性能优秀，旨在给大家一个宏观的感知。

Prologue

##前置知识##：

cutlass主要对运用tensorcore对密集计算进行优化，cuda是什么，tensorcore是什么，为什么要用tensorcore，GPU的硬件架构等知识已经有很多其他博主总结过了，此处不再过多赘述了~

为什么要用Cutlass，它解决了什么问题?

在深度学习框架中，一个非常需要解决的问题是，密集计算的性能问题，如conv2d，dense，matmul，transformers(attention)等，通常写一个极致优化的密集计算算子是非常不容易的，cutlass为我们提供了一个模板库的方式来实现高性能的密集计算算子，并且是完全白盒的，已经可以在很多case上超越cudnn了，白盒控制对于工程集成和正向研发来说非常重要!!

from github cutlass repo
不同的实现路径选择的优缺点是什么

目前我们一般有两种途径来完成一个高性能密集计算算子

基于DSL的编译器路径，如triton，tvm script
模块化的C++模板库，如cutlass

虽然他们在底层的优化思路大同小异，但是从整个优化链路上来讲是有非常大的区别的，DSL在软件架构上比较复杂，需要经过多层lower以及很多pass变换完成最终的代码生成，但每个pass做的事情比较清晰。模板库需要实现功能复杂的组件，但是软件架构比较简单。

对于第一种来讲，基于DSL的编译器路径在前端写算子的时候相对比较轻松，新增算子也比较容易，作为使用者来讲，不用太多去关注内部的一些实现以及领域知识（基于DSL本身的优化能力可以满足需求为前提）。如果优化能力不足需要自己做二次开发的话，认知成本和修改量是比较大的，比如新增IR的优化pass，增加新的指令codegen等等。我们用triton的代码作为DSL的示例(截取自官网):
此处用triton的softmax做解释，用很少的代码量就可以实现一个性能不错的kernel

2. 对于第二种而言，在写算子以及新增算子的时候就不那么容易了，也许需要实现新的组件来满足自己特定的需求（如fusion kernel），但由于整个库的软件架构比较简单，所见即所得，新增一些自己的优化思路等改动相对DSL来说比较容易，也更容易debug

MAIN
密集计算优化问题描述

针对于密集计算的优化，都可以归类于如何优化一个矩阵乘法，Conv2d(implicit gemm)，attention(batch matmul)

矩阵乘法的定义：Mat(M,K) x Mat(K, N) = Mat(M,N)

在硬件上的计算pipeline:

load(left matrix and Right matrix) -> 2. compute(mul and reduce sum) -> 3. write(global memory)
在cutlass中矩阵乘法的high level计算逻辑

想要达成的目标

最大化spatial 和 locality的数学期望(即较高的并行度和较好的局部性)，并且尽可能的打满访存带宽和tensorcore计算资源

CUTLASS的实现好在哪里，为什么性能可以超越cudnn？

这一part我会从high level的角度来讲解cutlass的性能为什么好，（每一个细节都会在后续的文章中详细展开叙述，此处先按下不表，感兴趣的铁铁们可以动动小手点点关注哈哈哈~）

因为我手上没有H100，所以此处都是基于A100的硬件架构来讲解的（focus在A100的tensorcore运用），H100的优化思路有所不同，但设计理念并不会有特别大的gap

比较重要的指令介绍

ldmatrix指令，搬运4个8x8的矩阵，从shared memory 到 local(register)

mma指令，计算一个小块的矩阵乘，比如mma.m16n8k16，会完成一个rowmajor (16,16) x colmajor (8, 16)的矩阵乘法

这些指令的用法以及解读都可以从NVidia的PTX ISA手册里获得:

所有优化手段都是围绕着这两个核心指令展开的

优化手段overview(按重要性排列)

bank conflict free的shared memory layout：cutlass首先提出的一种同时消除shared memory 存(global memory->shared memory)&&读(shared memory->register)的手段
block swizzle：这个优化对于中大型矩阵乘法比较明显，更改了发射block的顺序，以增加locality，从而提高l2cache的命中率，实现上非常简单，核心代码就是一个取余操作，但有用
多级流水线(software pipeline)：2条可以不要async.copy这个指令(sm80才有的)，大于2条流水就需要了，原理上没什么，和CPU的多级流水一个道理，主要是指令的应用。
predicate iterator：这个是一个软件层组件写法的优化，叫predicate的原因是，这个iterator会返回一个布尔值，在gpu的指令里是一个special register，用来表示这块内存是不是需要load，这个在软件层会涉及一些优化手段，比较有趣的是会在host侧precompute了哪些下标需要load，用位运算来mask，计算开销(位运算在gpu里开销较小)和存储开销(一个byte可以存8个mask值)都很小。为什么需要让存储开销很小？因为在gpu架构里，register是很贵的，一个thread只能使用255个register，如果超出了就会存在local memory里，register读取很快，一个cycle就可以完成，而local memory就会慢非常多，register用超了会非常非常影响性能！
shared memory重排搬出：mma指令计算完成之后，结果是存在register里的，且register中存储的数据是不连续的(32bits连续)，原因是由于mma指令造成的，我们知道vectorize load/store会提高访存带宽，所以我们可以在shared memory里重新排序，一并搬出。但并不是什么情况下重排都是正优化，因为重排还是会增加一次shared memory store/load，比如在小channel的conv2d中，直接从register搬出到global memory性能会更好
cooperative fetching和vectorize load：这两个是GPU的一些基本优化方法，即尽量用更大的data type来搬运，以及尽量让一个warp里的不同线程是连续地访存同一块内存地址，原理可以参考

7. tiling description: 提供了实例化方法，来调整block计算量和warp计算量，也就是说用模版参数来优化spacial，主要贡献在于给用户提供了一种自定义循环切分的方法，来定义循环切分的搜索空间，针对不同的workload搜索一个性能最优的选择，对于刚接触cutlass的同学而言，增加tiling description是一个比较容易的方法来提高kernel性能（因为原始的tiling确实太少了），在这个问题上多说一些，loop tiling是非常经典的编译器优化问题，目前除了polyhedral的方法以外都是tuning base的，只不过生成实例化的方法不同(预定义的options，机器学习base的搜索如基因算法(tvm-ansor)), 对于tensorcore的矩阵乘法优化问题，我们是有强先验的，即tensorcore的计算访存指令都是固定的(mma, ldmatrix)，相当于looptiling的子问题是一个确定的解，所以搜索空间并不会特别大(但也很大了。。)

Epilogue

CUTLASS的源码还是比较难读的，对于刚开始接触的同学建议把example里面的例子仔细看看，如果打算开发一个cutlass算子，可以先考虑从加一个cutlass的fusion kernel开始，比如已经知道了怎么跑通一个gemm算子，那么可以研究一下加一个gemm + bias_add + relu的融合算子应该怎么写(hint: 在Epilogue里加)。

CUTLASS的主逻辑(mma部分)其实能改的东西不多，不过也有可能需要做一些动作，一些debug的tips是通过nsight compute来分析这个kernel具体是memory bound还是compute bound，然后分析一下哪些指令bound了，是不是符合预期，然后再针对性的进行实验&&修改




感谢阅读，后续会不定期继续将这个系列补全，如果对大家有所帮助不妨点个赞呗~ ，我是Joe，是一名AI编译器从业者，如果大家对AI编译，mlsys感兴趣，可以关注一下哟，后续会继续分享CUTLASS的相关知识，也会考虑分享TVM，MLIR，量化算法，分布式推理等相关内容~







相关内容导览:
