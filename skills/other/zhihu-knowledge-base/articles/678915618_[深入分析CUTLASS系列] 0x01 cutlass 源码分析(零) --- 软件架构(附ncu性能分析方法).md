# [深入分析CUTLASS系列] 0x01 cutlass 源码分析(零) --- 软件架构(附ncu性能分析方法)

**作者**: JoeNomad​​新南威尔士大学 信息技术硕士

**原文链接**: https://zhuanlan.zhihu.com/p/678915618

---

​
目录
收起
开篇
CUTLASS的软件架构
Overview
Device
Kernel
Threadblock
Warp&&Thread
如何分析cutlass的性能瓶颈
生成nsight compute(ncu)的report
在GUI中打开report
开篇

大家好，我是joe，上一篇文章中着重从high level的层面分析了cutlass这个工作，以及整个优化手段的overview，

接下来会step by step地剖析cutlass的各个组件以及优化手段，本文作为源码分析的第一篇，会先从整个软件架构&&调用链路来说明每个组件做了什么，并且分享一些debug性能问题的方法

本篇文章focus的内容：

分析cutlass的主要组件，软件架构，旨在能够让大家快速warm up，减少认知成本
提供debug的方法参考
CUTLASS的软件架构
Overview

本文会以cutlass的example 08为例(sm75架构上的矩阵乘法)，自顶向下地来梳理cutlass的软件架构

# example dir
cutlass/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu
从计算逻辑上可以分为两个部分:
MMA: 矩阵乘法的乘累加部分，覆盖范围是load(global)->store(shared)->mma(矩阵乘法的结果存在寄存器里)
Epilogue: 将矩阵乘法的结果拿到，并进行后续的计算(cast, fusion kernel如bias,relu等)&&搬出, 覆盖的范围是 (store(shared),如果重排的话) -> Epilogue compute -> store(global)
cutlass的软件分层跟GPU的硬件架构基本相同:

device(主要是host侧调用的代码) -> kernel(针对不同的workload, dispatch需要用到的mma和epilogue && 定义kernel的mma计算逻辑以及调用epilogue) -> threadblock(定义一个block内mma的计算) -> warp-> thread

对于理解cutlass的行为，我们主要关注device, kernel, threadblock即可，warp和thread里做的事情就是简单的矩阵乘法，会存在不同的实现如cudacore, dp4a(cudacore的simd指令)，tensorcore(不同sm的指令不同如fp16在sm75: m16n8k8，sm80: m16n8k16，注：前向兼容，即sm80的硬件支持sm75)

Device

在这一部分中，主要包括了host侧的调用代码，Arguments结构体的定义，因此我们在这一层可以把device的代码和host的代码隔离开，比如我们想要预编译一些cutlass kernel在其他项目中调用，我们就可以把这一层的host侧代码封装起来，在调用时link，就不需要用到nvcc了。

这一层中我们需要关注的文件是(针对turing_tensorop_gemm.cu, 后文相同,不再mark):

cutlass/include/cutlass/gemm/device/gemm.h

API从命名就可以很容易地理解

gemm device的主要api
can_implement: 主要是对iterator做check，这个地方涉及L/R matrix的vectorize loading，检查是不是符合align，比如iterA的align要求是16，K维度是24，K无法被16整除，这里就会报错
get_workspace_size: 只跟kernel外splitk有关(即把k维分成n份做reduce sum)，如果不起用splitk，workspace是0，在这个问题上多说两句，splitk可以分为kernel内规约和kernel外规约(或者同时启用)，是为了提高kernel的并行度，kernel内是把n份的结果存在shared memory中，然后做reduce sum，kernel内会引入一个信号量(semaphore)，确保每一份矩阵乘法计算完成。kernel外则是把n份的结果搬出存在global memory里，再用一个reduce cuda kernel把n份结果做一个sum，kernel外会引入另外一个cuda kernel的开销。这个在k维比较大，m,n维都比较小的时候会有收益，比如resnet的最后一层，input channel是512，但h,w只有7

剩下的api就不多说了，看名字就能明白了，代码很好理解

Kernel

这一部分中，主要包含了kernel的模版特化，以及kernel执行的主要逻辑，我们需要关注的文件是

cutlass/include/cutlass/gemm/kernel/default_gemm.h
cutlass/include/cutlass/gemm/kernel/gemm.h
default_gemm.h中的模版特化

模版特化没什么太多可说的，声明了threadblock中的mma以及Epilogue

我们再来看计算的主要逻辑，这个地方就可以映射到在本文刚开始提到的计算逻辑上的分层(##重要##)

gemm.h中的核心代码(mma部分)

由于我们这个case中没有用到splitk，所以会执行275行，这行代码做的事情就是load->compute，accumulators就是一个申请的register，在cuda中，local memory和register的声明方法是相同的，即c++中的定长数组声明，超出255就会去用local memory, nvcc会自己去做寄存器复用分析，对用户是不可见的。

gemm.h中的核心代码(epilogue部分)




当mma计算完成后，我们会把乘累加的结果送进epilogue里，做后续的计算，即351行。

linear_combination.h中的核心代码

我们以linear_combination为例(y = ax)来展开说明epilogue里做了什么:

在epilogue里我们把register里的乘累加结果拿到，做一些自定义的后处理，在这里我们会显式的做cast，保证计算中数据的dtype是一致的，ElementCompute即我们希望的计算dtype，ElementD即我们希望的output dtype，比如我们希望一个fp16的矩阵乘法在后面的bias add中用fp32的精度来计算，以确保不会超出表达范围，我们就可以在实例化时传参ElementCompute=float

Threadblock

在一层中有好几个文件我们都需要用到，但我们主要关注两个：

cutlass/include/cutlass/gemm/threadblock/default_mma.h
cutlass/include/cutlass/gemm/threadblock/mma_singlestage.h

##注意##，这里会存在一些不一致性，因为在example 08里，numstage给的是2，但是我个人觉得对于刚接触cutlass的同学来说，singlestage的逻辑非常清晰，numstage的个数即代表有多少个流水线，即上一篇文章中提到的多级流水线的概念，单流水线有助于我们去梳理cutlass的计算逻辑，我们可以把numstage改成1，就会走到singlestage的调用里

default_mma.h中的一个模版特化

在这里定义了mma的特化传参，以及threadblock的计算流(single,1/pipelined,2/multistage,N)，bank conflict free的下标计算就在ThreadMapA/B里，这里我们先不去关心优化手段的细节，后续的文章在详细说明，我们只要知道，在这里声明了如何load&&compute即可

我们跳转到MmaSingleStage中在来看:

singlestage中的核心代码

这里我们可以清晰地看到，其实就是load matrixA/B和mma计算，这里多说一下unroll这个操作

pragma unroll: 这是一个非常常见的优化手段，广泛的存在于各种编译器当中，做的事情就是把for循环展开，这样遍历的下标可以在编译期被尽可能地推断成常量，减少执行期再去计算下标引入的scalar mul,add等的开销
Warp&&Thread

这里就不过多展开了，主要是怎么算一个warp内的mma以及每个thread怎么写，本质上是指令的应用。不过大家可以花点时间关注一下global load的细节，如下:

memory_sm80.h中global load的细节

这里的@p，就是上一篇中提到的，用一个special register来判断这个地方是不是需要被load

本文至此，软件层的架构梳理就结束了，每个组件的优化细节会在后续的文章中再进一步讲解~

下面会分享一些debug和分析的方法

如何分析cutlass的性能瓶颈

我们一般会用nsight compute来分析一个cuda kernel的性能

生成nsight compute(ncu)的report
# 编译kernel时需要加上-lineinfo flag，这样在report中才能看到sass对应的source，以sm80为例
nvcc -arch=sm_80 -lineinfo xxx.cu -I(需要的头文件)
# ncu profile 命令
# -o表示输出文件名，import-source表示我们希望源码在report里可以看到
# set full意思是说我们需要所有ncu提供的metrics
ncu -o xxx --import-source 1 --set full ./a.out
在GUI中打开report

查看report文件我们需要从英伟达官网下载并安装nsight compute的图形化界面

安装完成后我们用GUI打开report即可

report detail页面

在detail页面中我们会看到很多metrics，比如不同硬件利用率，指令计数等等都可以看到，教程可以参考

如果大家需要一些分析的知识，后续可能也会考虑详细的写一篇文章来讲解如何分析kernel的性能瓶颈

本篇文章至此结束，如果对大家有所帮助不妨点个赞呗~ ，我是Joe，是一名AI编译器从业者，如果大家对AI编译，mlsys感兴趣，可以关注一下哟，后续会继续分享CUTLASS的相关知识，也会考虑分享TVM，MLIR，量化算法，分布式推理等相关内容~

相关内容导览：



