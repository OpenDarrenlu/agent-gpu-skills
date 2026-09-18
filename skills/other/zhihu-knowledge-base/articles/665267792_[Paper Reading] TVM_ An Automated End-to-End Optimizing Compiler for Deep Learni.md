# [Paper Reading] TVM: An Automated End-to-End Optimizing Compiler for Deep Learning

**作者**: Mmaxwell想做一点微小的工作

**原文链接**: https://zhuanlan.zhihu.com/p/665267792

---

最近仔细阅读了一下 TVM 这篇经典论文，在这里简单写一下自己的理解，做个记录，希望能帮助到需要的同学，也欢迎大佬指出其中的错误。

paper 的整体行文逻辑是：提出需要解决的问题，如何将深度学习负载高性能地部署到不同的设备上-->分析当前框架存在的不足-->提出兼具 graph- and operator-level optimization 的TVM(introduction)-->介绍TVM的整体架构(overview)-->详细介绍 graph-level optimization(optimizing computational graph)-->详细介绍 operator-level optimization，这里分为两部分，一部分是如何为一个给定的 operator 生成具体实现(generating tensor operations)，另一部分是如何在众多具体实现中挑出最优者(automating optimization)-->最终通过一系列实验证明TVM强大的能力 。

接下来详细介绍一下 paper 的每一节。

Introduction

首先介绍一下背景，深度学习模型解决问题的能力强，应用越来越广泛。很自然地诞生了需求，将这些深度学习应用部署到不同的设备上。但是由于硬件后端的多样化，将深度学习负载部署到不同的设备上是很困难的，这就是需要解决的问题。

其次对现有的工作进行分析，找到其不足。具体来说，现有的深度学习框架如 TensorFlow、PyTorch等，可以将深度学习负载部署到一小类服务器级GPU设备上，这里存在的不足就是部署的设备范围很有限。这些框架依赖于计算图进行图级别的优化，这种高层次的优化无法利用硬件后端特性。为了利用硬件后端特性，这些框架需要用到供应商提供的算子库，这些算子库的构建和优化需要大量的人工调试，并且不能在不同的硬件后端上通用，这里存在的不足是需要为不同的硬件后端提供 operator libraries，工程量巨大。

然后作者针对这些不足，提出了能够为不同硬件后端提供图级别和算子级别优化的框架TVM。这里给出作者对TVM的描述： We built TVM, a compiler that takes a high-level specification of a deep learning program from existing frameworks and generates low-level optimized code for a diverse set of hardware backends. 为了获得堪比手工优化算子库的性能，需要解决以下挑战：

如何利用硬件特性：对于一个给定的算子，如何生成最优的代码，进而有效利用硬件原语达到加速效果；对于缺少控制流的硬件，如何在编译栈进行有效调度。
如何搜索巨大的优化空间：众多的优化选择，例如循环展开、循环分块等，构成了巨大的优化空间，如何进行有效地搜索至关重要。

那么TVM是如何解决这些挑战的呢？它定义了三个关键模块：

TVM 引入了 tensor expression language 用来表示算子，并且引入了一系列 transformation primitives 用来产生不同版本的算子实现。
TVM 引入了 automated program optimization framework 来进行优化空间探索。
不太明白这里的 graph rewriter 是什么

最后总结了一下文章的贡献，并且吹了一下实验效果。文章主要有以下几点贡献：

确定了优化不同硬件后端部署深度学习负载面临的主要挑战：算子级别的优化需要手工进行
引入了新的 schedule primitives
引入了基于机器学习的优化系统，进而能够自动且高效探索优化空间
提供了能够将深度学习负载部署到不同硬件后端的端到端编译栈
Overview

这一章介绍了 tvm 的整体架构，大致介绍了端到端流程。下图描述了 tvm 的系统架构：

System overview of TVM

tvm 的输入是主流格式的预训练模型。首先将模型转化为计算图，tvm 中用 relay 来描述计算图，对计算图进行优化；其次将计算图中的每个 operator 表示成张量表达式，到这一步还没有确定每个 operator 如何在硬件上进行计算，只是描述了计算规则；然后通过添加 primitives 来对 operator 的具体执行进行优化，用机器学习的方法对优化空间进行探索；最后可以得到能够部署的模块。

Optimizing Computational Graphs

计算图可以用来表示深度学习程序。计算图提供了程序的全局视角，但是并没有详细说明每个算子是如何实现的。TVM中，计算图的每个节点表示一个算子，每条边表示节点间的数据依赖关系。这里给出一个计算图的例子：

Example computational graph of a two-layer convolutional neural network

可以在计算图上进行一系列图级别的优化，文中详细介绍了 operator fusion 和 data layout transformation。

算子融合(Operator Fusion)：算子融合简单讲就是将多个算子融合为一个算子，通过减少对中间结果的存储和读取来提高性能。论文将图算子分为四类：injective；reduction；complex-out-fusable；opaque，之后又在分类的基础上提出了融合的基本规则，例如：多个 injective 算子可以融合为一个 injective 算子；reduction 算子可以和输入的 injective 算子融合；complex-out-fusable 算子可以和输出的 element-wise 算子融合。

Data Layout Transformation：这个优化主要讲的是如何在内存中存储一个给定的 tensor。首先指定每个算子想要的数据布局，当生产者和消费者之间的数据布局不匹配时，就进行数据布局转换。

虽然图优化能够大幅提升性能，但终究还是受限于算子库的能力。当图优化进行算子融合产生的新算子不能被算子库所支持时，就不能取得预期的性能提升。现在主流的框架通过人工对新算子进行支持，但随着网络中算子数量增加以及硬件后端类型增加，人工维护算子库的方法不再可行。基于这个问题，TVM提出了对指定算子自动生成代码的方法。

Generating Tensor Operations

承接上节，TVM 为一个算子生成多种实现并从中选出最佳实现，进而生成代码。那么是如何生成多种实现的呢？基本思想基于 Halide 提出的描述与计算规则分离，首先对算子进行描述，其次对描述添加一系列的转化（即添加 schedule primitive），不同的转化序列对应不同的实现。

Tensor Expression and Schedule Space

首先介绍了什么是 Tensor Expression。Tensor Expression 描述 output tensor 的shape，给出一个表达式描述output tensor 中的每个元素如何计算，但是并不包括循环结构或者具体执行细节，这就是 compute/schedule decouple 中的 compute 部分。下图是一个矩阵乘的 Tensor Expression 表示：

Tensor expression example

对于 compute/schedule decouple 中的 schedule 部分，TVM 中的 schedule 表示一种特定的 Tensor Expression 到 low-level code 的映射。TVM 通过对一个 Tensor Expression 逐步添加 schedule primitive 得到一个 schedule。其实 Halide 就已经提出了 schedule primitive，TVM 在这个基础上的主要增强是支持了更多的 schedule primitive 以在更多的硬件后端上进行优化。接下来的三个小节介绍了新添加的三种schedule primitive。

Nested Parallelism with Cooperation

这一小节主要讲的就是 tvm 如何在 schedule 中实现对目标硬件memory的利用。

GPU 提供了大量的并行手段，这就给作者抛出了难题，如何在 schedule transformation 描述并行模式？大多数现存的解决方案会采用 nested parallelism 这个模型，该模型通过一个 schedule primitive 来将一个数据并行任务划分为多个并行子任务，子任务再递归被划分，直到能够充分利用目标架构的多级线程层次结构。

之后举例说明利用嵌套并行性的一个例子是，所有线程一起协作取数据。具体的例子就是，gpu进行矩阵乘法时，一个线程组内的所有线程共同取数据，存进 shared memory，从而能够利用 gpu 的内存层次结构（线程访存shared memory的速度>>访存 global memory），提高数据在不同线程间的复用，详细 cuda 代码如下：

matmul cuda example

结合这张图，应该容易看懂 cuda 代码，该图来源于https://blog.csdn.net/qq_37764141/article/details/122609942

论文中的那段代码实在是没看懂。。。。

举例之后，作者介绍了一下如何设计schedule。作者在 schedule 中加入了 memory scope 来表示当前 scope 是共享的还是线程局部的，如果是共享的，那么必须计算所有组内线程的依赖关系。同时，必须能够在合适的地方插入内存同步操作。引入了 memory scope 这个概念后，还能够更好地利用专用加速器的内存层次结构。

Tensorization

DL workloads 通常可以拆分为一系列的 tensor operators，近些年的工作诞生了一系列相应的 tensor compute primitive 来计算这些 operators。这些新诞生的 primitive 要如何集成进 schedule-based 的 tvm 框架进而提升性能呢？因为 tvm 要支持很多种不同的硬件后端，不同的硬件后端有不同的 tensor 指令，所以 tvm 不能只简单支持一系列固定的 primitive。作者提出的解决方案就是采用一种可拓展的方式。

可拓展方案的主要想法是通过 tensor-intrinsic declaration 将 hardware intrinsic 和 schedule 分离。声明一个tensor intrinsic 要做两件事：通过张量表达式声明它的行为；定义对应的 lowering 规则，也就是如何将张量表达式做的事转化为硬件后端的intrinsic。下面是声明 gemm8x8 的代码：

有了这种可拓展方案，就可以加入 tensorize schedule primitive 将计算转化为对应的 intrinsic。具体的转化过程就是编译器匹配计算模式和 intrinsic 声明，如果匹配的话，就 lower 为对应的 intrinsic。

Explicit Memory Latency Hiding

首先介绍了一下 latency hiding 这个概念，通过重叠执行内存和计算操作，最大化利用内存和计算资源。对于不同的硬件后端，需要不同的策略。CPU 通过 simultaneous multithreading 实现 latency hiding；GPU 通过调度 warp 实现 latency hiding；TPU 通过 decoupled access-execute 架构实现 latency hiding，并且把细粒度的同步问题交给软件端。下图表示的就是 DAE pipeline：

Decoupled Access-Execute Pipeline

其中值得注意的是，DAE 架构中的指令流有一些用于同步的操作。例如 ld0 后一条 push 操作，表示 ex0 相对于ld0 的 RAW 依赖可以满足；ld1 之后隔了一段时间才开始 ld2，是因为 ld2 执行之前首先要检查 ld2 对于 ex0 的 WAR 是否能满足，如果不满足，就会被阻塞，直到 ex0 执行完后再执行 push 操作，进而触发 ld2，这实际上也保证了 ex1 能读到正确的 ld1 数据，而不是 ld2 的数据。这些操作都是由编译器插入进指令流的。

在对 DAE 架构加速器进行编程时，需要显式进行低层次的同步是很困难的。为了降低编程难度，作者引入了虚拟线程这个概念，通过使用 virtual threading scheduling primitive，编程者可以像在提供 multithreading 的硬件上编程。virtual threading 的使用可以参考下图代码：

左侧代码是虚拟线程并行代码；通过在每个线程内插入同步操作进而保证正确性，就得到了中间的代码；最终将所有虚拟线程的操作汇聚到单一指令流，最右侧代码的执行流程就如上上图 DAE pipeline 所示。

Automating Optimization

这一节详细介绍了 automated optimization framework，该框架的作用是 Schedule Explorer 在 ML Cost Model 的指导下逐步生成更优的 schedule，整体框图如下：

Overview of automated optimization framework

大致的工作流程就是，每一轮 Schedule Explorer 根据 ML Cost Model 选出一系列更优的 schedule 放到 device 上真实运行，得到运行数据，用来训练 ML Cost Model，之后进行下一轮。

Evaluation

通过一系列实验回答了四个问题：

• Can TVM optimize DL workloads over multiple platforms? —— 在 Server-Class GPU、Embedded CPU、Embedded GPU、FPGA accelerator 上进行端到端测试

• How does TVM compare to existing DL frameworks on each back-end？—— 取得一定的性能提升

• Can TVM support new, emerging DL workloads (e.g., depthwise convolution, low precision operations)?

• Can TVM support and optimize for new specialized accelerators? —— test on VDLA

Reference：

Chen T, Moreau T, Jiang Z, et al. {TVM}: An automated {End-to-End} optimizing compiler for deep learning[C]//13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18). 2018: 578-594.
