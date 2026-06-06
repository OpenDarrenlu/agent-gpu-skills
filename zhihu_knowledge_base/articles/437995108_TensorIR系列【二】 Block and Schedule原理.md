# TensorIR系列【二】 Block and Schedule原理

**作者**: 冯思远​上海交通大学 计算机科学与技术博士

**原文链接**: https://zhuanlan.zhihu.com/p/437995108

---

关于TensorIR的Motivation和简介，详见TensorIR第一篇文章：

TensorIR系列【一】 背景与简介
153 赞同 · 10 评论 文章

在介绍TensorIR Schedule时，本文会对比TVM TE Schedule来展开。因此如果有相关背景知识会对TensorIR的理解更有帮助。转一篇其他人写的不错的讲解TVM TE Schedule的文章：

tvm schedule详细举例
467 赞同 · 20 评论 文章
1. Block

TensorIR是一个可以被Schedule的IR。而IR的核心本身是我们新引入的数据结构Block。Block包含的主要数据结构在下图，比较重要的是block iterator，block的访问buffer和block的body，接下来我会一一详细介绍。

Block数据结构
1.1 Block Iterator

每个Block包含若干个iterator，共同定义了这个block所需要的迭代空间。注意，这里每一维迭代空间必须是连续的，是不允许跳跃的。如上图的例子所示，虽然这段程序计算的是 
64
∗
64
∗
64
 的矩阵乘法，但是block的迭代空间是 
16
∗
16
∗
16
 （因为block每次执行所读写的区间是一个 
4
∗
4
 的空间）。

一个常见错误如下。在这个例子里我们定义的迭代空间是 
64
∗
64
∗
64
 ，但是迭代并不连续。在这种情况下这是一个非法的block定义。

for yo, xo, ko in T.grid(16, 16, 16):
    with T.block():
        # The iterator is not continuous
        vy = T.axis.spatial(64, yo * 4)
        vx = T.axis.spatial(64, xo * 4)
        vk = T.axis.reduce (64, ko * 4)
        ...

除了迭代空间，还有迭代类型。一般来说，常用的迭代空间有以下三种：Spatial, Reduce, Scan。

Spatial：A[i] = B[i]。也叫data parallel，指可以数据并行的axis；
Reduce：C[i] += A[i, k] * B[i, k]的k。这个维度通常不能被并行，但可以执行rfactor和allreduce；
Scan：A[i] = A[i - 1]。这个维度只能顺序执行，Schedule空间较小。

注意：TensorIR不会也不能检查Iterator标注是否正确，如果Iterator信息标注错误，可能导致Schedule结果出错。

1.2 Block Access Region

除了Block Iterator，Block另一个重要的信息是Block对Buffer的访问信息。每个block会有他的读/写区间，根据这些读写区间，我们就可以判断出block之间的生产消费者关系，这对程序优化是至关重要的。

对于block的读写关系我们有以下严格要求：

Block 标注的read区间必须大于等于Block body实际的read区间；
Block 标注的write区间必须小于等于Block body实际的write区间。
如果标注符合要求但不完全符合body的实际需求，会导致计算结果正确但性能较差。通常要求标注严格正确。
1.3 Block Body

Block的body其实比较简单，与通常的stmt无较大差别。

唯一要注意的是，一个schedulable block内部的index计算仅可用block iterator表示，不可使用外部循环变量。

1.4 Conclusion

了解完Block的定义，我们回过头来想一下，我们为什么需要这样一个数据结构，为什么需要这样设计。

Block signature（包括Iterator和access region）对外申明了内部程序的执行抽象，而不需要关注Block的内部实现，从而能够对一些复杂计算（opaque intrinsic）提供Schedule能力。其他block只需要关心这个block的读写关系，和其需要的迭代空间内。这与申明了一个函数类似，将内部实现与外层隔离开，仅需提供参数（iterator）和数据（buffer）即可。

Block对内也保护了内部程序的独立性，使其不受外部的影响。只要内部的计算满足Block signature的声明，那么block内部的Schedule时完全与外部独立的。也就是说我们可以分别优化Block内外两部分而互不干涉，TensorIR的核心也就在此。

TensorIR的分治思想
2. Schedule

Schedule的含义是在不改变程序运行结果的情况下，对程序语句的执行方法、执行顺序进行更改，从而使得程序运行效率更高。为了达成Schedule的效果，我们有不同的实现方式（TE Schedule和TensorIR Schedule）。

2.1 TE Schedule

关于TE Schedule的原理，本文不打算深度阐述。简要来说，主要分为以下几步：

建立Stage和Schedule Tree（如下图）；
每个Primitive都是对Stage和Schedule Tree的修改；
将变换完成的Schedule Tree生成IR。

从这个步骤来说，这有以下特点：

依赖额外数据结构（Schedule Tree）；
操作是lazy的，不依赖于操作顺序，但不能即时反映操作结果；
Primitive之间相互影响，各个Primitive之间强相关，且容易产生bug；
TE Schedule Tree
2.2 TensorIR Schedule

TensorIR的实现方式和TE完全不同，并没有采用类似Schedule Tree的额外数据结构，而是直接基于Block和IR本身实现Schedule的各个变换。每一个Primitive更像是一个用户指导下的pass，从一个IR变换到另一个IR，从而实现优化。

TensorIR Schedule有也有以下特点：

实时变换IR，即时反馈变换结果；
Primitive之间相互独立，更易于添加或更改Primitive实现；
对Tensorized计算的原生支持。

此外，基于IR的schedule提供了一种全新的可能：手写程序+Schedule优化。

由于原理上摆脱了Stage和Schedule Tree的限制，Schedule的input不再局限于te.compute生成的简单描述。任意合法的TVM program（导入自TVM Script）都能够进入Schedule进行优化，这也就提供了手写和自动生成以外的第三种编程范式：手写+Schedule（自动）优化。

2.3 Schedule API

虽然TensorIR的实现方式和TE截然不同，但是保持了和TE Schedule的高度相似的API。从用户侧来说，迁移成本较低，以下展示一个简单的TE Schedule和TensorIR Schedule的Schedule对比，例子源于TVM docs的CPU GEMM优化:

# TE Schedule
s = te.create_schedule(C.op)CC = s.cache_write(C, "global")i, j = C.op.axisio, ii = s[C].split(i, factor=32)jo, ji = s[C].split(j, factor=32)s[C].reorder(io, jo, ii, ji)s[CC].compute_at(s[C], jo)ic, jc = s[CC].op.axisk, = s[CC].op.reduce_axisko, ki = s[CC].split(k, factor=4)s[CC].reorder(ko, ic, ki,jc)s[CC].vectorize(jc)s[CC].unroll(ki)s[C].parallel(io)x, y, z = s[packedB].op.axiss[packedB].vectorize(z)s[packedB].parallel(x)	# TIR Schedule
func = te.create_prim_func([A, B, C])s = tir.Schedule(func)packedB = s.get_block("packedB")C = s.get_block("C")C_global = s.cache_write(C, 0, "global")i, j = s.get_loops(C_global)io, ii = s.split(i, [None, 32])jo, ji = s.split(j, [None, 32])s.reorder(io, jo, ii, ji)s.compute_at(C, jo)ic, jc, k = s.get_loops(C)[-3:]ko, ki = s.split(k, [None, 4])s.reorder(ko, ic, ki, jc)s.unroll(ki)s.vectorize(jc)s.parallel(io)x, y, z = s.get_loops(packedB)s.vectorize(z)s.parallel(x)

完整程序和原tutorial链接如下：

小结

TensorIR Schedule采用了以Block为核心的原理进行优化，摆脱了原本的Stage和Schedule Tree，并且引入了全新的编程范式：手写程序+Schedule优化。在API设计上尽可能实现了向下兼容，减少了用户的迁移成本。TensorIR在Tensorize甚至Sparse TIR上的应用以及自动化优化等技术将在后续文章介绍。此外2021 TVMCon也将有更多关于TensorIR的介绍。

合作者
侯博涵@CMU
邵俊儒@OctoML
林武威@OctoML
金弘义@SJTU
赖睿航@SJTU
叶子豪@UW
陈天奇@CMU & OctoML
