# [深入分析CUTLASS系列] 0x02 cutlass 源码分析(一) --- block swizzle 和 tile iterator (附tvm等价code)

**作者**: JoeNomad​​新南威尔士大学 信息技术硕士

**原文链接**: https://zhuanlan.zhihu.com/p/679929705

---

​
目录
收起
Overview
Block Swizzle
代码指引
Block Swizzle逻辑分析
tvm等价代码
Tile Iterator
解决的问题描述
代码指引
Tile Iterator逻辑分析

开篇

大家好，我是joe，在上一篇文章中我们基于一个gemm的case，梳理了整个repo的软件架构和部分细节，

接下来我会对之前提到的每一个组件进行拆解，并分析其中的逻辑，由于整个repo的组件较多，放在一篇文章中叙述内容太多，并不利于大家按需去找到想要深入看的part，所以会按照逻辑上的执行过程拆解成多个part，本文主要聚焦在block swizzle和tile iterator上，也是整个执行过程中的开始

本文focus的内容：

Block Swizzle的内部逻辑以及源码分析(附tvm等价的for loop变换code)
从Conv2d tile iterator来分析cutlass中tile iterator相关的优化手段




Overview

首先简单描述一下block swizzle和iterator做了什么事情：

block swizzle: 更改了线程块的发射顺序，以增加L2 cache 命中率
tile_iterator: 针对不同的分块计算了global load的下标，并根据threadMap提供了load/store function，指导每个warp做load/store的动作
Block Swizzle
代码指引

block swizzle的逻辑相对简单，其原理是以一定步长来做一个换行操作，核心的逻辑就是一个取余操作，会有一个超参来限制swizzle的步长是多少，我们需要关注的文件是:

include/cutlass/gemm/threadblock/threadblock_swizzle.h
include/cutlass/gemm/kernel/gemm.h
block swizzle的核心代码

在GPU中，block发射顺序也是按照x->y->z的顺序来发射的，上图中((block_idx_x) & ((1 << (log_tile)) - 1))其实就是对x轴做取余操作，超参是2的n次方，所以我们可以用位运算来等价取余操作，位运算的开销更小

gemm.h中的调用逻辑
Block Swizzle逻辑分析

我们先来回顾一下gemm的计算逻辑，下面是一个 4096 x 4096 x1024的矩阵乘法，后面都会以这个矩阵乘法来做示例

original的矩阵乘，三层for循环，(M,K) x (K, N), M=N=4096 K=1024，

我们假设 threadblock tile大小是(64, 64)，我们的计算逻辑会变成这样

##注意##,这里为了排除无关信息，先不考虑k维，实际在cutlass的tile中是三个维度，对M，N，K都会切分，K是reduce轴，并不影响swizzle

cutlass中的tile声明, 与上下文计算无关
经过分块后的计算逻辑，等价于不做block swizzle

如果我们不做block swizzle，用代数表达每个threadblock的tile是(tbm, tbn)，那么我们可以看出线程块是先按照axis n发射 \frac{n +(tbn-1)}{tbn} 个, 然后再遍历 axis m，如果n非常大，那么我们相当于先做了一个长方形的矩阵乘法，那么每一个发射的block读取的右矩阵的global位置都是不同的，访存量用公式表示为:mem = leftmem * lnums + rightmem*rnums\\ leftmem = tbm * k\\ rightmem = k * tbn\\

如果不apply block swizzle，lnums = 1， rnums = n
如果apply了block swizzle我们就相当于在给定N的步长内，我们会换到下一行去计算，我们可以得到
lnums = N, rnums = n/N，假设L2是空的情况下，右矩阵命中了3次

显而易见，当我们swizzle了之后，在计算量相同的情况下，单个tile的访存量变小了(对整个gemm来说总量并不变)，且在load过程中减少了cache miss

假设我们的步长是4，计算逻辑示例如下:

step=4时的计算逻辑

我们知道片上的cache是比较贵的资源，相对来说存储空间较小，如果我们的访存量较大超出了L2cache的容量，那么之前存储的内容就会被挤掉，当我们需要再次访问之前存储的内容时，就会发生cache miss，我们就需要去HBM里访存，load的周期就会变长

tvm等价代码

上文中的逻辑示例是用tvm来写的，用tvm写的原因是认为会简洁清晰很多，full code如下

from tvm import te, tir, topi

A = te.placeholder([4096, 1024], "float")
B = te.placeholder([1024, 4096], "float")
C = topi.nn.matmul(A, B)

func = te.create_prim_func([A, B, C])
sch = tir.Schedule(func)
sch.show() # print compute and schedule
mm_b = sch.get_block("T_matmul_NN")
v_m, v_n, k = sch.get_loops(mm_b)

# with block tiling
tb_m, tb_n = (64, 64)
v_m_o, v_m_i = sch.split(v_m, [None, tb_m])
v_n_o, v_n_i = sch.split(v_n, [None, tb_n])
sch.reorder(v_m_o, v_n_o, v_m_i, v_n_i)
sch.show()

# with block swizzle
N = 4  # step
v_m_o_o, v_m_o_i = sch.split(v_m_o, [None, N])
v_n_o_o, v_n_o_i = sch.split(v_n_o, [None, N])
sch.reorder(v_m_o_o, v_n_o_o, v_m_o_i, v_n_o_i, v_m_i, v_n_i)
sch.show()
Tile Iterator
解决的问题描述

Tile iterator提供了左右矩阵的load/store方法，我们这里用conv2d的iterator来讲解，因为conv2d的iterator相对复杂，包含了要讲到的所有优化手段，基本的逻辑是，当我们focus在某一个分块时，我们需要关注这个分块每个thread所load的位置是不是需要被load(这里有点绕)，有如下两点原因:

Conv2d存在padding的情况，假设kernel的大小是3x3，那么当我们滑动窗口在左上角的位置，其实前几个元素都是padding，并不是实际要去load数据，我们直接补0即可
Conv2d和gemm都存在的常见问题——尾块处理，当我们用一个分块不能整除的时候，在load features的末尾，会有超出的部分，我们补0做乘累加也是零，只是多了点计算开销
代码指引

在conv2d iterator中有两个common case的实现，分别是analytic和optimized，他们的区别就是有没有用mask的pre-compute来减少计算load下标的开销，我们需要关注的文件是:

include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h
include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h
include/cutlass/conv/threadblock/conv2d_tile_iterator.h

还有一些其他的iterator没有列出来，如filter的iterator(逻辑跟activation比较起来简单许多)，fewchannel(具有强先验)等，大家有兴趣可以针对性地去看，比如对于小channel而言，fewchannel这个iterator性能会更好，上述三个文件就可以把iterator的所有优化方法描述清楚

conv2d_tile_iterator作为一个common的类，实现了如何load/store

conv2d_tile_iterator.h中的核心代码

各种iterator的文件中会根据自己的需求进行一些特定的判断，都是计算load pointer的下标

overview
Tile Iterator逻辑分析

首先我们先说明一下shared memory是怎么被load的，我们每个threadblock的大小会被等分到每个warp上，每个warp需要去读一个切分的块，gpu的访存指令最大带宽是128bit即16byte，即针对fp16的数据，每个warp每次最多load 32 * 8个值，显然一次load是没办法把所有所需的值load完的，所以我们可以看到threadmap里面会有一个kStride，来表示每个线程的下一次访存步长。我们可以看到还有一个kContiguous参数，在用最大访存指令的情况下这个值是1，在cutlass里每contiguous处理128bit，如果align不满足的情况下我们每次就会循环128bits/align次，尽量使用大的align会让性能更好，比如k=33和k=32的性能差距非常大，虽然他们的计算量并没有差多少

我们先来看一下analytic的iterator，这里是比较基础的实现，逻辑很清晰，有助于我们去梳理iterator的逻辑

conv2d_fprop_activation_tile_access_iterator_analytic.h的核心逻辑

在这里n,p,q,k,r,s这些符号代表的含义，可以参考cutlass的文档，即implicit_gemm的逻辑:

在iterator的构造函数中会预先针对每个stride算好n,p,q的值，在at这个函数中，会通过当前访存到的滑动窗口的哪个位置来反推出feature的n,h,w,c然后输出，valid的函数就比较简单了，即判断n,h,w,c是不是满足条件，输出一个布尔值给到special register然后在执行期判断是否需要被load，相信这很容易理解。这里我们会发现，计算n,h,w,c以及判断的时候，有很多scalar操作，在gpu里这些操作是有比较大的开销的，尤其是&&这种判断操作，我们想要尽可能的优化掉不必要的scalar操作，于是我们来看optimize的iterator里是怎么做的，这里的处理还是非常巧妙的

conv2d_fprop_activation_tile_access_iterator_optimized.h中bit masks的计算逻辑, 在构造函数中precompute

我们针对每个stride是可以预先算出当他访存到特定的滑动窗口位置时，是否是valid，因为是布尔值，所以我们可以用位运算在一个int32中表达，比如以3x3的kernel为例，假设对于kw的mask是00000011(只展示前8个bit)，则代表当kw访存到滑动窗口的(x,0),(x,1)位置时读取,(x,2)则不读。本质上是一个以空间换时间的方法。

masks开辟的总空间大小，2是因为我们需要考虑滑动窗口的h,w
输出predicates

我们可以看到在代码中减少了很多scalar操作，这让我们在load的时候更少的bound在下标的计算上。但其实把scope放在整个gemm的for loop访存下标计算当中，还是有一些可优化的地方，但由于是分组件的，我们就没办法像DSL一样获得一个overall的symbolic表达式去做各种代数化简，公共子表达式消除等，只能依赖nvcc编译器内部的分析能力，且这种写法本身就会存在一些局限性，比如之前就发现过bound在一些奇怪的指令上如CS2R，有时候编译器内部很难去做这种依赖分析，对于nvcc可能也没办法很好的优化，这也是我觉得的一个局限性所在。

本篇文章至此结束，如果对大家有所帮助不妨点个赞呗~ ，我是Joe，是一名AI编译器从业者，如果大家对AI编译，mlsys感兴趣，可以关注一下哟，后续会继续分享CUTLASS的相关知识，也会考虑分享TVM，MLIR，量化算法，分布式推理等相关内容~




相关内容导览：
