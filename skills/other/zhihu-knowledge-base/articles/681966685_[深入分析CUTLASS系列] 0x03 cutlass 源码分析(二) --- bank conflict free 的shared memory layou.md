# [深入分析CUTLASS系列] 0x03 cutlass 源码分析(二) --- bank conflict free 的shared memory layout (附tvm等价pass)

**作者**: JoeNomad​​新南威尔士大学 信息技术硕士

**原文链接**: https://zhuanlan.zhihu.com/p/681966685

---

​
目录
收起
开篇
Prologue
overview
Bank Conflict造成的原因以及common解
ldmatrix(LDSM)指令是如何load data的
MAIN
问题化简
核心layout变换，bank conflict free的layout
代码指引
TVM中的等价pass变化
Epilogue
Reference
开篇

大家好，我是joe，在上一篇我们分析block swizzle和iterator计算global ptr下标的逻辑，本文会进一步分析global->shared以及shared->register中非常重要的优化手段，bank conflict free的smem layout, 也可以说是smem的swizzle(3.0 cute api中有swizzle的IR抽象)，在这个优化中同时消除了shared store/load的bank conflict以提高访存的效率。

本文focus的内容：

1. 为什么需要swizzle，以及矩阵乘中造成bank conflict的原因
2. swizzle是如何完成消除bank conflict的工作的
3. swizzle的数学表达式推导以及核心代码指路

Prologue
overview

总的来说，矩阵乘中的shared memory的bank conflict是由于for loop tiling导致的load/store的轴交换造成的，其中ldmatrix的load行为属于tiling的强先验，会限制for loop如何去做tiling。

我们需要先confirm两点：

1. bank conflict的原因以及common解法
2. ldmatrix的工作原理

Bank Conflict造成的原因以及common解

快速过一下这个部分，细节可以参考nv的文档，shared memory被分为32bank，每个bank的位宽是4bytes，如果同一个warp中的不同线程访存到同一个bank中，会造成bank conflict，但当GPU每个线程访存大于4bytes即每个warp大于128bytes时, 每个wrap则会分割成多个transaction去执行，每个transaction保证线程内的访存不落在同一bank即可，所以当我们用最大访存指令时，需要保证1/4个连续线程不会存在地址重叠。

比较常见的case是transpose算子，假设我们访存(32,32)大小的float矩阵，smem也申请相同大小array，我们store的时候确实不存在bank conflict，但是当我们load的时候，我们要去访存一列以实现transpose，那么32个线程将落在同一个bank发生争抢，造成bank conflict。

    // 在transpose算子中，我们一般会用smem padding的方式去解决，如下:
    // 访存一列的时候，column id每一行都会有向右的偏移，则避免了bank conflict
    // 可参考 https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
    __shared__ float tile[TILE_DIM][TILE_DIM+1];
    


矩阵乘法的bank conflict逻辑也是一样的，我们需要一行乘一列(此处行列均指代一个长方形的块，宽不为1)，所以也会存在bank conflict，我们也可以用padding来解决矩阵乘的bank conflict，但是由于矩阵乘的shared memory用量很大，padding的填充其实只是为了下标偏移，没有实际存储的作用，降低了occupancy，过多的浪费会导致kernel性能下降。

ldmatrix(LDSM)指令是如何load data的

ldmatrix指令最大可以load 4个8x8的矩阵(16bits位宽的数据)到register中，每一个thread会负责load 128bits的数据并广播到临近的4个线程当中(个人理解原理应该和__shfl_sync差不多)，每个线程最终会拿到4x32bits的数据(对应4个8x8矩阵)，此处我觉得graphene(ASPLOS'23 nv的论文)里的图解是我看过最直观的，这个图也是偶然在tensorcore中ldmatrix指令的优势是什么？中reed老哥的回答中看到的，当时觉得这个图非常好，我看到标注是graphene这篇论文，随后也去读了一下这篇，这里也推荐reed老哥的专栏写的很好 CUDA高性能编程

ldmatrix 图解，论文中有对应code，不在此处增加篇幅了

不过这个图里广播的行为有点misunderstanding，图中T0会读取8个值，并分成四份，共享给T0-T3，所以绿色的四个32bits数据其实来源于{T0, T8, T16, T24}，所以我们每个线程中拿到的数据实际是非连续的。




MAIN
问题化简


我们已知每 1/4个warp访存不同的bank就可以达到bank conflict free，所以我们可以推导出一个子问题，只要我们能够在(4,64)的块里保证没有bank conflict即可，左右矩阵只需用这个块去分割成n个tile

核心layout变换，bank conflict free的layout

cutlass文档中以row major的filter为例，假设我们load一个32x8，并transpose后以row major存在shared memory当中(ldmatrix可以加.trans在指令内部做transpose)

我们可以看到，此时我们store时每1/4个warp在同一行，即不同bank

当我们load的时候，每一个线程的fragment的index也指向不同的列，所以也是没有bank conflict

// 此时(4,64)的块中对于每个线程分配的块的表达式为
  int row = (lane_id >> 1) & 3
  int store_column = (lane_id % 8) ^ (lane_id / 8);


我们可以理解为每一行是一个group，每四行是一个循环，XOR是满足结合律的，所以我们可以先算出第一个group的column值，然后把表达式改写成递归的形式

// k = {0,1,2,3}
// ^1 advances from k=0 to k=1
// ^3 advances from k=1 to k=2
// ^1 advances from k=2 to k=3
// ^3 advances from k=3 to k=0
// 推导: x ^ 2 = x ^ (1 ^ 3) = (x ^ 1) ^ 3 = x1 ^ 3
  int store_column_next = k & 1 == 0 ? store_column ^ 1 : store_column ^ 3;

代码指引

我们需要关注的文件主要如下:

cutlass/layout/tensor_op_multiplicand_sm75.h
cutlass/gemm/warp/mma_tensor_op_tile_iterator.h
cutlass/gemm/threadblock/mma_pipelined.h

以2 stage为例，shared memory store的init offset部分在构造函数中完成

mma_pipelined.h中初始化smem_iterator A,B
tensor_op_multiplicand_sm75.h 中的核心逻辑

在这里不妨cuda-gdb打个断点，back trace会清晰很多

mma_tensor_op_tile_iterator.h 中核心逻辑
TVM中的等价pass变化

tvm中有一个核心思想相同的pass，只要看注释就能很容易的明白这个layout具体做了什么以及什么case下会有用

inject_permute_layout pass，数据类型是fp16

如上图所示，假设我们load一个(64,64)的块，显然一次ldmatrix是没法全部读进去的，我们假设想要把矩阵乘法的子问题变成(8,32)x(32,8)，即一次ldmatrix的访存量，对于左矩阵的{T0-T7}来说，读取的是一个对角线上的位置，是没有bank conflict的

# ut可参考tvm/tests/python/tir-transform/test_tir_transform_inject_permuted_layout.py

# schedule example
sch.annotate(block_or_loop=b53, ann_key="permuted_layout", ann_val="g2s_A")

# primfunc block example
T.block_attr({"permuted_layout": "g2s_A"})
for ax0_ax1_fused_0 in range(4):
    for ax0_ax1_fused_3 in T.vectorized(8):
        X_reindex_shared_dyn[ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, threadIdx_x % 4 * 8 + ax0_ax1_fused_3] = X[blockIdx_y // 8 * 128 + ax0_ax1_fused_0 * 32 + threadIdx_y * 8 + threadIdx_x // 4, ax2_0_0 * 32 + threadIdx_x % 4 * 8 + ax0_ax1_fused_3]
Epilogue

新的一年到啦，祝大家新年快乐~




本篇文章至此结束，如果对大家有所帮助不妨点个赞呗~ ，我是Joe，是一名AI编译器从业者，如果大家对AI编译，mlsys感兴趣，可以关注一下哟，后续会继续分享CUTLASS的相关知识，也会考虑分享TVM，MLIR，量化算法，分布式推理等相关内容~




相关内容导览：

Reference

An Efficient Matrix Transpose in CUDA C/C++ | NVIDIA Technical Blog

tensorcore中ldmatrix指令的优势是什么？

Graphene: An IR for Optimized Tensor Computations on GPUs | Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3
