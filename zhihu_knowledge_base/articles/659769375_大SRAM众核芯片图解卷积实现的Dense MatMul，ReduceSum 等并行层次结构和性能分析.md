# 大SRAM众核芯片图解卷积实现的Dense MatMul，ReduceSum 等并行层次结构和性能分析

**作者**: maja性别男，爱好女

**原文链接**: https://zhuanlan.zhihu.com/p/659769375

---

1. 问题引入

下午4点是英国的早上，延续了一天的思考和 GodFrey 博士的讨论，促成了这篇文章的核心观点 ：大SRAM芯片矩阵乘并行层次分析；而当 FlashAttention 流行起来，我们一致认为 “不同运算适用不同的内存排布” 有必要让更多人了解

CUTLASS 在2017([1])，发布了NV GPU设备上GEMM并行层次结构划分方案，在shared memory上引入了外积Tiling划分方案提高cache命中率，并通过addr-bank-thread映射缓解了NV GPU分支性能退化问题。

几乎同时，TVM率先针对一般GPU, CPU设备内存层级, 完成并行层次结构(Hirarchy)建模，将数据划分为InnerDimensions, OuterDimensions。

由于 GPT-3 等模型性能瓶颈在IO密集算子日益明显，OSDI 2021 OpenAI TRITON、OSDI 2022 Roller 结合内存层级，内存事务特点，抽象出了基于 {Load, Compute, Store} 设备代码优化和生成模型解决上述问题。

随后，FlashAttention等作品进一步通过提高数据片上驻留时间，参与更多运算，缓解了NV GPU的访存压力。

我们尤其注意到2021～2022年芯片设计革新带来的变化，针对具有近存计算能力的大SRAM众核芯片的算法设计，鲜有讨论。

转眼间，OSDI 2023三星提出近存HBM, 分析发现对访存密集算子有望提供5~8x 速度提升；同时地平线等企业证实了车规级软件运行 Bev中等模型的可能性。

近存芯片先驱 Graphcore IPU 在 FP16上可以提供280 TFLOPS，并在字节，阿里等企业用户模型近一步获得验证，是一个比较可靠的研究对象。

矩阵乘和卷积通常被认为是一个问题，这样就可以共享硬件加速单元。这里涉及 1x1 卷积的两个性质：（1）通道数变换，(2）加权平均。

对于第一个性质，近存芯片IPU中，通过将矩乘问题

(Groups(G), M, K) x (Groups(G), K, N)

映射为 1 维卷积问题 ：

(G, M/*batch_size*/, K/*输入通道*/, 1/*数据空间尺度*/) conv 
(G, N/*输出通道*/, K/*输入通道*/, 1/*卷积核空间尺度*/)

实现等价变换。

这里 空间尺度是 1 维的 (1, )，对应卷积核 kernel shape ；如果空间尺度（spatial shape）是二维的 (1/*H*/, 1/*W*/)，则对应的卷积核 kernel shape (1, 1) 正好就是我们在图像领域熟悉的 1x1 卷积。

对于第二个性质，是本文关于重映射(Remapping)策略讨论的关键议题。

通常针对矩阵/卷积的加速单元，如：intel AXM, Graphcore AMP, NV TensorCore 和 针对 ReduceSum 加速单元，如 ：Graphcore/Intel/AMD VectorCore, NV CUDACore，是不同的。

我们讨论了，大SRAM众核芯片上，矩阵乘并行层次结构并讨论和CUTLASS，TVM, OpenAI TRITON 等针对NV GPU设备实现的GEMM技术区别。

接着我们分析了在大SRAM众核芯片上ReduceSum并行层次结构和并讨论通过(1,1)卷积实现和性能。

最后我们总结了计算范式，由于 FlashAttention 延长了数据分片片上时间，并参与更多运算，而“不同运算适用不同的内存排布”，有理由认为全新的 编程模型抽象层级 {Load, Remap, Compute, Store} 将有助于生成更高性能设备代码。

2. Graphcore IPU GEMM 并行层次

首先任意 shape 的矩阵乘，当我们遵循 pytorch group matmul 和 broadcasting 规则，可以转换为

(G, M, K) x (G, K, N)

       B | HEAD | Spatial
LHS = (6 | 12   | 1 4 1 512 64 )
RHS = (6 | 12   | 3 1 6 64  512)

       G    | Broadcast | Spatial
LHS = (6 12 | 1 4 1     | 512 64 )
RHS = (6 12 | 3 1 6     | 64  512)

       G  | Broadcast | Spatial
LHS = (72 | 4         | 512 64 )
RHS = (72 | 18        | 64  512)

       G  | Spatial
LHS = (72 | 2048 64  )
RHS = (72 | 64   9216)


这里 G 表示 分组（groups），用来对齐两个矩阵相同的维度，比如batch_size, head_size 都可以放入这里。Broadcasting 是从 spatial dimension 里面按照 pytorch broadcasting 规则依次对齐。

在NVIDIA GPU GA100架构 共有 192 个 Streaming Processor[2]；Graphcore IPU IPU21架构中可以对应的就是 1472 个面积更小的双通道多线程分时处理单元：Tile。

并行层次的划分（Partition） 包含不同的层级(hirarchy ：Partition.level)。 每层的划分（Split）可以按照实际并行程度标记为 Parallel（比如不同核心/线程对应的划分）, Serial（比如单线程中For Loop 对应的划分）。

我们首先描述 IPU IPU21 架构并行层级，并通过 CUTLASS 在 GA100 Ampere 架构对照说明（ GV100 Volta 有相同的层次结构）:

IPU GEMM 并行层次结构

整体上核心映射(tileMapping)，分为 tile-level, system-level。

Tile-Level 就是将输入数据加载到输出数据所在的核心；system-level就是将输出数据分片划分到具体核心，并标记需加载的输入分片：

for (auto g1/*outter dim of groups*/=0; g1 < G1; g1++) {
  for (auto b1/*outter dim of batches*/=0; b1 < M1; b1++) {
    for (auto ic/*input channels*/=0; ic < K1; ic++) {
      for (auto oc/*output channels*/=0; oc < N1; oc++) {
        // map tensor slice
      } 
    }
  }
}

在RowMajorLayout下，数据被划分为(outterDimensions..., innerDimensions...) 可以递归。


2.1 Groups Level


片上(on-chip)分布式Tensor计算，需要将参与计算的tensor slice拷贝到一个 核心 (Tile) 并在上面创建设备代码（Vertex/MultiVertex）, 因此一个 Group 的数据需要映射到相邻的核心，保证临近传输效率。同理一个 Batch 的数据。

IPU srcTile-destTile (inter chip) 传输，首先将 srcTile 数据分片 (tensor slice) 通过所在srcColumn通道拷贝到 Exchange, 再定位到destTile所在的 destColumn，并传输到 destTile。

该模型导致逻辑上临近的 核心 和 物理上临近的 核心 有所区别，可以通过一个映射表解决。

尽管创建的 data input 视图具有卷积形状：


(G, M/*batch_size*/, K/*输入通道*/, 1/*数据空间尺度*/)

但在IPU中遵循通道后置，和 innerMostDimension, outMostDimensions 划分原则。

仍然用二元算符 ~(A,B):TensorView -> TesnorView 表示 A 是 B 的一个视图：

TensorView(G, M/*batch_size*/, K/*输入通道*/, 1/*数据空间尺度*/)
~ TensorView(G=G1*G2, M=M1*M2, 1, K=K1*K2)
~ TensorView(G1, M1, K1, 1, G2, M2, K2)

当 G 远小于核心数(比如IPU21架构中的1472)，则核心会被均匀分配给每一个Group (G1=72, G2=1)。举例 attention 中的矩阵乘：

(6/*batch*/*, 12/*head size*/, 512, 64) * (6/*batch*/*, 12/*head size*/, 64, 512)

对应 72 个 groups，假设1440 个 核心参与计算，平均每个矩阵乘

(512，64) * (64，512)

分配到20个核心用于并行计算。




NV GPU CUTLASS Block Partition[1]

IPU Group Level partition 类似 NV Block Level partition 输入数据，输出数据从逻辑上划分到的 SM 网格中：

每个SM沿着 K 维度在 For loop 中依次从Global Memory加载 (BlockItemsY, BlockItemsK)，(BlockItemsK, BlockItemsX) 以及 (BlockItemsY, BlockItemsX) 到片上shared memory, 将计算结果累加到 (BlockItemsY, BlockItemsX) 并写回 Global Memory。

这一阶段 Graphcore IPU 有以下特点：


- 核心数众多并发层级高；

- 数据运行时已经在片上，部分或者全部在对应目标核心；




2.2 Spatial Level


卷积主要按照 批(batch_size), 输入通道(input_channels)，输出(output_channels) 进行空间尺度划分。data input 切片(actSlice) 和 weights input (weightSlice) 在每个核心默认采用 Row Major排布 , 矩阵乘相当于 Row Major x Column Major ：

IPU GEMM Spatial Partition

每个 Tile 包含了输出数据分片 outSlice，以及对应输入分片actSlice, weightSlice， 并沿输入通道方向(input_channel) 累加 (ReduceSum)。

(512/*M=batch_size*/, 64/*K=input_channel*/) conv 
(512/*N=output_channel*/, 64/*K=input_channel*)




卷积被分配到 20(=1440 / 72) 个核心，我们需要决定 (batch, inputChannel, outputChannel)，空间划分 (spatial partition)，比如：

M(72)=M1(4)*M2(128)
K=K1(1)*K2(64)
N(72)=N1(5)*N2 // 不是常量，划分为5份，每份可取{102, 103}，保证数据均衡

在 RowMajor Layout下，(M1=4, K1=1) 和 (N1=5, K1=1) 表示 outterDimensions， (M2=128, K2=64) 和 (N2={102, 103}, K2=64) 表示 约 32KB 和 25 KB 的 innerDimensions。对innerDimensions递归该划分：

// {m,k,n}_i partition index at level i
// {M,K,N}_i partition size at level i

for (auto i/*level*/=0; i < LEVEL; i++) {
  for (auto k_i=0; k_i < K_i; k_i++) {
     // m=M/{M_0*M_1...M_i}
     // k=K/{K_0*K_1...K_i)
     // (m, k, 1)
     auto& actSlice = sliceAct();
     // n=N/{N_0*N_1...N_i}
     // k=K/{K_0*K_1...K_i)
     // (n, k, 1)
     auto& weightSlice = sliceWeight();
     // (m, n, 1)
     auto& outSlice = sliceOut();
     // accumulating partial sum matrix
     // create outSlice += actSlice conv weightSlice Vertex
     auto& vtx = createParitalConvVertex()
     // mapping parallel execution onto tiles
     mapping(actSlice, weightSlice, outSlice, vtx);
  }
}

现代编译器基于 Cost Model 来优化，Graphcore 在IPU上最为出色的工作之一，就是构建了AMP的指令模型。体现在CPU设备上，可以通过IPUModel 进行 带内存约束的 cycles极小目标 估计，得出一个划分 (M1, K1, N1)。

NV GPU CUTLASS Thread Block Partition[1]

在这一个并发层级，NV GPU 中数据从Global Memory 搬运到SM片上的 shared memory，在 GA100架构芯片，每个SM大概能分到 300 KB 大小的片上存储。SM 内通过以线程组(Warps)提供数据并行能力。

进一步采用了外积（列排布标记行号，行排布标记列号）方式来划分 warps , 每个warp只需要加载一次数据：

Warp(i,j) = Row(A, i) * Col(B, j)


这一阶段 IPU Spatial Partition 从空间维度上，进一步划分加载入一个核心 627 KB 的划分，确保只需要较少或者不需要片内数据拷贝。


2.3 Tile-Level

IPU每个核心是一个支持数据并行（MD）的多线程（MI）分时处理器。因此一个核心可以创建多个 Vertex (6 个)，或者创建一个 MultiVertex 来执行PartialConv。

线程以轮询（Round-Robin）方式执行，主要用来填充指令流水(instruction pipeline), 增加单位时间的指令吞吐，从而提高执行效率。


每个核心上的 Vertex 函数处理以下卷积：

              | outter dims                                  | inner dims (AMP)
actSlice:     (G2 /*conv groups*/, K2,                     M2)
weightsSlice: (G2 /*conv groups*/, K2,                     N2)
outSlice    : (G2 /*conv groups*/, N2,                     M2)
              | outter dims                                  | inner dims (AMP)
actSlice:     (G2 /*conv groups*/, K2/16 /*ig*/,           M2,           16/*ic*/)
weightsSlice: (G2 /*conv groups*/, K2/16 /*ig*/, N2/16 /*og*/, 16/*oc*/, 16/*ic*/)
outSlice    : (G2 /*conv groups*/, N2/16 /*og*/,           M2, 16/*oc*/)

包含了一组独立以 16 x 16 或 16 x 32 为单位循环，


for (auto cg = 0; cg != G2; cg++) {
  for (auto og = 0; og != N2 / 16; og++) {
    for (auto ig = 0; ig != K2 / 16; ig++) {
      for(auto i=0; i != M; i++) {
        auto& inSlice = in[cg, ig][i, :/*ic*/];
        auto& wSlice = w[cg, ig, og][:/*oc*/,:/*ic*/];
        out[cg, og][i, :/*oc*/] += AMPConv(inSlice, wSlice);
      }
    }
  }
}


IPU 提供了 FLOAT2, Half4, Half2等向量数据类型，并提供了完全重载的能力， 代码十分接近CPU线程的写法；

沿 batch 方向 参数被复用起来了，AX(L)U 提供了64 x 64-bit专用的寄存器。

这一阶段 IPU 同时提供了多线程(6x)，向量数据指令两种并行能力。


NV GPU CUTLASS Warp Tile Partition[1]

NV GPU 这一层级由于采用了外积，因此 A 数据在 shared memory 排布按照 Column-Major，充分利用数据从 Shared-Memory 到 RF 内存事务效率；同理B 在 shared memory 采用行排布。


同时 通过 WMMA-API NVGPU 会进一步按照 TensorCore 尺度进行划分（GV100 Volta 架构 4x4x128-bit, GA100 Ampere 架构 8x8x128-bit）


这一阶段 IPU 由于宽裕的 片上 存储，有以下特点：


- Memory Layout : actSlice 和 weightsSlice 采用行连续存储；由于尺度较宽不会造成内存事务padding, 导致低效；


- 卷积形态下，对贡献参数和数据流进行了划分




2.4 矩阵乘加速 AMP




AMP

在 IPU21 架构中，卷积参数首先被加载如 64 x 64-bit 寄存器，对于FP16数据，一次可加载64x4=16x16个参数（被16个AMP Unit共享），并流式读入16x1的数据输入[3]，并得到16x1的输出，执行时间为1个cycle。


3. ReduceSum


类似于 Matmul 首先我们按 Block/Group/Spatial 进行划分, 数据将被均匀的分布在 1472/1440 个核心上（第一层并行）, 每个核心划分到 6 个线程计算 partial sum （提高指令流水线吞吐），线程内部采用向量数据指令累加(数据并行)。


我们知道 1x1 卷积如果输出通道为1，那么相当于对原数据按输入通道方向进行加权, 卷积值为 1.f/inChannels 则为加权平均。


首先在 Numpy/大SRAM 芯片 上分别实现了任意 axis , 通过 Conv1x1 完成 Reduce 的操作，可以证明精度是没有问题的:

# x : (B, inChannels, H, W/*inner most dim*/)
# channel_dim : 1
# reduce_dim : 3

# xShuffleView : (B, W, H, inChannels)
xShuffleView = x.dimShuffle({channel_dim, reduce_dim})

# kernel : (1, W, 1, 1)
kernel = createConv1x1({1, W, 1, 1})

# (B, W/*inner most dim*/, H, inChannels) conv 
# (1, W, 1, 1)
outView = conv(aShuffleView, kernel)

# outShuffleView (B, H, inChannels, 1)
outShufleView = out.dimShuffle({reduce_dim, channel_dim})

// reorder ?

在 Graphcore IPU GEMM 并行层次结构 小节，知晓卷积内层排布：

act     : (1,      W / 16 /*ig*/,                  B, H, inChans, 16)
weights : (1,      W / 16 /*ig*/, outChans/16 /*og*/, 1,       1, 16, 16)
out     : (1, outChans/16 /*og*/,                  B, H, inChans, 16)




我们可以通过对 (W, B, H) 划分，调整内存尺度；容易看到，这里存在两个问题


- outChans ：这里实际为1，不能直接以 16 (oc) 为单位分出 该并行层级的外层划分 (og)

- reOrder ：我们标记了连续内存维度，shuffle并没有改变捏出视图，并不是 Reduce 最佳分布式状态，这意味着Copy中增加了Reorder(依赖 H， W 形状)，增加了tradeoff

对于第一个问题的解决，我们可以在(1,1)临近 维度ig匀出 (16) 维度，这意味着 W 需要是 16x16 的整数倍，或者软件层面 padding 15 个维度 ，但必然导致解决方案不通用，或者计算浪费。


对于第二个问题的解决，需要在计算开始前，通过识别连续内存维度(detectContinuousDimensions)，重映射(remap) 输入向量，减少片上传输(exchange)。

4. 具有重映射特征的设备代码生成抽象层

分布式排布

对于片上分布式Tensor，均匀地分布在片上每一个核心的 SRAM 上。在 GPU 中分配到每一个SM的 shared memory 作为缓存用于加载对应的 数据分片，通常统一内存排布格式。


视图操作"～"使目标张量的即便具有相同的形状(shape)，但却无法具有相同的连续内存维度。Flash Attention, 延长了传统GPU一个数据分片驻留时间，参与不同的操作。

通过IPU实算分析，有充分理由认为 “不同的操作适合不同内存排布” ；同一个排布参与不同的运算并不是最优，在片上重新排布，以极少的代价(约几百cycles)，获得整体最佳性能。

GH100 架构新增加了一个功能，SM-SM 通信，使得数据分片，可在片上重新拉取、排布；而片上重新拉取排布的速度约 GPU 访存带宽的 10x。

单独对一个算子输入 Remap 并评估收益/损失，并不困难。但设计一个通用 Remap 操作，十分困难需要从编译器层面解决，就如同Graphcore从LLVM编译器层面解决 memroy bank 问题。

但幸运的这并不是不可能。

- 首先每个片上的分布式状态，可以重新划分，考虑FP16列排布输入（innerDim, 72, 1024） ReduceSum + broadcast sub + elementWiseExp 操作：按 64-bit partial sum，innerDim=(4*6)^2=576 映射到一个核心操作，只需要两轮 partial sum 就可以完成 Reduce操作，处理576个数；如果innerDim=240，第一轮下来只有10个数，第二轮对10个数做reduce，同样两轮，却只处理了一半的数据。因此对boradcast sub 合适的排布，对 ReduceSum并不合适。

- 片上重排布，考虑FP16 列排布，和行排布矩阵运算：A*B + a*A+b*B，分布式载入tileA=load(A), tileB(B)到多个 SMs，每个SM不仅可以执行外积累加，多个SM上的分布式 tileA，tileB 还在对应内存区域执行加法。




内存重排

也许多年后的日光和煦的下午，我们会想起2023年服务器上，芯片点亮的那个下午，我们一边数着寄存器，一边测试一寸一寸织出地设备代码。




https://weixin.qq.com/g/AQYAAJfN-ibh2PA1b710gCNup-lKmlDNKS-eyxaxVO14iQ2FCfEh4wAglaYZEMS7 (二维码自动识别)




参考文献：
1. CUTLASS GEMM : https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/， Retrieved on 1st Oct 2023

2. PTX ：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#programming-model

3. Graphcore AMP : https://docs.graphcore.ai/projects/ai-float-white-paper/en/latest/ai-float.html

4. CUTLASS Ampere : https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
