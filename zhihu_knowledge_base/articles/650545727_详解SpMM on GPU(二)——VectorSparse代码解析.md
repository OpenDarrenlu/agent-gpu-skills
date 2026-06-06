# 详解SpMM on GPU(二)——VectorSparse代码解析

**作者**: MASA-XUEzy

**原文链接**: https://zhuanlan.zhihu.com/p/650545727

---

​
目录
收起
以V=8为例，介绍wmma_spmm
整体内核函数的流程：
输入
进入主循环：
处理剩余不满Tile_N(32)的数据
写回
结构、函数解析
结构wmmaSparseTile
结构wmmaDenseTile
结构wmmaComputeUtils8
结构wmmaOutputTile8
注：

vectorSparse[1]是首个在Tensor Core上做结构化稀疏矩阵乘的工作，代码也是完成度高、可读性高。

vectorSparse
github.com/apuaaChen/vectorSparse.git

其主要功能在上一篇中已经介绍，在读这一篇时，建议结合源码和上一篇，尤其是图11。




以V=8为例，介绍wmma_spmm

其实际功能是实现半精度浮点-结构化稀疏矩阵乘(Structured SpMM)(C[M*N]=A[M*K]B[K*N])，其中A矩阵是由长度为8(V)的向量构成的向量稀疏矩阵，如图1是V=2的向量稀疏。

图1 A矩阵，向量稀疏矩阵

首先看一下线程是如何划分的：

Thread Block高Tile_M = 1，宽Tile_K = 64，每次K维迭代Tile_N=32。BlockWidth(32)*Tile_M=32，一个Block启动了32个线程。

dim3 grid_dim(ceil(static_cast<float>(m_vec) / Tile_M), ceil(static_cast<float>(k) / Tile_K), 1);
dim3 block_dim(BlockWidth, Tile_M, 1);

整体内核函数的流程：
定义坐标、划分线程；
根据重新换行后的行索引，确定每个线程真正对应的行；
声明Shared Memory空间给稀疏矩阵，声明Register给稠密矩阵；
大循环直至每行数据不足Tile_N(32)： 加载稀疏矩阵到Shared Memory 循环8次：加载稠密矩阵(4*8) 循环8次：mma.884计算
处理不满Tile_N的数据
写回
输入
typename LoadType, typename IndexType, typename VecType, 
typename OutType, typename StoreType,
int Tile_N, int Tile_K, int BlockWidth, int VecLength>
(
    int m, //m_vec
    int k, //N
    int n, //K
    const int* __restrict__ row_indices, //m_vec * sizeof(int)
    const int* __restrict__ row_offsets, //(m_vec + 1) * sizeof(int)
    const int* __restrict__ column_indices,//nonzeros_vec * sizeof(int)
    const half* __restrict__ values,
    const half* __restrict__ rhs_matrix,
    OutType* __restrict__ output_matrix)


因为以V=8为例，8个fp16刚好是128bit，GPU最大的访存指令宽度为128bit，每条访存指令刚好读取一个向量，因此VecType是Float4(128bit)。而为了最大化利用带宽，加载数据用的LoadType和StoreType都是Float4。

输入矩阵乘规模M-K-N，其中，m为稀疏矩阵向量高度(实际的矩阵高度M = m*VecLength)，n为稀疏矩阵宽度/稠密矩阵高度(即K,这里由于源码写法的原因，将实际的K写成了n，后面源码都是如此)，k为稠密矩阵宽度(即N，与之前一样，写法与通常理解不同)。

row_indices：稀疏矩阵行号，在Sputnik[2]中引入了一种解决负载平衡的方式——Row_Swizzle，会根据稀疏行的长度来排序，让相邻行负载近似。因此行号不一定是顺序的。

row_offsets：CSR格式行偏移数组。column_indices：CSR格式列索引数组。(CSR格式不详细讲，不了解的可以看链接：稀疏矩阵主要存储格式[3])values：稀疏矩阵值，只有非零值。

rhs_matrix：稠密矩阵。output_matrix：C矩阵值，输出。

参数：

int m_index_vec = blockIdx.x; //线程块X坐标
int k_index = blockIdx.y * Tile_K; //线程块Y坐标*64
const int lane_id = threadIdx.x % 4; //线程组内序号 0-3
const int thread_group = threadIdx.x / 4; //线程组号 0-7
/////////////////////////////////////////////////////
if (m_index_vec >= m) return;//计算结束


1.如果当 前线程块X坐标大于等于矩阵高(m_index_vec >= m_vec)，则计算结束。

2.根据当前线程块号，到稀疏矩阵行索引数组(row_indices)(换行后行索引并不是顺序的)中，取到该行实际对应的索引。

m_index_vec = __ldg(row_indices + m_index_vec);


3.再根据新的行号(m_index_vec)，取到CSR格式偏移数组(row_offsets)中，该行的偏移(row_offset_vec)和该行的非零数(nonzeros)。

int row_offset_vec = __ldg(row_offsets+m_index_vec);
int nonzeros = __ldg(row_offsets+m_index_vec+1) - row_offset_vec;


4.声明共享内存，稀疏矩阵分块数组(values_tile_array)和该分块索引数组(column_indices_tile_array)。每个Thread Block负责V*Tile_K的C矩阵分块。

其实这里values_tile_array已经声明的是float4，大小不需要再乘VecLength了，__shared__ float4 values_tile_array[Tile_N];即可。

__shared__ float4 values_tile_array[VecLength * Tile_N];
__shared__ int column_indices_tile_array[Tile_N];

float4 * values_tile = values_tile_array;//ptr
int* column_indices_tile = column_indices_tile_array;//ptr


5.创建名为sparse_tile_loader的wmmaSparseTile结构，并使用给定的参数进行初始化。

wmmaSparseTile<LoadType, VecType, VecLength, Tile_N, BlockWidth> sparse_tile_loader(
k, row_offset_vec, threadIdx.x, values, column_indices,
values_tile, column_indices_tile
);


6.定义稠密矩阵能够加载的大小。Tile_N / 4 * 8 = 32 / 4 * 8 = 64。

constexpr int kDenseFragmentSize = Tile_N / 4 * 8;
__align__(16) half dense_matrix_fragment[kDenseFragmentSize];


7.创建名为dense_tile_loader的wmmaDenseTile结构，并使用给定的参数进行初始化。

wmmaDenseTile<LoadType, Tile_N, Tile_K, BlockWidth> dense_tile_loader(
k, k_index, lane_id, thread_group, 
rhs_matrix, column_indices_tile, dense_matrix_fragment
);


8.设置输出寄存器为16个fp32。

创建名为computer的wmmaComputeUtils8结构，并使用给定参数进行初始化。

constexpr int kOutputFragmentSize = 16;
__align__(16) float output_fragment[kOutputFragmentSize] = {};
wmmaComputeUtils8<VecType, Tile_N> computer(values_tile, dense_matrix_fragment, output_fragment, lane_id, thread_group);

进入主循环：

大循环按照每行非零数数量整除Tile_N(nonzeros/32)次的K维循环。每个大循环内部还有小循环InnerSteps(Tile_N/4=8)次。一次小循环加载4行，4*Tile_K = 4*64=256个数据。因为启动了32个线程，每个线程对应8个fp16数据，而8个fp16刚好是一个LoadType，也是GPU最宽的访存指令LDG.128。因为迭代8次，就是64个fp16数据，正好对应了为什么上面kDenseFragmentSize = 64。

先将稀疏矩阵的数据加载到Shared Memory上，然后分8次加载稠密矩阵到寄存器，再分8次执行矩阵乘加操作。整个过程循环展开。

constexpr int InnerSteps = Tile_N / 4;//32/4=8
for (; nonzeros >= Tile_N; nonzeros -= Tile_N){
    sparse_tile_loader.Load();//将稀疏矩阵的数据加载到Shared Memory上
    __syncthreads();
    #pragma unroll//分8次加载稠密矩阵到寄存器
    for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
        dense_tile_loader.LoadRow(n_group_idx);
    }
    __threadfence_block();
    #pragma unroll//分8次执行mma.884操作
    for (int n_group_idx = 0; n_group_idx < InnerSteps; n_group_idx ++){
        computer.TileMAC(n_group_idx);
    }
    __syncthreads();
}
asm("");

处理剩余不满Tile_N(32)的数据

先将稀疏矩阵的Shared Memory清空，将剩余的稀疏数据加载到Shared Memory上。再将剩余的稠密矩阵加载到寄存器，若果剩余行数小于4，则需要更小的加载指令，并计算结果。

sparse_tile_loader.ZeroTiles();//将稀疏矩阵的Shared Memory清空
__syncthreads();
sparse_tile_loader.Residue(nonzeros);//将剩余的稀疏数据加载到Shared Memory上
__syncthreads();

int n_group_idx = 0;

#pragma unroll 8
for (; n_group_idx < InnerSteps; n_group_idx ++){
    if (nonzeros < 4) break;
    dense_tile_loader.LoadRow(n_group_idx);//按4行加载稠密矩阵
    computer.TileMAC(n_group_idx);//计算
    nonzeros -= 4;
}
asm("");

dense_tile_loader.ResidueLoad(n_group_idx, nonzeros);//加载剩余矩阵中不满4的行
computer.TileMACResidue(n_group_idx);//计算剩余的数据

写回

最后创建名为output_tile_storer的wmmaOutputTile8结构，并使用给定的参数进行初始化。并将结果矩阵写回。

wmmaOutputTile8<OutType, StoreType> output_tile_storer(lane_id, thread_group, m_index_vec, 
    k_index, k, output_fragment, output_matrix);
output_tile_storer.Store();

结构、函数解析
结构wmmaSparseTile
//每个__ldg()指令可以加载多少个fp16数据
static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
//每个线程需要处理K维几个向量(这里是32/32=1，即一个线程处理一列向量)
static constexpr int kThreadItemsN_ = Tile_N / BlockWidth;
//稠密矩阵列方向实际需要的加载次数。需要加载N/8次
const int rhs_columns_ = (rhs_columns / kValuesPerLoad_);
//当前线程加载哪一个向量(0-31)，一个线程加载一个float4。(Global的地址)
const VecType* values_ = (reinterpret_cast<const VecType *>(values) + row_offset_vec + thread_idx_x);
//当前线程加载哪一个列索引(0-31)，一个线程加载一个索引。Global的地址
const int * column_idxs_ = (reinterpret_cast<const int *>(column_idxs) + row_offset_vec + thread_idx_x);
//当前线程读取的向量，需要写入Shared Memory的地址
VecType* values_tile_base_ = (reinterpret_cast<VecType *>(values_tile) + thread_idx_x);
//当前线程读取的列索引，需要写入Shared Memory的地址
int *column_idxs_tile_base_ = (reinterpret_cast<int *>(column_idxs_tile) + thread_idx_x);


sparse_tile_loader.Load()

从Global Memory加载整个稀疏矩阵tile（8*32）到Shared Memory上。每个线程加载一个V8向量，即每个线程加载一个float4数据。values_ 指向地址即图中五角星，线程tid加载红色箭头的向量。

sparse_tile_loader.Load()
//将指针value_tile指向当前线程要写入的Shared Memory的地址
VecType *values_tile = values_tile_base_;
//将指针column_idxs_tile指向当前线程要写入的Shared Memory的地址
int* column_idxs_tile = column_idxs_tile_base_;
 
//迭代kThreadItemsN_次，其实就一次，即稀疏宽为kThreadItemsN_*Block_size
#pragma unroll 
for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
*(values_tile) = __ldg(values_);//当前线程从Global中加载向量到Shared
*(column_idxs_tile) = rhs_columns_ * __ldg(column_idxs_);//加载当前线程对应的索引
//指向下一个线程块对应地址
values_ += BlockWidth;
column_idxs_ += BlockWidth;
values_tile += BlockWidth;
column_idxs_tile += BlockWidth;
}


sparse_tile_loader.ZeroTiles()

*(values_tile) = reinterpret_cast<const VecType*>(kZeroValues)[0];

*(column_idxs_tile) = 0;

将共享内存中的Tile清零。

VecType *values_tile = values_tile_base_;//同上
int *column_idxs_tile = column_idxs_tile_base_;//同上

const half kZeroValues[VecLength] = {};

#pragma unrill
for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
    *(values_tile) = reinterpret_cast<const VecType*>(kZeroValues)[0];
    *(column_idxs_tile) = 0;
    values_tile += BlockWidth;
    column_idxs_tile += BlockWidth;
}


sparse_tile_loader.Residue(residue)

用于加载residue个残余数据。含义同sparse_tile_loader.Load()

VecType* values_tile = values_tile_base_;
int *column_idxs_tile = column_idxs_tile_base_;

#pragma unroll
for (int n_item_idx = 0; n_item_idx < kThreadItemsN_; n_item_idx ++){
    if (residue <= threadIdx.x) return;//一个线程对应一个残余，超过的线程不执行
//其余与Load一样
    *(values_tile) = __ldg(values_);
    *(column_idxs_tile) = __ldg(column_idxs_) * rhs_columns_;

    values_ += BlockWidth;
    column_idxs_ += BlockWidth;
    values_tile += BlockWidth;
    column_idxs_tile += BlockWidth;
    residue -= BlockWidth;
}





结构wmmaDenseTile

matrix+offset其实就是稠密矩阵首地址+当前线程块的Y坐标，等于当前线程块的首地址。再加上当前线程组的序号(每个线程组宽度为8个数据，正好是一个LoadType float4的长度)，最终matrix_base_就是当前线程组的首地址，即图中五角星位置。row_offsets_base_ 为lane_id一样的线程对应的一行的索引号。

//每个__ldg()指令可以加载多少个fp16数据
static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);
//K维一共加载次数-1
static constexpr int kTotalStep = Tile_N / 4 - 1;
//稠密矩阵列方向实际需要的加载次数。需要加载N/8次
const int rhs_columns_ = (rhs_columns / kValuesPerLoad_);
const int lane_id_;
const LoadType * matrix_base_ = (reinterpret_cast<const LoadType *>
(matrix + offset) + thread_group);
const int * row_offsets_base_ = (row_offsets + lane_id);
LoadType * matrix_fragment_ = (reinterpret_cast<LoadType *>(matrix_fragment));


wmmaDenseTile.LoadRow(int row_group_idx)和wmmaDenseTile.ResidueLoad(int row_group_idx, int residue)

row_group_idx是小循环迭代轮数，一次迭代加载四行，对应一个线程组的四个线程。row_offsets 是在row_offsets_base_的基础上，每多迭代一次，就向后四行。再根据matrix_base_加上row_offsets 对应行的地址，将该地址对应的一个LoadType加载到稠密矩阵的寄存器上。

而ResidueLoad是用于剩余行数小于4的情况。

__device__ __forceinline__ void LoadRow(int row_group_idx){
    const int *row_offsets = row_offsets_base_ + row_group_idx * 4;
    *(matrix_fragment_ + row_group_idx) = __ldg(matrix_base_ + *(row_offsets));
}

// Load the residual and compute the matrix product
__device__ __forceinline__ void ResidueLoad(int row_group_idx, int residue){
    if (lane_id_ >= residue) return;
    const int *row_offsets = row_offsets_base_ + row_group_idx * 4;
    *(matrix_fragment_ + kTotalStep) = __ldg(matrix_base_ + *(row_offsets));
}

结构wmmaComputeUtils8
static constexpr int kTotalStep = Tile_N / 4 - 1;//同上

// Shared memory buffer storing the lhs tile values
const float2* lhs_tile_ = 
(reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 2 + thread_group / 4);
// Register file fragment storing the rhs tile
const half* rhs_fragment_ = (rhs_fragment);//稠密矩阵寄存器地址
// Register file fragment to accumulate results into.
float* output_fragment_ = (output_fragment);//输出矩阵寄存器地址

const float2* lhs_tile_ = (reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 2 + thread_group / 4);先将lhs_tile从float4指针强转为float2，如下图。

reinterpret_cast&lt;const float2 *&gt;(lhs_tile)

而const float2* lhs_tile_ = (reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 2 + thread_group / 4);如下图所示，相同颜色代表原来一个VecType，并且同一块中的线程同时指向同一个float2数据。此外，小循环8次，刚好能计算完整个lhs_tile。

const float2* lhs_tile_

TileMAC根据小循环的进度，每个线程计算16个结果，存在线程独享的寄存器output_fragment_中。

__device__ __forceinline__ void TileMAC(int n_group_idx){
    float lhs_fragment[2];
            //声明两个float寄存器，lhs_fragment_float2指向这两个寄存器
    float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
    *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);//如上图
    int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
    const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);
            
    #pragma unroll
    for (int i = 0; i < 2; i++){
        asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"//D = A * B + C
            "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"//D
            "{%8, %9}, \t"                        //A
            "{%10, %11}, \t"                      //B
            "{%0, %1, %2, %3, %4, %5, %6, %7}; ": //C
            "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
            "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
            "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
            "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
            "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
            "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
        );
    }
}


为了对应mma.884的计算格式，采用了把float4转换成float2的方式。而mma的输入格式如下图，具体解释可以查看ptx文档[4]。

mma.sync.aligned.m8n8k4
结构wmmaOutputTile8
static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);
static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);

int lane_id_ = lane_id;
int thread_group_ = thread_group;
// The register file fragment with the results to store
float2* output_fragment_  = reinterpret_cast<float2 *>(output_fragment);
const int output_offset = (row_offset_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
StoreType* output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);
// The number of columns in the rhs matrix
int rhs_columns_ = cols / kValuesPerStore_;


输出数据类型OutType是float，存储类型StoreType是float4，而一共8*64=512个float输出数据。因此32个线程每个线程要存16个float，即4条_ldg()指令。

const int output_offset = (row_offset_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;计算的是每个线程在输出矩阵中第一个_ldg()地址，如下图。而Store()函数中的*(output_matrix_)、*(output_matrix_+1)、*(output_matrix_+8)、*(output_matrix_+9)对应的是图中4次存储数据的位置。




store示意图




__device__ __forceinline__ void Store(){
    // Step 1: warp shuffle to align the memory access
    int src_line = (lane_id_ + 2) % 4 + thread_group_ * 4;//2 3 0 1 6 7 4 5 ...... 30 31 28 29

    #pragma unroll
    for (int i = 0; i < 4; i++){
        __align__(8) float temp[2];
        float2* temp_float2 = reinterpret_cast<float2 *>(temp);
        //线程混洗，交换不同线程的值    
        if (lane_id_ < 2) *(temp_float2) = output_fragment_[i * 2 + 1];
        else *(temp_float2) = output_fragment_[i * 2];
        temp[0] = __shfl_sync(0xffffffff, temp[0], src_line, 32);
        temp[1] = __shfl_sync(0xffffffff, temp[1], src_line, 32);
        if (lane_id_ < 2) output_fragment_[i * 2 + 1] = *(temp_float2);
        else output_fragment_[i * 2] = *(temp_float2);
    }

    if (kTypeConvert != 1){
        float* output_fragment_float = reinterpret_cast<float *>(output_fragment_);
        OutType* output_fragment_outType = reinterpret_cast<OutType *>(output_fragment_);
        #pragma unroll
        for(int i = 0; i < 16; i++){
            output_fragment_outType[i] = (OutType)output_fragment_float[i];
        }
    }

    StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
    *(output_matrix_) = *(output_fragment_storetype);
    *(output_matrix_ + 1) = *(output_fragment_storetype + 2);
    *(output_matrix_ + 8) = *(output_fragment_storetype + 1);
    *(output_matrix_ + 9) = *(output_fragment_storetype + 3);
}


但由于每个线程输出的数据如下图所示，如果直接将output_fragment的数据按顺序存储，则结果不正确。所以利用了行重索引，在加上__shfl_sync()去混洗数值，将正确结果写回，如下图。

mma.884输出和混洗示意图

这段写的很巧妙，建议亲自去看一下，我也是看了好久。建议想搞明白的同学，也自己去理解一下。




注：

写这些注释的前后时间跨度很大，写法上肯定有问题，内容也可能有错误，欢迎大家在评论区讨论、指正。

参考
^Zhaodong Chen et al. "Efficient Tensor Core-Based GPU Kernels for Structured Sparsity under Reduced Precision". In: SC (International Conference for High Performance Computing, Networking, Storage, and Analysis). 2021. https://ieeexplore.ieee.org/document/9910106
^Trevor Gale et al. "Sparse GPU Kernels for Deep Learning". In: SC (International Conference for High Performance Computing, Networking, Storage, and Analysis). 2020. https://ieeexplore.ieee.org/document/9355309
^Sparse稀疏矩阵主要存储格式总结 https://zhuanlan.zhihu.com/p/188700729
^Parallel Thread Execution ISA Version 8.2 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
