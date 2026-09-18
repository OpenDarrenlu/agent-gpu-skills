# cute swizzle

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/706796240

---

继续学习 cute 系列！

bank conflict

首先理解什么是 bank 冲突。这里 read 大佬的这篇 blog 的前面讲述的很明白。

reed：cute 之 Swizzle
560 赞同 · 73 评论 文章

在NVidia的架构中，shared memory包含32个bank，bank中可寻址的基本单元为4byte，如图1所示，每个bank为黑框所包含的单元，用户看到的地址空间为箭头所示的方向，即相邻的4byte占用不同的bank。

Figure 1. 共享内存bank结构和地址连续方向

同一个 Wrap 中的线程访问同一个 bank 时，就会产生 bank 冲突。如下图：

T0 和 T2 同时访问 bank1 ，则这两次访问会被排队执行，即先访问该bank的一个地址，然后再访问第二个地址，这样两次访问在发射任务维度上（产生访问请求指令）时间维度上是并行的，但是在真正bank读写数据在时间维度上是串行的。这就是所谓的bank conflict。

为了提高 load 效率，一般会使用向量化的读取命令，一次读取 128bit，也就是 16byte，对应四个bank。那么 8 个线程就可以一次完成 32 个bank 的load，所以问题简化为研究 T0 - T7 or T8 - T15 or T16 - T23 or T24 - T32 这 8 个线程内没有bank 冲突（此时32线程内的bank冲突无法避免，这样能够最小化bank冲突，每8个线程成称为一个phase）。假设，我们定义的 shared memory layout 为（bM，64 or 32）, 数据类型为 half， 如果读取方式按照以下的方式读取：

每个 phase 没有bank 冲突

在 global --> smem 的过程中，往往是这种写入方式（通过定义 tiled_copy 控制），所以这个过程中，一般没有 bank 冲突。但是在 smem --> register 中，是通过 ldmatrix 指令进行copy的，那么有必要先熟悉下这个指令。

官方文档在这里：

可以看看这个作者提供的 demo：

以及 read 大佬的回答：

以 ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ] 指令为例，

单线程执行该指令表示从 [ %4 ] 地址加载连续的 128bit（8个b16数据）到自己的4个寄存器{ %0, %1, %2, %3 }中，每个寄存器是32bit，所以刚好存下。如果warp level执行该指令，则该指令可以一次 copy 4 个 8x8 的半精度矩阵，即 16x16 的半精度矩阵。 这时warp中每个线程需要 load 16x16/32 = 8 个半精度元素，正好是 16x8=128bit，每个线程执行一次该指令。

如上图所示（这个图官方的出处在哪我还没找到……^_^，不知道有没有大佬知道）， 左边正好是一个 16x16的fp16 矩阵，右边是把左边 smem 对应部分的数据，加载到每个线程自己的寄存器中。右边寄存器的数据排布方式是这个 mma.m16n8k16 mma 指令集对输入矩阵A要求的数据在warp各个线程内部寄存器中的布局。

更新：

上面这个图可能有一点不对（大差不差），通过下面的demo打印latex验证了下：

    // copy from shared memory to register
    // use mma tiled ,so no tiled here
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // mma
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 1;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 16;
    static constexpr int kMmaPN = 8;
    static constexpr int kMmaPK = 16;
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
    
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    print_latex(s2r_tiled_copy_a);


结果如下：

T0 线程加载左边 T0 绿色部分的8个fp16 元素（这8个fp16必须连续），然后分到T1，T2，T3这三个线程的寄存器中。（这点不是很明白，寄存器不是每个线程私有的吗，为啥还可以这样分配，有一些说法是每个线程执行 ldmatrix 指令，加载对应部分连续8个元素到自己的寄存器中，然后 warp 内部进行数据交换，最终达到期望的布局， 但是我在指令中并未看到 shfl.sync之内的命令）。然后 tensor core 直接使用warp内所有线程的寄存器数值进行矩阵乘法计算，如下图。

tensor core 可以同时 load warp 线程中的所有寄存器，看起来就相当于warp之间寄存器数据共享，这样可以增加传输带宽和减少数据冗余。ok，既然 ldmatrix 指令 load sgmem 数据到寄存器的方式确定了，那就再看看有没有 bank 冲突。

实际测试中，也并不需要线程交换指令，只使用 ldmatrix 指令，就可以完成 MMA 指令要求的数据布局，如下 demo： 神叨：cuda的ldmatrix指令的详细解释

线程从 smem 读取的方式如下图：

右边的smem的K维度，不一定是32（由开发者自定义），但依然存在bank冲突

很显然，T0、T2、T4、T6 之间 T1、T3、T5、T7 等线程之间存在bank 冲突。cute swizzle语义就是解决这个问题。很显然，我们改变一下 shared memory 的布局，以及每个线程读取的地址就行了。

这里引用 read 大佬对 swizzle 语义的解释：

swizzle<B,M,S> 会作用在 shared memory 的 layout 。比如定义了 Swizzle<3,3,3> 后，Swizzle中M为3，所以8个元素形成一个新的最小的元素，即8x2byte = 16byte， 刚好等于每个thread load的最大长度；Swizzle中S为3，所以2D空间中一行包含8个元素（8列），则有8x16byte = 128byte，128byte为shared memory一个phase无conflict访问所有bank的最大宽度；Swizzle中B为3，则2D空间irow更新的间隔为8。如此则实现了将一个逻辑的空间向2D的shared memory空间的映射，其中列的宽度为128byte占满所有的bank，行列异或后得到新的列号，避免了在bank方向（亦即icol方向）的冲突。

仍然以类似上面的shared memory layout为例，假设为smem 为（16，32）：（32，1），然后在这个 layout 的基础上使用 swizzle， cute 写法类似如下：

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<32>{}),
                  make_stride(Int<32>{}, Int<1>{}))));
  
  // 这是未使用 Swizzle 语义的 smem layout
  // using SmemLayoutAtom = decltype(
  //     make_layout(make_shape(Int<8>{}, Int<32>{}),
  //                 make_stride(Int<32>{}, Int<1>{})));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<8>{}, Int<32>{})));


首先形成下图(只画出重要部分，其他部分可以类推)：

接下来牛逼的来的，经过一次异或计算（同是0，异是1），ibank = irow^icol, 比如第一行，irow=0, 所以第一行保持不变，第二行开始，就要变了，变化如下：

经过swizzle变换之后的smem layout

相同颜色的线程序列号属于同一phase，可以看出此时没有了 bank 冲突。这个图显示，大部分线程加载的数据起始地址都发生了改变（相同颜色表示相同的数据内容）。再来看看 read 大佬的图，就很清楚了。

read 大佬的图和我的分析稍有出入，但是表达的意思都是一样的，通过swizzle 语义，改变了 smem 中的数据排布（或者说相对位置）。

下面写个 demo 验证一下：

定义一个 smem , layout 为（8，32）：（32，1），然后通过 tiled_copy 把一个相同 layout 的gmem tensor 从拷贝到这个smem上，gmem tensor 初始化为 0——256，拷贝完成后打印验证 smem 。

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

template<class T, class SmemLayoutA, class G2SCopyA>
__global__ void gemm_device(const T* Aptr, int m, int k) 
{
  using namespace cute;
  Tensor gA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  // Shared memory buffers
  __shared__ T smemA[size(SmemLayoutA{})];
  Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});        // (BLK_M,BLK_K)

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
  const auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY_M, CPY_K)
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY_M, CPY_K)
  cute::copy(G2SCopyA{}, tAgA_copy, tAsA_copy);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  if(threadIdx.x==0){
    print(gA); printf("\n");
    for (int i = 0; i < sA.size(); i++) {
        if(i % k == 0)
            printf("\n\n");
        printf("%f ", __half2float(sA.data()[i]));
    }
    printf("\n");
  }
}

int main(int argc, char** argv)
{

  using namespace cute;
  using T = half;
  
  int M = 8;
  int K = 32;
  thrust::host_vector<T> h_A(M*K);
  for (int i = 0; i < h_A.size(); ++i) {
    h_A[i] = static_cast<T>(i);
  }
  thrust::device_vector<T> d_A = h_A;
  const T* Aptr = thrust::raw_pointer_cast(d_A.data());

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<8>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  
  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;
//   使用 Swizzle 语义的 smem layout
//   using SmemLayoutAtom = decltype(composition(
//       Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
//       make_layout(make_shape(Int<8>{}, Int<32>{}),
//                   make_stride(Int<32>{}, Int<1>{}))));
  
//   这是未使用 Swizzle 语义的 smem layout
  using SmemLayoutAtom = decltype(
      make_layout(make_shape(Int<8>{}, Int<32>{}),
                  make_stride(Int<32>{}, Int<1>{})));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<8>{}, Int<32>{})));
  dim3 gridDim(1);
  dim3 blockDim(size(G2SCopyA{}));

  print(size(G2SCopyA{})); printf("\n");
  gemm_device<T, SmemLayoutA, G2SCopyA>
              <<<gridDim, blockDim>>>(Aptr, M, K);
  cudaDeviceSynchronize();
  
  // print_layout(SmemLayoutA{});
}


输出结果如下，符合期望：

未使用swizzle语义的 smem

作为对比，使用 swizzle<3,3,3> 作用于smem layout 后：

使用 swizzle 之后的 smem

不妨打印一下两个 smem 的layout，看看为啥 tiled_copy + smem layout 能够做到这件事。

  //   使用 Swizzle 语义的 smem layout
  using SmemLayoutAtom_swillze = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<32>{}),
                  make_stride(Int<32>{}, Int<1>{}))));
  
  //   这是未使用 Swizzle 语义的 smem layout
  using SmemLayoutAtom = decltype(
      make_layout(make_shape(Int<8>{}, Int<32>{}),
                  make_stride(Int<32>{}, Int<1>{})));
  print_layout(SmemLayoutAtom{});
  print_layout(SmemLayoutAtom_swillze{});


输出结果为：

swizzle 对 layout 的改变

这个问题验证之后，还需要验证一个事情，那就是通过 ldmatrix 指令，每个 thread 的寄存器得到的数据，是期望的吗。下面是一个单测，定义一个 gmem 的（16， 32） 的tensor（这里M定义为16的原因是MMA指令能够处理A的shape最小为16x16），拷贝到 swillze 之后的 smem中，然后通过 ldmatrix 指令 load 到线程的寄存器中，然后打印寄存器里面的数值。

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>


template<class T, class SmemLayoutA, class G2SCopyA, class S2RCopyAtomA, class TiledMMA, int m ,int k>
__global__ void gemm_device(const T* Aptr) 
{
  using namespace cute;
  Tensor gA = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<m>{}, Int<k>{}), make_stride(Int<k>{}, Int<1>{}));
  // Shared memory buffers
  __shared__ T smemA[size(SmemLayoutA{})];
  Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});        // (M,K)
  
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _));  // (MMA, MMA_M, MMA_K)


  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
  const auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY_M, CPY_K)
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY_M, CPY_K)
  
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
  cute::copy(G2SCopyA{}, tAgA_copy, tAsA_copy);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // (CPY, CPY_M, CPY_K)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

  cute::copy(S2RCopyAtomA{}, tAsA, tCrA_view);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  if(threadIdx.x==0){
    print_tensor(tCrA_view);
  }
}

int main(int argc, char** argv)
{

  using namespace cute;
  using T = half;
  
  static constexpr int M = 16;
  static constexpr int K = 32;
  thrust::host_vector<T> h_A(M*K);
  for (int i = 0; i < h_A.size(); ++i) {
    h_A[i] = static_cast<T>(i);
  }
  thrust::device_vector<T> d_A = h_A;
  const T* Aptr = thrust::raw_pointer_cast(d_A.data());
  
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 1;
  static constexpr int kMmaEURepeatN = 1;
  static constexpr int kMmaEURepeatK = 1;

  static constexpr int kMmaVRepeatM = 1;
  static constexpr int kMmaVRepeatN = 1;
  static constexpr int kMmaVRepeatK = 1;

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_V_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaVRepeatM>{}, Int<kMmaVRepeatN>{}, Int<kMmaVRepeatK>{})));
  using TiledMMA =
      decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_V_RepeatT{}));

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<8>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  
  // shared memory to register copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  using S2RCopyAtomA = s2r_copy_atom;
  

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<64>{}),
                  make_stride(Int<64>{}, Int<1>{}))));
  
  // using SmemLayoutAtom = decltype(
  //     make_layout(make_shape(Int<16>{}, Int<32>{}),
  //                 make_stride(Int<32>{}, Int<1>{})));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<16>{}, Int<32>{})));
  static_assert(size(TiledMMA{}) == size(G2SCopyA{}));
  dim3 gridDim(1);
  dim3 blockDim(size(TiledMMA{}));

  print(size(G2SCopyA{})); printf("\n");
  gemm_device<T, SmemLayoutA, G2SCopyA, S2RCopyAtomA, TiledMMA, M, K>
              <<<gridDim, blockDim>>>(Aptr);
  cudaDeviceSynchronize();
  
  // print_layout(SmemLayoutA{});

}



输出结果如下：

线程0寄存器里面实际的数值
线程0期望得到的数据

符合预期！
