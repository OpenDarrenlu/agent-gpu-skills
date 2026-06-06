# cute TiledMMA

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/699255051

---

继续 cute 系列的学习，之前的学习链接在这里：

HPC小菜鸟：Cute TiledCopy
30 赞同 · 1 评论 文章

先阅读下 read 大佬关于 cute TiledMMA 的解释：

reed：cute 之 MMA抽象
298 赞同 · 41 评论 文章
MMA 抽象

下面简单总结一下MMA 抽象。整体上如下面这个图：

MMA 抽象

最底层是 硬件和对应的指令集， 针对不同的GPU架构，NVidia提供了不同的指令（MMAOperation）来使用Tensor Core。MMA_Atom 如其名字中的Atom所表达的“原子”，这是硬件提供的能执行的矩阵乘法的最小的单位，其能完成一个特定规格(如MNK)的矩阵乘法问题D = A x B + C，可以认为是 MMAOperation 的简单封装， MMA_Traits 便是这种封装的桥梁，提供了封装需要但是 MMAOperation 没有的信息。如封装 SM80_16x8x16_F32F16F16F32_TN 得到的 MMA_Atom，能完成 MNK = 16，8，16 大小的矩阵乘法计算。

在MMA_Atom之上进行扩展，形成更大的矩阵乘法能力，即TiledMMA，它应当是原子的整数倍。这种扩展可以是执行单元层面的（需要更多的执行线程）也可以是对Atom重复执行（warp level需要更多的循环）。无论哪一个种扩展都能提供更大的矩阵乘法。对于实际需要的计算问题大小，再定义完 TiledMMA 后，便可以通过 get_thread_slice(threadID) 接口，根据实际问题规模的大小获得每个线程所需要负责计算的部分。ThrMMA 就是完成这个工作：将逻辑的矩阵根据提供的线程号（threadIdx.x）来获得自己这个线程的任务（可以是warp level的等）。在ThrMMA得到各个线程的计算任务后，各个线程同时调用cute::gemm函数完成各个线程级别的任务下发，最终所有线程等结果体现为大块的D = A x B + C的任务完成计算。

一个简单的 TiledMMA

TileMMA 定义如下：

实际应用中，可以使用 make_tiled_mma api 简化代码。

不妨直接定义一个 TiledMMA 看看：

  using MMA_Atom_Arch  = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  TiledMMA mmaC = make_tiled_mma(MMA_Atom_Arch{});
  print(size(mmaC)); printf("\n");
  print_latex(mmaC);


正如它的名字那样，它可以处理 16x8x8大小的矩阵乘法计算。

注意的是：size（mmaC） = 32， 因为安培架构上的tensor core 指令，是以warp 为单位执行的，即一个 warp内的32个线程，协同作业完成 16x8x8大小的矩阵乘法计算。 到底是如何协同作业的呢，看看 print_latex 的输出结果就知道了。

SM80_16x8x16_F32F16F16F32_TN

上面的 latex 图，描述了一个 warp 内如何计算 MNK=16,8,16 的矩阵问题，其中 A = (M,K), row-major，B=（K，N), col-major， C = (M, N), row-major。 实际使用过程中，数据是否是 row-major 和 col-major 并不影响计算结果的正确是，但是会影响计算效率（应该主要影响带宽的效率）。

对照 PTX 指令集的文档看一看，可以看出 latex 中描述的线程之间的协作方式和指令集文档中的描述一致。

第一次看到这个图的时候有一个疑问，比如计算 C 中的第0行第0列的结果时，按照以前 cuda core 的编程思想，寄存器是单个线程私有的，T0 线程负责这个结果就需要load A 中的完整第一行和B中的完整第一列到自己的线程level寄存器中。但是从图上看，这些数据被不同的线程 load 了。其实在第三代 Tensor Core 架构中， Tensor Core 可以操作32个线程的所有寄存器。可以查看安培架构的白皮书验证。

图中，最左边的 FFMA， 每个绿色小方块的表示一个 cuda core，只能操作自己的thread level 寄存器，对应的每一个蓝色小方块，最右边的A100 上的 tensor core， 整体 TC 可以操作32 个 Thread 的所有寄存器。这样可以显著减少寄存器访问次数以及数据冗余问题。

在这个问答帖中，read 大佬也做了相关的描述：

拓展型的 TiledMMA

前面定义的 TiledMMA 并未进行拓展（只能处理16x8x16这么大小问题规模），下面定义一个拓展的：

  static constexpr int kNWarps = 4;
  using MMA_Atom_Arch  = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  TiledMMA mmaC = make_tiled_mma(MMA_Atom_Arch{},
                                Layout<Shape<Int<kNWarps>,_1,_1>>());
  print(size(mmaC)); printf("\n");
  print_latex(mmaC);

这种，执行单元层面上的重复需要更多的执行线程，在 M 方向拓展 4次，因此 size(mmaC) = 128。如果 Layout 的shape太大了，线程数会大大增加，一般不超过256个，大多数GPU 一个 CTA 最大限制的线程数是1024。

由于打印的 latex 图较大，这里只放矩阵 A 和 C 的部分截图，可以看出在 M 方向上的拓展。

Atom 重复执行：

  static constexpr int kNWarps = 4;
  using MMA_Atom_Arch  = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMma = TiledMMA<
                        MMA_Atom_Arch,
                        Layout<Shape<Int<kNWarps>,_1,_1>>,
                        Tile<Int<16 * kNWarps>, _16, _16>>;

  TiledMMA mmaC = TiledMma{};
  print(size(mmaC)); printf("\n");
  print_latex(mmaC);

size(mmaC) 还是等于 128，Tile<Int<16 * kNWarps>, _16, _16> 表示这个 TiledMMA 最终需要划分的问题规模 MNK 为 64，16，16。因此除了执行单元层面上重复四次，在 N 方向上也要循环执行两次。打印看看 B 矩阵的 latex 图：

可以看出 N 方向上的重复，而前面不同的是，这不是执行单元的重复，所以线程号没有增加。

（这里提出一个问题，因为 M 方向上已经重复了四次，有四个warp，但是B矩阵看起来总是第一个 Warp 来load，B 矩阵应该每个 warp 都要来 load 吧）。

该例子正是 vllm_flash_attention 中的用法: https://github.com/vllm-project/flash-attention/blob/main/csrc/flash_attn/src/kernel_traits.h#L74

一个简单的 gemm

写一个简单的 gemm 用来验证上述 MMA 的理解，用基本策略的sliced-k方法，但是不使用shared_memory(为了简单)。

demo 使用 TC 的 SM80_16x8x16_F16F16F16F16_TN 指令集，该指令集处理的问题规模还是 16x16x8，但是输入和输出都是 FP16。

gemm_device kernel

逻辑很简单，在 k 维度的循环里，每次循环时，每个 CTA 只需要处理 A中的（bm, bk），B中的（bn，bk），以及 C中的（bm，bn）。

整体逻辑如下：

先得到每个 CTA 负责计算的 ABC 上的数据，使用 local_tile api， 得到 gA, gB, gC.
根据 TiledMMA 划分逻辑，得到每个线程的划分逻辑 ThrMMA
根据实际的问题规模，ThrMMA 会划分每个CTA 上对应的gA，gB，fC矩阵，得到 tAgA， tBgB ，tCgC
实际问题规模一般是 TiledMMA 重复逻辑后对应的MNK shape 的整数倍
创建对应大小的寄存器矩阵，tArA，tBrB，tCrC，
将数据从global memory 拷贝到寄存器，调用mma指令，计算得到
将计算结果从shared memory 拷贝回 global memory
template<class T, int bM, int bN, int bK,
        class TiledMMA>
__global__ void gemm_device(const T* Aptr, const T* Bptr, T* Cptr, 
                            int m, int n, int k,
                            TiledMMA tiled_mma) {
  using namespace cute;
  using TA = float;
  using TB = float;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{})); //(m,k) row-major
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{})); //(n,k) row-major
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{})); //(m,n) row-major

  // Get the appropriate blocks for this thread block
  int ix = blockIdx.x;
  int iy = blockIdx.y;             
  Tensor gA = local_tile(A, make_tile(Int<bM>{}, Int<bK>{}), make_coord(ix, _));  // (b_M,b_K,num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<bN>{}, Int<bK>{}), make_coord(iy, _));  // (b_N,b_K,num_tile_k)
  Tensor gC = local_tile(C, make_tile(Int<bM>{}, Int<bN>{}), make_coord(ix, iy)); // (b_M,b_N)

  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  Tensor tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
  

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_K, MMA_N)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)

  clear(tCrC); 
  int num_tile_k = size<2>(gA);

  #pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

main函数代码：
int main(int argc, char** argv)
{
  using namespace cute;
  using Element = __half;


  constexpr int M = 4096;
  constexpr int N = 1024;
  constexpr int K = 512;

  // Define a tensor shape with dynamic extents (m, n)
  // Allocate and initialize
  thrust::host_vector<Element> h_A(M*K);
  thrust::host_vector<Element> h_B(K*N);
  thrust::host_vector<Element> h_C(M*N);
  thrust::host_vector<Element> h_C_ref(M*N);

  for (size_t i = 0; i < h_A.size(); ++i) {
    auto rand_value = rand() % 10 - 5;
    h_A[i] = static_cast<Element>(rand_value);
  }
  for (size_t i = 0; i < h_B.size(); ++i) {
    auto rand_value = rand() % 10 - 5;
    h_B[i] = static_cast<Element>(rand_value);
  }
  for (size_t i = 0; i < h_C.size(); ++i) {
    h_C[i] = static_cast<Element>(0.0f);
    h_C_ref[i] = static_cast<Element>(0.0f);
  }

  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Element> d_B = h_B;
  thrust::device_vector<Element> d_C = h_C;
  thrust::device_vector<Element> d_C_ref = h_C_ref;

  // Make tensors, default col-major
  const int bM = 128;
  const int bN = 128;
  const int bK = 32;

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using MMA_Atom_Arch = MMA_Atom<mma_traits>;
  // 直接可以简写为下面这句
  // using MMA_Atom_Arch  = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
  static constexpr int kNWarps = 4;
  using TiledMma = TiledMMA<
                        MMA_Atom_Arch,
                        Layout<Shape<Int<kNWarps>,_1,_1>>,
                        Tile<Int<16 * kNWarps>, _16, _16>>;
  TiledMMA mmaC = TiledMma{};
  // print(mmaC);
  make_tiled_mma(MMA_Atom_Arch{});
  dim3 dimGrid(size(ceil_div(M, bM)), 
               size(ceil_div(N, bN)));
  dim3 dimBlock(size(mmaC));
  print(size(mmaC)); printf("\n");
  const Element* Aptr = thrust::raw_pointer_cast(d_A.data());
  const Element* Bptr = thrust::raw_pointer_cast(d_B.data());
  Element* Cptr = thrust::raw_pointer_cast(d_C.data());
  gemm_device<Element, bM, bN, bK><<<dimGrid, dimBlock, 0, 0>>>(Aptr, Bptr, Cptr, 
                                            M, N, K,
                                            mmaC);
  cudaDeviceSynchronize();


  // 使用 cublas 库计算
  // Initialize cuBLAS
  cudaSetDevice(0);
  cublasHandle_t handle;
  checkCublasError(cublasCreate(&handle), "cuBLAS initialization failed");
  half alpha = half(1.0f);
  half beta = half(0.0f);
  Element* Cptr_ref = thrust::raw_pointer_cast(d_C_ref.data());
  checkCublasError(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    Bptr, K,
                    Aptr, K,
                    &beta,
                    Cptr_ref, N), "cuBLAS SGEMM failed");

  h_C = d_C;
  h_C_ref = d_C_ref;
  for (int i = 0; i < M*N; i++) {
    if (std::abs(__half2float(h_C[i]) - __half2float(h_C_ref[i])) > 0.01) {
      std::cerr << "Error. h_C[" << i << "]: " << __half2float(h_C[i]) << ",   h_C_ref[" << i << "]: " << __half2float(h_C_ref[i]) << std::endl;
      return -1;
    }
  }
  printf("Success!\n");
  cudaDeviceSynchronize();
  return 0;
}


TileMMA 通过执行单元的重复以及执行次数的重复，可以完成 MNK shape 为 <64, 16, 16>的任务，kernel 内部中，具体问题大小 bm bn bk 为（128， 128， 32），ThrMMA 的 partition api 会进一步增加执行次数的重复，来完整完成 bm bn bk 的计算（个人理解，可能有不当的地方）

最后的执行结果：

每个CTA 128个线程，输出结果于cublas api 对齐。
