# Cute TiledCopy

**作者**: weishengying

**原文链接**: https://zhuanlan.zhihu.com/p/705573338

---

本 blog 是 cute TiledCopy 的基础实践，非常基础，大佬可略过。在阅读 read 大佬以及一些其他文章后，自己写的一些 demo 来实践，加深理解。

建议先看看read 大佬系列：

reed：cute 之 Copy抽象

还有这个：

Anonymous：CUTLASS CuTe GEMM细节分析（二）——TiledCopy与cp.async

blog中的 demo 代码在仓库

tiled_copy.cu

TiledCopy参数的理解

先从一个非常简单的TiledCopy入手：

    using g2r_copy_op = UniversalCopy<T>;
    using g2r_copy_traits = Copy_Traits<g2r_copy_op>;
    using g2r_copy_atom = Copy_Atom<g2r_copy_traits, T>;

    using G2RCopy = decltype(make_tiled_copy(g2r_copy_atom{},
                        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                        Layout<Shape<_1, _4>>{}));
    print(G2RCopy{})


这样定义的意思是：

定义一个线程组（CTA），该CTA包含16*8 个线程，线程的组织方式为row-major，也即线程 id 增大的方向是行，每个线程负责拷贝 shape为（1，4）的子 tensor，即每个线程负责一行中的4 个元素，拷贝指令使用 g2r_copy_atom。
概括的说，定义一个 CTA，在 CTA level 和 thread level 层面定义了每个 CTA 和 thread 的任务

细心的同学，可能会发现，

 Layout<Shape<_1, _4>>{}

这句并未指定 stride，因此默认是 col-major，看起来和线程以 row-major 的组织方式不同。个人实践证明， Layout<Shape<_1, _4>>{} 不管 stride 的定义，即 stride 不管定义成多少都可以，cute内部不需要 stride 值，有懂的大佬可以教一下我原因……^_^。

将模板参数T指定为 float 类型，并利用CuTe的print函数打印这个TiledCopy的实例，得到如下结果：

图中的 Tiler_MN（16， 32） 正是前面所说有，在 CTA level层面，一次迭代需要拷贝（16，32）个元素。如果每个线程每次拷贝只拷贝一个元素，那么在 thread level 层面，就要循环四次。

TiledLayout_TV 的含义，这里引用 Anonymous 的解读：

TiledLayout_TV是一个复合的Layout，Layout本质上是一个映射，它能够将一个由整数构成的坐标转换为一个标量offset。TiledLayout_TV所表达的含义为：给定一个Thread的ID，以及这个Thread所负责的Tensor分块中某个元素的坐标，返回这个元素在Tiled_MN中的坐标。
等等，Layout输出不是一个标量offset吗？为什么返回的是一个坐标？在CuTe Layout官方文档中给出了说明，对于一个标量，我们可以根据Shape各个维度的大小，将一个标量值转换为一个坐标。例如，对于Shape(3, 3)，给定一个值5，它可以转换为坐标(2, 1)。
我们结合一个具体的例子来说明TiledLayout_TV的作用。假设，TiledLayout_TV为上文中的((8, 16), 4):((64, 1), 16)，我们想知道ID为9的Thread，它拷贝的Tensor分块中，坐标为(0, 2)的元素，对应Tiler_MN中的坐标是多少？
1. 首先，将Thread ID 9转换为Shape(8, 16)的坐标：(1, 1)，坐标(0, 2)是Shape(1, 4)的坐标，这个Shape可以简化为(4,)，因此坐标也可以简化为(2,)。因此，输入的坐标为((1, 1), 2)。
2.计算offset = 1 x 64 + 1 x 1 + 2 x 16 = 97（坐标乘以步长，然后求和）
3.将97转换为Shape(16, 32)的坐标，(16, 32)即Tiler_MN，结果为：(1, 6)
因此，对于Thread 9这个线程来说，它拷贝的Tensor分块中坐标为(0, 2)的元素位于Tiler_MN的(1, 6)位置处。事实上，Thread 9负责Tiler_MN上(1, 4:8)这个分块。

通过 print_lattex 直接打印 G2RCopy 也可以看出线程组拷贝的协作方式。以下是部分截图。

正如前面所说，一个线程负责拷贝4个行元素，线程id增大的方向是行，第9个线程拷贝对应子tensor 的第2个元素，对应的Tile_MN坐标也就是（1，6），它负责（1，4:8）这个分块！

Thr_copy

如果说 Tiled_copy 描述了 CTA 在一次迭代过程中的线程之间的协作方式，那么Thr_copy 就是 CTA 中每个 thread 需要负责的元素（线程level）。即正如 read 大佬所说：

TildCopy提供的是逻辑上的拷贝的概念，在具体的kernel执行之时，为了复合CUDA的编程范式，需要写成线程级别的指令，ThrCopy可以实现将大块的数据根据TiledCopy所描述的划分规则，通过提供当前线程的线程号threadIdx.x对大块的Tensor进行划分，得到当前线程为了完成D = S 拷贝所需要该线程做的任务；


假如有一个 tensor shape 为(32, 32)， 前面定义的 Tiled_copy 描述了一个 thread_block 有（16， 8）个线程，每个线程负责4个元素，则thread_block level 层面迭代一次可以处理（16， 8*4）个原数，那么需要迭代两次才能处理完（32， 32）的tensor，每个 thread 共需要处理 8 个元素。写个demo验证下：

int main(int argc, char** argv)
{
  using namespace cute;
  using Element = float;

  Layout thr_layout = make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
  Layout vec_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
  using AccessType = Element;
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      thr_layout,                  // thread layout
      vec_layout);                 // vector layout (e.g. 4x1)
    
  auto tensor_shape = make_shape(32, 32);
  auto tensor_stride = make_stride(Int<32>{}, Int<1>{});

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape));
  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
  }

  Tensor tensor_S = make_tensor(h_S.data(), make_layout(tensor_shape, tensor_stride));
  auto thr_copy = tiled_copy.get_thread_slice(0);

  Tensor thr_tile_S = thr_copy.partition_S(tensor_S);
  print_tensor(tensor_S);
  print_tensor(thr_tile_S);
  return 0;
}


很显然， id=0 的tread，需要拷贝以下元素（第0行的前四个+第16行的前四个）

向量化读取

正如前面所描述， 一个大小为 （16， 8）thread_block 中，一次 thread_block level 层面的迭代过程中每个thread负责4个元素的 copy，如果每次只拷贝一个元素，那么在thread level层面需要循环四次，才能完成4个元素的拷贝。 那这4个元素是否能够使用向量化 load 指令，一个load 指令（每个thread只需要循环一次）就可以完成4个元素的拷贝呢，答案是当然。稍微改改 Tiled_copy 的定义即可。

    using T = cutlass::AlignedArray<Element, 4>;
    using g2r_copy_op = UniversalCopy<T>;
    using g2r_copy_traits = Copy_Traits<g2r_copy_op>;
    using g2r_copy_atom = Copy_Atom<g2r_copy_traits, T>;

    using G2RCopy = decltype(make_tiled_copy(g2r_copy_atom{},
                        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                        Layout<Shape<_1, _4>>{}));
    print(G2RCopy{})


写个 demo 验证下：

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Slice the tensors to obtain a view into each tile.
  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)

  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)

  // Construct a register-backed Tensor with the same shape as each thread's partition
  // Use make_fragment because the first mode is the instruction-local mode
  Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(tiled_copy, thr_tile_S, fragment);
  copy(tiled_copy, fragment, thr_tile_D);
}


int main(int argc, char** argv)
{
  // Given a 2D shape, perform an efficient copy
  using namespace cute;
  using Element = float;

  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(128, 64);
  auto tensor_stride = make_stride(Int<64>{}, Int<1>{});

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  // Make tensors, default col-major
  Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape, tensor_stride));
  Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape, tensor_stride));


  // Tile tensors
  // Define a statically sized block (M, N).
  // Note, by convention, capital letters are used to represent static modes.
  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
    return -1;
  }

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  // These will be used to determine the CUDA kernel grid dimensions.
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

  // Thread arrangement
  Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
  Layout vec_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
  using AccessType = cutlass::AlignedArray<Element, size(vec_layout)>;
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      thr_layout,               // thread layout
      vec_layout);                 // vector layout (e.g. 4x1)

  // Determine grid and block dimensions
  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));

  // Launch the kernel
  copy_kernel_vectorized<<< gridDim, blockDim >>>(
    tiled_tensor_S,
    tiled_tensor_D,
    tiled_copy);

  cudaError result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  // Verify
  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}


使用 nvcc -ptx 指令验证下到底是不是如期望的那样，load指令集如下：

可以看出，一次循环迭代中一行的 8 个 thread，只发射了8个load指令，v4.f32 表示4个float组成的vector

如果将测试demo中 AccessType 改为：

using AccessType = float

则对应的load指令集如下：

有些理解可能不对，欢迎大佬们指正～

实际应用

最后看看在 flash_attention 的应用：

这里定义了拷贝部分 QKV 到shared memory 的方式，每个 CTA 中有 kNThreads 个线程（处理row-major的tenosr），每个线程负责拷贝一行中的8个元素，同时使用向量化的 copy 指令，一个可以拷贝 128bit，对于 half 数据，正好一次可以拷贝完。如果是 float数据，每个线程需要循环执行两次才能完成8个元素的拷贝。




更新

进一步学习之后，上述理解存在偏差，对于 tiledcopy 这些理解"还行"，但是严格来说是不准确的。后续学习完 tiledmma 之后会继续更新该 blog
