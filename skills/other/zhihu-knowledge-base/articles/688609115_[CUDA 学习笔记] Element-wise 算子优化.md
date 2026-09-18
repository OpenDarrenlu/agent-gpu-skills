# [CUDA 学习笔记] Element-wise 算子优化

**作者**: PicoPika一入infra深似海

**原文链接**: https://zhuanlan.zhihu.com/p/688609115

---

Element-wise 算子优化学习笔记
注: 本文主要是对文章 【BBuf 的CUDA笔记】一，解析OneFlow Element-Wise 算子实现 - 知乎 的学习整理

Element-wise 算子即针对输入 Tensor(可能有多个) 进行逐元素操作. 如 ReLU 操作.

朴素实现
__global__ void relu_kernel(float* input, float* output){
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  output[idx] = input[idx] < 0 ? 0 : input[idx];
}

问题分析

Element-wise 操作一般需要读取较多数据(tensor 中的元素), 然后对每个数据逐一计算. 一般是从全局内存加载, 因此容易看出, 这个操作是 memory bound 的.
朴素实现就是逐元素的"读取-计算-写入", 这样每个元素大部分时间必然花在内存的读写上.

对于 memory bound 的内核, 首先考虑的就是向量化内存访问, 即每个线程一次性加载更多的数据 (GPU 上最大是 128 比特即 16 字节)并计算. 这样也能增大计算强度.
在 Oneflow 的 Element-wise 算子设计中, 内核启动时的 grid_size 和 block_size 的选择也是有一定的学问.

Oneflow 实现

实现代码: oneflow/elementwise.cuh, how-to-optim-algorithm-in-cuda/elementwise.cu

ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<R, pack_size> ret;
  // 对向量化读取的元素两个两个的处理
  #pragma unroll
  for (int j = 0; j < pack_size; j += 2) { functor.Apply2(ret.elem + j, (in.elem + j)...); }
  return ret;
}

template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == false || pack_size % 2 != 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<R, pack_size> ret;
  // 对向量化读取的元素逐一处理
  #pragma unroll
  for (int j = 0; j < pack_size; ++j) { ret.elem[j] = functor((in.elem[j])...); }
  return ret;
}

template<int pack_size, typename FactoryT, typename R, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(FactoryT factory, int64_t n_pack, Packed<R, pack_size>* pack_r,
                 const Packed<IN, pack_size>*... pack_in, int64_t n_tail, R* tail_r,
                 const IN*... tail_in) {
  auto functor = factory(); // 仿函数
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  // 处理向量化的元素
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    pack_r[i] = ApplyPack<pack_size, decltype(functor), R, IN...>(functor, (pack_in[i])...);
  }
  // 处理尾部不够向量化的元素
  if (global_tid < n_tail) { tail_r[global_tid] = functor((tail_in[global_tid])...); }
}

template<size_t pack_size, typename FactoryT, typename R, typename... IN>
cudaError_t LaunchKernel(FactoryT factory, int64_t n, R* r, const IN*... in) {
  // 向量化后的分组数
  const int64_t n_pack = n / pack_size;
  const int64_t tail_offset = n_pack * pack_size;
  // 剩余不够向量化的个数
  const int64_t n_tail = n - tail_offset;
  int num_blocks;
  {
    cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
    if (err != cudaSuccess) { return err; }
  }
  ApplyGeneric<pack_size, FactoryT, R, IN...><<<num_blocks, kBlockSize, 0>>>(
      factory, n_pack, reinterpret_cast<Packed<R, pack_size>*>(r),
      (reinterpret_cast<const Packed<IN, pack_size>*>(in))..., n_tail, r + tail_offset,
      (in + tail_offset)...);
  return cudaPeekAtLastError();
}


值得一提的是, 在 Oneflow 的实现代码中, 像 pack_size 这种常量, 会作为模板参数传入; 而像 n_pack, n_tail 等所有线程公共的变量, 并不是在 kernel 中计算, 而是由 CPU 计算后作为参数传至 kernel (即 ApplyGeneric()) 中; 这样可以一定程度上减轻 GPU 寄存器压力并减少 kernel 中重复的公共计算.

Oneflow 向量化访存通用数据结构

OneFlow 针对不同数据类型提供了一个 Pack 数据结构, 以通用支持不同数据类型向量化.

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, int pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, pack_size> storage;
  T elem[pack_size];
};


template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
};

constexpr int kMaxPackBytes = 128 / 8;
constexpr int kMaxPackSize = 8;

constexpr int Min(int a, int b) { return a < b ? a : b; }

template<typename T>
constexpr int PackSize() {
  return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

template<typename T, typename U, typename... Args>
constexpr int PackSize() {
  return Min(PackSize<T>(), PackSize<U, Args...>());
}


上述代码中:

PackType<T, pack_size>, 定义了对总共 pack_size (元素总个数, 分为多个序列化访问)个 T 类型数据的序列化访问, 底层类型使用了 C++ 11 的 std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>, 即地址对齐的数据存储.
Pack 联合体主要是用在 Kernel 启动之前判断 Element-Wise 操作的输入输出 Tensor 对应的数据指针地址是否满足内存对齐的条件
PackSize<T>() 用于计算对 T 类型最大的向量化访问的元素个数. kMaxPackBytes 即 CUDA 向量化最大的访问粒度, 即上文提到的 128 比特(16 字节); kMaxPackBytes 定义了一个序列化访问的个数上限.
Packed 结构体即实际进行序列化访存的向量化的元素.
实际实现中, 通过 std::enable_if 判断算子是否包含 Apply2() 函数以执行相应的代码(具体见"【BBuf 的CUDA笔记】一，解析OneFlow Element-Wise 算子实现 0x3.2 向量化数据访问提升带宽"一节) , Apply2() 一般针对像 half 这种 CUDA 提供了 __hmul2() 函数可以直接两个一起算的情况.
相关手写 kernel
kernel 0
template <int pack_size>
__global__ void mul_coalesced(half *x, half *y, half* z, int64_t n){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t n_pack = n / pack_size;
  int64_t pack_off = n_pack * pack_size;

  auto pack_x = (reinterpret_cast<Packed<half, pack_size>*>(x));
  auto pack_y = (reinterpret_cast<Packed<half, pack_size>*>(y));
  auto pack_z = (reinterpret_cast<Packed<half, pack_size>*>(z));
  for (int i = idx; i < n_pack; i += gridDim.x * blockDim.x) {
    auto half_x = pack_x[i];
    auto half_y = pack_y[i];
    Packed<half, pack_size> half_z;
    #pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      half_z.elem[j] = half_x.elem[j] * half_y.elem[j];
    }
    pack_z[i] = half_z;
  }
  for (int i = pack_off + idx; i < n; i += gridDim.x * blockDim.x) {
    z[i] = x[i] * y[i];
  }
}


mul_coalesced() 是按照 Oneflow 实现手写的 kernel.
性能与 Oneflow 的实现相当. 间接证明了复杂的模板实际上并不影响最后的性能.
但如果使用 -G 选项编译, 则 Oneflow 实现(最差)和 mul_coalesced() 实现都不如朴素实现性能, 考虑到应该是代码的复杂度会增加 debug 模式编译生成的二进制的内容, 从而影响性能.

kernel 1
Packed<half, pack_size> *pack_x = (reinterpret_cast<Packed<half, pack_size>*>(x));
  Packed<half, pack_size> *pack_y = (reinterpret_cast<Packed<half, pack_size>*>(y));
  Packed<half, pack_size> *pack_z = (reinterpret_cast<Packed<half, pack_size>*>(z));
  for (int i = idx; i < n_pack; i += gridDim.x * blockDim.x) {
    // auto half_x = pack_x[i];
    // auto half_y = pack_y[i];
    // Packed<half, pack_size> half_z;
    #pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      pack_z[i].elem[j] = pack_x[i].elem[j] * pack_y[i].elem[j];
    }
    // pack_z[i] = half_z;
  }


mul_coalesced() 中直接逐一读写 Packed 结构体的内容. 经过 profile 可以看出, 这样 unroll 可以成功展开, 但是并没有达到向量化读写的效果, 因为每次都是按照 1 个元素的粒度直接读写的.

kernel 2
for (int i = idx; i < n_pack; i += gridDim.x * blockDim.x) {
    auto half_x = (reinterpret_cast<Packed<half, pack_size>*>(x))[i];
    auto half_y = (reinterpret_cast<Packed<half, pack_size>*>(y))[i];
    Packed<half, pack_size> half_z;
    #pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      half_z.elem[j] = half_x.elem[j] * half_y.elem[j];
    }
    (reinterpret_cast<Packed<half, pack_size>*>(z))[i] = half_z;
  }


mul_coalesced() 中没有在循环之前强制类型转换输入输出的指针 x y z. 而是在每次循环时临时转换并读写.
性能与 kernel 0 一致, 且实际 profile 发现生成的 SASS 与 kernel 0 中也是相同的. 也就是说, 强制类型转换只涉及地址计算的问题, 转换为汇编后并无差别.

kernel 3
for (int i = idx * pack_size; i < pack_off; i += gridDim.x * blockDim.x * pack_size) {
    auto half_x = *(reinterpret_cast<Packed<half, pack_size>*>(&x[i]));
    auto half_y = *(reinterpret_cast<Packed<half, pack_size>*>(&y[i]));
    Packed<half, pack_size> half_z;
    #pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      half_z.elem[j] = half_x.elem[j] * half_y.elem[j];
    }
    *(reinterpret_cast<Packed<half, pack_size>*>(&z[i])) = half_z;
  }


mul_coalesced() 与 kernel 2 类似, 区别在循环变量换成了以 half 为单位, 因此初始值和步幅都要 * pack_size 且结束条件由 n_pack 变为了 pack_off.
性能与 Oneflow 实现以及 mul_coalesced() kernel 0 基本一致. profile 显示生成的 SASS 与 kernel 0 有细微差别, 增加了一些指令, 应该是 * pack_size 引起的. 因此从代码上来看, 向量化读写时, 最好以向量化的数据结构(如 Packed)为单位进行索引, 以减少不必要的索引计算指令.

kernel 4
for (int i = idx; i < n_pack; i += gridDim.x * blockDim.x) {
    int4 tmp_x = (reinterpret_cast<int4*>(x))[i];
    int4 tmp_y = (reinterpret_cast<int4*>(y))[i];
    half *half_x = reinterpret_cast<half*>(&tmp_x);
    half *half_y = reinterpret_cast<half*>(&tmp_y);
    half half_z[pack_size];
    #pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      half_z[j] = half_x[j] * half_y[j];
    }
    (reinterpret_cast<int4*>(z))[i] = *(reinterpret_cast<int4*>(&half_z));
  }


mul_coalesced() 与 kernel 2 相似, 区别在没使用 Oneflow 定义的 Packed 结构体, 而是使用 int4 结构体作为容器来向量化读写, 在实际计算时在使用指针强转成 half 类型.
性能与 kernel 2 和 kernel 0 一致, 且实际 profile 发现生成的 SASS 与 kernel 0 中也是相同的. 因此, 向量化读取的"容器"数据结构也不影响实际 SASS 汇编的生成和实际的性能. Oneflow 中的 Packed 结构体的优势在于不需要像上面一样繁杂的强制类型转换(容易出错), 因为其结构体内部实际上还是以计算的类型(这里即 half)的元素组成的数组, 访问起来不需要再类型转换, 而且利用结构体赋值的优势来实现向量化读写.

Oneflow grid_size 和 block_size 选择

Oneflow 的 Element-wise 算子的 block_size 是一个常量 256. grid_size 根据数据量进行选择.
选择的原因具体可以见文章: 如何设置CUDA Kernel中的grid_size和block_size？ - 知乎

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;
/// @brief 获取kernel启动的grid_size大小
/// @param n element-wise处理的数据总数
/// @param[out] num_blocks 设置的线程块数
inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count; // SM个数
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;  // SM中线程最大数
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize, // 按数据个数取整划分线程块(数据量比较小)
                                                   sm_count * tpm / kBlockSize * kNumWaves)); // 按GPU线程处理量划分(数据量比较大)
  return cudaSuccess;
}

向量化内存访问
优化方面

带宽瓶颈

向量化内存访问会提高带宽, 但会降低总体并行性并增加寄存器用量. 因为相当于每个线程串行处理了多个数据.
不适用于内核已受到寄存器限制或并行度非常低的情况
更适合每个线程对数据的操作比较简单的情况(如 Element-wise 操作). 因为这样增加读写带宽的收益要大于串行降低并行性的损失.
具体说明

CUDA 每个线程一次性至多可以读写 128 比特(16 字节)的数据. 具体而言, 在 SASS 指令中, LD.E 和 ST.E 指令可以读取 32 比特(4 字节)数据, 可以替换为 LD.E.64 和 ST.E.64 指令读取 64 比特(8 字节)数据, 以及 LD.E.128 和 ST.E.128 指令读取 128 比特(16 字节)数据.

注: 需要地址对齐. 即读取 8 字节时, 数据地址需 8字节对齐.
CUDA 相关实现

int2, int4, uint2, uint4, float2 等.
可以直接使用reinterpret_cast<int2*> 或 (int2*) 将 int* 类型的指针转换为 int2* 类型的指针.

例子:

__global__ void device_copy_scalar_kernel(int* d_in, int* d_out, int N) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) { 
    d_out[i] = d_in[i]; 
  } 
}


使用 int2 序列化, 改为:

__global__ void device_copy_vector2_kernel(int* d_in, int* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < N/2; i += blockDim.x * gridDim.x) {
    reinterpret_cast<int2*>(d_out)[i] = reinterpret_cast<int2*>(d_in)[i];
  }

  // in only one thread, process final element (if there is one)
  if (idx==N/2 && N%2==1)
    d_out[N-1] = d_in[N-1];
}


整体效果是将循环减少了 N/2 次, 而每次迭代每个线程一次处理 2 个元素. 从而减少了指令的发射数, 提高了数据读取带宽.

参考资料
【BBuf 的CUDA笔记】一，解析OneFlow Element-Wise 算子实现 - 知乎
CUDA Pro Tip: Increase Performance with Vectorized Memory Access | NVIDIA Technical Blog
高效、易用、可拓展我全都要：OneFlow CUDA Elementwise 模板库的设计优化思路 - 知乎
