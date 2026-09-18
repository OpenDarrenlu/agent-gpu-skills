# [CUDA 学习笔记] half 类型的 atomicAdd 操作

**作者**: PicoPika一入infra深似海

**原文链接**: https://zhuanlan.zhihu.com/p/692009543

---

half 类型的 atomicAdd 操作
注: 本文主要是对文章 【BBuf的CUDA笔记】四，介绍三个高效实用的CUDA算法实现（OneFlow ElementWise模板，FastAtomicAdd模板，OneFlow UpsampleNearest2d模板） - 知乎 "0x2 FastAtomicAdd" 部分的学习整理, 参考代码 how-to-optim-algorithm-in-cuda/FastAtomicAdd

本文主要包括三个 half 数据类型的原子加操作 atomicAdd 的实现, 理论上可以拓展到 half 类型的其他原子操作, 如 atomicSub 等.

Kernel 0: CUDA atomicAdd() for half

算力 7.0 及以上的设备, CUDA 库中提供了用于 half 类型的 atomicAdd() 函数, 可以直接使用, 但性能较差.

__half atomicAdd(__half *address, __half val);

Kernel 1: pack half as half2
template<typename T, size_t pack_size>
struct alignas(sizeof(T) * pack_size) Pack {
  T elem[pack_size];
};

template<typename T, int32_t pack_size>
__device__ __inline__ void AtomicAdd(Pack<T, pack_size>* address,
                                     T val) {
  #pragma unroll
  for (int i = 0; i < pack_size; ++i) {
    atomicAdd(reinterpret_cast<T*>(address) + i, static_cast<T>(val));
  }
}

template<>
__device__ __inline__ void AtomicAdd<half, 2>(Pack<half, 2>* address, half val) {
  half2 h2_val;
  h2_val.x = static_cast<half>(val);
  h2_val.y = static_cast<half>(val);
  atomicAdd(reinterpret_cast<half2*>(address), h2_val);
}


kernel 1 的实现策略是通过 Pack<half, 2> 结构合并访问 2 个 half 元素, 从而使用 CUDA 库中 half2 的 atomicAdd() 函数.
但在 AtomicAdd() 函数中, 相当于对 address 地址处的两个 half 元素都进行原子加的操作, 即会影响相邻元素的值, 且至少要分配 2 个 half 的大小.

Kernel 2: Pytorch fastSpecializedAtomicAdd()

Pytorch 中针对 half 数据类型提供了 fastSpecializedAtomicAdd() 的实现.

// FastAdd is referenced from
// https://github.com/pytorch/pytorch/blob/396c3b1d88d7624938a2bb0b287f2a19f1e89bb4/aten/src/ATen/native/cuda/KernelUtils.cuh#L29
template<typename T, typename std::enable_if<std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void FastSpecializedAtomicAdd(T* base, size_t offset,
                                                         const size_t length, T value) {
#if ((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) \
     || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  atomicAdd(reinterpret_cast<half*>(base) + offset, static_cast<half>(value));
#else
  // Accounts for the chance base falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __half* target_addr = reinterpret_cast<__half*>(base + offset);
  // target_addr是否满足half2的内存对齐
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && offset < (length - 1)) {    // 内存对齐且非尾元素
    __half2 value2;
    value2.x = value;
    value2.y = __float2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && offset > 0) {    // 内存不对齐且非首元素
    __half2 value2;
    value2.x = __float2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {    // 首元素不对齐 或 尾元素对齐
    atomicAdd(reinterpret_cast<__half*>(base) + offset, static_cast<__half>(value));
  }
#endif
}

template<typename T, typename std::enable_if<!std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void FastSpecializedAtomicAdd(T* base, size_t offset,
                                                         const size_t length, T value) {
  atomicAdd(base + offset, value);
}

template<class T>
__device__ __forceinline__ void FastAdd(T* base, size_t offset, const size_t length, T value) {
  FastSpecializedAtomicAdd(base, offset, length, value);
}


FastSpecializedAtomicAdd() 函数的参数: base 表示写入的起始地址, offset 为实际写入位置距起始位置 base 的偏移, length 为 base 数组长度, value 为原子增加的值.
实现的核心也是使用 CUDA 库中 half2 的 atomicAdd() 函数, 与 kernel 1 不同的有两点:

kernel 2 使用 0 填充的 half2 中的另一个 half 元素, 这样不会影响相邻元素的值.
kernel 2 根据当前位置 offset 与 length 的大小关系以及 half2 的内存对齐条件, 选择 base+offset 与 base+offset+1(内存对齐且非尾元素) 或是 base+offset-1(内存不对齐且非首元素) 的元素合并为 half2 元素. 在极端情况下(首元素不对齐或尾元素对齐), 仍会退化为 half 类型的 atomicAdd() 函数. 因此, 该函数优化是针对一个 half 元素的数组的, 这也是函数名中带有 "Specialized" 的原因.

注: 参考代码 fast_atomic_add_half.cu 中, 笔者认为存在一些错误, 包括 main() 函数中 output_device 需要至少分配 2 个 half 元素大小, 即 sizeof(half)*2; 同时 dot() 函数调用 FastAdd() 时第三个参数应为 output_device 的大小 2 而非 N. 选择 2 的原因正是为了让 FastSpecializedAtomicAdd() 函数进入 half2 类型的 atomicAdd() 的分支, 而 1 个的话会退化为 half 类型的 atomicAdd(). (错误已修正)

性能笔记与总结

在 V100 上笔者测试 3 种实现性能如下:

kernel	性能(ms)
kernel 0: half atomicAdd()	182.45
kernel 1: pack half as half2	82.37
kernel 2: FastSpecializedAtomicAdd()	82.36

kernel 0: half atomicAdd() :

优点: 可以直接使用.
缺点: 性能很差, 不如对 2 个 half 的 haf2 类型的 atomicAdd().

kernel 1: pack half as half2:

优点: 性能较高
缺点: 必须两个 half 一起处理, 从而需要满足 half2 的内存对齐, 也会修改相邻的 half 元素

kernel 2: FastSpecializedAtomicAdd()

优点: 性能较高, 不影响相邻 half 元素的值
缺点: 适合 half 数组的情况, 极端情况下仍会退化为 atomicAdd(); 多个线程写入偏移不同时, 可能会造成 warp divergence.

额外一提, 在参考代码中, N 的值被设置为 32*1024*1024, 在测试过程中发现代码最后的计算结果并不正确, 笔者考虑应该是 half 类型精度导致的, 改为 double 便可得到正确结果, 或者 N 设置为 2048(及以下) 也能获得近似的正确结果. 不过 kernel 1 得到的结果好像比 kernel 0 和 2 更精确, 比如 N 设置为 4096 时, 其还能得到近似正确的结果. 笔者对计算精度不太了解, 此处仍存有疑问.

参考资料
【BBuf的CUDA笔记】四，介绍三个高效实用的CUDA算法实现（OneFlow ElementWise模板，FastAtomicAdd模板，OneFlow UpsampleNearest2d模板） - 知乎
