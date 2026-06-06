# 搞懂 CUDA Shared Memory 上的 bank conflicts 和向量化指令（LDS.128 / float4）的访存特点

**作者**: Alan 小分享​香港科技大学 资讯科技硕士

**原文链接**: https://zhuanlan.zhihu.com/p/690052715

---

​
目录
收起
1、shared memory 结构
简单情况
2、64 位宽的访存指令
Case 1
Case 2
3、128 位宽的访存指令
Case 1
Case 2
Case 3
Case 4
Case 5
4、其他发现
5、GEMM 中 warp tile 内的线程布局

在做算子优化的时候（比如 GEMM），我们需要充分利用 shared memory 的 broadcast 机制，以及避免 bank conflicts 的出现。同时还会用 LDS.64 或 LDS.128 指令（也可以直接用 float2、float4）等一次访问 8 或 16 个 bytes。

问题来了！！！官方文档中，只介绍了每个 thread 访问 4 byte（即 32bit）时的 broadcast 机制和 bank conflict 情形。可以看这里：（Compute Capability 5.x 到 9.x 都是一样的设计）

但是，对于使用 LDS.64 或 LDS.128 指令时的情况（即每个 thread 访问超过 4 个 bytes），却很难找到官方文档。生气 && 难过！！

于是，我大量查阅了网上的博客、问答等，以及使用 Nsight Compute 进行大量测试，总结出了使用这两个指令时的访存特点（memory transaction 数量、什么时候会出现 bank conflict 等）。

（下面的示例很多来自 Reference 中的第一篇文章，不过本文尝试把这些内容讲得更清晰，当时看各个博客的时候晕了很久）

看完本文，就可以理解，为什么 GEMM 优化中，访问 shared memory 时每个 warp 内的线程，需要组织成 4 * 8（或者 8 * 4），并且按类似下面这种顺序排列了（Z-Order）：（红色框内的数字表示 thread id）




注意：对于 shared memory，我们分析时，一般都是关注单个 warp 内的情况～～～




测试环境：

A100，Compute Capability 8.x

1、shared memory 结构

从上面的官方文档中可以知道，放在 shared memory 中的数据是以 4 bytes（即 32 bits）作为 1 个 word，依次放在 32 个 banks 中。所以，第 i 个 word，就存放在第 ( i mod 32 ) 个 bank 上。

每个 bank 在每个 cycle 的 bandwidth 为 32 bits。

所以 shared memory 在每个 cycle 的 bandwidth 为 32 * 32 bits = 32 * 4 bytes = 128 bytes。

那么关键就来了！！每次 memory transaction 最多访问 128 bytes 的数据。（这个数据是决定了 LDS.64 和 LDS.128 的访存特点）

初次接触的同学，可以看看 YouTube 上的这个视频：

简单情况

看看单次对于 shared memory 的数据请求～～～如果 warp 中每个 thread 只需要访问 4 bytes，则 broadcast 和 bank conflicts 的机制很简单：

当多个 thread 访问同一个 bank 内的同一个 word，就会触发 broadcast 机制。这个 word 会同时发给对应的 thread；
当多个 thread 访问同一个 bank 内的不同 word 时，就会产生 bank conflict。于是请求会被拆分成多次 memory transaction，串行地被发射（issue）出去执行。（比如 2-way bank conflict，就拆分成 2 次 transaction）




小结一下，单次请求中，warp 内 32 个 thread，每个访问 4 bytes，那么总的数据需求就是最多 128 bytes。只要不产生 bank conflict，一次 memory transaction 就够了。取回来 128 bytes 的数据，warp 内怎么分都可以。




2、64 位宽的访存指令

使用 LDS.64 指令（或者通过 float2、uint2 等类型）取数据时，每个 thread 请求 64 bits（即 8 bytes）数据，那么每 16 个 thread 就需要请求 128 bytes 的数据。

所以 CUDA 会默认将一个 warp 拆分为两个 half warp，每个 half warp 产生一次 memory transaction。即一共两次 transaction。

只有以下两个条件之一满足时，这两个 half warp 的访问才会合并成一次 memory transaction：

对于 Warp 内所有活跃的第 i 号线程，第 i xor 1 号线程不活跃或者访存地址和其一致；(i.e. T0==T1, T2==T3, T4==T5, T6==T7, T8 == T9, ......, T30 == T31, etc.)
对于 Warp 内所有活跃的第 i 号线程，第 i xor 2 号线程不活跃或者访存地址和其一致；(i.e. T0==T2, T1==T3, T4==T6, T5==T7 etc.)

（活跃是指有访存需求）




为什么呢？？

简单理解一下，当上面两种情况发生时，硬件就可以判断（具体是硬件还是编译器的功劳，我也不确定，先归给硬件吧），单个 half warp 内，最多需要 64 bytes 的数据，那么两个 half warp 就可以合并起来，通过一次 memory transaction，拿回 128 bytes 的数据。然后线程之间怎么分都可以（broadcast 机制）。

当然，这里的前提是没有产生 bank conflict。即没有从单个 bank 请求超过 1 个 word。




看几个栗子～

Case 1

每个线程依次访问连续的 uint2。即第 tid 个线程，访问第 tid 个 uint2。

这时，并没有触发合并的条件，每个 half warp 分别执行一次 memory transaction，一共两次。也没有产生 bank conflict。

看下第一个 half warp 访问的数据位置：（第一行中的 32 个 word，黄色部分）

上半部分的 Bank 表示数据排列，每个格子表示 1 个 word；下半部分表示线程排列；

第二个 half warp 则是访问第二行的 32 个 word。




注意！！

其实 bank conflict 是针对单次 memory transaction 而言的。如果单次 memory transaction 需要访问的 128 bytes 中有多个 word 属于同一个 bank，就产生了 bank conflict，从而需要拆分为多次 transaction。

比如这里，第一次访问了 0 - 31 个 word，第二次访问了 32 - 63 个 word，每次 transaction 内部并没有 bank conflict。




代码：

__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid];
}





看下 nsight compute 的结果：




l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum 表示总的 conflict 数量；

l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum 表示总的 shared memory load transaction 的数量；




Case 2

这个模式就是符合了合并条件中的第一条。

所以两个 half warp 的访问合并，一共只有 1 次 memory transaction，没有 bank conflict。

代码：

__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid / 2];
}





3、128 位宽的访存指令

使用 LDS.128 指令（或者通过 float4、uint4 等类型）取数据时，每个 thread 请求 128 bits（即 16 bytes）数据，那么每 8 个 thread 就需要请求 128 bytes 的数据。

所以，CUDA 会默认把每个 half warp 进一步切分成两个 quarter warp，每个包含 8 个 thread。每个 quarter warp 产生一次 memory transaction。所以每个 warp 每次请求，默认会有 4 次 memory transaction。（没有 bank conflict 的情况下）。

类似 64 位宽的情况，当满足特定条件时，一个 half warp 内的两个 quarter warp 的访存请求会合并为 1 次 memory transaction。但是两个 half warp 不会再进一步合并了。（划重点！！！）

具体条件和 64 位宽一样：

对于 Warp 内所有活跃的第 i 号线程，第 i xor 1 号线程不活跃或者访存地址和其一致；(i.e. T0==T1, T2==T3, T4==T5, T6==T7, T8 == T9, ......, T30 == T31, etc.)
对于 Warp 内所有活跃的第 i 号线程，第 i xor 2 号线程不活跃或者访存地址和其一致；(i.e. T0==T2, T1==T3, T4==T6, T5==T7 etc.)

（活跃是指有访存需求）




看几个栗子～～～

Case 1

只激活第 15 和 16 号线程，访问第 4 个 uint4（每个元素 16 bytes，对应下面 bank 中 4 个连续的格子）

上半部分的 Bank 表示数据排列，每个格子表示 1 个 word；下半部分表示线程排列；

这时只有两个 quarter-warp 活跃，分别需要一次 memory transaction，一共 2 次。没有 bank conflict。

注：这两个 quarter-warp 属于两个不同的 half warp，所以不会合并访问。

__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 15 || tid == 16) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}





Case 2

只激活第 0 和第 15 号线程，访问第 4 个 uint4：

这是满足合并条件第一条，所以前两个 quarter warp 的访存请求合并成 1 次 memory transaction。没有 bank conflict。

__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 0 || tid == 15) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}





Case 3

满足合并条件第一条，前两个 quarter warp 和后两个 quarter warp 分别合并，分别需要 1 个 memory transaction（即每个 half warp 需要 1 个 transaction）。一共 2 个 transaction，没有 bank conflict。

__global__ void smem_3(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
      smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}





Case 4

这个排布有点意思，第一个 half warp 满足合并条件 1，第二个half warp 满足合并条件 2。但是需要整个 warp 都满足条件 1，或者条件 2，或者 1、2 同时满足，这样才可以合并。

所以这里仍然是每个 quarter warp 需要 1 次 memory transaction，一共 4 次。没有 bank conflict。

__global__ void smem_4(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr;
  if (tid < 16) {
    addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
  } else {
    addr = (tid / 8) * 2 + ((tid % 8) % 2);
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
  // printf("tid: %d, addr: %d\n", tid, addr);
}





Case 5

(注：下面第一篇博客中，这个栗子的代码和解释都有点问题，我这里修正了一下）

thread 0 - 3 访问第 0 个 uint4， thread 4 - 7 访问第 8 个 uint4（到了第二行）；

thread 8 - 11 访问第 1 个 uint4， thread 12 - 15 访问第 9 个 uint4（到了第二行）；

依次类推；（可以在 kernel 内通过 printf 打印 tid 和 addr）




这里符合合并条件 1，所以前两个和后两个 quarter warp 分别合并。但是每个 half warp 内，产生了 2-way bank conflict，所以需要拆成 2 次 transaction。

即一共 2 个 bank conflict， 4 次 transaction。

__global__ void smem_5(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8;
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
  printf("tid: %d, addr: %d\n", tid, addr);
}





完整代码：




4、其他发现

对于单个简单 kernel，其实 2 次 memory transaction 和 4 次，所需的耗时并不会差很远（测试的时候差了 5% 左右）。

因为每个 transaction 可以在每个 cycle 发射一次，类似于流水线式地派发出去执行。

而单次 transaction 可能大概需要 30 - 40 个 cycle，所以多 2 次 transaction，可能也就多 2 个 cycle，整体影响不大。（注意：这只是针对非常简单的 kernel！）




但是，当 GPU 满流水运行时（计算时间可以比较好地掩盖访存延迟），2 次和 4 次 transaction 造成的整体 kernel 性能差异就比较大了。（比如 GEMM 中，对 warp 内的线程进行适当的排列，可以有 13% 左右的提升，参考这篇论文第 3.1.4 节：https://arxiv.org/pdf/2305.01024.pdf ）




说明 1：30 - 40 个 cycle 怎么算出来的？

以 A100 为例，网上的测评给出的访问 shared memory 的 latency 大概是 22.4 ns，GPU 时钟频率是 1.5GHz，算下来单次访问就是 22.4 / (1 / 1.5GHz) = 33.6 cycle。




5、GEMM 中 warp tile 内的线程布局

现在就可以解释下开头的问题 ~~ 为什么 GEMM 优化中，从 shared memory 读数据到 register 并计算时，每个 warp tile 内的线程，需要组织长 4 * 8 （或者 8 * 4）的格局，并按 Z-Order 方式排列呢？

例如：

比如看看对 B_tile（B 矩阵在 shared memory 上的一个 block tile） 的访问，每一轮迭代（即对 B_tile 一行一行地迭代）中，线程 0、2、4、6 都是访问相同的 4 个 float，线程 1、3、5、7 也是访问相同的 4 个 float，其他线程同理。

这时候就符合前面说的合并条件 2，所以线程 0 - 7，以及线程 8 - 15 的访存请求，合并为一次 memory transaction。线程 16 -31 同理。所以单个 warp 内一共只需要 2 次 memory transaction，去读取 B_tile 中的 8 * 4 个 float。

如果没有触发合并条件，则需要更多的 memory transaction。




另外，Z-Order 的顺序不是只有一种，能触发合并条件就行，比如下面这样子也可以：




如果哪里写的不对，请大家指正～～




Reference

[1] https://code.hitori.moe/post/cuda-shared-memory-access-mechanism-with-vectorized-instructions/

[2] Unexpected shared memory bank conflict.

[3] How to understand the bank conflict of shared_mem

[4] 4. Nsight Compute CLI

[5] nvprof - Metrics for Capability 7.x

[6] https://www.youtube.com/watch?v=CZgM3DEBplE
