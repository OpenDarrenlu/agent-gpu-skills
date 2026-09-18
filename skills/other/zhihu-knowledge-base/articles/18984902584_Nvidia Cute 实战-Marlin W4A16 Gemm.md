# Nvidia Cute 实战-Marlin W4A16 Gemm

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/18984902584

---

前言

Cute：

Nvidia Cutlass3.0 中推出了 新的编程库Cute，通过对layout, tensor, copy, mma 等在cuda编程中常见的对象和操作进行高级抽象，让开发者可以高效的处理复杂的坐标计算，以及CTA->warp->thread 不同执行层级的计算数据拆分。相比cutlass， cute可以更灵活的设计Epilog 以及设置一些在cutlass内难以处理的layout操作并忽略layout底层坐标计算的细节。

cutlass repo reed 大佬主页

cute基本的概念，有需要可以看cute 的官方文档及代码示例和reed 大佬的cute 解析文章，学习Cute前置知识。

Marlin W4A16 Gemm

MARLIN(Mixed-precision AutoRegressive LINear kernels) 是面向nvidia ampere/ ada 架构在LLM Inference 中 Linear Layer 优化的 W4A16 Gemm极致优化实现。在LLM decoding phase 通常是memory bound，低bit量化可以减少访存量从而提高LLM的吞吐。Marlin 通过Gpu计算流的优化、SM间的任务拆分、以及对于量化友好的权重内存布局的重排等优化手段，大幅提高了W4A16 LLM inference 性能。

Marlin 官方repo， Marlin 的官方实现采用最朴素的Cuda C 实现，其中涉及了大量、复杂且内存排布相关的坐标计算，且为了保证GPU 访存的高效，统一采用了int128_t 的数据格式来进行坐标计算，而Marlin W4A16 gemm 涉及了int4_t、fp16、 fp32 等不同的数据类型，不同数据类型位宽和int128_t之间的映射长度不同，这导致了Marlin 的官方实现可读性很糟糕，不利于学习和理解。

实现

本文代码和cute latex图发布在 CalebDu/Awesome-Cute， 需要注意的cute 实现中小部分的内存/线程排布和官方的实现并不是完全一致的。

内存排布预处理

Marlin 的实现中，为了最大化读取weight 的带宽，对weight做了内存排布的预处理，这与marlin 实现中的warp tile(16, 64) 紧密相关。

# gemm/marlin_gemm/marlin.py
# class Linear def pack, weight 预处理代码
# 省略量化部分代码
w = w.reshape((self.k // tile, tile, self.n // tile, tile))
w = w.permute((0, 2, 1, 3))
w = w.reshape((self.k // tile, self.n * tile))
res = w
res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)

#_perm = [0,  128,    8,  136,   16,  144,   24,  152,  256,  384,  264,
# 392,  272,  400,  280,  408, 512,  640,  520,  648,  528,  656,
# 536,  664,  768,  896,  776, 904,  784,  912,  792,  920....] total 1024 element = 16 * 64

weight 部分的预处理，现将量化之后的[k, n] 的weight，reshape+transpose 为 [k/16, n/16, 16(k), 16(n)] 保证每个warp 处理的[16,64] tile 的内存连续，在对于tile内的1024 个元素根据 mma指令每个thread fragment B的内存排布对weight 进行处理(对应_perm 的坐标）。

（16，（16，4））tile B layout
（16，64）mma tile B layout

weight 预处理保证每个thread 8个m16n8k16 所需的(8*4 fp16) fragB 对应的权重(32 int4) 在内存中 连续，一次int128 访存既可以全部读入。对于一个绿框中2次mma所需的int4 fragB 按照0,128,8,136,16,144,24,152 的顺序用于实现int4->fp16快速转换，详细可以参考DefTruth：[LLM推理优化] WINT8/4-(03): LOP3指令详解及INT4转FP16/BF16分析。

// gemm/marlin_gemm/marlin_cute_trait.h
DEVICE static auto dequant(int q) {
  auto half4_frag = make_tensor<half2>(make_shape(_2{})); // fragment B
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  half4_frag[0] = __hsub2(*reinterpret_cast<half2 *>(&lo),
                          *reinterpret_cast<const half2 *>(&SUB));
  half4_frag[1] = __hfma2(*reinterpret_cast<half2 *>(&hi),
                          *reinterpret_cast<const half2 *>(&MUL),
                          *reinterpret_cast<const half2 *>(&ADD));
  return half4_frag;
}


简单的来说，8个int4在int32中 从高地址到低地址的顺序是 q = e_{152}e_{24}e_{144}e_{16}e_{136}e_{8}e_{128}e_{0} ，一次dequant计算可以把 e_{0}e_{16}e_{128}e_{144} 转化成4个fp16 对应 第一个mma中 thread0 的v0 v1 v2 v4，再对q>>8进行dequant计算，可以实现 e_ {8}e_{24}e_{136}e_{152} fp16 的转换既第二个mma中 thread0 的v0 v1 v2 v4。

因为每个thread 负责读取32个int4(1 int128_t)B, 在B tensor 的layout设置也要符合这个要求。对应B tensor 的shape为make_shape(k / _16{}, n / _64{}, _32{}, _32{}) ，保证每个warp 的32thread 读取连续的1024 个int4 weight。

// [ctak/16, ctan/64, 16(k), 64(n)]->
// [ctak/16, ctan/64, 32(warp), 32(32int4=int128)]
using SmemBLayoutAtom = decltype(make_layout(
      make_shape(Int<kCTAK / kWarpK>{}, Int<kCTAN / kWarpN>{}, _32{}, _32{}),
      LayoutRight{}));

// G2S B copy
using G2SBCopyOp = SM80_CP_ASYNC_CACHEGLOBAL_EVICT<cute::uint128_t>;
using G2SBCopyTraits = Copy_Traits<G2SBCopyOp>;
using G2SBCopyAtom = Copy_Atom<G2SBCopyTraits, Btype>;
static constexpr int G2SBvecLen =
    sizeof_bits<uint128_t>::value / sizeof_bits<Btype>::value; // 32 int4

// [8/warpn, warpn, 32(warp), 1]
using G2SBThrLayout = decltype(make_layout(
    make_shape(Int<kMmaThrLayoutK>{}, Int<kMmaThrLayoutN>{}, _32{}, _1{}),
    LayoutRight{}));
//[1, 1, 1, 32(32int4)]
using G2SBThrValLayout =
    decltype(make_layout(make_shape(_1{}, _1{}, _1{}, Int<G2SBvecLen>{})));

// B tensor layout
auto B_tensor = make_tensor(make_gmem_ptr<Btype>(args.B),
                            make_shape(k / _16{}, n / _64{}, _32{}, _32{}),
                            LayoutRight{});


Marlin 对L2 Cache 的利用也有精细的设计，保证了A矩阵保留在L2 Cache内，B、scale矩阵不保留在L2 Cache，来最大化的利用L2 Cache 的带宽，实现L2->L1 和Dram->L2的并行。

// cp_async evict B/scale in L2 Cache
asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
);


由于weight进行了预处理，相应的反量化scale也需要对应的处理， 针对per channel 和per group(128) 两种量化方式，scale 需要不同的处理。

// gemm/marlin_gemm/marlin.py
if self.groupsize != self.k: //per group
  w = w.reshape((self.groupsize, -1, self.n))
  w = w.permute(1, 0, 2)
  w = w.reshape((self.k, self.n)).contiguous()
  s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
else: // per channel
  s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
#_scale_perm = [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50,
#58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 
#53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63] # 64 index for per group
#_scale_perm_single = [0, 1, 8, 9, 16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27, 4, 5, 12,
#13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31] # 32 index for per channel
（16，64）mma tile C frag

per group 反量化需要在gemm main loop 内进行计算，所以scale需要按照frag b 的排布，每个thread 对应8列，如thread0 需要（第0, 8, 16, 24, 32, 40, 48, 56 列） 共8个fp16 scale，可以一次int128 访存完整读取。 而per channel 可以在main loop 结束的epilog 进行计算，需要按照frag c排布（第0, 1, 8, 9, 16, 17, 24, 25列）对应上图，每个thread 对应16列，对应16个fp16 scale，需要2次int128 访存，为了保证不同线程间的memory coalescing，两次int128 访存间隔一行(32fp16)。

需要注意的是，属于同一列的thread，对应的scale 值是相同的，由于fragB和fragC的thread排布是不同的，所以per channel 和per group 每个thread对应读取的scale 的idx也是不同的。如per group 量化thread0 1 2 3 的对应scale 是相同的， per channel 量化 thread0 4 8 12 16 20 24 28 对应的scale是相同。

// gemm/marlin_gemm/marlin_cute_trait.h
int s2r_sScale_tile_idx;
if constexpr (GroupSize != -1) {
  // (warp_idx % (ctan/64)) *  8(8fp16) + (lane_idx) / 4
  s2r_sScale_tile_idx =
     _8{} * ((tidx >> 5) % (kMmaThrLayoutN)) + ((tidx & 31) >> 2);
} else {
  // (warp_idx % (ctan/64)) * 8(8fp16) + (lane_idx) % 4
  s2r_sScale_tile_idx =
     _8{} * ((tidx >> 5) % (kMmaThrLayoutN)) + ((tidx & 31) & 3);
}

mma计算

nvidia gpu ampere/ada 架构 每个sm拥有4个warp scheduler 和64k 个fp32 register， 且每个thread 最多可以使用256 个register，由此若每个thread 使用最多的register数目，每个sm可以驻留256个thread（8个warp），每个warp scheduler可以调度2个warp。Marlin 采用了两种cta tile 配置 (m=16/32/48/64, n=128, k=128)/(m=16/32/48/64, n=256, k=64)。对应B warp layout 为（4warp[k-dim]，2warp[n-dim])/ (2warp[k-dim], 4warp[n-dim])，每个warp在k iteration 可以展开2次k inner loop进行pipeline。

因为每个warp 负责计算的是n维度连续的8个m16n8k16 mma， 所以在设计tiled mma的时候要符合对应的排布。 using kMmaPermuteNLayout = Layout<Shape<_2, _4, Int<kMmaThrLayoutN>, _8>, Stride<_1, _2, _64, _8>>;让每个warp负责处理的mma n维度连续排布 。

// gemm/marlin_gemm/marlin_cute_trait.h
static constexpr int kMmaThrLayoutM = 1;
static constexpr int kMmaThrLayoutN = kCTAN / kWarpN;
static constexpr int kMmaThrLayoutK = kWarp / kMmaThrLayoutN;
using MmaThrLayout = decltype(make_layout(make_shape(
    Int<kMmaThrLayoutM>{}, Int<kMmaThrLayoutN>{}, Int<kMmaThrLayoutK>{})));
static constexpr int kMmaPermuteM = kMmaThrLayoutM * get<0>(mma_atom_shape{});
static constexpr int kMmaPermuteN = kCTAN;
static constexpr int kMmaPermuteK = kMmaThrLayoutK * get<2>(mma_atom_shape{});
static_assert(kMmaThrLayoutM * kMmaThrLayoutN * kMmaThrLayoutK == kWarp,
            "warp num mismatch");
using kMmaPermuteNLayout =
    Layout<Shape<_2, _4, Int<kMmaThrLayoutN>, _8>, Stride<_1, _2, _64, _8>>;
using MmaPermutations = decltype(make_tile(
    Int<kMmaPermuteM>{}, kMmaPermuteNLayout{}, Int<kMmaPermuteK>{}));
using MMA =
    decltype(make_tiled_mma(mma_atom{}, MmaThrLayout{}, MmaPermutations{}));


每个warp在n维度处理8个mma，而一个int32 包含2个mma所需的frag b， 所以只需要4次n循环即可计算一轮内层k 循环

// gemm/marlin_gemm/marlin_cute_trait.h
auto launch_gemm = [&](int k_inner_idx) {
#pragma unroll
    for (int n_idx = 0; n_idx < size<2>(tCrC_mma); n_idx += 2) {
    // warp tile [16, 64] = 8 * [16, 8] mma, for each thread holds
    // 8frag_b(4fp16) so int32 = 8 int4 = 2 quant frag_b(4int4), load 1
    // int32 compute 2 times mma op
    int quant_w = s2r_tBrB_copy_view_i32(n_idx >> 1, _0{}, k_inner_idx & 1);
    int quant_w_shift = quant_w >> 8;
    auto dequant_w = dequant(quant_w);
    auto dequant_w_shift = dequant(quant_w_shift);
    auto tBrB_mma_col0 = recast<Atype>(dequant_w);
    auto tBrB_mma_col1 = recast<Atype>(dequant_w_shift);
    if constexpr (GroupSize != -1) {
        half2 scale0_pack = __half2half2(tSrS_native(n_idx, k_inner_idx & 1));
        half2 scale1_pack =
            __half2half2(tSrS_native(n_idx + 1, k_inner_idx & 1));
#pragma unroll
        for (int i = 0; i < size<0>(dequant_w); i++) {
            dequant_w(i) = __hmul2(dequant_w(i), scale0_pack);
            dequant_w_shift(i) = __hmul2(dequant_w_shift(i), scale1_pack);
        }
    }
#pragma unroll
    for (int m_idx = 0; m_idx < size<1>(tCrC_mma); m_idx++) {
        gemm(mma, tArA_mma(_, m_idx, k_inner_idx & 1), tBrB_mma_col0,
            tCrC_mma(_, m_idx, n_idx));
        gemm(mma, tArA_mma(_, m_idx, k_inner_idx & 1), tBrB_mma_col1,
            tCrC_mma(_, m_idx, n_idx + 1));
    }
    }
};

对于gemm main loop，marlin 以stage 为单位，把k iteration 进行静态的循环展开，以实现更好的g2s/s2r/mma之间的并行和编译期的常量优化。

// gemm/marlin_gemm/marlin_cute_trait.h
#pragma unroll 1
while (tile_work.k_iters_remaining) {
#pragma unroll
  for (int pipe_idx = 0; pipe_idx < kStage;) {
#pragma unroll
    for (int k_inner_idx = 0; k_inner_idx < k_inner_cnt; k_inner_idx++) {
      launch_s2r(pipe_idx % kStage, k_inner_idx + 1);
      if (k_inner_idx == k_inner_cnt - 2) {
        // load next stage
        launch_g2s((pipe_idx + kStage - 1) % kStage, pipe_idx,
                    tile_work.k_iters_remaining >= kStage);
        pipe_idx++;
        wait_stage();
      }
      launch_gemm(k_inner_idx);
      gemm_cnt++;
    }
    tile_work.k_iters_remaining--;
    if (tile_work.k_iters_remaining == 0) {
      break;
    }
  }
  g2s_k_main_loop_offset += kStage;
}

intra-CTA reduce

由于在k 维度 采用了多个warp并行，所以在main loop 之后的epilog 需要把warp之间k维度的partial sum进行reduce 以得到完整的cta result。warp间的reduce 对应代码中launch_epilog_cta_reduce 函数。

核心逻辑采用logarithmic shared memory reduction。以CTA_N=128 为例: warp4, warp 6把每个元素拥有的partial sum frag c存储到smem， warp2 从smem读取warp4 warp6 的结果进行累加后存储到smem，最后warp0 把warp2 的结果累加后得到完整的结果

logarithmic shared memory reduction
inter-CTA reduce

在LLM decoding phase中A 的m维度往往是比较小，这导致了传统的data parallel gemm 不能很好的利用gpu的SM资源。为了充分利用SM资源，Marlin仿照了streamk的方式（streamk gemm 可以参考之前的文章CalebDu：Nvidia Cute 实战- StreamK Gemm，细节本文省略），对m n k 维度的k iteration 总数在block 中进行均分(per group 需要对齐到group size），使得SM 的资源都分配到计算的任务。这样的并行策略会导致一个tile 的完整k main loop 被多个cta负责计算，为了得到完整的k main loop 结果需要跨cta 之间的结果累加，所以marlin设置了dram 的lock数组 记录每个tile 内负责block 累加的次数，保证cta按顺序累加。 除此之外，在cute 的实现中仿照streamk 中的skew 偏移，如下图中的黑色序号，每个cta 负责的tile 按反向的顺序进行计算，通过逆向的计算顺序，缓解cta 顺序计算导致的通信等待开销。

strip partition

由于输出的C 为fp16， 在mma 计算的frag C为fp32 shape ((2[n_dim],2[m_dim]), m, 8)，在R2G store的之前需要 先把fp32转换成fp16，为了调用int128 实现高效的访存，需要把转换为fp16 的frag 转换layout，转换成(8, 2[m_dim]*m, 2[n_dim]), 如下图为 CTA tile (m,128, 128) 的G2S/R2G对应的线程排布。

tile(16, 64) G2S_epilog_global_reduce_copy
using G2SGlobalReduceCCopy = decltype(make_tiled_copy(
    G2SGlobalReduceCCopyAtom{},
    make_layout(make_shape(_8{}, make_shape(_4{}, Int<kMmaThrLayoutN>{})),
                make_stride(_4{}, make_stride(_1{}, _32{}))),
    make_layout(make_shape(Int<1>{}, Int<R2SCvecLen>{}))));


为了保证S2R读取的bank conflcit free，smem layout 设置为每个warp 中的local tile在smem中连续，避免在跨行stride 导致的bank conflict。（ncu 测试发现这里会触发不规则的bank conflict，没想清楚为什么）

// ((16, ctam/16), (32, ctan/32))  make (16, 32) sub tile continuous
// to avoid bank conflict
using SmemEpilogGlobalReduceCLayout =
    decltype(make_layout(make_shape(make_shape(_16{}, Int<kCTAM / 16>{}),
                                    make_shape(_32{}, Int<kCTAN / 32>{})),
                        make_stride(make_stride(_32{}, Int<16 * kCTAN>{}),
                                    make_stride(_1{}, _512{}))));

tile(m, 128) epilog_global_reduce_copy_smemC
python 接口

在官方的python 接口上增加了 cute_version flag，用于指定使用cute version 的实现。

def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16, cute_version=True):
    """Marlin FP16xINT4 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    @cute_version: use cute version
    """
    marlin_lib.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par, cute_version)
实验

实验环境 rtx4090（specdram bw 1,008GBps，tensor core fp16 330tflops），cuda_12.1.r12.1。

shape(m_n_k_group)	marlin_cute	marlin_official
7B_1_12288_4096_-1	15.057 us	14.629 us
7B_16_12288_4096_-1	15.941 us	15.291 us
7B_1_12288_4096_128	15.744 us	14.921 us
7B_16_12288_4096_128	16.550 us	15.637 us
7B_1_4096_4096_-1	8.359 us	8.189 us
7B_16_4096_4096_-1	9.035 us	8.936 us
7B_1_4096_4096_128	8.714 us	8.431 us
7B_16_4096_4096_128	9.393 us	9.113 us
7B_1_21504_4096_-1	23.983 us	23.464 us
7B_16_21504_4096_-1	24.858 us	24.171 us
7B_1_21504_4096_128	25.268 us	24.068 us
7B_16_21504_4096_128	26.104 us	24.861 us
7B_1_4096_10752_-1	14.108 us	13.845 us
7B_16_4096_10752_-1	14.916 us	14.617 us
7B_1_4096_10752_128	14.872 us	14.172 us
7B_16_4096_10752_128	15.714 us	15.224 us

根据上述实验数据，cute version 的实现性能略微弱于official version。
