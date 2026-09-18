# Nvidia CUTE 实战1：ABQ-LLM GEMM Kernel

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/3904815819

---

前言

Cute：

Nvidia Cutlass3.0 中推出了 新的编程库Cute，通过对layout, tensor, copy, mma 等在cuda编程中常见的对象和操作进行高级抽象，让开发者可以高效的处理复杂的坐标计算，以及CTA->warp->thread 不同执行层级的计算数据拆分。相比cutlass， cute可以更灵活的设计Epilog 以及设置一些在cutlass内难以处理的layout操作（如swizzling）并忽略layout底层坐标计算的细节。

cutlass repo reed 大佬主页

cute基本的概念，有需要可以看cute 的官方文档及代码示例和reed 大佬的cute 解析文章，学习Cute前置知识。

ABQ(Arbitrary-Bit Quantized)-LLM：是字节跳动ByteNN 团队提出的任意bit精度量化的工作，作者的详细解析：liusongwei：ABQ-LLM : 字节开源，LLM领域首次实现量化推理自由，效果和性能双SOTA. 为了实现任意bit的精度的计算ABQ-LLM 实现了的对应的ABQ-LLM customizedGEMM， 通过nvidia isa 中提供的 1bit mma 指令，巧妙的将WpAq 的gemm 计算转化成对应的 1bit gemm，并在Epilog 阶段将对应的pbit、qbit 的结果进行reduce，得到完整的WpAq gemm的结果。计算逻辑如下图

ABQ-LLM GEMM
// pseudo code
// WpAq gemm
[m, n] 32bit = [m, k] p bit * [n, k]T q bit
// abq-llm gemm mainloop for WpAq gemm
A：[p, m, k] 1bit = pack([m, k] p bit)
B: [q, n, k] 1bit = pack([n, k] q bit)
gemm_output: [p * m, q * n] 32bit= matmul([p * m, k] 1bit, [q * n, k]T 1bit)
// abq-llm gemm epilog for WpAq gemm bit reduction 【omit sign】
for m_i = 0 to m do
  for n_i = 0 to n do
    p_multiplier <- 1
    for p_i = 0 to p do
      q_multiplier <- 1
      for q_i = 0 to q do
        output[m_i, n_i] += gemm_output[p_i*m + m_i, q_i*n+n_i]*p_multiplier*q_multiplier
	q_multiplier <- q_multiplier * 2
      end for
      p_multiplier <- p_multiplier * 2
    end for
  end for
end for


ABQ-LLM 官方repo提供了基于nv官方提供的WMMA api 和BMMA customized ptx wrapper api 两套实现的customized GEMM. 相比 WMMA的粗粒度封装,BMMA 实现直接调用 ptx mma 汇编可以进行更细粒度的线程操作。

除了官方提供的WMMA, BMMA的两种实现，本文将介绍通过使用Cute 框架来重新实现ABQ-LLM customized GEMM，对比性能及总结cute的优缺点。Cute 版本的代码 发布在 GitHub - CalebDu/ABQ-LLM at caleb_dev

实现

cute版本的核心kernel代码为ABQ-LLM/engine/mma_any/aq_cute_kernel.h、ABQ-LLM/engine/mma_any/aq_cute_atom.h,其余host处理代码与bmma/wmma版本保持一致。

官方的实现采用int32 来表示32 个int1b，cute/cutlass 中提供了subbyte 数据类型，所以可以直接使用uint1_t 来表示。

auto A_tensor =
        make_tensor(make_gmem_ptr<type>(X), make_shape(make_shape(M, Int<X_BITS>{}), main_loop_k),
                    make_stride(make_stride(main_loop_k, M * main_loop_k), _1{}));
auto B_tensor =
        make_tensor(make_gmem_ptr<type>(W), make_shape(make_shape(N, Int<W_BITS>{}), main_loop_k),
                    make_stride(make_stride(main_loop_k, N * main_loop_k), _1{}));


由此利用指针uint1_b*定义global tensor A B, 形状定义为((m,x_bits), k) , ((n, w_bits), k) 来表示(m*x_bits,k)和(n*w_bits,k)的真实形状。需要注意的make_gmem_ptr<type>(X) 中的<type> 不可省略，否则无法正确的处理subbyte 数据类型，会调用错误的make_gmem_ptr函数重载。

对于每个cta block mainloop 每次循环需要进行mnk为(BLOCK_M*x_bits, BLOCK_N*w_bits, BLOCK_K)的mma计算，因此对于A_tensor, B_tensor 需要根据blockIdx 进行tile 分块。

auto gA =
        local_tile(A_tensor,
                   make_tile(make_tile(Int<BLOCK_M>{}, Int<X_BITS>{}), Int<MainLoop_BLOCK_K>{}),
                   make_coord(bidx_m, _)); //[(block_m, P), block_k, k_loop]
auto gB =
        local_tile(B_tensor,
                   make_tile(make_tile(Int<BLOCK_N>{}, Int<W_BITS>{}), Int<MainLoop_BLOCK_K>{}),
                   make_coord(bidx_n, _)); //[(block_n, Q), block_k, k_loop]

Smem

对于G2S copy，cuda支持最高每个thread 128bit(16byte)的向量化访存，而shared memory 以4byte为单位分为32个bank，即一个warp中的32个thread 最多能够以8个thread为一组同时通过16byte(4bank)访问32个bank, 32个thread分为4个phase来执行 128bit向量化访存。 但是对于每个block_k 通常是mma_k的整数倍，block_k的低位stride导致了每次从[BLOCK_M, BLOCK_K] 的smem中 进行S2R copy [mma_m, mma_k] 的数据会触发load bank conflict。

以uint1_t block_k=512为例，mma 指令的最小tile 是m8n8k128。下图中的一个数字代表一次128bit (4bank)向量化访存的数据，即128 个uint1。 mma 第一次循环时 32个thread需要s2r copy 0 4 8 .... 24 28 共[8, 128]数据，导致了[0 8 16 24] [4 12 20 28] 处于相同的bank， 触发了4-way bank conflict。

cute 提供了swizzle 的抽象封装，通过 Swizzle<2, 7, 3>的映射处理 可以把每次需要copy的数据均匀的分配到32个bank内。<2, 7, 3>代表了 2^2=4 行, 2^7=128 个元素， 2^3=8 列 对应了上图的view 操作，通过(icol = irow ^ icol)计算view后的新的列坐标，实现每次mma计算访问的smem数据bank conflict free。

注意，smem [m, k* 2^M ]形状绑定swizzle 需要和<B, M, S>匹配，满足 m*k = a*2^B*2^S ，否则无法正确的避免bank conflict 甚至会产生static_assert fail。

基于以上原则，对于不同的block_k 如 128 ,256, 512,1024 分别采用swizzle<0, 7, 3>,swizzle<1, 7, 3>,swizzle<2, 7, 3>,swizzle<3, 7, 3>, 如果是block_k/128 不满足2的幂次，天然的可以做到bank conflict free，则直接使用swizzle<0, 7, 3>。

template <int M, int N, int K> struct SwizzleAtom {
    constexpr static bool enable_swizzle = cutlass::is_pow2<K / 128>::value; // check K/128 is 2^n
    constexpr static int AB_Swizzle_B =
        std::conditional_t<enable_swizzle, cutlass::log2_up<K / 128>, //swizzle<n, 7, 3>
                           cutlass::log2_up<1>>::value; // swizzle<0, 7, 3> (no swizzle)
    constexpr static int AB_Swizzle_M = 7; // 2^7 = 128 uint1b(16B)
    constexpr static int AB_Swizzle_S = 3; // 2^3 = 8 bank(16B per bank)
    using AB_Swizzle = Swizzle<AB_Swizzle_B, AB_Swizzle_M, AB_Swizzle_S>;
};
// AB Swizzle
using SwizzleAtom = SwizzleAtom<BLOCK_M, BLOCK_N, BLOCK_K>;
using AB_Swizzle = typename SwizzleAtom::AB_Swizzle;
// SmemLayoutAtom [8, block_k]
using SmemABLayoutAtom = decltype(composition(
    AB_Swizzle{}, make_layout(make_shape(Int<8>{}, Int<MainLoop_BLOCK_K>{}),
                              make_stride(Int<MainLoop_BLOCK_K>{}, Int<1>{}))));

// SmemALayout [x_bit * block_m, block_k, stage]
using SmemALayout = decltype(tile_to_shape(
    SmemABLayoutAtom{},
    make_shape(Int<MainLoop_BLOCK_M>{}, Int<MainLoop_BLOCK_K>{}, Int<kThreadBlockStage>{})));
// SmemBLayout [w_bit * block_n, block_k, stage]
using SmemBLayout = decltype(tile_to_shape(
    SmemABLayoutAtom{},
    make_shape(Int<MainLoop_BLOCK_N>{}, Int<MainLoop_BLOCK_K>{}, Int<kThreadBlockStage>{})));


因为mma指令时以m=8 为单位，所以只需要保证从[8, block_k] 访问s2r bank conflict free，即可以保证[block_m, block_k] 也可以s2r bank conflict free。

ABsmem 设置cp.async多级流水来有效的隐藏后续tile的g2s访问延迟。

为了处理block_m*xbit不满足mma_m 的情况，会对 block_m*xbit 向上取整到mma_m * warp_m 的整数倍MainLoop_BLOCK_M，对于CSmem mainloop 计算的结果为[MainLoop_BLOCK_M, MainLoop_BLOCK_N], 在后续的epilog计算中只需要其中真实的[xbitBLOCK_M, wbitBLOCK_N]，每次循环只需要访问[block_m, block_n] 的数据，同样面临load bank conflict的问题。这里采用传统的padding的方法，设置skew=8 int32和限制block_n为32的倍数，即保证了访问[block_m, block_n] 可以bank conflict free。

static constexpr int SmemCLayoutSkew = 8;
using SmemCLayout =
    decltype(make_layout(make_shape(Int<MainLoop_BLOCK_M>{}, Int<MainLoop_BLOCK_N>{}),
                          make_stride(Int<MainLoop_BLOCK_N + SmemCLayoutSkew>{}, _1{})));
using SmemEpilogLayout =
    decltype(make_layout(make_shape(Int<BLOCK_M * X_BITS>{}, Int<BLOCK_N * W_BITS>{}),
                          make_stride(Int<MainLoop_BLOCK_N + SmemCLayoutSkew>{}, _1{})));

G2Scopy
using G2SCopyOp = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>; // 128 uint1b_t per thread
using G2SCopyTraits = Copy_Traits<G2SCopyOp>;
using G2SCopyAtom = Copy_Atom<G2SCopyTraits, type>;

static constexpr int G2SCopy_thread_k =
    ROUND_UP(ThreadBlockShape::K / 128, 2); // 128 uint1b_t(int128_t) per thread

static constexpr int G2SCopy_thread_m = kThread / G2SCopy_thread_k;
static constexpr int G2SCopy_thread_n = kThread / G2SCopy_thread_k;

// CTA block G2S load copy [P * block_m, block_k] tile from [P * m, k]
using G2SCopyA = decltype(make_tiled_copy(
    G2SCopyAtom{},
    make_layout(make_shape(Int<G2SCopy_thread_m>{}, Int<G2SCopy_thread_k>{}),
                make_stride(Int<G2SCopy_thread_k>{}, _1{})),
    make_layout(make_shape(_1{}, _128{}))));
using G2SCopyB = decltype(make_tiled_copy(
    G2SCopyAtom{},
    make_layout(make_shape(Int<G2SCopy_thread_n>{}, Int<G2SCopy_thread_k>{}),
                make_stride(Int<G2SCopy_thread_k>{}, _1{})),
    make_layout(make_shape(_1{}, _128{}))));


用cp.async 定义tiled_copy, 所有的thread每次搬运完整的block_k 和余下的block_m，例如128 thread， block_k=256, tile copy 为threadlayout[64, 2] copy [64, 256] data。

对于G2Scopy 时需要保证边界处理的正确，否则会导致smem写越界和得到错误的gemm结果。如下图所示，tiled copy 需要strip 当前block 不需要的数据防止越界（黑色部分）， 并对block 不能填满的部分进行zero-filling 保证mma指令得到正常的累加结果（黄色部分）。

边界处理
template <class... CopyArgs, class PredTensor, class SrcEngine, class SrcLayout, class DstEngine,
          class DstLayout, class StripTuple, class ZfillTuple>
__device__ __forceinline__ static void
copy_strip_zfill(Copy_Atom<CopyArgs...> const &copy, PredTensor const &pred,
                 Tensor<SrcEngine, SrcLayout> const &src, Tensor<DstEngine, DstLayout> dst,
                 StripTuple const &strip_bound, ZfillTuple const &zfill_bound)
{
    static_assert(SrcLayout::rank == DstLayout::rank, "dst and src mismatch rank ");
    constexpr int Rank = SrcLayout::rank;
    // print_type(Rank);
    auto src_v = group_modes<1, Rank>(src); // [copy, copy_m * copy_n]
    auto dst_v = group_modes<1, Rank>(dst); // [copy, copy_m * copy_n]
    auto pred_v = group_modes<1, Rank>(pred); // [copy, copy_m * copy_n]
#pragma unroll
    for (int idx = 0; idx < size<1>(pred_v); idx++) {
        auto pred_coord = pred_v(_0{}, idx);
        // strip data OOB block tile
        if (elem_less(pred_coord, strip_bound)) {
            // fill zeros OOB global shape into block tile
            copy_if(
                copy,
                [&](auto... coords) { return elem_less(pred_v(_0{}, coords...), zfill_bound); },
                src_v(_, _), dst_v(_, _));
        }
    }
}


cute 提供了predicate tensor 机制来处理标界，用make_identity_tensor 接口创建对应tensor的predicate，其中的值为对应tensor 元素在矩阵内的坐标，如(0,128)。通过比较thread 被tiled copy分配到的pred和 tensor 全局的pred 的关系，来判断是否需要边界处理。如上代码通过strip_bound和zfill_bound来判断哪些数据需要剥离哪些数据需要填零。

tiledmma

不同于cutlass 中的mma层级(block tiling->warp tiling->mma tiling), cute 采用了tiled mma的封装用于构建多个warp组成的基本mma tiling， 通过多个tiled mma 组成完整的block tiling，所以开发者可以忽略warp-tiling 这一层级，由cute来计算所需的warp tiling。此外tiled mma可以调整warp中thread在m n维度计算的排布，更高效处理一些epilog 的场景，如下，引用自[QST] What is PermutationMNK in TiledMMA in CUTLASS 3.4 changes? · NVIDIA/cutlass · Discussion #1345。

TiledMMA tiled_mma = make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{},
                                        Layout<Shape<_1,_1,_1>>{},     // AtomLayout
                                        Tile<_8,_16,_8>{});            // Tiler
print_latex(tiled_mma);

TiledMMA tiled_mma = make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{},
                                        Layout<Shape<_1,_1,_1>>{},     // AtomLayout
                                        Tile<_8,                       // Permutation on M, equivalent to 8:1, identity
                                             Layout<Shape <_2,_4,_2>, 
                                                    Stride<_1,_4,_2>>, // Permutation on N, size 16
                                             _8>{});                   // Permutation on K, equivalent to 8:1, identity
print_latex(tiled_mma);

实验

实验环境 rtx4090（peak dram bw 1,008GBps），cuda_12.1.r12.1，cute版本 采用和bmma 最好性能的相同的tiling配置（适当调大stage 数目）。

1x4096x4096	w2a2	w3a3	w4a4	w5a5	w6a6	w7a7	w2a8
bmma version	10.02Tops
1257GBps	8.85Tops
1665GBps	7.01Tops
1758GBps	7.19Tops
2251GBps	6.48Tops
2437GBps	5.96Tops
2612GBps	10.02Tops
5019GBps
cute version	11.11Tops
1394GBps	9.04Tops
1701GBps	7.92Tops
1985GBps	6.81Tops
2133GBps	5.6Tops
2112GBps	4.22Tops
1848GBps	10.84Tops
5429GBps

由于有L2 cache，benchmark多次执行会放大dram带宽，所以实测带宽> dram spec带宽。后续需要ncu profiling分析w5a5/w6a6/w7a7 case下与bmma 版本的差异。

在rtx3090环境里做实验发现，3090上cp.async.zfill会出现随机的 bank conflict，但是4090上不会，不确定是不是3090硬件的问题。

小结

实际使用下来，Cute有更好的灵活性，可以更方便的拓展epilog和处理复杂的tensor layout，但是cute官方的示例还是比较少，文档的说明不太够(好在github 提问题官方回复的速度比较快)。cute 内部还是有些bug和支持不完善的地方，我在使用的过程中就发现了cp_async_zfill 的pred bug，导致cp async 总是zero filling，和uint1_t mma_trait 里AB Layout 定义错了导致结果mismtach以及cute支持的uint1_t mma 类型不完善，顺手给官方提了PR。
