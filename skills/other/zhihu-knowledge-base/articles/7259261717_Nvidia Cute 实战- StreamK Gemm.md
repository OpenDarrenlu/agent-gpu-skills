# Nvidia Cute 实战- StreamK Gemm

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/7259261717

---

前言

Cute：

Nvidia Cutlass3.0 中推出了 新的编程库Cute，通过对layout, tensor, copy, mma 等在cuda编程中常见的对象和操作进行高级抽象，让开发者可以高效的处理复杂的坐标计算，以及CTA->warp->thread 不同执行层级的计算数据拆分。相比cutlass， cute可以更灵活的设计Epilog 以及设置一些在cutlass内难以处理的layout操作并忽略layout底层坐标计算的细节。

cutlass repo reed 大佬主页

cute基本的概念，有需要可以看cute 的官方文档及代码示例和reed 大佬的cute 解析文章，学习Cute前置知识。

StreamK-Gemm

paper : Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

StreamK-Gemm是cutlass 官方提供的一种Gemm(General Matrix Multiply)的一种高效实现，针对在GPU上传统DP(Data Parallel)-Gemm面临的wave quantization 导致的GPU SM计算资源利用率低的问题。

wave quantization：Cuda编程中每个CTA(thread block)会被分配到特定的SM硬件单元中，所有的SM可以并行执行的CTA 称为一个wave。对应不同problem shape的Gemm，传统的DP-Gemm会把problem shape的Gemm问题拆分到若干个固定tile 的CTA并行的来计算，但是通常DP的CTA数和SM的数据无法做到完美匹配，这会导致tailing wave 无法用满全部的SM 单元，导致算力的浪费，如下图。

wave quantization
DP-Gemm
StreamK-Gemm

DP-Gemm： 对于(m,n,k)的gemm， 每个CTA 负责计算（CTA_m,CTA_n,CTA_k）的tile， 共需要ceil_div(m, CTA_M)*ceil_div(n, CTA_n) 个CTA，每个CTA 的mainloop 需要ceil_div(k, CTA_k) 次迭代。

Basic SK-Gemm：已知(m,n,k)的gemm， 共需要ceil_div(m, CTA_M)*ceil_div(n, CTA_n) *ceil_div(k, CTA_k) =A次迭代，为了用满所有的SM， 把所有的A次mainloop 迭代，均分到所有的SM上，每个CTA 负责ceil_div(A, nSM) 次迭代。但是会引入一个问题，每个CTA负责计算的部分可能会是不同tile的 一部分mainloop，CTA只计算了当前tile的partial sum，所以需要跨CTA之间的通信和reduce来获得完整的tile结果，对应上图的FixUp。因为每个CTA负责计算部分存在Skew，可以避免CTA之间的等待，例如上图中CTA1 会先计算2 tile的partial sum存入 dram的 workspace，CTA0 会先计算0 1 tile最后计算2 tile 并与workspace的partial sum做reduce。

DP-SK-Gemm：论文中提到Basic SK-Gemm 中CTA之间的Skew对于GPU cache有负面的影响，所以为了最小化这个负面影响，对于可以填满full wave的tile 还是采用DP来计算，对于不能填满wave的remaining tile 采用SK的策略。

DP-SK-Gemm
SK-Gemm 伪代码
实现

Cutlass 内开源了StreamK+DP+SplitK Gemm 的实现，核心实现在以下路径cutlass/include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h、cutlass/include/cutlass/gemm/kernel/gemm_universal_streamk.h

本文Cute的实现参考官方实现，并进行了适当的简化，仅实现了DP+SK，去除了SplitK。对one tile SK+DP/ two ti le SK+DP 的策略进行分离，方便对比。发布在CalebDu/Awesome-Cute

gemm/streamk_gemm/gemm_streamk.h：定义Cute Gemm Trait 及mainloop/epilog/sk tile reduce核心逻辑

gemm/streamk_gemm/dp_sk_block.h： 定义block->tile 之间的映射，如何设置dp/sk block，heuristic search sk block number。

SK_DP_Block_Wrapper

SK_DP_Block_Wrapper 的作用是根据problem size 和 CTA tile 的shape 和1sk tile/ 2sk tile strategy计算出当前最优的sk/dp block。avail_sm 为当前GPU 拥有的SM数， sm_occupancy为 每个SM 可以驻留的CTA 数目（与SM的smem 大小和CTA使用的smem大小有关）。dp_tiles 代表被avail_sms 整除的tile 数目， 剩下的余数tile(sk_tile)采用streamk gemm来计算，如果是2tile sk 模式，从dp_tiles中拿出一个full wave 给sk tile,避免超过2个sk block处理一个tile的情况，造成较大的block之间的通信开销。get_sk_blocks 函数采用cutlass 官方的实现，用于在 给定sk_tiles 的数目下，heuristic search 最优的sk block 数目。得到launch kernel 的grid 即为 (sk_blocks+dp_blocks)。

// SK_DP_Block_Wrapper
static void get_blocks(int &dp_tiles, int &sk_blocks, int output_tiles,
                         int iter_per_tile, int avail_sms, int sm_occupancy,
                         SK_DP_Block_Strategy strategy) {
    dp_tiles = output_tiles;
    int full_waves = output_tiles / avail_sms;
    int full_wave_tiles = full_waves * avail_sms;
    int partial_wave_tiles = output_tiles - full_wave_tiles;
    int score = -1;
    if (partial_wave_tiles == 0) {
      // Perfect quantization
      return;
    }
    if (strategy == SK_DP_Block_Strategy::sk1tile_dp) {
      int max_sk_occupancy = sm_occupancy - ((full_waves) % sm_occupancy);
      dp_tiles = full_wave_tiles;

      get_sk_blocks(sk_blocks, score, partial_wave_tiles, iter_per_tile,
                    avail_sms, max_sk_occupancy, true);
      if (score < 0) {
        printf("disable streamk\n");
        sk_blocks = 0;
        dp_tiles = output_tiles;
      }
    } else if (strategy == SK_DP_Block_Strategy::sk2tile_dp) {
      int max_sk_occupancy = sm_occupancy - ((full_waves - 1) % sm_occupancy);
      dp_tiles = full_wave_tiles - avail_sms;
      get_sk_blocks(sk_blocks, score, partial_wave_tiles + avail_sms,
                    iter_per_tile, avail_sms, max_sk_occupancy, true);
      if (score < 0) {
        printf("disable streamk\n");
        sk_blocks = 0;
        dp_tiles = output_tiles;
      }
    }
  }


SK_DP_Block_Wrapper 中的big_sk_block、normal_sk_block代表sk iter 不能被sk block 整除的情况，big_sk_block 需要比normal_sk_block多处理一次迭代。由于cute实现的简化的streamk 去除了splitK，所以不需要官方实现中的 sk_region 的定义。

GemmTraits

GemmTraits 中定义了gemm 所需的 copy mma layout 等类型， 和用于保存参数和管理workspace的Argument 类型。

GemmTraits::mainloop,GemmTraits::epilog 的核心逻辑参考 reed：cute 之 高效GEMM实现

Argument 负责管理StreamK 所需的workspace 空间，本文的实现和Cutlass官方实践有一些不同，做了一些简化，已知sk tile的数目，所以workspace 只需要对每个sk tile 分配[CTA_m, CTA_n] 的tensor和一个barrier（用于当前sk tile所需的所有sk block之间的通信）。官方实现好像是对每个sk block 都分配，感觉是多余的。

对于dp/sk block的处理，参考官方的实现，用TileWorkDesc 来记录每个block 负责的当前tile 中mainloop k iter 开始与结束的范围，GemmTraits::mainloop 根据这个范围来计算CTA中每个thread 负责的 c_frag。

sk_tile_reduce 函数用于sk tile 负责所有sk block 的reduce 各自的partial sum得到完整的sk tile的结果。具体的实现和paper 描述的有一些差异。paper中伪代码sk block是从 first tile to last tile的迭代遍历顺序，对应下图红色箭头，官方的代码实现sk block是从last tile to first tile的迭代遍历顺序，对应下图的绿色箭头。为了保证cta之间的skew，sk block之间通信逻辑和伪代码是相反的。

  template <typename AccEngine, typename AccLayout, typename PartialEngine,
            typename PartialLayout>
  DEVICE static void
  sk_tile_reduce(Arguments const &args, TileWorkDesc &tile_work, int block_idx,
                 Tensor<AccEngine, AccLayout> &acc,
                 Tensor<PartialEngine, PartialLayout> &partial_sum) {
    int first_iter = tile_work.tile_idx * args.block_wrapper.iter_per_tile;
    int first_block = args.block_wrapper.get_sk_block_idx(first_iter);
    if (!tile_work.tile_finished(args)) {
      share_accumulators(args, tile_work, block_idx, first_block, acc,
                         partial_sum);
    } else if (!tile_work.tile_started()) {
      acquire_accumulators(args, tile_work, block_idx, first_block, acc,
                           partial_sum);
    }
  }


非tile 最后一个iter 的sk block负责调用share_accumulators，把partial sum 存储/atomic add到workspace，负责最后一个iter的sk block 负责把workspace中存储的其余sk block的partial sum做reduce 得到完整的sk tile 结果。

  template <typename AccEngine, typename AccLayout, typename PartialEngine,
            typename PartialLayout>
  DEVICE static void
  share_accumulators(Arguments const &args, TileWorkDesc &tile_work,
                     int block_idx, int first_block,
                     Tensor<AccEngine, AccLayout> const &acc,
                     Tensor<PartialEngine, PartialLayout> &partial_sum) {
    int tidx = threadIdx.x;
    auto copy_acc = make_tiled_copy_C(R2SCopyAtomC{}, MMA{});
    auto thr_copy_acc = copy_acc.get_slice(tidx);
    auto copy_acc_s = thr_copy_acc.retile_S(acc);
    auto copy_acc_d = thr_copy_acc.partition_D(partial_sum);

    if (block_idx == first_block) {
      // store acc to dram partial_sum
      copy(copy_acc, copy_acc_s, copy_acc_d);
    } else {
      int wait_block_count = block_idx - first_block;
      auto pack_copy_acc_s = recast<Ctype_pack>(copy_acc_s);
      auto pack_copy_acc_d = recast<Ctype_pack>(copy_acc_d);
      // Turnstile reduction order deterministicly: wait all previous block
      // complete
      Barrier::wait_eq(args.barrier_workspace, tidx, tile_work.tile_idx,
                       wait_block_count);
      // atomic reduce
#pragma unroll
      for (int idx = 0; idx < size(pack_copy_acc_s); idx++) {
        atomicAdd(&pack_copy_acc_d(idx), pack_copy_acc_s(idx));
      }
    }
    // arrive counter ++
    Barrier::arrive_inc(args.barrier_workspace, tidx, tile_work.tile_idx);
  }


share_accumulators 采用了官方实现的Turnstile 策略，通过barrier spin clock保证了sk block之间安装block id的顺序进行atomic add，保证结果deterministic。

acquire_accumulators 需要保证当前sk tile 其余的sk block 都把partial sum 存储到workspace 后再进行reduce，同样用barrier来保证上述的这个过程。

实验

实验环境 rtx4090（spec dram bw 1,008GBps，tensor core fp16 330tflops），cuda_12.1.r12.1。

A100 tensor core吞吐

flops=2\times freq\times n\_sm\times fma/sm/cycle

根据tensor core fp16 330tflops, boost freq 2520mhz,128 sm 可反推得4090 tensor core fp16 fma 吞吐为512 fma/cycle/sm=128fma/cycle/tensor_core。

测试case：tiles 数目不整除sm数目	cublas	cute: gemm_multistage	cute: gemm_streamk_1sk_dp	cute: gemm_streamk_2sk_dp_128*256*32_stage3	cutlass官方实现:example/47_ampere_gemm_universal_streamk
mnk
(4096,4352,4096)	235tflops/0.619 ms	249tflops/0.585ms(128*128*32_stage3）	257tflops/0.566ms(_128*256*32_stage3)	271tflops/0.538	270tflops/0.553ms(default load-balancing)
mnk
(4096,4352,10240)	235tflops/1.545ms	239tflops/1.521ms(128*256*32_stage3）	258tflops/1.414ms(_128*256*32_stage3)	265tflops/1.373ms	263tflops/1.384ms(default load-balancing)
mnk
(1152,4352,4096)	219tflops/0.186ms	218tflops/0.187ms(128*128*32_stage3)	255tflops/0.160ms(gemm_streamk_1sk_dp_128*128*32_stage3)	268tflops/0.153ms	272tflops/1.504ms(default load-balancing)
