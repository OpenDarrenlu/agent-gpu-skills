# Nvidia Cute 实战-WarpSpecialization Gemm for Hopper

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/1905383022901059783

---

这篇文章拖延了好久，拖到Cute python DSL release了
前言

Cute：

Nvidia Cutlass3.0 中推出了 新的编程库Cute，通过对layout, tensor, copy, mma 等在cuda编程中常见的对象和操作进行高级抽象，让开发者可以高效的处理复杂的坐标计算，以及CTA->warp->thread 不同执行层级的计算数据拆分。相比cutlass， cute可以更灵活的设计Epilog 以及设置一些在cutlass内难以处理的layout操作并忽略layout底层坐标计算的细节。

cutlass repo reed 大佬主页

cute基本的概念，有需要可以看cute 的官方文档及代码示例和reed 大佬的cute 解析文章，学习Cute前置知识。

WarpSpecialization Gemm for Hopper：

CalebDu：Nvidia Cute 实战-WarpSpecialization Gemm

在之前的文章中，我们尝试在Ada 架构的RTX 4090 实现了Naive WarpSpecialization Gemm，在Hopper之前的架构，WarpSpecialization 的编程范式由于以下的原因无法获得理想的性能：

缺少setmaxnreg ptx指令来为异构的warp group管理register，导致了producer warp分配了大量的mma register影响了sm 的occupancy。
缺少TMA(Tensor Memory Accelerator),produder 仍然通过SIMT的方式进行数据搬运，需要提高并行度来隐藏latency，要求计算每个thread对应的offset 来通过cp.async拷贝对应的数据，TMA减少了坐标计算的cuda core需求和所需寄存器。
在Hopper之前的架构barrier 通过自旋锁实现，Hopper架构提供了硬件加速的高效的mbarrier。

在本文中，基于cultass/cute 封装的低级api尝试结合Hopper架构的新特性和常见的优化手段：setmaxnreg，mbarrier, TMA_multicast, WGMMA，CTA Cluster，CTA swizzle，Persistent Scheduler等等，来实现WarpSpecialization/WarpSpecialization Cooperative/WarpSpecialization PingPong Gemm。（本文中不会使用cutlass pipeline 接口，直接通过barrier来实现produer和consumer 之间 或 pingpong warp group 之间的同步）

实现

本文参考cutlass和deepgemm 的实现，代码和ncu report发布在GitHub - CalebDu/Awesome-Cute

Kernel Tag

本文的实现采用WASP/WASP_COOP/WASP_PIPO 枚举来代表3种WarpSpeicalization Gemm实现

enum class KernelTag {
  WASP = 0,      // warp specialization
  WASP_COOP = 1, // warp specialization cooperative
  WASP_PIPO = 2, // warp specialization pingpong
};


CTA Cluster

从Hopper(SM90)开始，Nvidia引入了CTA cluster，允许最大16个CTA调度到同一个GPC(graphic processor cluster)内的SM组成Cluster，同时支持了Cluster内CTA 的SMEM 组成distributed SMEM，可以CTA间互相访问或TMA 在cluster之间进行multicast 节约访问global memory的带宽。

由于Hopper 的GPC 最大拥有18个SM，所以如果cluster size>2 无法launch 全部的SM，在本文的实现中限制cluster size<=2。

h100 white paper

Persistent Scheduler and CTA Swizzle

Persistent Scheduler 不同于传统的data parallel, grid 固定launch CTA 数目=SM数目（cluster size=2条件下最优的配置），保证每个CTA 运行多个Gemm Tile 从而可以从第二个Tile开始隐藏prologue的开销。

// gemm/warp_speicalization_gemm/gemm_ws_scheduler_sm90a.h  
template <typename Cluster_shape>
  static dim3 get_grid_dim(Cluster_shape cluster_shape) {
    cudaDeviceProp prop;
    int device_id;
    cudaGetDevice(&device_id);
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    auto cluster_size = size<0>(cluster_shape) * size<1>(cluster_shape);
    auto sm_count = prop.multiProcessorCount;
    auto grid_size = sm_count / cluster_size * cluster_size;
    return dim3(grid_size, 1, 1);
  }


由于Nvidia GPU存在L2 Cache 可以缓存SM之间访问过的局部数据，在H100 上L2 cache 可以提供10TB/s 的带宽，所以如何高效的利用L2 Cache提高数据复用率是Gemm kernel 优化的一个重点。

Wave表示GPU上可以同时并行的CTA的数目，我们可以把wave并行的分配到C矩阵的M和N维度上，得到 Wave_M、Wave_N ，对于Persistent Scheduler Wave_M\times Wave_N \approx N\_SM(132SM\ in \ H100) 。对于一次Wave 的global memory 的访问量为 (tile_n+tile_m)\times k\times wave ，L2 cache 的访问量为 (wave_m\times tile_m+wave_n\times tile_n)\times k ， 为了L2 cache 最大化命中率，我们需要最小化L2 cache 的访问量，所以可以转化为 在 Wave_M\times Wave_N =N\_SM 的约束下，求 (wave_m\times tile_m+wave_n\times tile_n) 最小。 我们可以得到当 wave_m\times tile_m 与 wave_n\times tile_n 之间的差值最小时，L2 cache的命中率最大。当CTA tile=128x128x64时，wave 推荐为16x8或8x16的CTA 排布。本文实现中默认采用如下图的N-dim Along 的CTA swizzle 策略。同时实现中没有采用DeepGemm 中Scheduler 对Cluster size 不对齐情况的判断，默认计算对齐到Cluster size 情况， 例如 tile的数目为<3, 2> , swizzle=2，cluster size 为<2, 1, 1>, 默认把tile 对齐 <2x2,2>, 会产生冗余tile计算(TMA 会判断out of bound 的数据搬运）。

example: C tensor with cluster(2, 1, 1) swizzle=4 N-dim Along
// block_idx to tile coordinate
// gemm/warp_speicalization_gemm/gemm_ws_scheduler_sm90a.h 
DEVICE
  TileInfo get_tile_id() {
    if (linear_block_idx >= param_.problem_tiles) {
      return {-1u, -1u, false};
    }
    auto linear_cluster_idx = linear_block_idx / param_.cluster_size;
    auto block_offset = linear_block_idx % param_.cluster_size;

    auto block_m_offset = block_offset % param_.cluster_m_shape;
    auto block_n_offset = block_offset / param_.cluster_m_shape;

    auto cluster_swizzle_offset = linear_cluster_idx % param_.swizzle_;
    auto cluster_swizzle_extra = linear_cluster_idx / param_.swizzle_;

    auto cluster_m_swizzle_offset =
        cluster_swizzle_extra / param_.cluster_n_blocks;
    auto cluster_m_idx =
        cluster_m_swizzle_offset * param_.swizzle_ + cluster_swizzle_offset;
    auto cluster_n_idx = cluster_swizzle_extra % param_.cluster_n_blocks;

    auto tile_m = cluster_m_idx * param_.cluster_m_shape + block_m_offset;
    auto tile_n = cluster_n_idx * param_.cluster_n_shape + block_n_offset;

    return {tile_m, tile_n, true};
  }

ncu L2cache profile

Mainloop Barrier Pipeline

producer thread分配40 个register， consumer thread 分配232 个register，40x128+232x256 = 64512 register，尽可能用满65536个register。本文的实现WASP/WASP_COOP共用一个consumer 函数，WASP_PIPO 调用ws_pipo_consumer实现两个2wg mainloop与epilog的并行。

    // WASP: consumer wg0, produer wg1
    // WASP_COOP: consumer wg0 wg1, producer wg2
    // WASP_PIPO: consumer wg0 wg1, producer wg2
    if (warp_group_idx == WarpGroupCnt - 1) {
      // producer
      // alloc 40 register for tma load
      cutlass::arch::warpgroup_reg_dealloc<40>();
      // elect 1 thread issue tma load
      if (warp_idx_in_group == 0 && elect_one_sync()) {
        producer(param, shared_storage, block_rank_in_cluster);
      }
    } else {
      // consumer
      // alloc 232 register for mma compute
      cutlass::arch::warpgroup_reg_alloc<232>();

      if constexpr (kernel_tag == KernelTag::WASP ||
                    kernel_tag == KernelTag::WASP_COOP) {
        ws_consumer(param, shared_storage);
      } else if constexpr (kernel_tag == KernelTag::WASP_PIPO) {
        ws_pipo_consumer(param, shared_storage);
      }
    }


采用多级的mbarrier来实现multistage producer 和consumer 之间的同步。1个thread 初始化mbarrier，需要提供arrival count，来设置当前mbarrier调用wait(phase)时需要等待多少个arrive 到达才完成当前phase。对于TMA 的数据传输，通过expect_tx 来设置mbarrier 需要等待tma搬运多少bytes 的数据完成。

ptx isa

当mbarrier.wait(phase) arrival count 和tx count都到达0时，当前phase完成，当采用parity mode 时phase将更新为phase=phase^1。

ptx isa

对于Stage的pipeline，需要full[Stage] 来表示producer完成数据搬运和empty[Stage] 表示consumer 完成当前iteration 的smem数据mma计算。producer 的arrival count=1表示1个thread 发起tma 指令，consumer arrival count=size(TiledMma{})*size(ClusterShape{})/ WarpSize 表示cluster 内的tiled mma中的thread以warp 为单位发起empty.arrive。

需要注意：Producer 的PipelineState 把phase初始化为1，第一次 phase != 0, empty.wait(phase) 不触发等待直接开始搬运tma。

//producer 
    PipelineState<Stage> pipeline_states{1, 0};
    while (tile_info.is_valid) {
#pragma unroll 1
      for (int k_idx = 0; k_idx < k_loop_cnt; k_idx++) {
        uint64_t *full_barrier_ptr = reinterpret_cast<uint64_t *>(
            &shared_storage.pipelines
                 .mainloop_full_bar[pipeline_states.stage_idx]);
        // wait consumer
        shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx]
            .wait(pipeline_states.phase);
        // notify consumer
        shared_storage.pipelines.mainloop_full_bar[pipeline_states.stage_idx]
            .arrive_and_expect_tx(TmaLoadTotalBytes);
        // tma load
        copy(param.tma_a.with(*full_barrier_ptr, mcast_mask_a),
             tAgA(_, _, _, k_idx), tAsA(_, _, _, pipeline_states.stage_idx));
        copy(param.tma_b.with(*full_barrier_ptr, mcast_mask_b),
             tBgB(_, _, _, k_idx), tBsB(_, _, _, pipeline_states.stage_idx));
        // update pipeline states
        pipeline_states++;
      }
      // next tile
      scheduler.advance_next_tile();
      tile_info = scheduler.get_tile_id();
    }
    // load tail: make sure all consumers have use data
#pragma unroll
    for (int i = 0; i < Stage; i++) {
      shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx]
          .wait(pipeline_states.phase);
      pipeline_states++;
    }


Consumer 计算mma时需要保持一个wgmma commit in-flight，在mma 结束调用mma tail，等待一轮iteration 的全部wgmma commit完成后最后一个empty.arrive.

//consumer mma
  // fisrt mma with no accumulation to avoid init zeros
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  {
      // prologue mma
      // wait producer
      bool wait_complete =
          shared_storage.pipelines
              .mainloop_full_bar[pipeline_states_read.stage_idx]
              .try_wait(pipeline_states_read.phase);
      if (!wait_complete) {
        shared_storage.pipelines
            .mainloop_full_bar[pipeline_states_read.stage_idx]
            .wait(pipeline_states_read.phase);
      }
#pragma unroll
      for (int k_inner = 0; k_inner < size<2>(tCrA); k_inner++) {
        gemm(tiled_mma, tCrA(_, _, k_inner, pipeline_states_read.stage_idx),
             tCrB(_, _, k_inner, pipeline_states_read.stage_idx), acc);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      pipeline_states_read++;
    }
    warpgroup_fence_operand(acc);

#pragma unroll 1
    for (int k_idx = 1; k_idx < param.scheduler.k_loop_cnt; k_idx++) {
      // wait producer
      shared_storage.pipelines.mainloop_full_bar[pipeline_states_read.stage_idx]
          .wait(pipeline_states_read.phase);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      // wgmma
      warpgroup_commit_batch();
      // keep 1 wgmma commit in-flight
      warpgroup_wait<1>();
      warpgroup_fence_operand(acc);
      // notify producer
      shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx]
          .arrive(target_cta, pred_arrive);
      // update pipeline states
      pipeline_states++;
      pipeline_states_read++;
    }
    warpgroup_fence_operand(acc);

// consumer mma tail
    warpgroup_wait<0>();
    shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx]
        .arrive(target_cta, pred_arrive);
    pipeline_states++;

example:stage3 mainloop pipeline（省略mma tail）

TMA MultiCast

TMA 提供了MultiCast功能允许一次tma读取的数据 多播到 cluster 内的多个CTA，通过16bit(一个cluster最大16个CTA) 的CTAMask表示需要多播的CTA id.

ptx isa:cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast
    // init tma multicast mask
    uint16_t mcast_mask_a = 0, mcast_mask_b = 0;
    if constexpr (is_same_v<TmaG2STiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
#pragma unroll
      for (int n = 0; n < size<1>(cluster_layout); n++) {
        mcast_mask_a |= (static_cast<uint16_t>(1)
                         << cluster_layout(cluster_idx.x, n, _0{}));
      }
    }
    if constexpr (is_same_v<TmaG2STiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
#pragma unroll
      for (int m = 0; m < size<0>(cluster_layout); m++) {
        mcast_mask_b |= (static_cast<uint16_t>(1)
                         << cluster_layout(m, cluster_idx.y, _0{}));
      }
    }
...
   copy(param.tma_a.with(*full_barrier_ptr, mcast_mask_a),
             tAgA(_, _, _, k_idx), tAsA(_, _, _, pipeline_states.stage_idx));
   copy(param.tma_b.with(*full_barrier_ptr, mcast_mask_b),
             tBgB(_, _, _, k_idx), tBsB(_, _, _, pipeline_states.stage_idx));

example:cluster(2,1,1) /cluster(1,2,1) multicast

Tiled MMA

Hopper架构引入了新的tensor core 计算指令wgmma，支持4个warp组成warp group 异步的执行tensor core计算，wgmma 支持m64nXk16 fp16 shape（X为8至256，步长为8），可以看到wgmma 实际上是4个warp 的m16n8k16在m维度上拼成m16，再在n维度拓展。

wgmma m64nXk16

WASP_COOP采用2个warp group 来组成tiled mma，WASP和WASP_PIPO 的tiled mma仅使用一个warp group

  // ws_cooperative use 2 warp group
  using AtomLayoutMNK =
      std::conditional_t<kernel_tag == KernelTag::WASP_COOP,
                         Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<ABtype, ABtype, Acctype, CtaTile, GmmaMajorA,
                                 GmmaMajorB>(),
      AtomLayoutMNK{}));

128*128 Tiled MMA

Epilog Pipeline

Epilog 的实现采用Stage=4的pipeline来实现r2s 和tma s2g 的并行。通过cutlass::arch::NamedBarrier(MmaThreads).sync(); 实现tiled mma内的thread 同步。

#pragma unroll
    for (int epilog_m = 0; epilog_m < epilog_m_loop; epilog_m++) {
#pragma unroll
      for (int epilog_n = 0; epilog_n < epilog_n_loop; epilog_n++) {
        int mma_m = epilog_m;
        int mma_n = epilog_n * epilog_tile_n / mma_tile_m;
        int reg_offset =
            (epilog_n % (mma_tile_n / epilog_tile_n)) * size(r2s_rC_frag);

        Tensor cur_r2s_rC = r2s_rC(_, mma_m, mma_n);
        // convert acc fp32 to fp16 r2r copy
#pragma unroll
        for (int i = 0; i < size(r2s_rC_frag); i++) {
          r2s_rC_frag(i) = __float2half(cur_r2s_rC(reg_offset + i));
        }
        if (is_issue_tma) {
          // wait smem available
          if (pipeline_states.count > EpilogStage - 1) {
            // keep EpilogStage-1 tma in-flight
            tma_store_wait<EpilogStage - 1>();
          }
        }
        cutlass::arch::NamedBarrier(MmaThreads).sync();
        // r2s copy
        copy(r2s_copy_c, r2s_rC_frag,
             r2s_sC(_, _, _, pipeline_states.stage_idx));
        // fence for visiblity
        cutlass::arch::fence_view_async_shared();
        // sync current consumer threads
        cutlass::arch::NamedBarrier(MmaThreads).sync();
        if (is_issue_tma) {
          // issue tma s2g store
          copy(param.tma_c, s2g_sC(_, _, _, pipeline_states.stage_idx),
               s2g_gC(_, _, _, epilog_m, epilog_n));
          // commit s2g tma
          tma_store_arrive();
        }
        pipeline_states++;
      }
    }
    if (is_issue_tma) {
      // store tail
      tma_store_wait<0>();
    }

epilog pipeline


PingPong Barrier Pipeline

WASP_PIPO 通过pipo_mainloop_bar[2] pipo_epilog_bar[2] 4个barrier控制一个CTA之间的两个wg 有序的计算2个Tile。在一个wg 发起mma tail之前通过mainloop barrier 通知另一个wg 发起mma计算，同理epilog 也是同样的方法，从而实现在用一个wg 的mainloop来与另一个wg的epilog 并行，用mainloop隐藏掉epilog的操作。Pipo_states wg0初始化phase=1 wg1 初始化phase=0，保证wg0 直接发起mainloop和epilog，需要注意的Pipo_states 和scheduler每次都需要额外多advance 一次iteration 来跳过另一个wg 负责的状态，否则barrier 可能会触发hang。

  DEVICE void ws_pipo_consumer(Param const &param,
                               SharedStorage &shared_storage) {
    Scheduler scheduler(param.scheduler);
    auto warp_group_idx =
        __shfl_sync(0xffffffff, threadIdx.x / WarpGroupSize, 0);
    auto other_warp_group_idx =
        __shfl_sync(0xffffffff, ((warp_group_idx + 1) & 1), 0);
    // wg0 phase = 1, wg1 phase = 0 (wait wg0)
    PipelineState<PipoOrderedStage> pipo_states{warp_group_idx == 0, 0};
    PipelineState<Stage> pipeline_states{0, 0};
    // wg1 advance next tile
    if (warp_group_idx != 0) {
      scheduler.advance_next_tile();
      pipeline_states.advance(param.scheduler.k_loop_cnt);
    }
    while (tile_info.is_valid) {
      // 2 wg orderedly execute mainloop
      shared_storage.pipelines.pipo_mainloop_bar[warp_group_idx].wait(
          pipo_states.phase);

      issue_mma(param, shared_storage, acc, pipeline_states, tiled_mma);
      // notify other wg execute mainloop
      shared_storage.pipelines.pipo_mainloop_bar[other_warp_group_idx].arrive();
      pipo_states++; // stage_idx ++
      mma_tail(shared_storage, pipeline_states);
      // pipeline state extra advance k_loop_cnt
      pipeline_states.advance(param.scheduler.k_loop_cnt);
      // 2 wg orderedly execute epilog
      shared_storage.pipelines.pipo_epilog_bar[warp_group_idx].wait(
          pipo_states.phase);
      issue_epilog(param, shared_storage, tile_info, acc, thread_idx_in_mma);
      // notify other wg execute epilog
      shared_storage.pipelines.pipo_epilog_bar[other_warp_group_idx].arrive();
      pipo_states++; // stage_idx ++, phase reverse
      // advance 2 tiles
      scheduler.advance_next_tile();
      scheduler.advance_next_tile();
      tile_info = scheduler.get_tile_id();
    }
  }





consumer pingpong pipeline
实验

测试环境 H100 PCIE 80GB （peak 800tflops fp16 tensor core）， cuda_12.8.r12.8

	2048x2048x2048	4096x4095x4096	8192x8192x8192	备注
cublas gemm	ncu: 49788cycle
stream: 626tflops 0.027419 ms	ncu: 282381cycle
stream:775tflops 0.177158 ms	ncu: 2270766cycle
stream:716tflops 1.533956 ms	stream测出cublas的latency更少算力更高，但是ncu 显示的cublas 的cycle数并没有明显的减少
cutlass_ws gemm	ncu: 51681cycle
stream: 521tflops 0.032957 ms	ncu: 325350cycle
stream: 596tflops 0.230587 ms	ncu: 2825726cycle
stream:516tflops 1.815191 ms	cutlass ws 还是采用data parallel的策略而非persistent
cutlass_ws_coop gemm	ncu: 47620cycle
stream:566tflops 0.030308 ms	ncu: 293475cycle
stream:679tflops 0.202164 ms	ncu: 2291953cycle
stream:598tflops 1.837878 ms	
cutlass_ws_pipo gemm	ncu: 46299cycle
stream:564tflops 0.030417 ms	ncu: 279205cycle
stream:695tflops 0.197744 ms	ncu: 2237864cycle
stream:608tflops 1.805988 ms	
my_ws gemm	ncu: 44868cycle
stream: 628tflops 0.027348 ms	ncu: 291775cycle
stream: 694tflops 0.197875 ms	ncu: 2195027cycle
stream:605tflops 1.815191 ms	
my_ws_coop gemm	ncu: 43795cycle
stream: 629tflops 0.027309 ms	ncu: 287676cycle
stream:683tflops 0.201130 ms	ncu: 2173536cycle
stream: 626tflops 1.753922 ms	
my_ws_pipo gemm	ncu: 42447cycle
stream: 629tflops 0.026972 ms	ncu: 275501cycle
stream: 710tflops 0.193317 ms	ncu: 2129913cycle
stream:638tflops 1.722661 ms	

cublas 的实现 ncu 抓的cycle数没有明显更少，但是用stream实测的性能会更好，需要后续再研究一下。

需要注意的cultass编译的时候需要加上-DNDEBUG 的flag，否则会插入debug 的一些信息导致kernel性能大幅下降（坑）。

后续更新的话应该就考虑用Cute python DSL了，模版还是太折磨人了
