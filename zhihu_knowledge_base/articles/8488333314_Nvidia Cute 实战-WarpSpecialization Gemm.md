# Nvidia Cute 实战-WarpSpecialization Gemm

**作者**: CalebDu

**原文链接**: https://zhuanlan.zhihu.com/p/8488333314

---

前言

Cute：

Nvidia Cutlass3.0 中推出了 新的编程库Cute，通过对layout, tensor, copy, mma 等在cuda编程中常见的对象和操作进行高级抽象，让开发者可以高效的处理复杂的坐标计算，以及CTA->warp->thread 不同执行层级的计算数据拆分。相比cutlass， cute可以更灵活的设计Epilog 以及设置一些在cutlass内难以处理的layout操作并忽略layout底层坐标计算的细节。

cutlass repo reed 大佬主页

cute基本的概念，有需要可以看cute 的官方文档及代码示例和reed 大佬的cute 解析文章，学习Cute前置知识。

WarpSpecialization Gemm

WarpSpecialization(WS)最早可以追溯到2011年发表的 CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization，在Fermi架构的GPU上首次提出。Nvidia GPU 以Warp(32 threads) 作为最小的执行单位，不同于传统的GPU编程模型中所有的warp同时负责G2S(Global memory to Shared memory) / S2R(Shared memory to Register)数据搬运和计算，WarpSpecialization 把CTA(Thread block) 中的warp分为Producer Warp(G2S) 和Consumer Warp(S2R/compute)。

CudaDMA paper 表示通过WS把不同warp拆分为异构的指令流，1. 可以提高不同内存层级的并行(MLP, Memory level Parallelism)[注：由于Ampere/Hopper架构开始引入了cp.async/tma 等异步内存拷贝指令,个人感觉WS在新架构上不一定能提高MLP ] 2. 避免了CTA层级的的粗粒度的同步操作，采用细粒度的barrier ptx 指令，减少了同步开销。3. WS分离了G2S/ compute 的复杂指令流，便于编译器优化两个独立的指令流， 提高ILP(Instruction Level Parallelism)。

WarpSpecialization

从Hopper 架构，Nvidia官方开始主推WS 的编程范式[感兴趣可以看一下官方的talk CUTLASS 2.x 与 3.x 的入门使用_哔哩哔哩_bilibili 大概从19分钟开始讲cutlass3.0 中的WS]。

Spec	Cuda Core
FP16	Tensor Core
FP16	DRAM BandWidth
A100 SXM	78TFLOPS	312TFLOPS	1555GBps
H100 SXM	133.8TFLOPS (1.7x)	989.4TFLOPS (3.17x)	3352GPps(2.1x)
H200 SXM	133.8TFLOPS (1.7x)	989.4TFLOPS(3.17x)	4800GBps(3.0x)

对于Gemm kernel， 可以分为prologue、mainloop、epilogue3个部分，Tensor Core主要发挥在mainloop的稳定流水中，epilogue 包含一些常见的Element-wise OP Fusion(Cuda Core) 和Store DRAM。随着新一代的GPU架构Tensor Core 的算力成倍提高，mainloop的耗时成倍下降，相比之下prologue和epilogue 的时间 相对固定，因此如何能隐藏prologue和epilogue 的时间对优化整体的kernel耗时显得更重要。如下图，在Hopper 架构上Nv 提出了 Persistent-Cooperative WS gemm有效的隐藏了prologue和epilogue 的时间。Persistent指 CTA不再是传统的Data Parallel 的方式只计算一个tile 就退出，CTAs占用全部的SM并每个CTA循环的计算多个tile直至计算完所有的tile，这样来隐藏prologue。Cooperative指一个CTA中存在2组Consumer Warp，当一个Consumer mainloop结束的时候进入epilogue，则切换到另一个Consumer继续计算mainloop，这通过两个Consumer 的mainloop pingpong隐藏掉epilogue。这样的新的编程范式可以保证GPU架构迭代的过程中有效的隐藏prologue和epilogue充分利用架构升级Scaling的Tensor Core算力 ，Tensor Core 的计算占到Gemm Kernel的耗时的主要部分。

WS_Persistent_PingPong

Hopper(sm90) 相比Ada(sm89)/Ampere(sm80) 引入了很多新的特性，由于本人手边只有Ada 和Ampere 的卡用，对Hopper架构的新特性不太熟悉，在实现之前需要通过官方的programming guide/ISA 梳理一下sm90的这些新特性对在sm80/sm89上实现Gemm-WS的影响(理解不一定正确，欢迎指正）。

TMA：Hopper架构中引入了TMA(Tensor Memory Accelerator)来提高GPU对多维Tensor的数据搬运效率，只需要warp中的一个thread 发起TMA指令，相比cp.async SIMIT指令可以大幅减少地址偏移计算的开销和提高数据搬运的效率。配和TMA，Hopper架构的上提供了新的mbarrier 异步通信指令。由于引入了TMA支持S2G 的async store，支持更高效的同步指令以及更大程度的计算/copy 的并行(如Cooperative-pingpong中mainloop与epilog的并行，由于cp.async 不能支持S2G的async cp，个人感觉sm8x上无法实现两组consumer的mainloop/epilog的并行）
https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
Programming Guide：Async Copy Ampere vs Hopper

2. WGMMA(Warp Group Matrix Multiply Accumalate) 是Hooper上提出的新的Tensor Core 计算指令，不同于之前架构的wmma/mma 单个warp调用一个sm 中单个Tensor Core进行同步的C=A*B+C 的计算，wgmma 指令允许四个连续的warp组成warp group 异步的调用一个sm中全部4个Tensor Core进行计算， 通过异步的wgmma计算指令更好的实现计算与数据搬运并行。

3. CTA reconfiguration: 由于WS不同的warp异构的负责不同的指令流，导致不同的warp的thread的所需使用的register数目是不同的，producer thread所需的register会远小于consumer thread。所以sm90 提供了setmaxnreg 指令手动分配和释放不需要的register，避免register的浪费。

实现

受限于sm8x架构的本文只实现了最基础的WS-Gemm。 代码发布在GitHub - CalebDu/Awesome-Cute

由于Cute 对于sm90之前的架构没有封装barrier ptx的api。实现中直接调用libcu++中封装barrier的api Synchronization Primitives/pipeline 来实现producer/consumer之间的同步。

  // initialize pipeline
  auto block = cooperative_groups::this_thread_block();
  const cuda::pipeline_role thread_role =
      block.thread_rank() < GemmTraits::kProducerThread
          ? cuda::pipeline_role::producer
          : cuda::pipeline_role::consumer;
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block,
                                         kStage>
      shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state, thread_role);


producer：调用pipeline.producer_acquire()/producer_commit() 来同步producer G2S搬运multistage数据。

   for (int k_main_loop_idx = 0, multi_stage_idx = 0;
         k_main_loop_idx < k_main_loop_cnt; k_main_loop_idx++) {

      for (; multi_stage_idx < k_main_loop_cnt &&
             multi_stage_idx < (k_main_loop_idx + kStage);
           multi_stage_idx++) {
        auto a_tile_bound =
            make_tuple(m_tile_bound, (g2s_g_read_cnt + 1) * kCTAK);
        auto b_tile_bound =
            make_tuple(n_tile_bound, (g2s_g_read_cnt + 1) * kCTAK);

        pipeline.producer_acquire();
        if constexpr (kBound_Check) {
          copy_strip_zfill(g2s_copy_a,
                           g2s_tAgA_copy_pred(_, _, _, g2s_g_read_cnt),
                           g2s_tAgA_copy(_, _, _, g2s_g_read_cnt),
                           g2s_tAsA_copy(_, _, _, g2s_s_write_cnt),
                           a_tile_bound, select<0, 2>(args.problem_shape));
          copy_strip_zfill(g2s_copy_b,
                           g2s_tBgB_copy_pred(_, _, _, g2s_g_read_cnt),
                           g2s_tBgB_copy(_, _, _, g2s_g_read_cnt),
                           g2s_tBsB_copy(_, _, _, g2s_s_write_cnt),
                           b_tile_bound, select<1, 2>(args.problem_shape));
        } else {
          copy(g2s_copy_a, g2s_tAgA_copy(_, _, _, g2s_g_read_cnt),
               g2s_tAsA_copy(_, _, _, g2s_s_write_cnt));
          copy(g2s_copy_b, g2s_tBgB_copy(_, _, _, g2s_g_read_cnt),
               g2s_tBsB_copy(_, _, _, g2s_s_write_cnt));
        }
        // cp_async_fence();
        pipeline.producer_commit();
        g2s_g_read_cnt++;
        g2s_s_write_cnt = (g2s_s_write_cnt + 1) % kStage;
        // if (thread0()) {
        //   print("produer %d\n", multi_stage_idx);
        // }
      }
    }


consumer：调用pipeline.consumer_wait()/consumer_release() 来同步consumer 等待 producer 搬运完数据和通知producer继续搬运数据。

    for (int k_main_loop_idx = 0; k_main_loop_idx < k_main_loop_cnt;
         k_main_loop_idx++) {
#pragma unroll
      for (int k_inner_loop_idx = 0; k_inner_loop_idx < k_inner_loop_cnt;
           k_inner_loop_idx++) {
        int next_k_inner_loop_idx = (k_inner_loop_idx + 1) % k_inner_loop_cnt;
        // wait next stage commit
        if (k_inner_loop_idx == k_inner_loop_cnt - 1 &&
            k_main_loop_idx < k_main_loop_cnt - 1) {
          pipeline.consumer_wait(); // wait producer
          s2r_s_read_cnt = next_s2r_s_read_cnt;
          // s2r_s_read_cnt = (s2r_s_read_cnt + 1) % kStage;
        }
        // s2r pipeline
        copy(s2r_copy_a,
             s2r_tAsA_copy(_, _, next_k_inner_loop_idx, s2r_s_read_cnt),
             s2r_tArA_copy(_, _, next_k_inner_loop_idx));
        copy(s2r_copy_b,
             s2r_tBsB_copy(_, _, next_k_inner_loop_idx, s2r_s_read_cnt),
             s2r_tBrB_copy(_, _, next_k_inner_loop_idx));
        if (k_inner_loop_idx == 0) {
          pipeline.consumer_release(); // trigger producer
          next_s2r_s_read_cnt = (s2r_s_read_cnt + 1) % kStage;
        }
        // gemm
        gemm(mma, tArA(_, _, k_inner_loop_idx), tBrB(_, _, k_inner_loop_idx),
             tCrC);
      }
    }

实验

实验环境 rtx4090（spec dram bw 1,008GBps，tensor core fp16 330tflops），cuda_12.1.r12.1。

	cublas	cute: gemm_multistage_128*256*32_stage3	cute: gemm_ws_producer32_128*256*32_stage3	cute: gemm_ws_producer64_128*256*32_stage3	cute: gemm_ws_producer128_128*256*32_stage3
mnk(2048, 2048,2048)	218tflops/0.078ms	275tflops/0.062ms	235tflops/0.072ms	246tflops/0.069ms	252tflops/0.068ms
mnk(4096,4096,4096)	260tflops/0.525ms	289tflops/0.475ms	247tflops/0.556ms	259tflops/0.528ms	267tflops/0.514ms
mnk(8192,8192,8192)	252tflops/4.353ms	268tflops/4.101ms	211tflops/5.206ms	247tflops/4.436ms	255tflops/4.304ms

ws相比于multistage反而有性能性能下降。看ncu profile报告，ws的版本比 multistage 的版本sm吞吐和带宽都低不少，整体的指令数目增加了很多，有一些疑惑。
